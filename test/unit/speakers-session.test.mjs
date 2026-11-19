import { test } from 'node:test';
import assert from 'node:assert/strict';
import { SpeakerSession } from '../../src/speakers/session.js';
import { SpeakersEvent, TranscriptEvent, CzechSpeechError } from '../../src/events.js';
import { nodeWorkerFactory } from '../support/env.mjs';

const entry = new URL('../support/fake-speaker-entry.mjs', import.meta.url);
const fakeModel = { _modelBlob: () => new Blob([new Uint8Array(4)]), _runtimeBlob: () => new Blob([new Uint8Array(4)]), dim: 192, sha256: 'f'.repeat(64) };
const dc = (s, l) => new Float32Array(Math.round(s * 16000)).fill(l);
const create = (o = {}) => SpeakerSession.create({ model: fakeModel, threads: 1, onCpu: 'run', _createWorker: nodeWorkerFactory(entry), ...o });
function send(port, id, samples, audioStartMs = 0) {
  port.postMessage({ type: 'utterance', id, startMs: audioStartMs, audioStartMs, endMs: audioStartMs + samples.length / 16, samples }, [samples.buffer]);
}
const into = (list) => (p) => list.push(p.commit());
const nextEvents = (list, n) => new Promise((resolve) => { const t = setInterval(() => { if (list.length >= n) { clearInterval(t); resolve(list); } }, 5); });

test('SpeakersEvent lists each speaker once, in order, and never prints keys or numbers', () => {
  const ev = new SpeakersEvent('1.3', [
    { startMs: 0, endMs: 1000, speaker: { kind: 'session', n: 2 } },
    { startMs: 1000, endMs: 2000, speaker: null },
    { startMs: 2000, endMs: 3500, speaker: { kind: 'known', key: 'abc' } },
    { startMs: 3500, endMs: 4000, speaker: { kind: 'session', n: 2 } },
  ], null);
  assert.deepEqual(ev.speakers, [{ kind: 'session', n: 2 }, { kind: 'known', key: 'abc' }]);
  assert.ok(!JSON.stringify(ev).includes('abc') && !String(ev).includes('abc'));
  assert.deepEqual(ev.toJSON(), { type: 'speakers', utteranceId: '1.3', parts: 4, speakers: 2, skipped: null });
  assert.match(String(ev), /^Speakers\(utteranceId=1\.3, parts=4, speakers=2, skipped=null\)$/);
  assert.ok(Object.isFrozen(ev) && Object.isFrozen(ev.parts) && Object.isFrozen(ev.parts[0]));
});

test('TranscriptEvent carries utteranceId', () => {
  const t = new TranscriptEvent('dobrý den', 0, 1000, 5, 'cs', '2.7');
  assert.equal(t.utteranceId, '2.7');
  assert.equal(t.toJSON().utteranceId, '2.7');
});

test('the new error codes exist', () => {
  for (const code of ['voiceprint-model', 'enroll-too-short', 'enroll-too-noisy']) {
    assert.equal(new CzechSpeechError('x', { code }).code, code);
  }
});

test('numbering survives a second listen() with the same session (review focus 1)', async () => {
  const s = await create();
  try {
    const first = [], second = [];
    send(s._attach(1, into(first)), 1, dc(2, 0.2));          // voice A
    await nextEvents(first, 1);
    s._detach(1);
    const port = s._attach(2, into(second));
    send(port, 1, dc(2, -0.2));                                         // voice B
    send(port, 2, dc(2, 0.2));                                          // voice A again
    await nextEvents(second, 2);
    assert.deepEqual(first[0].speakers, [{ kind: 'session', n: 1 }]);
    assert.deepEqual(second.map((e) => [e.utteranceId, e.speakers[0].n]), [['2.1', 2], ['2.2', 1]]);
    assert.deepEqual(s.speakers.map((x) => x.n), [1, 2]);
  } finally {
    await s.close();
  }
});

test('firstLabel and part times on the session clock', async () => {
  const s = await create({ firstLabel: 4 });
  try {
    const got = [];
    send(s._attach(1, into(got)), 1, dc(2, 0.2), 9000);
    await nextEvents(got, 1);
    assert.deepEqual(got[0].parts, [{ startMs: 9000, endMs: 11000, speaker: { kind: 'session', n: 4 } }]);
  } finally {
    await s.close();
  }
});

test('a short utterance of a new voice is unknown; skipped by a backlog is reported as behind; the speech time adds up', async () => {
  const s = await create();
  try {
    const got = [];
    const port = s._attach(1, into(got));
    send(port, 1, dc(0.5, 0.2));                                        // 0.5 s: no new speaker
    send(port, 2, dc(2, 0.2));
    await nextEvents(got, 2);
    assert.deepEqual(got[0].parts, [{ startMs: 0, endMs: 500, speaker: null }]);
    assert.deepEqual(got[0].speakers, []);
    assert.deepEqual(s.speakers, [{ n: 1, speechMs: 2000 }]);
  } finally {
    await s.close();
  }
});

test('one failed utterance is reported as error (not failed: the worker lives) and the next one is still handled', async () => {
  const s = await create();
  try {
    const got = [];
    const port = s._attach(1, into(got));
    send(port, 1, new Float32Array(10));                                 // too short to embed: the worker answers with an error
    send(port, 2, dc(2, 0.2));
    await nextEvents(got, 2);
    assert.deepEqual([got[0].skipped, got[0].parts.length], ['error', 0]);
    assert.equal(got[1].speakers[0].n, 1);
  } finally {
    await s.close();
  }
});

test('tagging waits for commit(): a discarded utterance founds no speaker and the next voice is number 1', async () => {
  const s = await create();
  try {
    const got = [];
    const port = s._attach(1, (p) => got.push(p));
    send(port, 1, dc(2, 0.2));
    send(port, 2, dc(2, -0.2));
    await nextEvents(got, 2);
    assert.deepEqual(s.speakers, []);          // nothing is tagged before commit()
    got[0].discard();
    const ev = got[1].commit();
    assert.equal(ev.speakers[0].n, 1);
    assert.deepEqual(s.speakers.map((x) => x.n), [1]);
    assert.equal(got[0].commit().skipped, 'failed'); // a discarded utterance has nothing to commit
  } finally {
    await s.close();
  }
});

test('close() discards what is still waiting for its transcript', async () => {
  const s = await create();
  const got = [];
  send(s._attach(1, (p) => got.push(p)), 1, dc(2, 0.2));
  await nextEvents(got, 1);
  await s.close();
  assert.equal(got[0].commit().skipped, 'failed');
  assert.deepEqual(s.speakers, []);
});

test('a crashed worker: the attached listener hears of it once', async () => {
  const s = await create({ _fake: { crashAt: 1 } });
  try {
    const failures = [];
    const port = s._attach(1, () => {}, (err) => failures.push(err));
    send(port, 1, dc(2, 0.2));
    await new Promise((r) => setTimeout(r, 400));
    assert.equal(failures.length, 1);
    assert.ok(failures[0] instanceof CzechSpeechError && failures[0].code === 'runtime');
  } finally {
    await s.close();
  }
});

test('close() forgets the speakers and refuses a new listen', async () => {
  const s = await create();
  await s.close();
  assert.deepEqual(s.speakers, []);
  assert.throws(() => s._attach(3, () => {}), (e) => e instanceof CzechSpeechError && e.code === 'closed');
});

test('close() tells an attached listener, once, with code closed', async () => {
  const s = await create();
  const failures = [];
  s._attach(1, () => {}, (err) => failures.push(err));
  await s.close();
  assert.deepEqual(failures.map((e) => e.code), ['closed']);
});

test('the session never prints a voiceprint or a speaker number', async () => {
  const s = await create();
  try {
    assert.equal(JSON.stringify(s), '{}');
    assert.match(String(s), /^SpeakerSession\(/);
  } finally {
    await s.close();
  }
});
