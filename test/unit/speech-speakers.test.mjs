// listen({ speakers }): the speakers event after each transcript, the bounded wait at the end, the numbers across two
// listen() calls, the CPU rule. Fake recogniser and fake speaker worker in real worker threads; "speech" is any frame with
// a sample above 0.05 in magnitude, and a "voice" is the sign of a constant level (fake speaker worker).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { setupWithSpeakers } from '../support/speech-setup.mjs';
import { ArrayAudioSource, Signals, within } from '../support/env.mjs';

const dc = (s, l) => new Float32Array(Math.round(s * 16000)).fill(l);
async function collect(iterable, onEvent) {
  const events = [];
  for await (const ev of iterable) {
    events.push(ev);
    if (onEvent) await onEvent(ev, events);
  }
  return events;
}
async function withSpeakers(options, fn) {
  const t = await setupWithSpeakers(options);
  try {
    return await fn(t);
  } finally {
    await t.close();
  }
}
const index = (events, type, id) => events.findIndex((e) => e.type === type && e.utteranceId === id);
const oneVoice = () => new ArrayAudioSource(Signals.concat(dc(2, 0.2), Signals.zeros(1)));

test('each transcript is followed by exactly one speakers event with its utteranceId, never before it', () =>
  withSpeakers({ speaker: { delayMs: 300 } }, async ({ speech, speakerSession }) => {
    const source = new ArrayAudioSource(Signals.concat(dc(2, 0.2), Signals.zeros(1), dc(2, -0.2), Signals.zeros(1)));
    const events = await collect(speech.listen({ _source: source, speakers: speakerSession }));
    const transcripts = events.filter((e) => e.type === 'transcript');
    const speakers = events.filter((e) => e.type === 'speakers');
    assert.deepEqual(transcripts.map((e) => e.utteranceId), ['1.1', '1.2']);
    assert.deepEqual(speakers.map((e) => e.utteranceId), ['1.1', '1.2']);
    for (const id of ['1.1', '1.2']) assert.ok(index(events, 'speakers', id) > index(events, 'transcript', id), id);
    assert.deepEqual(speakers.map((e) => e.speakers[0].n), [1, 2]);
  }));

test('a fast speaker worker still delivers speakers after its transcript', () =>
  withSpeakers({ speaker: { delayMs: 0 }, engine: { workMs: 300 } }, async ({ speech, speakerSession }) => {
    const events = await collect(speech.listen({ _source: oneVoice(), speakers: speakerSession }));
    assert.deepEqual(events.filter((e) => e.type !== 'speechStarted' && e.type !== 'speechEnded').map((e) => e.type), ['transcript', 'speakers']);
  }));

test('an utterance without a transcript has no speakers event', () =>
  withSpeakers({ engine: { emptyTextAt: 1 } }, async ({ speech, speakerSession }) => {
    const events = await collect(speech.listen({ _source: oneVoice(), speakers: speakerSession }));
    assert.equal(events.filter((e) => e.type === 'transcript').length, 0);
    assert.equal(events.filter((e) => e.type === 'speakers').length, 0);
  }));

test('an utterance without a transcript founds no speaker: the next new voice is still number 1', () =>
  withSpeakers({ engine: { emptyTextAt: 1 } }, async ({ speech, speakerSession }) => {
    const source = new ArrayAudioSource(Signals.concat(dc(2, 0.2), Signals.zeros(1), dc(2, -0.2), Signals.zeros(1)));
    const events = await collect(speech.listen({ _source: source, speakers: speakerSession }));
    assert.deepEqual(events.filter((e) => e.type === 'transcript').map((e) => e.utteranceId), ['1.2']);
    assert.deepEqual(events.filter((e) => e.type === 'speakers').map((e) => e.speakers.map((x) => x.n)), [[1]]);
    assert.deepEqual(speakerSession.speakers.map((x) => x.n), [1]);
  }));

test('stop(): the last utterance gets its speaker before the loop ends (review focus 4)', () =>
  withSpeakers({ speaker: { delayMs: 800 } }, async ({ speech, speakerSession }) => {
    // stop() comes right after the transcript, while the speaker worker still computes for 800 ms
    const source = new ArrayAudioSource(Signals.concat(dc(3, 0.2), Signals.zeros(1)), { tail: () => 0 });
    const events = await collect(speech.listen({ _source: source, speakers: speakerSession }), (e) => { if (e.type === 'transcript') speech.stop(); });
    assert.equal(events.at(-1).type, 'speakers');
    assert.equal(speech.state.speakers.lost, 0);
  }));

test('stop(): a speaker worker that never answers is waited for a bounded time and counted as lost', () =>
  withSpeakers({ speaker: { delayMs: 60000 }, extra: { _speakersDrainMs: 300 } }, async ({ speech, speakerSession }) => {
    const t0 = Date.now();
    const events = await collect(speech.listen({ _source: oneVoice(), speakers: speakerSession }));
    assert.ok(Date.now() - t0 < within(5000));
    assert.equal(events.filter((e) => e.type === 'transcript').length, 1);
    assert.equal(events.filter((e) => e.type === 'speakers').length, 0);
    assert.equal(speech.state.speakers.lost, 1);
  }));

test('a crashed speaker worker does not block the end; transcripts all arrive (review focus 4)', () =>
  withSpeakers({ speaker: { crashAt: 1 } }, async ({ speech, speakerSession }) => {
    const t0 = Date.now();
    const events = await collect(speech.listen({ _source: new ArrayAudioSource(Signals.concat(dc(2, 0.2), Signals.zeros(1), dc(2, 0.2), Signals.zeros(1))), speakers: speakerSession }));
    assert.ok(Date.now() - t0 < within(10000));
    assert.equal(events.filter((e) => e.type === 'transcript').length, 2);
    assert.deepEqual(events.filter((e) => e.type === 'speakers').map((e) => e.skipped), ['failed', 'failed']);
  }));

test('one failing embedding is reported as error, not failed: the worker lives and the next utterance is handled', () =>
  withSpeakers({ speaker: { failFirst: 1 } }, async ({ speech, speakerSession }) => {
    const source = new ArrayAudioSource(Signals.concat(dc(2, 0.2), Signals.zeros(1), dc(2, 0.2), Signals.zeros(1)));
    const events = await collect(speech.listen({ _source: source, speakers: speakerSession }));
    const sp = events.filter((e) => e.type === 'speakers');
    assert.deepEqual(sp.map((e) => e.skipped), ['error', null]);
    assert.equal(sp[1].speakers[0].n, 1);
  }));

test('a speaker worker that died before listen(): every transcript is followed by a failed speakers event', () =>
  withSpeakers({ speaker: { crashAt: 1 } }, async ({ speech, speakerSession }) => {
    await collect(speech.listen({ _source: oneVoice(), speakers: speakerSession }));
    const events = await collect(speech.listen({ _source: oneVoice(), speakers: speakerSession }));
    assert.deepEqual(events.filter((e) => e.type === 'speakers').map((e) => [e.utteranceId, e.skipped]), [['2.1', 'failed']]);
  }));

test('the same SpeakerSession across two listen() calls keeps the numbers (review focus 1)', () =>
  withSpeakers({}, async ({ speech, speakerSession }) => {
    const a = await collect(speech.listen({ _source: oneVoice(), speakers: speakerSession }));
    const b = await collect(speech.listen({ _source: new ArrayAudioSource(Signals.concat(dc(2, -0.2), Signals.zeros(1), dc(2, 0.2), Signals.zeros(1))), speakers: speakerSession }));
    const ns = (evs) => evs.filter((e) => e.type === 'speakers').map((e) => [e.utteranceId, e.speakers[0].n]);
    assert.deepEqual(ns(a), [['1.1', 1]]);
    assert.deepEqual(ns(b), [['2.1', 2], ['2.2', 1]]);
  }));

test('on the CPU with onCpu skip: speakers are not computed and every event says cpu', () =>
  withSpeakers({ onCpu: 'skip' }, async ({ speech, speakerSession }) => {
    const events = await collect(speech.listen({ _source: oneVoice(), speakers: speakerSession }));
    assert.deepEqual(events.filter((e) => e.type === 'speakers').map((e) => [e.skipped, e.parts.length]), [['cpu', 0]]);
    assert.deepEqual(speakerSession.speakers, []);
  }));

test('the capture worker sends the speech span only: no trailing silence', () => {
  const lengths = [];
  return withSpeakers({ speaker: { reportLengths: true }, onWorkerMessage: (m) => { if (m.type === 'fake' && m.length) lengths.push(m.length); } },
    async ({ speech, speakerSession }) => {
      await collect(speech.listen({ _source: new ArrayAudioSource(Signals.concat(Signals.zeros(0.32), dc(2, 0.2), Signals.zeros(1))), speakers: speakerSession }));
      // 320 ms of pre-roll (10 frames) + 2 s of speech (63 frames), without the 600 ms of silence that follows
      assert.deepEqual(lengths, [(10 + Math.ceil(2000 / 32)) * 512]);
    });
});

test('state.speakers describes the last listen() with speakers; none without', () =>
  withSpeakers({}, async ({ speech, speakerSession }) => {
    assert.equal(speech.state.speakers, null);
    await collect(speech.listen({ _source: oneVoice(), speakers: speakerSession }));
    assert.deepEqual(speech.state.speakers, { active: false, threads: 1, pending: 0, unsplit: 1, skipped: 0, lost: 0 });
  }));

test('listen() refuses something that is not a SpeakerSession', () =>
  withSpeakers({}, async ({ speech }) => {
    await assert.rejects(collect(speech.listen({ _source: new ArrayAudioSource(Signals.zeros(1)), speakers: {} })), TypeError);
  }));

test('without speakers listen() is as before: transcripts carry an utteranceId and no speakers events come', () =>
  withSpeakers({}, async ({ speech }) => {
    const events = await collect(speech.listen({ _source: oneVoice() }));
    assert.equal(events.find((e) => e.type === 'transcript').utteranceId, '1.1');
    assert.equal(events.filter((e) => e.type === 'speakers').length, 0);
  }));
