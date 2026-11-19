import { test } from 'node:test';
import assert from 'node:assert/strict';
import { SpeakerEngine } from '../../src/speakers/engine.js';
import { STRICT } from '../../src/speakers/points.js';
import { CzechSpeechError } from '../../src/events.js';
import { Signals, nodeWorkerFactory } from '../support/env.mjs';

const entry = new URL('../support/fake-speaker-entry.mjs', import.meta.url);
const dc = (seconds, level) => new Float32Array(Math.round(seconds * 16000)).fill(level);
const fakeModel = { _modelBlob: () => new Blob([new Uint8Array(4)]), _runtimeBlob: () => new Blob([new Uint8Array(4)]), dim: 192, sha256: 'f'.repeat(64) };
const make = (fake = {}) => SpeakerEngine.create({ model: fakeModel, threads: 1, point: STRICT, _createWorker: nodeWorkerFactory(entry), _fake: fake });

test('embed() returns one normalised vector', async () => {
  const e = await make();
  const v = await e.embed(dc(1, 0.2));
  assert.equal(v.length, 192);
  assert.deepEqual([v[0], v[1]], [1, 0]);
  await e.close();
});

test('an utterance of two voices is split where the voice changes (2 s + 2 s)', async () => {
  const e = await make();
  const r = await e.embedUtterance(Signals.concat(dc(2, 0.2), dc(2, -0.2)));
  assert.equal(r.mode, 'split');
  assert.deepEqual(r.pieces.map((p) => [p.startMs, p.endMs]), [[0, 2000], [2000, 4000]]);
  assert.deepEqual([r.pieces[0].vector[0], r.pieces[1].vector[1]], [1, 1]);
  await e.close();
});

test('utterances shorter than 3 s are not split', async () => {
  const e = await make();
  const r = await e.embedUtterance(Signals.concat(dc(1.4, 0.2), dc(1.4, -0.2)));
  assert.deepEqual([r.mode, r.pieces.length], ['whole', 1]);
  await e.close();
});

test('a backlog first stops splitting, then skips (oldest waiting first)', async () => {
  // #1 starts at once (nothing waits yet) and takes 50 ms; #2-#7 arrive meanwhile.
  // #2 starts with 5 waiting -> skipped; #3-#5 with 4, 3, 2 waiting -> whole; #6, #7 with 1, 0 -> split.
  const e = await make({ delayMs: 50 });
  const four = () => Signals.concat(dc(2, 0.2), dc(2, -0.2));
  const modes = (await Promise.all(Array.from({ length: 7 }, () => e.embedUtterance(four())))).map((r) => r.mode);
  assert.deepEqual(modes, ['split', 'skipped', 'whole', 'whole', 'whole', 'split', 'split']);
  await e.close();
});

test('close() zeroes what the worker holds, terminates it and rejects pending work', async () => {
  const factory = nodeWorkerFactory(entry);
  const zeroed = [];
  const e = await SpeakerEngine.create({ model: fakeModel, threads: 1, point: STRICT, _createWorker: factory,
    _fake: { delayMs: 500, reportZeroed: true, reportLengths: true },
    _onWorkerMessage: (m) => { if (m.type === 'fake' && m.zeroed !== undefined) zeroed.push(m.zeroed); else if (m.type === 'fake') { embedsAfterRelease.push(zeroed.length); } } });
  const embedsAfterRelease = [];
  const pending = e.embedUtterance(dc(4, 0.2));
  const rejected = assert.rejects(pending, (err) => err instanceof CzechSpeechError && err.code === 'closed');
  await new Promise((r) => setTimeout(r, 100));
  await e.close();
  await new Promise((r) => setTimeout(r, 700));
  assert.ok(embedsAfterRelease.length > 0, 'an embedding was running');
  assert.ok(embedsAfterRelease.every((n) => n === 0), 'no embedding started after the release');
  await rejected;
  assert.deepEqual(zeroed, [true]);
  assert.ok(factory.created.every((w) => w.terminated));
});

test('a backlog is noticed although the embedder blocks the worker (busy-wait, like the real runtime)', async () => {
  const e = await make({ busyMs: 50 });
  const four = () => Signals.concat(dc(2, 0.2), dc(2, -0.2));
  const modes = (await Promise.all(Array.from({ length: 7 }, () => e.embedUtterance(four())))).map((r) => r.mode);
  assert.ok(modes.includes('skipped'), modes.join());
  assert.ok(modes.includes('whole'), modes.join());
  await e.close();
});

test('close() during blocking work rejects pending work promptly and does not embed after the release', async () => {
  const e = await make({ busyMs: 100 });
  const pending = e.embedUtterance(Signals.concat(dc(10, 0.2), dc(10, -0.2)));
  await new Promise((r) => setTimeout(r, 150));
  const t0 = performance.now();
  await e.close();
  assert.ok(performance.now() - t0 < 1000, `close took ${performance.now() - t0} ms`);
  await assert.rejects(pending, (err) => err instanceof CzechSpeechError && err.code === 'closed');
});

// --- start-up -------------------------------------------------------------------------------------------------------

test('a worker that never answers ready fails the start-up with code runtime and is terminated', async () => {
  const factory = nodeWorkerFactory(entry);
  await assert.rejects(SpeakerEngine.create({ model: fakeModel, threads: 1, point: STRICT, _createWorker: factory,
    _fake: { neverReady: true }, _startupTimeoutMs: 150 }), (err) => err instanceof CzechSpeechError && err.code === 'runtime');
  assert.ok(factory.created.every((w) => w.terminated));
});

test('a failure before the init message is sent (unreadable runtime file) terminates the worker', async () => {
  const factory = nodeWorkerFactory(entry);
  // something that cannot be sent to a worker: postMessage throws before anything is answered
  const model = { ...fakeModel, _runtimeBlob: () => () => {} };
  await assert.rejects(SpeakerEngine.create({ model, threads: 1, point: STRICT, _createWorker: factory }),
    (err) => err instanceof CzechSpeechError && err.code === 'runtime');
  assert.ok(factory.created.every((w) => w.terminated));
});

// --- attached listeners -----------------------------------------------------------------------------------------------

test('an attached listener hears of a worker failure, and of a close', async () => {
  const e = await make({ crashAt: 1 });
  const failures = [];
  e.attach(1, () => {}, (err) => failures.push(err));
  await assert.rejects(e.embedUtterance(dc(4, 0.2)), (err) => err.code === 'runtime');
  assert.equal(failures.length, 1);
  assert.equal(failures[0].code, 'runtime');
  await e.close();

  const e2 = await make();
  const closed = [];
  e2.attach(1, () => {}, (err) => closed.push(err.code));
  await e2.close();
  assert.deepEqual(closed, ['closed']);
});

test('detach() ends one listener: no more calls, its id can be used again', async () => {
  const e = await make();
  const got = [];
  const port = e.attach(1, (m) => got.push(['a', m.id]), () => {});
  port.postMessage({ type: 'utterance', id: 1, samples: dc(1, 0.2) });
  await new Promise((r) => setTimeout(r, 200));
  assert.deepEqual(got, [['a', 1]]);
  e.detach(1);
  e.detach(1); // twice, or an unknown id: nothing happens
  e.detach(99);
  const port2 = e.attach(1, (m) => got.push(['b', m.id]), () => {});
  port2.postMessage({ type: 'utterance', id: 2, samples: dc(1, 0.2) });
  await new Promise((r) => setTimeout(r, 200));
  assert.deepEqual(got, [['a', 1], ['b', 2]]);
  await e.close();
});

test('close() after a failure does not wait for the worker', async () => {
  const e = await make({ crashAt: 1 });
  await assert.rejects(e.embedUtterance(dc(4, 0.2)));
  const t0 = performance.now();
  await e.close();
  assert.ok(performance.now() - t0 < 1000);
});

// --- input ------------------------------------------------------------------------------------------------------------

test('bad input is refused for that call only; the engine keeps working', async () => {
  const e = await make();
  await assert.rejects(e.embed(new Float32Array(50)), RangeError);
  await assert.rejects(e.embed(new Float32Array(0)), RangeError);
  await assert.rejects(e.embed(new Float64Array(1000)), TypeError);
  await assert.rejects(e.embed(null), TypeError);
  await assert.rejects(e.embedUtterance(new Float32Array(10)), RangeError);
  const detached = new Float32Array(1000);
  structuredClone(detached, { transfer: [detached.buffer] });
  await assert.rejects(e.embed(detached), RangeError);
  assert.equal((await e.embed(dc(1, 0.2))).length, 192);
  await e.close();
});

test('a failing embedding fails that request only (embed, embedUtterance, attached listener)', async () => {
  const e = await make({ failFirst: 1 });
  await assert.rejects(e.embed(dc(1, 0.2)), (err) => err instanceof CzechSpeechError && err.code === 'runtime');
  assert.equal((await e.embed(dc(1, 0.2))).length, 192);
  await e.close();

  const e2 = await make({ failFirst: 1 });
  await assert.rejects(e2.embedUtterance(dc(1, 0.2)), (err) => err instanceof CzechSpeechError && err.code === 'runtime');
  assert.equal((await e2.embedUtterance(dc(1, 0.2))).pieces.length, 1);
  await e2.close();

  const e3 = await make({ failFirst: 1 });
  const got = [];
  const port = e3.attach(1, (m) => got.push(m), () => {});
  port.postMessage({ type: 'utterance', id: 1, samples: dc(1, 0.2) });
  port.postMessage({ type: 'utterance', id: 2, samples: dc(1, 0.2) });
  await new Promise((r) => setTimeout(r, 300));
  assert.equal(got.length, 2);
  assert.equal(typeof got[0].error, 'string');
  assert.equal(got[1].error, undefined);
  assert.equal(got[1].pieces.length, 1);
  await e3.close();
});

// --- the input cap ----------------------------------------------------------------------------------------------------

const lengthsOf = async (maxInputMs, samples) => {
  const lengths = [];
  const e = await SpeakerEngine.create({ model: fakeModel, threads: 1, point: STRICT, maxInputMs, _createWorker: nodeWorkerFactory(entry),
    _fake: { reportLengths: true }, _onWorkerMessage: (m) => m.type === 'fake' && lengths.push(m.length) });
  const v = await e.embed(samples);
  await e.close();
  return { lengths, v };
};

test('an embedding is never computed from more than 10 s: longer audio goes in equal chunks, averaged', async () => {
  const exact = await lengthsOf(undefined, dc(10, 0.2));
  assert.deepEqual(exact.lengths, [160000]);
  const long = await lengthsOf(undefined, dc(25, 0.2));
  assert.deepEqual(long.lengths, [133334, 133334, 133332]);
  assert.ok(Math.abs(long.v[0] - 1) < 1e-6 && Math.abs(long.v[1]) < 1e-6);
  assert.deepEqual((await lengthsOf(undefined, dc(10.001, 0.2))).lengths.length, 2);
});

test('maxInputMs sets the cap (and must be at least the 1.5 s change-search window)', async () => {
  assert.deepEqual((await lengthsOf(15000, dc(25, 0.2))).lengths, [200000, 200000]);
  await assert.rejects(SpeakerEngine.create({ model: fakeModel, point: STRICT, maxInputMs: 10, _createWorker: nodeWorkerFactory(entry) }), RangeError);
  await assert.rejects(SpeakerEngine.create({ model: fakeModel, point: STRICT, maxInputMs: 1499, _createWorker: nodeWorkerFactory(entry) }), RangeError);
  const e = await SpeakerEngine.create({ model: fakeModel, point: STRICT, maxInputMs: 1500, _createWorker: nodeWorkerFactory(entry) });
  await e.close();
});

test('a long utterance with no change point is embedded in capped chunks, a window never exceeds the cap', async () => {
  const lengths = [];
  const e = await SpeakerEngine.create({ model: fakeModel, threads: 1, point: STRICT, _createWorker: nodeWorkerFactory(entry),
    _fake: { reportLengths: true }, _onWorkerMessage: (m) => m.type === 'fake' && lengths.push(m.length) });
  const r = await e.embedUtterance(dc(24, 0.2));
  await e.close();
  assert.equal(r.pieces.length, 1);
  assert.ok(lengths.length > 0, 'the embedder was called');
  assert.ok(lengths.every((n) => n <= 160000), lengths.join());
});

// --- re-review fixes ----------------------------------------------------------------------------------------------------

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const until = async (cond, ms = 3000) => {
  const t0 = performance.now();
  while (!cond()) {
    if (performance.now() - t0 > ms) throw new Error('timed out waiting');
    await sleep(10);
  }
};

test('a re-used listenId gets nothing of the previous attachment: queued and in-flight work is dropped', async () => {
  const embeds = [];
  const e = await SpeakerEngine.create({ model: fakeModel, threads: 1, point: STRICT, _createWorker: nodeWorkerFactory(entry),
    _fake: { delayMs: 40, reportLengths: true }, _onWorkerMessage: (m) => m.type === 'fake' && m.length && embeds.push(m.length) });
  const first = [];
  const port = e.attach(1, (m) => first.push(m.id), () => {});
  port.postMessage({ type: 'utterance', id: 7, samples: Signals.concat(dc(2, 0.2), dc(2, -0.2)) }); // in flight: many window embeddings
  port.postMessage({ type: 'utterance', id: 8, samples: dc(1, 0.2) });                                  // queued behind it
  await until(() => embeds.length >= 2);
  e.detach(1);
  const second = [];
  const port2 = e.attach(1, (m) => second.push(m.id), () => {});
  await sleep(600); // the in-flight utterance would need ~40 windows x 40 ms
  const seen = embeds.length;
  await sleep(300);
  assert.equal(embeds.length, seen, 'no embedding is computed any more');
  assert.ok(seen < 12, `the in-flight utterance was stopped early (${seen} embeddings)`);
  assert.deepEqual([first, second], [[], []]);
  port2.postMessage({ type: 'utterance', id: 9, samples: dc(1, 0.2) });
  await until(() => second.length === 1);
  assert.deepEqual(second, [9]);
  await e.close();
});

test('an aborted utterance leaves no embedding unzeroed in the worker', async () => {
  const msgs = [];
  const factory = nodeWorkerFactory(entry);
  const e = await SpeakerEngine.create({ model: fakeModel, threads: 1, point: STRICT, _createWorker: factory,
    _fake: { delayMs: 30, trackOut: true, reportLengths: true }, _onWorkerMessage: (m) => m.type === 'fake' && msgs.push(m) });
  const port = e.attach(1, () => {}, () => {});
  port.postMessage({ type: 'utterance', id: 1, samples: Signals.concat(dc(2, 0.2), dc(2, -0.2)) });
  await until(() => msgs.filter((m) => m.length).length >= 3);
  e.detach(1);
  await sleep(300);
  factory.created[0].postMessage({ type: 'fakeCheck' });
  await until(() => msgs.some((m) => m.out));
  const out = msgs.find((m) => m.out).out;
  assert.ok(out.count >= 3, `vectors were kept (${out.count})`);
  assert.equal(out.zero, true);
  await e.close();
});

test('close() resolves promptly when the worker dies while it is closing', async () => {
  const factory = nodeWorkerFactory(entry);
  const e = await SpeakerEngine.create({ model: fakeModel, threads: 1, point: STRICT, _createWorker: factory, _fake: { busyMs: 1500 } });
  const pending = e.embedUtterance(dc(1, 0.2)).catch(() => {});
  await sleep(100);
  const closing = e.close();
  await sleep(50);
  const t0 = performance.now();
  factory.created[0].onerror({ message: 'boom' });
  await closing;
  assert.ok(performance.now() - t0 < 500, `close took ${performance.now() - t0} ms after the crash`);
  await pending;
});

test('utterances on a port are validated: a bad one gets its own error, the queue and a later close are fine', async () => {
  const e = await make();
  const got = [];
  const failures = [];
  const port = e.attach(1, (m) => got.push(m), (err) => failures.push(err.code));
  port.postMessage({ type: 'utterance', id: 1, samples: new Float64Array(1000) });
  port.postMessage({ type: 'utterance', id: 2, samples: new Float32Array(10) });
  port.postMessage({ type: 'utterance', id: 3 });
  port.postMessage({ type: 'utterance', id: 4, samples: dc(1, 0.2) });
  await until(() => got.length === 4);
  assert.deepEqual(got.map((m) => m.id).sort(), [1, 2, 3, 4]);
  for (const id of [1, 2, 3]) assert.equal(typeof got.find((m) => m.id === id).error, 'string');
  assert.equal(got.find((m) => m.id === 4).error, undefined);
  await e.close();
  assert.deepEqual(failures, ['closed']);
});

test('after close() no piece reaches a listener; it hears one closed failure', async () => {
  const factory = nodeWorkerFactory(entry);
  const e = await SpeakerEngine.create({ model: fakeModel, threads: 1, point: STRICT, _createWorker: factory });
  const events = [];
  const sent = [];
  const w = factory.created[0];
  const post = w.postMessage.bind(w);
  w.postMessage = (m, t) => { if (m && m.type === 'attach') sent.push(m.listenId); return post(m, t); };
  e.attach(1, (m) => events.push(['pieces', m.id]), (err) => events.push(['failure', err.code]));
  const vector = Float32Array.of(1, 0);
  const closing = e.close();
  // a piece message that was already on its way when close() was called
  w.onmessage({ data: { type: 'pieces', listenId: sent[0], id: 5, mode: 'whole', pieces: [{ startMs: 0, endMs: 1, vector }], computeMs: 1 } });
  await closing;
  assert.deepEqual(events, [['failure', 'closed']]);
  assert.deepEqual([...vector], [0, 0], 'the late piece was zeroed');
});

test('audio over a SharedArrayBuffer is refused with a clear error', async () => {
  const e = await make();
  const shared = new Float32Array(new SharedArrayBuffer(4000));
  await assert.rejects(e.embed(shared), (err) => err instanceof TypeError && /SharedArrayBuffer/.test(err.message));
  await assert.rejects(e.embedUtterance(shared), TypeError);
  assert.equal((await e.embed(dc(1, 0.2))).length, 192);
  await e.close();
});

test('a postMessage that throws (DataCloneError) rejects with a CzechSpeechError and the engine goes on', async () => {
  const factory = nodeWorkerFactory(entry);
  const e = await SpeakerEngine.create({ model: fakeModel, threads: 1, point: STRICT, _createWorker: factory });
  const w = factory.created[0];
  const post = w.postMessage.bind(w);
  let fail = true;
  w.postMessage = (m, t) => { if (fail && m && (m.type === 'embed' || m.type === 'utterance')) throw new DOMException('cannot clone', 'DataCloneError'); return post(m, t); };
  await assert.rejects(e.embed(dc(1, 0.2)), (err) => err instanceof CzechSpeechError && /could not be sent/.test(err.message));
  await assert.rejects(e.embedUtterance(dc(1, 0.2)), (err) => err instanceof CzechSpeechError);
  fail = false;
  assert.equal((await e.embed(dc(1, 0.2))).length, 192);
  await e.close();
});
