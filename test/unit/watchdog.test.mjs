// The WebGPU watchdog and the way back to WebGPU after a failure, with fake time: superviseCall (src/worker-core.js)
// stops a call that overruns its limit cooperatively and kills the engine only when the call does not stop within a
// grace period; GpuRecovery (src/backend.js) decides when 'auto' tries WebGPU again after a failure and when it gives
// up. The same behaviour through the whole pipeline with fake engines is in test/unit/backend.test.mjs.
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { superviseCall } from '../../src/worker-core.js';
import { GpuRecovery } from '../../src/backend.js';

/** setTimeout/clearTimeout on a clock that only moves when advance() is called. */
function fakeClock() {
  let now = 0;
  let seq = 0;
  const timers = new Map();
  const flush = async () => { for (let i = 0; i < 5; i++) await Promise.resolve(); };
  return {
    now: () => now,
    setTimeout(fn, ms) {
      const id = ++seq;
      timers.set(id, { at: now + ms, fn });
      return id;
    },
    clearTimeout(id) {
      timers.delete(id);
    },
    get pending() {
      return timers.size;
    },
    /** Moves the clock forward by `ms`, running due timers in order (and the promise jobs they start). */
    async advance(ms) {
      const end = now + ms;
      await flush();
      for (;;) {
        const due = [...timers].filter(([, t]) => t.at <= end).sort((a, b) => a[1].at - b[1].at)[0];
        if (!due) break;
        timers.delete(due[0]);
        now = due[1].at;
        due[1].fn();
        await flush();
      }
      now = end;
      await flush();
    },
  };
}

/** A call the test settles by hand. */
function pendingCall() {
  const c = {};
  c.promise = new Promise((resolve, reject) => { c.resolve = resolve; c.reject = reject; });
  return c;
}

function supervised(clock, call, { stop = true, graceMs = 3000 } = {}) {
  const log = [];
  const p = superviseCall(() => call.promise, {
    limitMs: 10000, graceMs, what: 'WebGPU: recognition', timers: clock,
    stop: stop ? () => log.push(`stop@${clock.now()}`) : null,
    kill: (err) => log.push(`kill@${clock.now()}: ${err.message}${err.watchdog ? ' (watchdog)' : ''}`),
  });
  const outcome = { settled: false };
  p.then((v) => Object.assign(outcome, { settled: true, value: v }), (e) => Object.assign(outcome, { settled: true, error: e }));
  return { log, outcome };
}

describe('superviseCall', () => {
  test('a call that ends within its limit: its result, nothing stopped or killed, no timer left', async () => {
    const clock = fakeClock();
    const call = pendingCall();
    const { log, outcome } = supervised(clock, call);
    await clock.advance(9999);
    assert.equal(outcome.settled, false);
    call.resolve({ status: 0 });
    await clock.advance(1);
    assert.deepEqual(outcome.value, { value: { status: 0 }, stopped: false });
    assert.deepEqual(log, []);
    assert.equal(clock.pending, 0);
    await clock.advance(60000);
    assert.deepEqual(log, [], 'no late stop or kill');
  });

  test('a call that fails within its limit: its error', async () => {
    const clock = fakeClock();
    const call = pendingCall();
    const { log, outcome } = supervised(clock, call);
    call.reject(new Error('ggml aborted'));
    await clock.advance(1);
    assert.equal(outcome.error.message, 'ggml aborted');
    assert.deepEqual(log, []);
    assert.equal(clock.pending, 0);
  });

  test('over its limit the call is asked to stop; when it stops within the grace, nothing is killed', async () => {
    const clock = fakeClock();
    const call = pendingCall();
    const { log, outcome } = supervised(clock, call);
    await clock.advance(10000);
    assert.deepEqual(log, ['stop@10000'], 'asked to stop exactly at the limit');
    await clock.advance(2999);
    assert.equal(outcome.settled, false);
    call.resolve({ status: 1 });
    await clock.advance(1);
    assert.deepEqual(outcome.value, { value: { status: 1 }, stopped: true });
    await clock.advance(60000);
    assert.deepEqual(log, ['stop@10000'], 'never killed');
    assert.equal(clock.pending, 0);
  });

  test('a call that does not stop within the grace is killed, and its late end changes nothing', async () => {
    const clock = fakeClock();
    const call = pendingCall();
    const { log, outcome } = supervised(clock, call);
    await clock.advance(12999);
    assert.deepEqual(log, ['stop@10000']);
    assert.equal(outcome.settled, false);
    await clock.advance(1);
    assert.deepEqual(log, ['stop@10000', 'kill@13000: WebGPU: recognition did not finish in 10 s (watchdog)']);
    assert.equal(outcome.error.message, 'WebGPU: recognition did not finish in 10 s');
    assert.equal(outcome.error.watchdog, true);
    call.resolve({ status: 0 });
    await clock.advance(1);
    assert.equal(outcome.value, undefined);
    assert.equal(clock.pending, 0);
  });

  test('a call that fails while it is being stopped gives that failure, and is not killed again', async () => {
    const clock = fakeClock();
    const call = pendingCall();
    const { log, outcome } = supervised(clock, call);
    await clock.advance(10500);
    call.reject(new Error('WebGPU device lost'));
    await clock.advance(1);
    assert.equal(outcome.error.message, 'WebGPU device lost');
    await clock.advance(60000);
    assert.deepEqual(log, ['stop@10000']);
  });

  test('without a way to stop it (model load, free), the call is killed at the limit', async () => {
    const clock = fakeClock();
    const call = pendingCall();
    const { log, outcome } = supervised(clock, call, { stop: false });
    await clock.advance(9999);
    assert.deepEqual(log, []);
    await clock.advance(1);
    assert.deepEqual(log, ['kill@10000: WebGPU: recognition did not finish in 10 s (watchdog)']);
    assert.equal(outcome.error.watchdog, true);
  });
});

describe('GpuRecovery', () => {
  const delaysMs = [10000, 60000, 300000];

  test('nothing to recover before a failure', () => {
    const clock = fakeClock();
    const r = new GpuRecovery({ delaysMs, now: clock.now });
    assert.equal(r.due, false);
    assert.equal(r.gaveUp, false);
  });

  test('after a failure WebGPU is due again after the first delay, not before', async () => {
    const clock = fakeClock();
    const r = new GpuRecovery({ delaysMs, now: clock.now });
    await clock.advance(5000);
    r.failed();
    await clock.advance(9999);
    assert.equal(r.due, false);
    await clock.advance(1);
    assert.equal(r.due, true);
    assert.equal(r.gaveUp, false);
  });

  test('each further failure in a row waits longer; after one failure more than there are delays it gives up', async () => {
    const clock = fakeClock();
    const r = new GpuRecovery({ delaysMs, now: clock.now });
    r.failed();                     // t = 0: due at 10 s
    await clock.advance(10000);
    assert.equal(r.due, true);
    r.failed();                     // the re-created engine failed too: due 60 s later
    await clock.advance(59999);
    assert.equal(r.due, false);
    await clock.advance(1);
    assert.equal(r.due, true);
    r.failed();                     // third in a row: 300 s
    await clock.advance(299999);
    assert.equal(r.due, false);
    await clock.advance(1);
    assert.equal(r.due, true);
    r.failed();                     // fourth in a row: the CPU from now on
    assert.equal(r.gaveUp, true);
    await clock.advance(1e9);
    assert.equal(r.due, false, 'never due again');
  });

  test('the state travels between the page and a recogniser as failures and the time left, whatever each clock reads', async () => {
    // the page owns it (src/speech.js); each recogniser starts from it and reports every change back
    const page = fakeClock();
    const worker = fakeClock();
    await page.advance(123456);
    await worker.advance(42);
    const w = new GpuRecovery({ delaysMs, now: worker.now });
    w.adopt({ failures: 0, retryInMs: null });
    assert.deepEqual(w.snapshot(), { failures: 0, retryInMs: null, gaveUp: false });
    w.failed();
    assert.deepEqual(w.snapshot(), { failures: 1, retryInMs: 10000, gaveUp: false });
    const p = new GpuRecovery({ delaysMs, now: page.now });
    p.adopt(w.snapshot());
    await page.advance(4000);
    assert.deepEqual(p.snapshot(), { failures: 1, retryInMs: 6000, gaveUp: false });
    assert.equal(p.due, false);
    const next = new GpuRecovery({ delaysMs, now: worker.now }); // the next recogniser
    next.adopt(p.snapshot());
    await worker.advance(5999);
    assert.equal(next.due, false);
    await worker.advance(1);
    assert.equal(next.due, true);
    next.failed();
    assert.deepEqual(next.snapshot(), { failures: 2, retryInMs: 60000, gaveUp: false }, 'the count goes on');
    for (let i = 0; i < 2; i++) next.failed();
    p.adopt(next.snapshot());
    assert.deepEqual(p.snapshot(), { failures: 4, retryInMs: null, gaveUp: true });
  });

  test('a recognition that succeeds on WebGPU resets the count', async () => {
    const clock = fakeClock();
    const r = new GpuRecovery({ delaysMs, now: clock.now });
    r.failed();
    await clock.advance(10000);
    r.failed();
    await clock.advance(60000);
    r.succeeded();
    assert.equal(r.due, false, 'nothing to recover now');
    r.failed();
    await clock.advance(9999);
    assert.equal(r.due, false);
    await clock.advance(1);
    assert.equal(r.due, true, 'back to the first delay');
    for (let i = 0; i < 3; i++) r.failed();
    assert.equal(r.gaveUp, true, 'and to the full number of tries');
  });
});
