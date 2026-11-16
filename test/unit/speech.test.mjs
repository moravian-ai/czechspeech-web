// The listen()/transcribe() pipeline with a fake engine in real worker threads: the Android SDK's SpeechEngineTest
// ported, plus the browser-specific lifecycle (abort signals, the no-SharedArrayBuffer fallback, worker
// termination). "Speech" is any frame with a sample above 0.05 in magnitude. No model, no WebAssembly: always runs.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import v8 from 'node:v8';
import { CzechSpeech, CzechSpeechModel, CzechSpeechError } from '../../src/index.js';
import { ArrayAudioSource, FRAME_SAMPLES, Signals, TestEnv, nodeWorkerFactory, within } from '../support/env.mjs';

const fakeEntry = new URL('../support/fake-worker-entry.mjs', import.meta.url);
const vadModel = fs.readFileSync(TestEnv.vadAsset);
const utterance = Signals.concat(Signals.zeros(0.5), Signals.tone(1.0), Signals.zeros(1.0));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function setup({ engine, config, keep, noSab, modelPath = 'fake-model.bin', abortGraceMs } = {}) {
  const fake = { capture: null, recognizer: null, events: [] };
  const logs = [];
  const factory = nodeWorkerFactory(fakeEntry);
  const speech = await CzechSpeech.create({
    model: CzechSpeechModel._fromPath(modelPath, 1000),
    config,
    keepModelLoaded: keep,
    log: (level, message) => logs.push(`${level}: ${message}`),
    _createWorker: factory,
    _vadModel: vadModel,
    _engineOptions: engine,
    _noSharedMemory: noSab,
    _abortGraceMs: abortGraceMs,
    _onWorkerMessage: (role, m) => {
      if (m.type === 'fake') {
        fake[role] = m;
        fake.events.push(`${role}:${m.event}`);
      }
    },
  });
  return { speech, factory, fake, logs };
}

async function collect(iterable, onEvent) {
  const events = [];
  for await (const ev of iterable) {
    events.push(ev);
    if (onEvent && (await onEvent(ev, events)) === 'break') break;
  }
  return events;
}

async function until(cond, ms = 10000, what = 'condition') {
  const t0 = Date.now();
  while (!cond()) {
    if (Date.now() - t0 > within(ms)) throw new Error(`timeout waiting for ${what}`);
    await sleep(5);
  }
}

const allTerminated = (factory) => factory.created.every((w) => w.terminated);
const live = (factory) => factory.created.filter((w) => !w.terminated).length;
const noTextInLogs = (logs) => assert.ok(!logs.some((l) => /dobrý|den/.test(l)), logs.join('\n'));

/**
 * Workers without a thread, for races that need exact timing: `init` is answered with `ready` synchronously (or never,
 * with `start: 'never'`; `start` may also be { recognizer, capture }), `transcribe`, `free` and `stop` on the next
 * microtask. `posted` lists what each worker received, `delivered` the replies it has handed to the page.
 */
function scriptedWorkers({ start = 'instant' } = {}) {
  const created = [];
  const factory = (role) => {
    const w = {
      role,
      terminated: false,
      posted: [],
      delivered: [],
      inits: [],
      onmessage: null,
      postMessage(m) {
        if (w.terminated) return;
        w.posted.push(m.type);
        if (m.type === 'init') w.inits.push(m);
        const reply = (data) => queueMicrotask(() => {
          if (w.terminated) return;
          w.delivered.push(data.type);
          w.onmessage({ data });
        });
        const mode = typeof start === 'object' ? start[role] : start;
        if (m.type === 'init' && mode === 'instant') w.onmessage({ data: { type: 'ready', backend: 'cpu', heapBytes: 1 } });
        if (m.type === 'transcribe') reply({ type: 'transcribed', id: m.id, text: 'dobrý den' });
        if (m.type === 'free') reply({ type: 'freed', heapBytes: 1 });
        if (m.type === 'stop') reply({ type: 'stopped', heapBytes: 1 });
      },
      terminate() { w.terminated = true; },
    };
    created.push(w);
    return w;
  };
  factory.created = created;
  return factory;
}

async function setupScripted(options, extra = {}) {
  const factory = scriptedWorkers(options);
  const speech = await CzechSpeech.create({ model: CzechSpeechModel._fromPath('fake-model.bin', 1000), log: false,
    _createWorker: factory, _vadModel: vadModel, ...extra });
  return { speech, factory };
}

test('listen emits started, ended, transcript, then closes the source and frees the model', async () => {
  const { speech, factory, fake, logs } = await setup();
  const source = new ArrayAudioSource(utterance);
  const events = await collect(speech.listen({ _source: source }));

  assert.deepEqual(events.map((e) => e.type), ['speechStarted', 'speechEnded', 'transcript']);
  const t = events[2];
  assert.equal(t.text, 'dobrý den');
  assert.equal(t.startMs, 480); // the tone starts at 0.5 s, inside frame 15 (480..512 ms)
  assert.ok(t.endMs >= 1500 && t.endMs <= 1540, `endMs ${t.endMs}`);
  assert.ok(!String(t).includes('dobrý') && !JSON.stringify(t).includes('dobrý'));

  assert.ok(source.opened && source.closed);
  assert.equal(speech.state.listening, false);
  assert.equal(speech.state.modelLoaded, false);
  assert.equal(speech.state.modelLoads, 1);
  await until(() => allTerminated(factory), 2000, 'workers terminated');
  assert.equal(fake.recognizer.counters.whisperInits, 1);
  assert.equal(fake.recognizer.counters.whisperFrees, 1);
  assert.equal(fake.capture.counters.vadInits + fake.recognizer.counters.vadInits, 2, 'streaming VAD + gate VAD');
  assert.equal(fake.capture.counters.vadFrees + fake.recognizer.counters.vadFrees, 2);
  assert.equal(fake.capture.counters.vadResets, 1, 'reset after the utterance (the reset after init is in C)');
  assert.equal(fake.capture.counters.wipes, 1);
  assert.equal(fake.recognizer.counters.wipes, 1);
  assert.ok(fake.capture.framesZeroed, 'every captured frame zeroed after use');
  assert.equal(fake.recognizer.whisperInputs, 1);
  assert.ok(fake.recognizer.whisperInputsZeroed, 'utterance buffer zeroed after use');
  noTextInLogs(logs);
});

test('listen() events carry the language the engine recognised in; the engine gets the languages of the model', async () => {
  const langs = [];
  const mk = async (meta, listenOptions) => {
    const speech = await CzechSpeech.create({
      model: CzechSpeechModel._fromPath('fake-model.bin', 1000, meta), log: false,
      _createWorker: nodeWorkerFactory(fakeEntry), _vadModel: vadModel,
      _onWorkerMessage: (role, m) => { if (m.type === 'fake' && m.event === 'transcribeStarted') langs.push(m.languages); },
    });
    try {
      return await collect(speech.listen({ _source: new ArrayAudioSource(utterance), ...listenOptions }));
    } finally {
      await speech.close();
    }
  };
  const cs = await mk({});
  assert.equal(cs[2].language, 'cs');
  const pl = await mk({ languages: ['pl', 'en'] }, { language: 'en' });
  assert.equal(pl[2].language, 'en');
  const both = await mk({ languages: ['pl', 'en'] });
  assert.equal(both[2].language, 'pl');
  assert.deepEqual(langs, ['cs', 'en', 'pl,en']);
});

test('listen({ language }) outside the languages of the model is a RangeError at the call', async () => {
  const { speech, fake } = await setup();
  await assert.rejects(collect(speech.listen({ language: 'de', _source: new ArrayAudioSource(utterance) })), RangeError);
  assert.equal(fake.recognizer, null, 'no recogniser was started');
  assert.equal(speech.state.listening, false);
  const ok = await collect(speech.listen({ language: 'cs', _source: new ArrayAudioSource(utterance) }));
  assert.equal(ok[2].language, 'cs', 'a refused call leaves the instance usable');
});

test('no speech means no events and the recogniser never runs', async () => {
  const { speech, fake } = await setup();
  const events = await collect(speech.listen({ _source: new ArrayAudioSource(Signals.zeros(10)) }));
  assert.equal(events.length, 0);
  assert.equal(fake.recognizer.whisperInputs, 0);
  assert.equal(speech.state.modelLoaded, false);
  assert.equal(speech.state.modelLoads, 0, 'a session without speech never loads the model');
});

test('transcribe() gates before loading the model', async () => {
  const { speech, fake } = await setup();
  assert.equal(await speech.transcribe(Signals.zeros(3)), '');
  assert.equal(speech.state.modelLoads, 0, 'model not loaded for silence');
  assert.equal(fake.recognizer.counters.gateCalls, 1);
  assert.equal(fake.recognizer.counters.whisperInits, 0);
  assert.equal(await speech.transcribe(new Float32Array(0)), '');
  await assert.rejects(speech.transcribe([0, 1]), TypeError);
});

test('transcribe() loads and frees the model and never touches the caller\'s array', async () => {
  const { speech, factory } = await setup();
  const samples = Signals.tone(1.0);
  const copy = samples.slice();
  assert.equal(await speech.transcribe(samples), 'dobrý den');
  assert.deepEqual(samples, copy);
  assert.equal(speech.state.modelLoaded, false);
  assert.equal(speech.state.modelLoads, 1);
  await speech.transcribe(samples);
  assert.equal(speech.state.modelLoads, 2, 'loaded again for the second call');
  assert.ok(allTerminated(factory));
});

test('results without letters or digits are dropped', async () => {
  const { speech } = await setup({ engine: { text: ' ... , ...' } });
  assert.equal(await speech.transcribe(Signals.tone(1.0)), '');
  const events = await collect(speech.listen({ _source: new ArrayAudioSource(utterance) }));
  assert.deepEqual(events.map((e) => e.type), ['speechStarted', 'speechEnded']);
  const digits = await setup({ engine: { text: '91' } });
  assert.equal(await digits.speech.transcribe(Signals.tone(1.0)), '91');
});

test('an empty result is dropped silently and the utterance is decoded once', async () => {
  // a model that learned to answer "no speech" with an empty transcript (suppress_blank is off in native/)
  for (const empty of ['', ' ', '\n']) {
    const { speech, fake } = await setup({ engine: { text: empty } });
    assert.equal(await speech.transcribe(Signals.tone(1.0)), '', JSON.stringify(empty));
    assert.equal(fake.recognizer.whisperInputSamples.length, 1);
    const events = await collect(speech.listen({ _source: new ArrayAudioSource(utterance) }));
    assert.deepEqual(events.map((e) => e.type), ['speechStarted', 'speechEnded'], JSON.stringify(empty));
    assert.equal(fake.recognizer.whisperInputSamples.length, 1, 'one pass for the utterance (the session has its own recogniser)');
  }
});

test('a repetition loop is removed before the transcript reaches the app; the log has counts only', async () => {
  const loop = 'pojišťování vývozu se státní podporou v roce 2012 až se státní podporou v roce 2012 až se státní ' +
    'podporou v roce 2012 až se státní podporou v roce 2012 až se st';
  const cleaned = 'pojišťování vývozu se státní podporou v roce 2012 až';
  const { speech, logs } = await setup({ engine: { text: loop } });
  const events = await collect(speech.listen({ _source: new ArrayAudioSource(utterance) }));
  assert.deepEqual(events.map((e) => e.type), ['speechStarted', 'speechEnded', 'transcript']);
  assert.equal(events[2].text, cleaned);
  assert.equal(await speech.transcribe(Signals.tone(1.0)), cleaned);
  const removed = logs.filter((l) => l.includes('repetition loop removed'));
  assert.deepEqual(removed, ['info: utterance: repetition loop removed (7 words, 3 repeats, partial copy at the end)',
    'info: transcribe: repetition loop removed (7 words, 3 repeats, partial copy at the end)']);
  assert.ok(!logs.some((l) => /pojišťování|státní|podporou/.test(l)), logs.join('\n'));
});

test('transcribe() during listen() uses the session\'s model', async () => {
  const { speech } = await setup();
  const source = new ArrayAudioSource(utterance, { tail: () => 0, frameDelayMs: 1 });
  let during = null;
  await collect(speech.listen({ _source: source }), async (ev) => {
    if (ev.type !== 'transcript') return null;
    assert.equal(speech.state.modelLoaded, true);
    during = await speech.transcribe(Signals.tone(0.5));
    assert.equal(speech.state.modelLoaded, true, 'the session still holds the model');
    return 'break';
  });
  assert.equal(during, 'dobrý den');
  assert.equal(speech.state.modelLoads, 1);
  assert.equal(speech.state.modelLoaded, false);
  assert.ok(source.closed);
});

test('leaving the loop aborts a running recognition through SharedArrayBuffer and frees the model', async () => {
  const { speech, fake, factory } = await setup({ engine: { blockUntilAborted: true } });
  const source = new ArrayAudioSource(utterance, { tail: () => 0, frameDelayMs: 1 });
  const ac = new AbortController();
  const run = collect(speech.listen({ _source: source, signal: ac.signal }));
  await until(() => fake.events.includes('recognizer:transcribeStarted'), 10000, 'transcription start');
  const t0 = Date.now();
  ac.abort();
  await assert.rejects(run, (e) => e.name === 'AbortError');
  const stopMs = Date.now() - t0;
  assert.ok(source.closed);
  assert.equal(speech.state.abortedRecognitions, 1, 'stopped by the abort callback');
  assert.equal(speech.state.modelLoaded, false);
  assert.equal(speech.state.listening, false);
  assert.equal(fake.recognizer.counters.whisperFrees, 1, 'freed gracefully (not terminated)');
  assert.ok(fake.recognizer.whisperInputsZeroed);
  assert.ok(stopMs < within(1000), `stopped ${stopMs} ms after the abort`);
  await until(() => allTerminated(factory), 2000, 'workers terminated');
});

test('without SharedArrayBuffer a busy recogniser is terminated at once', async () => {
  const { speech, fake, factory } = await setup({ engine: { blockUntilAborted: true }, noSab: true });
  assert.equal(speech.state.runtime, 'wasm-single');
  assert.equal(speech.state.threads, 1);
  const source = new ArrayAudioSource(utterance, { tail: () => 0, frameDelayMs: 1 });
  const t0 = { v: 0 };
  await collect(speech.listen({ _source: source }), async () => {
    await until(() => fake.events.includes('recognizer:transcribeStarted'), 10000, 'transcription start');
    t0.v = Date.now();
    return 'break';
  });
  const stopMs = Date.now() - t0.v;
  assert.ok(stopMs < within(1500), `stopped ${stopMs} ms after break (the fake would block for 10 s)`);
  assert.ok(source.closed);
  assert.equal(speech.state.modelLoaded, false);
  assert.ok(allTerminated(factory));
});

test('a recognition that ignores the abort is terminated after the grace period', async () => {
  // blockMs 3000 with no abort check: the fake never looks at the flag
  const { speech, fake, factory } = await setup({ engine: { workMs: 3000 }, abortGraceMs: 300 });
  const source = new ArrayAudioSource(utterance, { tail: () => 0, frameDelayMs: 1 });
  let t0 = 0;
  await collect(speech.listen({ _source: source }), async () => {
    await until(() => fake.events.includes('recognizer:transcribeStarted'), 10000, 'transcription start');
    t0 = Date.now();
    return 'break';
  });
  const stopMs = Date.now() - t0;
  assert.ok(stopMs < within(1500), `stopped ${stopMs} ms after break`);
  assert.equal(speech.state.modelLoaded, false);
  assert.ok(allTerminated(factory));
});

test('one session at a time', async () => {
  const { speech } = await setup();
  const source = new ArrayAudioSource(Signals.zeros(1), { tail: () => 0, frameDelayMs: 1 });
  const ac = new AbortController();
  const first = collect(speech.listen({ _source: source, signal: ac.signal }));
  await until(() => speech.state.listening, 5000, 'listening');
  await assert.rejects(speech.listen({ _source: new ArrayAudioSource(Signals.zeros(1)) }).next(),
    (e) => e instanceof CzechSpeechError && e.code === 'busy' && /already/.test(e.message));
  ac.abort();
  await assert.rejects(first, (e) => e.name === 'AbortError');
  assert.equal(speech.state.listening, false);
  assert.ok(source.closed);
  // and a new session can start afterwards
  const events = await collect(speech.listen({ _source: new ArrayAudioSource(utterance) }));
  assert.equal(events.filter((e) => e.type === 'transcript').length, 1);
});

/** Every transcript comes after the speechEnded of its own speech (a fast source may run ahead of recognition). */
function assertTranscriptsFollowTheirSpeech(events) {
  const ends = [];
  events.forEach((e, i) => { if (e.type === 'speechEnded') ends.push({ i, atMs: e.atMs }); });
  events.forEach((e, i) => {
    if (e.type !== 'transcript' && e.type !== 'error') return;
    if (e.type === 'transcript') {
      const own = ends.find((x) => x.atMs === e.endMs);
      assert.ok(own && own.i < i, `transcript ${e} before its speechEnded`);
    }
  });
  const speech = events.filter((e) => e.type.startsWith('speech')).map((e) => e.type);
  for (let k = 0; k < speech.length; k++) assert.equal(speech[k], k % 2 === 0 ? 'speechStarted' : 'speechEnded');
}

test('close() cancels the session and rejects new work', async () => {
  const { speech, factory } = await setup();
  const source = new ArrayAudioSource(utterance, { tail: () => 0, frameDelayMs: 1 });
  const run = collect(speech.listen({ _source: source }));
  await until(() => speech.state.modelLoaded, 5000, 'model loaded');
  await speech.close();
  await assert.rejects(run, (e) => e.code === 'closed');
  assert.equal(speech.state.modelLoaded, false);
  assert.ok(source.closed);
  await assert.rejects(speech.transcribe(Signals.tone(1.0)), (e) => e.code === 'closed');
  await assert.rejects(speech.listen({ _source: new ArrayAudioSource(utterance) }).next(), (e) => e.code === 'closed');
  assert.ok(allTerminated(factory));
});

test('close() during the recogniser start-up cancels it: the waiting call rejects with closed, nothing is loaded', async () => {
  const { speech, factory } = await setup({ engine: { startDelayMs: 400 } });
  const call = speech.transcribe(Signals.tone(1.0));
  call.catch(() => {});
  await sleep(50); // the recogniser worker is starting
  assert.equal(live(factory), 1);
  const t0 = Date.now();
  await speech.close();
  const closeMs = Date.now() - t0;
  assert.equal(live(factory), 0, 'close() resolved only after the worker was terminated');
  assert.ok(closeMs < within(300), `close() waited ${closeMs} ms instead of cancelling the start-up`);
  await assert.rejects(call, (e) => e instanceof CzechSpeechError && e.code === 'closed');
  await sleep(500); // past the end of the start-up the worker would have had
  assert.equal(speech.state.modelLoads, 0, 'the model was never loaded');
  assert.equal(speech.state.modelLoaded, false);
});

test('close() as the recogniser becomes ready: the waiting call rejects with closed and is never sent', async () => {
  const { speech, factory } = await setupScripted();
  // the worker answers 'ready' synchronously: transcribe() is waiting for that answer when close() is called
  const call = speech.transcribe(Signals.tone(1.0));
  call.catch(() => {});
  await speech.close();
  await assert.rejects(call, (e) => e.code === 'closed');
  const [rec] = factory.created;
  assert.ok(!rec.posted.includes('transcribe'), rec.posted.join(','));
  assert.ok(rec.terminated);
});

test('close() or an abort in the same tick as the first next() of listen() leaves no worker running', async () => {
  // Regression: the capture worker was created a microtask later, after the session had already stopped
  const liveRoles = (factory) => factory.created.filter((w) => !w.terminated).map((w) => w.role);
  for (const start of ['instant', 'never']) {
    for (const how of ['close', 'abort']) {
      const what = `${how}, ${start} start-up`;
      const { speech, factory } = await setupScripted({ start });
      const ac = new AbortController();
      const source = new ArrayAudioSource(utterance, { tail: () => 0, frameDelayMs: 1 });
      const it = speech.listen({ signal: ac.signal, _source: source })[Symbol.asyncIterator]();
      const first = it.next();
      first.catch(() => {});
      if (how === 'close') {
        await speech.close();
      } else {
        ac.abort();
        await assert.rejects(first, (e) => e.name === 'AbortError', what);
      }
      assert.deepEqual(liveRoles(factory), [], `${what}: no worker alive`);
      assert.deepEqual(factory.created.map((w) => w.role).sort(), ['capture', 'recognizer'], what);
      await sleep(20);
      assert.deepEqual(liveRoles(factory), [], `${what}: none started later either`);
      if (how === 'close') await assert.rejects(first, (e) => e.code === 'closed', what);
      await speech.close();
      assert.ok(source.opened ? source.closed : true, `${what}: the source is closed`);
    }
  }
});

test('close() a few microtasks after transcribe(): nothing is sent after close() began, the call rejects with closed', async () => {
  // Regression: with close() one microtask after the call, the audio was still posted and the call resolved
  const cases = new Set();
  for (let k = 0; k <= 8; k++) {
    const { speech, factory } = await setupScripted();
    const call = speech.transcribe(Signals.tone(1.0));
    call.catch(() => {});
    let tick = Promise.resolve();
    for (let i = 0; i < k; i++) tick = tick.then(() => {});
    let postedBeforeClose = null;
    let answeredBeforeClose = null;
    const closing = tick.then(() => {
      postedBeforeClose = factory.created[0].posted.length;
      answeredBeforeClose = factory.created[0].delivered.includes('transcribed');
      return speech.close();
    });
    const outcome = await call.then((text) => `resolved "${text}"`, (e) => `rejected ${e.code}`);
    await closing;
    const [rec] = factory.created;
    const sentAfterClose = rec.posted.slice(postedBeforeClose);
    assert.ok(!sentAfterClose.includes('transcribe'), `k=${k}: sent after close(): ${sentAfterClose}`);
    const sentBefore = rec.posted.slice(0, postedBeforeClose).includes('transcribe');
    // the transcript only when the recogniser had answered before close(); else closed, whether the audio was never
    // sent or sent and not answered yet (close() rejects every call it has not seen answered)
    const what = !sentBefore ? 'not sent' : answeredBeforeClose ? 'answered' : 'sent, not answered';
    cases.add(what);
    assert.equal(outcome, what === 'answered' ? 'resolved "dobrý den"' : 'rejected closed', `k=${k}: ${what} before close()`);
    assert.ok(rec.terminated, `k=${k}`);
  }
  assert.deepEqual([...cases].sort(), ['answered', 'not sent', 'sent, not answered'], 'the sweep meets every case');
});

test('every worker gets the URLs of its modules and their .wasm files (the glue never looks for one by its name)', async () => {
  for (const extra of [{}, { assetBaseUrl: 'https://app.example/static/czechspeech/' }]) {
    const factory = scriptedWorkers();
    const speech = await CzechSpeech.create({ model: CzechSpeechModel._fromPath('fake-model.bin', 1000), log: false,
      _createWorker: factory, _vadModel: vadModel, ...extra });
    speech.listen({ _source: new ArrayAudioSource(utterance, { tail: () => 0, frameDelayMs: 1 }) }).next().catch(() => {});
    await speech.close();
    const inits = factory.created.flatMap((w) => w.inits);
    assert.deepEqual(inits.map((m) => m.role).sort(), ['capture', 'recognizer']);
    for (const m of inits) {
      const urls = m.role === 'recognizer' ? [m.moduleUrl, m.wasmUrl, m.gpuModuleUrl, m.gpuWasmUrl] : [m.moduleUrl, m.wasmUrl];
      assert.ok(urls.every((u) => typeof u === 'string' && /\.(js|wasm)$/.test(u)), `${m.role}: ${urls}`);
      if (extra.assetBaseUrl) assert.ok(urls.every((u) => u.startsWith(extra.assetBaseUrl)), urls.join(', '));
    }
  }
});

test('concurrent close() calls resolve together, after the real close', async () => {
  const { speech, factory } = await setup({ engine: { workMs: 800 } });
  const source = new ArrayAudioSource(utterance, { tail: () => 0, frameDelayMs: 1 });
  const run = collect(speech.listen({ _source: source }));
  run.catch(() => {});
  await until(() => speech.state.modelLoaded, 5000, 'model loaded');
  const resolved = [];
  const first = speech.close().then(() => resolved.push(`first (${live(factory)} workers alive)`));
  const second = speech.close().then(() => resolved.push(`second (${live(factory)} workers alive)`));
  await Promise.all([first, second]);
  assert.deepEqual(resolved, ['first (0 workers alive)', 'second (0 workers alive)']);
  await speech.close(); // and later calls resolve too
  await assert.rejects(run, (e) => e.code === 'closed');
});

/**
 * nodeWorkerFactory(fakeEntry) whose next `n` workers of `role` cannot be constructed (`how: 'construct'`, as browsers
 * refuse a cross-origin module worker) or refuse to send one message type (`how: 'init'`, 'transcribe'...; as
 * postMessage throws a DataCloneError): they throw `error`.
 */
function failingWorkers({ role, error, how = 'construct', n = 1 }) {
  const inner = nodeWorkerFactory(fakeEntry);
  let left = n;
  const factory = (r) => {
    if (r === role && how === 'construct' && left > 0) {
      left--;
      throw error;
    }
    const w = inner(r);
    if (r === role && how !== 'construct' && left > 0) {
      left--;
      const post = w.postMessage;
      w.postMessage = (m, t) => {
        if (m.type === how) throw error;
        post.call(w, m, t);
      };
    }
    return w;
  };
  factory.created = inner.created;
  return factory;
}

async function setupWith(factory, extra = {}) {
  return CzechSpeech.create({ model: CzechSpeechModel._fromPath('fake-model.bin', 1000), log: false, _createWorker: factory,
    _vadModel: vadModel, ...extra });
}

test('a recogniser worker that cannot be constructed or initialised: a runtime error, and no lease is kept', async (t) => {
  for (const how of ['construct', 'init']) {
    const error = how === 'construct'
      ? new DOMException('Failed to construct \'Worker\': the script cannot be accessed from origin', 'SecurityError')
      : new DOMException('the model could not be cloned', 'DataCloneError');
    const factory = failingWorkers({ role: 'recognizer', error, how });
    const speech = await setupWith(factory);
    t.after(() => speech.close());
    const failure = await speech.transcribe(Signals.tone(1.0)).then(() => null, (e) => e);
    // the next call starts a recogniser and frees it afterwards, as without the failure
    assert.equal(await speech.transcribe(Signals.tone(1.0)), 'dobrý den');
    assert.equal(speech.state.modelLoaded, false, `${how}: the model was freed after the call`);
    assert.equal(live(factory), 0, `${how}: the recogniser was terminated`);
    assert.ok(failure instanceof CzechSpeechError && failure.code === 'runtime' && failure.cause === error,
      `${how}: ${failure && failure.name} code=${failure && failure.code}`);
  }
});

test('a transcribe() that could not be posted is not in flight: the recogniser is freed at once, not killed', async (t) => {
  // regression (6ce51a4): counted as in flight before it was posted, the call left the recogniser "busy", so it was not
  // freed but terminated after the abort grace period (1 s)
  const error = new DOMException('could not be cloned', 'DataCloneError');
  const factory = failingWorkers({ role: 'recognizer', error, how: 'transcribe' });
  const events = [];
  const speech = await setupWith(factory, { _onWorkerMessage: (role, m) => { if (m.type === 'fake') events.push(`${role}:${m.event}`); } });
  t.after(() => speech.close());
  const t0 = Date.now();
  await assert.rejects(speech.transcribe(Signals.tone(1.0)), (e) => e.code === 'runtime' && e.cause === error);
  const ms = Date.now() - t0;
  assert.ok(events.includes('recognizer:wipe'), `freed by the recogniser (heap wiped), not terminated: ${events}`);
  assert.ok(ms < within(800), `the call took ${ms} ms`);
  assert.equal(live(factory), 0);
});

test('a capture worker that cannot be constructed ends listen() with a runtime error and frees the recogniser', async (t) => {
  const error = new DOMException('Failed to construct \'Worker\'', 'SecurityError');
  const factory = failingWorkers({ role: 'capture', error });
  const speech = await setupWith(factory);
  t.after(() => speech.close());
  const source = new ArrayAudioSource(utterance, { tail: () => 0, frameDelayMs: 1 });
  await assert.rejects(collect(speech.listen({ _source: source })),
    (e) => e instanceof CzechSpeechError && e.code === 'runtime' && e.cause === error);
  assert.ok(source.closed);
  assert.equal(live(factory), 0);
  assert.equal(speech.state.listening, false);
  assert.equal(speech.state.modelLoaded, false);
});

test('without Web Workers (Node.js, server-side rendering) create() fails with code unsupported, before any fetch', async () => {
  assert.equal(typeof globalThis.Worker, 'undefined');
  const saved = globalThis.fetch;
  let fetched = 0;
  globalThis.fetch = async (...args) => { fetched++; return saved(...args); };
  try {
    await assert.rejects(CzechSpeech.create({ model: CzechSpeechModel._fromPath('fake-model.bin', 1000), log: false }),
      (e) => e instanceof CzechSpeechError && e.code === 'unsupported' && /Web Workers/.test(e.message));
  } finally {
    globalThis.fetch = saved;
  }
  assert.equal(fetched, 0, 'the VAD asset is not fetched');
});

/** Workers of the roles in `never` never finish starting (scripted, no thread); the others run the fake engine. */
function neverStarting(never) {
  const stuck = scriptedWorkers({ start: 'never' });
  const fake = nodeWorkerFactory(fakeEntry);
  const created = [];
  const factory = (role) => {
    const w = never.includes(role) ? stuck(role) : fake(role);
    created.push(w);
    return w;
  };
  factory.created = created;
  return factory;
}

// "About 100 ms" after the signal: nothing on the cancel path may wait for a start-up.
const CANCEL_MS = within(100);

test('aborting listen() while the workers are starting ends it at once and terminates them', async () => {
  for (const never of [['recognizer'], ['recognizer', 'capture']]) {
    const factory = neverStarting(never);
    const speech = await setupWith(factory);
    const ac = new AbortController();
    const source = new ArrayAudioSource(utterance, { tail: () => 0, frameDelayMs: 1 });
    const run = collect(speech.listen({ _source: source, signal: ac.signal }));
    run.catch(() => {});
    await until(() => source.opened && factory.created.length === 2, 5000, 'microphone open, workers starting');
    await sleep(50);
    const t0 = Date.now();
    ac.abort();
    await assert.rejects(run, (e) => e.name === 'AbortError');
    const ms = Date.now() - t0;
    assert.ok(ms <= CANCEL_MS, `${never}: the loop ended ${ms} ms after abort()`);
    assert.ok(source.closed);
    assert.equal(live(factory), 0, `${never}: every worker terminated`);
    assert.equal(speech.state.listening, false);
  }
});

test('close() while a session\'s workers are starting resolves at once', async () => {
  const factory = neverStarting(['recognizer', 'capture']);
  const speech = await setupWith(factory);
  const run = collect(speech.listen({ _source: new ArrayAudioSource(utterance, { tail: () => 0, frameDelayMs: 1 }) }));
  run.catch(() => {});
  await until(() => factory.created.length === 2, 5000, 'workers starting');
  const t0 = Date.now();
  await speech.close();
  const ms = Date.now() - t0;
  await assert.rejects(run, (e) => e.code === 'closed');
  assert.ok(ms <= CANCEL_MS, `close() resolved ${ms} ms after the call`);
  assert.equal(live(factory), 0);
});

test('an aborted transcribe() does not wait for the recogniser start-up', async () => {
  const factory = neverStarting(['recognizer']);
  const speech = await setupWith(factory);
  // two calls wait for the same start-up: the worker goes when the last of them is aborted
  const first = new AbortController();
  const second = new AbortController();
  const a = speech.transcribe(Signals.tone(1.0), { signal: first.signal });
  const b = speech.transcribe(Signals.tone(1.0), { signal: second.signal });
  await sleep(50);
  let t0 = Date.now();
  first.abort();
  await assert.rejects(a, (e) => e.name === 'AbortError');
  assert.ok(Date.now() - t0 <= CANCEL_MS, `the first call rejected ${Date.now() - t0} ms after abort()`);
  assert.equal(live(factory), 1, 'still starting for the second call');
  t0 = Date.now();
  second.abort();
  await assert.rejects(b, (e) => e.name === 'AbortError');
  assert.ok(Date.now() - t0 <= CANCEL_MS, `the second call rejected ${Date.now() - t0} ms after abort()`);
  assert.equal(live(factory), 0, 'nobody waits for the start-up any more: terminated');
  await speech.close();
});

// ---- the start-up limit: a worker that never answers 'ready' (a module that cannot start) ----

const STARTUP_MS = within(20);
const startupLimit = { _timeouts: { startup: STARTUP_MS } };
const LIMITED = { timeout: within(10000) }; // without the limit these calls never end
const notStarted = (who) => (e) => e instanceof CzechSpeechError && e.code === 'runtime' &&
  new RegExp(`^The ${who} did not start within [\\d.]+ s$`).test(e.message) && e.cause && e.cause.name === 'TimeoutError';

test('a recogniser that never starts fails the call with runtime after the start-up limit; the next call starts a new one', LIMITED, async () => {
  const { speech, factory } = await setupScripted({ start: 'never' }, startupLimit);
  await assert.rejects(speech.transcribe(Signals.tone(1.0)), notStarted('recogniser'));
  assert.equal(factory.created.length, 1);
  assert.ok(factory.created[0].terminated, 'the worker that did not start is terminated');
  await assert.rejects(speech.transcribe(Signals.tone(1.0)), notStarted('recogniser'));
  assert.equal(factory.created.length, 2, 'the next call started a new worker');
  assert.ok(factory.created[1].terminated);
  await speech.close();
});

test('a capture worker that never starts ends listen() with runtime after the start-up limit; the next session starts a new one', LIMITED, async () => {
  const { speech, factory } = await setupScripted({ start: { recognizer: 'instant', capture: 'never' } }, startupLimit);
  const roles = () => factory.created.map((w) => w.role).sort();
  for (let session = 1; session <= 2; session++) {
    const source = new ArrayAudioSource(utterance, { tail: () => 0, frameDelayMs: 1 });
    await assert.rejects(collect(speech.listen({ _source: source })), notStarted('capture worker'));
    assert.equal(live(factory), 0, `session ${session}: every worker terminated`);
    assert.ok(source.closed);
  }
  assert.deepEqual(roles(), ['capture', 'capture', 'recognizer', 'recognizer']);
  await speech.close();
});

test('the start-up limit against an abort and close(): the earlier one wins, and close() gives closed', LIMITED, async () => {
  const later = within(100);
  // an abort before the limit: AbortError, and the limit does nothing afterwards
  for (const start of [{ start: 'never' }, { start: { recognizer: 'instant', capture: 'never' } }]) {
    const { speech, factory } = await setupScripted(start, { _timeouts: { startup: later } });
    const ac = new AbortController();
    const call = start.start === 'never' ? speech.transcribe(Signals.tone(1.0), { signal: ac.signal })
      : collect(speech.listen({ signal: ac.signal, _source: new ArrayAudioSource(utterance, { tail: () => 0, frameDelayMs: 1 }) }));
    await sleep(10);
    ac.abort();
    await assert.rejects(call, (e) => e.name === 'AbortError');
    await sleep(later + within(50));
    assert.equal(live(factory), 0);
    await speech.close();
  }
  // the limit before an abort: runtime, and the abort afterwards changes nothing
  {
    const { speech, factory } = await setupScripted({ start: 'never' }, startupLimit);
    const ac = new AbortController();
    await assert.rejects(speech.transcribe(Signals.tone(1.0), { signal: ac.signal }), notStarted('recogniser'));
    ac.abort();
    assert.equal(live(factory), 0);
    await speech.close();
  }
  // close() before the limit: closed, for the call and for the session
  {
    const { speech, factory } = await setupScripted({ start: { recognizer: 'never', capture: 'never' } }, { _timeouts: { startup: later } });
    const call = speech.transcribe(Signals.tone(1.0));
    const session = collect(speech.listen({ _source: new ArrayAudioSource(utterance, { tail: () => 0, frameDelayMs: 1 }) }));
    await sleep(10);
    await speech.close();
    await assert.rejects(call, (e) => e.code === 'closed');
    await assert.rejects(session, (e) => e.code === 'closed');
    await sleep(later + within(50));
    assert.equal(live(factory), 0);
  }
  // close() after the limit has failed the call: the call keeps runtime, close() resolves
  {
    const { speech, factory } = await setupScripted({ start: 'never' }, startupLimit);
    await assert.rejects(speech.transcribe(Signals.tone(1.0)), notStarted('recogniser'));
    await speech.close();
    assert.equal(live(factory), 0);
  }
});

test('a failed utterance is an error event and the session goes on', async () => {
  const { speech } = await setup({ engine: { failNext: 1 } });
  const events = await collect(speech.listen({ _source: new ArrayAudioSource(Signals.concat(utterance, utterance)) }));
  const errors = events.filter((e) => e.type === 'error');
  assert.equal(errors.length, 1);
  assert.ok(errors[0].error instanceof CzechSpeechError);
  assert.equal(events.filter((e) => e.type === 'transcript').length, 1);
  assertTranscriptsFollowTheirSpeech(events);
  // the error belongs to the first utterance: it comes after the first speechEnded
  assert.ok(events.findIndex((e) => e.type === 'error') > events.findIndex((e) => e.type === 'speechEnded'));
  assert.equal(speech.state.modelLoaded, false);
});

test('a model that cannot be loaded ends the session with an exception', async () => {
  const { speech, fake } = await setup({ modelPath: '/nonexistent/model.bin' });
  const source = new ArrayAudioSource(utterance, { tail: () => 0, frameDelayMs: 1 });
  await assert.rejects(collect(speech.listen({ _source: source })), (e) => e.code === 'model' && /not found/.test(e.message));
  assert.ok(source.closed, 'microphone closed');
  assert.equal(speech.state.listening, false);
  assert.equal(fake.recognizer ? fake.recognizer.counters.whisperInits : 0, 0);
});

test('a file that is not a ggml model fails at session start, before any speech', async () => {
  const { speech } = await setup({ modelPath: 'not-ggml.bin' });
  const source = new ArrayAudioSource(Signals.zeros(1), { tail: () => 0, frameDelayMs: 1 });
  const t0 = Date.now();
  await assert.rejects(collect(speech.listen({ _source: source })), (e) => e.code === 'model' && /not a whisper.cpp/.test(e.message));
  assert.ok(Date.now() - t0 < within(3000));
  assert.ok(source.closed);
  assert.equal(speech.state.modelLoads, 0);
});

test('microphone errors end the session', async () => {
  const { speech, factory } = await setup();
  const denied = new CzechSpeechError('Microphone access was denied', { code: 'microphone-denied' });
  await assert.rejects(collect(speech.listen({ _source: new ArrayAudioSource(utterance, { failOpen: denied }) })),
    (e) => e === denied);
  assert.equal(speech.state.listening, false);
  assert.equal(speech.state.modelLoaded, false);
  await until(() => allTerminated(factory), 3000, 'workers terminated');
});

test('break closes the source at once and frees the model', async () => {
  const { speech, factory } = await setup();
  const source = new ArrayAudioSource(utterance, { tail: () => 0, frameDelayMs: 1 });
  const events = await collect(speech.listen({ _source: source }), (ev) => (ev.type === 'speechStarted' ? 'break' : null));
  assert.equal(events.length, 1);
  assert.ok(source.closed);
  assert.equal(speech.state.modelLoaded, false);
  assert.ok(allTerminated(factory));
});

test('events keep their order over two utterances; transcripts() yields only transcripts', async () => {
  const { speech } = await setup();
  const audio = Signals.concat(utterance, Signals.zeros(1), utterance);
  const events = await collect(speech.listen({ _source: new ArrayAudioSource(audio) }));
  assert.equal(events.length, 6);
  assertTranscriptsFollowTheirSpeech(events);
  // at real-time pace (a microphone) recognition finishes before the next speech: S E T S E T
  const paced = await collect(speech.listen({ _source: new ArrayAudioSource(audio, { frameDelayMs: 4 }) }));
  assert.deepEqual(paced.map((e) => e.type[0] + (e.type === 'speechEnded' ? 'e' : '')), ['s', 'se', 't', 's', 'se', 't']);
  const transcripts = await collect(speech.transcripts({ _source: new ArrayAudioSource(audio) }));
  assert.equal(transcripts.length, 2);
  assert.ok(transcripts.every((t) => t.type === 'transcript'));
  assert.ok(transcripts[1].startMs > transcripts[0].endMs);
});

test('an aborted transcribe() rejects at once and the recogniser is freed in the background', async () => {
  const { speech, factory } = await setup({ engine: { blockUntilAborted: true } });
  const ac = new AbortController();
  const p = speech.transcribe(Signals.tone(1.0), { signal: ac.signal });
  await until(() => speech.state.modelLoaded, 5000, 'model loaded');
  await sleep(20);
  ac.abort();
  await assert.rejects(p, (e) => e.name === 'AbortError');
  await until(() => allTerminated(factory), 3000, 'recogniser terminated');
  assert.equal(speech.state.abortedRecognitions, 1);
  assert.equal(speech.state.modelLoaded, false);
});

/**
 * Live objects made by `ctor` after a full garbage collection. v8.queryObjects() is experimental (Node.js 22+): its
 * ExperimentalWarning is kept out of the test output; only the count is used.
 */
function liveObjects(ctor) {
  const emit = process.emitWarning;
  process.emitWarning = (warning, ...rest) => {
    if (!/queryObjects/.test(String(warning && warning.message ? warning.message : warning))) emit.call(process, warning, ...rest);
  };
  try {
    return v8.queryObjects(ctor, { format: 'count' });
  } finally {
    process.emitWarning = emit;
  }
}

test('aborted transcribe() calls leave nothing behind when the recogniser is terminated instead of answering',
  { skip: typeof v8.queryObjects !== 'function' && 'v8.queryObjects() is not available (Node.js 22 or later)' }, async () => {
  // Without SharedArrayBuffer a busy recogniser cannot be told to stop: it is terminated and never answers the call.
  const { speech, factory } = await setup({ engine: { blockUntilAborted: true }, noSab: true });
  const abortedCall = async () => {
    const ac = new AbortController();
    const call = speech.transcribe(Signals.tone(1.0), { signal: ac.signal });
    await until(() => speech.state.modelLoaded, 5000, 'recognition running');
    ac.abort();
    await assert.rejects(call, (e) => e.name === 'AbortError');
    await until(() => live(factory) === 0, 3000, 'recogniser terminated');
  };
  // live promises after a full garbage collection: each call that is still remembered keeps about four
  const livePromises = () => liveObjects(Promise);
  await abortedCall();
  const before = livePromises();
  const calls = 20;
  for (let i = 0; i < calls; i++) await abortedCall();
  const grown = livePromises() - before;
  assert.ok(grown < calls / 2, `${grown} more live promises after ${calls} aborted calls`);
  await speech.close();
});

test('keepModelLoaded keeps one model across calls until close()', async () => {
  const { speech, factory } = await setup({ keep: true });
  await speech.transcribe(Signals.tone(1.0));
  await speech.transcribe(Signals.tone(1.0));
  await collect(speech.listen({ _source: new ArrayAudioSource(utterance) }));
  assert.equal(speech.state.modelLoads, 1);
  assert.equal(speech.state.modelLoaded, true);
  await speech.close();
  assert.equal(speech.state.modelLoaded, false);
  assert.ok(allTerminated(factory));
});

// ---- stop(): finish a session without losing the last utterance (the Android SDK's stop* cases, plus start-up) ----

const types = (events) => events.map((e) => e.type);
// A guard so that a stop() that does not work fails the test instead of hanging it.
const guard = () => AbortSignal.timeout(within(15000));

test('stop() finishes the utterance in progress, transcribes it, and the loop ends normally', async () => {
  const { speech, fake, factory, logs } = await setup();
  // Speech that never pauses: without stop() the utterance would only end at the length cap.
  const source = new ArrayAudioSource(Signals.zeros(0.5), { tail: (i) => 0.3 * Math.sin(i * 0.0864), frameDelayMs: 1 });
  const events = await collect(speech.listen({ _source: source, signal: guard() }), (ev) => {
    if (ev.type === 'speechStarted') speech.stop(); // speech confirmed and still going on
    return null;
  });
  assert.deepEqual(types(events), ['speechStarted', 'speechEnded', 'transcript'], 'the loop ended by itself');
  assert.equal(events[2].text, 'dobrý den');
  assert.equal(source.finishCalls, 1, 'the source was finished, not just closed');
  assert.ok(source.closed);
  assert.equal(speech.state.listening, false);
  assert.equal(speech.state.modelLoaded, false);
  assert.equal(speech.state.abortedRecognitions, 0, 'nothing aborted');
  await until(() => allTerminated(factory), 2000, 'workers terminated');
  assert.equal(fake.recognizer.counters.whisperFrees, 1);
  assert.ok(fake.recognizer.whisperInputsZeroed);
  assert.ok(fake.capture.framesZeroed);
  noTextInLogs(logs);
});

test('stop() transcribes utterances still waiting in the queue', async () => {
  // The first recognition takes 3 s, so the second utterance waits in the recogniser's queue.
  const { speech, fake } = await setup({ engine: { workMs: 3000 } });
  let tailFrames = 0;
  const source = new ArrayAudioSource(Signals.concat(utterance, utterance), {
    tail: (i) => { if (i % 512 === 0) tailFrames++; return 0; },
    frameDelayMs: 1,
  });
  const events = [];
  const run = collect(speech.listen({ _source: source, signal: guard() }), (ev) => { events.push(ev); return null; });
  await until(() => fake.events.includes('recognizer:transcribeStarted'), 10000, 'first recognition');
  await until(() => tailFrames >= 10, 10000, 'second utterance ended');   // it ended 1 s of audio before the tail
  assert.equal(events.filter((e) => e.type === 'transcript').length, 0, 'nothing recognised yet');
  speech.stop();
  const all = await run;
  assert.equal(all.filter((e) => e.type === 'transcript').length, 2, types(all).join(','));
  assert.ok(source.closed);
  assert.equal(speech.state.listening, false);
  assert.equal(speech.state.modelLoaded, false);
  assert.ok(fake.recognizer.whisperInputsZeroed);
});

test('stop() without a session does not stop the next one', async () => {
  const { speech } = await setup();
  speech.stop();
  const first = new ArrayAudioSource(utterance);
  const events = await collect(speech.listen({ _source: first, signal: guard() }));
  assert.equal(events.filter((e) => e.type === 'transcript').length, 1);
  assert.equal(first.finishCalls, 0, 'not finished by the earlier stop()');
  // and a stop() of one session does not reach the next one
  const stopped = new ArrayAudioSource(Signals.zeros(0.5), { tail: () => 0, frameDelayMs: 1 });
  const run = collect(speech.listen({ _source: stopped, signal: guard() }));
  await until(() => speech.state.listening && stopped.framesSent > 5, 5000, 'listening');
  speech.stop();
  speech.stop(); // twice is fine
  assert.deepEqual(await run, []);
  const next = new ArrayAudioSource(utterance, { frameDelayMs: 1 });
  const later = await collect(speech.listen({ _source: next, signal: guard() }));
  assert.deepEqual(types(later), ['speechStarted', 'speechEnded', 'transcript']);
  assert.equal(next.finishCalls, 0);
});

test('stop() during start-up ends the session cleanly', async () => {
  // (a) before the microphone (source) is open: it is closed as soon as it opens; nothing is recognised
  const a = await setup();
  const early = new ArrayAudioSource(utterance, { openDelayMs: 100 });
  const runA = collect(a.speech.listen({ _source: early, signal: guard() }));
  a.speech.stop();
  assert.deepEqual(await runA, []);
  assert.equal(early.framesSent === 0 || early.closed, true);
  assert.ok(early.closed, 'opened, then closed at once');
  assert.equal(a.speech.state.listening, false);
  assert.equal(a.speech.state.modelLoads, 0);
  await until(() => allTerminated(a.factory), 3000, 'workers terminated');
  // (b) the source is open, the workers are still starting: the audio captured meanwhile is still recognised
  const b = await setup({ engine: { startDelayMs: 600 } });
  const open = new ArrayAudioSource(utterance, { tail: () => 0, frameDelayMs: 1 });
  const runB = collect(b.speech.listen({ _source: open, signal: guard() }));
  await until(() => open.framesSent > 100, 5000, 'the utterance was captured');
  assert.equal(b.speech.state.listening, true);
  b.speech.stop();
  const events = await runB;
  assert.deepEqual(types(events), ['speechStarted', 'speechEnded', 'transcript']);
  assert.equal(open.finishCalls, 1);
  assert.ok(open.closed);
  assert.equal(b.speech.state.modelLoaded, false);
});

test('after stop(), leaving the loop still discards (stop does not change break/abort/close)', async () => {
  const { speech, fake } = await setup({ engine: { blockUntilAborted: true } });
  const source = new ArrayAudioSource(utterance, { tail: () => 0, frameDelayMs: 1 });
  const ac = new AbortController();
  const run = collect(speech.listen({ _source: source, signal: ac.signal }));
  await until(() => fake.events.includes('recognizer:transcribeStarted'), 10000, 'transcription start');
  speech.stop();
  ac.abort();
  await assert.rejects(run, (e) => e.name === 'AbortError');
  assert.equal(speech.state.abortedRecognitions, 1, 'the running recognition was aborted');
  assert.equal(speech.state.modelLoaded, false);
});

// stop() during speech waits for the end of the word: a pause of 256 ms (8 frames), at most 1 s (the Android SDK's
// SpeechEngineTest has the same cases). The source calls stop() right after sending frame number STOP_FRAME (1504 ms
// of audio); the stop request and the audio travel on different ports, so the capture worker sees it within a frame
// of that and logs where. Speech (a tone) starts at frame 16; the utterance's pre-roll at frame 6.
const STOP_FRAME = 47;
const STOP_MS = STOP_FRAME * 32;
const SPEECH_FROM = 16;
const PRE_ROLL_FROM = 6;
const endlessTone = (i) => 0.3 * Math.sin((2 * Math.PI * 220 * i) / 16000);
const toneSamples = (n) => Float32Array.from({ length: n }, (_, i) => endlessTone(i));
const silence = (frames) => new Float32Array(frames * FRAME_SAMPLES);

/**
 * Calls stop() and holds the source back until the request has reached the capture worker. The stop request and the
 * audio travel on different ports; a source that goes on sending (here: a frame per millisecond) would let a slow
 * machine run ahead of the request, and the worker would see stop() frames later than the test expects.
 */
async function stopAndWait(speech) {
  speech.stop();
  await sleep(within(100));
}

/** The capture worker's stop() line: where it saw stop() (ms of audio), and the line itself. */
function stopLine(logs) {
  const line = logs.find((l) => l.includes('capture: stop() at')) || '';
  const m = /stop\(\) at (\d+) ms of audio/.exec(line);
  return { line, atMs: m ? Number(m[1]) : NaN };
}

/** stop() did not wait, and the capture worker processed no frame after the one in flight when it was called. */
function assertNoWait(logs, stopFrame) {
  assert.match(stopLine(logs).line, /no word in progress/, logs.join('\n'));
  const cap = logs.map((l) => /capture: (\d+) ms of audio, VAD/.exec(l)).find(Boolean);
  assert.ok(cap && Number(cap[1]) <= (stopFrame + 1) * 32, `the capture worker processed ${cap && cap[1]} ms of audio`);
}

test('stop() during speech waits for a pause; the utterance keeps the speech after stop()', async () => {
  const { speech, fake, logs } = await setup();
  // the speech goes on for 400 ms after stop(), then silence
  const audio = Signals.concat(silence(SPEECH_FROM), toneSamples((STOP_FRAME - SPEECH_FROM) * FRAME_SAMPLES + 6400), silence(100));
  const source = new ArrayAudioSource(audio, { frameDelayMs: 1, onFrame: (n) => (n === STOP_FRAME ? stopAndWait(speech) : undefined) });
  const events = await collect(speech.listen({ _source: source, signal: guard() }));
  assert.deepEqual(types(events), ['speechStarted', 'speechEnded', 'transcript'], 'the loop ended by itself');
  // the speech ends 400 ms after stop() (1904 ms, inside frame 59); endMs is the end of that frame
  assert.equal(events[2].endMs, STOP_MS + 416);
  // the utterance: pre-roll from frame 6, the speech to frame 59, then the 8 frames of the pause (60..67)
  assert.deepEqual(fake.recognizer.whisperInputSamples, [(68 - PRE_ROLL_FROM) * FRAME_SAMPLES]);
  assert.match(stopLine(logs).line, /waited \d+ ms for a pause/);
  assert.equal(source.finishCalls, 1, 'the microphone ended through finish()');
  assert.ok(source.closed);
  assert.equal(speech.state.modelLoaded, false);
  noTextInLogs(logs);
});

test('stop() during speech that never pauses ends the utterance 1 s after stop()', async () => {
  const { speech, logs } = await setup();
  const source = new ArrayAudioSource(silence(SPEECH_FROM), {
    tail: endlessTone,
    frameDelayMs: 1,
    // a second stop() during the wait changes nothing
    onFrame: (n) => (n === STOP_FRAME ? stopAndWait(speech) : n === STOP_FRAME + 10 ? speech.stop() : undefined),
  });
  const events = await collect(speech.listen({ _source: source, signal: guard() }));
  const t = events.filter((e) => e.type === 'transcript');
  assert.equal(t.length, 1, types(events).join(','));
  const s = stopLine(logs);
  assert.ok(Math.abs(s.atMs - STOP_MS) <= 32, `the capture worker saw stop() at ${s.atMs} ms of audio`);
  const waited = t[0].endMs - s.atMs;
  assert.ok(waited >= 1000 - 32 && waited <= 1000 + 32, `ended ${waited} ms after stop()`);
  assert.match(s.line, /\(the limit\)/);
});

test('stop() without speech in progress finishes at once', async () => {
  // after an utterance has ended (more than minSilenceMs of silence) no word is in progress
  const a = await setup();
  const sa = new ArrayAudioSource(utterance, { tail: () => 0, frameDelayMs: 1, onFrame: (n) => (n === 100 ? stopAndWait(a.speech) : undefined) });
  assert.deepEqual(types(await collect(a.speech.listen({ _source: sa, signal: guard() }))), ['speechStarted', 'speechEnded', 'transcript']);
  assertNoWait(a.logs, 100);
  // nor in pure silence
  const b = await setup();
  const sb = new ArrayAudioSource(new Float32Array(0), { tail: () => 0, frameDelayMs: 1, onFrame: (n) => (n === 40 ? stopAndWait(b.speech) : undefined) });
  assert.deepEqual(await collect(b.speech.listen({ _source: sb, signal: guard() })), []);
  assertNoWait(b.logs, 40);
});

test('a pause under way counts, so stop() waits only for the rest of it', async () => {
  // speech up to frame 46, silence from frame 47
  const audio = Signals.concat(silence(SPEECH_FROM), toneSamples((STOP_FRAME - SPEECH_FROM) * FRAME_SAMPLES), silence(100));
  // stop() 96 ms into the pause waits for the other 160 ms
  const a = await setup();
  const sa = new ArrayAudioSource(audio, { frameDelayMs: 1, onFrame: (n) => (n === STOP_FRAME + 3 ? stopAndWait(a.speech) : undefined) });
  const ea = await collect(a.speech.listen({ _source: sa, signal: guard() }));
  assert.equal(ea.find((e) => e.type === 'transcript').endMs, STOP_MS);
  assert.deepEqual(a.fake.recognizer.whisperInputSamples, [(STOP_FRAME + 8 - PRE_ROLL_FROM) * FRAME_SAMPLES]);
  // stop() 320 ms into the pause finishes at once
  const b = await setup();
  const sb = new ArrayAudioSource(audio, { frameDelayMs: 1, onFrame: (n) => (n === STOP_FRAME + 10 ? stopAndWait(b.speech) : undefined) });
  const eb = await collect(b.speech.listen({ _source: sb, signal: guard() }));
  assert.equal(eb.filter((e) => e.type === 'transcript').length, 1);
  assertNoWait(b.logs, STOP_FRAME + 10);
});

test('leaving the loop while stop() waits for a pause still discards', async () => {
  const { speech, fake } = await setup();
  const ac = new AbortController();
  const source = new ArrayAudioSource(silence(SPEECH_FROM), {
    tail: endlessTone,
    frameDelayMs: 5,
    onFrame: (n) => {
      if (n === STOP_FRAME) speech.stop();
      if (n === STOP_FRAME + 5) ac.abort();
    },
  });
  await assert.rejects(collect(speech.listen({ _source: source, signal: ac.signal })), (e) => e.name === 'AbortError');
  assert.ok(source.closed);
  assert.ok(source.framesSent < STOP_FRAME + 31, `aborted within the wait (${source.framesSent} frames)`);
  assert.equal(fake.recognizer ? fake.recognizer.whisperInputs : 0, 0, 'nothing recognised');
  assert.equal(speech.state.listening, false);
  assert.equal(speech.state.modelLoaded, false);
});

test('create() validates its input and the bundled VAD model', async () => {
  await assert.rejects(CzechSpeech.create({}), TypeError);
  await assert.rejects(CzechSpeech.create({ model: {} }), TypeError);
  await assert.rejects(CzechSpeech.create({ model: CzechSpeechModel._fromPath('x', 1), config: { threshold: 2 } }), RangeError);
  await assert.rejects(CzechSpeech.create({ model: CzechSpeechModel._fromPath('x', 1), _vadModel: new Uint8Array(10),
    _createWorker: nodeWorkerFactory(fakeEntry) }),
    (e) => /SHA-256/.test(e.message));
  assert.throws(() => new CzechSpeech({}), TypeError);
});
