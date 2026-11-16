// Backend selection (WebGPU or CPU) and the WebGPU fallback: the pure helpers of src/backend.js with fake GPU objects,
// and the whole pipeline with the fake engine of test/support/fake-worker-entry.mjs, whose simulated WebGPU engine is
// asynchronous like the real one (JSPI) and can fail to start, fail or hang in the middle of a recognition.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { CzechSpeech, CzechSpeechModel, CzechSpeechError } from '../../src/index.js';
import { adapterLabel, backendPlan, jspiSupported, probeWebGPU } from '../../src/backend.js';
import { resolveConfig } from '../../src/config.js';
import { ArrayAudioSource, Signals, TestEnv, nodeWorkerFactory, within } from '../support/env.mjs';

const fakeEntry = new URL('../support/fake-worker-entry.mjs', import.meta.url);
const vadModel = fs.readFileSync(TestEnv.vadAsset);
const utterance = Signals.concat(Signals.zeros(0.5), Signals.tone(1.0), Signals.zeros(1.0));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---- src/backend.js ----

const jspi = { Suspending: function Suspending() {}, promising: () => {} };
const adapter = (features, info = {}) => ({ features: new Set(features), info: { vendor: 'intel', architecture: 'xe-lpg', ...info } });
const gpuWith = (a) => ({ requestAdapter: async () => (typeof a === 'function' ? a() : a) });

test('probeWebGPU: every reason not to use WebGPU, and the adapter when it can', async () => {
  assert.equal(jspiSupported({}), false);
  assert.equal(jspiSupported(jspi), true);
  const why = async (gpu, wasm = jspi) => (await probeWebGPU({ gpu, wasm })).reason;
  assert.match(await why(gpuWith(adapter(['shader-f16'])), {}), /JSPI/);
  assert.match(await why(undefined), /navigator\.gpu is not available/);
  assert.match(await why(gpuWith(() => { throw new Error('GPU process crashed'); })), /requestAdapter\(\) failed: GPU process crashed/);
  assert.match(await why(gpuWith(null)), /gave no adapter/);
  assert.match(await why(gpuWith(adapter(['shader-f16'], { isFallbackAdapter: true, vendor: 'google', architecture: 'swiftshader' }))),
    /software fallback \(google swiftshader\)/);
  assert.match(await why(gpuWith(adapter(['subgroups']))), /intel xe-lpg\) has no shader-f16/);
  const ok = await probeWebGPU({ gpu: gpuWith(adapter(['shader-f16', 'subgroups'])), wasm: jspi });
  assert.deepEqual(ok, { ok: true, adapter: { vendor: 'intel', architecture: 'xe-lpg', device: '', description: '' }, subgroups: true });
  assert.equal(adapterLabel(ok.adapter), 'intel xe-lpg');
  assert.equal(adapterLabel(null), 'unknown adapter');
});

test('backendPlan: cpu never tries the GPU, webgpu insists, auto falls back and remembers a failure', () => {
  assert.deepEqual(backendPlan('cpu'), { tryGpu: false, onUnavailable: 'cpu', reason: null });
  assert.deepEqual(backendPlan('auto'), { tryGpu: true, onUnavailable: 'cpu', reason: null });
  assert.deepEqual(backendPlan('webgpu'), { tryGpu: true, onUnavailable: 'fail', reason: null });
  assert.deepEqual(backendPlan('auto', { blocked: 'WebGPU failed: lost' }), { tryGpu: false, onUnavailable: 'cpu', reason: 'WebGPU failed: lost' });
  assert.equal(backendPlan('webgpu', { blocked: 'WebGPU failed: lost' }).tryGpu, true, 'webgpu tries again');
  assert.match(backendPlan('auto', { gpuModule: false }).reason, /not part of the assets/);
  // after a WebGPU failure the instance waits before it tries WebGPU again (src/backend.js GpuRecovery, owned by the page)
  assert.deepEqual(backendPlan('auto', { retryInMs: 5000, failure: 'WebGPU failed: lost' }),
    { tryGpu: false, onUnavailable: 'cpu', reason: 'WebGPU failed: lost (WebGPU is tried again in 5 s)' });
  assert.equal(backendPlan('auto', { retryInMs: 0, failure: 'WebGPU failed: lost' }).tryGpu, true, 'due: tried');
  assert.equal(backendPlan('webgpu', { retryInMs: 5000 }).tryGpu, true, 'webgpu does not wait');
});

test('config.backend: auto by default, validated', () => {
  assert.equal(resolveConfig().backend, 'auto');
  for (const b of ['auto', 'webgpu', 'cpu']) assert.equal(resolveConfig({ backend: b }).backend, b);
  for (const b of ['gpu', 'wasm', '', null, 1]) assert.throws(() => resolveConfig({ backend: b }), RangeError);
});

// ---- the pipeline with the fake engine ----

async function setup({ engine, backend, keep, noSab, gpuTimeouts, now } = {}) {
  const fake = { capture: null, recognizer: null, events: [] };
  const logs = [];
  const factory = nodeWorkerFactory(fakeEntry);
  const speech = await CzechSpeech.create({
    model: CzechSpeechModel._fromPath('fake-model.bin', 1000),
    config: backend ? { backend } : undefined,
    keepModelLoaded: keep,
    log: (level, message) => logs.push(`${level}: ${message}`),
    _createWorker: factory,
    _vadModel: vadModel,
    _engineOptions: engine,
    _noSharedMemory: noSab,
    _gpuTimeouts: gpuTimeouts,
    _now: now,
    _onWorkerMessage: (role, m) => {
      if (m.type === 'fake') {
        fake[role] = m;
        fake.events.push(`${role}:${m.backend}:${m.event}`);
      }
    },
  });
  return { speech, factory, fake, logs };
}

async function collect(iterable) {
  const events = [];
  for await (const ev of iterable) events.push(ev);
  return events;
}

const types = (events) => events.map((e) => e.type);
const results = (events) => types(events).filter((t) => t === 'transcript' || t === 'error');
const allTerminated = (factory) => factory.created.every((w) => w.terminated);
const notices = (logs) => logs.filter((l) => /WebGPU not used/.test(l));
const noText = (logs) => assert.ok(!logs.some((l) => /dobrý|den|grafice/.test(l)), logs.join('\n'));

test('auto without WebGPU: the CPU, silently, with one notice and the reason in state', async () => {
  const { speech, fake, logs } = await setup();
  assert.equal(speech.state.backend, null, 'not decided before the recogniser starts');
  assert.equal(await speech.transcribe(Signals.tone(1.0)), 'dobrý den');
  assert.equal(speech.state.backend, 'cpu');
  assert.match(speech.state.backendFallback, /navigator\.gpu is not available/);
  assert.equal(speech.state.gpuAdapter, null);
  assert.ok(fake.events.every((e) => !e.includes(':webgpu:')), 'no GPU engine');
  const events = await collect(speech.listen({ _source: new ArrayAudioSource(utterance) }));
  assert.deepEqual(types(events), ['speechStarted', 'speechEnded', 'transcript']);
  assert.equal(notices(logs).length, 2, 'the decision is logged per recogniser start');
  assert.ok(!logs.some((l) => l.startsWith('warn')), logs.join('\n'));
});

test('the default logger prints the fallback once with console.info, and no transcript text', async () => {
  const lines = { info: [], debug: [], warn: [] };
  const saved = { info: console.info, debug: console.debug, warn: console.warn };
  for (const k of Object.keys(lines)) console[k] = (...a) => lines[k].push(a.join(' '));
  try {
    const speech = await CzechSpeech.create({ model: CzechSpeechModel._fromPath('fake-model.bin', 1000),
      _createWorker: nodeWorkerFactory(fakeEntry), _vadModel: vadModel });
    await speech.transcribe(Signals.tone(1.0));
    await speech.transcribe(Signals.tone(1.0));
    await collect(speech.listen({ _source: new ArrayAudioSource(utterance) }));
  } finally {
    Object.assign(console, saved);
  }
  assert.equal(lines.info.length, 1, lines.info.join('\n'));
  assert.match(lines.info[0], /^CzechSpeech: WebGPU not used \(navigator\.gpu is not available in the worker\); recognising on the CPU/);
  assert.equal(lines.warn.length, 0);
  assert.ok(![...lines.info, ...lines.debug].some((l) => /dobrý|den/.test(l)));
});

test('backend cpu never consults WebGPU', async () => {
  const { speech, fake, logs } = await setup({ backend: 'cpu', engine: { gpu: 'ok' } });
  assert.equal(await speech.transcribe(Signals.tone(1.0)), 'dobrý den');
  assert.equal(speech.state.backend, 'cpu');
  assert.equal(speech.state.backendFallback, null);
  assert.equal(fake.recognizer.counters.gpuStarts, 0);
  assert.equal(notices(logs).length, 0);
});

test('auto with WebGPU: the asynchronous GPU engine recognises, is wiped and freed like the CPU one', async () => {
  const { speech, fake, factory, logs } = await setup({ engine: { gpu: 'ok', gpuText: 'na grafice' } });
  const source = new ArrayAudioSource(utterance);
  const events = await collect(speech.listen({ _source: source }));
  assert.deepEqual(types(events), ['speechStarted', 'speechEnded', 'transcript']);
  assert.equal(events[2].text, 'na grafice');
  assert.equal(speech.state.backend, 'webgpu');
  assert.deepEqual(speech.state.gpuAdapter, { vendor: 'fakevendor', architecture: 'fakearch', device: '', description: '' });
  assert.equal(speech.state.backendFallback, null);
  assert.ok(speech.state.peakGpuBytes > 0, 'GPU memory reported');
  assert.equal(speech.state.modelLoaded, false);
  assert.ok(source.closed);
  await new Promise((r) => setTimeout(r, 50));
  assert.ok(allTerminated(factory));
  assert.ok(fake.events.includes('recognizer:webgpu:gpuWipe'), 'GPU engine wiped at the end');
  assert.equal(fake.recognizer.counters.whisperFrees, 1);
  assert.ok(fake.recognizer.whisperInputsZeroed, 'utterance zeroed after the GPU recognition');
  assert.ok(fake.capture.framesZeroed);
  assert.ok(fake.events.includes('capture:cpu:wipe'), 'speech detection stays on the CPU module');
  assert.ok(logs.some((l) => /recogniser on WebGPU \(fakevendor fakearch\)/.test(l)));
  assert.equal(notices(logs).length, 0);
});

test('on WebGPU listen() loads the model at session start; on the CPU only with the first utterance', async () => {
  const gpu = await setup({ engine: { gpu: 'ok' } });
  const silence = new ArrayAudioSource(Signals.zeros(1), { tail: () => 0, frameDelayMs: 1 });
  const ac = new AbortController();
  const run = collect(gpu.speech.listen({ _source: silence, signal: ac.signal }));
  const t0 = Date.now();
  while (!gpu.speech.state.modelLoaded && Date.now() - t0 < within(5000)) await sleep(5);
  assert.equal(gpu.speech.state.modelLoaded, true, 'loaded before any speech');
  ac.abort();
  await assert.rejects(run, (e) => e.name === 'AbortError');
  assert.equal(gpu.speech.state.modelLoads, 1);
  assert.equal(gpu.speech.state.modelLoaded, false, 'freed at the end');
  const cpu = await setup({ engine: { gpu: 'ok' }, backend: 'cpu' });
  await collect(cpu.speech.listen({ _source: new ArrayAudioSource(Signals.zeros(3)) }));
  assert.equal(cpu.speech.state.modelLoads, 0, 'CPU: a session without speech never loads the model');
  // the model failing on the GPU at session start: 'auto' goes on with the CPU, which loads with the first utterance
  const failing = await setup({ engine: { gpu: 'ok', gpuFailLoad: true } });
  const events = await collect(failing.speech.listen({ _source: new ArrayAudioSource(utterance) }));
  assert.deepEqual(types(events), ['speechStarted', 'speechEnded', 'transcript']);
  assert.equal(failing.speech.state.backend, 'cpu');
  assert.match(failing.speech.state.backendFallback, /out of GPU memory while loading/);
  assert.equal(failing.speech.state.modelLoads, 1, 'loaded once, on the CPU');
});

test('WebGPU uses whisper.cpp attention without flash attention (subgroups or not); an explicit option wins', async () => {
  const without = await setup({ engine: { gpu: 'ok-no-subgroups' } });
  assert.equal(await without.speech.transcribe(Signals.tone(1.0)), 'dobrý den');
  assert.equal(without.speech.state.backend, 'webgpu');
  assert.equal(without.fake.recognizer.flashAttn, 0);
  const withSg = await setup({ engine: { gpu: 'ok' } });
  await withSg.speech.transcribe(Signals.tone(1.0));
  assert.equal(withSg.fake.recognizer.flashAttn, 0);
  const forced = await setup({ engine: { gpu: 'ok', flashAttn: 1 } });
  await forced.speech.transcribe(Signals.tone(1.0));
  assert.equal(forced.fake.recognizer.flashAttn, 1);
});

test('the heap is reported after each recognition, so its peak includes the compute buffers', async () => {
  // regression (e91886f): the heap was reported at start-up, model load and free only; with keepModelLoaded the
  // compute buffers whisper.cpp allocates during the first recognition never showed in state
  const base = 1 << 20;
  const growth = 64 << 20;
  for (const gpu of [undefined, 'ok']) {
    const { speech } = await setup({ engine: { gpu, heapGrowth: growth }, keep: true });
    await speech.transcribe(Signals.tone(1.0));
    assert.equal(speech.state.backend, gpu ? 'webgpu' : 'cpu');
    assert.equal(speech.state.heapBytes, base + growth, `${speech.state.backend}: after transcribe()`);
    assert.equal(speech.state.peakHeapBytes, base + growth);
    await speech.close();
    // and after an utterance of listen()
    const s = await setup({ engine: { gpu, heapGrowth: growth }, keep: true });
    await collect(s.speech.listen({ _source: new ArrayAudioSource(utterance) }));
    assert.equal(s.speech.state.heapBytes, base + growth, `${s.speech.state.backend}: after listen()`);
    await s.speech.close();
  }
});

test('backend webgpu without WebGPU fails with code unsupported, in listen() and transcribe()', async () => {
  const { speech } = await setup({ backend: 'webgpu', engine: { gpu: 'no-f16' } });
  await assert.rejects(speech.transcribe(Signals.tone(1.0)),
    (e) => e instanceof CzechSpeechError && e.code === 'unsupported' && /no shader-f16/.test(e.message));
  const source = new ArrayAudioSource(utterance, { tail: () => 0, frameDelayMs: 1 });
  await assert.rejects(collect(speech.listen({ _source: source })), (e) => e.code === 'unsupported');
  assert.ok(source.closed, 'microphone closed');
  assert.equal(speech.state.listening, false);
});

test('auto: a WebGPU module that does not start falls back, and later sessions do not try it again', async () => {
  const { speech, fake, logs } = await setup({ engine: { gpu: 'ok', gpuStartFails: true } });
  assert.equal(await speech.transcribe(Signals.tone(1.0)), 'dobrý den');
  assert.equal(speech.state.backend, 'cpu');
  assert.match(speech.state.backendFallback, /the WebGPU module did not start: fake WebGPU backend did not start/);
  assert.equal(fake.recognizer.counters.gpuStarts, 1);
  assert.equal(await speech.transcribe(Signals.tone(1.0)), 'dobrý den');
  assert.equal(fake.recognizer.counters.gpuStarts, 0, 'second recogniser: WebGPU not tried');
  assert.equal(notices(logs).length, 2);
});

test('auto: WebGPU failing in the middle of an utterance gives the transcript from the CPU, then WebGPU again', async () => {
  // before: 'auto' stayed on the CPU for the rest of the instance after any WebGPU failure (the 3-hour soak of the
  // meeting app fell to the CPU for good after one runaway decode)
  const { speech, fake, logs } = await setup({ engine: { gpu: 'ok', gpuFailAt: 1, gpuText: 'na grafice' },
    gpuTimeouts: { gpuRetryMs: [0, 0, 0] } });
  const events = await collect(speech.listen({ _source: new ArrayAudioSource(Signals.concat(utterance, utterance, utterance)) }));
  // results come in utterance order; with this fast source later utterances' speech events may come first
  assert.deepEqual(results(events), ['transcript', 'transcript', 'transcript']);
  assert.deepEqual(events.filter((e) => e.type === 'transcript').map((e) => e.text), ['dobrý den', 'na grafice', 'na grafice'],
    'the failed utterance redone on the CPU, the next ones on the WebGPU module started again');
  assert.equal(speech.state.backend, 'webgpu');
  assert.equal(speech.state.backendFallback, null);
  assert.ok(logs.some((l) => /^warn: WebGPU failed \(ggml aborted: fake device error\); recognising on the CPU, WebGPU is tried again in 0 s$/.test(l)),
    logs.join('\n'));
  assert.ok(logs.some((l) => /^info: WebGPU started again after a failure$/.test(l)), logs.join('\n'));
  assert.ok(fake.events.includes('recognizer:webgpu:gpuWipe'), 'the failed GPU engine was released');
  assert.ok(fake.events.includes('recognizer:cpu:wipe'), 'and the CPU engine once WebGPU was back');
  assert.equal(fake.recognizer.counters.gpuStarts, 2, 'the WebGPU module was started a second time');
  assert.ok(fake.recognizer.whisperInputsZeroed);
  assert.equal(speech.state.modelLoaded, false);
});

test('auto: WebGPU is started again only after the retry delay; until then the CPU recognises', async (t) => {
  const { speech, fake } = await setup({ engine: { gpu: 'ok', gpuFailAt: 1, gpuText: 'na grafice' }, keep: true,
    gpuTimeouts: { gpuRetryMs: [within(1500), 0, 0] } });
  t.after(() => speech.close());
  assert.equal(await speech.transcribe(Signals.tone(1.0)), 'dobrý den', 'redone on the CPU');
  assert.equal(await speech.transcribe(Signals.tone(1.0)), 'dobrý den', 'still on the CPU: the delay has not passed');
  assert.equal(speech.state.backend, 'cpu');
  assert.match(speech.state.backendFallback, /WebGPU failed: ggml aborted: fake device error/);
  assert.equal(fake.recognizer.counters.gpuStarts, 1);
  await sleep(within(1500));
  assert.equal(await speech.transcribe(Signals.tone(1.0)), 'na grafice');
  assert.equal(speech.state.backend, 'webgpu');
  assert.equal(fake.recognizer.counters.gpuStarts, 2);
});

test('auto: a WebGPU module that fails again each time it is started: the CPU after the third retry, also for later sessions', async () => {
  // the first engine breaks in its first recognition; every later start of the WebGPU module fails
  const { speech, fake, logs } = await setup({ engine: { gpu: 'ok', gpuFailAt: 1, gpuStartFailsAfter: 1 },
    gpuTimeouts: { gpuRetryMs: [0, 0, 0] } });
  const audio = Signals.concat(...Array(6).fill(utterance));
  const events = await collect(speech.listen({ _source: new ArrayAudioSource(audio) }));
  assert.deepEqual(events.filter((e) => e.type === 'transcript').map((e) => e.text), Array(6).fill('dobrý den'));
  assert.equal(fake.recognizer.counters.gpuStarts, 4, 'the first start and three retries, then no more');
  assert.equal(speech.state.backend, 'cpu');
  assert.ok(logs.some((l) => /^warn: WebGPU could not be started again \(.*did not start\); recognising on the CPU from now on$/.test(l)),
    logs.join('\n'));
  // the next recogniser of this instance does not try WebGPU any more
  assert.equal(await speech.transcribe(Signals.tone(1.0)), 'dobrý den');
  assert.equal(fake.recognizer.counters.gpuStarts, 0);
  assert.match(speech.state.backendFallback, /did not start/);
});

// ---- the recovery state belongs to the instance (keepModelLoaded: false gives every call a new recogniser) ----

const gpuStarts = (fake) => fake.events.filter((e) => e === 'recognizer:webgpu:gpuStart').length;

test('auto, a new recogniser per call: WebGPU is tried again after 10 s, 60 s and 300 s, then never; in between the CPU at once', async () => {
  // before: each new recogniser started with a fresh count, so a GPU that fails every time was tried on every call
  // (WebGPU start, GPU model load, failure, CPU start, CPU model load)
  let now = 0;
  const { speech, fake, logs } = await setup({ engine: { gpu: 'ok', gpuFailAt: 1 }, now: () => now });
  const call = async (atMs) => {
    now = atMs;
    const before = gpuStarts(fake);
    assert.equal(await speech.transcribe(Signals.tone(1.0)), 'dobrý den', `at ${atMs} ms`);
    return gpuStarts(fake) - before;
  };
  assert.equal(await call(0), 1, 'the first call tries WebGPU (and fails)');
  assert.equal(speech.state.backend, 'cpu');
  // (the instance's clock is fake, but a recogniser counts the time left on its own clock: the calls before a retry time
  // leave seconds to spare; the exact boundaries are tested with fake time in test/unit/watchdog.test.mjs)
  assert.equal(await call(5000), 0, 'before the first delay: the CPU, WebGPU not tried');
  assert.match(speech.state.backendFallback, /WebGPU failed: ggml aborted: fake device error \(WebGPU is tried again in 5 s\)/);
  assert.equal(await call(10000), 1, 'after 10 s: tried again (and fails)');
  assert.equal(await call(40000), 0);
  assert.equal(await call(70000), 1, 'then 60 s later');
  assert.equal(await call(200000), 0);
  assert.equal(await call(370000), 1, 'then 300 s later: the fourth failure in a row');
  assert.equal(await call(10000000), 0, 'given up: never again in this instance');
  assert.equal(gpuStarts(fake), 4);
  assert.ok(logs.some((l) => /recognising on the CPU from now on/.test(l)), logs.join('\n'));
});

test('auto, a new recogniser per call: a recognition that succeeds on WebGPU resets the count', async () => {
  let now = 0;
  // the GPU recognitions of all recognisers of this instance, in order: fail, fail, succeed, fail
  const seq = new Int32Array(new SharedArrayBuffer(4));
  const { speech, fake } = await setup({ engine: { gpu: 'ok', gpuText: 'na grafice', gpuFailSequence: [true, true, false, true], gpuSeq: seq },
    now: () => now });
  const call = async (atMs) => {
    now = atMs;
    const before = gpuStarts(fake);
    const text = await speech.transcribe(Signals.tone(1.0));
    return { text, tried: gpuStarts(fake) - before };
  };
  assert.deepEqual(await call(0), { text: 'dobrý den', tried: 1 });
  assert.deepEqual(await call(10000), { text: 'dobrý den', tried: 1 });
  assert.deepEqual(await call(70000), { text: 'na grafice', tried: 1 }, 'the third try succeeds');
  assert.deepEqual(await call(70001), { text: 'dobrý den', tried: 1 }, 'no wait after a success; this one fails');
  assert.deepEqual(await call(75000), { text: 'dobrý den', tried: 0 }, 'the first delay again (10 s), not the third');
  assert.deepEqual(await call(80001), { text: 'na grafice', tried: 1 });
});

test('auto: repeated failures in the middle of recognitions give up after the fourth in a row, also for later recognisers', async () => {
  const seq = new Int32Array(new SharedArrayBuffer(4));
  const { speech, fake, logs } = await setup({ engine: { gpu: 'ok', gpuFailSequence: Array(10).fill(true), gpuSeq: seq },
    gpuTimeouts: { gpuRetryMs: [0, 0, 0] } });
  const audio = Signals.concat(...Array(6).fill(utterance));
  const events = await collect(speech.listen({ _source: new ArrayAudioSource(audio) }));
  assert.deepEqual(events.filter((e) => e.type === 'transcript').map((e) => e.text), Array(6).fill('dobrý den'));
  assert.equal(gpuStarts(fake), 4, 'the first start and three restarts');
  assert.ok(logs.some((l) => /^warn: WebGPU failed \(ggml aborted: fake device error\); recognising on the CPU from now on$/.test(l)), logs.join('\n'));
  assert.equal(await speech.transcribe(Signals.tone(1.0)), 'dobrý den');
  assert.equal(gpuStarts(fake), 4, 'the next recogniser does not try WebGPU');
});

test('auto: the simulated device loss (test hook) hits as many WebGPU engines as asked, then WebGPU recognises', async (t) => {
  for (const [count, texts] of [[undefined, ['dobrý den', 'na grafice', 'na grafice']], [2, ['dobrý den', 'dobrý den', 'na grafice']]]) {
    const { speech } = await setup({ engine: { gpu: 'ok', gpuText: 'na grafice', testFailRecognitionMs: 5, testFailRecognitionCount: count },
      keep: true, gpuTimeouts: { gpuRetryMs: [0, 0, 0] } });
    t.after(() => speech.close());
    const got = [];
    for (let i = 0; i < 3; i++) got.push(await speech.transcribe(Signals.tone(1.0)));
    assert.deepEqual(got, texts, `testFailRecognitionCount ${count}`);
  }
});

test('auto: a WebGPU module started again that cannot be used is released, and the CPU goes on', async (t) => {
  // the restored engine fails while it is attached (its speech gate does not load): before, it was left behind
  const { speech, fake, logs } = await setup({ engine: { gpu: 'ok', gpuFailAt: 1, gpuAttachFailsAfter: 1 }, keep: true,
    gpuTimeouts: { gpuRetryMs: [0, 60000, 60000] } }); // one retry, at once
  t.after(() => speech.close());
  assert.equal(await speech.transcribe(Signals.tone(1.0)), 'dobrý den');
  const wipesBefore = fake.events.filter((e) => e === 'recognizer:webgpu:gpuWipe').length;
  assert.equal(await speech.transcribe(Signals.tone(1.0)), 'dobrý den', 'still recognised, on the CPU');
  assert.equal(speech.state.backend, 'cpu');
  assert.equal(fake.events.filter((e) => e === 'recognizer:webgpu:gpuWipe').length, wipesBefore + 1, 'the new GPU engine was released');
  assert.ok(logs.some((l) => /^warn: WebGPU could not be started again \(.*fake WebGPU speech gate failed.*\)/.test(l)), logs.join('\n'));
});

test('auto: when the CPU module does not start after a WebGPU failure, the call fails clearly and the next starts it', async (t) => {
  const wipes = (fake) => fake.events.filter((e) => e === 'recognizer:webgpu:gpuWipe').length;
  {
    // before its retry time: the next call starts the CPU module again, and WebGPU is not tried outside the backoff
    const { speech, fake, logs } = await setup({ engine: { gpu: 'ok', gpuFailAt: 1, cpuStartFails: 1 }, keep: true });
    t.after(() => speech.close());
    await assert.rejects(speech.transcribe(Signals.tone(1.0)), (e) => e.code === 'runtime' &&
      /^WebGPU failed \(ggml aborted: fake device error\) and the CPU module did not start: fake CPU module did not start$/.test(e.message));
    assert.equal(speech.state.backend, 'cpu');
    assert.match(speech.state.backendFallback, /WebGPU failed: ggml aborted/);
    assert.equal(await speech.transcribe(Signals.tone(1.0)), 'dobrý den', 'the next call starts the CPU module again');
    assert.equal(speech.state.backend, 'cpu');
    assert.equal(gpuStarts(fake), 1, 'WebGPU is not started again before its retry time');
    assert.equal(wipes(fake), 1);
    assert.ok(!logs.some((l) => /releasing the CPU module/.test(l)), logs.join('\n'));
  }
  {
    // the retry is due at once: WebGPU is started again with nothing to release on the CPU
    const { speech, fake, logs } = await setup({ engine: { gpu: 'ok', gpuFailAt: 1, cpuStartFails: 1, gpuText: 'na grafice' }, keep: true,
      gpuTimeouts: { gpuRetryMs: [0, 60000, 60000] } });
    t.after(() => speech.close());
    await assert.rejects(speech.transcribe(Signals.tone(1.0)), (e) => e.code === 'runtime' && /the CPU module did not start/.test(e.message));
    assert.equal(await speech.transcribe(Signals.tone(1.0)), 'na grafice');
    assert.equal(speech.state.backend, 'webgpu');
    assert.equal(gpuStarts(fake), 2);
    assert.ok(!logs.some((l) => /releasing the CPU module/.test(l)), logs.join('\n'));
  }
});

test('webgpu: a module restarted after a failure whose speech gate does not load is released, never used', async (t) => {
  const { speech, fake } = await setup({ backend: 'webgpu', engine: { gpu: 'ok', gpuFailAt: 1, gpuAttachFailsAfter: 1 }, keep: true });
  t.after(() => speech.close());
  const wipes = () => fake.events.filter((e) => e === 'recognizer:webgpu:gpuWipe').length;
  await assert.rejects(speech.transcribe(Signals.tone(1.0)), /WebGPU failed: ggml aborted/);
  assert.equal(wipes(), 1, 'the failed engine');
  for (let call = 2; call <= 3; call++) {
    await assert.rejects(speech.transcribe(Signals.tone(1.0)), (e) => e.code === 'unsupported' &&
      /the WebGPU module could not be used: fake WebGPU speech gate failed/.test(e.message));
    assert.equal(wipes(), call, `call ${call}: the restarted engine was released`);
    assert.equal(gpuStarts(fake), call, `call ${call}: started again, not left attached without a speech gate`);
  }
});

test('auto: a stop that races the end of the recognition keeps its transcript', async (t) => {
  const { speech } = await setup({ engine: { gpu: 'ok', gpuSlowAt: 1, gpuSlowMs: 20000, gpuSlowCompletes: true, gpuText: 'na grafice' },
    keep: true, gpuTimeouts: { recognizeBase: 300, recognizePerSecond: 0 } });
  t.after(() => speech.close());
  assert.equal(await speech.transcribe(Signals.tone(1.0)), 'na grafice', 'the call ended normally while it was being stopped');
  assert.equal(speech.state.abortedRecognitions, 0);
  assert.equal(speech.state.backend, 'webgpu');
});

test('close() during the stop grace and during a restart of WebGPU resolves at once and terminates every worker', async (t) => {
  {
    const { speech, factory } = await setup({ engine: { gpu: 'ok', gpuHangAt: 1 },
      gpuTimeouts: { recognizeBase: 200, recognizePerSecond: 0, recognizeStopGrace: 60000 } });
    t.after(() => speech.close());
    const p = speech.transcribe(Signals.tone(1.0));
    p.catch(() => {});
    await sleep(500); // past the limit: in the grace
    const t0 = Date.now();
    await speech.close();
    assert.ok(Date.now() - t0 < within(3000), `close() took ${Date.now() - t0} ms`);
    await assert.rejects(p, (e) => e.code === 'closed');
    assert.ok(allTerminated(factory));
  }
  {
    const { speech, factory, fake } = await setup({ engine: { gpu: 'ok', gpuFailAt: 1, gpuRestartDelayMs: 60000 }, keep: true,
      gpuTimeouts: { gpuRetryMs: [0, 0, 0] } });
    t.after(() => speech.close());
    assert.equal(await speech.transcribe(Signals.tone(1.0)), 'dobrý den');
    const p = speech.transcribe(Signals.tone(1.0));
    p.catch(() => {});
    while (gpuStarts(fake) < 2) await sleep(5); // the restart is under way
    const t0 = Date.now();
    await speech.close();
    assert.ok(Date.now() - t0 < within(3000), `close() took ${Date.now() - t0} ms`);
    await assert.rejects(p, (e) => e.code === 'closed');
    assert.ok(allTerminated(factory));
  }
});

test('a decode stopped by a token limit is logged with the limit that stopped it and counted, without text', async (t) => {
  {
    // the cap for 1 s of audio: max_tokens 32, so a decode stops after 33 text tokens
    const { speech, logs } = await setup({ engine: { gpu: 'ok', capped: true }, keep: true });
    t.after(() => speech.close());
    await speech.transcribe(Signals.tone(1.0));
    await speech.transcribe(Signals.tone(1.0));
    assert.equal(speech.state.cappedRecognitions, 2);
    assert.ok(logs.some((l) => /^info: transcribe: token limit reached \(33 tokens; the cap allows 33\)$/.test(l)), logs.join('\n'));
    noText(logs);
  }
  {
    // a cap above whisper.cpp's own limit (as from 8.8 s of audio up): whisper.cpp stopped it at 220, not the cap
    const { speech, logs } = await setup({ engine: { gpu: 'ok', capped: true, cappedTokens: 220, maxTokens: 250 }, keep: true });
    t.after(() => speech.close());
    await speech.transcribe(Signals.tone(1.0));
    assert.equal(speech.state.cappedRecognitions, 1);
    assert.ok(logs.some((l) => /^info: transcribe: token limit reached \(220 tokens; whisper\.cpp allows 220\)$/.test(l)), logs.join('\n'));
    noText(logs);
  }
});

test('auto: a WebGPU recognition over its time limit is stopped and the GPU kept: no transcript for it, the next on WebGPU', async (t) => {
  // before: the watchdog marked the engine broken and destroyed the device, and 'auto' went to the CPU for good
  const { speech, fake, logs } = await setup({ engine: { gpu: 'ok', gpuSlowAt: 1, gpuSlowMs: 20000, gpuText: 'na grafice' }, keep: true,
    gpuTimeouts: { recognizeBase: 300, recognizePerSecond: 0, recognizeStopGrace: 5000 } });
  t.after(() => speech.close());
  const t0 = Date.now();
  assert.equal(await speech.transcribe(Signals.tone(1.0)), '');
  const ms = Date.now() - t0;
  assert.ok(ms < within(2000), `stopped after ${ms} ms`);
  assert.equal(speech.state.backend, 'webgpu');
  assert.equal(speech.state.backendFallback, null);
  assert.equal(speech.state.abortedRecognitions, 1);
  assert.ok(logs.some((l) => /^warn: transcribe: 1000 ms of audio, the recogniser did not finish in 0 s and was stopped \(WebGPU stays in use\), no transcript$/.test(l)),
    logs.join('\n'));
  assert.ok(!fake.events.includes('recognizer:webgpu:gpuWipe'), 'the GPU engine was not released');
  assert.equal(await speech.transcribe(Signals.tone(1.0)), 'na grafice', 'the next recognition on the same WebGPU module');
  assert.equal(fake.recognizer.counters.gpuStarts, 1);
  assert.equal(fake.recognizer.counters.gpuTranscribes, 2);
  noText(logs);
});

test('auto: a recognition that does not stop within the grace is given up: device released, transcript from the CPU', async () => {
  const { speech, fake } = await setup({ engine: { gpu: 'ok', gpuHangAt: 1 },
    gpuTimeouts: { recognizeBase: 300, recognizePerSecond: 0, recognizeStopGrace: 200 } });
  const t0 = Date.now();
  assert.equal(await speech.transcribe(Signals.tone(1.0)), 'dobrý den');
  const ms = Date.now() - t0;
  assert.ok(ms >= 450 && ms < within(3000), `took ${ms} ms: the limit, then the grace`);
  assert.equal(speech.state.backend, 'cpu');
  assert.match(speech.state.backendFallback, /WebGPU failed: WebGPU: recognition did not finish in 0 s/);
  assert.ok(fake.events.includes('recognizer:webgpu:gpuWipe'), 'the failed GPU engine was released');
});

test('webgpu: a recognition over its time limit is stopped and the GPU kept; the session goes on', async () => {
  const { speech, fake } = await setup({ backend: 'webgpu', engine: { gpu: 'ok', gpuSlowAt: 1, gpuSlowMs: 20000, gpuText: 'na grafice' },
    gpuTimeouts: { recognizeBase: 300, recognizePerSecond: 0, recognizeStopGrace: 5000 } });
  const events = await collect(speech.listen({ _source: new ArrayAudioSource(Signals.concat(utterance, utterance)) }));
  assert.deepEqual(results(events), ['transcript'], 'the stopped utterance gives nothing, the next its transcript');
  assert.equal(events.find((e) => e.type === 'transcript').text, 'na grafice');
  assert.equal(fake.recognizer.counters.gpuStarts, 1, 'no restart');
  assert.equal(speech.state.abortedRecognitions, 1);
});

test('webgpu: a failed utterance is an error event, the GPU module restarts and the session goes on', async () => {
  const { speech, logs } = await setup({ backend: 'webgpu', engine: { gpu: 'ok', gpuFailAt: 1, gpuText: 'na grafice' } });
  const events = await collect(speech.listen({ _source: new ArrayAudioSource(Signals.concat(utterance, utterance)) }));
  const r = events.filter((e) => e.type === 'transcript' || e.type === 'error');
  assert.deepEqual(r.map((e) => e.type), ['error', 'transcript']);
  assert.match(r[0].error.message, /WebGPU failed: ggml aborted/);
  assert.equal(r[1].text, 'na grafice');
  assert.equal(speech.state.backend, 'webgpu');
  assert.ok(logs.some((l) => /WebGPU module restarted/.test(l)));
});

test('webgpu: when the GPU module cannot be restarted after a failure the session ends with unsupported', async () => {
  // gpuFailAt 1 breaks the first engine; the restart fails because every later start fails (gpuStartFails after 1)
  const { speech } = await setup({ backend: 'webgpu', engine: { gpu: 'ok', gpuFailAt: 1, gpuStartFailsAfter: 1 } });
  await assert.rejects(collect(speech.listen({ _source: new ArrayAudioSource(Signals.concat(utterance, utterance)) })),
    (e) => e.code === 'unsupported' && /did not start/.test(e.message));
  assert.equal(speech.state.listening, false);
});

test('the GPU engine gets one utterance at a time, in order, while the worker stays responsive', async () => {
  const { speech, fake } = await setup({ engine: { gpu: 'ok', gpuWorkMs: 150 } });
  const audio = Signals.concat(utterance, Signals.zeros(0.7), utterance, Signals.zeros(0.7), utterance);
  const events = await collect(speech.listen({ _source: new ArrayAudioSource(audio) })); // fast source: queued utterances
  const ts = events.filter((e) => e.type === 'transcript');
  assert.equal(ts.length, 3, types(events).join(','));
  assert.ok(ts[0].startMs < ts[1].startMs && ts[1].startMs < ts[2].startMs);
  assert.ok(!events.some((e) => e.type === 'error'), 'no re-entry');
  assert.equal(fake.recognizer.counters.gpuTranscribes, 3);
});

test('an aborted transcribe() on the GPU rejects at once and the recogniser is freed', async () => {
  const { speech, factory } = await setup({ engine: { gpu: 'ok', gpuWorkMs: 400 } });
  const ac = new AbortController();
  const p = speech.transcribe(Signals.tone(1.0), { signal: ac.signal });
  while (!speech.state.modelLoaded) await sleep(5);
  ac.abort();
  await assert.rejects(p, (e) => e.name === 'AbortError');
  const t0 = Date.now();
  while (!allTerminated(factory) && Date.now() - t0 < within(3000)) await sleep(10);
  assert.ok(allTerminated(factory));
  assert.equal(speech.state.modelLoaded, false);
  assert.equal(speech.state.abortedRecognitions, 1, 'stopped through the abort flag after the GPU step');
});

test('keepModelLoaded on the GPU: one model load until close(), which wipes the GPU engine', async () => {
  const { speech, fake, factory } = await setup({ engine: { gpu: 'ok' }, keep: true });
  await speech.transcribe(Signals.tone(1.0));
  await speech.transcribe(Signals.tone(1.0));
  assert.equal(speech.state.modelLoads, 1);
  assert.equal(speech.state.modelLoaded, true);
  assert.ok(!fake.events.includes('recognizer:webgpu:gpuWipe'));
  await speech.close();
  assert.ok(fake.events.includes('recognizer:webgpu:gpuWipe'));
  assert.equal(speech.state.modelLoaded, false);
  assert.ok(allTerminated(factory));
});
