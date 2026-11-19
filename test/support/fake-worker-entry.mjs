// Worker entry for the unit tests: worker-core with a fake engine (no WebAssembly, no model). The engine reports
// its calls to the page as { type: 'fake', role, ... } messages, which the tests collect through _onWorkerMessage.
//
// "Speech" is any frame with a sample above 0.05 in magnitude (as in the Android SpeechEngineTest's FakeNative).
//
// WebGPU is simulated too (engine options, all optional):
//   gpu: 'ok' | 'ok-no-subgroups' | 'no-adapter' | 'no-f16'   what the WebGPU probe finds (default: no navigator.gpu)
//   gpuStartFails: true                   the WebGPU module does not start (like cs_backend_init failing)
//   gpuStartFailsAfter: n                 every WebGPU start after the n-th fails (a restart after a failure)
//   gpuFailAt: n                          the n-th GPU recognition (1-based, counted per worker) rejects like a ggml abort
//   gpuHangAt: n                          the n-th GPU recognition never settles, not even when asked to stop (the
//                                         watchdog must give it up)
//   gpuSlowAt: n, gpuSlowMs: ms           the n-th GPU recognition takes that long, but ends at once with status 1 (like
//                                         whisper.cpp's abort check) when the watchdog asks it to stop (engine.stop())
//   gpuSlowCompletes: true                ... except that, asked to stop, it ends normally (the stop raced its end)
//   gpuFailSequence: [bool], gpuSeq: Int32Array(SharedArrayBuffer)   GPU recognitions across all workers of a test, in
//                                         order: true rejects like a ggml abort (gpuSeq is the shared position)
//   gpuAttachFailsAfter: n                every WebGPU engine after the n-th (per worker) fails to load its speech gate
//   gpuRestartDelayMs: ms                 every WebGPU start after the first (per worker) takes this long
//   testFailRecognitionMs: ms             as in the real engine: the device is lost this many ms into the engine's first
//                                         recognition (src/worker-core.js Backends passes it to the first engines only)
//   emptyTextAt: n                        the n-th CPU recognition (1-based, per worker) returns no text
//   capped: true                          every recognition reports that it reached a token limit (33 tokens)
//   cappedTokens: n                       ... with that many tokens instead
//   cpuStartFails: n                      the first n starts of the CPU module after a WebGPU start (per worker) fail
//   gpuFailLoad: true                     loading the model onto the GPU rejects
//   gpuWorkMs: ms                         time a GPU recognition takes (setTimeout: the worker stays responsive)
//   gpuText: string                       what the GPU engine "recognises" (default: the CPU text)
// Start-up: startDelayMs: ms, every engine (capture and recogniser) takes this long to load.
// Every recognition reports the decoding limits it was given ({ maxTokens, temperatureFallback }) as `limits` and the
// languages it was given (comma-separated codes) as `languages`; the result's `language` is the first of them.
// Heap: heapGrowth: bytes, the heap grows by this much during the first recognition (whisper.cpp's compute buffers).
// Failures: failLoad: true, every engine fails to load; failIn: 'transcribe' (whisperTranscribe throws) or 'crash' (the
// worker dies of an uncaught error during a recognition). errorKind: 'dom' (a DOMException, numeric code 19) or 'node'
// (an Error with code 'ENOENT') is what is thrown; default an Error without a code.
import { installWorker } from '../../src/worker-core.js';

const sleepView = new Int32Array(new SharedArrayBuffer(4));
const sleep = (ms) => Atomics.wait(sleepView, 0, 0, ms);
const peak = (a) => a.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
const gpuCounters = { starts: 0, transcribes: 0, cpuStartFailures: 0 }; // per worker, across GPU engine restarts

function fakeError(o, message) {
  if (o.errorKind === 'dom') return new DOMException(message, 'NetworkError');
  if (o.errorKind === 'node') return Object.assign(new Error(`ENOENT: ${message}`), { code: 'ENOENT' });
  return new Error(message);
}

async function loadFakeEngine({ abortRequested, post, options, gpu }) {
  const o = options || {};
  if (o.failLoad) throw fakeError(o, 'fake engine failed to load');
  if (o.startDelayMs) await new Promise((r) => setTimeout(r, o.startDelayMs));
  if (gpu) {
    gpuCounters.starts++;
    post({ type: 'fake', event: 'gpuStart', backend: 'webgpu' });
    if (o.gpuRestartDelayMs && gpuCounters.starts > 1) await new Promise((r) => setTimeout(r, o.gpuRestartDelayMs));
    if (o.gpuStartFails || (o.gpuStartFailsAfter && gpuCounters.starts > o.gpuStartFailsAfter)) {
      throw new Error('fake WebGPU backend did not start');
    }
  }
  if (!gpu && gpuCounters.starts > 0 && gpuCounters.cpuStartFailures < (o.cpuStartFails || 0)) {
    gpuCounters.cpuStartFailures++;
    throw new Error('fake CPU module did not start');
  }
  let failNext = o.failNext || 0;
  const capInfo = o.capped ? { tokens: o.cappedTokens ?? 33, capped: true } : {};
  let handles = 100;
  const live = new Set();
  const counters = { vadInits: 0, vadFrees: 0, vadResets: 0, vadFrames: 0, gateCalls: 0, whisperInits: 0,
    whisperFrees: 0, transcribes: 0, wipes: 0, gpuStarts: gpuCounters.starts, gpuTranscribes: 0 };
  const seenFrames = [];
  const passedToWhisper = [];
  const report = (event, extra = {}) => post({
    ...extra, type: 'fake', event, backend: gpu ? 'webgpu' : 'cpu',
    counters: { ...counters, gpuStarts: gpuCounters.starts, gpuTranscribes: gpuCounters.transcribes },
    flashAttn: o.flashAttn,
    framesZeroed: seenFrames.every((f) => f.every((v) => v === 0)),
    whisperInputsZeroed: passedToWhisper.every((f) => f.every((v) => v === 0)),
    whisperInputs: passedToWhisper.length,
    whisperInputSamples: passedToWhisper.map((f) => f.length),
  });
  const engine = {
    heapBytes: 1 << 20,
    async: !!gpu,
    backend: gpu ? 'webgpu' : 'cpu',
    adapter: gpu ? { vendor: 'fakevendor', architecture: 'fakearch', device: '', description: '' } : undefined,
    gpuBytes: 0,
    failure: null,
    fail(err) {
      if (!engine.failure) engine.failure = err;
    },
    systemInfo: () => 'fake',
    vadInit() {
      counters.vadInits++;
      const h = ++handles;
      live.add(h);
      return h;
    },
    vadFree(h) {
      counters.vadFrees++;
      if (!live.delete(h)) throw new Error('double free (VAD)');
    },
    vadReset() {
      counters.vadResets++;
    },
    vadFrame(h, frame) {
      if (!live.has(h)) throw new Error('VAD use after free');
      counters.vadFrames++;
      seenFrames.push(frame);
      return peak(frame) > 0.05 ? 0.9 : 0.02;
    },
    vadSpeechSegments(h, samples) {
      if (!live.has(h)) throw new Error('VAD use after free');
      counters.gateCalls++;
      report('gate');
      return peak(samples) > 0.05 ? 1 : 0;
    },
    whisperInit(reader) {
      if (String(reader.path || '').includes('nonexistent')) throw new Error('Model file not found');
      counters.whisperInits++;
      const h = ++handles;
      live.add(h);
      report('whisperInit');
      return h;
    },
    whisperFree(h) {
      counters.whisperFrees++;
      if (!live.delete(h)) throw new Error('double free (whisper)');
    },
    whisperTranscribe(h, token, samples, threads, limits, languages) {
      if (!live.has(h)) throw new Error('use of a freed context');
      counters.transcribes++;
      passedToWhisper.push(samples);
      report('transcribeStarted', { limits, languages });
      if (o.failIn === 'transcribe') throw fakeError(o, 'fake recognition failed');
      if (o.failIn === 'crash') {
        setTimeout(() => { throw fakeError(o, 'fake worker crashed'); });
        return new Promise(() => {});
      }
      if (o.blockUntilAborted) {
        const deadline = Date.now() + (o.blockMs || 10000);
        while (!abortRequested()) {
          if (Date.now() > deadline) return { status: 0, text: 'never aborted' };
          sleep(2);
        }
        return { status: 1 };
      }
      if (o.workMs) sleep(o.workMs);
      if (o.heapGrowth && counters.transcribes === 1) engine.heapBytes += o.heapGrowth;
      if (failNext > 0) {
        failNext--;
        return { status: -1 };
      }
      return { status: 0, text: o.emptyTextAt === counters.transcribes ? '' : (o.text ?? 'dobrý den'), ...capInfo, language: o.noLanguage ? '' : String(languages).split(',')[0] };
    },
    wipe() {
      counters.wipes++;
      report('wipe');
    },
  };
  if (!gpu) return engine;

  // The GPU engine: the whisper calls return Promises and settle later (JSPI); everything else stays synchronous.
  const cpu = { init: engine.whisperInit, free: engine.whisperFree };
  let busy = false;
  let stopRequested = false;
  const startNo = gpuCounters.starts;
  let testFailMs = o.testFailRecognitionMs || 0;
  const vadInit = engine.vadInit;
  engine.vadInit = (model) => {
    if (o.gpuAttachFailsAfter && startNo > o.gpuAttachFailsAfter) throw new Error('fake WebGPU speech gate failed');
    return vadInit(model);
  };
  const later = (ms, fn) => new Promise((resolve, reject) => setTimeout(() => {
    try { resolve(fn()); } catch (e) { reject(e); }
  }, ms));
  engine.whisperInit = (reader) => {
    if (engine.failure) return Promise.reject(engine.failure);
    if (o.gpuFailLoad) return later(1, () => { throw new Error('fake WebGPU: out of GPU memory while loading'); });
    return later(2, () => {
      const h = cpu.init(reader);
      engine.gpuBytes = 480 << 20;
      return h;
    });
  };
  engine.whisperTranscribe = (h, token, samples, threads, limits, languages) => {
    if (engine.failure) return Promise.reject(engine.failure);
    if (busy) return Promise.reject(new Error('fake WebGPU: re-entered while a recognition was running'));
    if (!live.has(h)) return Promise.reject(new Error('use of a freed context'));
    const n = ++gpuCounters.transcribes;
    counters.transcribes++;
    passedToWhisper.push(samples);
    report('gpuTranscribeStarted', { limits, languages });
    if (o.gpuHangAt === n) return new Promise(() => {});
    busy = true;
    stopRequested = false;
    if (o.gpuSlowAt === n) {
      // whisper.cpp checks its abort flag between decoder steps: a stop request ends the call within a step
      return new Promise((resolve) => {
        const t0 = Date.now();
        const step = setInterval(() => {
          if (!stopRequested && !abortRequested() && Date.now() - t0 < o.gpuSlowMs) return;
          clearInterval(step);
          busy = false;
          const text = o.gpuText ?? o.text ?? 'dobrý den';
          if (stopRequested && o.gpuSlowCompletes) resolve({ status: 0, text, ...capInfo, language: String(languages).split(',')[0] });
          else resolve(Date.now() - t0 < o.gpuSlowMs ? { status: 1 } : { status: 0, text, ...capInfo, language: String(languages).split(',')[0] });
        }, 5);
      });
    }
    const simulatedLoss = testFailMs > 0;
    testFailMs = 0;
    return later(o.gpuWorkMs ?? 5, () => {
      busy = false;
      if (o.heapGrowth && counters.transcribes === 1) engine.heapBytes += o.heapGrowth;
      const failsInSequence = o.gpuFailSequence ? !!o.gpuFailSequence[Atomics.add(o.gpuSeq, 0, 1)] : false;
      if (o.gpuFailAt === n || failsInSequence) {
        engine.fail(new Error('ggml aborted: fake device error'));
        throw engine.failure;
      }
      if (simulatedLoss) {
        engine.fail(new Error('simulated WebGPU device loss (test hook)'));
        throw engine.failure;
      }
      if (abortRequested()) return { status: 1 };
      return { status: 0, text: o.gpuText ?? o.text ?? 'dobrý den', ...capInfo, language: String(languages).split(',')[0] };
    });
  };
  engine.stop = () => {
    if (busy) stopRequested = true;
  };
  engine.whisperFree = (h) => later(1, () => cpu.free(h));
  engine.wipe = () => later(1, () => {
    counters.wipes++;
    report('gpuWipe');
  });
  return engine;
}

const GGML_MAGIC = new Uint8Array([0x6c, 0x6d, 0x67, 0x67]);

async function probeFakeGpu({ options } = {}) {
  const o = options || {};
  const adapter = { vendor: 'fakevendor', architecture: 'fakearch', device: '', description: '' };
  if (o.gpu === 'ok') return { ok: true, adapter, subgroups: true };
  if (o.gpu === 'ok-no-subgroups') return { ok: true, adapter, subgroups: false };
  if (o.gpu === 'no-adapter') return { ok: false, reason: 'requestAdapter() gave no adapter' };
  if (o.gpu === 'no-f16') return { ok: false, reason: 'the adapter (fakevendor fakearch) has no shader-f16' };
  return { ok: false, reason: 'navigator.gpu is not available in the worker' };
}

installWorker(self, {
  loadEngine: loadFakeEngine,
  openModel(source) {
    if (String(source.path || '').includes('nonexistent')) throw new Error('Model file not found');
    const head = String(source.path || '').includes('not-ggml') ? new Uint8Array(4) : GGML_MAGIC;
    return { size: source.size || 1, path: source.path, read: (pos, n) => head.slice(pos, pos + n) };
  },
  probeGpu: probeFakeGpu,
});
