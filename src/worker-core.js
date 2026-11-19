// Message handling of the two CzechSpeech workers. The engine (WebAssembly, or a fake in the unit tests) and the
// model reader are injected by the entry module (src/worker.js), so this file has no environment dependencies.
//
//   capture worker:    512-sample frames (from the AudioWorklet, through a MessagePort) -> streaming Silero VAD ->
//                      Endpointer -> utterances, sent straight to the recogniser through a second MessagePort.
//                      Always the single-threaded module: the VAD is tiny.
//   recogniser worker: speech gate (Silero VAD over the whole utterance) -> Whisper -> letters/digits filter ->
//                      repetition-loop guard (src/repetition.js).
//                      Multi-threaded module when the page is cross-origin isolated.
//
// This mirrors the two threads of the Android SDK (capture thread, whisper thread). Audio never passes through the
// page's main thread during listen(), and every array that held audio is zeroed once it has been used or dropped.
//
// The recogniser's backend (src/backend.js): WebGPU (czechspeech-webgpu, an async engine: its whisper calls return
// Promises) or the CPU modules (synchronous). The capture worker always uses the single-threaded CPU module, so speech
// detection is the same on every backend. Recogniser work is serialised: with an async engine the worker keeps
// handling messages while whisper.cpp waits for the GPU, and a second utterance must not enter whisper.cpp then.

import { GPU_RETRY_DELAYS_MS, GpuRecovery, backendPlan } from './backend.js';
import { Endpointer, FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE, STOP_TAIL_MAX_FRAMES } from './endpointer.js';
import { WHISPER_TOKEN_LIMIT, decodingLimits, tokenLimit } from './decoding.js';
import { describeLoops, removeRepetitionLoops } from './repetition.js';
import { errorCode } from './events.js';
import { speechSpan } from './speakers/maths.js';

// Kotlin's Char.isLetterOrDigit: letters (L*) and decimal digits (Nd).
const LETTER_OR_DIGIT = /[\p{L}\p{Nd}]/u;

/** Slots of the shared cancellation table (src/speech.js): token t is cancelled when slot t % SLOTS holds t. */
export const CANCEL_SLOTS = 256;

/**
 * Watchdog of the async (WebGPU) engine, ms (superviseCall). Recognition: base + per second of audio (a GPU slower than
 * three times real time is slower than the CPU fallback anyway); a recognition over its limit is asked to stop and
 * given recognizeStopGrace to do so (whisper.cpp checks after the encoder pass and after each decoder step) before the
 * engine is marked broken. Start, load and free cannot be stopped: over the limit the engine is marked broken at once.
 * gpuRetryMs: 'auto' starts WebGPU again this long after a failure (src/backend.js GpuRecovery). The internal option
 * _gpuTimeouts may also set recognizeLimitFirstMs: [ms, ...], the limits of the first recognitions (a test hook).
 */
export const ASYNC_TIMEOUTS = Object.freeze({ start: 30000, load: 120000, recognizeBase: 20000, recognizePerSecond: 3000,
  recognizeStopGrace: 5000, free: 10000, gpuRetryMs: GPU_RETRY_DELAYS_MS });

const now = () => performance.now();
const since = (t0) => Math.round(now() - t0);

/**
 * @param {object} scope   the worker global (self) or an equivalent with postMessage/onmessage
 * @param {object} deps
 * @param {(options: object) => Promise<object>} deps.loadEngine
 * @param {(source: object) => { size: number, read(position: number, length: number): Uint8Array }} deps.openModel
 * @param {() => Promise<{ ok: boolean, reason?: string, adapter?: object }>} [deps.probeGpu]  src/backend.js probeWebGPU
 */
export function installWorker(scope, { loadEngine, openModel, probeGpu }) {
  let impl = null;
  let initializing = null;
  const post = (msg, transfer) => scope.postMessage(msg, transfer || []);
  const log = (level, message) => post({ type: 'log', level, message });
  const fatal = (err, fallback = 'runtime') => post({ type: 'fatal', code: errorCode(err && err.code, fallback),
    message: errorMessage(err) });

  scope.onmessage = (e) => {
    const msg = e.data;
    if (msg && msg.type === 'init') {
      initializing = init(msg).catch((err) => fatal(err, 'runtime'));
      return;
    }
    const run = () => {
      if (!impl) return dropAudio(msg);
      try {
        impl.handle(msg);
      } catch (err) {
        fatal(err);
      }
    };
    if (initializing && !impl) initializing.then(run);
    else run();
  };

  async function init(msg) {
    const t0 = now();
    const abortView = msg.abort ? new Int32Array(msg.abort) : null;
    let current = null;
    // Without SharedArrayBuffer (no cross-origin isolation) nothing can reach a busy worker; the page terminates it.
    const cancelled = (token) => !!abortView && token > 0 && Atomics.load(abortView, token % CANCEL_SLOTS) === token;
    const engineArgs = (gpu) => ({
      moduleUrl: gpu ? msg.gpuModuleUrl : msg.moduleUrl,
      wasmUrl: gpu ? msg.gpuWasmUrl : msg.wasmUrl,
      threads: gpu ? 1 : msg.threads || 1,
      abortRequested: () => cancelled(current),
      log,
      post,
      options: msg.engineOptions,
      gpu,
    });
    if (msg.role === 'capture') {
      const engine = await loadEngine(engineArgs(false));
      const vad = engine.vadInit(new Uint8Array(msg.vadModel));
      impl = new Capture(engine, vad, msg.config, post, log);
      post({ type: 'ready', ms: since(t0), heapBytes: engine.heapBytes, systemInfo: engine.systemInfo() });
      return;
    }
    const backends = new Backends(msg, { loadEngine, probeGpu, engineArgs, log });
    const chosen = await backends.start();
    const rec = new Recognizer(chosen, backends, msg, openModel, cancelled, (token) => { current = token; }, post, log);
    impl = rec;
    post({ type: 'ready', ms: since(t0), heapBytes: chosen.engine.heapBytes, systemInfo: chosen.engine.systemInfo(),
      ...rec.backendState(), gpuModuleFailed: !!chosen.moduleFailed });
  }
}

/** Starts recogniser engines: WebGPU first where the plan says so, the CPU module otherwise or as the fallback. */
class Backends {
  constructor(msg, { loadEngine, probeGpu, engineArgs, log }) {
    this.requested = msg.backend || 'cpu';
    // 'auto' after WebGPU failures: the instance's recovery state (src/speech.js); before its retry time the recogniser
    // starts on the CPU without trying WebGPU
    this.recovery = msg.gpuRecovery || null;
    this.plan = backendPlan(this.requested, { gpuModule: !!msg.gpuModuleUrl, blocked: msg.gpuBlocked || null,
      retryInMs: this.recovery ? this.recovery.retryInMs : null, failure: msg.gpuFailure || null });
    this.loadEngine = loadEngine;
    this.probeGpu = probeGpu;
    this.engineArgs = engineArgs;
    this.engineOptions = msg.engineOptions || null;
    this.log = log;
    this.timeouts = { ...ASYNC_TIMEOUTS, ...(msg.timeouts || {}) };
    // test hook: the simulated device loss (engine option testFailRecognitionMs) goes to this many WebGPU engines of the
    // recogniser (testFailRecognitionCount, default 1), so that the module started again after it recognises
    const o = this.engineOptions || {};
    this.testFailuresLeft = o.testFailRecognitionMs ? (Number.isInteger(o.testFailRecognitionCount) ? o.testFailRecognitionCount : 1) : 0;
  }

  /** { engine, backend, adapter, fallback }; throws code 'unsupported' when WebGPU is required and not available. */
  async start() {
    let reason = this.plan.reason;
    let moduleFailed = false;
    if (this.plan.tryGpu) {
      const gpu = await this.startGpu();
      if (gpu.engine) return gpu;
      reason = gpu.reason;
      moduleFailed = !!gpu.moduleFailed;
    }
    if (this.plan.onUnavailable === 'fail') throw unsupported(reason);
    return { engine: await this.startCpu(), backend: 'cpu', adapter: null, fallback: reason, moduleFailed };
  }

  /** { engine, backend: 'webgpu', adapter } or { engine: null, reason, moduleFailed }. Never throws. */
  async startGpu() {
    let probe;
    try {
      probe = this.probeGpu ? await this.probeGpu({ options: this.engineOptions })
        : { ok: false, reason: 'no WebGPU probe in this worker' };
    } catch (err) {
      probe = { ok: false, reason: `the WebGPU probe failed: ${errorMessage(err)}` };
    }
    if (!probe || !probe.ok) return { engine: null, reason: (probe && probe.reason) || 'WebGPU is not available' };
    // Attention on WebGPU: whisper.cpp's matrix-multiplication path, not flash attention. With the patched Q4_0 kernel
    // (native/patches/) it was faster on the Intel Xe-LPG (encoder 0.88 s instead of 1.09 s for e-rec; 150 MiB more
    // GPU buffers), and it needs no subgroups (ggml's flash-attention shader does). An explicit engine option wins.
    const args = this.engineArgs(true);
    args.options = { flashAttn: 0, ...(args.options || {}) };
    if (this.testFailuresLeft > 0) this.testFailuresLeft--;
    else delete args.options.testFailRecognitionMs;
    const loading = Promise.resolve().then(() => this.loadEngine(args));
    try {
      const engine = await withWatchdog(loading, this.timeouts.start, 'starting the WebGPU module');
      return { engine, backend: 'webgpu', adapter: engine.adapter || probe.adapter || null };
    } catch (err) {
      // a module that starts after the watchdog gave up is released at once
      loading.then((late) => late && late.wipe && late.wipe(), () => {});
      return { engine: null, reason: `the WebGPU module did not start: ${errorMessage(err)}`, moduleFailed: true };
    }
  }

  async startCpu() {
    return this.loadEngine(this.engineArgs(false));
  }
}

function unsupported(reason) {
  return Object.assign(new Error(`WebGPU is not available: ${reason || 'unknown reason'}`), { code: 'unsupported', fatal: true });
}

/** Resolves like `promise`, or rejects after `ms` (calling onTimeout first). */
function withWatchdog(promise, ms, what, onTimeout) {
  return superviseCall(() => promise, { limitMs: ms, what, kill: onTimeout || (() => {}) }).then((r) => r.value);
}

/**
 * The watchdog of an async (WebGPU) engine call. `call()` starts the call and returns its Promise. Within `limitMs`
 * its result is passed on as { value, stopped: false }. Past the limit, `stop()` (when given) asks it to end
 * cooperatively (whisper.cpp's abort check: the engine and its device stay usable); a call that then ends within
 * `graceMs` gives { value, stopped: true }. A call that cannot be stopped, or does not end within the grace, is given up:
 * `kill(err)` (the engine is marked broken and its device destroyed) and the Promise rejects with `err`, which has
 * `watchdog: true`. A call that fails before that rejects with its own error. Timers are injectable for the tests.
 */
export function superviseCall(call, { limitMs, graceMs = 0, what, stop = null, kill, timers = globalThis }) {
  return new Promise((resolve, reject) => {
    let done = false;
    let stopping = false;
    let timer = null;
    const finish = (fn, arg) => {
      if (done) return;
      done = true;
      if (timer !== null) timers.clearTimeout(timer);
      timer = null;
      fn(arg);
    };
    const giveUp = () => {
      timer = null;
      const err = Object.assign(new Error(`${what} did not finish in ${Math.round(limitMs / 1000)} s`), { watchdog: true });
      kill(err);
      finish(reject, err);
    };
    timer = timers.setTimeout(() => {
      if (!stop) return giveUp();
      stopping = true;
      stop();
      timer = timers.setTimeout(giveUp, graceMs);
      return undefined;
    }, limitMs);
    let pending;
    try {
      pending = Promise.resolve(call());
    } catch (err) {
      pending = Promise.reject(err);
    }
    pending.then((value) => finish(resolve, { value, stopped: stopping }), (err) => finish(reject, err));
  });
}

function errorMessage(err) {
  if (err instanceof Error) return err.message;
  if (typeof WebAssembly !== 'undefined' && err instanceof WebAssembly.RuntimeError) return `WebAssembly: ${err.message}`;
  return String(err);
}

function dropAudio(msg) {
  if (!msg) return;
  if (msg.samples instanceof Float32Array) msg.samples.fill(0);
}

// ---------------------------------------------------------------------------------------------------------------------

class Capture {
  constructor(engine, vad, config, post, log) {
    this.engine = engine;
    this.vad = vad;
    this.config = config;
    this.post = post;
    this.log = log;
    this.seq = 0;
    this.ids = 0;
    this.sent = 0;
    this.frames = 0;
    this.vadMs = 0;
    this.stopped = false;
    this.ended = false;
    this.endpointer = null;
    this.audioPort = null;
    this.utterancePort = null;
    this.speakerPort = null;   // a copy of each utterance's speech goes here (listen({ speakers })), or null
    this.speakers = false;     // false while the page asked not to compute speakers (the recogniser runs on the CPU...)
    this.tail = -1;          // stop(): frames since it while the word in progress finishes; -1 when not waiting
    this.stopAtFrame = 0;    // stop(): frames processed when it came
    this.drained = false;    // stop(): the wait is over; audio after this is dropped until the end marker
  }

  handle(msg) {
    switch (msg.type) {
      case 'start': return this.#start(msg);
      case 'drain': return this.#drain();
      case 'speakers': this.speakers = !!msg.enabled; return undefined;
      case 'stop': return this.#stop();
      // stop() when the end marker could not come on the audio port (see Session.finish in src/speech.js)
      case 'finish': return this.#end();
      default: throw new Error(`capture worker: unknown message ${msg.type}`);
    }
  }

  /** No more audio: an utterance in progress ends as if silence followed, and the page learns how many there were. */
  #end() {
    if (this.stopped || this.ended || !this.endpointer) return;
    this.ended = true;
    this.endpointer.finish();
    this.#logStats();
    this.post({ type: 'ended', utterances: this.sent, frames: this.frames });
  }

  /**
   * stop() during listen() (the Android SDK's capture loop does the same): with speech in progress, capture goes on
   * until a pause of STOP_TAIL_SILENCE_MS (one under way counts) or STOP_TAIL_MAX_MS of audio, so that the word in
   * progress is not cut off; then, or at once without speech, the page is told 'drained' and ends the microphone
   * through its finish path (the end marker after the last frame). Audio arriving after 'drained' is dropped, so
   * the utterance ends where the wait ended.
   */
  #drain() {
    if (this.stopped || this.ended || this.drained || this.tail >= 0) return;
    if (!this.endpointer) {
      this.drained = true;
      this.post({ type: 'drained' });
      return;
    }
    this.stopAtFrame = this.frames;
    this.tail = 0;
    this.#checkTail();
  }

  #checkTail() {
    const atPause = this.endpointer.atPause;
    if (!atPause && this.tail < STOP_TAIL_MAX_FRAMES) return;
    const at = `capture: stop() at ${this.stopAtFrame * FRAME_MS} ms of audio`;
    this.log('info', this.tail === 0 ? `${at}, no word in progress`
      : `${at}, waited ${this.tail * FRAME_MS} ms ${atPause ? 'for a pause' : '(the limit)'}`);
    this.tail = -1;
    this.drained = true;
    this.post({ type: 'drained' });
  }

  #start({ audioPort, utterancePort, speakerPort = null, speakers = false }) {
    this.endpointer = new Endpointer(this.config, (ev) => this.#onEndpointer(ev));
    this.audioPort = audioPort;
    this.utterancePort = utterancePort;
    this.speakerPort = speakerPort;
    this.speakers = !!speakerPort && !!speakers;
    audioPort.onmessage = (e) => this.#onAudio(e.data);
  }

  #onAudio(d) {
    if (d instanceof Float32Array) {
      if (this.stopped || this.ended || this.drained) {
        d.fill(0);
        return;
      }
      if (d.length !== FRAME_SAMPLES) {
        d.fill(0);
        this.post({ type: 'fatal', code: 'runtime', message: `audio frames must have ${FRAME_SAMPLES} samples` });
        return;
      }
      try {
        const t0 = now();
        const prob = this.engine.vadFrame(this.vad, d);
        this.vadMs += now() - t0;
        this.endpointer.accept(d, prob);
        this.frames++;
        if (this.tail >= 0) {
          this.tail++;
          this.#checkTail();
        }
      } catch (err) {
        this.post({ type: 'fatal', code: 'runtime', message: errorMessage(err) });
      } finally {
        d.fill(0);
      }
      return;
    }
    if (!d || this.stopped || this.ended) return;
    if (d.type === 'end') {
      // The source ended (stop(): the microphone's last frame came before this; or a finite test source).
      this.#end();
    } else if (d.type === 'error') {
      this.post({ type: 'fatal', code: errorCode(d.code, 'microphone-unavailable'), message: d.message || 'audio source failed' });
    }
  }

  #onEndpointer(ev) {
    if (ev.kind === 'speechStarted') {
      this.post({ type: 'event', seq: ++this.seq, event: 'speechStarted', atMs: ev.atMs });
    } else if (ev.kind === 'speechEnded') {
      this.engine.vadReset(this.vad);
      if (ev.confirmed) this.post({ type: 'event', seq: ++this.seq, event: 'speechEnded', atMs: ev.atMs });
    } else if (ev.kind === 'utterance') {
      if (!this.utterancePort) {
        ev.samples.fill(0);
        return;
      }
      const id = ++this.ids;
      const meta = { id, afterSeq: this.seq, startMs: ev.startMs, endMs: ev.endMs };
      // The speech span only (no trailing silence), as a copy: the recogniser gets `samples` itself, by transfer
      let speaker = false;
      if (this.speakerPort && this.speakers) {
        const copy = speechSpan(ev.samples, ev.audioStartMs, ev.endMs).slice();
        this.speakerPort.postMessage({ type: 'utterance', id, startMs: ev.startMs, audioStartMs: ev.audioStartMs, endMs: ev.endMs, samples: copy }, [copy.buffer]);
        speaker = true;
      }
      this.post({ type: 'utterance', ...meta, samples: ev.samples.length, speaker });
      this.utterancePort.postMessage({ type: 'utterance', ...meta, samples: ev.samples }, [ev.samples.buffer]);
      this.sent++;
    }
  }

  #logStats() {
    const perFrameUs = this.frames > 0 ? Math.round((this.vadMs * 1000) / this.frames) : 0;
    this.log('info', `capture: ${this.frames * 32} ms of audio, VAD ${perFrameUs} us per 32 ms frame`);
  }

  #stop() {
    if (!this.ended) this.#logStats();
    this.stopped = true;
    this.endpointer?.clear();
    if (this.audioPort) {
      this.audioPort.onmessage = null;
      this.audioPort.close();
    }
    if (this.utterancePort) {
      // Last message on the utterance channel: the recogniser has seen every utterance once it reads this. The
      // recogniser closes the channel; closing it here could drop this message while it is in flight.
      this.utterancePort.postMessage({ type: 'close' });
    }
    const heapBytes = this.engine.heapBytes;
    this.engine.vadFree(this.vad);
    this.engine.wipe();
    this.post({ type: 'stopped', heapBytes });
  }
}

// ---------------------------------------------------------------------------------------------------------------------

class Recognizer {
  constructor(chosen, backends, init, openModel, cancelled, setCurrent, post, log) {
    this.backends = backends;
    this.requested = backends.requested;
    this.timeouts = backends.timeouts;
    this.vadModel = new Uint8Array(init.vadModel);
    this.config = init.config;
    this.threads = init.threads || 1;
    this.modelSource = init.model;
    this.engineOptions = init.engineOptions || {};
    this.openModel = openModel;
    this.cancelled = cancelled;
    this.setCurrent = setCurrent;
    this.post = post;
    this.log = log;
    this.engine = null;
    this.gate = 0;
    this.whisper = 0;
    this.backend = chosen.backend;
    this.adapter = chosen.adapter || null;
    this.fallback = chosen.fallback || null;
    this.freed = false;
    this.ports = new Map();
    this.tail = Promise.resolve();
    // test hook (internal timeouts option): the limits of the first recognitions, before the usual ones
    this.firstLimitsMs = Array.isArray(this.timeouts.recognizeLimitFirstMs) ? [...this.timeouts.recognizeLimitFirstMs] : [];
    // 'auto' starts the WebGPU module again some time after a failure (src/backend.js GpuRecovery), from the state of
    // the instance (src/speech.js), to which it reports every change
    this.recovery = null;
    if (this.requested === 'auto') {
      this.recovery = new GpuRecovery({ delaysMs: this.timeouts.gpuRetryMs });
      if (backends.recovery) this.recovery.adopt(backends.recovery);
      // a WebGPU module that does not start when it is tried again after a failure is one more failure in a row
      if (chosen.moduleFailed && this.recovery.failures > 0) {
        this.recovery.failed();
        this.#reportRecovery(`WebGPU failed: ${chosen.fallback}`);
      }
    }
    this.#attach(chosen.engine);
  }

  /** Sends the recovery state to the page (src/speech.js #onGpuRecovery), with the reason of the last failure. */
  #reportRecovery(reason = null) {
    this.post({ type: 'gpuRecovery', ...this.recovery.snapshot(), reason });
  }

  /** What the page shows in speech.state (no content). */
  backendState() {
    return { backend: this.backend, gpuAdapter: this.adapter, backendFallback: this.fallback };
  }

  /** Takes `engine` into use with a speech gate of its own; when the gate does not load, nothing changes and it throws. */
  #attach(engine) {
    this.#use(engine, engine.vadInit(this.vadModel));
  }

  #use(engine, gate) {
    this.engine = engine;
    this.gate = gate;
  }

  /**
   * The speech gate of an engine started after the recogniser's start-up. An engine whose gate does not load is
   * released and the error thrown: it is never left in use without a gate.
   */
  async #newGate(engine, what) {
    try {
      return engine.vadInit(this.vadModel);
    } catch (err) {
      try {
        await engine.wipe();
      } catch (e) {
        this.log('warn', `releasing the ${what}: ${errorMessage(e)}`);
      }
      throw err;
    }
  }

  /** Starts the CPU module and takes it into use ('auto' after a WebGPU failure); throws when it cannot. */
  async #startCpu() {
    const engine = await this.backends.startCpu();
    this.#use(engine, await this.#newGate(engine, 'CPU module'));
  }

  handle(msg) {
    switch (msg.type) {
      case 'load': return this.#enqueue(() => this.#load());
      case 'check': return this.#check();
      case 'listen': return this.#listen(msg);
      case 'transcribe': return this.#enqueue(() => this.#transcribe(msg));
      case 'free': return this.#enqueue(() => this.#free());
      default: throw new Error(`recogniser worker: unknown message ${msg.type}`);
    }
  }

  /**
   * Runs recogniser jobs one after the other. With the CPU engine a job never yields to the event loop (as in 0.1:
   * the model is loaded in the same task as the first recognition); with the WebGPU engine it does, while whisper.cpp
   * waits for the GPU, and later messages wait here.
   */
  #enqueue(job) {
    const run = this.tail.then(job);
    this.tail = run.catch((err) => {
      this.post({ type: 'fatal', code: errorCode(err && err.code, 'runtime'), message: errorMessage(err) });
    });
  }

  /**
   * An engine call. Async (WebGPU) engines run under the watchdog (superviseCall): a recognition (`stoppable`) that
   * overruns its limit is asked to stop at whisper.cpp's next abort check and resolves { value, stopped: true }; only a
   * call that does not end within the grace, or one that cannot be stopped (model load, free), marks the engine broken
   * and destroys its device. Stoppable calls resolve { value, stopped }, the others their value.
   */
  async #call(what, ms, fn, { stoppable = false } = {}) {
    const engine = this.engine;
    if (!engine.async) {
      const value = await fn(engine);
      return stoppable ? { value, stopped: false } : value;
    }
    const r = await superviseCall(() => fn(engine), {
      limitMs: ms, graceMs: this.timeouts.recognizeStopGrace, what: `WebGPU: ${what}`,
      stop: stoppable && typeof engine.stop === 'function' ? () => engine.stop() : null,
      kill: (err) => { if (typeof engine.fail === 'function') engine.fail(err); },
    });
    return stoppable ? r : r.value;
  }

  /** ' from now on' or ', WebGPU is tried again in N s' ('auto' after a WebGPU failure), for the log. */
  #retryNote() {
    if (!this.recovery || this.recovery.gaveUp) return ' from now on';
    return `, WebGPU is tried again in ${this.recovery.retryInSeconds} s`;
  }

  /**
   * The WebGPU engine failed (ggml abort, device lost, a watchdog that had to give up): release it and go on according
   * to config.backend. 'auto' goes on with the CPU module and starts WebGPU again later (#ensureEngine). When the CPU
   * module does not start, the recogniser stays without an engine and this throws (code 'runtime', both reasons): the
   * caller's utterance fails, and the next one starts the CPU module again, or WebGPU once its retry time has come.
   */
  async #gpuFailed(err) {
    const reason = errorMessage(err);
    const old = this.engine;
    this.engine = null;
    this.gate = 0;
    this.whisper = 0;
    try {
      // clears (when it still can) and destroys the GPU buffers and the device, zeroes the WebAssembly heap
      await old.wipe();
    } catch (e) {
      this.log('warn', `releasing the failed WebGPU module: ${errorMessage(e)}`);
    }
    if (this.requested === 'auto') {
      this.recovery.failed();
      this.log('warn', `WebGPU failed (${reason}); recognising on the CPU${this.#retryNote()}`);
      this.backend = 'cpu';
      this.adapter = null;
      this.fallback = `WebGPU failed: ${reason}`;
      this.#reportRecovery(this.fallback);
      let cpuError = null;
      try {
        await this.#startCpu();
      } catch (e) {
        cpuError = e;
        this.log('warn', `the CPU module did not start (${errorMessage(e)}); it is started again for the next utterance`);
      }
      this.post({ type: 'backend', ...this.backendState(), gpuFailed: true, reason });
      if (cpuError) {
        throw Object.assign(new Error(`WebGPU failed (${reason}) and the CPU module did not start: ${errorMessage(cpuError)}`),
          { code: 'runtime' });
      }
      return;
    }
    this.log('warn', `WebGPU failed (${reason}); the WebGPU module is restarted for the next utterance`);
    this.post({ type: 'backend', ...this.backendState(), gpuFailed: true, reason });
  }

  /**
   * 'auto', on the CPU after a WebGPU failure, when the retry delay has passed: starts the WebGPU module again and, when
   * it starts and takes its speech gate, releases the CPU engine (the model is loaded onto the GPU with the next
   * recognition). When it does not, the new engine is released and that counts as one more failure in a row. There is
   * no CPU engine to release when the CPU module did not start after the failure (#gpuFailed).
   */
  async #restoreGpu() {
    const gpu = await this.backends.startGpu();
    let gate = 0;
    let failure = gpu.engine ? null : gpu.reason;
    if (gpu.engine) {
      try {
        gate = await this.#newGate(gpu.engine, 'WebGPU module');
      } catch (err) {
        failure = `the WebGPU module could not be used: ${errorMessage(err)}`;
      }
    }
    if (failure) {
      this.recovery.failed();
      this.log('warn', `WebGPU could not be started again (${failure}); recognising on the CPU${this.#retryNote()}`);
      this.fallback = `WebGPU failed: ${failure}`;
      this.#reportRecovery(this.fallback);
      this.post({ type: 'backend', ...this.backendState(), gpuFailed: true, reason: failure });
      return;
    }
    if (this.engine) {
      try {
        await this.#releaseEngine();
      } catch (err) {
        this.log('warn', `releasing the CPU module: ${errorMessage(err)}`);
      }
    }
    this.#use(gpu.engine, gate);
    this.backend = 'webgpu';
    this.adapter = gpu.adapter;
    this.fallback = null;
    this.log('info', 'WebGPU started again after a failure');
    this.post({ type: 'backend', ...this.backendState(), gpuRestored: true });
  }

  /**
   * Releases the current engine: frees the model and the speech gate, then (whatever failed before) wipe() zeroes the
   * heap, and a threaded CPU module tears its runtime down, so that its memory can be collected (src/engine-wasm.js).
   * The one release path of the recogniser: at the end (#free) and when WebGPU replaces the CPU module (#restoreGpu).
   */
  async #releaseEngine() {
    const engine = this.engine;
    if (!engine) return;
    try {
      if (this.whisper) {
        try {
          await this.#call('freeing the model', this.timeouts.free, (e) => e.whisperFree(this.whisper));
        } catch (err) {
          this.log('warn', `freeing the model: ${errorMessage(err)}`);
        }
        this.whisper = 0;
        this.log('info', 'model freed');
      }
      try {
        engine.vadFree(this.gate);
      } catch (err) {
        if (!engine.failure) throw err;
      }
    } finally {
      this.engine = null;
      this.gate = 0;
      await engine.wipe();
    }
  }

  /**
   * Restarts the WebGPU engine after a failure (config.backend 'webgpu'); fatal 'unsupported' when it cannot start or
   * its speech gate does not load (that engine is released). 'auto' on the CPU after a failure starts it again once the
   * retry delay has passed, and starts the CPU module when it did not start after the failure.
   */
  async #ensureEngine() {
    if (this.engine && this.engine.failure) await this.#gpuFailed(this.engine.failure);
    if (this.recovery && this.backend === 'cpu' && this.recovery.due) await this.#restoreGpu();
    if (this.engine) return;
    if (this.requested !== 'webgpu') {
      try {
        await this.#startCpu();
      } catch (err) {
        throw Object.assign(new Error(`the CPU module did not start: ${errorMessage(err)}`), { code: 'runtime' });
      }
      this.log('info', 'CPU module started again after a failure');
      return;
    }
    const gpu = await this.backends.startGpu();
    if (!gpu.engine) throw unsupported(gpu.reason);
    let gate;
    try {
      gate = await this.#newGate(gpu.engine, 'WebGPU module');
    } catch (err) {
      throw unsupported(`the WebGPU module could not be used: ${errorMessage(err)}`);
    }
    this.#use(gpu.engine, gate);
    this.adapter = gpu.adapter;
    this.log('info', 'WebGPU module restarted');
    this.post({ type: 'backend', ...this.backendState() });
  }

  /** Runs job(); a WebGPU failure is handled by #gpuFailed, then 'auto' runs the job again on the CPU. */
  async #withGpuFallback(job) {
    await this.#ensureEngine();
    try {
      return await job();
    } catch (err) {
      if (!err || !err.gpu) throw err;
      await this.#gpuFailed(err);
      if (this.requested !== 'auto') {
        throw Object.assign(new Error(`WebGPU failed: ${errorMessage(err)}`), { code: 'recognition' });
      }
      return job();
    }
  }

  async #ensureWhisper() {
    if (this.whisper) return this.whisper;
    if (this.freed) throw Object.assign(new Error('the recogniser was freed'), { fatal: true });
    const t0 = now();
    const fatalModel = (err) => Object.assign(new Error(`Could not load the model: ${errorMessage(err)}`), { fatal: true, code: 'model' });
    let reader;
    try {
      reader = this.openModel(this.modelSource);
    } catch (err) {
      throw fatalModel(err);
    }
    try {
      this.whisper = await this.#call('loading the model', this.timeouts.load, (e) => e.whisperInit(reader));
    } catch (err) {
      if (this.engine.async) throw gpuError(err);
      throw fatalModel(err);
    }
    const loadMs = since(t0);
    const heapBytes = this.engine.heapBytes;
    const gpuBytes = this.engine.gpuBytes || 0;
    const where = this.engine.async ? `WebGPU, GPU buffers ${mib(gpuBytes)} MiB`
      : `${this.threads} thread${this.threads > 1 ? 's' : ''}`;
    this.log('info', `model loaded in ${loadMs} ms (${Math.round(this.modelSource.size / 1e6)} MB, ${where}, heap ${mib(heapBytes)} MiB)`);
    this.post({ type: 'modelLoaded', loadMs, heapBytes, gpuBytes, backend: this.backend });
    return this.whisper;
  }

  /**
   * Loads the model at session start (the page asks for it on WebGPU). When that fails on the GPU, the engine is
   * replaced as after any WebGPU failure but the model is not loaded on the CPU yet: there it loads with the first
   * utterance (its fast path, see src/speech.js).
   */
  async #load() {
    try {
      await this.#ensureEngine();
      try {
        await this.#ensureWhisper();
      } catch (err) {
        if (!err || !err.gpu) throw err;
        await this.#gpuFailed(err);
      }
    } catch (err) {
      this.post({ type: 'fatal', code: errorCode(err && err.code, 'model'), message: errorMessage(err) });
    }
  }

  /** Cheap check at session start that the model is a ggml file, so a wrong file fails before any speech. */
  #check() {
    try {
      const head = this.openModel(this.modelSource).read(0, 4);
      // GGML_FILE_MAGIC 0x67676d6c, little-endian
      if (head.length !== 4 || head[0] !== 0x6c || head[1] !== 0x6d || head[2] !== 0x67 || head[3] !== 0x67) {
        throw new Error('not a whisper.cpp (ggml) model file');
      }
      this.post({ type: 'checked' });
    } catch (err) {
      this.post({ type: 'fatal', code: 'model', message: `Could not load the model: ${errorMessage(err)}` });
    }
  }

  #listen({ session, token, port, languages }) {
    this.ports.set(session, port);
    port.onmessage = (e) => this.#onUtterance(e.data, session, token, port, languages);
  }

  #onUtterance(d, session, token, port, languages) {
    if (!d) return;
    if (d.type === 'close') {
      // after the utterances queued before it, like every other message of the channel
      this.#enqueue(() => {
        port.onmessage = null;
        port.close();
        this.ports.delete(session);
        this.post({ type: 'listenClosed', session });
      });
      return;
    }
    if (d.type !== 'utterance') return;
    this.#enqueue(() => this.#utterance(d, session, token, languages));
  }

  async #utterance(d, session, token, languages) {
    const { samples } = d;
    const base = { type: 'result', session, id: d.id, afterSeq: d.afterSeq, startMs: d.startMs, endMs: d.endMs };
    let reply;
    if (this.freed || this.cancelled(token)) {
      samples.fill(0);
      this.post({ ...base, text: '', processingMs: 0, cancelled: true });
      return;
    }
    this.post({ type: 'busy', busy: true });
    try {
      const r = await this.#recognize(samples, token, 'utterance', languages);
      reply = { ...base, text: r.text, language: r.language, processingMs: r.processingMs, cancelled: !!r.cancelled };
    } catch (err) {
      reply = err && err.fatal ? { type: 'fatal', code: errorCode(err.code, 'model'), message: errorMessage(err) }
        : { ...base, error: errorMessage(err) };
    } finally {
      samples.fill(0);
    }
    // 'busy: false' first: the page may release the recogniser as soon as it has the reply.
    this.post({ type: 'busy', busy: false, heapBytes: this.engine ? this.engine.heapBytes : undefined });
    this.post(reply);
  }

  async #transcribe({ id, samples, token, languages }) {
    let reply;
    this.post({ type: 'busy', busy: true });
    try {
      const r = await this.#recognize(samples, token, 'transcribe', languages);
      reply = { type: 'transcribed', id, text: r.text, language: r.language, processingMs: r.processingMs, cancelled: !!r.cancelled };
    } catch (err) {
      reply = { type: 'transcribed', id, error: errorMessage(err), code: errorCode(err && err.code, 'recognition') };
    } finally {
      samples.fill(0);
    }
    this.post({ type: 'busy', busy: false, heapBytes: this.engine ? this.engine.heapBytes : undefined });
    this.post(reply);
  }

  /** Speech gate, Whisper and filter (the Android SDK's SpeechEngine.recognize). */
  async #recognize(samples, token, what, languages) {
    const t0 = now();
    if (samples.length === 0) return { text: '', processingMs: 0 };
    if (this.cancelled(token)) return { text: '', processingMs: 0, cancelled: true };
    if (this.freed) throw Object.assign(new Error('the recogniser was freed'), { fatal: true });
    await this.#ensureEngine();
    const audioMs = Math.floor((samples.length * 1000) / SAMPLE_RATE);
    // The gate (Silero VAD over the whole utterance) runs on ggml's CPU backend in every module.
    const segments = this.engine.vadSpeechSegments(this.gate, samples, this.config.threshold);
    const gateMs = since(t0);
    if (segments < 0) throw new Error('Voice activity detection failed');
    if (segments === 0) {
      this.log('info', `${what}: ${audioMs} ms of audio, no speech (gate ${gateMs} ms), recogniser not run`);
      return { text: '', processingMs: since(t0) };
    }
    return this.#withGpuFallback(() => this.#whisper(samples, token, what, { t0, audioMs, segments, gateMs, languages }));
  }

  async #whisper(samples, token, what, { t0, audioMs, segments, gateMs, languages }) {
    const whisper = await this.#ensureWhisper();
    if (this.cancelled(token)) return { text: '', processingMs: since(t0), cancelled: true };
    const t1 = now();
    const gpu = this.engine.async;
    let r;
    let stopped;
    // max_tokens relative to the audio length, no temperature fallback (src/decoding.js)
    const limits = decodingLimits(samples.length, this.engineOptions, this.modelSource.tokensPerSecond);
    // 'cs', or 'pl,en': one language is forced, several are chosen among by whisper.cpp
    const langs = languages || (this.modelSource.languages || ['cs']).join(',');
    const ms = this.firstLimitsMs.length > 0 ? this.firstLimitsMs.shift()
      : this.timeouts.recognizeBase + Math.ceil(samples.length / SAMPLE_RATE) * this.timeouts.recognizePerSecond;
    this.setCurrent(token);
    try {
      // the WebGPU module is single-threaded: the GPU does the heavy work
      const call = await this.#call('recognition', ms, (e) => e.whisperTranscribe(whisper, token, samples, gpu ? 1 : this.threads, limits, langs),
        { stoppable: true });
      r = call.value;
      stopped = call.stopped;
    } catch (err) {
      throw gpu ? gpuError(err) : err;
    } finally {
      this.setCurrent(null);
    }
    const whisperMs = since(t1);
    if (r.status === 1 && stopped && !this.cancelled(token)) {
      // The watchdog stopped it (whisper.cpp's abort check): no transcript, and the WebGPU engine stays in use.
      this.post({ type: 'aborted' });
      this.log('warn', `${what}: ${audioMs} ms of audio, the recogniser did not finish in ${Math.round(ms / 1000)} s and was ` +
        'stopped (WebGPU stays in use), no transcript');
      return { text: '', processingMs: since(t0), cancelled: true };
    }
    if (r.status === 1 || this.cancelled(token)) {
      if (r.status === 1) this.post({ type: 'aborted' });
      this.log('info', `${what}: ${audioMs} ms of audio, recogniser cancelled after ${whisperMs} ms`);
      return { text: '', processingMs: since(t0), cancelled: true };
    }
    if (r.status !== 0) {
      const err = Object.assign(new Error('Recognition failed'), { code: 'recognition' });
      throw gpu ? gpuError(err) : err;
    }
    if (!r.language) {
      // The native layer sets the language on every successful call (the forced one, or the detected one); an empty
      // string means the module and the JS side disagree. That is a failed recognition, like any other status the
      // module should not return, not a language to make up.
      const err = Object.assign(new Error('Recognition failed'), { code: 'recognition' });
      throw gpu ? gpuError(err) : err;
    }
    if (gpu && this.recovery && this.recovery.failures > 0) {
      this.recovery.succeeded();
      this.#reportRecovery();
    }
    if (r.capped) {
      // a repetition loop (or, in theory, speech faster than src/decoding.js allows) that a token limit stopped: the cap
      // (max_tokens), or whisper.cpp's own limit when that is smaller; the log names the one that stopped it
      const byCap = limits.maxTokens > 0 && limits.maxTokens + 1 <= WHISPER_TOKEN_LIMIT;
      this.post({ type: 'capped' });
      this.log('info', `${what}: token limit reached (${r.tokens} tokens; ${byCap ? 'the cap' : 'whisper.cpp'} allows ` +
        `${tokenLimit(limits.maxTokens)})`);
    }
    const text = r.text.trim();
    // Silence and noise that slip through make Whisper emit punctuation only (",", "... ..."); drop it.
    const kept = LETTER_OR_DIGIT.test(text);
    const t = r.timings ? ` (encoder ${r.timings.encodeMs} ms, decoder ${r.timings.decodeMsPerToken} ms/token)` : '';
    this.log('info', `${what}: ${audioMs} ms of audio, ${segments} speech segment(s), gate ${gateMs} ms, ` +
      `recogniser ${whisperMs} ms${t}${gpu ? ' on WebGPU' : ''}, ${kept ? 'transcript kept' : 'no letters or digits, dropped'}`);
    if (!kept) return { text: '', processingMs: since(t0) };
    // Audio that ends mid-word (stop(), the length cap) can make greedy decoding loop; keep one copy (src/repetition.js).
    const guarded = removeRepetitionLoops(text);
    if (guarded.loops.length > 0) this.log('info', `${what}: ${describeLoops(guarded.loops)}`);
    return { text: guarded.text, language: r.language, processingMs: since(t0) };
  }

  async #free() {
    for (const port of this.ports.values()) {
      port.onmessage = null;
      port.close();
    }
    this.ports.clear();
    const engine = this.engine;
    const heapBytes = engine ? engine.heapBytes : 0;
    if (engine) await this.#releaseEngine();
    this.freed = true;
    this.post({ type: 'freed', heapBytes });
  }
}

/** Marks an error of the WebGPU engine, so that the recogniser falls back instead of failing the utterance. */
function gpuError(err) {
  const e = err instanceof Error ? err : new Error(String(err));
  e.gpu = true;
  return e;
}

const mib = (bytes) => Math.round(bytes / 2 ** 20);
