// CzechSpeech: the public speech API on the page's main thread. It owns no audio during listen(): the AudioWorklet
// sends frames to the capture worker, which sends utterances to the recogniser worker; the page only receives
// events and transcripts. See src/worker-core.js for the workers.

import { GPU_RETRY_DELAYS_MS, GpuRecovery, adapterLabel } from './backend.js';
import { resolveConfig, threadCount } from './config.js';
import { STOP_TAIL_MAX_MS } from './endpointer.js';
import { CzechSpeechError, SpeakersEvent, SpeechEndedEvent, SpeechErrorEvent, SpeechStartedEvent, TranscriptEvent, errorCode } from './events.js';
import { openMicrophone } from './mic.js';
import { CzechSpeechModel } from './model.js';
import { sha256Hex } from './sha256.js';
import { SpeakerSession } from './speakers/session.js';
import { CANCEL_SLOTS } from './worker-core.js';

export const VAD_SHA256 = '2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987';
const FREE_TIMEOUT_MS = 5000;
const STOP_TIMEOUT_MS = 3000;
// stop() during speech: the capture worker waits at most STOP_TAIL_MAX_MS of audio for the word in progress to end.
// If it has not answered after this long (no audio arriving, e.g. a suspended audio context), the source is ended anyway.
const STOP_TAIL_WAIT_MS = STOP_TAIL_MAX_MS + 1000;
// A cancelled recognition stops at whisper.cpp's next check (after the encoder pass, after each decoder step). The
// encoder of a long utterance takes many seconds in WebAssembly, so after this grace period the worker is
// terminated instead: that stops the CPU work and releases the memory at once.
const ABORT_GRACE_MS = 1000;
// Without SharedArrayBuffer a busy recogniser cannot be interrupted; it is terminated if it does not answer in time.
const NO_SAB_CLOSE_TIMEOUT_MS = 300;
// A worker that has not answered 'ready' after this long is taken as one that cannot start (a module whose threads
// cannot be loaded waits for them for ever, a fetch may never end): it is terminated and its start-up fails, so that
// no call waits for it without end. The recogniser's own start includes trying WebGPU (its watchdog: 30 s) and then
// starting the CPU module.
const STARTUP_TIMEOUT_MS = 60000;
// stop(): after the last transcript the loop waits at most this long for the speakers of the utterances that have not got
// theirs yet; what does not come is not delivered and is counted in state.speakers.lost.
const SPEAKERS_DRAIN_MS = 10000;

function defaultAssets(base) {
  if (base) {
    const u = (p) => new URL(p, base).href;
    return {
      mt: u('wasm/czechspeech-mt.js'), mtWasm: u('wasm/czechspeech-mt.wasm'),
      st: u('wasm/czechspeech-st.js'), stWasm: u('wasm/czechspeech-st.wasm'),
      gpu: u('wasm/czechspeech-webgpu.js'), gpuWasm: u('wasm/czechspeech-webgpu.wasm'),
      vad: u('assets/ggml-silero-v6.2.0.bin'), worklet: u('worklet.js'), worker: u('worker.js'),
    };
  }
  // Literal `new URL('./x', import.meta.url)` patterns, so that bundlers can see and copy the files.
  return {
    mt: new URL('./wasm/czechspeech-mt.js', import.meta.url).href,
    mtWasm: new URL('./wasm/czechspeech-mt.wasm', import.meta.url).href,
    st: new URL('./wasm/czechspeech-st.js', import.meta.url).href,
    stWasm: new URL('./wasm/czechspeech-st.wasm', import.meta.url).href,
    gpu: new URL('./wasm/czechspeech-webgpu.js', import.meta.url).href,
    gpuWasm: new URL('./wasm/czechspeech-webgpu.wasm', import.meta.url).href,
    vad: new URL('./assets/ggml-silero-v6.2.0.bin', import.meta.url).href,
    worklet: new URL('./worklet.js', import.meta.url).href,
    worker: null,
  };
}

/** A capture or recogniser worker, named after its role for the browser's developer tools. */
function createBrowserWorker(role, assets) {
  if (assets.worker) return new Worker(assets.worker, { type: 'module', name: `czechspeech-${role}` });
  // Literal URL and options, one construction per role: bundlers (Vite, webpack) find the worker only in exactly this
  // form, and Vite stops the build when the options are not a literal object.
  if (role === 'capture') return new Worker(new URL('./worker.js', import.meta.url), { type: 'module', name: 'czechspeech-capture' });
  return new Worker(new URL('./worker.js', import.meta.url), { type: 'module', name: 'czechspeech-recognizer' });
}

/** `err` itself when it is a CzechSpeechError, else one with code 'runtime' and `err` as its cause. */
function asCzechSpeechError(err, what) {
  if (err instanceof CzechSpeechError) return err;
  return new CzechSpeechError(`${what}: ${err && err.message ? err.message : String(err)}`, { code: 'runtime', cause: err });
}

/** A worker that could not be constructed or sent its init message, as a CzechSpeechError (the original as cause). */
function workerStartError(role, err) {
  // Browsers construct module workers from the page's origin only: a cross-origin assetBaseUrl is a SecurityError.
  const hint = err && err.name === 'SecurityError' ? ' (workers must be served from the page\'s origin; see assetBaseUrl)' : '';
  return asCzechSpeechError(err, `Could not start the ${role} worker${hint}`);
}

const defaultLog = (level, message) => {
  if (level === 'warn') console.warn(`CzechSpeech: ${message}`);
  else console.debug(`CzechSpeech: ${message}`);
};
// The default logger prints the backend fallback once with console.info (the rest of 'info' goes to console.debug).
const defaultNotice = (message) => console.info(`CzechSpeech: ${message}`);

/** A worker with a `ready` handshake, message routing and waits for single replies. */
class Remote {
  /** `startupMs`: how long it may take to answer 'ready' (0: no limit). */
  constructor(worker, role, onMessage, startupMs = 0) {
    this.worker = worker;
    this.role = role;
    this.onMessage = onMessage;
    this.terminated = false;
    this.started = false; // its 'ready' came
    this.failed = false;  // it failed or was terminated before that
    this.waits = [];
    this.startupTimer = null;
    this.ready = new Promise((resolve, reject) => {
      this.resolveReady = (m) => {
        if (this.failed) return;
        this.started = true;
        this.#clearStartupTimer();
        resolve(m);
      };
      this.rejectReady = (err) => {
        if (this.started) return;
        this.failed = true;
        this.#clearStartupTimer();
        reject(err);
      };
    });
    this.ready.catch(() => {});
    if (startupMs > 0) this.startupTimer = setTimeout(() => this.#startupTimedOut(startupMs), startupMs);
    worker.onmessage = (e) => this.#dispatch(e.data);
    worker.onerror = (e) => {
      if (e && typeof e.preventDefault === 'function') e.preventDefault();
      const message = (e && (e.message || (e.error && e.error.message))) || 'worker error';
      this.#dispatch({ type: 'fatal', code: 'runtime', message: `${role} worker: ${message}` });
    };
    worker.onmessageerror = () => this.#dispatch({ type: 'fatal', code: 'runtime', message: `${role} worker: message error` });
  }

  #clearStartupTimer() {
    if (this.startupTimer === null) return;
    clearTimeout(this.startupTimer);
    this.startupTimer = null;
  }

  /** No 'ready' in time: the start-up fails with its own error (before terminate() would give a generic one). */
  #startupTimedOut(ms) {
    this.startupTimer = null;
    const who = this.role === 'recognizer' ? 'The recogniser' : `The ${this.role} worker`;
    const cause = new DOMException(`no 'ready' from the ${this.role} worker in ${ms} ms`, 'TimeoutError');
    this.rejectReady(new CzechSpeechError(`${who} did not start within ${ms / 1000} s`, { code: 'runtime', cause }));
    this.terminate();
  }

  #dispatch(m) {
    if (!m || this.terminated) return;
    if (m.type === 'ready') this.resolveReady(m);
    if (m.type === 'fatal') {
      this.rejectReady(new CzechSpeechError(m.message, { code: m.code }));
      for (const w of this.waits.splice(0)) w.reject(new CzechSpeechError(m.message, { code: m.code }));
    }
    for (const w of [...this.waits]) {
      if (w.match(m)) {
        this.waits.splice(this.waits.indexOf(w), 1);
        w.resolve(m);
      }
    }
    this.onMessage(m, this);
  }

  post(msg, transfer) {
    if (this.terminated) throw new CzechSpeechError(`${this.role} worker is gone`, { code: 'runtime' });
    this.worker.postMessage(msg, transfer || []);
  }

  /** Next message matching `match` (rejects on fatal, termination or timeout). */
  waitFor(match, timeoutMs) {
    return new Promise((resolve, reject) => {
      const w = { match, resolve, reject };
      if (timeoutMs) {
        const timer = setTimeout(() => {
          const i = this.waits.indexOf(w);
          if (i >= 0) this.waits.splice(i, 1);
          reject(new CzechSpeechError(`${this.role} worker did not answer`, { code: 'runtime' }));
        }, timeoutMs);
        w.resolve = (m) => { clearTimeout(timer); resolve(m); };
        w.reject = (e) => { clearTimeout(timer); reject(e); };
      }
      this.waits.push(w);
    });
  }

  terminate() {
    if (this.terminated) return;
    this.terminated = true;
    this.#clearStartupTimer();
    this.worker.terminate();
    // nothing comes any more; whoever still holds the Worker object does not keep this one alive
    this.worker.onmessage = this.worker.onerror = this.worker.onmessageerror = null;
    const err = new CzechSpeechError(`${this.role} worker was terminated`, { code: 'runtime' });
    this.rejectReady(err);
    for (const w of this.waits.splice(0)) w.reject(err);
  }
}

/** Offline Czech speech-to-text in the browser. Create with `await CzechSpeech.create({ model })`. */
export class CzechSpeech {
  #model;
  #config;
  #options;
  #assets;
  #log;
  #vadBytes;
  #variant;
  #threads;
  #cancel;            // Int32Array over a SharedArrayBuffer, or null (no cross-origin isolation)
  #nextToken = 0;
  #closed = false;
  #closedError = null; // what close() rejects pending work with
  #closing = null;     // the promise of the first close(), which every close() returns
  #session = null;
  #listenCount = 0;    // listen() calls so far: the first part of every utteranceId ('<listen call>.<n>')
  #speakerState = null; // state.speakers: of the last listen() with speakers
  #recognizer = null;  // Remote
  #recognizerUsers = 0;
  #inFlight = 0;       // transcribe() requests sent to the recogniser and not answered yet
  #recognizerBusyFlag = false; // the recogniser's own report: a recognition is running
  #recognizerClosing = null;
  #pending = new Map(); // transcribe() id -> { rec, resolve, reject }: posted to `rec`, not answered yet
  #notice;             // the backend fallback: once per instance
  #noticed = false;
  #gpuBlocked = null;  // why this instance stopped trying WebGPU ('auto' after a WebGPU failure), or null
  // 'auto' after a WebGPU failure: the failures in a row and when WebGPU may be tried again, across recognisers
  #gpuRecovery;
  #gpuFailure = null;  // the reason of the last WebGPU failure
  #stats = { modelLoaded: false, modelLoads: 0, abortedRecognitions: 0, cappedRecognitions: 0, lastLoadMs: null,
    heapBytes: 0, peakHeapBytes: 0,
    backend: null, gpuAdapter: null, backendFallback: null, gpuBytes: 0, peakGpuBytes: 0 };

  /**
   * @param {object} options
   * @param {CzechSpeechModel} options.model   from CzechSpeechModel.download() / fromUrl() / fromBytes()
   * @param {object} [options.config]         speech detection settings (see DEFAULT_CONFIG)
   * @param {boolean} [options.keepModelLoaded] keep the model in memory between sessions and transcribe() calls
   *        until close() (default false: loaded per session/call and freed afterwards, like the Android SDK)
   * @param {((level: 'info'|'warn', message: string) => void)|false} [options.log] never receives transcript text
   * @param {string|URL} [options.assetBaseUrl] where worker.js, worklet.js, wasm/ and assets/ are served, when a
   *        bundler does not resolve them next to this module
   */
  static async create(options = {}) {
    if (!options || !options.model) throw new TypeError('CzechSpeech.create({ model }) needs a model');
    const speech = new CzechSpeech(options, CzechSpeech.#token);
    await speech.#init();
    return speech;
  }

  static #token = Symbol('CzechSpeech');

  constructor(options, token) {
    if (token !== CzechSpeech.#token) throw new TypeError('use CzechSpeech.create()');
    const model = options.model;
    if (!(model instanceof CzechSpeechModel)) throw new TypeError('model must be a CzechSpeechModel');
    this.#model = model;
    this.#config = resolveConfig(options.config);
    this.#options = options;
    this.#log = options.log === false ? () => {} : typeof options.log === 'function' ? options.log : defaultLog;
    this.#notice = options.log === false ? () => {} : typeof options.log === 'function'
      ? (message) => options.log('info', message) : defaultNotice;
    this.#assets = { ...defaultAssets(options.assetBaseUrl), ...(options._assets || {}) };
    const g = globalThis;
    const sab = !options._noSharedMemory && typeof g.SharedArrayBuffer === 'function' && g.crossOriginIsolated !== false;
    this.#variant = options._variant || (sab ? 'mt' : 'st');
    this.#cancel = sab ? new Int32Array(new g.SharedArrayBuffer(CANCEL_SLOTS * 4)) : null;
    this.#threads = threadCount(this.#config, this.#variant === 'mt', g.navigator && g.navigator.hardwareConcurrency);
    const t = options._gpuTimeouts;
    this.#gpuRecovery = new GpuRecovery({ delaysMs: (t && t.gpuRetryMs) || GPU_RETRY_DELAYS_MS,
      now: typeof options._now === 'function' ? options._now : () => performance.now() });
  }

  async #init() {
    // Everything runs in workers: without them (server-side rendering, Node.js) nothing can work, so fail at once.
    if (!this.#options._createWorker && typeof Worker !== 'function') {
      throw new CzechSpeechError('Web Workers are not available here (CzechSpeech runs in a browser page)', { code: 'unsupported' });
    }
    if (this.#options._vadModel) {
      this.#vadBytes = new Uint8Array(this.#options._vadModel).slice();
    } else {
      let res;
      try {
        res = await fetch(this.#assets.vad);
        if (res.ok) this.#vadBytes = new Uint8Array(await res.arrayBuffer());
      } catch (err) {
        throw new CzechSpeechError(`Could not fetch the Silero VAD model: ${err && err.message}`, { code: 'runtime', cause: err });
      }
      if (!res.ok) throw new CzechSpeechError(`Could not fetch the Silero VAD model (HTTP ${res.status})`, { code: 'runtime' });
    }
    if (sha256Hex(this.#vadBytes) !== VAD_SHA256) {
      throw new CzechSpeechError('The Silero VAD model file is corrupt (SHA-256 mismatch)', { code: 'runtime' });
    }
    const cpu = this.#variant === 'mt' ? `multi-threaded WebAssembly, ${this.#threads} threads`
      : 'single-threaded WebAssembly (the page is not cross-origin isolated)';
    this.#log('info', `ready: backend ${this.#config.backend}, CPU: ${cpu}`);
  }

  /** The speech detection settings in use (defaults filled in). */
  get config() {
    return this.#config;
  }

  /**
   * Diagnostics: { listening, modelLoaded, modelLoads, abortedRecognitions, cappedRecognitions, backend, gpuAdapter,
   * backendFallback,
   * runtime, threads, lastLoadMs, heapBytes, peakHeapBytes, gpuBytes, peakGpuBytes, closed }. No content.
   * backend is null until the recogniser has started once (first listen() or transcribe()).
   */
  get state() {
    return {
      listening: !!this.#session,
      modelLoaded: this.#stats.modelLoaded,
      modelLoads: this.#stats.modelLoads,
      abortedRecognitions: this.#stats.abortedRecognitions,
      cappedRecognitions: this.#stats.cappedRecognitions,
      backend: this.#stats.backend,
      gpuAdapter: this.#stats.gpuAdapter,
      backendFallback: this.#stats.backendFallback,
      runtime: this.#variant === 'mt' ? 'wasm-threads' : 'wasm-single',
      threads: this.#threads,
      lastLoadMs: this.#stats.lastLoadMs,
      heapBytes: this.#stats.heapBytes,
      peakHeapBytes: this.#stats.peakHeapBytes,
      gpuBytes: this.#stats.gpuBytes,
      peakGpuBytes: this.#stats.peakGpuBytes,
      speakers: this.#speakerState ? { ...this.#speakerState } : null,
      closed: this.#closed,
    };
  }

  /**
   * Listens to the microphone and yields speech events and transcripts, one transcript per utterance.
   *
   * The iterable is cold: iterating it opens the microphone and loads the model in parallel. To finish, call
   * stop(): the last utterance is still recognised and the loop ends normally. Leaving the loop (`break`,
   * `return`), aborting `signal` or calling close() instead closes the microphone at once, discards speech that
   * has not been recognised yet, stops a recognition in progress and frees the model; the loop's exit (or the
   * thrown abort reason) waits for that. One session at a time per instance.
   *
   * @param {{ signal?: AbortSignal, audio?: MediaTrackConstraints, stream?: MediaStream, language?: string }} [options]
   *   audio: getUserMedia constraints merged over the dictation defaults; stream: use this MediaStream instead of
   *   opening the microphone (it is not stopped at the end); language: force one of the model's languages (default:
   *   all of them, and the recogniser picks per utterance); any other code is a RangeError, thrown by the first next().
   * @returns {AsyncGenerator}
   */
  listen(options = {}) {
    return this.#run(options);
  }

  /**
   * Finishes the active listen() session. It finishes the word in progress: with speech in progress the microphone
   * stays on until a short pause (STOP_TAIL_SILENCE_MS, 256 ms), at most STOP_TAIL_MAX_MS (1 s); without speech it
   * stops delivering audio at once. The utterance in progress then ends as if silence followed, everything still
   * waiting is recognised and yielded, the loop ends normally and the model is freed. Returns immediately. Does
   * nothing without an active session (it never stops a later one). To discard instead, leave the loop, abort the
   * signal or call close().
   */
  stop() {
    if (this.#session) this.#session.finish();
  }

  /** Only the transcript events of listen(). */
  async *transcripts(options = {}) {
    for await (const ev of this.listen(options)) {
      if (ev.type === 'transcript') yield ev;
    }
  }

  /** The `languages` argument of the engine: the forced language, else all of the model's, comma-separated. */
  #languagesFor(language) {
    const languages = this.#model.languages;
    if (language === undefined || language === null) return languages.join(',');
    if (typeof language !== 'string' || !languages.includes(language)) {
      throw new RangeError(`language must be one of the model's languages (${languages.join(', ')})`);
    }
    return language;
  }

  async *#run({ signal, audio, stream, language, speakers, _source } = {}) {
    this.#ensureOpen();
    const languages = this.#languagesFor(language);
    if (speakers !== undefined && speakers !== null && !(speakers instanceof SpeakerSession)) {
      throw new TypeError('speakers must be a SpeakerSession (SpeakerSession.create())');
    }
    if (this.#session) {
      throw new CzechSpeechError('listen() is already being iterated; one session at a time', { code: 'busy' });
    }
    if (signal) signal.throwIfAborted();
    const session = new Session(this.#hooks(), this.#config, { listenId: ++this.#listenCount, speakers: speakers || null });
    this.#session = session;
    const onAbort = () => session.cancel(signal.reason);
    if (signal) signal.addEventListener('abort', onAbort, { once: true });
    try {
      session.start({ audio, stream, source: _source, languages });
      for (;;) {
        const ev = await session.next();
        if (ev === null) return;
        yield ev;
      }
    } finally {
      if (signal) signal.removeEventListener('abort', onAbort);
      await session.stop();
      if (this.#session === session) this.#session = null;
    }
  }

  /**
   * Transcribes audio the app already has: 16 kHz mono samples in [-1, 1]. Returns '' when there is no speech; the
   * model is not even loaded then. The array is copied and never modified; zero it yourself when you are done.
   *
   * @param {Float32Array} samples
   * @param {{ signal?: AbortSignal, language?: string }} [options] language: force one of the model's languages
   *   (default: all of them); any other code is a RangeError
   */
  async transcribe(samples, { signal, language } = {}) {
    this.#ensureOpen();
    if (!(samples instanceof Float32Array)) throw new TypeError('samples must be a Float32Array (16 kHz mono)');
    const languages = this.#languagesFor(language);
    if (samples.length === 0) return '';
    if (signal) signal.throwIfAborted();
    const copy = samples.slice();
    let rec = null;
    let token = 0;
    let onAbort = null;
    let abandoned = false;
    try {
      // an abort during the recogniser's start-up rejects at once (the start-up is not waited for)
      rec = await this.#acquireRecognizer(signal);
      // close() may have begun between the lease and this continuation: then nothing is sent (the lease goes below)
      if (this.#closed) throw this.#closedError;
      token = ++this.#nextToken;
      const id = token;
      const result = new Promise((resolve, reject) => this.#pending.set(id, { rec, resolve, reject }));
      try {
        rec.post({ type: 'transcribe', id, token, samples: copy, languages }, [copy.buffer]);
      } catch (err) {
        this.#pending.delete(id);
        throw asCzechSpeechError(err, 'Could not send the audio to the recogniser');
      }
      this.#inFlight++;
      const aborted = signal ? new Promise((_, reject) => {
        onAbort = () => {
          this.#cancelToken(token);
          abandoned = true;
          // the caller is answered now; a late reply finds nothing (and the recogniser may never send one)
          this.#pending.delete(id);
          reject(signal.reason);
        };
        signal.addEventListener('abort', onAbort, { once: true });
      }) : null;
      const r = await (aborted ? Promise.race([result, aborted]) : result);
      if (r.error) throw new CzechSpeechError(r.error, { code: errorCode(r.code, 'recognition') });
      return r.text;
    } finally {
      if (onAbort) signal.removeEventListener('abort', onAbort);
      if (copy.byteLength > 0) copy.fill(0); // never transferred
      if (rec) {
        // An aborted call returns at once; the recogniser is stopped and freed in the background.
        const release = this.#releaseRecognizer();
        if (abandoned) release.catch(() => {});
        else await release;
      }
    }
  }

  /**
   * Cancels an active listen() session and transcribe() calls, frees everything and rejects further calls. Resolves
   * once every worker is terminated; a recogniser that is still starting is terminated without waiting for it.
   * Every call resolves with the first one, when that close has finished.
   */
  close() {
    if (!this.#closing) this.#closing = this.#close();
    return this.#closing;
  }

  async #close() {
    this.#closed = true;
    const err = new CzechSpeechError('CzechSpeech was closed', { code: 'closed' });
    this.#closedError = err;
    const session = this.#session;
    if (session) session.cancel(err);
    for (const p of this.#pending.values()) p.reject(err);
    this.#pending.clear();
    if (this.#cancel) for (let t = Math.max(1, this.#nextToken - CANCEL_SLOTS + 1); t <= this.#nextToken; t++) this.#cancelToken(t);
    // A recogniser still starting is not waited for: its start-up is cancelled, and whoever waits for it gets `err`.
    const starting = this.#recognizer;
    if (starting && !starting.started) this.#discardRecognizer(starting);
    if (session) await session.stop();
    // Then the recogniser goes, whoever still holds it (every call holding it has been rejected above).
    await this.#maybeFreeRecognizer(true);
  }

  #ensureOpen() {
    if (this.#closed) throw new CzechSpeechError('CzechSpeech is closed', { code: 'closed' });
  }

  #cancelToken(token) {
    if (this.#cancel && token > 0) Atomics.store(this.#cancel, token % CANCEL_SLOTS, token);
  }

  /** A started worker (init message sent), or a CzechSpeechError; nothing is left running when it throws. */
  #newWorker(role, onMessage) {
    const create = this.#options._createWorker || ((r) => createBrowserWorker(r, this.#assets));
    let remote = null;
    try {
      remote = new Remote(create(role), role, onMessage, this.#startupTimeoutMs());
      remote.post(this.#initMessage(role));
    } catch (err) {
      if (remote) remote.terminate();
      throw workerStartError(role, err);
    }
    return remote;
  }

  #initMessage(role) {
    const rec = role === 'recognizer';
    const mt = rec && this.#variant === 'mt';
    return {
      type: 'init',
      role,
      moduleUrl: mt ? this.#assets.mt : this.#assets.st,
      wasmUrl: mt ? this.#assets.mtWasm : this.#assets.stWasm,
      // the recogniser's backend (src/backend.js); the capture worker always uses the single-threaded CPU module
      backend: rec ? this.#config.backend : 'cpu',
      gpuModuleUrl: rec ? this.#assets.gpu || null : null,
      gpuWasmUrl: rec ? this.#assets.gpuWasm || null : null,
      gpuBlocked: rec ? this.#gpuBlocked : null,
      // 'auto': the instance's recovery state after WebGPU failures (src/backend.js GpuRecovery)
      gpuRecovery: rec && this.#config.backend === 'auto' ? this.#gpuRecovery.snapshot() : null,
      gpuFailure: rec ? this.#gpuFailure : null,
      timeouts: rec ? this.#options._gpuTimeouts || null : null,
      threads: rec ? this.#threads : 1,
      vadModel: this.#vadBytes.slice().buffer,
      config: this.#config,
      model: role === 'recognizer' ? this.#model._workerSource() : null,
      abort: role === 'recognizer' && this.#cancel ? this.#cancel.buffer : null,
      engineOptions: this.#options._engineOptions || null,
    };
  }

  // ---- recogniser lease: created on first use, freed when the last user releases it (unless keepModelLoaded) ----

  /**
   * A lease on the recogniser, which is started when there is none. Resolves once it is ready. Rejects with code
   * 'closed' when close() comes first, even when the start-up has just finished, and with `signal.reason` as soon as
   * `signal` aborts: the lease goes at once, and a recogniser whose start-up nobody waits for any more is terminated.
   */
  async #acquireRecognizer(signal) {
    if (this.#recognizerClosing) await abortable(this.#recognizerClosing, signal);
    this.#ensureOpen();
    if (signal) signal.throwIfAborted();
    // a worker that cannot be started throws here, before a lease is taken
    if (!this.#recognizer) this.#recognizer = this.#newWorker('recognizer', (m, r) => this.#onRecognizerMessage(m, r));
    const rec = this.#recognizer;
    this.#recognizerUsers++;
    try {
      await abortable(rec.ready, signal);
    } catch (err) {
      this.#recognizerUsers--;
      if (this.#recognizer === rec) {
        if (rec.failed || (!rec.started && this.#recognizerUsers === 0)) this.#discardRecognizer(rec);
        else this.#maybeFreeRecognizer(false); // started just now, and nobody holds it
      }
      throw this.#closed ? this.#closedError : err;
    }
    if (this.#closed) {
      // close() came while the start-up finished; close() frees the recogniser
      this.#recognizerUsers--;
      throw this.#closedError;
    }
    return rec;
  }

  async #releaseRecognizer() {
    this.#recognizerUsers = Math.max(0, this.#recognizerUsers - 1);
    await this.#maybeFreeRecognizer(false);
  }

  /**
   * Frees the recogniser when nobody holds it any more (unless keepModelLoaded), or at once with `force` (close()).
   * Resolves when no recogniser is being freed.
   */
  async #maybeFreeRecognizer(force) {
    const rec = this.#recognizer;
    const unused = this.#recognizerUsers === 0 && (!this.#options.keepModelLoaded || this.#closed);
    if (rec && (force || unused)) {
      this.#recognizer = null;
      const done = this.#freeRecognizer(rec);
      this.#recognizerClosing = done;
      done.then(() => { if (this.#recognizerClosing === done) this.#recognizerClosing = null; });
    }
    await this.#recognizerClosing;
  }

  /** Lets the recogniser free the model and zero its memory, then terminates it. Never rejects. */
  async #freeRecognizer(rec) {
    if (rec.started && !rec.terminated && this.#busy() && this.#cancel) {
      // a cancelled recognition may still be running: give whisper.cpp until its next abort check
      const until = Date.now() + this.#abortGraceMs();
      while (this.#busy() && !rec.terminated && Date.now() < until) {
        await new Promise((r) => setTimeout(r, 10));
      }
    }
    // A recogniser that has not started yet has nothing loaded: it is terminated without waiting for it.
    if (rec.started && !rec.terminated && !this.#busy()) {
      try {
        const freed = rec.waitFor((m) => m.type === 'freed', FREE_TIMEOUT_MS);
        rec.post({ type: 'free' });
        await freed;
      } catch {
        // terminated below
      }
    }
    this.#discardRecognizer(rec);
  }

  /**
   * Terminates the recogniser at once (no free, no wait): a failed start-up, a cancelled one, the end of freeing, a
   * fatal error. transcribe() calls it has not answered fail with `err`: nothing will answer them now.
   */
  #discardRecognizer(rec, err = new CzechSpeechError('The recogniser was terminated', { code: 'runtime' })) {
    if (this.#recognizer === rec) this.#recognizer = null;
    rec.terminate();
    for (const [id, p] of this.#pending) {
      if (p.rec !== rec) continue;
      this.#pending.delete(id);
      p.reject(err);
    }
    this.#inFlight = 0;
    this.#recognizerBusyFlag = false;
    this.#stats.modelLoaded = false;
  }

  #startupTimeoutMs() {
    const t = this.#options._timeouts;
    return t && t.startup !== undefined ? t.startup : STARTUP_TIMEOUT_MS;
  }

  #abortGraceMs() {
    return this.#options._abortGraceMs ?? ABORT_GRACE_MS;
  }

  #onRecognizerMessage(m, rec) {
    switch (m.type) {
      case 'log': return this.#log(m.level, m.message);
      case 'ready':
        this.#onBackend(m, true);
        return this.#heap(m.heapBytes);
      case 'backend': return this.#onBackend(m, false);
      case 'modelLoaded':
        this.#stats.modelLoaded = true;
        this.#stats.modelLoads++;
        this.#stats.lastLoadMs = m.loadMs;
        this.#gpu(m.gpuBytes);
        return this.#heap(m.heapBytes);
      case 'freed': return this.#heap(m.heapBytes);
      case 'checked': return;
      case 'aborted':
        this.#stats.abortedRecognitions++;
        return;
      case 'capped':
        this.#stats.cappedRecognitions++;
        return;
      case 'gpuRecovery':
        return this.#onGpuRecovery(m);
      case 'busy':
        this.#recognizerBusyFlag = !!m.busy;
        return this.#heap(m.heapBytes);
      case 'transcribed': {
        this.#inFlight = Math.max(0, this.#inFlight - 1);
        const p = this.#pending.get(m.id);
        if (p) {
          this.#pending.delete(m.id);
          p.resolve(m);
        }
        return;
      }
      case 'result':
        if (this.#session && this.#session.id === m.session) this.#session.onResult(m);
        return;
      case 'listenClosed':
        if (this.#session && this.#session.id === m.session) this.#session.onListenClosed();
        return;
      case 'fatal': {
        const err = new CzechSpeechError(m.message, { code: m.code });
        if (this.#session) this.#session.fail(err);
        this.#discardRecognizer(rec, err);
        return;
      }
      default:
        if (this.#options._onWorkerMessage) this.#options._onWorkerMessage('recognizer', m);
    }
  }

  #busy() {
    return this.#inFlight > 0 || this.#recognizerBusyFlag;
  }

  #heap(bytes) {
    if (typeof bytes !== 'number') return;
    this.#stats.heapBytes = bytes;
    this.#stats.peakHeapBytes = Math.max(this.#stats.peakHeapBytes, bytes);
  }

  #gpu(bytes) {
    if (typeof bytes !== 'number') return;
    this.#stats.gpuBytes = bytes;
    this.#stats.peakGpuBytes = Math.max(this.#stats.peakGpuBytes, bytes);
  }

  /** The recogniser's backend, from its 'ready' message or a later switch ('backend'). */
  #onBackend(m, starting) {
    if (!m.backend) return;
    const requested = this.#config.backend;
    const before = this.#stats.backend;
    this.#stats.backend = m.backend;
    if (this.#session) this.#session.onBackend(m.backend);
    this.#stats.gpuAdapter = m.gpuAdapter || null;
    this.#stats.backendFallback = m.backendFallback || null;
    if (m.backend !== 'webgpu') this.#stats.gpuBytes = 0;
    // 'auto' stops trying WebGPU in this instance when the WebGPU module did not start at all the first time (it is
    // several MB to start); a missing adapter is checked again by the next recogniser, which costs a few milliseconds.
    // After a WebGPU failure the recovery state decides (#onGpuRecovery).
    if (requested === 'auto' && m.gpuModuleFailed && this.#gpuRecovery.failures === 0) {
      this.#gpuBlocked = m.backendFallback || m.reason || 'WebGPU failed';
    }
    const cpu = this.#variant === 'mt' ? `WebAssembly, ${this.#threads} threads` : 'WebAssembly, single-threaded';
    if (m.backend === 'webgpu') {
      this.#log('info', `recogniser on WebGPU (${adapterLabel(m.gpuAdapter)})`);
    } else if (starting && requested === 'auto' && m.backendFallback) {
      const line = `WebGPU not used (${m.backendFallback}); recognising on the CPU (${cpu})`;
      if (!this.#noticed) this.#notice(line);
      else this.#log('info', line);
      this.#noticed = true;
    } else if (!starting && m.gpuFailed && before !== 'cpu') {
      this.#log('info', `recogniser on the CPU now (${cpu})`);
    }
  }

  /**
   * 'auto': a recogniser reports the recovery state after a WebGPU failure, a failed restart or a success on WebGPU. The
   * instance keeps it for its next recognisers: one started before the retry time starts on the CPU at once, and after
   * too many failures in a row none tries WebGPU.
   */
  #onGpuRecovery(m) {
    if (this.#config.backend !== 'auto') return;
    this.#gpuRecovery.adopt(m);
    if (m.reason) this.#gpuFailure = m.reason;
    if (this.#gpuRecovery.failures === 0) this.#gpuFailure = null;
    if (this.#gpuRecovery.gaveUp) this.#gpuBlocked = this.#gpuFailure || 'WebGPU failed';
  }

  /** What a Session needs from its CzechSpeech (keeps the private fields private). */
  #hooks() {
    return {
      log: (level, message) => this.#log(level, message),
      acquireRecognizer: (signal) => this.#acquireRecognizer(signal),
      releaseRecognizer: () => this.#releaseRecognizer(),
      newToken: () => ++this.#nextToken,
      cancelToken: (t) => this.#cancelToken(t),
      canInterrupt: () => !!this.#cancel,
      abortGraceMs: () => this.#abortGraceMs(),
      // internal _eagerLoad: true/false forces it; by default the model is loaded at session start on WebGPU only
      eagerLoad: () => (this.#options._eagerLoad !== undefined ? !!this.#options._eagerLoad : this.#stats.backend === 'webgpu'),
      recognizerBusy: () => this.#busy(),
      terminateRecognizer: (rec) => this.#discardRecognizer(rec,
        new CzechSpeechError('The recogniser was stopped together with the listen() session', { code: 'closed' })),
      newCapture: (onMessage) => this.#newWorker('capture', onMessage),
      openMicrophone: (o) => openMicrophone({ workletUrl: this.#assets.worklet, ...o }),
      onWorkerMessage: this.#options._onWorkerMessage || null,
      backend: () => this.#stats.backend,
      speakersDrainMs: () => this.#options._speakersDrainMs ?? SPEAKERS_DRAIN_MS,
      speakerState: (st) => { this.#speakerState = st; },
    };
  }
}

let sessionIds = 0;

/** One listen() session: microphone (or test source) + capture worker + a lease on the recogniser. */
class Session {
  constructor(hooks, config, { listenId = 0, speakers = null } = {}) {
    this.id = ++sessionIds;
    this.h = hooks;
    this.config = config;
    this.token = hooks.newToken();
    this.queue = [];
    this.waiter = null;
    this.error = null;
    this.done = false;
    this.stopping = null;
    this.source = null;
    this.capture = null;
    this.recognizer = null;
    this.recognizerAcquisition = null;
    this.startup = new AbortController(); // aborted when the session stops: its recogniser lease is not waited for
    this.listening = false;
    this.listenClosed = new Promise((resolve) => { this.onListenClosed = resolve; });
    this.lastSeq = 0;
    this.held = [];          // results waiting for the capture event they follow
    this.expected = null;    // utterance count once a finite source ended (or stop() finished the capture)
    this.results = 0;
    this.finishing = false;  // stop(): no audio after this, what was captured is still recognised
    this.abandoned = false;  // stop() came before the source was open: it was closed unused
    this.drainTimer = null;  // stop(): set while the capture worker lets the word in progress finish
    // speakers (listen({ speakers })): one speakers event follows each transcript, whatever order the two arrive in
    this.listenId = listenId;
    this.ss = speakers;      // the SpeakerSession, or null
    this.speakerAttached = false;
    this.speakerOff = null;  // null, 'cpu' or 'failed': new utterances are not sent to the speaker worker
    this.utts = new Map();   // utterance id of the capture worker -> what is known of its speakers (only with speakers)
    this.waiting = new Set(); // ids whose transcript is out and whose speakers event is not
    this.speakerStats = { unsplit: 0, skipped: 0, lost: 0 };
    this.speakersDrainTimer = null;
  }

  // ---- consumer side ----

  next() {
    if (this.queue.length > 0) return Promise.resolve(this.queue.shift());
    if (this.error) return Promise.reject(this.error);
    if (this.done) return Promise.resolve(null);
    return new Promise((resolve, reject) => { this.waiter = { resolve, reject }; });
  }

  #push(ev) {
    if (this.done || this.error) return;
    if (this.waiter) {
      const w = this.waiter;
      this.waiter = null;
      w.resolve(ev);
    } else {
      this.queue.push(ev);
    }
  }

  #finish() {
    if (this.done || this.error) return;
    this.done = true;
    this.#clearDrain();
    this.#clearSpeakersDrain();
    this.#reportSpeakers();
    if (this.waiter) {
      const w = this.waiter;
      this.waiter = null;
      w.resolve(null);
    }
  }

  /** Ends the session with an error (thrown from the iterator) and starts the cleanup at once. */
  fail(err) {
    if (this.error || this.done) {
      this.stop();
      return;
    }
    this.error = err;
    this.queue.length = 0;
    if (this.waiter) {
      const w = this.waiter;
      this.waiter = null;
      w.reject(err);
    }
    this.stop();
  }

  cancel(reason) {
    this.fail(reason);
  }

  /**
   * stop(): first the capture worker lets the word in progress finish (#drain); then the source delivers no audio
   * after this; the capture worker ends the utterance in progress as if silence followed and reports how many
   * utterances there were, the recogniser works through them, and the session then ends like one whose source
   * ended (#maybeComplete). Before the session is wired the source is finished right after wiring, without waiting
   * for a pause; before the source is open, the source is closed as soon as it opens.
   */
  finish() {
    if (this.finishing || this.stopping || this.done || this.error) return;
    this.finishing = true;
    this.h.log('info', 'listen: stop(), finishing the session');
    if (this.listening) this.#drain();
  }

  /**
   * The capture worker answers 'drained' at once without speech in progress, otherwise after a pause of
   * STOP_TAIL_SILENCE_MS or STOP_TAIL_MAX_MS of audio (src/worker-core.js); the microphone stays open until then.
   */
  #drain() {
    this.drainTimer = setTimeout(() => this.#drained(), STOP_TAIL_WAIT_MS);
    try {
      this.capture.post({ type: 'drain' });
    } catch {
      this.#drained();
    }
  }

  #drained() {
    if (this.drainTimer === null) return;
    this.#clearDrain();
    if (!this.stopping && !this.done && !this.error) this.#finishSource();
  }

  #clearDrain() {
    if (this.drainTimer === null) return;
    clearTimeout(this.drainTimer);
    this.drainTimer = null;
  }

  async #finishSource() {
    const s = this.source;
    let delivered;
    try {
      // the end marker travels on the audio port after the last frame, so no frame is lost
      delivered = typeof s.finish === 'function' ? !!(await s.finish()) : false;
    } catch {
      delivered = false;
    }
    if (!delivered && !this.stopping && this.capture) {
      // No acknowledged end marker (a source without finish(), or the worklet did not answer in time): tell the
      // capture worker directly. Frames still in flight on the audio port may be lost then.
      try { this.capture.post({ type: 'finish' }); } catch { /* the worker is gone */ }
      try { await s.close(); } catch { /* closed */ }
    }
  }

  // ---- start ----

  start({ audio, stream, source, languages }) {
    const t0 = performance.now();
    // The recogniser worker starts with the microphone, but the model is loaded in the same task as the first
    // recognition (deviation from the Android SDK, which loads it at once). When the 0.5 GB model was loaded in a
    // task of its own, V8's dynamic tier-up often left the whole first encoder pass in baseline (Liftoff) code,
    // 2.7x slower (26 s instead of 10 s for a 7 s utterance in Chrome and Node.js); loading it with the first
    // utterance gave the fast path every time. It also means a session without speech never loads the model.
    // A wrong file still fails at once: 'check' reads the file header.
    // On WebGPU the model is loaded at once instead: the encoder runs on the GPU, the first recognition is as fast
    // after an early load (5.7 s for a 7.6 s utterance either way, Intel Xe-LPG), and the transcript then comes
    // 1.2 to 2 s sooner (the load, incl. the upload to the GPU, happens while the person speaks).
    const rec = this.h.acquireRecognizer(this.startup.signal).then((r) => {
      this.recognizer = r;
      if (!this.stopping) r.post({ type: this.h.eagerLoad() ? 'load' : 'check' });
      return r;
    });
    this.recognizerAcquisition = rec.catch(() => null);
    // The capture worker is created now, not a task later: a close() or abort right after the first next() must find
    // it to terminate it.
    let cap;
    try {
      const c = this.h.newCapture((m) => this.#onCaptureMessage(m));
      this.capture = c;
      cap = c.ready.then(() => c);
    } catch (err) {
      cap = Promise.reject(err);
    }
    const src = Promise.resolve().then(() => (source ? source.open() : this.h.openMicrophone({ audio, stream })))
      .then((s) => {
        this.source = s;
        if (this.stopping) {
          s.close();
        } else if (this.finishing) {
          // stop() before the microphone was open: nothing was captured; close it and end the session
          this.abandoned = true;
          s.close();
          this.expected = 0;
          this.#maybeComplete();
        } else if (typeof s.onEnded === 'function') {
          s.onEnded((err) => this.fail(err));
        }
        return s;
      });
    Promise.all([rec, cap, src]).then(([r, c, s]) => {
      if (this.stopping || this.abandoned) return;
      const channel = new MessageChannel();
      const speakerPort = this.#attachSpeakers();
      const transfer = speakerPort ? [s.port, channel.port1, speakerPort] : [s.port, channel.port1];
      c.post({ type: 'start', audioPort: s.port, utterancePort: channel.port1, speakerPort, speakers: this.speakerOff === null }, transfer);
      r.post({ type: 'listen', session: this.id, token: this.token, port: channel.port2, languages }, [channel.port2]);
      this.listening = true;
      this.#reportSpeakers();
      const rate = s.sampleRate ? `, capture at ${s.sampleRate} Hz` : '';
      this.h.log('info', `listen: session started in ${Math.round(performance.now() - t0)} ms${rate}`);
      // stop() during start-up: the audio captured so far is still recognised
      if (this.finishing) this.#finishSource();
    }).catch((err) => this.fail(asCzechSpeechError(err, 'listen() could not start')));
  }

  // ---- worker messages ----

  #onCaptureMessage(m) {
    switch (m.type) {
      case 'log': return this.h.log(m.level, m.message);
      case 'event': {
        this.#push(m.event === 'speechStarted' ? new SpeechStartedEvent(m.atMs) : new SpeechEndedEvent(m.atMs));
        this.lastSeq = m.seq;
        this.#flushHeld();
        return;
      }
      case 'utterance':
        if (this.ss) this.#utt(m.id).speaker = !!m.speaker;
        return;
      case 'drained':
        return this.#drained();
      case 'ended':
        this.expected = m.utterances;
        this.#maybeComplete();
        return;
      case 'fatal':
        return this.fail(new CzechSpeechError(m.message, { code: m.code }));
      case 'ready':
      case 'stopped':
        return;
      default:
        if (this.h.onWorkerMessage) this.h.onWorkerMessage('capture', m);
    }
  }

  onResult(m) {
    this.results++;
    if (m.afterSeq <= this.lastSeq) this.#deliver(m);
    else this.held.push(m);
    this.#maybeComplete();
  }

  #flushHeld() {
    if (this.held.length === 0) return;
    const ready = this.held.filter((m) => m.afterSeq <= this.lastSeq);
    this.held = this.held.filter((m) => m.afterSeq > this.lastSeq);
    for (const m of ready) this.#deliver(m);
    this.#maybeComplete();
  }

  #deliver(m) {
    if (m.error) {
      this.h.log('warn', `utterance failed: ${m.error}`);
      this.#push(new SpeechErrorEvent(new CzechSpeechError(m.error, { code: 'recognition' })));
    } else if (m.text && !m.cancelled) {
      this.#push(new TranscriptEvent(m.text, m.startMs, m.endMs, m.processingMs, m.language, `${this.listenId}.${m.id}`));
      this.#afterTranscript(m.id);
      return;
    }
    this.#noTranscript(m.id);
  }

  // ---- speakers ----

  #utt(n) {
    let st = this.utts.get(n);
    if (!st) {
      st = { speaker: null, transcript: false, none: false, ev: null }; // speaker: did the capture worker send a copy
      this.utts.set(n, st);
    }
    return st;
  }

  /** The speaker port for the capture worker, or null (no speakers, or their worker is gone: transcripts go on). */
  #attachSpeakers() {
    if (!this.ss) return null;
    this.onBackend(this.h.backend());
    try {
      const port = this.ss._attach(this.listenId, (ev, info) => this.#onSpeakers(ev, info), (err) => this.#onSpeakerFailure(err));
      this.speakerAttached = true;
      return port;
    } catch (err) {
      if (err && err.code === 'closed') throw err; // a closed session is a mistake of the caller
      this.speakerOff = 'failed';                  // its worker died earlier: no speakers, the transcript goes on
      return null;
    }
  }

  /** The recogniser's backend (at start and when it changes): on the CPU the app may ask for no speakers (onCpu 'skip'). */
  onBackend(backend) {
    if (!this.ss || this.speakerOff === 'failed') return;
    const off = this.ss.onCpu === 'skip' && backend === 'cpu' ? 'cpu' : null;
    if (off === this.speakerOff) return;
    this.speakerOff = off;
    if (this.listening && this.capture) {
      try { this.capture.post({ type: 'speakers', enabled: off === null }); } catch { /* the worker is gone */ }
    }
    this.#reportSpeakers();
  }

  /** The speakers of one utterance, from the speaker worker: out at once when its transcript is, else held for it. */
  #onSpeakers(ev, info) {
    const n = Number(ev.utteranceId.slice(ev.utteranceId.indexOf('.') + 1));
    if (info && info.mode === 'whole') this.speakerStats.unsplit++;
    const st = this.#utt(n);
    if (st.none) {
      ev.discard();        // the utterance has no transcript (the speech gate, empty text, an error): it founds no speaker
      this.utts.delete(n);
    } else if (st.transcript) {
      this.#emitSpeakers(n, ev);
      this.#maybeComplete();
    } else {
      st.ev = ev;
    }
  }

  #afterTranscript(n) {
    if (!this.ss) return;
    const st = this.#utt(n);
    st.transcript = true;
    if (st.ev) return this.#emitSpeakers(n, st.ev);
    // not sent to the speaker worker (cpu, or its worker is gone), or it is gone since: nothing will come
    let skipped = null;
    if (st.speaker === false || (st.speaker === null && this.speakerOff !== null)) skipped = this.speakerOff ?? 'failed';
    else if (this.speakerOff === 'failed') skipped = 'failed';
    if (skipped) return this.#emitSpeakers(n, new SpeakersEvent(`${this.listenId}.${n}`, [], skipped));
    this.waiting.add(n);
    this.#reportSpeakers();
    return undefined;
  }

  #noTranscript(n) {
    if (!this.ss) return;
    const st = this.#utt(n);
    if (st.ev || st.speaker === false) {                    // held speakers are dropped (and zeroed); none are coming
      if (st.ev) st.ev.discard();
      this.utts.delete(n);
    }
    else st.none = true;                                      // they may still arrive: dropped then
  }

  #emitSpeakers(n, ev) {
    const out = ev instanceof SpeakersEvent ? ev : ev.commit(); // the pieces are tagged now, in the order of the transcripts
    this.#push(out);
    if (out.skipped) this.speakerStats.skipped++;
    this.utts.delete(n);
    this.waiting.delete(n);
    this.#reportSpeakers();
  }

  /** The speaker worker failed or the SpeakerSession was closed: nothing more comes from it. */
  #onSpeakerFailure() {
    if (this.speakerOff === 'failed') return;
    this.speakerOff = 'failed';
    if (this.listening && this.capture) {
      try { this.capture.post({ type: 'speakers', enabled: false }); } catch { /* the worker is gone */ }
    }
    for (const n of [...this.waiting]) this.#emitSpeakers(n, new SpeakersEvent(`${this.listenId}.${n}`, [], 'failed'));
    this.#maybeComplete();
  }

  #reportSpeakers() {
    if (!this.ss) return;
    this.h.speakerState({
      active: this.speakerAttached && !this.stopping && !this.done && !this.error && this.speakerOff === null,
      threads: this.ss.threads, pending: this.waiting.size, ...this.speakerStats,
    });
  }

  /** stop(): the transcripts are all out; the speakers of the last utterances are waited for, but not for ever. */
  #startSpeakersDrain() {
    if (this.speakersDrainTimer !== null) return;
    this.speakersDrainTimer = setTimeout(() => {
      this.speakersDrainTimer = null;
      this.speakerStats.lost += this.waiting.size;
      this.waiting.clear();
      this.#finish();
    }, this.h.speakersDrainMs());
  }

  #clearSpeakersDrain() {
    if (this.speakersDrainTimer === null) return;
    clearTimeout(this.speakersDrainTimer);
    this.speakersDrainTimer = null;
  }

  #maybeComplete() {
    if (this.expected !== null && this.results >= this.expected && this.held.length === 0) {
      if (this.waiting.size > 0) this.#startSpeakersDrain();
      else this.#finish();
    }
  }

  // ---- stop ----

  /** Closes the microphone, stops the capture worker and releases the recogniser. Idempotent. */
  stop() {
    if (!this.stopping) this.stopping = this.#stop();
    return this.stopping;
  }

  async #stop() {
    this.#clearDrain();
    // 1. The microphone closes at once.
    const closing = this.source ? Promise.resolve(this.source.close()).catch(() => {}) : null;
    // 2. Recognitions of this session stop at whisper.cpp's next check, and a recogniser still starting is not
    //    waited for (the session's lease goes at once; the recogniser is terminated if nobody else waits for it).
    this.h.cancelToken(this.token);
    this.startup.abort(new CzechSpeechError('the listen() session ended', { code: 'closed' }));
    // 3. The capture worker zeroes its buffers and its heap, and tells the recogniser that no more utterances come.
    //    One still starting has received no audio yet: it is terminated without waiting for it.
    const cap = this.capture;
    if (cap && !cap.started) {
      cap.terminate();
    } else if (cap) {
      try {
        await withTimeout(cap.ready, STOP_TIMEOUT_MS);
        const stopped = cap.waitFor((m) => m.type === 'stopped', STOP_TIMEOUT_MS);
        cap.post({ type: 'stop' });
        await stopped;
      } catch {
        // terminated below
      }
      cap.terminate();
    }
    if (this.speakerAttached) {
      // the speaker worker drops (and zeroes) what this listen() still had queued there
      this.speakerAttached = false;
      try { this.ss._detach(this.listenId); } catch { /* closed */ }
    }
    this.#clearSpeakersDrain();
    for (const st of this.utts.values()) if (st.ev) st.ev.discard(); // never transcribed: nothing to tag
    this.utts.clear();
    this.#reportSpeakers();
    if (closing) await closing;
    // 4. The recogniser drops what is queued for this session (it reads the capture worker's last message, 'close',
    //    after every utterance), then the lease is released, which frees the model.
    const r = this.recognizerAcquisition ? await this.recognizerAcquisition : null;
    if (r) {
      if (this.listening && !r.terminated) {
        const interruptible = this.h.canInterrupt();
        if (!interruptible && this.h.recognizerBusy()) {
          // No SharedArrayBuffer: whisper.cpp cannot be told to stop, and a single-threaded recognition can take
          // longer than the utterance itself. Terminating the worker releases its memory at once.
          this.h.terminateRecognizer(r);
        } else {
          try {
            await withTimeout(this.listenClosed, interruptible ? this.h.abortGraceMs() : NO_SAB_CLOSE_TIMEOUT_MS);
          } catch {
            this.h.terminateRecognizer(r);
          }
        }
      }
      await this.h.releaseRecognizer();
    }
    // Whatever capture worker the session has by now goes too (terminate() does nothing the second time).
    if (this.capture) this.capture.terminate();
    this.h.log('info', 'listen: session ended');
  }
}

/** `promise`, or a rejection with `signal.reason` as soon as `signal` aborts (no signal: `promise`). */
function abortable(promise, signal) {
  if (!signal) return promise;
  if (signal.aborted) return Promise.reject(signal.reason);
  let onAbort;
  const aborted = new Promise((_, reject) => {
    onAbort = () => reject(signal.reason);
    signal.addEventListener('abort', onAbort, { once: true });
  });
  return Promise.race([promise, aborted]).finally(() => signal.removeEventListener('abort', onAbort));
}

function withTimeout(promise, ms) {
  let timer;
  return Promise.race([
    promise,
    new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('timeout')), ms); }),
  ]).finally(() => clearTimeout(timer));
}
