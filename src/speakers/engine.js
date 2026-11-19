// The page side of the speaker worker: starts it, sends it audio (by transfer, the caller loses the arrays) and routes
// its answers. One engine serves one model; close() zeroes what the worker holds and terminates it.
// Terminating the worker is the scrub of everything we cannot reach: the runtime's features, activations and heap, and the
// fbank buffers only go away with the worker, so close() (and a failure) always ends with terminate().
import { CzechSpeechError } from '../events.js';

const CLOSE_GRACE_MS = 3000;
// A worker that has not answered 'ready' after this long cannot start (a runtime whose threads cannot be loaded waits for
// them for ever, a read may never end): it is terminated and create() fails, as the recogniser's start-up does (src/speech.js).
const STARTUP_TIMEOUT_MS = 60000;
// The fbank front end needs one 25 ms frame (400 samples at 16 kHz); the model needs at least one frame.
const MIN_SAMPLES = 400;
// The change search embeds windows of splitWindowMs in one go, so the input cap may not be lower than that window.
// The windows are not cut to the cap; a cap below them is refused instead (the simpler of the two ways out).

function createBrowserWorker(assetBaseUrl) {
  if (assetBaseUrl) {
    return new Worker(new URL('speakers/worker.js', assetBaseUrl), { type: 'module', name: 'czechspeech-speaker' });
  }
  return new Worker(new URL('./worker.js', import.meta.url), { type: 'module', name: 'czechspeech-speaker' });
}

/** The input of embed() / embedUtterance(), checked here so that a bad call fails alone and never reaches the worker. */
function checkSamples(samples) {
  if (!(samples instanceof Float32Array) && Object.prototype.toString.call(samples) !== '[object Float32Array]') {
    throw new TypeError('audio must be a Float32Array (16 kHz mono)');
  }
  if (samples.buffer && typeof SharedArrayBuffer === 'function' && samples.buffer instanceof SharedArrayBuffer) {
    throw new TypeError('audio must not be backed by a SharedArrayBuffer (it cannot be transferred to the worker); copy it into a plain Float32Array');
  }
  if (samples.length < MIN_SAMPLES) {
    throw new RangeError(`audio must have at least ${MIN_SAMPLES} samples (${samples.length}; an array whose buffer was transferred away counts as empty)`);
  }
}

/** An array the caller hands over: a view of a larger buffer is copied (and the view zeroed) so only its bytes move. */
function transferable(samples) {
  if (samples.byteOffset === 0 && samples.buffer.byteLength === samples.byteLength) return samples;
  const copy = samples.slice();
  samples.fill(0);
  return copy;
}

export class SpeakerEngine {
  #worker;
  #closed = false;
  #threads = 1;
  #nextId = 1;
  #pending = new Map();      // id -> { resolve, reject } of embed()
  #pendingUtt = new Map();   // id -> { resolve, reject } of embedUtterance()
  #listeners = new Map();    // listenId (the app's) -> { channel, onPieces, onFailure, port }
  #channels = new Map();     // channel -> the same entry. The channel is the id the worker knows: never reused, so a
                             // listenId that is attached again cannot receive what the previous attachment had in flight
  #nextChannel = 1;
  #closing = null;
  #closeWaiter = null;
  #onWorkerMessage;

  constructor(token, worker, onWorkerMessage) {
    if (token !== SpeakerEngine.#token) throw new TypeError('use SpeakerEngine.create()');
    this.#worker = worker;
    this.#onWorkerMessage = onWorkerMessage || null;
  }

  static #token = Symbol('SpeakerEngine');

  /**
   * Starts the worker and loads the model into it. `model` is a VoiceprintModel, `threads` 1 or 2, `point` the operating
   * point of the model (src/speakers/points.js). `assetBaseUrl`: where speakers/worker.js is served from (default: next
   * to this module).
   */
  static async create({ model, threads = 1, point, assetBaseUrl, _createWorker, _fake, _onWorkerMessage, maxInputMs, _startupTimeoutMs = STARTUP_TIMEOUT_MS } = {}) {
    if (!model || typeof model._modelBlob !== 'function') throw new TypeError('SpeakerEngine.create() needs a VoiceprintModel');
    if (!point) throw new TypeError('SpeakerEngine.create() needs an operating point');
    if (!Number.isInteger(threads) || threads < 1 || threads > 2) throw new RangeError('threads must be 1 or 2');
    const minInputMs = Math.max(1000, point.splitWindowMs || 0);
    if (maxInputMs !== undefined && !(maxInputMs >= minInputMs)) {
      throw new RangeError(`maxInputMs must be at least ${minInputMs} (the window of the change search)`);
    }
    let worker;
    try {
      worker = _createWorker ? _createWorker('speaker') : createBrowserWorker(assetBaseUrl);
    } catch (err) {
      throw new CzechSpeechError(`The speaker worker could not start: ${err && err.message}`, { code: 'runtime', cause: err });
    }
    const engine = new SpeakerEngine(SpeakerEngine.#token, worker, _onWorkerMessage);
    await engine.#init({ model, threads, point, fake: _fake, timeoutMs: _startupTimeoutMs, maxInputMs });
    return engine;
  }

  /** Whatever goes wrong while starting (also a timeout), the worker is terminated before create() rejects. */
  async #init({ model, threads, point, fake, timeoutMs, maxInputMs }) {
    let timer;
    const timeout = new Promise((_, reject) => {
      timer = setTimeout(() => reject(new CzechSpeechError(
        `The speaker worker did not start within ${Math.round(timeoutMs / 1000)} s`, { code: 'runtime' })), timeoutMs);
    });
    try {
      await Promise.race([this.#start({ model, threads, point, fake, maxInputMs }), timeout]);
    } catch (err) {
      this.#failed = err;
      this.#terminate();
      throw err instanceof CzechSpeechError ? err
        : new CzechSpeechError(`The speaker worker could not start: ${(err && err.message) || err}`, { code: 'runtime', cause: err });
    } finally {
      clearTimeout(timer);
    }
  }

  async #start({ model, threads, point, fake, maxInputMs }) {
    const ready = new Promise((resolve, reject) => {
      this.#initWaiter = { resolve, reject };
    });
    this.#worker.onmessage = (e) => this.#onMessage(e.data);
    this.#worker.onerror = (e) => this.#fail(new CzechSpeechError(`The speaker worker failed: ${(e && e.message) || 'error'}`, { code: 'runtime' }));
    this.#worker.onmessageerror = () => this.#fail(new CzechSpeechError('The speaker worker sent an unreadable message', { code: 'runtime' }));
    // Blobs go by reference: the page keeps no copy of the 15 MB runtime, the worker makes a URL of it for the loader
    const msg = { type: 'init', model: model._modelBlob(), runtime: model._runtimeBlob(), threads, point, maxInputMs };
    if (fake) msg._fake = fake;
    this.#worker.postMessage(msg);
    await ready;
  }

  #initWaiter = null;

  #onMessage(m) {
    if (this.#onWorkerMessage) this.#onWorkerMessage(m);
    if (!m) return;
    switch (m.type) {
      case 'ready':
        this.#threads = m.threads;
        this.#initWaiter?.resolve();
        this.#initWaiter = null;
        break;
      case 'embedded': {
        const p = this.#pending.get(m.id);
        this.#pending.delete(m.id);
        if (!p || this.#closed) { m.vector?.fill(0); p?.reject(new CzechSpeechError('The speaker engine was closed', { code: 'closed' })); } // closed meanwhile
        else if (m.error !== undefined) p.reject(new CzechSpeechError(`The speaker worker could not embed the audio: ${m.error}`, { code: 'runtime' }));
        else p.resolve(m.vector);
        break;
      }
      case 'pieces': {
        if (m.listenId === 0) {
          const p = this.#pendingUtt.get(m.id);
          this.#pendingUtt.delete(m.id);
          if (!p || this.#closed) {
            for (const x of m.pieces || []) x.vector.fill(0);
            p?.reject(new CzechSpeechError('The speaker engine was closed', { code: 'closed' }));
          }
          else if (m.error !== undefined) p.reject(new CzechSpeechError(`The speaker worker could not embed the utterance: ${m.error}`, { code: 'runtime' }));
          else p.resolve({ mode: m.mode, pieces: m.pieces, computeMs: m.computeMs });
        } else {
          const l = this.#channels.get(m.listenId);
          if (l && !this.#closed) {
            try { l.onPieces(m); } catch { /* the listener's own error is its business */ }
          }
          else for (const x of m.pieces || []) x.vector.fill(0); // detached or closed meanwhile
        }
        break;
      }
      case 'closed':
        this.#closeWaiter?.();
        break;
      case 'fatal':
        this.#fail(new CzechSpeechError(`The speaker worker failed: ${m.message}`, { code: 'runtime' }));
        break;
      default:
        break;
    }
  }

  #fail(err) {
    if (this.#initWaiter) {
      this.#initWaiter.reject(err);
      this.#initWaiter = null;
    }
    this.#rejectAll(err);
    if (!this.#failed) {
      this.#failed = err;
      this.#endListeners(err);
      this.#terminate(); // nothing more is expected of it, and terminating is what frees its memory
    }
    this.#closeWaiter?.(); // a close() that waits for the worker's answer will not get one
  }

  /** Tells every attached listener that it will hear nothing more, and closes the ports. */
  #endListeners(err) {
    const all = [...this.#listeners.values()];
    this.#listeners.clear();
    this.#channels.clear();
    for (const l of all) {
      try { l.port.close(); } catch { /* closed */ }
      try { l.onFailure?.(err); } catch { /* the listener's own error is its business */ }
    }
  }

  #failed = null;

  #rejectAll(err) {
    for (const p of this.#pending.values()) p.reject(err);
    for (const p of this.#pendingUtt.values()) p.reject(err);
    this.#pending.clear();
    this.#pendingUtt.clear();
  }

  #check() {
    if (this.#closed) throw new CzechSpeechError('The speaker engine is closed', { code: 'closed' });
    if (this.#failed) throw this.#failed;
  }

  #terminate() {
    try {
      this.#worker.terminate();
    } catch { /* already gone */ }
  }

  get closed() { return this.#closed; }

  /** Threads the runtime actually uses (1 when the page is not cross-origin isolated). */
  get threads() { return this.#threads; }

  /** One embedding (192 numbers, unit length) of the whole array. The array is handed over. */
  embed(samples) {
    try {
      this.#check();
      checkSamples(samples);
    } catch (err) {
      return Promise.reject(err);
    }
    const s = transferable(samples);
    const id = this.#nextId++;
    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject });
      this.#send(() => this.#pending.delete(id), reject, s, { type: 'embed', id, samples: s });
    });
  }

  /** The pieces of one utterance and their embeddings, relative to its start. The array is handed over. */
  embedUtterance(samples) {
    try {
      this.#check();
      checkSamples(samples);
    } catch (err) {
      return Promise.reject(err);
    }
    const s = transferable(samples);
    const id = this.#nextId++;
    return new Promise((resolve, reject) => {
      this.#pendingUtt.set(id, { resolve, reject });
      this.#send(() => this.#pendingUtt.delete(id), reject, s, { type: 'utterance', id, samples: s, listenId: 0 });
    });
  }

  /** Posts a request with its array transferred; when that throws (DataCloneError...) nothing stays pending and the array is zeroed. */
  #send(unregister, reject, samples, msg) {
    try {
      this.#worker.postMessage(msg, [samples.buffer]);
    } catch (err) {
      unregister();
      try { samples.fill(0); } catch { /* detached */ }
      reject(new CzechSpeechError(`The audio could not be sent to the speaker worker: ${(err && err.message) || err}`, { code: 'runtime', cause: err }));
    }
  }

  /**
   * A port for a capture worker: it posts `{ type: 'utterance', id, samples }` on it and `onPieces(msg)` gets the
   * worker's `pieces` answer with the same `id` (`msg.error` is a string when that utterance failed; the others go on).
   * `onFailure(error)` is called once when the engine fails or is closed: nothing more will come. detach() ends it.
   */
  attach(listenId, onPieces, onFailure) {
    this.#check();
    if (!Number.isInteger(listenId) || listenId < 1) throw new RangeError('listenId must be a positive integer');
    if (this.#listeners.has(listenId)) throw new RangeError(`listenId ${listenId} is attached already`);
    const { port1, port2 } = new MessageChannel();
    const channel = this.#nextChannel++;
    const entry = { channel, onPieces, onFailure, port: port2 };
    this.#listeners.set(listenId, entry);
    this.#channels.set(channel, entry);
    try {
      this.#worker.postMessage({ type: 'attach', port: port1, listenId: channel }, [port1]);
    } catch (err) {
      this.#listeners.delete(listenId);
      this.#channels.delete(channel);
      try { port2.close(); } catch { /* closed */ }
      throw new CzechSpeechError(`The speaker worker could not take the port: ${(err && err.message) || err}`, { code: 'runtime', cause: err });
    }
    return port2;
  }

  /** Ends one listener: its port is closed on both sides and nothing is delivered to it any more. Unknown id: nothing. */
  detach(listenId) {
    const l = this.#listeners.get(listenId);
    if (!l) return;
    this.#listeners.delete(listenId);
    this.#channels.delete(l.channel);
    try { l.port.close(); } catch { /* closed */ }
    if (!this.#closed && !this.#failed) {
      try { this.#worker.postMessage({ type: 'detach', listenId: l.channel }); } catch { /* the worker is gone */ }
    }
  }

  /** Zeroes what the worker holds, stops it and rejects pending work with code 'closed'. */
  close() {
    if (this.#closing) return this.#closing;
    this.#closed = true;
    const err = new CzechSpeechError('The speaker engine was closed', { code: 'closed' });
    this.#closing = new Promise((resolve) => {
      let timer;
      const done = () => {
        clearTimeout(timer);
        this.#terminate();
        this.#endListeners(err);
        this.#rejectAll(err); // after the worker has zeroed and stopped: whoever waited hears of it now
        resolve();
      };
      this.#closeWaiter = done;
      if (this.#failed) return done(); // the worker is terminated already: nobody to wait for
      timer = setTimeout(done, CLOSE_GRACE_MS);
      try {
        this.#worker.postMessage({ type: 'close' });
      } catch {
        done();
      }
    });
    return this.#closing;
  }
}
