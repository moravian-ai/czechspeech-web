// The WebAssembly engine as the workers use it: whisper.cpp + Silero VAD through native/czechspeech_wasm.c.
// Runs inside a worker (src/worker.js) or, for the integration tests, a Node.js worker thread.
//
// Every audio copy in the WebAssembly heap is made here and zeroed and freed in a `finally` right after the call;
// wipe() zeroes the whole heap before the worker is terminated, which also covers whisper.cpp's internal buffers
// (mel spectrogram, activations, decoded segment strings) that the C API cannot reach.
//
// Two kinds of engine:
//   WasmEngine  the CPU modules (czechspeech-mt / -st): every call is synchronous.
//   GpuEngine   czechspeech-webgpu: start, whisperInit, whisperTranscribe and whisperFree return Promises (JSPI:
//               whisper.cpp waits for the GPU). They reject at once when ggml aborts or the device is lost (and the
//               recogniser runs them under a watchdog, src/worker-core.js), so a broken GPU becomes an exception,
//               never a hang. Its GPU buffers are tracked:
//               their sizes give the GPU memory figures, and when the model is freed every buffer is cleared on the
//               GPU before it is released (whisper.cpp's buffers hold the mel spectrogram, the encoder output and the
//               decoder state of the last utterance), then the device is destroyed.

const MODEL_READ_CHUNK = 8 * 1024 * 1024;
const READ_AHEAD = 1024 * 1024;

/** cs_whisper_transcribe's max_tokens and temperature_fallback arguments. */
const decodingArgs = ({ maxTokens = 0, temperatureFallback = false } = {}) => [maxTokens > 0 ? maxTokens | 0 : 0, temperatureFallback ? 1 : 0];

/**
 * Loads one Emscripten module (czechspeech-mt.js, czechspeech-st.js or czechspeech-webgpu.js, or the -node test
 * builds of the CPU modules).
 *
 * @param {object} o
 * @param {string} o.moduleUrl     URL of the Emscripten ES module
 * @param {string} [o.wasmUrl]     URL of its .wasm (default: next to the module)
 * @param {number} o.threads       recogniser threads (pthread pool = threads + 1; ignored by the single-threaded builds)
 * @param {() => boolean} [o.abortRequested]  polled by whisper.cpp between stages
 * @param {(level: string, message: string) => void} o.log
 * @param {boolean} [o.gpu]        the WebGPU module: resolves a GpuEngine once ggml has a GPU device, else rejects
 * @param {object} [o.options]     engine options (tests and measurements): { flashAttn, verbose }
 */
export async function loadWasmEngine({ moduleUrl, wasmUrl, threads, abortRequested, log, gpu = false, options }) {
  const o = options || {};
  const hooks = gpu ? GpuTracker.install(log) : null;
  let onAbort = () => {};
  // whisper.cpp stops at its next abort check when the WebGPU engine failed (broken) or its recognition in flight is
  // asked to stop (the watchdog, GpuEngine.stop()), and when the page cancelled the work (abortRequested)
  const flags = { broken: false, stop: false };
  // The module is an asset (src/speech.js): bundlers must leave this import to the browser.
  const { default: createModule } = await import(/* webpackIgnore: true */ /* @vite-ignore */ moduleUrl);
  const M = await createModule({
    czechspeechPool: Math.max(1, threads) + 1,
    czechspeechAbortRequested: () => flags.broken || flags.stop || (abortRequested ? abortRequested() : false),
    czechspeechModelRead: () => 0,
    czechspeechModelEof: () => true,
    // whisper.cpp writes warnings and errors to stderr (cs_log_callback filters INFO/DEBUG, which can contain
    // decoded tokens). stdout is dropped: nothing in the configuration prints there (only the development profiling
    // build of the WebGPU module does, shown with the internal verbose option).
    print: o.verbose ? (text) => log('info', `stdout: ${String(text).trim()}`) : () => {},
    printErr: (text) => log('warn', `whisper.cpp: ${String(text).trim()}`),
    onAbort: (what) => onAbort(what),
    ...(wasmUrl ? { locateFile: (path) => (path.endsWith('.wasm') ? wasmUrl : path) } : {}),
  });
  // src/wasm/ is not in git: a module built before the language support has no _cs_whisper_language, and every
  // recognition, Czech included, would then fail with an opaque error
  if (typeof M._cs_whisper_language !== 'function') {
    throw new Error('the WebAssembly module is older than the SDK (it has no _cs_whisper_language); rebuild it with native/build-wasm.sh');
  }
  M._cs_init();
  // development only (an internal option): whisper.cpp/ggml INFO lines, which carry sizes and timings, never text
  if (o.verbose && typeof M._cs_set_verbose === 'function') M._cs_set_verbose(1);
  if (!gpu) return new WasmEngine(M, log);
  const engine = new GpuEngine(M, log, hooks, o, flags);
  onAbort = (what) => engine.fail(new Error(`ggml aborted: ${String(what || 'abort').slice(0, 300)}`));
  try {
    await engine.start();
  } catch (err) {
    await engine.wipe(); // releases a device that was created before the failure
    throw err;
  }
  return engine;
}

export class WasmEngine {
  constructor(M, log) {
    this.M = M;
    this.log = log;
    this.dead = false;
    this.async = false;
    this.backend = 'cpu';
    this.frameIn = 0;   // 512-sample scratch buffer for the streaming VAD
    this.frameOut = 0;
  }

  _views() {
    const buffer = this.M.wasmMemory ? this.M.wasmMemory.buffer : this.M.HEAPU8.buffer;
    if (buffer !== this._buffer || this._u8.length !== buffer.byteLength) {
      this._buffer = buffer;
      this._u8 = new Uint8Array(buffer);
      this._f32 = new Float32Array(buffer, 0, buffer.byteLength >>> 2);
    }
    return this;
  }

  _alive() {
    if (this.dead) throw new Error('engine was wiped');
  }

  get heapBytes() {
    return this._views()._u8.length;
  }

  get gpuBytes() {
    return 0;
  }

  systemInfo() {
    this._alive();
    return this.M.UTF8ToString(this.M._cs_system_info());
  }

  /** Copies samples into a fresh heap buffer and returns its pointer; release with _release(ptr, samples). */
  _copyIn(samples) {
    const bytes = samples.length * 4;
    const ptr = this.M._malloc(Math.max(4, bytes));
    if (!ptr) throw new Error('out of WebAssembly memory for the audio buffer');
    this._views()._f32.set(samples, ptr >>> 2);
    return ptr;
  }

  _release(ptr, samples) {
    this.M._cs_zero_free(ptr, samples.length * 4);
  }

  /** Copies samples into a fresh heap buffer, runs fn(ptr), then zeroes and frees the buffer. */
  _withSamples(samples, fn) {
    const ptr = this._copyIn(samples);
    try {
      return fn(ptr);
    } finally {
      this._release(ptr, samples);
    }
  }

  vadInit(modelBytes) {
    this._alive();
    const ptr = this.M._malloc(modelBytes.length);
    if (!ptr) throw new Error('out of WebAssembly memory for the VAD model');
    try {
      this._views()._u8.set(modelBytes, ptr);
      const handle = this.M._cs_vad_init(ptr, modelBytes.length, 1);
      if (!handle) throw new Error('Could not load the Silero VAD model');
      return handle;
    } finally {
      this.M._free(ptr);
    }
  }

  vadFree(handle) {
    if (!this.dead && handle) this.M._cs_vad_free(handle);
  }

  vadReset(handle) {
    this._alive();
    this.M._cs_vad_reset(handle);
  }

  /** Speech probability of one 512-sample frame, continuing the LSTM state. */
  vadFrame(handle, frame) {
    this._alive();
    if (!this.frameIn) {
      this.frameIn = this.M._malloc(frame.length * 4);
      this.frameOut = this.M._malloc(16);
      if (!this.frameIn || !this.frameOut) throw new Error('out of WebAssembly memory');
    }
    const v = this._views();
    const at = this.frameIn >>> 2;
    v._f32.set(frame, at);
    const n = this.M._cs_vad_probs(handle, this.frameIn, frame.length, this.frameOut, 4);
    const prob = this._views()._f32[this.frameOut >>> 2];
    this._views()._f32.fill(0, at, at + frame.length);
    if (n !== 1) throw new Error('Voice activity detection failed');
    return prob;
  }

  /** Speech segments in the samples (fresh VAD state) at `threshold`; -1 on error. */
  vadSpeechSegments(handle, samples, threshold) {
    this._alive();
    return this._withSamples(samples, (ptr) => this.M._cs_vad_speech_segments(handle, ptr, samples.length, threshold));
  }

  /** Points the module's loader callbacks at `reader`; returns the function that detaches them. */
  _attachReader(reader) {
    // whisper.cpp's loader makes about 100,000 tiny reads (vocabulary strings, tensor headers) between the large
    // tensor reads. Each read of the source is a synchronous call into the browser (FileReaderSync), so small
    // reads are served from a read-ahead window and large ones go straight into the heap.
    let position = 0;
    let window = new Uint8Array(0);
    let windowStart = 0;
    this.M.czechspeechModelEof = () => position >= reader.size;
    this.M.czechspeechModelRead = (dst, n) => {
      let done = 0;
      while (done < n && position < reader.size) {
        const offset = position - windowStart;
        if (offset >= 0 && offset < window.length) {
          const take = Math.min(n - done, window.length - offset);
          this._views()._u8.set(window.subarray(offset, offset + take), dst + done);
          done += take;
          position += take;
          continue;
        }
        const want = n - done;
        if (want >= READ_AHEAD) {
          const chunk = reader.read(position, Math.min(want, MODEL_READ_CHUNK, reader.size - position));
          if (chunk.length === 0) break;
          this._views()._u8.set(chunk, dst + done);
          done += chunk.length;
          position += chunk.length;
        } else {
          window = reader.read(position, Math.min(READ_AHEAD, reader.size - position));
          windowStart = position;
          if (window.length === 0) break;
        }
      }
      return done;
    };
    return () => {
      this.M.czechspeechModelRead = () => 0;
      this.M.czechspeechModelEof = () => true;
    };
  }

  /**
   * Loads the whisper model through `reader` ({ size, read(position, length) -> Uint8Array }). Returns the handle,
   * or throws.
   */
  whisperInit(reader) {
    this._alive();
    const detach = this._attachReader(reader);
    try {
      const handle = this.M._cs_whisper_init(0, 1);
      if (!handle) throw new Error('whisper.cpp could not load the model');
      return handle;
    } finally {
      detach();
    }
  }

  whisperFree(handle) {
    if (!this.dead && handle) this.M._cs_whisper_free(handle);
  }

  /**
   * Whisper on 16 kHz mono samples. Returns { status: 0, text } | { status: 1 } (cancelled) | { status: -1 }.
   * The decoded bytes are copied out of the heap, decoded, and both copies are zeroed. `limits`: { maxTokens (0: none),
   * temperatureFallback } (src/decoding.js decodingLimits). `languages`: comma-separated whisper language codes ("cs",
   * "pl,en"); one forces that language, several make whisper.cpp pick among them. The result's `language` is the one
   * the text was recognised in.
   */
  whisperTranscribe(handle, generation, samples, threads, limits = {}, languages = 'cs') {
    this._alive();
    const status = this._withSamples(samples, (ptr) => this._withCString(languages, (lang) =>
      this.M._cs_whisper_transcribe(handle, generation, ptr, samples.length, threads, ...decodingArgs(limits), lang)));
    return this._result(handle, status);
  }

  /** Copies a short ASCII string into the heap as a C string, runs fn(ptr), then zeroes and frees it. */
  _withCString(text, fn) {
    const ptr = this._cString(text);
    try {
      return fn(ptr);
    } finally {
      this._releaseCString(ptr, text);
    }
  }

  _cString(text) {
    const bytes = new TextEncoder().encode(String(text));
    const ptr = this.M._malloc(bytes.length + 1);
    if (!ptr) throw new Error('out of WebAssembly memory for the language list');
    const u8 = this._views()._u8;
    u8.set(bytes, ptr);
    u8[ptr + bytes.length] = 0;
    return ptr;
  }

  _releaseCString(ptr, text) {
    this._views()._u8.fill(0, ptr, ptr + new TextEncoder().encode(String(text)).length + 1);
    this.M._free(ptr);
  }

  _result(handle, status) {
    if (status !== 0) {
      this.M._cs_text_release(handle);
      return { status };
    }
    try {
      const timings = this._timings(handle);
      const ptr = this.M._cs_text_ptr(handle);
      const len = this.M._cs_text_len(handle);
      // text tokens decoded, and whether a token limit stopped the decode (src/decoding.js): numbers, no content
      const decoded = typeof this.M._cs_whisper_tokens === 'function'
        ? { tokens: this.M._cs_whisper_tokens(handle), capped: this.M._cs_whisper_capped(handle) !== 0 } : {};
      const language = this.M.UTF8ToString(this.M._cs_whisper_language(handle));
      if (!ptr || len === 0) return { status, text: '', timings, language, ...decoded };
      const bytes = this._views()._u8.slice(ptr, ptr + len); // a non-shared copy: TextDecoder rejects shared memory
      try {
        return { status, text: new TextDecoder('utf-8').decode(bytes), timings, language, ...decoded };
      } finally {
        bytes.fill(0);
      }
    } finally {
      this.M._cs_text_release(handle);
    }
  }

  /** { encodeMs, decodeMsPerToken } of the last transcription (no content). */
  _timings(handle) {
    const ptr = this.M._malloc(16);
    if (!ptr) return null;
    try {
      this.M._cs_whisper_timings(handle, ptr);
      const f = this._views()._f32;
      return { encodeMs: Math.round(f[ptr >>> 2]), decodeMsPerToken: Math.round(f[(ptr >>> 2) + 1] * 10) / 10 };
    } finally {
      this.M._free(ptr);
    }
  }

  _zeroHeap() {
    try {
      this._views()._u8.fill(0);
    } catch (e) {
      this.log('warn', `wiping the WebAssembly heap failed: ${e && e.message}`);
    }
  }

  /**
   * Zeroes the entire WebAssembly heap. The engine is unusable afterwards. The threaded module then tears its runtime
   * down (native/threads-teardown.js): it terminates its pthread workers (threads + 1, each holding the module's shared
   * memory: about 1.2 GB with the model loaded) and wakes its main thread's mailbox wait, whose pending promise would
   * keep the whole module reachable. So nothing of it outlives the engine when the recogniser goes on without it
   * (WebGPU back after a failure): the memory goes with the last reference to the engine (measured:
   * test/integration/teardown.test.mjs, test/e2e/run.mjs D6).
   */
  wipe() {
    if (this.dead) return;
    this.dead = true;
    this._zeroHeap();
    if (typeof this.M.czechspeechTerminateRuntime === 'function') {
      try {
        this.M.czechspeechTerminateRuntime();
      } catch (e) {
        this.log('warn', `tearing the WebAssembly runtime down failed: ${e && e.message}`);
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// WebGPU

const COPY_DST = 0x0008; // GPUBufferUsage.COPY_DST


/**
 * Knows the GPUDevice that ggml (through emdawnwebgpu) creates for one engine, and every buffer on it. The worker's
 * navigator.gpu.requestAdapter -> adapter.requestDevice is wrapped once; a device goes to the tracker of the engine
 * that is starting (`current`), so a released engine never touches the device of a later one.
 */
class GpuTracker {
  static #hooked = new WeakSet();
  static #current = null;

  static install(log) {
    const gpu = globalThis.navigator && globalThis.navigator.gpu;
    if (!gpu || typeof gpu.requestAdapter !== 'function') throw new Error('navigator.gpu is not available in the worker');
    if (!GpuTracker.#hooked.has(gpu)) {
      GpuTracker.#hooked.add(gpu);
      const requestAdapter = gpu.requestAdapter.bind(gpu);
      gpu.requestAdapter = async (options) => {
        const adapter = await requestAdapter(options);
        if (adapter && typeof adapter.requestDevice === 'function') {
          const requestDevice = adapter.requestDevice.bind(adapter);
          adapter.requestDevice = async (descriptor) => {
            const device = await requestDevice(descriptor);
            const owner = GpuTracker.#current;
            if (owner && !owner.device) owner.#attach(device, adapter);
            return device;
          };
        }
        return adapter;
      };
    }
    const t = new GpuTracker(log);
    GpuTracker.#current = t;
    return t;
  }

  constructor(log) {
    this.log = log;
    this.device = null;
    this.adapterInfo = null;
    this.buffers = new Set();
    this.liveBytes = 0;
    this.peakBytes = 0;
    this.onLost = null;
  }

  #attach(device, adapter) {
    this.device = device;
    const i = adapter.info || {};
    this.adapterInfo = { vendor: i.vendor || '', architecture: i.architecture || '', device: i.device || '', description: i.description || '' };
    const createBuffer = device.createBuffer.bind(device);
    device.createBuffer = (descriptor) => {
      const buffer = createBuffer(descriptor);
      if (this.device === device) this.#track(buffer, Number(descriptor && descriptor.size) || buffer.size || 0);
      return buffer;
    };
    device.lost.then((info) => {
      if (this.device === device && this.onLost && !(info && info.reason === 'destroyed')) {
        this.onLost(new Error(`WebGPU device lost${info && info.message ? `: ${info.message}` : ''}`));
      }
    }, () => {});
  }

  #track(buffer, size) {
    this.buffers.add(buffer);
    this.liveBytes += size;
    this.peakBytes = Math.max(this.peakBytes, this.liveBytes);
    const destroy = buffer.destroy.bind(buffer);
    buffer.destroy = () => {
      if (this.buffers.delete(buffer)) this.liveBytes -= size;
      destroy();
    };
  }

  /** Clears every live buffer on the GPU (zeroes) and waits for it; errors are reported, never thrown. */
  async clearAll(timeoutMs = 3000) {
    const device = this.device;
    if (!device) return 0;
    let cleared = 0;
    try {
      device.pushErrorScope('validation');
      const encoder = device.createCommandEncoder();
      for (const b of this.buffers) {
        if ((b.usage & COPY_DST) && (b.mapState === undefined || b.mapState === 'unmapped') && b.size % 4 === 0) {
          encoder.clearBuffer(b);
          cleared += b.size;
        }
      }
      device.queue.submit([encoder.finish()]);
      const error = await device.popErrorScope();
      if (error) this.log('warn', `clearing the GPU buffers: ${error.message}`);
      await withTimeout(device.queue.onSubmittedWorkDone(), timeoutMs, 'clearing the GPU buffers');
    } catch (e) {
      this.log('warn', `clearing the GPU buffers failed: ${e && e.message}`);
    }
    return cleared;
  }

  /** Destroys every buffer and the device: the GPU memory goes back to the browser. */
  destroyAll() {
    const device = this.device;
    for (const b of [...this.buffers]) {
      try { b.destroy(); } catch { /* already gone */ }
    }
    this.buffers.clear();
    this.liveBytes = 0;
    if (device) {
      this.onLost = null;
      try { device.destroy(); } catch { /* already lost */ }
    }
    this.device = null;
  }
}

function withTimeout(promise, ms, what) {
  let timer;
  return Promise.race([
    promise,
    new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`${what} did not finish in ${Math.round(ms / 1000)} s`)), ms); }),
  ]).finally(() => clearTimeout(timer));
}

export class GpuEngine extends WasmEngine {
  /** `flags` ({ broken, stop }) is what the module's abort callback reads (loadWasmEngine). */
  constructor(M, log, tracker, options = {}, flags = { broken: false, stop: false }) {
    super(M, log);
    this.flags = flags;
    // test hook (internal engine option): the device is lost this many ms into the engine's first recognition; the
    // recogniser passes it to its first WebGPU engines only (src/worker-core.js Backends, testFailRecognitionCount)
    this.testFailMs = options.testFailRecognitionMs || 0;
    this.async = true;
    this.backend = 'webgpu';
    this.tracker = tracker;
    this.flashAttn = options.flashAttn ? 1 : 0; // the recogniser passes 0 by default (src/worker-core.js)
    this.failure = null;
    this.#failed = new Promise((_, reject) => { this.#rejectFailed = reject; });
    this.#failed.catch(() => {});
    tracker.onLost = (err) => this.fail(err);
    // After ggml aborts inside a WebGPU callback (an uncaptured device error), the exception surfaces as an uncaught
    // error of this worker; it is reported through fail() instead.
    this.#onError = (e) => {
      if (this.failure && e && typeof e.preventDefault === 'function') e.preventDefault();
    };
    if (typeof globalThis.addEventListener === 'function') globalThis.addEventListener('error', this.#onError);
  }

  #failed;
  #rejectFailed;
  #onError;
  #pending = null; // the JSPI call in flight
  #recognizing = false; // whisperTranscribe() is running
  #cleared = 0;    // bytes of GPU buffers zeroed before release

  get adapter() {
    return this.tracker.adapterInfo;
  }

  get gpuBytes() {
    return this.tracker.liveBytes;
  }

  get peakGpuBytes() {
    return this.tracker.peakBytes;
  }

  /**
   * Marks the engine broken (ggml abort, device loss, watchdog); pending and later calls reject with `err`. whisper.cpp
   * is told to stop at its next abort check and the device is destroyed, so that a call still suspended in the module
   * wakes up and ends (ggml aborts on the failed wait) instead of waiting for the GPU.
   */
  fail(err) {
    if (this.failure) return;
    this.failure = err;
    this.flags.broken = true;
    this.#rejectFailed(err);
    this.tracker.destroyAll();
  }

  /**
   * Asks the recognition in flight to end at whisper.cpp's next abort check (after the encoder pass, after each
   * decoder step); it then resolves with { status: 1 } like a cancelled one. Unlike fail(), the engine and its device
   * stay usable. The request ends with the call; between calls it does nothing.
   */
  stop() {
    if (this.#recognizing) this.flags.stop = true;
  }

  _alive() {
    super._alive();
    if (this.failure) throw this.failure;
  }

  /** Runs one JSPI call; rejects when it fails, ggml aborts or the device is lost (the engine is broken then). */
  async #guard(call) {
    const pending = Promise.resolve().then(call);
    this.#pending = pending;
    try {
      return await Promise.race([pending, this.#failed]);
    } catch (err) {
      this.fail(err instanceof Error ? err : new Error(String(err)));
      throw this.failure;
    }
  }

  /** ggml's backend registry: requests the adapter and the device; rejects unless a GPU device started. */
  async start() {
    const n = await this.#guard(() => this.M._cs_backend_init());
    if (n < 1 || !this.tracker.device) {
      throw new Error(n < 0 ? 'the GPU backend did not start' : 'ggml found no WebGPU device (adapter without shader-f16?)');
    }
  }

  async whisperInit(reader) {
    this._alive();
    const detach = this._attachReader(reader);
    let handle;
    try {
      handle = await this.#guard(() => this.M._cs_whisper_init(1, this.flashAttn));
    } finally {
      detach();
    }
    if (!handle) throw new Error('whisper.cpp could not load the model');
    // whisper.cpp keeps the weights on the CPU when the GPU backend fails to start (it only logs an error); then
    // this would be a slow single-threaded CPU engine. The weights (about half the file size) must be on the GPU.
    if (this.tracker.liveBytes < reader.size / 2) {
      throw new Error(`the model was not placed on the GPU (${Math.round(this.tracker.liveBytes / 2 ** 20)} MiB of GPU buffers)`);
    }
    return handle;
  }

  async whisperTranscribe(handle, generation, samples, threads, limits = {}, languages = 'cs') {
    this._alive();
    const ptr = this._copyIn(samples);
    let lang = 0;
    let status;
    let timer = null;
    if (this.testFailMs) {
      timer = setTimeout(() => this.fail(new Error('simulated WebGPU device loss (test hook)')), this.testFailMs);
      this.testFailMs = 0;
    }
    this.flags.stop = false;
    this.#recognizing = true;
    try {
      lang = this._cString(languages);
      status = await this.#guard(() => this.M._cs_whisper_transcribe(handle, generation, ptr, samples.length, threads,
        ...decodingArgs(limits), lang));
    } finally {
      if (lang && !this.failure) this._releaseCString(lang, languages); // after a failure wipe() zeroes the heap
      this.#recognizing = false;
      this.flags.stop = false;
      if (timer) clearTimeout(timer);
      // A failed module may be left suspended; its heap is zeroed by wipe(). Otherwise free the copy now.
      if (!this.failure) this._release(ptr, samples);
      else this._views()._f32.fill(0, ptr >>> 2, (ptr >>> 2) + samples.length);
    }
    return this._result(handle, status);
  }

  async whisperFree(handle) {
    if (this.dead || !handle) return;
    if (this.failure) return; // the module is broken; wipe() releases the GPU memory
    // Zero whisper.cpp's GPU buffers (mel spectrogram, encoder output, decoder state) before they are released.
    this.#cleared += await this.tracker.clearAll();
    await this.#guard(() => this.M._cs_whisper_free(handle));
  }

  /** Clears and destroys the GPU buffers and the device, then zeroes the heap. */
  async wipe() {
    if (this.dead) return;
    if (!this.failure) this.#cleared += await this.tracker.clearAll();
    this.tracker.destroyAll();
    if (this.failure && this.#pending) {
      // A call of the broken module may still be suspended: let it run to its end (it aborts or stops now that the
      // device is gone) before the heap under it is zeroed. It is bounded: a call that never wakes up is harmless.
      await Promise.race([this.#pending.then(() => {}, () => {}), new Promise((r) => setTimeout(r, 5000))]);
    }
    this.dead = true;
    this._zeroHeap();
    this.log('info', this.failure ? 'WebGPU module released after a failure (GPU buffers destroyed, heap zeroed)'
      : `GPU buffers cleared (${Math.round(this.#cleared / 2 ** 20)} MiB) and released, heap zeroed`);
    if (typeof globalThis.removeEventListener === 'function') {
      // keep swallowing the uncaught errors of a failed module a little longer (late callbacks)
      setTimeout(() => globalThis.removeEventListener('error', this.#onError), this.failure ? 5000 : 0);
    }
  }
}

/** Synchronous reader over a Blob (dedicated workers only: FileReaderSync). */
export function openBlobReader(blob) {
  const reader = new FileReaderSync();
  return {
    size: blob.size,
    read(position, length) {
      return new Uint8Array(reader.readAsArrayBuffer(blob.slice(position, position + length)));
    },
  };
}
