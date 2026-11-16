// Backend selection for the recogniser: WebGPU (ggml's WebGPU backend in czechspeech-webgpu.wasm) or the CPU
// WebAssembly modules. Runs in the recogniser worker, so it sees the worker's own navigator.gpu; no environment
// dependencies beyond what is passed in, so the unit tests can feed it fake GPU objects.
//
// 'auto' (default) uses WebGPU only when all of these hold, and otherwise the CPU modules, with one notice:
//   - JSPI (WebAssembly.Suspending / WebAssembly.promising): the WebGPU module waits for the GPU through it;
//   - navigator.gpu exists in the worker;
//   - requestAdapter() gives an adapter, and it is not a software ("fallback") adapter;
//   - the adapter has 'shader-f16' (ggml-webgpu registers no device without it);
//   - the WebGPU module starts: ggml requests the device and builds its first pipeline (checked by the caller).
// 'webgpu' insists on WebGPU (CzechSpeechError code 'unsupported' otherwise); 'cpu' never touches the GPU.

export const BACKENDS = Object.freeze(['auto', 'webgpu', 'cpu']);

/** True when the WebAssembly JS Promise Integration API is available. */
export function jspiSupported(wasm = globalThis.WebAssembly) {
  return !!wasm && typeof wasm.Suspending === 'function' && typeof wasm.promising === 'function';
}

/** Adapter description without anything that identifies the user beyond what the browser already exposes. */
export function adapterSummary(adapter) {
  const i = (adapter && adapter.info) || {};
  const s = (v) => (typeof v === 'string' ? v : '');
  return { vendor: s(i.vendor), architecture: s(i.architecture), device: s(i.device), description: s(i.description) };
}

export function adapterLabel(a) {
  if (!a) return 'unknown adapter';
  return [a.vendor, a.architecture].filter(Boolean).join(' ') || a.description || a.device || 'unknown adapter';
}

/**
 * Whether this worker can run the WebGPU module. Resolves { ok: true, adapter } or { ok: false, reason }; never
 * rejects. `gpu` is navigator.gpu (or undefined), `wasm` the WebAssembly namespace.
 */
export async function probeWebGPU({ gpu, wasm = globalThis.WebAssembly } = {}) {
  if (!jspiSupported(wasm)) return { ok: false, reason: 'WebAssembly JSPI (WebAssembly.Suspending) is not available' };
  if (!gpu || typeof gpu.requestAdapter !== 'function') return { ok: false, reason: 'navigator.gpu is not available in the worker' };
  let adapter;
  try {
    adapter = await gpu.requestAdapter();
  } catch (e) {
    return { ok: false, reason: `requestAdapter() failed: ${(e && e.message) || e}` };
  }
  if (!adapter) return { ok: false, reason: 'requestAdapter() gave no adapter' };
  const info = adapter.info || {};
  const summary = adapterSummary(adapter);
  if (info.isFallbackAdapter === true || adapter.isFallbackAdapter === true) {
    return { ok: false, reason: `the adapter is a software fallback (${adapterLabel(summary)})`, adapter: summary };
  }
  const features = adapter.features;
  if (!features || typeof features.has !== 'function' || !features.has('shader-f16')) {
    return { ok: false, reason: `the adapter (${adapterLabel(summary)}) has no shader-f16`, adapter: summary };
  }
  return { ok: true, adapter: summary, subgroups: features.has('subgroups') };
}

/**
 * 'auto' after a WebGPU failure mid-session: how long the instance stays on the CPU before a recogniser starts the WebGPU
 * module again, after the first, second and third failure in a row (a failure of the re-created module counts too, a
 * recognition that succeeds on WebGPU ends the row). One failure more and it stays on the CPU.
 */
export const GPU_RETRY_DELAYS_MS = Object.freeze([10000, 60000, 300000]);

/**
 * When 'auto' tries WebGPU again after a failure (fake `now` in the tests). The CzechSpeech instance owns this state
 * (src/speech.js): each recogniser starts from its snapshot() and reports every change back, so that the row and the
 * delays hold across recognisers (with keepModelLoaded: false every call has one). A snapshot carries the time left
 * rather than a time, because the page and the worker have different clocks.
 */
export class GpuRecovery {
  constructor({ delaysMs = GPU_RETRY_DELAYS_MS, now = () => performance.now() } = {}) {
    this.delaysMs = delaysMs;
    this.now = now;
    this.failures = 0;  // in a row, without a successful WebGPU recognition in between
    this.retryAt = null;
  }

  /** The WebGPU engine failed, or could not be started again. */
  failed() {
    this.failures++;
    this.retryAt = this.gaveUp ? null : this.now() + this.delaysMs[this.failures - 1];
  }

  /** A recognition succeeded on WebGPU. */
  succeeded() {
    this.failures = 0;
    this.retryAt = null;
  }

  /** Too many failures in a row: the CPU from now on. */
  get gaveUp() {
    return this.failures > this.delaysMs.length;
  }

  /** WebGPU should be started again now. */
  get due() {
    return this.retryAt !== null && this.now() >= this.retryAt;
  }

  /** { failures, retryInMs (null: no try pending), gaveUp }: what goes between the page and a recogniser. */
  snapshot() {
    const retryInMs = this.retryAt === null ? null : Math.max(0, Math.round(this.retryAt - this.now()));
    return { failures: this.failures, retryInMs, gaveUp: this.gaveUp };
  }

  /** Takes over a snapshot (of the page, or of a recogniser). */
  adopt({ failures = 0, retryInMs = null } = {}) {
    this.failures = Number.isInteger(failures) && failures > 0 ? failures : 0;
    const pending = !this.gaveUp && retryInMs !== null && retryInMs !== undefined;
    this.retryAt = pending ? this.now() + Math.max(0, retryInMs) : null;
  }

  /** Seconds until the next try, for the log; null when there is none. */
  get retryInSeconds() {
    return this.retryAt === null ? null : Math.max(0, Math.round((this.retryAt - this.now()) / 1000));
  }
}

/**
 * The plan for one recogniser start: which module to try first and what to do when WebGPU is not available.
 *   requested   config.backend
 *   gpuModule   true when the WebGPU module is part of the assets
 *   blocked     reason why this CzechSpeech instance stopped using WebGPU (a failure earlier on), or null
 *   retryInMs   'auto' after a WebGPU failure: ms until WebGPU may be tried again (GpuRecovery), or null
 *   failure     the reason of that failure, for backendFallback
 * Returns { tryGpu, onUnavailable: 'cpu' | 'fail', reason }.
 */
export function backendPlan(requested, { gpuModule = true, blocked = null, retryInMs = null, failure = null } = {}) {
  if (requested === 'cpu') return { tryGpu: false, onUnavailable: 'cpu', reason: null };
  const onUnavailable = requested === 'webgpu' ? 'fail' : 'cpu';
  if (!gpuModule) return { tryGpu: false, onUnavailable, reason: 'the WebGPU module is not part of the assets' };
  if (blocked && requested === 'auto') return { tryGpu: false, onUnavailable, reason: blocked };
  if (requested === 'auto' && retryInMs > 0) {
    const why = failure || 'WebGPU failed';
    const inSeconds = Math.round(retryInMs / 1000);
    return { tryGpu: false, onUnavailable, reason: `${why} (WebGPU is tried again in ${inSeconds} s)` };
  }
  return { tryGpu: true, onUnavailable, reason: null };
}
