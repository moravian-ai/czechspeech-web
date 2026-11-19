// Whether this browser can tell speakers apart: onnxruntime-web needs WebAssembly SIMD and the speaker worker needs Web
// Workers. Two runtime threads only on a cross-origin isolated page with 8 or more logical cores (the recogniser keeps
// some of them, and on WebGPU the GPU does the recogniser's work).

// A module of one function that uses a v128 instruction: valid only where WebAssembly SIMD exists.
// (i32.const 0, i8x16.splat, i8x16.neg: the smallest module that needs a vector instruction).
const SIMD_PROBE = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]);

/**
 * @param {object} [env]  globalThis by default (a test passes its own)
 * @returns {{ supported: boolean, threads: number, reason: 'no-webassembly-simd' | 'no-workers' | null }}
 */
export function speakerSupport(env = globalThis) {
  let simd;
  try {
    simd = !!env.WebAssembly && typeof env.WebAssembly.validate === 'function' && env.WebAssembly.validate(SIMD_PROBE);
  } catch {
    simd = false;
  }
  if (!simd) return { supported: false, threads: 0, reason: 'no-webassembly-simd' };
  if (typeof env.Worker !== 'function') return { supported: false, threads: 0, reason: 'no-workers' };
  const cores = env.navigator && env.navigator.hardwareConcurrency;
  const threads = env.crossOriginIsolated === true && typeof cores === 'number' && cores >= 8 ? 2 : 1;
  return { supported: true, threads, reason: null };
}
