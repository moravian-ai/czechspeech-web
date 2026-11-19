// The vendored onnxruntime-web (MIT): the WebAssembly runtime of the speaker worker, shipped inside the package and
// never loaded from a CDN. Its sha256 is pinned; scripts/check-package.mjs and VoiceprintModel check it.
export const ORT_VERSION = '1.31.1';

export const ORT_RUNTIME = Object.freeze({
  path: 'speakers/ort/ort-wasm-simd-threaded.wasm',
  size: 14848852,
  sha256: '4b884c203b0fa48d057be3332c65513ff85aa3c5bddd6946acc0fab7ecda001d',
});

/** JavaScript modules next to the wasm (paths relative to src/). */
export const ORT_MODULES = Object.freeze(['speakers/ort/ort.wasm.min.mjs', 'speakers/ort/ort-wasm-simd-threaded.mjs']);
