// Entry module of the speaker worker: onnxruntime-web (WebAssembly, no proxy worker, 1-2 threads) computing the
// voiceprint embedding of an audio array. The runtime files are the ones shipped in ./ort/ (never a CDN): the wasm comes
// from the page as a Blob (it is in the model cache) and is handed to the loader as an object URL, so that no copy of it
// stays in JavaScript memory and it compiles while streaming; the JavaScript modules come from this package.
//
// Privacy: the input array is zeroed whether the run succeeds or throws. The log-mel features, the activations and the
// WebAssembly heap of the runtime are not ours to scrub; terminating the worker (SpeakerEngine.close()) is what frees them.
import { installSpeakerWorker } from './worker-core.js';
import { fbank } from './fbank.js';
import { unit } from './maths.js';

async function loadEmbedder({ model, runtime, threads }) {
  const ort = await import('./ort/ort.wasm.min.mjs');
  const env = ort.env;
  env.logLevel = 'error';
  env.wasm.numThreads = threads;
  // `new URL('./x', import.meta.url)` with a literal file name: bundlers recognise it and emit the file
  const wasmUrl = URL.createObjectURL(runtime.slice(0, runtime.size, 'application/wasm'));
  env.wasm.wasmPaths = { mjs: new URL('./ort/ort-wasm-simd-threaded.mjs', import.meta.url).href, wasm: wasmUrl };
  env.wasm.proxy = false;
  let session;
  try {
    session = await ort.InferenceSession.create(new Uint8Array(await model.arrayBuffer()), {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
  } finally {
    URL.revokeObjectURL(wasmUrl);
  }
  return {
    // numThreads after the runtime started: 1 when the page is not cross-origin isolated
    threads: env.wasm.numThreads,
    async embed(samples) {
      const { data, frames } = fbank(samples);
      try {
        if (frames < 1) throw new RangeError('audio too short for one frame');
        const input = new ort.Tensor('float32', data, [1, frames, 80]);
        const output = await session.run({ x: input });
        const raw = output.embedding.data;
        try {
          return unit(raw);
        } finally {
          raw.fill(0); // the runtime's own copy of the voiceprint
        }
      } finally {
        data.fill(0);
      }
    },
    release() {
      session.release().catch(() => { /* already released */ });
    },
  };
}

installSpeakerWorker(self, { loadEmbedder });
