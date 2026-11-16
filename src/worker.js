// Entry module of the CzechSpeech workers (capture and recogniser; the role comes with the init message).
import { installWorker } from './worker-core.js';
import { loadWasmEngine, openBlobReader } from './engine-wasm.js';
import { probeWebGPU } from './backend.js';

installWorker(self, {
  loadEngine: loadWasmEngine,
  openModel(source) {
    if (source && typeof Blob !== 'undefined' && source.blob instanceof Blob) return openBlobReader(source.blob);
    throw new Error('the model must be a Blob in the browser');
  },
  probeGpu: () => probeWebGPU({ gpu: self.navigator && self.navigator.gpu }),
});
