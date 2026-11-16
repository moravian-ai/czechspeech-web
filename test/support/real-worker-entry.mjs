// Worker entry for the Node.js integration tests: the real WebAssembly engine (Node builds of the same objects,
// build/wasm-node/) with the model read from a file path instead of a Blob (Node has no FileReaderSync).
import fs from 'node:fs';
import { installWorker } from '../../src/worker-core.js';
import { loadWasmEngine } from '../../src/engine-wasm.js';
import { probeWebGPU } from '../../src/backend.js';

const fds = new Map();

installWorker(self, {
  loadEngine: loadWasmEngine,
  openModel(source) {
    if (!source || !source.path) throw new Error('the Node.js test worker needs a model path');
    const fd = fds.get(source.path) ?? fs.openSync(source.path, 'r');
    fds.set(source.path, fd);
    const size = fs.fstatSync(fd).size;
    return {
      size,
      read(position, length) {
        const buf = Buffer.allocUnsafe(length);
        const n = fs.readSync(fd, buf, 0, length, position);
        return new Uint8Array(buf.buffer, buf.byteOffset, n);
      },
    };
  },
  // Node.js has no navigator.gpu: config.backend 'auto' falls back to the CPU here, the same code path as a browser
  // without WebGPU.
  probeGpu: () => probeWebGPU({ gpu: globalThis.navigator && globalThis.navigator.gpu }),
});
