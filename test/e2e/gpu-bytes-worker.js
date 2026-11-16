// The SDK's worker entry (src/worker.js) with the WebGPU buffers it creates counted (gpu-bytes.js), for
// test/e2e/detect-cost.mjs. Static imports, evaluated in this order (see counting-worker.js).
import './gpu-bytes.js';
import '../../src/worker.js';
