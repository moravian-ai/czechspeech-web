// The SDK's worker entry (src/worker.js) with the Workers it creates counted (count-workers.js), for test/e2e/run.mjs:
// the threaded CPU module starts its pthread workers from here (threads + 1, each holding the module's shared memory).
// Static imports, evaluated in this order: a dynamic import would install the worker's message handler too late for the
// init message.
import './count-workers.js';
import '../../src/worker.js';
