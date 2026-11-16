// Appended to the threaded CPU module (--post-js, native/CMakeLists.txt), inside the module's scope: lets
// src/engine-wasm.js tear the module's runtime down when it releases the module, e.g. when the recogniser goes back
// from the CPU to WebGPU after a WebGPU failure. It is Emscripten's own teardown, the one exit() runs:
//   1. terminate the pthread workers (threads + 1, each holding the module's shared memory);
//   2. clear the main runtime thread's state and wake its mailbox wait. That wait is an Atomics.waitAsync on the shared
//      memory, armed at start-up and armed again after every wake-up while the thread has an id; its pending promise
//      is kept by the browser until it is woken, and its reaction closes over the whole module, memory included.
// With step 1 alone the memory (about 1.2 GB with the model loaded) stayed reachable for as long as the worker lived:
// test/integration/teardown.test.mjs and test/e2e/run.mjs (D6) measure it. Only call it when no thread runs (between
// recognitions); the module cannot be used afterwards.
Module['czechspeechTerminateRuntime'] = () => {
  PThread.terminateRuntime();
};
