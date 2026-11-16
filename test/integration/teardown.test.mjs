// A released threaded CPU module leaves nothing reachable (src/engine-wasm.js wipe(), native/threads-teardown.js): its
// pthread workers are terminated and its WebAssembly memory (about 1.2 GB with the model loaded) can be collected, so
// that a recogniser that goes back from the CPU to WebGPU after a WebGPU failure does not keep a CPU module per
// cycle. Observed with WeakRefs to each module's WebAssembly.Memory and forced garbage collections, on the Node link
// of the threaded module (build/wasm-node/); test/e2e/run.mjs (D6) checks the same in the browser through the whole
// recogniser. No model needed: the module's own runtime is what could keep it alive.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import v8 from 'node:v8';
import vm from 'node:vm';
import { loadWasmEngine } from '../../src/engine-wasm.js';
import { ROOT } from '../support/env.mjs';

const mtNode = path.join(ROOT, 'build/wasm-node/czechspeech-mt-node.js');
const skip = fs.existsSync(mtNode) ? false : 'build/wasm-node/czechspeech-mt-node.js not built; run native/build-wasm.sh';
if (skip) console.log(`SKIP: ${skip}`);

// gc() without --expose-gc on the command line: the flag set at run time applies to contexts created afterwards
v8.setFlagsFromString('--expose-gc');
const gc = vm.runInNewContext('gc');
const nextTask = () => new Promise((resolve) => setTimeout(resolve, 10));

async function collect() {
  // a WeakRef keeps its target until the end of the task that read it; a notified Atomics.waitAsync settles in a task
  for (let i = 0; i < 6; i++) {
    await nextTask();
    gc();
  }
  await nextTask();
}

const load = () => loadWasmEngine({ moduleUrl: pathToFileURL(mtNode).href, threads: 2, log: () => {} });

/** Loads an engine, releases it as the recogniser does, and returns a WeakRef to its memory. */
async function releasedModule() {
  // in a function of its own: a suspended async function can keep a dead local (the engine) in its saved registers
  const engine = await load();
  const memory = new WeakRef(engine.M.wasmMemory);
  engine.wipe();
  return memory;
}

test('a released threaded CPU module leaves its memory collectable, cycle after cycle', { skip }, async () => {
  const released = [await releasedModule(), await releasedModule()];
  const inUse = await load(); // the control: the check must see a module that is still in use
  const control = new WeakRef(inUse.M.wasmMemory);
  try {
    await collect();
    assert.ok(control.deref(), 'the memory of the module in use is reachable');
    assert.deepEqual(released.map((r) => (r.deref() ? 'reachable' : 'collected')), ['collected', 'collected'],
      'the memory of each released module was collected');
  } finally {
    inUse.wipe();
  }
});
