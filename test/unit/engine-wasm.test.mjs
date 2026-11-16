// src/engine-wasm.js without WebAssembly: the engines run against fake Emscripten modules (and the WebGPU engine
// against a fake navigator.gpu), which record what the engine asks of them.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import { WasmEngine, loadWasmEngine } from '../../src/engine-wasm.js';

const MiB = 1024 * 1024;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** A seeded generator (xorshift32): a failure can be reproduced. */
function random(seed) {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0;
    return s / 2 ** 32;
  };
}

/**
 * The read pattern of whisper.cpp's model loader: the magic and hyperparameters, the vocabulary (a length and a short
 * string per token, about 100,000 reads), then per tensor a header of small reads and its data in one read.
 */
function loaderReads(seed) {
  const r = random(seed);
  const reads = [4, ...Array(11).fill(4)];
  for (let i = 0; i < 51866; i++) reads.push(4, 1 + Math.floor(r() * 12));
  for (let t = 0; t < 40; t++) {
    reads.push(4, 4, 4, 4, 4, 8 + Math.floor(r() * 24));
    reads.push(t % 4 === 0 ? 2 * MiB + Math.floor(r() * MiB) : 4096 + Math.floor(r() * 400000));
  }
  reads.push(9 * MiB + 3); // larger than one engine read (8 MiB)
  return reads;
}

test('model load: whisper.cpp\'s 100,000 small reads come from a read-ahead window, each byte read once', () => {
  // regression (028da8d): every small read was one synchronous FileReaderSync call; the load took 34 s in Chromium
  const reads = loaderReads(7);
  const size = reads.reduce((a, b) => a + b, 0);
  const model = crypto.randomBytes(size);
  const reader = { size, calls: 0, bytes: 0, read(position, length) {
    this.calls++;
    this.bytes += length;
    return new Uint8Array(model.buffer, model.byteOffset + position, Math.min(length, size - position));
  } };
  const heap = new Uint8Array(16 * MiB);
  let mismatches = 0;
  let eof = null;
  const M = {
    HEAPU8: heap,
    // whisper.cpp's loader: reads through the callbacks the engine installs, here always into the same buffer
    _cs_whisper_init() {
      let position = 0;
      for (const n of reads) {
        const got = M.czechspeechModelRead(64, n);
        if (got !== n || !Buffer.from(heap.buffer, 64, n).equals(model.subarray(position, position + n))) mismatches++;
        position += n;
      }
      eof = M.czechspeechModelEof();
      return 1;
    },
  };
  const engine = new WasmEngine(M, () => {});
  assert.equal(engine.whisperInit(reader), 1);
  assert.equal(mismatches, 0, 'every read got the right bytes');
  assert.equal(eof, true);
  assert.equal(reader.bytes, size, 'each byte of the file read once');
  assert.ok(reader.calls < 100, `${reader.calls} reads of the file for ${reads.length} reads of the loader`);
  // afterwards the loader callbacks are detached
  assert.equal(M.czechspeechModelRead(0, 10), 0);
  assert.equal(M.czechspeechModelEof(), true);
});

// ---- the WebGPU engine (GpuEngine) with a fake navigator.gpu and a fake Emscripten module ----

const FAKE_MODULE = new URL('../support/fake-emscripten-module.mjs', import.meta.url).href;
const COPY_DST = 0x08;
const STORAGE = 0x80;

/** A fake navigator.gpu: every device it creates is in `devices`, with its buffers and a log of what was done. */
function installGpu(t) {
  const devices = [];
  const gpu = {
    devices,
    async requestAdapter() {
      return {
        info: { vendor: 'fakevendor', architecture: 'fakearch' },
        features: new Set(['shader-f16']),
        async requestDevice() {
          let lost;
          const device = {
            destroyed: false,
            buffers: [],
            log: [],
            lost: new Promise((r) => { lost = r; }),
            createBuffer({ size, usage }) {
              const b = { size, usage, mapState: 'unmapped', destroyed: false };
              b.destroy = () => { b.destroyed = true; device.log.push(`destroy ${size}`); };
              device.buffers.push(b);
              return b;
            },
            createCommandEncoder() {
              const cleared = [];
              return { clearBuffer: (b) => cleared.push(b), finish: () => cleared };
            },
            queue: {
              submit: (commands) => { for (const c of commands) for (const b of c) device.log.push(`clear ${b.size}`); },
              onSubmittedWorkDone: async () => {},
            },
            pushErrorScope() {},
            popErrorScope: async () => null,
            destroy() {
              if (device.destroyed) return;
              device.destroyed = true;
              device.log.push('destroy device');
              lost({ reason: 'destroyed', message: '' });
            },
          };
          devices.push(device);
          return device;
        },
      };
    },
  };
  Object.defineProperty(globalThis.navigator, 'gpu', { configurable: true, value: gpu });
  t.after(() => { delete globalThis.navigator.gpu; });
  return gpu;
}

/**
 * The WebGPU module as the engine sees it (JSPI calls return Promises). ggml's backend start requests the adapter
 * and the device; the model load allocates GPU buffers for the weights and whisper.cpp's state in the heap (marked);
 * a recognition waits for the GPU until the device goes away, then runs on and records what it found.
 */
function fakeGpuModule(overrides) {
  const heap = new Uint8Array(MiB);
  let next = 1024;
  const M = {
    ...overrides,
    HEAPU8: heap,
    device: null,
    state: 0,
    woke: null, // what the recognition found when it woke up: { heapIntact, abortRequested }
    _malloc(n) {
      const p = next;
      next += (n + 7) & ~7;
      return p;
    },
    _free() {},
    _cs_zero_free(p, n) { heap.fill(0, p, p + n); },
    _cs_init() {},
    _cs_whisper_language() { return 0; },
    UTF8ToString() { return "cs"; },
    _cs_text_release() {},
    async _cs_backend_init() {
      const adapter = await globalThis.navigator.gpu.requestAdapter();
      M.device = await adapter.requestDevice();
      M.device.createBuffer({ size: 4096, usage: COPY_DST });
      return 1;
    },
    async _cs_whisper_init() {
      M.device.createBuffer({ size: MiB, usage: STORAGE | COPY_DST });
      M.device.createBuffer({ size: 256, usage: STORAGE });
      M.state = M._malloc(256);
      heap.fill(0xab, M.state, M.state + 256);
      return 7;
    },
    _cs_whisper_transcribe() {
      return new Promise((resolve) => {
        M.device.lost.then(() => setTimeout(() => {
          M.woke = { heapIntact: heap.subarray(M.state, M.state + 256).every((v) => v === 0xab),
            abortRequested: M.czechspeechAbortRequested() };
          resolve(1);
        }, 20));
      });
    },
    async _cs_whisper_free() {},
  };
  return M;
}

async function startGpuEngine(options) {
  let M = null;
  globalThis.czechspeechFakeModule = (overrides) => (M = fakeGpuModule(overrides));
  const logs = [];
  const engine = await loadWasmEngine({ moduleUrl: FAKE_MODULE, threads: 1, gpu: true, abortRequested: () => false, options,
    log: (level, message) => logs.push(`${level}: ${message}`) });
  const handle = await engine.whisperInit({ size: MiB, read: () => new Uint8Array(0) });
  return { engine, M, handle, logs };
}

test('the module finds its .wasm through locateFile, under whatever name a bundler gave it', async () => {
  // the glue's own fallback, new URL("czechspeech-st.wasm", import.meta.url), is never used (scripts/fix-glue.mjs)
  let overrides = null;
  globalThis.czechspeechFakeModule = (o) => { overrides = o; return { HEAPU8: new Uint8Array(64), _cs_init() {}, _cs_whisper_language() {} }; };
  const engine = await loadWasmEngine({ moduleUrl: FAKE_MODULE, wasmUrl: 'https://app.example/assets/3f9a.wasm', threads: 4,
    log: () => {} });
  assert.equal(engine.backend, 'cpu');
  assert.equal(overrides.locateFile('czechspeech-st.wasm', 'https://app.example/assets/'), 'https://app.example/assets/3f9a.wasm');
  assert.equal(overrides.czechspeechPool, 5, 'the pthread pool: threads + 1');
});

test('a WebAssembly module older than the SDK (no _cs_whisper_language) fails at load and names the rebuild', async (t) => {
  installGpu(t);
  for (const gpu of [false, true]) {
    globalThis.czechspeechFakeModule = () => ({ HEAPU8: new Uint8Array(64), _cs_init() { throw new Error('must not be initialised'); } });
    await assert.rejects(loadWasmEngine({ moduleUrl: FAKE_MODULE, threads: 1, gpu, log: () => {} }),
      /WebAssembly module is older than the SDK.*native\/build-wasm\.sh/);
  }
});

test('a failed WebGPU module is told to stop, its device goes at once, its heap is zeroed only after the call ended', async (t) => {
  // regression (84a4557): the heap was zeroed under a call still suspended in the module, which shares the worker
  // with the CPU fallback; whisper.cpp was not told to stop, and the device stayed until the release
  const gpu = installGpu(t);
  const { engine, M, handle } = await startGpuEngine();
  const running = engine.whisperTranscribe(handle, 1, new Float32Array(16000).fill(0.1), 1);
  running.catch(() => {});
  await sleep(10);
  assert.equal(M.czechspeechAbortRequested(), false);
  engine.fail(new Error('ggml aborted: fake device error'));
  assert.equal(M.czechspeechAbortRequested(), true, 'whisper.cpp stops at its next abort check');
  assert.equal(gpu.devices[0].destroyed, true, 'the device is destroyed at once, so the suspended call wakes up');
  await assert.rejects(running, /ggml aborted: fake device error/);
  await assert.rejects(engine.whisperTranscribe(handle, 2, new Float32Array(10), 1), /ggml aborted/, 'later calls too');
  assert.equal(M.woke, null, 'the call is still running');
  await engine.wipe();
  assert.deepEqual(M.woke, { heapIntact: true, abortRequested: true }, 'the call ended on an intact heap');
  assert.ok(M.HEAPU8.every((v) => v === 0), 'then the heap was zeroed');
});

test('the WebGPU engine passes the languages as the trailing C string, frees it after a call, keeps it until wipe() after a failure', async (t) => {
  installGpu(t);
  const { engine, M, handle } = await startGpuEngine();
  Object.assign(M, { _cs_whisper_timings() {}, _cs_text_ptr: () => 0, _cs_text_len: () => 0 });
  const freed = [];
  M._free = (p) => freed.push(p);
  const cstring = (p) => { let e = p; while (M.HEAPU8[e] !== 0) e++; return Buffer.from(M.HEAPU8.subarray(p, e)).toString('utf8'); };
  const seen = [];
  M._cs_whisper_transcribe = async (...a) => { seen.push({ n: a.length, ptr: a[7], text: cstring(a[7]) }); return 0; };
  await engine.whisperTranscribe(handle, 1, new Float32Array(160), 1, {}, 'pl,en');
  assert.deepEqual(seen.map((s) => [s.n, s.text]), [[8, 'pl,en']], 'languages is the last argument');
  assert.ok(freed.includes(seen[0].ptr), 'freed after a normal return');
  assert.ok(M.HEAPU8.subarray(seen[0].ptr, seen[0].ptr + 6).every((v) => v === 0), 'and wiped');

  freed.length = 0;
  M._cs_whisper_transcribe = (...a) => {
    seen.push({ n: a.length, ptr: a[7], text: cstring(a[7]) });
    return new Promise(() => {});
  };
  const running = engine.whisperTranscribe(handle, 2, new Float32Array(160), 1, {}, 'cs');
  running.catch(() => {});
  await sleep(10);
  const failing = seen[1];
  assert.equal(failing.text, 'cs');
  engine.fail(new Error('ggml aborted: fake device error'));
  await assert.rejects(running, /ggml aborted/);
  assert.ok(!freed.includes(failing.ptr), 'not freed after a device failure: the suspended call may still read it');
  assert.equal(cstring(failing.ptr), 'cs', 'and not touched before wipe()');
  await engine.wipe();
  assert.ok(M.HEAPU8.every((v) => v === 0), 'wipe() zeroes the heap, string included');
});

test('a WebGPU recognition asked to stop ends at whisper.cpp\'s next abort check; the device stays, the next call runs', async (t) => {
  // the watchdog's first step (src/worker-core.js superviseCall): before, an overrunning call could only be failed,
  // which destroyed the device
  const gpu = installGpu(t);
  const { engine, M, handle } = await startGpuEngine();
  Object.assign(M, { _cs_whisper_timings() {}, _cs_text_ptr: () => 0, _cs_text_len: () => 0 });
  // whisper.cpp polls the abort flag between decoder steps; a call that is not stopped ends after 300 ms
  M._cs_whisper_transcribe = () => new Promise((resolve) => {
    const poll = setInterval(() => {
      if (M.czechspeechAbortRequested()) {
        clearInterval(poll);
        resolve(1);
      }
    }, 2);
    setTimeout(() => { clearInterval(poll); resolve(0); }, 300);
  });
  const running = engine.whisperTranscribe(handle, 1, new Float32Array(1600), 1);
  await sleep(20);
  assert.equal(M.czechspeechAbortRequested(), false);
  engine.stop();
  assert.equal(M.czechspeechAbortRequested(), true, 'whisper.cpp stops at its next abort check');
  const t0 = Date.now();
  assert.deepEqual(await running, { status: 1 });
  assert.ok(Date.now() - t0 < 200, 'at once, not at the end of the call');
  assert.equal(gpu.devices[0].destroyed, false, 'the device stays');
  assert.equal(engine.failure, null, 'the engine is not broken');
  assert.equal(M.czechspeechAbortRequested(), false, 'the request ends with its call');
  assert.equal((await engine.whisperTranscribe(handle, 2, new Float32Array(1600), 1)).status, 0, 'the next call runs to its end');
  engine.stop(); // between calls: nothing to stop, and the next call is not affected
  assert.equal((await engine.whisperTranscribe(handle, 3, new Float32Array(1600), 1)).status, 0);
});

test('the simulated device loss (test hook) hits the first recognition of the engine that has it; one without it recognises', async (t) => {
  // how many engines get the hook is decided by the recogniser (src/worker-core.js Backends, testFailRecognitionCount)
  installGpu(t);
  const first = await startGpuEngine({ testFailRecognitionMs: 20 });
  await assert.rejects(first.engine.whisperTranscribe(first.handle, 1, new Float32Array(160), 1), /simulated WebGPU device loss/);
  await first.engine.wipe();
  const second = await startGpuEngine({});
  Object.assign(second.M, { _cs_whisper_timings() {}, _cs_text_ptr: () => 0, _cs_text_len: () => 0,
    _cs_whisper_transcribe: async () => { await sleep(60); return 0; } });
  assert.equal((await second.engine.whisperTranscribe(second.handle, 1, new Float32Array(160), 1)).status, 0);
  assert.equal(second.engine.failure, null);
});

test('releasing a CPU engine tears the threaded module\'s runtime down, after its heap is zeroed', () => {
  // the threaded module's pthread workers and its main thread's mailbox wait keep its shared memory (about 1.2 GB with
  // the model) reachable (native/threads-teardown.js); before, wipe() only zeroed the heap, and every WebGPU failure
  // and recovery left one such module behind. That the memory is then collected: test/integration/teardown.test.mjs
  const log = [];
  const heap = new Uint8Array(4096).fill(7);
  const M = { HEAPU8: heap, czechspeechTerminateRuntime: () => log.push(heap.every((v) => v === 0) ? 'teardown after zeroing' : 'teardown before zeroing') };
  const engine = new WasmEngine(M, () => {});
  engine.wipe();
  engine.wipe();
  assert.deepEqual(log, ['teardown after zeroing'], 'once');
  const single = new WasmEngine({ HEAPU8: new Uint8Array(64) }, () => {});
  single.wipe(); // the single-threaded module has no threads: nothing to tear down
});

test('each WebGPU engine has its own device: releasing a failed one later leaves the next one alone', async (t) => {
  // regression (84a4557): one GPU tracker per worker, reset by the next engine, so the late release of a failed engine
  // destroyed the device and buffers of the engine that replaced it
  const gpu = installGpu(t);
  const first = await startGpuEngine();
  first.engine.fail(new Error('WebGPU device lost'));
  const second = await startGpuEngine();
  assert.equal(gpu.devices.length, 2);
  const gpuBytes = second.engine.gpuBytes;
  assert.ok(gpuBytes > MiB);
  await first.engine.wipe();
  assert.equal(gpu.devices[1].destroyed, false, 'the second engine\'s device survives the first one\'s release');
  assert.ok(gpu.devices[1].buffers.every((b) => !b.destroyed));
  assert.equal(second.engine.gpuBytes, gpuBytes);
  assert.deepEqual(second.engine.adapter, { vendor: 'fakevendor', architecture: 'fakearch', device: '', description: '' });
  await second.engine.wipe();
  assert.equal(gpu.devices[1].destroyed, true);
});

test('releasing a healthy WebGPU engine clears its buffers on the GPU, destroys them and the device, zeroes the heap', async (t) => {
  const gpu = installGpu(t);
  const { engine, M, handle, logs } = await startGpuEngine();
  await engine.whisperFree(handle);
  await engine.wipe();
  const { log } = gpu.devices[0];
  const lastClear = log.map((l) => l.startsWith('clear')).lastIndexOf(true);
  const firstDestroy = log.findIndex((l) => l.startsWith('destroy'));
  assert.ok(lastClear >= 0 && lastClear < firstDestroy, log.join(', '));
  assert.ok(log.includes(`clear ${MiB}`) && log.includes('clear 4096'), 'the buffers the GPU can copy into are cleared');
  assert.equal(log.at(-1), 'destroy device');
  assert.ok(gpu.devices[0].buffers.every((b) => b.destroyed));
  assert.equal(engine.gpuBytes, 0);
  assert.ok(M.HEAPU8.every((v) => v === 0));
  assert.ok(logs.some((l) => /GPU buffers cleared \(\d+ MiB\) and released, heap zeroed/.test(l)), logs.join('\n'));
});

test('the CPU engine passes the languages as the trailing C string, frees it after the call (also when it throws), and returns the language of the text', () => {
  const heap = new Uint8Array(1 << 16);
  let next = 1024;
  const freed = [];
  const seen = [];
  const cstring = (p) => { let e = p; while (heap[e] !== 0) e++; return Buffer.from(heap.subarray(p, e)).toString('utf8'); };
  const M = {
    HEAPU8: heap,
    _malloc: (n) => { const p = next; next += (n + 15) & ~15; return p; },
    _free: (p) => freed.push(p),
    _cs_zero_free: (p, n) => { heap.fill(0, p, p + n); },
    _cs_whisper_tokens: () => 0,
    _cs_whisper_capped: () => 0,
    _cs_whisper_transcribe: (...a) => { seen.push({ n: a.length, ptr: a[7], text: cstring(a[7]) }); return 0; },
    _cs_whisper_timings() {},
    _cs_text_ptr: () => 0,
    _cs_text_len: () => 0,
    _cs_text_release() {},
    _cs_whisper_language: () => 2048,
    UTF8ToString: (p) => (p === 2048 ? 'pl' : ''),
  };
  heap.set(new TextEncoder().encode('pl\0'), 2048);
  const engine = new WasmEngine(M, () => {});
  const result = engine.whisperTranscribe(1, 1, new Float32Array(160), 1, {}, 'pl,en');
  assert.deepEqual(seen.map((s) => [s.n, s.text]), [[8, 'pl,en']], 'languages is the last argument');
  assert.equal(result.language, 'pl');
  assert.ok(freed.includes(seen[0].ptr), 'freed after the call');
  assert.ok(heap.subarray(seen[0].ptr, seen[0].ptr + 6).every((v) => v === 0), 'and wiped');

  freed.length = 0;
  M._cs_whisper_transcribe = (...a) => { seen.push({ ptr: a[7] }); throw new Error('boom'); };
  assert.throws(() => engine.whisperTranscribe(1, 2, new Float32Array(160), 1, {}, 'cs'), /boom/);
  assert.ok(freed.includes(seen.at(-1).ptr), 'freed when the call throws');
});
