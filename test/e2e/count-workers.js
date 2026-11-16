// Counts the Workers this worker creates and terminates (imported first by counting-worker.js, before the SDK's worker
// entry, so that the threaded CPU module's pthread workers are counted). Every change is reported to the page as
// { type: 'workers', created, terminated }.
//
// It also keeps a WeakRef to every shared WebAssembly.Memory created here: the threaded CPU module's memory (about
// 1.2 GB with the model loaded). Asked on the BroadcastChannel 'czechspeech-e2e-memories' with { type: 'check', id },
// it runs full garbage collections (gc(): the browser is started with --js-flags=--expose-gc) and answers
// { type: 'memories', id, worker, created, live, liveBytes }: how many of those memories are still reachable from
// anywhere in this worker. A released module whose memory stays reachable (by the module's own closures, for example)
// stays in `live`.
const NativeWorker = globalThis.Worker;
let created = 0;
let terminated = 0;
const report = () => self.postMessage({ type: 'workers', created, terminated });

globalThis.Worker = class CountedWorker extends NativeWorker {
  constructor(...args) {
    super(...args);
    created++;
    report();
  }

  terminate() {
    if (!this.terminatedOnce) {
      this.terminatedOnce = true;
      terminated++;
      report();
    }
    return super.terminate();
  }
};

const memories = [];
const NativeMemory = WebAssembly.Memory;
WebAssembly.Memory = new Proxy(NativeMemory, {
  construct(target, args) {
    const memory = Reflect.construct(target, args);
    if (args[0] && args[0].shared) memories.push(new WeakRef(memory));
    return memory;
  },
});

const nextTask = () => new Promise((resolve) => setTimeout(resolve, 0));
const channel = new BroadcastChannel('czechspeech-e2e-memories');
channel.onmessage = async (e) => {
  if (!e.data || e.data.type !== 'check') return;
  const reply = { type: 'memories', id: e.data.id, worker: self.name };
  if (typeof globalThis.gc !== 'function') {
    channel.postMessage({ ...reply, error: 'gc() is not exposed (start the browser with --js-flags=--expose-gc)' });
    return;
  }
  // Several rounds, each in a task of its own: a WeakRef keeps its target until the end of the task that read it, and
  // what the collection depends on may run in between (a notified Atomics.waitAsync settles in a task).
  for (let i = 0; i < 5; i++) {
    await nextTask();
    globalThis.gc();
  }
  await nextTask();
  const live = memories.map((r) => r.deref()).filter(Boolean);
  channel.postMessage({ ...reply, created: memories.length, live: live.length,
    liveBytes: live.reduce((sum, m) => sum + m.buffer.byteLength, 0) });
};
