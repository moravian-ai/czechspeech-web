// Runs a worker module written for Web Workers (self.onmessage / self.postMessage) in a node:worker_threads thread.
import { parentPort, workerData } from 'node:worker_threads';

globalThis.self = globalThis;
globalThis.postMessage = (msg, transfer) => parentPort.postMessage(msg, transfer);
parentPort.on('message', (data) => {
  if (typeof globalThis.onmessage === 'function') globalThis.onmessage({ data });
});
await import(workerData.entry);
