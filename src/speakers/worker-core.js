// Message handling of the speaker worker. The embedder (onnxruntime-web, or a fake in the unit tests) is injected by
// the entry module (src/speakers/worker.js), so this file has no environment dependencies.
//
//   page -> worker:  init { model: Blob, runtime: Blob, threads, point, maxInputMs }
//                    embed { id, samples }                    one embedding of the whole array (parity and benchmark tests)
//                    utterance { id, samples, listenId }      one utterance (listenId 0: asked by the page directly)
//                    attach { port, listenId }                a port from the capture worker; messages on it are `utterance`
//                    detach { listenId }                      closes that port
//                    close
//   worker -> page:  ready { threads }, embedded { id, vector }, pieces { listenId, id, mode, pieces, computeMs },
//                    (audioStartMs: echoed from the utterance, so the page can put the pieces on its clock)
//                    closed, fatal { message }
//                    One request that fails is answered with `embedded { id, error }` / `pieces { listenId, id, error }`
//                    (error: a string) and the worker goes on; `fatal` is for what ends the worker's life.
//
// Utterances are handled one at a time, oldest first. When a backlog builds up the worker degrades instead of falling
// behind: with SKIP_AT or more utterances waiting behind the one being started it computes nothing (`skipped`), with
// WHOLE_AT or more it embeds the utterance as a whole instead of looking for speaker changes (`whole`).
// One embedding is never computed from more than `maxInputMs` of audio (default MAX_INPUT_MS): the activations of the
// network grow linearly with the length and WebAssembly memory never shrinks. A longer array is cut into the fewest equal
// chunks that fit, and the unit embeddings of the chunks are averaged (and normalised again).
// Every array that held audio is zeroed as soon as it has been used or dropped. Nothing is logged: no audio, no
// embedding and no speaker data ever leaves this file except as the messages above.
import { dot, changePoints, validCuts, windowStarts, unit } from './maths.js';

export const WHOLE_AT = 2;
export const SKIP_AT = 5;
export const MAX_INPUT_MS = 10000;

const SAMPLE_RATE = 16000;
// The real embedder (onnxruntime-web) computes inside synchronous WebAssembly calls: its promises settle through
// microtasks alone, so no message (a new utterance, close) is handled while it keeps going. A task-queue turn between
// two computations lets those messages in; without it the backlog is never seen and close() cannot interrupt.
// A MessageChannel message is a task of its own that is neither clamped to 4 ms (nested timers) nor throttled in a hidden
// tab (timers are); a timer is the fallback where there is no MessageChannel.
const yieldToMessages = () => new Promise((resolve) => {
  if (typeof MessageChannel === 'function') {
    try {
      const { port1, port2 } = new MessageChannel();
      port1.onmessage = () => {
        port1.onmessage = null;
        port1.close();
        resolve();
      };
      port2.postMessage(0);
      return;
    } catch { /* fall through to the timer */ }
  }
  setTimeout(resolve, 0);
});
const MIN_SAMPLES = 400; // one 25 ms fbank frame

/** Why `samples` cannot be embedded, or null. Anything can arrive on a port. */
function badSamples(samples) {
  if (!(samples instanceof Float32Array)) return 'audio must be a Float32Array';
  if (samples.length < MIN_SAMPLES) return `audio must have at least ${MIN_SAMPLES} samples`;
  return null;
}
const WINDOW_AHEAD = 3; // windows between the two windows compared: 1.5 s window, 0.5 s hop -> 3

/**
 * @param {object} scope   the worker global (self) or an equivalent with postMessage/onmessage
 * @param {object} deps
 * @param {(o: { model: Blob, runtime: Blob, threads: number }) => Promise<{ embed(samples: Float32Array): Promise<Float32Array> | Float32Array, release(): void, threads?: number }>} deps.loadEmbedder
 */
export function installSpeakerWorker(scope, { loadEmbedder }) {
  let embedder = null;
  let point = null;
  let maxInput = Math.round((MAX_INPUT_MS * SAMPLE_RATE) / 1000); // samples
  let initializing = null;
  let closing = false;
  const queue = [];       // utterances waiting, oldest first
  let current = null;     // the utterance being handled
  let pumping = false;
  const ports = new Map(); // listenId -> port (the page's internal channel id: never reused, so one id is one attachment)

  const post = (msg, transfer) => scope.postMessage(msg, transfer || []);
  const message = (err) => String((err && err.message) || err || 'failed');
  const fatal = (err) => post({ type: 'fatal', message: String((err && err.message) || err || 'speaker worker failed') });

  scope.onmessage = (e) => {
    const msg = e.data;
    if (!msg) return;
    if (msg.type === 'init') {
      initializing = init(msg).catch(fatal);
      return;
    }
    const run = () => {
      try {
        handle(msg);
      } catch (err) {
        fatal(err);
      }
    };
    if (initializing && !embedder) initializing.then(run);
    else run();
  };

  async function init(msg) {
    point = msg.point;
    if (msg.maxInputMs > 0) {
      // a window of the change search is embedded in one go, so the cap is never lower than it
      maxInput = Math.round((Math.max(msg.maxInputMs, point?.splitWindowMs || 0) * SAMPLE_RATE) / 1000);
    }
    embedder = await loadEmbedder({ model: msg.model, runtime: msg.runtime, threads: msg.threads });
    post({ type: 'ready', threads: embedder.threads ?? msg.threads });
  }

  function handle(msg) {
    if (closing && msg.type !== 'close') return drop(msg);
    switch (msg.type) {
      case 'embed': return void embedOne(msg);
      case 'utterance': {
        const bad = badSamples(msg.samples);
        if (bad) {
          drop(msg);
          post({ type: 'pieces', listenId: msg.listenId, id: msg.id, error: bad });
          return;
        }
        return void enqueue(msg);
      }
      case 'attach': return void attach(msg);
      case 'detach': return void detach(msg);
      case 'close': return void close();
      default: return drop(msg);
    }
  }

  function drop(msg) {
    if (msg && msg.samples && typeof msg.samples.fill === 'function') {
      try { msg.samples.fill(0); } catch { /* not ours to zero */ }
    }
    if (msg && msg.port) try { msg.port.close(); } catch { /* already closed */ }
  }

  function detach({ listenId }) {
    const port = ports.get(listenId);
    if (!port) return;
    ports.delete(listenId);
    port.onmessage = null;
    try { port.close(); } catch { /* already closed */ }
    // what that listener has queued is dropped (and zeroed); the utterance being handled stops at its next check
    for (let i = queue.length - 1; i >= 0; i--) {
      if (queue[i].listenId !== listenId) continue;
      queue[i].samples.fill(0);
      queue.splice(i, 1);
    }
    if (current && current.listenId === listenId) current.aborted = true;
  }

  function attach({ port, listenId }) {
    ports.set(listenId, port);
    port.onmessage = (e) => {
      const m = e.data;
      if (!m || m.type !== 'utterance') return;
      try {
        if (closing || !embedder || ports.get(listenId) !== port) return drop(m);
        const bad = badSamples(m.samples);
        if (bad) {
          drop(m);
          post({ type: 'pieces', listenId, id: m.id, error: bad });
          return;
        }
        enqueue({ ...m, listenId });
      } catch (err) {
        fatal(err);
      }
    };
  }

  async function embedOne({ id, samples }) {
    const emb = embedder;
    const bad = badSamples(samples);
    if (bad) {
      drop({ samples });
      if (!closing) post({ type: 'embedded', id, error: bad });
      return;
    }
    try {
      const vector = await embedCapped(emb, samples, () => closing);
      if (!closing) post({ type: 'embedded', id, vector }, [vector.buffer]);
      else vector.fill(0);
    } catch (err) {
      if (!closing) post({ type: 'embedded', id, error: message(err) });
    } finally {
      samples.fill(0);
    }
  }

  function enqueue(msg) {
    queue.push(msg);
    if (!pumping) void pump();
  }

  async function pump() {
    pumping = true;
    try {
      while (queue.length && !closing) {
        current = queue.shift();
        const job = current;
        const waiting = queue.length; // behind the utterance being started
        try {
          await handleUtterance(job, waiting, embedder);
        } catch (err) {
          if (!closing && !job.aborted) post({ type: 'pieces', listenId: job.listenId, id: job.id, error: message(err) });
        } finally {
          drop(job); // robust: zeroes what can be zeroed, never throws
          current = null;
        }
        await yieldToMessages(); // messages that arrived meanwhile are queued (or close) before the next one starts
      }
    } finally {
      pumping = false;
    }
  }

  async function handleUtterance(job, waiting, emb) {
    const { id, samples, listenId, audioStartMs } = job;
    const stop = () => closing || job.aborted === true; // the session is released, or this listener is gone
    const t0 = performance.now();
    const durMs = Math.round((samples.length * 1000) / SAMPLE_RATE);
    let mode = waiting >= SKIP_AT ? 'skipped' : waiting >= WHOLE_AT ? 'whole' : 'split';
    const pieces = [];
    let sent = false;
    try {
      if (mode === 'split' && samples.length < 2 * Math.round((point.splitWindowMs * SAMPLE_RATE) / 1000)) mode = 'whole';
      if (mode === 'split') {
        const cuts = await findCuts(samples, durMs, emb, stop);
        if (stop()) return; // no more computing
        if (cuts.length === 0) {
          pieces.push({ startMs: 0, endMs: durMs, vector: await embedCapped(emb, samples, stop) });
        } else {
          const bounds = [0, ...cuts, durMs];
          for (let i = 0; i + 1 < bounds.length; i++) {
            await yieldToMessages();
            if (stop()) return;
            const a = Math.round((bounds[i] * SAMPLE_RATE) / 1000);
            const b = Math.round((bounds[i + 1] * SAMPLE_RATE) / 1000);
            pieces.push({ startMs: bounds[i], endMs: bounds[i + 1], vector: await embedCapped(emb, samples.subarray(a, b), stop) });
          }
        }
      } else if (mode === 'whole') {
        pieces.push({ startMs: 0, endMs: durMs, vector: await embedCapped(emb, samples, stop) });
      }
      if (stop()) return;
      const buffers = pieces.map((p) => p.vector.buffer);
      post({ type: 'pieces', listenId, id, mode, pieces, computeMs: Math.round(performance.now() - t0), audioStartMs }, buffers);
      sent = true;
    } finally {
      // whatever was not sent (a closing return, a throw, an aborted listener) is zeroed; sent buffers are detached
      if (!sent) for (const p of pieces) p.vector.fill(0);
    }
  }

  /** One embedding of `samples`, from chunks of at most maxInput samples when it is longer. `stop()` ends it with an error. */
  async function embedCapped(emb, samples, stop) {
    if (samples.length <= maxInput) return emb.embed(samples);
    const n = Math.ceil(samples.length / maxInput);
    const size = Math.ceil(samples.length / n);
    let sum = null;
    try {
      for (let i = 0; i < n; i++) {
        await yieldToMessages();
        if (stop()) throw new Error('closed');
        const v = await emb.embed(samples.subarray(i * size, Math.min(samples.length, (i + 1) * size)));
        if (!sum) sum = new Float32Array(v.length);
        for (let k = 0; k < v.length; k++) sum[k] += v[k];
        v.fill(0);
      }
      return unit(sum);
    } finally {
      if (sum) sum.fill(0);
    }
  }

  async function findCuts(samples, durMs, emb, stop) {
    const win = Math.round((point.splitWindowMs * SAMPLE_RATE) / 1000);
    const hop = Math.round((point.splitHopMs * SAMPLE_RATE) / 1000);
    const starts = windowStarts(samples.length, win, hop);
    const vectors = [];
    try {
      for (const s of starts) {
        await yieldToMessages();
        if (stop()) return [];
        vectors.push(await emb.embed(samples.subarray(s, s + win)));
      }
      if (stop()) return [];
      const scores = [];
      for (let i = 0; i + WINDOW_AHEAD < vectors.length; i++) scores.push(1 - dot(vectors[i], vectors[i + WINDOW_AHEAD]));
      const cuts = changePoints(scores, { firstMs: point.splitWindowMs, hopMs: point.splitHopMs, winMs: point.splitWindowMs, drop: point.drop });
      return validCuts(cuts, durMs, point.minMs);
    } finally {
      for (const v of vectors) v.fill(0);
    }
  }

  function close() {
    closing = true;
    for (const m of queue.splice(0)) m.samples.fill(0);
    if (current) current.samples.fill(0);
    for (const p of ports.values()) {
      p.onmessage = null;
      try { p.close(); } catch { /* already closed */ }
    }
    ports.clear();
    try {
      if (embedder) embedder.release();
    } catch (err) {
      fatal(err);
    }
    embedder = null;
    post({ type: 'closed' });
  }
}
