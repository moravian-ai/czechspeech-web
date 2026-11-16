// Test inputs come only from the environment (no defaults, nothing machine-specific in the repository); tests that
// need one skip with a message naming the variable when it is unset or points nowhere. Same rules as the Android
// SDK's TestSupport.kt.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Worker } from 'node:worker_threads';

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
export const SAMPLE_RATE = 16000;
export const FRAME_SAMPLES = 512;

/** Windows paths ("C:\...") become /mnt/c/... when the tests run in WSL. */
export function localPath(p) {
  const m = /^([A-Za-z]):[\\/](.*)$/.exec(p);
  if (!m || process.platform === 'win32') return p;
  return `/mnt/${m[1].toLowerCase()}/${m[2].replace(/\\/g, '/')}`;
}

function fromEnv(name) {
  const v = process.env[name];
  return v && v.trim() ? localPath(v.trim()) : null;
}

/**
 * Wall-clock limits of the tests (how long a cancellation may take, how long to wait for a worker) are multiplied by
 * CZECHSPEECH_TEST_TIME_SCALE, default 1: on a slow or loaded machine (a CI runner with two cores) set e.g. 3. The limits
 * check that nothing waits for what it should not, not how fast the machine is.
 */
export const TIME_SCALE = (() => {
  const v = Number(process.env.CZECHSPEECH_TEST_TIME_SCALE);
  return Number.isFinite(v) && v > 0 ? v : 1;
})();
export const within = (ms) => ms * TIME_SCALE;

export const TestEnv = {
  get model() { return fromEnv('CZECHSPEECH_MODEL'); },
  get audioDir() { return fromEnv('CZECHSPEECH_TEST_AUDIO'); },
  get manifest() { return fromEnv('CZECHSPEECH_TEST_MANIFEST'); },
  get threads() {
    const n = Number.parseInt(process.env.CZECHSPEECH_TEST_THREADS || '', 10);
    return Number.isFinite(n) ? Math.min(8, Math.max(1, n)) : 6;
  },
  vadAsset: path.join(ROOT, 'src/assets/ggml-silero-v6.2.0.bin'),
};

/** Reason to skip, or null: names the variable when it is unset, the path when it does not exist. */
export function missing(name, p, what) {
  if (!p) return `${what}: ${name} not set`;
  if (!fs.existsSync(p)) return `${what} not found at ${p} (${name})`;
  return null;
}

/** Minimal PCM WAV reader: 16-bit or 32-bit float, mono, 16 kHz. */
export function readWav(file) {
  const b = fs.readFileSync(file);
  if (b.toString('ascii', 0, 4) !== 'RIFF' || b.toString('ascii', 8, 12) !== 'WAVE') throw new Error(`${file} is not a WAV file`);
  let pos = 12;
  let format = 0;
  let channels = 0;
  let rate = 0;
  let bits = 0;
  while (pos + 8 <= b.length) {
    const id = b.toString('ascii', pos, pos + 4);
    const size = b.readUInt32LE(pos + 4);
    const body = pos + 8;
    if (id === 'fmt ') {
      format = b.readUInt16LE(body);
      channels = b.readUInt16LE(body + 2);
      rate = b.readUInt32LE(body + 4);
      bits = b.readUInt16LE(body + 14);
    } else if (id === 'data') {
      if (channels !== 1 || rate !== SAMPLE_RATE) throw new Error(`${file}: need 16 kHz mono, got ${rate} Hz x ${channels}`);
      const n = Math.min(size, b.length - body);
      if (format === 1 && bits === 16) {
        const out = new Float32Array(n / 2);
        for (let i = 0; i < out.length; i++) out[i] = b.readInt16LE(body + 2 * i) / 32768;
        return out;
      }
      if (format === 3 && bits === 32) {
        const out = new Float32Array(n / 4);
        for (let i = 0; i < out.length; i++) out[i] = b.readFloatLE(body + 4 * i);
        return out;
      }
      throw new Error(`${file}: unsupported WAV format ${format}/${bits} bit`);
    }
    pos = body + size + (size & 1);
  }
  throw new Error(`${file}: no data chunk`);
}

/** The reference transcript of a ParCzech segment, from the manifest. */
export function reference(manifest, id) {
  const line = fs.readFileSync(manifest, 'utf8').split('\n')
    .find((l) => l.includes(`"id": "${id}"`) || l.includes(`"id":"${id}"`));
  if (!line) return null;
  return JSON.parse(line).ref;
}

export const Signals = {
  zeros: (seconds) => new Float32Array(Math.round(seconds * SAMPLE_RATE)),
  tone(seconds, hz = 220, amplitude = 0.3) {
    const out = new Float32Array(Math.round(seconds * SAMPLE_RATE));
    for (let i = 0; i < out.length; i++) out[i] = amplitude * Math.sin((2 * Math.PI * hz * i) / SAMPLE_RATE);
    return out;
  },
  whiteNoise(seconds, amplitude = 0.1, seed = 1) {
    let s = seed >>> 0 || 1;
    const out = new Float32Array(Math.round(seconds * SAMPLE_RATE));
    for (let i = 0; i < out.length; i++) {
      s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0;
      out[i] = amplitude * ((s / 0xffffffff) * 2 - 1);
    }
    return out;
  },
  concat(...parts) {
    const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
    let pos = 0;
    for (const p of parts) {
      out.set(p, pos);
      pos += p.length;
    }
    return out;
  },
};

/** Lower-case words without punctuation (the normalisation of wordRecall, wer and wordDiff). */
export const words = (s) => String(s).toLowerCase().split(/[^\p{L}\p{N}]+/u).filter(Boolean);

/** Word-level edit alignment of b against a: { distance, ops: [{ op: 'sub' | 'del' | 'ins', a?, b? }] }. */
export function wordAlign(a, b) {
  const n = a.length;
  const m = b.length;
  const d = Array.from({ length: n + 1 }, (_, i) => Int32Array.from({ length: m + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : 0)));
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
  }
  const ops = [];
  for (let i = n, j = m; i > 0 || j > 0;) {
    if (i > 0 && j > 0 && d[i][j] === d[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1)) {
      if (a[i - 1] !== b[j - 1]) ops.push({ op: 'sub', a: a[i - 1], b: b[j - 1] });
      i--; j--;
    } else if (i > 0 && d[i][j] === d[i - 1][j] + 1) {
      ops.push({ op: 'del', a: a[i - 1] });
      i--;
    } else {
      ops.push({ op: 'ins', b: b[j - 1] });
      j--;
    }
  }
  return { distance: d[n][m], ops: ops.reverse() };
}

/** Word error rate of hyp against ref (substitutions + deletions + insertions over reference words). */
export function wer(ref, hyp) {
  const r = words(ref);
  const { distance } = wordAlign(r, words(hyp));
  return r.length === 0 ? (words(hyp).length ? 1 : 0) : distance / r.length;
}

/** Word differences between two transcripts: { same, ops, summary } (e.g. "a -> b, -c, +d"). */
export function wordDiff(a, b) {
  const { ops } = wordAlign(words(a), words(b));
  const summary = ops.map((o) => (o.op === 'sub' ? `${o.a} -> ${o.b}` : o.op === 'del' ? `-${o.a}` : `+${o.b}`)).join(', ');
  return { same: ops.length === 0, ops, summary };
}

/** Share of reference words found in the hypothesis (case and punctuation ignored, multiset). */
export function wordRecall(ref, hyp) {
  const r = words(ref);
  const counts = new Map();
  for (const w of words(hyp)) counts.set(w, (counts.get(w) || 0) + 1);
  let hit = 0;
  for (const w of r) {
    const c = counts.get(w) || 0;
    if (c > 0) {
      hit++;
      counts.set(w, c - 1);
    }
  }
  return r.length === 0 ? 1 : hit / r.length;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * Plays a Float32Array as 512-sample frames through a MessagePort (in place of the microphone), then sends 'end',
 * or keeps sending `tail` frames until closed. The last partial frame is zero-padded. finish() ends it like the
 * microphone's finish(): no frame after the current one, then 'end' on the same port. `onFrame(n)` runs after the
 * n-th frame was sent (e.g. to call stop() at a known point of the audio); when it returns a promise, no further
 * frame is sent before it settles.
 */
export class ArrayAudioSource {
  constructor(samples, { tail = null, frameDelayMs = 0, failOpen = null, openDelayMs = 0, onFrame = null } = {}) {
    this.samples = samples;
    this.tail = tail;
    this.frameDelayMs = frameDelayMs;
    this.failOpen = failOpen;
    this.openDelayMs = openDelayMs;
    this.onFrame = onFrame;
    this.opened = false;
    this.closed = false;
    this.ended = false;
    this.finishCalls = 0;
    this.framesSent = 0;
    this.finishRequested = false;
    this.onFinished = null;
  }

  async open() {
    if (this.openDelayMs > 0) await sleep(this.openDelayMs);
    if (this.failOpen) throw this.failOpen;
    const channel = new MessageChannel();
    this.opened = true;
    this.port = channel.port1;
    this.#pump();
    return {
      port: channel.port2,
      sampleRate: SAMPLE_RATE,
      close: () => {
        this.closed = true;
        try { this.port.close(); } catch { /* closed */ }
        if (this.onFinished) this.onFinished(false);
      },
      finish: () => {
        this.finishCalls++;
        if (this.closed || this.ended) return Promise.resolve(false);
        this.finishRequested = true;
        return new Promise((resolve) => { this.onFinished = resolve; });
      },
    };
  }

  #end() {
    this.port.postMessage({ type: 'end' });
    this.ended = true;
    if (this.onFinished) this.onFinished(true);
  }

  async #pump() {
    let tailIndex = 0;
    for (let pos = 0; ; pos += FRAME_SAMPLES) {
      if (this.closed) return;
      if (this.finishRequested) {
        this.#end();
        return;
      }
      const frame = new Float32Array(FRAME_SAMPLES);
      if (pos < this.samples.length) {
        frame.set(this.samples.subarray(pos, Math.min(pos + FRAME_SAMPLES, this.samples.length)));
      } else if (this.tail) {
        for (let i = 0; i < FRAME_SAMPLES; i++) frame[i] = this.tail(tailIndex++);
      } else {
        this.#end();
        return;
      }
      this.port.postMessage(frame, [frame.buffer]);
      this.framesSent++;
      if (this.onFrame) await this.onFrame(this.framesSent);
      if (this.frameDelayMs > 0) await sleep(this.frameDelayMs);
      else if (this.framesSent % 64 === 0) await sleep(0);
    }
  }
}

/** A Web-Worker-like object over node:worker_threads, running `entry` (a worker module using self.onmessage). */
export function nodeWorkerFactory(entry) {
  const bootstrap = new URL('./node-worker-bootstrap.mjs', import.meta.url);
  const created = [];
  const factory = (role) => {
    const w = new Worker(bootstrap, { workerData: { entry: String(entry) }, name: `czechspeech-${role}` });
    const wrapper = {
      role,
      terminated: false,
      onmessage: null,
      onerror: null,
      onmessageerror: null,
      postMessage(msg, transfer) { w.postMessage(msg, transfer); },
      terminate() {
        wrapper.terminated = true;
        w.terminate();
      },
    };
    w.on('message', (data) => wrapper.onmessage && wrapper.onmessage({ data }));
    w.on('error', (error) => wrapper.onerror && wrapper.onerror({ message: error.message, error, preventDefault() {} }));
    created.push(wrapper);
    return wrapper;
  };
  factory.created = created;
  return factory;
}
