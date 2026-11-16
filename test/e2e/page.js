// Browser side of the end-to-end tests: functions that test/e2e/run.mjs calls through page.evaluate().
import { CzechSpeech, CzechSpeechModel } from '../../src/index.js';

let model = null;
let speech = null;
const logs = [];
const streams = [];
let captured = []; // captureMic(): 16 kHz frames of the microphone, in memory only
let workerCounts = {}; // createCounting(): Workers created and terminated by each worker
let memoryChecks = 0; // memories(): ids of the questions to the recogniser worker
let gpuChecks = 0; // gpuBuffers(): ids of the questions to the recogniser worker
const gpuDestroyed = []; // what gpu-bytes.js reported when a device was destroyed
const gpuChannel = new BroadcastChannel('czechspeech-e2e-gpu');
gpuChannel.addEventListener('message', (e) => { if (e.data && e.data.type === 'gpu-destroyed') gpuDestroyed.push(e.data); });

// Keep the streams the SDK opens, to check afterwards that their tracks were stopped.
const gum = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
navigator.mediaDevices.getUserMedia = async (constraints) => {
  const s = await gum(constraints);
  streams.push({ stream: s, constraints });
  return s;
};

function parseWav(buf) {
  const v = new DataView(buf);
  let pos = 12;
  while (pos + 8 <= v.byteLength) {
    const id = String.fromCharCode(v.getUint8(pos), v.getUint8(pos + 1), v.getUint8(pos + 2), v.getUint8(pos + 3));
    const size = v.getUint32(pos + 4, true);
    if (id === 'data') {
      const out = new Float32Array(size / 2);
      for (let i = 0; i < out.length; i++) out[i] = v.getInt16(pos + 8 + 2 * i, true) / 32768;
      return out;
    }
    pos += 8 + size + (size & 1);
  }
  throw new Error('no data chunk');
}

async function transcribeSamples(samples) {
  const copy = samples.slice();
  const t0 = performance.now();
  const text = await speech.transcribe(samples);
  const ms = Math.round(performance.now() - t0);
  const unchanged = samples.every((v, i) => v === copy[i]);
  return { text, ms, audioMs: Math.round((samples.length / 16000) * 1000), unchanged, state: speech.state, logs: [...logs] };
}

window.e2e = {
  ready: true,

  info: () => ({
    crossOriginIsolated: self.crossOriginIsolated,
    sharedArrayBuffer: typeof SharedArrayBuffer === 'function',
    hardwareConcurrency: navigator.hardwareConcurrency,
    userAgent: navigator.userAgent,
  }),

  async model({ url, sha256, size, languages }) {
    const t0 = performance.now();
    let lastProgress = null;
    model = await CzechSpeechModel.fromUrl(url, { sha256, size, ...(languages ? { languages } : {}), onProgress: (d, t) => { lastProgress = [d, t]; } });
    return { ms: Math.round(performance.now() - t0), cached: model.cached, size: model.size, sha256: model.sha256, lastProgress,
      isDownloaded: await CzechSpeechModel.isDownloaded({ url, sha256, size }) };
  },

  /** A new CzechSpeech (the previous one is closed). defaultLog: true keeps the SDK's default console logger. */
  async create(options = {}) {
    if (speech) await speech.close();
    logs.length = 0;
    const t0 = performance.now();
    const { defaultLog, ...rest } = options;
    const log = defaultLog ? undefined : (level, message) => logs.push(`${level}: ${message}`);
    speech = await CzechSpeech.create({ model, ...(log ? { log } : {}), ...rest });
    return { ms: Math.round(performance.now() - t0), state: speech.state };
  },

  /**
   * A new CzechSpeech whose workers count the Workers they create and terminate (counting-worker.js): workerCounts()
   * then gives { capture, recognizer } as { created, terminated }.
   */
  async createCounting(options = {}) {
    if (speech) await speech.close();
    logs.length = 0;
    workerCounts = {};
    speech = await CzechSpeech.create({
      model, log: (level, message) => logs.push(`${level}: ${message}`), ...options,
      _assets: { worker: new URL('./counting-worker.js', import.meta.url).href },
      _onWorkerMessage: (role, m) => { if (m.type === 'workers') workerCounts[role] = { created: m.created, terminated: m.terminated }; },
    });
    return { state: speech.state };
  },
  workerCounts: () => ({ ...workerCounts }),

  /**
   * (test/e2e/detect-cost.mjs only) A new CzechSpeech whose recogniser worker is gpu-bytes-worker.js: it counts the GPU
   * buffers the worker creates and destroys, which gpuBuffers() reads. The SDK itself reports gpuBytes only when the
   * model is loaded.
   */
  async createGpuTracked(options = {}) {
    if (speech) await speech.close();
    logs.length = 0;
    gpuDestroyed.length = 0;
    speech = await CzechSpeech.create({
      model, log: (level, message) => logs.push(`${level}: ${message}`), ...options,
      _assets: { worker: new URL('./gpu-bytes-worker.js', import.meta.url).href },
    });
    return { state: speech.state };
  },

  /** The live GPU buffers of the recogniser worker (gpu-bytes.js), or an error when it does not answer. */
  async gpuBuffers(timeoutMs = 10000) {
    const id = ++gpuChecks;
    return new Promise((resolve) => {
      const timer = setTimeout(() => { gpuChannel.removeEventListener('message', on); resolve({ error: 'no answer' }); }, timeoutMs);
      const on = (e) => {
        if (!e.data || e.data.type !== 'gpu' || e.data.id !== id || e.data.worker !== 'czechspeech-recognizer') return;
        clearTimeout(timer);
        gpuChannel.removeEventListener('message', on);
        resolve(e.data);
      };
      gpuChannel.addEventListener('message', on);
      gpuChannel.postMessage({ type: 'check', id });
    });
  },
  /** What the recogniser worker reported when a WebGPU device was destroyed (its release at the end). */
  gpuDestroyed: () => [...gpuDestroyed],

  /**
   * (test/e2e/detect-cost.mjs only) One transcribe() per entry of `plan` (seconds of audio: the first `seconds` of the
   * e-rec fixture, 13.9 for e-rec twice in a row), in order, with the time of each: total, and the recogniser's own
   * (the line in the SDK's log: no model load, no speech gate). With `trackGpu` the live GPU buffers are read after
   * each call. Returns no more than the word count of each transcript and its text, for comparing the two settings.
   */
  async run({ plan, wavUrl, trackGpu = false }) {
    const wav = parseWav(await (await fetch(wavUrl)).arrayBuffer());
    const out = [];
    for (let i = 0; i < plan.length; i++) {
      const seconds = plan[i];
      const samples = seconds > wav.length / 16000 ? Float32Array.from([...wav, ...wav]) : wav.slice(0, Math.round(seconds * 16000));
      const before = logs.length;
      const t0 = performance.now();
      const text = await speech.transcribe(samples);
      const ms = Math.round(performance.now() - t0);
      const re = /recogniser (\d+) ms \(encoder (\d+) ms, decoder ([\d.]+) ms\/token\)/;
      const line = logs.slice(before).find((l) => re.test(l));
      const m = line && re.exec(line);
      const call = { i, seconds, ms, recogniserMs: m ? Number(m[1]) : null, encoderMs: m ? Number(m[2]) : null, text, words: text.split(/\s+/).filter(Boolean).length };
      if (trackGpu) {
        call.gpu = await window.e2e.gpuBuffers();
        call.stateGpuBytes = speech.state.gpuBytes;
      }
      out.push(call);
    }
    return { calls: out, state: speech.state, logs: [...logs] };
  },

  /**
   * After forced garbage collections in the recogniser worker of createCounting(): how many shared WebAssembly memories
   * (threaded CPU modules) it created and how many are still reachable ({ created, live, liveBytes }, count-workers.js).
   */
  async memories(timeoutMs = 20000) {
    const id = ++memoryChecks;
    const channel = new BroadcastChannel('czechspeech-e2e-memories');
    try {
      return await new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error('no answer from the recogniser worker')), timeoutMs);
        channel.onmessage = (e) => {
          if (!e.data || e.data.id !== id || e.data.worker !== 'czechspeech-recognizer') return;
          clearTimeout(timer);
          resolve(e.data);
        };
        channel.postMessage({ type: 'check', id });
      });
    } finally {
      channel.close();
    }
  },

  async transcribeUrl(wavUrl) {
    const samples = parseWav(await (await fetch(wavUrl)).arrayBuffer());
    return transcribeSamples(samples);
  },

  /** transcribe() of digital silence. */
  transcribeZeros: (seconds) => transcribeSamples(new Float32Array(Math.round(seconds * 16000))),

  /** transcribe() of any audio file the browser can decode (the ParCzech FLAC files), resampled to 16 kHz mono. */
  async transcribeDecoded(url) {
    const data = await (await fetch(url)).arrayBuffer();
    const ctx = new OfflineAudioContext(1, 16000, 16000);
    const buffer = await ctx.decodeAudioData(data);
    const samples = new Float32Array(buffer.getChannelData(0));
    return transcribeSamples(samples);
  },

  /** What WebGPU looks like from a dedicated worker (where the recogniser runs). */
  gpuInfo: () => new Promise((resolve) => {
    const w = new Worker(new URL('./gpu-probe-worker.js', import.meta.url), { type: 'module' });
    w.onmessage = (e) => { w.terminate(); resolve(e.data); };
    w.onerror = (e) => { w.terminate(); resolve({ error: e.message || 'worker error' }); };
    w.postMessage(1);
  }),

  /** The console.info lines the default logger printed, captured by consoleSpy(). */
  consoleSpy() {
    const info = console.info.bind(console);
    window.e2eInfo = [];
    console.info = (...a) => { window.e2eInfo.push(a.join(' ')); info(...a); };
    return true;
  },
  consoleInfo: () => [...(window.e2eInfo || [])],
  logs: () => [...logs],

  /**
   * listen() until `transcripts` transcripts arrived or `maxMs` passed, then leave the loop. With `stopAfterStartMs`,
   * speech.stop() is called that long after the first speechStarted instead, and the loop must end by itself
   * (`endedByItself`); `stoppedAt` is when stop() was called.
   */
  async listen({ transcripts = 1, maxMs = 120000, settleMs = 0, stopAfterStartMs = null } = {}) {
    const before = streams.length;
    const events = [];
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(new Error('timeout')), maxMs);
    const t0 = performance.now();
    let error = null;
    let n = 0;
    let doneAt = null;
    let stoppedAt = null;
    let stopTimer = null;
    let endedByItself = false;
    try {
      for await (const ev of speech.listen({ signal: ac.signal })) {
        const at = Math.round(performance.now() - t0);
        events.push(ev.type === 'transcript'
          ? { type: ev.type, text: ev.text, language: ev.language, startMs: ev.startMs, endMs: ev.endMs, processingMs: ev.processingMs, at, str: String(ev) }
          : { type: ev.type, atMs: ev.atMs, at, message: ev.error && ev.error.message });
        if (stopAfterStartMs !== null) {
          if (ev.type === 'speechStarted' && stopTimer === null) {
            stopTimer = setTimeout(() => { stoppedAt = Math.round(performance.now() - t0); speech.stop(); }, stopAfterStartMs);
          }
          continue;
        }
        if (ev.type === 'transcript' && ++n >= transcripts) {
          if (settleMs > 0 && doneAt === null) {
            doneAt = at;
            setTimeout(() => ac.abort(new Error('settled')), settleMs);
            continue;
          }
          break;
        }
      }
      endedByItself = stopAfterStartMs !== null;
    } catch (e) {
      if (!(e && (e.message === 'settled'))) error = { name: e && e.name, message: e && e.message, code: e && e.code };
    } finally {
      clearTimeout(timer);
      if (stopTimer !== null) clearTimeout(stopTimer);
    }
    const exitMs = Math.round(performance.now() - t0);
    const opened = streams.slice(before);
    return {
      events, error, exitMs, stoppedAt, endedByItself,
      tracks: opened.flatMap((s) => s.stream.getTracks().map((t) => t.readyState)),
      constraints: opened.map((s) => s.constraints),
      state: speech.state, logs: [...logs],
    };
  },

  /**
   * Records `seconds` of the (fake) microphone through the SDK's own path (getUserMedia with its constraints, the
   * AudioWorklet and its resampler): the 16 kHz frames a listen() session would get, kept in this page's memory only.
   */
  async captureMic(seconds) {
    const { openMicrophone } = await import('../../src/mic.js');
    for (const f of captured) f.fill(0);
    captured = [];
    const mic = await openMicrophone({ workletUrl: new URL('../../src/worklet.js', import.meta.url).href });
    mic.port.onmessage = (e) => { if (e.data instanceof Float32Array) captured.push(e.data); };
    await new Promise((r) => setTimeout(r, seconds * 1000));
    await mic.close();
    return captured.length;
  },

  /**
   * listen() over the frames captureMic() recorded, as fast as the recogniser goes: the same utterances and the same
   * recognitions as the live session (the capture worker cuts them). Returns the events (no text), the number of
   * words per transcript and the logs of this session.
   */
  async replayCapture() {
    const frames = captured;
    const source = {
      async open() {
        const ch = new MessageChannel();
        (async () => {
          for (let i = 0; i < frames.length; i++) {
            ch.port1.postMessage(frames[i].slice());
            if (i % 64 === 63) await new Promise((r) => setTimeout(r, 0));
          }
          ch.port1.postMessage({ type: 'end' });
        })();
        return { port: ch.port2, sampleRate: 16000, close: () => ch.port1.close(), finish: async () => false };
      },
    };
    const from = logs.length;
    const events = [];
    let error = null;
    try {
      for await (const ev of speech.listen({ _source: source })) {
        events.push(ev.type === 'transcript'
          ? { type: ev.type, startMs: ev.startMs, endMs: ev.endMs, processingMs: ev.processingMs, words: ev.text.split(/\s+/).filter(Boolean).length }
          : { type: ev.type, atMs: ev.atMs });
      }
    } catch (e) {
      error = { name: e && e.name, message: e && e.message, code: e && e.code };
    }
    return { events, error, logs: logs.slice(from), state: speech.state };
  },

  async memory() {
    if (self.crossOriginIsolated && performance.measureUserAgentSpecificMemory) {
      try {
        const m = await performance.measureUserAgentSpecificMemory();
        return { bytes: m.bytes };
      } catch (e) {
        return { error: String(e) };
      }
    }
    return { error: 'measureUserAgentSpecificMemory needs cross-origin isolation' };
  },

  async close() {
    if (speech) await speech.close();
    return speech ? speech.state : null;
  },
};
