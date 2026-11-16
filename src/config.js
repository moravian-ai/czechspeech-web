// Speech detection settings: the same fields, defaults and limits as the Android SDK's CzechSpeechConfig, plus the
// recogniser backend (web only).

import { BACKENDS } from './backend.js';

/**
 * Defaults are the validated ones; most apps never change them.
 *
 * - threshold 0.25: an utterance starts at the first 32 ms frame with Silero probability >= this; the same value
 *   gates Whisper. 0.25 rather than Silero's usual 0.5: on 2,181 test recordings with speech, 0.25 dropped none,
 *   0.5 dropped 5 of 729 recordings with reverberation and babble noise; all synthetic non-speech was still rejected.
 * - negThreshold 0.10: once started, the utterance continues while the probability stays >= this.
 * - minSilenceMs 600: the utterance ends after this much audio below negThreshold.
 * - preRollMs 300: audio before the start of speech passed to the recogniser.
 * - minSpeechMs 250: utterances with less audio >= threshold are dropped (clicks, coughs).
 * - maxUtteranceMs 28000: longer speech is cut at the quietest 32 ms frame of the last 2 s.
 * - threads 0: recogniser threads; 0 = automatic (half the logical cores, 1 to 8; 1 without cross-origin isolation).
 * - backend 'auto': the recogniser on the GPU through WebGPU where the browser can (src/backend.js), else on the CPU
 *   (WebAssembly); 'webgpu' requires the GPU (error code 'unsupported' otherwise), 'cpu' never uses it.
 */
export const DEFAULT_CONFIG = Object.freeze({
  threshold: 0.25,
  negThreshold: 0.1,
  minSilenceMs: 600,
  preRollMs: 300,
  minSpeechMs: 250,
  maxUtteranceMs: 28000,
  threads: 0,
  backend: 'auto',
});

const isInt = (v) => Number.isInteger(v);

/** Defaults filled in and validated like the Kotlin `init` block. Throws RangeError / TypeError. */
export function resolveConfig(config = {}) {
  if (config === null || typeof config !== 'object') throw new TypeError('config must be an object');
  for (const key of Object.keys(config)) {
    if (!(key in DEFAULT_CONFIG)) throw new TypeError(`unknown config field "${key}"`);
  }
  const c = { ...DEFAULT_CONFIG, ...config };
  const check = (ok, message) => {
    if (!ok) throw new RangeError(message);
  };
  check(typeof c.threshold === 'number' && c.threshold > 0 && c.threshold < 1, 'threshold must be in (0, 1)');
  check(typeof c.negThreshold === 'number' && c.negThreshold >= 0 && c.negThreshold <= c.threshold,
    'negThreshold must be in [0, threshold]');
  check(isInt(c.minSilenceMs) && c.minSilenceMs >= 32 && c.minSilenceMs <= 10000, 'minSilenceMs must be in 32..10000');
  check(isInt(c.preRollMs) && c.preRollMs >= 0 && c.preRollMs <= 2000, 'preRollMs must be in 0..2000');
  check(isInt(c.minSpeechMs) && c.minSpeechMs >= 0 && c.minSpeechMs <= 10000, 'minSpeechMs must be in 0..10000');
  check(isInt(c.maxUtteranceMs) && c.maxUtteranceMs >= 4000 && c.maxUtteranceMs <= 29000,
    'maxUtteranceMs must be in 4000..29000');
  check(isInt(c.threads) && c.threads >= 0 && c.threads <= 64, 'threads must be in 0..64 (0 = automatic)');
  check(BACKENDS.includes(c.backend), `backend must be one of ${BACKENDS.map((b) => `"${b}"`).join(', ')}`);
  return Object.freeze(c);
}

/** Recogniser threads for a config on this machine. Single-threaded without SharedArrayBuffer. */
export function threadCount(config, multiThreaded, hardwareConcurrency) {
  if (!multiThreaded) return 1;
  if (config.threads > 0) return config.threads;
  const cores = Number.isInteger(hardwareConcurrency) && hardwareConcurrency > 0 ? hardwareConcurrency : 2;
  return Math.max(1, Math.min(8, Math.floor(cores / 2)));
}
