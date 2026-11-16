// Public event and error types. Transcript text is personal data: toString(), toJSON() and Node's inspection show
// its length only, so it does not end up in logs or serialised telemetry by accident. Read `event.text` explicitly.

/** The codes of CzechSpeechError, as in types/index.d.ts (CzechSpeechErrorCode). */
export const ERROR_CODES = Object.freeze(['microphone-denied', 'microphone-unavailable', 'microphone-ended', 'model',
  'download', 'storage', 'runtime', 'recognition', 'busy', 'closed', 'unsupported', 'unknown-model']);

/**
 * `code` when it is one of ERROR_CODES, else `fallback`. Errors of the platform carry codes of their own (a
 * DOMException's number, Node's 'ENOENT'), which must not reach the app as a CzechSpeechError code.
 */
export function errorCode(code, fallback = 'runtime') {
  return ERROR_CODES.includes(code) ? code : fallback;
}

/** Thrown (or rejected) when the model, the microphone or the WebAssembly runtime cannot be used. */
export class CzechSpeechError extends Error {
  /**
   * @param {string} message
   * @param {{ code?: string, cause?: unknown }} [options]
   *   code: one of ERROR_CODES; anything else (or none) becomes 'runtime'
   */
  constructor(message, options = {}) {
    super(message, options.cause !== undefined ? { cause: options.cause } : undefined);
    this.name = 'CzechSpeechError';
    this.code = errorCode(options.code);
  }
}

const INSPECT = Symbol.for('nodejs.util.inspect.custom');

export class SpeechStartedEvent {
  constructor(atMs) {
    this.type = 'speechStarted';
    this.atMs = atMs;
    Object.freeze(this);
  }
}

export class SpeechEndedEvent {
  constructor(atMs) {
    this.type = 'speechEnded';
    this.atMs = atMs;
    Object.freeze(this);
  }
}

/**
 * One recognised utterance. `text` is never blank. Times are ms of audio since the session started. `language` is the
 * language the text was recognised in (a whisper language code such as 'cs'; one of the model's languages).
 */
export class TranscriptEvent {
  constructor(text, startMs, endMs, processingMs, language) {
    this.type = 'transcript';
    this.text = text;
    this.startMs = startMs;
    this.endMs = endMs;
    this.processingMs = processingMs;
    this.language = language;
    Object.freeze(this);
  }

  toString() {
    return `Transcript(text=<${this.text.length} chars>, language=${this.language}, startMs=${this.startMs}, ` +
      `endMs=${this.endMs}, processingMs=${this.processingMs})`;
  }

  toJSON() {
    return { type: this.type, text: `<${this.text.length} chars>`, language: this.language, startMs: this.startMs,
      endMs: this.endMs, processingMs: this.processingMs };
  }

  [INSPECT]() {
    return this.toString();
  }
}

/** Recognition of one utterance failed; the session goes on. Fatal errors are thrown from the iterator instead. */
export class SpeechErrorEvent {
  constructor(error) {
    this.type = 'error';
    this.error = error;
    Object.freeze(this);
  }
}
