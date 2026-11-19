// Public event and error types. Transcript text is personal data: toString(), toJSON() and Node's inspection show
// its length only, so it does not end up in logs or serialised telemetry by accident. Read `event.text` explicitly.

/** The codes of CzechSpeechError, as in types/index.d.ts (CzechSpeechErrorCode). */
export const ERROR_CODES = Object.freeze(['microphone-denied', 'microphone-unavailable', 'microphone-ended', 'model',
  'download', 'storage', 'runtime', 'recognition', 'busy', 'closed', 'unsupported', 'unknown-model', 'voiceprint-model',
  'enroll-too-short', 'enroll-too-noisy']);

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
 * `utteranceId` names the utterance for the SpeakersEvent that follows it (listen({ speakers })): '<listen call>.<n>'.
 */
export class TranscriptEvent {
  constructor(text, startMs, endMs, processingMs, language, utteranceId) {
    this.type = 'transcript';
    this.text = text;
    this.startMs = startMs;
    this.endMs = endMs;
    this.processingMs = processingMs;
    this.language = language;
    this.utteranceId = utteranceId;
    Object.freeze(this);
  }

  toString() {
    return `Transcript(text=<${this.text.length} chars>, language=${this.language}, startMs=${this.startMs}, ` +
      `endMs=${this.endMs}, processingMs=${this.processingMs})`;
  }

  toJSON() {
    return { type: this.type, text: `<${this.text.length} chars>`, language: this.language, startMs: this.startMs,
      endMs: this.endMs, processingMs: this.processingMs, utteranceId: this.utteranceId };
  }

  [INSPECT]() {
    return this.toString();
  }
}

/**
 * Who spoke in one utterance, after its transcript (same `utteranceId`). `parts` are the stretches of the utterance with
 * the speaker found for each (null: not sure), times in ms of audio since the session started; `speakers` the distinct
 * speakers in the order they spoke. `skipped`: why there are no parts ('cpu': the transcript runs on the CPU and the
 * app asked not to compete with it; 'behind': the speaker worker was behind; 'error': this utterance failed and the
 * speaker worker lives on; 'failed': the speaker worker is dead or the session closed). The speaker labels are the
 * app's data: toString(), toJSON() and Node's inspection show counts only, never a key or a number.
 */
export class SpeakersEvent {
  constructor(utteranceId, parts, skipped) {
    this.type = 'speakers';
    this.utteranceId = utteranceId;
    this.parts = Object.freeze(parts.map((p) => Object.freeze({
      startMs: p.startMs, endMs: p.endMs, speaker: p.speaker ? Object.freeze({ ...p.speaker }) : null })));
    const seen = new Set();
    const speakers = [];
    for (const p of this.parts) {
      if (!p.speaker) continue;
      const id = p.speaker.kind === 'known' ? `k:${p.speaker.key}` : `s:${p.speaker.n}`;
      if (seen.has(id)) continue;
      seen.add(id);
      speakers.push(p.speaker);
    }
    this.speakers = Object.freeze(speakers);
    this.skipped = skipped ?? null;
    Object.freeze(this);
  }

  toString() {
    return `Speakers(utteranceId=${this.utteranceId}, parts=${this.parts.length}, speakers=${this.speakers.length}, skipped=${this.skipped})`;
  }

  toJSON() {
    return { type: this.type, utteranceId: this.utteranceId, parts: this.parts.length, speakers: this.speakers.length, skipped: this.skipped };
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
