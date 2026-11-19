// Endpointer: per-frame speech probabilities -> utterances. A line-by-line port of the Android SDK's
// internal/Endpointer.kt (CzechSpeech for Android 0.1.0), so both SDKs cut speech at the same frames.

export const SAMPLE_RATE = 16000;

/** Silero VAD v6 window at 16 kHz (n_window in the ggml model header): one probability per 512 samples. */
export const FRAME_SAMPLES = 512;
export const FRAME_MS = 32;

/** Search window for the cut when an utterance reaches maxUtteranceMs. */
export const CAP_SEARCH_MS = 2000;

/**
 * stop() with speech in progress lets the word in progress finish: capture goes on until a pause of this long
 * (frames below negThreshold; a pause already under way counts) ...
 */
export const STOP_TAIL_SILENCE_MS = 256;
/** ... or until this much audio after stop(), whichever comes first. An abrupt cut makes Whisper invent words. */
export const STOP_TAIL_MAX_MS = 1000;
/** STOP_TAIL_MAX_MS in frames, rounded down: at most 1 s. */
export const STOP_TAIL_MAX_FRAMES = Math.floor(STOP_TAIL_MAX_MS / FRAME_MS);

const ceilDiv = (a, b) => Math.floor((a + b - 1) / b);

/**
 * Turns per-frame speech probabilities into utterances. Plain JavaScript, one thread, no allocation per frame.
 *
 * - Speech starts at the first frame with probability >= threshold and stays on while frames are >= negThreshold.
 * - It ends after minSilenceMs below negThreshold. The utterance audio runs from preRollMs before the start to the
 *   end of that silence; startMs/endMs report the speech itself (trigger frame to last frame >= negThreshold).
 * - A run with fewer than minSpeechMs of frames >= threshold is discarded silently.
 * - At maxUtteranceMs of audio the utterance is cut after the frame with the lowest probability in the last
 *   CAP_SEARCH_MS and the rest continues as the next utterance of the same run.
 *
 * Every audio buffer is zeroed once it has been copied out or discarded. Emitted utterance samples are copies owned
 * by the receiver.
 *
 * Events passed to `sink`:
 *   { kind: 'speechStarted', atMs }               speech confirmed (minSpeechMs reached); atMs is where it started
 *   { kind: 'speechEnded', atMs, confirmed }      a run ended; confirmed is false for runs too short to report
 *   { kind: 'utterance', samples, startMs, endMs, audioStartMs }
 *                                                 audioStartMs is where `samples` begins (pre-roll included); the first
 *                                                 (endMs - audioStartMs) ms of `samples` end with the speech, the rest is silence
 */
export class Endpointer {
  constructor(config, sink) {
    this.sink = sink;
    // Float32 like the Kotlin Float fields, so comparisons match the Android SDK bit for bit.
    this.threshold = Math.fround(config.threshold);
    this.negThreshold = Math.fround(config.negThreshold);
    this.preRollFrames = ceilDiv(config.preRollMs, FRAME_MS);
    this.minSilenceFrames = Math.max(1, ceilDiv(config.minSilenceMs, FRAME_MS));
    this.minSpeechFrames = ceilDiv(config.minSpeechMs, FRAME_MS);
    this.maxFrames = Math.floor(config.maxUtteranceMs / FRAME_MS);
    this.capSearchFrames = Math.min(this.maxFrames, ceilDiv(CAP_SEARCH_MS, FRAME_MS));
    this.stopTailSilenceFrames = ceilDiv(STOP_TAIL_SILENCE_MS, FRAME_MS);

    // Pre-roll ring buffer, filled while there is no speech.
    this.ring = Array.from({ length: this.preRollFrames }, () => new Float32Array(FRAME_SAMPLES));
    this.ringProbs = new Float32Array(this.preRollFrames);
    this.ringHead = 0;
    this.ringSize = 0;

    // Current utterance: pre-roll + speech + trailing silence, at most maxFrames.
    this.audio = new Float32Array(this.maxFrames * FRAME_SAMPLES);
    this.probs = new Float32Array(this.maxFrames);
    this.nFrames = 0;
    this.bufferFirstFrame = 0;

    this.frameIndex = 0;
    this.inSpeech = false;
    this.runStartFrame = 0;
    this.runConfirmed = false;
    this.pieceStartMs = 0;
    this.pieceSpeechFrames = 0;
    this.silenceRun = 0;
    this.lastActiveFrame = 0;
  }

  /** Feeds one frame of FRAME_SAMPLES samples and its speech probability. The frame is copied. */
  accept(frame, prob) {
    if (frame.length !== FRAME_SAMPLES) throw new RangeError(`frame must have ${FRAME_SAMPLES} samples`);
    prob = Math.fround(prob);
    const index = this.frameIndex++;
    if (!this.inSpeech) {
      if (prob >= this.threshold) this.#startRun(index, frame, prob);
      else this.#pushRing(frame, prob);
      return;
    }
    if (this.nFrames === this.maxFrames) this.#cutAtCap();
    this.#append(frame, prob);
    if (prob >= this.threshold) this.pieceSpeechFrames++;
    if (prob >= this.negThreshold) {
      this.silenceRun = 0;
      this.lastActiveFrame = index;
    } else {
      this.silenceRun++;
    }
    this.#confirmIfReady();
    if (this.silenceRun >= this.minSilenceFrames) this.#endRun();
  }

  /** End of the audio stream: an utterance in progress ends as if silence followed. */
  finish() {
    if (this.inSpeech) this.#endRun();
  }

  /** No speech in progress, or it has paused for STOP_TAIL_SILENCE_MS: stop() need not wait for the word to end. */
  get atPause() {
    return !this.inSpeech || this.silenceRun >= this.stopTailSilenceFrames;
  }

  /** Zeroes and drops all buffered audio (cancellation). */
  clear() {
    for (const frame of this.ring) frame.fill(0);
    this.ringProbs.fill(0);
    this.ringHead = 0;
    this.ringSize = 0;
    this.audio.fill(0);
    this.probs.fill(0);
    this.nFrames = 0;
    this.inSpeech = false;
    this.runConfirmed = false;
    this.pieceSpeechFrames = 0;
    this.silenceRun = 0;
  }

  #startRun(index, frame, prob) {
    this.inSpeech = true;
    this.nFrames = 0;
    this.bufferFirstFrame = index - this.ringSize;
    for (let k = 0; k < this.ringSize; k++) {
      const slot = (this.ringHead + k) % this.preRollFrames;
      this.#append(this.ring[slot], this.ringProbs[slot]);
      this.ring[slot].fill(0);
      this.ringProbs[slot] = 0;
    }
    this.ringHead = 0;
    this.ringSize = 0;
    this.#append(frame, prob);
    this.runStartFrame = index;
    this.runConfirmed = false;
    this.pieceStartMs = index * FRAME_MS;
    this.pieceSpeechFrames = 1;
    this.silenceRun = 0;
    this.lastActiveFrame = index;
    this.#confirmIfReady();
  }

  #confirmIfReady() {
    if (!this.runConfirmed && this.pieceSpeechFrames >= this.minSpeechFrames) {
      this.runConfirmed = true;
      this.sink({ kind: 'speechStarted', atMs: this.runStartFrame * FRAME_MS });
    }
  }

  #endRun() {
    const endMs = (this.lastActiveFrame + 1) * FRAME_MS;
    const confirmed = this.runConfirmed;
    this.sink({ kind: 'speechEnded', atMs: endMs, confirmed });
    if (confirmed && this.pieceSpeechFrames >= this.minSpeechFrames) this.#emitPiece(this.nFrames, endMs);

    // The trailing silence becomes the pre-roll of a quick next utterance.
    const keep = Math.min(this.preRollFrames, this.nFrames);
    for (let k = 0; k < keep; k++) {
      const src = this.nFrames - keep + k;
      this.ring[k].set(this.audio.subarray(src * FRAME_SAMPLES, (src + 1) * FRAME_SAMPLES));
      this.ringProbs[k] = this.probs[src];
    }
    this.ringHead = 0;
    this.ringSize = keep;

    this.audio.fill(0, 0, this.nFrames * FRAME_SAMPLES);
    this.probs.fill(0, 0, this.nFrames);
    this.nFrames = 0;
    this.inSpeech = false;
    this.runConfirmed = false;
    this.pieceSpeechFrames = 0;
    this.silenceRun = 0;
  }

  #cutAtCap() {
    const from = this.nFrames - this.capSearchFrames;
    let lowest = from;
    for (let k = from; k < this.nFrames; k++) if (this.probs[k] < this.probs[lowest]) lowest = k;
    const cut = lowest + 1;
    let speech = 0;
    for (let k = 0; k < cut; k++) if (this.probs[k] >= this.threshold) speech++;
    if (this.runConfirmed && speech >= this.minSpeechFrames) this.#emitPiece(cut, (this.bufferFirstFrame + cut) * FRAME_MS);

    const rest = this.nFrames - cut;
    this.audio.copyWithin(0, cut * FRAME_SAMPLES, this.nFrames * FRAME_SAMPLES);
    this.audio.fill(0, rest * FRAME_SAMPLES, this.nFrames * FRAME_SAMPLES);
    this.probs.copyWithin(0, cut, this.nFrames);
    this.probs.fill(0, rest, this.nFrames);
    this.nFrames = rest;
    this.bufferFirstFrame += cut;
    this.pieceStartMs = this.bufferFirstFrame * FRAME_MS;
    this.pieceSpeechFrames = 0;
    for (let k = 0; k < rest; k++) if (this.probs[k] >= this.threshold) this.pieceSpeechFrames++;
    if (!this.runConfirmed) this.runStartFrame = this.bufferFirstFrame;
  }

  #emitPiece(frames, endMs) {
    const samples = this.audio.slice(0, frames * FRAME_SAMPLES);
    this.sink({ kind: 'utterance', samples, startMs: this.pieceStartMs, endMs, audioStartMs: this.bufferFirstFrame * FRAME_MS });
  }

  #append(frame, prob) {
    this.audio.set(frame, this.nFrames * FRAME_SAMPLES);
    this.probs[this.nFrames] = prob;
    this.nFrames++;
  }

  #pushRing(frame, prob) {
    if (this.preRollFrames === 0) return;
    let slot;
    if (this.ringSize < this.preRollFrames) {
      slot = (this.ringHead + this.ringSize) % this.preRollFrames;
      this.ringSize++;
    } else {
      slot = this.ringHead;
      this.ringHead = (this.ringHead + 1) % this.preRollFrames;
    }
    this.ring[slot].set(frame);
    this.ringProbs[slot] = prob;
  }
}
