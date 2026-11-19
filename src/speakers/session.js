// The speakers of one meeting. The state (who is "Speaker 1", "Speaker 2"...) belongs to the meeting, not to one
// listen() call, so it lives here and survives pauses: listen({ speakers: session }) attaches to it for as long as it
// runs, and the next listen() with the same session goes on with the same numbers.
//
// What it holds is in memory only: the speaker worker (the model and the pieces being computed) and the Tagger (the
// centres of the speakers). close() zeroes the centres and terminates the worker; nothing is ever stored or sent.
// The labels are the app's data (it knows the names): toString(), toJSON() and Node's inspection show nothing of them.
import { CzechSpeechError, SpeakersEvent } from '../events.js';
import { SpeakerEngine } from './engine.js';
import { DEFAULT_VOICEPRINT_ID, voiceprintEntry } from './catalog.js';
import { Tagger } from './tagger.js';
import { speakerSupport } from './support.js';

const INSPECT = Symbol.for('nodejs.util.inspect.custom');

export class SpeakerSession {
  static #token = Symbol('SpeakerSession');
  #engine;
  #tagger;
  #onCpu;
  #closed = false;
  #attached = new Map(); // listenId -> true while attached
  #held = new Set();     // pending utterances: pieces that have not been tagged (or discarded) yet

  constructor(token, engine, tagger, onCpu) {
    if (token !== SpeakerSession.#token) throw new TypeError('use SpeakerSession.create()');
    this.#engine = engine;
    this.#tagger = tagger;
    this.#onCpu = onCpu;
  }

  /**
   * @param {object} o
   * @param {object} o.model          a VoiceprintModel (downloaded)
   * @param {number} [o.firstLabel]   number of the first new speaker (default 1; after a recovery the next free one)
   * @param {number} [o.threads]      1 or 2 (default: what speakerSupport() advises)
   * @param {'skip'|'run'} [o.onCpu]  'skip' (default): while the recogniser runs on the CPU, speakers are not computed
   * @param {string|URL} [o.assetBaseUrl]  where speakers/worker.js is served from
   */
  static async create({ model, firstLabel = 1, threads = speakerSupport().threads || 1, onCpu = 'skip', assetBaseUrl, _createWorker, _fake, _onWorkerMessage, _startupTimeoutMs } = {}) {
    if (!model || typeof model._modelBlob !== 'function') throw new TypeError('SpeakerSession.create() needs a VoiceprintModel');
    if (!Number.isInteger(firstLabel) || firstLabel < 1) throw new RangeError('firstLabel must be a positive whole number');
    if (onCpu !== 'skip' && onCpu !== 'run') throw new RangeError("onCpu must be 'skip' or 'run'");
    const point = voiceprintEntry(model.id ?? DEFAULT_VOICEPRINT_ID).operatingPoints.strict;
    const engine = await SpeakerEngine.create({ model, threads, point, assetBaseUrl, _createWorker, _fake, _onWorkerMessage, _startupTimeoutMs });
    return new SpeakerSession(SpeakerSession.#token, engine, new Tagger({ point, firstLabel }), onCpu);
  }

  /** The speakers found so far in this meeting: their number and how much they have spoken. */
  get speakers() {
    return this.#tagger.speakers.map((s) => ({ n: s.n, speechMs: s.speechMs }));
  }

  /** 'skip' or 'run' (see create()). */
  get onCpu() { return this.#onCpu; }

  /** Threads the runtime uses (1 when the page is not cross-origin isolated). */
  get threads() { return this.#engine.threads; }

  get closed() { return this.#closed; }

  /**
   * For listen(): a port for the capture worker (it posts `{ type: 'utterance', id, startMs, audioStartMs, endMs,
   * samples }` on it) and `onEvent(pending, { mode, computeMs })` for each answer. `pending` is `{ utteranceId, skipped,
   * commit(), discard() }`: nothing is tagged yet. listen() calls `commit()` (tags the pieces, in the order the transcripts
   * come out, and returns the SpeakersEvent) when the utterance's transcript is out, or `discard()` when it will have none
   * (an utterance that is never transcribed founds no speaker and moves no centre). Either one zeroes the vectors.
   * `onFailure(error)` is called once when the speaker worker fails (code 'runtime') or the session is closed (code
   * 'closed'): nothing more comes then. An utterance that fails alone is answered with `skipped` 'error' (the worker
   * lives); 'failed' is for a dead worker or a closed session.
   */
  _attach(listenId, onEvent, onFailure) {
    if (this.#closed) throw new CzechSpeechError('The speaker session is closed', { code: 'closed' });
    const port = this.#engine.attach(listenId, (m) => this.#onPieces(listenId, onEvent, m), onFailure);
    this.#attached.set(listenId, true);
    return port;
  }

  _detach(listenId) {
    this.#attached.delete(listenId);
    this.#engine.detach(listenId);
  }

  #onPieces(listenId, onEvent, m) {
    const id = `${listenId}.${m.id}`;
    if (this.#closed) return;
    if (m.error !== undefined) {
      onEvent(this.#pending(id, [], 0, 'error'), { mode: 'failed', computeMs: 0 });
      return;
    }
    const offset = Number.isFinite(m.audioStartMs) ? m.audioStartMs : 0;
    onEvent(this.#pending(id, m.pieces || [], offset, m.mode === 'skipped' ? 'behind' : null), { mode: m.mode, computeMs: m.computeMs });
  }

  /** The answer for one utterance, held until its transcript decides: commit() tags it, discard() forgets it. */
  #pending(id, pieces, offset, skipped) {
    let state = 'held'; // 'held' | 'done' | 'gone'
    let event = null;
    const zero = () => { for (const p of pieces) p.vector?.fill(0); };
    const pending = {
      utteranceId: id,
      skipped,
      commit: () => {
        if (state === 'done') return event;
        if (state === 'gone') return new SpeakersEvent(id, [], 'failed');
        state = 'done';
        this.#held.delete(pending);
        const parts = [];
        for (const p of pieces) {
          const speaker = this.#tagger.tag(p.vector, p.endMs - p.startMs);
          parts.push({ startMs: offset + p.startMs, endMs: offset + p.endMs, speaker });
        }
        zero(); // the centre has taken what it needs
        event = new SpeakersEvent(id, parts, skipped);
        return event;
      },
      discard: () => {
        if (state !== 'held') return;
        state = 'gone';
        this.#held.delete(pending);
        zero();
      },
    };
    if (pieces.length > 0) this.#held.add(pending);
    else state = 'held';
    return pending;
  }

  /** Forgets the speakers (their centres are zeroed), stops the worker and tells attached listeners. */
  async close() {
    if (this.#closed) return;
    this.#closed = true;
    for (const pending of [...this.#held]) pending.discard(); // waiting for a transcript that will not matter now
    await this.#engine.close();
    this.#tagger.clear();
    this.#attached.clear();
  }

  toString() { return `SpeakerSession(closed=${this.#closed})`; }
  toJSON() { return {}; }
  [INSPECT]() { return this.toString(); }
}
