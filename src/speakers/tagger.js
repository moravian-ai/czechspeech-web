// Speaker assignment of one meeting, port of the measured rules (improve.tag_stream, fast path, and _Sessions.assign).
// Everything is held in memory only; clear() zeroes the sums.
import { dot, namedThreshold, unit } from './maths.js';

export class Tagger {
  #point;
  #known;
  #next;
  #maxLabels;
  #sessions = []; // { n, sum, centroid, count, speechMs }

  /** known: Array<{ key, vector }> (stored voices). firstLabel: number of the first new "Speaker n". */
  constructor({ point, known = [], firstLabel = 1, maxLabels = 30 }) {
    this.#point = point;
    this.#known = known.map((k) => ({ key: k.key, vector: unit(k.vector) }));
    this.#next = firstLabel;
    this.#maxLabels = maxLabels;
  }

  /** `{ kind: 'known', key }`, `{ kind: 'session', n }` or null (unsure). `vector` must be a unit vector. */
  tag(vector, durMs) {
    const p = this.#point;
    if (this.#known.length) {
      let best = -Infinity;
      let second = -Infinity;
      let bi = -1;
      for (let i = 0; i < this.#known.length; i++) {
        const s = dot(this.#known[i].vector, vector);
        if (s > best) { second = best; best = s; bi = i; } else if (s > second) second = s;
      }
      if (best >= namedThreshold(p.named, durMs, p.lengthSlope)) {
        // above the threshold but without the margin: unsure, and no new speaker is founded
        return best - second >= p.margin ? { kind: 'known', key: this.#known[bi].key } : null;
      }
    }
    let bj = -1;
    let bs = -Infinity;
    for (let j = 0; j < this.#sessions.length; j++) {
      const s = dot(this.#sessions[j].centroid, vector);
      if (s > bs) { bs = s; bj = j; }
    }
    if (bj >= 0 && bs >= p.session) {
      const sp = this.#sessions[bj];
      for (let i = 0; i < sp.sum.length; i++) sp.sum[i] += vector[i];
      const old = sp.centroid;
      sp.centroid = unit(sp.sum);
      old.fill(0);
      sp.count += 1;
      sp.speechMs += durMs;
      return { kind: 'session', n: sp.n };
    }
    if (durMs >= p.minMs && this.#sessions.length < this.#maxLabels) {
      const sp = { n: this.#next++, sum: Float32Array.from(vector), centroid: unit(vector), count: 1, speechMs: durMs };
      this.#sessions.push(sp);
      return { kind: 'session', n: sp.n };
    }
    return null;
  }

  get speakers() {
    return this.#sessions.map((s) => ({ n: s.n, count: s.count, speechMs: s.speechMs }));
  }

  /** Normalised sum of the pieces of speaker `n` (a copy). */
  centroid(n) {
    const sp = this.#sessions.find((s) => s.n === n);
    if (!sp) throw new RangeError(`no speaker ${n}`);
    return sp.centroid.slice();
  }

  clear() {
    for (const s of this.#sessions) { s.sum.fill(0); s.centroid.fill(0); }
    this.#sessions = [];
    for (const k of this.#known) k.vector.fill(0);
    this.#known = [];
  }
}
