// Small numeric helpers of the speaker pipeline: vectors, the length-dependent name threshold and the cutting of an
// utterance at speaker changes. Ports of the measured Python code (assign.change_points, run._valid_cuts).

/** Unit-length copy of `v` (a zero vector stays zero). */
export function unit(v) {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  const out = new Float32Array(v.length);
  if (s === 0) return out;
  const inv = 1 / Math.sqrt(s);
  for (let i = 0; i < v.length; i++) out[i] = v[i] * inv;
  return out;
}

export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** Threshold for a stored voice: `named + slope * ln(max(durMs, 200) / 3000)`. */
export function namedThreshold(named, durMs, slope) {
  return named + slope * Math.log(Math.max(durMs, 200) / 3000);
}

/** Start samples of the analysis windows `k * hop` for `k * hop <= length - win`; none below two windows. */
export function windowStarts(lengthSamples, winSamples, hopSamples) {
  const out = [];
  if (lengthSamples < 2 * winSamples) return out;
  for (let s = 0; s <= lengthSamples - winSamples; s += hopSamples) out.push(s);
  return out;
}

/**
 * Speaker changes from the dissimilarity scores (1 - cosine of the windows before and after a point). Score i belongs
 * to the point `firstMs + i * hopMs`. A change is a local maximum (against its neighbours) at or above `drop`; a
 * stronger change pushes out a weaker one closer than `winMs`.
 */
export function changePoints(scores, { firstMs, hopMs, winMs, drop }) {
  const peaks = [];
  for (let i = 0; i < scores.length; i++) {
    const sc = scores[i];
    if (sc < drop) continue;
    let max = -Infinity;
    for (let j = Math.max(0, i - 1); j < Math.min(scores.length, i + 2); j++) max = Math.max(max, scores[j]);
    if (sc >= max) peaks.push([i, sc]);
  }
  const out = [];
  for (const [i, sc] of peaks) {
    const last = out[out.length - 1];
    if (last && (i - last[0]) * hopMs < winMs) {
      if (sc > last[1]) out[out.length - 1] = [i, sc];
      continue;
    }
    out.push([i, sc]);
  }
  return out.map(([i]) => Math.trunc(firstMs + i * hopMs));
}

/** Cuts that keep at least `minMs` from the start, from the previous kept cut and from the end. */
export function validCuts(cutsMs, durMs, minMs) {
  const out = [];
  let last = 0;
  for (const t of cutsMs) {
    if (t - last >= minMs && durMs - t >= minMs) {
      out.push(t);
      last = t;
    }
  }
  return out;
}

/** The part of an utterance buffer up to `endMs` (the buffer starts at `audioStartMs`); a view, no copy. */
export function speechSpan(samples, audioStartMs, endMs) {
  return samples.subarray(0, Math.min(samples.length, Math.max(0, Math.round((endMs - audioStartMs) * 16))));
}
