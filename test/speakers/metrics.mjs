// Metrics of the speaker assignment against the true speech turns of the synthetic meetings: a port of the measured
// Python code (research/speakers/metrics.py `score`, `_speech_in`, `_map_session_labels`, run.py `outcomes`), counts only.
import { maxAssignment } from './hungarian.mjs';

/** Milliseconds of speech of every person inside [startMs, endMs); the keys are in the order of `turns`. */
export function speechIn(turns, startMs, endMs) {
  const out = new Map();
  for (const t of turns) {
    const ms = Math.min(endMs, t.endMs) - Math.max(startMs, t.startMs);
    if (ms > 0) out.set(t.speaker, (out.get(t.speaker) || 0) + ms);
  }
  return out;
}

/** One-to-one assignment of the labels "Speaker n" to persons that maximises the overlapping speech. */
function mapSessionLabels(speech, results) {
  const overlap = new Map();
  results.forEach((r, i) => {
    if (r.label && r.label.kind === 'session') {
      const d = overlap.get(r.label.n) || new Map();
      for (const [person, ms] of speech[i]) d.set(person, (d.get(person) || 0) + ms);
      overlap.set(r.label.n, d);
    }
  });
  const ns = [...overlap.keys()].sort((a, b) => a - b);
  const persons = [...new Set([...overlap.values()].flatMap((d) => [...d.keys()]))].sort();
  if (!ns.length || !persons.length) return new Map();
  const m = ns.map((n) => persons.map((p) => overlap.get(n).get(p) || 0));
  return new Map(maxAssignment(m).map(([i, j]) => [ns[i], persons[j]]));
}

function truthOf(sp) {
  let best = null;
  let bestMs = -Infinity;
  for (const [p, ms] of sp) if (ms > bestMs) { best = p; bestMs = ms; }
  return best;
}

/**
 * results: [{ startMs, endMs, label }] with label { kind: 'named', name } | { kind: 'session', n } | null (unsure);
 * turns: [{ speaker, startMs, endMs }]; enrolled: Set of names with a stored voice.
 * The true speaker of a piece is the one with the most speech inside it (the first in `turns` on a tie).
 * namedOk: pieces with speech whose name is the true speaker; correct: pieces whose name or mapped "Speaker n" is the
 * true speaker; confusion: pieces (also without speech) with the name of an enrolled person who is not the true speaker.
 */
export function scoreMeeting(results, turns, enrolled) {
  const speech = results.map((r) => speechIn(turns, r.startMs, r.endMs));
  const mapping = mapSessionLabels(speech, results);
  let nScored = 0;
  let namedOk = 0;
  let correct = 0;
  let confusion = 0;
  let uncertain = 0;
  results.forEach((r, i) => {
    const sp = speech[i];
    const label = r.label;
    const named = label && label.kind === 'named';
    if (!sp.size) {
      if (named && enrolled.has(label.name)) confusion++;
      return;
    }
    nScored++;
    const truth = truthOf(sp);
    const who = named ? label.name : label && label.kind === 'session' ? mapping.get(label.n) : undefined;
    if (!label) uncertain++;
    else if (who === truth) correct++;
    if (named && who === truth) namedOk++;
    if (named && enrolled.has(label.name) && who !== truth) confusion++;
  });
  return { nScored, nAll: results.length, namedOk, correct, confusion, uncertain };
}
