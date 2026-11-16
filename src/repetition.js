// Repetition loops in recognition results. Greedy decoding sometimes falls into a loop when the audio ends abruptly in
// the middle of a word (stop(), or speech cut at maxUtteranceMs): "… v roce 2012 až se státní podporou v roce 2012
// až se státní podporou …" until the token limit. whisper.cpp's temperature fallback does not prevent it. The
// recogniser applies this to every result, after the letters/digits filter (src/worker-core.js); the Android SDK's
// RepetitionGuard.kt implements the same rule.
//
// The rule:
// - Words are separated by whitespace and compared without case and punctuation (only letters and decimal digits
//   count; a token without any, such as a dash, belongs to the word before it). The text kept is the original.
// - A phrase of 2 to 12 words that occurs at least 3 times in immediate succession keeps its first occurrence; the
//   repeats are dropped. When they end the text, a trailing partial copy (a prefix of the phrase, its last word
//   possibly cut off, as "se st") is dropped too.
// - Single repeated words ("ne, ne, ne") and phrases repeated only twice stay. Scanning from the left, the smallest
//   phrase that explains a run wins; a phrase that is itself a repetition of a shorter one ("ne ne") does not count,
//   so "ne ne ne ne ne ne" stays as well.
// - The rule is applied again until nothing changes (a loop of loops).

export const LOOP_MIN_WORDS = 2;
export const LOOP_MAX_WORDS = 12;
export const LOOP_MIN_COPIES = 3;

const NOT_LETTER_OR_DIGIT = /[^\p{L}\p{Nd}]+/gu;

/**
 * @param {string} text  a recognition result
 * @returns {{ text: string, loops: { words: number, repeats: number, partial: boolean }[] }}  the text without the
 *   loops (the same string when there are none), and per loop removed: its phrase length, the copies dropped after the
 *   first, and whether a partial copy at the end was dropped
 */
export function removeRepetitionLoops(text) {
  let words = splitWords(text);
  const loops = [];
  for (;;) {
    const pass = collapse(words);
    if (pass.loops.length === 0) break;
    loops.push(...pass.loops);
    words = pass.words;
  }
  if (loops.length === 0) return { text, loops };
  return { text: words.map((w, i) => (i > 0 ? w.sep : '') + text.slice(w.start, w.end)).join(''), loops };
}

/** The log line for removed loops: counts only, never text. */
export function describeLoops(loops) {
  const one = (l) => `${l.words} words, ${l.repeats} repeats${l.partial ? ', partial copy at the end' : ''}`;
  return loops.length === 1 ? `repetition loop removed (${one(loops[0])})`
    : `${loops.length} repetition loops removed (${loops.map(one).join('; ')})`;
}

/** Words with their place in the text, the whitespace before them and their comparison key. */
function splitWords(text) {
  const words = [];
  let pendingStart = -1;
  let lastEnd = 0;
  for (const m of text.matchAll(/\S+/g)) {
    const start = m.index;
    const end = start + m[0].length;
    const key = m[0].toLowerCase().replace(NOT_LETTER_OR_DIGIT, '');
    if (!key) {
      // punctuation on its own: part of the word before it (or, at the start, of the word after it)
      if (words.length > 0) {
        words[words.length - 1].end = end;
        lastEnd = end;
      } else if (pendingStart < 0) {
        pendingStart = start;
      }
      continue;
    }
    const first = pendingStart >= 0 ? pendingStart : start;
    words.push({ start: first, end, sep: text.slice(lastEnd, first), key });
    pendingStart = -1;
    lastEnd = end;
  }
  return words;
}

/** One pass from the left: every run collapsed to its first copy. */
function collapse(words) {
  const out = [];
  const loops = [];
  const n = words.length;
  let i = 0;
  while (i < n) {
    const run = findRun(words, i);
    if (!run) {
      out.push(words[i++]);
      continue;
    }
    for (let x = i; x < i + run.k; x++) out.push(words[x]);
    let next = i + run.k * run.copies;
    const partial = next < n && isTrailingPartial(words, i, run.k, next);
    if (partial) next = n;
    loops.push({ words: run.k, repeats: run.copies - 1, partial });
    i = next;
  }
  return { words: out, loops };
}

/** The run starting at word i with the smallest phrase length k, or null. */
function findRun(words, i) {
  const n = words.length;
  for (let k = LOOP_MIN_WORDS; k <= LOOP_MAX_WORDS && i + LOOP_MIN_COPIES * k <= n; k++) {
    if (!isPrimitive(words, i, k)) continue;
    let copies = 1;
    while (i + (copies + 1) * k <= n && samePhrase(words, i, i + copies * k, k)) copies++;
    if (copies >= LOOP_MIN_COPIES) return { k, copies };
  }
  return null;
}

function samePhrase(words, a, b, k) {
  for (let x = 0; x < k; x++) if (words[a + x].key !== words[b + x].key) return false;
  return true;
}

/** False when the k words at i are a shorter phrase repeated ("ne ne", "a b a b"): that one explains the run. */
function isPrimitive(words, i, k) {
  for (let d = 1; d < k; d++) {
    if (k % d !== 0) continue;
    let periodic = true;
    for (let x = d; x < k && periodic; x++) periodic = words[i + x].key === words[i + x - d].key;
    if (periodic) return false;
  }
  return true;
}

/** Whether the words from `from` to the end are a prefix of the phrase at i (k words), the last one possibly cut. */
function isTrailingPartial(words, i, k, from) {
  const m = words.length - from;
  if (m < 1 || m > k) return false;
  for (let x = 0; x < m - 1; x++) if (words[from + x].key !== words[i + x].key) return false;
  return words[i + m - 1].key.startsWith(words[from + m - 1].key);
}
