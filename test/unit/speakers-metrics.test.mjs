import { test } from 'node:test';
import assert from 'node:assert/strict';
import { maxAssignment } from '../speakers/hungarian.mjs';
import { scoreMeeting, speechIn } from '../speakers/metrics.mjs';

test('maxAssignment maximises the sum over a rectangular matrix', () => {
  assert.deepEqual(maxAssignment([[3, 1], [2, 4], [0, 5]]).sort(), [[0, 0], [2, 1]]);
});

test('maxAssignment equals brute force on random small matrices', () => {
  let seed = 7;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) % 100);
  const brute = (m) => {
    const cols = m[0].length; let best = -1;
    const go = (r, used, sum) => { if (r === m.length) { best = Math.max(best, sum); return; }
      go(r + 1, used, sum); for (let c = 0; c < cols; c++) if (!used.has(c)) go(r + 1, new Set([...used, c]), sum + m[r][c]); };
    go(0, new Set(), 0); return best;
  };
  for (let k = 0; k < 50; k++) {
    const rows = 1 + (k % 5), cols = 1 + ((k * 3) % 4);
    const m = Array.from({ length: rows }, () => Array.from({ length: cols }, rnd));
    const sum = maxAssignment(m).reduce((s, [r, c]) => s + m[r][c], 0);
    assert.equal(sum, brute(m));
  }
});

test('scoreMeeting matches the Python definitions on a tiny meeting', () => {
  const turns = [{ speaker: 'A', startMs: 0, endMs: 4000 }, { speaker: 'B', startMs: 4000, endMs: 8000 }];
  const results = [
    { startMs: 0, endMs: 3000, label: { kind: 'named', name: 'A' } },       // right name
    { startMs: 3000, endMs: 5000, label: { kind: 'named', name: 'B' } },    // tie 1000/1000: truth is A (first in turns) -> confused
    { startMs: 5000, endMs: 8000, label: { kind: 'session', n: 1 } },       // mapped to B -> correct
    { startMs: 8000, endMs: 9000, label: { kind: 'named', name: 'A' } },    // no speech: confused, not scored
  ];
  assert.deepEqual([...speechIn(turns, 3000, 5000)], [['A', 1000], ['B', 1000]]);
  assert.deepEqual(scoreMeeting(results, turns, new Set(['A', 'B'])),
    { nScored: 3, nAll: 4, namedOk: 1, correct: 2, confusion: 2, uncertain: 0 });
});
