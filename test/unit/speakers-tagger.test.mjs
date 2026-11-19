import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Tagger } from '../../src/speakers/tagger.js';
import { STRICT } from '../../src/speakers/points.js';
import { unit } from '../../src/speakers/maths.js';

const v = (...xs) => unit(Float32Array.from(xs));
const e1 = v(1, 0, 0, 0), e2 = v(0, 1, 0, 0), e3 = v(0, 0, 1, 0);

test('voices get Mluvci 1, 2 in order of first speech; short unknown pieces stay unknown', () => {
  const t = new Tagger({ point: STRICT });
  assert.deepEqual(t.tag(e1, 2000), { kind: 'session', n: 1 });
  assert.deepEqual(t.tag(e2, 2000), { kind: 'session', n: 2 });
  assert.deepEqual(t.tag(e1, 500), { kind: 'session', n: 1 });   // short, but similar to an existing speaker
  assert.equal(t.tag(e3, 999), null);                             // short and new: no new speaker
  assert.deepEqual(t.tag(e3, 1000), { kind: 'session', n: 3 });
});

test('firstLabel continues the numbering', () => {
  assert.deepEqual(new Tagger({ point: STRICT, firstLabel: 4 }).tag(e1, 2000), { kind: 'session', n: 4 });
});

test('the centroid is the normalised sum of its pieces', () => {
  const t = new Tagger({ point: STRICT });
  t.tag(e1, 2000);
  t.tag(v(1, 1, 0, 0), 2000);                                      // similarity 0.707 >= 0.28
  const c = t.centroid(1);
  assert.ok(Math.abs(c[0] - 0.9238795) < 1e-6 && Math.abs(c[1] - 0.3826834) < 1e-6);
  assert.deepEqual(t.speakers, [{ n: 1, count: 2, speechMs: 4000 }]);
});

test('a known voice above the length-dependent threshold wins over session speakers', () => {
  const known = [{ key: 'A', vector: e1 }];
  const t = new Tagger({ point: STRICT, known });
  const at = (cos) => v(cos, Math.sqrt(1 - cos * cos), 0, 0);
  assert.deepEqual(t.tag(at(0.39), 3000), { kind: 'known', key: 'A' });    // 0.39 >= 0.38
  assert.deepEqual(t.tag(at(0.36), 1500), { kind: 'known', key: 'A' });    // 1.5 s: threshold 0.3453
  assert.deepEqual(t.tag(at(0.36), 3000), { kind: 'session', n: 1 });      // 3 s: 0.36 < 0.38
});

test('above the threshold without margin is unknown and founds no speaker', () => {
  const point = { ...STRICT, margin: 0.05 };
  const t = new Tagger({ point, known: [{ key: 'A', vector: e1 }, { key: 'B', vector: v(0.96, 0.28, 0, 0) }] });
  assert.equal(t.tag(v(0.99, 0.141, 0, 0), 3000), null);
  assert.deepEqual(t.speakers, []);
});

test('no more than maxLabels speakers', () => {
  const t = new Tagger({ point: STRICT, maxLabels: 2 });
  t.tag(e1, 2000); t.tag(e2, 2000);
  assert.equal(t.tag(e3, 2000), null);
});

test('clear() zeroes the sums', () => {
  const t = new Tagger({ point: STRICT });
  t.tag(e1, 2000);
  t.clear();
  assert.deepEqual(t.speakers, []);
});

test('clear() forgets the stored voices as well as the session speakers', () => {
  const t = new Tagger({ point: STRICT, known: [{ key: 'ana', vector: e1 }] });
  assert.deepEqual(t.tag(e1, 3000), { kind: 'known', key: 'ana' });
  t.clear();
  assert.notEqual(t.tag(e1, 3000)?.kind, 'known');
});

// Float32Arrays made while `fn` runs (the tagger's own arrays: they are not reachable otherwise)
function trackArrays(fn) {
  const Orig = globalThis.Float32Array;
  const made = [];
  globalThis.Float32Array = new Proxy(Orig, { construct(target, args) { const a = Reflect.construct(target, args); made.push(a); return a; } });
  try { fn(); } finally { globalThis.Float32Array = Orig; }
  return made;
}
const isZero = (a) => a.every((x) => x === 0);

test('the centroid array replaced by a new piece is zeroed', () => {
  const t = new Tagger({ point: STRICT });
  const made = trackArrays(() => {
    t.tag(e1, 2000);
    t.tag(v(1, 1, 0, 0), 2000);
  });
  const zero = made.filter(isZero);
  assert.ok(made.length >= 3, 'arrays were seen');
  assert.equal(zero.length, 1, 'exactly the old centroid is zero');
});

test('clear() zeroes the known voices and the session arrays themselves', () => {
  let t;
  const made = trackArrays(() => {
    t = new Tagger({ point: STRICT, known: [{ key: 'ana', vector: e1 }] });
    t.tag(e2, 2000);
  });
  assert.ok(made.filter((a) => !isZero(a)).length >= 3, 'known copy, sum and centroid are alive');
  t.clear();
  assert.ok(made.every(isZero), 'every array the tagger made is zero after clear()');
});
