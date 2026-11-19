import { test } from 'node:test';
import assert from 'node:assert/strict';
import { changePoints, dot, namedThreshold, speechSpan, unit, validCuts, windowStarts } from '../../src/speakers/maths.js';

const close = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);

test('namedThreshold grows with the log of the duration (strict point: named 0.38, slope 0.05)', () => {
  close(namedThreshold(0.38, 3000, 0.05), 0.38);
  close(namedThreshold(0.38, 1500, 0.05), 0.3453426);
  close(namedThreshold(0.38, 10000, 0.05), 0.4401986);
  close(namedThreshold(0.38, 100, 0.05), 0.2445975);   // below 200 ms counts as 200 ms
});

test('unit and dot', () => {
  const u = unit(Float32Array.from([3, 4]));
  close(u[0], 0.6); close(u[1], 0.8);
  close(dot(u, Float32Array.from([1, 0])), 0.6);
  assert.deepEqual([...unit(new Float32Array(2))], [0, 0]);
});

test('windowStarts: none below two windows, then every hop', () => {
  assert.deepEqual(windowStarts(47999, 24000, 8000), []);
  assert.deepEqual(windowStarts(48000, 24000, 8000), [0, 8000, 16000, 24000]);
  assert.deepEqual(windowStarts(64000, 24000, 8000), [0, 8000, 16000, 24000, 32000, 40000]);
});

test('changePoints: local maxima at or above drop', () => {
  const opts = { firstMs: 1500, hopMs: 500, winMs: 1500, drop: 0.4 };
  assert.deepEqual(changePoints([0.1, 0.5, 0.3, 0.45, 0.9, 0.2], opts), [2000, 3500]);
});

test('changePoints: a stronger change pushes out a weaker one closer than one window', () => {
  const opts = { firstMs: 1500, hopMs: 500, winMs: 1500, drop: 0.4 };
  assert.deepEqual(changePoints([0.5, 0.2, 0.6], opts), [2500]);
});

test('validCuts keeps at least minMs from the start, the previous cut and the end', () => {
  assert.deepEqual(validCuts([800, 2000, 2600, 4200], 5000, 1000), [2000]);
  assert.deepEqual(validCuts([1000, 2000], 3000, 1000), [1000, 2000]);
});

test('speechSpan drops the trailing silence the recogniser gets', () => {
  const s = new Float32Array(44 * 512);
  assert.equal(speechSpan(s, 0, 800).length, 12800);
  assert.equal(speechSpan(s, 160, 1120).length, 15360);
  assert.equal(speechSpan(s, 0, 999999).length, s.length);
});
