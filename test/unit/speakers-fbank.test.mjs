import { test } from 'node:test';
import assert from 'node:assert/strict';
import { FBANK, fbank, logMel } from '../../src/speakers/fbank.js';
import { Signals } from '../support/env.mjs';

test('FBANK is the front end the model was trained with (sherpa-onnx settings)', () => {
  assert.deepEqual({ ...FBANK }, { sampleRate: 16000, bins: 80, frameSamples: 400, shiftSamples: 160, fftSize: 512,
    lowHz: 20, highHz: 7600, preemphasis: 0.97, windowPower: 0.85, logFloor: 1.1920929e-7 });
});

test('frames = floor((n + 80) / 160), 80 values each', () => {
  for (const [n, frames] of [[16000, 100], [400, 3], [24000, 150], [79, 0]]) {
    const r = fbank(new Float32Array(n).fill(0.01));
    assert.equal(r.frames, frames, `n=${n}`);
    assert.equal(r.data.length, frames * 80);
  }
});

test('every bin has zero mean over the frames (global-mean normalisation)', () => {
  const r = fbank(Signals.concat(Signals.tone(0.5, 440), Signals.whiteNoise(0.5, 0.1, 7)));
  for (let b = 0; b < 80; b++) {
    let m = 0;
    for (let t = 0; t < r.frames; t++) m += r.data[t * 80 + b];
    assert.ok(Math.abs(m / r.frames) < 1e-4, `bin ${b}`);
  }
});

test('silence gives all zeros once the mean is subtracted', () => {
  const r = fbank(Signals.zeros(1));
  assert.ok(r.data.every((v) => v === 0));
});

test('a tone at the centre frequency of mel bin 20 peaks in bin 20', () => {
  // centre of bin b: mel(20) + (b + 1) * (mel(7600) - mel(20)) / 81, mel(f) = 1127 ln(1 + f/700)
  const mel = (f) => 1127 * Math.log(1 + f / 700);
  const centre = mel(20) + 21 * (mel(7600) - mel(20)) / 81;
  const hz = 700 * (Math.exp(centre / 1127) - 1);          // about 657 Hz
  const lm = logMel(Signals.tone(1, hz, 0.3));
  const row = lm.data.subarray(50 * 80, 51 * 80);
  assert.equal(row.indexOf(Math.max(...row)), 20);
});

test('the input is not modified', () => {
  const x = Signals.tone(0.3, 300);
  const copy = x.slice();
  fbank(x);
  assert.deepEqual(x, copy);
});

test('audio of 80 to 399 samples (frames reflect more than once at the edges) gives finite numbers', () => {
  for (const n of [80, 100, 139, 200, 399]) {
    const r = fbank(Signals.tone(n / 16000, 300));
    assert.equal(r.frames, Math.floor((n + 80) / 160), `n=${n}`);
    assert.ok(r.data.every(Number.isFinite), `n=${n}`);
  }
});
