import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Framer, Resampler } from '../../src/resampler.js';
import { generateWorklet } from '../../scripts/build-worklet.mjs';
import fs from 'node:fs';
import { ROOT } from '../support/env.mjs';

function sine(rate, hz, seconds, amplitude = 0.5) {
  const out = new Float32Array(Math.round(rate * seconds));
  for (let i = 0; i < out.length; i++) out[i] = amplitude * Math.sin((2 * Math.PI * hz * i) / rate);
  return out;
}

function resampleAll(inRate, input, blockSizes = [input.length], outRate = 16000) {
  const r = new Resampler(inRate, outRate);
  const parts = [];
  let pos = 0;
  let k = 0;
  while (pos < input.length) {
    const n = Math.min(blockSizes[k++ % blockSizes.length], input.length - pos);
    const out = new Float32Array(r.maxOutput(n));
    const m = r.process(input.subarray(pos, pos + n), out);
    parts.push(out.slice(0, m));
    pos += n;
  }
  const total = parts.reduce((s, p) => s + p.length, 0);
  const all = new Float32Array(total);
  let o = 0;
  for (const p of parts) {
    all.set(p, o);
    o += p.length;
  }
  return all;
}

const rms = (a, from = 0, to = a.length) => {
  let s = 0;
  for (let i = from; i < to; i++) s += a[i] * a[i];
  return Math.sqrt(s / (to - from));
};

/** Amplitude of frequency `hz` in `a` (sampled at `rate`) by correlation with sine and cosine. */
function toneAmplitude(a, rate, hz, from, to) {
  let c = 0;
  let s = 0;
  for (let i = from; i < to; i++) {
    c += a[i] * Math.cos((2 * Math.PI * hz * i) / rate);
    s += a[i] * Math.sin((2 * Math.PI * hz * i) / rate);
  }
  return (2 * Math.hypot(c, s)) / (to - from);
}

test('48 kHz -> 16 kHz keeps a 1 kHz tone (amplitude within 1 %)', () => {
  const out = resampleAll(48000, sine(48000, 1000, 1));
  assert.ok(out.length <= 16000 && out.length >= 16000 - 20, `length ${out.length}`); // minus the filter delay
  const amp = toneAmplitude(out, 16000, 1000, 1000, 15000);
  assert.ok(Math.abs(amp - 0.5) < 0.005, `amplitude ${amp}`);
});

test('48 kHz -> 16 kHz removes a 9 kHz tone (above the new Nyquist frequency) by more than 60 dB', () => {
  const out = resampleAll(48000, sine(48000, 9000, 1));
  const level = rms(out, 1000, 15000) / (0.5 / Math.SQRT2);
  assert.ok(20 * Math.log10(level) < -60, `alias level ${(20 * Math.log10(level)).toFixed(1)} dB`);
});

test('44.1 kHz -> 16 kHz: exact output length, 3 kHz tone kept', () => {
  const input = sine(44100, 3000, 2);
  const out = resampleAll(44100, input);
  const expected = Math.floor(((input.length - 1) * 16000) / 44100);
  assert.ok(Math.abs(out.length - expected) <= 40, `length ${out.length} vs ${expected}`); // minus the filter delay
  const amp = toneAmplitude(out, 16000, 3000, 1000, 30000);
  assert.ok(Math.abs(amp - 0.5) < 0.005, `amplitude ${amp}`);
});

test('the output does not depend on how the input is split into blocks', () => {
  const input = sine(44100, 440, 0.5);
  for (let i = 0; i < input.length; i++) input[i] += 0.1 * Math.sin(i * 0.37);
  const whole = resampleAll(44100, input);
  const blocks = resampleAll(44100, input, [128, 7, 1000, 1, 333]);
  assert.equal(blocks.length, whole.length);
  for (let i = 0; i < whole.length; i++) assert.equal(blocks[i], whole[i]);
});

test('16 kHz input passes through unchanged', () => {
  const input = sine(16000, 500, 0.1);
  const out = resampleAll(16000, input, [128]);
  assert.deepEqual(Array.from(out), Array.from(input));
});

test('8 kHz -> 16 kHz upsampling keeps a 1 kHz tone', () => {
  const out = resampleAll(8000, sine(8000, 1000, 1), [128]);
  assert.ok(Math.abs(out.length - 16000) <= 70);
  const amp = toneAmplitude(out, 16000, 1000, 1000, 15000);
  assert.ok(Math.abs(amp - 0.5) < 0.01, `amplitude ${amp}`);
});

test('clear() zeroes the buffered input', () => {
  const r = new Resampler(48000);
  const out = new Float32Array(r.maxOutput(128));
  r.process(sine(48000, 1000, 128 / 48000), out);
  r.clear();
  assert.ok(r.hist.every((v) => v === 0));
});

test('Framer emits 512-sample frames in order and zeroes a partial frame on clear()', () => {
  const frames = [];
  const f = new Framer((frame) => frames.push(frame));
  const input = Float32Array.from({ length: 1300 }, (_, i) => i);
  f.push(input.subarray(0, 100));
  f.push(input.subarray(100, 1100));
  f.push(input, 1300 - 1100 + 1100); // push(samples, count): the first 1300 again
  assert.equal(frames.length, Math.floor((100 + 1000 + 1300) / 512));
  assert.equal(frames[0][0], 0);
  assert.equal(frames[0][511], 511);
  assert.equal(frames[1][0], 512);
  assert.ok(frames.every((fr) => fr.length === 512));
  f.clear();
  assert.ok(f.frame.every((v) => v === 0));
});

test('src/worklet.js is generated from src/resampler.js (run node scripts/build-worklet.mjs)', () => {
  assert.equal(fs.readFileSync(`${ROOT}/src/worklet.js`, 'utf8'), generateWorklet());
});
