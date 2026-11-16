import { test } from 'node:test';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import util from 'node:util';
import { Sha256, sha256Hex } from '../../src/sha256.js';
import { DEFAULT_CONFIG, resolveConfig, threadCount } from '../../src/config.js';
import { TranscriptEvent, CzechSpeechError } from '../../src/events.js';

const ref = (b) => crypto.createHash('sha256').update(b).digest('hex');

test('SHA-256 matches node:crypto for empty, short, block-boundary and random inputs', () => {
  assert.equal(sha256Hex(new Uint8Array(0)), ref(Buffer.alloc(0)));
  assert.equal(sha256Hex(new TextEncoder().encode('abc')), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
  for (const n of [55, 56, 63, 64, 65, 119, 120, 1000, 100003]) {
    const b = crypto.randomBytes(n);
    assert.equal(sha256Hex(b), ref(b), `n=${n}`);
  }
});

test('SHA-256 is the same however the input is split', () => {
  const b = crypto.randomBytes(300001);
  const h = new Sha256();
  for (let pos = 0, k = 1; pos < b.length; pos += k, k = (k * 7 + 3) % 4099 + 1) h.update(b.subarray(pos, pos + k));
  assert.equal(h.hex(), ref(b));
});

test('config defaults are the Android CzechSpeechConfig defaults (plus the web-only backend)', () => {
  assert.deepEqual({ ...resolveConfig() }, {
    threshold: 0.25, negThreshold: 0.1, minSilenceMs: 600, preRollMs: 300, minSpeechMs: 250,
    maxUtteranceMs: 28000, threads: 0, backend: 'auto',
  });
  assert.ok(Object.isFrozen(DEFAULT_CONFIG));
});

test('config is validated like the Kotlin init block', () => {
  assert.throws(() => resolveConfig({ threshold: 0 }), RangeError);
  assert.throws(() => resolveConfig({ threshold: 0.2, negThreshold: 0.3 }), RangeError);
  assert.throws(() => resolveConfig({ minSilenceMs: 31 }), RangeError);
  assert.throws(() => resolveConfig({ preRollMs: 2001 }), RangeError);
  assert.throws(() => resolveConfig({ maxUtteranceMs: 30000 }), RangeError);
  assert.throws(() => resolveConfig({ threads: 65 }), RangeError);
  assert.throws(() => resolveConfig({ minSilence: 600 }), TypeError);
  assert.equal(resolveConfig({ minSilenceMs: 800 }).minSilenceMs, 800);
});

test('thread count: half the logical cores, 1..8; 1 without SharedArrayBuffer; explicit value wins', () => {
  const c = resolveConfig();
  assert.equal(threadCount(c, true, 22), 8);
  assert.equal(threadCount(c, true, 8), 4);
  assert.equal(threadCount(c, true, 2), 1);
  assert.equal(threadCount(c, true, undefined), 1);
  assert.equal(threadCount(c, false, 16), 1);
  assert.equal(threadCount(resolveConfig({ threads: 6 }), true, 4), 6);
});

test('transcripts do not print their text in toString, JSON or util.inspect', () => {
  const t = new TranscriptEvent('dobrý den', 480, 1520, 33);
  assert.equal(t.text, 'dobrý den');
  for (const s of [String(t), JSON.stringify(t), util.inspect(t), `${t}`, JSON.stringify({ ev: t })]) {
    assert.ok(!s.includes('dobrý'), s);
    assert.ok(s.includes('9 chars'), s);
  }
  assert.ok(Object.isFrozen(t));
});

test('transcripts name their language in toString and JSON, still without the text', () => {
  const t = new TranscriptEvent('dobrý den', 480, 1520, 33, 'cs');
  assert.equal(t.language, 'cs');
  assert.ok(JSON.stringify(t).includes('"language":"cs"'), JSON.stringify(t));
  assert.ok(!JSON.stringify(t).includes('dobrý'));
  assert.ok(String(t).includes('language=cs'), String(t));
});

test('CzechSpeechError carries a code and a cause', () => {
  const cause = new Error('x');
  const e = new CzechSpeechError('boom', { code: 'model', cause });
  assert.equal(e.name, 'CzechSpeechError');
  assert.equal(e.code, 'model');
  assert.equal(e.cause, cause);
  assert.ok(e instanceof Error);
});
