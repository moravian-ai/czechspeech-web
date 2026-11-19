import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { VOICEPRINT_MODELS, DEFAULT_VOICEPRINT_ID, voiceprintEntry } from '../../src/speakers/catalog.js';
import { MODELS } from '../../src/catalog.js';
import { CzechSpeechError } from '../../src/events.js';

test('the voiceprint model entry: not public, 192 numbers, strict operating point', () => {
  const e = VOICEPRINT_MODELS['voiceprint-cs'];
  assert.equal(DEFAULT_VOICEPRINT_ID, 'voiceprint-cs');
  assert.equal(e.url, null);
  assert.equal(e.dim, 192);
  assert.match(e.sha256, /^[0-9a-f]{64}$/);
  assert.notEqual(e.sha256, '34ecdabf67fabfe66924e1252ad45a3585be874849740ac7d0ac0d29a8cdeecc'); // metadata fixed
  assert.deepEqual([e.operatingPoints.strict.named, e.operatingPoints.strict.session, e.operatingPoints.strict.drop], [0.38, 0.28, 0.4]);
  assert.ok(Object.isFrozen(e) && Object.isFrozen(e.operatingPoints.strict));
});

test('the Czech speech model points at its voiceprint model', () => {
  assert.equal(MODELS.cs.voiceprint, 'voiceprint-cs');
});

test('unknown id', () => {
  assert.throws(() => voiceprintEntry('nope'), (e) => e instanceof CzechSpeechError && e.code === 'unknown-model');
});

test('the catalog matches the file, when CZECHSPEECH_VOICEPRINT_MODEL is set', { skip: !process.env.CZECHSPEECH_VOICEPRINT_MODEL && 'CZECHSPEECH_VOICEPRINT_MODEL not set' }, () => {
  const bytes = fs.readFileSync(process.env.CZECHSPEECH_VOICEPRINT_MODEL);
  const e = VOICEPRINT_MODELS['voiceprint-cs'];
  assert.equal(bytes.length, e.size);
  assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'), e.sha256);
});
