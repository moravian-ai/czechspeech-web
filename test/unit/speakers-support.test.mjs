import { test } from 'node:test';
import assert from 'node:assert/strict';
import { speakerSupport } from '../../src/speakers/support.js';
import * as api from '../../src/index.js';

const env = (o = {}) => ({ WebAssembly, Worker: class {}, crossOriginIsolated: true, navigator: { hardwareConcurrency: 22 }, ...o });

test('two threads on an isolated page with 8+ cores, else one', () => {
  assert.deepEqual(speakerSupport(env()), { supported: true, threads: 2, reason: null });
  assert.equal(speakerSupport(env({ crossOriginIsolated: false })).threads, 1);
  assert.equal(speakerSupport(env({ navigator: { hardwareConcurrency: 4 } })).threads, 1);
});

test('no WebAssembly SIMD or no workers: not supported, with the reason', () => {
  assert.deepEqual(speakerSupport(env({ WebAssembly: { validate: () => false } })), { supported: false, threads: 0, reason: 'no-webassembly-simd' });
  assert.deepEqual(speakerSupport(env({ Worker: undefined })), { supported: false, threads: 0, reason: 'no-workers' });
});

test('this runtime knows WebAssembly SIMD (the module it probes with is valid)', () => {
  assert.equal(speakerSupport({ WebAssembly, Worker: class {} }).supported, true);
});

test('the package exports the speaker API', () => {
  for (const name of ['SpeakerSession', 'SpeakersEvent', 'VoiceprintModel', 'VOICEPRINT_MODELS', 'DEFAULT_VOICEPRINT_ID', 'speakerSupport']) assert.ok(name in api, name);
  assert.equal(api.VERSION, '0.3.0');
});

test('the Czech model names its voiceprint model', () => {
  assert.equal(api.MODELS.cs.voiceprint, 'voiceprint-cs');
});
