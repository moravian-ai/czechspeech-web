// The model catalog: models are picked by id; an entry carries its source, languages and token rate.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { MODELS, DEFAULT_MODEL_ID, catalogEntry } from '../../src/catalog.js';
import { DEFAULT_MODEL } from '../../src/model.js';
import { CzechSpeechError } from '../../src/events.js';

test('the catalog has the Czech model, frozen, with the pinned source', () => {
  assert.equal(DEFAULT_MODEL_ID, 'cs');
  assert.deepEqual(MODELS.cs.source, DEFAULT_MODEL);
  assert.deepEqual(MODELS.cs.languages, ['cs']);
  assert.equal(MODELS.cs.tokensPerSecond, 25);
  assert.equal(MODELS.cs.id, 'cs');
  assert.ok(Object.isFrozen(MODELS));
  assert.ok(Object.isFrozen(MODELS.cs));
  assert.ok(Object.isFrozen(MODELS.cs.languages));
  assert.equal(catalogEntry('cs'), MODELS.cs);
});

test('every entry has a positive whole-number token rate', () => {
  for (const e of Object.values(MODELS)) assert.ok(Number.isInteger(e.tokensPerSecond) && e.tokensPerSecond > 0, e.id);
});

test('catalogEntry() rejects an unknown id with code unknown-model', () => {
  for (const id of ['xx', '', null, undefined, 'toString', '__proto__']) {
    assert.throws(() => catalogEntry(id), (e) => e instanceof CzechSpeechError && e.code === 'unknown-model', String(id));
  }
});
