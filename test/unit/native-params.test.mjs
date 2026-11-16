// The whisper_full parameters live in C and cannot be read back without a model, so this guards the source: whisper.cpp
// keeps suppress_blank on by default, which forbids an empty transcript as the first token. A model that learned to
// answer a segment without speech with an empty transcript then has to write something into silence.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { ROOT } from '../support/env.mjs';

test('suppress_blank is off in cs_whisper_transcribe', () => {
  const src = fs.readFileSync(path.join(ROOT, 'native/czechspeech_wasm.c'), 'utf8');
  assert.match(src, /params\.suppress_blank\s*=\s*false\s*;/);
});
