// The encoder window whisper.cpp gets for an utterance (cs_audio_ctx in native/czechspeech_wasm.c), asked of the real
// WebAssembly (the Node.js link of the CPU objects, no model). The model was never trained on a window shorter than
// 512 frames (about 10 s); trimmed shorter, it invents and repeats sentences on short utterances. So the window is
// trimmed to the utterance only above that floor. Skips when build/wasm-node is not built (native/build-wasm.sh).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { ROOT } from '../support/env.mjs';

const glue = path.join(ROOT, 'build/wasm-node/czechspeech-st-node.js');
const skip = fs.existsSync(glue) ? false : 'build/wasm-node/czechspeech-st-node.js not built; run native/build-wasm.sh';
if (skip) console.log(`SKIP: ${skip}`);

async function audioCtx() {
  const { default: create } = await import(pathToFileURL(glue).href);
  const mod = await create();
  return (seconds) => mod._cs_audio_ctx(Math.floor(seconds * 16000));
}

test('a short utterance gets the floor of 512 frames', { skip }, async () => {
  const ctx = await audioCtx();
  for (const s of [0, 0.5, 1, 2, 3.9, 4, 8, 9.5]) assert.equal(ctx(s), 512, `${s} s`);
});

test('a longer utterance is trimmed to its length', { skip }, async () => {
  const ctx = await audioCtx();
  assert.equal(ctx(10), 544); // 500 + 32 frames, up to a multiple of 16
  assert.equal(ctx(19.5), 1008);
  assert.equal(ctx(29.1), 1488);
});

test('the full window from 30 s up', { skip }, async () => {
  const ctx = await audioCtx();
  for (const s of [29.5, 30, 120]) assert.equal(ctx(s), 0, `${s} s`);
});
