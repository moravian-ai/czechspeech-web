// The decoding limits of one utterance (src/decoding.js): the token cap relative to the audio length, and that the
// recogniser hands the cap and the temperature fallback setting to the engine, which hands them to whisper.cpp.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as decoding from '../../src/decoding.js';
import { tokenCap } from '../../src/decoding.js';
import { WasmEngine } from '../../src/engine-wasm.js';
import { CzechSpeech, CzechSpeechModel } from '../../src/index.js';
import { ArrayAudioSource, Signals, TestEnv, nodeWorkerFactory } from '../support/env.mjs';

const samplesOf = (ms) => Math.round((ms * 16000) / 1000);

test('the cap grows with the audio: 25 tokens per second, rounded up', () => {
  assert.equal(tokenCap(samplesOf(3904)), 98, 'the runaway utterance of the 3-hour soak: 97.6 -> 98');
  assert.equal(tokenCap(samplesOf(6942)), 174, 'e-rec: 173.55 -> 174');
  assert.equal(tokenCap(samplesOf(1300)), 33, 'just above the floor: 32.5 -> 33');
});

test('the token rate comes from the model: tokenCap(n, tokensPerSecond), default 25', () => {
  assert.equal(tokenCap(16000 * 10, 30), 300);
  assert.equal(tokenCap(16000 * 10), 250);
  assert.equal(decoding.decodingLimits(16000 * 10, {}, 30).maxTokens, 300);
  assert.equal(decoding.decodingLimits(16000 * 10, { maxTokens: 5 }, 30).maxTokens, 5, 'a fixed cap still wins');
  assert.equal(decoding.decodingLimits(16000 * 10, { tokenCap: false }, 30).maxTokens, 0);
});

test('short utterances get the floor of 32 tokens, and so does a length that is not a finite number', () => {
  assert.equal(tokenCap(1), 32);
  for (const n of [NaN, Infinity, -Infinity, -5, undefined]) assert.equal(tokenCap(n), 32, String(n));
  assert.equal(tokenCap(samplesOf(500)), 32);
  assert.equal(tokenCap(samplesOf(1280)), 32, '25 x 1.28 s = 32');
});

test('a decode stopped by a token limit has max_tokens + 1 text tokens, at most whisper.cpp\'s own 220', () => {
  const { tokenLimit } = decoding;
  assert.equal(typeof tokenLimit, 'function', 'src/decoding.js exports tokenLimit');
  assert.equal(tokenLimit(5), 6, 'the integration test\'s cap of 5');
  assert.equal(tokenLimit(32), 33, 'the floor: below 1.28 s of audio');
  assert.equal(tokenLimit(98), 99, 'the soak\'s runaway utterance');
  assert.equal(tokenLimit(219), 220, 'both limits at once');
  assert.equal(tokenLimit(220), 220, 'from 8.8 s of audio up whisper.cpp stops first');
  assert.equal(tokenLimit(750), 220);
  assert.equal(tokenLimit(0), 220, 'no cap (tokenCap: false)');
});

test('audio longer than Whisper\'s 30 s window is capped per window, like 30 s', () => {
  assert.equal(tokenCap(samplesOf(30000)), 750);
  assert.equal(tokenCap(samplesOf(95000)), 750);
});

test('the cap is at least twice the tokens of every legitimate fixture transcript measured on WebGPU', () => {
  // [audio ms, text tokens of the transcript] from the measurement behind the cap (README, "Decoding limits"): the
  // highest token rates among 654 clean decodes of ParCzech4Speech (test/e2e fixtures and the numbers bucket) and FLEURS
  // (the meeting app's fixtures), plus the meeting fixture and the transcripts that doubled a phrase
  const measured = [
    [8183, 70],    // ParCzech numbers, 8.55 tokens/s, the highest legitimate rate
    [15611, 123],  // ParCzech numbers, the most tokens
    [6942, 38],    // e-rec
    [3872, 30],    // FLEURS, the soak's runaway sentence when it decodes normally
    [22780, 108],  // the meeting app's fixture schuze-cs (4 utterances in one call)
    [2581, 13],    // the shortest ParCzech segment
  ];
  for (const [ms, tokens] of measured) assert.ok(tokenCap(samplesOf(ms)) >= 2 * tokens, `${ms} ms, ${tokens} tokens: cap ${tokenCap(samplesOf(ms))}`);
  // transcripts that repeated a phrase twice (the repetition guard keeps two copies): not cut either
  for (const [ms, tokens] of [[4384, 50], [5120, 52]]) assert.ok(tokenCap(samplesOf(ms)) > tokens);
});

// ---- the recogniser hands the limits to the engine; the engine to whisper.cpp ----

const fakeEntry = new URL('../support/fake-worker-entry.mjs', import.meta.url);
const vadModel = fs.readFileSync(TestEnv.vadAsset);

async function speechWith(t, engine, meta) {
  const calls = [];
  const languages = [];
  const speech = await CzechSpeech.create({
    model: CzechSpeechModel._fromPath('fake-model.bin', 1000, meta), keepModelLoaded: true, log: () => {},
    _createWorker: nodeWorkerFactory(fakeEntry), _vadModel: vadModel, _engineOptions: engine,
    _onWorkerMessage: (role, m) => { if (m.type === 'fake' && /transcribeStarted/i.test(m.event)) { calls.push(m.limits); languages.push(m.languages); } },
  });
  t.after(() => speech.close()); // also when an assertion fails: the workers would keep the test process alive
  return { speech, calls, languages };
}

for (const gpu of [undefined, 'ok']) {
  const where = gpu ? 'WebGPU' : 'CPU';
  test(`${where}: each recognition gets the cap of its audio and no temperature fallback`, async (t) => {
    const { speech, calls } = await speechWith(t, { gpu });
    await speech.transcribe(Signals.tone(1.0));
    await speech.transcribe(Signals.concat(Signals.tone(3.904)));
    const utt = Signals.concat(Signals.zeros(0.5), Signals.tone(2.0), Signals.zeros(1.0));
    await (async () => { for await (const ev of speech.listen({ _source: new ArrayAudioSource(utt) })) void ev; })();
    assert.equal(calls.length, 3);
    assert.deepEqual(calls[0], { maxTokens: 32, temperatureFallback: false });
    assert.deepEqual(calls[1], { maxTokens: 98, temperatureFallback: false });
    assert.equal(calls[2].temperatureFallback, false);
    assert.ok(calls[2].maxTokens >= 70 && calls[2].maxTokens <= 80, `an utterance of about 2.9 s: ${calls[2].maxTokens}`);
  });
}

test('the cap uses the token rate of the model: 30 tokens per second give 300 on a 10 s utterance', async (t) => {
  const { speech, calls } = await speechWith(t, undefined, { tokensPerSecond: 30 });
  await speech.transcribe(Signals.tone(10));
  assert.equal(calls[0].maxTokens, 300);
});

test('each recognition gets the languages of the model, or the one forced with `language`', async (t) => {
  const { speech, languages } = await speechWith(t, undefined, { languages: ['pl', 'en'] });
  await speech.transcribe(Signals.tone(1.0));
  await speech.transcribe(Signals.tone(1.0), { language: 'en' });
  assert.deepEqual(languages, ['pl,en', 'en']);
  await assert.rejects(speech.transcribe(Signals.tone(1.0), { language: 'de' }), RangeError);
  await assert.rejects(speech.transcribe(Signals.tone(1.0), { language: 'EN' }), RangeError);
  assert.equal(languages.length, 2, 'a refused language never reaches the engine');
});

test('internal engine options switch the cap off and the fallback on (measurements: decoding as before 0.2.0)', async (t) => {
  const { speech, calls } = await speechWith(t, { gpu: 'ok', tokenCap: false, temperatureFallback: true });
  await speech.transcribe(Signals.tone(3.904));
  assert.deepEqual(calls[0], { maxTokens: 0, temperatureFallback: true });
});

test('an internal engine option sets a fixed cap (the integration tests check that whisper.cpp honours it)', async (t) => {
  const { speech, calls } = await speechWith(t, { maxTokens: 5 });
  await speech.transcribe(Signals.tone(3.904));
  assert.deepEqual(calls[0], { maxTokens: 5, temperatureFallback: false });
});

test('the engine passes the languages as a C string and reads the language of the result', () => {
  const heap = new Uint8Array(1 << 20);
  let next = 1024;
  const freed = [];
  const passed = [];
  const M = {
    HEAPU8: heap,
    _malloc: (n) => { const p = next; next += (n + 7) & ~7; return p; },
    _free: (p) => freed.push(p),
    _cs_zero_free() {},
    _cs_text_release() {},
    _cs_whisper_timings() {},
    _cs_text_ptr: () => 0,
    _cs_text_len: () => 0,
    _cs_whisper_transcribe: (...a) => {
      let end = a[7];
      while (heap[end] !== 0) end++;
      passed.push({ nArgs: a.length, text: Buffer.from(heap.subarray(a[7], end)).toString('utf8'), ptr: a[7] });
      return 0;
    },
    _cs_whisper_language: () => 2000,
    UTF8ToString: (p) => { let e = p; while (heap[e] !== 0) e++; return Buffer.from(heap.subarray(p, e)).toString('utf8'); },
  };
  heap.set(Buffer.from('en\0'), 2000);
  const engine = new WasmEngine(M, () => {});
  const r = engine.whisperTranscribe(7, 1, new Float32Array(160), 4, { maxTokens: 98 }, 'pl,en');
  assert.deepEqual(passed.map((p) => [p.nArgs, p.text]), [[8, 'pl,en']], 'languages is the last argument');
  assert.ok(freed.includes(passed[0].ptr), 'the string is freed after the call');
  assert.equal(heap[passed[0].ptr], 0, 'and wiped');
  assert.equal(r.language, 'en', 'from cs_whisper_language');
});

test('the engine passes the cap and the fallback setting to cs_whisper_transcribe', () => {
  const args = [];
  const heap = new Uint8Array(1 << 20);
  let next = 1024;
  const M = {
    HEAPU8: heap,
    _malloc: (n) => { const p = next; next += (n + 7) & ~7; return p; },
    _free() {},
    _cs_zero_free() {},
    _cs_text_release() {},
    _cs_whisper_timings() {},
    _cs_text_ptr: () => 0,
    _cs_text_len: () => 0,
    _cs_whisper_transcribe: (...a) => { args.push(a.slice(4)); return 0; },
    _cs_whisper_tokens: () => 99,
    _cs_whisper_capped: () => 1,
    _cs_whisper_language: () => 0,
    UTF8ToString: () => "cs",
  };
  const engine = new WasmEngine(M, () => {});
  const r = engine.whisperTranscribe(7, 1, new Float32Array(160), 4, { maxTokens: 98, temperatureFallback: false });
  engine.whisperTranscribe(7, 2, new Float32Array(160), 4, { maxTokens: 0, temperatureFallback: true });
  assert.deepEqual(args.map((a) => a.slice(0, 3)), [[4, 98, 0], [4, 0, 1]], '(threads, max_tokens, temperature fallback)');
  assert.equal(r.tokens, 99, 'the tokens decoded, from cs_whisper_tokens');
  assert.equal(r.capped, true, 'the decode stopped at the cap, from cs_whisper_capped');
});
