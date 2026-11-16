// The whole pipeline in Node.js: real whisper.cpp + Silero VAD WebAssembly (the Node links of the same objects,
// build/wasm-node/), the real CzechSpeech model and recordings fed through ArrayAudioSource in place of the
// microphone, with the capture and recogniser workers in worker threads. Same cases as the Android SDK's
// IntegrationTest. Skips, naming the variable, when an input is missing.
//
// Env: CZECHSPEECH_MODEL, CZECHSPEECH_TEST_AUDIO, CZECHSPEECH_TEST_MANIFEST, CZECHSPEECH_TEST_THREADS (default 6).
//
// The test prints transcripts of the (public) parliament recording to check them by eye; the SDK never logs text.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { CzechSpeech, CzechSpeechModel } from '../../src/index.js';
import { loadWasmEngine } from '../../src/engine-wasm.js';
import { ArrayAudioSource, ROOT, Signals, TestEnv, missing, nodeWorkerFactory, readWav, reference, wordRecall }
  from '../support/env.mjs';

const SPEECH_ID = 'test_0000-0000000333';
const nodeWasm = path.join(ROOT, 'build/wasm-node');
const realEntry = pathToFileURL(path.join(ROOT, 'test/support/real-worker-entry.mjs'));

const skip = missing('CZECHSPEECH_MODEL', TestEnv.model, 'CzechSpeech model') ||
  missing('CZECHSPEECH_TEST_AUDIO', TestEnv.audioDir, 'test audio directory') ||
  (fs.existsSync(path.join(nodeWasm, 'czechspeech-mt-node.js')) ? null
    : 'build/wasm-node/czechspeech-mt-node.js not built; run native/build-wasm.sh') || false;
if (skip) console.log(`SKIP: ${skip}`);

const vadModel = fs.readFileSync(TestEnv.vadAsset);
const audio = (name) => readWav(path.join(TestEnv.audioDir, name));
function ref() {
  const why = missing('CZECHSPEECH_TEST_MANIFEST', TestEnv.manifest, 'ParCzech manifest');
  if (why) return { skip: why };
  const r = reference(TestEnv.manifest, SPEECH_ID);
  return r ? { text: r } : { skip: `manifest entry ${SPEECH_ID} not found` };
}

async function engine(extra = {}, modelOptions) {
  const logs = [];
  const size = fs.statSync(TestEnv.model).size;
  const speech = await CzechSpeech.create({
    model: CzechSpeechModel._fromPath(TestEnv.model, size, modelOptions),
    config: { threads: TestEnv.threads },
    log: (level, message) => logs.push(`${level}: ${message}`),
    _createWorker: nodeWorkerFactory(realEntry),
    _vadModel: vadModel,
    _assets: {
      mt: pathToFileURL(path.join(nodeWasm, 'czechspeech-mt-node.js')).href, mtWasm: null,
      st: pathToFileURL(path.join(nodeWasm, 'czechspeech-st-node.js')).href, stWasm: null,
    },
    ...extra,
  });
  return { speech, logs };
}

const seconds = (n) => (n / 16000).toFixed(1);

async function listen(name, samples, extra, modelOptions, listenOptions) {
  const { speech, logs } = await engine(extra, modelOptions);
  const t0 = performance.now();
  const events = [];
  for await (const ev of speech.listen({ _source: new ArrayAudioSource(samples), ...listenOptions })) events.push(ev);
  const ms = Math.round(performance.now() - t0);
  console.log(`listen(${name}, ${seconds(samples.length)} s of audio) took ${ms} ms:`);
  for (const e of events) {
    if (e.type === 'transcript') console.log(`  Transcript ${e.startMs}-${e.endMs} ms, ${e.processingMs} ms: "${e.text}"`);
    else console.log(`  ${e.type}${e.atMs !== undefined ? ` at ${e.atMs} ms` : ''}${e.error ? `: ${e.error.message}` : ''}`);
  }
  for (const l of logs.filter((x) => /model loaded|recogniser|capture:/.test(x))) console.log(`    ${l}`);
  assert.equal(speech.state.modelLoaded, false, 'model freed after the session');
  assert.ok(!logs.some((l) => /pojišťování|vývozu|Děkuji/.test(l)), 'no transcript text in the logs');
  return { events, speech, logs };
}

const transcripts = (events) => events.filter((e) => e.type === 'transcript');

test('non-speech recordings give no transcript, and transcribe() does not even load the model', { skip }, async () => {
  const cases = ['a-silence-5s.wav', 'b-pink-noise-2s.wav', 'c-hum-2s.wav', 'd-clicks-6s.wav']
    .map((n) => [n, audio(n)]).concat([['digital zeros 10 s', Signals.zeros(10)]]);
  for (const [name, samples] of cases) {
    const { events } = await listen(name, samples);
    assert.equal(transcripts(events).length, 0, `${name}: no transcript`);
    assert.ok(!events.some((e) => e.type === 'error'), `${name}: no error`);
    const { speech } = await engine();
    assert.equal(await speech.transcribe(samples), '', `${name}: transcribe() gives ""`);
    assert.equal(speech.state.modelLoads, 0, `${name}: the gate keeps the model unloaded`);
  }
});

test('parliament speech is transcribed (word recall >= 0.8)', { skip }, async (t) => {
  const r = ref();
  if (r.skip) return t.skip(r.skip);
  const samples = audio('e-rec.wav');
  const copy = samples.slice();
  const { speech, logs } = await engine();
  const t0 = performance.now();
  const text = await speech.transcribe(samples);
  const ms = Math.round(performance.now() - t0);
  const recall = wordRecall(r.text, text);
  console.log(`transcribe(e-rec.wav, ${seconds(samples.length)} s) in ${ms} ms with ${speech.state.threads} threads ` +
    `(model load ${speech.state.lastLoadMs} ms, peak heap ${Math.round(speech.state.peakHeapBytes / 2 ** 20)} MiB): "${text}"`);
  console.log(`reference: "${r.text}"`);
  console.log(`word recall ${recall.toFixed(2)}`);
  for (const l of logs.filter((x) => /recogniser|model loaded/.test(x))) console.log(`    ${l}`);
  assert.ok(recall >= 0.8, `word recall ${recall}`);
  assert.deepEqual(samples, copy, 'the caller\'s samples are not modified');
  assert.equal(speech.state.modelLoaded, false);
  assert.equal(speech.state.modelLoads, 1);
  // config.backend 'auto' in Node.js: no JSPI (Node 24) and no navigator.gpu, so the CPU module, as in a browser
  // without WebGPU
  assert.equal(speech.state.backend, 'cpu');
  assert.match(speech.state.backendFallback, /JSPI .* is not available|navigator\.gpu is not available/);
  assert.equal(logs.filter((l) => /WebGPU not used/.test(l)).length, 1, 'one notice');
});

test('whisper.cpp stops at the token cap; the default cap leaves the transcript whole', { skip }, async (t) => {
  // src/decoding.js: max_tokens relative to the audio length reaches whisper.cpp through cs_whisper_transcribe (before
  // the glue had no such argument, and a repetition loop decoded until 220 tokens)
  const r = ref();
  if (r.skip) return t.skip(r.skip);
  const samples = audio('e-rec.wav');
  const runs = [];
  const run = async (engineOptions) => {
    const { speech, logs } = await engine({ keepModelLoaded: true, _engineOptions: engineOptions });
    try {
      const text = await speech.transcribe(samples);
      runs.push({ capped: speech.state.cappedRecognitions, logs });
      return text;
    } finally {
      await speech.close();
    }
  };
  const capped = await run(undefined);           // the default: 25 tokens per second, 174 for these 6.9 s
  const uncapped = await run({ tokenCap: false, temperatureFallback: true }); // whisper.cpp's defaults
  const five = await run({ maxTokens: 5 });
  const n = (s) => s.split(/\s+/).filter(Boolean).length;
  console.log(`e-rec: ${n(capped)} words with the default cap, ${n(uncapped)} without a cap, ${n(five)} with a cap of 5 tokens`);
  assert.equal(capped, uncapped, 'the cap does not cut a legitimate transcript');
  assert.ok(wordRecall(r.text, capped) >= 0.8);
  assert.ok(n(five) >= 1 && n(five) <= 5, `5 tokens: ${n(five)} words`);
  assert.ok(wordRecall(r.text, five) < 0.5, 'cut off after 5 tokens');
  assert.deepEqual(runs.map((x) => x.capped), [0, 0, 1], 'only the decode with the cap of 5 tokens reached its cap');
  assert.ok(runs[2].logs.some((l) => /^info: transcribe: token limit reached \(6 tokens; the cap allows 6\)$/.test(l)), runs[2].logs.join('\n'));
});

test('config.backend webgpu without WebGPU fails with code unsupported before anything is loaded', { skip }, async () => {
  const { speech } = await engine({ config: { threads: TestEnv.threads, backend: 'webgpu' } });
  await assert.rejects(speech.transcribe(audio('e-rec.wav')),
    (e) => e.code === 'unsupported' && /JSPI .* is not available|navigator\.gpu is not available/.test(e.message));
  assert.equal(speech.state.modelLoads, 0);
});

// The WebGPU recogniser needs a browser: Node.js has no navigator.gpu (no WebGPU adapter), and czechspeech-webgpu is
// built for browser workers only. Its integration cases (these fixtures on the GPU, CPU vs GPU word differences) run
// in headless Chromium: npm run test:e2e (part D) and test/e2e/accuracy.mjs.
test('WebGPU: the fixtures on the GPU', {
  skip: 'no WebGPU adapter in Node.js (navigator.gpu is undefined); the WebGPU path is tested in the browser: npm run test:e2e',
}, () => {});

test('sessions are deterministic', { skip }, async () => {
  const speech = audio('e-rec.wav');
  const samples = Signals.concat(Signals.whiteNoise(0.5, 0.01), speech, Signals.zeros(2), speech);
  const timeline = (events) => events.map((e) => (e.type === 'speechStarted' ? `S${e.atMs}` : e.type === 'speechEnded'
    ? `E${e.atMs}` : e.type === 'transcript' ? `T${e.startMs}-${e.endMs}` : 'error'));
  const first = timeline((await listen('determinism run 1', samples)).events);
  const second = timeline((await listen('determinism run 2', samples)).events);
  assert.deepEqual(second, first);
  assert.equal(first.filter((x) => x.startsWith('T')).length, 2);
});

test('cancelling a transcription stops whisper.cpp at its abort check, and the engine still works', { skip }, async (t) => {
  const r = ref();
  if (r.skip) return t.skip(r.skip);
  const speech = audio('e-rec.wav');
  const long = Signals.concat(speech, speech, speech, speech); // ~28 s, many seconds of encoder work
  // A long grace period, so that the worker is not terminated and the abort callback has to do the job.
  const { speech: s, logs } = await engine({ _abortGraceMs: 600000 });
  const ac = new AbortController();
  let t0 = performance.now();
  const p = s.transcribe(long, { signal: ac.signal });
  while (!s.state.modelLoaded) await new Promise((res) => setTimeout(res, 5));
  await new Promise((res) => setTimeout(res, 300)); // whisper is encoding now
  const loadedAfter = Math.round(performance.now() - t0);
  t0 = performance.now();
  ac.abort();
  await assert.rejects(p, (e) => e.name === 'AbortError');
  const rejectMs = Math.round(performance.now() - t0);
  while (s.state.modelLoaded) await new Promise((res) => setTimeout(res, 20));
  const freedMs = Math.round(performance.now() - t0);
  console.log(`cancelled ${loadedAfter} ms into a 28 s transcription: rejected after ${rejectMs} ms, model freed ` +
    `${freedMs} ms after the cancel (the encoder pass finishes first)`);
  for (const l of logs.filter((x) => /cancelled|model/.test(x))) console.log(`    ${l}`);
  assert.equal(s.state.abortedRecognitions, 1, 'whisper stopped by the abort callback');
  assert.ok(rejectMs < 100, `transcribe() rejected ${rejectMs} ms after the abort`);
  // the instance still works afterwards
  const text = await s.transcribe(speech);
  assert.ok(wordRecall(r.text, text) >= 0.8);
});

test('with the default grace period a cancelled encoder pass is cut short by terminating the worker', { skip }, async () => {
  const speech = audio('e-rec.wav');
  const { speech: s } = await engine();
  const ac = new AbortController();
  const p = s.transcribe(Signals.concat(speech, speech, speech, speech), { signal: ac.signal });
  while (!s.state.modelLoaded) await new Promise((res) => setTimeout(res, 5));
  await new Promise((res) => setTimeout(res, 300));
  const t0 = performance.now();
  ac.abort();
  await assert.rejects(p, (e) => e.name === 'AbortError');
  while (s.state.modelLoaded) await new Promise((res) => setTimeout(res, 20));
  const freedMs = Math.round(performance.now() - t0);
  console.log(`default grace: model freed ${freedMs} ms after the cancel`);
  assert.ok(freedMs < 2500, `freed after ${freedMs} ms`);
});

test('speech followed by 15 s of silence gives exactly one transcript', { skip }, async (t) => {
  const r = ref();
  if (r.skip) return t.skip(r.skip);
  const { events } = await listen('f-rec-plus-silence15s.wav', audio('f-rec-plus-silence15s.wav'));
  const ts = transcripts(events);
  assert.equal(ts.length, 1);
  assert.equal(events.filter((e) => e.type === 'speechStarted').length, 1);
  assert.equal(events.filter((e) => e.type === 'speechEnded').length, 1);
  assert.ok(wordRecall(r.text, ts[0].text) >= 0.8);
  assert.ok(ts[0].startMs < ts[0].endMs && ts[0].endMs <= 7200);
});

test('two copies two seconds apart give two transcripts', { skip }, async (t) => {
  const r = ref();
  if (r.skip) return t.skip(r.skip);
  const speech = audio('e-rec.wav');
  const { events } = await listen('e-rec + 2 s zeros + e-rec', Signals.concat(speech, Signals.zeros(2), speech));
  const ts = transcripts(events);
  assert.equal(ts.length, 2);
  for (const x of ts) assert.ok(wordRecall(r.text, x.text) >= 0.8, x.text);
  assert.ok(ts[1].startMs >= ts[0].endMs + 1500);
});

test('the language of every transcript: forced for a one-language model, detected within the list of the model, or forced by the caller', { skip }, async (t) => {
  const r = ref();
  if (r.skip) return t.skip(r.skip);
  const samples = audio('e-rec.wav');
  // Czech model: forced to cs, no detection pass, the text as before
  const forced = transcripts((await listen('e-rec, model cs', samples)).events);
  assert.equal(forced.length, 1);
  assert.equal(forced[0].language, 'cs');
  assert.ok(wordRecall(r.text, forced[0].text) >= 0.8);
  // the same model declared as cs + en: the language is detected within the two, and a Czech utterance is Czech
  const detected = transcripts((await listen('e-rec, languages cs+en', samples, undefined, { languages: ['cs', 'en'] })).events);
  assert.equal(detected.length, 1);
  assert.equal(detected[0].language, 'cs');
  assert.ok(wordRecall(r.text, detected[0].text) >= 0.8);
  // the caller forces another language of the list
  const english = transcripts((await listen('e-rec, language en', samples, undefined, { languages: ['cs', 'en'] }, { language: 'en' })).events);
  assert.equal(english.length, 1);
  assert.equal(english[0].language, 'en');
});

test('a language list: a call cancelled before it starts pays no detection, a malformed list fails, the engine then still decodes', { skip }, async (t) => {
  const r = ref();
  if (r.skip) return t.skip(r.skip);
  let cancelled = false;
  const logs = [];
  const engineUnderTest = await loadWasmEngine({
    moduleUrl: pathToFileURL(path.join(nodeWasm, 'czechspeech-st-node.js')).href,
    threads: 1,
    abortRequested: () => cancelled,
    log: (level, message) => logs.push(`${level}: ${message}`),
  });
  const fd = fs.openSync(TestEnv.model, 'r');
  const size = fs.fstatSync(fd).size;
  const reader = { size, read(position, length) {
    const buf = Buffer.allocUnsafe(length);
    const n = fs.readSync(fd, buf, 0, length, position);
    return new Uint8Array(buf.buffer, buf.byteOffset, n);
  } };
  const handle = engineUnderTest.whisperInit(reader);
  try {
    const samples = audio('e-rec.wav');
    const limits = { maxTokens: 200, temperatureFallback: false };
    cancelled = true;
    const t0 = performance.now();
    const aborted = engineUnderTest.whisperTranscribe(handle, 1, samples, 1, limits, 'cs,en');
    const abortedMs = Math.round(performance.now() - t0);
    cancelled = false;
    assert.equal(aborted.status, 1, 'cancelled');
    assert.ok(!aborted.language, 'no language for a cancelled call');
    assert.ok(abortedMs < 1000, `a call cancelled before it starts returns at once (${abortedMs} ms)`);
    for (const bad of ['cs,', ',cs', 'cs,,en', ',', '', 'xx', 'cs,xx', 'czech', 'czech,english', 'cs,english']) {
      assert.equal(engineUnderTest.whisperTranscribe(handle, 2, samples, 1, limits, bad).status, -1, `list ${JSON.stringify(bad)} is refused`);
    }
    const ok = engineUnderTest.whisperTranscribe(handle, 3, samples, 1, limits, 'cs,en');
    assert.equal(ok.status, 0);
    assert.equal(ok.language, 'cs');
    assert.ok(wordRecall(r.text, ok.text) >= 0.8, ok.text);
  } finally {
    engineUnderTest.whisperFree(handle);
    fs.closeSync(fd);
    engineUnderTest.wipe();
  }
});
