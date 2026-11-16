// Browser end-to-end tests in headless Chromium (playwright-core), served by scripts/serve.mjs:
//
//   A. cross-origin isolated page (COOP/COEP): the model is served locally and cached in Cache Storage;
//      transcribe() of e-rec; listen() through the real getUserMedia + AudioWorklet path with Chromium's fake
//      microphone playing f-rec-plus-silence15s.wav; the microphone tracks are stopped afterwards.
//   B. the same origin without COOP/COEP (?coi=0): the single-threaded fallback, transcribe() and listen().
//   C. the example page.
//   A to C run in Chromium's headless shell, which has navigator.gpu but no WebGPU adapter: config.backend 'auto'
//   falls back to the CPU modules there (checked), so these are the CPU path.
//   D. WebGPU, in a second browser started after the first one is closed: full Chromium in its new headless mode
//      (D3D12 on Windows). The worker's adapter is reported; without a hardware adapter D is skipped with a message.
//      The fixtures on the GPU (non-speech gives no transcript and no model load, e-rec word recall >= 0.9 and the
//      words compared with the CPU transcript of A), listen() with the fake microphone on 'auto', the GPU memory
//      cleared and released at the end, a simulated device loss in the middle of a recognition ('auto' then
//      gives the transcript from the CPU, within a bounded time, and the next recognition after the retry delay is on
//      WebGPU again), a recognition over the watchdog's limit (shortened) stopped with the device kept and the next one
//      transcribed on it, and two failure-and-recovery cycles that must leave no pthread worker of a CPU module behind
//      and no CPU module's memory reachable (WeakRefs and forced garbage collections in the recogniser worker).
//      The runaway decode of the meeting app's soak test is reproduced in test/e2e/runaway.mjs (opt-in).
//   Throughout: every network request of the pages is recorded; only the local server may be contacted.
//
// Env: CZECHSPEECH_MODEL, CZECHSPEECH_TEST_AUDIO, CZECHSPEECH_TEST_MANIFEST, CZECHSPEECH_TEST_THREADS (default 6),
//      CZECHSPEECH_E2E_SKIP_FALLBACK=1 to skip part B, CZECHSPEECH_E2E_SKIP_EXAMPLE=1 to skip C,
//      CZECHSPEECH_E2E_SKIP_GPU=1 to skip D. Chromium: `npx playwright-core install chromium`.
// GPU work runs one recognition at a time, with short pauses in between (CZECHSPEECH_E2E_GPU_PAUSE_MS, default 3000).
// Usage: node test/e2e/run.mjs
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { chromium } from 'playwright-core';
import { startServer } from '../../scripts/serve.mjs';
import { DEFAULT_MODEL } from '../../src/model.js';
import { TestEnv, missing, reference, wordDiff, wordRecall } from '../support/env.mjs';

const skip = missing('CZECHSPEECH_MODEL', TestEnv.model, 'CzechSpeech model') ||
  missing('CZECHSPEECH_TEST_AUDIO', TestEnv.audioDir, 'test audio directory') ||
  missing('CZECHSPEECH_TEST_MANIFEST', TestEnv.manifest, 'ParCzech manifest');
if (skip) {
  console.log(`SKIP: ${skip}`);
  process.exit(0);
}
const REF = reference(TestEnv.manifest, 'test_0000-0000000333');
const fakeMic = path.join(TestEnv.audioDir, 'f-rec-plus-silence15s.wav');
const modelSize = fs.statSync(TestEnv.model).size;
const threads = TestEnv.threads;
const gpuPauseMs = Number(process.env.CZECHSPEECH_E2E_GPU_PAUSE_MS || 3000);
const NON_SPEECH = ['a-silence-5s.wav', 'b-pink-noise-2s.wav', 'c-hum-2s.wav', 'd-clicks-6s.wav'];
const TEXT = /pojišťování|vývozu|Děkuji/;

let failures = 0;
const results = {};
const check = (ok, message) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${message}`);
  if (!ok) failures++;
};
// The language of every transcript event of the Czech model: forced 'cs', as before the catalog (no detection pass).
const allCs = (events) => events.filter((e) => e.type === 'transcript').every((e) => e.language === 'cs');
const mib = (b) => `${Math.round(b / 2 ** 20)} MiB`;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const server = await startServer({ port: 0, model: TestEnv.model, fixtures: TestEnv.audioDir, quiet: true });
const origin = `http://127.0.0.1:${server.port}`;
const modelUrl = `${origin}/models/${path.basename(TestEnv.model)}`;
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'czechspeech-e2e-'));
const requests = [];
const failedResponses = [];
const consoleLines = [];

/** headless shell (default) or channel 'chromium' (new headless, with a GPU); the profile (and its cache) is shared. */
async function launch(channel) {
  const ctx = await chromium.launchPersistentContext(profile, {
    headless: true,
    ...(channel ? { channel } : {}),
    args: [
      '--use-fake-ui-for-media-stream',
      '--use-fake-device-for-media-stream',
      `--use-file-for-fake-audio-capture=${fakeMic}%noloop`,
      '--autoplay-policy=no-user-gesture-required',
      // gc() in the workers, for D6's check that a released CPU module's memory can be collected
      '--js-flags=--expose-gc',
    ],
  });
  ctx.on('request', (r) => requests.push({ url: r.url(), method: r.method() }));
  ctx.on('response', (r) => { if (r.status() >= 400) failedResponses.push(`${r.status()} ${r.url()}`); });
  return ctx;
}

async function open(ctx, query = '') {
  const page = await ctx.newPage();
  page.on('console', (m) => {
    const where = m.type() === 'error' && m.location() && m.location().url ? ` (${m.location().url})` : '';
    consoleLines.push(`${m.type()}: ${m.text()}${where}`);
  });
  page.on('pageerror', (e) => consoleLines.push(`pageerror: ${e.message}`));
  await page.goto(`${origin}/test/e2e/page.html${query}`);
  await page.waitForFunction(() => window.e2e && window.e2e.ready);
  return page;
}

const timed = (s) => `${(s / 1000).toFixed(1)} s`;
const recogniserMs = (logs, what) => {
  // the last recognition that ran (an instance's log also has the gate-only lines of earlier calls)
  const re = /recogniser (\d+) ms \(encoder (\d+) ms, decoder ([\d.]+) ms\/token\)/;
  const l = [...logs].reverse().find((x) => x.includes(`${what}:`) && re.test(x));
  const m = l && re.exec(l);
  return m ? { recogniserMs: Number(m[1]), encoderMs: Number(m[2]), decoderMsPerToken: Number(m[3]) } : null;
};
const modelLoadMs = (logs) => {
  const l = logs.find((x) => x.includes('model loaded in'));
  const m = l && /model loaded in (\d+) ms/.exec(l);
  return m ? Number(m[1]) : null;
};
const modelArgs = { url: modelUrl, sha256: DEFAULT_MODEL.sha256, size: modelSize };

let context = null;
try {
  context = await launch(null);
  // ---- A: cross-origin isolated ----
  const page = await open(context);
  const info = await page.evaluate(() => window.e2e.info());
  results.browser = info.userAgent;
  check(info.crossOriginIsolated === true && info.sharedArrayBuffer, `A: page is cross-origin isolated (${info.hardwareConcurrency} logical cores)`);
  const shellGpu = await page.evaluate(() => window.e2e.gpuInfo());
  results.headlessShellGpu = shellGpu;
  console.log(`      headless shell, WebGPU in a worker: ${shellGpu.probe && shellGpu.probe.ok ? 'available' : shellGpu.probe ? shellGpu.probe.reason : shellGpu.error}`);

  const dl = await page.evaluate((a) => window.e2e.model(a), modelArgs);
  results.firstModelFetch = dl;
  check(dl.size === DEFAULT_MODEL.size && dl.sha256 === DEFAULT_MODEL.sha256 && dl.cached && dl.isDownloaded,
    `A: model from the local server, SHA-256 verified, stored in Cache Storage in ${timed(dl.ms)}`);
  const again = await page.evaluate((a) => window.e2e.model(a), modelArgs);
  const modelRequests = requests.filter((r) => r.url.startsWith(modelUrl)).length;
  check(again.cached && again.ms < 3000 && modelRequests === 1,
    `A: second fromUrl() served from the cache in ${again.ms} ms without a request (${modelRequests} request in total)`);

  const created = await page.evaluate((t) => window.e2e.create({ config: { threads: t } }), threads);
  check(created.state.runtime === 'wasm-threads' && created.state.threads === threads && created.state.backend === null,
    `A: CzechSpeech.create() in ${created.ms} ms: ${created.state.runtime}, ${created.state.threads} threads, backend decided later`);

  const tr = await page.evaluate((u) => window.e2e.transcribeUrl(u), `${origin}/fixtures/e-rec.wav`);
  const rec = wordRecall(REF, tr.text);
  const timing = recogniserMs(tr.logs, 'transcribe');
  results.isolatedTranscribe = { text: tr.text, ms: tr.ms, audioMs: tr.audioMs, threads, recall: rec, modelLoadMs: modelLoadMs(tr.logs),
    ...timing, peakHeapBytes: tr.state.peakHeapBytes, backend: tr.state.backend, backendFallback: tr.state.backendFallback };
  console.log(`      transcribe(e-rec, ${tr.audioMs} ms): ${tr.ms} ms total, model load ${modelLoadMs(tr.logs)} ms, ` +
    `${JSON.stringify(timing)}, peak wasm heap ${mib(tr.state.peakHeapBytes)}: "${tr.text}"`);
  check(rec >= 0.8, `A: transcribe() word recall ${rec.toFixed(2)}`);
  check(tr.unchanged && !tr.state.modelLoaded, 'A: caller\'s samples unchanged, model freed afterwards');
  check(!tr.logs.some((l) => TEXT.test(l)), 'A: no transcript text in the SDK logs');
  if (!(shellGpu.probe && shellGpu.probe.ok)) {
    check(tr.state.backend === 'cpu' && !!tr.state.backendFallback && tr.logs.filter((l) => /WebGPU not used/.test(l)).length === 1,
      `A: backend 'auto' without a WebGPU adapter: the CPU, one notice (${tr.state.backendFallback})`);
  }

  const ls = await page.evaluate(() => window.e2e.listen({ transcripts: 1, maxMs: 150000 }));
  const t = ls.events.filter((e) => e.type === 'transcript');
  results.isolatedListen = { events: ls.events, exitMs: ls.exitMs, tracks: ls.tracks, error: ls.error,
    timing: recogniserMs(ls.logs, 'utterance'), modelLoadMs: modelLoadMs(ls.logs) };
  for (const e of ls.events) console.log(`      +${e.at} ms ${e.type}${e.atMs !== undefined ? ` at ${e.atMs} ms` : ''}${e.text ? ` ${e.startMs}-${e.endMs} ms: "${e.text}"` : ''}`);
  for (const l of ls.logs.filter((x) => /listen:|capture:|recogniser|model loaded/.test(x))) console.log(`        ${l}`);
  check(!ls.error, `A: listen() ran without a fatal error${ls.error ? `: ${JSON.stringify(ls.error)}` : ''}`);
  check(ls.events.map((e) => e.type).join(',') === 'speechStarted,speechEnded,transcript',
    `A: listen() events: ${ls.events.map((e) => e.type).join(', ')}`);
  check(t.length === 1 && wordRecall(REF, t[0].text) >= 0.8, `A: listen() transcript word recall ${t[0] ? wordRecall(REF, t[0].text).toFixed(2) : '-'}`);
  check(t.length === 1 && !t[0].str.includes('pojišťování'), 'A: String(transcript) does not contain the text');
  check(t.length === 1 && allCs(ls.events) && t[0].str.includes('cs'), `A: listen() transcript language is 'cs' (${t.map((e) => e.language).join(', ')})`);
  check(ls.tracks.length > 0 && ls.tracks.every((s) => s === 'ended'), `A: microphone tracks stopped after break (${ls.tracks.join(', ')})`);
  check(!ls.state.modelLoaded && !ls.state.listening, 'A: model freed and session over after break');
  check(JSON.stringify(ls.constraints[0]).includes('"noiseSuppression":false'), `A: getUserMedia constraints ${JSON.stringify(ls.constraints[0])}`);

  // A, stop(): through the real worklet, the utterance in progress is not lost and the loop ends by itself.
  // (1) 6.9 s after speechStarted the sentence is over, but the endpointer would end the utterance only after 600 ms
  //     of silence: stop() ends it, it is transcribed, and the loop ends without break (the fake microphone would go
  //     on with 15 s of silence).
  const s1 = await page.evaluate(() => window.e2e.listen({ stopAfterStartMs: 6900, maxMs: 150000 }));
  const t1 = s1.events.filter((e) => e.type === 'transcript');
  results.stopAtEnd = { events: s1.events, stoppedAt: s1.stoppedAt, exitMs: s1.exitMs, error: s1.error };
  for (const e of s1.events) console.log(`      +${e.at} ms ${e.type}${e.atMs !== undefined ? ` at ${e.atMs} ms` : ''}${e.text ? ` ${e.startMs}-${e.endMs} ms: "${e.text}"` : ''}`);
  console.log(`      stop() at +${s1.stoppedAt} ms, loop ended at +${s1.exitMs} ms`);
  check(!s1.error && s1.endedByItself && s1.events.map((e) => e.type).join(',') === 'speechStarted,speechEnded,transcript',
    `A: stop() at the end of the sentence: ${s1.events.map((e) => e.type).join(', ')}, the loop ended by itself${s1.error ? ` (error ${JSON.stringify(s1.error)})` : ''}`);
  check(t1.length === 1 && wordRecall(REF, t1[0].text) >= 0.8, `A: stop(): the last utterance was transcribed, word recall ${t1[0] ? wordRecall(REF, t1[0].text).toFixed(2) : '-'}`);
  check(t1.length === 1 && allCs(s1.events), `A: stop(): transcript language is 'cs' (${t1.map((e) => e.language).join(', ')})`);
  check(s1.tracks.length > 0 && s1.tracks.every((s) => s === 'ended') && !s1.state.modelLoaded && !s1.state.listening,
    'A: stop(): microphone stopped, model freed, session over');
  // (2) 3.5 s after speechStarted, in the middle of the sentence: what was said up to then is transcribed, with the
  //     word in progress (stop() waits for a pause, at most 1 s).
  const s2 = await page.evaluate(() => window.e2e.listen({ stopAfterStartMs: 3500, maxMs: 150000 }));
  const t2 = s2.events.filter((e) => e.type === 'transcript');
  const precision2 = t2[0] ? wordRecall(t2[0].text, REF) : 0; // share of the transcript's words that are in the reference
  results.stopMidSentence = { events: s2.events, stoppedAt: s2.stoppedAt, exitMs: s2.exitMs, error: s2.error, precision: precision2 };
  for (const e of s2.events) console.log(`      +${e.at} ms ${e.type}${e.atMs !== undefined ? ` at ${e.atMs} ms` : ''}${e.text ? ` ${e.startMs}-${e.endMs} ms: "${e.text}"` : ''}`);
  check(!s2.error && s2.endedByItself && t2.length === 1 && t2[0].endMs < 6000 && precision2 >= 0.8,
    `A: stop() in the middle of the sentence: the part said so far is transcribed (${t2[0] ? `${t2[0].endMs} ms of speech, ${precision2.toFixed(2)} of its words in the reference` : 'no transcript'}), the loop ended by itself`);
  check(t2.length === 1 && allCs(s2.events), `A: stop() mid-sentence: transcript language is 'cs' (${t2.map((e) => e.language).join(', ')})`);
  check(!s2.state.modelLoaded && !s2.state.listening && s2.tracks.every((s) => s === 'ended'), 'A: stop() mid-sentence: microphone stopped, model freed');
  results.memory = await page.evaluate(() => window.e2e.memory());
  await page.evaluate(() => window.e2e.close());
  await page.close();

  // ---- B: not cross-origin isolated: single-threaded fallback ----
  if (process.env.CZECHSPEECH_E2E_SKIP_FALLBACK !== '1') {
    const pb = await open(context, '?coi=0');
    const ib = await pb.evaluate(() => window.e2e.info());
    check(ib.crossOriginIsolated === false && !ib.sharedArrayBuffer, 'B: page is not cross-origin isolated, no SharedArrayBuffer');
    const db = await pb.evaluate((a) => window.e2e.model(a), modelArgs);
    check(db.cached && db.ms < 3000, `B: model from the same origin's cache in ${db.ms} ms`);
    // backend 'cpu': this part is about the single-threaded CPU module (WebGPU would not need isolation)
    const cb = await pb.evaluate(() => window.e2e.create({ config: { backend: 'cpu' } }));
    check(cb.state.runtime === 'wasm-single' && cb.state.threads === 1, `B: runtime ${cb.state.runtime}, ${cb.state.threads} thread`);
    const tb = await pb.evaluate((u) => window.e2e.transcribeUrl(u), `${origin}/fixtures/e-rec.wav`);
    const rb = wordRecall(REF, tb.text);
    const timingB = recogniserMs(tb.logs, 'transcribe');
    results.fallbackTranscribe = { text: tb.text, ms: tb.ms, recall: rb, modelLoadMs: modelLoadMs(tb.logs), ...timingB,
      peakHeapBytes: tb.state.peakHeapBytes };
    console.log(`      transcribe(e-rec) single-threaded: ${tb.ms} ms, ${JSON.stringify(timingB)}, peak wasm heap ${mib(tb.state.peakHeapBytes)}: "${tb.text}"`);
    check(rb >= 0.8 && !tb.state.modelLoaded && tb.state.backend === 'cpu', `B: transcribe() word recall ${rb.toFixed(2)}, model freed`);
    const lb = await pb.evaluate(() => window.e2e.listen({ transcripts: 1, maxMs: 240000 }));
    const tlb = lb.events.filter((e) => e.type === 'transcript');
    results.fallbackListen = { events: lb.events, exitMs: lb.exitMs, error: lb.error, timing: recogniserMs(lb.logs, 'utterance') };
    for (const e of lb.events) console.log(`      +${e.at} ms ${e.type}${e.text ? `: "${e.text}"` : ''}`);
    check(!lb.error && tlb.length === 1 && wordRecall(REF, tlb[0].text) >= 0.8,
      `B: listen() single-threaded: ${lb.events.map((e) => e.type).join(', ')}${lb.error ? ` error ${JSON.stringify(lb.error)}` : ''}`);
    check(tlb.length === 1 && allCs(lb.events), `B: transcript language is 'cs' (${tlb.map((e) => e.language).join(', ')})`);
    check(lb.tracks.every((s) => s === 'ended') && !lb.state.modelLoaded, 'B: microphone stopped and model freed');
    await pb.evaluate(() => window.e2e.close());
    await pb.close();
  }

  // ---- C: the example page (model from the cache filled in A, Start, a transcript, Stop) ----
  if (process.env.CZECHSPEECH_E2E_SKIP_EXAMPLE !== '1') {
    const pc = await context.newPage();
    pc.on('console', (m) => consoleLines.push(`${m.type()}: ${m.text()}`));
    pc.on('pageerror', (e) => consoleLines.push(`pageerror: ${e.message}`));
    await pc.goto(`${origin}/examples/basic/?model=/models/${path.basename(TestEnv.model)}`);
    await pc.waitForSelector('#start:not([disabled])', { timeout: 60000 });
    const status = await pc.textContent('#model-status');
    check(/ready/.test(status) && /stored in this browser/.test(status), `C: example page found the stored model (${status})`);
    await pc.click('#start');
    await pc.waitForSelector('#transcripts li', { timeout: 150000 });
    const line = await pc.textContent('#transcripts li');
    await pc.click('#stop');
    await pc.waitForSelector('#start:not([disabled])', { timeout: 10000 });
    const indicator = await pc.textContent('#indicator');
    check(wordRecall(REF, line) >= 0.8 && indicator === '', `C: example page Start -> transcript -> Stop ("${line.trim()}")`);
    await pc.close();
  }
  await context.close();
  context = null;

  // ---- D: WebGPU (full Chromium, new headless mode), after the first browser is closed ----
  if (process.env.CZECHSPEECH_E2E_SKIP_GPU !== '1') {
    context = await launch('chromium');
    const pd = await open(context);
    const gi = await pd.evaluate(() => window.e2e.gpuInfo());
    results.gpu = { adapter: gi.info, features: gi.features, limits: gi.limits, probe: gi.probe, userAgent: (await pd.evaluate(() => window.e2e.info())).userAgent };
    console.log(`      WebGPU in a worker: ${JSON.stringify(gi.info)} ${gi.probe ? JSON.stringify(gi.probe) : gi.error}`);
    if (!gi.probe || !gi.probe.ok) {
      console.log(`SKIP  D: no usable WebGPU adapter in a worker of this browser (${gi.probe ? gi.probe.reason : gi.error}); the WebGPU cases did not run`);
      results.gpu.skipped = true;
    } else {
      const software = gi.info.isFallbackAdapter || /swiftshader|llvmpipe|software|microsoft basic/i.test(`${gi.info.vendor} ${gi.info.architecture} ${gi.info.description}`);
      check(!software, `D: the worker sees a hardware adapter (${gi.info.vendor} ${gi.info.architecture}), not a software one`);
      const dd = await pd.evaluate((a) => window.e2e.model(a), modelArgs);
      check(dd.cached, `D: model from Cache Storage in ${dd.ms} ms`);

      // D1: non-speech on the WebGPU backend: no transcript, the model is not even loaded
      await pd.evaluate((t) => window.e2e.create({ config: { threads: t, backend: 'webgpu' } }), threads);
      const quiet = [];
      for (const name of NON_SPEECH) quiet.push([name, await pd.evaluate((u) => window.e2e.transcribeUrl(u), `${origin}/fixtures/${name}`)]);
      quiet.push(['digital zeros 10 s', await pd.evaluate(() => window.e2e.transcribeZeros(10))]);
      for (const [name, r] of quiet) {
        check(r.text === '' && r.state.modelLoads === 0 && r.state.backend === 'webgpu',
          `D: ${name} on WebGPU: no transcript, model not loaded (backend ${r.state.backend})`);
      }

      // D2: e-rec on the WebGPU backend, compared with the CPU transcript of A
      await sleep(gpuPauseMs);
      const tg = await pd.evaluate((u) => window.e2e.transcribeUrl(u), `${origin}/fixtures/e-rec.wav`);
      const rg = wordRecall(REF, tg.text);
      const tgTiming = recogniserMs(tg.logs, 'transcribe');
      const diff = wordDiff(results.isolatedTranscribe.text, tg.text);
      results.gpuTranscribe = { text: tg.text, ms: tg.ms, recall: rg, modelLoadMs: modelLoadMs(tg.logs), ...tgTiming,
        peakHeapBytes: tg.state.peakHeapBytes, peakGpuBytes: tg.state.peakGpuBytes, adapter: tg.state.gpuAdapter, vsCpu: diff };
      console.log(`      transcribe(e-rec) on WebGPU: ${tg.ms} ms, model load ${modelLoadMs(tg.logs)} ms, ${JSON.stringify(tgTiming)}, ` +
        `peak wasm heap ${mib(tg.state.peakHeapBytes)}, GPU buffers ${mib(tg.state.peakGpuBytes)}: "${tg.text}"`);
      check(rg >= 0.9, `D: e-rec on WebGPU, word recall ${rg.toFixed(2)}`);
      check(tg.state.backend === 'webgpu' && tg.state.peakGpuBytes > 400 * 2 ** 20 && !tg.state.modelLoaded,
        `D: the model was on the GPU (${mib(tg.state.peakGpuBytes)} of GPU buffers) and is freed`);
      check(diff.same, `D: WebGPU transcript ${diff.same ? 'identical to' : `differs from (${diff.summary})`} the CPU transcript of A`);
      // the last such line is this call's (the no-speech calls before it started and released a GPU module without a model)
      const clearedLine = [...tg.logs].reverse().find((l) => /GPU buffers cleared/.test(l)) || '-';
      check(/GPU buffers cleared \([1-9]\d* MiB\) and released, heap zeroed/.test(clearedLine),
        `D: GPU buffers cleared and released at the end (${clearedLine.replace(/^info: /, '')})`);
      check(tg.unchanged && !tg.logs.some((l) => TEXT.test(l)), 'D: caller\'s samples unchanged, no transcript text in the SDK logs');

      // D3: listen() with the fake microphone on 'auto' (WebGPU here)
      await sleep(gpuPauseMs);
      await pd.evaluate((t) => window.e2e.create({ config: { threads: t } }), threads);
      const lg = await pd.evaluate(() => window.e2e.listen({ transcripts: 1, maxMs: 150000 }));
      const tlg = lg.events.filter((e) => e.type === 'transcript');
      results.gpuListen = { events: lg.events, exitMs: lg.exitMs, error: lg.error, timing: recogniserMs(lg.logs, 'utterance'),
        modelLoadMs: modelLoadMs(lg.logs), backend: lg.state.backend };
      for (const e of lg.events) console.log(`      +${e.at} ms ${e.type}${e.atMs !== undefined ? ` at ${e.atMs} ms` : ''}${e.text ? ` ${e.startMs}-${e.endMs} ms: "${e.text}"` : ''}`);
      for (const l of lg.logs.filter((x) => /listen:|capture:|recogniser|model loaded|WebGPU/.test(x))) console.log(`        ${l}`);
      check(!lg.error && lg.events.map((e) => e.type).join(',') === 'speechStarted,speechEnded,transcript' && lg.state.backend === 'webgpu',
        `D: listen() on 'auto' used WebGPU: ${lg.events.map((e) => e.type).join(', ')}${lg.error ? ` error ${JSON.stringify(lg.error)}` : ''}`);
      check(tlg.length === 1 && wordRecall(REF, tlg[0].text) >= 0.9, `D: listen() transcript word recall ${tlg[0] ? wordRecall(REF, tlg[0].text).toFixed(2) : '-'}`);
      check(tlg.length === 1 && allCs(lg.events), `D: listen() transcript language is 'cs' (${tlg.map((e) => e.language).join(', ')})`);
      check(lg.tracks.length > 0 && lg.tracks.every((s) => s === 'ended') && !lg.state.modelLoaded && !lg.state.listening,
        'D: microphone stopped, model freed, session over');

      // D4: the device is lost in the middle of a recognition (test hook): 'auto' redoes it on the CPU, then, after the
      // retry delay (shortened to 2 s here; 10 s by default), starts WebGPU again for the next recognition
      await sleep(gpuPauseMs);
      await pd.evaluate((t) => window.e2e.create({ config: { threads: t }, keepModelLoaded: true,
        _engineOptions: { testFailRecognitionMs: 400 }, _gpuTimeouts: { gpuRetryMs: [2000, 2000, 2000] } }), threads);
      const tf = await pd.evaluate((u) => window.e2e.transcribeUrl(u), `${origin}/fixtures/e-rec.wav`);
      const rf = wordRecall(REF, tf.text);
      results.gpuFailure = { text: tf.text, ms: tf.ms, recall: rf, backend: tf.state.backend, backendFallback: tf.state.backendFallback,
        logs: tf.logs.filter((l) => /WebGPU|model loaded|recogniser|released/.test(l)) };
      for (const l of results.gpuFailure.logs) console.log(`        ${l}`);
      check(rf >= 0.9 && tf.state.backend === 'cpu' && /simulated WebGPU device loss/.test(tf.state.backendFallback || '') && tf.ms < 90000,
        `D: device lost mid-recognition: transcript from the CPU in ${timed(tf.ms)}, word recall ${rf.toFixed(2)}, backend now ${tf.state.backend}`);
      check(tf.logs.some((l) => /WebGPU module released after a failure/.test(l)), 'D: the failed WebGPU module was released');
      await sleep(Math.max(gpuPauseMs, 2500));
      const tb = await pd.evaluate((u) => window.e2e.transcribeUrl(u), `${origin}/fixtures/e-rec.wav`);
      const rb = wordRecall(REF, tb.text);
      const back = tb.logs.slice(tf.logs.length);
      results.gpuRecovery = { ms: tb.ms, recall: rb, backend: tb.state.backend, backendFallback: tb.state.backendFallback,
        logs: back.filter((l) => /WebGPU|model loaded|recogniser|released/.test(l)) };
      for (const l of results.gpuRecovery.logs) console.log(`        ${l}`);
      check(rb >= 0.9 && tb.state.backend === 'webgpu' && tb.state.backendFallback === null && back.some((l) => /WebGPU started again after a failure/.test(l)),
        `D: after the retry delay the next recognition is on WebGPU again (${timed(tb.ms)}, word recall ${rb.toFixed(2)}, backend ${tb.state.backend})`);
      await pd.evaluate(() => window.e2e.close());

      // D5: a recognition over its time limit (1 ms here, for the first recognition only: the stop lands at whisper.cpp's
      // first abort check, after the encoder pass) is stopped; the WebGPU module and its device stay: no transcript for
      // it, no failure, and the next recognition on the same module gives the transcript; the model is loaded once
      await sleep(gpuPauseMs);
      await pd.evaluate((t) => window.e2e.create({ config: { threads: t }, keepModelLoaded: true,
        _gpuTimeouts: { recognizeLimitFirstMs: [1] } }), threads);
      const ts1 = await pd.evaluate((u) => window.e2e.transcribeUrl(u), `${origin}/fixtures/e-rec.wav`);
      const ts2 = await pd.evaluate((u) => window.e2e.transcribeUrl(u), `${origin}/fixtures/e-rec.wav`);
      const rs2 = wordRecall(REF, ts2.text);
      results.gpuStopped = { ms: [ts1.ms, ts2.ms], recall: rs2, state: ts2.state,
        logs: ts2.logs.filter((l) => /WebGPU|stopped|released|model loaded/.test(l)) };
      for (const l of results.gpuStopped.logs) console.log(`        ${l}`);
      check(ts1.text === '' && ts1.state.abortedRecognitions === 1 && ts1.ms < 10000,
        `D: a recognition over its time limit stopped in ${timed(ts1.ms)}, no transcript (aborted ${ts1.state.abortedRecognitions})`);
      check(rs2 >= 0.9 && ts2.state.abortedRecognitions === 1,
        `D: the next recognition on the same WebGPU module gives the transcript (${timed(ts2.ms)}, word recall ${rs2.toFixed(2)})`);
      check(ts2.state.backend === 'webgpu' && ts2.state.backendFallback === null && ts2.state.modelLoads === 1 &&
        !ts2.logs.some((l) => /WebGPU failed|released after a failure/.test(l)),
        `D: the WebGPU module and its device were kept (backend ${ts2.state.backend}, ${ts2.state.modelLoads} model load)`);
      await pd.evaluate(() => window.e2e.close());

      // D6: two WebGPU failures in a row, each followed by WebGPU again (test hook twice, retry delays 0): each failure
      // starts the threaded CPU module (threads + 1 pthread workers holding its shared memory, about 1.2 GB with the
      // model); going back to WebGPU must terminate them and leave the module's memory unreachable, so that no CPU module
      // of an earlier cycle is left behind. The memory is observed in the recogniser worker (count-workers.js): a WeakRef
      // to every shared WebAssembly.Memory, read after forced garbage collections. After the second cycle the current CPU
      // module must be reachable (the check sees a module in use) and the first one not; on WebGPU at the end, neither.
      await sleep(gpuPauseMs);
      await pd.evaluate((t) => window.e2e.createCounting({ config: { threads: t }, keepModelLoaded: true,
        _engineOptions: { testFailRecognitionMs: 400, testFailRecognitionCount: 2 }, _gpuTimeouts: { gpuRetryMs: [0, 0, 0] } }), threads);
      const cycles = [];
      for (let i = 0; i < 3; i++) {
        const r = await pd.evaluate((u) => window.e2e.transcribeUrl(u), `${origin}/fixtures/e-rec.wav`);
        cycles.push({ ms: r.ms, recall: wordRecall(REF, r.text), backend: r.state.backend,
          workers: (await pd.evaluate(() => window.e2e.workerCounts())).recognizer || { created: 0, terminated: 0 },
          memories: await pd.evaluate(() => window.e2e.memories()) });
      }
      results.gpuCycles = cycles;
      const memoryLine = (m) => (m.error ? m.error : `${m.created} CPU module memories created, ${m.live} reachable (${mib(m.liveBytes)})`);
      for (const c of cycles) console.log(`        ${c.backend}: ${timed(c.ms)}, word recall ${c.recall.toFixed(2)}, recogniser workers created ${c.workers.created}, terminated ${c.workers.terminated}; ${memoryLine(c.memories)}`);
      const last = cycles[2].workers;
      check(cycles.every((c) => c.recall >= 0.9) && cycles.map((c) => c.backend).join(',') === 'cpu,cpu,webgpu',
        `D: two WebGPU failures, each redone on the CPU, then WebGPU again (${cycles.map((c) => c.backend).join(', ')})`);
      check(last.created >= 2 * (threads + 1) && last.created === last.terminated,
        `D: no pthread worker of the two CPU modules is left (${last.created} created, ${last.terminated} terminated)`);
      const [m1, m2, m3] = cycles.map((c) => c.memories);
      check(!m1.error && m1.created === 1 && m1.live === 1 && m2.created === 2 && m2.live === 1,
        `D: the first CPU module's memory is collected once WebGPU replaced it; the module in use is reachable (after cycle 1: ${memoryLine(m1)}; after cycle 2: ${memoryLine(m2)})`);
      check(!m3.error && m3.created === 2 && m3.live === 0,
        `D: back on WebGPU, no CPU module's memory is reachable (${memoryLine(m3)})`);
      await pd.evaluate(() => window.e2e.close());
    }
    await pd.close();
    await context.close();
    context = null;
  }

  // ---- network and console audit ----
  const foreign = requests.filter((r) => !r.url.startsWith(`${origin}/`));
  check(foreign.length === 0, `no request left the local server (${requests.length} requests${foreign.length ? `; foreign: ${foreign.map((r) => r.url).join(', ')}` : ''})`);
  const allowed = ['/test/e2e/', '/src/', '/models/', '/fixtures/', '/examples/basic/'];
  const odd = requests.filter((r) => r.method !== 'GET' || !allowed.some((p) => new URL(r.url).pathname.startsWith(p)));
  check(odd.length === 0, `only GET requests for the harness, the package, the model and the fixtures${odd.length ? `: ${JSON.stringify(odd)}` : ''}`);
  check(failedResponses.length === 0, `no failed requests${failedResponses.length ? `: ${failedResponses.join(', ')}` : ''}`);
  check(!consoleLines.some((l) => TEXT.test(l)), `no transcript text on the console (${consoleLines.length} lines)`);
  const errors = consoleLines.filter((l) => l.startsWith('pageerror') || l.startsWith('error'));
  check(errors.length === 0, `no page errors${errors.length ? `: ${errors.join(' | ')}` : ''}`);
} catch (e) {
  failures++;
  console.log(`FAIL  ${e && e.stack ? e.stack : e}`);
} finally {
  if (context) await context.close();
  await server.close();
  fs.rmSync(profile, { recursive: true, force: true });
}

const out = path.join(TestEnv.audioDir ? os.tmpdir() : '.', 'czechspeech-e2e-results.json');
fs.writeFileSync(out, JSON.stringify(results, null, 2));
console.log(`results: ${out}`);
console.log(failures ? `${failures} FAILED` : 'ALL PASSED');
process.exit(failures ? 1 : 0);
