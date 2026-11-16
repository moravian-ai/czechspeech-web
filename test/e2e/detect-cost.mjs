// What language detection costs, against a forced language, and whether it leaks GPU memory (opt-in, not part of
// npm run test:e2e).
//
// Gate: CZECHSPEECH_E2E_DETECT_COST=1 (without it the script prints SKIP and exits 0). Run it on an otherwise idle
// machine; on the shared notebook only through heavy.sh.
//
// The same recordings go through the pinned Czech model in two configurations:
//   forced  languages ['cs']: one language, whisper.cpp is told to use it, no detection pass (what Czech does today);
//   detect  languages ['cs', 'en'], no language given: a fresh whisper state, the mel spectrogram, a full 30 s encoder
//           pass and one decoder step choose between cs and en, then the transcript is decoded as with forced.
// in the order forced, detect, forced (the second forced run shows whether the machine drifted), each with
// CZECHSPEECH_DETECT_COST_REPS (default 5) repetitions of every utterance. The utterances are the first 2, 3 and 4 s of
// the e-rec fixture, all of it (6.9 s) and e-rec twice in a row (13.9 s). The model stays loaded (keepModelLoaded), so the
// first call of a run is the cold one (module start-up, first use) and the others are steady state.
//   A. the CPU path: WebAssembly (threads, CZECHSPEECH_TEST_THREADS, default 6) in Chromium's headless shell;
//   B. the WebGPU path: full Chromium in its new headless mode, as part D of test/e2e/run.mjs (skipped, with a message,
//      without a hardware adapter);
//   C. a WebGPU smoke test: CZECHSPEECH_DETECT_COST_SMOKE (default 24) consecutive recognitions with the 'detect'
//      model, the live GPU buffers read after each one (test/e2e/gpu-bytes.js counts them in the recogniser worker; the
//      SDK itself reports gpuBytes only when the model loads) and when the device is destroyed at close().
// Report: per utterance the median recogniser time (the SDK's log line, which has neither the model load nor the speech
// gate) of forced and detect and their ratio, the first call against the steady state, and the series of GPU bytes.
// No transcript text is printed (only a comparison of the two settings); the raw numbers go to a JSON file.
//
// Env: CZECHSPEECH_E2E_DETECT_COST=1, CZECHSPEECH_MODEL, CZECHSPEECH_TEST_AUDIO (the e-rec.wav fixture),
//      CZECHSPEECH_TEST_THREADS, CZECHSPEECH_DETECT_COST_REPS, CZECHSPEECH_DETECT_COST_SMOKE,
//      CZECHSPEECH_DETECT_COST_SKIP_CPU=1, CZECHSPEECH_DETECT_COST_SKIP_GPU=1.
// Usage: CZECHSPEECH_E2E_DETECT_COST=1 node test/e2e/detect-cost.mjs
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { chromium } from 'playwright-core';
import { startServer } from '../../scripts/serve.mjs';
import { DEFAULT_MODEL } from '../../src/model.js';
import { TestEnv, missing } from '../support/env.mjs';

if (process.env.CZECHSPEECH_E2E_DETECT_COST !== '1') {
  console.log('SKIP: set CZECHSPEECH_E2E_DETECT_COST=1 to run the language detection cost measurement (opt-in)');
  process.exit(0);
}
const skip = missing('CZECHSPEECH_MODEL', TestEnv.model, 'CzechSpeech model') ||
  missing('CZECHSPEECH_TEST_AUDIO', TestEnv.audioDir, 'test audio directory');
if (skip) {
  console.log(`SKIP: ${skip}`);
  process.exit(0);
}
const REPS = Math.max(1, Number(process.env.CZECHSPEECH_DETECT_COST_REPS || 5));
const SMOKE = Math.max(2, Number(process.env.CZECHSPEECH_DETECT_COST_SMOKE || 24));
const PLAN_SECONDS = [2, 3, 4, 6.9, 13.9];
const pauseMs = Number(process.env.CZECHSPEECH_E2E_GPU_PAUSE_MS || 3000);
const threads = TestEnv.threads;

let failures = 0;
const results = { machine: { cpus: os.cpus()[0] && os.cpus()[0].model, logicalCores: os.cpus().length, memoryGiB: Math.round(os.totalmem() / 2 ** 30), threads }, reps: REPS };
const check = (ok, message) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${message}`);
  if (!ok) failures++;
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length === 0 ? NaN : s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
};
const f1 = (x) => (Number.isFinite(x) ? x.toFixed(0) : '-');

const server = await startServer({ port: 0, model: TestEnv.model, fixtures: TestEnv.audioDir, quiet: true });
const origin = `http://127.0.0.1:${server.port}`;
const modelUrl = `${origin}/models/${path.basename(TestEnv.model)}`;
const wavUrl = `${origin}/fixtures/e-rec.wav`;
const modelSize = fs.statSync(TestEnv.model).size;
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'czechspeech-detect-cost-'));
const pageErrors = [];

async function launch(channel) {
  return chromium.launchPersistentContext(profile, { headless: true, ...(channel ? { channel } : {}) });
}
async function open(ctx) {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => pageErrors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error') pageErrors.push(m.text()); });
  await page.goto(`${origin}/test/e2e/page.html`);
  await page.waitForFunction(() => window.e2e && window.e2e.ready);
  return page;
}
const MODELS = { forced: ['cs'], detect: ['cs', 'en'] };
const plan = PLAN_SECONDS.flatMap((s) => Array(REPS).fill(s));

/** One run: a model with the languages of `setting`, a recogniser of `backend`, `plan` through transcribe(). */
async function run(page, setting, backend, { tracked = false, calls = plan } = {}) {
  await page.evaluate((a) => window.e2e.model(a), { url: modelUrl, sha256: DEFAULT_MODEL.sha256, size: modelSize, languages: MODELS[setting] });
  const options = { config: { threads, backend }, keepModelLoaded: true };
  await page.evaluate(([t, o]) => (t ? window.e2e.createGpuTracked(o) : window.e2e.create(o)), [tracked, options]);
  const r = await page.evaluate((a) => window.e2e.run(a), { plan: calls, wavUrl, trackGpu: tracked });
  return r;
}

/** Medians of one run: per utterance length (the first call of the run left out of its own length), and the first call. */
function summarise(r) {
  const calls = r.calls;
  const first = calls[0];
  const per = {};
  for (const s of PLAN_SECONDS) {
    const own = calls.filter((c) => c.seconds === s && c !== first);
    per[s] = { n: own.length, recogniserMs: median(own.map((c) => c.recogniserMs)), totalMs: median(own.map((c) => c.ms)),
      encoderMs: median(own.map((c) => c.encoderMs)), words: median(own.map((c) => c.words)) };
  }
  return { first: { seconds: first.seconds, recogniserMs: first.recogniserMs, totalMs: first.ms, encoderMs: first.encoderMs }, per };
}

async function measure(label, page, backend) {
  const runs = {};
  const order = ['forced', 'detect', 'forced2'];
  for (const name of order) {
    const setting = name === 'detect' ? 'detect' : 'forced';
    if (backend === 'webgpu') await sleep(pauseMs);
    const r = await run(page, setting, backend);
    await page.evaluate(() => window.e2e.close());
    runs[name] = { ...r, summary: summarise(r) };
    console.log(`      ${label} ${name}: ${r.calls.length} recognitions, first call ${runs[name].summary.first.recogniserMs} ms (${runs[name].summary.first.seconds} s of audio)`);
  }
  // the same transcripts in the two settings (the language list changes the decoding of Czech speech not at all)
  const same = runs.forced.calls.every((c, i) => c.text === runs.detect.calls[i].text);
  check(runs.forced.calls.every((c) => c.words > 0) && runs.detect.calls.every((c) => c.words > 0),
    `${label}: every recognition gave a transcript (forced and detect)`);
  check(same, `${label}: the transcripts of 'detect' are word for word those of 'forced'`);
  const rows = [];
  console.log(`      ${label}: median recogniser ms per utterance (reps ${REPS}; the run's first call left out of its length)`);
  console.log('        audio s | forced | detect | detect/forced | forced again | encoder forced | encoder detect');
  for (const s of PLAN_SECONDS) {
    const a = runs.forced.summary.per[s];
    const b = runs.detect.summary.per[s];
    const c = runs.forced2.summary.per[s];
    const row = { seconds: s, forcedMs: a.recogniserMs, detectMs: b.recogniserMs, ratio: b.recogniserMs / a.recogniserMs,
      forcedAgainMs: c.recogniserMs, forcedEncoderMs: a.encoderMs, detectEncoderMs: b.encoderMs,
      forcedTotalMs: a.totalMs, detectTotalMs: b.totalMs };
    rows.push(row);
    console.log(`        ${String(s).padStart(7)} | ${f1(a.recogniserMs).padStart(6)} | ${f1(b.recogniserMs).padStart(6)} | ${(b.recogniserMs / a.recogniserMs).toFixed(2).padStart(13)} | ${f1(c.recogniserMs).padStart(12)} | ${f1(a.encoderMs).padStart(14)} | ${f1(b.encoderMs).padStart(14)}`);
  }
  const diffs = rows.map((r) => r.detectMs - r.forcedMs);
  const ratio = median(rows.map((r) => r.ratio));
  console.log(`      ${label}: detection adds a median of ${f1(median(diffs))} ms per utterance (median ratio ${ratio.toFixed(2)}); ` +
    `first call: forced ${runs.forced.summary.first.recogniserMs} ms, detect ${runs.detect.summary.first.recogniserMs} ms; ` +
    `steady state of the same utterance (${PLAN_SECONDS[0]} s): forced ${f1(runs.forced.summary.per[PLAN_SECONDS[0]].recogniserMs)} ms, ` +
    `detect ${f1(runs.detect.summary.per[PLAN_SECONDS[0]].recogniserMs)} ms`);
  return { rows, medianExtraMs: median(diffs), medianRatio: ratio, runs };
}

let context = null;
try {
  // ---- A: CPU (WebAssembly) ----
  if (process.env.CZECHSPEECH_DETECT_COST_SKIP_CPU !== '1') {
    context = await launch(null);
    const page = await open(context);
    results.cpuBrowser = (await page.evaluate(() => window.e2e.info())).userAgent;
    results.cpu = await measure('A: CPU', page, 'cpu');
    await page.close();
    await context.close();
    context = null;
  }

  // ---- B and C: WebGPU ----
  if (process.env.CZECHSPEECH_DETECT_COST_SKIP_GPU !== '1') {
    context = await launch('chromium');
    const page = await open(context);
    const gi = await page.evaluate(() => window.e2e.gpuInfo());
    results.gpu = { adapter: gi.info, probe: gi.probe };
    console.log(`      WebGPU in a worker: ${JSON.stringify(gi.info)} ${gi.probe ? JSON.stringify(gi.probe) : gi.error}`);
    if (!gi.probe || !gi.probe.ok) {
      console.log(`SKIP  B, C: no usable WebGPU adapter in a worker of this browser (${gi.probe ? gi.probe.reason : gi.error})`);
      results.gpu.skipped = true;
    } else {
      check(!(gi.info.isFallbackAdapter || /swiftshader|llvmpipe|software|microsoft basic/i.test(`${gi.info.vendor} ${gi.info.architecture} ${gi.info.description}`)),
        `B: the worker sees a hardware adapter (${gi.info.vendor} ${gi.info.architecture})`);
      results.webgpu = await measure('B: WebGPU', page, 'webgpu');

      // C: consecutive multi-language recognitions on WebGPU, GPU buffers after each
      await sleep(pauseMs);
      const cycle = [2, 3, 4, 6.9];
      const calls = Array.from({ length: SMOKE }, (_, i) => cycle[i % cycle.length]);
      const r = await run(page, 'detect', 'webgpu', { tracked: true, calls });
      const series = r.calls.map((c) => ({ i: c.i + 1, seconds: c.seconds, ms: c.ms, recogniserMs: c.recogniserMs,
        liveBytes: c.gpu.liveBytes, liveBuffers: c.gpu.liveBuffers, peakBytes: c.gpu.peakBytes, devices: c.gpu.devices, sdkGpuBytes: c.stateGpuBytes, error: c.gpu.error }));
      const stateAfter = r.state;
      await page.evaluate(() => window.e2e.close());
      await sleep(500);
      const destroyed = await page.evaluate(() => window.e2e.gpuDestroyed());
      const mib = (b) => (b / 2 ** 20).toFixed(1);
      console.log(`      C: GPU buffers (MiB) after each of ${SMOKE} consecutive recognitions, 'detect' model, WebGPU:`);
      console.log(`        ${series.map((s) => `${s.i}:${s.error ? 'ERR' : mib(s.liveBytes)}/${s.liveBuffers}`).join('  ')}   (MiB/buffers)`);
      console.log(`        peak ${mib(Math.max(...series.map((s) => s.peakBytes || 0)))} MiB; SDK state.gpuBytes after the run ${mib(stateAfter.gpuBytes)} MiB, peakGpuBytes ${mib(stateAfter.peakGpuBytes)} MiB`);
      console.log(`        at close(): device destroyed ${JSON.stringify(destroyed)}`);
      results.smoke = { series, destroyed, state: { gpuBytes: stateAfter.gpuBytes, peakGpuBytes: stateAfter.peakGpuBytes, modelLoads: stateAfter.modelLoads } };
      const base = series[0].liveBytes;
      const last = series[series.length - 1].liveBytes;
      check(series.every((s) => !s.error && s.liveBytes > 400 * 2 ** 20), 'C: the model stayed on the GPU through every recognition');
      check(series.every((s) => s.liveBytes === base && s.liveBuffers === series[0].liveBuffers),
        `C: no growth per recognition: ${mib(base)} MiB / ${series[0].liveBuffers} buffers after the first, ${mib(last)} MiB / ${series[series.length - 1].liveBuffers} after the last`);
      check(destroyed.length >= 1, `C: the device was destroyed at close() (${destroyed.length} report)`);
      check(destroyed.length >= 1 && destroyed.every((d) => d.leftBytes === 0),
        `C: no buffer was left alive when the device was destroyed (${destroyed.map((d) => `${mib(d.leftBytes)} MiB in ${d.leftBuffers} buffers`).join(', ') || 'no report'})`);
      check(stateAfter.modelLoads === 1, `C: one model load for all ${SMOKE} recognitions`);
    }
    await page.close();
    await context.close();
    context = null;
  }
  check(pageErrors.length === 0, `no page errors${pageErrors.length ? `: ${pageErrors.join(' | ')}` : ''}`);
} catch (e) {
  failures++;
  console.log(`FAIL  ${e && e.stack ? e.stack : e}`);
} finally {
  if (context) await context.close();
  await server.close();
  fs.rmSync(profile, { recursive: true, force: true });
}

const out = path.join(os.tmpdir(), 'czechspeech-detect-cost.json');
// the transcripts are not written out: only their lengths
const strip = (o) => JSON.parse(JSON.stringify(o, (k, v) => (k === 'text' ? undefined : k === 'logs' ? undefined : v)));
fs.writeFileSync(out, JSON.stringify(strip(results), null, 2));
console.log(`results: ${out}`);
console.log(failures ? `${failures} FAILED` : 'ALL PASSED');
process.exit(failures ? 1 : 0);
