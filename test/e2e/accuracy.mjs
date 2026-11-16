// Accuracy of the WebGPU backend against the CPU one, in Chromium (opt-in, not part of npm run test:e2e):
// N clean ParCzech4Speech segments from the manifest (round-robin over its buckets, at most 25 s each) are decoded
// in the page and transcribed with config.backend 'cpu' (multi-threaded WebAssembly) and then 'webgpu', one at a time,
// and on WebGPU once more without the token cap of src/decoding.js (no temperature fallback in either run): every
// transcript must be the same with the cap and without it.
// Prints both word error rates against the reference and every word that differs between CPU and GPU, and writes
// czechspeech-accuracy.json to the temporary directory. The ParCzech transcripts printed here are public (CC BY 4.0);
// the SDK itself never logs text.
//
// Env: CZECHSPEECH_MODEL, CZECHSPEECH_TEST_MANIFEST (its audio/ files are served), CZECHSPEECH_TEST_THREADS (default 6),
//      CZECHSPEECH_ACCURACY_N (default 20), CZECHSPEECH_E2E_GPU_PAUSE_MS (pause between GPU recognitions, default 2000).
// Usage: node test/e2e/accuracy.mjs
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { chromium } from 'playwright-core';
import { startServer } from '../../scripts/serve.mjs';
import { TestEnv, missing, wer, wordAlign, wordDiff, words } from '../support/env.mjs';

const skip = missing('CZECHSPEECH_MODEL', TestEnv.model, 'CzechSpeech model') ||
  missing('CZECHSPEECH_TEST_MANIFEST', TestEnv.manifest, 'ParCzech manifest');
if (skip) {
  console.log(`SKIP: ${skip}`);
  process.exit(0);
}
const N = Number(process.env.CZECHSPEECH_ACCURACY_N || 20);
const pauseMs = Number(process.env.CZECHSPEECH_E2E_GPU_PAUSE_MS || 2000);
const manifestDir = path.dirname(TestEnv.manifest);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// round-robin over the buckets in manifest order: a reproducible, varied sample
const entries = fs.readFileSync(TestEnv.manifest, 'utf8').trim().split('\n').map((l) => JSON.parse(l))
  .filter((e) => (e.condition || 'clean') === 'clean' && e.dur <= 25 && fs.existsSync(path.join(manifestDir, e.path)));
const buckets = new Map();
for (const e of entries) {
  if (!buckets.has(e.bucket)) buckets.set(e.bucket, []);
  buckets.get(e.bucket).push(e);
}
const sample = [];
for (let i = 0; sample.length < N && [...buckets.values()].some((b) => b.length > i); i++) {
  for (const b of buckets.values()) if (b[i] && sample.length < N) sample.push(b[i]);
}

const server = await startServer({ port: 0, model: TestEnv.model, mounts: [{ prefix: '/parczech/', dir: manifestDir }], quiet: true });
const origin = `http://127.0.0.1:${server.port}`;
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'czechspeech-accuracy-'));
const context = await chromium.launchPersistentContext(profile, { headless: true, channel: 'chromium' });
const requests = [];
context.on('request', (r) => requests.push(r.url()));
const out = { browser: null, adapter: null, n: sample.length, segments: [] };
let failed = false;
try {
  const page = await context.newPage();
  await page.goto(`${origin}/test/e2e/page.html`);
  await page.waitForFunction(() => window.e2e && window.e2e.ready);
  out.browser = (await page.evaluate(() => window.e2e.info())).userAgent;
  const gi = await page.evaluate(() => window.e2e.gpuInfo());
  out.adapter = gi.info;
  if (!gi.probe || !gi.probe.ok) {
    console.log(`SKIP: no usable WebGPU adapter in a worker (${gi.probe ? gi.probe.reason : gi.error})`);
    process.exitCode = 0;
  } else {
    console.log(`adapter: ${JSON.stringify(gi.info)}; ${sample.length} segments, ${sample.reduce((s, e) => s + e.dur, 0).toFixed(0)} s of audio`);
    // with its SHA-256 the model is verified and cached in Cache Storage, as an app would use it (without one,
    // fromUrl() keeps it in an in-memory Blob, which the recogniser's first read failed on in Chrome 156)
    const sha256 = crypto.createHash('sha256').update(fs.readFileSync(TestEnv.model)).digest('hex');
    await page.evaluate((a) => window.e2e.model(a), { url: `${origin}/models/${path.basename(TestEnv.model)}`, size: fs.statSync(TestEnv.model).size, sha256 });
    const run = async (backend, engineOptions = null) => {
      await page.evaluate(({ t, b, o }) => window.e2e.create({ config: { threads: t, backend: b }, keepModelLoaded: true, _engineOptions: o }),
        { t: TestEnv.threads, b: backend, o: engineOptions });
      const texts = [];
      for (const e of sample) {
        const r = await page.evaluate((u) => window.e2e.transcribeDecoded(u), `${origin}/parczech/${e.path}`);
        if (r.state.backend !== backend) throw new Error(`${e.id}: ran on ${r.state.backend}, not ${backend}`);
        texts.push({ text: r.text, ms: r.ms });
        console.log(`  ${backend.padEnd(6)} ${e.id} ${e.dur.toFixed(1).padStart(5)} s  ${String(r.ms).padStart(6)} ms  WER ${wer(e.ref, r.text).toFixed(3)}`);
        if (backend === 'webgpu') await sleep(pauseMs);
      }
      await page.evaluate(() => window.e2e.close());
      return texts;
    };
    const cpu = await run('cpu');
    await sleep(Math.max(pauseMs, 5000));
    const gpu = await run('webgpu');
    await sleep(Math.max(pauseMs, 5000));
    // the same decoding without the token cap (both runs without the temperature fallback): the cap must not change a
    // legitimate transcript
    const unlimited = await run('webgpu', { tokenCap: false });
    let refWords = 0;
    let cpuErr = 0;
    let gpuErr = 0;
    let identical = 0;
    for (let i = 0; i < sample.length; i++) {
      const e = sample[i];
      const r = words(e.ref);
      const ce = wordAlign(r, words(cpu[i].text)).distance;
      const ge = wordAlign(r, words(gpu[i].text)).distance;
      refWords += r.length;
      cpuErr += ce;
      gpuErr += ge;
      const diff = wordDiff(cpu[i].text, gpu[i].text);
      if (diff.same) identical++;
      out.segments.push({ id: e.id, bucket: e.bucket, dur: e.dur, ref: e.ref, cpu: cpu[i], gpu: gpu[i],
        werCpu: ce / r.length, werGpu: ge / r.length, cpuVsGpu: diff.summary || null });
    }
    const changed = sample.filter((e, i) => gpu[i].text !== unlimited[i].text).map((e) => e.id);
    out.tokenCap = { changed };
    out.summary = { refWords, werCpu: cpuErr / refWords, werGpu: gpuErr / refWords, identical, of: sample.length,
      cpuMs: cpu.reduce((s, x) => s + x.ms, 0), gpuMs: gpu.reduce((s, x) => s + x.ms, 0) };
    console.log(`\nWER (corpus, ${refWords} reference words): CPU ${(100 * out.summary.werCpu).toFixed(2)} %, WebGPU ${(100 * out.summary.werGpu).toFixed(2)} %`);
    console.log(`identical transcripts: ${identical} of ${sample.length}; recognition time CPU ${(out.summary.cpuMs / 1000).toFixed(0)} s, WebGPU ${(out.summary.gpuMs / 1000).toFixed(0)} s`);
    console.log(`${changed.length ? 'FAIL' : 'PASS'}  WebGPU with the token cap and without it: ${sample.length - changed.length} of ${sample.length} transcripts identical${changed.length ? ` (differ: ${changed.join(', ')})` : ''}`);
    if (changed.length) failed = true;
    for (const s of out.segments.filter((x) => x.cpuVsGpu)) {
      console.log(`  ${s.id}: ${s.cpuVsGpu}   (WER CPU ${s.werCpu.toFixed(3)}, WebGPU ${s.werGpu.toFixed(3)})`);
      console.log(`    CPU:    "${s.cpu.text}"\n    WebGPU: "${s.gpu.text}"`);
    }
    const foreign = requests.filter((u) => !u.startsWith(`${origin}/`));
    if (foreign.length) {
      failed = true;
      console.log(`FAIL  requests left the local server: ${foreign.join(', ')}`);
    }
  }
} catch (e) {
  failed = true;
  console.log(`FAIL  ${e && e.stack ? e.stack : e}`);
} finally {
  await context.close();
  await server.close();
  fs.rmSync(profile, { recursive: true, force: true });
}
const file = path.join(os.tmpdir(), 'czechspeech-accuracy.json');
fs.writeFileSync(file, JSON.stringify(out, null, 2));
console.log(`results: ${file}`);
process.exit(failed ? 1 : 0);
