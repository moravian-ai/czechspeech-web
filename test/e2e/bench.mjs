// Speed and memory of the 6.9 s e-rec clip in Chromium (opt-in measurement, not a test): the CPU modules
// (multi-threaded, single-threaded) and WebGPU, one configuration after the other in one browser, with pauses.
//
// Per configuration, at most 5 timed transcribe() calls:
//   run 1      a new CzechSpeech (keepModelLoaded): recogniser start + model load + first recognition, which is what
//              every listen() session and transcribe() call costs with the default keepModelLoaded: false
//   runs 2..   the same instance, model loaded ("warm")
//   WebGPU     also one more new instance at the end: a first run once the browser has compiled the shaders before
// Recorded: wall time, model load, recogniser, encoder, decoder per token, peak WebAssembly heap, GPU buffers
// (WebGPU), and the page's memory (measureUserAgentSpecificMemory, cross-origin isolated pages) with the model loaded.
//
// Env: CZECHSPEECH_MODEL, CZECHSPEECH_TEST_AUDIO, CZECHSPEECH_TEST_THREADS (default 6),
//      CZECHSPEECH_BENCH (configurations, default "cpu-mt,cpu-st,webgpu"), CZECHSPEECH_BENCH_PAUSE_MS (default 45000).
// Usage: node test/e2e/bench.mjs      (writes czechspeech-bench.json to the temporary directory)
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { chromium } from 'playwright-core';
import { startServer } from '../../scripts/serve.mjs';
import { TestEnv, missing } from '../support/env.mjs';

const skip = missing('CZECHSPEECH_MODEL', TestEnv.model, 'CzechSpeech model') ||
  missing('CZECHSPEECH_TEST_AUDIO', TestEnv.audioDir, 'test audio directory');
if (skip) {
  console.log(`SKIP: ${skip}`);
  process.exit(0);
}
const configs = (process.env.CZECHSPEECH_BENCH || 'cpu-mt,cpu-st,webgpu').split(',').map((s) => s.trim()).filter(Boolean);
const pauseMs = Number(process.env.CZECHSPEECH_BENCH_PAUSE_MS || 45000);
const RUNS = { 'cpu-mt': 3, 'cpu-st': 2, webgpu: 4 }; // + 1 extra new instance for WebGPU = 5
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const mib = (b) => (b ? Math.round(b / 2 ** 20) : null);

const server = await startServer({ port: 0, model: TestEnv.model, fixtures: TestEnv.audioDir, quiet: true });
const origin = `http://127.0.0.1:${server.port}`;
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'czechspeech-bench-'));
const context = await chromium.launchPersistentContext(profile, { headless: true, channel: 'chromium' });
const out = { date: new Date().toISOString(), browser: null, adapter: null, cores: null, threads: TestEnv.threads, configs: {} };

function parse(logs) {
  const load = logs.map((l) => /model loaded in (\d+) ms/.exec(l)).find(Boolean);
  const rec = logs.map((l) => /transcribe: \d+ ms of audio, .*gate (\d+) ms, recogniser (\d+) ms \(encoder (\d+) ms, decoder ([\d.]+) ms\/token\)/.exec(l))
    .filter(Boolean).pop();
  return { modelLoadMs: load ? Number(load[1]) : null, gateMs: rec ? Number(rec[1]) : null, recogniserMs: rec ? Number(rec[2]) : null,
    encoderMs: rec ? Number(rec[3]) : null, decoderMsPerToken: rec ? Number(rec[4]) : null };
}

try {
  let first = true;
  for (const c of configs) {
    if (!first) {
      console.log(`(pause ${pauseMs / 1000} s)`);
      await sleep(pauseMs);
    }
    first = false;
    const page = await context.newPage();
    await page.goto(`${origin}/test/e2e/page.html${c === 'cpu-st' ? '?coi=0' : ''}`);
    await page.waitForFunction(() => window.e2e && window.e2e.ready);
    const info = await page.evaluate(() => window.e2e.info());
    out.browser = info.userAgent;
    out.cores = info.hardwareConcurrency;
    if (c === 'webgpu') {
      const gi = await page.evaluate(() => window.e2e.gpuInfo());
      out.adapter = gi.info;
      if (!gi.probe || !gi.probe.ok) {
        console.log(`SKIP webgpu: ${gi.probe ? gi.probe.reason : gi.error}`);
        await page.close();
        continue;
      }
    }
    await page.evaluate((a) => window.e2e.model(a), { url: `${origin}/models/${path.basename(TestEnv.model)}`, size: fs.statSync(TestEnv.model).size });
    const options = { config: { threads: TestEnv.threads, backend: c === 'webgpu' ? 'webgpu' : 'cpu' }, keepModelLoaded: true };
    const runs = [];
    const once = async (label) => {
      const before = await page.evaluate(() => window.e2e.logs().length);
      const r = await page.evaluate((u) => window.e2e.transcribeUrl(u), `${origin}/fixtures/e-rec.wav`);
      const p = parse(r.logs.slice(before));
      const row = { label, wallMs: r.ms, ...p, peakHeapMiB: mib(r.state.peakHeapBytes), gpuMiB: mib(r.state.peakGpuBytes),
        backend: r.state.backend, chars: r.text.length };
      runs.push(row);
      console.log(`${c.padEnd(7)} ${label.padEnd(26)} ${JSON.stringify(row)}`);
      return r;
    };
    await page.evaluate((o) => window.e2e.create(o), options);
    await once('run 1 (new instance)');
    for (let i = 2; i <= RUNS[c]; i++) {
      await sleep(c === 'webgpu' ? 3000 : 1000);
      await once(`run ${i} (warm)`);
    }
    const memory = await page.evaluate(() => window.e2e.memory());
    if (c === 'webgpu') {
      await sleep(3000);
      await page.evaluate((o) => window.e2e.create(o), options);
      await once('run 5 (new instance again)');
    }
    await page.evaluate(() => window.e2e.close());
    await page.close();
    out.configs[c] = { runs, memoryMiB: memory && memory.bytes ? mib(memory.bytes) : null, memoryNote: memory && memory.error };
  }
} catch (e) {
  console.log(`FAIL  ${e && e.stack ? e.stack : e}`);
  process.exitCode = 1;
} finally {
  await context.close();
  await server.close();
  fs.rmSync(profile, { recursive: true, force: true });
}
const file = path.join(os.tmpdir(), 'czechspeech-bench.json');
fs.writeFileSync(file, JSON.stringify(out, null, 2));
console.log(`results: ${file}`);
