// The runaway decode of the meeting app's 3-hour soak test, reproduced through the real audio path, and the proof that
// it now ends in bounded time on WebGPU with the device kept (opt-in, not part of npm run test:e2e).
//
// Chrome's fake microphone plays CZECHSPEECH_TEST_LOOP (the soak loop of the meeting app: 32 FLEURS sentences,
// test/soak/out/smycka-cs.wav there; any speech recording works). The page records it through the SDK's own microphone
// path (getUserMedia, the AudioWorklet, its resampler) into memory, then runs listen() over the recorded frames twice on
// WebGPU, so that both runs get exactly the same utterances:
//   1. with whisper.cpp's decoding as before 0.2.0 (no token cap, temperature fallback; internal engine options) and no
//      watchdog in the way: the utterances whose recognition took longer than RUNAWAY_MS are the runaways;
//   2. as shipped (token cap, no fallback, the default watchdog, backend 'auto'): every recognition, the runaways
//      included, must end within BOUND_MS, on WebGPU throughout, without a WebGPU failure and without loading the model
//      again (so with the same device).
// The runaway does not come in every pass of the loop (in the soak about one pass in three), so up to
// CZECHSPEECH_RUNAWAY_CAPTURES captures are tried; without a runaway the script says so and passes nothing.
// No transcript text is printed: word counts and timings only.
//
// Env: CZECHSPEECH_MODEL, CZECHSPEECH_TEST_LOOP, CZECHSPEECH_RUNAWAY_SECONDS (per capture, default 560: three passes of
//      the soak loop), CZECHSPEECH_RUNAWAY_CAPTURES (default 2), CZECHSPEECH_E2E_CHANNEL (default 'chromium'; 'chrome' for
//      the installed Chrome), CZECHSPEECH_TEST_THREADS.
// Usage: node test/e2e/runaway.mjs
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { chromium } from 'playwright-core';
import { startServer } from '../../scripts/serve.mjs';
import { DEFAULT_MODEL } from '../../src/model.js';
import { TestEnv, missing } from '../support/env.mjs';

const LOOP = process.env.CZECHSPEECH_TEST_LOOP ? path.resolve(process.env.CZECHSPEECH_TEST_LOOP) : null;
const skip = missing('CZECHSPEECH_MODEL', TestEnv.model, 'CzechSpeech model') ||
  missing('CZECHSPEECH_TEST_LOOP', LOOP, 'speech recording for the fake microphone');
if (skip) {
  console.log(`SKIP: ${skip}`);
  process.exit(0);
}
const SECONDS = Number(process.env.CZECHSPEECH_RUNAWAY_SECONDS || 560);
const CAPTURES = Number(process.env.CZECHSPEECH_RUNAWAY_CAPTURES || 2);
const RUNAWAY_MS = 10000;
const BOUND_MS = 5000;

let failures = 0;
const check = (ok, message) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${message}`);
  if (!ok) failures++;
};

/** One entry per recognition of a session, in order: audio ms and recogniser ms (the time whisper.cpp took). */
function recognitions(logs) {
  const out = [];
  for (const l of logs) {
    const m = /utterance: (\d+) ms of audio, (.*)$/.exec(l);
    if (!m || /no speech/.test(m[2])) continue;
    const r = /recogniser (\d+) ms/.exec(m[2]);
    out.push({ audioMs: Number(m[1]), recogniserMs: r ? Number(r[1]) : null, stopped: /was stopped/.test(m[2]) });
  }
  return out;
}

const server = await startServer({ port: 0, model: TestEnv.model, quiet: true });
const origin = `http://127.0.0.1:${server.port}`;
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'czechspeech-runaway-'));
const requests = [];
const context = await chromium.launchPersistentContext(profile, {
  headless: true,
  channel: process.env.CZECHSPEECH_E2E_CHANNEL || 'chromium',
  args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', `--use-file-for-fake-audio-capture=${LOOP}`,
    '--autoplay-policy=no-user-gesture-required'],
});
context.on('request', (r) => requests.push(r.url()));
try {
  const page = await context.newPage();
  await page.goto(`${origin}/test/e2e/page.html`);
  await page.waitForFunction(() => window.e2e && window.e2e.ready);
  const gi = await page.evaluate(() => window.e2e.gpuInfo());
  if (!gi.probe || !gi.probe.ok) {
    console.log(`SKIP: no usable WebGPU adapter in a worker (${gi.probe ? gi.probe.reason : gi.error})`);
  } else {
    console.log(`adapter: ${JSON.stringify(gi.info)}; ${(await page.evaluate(() => window.e2e.info())).userAgent}`);
    await page.evaluate((a) => window.e2e.model(a),
      { url: `${origin}/models/${path.basename(TestEnv.model)}`, sha256: DEFAULT_MODEL.sha256, size: fs.statSync(TestEnv.model).size });
    let before = null;
    let runaways = [];
    for (let k = 1; k <= CAPTURES && runaways.length === 0; k++) {
      const frames = await page.evaluate((s) => window.e2e.captureMic(s), SECONDS);
      console.log(`capture ${k}: ${(frames * 0.032).toFixed(0)} s of the fake microphone through the SDK's microphone path`);
      await page.evaluate((t) => window.e2e.create({ config: { threads: t, backend: 'webgpu' }, keepModelLoaded: true,
        _engineOptions: { tokenCap: false, temperatureFallback: true }, _gpuTimeouts: { recognizeBase: 600000, recognizePerSecond: 0 } }),
      TestEnv.threads);
      const r = await page.evaluate(() => window.e2e.replayCapture());
      before = recognitions(r.logs);
      runaways = before.map((x, i) => ({ ...x, i })).filter((x) => x.recogniserMs > RUNAWAY_MS);
      console.log(`  decoding as before 0.2.0: ${before.length} recognitions, ${runaways.length} took longer than ${RUNAWAY_MS / 1000} s` +
        `${runaways.length ? `: ${runaways.map((x) => `#${x.i} ${x.audioMs} ms of audio in ${x.recogniserMs} ms`).join(', ')}` : ''}`);
    }
    if (runaways.length === 0) {
      console.log(`SKIP: no runaway decode in ${CAPTURES} capture(s) of ${SECONDS} s; nothing to compare`);
    } else {
      await page.evaluate((t) => window.e2e.create({ config: { threads: t }, keepModelLoaded: true }), TestEnv.threads);
      const r = await page.evaluate(() => window.e2e.replayCapture());
      const after = recognitions(r.logs);
      check(!r.error, `the session ran without an error${r.error ? `: ${JSON.stringify(r.error)}` : ''}`);
      check(after.length === before.length && after.every((x, i) => x.audioMs === before[i].audioMs),
        `the same ${before.length} utterances (${after.length} now)`);
      for (const x of runaways) {
        const a = after[x.i];
        check(a && a.recogniserMs !== null && a.recogniserMs < BOUND_MS,
          `runaway #${x.i} (${x.audioMs} ms of audio): ${x.recogniserMs} ms before, ${a ? `${a.recogniserMs ?? 'stopped'} ms` : 'missing'} now`);
      }
      const slowest = Math.max(...after.map((x) => x.recogniserMs || 0));
      check(slowest < BOUND_MS, `every recognition within ${BOUND_MS / 1000} s (slowest ${slowest} ms)`);
      check(after.every((x) => !x.stopped), 'no recognition needed the watchdog');
      check(r.state.backend === 'webgpu' && r.state.backendFallback === null && !r.logs.some((l) => /WebGPU failed/.test(l)),
        `on WebGPU throughout (backend ${r.state.backend}, fallback ${r.state.backendFallback})`);
      check(r.state.modelLoads === 1, `the model was loaded once: the WebGPU module and its device were kept (${r.state.modelLoads} loads)`);
      const words = r.events.filter((e) => e.type === 'transcript').map((e) => e.words);
      console.log(`      ${words.length} transcripts, ${words.reduce((s, n) => s + n, 0)} words`);
    }
    await page.evaluate(() => window.e2e.close());
  }
  const foreign = requests.filter((u) => !u.startsWith(`${origin}/`));
  check(foreign.length === 0, `no request left the local server (${requests.length} requests)`);
} catch (e) {
  failures++;
  console.log(`FAIL  ${e && e.stack ? e.stack : e}`);
} finally {
  await context.close();
  await server.close();
  fs.rmSync(profile, { recursive: true, force: true });
}
console.log(failures ? `${failures} FAILED` : 'done');
process.exit(failures ? 1 : 0);
