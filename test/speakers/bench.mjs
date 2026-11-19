// Time and memory of the voiceprint in the browser: for 1 and 2 threads the median of 5 runs (2 warm-up) of
// SpeakerEngine.embed for 2, 5, 10 and 28 s of audio (e-rec.wav repeated) and of embedUtterance (with the search for
// speaker changes) for 5 and 10 s, in headless Chromium on a cross-origin isolated page.
// Memory, for 1 and 2 threads separately: the baseline is taken after the engine is created and warmed up with two 10 s
// embeddings (so the model, the runtime and the first activations are in it); growth is the value after the 28 s input
// and after an utterance of 28 s minus that baseline. Two measures: the page's own `measureUserAgentSpecificMemory()`
// (the page, its workers and the WebAssembly memory, in bytes) and the working set of the process tree of the browser
// this script launched (Windows only; never other browser processes of the machine).
// Thresholds apply only on the reference notebook (CZECHSPEECH_BENCH_REFERENCE=1); otherwise the table is only printed:
//   embed 10 s with 2 threads <= 1500 ms, embedUtterance 10 s with 2 threads <= 3500 ms,
//   growth of the memory after the 28 s input <= 400 MB (both thread counts; the page's measure where the browser has
//   it, else the working set of our browser's process tree).
// Env: CZECHSPEECH_VOICEPRINT_MODEL (fixed fine-tuned model), CZECHSPEECH_TEST_AUDIO (directory with e-rec.wav).
// Usage: node test/speakers/bench.mjs
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { chromium } from 'playwright-core';
import { ROOT, TestEnv, localPath, missing } from '../support/env.mjs';
import { startSpeakerServer } from './server.mjs';

const model = process.env.CZECHSPEECH_VOICEPRINT_MODEL && localPath(process.env.CZECHSPEECH_VOICEPRINT_MODEL.trim());
const audioDir = TestEnv.audioDir;
if (!model || !audioDir) {
  console.log(`SKIP: ${!model ? 'CZECHSPEECH_VOICEPRINT_MODEL' : 'CZECHSPEECH_TEST_AUDIO'} not set`);
  process.exit(0);
}
const absent = missing('CZECHSPEECH_VOICEPRINT_MODEL', model, 'voiceprint model') || missing('CZECHSPEECH_TEST_AUDIO', path.join(audioDir, 'e-rec.wav'), 'e-rec.wav');
if (absent) { console.error(`FAIL: ${absent}`); process.exit(1); }

const REFERENCE = process.env.CZECHSPEECH_BENCH_REFERENCE === '1';
// CZECHSPEECH_BENCH_MAX_INPUT_MS: the input cap of the engine (default 10 000); a large value shows what the cap saves
const MAX_INPUT_MS = process.env.CZECHSPEECH_BENCH_MAX_INPUT_MS ? Number(process.env.CZECHSPEECH_BENCH_MAX_INPUT_MS) : undefined;
const RUNS = 5;
const WARMUP = 2;
const failures = [];
const outside = [];
const rows = [];

/**
 * Working set (MB) of the process tree of the browser launched with the profile directory `token` in its command line:
 * the sum over the browser process and all its descendants. Windows only; null elsewhere or when the root is not found.
 */
function treeMB(token) {
  if (process.platform !== 'win32') return null;
  const script = `$all=Get-CimInstance Win32_Process;$root=$all|Where-Object{$_.CommandLine -and $_.CommandLine.Contains('${token}') -and -not $_.CommandLine.Contains('--type=')}|Select-Object -First 1;` +
    `if(-not $root){return};$ids=@{};$ids[[int]$root.ProcessId]=1;do{$n=0;foreach($p in $all){if($ids.ContainsKey([int]$p.ParentProcessId) -and -not $ids.ContainsKey([int]$p.ProcessId)){$ids[[int]$p.ProcessId]=1;$n++}}}while($n -gt 0);` +
    `[math]::Round((($all|Where-Object{$ids.ContainsKey([int]$_.ProcessId)}|Measure-Object WorkingSetSize -Sum).Sum)/1MB)`;
  const r = spawnSync('powershell', ['-NoProfile', '-Command', script], { encoding: 'utf8' });
  const v = Number(r.stdout.trim());
  return Number.isFinite(v) && v > 0 ? v : null;
}

const bytes = fs.readFileSync(model);
const pins = { model: { size: bytes.length, sha256: crypto.createHash('sha256').update(bytes).digest('hex') } };
const growth = [];

for (const threads of [1, 2]) {
  const server = await startSpeakerServer({ isolated: true, files: {
    '/page/bench.html': path.join(ROOT, 'test/speakers/page/bench.html'),
    '/files/model.onnx': model,
    '/files/e-rec.wav': path.join(audioDir, 'e-rec.wav'),
  } });
  // a persistent context in a profile directory of our own: its name marks the browser processes that are ours
  const token = `czechspeech-bench-${process.pid}-${threads}`;
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), `${token}-`));
  const context = await chromium.launchPersistentContext(profile, { headless: true });
  try {
    const page = await context.newPage();
    await page.route('**/*', (route) => {
      const url = route.request().url();
      if (url.startsWith(`${server.origin}/`) || url.startsWith('blob:') || url.startsWith('data:')) return route.continue();
      outside.push(url);
      return route.abort();
    });
    page.on('pageerror', (e) => failures.push(`page error: ${e.message}`));
    await page.goto(`${server.origin}/page/bench.html`);
    await page.waitForFunction(() => window.benchReady);
    const info = await page.evaluate((a) => window.create(a), { modelUrl: `${server.origin}/files/model.onnx`, pins, threads, maxInputMs: MAX_INPUT_MS, audioUrl: `${server.origin}/files/e-rec.wav` });
    if (info.threads !== threads) failures.push(`${threads} threads requested, ${info.threads} in use`);
    await page.evaluate((a) => window.time(a), { kind: 'embed', seconds: 10, runs: 2, warmup: 0 }); // warm-up: not timed
    const mem = async () => ({ page: await page.evaluate(() => window.memoryMB()), tree: treeMB(profile) });
    const base = await mem();
    for (const [kind, seconds] of [['embed', 2], ['embed', 5], ['embed', 10], ['embed', 28], ['utterance', 5], ['utterance', 10]]) {
      const ms = await page.evaluate((a) => window.time(a), { kind, seconds, runs: RUNS, warmup: WARMUP });
      rows.push({ threads, kind, seconds, ms });
      await new Promise((r) => setTimeout(r, 1500)); // short pauses between the measurements
    }
    await page.evaluate((a) => window.time(a), { kind: 'embed', seconds: 28, runs: 1, warmup: 0 });
    const after = await mem();
    const d = (a, b) => (a == null || b == null ? null : b - a);
    growth.push({ threads, base, after, pageGrowthMB: d(base.page, after.page), treeGrowthMB: d(base.tree, after.tree) });
    console.log(`${threads} thread(s): isolated ${info.isolated}, model download ${info.downloadMs} ms, engine ready ${info.createMs} ms`);
    await page.evaluate(() => window.shutdown());
  } catch (err) {
    failures.push(`${threads} threads: ${err.message}`);
  } finally {
    await context.close();
    fs.rmSync(profile, { recursive: true, force: true });
    await server.close();
  }
}

console.table(rows);
for (const g of growth) {
  console.log(`memory with ${g.threads} thread(s), MB (page = measureUserAgentSpecificMemory, tree = working set of our browser): ` +
    `after create + warm-up ${JSON.stringify(g.base)}, after the 28 s input ${JSON.stringify(g.after)}, growth page ${g.pageGrowthMB} tree ${g.treeGrowthMB}`);
}
const at = (threads, kind, seconds) => rows.find((r) => r.threads === threads && r.kind === kind && r.seconds === seconds)?.ms;
if (REFERENCE) {
  if (!(at(2, 'embed', 10) <= 1500)) failures.push(`embed 10 s with 2 threads ${at(2, 'embed', 10)} ms > 1500 ms`);
  if (!(at(2, 'utterance', 10) <= 3500)) failures.push(`embedUtterance 10 s with 2 threads ${at(2, 'utterance', 10)} ms > 3500 ms`);
  for (const g of growth) {
    // the page's own measure where the browser has it (not the headless shell), else the working set of our browser's processes
    const grown = g.pageGrowthMB ?? g.treeGrowthMB;
    if (grown == null) failures.push(`${g.threads} threads: memory could not be measured`);
    else if (!(grown <= 400)) failures.push(`${g.threads} threads: memory growth ${grown} MB > 400 MB`);
  }
} else {
  console.log('thresholds not applied (CZECHSPEECH_BENCH_REFERENCE is not 1)');
}
if (outside.length) failures.push(`request outside the local server: ${outside.join(', ')}`);
if (failures.length) { console.error(failures.join('\n')); process.exit(1); }
console.log('PASS');
