// Parity of the browser embeddings with sherpa-onnx: for the fine-tuned and the base model, the first 0.3 .. 10 s of
// e-rec.wav embedded by SpeakerEngine (worker, onnxruntime-web, fbank in JavaScript) in headless Chromium must have a
// cosine of at least 0.9999 with the sherpa-onnx (Python) embedding: on an isolated page with 1 and 2 threads and on a
// page that is not isolated (2 threads asked for: the runtime must fall back to 1 and say so through engine.threads).
// Every request of the pages is recorded; only the local server may be contacted.
// Env: CZECHSPEECH_VOICEPRINT_MODEL (fixed fine-tuned model), CZECHSPEECH_VOICEPRINT_BASE_MODEL, CZECHSPEECH_TEST_AUDIO
// (directory with e-rec.wav); CZECHSPEECH_PYTHON (default: the research virtual environment is NOT assumed: set it).
// Unset: SKIP, exit 0; a path that does not exist: exit 1. The reference embeddings (biometric data) go to
// test/speakers/out/ (git-ignored) and are deleted as soon as they are read, whatever the outcome.
// Usage: node test/speakers/parity.mjs
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { chromium } from 'playwright-core';
import { ROOT, TestEnv, localPath, missing } from '../support/env.mjs';
import { startSpeakerServer } from './server.mjs';

const fromEnv = (name) => (process.env[name] && process.env[name].trim() ? localPath(process.env[name].trim()) : null);
const models = { finetuned: fromEnv('CZECHSPEECH_VOICEPRINT_MODEL'), base: fromEnv('CZECHSPEECH_VOICEPRINT_BASE_MODEL') };
const python = fromEnv('CZECHSPEECH_PYTHON');
const audioDir = TestEnv.audioDir;
const unset = [['CZECHSPEECH_VOICEPRINT_MODEL', models.finetuned], ['CZECHSPEECH_VOICEPRINT_BASE_MODEL', models.base],
  ['CZECHSPEECH_TEST_AUDIO', audioDir], ['CZECHSPEECH_PYTHON', python]].find(([, v]) => !v);
if (unset) {
  console.log(`SKIP: ${unset[0]} not set`);
  process.exit(0);
}
const absent = missing('CZECHSPEECH_VOICEPRINT_MODEL', models.finetuned, 'fine-tuned voiceprint model') ||
  missing('CZECHSPEECH_VOICEPRINT_BASE_MODEL', models.base, 'base voiceprint model') ||
  missing('CZECHSPEECH_TEST_AUDIO', path.join(audioDir, 'e-rec.wav'), 'e-rec.wav') || missing('CZECHSPEECH_PYTHON', python, 'python');
if (absent) {
  console.error(`FAIL: ${absent}`);
  process.exit(1);
}

const CUTS = [0.3, 0.6, 1, 1.5, 2, 3, 5, 10];
// `asked` threads are requested; `got` are expected (a page that is not cross-origin isolated has no shared memory: 1)
const RUNS = [{ asked: 1, got: 1, isolated: true }, { asked: 2, got: 2, isolated: true }, { asked: 1, got: 1, isolated: false },
  { asked: 2, got: 1, isolated: false }];
const OUT = path.join(ROOT, 'test', 'speakers', 'out');
fs.mkdirSync(OUT, { recursive: true });
const cosine = (a, b) => { let s = 0, x = 0, y = 0; for (let i = 0; i < a.length; i++) { s += a[i] * b[i]; x += a[i] * a[i]; y += b[i] * b[i]; } return s / Math.sqrt(x * y); };
const failures = [];
const outside = [];
const notes = [];
let checks = 0;
const dropReferences = () => { for (const name of Object.keys(models)) fs.rmSync(path.join(OUT, `ref-${name}.json`), { force: true }); };
process.on('exit', dropReferences);

for (const [modelName, modelPath] of Object.entries(models)) {
  const bytes = fs.readFileSync(modelPath);
  const pins = { model: { size: bytes.length, sha256: crypto.createHash('sha256').update(bytes).digest('hex') } };
  const refFile = path.join(OUT, `ref-${modelName}.json`);
  const r = spawnSync(python, ['-I', path.join(ROOT, 'test', 'speakers', 'ref', 'ref_embeddings.py'), modelPath, path.join(audioDir, 'e-rec.wav'), refFile],
    { encoding: 'utf8' });
  if (r.status !== 0) { console.error(r.stderr); failures.push(`${modelName}: the reference script failed`); continue; }
  const reference = JSON.parse(fs.readFileSync(refFile, 'utf8'));
  fs.rmSync(refFile, { force: true });
  if (JSON.stringify(Object.keys(reference).sort()) !== JSON.stringify(CUTS.map(String).sort())) {
    failures.push(`${modelName}: the reference holds ${Object.keys(reference).join(', ')}, expected ${CUTS.join(', ')}`);
    continue;
  }

  for (const run of RUNS) {
    const server = await startSpeakerServer({ isolated: run.isolated, files: {
      '/page/parity.html': path.join(ROOT, 'test/speakers/page/parity.html'),
      '/files/model.onnx': modelPath,
      '/files/e-rec.wav': path.join(audioDir, 'e-rec.wav'),
    } });
    const browser = await chromium.launch({ headless: true });
    try {
      const context = await browser.newContext();
      const page = await context.newPage();
      await page.route('**/*', (route) => {
        const url = route.request().url();
        if (url.startsWith(`${server.origin}/`) || url.startsWith('blob:') || url.startsWith('data:')) return route.continue();
        outside.push(url);
        return route.abort();
      });
      page.on('pageerror', (e) => failures.push(`${modelName}: page error: ${e.message}`));
      await page.goto(`${server.origin}/page/parity.html`);
      await page.waitForFunction(() => window.parityReady);
      const got = await page.evaluate((args) => window.runParity(args), {
        modelUrl: `${server.origin}/files/model.onnx`, pins, threads: run.asked, cuts: CUTS, audioUrl: `${server.origin}/files/e-rec.wav` });
      const label = `${modelName} ${run.asked} thr. asked, ${run.isolated ? 'isolated' : 'not isolated'}`;
      for (const [seconds, ref] of Object.entries(reference)) {
        if (!got[seconds] || got[seconds].length !== ref.length) { failures.push(`${label} ${seconds} s: no embedding of the right length`); continue; }
        const c = cosine(got[seconds], ref);
        checks++;
        if (!(c >= 0.9999)) failures.push(`${label} ${seconds} s: cosine ${c.toFixed(6)}`);
      }
      if (got.isolated !== run.isolated) failures.push(`crossOriginIsolated is ${got.isolated}, expected ${run.isolated}`);
      if (got.threads !== run.got) failures.push(`${label}: ${got.threads} threads in use, expected ${run.got}`);
      notes.push(`${label}: ${got.threads} thr., min cosine ${Math.min(...Object.entries(reference).map(([s, ref]) => cosine(got[s], ref))).toFixed(7)}, ms ${JSON.stringify(got.ms)}`);
    } catch (err) {
      failures.push(`${modelName} ${run.asked} ${run.isolated}: ${err.message}`);
    } finally {
      await browser.close();
      await server.close();
    }
  }
}
console.log(notes.join('\n'));
if (outside.length) failures.push(`request outside the local server: ${outside.join(', ')}`);
if (failures.length) { console.error(failures.join('\n')); process.exit(1); }
const expected = Object.keys(models).length * RUNS.length * CUTS.length;
if (checks !== expected) { console.error(`only ${checks} of ${expected} cosine checks were made`); process.exit(1); }
console.log(`PASS: ${checks} cosine checks, 0 requests outside`);
