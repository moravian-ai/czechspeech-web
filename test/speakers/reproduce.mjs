// Acceptance test of the browser runtime: the measured name accuracy on the 20 four-person test meetings is reproduced
// by the SDK's own parts (Endpointer, SpeakerEngine in a worker with onnxruntime-web, Tagger) in headless Chromium.
//
//   Level 1: speech probabilities from the Silero v6.2 ONNX model through onnxruntime-web, exactly as the measurement
//            (research/speakers/vad.py): this isolates the speaker runtime and must reproduce the measured numbers.
//   Level 2: speech probabilities from the SDK's WebAssembly VAD path (test/speakers/sdk-vad.mjs): what the product does;
//            only reported.
//   Modes:   `enrolled` (every speaker with a stored voice made from the enrolment recording) and `anon` (nobody).
//
// Env: CZECHSPEECH_VOICEPRINT_MODEL (fixed fine-tuned model), CZECHSPEECH_VOICEPRINT_BASE_MODEL, CZECHSPEECH_SILERO_ONNX,
//      CZECHSPEECH_SPEAKER_MEETINGS (directory with test-S4-NN.json/.wav and enroll.<speaker>.wav).
// Output: summary numbers to the console and test/speakers/out/reproduce.json; no names, no voiceprints.
// Usage: node test/speakers/reproduce.mjs
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright-core';
import { ROOT, localPath, missing } from '../support/env.mjs';
import { Tagger } from '../../src/speakers/tagger.js';
import { BASE_DEPLOYABLE, STRICT } from '../../src/speakers/points.js';
import { VOICEPRINT_MODELS } from '../../src/speakers/catalog.js';
import { enrollFile, readMeetingMeta, s4TestIds } from './meetings.mjs';
import { scoreMeeting } from './metrics.mjs';
import { startSpeakerServer } from './server.mjs';

export const ACCEPT = {
  finetuned: { pieces: 2495, namedOk: 0.880561, maxConfused: 19 },   // measured: 2197 / 2495, 16 confused
  base:      { pieces: 2546, namedOk: 0.653574, maxConfused: 16 },   // measured: 1664 / 2546, 13 confused
  namedOkTolerance: 0.005,
  piecesTolerance: 0.01,
};

const fromEnv = (name) => (process.env[name] && process.env[name].trim() ? localPath(process.env[name].trim()) : null);
const paths = {
  finetuned: fromEnv('CZECHSPEECH_VOICEPRINT_MODEL'),
  base: fromEnv('CZECHSPEECH_VOICEPRINT_BASE_MODEL'),
  silero: fromEnv('CZECHSPEECH_SILERO_ONNX'),
  meetings: fromEnv('CZECHSPEECH_SPEAKER_MEETINGS'),
};
const names = { finetuned: 'CZECHSPEECH_VOICEPRINT_MODEL', base: 'CZECHSPEECH_VOICEPRINT_BASE_MODEL', silero: 'CZECHSPEECH_SILERO_ONNX', meetings: 'CZECHSPEECH_SPEAKER_MEETINGS' };
const unset = Object.keys(paths).find((k) => !paths[k]);
if (unset) {
  console.log(`SKIP: ${names[unset]} not set`);
  process.exit(0);
}
for (const k of Object.keys(paths)) {
  const absent = missing(names[k], paths[k], k);
  if (absent) { console.error(`FAIL: ${absent}`); process.exit(1); }
}

const OUT = path.join(ROOT, 'test', 'speakers', 'out');
const VAD_DIR = path.join(OUT, 'vad-sdk');
fs.mkdirSync(OUT, { recursive: true });
const ids = s4TestIds(paths.meetings);
const metas = ids.map((id) => readMeetingMeta(paths.meetings, id));
const level2 = ids.every((id) => fs.existsSync(path.join(VAD_DIR, `${id}.f32`)));
if (!level2) console.log(`level 2 skipped: run test/speakers/sdk-vad.mjs first (no probabilities in ${VAD_DIR})`);

const POINTS = { finetuned: STRICT, base: BASE_DEPLOYABLE };
const summary = {};
const failures = [];
const outside = [];

for (const modelName of ['finetuned', 'base']) {
  const bytes = fs.readFileSync(paths[modelName]);
  const pins = { model: { size: bytes.length, sha256: crypto.createHash('sha256').update(bytes).digest('hex') } };
  if (modelName === 'finetuned') {
    // the file under test must be the one the catalog (and its operating point) is bound to
    const entry = VOICEPRINT_MODELS['voiceprint-cs'];
    if (pins.model.sha256 !== entry.sha256 || pins.model.size !== entry.size) {
      console.error(`FAIL: CZECHSPEECH_VOICEPRINT_MODEL is not the catalog's voiceprint-cs (sha256 ${pins.model.sha256}, size ${pins.model.size})`);
      process.exit(1);
    }
  }
  const files = {
    '/page/reproduce.html': path.join(ROOT, 'test/speakers/page/reproduce.html'),
    '/files/model.onnx': paths[modelName],
    '/files/silero_vad.onnx': paths.silero,
  };
  const enrollUrl = (id, s) => `/e/${id}/${encodeURIComponent(s)}.wav`;
  for (const m of metas) {
    files[`/m/${m.id}.wav`] = path.join(paths.meetings, `${m.id}.wav`);
    files[`/v/${m.id}.f32`] = path.join(VAD_DIR, `${m.id}.f32`);
    for (const s of m.speakers) files[enrollUrl(m.id, s)] = enrollFile(paths.meetings, m.id, s);
  }
  const server = await startSpeakerServer({ isolated: true, files });
  const browser = await chromium.launch({ headless: true });
  const t0 = Date.now();
  try {
    const page = await (await browser.newContext()).newPage();
    await page.route('**/*', (route) => {
      const url = route.request().url();
      if (url.startsWith(`${server.origin}/`) || url.startsWith('blob:') || url.startsWith('data:')) return route.continue();
      outside.push(url);
      return route.abort();
    });
    page.on('pageerror', (e) => failures.push(`${modelName}: page error: ${e.message}`));
    await page.goto(`${server.origin}/page/reproduce.html`);
    await page.waitForFunction(() => window.reproduceReady);
    const info = await page.evaluate((a) => window.setup(a), {
      modelUrl: `${server.origin}/files/model.onnx`, pins, threads: 2, point: POINTS[modelName], sileroUrl: `${server.origin}/files/silero_vad.onnx` });
    console.log(`${modelName}: ${JSON.stringify(info)}`);

    // stored voices (once per enrolment recording)
    const prints = new Map();   // meeting id -> Map(name -> Float32Array)
    const failedEnroll = {};
    const memo = new Map();
    for (const m of metas) {
      const mine = new Map();
      for (const s of m.speakers) {
        const file = enrollFile(paths.meetings, m.id, s);
        if (!memo.has(file)) memo.set(file, await page.evaluate((u) => window.enrollPrint({ wavUrl: u }), `${server.origin}${enrollUrl(m.id, s)}`));
        const r = memo.get(file);
        if (r.vector) mine.set(s, Float32Array.from(r.vector));
        else failedEnroll[r.error] = (failedEnroll[r.error] || 0) + 1;
      }
      prints.set(m.id, mine);
    }
    console.log(`${modelName}: ${memo.size} enrolment recordings, refused: ${JSON.stringify(failedEnroll)}`);

    for (const level of level2 ? [1, 2] : [1]) {
      const all = [];
      const modes = {};
      let computeMs = 0;
      for (const m of metas) {
        const r = await page.evaluate((a) => window.embedMeeting(a), {
          wavUrl: `${server.origin}/m/${m.id}.wav`, vadUrl: level === 2 ? `${server.origin}/v/${m.id}.f32` : null });
        all.push({ meta: m, items: r.items.map((it) => ({ ...it, vector: Float32Array.from(it.vector) })) });
        for (const [k, v] of Object.entries(r.modes)) modes[k] = (modes[k] || 0) + v;
        computeMs += r.computeMs;
        console.log(`${modelName} level ${level}: ${m.id} ${r.utterances} utterances -> ${r.items.length} pieces (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
      }
      for (const mode of ['enrolled', 'anon']) {
        const total = { nScored: 0, nAll: 0, namedOk: 0, correct: 0, confusion: 0, uncertain: 0 };
        for (const { meta, items } of all) {
          const known = mode === 'enrolled' ? [...prints.get(meta.id)].map(([key, vector]) => ({ key, vector })) : [];
          const tagger = new Tagger({ point: POINTS[modelName], known });
          const results = items.map((it) => {
            const t = tagger.tag(it.vector, it.endMs - it.startMs);
            return { startMs: it.startMs, endMs: it.endMs, label: !t ? null : t.kind === 'known' ? { kind: 'named', name: t.key } : { kind: 'session', n: t.n } };
          });
          const s = scoreMeeting(results, meta.turns, new Set(known.map((k) => k.key)));
          for (const k of Object.keys(total)) total[k] += s[k];
        }
        const row = { pieces: total.nScored, sentences: total.nAll, namedOk: total.namedOk / total.nScored, namedOkCount: total.namedOk,
          confused: total.confusion, correct: total.correct / total.nScored, unsure: total.uncertain / total.nScored, modes };
        summary[`${modelName} level ${level} ${mode}`] = row;
        console.log(`${modelName} level ${level} ${mode}: pieces ${row.pieces}, named ok ${row.namedOk.toFixed(6)} (${row.namedOkCount}), confused ${row.confused}, correct ${row.correct.toFixed(4)}, unsure ${row.unsure.toFixed(4)}, modes ${JSON.stringify(modes)}`);
        if (level === 1 && mode === 'enrolled') {
          const a = ACCEPT[modelName];
          if (Math.abs(row.pieces / a.pieces - 1) > ACCEPT.piecesTolerance) failures.push(`${modelName}: ${row.pieces} pieces, measured ${a.pieces} (+-${ACCEPT.piecesTolerance * 100} %)`);
          if (Math.abs(row.namedOk - a.namedOk) > ACCEPT.namedOkTolerance) failures.push(`${modelName}: named ok ${row.namedOk.toFixed(6)}, measured ${a.namedOk} (+-${ACCEPT.namedOkTolerance})`);
          if (row.confused > a.maxConfused) failures.push(`${modelName}: ${row.confused} foreign names, at most ${a.maxConfused}`);
        }
      }
      summary[`${modelName} level ${level} computeMs`] = computeMs;
    }
    await page.evaluate(() => window.finish());
  } catch (err) {
    failures.push(`${modelName}: ${err.message}`);
  } finally {
    await browser.close();
    await server.close();
  }
  console.log(`${modelName}: ${((Date.now() - t0) / 60000).toFixed(1)} min`);
}
if (outside.length) failures.push(`request outside the local server: ${outside.join(', ')}`);
fs.writeFileSync(path.join(OUT, 'reproduce.json'), JSON.stringify({ accept: ACCEPT, summary, failures }, null, 1));
if (failures.length) { console.error(failures.join('\n')); process.exit(1); }
console.log('PASS: level 1 reproduces the measured name accuracy of both models within the tolerance');
