// Checks before `npm pack`: every file listed in package.json "files" exists, the WebAssembly modules are built (the
// WebGPU one with the patched Q4_0 kernel), the Silero VAD asset has the pinned hash, versions agree, and no shipped
// file contains a build-machine path.
// Usage: node scripts/check-package.mjs   (run by `npm pack` through the prepack script)
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { fixGlue } from './fix-glue.mjs';
import { changelogDateProblem } from './changelog-date.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const pkg = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'));
let failures = 0;
const check = (ok, message) => {
  console.log(`${ok ? 'ok  ' : 'FAIL'}  ${message}`);
  if (!ok) failures++;
};

for (const f of pkg.files) check(fs.existsSync(path.join(root, f)), `${f} exists${f.includes('wasm/') ? ' (native/build-wasm.sh)' : ''}`);

const vad = path.join(root, 'src/assets/ggml-silero-v6.2.0.bin');
if (fs.existsSync(vad)) {
  const sha = crypto.createHash('sha256').update(fs.readFileSync(vad)).digest('hex');
  check(sha === '2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987', `Silero VAD sha256 ${sha.slice(0, 12)}…`);
}

// The vendored onnxruntime-web wasm of the speaker worker has the pinned hash (src/speakers/runtime.js).
const ortRuntime = path.join(root, 'src/speakers/runtime.js');
if (fs.existsSync(ortRuntime)) {
  const { ORT_RUNTIME } = await import(pathToFileURL(ortRuntime).href);
  const ortWasm = path.join(root, 'src', ORT_RUNTIME.path);
  if (fs.existsSync(ortWasm)) {
    const sha = crypto.createHash('sha256').update(fs.readFileSync(ortWasm)).digest('hex');
    check(sha === ORT_RUNTIME.sha256, `onnxruntime-web wasm sha256 ${sha.slice(0, 12)}…`);
  }
}

// The WebGPU module is built with native/patches/ggml-webgpu-q4_0-mat-mat.patch; without it ggml's slower generic
// kernel runs instead, which nothing else would notice.
const webgpu = path.join(root, 'src/wasm/czechspeech-webgpu.wasm');
if (fs.existsSync(webgpu)) {
  check(fs.readFileSync(webgpu).includes('mul_mat_q4_0_tile'), 'src/wasm/czechspeech-webgpu.wasm has the patched Q4_0 kernel');
}

// The threaded module must be able to tear its runtime down (native/threads-teardown.js): terminate its pthread workers
// and wake its main thread's mailbox wait. Otherwise every return from the CPU to WebGPU after a WebGPU failure leaves
// a pool, or the module's shared memory, behind (a glue built before that teardown only had the first step).
const mtGlue = path.join(root, 'src/wasm/czechspeech-mt.js');
if (fs.existsSync(mtGlue)) {
  const glue = fs.readFileSync(mtGlue, 'utf8');
  check(glue.includes('czechspeechTerminateRuntime') && glue.includes('PThread.terminateRuntime()') && glue.includes('clearMailboxAwait'),
    'src/wasm/czechspeech-mt.js can tear its runtime down (pthread workers and the mailbox wait)');
}

// The browser glue must start its pthread workers from its own URL (scripts/fix-glue.mjs): bundlers rename it.
for (const v of ['mt', 'st', 'webgpu']) {
  const glue = path.join(root, `src/wasm/czechspeech-${v}.js`);
  if (!fs.existsSync(glue)) continue;
  let fixed;
  try {
    fixed = !fixGlue(fs.readFileSync(glue, 'utf8'), glue).changed;
  } catch {
    fixed = false;
  }
  check(fixed, `src/wasm/czechspeech-${v}.js loads its pthread workers from its own URL (node scripts/fix-glue.mjs)`);
}

const index = fs.readFileSync(path.join(root, 'src/index.js'), 'utf8');
check(index.includes(`VERSION = '${pkg.version}'`), `src/index.js VERSION is ${pkg.version}`);

// The release's CHANGELOG heading carries a real date (not the YYYY-MM-DD placeholder), so that a tarball packed before
// release day is refused.
const changelog = path.join(root, 'CHANGELOG.md');
if (fs.existsSync(changelog)) {
  const problem = changelogDateProblem(fs.readFileSync(changelog, 'utf8'), pkg.version);
  check(problem === null, problem ?? `CHANGELOG.md has a dated heading for ${pkg.version}`);
}

// No user names, home directories, drive letters or WSL mounts in what is published.
const user = os.userInfo().username.replace(/[^A-Za-z0-9_.-]/g, '');
const patterns = [/\/mnt\/[a-z]\//, /[A-Za-z]:[\\/]+Users[\\/]/, /\/home\/[a-z][a-z0-9_-]*\//,
  ...(user.length >= 3 ? [new RegExp(user, 'i')] : [])];
for (const f of pkg.files) {
  const p = path.join(root, f);
  if (!fs.existsSync(p) || fs.statSync(p).isDirectory()) continue;
  const text = fs.readFileSync(p).toString('latin1');
  const hit = patterns.find((re) => re.test(text));
  check(!hit, `${f}: no build-machine paths${hit ? ` (matches ${hit})` : ''}`);
}

if (failures) {
  console.error(`${failures} check(s) failed`);
  process.exit(1);
}
