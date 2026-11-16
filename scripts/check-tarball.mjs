// Packs the package (npm pack), installs the tarball into a fresh consumer project in a temporary directory and
// checks it the way an app would use it: the ES module imports in Node.js, the TypeScript declarations resolve and
// type-check a consumer, and no packed file contains a build-machine path.
// Usage: node scripts/check-tarball.mjs [--keep]
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';
// npm.cmd needs a shell on Windows; node itself must run without one (its path may contain spaces).
const run = (cmd, args, cwd) => execFileSync(cmd, args, { cwd, encoding: 'utf8', shell: cmd === npm && process.platform === 'win32' });
let failures = 0;
const check = (ok, message) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${message}`);
  if (!ok) failures++;
};

// The prepack checks, run here so that their output does not mix with `npm pack --json`.
run(process.execPath, ['scripts/build-worklet.mjs', '--check'], root);
run(process.execPath, ['scripts/check-package.mjs'], root);
check(true, 'prepack checks (worklet up to date, files present, VAD hash, versions, no machine paths)');
const packed = JSON.parse(run(npm, ['pack', '--json', '--ignore-scripts', '--pack-destination', root], root))[0];
const tarball = path.join(root, packed.filename);
console.log(`${packed.filename}: ${packed.size} bytes packed, ${packed.unpackedSize} unpacked, ${packed.files.length} files`);
for (const f of packed.files) console.log(`  ${String(f.size).padStart(9)}  ${f.path}`);

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'czechspeech-consumer-'));
try {
  fs.writeFileSync(path.join(dir, 'package.json'), JSON.stringify({ name: 'consumer', private: true, type: 'module' }));
  run(npm, ['install', '--no-audit', '--no-fund', '--ignore-scripts', tarball], dir);
  const installed = path.join(dir, 'node_modules/@moravianai/czechspeech');
  const modules = ['mt', 'st', 'webgpu'].map((v) => `src/wasm/czechspeech-${v}.wasm`);
  check(modules.every((m) => fs.existsSync(path.join(installed, m))), `installed package contains the WebAssembly modules (${modules.join(', ')})`);
  // a module built before the language support lacks the export, and the engine then refuses to start (src/engine-wasm.js)
  const glue = modules.map((m) => m.replace(/\.wasm$/, '.js'));
  check(glue.every((g) => fs.readFileSync(path.join(installed, g), 'utf8').includes('_cs_whisper_language')),
    `the WebAssembly glue exports _cs_whisper_language (${glue.join(', ')}); rebuild with native/build-wasm.sh if not`);

  // No build-machine paths in any packed file.
  const user = os.userInfo().username.replace(/[^A-Za-z0-9_.-]/g, '');
  const patterns = [/\/mnt\/[a-z]\//, /[A-Za-z]:\\\\?Users\\/, /\/home\/[a-z][a-z0-9_-]*\//,
    ...(user.length >= 3 ? [new RegExp(user, 'i')] : [])];
  const bad = [];
  for (const f of packed.files) {
    const text = fs.readFileSync(path.join(installed, f.path)).toString('latin1');
    const hit = patterns.find((re) => re.test(text));
    if (hit) bad.push(`${f.path} (${hit})`);
  }
  check(bad.length === 0, `no build-machine paths in the ${packed.files.length} packed files${bad.length ? `: ${bad.join(', ')}` : ''}`);

  // ESM consumer
  fs.writeFileSync(path.join(dir, 'consumer.mjs'), `
import { CzechSpeech, CzechSpeechModel, CzechSpeechError, DEFAULT_CONFIG, DEFAULT_MODEL, DEFAULT_AUDIO_CONSTRAINTS,
  TranscriptEvent, VERSION } from '@moravianai/czechspeech';
import pkg from '@moravianai/czechspeech/package.json' with { type: 'json' };
const ok = typeof CzechSpeech.create === 'function' && typeof CzechSpeechModel.download === 'function'
  && DEFAULT_MODEL.sha256.startsWith('04ead666') && DEFAULT_CONFIG.threshold === 0.25 && DEFAULT_CONFIG.backend === 'auto' && VERSION === pkg.version
  && DEFAULT_AUDIO_CONSTRAINTS.noiseSuppression === false && new CzechSpeechError('x', { code: 'model' }).code === 'model'
  && !String(new TranscriptEvent('ahoj', 0, 1, 2)).includes('ahoj');
console.log(ok ? 'esm-ok ' + VERSION : 'esm-bad');
`);
  const out = run(process.execPath, ['consumer.mjs'], dir).trim();
  check(out === `esm-ok ${JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8')).version}`,
    `an ES module consumer imports the package (${out})`);

  // TypeScript consumer (types resolve through "exports")
  fs.writeFileSync(path.join(dir, 'consumer.ts'), `
import { CzechSpeech, CzechSpeechModel, type SpeechEvent, type TranscriptEvent, type CzechSpeechConfig, type CzechSpeechBackend,
  type GpuAdapterInfo } from '@moravianai/czechspeech';

export async function demo(signal: AbortSignal): Promise<string[]> {
  const model: CzechSpeechModel = await CzechSpeechModel.download({ onProgress: (done: number, total: number) => void (done / total) });
  const backend: CzechSpeechBackend = 'auto';
  const config: Partial<CzechSpeechConfig> = { minSilenceMs: 800, backend };
  const speech = await CzechSpeech.create({ model, config, keepModelLoaded: false, log: false });
  const texts: string[] = [];
  for await (const ev of speech.listen({ signal, audio: { autoGainControl: true } })) {
    const e: SpeechEvent = ev;
    if (e.type === 'transcript') texts.push(e.text, String(e.startMs + e.endMs + e.processingMs));
    else if (e.type === 'error') texts.push(e.error.code);
    else texts.push(String(e.atMs));
  }
  for await (const t of speech.transcripts()) { const x: TranscriptEvent = t; texts.push(x.text); speech.stop(); }
  const nothing: void = speech.stop();
  void nothing;
  texts.push(await speech.transcribe(new Float32Array(16000)));
  const loaded: boolean = speech.state.modelLoaded;
  const where: 'webgpu' | 'cpu' | null = speech.state.backend;
  const adapter: GpuAdapterInfo | null = speech.state.gpuAdapter;
  texts.push(String(where), adapter ? adapter.vendor : '', speech.state.backendFallback ?? '', String(speech.state.peakGpuBytes));
  await speech.close();
  const cached: boolean = await CzechSpeechModel.isDownloaded();
  return loaded && cached ? texts : [];
}
`);
  fs.writeFileSync(path.join(dir, 'tsconfig.json'), JSON.stringify({
    compilerOptions: { target: 'es2022', module: 'nodenext', moduleResolution: 'nodenext', strict: true, noEmit: true,
      lib: ['es2022', 'dom', 'dom.iterable'], skipLibCheck: false, types: [] },
    files: ['consumer.ts'],
  }));
  let tscOut = '';
  let tscOk = true;
  try {
    tscOut = run(process.execPath, [path.join(root, 'node_modules/typescript/bin/tsc'), '-p', 'tsconfig.json'], dir);
  } catch (e) {
    tscOk = false;
    tscOut = `${e.stdout || ''}${e.stderr || ''}`;
  }
  check(tscOk, `the TypeScript declarations resolve and type-check a consumer${tscOk ? '' : `:\n${tscOut}`}`);
} finally {
  if (!process.argv.includes('--keep')) fs.rmSync(dir, { recursive: true, force: true });
}
console.log(`tarball: ${tarball}`);
console.log(failures ? `${failures} FAILED` : 'ALL PASSED');
process.exit(failures ? 1 : 0);
