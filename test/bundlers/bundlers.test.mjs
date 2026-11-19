// Bundler compatibility: a tiny app that imports the package is built with Vite 8 and with webpack 5 in a temporary
// folder. The bundle must contain the worker (bundled, as a chunk of its own) and reference it, and the worklet, the
// WebAssembly modules (Emscripten JS and .wasm) and the Silero VAD model must be emitted unchanged and referenced.
// Each emitted Emscripten module must find what it loads under the name the bundler gave it: no file is resolved next
// to it by a fixed name except its own .wasm, which the SDK always passes through locateFile.
// The speakers (src/speakers/engine.js, model.js and the worker behind them) are not exported from the package yet, so a
// second app imports their files by path: the speaker worker must be bundled, and onnxruntime's wasm and its Emscripten
// loader emitted unchanged and referenced (a bundler never sees a file that is only named by a directory).
// No model, no browser; heavier than the unit tests (npm run test:bundlers). The WebAssembly modules are built by
// native/build-wasm.sh and not in the repository: where they are missing, stand-ins with the same references to
// themselves take their place, post-processed as the build does (scripts/fix-glue.mjs).
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fixGlue } from '../../scripts/fix-glue.mjs';
import { ROOT } from '../support/env.mjs';

const pkg = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8'));
const NAME = pkg.name;
// What `new URL('./…', import.meta.url)` in src/speech.js points at, relative to src/
const ASSETS = ['worklet.js', 'wasm/czechspeech-mt.js', 'wasm/czechspeech-mt.wasm', 'wasm/czechspeech-st.js',
  'wasm/czechspeech-st.wasm', 'wasm/czechspeech-webgpu.js', 'wasm/czechspeech-webgpu.wasm', 'assets/ggml-silero-v6.2.0.bin'];
// What the speaker app must find in the output: the files of onnxruntime-web, by their path in the package
const ORT_ASSETS = ['speakers/ort/ort-wasm-simd-threaded.wasm', 'speakers/ort/ort-wasm-simd-threaded.mjs'];
// A string of src/speakers/worker.js: the chunk that has it is the bundled speaker worker
const SPEAKER_WORKER_MARKERS = ['runtime.slice(0, runtime.size'];
// Strings of src/worker-core.js and src/engine-wasm.js: the chunk that has them is the bundled worker
const WORKER_MARKERS = ['recogniser worker: unknown message', 'engine was wiped'];

let tmp;
let pkgDir;
const sources = new Map(); // asset (relative to src/) -> its bytes in the installed package

/**
 * A stand-in for a missing WebAssembly build product: distinct bytes per file (so that no bundler merges two), and
 * larger than the size below which bundlers inline an asset as a data: URL (Vite: 4 KiB), as the real files are. A
 * glue stand-in has the references that Emscripten's output (6.0.13, minified) has to itself: its .wasm and, in the
 * multi-threaded module, the script of its pthread workers.
 */
function standIn(file) {
  const padding = `// ${'-'.repeat(1000)}\n`.repeat(64);
  if (file.endsWith('.js')) {
    const name = path.basename(file);
    const wasm = name.replace(/\.js$/, '.wasm');
    const threads = name.includes('-mt') ? 'function allocateUnusedWorker(){var worker;' +
      `worker=new Worker(new URL("${name}",import.meta.url),{type:"module",name:"em-pthread"});return worker}\n` : '';
    return Buffer.from(`// stand-in for ${file} (native/build-wasm.sh)\n` +
      `function findWasmBinary(){if(Module["locateFile"])return locateFile("${wasm}");` +
      `return new URL("${wasm}",import.meta.url).href}\n${threads}` +
      `export default async function createModule() {\n  throw new Error('${file} is a stand-in');\n}\n${padding}`);
  }
  return Buffer.concat([Buffer.from([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]), Buffer.from(file + padding)]);
}

before(() => {
  tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'czechspeech-bundlers-'));
  pkgDir = path.join(tmp, 'node_modules', ...NAME.split('/'));
  for (const f of ['package.json', ...pkg.files]) {
    const from = path.join(ROOT, f);
    const to = path.join(pkgDir, f);
    fs.mkdirSync(path.dirname(to), { recursive: true });
    if (fs.existsSync(from)) fs.copyFileSync(from, to);
    else if (f.endsWith('.js') && f.startsWith('src/wasm/')) fs.writeFileSync(to, fixGlue(standIn(f).toString(), f).text);
    else if (f.startsWith('src/wasm/')) fs.writeFileSync(to, standIn(f));
    else throw new Error(`${f} is listed in package.json "files" but missing`);
  }
  for (const a of ASSETS) sources.set(a, fs.readFileSync(path.join(pkgDir, 'src', a)));
  fs.writeFileSync(path.join(tmp, 'package.json'), JSON.stringify({ name: 'app', private: true, type: 'module' }));
  fs.writeFileSync(path.join(tmp, 'main.js'),
    `import { CzechSpeech, CzechSpeechModel } from '${NAME}';\nglobalThis.app = { CzechSpeech, CzechSpeechModel };\n`);
  fs.writeFileSync(path.join(tmp, 'speakers.js'),
    `import { SpeakerEngine } from './node_modules/${NAME}/src/speakers/engine.js';\n` +
    `import { VoiceprintModel } from './node_modules/${NAME}/src/speakers/model.js';\n` +
    'globalThis.speakers = { SpeakerEngine, VoiceprintModel };\n');
  fs.writeFileSync(path.join(tmp, 'speakers.html'), '<!doctype html><script type="module" src="/speakers.js"></script>\n');
  fs.writeFileSync(path.join(tmp, 'index.html'), '<!doctype html><script type="module" src="/main.js"></script>\n');
});

// CZECHSPEECH_KEEP_BUNDLES=1 keeps the folder (its path is printed) to look at what the bundlers made
after(() => {
  if (process.env.CZECHSPEECH_KEEP_BUNDLES) console.log(`bundles kept in ${tmp}`);
  else fs.rmSync(tmp, { recursive: true, force: true });
});

function filesIn(dir) {
  const out = [];
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) out.push(...filesIn(p));
    else out.push(p);
  }
  return out;
}

/** What a module resolves next to itself by name: `new URL("<name>", import.meta.url)`. */
const resolvedByName = (code) => [...code.matchAll(/new URL\((["'])([^"']+)\1\s*,\s*import\.meta\.url\)/g)].map((m) => m[2]);

/**
 * An emitted Emscripten module (`asset`, emitted as `emitted`): the bundler renamed it, so a file it resolved next to
 * itself by a fixed name would not be found. The only name left may be its own .wasm (the fallback without
 * locateFile, which the SDK always passes); the multi-threaded module starts its pthread workers from its own URL.
 */
function checkGlue(asset, emitted, code) {
  const ownWasm = path.basename(asset).replace(/\.js$/, '.wasm');
  const others = resolvedByName(code).filter((n) => n !== ownWasm);
  assert.deepEqual(others, [], `${asset}, emitted as ${emitted}, loads files by a fixed name`);
  if (code.includes('em-pthread')) {
    assert.match(code, /new Worker\(new URL\(import\.meta\.url\)\s*,\s*\{[^}]*name:\s*"em-pthread"/,
      `${asset}: its pthread workers start from the module's own URL`);
  }
}

/**
 * The checks on one bundler's output: every asset emitted byte for byte, once, and its name in the code that refers
 * to it; every emitted Emscripten module independent of its file name (checkGlue); the worker bundled into a chunk of
 * its own, which the entry refers to (`workerRef(chunkFile)` gives what the entry code contains for it).
 */
function checkOutput(outDir, { entry, workerRef }) {
  const files = filesIn(outDir);
  const js = files.filter((f) => f.endsWith('.js'));
  const text = new Map(js.map((f) => [f, fs.readFileSync(f, 'utf8')]));
  const entryCode = text.get(entry);
  assert.ok(entryCode, `entry ${path.basename(entry)} not among ${files.map((f) => path.relative(outDir, f))}`);
  for (const [asset, bytes] of sources) {
    const copies = files.filter((f) => fs.readFileSync(f).equals(bytes));
    assert.equal(copies.length, 1, `${asset}: emitted ${copies.length} times unchanged (${files.map((f) => path.basename(f))})`);
    const name = path.basename(copies[0]);
    assert.ok(entryCode.includes(name), `${asset} emitted as ${name}, which the entry does not refer to`);
    if (asset.startsWith('wasm/') && asset.endsWith('.js')) checkGlue(asset, name, bytes.toString('utf8'));
  }
  const workers = js.filter((f) => WORKER_MARKERS.every((m) => text.get(f).includes(m)));
  assert.equal(workers.length, 1, `the worker bundled into ${workers.length} chunks`);
  const [worker] = workers;
  assert.notEqual(worker, entry, 'the worker has a chunk of its own');
  assert.ok(!/from\s*['"]\.\/worker-core\.js['"]/.test(text.get(worker)), 'the worker chunk is bundled (no import left)');
  assert.match(entryCode, /new Worker\(/);
  assert.ok(entryCode.includes(workerRef(worker)), `the entry does not refer to the worker chunk (${workerRef(worker)})`);
  assert.ok(!entryCode.includes("'./worker.js'") && !entryCode.includes('"./worker.js"'), 'an unresolved ./worker.js');
}

test('Vite 8 builds an app with the package: worker bundled, worklet, WebAssembly and VAD emitted and referenced', async () => {
  const { build, version } = await import('vite');
  assert.match(version, /^8\./);
  const outDir = path.join(tmp, 'dist-vite');
  const warnings = [];
  const logger = {
    info() {}, clearScreen() {}, hasErrorLogged: () => false, hasWarned: false,
    warn: (m) => warnings.push(m), warnOnce: (m) => warnings.push(m), error: (m) => warnings.push(m),
  };
  await build({ root: tmp, configFile: false, logLevel: 'warn', customLogger: logger,
    build: { outDir, emptyOutDir: true, minify: false } });
  assert.deepEqual(warnings.filter((w) => /czechspeech/i.test(w)), [], warnings.join('\n'));
  const entry = filesIn(outDir).find((f) => /[\\/]assets[\\/]index-[^\\/]*\.js$/.test(f));
  checkOutput(outDir, { entry, workerRef: (f) => path.basename(f) });
});

test('webpack 5 builds an app with the package: worker bundled, worklet, WebAssembly and VAD emitted and referenced', async () => {
  const { default: webpack } = await import('webpack');
  assert.match(webpack.version, /^5\./);
  const outDir = path.join(tmp, 'dist-webpack');
  const stats = await new Promise((resolve, reject) => {
    webpack({
      mode: 'production',
      context: tmp,
      entry: './main.js',
      // webpack's default asset names ([hash][ext]): no Emscripten module keeps its own file name
      output: { path: outDir, filename: 'main.js', chunkFilename: '[id].chunk.js', clean: true },
      optimization: { minimize: false },
      devtool: false,
      performance: { hints: false },
    }, (err, s) => (err ? reject(err) : resolve(s)));
  });
  const info = stats.toJson({ all: false, errors: true, warnings: true, chunks: true });
  assert.deepEqual(info.errors.map((e) => e.message), []);
  assert.deepEqual(info.warnings.map((w) => `${w.moduleName}: ${w.message}`), [], 'no warnings (e.g. "Critical dependency")');
  // webpack loads a worker chunk by its id: the entry refers to it as `__webpack_require__.u(<id>)`
  checkOutput(outDir, { entry: path.join(outDir, 'main.js'),
    workerRef: (f) => `__webpack_require__.u(${JSON.stringify(path.basename(f).replace(/\.chunk\.js$/, ''))}`.replace(/"(\d+)"/, '$1') });
});

/**
 * The speaker app's output: the speaker worker is a chunk of its own that the entry refers to, and the two onnxruntime
 * files are emitted once, byte for byte, and named in the code that loads them (the worker's chunk or the entry's: the
 * wasm is named by model.js in the entry, the loader by the worker).
 */
function checkSpeakerOutput(outDir, { entry }) {
  const files = filesIn(outDir);
  const js = files.filter((f) => f.endsWith('.js') || f.endsWith('.mjs'));
  const text = new Map(js.map((f) => [f, fs.readFileSync(f, 'utf8')]));
  const all = [...text.values()].join('\n');
  for (const asset of ORT_ASSETS) {
    const bytes = fs.readFileSync(path.join(pkgDir, 'src', asset));
    const copies = files.filter((f) => fs.statSync(f).size === bytes.length && fs.readFileSync(f).equals(bytes));
    assert.equal(copies.length, 1, `${asset}: emitted ${copies.length} times unchanged (${files.map((f) => path.basename(f))})`);
    assert.ok(all.includes(path.basename(copies[0])), `${asset} emitted as ${path.basename(copies[0])}, which no code refers to`);
  }
  const workers = js.filter((f) => SPEAKER_WORKER_MARKERS.every((m) => text.get(f).includes(m)));
  assert.equal(workers.length, 1, `the speaker worker bundled into ${workers.length} chunks`);
  assert.notEqual(workers[0], entry, 'the speaker worker has a chunk of its own');
  assert.ok(!/from\s*['"]\.\/worker-core\.js['"]/.test(text.get(workers[0])), 'the speaker worker chunk is bundled (no import left)');
  assert.ok(!all.includes("'./ort/'") && !all.includes('"./ort/"'), 'a directory URL left in the output');
}

test('Vite 8 builds an app with the speakers: worker bundled, onnxruntime wasm and loader emitted and referenced', async () => {
  const { build } = await import('vite');
  const outDir = path.join(tmp, 'dist-vite-speakers');
  const warnings = [];
  const logger = {
    info() {}, clearScreen() {}, hasErrorLogged: () => false, hasWarned: false,
    warn: (m) => warnings.push(m), warnOnce: (m) => warnings.push(m), error: (m) => warnings.push(m),
  };
  await build({ root: tmp, configFile: false, logLevel: 'warn', customLogger: logger,
    build: { outDir, emptyOutDir: true, minify: false, rollupOptions: { input: path.join(tmp, 'speakers.html') } } });
  assert.deepEqual(warnings.filter((w) => /czechspeech|speakers/i.test(w)), [], warnings.join('\n'));
  const entry = filesIn(outDir).find((f) => /[\\/]assets[\\/]speakers-[^\\/]*\.js$/.test(f));
  assert.ok(entry, 'the speakers entry is not in the output');
  checkSpeakerOutput(outDir, { entry });
});

test('webpack 5 builds an app with the speakers: worker bundled, onnxruntime wasm and loader emitted and referenced', async () => {
  const { default: webpack } = await import('webpack');
  const outDir = path.join(tmp, 'dist-webpack-speakers');
  const stats = await new Promise((resolve, reject) => {
    webpack({
      mode: 'production', context: tmp, entry: './speakers.js',
      output: { path: outDir, filename: 'speakers-main.js', chunkFilename: '[id].chunk.js', clean: true },
      optimization: { minimize: false }, devtool: false, performance: { hints: false },
    }, (err, s) => (err ? reject(err) : resolve(s)));
  });
  const info = stats.toJson({ all: false, errors: true, warnings: true });
  assert.deepEqual(info.errors.map((e) => e.message), []);
  assert.deepEqual(info.warnings.map((w) => `${w.moduleName}: ${w.message}`), [], 'no warnings (e.g. "Critical dependency")');
  checkSpeakerOutput(outDir, { entry: path.join(outDir, 'speakers-main.js') });
});
