// Browser end-to-end test of the multi-threaded CPU module in a bundled app (no model, a few seconds). A small app that
// imports the package is built with Vite, which renames every asset with a hash, and opened cross-origin isolated in
// headless Chromium. transcribe() of one second of silence starts the recogniser on the threaded module: the module
// starts only once its pool of pthread workers (threads + 1, native/CMakeLists.txt) has loaded, and the speech gate
// then finds no speech, so the model is never loaded. The pthread workers must load the module's own, renamed file
// (scripts/fix-glue.mjs). A control app built with the glue as Emscripten emits it (pthread workers from
// "czechspeech-mt.js") must fail: that shows the test sees the defect.
//
// Needs the built browser modules in src/wasm/ (native/build-wasm.sh) and Chromium (npx playwright-core install chromium).
// Usage: node test/e2e/threads.mjs
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { chromium } from 'playwright-core';
import { startServer } from '../../scripts/serve.mjs';
import { fixGlue } from '../../scripts/fix-glue.mjs';
import { ROOT } from '../support/env.mjs';

const THREADS = 4;
const TIMEOUT_MS = 30000;
const glue = path.join(ROOT, 'src/wasm/czechspeech-mt.js');
if (!fs.existsSync(glue)) {
  console.log('SKIP: src/wasm/czechspeech-mt.js is not built (native/build-wasm.sh)');
  process.exit(0);
}

let failures = 0;
const check = (ok, message) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${message}`);
  if (!ok) failures++;
};

const pkg = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8'));
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'czechspeech-threads-'));

/** Builds the app with Vite into <tmp>/<name>/dist (served at /<name>/); `mtGlue` replaces the module's JS. */
async function buildApp(name, mtGlue) {
  const dir = path.join(tmp, name);
  const pkgDir = path.join(dir, 'node_modules', ...pkg.name.split('/'));
  for (const f of ['package.json', ...pkg.files]) {
    fs.mkdirSync(path.dirname(path.join(pkgDir, f)), { recursive: true });
    fs.copyFileSync(path.join(ROOT, f), path.join(pkgDir, f));
  }
  fs.writeFileSync(path.join(pkgDir, 'src/wasm/czechspeech-mt.js'), mtGlue);
  fs.writeFileSync(path.join(dir, 'package.json'), JSON.stringify({ name, private: true, type: 'module' }));
  fs.writeFileSync(path.join(dir, 'index.html'), '<!doctype html><script type="module" src="./main.js"></script>\n');
  fs.writeFileSync(path.join(dir, 'main.js'), `import { CzechSpeech, CzechSpeechModel } from '${pkg.name}';
window.run = async () => {
  const logs = [];
  const model = await CzechSpeechModel.fromBytes(new Uint8Array(1024)); // never loaded: silence has no speech
  const speech = await CzechSpeech.create({ model, config: { backend: 'cpu', threads: ${THREADS} },
    log: (level, message) => logs.push(level + ': ' + message) });
  const t0 = performance.now();
  try {
    const text = await Promise.race([speech.transcribe(new Float32Array(16000)),
      new Promise((_, reject) => setTimeout(() => reject(new Error('no answer in ${TIMEOUT_MS / 1000} s')), ${TIMEOUT_MS}))]);
    return { ok: true, text, ms: Math.round(performance.now() - t0), state: speech.state, logs, isolated: crossOriginIsolated };
  } catch (e) {
    return { ok: false, error: String(e && e.message || e), code: e && e.code, ms: Math.round(performance.now() - t0),
      state: speech.state, logs, isolated: crossOriginIsolated };
  } finally {
    speech.close();
  }
};
window.ready = true;
`);
  const { build } = await import('vite');
  await build({ root: dir, base: `/${name}/`, configFile: false, logLevel: 'warn',
    build: { outDir: path.join(dir, 'dist'), emptyOutDir: true, minify: false } });
  const assets = fs.readdirSync(path.join(dir, 'dist/assets'));
  return { dist: path.join(dir, 'dist'), mtGlue: assets.find((f) => /^czechspeech-mt-.*\.js$/.test(f)) };
}

const fixedText = fs.readFileSync(glue, 'utf8');
check(!fixGlue(fixedText, glue).changed, 'src/wasm/czechspeech-mt.js is post-processed (scripts/fix-glue.mjs)');
const emscriptenText = fixedText.replace('new Worker(new URL(import.meta.url),', 'new Worker(new URL("czechspeech-mt.js",import.meta.url),');
check(emscriptenText !== fixedText, 'the control glue has the fixed-name pthread worker of Emscripten\'s output');
const app = await buildApp('app', fixedText);
const control = await buildApp('control', emscriptenText);
console.log(`      built with Vite: the threaded module emitted as ${app.mtGlue} (control: ${control.mtGlue})`);

const server = await startServer({ port: 0, quiet: true, mounts: [{ prefix: '/app/', dir: app.dist }, { prefix: '/control/', dir: control.dist }] });
const origin = `http://127.0.0.1:${server.port}`;
// Every request as the server answered it: the pthread workers' script loads come from nested workers, which the
// browser automation may not report.
const served = [];
server.server.on('request', (req, res) => {
  const p = new URL(req.url, origin).pathname;
  res.on('finish', () => served.push({ path: p, status: res.statusCode }));
});
const browser = await chromium.launch({ headless: true });
try {
  for (const [name, expectStart] of [['app', true], ['control', false]]) {
    const context = await browser.newContext();
    served.length = 0;
    const page = await context.newPage();
    const workers = [];
    page.on('worker', (w) => workers.push(new URL(w.url()).pathname));
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.goto(`${origin}/${name}/index.html`);
    await page.waitForFunction(() => window.ready);
    const r = await page.evaluate(() => window.run());
    await new Promise((resolve) => setTimeout(resolve, 200)); // the last responses finish
    const mt = `/${name}/assets/${name === 'app' ? app.mtGlue : control.mtGlue}`;
    const glueLoads = served.filter((x) => x.path === mt && x.status === 200).length;
    const failed = served.filter((x) => x.status >= 400 && !x.path.endsWith('/favicon.ico')).map((x) => `${x.status} ${x.path}`);
    console.log(`      ${name}: ${JSON.stringify({ ok: r.ok, text: r.text, error: r.error, code: r.code, ms: r.ms,
      runtime: r.state.runtime, threads: r.state.threads, modelLoads: r.state.modelLoads, isolated: r.isolated })}`);
    console.log(`      ${name}: requests of ${path.basename(mt)}: ${glueLoads}; failed responses: ${failed.join(', ') || 'none'}; ` +
      `workers reported by the browser: ${workers.length}`);
    if (expectStart) {
      check(r.isolated, `${name}: the page is cross-origin isolated`);
      check(r.ok && r.text === '', `${name}: transcribe() of silence on the threaded module answered '' in ${r.ms} ms`);
      check(r.state.runtime === 'wasm-threads' && r.state.threads === THREADS,
        `${name}: runtime ${r.state.runtime}, ${r.state.threads} recogniser threads`);
      check(r.state.modelLoads === 0, `${name}: the model was not loaded (no speech)`);
      check(glueLoads >= THREADS + 2, `${name}: the renamed module was loaded ${glueLoads} times (the recogniser and ` +
        `${THREADS + 1} pthread workers)`);
      check(failed.length === 0, `${name}: no failed request`);
      check(errors.length === 0, `${name}: no page error ${errors.join('; ')}`);
    } else {
      check(!r.ok, `${name}: with the glue as Emscripten emits it, the threaded module does not start (${r.error})`);
      check(failed.some((f) => f.endsWith('/assets/czechspeech-mt.js')),
        `${name}: its pthread workers asked for the name it no longer has (${failed.join(', ')})`);
    }
    await context.close();
  }
} finally {
  await browser.close();
  await server.close();
  fs.rmSync(tmp, { recursive: true, force: true });
}
console.log(failures ? `${failures} check(s) failed` : 'all checks passed');
process.exit(failures ? 1 : 0);
