// scripts/fix-glue.mjs: the post-processing of the Emscripten glue that native/build-wasm.sh applies to every built
// module. The multi-threaded module must start its pthread workers from its own URL, not from a fixed file name that
// a bundler's renaming (Vite always, webpack by default) leaves pointing at nothing.
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { ROOT } from '../support/env.mjs';

const { fixGlue, fixedNameScripts } = await import('../../scripts/fix-glue.mjs');

// The two references Emscripten 6.0.13's minified ES-module glue has to itself (src/wasm/czechspeech-mt.js)
const MT = 'var _scriptName=import.meta.url;function findWasmBinary(){if(Module["locateFile"])return locateFile(' +
  '"czechspeech-mt.wasm");return new URL("czechspeech-mt.wasm",import.meta.url).href}' +
  'allocateUnusedWorker(){var worker;worker=new Worker(new URL("czechspeech-mt.js",import.meta.url),' +
  '{type:"module",name:"em-pthread"});PThread.unusedWorkers.push(worker);return worker}';
const ST = 'var _scriptName=import.meta.url;function findWasmBinary(){if(Module["locateFile"])return locateFile(' +
  '"czechspeech-st.wasm");return new URL("czechspeech-st.wasm",import.meta.url).href}';

test('the multi-threaded glue starts its pthread workers from its own URL; nothing else changes', () => {
  const r = fixGlue(MT, 'src/wasm/czechspeech-mt.js');
  assert.equal(r.changed, true);
  assert.equal(r.text, MT.replace('new URL("czechspeech-mt.js",import.meta.url)', 'new URL(import.meta.url)'));
  assert.deepEqual(fixedNameScripts(r.text), []);
  assert.match(r.text, /new Worker\(new URL\(import\.meta\.url\),\{type:"module",name:"em-pthread"\}\)/);
  assert.ok(r.text.includes('new URL("czechspeech-mt.wasm",import.meta.url)'), 'the .wasm fallback stays');
  // the same in Emscripten's unminified output
  const pretty = 'worker = new Worker(new URL(\'czechspeech-mt.js\', import.meta.url), { type: \'module\', name: \'em-pthread\' });';
  assert.equal(fixGlue(pretty, 'czechspeech-mt.js').text,
    'worker = new Worker(new URL(import.meta.url), { type: \'module\', name: \'em-pthread\' });');
});

test('fixGlue is idempotent and leaves a glue without threads alone', () => {
  const once = fixGlue(MT, 'czechspeech-mt.js').text;
  assert.deepEqual(fixGlue(once, 'czechspeech-mt.js'), { text: once, changed: false });
  assert.deepEqual(fixGlue(ST, 'czechspeech-st.js'), { text: ST, changed: false });
});

test('a glue it cannot make independent of its file name is an error, not a silent pass', () => {
  // a script of another name resolved next to the module
  assert.throws(() => fixGlue(MT.replace('"czechspeech-mt.js"', '"czechspeech-mt.worker.js"'), 'czechspeech-mt.js'),
    /czechspeech-mt\.js: loads other scripts by name: czechspeech-mt\.worker\.js/);
  // pthread workers created in a form it does not know
  assert.throws(() => fixGlue('worker=new Worker(pthreadUrl,{type:"module",name:"em-pthread"})', 'czechspeech-mt.js'),
    /czechspeech-mt\.js: its pthread workers are not started from new URL\(import\.meta\.url\)/);
});

test('only the module itself, in its own directory, is rewritten; the threaded glue must create pthread workers', () => {
  // ./czechspeech-mt.js is the module itself
  const dotted = MT.replace('"czechspeech-mt.js"', '"./czechspeech-mt.js"');
  assert.match(fixGlue(dotted, 'czechspeech-mt.js').text, /new Worker\(new URL\(import\.meta\.url\),\{type:"module",name:"em-pthread"\}\)/);
  // the same file name in another directory is another file
  for (const other of ['sub/czechspeech-mt.js', '../czechspeech-mt.js', './wasm/czechspeech-mt.js']) {
    assert.throws(() => fixGlue(MT.replace('"czechspeech-mt.js"', `"${other}"`), 'czechspeech-mt.js'),
      (e) => e.message === `czechspeech-mt.js: loads other scripts by name: ${other}`, other);
  }
  // a multi-threaded glue without pthread workers is not the module the build should have made
  assert.throws(() => fixGlue(ST.replace(/czechspeech-st/g, 'czechspeech-mt'), 'src/wasm/czechspeech-mt.js'),
    /czechspeech-mt\.js: the multi-threaded glue creates no pthread workers \(em-pthread\)/);
  assert.equal(fixGlue(ST, 'czechspeech-st.js').changed, false, 'the single-threaded glue needs none');
});

/** Runs `node scripts/fix-glue.mjs <args>` on copies of `files` ({ name: text }) in a temporary folder. */
function cli(args, files) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'czechspeech-glue-'));
  try {
    for (const [name, text] of Object.entries(files)) fs.writeFileSync(path.join(dir, name), text);
    const r = spawnSync(process.execPath, [path.join(ROOT, 'scripts/fix-glue.mjs'), ...args,
      ...Object.keys(files).map((n) => path.join(dir, n))], { encoding: 'utf8' });
    const after = Object.fromEntries(Object.keys(files).map((n) => [n, fs.readFileSync(path.join(dir, n), 'utf8')]));
    return { status: r.status, out: r.stdout + r.stderr, after };
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

test('the command fixes files in place, and with --check only reports', () => {
  const files = { 'czechspeech-mt.js': MT, 'czechspeech-st.js': ST };
  const check = cli(['--check'], files);
  assert.equal(check.status, 1);
  assert.match(check.out, /czechspeech-mt\.js: starts its pthread workers from a fixed file name/);
  assert.deepEqual(check.after, files, '--check changes nothing');
  const fix = cli([], files);
  assert.equal(fix.status, 0, fix.out);
  assert.equal(fix.after['czechspeech-st.js'], ST);
  assert.deepEqual(fixedNameScripts(fix.after['czechspeech-mt.js']), []);
  assert.equal(cli(['--check'], fix.after).status, 0, 'a fixed file passes --check');
  assert.equal(cli([], { 'czechspeech-mt.js': 'new Worker(x,{name:"em-pthread"})' }).status, 1, 'an error exits 1');
});

describe('static guards (no native build): the post-processing is wired into the build and the package check', () => {
  const build = fs.readFileSync(path.join(ROOT, 'native/build-wasm.sh'), 'utf8');
  test('native/build-wasm.sh fixes every browser glue after copying it and checks it', () => {
    assert.match(build, /cp "\$dir\/czechspeech-\$v\.js" "\$dir\/czechspeech-\$v\.wasm" "\$OUT\/"\n\s*node "\$ROOT\/scripts\/fix-glue\.mjs" "\$OUT\/czechspeech-\$v\.js" \|\| die/);
    assert.match(build, /node "\$ROOT\/scripts\/fix-glue\.mjs" --check "\$js" \|\| die/);
  });

  test('scripts/check-package.mjs refuses a glue that loads its threads by a fixed name', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'czechspeech-check-'));
    try {
      for (const f of ['scripts/check-package.mjs', 'scripts/fix-glue.mjs', 'package.json', 'src/index.js']) {
        fs.mkdirSync(path.dirname(path.join(dir, f)), { recursive: true });
        fs.copyFileSync(path.join(ROOT, f), path.join(dir, f));
      }
      fs.mkdirSync(path.join(dir, 'src/wasm'));
      const run = (mt) => {
        fs.writeFileSync(path.join(dir, 'src/wasm/czechspeech-mt.js'), mt);
        const r = spawnSync(process.execPath, [path.join(dir, 'scripts/check-package.mjs')], { encoding: 'utf8' });
        return r.stdout.split('\n').find((l) => l.includes('czechspeech-mt.js') && l.includes('own URL')) || '';
      };
      assert.match(run(MT), /^FAIL/);
      assert.match(run(fixGlue(MT, 'czechspeech-mt.js').text), /^ok/);
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});

test('the built browser glue is fixed (when it is built)', (t) => {
  const dir = path.join(ROOT, 'src/wasm');
  if (!fs.existsSync(path.join(dir, 'czechspeech-mt.js'))) {
    t.skip('src/wasm is not built (native/build-wasm.sh)');
    return;
  }
  for (const v of ['mt', 'st', 'webgpu']) {
    const file = path.join(dir, `czechspeech-${v}.js`);
    assert.equal(fixGlue(fs.readFileSync(file, 'utf8'), file).changed, false, `czechspeech-${v}.js needs fixing`);
  }
});
