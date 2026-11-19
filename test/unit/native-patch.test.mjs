// The ggml patch of the WebGPU module (c58c0cb: a Q4_0 x F32 matrix multiplication, 3.99 s -> 0.47 s of GPU time
// per utterance). A native build takes Emscripten and minutes of CPU, so these tests check what can break without
// one: the patch is there, well-formed and pinned to the whisper.cpp commit the build uses; the build applies it to the
// webgpu variant only and checks that the kernel is in the module; the package check refuses a WebGPU module without
// it. Where the submodule or the built module are present, the patch is applied (--check) and the module checked.
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { ROOT } from '../support/env.mjs';

const PATCH = path.join(ROOT, 'native/patches/ggml-webgpu-q4_0-mat-mat.patch');
const BUILD = fs.readFileSync(path.join(ROOT, 'native/build-wasm.sh'), 'utf8');
const KERNEL = 'mul_mat_q4_0_tile';
const WHISPER_COMMIT = /^WHISPER_COMMIT=([0-9a-f]{40})$/m.exec(BUILD)[1];

/** The files of a unified diff, with their hunks' added lines; throws when a hunk's line counts are wrong. */
function parsePatch(text) {
  const lines = text.split('\n');
  const files = [];
  for (let i = 0; i < lines.length; i++) {
    if (!lines[i].startsWith('+++ ')) continue;
    const file = { name: lines[i].slice(4).replace(/^b\//, '').trim(), added: [] };
    files.push(file);
    for (i++; i < lines.length && lines[i].startsWith('@@'); ) {
      const m = /^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@/.exec(lines[i]);
      assert.ok(m, `${file.name}: bad hunk header ${lines[i]}`);
      let old = Number(m[1] ?? 1);
      let neu = Number(m[2] ?? 1);
      for (i++; old > 0 || neu > 0; i++) {
        const l = lines[i];
        assert.ok(l !== undefined, `${file.name}: hunk ends early`);
        if (l.startsWith('+')) { neu--; file.added.push(l.slice(1)); } else if (l.startsWith('-')) old--;
        else if (l.startsWith(' ') || l === '') { old--; neu--; } else assert.fail(`${file.name}: unexpected line in hunk: ${l}`);
      }
      assert.ok(old === 0 && neu === 0, `${file.name}: hunk line counts do not match`);
    }
    i--;
  }
  return files;
}

/** Runs a copy of scripts/check-package.mjs in a folder with `webgpuWasm` as src/wasm/czechspeech-webgpu.wasm. */
function checkPackage(webgpuWasm) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'czechspeech-check-'));
  try {
    fs.mkdirSync(path.join(dir, 'scripts'));
    fs.mkdirSync(path.join(dir, 'src/wasm'), { recursive: true });
    fs.copyFileSync(path.join(ROOT, 'scripts/check-package.mjs'), path.join(dir, 'scripts/check-package.mjs'));
    fs.copyFileSync(path.join(ROOT, 'scripts/fix-glue.mjs'), path.join(dir, 'scripts/fix-glue.mjs'));
    fs.copyFileSync(path.join(ROOT, 'scripts/changelog-date.mjs'), path.join(dir, 'scripts/changelog-date.mjs'));
    fs.copyFileSync(path.join(ROOT, 'package.json'), path.join(dir, 'package.json'));
    fs.copyFileSync(path.join(ROOT, 'src/index.js'), path.join(dir, 'src/index.js'));
    fs.writeFileSync(path.join(dir, 'src/wasm/czechspeech-webgpu.wasm'), webgpuWasm);
    const r = spawnSync(process.execPath, [path.join(dir, 'scripts/check-package.mjs')], { encoding: 'utf8' });
    return r.stdout.split('\n').find((l) => l.includes('czechspeech-webgpu.wasm') && l.includes('Q4_0')) || '';
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

describe('static guards (no native build): the patch, the build script, the package check', () => {
  test('the Q4_0 patch is well-formed, adds the kernel and uses it, and is pinned to the build\'s whisper.cpp', () => {
    // regression (c58c0cb): the patch is the whole fix; without it the module silently uses ggml's slower kernel
    assert.ok(fs.existsSync(PATCH), 'native/patches/ggml-webgpu-q4_0-mat-mat.patch');
    const text = fs.readFileSync(PATCH, 'utf8');
    assert.match(text, new RegExp(`^Applies to: whisper\\.cpp ${WHISPER_COMMIT}`, 'm'), 'the commit native/build-wasm.sh pins');
    const files = parsePatch(text);
    assert.deepEqual(files.map((f) => f.name).sort(), ['ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      `ggml/src/ggml-webgpu/wgsl-shaders/${KERNEL}.wgsl`]);
    const shader = files.find((f) => f.name.endsWith('.wgsl')).added.join('\n');
    assert.match(shader, /^enable f16;/m);
    assert.match(shader, /@compute @workgroup_size\(/);
    const lib = files.find((f) => f.name.endsWith('.hpp')).added.join('\n');
    assert.match(lib, new RegExp(`ggml_webgpu_create_pipeline\\(device, processed, "${KERNEL}"\\)`));
    assert.match(lib, /src0->type == GGML_TYPE_Q4_0 && context\.src1->type == GGML_TYPE_F32/);
    assert.match(lib, /return get_mul_mat_q4_0_tile_pipeline\(\);/);
    // the submodule the repository pins is the same commit
    const pinned = execFileSync('git', ['rev-parse', 'HEAD:third_party/whisper.cpp'], { cwd: ROOT, encoding: 'utf8' }).trim();
    assert.equal(pinned, WHISPER_COMMIT);
  });

  test('native/build-wasm.sh applies the patches to the webgpu variant only and checks the kernel is in its module', () => {
    const prepare = /prepare_webgpu_sources\(\) \{([\s\S]*?)\n\}/.exec(BUILD);
    assert.ok(prepare, 'prepare_webgpu_sources()');
    assert.match(prepare[1], /for p in "\$ROOT"\/native\/patches\/ggml-webgpu-\*\.patch; do/);
    assert.match(prepare[1], /patch -d "\$WEBGPU_SRC" -p1 --forward --batch --quiet < "\$p"[^\n]*\|\| die/);
    const calls = [...BUILD.matchAll(/^\s*prepare_webgpu_sources$/gm)];
    assert.equal(calls.length, 1, 'called once');
    const branch = /if \[ "\$v" = webgpu \]; then\n([\s\S]*?)\n\s*fi\n/.exec(BUILD);
    assert.ok(branch && /prepare_webgpu_sources/.test(branch[1]) && /-DWHISPER_CPP_DIR="\$WEBGPU_SRC"/.test(branch[1]),
      'the webgpu variant (only) is built from the patched copy');
    assert.match(BUILD, new RegExp(`grep -qa "${KERNEL}" "\\$OUT/czechspeech-\\$v\\.wasm" \\|\\| die`));
  });

  test('the package check refuses a WebGPU module without the patched kernel', () => {
    const header = Buffer.from([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);
    assert.match(checkPackage(Buffer.concat([header, Buffer.from('mul_mat_reg_tile')])), /^FAIL/);
    assert.match(checkPackage(Buffer.concat([header, Buffer.from(`...${KERNEL}...`)])), /^ok/);
  });
});

test('the patch applies to the pinned whisper.cpp (when the submodule is checked out)', (t) => {
  const dir = path.join(ROOT, 'third_party/whisper.cpp');
  if (!fs.existsSync(path.join(dir, 'include/whisper.h'))) {
    t.skip('third_party/whisper.cpp is not checked out (git submodule update --init)');
    return;
  }
  const head = execFileSync('git', ['-C', dir, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
  assert.equal(head, WHISPER_COMMIT);
  execFileSync('git', ['-C', dir, 'apply', '--check', PATCH]);
});

test('the built WebGPU module contains the patched kernel (when it is built)', (t) => {
  const wasm = path.join(ROOT, 'src/wasm/czechspeech-webgpu.wasm');
  if (!fs.existsSync(wasm)) {
    t.skip('src/wasm/czechspeech-webgpu.wasm is not built (native/build-wasm.sh webgpu)');
    return;
  }
  assert.ok(fs.readFileSync(wasm).includes(KERNEL));
});
