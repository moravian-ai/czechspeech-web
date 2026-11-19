import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { ROOT } from '../support/env.mjs';
import { ORT_RUNTIME, ORT_MODULES, ORT_VERSION } from '../../src/speakers/runtime.js';

test('the vendored onnxruntime-web wasm is the pinned file', () => {
  const bytes = fs.readFileSync(path.join(ROOT, 'src', ORT_RUNTIME.path));
  assert.equal(ORT_VERSION, '1.31.1');
  assert.equal(bytes.length, ORT_RUNTIME.size);
  assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'), ORT_RUNTIME.sha256);
});

test('the vendored modules and the MIT licence are present and listed in package.json files', () => {
  const pkg = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8'));
  for (const rel of [...ORT_MODULES, ORT_RUNTIME.path, 'speakers/ort/LICENSE', 'speakers/ort/ThirdPartyNotices.txt']) {
    assert.ok(fs.existsSync(path.join(ROOT, 'src', rel)), rel);
    assert.ok(pkg.files.includes(`src/${rel}`), `package.json files: src/${rel}`);
  }
  assert.match(fs.readFileSync(path.join(ROOT, 'src/speakers/ort/LICENSE'), 'utf8'), /MIT License/);
  assert.match(fs.readFileSync(path.join(ROOT, 'THIRD_PARTY_NOTICES'), 'utf8'), /ONNX Runtime/);
});
