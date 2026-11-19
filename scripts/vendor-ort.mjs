// Copies the three onnxruntime-web files the speaker worker needs from node_modules into src/speakers/ort/ and prints
// the size and sha256 of the wasm (pinned in src/speakers/runtime.js). The package ships neither LICENSE nor
// ThirdPartyNotices.txt, so those two files in src/speakers/ort/ are taken from the upstream tag v1.31.1 by hand
// (https://github.com/microsoft/onnxruntime/tree/v1.31.1); this script keeps them if they are there and copies them from
// the package root when a later version ships them.
// Usage: node scripts/vendor-ort.mjs
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const pkgDir = path.join(root, 'node_modules', 'onnxruntime-web');
const out = path.join(root, 'src', 'speakers', 'ort');

const version = JSON.parse(fs.readFileSync(path.join(pkgDir, 'package.json'), 'utf8')).version;
if (version !== '1.31.1') {
  console.error(`onnxruntime-web ${version} is installed, 1.31.1 is pinned`);
  process.exit(1);
}
fs.mkdirSync(out, { recursive: true });
for (const f of ['ort.wasm.min.mjs', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']) {
  fs.copyFileSync(path.join(pkgDir, 'dist', f), path.join(out, f));
}
for (const f of ['LICENSE', 'ThirdPartyNotices.txt']) {
  const from = path.join(pkgDir, f);
  if (fs.existsSync(from)) fs.copyFileSync(from, path.join(out, f));
  else if (!fs.existsSync(path.join(out, f))) console.warn(`${f}: not in the package and not in ${out}; take it from the upstream tag`);
}
const wasm = fs.readFileSync(path.join(out, 'ort-wasm-simd-threaded.wasm'));
console.log(`ort-wasm-simd-threaded.wasm size=${wasm.length} sha256=${crypto.createHash('sha256').update(wasm).digest('hex')}`);
