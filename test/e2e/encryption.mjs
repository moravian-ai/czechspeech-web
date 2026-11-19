// Real-browser check of the decryption of encrypted model files (AES-256-CTR through the browser's own crypto.subtle):
// the NIST SP 800-38A vector, a 1 MiB file decrypted at offsets 0, 1, 15, 16, 17 and in odd-sized chunks, for an IV
// whose counter carries over the low 64 bits and one that wraps all 128 bits, and a download through fromUrl() into the
// browser's Cache Storage. Needs no model, no GPU and no network beyond the local server; headless Chromium
// (`npx playwright-core install chromium`).
// Usage: node test/e2e/encryption.mjs
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { chromium } from 'playwright-core';
import { startServer } from '../../scripts/serve.mjs';

const KEY = '603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4';
const NIST_IV = 'f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff';
const NIST_PLAIN = '6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e5130c81c46a35ce411e5fbc1191a0a52eff69f2445df4f9b17ad2b417be66c3710';
const NIST_CIPHER = '601ec313775789a5b7a7f504bbf3d228f443e3ca4d62b59aca84e990cacaf5c52b0930daa23de94ce87017ba2d84988ddfc9c58db67aada613c2dd08457941a6';
const IVS = [NIST_IV, `${'00'.repeat(8)}${'ff'.repeat(8)}`, 'ff'.repeat(16)];

const plain = crypto.randomBytes(1024 * 1024);
const cipherFor = (iv) => {
  const c = crypto.createCipheriv('aes-256-ctr', Buffer.from(KEY, 'hex'), Buffer.from(iv, 'hex'));
  return Buffer.concat([c.update(plain), c.final()]);
};
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'czechspeech-enc-'));
fs.writeFileSync(path.join(dir, 'm.bin.enc'), cipherFor(IVS[1]));
const sha = crypto.createHash('sha256').update(plain).digest('hex');

const server = await startServer({ port: 0, quiet: true, mounts: [{ prefix: '/data/', dir }] });
const origin = `http://127.0.0.1:${server.port}`;
const browser = await chromium.launch({ headless: true });
let failed = 0;
try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.log(`page error: ${e.message}`));
  await page.goto(`${origin}/test/e2e/encryption.html`);
  await page.waitForFunction(() => window.encReady === true);
  const ok = async (name, fn) => {
    const r = await fn();
    console.log(`${r === true ? 'ok  ' : 'FAIL'}  ${name}${r === true ? '' : ` (${r})`}`);
    if (r !== true) failed++;
  };
  await ok('crypto.subtle is available (secure context)', () => page.evaluate(() => !!(globalThis.crypto && crypto.subtle)));
  await ok('NIST SP 800-38A F.5.5 vector', () => page.evaluate(async ([key, iv, c, p]) => {
    const hex = (u) => [...u].map((b) => b.toString(16).padStart(2, '0')).join('');
    const out = await window.enc.createDecryptor({ algorithm: 'AES-256-CTR', key, iv })(Uint8Array.from(c.match(/../g), (h) => parseInt(h, 16)), 0);
    return hex(out) === p || 'wrong plaintext';
  }, [KEY, NIST_IV, NIST_CIPHER, NIST_PLAIN]));
  for (const iv of IVS) {
    const cipher = cipherFor(iv).toString('base64');
    await ok(`1 MiB at offsets 0, 1, 15, 16, 17 in odd chunks, IV ${iv.slice(0, 16)}…`, () => page.evaluate(async ([key, ivHex, b64, expectedB64]) => {
      const bytes = (s) => Uint8Array.from(atob(s), (ch) => ch.charCodeAt(0));
      const cipherBytes = bytes(b64);
      const expected = bytes(expectedB64);
      const decrypt = window.enc.createDecryptor({ algorithm: 'AES-256-CTR', key, iv: ivHex });
      for (const start of [0, 1, 15, 16, 17]) {
        let pos = start;
        const parts = [];
        for (let step = 1; pos < cipherBytes.length; step++) {
          const len = Math.min(cipherBytes.length - pos, [4093, 70001, 15, 65536][step % 4]);
          parts.push(await decrypt(cipherBytes.subarray(pos, pos + len), pos));
          pos += len;
        }
        const got = new Uint8Array(parts.reduce((n, a) => n + a.length, 0));
        let o = 0;
        for (const a of parts) { got.set(a, o); o += a.length; }
        const want = expected.subarray(start);
        if (got.length !== want.length || got.some((v, i) => v !== want[i])) return `mismatch from offset ${start}`;
      }
      return true;
    }, [KEY, iv, cipher, plain.toString('base64')]));
  }
  await ok('fromUrl() downloads, decrypts and caches the plaintext', () => page.evaluate(async ([url, key, iv, hash, size]) => {
    const model = await window.enc.CzechSpeechModel.fromUrl(url, { sha256: hash, size, encryption: { algorithm: 'AES-256-CTR', key, iv } });
    const again = await window.enc.CzechSpeechModel.isDownloaded({ url, sha256: hash });
    await window.enc.CzechSpeechModel.delete({ url, sha256: hash });
    return (model.sha256 === hash && model.size === size && again === true) || 'wrong model';
  }, [`${origin}/data/m.bin.enc`, KEY, IVS[1], sha, plain.length]));
} finally {
  await browser.close();
  server.close?.();
  fs.rmSync(dir, { recursive: true, force: true });
}
console.log(failed ? `${failed} FAILED` : 'ALL PASSED');
process.exit(failed ? 1 : 0);
