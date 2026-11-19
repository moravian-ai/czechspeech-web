// Encrypted model files: AES-256-CTR known answer, round trip against Node's crypto, resume at any offset, wrong key.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import { CzechSpeechModel, CACHE_NAME } from '../../src/model.js';
import { CzechSpeechError } from '../../src/events.js';
import { normalizeEncryption, createDecryptor, counterBlock } from '../../src/encryption.js';
import { MemoryCacheStorage } from '../support/memory-caches.mjs';

// NIST SP 800-38A, F.5.5 (CTR-AES256.Encrypt, the same vectors decrypt in F.5.6): the fixed test vectors, the only key material allowed in test output
const NIST_KEY = '603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4';
const NIST_IV = 'f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff';
const NIST_PLAIN = '6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e5130c81c46a35ce411e5fbc1191a0a52eff69f2445df4f9b17ad2b417be66c3710';
const NIST_CIPHER = '601ec313775789a5b7a7f504bbf3d228f443e3ca4d62b59aca84e990cacaf5c52b0930daa23de94ce87017ba2d84988ddfc9c58db67aada613c2dd08457941a6';

const enc = (key = NIST_KEY, iv = NIST_IV) => ({ algorithm: 'AES-256-CTR', key, iv });
const reference = (data, key = NIST_KEY, iv = NIST_IV) => {
  const c = crypto.createCipheriv('aes-256-ctr', Buffer.from(key, 'hex'), Buffer.from(iv, 'hex'));
  return Buffer.concat([c.update(data), c.final()]);
};

test('NIST SP 800-38A F.5.5 known answer', async () => {
  const decrypt = createDecryptor(enc());
  assert.equal(Buffer.from(await decrypt(Buffer.from(NIST_CIPHER, 'hex'), 0)).toString('hex'), NIST_PLAIN);
});

test('decrypts at any offset and for any chunk length like Node crypto, also across the 128-bit counter wrap', async () => {
  const data = crypto.randomBytes(300);
  for (const iv of [NIST_IV, '00'.repeat(16), 'ff'.repeat(16), `${'00'.repeat(8)}${'ff'.repeat(8)}`]) {
    const cipher = reference(data, NIST_KEY, iv);
    const decrypt = createDecryptor(enc(NIST_KEY, iv));
    for (const offset of [0, 1, 15, 16, 17, 31, 32, 100, 255]) {
      for (const len of [1, 5, 16, 17, 45]) {
        const end = Math.min(offset + len, data.length);
        assert.deepEqual(Buffer.from(await decrypt(cipher.subarray(offset, end), offset)), data.subarray(offset, end), `iv ${iv} offset ${offset} len ${len}`);
      }
    }
    assert.equal((await decrypt(new Uint8Array(0), 5)).length, 0);
  }
});

test('counterBlock adds the block number as a 128-bit big-endian integer', () => {
  assert.equal(Buffer.from(counterBlock(`${'00'.repeat(15)}ff`, 1)).toString('hex'), `${'00'.repeat(14)}0100`);
  assert.equal(Buffer.from(counterBlock('ff'.repeat(16), 1)).toString('hex'), '00'.repeat(16));
});

test('encryption must be exactly { algorithm, key: 64 hex, iv: 32 hex }, and the message never quotes the key', () => {
  const bad = [null, 'x', {}, { ...enc(), algorithm: 'AES-128-CTR' }, enc(NIST_KEY.slice(2)), enc(`${NIST_KEY}00`), enc('zz'.repeat(32)),
    enc(NIST_KEY, NIST_IV.slice(2)), enc(NIST_KEY, 'g'.repeat(32)), { ...enc(), key: 5 }, { ...enc(), iv: 5 }];
  for (const b of bad) {
    assert.throws(() => normalizeEncryption(b), (err) => err instanceof TypeError && !/603deb10|f0f1f2f3/.test(err.message));
  }
  assert.deepEqual({ ...normalizeEncryption(enc(NIST_KEY.toUpperCase(), NIST_IV.toUpperCase())) }, enc());
});

test('without WebCrypto the error says so (and names no key)', () => {
  assert.throws(() => createDecryptor(enc(), null), (err) => err instanceof CzechSpeechError && err.code === 'unsupported' && !err.message.includes('603deb'));
});

test('a failing WebCrypto becomes a download error without the key', async () => {
  const subtle = { importKey: () => Promise.reject(new Error(`broken ${NIST_KEY}`)), decrypt: () => Promise.reject(new Error('x')) };
  await assert.rejects(createDecryptor(enc(), subtle)(new Uint8Array(4), 0), (err) => err.code === 'download' && err.cause === undefined && !err.message.includes('603deb'));
});

// ---- the download -------------------------------------------------------------------------------------------------

const PART = 16 * 1024 * 1024;
const SIZE = 2 * PART + 12345;
const plain = crypto.randomBytes(SIZE);
const SHA = crypto.createHash('sha256').update(plain).digest('hex');
const cipherText = reference(plain);
const URL_ = 'https://models.example/m.bin.enc';

/** A fetch that serves the ciphertext in odd-sized chunks, honouring Range or not, and can break off. */
function fakeFetch({ honourRange = true, chunk = 70001, cutAfter = null } = {}) {
  const log = [];
  const fn = async (url, init) => {
    const range = init.headers && init.headers.Range;
    log.push(range || null);
    const m = range && /bytes=(\d+)-/.exec(range);
    const start = m && honourRange ? Number(m[1]) : 0;
    let pos = start;
    const body = new ReadableStream({
      pull(controller) {
        if (cutAfter !== null && pos - start >= cutAfter) {
          controller.error(new Error('connection lost'));
          return;
        }
        if (pos >= cipherText.length) {
          controller.close();
          return;
        }
        controller.enqueue(new Uint8Array(cipherText.subarray(pos, Math.min(pos + chunk, cipherText.length))));
        pos += chunk;
      },
    });
    const headers = { 'content-length': String(cipherText.length - start) };
    if (start > 0) headers['content-range'] = `bytes ${start}-${cipherText.length - 1}/${cipherText.length}`;
    return new Response(body, { status: start > 0 ? 206 : 200, headers });
  };
  fn.log = log;
  return fn;
}

const opts = (fetch, caches) => ({ sha256: SHA, size: SIZE, encryption: enc(), _fetch: fetch, _caches: caches });

test('fromUrl decrypts a multi-part download; the cache and the model hold the plaintext', async () => {
  const caches = new MemoryCacheStorage();
  const f = fakeFetch();
  const model = await CzechSpeechModel.fromUrl(URL_, opts(f, caches));
  assert.equal(model.size, SIZE);
  assert.equal(model.sha256, SHA);
  assert.equal(model.cached, true);
  assert.ok(Buffer.from(await model._workerSource().blob.arrayBuffer()).equals(plain));
  // what is stored is the plaintext, so isDownloaded() finds it by { url, sha256 } and nothing more is fetched
  const cache = await caches.open(CACHE_NAME);
  const first = Buffer.from(await (await cache.match(`https://czechspeech.invalid/${SHA}/part-0000`)).arrayBuffer());
  assert.ok(first.equals(plain.subarray(0, PART)));
  assert.equal(await CzechSpeechModel.isDownloaded({ url: URL_, sha256: SHA }, { _caches: caches }), true);
  await CzechSpeechModel.fromUrl(URL_, opts(f, caches));
  assert.equal(f.log.length, 1);
});

for (const honourRange of [true, false]) {
  test(`an interrupted encrypted download resumes at a part boundary (server ${honourRange ? 'honours' : 'ignores'} Range)`, async () => {
    const caches = new MemoryCacheStorage();
    await assert.rejects(CzechSpeechModel.fromUrl(URL_, opts(fakeFetch({ cutAfter: PART + 5000 }), caches)), (err) => err.code === 'download');
    const f = fakeFetch({ honourRange });
    const model = await CzechSpeechModel.fromUrl(URL_, opts(f, caches));
    assert.deepEqual(f.log, [`bytes=${PART}-`]);
    assert.ok(Buffer.from(await model._workerSource().blob.arrayBuffer()).equals(plain));
  });
}

test('a wrong key is a SHA-256 mismatch, nothing stays in the cache, and the key is not in the message', async () => {
  const caches = new MemoryCacheStorage();
  const wrong = `00${NIST_KEY.slice(2)}`;
  await assert.rejects(
    CzechSpeechModel.fromUrl(URL_, { ...opts(fakeFetch(), caches), encryption: enc(wrong) }),
    (err) => err.code === 'download' && /SHA-256 mismatch/.test(err.message) && !err.message.includes(wrong.slice(0, 16)) && !err.message.includes(NIST_KEY.slice(0, 16)),
  );
  assert.equal((await caches.open(CACHE_NAME)).entries.size, 0);
});

test('an encrypted source needs its sha256; a malformed encryption is a TypeError before any request', async () => {
  const f = fakeFetch();
  await assert.rejects(CzechSpeechModel.fromUrl(URL_, { encryption: enc(), _fetch: f }), TypeError);
  await assert.rejects(CzechSpeechModel.fromUrl(URL_, { sha256: SHA, encryption: enc('abc'), _fetch: f }), TypeError);
  await assert.rejects(CzechSpeechModel.fromUrl(URL_, { sha256: SHA, encryption: { ...enc(), algorithm: 'ROT13' }, _fetch: f }), TypeError);
  assert.equal(f.log.length, 0);
});

test('download({ source }) takes the encryption too, and { id } with encryption is refused', async () => {
  const model = await CzechSpeechModel.download({ source: { url: URL_, sha256: SHA, size: SIZE, encryption: enc() },
    _fetch: fakeFetch(), _caches: new MemoryCacheStorage() });
  assert.equal(model.sha256, SHA);
  await assert.rejects(CzechSpeechModel.download({ source: { id: 'cs', encryption: enc() } }), TypeError);
});

test('the pinned model is not served encrypted: a mirror with encryption and no (or the pinned) sha256 is refused', async () => {
  const f = fakeFetch();
  await assert.rejects(CzechSpeechModel.download({ source: { url: URL_, encryption: enc() }, _fetch: f }), TypeError);
  await assert.rejects(CzechSpeechModel.download({ source: { url: URL_, sha256: CzechSpeechModel.DEFAULT.sha256, encryption: enc() }, _fetch: f }), TypeError);
  assert.equal(f.log.length, 0);
});

test('without WebCrypto an encrypted download fails before it starts', async () => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
  Object.defineProperty(globalThis, 'crypto', { value: undefined, configurable: true });
  try {
    const f = fakeFetch();
    await assert.rejects(CzechSpeechModel.fromUrl(URL_, opts(f, new MemoryCacheStorage())), (err) => err.code === 'unsupported');
  } finally {
    Object.defineProperty(globalThis, 'crypto', descriptor);
  }
});
