// CzechSpeechModel: download, Cache Storage parts, SHA-256, resume, redirects and errors against a local HTTP server.
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import http from 'node:http';
import { CzechSpeechModel, CACHE_NAME, DEFAULT_MODEL } from '../../src/model.js';
import { CzechSpeechError } from '../../src/events.js';
import { MemoryCacheStorage } from '../support/memory-caches.mjs';

const SIZE = 40 * 1024 * 1024 + 12345; // three 16 MiB parts, the last one partial
const PART = 16 * 1024 * 1024;
const data = crypto.randomBytes(SIZE);
const SHA = crypto.createHash('sha256').update(data).digest('hex');
const PINNED = DEFAULT_MODEL.size;

let server;
let base;
const log = [];
const behaviour = { cutAfter: null, ignoreRange: false, corrupt: false };

before(async () => {
  server = http.createServer((req, res) => {
    const url = new URL(req.url, 'http://x');
    log.push({ path: url.pathname, range: req.headers.range || null });
    if (url.pathname === '/redirect') {
      res.writeHead(302, { Location: '/model.bin' });
      res.end();
      return;
    }
    // /pinned.bin: a mirror of the pinned model that announces its 474 MB but breaks off after the SIZE bytes of
    // `data` (the unit tests never transfer the real model).
    const pinned = url.pathname === '/pinned.bin';
    if (url.pathname !== '/model.bin' && !pinned) {
      res.writeHead(404);
      res.end();
      return;
    }
    const total = pinned ? PINNED : SIZE;
    let start = 0;
    const m = /bytes=(\d+)-/.exec(req.headers.range || '');
    const body = behaviour.corrupt ? Buffer.from(data).fill(7, 1000, 1001) : data;
    if (m && !behaviour.ignoreRange) {
      start = Number(m[1]);
      if (start >= total) {
        res.writeHead(416);
        res.end();
        return;
      }
      res.writeHead(206, { 'Content-Length': total - start, 'Content-Range': `bytes ${start}-${total - 1}/${total}`,
        'Accept-Ranges': 'bytes' });
    } else {
      res.writeHead(200, { 'Content-Length': total, 'Accept-Ranges': 'bytes' });
    }
    const cut = behaviour.cutAfter;
    behaviour.cutAfter = null;
    let pos = start;
    const step = 256 * 1024;
    const pump = () => {
      if (pinned && pos >= SIZE) {
        res.destroy(); // nothing (more) to send
        return;
      }
      while (pos < SIZE) {
        if (cut !== null && pos - start >= cut) {
          res.destroy();
          return;
        }
        const chunk = body.subarray(pos, Math.min(pos + step, SIZE));
        pos += chunk.length;
        // the pinned mirror breaks off after its last chunk has gone out, not while it is still buffered
        const last = pinned && pos >= SIZE;
        if (!res.write(chunk, last ? () => res.destroy() : undefined)) {
          if (!last) res.once('drain', pump);
          return;
        }
      }
      if (!pinned) res.end();
    };
    pump();
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  base = `http://127.0.0.1:${server.address().port}`;
});

after(() => new Promise((r) => server.close(r)));

const reset = () => {
  log.length = 0;
  Object.assign(behaviour, { cutAfter: null, ignoreRange: false, corrupt: false });
};

async function blobBytes(model) {
  return new Uint8Array(await model._workerSource().blob.arrayBuffer());
}

test('downloads, verifies, stores in parts and reports progress', async () => {
  reset();
  const caches = new MemoryCacheStorage();
  const progress = [];
  const model = await CzechSpeechModel.fromUrl(`${base}/model.bin`, { sha256: SHA, size: SIZE, _caches: caches,
    onProgress: (done, total) => progress.push([done, total]) });
  assert.equal(model.size, SIZE);
  assert.equal(model.sha256, SHA);
  assert.equal(model.cached, true);
  assert.ok(Buffer.from(await blobBytes(model)).equals(data));
  const cache = await caches.open(CACHE_NAME);
  const keys = (await cache.keys()).map((r) => r.url).sort();
  assert.deepEqual(keys.map((k) => k.split('/').pop()), ['meta', 'part-0000', 'part-0001', 'part-0002']);
  assert.ok(keys.every((k) => k.startsWith(`https://czechspeech.invalid/${SHA}/`)));
  assert.ok(progress.length >= 40, `${progress.length} progress calls`);
  assert.deepEqual(progress[progress.length - 1], [SIZE, SIZE]);
  assert.ok(progress.every(([d, t], i) => t === SIZE && (i === 0 || d >= progress[i - 1][0])));
  assert.equal(log.length, 1);
  assert.equal(log[0].range, null);

  // second call: from the cache, no network, no hashing
  log.length = 0;
  const again = await CzechSpeechModel.fromUrl(`${base}/model.bin`, { sha256: SHA, size: SIZE, _caches: caches });
  assert.equal(log.length, 0);
  assert.equal(again.size, SIZE);
  assert.ok(await CzechSpeechModel.isDownloaded({ url: `${base}/model.bin`, sha256: SHA, size: SIZE }, { _caches: caches }));

  assert.equal(await CzechSpeechModel.delete({ url: `${base}/model.bin`, sha256: SHA }, { _caches: caches }), true);
  assert.equal(await CzechSpeechModel.isDownloaded({ url: `${base}/model.bin`, sha256: SHA, size: SIZE }, { _caches: caches }), false);
  assert.equal((await cache.keys()).length, 0);
  assert.equal(await CzechSpeechModel.delete({ url: `${base}/model.bin`, sha256: SHA }, { _caches: caches }), false);
});

test('an interrupted download resumes from the last stored part with a Range request', async () => {
  reset();
  const caches = new MemoryCacheStorage();
  behaviour.cutAfter = 20 * 1024 * 1024;
  await assert.rejects(
    CzechSpeechModel.fromUrl(`${base}/model.bin`, { sha256: SHA, size: SIZE, _caches: caches }),
    (e) => e.name === 'CzechSpeechError' && e.code === 'download',
  );
  assert.equal(await CzechSpeechModel.isDownloaded({ url: `${base}/model.bin`, sha256: SHA, size: SIZE }, { _caches: caches }), false);
  const model = await CzechSpeechModel.fromUrl(`${base}/model.bin`, { sha256: SHA, size: SIZE, _caches: caches });
  assert.equal(log.length, 2);
  assert.equal(log[1].range, `bytes=${16 * 1024 * 1024}-`);
  assert.ok(Buffer.from(await blobBytes(model)).equals(data));
});

test('a server that ignores the Range header makes the download start over', async () => {
  reset();
  const caches = new MemoryCacheStorage();
  behaviour.cutAfter = 17 * 1024 * 1024;
  await assert.rejects(CzechSpeechModel.fromUrl(`${base}/model.bin`, { sha256: SHA, size: SIZE, _caches: caches }));
  behaviour.ignoreRange = true;
  const model = await CzechSpeechModel.fromUrl(`${base}/model.bin`, { sha256: SHA, size: SIZE, _caches: caches });
  assert.equal(log[1].range, `bytes=${16 * 1024 * 1024}-`);
  assert.ok(Buffer.from(await blobBytes(model)).equals(data));
  assert.equal(model.sha256, SHA);
});

test('a hash mismatch fails and deletes the stored parts', async () => {
  reset();
  const caches = new MemoryCacheStorage();
  behaviour.corrupt = true;
  await assert.rejects(
    CzechSpeechModel.fromUrl(`${base}/model.bin`, { sha256: SHA, size: SIZE, _caches: caches }),
    (e) => e.code === 'download' && /SHA-256 mismatch/.test(e.message),
  );
  const cache = await caches.open(CACHE_NAME);
  assert.equal((await cache.keys()).length, 0);
});

test('redirects are followed (with the Range header on resume)', async () => {
  reset();
  const caches = new MemoryCacheStorage();
  behaviour.cutAfter = 18 * 1024 * 1024;
  await assert.rejects(CzechSpeechModel.fromUrl(`${base}/redirect`, { sha256: SHA, size: SIZE, _caches: caches }));
  const model = await CzechSpeechModel.fromUrl(`${base}/redirect`, { sha256: SHA, size: SIZE, _caches: caches });
  assert.deepEqual(log.map((l) => l.path), ['/redirect', '/model.bin', '/redirect', '/model.bin']);
  assert.equal(log[3].range, `bytes=${16 * 1024 * 1024}-`);
  assert.equal(model.sha256, SHA);
});

test('an abort signal stops the download; the stored parts are kept for the next call', async () => {
  reset();
  const caches = new MemoryCacheStorage();
  const ac = new AbortController();
  let aborted = false;
  await assert.rejects(
    CzechSpeechModel.fromUrl(`${base}/model.bin`, { sha256: SHA, size: SIZE, _caches: caches, signal: ac.signal,
      onProgress: (done) => { if (done > 20 * 1024 * 1024 && !aborted) { aborted = true; ac.abort(); } } }),
    (e) => e.name === 'AbortError',
  );
  const model = await CzechSpeechModel.fromUrl(`${base}/model.bin`, { sha256: SHA, size: SIZE, _caches: caches });
  assert.ok(log[1].range && log[1].range.startsWith('bytes='));
  assert.equal(model.sha256, SHA);
});

test('the download sends no request header but Range (no Accept-Encoding, which browsers forbid)', async () => {
  // regression (6ce51a4): 'Accept-Encoding: identity' was sent; browsers drop forbidden headers silently
  reset();
  const caches = new MemoryCacheStorage();
  const sent = [];
  const spy = (url, init) => {
    sent.push([...new Headers(init.headers).keys()]);
    return fetch(url, init);
  };
  behaviour.cutAfter = 20 * 1024 * 1024;
  await assert.rejects(CzechSpeechModel.fromUrl(`${base}/model.bin`, { sha256: SHA, size: SIZE, _caches: caches, _fetch: spy }));
  await CzechSpeechModel.fromUrl(`${base}/model.bin`, { sha256: SHA, size: SIZE, _caches: caches, _fetch: spy });
  assert.deepEqual(sent, [[], ['range']]);
});

test('a size that does not match the server fails', async () => {
  reset();
  await assert.rejects(
    CzechSpeechModel.fromUrl(`${base}/model.bin`, { sha256: SHA, size: SIZE + 1, _caches: new MemoryCacheStorage() }),
    (e) => e.code === 'download',
  );
});

test('HTTP errors are reported with the status', async () => {
  reset();
  await assert.rejects(
    CzechSpeechModel.fromUrl(`${base}/missing.bin`, { _caches: new MemoryCacheStorage() }),
    (e) => e.code === 'download' && /HTTP 404/.test(e.message),
  );
});

test('without Cache Storage the model is downloaded into memory (no resume, not cached)', async () => {
  reset();
  const model = await CzechSpeechModel.fromUrl(`${base}/model.bin`, { sha256: SHA, size: SIZE });
  assert.equal(model.cached, false);
  assert.ok(Buffer.from(await blobBytes(model)).equals(data));
  assert.equal(await CzechSpeechModel.isDownloaded(), false);
});

test('a URL without a hash is downloaded into memory, not cached, and its SHA-256 computed', async () => {
  // Regression: cached under its URL, such a model could not be checked or deleted any more (isDownloaded(url)
  // and delete(url) mean the pinned model), so up to 474 MB would stay in the origin's storage for good
  reset();
  const caches = new MemoryCacheStorage();
  const model = await CzechSpeechModel.fromUrl(`${base}/model.bin`, { _caches: caches });
  assert.equal(model.sha256, SHA);
  assert.equal(model.cached, false);
  assert.ok(Buffer.from(await blobBytes(model)).equals(data));
  assert.equal(caches.map.size, 0, 'Cache Storage not even opened');
  await CzechSpeechModel.fromUrl(`${base}/model.bin`, { _caches: caches });
  assert.equal(log.length, 2, 'downloaded again');
});

// ---- download(): the pinned model, also from a mirror ----

const MIRROR = 'https://mirror.example/czechspeech-q4_0.bin';
const pinnedKey = (name) => `https://czechspeech.invalid/${DEFAULT_MODEL.sha256}/${name}`;
const keysOf = async (caches) => (await (await caches.open(CACHE_NAME)).keys()).map((r) => r.url).sort();
const fiveBytes = (headers) => async () => new Response(new Uint8Array([1, 2, 3, 4, 5]), { headers });

test('download() from a mirror rejects a file whose size is not the pinned model\'s, and a contradicting size', async () => {
  for (const source of [{ url: MIRROR }, MIRROR, { url: MIRROR, sha256: DEFAULT_MODEL.sha256 }, { url: MIRROR, size: PINNED }]) {
    const what = JSON.stringify(source);
    // five bytes that are not the model, announced as such
    await assert.rejects(
      CzechSpeechModel.download({ source, _caches: new MemoryCacheStorage(), _fetch: fiveBytes({ 'content-length': '5' }) }),
      (e) => e.name === 'CzechSpeechError' && e.code === 'download' && e.message.includes(`expected ${PINNED}`), what,
    );
    // ... and without a Content-Length: the size is checked when the body ends
    await assert.rejects(
      CzechSpeechModel.download({ source, _caches: new MemoryCacheStorage(), _fetch: fiveBytes() }),
      (e) => e.code === 'download' && e.message.includes(`5 of ${PINNED} bytes`), what,
    );
  }
  // a mirror is the same file: a size that contradicts the pinned one is a programming error
  await assert.rejects(CzechSpeechModel.download({ source: { url: MIRROR, size: 5 }, _caches: new MemoryCacheStorage(),
    _fetch: fiveBytes() }), (e) => e instanceof TypeError && /fromUrl/.test(e.message));
});

test('download() with a source that has a sha256 of its own verifies that file, like fromUrl()', async () => {
  reset();
  const caches = new MemoryCacheStorage();
  const model = await CzechSpeechModel.download({ source: { url: `${base}/model.bin`, sha256: SHA, size: SIZE }, _caches: caches });
  assert.equal(model.sha256, SHA);
  assert.ok(Buffer.from(await blobBytes(model)).equals(data));
  assert.equal(await CzechSpeechModel.isDownloaded({ url: `${base}/model.bin`, sha256: SHA }, { _caches: caches }), true);
  behaviour.corrupt = true;
  await assert.rejects(
    CzechSpeechModel.download({ source: { url: `${base}/model.bin`, sha256: SHA }, _caches: new MemoryCacheStorage() }),
    (e) => e.code === 'download' && /SHA-256 mismatch/.test(e.message),
  );
});

test('download(): progress, resume and abort through a mirror, stored under the pinned model\'s key', async () => {
  reset();
  const caches = new MemoryCacheStorage();
  const source = { url: `${base}/pinned.bin` };
  const progress = [];
  const onProgress = (done, total) => progress.push([done, total]);
  // 1. the mirror breaks off after SIZE bytes: the two whole parts are kept
  await assert.rejects(CzechSpeechModel.download({ source, _caches: caches, onProgress }),
    (e) => e.name === 'CzechSpeechError' && e.code === 'download');
  assert.deepEqual(progress[0], [0, PINNED]);
  assert.ok(progress.length >= 40, `${progress.length} progress calls`);
  assert.ok(progress.every(([d, t], i) => t === PINNED && (i === 0 || d >= progress[i - 1][0])));
  assert.deepEqual(await keysOf(caches), [pinnedKey('meta'), pinnedKey('part-0000'), pinnedKey('part-0001')]);
  assert.equal(await CzechSpeechModel.isDownloaded(source, { _caches: caches }), false);
  // 2. the next call resumes after them with a Range request
  progress.length = 0;
  log.length = 0;
  await assert.rejects(CzechSpeechModel.download({ source, _caches: caches, onProgress }), (e) => e.code === 'download');
  assert.equal(log[0].range, `bytes=${2 * PART}-`);
  assert.deepEqual(progress[0], [2 * PART, PINNED]);
  // 3. an abort rejects with the signal's reason and keeps the parts
  const ac = new AbortController();
  await assert.rejects(CzechSpeechModel.download({ source, _caches: caches, signal: ac.signal,
    onProgress: (done) => { if (done > 2 * PART) ac.abort(); } }), (e) => e.name === 'AbortError');
  assert.deepEqual(await keysOf(caches), [pinnedKey('meta'), pinnedKey('part-0000'), pinnedKey('part-0001')]);
  // 4. delete() with the same source removes the partial download
  assert.equal(await CzechSpeechModel.delete(source, { _caches: caches }), true);
  assert.deepEqual(await keysOf(caches), []);
});

test('isDownloaded() and delete() resolve a source the way download() does', async () => {
  const caches = new MemoryCacheStorage();
  const cache = await caches.open(CACHE_NAME);
  // the pinned model as download() stores it (stand-in parts: isDownloaded() does not read them)
  await cache.put(pinnedKey('meta'), Response.json({ v: 1, url: MIRROR, size: PINNED, sha256: DEFAULT_MODEL.sha256,
    partBytes: PART, parts: 2, complete: true }));
  await cache.put(pinnedKey('part-0000'), new Response('a'));
  await cache.put(pinnedKey('part-0001'), new Response('b'));
  for (const source of [undefined, MIRROR, { url: MIRROR }, { url: MIRROR, sha256: DEFAULT_MODEL.sha256.toUpperCase() }]) {
    assert.equal(await CzechSpeechModel.isDownloaded(source, { _caches: caches }), true, JSON.stringify(source));
  }
  assert.equal(await CzechSpeechModel.isDownloaded({ url: MIRROR, sha256: SHA }, { _caches: caches }), false, 'another file');
  assert.equal(await CzechSpeechModel.delete({ url: MIRROR, sha256: SHA }, { _caches: caches }), false, 'another file');
  assert.equal(await CzechSpeechModel.delete({ url: MIRROR }, { _caches: caches }), true);
  assert.deepEqual(await keysOf(caches), []);
  assert.equal(await CzechSpeechModel.isDownloaded(undefined, { _caches: caches }), false);
  // a source download() refuses is refused here too
  await assert.rejects(CzechSpeechModel.isDownloaded({ url: MIRROR, size: 5 }, { _caches: caches }), TypeError);
  await assert.rejects(CzechSpeechModel.delete({ url: MIRROR, size: 5 }, { _caches: caches }), TypeError);
});

test('fromBytes accepts ArrayBuffer, typed arrays and Blobs and checks an optional hash', async () => {
  const small = crypto.randomBytes(1000);
  const sha = crypto.createHash('sha256').update(small).digest('hex');
  for (const input of [small.buffer.slice(small.byteOffset, small.byteOffset + 1000), new Uint8Array(small), new Blob([small])]) {
    const m = await CzechSpeechModel.fromBytes(input, { sha256: sha });
    assert.equal(m.size, 1000);
    assert.equal(m.sha256, sha);
    assert.equal(m.cached, false);
  }
  await assert.rejects(CzechSpeechModel.fromBytes(small, { sha256: '0'.repeat(64) }), (e) => e.code === 'model');
  await assert.rejects(CzechSpeechModel.fromBytes('nope'), TypeError);
});

/** Cache Storage that fails at `where` ('open', 'match' or 'keys') with `error`, as browsers do in private windows. */
function failingCaches(where, error) {
  const inner = new MemoryCacheStorage();
  return {
    async open(name) {
      if (where === 'open') throw error;
      const cache = await inner.open(name);
      return new Proxy(cache, {
        get(target, prop) {
          if (prop === where) return async () => { throw error; };
          const v = target[prop];
          return typeof v === 'function' ? v.bind(target) : v;
        },
      });
    },
  };
}

test('Cache Storage failures are CzechSpeechError storage, with the browser\'s error as cause', async () => {
  reset();
  const error = new DOMException('The operation is insecure.', 'SecurityError');
  const storage = (what) => (e) => e instanceof CzechSpeechError && e.code === 'storage' && e.cause === error || assert.fail(`${what}: ${e}`);
  for (const where of ['open', 'match']) {
    const _caches = failingCaches(where, error);
    await assert.rejects(CzechSpeechModel.fromUrl(`${base}/model.bin`, { sha256: SHA, _caches }), storage(`fromUrl, ${where}`));
    await assert.rejects(CzechSpeechModel.download({ source: { url: MIRROR }, _caches, _fetch: fiveBytes() }), storage(`download, ${where}`));
    await assert.rejects(CzechSpeechModel.isDownloaded(undefined, { _caches }), storage(`isDownloaded, ${where}`));
    await assert.rejects(CzechSpeechModel.delete(undefined, { _caches }), storage(`delete, ${where}`));
  }
  await assert.rejects(CzechSpeechModel.delete(undefined, { _caches: failingCaches('keys', error) }), storage('delete, keys'));
  // a page where reading `caches` itself throws (an opaque origin)
  Object.defineProperty(globalThis, 'caches', { configurable: true, get() { throw error; } });
  try {
    await assert.rejects(CzechSpeechModel.isDownloaded(), storage('isDownloaded, caches'));
  } finally {
    delete globalThis.caches;
  }
  assert.equal(log.length, 0, 'nothing was downloaded');
});

test('the pinned default is the Android SDK model', () => {
  assert.deepEqual({ ...CzechSpeechModel.DEFAULT }, {
    url: 'https://huggingface.co/moravianai/CzechSpeech/resolve/a4e21bbf9c143d3cfa42488472fed79e9eeba8a9/ggml-czechspeech-q4_0.bin',
    size: 473992235,
    sha256: '04ead6662325354947015bba8604c0e3fbfb9fee598bca1d98340834cb8330a3',
  });
  assert.throws(() => new CzechSpeechModel(Symbol('x'), {}), TypeError);
});

test('download({ id }) and download() target the same pin; id and source together are refused', async () => {
  const caches = new MemoryCacheStorage();
  const cache = await caches.open(CACHE_NAME);
  await cache.put(pinnedKey('meta'), Response.json({ v: 1, url: MIRROR, size: PINNED, sha256: DEFAULT_MODEL.sha256,
    partBytes: PART, parts: 1, complete: true }));
  await cache.put(pinnedKey('part-0000'), new Response('a'));
  const before = log.length;
  for (const options of [{}, { id: 'cs' }]) {
    const model = await CzechSpeechModel.download({ ...options, _caches: caches });
    assert.equal(model.sha256, DEFAULT_MODEL.sha256);
    assert.equal(model.id, 'cs');
    assert.deepEqual([...model.languages], ['cs']);
    assert.equal(model.tokensPerSecond, 25);
  }
  assert.equal(log.length, before, 'served from the cache');
  await assert.rejects(CzechSpeechModel.download({ id: 'cs', source: { url: MIRROR } }), TypeError);
  await assert.rejects(CzechSpeechModel.download({ id: 'xx' }), (e) => e.code === 'unknown-model');
});

test('a model from fromUrl() and fromBytes() is Czech by default and has no id', async () => {
  const model = await CzechSpeechModel.fromUrl(`${base}/model.bin`, { sha256: SHA, _caches: new MemoryCacheStorage() });
  assert.equal(model.id, null);
  assert.deepEqual([...model.languages], ['cs']);
  assert.equal(model.tokensPerSecond, 25);
  assert.ok(Object.isFrozen(model.languages));
  assert.deepEqual(model._workerSource().languages, ['cs']);
  assert.equal(model._workerSource().tokensPerSecond, 25);
  const own = await CzechSpeechModel.fromBytes(new Uint8Array(4), { languages: ['pl', 'en'], tokensPerSecond: 18 });
  assert.deepEqual([...own.languages], ['pl', 'en']);
  assert.equal(own.tokensPerSecond, 18);
  await assert.rejects(CzechSpeechModel.fromUrl(`${base}/model.bin`, { sha256: SHA, languages: [] }), RangeError);
  await assert.rejects(CzechSpeechModel.fromBytes(new Uint8Array(4), { languages: [] }), RangeError);
  await assert.rejects(CzechSpeechModel.fromBytes(new Uint8Array(4), { languages: ['p'] }), RangeError);
  await assert.rejects(CzechSpeechModel.fromBytes(new Uint8Array(4), { tokensPerSecond: 0 }), RangeError);
  // a whole number, as on Android (an Int)
  await assert.rejects(CzechSpeechModel.fromBytes(new Uint8Array(4), { tokensPerSecond: 27.5 }), RangeError);
  await assert.rejects(CzechSpeechModel.fromUrl(`${base}/model.bin`, { sha256: SHA, tokensPerSecond: 27.5 }), RangeError);
  assert.equal((await CzechSpeechModel.fromBytes(new Uint8Array(4), { tokensPerSecond: 27 })).tokensPerSecond, 27);
});

test('a self-hosted model with languages is accepted and verified by its SHA-256 (the README snippet)', async () => {
  reset();
  const caches = new MemoryCacheStorage();
  const url = `${base}/model.bin`;
  const model = await CzechSpeechModel.fromUrl(url, { sha256: SHA, size: SIZE, languages: ['cs', 'sk', 'en'], tokensPerSecond: 25,
    _caches: caches });
  assert.equal(model.id, null);
  assert.equal(model.sha256, SHA);
  assert.deepEqual([...model.languages], ['cs', 'sk', 'en']);
  assert.equal(model.tokensPerSecond, 25);
  assert.ok(Buffer.from(await blobBytes(model)).equals(data));
  assert.equal(await CzechSpeechModel.isDownloaded({ url, sha256: SHA, size: SIZE }, { _caches: caches }), true);
  // a file that does not match the given hash is refused and not kept
  behaviour.corrupt = true;
  const other = new MemoryCacheStorage();
  await assert.rejects(
    CzechSpeechModel.fromUrl(url, { sha256: SHA, size: SIZE, languages: ['cs', 'sk', 'en'], _caches: other }),
    (e) => e.code === 'download' && /SHA-256 mismatch/.test(e.message),
  );
  assert.equal(await CzechSpeechModel.isDownloaded({ url, sha256: SHA, size: SIZE }, { _caches: other }), false);
});

test('isDownloaded({ id }) and delete({ id }) find the file stored under the pinned sha256', async () => {
  const caches = new MemoryCacheStorage();
  const cache = await caches.open(CACHE_NAME);
  await cache.put(pinnedKey('meta'), Response.json({ v: 1, url: MIRROR, size: PINNED, sha256: DEFAULT_MODEL.sha256,
    partBytes: PART, parts: 1, complete: true }));
  await cache.put(pinnedKey('part-0000'), new Response('a'));
  assert.equal(await CzechSpeechModel.isDownloaded({ id: 'cs' }, { _caches: caches }), true);
  await assert.rejects(CzechSpeechModel.isDownloaded({ id: 'xx' }, { _caches: caches }), (e) => e.code === 'unknown-model');
  assert.equal(await CzechSpeechModel.delete({ id: 'cs' }, { _caches: caches }), true);
  assert.equal(await CzechSpeechModel.isDownloaded({ id: 'cs' }, { _caches: caches }), false);
});
