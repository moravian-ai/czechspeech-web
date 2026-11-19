import { test } from 'node:test';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import { VoiceprintModel } from '../../src/speakers/model.js';
import { CzechSpeechError } from '../../src/events.js';
import { MemoryCacheStorage } from '../support/memory-caches.mjs';

const bytes = (n, seed) => Uint8Array.from({ length: n }, (_, i) => (i * 31 + seed) & 0xff);
const sha = (b) => crypto.createHash('sha256').update(b).digest('hex');
const MODEL = bytes(3000, 1), RUNTIME = bytes(2000, 2);
const pins = { model: { size: MODEL.length, sha256: sha(MODEL) }, runtime: { size: RUNTIME.length, sha256: sha(RUNTIME) } };
const files = { 'https://app.test/models/voiceprint/x.onnx': MODEL, 'https://app.test/czechspeech/speakers/ort/ort-wasm-simd-threaded.wasm': RUNTIME };
function fakeFetch(log) {
  return async (url) => {
    log.push(url);
    const body = files[url];
    return body ? new Response(body, { status: 200, headers: { 'content-length': String(body.length) } }) : new Response('', { status: 404 });
  };
}
const opts = (caches, log) => ({ url: 'https://app.test/models/voiceprint/x.onnx', assetBaseUrl: 'https://app.test/czechspeech/',
  _caches: caches, _fetch: fakeFetch(log), _pins: pins });

test('download() stores the model and the runtime, verified, with progress over both', async () => {
  const caches = new MemoryCacheStorage(), log = [], progress = [];
  const m = await VoiceprintModel.download({ ...opts(caches, log), onProgress: (d, t) => progress.push([d, t]) });
  assert.equal(m.sha256, pins.model.sha256);
  assert.equal(log.length, 2);
  assert.deepEqual(progress.at(-1), [5000, 5000]);
  assert.equal(await VoiceprintModel.isDownloaded({ _caches: caches, _pins: pins }), true);
});

test('offline after one download: no request at all, both files from the cache (review focus 5)', async () => {
  const caches = new MemoryCacheStorage(), log = [];
  await VoiceprintModel.download(opts(caches, log));
  const offline = [];
  const m = await VoiceprintModel.download({ ...opts(caches, offline), _fetch: async (u) => { offline.push(u); throw new TypeError('offline'); } });
  assert.equal(offline.length, 0);
  assert.equal((await m._modelBlob().arrayBuffer()).byteLength, 3000);
  assert.equal((await m._runtimeBlob().arrayBuffer()).byteLength, 2000);
});

test('a wrong file is refused and nothing complete stays', async () => {
  const caches = new MemoryCacheStorage();
  const bad = { ...pins, model: { size: 3000, sha256: '0'.repeat(64) } };
  await assert.rejects(VoiceprintModel.download({ ...opts(caches, []), _pins: bad }), (e) => e instanceof CzechSpeechError && e.code === 'download');
  assert.equal(await VoiceprintModel.isDownloaded({ _caches: caches, _pins: bad }), false);
});

test('delete() removes both files', async () => {
  const caches = new MemoryCacheStorage();
  await VoiceprintModel.download(opts(caches, []));
  assert.equal(await VoiceprintModel.delete({ _caches: caches, _pins: pins }), true);
  assert.equal(await VoiceprintModel.isDownloaded({ _caches: caches, _pins: pins }), false);
});

test('url is required; unknown id is unknown-model', async () => {
  await assert.rejects(VoiceprintModel.download({ _caches: new MemoryCacheStorage() }), TypeError);
  await assert.rejects(VoiceprintModel.download({ id: 'x', url: 'https://a/b' }), (e) => e.code === 'unknown-model');
});

test('the model fields are read-only', async () => {
  const m = await VoiceprintModel.download(opts(new MemoryCacheStorage(), []));
  assert.throws(() => { m.sha256 = 'x'; }, TypeError);
  assert.equal(m.sha256, pins.model.sha256);
  assert.equal(m.dim, 192);
});

test('runtimeUrl: the app hosts the wasm itself; it wins over assetBaseUrl', async () => {
  const caches = new MemoryCacheStorage(), log = [];
  const host = 'https://cdn.test/wasm/ort.wasm';
  files[host] = RUNTIME;
  try {
    await VoiceprintModel.download({ ...opts(caches, log), runtimeUrl: host });
    assert.deepEqual(log, ['https://app.test/models/voiceprint/x.onnx', host]);
  } finally {
    delete files[host];
  }
});

test('a relative runtimeUrl without a page location is a CzechSpeechError, not a raw "Invalid URL"', async () => {
  const saved = Object.getOwnPropertyDescriptor(globalThis, 'location');
  delete globalThis.location;
  try {
    await assert.rejects(VoiceprintModel.download({ ...opts(new MemoryCacheStorage(), []), runtimeUrl: 'wasm/ort.wasm' }),
      (e) => e instanceof CzechSpeechError && /runtimeUrl/.test(e.message));
  } finally {
    if (saved) Object.defineProperty(globalThis, 'location', saved);
  }
});
