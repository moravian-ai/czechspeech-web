// The CzechSpeech model (whisper.cpp q4_0, 474 MB): pinned download, SHA-256 verification and caching in the
// browser's Cache Storage. This file holds the only network code of the package besides fetching its own assets.
//
// Storage layout (Cache Storage, cache "czechspeech-models"): the file is stored in parts of 16 MiB under synthetic
// keys https://czechspeech.invalid/<id>/part-NNNN plus a small JSON entry .../meta; <id> is the file's SHA-256. Only a
// download with a known SHA-256 is stored: anything else could not be found again by isDownloaded() or delete().
// Parts make the download resumable (an interrupted download continues from the last stored part with an HTTP Range
// request) and are joined into one Blob for the recogniser worker without copying the data.

import { CzechSpeechError } from './events.js';
import { Sha256 } from './sha256.js';
import { MODELS, DEFAULT_MODEL_ID, catalogEntry } from './catalog.js';

export const CACHE_NAME = 'czechspeech-models';
const KEY_BASE = 'https://czechspeech.invalid/';
const PART_BYTES = 16 * 1024 * 1024;
const PROGRESS_STEP = 1024 * 1024;
const HEX64 = /^[0-9a-f]{64}$/;

/** The pinned CzechSpeech q4_0 model on Hugging Face (same file and hash as the Android SDK). */
export const DEFAULT_MODEL = MODELS[DEFAULT_MODEL_ID].source;

const LANGUAGE = /^[a-z]{2,3}$/;

/** The languages of a model given by the app (default: Czech), as a frozen array; RangeError when unusable. */
function normalizeLanguages(languages) {
  if (languages === undefined) return MODELS[DEFAULT_MODEL_ID].languages;
  if (!Array.isArray(languages) || languages.length === 0 || !languages.every((l) => typeof l === 'string' && LANGUAGE.test(l))) {
    throw new RangeError('languages must be a non-empty array of whisper language codes (two or three lower-case letters) such as ["pl", "en"]');
  }
  return Object.freeze([...new Set(languages)]);
}

function normalizeRate(tokensPerSecond) {
  if (tokensPerSecond === undefined) return MODELS[DEFAULT_MODEL_ID].tokensPerSecond;
  if (!(Number.isInteger(tokensPerSecond) && tokensPerSecond > 0)) throw new RangeError('tokensPerSecond must be a positive whole number');
  return tokensPerSecond;
}

/** The catalog id of a { id } source argument (TypeError when mixed with url, sha256 or size), else undefined. */
function idOf(source) {
  if (!source || typeof source !== 'object' || !('id' in source)) return undefined;
  if ('url' in source || 'sha256' in source || 'size' in source) throw new TypeError('pass either { id } or { url, size?, sha256? }, not both');
  return source.id;
}

function normalizeSource(source) {
  if (source === undefined) return DEFAULT_MODEL;
  if (typeof source === 'string') return { url: source, size: null, sha256: null };
  if (!source || typeof source.url !== 'string') throw new TypeError('source must be a URL or { url, size?, sha256? }');
  const sha256 = source.sha256 ? String(source.sha256).toLowerCase() : null;
  if (sha256 && !HEX64.test(sha256)) throw new TypeError('sha256 must be 64 hex digits');
  if (source.size != null && !(Number.isSafeInteger(source.size) && source.size > 0)) {
    throw new TypeError('size must be a positive integer');
  }
  return { url: source.url, size: source.size ?? null, sha256 };
}

/**
 * The source of download(), isDownloaded() and delete(): the pinned model, by default from Hugging Face. A mirror
 * (a URL, or { url }) serves the same file, so the pinned size and SHA-256 stay enforced. A source with a SHA-256
 * of its own is another file and is handled as fromUrl() handles it.
 */
function pinnedSource(source) {
  if (source === undefined) return DEFAULT_MODEL;
  const id = idOf(source);
  if (id !== undefined) return catalogEntry(id).source;
  const src = normalizeSource(source);
  if (src.sha256 && src.sha256 !== DEFAULT_MODEL.sha256) return src;
  if (src.size !== null && src.size !== DEFAULT_MODEL.size) {
    throw new TypeError(`a mirror serves the pinned model (${DEFAULT_MODEL.size} bytes); for another file pass its ` +
      'sha256, or use CzechSpeechModel.fromUrl()');
  }
  return { url: src.url, size: DEFAULT_MODEL.size, sha256: DEFAULT_MODEL.sha256 };
}

function checkUrl(url) {
  let u;
  try {
    u = new URL(url, globalThis.location ? globalThis.location.href : undefined);
  } catch {
    throw new TypeError(`invalid model URL: ${url}`);
  }
  if (u.protocol !== 'https:' && u.protocol !== 'http:') throw new TypeError('the model URL must be http(s)');
  return u.href;
}

const keyOf = (id, name) => `${KEY_BASE}${id}/${name}`;
const partName = (i) => `part-${String(i).padStart(4, '0')}`;

function cacheStorage(override) {
  if (override) return override;
  try {
    return typeof globalThis.caches !== 'undefined' ? globalThis.caches : null;
  } catch (err) {
    throw storageError(err, true); // reading `caches` throws on an opaque origin (a sandboxed iframe)
  }
}

/** Runs a Cache Storage read; a failure (a private window, storage disabled) becomes a CzechSpeechError 'storage'. */
async function cacheOp(fn) {
  try {
    return await fn();
  } catch (err) {
    throw storageError(err, true);
  }
}

const openCache = (cs) => cacheOp(() => cs.open(CACHE_NAME));

async function readMeta(cache, id) {
  const res = await cacheOp(() => cache.match(keyOf(id, 'meta')));
  if (!res) return null;
  try {
    return await res.json();
  } catch {
    return null;
  }
}

const writeMeta = (cache, id, meta) =>
  cache.put(keyOf(id, 'meta'), new Response(JSON.stringify(meta), { headers: { 'content-type': 'application/json' } }));

const deleteEntries = (cache, id, parts) => cacheOp(async () => {
  const keys = await cache.keys();
  const prefix = `${KEY_BASE}${id}/`;
  let n = 0;
  for (const req of keys) {
    const url = typeof req === 'string' ? req : req.url;
    if (url.startsWith(prefix)) {
      await cache.delete(req);
      n++;
    }
  }
  // Some Cache implementations do not list keys reliably during writes; delete the expected names too.
  for (let i = 0; i < (parts || 0); i++) await cache.delete(keyOf(id, partName(i)));
  await cache.delete(keyOf(id, 'meta'));
  return n;
});

/** The stored parts of a complete model joined into one Blob, or null when something is missing. */
const blobFromCache = (cache, id, meta) => cacheOp(async () => {
  const blobs = [];
  for (let i = 0; i < meta.parts; i++) {
    const res = await cache.match(keyOf(id, partName(i)));
    if (!res) return null;
    blobs.push(await res.blob());
  }
  const blob = new Blob(blobs);
  return blob.size === meta.size ? blob : null;
});

function storageError(err, reading = false) {
  if (err instanceof CzechSpeechError) return err;
  if (err && err.name === 'QuotaExceededError') {
    return new CzechSpeechError('Not enough storage for the model (the browser refused to store it)',
      { code: 'storage', cause: err });
  }
  const what = reading ? 'Could not read the model cache' : 'Could not store the model';
  return new CzechSpeechError(`${what}: ${err && err.message}`, { code: 'storage', cause: err });
}

/**
 * Downloads `src` into the cache (resuming from stored parts), verifies size and SHA-256, and returns the Blob.
 * Without Cache Storage, or without a known SHA-256, it downloads into memory.
 */
async function download(src, { onProgress, signal, caches: cachesOverride, fetch: fetchImpl }) {
  const doFetch = fetchImpl || globalThis.fetch;
  const progress = typeof onProgress === 'function' ? onProgress : () => {};
  const id = src.sha256;
  const cs = id ? cacheStorage(cachesOverride) : null;
  const cache = cs ? await openCache(cs) : null;

  let meta = cache ? await readMeta(cache, id) : null;
  if (meta && meta.complete && (!src.size || meta.size === src.size) && meta.sha256 === id) {
    const blob = await blobFromCache(cache, id, meta);
    if (blob) {
      progress(blob.size, blob.size);
      return { blob, cached: true, sha256: meta.sha256 };
    }
  }
  // Resume only when the stored parts are known to belong to the same file (pinned hash and size).
  const resumable = !!(cache && meta && !meta.complete && meta.sha256 === id &&
    src.size && meta.size === src.size && meta.partBytes === PART_BYTES && meta.url === src.url);
  if (cache && meta && !resumable) await deleteEntries(cache, id, meta.parts);
  if (!resumable) meta = null;

  let hasher = new Sha256();
  let parts = 0;
  let have = 0;
  if (resumable) {
    for (; parts < meta.parts; parts++) {
      const res = await cacheOp(() => cache.match(keyOf(id, partName(parts))));
      if (!res) break;
      const bytes = new Uint8Array(await cacheOp(() => res.arrayBuffer()));
      if (bytes.length !== PART_BYTES) break;
      hasher.update(bytes);
      have += bytes.length;
    }
  }
  let total = src.size || 0;
  progress(have, total);

  const memoryParts = [];
  const headers = {};
  if (have > 0) headers.Range = `bytes=${have}-`;
  let res;
  try {
    res = await doFetch(src.url, { headers, signal, mode: 'cors', cache: 'no-store', redirect: 'follow' });
  } catch (err) {
    if (signal && signal.aborted) throw signal.reason;
    throw new CzechSpeechError(`Model download failed: ${err && err.message}`, { code: 'download', cause: err });
  }
  if (src.url.startsWith('https:') && res.url && res.url.startsWith('http:')) {
    throw new CzechSpeechError('Refusing a redirect from https to http', { code: 'download' });
  }
  if (res.status === 206 && have > 0) {
    const range = res.headers.get('content-range'); // exposed by Hugging Face; optional elsewhere
    const m = range && /bytes\s+(\d+)-(\d+)\/(\d+|\*)/.exec(range);
    if (range && (!m || Number(m[1]) !== have || (m[3] !== '*' && total && Number(m[3]) !== total))) {
      throw new CzechSpeechError(`Server resumed at the wrong position (${range})`, { code: 'download' });
    }
  } else if (res.status === 200) {
    if (have > 0) {
      // the server ignored the range request: start over
      await deleteEntries(cache, id, parts);
      hasher = new Sha256();
      parts = 0;
      have = 0;
    }
    const length = Number(res.headers.get('content-length'));
    if (length > 0 && total && length !== total) {
      throw new CzechSpeechError(`Server offers ${length} bytes, expected ${total}`, { code: 'download' });
    }
    if (!total && length > 0) total = length;
  } else if (res.status === 416) {
    if (cache) await deleteEntries(cache, id, parts);
    throw new CzechSpeechError('Server rejected the resume request (HTTP 416); try again', { code: 'download' });
  } else {
    throw new CzechSpeechError(`Model download failed: HTTP ${res.status}`, { code: 'download' });
  }
  if (!res.body) throw new CzechSpeechError('Model download failed: no response body', { code: 'download' });

  meta = { v: 1, url: src.url, size: total || null, sha256: src.sha256, partBytes: PART_BYTES, parts, complete: false };
  if (cache) {
    try {
      await writeMeta(cache, id, meta);
    } catch (err) {
      throw storageError(err);
    }
  }

  let pending = [];
  let pendingBytes = 0;
  let reported = have;
  const flushPart = async () => {
    const blob = new Blob(pending);
    pending = [];
    pendingBytes = 0;
    if (cache) {
      try {
        await cache.put(keyOf(id, partName(parts)), new Response(blob));
        parts++;
        meta.parts = parts;
        await writeMeta(cache, id, meta);
      } catch (err) {
        throw storageError(err);
      }
    } else {
      memoryParts.push(blob);
      parts++;
    }
  };

  const reader = res.body.getReader();
  try {
    for (;;) {
      let chunk;
      try {
        chunk = await reader.read();
      } catch (err) {
        if (signal && signal.aborted) throw signal.reason;
        throw new CzechSpeechError(`Model download interrupted: ${err && err.message}`, { code: 'download', cause: err });
      }
      if (chunk.done) break;
      let bytes = chunk.value;
      if (total && have + bytes.length > total) {
        throw new CzechSpeechError(`Server sent more than ${total} bytes`, { code: 'download' });
      }
      hasher.update(bytes);
      have += bytes.length;
      while (bytes.length > 0) {
        const take = Math.min(bytes.length, PART_BYTES - pendingBytes);
        pending.push(take === bytes.length ? bytes : bytes.subarray(0, take));
        pendingBytes += take;
        bytes = bytes.subarray(take);
        if (pendingBytes === PART_BYTES) await flushPart();
      }
      if (have - reported >= PROGRESS_STEP) {
        reported = have;
        progress(have, total);
      }
    }
  } finally {
    reader.releaseLock?.();
  }
  if (pendingBytes > 0) await flushPart();
  if (total && have !== total) {
    throw new CzechSpeechError(`Download incomplete: ${have} of ${total} bytes; call again to resume`, { code: 'download' });
  }
  const actual = hasher.hex();
  if (src.sha256 && actual !== src.sha256) {
    if (cache) await deleteEntries(cache, id, parts);
    throw new CzechSpeechError(`SHA-256 mismatch: got ${actual}, expected ${src.sha256}`, { code: 'download' });
  }
  let blob;
  if (cache) {
    meta = { ...meta, size: have, sha256: id, complete: true };
    try {
      await writeMeta(cache, id, meta);
    } catch (err) {
      throw storageError(err);
    }
    blob = await blobFromCache(cache, id, meta);
    if (!blob) throw new CzechSpeechError('The stored model is incomplete; call again', { code: 'storage' });
  } else {
    blob = new Blob(memoryParts);
  }
  progress(have, have);
  return { blob, cached: !!cache, sha256: actual };
}

/** A CzechSpeech model ready for CzechSpeech.create(). Create with download(), fromUrl() or fromBytes(). */
export class CzechSpeechModel {
  #blob;
  #path;

  constructor(token, { blob = null, path = null, size, sha256 = null, cached = false, id = null, languages, tokensPerSecond }) {
    if (token !== CzechSpeechModel.#token) throw new TypeError('use CzechSpeechModel.download(), fromUrl() or fromBytes()');
    this.#blob = blob;
    this.#path = path;
    /** Size in bytes. */
    this.size = size;
    /** Lower-case hex SHA-256 when it was verified or computed, else null. */
    this.sha256 = sha256;
    /** True when the model is stored in the browser's Cache Storage (no download on the next visit). */
    this.cached = cached;
    /** The catalog id ('cs'), or null for a model of fromUrl(), fromBytes() or a source of its own. */
    this.id = id;
    /** The languages the model speaks (whisper language codes, two or three lower-case letters; frozen); only these can be forced with `language`. */
    this.languages = normalizeLanguages(languages);
    /** At most how many tokens per second of audio a correct transcript needs (the decoding cap derives from it). */
    this.tokensPerSecond = normalizeRate(tokensPerSecond);
    Object.freeze(this);
  }

  static #token = Symbol('CzechSpeechModel');

  /** The pinned model: { url, size, sha256 }. */
  static get DEFAULT() {
    return DEFAULT_MODEL;
  }

  /**
   * Downloads a model of the catalog (default 'cs': the pinned Czech model, 474 MB) unless it is already cached, and
   * returns it. Resumable, SHA-256-verified, idempotent. `onProgress(done, total)` is called about every MiB.
   *
   * @param {{ id?: string, onProgress?: (done: number, total: number) => void, signal?: AbortSignal,
   *           source?: string | { url: string, size?: number, sha256?: string } }} [options]
   *   id: a catalog id ('cs'); an unknown one rejects with code 'unknown-model'. Not together with `source`.
   *   source: a mirror of the same file; the pinned size and SHA-256 are enforced. A source with a sha256 of its
   *   own is another file, verified against that hash (as with fromUrl()).
   */
  static async download(options = {}) {
    if (options.id !== undefined && options.source !== undefined) throw new TypeError('pass either id or source, not both');
    if (options.id !== undefined) {
      const entry = catalogEntry(options.id);
      return CzechSpeechModel.#fetch(entry.source, options, entry);
    }
    const src = options.source ? pinnedSource(options.source) : DEFAULT_MODEL;
    // a mirror of the pinned model is that model: it carries its id
    return CzechSpeechModel.#fetch(src, options, src.sha256 === DEFAULT_MODEL.sha256 ? MODELS[DEFAULT_MODEL_ID] : undefined);
  }

  /**
   * Downloads a model from any URL (`languages`, default ['cs'], and `tokensPerSecond`, default 25, say what it
   * speaks and how fast; it has no id). With `sha256` the file is verified, stored in Cache Storage (found again by
   * isDownloaded() and delete() with the same { url, sha256 }) and the download resumable. Without it, the file is
   * downloaded into memory each time and not stored (nothing could find it again: a URL alone means the pinned model
   * to isDownloaded() and delete()); its SHA-256 is computed.
   */
  static async fromUrl(url, options = {}) {
    const src = normalizeSource({ url, size: options.size, sha256: options.sha256 });
    normalizeLanguages(options.languages);
    normalizeRate(options.tokensPerSecond);
    return CzechSpeechModel.#fetch(src, options, { languages: options.languages, tokensPerSecond: options.tokensPerSecond });
  }

  static async #fetch(src, options, entry = {}) {
    const s = { ...src, url: checkUrl(src.url) };
    const r = await download(s, { onProgress: options.onProgress, signal: options.signal, caches: options._caches,
      fetch: options._fetch });
    return new CzechSpeechModel(CzechSpeechModel.#token, { blob: r.blob, size: r.blob.size, sha256: r.sha256, cached: r.cached,
      id: entry.id ?? null, languages: entry.languages, tokensPerSecond: entry.tokensPerSecond });
  }

  /** A model from bytes the app already has (ArrayBuffer, typed array or Blob/File). Verified when sha256 is given. */
  static async fromBytes(bytes, options = {}) {
    let blob;
    if (typeof Blob !== 'undefined' && bytes instanceof Blob) blob = bytes;
    else if (bytes instanceof ArrayBuffer || ArrayBuffer.isView(bytes)) blob = new Blob([bytes]);
    else throw new TypeError('fromBytes() takes an ArrayBuffer, a typed array or a Blob');
    let sha256 = null;
    if (options.sha256) {
      const expected = String(options.sha256).toLowerCase();
      if (!HEX64.test(expected)) throw new TypeError('sha256 must be 64 hex digits');
      sha256 = await hashBlob(blob);
      if (sha256 !== expected) {
        throw new CzechSpeechError(`SHA-256 mismatch: got ${sha256}, expected ${expected}`, { code: 'model' });
      }
    }
    return new CzechSpeechModel(CzechSpeechModel.#token, { blob, size: blob.size, sha256, cached: false,
      languages: options.languages, tokensPerSecond: options.tokensPerSecond });
  }

  /**
   * True when the model (default: the pinned one) is completely in the cache. Cheap: no hashing, no network.
   * `source` means what it means for download(): a mirror URL alone stands for the pinned model. A model of fromUrl()
   * is found by its { url, sha256 }; one without sha256 is never cached.
   */
  static async isDownloaded(source, options = {}) {
    const src = pinnedSource(source);
    const cs = cacheStorage(options._caches);
    if (!cs) return false;
    const cache = await openCache(cs);
    const id = src.sha256;
    const meta = await readMeta(cache, id);
    if (!meta || !meta.complete || (src.size && meta.size !== src.size)) return false;
    for (let i = 0; i < meta.parts; i++) if (!(await cacheOp(() => cache.match(keyOf(id, partName(i)))))) return false;
    return true;
  }

  /**
   * Deletes the cached model (default: the pinned one) and any partial download. True when a model was stored.
   * `source` means what it means for download(): delete(url) deletes the pinned model. A model of fromUrl() is
   * deleted by its { url, sha256 }; one without sha256 is never cached.
   */
  static async delete(source, options = {}) {
    const src = pinnedSource(source);
    const cs = cacheStorage(options._caches);
    if (!cs) return false;
    const cache = await openCache(cs);
    const id = src.sha256;
    const meta = await readMeta(cache, id);
    const n = await deleteEntries(cache, id, meta ? meta.parts : 0);
    return !!(meta || n > 0);
  }

  /** Node.js tests only: a model read from a file path inside a worker thread. */
  static _fromPath(path, size, { languages, tokensPerSecond } = {}) {
    return new CzechSpeechModel(CzechSpeechModel.#token, { path, size, sha256: null, cached: false, languages, tokensPerSecond });
  }

  /** What the recogniser worker receives (a Blob is passed by reference, not copied). */
  _workerSource() {
    const meta = { size: this.size, languages: this.languages, tokensPerSecond: this.tokensPerSecond };
    return this.#path ? { path: this.#path, ...meta } : { blob: this.#blob, ...meta };
  }
}

async function hashBlob(blob) {
  const hasher = new Sha256();
  const step = 16 * 1024 * 1024;
  for (let pos = 0; pos < blob.size; pos += step) {
    hasher.update(new Uint8Array(await blob.slice(pos, pos + step).arrayBuffer()));
  }
  return hasher.hex();
}
