// The voiceprint model files in the browser's model cache: the ONNX model and the onnxruntime wasm next to it, both
// downloaded with the same resumable, hash-checked mechanism as the speech model (src/model.js), so that speaker
// recognition also works offline once downloaded. Nothing else is stored; the model is the same for everyone.
import { downloadFile, isFileCached, deleteCachedFile, checkUrl } from '../model.js';
import { DEFAULT_VOICEPRINT_ID, voiceprintEntry } from './catalog.js';
import { ORT_RUNTIME } from './runtime.js';
import { CzechSpeechError } from '../events.js';

// Where the onnxruntime wasm is when neither `runtimeUrl` nor `assetBaseUrl` says otherwise: next to this module. The URL
// is written out in full, as a literal, so that a bundler sees it as an asset and emits the file (and so that nothing
// resolves a directory: `new URL('../', import.meta.url)` is not an asset reference and breaks bundlers).
const defaultRuntimeUrl = () => new URL('./ort/ort-wasm-simd-threaded.wasm', import.meta.url).href;

function pins(id, override) {
  const e = voiceprintEntry(id);
  return {
    entry: e,
    model: override?.model ?? { size: e.size, sha256: e.sha256 },
    runtime: override?.runtime ?? { size: ORT_RUNTIME.size, sha256: ORT_RUNTIME.sha256 },
  };
}

export class VoiceprintModel {
  static #token = Symbol('VoiceprintModel');
  #id;
  #sha256;
  #size;
  #dim;
  #model;
  #runtime;

  constructor(token, { id, sha256, size, dim, model, runtime }) {
    if (token !== VoiceprintModel.#token) throw new TypeError('use VoiceprintModel.download()');
    this.#id = id;
    this.#sha256 = sha256;
    this.#size = size;
    this.#dim = dim;
    this.#model = model;
    this.#runtime = runtime;
  }

  /** @returns {string} */ get id() { return this.#id; }
  /** @returns {string} */ get sha256() { return this.#sha256; }
  /** @returns {number} */ get size() { return this.#size; }
  /** @returns {number} */ get dim() { return this.#dim; }

  /**
   * Downloads the model from `url` (pinned size and sha256 from the catalog; the file is not public, the app says where
   * it serves it from) and the onnxruntime wasm into the model cache. The wasm comes from `runtimeUrl` when the app hosts
   * it itself (next to the model, say; its size and sha256 are pinned, so any copy of the right file will do), else from
   * `speakers/ort/` under `assetBaseUrl`, else from this package. Resumes an interrupted download. `onProgress(done, total)` counts the bytes of both files.
   */
  static async download({ id = DEFAULT_VOICEPRINT_ID, url, assetBaseUrl, runtimeUrl, onProgress, signal, _caches, _fetch, _pins } = {}) {
    const p = pins(id, _pins);
    if (typeof url !== 'string' || !url) throw new TypeError('VoiceprintModel.download() needs the url of the model file');
    const modelSrc = { url: checkUrl(url), size: p.model.size, sha256: p.model.sha256 };
    let runtimeHref;
    if (runtimeUrl != null) {
      try {
        runtimeHref = new URL(String(runtimeUrl), globalThis.location?.href).href;
      } catch (err) {
        throw new CzechSpeechError(`runtimeUrl must be an absolute URL when there is no page location to resolve it against: ${runtimeUrl}`, { code: 'runtime', cause: err });
      }
    } else runtimeHref = assetBaseUrl != null ? new URL(ORT_RUNTIME.path, assetBaseUrl).href : defaultRuntimeUrl();
    const runtimeSrc = { url: checkUrl(runtimeHref), size: p.runtime.size, sha256: p.runtime.sha256 };
    const total = modelSrc.size + runtimeSrc.size;
    const progress = typeof onProgress === 'function' ? onProgress : () => {};
    const model = await downloadFile(modelSrc, { onProgress: (d) => progress(d, total), signal, caches: _caches, fetch: _fetch });
    const runtime = await downloadFile(runtimeSrc, {
      onProgress: (d) => progress(modelSrc.size + d, total), signal, caches: _caches, fetch: _fetch });
    progress(total, total);
    return new VoiceprintModel(VoiceprintModel.#token, {
      id: p.entry.id, sha256: p.model.sha256, size: modelSrc.size, dim: p.entry.dim, model: model.blob, runtime: runtime.blob });
  }

  /** True when both files are completely in the cache. Cheap: no hashing, no network. */
  static async isDownloaded({ id = DEFAULT_VOICEPRINT_ID, _caches, _pins } = {}) {
    const p = pins(id, _pins);
    return (await isFileCached(p.model, { caches: _caches })) && (await isFileCached(p.runtime, { caches: _caches }));
  }

  /** Deletes both files and any partial download. True when something was stored. */
  static async delete({ id = DEFAULT_VOICEPRINT_ID, _caches, _pins } = {}) {
    const p = pins(id, _pins);
    const a = await deleteCachedFile(p.model, { caches: _caches });
    const b = await deleteCachedFile(p.runtime, { caches: _caches });
    return a || b;
  }

  /** What the speaker worker receives (a Blob is passed by reference). */
  _modelBlob() { return this.#model; }

  _runtimeBlob() { return this.#runtime; }
}
