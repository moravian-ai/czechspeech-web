// The model catalog: models are picked by id ('cs'), and an entry carries what the engine needs to know about a
// model: where its file is (pinned size and SHA-256), the languages it speaks and how fast a correct transcript
// produces tokens; `voiceprint` is the id of its speaker voiceprint model (src/speakers/catalog.js), or null.

import { CzechSpeechError } from './events.js';

/** The id of the model used when none is given. */
export const DEFAULT_MODEL_ID = 'cs';

/** The catalog: { id, source: { url, size, sha256 }, languages, tokensPerSecond } by id. */
export const MODELS = Object.freeze({
  cs: Object.freeze({
    id: 'cs',
    source: Object.freeze({
      url: 'https://huggingface.co/moravianai/CzechSpeech/resolve/a4e21bbf9c143d3cfa42488472fed79e9eeba8a9/ggml-czechspeech-q4_0.bin',
      size: 473992235,
      sha256: '04ead6662325354947015bba8604c0e3fbfb9fee598bca1d98340834cb8330a3',
    }),
    languages: Object.freeze(['cs']),
    tokensPerSecond: 25,
    voiceprint: 'voiceprint-cs',
  }),
});

/** The entry of `id`; a CzechSpeechError with code 'unknown-model' when there is none. */
export function catalogEntry(id) {
  if (typeof id === 'string' && Object.hasOwn(MODELS, id)) return MODELS[id];
  throw new CzechSpeechError(`Unknown model id ${JSON.stringify(id)}; known ids: ${Object.keys(MODELS).join(', ')}`,
    { code: 'unknown-model' });
}
