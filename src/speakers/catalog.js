// The voiceprint model catalog. The entry is bound to its file (size and sha256 of the file after the metadata fix, see
// tools/voiceprint/fix-metadata.mjs) and carries the operating points measured for that file. The model is not hosted
// publicly: `url` is null and the application passes where it serves the file from (VoiceprintModel.download({ url })).
import { CzechSpeechError } from '../events.js';
import { STRICT } from './points.js';

export const VOICEPRINT_MODELS = Object.freeze({
  'voiceprint-cs': Object.freeze({
    id: 'voiceprint-cs',
    size: 26485420,
    sha256: '750339c64b30fa8dc0939fb030efc42d848987cca4cfe80b5703039af2d95983',
    url: null,
    dim: 192,
    operatingPoints: Object.freeze({ strict: STRICT }),
  }),
});

export const DEFAULT_VOICEPRINT_ID = 'voiceprint-cs';

/** The entry of `id`; a CzechSpeechError with code 'unknown-model' when there is none. */
export function voiceprintEntry(id) {
  if (typeof id === 'string' && Object.hasOwn(VOICEPRINT_MODELS, id)) return VOICEPRINT_MODELS[id];
  throw new CzechSpeechError(`Unknown voiceprint model id ${JSON.stringify(id)}; known ids: ${Object.keys(VOICEPRINT_MODELS).join(', ')}`,
    { code: 'unknown-model' });
}
