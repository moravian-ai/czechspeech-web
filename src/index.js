// @moravianai/czechspeech: offline Czech speech-to-text in the browser (whisper.cpp + Silero VAD in WebAssembly, on the
// GPU through WebGPU where available).
// Audio never leaves the page. Free for noncommercial use (PolyForm Noncommercial 1.0.0); commercial licence: info@moravian.ai.
export { CzechSpeech } from './speech.js';
export { CzechSpeechModel, DEFAULT_MODEL } from './model.js';
export { MODELS, DEFAULT_MODEL_ID } from './catalog.js';
export { DEFAULT_CONFIG } from './config.js';
export { DEFAULT_AUDIO_CONSTRAINTS } from './mic.js';
export { CzechSpeechError, SpeechStartedEvent, SpeechEndedEvent, TranscriptEvent, SpeechErrorEvent, SpeakersEvent } from './events.js';
export { SpeakerSession } from './speakers/session.js';
export { VoiceprintModel } from './speakers/model.js';
export { VOICEPRINT_MODELS, DEFAULT_VOICEPRINT_ID } from './speakers/catalog.js';
export { speakerSupport } from './speakers/support.js';

export const VERSION = '0.3.0';
