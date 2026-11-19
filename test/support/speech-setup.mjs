// Shared set-up of the listen()/transcribe() tests: a CzechSpeech over the fake engine in real worker threads, and
// (setupWithSpeakers) the same with a SpeakerSession over the fake speaker worker.
import fs from 'node:fs';
import { CzechSpeech, CzechSpeechModel } from '../../src/index.js';
import { SpeakerSession } from '../../src/speakers/session.js';
import { TestEnv, nodeWorkerFactory } from './env.mjs';

export const fakeEntry = new URL('./fake-worker-entry.mjs', import.meta.url);
export const fakeSpeakerEntry = new URL('./fake-speaker-entry.mjs', import.meta.url);
export const vadModel = fs.readFileSync(TestEnv.vadAsset);

export async function setup({ engine, config, keep, noSab, modelPath = 'fake-model.bin', abortGraceMs, extra } = {}) {
  const fake = { capture: null, recognizer: null, events: [] };
  const logs = [];
  const factory = nodeWorkerFactory(fakeEntry);
  const speech = await CzechSpeech.create({
    model: CzechSpeechModel._fromPath(modelPath, 1000),
    config,
    keepModelLoaded: keep,
    log: (level, message) => logs.push(`${level}: ${message}`),
    _createWorker: factory,
    _vadModel: vadModel,
    _engineOptions: engine,
    _noSharedMemory: noSab,
    _abortGraceMs: abortGraceMs,
    ...extra,
    _onWorkerMessage: (role, m) => {
      if (m.type === 'fake') {
        fake[role] = m;
        fake.events.push(`${role}:${m.event}`);
      }
    },
  });
  return { speech, factory, fake, logs };
}

const fakeVoiceModel = { _modelBlob: () => new Blob([new Uint8Array(4)]), _runtimeBlob: () => new Blob([new Uint8Array(4)]), dim: 192, sha256: 'f'.repeat(64) };

/**
 * setup() plus a SpeakerSession whose worker is the fake speaker entry. `speaker` are the fake's options (delayMs,
 * crashAt, reportLengths...), `onCpu` the session's, `onWorkerMessage(m)` sees the fake speaker worker's messages.
 * `close()` closes both.
 */
export async function setupWithSpeakers({ speaker, engine, onCpu = 'run', onWorkerMessage, firstLabel, speech: speechOptions, extra } = {}) {
  const base = await setup({ ...speechOptions, engine, extra });
  const speakerFactory = nodeWorkerFactory(fakeSpeakerEntry);
  const speakerSession = await SpeakerSession.create({ model: fakeVoiceModel, threads: 1, onCpu, firstLabel,
    _createWorker: speakerFactory, _fake: speaker, _onWorkerMessage: onWorkerMessage });
  return { ...base, speakerSession, speakerFactory,
    async close() { await speakerSession.close(); await base.speech.close(); } };
}
