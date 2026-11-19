// Reading the synthetic test meetings (JSON with the turns, WAV with the audio, WAV enrolment recordings per speaker).
import fs from 'node:fs';
import path from 'node:path';
import { readWav } from '../support/env.mjs';

/** Ids of the meetings of the four-person scenario (test-S4-NN, without the follow-up "-b" ones), sorted. */
export function s4TestIds(dir) {
  return fs.readdirSync(dir).map((f) => /^(test-S4-\d+)\.json$/.exec(f)).filter(Boolean).map((m) => m[1]).sort();
}

/** Where the enrolment recording of speaker `s` for meeting `id` is (own file first, else the shared one). */
export function enrollFile(dir, id, s) {
  const own = path.join(dir, `${id}.enroll.${s}.wav`);
  return fs.existsSync(own) ? own : path.join(dir, `enroll.${s}.wav`);
}

export function readMeetingMeta(dir, id) {
  const meta = JSON.parse(fs.readFileSync(path.join(dir, `${id}.json`), 'utf8'));
  return {
    id,
    speakers: Object.keys(meta.enroll_ids),
    turns: meta.turns.map((t) => ({ speaker: t.speaker, startMs: t.start_ms, endMs: t.end_ms })),
  };
}

export function readMeeting(dir, id) {
  const meta = readMeetingMeta(dir, id);
  return {
    id,
    audio: readWav(path.join(dir, `${id}.wav`)),
    turns: meta.turns,
    enroll: new Map(meta.speakers.map((s) => [s, readWav(enrollFile(dir, id, s))])),
  };
}
