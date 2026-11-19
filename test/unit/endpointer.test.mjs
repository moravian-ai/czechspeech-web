// Endpointer with synthetic probability sequences: the 14 tests of the Android SDK's EndpointerTest.kt, ported 1:1.
// Each frame's samples are filled with its frame index, so the audio of an utterance shows which frames it contains.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Endpointer, FRAME_MS, FRAME_SAMPLES } from '../../src/endpointer.js';
import { DEFAULT_CONFIG, resolveConfig } from '../../src/config.js';
import { speechSpan } from '../../src/speakers/maths.js';

const config = resolveConfig();
const frameMs = FRAME_MS;

class Run {
  constructor(cfg) {
    this.events = [];
    this.endpointer = new Endpointer(cfg, (e) => this.events.push(e));
    this.index = 0;
    this.frame = new Float32Array(FRAME_SAMPLES);
  }

  feed(prob, frames) {
    for (let i = 0; i < frames; i++) {
      this.frame.fill(this.index);
      this.endpointer.accept(this.frame, prob);
      this.index++;
    }
  }

  feedMs(prob, ms) {
    this.feed(prob, Math.floor(ms / FRAME_MS));
  }

  get utterances() { return this.events.filter((e) => e.kind === 'utterance'); }
  get started() { return this.events.filter((e) => e.kind === 'speechStarted'); }
  get ended() { return this.events.filter((e) => e.kind === 'speechEnded'); }

  /** Frame indices contained in an utterance, from the sample values. */
  frames(u) {
    return Array.from({ length: u.samples.length / FRAME_SAMPLES }, (_, i) => u.samples[i * FRAME_SAMPLES]);
  }
}

const range = (a, b) => Array.from({ length: b - a + 1 }, (_, i) => a + i);
const withConfig = (patch) => resolveConfig({ ...DEFAULT_CONFIG, ...patch });

test('startAndEnd', () => {
  const run = new Run(config);
  run.feed(0.02, 50); // 1.6 s silence, frames 0..49
  run.feed(0.9, 40); // speech, frames 50..89
  run.feed(0.02, 30); // silence

  assert.deepEqual(run.started.map((e) => e.atMs), [50 * frameMs]);
  assert.equal(run.ended.length, 1);
  assert.equal(run.ended[0].atMs, 90 * frameMs);
  assert.ok(run.ended[0].confirmed);

  const [u] = run.utterances;
  assert.equal(run.utterances.length, 1);
  assert.equal(u.startMs, 50 * frameMs);
  assert.equal(u.endMs, 90 * frameMs);
  // pre-roll (10 frames = 320 ms >= 300 ms) + 40 speech + 19 silence frames (608 ms >= 600 ms)
  assert.deepEqual(run.frames(u), range(40, 108));
  // order: started, ended, utterance
  assert.deepEqual(run.events.slice(0, 3).map((e) => e.kind), ['speechStarted', 'speechEnded', 'utterance']);
});

test('speechContinuesWhileAboveNegThreshold', () => {
  const run = new Run(config);
  run.feed(0.9, 20);
  run.feed(0.15, 40); // between negThreshold (0.10) and threshold (0.25): still speech
  run.feed(0.9, 20);
  run.feed(0.02, 25);
  assert.equal(run.utterances.length, 1);
  assert.equal(run.utterances[0].endMs, 80 * frameMs);
});

test('shortSilenceDoesNotEndTheUtterance', () => {
  const run = new Run(config);
  run.feed(0.9, 20);
  run.feed(0.02, 18); // 576 ms < 600 ms
  run.feed(0.9, 20);
  run.feed(0.02, 19);
  assert.equal(run.utterances.length, 1);
  assert.equal(run.ended.length, 1);
});

test('shortBlipsAreDiscarded', () => {
  const run = new Run(config);
  run.feed(0.02, 20);
  run.feed(0.95, 3); // 96 ms click
  run.feed(0.02, 30);
  run.feed(0.6, 7); // 224 ms < 250 ms of speech frames
  run.feed(0.02, 30);
  run.feed(0.2, 100); // never reaches threshold
  run.feed(0.02, 30);
  assert.equal(run.started.length, 0);
  assert.equal(run.utterances.length, 0);
  assert.ok(run.ended.every((e) => !e.confirmed));
  assert.equal(run.ended.length, 2); // the two runs above threshold end (VAD reset), unreported
});

test('eightSpeechFramesAreEnough', () => {
  const run = new Run(config);
  run.feed(0.6, 8); // 256 ms >= 250 ms
  run.feed(0.02, 20);
  assert.equal(run.utterances.length, 1);
});

test('preRollCoversTheAudioBeforeTheTrigger', () => {
  const run = new Run(withConfig({ preRollMs: 300 }));
  run.feed(0.02, 30);
  run.feed(0.9, 20); // trigger at frame 30
  run.feed(0.02, 19);
  const [u] = run.utterances;
  assert.equal(run.utterances.length, 1);
  assert.equal(run.frames(u)[0], 20); // 10 frames of pre-roll: 20..29
  assert.equal(u.startMs, 30 * frameMs);
});

test('preRollIsShorterAtTheStartOfTheStream', () => {
  const run = new Run(config);
  run.feed(0.02, 3);
  run.feed(0.9, 20);
  run.feed(0.02, 19);
  assert.equal(run.utterances.length, 1);
  assert.equal(run.frames(run.utterances[0])[0], 0);
});

test('noPreRoll', () => {
  const run = new Run(withConfig({ preRollMs: 0 }));
  run.feed(0.02, 30);
  run.feed(0.9, 20);
  run.feed(0.02, 19);
  assert.equal(run.utterances.length, 1);
  assert.equal(run.frames(run.utterances[0])[0], 30);
});

test('twoUtterancesSeparatedByTwoSeconds', () => {
  const run = new Run(config);
  run.feedMs(0.02, 1000); // frames 0..30
  run.feedMs(0.9, 2000); // 31..92
  run.feedMs(0.02, 2000); // 93..154
  run.feedMs(0.9, 2000); // 155..216
  run.feedMs(0.02, 1000);

  assert.equal(run.started.length, 2);
  assert.equal(run.ended.length, 2);
  const [a, b] = run.utterances;
  assert.equal(a.startMs, 31 * frameMs);
  assert.equal(a.endMs, 93 * frameMs);
  assert.equal(b.startMs, 155 * frameMs);
  assert.equal(b.endMs, 217 * frameMs);
  assert.equal(run.frames(b)[0], 145); // pre-roll from the silence in between
  // events alternate: started, ended, utterance, started, ended, utterance
  const letter = { speechStarted: 'S', speechEnded: 'E', utterance: 'U' };
  assert.deepEqual(run.events.map((e) => letter[e.kind]), ['S', 'E', 'U', 'S', 'E', 'U']);
});

test('longSpeechIsCutAtTheQuietestFrameBeforeTheCap', () => {
  const run = new Run(config);
  const maxFrames = Math.floor(config.maxUtteranceMs / frameMs); // 875 frames = 28.0 s
  run.feed(0.9, 843);
  run.feed(0.3, 1); // frame 843: the quietest within the last 2 s
  run.feed(0.9, 1250 - 844); // speech until 40 s
  run.feed(0.02, 19);

  assert.equal(run.started.length, 1);
  assert.equal(run.ended.length, 1);
  const [first, second] = run.utterances;
  assert.equal(first.samples.length, 844 * FRAME_SAMPLES);
  assert.equal(first.startMs, 0);
  assert.equal(first.endMs, 844 * frameMs);
  assert.deepEqual(run.frames(first), range(0, 843));
  assert.equal(second.startMs, 844 * frameMs);
  assert.equal(second.endMs, 1250 * frameMs);
  assert.equal(run.frames(second)[0], 844);
  assert.ok(run.utterances.every((u) => u.samples.length <= maxFrames * FRAME_SAMPLES));
});

test('capWithoutAQuietFrameCutsAtTheStartOfTheSearchWindow', () => {
  const run = new Run(config);
  run.feed(0.9, 2000); // 64 s of uniform speech
  run.feed(0.02, 19);
  const us = run.utterances;
  assert.ok(us.length >= 3);
  assert.ok(us.every((u) => u.samples.length <= 875 * FRAME_SAMPLES));
  // pieces are contiguous and cover the whole run
  for (let i = 1; i < us.length; i++) assert.equal(us[i].startMs, us[i - 1].endMs);
  assert.equal(us[us.length - 1].endMs, 2000 * frameMs);
  assert.equal(run.started.length, 1);
});

test('finishClosesAnOpenUtterance', () => {
  const run = new Run(config);
  run.feed(0.02, 10);
  run.feed(0.9, 30);
  run.endpointer.finish();
  assert.equal(run.utterances.length, 1);
  assert.equal(run.utterances[0].endMs, 40 * frameMs);
  assert.equal(run.ended.length, 1);
});

test('clearDropsEverything', () => {
  const run = new Run(config);
  run.feed(0.9, 30);
  run.endpointer.clear();
  run.endpointer.finish();
  assert.equal(run.utterances.length, 0);
  // and the internal buffers are zeroed
  assert.ok(run.endpointer.audio.every((v) => v === 0));
  assert.ok(run.endpointer.ring.every((f) => f.every((v) => v === 0)));
});

test('emittedUtterancesAreCopies', () => {
  const run = new Run(config);
  run.feed(0.9, 20);
  run.feed(0.02, 19);
  const [u] = run.utterances;
  run.feed(0.9, 20); // reuse of the internal buffer must not touch the emitted copy
  run.feed(0.02, 19);
  assert.equal(run.frames(u)[0], 0);
});

// --- audioStartMs: where the audio of an utterance begins (speaker worker) ---
function runProbs(probs, cfg = DEFAULT_CONFIG) {
  const out = [];
  const ep = new Endpointer(cfg, (ev) => { if (ev.kind === 'utterance') out.push(ev); });
  for (const p of probs) ep.accept(new Float32Array(512).fill(0.1), p);
  ep.finish();
  return out;
}
const frameList = (n, p) => Array(n).fill(p);

test('utterance carries audioStartMs: the start of its audio, pre-roll included', () => {
  const [u] = runProbs([...frameList(5, 0), ...frameList(20, 0.9), ...frameList(25, 0)]);
  assert.deepEqual([u.audioStartMs, u.startMs, u.endMs, u.samples.length], [0, 160, 800, 44 * 512]);
  assert.equal(speechSpan(u.samples, u.audioStartMs, u.endMs).length, 25 * 512);
});

test('pre-roll is at most 300 ms (10 frames)', () => {
  const [u] = runProbs([...frameList(15, 0), ...frameList(20, 0.9), ...frameList(25, 0)]);
  assert.deepEqual([u.audioStartMs, u.startMs, u.endMs], [160, 480, 1120]);
  assert.equal(speechSpan(u.samples, u.audioStartMs, u.endMs).length, 30 * 512);
});

test('a piece cut at maxUtteranceMs: its speech span is the whole piece', () => {
  const probs = frameList(200, 0.9);
  probs[110] = 0.3;                                  // the quietest frame of the last 2 s before the cap
  const [first] = runProbs(probs, { ...DEFAULT_CONFIG, maxUtteranceMs: 4000 });
  assert.equal(first.audioStartMs, 0);
  assert.equal(speechSpan(first.samples, first.audioStartMs, first.endMs).length, first.samples.length);
});
