// src/mic.js with fake Web Audio and getUserMedia objects (Node.js has neither): when the audio graph is built
// relative to the stream, the getUserMedia constraints, the error mapping, a track that ends, finish() and close().
import { test, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_AUDIO_CONSTRAINTS, openMicrophone } from '../../src/mic.js';
import { CzechSpeechError } from '../../src/events.js';
import { within } from '../support/env.mjs';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

class FakeTrack {
  constructor() {
    this.readyState = 'live';
    this.label = 'fake microphone';
    this.stopped = false;
    this.listeners = new Map();
  }

  addEventListener(type, fn) { this.listeners.set(type, fn); }

  removeEventListener(type, fn) { if (this.listeners.get(type) === fn) this.listeners.delete(type); }

  getSettings() { return { sampleRate: 48000, channelCount: 1 }; }

  stop() {
    this.stopped = true;
    this.readyState = 'ended';
  }

  /** The browser ends the track (device unplugged, permission revoked). */
  end() {
    this.readyState = 'ended';
    const fn = this.listeners.get('ended');
    if (fn) fn();
  }
}

/**
 * Installs fake AudioContext, AudioWorkletNode and navigator.mediaDevices. `events` records what happened when
 * (ms since install): the audio graph must be ready when the stream is, or its first audio is lost.
 */
function installFakes({ streamAfterMs = 0, workletAfterMs = 0, gumError = null, tracks = 1, startState = 'running',
  resumable = true, acknowledgeFinish = true } = {}) {
  const t0 = performance.now();
  const env = { events: [], ctx: null, node: null, constraints: null, stream: null };
  const at = (what) => env.events.push({ what, ms: performance.now() - t0 });
  env.stream = { tracks: Array.from({ length: tracks }, () => new FakeTrack()), getAudioTracks() { return this.tracks; } };
  const graphNode = () => ({ connected: [], connect(n) { this.connected.push(n); }, disconnect() { this.connected = []; } });

  class FakeAudioContext {
    constructor(options) {
      at('AudioContext');
      this.options = options;
      this.state = startState;
      this.sampleRate = 48000;
      this.destination = graphNode();
      this.audioWorklet = {
        addModule: (url) => {
          at('addModule');
          this.workletUrl = url;
          return sleep(workletAfterMs).then(() => at('workletReady'));
        },
      };
      env.ctx = this;
    }

    createMediaStreamSource(stream) {
      at('createMediaStreamSource');
      return Object.assign(graphNode(), { stream });
    }

    createGain() { return Object.assign(graphNode(), { gain: { value: 1 } }); }

    async resume() { if (resumable) this.state = 'running'; }

    async close() {
      at('close');
      this.state = 'closed';
    }
  }

  class FakeAudioWorkletNode {
    constructor(ctx, name, options) {
      at('AudioWorkletNode');
      Object.assign(this, graphNode(), { name, options });
      const port = {
        posted: [],
        onmessage: null,
        postMessage: (m) => {
          port.posted.push(m.type);
          if (m.type === 'finish' && acknowledgeFinish) {
            queueMicrotask(() => port.onmessage && port.onmessage({ data: { type: 'finished', delivered: true } }));
          }
        },
      };
      this.port = port;
      env.node = this;
    }
  }

  globalThis.AudioContext = FakeAudioContext;
  globalThis.AudioWorkletNode = FakeAudioWorkletNode;
  Object.defineProperty(globalThis.navigator, 'mediaDevices', {
    configurable: true,
    value: {
      getUserMedia(constraints) {
        at('getUserMedia');
        env.constraints = constraints;
        return sleep(streamAfterMs).then(() => {
          at('stream');
          if (gumError) throw gumError;
          return env.stream;
        });
      },
    },
  });
  return env;
}

afterEach(() => {
  delete globalThis.AudioContext;
  delete globalThis.AudioWorkletNode;
  delete globalThis.navigator.mediaDevices;
});

const when = (env, what) => {
  const e = env.events.find((x) => x.what === what);
  assert.ok(e, `${what} never happened: ${env.events.map((x) => x.what).join(', ')}`);
  return e.ms;
};

test('the audio graph is ready when the stream is: context and worklet are prepared during getUserMedia', async () => {
  // regression (6ce51a4): built after getUserMedia, the graph missed the first ~0.2 s, often the first word
  const env = installFakes({ streamAfterMs: 150, workletAfterMs: 100 });
  const mic = await openMicrophone({ workletUrl: 'https://app.example/worklet.js' });
  const order = env.events.map((e) => e.what);
  assert.ok(order.indexOf('AudioContext') < order.indexOf('getUserMedia'), `${order}: the context comes first (user gesture)`);
  assert.ok(order.indexOf('addModule') < order.indexOf('getUserMedia'), `${order}: the worklet loads during getUserMedia`);
  const gap = when(env, 'createMediaStreamSource') - when(env, 'stream');
  assert.ok(gap < within(50), `the source node came ${Math.round(gap)} ms after the stream (the worklet takes 100 ms)`);
  assert.equal(env.ctx.workletUrl, 'https://app.example/worklet.js');
  assert.deepEqual(env.constraints, { audio: { ...DEFAULT_AUDIO_CONSTRAINTS }, video: false });
  assert.equal(mic.sampleRate, 48000);
  assert.ok(mic.port);
  await mic.close();
});

test('getUserMedia constraints: the dictation defaults, overridden by listen({ audio })', async () => {
  const env = installFakes();
  const mic = await openMicrophone({ workletUrl: 'w.js', audio: { autoGainControl: true, deviceId: { exact: 'x' } } });
  assert.deepEqual(env.constraints.audio, { ...DEFAULT_AUDIO_CONSTRAINTS, autoGainControl: true, deviceId: { exact: 'x' } });
  await mic.close();
});

test('microphone errors: denied, unavailable, no audio track, an audio context that cannot start, no API', async () => {
  for (const name of ['NotAllowedError', 'SecurityError', 'PermissionDeniedError']) {
    const err = new DOMException('no', name);
    const env = installFakes({ gumError: err });
    await assert.rejects(openMicrophone({ workletUrl: 'w.js' }),
      (e) => e instanceof CzechSpeechError && e.code === 'microphone-denied' && e.cause === err, name);
    assert.equal(env.ctx.state, 'closed', `${name}: the audio context is closed again`);
  }
  const busy = new DOMException('Could not start audio source', 'NotReadableError');
  installFakes({ gumError: busy });
  await assert.rejects(openMicrophone({ workletUrl: 'w.js' }),
    (e) => e.code === 'microphone-unavailable' && e.cause === busy && /Could not start audio source/.test(e.message));
  installFakes({ tracks: 0 });
  await assert.rejects(openMicrophone({ workletUrl: 'w.js' }), (e) => e.code === 'microphone-unavailable' && /no audio track/.test(e.message));
  // resume() settles but the context stays suspended (no user activation)
  const suspended = installFakes({ startState: 'suspended', resumable: false });
  await assert.rejects(openMicrophone({ workletUrl: 'w.js' }), (e) => e.code === 'microphone-unavailable' && /user gesture/.test(e.message));
  assert.ok(suspended.stream.tracks.every((t) => t.stopped), 'the microphone it opened is stopped again');
  installFakes({ startState: 'suspended', resumable: true });
  const resumed = await openMicrophone({ workletUrl: 'w.js' });
  await resumed.close();
  delete globalThis.AudioWorkletNode;
  await assert.rejects(openMicrophone({ workletUrl: 'w.js' }), (e) => e.code === 'unsupported' && /AudioWorklet/.test(e.message));
  delete globalThis.navigator.mediaDevices;
  await assert.rejects(openMicrophone({ workletUrl: 'w.js' }), (e) => e.code === 'unsupported' && /secure context/.test(e.message));
});

test('a track that ends reports microphone-ended once; close() stops only a stream it opened itself', async () => {
  const env = installFakes();
  const mic = await openMicrophone({ workletUrl: 'w.js' });
  const ended = [];
  mic.onEnded((e) => ended.push(e));
  env.stream.tracks[0].end();
  assert.equal(ended.length, 1);
  assert.ok(ended[0] instanceof CzechSpeechError && ended[0].code === 'microphone-ended');
  await mic.close();
  await mic.close(); // twice is fine
  assert.ok(env.stream.tracks.every((t) => t.stopped));
  assert.equal(env.ctx.state, 'closed');
  assert.ok(env.node.port.posted.includes('stop'));
  env.stream.tracks[0].end();
  assert.equal(ended.length, 1, 'nothing after close()');
  // the app's own stream: used, not stopped
  const own = installFakes();
  const appStream = own.stream;
  const mic2 = await openMicrophone({ workletUrl: 'w.js', stream: appStream });
  assert.ok(!own.events.some((e) => e.what === 'getUserMedia'));
  await mic2.close();
  assert.ok(appStream.tracks.every((t) => !t.stopped), 'the app stream keeps running');
  // a track that had already ended when the session subscribes
  const late = installFakes();
  const mic3 = await openMicrophone({ workletUrl: 'w.js' });
  late.stream.tracks[0].readyState = 'ended';
  const lateEnded = [];
  mic3.onEnded((e) => lateEnded.push(e.code));
  assert.deepEqual(lateEnded, ['microphone-ended']);
  await mic3.close();
});

test('finish(): the worklet ends the audio with an end marker and acknowledges, then the microphone closes', async () => {
  const env = installFakes();
  const mic = await openMicrophone({ workletUrl: 'w.js' });
  assert.equal(await mic.finish(), true);
  assert.deepEqual(env.node.port.posted.filter((t) => t !== 'port'), ['finish', 'stop']);
  assert.equal(env.ctx.state, 'closed');
  assert.equal(await mic.finish(), false, 'nothing to finish after close');
  // a worklet that does not answer: finish() gives up after its timeout and still closes
  const silent = installFakes({ acknowledgeFinish: false });
  const mic2 = await openMicrophone({ workletUrl: 'w.js' });
  const t0 = Date.now();
  assert.equal(await mic2.finish(), false);
  assert.ok(Date.now() - t0 >= 900, 'waited for the acknowledgement');
  assert.equal(silent.ctx.state, 'closed');
});
