// The microphone of listen(): getUserMedia -> MediaStreamAudioSourceNode -> AudioWorkletNode (src/worklet.js) ->
// 512-sample 16 kHz frames on a MessagePort, which the page transfers to the capture worker unread.

import { CzechSpeechError } from './events.js';

/**
 * getUserMedia constraints for dictation. Each is a hint: browsers may ignore or couple them (Safari's voice
 * processing switches echo cancellation, noise suppression and gain control together).
 *
 * - channelCount 1: one voice, one channel; the node downmixes anyway.
 * - echoCancellation true: removes what the page or the system plays (text-to-speech read-back, notification
 *   sounds, a video) so it is not transcribed. Without anything playing it is close to transparent.
 * - noiseSuppression false: the recogniser was trained on noisy speech; suppression artefacts (musical noise,
 *   clipped word onsets) hurt recognition more than the noise itself. This matches the Android SDK, which records
 *   from VOICE_RECOGNITION (no noise suppression, no gain control).
 * - autoGainControl false: gain pumping raises the background between words, which keeps the speech detector on
 *   longer and makes utterances run together; Whisper's log-mel features do not need a normalised level.
 *
 * Override any of them with `listen({ audio: { ... } })`, e.g. `{ deviceId: { exact: id } }` or
 * `{ autoGainControl: true }` for a very quiet microphone.
 */
export const DEFAULT_AUDIO_CONSTRAINTS = Object.freeze({
  channelCount: 1,
  echoCancellation: true,
  noiseSuppression: false,
  autoGainControl: false,
});

const RESUME_TIMEOUT_MS = 3000;
// stop(): how long the worklet may take to send its last frame and the end marker before the microphone is closed
// anyway (the session then tells the capture worker directly).
const FINISH_TIMEOUT_MS = 1000;

function micError(err) {
  const name = err && err.name;
  if (name === 'NotAllowedError' || name === 'SecurityError' || name === 'PermissionDeniedError') {
    return new CzechSpeechError('Microphone access was denied', { code: 'microphone-denied', cause: err });
  }
  return new CzechSpeechError(`The microphone could not be opened${err && err.message ? `: ${err.message}` : ''}`,
    { code: 'microphone-unavailable', cause: err });
}

/**
 * Opens the microphone (or uses `stream`) and starts the worklet.
 * Returns { port, sampleRate, close(), finish(), onEnded(callback) }. `port` delivers Float32Array(512) frames at
 * 16 kHz; finish() ends them with { type: 'end' } after the last frame and then closes (stop()).
 * A stream passed in by the app is not stopped by close(); one opened here is.
 */
export async function openMicrophone({ workletUrl, audio, stream }) {
  const g = globalThis;
  if (!stream && !(g.navigator && g.navigator.mediaDevices && g.navigator.mediaDevices.getUserMedia)) {
    throw new CzechSpeechError('getUserMedia is not available (it needs a secure context: https or localhost)',
      { code: 'unsupported' });
  }
  const AudioContextClass = g.AudioContext || g.webkitAudioContext;
  if (!AudioContextClass || typeof g.AudioWorkletNode !== 'function') {
    throw new CzechSpeechError('AudioWorklet is not available in this browser', { code: 'unsupported' });
  }

  const ownStream = !stream;
  let ctx = null;
  let node = null;
  let source = null;
  let mute = null;
  let ended = null;
  let closed = false;
  const tracks = [];
  const onTrackEnded = () => {
    if (!closed && ended) {
      ended(new CzechSpeechError('The microphone stopped delivering audio', { code: 'microphone-ended' }));
    }
  };

  const close = async () => {
    if (closed) return;
    closed = true;
    try { node?.port.postMessage({ type: 'stop' }); } catch { /* already gone */ }
    for (const t of tracks) t.removeEventListener('ended', onTrackEnded);
    try { source?.disconnect(); } catch { /* not connected */ }
    try { node?.disconnect(); } catch { /* not connected */ }
    try { mute?.disconnect(); } catch { /* not connected */ }
    if (ownStream) for (const t of tracks) t.stop();
    if (ctx && ctx.state !== 'closed') await ctx.close().catch(() => {});
  };

  // stop(): the worklet sends the frame in progress and an end marker after its last frame, on the same port as
  // the audio, and acknowledges; then the microphone is closed. Resolves true when the end marker was sent.
  const finish = async () => {
    if (closed || !node) return false;
    let timer;
    const acked = new Promise((resolve) => {
      node.port.onmessage = (e) => { if (e.data && e.data.type === 'finished') resolve(!!e.data.delivered); };
    });
    const timeout = new Promise((resolve) => { timer = setTimeout(() => resolve(false), FINISH_TIMEOUT_MS); });
    let delivered = false;
    try {
      node.port.postMessage({ type: 'finish' });
      delivered = await Promise.race([acked, timeout]);
    } catch {
      delivered = false;
    } finally {
      clearTimeout(timer);
      node.port.onmessage = null;
    }
    await close();
    return delivered;
  };

  try {
    // The audio context (at the device's native rate; the worklet resamples to 16 kHz itself) and the worklet
    // module are prepared while the browser opens the microphone: audio captured before the source node exists is
    // not buffered anywhere, so the graph must be ready the moment the stream is. Creating the context first also
    // keeps it within the user gesture that started listen(), which some browsers require to let it run.
    ctx = new AudioContextClass({ latencyHint: 'interactive' });
    const workletReady = ctx.audioWorklet.addModule(workletUrl);
    workletReady.catch(() => {}); // awaited below
    if (!stream) {
      try {
        stream = await g.navigator.mediaDevices.getUserMedia({
          audio: { ...DEFAULT_AUDIO_CONSTRAINTS, ...(audio || {}) },
          video: false,
        });
      } catch (err) {
        throw micError(err);
      }
    }
    for (const t of stream.getAudioTracks()) {
      tracks.push(t);
      t.addEventListener('ended', onTrackEnded);
    }
    if (tracks.length === 0) {
      throw new CzechSpeechError('The stream has no audio track', { code: 'microphone-unavailable' });
    }
    await workletReady;
    source = ctx.createMediaStreamSource(stream);
    node = new g.AudioWorkletNode(ctx, 'czechspeech-capture', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
      channelCount: 1,
      channelCountMode: 'explicit',
      channelInterpretation: 'speakers',
    });
    // Connected to the destination through a muted gain node so that every browser keeps pulling the graph.
    mute = ctx.createGain();
    mute.gain.value = 0;
    source.connect(node);
    node.connect(mute);
    mute.connect(ctx.destination);

    const channel = new MessageChannel();
    node.port.postMessage({ type: 'port', port: channel.port1 }, [channel.port1]);

    if (ctx.state !== 'running') {
      let timer;
      const timeout = new Promise((resolve) => { timer = setTimeout(resolve, RESUME_TIMEOUT_MS); });
      await Promise.race([ctx.resume().catch(() => {}), timeout]);
      clearTimeout(timer);
      if (ctx.state !== 'running') {
        throw new CzechSpeechError('The audio context could not start; call listen() from a user gesture (a click)',
          { code: 'microphone-unavailable' });
      }
    }
    const label = tracks[0].label || '';
    return {
      port: channel.port2,
      sampleRate: ctx.sampleRate,
      label,
      settings: typeof tracks[0].getSettings === 'function' ? tracks[0].getSettings() : {},
      close,
      finish,
      onEnded(callback) {
        ended = callback;
        if (tracks.some((t) => t.readyState === 'ended')) onTrackEnded();
      },
    };
  } catch (err) {
    await close();
    throw err instanceof CzechSpeechError ? err : micError(err);
  }
}
