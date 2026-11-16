// AudioWorklet processor of listen(): microphone audio at the context's rate -> 16 kHz -> 512-sample frames, posted
// (transferred) through a MessagePort straight to the capture worker. Nothing is buffered beyond the resampler's
// filter history and one partial frame, and both are zeroed on stop.
//
// src/worklet.js is generated from this file and src/resampler.js by scripts/build-worklet.mjs, so that the
// worklet is one self-contained module (bundlers copy worklet files as assets without following their imports).

import { Resampler, Framer } from '../src/resampler.js';

class CzechSpeechCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // `sampleRate` is a global of AudioWorkletGlobalScope: the context's real rate, whatever was requested.
    this.resampler = new Resampler(sampleRate);
    this.out = new Float32Array(this.resampler.maxOutput(4096));
    this.target = null;
    this.stopped = false;
    this.framer = new Framer((frame) => {
      if (this.target) this.target.postMessage(frame, [frame.buffer]);
      else frame.fill(0);
    });
    this.port.onmessage = (e) => {
      const d = e.data;
      if (d && d.type === 'port') this.target = d.port;
      else if (d && d.type === 'finish') this.finish();
      else if (d && d.type === 'stop') this.stop();
    };
  }

  /**
   * stop(): the frame in progress (zero-padded) and then { type: 'end' } go to the capture worker on the same port,
   * after every frame before them, so nothing is lost; then no more audio. Acknowledged with 'finished'. The port
   * is left open: the capture worker reads the end marker and closes its side.
   */
  finish() {
    const delivered = !this.stopped && !!this.target;
    if (delivered) {
      if (this.framer.fill > 0) this.framer.push(new Float32Array(this.framer.frameSamples - this.framer.fill));
      this.target.postMessage({ type: 'end' });
    }
    this.stopped = true;
    this.resampler.clear();
    this.framer.clear();
    this.out.fill(0);
    this.target = null;
    this.port.postMessage({ type: 'finished', delivered });
  }

  stop() {
    if (this.stopped) return;
    this.stopped = true;
    this.resampler.clear();
    this.framer.clear();
    this.out.fill(0);
    if (this.target) this.target.close();
    this.target = null;
  }

  process(inputs) {
    if (this.stopped) return false;
    const input = inputs[0];
    // The node is configured with channelCount 1 and channelCountMode 'explicit': the browser downmixes to mono.
    if (this.target && input && input.length > 0 && input[0].length > 0) {
      const channel = input[0];
      if (this.out.length < this.resampler.maxOutput(channel.length)) {
        this.out = new Float32Array(this.resampler.maxOutput(channel.length));
      }
      const n = this.resampler.process(channel, this.out);
      this.framer.push(this.out, n);
      this.out.fill(0, 0, n);
    }
    return true;
  }
}

registerProcessor('czechspeech-capture', CzechSpeechCaptureProcessor);
