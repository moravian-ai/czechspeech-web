// Streaming resampler to 16 kHz and a 512-sample framer, used inside the AudioWorklet. Self-contained (no imports)
// so the worklet module stays a single small file.
//
// Browsers capture at whatever rate the audio device runs (44.1, 48, 96 kHz ...), and an AudioContext created with
// sampleRate 16000 is not reliable across browsers when a microphone stream is connected to it. So the context runs
// at its native rate and this resampler converts: a Kaiser-windowed sinc low-pass (cut-off 92 % of the lower
// Nyquist frequency, 16 zero crossings, beta 8: about 80 dB stop-band) evaluated at the exact rational position of
// every output sample, with the kernel tabulated at 256 phases per input sample and interpolated linearly.

export const TARGET_RATE = 16000;
export const FRAME_SAMPLES = 512;

function besselI0(x) {
  let sum = 1;
  let term = 1;
  const q = (x * x) / 4;
  for (let k = 1; k < 50; k++) {
    term *= q / (k * k);
    sum += term;
    if (term < sum * 1e-12) break;
  }
  return sum;
}

export class Resampler {
  /**
   * @param {number} inRate input sample rate in Hz (integer)
   * @param {number} [outRate] output sample rate in Hz (integer), 16000 by default
   */
  constructor(inRate, outRate = TARGET_RATE, { zeroCrossings = 16, rolloff = 0.92, beta = 8, phases = 256 } = {}) {
    if (!Number.isInteger(inRate) || inRate <= 0 || !Number.isInteger(outRate) || outRate <= 0) {
      throw new RangeError('sample rates must be positive integers');
    }
    this.inRate = inRate;
    this.outRate = outRate;
    this.passthrough = inRate === outRate;
    // cut-off in cycles per input sample
    const fc = 0.5 * Math.min(1, outRate / inRate) * rolloff;
    const W = Math.ceil(zeroCrossings / (2 * fc));
    this.halfWidth = W;
    this.phases = phases;
    const n = W * phases + 2;
    const table = new Float32Array(n);
    const i0b = besselI0(beta);
    for (let i = 0; i < n; i++) {
      const x = i / phases;
      if (x >= W) continue;
      const u = 2 * fc * x;
      const sinc = u === 0 ? 1 : Math.sin(Math.PI * u) / (Math.PI * u);
      const r = x / W;
      table[i] = 2 * fc * sinc * (besselI0(beta * Math.sqrt(1 - r * r)) / i0b);
    }
    this.table = table;
    // History of input samples; hist[0] is absolute input index histStart. Starts with W zeros before index 0.
    this.hist = new Float32Array(Math.max(4096, 4 * W));
    this.histStart = -W;
    this.histLen = W;
    this.produced = 0; // output samples produced so far
  }

  /** Upper bound of the output count for `inputLength` more input samples. */
  maxOutput(inputLength) {
    return Math.ceil((inputLength * this.outRate) / this.inRate) + 2;
  }

  /**
   * Resamples the next block of input into `output` (capacity >= maxOutput(input.length)); returns the number of
   * samples written. Blocks may have any length; the result does not depend on how the input is split.
   */
  process(input, output) {
    if (this.passthrough) {
      output.set(input);
      return input.length;
    }
    const W = this.halfWidth;
    if (this.histLen + input.length > this.hist.length) {
      const bigger = new Float32Array(Math.max(this.hist.length * 2, this.histLen + input.length));
      bigger.set(this.hist.subarray(0, this.histLen));
      this.hist.fill(0);
      this.hist = bigger;
    }
    this.hist.set(input, this.histLen);
    this.histLen += input.length;

    const { hist, table, phases, inRate, outRate } = this;
    const lastAbs = this.histStart + this.histLen - 1;
    let count = 0;
    for (;;) {
      const num = this.produced * inRate;
      const center = Math.floor(num / outRate);
      if (center + W > lastAbs) break;
      const frac = (num - center * outRate) / outRate;
      const pos = center + frac;
      let acc = 0;
      let idx = center - W + 1 - this.histStart;
      for (let k = center - W + 1; k <= center + W; k++, idx++) {
        const p = Math.abs(pos - k) * phases;
        const i = p | 0;
        const h = table[i] + (table[i + 1] - table[i]) * (p - i);
        acc += hist[idx] * h;
      }
      output[count++] = acc;
      this.produced++;
    }

    // Drop input that no future output needs, zeroing the vacated tail.
    const nextCenter = Math.floor((this.produced * inRate) / outRate);
    const drop = nextCenter - W + 1 - this.histStart;
    if (drop > 0) {
      const keep = this.histLen - drop;
      hist.copyWithin(0, drop, this.histLen);
      hist.fill(0, keep, this.histLen);
      this.histLen = keep;
      this.histStart += drop;
    }
    return count;
  }

  /** Zeroes the buffered audio. */
  clear() {
    this.hist.fill(0);
    this.histLen = 0;
  }
}

/** Collects samples into frames of FRAME_SAMPLES and hands each new frame to `emit` (the receiver owns it). */
export class Framer {
  constructor(emit, frameSamples = FRAME_SAMPLES) {
    this.emit = emit;
    this.frameSamples = frameSamples;
    this.frame = new Float32Array(frameSamples);
    this.fill = 0;
  }

  push(samples, count = samples.length) {
    let i = 0;
    while (i < count) {
      const n = Math.min(count - i, this.frameSamples - this.fill);
      this.frame.set(samples.subarray(i, i + n), this.fill);
      this.fill += n;
      i += n;
      if (this.fill === this.frameSamples) {
        const out = this.frame;
        this.frame = new Float32Array(this.frameSamples);
        this.fill = 0;
        this.emit(out);
      }
    }
  }

  /** Zeroes a partly filled frame. */
  clear() {
    this.frame.fill(0);
    this.fill = 0;
  }
}
