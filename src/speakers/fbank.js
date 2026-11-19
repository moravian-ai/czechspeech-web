// Kaldi log-mel filter bank, the front end the voiceprint model was trained with (sherpa-onnx settings):
// 80 bins, 25 ms frames every 10 ms, snip_edges = false (edges mirrored), Povey window, pre-emphasis 0.97,
// per-frame mean removal, 512-point FFT, 20 Hz .. 7600 Hz, no dither. `fbank` additionally subtracts the mean of
// every bin over the frames ("global-mean", feature_normalize_type of the model).

export const FBANK = Object.freeze({
  sampleRate: 16000,
  bins: 80,
  frameSamples: 400,
  shiftSamples: 160,
  fftSize: 512,
  lowHz: 20,
  highHz: 7600,
  preemphasis: 0.97,
  windowPower: 0.85,
  logFloor: 1.1920929e-7,
});

const { sampleRate: SR, bins: NBINS, frameSamples: FRAME_LEN, shiftSamples: SHIFT, fftSize: NFFT } = FBANK;
const HALF = NFFT / 2;

const mel = (f) => 1127 * Math.log(1 + f / 700);

const WINDOW = Float64Array.from({ length: FRAME_LEN },
  (_, i) => Math.pow(0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (FRAME_LEN - 1)), FBANK.windowPower));

const MEL_BANK = (() => {
  const binWidth = SR / NFFT;
  const lowMel = mel(FBANK.lowHz);
  const highMel = mel(FBANK.highHz);
  const delta = (highMel - lowMel) / (NBINS + 1);
  const banks = [];
  for (let b = 0; b < NBINS; b++) {
    const left = lowMel + b * delta;
    const centre = lowMel + (b + 1) * delta;
    const right = lowMel + (b + 2) * delta;
    const w = new Float64Array(HALF);
    for (let k = 0; k < HALF; k++) {
      const m = mel(binWidth * k);
      if (m > left && m < right) w[k] = m <= centre ? (m - left) / (centre - left) : (right - m) / (right - centre);
    }
    banks.push(w);
  }
  return banks;
})();

const COS = new Float64Array(HALF);
const SIN = new Float64Array(HALF);
for (let i = 0; i < HALF; i++) {
  COS[i] = Math.cos((2 * Math.PI * i) / NFFT);
  SIN[i] = -Math.sin((2 * Math.PI * i) / NFFT);
}
const REV = new Uint16Array(NFFT);
for (let i = 0, bits = Math.log2(NFFT); i < NFFT; i++) {
  let r = 0;
  for (let b = 0; b < bits; b++) if (i & (1 << b)) r |= 1 << (bits - 1 - b);
  REV[i] = r;
}

function fft(re, im) {
  for (let i = 0; i < NFFT; i++) {
    const j = REV[i];
    if (j > i) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= NFFT; len <<= 1) {
    const half = len >> 1;
    const step = NFFT / len;
    for (let i = 0; i < NFFT; i += len) {
      for (let k = 0; k < half; k++) {
        const wr = COS[k * step];
        const wi = SIN[k * step];
        const a = i + k;
        const b = a + half;
        const xr = re[b] * wr - im[b] * wi;
        const xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr; im[b] = im[a] - xi;
        re[a] += xr; im[a] += xi;
      }
    }
  }
}

/** Log mel energies without mean subtraction: frame by frame, 80 values each. */
export function logMel(samples) {
  const n = samples.length;
  const frames = Math.floor((n + SHIFT / 2) / SHIFT);
  const out = new Float32Array(frames * NBINS);
  const fr = new Float64Array(FRAME_LEN);
  const re = new Float64Array(NFFT);
  const im = new Float64Array(NFFT);
  const power = new Float32Array(HALF);
  for (let t = 0; t < frames; t++) {
    const start = t * SHIFT + SHIFT / 2 - FRAME_LEN / 2;
    let mean = 0;
    for (let i = 0; i < FRAME_LEN; i++) {
      let j = start + i;
      // reflect as often as needed (Kaldi does): audio shorter than a frame is mirrored several times at its edges
      while (j < 0 || j >= n) j = j < 0 ? -j - 1 : 2 * n - 1 - j;
      fr[i] = samples[j];
      mean += fr[i];
    }
    mean /= FRAME_LEN;
    for (let i = 0; i < FRAME_LEN; i++) fr[i] -= mean;
    for (let i = FRAME_LEN - 1; i > 0; i--) fr[i] -= FBANK.preemphasis * fr[i - 1];
    fr[0] -= FBANK.preemphasis * fr[0];
    re.fill(0);
    im.fill(0);
    for (let i = 0; i < FRAME_LEN; i++) re[i] = fr[i] * WINDOW[i];
    fft(re, im);
    for (let k = 0; k < HALF; k++) power[k] = re[k] * re[k] + im[k] * im[k];
    for (let b = 0; b < NBINS; b++) {
      let s = 0;
      const w = MEL_BANK[b];
      for (let k = 0; k < HALF; k++) s += w[k] * power[k];
      out[t * NBINS + b] = Math.log(Math.max(s, FBANK.logFloor));
    }
  }
  return { data: out, frames };
}

/** Log mel energies with the mean of every bin subtracted: the model input. */
export function fbank(samples) {
  const r = logMel(samples);
  const { data, frames } = r;
  for (let b = 0; b < NBINS; b++) {
    let m = 0;
    for (let t = 0; t < frames; t++) m += data[t * NBINS + b];
    m /= frames;
    for (let t = 0; t < frames; t++) data[t * NBINS + b] -= m;
  }
  return r;
}
