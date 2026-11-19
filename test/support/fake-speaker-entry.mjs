// Worker entry for the unit tests: the speaker worker core with a fake embedder (no WebAssembly, no model).
//
// The "voiceprint" of an array is v[0] = share of samples above 0.05, v[1] = share below -0.05, the rest 0, normalised
// (192 numbers), so DC levels make distinguishable "voices". Options come from `init._fake`:
//   delayMs       time an embedding takes (setTimeout: the worker stays responsive)
//   busyMs        time an embedding takes as a synchronous busy-wait (like the real runtime: no message is handled meanwhile)
//   reportLengths every embedding posts { type: 'fake', length } (samples it was given)
//   neverReady    the embedder never finishes loading (init never answers)
//   failFirst     the first n embeddings throw (a failure of one request, not of the worker)
//   crashAt       the n-th utterance throws out of the message handler (the worker dies: onerror on the page)
//   trackOut      remember every vector the embedder returns; a { type: 'fakeCheck' } message (sent to the worker by the test)
//                 answers { type: 'fake', out: { count, zero } }: how many are kept and whether all are zero by now
//   reportZeroed  after close(), the fake posts { type: 'fake', zeroed } : true when every array it was given is zero
import { installSpeakerWorker } from '../../src/speakers/worker-core.js';

let options = {};
let utterances = 0;
const held = [];
const heldOut = [];

installSpeakerWorker(self, {
  async loadEmbedder() {
    if (options.neverReady) await new Promise(() => {});
    return {
      threads: 1,
      async embed(samples) {
        if (options.failFirst && options.failFirst-- > 0) throw new Error('fake embedding failure');
        if (options.reportZeroed) held.push(samples);
        if (options.reportLengths) self.postMessage({ type: 'fake', length: samples.length });
        if (options.busyMs) { const t = performance.now(); while (performance.now() - t < options.busyMs) { /* busy */ } }
        if (options.delayMs) await new Promise((r) => setTimeout(r, options.delayMs));
        let pos = 0;
        let neg = 0;
        for (const x of samples) {
          if (x > 0.05) pos++;
          else if (x < -0.05) neg++;
        }
        const v = new Float32Array(192);
        const n = samples.length || 1;
        v[0] = pos / n;
        v[1] = neg / n;
        const norm = Math.hypot(v[0], v[1]) || 1;
        v[0] /= norm;
        v[1] /= norm;
        if (options.trackOut) heldOut.push(v);
        return v;
      },
      release() {
        if (options.reportZeroed) self.postMessage({ type: 'fake', zeroed: held.every((a) => a.every((x) => x === 0)) });
      },
    };
  },
});

// Options travel with init; also count utterances for crashAt. Wrap the core's handler.
const core = self.onmessage;
self.onmessage = (e) => {
  const m = e.data;
  if (m && m.type === 'init') options = m._fake || {};
  if (m && m.type === 'fakeCheck') {
    self.postMessage({ type: 'fake', out: { count: heldOut.length, zero: heldOut.every((a) => a.every((x) => x === 0)) } });
    return;
  }
  if (m && m.type === 'utterance' && options.crashAt && ++utterances === options.crashAt) throw new Error('fake crash');
  core(e);
  // utterances that come on an attached port count for crashAt as well
  if (m && m.type === 'attach' && m.port && m.port.onmessage) {
    const handler = m.port.onmessage;
    m.port.onmessage = (ev) => {
      if (ev.data && ev.data.type === 'utterance' && options.crashAt && ++utterances === options.crashAt) throw new Error('fake crash');
      handler(ev);
    };
  }
};
