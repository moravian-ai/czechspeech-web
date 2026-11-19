// Speech probabilities of the test meetings through the SDK's own path: the WebAssembly Silero VAD module (single
// thread, as in the capture worker), one probability per 512-sample frame, with the streaming reset after every end of
// speech that the Endpointer reports (exactly what Capture does in src/worker-core.js). Writes <id>.f32 (little-endian
// Float32) into the output directory, which must be under test/speakers/out/ (git-ignored).
// Usage: node test/speakers/sdk-vad.mjs <meetings directory> <output directory>
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { loadWasmEngine } from '../../src/engine-wasm.js';
import { Endpointer, FRAME_SAMPLES } from '../../src/endpointer.js';
import { DEFAULT_CONFIG } from '../../src/config.js';
import { ROOT, TestEnv, readWav } from '../support/env.mjs';
import { s4TestIds } from './meetings.mjs';

const [meetingsDir, outDir] = process.argv.slice(2);
if (!meetingsDir || !outDir) {
  console.error('usage: node test/speakers/sdk-vad.mjs <meetings directory> <output directory>');
  process.exit(2);
}
const out = path.resolve(outDir);
if (!out.startsWith(path.join(ROOT, 'test', 'speakers', 'out') + path.sep)) {
  console.error(`the output directory must be under test/speakers/out/ (git-ignored): ${out}`);
  process.exit(2);
}
fs.mkdirSync(out, { recursive: true });

const engine = await loadWasmEngine({
  moduleUrl: pathToFileURL(path.join(ROOT, 'build', 'wasm-node', 'czechspeech-st-node.js')).href,
  threads: 1,
  log: () => {},
});
const vad = engine.vadInit(new Uint8Array(fs.readFileSync(TestEnv.vadAsset)));
let total = 0;
for (const id of s4TestIds(meetingsDir)) {
  const audio = readWav(path.join(meetingsDir, `${id}.wav`));
  const frames = Math.floor(audio.length / FRAME_SAMPLES);
  const probs = new Float32Array(frames);
  engine.vadReset(vad);
  const ep = new Endpointer({ ...DEFAULT_CONFIG }, (ev) => {
    if (ev.kind === 'speechEnded') engine.vadReset(vad);
  });
  const frame = new Float32Array(FRAME_SAMPLES);
  for (let i = 0; i < frames; i++) {
    frame.set(audio.subarray(i * FRAME_SAMPLES, (i + 1) * FRAME_SAMPLES));
    probs[i] = engine.vadFrame(vad, frame);
    ep.accept(frame, probs[i]);
  }
  ep.finish();
  fs.writeFileSync(path.join(out, `${id}.f32`), Buffer.from(probs.buffer));
  total += frames;
  console.log(`${id}: ${frames} frames`);
}
engine.vadFree(vad);
console.log(`done: ${total} frames`);
process.exit(0);
