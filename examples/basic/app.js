// CzechSpeech example: model download with progress, Start/Stop and a list of transcripts.
//
// Run it from the repository:  node scripts/serve.mjs        then open http://localhost:8790/examples/basic/
// With a local copy of the model instead of downloading it from Hugging Face:
//   node scripts/serve.mjs --model <path>/ggml-czechspeech-q4_0.bin
//   http://localhost:8790/examples/basic/?model=/models/ggml-czechspeech-q4_0.bin
// ?backend=cpu|webgpu|auto chooses the recogniser backend (default auto: WebGPU where the browser offers it).
//
// In an app, import from the package instead: import { CzechSpeech, CzechSpeechModel } from '@moravianai/czechspeech';
import { CzechSpeech, CzechSpeechModel } from '../../src/index.js';

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
// A mirror (here: the dev server) of the same file; the pinned size and SHA-256 are still enforced.
const source = params.get('model')
  ? { url: new URL(params.get('model'), location.href).href, size: CzechSpeechModel.DEFAULT.size, sha256: CzechSpeechModel.DEFAULT.sha256 }
  : undefined;

const backend = ['auto', 'webgpu', 'cpu'].includes(params.get('backend')) ? params.get('backend') : 'auto';

let model = null;
let speech = null;
let session = null;

$('env').textContent = self.crossOriginIsolated
  ? `Cross-origin isolated: multi-threaded WebAssembly (${navigator.hardwareConcurrency} logical cores).`
  : 'Not cross-origin isolated: single-threaded WebAssembly, several times slower. Serve the page with ' +
    'Cross-Origin-Opener-Policy: same-origin and Cross-Origin-Embedder-Policy: require-corp.';
if (!self.crossOriginIsolated) $('env').className = 'warn';

const mb = (b) => `${Math.round(b / 1e6)} MB`;

function setModelStatus(text) {
  $('model-status').textContent = text;
}

async function getModel(onProgress) {
  return source ? CzechSpeechModel.download({ source, onProgress }) : CzechSpeechModel.download({ onProgress });
}

async function loadModel() {
  $('download').disabled = true;
  $('progress').hidden = false;
  const t0 = performance.now();
  try {
    model = await getModel((done, total) => {
      $('progress').value = total ? done / total : 0;
      setModelStatus(`${mb(done)} of ${mb(total)}`);
    });
    setModelStatus(`ready (${mb(model.size)}, ${model.cached ? 'stored in this browser' : 'in memory only'}, ` +
      `${((performance.now() - t0) / 1000).toFixed(1)} s)`);
    $('start').disabled = false;
    $('delete').disabled = false;
    $('progress').hidden = true;
  } catch (e) {
    setModelStatus(`download failed: ${e.message}. Click again to resume.`);
    $('download').disabled = false;
  }
}

$('download').onclick = loadModel;

$('delete').onclick = async () => {
  if (session) session.abort();
  if (speech) await speech.close();
  speech = null;
  model = null;
  await CzechSpeechModel.delete(source);
  setModelStatus('deleted');
  $('start').disabled = true;
  $('delete').disabled = true;
  $('download').disabled = false;
};

function addLine(text, note, cls) {
  const li = document.createElement('li');
  li.textContent = text;
  if (note) {
    const small = document.createElement('small');
    small.textContent = note;
    li.append(small);
  }
  if (cls) li.className = cls;
  $('transcripts').append(li);
}

let finishing = false;

$('start').onclick = async () => {
  $('start').disabled = true;
  $('stop').disabled = false;
  finishing = false;
  // Aborted only when the model is deleted (discards); the Stop button finishes the session with speech.stop().
  session = new AbortController();
  try {
    speech ??= await CzechSpeech.create({ model, config: { backend } });
    $('indicator').textContent = 'listening…';
    for await (const ev of speech.listen({ signal: session.signal })) {
      if (ev.type === 'speechStarted') $('indicator').textContent = 'speech…';
      else if (ev.type === 'speechEnded') $('indicator').textContent = 'recognising…';
      else if (ev.type === 'transcript') {
        const on = speech.state.backend === 'webgpu' ? ' on the GPU (WebGPU)' : ' on the CPU';
        addLine(ev.text, `${(ev.startMs / 1000).toFixed(1)}–${(ev.endMs / 1000).toFixed(1)} s, recognised in ${(ev.processingMs / 1000).toFixed(1)} s${on}`);
        if (!finishing) $('indicator').textContent = 'listening…';
      } else if (ev.type === 'error') {
        addLine(`one utterance failed: ${ev.error.message}`, '', 'error');
      }
    }
  } catch (e) {
    if (e.name !== 'AbortError') addLine(`${e.code ? `${e.code}: ` : ''}${e.message}`, '', 'error');
  } finally {
    $('indicator').textContent = '';
    $('start').disabled = !model;
    $('stop').disabled = true;
    session = null;
  }
};

// Stop: the microphone closes, what was said is still transcribed, then the loop above ends by itself.
$('stop').onclick = () => {
  if (!session) return;
  if (!speech) { session.abort(); return; } // still creating it: nothing was captured yet
  finishing = true;
  speech.stop();
  $('stop').disabled = true;
  $('indicator').textContent = 'finishing…';
};

// A model stored on an earlier visit is ready without a download.
if (await CzechSpeechModel.isDownloaded(source)) await loadModel();
else setModelStatus('not downloaded yet');
