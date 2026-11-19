# CzechSpeech for the web

Offline Czech speech-to-text for web apps. CzechSpeech is Whisper large-v3-turbo fine-tuned on Czech by
[Moravian AI](https://moravian.ai/czechspeech/), quantised to 4 bits (474 MB). It runs in the browser through
[whisper.cpp](https://github.com/ggml-org/whisper.cpp) compiled to WebAssembly, **on the GPU through WebGPU** where
the browser offers it and on the CPU otherwise, with [Silero VAD](https://github.com/snakers4/silero-vad) to find
speech.

You install one package, iterate `listen()` and get Czech transcripts. To stop, leave the loop. **Audio never leaves
the page**: it is recognised in Web Workers on the user's machine and is never stored or sent anywhere.

Version 0.3.0, the first public release (a 0.2.0 was prepared but never published): the web counterpart of CzechSpeech for Android 0.3.0
(same model, same speech detection, same results), with a WebGPU backend (see [Backend: WebGPU or CPU](#backend-webgpu-or-cpu)) and
speaker recognition on the device (see [Speakers](#speakers)). See [What has been tested](#what-has-been-tested).

## Install

```
npm install @moravianai/czechspeech
```

```js
import { CzechSpeech, CzechSpeechModel } from '@moravianai/czechspeech';
```

The package is plain ES modules with TypeScript declarations and no dependencies. Besides `src/index.js` it ships
two worker modules (`worker.js`, `worklet.js`), three WebAssembly modules (`wasm/czechspeech-mt.*` and
`wasm/czechspeech-st.*` for the CPU, 1.3 MB each, and `wasm/czechspeech-webgpu.*` for the GPU, 3.2 MB; a browser
downloads only the ones it uses) and the Silero VAD model (`assets/ggml-silero-v6.2.0.bin`, 885 kB). The library
finds them with `new URL('./…', import.meta.url)`, a pattern that bundlers recognise. If yours does not copy them,
serve the package's `src/` directory as static files and pass its URL:

```js
await CzechSpeech.create({ model, assetBaseUrl: '/static/czechspeech/' });
```

Browsers start module workers only from the page's own origin, so `assetBaseUrl` must be on that origin (not a CDN);
a worker that cannot be started fails the call with a `CzechSpeechError` with code `runtime` (the browser's error is
its `cause`). Where Web Workers are not available at all (for example during server-side rendering, or in Node.js),
`CzechSpeech.create()` fails with code `unsupported`; importing the package works everywhere.

Requirements: a secure context (https or localhost) for the microphone and the model cache, and **about 1.2 GB of
memory** for the recogniser while a model is loaded (474 MB of weights plus whisper.cpp's buffers): on the CPU all of
it in the WebAssembly heap, on WebGPU about 520 MB of heap plus 830 MB of GPU buffers.

## Required headers: cross-origin isolation

Multi-threaded WebAssembly needs `SharedArrayBuffer`, which browsers only give to cross-origin isolated pages.
Serve the page **and all of its scripts, workers and the .wasm files** with:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

With `require-corp`, every cross-origin resource of the page must allow it (CORS, or `Cross-Origin-Resource-Policy`);
`Cross-Origin-Embedder-Policy: credentialless` is an alternative in Chromium and Firefox. Examples:

```
# Cloudflare Pages / Netlify: _headers
/*
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp
```

```nginx
add_header Cross-Origin-Opener-Policy same-origin always;
add_header Cross-Origin-Embedder-Policy require-corp always;
```

```js
// vite.config.js
export default { server: { headers: { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' } } };
```

Check with `self.crossOriginIsolated` (or `speech.state.runtime`).

**Without these headers it still works**: WebGPU does not need them, and on the CPU the library picks the
single-threaded WebAssembly module, several times slower (see [Speed and memory](#speed-and-memory)). A recognition
in progress cannot be interrupted then, so cancelling terminates the recogniser worker instead.

## Backend: WebGPU or CPU

```js
await CzechSpeech.create({ model, config: { backend: 'auto' } });   // 'auto' (default) | 'webgpu' | 'cpu'
```

The recogniser (Whisper) runs on the GPU through WebGPU when it can and on the CPU (WebAssembly) otherwise. With
`'auto'` the recogniser worker uses WebGPU only if all of these hold, checked inside the worker itself:

1. the browser has WebAssembly JSPI (`WebAssembly.Suspending`): the WebGPU module waits for the GPU through it;
2. `navigator.gpu` exists in the worker and `requestAdapter()` gives a hardware adapter (not a software fallback)
   with the `shader-f16` feature;
3. the WebGPU module (`czechspeech-webgpu`) starts: ggml requests the device and builds its first pipeline;
4. after the model is loaded, its weights are really in GPU memory.

Otherwise the CPU modules run, and the SDK says so once per instance with `console.info`
(`CzechSpeech: WebGPU not used (requestAdapter() gave no adapter); recognising on the CPU (…)`), or through your
`log` callback at level `info`. Nothing else changes for the app: the same events and, in our tests, the same
transcripts (see [Speed and memory](#speed-and-memory)).

- **`'webgpu'`** insists on the GPU: without it `listen()` throws and `transcribe()` rejects with a
  `CzechSpeechError` with code `unsupported`.
- **`'cpu'`** never touches `navigator.gpu`.
- **Speech detection always runs on the CPU** (the capture worker's Silero VAD, the endpointer and the speech gate
  before Whisper), so what counts as speech, and the hallucination gate, are identical on both backends. WebGPU does
  not need cross-origin isolation (its module is single-threaded).
- **Which GPU:** the one the browser gives to `requestAdapter()`. On a laptop with an integrated and a discrete GPU,
  Chromium on Windows gave the integrated one in our tests (even when asked for `high-performance`); started with
  `--force_high_performance_gpu` it gave the discrete one (a browser setting, not something a page can ask for).
- **A recognition that takes too long on WebGPU** (more than 20 s plus 3 s per second of audio; with the
  [decoding limits](#decoding-limits) a healthy GPU stays far below that) is stopped at whisper.cpp's next abort check,
  as a cancelled one is: that utterance gives no transcript (`transcribe()` resolves `''`), `abortedRecognitions` counts
  it, a warning is logged, and the WebGPU module and its device stay in use for the next utterance. Only a recognition
  that does not stop within 5 s more is treated as a WebGPU failure.
- **When WebGPU fails mid-session** (the device is lost, the driver resets, ggml aborts, or a call cannot be
  stopped): with `'auto'` the utterance is recognised again on the CPU module in the same worker, so it still gives
  its transcript (late: in our tests a 7 s utterance took 25 to 31 s, as the fresh CPU module ran its first
  recognition in V8's baseline tier; every fallback starts a fresh CPU module, so a later one is as slow as the
  first), and a warning is logged. WebGPU is tried again once a retry delay has passed (10 s after the first
  failure), with no timer: nothing is tried while no `listen()` session or `transcribe()` call is running. The try
  comes in one of two ways:
  - A recogniser that is still running (during a `listen()` session, or kept by `keepModelLoaded: true`) starts the
    WebGPU module again with its next utterance or `transcribe()` call after the delay, before the speech gate, and
    releases the CPU module at once: its pthread workers are terminated and its memory (about 1.2 GB with the model
    loaded) is no longer reachable, so the garbage collector frees it (checked in the browser tests). It loads the
    model onto the GPU with the first utterance that has speech; when that is the utterance that started the module,
    it pays both (in the browser tests `e-rec` took 4.9 and 7.1 s, 3.1 and 3.7 s of it loading the model).
    `state.backend` is `'webgpu'` again.
  - A recogniser started after the delay starts on WebGPU, as the first one did: its start-up includes the start of the
    WebGPU module; `listen()` then loads the model when the session starts, `transcribe()` with its first recognition.

  If WebGPU fails again before a recognition succeeds on it, or does not start, the next try waits 60 s, then 300 s;
  after the fourth failure in a row the instance stays on the CPU. This state belongs to the `CzechSpeech` instance,
  not to a recogniser. With the default `keepModelLoaded: false` the recogniser is released when the last
  `listen()` session or `transcribe()` call using it ends (calls that overlap share one), and the next call starts a
  new one; one started before the retry time starts on the CPU without trying WebGPU. If the CPU module itself does
  not start after a WebGPU failure, a `transcribe()` call rejects with code `runtime` and a `listen()` utterance is
  an `error` event (code `recognition`), both with a message naming the two failures (a `listen()` session that was
  loading the model at its start ends with code `runtime`); the next utterance or call starts the CPU module again,
  or WebGPU once its retry time has come.
  With `'webgpu'` that utterance is an `error` event and the WebGPU module is started again for the next one (if it
  cannot start or its speech gate does not load, the session ends with code `unsupported`, and a `transcribe()` call
  rejects with it). Nothing waits for a broken GPU: every GPU call runs under a watchdog and fails at once when ggml
  aborts or the device is lost.
- **`speech.state`** tells what happened: `backend` (`'webgpu'`, `'cpu'`, or `null` before the recogniser first
  started), `gpuAdapter` (`{ vendor, architecture, device, description }` as the browser reports it),
  `backendFallback` (why `'auto'` did not use WebGPU) and `gpuBytes`/`peakGpuBytes` (GPU memory of the recogniser).

## The model

The model is downloaded once and kept in the browser's Cache Storage of your origin:

```js
const model = await CzechSpeechModel.download({
  onProgress: (done, total) => { progress.value = done / total; },   // about every MiB
});
```

- **Pinned:** it comes from a fixed revision of [`moravianai/CzechSpeech`](https://huggingface.co/moravianai/CzechSpeech)
  on Hugging Face (`CzechSpeechModel.DEFAULT`: 473,992,235 bytes, SHA-256 `04ead666…30a3`), the same file as the
  Android SDK. Hugging Face and its CDN send CORS headers that work from a cross-origin isolated page.
- **Verified:** size and SHA-256 are checked while the file streams in; a mismatch deletes it.
- **Resumable:** it is stored in 16 MiB parts; an interrupted or aborted download (`signal`) continues on the next
  call with an HTTP Range request.
- **Idempotent:** a stored model is returned at once, without network and without hashing.
- **Mirrorable:** serve the same file yourself; the pinned size and SHA-256 are still enforced:
  `CzechSpeechModel.download({ source: { url: 'https://models.example.com/czechspeech-q4_0.bin' } })`.
  `isDownloaded(source)` and `delete(source)` read a source the same way. A source with a `sha256` of its own is
  another file, verified against that hash (as with `fromUrl()`).

Also: `CzechSpeechModel.fromUrl(url, { sha256?, size?, languages?, tokensPerSecond? })` for another file, `CzechSpeechModel.fromBytes(arrayBufferOrBlob,
{ sha256?, languages?, tokensPerSecond? })` if you ship the model your own way, `CzechSpeechModel.isDownloaded()` (cheap) and
`CzechSpeechModel.delete()`. `fromUrl()` caches the file only with `sha256`; `isDownloaded()` and `delete()` then find it
by `{ url, sha256 }`. Without `sha256` the file is downloaded into memory each time, as nothing could find a cached copy
again: a URL alone means the pinned model to `isDownloaded(url)` and `delete(url)`.

Storage: browsers grant an origin a share of the free disk space (Chrome and Edge up to 60 % of the disk, Firefox
up to 10 %, Safari asks or limits); 474 MB fits on most machines. Storage can be evicted under pressure unless you
call `navigator.storage.persist()`. Without Cache Storage (an insecure context) the model is downloaded into
memory for the page's lifetime.

## Models and languages

Models are picked by id from a catalog:

```js
const model = await CzechSpeechModel.download({ id: 'cs' });   // the default: the Czech model above
model.id;               // 'cs'
model.languages;        // ['cs']: the languages it speaks
model.tokensPerSecond;  // 25: the cap of tokens per second of audio (see "Decoding limits")
```

- `MODELS` lists the catalog and `DEFAULT_MODEL_ID` is `'cs'`. An unknown id fails at once, when you ask for it, with
  a `CzechSpeechError` of code `'unknown-model'`. `id` and `source` are alternatives (both is a `TypeError`).
  `isDownloaded({ id })` and `delete({ id })` take an id too. A file stored by an earlier version is found without
  downloading again (the cache is keyed by SHA-256, which has not changed).
- A language is a whisper language code: two or three lower-case letters (`'cs'`, `'pl'`, `'en'`, `'haw'`). A model
  from your own URL or bytes (`fromUrl()`, `fromBytes()`) says what it speaks with `languages` (default `['cs']`,
  never empty) and `tokensPerSecond` (a positive whole number, default 25); such a model has `id` `null`.
- **Forcing a language per call.** `listen({ language })` and `transcribe(samples, { language })` force one of the
  model's `languages`. Another code is a `RangeError` (`listen()` throws it from the first `next()`); nothing is
  switched silently. Without `language`, the model decides: a model with one language (Czech) is forced to it exactly
  as before, a model with several detects the language among its own list, never outside it.
- **The language of each transcript** is `ev.language` on the `transcript` events of `listen()`, a code from the
  model's list. `transcribe()` still resolves with the text only.
- **What detection costs.** A forced language costs nothing extra. Choosing among several languages makes
  whisper.cpp run the encoder once more over a full 30 s window plus one decoder step before it decodes, and the
  recogniser starts a fresh state for it, so every utterance of a multi-language model pays that again, whatever
  its length (short utterances most, relatively). Measured once, on one machine, with the current model and decoder (the CzechSpeech q4_0 model declared as cs + en, 5 repetitions, 2 to 13.9 s of audio, an idle 22-logical-core notebook with an Intel Xe-LPG GPU): detection adds about 54 s per utterance on the CPU (WebAssembly, 6 threads; a 2 s utterance takes 5 s forced and 60 s with detection) and about 4.3 s on WebGPU (1.1 s forced, 6.2 s with detection), independent of the utterance length, because it always encodes a full 30 s window; the detection overhead is the same on the first call of a session as on later ones (the first call itself is slower for start-up, e.g. 2.7 s against 1.1 s for a forced 2 s utterance on WebGPU). The detected transcripts were word for word those of the forced runs, and over 24 consecutive detect recognitions on WebGPU the GPU buffers stayed flat at 833.4 MiB in 13 buffers, none left after close(). On the CPU a multi-language model is therefore impractical for now; `CZECHSPEECH_E2E_DETECT_COST=1 node test/e2e/detect-cost.mjs` measures it on your machine (opt-in, see [Building from source](#building-from-source)). Force `language` when the app knows it.
  Czech (one language) pays nothing.

### Your own or a licensed model

The public catalogue holds the free Czech model only. A model that is not in it, for example a model of several languages
that a company licensed from Moravian AI (the Czech, Slovak and English enterprise model is gated on Hugging Face and
its files go only to companies with a licence), is a file you host yourself: on your server, your CDN or in
your app. You pass its URL and SHA-256 to the SDK, which verifies the file while it streams in and deletes it on a
mismatch. No licence server is involved and nothing is sent to us.

```js
const model = await CzechSpeechModel.fromUrl('https://models.example.com/my-model.bin', {
  sha256: '<64 hex digits of your file>',   // verified; with it the file is also cached and resumable
  size: 474000000,                          // optional: the expected size in bytes
  languages: ['cs', 'sk', 'en'],            // what the model speaks; the default is ['cs']
  tokensPerSecond: 25,                      // the decoding cap, see "Decoding limits"
});
const speech = await CzechSpeech.create({ model });
```

There is no language rule on the web yet: a model of several languages picks its language in one pass among its
`languages`, and you force one with `language` per call (see above). The enterprise model and its licence are available
on request at info@moravian.ai.

#### Encrypting the hosted file

If the file sits at a public URL that you do not want to advertise, host it encrypted so that a found URL is useless
without the key, and pass the key with the source. The SDK decrypts while the file streams in (chunk by chunk with
WebCrypto, resume with `Range` works), the cache holds the plain model, and `sha256` is the hash of the plain model,
verified after decryption as always; a wrong key is a "SHA-256 mismatch" and nothing is kept. Resuming an interrupted download needs `size`.

```js
const model = await CzechSpeechModel.fromUrl('https://models.example.com/my-model.bin.enc', {
  sha256: '<64 hex digits of the PLAIN model>',   // required with encryption
  size: 474000000,                                 // optional; encrypted and plain files have the same size
  encryption: { algorithm: 'AES-256-CTR', key: '<64 hex digits>', iv: '<32 hex digits>' },
});
```

The format is AES-256-CTR over the whole file: the ciphertext is as long as the plaintext, the 16-byte IV is the initial
counter block, incremented as one 128-bit big-endian integer per 16-byte block (what WebCrypto's `{ name: 'AES-CTR',
length: 128 }`, Java's `AES/CTR/NoPadding` and OpenSSL do). There is no MAC; the plaintext SHA-256 is the integrity
check. **Never encrypt two different files with the same key and IV** (a new version of a model gets a new key, or at
least a new IV): the two ciphertexts would reveal the XOR of the two plaintexts. The key and the IV belong to the hosted
file: publish a re-encrypted file under a new URL, and never replace the file behind a URL that released apps use. To
encrypt with OpenSSL, keeping the key and the IV in a file only you can read:

```
key=$(openssl rand -hex 32); iv=$(openssl rand -hex 16)
openssl enc -aes-256-ctr -K "$key" -iv "$iv" -in model.bin -out model.bin.enc
(umask 077; printf 'key=%s\niv=%s\n' "$key" "$iv" > model.key)
```

The Android SDK repository (github.com/moravian-ai/czechspeech-android) has `scripts/encrypt-model.sh` for this, which also
streams (`-` for stdin and stdout) and checks the result against the hash and size you know. Run it on a machine you do
not share, as OpenSSL takes the key on its command line. The length of the encrypted file reveals the size of the model. Decryption needs `crypto.subtle`, which browsers offer only in a secure context (https or localhost); without it
the download fails with the error code `'unsupported'`. The key is never put into an error message.

**What this protects, exactly:** the file on your server or CDN. Someone who finds the URL gets noise. It does not
protect the model in the browser: the decrypted model is in the page's Cache Storage and the key is part of your page, so
anyone who opens the developer tools can get both. It is access control for hosting, not copy protection.

## Listen

```js
const speech = await CzechSpeech.create({ model });     // cheap: nothing is loaded yet

stopButton.onclick = () => speech.stop();   // finishes: the last utterance is still transcribed, then the loop ends
const leave = new AbortController();        // leave.abort() discards instead (e.g. when the user leaves the page)
try {
  for await (const ev of speech.listen({ signal: leave.signal })) {   // opens the microphone
    if (ev.type === 'speechStarted') showListening();
    else if (ev.type === 'speechEnded') showRecognising();
    else if (ev.type === 'transcript') append(ev.text);
    else if (ev.type === 'error') console.warn('one utterance failed', ev.error);
  }
} catch (e) {
  if (e.name !== 'AbortError') showError(e);   // e.code: 'microphone-denied', 'model', ...
}
```

If you only want the text, iterate `speech.transcripts()`.

How a session behaves:

- **Cold and single.** Iterating starts a session; one session at a time per `CzechSpeech` (a second one throws
  a `CzechSpeechError` with code `busy`). Call `listen()` from a user gesture, so that the audio context may start.
- **The recogniser starts with the microphone; the model loads with the first utterance on the CPU, at once on
  WebGPU.** The recogniser worker and its WebAssembly start in parallel with `getUserMedia`, the backend is chosen,
  and the file header is checked at once. On the CPU the model itself (about 1 to 1.3 s from Cache Storage) is
  loaded when the first utterance is complete. This differs from the Android SDK on purpose: loaded in a task of
  its own, V8 often left the first recognition in its baseline compiler tier, 2.7 times slower; and a CPU session
  without speech never loads the model. On WebGPU the encoder runs on the GPU, that effect does not occur, and the
  model (1.2 to 2 s including the upload to the GPU) is loaded while the person speaks, so the first transcript comes
  that much sooner.
- **Order of events.** Each `speechStarted` is followed by a `speechEnded`. The `transcript` comes after its
  `speechEnded`, because recognition starts once the utterance is complete. No transcript follows if the speech
  gate or the recogniser found no words. Speech longer than `maxUtteranceMs` gives several transcripts.
- **`speechStarted` can fire on noise.** The speech gate then stops it before it produces text.
- **Fatal errors are thrown** from the loop: `microphone-denied`, `microphone-unavailable`, `microphone-ended`
  (the device went away), `model`, `runtime`. An `error` event reports one utterance whose recognition failed; the
  session continues.
- **Stopping with `stop()`.** Use this for a Stop button. It finishes the word in progress: if someone is
  speaking, the microphone stays on until a short pause (256 ms), at most 1 s, and that audio is part of the
  utterance (cut off mid-word, Whisper tends to invent an ending); without speech it stops delivering audio at
  once. The utterance in progress then ends as if silence followed, and everything still waiting is transcribed and
  yielded. Then the loop ends normally (nothing is thrown) and the model is freed. `stop()` returns immediately and
  does nothing without an active session; it never stops a later one. Called while the session is still starting,
  it ends the session as well, without waiting for a pause: audio the microphone already captured is still
  transcribed, and a microphone that was not open yet is closed as soon as it opens.
- **Cancelling** (`break`, `return`, aborting `signal`, or `speech.close()`): the microphone closes at once and
  speech not yet recognised is discarded. A recognition in progress is stopped at whisper.cpp's next check (after
  the encoder pass, after each decoder step); if that takes more than 1 s, the recogniser worker is terminated.
  Workers that are still starting (a slow WebGPU start, a stalled download of a module) are not waited for: they
  are terminated at once. Then the model is freed and the loop exits. Aborting throws `signal.reason`; `close()` throws a
  `CzechSpeechError` with code `closed`; `break` just ends the loop.
- **A worker that does not start.** A capture or recogniser worker that has not finished starting after 60 s (the
  recogniser's start includes trying WebGPU, whose own limit is 30 s) is terminated, and the session or call that
  waited for it fails with a `CzechSpeechError` with code `runtime`; the next session or call starts a new worker.
- **Timestamps.** `startMs` and `endMs` are milliseconds of audio since the session started; `processingMs` is
  the time spent on recognition.
- **Model memory.** The model is freed at the end of every session and `transcribe()` call: the recogniser worker
  zeroes its GPU buffers and its WebAssembly heap and is terminated, which returns the memory to the system. With
  `CzechSpeech.create({ model, keepModelLoaded: true })` it stays loaded until `close()`, which saves the reload of
  every session (about 1.5 s on the CPU: model 1.3 s plus worker start; 2 to 4 s on WebGPU, which also uploads it
  to the GPU and builds its shader pipelines again) at the price of keeping 1.2 to 1.35 GB. Recommended for
  dictation on WebGPU, see [Speed and memory](#speed-and-memory).

### Microphone settings

`listen()` asks for `DEFAULT_AUDIO_CONSTRAINTS`:

| constraint | value | why |
|---|---|---|
| `channelCount` | 1 | one voice; the worklet downmixes anyway |
| `echoCancellation` | `true` | what the page or the system plays (read-back, notification sounds, a video) is not transcribed; with nothing playing it is close to transparent |
| `noiseSuppression` | `false` | the recogniser was trained on noisy speech; suppression artefacts hurt it more than the noise. The Android SDK records the same way (VOICE_RECOGNITION) |
| `autoGainControl` | `false` | gain pumping raises the background between words, which keeps the speech detector on longer |

Override any of them, or pick a device: `speech.listen({ audio: { deviceId: { exact: id }, autoGainControl: true } })`.
Or bring your own stream: `speech.listen({ stream })` (it is not stopped at the end of the session).

The audio context runs at the device's native rate (44.1 or 48 kHz usually); the worklet resamples to 16 kHz
itself with a windowed-sinc filter, because a 16 kHz `AudioContext` fed by a microphone does not work in every
browser.

## Speakers

`listen()` can also tell **who** spoke: "Speaker 1", "Speaker 2" and so on, on the device. It is optional and off unless you
pass a `SpeakerSession`. The SDK only computes; names and anything stored are the app's business. **Turn speakers on only after everyone whose voice will be processed has agreed (show a consent screen): a voiceprint is personal data, and biometric data when it is used to recognise a person. The SDK keeps voiceprints in the page's worker memory only, never stores, sends or logs them, and zeroes them on `close()`; it does not check consent, that is the app's duty.** A voiceprint
is a row of numbers worked out from how a voice sounds; this SDK never stores one, never sends one and never writes
one to a log, and it does not look at emotion or tone.

```js
import { CzechSpeech, SpeakerSession, VoiceprintModel, speakerSupport } from '@moravianai/czechspeech';

if (speakerSupport().supported) {
  // the voiceprint model is not public: your app serves the file and says where (size and SHA-256 are pinned)
  const voices = await VoiceprintModel.download({ url: '/models/voiceprint/<sha256>.onnx', onProgress });
  const speakers = await SpeakerSession.create({ model: voices });          // one per meeting
  for await (const ev of speech.listen({ speakers })) {
    if (ev.type === 'transcript') append(ev.utteranceId, ev.text);
    else if (ev.type === 'speakers') label(ev.utteranceId, ev.speakers);    // [{ kind: 'session', n: 2 }, ...] in speaking order
  }
  // pause, then listen({ speakers }) again with the same session: "Speaker 2" stays "Speaker 2"
  await speakers.close();                                                   // zeroes the centres, ends the worker
}
```

- **One `speakers` event follows each transcript** with the same `utteranceId`, never before it, even when the speaker
  worker is faster: the transcript never waits for the speakers. An utterance without a transcript has no
  `speakers` event. `parts` are the stretches of the utterance with the speaker found for each (`null`: not sure);
  `speakers` lists the distinct ones in the order they spoke. The text of one utterance is not divided between speakers.
- **The numbers belong to the meeting, not to one `listen()`.** The same `SpeakerSession` across `stop()` and the next
  `listen()` goes on numbering. A label does not change once delivered: nothing is regrouped afterwards. After a recovery
  pass `firstLabel` (the next free number) so that new speakers do not take numbers already in use.
- **Speakers can be skipped, and the event says why** (`skipped`): `'cpu'` when the recogniser runs on the CPU and the
  session was made with the default `onCpu: 'skip'` (the speaker worker would compete with the recogniser for the
  processor; `onCpu: 'run'` computes anyway), `'behind'` when the speaker worker had too much waiting (it first stops
  looking for speaker changes within an utterance, then skips utterances), `'error'` when this utterance failed (the speaker
  worker lives on and the next utterance is handled), `'failed'` when the speaker worker is dead or the session was closed
  (nothing more comes from it until a new `SpeakerSession`). The transcripts are not affected by any of it. `state.speakers` has the counts.
- **`stop()` waits for the last speakers** for at most 10 s; what does not come in time is not delivered and is counted in
  `state.speakers.lost`. A dead speaker worker does not block the end: it is reported as `'failed'`.
- **Nobody is numbered for an utterance that has no transcript.** The speaker of an utterance is worked out when its transcript
  is out: an utterance that the recogniser drops (a cough, a scrape) founds no "Speaker n" and moves no speaker's centre.
- **Short pieces.** A piece shorter than 1 s never starts a new speaker; it joins one that sounds alike, or stays `null`.
  Speech over each other is not detected.
- **The voiceprint model and the runtime** (onnxruntime-web 1.31.1, MIT, bundled in `speakers/ort/`) are cached in the
  browser like the speech model (`VoiceprintModel.download()`, `isDownloaded()`, `delete()`), so it all works offline
  after one download. `runtimeUrl` lets the app serve the 15 MB WebAssembly file itself.
- **Requirements.** WebAssembly SIMD and Web Workers (`speakerSupport()`). Two runtime threads on a cross-origin isolated
  page with 8 or more cores, else one. The model needs about 40 MB (the model plus the runtime) and the worker a few
  hundred MB while it runs; it holds at most 10 s of audio in one embedding, longer pieces are averaged from chunks.
- **What it is measured on.** The settings (the `strict` operating point of `voiceprint-cs`) were chosen on synthetic
  meetings of four to ten people made from parliament recordings, not on real meetings; speech from a loudspeaker, online
  meetings and overlapping speech were not measured. Do not promise accuracy to your users from these numbers.
- **Memory is zeroed, then released.** Every array that held audio or a voiceprint is filled with zeros when it has been
  used; terminating the worker (`close()`) frees what the runtime holds that the SDK cannot reach.

## Transcribe audio you already have

```js
const text = await speech.transcribe(samples);   // Float32Array, 16 kHz mono, values in [-1, 1]
```

The same speech gate and filter apply, and the result is `''` when there is no speech. If Silero VAD finds no
speech, the model is not even loaded. Otherwise it is loaded for the call and freed afterwards, unless a `listen()`
session holds it. `transcribe(samples, { signal })` can be aborted, also while the recogniser is still starting.

**You own `samples`.** The SDK copies the array and never modifies it; zero it yourself when you are done. Longer
inputs run through Whisper's 30 s window.

## Configuration

```js
await CzechSpeech.create({ model, config: { minSilenceMs: 800 } });
```

| field | default | meaning |
|---|---|---|
| `threshold` | 0.25 | speech starts at a 32 ms frame with Silero probability ≥ this; also the threshold of the speech gate before Whisper |
| `negThreshold` | 0.10 | speech continues while probability ≥ this |
| `minSilenceMs` | 600 | the utterance ends after this much audio below `negThreshold` |
| `preRollMs` | 300 | audio before the start of speech that is passed to the recogniser |
| `minSpeechMs` | 250 | utterances with less audio ≥ `threshold` are dropped (clicks, coughs) |
| `maxUtteranceMs` | 28 000 | longer speech is cut at the quietest 32 ms frame of the last 2 s and continues as the next utterance |
| `threads` | 0 | recogniser threads; 0 = half the logical cores, 1 to 8 (always 1 without cross-origin isolation) |

These are the Android SDK's defaults and limits. **Why the threshold is 0.25 and not Silero's usual 0.5.** On 2,181
recordings with speech, 0.25 dropped none; 0.5 dropped 5 of the 729 recordings with reverberation and babble noise.
Every synthetic non-speech signal was still rejected at 0.25: silence, white, pink and brown noise, 50 Hz hum,
clicks, bursts and beeps.

## Speed and memory

Whisper's encoder dominates: it processes the whole utterance (in a window trimmed to its length, but never shorter
than 512 frames, about 10 s: the model was not trained on shorter ones) before the first word is decoded. Measured on a laptop with an Intel Core Ultra 7 155H (22 logical cores) and its integrated Intel Arc GPU
(Xe-LPG, shared memory), Chromium 156 headless, for the 6.9 s parliament recording `e-rec` (`test/e2e/bench.mjs`;
one configuration after the other, at most 5 runs each):

| backend | run | total | model load | recognition | encoder | decoder | WebAssembly heap | GPU buffers |
|---|---|---|---|---|---|---|---|---|
| **WebGPU** | first (new instance) | 5.4 s | 2.2 s | 2.8 s | 1.7 s | 18 ms/token | 518 MiB | 833 MiB |
| **WebGPU** | warm (model loaded), 3 runs | 1.33 to 1.38 s | | 1.25 to 1.30 s | 0.71 to 0.80 s | 11.5 to 12.7 ms/token | 518 MiB | 833 MiB |
| **WebGPU** | new instance, shaders compiled before | 3.3 s | 1.5 s | 1.5 s | 0.91 s | 12.5 ms/token | | |
| WebGPU, stock ggml kernels (before the patch) | warm, 3 runs | 5.1 to 5.2 s | | 5.1 to 5.2 s | 4.5 to 4.6 s | 12.2 ms/token | 518 MiB | 684 MiB |
| CPU, multi-threaded, 6 threads | first | 10.6 s | 1.0 s | 9.3 s | 8.9 s | 8.2 ms/token | 1.2 GB | |
| CPU, multi-threaded, 6 threads | warm, 2 runs | 9.6 to 10.7 s | | 9.5 to 10.6 s | 9.1 to 10.2 s | 7.8 ms/token | 1.2 GB | |
| CPU, single-threaded (no COOP/COEP) | first / warm | 40.0 / 37.7 s | 1.4 s | 38.4 / 37.6 s | 37.2 / 36.4 s | 27.3 ms/token | 1.2 GB | |

"First" is what every `listen()` session and `transcribe()` call costs with the default `keepModelLoaded: false`:
the recogniser starts, the model is loaded (onto the GPU) and the first recognition runs; "warm" is with the model
kept loaded. With `listen()` on WebGPU the model is loaded while the person speaks, so the transcript of a 7.6 s
utterance came **1.9 s after the speech ended** (11.7 s on the CPU). **On WebGPU a dictated sentence is
transcribed several times faster than it was spoken on this laptop**; on the CPU it is not. The very first
recognition in a new browser profile takes a few seconds longer while the browser compiles the WGSL shaders;
Chromium caches them.

**A discrete GPU** (the laptop's NVIDIA RTX 4050, which Chromium uses only when started with
`--force_high_performance_gpu`): warm 0.87 to 0.94 s for `e-rec` (encoder 0.50 to 0.58 s, decoder 8.2 ms/token),
first run 5.0 s with the model load. One correctness run and three timed runs; on that GPU the transcript had the
same words as the CPU's but different punctuation (`2012 /sněmovní tisk 91/` instead of `2012, sněmovní tisk 91.`).

**The ggml patch.** With ggml's stock WebGPU kernels, 84 % of the GPU time went into the encoder's matrix
multiplications with the 4-bit weights (`mul_mat_reg_tile_q4_0_f32`, 4.0 s, about 120 GFLOPS, a few per cent of
this GPU's peak): that kernel keeps its accumulators in loop-indexed arrays, which the shader compiler can leave in
scratch memory, and reads shared memory with bank conflicts. `native/patches/ggml-webgpu-q4_0-mat-mat.patch` adds
a Q4_0 kernel with the same arithmetic that keeps its values in registers and reads shared memory as whole vectors:
0.47 s instead of 4.0 s of GPU time. Attention then became the largest cost, and whisper.cpp's matrix-multiplication
attention (0.23 s) beat ggml's flash attention (0.49 s), so the SDK uses it on WebGPU (it costs 150 MiB more GPU
buffers). Remaining GPU time per warm run of `e-rec`: about 0.5 s matrix multiplications, 0.23 s attention, 0.11 s
the decoder's matrix-vector products. The decoder is slower per token on the GPU than on the CPU (12 vs 8 ms):
each token waits for the GPU and reads back the logits. For comparison, native whisper.cpp (AVX2, 6 threads) takes
2.3 s on this laptop.

Memory: with WebGPU the weights and whisper.cpp's buffers live in GPU memory (833 MiB of buffers: 473 MB of weights,
the key/value caches and the compute buffers) and the recogniser's WebAssembly heap is 518 MiB. On the CPU the heap
grows to about 1.2 GB (452 MiB of weights, the rest whisper.cpp's compute buffers, allocated for the full 30 s
window). Both are returned when the recogniser worker is terminated at the end of a session (GPU buffers after
being zeroed, see [Privacy](#privacy)). On an integrated GPU the GPU buffers are system memory too. The capture worker
(speech detection) needs about 64 MB.

**For dictation on WebGPU, consider `keepModelLoaded: true`.** Each new session then skips starting the WebGPU
module, uploading the model to the GPU and building its pipelines: the first transcript of a session costs about as
much as any later one (1.3 s instead of 3.3 to 5.4 s for `e-rec` above). The price is that the model's memory
(about 1.35 GB on WebGPU, 1.2 GB on the CPU) stays allocated until `close()`, and whisper.cpp's GPU buffers are
zeroed only then.

**Accuracy is the same on both backends.** On 20 clean ParCzech4Speech segments (205 s of audio, 4 to 18 s each,
round-robin over the manifest's numbers/legal/random buckets; `test/e2e/accuracy.mjs`) the WebGPU transcripts were
identical to the CPU ones, word for word and character for character, in 20 of 20 segments, with the patched kernel
and without it: word error rate 2.43 % on both (371 reference words). Recognition took 304 s on the CPU (6 threads)
and 58 s on WebGPU (170 s with ggml's stock kernels), model kept loaded. Through `listen()` (the fake microphone,
resampled from 48 kHz) the one recording tested always gave the same words, but its punctuation varied from run
to run on both backends (`2012, sněmovní tisk 91.` or `2012 /sněmovní tisk 91/`): the microphone path does not
give bit-identical samples, and that is a close call for the model. On the RTX 4050, `e-rec` through `transcribe()`
gave the second variant: there the GPU's different order of computation decided it.

Other numbers: the first `fromUrl()` of the 474 MB model from a local server, with SHA-256 verification and storage
in Cache Storage, took 4.3 to 5.3 s (a cached model then loads without a request in about 10 ms); `listen()` had
the microphone and both workers running about 300 ms after the first iteration. Longer utterances take
proportionally longer.

The default CPU thread count is half the logical cores (at most 8); set `config.threads` if you know your users'
hardware better. The WebGPU module uses one CPU thread besides the GPU.

## Why whole utterances and not words as you speak

Whisper is not a streaming model; it transcribes a finished piece of audio. Partial results would mean running the
recogniser again and again on the growing buffer while the person speaks, which would not keep up on the CPU and
would keep the GPU busy the whole time.
So CzechSpeech waits for the end of an utterance (600 ms of silence) and transcribes it once. **Partial results
are a non-goal for 0.2.**

## The hallucination gate

On silence and noise, Whisper models invent text. Without a gate, this model "transcribed" all 66 of 66 test
signals without speech, producing things like "Titulky vytvořil JohnnyX", "," or "... ...". CzechSpeech therefore
applies two checks, in `listen()` and in `transcribe()` alike:

1. **Before Whisper:** the utterance must contain at least one speech segment according to Silero VAD
   (`whisper_vad_segments_from_samples`, threshold 0.25). If it doesn't, Whisper does not run.
2. **After Whisper:** a result with no letter or digit is dropped.
   A model that has learned to answer a segment without speech with an empty transcript can do so: the decoder
   is not forbidden to end the text as its first token (`suppress_blank` is off). An empty result is dropped without
   an event or an error, and is not decoded again.

The integration tests repeat this on recorded silence, pink noise, mains hum, clicks and 10 s of digital zeros.

**Repetition loops:** audio that ends mid-word (`stop()`, the length cap), and now and then an utterance that ends
normally, can make Whisper repeat a phrase until a token limit (see [Decoding limits](#decoding-limits)), so a phrase
of 2 to 12 words repeated three or more times in a row is kept once, and a cut-off copy at the end is dropped.

## Decoding limits

Whisper decodes greedily, one token after the other, until it predicts the end of the text. On some utterances it
falls into a repetition loop instead. whisper.cpp alone lets a loop run to 220 tokens (half its text context), and a
decode that reaches 220 tokens without an end counts as failed: its temperature fallback then decodes the utterance
again at temperatures 0.2, 0.4, ... 1.0, with 5 decoders each, again up to 220 tokens. The meeting app's 3-hour soak
test (WebGPU, Intel Xe-LPG) met this on one 3.9 s sentence of its test loop. Reproduced through the same audio path,
that recognition took 26 s: 220 tokens at 12.9 ms, then two fallback passes of 220 steps at 53 ms each. Its text after
the [repetition guard](#the-hallucination-gate) was the same as after the first pass alone (3.4 s). In other passes
of the loop it took up to 45 s, past the watchdog's limit of 32 s; the watchdog then destroyed the GPU device and the
session went on on the CPU, which is how the soak test lost WebGPU, twice.

So CzechSpeech limits the decoding of each utterance (`src/decoding.js`):

- **At most 25 tokens per second of audio, at least 32** (whisper.cpp's `max_tokens`; its own limit of 220 tokens
  takes over from 8.8 s up). 25 is the Czech model's `tokensPerSecond`; every model of the catalog brings its own rate
  (see [Models and languages](#models-and-languages)), and the figures below are for 25. Exactly: `max_tokens` = max(32, ⌈rate·s⌉) with rate 25 for s seconds of audio (at most 30: the cap
  applies per 30 s window), and whisper.cpp stops after the token with that index, so a decode stopped by the cap has
  `max_tokens` + 1 text tokens. whisper.cpp's own limit stops any decode after 220 text tokens, so a decode stopped
  by a token limit has min(`max_tokens` + 1, 220) text tokens per window: 33 for audio up to 1.28 s, 220 above
  8.76 s. Among 654 decodes of test recordings on WebGPU, the highest rate a correct transcript needed was 8.55
  tokens per second (a ParCzech4Speech segment full of numbers; the median was 5.4 to 5.7); transcripts that doubled
  a phrase reached 11.4. 25 is 2.9 times the highest legitimate rate, and leaves a loop room for the three copies the
  repetition guard needs to remove it, usually (a long phrase in a short utterance may get only two, and then stays
  twice). A decode stopped by either limit is logged with the limit that stopped it, as
  `token limit reached (99 tokens; the cap allows 99)` or `token limit reached (220 tokens; whisper.cpp allows 220)`,
  and counted in `state.cappedRecognitions`. The recordings: `e-rec`, the 20 segments of `test/e2e/accuracy.mjs`
  and the 129 segments of the ParCzech4Speech numbers bucket, and the meeting app's FLEURS recordings (its meeting
  recording, its sample, and the 32 sentences of its soak loop, each cut at 16 frame phases).
- **No temperature fallback**: one greedy decode. On the soak's runaway it only repeated the loop, at four times the
  cost per step. In the test recordings it ran on 4 of the 129 numbers segments (repeated digits look like a loop to
  it) and took 3 to 19 s for each; without it they took 1 to 2 s, with 4 word errors instead of 5.

The decoding of an utterance thus takes at most the encoder pass plus max(32, ⌈25·s⌉) + 1 tokens for s seconds of
audio (220 tokens from 8.8 s up), about 13 ms per token on WebGPU on the laptop of [Speed and memory](#speed-and-memory). The soak's runaway
now takes 1.6 to 1.7 s instead of 23 to 39 s (`test/e2e/runaway.mjs`), on WebGPU throughout.

**The cap does not cut transcripts.** On 208 test recordings on WebGPU (the 142 ParCzech4Speech segments above, among
them `e-rec`, the meeting app's two recordings and its 32 soak sentences at two frame phases) the transcripts with the
cap and without it were the same, word for word, except for two that looped: a soak sentence and a ParCzech4Speech
numbers segment. Those came out with fewer word errors with the cap (1 instead of 8, and 15 instead of 37), because the
repetition guard could remove the shorter loop. Against the decoding before (no cap, temperature fallback), the 206
recordings with a reference (3 537 words) went from 225 to 195 word errors, 6.36 % to 5.51 % (ParCzech4Speech 3.03 % to
2.24 %). 29 of the 30 fewer errors are those two loops; removing the fallback alone went from 225 to 224. The slowest
recognition of a single utterance went from 18.6 s (a 4.4 s ParCzech4Speech segment) to 4.8 s (a 19 s one); the slowest
of all, 5.6 s, is the meeting app's 22.8 s recording, which took as long before. All of these are clean recordings:
noisy speech and real rooms were not measured.

The watchdog stays as a safety net for a GPU far slower than expected (see [Backend](#backend-webgpu-or-cpu)).

## Privacy

What the SDK guarantees:

- **Audio stays in memory, in this page's workers.** During `listen()` the audio goes from the AudioWorklet to the
  capture worker and on to the recogniser worker through message ports; the page's main thread never holds it.
  Audio is never written to any storage (no IndexedDB, Cache Storage, OPFS or files) and never sent anywhere.
- **No network except the model.** The only requests the library makes are for its own files (workers, WebAssembly,
  the VAD model) and the model download from the pinned URL or the URL you give it. The browser tests record every
  request of the page and check this.
- **No telemetry:** no analytics, no crash reporting, no usage pings.
- **Audio is zeroed:** every buffer that held audio is filled with zeros as soon as it is no longer needed: the
  worklet's resampler history and frames, the capture frames, the pre-roll ring and the utterance buffer, the
  utterances queued or discarded on cancel, the copy passed to `transcribe()`, and every copy in the WebAssembly
  heap. Before a worker is terminated, its whole WebAssembly heap is zeroed; that also covers whisper.cpp's internal
  buffers (mel spectrogram, activations, decoded text), which the Android SDK cannot reach.
- **GPU memory is cleared, then released:** on WebGPU, the mel spectrogram, the activations and the encoder and
  decoder state live in GPU buffers. When the model is freed (end of a session or `transcribe()` call, or
  `close()` with `keepModelLoaded`), every buffer of the recogniser's GPU device is overwritten with zeros on the
  GPU, the buffers are destroyed and the device is destroyed, before the worker is terminated. The raw audio never
  goes to the GPU: whisper.cpp computes the mel spectrogram on the CPU, in the WebAssembly heap, and uploads that.
- **No transcript text in logs:** the `log` callback (default `console.debug`/`console.warn`, and `console.info`
  once for the backend fallback) only receives
  timings, counts and states. A transcript's `toString()`, `toJSON()` (so also `JSON.stringify`) and Node's
  `util.inspect` print its length only; read `ev.text` explicitly. whisper.cpp's debug output, which includes
  decoded tokens, is suppressed; only its warnings and errors are passed on.
- **The model is held only while needed:** loaded during `listen()` sessions and `transcribe()` calls, freed
  afterwards, unless you choose `keepModelLoaded`.
- **No emotion or tone analysis.** Speakers are told apart only when the app asks for it ([Speakers](#speakers)), on the device, and an app may compute or keep a voiceprint of a person only with that person's explicit consent (the SDK does not check it);
  the SDK never stores or sends a voiceprint, and a speaker's label, key or number is not in any log or `toString()`.

**Limits.** JavaScript strings cannot be zeroed: the transcript text you receive lives until it is garbage
collected. The browser's own audio pipeline (getUserMedia, the audio thread) is outside the SDK's control. When a
recogniser has to be terminated without the clean-up (cancelling during a long encoder pass, or without
cross-origin isolation), its heap and GPU buffers are released to the browser without being zeroed first. After a
WebGPU failure the failed module's GPU buffers are destroyed without being cleared (the device may be gone); its heap
is zeroed. WebGPU itself guarantees that released GPU memory reaches other pages only zero-initialised, but the
SDK cannot control the driver.

## Browser support

The library needs WebAssembly SIMD, module workers, AudioWorklet, `FileReaderSync` in workers and (for the cache)
Cache Storage; multi-threading also needs cross-origin isolation. The WebGPU backend also needs **WebGPU in dedicated
workers with `shader-f16`** and **WebAssembly JSPI** (JavaScript Promise Integration, `WebAssembly.Suspending`):
whisper.cpp's WebGPU backend waits for the GPU synchronously, and JSPI lets the WebAssembly module suspend there
without blocking the worker. Where either is missing, `'auto'` uses the CPU modules.

| browser | WebGPU backend | CPU backend |
|---|---|---|
| Chrome, Edge 137+ (JSPI on by default since 137) | yes, where the GPU has `shader-f16`; **tested**: Chromium 156 on Windows (D3D12), Intel Arc iGPU (Xe-LPG); briefly NVIDIA RTX 4050 | yes, tested |
| Firefox | not tested: Playwright's Firefox 157 has JSPI and `navigator.gpu` in workers, but headless it blocklists WebGPU (no adapter), so `'auto'` fell back to the CPU | yes: `transcribe()` tested in Firefox 157 (Playwright build, headless, Windows), multi-threaded, 9.2 s for `e-rec`, same transcript |
| Safari | not yet: no JSPI in a stable Safari when this was written (not verified by us) | not tested (Safari 16.4+) |

- **Tested:** Chromium 156 on Windows (headless, with a fake microphone), cross-origin isolated and not; WebGPU
  in full Chromium's new headless mode, CPU in its headless shell (which has `navigator.gpu` but no adapter, so it
  also covers the fallback). The same code paths run in Chrome and Edge.
- **Not tested yet:** WebGPU in Firefox, Safari, any mobile browser, AMD and Apple GPUs, Linux and macOS; the
  RTX 4050 only with one correctness run and three timed runs. A phone with 4 GB of RAM or less is likely
  to fail to allocate the 1.2 GB heap of the CPU backend.
- **Asyncify** (Emscripten's alternative to JSPI, which works in every browser with WebGPU) is not shipped: it
  would be a fourth, larger and slower WebAssembly module.

## Diagnostics

`speech.state` gives `{ listening, modelLoaded, modelLoads, abortedRecognitions, cappedRecognitions, backend, gpuAdapter,
backendFallback, runtime, threads, lastLoadMs, heapBytes, peakHeapBytes, gpuBytes, peakGpuBytes, closed }`. The `log` option receives
lines like `recogniser on WebGPU (intel xe-lpg)` and
`utterance: 7584 ms of audio, 1 speech segment(s), gate 148 ms, recogniser 1713 ms (encoder 1165 ms, decoder 12.2 ms/token) on WebGPU, transcript kept`.
A decode stopped by a token limit logs `utterance: token limit reached (99 tokens; the cap allows 99)` (or
`whisper.cpp allows 220`, see [Decoding limits](#decoding-limits)) at level `info` and counts in
`cappedRecognitions`. The WebGPU watchdog and recovery log warnings like
`utterance: 3904 ms of audio, the recogniser did not finish in 32 s and was stopped (WebGPU stays in use), no transcript`,
`WebGPU failed (WebGPU device lost); recognising on the CPU, WebGPU is tried again in 10 s` and, at level `info`,
`WebGPU started again after a failure`. None of them contains transcript text.

## Building from source

Development, the tests and the tools of this repository need Node.js 22 or later (`devEngines` in `package.json`:
`npm install` refuses an older one). The package itself runs in browsers and declares no Node.js version.

```bash
git clone --recurse-submodules <this repository>
native/build-wasm.sh          # Linux or WSL: emsdk 6.0.13 in ~/emsdk (EMSDK=...), cmake, ninja, python3; -j8 by default
npm install                   # dev tools only: typescript, playwright-core, vite, webpack
npm test                      # unit tests (no model needed)
npm run test:integration      # Node.js, real WebAssembly and model (env below)
npm run test:e2e              # headless Chromium (npx playwright-core install chromium), CPU and WebGPU
npm run test:e2e:threads      # the threaded CPU module of an app built by Vite, in headless Chromium (no model)
npm run test:bundlers         # builds a small app with Vite 8 and webpack 5 (no browser)
node test/e2e/accuracy.mjs    # WebGPU vs CPU on 20 ParCzech segments (word error rates, word differences)
node test/e2e/bench.mjs       # speed and memory of e-rec on WebGPU, CPU multi- and single-threaded
node test/e2e/runaway.mjs     # the soak test's runaway decode through the real audio path, before and after the limits
node scripts/check-tarball.mjs  # npm pack + install into a fresh project + ESM import + TypeScript check
```

- **whisper.cpp:** pinned as the submodule `third_party/whisper.cpp` at commit `482279b`, the same as the Android SDK.
  `native/czechspeech_wasm.c` is the Android JNI glue rewritten as a plain C API.
- **ggml patch (WebGPU module only):** `native/patches/ggml-webgpu-*.patch` are applied by `native/build-wasm.sh` to a
  copy of the submodule (`<build dir>-src/whisper.cpp`) for the `webgpu` variant; the submodule stays unmodified and
  the CPU modules are built from it unchanged. Each patch starts with a header saying what it changes, why, and
  where it comes from. Today there is one: `ggml-webgpu-q4_0-mat-mat.patch`, a faster Q4_0 matrix multiplication
  (see [Speed and memory](#speed-and-memory)).
- **WebAssembly modules:** built by `native/build-wasm.sh` into `src/wasm/` (not committed): `mt` and `st` (CPU) and
  `webgpu` (`native/build-wasm.sh webgpu` alone): whisper.cpp with ggml's WebGPU backend, single-threaded, JSPI. Its
  WebGPU bindings (Dawn's emdawnwebgpu) come from Emscripten's built-in port, which downloads a pinned, hash-checked
  release from GitHub on the first build. The script checks that no build-machine path ends up in the modules, and
  post-processes each module's JS (`scripts/fix-glue.mjs`) so that the threaded module starts its threads from its
  own URL rather than from the file name `czechspeech-mt.js`, which bundlers change.
  `build/wasm-node/` gets Node.js links of the CPU objects for the integration tests (Node.js has no WebGPU).
- **Tests:** integration and browser tests read their inputs from environment variables only and skip, naming the
  variable, when one is missing:
  - `CZECHSPEECH_MODEL`: a CzechSpeech `.bin` (whisper.cpp) model.
  - `CZECHSPEECH_TEST_AUDIO`: a folder of 16 kHz mono WAV files: `a-silence-5s`, `b-pink-noise-2s`, `c-hum-2s` and
    `d-clicks-6s` (synthetic silence, pink noise, mains hum, clicks), `e-rec` (ParCzech4Speech segment
    `test_0000-0000000333`, CC BY 4.0) and `f-rec-plus-silence15s` (`e-rec` followed by 15 s of digital silence).
  - `CZECHSPEECH_TEST_MANIFEST`: the ParCzech4Speech manifest (`.jsonl`) with the reference transcript of `e-rec`.
  - `CZECHSPEECH_TEST_THREADS`: recogniser threads for the tests (default 6).
  - `CZECHSPEECH_TEST_LOOP` (`test/e2e/runaway.mjs` only): a speech recording that Chromium's fake microphone plays in
    a loop, e.g. the meeting app's soak loop of 32 FLEURS sentences. `CZECHSPEECH_E2E_CHANNEL=chrome` runs it in the
    installed Chrome.

  - `CZECHSPEECH_E2E_DETECT_COST=1` (`test/e2e/detect-cost.mjs`, opt-in, not part of `npm run test:e2e`): measures what
    language detection costs against a forced language on the CPU and on WebGPU, and checks that GPU memory does not
    grow over consecutive recognitions. Run it on an otherwise idle machine.

  Never commit models or audio; `.gitignore` blocks them.
- **Example:** `node scripts/serve.mjs` serves the repository with the isolation headers; open
  `http://localhost:8790/examples/basic/`. With `--model <file>` it also serves a local model, used by
  `?model=/models/<file name>` instead of the Hugging Face download; `?backend=cpu|webgpu|auto` chooses the backend,
  and each transcript says where it was recognised.

## What has been tested

- **Unit tests (Node.js):** the endpointer (the Android SDK's 14 cases, ported 1:1), the resampler and framer,
  SHA-256, configuration, events, the model download against a local HTTP server (parts, resume, redirects, hash
  mismatch, abort) and the whole pipeline with a fake engine in real worker threads (event order, microphone closed
  and model freed on break/abort/close, buffers zeroed, one session at a time, the no-SharedArrayBuffer path). The
  backend choice and the WebGPU fallback with a simulated asynchronous GPU engine: every reason not to use WebGPU,
  'webgpu' failing with `unsupported`, a GPU module that does not start, a recognition that fails or hangs in the
  middle (the transcript then comes from the CPU), restarts, one utterance at a time, abort, `keepModelLoaded`. The
  watchdog with fake time (an overrunning recognition is stopped and the GPU kept; one that does not stop is given up
  after the grace) and the way back to WebGPU after a failure (the retry delays, giving up after four failures in a
  row; a module that cannot take over is released); the token cap of every recognition, that it reaches whisper.cpp's
  `max_tokens`, and the log of the limit that stopped a decode.
- **Integration tests (Node.js):** the real WebAssembly and model, with the Android SDK's cases. The transcripts and
  speech timestamps are identical to the Android SDK's JVM tests (same engine, same model, same speech detection).
  Node.js has no WebGPU: there `'auto'` falls back to the CPU (checked) and the GPU case is skipped with a message.
  whisper.cpp stops at the token cap, and the default cap leaves `e-rec` word for word as without it. A released
  threaded CPU module leaves its memory collectable (`test/integration/teardown.test.mjs`, no model needed).
- **Browser tests (headless Chromium):** `listen()` through the real getUserMedia + AudioWorklet path with a fake
  microphone playing a recording, `transcribe()` in the page, the single-threaded fallback on a page without
  isolation, the model served locally and cached in Cache Storage, and an audit of every network request. On
  WebGPU (a real adapter, reported by the test): the non-speech recordings give no transcript and do not load the
  model, `e-rec` gives the CPU's transcript word for word, `listen()` with the fake microphone, the GPU memory
  cleared and released afterwards, a simulated device loss in the middle of a recognition (the transcript then
  comes from the CPU, and the next recognition after the retry delay from WebGPU again), and recognitions over a
  shortened watchdog limit, stopped with the WebGPU module and its device kept; two failure-and-recovery cycles in a
  row leave no pthread worker and no memory of a CPU module behind (WeakRefs and forced garbage collections in the
  recogniser worker). `test/e2e/runaway.mjs` reproduces the
  soak test's runaway decode through the real audio path and checks that it now ends within 5 s on WebGPU, with the
  device kept.
- **Accuracy on the GPU:** `test/e2e/accuracy.mjs`, see [Speed and memory](#speed-and-memory).
- **Bundlers (`npm run test:bundlers`):** a small app that imports the package is built with Vite 8 and webpack 5;
  the worker must be bundled into a chunk of its own, and the worklet, the WebAssembly modules and the VAD model
  emitted unchanged and referenced by the bundle, and every emitted Emscripten module independent of the name it was
  emitted under. `npm run test:e2e:threads` runs the threaded CPU module of an app built by Vite in headless Chromium
  (cross-origin isolated, no model): its pthread workers start from the renamed file. The WebGPU module and `listen()`
  have not been run in a bundled app.

**Not tested yet:** real microphones and real rooms, WebGPU and listen() in Firefox, Safari, mobile browsers, discrete GPUs beyond a
short check, `listen()` and WebGPU in a bundled app, and machines other than the one above.

## Enterprise edition

This SDK and the free Czech model are all you need for Czech. For companies, Moravian AI also offers an enterprise
edition on request:

- **CzechSpeech Enterprise cs+sk+en (beta):** one model for Czech, Slovak and English. Its Czech word error rate is the free model's
  on Czech parliament speech (1.9 %), lower on FLEURS (7.3 % against 7.8 %) and 6.8 % against 6.7 % on noisy speech,
  a difference within the margin of the measurement; Slovak FLEURS 6.6 %, English FLEURS 6.6 %. It is a beta: see
  its model card at [huggingface.co/moravianai/CzechSpeech-Enterprise](https://huggingface.co/moravianai/CzechSpeech-Enterprise)
  for where it falls short. Force the language per call (see [Models and languages](#models-and-languages)).
- **CzechSpeech Voiceprint:** speaker recognition (who said what), fine-tuned on Czech speech. The SDK supports it (see
  [Speakers](#speakers)); the model file is not public.
- **Fine-tuning on your vocabulary and your recordings,** with the accuracy measured on your own data before you
  decide.
- **Help with integration,** e-mail support and model updates.

It uses this same public SDK. An enterprise model is a file that you host yourself or ship in your app, and you pass
it as described in [Your own or a licensed model](#your-own-or-a-licensed-model). There is no licence server and no telemetry: audio and transcripts
stay on the device, as with the free model.

Write to info@moravian.ai.

*Česky:* toto SDK s volným českým modelem stačí na češtinu. Firmám nabízí Moravian AI na vyžádání podnikovou verzi:
- jeden model pro češtinu, slovenštinu a angličtinu, beta (čeština: v parlamentních záznamech stejná chybovost jako volný
  model, na FLEURS nižší, v šumu 6,8 % proti 6,7 %, rozdíl v mezích měření);
- rozpoznávání řečníků doladěné na češtině (SDK ho podporuje, soubor modelu není veřejný);
- doladění na váš slovník a vaše nahrávky s přesností změřenou na vašich datech;
- pomoc s nasazením.

Funguje se stejným veřejným SDK: model hostujete sami a předáte ho s kontrolním součtem SHA-256. Žádný licenční server
ani telemetrie, zvuk zůstává v zařízení. Napište na info@moravian.ai.

## Licence

**Dual licence: free for noncommercial use (PolyForm Noncommercial 1.0.0); commercial licence: info@moravian.ai**

- **This SDK:** © 2026 Jan Rosecký (Moravian AI), under the dual licence above. See [LICENSE](LICENSE).
- **The model weights:** licensed separately (free for non-commercial use under CC BY-NC 4.0; commercial licence:
  info@moravian.ai), on
  [Hugging Face](https://huggingface.co/moravianai/CzechSpeech). The base model, Whisper large-v3-turbo, is © 2022
  OpenAI, MIT licence.
- **Bundled third-party components:** whisper.cpp and ggml (MIT, © The ggml authors; the WebGPU module contains a
  modified ggml WebGPU backend, see `native/patches/`), the Silero VAD v6.2.0 model
  (MIT, © Silero Team), Dawn's emdawnwebgpu WebGPU bindings (BSD 3-Clause, © The Dawn & Tint Authors, and
  Emscripten's licence), ONNX Runtime (onnxruntime-web 1.31.1, MIT, © Microsoft Corporation; the WebAssembly runtime
  of speaker recognition, unchanged, in `src/speakers/ort/` with its own notices) and the Emscripten runtime and system
  libraries in the WebAssembly modules. See [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
