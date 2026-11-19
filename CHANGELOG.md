# Changelog

## 0.3.0 (2026-11-19)

The first version published on npm (0.2.0 was prepared but never published; its changes are below). Speakers. Everything is optional: without `speakers` nothing changes except the new `utteranceId`.

- **Encrypted self-hosted model files (additive).** `CzechSpeechModel.fromUrl(url, { sha256, encryption: { algorithm: 'AES-256-CTR', key, iv } })` (and a `source` of `download()` with a `sha256` of its own) decrypts an AES-256-CTR file while it streams in, resumes with `Range` as before, caches the plaintext and verifies the plaintext `sha256` (required with `encryption`). It protects the file on the server only, not in the browser, and a key and IV must never be used for a second file. New `ModelEncryption` and `PinnedModelSource` types, `ModelSource.encryption`; needs `crypto.subtle` (a secure context), else error code `'unsupported'`. See the README, "Encrypting the hosted file".
- **Consent is the app's duty.** A voiceprint is personal data: use speakers only after everyone whose voice is processed has agreed. The SDK keeps voiceprints on the device only and never stores or sends them; it does not check consent.
- **`listen({ speakers })`** with a `SpeakerSession` tells who spoke, on the device: one `SpeakersEvent` follows each
  transcript with the same `utteranceId` ("Speaker 1", "Speaker 2"...). The numbers belong to the meeting and survive
  `stop()` and the next `listen()` with the same session. `stop()` waits at most 10 s for the last speakers.
- **Speakers are numbered only for utterances that get a transcript**, and `SpeakersEvent.skipped` tells one failed
  utterance (`'error'`, the speaker worker lives on) from a dead worker or a closed session (`'failed'`).
- **`VoiceprintModel`** downloads and caches the voiceprint model and the onnxruntime-web 1.31.1 (MIT) runtime;
  `SpeakerSession`, `SpeakersEvent`, `VOICEPRINT_MODELS`, `DEFAULT_VOICEPRINT_ID` and `speakerSupport()` are exported.
  `MODELS.cs.voiceprint` names the voiceprint model of the Czech model.
- **`TranscriptEvent.utteranceId`** (always set), `CzechSpeechState.speakers`, new error codes `'voiceprint-model'`,
  `'enroll-too-short'`, `'enroll-too-noisy'`.
- The package grows by the bundled onnxruntime runtime (about 3.8 MB gzipped), whether you use speakers or not.

## 0.2.0 (prepared 17 Nov 2026, never published)

Prepared and never published: its changes are part of 0.3.0, the first version on npm. No 0.1.0 was published either; compared with that first local build this release adds the WebGPU backend, `stop()`, the repetition-loop guard and
the new `state` fields (see the README). The sections below are the behaviour changes since the 0.2.0 code was first
assembled; they matter if you tried a pre-release build.

### Model catalog

- **Models are picked by id.** `CzechSpeechModel.download({ id: 'cs' })` (the default, so Czech behaves as before),
  `isDownloaded({ id })` and `delete({ id })`; `MODELS` and `DEFAULT_MODEL_ID` are exported. An unknown id fails at once
  with a `CzechSpeechError` of the new code `'unknown-model'`. `id` and `source` together are a `TypeError`.
- **A model carries `id`, `languages` and `tokensPerSecond`.** `fromUrl()` and `fromBytes()` take `languages` (default
  `['cs']`, non-empty) and `tokensPerSecond` (default 25); such a model has `id` null. A file cached by an earlier
  version (keyed by SHA-256) is found as before.
- **Language per call.** `listen({ language })` and `transcribe(samples, { language })` force one of the model's
  `languages` (default: all of them; for Czech that is `'cs'`, forced as before). Another code is a `RangeError`
  (`listen()` throws it from the first `next()`). `TranscriptEvent` has a new `language` field (also in `toString()` and
  `toJSON()`, which still hide the text). `transcribe()` still returns a string.
- **The token cap uses the model's `tokensPerSecond`** (25 for Czech, so nothing changes there).
- **A model with several languages detects the language within its own list.** One language is forced (Czech: no
  detection pass, as before). Several: a fresh whisper state, a full 30 s encoder pass and one decoder step choose the
  most probable language of the list, which the transcript then uses. The cost is in the README. Cancelling (an abort
  `signal`, leaving the loop, `close()`) during the detection discards the result when the detection returns, and the engine stays
  usable; `stop()` never cancels a recognition, so a detection under way finishes and its transcript is delivered.
- **A language code is two or three lower-case letters** (whisper's codes, such as `'cs'` or `'haw'`), not only ISO 639-1;
  full names such as `'czech'` are refused.
- **`tokensPerSecond` is a positive whole number** (an `Int` on Android); `fromUrl()` and `fromBytes()` refuse 27.5 with a
  `RangeError`, as they already did for a non-positive rate.
- **A WebAssembly module older than the SDK fails at load.** `src/wasm/` is not in git; a module built before the
  language support has no `_cs_whisper_language`, so the engine now refuses to start with a message that names
  `native/build-wasm.sh`, instead of every recognition failing with a generic error. `npm run check:tarball` asserts the
  export in the packed glue.
- **A recognition that succeeds without a language is a failed recognition** (code `'recognition'`), not a transcript
  with a language made up by the worker; the native layer always sets it, so this only guards the two against each other.

### Decoding limits and the WebGPU watchdog

The meeting app's 3-hour soak test lost WebGPU twice, at 33 and at 175 minutes, for the rest of the meeting: one
3.9 s sentence made greedy decoding loop to whisper.cpp's limit of 220 tokens, its temperature fallback decoded it
again and again with 5 decoders (23 to 45 s, reproduced through the same audio path), and past the watchdog's 32 s the
watchdog destroyed the GPU device and `'auto'` stayed on the CPU, which could not keep up.

- **Each recognition decodes at most about 25 tokens per second of audio, with no temperature fallback.** The exact
  cap is max(32, ⌈25·s⌉) + 1 tokens for s seconds of audio, and whisper.cpp's own 220 above that.
  A repetition loop now ends by itself: the soak's runaway takes 1.6 to 1.7 s, on WebGPU throughout. The cap is 2.9
  times the highest rate a correct transcript needed in 654 decodes of the test recordings. On 208 clean test
  recordings the transcripts were the same with and without the cap except for two repetition loops, which came out
  with fewer errors; against the decoding before, the word errors went from 225 to 195 (6.36 % to 5.51 %), 29 of the 30
  from those two loops (removing the fallback alone: 225 to 224), and the slowest recognition of an utterance from
  18.6 s to 4.8 s. Noisy speech and real rooms were not measured. A decode stopped by a token limit (the cap, or
  whisper.cpp's own 220) is logged with the limit that stopped it and counted in the new `state.cappedRecognitions`.
  See the README, "Decoding limits".
- **A recognition over the watchdog's limit is stopped, and WebGPU kept.** whisper.cpp stops at its next abort check;
  the utterance gives no transcript (`transcribe()` resolves `''`) and counts in `abortedRecognitions`, and the next
  one runs on the same WebGPU module. Only a recognition that does not stop within 5 s more is a WebGPU failure. Before,
  every overrun destroyed the device.
- **`'auto'` goes back to WebGPU after a failure.** The failed utterance is still redone on the CPU. WebGPU is tried
  again at least 10 s later, with no timer. A recogniser that is still running starts the WebGPU module with its next
  utterance or call and releases the CPU module at once: its pthread workers are terminated and its memory is no
  longer reachable, so the garbage collector frees it (checked in the browser tests). It loads the model with the
  first utterance that has speech (`e-rec` took 4.9 and 7.1 s in the browser tests, 3.1 and 3.7 s of it the model
  load). Every fallback to the CPU starts a fresh CPU module, so a later one is as slow as the first. A recogniser
  started later starts on WebGPU as the first one did (`listen()` loads the model when the session starts).
  `state.backend` is `'webgpu'` again, `backendFallback` `null`. Further failures in a row wait 60 s and 300 s; after
  the fourth the instance stays on the CPU. The instance keeps this state across its recognisers, so with
  `keepModelLoaded: false` a recogniser started before the retry time starts on the CPU at once. Before, any WebGPU
  failure kept the instance on the CPU for good.
- **A module that cannot take over is never left in use.** With `'auto'`, when the CPU module does not start after a
  WebGPU failure, a `transcribe()` call rejects with code `runtime` and a `listen()` utterance is an `error` event,
  both naming the two failures, and the next one starts the CPU module again (or WebGPU once its retry time has come). With `'webgpu'`, a WebGPU module restarted after a
  failure whose speech gate does not load is released and the call fails with code `unsupported`; before, it stayed
  in use without a speech gate.

### Encoder window

- **The encoder window is never shorter than 512 frames (about 10 s).** It was trimmed to the utterance, and the
  CzechSpeech model was never trained on a window that short: under 4 s of speech it invented or repeated sentences
  in 15 to 46 % of the utterances. Now the window is the utterance's length rounded up as before, but at least 512
  frames; from 1500 frames (30 s) up it is still the full window. Measured on the Android SDK's native code (the same
  glue and whisper.cpp) on short utterances: word error rate 6.28 to 3.90 % on clean audio and 10.82 to 9.89 % in
  noise, the invented-word bursts 8 to 0 and 2 to 0, total processing time 0.97 times the old one (the temperature
  fallback and its loops no longer happen), latency hardly changed. The WebAssembly modules were rebuilt for this;
  `test/unit/audio-ctx.test.mjs` checks the window in the real WebAssembly. The decoding limits above are unchanged.

### suppress_blank

- **`suppress_blank` is off.** whisper.cpp forbids the end of the text as the first token, so a model that learned to
  answer a segment without speech with an empty transcript had to write something into silence:
  100 % of 360 segments without speech with it on, 0 % with it off (past the speech gate 12.2 % against 0 %).
  Speech transcripts do not change (parliament, FLEURS cs and short utterances the same; 0 of 1,952 utterances
  dropped), and the CzechSpeech model is not affected. An empty result is dropped without a `Transcript` event, an
  error or a retry: an empty decode ends at its first token with a log-probability near 0, which is not a failure for
  whisper.cpp's temperature fallback. Covered by `test/unit/speech.test.mjs` and `test/unit/native-params.test.mjs`. The WebAssembly modules were rebuilt.

### Model download

- **A mirror keeps the pinned size and SHA-256.** `CzechSpeechModel.download({ source })` with a URL or `{ url }`
  now verifies the pinned size and SHA-256 (before, a mirror source dropped both and any bytes were accepted). A source
  with a `sha256` of its own is another file, verified against that hash as `fromUrl()` does. A source that gives a
  `size` other than the pinned one and no `sha256` of its own is a `TypeError`. The download is stored under the
  pinned model's key, whichever mirror it came from.
- **`isDownloaded(source)` and `delete(source)` read a source as `download()` does.** A URL alone means the pinned
  model (`delete(url)` deletes the pinned model); a model of `fromUrl()` is found by its `{ url, sha256 }`. They
  reject with a `TypeError` for a source `download()` would refuse.
- **`fromUrl(url)` without `sha256` is no longer cached.** It is downloaded into memory every time, as nothing could
  find a copy cached under its URL again; its SHA-256 is still computed. With `sha256` nothing changes: verified,
  cached, resumable.
- **Cache Storage failures are a `CzechSpeechError` with code `storage`** (the browser's error as `cause`): opening
  the cache, reading from it and listing it, and reading `caches` itself, in `download()`, `fromUrl()`, `isDownloaded()`
  and `delete()`. Before, only failed writes were wrapped.

### Lifecycle

- **`close()` resolves once every worker is terminated, and every call of it resolves together.** A second
  `close()` used to resolve at once, with the workers still running.
- **`close()` cancels a recogniser that is still starting** instead of waiting for it; a `transcribe()` waiting for
  that start-up rejects with code `closed` and never sends its audio, also when the start-up finishes just as
  `close()` is called, or `close()` comes between the call's lease and its message to the recogniser.
- **Cancelling does not wait for a worker's start-up.** Leaving the loop, aborting `listen()`'s signal, `close()`, and
  aborting a `transcribe()` take effect at once (about 2 ms instead of the length of the start-up, which had no
  limit); a worker that is still starting and that nobody waits for any more is terminated.
- **A worker that does not start within 60 s** (for example a threaded module whose threads cannot be loaded, which
  waits for them for ever) is terminated, and the `transcribe()` or `listen()` that waited for it fails with code
  `runtime` ("The recogniser did not start within 60 s", the cause a `TimeoutError`); the next call starts a new
  worker. Before, the call waited without end. An abort or `close()` that comes first still wins.
- **`close()` or an abort in the same tick as `listen()`'s first `next()`** terminates the capture worker too (it
  stayed alive).
- **An aborted `transcribe()` leaves nothing behind** when the recogniser is terminated instead of answering.
- **A worker that cannot be started** (for example a cross-origin `assetBaseUrl`, which browsers refuse for module
  workers) fails the call with a `CzechSpeechError` with code `runtime` and the browser's error as `cause`, and no
  longer keeps the model loaded for the rest of the instance's life.
- **`create()` fails with code `unsupported` where Web Workers are not available** (server-side rendering, Node.js),
  before it fetches anything. It failed with `runtime` before.

### Errors

- **`CzechSpeechError.code` is always one of the 11 documented codes.** A code of the error behind it (a
  DOMException's number, Node's `'ENOENT'`) is replaced by the code that fits the place, and the constructor turns any
  other code into `runtime`.
- **No raw errors** from `create()` (the VAD asset fetch), `transcribe()` (a message the browser refuses to send) or a
  `listen()` start-up: they become a `CzechSpeechError` with code `runtime` and the original as `cause`. Invalid
  arguments are still `TypeError` and `RangeError`.

### Bundlers

- **Workers and assets are constructed in the form bundlers analyse**: `new Worker(new URL('./worker.js',
  import.meta.url), { type: 'module', name: 'czechspeech-capture' })` (and `'czechspeech-recognizer'`) with literal
  options, which Vite 8 requires; the import of the WebAssembly module carries `webpackIgnore`.
- **The multi-threaded module starts its threads under any file name.** Emscripten's glue started its pthread workers
  from the file name `czechspeech-mt.js`, which bundlers change (Vite always, webpack by default); the glue is now
  post-processed (`scripts/fix-glue.mjs`, run by `native/build-wasm.sh`) to start them from the module's own URL. For
  modules built earlier: `node scripts/fix-glue.mjs src/wasm/czechspeech-*.js`. The package check refuses a glue that
  still has the fixed name.
- Tested: builds with Vite 8 and webpack 5 (worker, worklet, WebAssembly and VAD emitted and referenced, the glue
  independent of its name) and one browser run of the threaded CPU module in an app built by Vite.

### Package

- No `engines` field: the package runs in browsers, and `engines` would make engine-strict npm installs and Yarn
  refuse it on an older Node.js. Development, tests and tools of the repository need Node.js 22 or later
  (`devEngines`, enforced by `npm install` in the repository only).
- The package check also refuses a WebGPU module without the patched Q4_0 kernel.
- `package.json` has `repository`, `bugs` and `publishConfig.access: "public"` (a scoped package is otherwise
  published as restricted).

### Tests

- The recordings the integration and browser tests read from `CZECHSPEECH_TEST_AUDIO` have English names:
  `a-silence-5s`, `b-pink-noise-2s`, `c-hum-2s`, `d-clicks-6s`, `e-rec` and `f-rec-plus-silence15s` (before:
  `a-ticho-5s`, `b-ruzovy-sum-2s`, `c-brum-2s`, `d-cvaky-6s`, `f-rec-plus-ticho15s`). The audio is the same; rename
  the files.

### Known limits

- When Cache Storage exists but fails (some private windows), `download()` fails with code `storage`; there is no
  fallback to a download into memory.
- The Emscripten glue still names its own `.wasm` as a fallback for when no `locateFile` is given; the SDK always
  gives one, so the bundler's name is used.
- In a bundled app only the threaded CPU module has been run in a browser; the WebGPU module and `listen()` have not.
- A threaded module that cannot start is not replaced by the single-threaded one: the call fails after the start-up
  limit (a fallback from `mt` to `st` is a possible follow-up). The failure has code `runtime`, not a code of its own.
