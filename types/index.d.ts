// Type declarations of @moravianai/czechspeech 0.3.0.

/**
 * Where the recogniser runs. 'auto': on the GPU through WebGPU when the browser offers it in a worker (JSPI, an adapter
 * with shader-f16) and the WebGPU module starts, else on the CPU (WebAssembly), with one console.info. 'webgpu': the GPU
 * or a CzechSpeechError with code 'unsupported'. 'cpu': never the GPU.
 */
export type CzechSpeechBackend = 'auto' | 'webgpu' | 'cpu';

/** Speech detection settings; the defaults are the validated ones (same as the Android SDK), plus the backend. */
export interface CzechSpeechConfig {
  /** Speech starts at a 32 ms frame with Silero probability >= this; also the threshold of the speech gate. Default 0.25. */
  readonly threshold: number;
  /** Speech continues while the probability is >= this. Default 0.10. */
  readonly negThreshold: number;
  /** The utterance ends after this much audio below negThreshold. Default 600, range 32..10000. */
  readonly minSilenceMs: number;
  /** Audio before the start of speech passed to the recogniser. Default 300, range 0..2000. */
  readonly preRollMs: number;
  /** Utterances with less speech than this are dropped (clicks, coughs). Default 250, range 0..10000. */
  readonly minSpeechMs: number;
  /** Longer speech is cut at the quietest frame of the last 2 s and continues as the next utterance. Default 28000, range 4000..29000. */
  readonly maxUtteranceMs: number;
  /** Recogniser threads; 0 = half the logical cores (1 to 8), always 1 without cross-origin isolation. Default 0. */
  readonly threads: number;
  /** Recogniser backend. Default 'auto'. */
  readonly backend: CzechSpeechBackend;
}

export declare const DEFAULT_CONFIG: Readonly<CzechSpeechConfig>;

/** getUserMedia constraints used by listen() unless overridden (see the README for the reasons). */
export declare const DEFAULT_AUDIO_CONSTRAINTS: Readonly<MediaTrackConstraints>;

export declare const VERSION: string;

export type CzechSpeechErrorCode =
  | 'microphone-denied'
  | 'microphone-unavailable'
  | 'microphone-ended'
  | 'model'
  | 'download'
  | 'storage'
  | 'runtime'
  | 'recognition'
  | 'busy'
  | 'closed'
  | 'unsupported'
  /** The model id is not in the catalog. */
  | 'unknown-model'
  /** The voiceprint model could not be used. */
  | 'voiceprint-model'
  /** Reserved, not produced in 0.3.0: a recording is too short to take a voiceprint from. */
  | 'enroll-too-short'
  /** Reserved, not produced in 0.3.0: a recording has more noise than speech. */
  | 'enroll-too-noisy';

/** Thrown when the model, the microphone or the WebAssembly runtime cannot be used. */
export declare class CzechSpeechError extends Error {
  constructor(message: string, options?: { code?: CzechSpeechErrorCode; cause?: unknown });
  readonly name: 'CzechSpeechError';
  readonly code: CzechSpeechErrorCode;
}

/** Speech started at `atMs` milliseconds of audio since the session started. */
export declare class SpeechStartedEvent {
  readonly type: 'speechStarted';
  readonly atMs: number;
}

/** Speech ended at `atMs`; emitted after minSilenceMs of silence. */
export declare class SpeechEndedEvent {
  readonly type: 'speechEnded';
  readonly atMs: number;
}

/**
 * One recognised utterance. `text` is never blank. toString(), toJSON() and Node's inspection print the length of
 * the text only, so transcripts do not end up in logs by accident.
 */
export declare class TranscriptEvent {
  readonly type: 'transcript';
  readonly text: string;
  /** Start of speech, ms of audio since the session started. */
  readonly startMs: number;
  /** End of speech, on the same clock. */
  readonly endMs: number;
  /** Time spent on this utterance by the speech gate and the recogniser (wall clock). */
  readonly processingMs: number;
  /** The language the text was recognised in (a whisper language code, one of the model's `languages`), for example 'cs'. */
  readonly language: string;
  /** Names the utterance for the SpeakersEvent that follows it: '<listen() call>.<n>', unique within one CzechSpeech. */
  readonly utteranceId: string;
  toString(): string;
  toJSON(): { type: 'transcript'; text: string; language: string; startMs: number; endMs: number; processingMs: number; utteranceId: string };
}

/** Who a stretch of speech was: a speaker of this meeting ("Speaker n"), or an app-defined key (voices the app knows; reserved, not produced in 0.3.0). */
export type SpeakerLabel = { readonly kind: 'session'; readonly n: number } | { readonly kind: 'known'; readonly key: string };

/** A stretch of one utterance and the speaker found for it (null: not sure). Times: ms of audio since the session started. */
export interface SpeakerPart {
  readonly startMs: number;
  readonly endMs: number;
  readonly speaker: SpeakerLabel | null;
}

/**
 * Who spoke in one utterance. Comes after the TranscriptEvent with the same `utteranceId`, exactly once per transcript
 * (when listen() got `speakers`), also when it has nothing to say (`skipped`). An utterance without a transcript has
 * none. toString(), toJSON() and Node's inspection show counts only, never a number or key of a speaker.
 */
export declare class SpeakersEvent {
  readonly type: 'speakers';
  readonly utteranceId: string;
  /** The stretches in order; empty when `skipped`. */
  readonly parts: readonly SpeakerPart[];
  /** The distinct speakers in the order they spoke; empty: nobody could be told. */
  readonly speakers: readonly SpeakerLabel[];
  /**
   * Why there are no parts: 'cpu' (the recogniser runs on the CPU and the SpeakerSession was made with onCpu 'skip'),
   * 'behind' (the speaker worker had too much waiting), 'error' (this utterance failed; the speaker worker lives on and
   * the next one is handled) or 'failed' (the speaker worker is dead, or the session was closed: nothing more comes
   * from it). Else null.
   */
  readonly skipped: 'cpu' | 'behind' | 'error' | 'failed' | null;
  toString(): string;
  toJSON(): { type: 'speakers'; utteranceId: string; parts: number; speakers: number; skipped: 'cpu' | 'behind' | 'error' | 'failed' | null };
}

/** Recognition of one utterance failed; the session goes on. */
export declare class SpeechErrorEvent {
  readonly type: 'error';
  readonly error: CzechSpeechError;
}

export type SpeechEvent = SpeechStartedEvent | SpeechEndedEvent | TranscriptEvent | SpeechErrorEvent | SpeakersEvent;

/**
 * How a hosted model file is encrypted: AES-256-CTR over the whole file (the ciphertext is as long as the plaintext);
 * `iv` is the initial 16-byte counter block, incremented as one 128-bit big-endian integer per 16-byte block, as
 * `openssl enc -aes-256-ctr -K <key> -iv <iv>` does. The download is decrypted while it streams in. This keeps a
 * found URL useless; it does not protect the model in the browser, where the key is part of your page. The key never
 * appears in an error message. Never encrypt two different files with the same key and IV (a new version of a model
 * gets a new key, or at least a new IV): the two ciphertexts would reveal the XOR of the two plaintexts.
 */
export interface ModelEncryption {
  readonly algorithm: 'AES-256-CTR';
  /** 64 hex digits (32 bytes). */
  readonly key: string;
  /** 32 hex digits (16 bytes). */
  readonly iv: string;
}

/** Where a model is downloaded from. */
export interface ModelSource {
  readonly url: string;
  /** Size in bytes; checked when given. */
  readonly size?: number;
  /** Lower-case hex SHA-256; verified when given (and makes the download resumable). */
  readonly sha256?: string;
  /** The hosted file is encrypted; `sha256` (required with it) and `size` are those of the plaintext. */
  readonly encryption?: ModelEncryption;
}

/** The pinned model of the catalog: never encrypted. */
export interface PinnedModelSource {
  readonly url: string;
  readonly size: number;
  readonly sha256: string;
}

export declare const DEFAULT_MODEL: Readonly<PinnedModelSource>;

/** A model of the catalog. */
export interface CatalogEntry {
  readonly id: string;
  /** Where the file is, with its pinned size and SHA-256. */
  readonly source: Readonly<PinnedModelSource>;
  /** The languages the model speaks (whisper language codes); only these can be forced with `language`. */
  readonly languages: readonly string[];
  /** At most how many tokens per second of audio a correct transcript needs (a positive whole number). */
  readonly tokensPerSecond: number;
  /** The id (in VOICEPRINT_MODELS) of the voiceprint model that belongs to this one, or null. */
  readonly voiceprint: string | null;
}

/** The model catalog by id. Today: 'cs', the pinned Czech model. */
export declare const MODELS: Readonly<Record<string, CatalogEntry>>;
/** The id used when none is given: 'cs'. */
export declare const DEFAULT_MODEL_ID: 'cs';

/** What a model of your own speaks; the defaults are Czech and 25. */
export interface ModelTraits {
  /** Whisper language codes (two or three lower-case letters, such as 'cs' or 'haw'), non-empty (RangeError otherwise). Default ['cs']. */
  languages?: readonly string[];
  /** Positive whole number (RangeError otherwise, as an Int on Android). Default 25. */
  tokensPerSecond?: number;
}

export interface DownloadOptions {
  /** Bytes done and total, about every MiB. */
  onProgress?: (done: number, total: number) => void;
  signal?: AbortSignal;
}

/** The CzechSpeech model (whisper.cpp q4_0, 474 MB), ready for CzechSpeech.create(). */
export declare class CzechSpeechModel {
  private constructor();
  /** The pinned model on Hugging Face: url, size 473992235, sha256 04ead666…30a3. */
  static readonly DEFAULT: Readonly<PinnedModelSource>;
  /**
   * Downloads a model of the catalog (default `id: 'cs'`, the pinned Czech model) unless it is already in the
   * browser's Cache Storage. Resumable, SHA-256-verified, idempotent. An unknown `id` rejects with a
   * CzechSpeechError of code 'unknown-model'; `id` together with `source` rejects with a TypeError. `source` (a URL or { url }) serves the same file from a mirror: the pinned size and SHA-256 are still
   * enforced. A source with a `sha256` of its own is another file, verified against that hash (as with fromUrl()).
   * Rejects with a TypeError when the source gives a `size` other than the pinned one and no `sha256` of its own.
   */
  static download(options?: DownloadOptions & ({ id?: string; source?: undefined } | { id?: undefined; source?: ModelSource | string })): Promise<CzechSpeechModel>;
  /**
   * Downloads a model from any URL. With `sha256` it is verified, cached and resumable, and isDownloaded() and delete()
   * find it by { url, sha256 }. Without `sha256` it is downloaded into memory every time and not cached: a URL alone
   * means the pinned model to isDownloaded() and delete(). `languages` and `tokensPerSecond` say what the model
   * speaks (default Czech, 25); it has no `id`.
   */
  static fromUrl(url: string, options?: DownloadOptions & ModelTraits & { sha256?: string; size?: number; encryption?: ModelEncryption }): Promise<CzechSpeechModel>;
  /** A model from bytes the app already has. Verified when `sha256` is given. Czech by default, like fromUrl(). */
  static fromBytes(bytes: ArrayBuffer | ArrayBufferView | Blob, options?: ModelTraits & { sha256?: string }): Promise<CzechSpeechModel>;
  /**
   * True when the model (default: the pinned one) is completely cached. No network, no hashing. `source` as in
   * download(): a mirror URL alone stands for the pinned model; a model of fromUrl() is found by its { url, sha256 }
   * (one without `sha256` is never cached). Rejects with a TypeError for a source download() would refuse (a `size`
   * other than the pinned one without a `sha256` of its own).
   */
  static isDownloaded(source?: ModelSource | string | { id: string }): Promise<boolean>;
  /**
   * Deletes the cached model (default: the pinned one) and any partial download. True when one was stored. `source`
   * as in download(): delete(url) deletes the pinned model (when it came from that mirror or any other); a model of
   * fromUrl() is deleted by its { url, sha256 } (one without `sha256` is never cached). Rejects with a TypeError for a
   * source download() would refuse.
   */
  static delete(source?: ModelSource | string | { id: string }): Promise<boolean>;
  /** Size in bytes. */
  readonly size: number;
  /** SHA-256 (lower-case hex) when verified or computed during the download, else null. */
  readonly sha256: string | null;
  /** True when the model is stored in Cache Storage. */
  readonly cached: boolean;
  /** The catalog id ('cs'), or null for a model of fromUrl(), fromBytes() or a source of its own. */
  readonly id: string | null;
  /** The languages the model speaks (frozen). */
  readonly languages: readonly string[];
  /** At most how many tokens per second of audio a correct transcript needs. */
  readonly tokensPerSecond: number;
}

export type CzechSpeechLogger = (level: 'info' | 'warn', message: string) => void;

export interface CzechSpeechOptions {
  model: CzechSpeechModel;
  config?: Partial<CzechSpeechConfig>;
  /**
   * Keep the model loaded between listen() sessions and transcribe() calls until close(). Default false: the model
   * is loaded when a session or call starts and freed (worker terminated, memory returned) when it ends.
   */
  keepModelLoaded?: boolean;
  /** Receives timings, counts and states only, never transcript text. false: no logging. Default: console.debug/warn. */
  log?: CzechSpeechLogger | false;
  /** Where worker.js, worklet.js, wasm/ and assets/ of this package are served, if a bundler does not resolve them. */
  assetBaseUrl?: string | URL;
}

export interface ListenOptions {
  /** Aborting it closes the microphone, frees the model and makes the iterator throw `signal.reason`. */
  signal?: AbortSignal;
  /** getUserMedia audio constraints, merged over DEFAULT_AUDIO_CONSTRAINTS. */
  audio?: MediaTrackConstraints;
  /** Use this stream instead of opening the microphone. It is not stopped at the end of the session. */
  stream?: MediaStream;
  /**
   * Force one of the model's `languages`, for example 'en' for a Polish and English model. Default: the recogniser
   * picks among all of them for each utterance (a Czech-only model is always forced to Czech). With several languages
   * and no `language`, every utterance pays one more encoder pass for the detection (see the README). Any other code
   * is a RangeError, thrown by the first next() of the iterator.
   */
  language?: string;
  /**
   * Tell the speakers apart. The same SpeakerSession for every listen() of one meeting keeps "Speaker 1, 2..." across
   * pauses. Each transcript is then followed by one SpeakersEvent. Without it nothing is computed and no event comes.
   * Consent: use it only after everyone whose voice will be processed has explicitly agreed (show a consent screen); a
   * voiceprint is personal data. The SDK keeps voiceprints on the device (worker memory), never stores or sends them and
   * does not check consent: that is the app's duty.
   */
  speakers?: SpeakerSession;
}

/** The WebGPU adapter as the browser describes it (GPUAdapterInfo; browsers may leave fields empty). */
export interface GpuAdapterInfo {
  readonly vendor: string;
  readonly architecture: string;
  readonly device: string;
  readonly description: string;
}

export interface CzechSpeechState {
  readonly listening: boolean;
  readonly modelLoaded: boolean;
  readonly modelLoads: number;
  /**
   * Recognitions whisper.cpp stopped before their end: cancelled (an abort, leaving the loop, close()), or stopped by
   * the WebGPU watchdog because they took longer than their time limit (no transcript for those, the GPU stays in use).
   */
  readonly abortedRecognitions: number;
  /**
   * Recognitions whose decoding was stopped by a token limit: the cap (max(32, ⌈25·s⌉) + 1 tokens for s seconds), or
   * whisper.cpp's own limit of 220 tokens (README, "Decoding limits"). Repetition loops, as a rule. Their transcript is
   * what was decoded up to the limit, after the repetition guard: a phrase of 2 to 12 words repeated three or more
   * times in a row is kept once, so a loop that got only two copies before the limit stays twice.
   */
  readonly cappedRecognitions: number;
  /**
   * The recogniser's backend in use; null until the recogniser has started once (first listen() or transcribe()). With
   * config.backend 'auto' it is 'cpu' for a while after a WebGPU failure, then 'webgpu' again.
   */
  readonly backend: 'webgpu' | 'cpu' | null;
  /** The adapter when backend is 'webgpu', else null. */
  readonly gpuAdapter: GpuAdapterInfo | null;
  /**
   * Why WebGPU is not used although config.backend is 'auto' (no adapter, no shader-f16, a failure...), else null; null
   * again once WebGPU is back after a failure.
   */
  readonly backendFallback: string | null;
  /** The CPU WebAssembly module of this page (the recogniser's when backend is 'cpu', always the capture worker's). */
  readonly runtime: 'wasm-threads' | 'wasm-single';
  /** CPU recogniser threads (the WebGPU module uses one CPU thread besides the GPU). */
  readonly threads: number;
  readonly lastLoadMs: number | null;
  /** WebAssembly heap of the recogniser, bytes (last report) and the peak seen. */
  readonly heapBytes: number;
  readonly peakHeapBytes: number;
  /** GPU buffers of the recogniser (WebGPU backend; weights, caches, compute buffers), bytes after the last model load, and the peak. */
  readonly gpuBytes: number;
  readonly peakGpuBytes: number;
  /**
   * The speakers of the last listen() that had `speakers`, else null: `active` (being computed now), `threads` of the
   * runtime, `pending` (transcripts still waiting for their speakers), `unsplit` (utterances not looked at for speaker
   * changes because the speaker worker was behind), `skipped` (events with a `skipped` reason), `lost` (speakers
   * still missing when stop() ended the wait). Counts only.
   */
  readonly speakers: { readonly active: boolean; readonly threads: number; readonly pending: number; readonly unsplit: number; readonly skipped: number; readonly lost: number } | null;
  readonly closed: boolean;
}

/** Offline Czech speech-to-text in the browser. Audio never leaves the page. */
export declare class CzechSpeech {
  private constructor();
  /** Cheap: checks the configuration and fetches the bundled Silero VAD (885 kB); the model is loaded later. */
  static create(options: CzechSpeechOptions): Promise<CzechSpeech>;
  readonly config: Readonly<CzechSpeechConfig>;
  /** Diagnostics; no content. */
  readonly state: CzechSpeechState;
  /**
   * Listens to the microphone. Iterating opens the microphone (getUserMedia) and loads the model in parallel. To
   * finish, call stop(): the last utterance is still transcribed and the loop ends. Leaving the loop, aborting the
   * signal or close() instead closes the microphone at once, discards speech not yet recognised and frees the
   * model. One session at a time per instance. Fatal errors (microphone denied, model or WebAssembly failure) are
   * thrown; an 'error' event is one failed utterance and the session goes on.
   */
  listen(options?: ListenOptions): AsyncGenerator<SpeechEvent, void, undefined>;
  /**
   * Finishes the active listen() session. It finishes the word in progress: with speech in progress the microphone
   * stays on until a short pause (256 ms), at most 1 s; without speech it stops delivering audio at once. The
   * utterance in progress then ends as if silence followed, everything still waiting is recognised and yielded, the
   * loop ends normally and the model is freed. Returns immediately. Does nothing without an active session (it
   * never stops a later one). To discard instead, leave the loop, abort the signal or call close().
   */
  stop(): void;
  /** Only the transcript events of listen(). */
  transcripts(options?: ListenOptions): AsyncGenerator<TranscriptEvent, void, undefined>;
  /**
   * Transcribes 16 kHz mono samples in [-1, 1]. '' when there is no speech (the model is not even loaded then).
   * The array is copied and never modified. `language` forces one of the model's `languages` (default: all of them;
   * any other code is a RangeError); the language recognised is on the events of listen().
   */
  transcribe(samples: Float32Array, options?: { signal?: AbortSignal; language?: string }): Promise<string>;
  /**
   * Cancels a session and pending calls (they reject with code 'closed'), frees everything, rejects further calls.
   * Resolves once every worker is terminated; every call of close() resolves then.
   */
  close(): Promise<void>;
}

// ---- Speakers: who said what, on the device. Nothing is stored or sent; the app decides about names. ----

/** One setting of the speaker recognition, measured with one voiceprint model (the operating point of the model). */
export interface VoiceprintOperatingPoint {
  readonly id: string;
  readonly named: number;
  readonly lengthSlope: number;
  readonly margin: number;
  readonly session: number;
  readonly minMs: number;
  readonly drop: number;
  readonly splitWindowMs: number;
  readonly splitHopMs: number;
  readonly enrollWindowMs: number;
  readonly enrollHopMs: number;
}

/** A voiceprint model of the catalog. */
export interface VoiceprintModelEntry {
  readonly id: string;
  /** Size and SHA-256 of the file. */
  readonly size: number;
  readonly sha256: string;
  /** null: the model is not public and the app says where it serves it from (VoiceprintModel.download({ url })). */
  readonly url: string | null;
  readonly dim: number;
  readonly operatingPoints: Readonly<Record<'strict', VoiceprintOperatingPoint>>;
}

export declare const VOICEPRINT_MODELS: Readonly<Record<string, VoiceprintModelEntry>>;
/** 'voiceprint-cs'. */
export declare const DEFAULT_VOICEPRINT_ID: 'voiceprint-cs';

/**
 * Whether this browser can tell speakers apart (WebAssembly SIMD and workers), and how many threads the runtime should
 * get: 2 on a cross-origin isolated page with 8 or more logical cores, else 1.
 */
export declare function speakerSupport(): { supported: boolean; threads: number; reason: 'no-webassembly-simd' | 'no-workers' | null };

/** The voiceprint model and the onnxruntime runtime next to it, in the browser's model cache (offline after one download). */
export declare class VoiceprintModel {
  private constructor();
  /**
   * Downloads the model from `url` (size and SHA-256 pinned by the catalog) and the onnxruntime WebAssembly runtime (pinned
   * too) into Cache Storage; resumable. The runtime comes from `runtimeUrl` if the app hosts it, else from
   * `speakers/ort/` under `assetBaseUrl`, else from this package. `onProgress` counts the bytes of both files.
   */
  static download(options: { id?: string; url: string; assetBaseUrl?: string | URL; runtimeUrl?: string | URL; onProgress?: (done: number, total: number) => void; signal?: AbortSignal }): Promise<VoiceprintModel>;
  static isDownloaded(options?: { id?: string }): Promise<boolean>;
  static delete(options?: { id?: string }): Promise<boolean>;
  readonly id: string;
  readonly sha256: string;
  readonly size: number;
  readonly dim: number;
}

/**
 * The speakers of one meeting: who is "Speaker 1", "Speaker 2"... The state belongs to the meeting, not to one
 * listen(): pass the same session to every listen() of the meeting. Everything is in memory (a worker with the model and
 * the centres of the speakers); close() zeroes it and ends the worker. Nothing is stored or sent; toString(), toJSON()
 * and Node's inspection show nothing of the speakers. Consent: an app may compute or store a voiceprint of a person only
 * with that person's explicit consent (and should show a consent screen); the SDK keeps voiceprints on the device, never
 * sends them, and does not check consent itself.
 */
export declare class SpeakerSession {
  private constructor();
  static create(options: {
    model: VoiceprintModel;
    /** The number of the first new speaker; after a recovered meeting, the next free number. Default 1. */
    firstLabel?: number;
    /** 1 or 2. Default: speakerSupport().threads. */
    threads?: number;
    /** 'skip' (default): while the recogniser runs on the CPU the speakers are not computed (SpeakersEvent.skipped 'cpu'). */
    onCpu?: 'skip' | 'run';
    assetBaseUrl?: string | URL;
  }): Promise<SpeakerSession>;
  /** The speakers found so far and how many ms each has spoken. */
  readonly speakers: ReadonlyArray<{ readonly n: number; readonly speechMs: number }>;
  readonly onCpu: 'skip' | 'run';
  readonly threads: number;
  readonly closed: boolean;
  /** Zeroes the centres, stops the worker; a listen() that has this session hears the SpeakersEvents stop ('failed'). */
  close(): Promise<void>;
  toString(): string;
}
