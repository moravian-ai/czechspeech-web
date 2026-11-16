// Decoding limits of one recognition, so that greedy decoding stops by itself in bounded time.
//
// Why: greedy decoding sometimes falls into a repetition loop on an utterance (the same phrase again and again). In
// whisper.cpp nothing stops it before 220 tokens (half the text context); a decoder that reaches that without an end
// token counts as failed, and the temperature fallback then decodes the window again at temperatures 0.2, 0.4, ... 1.0
// with 5 decoders each, again up to 220 tokens. Measured on WebGPU (Intel Xe-LPG, Chrome 156) with the real audio path
// of the meeting app's 3-hour soak test, a 3.9 s utterance took 26 s that way (220 tokens at 12.9 ms, then two fallback
// passes of 220 steps at about 53 ms each), and its final text was the same as without the fallback; in other passes of
// the loop it took up to 45 s, past the watchdog's 32 s, which then destroyed the GPU device.
//
// The cap, from the data (README, "Decoding limits"): 654 clean decodes on WebGPU of the SDK's ParCzech4Speech fixtures
// (e-rec, the 20 accuracy segments, the 129 segments of the numbers bucket) and the meeting app's FLEURS fixtures (its
// meeting recording, its sample, the 32 sentences of the soak loop cut at 16 frame phases) needed at most 8.55 text
// tokens per second of audio (a segment full of numbers); transcripts that doubled a phrase reached 11.4. 25 tokens per
// second is 2.9 times the highest legitimate rate, and leaves a loop room for the three copies the repetition guard
// (src/repetition.js) usually needs to recognise it (a long phrase in a short utterance may get only two). Very short
// utterances get at least 32 tokens.
//
// The exact rule: max_tokens = max(32, ceil(25 * s)) for s seconds of audio (at most 30), and whisper.cpp stops after
// the token with that index, so a decode stopped by the cap has max_tokens + 1 text tokens. whisper.cpp's own limit
// stops any decode after 220 text tokens, which comes first above 8.76 s of audio (tokenLimit).

/** Text tokens per second of audio that whisper.cpp may decode for one utterance (max_tokens). */
export const TOKENS_PER_SECOND = 25;
/** The smallest cap, for short utterances. */
export const MIN_TOKENS = 32;
/** whisper.cpp applies max_tokens per 30 s window. */
const WINDOW_SECONDS = 30;
/** whisper.cpp decodes at most this many text tokens per window, whatever max_tokens says (n_text_ctx / 2 - 4). */
export const WHISPER_TOKEN_LIMIT = 220;

/**
 * max_tokens for `nSamples` of 16 kHz audio at `tokensPerSecond` (the model's rate, default 25); the floor for
 * anything that is not a positive finite length.
 */
export function tokenCap(nSamples, tokensPerSecond = TOKENS_PER_SECOND) {
  if (!Number.isFinite(nSamples) || nSamples <= 0) return MIN_TOKENS;
  const seconds = Math.min(nSamples / 16000, WINDOW_SECONDS);
  return Math.max(MIN_TOKENS, Math.ceil(tokensPerSecond * seconds));
}

/**
 * How many text tokens a decode stopped by a token limit has per window: max_tokens + 1, or whisper.cpp's own limit
 * when that is smaller (from max_tokens 220 up, and without a cap).
 */
export function tokenLimit(maxTokens) {
  return maxTokens > 0 ? Math.min(maxTokens + 1, WHISPER_TOKEN_LIMIT) : WHISPER_TOKEN_LIMIT;
}

/**
 * What cs_whisper_transcribe gets for one recognition: { maxTokens (0: none), temperatureFallback }. No temperature
 * fallback: on the runaway it only repeated the loop at 4 times the cost per step, and on the four fixtures where it
 * ran (ParCzech numbers) the capped greedy result was as good or better (README, "Decoding limits"). The internal
 * engine options `tokenCap: false` and `temperatureFallback: true` restore whisper.cpp's defaults, for measurements;
 * `maxTokens: n` sets a fixed cap (tests). `tokensPerSecond` is the model's rate (src/catalog.js).
 */
export function decodingLimits(nSamples, options = {}, tokensPerSecond = TOKENS_PER_SECOND) {
  const o = options || {};
  let maxTokens = tokenCap(nSamples, tokensPerSecond);
  if (o.tokenCap === false) maxTokens = 0;
  else if (Number.isInteger(o.maxTokens) && o.maxTokens > 0) maxTokens = o.maxTokens;
  return { maxTokens, temperatureFallback: o.temperatureFallback === true };
}
