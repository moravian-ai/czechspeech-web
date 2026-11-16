// CzechSpeech WebAssembly glue: a plain C API between the JavaScript workers (src/engine-wasm.js) and whisper.cpp.
//
// Adapted from the CzechSpeech Android SDK 0.1.0 (native/src/czechspeech_jni.c in czechspeech-android, same
// licence as this package): the whisper.cpp parameters, the streaming Silero VAD (no_reset + an explicit reset right
// after init), the speech gate through whisper_vad_segments_from_samples, the trimmed encoder window (audio_ctx) and
// the abort / encoder_begin callbacks are the same. The decoding limits differ: max_tokens relative to the audio
// length and no temperature fallback (cs_whisper_transcribe). The JNI layer is replaced by exported C functions that take
// pointers into the WebAssembly heap. The Android glue in turn follows parts of whisper.cpp
// examples/whisper.android/lib/src/main/jni/whisper/jni.c (MIT licence, Copyright (c) 2023-2026 The ggml authors).
// See THIRD_PARTY_NOTICES.
//
// Privacy rules implemented here and in src/engine-wasm.js:
//   - Audio arrives in a buffer that the JavaScript side allocates in the WebAssembly heap, fills, and zeroes and
//     frees right after the call. This file never copies audio anywhere else.
//   - The transcript is concatenated into one malloc'd buffer that JavaScript copies out; cs_text_release() zeroes
//     and frees it. Nothing is logged except whisper.cpp/ggml warnings and errors, which never contain transcript
//     text (INFO and DEBUG are dropped: DEBUG output of whisper.cpp includes decoded tokens).
//   - No file or network I/O: the model is pulled through a loader callback (cs_js_model_read) and the VAD model is
//     passed as bytes.
//
// The same file is compiled into the CPU modules and the WebGPU module (native/CMakeLists.txt). In the WebGPU module
// cs_backend_init, cs_whisper_init, cs_whisper_transcribe and cs_whisper_free can wait for the GPU and are exported
// through JSPI (they return Promises); the Silero VAD always runs on ggml's CPU backend.

#include <emscripten/emscripten.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml-backend.h"
#include "whisper.h"

// ---------------------------------------------------------------------------------------------------------------------
// JavaScript callbacks (implemented by src/engine-wasm.js on the Module object)

// Non-zero when the page asked to cancel work enqueued at `epoch` (a shared counter moved past it). Called on the
// thread that runs whisper_full, which is the worker that owns the module.
EM_JS(int, cs_js_abort_requested, (int epoch), {
    return Module.czechspeechAbortRequested ? (Module.czechspeechAbortRequested(epoch) ? 1 : 0) : 0;
});

// Copies the next `n` bytes of the model into the heap at `dst`; returns the number of bytes copied.
EM_JS(size_t, cs_js_model_read, (void *dst, size_t n), {
    return Module.czechspeechModelRead(dst, n);
});

EM_JS(int, cs_js_model_eof, (void), {
    return Module.czechspeechModelEof() ? 1 : 0;
});

// ---------------------------------------------------------------------------------------------------------------------
// helpers

static void cs_secure_zero(void *p, size_t n) {
    if (p == NULL || n == 0) {
        return;
    }
    memset(p, 0, n);
    // Keep the compiler from dropping the memset as a dead store before free().
    __asm__ __volatile__("" : : "r"(p) : "memory");
}

// Development diagnostics (cs_set_verbose, reachable only through an internal engine option): INFO lines too. They
// carry sizes, buffer placement and timings, never text; DEBUG (which includes decoded tokens) is always dropped.
static int cs_verbose = 0;

// whisper.cpp and ggml log through this callback (set in cs_init). Only warnings and errors pass; they go to stderr,
// which the worker forwards to the page's logger.
static void cs_log_callback(enum ggml_log_level level, const char *text, void *user_data) {
    (void) user_data;
    static _Thread_local int keep_continuation = 0;
    int keep;
    if (level == GGML_LOG_LEVEL_CONT) {
        keep = keep_continuation;
    } else {
        keep = level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR ||
               (cs_verbose && level == GGML_LOG_LEVEL_INFO);
        keep_continuation = keep;
    }
    if (!keep || text == NULL) {
        return;
    }
    fputs(text, stderr);
}

EMSCRIPTEN_KEEPALIVE void cs_init(void) {
    whisper_log_set(cs_log_callback, NULL);
}

EMSCRIPTEN_KEEPALIVE void cs_set_verbose(int on) {
    cs_verbose = on != 0;
}

EMSCRIPTEN_KEEPALIVE const char *cs_system_info(void) {
    return whisper_print_system_info();
}

// Zeroes and frees a buffer the JavaScript side allocated (audio copies).
EMSCRIPTEN_KEEPALIVE void cs_zero_free(void *p, size_t n) {
    cs_secure_zero(p, n);
    free(p);
}

// Builds ggml's backend registry and returns the number of GPU devices whose backend could be started (0 in the CPU
// modules, or when WebGPU gave no adapter/device; -1 when a GPU device was found but its backend did not start).
// In the WebGPU module this is the first call that reaches ggml: registering the WebGPU backend requests the adapter
// and creates the device (ggml-webgpu keeps one device for the module's lifetime); starting and freeing a backend
// once also compiles its first pipeline, so a GPU that cannot run ggml fails here and not with the first utterance.
EMSCRIPTEN_KEEPALIVE int cs_backend_init(void) {
    int n_gpu = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }
        ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
        if (backend == NULL) {
            return -1;
        }
        ggml_backend_free(backend);
        n_gpu++;
    }
    return n_gpu;
}

// ---------------------------------------------------------------------------------------------------------------------
// whisper

struct cs_whisper {
    struct whisper_context *ctx;
    char *text;       // last transcript (UTF-8, not NUL-terminated), owned until cs_text_release
    size_t text_len;
    int tokens;       // text tokens of the last transcript
    int capped;       // the last decode stopped at a token limit instead of its end token
    const char *language;  // language of the last transcript: a whisper_lang_str entry (static), NULL when none
};

struct cs_abort {
    int epoch;
};

// whisper.cpp (pinned commit) checks abort_callback after the encoder pass and after each decoder step, not inside
// the encoder graph; encoder_begin_callback adds a check before the encoder starts.
static bool cs_abort_callback(void *data) {
    const struct cs_abort *a = (const struct cs_abort *) data;
    return cs_js_abort_requested(a->epoch) != 0;
}

static bool cs_encoder_begin_callback(struct whisper_context *ctx, struct whisper_state *state, void *data) {
    (void) ctx;
    (void) state;
    return !cs_abort_callback(data);
}

// Encoder window trimmed to the utterance: 20 ms mel frames + 32, rounded up to a multiple of 16; the full
// window (0) from 1500 frames (30 s) up. Never shorter than CS_MIN_AUDIO_CTX: the model was never trained on a
// window that short, and below 4 s of speech it then invents or repeats sentences (15-46 % of utterances).
// A floor of 512 frames (~10 s) cut WER on short utterances from 6.28 to 3.90 % (clean) and 10.82 to 9.89 %
// (noise) and removed the invented-word bursts, against the trimmed window on the same short utterances (CHANGELOG,
// "Encoder window").
#define CS_MIN_AUDIO_CTX 512

static int cs_audio_ctx_for(int n_samples) {
    int frames = (n_samples + 319) / 320 + 32;
    frames = (frames + 15) / 16 * 16;
    if (frames < CS_MIN_AUDIO_CTX) frames = CS_MIN_AUDIO_CTX;
    return frames < 1500 ? frames : 0;
}

// The window whisper.cpp gets for an utterance of n_samples samples (exported for the tests).
EMSCRIPTEN_KEEPALIVE int cs_audio_ctx(int n_samples) {
    return cs_audio_ctx_for(n_samples);
}

static size_t cs_loader_read(void *ctx, void *output, size_t read_size) {
    (void) ctx;
    return cs_js_model_read(output, read_size);
}

static bool cs_loader_eof(void *ctx) {
    (void) ctx;
    return cs_js_model_eof() != 0;
}

static void cs_loader_close(void *ctx) {
    (void) ctx;
}

// Loads the model through the JavaScript reader (Module.czechspeechModelRead). NULL on failure.
// use_gpu: weights and compute on the first GPU device (WebGPU module), else the CPU backend. flash_attn: whisper.cpp's
// default (1) everywhere so far; 0 selects the matrix-multiplication attention path.
EMSCRIPTEN_KEEPALIVE struct cs_whisper *cs_whisper_init(int use_gpu, int flash_attn) {
    struct whisper_model_loader loader = {
        .context = NULL,
        .read    = cs_loader_read,
        .eof     = cs_loader_eof,
        .close   = cs_loader_close,
    };
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu    = use_gpu != 0;
    cparams.flash_attn = flash_attn != 0;
    struct whisper_context *ctx = whisper_init_with_params(&loader, cparams);
    if (ctx == NULL) {
        return NULL;
    }
    struct cs_whisper *w = (struct cs_whisper *) calloc(1, sizeof(struct cs_whisper));
    if (w == NULL) {
        whisper_free(ctx);
        return NULL;
    }
    w->ctx = ctx;
    return w;
}

EMSCRIPTEN_KEEPALIVE void cs_text_release(struct cs_whisper *w) {
    if (w == NULL || w->text == NULL) {
        return;
    }
    cs_secure_zero(w->text, w->text_len);
    free(w->text);
    w->text = NULL;
    w->text_len = 0;
}

EMSCRIPTEN_KEEPALIVE void cs_whisper_free(struct cs_whisper *w) {
    if (w == NULL) {
        return;
    }
    cs_text_release(w);
    whisper_free(w->ctx);
    free(w);
}

EMSCRIPTEN_KEEPALIVE const char *cs_text_ptr(struct cs_whisper *w) {
    return w != NULL ? w->text : NULL;
}

EMSCRIPTEN_KEEPALIVE int cs_text_len(struct cs_whisper *w) {
    return w != NULL ? (int) w->text_len : 0;
}

#define CS_MAX_LANGUAGES 128

// Splits a comma-separated list of whisper language codes ("cs", "pl,en") into whisper.cpp language ids. Returns the count,
// or -1 when the list is empty or holds a code whisper.cpp does not know (or more than max entries).
static int cs_parse_languages(const char *languages, int *ids, int max) {
    int n = 0;
    const char *p = languages;
    while (p != NULL && *p != '\0') {
        const char *end = strchr(p, ',');
        const size_t len = end != NULL ? (size_t) (end - p) : strlen(p);
        char code[8];
        if (len == 0 || len >= sizeof(code) || n >= max) {
            return -1;
        }
        memcpy(code, p, len);
        code[len] = '\0';
        const int id = whisper_lang_id(code);
        if (id < 0 || strcmp(whisper_lang_str(id), code) != 0) {
            return -1;  // unknown, or a full name ("czech") that whisper_lang_id also accepts
        }
        ids[n++] = id;
        if (end != NULL && end[1] == '\0') {
            return -1;  // a trailing comma
        }
        p = end != NULL ? end + 1 : p + len;
    }
    return n > 0 ? n : -1;
}

// Runs Whisper on n_samples of 16 kHz mono audio at pcm (owned by the caller, not modified).
// max_tokens: the most tokens decoded per 30 s window (0: whisper.cpp's limit, half the text context, 220 tokens);
// src/decoding.js derives it from the audio length, so that a repetition loop ends by itself. temperature_fallback:
// 0 decodes once, greedily at temperature 0; 1 keeps whisper.cpp's fallback (a failed decode is repeated at
// temperatures 0.2 ... 1.0 with 5 decoders each), which on a looping utterance cost 4 times as much per step and gave
// the same text.
// Returns 0 with the transcript in cs_text_ptr/cs_text_len (possibly empty), 1 when cancelled through the abort
// epoch (by the page, or by the recogniser's watchdog), -1 when whisper_full failed. cs_whisper_tokens and
// cs_whisper_capped then tell how many text tokens it has and whether a token limit stopped the decode (max_tokens,
// or whisper.cpp's own 220).
// languages: comma-separated whisper language codes (two or three lower-case letters: "cs", "pl,en"), the languages
// the model speaks and the caller allows.
// One code forces that language, exactly as before (no detection pass). Several: the language is detected first (the
// mel spectrogram, one encoder pass and one decoder step) and the most probable one of the list is used. whisper.cpp's
// own detection cannot be limited to a list, and runs before audio_ctx applies. An unknown or empty list gives -1.
// The detection cannot be interrupted (whisper_lang_auto_detect_with_state takes no abort callback), so the abort
// epoch is checked twice: before the detection state is allocated, and right after the detection. A call cancelled
// at either point returns 1 without decoding. cs_whisper_language then tells the language of the transcript (valid
// after a return of 0).
EMSCRIPTEN_KEEPALIVE int cs_whisper_transcribe(struct cs_whisper *w, int epoch, const float *pcm, int n_samples,
                                               int n_threads, int max_tokens, int temperature_fallback,
                                               const char *languages) {
    if (w == NULL || w->ctx == NULL || pcm == NULL || n_samples < 0) {
        return -1;
    }
    cs_text_release(w);
    w->tokens = 0;
    w->capped = 0;
    w->language = NULL;

    int ids[CS_MAX_LANGUAGES];
    const int n_languages = cs_parse_languages(languages, ids, CS_MAX_LANGUAGES);
    if (n_languages < 0) {
        return -1;
    }
    struct cs_abort abort_data = { epoch };

    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.n_threads        = n_threads > 0 ? n_threads : 1;
    params.translate        = false;
    params.language         = whisper_lang_str(ids[0]);
    params.detect_language  = false;
    params.no_context       = true;   // no text carried over between utterances
    params.no_timestamps    = true;
    // whisper.cpp keeps suppress_blank on, which forbids the end of the text as the first token. A model that learned to
    // answer a segment without speech with an empty transcript then has to write something into silence:
    // 100 % of 360 such segments with it on, 0 % off; speech transcripts of the CzechSpeech model unchanged (measured
    // on 12 Nov 2026). The empty result is dropped by the callers.
    params.suppress_blank   = false;
    params.single_segment   = false;
    params.offset_ms        = 0;
    params.print_special    = false;
    params.print_progress   = false;
    params.print_realtime   = false;  // would printf the text
    params.print_timestamps = false;
    params.audio_ctx        = cs_audio_ctx(n_samples);
    params.max_tokens       = max_tokens > 0 ? max_tokens : 0;
    if (!temperature_fallback) {
        params.temperature_inc = 0.0f;  // one greedy decode at temperature 0
    }

    params.abort_callback                   = cs_abort_callback;
    params.abort_callback_user_data         = &abort_data;
    params.encoder_begin_callback           = cs_encoder_begin_callback;
    params.encoder_begin_callback_user_data = &abort_data;

    whisper_reset_timings(w->ctx);
    if (n_languages > 1) {
        if (cs_abort_callback(&abort_data)) {
            return 1;  // already cancelled: no state, mel or encoder pass
        }
        // A fresh state per call is deliberate (a long-lived second one would double the steady memory); its cost is measured in W4.
        struct whisper_state *state = whisper_init_state(w->ctx);
        if (state == NULL) {
            return -1;
        }
        // A state of its own: the context's default one keeps the audio_ctx of the previous whisper_full, which
        // the detection would inherit; a fresh one reads the whole 30 s window. Freed again below (not kept).
        int best = -1;
        const int max_id = whisper_lang_max_id();
        float *probs = (float *) calloc((size_t) max_id + 1, sizeof(float));
        if (probs != NULL && whisper_pcm_to_mel_with_state(w->ctx, state, pcm, n_samples, params.n_threads) == 0 &&
            whisper_lang_auto_detect_with_state(w->ctx, state, 0, params.n_threads, probs) >= 0) {
            for (int i = 0; i < n_languages; i++) {
                if (best < 0 || probs[ids[i]] > probs[best]) {
                    best = ids[i];
                }
            }
        }
        free(probs);
        whisper_free_state(state);
        if (cs_abort_callback(&abort_data)) {
            return 1;
        }
        if (best < 0) {
            return -1;
        }
        params.language = whisper_lang_str(best);
    }
    const int rc = whisper_full(w->ctx, params, pcm, n_samples);
    if (cs_abort_callback(&abort_data)) {
        return 1;
    }
    if (rc != 0) {
        return -1;
    }

    const int n_segments = whisper_full_n_segments(w->ctx);
    // With no_timestamps each 30 s window is one segment, ending with the end token unless a token limit stopped it:
    // max_tokens (a capped decode has max_tokens + 1 text tokens) or, from 8.8 s of audio up, whisper.cpp's own 220.
    const whisper_token eot = whisper_token_eot(w->ctx);
    size_t total = 0;
    for (int i = 0; i < n_segments; i++) {
        const char *seg = whisper_full_get_segment_text(w->ctx, i);
        if (seg != NULL) {
            total += strlen(seg);
        }
        const int n = whisper_full_n_tokens(w->ctx, i);
        for (int j = 0; j < n; j++) {
            if (whisper_full_get_token_id(w->ctx, i, j) < eot) {
                w->tokens++;
            }
        }
        if (n > 0 && whisper_full_get_token_id(w->ctx, i, n - 1) < eot) {
            w->capped = 1;
        }
    }
    char *text = (char *) malloc(total > 0 ? total : 1);
    if (text == NULL) {
        return -1;
    }
    size_t pos = 0;
    for (int i = 0; i < n_segments; i++) {
        const char *seg = whisper_full_get_segment_text(w->ctx, i);
        if (seg != NULL) {
            const size_t len = strlen(seg);
            memcpy(text + pos, seg, len);
            pos += len;
        }
    }
    w->text = text;
    w->text_len = total;
    w->language = params.language;
    return 0;
}

// Language of the last transcript (whisper language code; a static string owned by whisper.cpp); "" when there is none.
EMSCRIPTEN_KEEPALIVE const char *cs_whisper_language(struct cs_whisper *w) {
    return w != NULL && w->language != NULL ? w->language : "";
}

// Text tokens of the last transcript (diagnostics, no content).
EMSCRIPTEN_KEEPALIVE int cs_whisper_tokens(struct cs_whisper *w) {
    return w != NULL ? w->tokens : 0;
}

// 1 when the last decode was stopped by a token limit (src/decoding.js, or whisper.cpp's own) instead of ending.
EMSCRIPTEN_KEEPALIVE int cs_whisper_capped(struct cs_whisper *w) {
    return w != NULL ? w->capped : 0;
}

// Timings of the last cs_whisper_transcribe (diagnostics, no content): out[0] encoder ms per encode pass,
// out[1] decoder ms per token, out[2] batched (prompt) decode ms per batch.
EMSCRIPTEN_KEEPALIVE void cs_whisper_timings(struct cs_whisper *w, float *out) {
    if (out == NULL) {
        return;
    }
    out[0] = out[1] = out[2] = 0.0f;
    if (w == NULL || w->ctx == NULL) {
        return;
    }
    struct whisper_timings *t = whisper_get_timings(w->ctx);  // allocated by whisper.cpp with new (malloc)
    if (t != NULL) {
        out[0] = t->encode_ms;
        out[1] = t->decode_ms;
        out[2] = t->batchd_ms;
        free(t);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Silero VAD

struct cs_mem_reader {
    const uint8_t *data;
    size_t size;
    size_t pos;
};

static size_t cs_mem_read(void *ctx, void *output, size_t read_size) {
    struct cs_mem_reader *r = (struct cs_mem_reader *) ctx;
    const size_t left = r->size - r->pos;
    const size_t n = read_size < left ? read_size : left;
    memcpy(output, r->data + r->pos, n);
    r->pos += n;
    return n;
}

static bool cs_mem_eof(void *ctx) {
    const struct cs_mem_reader *r = (const struct cs_mem_reader *) ctx;
    return r->pos >= r->size;
}

static void cs_mem_close(void *ctx) {
    (void) ctx;
}

// Loads Silero VAD from the ggml file bytes at model (owned by the caller). NULL on failure.
EMSCRIPTEN_KEEPALIVE struct whisper_vad_context *cs_vad_init(const uint8_t *model, size_t size, int n_threads) {
    if (model == NULL || size == 0) {
        return NULL;
    }
    struct cs_mem_reader reader = { model, size, 0 };
    struct whisper_model_loader loader = {
        .context = &reader,
        .read    = cs_mem_read,
        .eof     = cs_mem_eof,
        .close   = cs_mem_close,
    };
    struct whisper_vad_context_params cparams = whisper_vad_default_context_params();
    cparams.n_threads = n_threads > 0 ? n_threads : 1;
    cparams.use_gpu   = false;
    struct whisper_vad_context *vctx = whisper_vad_init_with_params(&loader, cparams);
    if (vctx != NULL) {
        // whisper_vad_init allocates the LSTM state without zeroing it; the streaming calls do not reset it, so
        // without this the first seconds would depend on whatever the heap held (e.g. a previous session's state).
        whisper_vad_reset_state(vctx);
    }
    return vctx;
}

EMSCRIPTEN_KEEPALIVE void cs_vad_free(struct whisper_vad_context *vctx) {
    if (vctx != NULL) {
        whisper_vad_free(vctx);
    }
}

EMSCRIPTEN_KEEPALIVE void cs_vad_reset(struct whisper_vad_context *vctx) {
    if (vctx != NULL) {
        whisper_vad_reset_state(vctx);
    }
}

// Speech probabilities for consecutive 512-sample windows (the last one zero-padded), continuing the LSTM state of
// earlier calls. Writes up to cap values to out and returns the number of windows, or -1 on error.
EMSCRIPTEN_KEEPALIVE int cs_vad_probs(struct whisper_vad_context *vctx, const float *pcm, int n_samples, float *out,
                                      int cap) {
    if (vctx == NULL || pcm == NULL || out == NULL || n_samples < 0) {
        return -1;
    }
    if (!whisper_vad_detect_speech_no_reset(vctx, pcm, n_samples)) {
        return -1;
    }
    const int n_probs = whisper_vad_n_probs(vctx);
    const int n_out = n_probs < cap ? n_probs : cap;
    if (n_out > 0) {
        memcpy(out, whisper_vad_probs(vctx), (size_t) n_out * sizeof(float));
    }
    return n_probs;
}

// Number of speech segments in the samples by whisper_vad_segments_from_samples (fresh LSTM state, default
// segment parameters except the threshold), or -1 on error.
EMSCRIPTEN_KEEPALIVE int cs_vad_speech_segments(struct whisper_vad_context *vctx, const float *pcm, int n_samples,
                                                float threshold) {
    if (vctx == NULL || pcm == NULL || n_samples < 0) {
        return -1;
    }
    struct whisper_vad_params params = whisper_vad_default_params();
    params.threshold = threshold;
    struct whisper_vad_segments *segments = whisper_vad_segments_from_samples(vctx, params, pcm, n_samples);
    if (segments == NULL) {
        return -1;
    }
    const int n = whisper_vad_segments_n_segments(segments);
    whisper_vad_free_segments(segments);
    return n;
}
