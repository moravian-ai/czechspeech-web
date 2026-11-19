# Speaker voiceprints in the browser: reproduction of the measured accuracy

`test/speakers/reproduce.mjs` runs the SDK's own parts in headless Chromium (Endpointer, `SpeakerEngine` in a Web Worker
with onnxruntime-web 1.31.1, 2 threads, cross-origin isolated page, `Tagger`) over the 20 four-person test meetings
(`test-S4-00` .. `test-S4-19`, synthetic meetings of 4 speakers, about 5 minutes each) and compares the result with the
measurement made in Python (sherpa-onnx). No names and no voiceprints are written anywhere; the test audio is only read
from the directory in `CZECHSPEECH_SPEAKER_MEETINGS`.

## Parity of the embeddings (`test/speakers/parity.mjs`)

The first 0.3 .. 10 s of one recording embedded in the browser (fbank in JavaScript + onnxruntime-web) against sherpa-onnx:
cosine 1.0000000 for every cut, both models, on an isolated page with 1 and with 2 threads and on a page that is not
isolated (1 thread asked for, and 2 asked for: the runtime falls back to 1 and `engine.threads` says so): 64 of 64 checks
counted, 0 requests outside the local server. The reference embeddings are deleted as soon as they are read.

## Level 1: speech probabilities as in the measurement (Silero v6.2 ONNX through onnxruntime-web)

Mode `enrolled`: every speaker has a stored voice made from the enrolment recording (windows of 3 s every 1 s of the speech
found by the VAD, averaged). Mode `anon`: nobody has one (only "Speaker n").

| Model, operating point | pieces | named ok (measured) | named ok (browser) | foreign names (measured / browser) | correct speaker incl. "Speaker n" (measured / browser) |
|---|---|---|---|---|---|
| fine-tuned, `strict` | 2495 (2495) | 0.880561 (2197) | 0.880561 (2197) | 16 / 16 | 0.943487 / 0.9435 |
| base, deployable | 2546 (2546) | 0.653574 (1664) | 0.653574 (1664) | 13 / 13 | 0.858209 / 0.8582 |

Mode `anon`, correct speaker: fine-tuned 0.894589 measured, 0.8946 in the browser; base 0.811076 measured, 0.8111 in the browser.
The reproduction is exact (the tolerance of the acceptance test is +-0.005 and 1 % of the pieces).

## Level 2: speech probabilities through the SDK's own path (WebAssembly Silero VAD, streaming with resets)

Only reported, not an acceptance criterion. The probabilities differ slightly from the ONNX ones, so the segmentation does too.

| Model | pieces | named ok | foreign names | correct (`enrolled`) | correct (`anon`) |
|---|---|---|---|---|---|
| fine-tuned | 2482 | 0.878324 (2180) | 17 | 0.9388 | 0.8699 |
| base | 2540 | 0.654724 (1663) | 7 | 0.8575 | 0.8079 |

## Accuracy by utterances

An utterance can contain several speakers; the pieces after the search for speaker changes are what is scored. For
the fine-tuned model at level 1, 509 of 623 utterances were split into pieces (the search ran), 114 were short and kept whole.
A sentence in the application shows all speakers it contains; this table is for the author and makes no claim about it.

## Input cap and memory

One embedding is computed from at most 10 s of audio (`maxInputMs`, never set below the 1.5 s window of the change search, so at least 1500: a lower value is refused with a RangeError); a longer piece goes through in equal chunks whose unit
embeddings are averaged. With the cap, level 1 gives exactly the numbers above (fine-tuned 2197 / 2495 named ok, 16 foreign
names; base 1664 / 2546, 13), so it costs nothing measurable in accuracy; level 2 is unchanged as well (2482 pieces, 0.878324, 17;
2540, 0.654724, 7).

Memory (`test/speakers/bench.mjs`; working set of the process tree of the browser the script launched, baseline after
creating the engine and two 10 s embeddings, then all timings and one more 28 s input; MB):

| threads | cap | after create + warm-up | after the 28 s input | growth |
|---|---|---|---|---|
| 1 | 10 s | 550 | 472 | -78 (none) |
| 2 | 10 s | 563 | 489 | -74 (none) |
| 1 | none (100 s) | 553 | 680 | +127 |
| 2 | none (100 s) | 559 | 681 | +122 |

The absolute figures are the whole headless browser (several processes), not the speaker engine alone. The working set is the
current one at the moment of reading, so a falling value says only that nothing grew. Earlier figures (464 MB growth) took the
peak of the largest of all Chrome processes on the machine against a baseline taken before the model was loaded and are not
comparable. `performance.measureUserAgentSpecificMemory()` is not available in the headless shell; the script uses it where it is.

Time with the cap on the reference notebook, 2 threads: `embed` 10 s 734 ms, 28 s 2018 ms (three chunks), `embedUtterance` 10 s
2627 ms; 1 thread: 1093, 2952 and 3960 ms.

## Versions and time

onnxruntime-web 1.31.1 (wasm 14 848 852 B), Chromium of playwright-core 1.64.1, 2 threads. Each model took about 50 minutes for
both levels on the reference notebook, running one embedding at a time (no backlog, so splitting is always on).
