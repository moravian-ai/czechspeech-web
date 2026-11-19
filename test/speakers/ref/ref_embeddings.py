"""Reference embeddings from sherpa-onnx (Python, 1 thread) of the first 0.3 .. 10 s of e-rec.wav (10 s: the recording
repeated from its start, as the benchmark does). They are biometric data: written only to test/speakers/out/ (git-ignored).

    python -I test/speakers/ref/ref_embeddings.py <model.onnx> <e-rec.wav> <out.json>
"""
import json
import sys
import wave

import numpy as np
import sherpa_onnx

CUTS = (0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)


def main() -> int:
    model, wav, out = sys.argv[1:4]
    with wave.open(wav) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        src = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    ex = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=model, num_threads=1))
    result = {}
    for s in CUTS:
        x = np.resize(src, round(s * 16000))
        st = ex.create_stream()
        st.accept_waveform(sample_rate=16000, waveform=x)
        st.input_finished()
        v = np.asarray(ex.compute(st), dtype=np.float64)
        result[f"{s:g}"] = (v / np.linalg.norm(v)).tolist()
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f)
    print(f"wrote {len(result)} reference embeddings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
