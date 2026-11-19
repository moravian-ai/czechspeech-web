"""Checks that two ONNX files give bit-identical outputs: python -I check-onnx.py <original> <fixed>.
Three random inputs [1, T, 80], T = 150, 300, 1000, seed 20261116. Needs only onnxruntime and numpy."""
import sys

import numpy as np
import onnxruntime as ort


def main() -> int:
    a, b = sys.argv[1:3]
    sa = ort.InferenceSession(a, providers=["CPUExecutionProvider"])
    sb = ort.InferenceSession(b, providers=["CPUExecutionProvider"])
    name_a, name_b = sa.get_inputs()[0].name, sb.get_inputs()[0].name
    rng = np.random.default_rng(20261116)
    worst, identical = 0.0, True
    for t in (150, 300, 1000):
        x = rng.standard_normal((1, t, 80)).astype(np.float32)
        ya = sa.run(None, {name_a: x})
        yb = sb.run(None, {name_b: x})
        for u, v in zip(ya, yb):
            worst = max(worst, float(np.max(np.abs(u - v))))
            identical &= bool(np.array_equal(u, v))
    print(f"max_abs_diff={worst}")
    print(f"outputs_identical={identical}")
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
