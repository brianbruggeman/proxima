"""Re-proves proxima-onnx's own accuracy claim for mnist.onnx
(tests/real_mnist_accuracy.rs: 0.9900, 990/1000 t10k images) and its
zero-input logit parity gate (tests/real_mnist_checkpoint.rs) against the
real PyTorch execution of the same checkpoint. See README.md for the
recorded reference numbers this guards.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import dataset_present, load_labels, load_normalized_images, test_images_path, test_labels_path  # noqa: E402
from model import load_model  # noqa: E402

TEST_IMAGES_COUNT = 1000
LOGIT_PARITY_TOLERANCE = 1e-3

# proxima-onnx/tests/real_mnist_checkpoint.rs's own zero-input evaluation,
# `real_mnist evaluation SUCCEEDED: ... values=[...]`
REFERENCE_ZERO_INPUT_LOGITS = [
    -1.422484,
    -2.4826293,
    -1.3939257,
    -3.2908063,
    -3.8867984,
    -1.478745,
    -3.6286578,
    -3.1661224,
    -3.1980472,
    -3.440639,
]


def main() -> None:
    if not dataset_present():
        print(f"skipping: no host-local MNIST idx dataset under {test_images_path().parent.parent}")
        return

    model = load_model()

    with torch.no_grad():
        zero_input = torch.zeros(1, 1, 28, 28, dtype=torch.float32)
        zero_logits = model(zero_input).squeeze(0).numpy()
    reference = np.array(REFERENCE_ZERO_INPUT_LOGITS, dtype=np.float32)
    max_logit_diff = float(np.max(np.abs(zero_logits - reference)))
    print(f"zero-input logits: {zero_logits.tolist()}")
    print(f"max abs diff vs proxima-onnx reference: {max_logit_diff:.3e}")

    images = load_normalized_images(test_images_path(), TEST_IMAGES_COUNT)
    labels = load_labels(test_labels_path(), TEST_IMAGES_COUNT)
    assert len(images) == len(labels), "same number of images and labels"

    batch = torch.from_numpy(images).unsqueeze(1)
    start = time.perf_counter()
    with torch.no_grad():
        logits = model(batch)
    elapsed = time.perf_counter() - start
    predicted = logits.argmax(dim=1).numpy()
    correct = int((predicted == labels).sum())
    accuracy = correct / len(labels)

    print(f"accuracy: {accuracy:.4f} ({correct}/{len(labels)} images) in {elapsed:.4f}s")

    if max_logit_diff > LOGIT_PARITY_TOLERANCE:
        raise SystemExit(f"logit parity gate failed: max_diff={max_logit_diff:.3e} exceeds {LOGIT_PARITY_TOLERANCE:.0e}")


if __name__ == "__main__":
    main()
