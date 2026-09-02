"""Cross-checks MnistNet's (model.py) per-layer MAC counts against the
figures `proxima-tensor/docs/discipline.md` cites for the mnist f32 lane
(conv1 48,672 / conv2 663,552 / conv3 1,672,704 / fc1 371,712 / fc2 320,
total 2,756,960 MACs/image) -- computed from the SAME module the bench
scripts actually run, not re-derived by hand and hoped to match.
"""

from __future__ import annotations

from model import load_model

EXPECTED = {
    "conv1": 48_672,
    "conv2": 663_552,
    "conv3": 1_672_704,
    "fc1": 371_712,
    "fc2": 320,
}
EXPECTED_TOTAL = 2_756_960


def conv_macs(out_channels: int, in_channels: int, kernel: int, out_h: int, out_w: int) -> int:
    return out_channels * in_channels * kernel * kernel * out_h * out_w


def linear_macs(in_features: int, out_features: int) -> int:
    return in_features * out_features


def main() -> None:
    model = load_model()

    # spatial extents derived from the same 28x28 input and kernel=3, stride=1,
    # no padding chain the module (and mnist.onnx) both use
    input_size = 28
    conv1_out = input_size - 2  # 26
    conv2_out = conv1_out - 2  # 24
    conv3_out = conv2_out - 2  # 22

    measured = {
        "conv1": conv_macs(model.conv1.out_channels, model.conv1.in_channels, 3, conv1_out, conv1_out),
        "conv2": conv_macs(model.conv2.out_channels, model.conv2.in_channels, 3, conv2_out, conv2_out),
        "conv3": conv_macs(model.conv3.out_channels, model.conv3.in_channels, 3, conv3_out, conv3_out),
        "fc1": linear_macs(model.fc1.in_features, model.fc1.out_features),
        "fc2": linear_macs(model.fc2.in_features, model.fc2.out_features),
    }
    measured_total = sum(measured.values())

    print("layer      measured        expected        match")
    all_match = True
    for layer, expected_value in EXPECTED.items():
        measured_value = measured[layer]
        match = measured_value == expected_value
        all_match = all_match and match
        print(f"{layer:<10} {measured_value:>12,}  {expected_value:>12,}  {'PASS' if match else 'FAIL'}")

    total_match = measured_total == EXPECTED_TOTAL
    all_match = all_match and total_match
    print(f"{'total':<10} {measured_total:>12,}  {EXPECTED_TOTAL:>12,}  {'PASS' if total_match else 'FAIL'}")

    if not all_match:
        raise SystemExit("MAC-count cross-check FAILED: model.py's architecture does not match discipline.md's cited MAC counts -- do not report timing numbers for this model as ROW 189's network")

    print("MAC-count cross-check PASSED: model.py's architecture matches discipline.md's cited per-layer MAC counts exactly")


if __name__ == "__main__":
    main()
