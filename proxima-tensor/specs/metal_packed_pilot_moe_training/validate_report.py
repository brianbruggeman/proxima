import json
import sys
from pathlib import Path


EXPECTED_CODECS = {"Bf8E5M2", "Bf4E2M1"}
EXPECTED_TOKEN_IDS = [0, 1, 2, 3, 0, 1, 2, 3]
EXPECTED_TARGET_IDS = [1, 2, 3, 0, 1, 2, 3, 0]
EXPECTED_PACKED_LENGTHS = {"Bf8E5M2": 32, "Bf4E2M1": 16}
BACKENDS = {
    "metal": "metal",
    "cpu_reference": "cpu",
    "scalar_reference": "scalar",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_values(record, label, backend):
    values = record if label == "metal" else record[label]
    require(values["backend"] == backend, f"{label}: backend mismatch")
    require(len(values["logits"]) == 8 and all(len(row) == 4 for row in values["logits"]), f"{label}: logits shape")
    require(len(values["token_losses"]) == 8, f"{label}: token loss shape")
    require(isinstance(values["mean_loss"], (int, float)), f"{label}: mean loss missing")
    require(len(values["compact_gradients"]) == 8 and all(len(row) == 16 for row in values["compact_gradients"]), f"{label}: compact gradient shape")
    require(len(values["coalesced_gradients"]) == 2 and all(len(row) == 16 for row in values["coalesced_gradients"]), f"{label}: coalesced gradient shape")
    for field in ("updated_masters", "first_moment", "second_moment"):
        require(len(values[field]) == 32, f"{label}: {field} shape")


def validate(path):
    records = json.loads(Path(path).read_text())
    require(isinstance(records, list) and len(records) == 2, "expected exactly two records")
    require({record["codec"] for record in records} == EXPECTED_CODECS, "codec set mismatch")
    for record in records:
        require(record["seed"] == 17 and record["step"] == 1, "seed/step mismatch")
        require(record["routes"] == [0, 1, 0, 1, 0, 1, 0, 1], "routes mismatch")
        require(record["token_ids"] == EXPECTED_TOKEN_IDS, "token IDs mismatch")
        require(record["target_ids"] == EXPECTED_TARGET_IDS, "target IDs mismatch")
        require(len(record["targets"]) == 8 and all(len(row) == 4 for row in record["targets"]), "target shape")
        require(len(record["initial_masters"]) == 32, "master shape")
        require(record["packed_byte_len"] == EXPECTED_PACKED_LENGTHS[record["codec"]], "codec packed length mismatch")
        require(len(record["packed_bytes"]) == EXPECTED_PACKED_LENGTHS[record["codec"]], "packed byte payload length mismatch")
        for label, backend in BACKENDS.items():
            validate_values(record, label, backend)


if __name__ == "__main__":
    validate(sys.argv[1])
