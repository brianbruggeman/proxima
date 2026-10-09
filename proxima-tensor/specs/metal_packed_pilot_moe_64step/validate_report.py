#!/usr/bin/env python3
"""Validate retained Metal BF8/BF4 64-step training payloads."""

import argparse
import importlib.util
import json
import math
from pathlib import Path

ENCODER_PATH = (
    Path(__file__).parents[1]
    / "packed_pilot_moe_training"
    / "validate_report.py"
)
ENCODER_SPEC = importlib.util.spec_from_file_location("packed_pilot_validator", ENCODER_PATH)
if ENCODER_SPEC is None or ENCODER_SPEC.loader is None:
    raise RuntimeError(f"cannot load codec encoder at {ENCODER_PATH}")
ENCODER = importlib.util.module_from_spec(ENCODER_SPEC)
ENCODER_SPEC.loader.exec_module(ENCODER)

CODECS = ("Bf8E5M2", "Bf4E2M1")
CODEC_FORMATS = {"Bf8E5M2": "bf8_e5m2", "Bf4E2M1": "bf4_e2m1"}
BYTE_LENGTHS = {"Bf8E5M2": 32, "Bf4E2M1": 16}
TRAIN_TOKENS = [0, 1, 2, 3, 0, 1, 2, 3]
TRAIN_TARGETS = [1, 2, 3, 0, 1, 2, 3, 0]
TRAIN_ROUTES = [0, 1, 0, 1, 0, 1, 0, 1]
HELD_OUT_TOKENS = [0, 0, 1, 1, 2, 2, 3, 3]
HELD_OUT_TARGETS = [0, 1, 1, 2, 2, 3, 3, 0]
HELD_OUT_ROUTES = [0, 0, 1, 1, 0, 0, 1, 1]
TOP_FIELDS = {
    "codec", "seed", "step", "backend", "plan_build_count", "plan_execution_count",
    "master_input", "first_moment_input",
    "second_moment_input", "train_packed_bytes", "routes", "token_ids",
    "target_ids", "train_logits", "train_token_losses", "train_mean_loss",
    "compact_gradients", "coalesced_gradients", "updated_masters",
    "updated_first_moment", "updated_second_moment", "held_out_packed_bytes",
    "held_out_routes", "held_out_token_ids", "held_out_target_ids",
    "held_out_logits", "held_out_token_losses", "held_out_mean_loss",
    "cpu_reference", "scalar_reference",
}
NESTED_FIELDS = {
    "backend", "train_logits", "train_token_losses", "train_mean_loss",
    "compact_gradients", "coalesced_gradients", "updated_masters",
    "updated_first_moment", "updated_second_moment", "held_out_logits",
    "held_out_token_losses", "held_out_mean_loss",
}
NUMERIC_FIELDS = (
    "train_logits", "train_token_losses", "train_mean_loss", "compact_gradients",
    "coalesced_gradients", "updated_masters", "updated_first_moment",
    "updated_second_moment", "held_out_logits", "held_out_token_losses",
    "held_out_mean_loss",
)


def flatten(value):
    if isinstance(value, list):
        for item in value:
            yield from flatten(item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)


def require_shape(value, shape, label):
    if not shape:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{label} must be a number")
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite")
        return
    if not isinstance(value, list) or len(value) != shape[0]:
        raise ValueError(f"{label} must have shape {shape}")
    for index, item in enumerate(value):
        require_shape(item, shape[1:], f"{label}[{index}]")


def close_payload(actual, expected, label):
    actual_values = list(flatten(actual))
    expected_values = list(flatten(expected))
    if len(actual_values) != len(expected_values):
        raise ValueError(f"{label} numeric payload lengths differ")
    for index, (actual_value, expected_value) in enumerate(zip(actual_values, expected_values)):
        if not math.isfinite(actual_value) or not math.isfinite(expected_value):
            raise ValueError(f"{label}[{index}] is non-finite")
        if abs(actual_value - expected_value) > 1e-6:
            raise ValueError(f"{label}[{index}] differs beyond 1e-6")


def validate(records):
    if not isinstance(records, list) or len(records) != 128:
        raise ValueError("report must contain exactly 128 records")
    for record_index, record in enumerate(records):
        codec_index, step_index = divmod(record_index, 64)
        codec = CODECS[codec_index]
        step = step_index + 1
        label = f"{codec} step {step}"
        if set(record) != TOP_FIELDS:
            raise ValueError(f"{label} top-level fields differ: {sorted(set(record) ^ TOP_FIELDS)}")
        if (record["codec"], record["seed"], record["step"], record["backend"]) != (codec, 17, step, "metal"):
            raise ValueError(f"{label} identity/order differs")
        if record["plan_build_count"] != 1 or record["plan_execution_count"] != step * 2:
            raise ValueError(f"{label} Metal plan build/execution counts differ")

        expected_vectors = {
            "master_input": [32], "first_moment_input": [32], "second_moment_input": [32],
            "train_packed_bytes": [BYTE_LENGTHS[codec]], "routes": [8], "token_ids": [8],
            "target_ids": [8], "train_logits": [8, 4], "train_token_losses": [8],
            "train_mean_loss": [], "compact_gradients": [8, 16],
            "coalesced_gradients": [2, 16], "updated_masters": [32],
            "updated_first_moment": [32], "updated_second_moment": [32],
            "held_out_packed_bytes": [BYTE_LENGTHS[codec]], "held_out_routes": [8],
            "held_out_token_ids": [8], "held_out_target_ids": [8],
            "held_out_logits": [8, 4], "held_out_token_losses": [8],
            "held_out_mean_loss": [],
        }
        for field, shape in expected_vectors.items():
            require_shape(record[field], shape, f"{label}.{field}")
        for field, expected in (("routes", TRAIN_ROUTES), ("token_ids", TRAIN_TOKENS),
                                ("target_ids", TRAIN_TARGETS), ("held_out_routes", HELD_OUT_ROUTES),
                                ("held_out_token_ids", HELD_OUT_TOKENS),
                                ("held_out_target_ids", HELD_OUT_TARGETS)):
            if record[field] != expected:
                raise ValueError(f"{label}.{field} differs from the fixed split")

        codec_format = CODEC_FORMATS[codec]
        for bytes_field, masters_field in (("train_packed_bytes", "master_input"),
                                           ("held_out_packed_bytes", "updated_masters")):
            expected_bytes = ENCODER.encode_weights(list(flatten(record[masters_field])), codec_format)
            if record[bytes_field] != expected_bytes:
                raise ValueError(f"{label}.{bytes_field} does not encode {masters_field}")

        if step == 1:
            pilot_path = (
                Path(__file__).parents[1]
                / "packed_pilot_moe_training"
                / "results"
                / "packed-tiny-pilot.json"
            )
            pilot = json.loads(pilot_path.read_text())
            format_name = CODEC_FORMATS[codec]
            pilot_arm = next(
                arm for arm in pilot["arms"]
                if arm["seed"] == 17 and arm["format"] == format_name
            )
            initial_masters = pilot_arm["initial_parameters"]
            if record["master_input"] != initial_masters:
                raise ValueError(f"{label} masters differ from seed-17 pilot artifact")
            if record["first_moment_input"] != [0.0] * 32 or record["second_moment_input"] != [0.0] * 32:
                raise ValueError(f"{label} initial Adam moments are not zero")
        else:
            prior = records[record_index - 1]
            for input_field, updated_field in (("master_input", "updated_masters"),
                                               ("first_moment_input", "updated_first_moment"),
                                               ("second_moment_input", "updated_second_moment")):
                if record[input_field] != prior[updated_field]:
                    raise ValueError(f"{label} {input_field} breaks state continuity")

        references = {"cpu_reference": "cpu", "scalar_reference": "scalar"}
        for reference_field, backend in references.items():
            reference = record[reference_field]
            if not isinstance(reference, dict) or set(reference) != NESTED_FIELDS:
                raise ValueError(f"{label}.{reference_field} fields differ")
            if reference["backend"] != backend:
                raise ValueError(f"{label}.{reference_field} backend differs")
            for field, shape in expected_vectors.items():
                if field in NUMERIC_FIELDS:
                    require_shape(reference[field], shape, f"{label}.{reference_field}.{field}")
                    close_payload(record[field], reference[field], f"{label}.{reference_field}.{field}")
    return len(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    arguments = parser.parse_args()
    count = validate(json.loads(arguments.report.read_text()))
    print(f"validated_records={count}")


if __name__ == "__main__":
    main()
