#!/usr/bin/env python3
"""Validate packed tiny-pilot payloads and retain the checked JSON report."""

import argparse
import json
import math
import struct
from pathlib import Path

FORMATS = {"bf8_e5m2", "bf4_e2m1", "fp32"}
SEEDS = {17, 29, 43}
NUMERIC_OUTPUT_FIELDS = (
    "logits",
    "token_losses",
    "train_loss",
    "held_out_logits",
    "held_out_token_losses",
    "held_out_loss",
    "compact_gradient_rows",
    "coalesced_gradient_rows",
    "updated_master_rows",
    "updated_first_moment_rows",
    "updated_second_moment_rows",
)
STEP_FIELDS = {
    "step",
    "backend",
    "master_input",
    "first_moment_input",
    "second_moment_input",
    "train_packed_bytes",
    "train_expert_byte_spans",
    "held_out_packed_bytes",
    "held_out_expert_byte_spans",
    "input_ids",
    "expert_ids",
    "target_ids",
    "logits",
    "token_losses",
    "train_loss",
    "held_out_input_ids",
    "held_out_expert_ids",
    "held_out_target_ids",
    "held_out_logits",
    "held_out_token_losses",
    "held_out_loss",
    "compact_gradient_rows",
    "coalesced_gradient_rows",
    "updated_master_rows",
    "updated_first_moment_rows",
    "updated_second_moment_rows",
    "non_finite_count",
    "wall_time_ns",
    "tokens_per_second",
    "scalar_reference",
}


def round_shift_ties_even(value: int, shift: int) -> int:
    if shift == 0:
        return value
    if shift > 32:
        return 0
    if shift == 32:
        return int(value > (1 << 31))
    quotient = value >> shift
    remainder = value & ((1 << shift) - 1)
    halfway = 1 << (shift - 1)
    return quotient + int(remainder > halfway or (remainder == halfway and quotient & 1 == 1))


def encode_bf8(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    sign = (bits >> 24) & 0x80
    exponent = (bits >> 23) & 0xFF
    fraction = bits & 0x7FFFFF
    if exponent == 0xFF:
        return sign | (0x7B if fraction == 0 else 0x7E)
    if exponent == 0:
        return sign
    unbiased = exponent - 127
    if unbiased > 15:
        return sign | 0x7B
    if unbiased < -14:
        rounded = round_shift_ties_even((1 << 23) | fraction, 7 - unbiased)
        return sign | min(rounded, 4)
    target_fraction = round_shift_ties_even(fraction, 21)
    target_exponent = unbiased + 15
    if target_fraction == 4:
        target_fraction = 0
        target_exponent += 1
    if target_exponent >= 31:
        return sign | 0x7B
    return sign | (target_exponent << 2) | target_fraction


def encode_bf4_nibble(value: float) -> int:
    if not math.isfinite(value):
        raise ValueError("BF4 master value must be finite")
    levels = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    magnitude = min(abs(value), 6.0)
    best_code = 0
    best_distance = math.inf
    for code, level in enumerate(levels):
        distance = abs(magnitude - level)
        if distance < best_distance or (distance == best_distance and code & 1 == 0):
            best_code = code
            best_distance = distance
    return (0x08 if math.copysign(1.0, value) < 0 else 0) | best_code


def encode_weights(values: list[float], format_name: str) -> list[int]:
    if format_name == "bf8_e5m2":
        return [encode_bf8(value) for value in values]
    if format_name == "bf4_e2m1":
        encoded = []
        for index in range(0, len(values), 2):
            low = encode_bf4_nibble(values[index])
            high = encode_bf4_nibble(values[index + 1])
            encoded.append(low | (high << 4))
        return encoded
    return []


def expected_spans(format_name: str) -> list[dict]:
    if format_name == "fp32":
        return []
    bytes_per_expert = {"bf8_e5m2": 16, "bf4_e2m1": 8, "fp32": 0}[format_name]
    return [
        {
            "expert_id": expert_id,
            "offset": expert_id * bytes_per_expert,
            "length": bytes_per_expert,
            "codec": format_name,
            "out_dim": 4,
            "in_dim": 4,
        }
        for expert_id in range(2)
    ]


def leaves(value):
    if isinstance(value, list):
        for item in value:
            yield from leaves(item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)


def close_payload(actual, expected, arm_key: str, step: int, field: str) -> None:
    def compare_values(actual_value, expected_value, path: str) -> None:
        if isinstance(actual_value, list) or isinstance(expected_value, list):
            if not isinstance(actual_value, list) or not isinstance(expected_value, list):
                raise ValueError(f"{arm_key} step {step} {field}{path}: array shape differs")
            if len(actual_value) != len(expected_value):
                raise ValueError(f"{arm_key} step {step} {field}{path}: array length differs")
            for index, (actual_item, expected_item) in enumerate(zip(actual_value, expected_value)):
                compare_values(actual_item, expected_item, f"{path}[{index}]")
            return
        if isinstance(actual_value, (int, float)) and isinstance(expected_value, (int, float)):
            if not math.isfinite(actual_value) or not math.isfinite(expected_value):
                raise ValueError(f"{arm_key} step {step} {field}{path} is non-finite")
            if abs(actual_value - expected_value) > 1e-6:
                raise ValueError(
                    f"{arm_key} step {step} {field}{path} differs: "
                    f"{actual_value} vs {expected_value}"
                )
            return
        raise ValueError(f"{arm_key} step {step} {field}{path}: non-numeric payload")

    compare_values(actual, expected, "")


def validate(report: dict) -> None:
    arms = report.get("arms")
    if not isinstance(arms, list) or len(arms) != 9:
        raise ValueError("report must contain exactly 9 arms")
    expected_pairs = {(format_name, seed) for format_name in FORMATS for seed in SEEDS}
    observed_pairs = {(arm.get("format"), arm.get("seed")) for arm in arms}
    if observed_pairs != expected_pairs:
        raise ValueError(f"arm pairs differ: {observed_pairs!r}")
    if sum(len(arm.get("steps", [])) for arm in arms) != 576:
        raise ValueError("report must contain exactly 576 steps")

    for arm in arms:
        format_name = arm["format"]
        arm_key = f"{format_name}:{arm['seed']}"
        if arm.get("parameters") != 32 or arm.get("hyperparameter_search_trials") != 0:
            raise ValueError(f"{arm_key} has an unexpected parameter or tuning count")
        steps = arm.get("steps")
        if not isinstance(steps, list) or len(steps) != 64:
            raise ValueError(f"{arm_key} must contain exactly 64 steps")
        if len(arm.get("initial_parameters", [])) != 32 or len(arm.get("final_parameters", [])) != 32:
            raise ValueError(f"{arm_key} must retain 32 initial and final masters")
        for expected_step, step_record in enumerate(steps, start=1):
            step = step_record.get("step")
            if step != expected_step:
                raise ValueError(f"{arm_key} step order expected {expected_step}, found {step}")
            missing = STEP_FIELDS - step_record.keys()
            if missing:
                raise ValueError(f"{arm_key} step {step} is missing {sorted(missing)}")
            if len(step_record["master_input"]) != 32:
                raise ValueError(f"{arm_key} step {step} master input length differs")
            if len(step_record["first_moment_input"]) != 32 or len(step_record["second_moment_input"]) != 32:
                raise ValueError(f"{arm_key} step {step} moment input length differs")
            if len(step_record["input_ids"]) != 8 or len(step_record["expert_ids"]) != 8 or len(step_record["target_ids"]) != 8:
                raise ValueError(f"{arm_key} step {step} training token payload lengths differ")
            if len(step_record["held_out_input_ids"]) != 8 or len(step_record["held_out_expert_ids"]) != 8 or len(step_record["held_out_target_ids"]) != 8:
                raise ValueError(f"{arm_key} step {step} held-out token payload lengths differ")

            for bytes_field, spans_field, master_field in (
                ("train_packed_bytes", "train_expert_byte_spans", "master_input"),
                ("held_out_packed_bytes", "held_out_expert_byte_spans", "updated_master_rows"),
            ):
                expected_bytes = encode_weights(list(leaves(step_record[master_field])), format_name)
                if step_record[bytes_field] != expected_bytes:
                    raise ValueError(f"{arm_key} step {step} {bytes_field} does not encode {master_field}")
                if step_record[spans_field] != expected_spans(format_name):
                    raise ValueError(f"{arm_key} step {step} {spans_field} offsets/lengths differ")

            scalar_reference = step_record["scalar_reference"]
            for field in NUMERIC_OUTPUT_FIELDS:
                if field not in scalar_reference:
                    raise ValueError(f"{arm_key} step {step} scalar_reference missing {field}")
                close_payload(step_record[field], scalar_reference[field], arm_key, step, field)
        if arm["final_parameters"] != list(leaves(steps[-1]["updated_master_rows"])):
            raise ValueError(f"{arm_key} final parameters differ from step 64 masters")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    report = json.loads(arguments.input.read_text())
    validate(report)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
