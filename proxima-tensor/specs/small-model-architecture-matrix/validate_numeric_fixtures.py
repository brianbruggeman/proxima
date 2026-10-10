#!/usr/bin/env python3
"""Validate exact-ID and f32 comparisons from pinned llama.cpp outputs."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from pathlib import Path


SPEC_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SPEC_DIRECTORY.parents[2]
FIXTURE_PATH = SPEC_DIRECTORY / "fixtures" / "numeric-pair-fixtures.json"
EVIDENCE_PATH = SPEC_DIRECTORY / "fixtures" / "numeric-pair-evidence.json"
IDS_SOURCE_PATH = REPOSITORY_ROOT / "proxima-model-interop" / "tests" / "fixtures" / "gemma4_e2b_llama_greedy_ids.json"
F32_SOURCE_PATH = REPOSITORY_ROOT / "proxima-model-interop" / "tests" / "fixtures" / "llama-parity" / "gemma4_e2b" / "n_probs.json"
F32_RUNTIME_REVISION = "f1ea206218210afb913ae2f5d2c51faed35915da"


def _payload(pair: dict[str, object], field: str) -> bytes:
    path_value = pair.get(f"{field}_payload_path")
    digest_value = pair.get(f"{field}_payload_sha256")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{field} payload path is missing")
    if not isinstance(digest_value, str) or len(digest_value) != 64:
        raise ValueError(f"{field} payload hash is invalid")
    payload_path = (SPEC_DIRECTORY / path_value).resolve()
    if not payload_path.is_relative_to(SPEC_DIRECTORY):
        raise ValueError(f"{field} payload path escapes the spec directory")
    payload = payload_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest_value:
        raise ValueError(f"{field} payload hash does not match its bytes")
    return payload


def _coordinates(flat_index: int, shape: list[int]) -> list[int]:
    coordinate = [0] * len(shape)
    remaining = flat_index
    for dimension_index in range(len(shape) - 1, -1, -1):
        coordinate[dimension_index] = remaining % shape[dimension_index]
        remaining //= shape[dimension_index]
    return coordinate


def compare_pair(pair: dict[str, object]) -> tuple[str | None, dict[str, object]]:
    required_text = (
        "case",
        "boundary",
        "comparator",
        "reference_runtime",
        "reference_revision",
        "reference_source",
    )
    if any(not isinstance(pair.get(field), str) or not pair[field] for field in required_text):
        return "missing_boundary" if not pair.get("boundary") else "missing_metadata", {}
    comparator = pair["comparator"]
    reference_payload = _payload(pair, "reference")
    output_payload = _payload(pair, "output")
    evidence: dict[str, object] = {
        "case": pair["case"],
        "boundary": pair["boundary"],
        "comparator": comparator,
        "reference_runtime": pair["reference_runtime"],
        "reference_revision": pair["reference_revision"],
        "reference_source": pair["reference_source"],
        "reference_payload_path": pair["reference_payload_path"],
        "reference_payload_sha256": pair["reference_payload_sha256"],
        "output_payload_path": pair["output_payload_path"],
        "output_payload_sha256": pair["output_payload_sha256"],
        "status": pair.get("status"),
        "output_origin": pair.get("output_origin"),
    }
    if comparator == "u32_le_exact_v1":
        if len(reference_payload) % 4 or len(output_payload) % 4:
            return "payload_shape_mismatch", evidence
        reference_ids = [value[0] for value in struct.iter_unpack("<I", reference_payload)]
        output_ids = [value[0] for value in struct.iter_unpack("<I", output_payload)]
        evidence["reference_ids"] = reference_ids
        evidence["output_ids"] = output_ids
        evidence["mismatches"] = [
            {"index": index, "reference": reference_id, "output": output_id}
            for index, (reference_id, output_id) in enumerate(zip(reference_ids, output_ids))
            if reference_id != output_id
        ]
        actual_status = "matched" if reference_ids == output_ids else "mismatched"
    elif comparator == "f32_le_allclose_v1":
        shape = pair.get("reference_shape")
        output_shape = pair.get("output_shape")
        absolute_tolerance = pair.get("atol")
        relative_tolerance = pair.get("rtol")
        if (
            not isinstance(shape, list)
            or not shape
            or any(not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0 for dimension in shape)
            or not isinstance(output_shape, list)
            or not output_shape
            or any(not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0 for dimension in output_shape)
            or not isinstance(absolute_tolerance, (int, float))
            or isinstance(absolute_tolerance, bool)
            or not math.isfinite(absolute_tolerance)
            or absolute_tolerance < 0
            or not isinstance(relative_tolerance, (int, float))
            or isinstance(relative_tolerance, bool)
            or not math.isfinite(relative_tolerance)
            or relative_tolerance < 0
        ):
            return "invalid_shape_or_tolerance", evidence
        element_count = math.prod(shape)
        if shape != output_shape or len(reference_payload) != element_count * 4 or len(output_payload) != element_count * 4:
            return "payload_shape_mismatch", evidence
        reference_values = [value[0] for value in struct.iter_unpack("<f", reference_payload)]
        output_values = [value[0] for value in struct.iter_unpack("<f", output_payload)]
        if not all(math.isfinite(value) for value in reference_values + output_values):
            return "non_finite_value", evidence
        mismatches = []
        for index, (reference_value, output_value) in enumerate(zip(reference_values, output_values)):
            if abs(output_value - reference_value) > absolute_tolerance + relative_tolerance * abs(reference_value):
                mismatches.append(
                    {
                        "coordinate": _coordinates(index, shape),
                        "reference": reference_value,
                        "output": output_value,
                    }
                )
        evidence["shape"] = shape
        evidence["output_shape"] = output_shape
        evidence["reference_values"] = reference_values
        evidence["output_values"] = output_values
        evidence["atol"] = absolute_tolerance
        evidence["rtol"] = relative_tolerance
        evidence["mismatches"] = mismatches
        actual_status = "matched" if not mismatches else "mismatched"
    else:
        return "unsupported_comparator", evidence
    if pair.get("status") != actual_status:
        return "comparison_inconsistent", evidence
    return None, evidence


def _verify_upstream_sources(pairs: list[dict[str, object]]) -> None:
    ids_source = json.loads(IDS_SOURCE_PATH.read_text())
    paris_ids = next(record for record in ids_source if record.get("name") == "paris")
    exact_pair = next(pair for pair in pairs if pair.get("comparator") == "u32_le_exact_v1")
    if (
        exact_pair.get("reference_ids") != paris_ids.get("oracle_ids")
        or exact_pair.get("reference_revision") != paris_ids.get("llama_commit")
    ):
        raise ValueError("exact-ID reference no longer matches the retained llama.cpp Paris record")

    f32_source = json.loads(F32_SOURCE_PATH.read_text())[0]
    f32_pair = next(pair for pair in pairs if pair.get("comparator") == "f32_le_allclose_v1")
    expected_values = [item["logprob"] for item in f32_source["steps"][0]["top"][:2]]
    rounded_values = [struct.unpack("<f", struct.pack("<f", value))[0] for value in expected_values]
    if (
        f32_pair.get("reference_values") != rounded_values
        or f32_pair.get("reference_revision") != F32_RUNTIME_REVISION
        or not str(f32_source.get("llama_commit", "")).startswith("f1ea20621")
    ):
        raise ValueError("f32 reference no longer matches the retained llama.cpp top-logprob record")


def run() -> int:
    fixture = json.loads(FIXTURE_PATH.read_text())
    pairs = fixture.get("pairs")
    wrong_output = fixture.get("wrong_output_control")
    if fixture.get("version") != 1 or not isinstance(pairs, list) or len(pairs) != 2 or not isinstance(wrong_output, dict):
        raise ValueError("numeric fixture must contain two positive pairs and one wrong-output control")
    _verify_upstream_sources(pairs)

    exact_result, exact_evidence = compare_pair(pairs[0])
    f32_result, f32_evidence = compare_pair(pairs[1])
    wrong_result, wrong_evidence = compare_pair(wrong_output)
    missing_boundary = dict(pairs[1])
    missing_boundary.pop("boundary", None)
    missing_boundary_result, _ = compare_pair(missing_boundary)
    wrong_output_rejected = wrong_result == "comparison_inconsistent"
    missing_boundary_rejected = missing_boundary_result == "missing_boundary"
    evidence = {
        "version": 1,
        "positive_pairs": [exact_evidence, f32_evidence],
        "wrong_output_rejected": wrong_output_rejected,
        "wrong_output_control": wrong_evidence,
        "missing_boundary_rejected": missing_boundary_rejected,
    }
    EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(
        "positive_pairs=2 "
        f"compared={int(exact_result is None) + int(f32_result is None)} "
        f"wrong_output_rejected={int(wrong_output_rejected)} "
        f"missing_boundary_rejected={int(missing_boundary_rejected)}"
    )
    print(f"retained={EVIDENCE_PATH.relative_to(SPEC_DIRECTORY)} mismatches={wrong_evidence.get('mismatches')}")
    return int(
        exact_result is not None
        or f32_result is not None
        or wrong_output_rejected is False
        or missing_boundary_rejected is False
        or wrong_evidence.get("mismatches") != [{"coordinate": [0], "reference": f32_evidence["reference_values"][0], "output": wrong_evidence["output_values"][0]}]
    )


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except (KeyError, OSError, StopIteration, TypeError, ValueError, struct.error) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
