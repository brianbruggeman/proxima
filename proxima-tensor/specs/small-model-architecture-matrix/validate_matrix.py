#!/usr/bin/env python3
"""Validate the pinned model inventory and retained trial evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import struct
import sys
from pathlib import Path


SPEC_DIRECTORY = Path(__file__).resolve().parent

INVENTORY_PATH = SPEC_DIRECTORY / "inventory.jsonl"

EXPECTED_CASES = {
    "lfm_text",
    "lfm_vision",
    "lfm_audio",
    "sori_audio",
    "nanbeige_text",
    "minicpm_text",
    "openbmb_sala_text",
    "nemotron_text",
    "granite4_micro_text",
    "granite4_h_tiny_text",
    "gemma4_e2b",
}

REQUIRED_FIELDS = {
    "case",
    "repo",
    "revision",
    "model_type",
    "parameters",
    "modalities",
    "training_status",
    "license",
    "gated",
    "availability",
    "config_url",
    "card_url",
    "architecture_evidence",
}

EXPECTED_GEMMA_INVENTORY = {
    "repo": "google/gemma-4-E2B",
    "revision": "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f",
    "model_type": "gemma4",
    "parameters": "2.3B effective; 5.1B total including per-layer embeddings",
    "modalities": ["text", "image", "audio", "video"],
    "training_status": "base",
    "license": "apache-2.0",
    "release_date": "2026-03-31",
    "release_url": "https://ai.google.dev/gemma/docs/releases?hl=en",
}

REQUEST_FIXTURE_PATH = SPEC_DIRECTORY / "fixtures" / "request-records.json"

REQUEST_FIELDS = {
    "version",
    "case",
    "checkpoint",
    "input",
    "proxima",
    "result",
    "semantic",
    "numeric",
}

PROXIMA_STATUSES = {"completed", "failed", "not_attempted"}

FAILURE_STAGES = {"access", "load", "bind", "invocation"}

def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )

def _is_revision(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )

def _has_text_fields(record: dict[str, object], fields: set[str]) -> bool:
    return all(isinstance(record.get(field_name), str) and record[field_name] for field_name in fields)

def _payload_bytes(root: Path, relative_path: object, expected_sha256: object) -> bytes | None:
    if not isinstance(relative_path, str) or not relative_path:
        return None
    artifact_path = Path(relative_path)
    if artifact_path.is_absolute() or ".." in artifact_path.parts:
        return None
    try:
        resolved_path = (root / artifact_path).resolve(strict=True)
        resolved_path.relative_to(root.resolve())
        payload = resolved_path.read_bytes()
    except (OSError, ValueError):
        return None
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        return None
    return payload

def _shape_size(shape: list[int]) -> int:
    element_count = 1
    for dimension in shape:
        element_count *= dimension
    return element_count

def validate_request_record(record: object, artifact_root: Path = SPEC_DIRECTORY) -> str | None:
    if not isinstance(record, dict):
        return "record_not_object"
    if "result" not in record:
        return "result_missing"
    if not REQUEST_FIELDS.issubset(record):
        return "missing_field"
    if type(record["version"]) is not int or record["version"] != 1:
        return "wrong_version"
    if not isinstance(record["case"], str) or not record["case"]:
        return "invalid_case"

    checkpoint = record["checkpoint"]
    model_input = record["input"]
    proxima = record["proxima"]
    result = record["result"]
    semantic = record["semantic"]
    numeric = record["numeric"]
    if not all(isinstance(value, dict) for value in (checkpoint, model_input, proxima, semantic, numeric)):
        return "missing_field"
    if (
        not _has_text_fields(checkpoint, {"repo"})
        or not _is_revision(checkpoint.get("revision"))
        or "weights_sha256" not in checkpoint
    ):
        return "missing_field"
    if not _is_sha256(checkpoint.get("config_sha256")):
        return "invalid_hash"
    if checkpoint.get("weights_sha256") is not None and not _is_sha256(checkpoint.get("weights_sha256")):
        return "invalid_hash"

    input_kind = model_input.get("kind")
    if input_kind == "text":
        token_ids = model_input.get("token_ids")
        if (
            not _is_sha256(model_input.get("prompt_utf8_sha256"))
            or not isinstance(token_ids, list)
            or not token_ids
            or any(not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0 for token_id in token_ids)
            or not _is_sha256(model_input.get("token_ids_sha256"))
        ):
            return "invalid_input"
    elif isinstance(input_kind, str) and input_kind in {"image", "audio", "video"}:
        if (
            not _is_sha256(model_input.get("payload_sha256"))
            or not _has_text_fields(model_input, {"format", "decoded_input_descriptor"})
            or not _is_sha256(model_input.get("decoded_input_sha256"))
        ):
            return "invalid_input"
    else:
        return "invalid_input"

    statuses = [proxima.get(f"{stage}_status") for stage in ("load", "bind", "invocation")]
    if not {"load_status", "bind_status", "invocation_status", "invocation"}.issubset(proxima):
        return "missing_field"
    if any(not isinstance(status, str) or status not in PROXIMA_STATUSES for status in statuses):
        return "invalid_proxima_status"
    load_status, bind_status, invocation_status = statuses
    if load_status != "completed" and (bind_status != "not_attempted" or invocation_status != "not_attempted"):
        return "stage_order_invalid"
    if bind_status != "completed" and invocation_status != "not_attempted":
        return "stage_order_invalid"
    invocation = proxima.get("invocation")
    if invocation_status == "not_attempted":
        if invocation is not None:
            return "invocation_evidence_inconsistent"
    else:
        if not isinstance(invocation, dict):
            return "invocation_evidence_missing"
        if set(invocation) != {"call_id", "request_sha256", "output_sha256"}:
            return "invocation_evidence_missing"
        if not _has_text_fields(invocation, {"call_id"}) or not _is_sha256(invocation.get("request_sha256")):
            return "invocation_evidence_missing"
        if invocation_status == "completed":
            if not _is_sha256(invocation.get("output_sha256")):
                return "invocation_evidence_missing"
        elif invocation.get("output_sha256") is not None:
            return "invocation_evidence_inconsistent"

    if not isinstance(result, dict):
        return "result_missing"
    result_status = result.get("status")
    if result_status == "completed":
        generated_ids = result.get("generated_ids")
        if (
            set(result) != {"status", "generated_ids", "output_utf8_sha256"}
            or invocation_status != "completed"
            or any(status != "completed" for status in statuses[:2])
            or not isinstance(generated_ids, list)
            or not generated_ids
            or any(not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0 for token_id in generated_ids)
            or not _is_sha256(result.get("output_utf8_sha256"))
        ):
            return "result_inconsistent"
    elif result_status == "failed":
        if (
            set(result) != {"status", "stage", "code", "message"}
            or not _has_text_fields(result, {"stage", "code", "message"})
            or result["stage"] not in FAILURE_STAGES
        ):
            return "result_missing"
        failure_stage = result["stage"]
        expected_status = {
            "load": proxima["load_status"],
            "bind": proxima["bind_status"],
            "invocation": invocation_status,
        }.get(failure_stage)
        if failure_stage == "access":
            if (
                checkpoint.get("weights_sha256") is not None
                or any(status != "not_attempted" for status in statuses)
                or invocation is not None
            ):
                return "result_inconsistent"
        elif expected_status != "failed":
            return "result_inconsistent"
    else:
        return "result_missing"

    if not _has_text_fields(semantic, {"rubric"}):
        return "missing_field"
    if result_status == "completed":
        if (
            set(semantic) != {"status", "rubric", "evidence_sha256"}
            or not isinstance(semantic.get("status"), str)
            or semantic.get("status") not in {"coherent", "incoherent"}
            or not _is_sha256(semantic.get("evidence_sha256"))
        ):
            return "semantic_missing"
    elif (
        set(semantic) != {"status", "rubric", "reason"}
        or semantic.get("status") != "unavailable"
        or not _has_text_fields(semantic, {"reason"})
    ):
        return "semantic_missing"

    if not _has_text_fields(numeric, {"boundary"}):
        return "missing_field"
    numeric_status = numeric.get("status")
    if isinstance(numeric_status, str) and numeric_status in {"matched", "mismatched"}:
        common_numeric_fields = {
            "status",
            "boundary",
            "comparator",
            "atol",
            "rtol",
            "reference_runtime",
            "reference_revision",
            "reference_payload_path",
            "reference_payload_sha256",
            "output_payload_path",
            "output_payload_sha256",
        }
        comparator = numeric.get("comparator")
        expected_numeric_fields = common_numeric_fields
        if comparator == "f32_le_allclose_v1":
            expected_numeric_fields = common_numeric_fields | {"reference_shape", "output_shape"}
        if (
            set(numeric) != expected_numeric_fields
            or result_status != "completed"
            or not _has_text_fields(
                numeric,
                {
                    "comparator",
                    "reference_runtime",
                    "reference_revision",
                    "reference_payload_path",
                    "output_payload_path",
                },
            )
            or comparator not in {"sha256_exact_v1", "f32_le_allclose_v1"}
            or not _is_sha256(numeric.get("reference_payload_sha256"))
            or not _is_sha256(numeric.get("output_payload_sha256"))
            or any(
                not _is_finite_number(numeric.get(tolerance)) or numeric[tolerance] < 0
                for tolerance in ("atol", "rtol")
            )
        ):
            return "reference_missing"
        if comparator == "sha256_exact_v1":
            reference_payload = _payload_bytes(
                artifact_root,
                numeric["reference_payload_path"],
                numeric["reference_payload_sha256"],
            )
            output_payload = _payload_bytes(
                artifact_root,
                numeric["output_payload_path"],
                numeric["output_payload_sha256"],
            )
            if reference_payload is None or output_payload is None:
                return "numeric_artifact_invalid"
            hashes_match = numeric["reference_payload_sha256"] == numeric["output_payload_sha256"]
            if (
                numeric["atol"] != 0
                or numeric["rtol"] != 0
                or (numeric_status == "matched") != hashes_match
            ):
                return "numeric_comparison_inconsistent"
        elif comparator == "f32_le_allclose_v1":
            reference_shape = numeric["reference_shape"]
            output_shape = numeric["output_shape"]
            if any(
                not isinstance(shape, list)
                or not shape
                or any(not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0 for dimension in shape)
                for shape in (reference_shape, output_shape)
            ):
                return "numeric_shape_invalid"
            reference_payload = _payload_bytes(
                artifact_root,
                numeric["reference_payload_path"],
                numeric["reference_payload_sha256"],
            )
            output_payload = _payload_bytes(
                artifact_root,
                numeric["output_payload_path"],
                numeric["output_payload_sha256"],
            )
            if reference_payload is None or output_payload is None:
                return "numeric_artifact_invalid"
            if (
                len(reference_payload) != _shape_size(reference_shape) * 4
                or len(output_payload) != _shape_size(output_shape) * 4
            ):
                return "numeric_payload_shape_mismatch"
            shapes_match = reference_shape == output_shape
            reference_values = [value[0] for value in struct.iter_unpack("<f", reference_payload)]
            output_values = [value[0] for value in struct.iter_unpack("<f", output_payload)]
            if not all(math.isfinite(value) for value in reference_values + output_values):
                return "numeric_value_nonfinite"
            values_match = shapes_match and all(
                abs(output_value - reference_value)
                <= numeric["atol"] + numeric["rtol"] * abs(reference_value)
                for reference_value, output_value in zip(reference_values, output_values)
            )
            if (numeric_status == "matched") != values_match:
                return "numeric_comparison_inconsistent"
    elif (
        set(numeric) != {"status", "boundary", "reason"}
        or numeric_status != "unavailable"
        or not _has_text_fields(numeric, {"reason"})
    ):
        return "numeric_missing"

    access_only = checkpoint.get("weights_sha256") is None
    if access_only and not (
        all(status == "not_attempted" for status in statuses)
        and invocation is None
        and result_status == "failed"
        and result.get("stage") == "access"
        and semantic.get("status") == "unavailable"
        and numeric_status == "unavailable"
    ):
        return "access_record_inconsistent"
    return None

def _request_fixture_parts(fixture: object) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if (
        not isinstance(fixture, dict)
        or set(fixture) != {"valid", "controls"}
        or not isinstance(fixture["valid"], list)
        or not isinstance(fixture["controls"], list)
        or any(not isinstance(record, dict) for record in fixture["valid"])
        or any(not isinstance(control, dict) for control in fixture["controls"])
        or any(not isinstance(record.get("case"), str) or not record["case"] for record in fixture["valid"])
        or len({record["case"] for record in fixture["valid"]}) != len(fixture["valid"])
    ):
        raise ValueError("request fixture must contain valid and controls arrays of objects")
    positive_records = fixture["valid"]
    controls = fixture["controls"]
    records_by_case = {record["case"]: record for record in positive_records}
    control_names: set[str] = set()
    for control in controls:
        if (
            set(control) != {"name", "base", "mutations"}
            or not isinstance(control.get("base"), str)
            or control["base"] not in records_by_case
            or not isinstance(control.get("name"), str)
            or not control.get("name")
            or control["name"] in control_names
            or not isinstance(control.get("mutations"), list)
        ):
            raise ValueError("request fixture control has an invalid shape or base")
        control_names.add(control["name"])
    return positive_records, controls

def validate_request_fixtures() -> int:
    fixture = json.loads(REQUEST_FIXTURE_PATH.read_text())
    positive_records, controls = _request_fixture_parts(fixture)
    accepted_results = [validate_request_record(record) for record in positive_records]
    records_by_case = {record["case"]: record for record in positive_records}
    control_results: dict[str, str | None] = {}
    for control in controls:
        control_record = copy.deepcopy(records_by_case[control["base"]])
        for mutation in control["mutations"]:
            if (
                not isinstance(mutation, dict)
                or not isinstance(mutation.get("path"), list)
                or not mutation["path"]
                or any(not isinstance(key, str) for key in mutation["path"])
                or ("remove" in mutation and mutation["remove"] is not True)
                or ("remove" not in mutation and "value" not in mutation)
            ):
                raise ValueError("request fixture mutation has an invalid path")
            target = control_record
            for key in mutation["path"][:-1]:
                if not isinstance(target, dict) or key not in target:
                    raise ValueError("request fixture mutation path does not exist")
                target = target[key]
            if not isinstance(target, dict):
                raise ValueError("request fixture mutation target is not an object")
            leaf = mutation["path"][-1]
            if mutation.get("remove", False):
                if leaf not in target:
                    raise ValueError("request fixture mutation removes a missing field")
                del target[leaf]
            else:
                target[leaf] = mutation["value"]
        control_results[control["name"]] = validate_request_record(control_record)
    expected_controls = {
        "invocation_evidence_missing": "invocation_evidence_missing",
        "result_missing": "result_missing",
        "reference_missing": "reference_missing",
        "invocation_key_missing": "missing_field",
        "weight_hash_key_missing": "missing_field",
        "failed_output_hash_key_missing": "invocation_evidence_missing",
        "completed_result_has_failure_fields": "result_inconsistent",
        "float_version": "wrong_version",
        "invalid_stage_order": "stage_order_invalid",
        "exact_hash_status_conflict": "numeric_comparison_inconsistent",
        "f32_value_status_conflict": "numeric_comparison_inconsistent",
        "f32_payload_length_conflict": "numeric_payload_shape_mismatch",
        "numeric_artifact_missing": "numeric_artifact_invalid",
        "unsupported_modality": "invalid_input",
    }
    accepted = sum(result is None for result in accepted_results)
    rejected_controls = sum(control_results.get(name) == code for name, code in expected_controls.items())
    missing_fields = sum(result in {"missing_field", "result_missing"} for result in accepted_results)
    rejection_counts = {
        error_code: sum(result == error_code for result in control_results.values())
        for error_code in set(expected_controls.values())
    }
    fixture_shape_rejections = 0
    malformed_fixtures = [None, {}, copy.deepcopy(fixture)]
    malformed_fixtures[-1]["controls"][0]["base"] = []
    for malformed_fixture in malformed_fixtures:
        try:
            _request_fixture_parts(malformed_fixture)
        except ValueError:
            fixture_shape_rejections += 1
    print(
        f"accepted={accepted} rejected_controls={rejected_controls} missing_fields={missing_fields} "
        f"error_classes={len(rejection_counts)} exact_control_codes={sum(result == expected_controls[name] for name, result in control_results.items())} "
        f"fixture_shape_rejections={fixture_shape_rejections}"
    )
    return int(
        accepted != 4
        or len(positive_records) != 4
        or len(controls) != 14
        or rejected_controls != 14
        or fixture_shape_rejections != 3
        or missing_fields != 0
        or set(control_results) != set(expected_controls)
        or any(result != expected_controls[name] for name, result in control_results.items())
    )

def read_inventory() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(INVENTORY_PATH.read_text().splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"inventory line {line_number} is invalid JSON: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"inventory line {line_number} is not an object")
        records.append(record)
    return records

def validate_inventory() -> int:
    records = read_inventory()
    case_names = [str(record.get("case", "")) for record in records]
    missing_fields = sum(
        field_name not in record or record[field_name] is None or record[field_name] == ""
        for record in records
        for field_name in REQUIRED_FIELDS
    )
    revision_mismatches = sum(
        not record.get("revision")
        or not str(record.get("config_url", "")).count(str(record.get("revision")))
        or not str(record.get("card_url", "")).count(str(record.get("revision")))
        for record in records
    )
    duplicate_cases = len(case_names) - len(set(case_names))
    missing_cases = len(EXPECTED_CASES.difference(case_names))
    unexpected_cases = len(set(case_names).difference(EXPECTED_CASES))
    empty_modalities = sum(not record.get("modalities") for record in records)
    gemma_records = [record for record in records if record.get("case") == "gemma4_e2b"]
    gemma_fact_errors = int(
        len(gemma_records) != 1
        or any(gemma_records[0].get(field_name) != expected for field_name, expected in EXPECTED_GEMMA_INVENTORY.items())
    )
    case_errors = duplicate_cases + missing_cases + unexpected_cases + empty_modalities + gemma_fact_errors
    print(
        f"candidates={len(records)} missing_fields={missing_fields} "
        f"revision_mismatches={revision_mismatches} case_errors={case_errors} gemma_fact_errors={gemma_fact_errors}"
    )
    return int(
        len(records) != len(EXPECTED_CASES)
        or missing_fields != 0
        or revision_mismatches != 0
        or case_errors != 0
    )

def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--schema-fixtures", action="store_true")
    arguments = parser.parse_args()
    if sum((arguments.inventory, arguments.schema_fixtures)) != 1:
        parser.error("select one implemented validation mode")
    try:
        if arguments.schema_fixtures:
            return validate_request_fixtures()
        return validate_inventory()
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
