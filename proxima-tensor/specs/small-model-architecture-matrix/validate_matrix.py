import argparse

import json

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

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", action="store_true")
    arguments = parser.parse_args()
    if not arguments.inventory:
        parser.error("select --inventory")
    try:
        return validate_inventory()
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
