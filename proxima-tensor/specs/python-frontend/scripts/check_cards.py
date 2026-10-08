#!/usr/bin/env python3
"""Check the Python frontend card set's required structure."""

from __future__ import annotations

import argparse
import re as regex
import sys
from pathlib import Path


REQUIRED_FIELDS = (
    "**Owner:**",
    "**Branch:**",
    "**Worktree:**",
    "**Base:**",
    "**Dependency:**",
    "**Commit:**",
    "## Goal",
    "## Changes",
    "## Acceptance criteria",
    "## Complete when",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--check-evidence-plan", action="store_true")
    parser.add_argument("--check-card-order", action="store_true")
    arguments = parser.parse_args()

    card_paths = sorted((arguments.root / "cards").glob("[0-9][0-9]-*.md"))
    cards = [(path, path.read_text(encoding="utf-8")) for path in card_paths]
    errors: list[str] = []
    branches: set[str] = set()
    worktrees: set[str] = set()
    ac_total = 0
    cards_with_ac = 0
    captured_commands = 0
    precommit_checks = 0
    finalizers = 0
    for path, contents in cards:
        for field in REQUIRED_FIELDS:
            if field not in contents:
                errors.append(f"{path}: missing {field}")
        branch = regex.search(r"^\*\*Branch:\*\* `([^`]+)`\s*$", contents, regex.M)
        worktree = regex.search(r"^\*\*Worktree:\*\* `([^`]+)`\s*$", contents, regex.M)
        if branch is None or worktree is None:
            continue
        if branch.group(1) in branches:
            errors.append(f"{path}: duplicate branch")
        branches.add(branch.group(1))
        if worktree.group(1) in worktrees:
            errors.append(f"{path}: duplicate worktree")
        worktrees.add(worktree.group(1))
        acceptance = regex.search(r"## Acceptance criteria\n(.*?)(?:\n## |\Z)", contents, regex.S)
        acceptance_text = acceptance.group(1) if acceptance else ""
        rows = regex.findall(r"^\|\s*AC\d{2}\.\d+\s*\|.*$", acceptance_text, regex.M)
        if not rows:
            errors.append(f"{path}: no numbered command acceptance criteria")
        else:
            cards_with_ac += 1
        ac_total += len(rows)
        for row in rows:
            command = regex.search(r"^\|\s*AC\d{2}\.\d+\s*\|\s*`([^`]+)`", row)
            if command is None:
                errors.append(f"{path}: acceptance criterion has no complete command field")
            elif "check_evidence.py" in command.group(1) and "--finalize" in command.group(1):
                finalizers += 1
            elif "capture_validation.py" in command.group(1):
                captured_commands += 1
                if "--precommit" in command.group(1):
                    precommit_checks += 1
            else:
                errors.append(f"{path}: acceptance command does not capture output or finalize evidence")
        if arguments.check_evidence_plan:
            if "/private/tmp/proxima-python-frontend/evidence/" not in contents:
                errors.append(f"{path}: evidence path missing or not external")
            if "full" not in contents.lower() or "check_evidence.py --finalize" not in contents:
                errors.append(f"{path}: full log/finalizer contract missing")
    if len(cards) != 14:
        errors.append(f"expected 14 cards, found {len(cards)}")
    if finalizers != len(cards):
        errors.append(f"expected one finalizer per card, found {finalizers}")
    if precommit_checks != len(cards):
        errors.append(f"expected one precommit check per card, found {precommit_checks}")
    if captured_commands < len(cards):
        errors.append(f"expected captured acceptance commands for all cards, found {captured_commands}")

    if arguments.check_card_order:
        titles = "\n".join(contents.splitlines()[0] for _, contents in cards)
        required = ("external validation capture", "architecture report", "cassette", "Rust symbolic", "PyO3", "Python operators", "CPU vertical slice", "TOML load", "program, bind, and explain", "Omega device", "model loading", "NumPy ownership", "core Python sugar", "extended Python sugar")
        positions = [titles.lower().find(value.lower()) for value in required]
        if any(position < 0 for position in positions) or positions != sorted(positions):
            errors.append("architecture, Rust seam, PyO3, CPU slice, Omega order is invalid")

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        print(f"cards={len(cards)} acceptance_rows={ac_total} errors={len(errors)}")
        return 1

    if arguments.check_evidence_plan:
        print(f"cards={len(cards)} captured_commands={captured_commands} precommit_checks={precommit_checks} postpush_finalizers={finalizers} external_evidence_paths={len(cards)} repository_evidence_paths=0 missing_fields=0")
    elif arguments.check_card_order:
        print("card_order=14_ordered_cards architecture_before_implementation=1 cpu_slice_before_omega=1 model_load_after_omega=1 core_sugar_before_extended_sugar=1")
    else:
        print(f"cards={len(cards)} unique_worktrees={len(worktrees)} unique_branches={len(branches)} cards_with_acceptance={cards_with_ac} acceptance_rows={ac_total} precommit_checks={precommit_checks} postpush_finalizers={finalizers} missing_fields=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
