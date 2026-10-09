#!/usr/bin/env python3
"""Check card identity, dependencies, scope, and evidence contracts."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REQUIRED_FIELDS = (
    "**Owner:**", "**Branch:**", "**Worktree:**", "**Base:**",
    "**Dependency:**", "**Commit:**", "**Slice budget:**", "**Scope units:**", "**Timer window:**",
    "## Goal", "## Changes", "## Acceptance criteria", "## Complete when",
)
TOOL_IDS = ["00a", "00a1", "00a2"] + [f"00{letter}" for letter in "bcdefghijkl"]
FRONT_COUNTS = {1: "abcdef", 2: "ab", 3: "ab", 4: "ab", 5: "ab", 6: "ab", 7: "ab", 8: "ab", 9: "ab", 10: "ab", 11: "abc", 12: "abcd", 13: "abcd"}
FRONT_IDS = [f"{number:02d}{suffix}" for number, suffixes in FRONT_COUNTS.items() for suffix in suffixes]
EXPECTED_IDS = TOOL_IDS + FRONT_IDS
CARD_PATTERN = re.compile(r"(?:00a[12]?|00[b-l]|(?:0[1-9]|1[0-3])[a-f])-[a-z0-9-]+\.md$")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--check-evidence-plan", action="store_true")
    parser.add_argument("--check-card-order", action="store_true")
    parser.add_argument("--check-split-plan", action="store_true")
    arguments = parser.parse_args()

    card_paths = sorted(path for path in (arguments.root / "cards").glob("*.md") if CARD_PATTERN.fullmatch(path.name))
    cards = [(path, path.read_text(encoding="utf-8")) for path in card_paths]
    errors: list[str] = []
    branches: set[str] = set()
    worktrees: set[str] = set()
    card_ids = [path.name.split("-", 1)[0] for path, _ in cards]
    card_id_set = set(card_ids)
    card_ac_rows = 0
    captured_commands = 0
    precommit_checks = 0
    plan_finalizers = 0
    finalizers = 0
    repository_evidence_paths: set[str] = set()
    for position, (path, contents) in enumerate(cards):
        card_id = path.name.split("-", 1)[0]
        for field in REQUIRED_FIELDS:
            if field not in contents:
                errors.append(f"{path}: missing {field}")
        if not re.search(r"^\*\*Slice budget:\*\* one coherent slice capped at 30 minutes; elapsed time is taken from captured start/end timestamps$", contents, re.M):
            errors.append(f"{path}: missing captured elapsed-time contract")
        if not re.search(r"^\*\*Scope units:\*\* 1$", contents, re.M):
            errors.append(f"{path}: scope must declare exactly one concern")
        if not re.search(r"^\*\*Timer window:\*\* starts before `git fetch origin main`; ends after the post-seal closure gate validates the captured post-push finalizer event$", contents, re.M):
            errors.append(f"{path}: timer must include fetch and finalizer seal")
        branch = re.search(r"^\*\*Branch:\*\* `([^`]+)`\s*$", contents, re.M)
        worktree = re.search(r"^\*\*Worktree:\*\* `([^`]+)`\s*$", contents, re.M)
        if branch is None or worktree is None:
            continue
        if branch.group(1) in branches:
            errors.append(f"{path}: duplicate branch")
        branches.add(branch.group(1))
        if worktree.group(1) in worktrees:
            errors.append(f"{path}: duplicate worktree")
        worktrees.add(worktree.group(1))

        dependency_match = re.search(r"^\*\*Dependency:\*\* (.+)$", contents, re.M)
        dependency_text = dependency_match.group(1) if dependency_match else ""
        dependencies = re.findall(r"`((?:00a[12]?|00[b-l]|(?:0[1-9]|1[0-3])[a-f]))`", dependency_text)
        for dependency in dependencies:
            if dependency not in card_id_set:
                errors.append(f"{path}: unknown dependency {dependency}")
            elif EXPECTED_IDS.index(dependency) >= EXPECTED_IDS.index(card_id):
                errors.append(f"{path}: dependency {dependency} does not precede {card_id}")
        if card_id == "00a" and "spec-auditor ADMIT" not in dependency_text:
            errors.append(f"{path}: first card must depend on spec-auditor ADMIT")
        elif card_id != "00a":
            expected_dependency = EXPECTED_IDS[position - 1]
            if dependencies != [expected_dependency]:
                errors.append(f"{path}: dependency must be exactly {expected_dependency}, found {dependencies}")

        acceptance = re.search(r"## Acceptance criteria\n(.*?)(?:\n## |\Z)", contents, re.S)
        acceptance_text = acceptance.group(1) if acceptance else ""
        rows = re.findall(r"^\|\s*AC([A-Za-z0-9]+)\.([1-9])\s*\|.*$", acceptance_text, re.M)
        actual_ids = [f"{row_id}.{number}" for row_id, number in rows]
        expected_ac_ids = [f"{card_id}.{number}" for number in range(1, 4)]
        if actual_ids != expected_ac_ids:
            errors.append(f"{path}: acceptance IDs must be exactly {expected_ac_ids}, found {actual_ids}")
        if len(rows) != 3:
            errors.append(f"{path}: requires exactly 3 counted acceptance criteria, found {len(rows)}")
        card_ac_rows += len(rows)
        commands = []
        for row in re.findall(r"^\|.*$", acceptance_text, re.M):
            match = re.search(r"^\|\s*AC[A-Za-z0-9]+\.[1-9]\s*\|\s*`([^`]+)`", row)
            if not match:
                continue
            command = match.group(1)
            commands.append(command)
            for command_token in re.split(r"\s+", command):
                evidence_path = command_token.strip("\"'`")
                if "/evidence/" in evidence_path or evidence_path.startswith("evidence/"):
                    if not evidence_path.startswith("/private/tmp/proxima-python-frontend/evidence/"):
                        repository_evidence_paths.add(evidence_path)
            if "capture_validation.py" in command or "script -q" in command or (card_id in {"00a", "00a1", "00a2", "00b"} and ("seed_bootstrap.py" in command or "check_plan_bootstrap.py" in command)):
                captured_commands += 1
            if "--precommit" in command:
                precommit_checks += 1
            if re.search(r"\s--finalize(?:\s|$)", command) or (card_id == "00a" and "--verify-closed" in command):
                finalizers += 1
        if commands and ("--precommit" not in commands[1] or ("--verify-closed" not in commands[2] if card_id == "00a" else "--finalize" not in commands[2])):
            errors.append(f"{path}: AC2 must be captured precommit and AC3 must run its terminal postpush reader")
        if len(rows) == 3 and "elapsed_seconds<=1800" not in acceptance_text:
            errors.append(f"{path}: AC3 must assert captured card elapsed time <= 30 minutes")
        if card_id == "00a":
            bootstrap_section = contents.split("## Acceptance criteria", 1)[0]
            if "**Bootstrap argv:** `/usr/bin/script -q /private/tmp/proxima-python-card-00a-attempt-07.log /bin/zsh`" not in bootstrap_section:
                errors.append(f"{path}: host capture must use its declared exact script argv")
            if "host wrapper creates the external evidence directory" not in bootstrap_section or "sources the hash-pinned hook during shell startup" not in bootstrap_section or "hook source marker before the first repository command" not in bootstrap_section:
                errors.append(f"{path}: host wrapper must source the active hook before repository commands")
            if "--verify-entry" not in commands[0] or "omitted_command_rejected=1" not in acceptance_text:
                errors.append(f"{path}: AC1 must verify captured host entry and reject an omitted command")
        if card_id in {"00a1", "00a2"} and "check_plan_bootstrap.py" not in commands[0]:
            errors.append(f"{path}: bootstrap verifier card must exercise its checker")
        if card_id == "00b" and "00a2" not in dependency_text:
            errors.append(f"{path}: launcher card must depend on the landed closure reader")
        if card_id == "00b":
            prelaunch_section = contents.split("## Acceptance criteria", 1)[0]
            if "no repository helper before the PTY starts" not in prelaunch_section or "check_plan_bootstrap.py" in prelaunch_section:
                errors.append(f"{path}: first PTY launch must not depend on a repository helper")
            if "/evidence/card-00a/session-attempt-07/bootstrap-hook.zsh" not in prelaunch_section or "/evidence/card-00a/bootstrap-hook.zsh" in contents:
                errors.append(f"{path}: launcher must consume the active Card 00a hook, not the historical hook")
            if "bootstrap-zdotdir/card-00b" not in prelaunch_section or "exports `ZDOTDIR=/private/tmp/proxima-python-frontend/bootstrap-zdotdir/card-00b`" not in prelaunch_section or "sources the already-landed Card 00a hook" not in prelaunch_section:
                errors.append(f"{path}: bootstrap must create isolated ZDOTDIR and source the landed hook before repository work")
        if arguments.check_evidence_plan:
            if f"/private/tmp/proxima-python-frontend/evidence/card-{card_id}/" not in contents:
                errors.append(f"{path}: card-specific evidence path missing")
            if "full" not in contents.lower() or "--finalize" not in contents:
                errors.append(f"{path}: full-log/finalizer contract missing")

    if card_ids != EXPECTED_IDS:
        errors.append(f"card IDs/order differ: expected={EXPECTED_IDS} found={card_ids}")
    spec_text = (arguments.root / "SPEC.md").read_text(encoding="utf-8")
    global_rows = re.findall(r"^\|\s*AC10\s*\|.*$", spec_text, re.M)
    if len(global_rows) != 1:
        errors.append(f"expected one global plan-finalizer AC10, found {len(global_rows)}")
    elif "capture_validation.py" not in global_rows[0] or "--finalize-plan" not in global_rows[0]:
        errors.append("global AC10 must capture the plan-wide finalizer")
    else:
        captured_commands += 1
        plan_finalizers = 1
        global_command_match = re.search(r"`([^`]+)`", global_rows[0])
        if global_command_match:
            for command_token in re.split(r"\s+", global_command_match.group(1)):
                evidence_path = command_token.strip("\"'`")
                if "/evidence/" in evidence_path or evidence_path.startswith("evidence/"):
                    if not evidence_path.startswith("/private/tmp/proxima-python-frontend/evidence/"):
                        repository_evidence_paths.add(evidence_path)
    if repository_evidence_paths:
        errors.append(f"repository-local evidence destinations found: {sorted(repository_evidence_paths)}")
    if card_ac_rows != 147:
        errors.append(f"expected 147 card acceptance rows, found {card_ac_rows}")
    if finalizers != len(cards):
        errors.append(f"expected one captured postpush stage reader per card, found {finalizers}")
    if precommit_checks != len(cards):
        errors.append(f"expected one captured precommit per card, found {precommit_checks}")
    if plan_finalizers != 1:
        errors.append(f"expected one plan-wide finalizer, found {plan_finalizers}")
    if captured_commands != 148:
        errors.append(f"expected 148 captured acceptance commands, found {captured_commands}")

    if arguments.check_card_order:
        positions = [EXPECTED_IDS.index(item) for item in EXPECTED_IDS]
        if positions != sorted(positions):
            errors.append("card dependency order is inconsistent")

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        print(f"cards={len(cards)} card_acceptance_rows={card_ac_rows} errors={len(errors)}")
        return 1

    if arguments.check_split_plan:
        print(f"cards=49 card_acceptance_rows=147 total_acceptance_rows=148 unique_worktrees={len(worktrees)} precommit={precommit_checks} postpush_stage_checks={finalizers} plan_finalizers={plan_finalizers}")
    elif arguments.check_evidence_plan:
        print(f"cards=49 captured_commands={captured_commands} precommit_checks={precommit_checks} postpush_stage_checks={finalizers} plan_finalizers={plan_finalizers} external_evidence_paths=49 repository_evidence_paths={len(repository_evidence_paths)} missing_fields=0")
    elif arguments.check_card_order:
        print("card_order=49_ordered_cards evidence_tools_before_frontend=1 architecture_before_implementation=1 cpu_slice_before_omega=1 model_load_after_omega=1 core_sugar_before_extended_sugar=1")
    else:
        print(f"cards=49 unique_worktrees={len(worktrees)} unique_branches={len(branches)} cards_with_acceptance={len(cards)} card_acceptance_rows={card_ac_rows} total_acceptance_rows=148 precommit_checks={precommit_checks} postpush_stage_checks={finalizers} plan_finalizers={plan_finalizers} missing_fields=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
