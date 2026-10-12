#!/usr/bin/env python3
"""Inspect captured command artifacts and enforce card gates."""
from __future__ import annotations
import argparse, base64, hashlib, json, shutil, subprocess, sys, os, re, shlex
from pathlib import Path
from typing import Any

FORBIDDEN_DIRS = {".venv", ".pytest_cache", "__pycache__", "target", "build", "dist", "tests/cassettes", "cassettes", "evidence"}

class EvidenceError(Exception):
    pass

def fail(message: str) -> None:
    raise EvidenceError(message)

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_events(root: Path) -> list[dict[str, Any]]:
    manifest = root / "manifest.jsonl"
    if not manifest.is_file():
        fail(f"missing manifest: {manifest}")
    events = []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            fail(f"empty manifest record at line {line_number}")
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as error:
            fail(f"invalid manifest record at line {line_number}: {error}")
    return events

def verify_bundle(root: Path, allow_inflight_argv: list[str] | None = None) -> list[dict[str, Any]]:
    root = root.resolve()
    events = load_events(root)
    if not events:
        fail("manifest contains no command records")
    if not (root / "starts.jsonl").is_file(): fail("missing starts.jsonl")
    if not (root / "receipt.json").is_file(): fail("missing final count receipt")
    full_log = root / "full.log"
    if not full_log.is_file():
        fail("missing full.log")
    log_records = []
    for line_number, line in enumerate(full_log.read_text(encoding="utf-8").splitlines(), 1):
        try:
            log_records.append(json.loads(line))
        except json.JSONDecodeError as error:
            fail(f"invalid full.log record at line {line_number}: {error}")
    if len(log_records) != len(events):
        fail(f"full.log records={len(log_records)} manifest records={len(events)}")
    for position, event in enumerate(events, 1):
        if event.get("event_id") != position:
            fail(f"nonsequential event id at position {position}")
        if not event.get("argv") or not event.get("cwd") or not event.get("started_at") or not event.get("ended_at"):
            fail(f"incomplete command metadata at event {position}")
        if event.get("started_unix_ns", 0) > event.get("ended_unix_ns", -1):
            fail(f"invalid timestamp order at event {position}")
        log_record = log_records[position - 1]
        if log_record.get("event") != event:
            fail(f"full.log event differs from manifest at event {position}")
        for stream_name in ("stdout", "stderr", "exit"):
            stream = event.get("streams", {}).get(stream_name)
            if not stream:
                fail(f"missing {stream_name} record at event {position}")
            artifact = (root / stream["path"]).resolve()
            if root not in artifact.parents or not artifact.is_file():
                fail(f"missing or external {stream_name} artifact at event {position}")
            if sha_file(artifact) != stream["sha256"]:
                fail(f"{stream_name} hash mismatch at event {position}")
            if stream_name in ("stdout", "stderr") and artifact.stat().st_size != stream["bytes"]:
                fail(f"{stream_name} truncated at event {position}")
        exit_record = json.loads((root / event["streams"]["exit"]["path"]).read_text(encoding="utf-8"))
        terminal_fields = ("event_id","exit_code","outcome","signal_number","started_at","ended_at","ended_unix_ns")
        if any(exit_record.get(field) != event.get(field) for field in terminal_fields):
            fail(f"exit terminal record differs from manifest at event {position}")
        stdout = base64.b64decode(log_record.get("stdout_base64", ""), validate=True)
        stderr = base64.b64decode(log_record.get("stderr_base64", ""), validate=True)
        if sha(stdout) != event["streams"]["stdout"]["sha256"] or sha(stderr) != event["streams"]["stderr"]["sha256"]:
            fail(f"full.log stream mismatch at event {position}")
        if "outcome" in event and event.get("outcome") not in ("completed", "spawn_error", "interrupted", "signaled"):
            fail(f"unsealed outcome at event {position}")
        if "outcome" in event and exit_record.get("outcome") != event.get("outcome"):
            fail(f"terminal outcome mismatch at event {position}")
        for item in event.get("immutable_artifacts", {}).values():
            artifact = (root / item["path"]).resolve()
            if root not in artifact.parents or not artifact.is_file() or sha_file(artifact) != item["sha256"]:
                fail(f"immutable event artifact mismatch at event {position}")
    command_dirs = sorted((root / "commands").glob("[0-9][0-9][0-9][0-9][0-9][0-9]"))
    sealed_dirs = [directory for directory in command_dirs if (directory / "exit.json").is_file()]
    unsealed_dirs = [directory for directory in command_dirs if not (directory / "exit.json").is_file()]
    if len(sealed_dirs) != len(events) or len(unsealed_dirs) > 1:
        fail(f"sealed command directories={len(sealed_dirs)} manifest events={len(events)} unsealed={len(unsealed_dirs)}")
    if unsealed_dirs:
        start_path = unsealed_dirs[0] / "start.json"
        if not start_path.is_file() or int(unsealed_dirs[0].name) != len(events) + 1:
            fail("unexpected unsealed in-flight command directory")
        pending = json.loads(start_path.read_text(encoding="utf-8"))
        if allow_inflight_argv is None or pending.get("argv") != allow_inflight_argv:
            fail("unsealed start does not match the active checker command")
    starts_path = root / "starts.jsonl"
    if not starts_path.is_file(): fail("missing starts.jsonl")
    start_lines = starts_path.read_text(encoding="utf-8").splitlines()
    expected_start_count = len(events) + len(unsealed_dirs)
    if len(start_lines) != expected_start_count:
        fail("start records do not match manifest events")
    start_fields = ("event_id","argv","capture_argv","cwd","started_at","started_unix_ns","environment_overrides")
    for position, event in enumerate(events, 1):
        command_start = root / "commands" / f"{position:06d}" / "start.json"
        if not command_start.is_file(): fail(f"missing command start payload at event {position}")
        start_payload = json.loads(command_start.read_text(encoding="utf-8"))
        indexed_start = json.loads(start_lines[position - 1])
        expected_start = {field: event.get(field) for field in start_fields}
        if start_payload != indexed_start or any(start_payload.get(field) != expected_start[field] for field in start_fields):
            fail(f"command start provenance mismatch at event {position}")
    if unsealed_dirs:
        pending_start = json.loads((unsealed_dirs[0] / "start.json").read_text(encoding="utf-8"))
        if pending_start != json.loads(start_lines[-1]):
            fail("pending command start differs from starts.jsonl")
    receipt_path = root / "receipt.json"
    if not receipt_path.is_file(): fail("missing final count receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required_receipt = {"event_count","command_directory_count","manifest_sha256","full_log_sha256","starts_sha256","artifacts_sha256","comparisons_sha256","last_event_id","last_event_sha256"}
    if not required_receipt.issubset(receipt): fail("incomplete final count receipt")
    if receipt["event_count"] != len(events) or receipt["command_directory_count"] not in (len(command_dirs), len(sealed_dirs)):
        fail("receipt count mismatch")
    for name, key in (("manifest.jsonl","manifest_sha256"),("full.log","full_log_sha256"),("artifacts.jsonl","artifacts_sha256"),("comparisons.jsonl","comparisons_sha256")):
        if sha_file(root / name) != receipt[key]: fail(f"receipt hash mismatch for {name}")
    starts_bytes = ("\n".join(start_lines[:len(events)]) + ("\n" if events else "")).encode()
    if sha(starts_bytes) != receipt["starts_sha256"]: fail("receipt hash mismatch for starts.jsonl")
    if receipt["last_event_id"] != events[-1]["event_id"] or receipt["last_event_sha256"] != sha(json.dumps(events[-1],sort_keys=True).encode()):
        fail("receipt last event record mismatch")
    artifact_path = root / "artifacts.jsonl"
    if not artifact_path.is_file():
        fail("missing typed artifacts.jsonl")
    artifact_rows = []
    artifact_ids = set()
    event_ids = {event["event_id"] for event in events}
    for line_number, line in enumerate(artifact_path.read_text(encoding="utf-8").splitlines(), 1):
        try: row = json.loads(line)
        except json.JSONDecodeError as error: fail(f"invalid artifact row {line_number}: {error}")
        required = {"id", "kind", "relative_path", "sha256", "producer_event_id"}
        if set(row) != required or not row["id"] or row["kind"] not in {"event_output", "actual", "payload", "cassette", "literal", "incumbent", "citation"}:
            fail(f"invalid typed artifact schema at row {line_number}")
        if row["id"] in artifact_ids or row["producer_event_id"] not in event_ids: fail(f"duplicate artifact id or missing producer at row {line_number}")
        artifact_ids.add(row["id"])
        path = (root / row["relative_path"]).resolve()
        if root not in path.parents or not path.is_file() or sha_file(path) != row["sha256"]: fail(f"artifact row does not match opened file at row {line_number}")
        artifact_rows.append(row)
    required_outputs = {(event["event_id"], name) for event in events for name in ("stdout", "stderr", "exit")}
    observed_outputs = {(row["producer_event_id"], row["kind"], row["relative_path"], row["sha256"]) for row in artifact_rows if row["kind"] == "event_output"}
    for event in events:
        for name in ("stdout", "stderr", "exit"):
            stream = event["streams"][name]
            if (event["event_id"], "event_output", stream["path"], stream["sha256"]) not in observed_outputs:
                fail(f"missing typed {name} artifact row for event {event['event_id']}")
    expected_immutable = {(event["event_id"], item["path"], item["sha256"]) for event in events for item in event.get("immutable_artifacts", {}).values()}
    observed_immutable = {(row["producer_event_id"], row["relative_path"], row["sha256"]) for row in artifact_rows if row["kind"] != "event_output"}
    if expected_immutable != observed_immutable: fail("typed immutable artifact rows do not match event snapshots")
    comparisons_path = root / "comparisons.jsonl"
    if not comparisons_path.is_file(): fail("missing comparisons.jsonl")
    artifact_by_id = {row["id"]: row for row in artifact_rows}
    for line_number, line in enumerate(comparisons_path.read_text(encoding="utf-8").splitlines(), 1):
        try: comparison = json.loads(line)
        except json.JSONDecodeError as error: fail(f"invalid comparison row {line_number}: {error}")
        if set(comparison) != {"id", "left_artifact_id", "right_artifact_id", "comparator", "producer_event_id"}:
            fail(f"invalid comparison schema at row {line_number}")
        if comparison["left_artifact_id"] not in artifact_ids or comparison["right_artifact_id"] not in artifact_ids or comparison["comparator"] not in {"sha256_equal", "exact_bytes", "exact_text"} or comparison["producer_event_id"] not in event_ids:
            fail(f"comparison references missing artifact or unsupported comparator at row {line_number}")
        left_path = root / artifact_by_id[comparison["left_artifact_id"]]["relative_path"]
        right_path = root / artifact_by_id[comparison["right_artifact_id"]]["relative_path"]
        left_bytes, right_bytes = left_path.read_bytes(), right_path.read_bytes()
        comparator = comparison["comparator"]
        equal = left_bytes == right_bytes if comparator in {"sha256_equal", "exact_bytes"} else left_bytes.decode("utf-8") == right_bytes.decode("utf-8") if comparator == "exact_text" else False
        if not equal: fail(f"comparison outcome failed at row {line_number}: {comparator}")
    return events

def repo_artifacts_from_status(status: str) -> list[str]:
    forbidden = {".venv", ".pytest_cache", "__pycache__", "target", "build", "dist", "cassettes", "evidence"}
    found = []
    records = status.splitlines() if isinstance(status, str) else status
    for record in records:
        relative = record[3:].strip().removeprefix("./")
        parts = Path(relative).parts
        if any(part in forbidden or part.endswith(".egg-info") for part in parts) or relative.endswith(".cassette") or Path(relative).name == ".capture.lock":
            found.append(relative)
    return sorted(set(found))

def repo_artifacts_from_events(events: list[dict[str, Any]], require_tree: bool = False, allow_terminal_finalizer: bool = False) -> list[str]:
    status_argv = ["git", "status", "--short", "--ignored", "--untracked-files=all"]
    diff_argv = ["git", "diff", "--name-only", "origin/main...HEAD"]
    tree_argv = ["git", "ls-tree", "-r", "--name-only", "HEAD"]
    commit_diff_argv = ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]
    def captured_output(event: dict[str, Any]) -> str:
        return (Path(event["_evidence_root"]) / event["streams"]["stdout"]["path"]).read_text(encoding="utf-8")
    status_events = [event for event in events if event.get("argv") == status_argv and event.get("exit_code") == 0]
    if not status_events: fail("missing captured generated-artifact status inventory")
    found = set(repo_artifacts_from_status(captured_output(status_events[-1])))
    diff_events = [event for event in events if event.get("argv") == diff_argv and event.get("exit_code") == 0]
    tree_events = [event for event in events if event.get("argv") == tree_argv and event.get("exit_code") == 0]
    commit_diff_events = [event for event in events if event.get("argv") == commit_diff_argv and event.get("exit_code") == 0]
    if require_tree and (not diff_events or not tree_events or not commit_diff_events): fail("missing captured branch-diff, committed-tree, or commit-diff inventory")
    if require_tree:
        inventory_tail = [status_events[-1], diff_events[-1], tree_events[-1], commit_diff_events[-1]]
        terminal = events[-1] if events else {}
        terminal_finalizer = allow_terminal_finalizer and terminal.get("exit_code") == 0 and any(Path(item).name == "check_evidence.py" for item in terminal.get("argv", [])) and "--finalize" in terminal.get("argv", [])
        inventory_matches = len(events) >= 4 and events[-4:] == inventory_tail
        finalizer_matches = len(events) >= 5 and events[-5:-1] == inventory_tail and terminal_finalizer
        if not inventory_matches and not finalizer_matches:
            fail("generated-artifact inventory is stale; only the exact sealed finalizer may follow the final status/diff/tree inventory")
        changed = set(captured_output(diff_events[-1]).splitlines()) | set(captured_output(commit_diff_events[-1]).splitlines())
        tree = set(captured_output(tree_events[-1]).splitlines())
        found.update(repo_artifacts_from_status("\n".join("?? " + line for line in sorted(tree & changed))))
    return sorted(found)

def self_check(destination: Path, repo: Path) -> str:
    sentinel = destination.with_name(destination.name + "-sentinel.json")
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    if not sentinel.exists(): sentinel.write_text("prior-attempt\n", encoding="utf-8")
    sentinel_hash = sha_file(sentinel)
    previous = set(destination.parent.glob(destination.name + "-attempt-*"))
    attempt = 1
    while destination.with_name(destination.name + f"-attempt-{attempt:04d}").exists(): attempt += 1
    destination = destination.with_name(destination.name + f"-attempt-{attempt:04d}")
    destination.mkdir(parents=True)
    from capture_validation import capture_one
    capture_one(destination, [sys.executable, "-c", "import sys;sys.stdout.buffer.write(b'out-one\\n');sys.stderr.buffer.write(b'err-one\\n')"])
    capture_one(destination, [sys.executable, "-c", "import sys;sys.stdout.write('out-two\\n');sys.stderr.write('err-two\\n');sys.exit(7)"])
    events = verify_bundle(destination)
    if len(events) != 2 or [item["exit_code"] for item in events] != [0, 7]: fail("self-check did not retain both success and failure exit records")
    missing_evidence_rejected = True
    for filename in ("receipt.json","starts.jsonl"):
        incomplete = destination.with_name(destination.name + "-missing-" + filename.replace(".","-"))
        shutil.copytree(destination, incomplete)
        (incomplete / filename).unlink()
        try: verify_bundle(incomplete)
        except EvidenceError: pass
        else: missing_evidence_rejected = False
    if not missing_evidence_rejected: fail("missing receipt or start stream was accepted")
    receipt_copy = destination.with_name(destination.name + "-receipt-hash-control")
    shutil.copytree(destination, receipt_copy)
    receipt_record = json.loads((receipt_copy / "receipt.json").read_text(encoding="utf-8"))
    receipt_record["manifest_sha256"] = "0" * 64
    (receipt_copy / "receipt.json").write_text(json.dumps(receipt_record,sort_keys=True)+"\n",encoding="utf-8")
    receipt_hash_rejected = False
    try: verify_bundle(receipt_copy)
    except EvidenceError as error: receipt_hash_rejected = "receipt hash mismatch" in str(error)
    if not receipt_hash_rejected: fail("edited receipt hash was accepted")
    start_copy = destination.with_name(destination.name + "-start-mismatch-control")
    shutil.copytree(destination, start_copy)
    start_record = json.loads((start_copy / "commands/000001/start.json").read_text(encoding="utf-8"))
    start_record["argv"] = ["tampered"]
    (start_copy / "commands/000001/start.json").write_text(json.dumps(start_record,sort_keys=True)+"\n",encoding="utf-8")
    start_mismatch_rejected = False
    try: verify_bundle(start_copy)
    except EvidenceError as error: start_mismatch_rejected = "start provenance mismatch" in str(error)
    if not start_mismatch_rejected: fail("mismatched command start payload was accepted")
    corrupted = destination.with_name(destination.name + "-corrupted")
    if corrupted.exists(): corrupted = corrupted.with_name(corrupted.name + f"-attempt-{attempt:04d}")
    shutil.copytree(destination, corrupted)
    with (corrupted / events[0]["streams"]["stdout"]["path"]).open("ab") as output: output.write(b"corruption")
    corrupted_rejected = False
    try: verify_bundle(corrupted)
    except EvidenceError: corrupted_rejected = True
    if not corrupted_rejected: fail("corrupted artifact was not rejected")
    comparison_copy = destination.with_name(destination.name + "-comparison-mismatch")
    shutil.copytree(destination, comparison_copy)
    comparison_events = load_events(comparison_copy)
    comparison = {"id":"negative-control","left_artifact_id":"event-000001-stdout","right_artifact_id":"event-000002-stdout","comparator":"exact_bytes","producer_event_id":2}
    (comparison_copy / "comparisons.jsonl").write_text(json.dumps(comparison) + "\n", encoding="utf-8")
    comparison_receipt = json.loads((comparison_copy / "receipt.json").read_text(encoding="utf-8"))
    comparison_receipt["comparisons_sha256"] = sha_file(comparison_copy / "comparisons.jsonl")
    (comparison_copy / "receipt.json").write_text(json.dumps(comparison_receipt,sort_keys=True) + "\n", encoding="utf-8")
    comparison_rejected = False
    try: verify_bundle(comparison_copy)
    except EvidenceError as error: comparison_rejected = "comparison outcome failed" in str(error)
    if not comparison_rejected: fail("mismatched exact-bytes comparison was accepted")
    tail_copy = destination.with_name(destination.name + "-tail-truncated")
    if tail_copy.exists(): fail(f"retained tail control already exists: {tail_copy}")
    shutil.copytree(destination, tail_copy)
    manifest_path = tail_copy / "manifest.jsonl"
    full_path = tail_copy / "full.log"
    manifest_lines = manifest_path.read_text(encoding="utf-8").splitlines()
    full_lines = full_path.read_text(encoding="utf-8").splitlines()
    first_event = json.loads(manifest_lines[0])
    manifest_path.write_text(manifest_lines[0] + "\n",encoding="utf-8")
    full_path.write_text(full_lines[0] + "\n",encoding="utf-8")
    start_path = tail_copy / "starts.jsonl"
    start_lines = start_path.read_text(encoding="utf-8").splitlines()
    start_path.write_text(start_lines[0] + "\n",encoding="utf-8")
    artifact_rows = [json.loads(line) for line in (tail_copy / "artifacts.jsonl").read_text(encoding="utf-8").splitlines()]
    artifact_rows = [row for row in artifact_rows if row["producer_event_id"] == 1]
    (tail_copy / "artifacts.jsonl").write_text("".join(json.dumps(row,sort_keys=True)+"\n" for row in artifact_rows),encoding="utf-8")
    tail_receipt = json.loads((tail_copy / "receipt.json").read_text(encoding="utf-8"))
    tail_receipt.update({"event_count":1,"command_directory_count":2,"manifest_sha256":sha_file(manifest_path),"full_log_sha256":sha_file(full_path),"starts_sha256":sha_file(start_path),"artifacts_sha256":sha_file(tail_copy / "artifacts.jsonl"),"last_event_id":1,"last_event_sha256":sha(json.dumps(first_event,sort_keys=True).encode())})
    (tail_copy / "receipt.json").write_text(json.dumps(tail_receipt,sort_keys=True)+"\n",encoding="utf-8")
    tail_rejected = False
    try: verify_bundle(tail_copy)
    except EvidenceError as error: tail_rejected = "sealed command directories=2 manifest events=1" in str(error)
    if not tail_rejected: fail("orphan sealed command directory was accepted")
    root_rejected = False
    try: capture_one(repo.resolve(), [sys.executable, "-c", "pass"])
    except SystemExit: root_rejected = True
    if not root_rejected: fail("repository-local root was accepted")
    old_cassette = os.environ.get("PROXIMA_CASSETTE_DIR"); cassette_rejected = False
    try:
        os.environ["PROXIMA_CASSETTE_DIR"] = str(repo.resolve() / "tests" / "cassettes")
        capture_one(destination / "cassette-control", [sys.executable, "-c", "pass"])
    except SystemExit: cassette_rejected = True
    finally:
        if old_cassette is None: os.environ.pop("PROXIMA_CASSETTE_DIR", None)
        else: os.environ["PROXIMA_CASSETTE_DIR"] = old_cassette
    if not cassette_rejected: fail("inherited repository cassette root was accepted")
    symlink_root = destination / "cassette-symlink-control"
    symlink_root.mkdir()
    (symlink_root / "cassettes").symlink_to(repo.resolve() / "tests" / "cassettes", target_is_directory=True)
    symlink_rejected = False
    try: capture_one(symlink_root, [sys.executable, "-c", "pass"])
    except SystemExit: symlink_rejected = True
    if not symlink_rejected: fail("repository-local cassette symlink was accepted")
    external_symlink_root = destination / "external-cassette-symlink-control"
    external_target = destination / "external-cassette-target"
    external_target.mkdir()
    external_symlink_root.mkdir()
    (external_symlink_root / "cassettes").symlink_to(external_target, target_is_directory=True)
    external_symlink_rejected = False
    try: capture_one(external_symlink_root, [sys.executable, "-c", "pass"])
    except SystemExit: external_symlink_rejected = True
    if not external_symlink_rejected: fail("external cassette symlink was accepted")
    spawn_root = destination / "spawn-control"; capture_one(spawn_root, ["missing-card00-executable"]); spawn_events = verify_bundle(spawn_root)
    spawn_sealed = len(spawn_events) == 1 and spawn_events[0].get("outcome") == "spawn_error" and spawn_events[0].get("exit_code") == 127
    if not spawn_sealed: fail("spawn failure did not produce a sealed event")
    fake_card = destination / "card-01"
    fake_card.mkdir()
    (fake_card / "finalize-result.json").write_text(json.dumps({"card_id":"01","findings":17,"success":True}) + "\n")
    future_rejected = False
    try: run_finalize(fake_card, repo)
    except EvidenceError: future_rejected = True
    if not future_rejected: fail("unregistered Card01 with forged findings was accepted")
    fake_registered_parent = destination.with_name(destination.name + "-registered-card-control")
    fake_registered_card = fake_registered_parent / "card-00"
    fake_session = fake_registered_card / "session-final"
    if fake_registered_parent.exists(): fail("registered-card control already exists")
    fake_registered_card.mkdir(parents=True)
    shutil.copytree(destination, fake_session)
    (fake_session / "artifacts.jsonl").write_text("", encoding="utf-8")
    (fake_registered_card / "finalize-result.json").write_text(json.dumps({"card_id":"00","success":True,"findings":17}) + "\n")
    registered_rejected = False
    try: run_finalize(fake_registered_card, repo)
    except EvidenceError as error: registered_rejected = "typed" in str(error) or "artifact" in str(error)
    if not registered_rejected: fail("registered Card00 with missing artifact proof was accepted")
    forged_plan = destination / "forged-plan"
    forged_plan.mkdir()
    (forged_plan / "finalize-result.json").write_text(json.dumps({"cards":14,"worktrees":14,"success":True}) + "\n")
    shutil.copytree(destination / "spawn-control", forged_plan / "session")
    fake_cards = destination / "forged-plan-spec-root"
    card_home = fake_cards / "cards"
    card_home.mkdir(parents=True)
    delimiter = chr(96)
    for card_number in range(14):
        card_file = card_home / f"{card_number:02d}-fake.md"
        card_file.write_text(f"# Card {card_number:02d}\n\n**Worktree:** {delimiter}/private/tmp/proxima-python-missing-card-{card_number:02d}{delimiter}\n")
    plan_rejected = False
    try: run_finalize_plan(forged_plan, fake_cards)
    except EvidenceError as error: plan_rejected = "missing .git" in str(error)
    if not plan_rejected: fail("forged 14-worktree summary did not fail on an absent worktree")
    plan_session_control = destination.with_name(destination.name + "-plan-session-control")
    plan_session_control.mkdir()
    shutil.copytree(destination, plan_session_control / "session")
    (plan_session_control / "session" / "receipt.json").unlink()
    plan_session_rejected = False
    try: run_finalize_plan(plan_session_control, fake_cards)
    except EvidenceError as error: plan_session_rejected = "missing final count receipt" in str(error)
    if not plan_session_rejected: fail("plan finalizer accepted a session missing its receipt")
    chain_reinit_rejected = False
    chain_control = destination / "chain-reinitialize-control"
    chain_control.mkdir()
    (chain_control / "segments.jsonl").write_text("retained\n",encoding="utf-8")
    try: write_segment_chain(chain_control)
    except EvidenceError as error: chain_reinit_rejected = "already initialized" in str(error)
    if not chain_reinit_rejected: fail("existing segment history was reinitialized")
    actual_worktree_head = worktree_head(repo.resolve())
    if not re.fullmatch(r"[0-9a-f]{40}", actual_worktree_head): fail("linked worktree HEAD could not be resolved")
    retry_counts = git_counts([
        {"argv":["git","push","origin","HEAD:main"],"exit_code":1,"cwd":str(repo.resolve())},
        {"argv":["git","push","origin","HEAD:main"],"exit_code":0,"cwd":str(repo.resolve())},
    ], repo, repo)
    wrong_worktree_rejected = git_counts([
        {"argv":["git","commit","-m","x"],"exit_code":0,"cwd":str(destination)},
        {"argv":["git","rebase","origin/main"],"exit_code":0,"cwd":str(destination)},
    ], repo) == (0,0,0)
    abort_rebase_rejected = git_counts([
        {"argv":["git","rebase","--abort"],"exit_code":0,"cwd":str(repo.resolve())},
    ], repo) == (0,0,0)
    if not wrong_worktree_rejected or not abort_rebase_rejected: fail("wrong-worktree or rebase-abort event counted as landed work")
    if retry_counts != (0,0,1): fail(f"failed push retry count={retry_counts}")
    wrong_ref_rejected = False
    try: git_counts([{"argv":["git","push","origin","HEAD:other"],"exit_code":0}])
    except EvidenceError: wrong_ref_rejected = True
    if not wrong_ref_rejected: fail("successful wrong-ref push was accepted")
    accepted_count_rejected = False
    try: require_accepted_count(1, 2)
    except EvidenceError: accepted_count_rejected = True
    if not accepted_count_rejected: fail("failed functional AC count was accepted")
    stale_root = destination.with_name(destination.name + "-stale-chain-control")
    if stale_root.exists(): fail("stale chain control already exists")
    stale_root.mkdir()
    chain_source = destination.parent / "segments.jsonl"
    if not chain_source.is_file(): fail("missing initialized segment chain for stale-row control")
    stale_rows = [json.loads(line) for line in chain_source.read_text(encoding="utf-8").splitlines()]
    for row in stale_rows:
        (stale_root / row["root"]).symlink_to(destination.parent / row["root"], target_is_directory=True)
    stale_rows[-1]["event_count"] += 1
    stale_rows[-1]["row_sha256"] = sha(json.dumps({key:value for key,value in stale_rows[-1].items() if key != "row_sha256"},sort_keys=True,separators=(",",":")).encode())
    (stale_root / "segments.jsonl").write_text("".join(json.dumps(row,sort_keys=True)+"\n" for row in stale_rows),encoding="utf-8")
    stale_rejected = False
    try: verify_segment_chain(stale_root)
    except EvidenceError as error: stale_rejected = "raw session segment inventory changed" in str(error)
    if not stale_rejected: fail("stale active segment row was accepted")
    fake_finalizer = stale_root / "card-00"
    fake_finalizer.mkdir()
    for root_name in ("session","session-invalid-archive","session-retry","session-failed-attempt-02","session-final"):
        target = destination if root_name == "session-final" else destination.parent / root_name
        (fake_finalizer / root_name).symlink_to(target,target_is_directory=True)
    shutil.copy2(stale_root / "segments.jsonl",fake_finalizer / "segments.jsonl")
    finalizer_rejected = False
    try: run_finalize(fake_finalizer,repo)
    except EvidenceError as error:
        finalizer_rejected = "raw session segment inventory changed" in str(error)
    if not finalizer_rejected: fail("terminal finalizer accepted stale segment history")
    cassette_root = destination / "cassette-jsonl-control"
    capture_one(cassette_root, [sys.executable, "-c", "from pathlib import Path;import os;path=Path(os.environ['PROXIMA_CASSETTE_DIR'])/'probe.jsonl';path.write_text('record')"])
    cassette_events = verify_bundle(cassette_root)
    cassette_items = cassette_events[0].get("immutable_artifacts", {})
    cassette_item = next((item for name,item in cassette_items.items() if name.endswith("probe.jsonl")), None)
    if cassette_item is None or not cassette_item["path"].startswith("commands/"): fail("JSONL cassette was not snapshotted")
    typed_copy = destination / "cassette-typed-row-control"
    shutil.copytree(cassette_root, typed_copy)
    typed_rows = [json.loads(line) for line in (typed_copy / "artifacts.jsonl").read_text(encoding="utf-8").splitlines()]
    typed_rows = [row for row in typed_rows if row["kind"] != "cassette"]
    (typed_copy / "artifacts.jsonl").write_text("".join(json.dumps(row,sort_keys=True)+"\n" for row in typed_rows),encoding="utf-8")
    typed_receipt = json.loads((typed_copy / "receipt.json").read_text(encoding="utf-8"))
    typed_receipt["artifacts_sha256"] = sha_file(typed_copy / "artifacts.jsonl")
    (typed_copy / "receipt.json").write_text(json.dumps(typed_receipt,sort_keys=True)+"\n",encoding="utf-8")
    typed_row_rejected = False
    try: verify_bundle(typed_copy)
    except EvidenceError as error: typed_row_rejected = "typed immutable" in str(error)
    if not typed_row_rejected: fail("missing typed cassette artifact row was accepted")
    (cassette_root / cassette_item["path"]).write_bytes(b"altered\n")
    cassette_tamper_rejected = False
    try: verify_bundle(cassette_root)
    except EvidenceError: cassette_tamper_rejected = True
    if not cassette_tamper_rejected: fail("modified JSONL cassette snapshot was accepted")
    inventory_control = destination / "committed-inventory-control"
    capture_one(inventory_control, [sys.executable, "-c", "print('')"])
    inventory_rows = load_events(inventory_control)
    inventory_rows[0]["argv"] = ["git","status","--short","--ignored","--untracked-files=all"]
    inventory_rows[0]["_evidence_root"] = str(inventory_control)
    (inventory_control / inventory_rows[0]["streams"]["stdout"]["path"]).write_text("",encoding="utf-8")
    diff_event = dict(inventory_rows[0]); diff_event["argv"] = ["git","diff","--name-only","origin/main...HEAD"]
    (inventory_control / diff_event["streams"]["stdout"]["path"]).write_text("",encoding="utf-8")
    tree_event = dict(inventory_rows[0]); tree_event["argv"] = ["git","ls-tree","-r","--name-only","HEAD"]
    (inventory_control / tree_event["streams"]["stdout"]["path"]).write_text("target/committed-file\n",encoding="utf-8")
    commit_diff_event = dict(inventory_rows[0]); commit_diff_event["argv"] = ["git","diff-tree","--no-commit-id","--name-only","-r","HEAD"]
    (inventory_control / commit_diff_event["streams"]["stdout"]["path"]).write_text("target/committed-file\n",encoding="utf-8")
    inventory_rows.extend([diff_event,tree_event,commit_diff_event])
    tracked_artifacts = repo_artifacts_from_events(inventory_rows, require_tree=True)
    if tracked_artifacts != ["target/committed-file"]: fail(f"committed artifact inventory={tracked_artifacts}")
    stale_status_rejected = False
    try: repo_artifacts_from_events(inventory_rows + [{"argv":["python3","-c","pass"],"exit_code":0}], require_tree=True)
    except EvidenceError as error: stale_status_rejected = "stale" in str(error)
    if not stale_status_rejected: fail("stale generated-artifact inventory was accepted")
    inventory_output = inventory_control / inventory_rows[0]["streams"]["stdout"]["path"]
    inventory_output.write_text("", encoding="utf-8")
    finalizer_tail = dict(inventory_rows[-1])
    finalizer_tail["argv"] = ["python3","proxima-tensor/specs/python-frontend/scripts/check_evidence.py","--finalize","/private/tmp/proxima-python-frontend/evidence/card-00","--repo-root",str(repo)]
    finalizer_tail["exit_code"] = 0
    if repo_artifacts_from_events(inventory_rows + [finalizer_tail], require_tree=True, allow_terminal_finalizer=True):
        fail("sealed finalizer inventory control reported unexpected repository artifacts")
    arbitrary_tail_rejected = False
    try: repo_artifacts_from_events(inventory_rows + [{"argv":["python3","-c","pass"],"exit_code":0}], require_tree=True, allow_terminal_finalizer=True)
    except EvidenceError as error: arbitrary_tail_rejected = "stale" in str(error)
    if not arbitrary_tail_rejected: fail("arbitrary post-inventory command was accepted")
    signal_root = destination / "signal-control"
    from capture_validation import capture_one
    capture_one(signal_root, [sys.executable, "-c", "import os,signal;os.kill(os.getpid(),signal.SIGTERM)"])
    signal_events = verify_bundle(signal_root)
    if len(signal_events) != 1 or signal_events[0].get("outcome") != "signaled" or signal_events[0].get("signal_number") != 15:
        fail("child signal did not produce a sealed signal outcome")
    preserved = all(path.exists() for path in previous) and sha_file(sentinel) == sentinel_hash
    if not preserved: fail("prior self-check attempt or sentinel was changed")
    inventory_root = destination / "repo-inventory"
    capture_one(inventory_root, ["git", "status", "--short", "--ignored", "--untracked-files=all"])
    inventory_event = load_events(inventory_root)[0]
    inventory_event["_evidence_root"] = str(inventory_root)
    if repo_artifacts_from_events([inventory_event]): fail("self-check found repository artifacts")
    return "captured_commands=2 stdout_records=2 stderr_records=2 exit_records=2 corrupted_artifact_rejected=1 tail_removal_rejected=1 repo_local_root_rejected=1 inherited_cassette_root_rejected=1 cassette_symlink_rejected=1 external_cassette_symlink_rejected=1 spawn_failure_sealed=1 signal_outcome_sealed=1 unregistered_card_rejected=1 registered_card_missing_artifact_rejected=1 forged_plan_missing_worktree_rejected=1 plan_session_receipt_rejected=1 failed_push_retry_counted=1 wrong_worktree_git_rejected=1 abort_rebase_rejected=1 wrong_ref_push_rejected=1 comparator_mismatch_rejected=1 failed_ac_count_rejected=1 cassette_jsonl_captured=1 cassette_jsonl_tamper_rejected=1 typed_cassette_row_required=1 committed_artifact_detected=1 stale_inventory_rejected=1 sealed_finalizer_inventory_accepted=1 stale_chain_rejected=1 terminal_finalizer_chain_rejected=1 linked_worktree_head_resolved=1 segment_reinitialize_rejected=1 missing_receipt_or_starts_rejected=1 receipt_hash_mismatch_rejected=1 start_provenance_mismatch_rejected=1 previous_self_checks_preserved=1 repository_artifacts=0"

def acceptance_allowlist(repo: Path, card_id: str) -> set[tuple[str, ...]]:
    card_files = sorted((repo / "proxima-tensor/specs/python-frontend/cards").glob(f"{card_id}-*.md"))
    if len(card_files) != 1: fail(f"expected one card file for {card_id}, found {len(card_files)}")
    commands = set()
    in_table = False
    for line in card_files[0].read_text(encoding="utf-8").splitlines():
        if "| id | command | expected |" in line:
            in_table = True; continue
        if in_table and not line.startswith("|"): break
        if not in_table or "| AC" not in line: continue
        cells = line.split("|")
        if len(cells) < 4: continue
        delimiter = chr(96)
        match = re.search(delimiter + r"([^" + delimiter + r"]+)" + delimiter, cells[2])
        if not match: fail(f"acceptance command lacks exact code argv in {card_files[0]}")
        argv = shlex.split(match.group(1))
        if "capture_validation.py" in " ".join(argv):
            if "--" not in argv: fail(f"capture command missing -- in {card_files[0]}")
            argv = argv[argv.index("--") + 1:]
        if "check_evidence.py" in " ".join(argv) and any(flag in argv for flag in ("--precommit", "--finalize", "--finalize-plan")): continue
        commands.add(tuple(argv))
    if not commands: fail(f"no functional acceptance commands found for card {card_id}")
    return commands

def functional_events(events: list[dict[str, Any]], allowlist: set[tuple[str, ...]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, ...], dict[str, Any]] = {}
    for event in events:
        argv = tuple(event.get("argv", []))
        if argv in allowlist: latest[argv] = event
    return list(latest.values())

def git_counts(events: list[dict[str, Any]], repo: Path | None = None, integration: Path | None = None) -> tuple[int, int, int]:
    commit = rebase = push = 0
    card_root = repo.resolve() if repo is not None else None
    integration_root = integration.resolve() if integration is not None else card_root
    for event in events:
        argv = event.get("argv", [])
        if event.get("exit_code") != 0 or not argv or Path(argv[0]).name != "git":
            continue
        cwd = Path(event.get("cwd", "")).resolve()
        if len(argv) > 1 and argv[1] == "commit" and (card_root is None or cwd == card_root):
            commit += 1
        if len(argv) > 1 and argv[1] == "rebase" and (card_root is None or cwd == card_root) and not any(flag in argv[2:] for flag in ("--abort","--continue","--skip","--edit-todo")):
            rebase += 1
        if len(argv) > 1 and argv[1] == "push":
            if argv[1:] != ["push", "origin", "HEAD:main"]:
                fail(f"successful push targets an unapproved ref: {argv}")
            if integration_root is None or cwd == integration_root:
                push += 1
    return commit, rebase, push

def declared_integration_worktree(card_file: Path) -> Path:
    marker = "**Integration worktree:**"
    for line in card_file.read_text(encoding="utf-8").splitlines():
        if line.startswith(marker):
            value = line.split(marker, 1)[1].strip()
            if value.startswith(chr(96)) and value.endswith(chr(96)):
                value = value[1:-1]
            return Path(value).resolve()
    fail(f"missing declared integration worktree in {card_file}")

def segment_inventory(root: Path, sequence: int, role: str, previous_hash: str | None, duplicate_of: int | None = None, snapshot_of: int | None = None, prefix_events: int | None = None, allow_pending: bool = False) -> dict[str, Any]:
    events = load_events(root)
    full_path = root / "full.log"
    full_rows = [json.loads(line) for line in full_path.read_text(encoding="utf-8").splitlines()]
    if len(events) != len(full_rows): fail(f"segment full-log count mismatch: {root}")
    for position, (event, full_record) in enumerate(zip(events, full_rows), 1):
        if full_record.get("event") != event: fail(f"segment full-log/manifest mismatch: {root} event {position}")
        for name in ("stdout", "stderr", "exit"):
            stream = event.get("streams", {}).get(name)
            if not stream: fail(f"segment missing {name} record: {root} event {position}")
            artifact = root / stream["path"]
            if not artifact.is_file() or sha_file(artifact) != stream["sha256"]:
                fail(f"segment {name} artifact hash mismatch: {root} event {position}")
            if name in ("stdout", "stderr"):
                payload_name = name + "_base64"
                payload = base64.b64decode(full_record.get(payload_name, ""), validate=True)
                if sha(payload) != stream["sha256"]: fail(f"segment full-log {name} payload mismatch: {root} event {position}")
        for artifact in event.get("immutable_artifacts", {}).values():
            artifact_path = root / artifact["path"]
            if not artifact_path.is_file() or sha_file(artifact_path) != artifact["sha256"]:
                fail(f"segment immutable artifact hash mismatch: {root} event {position}")
    timestamps = [event.get("started_at") for event in events if event.get("started_at")]
    ended = [event.get("ended_at") for event in events if event.get("ended_at")]
    commands = {}
    for file_path in sorted((root / "commands").rglob("*")):
        parts = file_path.relative_to(root).parts
        if allow_pending and len(parts) > 1 and parts[0] == "commands" and not (root / "commands" / parts[1] / "exit.json").exists():
            continue
        if file_path.is_file():
            relative = file_path.relative_to(root).as_posix()
            commands[relative] = {"bytes": file_path.stat().st_size, "sha256": sha_file(file_path)}
    failures = []
    for event in events:
        if event.get("exit_code") != 0:
            stderr = root / event["streams"]["stderr"]["path"]
            failures.append({"event_id": event["event_id"], "stderr_sha256": sha_file(stderr)})
    if snapshot_of is not None:
        source = root.parent / "session-retry"
        source_events = load_events(source)
        target_full = full_path.read_text(encoding="utf-8").splitlines()
        source_full = (source / "full.log").read_text(encoding="utf-8").splitlines()
        if prefix_events is None or events != source_events[:prefix_events] or target_full != source_full[:prefix_events]:
            fail(f"session prefix snapshot mismatch: {root}")
        pending_files = set()
        for relative, record in commands.items():
            parts = Path(relative).parts
            command_number = int(parts[1])
            source_path = source / relative
            if command_number <= prefix_events:
                if not source_path.is_file() or sha_file(source_path) != record["sha256"]:
                    fail(f"session snapshot command file mismatch: {relative}")
            else:
                if command_number != prefix_events + 1 or parts[-1] not in {"start.json", "stdout.bin", "stderr.bin"}:
                    fail(f"unexpected snapshot tail command artifact: {relative}")
                pending_files.add(relative)
        expected_pending = {f"commands/{prefix_events + 1:06d}/{name}" for name in ("start.json", "stdout.bin", "stderr.bin")}
        if pending_files != expected_pending:
            fail(f"session snapshot pending command files={sorted(pending_files)}")
        pending_start = json.loads((root / f"commands/{prefix_events + 1:06d}/start.json").read_text(encoding="utf-8"))
        if pending_start.get("event_id") != prefix_events + 1 or (root / f"commands/{prefix_events + 1:06d}/stdout.bin").stat().st_size or (root / f"commands/{prefix_events + 1:06d}/stderr.bin").stat().st_size:
            fail("session snapshot tail is not a start-only in-flight event")
    starts_path = root / "starts.jsonl"
    if starts_path.exists() and allow_pending:
        start_lines = starts_path.read_text(encoding="utf-8").splitlines()
        if len(start_lines) == len(events) + 1:
            starts_hash = sha(("\n".join(start_lines[:len(events)]) + ("\n" if events else "")).encode())
        else:
            starts_hash = sha_file(starts_path)
    else:
        starts_hash = sha_file(starts_path) if starts_path.exists() else None
    row = {"sequence": sequence, "root": root.name, "role": role, "event_count": len(events),
           "failed_event_count": len(failures), "first_started_at": timestamps[0] if timestamps else None,
           "last_ended_at": ended[-1] if ended else None, "manifest_sha256": sha_file(root / "manifest.jsonl"),
           "full_log_sha256": sha_file(full_path), "starts_sha256": starts_hash,
           "receipt_sha256": sha_file(root / "receipt.json") if (root / "receipt.json").exists() else None,
           "artifacts_catalog_sha256": sha_file(root / "artifacts.jsonl") if (root / "artifacts.jsonl").exists() else None,
           "comparisons_catalog_sha256": sha_file(root / "comparisons.jsonl") if (root / "comparisons.jsonl").exists() else None,
           "commands": commands, "failure_refs": failures, "previous_row_sha256": previous_hash,
           "duplicate_of": duplicate_of, "snapshot_of": snapshot_of, "prefix_events": prefix_events}
    row["row_sha256"] = sha(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
    return row

def write_segment_chain(card_root: Path) -> list[dict[str, Any]]:
    chain_path = card_root / "segments.jsonl"
    if chain_path.exists(): fail("segment chain is already initialized; refusing to replace history")
    definitions = [
        ("session", "historical_invalid", None),
        ("session-invalid-archive", "duplicate", 1),
        ("session-retry", "historical_superseded", None),
        ("session-failed-attempt-02", "snapshot", None),
        ("session-final", "active", None),
    ]
    rows = []
    previous = None
    for sequence, (relative, role, duplicate_of) in enumerate(definitions, 1):
        root = card_root / relative
        if not root.is_dir(): fail(f"missing retained session segment {relative}")
        snapshot_of = 3 if role == "snapshot" else None
        prefix_events = 26 if role == "snapshot" else None
        row = segment_inventory(root, sequence, role, previous, duplicate_of, snapshot_of, prefix_events)
        rows.append(row); previous = row["row_sha256"]
    with chain_path.open("x", encoding="utf-8") as chain_stream:
        chain_stream.write("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return rows

def verify_segment_chain(card_root: Path, refresh_active: bool = False) -> list[dict[str, Any]]:
    chain_path = card_root / "segments.jsonl"
    if not chain_path.is_file(): fail("missing initialized segments.jsonl")
    rows = [json.loads(line) for line in chain_path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 5: fail(f"segment chain rows={len(rows)} expected=5")
    previous = None
    for position, row in enumerate(rows, 1):
        recorded_hash = row.get("row_sha256")
        unhashed = dict(row); unhashed.pop("row_sha256", None)
        expected_hash = sha(json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode())
        if recorded_hash != expected_hash or row.get("sequence") != position or row.get("previous_row_sha256") != previous:
            fail(f"segment chain hash/order mismatch at row {position}")
        root = card_root / row["root"]
        if not root.is_dir(): fail(f"missing indexed session segment {row['root']}")
        pending = row.get("role") == "active" and any(not (directory / "exit.json").exists() for directory in (root / "commands").glob("[0-9][0-9][0-9][0-9][0-9][0-9]"))
        current = segment_inventory(root, position, row["role"], previous, row.get("duplicate_of"), row.get("snapshot_of"), row.get("prefix_events"), allow_pending=pending)
        if current != row: fail(f"raw session segment inventory changed: {row['root']}")
        if row.get("duplicate_of") is not None:
            source = rows[row["duplicate_of"] - 1]
            for key in ("event_count","failed_event_count","manifest_sha256","full_log_sha256","starts_sha256","receipt_sha256","artifacts_catalog_sha256","comparisons_catalog_sha256","commands","failure_refs"):
                if row.get(key) != source.get(key): fail(f"duplicate segment differs from source row: {row['root']} field={key}")
        if row.get("role") == "historical_invalid":
            ids = [event["event_id"] for event in load_events(root)]
            if ids == sorted(ids): fail("historical invalid segment no longer reproduces its event-order defect")
        previous = row["row_sha256"]
    if refresh_active:
        chain_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return rows

def require_accepted_count(accepted: int, expected: int) -> None:
    if accepted != expected: fail(f"accepted functional commands={accepted} expected={expected}")

def run_precommit(card_root: Path, repo: Path, card_id: str) -> str:
    session = card_root / "session-final" if card_id == "00" else (card_root / "session" if (card_root / "session").is_dir() else card_root)
    current_argv = ["python3", sys.argv[0], *sys.argv[1:]]
    events = verify_bundle(session, allow_inflight_argv=current_argv)
    allowlist = acceptance_allowlist(repo, card_id)
    functions = functional_events(events, allowlist)
    observed = {tuple(event.get("argv", [])) for event in functions}
    if observed != allowlist: fail(f"functional acceptance argv mismatch missing={sorted(allowlist - observed)} extra={sorted(observed - allowlist)}")
    accepted = sum(event["exit_code"] == 0 for event in functions)
    require_accepted_count(accepted, len(allowlist))
    segments = verify_segment_chain(card_root) if card_id == "00" else []
    history_events = sum(row["event_count"] for row in segments if row["role"] not in {"active", "duplicate", "snapshot"})
    history_failures = sum(row["failed_event_count"] for row in segments if row["role"] not in {"active", "duplicate", "snapshot"})
    for event in events: event["_evidence_root"] = str(session)
    artifacts = repo_artifacts_from_events(events, require_tree=True, allow_terminal_finalizer=True)
    if artifacts:
        fail(f"repository artifacts found: {artifacts}")
    result = f"card={card_id} functional_commands={len(functions)} accepted_commands={accepted} hashes_valid=1 generated_repo_artifacts=0"
    if card_id == "00":
        result += f" history_segments={len(segments)} history_events={history_events} history_failures={history_failures} history_hashes_valid=1"
    return result

def run_finalize(card_root: Path, repo: Path, write_result: bool = True) -> str:
    card_id = card_root.name.removeprefix("card-")
    validators = {"00": True}
    if card_id not in validators:
        fail(f"no evidence reader registered for card {card_id}")
    session = card_root / "session-final" if card_id == "00" else (card_root / "session" if (card_root / "session").is_dir() else card_root)
    current_argv = ["python3", sys.argv[0], *sys.argv[1:]]
    events = verify_bundle(session, allow_inflight_argv=current_argv)
    segments = verify_segment_chain(card_root)
    integration = declared_integration_worktree(next((repo / "proxima-tensor/specs/python-frontend/cards").glob("00-*.md")))
    commit, rebase, push = git_counts(events, repo.resolve(), integration)
    if not (card_root / "bootstrap.log").is_file():
        fail("missing copied bootstrap log")
    bootstrap_manifest_path = card_root / "bootstrap-manifest.json"
    if not bootstrap_manifest_path.is_file():
        fail("missing bootstrap manifest")
    bootstrap_manifest = json.loads(bootstrap_manifest_path.read_text(encoding="utf-8"))
    bootstrap_files = bootstrap_manifest.get("bootstrap_files", {})
    if "bootstrap.log" not in bootstrap_files:
        fail("bootstrap manifest does not hash the primary transcript")
    for relative, expected_hash in bootstrap_files.items():
        artifact = (card_root / relative).resolve()
        if card_root.resolve() not in artifact.parents or not artifact.is_file() or sha_file(artifact) != expected_hash:
            fail(f"bootstrap transcript hash mismatch: {relative}")
    functional = functional_events(events, acceptance_allowlist(repo, card_id))
    if len(functional) != len(acceptance_allowlist(repo, card_id)) or sum(item["exit_code"] == 0 for item in functional) != len(acceptance_allowlist(repo, card_id)):
        fail(f"expected exactly two successful functional validation events, found {len(functional)}")
    if commit != 1 or rebase < 1 or push != 1:
        fail(f"git events commit={commit} rebase={rebase} push={push}")
    for event in events: event["_evidence_root"] = str(session)
    artifacts = repo_artifacts_from_events(events, require_tree=True, allow_terminal_finalizer=True)
    if artifacts:
        fail(f"repository artifacts found: {artifacts}")
    head_events = [event for event in events if event.get("argv", [])[-2:] == ["rev-parse", "HEAD"]]
    if not head_events:
        fail("missing captured git rev-parse HEAD event")
    head_event = head_events[-1]
    commit_events = [event for event in events if event.get("exit_code") == 0 and event.get("argv", [])[:2] == ["git", "commit"] and Path(event.get("cwd", "")).resolve() == repo.resolve()]
    rebase_events = [event for event in events if event.get("exit_code") == 0 and event.get("argv", [])[:2] == ["git", "rebase"] and Path(event.get("cwd", "")).resolve() == repo.resolve() and "--abort" not in event.get("argv", [])]
    if not commit_events or not rebase_events or commit_events[-1]["event_id"] >= rebase_events[-1]["event_id"] or rebase_events[-1]["event_id"] >= head_event["event_id"]:
        fail("captured commit, rebase, and resulting HEAD are not ordered in the declared worktree")
    head_path = session / head_event["streams"]["stdout"]["path"]
    commit_hash = head_path.read_text(encoding="utf-8").strip()
    if len(commit_hash) != 40:
        fail("captured git rev-parse HEAD did not return a full commit id")
    actual_head = worktree_head(repo.resolve())
    if actual_head != commit_hash:
        fail(f"captured HEAD {commit_hash} differs from actual worktree HEAD {actual_head}")
    ancestry = [event for event in events if event.get("argv") == ["git", "merge-base", "--is-ancestor", "origin/main", "HEAD"] and event.get("exit_code") == 0]
    if not ancestry: fail("missing successful captured origin/main ancestry check")
    remote_events = [event for event in events if event.get("argv") == ["git", "ls-remote", "origin", "refs/heads/main"] and event.get("exit_code") == 0]
    if not remote_events: fail("missing successful captured origin main state")
    remote_text = (session / remote_events[-1]["streams"]["stdout"]["path"]).read_text(encoding="utf-8").strip()
    if not remote_text.startswith(commit_hash + "\trefs/heads/main"): fail(f"origin main does not point at pushed HEAD: {remote_text}")
    report = {"card_id": card_id, "repo_root": str(repo.resolve()), "commit": commit_hash, "commit_events": commit, "rebase_events": rebase, "push_events": push, "validation_events": len(functional), "full_log": True, "hashes_valid": True, "generated_repo_artifacts": 0, "literal_oracle_matches": 0, "rust_incumbent_controls": 0, "correctness_controls_not_applicable": 1, "history_segments": len(segments), "history_events": sum(row["event_count"] for row in segments if row["role"] not in {"active","duplicate","snapshot"}), "history_failures": sum(row["failed_event_count"] for row in segments if row["role"] not in {"active","duplicate","snapshot"}), "history_hashes_valid": True}
    result_path = card_root / "finalize-result.json"
    serialized_report = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if write_result:
        result_path.write_text(serialized_report, encoding="utf-8")
    elif not result_path.is_file() or json.loads(result_path.read_text(encoding="utf-8")) != report:
        fail(f"stored finalizer result differs from rerun reader: {result_path}")
    return f"bootstrap_records=1 validation_events={len(functional)} commit_events={commit} rebase_events>={rebase} push_events={push} full_log=1 hashes_valid=1 generated_repo_artifacts=0 literal_oracle_matches=0 rust_incumbent_controls=0 correctness_controls_not_applicable=1 history_segments={len(segments)} history_events={report["history_events"]} history_failures={report["history_failures"]} history_hashes_valid=1 git_head_matches=1 origin_main_matches_pushed_head=1 inventories_final=1"

def declared_worktree(card_file: Path) -> Path:
    delimiter = chr(96)
    pattern = r"\*\*Worktree:\*\*\s*" + delimiter + r"([^" + delimiter + r"]+)" + delimiter
    match = re.search(pattern, card_file.read_text(encoding="utf-8"))
    if not match: fail(f"card worktree declaration missing: {card_file}")
    return Path(match.group(1)).expanduser().resolve()

def worktree_head(worktree: Path) -> str:
    git_marker = worktree / ".git"
    if not git_marker.exists(): fail(f"declared worktree is missing .git: {worktree}")
    if git_marker.is_dir(): git_dir = git_marker
    else:
        marker = git_marker.read_text(encoding="utf-8").strip()
        if not marker.startswith("gitdir: "): fail(f"invalid worktree gitfile: {git_marker}")
        git_dir = (worktree / marker.removeprefix("gitdir: ")).resolve()
    common_marker = git_dir / "commondir"
    common_dir = (git_dir / common_marker.read_text(encoding="utf-8").strip()).resolve() if common_marker.is_file() else git_dir
    head_path = git_dir / "HEAD"
    if not head_path.is_file(): fail(f"worktree HEAD is missing: {head_path}")
    head = head_path.read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        reference_name = head.removeprefix("ref: ")
        reference = git_dir / reference_name
        if not reference.is_file(): reference = common_dir / reference_name
        if reference.is_file():
            head = reference.read_text(encoding="utf-8").strip()
        else:
            packed_refs = common_dir / "packed-refs"
            packed = next((line.split(" ", 1)[0] for line in packed_refs.read_text(encoding="utf-8").splitlines() if len(line.split(" ", 1)) == 2 and line.split(" ", 1)[1] == reference_name), None) if packed_refs.is_file() else None
            if packed is None: fail(f"worktree reference is missing: {reference_name}")
            head = packed
    if not re.fullmatch(r"[0-9a-f]{40}", head): fail(f"invalid worktree HEAD: {worktree}")
    return head

def run_finalize_plan(plan_root: Path, cards_dir: Path, active_argv: list[str] | None = None) -> str:
    plan_session = plan_root / "session"
    if not plan_session.is_dir(): fail("missing captured plan session")
    if active_argv is None:
        plan_events = verify_bundle(plan_session)
    else:
        plan_events = verify_bundle(plan_session, allow_inflight_argv=active_argv)
        pending_path = plan_session / "commands" / f"{len(plan_events) + 1:06d}" / "start.json"
        if not pending_path.is_file() or json.loads(pending_path.read_text(encoding="utf-8")).get("argv") != active_argv:
            fail("plan finalizer start is not the exact active captured invocation")
    if not cards_dir.is_dir(): fail("missing declared worktree directory")
    card_home = cards_dir / "cards" if (cards_dir / "cards").is_dir() else cards_dir
    card_files = sorted(card_home.glob("[0-9][0-9]-*.md"))
    if len(card_files) != 14: fail(f"card header count={len(card_files)} expected=14")
    cards = []
    for card_file in card_files:
        worktree = declared_worktree(card_file)
        head = worktree_head(worktree)
        card_id = card_file.name[:2]
        bundle = plan_root / f"card-{card_id}"
        if card_id not in {"00"}: fail(f"no evidence reader registered for card {card_id}")
        report = run_finalize(bundle, worktree, write_result=False)
        cards.append((card_id, worktree, head, report))
    if len({str(worktree) for _, worktree, _, _ in cards}) != 14: fail("declared worktree paths are not unique")
    commits = rebases = pushes = missing_full_logs = repository_artifacts = 0
    for card_id, worktree, head, _ in cards:
        result_path = plan_root / f"card-{card_id}" / "finalize-result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("commit") != head: fail(f"card {card_id} reported commit differs from actual worktree HEAD")
        commits += result.get("commit_events", 0)
        rebases += result.get("rebase_events", 0)
        pushes += result.get("push_events", 0)
        missing_full_logs += int(not result.get("full_log"))
        repository_artifacts += result.get("generated_repo_artifacts", 0)
    if missing_full_logs or repository_artifacts: fail(f"plan evidence incomplete: missing_full_logs={missing_full_logs} repository_artifacts={repository_artifacts}")
    return f"cards={len(cards)} worktrees={len({str(item[1]) for item in cards})} checked_bundles={len(cards)} commits={commits} rebases={rebases} pushes={pushes} missing_full_logs={missing_full_logs} repository_artifacts={repository_artifacts}"

def verify_sealed_plan(plan_root: Path) -> str:
    plan_session = plan_root / "session"
    events = verify_bundle(plan_session)
    if not events:
        fail("sealed plan session has no events")
    terminal = events[-1]
    argv = terminal.get("argv", [])
    if terminal.get("exit_code") != 0 or "--finalize-plan" not in argv or not any(Path(item).name == "check_evidence.py" for item in argv):
        fail("sealed plan session does not end with a successful plan finalizer")
    stdout = (plan_session / terminal["streams"]["stdout"]["path"]).read_text(encoding="utf-8")
    if not re.fullmatch(r"cards=14 worktrees=14 checked_bundles=14 commits=14 rebases=[0-9]+ pushes=14 missing_full_logs=0 repository_artifacts=0\n", stdout):
        fail("sealed plan finalizer output does not match the checked plan result")
    return f"plan_session_events={len(events)} terminal_plan_finalizer=1 hashes_valid=1"

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-check", type=Path)
    group.add_argument("--precommit", type=Path)
    group.add_argument("--finalize", type=Path)
    group.add_argument("--finalize-plan", type=Path)
    group.add_argument("--verify-plan", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--card-id")
    parser.add_argument("--cards", type=Path)
    args = parser.parse_args()
    try:
        if args.self_check:
            output = self_check(args.self_check, Path.cwd())
        elif args.precommit:
            if not args.repo_root or not args.card_id:
                fail("--precommit requires --repo-root and --card-id")
            output = run_precommit(args.precommit, args.repo_root, args.card_id)
        elif args.verify_plan:
            output = verify_sealed_plan(args.verify_plan)
        elif args.finalize:
            if not args.repo_root:
                fail("--finalize requires --repo-root")
            output = run_finalize(args.finalize, args.repo_root)
        else:
            if not args.cards:
                fail("--finalize-plan requires --cards")
            output = run_finalize_plan(args.finalize_plan, args.cards, ["python3", sys.argv[0], *sys.argv[1:]])
    except (EvidenceError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"evidence check failed: {error}", file=sys.stderr)
        return 1
    print(output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
