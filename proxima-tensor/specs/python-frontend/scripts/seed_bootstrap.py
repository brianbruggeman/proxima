#!/usr/bin/env python3
"""Capture and verify the first Proxima card shell session."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile
import time

DEFAULT_ROOT = pathlib.Path("/private/tmp/proxima-python-frontend/evidence/card-00a/session-attempt-10")
DEFAULT_RAW = pathlib.Path("/private/tmp/proxima-python-card-00a-attempt-10.log")
HOST_LOG = pathlib.Path("/private/tmp/proxima-python-card-00a-attempt-10-host.log")


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fail(message: str) -> None:
    raise ValueError(message)


def read_json(path: pathlib.Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail("invalid json {}: {}".format(path, error))


def read_rows(path: pathlib.Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            fail("invalid ledger row {}:{}: {}".format(path, line_number, error))
    return rows



def exact_line_positions(payload: bytes, marker: bytes) -> list[int]:
    positions = []
    offset = 0
    for line in payload.splitlines(keepends=True):
        if line.rstrip(bytes((13, 10))) == marker:
            positions.append(offset)
        offset += len(line)
    return positions


def captured_sha(payload: bytes, event_id: str) -> str:
    start_marker = ("PROXIMA_EVENT_START ns=" + event_id).encode()
    end_marker = ("PROXIMA_EVENT_END code=0 ns=" + event_id).encode()
    start = payload.find(start_marker)
    end = payload.find(end_marker, start + len(start_marker))
    if start < 0 or end < 0:
        fail("git output markers missing for event {}".format(event_id))
    section = payload[start + len(start_marker):end]
    matches = re.findall(rb"(?m)^([0-9a-f]{40})\r?$", section)
    if len(matches) != 1:
        fail("git output SHA count mismatch for event {}".format(event_id))
    return matches[0].decode()

def validate_entry(root: pathlib.Path, raw_path: pathlib.Path, repo_root: pathlib.Path, raw_bytes: bytes | None = None) -> tuple[dict, list[dict], bytes]:
    receipt = read_json(root / "launcher-receipt.json")
    host_bootstrap = read_json(root / "host-bootstrap.json")
    rows = read_rows(root / "bootstrap-events.jsonl")
    payload = raw_path.read_bytes() if raw_bytes is None else raw_bytes
    host_payload = HOST_LOG.read_bytes()
    expected_argv = ["/usr/bin/script", "-q", str(raw_path), "/bin/zsh"]
    if receipt.get("card_id") != "00a" or receipt.get("attempt") != "10" or receipt.get("raw_transcript") != str(raw_path):
        fail("receipt identity mismatch")
    if receipt.get("primary_script_argv") != expected_argv or receipt.get("cwd") != "/private/tmp":
        fail("bootstrap argv or cwd mismatch")
    runner_path = pathlib.Path(receipt.get("runner_path", ""))
    if runner_path != pathlib.Path("/private/tmp/proxima-python-card-00a-runner-10.py") or hashlib.sha256(runner_path.read_bytes()).hexdigest() != receipt.get("runner_sha256") or receipt.get("host_transcript") != str(HOST_LOG):
        fail("host runner receipt hash mismatch")
    if host_bootstrap != receipt:
        fail("host bootstrap receipt linkage mismatch")
    started = receipt.get("started_monotonic_ns")
    if not isinstance(started, int) or started <= 0:
        fail("bootstrap timer missing")
    zdotdir = pathlib.Path(receipt["zdotdir"])
    for name, expected in receipt["zdotdir_files"].items():
        if hashlib.sha256((zdotdir / name).read_bytes()).hexdigest() != expected:
            fail("zdotdir hash mismatch: {}".format(name))
    hook_path = root / "bootstrap-hook.zsh"
    hook_hash = hashlib.sha256(hook_path.read_bytes()).hexdigest()
    if receipt.get("hook_sha256") != hook_hash or receipt["zdotdir_files"].get("bootstrap-hook.zsh") != hook_hash:
        fail("active hook hash mismatch")
    bootstrap_line = ("ATTEMPT10_HOST_START monotonic_ns={} helper_sha256={}".format(started, read_json(root / "launcher-receipt.json").get("runner_sha256"))).encode()
    host_lines = [line.rstrip(bytes((13, 10))) for line in host_payload.splitlines(keepends=True)]
    if host_lines.count(bootstrap_line) != 1:
        fail("host bootstrap receipt line mismatch")
    bootstrap_position = host_payload.find(bootstrap_line)
    fetch_in_host = host_payload.find(b"fetch origin main")
    if bootstrap_position < 0 or fetch_in_host < 0 or bootstrap_position >= fetch_in_host:
        fail("host bootstrap order mismatch")
    hook_marker = b"PROXIMA_HOOK_SOURCED sha=" + hook_hash.encode()
    hook_offsets = []
    offset = 0
    for line in payload.splitlines(keepends=True):
        if line.rstrip(bytes((13, 10))) == hook_marker:
            hook_offsets.append(offset)
        offset += len(line)
    if len(hook_offsets) != 1:
        fail("hook source marker count mismatch")
    hook_position = hook_offsets[0]
    expected_repo = "/private/tmp/proxima-python-plan-amend-00a-split"
    fetch_starts = [row for row in rows if row.get("kind") == "start" and row.get("command") == "git -C " + expected_repo + " fetch origin main"]
    fetch_pairs = [(start, [end for end in rows if end.get("kind") == "end" and end.get("event_id") == start.get("event_id")]) for start in fetch_starts]
    fetch_successes = [(start, terminal_rows[0]) for start, terminal_rows in fetch_pairs if len(terminal_rows) == 1 and terminal_rows[0].get("exit_code") == 0 and terminal_rows[0].get("command") == start.get("command") and terminal_rows[0].get("cwd") == start.get("cwd") and terminal_rows[0].get("started_monotonic_ns") == start.get("started_monotonic_ns") and terminal_rows[0].get("ended_monotonic_ns", 0) >= start.get("started_monotonic_ns", 1)]
    if len(fetch_successes) != 1:
        fail("clean source fetch successful terminal count mismatch")
    fetch_event, fetch_terminal = fetch_successes[0]
    fetch_start_marker = ("PROXIMA_EVENT_START ns=" + fetch_event["event_id"]).encode()
    fetch_end_marker = ("PROXIMA_EVENT_END code=0 ns=" + fetch_event["event_id"]).encode()
    fetch_start_positions = exact_line_positions(payload, fetch_start_marker)
    fetch_end_positions = exact_line_positions(payload, fetch_end_marker)
    if len(fetch_start_positions) != 1 or len(fetch_end_positions) != 1:
        fail("fetch event raw marker count mismatch")
    fetch_position = fetch_start_positions[0]
    fetch_end_position = fetch_end_positions[0]
    fetch_output_position = payload.find(b"From github.com:brianbruggeman/proxima", fetch_position, fetch_end_position)
    first_repo = next((row for row in rows if row.get("kind") == "start" and row.get("cwd") == "/private/tmp" and row.get("command") == "git -C " + expected_repo + " status --short"), None)
    if first_repo is None:
        fail("first repository event missing")
    first_command = next((row for row in rows if row.get("kind") == "start"), None)
    if first_command is None or first_command.get("event_id") != first_repo.get("event_id"):
        fail("clean source status is not the first captured command")
    first_repo_terminals = [row for row in rows if row.get("kind") == "end" and row.get("event_id") == first_repo.get("event_id")]
    if len(first_repo_terminals) != 1:
        fail("first repository event terminal count mismatch")
    first_repo_end = first_repo_terminals[0]
    if first_repo_end.get("exit_code") != 0 or first_repo_end.get("cwd") != "/private/tmp" or first_repo_end.get("command") != first_repo.get("command") or first_repo_end.get("started_monotonic_ns") != first_repo.get("started_monotonic_ns") or first_repo_end.get("ended_monotonic_ns", 0) < first_repo.get("started_monotonic_ns", 1):
        fail("first repository event terminal mismatch")
    start_marker = ("PROXIMA_EVENT_START ns=" + first_repo["event_id"]).encode()
    end_marker = ("PROXIMA_EVENT_END code=0 ns=" + first_repo["event_id"]).encode()
    repo_start_positions = exact_line_positions(payload, start_marker)
    repo_end_positions = exact_line_positions(payload, end_marker)
    if len(repo_start_positions) != 1 or len(repo_end_positions) != 1:
        fail("first repository event raw marker count mismatch")
    repo_start_position = repo_start_positions[0]
    repo_end_position = repo_end_positions[0]
    status_span = payload[repo_start_position + len(start_marker):repo_end_position]
    status_output = re.sub(rb"\x1b\[[0-?]*[ -/]*[@-~]", b"", status_span)
    status_output = re.sub(rb"^\r\n% +\r \r$", b"", status_output)
    status_output = status_output.strip(bytes((13, 10)))
    if status_output:
        fail("clean source status emitted paths")
    if not hook_position < repo_start_position < repo_end_position < fetch_position < fetch_output_position < fetch_end_position:
        fail("hook or first repository event order mismatch")
    hook_events = [row for row in rows if row.get("kind") in {"start", "end"}]
    if len(hook_events) < 2:
        fail("hook event count mismatch")
    return receipt, rows, payload


def verify_entry(args: argparse.Namespace) -> None:
    root = args.verify_entry
    receipt, rows, payload = validate_entry(root, args.raw, args.repo_root)
    hook_marker = b"PROXIMA_HOOK_SOURCED sha=" + receipt["hook_sha256"].encode()
    omitted_rejected = 0
    try:
        validate_entry(root, args.raw, args.repo_root, payload.replace(hook_marker, b"PROXIMA_HOOK_OMITTED sha=" + receipt["hook_sha256"].encode()))
    except ValueError as error:
        omitted_rejected = int(str(error) == "hook source marker count mismatch")
    if omitted_rejected != 1:
        fail("omitted command control did not reject for expected reason")
    first_repo = next(row for row in rows if row.get("kind") == "start" and row.get("command", "").startswith("git -C /private/tmp/proxima-python-plan-amend-00a-split status --short"))
    status_start = ("PROXIMA_EVENT_START ns=" + first_repo["event_id"]).encode()
    status_end = ("PROXIMA_EVENT_END code=0 ns=" + first_repo["event_id"]).encode()
    status_start_positions = exact_line_positions(payload, status_start)
    status_end_positions = exact_line_positions(payload, status_end)
    if len(status_start_positions) != 1 or len(status_end_positions) != 1:
        fail("dirty status control marker count mismatch")
    status_start_at = status_start_positions[0]
    status_end_at = status_end_positions[0]
    clean_control = payload[:status_start_at + len(status_start)] + bytes((10,)) + b"?? dirty" + bytes((10,)) + payload[status_end_at:]
    dirty_rejected = 0
    try:
        validate_entry(root, args.raw, args.repo_root, clean_control)
    except ValueError as error:
        dirty_rejected = int(str(error) == "clean source status emitted paths")
    if dirty_rejected != 1:
        fail("dirty source control did not reject for expected reason")
    host_entries = sum(line.rstrip(bytes((13, 10))) == hook_marker for line in payload.splitlines(keepends=True))
    if host_entries != 1:
        fail("hook startup record count mismatch")
    hook_events = sum(row.get("kind") in {"start", "end"} for row in rows)
    print("hook_startup_records={} receipt_hashes_valid=1 hook_events>=2 observed_hook_events={} setup_cd=0 first_repo_after_hook=1 clean_source_status=1 omitted_command_rejected=1 dirty_source_rejected=1".format(host_entries, hook_events))

def precommit(args: argparse.Namespace) -> None:
    root = args.precommit
    _, rows, payload = validate_entry(root, args.raw, args.repo_root)
    precommit_starts = [row for row in rows if row.get("kind") == "start" and "--precommit" in row.get("command", "")]
    ended_ids = {row.get("event_id") for row in rows if row.get("kind") == "end"}
    pending = [row for row in precommit_starts if row.get("event_id") not in ended_ids]
    if len(pending) != 1:
        fail("expected exactly one pending precommit start")
    expected_ac1 = "python3 proxima-tensor/specs/python-frontend/scripts/seed_bootstrap.py --verify-entry {} --raw {} --repo-root {}".format(root, args.raw, args.repo_root)
    ac1_starts = [row for row in rows if row.get("kind") == "start" and row.get("command") == expected_ac1 and row.get("cwd") == str(args.repo_root)]
    accepted_ac1 = []
    pattern = re.compile(rb"(?m)^hook_startup_records=1 receipt_hashes_valid=1 hook_events>=2 observed_hook_events=[0-9]+ setup_cd=0 first_repo_after_hook=1 clean_source_status=1 omitted_command_rejected=1 dirty_source_rejected=1\r?$")
    for start_event in ac1_starts:
        terminals = [row for row in rows if row.get("kind") == "end" and row.get("event_id") == start_event.get("event_id")]
        if len(terminals) != 1:
            fail("AC00a.1 terminal count mismatch")
        terminal = terminals[0]
        if terminal.get("command") != start_event.get("command") or terminal.get("cwd") != start_event.get("cwd") or terminal.get("started_monotonic_ns") != start_event.get("started_monotonic_ns") or terminal.get("ended_monotonic_ns", 0) < start_event.get("started_monotonic_ns", 1):
            fail("AC00a.1 terminal provenance mismatch")
        start_marker = ("PROXIMA_EVENT_START ns=" + start_event["event_id"]).encode()
        end_marker = ("PROXIMA_EVENT_END code={} ns=".format(terminal.get("exit_code")) + start_event["event_id"]).encode()
        if terminal.get("exit_code") != 0:
            continue
        start_positions = exact_line_positions(payload, start_marker)
        end_positions = exact_line_positions(payload, end_marker)
        if len(start_positions) != 1 or len(end_positions) != 1:
            fail("AC00a.1 raw marker count mismatch for {}: starts={} ends={}".format(start_event["event_id"], len(start_positions), len(end_positions)))
        event_output = payload[start_positions[0] + len(start_marker):end_positions[0]]
        if pattern.search(event_output):
            accepted_ac1.append(start_event)
    if len(accepted_ac1) != 1:
        fail("expected exactly one successful AC00a.1 output event")
    prefix_path = root / "transcript-prefix.log"
    prefix_path.write_bytes(payload)
    prefix_matches = prefix_path.read_bytes() == args.raw.read_bytes()
    if not prefix_matches:
        fail("transcript prefix mismatch")
    precommit_event = pending[-1]
    prefix_record = {
        "event_id": precommit_event["event_id"],
        "prefix_size": len(payload),
        "prefix_sha256": hashlib.sha256(payload).hexdigest(),
    }
    (root / "precommit-check.json").write_text(json.dumps(prefix_record, sort_keys=True) + "\n", encoding="utf-8")
    print("card=00a functional=1 completed_functional_commands=1 pending_precommit_starts=1 prefix_snapshot=1 hashes_valid=1")


def finalize(args: argparse.Namespace, output: bool = True, write_record: bool = True) -> dict:
    root = args.finalize
    receipt, rows, payload = validate_entry(root, args.raw, args.repo_root)
    for token, marker in [
        ("--verify-entry", "hook_startup_records=1 receipt_hashes_valid=1"),
        ("--precommit", "card=00a functional=1 completed_functional_commands=1"),
    ]:
        if marker.encode() not in payload:
            fail("required AC output missing: {}".format(token))
    ended = {row.get("event_id"): row for row in rows if row.get("kind") == "end"}
    starts = [row for row in rows if row.get("kind") == "start"]
    precommit_record = read_json(root / "precommit-check.json")
    precommit_event_id = precommit_record.get("event_id")
    precommit_start = next((row for row in starts if row.get("event_id") == precommit_event_id), None)
    precommit_end = ended.get(precommit_event_id)
    if precommit_start is None or "--precommit" not in precommit_start.get("command", "") or precommit_end is None or precommit_end.get("command") != precommit_start.get("command") or precommit_end.get("cwd") != precommit_start.get("cwd") or precommit_end.get("started_monotonic_ns") != precommit_start.get("started_monotonic_ns") or precommit_end.get("exit_code") != 0:
        fail("accepted precommit event pair missing")
    landing_floor = precommit_end.get("ended_monotonic_ns", 0)
    landing_specs = [
        ("commit", lambda argv, cwd: argv[:2] == ["git", "commit"] and "-m" in argv and "--dry-run" not in argv and cwd == str(args.repo_root), str(args.repo_root)),
        ("rebase", lambda argv, cwd: argv == ["git", "rebase", "origin/main"] and cwd == str(args.repo_root), str(args.repo_root)),
        ("integration_ff", lambda argv, cwd: argv == ["git", "merge", "--ff-only", "codex/python-frontend-card-00a-final-10"] and cwd == "/private/tmp/proxima-python-card-00a-attempt-10-integration", "/private/tmp/proxima-python-card-00a-attempt-10-integration"),
        ("push", lambda argv, cwd: argv == ["git", "push", "origin", "HEAD:main"] and cwd == "/private/tmp/proxima-python-card-00a-attempt-10-integration", "/private/tmp/proxima-python-card-00a-attempt-10-integration"),
        ("remote", lambda argv, cwd: argv == ["git", "ls-remote", "origin", "refs/heads/main"] and cwd == "/private/tmp/proxima-python-card-00a-attempt-10-integration", "/private/tmp/proxima-python-card-00a-attempt-10-integration"),
    ]
    successful_landing = []
    for name, predicate, expected_cwd in landing_specs:
        matches = []
        for row in starts:
            try:
                command_argv = shlex.split(row.get("command", ""))
            except ValueError:
                continue
            if predicate(command_argv, row.get("cwd", "")) and row.get("started_monotonic_ns", 0) > landing_floor:
                matches.append(row)
        successful_matches = []
        for candidate in matches:
            terminal = ended.get(candidate.get("event_id"))
            if terminal is not None and terminal.get("exit_code") == 0 and terminal.get("command") == candidate.get("command") and terminal.get("cwd") == expected_cwd:
                successful_matches.append(terminal)
        if len(successful_matches) != 1:
            fail("landing successful command count mismatch: {}".format(name))
        successful_landing.append(successful_matches[0])
    if [row["started_monotonic_ns"] for row in successful_landing] != sorted(row["started_monotonic_ns"] for row in successful_landing):
        fail("landing command order mismatch")
    head_specs = [
        ("card_head", str(args.repo_root)),
        ("integration_head", "/private/tmp/proxima-python-card-00a-attempt-10-integration"),
    ]
    head_event_ids = []
    head_values = []
    head_starts = []
    for name, expected_cwd in head_specs:
        matches = []
        for row in starts:
            try:
                command_argv = shlex.split(row.get("command", ""))
            except ValueError:
                continue
            expected_after = successful_landing[1]["started_monotonic_ns"] if name == "card_head" else successful_landing[2]["started_monotonic_ns"]
            if command_argv == ["git", "rev-parse", "HEAD"] and row.get("cwd") == expected_cwd and row.get("started_monotonic_ns", 0) > expected_after:
                matches.append(row)
        successful_matches = []
        for candidate in matches:
            terminal = ended.get(candidate.get("event_id"))
            if terminal is not None and terminal.get("exit_code") == 0 and terminal.get("command") == candidate.get("command") and terminal.get("cwd") == expected_cwd:
                successful_matches.append((candidate, terminal))
        if len(successful_matches) != 1:
            fail("post-landing HEAD successful command count mismatch: {}".format(name))
        candidate, terminal = successful_matches[0]
        head_event_ids.append(candidate["event_id"])
        head_starts.append(candidate["started_monotonic_ns"])
        head_values.append(captured_sha(payload, candidate["event_id"]))
    remote_row = successful_landing[-1]
    remote_marker = ("PROXIMA_EVENT_START ns=" + remote_row["event_id"]).encode()
    remote_end = ("PROXIMA_EVENT_END code=0 ns=" + remote_row["event_id"]).encode()
    remote_start_pos = payload.find(remote_marker)
    remote_end_pos = payload.find(remote_end, remote_start_pos + len(remote_marker))
    if remote_start_pos < 0 or remote_end_pos < 0:
        fail("remote output markers missing")
    remote_output = payload[remote_start_pos + len(remote_marker):remote_end_pos]
    remote_matches = re.findall(rb"(?m)^([0-9a-f]{40})\s+refs/heads/main\r?$", remote_output)
    if len(remote_matches) != 1 or len(set(head_values + [remote_matches[0].decode()])) != 1:
        fail("card, integration, and remote HEAD mismatch")
    if not successful_landing[1]["started_monotonic_ns"] < head_starts[0] < successful_landing[2]["started_monotonic_ns"] < head_starts[1] < successful_landing[3]["started_monotonic_ns"] < successful_landing[4]["started_monotonic_ns"]:
        fail("HEAD and remote proof order mismatch")
    precommit_record = read_json(root / "precommit-check.json")
    precommit_event_id = precommit_record.get("event_id")
    precommit_start = next((row for row in starts if row.get("event_id") == precommit_event_id), None)
    precommit_end = ended.get(precommit_event_id)
    if precommit_start is None or "--precommit" not in precommit_start.get("command", "") or precommit_end is None or precommit_end.get("command") != precommit_start.get("command") or precommit_end.get("cwd") != precommit_start.get("cwd") or precommit_end.get("started_monotonic_ns") != precommit_start.get("started_monotonic_ns") or precommit_end.get("exit_code") != 0:
        fail("accepted precommit event pair missing")
    raw = args.raw.read_bytes()
    precommit_start_marker = ("PROXIMA_EVENT_START ns=" + precommit_event_id).encode()
    precommit_end_marker = ("PROXIMA_EVENT_END code=0 ns=" + precommit_event_id).encode()
    precommit_start_positions = exact_line_positions(raw, precommit_start_marker)
    precommit_end_positions = exact_line_positions(raw, precommit_end_marker)
    if len(precommit_start_positions) != 1 or len(precommit_end_positions) != 1 or precommit_start_positions[0] >= precommit_end_positions[0]:
        fail("accepted precommit raw marker pair missing")
    precommit_output = raw[precommit_start_positions[0] + len(precommit_start_marker):precommit_end_positions[0]]
    if b"card=00a functional=1 completed_functional_commands=1 pending_precommit_starts=1 prefix_snapshot=1 hashes_valid=1" not in precommit_output:
        fail("accepted precommit raw output mismatch")
    prefix_path = root / "transcript-prefix.log"
    prefix = prefix_path.read_bytes()
    if len(prefix) != precommit_record.get("prefix_size") or hashlib.sha256(prefix).hexdigest() != precommit_record.get("prefix_sha256"):
        fail("precommit prefix receipt mismatch")
    if not raw.startswith(prefix) or len(prefix) >= len(raw):
        fail("precommit prefix is not a strict prefix of final transcript")
    elapsed = (time.monotonic_ns() - receipt["started_monotonic_ns"]) / 1_000_000_000
    if precommit_end is None or elapsed > 1800:
        fail("postpush closure counts mismatch")
    record = {
        "card_id": "00a", "argv": sys.argv, "cwd": os.getcwd(),
        "started_monotonic_ns": receipt["started_monotonic_ns"], "ended_monotonic_ns": time.monotonic_ns(),
        "elapsed_seconds": elapsed, "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "transcript_prefix_sha256": digest(prefix_path), "raw_size": len(raw), "landing_event_ids": [row["event_id"] for row in successful_landing], "head_event_ids": head_event_ids, "head_sha256": head_values[0],
        "sealed_precommit_events": 1, "precommit_event_id": precommit_event_id, "exit_code": 0,
    }
    if write_record:
        (root / "postpush-check.json").write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    if output:
        print("postpush_check=1 captured_landing_records=5 sealed_precommit_events=1 elapsed_seconds<=1800")
    return record


def capture_closure(args: argparse.Namespace) -> None:
    argv = shlex.split(args.finalize_argv)
    started = time.monotonic_ns()
    process = subprocess.run(argv, cwd=args.repo_root, capture_output=True, text=True, check=False)
    ended = time.monotonic_ns()
    finalizer = {
        "argv": argv, "cwd": str(args.repo_root), "started_monotonic_ns": started, "ended_monotonic_ns": ended,
        "exit_code": process.returncode, "stdout": process.stdout, "stderr": process.stderr,
        "stdout_sha256": hashlib.sha256(process.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(process.stderr.encode()).hexdigest(),
    }
    args.finalizer_log.write_text(json.dumps(finalizer, sort_keys=True) + "\n", encoding="utf-8")
    if process.stdout:
        sys.stdout.write(process.stdout)
    if process.stderr:
        sys.stderr.write(process.stderr)
    if process.returncode != 0:
        raise SystemExit(process.returncode)


def postseal(args: argparse.Namespace) -> None:
    root = args.postseal
    receipt, rows, payload = validate_entry(root, args.raw, args.repo_root)
    ended = {row.get("event_id"): row for row in rows if row.get("kind") == "end"}
    closure_prefix = "python3 " + str(pathlib.Path(__file__).resolve()) + " --capture-postpush-closure "
    closure_starts = [row for row in rows if row.get("kind") == "start" and row.get("command", "").startswith(closure_prefix) and row.get("cwd") == str(args.repo_root)]
    closure_ends = [ended.get(row.get("event_id")) for row in closure_starts]
    successful_closures = [row for row in closure_ends if row is not None and row.get("exit_code") == 0]
    if len(successful_closures) != 1:
        fail("postpush closure successful terminal count mismatch")
    closure_end = successful_closures[0]
    check = read_json(root / "postpush-check.json")
    finalizer = read_json(root / "finalizer.json")
    expected_finalizer_argv = ["python3", str(pathlib.Path(__file__).resolve()), "--finalize", str(root)]
    if finalizer.get("argv", [])[:4] != expected_finalizer_argv or finalizer.get("cwd") != str(args.repo_root) or finalizer.get("exit_code") != 0:
        fail("captured finalizer invocation mismatch")
    if hashlib.sha256(finalizer.get("stdout", "").encode()).hexdigest() != finalizer.get("stdout_sha256") or hashlib.sha256(finalizer.get("stderr", "").encode()).hexdigest() != finalizer.get("stderr_sha256"):
        fail("captured finalizer stream hash mismatch")
    if "postpush_check=1 captured_landing_records=5 sealed_precommit_events=1 elapsed_seconds<=1800" not in finalizer.get("stdout", ""):
        fail("captured finalizer output mismatch")
    if check.get("exit_code") != 0 or hashlib.sha256(payload[:check.get("raw_size", 0)]).hexdigest() != check.get("raw_sha256"):
        fail("postpush check does not bind sealed raw transcript")
    args.sealed_transcript.parent.mkdir(parents=True, exist_ok=True)
    args.sealed_transcript.write_bytes(payload)
    record = {"closure_event_id": closure_end["event_id"], "raw_sha256": hashlib.sha256(payload).hexdigest(),
              "full_log_sha256": digest(args.sealed_transcript), "exit_code": closure_end["exit_code"], "closed": True}
    (root / "postseal.json").write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    if (time.monotonic_ns() - receipt["started_monotonic_ns"]) / 1_000_000_000 > 1800:
        fail("elapsed time exceeds cap")
    print("card=00a preclose_log=1 captured_landing_records>=5 sealed_precommit_events=1 sealed_postpush_events=1 elapsed_seconds<=1800")


def record_script_pid(args: argparse.Namespace) -> None:
    pid = args.script_pid
    if pid is None or pid < 1:
        fail("primary script PID missing")
    try:
        os.kill(pid, 0)
    except OSError as error:
        fail("primary script PID is not live: {}".format(error))
    root = args.record_script_pid
    raw = args.raw.read_bytes()
    record = {"pid": pid, "recorded_monotonic_ns": time.monotonic_ns(),
              "raw_sha256": hashlib.sha256(raw).hexdigest(), "raw_size": len(raw),
              "argv": ["/usr/bin/script", "-q", str(args.raw), "/bin/zsh"]}
    (root / "primary-script-shell-pid.json").write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    print("primary_script_pid_recorded=1")


def seal_closed(args: argparse.Namespace) -> None:
    root = args.seal_closed
    raw = args.raw.read_bytes()
    full = args.sealed_transcript
    if full is None:
        fail("closed transcript path missing")
    launch_receipt = read_json(root / "primary-script-pid.json")
    exit_receipt = read_json(root / "primary-script-exit.json")
    pid_receipt = read_json(root / "primary-script-shell-pid.json")
    expected_argv = ["/usr/bin/script", "-q", str(args.raw), "/bin/zsh"]
    pid_prefix = raw[:pid_receipt.get("raw_size", 0)]
    launch_prefix = raw[:launch_receipt.get("raw_size", 0)]
    if launch_receipt.get("argv") != expected_argv or not isinstance(launch_receipt.get("started_monotonic_ns"), int) or launch_receipt.get("raw_size", -1) < 0 or hashlib.sha256(launch_prefix).hexdigest() != launch_receipt.get("raw_prefix_sha256"):
        fail("primary script launch receipt mismatch")
    if exit_receipt.get("exit_code") != 0 or exit_receipt.get("pid") != launch_receipt.get("pid") or exit_receipt.get("argv") != expected_argv or exit_receipt.get("raw_sha256") != hashlib.sha256(raw).hexdigest() or exit_receipt.get("raw_size") != len(raw):
        fail("closed primary script wrapper receipt mismatch")
    if not read_json(root / "launcher-receipt.json")["started_monotonic_ns"] <= launch_receipt.get("started_monotonic_ns", 0) < pid_receipt.get("recorded_monotonic_ns", 0) < exit_receipt.get("ended_monotonic_ns", 0) or exit_receipt.get("started_monotonic_ns") != read_json(root / "launcher-receipt.json").get("started_monotonic_ns"):
        fail("primary script receipt chronology mismatch")
    if pid_receipt.get("pid") != launch_receipt.get("pid") or pid_receipt.get("argv") != expected_argv or not isinstance(pid_receipt.get("recorded_monotonic_ns"), int) or not read_json(root / "launcher-receipt.json")["started_monotonic_ns"] < pid_receipt["recorded_monotonic_ns"] < exit_receipt.get("ended_monotonic_ns", 0) or pid_receipt.get("raw_size", 0) <= 0 or hashlib.sha256(pid_prefix).hexdigest() != pid_receipt.get("raw_sha256"):
        fail("primary script PID receipt mismatch")
    host_payload = HOST_LOG.read_bytes()
    wrapper_line = ("host_wrapper_waited=1 script_pid={} script_exit=0 raw_sha256={} raw_size={}".format(exit_receipt["pid"], exit_receipt["raw_sha256"], exit_receipt["raw_size"])).encode()
    host_lines = [line.rstrip(bytes((13, 10))) for line in host_payload.splitlines(keepends=True)]
    if host_lines.count(wrapper_line) != 1:
        fail("closed host wrapper transcript receipt mismatch")
    if full.exists():
        prior = full.with_name("full.log.preclose")
        if not prior.exists():
            prior.write_bytes(full.read_bytes())
        prior_hash = hashlib.sha256(prior.read_bytes()).hexdigest()
        postseal_record = read_json(root / "postseal.json")
        if postseal_record.get("raw_sha256") != prior_hash or postseal_record.get("full_log_sha256") != prior_hash:
            fail("preclose copy does not match postseal hash receipt")
        if not raw.startswith(prior.read_bytes()):
            fail("preclose full log is not a prefix of closed raw")
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(raw)
    receipt = read_json(root / "launcher-receipt.json")
    record = {"raw_size": len(raw), "raw_sha256": hashlib.sha256(raw).hexdigest(),
              "host_log_size": len(host_payload), "host_log_sha256": hashlib.sha256(host_payload).hexdigest(),
              "sealed_monotonic_ns": time.monotonic_ns(), "started_monotonic_ns": receipt["started_monotonic_ns"]}
    (root / "closed-seal.json").write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    print("closed_raw_sealed=1 raw_size={}".format(len(raw)))


def verify_closed(args: argparse.Namespace) -> None:
    root = args.verify_closed
    receipt, rows, raw = validate_entry(root, args.raw, args.repo_root)
    full = args.sealed_transcript
    if full is None:
        fail("closed transcript path missing")
    copied = full.read_bytes()
    host_payload = HOST_LOG.read_bytes()
    seal = read_json(root / "closed-seal.json")
    postseal_record = read_json(root / "postseal.json")
    finalizer = read_json(root / "finalizer.json")
    check = read_json(root / "postpush-check.json")
    precommit = read_json(root / "precommit-check.json")
    script_exit = read_json(root / "primary-script-exit.json")
    launch_receipt = read_json(root / "primary-script-pid.json")
    pid_receipt = read_json(root / "primary-script-shell-pid.json")
    prefix_path = root / "transcript-prefix.log"
    prefix_bytes = prefix_path.read_bytes()
    expected_argv = ["/usr/bin/script", "-q", str(args.raw), "/bin/zsh"]
    pid_prefix = raw[:pid_receipt.get("raw_size", 0)]
    launch_prefix = raw[:launch_receipt.get("raw_size", 0)]
    if launch_receipt.get("argv") != expected_argv or not isinstance(launch_receipt.get("started_monotonic_ns"), int) or launch_receipt.get("raw_size", -1) < 0 or hashlib.sha256(launch_prefix).hexdigest() != launch_receipt.get("raw_prefix_sha256"):
        fail("primary script launch receipt mismatch")
    if script_exit.get("exit_code") != 0 or script_exit.get("pid") != launch_receipt.get("pid") or script_exit.get("argv") != expected_argv or script_exit.get("raw_sha256") != hashlib.sha256(raw).hexdigest() or script_exit.get("raw_size") != len(raw):
        fail("primary script wrapper exit record mismatch")
    if not receipt["started_monotonic_ns"] <= launch_receipt.get("started_monotonic_ns", 0) < pid_receipt.get("recorded_monotonic_ns", 0) < script_exit.get("ended_monotonic_ns", 0) or script_exit.get("started_monotonic_ns") != receipt["started_monotonic_ns"]:
        fail("primary script receipt chronology mismatch")
    if pid_receipt.get("pid") != launch_receipt.get("pid") or pid_receipt.get("argv") != expected_argv or not isinstance(pid_receipt.get("recorded_monotonic_ns"), int) or not receipt["started_monotonic_ns"] < pid_receipt["recorded_monotonic_ns"] < script_exit.get("ended_monotonic_ns", 0) or pid_receipt.get("raw_size", 0) <= 0 or hashlib.sha256(pid_prefix).hexdigest() != pid_receipt.get("raw_sha256"):
        fail("primary script PID receipt mismatch")
    wrapper_line = ("host_wrapper_waited=1 script_pid={} script_exit=0 raw_sha256={} raw_size={}".format(script_exit["pid"], script_exit["raw_sha256"], script_exit["raw_size"])).encode()
    host_lines = [line.rstrip(bytes((13, 10))) for line in host_payload.splitlines(keepends=True)]
    if host_lines.count(wrapper_line) != 1:
        fail("closed host wrapper wait event missing")
    if len(prefix_bytes) != precommit.get("prefix_size") or hashlib.sha256(prefix_bytes).hexdigest() != precommit.get("prefix_sha256") or not raw.startswith(prefix_bytes):
        fail("precommit prefix evidence mismatch")
    raw_prefix_size = check.get("raw_size", 0)
    if hashlib.sha256(raw[:raw_prefix_size]).hexdigest() != check.get("raw_sha256"):
        fail("postpush raw prefix evidence mismatch")
    closure_ends = [row for row in rows if row.get("kind") == "end" and row.get("exit_code") == 0 and row.get("command", "").startswith("python3 " + str(pathlib.Path(__file__).resolve()) + " --capture-postpush-closure")]
    if len(closure_ends) != 1:
        fail("closed transcript closure count mismatch")
    if b"card=00a preclose_log=1 captured_landing_records>=5 sealed_precommit_events=1 sealed_postpush_events=1 elapsed_seconds<=1800" not in raw:
        fail("closed transcript postseal output missing")
    if not copied == raw or hashlib.sha256(copied).hexdigest() != seal.get("raw_sha256") or seal.get("raw_size") != len(raw):
        fail("closed raw/full log byte or hash mismatch")
    if len(host_payload) != seal.get("host_log_size") or hashlib.sha256(host_payload).hexdigest() != seal.get("host_log_sha256"):
        fail("closed host wrapper log byte or hash mismatch")
    raw_hash = hashlib.sha256(raw).hexdigest()
    if postseal_record.get("closure_event_id") != closure_ends[0].get("event_id") or postseal_record.get("exit_code") != 0:
        fail("postseal record closure binding mismatch")
    preclose = full.with_name("full.log.preclose").read_bytes()
    if hashlib.sha256(preclose).hexdigest() != postseal_record.get("raw_sha256") or hashlib.sha256(preclose).hexdigest() != postseal_record.get("full_log_sha256") or not raw.startswith(preclose):
        fail("preclose transcript hash linkage mismatch")
    previous_finalize = args.finalize
    args.finalize = root
    derived = finalize(args, output=False, write_record=False)
    args.finalize = previous_finalize
    if derived.get("landing_event_ids") != check.get("landing_event_ids") or derived.get("head_event_ids") != check.get("head_event_ids") or derived.get("head_sha256") != check.get("head_sha256") or derived.get("transcript_prefix_sha256") != check.get("transcript_prefix_sha256") or derived.get("precommit_event_id") != check.get("precommit_event_id") or len(derived.get("landing_event_ids", [])) != 5 or derived.get("sealed_precommit_events") != 1:
        fail("postpush counts do not match recomputed ledger records")
    if "postpush_check=1 captured_landing_records=5 sealed_precommit_events=1 elapsed_seconds<=1800" not in finalizer.get("stdout", ""):
        fail("captured finalizer output fields mismatch")
    if finalizer.get("exit_code") != 0 or check.get("exit_code") != 0:
        fail("postpush finalizer outcome missing")
    if hashlib.sha256(finalizer.get("stdout", "").encode()).hexdigest() != finalizer.get("stdout_sha256") or hashlib.sha256(finalizer.get("stderr", "").encode()).hexdigest() != finalizer.get("stderr_sha256"):
        fail("finalizer stream hashes mismatch")
    elapsed = (time.monotonic_ns() - receipt["started_monotonic_ns"]) / 1_000_000_000
    if elapsed > 1800:
        fail("elapsed time exceeds cap")
    result = {"raw_size": len(raw), "full_log_size": len(copied), "raw_sha256": raw_hash,
              "full_log_sha256": hashlib.sha256(copied).hexdigest(), "elapsed_seconds": elapsed, "equal": True}
    (root / "closed-verified.json").write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    print("closed_raw_sealed=1 card=00a preclose_log=1 captured_landing_records=5 sealed_precommit_events=1 sealed_postpush_events=1 elapsed_seconds<=1800 host_wrapper_waited=1 script_exit=0 closed_raw_equals_full=1 host_log_hash_matches=1 precommit_prefix_matches=1 final_hash_matches=1 final_elapsed_seconds<=1800")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-entry", type=pathlib.Path)
    parser.add_argument("--precommit", type=pathlib.Path)
    parser.add_argument("--capture-postpush-closure", type=pathlib.Path)
    parser.add_argument("--finalize", type=pathlib.Path)
    parser.add_argument("--postseal", type=pathlib.Path)
    parser.add_argument("--record-script-pid", type=pathlib.Path)
    parser.add_argument("--script-pid", type=int)
    parser.add_argument("--seal-closed", type=pathlib.Path)
    parser.add_argument("--verify-closed", type=pathlib.Path)
    parser.add_argument("--card-id")
    parser.add_argument("--raw", type=pathlib.Path, default=DEFAULT_RAW)
    parser.add_argument("--sealed-transcript", type=pathlib.Path)
    parser.add_argument("--repo-root", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument("--finalizer-log", type=pathlib.Path)
    parser.add_argument("--finalize-argv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.verify_entry:
            verify_entry(args)
        elif args.precommit:
            precommit(args)
        elif args.capture_postpush_closure:
            if not args.sealed_transcript or not args.finalizer_log or not args.finalize_argv:
                fail("postpush closure arguments missing")
            capture_closure(args)
        elif args.finalize:
            finalize(args)
        elif args.postseal:
            if not args.sealed_transcript:
                fail("postseal transcript path missing")
            postseal(args)
        elif args.record_script_pid:
            record_script_pid(args)
        elif args.seal_closed:
            seal_closed(args)
        elif args.verify_closed:
            verify_closed(args)
        else:
            fail("one command mode is required")
    except (OSError, ValueError, KeyError) as error:
        print("seed_bootstrap.py: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
