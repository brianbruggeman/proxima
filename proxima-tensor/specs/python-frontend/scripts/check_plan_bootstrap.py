#!/usr/bin/env python3
"""Verify captured bootstrap receipts, shell events, and raw byte spans."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

SOURCE_ROOT = Path("/private/tmp/proxima-python-frontend/evidence/card-00a/session-attempt-10")
SOURCE_RAW = Path("/private/tmp/proxima-python-card-00a-attempt-10.log")
SOURCE_REPO = Path("/private/tmp/proxima-python-card-00a-attempt-10")
HOST_LOG = Path("/private/tmp/proxima-python-card-00a-attempt-10-host.log")
OUTPUT_AC1 = re.compile(rb"(?m)^hook_startup_records=1 receipt_hashes_valid=1 hook_events>=2 observed_hook_events=[0-9]+ setup_cd=0 first_repo_after_hook=1 clean_source_status=1 omitted_command_rejected=1 dirty_source_rejected=1\r?$")
OUTPUT_AC2 = b"card=00a functional=1 completed_functional_commands=1 pending_precommit_starts=1 prefix_snapshot=1 hashes_valid=1"


def fail(reason: str) -> None:
    raise ValueError(reason)


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail("malformed receipt: {}".format(error))


def exact_lines(payload: bytes, marker: bytes) -> list[int]:
    positions = []
    offset = 0
    for line in payload.splitlines(keepends=True):
        if line.rstrip(bytes((13, 10))) == marker:
            positions.append(offset)
        offset += len(line)
    return positions


def command_echo_window(payload: bytes, marker_position: int) -> bytes:
    previous_end = payload.rfind(b"PROXIMA_EVENT_END", 0, marker_position)
    previous_hook = payload.rfind(b"PROXIMA_HOOK_SOURCED", 0, marker_position)
    lower_bound = max(previous_end, previous_hook)
    window = payload[lower_bound:marker_position]
    window = re.sub(rb"\x1b\[[0-?]*[ -/]*[@-~]", b"", window)
    window = re.sub(rb".\x08", b"", window)
    window = window.replace(b"\r", b"")
    return re.sub(rb"([A-Za-z0-9]) ([A-Za-z0-9])\2", rb"\1\2", window)


def read_rows(root: Path) -> list[dict]:
    rows = []
    for number, line in enumerate((root / "bootstrap-events.jsonl").read_text(encoding="utf-8").splitlines(), 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            fail("malformed event row {}: {}".format(number, error))
    return rows


def validate(root: Path, raw_path: Path, repo_root: Path, receipt_override=None, rows_override=None, raw_override=None) -> int:
    receipt = load_json(root / "launcher-receipt.json") if receipt_override is None else receipt_override
    host_receipt = load_json(root / "host-bootstrap.json")
    if receipt != host_receipt:
        fail("receipt linkage mismatch")
    if receipt.get("card_id") != "00a" or receipt.get("attempt") != "10" or receipt.get("raw_transcript") != str(raw_path):
        fail("receipt identity mismatch")
    payload = raw_path.read_bytes() if raw_override is None else raw_override
    if raw_override is None and (root / "primary-script-exit.json").exists() and sha(payload) != read_json_field(root, "primary-script-exit.json", "raw_sha256"):
        fail("closed raw receipt mismatch")
    host_payload = Path(receipt["host_transcript"]).read_bytes()
    if not host_payload.startswith(b"\x1b") or b"ATTEMPT10_HOST_START" not in host_payload:
        fail("host transcript provenance missing")
    runner = Path(receipt["runner_path"])
    if sha(runner.read_bytes()) != receipt.get("runner_sha256"):
        fail("runner hash mismatch")
    hook = root / "bootstrap-hook.zsh"
    hook_bytes = hook.read_bytes()
    if sha(hook_bytes) != receipt.get("hook_sha256"):
        fail("hook hash mismatch")
    zdotdir = Path(receipt["zdotdir"])
    for name, expected in receipt["zdotdir_files"].items():
        target = zdotdir / name
        if not target.is_file() or sha(target.read_bytes()) != expected:
            fail("zdotdir hash mismatch")
    rows = read_rows(root) if rows_override is None else rows_override
    starts = [row for row in rows if row.get("kind") == "start"]
    ends = [row for row in rows if row.get("kind") == "end"]
    start_ids = [row.get("event_id") for row in starts]
    if len(start_ids) != len(set(start_ids)):
        fail("duplicate event start")
    orphan_starts = [start for start in starts if not any(row.get("event_id") == start.get("event_id") for row in ends)]
    if len(orphan_starts) > 1 or any(start.get("command") != "exit" for start in orphan_starts):
        fail("event terminal count mismatch")
    spans = 0
    for start in starts:
        terminals = [row for row in ends if row.get("event_id") == start.get("event_id")]
        if not terminals and start.get("command") == "exit":
            exit_marker = ("PROXIMA_EVENT_START ns=" + str(start["event_id"])).encode()
            if len(exact_lines(payload, exit_marker)) != 1:
                fail("exit event byte span mismatch")
            spans += 1
            continue
        if len(terminals) != 1:
            fail("event terminal count mismatch")
        terminal = terminals[0]
        command_parts = shlex.split(start.get("command", ""))
        expected_end_cwd = start.get("cwd")
        if command_parts and command_parts[0] == "cd":
            destination = Path(command_parts[1])
            expected_end_cwd = str(destination if destination.is_absolute() else Path(start["cwd"]) / destination)
        if terminal.get("command") != start.get("command") or terminal.get("cwd") != expected_end_cwd or terminal.get("started_monotonic_ns") != start.get("started_monotonic_ns") or terminal.get("ended_monotonic_ns", 0) < start.get("started_monotonic_ns", 1):
            fail("event terminal provenance mismatch")
        begin = ("PROXIMA_EVENT_START ns=" + str(start["event_id"])).encode()
        finish = ("PROXIMA_EVENT_END code={} ns=".format(terminal["exit_code"]) + str(start["event_id"])).encode()
        begin_positions = exact_lines(payload, begin)
        finish_positions = exact_lines(payload, finish)
        if len(begin_positions) != 1 or len(finish_positions) != 1 or begin_positions[0] >= finish_positions[0]:
            fail("event byte span mismatch")
        spans += 1
    hook_marker = b"PROXIMA_HOOK_SOURCED sha=" + receipt["hook_sha256"].encode()
    hook_positions = exact_lines(payload, hook_marker)
    if len(hook_positions) != 1:
        fail("hook marker mismatch")
    if b"check_plan_bootstrap.py" in payload[:hook_positions[0]]:
        fail("forbidden pre-PTY helper found")
    first = starts[0] if starts else {}
    source = "/private/tmp/proxima-python-plan-amend-00a-split"
    expected_status = "git -C " + source + " status --short"
    expected_fetch = "git -C " + source + " fetch origin main"
    expected_worktree = "git -C " + source + " worktree add -b codex/python-frontend-card-00a-final-10 /private/tmp/proxima-python-card-00a-attempt-10 origin/main"
    if first.get("command") != expected_status or first.get("cwd") != "/private/tmp":
        fail("first repository event mismatch")
    status_marker = ("PROXIMA_EVENT_START ns=" + first["event_id"]).encode()
    status_positions = exact_lines(payload, status_marker)
    if len(status_positions) != 1:
        fail("first status marker mismatch")
    status_echo = command_echo_window(payload, status_positions[0])
    if not status_echo.endswith(expected_status.encode() + b"\n"):
        fail("first command echo mismatch")
    commands = [row.get("command") for row in starts]
    if expected_fetch not in commands or expected_worktree not in commands:
        fail("bootstrap setup commands missing")
    if not hook_positions[0] < payload.find(("PROXIMA_EVENT_START ns=" + first["event_id"]).encode()):
        fail("hook precedes repository event check failed")
    ac1_rows = [row for row in starts if "--verify-entry" in row.get("command", "")]
    if len(ac1_rows) != 1:
        fail("AC00a.1 event count mismatch")
    ac1_start = ac1_rows[0]
    ac1_end = next(row for row in ends if row.get("event_id") == ac1_start["event_id"])
    if ac1_end.get("exit_code") != 0:
        fail("AC00a.1 failed")
    ac1_begin = ("PROXIMA_EVENT_START ns=" + ac1_start["event_id"]).encode()
    ac1_finish = ("PROXIMA_EVENT_END code=0 ns=" + ac1_start["event_id"]).encode()
    left = exact_lines(payload, ac1_begin)
    right = exact_lines(payload, ac1_finish)
    if len(left) != 1 or len(right) != 1:
        fail("AC00a.1 byte span mismatch")
    if not OUTPUT_AC1.search(payload[left[0] + len(ac1_begin):right[0]]):
        fail("AC00a.1 output mismatch")
    return spans


def read_json_field(root: Path, filename: str, key: str):
    return load_json(root / filename).get(key)


def copy_active_source(root: Path) -> None:
    source = Path(__file__).resolve()
    record = {"source_path": str(source), "source_sha256": sha(source.read_bytes())}
    (root / "active-verifier.json").write_text(json.dumps(record, sort_keys=True) + chr(10), encoding="utf-8")


def self_check(args: argparse.Namespace) -> None:
    root = args.source
    spans = validate(root, SOURCE_RAW, SOURCE_REPO)
    receipt = load_json(root / "launcher-receipt.json")
    rows = read_rows(root)
    raw = SOURCE_RAW.read_bytes()
    bad_receipt = dict(receipt)
    bad_receipt["hook_sha256"] = "0" * 64
    malformed_receipt_rejected = 0
    try:
        validate(root, SOURCE_RAW, SOURCE_REPO, receipt_override=bad_receipt)
    except ValueError as error:
        malformed_receipt_rejected = int(str(error) == "receipt linkage mismatch")
    bad_rows = [dict(row) for row in rows]
    bad_rows[0]["cwd"] = "/wrong"
    malformed_events_rejected = 0
    try:
        validate(root, SOURCE_RAW, SOURCE_REPO, rows_override=bad_rows)
    except ValueError as error:
        malformed_events_rejected = int(str(error) == "event terminal provenance mismatch")
    hook_position = exact_lines(raw, b"PROXIMA_HOOK_SOURCED sha=" + receipt["hook_sha256"].encode())[0]
    hostile = b"python3 check_plan_bootstrap.py --launch" + bytes((10,)) + raw
    forbidden_prepty_helpers_rejected = 0
    try:
        validate(root, SOURCE_RAW, SOURCE_REPO, raw_override=hostile)
    except ValueError as error:
        forbidden_prepty_helpers_rejected = int(str(error) == "forbidden pre-PTY helper found")
    wrong_reason = int(malformed_receipt_rejected != 1 or malformed_events_rejected != 1 or forbidden_prepty_helpers_rejected != 1)
    if wrong_reason:
        fail("self-check rejection reason mismatch")
    status_start = next(row for row in rows if row.get("kind") == "start")
    status_marker = ("PROXIMA_EVENT_START ns=" + status_start["event_id"]).encode()
    status_position = exact_lines(raw, status_marker)[0]
    status_echo_start = raw.rfind(b"git -C ", 0, status_position)
    status_echo_end = raw.find(b"status --short", status_echo_start, status_position)
    echo_mutation_rejected = 0
    if status_echo_start < 0 or status_echo_end < 0:
        fail("first command echo fixture missing")
    mutated = raw[:status_echo_end] + b"status --short --evil" + raw[status_echo_end + len(b"status --short"):]
    try:
        validate(root, SOURCE_RAW, SOURCE_REPO, raw_override=mutated)
    except ValueError as error:
        echo_mutation_rejected = int(str(error) == "first command echo mismatch")
    if echo_mutation_rejected != 1:
        fail("command echo mutation control failed")
    active_root = args.self_check_host_bootstrap.parent / "session"
    active_root.mkdir(parents=True, exist_ok=True)
    copy_active_source(active_root)
    source_path = Path(__file__).resolve()
    active = load_json(active_root / "active-verifier.json")
    output = "valid_segments=1 event_marker_pairs={} first_command_echo=1 malformed_receipts_rejected=1 malformed_events_rejected=1 forbidden_prepty_helpers_rejected=1 echo_mutation_rejected=1 wrong_reason_acceptances=0".format(spans)
    functional = {"output": output, "source_sha256": active["source_sha256"], "source_raw_sha256": sha(SOURCE_RAW.read_bytes()), "source_receipt_sha256": sha((root / "launcher-receipt.json").read_bytes())}
    (active_root / "functional-check.json").write_text(json.dumps(functional, sort_keys=True) + chr(10), encoding="utf-8")
    print(output)


def precommit(args: argparse.Namespace) -> None:
    root = args.precommit
    if args.card_id != "00a1":
        fail("card identity mismatch")
    validate(SOURCE_ROOT, SOURCE_RAW, SOURCE_REPO)
    active = load_json(root / "active-verifier.json")
    source_path = Path(active["source_path"])
    if source_path != Path(__file__).resolve() or sha(source_path.read_bytes()) != active.get("source_sha256"):
        fail("active source hash mismatch")
    functional = load_json(root / "functional-check.json")
    source_receipt_hash = sha((SOURCE_ROOT / "launcher-receipt.json").read_bytes())
    if functional.get("source_sha256") != active.get("source_sha256") or functional.get("source_raw_sha256") != sha(SOURCE_RAW.read_bytes()) or functional.get("source_receipt_sha256") != source_receipt_hash:
        fail("functional source hash link mismatch")
    raw = args.raw.read_bytes()
    output = functional.get("output", "").encode()
    if not output or output not in raw:
        fail("functional output missing from current transcript")
    control_fields = (b"malformed_receipts_rejected=1", b"malformed_events_rejected=1", b"forbidden_prepty_helpers_rejected=1", b"echo_mutation_rejected=1")
    if any(field not in output for field in control_fields):
        fail("functional controls incomplete")
    prefix = raw
    (root / "transcript-prefix.log").write_bytes(prefix)
    record = {"prefix_size": len(prefix), "prefix_sha256": sha(prefix), "functional_source_sha256": active["source_sha256"], "functional_raw_sha256": functional["source_raw_sha256"], "functional_receipt_sha256": source_receipt_hash, "functional_output_sha256": sha(output), "control_count": len(control_fields)}
    (root / "precommit-check.json").write_text(json.dumps(record, sort_keys=True) + chr(10), encoding="utf-8")
    print("card=00a1 functional=1 controls={} active_hash_links=1 hashes_valid=1".format(len(control_fields)))


def run_closure(args: argparse.Namespace) -> None:
    root = args.run_closure
    seed_script = args.seed_script.resolve()
    primary_raw = args.raw.read_bytes()
    precommit = load_json(root / "precommit-check.json")
    prefix = (root / "transcript-prefix.log").read_bytes()
    if len(prefix) != precommit.get("prefix_size") or sha(prefix) != precommit.get("prefix_sha256"):
        fail("closed primary precommit prefix receipt mismatch")
    if not primary_raw.startswith(prefix) or len(prefix) >= len(primary_raw):
        fail("closed primary raw does not extend precommit prefix")
    if load_json(root / "functional-check.json").get("output", "").encode() not in primary_raw:
        fail("closed primary raw lacks functional output")
    child_raw = root / "closure-segment.log"
    if child_raw.exists():
        fail("closure segment already exists")
    full_log = root / "full.log"
    finalizer_log = root / "finalizer.json"
    seed_command = [sys.executable, str(seed_script), "--capture-postpush-closure", str(root), "--card-id", "00a1",
                    "--raw", str(child_raw), "--precommit-raw", str(args.raw), "--sealed-transcript", str(full_log), "--repo-root", str(args.repo_root),
                    "--finalizer-log", str(finalizer_log)]
    nested_finalize = [sys.executable, str(seed_script), "--finalize", str(root), "--card-id", "00a1",
                       "--raw", str(child_raw), "--precommit-raw", str(args.raw), "--sealed-transcript", str(full_log), "--repo-root", str(args.repo_root)]
    seed_command.extend(["--finalize-argv", shlex.join(nested_finalize)])
    script_argv = ["/usr/bin/script", "-q", "-a", str(child_raw), "/bin/zsh", "-f"]
    started = time.monotonic_ns()
    process = subprocess.Popen(script_argv, cwd=args.repo_root, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    primary_pid = process.pid
    command_text = shlex.join(seed_command) + "\nexit\n"
    stdout, stderr = process.communicate(command_text.encode(), timeout=300)
    ended = time.monotonic_ns()
    raw_bytes = child_raw.read_bytes()
    exit_record = {"pid": primary_pid, "argv": script_argv, "cwd": str(args.repo_root), "started_monotonic_ns": started,
                   "ended_monotonic_ns": ended, "exit_code": process.returncode, "raw_size": len(raw_bytes),
                   "raw_sha256": sha(raw_bytes), "stdout_size": len(stdout), "stderr_size": len(stderr),
                   "stdout_sha256": sha(stdout), "stderr_sha256": sha(stderr)}
    (root / "closure-runner-receipt.json").write_text(json.dumps(exit_record, sort_keys=True) + chr(10), encoding="utf-8")
    if process.returncode != 0:
        fail("closure child script exited nonzero")
    expected_child_line = b"postpush_check=1 captured_landing_records>=5 elapsed_seconds<=1800"
    if expected_child_line not in raw_bytes:
        fail("closure child output missing from closed raw")
    if stdout and stdout not in raw_bytes:
        fail("closure child stdout missing from raw")
    full_log.write_bytes(primary_raw)
    if full_log.read_bytes() != primary_raw:
        fail("closed primary raw and full log mismatch")
    postpush = load_json(root / "postpush-check.json")
    finalizer = load_json(finalizer_log)
    if postpush.get("exit_code") != 0 or len(postpush.get("landing_operations", [])) != 5 or finalizer.get("exit_code") != 0:
        fail("postpush terminal artifacts invalid")
    timer_start = int((root / "timer-start-ns.txt").read_text(encoding="utf-8").strip())
    elapsed = (time.monotonic_ns() - timer_start) / 1_000_000_000
    if elapsed > 1800 or ended < started:
        fail("closure elapsed or process order invalid")
    receipt = dict(exit_record)
    receipt["finalizer_sha256"] = sha(finalizer_log.read_bytes())
    receipt["postpush_sha256"] = sha((root / "postpush-check.json").read_bytes())
    receipt["full_log_sha256"] = sha(full_log.read_bytes())
    receipt["primary_raw_sha256"] = sha(primary_raw)
    receipt["primary_raw_size"] = len(primary_raw)
    receipt["closure_segment"] = str(child_raw)
    receipt["closure_segment_sha256"] = sha(raw_bytes)
    receipt["closure_segment_size"] = len(raw_bytes)
    receipt["elapsed_seconds"] = elapsed
    (root / "closed-reader.json").write_text(json.dumps(receipt, sort_keys=True) + chr(10), encoding="utf-8")
    sys.stdout.write(stdout.decode(errors="replace"))
    if stderr:
        sys.stderr.write(stderr.decode(errors="replace"))
    print("card=00a1 primary_raw_equals_full=1 closure_segment_sealed=1 captured_landing_records=5 elapsed_seconds<=1800")

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check-host-bootstrap", type=Path)
    parser.add_argument("--source", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--precommit", type=Path)
    parser.add_argument("--card-id")
    parser.add_argument("--raw", type=Path, default=SOURCE_RAW)
    parser.add_argument("--repo-root", type=Path, default=SOURCE_REPO)
    parser.add_argument("--run-closure", type=Path)
    parser.add_argument("--seed-script", type=Path, default=Path("proxima-tensor/specs/python-frontend/scripts/seed_bootstrap.py"))
    args = parser.parse_args()
    try:
        if args.self_check_host_bootstrap:
            self_check(args)
        elif args.precommit:
            precommit(args)
        elif args.run_closure:
            run_closure(args)
        else:
            fail("one mode is required")
    except (OSError, ValueError, KeyError, StopIteration) as error:
        print("check_plan_bootstrap.py: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
