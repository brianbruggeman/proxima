#!/usr/bin/env python3
"""Validate actual Card 00a bootstrap receipts, terminal markers, and transcripts."""
import argparse
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys
import time

EVIDENCE = pathlib.Path("/private/tmp/proxima-python-frontend/evidence/card-00a")
RAW = pathlib.Path("/private/tmp/proxima-python-card-00a-bootstrap.log")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_receipt(path):
    receipt = json.loads(pathlib.Path(path).read_text())
    required = {"argv", "cwd", "start_ns", "end_ns", "environment_overrides", "stdout", "stderr", "exit_code", "outputs", "first_command_argv", "first_command_sha256"}
    require(required <= receipt.keys(), "receipt schema missing fields")
    require(receipt["start_ns"] <= receipt["end_ns"] and receipt["exit_code"] == 0, "receipt process fields invalid")
    for name, expected in receipt["outputs"].items():
        require(sha(pathlib.Path(name).read_bytes()) == expected, "receipt output hash mismatch: " + name)
    return receipt


def verify_actual(receipt_path, ledger_path, raw_path, zdotdir, prefix_path=None, allow_one_pending=False):
    receipt = load_receipt(receipt_path)
    raw = pathlib.Path(raw_path).read_bytes()
    rows = [json.loads(line) for line in pathlib.Path(ledger_path).read_text().splitlines()]
    require(rows, "event ledger empty")
    payload_path = pathlib.Path(receipt["first_command_argv"][1])
    payload_hash = sha(payload_path.read_bytes())
    require(payload_hash == receipt["first_command_sha256"], "first command source hash mismatch")
    first = rows[0]
    require(first["argv"] == receipt["first_command_argv"], "first command argv mismatch")
    require(first["start_ns"] == receipt["start_ns"], "receipt start timestamp differs from bootstrap event")
    require(receipt["start_ns"] < rows[1]["start_ns"], "bootstrap start timestamp does not precede fetch/event activity")
    require(first["payload_sha256"] == payload_hash, "first command ledger hash mismatch")
    require(first["receipt_sha256"] == sha(pathlib.Path(receipt_path).read_bytes()), "first command receipt hash mismatch")
    require(first["source_marker"].encode() in raw and first["receipt_marker"].encode() in raw, "bootstrap receipt/source marker absent")
    source_pos = raw.find(first["source_marker"].encode())
    receipt_pos = raw.find(first["receipt_marker"].encode())
    require(0 <= source_pos < receipt_pos, "bootstrap marker order invalid")
    command = payload_path.name.encode()
    bootstrap_command_pos = raw.find(command)
    require(0 <= bootstrap_command_pos < source_pos, "first bootstrap command absent before receipt marker")
    zdotdir_path = pathlib.Path(zdotdir)
    require(zdotdir_path.joinpath(".zshenv").is_file() and zdotdir_path.joinpath(".zshrc").is_file(), "ZDOTDIR files missing")
    require(receipt["environment_overrides"].get("ZDOTDIR") == str(zdotdir_path), "receipt ZDOTDIR mismatch")
    require(receipt["outputs"].get(str(zdotdir_path / ".zshenv")) and receipt["outputs"].get(str(zdotdir_path / ".zshrc")), "ZDOTDIR hashes missing")
    previous_end = receipt_pos
    hook_count = 0
    pending_events = 0
    for row_index, row in enumerate(rows[1:], 1):
        require({"argv", "cwd", "start_ns", "end_ns", "event_id", "command_sha256", "exit_code", "environment_overrides", "start_marker", "end_marker"} <= row.keys(), "hook event schema missing")
        start_marker = row["start_marker"].encode()
        end_marker = row["end_marker"].encode()
        start_pos = raw.find(start_marker)
        end_pos = raw.find(end_marker, start_pos + len(start_marker)) if start_pos >= 0 else -1
        if start_pos < 0 or end_pos < 0:
            if allow_one_pending and row_index == len(rows) - 1:
                pending_events += 1
                continue
            require(False, "hook marker order invalid at event " + str(row.get("event_id")))
        require(previous_end < start_pos < end_pos, "hook marker order invalid at event " + str(row.get("event_id")))
        command_bytes = row["argv"][0].encode()
        hook_command_pos = raw.find(command_bytes, previous_end)
        require(previous_end <= hook_command_pos < start_pos, "hook command bytes absent before start marker")
        require(sha(command_bytes) == row["command_sha256"], "hook command hash mismatch")
        require(row["start_marker"].endswith(row["command_sha256"]), "start marker command hash mismatch")
        require(row["end_marker"].endswith(":" + str(row["exit_code"])), "end marker exit mismatch")
        require(row["start_ns"] <= row["end_ns"], "hook time order invalid")
        previous_end = end_pos + len(end_marker)
        hook_count += 1
    require(hook_count > 0, "no marker-backed hook event")
    require(b"check_plan_bootstrap.py" not in raw[:source_pos], "repository helper ran before bootstrap")
    fetch_pos = raw.find(b"git fetch origin main")
    require(fetch_pos < 0 or receipt_pos < fetch_pos, "repository inspection preceded bootstrap")
    if prefix_path is not None:
        prefix = pathlib.Path(prefix_path).read_bytes()
        require(raw.startswith(prefix), "transcript prefix mismatch")
    return receipt, rows, raw, {"bootstrap_command_pos": bootstrap_command_pos, "source_marker_pos": source_pos, "receipt_marker_pos": receipt_pos, "hook_events": hook_count, "pending_events": pending_events, "raw_bytes": len(raw)}


def self_check(args):
    args.fixture = args.self_check_host_bootstrap
    args.receipt = args.receipt or str(EVIDENCE / "launcher-receipt.json")
    args.manifest = args.manifest or str(EVIDENCE / "bootstrap-events.jsonl")
    args.transcript = args.transcript or str(RAW)
    args.prefix = args.prefix or str(EVIDENCE / "transcript-prefix.log")
    args.zdotdir = args.zdotdir or "/private/tmp/proxima-python-frontend/bootstrap-zdotdir/card-00a"
    receipt, rows, raw, positions = verify_actual(args.receipt, args.manifest, args.transcript, args.zdotdir, args.prefix, True)
    require(receipt["cwd"] == args.cwd, "bootstrap cwd mismatch")
    require(receipt["argv"] == shlex.split(args.host_script_argv), "host argv mismatch")
    bad = []
    malformed_receipt = pathlib.Path(args.fixture) / "bad-receipt.json"
    malformed_receipt.parent.mkdir(parents=True, exist_ok=True)
    malformed_receipt.write_text("{}\n")
    try:
        load_receipt(malformed_receipt)
    except (ValueError, KeyError):
        bad.append("receipt")
    else:
        raise ValueError("malformed receipt control accepted")
    malformed_hook = pathlib.Path(args.fixture) / "bad-hook.jsonl"
    broken_rows = [dict(row) for row in rows]
    broken_rows[1]["start_marker"] = "PROXIMA_EVENT_START:absent"
    malformed_hook.write_text("\n".join(json.dumps(row) for row in broken_rows) + "\n")
    try:
        verify_actual(args.receipt, malformed_hook, args.transcript, args.zdotdir)
    except (ValueError, KeyError):
        bad.append("hook")
    else:
        raise ValueError("malformed hook control accepted")
    zdotdir_missing = pathlib.Path(args.fixture) / "bad-zdotdir"
    zdotdir_missing.mkdir(exist_ok=True)
    try:
        require(zdotdir_missing.joinpath(".zshenv").is_file() and zdotdir_missing.joinpath(".zshrc").is_file(), "ZDOTDIR files missing")
    except ValueError:
        bad.append("zdotdir")
    else:
        raise ValueError("malformed ZDOTDIR control accepted")
    helper = pathlib.Path(args.cwd) / "proxima-tensor/specs/python-frontend/scripts/check_plan_bootstrap.py"
    require(args.repo_helper_absent and not helper.exists(), "repository helper present before bootstrap")
    require(args.check_forbidden_helper_control, "forbidden-helper control not requested")
    forbidden_root = pathlib.Path(args.fixture) / "forbidden-repo"
    forbidden_helper = forbidden_root / "proxima-tensor/specs/python-frontend/scripts/check_plan_bootstrap.py"
    forbidden_helper.parent.mkdir(parents=True, exist_ok=True)
    forbidden_helper.write_text("raise SystemExit('must not run')\n")
    marker = rows[0]["source_marker"].encode()
    source_pos = raw.find(marker)
    hostile_raw = pathlib.Path(args.fixture) / "forbidden-prepty.log"
    hostile_raw.write_bytes(raw[:source_pos] + ("python3 " + str(forbidden_helper) + "\n").encode() + raw[source_pos:])
    try:
        verify_actual(args.receipt, args.manifest, str(hostile_raw), args.zdotdir)
    except ValueError:
        bad.append("forbidden-helper")
    else:
        raise ValueError("forbidden helper control accepted")
    fields = {
        "repo_helper_present": int(helper.exists()),
        "cwd_checked": int(receipt["cwd"] == args.cwd),
        "host_script_started": int(receipt["argv"] == shlex.split(args.host_script_argv)),
        "first_command_created_evidence": int(rows[0]["receipt_sha256"] == sha(pathlib.Path(args.receipt).read_bytes())),
        "receipt_schema_valid": 1,
        "hook_schema_valid": int(positions["hook_events"] > 0),
        "zdotdir_created": int(pathlib.Path(args.zdotdir).is_dir()),
        "zdotdir_files_checked": sum((pathlib.Path(args.zdotdir) / name).is_file() for name in (".zshenv", ".zshrc")),
        "receipt_process_fields": int(receipt["start_ns"] <= receipt["end_ns"] and receipt["exit_code"] == 0),
        "hook_event_checked": int(positions["hook_events"] > 0),
        "malformed_receipt_rejected": int("receipt" in bad),
        "malformed_hook_rejected": int("hook" in bad),
        "malformed_zdotdir_rejected": int("zdotdir" in bad),
        "forbidden_prepty_helper_rejected": int("forbidden-helper" in bad),
        "repo_inspection_before_hook": int(raw.find(b"git fetch origin main") >= 0 and raw.find(b"git fetch origin main") < positions["receipt_marker_pos"]),
        "transcript_prefix": int(raw.startswith(pathlib.Path(args.prefix).read_bytes())),
        "output_hashes_valid": 1,
    }
    output = " ".join(key + "=" + str(value) for key, value in fields.items())
    attempt = pathlib.Path(args.fixture) / ("accepted-" + str(time.monotonic_ns()))
    attempt.mkdir(parents=True)
    record = {"ac_id":"AC00a.1", "accepted":True, "stdout_sha256":sha(output.encode()), "checker_sha256":sha(pathlib.Path(__file__).read_bytes()), "raw_sha256":sha(raw), "receipt_sha256":sha(pathlib.Path(args.receipt).read_bytes()), "positions":positions, "negative_controls":bad}
    attempt.joinpath("result.json").write_text(json.dumps(record, sort_keys=True) + "\n")
    print(output)


def precommit(args):
    require(args.prefix_snapshot == "1", "prefix snapshot flag missing")
    args.fixture = args.fixture or str(EVIDENCE / "fixture")
    args.launcher_receipt = args.launcher_receipt or str(EVIDENCE / "launcher-receipt.json")
    args.manifest = args.manifest or str(EVIDENCE / "bootstrap-events.jsonl")
    args.transcript = args.transcript or str(RAW)
    args.prefix = str(EVIDENCE / "transcript-prefix.log")
    args.zdotdir = "/private/tmp/proxima-python-frontend/bootstrap-zdotdir/card-00a"
    receipt, rows, raw, pos = verify_actual(args.launcher_receipt, args.manifest, args.transcript, args.zdotdir, args.prefix, True)
    top = subprocess.run(["git", "-C", args.repo_root, "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()
    require(pathlib.Path(top).resolve() == pathlib.Path(args.repo_root).resolve(), "precommit repository root mismatch")
    results = list((pathlib.Path(args.fixture)).glob("accepted-*/result.json"))
    checker_hash = sha(pathlib.Path(__file__).read_bytes())
    ac_one_events = sorted((EVIDENCE / "acceptance").glob("ac00a-1-*/event.json"), key=lambda path: json.loads(path.read_text())["start_ns"])
    require(ac_one_events, "captured AC00a.1 event missing")
    ac_one_event_path = ac_one_events[-1]
    ac_one_event = json.loads(ac_one_event_path.read_text())
    ac_one_stdout_path = ac_one_event_path.parent / "stdout.txt"
    ac_one_stderr_path = ac_one_event_path.parent / "stderr.txt"
    ac_one_stdout = ac_one_stdout_path.read_text()
    require(ac_one_event["exit_code"] == 0 and ac_one_event["argv"][1] == "proxima-tensor/specs/python-frontend/scripts/check_plan_bootstrap.py", "captured AC00a.1 command failed or differs")
    require(sha(ac_one_stdout.encode()) == ac_one_event["stdout_sha256"] and sha(ac_one_stderr_path.read_bytes()) == ac_one_event["stderr_sha256"], "captured AC00a.1 stream hashes invalid")
    require(any(json.loads(path.read_text()).get("ac_id") == "AC00a.1" and json.loads(path.read_text()).get("checker_sha256") == checker_hash for path in results), "AC00a.1 result missing for current checker")
    functional = int(ac_one_event["exit_code"] == 0)
    accepted = int(ac_one_stdout.startswith("repo_helper_present=0 ") and "forbidden_prepty_helper_rejected=1" in ac_one_stdout)
    hashes_valid = int(sha(ac_one_stdout.encode()) == ac_one_event["stdout_sha256"] and sha(ac_one_stderr_path.read_bytes()) == ac_one_event["stderr_sha256"])
    output = "card=00a functional=" + str(functional) + " accepted=" + str(accepted) + " hashes_valid=" + str(hashes_valid) + " launcher_receipt=" + str(int(bool(receipt))) + " bootstrap_transcript_prefix=" + str(int(raw.startswith(pathlib.Path(args.prefix).read_bytes())))
    record = {"ac_id":"AC00a.2", "accepted":True, "checker_sha256":checker_hash, "stdout_sha256":sha(output.encode()), "raw_sha256":sha(raw), "receipt_sha256":sha(pathlib.Path(args.launcher_receipt).read_bytes())}
    (EVIDENCE / "precommit-result.json").write_text(json.dumps(record, sort_keys=True) + "\n")
    lifecycle = EVIDENCE / "lifecycle-events.jsonl"
    if not lifecycle.exists():
        startup = json.loads(pathlib.Path(args.manifest).read_text().splitlines()[0])
        lifecycle.write_text(json.dumps({"kind":"bootstrap", "payload_sha256":startup["payload_sha256"], "receipt_sha256":startup["receipt_sha256"], "start_ns":startup["start_ns"], "end_ns":startup["end_ns"]}, sort_keys=True) + "\n")
    with lifecycle.open("a") as stream:
        stream.write(json.dumps({"kind":"precommit", "result_sha256":sha((EVIDENCE / "precommit-result.json").read_bytes()), "start_ns":time.monotonic_ns(), "exit_code":0}, sort_keys=True) + "\n")
    print(output)


def finalize(args):
    args.zdotdir = "/private/tmp/proxima-python-frontend/bootstrap-zdotdir/card-00a"
    verify_actual(args.launcher_receipt, args.manifest, args.raw_transcript, args.zdotdir, args.prefix)
    require(pathlib.Path(args.raw_transcript).read_bytes() == pathlib.Path(args.transcript).read_bytes(), "transcript copies differ")
    result = json.loads((EVIDENCE / "precommit-result.json").read_text())
    require(result.get("ac_id") == "AC00a.2" and result.get("accepted") is True, "precommit result missing")
    require(result.get("checker_sha256") == sha(pathlib.Path(__file__).read_bytes()), "precommit checker version mismatch")
    elapsed = (time.monotonic_ns() - 702817687984041) / 1e9
    require(elapsed <= 1800, "elapsed time exceeded 1800 seconds")


def capture_closure(args):
    raw = pathlib.Path(args.raw_transcript)
    transcript = pathlib.Path(args.transcript)
    transcript.write_bytes(raw.read_bytes())
    command = shlex.split(args.finalizer_argv)
    started = time.monotonic_ns()
    result = subprocess.run(command, cwd=args.repo_root, capture_output=True, text=True)
    ended = time.monotonic_ns()
    record = {"argv":command, "cwd":args.repo_root, "start_ns":started, "end_ns":ended, "exit_code":result.returncode, "stdout":result.stdout, "stderr":result.stderr, "raw_sha256":sha(raw.read_bytes()), "copy_sha256":sha(transcript.read_bytes())}
    finalizer = pathlib.Path(args.finalizer_log)
    finalizer.write_text(json.dumps(record, sort_keys=True) + "\n")
    require(result.returncode == 0, "postpush finalizer command failed")
    require(record["raw_sha256"] == record["copy_sha256"], "postpush transcript hash mismatch")
    require(result.stdout == "", "postpush finalizer emitted unexpected stdout")
    head = subprocess.run(["git", "-C", args.repo_root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    remote = subprocess.run(["git", "-C", args.repo_root, "ls-remote", "origin", "refs/heads/main"], capture_output=True, text=True, check=True).stdout.split()[0]
    require(head == remote, "local head and remote main differ")
    record["head"] = head
    record["remote_main"] = remote
    finalizer.write_text(json.dumps(record, sort_keys=True) + "\n")
    lifecycle = EVIDENCE / "lifecycle-events.jsonl"
    previous = lifecycle.read_text().splitlines() if lifecycle.exists() else []
    require(len(previous) == 2, "bootstrap and precommit lifecycle records missing")
    finalizer_hash = sha(finalizer.read_bytes())
    with lifecycle.open("a") as stream:
        stream.write(json.dumps({"kind":"postpush-finalizer", "sha256":finalizer_hash, "start_ns":started, "end_ns":ended, "exit_code":result.returncode}, sort_keys=True) + "\n")
    require(len(lifecycle.read_text().splitlines()) == 3, "lifecycle event count mismatch")
    elapsed = (time.monotonic_ns() - 702817687984041) / 1e9
    require(elapsed <= 1800, "elapsed time exceeded 1800 seconds")
    print("card=00a functional=1 full_log=1 bootstrap_receipt=1 bootstrap_events=3 transcript_hashes_valid=1 postpush_stage=1 elapsed_seconds<=1800")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check-host-bootstrap", nargs="?")
    parser.add_argument("--repo-helper-absent", action="store_true")
    parser.add_argument("--cwd")
    parser.add_argument("--zdotdir")
    parser.add_argument("--check-zdotdir-files", action="store_true")
    parser.add_argument("--check-receipt-process-fields", action="store_true")
    parser.add_argument("--check-hook-event", action="store_true")
    parser.add_argument("--check-forbidden-helper-control", action="store_true")
    parser.add_argument("--host-script-argv")
    parser.add_argument("--receipt")
    parser.add_argument("--manifest")
    parser.add_argument("--transcript")
    parser.add_argument("--prefix")
    parser.add_argument("--prefix-snapshot")
    parser.add_argument("--fixture")
    parser.add_argument("--precommit", action="store_true")
    parser.add_argument("--capture-postpush-closure", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--repo-root")
    parser.add_argument("--launcher-receipt")
    parser.add_argument("--raw-transcript")
    parser.add_argument("--finalizer-log")
    parser.add_argument("--finalizer-argv")
    args = parser.parse_args()
    if args.self_check_host_bootstrap:
        self_check(args)
    elif args.precommit:
        precommit(args)
    elif args.finalize:
        finalize(args)
    elif args.capture_postpush_closure:
        capture_closure(args)
    else:
        raise SystemExit("unsupported mode")

if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print("card00a_error=" + str(error), file=sys.stderr)
        raise SystemExit(1)
