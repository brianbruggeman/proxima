#!/usr/bin/env python3
"""Capture complete command evidence outside the source checkout."""
from __future__ import annotations
import argparse, base64, hashlib, json, os, subprocess, sys, time, signal, fcntl, tempfile
from datetime import datetime, timezone
from pathlib import Path
def utc_now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z")
def sha(data): return hashlib.sha256(data).hexdigest()
def inside(path,parent):
    try: path.resolve().relative_to(parent.resolve()); return True
    except ValueError: return False
def repo_root(path):
    for candidate in (path,*path.parents):
        if (candidate/".git").exists(): return candidate.resolve()
    raise RuntimeError("current directory is not inside a git worktree")
def containing_worktree(path):
    for candidate in (path.resolve(), *path.resolve().parents):
        if (candidate / ".git").exists():
            return candidate.resolve()
    return None
def atomic_write(path, data):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(directory_fd)
    finally: os.close(directory_fd)

def refresh_active_segment(root):
    chain_path = root.resolve().parent / "segments.jsonl"
    if not chain_path.is_file():
        return
    from check_evidence import segment_inventory
    rows = [json.loads(line) for line in chain_path.read_text(encoding="utf-8").splitlines()]
    position = next((index for index, row in enumerate(rows) if row.get("root") == root.name and row.get("role") == "active"), None)
    if position is None:
        return
    previous = rows[position - 1]["row_sha256"] if position else None
    old = rows[position]
    row = segment_inventory(root.resolve(), position + 1, "active", previous, old.get("duplicate_of"), old.get("snapshot_of"), old.get("prefix_events"))
    row["row_sha256"] = sha(json.dumps({key: value for key, value in row.items() if key != "row_sha256"}, sort_keys=True, separators=(",", ":")).encode())
    rows[position] = row
    for index in range(position + 1, len(rows)):
        prior = rows[index - 1]["row_sha256"]
        rows[index]["previous_row_sha256"] = prior
        rows[index]["row_sha256"] = sha(json.dumps({key: value for key, value in rows[index].items() if key != "row_sha256"}, sort_keys=True, separators=(",", ":")).encode())
    atomic_write(chain_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode())

def _capture_one_locked(root,argv,capture_argv=None,relay=True):
    root=root.expanduser().resolve(); repository=repo_root(Path.cwd())
    if inside(root,repository): raise SystemExit("evidence root must be outside repository")
    inherited=os.environ.get("PROXIMA_CASSETTE_DIR")
    if inherited and inside(Path(inherited),repository): raise SystemExit("inherited cassette root is inside repository")
    root.mkdir(parents=True,exist_ok=True)
    executable_names = [Path(item).name for item in argv]
    if "check_evidence.py" in executable_names and any(flag in argv for flag in ("--precommit","--finalize","--finalize-plan")):
        for inventory_argv in (
            ["git","status","--short","--ignored","--untracked-files=all"],
            ["git","diff","--name-only","origin/main...HEAD"],
            ["git","ls-tree","-r","--name-only","HEAD"],
            ["git","diff-tree","--no-commit-id","--name-only","-r","HEAD"],
        ):
            _capture_one_locked(root, inventory_argv, ["capture_validation.py","terminal-inventory"], relay=False)
    for name in ("commands","target","uv-cache","uv-environment","python-cache","tmp","xdg-cache","cassettes"):
        output_path = root / name
        output_path.mkdir(parents=True,exist_ok=True)
        if output_path.is_symlink() or not output_path.is_dir() or containing_worktree(output_path.resolve()):
            raise SystemExit(f"evidence output directory is invalid or inside a repository worktree: {output_path}")
    number=len(list((root/"commands").glob("[0-9][0-9][0-9][0-9][0-9][0-9]")))+1; command_dir=root/"commands"/f"{number:06d}"; command_dir.mkdir()
    start_path=command_dir/"start.json"; stdout_path=command_dir/"stdout.bin"; stderr_path=command_dir/"stderr.bin"; exit_path=command_dir/"exit.json"
    env=os.environ.copy(); overrides={"CARGO_TARGET_DIR":str(root/"target"),"UV_CACHE_DIR":str(root/"uv-cache"),"UV_PROJECT_ENVIRONMENT":str(root/"uv-environment"),"PYTHONPYCACHEPREFIX":str(root/"python-cache"),"TMPDIR":str(root/"tmp"),"XDG_CACHE_HOME":str(root/"xdg-cache"),"PROXIMA_CASSETTE_DIR":str(root/"cassettes")}; env.update(overrides)
    started=utc_now(); started_ns=time.time_ns(); start={"event_id":number,"argv":argv,"capture_argv":capture_argv or [],"cwd":str(Path.cwd()),"started_at":started,"started_unix_ns":started_ns,"environment_overrides":overrides}
    start_bytes=(json.dumps(start,sort_keys=True)+"\n").encode(); start_path.write_bytes(start_bytes)
    with (root/"starts.jsonl").open("ab") as stream: stream.write(start_bytes); stream.flush(); os.fsync(stream.fileno())
    outcome="completed"; code=127
    try:
        with stdout_path.open("wb") as out,stderr_path.open("wb") as err:
            process=subprocess.Popen(argv,cwd=os.getcwd(),env=env,stdout=out,stderr=err)
            try:
                code=process.wait()
                if code < 0:
                    outcome="signaled"
            except KeyboardInterrupt:
                outcome="interrupted"; process.send_signal(signal.SIGINT)
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
                code=130
    except OSError as error:
        outcome="spawn_error"; stderr_path.write_text(f"{type(error).__name__}: {error}\n",encoding="utf-8")
    ended=utc_now(); terminal={"event_id":number,"exit_code":code,"outcome":outcome,"signal_number":-code if code < 0 else None,"started_at":started,"ended_at":ended,"ended_unix_ns":time.time_ns()}; exit_path.write_text(json.dumps(terminal,sort_keys=True)+"\n",encoding="utf-8")
    stdout=stdout_path.read_bytes(); stderr=stderr_path.read_bytes(); immutable={}; excluded={"commands","target","uv-cache","uv-environment","python-cache","tmp","xdg-cache"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or excluded.intersection(path.relative_to(root).parts) or path.is_relative_to(command_dir) or path.name in {"manifest.jsonl","full.log","starts.jsonl","receipt.json","artifacts.jsonl","comparisons.jsonl"}: continue
        if "cassettes" in path.relative_to(root).parts or path.suffix.lower() in {".npy",".npz",".safetensors",".cassette",".toml",".json"} or "payload" in path.parts:
            data=path.read_bytes(); destination=command_dir/"artifacts"/path.relative_to(root).as_posix().replace("/","__"); destination.parent.mkdir(parents=True,exist_ok=True); destination.write_bytes(data); immutable[path.relative_to(root).as_posix()]={"path":destination.relative_to(root).as_posix(),"sha256":sha(data),"bytes":len(data)}
    event={**start,**terminal,"streams":{"stdout":{"path":stdout_path.relative_to(root).as_posix(),"bytes":len(stdout),"sha256":sha(stdout)},"stderr":{"path":stderr_path.relative_to(root).as_posix(),"bytes":len(stderr),"sha256":sha(stderr)},"exit":{"path":exit_path.relative_to(root).as_posix(),"sha256":sha(exit_path.read_bytes())}},"immutable_artifacts":immutable}
    complete={"event":event,"stdout_base64":base64.b64encode(stdout).decode(),"stderr_base64":base64.b64encode(stderr).decode()}
    for name,record in (("manifest.jsonl",event),("full.log",complete)):
        with (root/name).open("a",encoding="utf-8") as stream: stream.write(json.dumps(record,sort_keys=True)+"\n"); stream.flush(); os.fsync(stream.fileno())
    artifact_rows = []
    for stream_name in ("stdout", "stderr", "exit"):
        stream_data = event["streams"][stream_name]
        artifact_rows.append({"id": f"event-{number:06d}-{stream_name}", "kind": "event_output", "relative_path": stream_data["path"], "sha256": stream_data["sha256"], "producer_event_id": number})
    for relative, artifact in immutable.items():
        suffix = Path(relative).suffix.lower()
        kind = "cassette" if "cassette" in relative.lower() else ("payload" if suffix in {".npy", ".npz", ".safetensors"} or "payload" in relative.lower() else "actual")
        artifact_rows.append({"id": f"event-{number:06d}-{len(artifact_rows):03d}", "kind": kind, "relative_path": artifact["path"], "sha256": artifact["sha256"], "producer_event_id": number})
    with (root/"artifacts.jsonl").open("a",encoding="utf-8") as stream:
        for artifact_row in artifact_rows:
            stream.write(json.dumps(artifact_row,sort_keys=True)+"\n")
    (root/"comparisons.jsonl").touch(exist_ok=True)
    receipt = {"event_count":number,"command_directory_count":number,"manifest_sha256":sha((root/"manifest.jsonl").read_bytes()),"full_log_sha256":sha((root/"full.log").read_bytes()),"starts_sha256":sha((root/"starts.jsonl").read_bytes()),"artifacts_sha256":sha((root/"artifacts.jsonl").read_bytes()),"comparisons_sha256":sha((root/"comparisons.jsonl").read_bytes()),"last_event_id":number,"last_event_sha256":sha(json.dumps(event,sort_keys=True).encode())}
    (root/"receipt.json").write_text(json.dumps(receipt,sort_keys=True)+"\n",encoding="utf-8")
    refresh_active_segment(root)
    if relay:
        sys.stdout.buffer.write(stdout); sys.stdout.buffer.flush(); sys.stderr.buffer.write(stderr); sys.stderr.buffer.flush()
    return code
def capture_one(root, argv, capture_argv=None):
    resolved = root.expanduser().resolve()
    repository = repo_root(Path.cwd())
    if containing_worktree(resolved):
        raise SystemExit("evidence root must be outside every repository worktree")
    inherited = os.environ.get("PROXIMA_CASSETTE_DIR")
    if inherited and containing_worktree(Path(inherited)):
        raise SystemExit("inherited cassette root is inside a repository worktree")
    resolved.mkdir(parents=True, exist_ok=True)
    lock_root = Path(tempfile.gettempdir()) / "proxima-python-evidence-locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / (sha(str(resolved).encode()) + ".lock")
    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        try:
            return _capture_one_locked(resolved, argv, capture_argv)
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--root",required=True,type=Path); parser.add_argument("--initialize-segments",action="store_true"); parser.add_argument("command",nargs=argparse.REMAINDER); args=parser.parse_args(); command=args.command[1:] if args.command[:1]==["--"] else args.command
    if args.initialize_segments:
        if containing_worktree(args.root.expanduser().resolve()):
            parser.error("segment evidence root must be outside every repository worktree")
        inherited = os.environ.get("PROXIMA_CASSETTE_DIR")
        if inherited and containing_worktree(Path(inherited)):
            parser.error("inherited cassette root is inside a repository worktree")
        from check_evidence import write_segment_chain
        write_segment_chain(args.root.resolve().parent)
        command = [sys.executable, "-c", "print('segment chain initialized')"]
    if not command: parser.error("a command is required after --")
    return capture_one(args.root,command,sys.argv)
if __name__=="__main__": raise SystemExit(main())
