#!/usr/bin/env python3
"""Decide which CI areas a change set touches, considering the Cargo
dependency graph when the repo is a Cargo workspace.

Repo-agnostic: areas are declared in `.github/ci-areas.toml`. Each area is
defined by root `packages` (cargo mode) and/or `paths` globs (works with or
without cargo). An area runs when a changed file is in its package's
dependency closure OR matches one of its path globs. Global paths (lockfile,
root manifest, the .github tree, this script, the areas config) force every
area on, as do `--force-all` (release tags / workflow_dispatch).

Outputs `area=true|false` lines to $GITHUB_OUTPUT (and stdout). Exit 0 always;
a detection error forces every area on (fail safe — never silently skip).
"""

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

GLOBAL_GLOBS = [
    "Cargo.lock",
    "Cargo.toml",
    ".github/workflows/affected.yml",
    "scripts/ci-affected.py",
    ".github/ci-areas.toml",
    "rust-toolchain*",
]


def changed_files(base: str | None, head: str | None) -> list[str]:
    if base and head:
        diff = subprocess.run(
            ["git", "diff", "--name-only", f"{base}", f"{head}"],
            capture_output=True, text=True, check=True,
        )
        return [line for line in diff.stdout.splitlines() if line]
    return []


def workspace_graph() -> tuple[dict[str, set[str]], dict[str, str]]:
    """Return (deps, dirs): per-package transitive workspace-dep closure and
    each package's directory (relative to repo root). Empty if not cargo."""
    if not Path("Cargo.toml").exists():
        return {}, {}
    meta = json.loads(subprocess.run(
        ["cargo", "metadata", "--format-version", "1"],
        capture_output=True, text=True, check=True,
    ).stdout)
    members = set(meta["workspace_members"])
    id_to_name = {p["id"]: p["name"] for p in meta["packages"]}
    dirs = {
        p["name"]: str(Path(p["manifest_path"]).parent.relative_to(Path.cwd()))
        for p in meta["packages"] if p["id"] in members
    }
    direct: dict[str, set[str]] = {}
    for node in meta["resolve"]["nodes"]:
        if node["id"] not in members:
            continue
        name = id_to_name[node["id"]]
        deps: set[str] = set()
        for dep in node["deps"]:
            if dep["pkg"] not in members:
                continue
            # dev-only deps (integration-test helpers, umbrella-for-tests)
            # would drag the whole graph into every leaf's closure
            kinds = {kind["kind"] for kind in dep.get("dep_kinds", [])}
            if kinds and kinds <= {"dev"}:
                continue
            deps.add(id_to_name[dep["pkg"]])
        direct[name] = deps

    closure: dict[str, set[str]] = {}

    def resolve(pkg: str, seen: set[str]) -> set[str]:
        if pkg in closure:
            return closure[pkg]
        if pkg in seen:
            return set()
        seen.add(pkg)
        acc = {pkg}
        for dep in direct.get(pkg, set()):
            acc |= resolve(dep, seen)
        closure[pkg] = acc
        return acc

    for pkg in direct:
        resolve(pkg, set())
    return closure, dirs


ROOT_SRC_PREFIXES = ("src/", "benches/", "tests/", "examples/")


def owning_packages(files: list[str], dirs: dict[str, str]) -> set[str]:
    """Map changed files to the workspace package whose dir is the longest
    matching path prefix. The root crate (dir ".") owns only its own source
    trees, not repo-infra files (scripts/, .github/, docs/, *.md) — those are
    routed by area path globs or the global force-all set instead."""
    owners: set[str] = set()
    ranked = sorted(dirs.items(), key=lambda kv: len(kv[1]), reverse=True)
    for path in files:
        for name, directory in ranked:
            if directory in (".", ""):
                if path.startswith(ROOT_SRC_PREFIXES):
                    owners.add(name)
                    break
                continue
            if path.startswith(directory + "/"):
                owners.add(name)
                break
    return owners


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--force-all", action="store_true")
    parser.add_argument("--changed", nargs="*", help="explicit file list (testing)")
    parser.add_argument("--config", default=".github/ci-areas.toml")
    args = parser.parse_args()

    areas = tomllib.loads(Path(args.config).read_text())["area"]

    def emit(values: dict[str, bool]) -> int:
        lines = [f"{name}={'true' if hit else 'false'}" for name, hit in values.items()]
        print("\n".join(lines))
        out = os.environ.get("GITHUB_OUTPUT")
        if out:
            Path(out).write_text("\n".join(lines) + "\n")
        return 0

    force = args.force_all or os.environ.get("FORCE_ALL") == "true"
    try:
        files = args.changed if args.changed is not None else changed_files(args.base, args.head)
    except subprocess.CalledProcessError:
        return emit({name: True for name in areas})  # fail safe

    if force or not files:
        return emit({name: True for name in areas})

    if any(any(fnmatch.fnmatch(path, glob) for glob in GLOBAL_GLOBS) for path in files):
        return emit({name: True for name in areas})

    try:
        closure, dirs = workspace_graph()
    except (subprocess.CalledProcessError, KeyError):
        return emit({name: True for name in areas})  # fail safe

    changed_pkgs = owning_packages(files, dirs)
    result: dict[str, bool] = {}
    for name, spec in areas.items():
        pkgs = set(spec.get("packages", []))
        reachable = set().union(*(closure.get(pkg, {pkg}) for pkg in pkgs)) if pkgs else set()
        by_dep = bool(reachable & changed_pkgs)
        by_path = any(
            fnmatch.fnmatch(path, glob)
            for glob in spec.get("paths", [])
            for path in files
        )
        result[name] = by_dep or by_path
    return emit(result)


if __name__ == "__main__":
    sys.exit(main())
