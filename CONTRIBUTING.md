# Contributing

Thanks for helping improve Proxima.

## License

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this repository is dual-licensed under MIT OR Apache-2.0, the
same terms as the project.

## Current Release State

The workspace is public-source oriented, but crates.io publication is not the
current release path. Do not add publish automation or crates.io-specific release
steps until the package naming policy is settled.

## Development Setup

Use Rust 1.96 or newer. The workspace records `rust-version = "1.96"` because
that is the version currently validated here.

Useful baseline commands:

```bash
cargo fmt --all
env -u RUSTC_WRAPPER -u CARGO_BUILD_RUSTC_WRAPPER cargo check --workspace --all-targets
env -u RUSTC_WRAPPER -u CARGO_BUILD_RUSTC_WRAPPER cargo test --workspace --no-run
env -u RUSTC_WRAPPER -u CARGO_BUILD_RUSTC_WRAPPER cargo clippy --workspace --all-targets -- -D warnings
```

The `env -u ...` wrapper is only needed on machines where a local `sccache` or
other `rustc` wrapper is misconfigured. If your wrapper works, normal `cargo`
commands are fine.

Some optional crates and tests need platform-specific support:

- DPDK and low-level network paths are Linux-oriented and may require system
  libraries or privileges.
- io_uring tests require Linux support.
- KVM / Hypervisor.framework VM paths are platform-gated.
- Integration tests that talk to real Redis/PostgreSQL/remote hosts should be
  kept opt-in or fixture-backed unless a CI service explicitly provides them.

## Change Guidelines

- Keep changes scoped. Avoid mixing behavior changes, formatting sweeps, and
  release hygiene in the same patch unless the task explicitly requires it.
- Prefer existing workspace primitives and patterns over new abstractions.
- Add tests in the crate that owns the behavior. Broaden tests when touching a
  shared trait, codec, config surface, or listener path.
- Keep generated or private agent/worktree files out of commits. Local AI
  assistant state directories, target directories, captures with secrets, and
  temporary worktrees must not be committed.
- Do not add new Git dependencies for code that is expected to be published as a
  crate later. Prefer crates.io dependencies or workspace crates.
- Redact fixtures before committing them. Captures must not contain real API
  keys, cookies, bearer tokens, account identifiers, customer data, or private
  prompts.

## Security Work

Do not disclose vulnerabilities in public issues or pull requests. Follow
[SECURITY.md](SECURITY.md).

Security fixes should include:

- a focused regression test where possible;
- an impact note in the pull request or commit message;
- fixture redaction notes if captures are involved.

## Documentation

Update docs when changing public behavior, feature flags, config keys, command
syntax, or release expectations. If you add vendored source or externally
derived fixtures, update [THIRD_PARTY.md](THIRD_PARTY.md).
