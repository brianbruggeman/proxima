//! `elf.rs` disciplined-component compare-bench, gated behind the
//! `elf-bench` feature (default-off — see `Cargo.toml`'s `[[bench]]`
//! `required-features`). Named incumbents: `object` and `goblin`, per
//! `tools/proxima-vm/ROADMAP.md`'s "Hand-rolling a spec parser" condition 3.
//!
//! Fixtures:
//! - the two real M1 guest ELFs (`aarch64-unknown-none` /
//!   `x86_64-unknown-none`), built on demand — `parse_elf`'s actual design
//!   point (a small, static, `ET_EXEC`, two-`PT_LOAD` bare-metal guest).
//! - `benches/fixtures/dynamic_aarch64_hello.elf`: a real, dynamically
//!   linked `aarch64-unknown-linux-gnu` std "hello world" binary
//!   (10 program headers: `PT_PHDR`/`PT_INTERP`/2x`PT_LOAD`/`PT_DYNAMIC`/
//!   `PT_NOTE`/`PT_TLS`/`PT_GNU_EH_FRAME`/`PT_GNU_STACK`/`PT_GNU_RELRO`,
//!   `ET_DYN`) — `object`/`goblin`'s home turf: a full dynamic executable a
//!   real dynamic loader would process. `parse_elf` legitimately rejects it
//!   (`UnsupportedType` — the module's own doc comment names `ET_DYN` as
//!   out of scope: no relocation processing). That arm is a **feature
//!   gap**, not a perf loss, and is reported as such (gate 13's "cannot run
//!   the arm" case), never as a throughput number.
//!
//! Every arm is labeled `design-favors: incumbent | ours | neutral` per the
//! disciplined-component skill's per-arm labeling requirement.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use object::Object;
use proxima_vm::elf::parse_elf;

const MAX_SEGMENTS: usize = 4;
const LARGE_FIXTURE: &[u8] = include_bytes!("fixtures/dynamic_aarch64_hello.elf");

fn target_dir() -> PathBuf {
    env::var("CARGO_TARGET_DIR").map_or_else(
        |_| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("tools dir")
                .parent()
                .expect("workspace root")
                .join("target")
        },
        PathBuf::from,
    )
}

fn guest_elf_bytes(target_triple: &str) -> Vec<u8> {
    let artifact = target_dir()
        .join(target_triple)
        .join("debug")
        .join("proxima-vm-guest-lambda");
    if !artifact.exists() {
        let status = Command::new("cargo")
            .args([
                "build",
                "-p",
                "proxima-vm-guest-lambda",
                "--target",
                target_triple,
            ])
            .status()
            .expect("run cargo build for the guest crate");
        assert!(status.success(), "cargo build for {target_triple} failed");
    }
    std::fs::read(&artifact).expect("read built guest elf")
}

fn bench_home_turf_small_guest(criterion: &mut Criterion) {
    let aarch64_guest = guest_elf_bytes("aarch64-unknown-none");
    let x86_64_guest = guest_elf_bytes("x86_64-unknown-none");

    for (label, image) in [
        ("aarch64_guest", &aarch64_guest),
        ("x86_64_guest", &x86_64_guest),
    ] {
        let mut group = criterion.benchmark_group(format!("small_guest_elf/{label}"));
        group.throughput(Throughput::Bytes(image.len() as u64));

        // design-favors: ours -- parse_elf's actual design point: a small,
        // static, ET_EXEC, two-PT_LOAD bare-metal guest.
        group.bench_function("proxima_vm_parse_elf", |bencher| {
            bencher.iter(|| {
                let (entry, segments) = parse_elf::<MAX_SEGMENTS>(std::hint::black_box(image))
                    .expect("real guest ELF parses");
                std::hint::black_box((entry, segments.len()));
            });
        });

        // design-favors: neutral -- object's ELF64 decode on the identical
        // small file; their machinery (relocation tables, symbol tables,
        // section indices) is not engaged by this input.
        group.bench_function("object_parse", |bencher| {
            bencher.iter(|| {
                let file = object::File::parse(std::hint::black_box(image.as_slice()))
                    .expect("object parses the real guest ELF");
                std::hint::black_box(file.entry());
            });
        });

        // design-favors: neutral -- same rationale as the object arm.
        group.bench_function("goblin_parse", |bencher| {
            bencher.iter(|| {
                let elf = goblin::elf::Elf::parse(std::hint::black_box(image))
                    .expect("goblin parses the real guest ELF");
                std::hint::black_box(elf.entry);
            });
        });

        group.finish();
    }
}

fn bench_home_turf_large_dynamic_binary(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("large_dynamic_elf/aarch64_linux_hello");
    group.throughput(Throughput::Bytes(LARGE_FIXTURE.len() as u64));

    // design-favors: incumbent -- a full dynamically linked ET_DYN
    // executable (PT_INTERP/PT_DYNAMIC/PT_TLS/PT_GNU_*) is exactly the
    // shape a real dynamic loader hands `object`: their headline use case.
    group.bench_function("object_parse", |bencher| {
        bencher.iter(|| {
            let file = object::File::parse(std::hint::black_box(LARGE_FIXTURE))
                .expect("object parses the real dynamic ELF");
            std::hint::black_box(file.entry());
        });
    });

    // design-favors: incumbent -- same rationale; goblin's ELF64 module is
    // built around exactly this shape (dynamic symbol table, PT_DYNAMIC).
    group.bench_function("goblin_parse", |bencher| {
        bencher.iter(|| {
            let elf = goblin::elf::Elf::parse(std::hint::black_box(LARGE_FIXTURE))
                .expect("goblin parses the real dynamic ELF");
            std::hint::black_box(elf.entry);
        });
    });

    group.finish();

    // Not a throughput arm: parse_elf legitimately rejects this input
    // (ET_DYN is out of scope per the module doc comment) -- a feature gap,
    // recorded once as a correctness fact, never averaged into the bench.
    let rejection = parse_elf::<MAX_SEGMENTS>(LARGE_FIXTURE)
        .expect_err("parse_elf must name ET_DYN as unsupported, not silently mis-map it");
    assert!(
        matches!(
            rejection,
            proxima_vm::elf::LoaderError::UnsupportedType { .. }
        ),
        "expected UnsupportedType, got {rejection:?}"
    );
}

criterion_group!(
    benches,
    bench_home_turf_small_guest,
    bench_home_turf_large_dynamic_binary
);
criterion_main!(benches);
