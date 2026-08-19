// links a statically-built ggml checkout for the `ggml-bench` criterion
// bench only; a no-op for every other build so a normal `cargo build` never
// needs ggml on disk.
use std::env;
use std::path::PathBuf;

fn main() {
    emit_dotprod_cfg();
    emit_avx2_cfg();

    if env::var_os("CARGO_FEATURE_GGML_BENCH").is_none() {
        return;
    }

    let ggml_root = env::var("GGML_BUILD_DIR").unwrap_or_else(|_| {
        panic!(
            "the `ggml-bench` feature needs a statically-built ggml checkout on disk; \
             set GGML_BUILD_DIR to its root (the directory whose `build/src/` holds \
             libggml*.a) -- there is no session-scoped default, every session's \
             scratchpad path differs"
        )
    });
    let build_dir = PathBuf::from(&ggml_root).join("build");

    println!("cargo:rustc-link-search=native={}", build_dir.join("src").display());
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rerun-if-env-changed=GGML_BUILD_DIR");
}

/// Declares the `q4k_dotprod` cfg `cpu::dot_q4k_q8k`'s NEON kernel (feature
/// `q4k-int8-dot`) keys off, in place of a runtime
/// `is_aarch64_feature_detected!("dotprod")` branch: `FEAT_DotProd` is not
/// in the aarch64 base ISA (unlike this crate's other NEON tiles, which
/// lean on `neon` itself being unconditional on aarch64 --
/// `proxima-tensor/src/sized.rs`'s `WIDTH_TILE_ROWS` doc states the same
/// assumption for baseline NEON), but every aarch64 target this workspace
/// actually builds for -- Apple M1 and later -- has it. That is a
/// build-time platform decision, declared once here rather than sniffed
/// per call: when the cfg is off, `dot_q4k_q8k`'s aarch64 arm simply is
/// not compiled and the portable scalar arm is the only one built, exactly
/// as this crate's existing `dot_q4k_f32` stays the production default
/// until the packed kernel earns the switch.
fn emit_dotprod_cfg() {
    println!("cargo:rustc-check-cfg=cfg(q4k_dotprod)");
    if env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("aarch64") {
        println!("cargo:rustc-cfg=q4k_dotprod");
    }
}

/// Declares the `q4k_avx2` cfg `cpu::dot_q4k_q8k`'s x86 kernel (feature
/// `q4k-int8-dot`) keys off -- the x86 sibling of `emit_dotprod_cfg`'s
/// aarch64 cfg above, but the availability signal is NOT the same shape.
/// `FEAT_DotProd` is absent from the aarch64 *base* ISA yet present on
/// every aarch64 target this workspace actually builds for, so
/// `emit_dotprod_cfg` can key on `CARGO_CFG_TARGET_ARCH` alone. AVX2 is
/// NOT in that position: it is absent from the x86-64 *baseline* ISA
/// (`x86_64-unknown-linux-gnu`'s default `CARGO_CFG_TARGET_FEATURE` is
/// `fxsr,sse,sse2` only -- confirmed via `rustc --print cfg --target
/// x86_64-unknown-linux-gnu`), so declaring it from `target_arch` alone
/// would be dishonest: it would compile `_mm256_*` intrinsics into a
/// binary the default x86-64 target doesn't guarantee support for. The
/// build only turns this cfg on when `CARGO_CFG_TARGET_FEATURE` (which
/// Cargo derives from `-C target-feature=+avx2` / `-C target-cpu=<v3 or
/// newer, or native on an AVX2 host>`) actually lists `avx2` -- i.e. the
/// caller opted in via RUSTFLAGS or `-C target-cpu`, not merely by
/// picking an x86_64 target triple. When the cfg is off (the default,
/// unqualified x86-64 build), `dot_q4k_q8k`'s x86 arm is simply not
/// compiled and the portable scalar arm remains the only one built, same
/// as any non-aarch64, non-AVX2 target.
fn emit_avx2_cfg() {
    println!("cargo:rustc-check-cfg=cfg(q4k_avx2)");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_FEATURE");
    let target_is_x86 = matches!(env::var("CARGO_CFG_TARGET_ARCH").as_deref(), Ok("x86_64" | "x86"));
    let target_feature_has_avx2 = env::var("CARGO_CFG_TARGET_FEATURE")
        .map(|features| features.split(',').any(|feature| feature == "avx2"))
        .unwrap_or(false);
    if target_is_x86 && target_feature_has_avx2 {
        println!("cargo:rustc-cfg=q4k_avx2");
    }
}
