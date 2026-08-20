// links a statically-built ggml checkout for the `ggml-bench` criterion
// bench only; a no-op for every other build so a normal `cargo build` never
// needs ggml on disk.
use std::env;
use std::fs;
use std::path::PathBuf;

use toml::Value;

fn main() {
    emit_dotprod_cfg();
    emit_avx2_cfg();
    emit_sizing_consts();

    if env::var_os("CARGO_FEATURE_GGML_BENCH").is_none() {
        return;
    }

    let ggml_root = match env::var("GGML_BUILD_DIR") {
        Ok(root) => root,
        Err(_) => {
            println!(
                "cargo:warning=`ggml-bench` is enabled but GGML_BUILD_DIR is unset; \
                 skipping the ggml link directives -- doc/clippy/check builds still \
                 succeed, but `cargo bench -p proxima-tensor --features ggml-bench` \
                 will fail at link time. set GGML_BUILD_DIR to a statically-built ggml \
                 checkout's root (the directory whose `build/src/` holds libggml*.a) \
                 to run the bench."
            );
            println!("cargo:rerun-if-env-changed=GGML_BUILD_DIR");
            return;
        }
    };
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

fn require_nonzero(name: &str, value: i64) -> usize {
    let value = usize::try_from(value)
        .unwrap_or_else(|_| panic!("{name} must be a non-negative integer; got {value}"));
    assert!(value > 0, "{name} must be non-zero");
    value
}

fn get_int(table: &Value, section: &str, key: &str) -> i64 {
    table
        .get(section)
        .and_then(|section_value| section_value.get(key))
        .and_then(Value::as_integer)
        .unwrap_or_else(|| {
            panic!("proxima-tensor-runtime.toml: missing or non-integer [{section}].{key}")
        })
}

/// Mirrors `prime/build.rs`'s `emit_sizing_consts` -- reads
/// `proxima-tensor-runtime.toml`, emits `OUT_DIR/proxima_tensor_sized.rs`,
/// always regardless of feature (cheap, matches prime's unconditional
/// emission). Only the execution-policy family from `src/sized.rs`'s
/// module doc traces here; the const-generic-shaped family
/// (`MAX_INLINE_RANK`, `TILE_ROWS`, `DOT_LANES`, ...) sizes fixed
/// array/`ArrayVec`/`SmallVec` capacities or kernel `const` generics and
/// stays hand-written, exactly as that doc states.
///
/// `NEON_COLUMN_PANEL_BUDGET_BYTES` is `target_arch = "aarch64"`-only in
/// `src/sized.rs`, so this only emits that line when
/// `CARGO_CFG_TARGET_ARCH` is `aarch64` -- on every other target the
/// generated file simply omits it, keeping the private `generated` module
/// free of an unreferenced (dead-code-linted) constant on those targets.
#[allow(clippy::expect_used)]
fn emit_sizing_consts() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set by cargo");
    let toml_path = PathBuf::from(&manifest_dir).join("proxima-tensor-runtime.toml");
    println!("cargo:rerun-if-changed=proxima-tensor-runtime.toml");

    let text = fs::read_to_string(&toml_path)
        .unwrap_or_else(|err| panic!("read {}: {err}", toml_path.display()));
    let root: Value = text
        .parse()
        .unwrap_or_else(|err| panic!("parse {}: {err}", toml_path.display()));

    let parallel_threshold =
        require_nonzero("parallel.threshold", get_int(&root, "parallel", "threshold"));
    let oversubscribe =
        require_nonzero("parallel.oversubscribe", get_int(&root, "parallel", "oversubscribe"));
    let row_oversubscribe = require_nonzero(
        "parallel.row_oversubscribe",
        get_int(&root, "parallel", "row_oversubscribe"),
    );
    let split_alignment = require_nonzero(
        "parallel.split_alignment",
        get_int(&root, "parallel", "split_alignment"),
    );
    let min_macs_per_chunk = require_nonzero(
        "parallel.min_macs_per_chunk",
        get_int(&root, "parallel", "min_macs_per_chunk"),
    );
    let min_quantize_blocks_for_dispatch = require_nonzero(
        "quantize.min_blocks_for_dispatch",
        get_int(&root, "quantize", "min_blocks_for_dispatch"),
    );
    let min_transpose_elements_for_dispatch = require_nonzero(
        "transpose.min_elements_for_dispatch",
        get_int(&root, "transpose", "min_elements_for_dispatch"),
    );
    let cohort_spin_polls =
        require_nonzero("cohort.spin_polls", get_int(&root, "cohort", "spin_polls"));

    let mut out = format!(
        "// AUTO-GENERATED by build.rs from proxima-tensor-runtime.toml. DO NOT EDIT.\n\
         pub const PARALLEL_THRESHOLD: usize = {parallel_threshold};\n\
         pub const OVERSUBSCRIBE: usize = {oversubscribe};\n\
         pub const ROW_OVERSUBSCRIBE: usize = {row_oversubscribe};\n\
         pub const SPLIT_ALIGNMENT: u64 = {split_alignment};\n\
         pub const MIN_MACS_PER_CHUNK: usize = {min_macs_per_chunk};\n\
         pub const MIN_QUANTIZE_BLOCKS_FOR_DISPATCH: usize = {min_quantize_blocks_for_dispatch};\n\
         pub const MIN_TRANSPOSE_ELEMENTS_FOR_DISPATCH: usize = {min_transpose_elements_for_dispatch};\n\
         pub const COHORT_SPIN_POLLS: u32 = {cohort_spin_polls};\n",
    );

    if env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("aarch64") {
        let column_panel_budget_bytes = require_nonzero(
            "neon.column_panel_budget_bytes",
            get_int(&root, "neon", "column_panel_budget_bytes"),
        );
        out.push_str(&format!(
            "pub const NEON_COLUMN_PANEL_BUDGET_BYTES: usize = {column_panel_budget_bytes};\n"
        ));
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR set by cargo"));
    let out_path = out_dir.join("proxima_tensor_sized.rs");
    fs::write(&out_path, out).unwrap_or_else(|err| panic!("write {}: {err}", out_path.display()));
}
