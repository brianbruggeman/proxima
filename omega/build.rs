// build-time sizing for omega's MSL kernel selection — mirrors
// `proxima-tensor/build.rs`'s `emit_sizing_consts`/`resolve_int` shape over
// `proxima-tensor-runtime.toml`, instantiated here for `omega-runtime.toml`
// instead. See `src/sized.rs`'s module doc for the two-family split this
// generated module feeds.
use std::env;
use std::fs;
use std::path::PathBuf;

use toml::Value;

fn main() {
    emit_sizing_consts();
}

fn require_nonzero(name: &str, value: i64) -> usize {
    let value = usize::try_from(value)
        .unwrap_or_else(|_| panic!("{name} must be a non-negative integer; got {value}"));
    assert!(value > 0, "{name} must be non-zero");
    value
}

/// Like [`require_nonzero`], but `0` is a legal value -- for keys where `0`
/// is a documented sentinel (`packed_row_block.split_k_max_rows`: `0`
/// disables the row ceiling; `cooperative_reduce.min_len`: `0` disables the
/// serial-route threshold, restoring the routing every build before that key
/// existed used) rather than a degenerate configuration.
fn require_nonneg(name: &str, value: i64) -> usize {
    usize::try_from(value)
        .unwrap_or_else(|_| panic!("{name} must be a non-negative integer; got {value}"))
}

/// Cross-axis validation for the tiled-GEMM geometry (principle 8: these
/// rules live with the profile type, i.e. right here alongside the values
/// they constrain, not scattered into the consuming kernel emitter).
/// `block_m`/`block_n` must be a whole number of `2 * TILE_DIM` (2
/// simdgroup halves, `TILE_DIM`(8)-wide `simdgroup_matrix` fragments each —
/// see `omega/src/msl.rs`'s `TILED_GEMM_NSG` doc) or the row/col-half
/// pointer arithmetic in `push_tiled_gemm_body` reads a fragment straddling
/// two logical halves. `block_k` must divide 256 (`Q4K_BLOCK_ELEMENTS`)
/// evenly: every real Q4_K reduction extent is a whole multiple of 256
/// (`classify_packed_row_block`'s own gate), so this is what guarantees the
/// tiled kernel's `k0` loop always lands exactly on `u.reduction_total`
/// without a per-iteration bounds check.
fn require_multiple_of_sixteen(name: &str, value: usize) -> usize {
    assert!(
        value.is_multiple_of(16),
        "{name} must be a multiple of 16; got {value}"
    );
    value
}

fn require_divides_q4k_block(name: &str, value: usize) -> usize {
    assert!(
        256usize.is_multiple_of(value),
        "{name} must evenly divide 256 (Q4K_BLOCK_ELEMENTS); got {value}"
    );
    value
}

/// ROW 113's weight-staging amortization (`push_tiled_gemm_body`'s
/// `q4k_header_for`/`q4k_run8` loop, replacing the per-element `q4k_element`
/// rederive) batches the nibble extract 8 elements at a time
/// (`q4k_run8`'s own fixed width). Combined with [`require_divides_q4k_block`]
/// (a power-of-two divisor of 256), this guarantees `block_k` is always
/// either <= the Q4_K sub-block width (32) or a whole multiple of it, so
/// `push_tiled_gemm_body`'s per-row staging loop never has a ragged
/// remainder chunk.
fn require_multiple_of_eight(name: &str, value: usize) -> usize {
    assert!(
        value.is_multiple_of(8),
        "{name} must be a multiple of 8; got {value}"
    );
    value
}

/// [`crate::sized::PACKED_ROW_BLOCK_SIMDGROUPS`]'s cross-axis rule: a power
/// of two (so `simdgroups * SIMD_WIDTH` is always a clean multiple of
/// `SIMD_WIDTH`, the invariant `msl.rs`'s row-blocked kernel body's
/// `gid / SIMD_WIDTH` output-index math depends on) and no larger than 32
/// (`simdgroups * SIMD_WIDTH` <= 1024, the per-threadgroup thread ceiling
/// every Apple GPU family this crate targets enforces).
fn require_power_of_two_le_32(name: &str, value: usize) -> usize {
    assert!(
        value.is_power_of_two(),
        "{name} must be a power of two; got {value}"
    );
    assert!(value <= 32, "{name} must be <= 32; got {value}");
    value
}

/// [`crate::sized::WIDE_COOPERATIVE_REDUCE_MAX_WIDTH`] must stay a whole
/// number of simdgroups: `msl::push_cooperative_reduce_tail`'s two-level
/// fold divides the chosen width by `SIMD_WIDTH` to size its `threadgroup`
/// partials array, and `msl::cooperative_reduce_width` only ever rounds UP
/// to a multiple of 32 before clamping against this cap — a cap that is not
/// itself a multiple of 32 would silently clamp a wide reduce down to a
/// width its own preceding `next_multiple_of(32)` never produces.
fn require_multiple_of_thirty_two(name: &str, value: usize) -> usize {
    assert!(
        value.is_multiple_of(32),
        "{name} must be a multiple of 32 (SIMD_WIDTH); got {value}"
    );
    value
}

/// Cross-axis validation for split-K's `max_split` (`[packed_row_split_k]`):
/// `max_split * SIMD_WIDTH`(32) must stay inside `crate::metal::dispatch`'s
/// clamp headroom (`.min(max_threadgroup)`, typically 1024 on Apple Silicon)
/// -- `grid_threads` commits to a total thread count that is an EXACT
/// multiple of `SIMD_WIDTH * split`, and if `dispatch`'s clamp ever shrank
/// the threadgroup width below what `grid_threads` assumed, the kernel's
/// `output_index = gid / tptg_width` group indexing would silently read the
/// wrong row group. 1024 is a conservative, real-hardware floor (Apple GPU
/// family docs), not this crate's own policy -- kept here rather than in
/// `sized.rs` because the value being validated is read once, at build time,
/// long before any device is queried.
fn require_split_k_headroom(max_split: usize) -> usize {
    const CONSERVATIVE_MAX_THREADGROUP: usize = 1024;
    const SIMD_WIDTH: usize = 32;
    assert!(
        max_split * SIMD_WIDTH <= CONSERVATIVE_MAX_THREADGROUP,
        "packed_row_split_k.max_split={max_split} * SIMD_WIDTH(32) must not exceed {CONSERVATIVE_MAX_THREADGROUP}"
    );
    max_split
}

fn get_int(table: &Value, section: &str, key: &str) -> i64 {
    table
        .get(section)
        .and_then(|section_value| section_value.get(key))
        .and_then(Value::as_integer)
        .unwrap_or_else(|| panic!("omega-runtime.toml: missing or non-integer [{section}].{key}"))
}

/// like `get_int`, but an `OMEGA_<SECTION>_<KEY>` env var overrides the TOML
/// value when present — mirrors `proxima-tensor/build.rs`'s `resolve_int`
/// (principle 12: every override consulted emits its own
/// `cargo:rerun-if-env-changed` line, so a cached build never ignores it).
fn resolve_int(table: &Value, section: &str, key: &str) -> i64 {
    let env_name = format!(
        "OMEGA_{section}_{key}",
        section = section.to_uppercase(),
        key = key.to_uppercase()
    );
    println!("cargo:rerun-if-env-changed={env_name}");
    match env::var(&env_name) {
        Ok(raw) => raw
            .parse::<i64>()
            .unwrap_or_else(|err| panic!("{env_name}={raw} must parse as i64: {err}")),
        Err(_) => get_int(table, section, key),
    }
}

/// Reads `omega-runtime.toml`, emits `OUT_DIR/omega_sized.rs`. A new
/// execution-policy key follows the same `resolve_int` + `require_nonzero` +
/// generated-`pub const` shape `packed_row_block.simdgroups` and
/// `tiled_gemm.min_tokens` already do (`cooperative_reduce.min_len` and
/// `packed_row_block.split_k_max_rows` are the exceptions that need
/// `require_nonneg` instead: 0 is a legal, meaningful value for both).
/// `TILED_GEMM_MIN_TOKENS` is `feature = "metal-tiled-gemm"`-only in
/// `src/sized.rs` (the tiled path has no meaning without it), so this only
/// emits the line when Cargo reports the feature active for THIS build — on
/// every other build the generated file simply omits it, keeping the private
/// `generated` module free of an unreferenced (dead-code-linted) constant,
/// the same convention `proxima-tensor/build.rs`'s
/// `CARGO_FEATURE_COHORT_STAGED_GRAPH` branch uses.
#[allow(clippy::expect_used)]
fn emit_sizing_consts() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set by cargo");
    let toml_path = PathBuf::from(&manifest_dir).join("omega-runtime.toml");
    println!("cargo:rerun-if-changed=omega-runtime.toml");

    let text = fs::read_to_string(&toml_path)
        .unwrap_or_else(|err| panic!("read {}: {err}", toml_path.display()));
    let root: Value = text
        .parse()
        .unwrap_or_else(|err| panic!("parse {}: {err}", toml_path.display()));

    let mut out =
        String::from("// AUTO-GENERATED by build.rs from omega-runtime.toml. DO NOT EDIT.\n");

    let packed_row_block_simdgroups = require_power_of_two_le_32(
        "packed_row_block.simdgroups",
        require_nonzero(
            "packed_row_block.simdgroups",
            resolve_int(&root, "packed_row_block", "simdgroups"),
        ),
    );
    out.push_str(&format!(
        "pub const PACKED_ROW_BLOCK_SIMDGROUPS: u64 = {packed_row_block_simdgroups};\n"
    ));

    let cooperative_reduce_min_len = require_nonneg(
        "cooperative_reduce.min_len",
        resolve_int(&root, "cooperative_reduce", "min_len"),
    );
    out.push_str(&format!(
        "pub const COOPERATIVE_REDUCE_MIN_LEN: u64 = {cooperative_reduce_min_len};\n"
    ));

    if env::var_os("CARGO_FEATURE_METAL_TILED_GEMM").is_some() {
        let min_tokens = require_nonzero(
            "tiled_gemm.min_tokens",
            resolve_int(&root, "tiled_gemm", "min_tokens"),
        );
        out.push_str(&format!(
            "pub const TILED_GEMM_MIN_TOKENS: u64 = {min_tokens};\n"
        ));

        let block_m = require_multiple_of_sixteen(
            "tiled_gemm.block_m",
            require_nonzero(
                "tiled_gemm.block_m",
                resolve_int(&root, "tiled_gemm", "block_m"),
            ),
        );
        let block_n = require_multiple_of_sixteen(
            "tiled_gemm.block_n",
            require_nonzero(
                "tiled_gemm.block_n",
                resolve_int(&root, "tiled_gemm", "block_n"),
            ),
        );
        let block_k = require_multiple_of_eight(
            "tiled_gemm.block_k",
            require_divides_q4k_block(
                "tiled_gemm.block_k",
                require_nonzero(
                    "tiled_gemm.block_k",
                    resolve_int(&root, "tiled_gemm", "block_k"),
                ),
            ),
        );
        out.push_str(&format!("pub const TILED_GEMM_BLOCK_M: u64 = {block_m};\n"));
        out.push_str(&format!("pub const TILED_GEMM_BLOCK_N: u64 = {block_n};\n"));
        out.push_str(&format!("pub const TILED_GEMM_BLOCK_K: u64 = {block_k};\n"));
    }

    if env::var_os("CARGO_FEATURE_METAL_WIDE_COOPERATIVE_REDUCE").is_some() {
        let max_width = require_multiple_of_thirty_two(
            "wide_cooperative_reduce.max_width",
            require_nonzero(
                "wide_cooperative_reduce.max_width",
                resolve_int(&root, "wide_cooperative_reduce", "max_width"),
            ),
        );
        out.push_str(&format!(
            "pub const WIDE_COOPERATIVE_REDUCE_MAX_WIDTH: u64 = {max_width};\n"
        ));
    }

    if env::var_os("CARGO_FEATURE_METAL_Q4K_SPLIT_K").is_some() {
        let target_simdgroups = require_nonzero(
            "packed_row_split_k.target_simdgroups",
            resolve_int(&root, "packed_row_split_k", "target_simdgroups"),
        );
        let max_split = require_split_k_headroom(require_nonzero(
            "packed_row_split_k.max_split",
            resolve_int(&root, "packed_row_split_k", "max_split"),
        ));
        out.push_str(&format!(
            "pub const PACKED_ROW_SPLIT_K_TARGET_SIMDGROUPS: u64 = {target_simdgroups};\n"
        ));
        out.push_str(&format!(
            "pub const PACKED_ROW_SPLIT_K_MAX_SPLIT: u64 = {max_split};\n"
        ));

        let split_k_max_rows = require_nonneg(
            "packed_row_block.split_k_max_rows",
            resolve_int(&root, "packed_row_block", "split_k_max_rows"),
        );
        out.push_str(&format!(
            "pub const PACKED_ROW_SPLIT_K_MAX_ROWS: u64 = {split_k_max_rows};\n"
        ));
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR set by cargo"));
    let out_path = out_dir.join("omega_sized.rs");
    fs::write(&out_path, out).unwrap_or_else(|err| panic!("write {}: {err}", out_path.display()));
}
