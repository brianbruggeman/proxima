// build-time sizing for omega's MSL kernel selection over
// `omega-runtime.toml` -- resolution mechanics (`SizingSource::resolve_int`,
// `require_nonzero`/`require_nonneg`) come from `proxima_build::sizing`, the
// one definition `proxima-tensor/build.rs` also calls; the cross-axis
// validators below are omega's own kernel-geometry rules. See
// `src/sized.rs`'s module doc for the two-family split this generated
// module feeds.
use std::env;
use std::fs;
use std::path::PathBuf;

use proxima_build::sizing::{require_nonneg, require_nonzero, SizingSource};

fn main() {
    emit_sizing_consts();
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

/// Thin panic-on-error wrapper over `proxima_build::sizing::SizingSource`'s
/// `resolve_int` (principle 12: every override consulted emits its own
/// `cargo:rerun-if-env-changed` line, so a cached build never ignores it) --
/// build.rs has no caller to propagate a `Result` to, so this is where the
/// shared library's typed error becomes the build-time panic every other
/// `require_*` rule below already assumes.
fn resolve_int(source: &SizingSource, section: &str, key: &str) -> i64 {
    source
        .resolve_int(section, key)
        .unwrap_or_else(|err| panic!("{err}"))
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
    let root = SizingSource::load(&manifest_dir, "omega-runtime.toml", "OMEGA")
        .unwrap_or_else(|err| panic!("{err}"));

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

    let uniform_cache_entries = require_nonzero(
        "spans.uniform_cache_entries",
        resolve_int(&root, "spans", "uniform_cache_entries"),
    );
    out.push_str(&format!(
        "pub const UNIFORM_CACHE_ENTRIES: u64 = {uniform_cache_entries};\n"
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

    let group = require_power_of_two_le_32(
        "packed_row_multi_activation.group",
        require_nonzero(
            "packed_row_multi_activation.group",
            resolve_int(&root, "packed_row_multi_activation", "group"),
        ),
    );
    out.push_str(&format!(
        "pub const PACKED_ROW_ACTIVATION_GROUP: u64 = {group};\n"
    ));

    if env::var_os("CARGO_FEATURE_METAL_BUFFER_POOL").is_some() {
        let max_per_bucket = require_nonzero(
            "output_pool.max_per_bucket",
            resolve_int(&root, "output_pool", "max_per_bucket"),
        );
        out.push_str(&format!(
            "pub const OUTPUT_POOL_MAX_PER_BUCKET: usize = {max_per_bucket};\n"
        ));
    }

    if env::var_os("CARGO_FEATURE_METAL_PLAN_STABLE_BUFFERS").is_some() {
        let transient_cap = require_nonzero(
            "arena.transient_cap",
            resolve_int(&root, "arena", "transient_cap"),
        );
        out.push_str(&format!(
            "pub const ARENA_TRANSIENT_CAP: usize = {transient_cap};\n"
        ));
    }

    let attention_context_chunks_keys_per_chunk = require_nonzero(
        "attention_context_chunks.keys_per_chunk",
        resolve_int(&root, "attention_context_chunks", "keys_per_chunk"),
    );
    out.push_str(&format!(
        "pub const ATTENTION_CONTEXT_KEYS_PER_CHUNK: u64 = {attention_context_chunks_keys_per_chunk};\n"
    ));
    let attention_context_chunks_cap = require_nonzero(
        "attention_context_chunks.cap",
        resolve_int(&root, "attention_context_chunks", "cap"),
    );
    out.push_str(&format!(
        "pub const ATTENTION_CONTEXT_CHUNK_CAP: u64 = {attention_context_chunks_cap};\n"
    ));

    let attention_block_width = require_multiple_of_thirty_two(
        "attention_block.width",
        require_nonzero(
            "attention_block.width",
            resolve_int(&root, "attention_block", "width"),
        ),
    );
    out.push_str(&format!(
        "pub const ATTENTION_BLOCK_WIDTH: u64 = {attention_block_width};\n"
    ));

    let attention_splits_keys_per_split = require_nonzero(
        "attention_splits.keys_per_split",
        resolve_int(&root, "attention_splits", "keys_per_split"),
    );
    out.push_str(&format!(
        "pub const ATTENTION_SPLIT_KEYS_PER_SPLIT: u64 = {attention_splits_keys_per_split};\n"
    ));
    let attention_splits_max = require_nonzero(
        "attention_splits.max",
        resolve_int(&root, "attention_splits", "max"),
    );
    out.push_str(&format!(
        "pub const ATTENTION_SPLIT_MAX: u64 = {attention_splits_max};\n"
    ));

    let workgroup_size = require_nonzero(
        "wgsl.workgroup_size",
        resolve_int(&root, "wgsl", "workgroup_size"),
    );
    out.push_str(&format!(
        "pub const WORKGROUP_SIZE: u32 = {workgroup_size};\n"
    ));

    if env::var_os("CARGO_FEATURE_METAL_PACKED_ROW_NSG2").is_some()
        || env::var_os("CARGO_FEATURE_METAL_Q4K_GGML_PORT").is_some()
    {
        let width = require_power_of_two_le_32(
            "packed_row_nsg.width",
            require_nonzero(
                "packed_row_nsg.width",
                resolve_int(&root, "packed_row_nsg", "width"),
            ),
        );
        out.push_str(&format!("pub const PACKED_ROW_NSG: usize = {width};\n"));
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR set by cargo"));
    let out_path = out_dir.join("omega_sized.rs");
    fs::write(&out_path, out).unwrap_or_else(|err| panic!("write {}: {err}", out_path.display()));
}
