//! The Metal emit A/B switches as one value.
//!
//! Teaching pointer: this is the data form of the `PROXIMA_*` environment reads that
//! `kernel_types_identity.rs` and `emit_and_classify.rs` still perform per call. It follows
//! `proxima_numeric::NumericPolicy`: one tier-1 struct, seeded from [`crate::sized`], with consuming
//! `with_*` setters, and under `std` serde plus `conflaguration::Settings` so
//! [`crate::config::RuntimeConfig`] nests it under the unchanged `PROXIMA_*` spellings. Compose it
//! through `RuntimeConfig::builder().with_emit(..)`, or read the environment once with
//! [`EmitKnobs::from_env`] and pass the value on.

use alloc::string::String;
use alloc::vec::Vec;

use proxima_numeric::float_layout::FloatLayout;

#[cfg(feature = "std")]
use core::convert::Infallible;

/// Which packed-row ops may bake their reduction length as a compiled literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(
    feature = "std",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "snake_case")
)]
pub enum ReductionLiteral {
    /// Every op reads the runtime `u.reduction_total` uniform.
    #[default]
    Off,
    /// Every admitted op bakes its reduction length (`PROXIMA_REDUCTION_LITERAL=1`).
    All,
    /// Only single-token ops bake it (`PROXIMA_REDUCTION_LITERAL=decode`).
    DecodeOnly,
}

/// Every A/B switch the Metal emitter honours, resolved to a value.
///
/// Defaults are the production defaults: the three geometry counts come from
/// [`crate::sized`], the measured-win switches are on, the experiments are off.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(
    feature = "std",
    derive(serde::Serialize, serde::Deserialize, conflaguration::Settings),
    settings(prefix = "PROXIMA"),
    serde(default, deny_unknown_fields)
)]
pub struct EmitKnobs {
    /// Rows one simdgroup folds for non-`Q6_K` packed codecs, such as `Some(8)`; `None` keeps the per-codec default.
    #[cfg_attr(feature = "std", setting(default, resolve_with = "parse_packed_rows"))]
    pub packed_rows: Option<usize>,
    /// Ceiling on simdgroups cooperating on one row group, such as `4`.
    #[cfg_attr(
        feature = "std",
        setting(default_str = "sized", resolve_with = "parse_packed_k_split")
    )]
    pub packed_k_split: u64,
    /// Rows one simdgroup folds when the K split is above 1, such as `2`.
    #[cfg_attr(
        feature = "std",
        setting(default_str = "sized", resolve_with = "parse_packed_rows_ksplit")
    )]
    pub packed_rows_ksplit: usize,
    /// Bytes of a block's low code plane one lane owns, such as `16`; `0` is off.
    #[cfg_attr(
        feature = "std",
        setting(default_str = "sized", resolve_with = "parse_packed_lane_run_bytes")
    )]
    pub packed_lane_run_bytes: usize,
    /// Hoist the `Q4_0` multi-row header decode out of the per-element loop.
    #[cfg_attr(feature = "std", setting(default = false, resolve_with = "parse_off_unless_one"))]
    pub q4_0_multi_row_hoist: bool,
    /// Render the `Q4_0` multi-row pair-lane body.
    #[cfg_attr(feature = "std", setting(default = true, resolve_with = "parse_on_unless_zero"))]
    pub q4_0_multi_row_pair_lane: bool,
    /// Codec spellings kept off the row-blocked matvec, such as `["q4_0", "q8_0"]`.
    #[cfg_attr(feature = "std", setting(default, resolve_with = "parse_codec_list"))]
    pub packed_row_disable: Vec<String>,
    /// Codec spellings kept off the tiled and expert-grouped GEMM, such as `["q4k"]`.
    #[cfg_attr(feature = "std", setting(default, resolve_with = "parse_codec_list"))]
    pub tiled_gemm_disable: Vec<String>,
    /// Admit `Q4_0` onto the tiled GEMM.
    #[cfg_attr(feature = "std", setting(default = true, resolve_with = "parse_on_unless_zero"))]
    pub tiled_gemm_q4_0: bool,
    /// Admit dense batched matmul onto the tiled GEMM.
    #[cfg_attr(feature = "std", setting(default = true, resolve_with = "parse_on_unless_zero"))]
    pub tiled_gemm_dense: bool,
    /// Load the interior activation tile one `float4` per thread.
    #[cfg_attr(feature = "std", setting(default = true, resolve_with = "parse_on_unless_zero"))]
    pub tiled_gemm_wide_act_load: bool,
    /// Alias the epilogue staging onto the weight and activation tiles.
    #[cfg_attr(feature = "std", setting(default = true, resolve_with = "parse_on_unless_zero"))]
    pub tiled_gemm_slim_tgmem: bool,
    /// Activation staging layout, such as `float16`; `None` keeps binary32.
    #[cfg_attr(feature = "std", setting(default, resolve_with = "parse_staging_layout"))]
    pub tiled_gemm_staging: Option<FloatLayout>,
    /// Store interior output tiles straight to device memory.
    #[cfg_attr(feature = "std", setting(default = true, resolve_with = "parse_on_unless_zero"))]
    pub tiled_gemm_direct_store: bool,
    /// Stage weights with a wide cooperative loop.
    #[cfg_attr(feature = "std", setting(default = true, resolve_with = "parse_on_unless_zero"))]
    pub tiled_gemm_wide_weight_stage: bool,
    /// Dispatch the tiled GEMM on a 2D grid.
    #[cfg_attr(feature = "std", setting(default = true, resolve_with = "parse_on_unless_zero"))]
    pub tiled_gemm_grid2d: bool,
    /// Lay the tiles out in the `simdgroup_matrix` multiply layout.
    #[cfg_attr(feature = "std", setting(default = true, resolve_with = "parse_on_unless_zero"))]
    pub tiled_gemm_mm_layout: bool,
    /// Size the tiled GEMM threadgroup memory at dispatch.
    #[cfg_attr(feature = "std", setting(default = true, resolve_with = "parse_on_unless_zero"))]
    pub tiled_gemm_dynamic_tgmem: bool,
    /// Emit literal accumulator indices in the multi-row body.
    #[cfg_attr(feature = "std", setting(default = true, resolve_with = "parse_on_unless_zero"))]
    pub multi_row_unroll: bool,
    /// Narrow the multi-row k loop's index arithmetic to `uint` where it fits.
    #[cfg_attr(feature = "std", setting(default = true, resolve_with = "parse_on_unless_zero"))]
    pub multi_row_index32: bool,
    /// Narrow the coordinate decomposition to `uint` where every extent fits.
    #[cfg_attr(feature = "std", setting(default = true, resolve_with = "parse_on_unless_zero"))]
    pub coord_index32: bool,
    /// Which packed-row ops bake their reduction length.
    #[cfg_attr(feature = "std", setting(default, resolve_with = "parse_reduction_literal"))]
    pub reduction_literal: ReductionLiteral,
    /// Read the cached softmax row bound from the runtime uniforms instead of a literal.
    #[cfg_attr(feature = "std", setting(default = false, resolve_with = "parse_off_unless_one"))]
    pub softmax_runtime_rows: bool,
    /// Escape knob: admit expert sources the classifier would reject. Presence alone enables it, even set to the empty string.
    #[cfg_attr(feature = "std", setting(default = false, resolve_with = "parse_presence"))]
    pub enable_unsafe_metal_expert_sources: bool,
}

impl Default for EmitKnobs {
    fn default() -> Self {
        Self {
            packed_rows: None,
            packed_k_split: crate::sized::PACKED_ROW_KSPLIT_MAX,
            packed_rows_ksplit: crate::sized::PACKED_ROW_KSPLIT_ROWS,
            packed_lane_run_bytes: crate::sized::PACKED_ROW_LANE_RUN_BYTES,
            q4_0_multi_row_hoist: false,
            q4_0_multi_row_pair_lane: true,
            packed_row_disable: Vec::new(),
            tiled_gemm_disable: Vec::new(),
            tiled_gemm_q4_0: true,
            tiled_gemm_dense: true,
            tiled_gemm_wide_act_load: true,
            tiled_gemm_slim_tgmem: true,
            tiled_gemm_staging: None,
            tiled_gemm_direct_store: true,
            tiled_gemm_wide_weight_stage: true,
            tiled_gemm_grid2d: true,
            tiled_gemm_mm_layout: true,
            tiled_gemm_dynamic_tgmem: true,
            multi_row_unroll: true,
            multi_row_index32: true,
            coord_index32: true,
            reduction_literal: ReductionLiteral::Off,
            softmax_runtime_rows: false,
            enable_unsafe_metal_expert_sources: false,
        }
    }
}

impl EmitKnobs {
    /// Set the packed rows-per-simdgroup override.
    #[must_use]
    pub fn with_packed_rows(mut self, value: Option<usize>) -> Self {
        self.packed_rows = value;
        self
    }

    /// Set the K-split ceiling.
    #[must_use]
    pub fn with_packed_k_split(mut self, value: u64) -> Self {
        self.packed_k_split = value;
        self
    }

    /// Set the rows folded per simdgroup when the K split is above 1.
    #[must_use]
    pub fn with_packed_rows_ksplit(mut self, value: usize) -> Self {
        self.packed_rows_ksplit = value;
        self
    }

    /// Set the bytes of the low code plane one lane owns.
    #[must_use]
    pub fn with_packed_lane_run_bytes(mut self, value: usize) -> Self {
        self.packed_lane_run_bytes = value;
        self
    }

    /// Set the `Q4_0` multi-row header-decode hoist.
    #[must_use]
    pub fn with_q4_0_multi_row_hoist(mut self, value: bool) -> Self {
        self.q4_0_multi_row_hoist = value;
        self
    }

    /// Set the `Q4_0` multi-row pair-lane body.
    #[must_use]
    pub fn with_q4_0_multi_row_pair_lane(mut self, value: bool) -> Self {
        self.q4_0_multi_row_pair_lane = value;
        self
    }

    /// Set the codec spellings kept off the row-blocked matvec.
    #[must_use]
    pub fn with_packed_row_disable(mut self, value: Vec<String>) -> Self {
        self.packed_row_disable = value;
        self
    }

    /// Set the codec spellings kept off the tiled GEMM.
    #[must_use]
    pub fn with_tiled_gemm_disable(mut self, value: Vec<String>) -> Self {
        self.tiled_gemm_disable = value;
        self
    }

    /// Set the `Q4_0` tiled GEMM admission.
    #[must_use]
    pub fn with_tiled_gemm_q4_0(mut self, value: bool) -> Self {
        self.tiled_gemm_q4_0 = value;
        self
    }

    /// Set the dense batched tiled GEMM admission.
    #[must_use]
    pub fn with_tiled_gemm_dense(mut self, value: bool) -> Self {
        self.tiled_gemm_dense = value;
        self
    }

    /// Set the vectorized activation-tile load.
    #[must_use]
    pub fn with_tiled_gemm_wide_act_load(mut self, value: bool) -> Self {
        self.tiled_gemm_wide_act_load = value;
        self
    }

    /// Set the aliased epilogue staging.
    #[must_use]
    pub fn with_tiled_gemm_slim_tgmem(mut self, value: bool) -> Self {
        self.tiled_gemm_slim_tgmem = value;
        self
    }

    /// Set the activation staging layout; `None` keeps binary32.
    #[must_use]
    pub fn with_tiled_gemm_staging(mut self, value: Option<FloatLayout>) -> Self {
        self.tiled_gemm_staging = value;
        self
    }

    /// Set the direct device store of interior output tiles.
    #[must_use]
    pub fn with_tiled_gemm_direct_store(mut self, value: bool) -> Self {
        self.tiled_gemm_direct_store = value;
        self
    }

    /// Set the wide cooperative weight staging.
    #[must_use]
    pub fn with_tiled_gemm_wide_weight_stage(mut self, value: bool) -> Self {
        self.tiled_gemm_wide_weight_stage = value;
        self
    }

    /// Set the 2D tiled GEMM grid.
    #[must_use]
    pub fn with_tiled_gemm_grid2d(mut self, value: bool) -> Self {
        self.tiled_gemm_grid2d = value;
        self
    }

    /// Set the multiply tile layout.
    #[must_use]
    pub fn with_tiled_gemm_mm_layout(mut self, value: bool) -> Self {
        self.tiled_gemm_mm_layout = value;
        self
    }

    /// Set the dispatch-sized threadgroup memory.
    #[must_use]
    pub fn with_tiled_gemm_dynamic_tgmem(mut self, value: bool) -> Self {
        self.tiled_gemm_dynamic_tgmem = value;
        self
    }

    /// Set the multi-row literal-index unroll.
    #[must_use]
    pub fn with_multi_row_unroll(mut self, value: bool) -> Self {
        self.multi_row_unroll = value;
        self
    }

    /// Set the multi-row `uint` index narrowing.
    #[must_use]
    pub fn with_multi_row_index32(mut self, value: bool) -> Self {
        self.multi_row_index32 = value;
        self
    }

    /// Set the coordinate `uint` index narrowing.
    #[must_use]
    pub fn with_coord_index32(mut self, value: bool) -> Self {
        self.coord_index32 = value;
        self
    }

    /// Set which ops bake their reduction length.
    #[must_use]
    pub fn with_reduction_literal(mut self, value: ReductionLiteral) -> Self {
        self.reduction_literal = value;
        self
    }

    /// Set the runtime softmax row bound.
    #[must_use]
    pub fn with_softmax_runtime_rows(mut self, value: bool) -> Self {
        self.softmax_runtime_rows = value;
        self
    }

    /// Set the unsafe expert-source escape knob.
    #[must_use]
    pub fn with_enable_unsafe_metal_expert_sources(mut self, value: bool) -> Self {
        self.enable_unsafe_metal_expert_sources = value;
        self
    }

    /// Read every `PROXIMA_*` switch from the process environment once, with the lenient rules
    /// the per-call reads used: a value that does not parse keeps the default, it never errors.
    ///
    /// `std` only; the alloc tier consumes explicit values. The same parse functions back the
    /// `conflaguration::Settings` derive, so [`crate::config::RuntimeConfig`] resolves identically.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn from_env() -> Self {
        let base = Self::default();
        Self {
            packed_rows: read_env("PROXIMA_PACKED_ROWS", parse_packed_rows, base.packed_rows),
            packed_k_split: read_env("PROXIMA_PACKED_K_SPLIT", parse_packed_k_split, base.packed_k_split),
            packed_rows_ksplit: read_env(
                "PROXIMA_PACKED_ROWS_KSPLIT",
                parse_packed_rows_ksplit,
                base.packed_rows_ksplit,
            ),
            packed_lane_run_bytes: read_env(
                "PROXIMA_PACKED_LANE_RUN_BYTES",
                parse_packed_lane_run_bytes,
                base.packed_lane_run_bytes,
            ),
            q4_0_multi_row_hoist: read_env(
                "PROXIMA_Q4_0_MULTI_ROW_HOIST",
                parse_off_unless_one,
                base.q4_0_multi_row_hoist,
            ),
            q4_0_multi_row_pair_lane: read_env(
                "PROXIMA_Q4_0_MULTI_ROW_PAIR_LANE",
                parse_on_unless_zero,
                base.q4_0_multi_row_pair_lane,
            ),
            packed_row_disable: read_env("PROXIMA_PACKED_ROW_DISABLE", parse_codec_list, base.packed_row_disable),
            tiled_gemm_disable: read_env("PROXIMA_TILED_GEMM_DISABLE", parse_codec_list, base.tiled_gemm_disable),
            tiled_gemm_q4_0: read_env("PROXIMA_TILED_GEMM_Q4_0", parse_on_unless_zero, base.tiled_gemm_q4_0),
            tiled_gemm_dense: read_env("PROXIMA_TILED_GEMM_DENSE", parse_on_unless_zero, base.tiled_gemm_dense),
            tiled_gemm_wide_act_load: read_env(
                "PROXIMA_TILED_GEMM_WIDE_ACT_LOAD",
                parse_on_unless_zero,
                base.tiled_gemm_wide_act_load,
            ),
            tiled_gemm_slim_tgmem: read_env(
                "PROXIMA_TILED_GEMM_SLIM_TGMEM",
                parse_on_unless_zero,
                base.tiled_gemm_slim_tgmem,
            ),
            tiled_gemm_staging: read_env("PROXIMA_TILED_GEMM_STAGING", parse_staging_layout, base.tiled_gemm_staging),
            tiled_gemm_direct_store: read_env(
                "PROXIMA_TILED_GEMM_DIRECT_STORE",
                parse_on_unless_zero,
                base.tiled_gemm_direct_store,
            ),
            tiled_gemm_wide_weight_stage: read_env(
                "PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE",
                parse_on_unless_zero,
                base.tiled_gemm_wide_weight_stage,
            ),
            tiled_gemm_grid2d: read_env("PROXIMA_TILED_GEMM_GRID2D", parse_on_unless_zero, base.tiled_gemm_grid2d),
            tiled_gemm_mm_layout: read_env(
                "PROXIMA_TILED_GEMM_MM_LAYOUT",
                parse_on_unless_zero,
                base.tiled_gemm_mm_layout,
            ),
            tiled_gemm_dynamic_tgmem: read_env(
                "PROXIMA_TILED_GEMM_DYNAMIC_TGMEM",
                parse_on_unless_zero,
                base.tiled_gemm_dynamic_tgmem,
            ),
            multi_row_unroll: read_env("PROXIMA_MULTI_ROW_UNROLL", parse_on_unless_zero, base.multi_row_unroll),
            multi_row_index32: read_env("PROXIMA_MULTI_ROW_INDEX32", parse_on_unless_zero, base.multi_row_index32),
            coord_index32: read_env("PROXIMA_COORD_INDEX32", parse_on_unless_zero, base.coord_index32),
            reduction_literal: read_env(
                "PROXIMA_REDUCTION_LITERAL",
                parse_reduction_literal,
                base.reduction_literal,
            ),
            softmax_runtime_rows: read_env(
                "PROXIMA_SOFTMAX_RUNTIME_ROWS",
                parse_off_unless_one,
                base.softmax_runtime_rows,
            ),
            // var_os, not var: the original read counts a non-unicode value as present
            enable_unsafe_metal_expert_sources: std::env::var_os("PROXIMA_ENABLE_UNSAFE_METAL_EXPERT_SOURCES")
                .is_some(),
        }
    }
}

#[cfg(feature = "std")]
fn read_env<T>(name: &str, parse: fn(&str) -> Result<T, Infallible>, fallback: T) -> T {
    std::env::var(name)
        .ok()
        .map_or(fallback, |raw| parse(&raw).unwrap_or_else(|never| match never {}))
}

#[cfg(feature = "std")]
fn parse_on_unless_zero(raw: &str) -> Result<bool, Infallible> {
    Ok(raw.trim() != "0")
}

#[cfg(feature = "std")]
fn parse_off_unless_one(raw: &str) -> Result<bool, Infallible> {
    Ok(raw.trim() == "1")
}

#[cfg(feature = "std")]
fn parse_presence(_raw: &str) -> Result<bool, Infallible> {
    Ok(true)
}

#[cfg(feature = "std")]
fn parse_packed_rows(raw: &str) -> Result<Option<usize>, Infallible> {
    Ok((raw.trim() == "8").then_some(8))
}

#[cfg(feature = "std")]
fn geometry_count(raw: &str) -> Option<usize> {
    raw.trim().parse::<usize>().ok().filter(|count| (1..=8).contains(count))
}

#[cfg(feature = "std")]
fn parse_packed_k_split(raw: &str) -> Result<u64, Infallible> {
    Ok(geometry_count(raw).map_or(crate::sized::PACKED_ROW_KSPLIT_MAX, |split| split as u64))
}

#[cfg(feature = "std")]
fn parse_packed_rows_ksplit(raw: &str) -> Result<usize, Infallible> {
    Ok(geometry_count(raw).unwrap_or(crate::sized::PACKED_ROW_KSPLIT_ROWS))
}

#[cfg(feature = "std")]
fn parse_packed_lane_run_bytes(raw: &str) -> Result<usize, Infallible> {
    Ok(raw
        .trim()
        .parse::<usize>()
        .ok()
        .filter(|bytes| *bytes == 0 || (bytes.is_power_of_two() && *bytes <= 32))
        .unwrap_or(crate::sized::PACKED_ROW_LANE_RUN_BYTES))
}

#[cfg(feature = "std")]
fn parse_codec_list(raw: &str) -> Result<Vec<String>, Infallible> {
    Ok(raw
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(String::from)
        .collect())
}

#[cfg(feature = "std")]
fn parse_staging_layout(raw: &str) -> Result<Option<FloatLayout>, Infallible> {
    Ok(FloatLayout::from_name(raw.trim())
        .filter(|layout| super::layout_render::hardware_type(layout).is_some())
        .filter(|layout| !layout.same_format(&FloatLayout::BF32)))
}

#[cfg(feature = "std")]
fn parse_reduction_literal(raw: &str) -> Result<ReductionLiteral, Infallible> {
    Ok(match raw.trim() {
        "1" => ReductionLiteral::All,
        "decode" => ReductionLiteral::DecodeOnly,
        _ => ReductionLiteral::Off,
    })
}

/// A record with every field moved off its default: the fixture the parity tests and the
/// emit-independence tests (`msl/tests.rs`) share.
#[cfg(test)]
pub(crate) fn every_knob_moved_by_the_builder() -> EmitKnobs {
    let staging = FloatLayout::from_name("float16");
    assert!(staging.is_some(), "float16 is a named layout");
    EmitKnobs::default()
        .with_packed_rows(Some(8))
        .with_packed_k_split(2)
        .with_packed_rows_ksplit(3)
        .with_packed_lane_run_bytes(16)
        .with_q4_0_multi_row_hoist(true)
        .with_q4_0_multi_row_pair_lane(false)
        .with_packed_row_disable(alloc::vec![String::from("q4_0"), String::from("q8_0")])
        .with_tiled_gemm_disable(alloc::vec![String::from("q4k")])
        .with_tiled_gemm_q4_0(false)
        .with_tiled_gemm_dense(false)
        .with_tiled_gemm_wide_act_load(false)
        .with_tiled_gemm_slim_tgmem(false)
        .with_tiled_gemm_staging(staging)
        .with_tiled_gemm_direct_store(false)
        .with_tiled_gemm_wide_weight_stage(false)
        .with_tiled_gemm_grid2d(false)
        .with_tiled_gemm_mm_layout(false)
        .with_tiled_gemm_dynamic_tgmem(false)
        .with_multi_row_unroll(false)
        .with_multi_row_index32(false)
        .with_coord_index32(false)
        .with_reduction_literal(ReductionLiteral::DecodeOnly)
        .with_softmax_runtime_rows(true)
        .with_enable_unsafe_metal_expert_sources(true)
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use core::fmt::Debug;

    use super::super::*;
    use super::{EmitKnobs, ReductionLiteral, every_knob_moved_by_the_builder};
    use crate::config::RuntimeConfig;

    const ENV_NAMES: [&str; 24] = [
        "PROXIMA_PACKED_ROWS",
        "PROXIMA_PACKED_K_SPLIT",
        "PROXIMA_PACKED_ROWS_KSPLIT",
        "PROXIMA_PACKED_LANE_RUN_BYTES",
        "PROXIMA_Q4_0_MULTI_ROW_HOIST",
        "PROXIMA_Q4_0_MULTI_ROW_PAIR_LANE",
        "PROXIMA_PACKED_ROW_DISABLE",
        "PROXIMA_TILED_GEMM_DISABLE",
        "PROXIMA_TILED_GEMM_Q4_0",
        "PROXIMA_TILED_GEMM_DENSE",
        "PROXIMA_TILED_GEMM_WIDE_ACT_LOAD",
        "PROXIMA_TILED_GEMM_SLIM_TGMEM",
        "PROXIMA_TILED_GEMM_STAGING",
        "PROXIMA_TILED_GEMM_DIRECT_STORE",
        "PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE",
        "PROXIMA_TILED_GEMM_GRID2D",
        "PROXIMA_TILED_GEMM_MM_LAYOUT",
        "PROXIMA_TILED_GEMM_DYNAMIC_TGMEM",
        "PROXIMA_MULTI_ROW_UNROLL",
        "PROXIMA_MULTI_ROW_INDEX32",
        "PROXIMA_COORD_INDEX32",
        "PROXIMA_REDUCTION_LITERAL",
        "PROXIMA_SOFTMAX_RUNTIME_ROWS",
        "PROXIMA_ENABLE_UNSAFE_METAL_EXPERT_SOURCES",
    ];

    const COMMON_VALUES: [&str; 4] = ["0", "1", " 1 ", "garbage"];

    const CODECS: [Codec; 5] = [Codec::Q4_0, Codec::Q4K, Codec::Q6K, Codec::Q8_0, Codec::Float16];

    fn with_only<T>(name: &str, value: Option<&str>, body: impl FnOnce() -> T) -> T {
        let variables: Vec<(&str, Option<&str>)> = ENV_NAMES
            .iter()
            .map(|candidate| (*candidate, if *candidate == name { value } else { None }))
            .collect();
        temp_env::with_vars(variables, body)
    }

    fn with_all_unset<T>(body: impl FnOnce() -> T) -> T {
        let variables: Vec<(&str, Option<&str>)> = ENV_NAMES.iter().map(|name| (*name, None)).collect();
        temp_env::with_vars(variables, body)
    }

    fn assert_parity<T: Debug + PartialEq>(
        name: &str,
        extra_values: &[&str],
        from_knobs: fn(&EmitKnobs) -> T,
        current_read: fn() -> T,
    ) {
        let values = core::iter::once(None)
            .chain(COMMON_VALUES.iter().chain(extra_values).map(|value| Some(*value)))
            .collect::<Vec<_>>();
        assert!(values.len() >= 5, "{name}: the value table shrank");
        for value in values {
            let (from_env, current) = with_only(name, value, || (from_knobs(&EmitKnobs::from_env()), current_read()));
            assert_eq!(from_env, current, "{name}={value:?}: from_env disagrees with the current read function");
        }
    }

    fn disabled_by(list: &[String]) -> Vec<bool> {
        CODECS
            .iter()
            .map(|codec| list.iter().any(|name| name.as_str() == codec_cache_key(*codec)))
            .collect()
    }

    #[test]
    fn defaults_equal_the_sized_constants() {
        let knobs = EmitKnobs::default();

        assert_eq!(knobs.packed_k_split, crate::sized::PACKED_ROW_KSPLIT_MAX);
        assert_eq!(knobs.packed_rows_ksplit, crate::sized::PACKED_ROW_KSPLIT_ROWS);
        assert_eq!(knobs.packed_lane_run_bytes, crate::sized::PACKED_ROW_LANE_RUN_BYTES);
        assert_eq!(knobs, with_all_unset(EmitKnobs::from_env));
    }

    #[test]
    fn packed_rows_matches_the_current_read_function() {
        assert_parity(
            "PROXIMA_PACKED_ROWS",
            &["8", " 8 ", "80", ""],
            |knobs| knobs.packed_rows.unwrap_or(PACKED_ROWS_PER_GROUP),
            || codec_rows_per_simdgroup(Codec::Q4_0),
        );
    }

    #[test]
    fn packed_k_split_matches_the_current_read_function() {
        assert_parity(
            "PROXIMA_PACKED_K_SPLIT",
            &["2", "8", "9", "-1", ""],
            |knobs| knobs.packed_k_split,
            ksplit_max,
        );
    }

    #[test]
    fn packed_rows_ksplit_matches_the_current_read_function() {
        assert_parity(
            "PROXIMA_PACKED_ROWS_KSPLIT",
            &["2", "8", "9", "-1", ""],
            |knobs| knobs.packed_rows_ksplit,
            ksplit_rows,
        );
    }

    #[test]
    fn packed_lane_run_bytes_matches_the_current_read_function() {
        assert_parity(
            "PROXIMA_PACKED_LANE_RUN_BYTES",
            &["16", "32", "64", "3", "-1", ""],
            |knobs| knobs.packed_lane_run_bytes,
            lane_run_bytes,
        );
    }

    #[test]
    fn packed_row_disable_matches_the_current_read_function() {
        assert_parity(
            "PROXIMA_PACKED_ROW_DISABLE",
            &["q4_0", " q4_0 , q8_0 ", "q4_0,,", "Q4_0", "q4k,q6k", ""],
            |knobs| disabled_by(&knobs.packed_row_disable),
            || CODECS.iter().map(|codec| packed_row_codec_disabled(*codec)).collect(),
        );
    }

    #[test]
    fn enable_unsafe_metal_expert_sources_matches_the_current_read_function() {
        assert_parity(
            "PROXIMA_ENABLE_UNSAFE_METAL_EXPERT_SOURCES",
            &[""],
            |knobs| knobs.enable_unsafe_metal_expert_sources,
            unsafe_metal_expert_sources_enabled,
        );
    }

    #[test]
    fn an_empty_value_still_enables_the_unsafe_expert_sources() {
        let knobs = with_only("PROXIMA_ENABLE_UNSAFE_METAL_EXPERT_SOURCES", Some(""), EmitKnobs::from_env);

        assert!(knobs.enable_unsafe_metal_expert_sources);
        assert!(!with_all_unset(EmitKnobs::from_env).enable_unsafe_metal_expert_sources);
    }

    #[cfg(feature = "metal-tiled-gemm")]
    #[test]
    fn tiled_gemm_disable_matches_the_current_read_function() {
        assert_parity(
            "PROXIMA_TILED_GEMM_DISABLE",
            &["q4_0", " q4k , q6k ", "q4k,,", "Q4K", "f16", ""],
            |knobs| disabled_by(&knobs.tiled_gemm_disable),
            || CODECS.iter().map(|codec| tiled_gemm_codec_disabled(*codec)).collect(),
        );
    }

    #[cfg(feature = "metal-tiled-gemm")]
    #[test]
    fn tiled_gemm_q4_0_matches_the_current_read_function() {
        assert_parity("PROXIMA_TILED_GEMM_Q4_0", &[""], |knobs| knobs.tiled_gemm_q4_0, tiled_gemm_q4_0_override);
    }

    #[cfg(feature = "metal-tiled-gemm")]
    #[test]
    fn tiled_gemm_dense_matches_the_current_read_function() {
        assert_parity("PROXIMA_TILED_GEMM_DENSE", &[""], |knobs| knobs.tiled_gemm_dense, tiled_gemm_dense_override);
    }

    #[cfg(feature = "metal-tiled-gemm")]
    #[test]
    fn tiled_gemm_wide_act_load_matches_the_current_read_function() {
        assert_parity(
            "PROXIMA_TILED_GEMM_WIDE_ACT_LOAD",
            &[""],
            |knobs| knobs.tiled_gemm_wide_act_load,
            wide_activation_load_override,
        );
    }

    #[cfg(feature = "metal-tiled-gemm")]
    #[test]
    fn tiled_gemm_slim_tgmem_matches_the_current_read_function() {
        assert_parity(
            "PROXIMA_TILED_GEMM_SLIM_TGMEM",
            &[""],
            |knobs| knobs.tiled_gemm_slim_tgmem,
            slim_tgmem_override,
        );
    }

    #[cfg(feature = "metal-tiled-gemm")]
    #[test]
    fn tiled_gemm_staging_matches_the_current_read_function() {
        assert_parity(
            "PROXIMA_TILED_GEMM_STAGING",
            &["float32", "float16", "bfloat16", " float16 ", "Float16", ""],
            |knobs| knobs.tiled_gemm_staging,
            tiled_gemm_staging_layout_override,
        );
    }

    #[cfg(feature = "metal-tiled-gemm")]
    #[test]
    fn tiled_gemm_direct_store_matches_the_current_read_function() {
        assert_parity(
            "PROXIMA_TILED_GEMM_DIRECT_STORE",
            &[""],
            |knobs| knobs.tiled_gemm_direct_store,
            tiled_gemm_direct_store_override,
        );
    }

    #[cfg(feature = "metal-tiled-gemm")]
    #[test]
    fn tiled_gemm_wide_weight_stage_matches_the_current_read_function() {
        assert_parity(
            "PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE",
            &[""],
            |knobs| knobs.tiled_gemm_wide_weight_stage,
            tiled_gemm_wide_weight_stage_override,
        );
    }

    #[cfg(feature = "metal-tiled-gemm")]
    #[test]
    fn tiled_gemm_grid2d_matches_the_current_read_function() {
        assert_parity(
            "PROXIMA_TILED_GEMM_GRID2D",
            &[""],
            |knobs| knobs.tiled_gemm_grid2d,
            tiled_gemm_grid2d_override,
        );
    }

    #[cfg(feature = "metal-tiled-gemm")]
    #[test]
    fn tiled_gemm_mm_layout_matches_the_current_read_function() {
        assert_parity(
            "PROXIMA_TILED_GEMM_MM_LAYOUT",
            &[""],
            |knobs| knobs.tiled_gemm_mm_layout,
            tiled_gemm_mm_layout_override,
        );
    }

    #[cfg(feature = "metal-tiled-gemm")]
    #[test]
    fn tiled_gemm_dynamic_tgmem_matches_the_current_read_function() {
        assert_parity(
            "PROXIMA_TILED_GEMM_DYNAMIC_TGMEM",
            &[""],
            |knobs| knobs.tiled_gemm_dynamic_tgmem,
            tiled_gemm_dynamic_tgmem_override,
        );
    }

    #[cfg(feature = "metal-reduction-literal")]
    #[test]
    fn reduction_literal_matches_the_current_read_functions() {
        assert_parity(
            "PROXIMA_REDUCTION_LITERAL",
            &["decode", " decode ", "Decode", ""],
            |knobs| {
                (
                    knobs.reduction_literal != ReductionLiteral::Off,
                    knobs.reduction_literal == ReductionLiteral::DecodeOnly,
                )
            },
            || (reduction_literal_override(), reduction_literal_decode_only()),
        );
    }

    #[test]
    fn reduction_literal_reads_one_decode_and_nothing_else() {
        let cases = [
            (None, ReductionLiteral::Off),
            (Some("0"), ReductionLiteral::Off),
            (Some("1"), ReductionLiteral::All),
            (Some(" 1 "), ReductionLiteral::All),
            (Some("decode"), ReductionLiteral::DecodeOnly),
            (Some(" decode "), ReductionLiteral::DecodeOnly),
            (Some("garbage"), ReductionLiteral::Off),
        ];

        for (value, expected) in cases {
            let knobs = with_only("PROXIMA_REDUCTION_LITERAL", value, EmitKnobs::from_env);
            assert_eq!(knobs.reduction_literal, expected, "PROXIMA_REDUCTION_LITERAL={value:?}");
        }
    }

    #[test]
    fn every_boolean_switch_follows_its_named_rule() {
        type Field = fn(&EmitKnobs) -> bool;
        let on_unless_zero: [(&str, Field); 13] = [
            ("PROXIMA_Q4_0_MULTI_ROW_PAIR_LANE", |knobs| knobs.q4_0_multi_row_pair_lane),
            ("PROXIMA_TILED_GEMM_Q4_0", |knobs| knobs.tiled_gemm_q4_0),
            ("PROXIMA_TILED_GEMM_DENSE", |knobs| knobs.tiled_gemm_dense),
            ("PROXIMA_TILED_GEMM_WIDE_ACT_LOAD", |knobs| knobs.tiled_gemm_wide_act_load),
            ("PROXIMA_TILED_GEMM_SLIM_TGMEM", |knobs| knobs.tiled_gemm_slim_tgmem),
            ("PROXIMA_TILED_GEMM_DIRECT_STORE", |knobs| knobs.tiled_gemm_direct_store),
            ("PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE", |knobs| knobs.tiled_gemm_wide_weight_stage),
            ("PROXIMA_TILED_GEMM_GRID2D", |knobs| knobs.tiled_gemm_grid2d),
            ("PROXIMA_TILED_GEMM_MM_LAYOUT", |knobs| knobs.tiled_gemm_mm_layout),
            ("PROXIMA_TILED_GEMM_DYNAMIC_TGMEM", |knobs| knobs.tiled_gemm_dynamic_tgmem),
            ("PROXIMA_MULTI_ROW_UNROLL", |knobs| knobs.multi_row_unroll),
            ("PROXIMA_MULTI_ROW_INDEX32", |knobs| knobs.multi_row_index32),
            ("PROXIMA_COORD_INDEX32", |knobs| knobs.coord_index32),
        ];
        let off_unless_one: [(&str, Field); 2] = [
            ("PROXIMA_Q4_0_MULTI_ROW_HOIST", |knobs| knobs.q4_0_multi_row_hoist),
            ("PROXIMA_SOFTMAX_RUNTIME_ROWS", |knobs| knobs.softmax_runtime_rows),
        ];
        let expectations = [(None, true), (Some("0"), false), (Some("1"), true), (Some(" 1 "), true), (Some(" 0 "), false), (Some("garbage"), true), (Some(""), true)];

        for (name, field) in on_unless_zero {
            for (value, expected) in expectations {
                assert_eq!(field(&with_only(name, value, EmitKnobs::from_env)), expected, "{name}={value:?}");
            }
        }
        for (name, field) in off_unless_one {
            for (value, _) in expectations {
                let expected = value.is_some_and(|text| text.trim() == "1");
                assert_eq!(field(&with_only(name, value, EmitKnobs::from_env)), expected, "{name}={value:?}");
            }
        }
    }

    fn every_knob_moved_by_the_environment() -> Vec<(&'static str, Option<&'static str>)> {
        vec![
            ("PROXIMA_PACKED_ROWS", Some("8")),
            ("PROXIMA_PACKED_K_SPLIT", Some("2")),
            ("PROXIMA_PACKED_ROWS_KSPLIT", Some("3")),
            ("PROXIMA_PACKED_LANE_RUN_BYTES", Some("16")),
            ("PROXIMA_Q4_0_MULTI_ROW_HOIST", Some("1")),
            ("PROXIMA_Q4_0_MULTI_ROW_PAIR_LANE", Some("0")),
            ("PROXIMA_PACKED_ROW_DISABLE", Some("q4_0, q8_0")),
            ("PROXIMA_TILED_GEMM_DISABLE", Some("q4k")),
            ("PROXIMA_TILED_GEMM_Q4_0", Some("0")),
            ("PROXIMA_TILED_GEMM_DENSE", Some("0")),
            ("PROXIMA_TILED_GEMM_WIDE_ACT_LOAD", Some("0")),
            ("PROXIMA_TILED_GEMM_SLIM_TGMEM", Some("0")),
            ("PROXIMA_TILED_GEMM_STAGING", Some("float16")),
            ("PROXIMA_TILED_GEMM_DIRECT_STORE", Some("0")),
            ("PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE", Some("0")),
            ("PROXIMA_TILED_GEMM_GRID2D", Some("0")),
            ("PROXIMA_TILED_GEMM_MM_LAYOUT", Some("0")),
            ("PROXIMA_TILED_GEMM_DYNAMIC_TGMEM", Some("0")),
            ("PROXIMA_MULTI_ROW_UNROLL", Some("0")),
            ("PROXIMA_MULTI_ROW_INDEX32", Some("0")),
            ("PROXIMA_COORD_INDEX32", Some("0")),
            ("PROXIMA_REDUCTION_LITERAL", Some("decode")),
            ("PROXIMA_SOFTMAX_RUNTIME_ROWS", Some("1")),
            ("PROXIMA_ENABLE_UNSAFE_METAL_EXPERT_SOURCES", Some("")),
        ]
    }

    #[test]
    fn the_builder_and_the_environment_build_the_same_knobs() {
        let built = every_knob_moved_by_the_builder();
        let loaded = temp_env::with_vars(every_knob_moved_by_the_environment(), EmitKnobs::from_env);

        assert_eq!(built, loaded);
        assert_ne!(built, EmitKnobs::default());
        assert_ne!(built.packed_k_split, EmitKnobs::default().packed_k_split);
        assert_ne!(built.packed_rows_ksplit, EmitKnobs::default().packed_rows_ksplit);
        assert_ne!(built.packed_lane_run_bytes, EmitKnobs::default().packed_lane_run_bytes);
    }

    #[test]
    fn the_settings_derive_resolves_the_same_knobs_as_from_env() {
        let derived = temp_env::with_vars(every_knob_moved_by_the_environment(), || {
            <EmitKnobs as conflaguration::Settings>::from_env().ok()
        });

        assert_eq!(derived, Some(every_knob_moved_by_the_builder()));
        assert_eq!(
            with_all_unset(|| <EmitKnobs as conflaguration::Settings>::from_env().ok()),
            Some(EmitKnobs::default())
        );
    }

    #[test]
    fn runtime_config_nests_the_knobs_under_unchanged_proxima_spellings() {
        let mut variables = every_knob_moved_by_the_environment();
        variables.extend([("OMEGA_ENGINE", None), ("OMEGA_GPU_DRIVER", None)]);

        let loaded = temp_env::with_vars(variables, || RuntimeConfig::from_env().ok());

        assert_eq!(
            loaded.map(|config| config.emit),
            Some(every_knob_moved_by_the_builder())
        );
    }

    #[test]
    fn the_runtime_config_builder_carries_the_knobs() {
        let built = RuntimeConfig::builder()
            .with_emit(every_knob_moved_by_the_builder())
            .build()
            .ok();

        assert_eq!(built.map(|config| config.emit), Some(every_knob_moved_by_the_builder()));
        assert_eq!(RuntimeConfig::default().emit, EmitKnobs::default());
    }

    #[test]
    fn the_knobs_round_trip_through_toml_and_reject_unknown_fields() {
        let built = every_knob_moved_by_the_builder();

        let text = toml::to_string(&built).ok();
        let parsed = text.as_deref().and_then(|text| toml::from_str::<EmitKnobs>(text).ok());
        let unknown = toml::from_str::<EmitKnobs>("not_a_knob = true");

        assert_eq!(parsed, Some(built));
        assert!(unknown.is_err(), "deny_unknown_fields must reject an unknown key");
        assert_eq!(toml::from_str::<EmitKnobs>("").ok(), Some(EmitKnobs::default()));
    }
}
