//! The first auto-tune step: derive a checkpoint's own device-memory
//! budget from its SHAPE alone -- weight bytes broken out by CLASS (dense,
//! mixture-of-experts, embedding/output tables), placed-KV bytes at the
//! requested context length, SSM recurrent-state bytes, and a fixed arena
//! allowance -- and compare the total against the host's own reported
//! limit, all BEFORE
//! `crate::generate::LoadedModel::generate_with_serving_config`'s own
//! private `apply_memory_fit_gate` lets `BackendRuntime::new` ask a device
//! for a single buffer.
//!
//! Weight bytes are separated by class, not folded into one blob, because
//! a follow-up feature (per-expert precision as a budget-constrained
//! allocation -- DynaExq's rule: a per-layer high-precision resident set
//! chosen by budget-feasible top-n) needs the expert class isolated from
//! the dense class it selects a subset of, and the table class isolated
//! from both (embedding/output tables are neither a per-layer weight nor a
//! routing target). This module does not implement that selection --
//! [`WeightClassBytes`] and [`MemoryBudget`] only carry the classes
//! separable, so that later step has a real envelope to allocate inside
//! rather than a single number to guess a split from.
//!
//! No new pipe, no new runtime type (guiding-principles principle 1): this
//! is two payload structs ([`WeightClassBytes`], [`MemoryBudget`]) plus
//! pure functions over them and [`HostMemoryLimit`], called once per
//! `generate_with_serving_config` call. [`HostMemoryLimit`] is a plain
//! `u64` pair rather than `omega::metal::SystemMemoryFacts` itself, so this
//! module's arithmetic is testable with a hand-built fake and compiles
//! under `--features std` alone, with no `omega`/`metal` dependency at all
//! -- the `metal`-gated caller in `generate.rs` is the only place a real
//! `omega::metal::system_memory_facts` probe ever produces one.
//!
//! The KV formula mirrors
//! [`crate::generate::LoadedModel::run_decode_loop_placed_kv`]'s own
//! `capacity_even_odd`/`capacity_v` sizing exactly: `block_count` layers,
//! each holding `k_even`+`k_odd`+`v` rows of `kv_heads * head_dim *
//! size_of::<f32>()` bytes per stored position and per k/v half, which
//! collapses to `block_count * kv_heads * head_dim * 8` bytes per position
//! (`proxima-tensor/docs/discipline.md` ROW 391's own per-step counter
//! table cites the same shape). Unlike that function, this module has no
//! prompt length yet at load time, so it prices the WORST case a caller's
//! own `context_length` could reach, not `positions_needed`.

use crate::error::InteropError;

/// Bytes-per-stored-position the placed-KV cache holds for one layer,
/// mirroring [`crate::generate::LoadedModel::run_decode_loop_placed_kv`]'s
/// own `row_bytes_even_odd`/`row_bytes_v` sum (`kv_heads * head_dim * 4`
/// each, for `k_even`, `k_odd`, and `v`).
#[must_use]
pub fn kv_row_bytes(kv_heads: u32, head_dim: u32) -> u64 {
    u64::from(kv_heads) * u64::from(head_dim) * 8
}

/// A checkpoint's own on-disk weight bytes, by class --
/// [`crate::bind::tensor_bytes_by_class`]'s own three-way split
/// (dense/experts/tables) plus the SSM recurrent-state bytes a qwen35
/// hybrid checkpoint's layers hold ([`crate::generate::Qwen35SsmShape`]'s
/// own doc; `0` for every other architecture). Never a context-dependent
/// class -- these bytes are fixed once a checkpoint is chosen, unlike
/// [`MemoryBudget::kv_cache_bytes`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WeightClassBytes {
    /// Every per-layer attention/FFN weight that is neither a
    /// mixture-of-experts stack nor an embedding/output table.
    pub dense_bytes: u64,
    /// `blk.{layer}.{proj}_exps.weight` stacks (llama.cpp's own naming) --
    /// `0` for a dense checkpoint.
    pub expert_bytes: u64,
    /// `token_embd.weight` + `output.weight`.
    pub table_bytes: u64,
    /// A qwen35 hybrid checkpoint's per-layer SSM conv-history plus
    /// recurrent-state bytes, summed across every layer -- `0` for a
    /// non-hybrid (dense or MoE-dense) checkpoint.
    pub ssm_state_bytes: u64,
}

impl WeightClassBytes {
    /// The sum of every class this record carries.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.dense_bytes + self.expert_bytes + self.table_bytes + self.ssm_state_bytes
    }
}

/// A checkpoint's device-memory budget, derived from its SHAPE alone --
/// never a live measurement, since this is computed before any weight
/// upload or forward pass exists to measure. Each field is one class the
/// load path will actually allocate; [`Self::total_bytes`] is what
/// [`fit_context_length`] compares against [`HostMemoryLimit::available_bytes`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryBudget {
    /// [`WeightClassBytes::dense_bytes`], carried through unchanged.
    pub dense_weights_bytes: u64,
    /// [`WeightClassBytes::expert_bytes`], carried through unchanged.
    pub expert_weights_bytes: u64,
    /// [`WeightClassBytes::table_bytes`], carried through unchanged.
    pub table_weights_bytes: u64,
    /// [`kv_row_bytes`] times `block_count` times the context length this
    /// budget was derived for -- the one class that varies with
    /// `context_length`.
    pub kv_cache_bytes: u64,
    /// [`WeightClassBytes::ssm_state_bytes`], carried through unchanged.
    pub ssm_state_bytes: u64,
    /// The fixed `BufferArena` allowance
    /// (`omega::sized::LOAD_TIME_FIT_ARENA_ALLOWANCE_BYTES`) a caller
    /// passes in -- this module never reads `omega` directly (see the
    /// module doc), so the caller resolves the constant and hands it here.
    pub arena_allowance_bytes: u64,
}

impl MemoryBudget {
    /// Derives a budget for `context_length` positions of placed KV cache
    /// across `block_count` layers, `weights`' own per-class byte counts,
    /// and `arena_allowance_bytes` of fixed arena headroom.
    #[must_use]
    pub fn derive(
        weights: WeightClassBytes,
        block_count: u32,
        kv_heads: u32,
        head_dim: u32,
        context_length: u32,
        arena_allowance_bytes: u64,
    ) -> Self {
        let kv_cache_bytes =
            kv_row_bytes(kv_heads, head_dim) * u64::from(block_count) * u64::from(context_length);
        Self {
            dense_weights_bytes: weights.dense_bytes,
            expert_weights_bytes: weights.expert_bytes,
            table_weights_bytes: weights.table_bytes,
            kv_cache_bytes,
            ssm_state_bytes: weights.ssm_state_bytes,
            arena_allowance_bytes,
        }
    }

    /// The sum of every class this budget carries.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.dense_weights_bytes
            + self.expert_weights_bytes
            + self.table_weights_bytes
            + self.kv_cache_bytes
            + self.ssm_state_bytes
            + self.arena_allowance_bytes
    }
}

/// The host limit a [`MemoryBudget`] is compared against -- a plain `u64`
/// pair, not `omega::metal::SystemMemoryFacts` itself (see the module doc
/// for why this module never depends on `omega`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostMemoryLimit {
    /// `min(recommendedMaxWorkingSetSize, hw.memsize)` -- the tighter of
    /// what the device advises and what the host physically has.
    pub limit_bytes: u64,
    /// Bytes reserved for the OS and every other process
    /// (`omega::sized::LOAD_TIME_FIT_OS_HEADROOM_BYTES`'s own doc).
    pub os_headroom_bytes: u64,
}

impl HostMemoryLimit {
    /// `limit_bytes` minus `os_headroom_bytes`, saturating at zero rather
    /// than underflowing when headroom alone exceeds the limit (a
    /// pathological but not impossible config on a tiny device).
    #[must_use]
    pub fn available_bytes(&self) -> u64 {
        self.limit_bytes.saturating_sub(self.os_headroom_bytes)
    }
}

/// What [`fit_context_length`] did to reach a fitting budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitOutcome {
    /// `context_length` fit as requested; no change.
    Fits,
    /// `context_length` alone pushed the budget over the limit -- reduced
    /// to the largest value that fits.
    ReducedContext { from: u32, to: u32 },
}

/// Fits `weights`' own per-class bytes plus placed-KV at
/// `requested_context_length` plus `arena_allowance_bytes` of fixed arena
/// headroom against `limit`.
///
/// Returns the context length to actually serve at (unchanged, or reduced)
/// and how it got there. When even a `context_length` of `1` cannot fit
/// (`weights.total_bytes()` plus the arena allowance alone exceed
/// `limit.available_bytes()`), returns [`InteropError::MemoryBudgetExceeded`]
/// naming every class and number involved -- the caller never uploads a
/// weight in that case.
///
/// # Errors
///
/// [`InteropError::MemoryBudgetExceeded`] when no context length, including
/// `1`, fits.
pub fn fit_context_length(
    weights: WeightClassBytes,
    block_count: u32,
    kv_heads: u32,
    head_dim: u32,
    requested_context_length: u32,
    arena_allowance_bytes: u64,
    limit: HostMemoryLimit,
) -> Result<(u32, FitOutcome), InteropError> {
    let available = limit.available_bytes();
    let fixed_bytes = weights.total_bytes() + arena_allowance_bytes;
    let row_bytes = kv_row_bytes(kv_heads, head_dim) * u64::from(block_count);

    let requested_budget = MemoryBudget::derive(
        weights,
        block_count,
        kv_heads,
        head_dim,
        requested_context_length,
        arena_allowance_bytes,
    );
    if requested_budget.total_bytes() <= available {
        return Ok((requested_context_length, FitOutcome::Fits));
    }

    let exceeded = |kv_cache_bytes: u64| InteropError::MemoryBudgetExceeded {
        dense_weights_bytes: weights.dense_bytes,
        expert_weights_bytes: weights.expert_bytes,
        table_weights_bytes: weights.table_bytes,
        kv_cache_bytes,
        ssm_state_bytes: weights.ssm_state_bytes,
        arena_allowance_bytes,
        limit_bytes: limit.limit_bytes,
        os_headroom_bytes: limit.os_headroom_bytes,
    };

    if fixed_bytes >= available || row_bytes == 0 {
        return Err(exceeded(requested_budget.kv_cache_bytes));
    }

    let max_context_length = (available - fixed_bytes) / row_bytes;
    if max_context_length == 0 {
        return Err(exceeded(requested_budget.kv_cache_bytes));
    }

    let reduced = u32::try_from(max_context_length).unwrap_or(u32::MAX);
    Ok((
        reduced,
        FitOutcome::ReducedContext {
            from: requested_context_length,
            to: reduced,
        },
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{
        FitOutcome, HostMemoryLimit, MemoryBudget, WeightClassBytes, fit_context_length,
        kv_row_bytes,
    };
    use crate::error::InteropError;

    const KV_HEADS: u32 = 2;
    const HEAD_DIM: u32 = 64;
    const BLOCK_COUNT: u32 = 2;
    const ARENA_ALLOWANCE_BYTES: u64 = 28_311_552;
    const OS_HEADROOM_BYTES: u64 = 8_589_934_592;

    fn weights(dense_bytes: u64) -> WeightClassBytes {
        WeightClassBytes {
            dense_bytes,
            expert_bytes: 0,
            table_bytes: 0,
            ssm_state_bytes: 0,
        }
    }

    /// The exact formula `run_decode_loop_placed_kv` derives its own
    /// `capacity_even_odd`/`capacity_v` from: `kv_heads * head_dim * 8`
    /// bytes per stored position, per layer.
    #[test]
    fn kv_row_bytes_matches_the_placed_kv_capacity_formula() {
        assert_eq!(
            kv_row_bytes(KV_HEADS, HEAD_DIM),
            2 * 64 * 8,
            "kv_row_bytes must equal kv_heads * head_dim * 8 (k_even + k_odd + v, f32)"
        );
    }

    #[test]
    fn budget_matches_hand_computed_formula_for_the_fixture_shape() {
        let weights = WeightClassBytes {
            dense_bytes: 700_000,
            expert_bytes: 200_000,
            table_bytes: 100_000,
            ssm_state_bytes: 4_096,
        };
        let context_length = 4096u32;
        let budget = MemoryBudget::derive(
            weights,
            BLOCK_COUNT,
            KV_HEADS,
            HEAD_DIM,
            context_length,
            ARENA_ALLOWANCE_BYTES,
        );
        let expected_kv_cache_bytes =
            u64::from(KV_HEADS) * u64::from(HEAD_DIM) * 8 * u64::from(BLOCK_COUNT) * u64::from(context_length);
        assert_eq!(budget.dense_weights_bytes, weights.dense_bytes);
        assert_eq!(budget.expert_weights_bytes, weights.expert_bytes);
        assert_eq!(budget.table_weights_bytes, weights.table_bytes);
        assert_eq!(budget.ssm_state_bytes, weights.ssm_state_bytes);
        assert_eq!(budget.kv_cache_bytes, expected_kv_cache_bytes);
        assert_eq!(budget.arena_allowance_bytes, ARENA_ALLOWANCE_BYTES);
        assert_eq!(
            budget.total_bytes(),
            weights.total_bytes() + expected_kv_cache_bytes + ARENA_ALLOWANCE_BYTES,
            "total_bytes must be the plain sum of every class"
        );
    }

    /// (a) fits -- generous limit, requested context length unchanged.
    #[test]
    fn fits_within_a_generous_limit_without_reducing_context() {
        let limit = HostMemoryLimit {
            limit_bytes: 64u64 * 1024 * 1024 * 1024,
            os_headroom_bytes: OS_HEADROOM_BYTES,
        };
        let (context_length, outcome) = fit_context_length(
            weights(1_000_000),
            BLOCK_COUNT,
            KV_HEADS,
            HEAD_DIM,
            131_072,
            ARENA_ALLOWANCE_BYTES,
            limit,
        )
        .expect("a 64 GiB limit must fit this fixture's tiny weights and kv cache");
        assert_eq!(context_length, 131_072);
        assert_eq!(outcome, FitOutcome::Fits);
    }

    /// (b) context too large -- reduced to the computed max, reported as
    /// [`FitOutcome::ReducedContext`].
    #[test]
    fn reduces_context_length_to_the_largest_value_that_fits() {
        // limit_bytes - os_headroom_bytes - weights - arena leaves exactly
        // room for 10 rows of kv cache at this shape's row_bytes (2048).
        let row_bytes = kv_row_bytes(KV_HEADS, HEAD_DIM) * u64::from(BLOCK_COUNT);
        let dense_bytes = 1_000_000u64;
        let available_for_test = dense_bytes + ARENA_ALLOWANCE_BYTES + row_bytes * 10;
        let limit = HostMemoryLimit {
            limit_bytes: available_for_test,
            os_headroom_bytes: 0,
        };
        let (context_length, outcome) = fit_context_length(
            weights(dense_bytes),
            BLOCK_COUNT,
            KV_HEADS,
            HEAD_DIM,
            131_072,
            ARENA_ALLOWANCE_BYTES,
            limit,
        )
        .expect("a limit sized for exactly 10 kv rows must still fit at a reduced context");
        assert_eq!(context_length, 10);
        assert_eq!(
            outcome,
            FitOutcome::ReducedContext {
                from: 131_072,
                to: 10
            }
        );
    }

    /// (c) weights alone exceed the limit -- typed error carrying every
    /// class and number.
    #[test]
    fn errors_with_every_class_when_weights_alone_exceed_the_limit() {
        let checkpoint_weights = WeightClassBytes {
            dense_bytes: 60u64 * 1024 * 1024 * 1024,
            expert_bytes: 30u64 * 1024 * 1024 * 1024,
            table_bytes: 10u64 * 1024 * 1024 * 1024,
            ssm_state_bytes: 0,
        };
        let limit = HostMemoryLimit {
            limit_bytes: 1024,
            os_headroom_bytes: 0,
        };
        let error = fit_context_length(
            checkpoint_weights,
            BLOCK_COUNT,
            KV_HEADS,
            HEAD_DIM,
            131_072,
            ARENA_ALLOWANCE_BYTES,
            limit,
        )
        .expect_err("weights bytes alone exceeding the limit must be a typed error, not a load");
        match error {
            InteropError::MemoryBudgetExceeded {
                dense_weights_bytes,
                expert_weights_bytes,
                table_weights_bytes,
                arena_allowance_bytes: reported_arena,
                limit_bytes: reported_limit,
                os_headroom_bytes: reported_headroom,
                ..
            } => {
                assert_eq!(dense_weights_bytes, checkpoint_weights.dense_bytes);
                assert_eq!(expert_weights_bytes, checkpoint_weights.expert_bytes);
                assert_eq!(table_weights_bytes, checkpoint_weights.table_bytes);
                assert_eq!(reported_arena, ARENA_ALLOWANCE_BYTES);
                assert_eq!(reported_limit, 1024);
                assert_eq!(reported_headroom, 0);
            }
            other => panic!("expected MemoryBudgetExceeded, got {other:?}"),
        }
    }
}
