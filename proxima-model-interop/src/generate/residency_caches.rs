use core::ops::ControlFlow;

use super::*;

pub enum NodeValuesSink<'sink> {
    Discard,
    Collect {
        nodes: &'sink [NodeId],
        steps: &'sink mut Vec<Vec<Vec<f32>>>,
    },
}

#[derive(Debug, PartialEq)]
pub(super) struct NonFiniteNodeValue {
    pub(super) node: NodeId,
    pub(super) index: usize,
    pub(super) value: f32,
    pub(super) shape: Vec<u64>,
}

pub(super) fn first_nonfinite_node_value(
    evaluated: &Evaluated,
    nodes: &[NodeId],
) -> Option<NonFiniteNodeValue> {
    nodes.iter().find_map(|node| {
        let (values, shape) = evaluated.get(*node)?;
        values
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
            .map(|(index, value)| NonFiniteNodeValue {
                node: *node,
                index,
                value: *value,
                shape: shape.to_vec(),
            })
    })
}

impl NodeValuesSink<'_> {
    pub fn nodes(&self) -> &[NodeId] {
        match self {
            Self::Discard => &[],
            Self::Collect { nodes, .. } => nodes,
        }
    }

    pub fn observe(&mut self, evaluated: &Evaluated) -> Result<(), InteropError> {
        let Self::Collect { nodes, steps } = self else {
            return Ok(());
        };
        let mut values = Vec::with_capacity(nodes.len());
        for &node in *nodes {
            let (data, _) = evaluated
                .get(node)
                .ok_or(InteropError::MissingEvaluatedNode { node })?;
            values.push(data.to_vec());
        }
        steps.push(values);
        Ok(())
    }
}

impl LogitsSink<'_> {
    /// `barriers_step` is this step's own `MetalStageTotals::barriers_emitted`
    /// (already read once, snapshot-and-reset, by the caller's `metal_stage`
    /// local) -- callers outside `feature = "instrument", feature = "metal",
    /// target_os = "macos"` pass `0`, matching every run where the counter
    /// itself never exists.
    pub(super) fn observe(&mut self, logits: &[f32], _barriers_step: u64) {
        match self {
            Self::Discard => {}
            Self::Collect(buffer) => buffer.push(logits.to_vec()),
            #[cfg(all(test, feature = "metal"))]
            Self::SumBarriers(total) => **total += _barriers_step,
        }
    }
}

/// This call's growable per-layer key/value cache -- `F32` only:
/// [`apply_serving_config`]'s own gate rejects any other
/// `kv_cache_key_quant`/`kv_cache_value_quant` before [`LoadedModel::call`]
/// ever reaches this loop, so there is no second precision for this type
/// to carry (contrast `bind.rs`'s own `real_openchat_file::LayerCache`,
/// which still probes the rejected `Q8_0` path directly against the
/// tensor seam that gate exists to keep unreachable here).
#[derive(Clone)]
pub(super) struct LayerCache {
    pub(super) k_even: Vec<f32>,
    pub(super) k_odd: Vec<f32>,
    pub(super) v: Vec<f32>,
}

impl LayerCache {
    pub(super) fn new() -> Self {
        Self {
            k_even: Vec::new(),
            k_odd: Vec::new(),
            v: Vec::new(),
        }
    }

    pub(super) fn append(&mut self, even: &[f32], odd: &[f32], value: &[f32]) {
        self.k_even.extend_from_slice(even);
        self.k_odd.extend_from_slice(odd);
        self.v.extend_from_slice(value);
    }

    /// [`Self::append`]'s inverse for speculative decode's KV-rewind: a
    /// verify forward writes K/V for every drafted position, but
    /// [`proxima_tokenizer::draft::speculative_accept_greedy`] only commits
    /// a prefix of them. `keep_positions` is the committed length in
    /// POSITIONS (this layer's `cached_len` after the rewind, not before
    /// this step's append) -- `even_odd_row`/`v_row` are the same per-
    /// position row widths [`KvPadShape`] already carries, so this call
    /// mirrors `append`'s own row-based growth exactly, just shrinking
    /// instead of extending. A no-op when `keep_positions` is not shorter
    /// than what is already cached (nothing to rewind).
    ///
    /// Called from [`super::decode::LoadedModel::run_decode_loop_observed_seeded`]'s
    /// speculative-decode verify branch: a forward over
    /// `[current, draft...]` appends `new_count` positions' worth of K/V,
    /// and this rewinds every layer back to the `verified.accepted + 1`
    /// that survived.
    pub(super) fn truncate(&mut self, keep_positions: usize, even_odd_row: usize, v_row: usize) {
        self.k_even.truncate(keep_positions * even_odd_row);
        self.k_odd.truncate(keep_positions * even_odd_row);
        self.v.truncate(keep_positions * v_row);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod layer_cache_truncate_tests {
    use super::LayerCache;

    /// The KV-rewind shape a speculative-decode verify step needs: a
    /// forward over `[current, draft_1, draft_2, draft_3]` (4 positions)
    /// appends 4 positions' worth of K/V, but only 2 drafts were accepted
    /// (plus the bonus token already sampled from position 2's own row,
    /// which needs no K/V of its own yet) -- rewinding to `keep_positions =
    /// 2` must land exactly where two real `append` calls would have.
    #[test]
    fn truncate_after_partial_accept_matches_incrementally_appended_prefix() {
        let even_odd_row = 3;
        let v_row = 2;
        let positions: [(Vec<f32>, Vec<f32>, Vec<f32>); 4] = [
            (vec![1.0, 1.1, 1.2], vec![1.3, 1.4, 1.5], vec![1.6, 1.7]),
            (vec![2.0, 2.1, 2.2], vec![2.3, 2.4, 2.5], vec![2.6, 2.7]),
            (vec![3.0, 3.1, 3.2], vec![3.3, 3.4, 3.5], vec![3.6, 3.7]),
            (vec![4.0, 4.1, 4.2], vec![4.3, 4.4, 4.5], vec![4.6, 4.7]),
        ];

        let mut incremental = LayerCache::new();
        for (even, odd, value) in positions.iter().take(2) {
            incremental.append(even, odd, value);
        }

        let mut speculative = LayerCache::new();
        let even_batch: Vec<f32> = positions.iter().flat_map(|(even, _, _)| even.clone()).collect();
        let odd_batch: Vec<f32> = positions.iter().flat_map(|(_, odd, _)| odd.clone()).collect();
        let value_batch: Vec<f32> = positions.iter().flat_map(|(_, _, value)| value.clone()).collect();
        speculative.append(&even_batch, &odd_batch, &value_batch);
        speculative.truncate(2, even_odd_row, v_row);

        assert_eq!(speculative.k_even, incremental.k_even);
        assert_eq!(speculative.k_odd, incremental.k_odd);
        assert_eq!(speculative.v, incremental.v);
    }

    /// Every draft accepted (`keep_positions == positions already written`):
    /// truncate is a no-op, the common case when the draft is fully right.
    #[test]
    fn truncate_at_full_length_is_a_no_op() {
        let mut cache = LayerCache::new();
        cache.append(&[1.0, 2.0], &[3.0, 4.0], &[5.0]);
        let before = cache.clone();

        cache.truncate(1, 2, 1);

        assert_eq!(cache.k_even, before.k_even);
        assert_eq!(cache.k_odd, before.k_odd);
        assert_eq!(cache.v, before.v);
    }

    /// Zero accepted (`keep_positions == 0`, the first draft token itself
    /// disagreed): rewinds all the way back to empty.
    #[test]
    fn truncate_to_zero_positions_empties_every_leaf() {
        let mut cache = LayerCache::new();
        cache.append(&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]);

        cache.truncate(0, 2, 2);

        assert!(cache.k_even.is_empty());
        assert!(cache.k_odd.is_empty());
        assert!(cache.v.is_empty());
    }
}

/// [`LayerCache`]'s bucket-padded mirror -- the two-range decode loop's own
/// fix for the plan-cache defect `Self::plans`' own doc on
/// [`BackendRuntime`] walks through: `LayerCache` grows by exactly
/// `cached_len` every step, so a plan keyed on it can never repeat, but the
/// fused `BoundOpKind::CachedAttention` op's own runtime bound
/// (`proxima_tensor::bind::cached_attention_candidates`'s own doc) makes any
/// reader that pads a copy of it out to a `kv_extent` bucket boundary
/// numerically identical to reading the exact, unpadded length. `fill`
/// copies `source`'s real content into a buffer at least `bound_extent` rows
/// long, zero-filling the remainder (never load-bearing -- the runtime bound
/// always excludes it before softmax, see that bound's own doc); `resize`
/// only grows when a step crosses into a new, larger bucket, the same
/// reuse-across-steps shape [`run_decode_loop_placed_kv`]'s own
/// single-range path's placeholder scratch.
pub(super) struct KvPadScratch {
    pub(super) k_even: Vec<f32>,
    pub(super) k_odd: Vec<f32>,
    pub(super) v: Vec<f32>,
}

impl KvPadScratch {
    pub(super) fn new() -> Self {
        Self {
            k_even: Vec::new(),
            k_odd: Vec::new(),
            v: Vec::new(),
        }
    }

    /// # Errors
    ///
    /// [`InteropError::CacheScratchShapeMismatch`] when `source` (this
    /// layer's real, unpadded cache) holds more elements in a leaf than
    /// `shape` sized that leaf's own scratch buffer to -- `shape`'s row
    /// widths come from the bound program's own declared cache-leaf
    /// extents ([`layer_pad_row_widths`]'s own doc), so this only fires
    /// when a foreign bind's program under-declares a leaf its own
    /// [`LayerCache::append`] then over-fills, not on any checkpoint whose
    /// program and cache stay in agreement.
    pub(super) fn fill(
        &mut self,
        source: &LayerCache,
        shape: &KvPadShape,
        layer: usize,
    ) -> Result<(), InteropError> {
        let even_odd_len = shape.even_odd_len();
        let v_len = shape.v_len();
        if self.k_even.len() < even_odd_len {
            self.k_even.resize(even_odd_len, 0.0);
        }
        if self.k_odd.len() < even_odd_len {
            self.k_odd.resize(even_odd_len, 0.0);
        }
        if self.v.len() < v_len {
            self.v.resize(v_len, 0.0);
        }
        copy_into_padded(&mut self.k_even, &source.k_even, layer, "k_even")?;
        copy_into_padded(&mut self.k_odd, &source.k_odd, layer, "k_odd")?;
        copy_into_padded(&mut self.v, &source.v, layer, "v")?;
        Ok(())
    }

    pub(super) fn named_blocks<'cache>(
        &'cache self,
        k_even_name: &'cache str,
        k_odd_name: &'cache str,
        v_name: &'cache str,
        shape: &KvPadShape,
    ) -> [(&'cache str, QuantizedBlock<'cache>); 3] {
        [
            (
                k_even_name,
                QuantizedBlock::Float32(&self.k_even[..shape.even_odd_len()]),
            ),
            (
                k_odd_name,
                QuantizedBlock::Float32(&self.k_odd[..shape.even_odd_len()]),
            ),
            (v_name, QuantizedBlock::Float32(&self.v[..shape.v_len()])),
        ]
    }
}

/// [`KvPadScratch`]'s own row-width parameters, grouped into one reference
/// rather than two positional `usize`s -- every [`KvPadScratch::fill`]/
/// [`KvPadScratch::named_blocks`] call site already computes both together
/// from `kv_bound_extent` and this layer's own [`LayerPadRowWidths`], so
/// one reference says what was already true by convention. `even_odd_row`/
/// `v_row` are each `kv_heads * width` for their own leaf, read back off
/// this layer's declared `Op::Input` shape by [`cache_leaf_row_elements`]
/// -- never derived from `ModelArchitecture` scalars a foreign bind may
/// leave zero/unset (that doc's own paragraph on why).
pub(super) struct KvPadShape {
    pub(super) bound_extent: usize,
    pub(super) even_odd_row: usize,
    pub(super) v_row: usize,
}

impl KvPadShape {
    pub(super) fn even_odd_len(&self) -> usize {
        self.bound_extent * self.even_odd_row
    }

    pub(super) fn v_len(&self) -> usize {
        self.bound_extent * self.v_row
    }
}

/// Copies `source` into `dest`'s own leading rows -- the shared bounds
/// check every [`KvPadScratch::fill`]/[`Qwen35DenseAttentionPadScratch::fill`]
/// leaf copy needs: `dest` was just resized to (at least) `shape`'s own
/// declared row width, so `source` (this layer's real, unpadded cache)
/// fitting inside it is the invariant the whole pad-scratch mechanism
/// depends on. Previously an unchecked `copy_from_slice`, panicking with
/// "range end index out of range" the moment a foreign bind's declared
/// shape undercounted a leaf's true width; now a named, typed error.
///
/// # Errors
///
/// [`InteropError::CacheScratchShapeMismatch`] when `source.len() >
/// dest.len()`.
pub(super) fn copy_into_padded(
    dest: &mut [f32],
    source: &[f32],
    layer: usize,
    leaf: &'static str,
) -> Result<(), InteropError> {
    let expected = dest.len();
    let found = source.len();
    if found > expected {
        return Err(InteropError::CacheScratchShapeMismatch {
            layer,
            leaf,
            expected,
            found,
        });
    }
    dest[..found].copy_from_slice(source);
    Ok(())
}

/// [`LayerCache`]'s 4-wide counterpart for a
/// [`Qwen35LayerRoots::DenseAttention`] layer -- this checkpoint's own
/// partial-rotary gap (`proxima_tensor::spec::append_qwen35_dense_attention_layer`'s
/// own doc) needs a third K component (`k_pass`, the untouched
/// `rotary_dim..attn_head_dim` remainder) alongside the rotated
/// `k_first`/`k_second` halves [`LayerCache`]'s `k_even`/`k_odd` already
/// name for the plain single-section-RoPE checkpoints.
#[derive(Clone)]
pub(super) struct Qwen35DenseAttentionCache {
    pub(super) k_first: Vec<f32>,
    pub(super) k_second: Vec<f32>,
    pub(super) k_pass: Vec<f32>,
    pub(super) v: Vec<f32>,
}

impl Qwen35DenseAttentionCache {
    pub(super) fn new() -> Self {
        Self {
            k_first: Vec::new(),
            k_second: Vec::new(),
            k_pass: Vec::new(),
            v: Vec::new(),
        }
    }

    pub(super) fn append(&mut self, first: &[f32], second: &[f32], pass: &[f32], value: &[f32]) {
        self.k_first.extend_from_slice(first);
        self.k_second.extend_from_slice(second);
        self.k_pass.extend_from_slice(pass);
        self.v.extend_from_slice(value);
    }
}

/// [`KvPadShape`]'s counterpart for a [`Qwen35DenseAttentionCache`] --
/// `k_first`/`k_second` share [`KvPadShape::even_odd_len`]'s row width
/// (both are `pairs`-wide, the same rotary half [`spec::append_qwen35_dense_attention_layer`]'s
/// `k_first_cache`/`k_second_cache` leaves declare), but `k_pass`/`v` are
/// `attn_head_dim`-based, not `head_dim`-based, so they need their own
/// widths rather than reusing [`KvPadShape::v_len`].
pub(super) struct Qwen35DenseAttentionPadShape {
    pub(super) bound_extent: usize,
    pub(super) even_odd_row: usize,
    pub(super) pass_row: usize,
    pub(super) v_row: usize,
}

impl Qwen35DenseAttentionPadShape {
    pub(super) fn even_odd_len(&self) -> usize {
        self.bound_extent * self.even_odd_row
    }

    pub(super) fn pass_len(&self) -> usize {
        self.bound_extent * self.pass_row
    }

    pub(super) fn v_len(&self) -> usize {
        self.bound_extent * self.v_row
    }
}

/// [`KvPadScratch`]'s counterpart for a [`Qwen35LayerRoots::DenseAttention`]
/// layer -- the same defect [`KvPadScratch`]'s own doc names
/// (`cached_len` growing 1:1 with the step index defeats
/// `proxima_tensor::bind::cached_attention_candidates`'s plan-key
/// bucketing unless the bound buffer this layer feeds the backend is padded
/// out to the SAME `kv_extent` boundary the `Attention` arm already pads
/// to) applies here identically: `qwen35_forward_program`'s dense-attention
/// layers declare `k_first_cache`/`k_second_cache`/`k_pass_cache`/`v_cache`
/// on the identical `Extent::Symbolic(1)` slot the `Attention` arm's
/// `kv_cache.{layer}.*` leaves use, so a bucketed `symbols[1]` value only
/// works when EVERY layer's bound buffer -- dense-attention included --
/// is actually that many rows long, zero-padded past the real
/// `cached_len`.
pub(super) struct Qwen35DenseAttentionPadScratch {
    pub(super) k_first: Vec<f32>,
    pub(super) k_second: Vec<f32>,
    pub(super) k_pass: Vec<f32>,
    pub(super) v: Vec<f32>,
}

impl Qwen35DenseAttentionPadScratch {
    pub(super) fn new() -> Self {
        Self {
            k_first: Vec::new(),
            k_second: Vec::new(),
            k_pass: Vec::new(),
            v: Vec::new(),
        }
    }

    /// # Errors
    ///
    /// [`InteropError::CacheScratchShapeMismatch`] when `source` (this
    /// layer's real, unpadded cache) holds more elements in a leaf than
    /// `shape` sized that leaf's own scratch buffer to -- see
    /// [`KvPadScratch::fill`]'s own doc for why this can only fire on a
    /// bind whose program under-declares a leaf its own
    /// [`Qwen35DenseAttentionCache::append`] then over-fills. This is the
    /// exact defect measured on the real `qwen3.6:35b-a3b` checkpoint: a
    /// foreign bind's [`crate::architecture::Architecture::step_state`]
    /// left `attn_head_dim` at the trait default (`Ok(None)`), which used
    /// to size `v`'s scratch to `0` while `source.v` held real data --
    /// this fill no longer reads `attn_head_dim` at all, so that failure
    /// mode is gone; the check stays as the general safety net.
    pub(super) fn fill(
        &mut self,
        source: &Qwen35DenseAttentionCache,
        shape: &Qwen35DenseAttentionPadShape,
        layer: usize,
    ) -> Result<(), InteropError> {
        let even_odd_len = shape.even_odd_len();
        let pass_len = shape.pass_len();
        let v_len = shape.v_len();
        if self.k_first.len() < even_odd_len {
            self.k_first.resize(even_odd_len, 0.0);
        }
        if self.k_second.len() < even_odd_len {
            self.k_second.resize(even_odd_len, 0.0);
        }
        if self.k_pass.len() < pass_len {
            self.k_pass.resize(pass_len, 0.0);
        }
        if self.v.len() < v_len {
            self.v.resize(v_len, 0.0);
        }
        copy_into_padded(&mut self.k_first, &source.k_first, layer, "k_first")?;
        copy_into_padded(&mut self.k_second, &source.k_second, layer, "k_second")?;
        copy_into_padded(&mut self.k_pass, &source.k_pass, layer, "k_pass")?;
        copy_into_padded(&mut self.v, &source.v, layer, "v")?;
        Ok(())
    }

    pub(super) fn named_blocks<'cache>(
        &'cache self,
        k_first_name: &'cache str,
        k_second_name: &'cache str,
        k_pass_name: &'cache str,
        v_name: &'cache str,
        shape: &Qwen35DenseAttentionPadShape,
    ) -> [(&'cache str, QuantizedBlock<'cache>); 4] {
        [
            (
                k_first_name,
                QuantizedBlock::Float32(&self.k_first[..shape.even_odd_len()]),
            ),
            (
                k_second_name,
                QuantizedBlock::Float32(&self.k_second[..shape.even_odd_len()]),
            ),
            (
                k_pass_name,
                QuantizedBlock::Float32(&self.k_pass[..shape.pass_len()]),
            ),
            (v_name, QuantizedBlock::Float32(&self.v[..shape.v_len()])),
        ]
    }
}

/// [`LayerCache`]'s counterpart for a [`Qwen35LayerRoots::Ssm`] layer --
/// `conv_history` is a fixed-size rolling window (the causal conv1d
/// kernel's own left context, `conv_history_len` elements total, oldest row
/// dropped as each new one is appended) rather than [`LayerCache`]'s
/// unbounded grow-forever history; `state` is the gated DeltaNet recurrent
/// state, fully replaced every step (never appended to) because the mixer
/// already folds every past position into it. Both lengths come from
/// [`LayerPadRowWidths::Ssm`] -- the program's own declared
/// `ssm_cache.{layer}.conv_history`/`.state` `Op::Input` shapes
/// ([`cache_leaf_total_elements`]'s own doc on why this, not
/// [`crate::architecture::Architecture::step_state`], is authoritative).
#[derive(Clone)]
pub(super) struct SsmLayerCache {
    pub(super) conv_history: Vec<f32>,
    pub(super) state: Vec<f32>,
}

impl SsmLayerCache {
    pub(super) fn new(conv_history_len: usize, state_len: usize) -> Self {
        Self {
            conv_history: alloc::vec![0.0f32; conv_history_len],
            state: alloc::vec![0.0f32; state_len],
        }
    }

    /// `qkv_mixed_new` is this step's own `new_count`-many freshly computed
    /// `qkv_mixed` rows; `state_new` is the mixer's full replacement state.
    /// Keeps only the most recent `conv_history_len` elements -- older rows
    /// fall out of the causal conv1d kernel's left context and are never
    /// read again.
    pub(super) fn advance(
        &mut self,
        qkv_mixed_new: &[f32],
        state_new: &[f32],
        conv_history_len: usize,
    ) {
        self.advance_conv_history(qkv_mixed_new, conv_history_len);
        if self.state.len() == state_new.len() {
            self.state.copy_from_slice(state_new);
        } else {
            self.state.clear();
            self.state.extend_from_slice(state_new);
        }
    }

    pub(super) fn advance_conv_history(&mut self, qkv_mixed_new: &[f32], conv_history_len: usize) {
        self.conv_history.extend_from_slice(qkv_mixed_new);
        let drop = self.conv_history.len().saturating_sub(conv_history_len);
        if drop != 0 {
            let retained = self.conv_history.len() - drop;
            self.conv_history.copy_within(drop.., 0);
            self.conv_history.truncate(retained);
        }
    }

    pub(super) fn named_blocks<'cache>(
        &'cache self,
        conv_history_name: &'cache str,
        state_name: &'cache str,
    ) -> [(&'cache str, QuantizedBlock<'cache>); 2] {
        [
            (
                conv_history_name,
                QuantizedBlock::Float32(self.conv_history.as_slice()),
            ),
            (state_name, QuantizedBlock::Float32(self.state.as_slice())),
        ]
    }
}

/// [`LayerCache::new`]/[`SsmLayerCache::new`] threaded per forward-program
/// layer, matching [`LoadedModel::layer_roots`]'s own per-layer discriminant
/// -- an attention layer's cache append/readback shape genuinely differs
/// from an ssm layer's, the same reason [`Qwen35LayerRoots`] itself is an
/// enum rather than a fixed-shape tuple.
#[derive(Clone)]
pub(super) enum LayerCacheState {
    Attention(LayerCache),
    DenseAttention(Qwen35DenseAttentionCache),
    Ssm(SsmLayerCache),
    /// gemma4 E2B's cross-layer shared-KV layer
    /// ([`Qwen35LayerRoots::SharedFromLayer`]'s own doc): no state of its
    /// own to grow, fill, or read back -- its `K`/`V` live entirely in the
    /// donor layer's own [`LayerCacheState`] entry.
    SharedFromLayer,
}

/// Diagnostic-only (history-carry bisection step 3): `(element_count,
/// abs_sum)` over every `f32` this layer's cache currently holds -- a cheap
/// stand-in for a real hash that still catches the two failure shapes this
/// bisection is looking for: `element_count` frozen step-over-step means the
/// cache never grew (state leaves fed zeros or the wrong buffer);
/// `abs_sum` frozen while `element_count` grows means new rows are being
/// appended but they are all-zero.
#[cfg(feature = "instrument")]
pub(super) fn layer_cache_checksum(cache: &LayerCacheState) -> (usize, f64) {
    let sum_abs =
        |values: &[f32]| -> f64 { values.iter().map(|value| f64::from(value.abs())).sum() };
    match cache {
        LayerCacheState::Attention(cache) => (
            cache.k_even.len() + cache.k_odd.len() + cache.v.len(),
            sum_abs(&cache.k_even) + sum_abs(&cache.k_odd) + sum_abs(&cache.v),
        ),
        LayerCacheState::DenseAttention(cache) => (
            cache.k_first.len() + cache.k_second.len() + cache.k_pass.len() + cache.v.len(),
            sum_abs(&cache.k_first)
                + sum_abs(&cache.k_second)
                + sum_abs(&cache.k_pass)
                + sum_abs(&cache.v),
        ),
        LayerCacheState::Ssm(cache) => (
            cache.conv_history.len() + cache.state.len(),
            sum_abs(&cache.conv_history) + sum_abs(&cache.state),
        ),
        LayerCacheState::SharedFromLayer => (0, 0.0),
    }
}

/// A cached prefix: the token ids [`LoadedModel::prefill_prefix`] ran one
/// forward pass over, and the per-layer `LayerCacheState` that pass left
/// behind -- the SAME `(ids, layer_caches, cached_len)` triple
/// `run_decode_loop_observed_seeded` already threads through
/// its own two-range decode loop as local bindings on every call, kept
/// alive across calls instead of dropped at function return. This is not a
/// new cache shape: composing it back in
/// ([`LoadedModel::generate_from_prefix`]) is exactly step 0 of
/// [`LoadedModel::generate_with_serving_config`]'s own loop, given a
/// nonzero `cached_len` and a suffix-only `next_ids` to start from rather
/// than the whole prompt at `cached_len == 0`.
///
/// Plain host `Vec<f32>` buffers throughout (`LayerCache`'s own field
/// list) -- the two-range decode path never registers a named,
/// device-resident buffer for the KV cache the way
/// `LoadedModel::resident_names`'s STATIC weights do (only re-uploads it
/// as an ordinary named block every step, `run_decode_loop_observed_seeded`'s
/// own `named_blocks.extend(kv_pad_scratch...)` call). Releasing this state
/// is therefore exactly Rust's own default `Drop` for a `Vec` -- there is
/// no device identity to unregister the way [`LoadedModel`]'s own `Drop`
/// must, so this type carries none.
pub struct PrefixState {
    pub(super) ids: Vec<u32>,
    pub(super) layer_caches: Vec<LayerCacheState>,
    pub(super) cached_len: usize,
}

impl PrefixState {
    /// The number of prompt/generated tokens this state's own KV cache
    /// covers -- [`LoadedModel::generate_from_prefix`]'s own resume point.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cached_len
    }

    /// `true` for a prefix that cached zero tokens -- never actually
    /// produced by [`LoadedModel::prefill_prefix`] against a non-empty
    /// prompt, but a real state a caller could still reach by prefilling
    /// an empty string.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cached_len == 0
    }
}

/// This call's own [`Op::Input`] names for one layer's cache, matching
/// [`LayerCacheState`]'s discriminant one-to-one -- built once per
/// [`LoadedModel::run_decode_loop`] call (never per step) since layer kind
/// and layer index never change within a call.
pub(super) enum LayerCacheNames {
    Attention {
        k_even: String,
        k_odd: String,
        v: String,
    },
    DenseAttention {
        k_first: String,
        k_second: String,
        k_pass: String,
        v: String,
    },
    Ssm {
        conv_history: String,
        state: String,
    },
    /// gemma4 E2B's cross-layer shared-KV layer -- declares no
    /// `Op::Input` leaf at all (`DeclaredCacheKind::SharedFromLayer`'s own
    /// doc), so this variant carries no names to feed at step time.
    SharedFromLayer,
}

/// Which of the three per-layer cache shapes a layer's `Op::Input` leaves
/// actually declare, at `layer` -- [`LayerCacheNames`]/[`LayerCacheState`]
/// are now built FROM this, not from [`Qwen35LayerRoots`]'s own
/// discriminant. A foreign [`crate::architecture::Architecture::bind`] can
/// tag that enum inconsistently with the ops it actually emitted (copy a
/// [`Qwen35DenseAttentionRoots`] tuple into the wrong variant, drop the
/// `k_pass` leaf); the program's own declared leaf names cannot lie about
/// what the decode loop must feed, so they are the single source of truth
/// this type is derived from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DeclaredCacheKind {
    Attention,
    DenseAttention,
    Ssm,
    /// gemma4 E2B's cross-layer shared-KV layer
    /// ([`Qwen35LayerRoots::SharedFromLayer`]'s own doc): this layer
    /// declares NO `kv_cache.{layer}.*`/`ssm_cache.{layer}.*` `Op::Input`
    /// leaves at all, by design -- its `K`/`V` are a donor layer's
    /// already-declared leaves, read a second time in-graph. The one
    /// `DeclaredCacheKind` a layer resolves to when
    /// [`declared_cache_kind`] finds none of the other three shapes AND
    /// `self.layer_roots` says this layer is bound `SharedFromLayer`.
    SharedFromLayer,
}

impl DeclaredCacheKind {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Attention => "kv_cache.{layer}.{k_even,k_odd,v}",
            Self::DenseAttention => "kv_cache.{layer}.{k_first,k_second,k_pass,v}",
            Self::Ssm => "ssm_cache.{layer}.{conv_history,state}",
            Self::SharedFromLayer => "(none -- reads a donor layer's own leaves)",
        }
    }

    /// The `Op::Input` leaf-name templates a program must declare for a
    /// layer bound to this kind -- the exact set
    /// [`declared_layer_cache_names_and_widths`] fills in `{layer}` from,
    /// surfaced verbatim in [`InteropError::LayerCacheLeavesMissing`] so
    /// the error names precisely what the architecture's `bind` failed to
    /// emit. Empty for [`Self::SharedFromLayer`] -- this layer must
    /// declare none, not "declares them differently".
    pub(super) fn expected_leaf_templates(self) -> &'static [&'static str] {
        match self {
            Self::Attention => &[
                "kv_cache.{layer}.k_even",
                "kv_cache.{layer}.k_odd",
                "kv_cache.{layer}.v",
            ],
            Self::DenseAttention => &[
                "kv_cache.{layer}.k_first",
                "kv_cache.{layer}.k_second",
                "kv_cache.{layer}.k_pass",
                "kv_cache.{layer}.v",
            ],
            Self::Ssm => &["ssm_cache.{layer}.conv_history", "ssm_cache.{layer}.state"],
            Self::SharedFromLayer => &[],
        }
    }
}

/// Probes `program_input_names` (every `Op::Input` name this call's own
/// program declares, collected once) for `layer`'s cache leaves -- checks
/// the widest/most specific shape (`DenseAttention`'s 4-wide `k_first`)
/// before the narrower ones so a layer that happens to declare both would
/// still resolve unambiguously. `None` when `layer` declares no recognized
/// cache leaf at all (a non-cached layer, or a name this function's own
/// three prefixes do not cover).
pub(super) fn declared_cache_kind(
    program_input_names: &BTreeSet<&str>,
    layer: usize,
) -> Option<DeclaredCacheKind> {
    if program_input_names.contains(alloc::format!("kv_cache.{layer}.k_first").as_str()) {
        Some(DeclaredCacheKind::DenseAttention)
    } else if program_input_names.contains(alloc::format!("kv_cache.{layer}.k_even").as_str()) {
        Some(DeclaredCacheKind::Attention)
    } else if program_input_names
        .contains(alloc::format!("ssm_cache.{layer}.conv_history").as_str())
    {
        Some(DeclaredCacheKind::Ssm)
    } else {
        None
    }
}

/// [`Qwen35LayerRoots`]'s own discriminant, read back as a
/// [`DeclaredCacheKind`] so it can be compared against
/// [`declared_cache_kind`]'s program-derived answer for the same layer.
pub(super) fn bound_cache_kind(roots: &Qwen35LayerRoots) -> DeclaredCacheKind {
    match roots {
        Qwen35LayerRoots::Attention(_) => DeclaredCacheKind::Attention,
        Qwen35LayerRoots::DenseAttention(_) => DeclaredCacheKind::DenseAttention,
        Qwen35LayerRoots::Ssm { .. } => DeclaredCacheKind::Ssm,
        Qwen35LayerRoots::SharedFromLayer(_) => DeclaredCacheKind::SharedFromLayer,
    }
}

/// `name`'s own declared [`Op::Input`] shape, read back out of the program
/// that emitted it, collapsed to the flat element count one cached
/// position occupies -- every `kv_cache.{layer}.*` leaf is
/// `[Extent::Symbolic(KV_BOUND), heads, width]`
/// (`proxima_tensor::spec`'s `append_qwen35_dense_attention_layer`/
/// `append_mistral_cached_layer` own `input_leaf` calls for these exact
/// names), so the row width [`KvPadShape`]/[`Qwen35DenseAttentionPadShape`]
/// need is the PRODUCT of every extent after the leading symbolic
/// bound-extent slot, not a single dimension. This is the single source of
/// truth those two shapes size their scratch buffers from -- never
/// `ModelArchitecture`/[`crate::architecture::Architecture::step_state`]
/// scalars a foreign bind may leave zero or unset -- the real defect this
/// function replaces: `LoadedModel` used to carry a single model-wide
/// `qwen35_attn_head_dim: Option<u32>`, read from a trait method whose
/// default impl is `Ok(None)`, silently sizing every `DenseAttention`
/// layer's `v`/`k_pass` scratch to zero on any foreign
/// [`crate::architecture::Architecture`] that never overrides it.
///
/// `None` when `name` is not declared at all, or when the program declared
/// it with an unexpected shape (fewer than two dimensions, or a second
/// symbolic extent) -- [`layer_pad_row_widths`] reads either case as row
/// width `0`, which the `fill`-time [`crate::error::InteropError::CacheScratchShapeMismatch`]
/// check then catches as soon as a real, nonzero-length cache actually
/// needs to copy into it.
pub(super) fn cache_leaf_row_elements(program: &[Op], name: &str) -> Option<usize> {
    let shape = program.iter().find_map(|op| match op {
        Op::Input {
            name: Some(leaf_name),
            shape,
            ..
        } if leaf_name == name => Some(shape.as_slice()),
        _ => None,
    })?;
    let (_bound_extent, row_dims) = shape.split_first()?;
    row_dims
        .iter()
        .try_fold(1usize, |product, extent| match extent {
            Extent::Static(value) => Some(product * (*value as usize)),
            Extent::Symbolic(_) => None,
        })
}

/// `name`'s own declared [`Op::Input`] shape, collapsed to its flat total
/// element count -- unlike [`cache_leaf_row_elements`], every dimension
/// counts (an SSM cache leaf like `ssm_cache.{layer}.conv_history`,
/// `[Static(d_conv-1), Static(qkv_dim)]`, has no growing bound-extent slot
/// the way a KV cache leaf does: it is a fixed-size rolling window from the
/// first decode step onward, so the leading dim is real window depth, not
/// something to skip). This is the single source of truth
/// [`layer_pad_row_widths`]'s own `Ssm` arm sizes
/// [`SsmLayerCache::new`]/[`SsmLayerCache::advance`] from -- never
/// [`crate::architecture::Architecture::step_state`]'s `ssm_shape`, whose
/// default impl a foreign architecture leaves `None` (the real defect this
/// function replaces, the `Ssm` sibling of [`cache_leaf_row_elements`]'s own
/// doc on the `DenseAttention` case).
///
/// `None` when `name` is not declared at all, or when the program declared
/// it with any symbolic extent -- [`layer_pad_row_widths`] reads either case
/// as `0`, caught by the same `fill`-time bounds check every other leaf
/// shape mismatch is.
pub(super) fn cache_leaf_total_elements(program: &[Op], name: &str) -> Option<usize> {
    let shape = program.iter().find_map(|op| match op {
        Op::Input {
            name: Some(leaf_name),
            shape,
            ..
        } if leaf_name == name => Some(shape.as_slice()),
        _ => None,
    })?;
    shape
        .iter()
        .try_fold(1usize, |product, extent| match extent {
            Extent::Static(value) => Some(product * (*value as usize)),
            Extent::Symbolic(_) => None,
        })
}

/// [`KvPadShape`]/[`Qwen35DenseAttentionPadShape`]'s own row widths for one
/// layer, read once (`self.program` never changes for the lifetime of a
/// decode call) rather than re-derived from architecture scalars every
/// step -- see [`cache_leaf_row_elements`]'s own doc for why this is the
/// authoritative source.
pub(super) enum LayerPadRowWidths {
    Attention {
        even_odd_row: usize,
        v_row: usize,
    },
    DenseAttention {
        even_odd_row: usize,
        pass_row: usize,
        v_row: usize,
    },
    /// [`SsmLayerCache::new`]/[`SsmLayerCache::advance`]'s own initial and
    /// steady-state window sizes -- the flat element count of
    /// `ssm_cache.{layer}.conv_history`/`.state` as the program itself
    /// declared them ([`cache_leaf_total_elements`]), never
    /// [`crate::architecture::Architecture::step_state`]'s `ssm_shape`.
    Ssm {
        conv_history_len: usize,
        state_len: usize,
    },
    /// gemma4 E2B's cross-layer shared-KV layer -- no leaf, no row width.
    SharedFromLayer,
}

/// Builds [`LayerPadRowWidths`] for one layer from its own
/// [`LayerCacheNames`] leaf names, looking each one's declared shape up in
/// `program`. A leaf [`cache_leaf_row_elements`] cannot resolve (not
/// declared, or an unexpected shape) reads as row width `0` here --
/// deliberately, not a setup-time error: `0` reproduces exactly the
/// starting state a genuinely absent leaf already left this scratch buffer
/// in before this function existed, so the SAME `fill`-time bounds check
/// ([`InteropError::CacheScratchShapeMismatch`]) catches it, at the point
/// the mismatch actually matters, instead of two divergent error paths for
/// what is the same defect.
pub(super) fn layer_pad_row_widths(program: &[Op], names: &LayerCacheNames) -> LayerPadRowWidths {
    match names {
        LayerCacheNames::Attention { k_even, v, .. } => LayerPadRowWidths::Attention {
            even_odd_row: cache_leaf_row_elements(program, k_even).unwrap_or(0),
            v_row: cache_leaf_row_elements(program, v).unwrap_or(0),
        },
        LayerCacheNames::DenseAttention {
            k_first, k_pass, v, ..
        } => LayerPadRowWidths::DenseAttention {
            even_odd_row: cache_leaf_row_elements(program, k_first).unwrap_or(0),
            pass_row: cache_leaf_row_elements(program, k_pass).unwrap_or(0),
            v_row: cache_leaf_row_elements(program, v).unwrap_or(0),
        },
        LayerCacheNames::Ssm {
            conv_history,
            state,
        } => LayerPadRowWidths::Ssm {
            conv_history_len: cache_leaf_total_elements(program, conv_history).unwrap_or(0),
            state_len: cache_leaf_total_elements(program, state).unwrap_or(0),
        },
        LayerCacheNames::SharedFromLayer => LayerPadRowWidths::SharedFromLayer,
    }
}

/// One step's KV-cache `Op::Input` leaves, named and padded off whatever
/// [`LayerCacheNames`]/[`LayerCacheState`]/[`LayerPadRowWidths`] this call's
/// own layers declared -- the ONE place either
/// [`LoadedModel::run_decode_loop_observed_seeded`] (a growing cache,
/// `kv_pad_scratch` reused across steps) or
/// [`LoadedModel::forward_node_values_on_backend`] (a fresh, empty cache,
/// one shot) turns cache state into named blocks, so a foreign
/// architecture's own leaf names (`k_first`/`k_second`/`k_pass` in place of
/// `k_even`/`k_odd`) are fed identically by both callers. Two passes over
/// the same `layer`/`cache` pairing, not one interleaved pass: see this
/// function's own former call-site comment (now here) on why
/// `KvPadScratch::named_blocks`'s borrow of `kv_pad_scratch[layer]` forces
/// fill-then-emit rather than an interleaved loop.
///
/// # Errors
///
/// Whatever [`KvPadScratch::fill`]/[`Qwen35DenseAttentionPadScratch::fill`]
/// can fail with.
pub(super) fn push_kv_named_blocks<'call>(
    cache_names: &'call [LayerCacheNames],
    layer_caches: &'call [LayerCacheState],
    layer_row_widths: &[LayerPadRowWidths],
    kv_bound_extent: usize,
    kv_pad_scratch: &'call mut [KvPadScratch],
    qwen35_dense_pad_scratch: &'call mut [Qwen35DenseAttentionPadScratch],
    named_blocks: &mut Vec<(&'call str, QuantizedBlock<'call>)>,
) -> Result<(), InteropError> {
    for (layer, cache) in layer_caches.iter().enumerate() {
        match (cache, &layer_row_widths[layer]) {
            (
                LayerCacheState::Attention(cache),
                LayerPadRowWidths::Attention {
                    even_odd_row,
                    v_row,
                },
            ) => {
                let shape = KvPadShape {
                    bound_extent: kv_bound_extent,
                    even_odd_row: *even_odd_row,
                    v_row: *v_row,
                };
                kv_pad_scratch[layer].fill(cache, &shape, layer)?;
            }
            (
                LayerCacheState::DenseAttention(cache),
                LayerPadRowWidths::DenseAttention {
                    even_odd_row,
                    pass_row,
                    v_row,
                },
            ) => {
                let shape = Qwen35DenseAttentionPadShape {
                    bound_extent: kv_bound_extent,
                    even_odd_row: *even_odd_row,
                    pass_row: *pass_row,
                    v_row: *v_row,
                };
                qwen35_dense_pad_scratch[layer].fill(cache, &shape, layer)?;
            }
            (LayerCacheState::Ssm(_), LayerPadRowWidths::Ssm { .. }) => {}
            (LayerCacheState::SharedFromLayer, LayerPadRowWidths::SharedFromLayer) => {}
            _ => unreachable!(
                "layer_row_widths built from the same cache_names as layer_caches, in lockstep"
            ),
        }
    }
    for (layer, names) in cache_names.iter().enumerate() {
        match (names, &layer_caches[layer], &layer_row_widths[layer]) {
            (
                LayerCacheNames::Attention { k_even, k_odd, v },
                LayerCacheState::Attention(_),
                LayerPadRowWidths::Attention {
                    even_odd_row,
                    v_row,
                },
            ) => {
                let shape = KvPadShape {
                    bound_extent: kv_bound_extent,
                    even_odd_row: *even_odd_row,
                    v_row: *v_row,
                };
                named_blocks.extend(kv_pad_scratch[layer].named_blocks(k_even, k_odd, v, &shape));
            }
            (
                LayerCacheNames::DenseAttention {
                    k_first,
                    k_second,
                    k_pass,
                    v,
                },
                LayerCacheState::DenseAttention(_),
                LayerPadRowWidths::DenseAttention {
                    even_odd_row,
                    pass_row,
                    v_row,
                },
            ) => {
                let shape = Qwen35DenseAttentionPadShape {
                    bound_extent: kv_bound_extent,
                    even_odd_row: *even_odd_row,
                    pass_row: *pass_row,
                    v_row: *v_row,
                };
                named_blocks.extend(
                    qwen35_dense_pad_scratch[layer]
                        .named_blocks(k_first, k_second, k_pass, v, &shape),
                );
            }
            (
                LayerCacheNames::Ssm {
                    conv_history,
                    state,
                },
                LayerCacheState::Ssm(cache),
                LayerPadRowWidths::Ssm { .. },
            ) => {
                named_blocks.extend(cache.named_blocks(conv_history, state));
            }
            (
                LayerCacheNames::SharedFromLayer,
                LayerCacheState::SharedFromLayer,
                LayerPadRowWidths::SharedFromLayer,
            ) => {}
            _ => unreachable!(
                "cache_names/layer_caches/layer_row_widths built from the same layer_roots, in lockstep"
            ),
        }
    }
    Ok(())
}

/// Every per-call input the cached forward program needs beyond the model
/// weights and the growing key/value cache: `ids_i32`/RoPE `cos`/`sin` for
/// only the `new` positions this call introduces, at their true absolute
/// angle (`start_position`, not 0 -- a generated token's position is
/// `cached_len`, never the start of the sequence), plus the
/// reduce-broadcast `eps` vector sized to match.
pub(super) struct PositionInputs {
    pub(super) ids_i32: Vec<i32>,
    pub(super) epsilon: Vec<f32>,
    pub(super) cos: Vec<f32>,
    pub(super) sin: Vec<f32>,
}

/// Builds the `ids`/`eps`/`rope_cos`/`rope_sin` step inputs every
/// architecture's builtin decode-loop leaves share. `rope_freqs`, when
/// present, is the checkpoint's own per-pair frequency-scaling factor
/// (GGUF `ROPE_FREQS`, `crate::gemma4::bind::Gemma4Arch::rope_freq_factors`)
/// that ggml divides each pair's angle by before taking `cos`/`sin` --
/// gemma4's full/global layers are the only architecture this crate binds
/// one for (`[1.0]*64 + [1e30]*192]` on the real checkpoint: dividing by
/// `1.0` is a no-op for the first 64 pairs, and dividing by `1e30` shrinks
/// `theta` for the remaining 192 pairs to a value so far below one radian
/// that `cos` rounds to exactly `1.0f32` and `sin` rounds to a value
/// indistinguishable from `0.0f32` at any downstream precision -- the same
/// observable effect this function used to get by skipping those pairs
/// outright (`rotary_pairs` truncation, since removed: this is the
/// data-driven replacement, not an additional code path). `None` (every
/// non-gemma4 architecture, and gemma4's own SWA layers via
/// `gemma4_sliding_rope_table`, which never calls this function) leaves
/// every pair's angle undivided -- full rotation, this function's only
/// behaviour before `rope_freqs` existed.
pub(super) fn build_position_inputs(
    new_ids: &[u32],
    start_position: usize,
    head_dim: u32,
    rope_freq_base: f32,
    rms_epsilon: f32,
    rope_freqs: Option<&[f32]>,
) -> PositionInputs {
    let new_count = new_ids.len();
    let pairs = head_dim as usize / 2;
    let ids_i32: Vec<i32> = new_ids.iter().map(|&id| id as i32).collect();
    let epsilon = alloc::vec![rms_epsilon; new_count];

    let mut cos = alloc::vec![1.0f32; new_count * pairs];
    let mut sin = alloc::vec![0.0f32; new_count * pairs];
    for offset in 0..new_count {
        let position = (start_position + offset) as f32;
        for pair in 0..pairs {
            // Qwen3.6 text positions use three MRoPE sections (11, 11, 10
            // pairs). Each section restarts its local frequency index; the
            // graph still consumes one flat table, so only angle generation
            // changes here.
            let frequency_pair = pair;
            let mut theta =
                position * rope_freq_base.powf(-((2 * frequency_pair) as f32) / (head_dim as f32));
            if let Some(factor) = rope_freqs.and_then(|freqs| freqs.get(pair)) {
                theta /= factor;
            }
            cos[offset * pairs + pair] = theta.cos();
            sin[offset * pairs + pair] = theta.sin();
        }
    }

    PositionInputs {
        ids_i32,
        epsilon,
        cos,
        sin,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod rope_freqs_tests {
    use super::build_position_inputs;

    /// gemma4's real checkpoint shape: `head_dim=512` (256 pairs),
    /// `rope_freq_base=1e6`, `rope_freqs.weight = [1.0]*64 + [1e30]*192`.
    /// Confirms the data-driven division path is numerically identical to
    /// the removed `rotary_pairs = 64` truncation this replaces: pairs
    /// `0..64` (factor `1.0`, a no-op divide) carry the same real angle as
    /// an undivided full rotation, and pairs `64..256` (factor `1e30`)
    /// collapse to the truncation's own exact `cos=1, sin=0` identity pass
    /// -- `cos` rounds bit-exact to `1.0f32`, `sin` lands at float noise
    /// (`< 1e-9`) far below any value that changes a downstream matmul.
    #[test]
    fn gemma4_full_layer_rope_freqs_matches_truncation_at_pair_boundary() {
        let head_dim = 512u32;
        let rope_freq_base = 1.0e6_f32;
        let mut rope_freqs = alloc::vec![1.0f32; 64];
        rope_freqs.extend(alloc::vec![1.0e30_f32; 192]);
        let positions = [1u32, 100, 4096];

        let full_rotation = build_position_inputs(&positions, 0, head_dim, rope_freq_base, 1e-5, None);
        let scaled = build_position_inputs(
            &positions,
            0,
            head_dim,
            rope_freq_base,
            1e-5,
            Some(&rope_freqs),
        );

        let pairs = head_dim as usize / 2;
        for offset in 0..positions.len() {
            for pair in 0..64 {
                let index = offset * pairs + pair;
                assert_eq!(
                    scaled.cos[index], full_rotation.cos[index],
                    "factor 1.0 must be a no-op divide for pair {pair} at offset {offset}"
                );
                assert_eq!(
                    scaled.sin[index], full_rotation.sin[index],
                    "factor 1.0 must be a no-op divide for pair {pair} at offset {offset}"
                );
            }
            for pair in 64..pairs {
                let index = offset * pairs + pair;
                assert_eq!(
                    scaled.cos[index], 1.0f32,
                    "factor 1e30 must round cos to exactly 1.0 for pair {pair} at offset {offset}"
                );
                assert!(
                    scaled.sin[index].abs() < 1e-9,
                    "factor 1e30 must collapse sin to float noise for pair {pair} at offset \
                     {offset}, got {}",
                    scaled.sin[index]
                );
            }
        }
    }
}

/// The fully-supported [`ServingConfig`]: every knob [`apply_serving_config`]
/// accepts today, `F32` key/value cache storage (the only precision the
/// cached-attention reduce's shared `kv_heads` axis can cross -- see
/// `bind.rs`'s own `q8_0_quantized_key_value_cache_cannot_cross_the_weight_matmul_quantized_seam`
/// for the gap this sidesteps by construction rather than by luck).
///
/// `gpu_layers` is the caller's own backend pick threaded straight through
/// to [`ServingConfig::gpu_layers`]/[`select_backend`] -- `0` for CPU,
/// [`GPU_LAYERS_ALL`] for Metal (only valid on a `metal`-featured build,
/// `apply_serving_config`'s own gate). `math_mode` is the same kind of
/// pass-through for [`ServingConfig::math_mode`] -- present only on a
/// `metal`-featured macOS build, the same gate that field carries, since
/// [`omega::MathMode`] itself does not exist otherwise. `crate::quality`'s
/// harness is the reason this is a parameter rather than always
/// `ServingConfig::default`'s `Relaxed`: without it, `quality_report` could
/// never measure any math mode other than the compiled-in default (ROW 356).
/// Every other caller here passes [`omega::MathMode::default`] to keep its
/// own behavior exactly as it was before this parameter existed.
/// `pub(crate)` rather than private: `crate::quality`'s reference/variant
/// harness needs the identical fully-supported knob set [`Self::generate`]
/// runs, with only the backend choice left open, so it reuses this function
/// instead of hand-copying its field list.
pub(crate) fn supported_serving_config(
    gpu_layers: i32,
    #[cfg(all(feature = "metal", target_os = "macos"))] math_mode: omega::MathMode,
) -> ServingConfig<'static> {
    ServingConfig {
        kv_cache_key_quant: GgmlType::F32,
        kv_cache_value_quant: GgmlType::F32,
        flash_attention: false,
        batch_size: 0,
        ubatch_size: 0,
        gpu_layers,
        reasoning_budget: 0,
        #[cfg(all(feature = "metal", target_os = "macos"))]
        math_mode,
        ..ServingConfig::default()
    }
}

/// ROW 356's own regression: proves [`supported_serving_config`] actually
/// carries its `math_mode` argument into [`ServingConfig::math_mode`]
/// rather than the `..ServingConfig::default()` tail silently overwriting
/// it -- the exact defect the quality harness had (`generate.rs:1397` built
/// its config with `..ServingConfig::default()` and no `math_mode` field at
/// all, so every quality measurement ran `Relaxed` no matter what
/// `PROXIMA_MATH_MODE` said). Structural, no checkpoint needed: the real
/// end-to-end check is `quality::real_openchat_file::metal_vs_cpu_reports_real_drift`,
/// `#[ignore]`d because it needs a host-local model.
#[cfg(all(test, feature = "metal", target_os = "macos"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
pub(super) mod supported_serving_config_tests {
    use super::supported_serving_config;

    #[test]
    pub(super) fn threads_the_requested_math_mode_into_the_serving_config() {
        let safe_config = supported_serving_config(0, omega::MathMode::Safe);
        let fast_config = supported_serving_config(0, omega::MathMode::Fast);

        assert_eq!(safe_config.math_mode, omega::MathMode::Safe);
        assert_eq!(fast_config.math_mode, omega::MathMode::Fast);
    }
}

/// `ServingConfig::gpu_layers` (`-ngl`, `serving.rs`) is this crate's
/// existing GPU-offload knob, so backend selection reads it rather than a
/// second mechanism -- `0` (cpu-only, [`supported_serving_config`]'s own
/// default) selects [`Engine::Cpu`]; [`GPU_LAYERS_ALL`] (`-ngl all`)
/// selects [`Engine::Gpu`]. `apply_serving_config` rejects every other
/// value before a forward ever runs, so those are the only two this match
/// needs to distinguish.
#[cfg(feature = "metal")]
pub(super) fn select_backend(config: &ServingConfig) -> Engine {
    if config.gpu_layers == GPU_LAYERS_ALL {
        Engine::Gpu
    } else {
        Engine::Cpu
    }
}

// `dequantize_unsupported_metal_weights`/`resolve_packed_block` (the
// per-call "convert this packed weight back to f32 because Metal has no
// unpack kernel for it yet" step) were deleted here: `Q4_K`/`Q5_K`/`Q6_K`
// -- every packed codec this checkpoint's weights actually carry -- now
// all stay packed straight to the GPU (`omega::msl::Q5K_UNPACK_MSL` is the
// row-blocked kernel `Q5_K` was still missing), so the conversion step had
// zero remaining callers. `named_blocks` below pushes `*block` directly,
// the same value this mechanism always resolved to once its lookup found
// nothing to convert.

/// Everything a decode step needs to actually run the program that is
/// backend-specific: which [`Engine`] to run it on, and the reusable state
/// each call to [`Self::evaluate`] persists across steps. Owns the plan
/// cache directly rather than through a trait object -- [`Engine`] is
/// already a closed, non-`dyn` enum (`omega::backend`'s own doc), and this
/// struct's whole job is picking one arm of it once per
/// [`LoadedModel::generate_with_serving_config`] call. This crate never links
/// `wgpu-backend`, so `Engine::Gpu` here always resolves to the Metal driver
/// through [`omega::backend::GpuDriver::for_target`].
/// [`BackendRuntime::math_mode`]/[`BackendRuntime::numeric_policy`]/
/// [`BackendRuntime::dispatch_type`] -- the three `ServingConfig` knobs
/// [`BackendRuntime::build_placed_plan`] applies together, in this order,
/// to every freshly built [`omega::metal::Plan`]. A parameter struct
/// rather than three positional arguments: every caller already copies
/// all three off `self` in one statement before the plan-cache build
/// closure, so one reference at the call site says what was already true
/// by convention, and keeps `build_placed_plan` under clippy's
/// `too_many_arguments` threshold without an `#[allow]`.
#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct PlanNumerics {
    pub(super) math_mode: omega::metal::MathMode,
    pub(super) numeric_policy: proxima_tensor::NumericPolicy,
    pub(super) dispatch_type: omega::metal::DispatchType,
    /// `ServingConfig::plan_time_constants` -- [`BackendRuntime::build_placed_plan`]'s
    /// own doc for how this reaches [`omega::metal::Plan::mark_plan_time_constants_resident`].
    pub(super) plan_time_constants: bool,
    /// `ServingConfig::command_buffer_chunks` -- reaches
    /// [`omega::metal::Plan::command_buffer_chunks`] through
    /// `omega::backend::set_command_buffer_chunks`, applied the same call
    /// site as `plan_time_constants`/`dispatch_type` above.
    pub(super) command_buffer_chunks: u32,
    /// Forwarded to `proxima_tensor::bind::bind_with_fusion`'s
    /// `fuse_cached_attention` argument -- see that function's own doc for
    /// what the bool controls. `true` at every construction site below
    /// except the `metal-fuse-attn-decode` parity probe, which builds one
    /// [`PlanNumerics`] with `false` to render the unfused chain for the
    /// same program/outputs.
    pub(super) fuse_cached_attention: bool,
}

#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
pub(super) struct SegmentMetalBindings<'buffers, 'source> {
    pub(super) input_placements: &'buffers [(NodeId, &'buffers PlacedBuffer, usize)],
    pub(super) output_placements: &'buffers [(NodeId, &'buffers PlacedBuffer, usize)],
    pub(super) expert_sources:
        &'buffers alloc::collections::BTreeMap<NodeId, proxima_tensor::cpu::ExpertSource<'source>>,
}

#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
pub(super) struct Qwen35SsmPlacement<'buffers> {
    pub(super) input_nodes: &'buffers [Option<NodeId>],
    pub(super) buffers: &'buffers [Option<(PlacedBuffer, PlacedBuffer)>],
    pub(super) maximum_layer: Option<usize>,
    pub(super) use_second_as_input: bool,
}

#[cfg(feature = "metal")]
pub(crate) struct BackendRuntime {
    pub(super) engine: Engine,
    /// Warmup deliberately retains the monolithic all-low source snapshot so
    /// the timed prefill can reuse its plans and Metal bindings. The real
    /// prefill flips this off before its first batch; the release boundary
    /// then hands ownership to the bounded DynaExq sources.
    pub(super) retain_monolithic_prefill_sources: bool,
    /// Keyed by `(new_count, kv_bound_extent)` -- the two symbols
    /// `mistral_cached_forward_program`'s cached-attention read extent
    /// resolves against (`Extent::Symbolic(1) == kv_bound_extent`). A
    /// [`Plan`] bakes concrete shapes from those symbols
    /// (`omega::backend::plan_named`'s own doc), and RAW `cached_len` grows
    /// by `new_count` every decode step, so a plan keyed on it directly was
    /// never valid for the next step -- the exact defect ROW 392 measured
    /// (one miss and one fresh `Plan`, with its own device output buffers,
    /// per token). `kv_bound_extent` is `cached_len` rounded up to
    /// `ServingConfig::kv_bucket_tokens` (`generate::kv_extent`'s own doc),
    /// which repeats for every step inside one bucket -- the fused
    /// `BoundOpKind::CachedAttention` op's own runtime bound
    /// (`proxima_tensor::bind::cached_attention_candidates`'s own doc) is
    /// what makes a `Plan` built for that rounded shape numerically correct
    /// for every real `cached_len` the bucket covers, so ordinary autoregressive
    /// decode now hits this cache `bucket_tokens - 1` times out of every
    /// `bucket_tokens` steps instead of never.
    ///
    /// [`Self::resolve_cached_plan`] clears this on every miss instead of
    /// accumulating entries: measured on a real decode before bucketing
    /// landed (`plan_cache_len` / `plan_misses` in `token_breakdown_metal`)
    /// this map grew 1:1 with the step index and `plan_hits` never left 0,
    /// so every step but the first was retaining a `Plan` that could never
    /// be looked up again for the rest of the call -- a Rust-heap leak
    /// (`phys_footprint_bytes` climbed while `omega::metal::current_allocated_size()`
    /// stayed flat over the same steps, proving the growth was not
    /// GPU-side). Clearing on miss keeps exactly the one entry worth
    /// keeping: the bucket a caller is currently inside.
    pub(super) plans: alloc::collections::BTreeMap<(usize, usize, Vec<NodeId>), Plan>,
    /// Plans for the stable pre-gather router/gather partitions. The segment
    /// programs reuse node IDs across layers, so this cache is keyed by the
    /// partition's address and shape rather than the ordinary decode key.
    pub(super) segment_plans:
        alloc::collections::BTreeMap<(usize, usize, usize, Vec<NodeId>), Plan>,
    /// `ServingConfig::math_mode`, read once at construction and narrowed
    /// into every freshly-built [`Plan`] below (`set_math_mode`'s own call
    /// sites) -- a plan-cache hit reuses a `Plan` already carrying it, same
    /// as `resident_names`/`mark_resident` above.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(super) math_mode: omega::metal::MathMode,
    /// `ServingConfig::numeric_policy`, read once at construction and
    /// passed INTO `plan_named`/`plan_named_placed` at build time (never a
    /// post-hoc setter -- `omega::metal::Plan::numeric_policy`'s own doc:
    /// the policy is fixed for a plan's whole life). Ungated, unlike
    /// `math_mode`/`dispatch_type` above: `ServingConfig::numeric_policy`
    /// is present unconditionally (its own doc), and `omega::backend::
    /// plan_named`'s signature now takes it on every engine arm, not only
    /// the Metal one.
    pub(super) numeric_policy: proxima_tensor::NumericPolicy,
    /// `ServingConfig::dispatch_type`, read once at construction and applied
    /// to every freshly-built [`Plan`] below (`set_dispatch_type`'s own call
    /// sites) -- same pattern as `math_mode` immediately above.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(super) dispatch_type: omega::metal::DispatchType,
    /// [`Self::evaluate_with_placements`]'s own plan cache -- same
    /// `(new_count, merged_len)` keying as `plans` above, but holding
    /// `omega::metal::Plan` directly rather than the backend-polymorphic
    /// `omega::backend::Plan` enum, since [`PlacedBuffer`] placement has no
    /// arm for CPU/wgpu and only ever runs against the Metal backend. A
    /// second map rather than a second variant on `plans`'s own `Plan`
    /// enum because the two plan types come from executing two entirely
    /// different programs (two-range vs. single-range) against the same
    /// `(new_count, merged_len)` shape space -- sharing one map would let a
    /// single-range plan satisfy a two-range lookup by coincidence of key.
    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
    pub(super) placed_plans:
        alloc::collections::BTreeMap<(usize, usize, Vec<NodeId>), omega::metal::Plan>,
    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
    pub(super) placed_segment_plans:
        alloc::collections::BTreeMap<(usize, usize, usize, Vec<NodeId>), omega::metal::Plan>,
    pub(crate) plan_hits: usize,
    pub(crate) plan_misses: usize,
    /// `ServingConfig::exact_activations`, read once at construction --
    /// `Self::evaluate`'s `Engine::Cpu` arm plans through
    /// `omega::backend::plan_named_exact` instead of `plan_named` when
    /// this is `true`, so a cross-backend quality harness's CPU reference
    /// carries the same zero activation-quantization error Metal's own
    /// kernels do (see `ServingConfig::exact_activations`'s own doc). No
    /// effect on `Engine::Gpu`: `plan_named_exact` is a no-op identity on
    /// that arm.
    pub(super) exact_activations: bool,
    /// `ServingConfig::plan_time_constants`, read once at construction and
    /// threaded into every freshly-built placed [`omega::metal::Plan`]
    /// through [`PlanNumerics`] -- same pattern as `math_mode`/`dispatch_type`
    /// above, gated the same as `placed_plans` since only
    /// [`Self::build_placed_plan`] reads it.
    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
    pub(super) plan_time_constants: bool,
    /// `ServingConfig::command_buffer_chunks`, read once at construction and
    /// threaded into every freshly-built placed [`omega::metal::Plan`]
    /// through [`PlanNumerics`] -- same pattern as `plan_time_constants`
    /// immediately above.
    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
    pub(super) command_buffer_chunks: u32,
}

#[cfg(feature = "metal")]
impl BackendRuntime {
    pub(crate) fn new(config: &ServingConfig) -> Self {
        Self {
            engine: select_backend(config),
            retain_monolithic_prefill_sources: false,
            plans: alloc::collections::BTreeMap::new(),
            segment_plans: alloc::collections::BTreeMap::new(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            math_mode: config.math_mode,
            numeric_policy: config.numeric_policy,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            dispatch_type: config.dispatch_type,
            #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
            placed_plans: alloc::collections::BTreeMap::new(),
            #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
            placed_segment_plans: alloc::collections::BTreeMap::new(),
            plan_hits: 0,
            plan_misses: 0,
            exact_activations: config.exact_activations,
            #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
            plan_time_constants: config.plan_time_constants,
            #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
            command_buffer_chunks: config.command_buffer_chunks,
        }
    }

    /// Whether this call's [`ServingConfig`] selected the Gpu engine --
    /// [`Self::engine`] is private (this struct's whole job is hiding which
    /// arm was picked), so [`Self::run_decode_loop`]'s own choice of the
    /// placed-KV decode path against [`LoadedModel::single_range`] needs
    /// this accessor rather than reading the field directly.
    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
    pub(crate) fn is_metal(&self) -> bool {
        matches!(self.engine, Engine::Gpu)
    }

    #[cfg(feature = "metal")]
    pub(super) fn uses_gpu(&self) -> bool {
        matches!(self.engine, Engine::Gpu)
    }

    /// `resident_names` -- the caller's own model-weight names, fixed for
    /// the whole [`LoadedModel::generate_with_serving_config`] call -- is
    /// handed to [`mark_resident`] exactly once per distinct [`Plan`] (right
    /// after it is built, never on a cache hit, since a hit reuses the SAME
    /// `Plan` object that was already marked). See `omega::metal::Plan::mark_resident`'s
    /// own doc for why this needs a name set the tensor program itself
    /// cannot derive.
    // A modified expert table crosses the same backend boundary on every
    // engine. CPU consumes it; a GPU without expert-source bindings returns
    // a typed error instead of silently reading the original full stack.
    pub(super) fn evaluate(
        &mut self,
        program: &[Op],
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
        resident_names: &BTreeSet<&str>,
        expert_sources: Option<
            &alloc::collections::BTreeMap<NodeId, proxima_tensor::cpu::ExpertSource<'_>>,
        >,
    ) -> Result<Evaluated, InteropError> {
        let shape = (symbols[0] as usize, symbols[1] as usize, outputs.to_vec());
        let exact_activations = self.exact_activations;
        let plan = Self::resolve_cached_plan(
            &mut self.plans,
            &mut self.plan_hits,
            &mut self.plan_misses,
            shape,
            || {
                let mut plan = if exact_activations {
                    plan_named_exact(
                        self.engine,
                        None,
                        program,
                        symbols,
                        named,
                        outputs,
                        self.numeric_policy,
                    )?
                } else {
                    plan_named(
                        self.engine,
                        None,
                        program,
                        symbols,
                        named,
                        outputs,
                        self.numeric_policy,
                    )?
                };
                mark_resident(&mut plan, resident_names);
                if self.plan_time_constants {
                    mark_plan_time_constants_resident(&mut plan);
                }
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    set_math_mode(&mut plan, self.math_mode)?;
                    // The backend-polymorphic executor historically opens a
                    // serial Metal encoder. Its stable-buffer implementation
                    // must preserve that ordering: the concurrent hazard
                    // schedule is proven only for the placed single-range
                    // program, not hybrid recurrent graphs.
                    set_dispatch_type(&mut plan, self.dispatch_type);
                    set_command_buffer_chunks(
                    &mut plan,
                    self.command_buffer_chunks,
                    symbols.first().copied() == Some(1),
                );
                }
                Ok(plan)
            },
        )?;
        Ok(execute_plan_named_with_expert_sources(
            plan,
            named,
            expert_sources,
        )?)
    }

    /// Evaluates one graph partition without allowing the ordinary
    /// shape-only decode-plan cache to alias a different partition having
    /// the same `(new_count, kv_bound_extent)` pair. Routed execution uses
    /// this for the router and gather programs surrounding one layer.
    pub(crate) fn evaluate_segment(
        &mut self,
        program: &[Op],
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
        resident_names: &BTreeSet<&str>,
        expert_sources: Option<
            &alloc::collections::BTreeMap<NodeId, proxima_tensor::cpu::ExpertSource<'_>>,
        >,
    ) -> Result<Evaluated, InteropError> {
        let host_timing = std::env::var_os("PROXIMA_DEBUG_SEGMENT_HOST").is_some();
        let resolve_started = std::time::Instant::now();
        let program_key = program.as_ptr() as usize;
        let new_count = symbols.first().copied().unwrap_or_default() as usize;
        let kv_bound_extent = symbols.get(1).copied().unwrap_or_default() as usize;
        let exact_activations = self.exact_activations;
        if std::env::var_os("PROXIMA_DEBUG_GDN_COMPARE").is_some()
            && program
                .iter()
                .any(|operation| operation.name() == Some("gdn_prefill.0.delta_out"))
            && let Ok(shapes) = proxima_tensor::infer(program, symbols)
            && let Ok(bound) = proxima_tensor::bind(program, &shapes, outputs, self.numeric_policy)
        {
            eprintln!("gdn_tail_bound {bound:#?}");
        }
        let plan = Self::resolve_segment_plan(
            &mut self.segment_plans,
            &mut self.plan_hits,
            &mut self.plan_misses,
            (program_key, new_count, kv_bound_extent, outputs.to_vec()),
            || {
                let mut plan = if exact_activations {
                    plan_named_exact(
                        self.engine,
                        None,
                        program,
                        symbols,
                        named,
                        outputs,
                        self.numeric_policy,
                    )?
                } else {
                    plan_named(
                        self.engine,
                        None,
                        program,
                        symbols,
                        named,
                        outputs,
                        self.numeric_policy,
                    )?
                };
                mark_resident(&mut plan, resident_names);
                if self.plan_time_constants {
                    mark_plan_time_constants_resident(&mut plan);
                }
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    set_math_mode(&mut plan, self.math_mode)?;
                    set_dispatch_type(&mut plan, self.dispatch_type);
                    set_command_buffer_chunks(
                    &mut plan,
                    self.command_buffer_chunks,
                    symbols.first().copied() == Some(1),
                );
                }
                Ok(plan)
            },
        )?;
        let resolve_elapsed_us = resolve_started.elapsed().as_micros();
        #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
        let device_before = if host_timing {
            omega::metal::current_allocated_size().unwrap_or_default()
        } else {
            0
        };
        let execute_started = std::time::Instant::now();
        let result = execute_plan_named_with_expert_sources(plan, named, expert_sources)
            .map_err(InteropError::from);
        if host_timing {
            eprintln!(
                "qwen35 segment host resolve_us={} execute_us={} plan_hits={} plan_misses={}",
                resolve_elapsed_us,
                execute_started.elapsed().as_micros(),
                self.plan_hits,
                self.plan_misses,
            );
            #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
            {
                let stage = metal_stage_totals();
                eprintln!(
                    "qwen35 segment metal prepare_ms={:.3} emit_ms={:.3} pipeline_lookup_ms={:.3} op_setup_ms={:.3} gpu_exec_ms={:.3} encode_dispatch_ms={:.3} readback_ms={:.3} block_upload_ms={:.3} device_before={} device_after={} device_delta={}",
                    ticks_to_nanos(stage.prepare_ticks) as f64 / 1_000_000.0,
                    ticks_to_nanos(stage.emit_ticks) as f64 / 1_000_000.0,
                    ticks_to_nanos(stage.pipeline_lookup_ticks) as f64 / 1_000_000.0,
                    ticks_to_nanos(stage.op_setup_ticks) as f64 / 1_000_000.0,
                    ticks_to_nanos(stage.gpu_exec_ticks) as f64 / 1_000_000.0,
                    ticks_to_nanos(stage.encode_dispatch_ticks) as f64 / 1_000_000.0,
                    ticks_to_nanos(stage.readback_ticks) as f64 / 1_000_000.0,
                    ticks_to_nanos(stage.block_upload_ticks) as f64 / 1_000_000.0,
                    device_before,
                    omega::metal::current_allocated_size().unwrap_or_default(),
                    omega::metal::current_allocated_size()
                        .unwrap_or_default()
                        .saturating_sub(device_before),
                );
            }
        }
        result
    }

    /// Diagnostic twin of [`Self::evaluate_segment`] for a routed gather.
    /// It resolves the identical cached segment plan and preserves its
    /// per-expert source substitutions, changing only command-buffer
    /// granularity so the caller can attribute GPU time to bound ops.
    #[cfg(all(feature = "instrument", target_os = "macos"))]
    pub(super) fn evaluate_segment_op_timed(
        &mut self,
        program: &[Op],
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
        resident_names: &BTreeSet<&str>,
        expert_sources: &alloc::collections::BTreeMap<
            NodeId,
            proxima_tensor::cpu::ExpertSource<'_>,
        >,
    ) -> Result<(Evaluated, Vec<OpGpuTiming>), InteropError> {
        let program_key = program.as_ptr() as usize;
        let new_count = symbols.first().copied().unwrap_or_default() as usize;
        let kv_bound_extent = symbols.get(1).copied().unwrap_or_default() as usize;
        let exact_activations = self.exact_activations;
        let mut profile_outputs = outputs.to_vec();
        for selected in [
            "PROXIMA_METAL_COMPARE_BOUND_NODE",
            "PROXIMA_METAL_MATERIALIZE_BOUND_NODE",
        ]
        .into_iter()
        .filter_map(|name| {
            std::env::var(name)
                .ok()
                .and_then(|value| value.parse::<u32>().ok())
                .map(NodeId)
        }) {
            if program
                .get(selected.0 as usize)
                .is_some_and(|operation| !matches!(operation, Op::Input { .. }))
                && !profile_outputs.contains(&selected)
            {
                profile_outputs.push(selected);
            }
        }
        let plan = Self::resolve_segment_plan(
            &mut self.segment_plans,
            &mut self.plan_hits,
            &mut self.plan_misses,
            (
                program_key,
                new_count,
                kv_bound_extent,
                profile_outputs.clone(),
            ),
            || {
                let mut plan = if exact_activations {
                    plan_named_exact(
                        self.engine,
                        None,
                        program,
                        symbols,
                        named,
                        &profile_outputs,
                        self.numeric_policy,
                    )?
                } else {
                    plan_named(
                        self.engine,
                        None,
                        program,
                        symbols,
                        named,
                        &profile_outputs,
                        self.numeric_policy,
                    )?
                };
                mark_resident(&mut plan, resident_names);
                if self.plan_time_constants {
                    mark_plan_time_constants_resident(&mut plan);
                }
                set_math_mode(&mut plan, self.math_mode)?;
                set_dispatch_type(&mut plan, omega::metal::DispatchType::Serial);
                set_command_buffer_chunks(
                    &mut plan,
                    self.command_buffer_chunks,
                    symbols.first().copied() == Some(1),
                );
                Ok(plan)
            },
        )?;
        let cpu_reference = std::env::var("PROXIMA_METAL_COMPARE_BOUND_NODE")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .map(NodeId)
            .and_then(|selected| {
                let mut scratch = Vec::new();
                let mut validated = None;
                let expected = evaluate_quantized_named_exact_with_scratch_and_experts(
                    program,
                    symbols,
                    named,
                    &profile_outputs,
                    &mut scratch,
                    &mut validated,
                    Some(expert_sources),
                )
                .ok()?;
                let (values, _) = expected.get(selected)?;
                let mut reference = BTreeMap::new();
                reference.insert(selected, values.to_vec());
                Some(reference)
            });
        Ok(execute_plan_named_metal_op_timed_with_expert_sources(
            plan,
            named,
            expert_sources,
            cpu_reference.as_ref(),
        )?)
    }

    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
    pub(super) fn evaluate_segment_with_placements_and_expert_sources(
        &mut self,
        program: &[Op],
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
        resident_names: &BTreeSet<&str>,
        bindings: &SegmentMetalBindings<'_, '_>,
    ) -> Result<Evaluated, InteropError> {
        let program_key = program.as_ptr() as usize;
        let new_count = symbols.first().copied().unwrap_or_default() as usize;
        let kv_bound_extent = symbols.get(1).copied().unwrap_or_default() as usize;
        let numerics = PlanNumerics {
            math_mode: self.math_mode,
            numeric_policy: self.numeric_policy,
            dispatch_type: self.dispatch_type,
            plan_time_constants: self.plan_time_constants,
            command_buffer_chunks: self.command_buffer_chunks,
            fuse_cached_attention: true,
        };
        let plan = Self::resolve_segment_plan(
            &mut self.placed_segment_plans,
            &mut self.plan_hits,
            &mut self.plan_misses,
            (program_key, new_count, kv_bound_extent, outputs.to_vec()),
            || {
                Self::build_placed_plan(
                    program,
                    symbols,
                    named,
                    outputs,
                    resident_names,
                    bindings.input_placements,
                    &numerics,
                )
            },
        )?;
        Ok(execute_plan_named_with_placements_and_expert_sources(
            plan,
            named,
            bindings.input_placements,
            bindings.output_placements,
            bindings.expert_sources,
        )?)
    }

    #[cfg(all(
        feature = "metal-output-placement",
        feature = "instrument",
        target_os = "macos"
    ))]
    pub(super) fn evaluate_segment_op_timed_with_placements(
        &mut self,
        program: &[Op],
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
        resident_names: &BTreeSet<&str>,
        placements: SegmentPlacements<'_>,
    ) -> Result<(Evaluated, Vec<OpGpuTiming>), InteropError> {
        let SegmentPlacements {
            input_placements,
            output_placements,
        } = placements;
        let shape = (
            program.as_ptr() as usize,
            symbols.first().copied().unwrap_or_default() as usize,
            symbols.get(1).copied().unwrap_or_default() as usize,
            outputs.to_vec(),
        );
        let numerics = PlanNumerics {
            math_mode: self.math_mode,
            numeric_policy: self.numeric_policy,
            dispatch_type: omega::metal::DispatchType::Serial,
            plan_time_constants: self.plan_time_constants,
            command_buffer_chunks: self.command_buffer_chunks,
            fuse_cached_attention: true,
        };
        let plan = Self::resolve_segment_plan(
            &mut self.placed_segment_plans,
            &mut self.plan_hits,
            &mut self.plan_misses,
            shape,
            || {
                Self::build_placed_plan(
                    program,
                    symbols,
                    named,
                    outputs,
                    resident_names,
                    input_placements,
                    &numerics,
                )
            },
        )?;
        Ok(execute_plan_named_with_placements_op_timed(
            plan,
            named,
            input_placements,
            output_placements,
        )?)
    }

    /// [`Self::evaluate`]'s placed-KV counterpart: same `(new_count,
    /// merged_len)` plan-cache bookkeeping (`plan_hits`/`plan_misses` stay
    /// meaningful across both paths -- a caller reading them after the loop
    /// cannot tell which one ran), but plans and executes directly against
    /// `omega::metal` (`plan_named_placed`/[`execute_plan_named_with_placements`])
    /// rather than through `omega::backend`'s polymorphic entry point, and
    /// routes `input_placements`/`output_placements` into the execute call
    /// -- the whole reason this method exists next to [`Self::evaluate`]
    /// rather than adding a placement parameter there, since every other
    /// backend arm has no such parameter to accept. Clears `placed_plans`
    /// on every miss, same as `plans`/[`Self::resolve_cached_plan`] -- the
    /// identical superseded-`Plan` leak that clearing fixed there applies
    /// here unchanged: ordinary decode's `merged_len` strictly increases,
    /// so a miss means the previous entry can never be looked up again.
    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn evaluate_with_placements(
        &mut self,
        program: &[Op],
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
        resident_names: &BTreeSet<&str>,
        input_placements: &[(NodeId, &PlacedBuffer, usize)],
        output_placements: &[(NodeId, &PlacedBuffer, usize)],
        expert_sources: Option<&BTreeMap<NodeId, proxima_tensor::cpu::ExpertSource<'_>>>,
    ) -> Result<Evaluated, InteropError> {
        let shape = (symbols[0] as usize, symbols[1] as usize, outputs.to_vec());
        let numerics = PlanNumerics {
            math_mode: self.math_mode,
            numeric_policy: self.numeric_policy,
            dispatch_type: self.dispatch_type,
            plan_time_constants: self.plan_time_constants,
            command_buffer_chunks: self.command_buffer_chunks,
            fuse_cached_attention: true,
        };
        let plan = Self::resolve_cached_plan(
            &mut self.placed_plans,
            &mut self.plan_hits,
            &mut self.plan_misses,
            shape,
            || {
                Self::build_placed_plan(
                    program,
                    symbols,
                    named,
                    outputs,
                    resident_names,
                    input_placements,
                    &numerics,
                )
            },
        )?;
        if let Some(expert_sources) = expert_sources {
            Ok(execute_plan_named_with_placements_and_expert_sources(
                plan,
                named,
                input_placements,
                output_placements,
                expert_sources,
            )?)
        } else {
            Ok(execute_plan_named_with_placements(
                plan,
                named,
                input_placements,
                output_placements,
            )?)
        }
    }

    /// Every [`Self::placed_plans`] build closure's shared body -- the class
    /// fix for the defect ROW 329's slice found: `evaluate_op_timed_with_placements`
    /// and `evaluate_dispatch_timed_with_placements` used to build their own
    /// `plan_named_placed` + `mark_resident` inline, never calling
    /// `set_math_mode`/`set_dispatch_type`, so a shape first resolved through
    /// either diagnostic path entered the cache carrying the DEFAULT math
    /// mode / dispatch type, and a later hit from [`Self::evaluate_with_placements`]
    /// (the production path) silently served that wrong mode. Folding all
    /// three closures through this one function makes every placed-plan
    /// build path correct by construction -- there is no longer a second
    /// closure body that can forget the call.
    ///
    /// Takes `numerics` as one reference rather than `math_mode`/
    /// `numeric_policy`/`dispatch_type` as three positional copies --
    /// [`PlanNumerics`] groups exactly the three [`ServingConfig`] knobs
    /// every caller below already reads and threads together, so the
    /// signature says that instead of leaving it to be true by convention
    /// (and drops the argument count back under clippy's threshold without
    /// an `#[allow]`).
    ///
    /// `numeric_policy` is now passed INTO `plan_named_placed` at
    /// construction, never set afterward (`omega::metal::Plan::
    /// numeric_policy`'s own doc: the policy is fixed for a plan's whole
    /// life) -- `set_math_mode` runs after construction only to NARROW the
    /// compiled `MathMode` within that already-bound policy, and is now
    /// fallible for exactly that reason.
    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
    pub(super) fn build_placed_plan(
        program: &[Op],
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
        resident_names: &BTreeSet<&str>,
        input_placements: &[(NodeId, &PlacedBuffer, usize)],
        numerics: &PlanNumerics,
    ) -> Result<omega::metal::Plan, InteropError> {
        let placed_input_nodes: Vec<NodeId> =
            input_placements.iter().map(|(node, _, _)| *node).collect();
        let mut plan = plan_named_with_placed_inputs(
            program,
            symbols,
            named,
            outputs,
            numerics.numeric_policy,
            &placed_input_nodes,
            numerics.fuse_cached_attention,
        )?;
        plan.mark_resident(resident_names);
        if numerics.plan_time_constants {
            plan.mark_plan_time_constants_resident();
        }
        plan.set_math_mode(numerics.math_mode)?;
        plan.set_dispatch_type(numerics.dispatch_type);
        // Decode-shaped: this plan's own new-token count (`symbols[0]`, the
        // same convention `resolve_cached_plan`'s own `shape` key reads) is
        // exactly `1`. The owner's own integration scoping: the measured
        // chunked-submission win covers only this shape, never prefill.
        plan.set_command_buffer_chunks(
            numerics.command_buffer_chunks,
            symbols.first().copied() == Some(1),
        );
        Ok(plan)
    }

    /// [`Self::evaluate_with_placements`]'s diagnostic counterpart, same
    /// relationship [`Self::evaluate_op_timed`] already has to
    /// [`Self::evaluate`]: identical plan-cache lookup against
    /// [`Self::placed_plans`], but
    /// [`omega::metal::execute_plan_named_with_placements_op_timed`] commits
    /// and waits on ITS OWN command buffer per op instead of the whole
    /// program's one, so [`OpGpuTiming`] comes back for the default decode
    /// path's placed-KV shape the same way [`Self::evaluate_op_timed`]
    /// already does for the two-range path. Reachable only behind the
    /// `instrument` feature and only from `run_decode_loop_placed_kv`'s own
    /// `PROXIMA_METAL_OP_PROFILE_STEP` branch.
    #[cfg(all(
        feature = "metal-output-placement",
        feature = "instrument",
        target_os = "macos"
    ))]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn evaluate_op_timed_with_placements(
        &mut self,
        program: &[Op],
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
        resident_names: &BTreeSet<&str>,
        input_placements: &[(NodeId, &PlacedBuffer, usize)],
        output_placements: &[(NodeId, &PlacedBuffer, usize)],
    ) -> Result<(Evaluated, Vec<OpGpuTiming>), InteropError> {
        let shape = (symbols[0] as usize, symbols[1] as usize, outputs.to_vec());
        let numerics = PlanNumerics {
            math_mode: self.math_mode,
            numeric_policy: self.numeric_policy,
            dispatch_type: self.dispatch_type,
            plan_time_constants: self.plan_time_constants,
            command_buffer_chunks: self.command_buffer_chunks,
            fuse_cached_attention: true,
        };
        let plan = Self::resolve_cached_plan(
            &mut self.placed_plans,
            &mut self.plan_hits,
            &mut self.plan_misses,
            shape,
            || {
                Self::build_placed_plan(
                    program,
                    symbols,
                    named,
                    outputs,
                    resident_names,
                    input_placements,
                    &numerics,
                )
            },
        )?;
        Ok(execute_plan_named_with_placements_op_timed(
            plan,
            named,
            input_placements,
            output_placements,
        )?)
    }

    /// [`Self::evaluate_with_placements`]'s per-dispatch GPU-timestamp
    /// counterpart -- same plan-cache lookup, but
    /// [`omega::execute_plan_named_with_placements_dispatch_timed`] submits
    /// the SAME single command buffer the production path does and reads
    /// `MTLCounterSampleBuffer` timestamps bracketing every dispatch
    /// instead of one command buffer per op
    /// ([`Self::evaluate_op_timed_with_placements`]'s own shape, which ROW
    /// 298's own finding says does not reproduce the batched buffer's
    /// cost). Reachable only behind `instrument` and only from
    /// `run_decode_loop_placed_kv`'s own `PROXIMA_METAL_DISPATCH_PROFILE_STEP`
    /// branch. That same call site also reads `PROXIMA_METAL_ENCODER_SPLIT_AT`
    /// (ROW 329, same one-env-var-per-diagnostic convention as
    /// `PROXIMA_DUPLICATE_HEAD`/`PROXIMA_METAL_OP_PROFILE_STEP` above) and
    /// applies it to the resolved plan via
    /// [`omega::metal::Plan::set_encoder_split_at`] AFTER the cache lookup
    /// below, never inside the build closure: measured directly (a
    /// `debug!` trace that showed `plan_encoder_split_at=None` reaching
    /// [`omega::metal::execute_plan_with_placements_dispatch_timed`] despite
    /// this call setting it), `ServingConfig::kv_bucket_tokens` rounds
    /// several consecutive steps' KV extents onto the SAME cache key, so
    /// the plan this call's own step reuses is often one
    /// [`Self::evaluate_with_placements`]'s closure already inserted --
    /// build-time-only would silently no-op on that hit.
    #[cfg(all(
        feature = "metal-output-placement",
        feature = "instrument",
        target_os = "macos"
    ))]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn evaluate_dispatch_timed_with_placements(
        &mut self,
        program: &[Op],
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
        resident_names: &BTreeSet<&str>,
        input_placements: &[(NodeId, &PlacedBuffer, usize)],
        output_placements: &[(NodeId, &PlacedBuffer, usize)],
    ) -> Result<omega::metal::DispatchTimedOutcome, InteropError> {
        let shape = (symbols[0] as usize, symbols[1] as usize, outputs.to_vec());
        let numerics = PlanNumerics {
            math_mode: self.math_mode,
            numeric_policy: self.numeric_policy,
            dispatch_type: self.dispatch_type,
            plan_time_constants: self.plan_time_constants,
            command_buffer_chunks: self.command_buffer_chunks,
            fuse_cached_attention: true,
        };
        let plan = Self::resolve_cached_plan(
            &mut self.placed_plans,
            &mut self.plan_hits,
            &mut self.plan_misses,
            shape,
            || {
                Self::build_placed_plan(
                    program,
                    symbols,
                    named,
                    outputs,
                    resident_names,
                    input_placements,
                    &numerics,
                )
            },
        )?;
        // Applied AFTER the cache lookup, not inside the build closure above:
        // this shape's `Plan` is just as likely to have been inserted by
        // `Self::evaluate_with_placements`'s own closure (a HIT here on a
        // step where `PROXIMA_METAL_DISPATCH_PROFILE_STEP` was unset, since
        // `ServingConfig::kv_bucket_tokens` rounds several consecutive
        // steps' KV extents to the SAME cache key) as by this function's
        // own closure -- a build-time-only `set_encoder_split_at` call
        // would silently no-op on that hit path. Cheap and always correct
        // to set unconditionally: unlike `set_math_mode`, this field never
        // invalidates `resolved_steps` (this same function's own doc).
        plan.set_encoder_split_at(encoder_split_at_from_env());
        Ok(execute_plan_named_with_placements_dispatch_timed(
            plan,
            named,
            input_placements,
            output_placements,
        )?)
    }

    /// [`Self::evaluate`]/[`Self::evaluate_op_timed`]/
    /// [`Self::evaluate_with_placements`]/
    /// [`Self::evaluate_op_timed_with_placements`]'s shared cache-lookup
    /// step, split out so the eviction policy lives in exactly one place and
    /// so every caller gets back the `Plan` it just resolved rather than a
    /// shape it must look up again -- the second lookup was the only reason
    /// `InteropError::PlanCacheEntryVanished` (now deleted) existed, for a
    /// state (a key missing immediately after this function inserted it)
    /// that cannot occur: [`alloc::collections::btree_map::Entry`] proves it
    /// at the type level instead.
    ///
    /// Generic over the cached `Plan` type because [`Self::plans`] and
    /// [`Self::placed_plans`] key different `Plan` types while sharing this
    /// exact hit/miss/evict policy. The output set is part of the key because
    /// decode can add the logits root only on the final batch.
    ///
    /// This struct's own [`Self::plans`] field comment already proved
    /// ordinary autoregressive decode's `cached_len` strictly increases, so
    /// a `Plan` keyed on it is NEVER looked up again once superseded --
    /// measured directly: `plan_cache_len`/`plan_misses` both grew 1:1 with
    /// the step index (`plan_hits` stayed 0) across a real decode, while
    /// `phys_footprint_bytes` climbed and `omega::metal::current_allocated_size()`
    /// stayed flat over the same steps -- the growth is a Rust-heap leak of
    /// superseded `Plan`s, not a Metal-driver allocation. Clearing the map on
    /// every miss keeps the one entry the doc's own rationale says is worth
    /// keeping (an immediate same-shape replay lands as a hit BEFORE the
    /// next miss would evict it) while making superseded entries collectible
    /// instead of retained for the rest of the call.
    pub(super) fn resolve_cached_plan<'cache, PlanKey, PlanType>(
        cache: &'cache mut alloc::collections::BTreeMap<PlanKey, PlanType>,
        plan_hits: &mut usize,
        plan_misses: &mut usize,
        shape: PlanKey,
        build: impl FnOnce() -> Result<PlanType, InteropError>,
    ) -> Result<&'cache mut PlanType, InteropError>
    where
        PlanKey: Ord,
    {
        use alloc::collections::btree_map::Entry;

        if !cache.contains_key(&shape) {
            cache.clear();
        }
        match cache.entry(shape) {
            Entry::Occupied(entry) => {
                *plan_hits += 1;
                Ok(entry.into_mut())
            }
            Entry::Vacant(entry) => {
                *plan_misses += 1;
                let plan = build()?;
                Ok(entry.insert(plan))
            }
        }
    }

    pub(super) fn resolve_segment_plan<'cache, PlanType>(
        cache: &'cache mut alloc::collections::BTreeMap<
            (usize, usize, usize, Vec<NodeId>),
            PlanType,
        >,
        plan_hits: &mut usize,
        plan_misses: &mut usize,
        shape: (usize, usize, usize, Vec<NodeId>),
        build: impl FnOnce() -> Result<PlanType, InteropError>,
    ) -> Result<&'cache mut PlanType, InteropError> {
        use alloc::collections::btree_map::Entry;

        match cache.entry(shape) {
            Entry::Occupied(entry) => {
                *plan_hits += 1;
                Ok(entry.into_mut())
            }
            Entry::Vacant(entry) => {
                *plan_misses += 1;
                let plan = build()?;
                Ok(entry.insert(plan))
            }
        }
    }

    /// Live entry count in [`Self::plans`] -- the direct witness that
    /// [`Self::resolve_cached_plan`]'s clear-on-miss policy keeps this bounded at 1
    /// through ordinary autoregressive decode's strictly increasing
    /// `cached_len`, rather than growing 1:1 with the step index as it did
    /// before that policy landed. See `token_breakdown_metal`'s
    /// `plan_cache_len` field.
    #[cfg(feature = "instrument")]
    pub(crate) fn plans_len(&self) -> usize {
        self.plans.len()
    }

    /// Retained output-slot bytes across the plan caches. Peak liveness is
    /// smaller than this when slots are reused within one plan; this census
    /// answers why a device allocation remains high after a step completes.
    #[cfg(feature = "instrument")]
    pub(crate) fn arena_allocated_bytes(&self) -> (usize, usize, usize) {
        let ordinary = self
            .plans
            .values()
            .map(omega::backend::plan_arena_allocated_bytes)
            .sum();
        let segments = self
            .segment_plans
            .values()
            .map(omega::backend::plan_arena_allocated_bytes)
            .sum();
        #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
        let placed = self
            .placed_plans
            .values()
            .chain(self.placed_segment_plans.values())
            .map(|plan| plan.arena_allocated_bytes().unwrap_or_default())
            .sum();
        #[cfg(not(all(feature = "metal-output-placement", target_os = "macos")))]
        let placed = 0;
        (ordinary, segments, placed)
    }

    /// [`Self::plans_len`]'s placed-KV counterpart -- [`Self::plans`] and
    /// [`Self::placed_plans`] are two SEPARATE maps
    /// ([`Self::evaluate_with_placements`] never touches [`Self::plans`] at
    /// all), so a caller on [`LoadedModel::run_decode_loop_placed_kv`]'s
    /// own arm reading [`Self::plans_len`] was always reading a map that
    /// stayed empty for the whole call, regardless of how many placed
    /// plans were actually cached.
    #[cfg(all(
        feature = "instrument",
        feature = "metal-output-placement",
        target_os = "macos"
    ))]
    pub(crate) fn placed_plans_len(&self) -> usize {
        self.placed_plans.len()
    }

    /// Diagnostic counterpart of [`Self::evaluate`]: same plan-cache lookup,
    /// but the Metal driver commits and waits on ONE command buffer PER
    /// `BoundOp` instead of once for the whole program, so each op's own
    /// GPU-only execution time comes back alongside the result -- see
    /// `omega::metal::execute_plan_op_timed`'s own doc for the cost this
    /// pays and why it must never replace [`Self::evaluate`] on the serving
    /// loop. Reachable only behind the `instrument` feature and only from
    /// this crate's own diagnostic call sites (`run_decode_loop`'s
    /// `PROXIMA_METAL_OP_PROFILE_STEP` branch).
    #[cfg(all(
        feature = "instrument",
        target_os = "macos",
        not(feature = "metal-output-placement")
    ))]
    pub(super) fn evaluate_op_timed(
        &mut self,
        program: &[Op],
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
        resident_names: &BTreeSet<&str>,
    ) -> Result<(Evaluated, Vec<OpGpuTiming>), InteropError> {
        let shape = (symbols[0] as usize, symbols[1] as usize, outputs.to_vec());
        let plan = Self::resolve_cached_plan(
            &mut self.plans,
            &mut self.plan_hits,
            &mut self.plan_misses,
            shape,
            || {
                let mut plan = plan_named(
                    self.engine,
                    None,
                    program,
                    symbols,
                    named,
                    outputs,
                    self.numeric_policy,
                )?;
                mark_resident(&mut plan, resident_names);
                if self.plan_time_constants {
                    mark_plan_time_constants_resident(&mut plan);
                }
                set_math_mode(&mut plan, self.math_mode)?;
                set_dispatch_type(&mut plan, self.dispatch_type);
                set_command_buffer_chunks(
                    &mut plan,
                    self.command_buffer_chunks,
                    symbols.first().copied() == Some(1),
                );
                Ok(plan)
            },
        )?;
        Ok(execute_plan_named_metal_op_timed(plan, named, None)?)
    }

    #[cfg(all(feature = "instrument", target_os = "macos"))]
    pub(super) fn evaluate_op_timed_with_expert_sources(
        &mut self,
        program: &[Op],
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
        resident_names: &BTreeSet<&str>,
        expert_sources: &BTreeMap<NodeId, proxima_tensor::cpu::ExpertSource<'_>>,
    ) -> Result<(Evaluated, Vec<OpGpuTiming>), InteropError> {
        let exact_activations = self.exact_activations;
        let mut profile_outputs = outputs.to_vec();
        for selected in [
            "PROXIMA_METAL_COMPARE_BOUND_NODE",
            "PROXIMA_METAL_MATERIALIZE_BOUND_NODE",
        ]
        .into_iter()
        .filter_map(|name| {
            std::env::var(name)
                .ok()
                .and_then(|value| value.parse::<u32>().ok())
                .map(NodeId)
        }) {
            if program
                .get(selected.0 as usize)
                .is_some_and(|operation| !matches!(operation, Op::Input { .. }))
                && !profile_outputs.contains(&selected)
            {
                profile_outputs.push(selected);
            }
        }
        let shape = (
            symbols[0] as usize,
            symbols[1] as usize,
            profile_outputs.clone(),
        );
        let plan = Self::resolve_cached_plan(
            &mut self.plans,
            &mut self.plan_hits,
            &mut self.plan_misses,
            shape,
            || {
                let mut plan = plan_named(
                    self.engine,
                    None,
                    program,
                    symbols,
                    named,
                    &profile_outputs,
                    self.numeric_policy,
                )?;
                mark_resident(&mut plan, resident_names);
                if self.plan_time_constants {
                    mark_plan_time_constants_resident(&mut plan);
                }
                set_math_mode(&mut plan, self.math_mode)?;
                set_dispatch_type(&mut plan, self.dispatch_type);
                set_command_buffer_chunks(
                    &mut plan,
                    self.command_buffer_chunks,
                    symbols.first().copied() == Some(1),
                );
                Ok(plan)
            },
        )?;
        let cpu_reference = std::env::var("PROXIMA_METAL_COMPARE_BOUND_NODE")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .map(NodeId)
            .and_then(|selected| {
                let mut scratch = Vec::new();
                let mut validated = None;
                let expected = if exact_activations {
                    evaluate_quantized_named_exact_with_scratch_and_experts(
                        program,
                        symbols,
                        named,
                        &profile_outputs,
                        &mut scratch,
                        &mut validated,
                        Some(expert_sources),
                    )
                } else {
                    evaluate_quantized_named_with_scratch_and_experts(
                        program,
                        symbols,
                        named,
                        &profile_outputs,
                        &mut scratch,
                        &mut validated,
                        Some(expert_sources),
                    )
                }
                .ok()?;
                let (values, _) = expected.get(selected)?;
                let mut reference = BTreeMap::new();
                reference.insert(selected, values.to_vec());
                Some(reference)
            });
        Ok(execute_plan_named_metal_op_timed_with_expert_sources(
            plan,
            named,
            expert_sources,
            cpu_reference.as_ref(),
        )?)
    }
}

/// The CPU-direct runtime a build without the `metal` feature keeps --
/// `omega` is not even a dependency in that build (`Cargo.toml`'s `metal`
/// feature is the only thing that turns `dep:omega` on), so this calls
/// [`evaluate_quantized_named_with_scratch`] exactly as
/// [`LoadedModel::generate`] always has. `free_buffers`/`validated_weight_nodes`
/// are the same scratch this loop's local variables used to own directly --
/// moved onto this struct so [`LoadedModel::generate_with_serving_config`]'s
/// loop body reads identically whether or not `metal` is compiled in.
#[cfg(not(feature = "metal"))]
pub(crate) struct BackendRuntime {
    pub(super) free_buffers: Vec<Vec<f32>>,
    pub(super) validated_weight_nodes: Option<BTreeSet<NodeId>>,
    /// `ServingConfig::exact_activations`, read once at construction --
    /// see that field's own doc.
    pub(super) exact_activations: bool,
}

#[cfg(not(feature = "metal"))]
impl BackendRuntime {
    pub(crate) fn new(_config: &ServingConfig) -> Self {
        Self {
            free_buffers: Vec::new(),
            validated_weight_nodes: None,
            exact_activations: _config.exact_activations,
        }
    }

    pub(super) fn uses_gpu(&self) -> bool {
        false
    }

    /// `resident_names` is unused on this backend: the CPU evaluator has no
    /// device buffer to cache, so there is nothing to mark resident. Carried
    /// anyway so both `BackendRuntime::evaluate` impls share one signature
    /// and the decode loop's call site never needs a `cfg` of its own.
    pub(super) fn evaluate(
        &mut self,
        program: &[Op],
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
        _resident_names: &BTreeSet<&str>,
        expert_sources: Option<
            &alloc::collections::BTreeMap<NodeId, proxima_tensor::cpu::ExpertSource<'_>>,
        >,
    ) -> Result<Evaluated, InteropError> {
        if self.exact_activations {
            return Ok(evaluate_quantized_named_exact_with_scratch_and_experts(
                program,
                symbols,
                named,
                outputs,
                &mut self.free_buffers,
                &mut self.validated_weight_nodes,
                expert_sources,
            )?);
        }
        Ok(evaluate_quantized_named_with_scratch_and_experts(
            program,
            symbols,
            named,
            outputs,
            &mut self.free_buffers,
            &mut self.validated_weight_nodes,
            expert_sources,
        )?)
    }

    /// Evaluates one graph partition through the same reusable CPU scratch
    /// state as [`Self::evaluate`]. The partition owns its node numbering and
    /// therefore cannot alias the decode path's plan cache; CPU has no plan
    /// cache, so the only safe reusable state is the evaluator's scratch pool
    /// and weight-validation set. Keeping this method beside the Metal
    /// implementation gives routed callers one backend-independent seam.
    pub(crate) fn evaluate_segment(
        &mut self,
        program: &[Op],
        symbols: &[u64],
        named: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
        resident_names: &BTreeSet<&str>,
        expert_sources: Option<
            &alloc::collections::BTreeMap<NodeId, proxima_tensor::cpu::ExpertSource<'_>>,
        >,
    ) -> Result<Evaluated, InteropError> {
        self.evaluate(
            program,
            symbols,
            named,
            outputs,
            resident_names,
            expert_sources,
        )
    }
}

impl<'file> Pipe for LoadedModel<'file> {
    type In = (String, usize);
    type Out = (Vec<u32>, String, bool);
    type Err = InteropError;

    fn call(
        &self,
        input: (String, usize),
    ) -> impl Future<Output = Result<(Vec<u32>, String, bool), InteropError>> {
        async move {
            let (prompt, max_tokens) = input;
            self.generate(&prompt, max_tokens)
        }
    }
}

/// One decode step surfaced to a caller AS it happens, instead of only
/// after [`LoadedModel::generate_streaming`] returns -- the payload
/// `decode_until_stop_or_budget` hands to its `on_token` callback every
/// step, teaching a caller (a CLI's "loading / thinking / answering"
/// indicator) exactly what that loop already knows at that point and
/// nothing it has to re-derive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenEvent<'piece> {
    pub token_id: u32,
    /// This token's own decoded text, continuing on from whatever
    /// `decode_until_stop_or_budget` has already handed back for earlier
    /// tokens -- concatenating every `text_piece` across a whole decode
    /// reproduces [`proxima_tokenizer::decode`]'s own output on the same
    /// ids exactly (`decode_streamed_piece`'s own doc: incomplete
    /// multi-byte tails carry forward instead of resolving to U+FFFD mid
    /// stream).
    pub text_piece: &'piece str,
    pub phase: Phase,
    /// `0`-indexed decode step this event belongs to.
    pub step: usize,
    /// Milliseconds since this call's decode loop started (`step` `0`'s own
    /// first [`std::time::Instant::now`] reading), not this step's own
    /// duration -- a caller computes both a running tok/s and a single
    /// step's latency from two consecutive events' `elapsed_ms` without
    /// this loop tracking either itself. Plain [`std::time::Instant`]
    /// rather than `proxima_tensor::instrument`'s tick counters: those only
    /// exist behind this crate's diagnostic-only `instrument` feature
    /// (`proxima-tensor/src/lib.rs`'s own `#[cfg(feature = "instrument")]`
    /// on that module), and a live "what's running" indicator must work on
    /// every build that reaches [`LoadedModel::generate_streaming`] at all,
    /// not only one compiled for op-level profiling.
    pub elapsed_ms: u64,
}

/// Allocation-free decode evidence assembled from the events a caller already
/// receives. The labels preserve provenance when a record is written beside
/// measurements from another runtime or benchmark harness.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeMetrics {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub elapsed_ms: u64,
    pub tokens_per_second: f64,
    pub per_token_latency_ms: f64,
    pub peak_rss_bytes: u64,
    pub cpu_percent: f64,
    pub error_count: u64,
    pub covariance: f64,
    pub timing_source: &'static str,
    pub memory_source: &'static str,
    pub resource_source: &'static str,
}

impl DecodeMetrics {
    /// Builds one record without allocating or re-reading the model output.
    /// `elapsed_ms` is the final cumulative [`TokenEvent`] timestamp, so the
    /// throughput and latency fields describe the same observed interval.
    pub fn from_events(
        events: &[TokenEvent<'_>],
        peak_rss_bytes: u64,
        cpu_percent: f64,
        error_count: u64,
        covariance: f64,
    ) -> Self {
        let mut prompt_tokens = 0;
        let mut generated_tokens = 0;
        let mut elapsed_ms = 0;
        for event in events {
            match event.phase {
                Phase::Prefill {
                    prompt_tokens: count,
                } => prompt_tokens = count,
                Phase::Token => generated_tokens += 1,
            }
            elapsed_ms = event.elapsed_ms;
        }
        let elapsed_seconds = elapsed_ms as f64 / 1000.0;
        let tokens_per_second = if elapsed_seconds > 0.0 {
            generated_tokens as f64 / elapsed_seconds
        } else {
            0.0
        };
        let per_token_latency_ms = if generated_tokens > 0 {
            elapsed_ms as f64 / generated_tokens as f64
        } else {
            0.0
        };
        Self {
            prompt_tokens,
            generated_tokens,
            elapsed_ms,
            tokens_per_second,
            per_token_latency_ms,
            peak_rss_bytes,
            cpu_percent,
            error_count,
            covariance,
            timing_source: "TokenEvent::elapsed_ms",
            memory_source: "caller_peak_rss_bytes",
            resource_source: "caller_cpu_percent_and_error_count",
        }
    }
}

/// [`TokenEvent::phase`]: which part of the decode loop produced this
/// event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Emitted exactly once, at decode step `0` -- the one step whose
    /// forward pass evaluates the whole encoded prompt rather than a single
    /// new token (`LoadedModel::run_decode_loop_observed`'s own doc on
    /// `new_positions == prompt_length` on the first step).
    Prefill {
        /// The encoded prompt's own token count (`ids.len()` before
        /// decoding starts), so a caller can show "127 prompt tokens"
        /// without re-encoding the prompt itself.
        prompt_tokens: usize,
    },
    /// Every decode step, prefill included -- carries the token that step
    /// produced.
    Token,
}

/// One [`TokenEvent::text_piece`] worth of text for `token_id`, carrying
/// forward any UTF-8 tail [`proxima_tokenizer::bpe::decode_ids`] left
/// incomplete in `pending` from a previous call -- byte-level BPE has no
/// obligation to keep a multibyte character inside one token
/// ([`proxima_tokenizer::pipe::decode`]'s own doc), so a caller watching
/// tokens arrive one at a time needs the same "flag, don't drop" contract
/// that one-shot decode gives the whole sequence, just deferred: an
/// incomplete tail waits here for the token that completes it instead of
/// resolving to U+FFFD before decoding is known to be finished.
///
/// Never returns [`proxima_tokenizer::TokenizerError::InvalidUtf8`] over a
/// pending tail that turns out unresolvable -- a decode step producing a
/// token id that does not validly continue an earlier incomplete lead byte
/// is a normal outcome of autoregressive sampling, not corruption
/// (guiding-principles principle 15: an error type is not the "correct
/// treatment" for a case the caller cannot act on). Draining is
/// [`proxima_tokenizer::drain_lossy_utf8`] itself -- the same routine
/// [`proxima_tokenizer::pipe::decode`] uses for the one-shot whole-sequence
/// case -- so a genuinely invalid run resolves to one U+FFFD and draining
/// resumes on whatever bytes remain after it here exactly as it does there;
/// the only difference is this call site leaves an incomplete trailing
/// sequence in `pending` for a future token to complete, instead of
/// flushing it immediately.
pub(super) fn decode_streamed_piece(
    vocab: &Vocab,
    token_id: u32,
    pending: &mut Vec<u8>,
) -> Result<String, InteropError> {
    let bytes = proxima_tokenizer::bpe::decode_ids(&[token_id], vocab)?;
    pending.extend_from_slice(&bytes);
    let mut piece = String::new();
    proxima_tokenizer::drain_lossy_utf8(pending, &mut piece);
    Ok(if vocab.is_unigram() {
        proxima_tokenizer::unigram::replace_space_markers(&piece)
    } else {
        piece
    })
}

/// The decode loop's termination policy, isolated from the forward pass
/// that produces each token: pulls up to `max_tokens` ids out of
/// `produce_next_token` (one call per step, `0`-indexed), appending each
/// to the result unless it is `vocab`'s end-of-sequence id, in which case
/// decoding stops immediately without appending that id. Returns the
/// accumulated ids plus whether the stop was the model's own signal
/// (`true`) rather than the budget running out (`false`) -- a caller
/// [`ControlFlow::Break`] collapses into the same `false` as budget
/// exhaustion, since neither is the model's own eos.
///
/// Every step, after producing that step's token, calls `on_token` once (an
/// extra [`Phase::Prefill`] call at step `0`, ahead of that step's own
/// [`Phase::Token`] call) -- [`LoadedModel::generate_with_serving_config`]'s
/// own `&mut |_| ControlFlow::Continue(())` never observes a difference
/// from this function's pre-streaming behavior;
/// [`LoadedModel::generate_streaming`] is the same loop with a real
/// callback.
///
/// Factored out so this policy -- the exact defect this module's
/// [`LoadedModel::generate`] fixed (a loop with no termination condition
/// besides the budget) -- is provable against a scripted token source,
/// without paying for a real forward pass per test.
pub(super) fn decode_until_stop_or_budget(
    vocab: &Vocab,
    max_tokens: usize,
    prompt_token_count: usize,
    mut produce_next_token: impl FnMut(usize) -> Result<u32, InteropError>,
    on_token: &mut dyn FnMut(TokenEvent<'_>) -> ControlFlow<(), ()>,
) -> Result<(Vec<u32>, bool), InteropError> {
    let mut generated_ids = Vec::with_capacity(max_tokens);
    let mut stopped_by_eos = false;
    let mut pending_bytes: Vec<u8> = Vec::new();
    let mut unigram_leading_space_trimmed = false;
    let loop_started = std::time::Instant::now();
    for step in 0..max_tokens {
        let token_id = produce_next_token(step)?;
        let elapsed_ms = u64::try_from(loop_started.elapsed().as_millis()).unwrap_or(u64::MAX);
        let is_eos = vocab.eos_token_id() == Some(token_id);
        // Control tokens (gemma4's `<turn|>`-shaped turn markers) are
        // structural, not content -- they must never appear in decoded
        // TEXT, but unlike eos they do not stop generation: the id still
        // enters `generated_ids` below, only its visible piece is empty.
        let is_control = vocab.token_type(token_id) == Some(TokenType::Control);
        let mut text_piece = if is_eos || is_control {
            String::new()
        } else {
            decode_streamed_piece(vocab, token_id, &mut pending_bytes)?
        };
        // Mirrors `proxima_tokenizer::pipe::decode`'s own one-time leading-
        // space trim (SentencePiece's `escape` always prepends one), applied
        // to the FIRST non-empty piece this whole call ever emits rather
        // than every piece -- a later piece starting with the space marker
        // is a real inter-word space, not that artifact.
        if !unigram_leading_space_trimmed && vocab.is_unigram() && !text_piece.is_empty() {
            unigram_leading_space_trimmed = true;
            if text_piece.starts_with(' ') {
                text_piece.remove(0);
            }
        }
        if step == 0 {
            let control = on_token(TokenEvent {
                token_id,
                text_piece: &text_piece,
                phase: Phase::Prefill {
                    prompt_tokens: prompt_token_count,
                },
                step,
                elapsed_ms,
            });
            if control == ControlFlow::Break(()) {
                break;
            }
        }
        if is_eos {
            stopped_by_eos = true;
            break;
        }
        generated_ids.push(token_id);
        let control = on_token(TokenEvent {
            token_id,
            text_piece: &text_piece,
            phase: Phase::Token,
            step,
            elapsed_ms,
        });
        if control == ControlFlow::Break(()) {
            break;
        }
    }
    Ok((generated_ids, stopped_by_eos))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod decode_control_suppression_tests {
    use alloc::string::String;
    use core::ops::ControlFlow;

    use super::{Phase, TokenEvent, TokenType, Vocab, decode_until_stop_or_budget};

    /// A small vocab with one [`TokenType::Control`] entry (`"<turn|>"`,
    /// gemma4's real turn-boundary marker) among ordinary text tokens --
    /// built directly, no model or checkpoint needed
    /// (`decode_until_stop_or_budget`'s own doc: "provable against a
    /// scripted token source").
    fn vocab_with_control_marker() -> Vocab {
        let tokens = alloc::vec![
            String::from("h"),
            String::from("i"),
            String::from("<turn|>"),
        ];
        let vocab =
            Vocab::new(tokens, &[], None, None, None).expect("small vocab with no merges builds");
        let token_types = alloc::vec![TokenType::Normal, TokenType::Normal, TokenType::Control];
        vocab
            .with_token_types(token_types)
            .expect("token type array length matches vocab length")
    }

    /// Pre-fix, this loop never consulted `token_type` and the control
    /// id's own bytes (`"<turn|>"`) leaked into `text_piece` like any other
    /// token, so this call's accumulated text was `"h<turn|>i"`. Post-fix
    /// the control id still enters `generated_ids` (it is structural
    /// signal a caller may want to see, just not render), but contributes
    /// zero characters to the decoded text, leaving `"hi"`.
    #[test]
    fn decode_until_stop_or_budget_suppresses_control_text_but_keeps_the_id() {
        let vocab = vocab_with_control_marker();
        let h_id = vocab.token_id("h").expect("h token exists");
        let i_id = vocab.token_id("i").expect("i token exists");
        let control_id = vocab.token_id("<turn|>").expect("control token exists");
        let script = [h_id, control_id, i_id];

        // Only `Phase::Token` events are accumulated -- step 0 additionally
        // fires a `Phase::Prefill` event carrying the same `text_piece`
        // (`decode_until_stop_or_budget`'s own doc), and a real caller
        // renders text from one phase, not both, to avoid double-counting
        // step 0's piece.
        let mut text = String::new();
        let (generated_ids, stopped_by_eos) = decode_until_stop_or_budget(
            &vocab,
            script.len(),
            0,
            |step| Ok(script[step]),
            &mut |event: TokenEvent<'_>| {
                if event.phase == Phase::Token {
                    text.push_str(event.text_piece);
                }
                ControlFlow::Continue(())
            },
        )
        .expect("decode succeeds against a scripted token source");

        assert_eq!(
            generated_ids,
            script.to_vec(),
            "the control id still enters generated_ids"
        );
        assert!(!stopped_by_eos, "no eos id in this script");
        assert_eq!(
            text, "hi",
            "the control token must contribute zero characters to decoded text"
        );
        assert!(
            !text.contains("<turn|>"),
            "no literal control marker leaks into decoded text"
        );
    }
}

/// [`Vocab::add_bos_token`]'s own fallback when the checkpoint's metadata
/// carries no `tokenizer.ggml.add_bos_token` opinion at all: default to
/// requesting BOS only when the vocab actually HAS a
/// [`Vocab::bos_token_id`] to add. Every dense checkpoint this crate has
/// bound so far declares one (openchat-3.5, SmolLM2), so this reproduces
/// this crate's pre-existing unconditional `true` default for them
/// byte-for-byte; the real Qwen3.5 checkpoint declares neither the policy
/// key nor a `tokenizer.ggml.bos_token_id` key at all (confirmed via
/// `strings` on the real file -- Qwen's own tokenizer has no BOS token,
/// chat turns open on `<|im_start|>` instead), so defaulting to `true`
/// there would ask [`proxima_tokenizer::encode_with_bos_eos`] to prepend an
/// id that does not exist, surfacing
/// [`proxima_tokenizer::TokenizerError::MissingMetadataKey`] on every
/// prompt rather than the tokenizer's own real, silent policy.
pub(super) fn wants_bos(vocab: &Vocab) -> bool {
    vocab
        .add_bos_token()
        .unwrap_or_else(|| vocab.bos_token_id().is_some())
}

pub(super) struct CurrentExpertSources {
    pub(super) decisions: [crate::residency::ServeDecision; 16],
    pub(super) len: usize,
}

impl CurrentExpertSources {
    pub(super) fn new() -> Self {
        Self {
            decisions: [crate::residency::ServeDecision {
                address: crate::residency::ExpertAddress {
                    layer: 0,
                    expert: 0,
                },
                precision: crate::residency::ServePrecision::Low,
            }; 16],
            len: 0,
        }
    }

    pub(super) fn clear(&mut self) {
        self.len = 0;
    }

    pub(super) fn push(
        &mut self,
        decision: crate::residency::ServeDecision,
    ) -> Result<(), InteropError> {
        let Some(slot) = self.decisions.get_mut(self.len) else {
            return Err(InteropError::PreGatherExecutionUnsupported {
                architecture: String::from("qwen35moe"),
                reason: String::from("router decision count exceeded the fixed staging bound"),
            });
        };
        *slot = decision;
        self.len += 1;
        Ok(())
    }

    pub(super) fn as_slice(&self) -> &[crate::residency::ServeDecision] {
        &self.decisions[..self.len]
    }
}

/// Every `LoadedModel::expert_slab` acquire, in one place -- recovers from
/// poisoning instead of panicking (see that field's own doc for why) rather
/// than each call site repeating the same `unwrap_or_else`.
pub(super) fn lock_expert_slab<'lock, 'file>(
    slab: &'lock std::sync::Mutex<crate::expert_slab::ExpertSlab<'file>>,
) -> std::sync::MutexGuard<'lock, crate::expert_slab::ExpertSlab<'file>> {
    slab.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// A router logits tensor and the `[positions, experts]` shape it was
/// evaluated with, grouped so the functions that consume both keep their
/// argument count under clippy's threshold.
pub(super) struct RouterLogits<'values> {
    pub(super) values: &'values [f32],
    pub(super) shape: &'values [u64],
}

/// The router's configured expert count and the top-k it selects per
/// position, grouped so the functions that consume both keep their
/// argument count under clippy's threshold.
pub(super) struct RouterExpertCounts {
    pub(super) expert_count: usize,
    pub(super) expert_used_count: usize,
}

pub(super) fn visit_qwen35moe_router_selections<BeforeGather>(
    layer: usize,
    position_offset: usize,
    logits: RouterLogits<'_>,
    counts: RouterExpertCounts,
    scratch: &mut Vec<crate::residency::RoutedExpert>,
    before_gather: &mut BeforeGather,
) -> Result<(), InteropError>
where
    BeforeGather: FnMut(usize, u64, &[crate::residency::RoutedExpert]) -> Result<(), InteropError>,
{
    let RouterLogits {
        values: logits,
        shape,
    } = logits;
    let RouterExpertCounts {
        expert_count,
        expert_used_count,
    } = counts;
    let [positions, shaped_experts] = shape else {
        return Err(InteropError::PreGatherExecutionUnsupported {
            architecture: String::from("qwen35moe"),
            reason: alloc::format!(
                "layer {layer} router logits have shape {shape:?}, expected [positions, experts]"
            ),
        });
    };
    let positions =
        usize::try_from(*positions).map_err(|_| InteropError::PreGatherExecutionUnsupported {
            architecture: String::from("qwen35moe"),
            reason: alloc::format!("layer {layer} router position extent does not fit usize"),
        })?;
    let shaped_experts = usize::try_from(*shaped_experts).map_err(|_| {
        InteropError::PreGatherExecutionUnsupported {
            architecture: String::from("qwen35moe"),
            reason: alloc::format!("layer {layer} router expert extent does not fit usize"),
        }
    })?;
    let expected_values = positions.checked_mul(expert_count).ok_or_else(|| {
        InteropError::PreGatherExecutionUnsupported {
            architecture: String::from("qwen35moe"),
            reason: alloc::format!("layer {layer} router shape overflows usize"),
        }
    })?;
    if shaped_experts != expert_count
        || logits.len() != expected_values
        || expert_used_count == 0
        || expert_used_count > expert_count
    {
        return Err(InteropError::PreGatherExecutionUnsupported {
            architecture: String::from("qwen35moe"),
            reason: alloc::format!(
                "layer {layer} router has shape {shape:?}, {} values, expert_count {expert_count}, and expert_used_count {expert_used_count}",
                logits.len()
            ),
        });
    }

    for (local_position, row) in logits.chunks_exact(expert_count).enumerate() {
        if std::env::var_os("PROXIMA_DEBUG_EXPERT_UPLOADS").is_some() {
            let nan_count = row.iter().filter(|value| value.is_nan()).count();
            let min = row.iter().copied().fold(f32::INFINITY, f32::min);
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "qwen35 router logits layer={} position={} min={} max={} nan_count={}",
                layer,
                position_offset.saturating_add(local_position),
                min,
                max,
                nan_count
            );
        }
        scratch.clear();
        for (expert, &importance) in row.iter().enumerate() {
            let candidate = crate::residency::RoutedExpert { expert, importance };
            if scratch.len() < expert_used_count {
                scratch.push(candidate);
            } else {
                let last = scratch[expert_used_count - 1];
                if importance.total_cmp(&last.importance).is_gt()
                    || (importance.total_cmp(&last.importance).is_eq() && expert < last.expert)
                {
                    scratch[expert_used_count - 1] = candidate;
                } else {
                    continue;
                }
            }
            scratch.sort_unstable_by(|left, right| {
                right
                    .importance
                    .total_cmp(&left.importance)
                    .then_with(|| left.expert.cmp(&right.expert))
            });
        }
        before_gather(
            layer,
            position_offset.saturating_add(local_position) as u64,
            scratch,
        )?;
    }
    Ok(())
}

pub(super) fn visit_qwen35moe_router_boundary<'file, BeforeGather>(
    layer: usize,
    position_offset: usize,
    logits: RouterLogits<'_>,
    counts: RouterExpertCounts,
    scratch: &mut Vec<crate::residency::RoutedExpert>,
    expert_slab: &mut crate::expert_slab::ExpertSlab<'file>,
    before_gather: &mut BeforeGather,
) -> Result<(), InteropError>
where
    BeforeGather: FnMut(
        usize,
        u64,
        &[crate::residency::RoutedExpert],
        &mut crate::expert_slab::ExpertSlab<'file>,
    ) -> Result<(), InteropError>,
{
    visit_qwen35moe_router_selections(
        layer,
        position_offset,
        logits,
        counts,
        scratch,
        &mut |layer, position, routes| {
            // Legitimately pauses the caller's own open `StepGuard` scope
            // for exactly this callback -- see [`ExpertSlab::open_step`]'s
            // own doc for why this is the one place a step reopens outside
            // [`ExpertSlab::begin_step`] itself.
            expert_slab.close_step();
            let boundary_result = before_gather(layer, position, routes, expert_slab);
            if boundary_result.is_ok() {
                expert_slab.open_step();
            }
            boundary_result
        },
    )
}
