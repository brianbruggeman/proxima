//! Device-resident KV for the two-range decode loop, prefill included.
//!
//! The host [`LayerCache`] is the source of truth between calls
//! ([`PrefixState`], the prompt cache), but inside one decode call it only
//! ever grew by the rows the device had just computed and was then copied
//! back to the device whole, every token: `KvPadScratch::fill` re-copied the
//! full history of every layer and each scratch became a fresh no-copy
//! `MTLBuffer` wrapper. [`DeviceKv`] keeps those rows in
//! [`PlacedBuffer`]s instead -- the same caller-owned-buffer mechanism the
//! recurrent-interval dense-attention placement and the single-range KV path already use
//! (`omega::execute_plan_named_with_placements`): the leaf the program reads
//! is an input placement over the buffer, the freshly computed rows are an
//! output placement written into the buffer's tail, and the host touches
//! neither. A call adopts its host caches once ([`DeviceKv::adopt`]), before its
//! first evaluation, so a prefill's K and V rows are written by the device into
//! the buffers at their cache offsets and never read back during the call, and
//! hands them back once ([`DeviceKv::flush`]), so everything that reads
//! `layer_caches` after the loop is unchanged.
//!
//! A full layer's rows live at their absolute position. A sliding layer keeps
//! the rows its host ring would hold (window plus rewind slack, `retain`) in a
//! linear buffer of `retain + step` rows rounded up to a compaction granule (`omega::sized::KV_WINDOW_COMPACTION_GRANULE_ROWS`): the leaf placement slides forward
//! by one row per token (the program reads `min(cached_len, window)` rows
//! starting `window` back), and when the read would run off the end the kept
//! rows are moved to the front. That trades the ring's modular indexing, which
//! the kernel does not do, for an amortised `retain`-row move once every
//! granule-slack tokens (`capacity - retain - step`, 255 for a 513-row window with 512-row steps).

use proxima_core::kv_carry::KvCommand;
use proxima_numeric::{DType, FormatCodec, NumericError, NumericFormat};
use proxima_tensor::spec::store_rows_as;

use super::*;
#[cfg(feature = "instrument")]
use crate::kv_cache_format;

// the emitter alone knows which formats a kernel reads, so its answer and its refusal are passed through rather than re-derived here
fn kernel_codecs(
    key: &NumericFormat,
    value: &NumericFormat,
) -> Result<(Option<NumericFormat>, Option<NumericFormat>), InteropError> {
    omega::cached_attention_kv_codecs_for(key, value).map_err(|error| match error {
        omega::EmitError::CachedAttentionKvCodecNotSupported { reason, .. } => {
            InteropError::UnsupportedServingConfig(alloc::format!(
                "device kv cache types key {key:?} value {value:?}: {reason}"
            ))
        }
        other => InteropError::UnsupportedServingConfig(alloc::format!(
            "device kv cache types key {key:?} value {value:?}: {other}"
        )),
    })
}

fn numeric_failure(format: &NumericFormat, error: NumericError) -> InteropError {
    InteropError::UnsupportedServingConfig(alloc::format!(
        "device kv cache format {format:?}: {error}"
    ))
}

/// The placement hook of the device kv: the allocator the three buffers of each layer come from; the default is [`omega::allocate_placed_buffer`]; [`omega::allocate_placed_buffer_over`] wraps memory a caller owns.
pub(super) type KvBufferSource = fn(usize) -> Result<PlacedBuffer, omega::MetalError>;

/// The three `kv_cache.{layer}.*` leaf nodes of one layer in one program.
#[derive(Debug, Clone, Copy)]
pub(super) struct KvLeafNodes {
    pub(super) k_even: NodeId,
    pub(super) k_odd: NodeId,
    pub(super) v: NodeId,
}

/// One step's placements for every device-resident layer, ready to hand to
/// [`BackendRuntime::evaluate_with_placements`].
pub(super) struct KvPlacements<'buffers> {
    pub(super) inputs: Vec<(NodeId, &'buffers PlacedBuffer, usize)>,
    pub(super) outputs: Vec<(NodeId, &'buffers PlacedBuffer, usize)>,
    /// The output nodes whose cache layout is not binary32: the plan encodes them into `outputs`'
    /// buffer in this format ([`BackendRuntime::set_encoded_outputs`]), on the device, in the step's
    /// own command buffer.
    pub(super) encoded: Vec<(NodeId, NumericFormat)>,
}

struct DeviceKvLayer {
    #[cfg(feature = "instrument")]
    layer: usize,
    k_even: PlacedBuffer,
    k_odd: PlacedBuffer,
    v: PlacedBuffer,
    /// Whether the K planes are stored in a layout the graph can neither write itself (see
    /// [`omega::device_row_store_dtype`]) nor leave as binary32; the plan then encodes this step's rows into the cache itself ([`KvPlacements::encoded`]).
    key_encoded: bool,
    /// [`Self::key_encoded`] for the V rows.
    value_encoded: bool,
    key_format: NumericFormat,
    /// The key planes' codec, resolved once at allocation.
    key_codec: FormatCodec,
    value_format: NumericFormat,
    /// The value rows' codec, resolved once at allocation.
    value_codec: FormatCodec,
    even_odd_row: usize,
    v_row: usize,
    even_odd_row_bytes: usize,
    v_row_bytes: usize,
    capacity_rows: usize,
    /// `Some` for a sliding layer: its buffer is a moving window.
    window: Option<usize>,
    /// Rows a sliding layer keeps behind the write head: its host ring's whole
    /// capacity, so a prefix state or prompt cache rewound after the call
    /// finds every row the ring would have held.
    retain: usize,
    /// Absolute position of buffer row 0. Always `0` for a full layer.
    base_position: usize,
}

/// What [`DeviceKv::adopt_resident`] found when it rewound a resident slot to `keep` positions.
/// Composes [`MissReason`] (why nothing is reusable) and [`RingCheckpoint`] positions (llama's
/// `do_reset`-or-restore branch, `tools/server/server-context.cpp:3276-3365`).
#[must_use]
pub(super) enum RewindOutcome {
    /// Every layer still holds the rows a rewind to `keep` keeps; nothing moved or was written.
    InPlace,
    /// A sliding layer lost rows the rewind needs: the newest checkpoint at or before `keep`,
    /// moved out of the slot so the caller restores it ([`PrefixState::restore_checkpoint`], the
    /// path `PromptCache::restore_nearest` takes) and prefills from [`RingCheckpoint::position`].
    Checkpoint(RingCheckpoint),
    /// No checkpoint covers the rewind either, or the slot cannot serve `keep` at all.
    Refused(MissReason),
}

impl core::fmt::Debug for RewindOutcome {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InPlace => formatter.write_str("InPlace"),
            Self::Checkpoint(checkpoint) => formatter.debug_tuple("Checkpoint").field(&checkpoint.position()).finish(),
            Self::Refused(reason) => formatter.debug_tuple("Refused").field(reason).finish(),
        }
    }
}

pub(super) struct DeviceKv {
    layers: Vec<Option<DeviceKvLayer>>,
    max_step_rows: usize,
    key_format: NumericFormat,
    value_format: NumericFormat,
}

#[cfg(feature = "instrument")]
fn acquire(
    source: &mut impl FnMut(usize) -> Result<PlacedBuffer, omega::MetalError>,
    layer: usize,
    buffer: &'static str,
    bytes: usize,
) -> Result<PlacedBuffer, InteropError> {
    let started = read_ticks();
    let placed = source(bytes)?;
    debug!(
        layer = layer as u64,
        buffer,
        bytes = bytes as u64,
        acquire_ns = ticks_to_nanos(elapsed_ticks(started)),
        "kv_adopt_buffer"
    );
    Ok(placed)
}

#[cfg(not(feature = "instrument"))]
fn acquire(
    source: &mut impl FnMut(usize) -> Result<PlacedBuffer, omega::MetalError>,
    _layer: usize,
    _buffer: &'static str,
    bytes: usize,
) -> Result<PlacedBuffer, InteropError> {
    Ok(source(bytes)?)
}

/// Rows of a sliding layer's linear buffer: `retain + step` is the least `make_room` can work in (its
/// `reach` is at most `retain + new_count`, `new_count <= step`), rounded up to `granule` rows. The
/// rounding is the slack between compactions, `granule - 1` tokens for a 513-row window with 512-row steps; see
/// `omega-runtime.toml` `[kv_window]` for the memory-versus-copy trade. A zero granule adds no slack.
fn window_capacity_rows(retain: usize, step: usize, granule: usize) -> usize {
    (retain + step).next_multiple_of(granule.max(1))
}

/// The widest evaluation of a call that prefills `new_ids` tokens: a decode step with its draft
/// (`draft_limit + 1` rows) or a prefill batch, whichever is wider. The one reader for the decode
/// loop's adoption and for the caller deciding whether a resident slot can serve the call.
pub(super) fn device_step_rows(serving_config: &ServingConfig, draft_limit: usize, new_ids: usize) -> usize {
    (draft_limit + 1).max(match serving_config.ubatch_size {
        0 => new_ids,
        ubatch => new_ids.min(ubatch as usize),
    })
}

impl DeviceKvLayer {
    fn allocate(
        layer: usize,
        cache: &LayerCache,
        (even_odd_row, v_row): (usize, usize),
        full_capacity_rows: usize,
        (window_step_rows, compaction_granule_rows): (usize, usize),
        source: &mut impl FnMut(usize) -> Result<PlacedBuffer, omega::MetalError>,
        (key_format, value_format): (&NumericFormat, &NumericFormat),
    ) -> Result<Self, InteropError> {
        let narrow_codecs = kernel_codecs(key_format, value_format)?;
        let encoded = |narrow: bool, format: &NumericFormat| narrow && omega::device_row_store_dtype(format).is_none();
        let slice_codec = |format: &NumericFormat| {
            FormatCodec::for_format(format).ok_or_else(|| {
                InteropError::UnsupportedServingConfig(alloc::format!(
                    "device kv cache format {format:?} has no slice codec"
                ))
            })
        };
        let key_codec = slice_codec(key_format)?;
        let value_codec = slice_codec(value_format)?;
        let row_bytes = |format: &NumericFormat, row: usize| {
            format.packed_len(row).ok_or_else(|| {
                InteropError::UnsupportedServingConfig(alloc::format!(
                    "device kv cache format {format:?}: a row of {row} elements is not whole blocks"
                ))
            })
        };
        let window = cache.ring_geometry().map(|ring| ring.window);
        let retain = cache.ring_geometry().map_or(0, |ring| ring.capacity);
        let capacity_rows = if window.is_some() {
            window_capacity_rows(
                retain,
                window_step_rows,
                compaction_granule_rows,
            )
            .min(full_capacity_rows)
        } else {
            full_capacity_rows
        };
        let even_odd_row_bytes = row_bytes(key_format, even_odd_row)?;
        let v_row_bytes = row_bytes(value_format, v_row)?;
        let k_even = acquire(source, layer, "k_even", capacity_rows * even_odd_row_bytes)?;
        let k_odd = acquire(source, layer, "k_odd", capacity_rows * even_odd_row_bytes)?;
        let v = acquire(source, layer, "v", capacity_rows * v_row_bytes)?;
        Ok(Self {
            #[cfg(feature = "instrument")]
            layer,
            k_even,
            k_odd,
            v,
            key_encoded: encoded(narrow_codecs.0.is_some(), key_format),
            value_encoded: encoded(narrow_codecs.1.is_some(), value_format),
            key_format: *key_format,
            key_codec,
            value_format: *value_format,
            value_codec,
            even_odd_row,
            v_row,
            even_odd_row_bytes,
            v_row_bytes,
            capacity_rows,
            window,
            retain,
            base_position: 0,
        })
    }

    /// The oldest position this layer keeps at `cached_len`: every one for a
    /// full layer, the last `retain` for a sliding one.
    fn first_kept(&self, cached_len: usize) -> usize {
        match self.window {
            Some(_) => cached_len - cached_len.min(self.retain),
            None => 0,
        }
    }

    /// Writes the kept host rows to the front of each buffer and zeroes the
    /// rest. The leaf reads past the live rows up to the bucket; a row never
    /// written must be a finite zero, not undefined memory, because the masked
    /// softmax weight 0.0 still multiplies it. Only the tail is zeroed: the
    /// head is overwritten here, so zeroing it first is a second pass over
    /// every byte of a long context.
    fn seed(&mut self, cache: &LayerCache, cached_len: usize) -> Result<(), InteropError> {
        #[cfg(feature = "instrument")]
        let write_started = read_ticks();
        let kept = self.write_kept(cache, cached_len)?;
        #[cfg(feature = "instrument")]
        let write_ns = ticks_to_nanos(elapsed_ticks(write_started));
        #[cfg(feature = "instrument")]
        let zero_started = read_ticks();
        self.zero_tail(kept)?;
        #[cfg(feature = "instrument")]
        let zero_ns = ticks_to_nanos(elapsed_ticks(zero_started));
        #[cfg(feature = "instrument")]
        debug!(
            layer = self.layer as u64,
            kept_rows = kept as u64,
            zero_bytes = ((self.capacity_rows - kept)
                * (2 * self.even_odd_row_bytes + self.v_row_bytes)) as u64,
            write_ns,
            zero_ns,
            "kv_adopt_seed"
        );
        Ok(())
    }

    fn write_kept(&mut self, cache: &LayerCache, cached_len: usize) -> Result<usize, InteropError> {
        let even_odd_row = self.even_odd_row;
        let v_row = self.v_row;
        let first = self.first_kept(cached_len);
        let kept = cached_len - first;
        self.base_position = first;
        for (slot_start, rows, target_row) in kept_segments(cache, first, kept)
            .into_iter()
            .filter(|(_, rows, _)| *rows > 0)
        {
            self.write_rows(cache, slot_start, rows, target_row, even_odd_row, v_row)?;
        }
        Ok(kept)
    }

    fn zero_tail(&self, kept: usize) -> Result<(), InteropError> {
        for (buffer, row_bytes) in [
            (&self.k_even, self.even_odd_row_bytes),
            (&self.k_odd, self.even_odd_row_bytes),
            (&self.v, self.v_row_bytes),
        ] {
            omega::metal::zero_placed_buffer_range(
                buffer,
                kept * row_bytes,
                (self.capacity_rows - kept) * row_bytes,
            )?;
        }
        Ok(())
    }

    fn write_rows(
        &self,
        cache: &LayerCache,
        slot_start: usize,
        rows: usize,
        target_row: usize,
        even_odd_row: usize,
        v_row: usize,
    ) -> Result<(), InteropError> {
        let mut wide = alloc::vec![0.0f32; rows * even_odd_row.max(v_row)];
        let even_odd = &mut wide[..rows * even_odd_row];
        cache.k_even.decode_range(slot_start * even_odd_row, even_odd)?;
        self.store_f32(&self.k_even, target_row * self.even_odd_row_bytes, even_odd)?;
        cache.k_odd.decode_range(slot_start * even_odd_row, even_odd)?;
        self.store_f32(&self.k_odd, target_row * self.even_odd_row_bytes, even_odd)?;
        let value = &mut wide[..rows * v_row];
        cache.v.decode_range(slot_start * v_row, value)?;
        self.store_value_f32(&self.v, target_row * self.v_row_bytes, value)
    }

    fn make_room(
        &mut self,
        cached_len: usize,
        new_count: usize,
        bound_extent: usize,
    ) -> Result<(), InteropError> {
        let Some(window) = self.window else {
            return Ok(());
        };
        let live_first_row = cached_len - cached_len.min(window) - self.base_position;
        let reach =
            (live_first_row + bound_extent).max(cached_len + new_count - self.base_position);
        if reach <= self.capacity_rows {
            return Ok(());
        }
        let kept_first = self.first_kept(cached_len);
        let kept_rows = cached_len - kept_first;
        let kept_first_row = kept_first - self.base_position;
        for (buffer, row_bytes) in [
            (&self.k_even, self.even_odd_row_bytes),
            (&self.k_odd, self.even_odd_row_bytes),
            (&self.v, self.v_row_bytes),
        ] {
            omega::move_placed_buffer_bytes(
                buffer,
                kept_first_row * row_bytes,
                0,
                kept_rows * row_bytes,
            )?;
        }
        self.base_position = kept_first;
        Ok(())
    }

    /// Whether the buffer still holds every row a rewind to `keep` positions would keep: a full
    /// layer always does, a sliding layer only when its `retain` rows behind `keep` lie at or after
    /// the buffer's first row (the same rows [`Self::first_kept`] hands [`DeviceKv::flush`]).
    fn holds_rewind_to(&self, keep: usize) -> bool {
        self.first_kept(keep) >= self.base_position
    }

    /// A rewind to `to_rows` positions needs no copy: the valid rows are the host `cached_len`. It
    /// only fails when a sliding layer already moved its window past the rewind point.
    fn truncate(&self, to_rows: usize) -> Result<(), InteropError> {
        if to_rows < self.base_position {
            return Err(InteropError::RewindIntoSealed {
                keep_positions: to_rows,
                sealed_end: self.base_position,
            });
        }
        Ok(())
    }

    /// The buffer is sized once at adopt; growth past it is a configuration error, not a reallocation.
    fn fits(&self, to_elements: usize) -> Result<(), InteropError> {
        let capacity_elements = self.capacity_rows * self.even_odd_row.max(1);
        if to_elements > capacity_elements {
            return Err(InteropError::UnsupportedServingConfig(alloc::format!(
                "device kv layer holds {capacity_elements} elements per field, a step asked for {to_elements}"
            )));
        }
        Ok(())
    }

    fn input_row(&self, cached_len: usize) -> usize {
        self.window.map_or(0, |window| {
            cached_len - cached_len.min(window) - self.base_position
        })
    }

    fn output_row(&self, cached_len: usize) -> usize {
        cached_len - self.base_position
    }

    /// Writes f32 rows into a key plane through the key format's codec.
    fn store_f32(
        &self,
        buffer: &PlacedBuffer,
        byte_offset: usize,
        values: &[f32],
    ) -> Result<(), InteropError> {
        omega::write_placed_buffer(buffer, byte_offset, &self.key_codec, values)
            .map_err(|error| numeric_failure(&self.key_format, error))
    }

    /// Reads f32 rows out of a key plane through the key format's codec.
    fn load_f32(
        &self,
        buffer: &PlacedBuffer,
        byte_offset: usize,
        count: usize,
    ) -> Result<Vec<f32>, InteropError> {
        omega::read_placed_buffer(buffer, byte_offset, &self.key_codec, count)
            .map_err(|error| numeric_failure(&self.key_format, error))
    }

    /// Writes f32 rows into the value buffer through the value format's codec.
    fn store_value_f32(
        &self,
        buffer: &PlacedBuffer,
        byte_offset: usize,
        values: &[f32],
    ) -> Result<(), InteropError> {
        omega::write_placed_buffer(buffer, byte_offset, &self.value_codec, values)
            .map_err(|error| numeric_failure(&self.value_format, error))
    }

    /// Reads f32 rows out of the value buffer through the value format's codec.
    fn load_value_f32(
        &self,
        buffer: &PlacedBuffer,
        byte_offset: usize,
        count: usize,
    ) -> Result<Vec<f32>, InteropError> {
        omega::read_placed_buffer(buffer, byte_offset, &self.value_codec, count)
            .map_err(|error| numeric_failure(&self.value_format, error))
    }
}

/// A granule of one rows is `retain + step` exactly, the tightest buffer: the tests that count compactions use it.
#[cfg(test)]
const BARE_MINIMUM_GRANULE_ROWS: usize = 1;
/// A decode program whose attention layers hand back their new K and V rows already in the
/// scalar layout the cache stores them as.
pub(super) struct DirectKvRows {
    pub(super) program: Vec<Op>,
    pub(super) layer_roots: Vec<LayerCacheRoots>,
}

/// `program` with each cache plane stored as a layout [`omega::device_row_store_dtype`] names
/// cast at its root, and `layer_roots` pointing at the casts: the producer values are unchanged
/// (the cast rounds once, as the host narrowing did) and [`DeviceKv::placements`] places them in
/// the cache rows with no f32 staging. `None` when neither plane qualifies, so an f32 or block
/// cache keeps its program. Composes [`proxima_tensor::spec::store_rows_as`].
///
/// # Errors
/// The tensor builder's refusal, which a rank-3 cache-row root cannot raise.
pub(super) fn with_direct_kv_rows(
    program: &[Op],
    layer_roots: &[LayerCacheRoots],
    (key_format, value_format): (&NumericFormat, &NumericFormat),
) -> Result<Option<DirectKvRows>, InteropError> {
    let key_store = omega::device_row_store_dtype(key_format);
    let value_store = omega::device_row_store_dtype(value_format);
    if key_store.is_none() && value_store.is_none() {
        return Ok(None);
    }
    let mut program = program.to_vec();
    let mut rewritten = Vec::with_capacity(layer_roots.len());
    for roots in layer_roots {
        let LayerCacheRoots::Attention((even, odd, value)) = roots else {
            rewritten.push(roots.clone());
            continue;
        };
        let mut store = |node: NodeId, dtype: Option<DType>| match dtype {
            Some(dtype) => store_rows_as(&mut program, node, dtype),
            None => Ok(node),
        };
        rewritten.push(LayerCacheRoots::Attention((
            store(*even, key_store)?,
            store(*odd, key_store)?,
            store(*value, value_store)?,
        )));
    }
    Ok(Some(DirectKvRows { program, layer_roots: rewritten }))
}

impl DeviceKv {
    /// Moves every attention layer's host rows onto the device, or `None`
    /// when this call's caches are not the plain attention shape
    /// (`SharedFromLayer` layers are fine: they own nothing) or a host cache
    /// does not hold exactly `cached_len` rows. Host copies are released here
    /// and rebuilt by [`Self::flush`], so a long context is held once.
    ///
    /// `capacity_positions` is the most positions any step can reach: the call's
    /// `positions_needed` plus the rows of a speculative draft. `window_step_rows`
    /// is the widest step of any kind, a prefill batch included, and sizes a sliding
    /// layer's linear buffer. `step_rows` is the widest step still to run after this
    /// adoption: it bounds every later step. A prefill batch ends inside
    /// `positions_needed`, so it adds no full-layer rows.
    ///
    /// `compaction_granule_rows` rounds a sliding layer's `retain + window_step_rows` buffer up to a
    /// multiple: the slack is how many tokens pass between compactions. A serving call passes
    /// `omega::sized::KV_WINDOW_COMPACTION_GRANULE_ROWS`, set in `omega-runtime.toml` `[kv_window]`.
    ///
    /// `key_format` stores the K planes and `value_format` the V rows; a block key layout beside
    /// a scalar value is the one pair that differs, see [`kernel_codecs`].
    pub(super) fn adopt_pair(
        layer_caches: &mut [LayerCacheState],
        layer_row_widths: &[LayerPadRowWidths],
        cached_len: usize,
        (capacity_positions, bucket_tokens): (usize, usize),
        (window_step_rows, step_rows, compaction_granule_rows): (usize, usize, usize),
        mut source: impl FnMut(usize) -> Result<PlacedBuffer, omega::MetalError>,
        (key_format, value_format): (NumericFormat, NumericFormat),
    ) -> Result<Option<Self>, InteropError> {
        let full_capacity_rows =
            kv_extent(capacity_positions, usize::MAX, bucket_tokens) + bucket_tokens;
        let mut layers: Vec<Option<DeviceKvLayer>> = Vec::with_capacity(layer_caches.len());
        for (index, (state, widths)) in layer_caches.iter().zip(layer_row_widths).enumerate() {
            match (state, widths) {
                (
                    LayerCacheState::Attention(cache),
                    LayerPadRowWidths::Attention {
                        even_odd_row,
                        v_row,
                    },
                ) => {
                    let host_rows = cache.k_even.elements() / (*even_odd_row).max(1);
                    let holds_every_position =
                        cache.ring_geometry().is_some() || host_rows == cached_len;
                    if !holds_every_position {
                        return Ok(None);
                    }
                    let mut device_layer = DeviceKvLayer::allocate(
                        index,
                        cache,
                        (*even_odd_row, *v_row),
                        full_capacity_rows,
                        (window_step_rows, compaction_granule_rows),
                        &mut source,
                        (&key_format, &value_format),
                    )?;
                    device_layer.seed(cache, cached_len)?;
                    layers.push(Some(device_layer));
                }
                (LayerCacheState::SharedFromLayer, LayerPadRowWidths::SharedFromLayer) => {
                    layers.push(None);
                }
                _ => return Ok(None),
            }
        }
        for state in layer_caches.iter_mut() {
            if let LayerCacheState::Attention(cache) = state {
                cache.k_even.release();
                cache.k_odd.release();
                cache.v.release();
            }
        }
        Ok(Some(Self {
            layers,
            max_step_rows: step_rows,
            key_format,
            value_format,
        }))
    }

    /// [`Self::adopt_pair`] with K and V stored in one `format`, the shape every uniform cache takes.
    #[cfg(test)]
    pub(super) fn adopt(
        layer_caches: &mut [LayerCacheState],
        layer_row_widths: &[LayerPadRowWidths],
        cached_len: usize,
        capacity: (usize, usize),
        max_step_rows: usize,
        source: impl FnMut(usize) -> Result<PlacedBuffer, omega::MetalError>,
        format: NumericFormat,
    ) -> Result<Option<Self>, InteropError> {
        Self::adopt_pair(
            layer_caches,
            layer_row_widths,
            cached_len,
            capacity,
            (max_step_rows, max_step_rows, BARE_MINIMUM_GRANULE_ROWS),
            source,
            (format, format),
        )
    }

    /// Gives up the cache buffers of every resident layer so the next [`Self::adopt`] can bind them
    /// instead of allocating.
    pub(super) fn into_buffers(self) -> Vec<PlacedBuffer> {
        self.layers
            .into_iter()
            .flatten()
            .flat_map(|layer| [layer.k_even, layer.k_odd, layer.v])
            .collect()
    }


    /// Takes a [`ResidentSlot`] back into a live cache and rewinds it to `keep` positions where it
    /// is: no seed (host to device) and no `zero_tail`, because a rewound layer only exposes rows
    /// it already wrote, finite values the program masks. A full layer is always [`RewindOutcome::InPlace`];
    /// a sliding layer is only when [`DeviceKvLayer::holds_rewind_to`], else the newest
    /// [`RingCheckpoint`] at or before `keep` is handed back in [`RewindOutcome::Checkpoint`], else
    /// the rewind is refused with [`MissReason::RingSlackExceeded`]. On anything but `InPlace` the
    /// returned cache is unchanged; the caller restores the checkpoint it was given. Staleness by [`CacheKey`] is the slot table's call, not this one's.
    ///
    /// # Errors
    /// `keep` past the positions the slot holds, which is a caller bug.
    pub(super) fn adopt_resident(
        slot: ResidentSlot,
        keep: usize,
    ) -> Result<(Self, RewindOutcome), InteropError> {
        let ResidentSlot { ids, key: _, device, checkpoints } = slot;
        let held = ids.len();
        if keep > held {
            return Err(InteropError::UnsupportedServingConfig(alloc::format!(
                "resident slot holds {held} positions, a rewind asked to keep {keep}"
            )));
        }
        let outcome = device.rewind_outcome(keep, held, checkpoints);
        Ok((device, outcome))
    }

    fn rewind_outcome(&self, keep: usize, held: usize, checkpoints: Vec<RingCheckpoint>) -> RewindOutcome {
        if keep == 0 {
            return RewindOutcome::Refused(MissReason::NoCommonPrefix);
        }
        let Some(short) = self.layers.iter().flatten().find(|layer| !layer.holds_rewind_to(keep)) else {
            return RewindOutcome::InPlace;
        };
        match checkpoints.into_iter().rev().find(|checkpoint| checkpoint.position() <= keep) {
            Some(checkpoint) => RewindOutcome::Checkpoint(checkpoint),
            None => RewindOutcome::Refused(MissReason::RingSlackExceeded {
                rewind_rows: held - keep,
                slack_rows: short.retain - short.window.unwrap_or(short.retain),
            }),
        }
    }

    /// Leaves this cache in place as a [`ResidentSlot`] holding `ids`, the positions its rows cover.
    /// Call it only after every command buffer of the request completed, since the slot changes
    /// threads between requests. `DeviceKv::adopt_resident` is the inverse.
    ///
    /// Each sliding layer's `base_position` is the oldest position its buffer holds: `0` until the
    /// buffer first compacts, then `len - capacity_rows` at most. A slot whose rows were written by
    /// its own steps, never seeded, keeps that value, so [`DeviceKvLayer::holds_rewind_to`] admits
    /// a rewind of up to `capacity_rows - retain` rows in place; only a seed pins the base at
    /// `first_kept(len)` and refuses every rewind.
    pub(super) fn into_slot(self, ids: Vec<u32>, key: CacheKey, checkpoints: Vec<RingCheckpoint>) -> ResidentSlot {
        ResidentSlot { ids, key, device: self, checkpoints }
    }

    /// Bytes of device KV this cache holds: the three planes of every resident layer, each sized as
    /// [`DeviceKvLayer::allocate`] asked the allocator for it (`capacity_rows` times the row bytes).
    pub(super) fn resident_bytes(&self) -> usize {
        self.layers
            .iter()
            .flatten()
            .map(|layer| layer.capacity_rows * (2 * layer.even_odd_row_bytes + layer.v_row_bytes))
            .sum()
    }

    pub(super) fn max_step_rows(&self) -> usize {
        self.max_step_rows
    }

    /// Whether these buffers, sized by an earlier request, hold what a request reaching
    /// `capacity_positions` (its `positions_needed` plus draft rows) with steps of up to `step_rows`
    /// would have allocated: [`Self::adopt_pair`]'s own sizing, read as a lower bound. A buffer is
    /// never grown after adoption, so a request that does not fit must not start from it.
    pub(super) fn fits_request(&self, capacity_positions: usize, bucket_tokens: usize, step_rows: usize) -> bool {
        let full_capacity_rows = kv_extent(capacity_positions, usize::MAX, bucket_tokens) + bucket_tokens;
        step_rows <= self.max_step_rows
            && self.layers.iter().flatten().all(|layer| match layer.window {
                Some(_) => layer.capacity_rows >= (layer.retain + step_rows).min(full_capacity_rows),
                None => layer.capacity_rows >= full_capacity_rows,
            })
    }

    /// The packed codec the leaf placements read this cache as, or `None` for
    /// f32 (a plain buffer needs no codec).
    ///
    /// # Errors
    /// The classifier's refusal, as at allocation: a format the cached-attention kernel cannot read.
    pub(super) fn cache_codec(&self) -> Result<Option<NumericFormat>, InteropError> {
        kernel_codecs(&self.key_format, &self.value_format).map(|(key, _)| key)
    }

    /// The packed codec the V leaf placements read, beside [`Self::cache_codec`] for the K planes.
    ///
    /// # Errors
    /// The classifier's refusal, as at allocation.
    pub(super) fn value_cache_codec(&self) -> Result<Option<NumericFormat>, InteropError> {
        kernel_codecs(&self.key_format, &self.value_format).map(|(_, value)| value)
    }

    /// Bytes `new_rows` rows of this step wrote into the device cache: every resident layer's K planes
    /// and V row, counted in the formats the cache was adopted with.
    #[cfg(feature = "instrument")]
    pub(super) fn step_write_bytes(&self, new_rows: usize) -> u64 {
        kv_cache_format::step_write_bytes(
            (&self.key_format, &self.value_format),
            new_rows as u64,
            self.layers
                .iter()
                .flatten()
                .map(|layer| (layer.even_odd_row as u64, layer.v_row as u64)),
        )
    }

    /// Runs one [`KvCommand`] the kv carry state machine decided for layer `input`: a `Truncate` moves
    /// no bytes (the valid rows are `cached_len`) and a `Grow` must fit the buffer sized at adopt. The
    /// copy, park and drain commands have no device-side counterpart here.
    pub(super) fn apply(&self, command: &KvCommand<usize>) -> Result<(), InteropError> {
        match *command {
            KvCommand::Truncate { input, to_rows } => self.resident(input)?.truncate(to_rows as usize),
            KvCommand::Grow { input, to_elements, .. } => self.resident(input)?.fits(to_elements as usize),
            _ => Ok(()),
        }
    }

    fn resident(&self, layer: usize) -> Result<&DeviceKvLayer, InteropError> {
        self.layers.get(layer).and_then(Option::as_ref).ok_or_else(|| {
            InteropError::UnsupportedServingConfig(alloc::format!(
                "layer {layer} keeps no device kv for a state machine command"
            ))
        })
    }

    /// One flag per layer, for the host paths that must skip a resident one.
    pub(super) fn resident_layers(&self) -> Vec<bool> {
        self.layers.iter().map(Option::is_some).collect()
    }

    /// Readies every sliding layer for a step at `cached_len` positions.
    /// `sliding_bound` is the program's sliding extent this step
    /// (`KvRing::bound_extent`). Separate from [`Self::placements`] so the
    /// placements borrow the cache shared while the step's placement lists are alive.
    pub(super) fn ready_step(
        &mut self,
        cached_len: usize,
        new_count: usize,
        sliding_bound: usize,
    ) -> Result<(), InteropError> {
        for layer in self.layers.iter_mut().flatten() {
            layer.make_room(cached_len, new_count, sliding_bound)?;
        }
        Ok(())
    }

    /// Names the placements a step runs under, after [`Self::ready_step`].
    pub(super) fn placements<'buffers>(
        &'buffers self,
        cached_len: usize,
        leaves: &[Option<KvLeafNodes>],
        layer_roots: &[LayerCacheRoots],
    ) -> KvPlacements<'buffers> {
        let mut inputs = Vec::with_capacity(self.layers.len() * 3);
        let mut outputs = Vec::with_capacity(self.layers.len() * 3);
        let mut encoded = Vec::new();
        for (index, slot) in self.layers.iter().enumerate() {
            let (
                Some(layer),
                Some(Some(leaf)),
                Some(LayerCacheRoots::Attention((even, odd, value))),
            ) = (slot, leaves.get(index), layer_roots.get(index))
            else {
                continue;
            };
            let input_row = layer.input_row(cached_len);
            let output_row = layer.output_row(cached_len);
            inputs.push((
                leaf.k_even,
                &layer.k_even,
                input_row * layer.even_odd_row_bytes,
            ));
            inputs.push((
                leaf.k_odd,
                &layer.k_odd,
                input_row * layer.even_odd_row_bytes,
            ));
            inputs.push((leaf.v, &layer.v, input_row * layer.v_row_bytes));
            outputs.push((*even, &layer.k_even, output_row * layer.even_odd_row_bytes));
            outputs.push((*odd, &layer.k_odd, output_row * layer.even_odd_row_bytes));
            outputs.push((*value, &layer.v, output_row * layer.v_row_bytes));
            if layer.key_encoded {
                encoded.push((*even, layer.key_format));
                encoded.push((*odd, layer.key_format));
            }
            if layer.value_encoded {
                encoded.push((*value, layer.value_format));
            }
        }
        KvPlacements { inputs, outputs, encoded }
    }

    /// Hands the rows back to the host caches so [`PrefixState`] and the
    /// prompt cache see exactly what a host-only run would have left.
    pub(super) fn flush(
        &self,
        layer_caches: &mut [LayerCacheState],
        cached_len: usize,
        positions_needed: usize,
    ) -> Result<(), InteropError> {
        #[cfg(feature = "instrument")]
        let flush_started = read_ticks();
        #[cfg(feature = "instrument")]
        let mut rows_written = 0u64;
        for (slot, state) in self.layers.iter().zip(layer_caches.iter_mut()) {
            let (Some(layer), LayerCacheState::Attention(cache)) = (slot, state) else {
                continue;
            };
            let first = layer.first_kept(cached_len);
            let live = cached_len - first;
            #[cfg(feature = "instrument")]
            {
                rows_written += live as u64;
            }
            let first_row = first - layer.base_position;
            let even_odd_row = layer.even_odd_row;
            let v_row = layer.v_row;
            let even = layer.load_f32(
                &layer.k_even,
                first_row * layer.even_odd_row_bytes,
                live * even_odd_row,
            )?;
            let odd = layer.load_f32(
                &layer.k_odd,
                first_row * layer.even_odd_row_bytes,
                live * even_odd_row,
            )?;
            let value =
                layer.load_value_f32(&layer.v, first_row * layer.v_row_bytes, live * v_row)?;
            cache.k_even.release();
            cache.k_odd.release();
            cache.v.release();
            cache.reserve_ring_rows(positions_needed)?;
            cache.append_at(first, &even, &odd, &value)?;
        }
        #[cfg(feature = "instrument")]
        debug!(
            rows_written,
            cached_len = cached_len as u64,
            flush_ns = ticks_to_nanos(elapsed_ticks(flush_started)),
            "kv_flush"
        );
        #[cfg(feature = "instrument")]
        debug!(
            host_mirror_bytes = host_mirror_bytes(layer_caches) as u64,
            cached_len = cached_len as u64,
            "kv_host_mirror"
        );
        Ok(())
    }
}

/// Host bytes the attention layers' K/V row vectors hold by allocated
/// capacity: the copy [`DeviceKv::flush`] leaves in the host caches.
#[cfg(feature = "instrument")]
fn host_mirror_bytes(layer_caches: &[LayerCacheState]) -> usize {
    layer_caches
        .iter()
        .filter_map(|state| match state {
            LayerCacheState::Attention(cache) => Some(cache.k_even.capacity_bytes() + cache.k_odd.capacity_bytes() + cache.v.capacity_bytes()),
            _ => None,
        })
        .sum()
}

/// The host-cache slot runs that hold positions `first..first + kept`, as
/// `(first slot, rows, device row)`. A full layer stores position `p` in slot
/// `p`: one run. A ring stores it in slot `p % capacity`, so the run wraps at
/// most once.
fn kept_segments(cache: &LayerCache, first: usize, kept: usize) -> [(usize, usize, usize); 2] {
    let Some(ring) = cache.ring_geometry() else {
        return [(first, kept, 0), (0, 0, kept)];
    };
    let start_slot = first % ring.capacity;
    let head_rows = kept.min(ring.capacity - start_slot);
    [(start_slot, head_rows, 0), (0, kept - head_rows, head_rows)]
}

/// Resolves every layer's `kv_cache.{layer}.*` leaf nodes in one pass over
/// `program` -- the per-program table [`DeviceKv::placements`] reads each
/// step, so no step searches the program by name.
pub(super) fn kv_leaf_nodes(
    program: &[Op],
    cache_names: &[LayerCacheNames],
) -> Vec<Option<KvLeafNodes>> {
    let mut by_name: BTreeMap<&str, NodeId> = BTreeMap::new();
    for (index, operation) in program.iter().enumerate() {
        if let Op::Input {
            name: Some(name), ..
        } = operation
        {
            by_name.insert(name.as_str(), NodeId(index as u32));
        }
    }
    cache_names
        .iter()
        .map(|names| match names {
            LayerCacheNames::Attention { k_even, k_odd, v } => Some(KvLeafNodes {
                k_even: *by_name.get(k_even.as_str())?,
                k_odd: *by_name.get(k_odd.as_str())?,
                v: *by_name.get(v.as_str())?,
            }),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
pub(in crate::generate) mod tests {
    use proxima_numeric::{FloatLayout, NumericPolicy};

    use super::*;
    use proxima_tensor::ScalarOp;

    const EVEN_ODD_ROW: usize = 4;
    const V_ROW: usize = 6;

    fn row_values(position: usize, width: usize, leaf: usize) -> Vec<f32> {
        (0..width)
            .map(|column| (position * 100 + column) as f32 + leaf as f32 * 0.5)
            .collect()
    }

    fn rows(first: usize, count: usize, width: usize, leaf: usize) -> Vec<f32> {
        (first..first + count)
            .flat_map(|position| row_values(position, width, leaf))
            .collect()
    }

    fn host_cache(window: Option<usize>, positions: usize) -> LayerCache {
        let mut cache = test_attention_cache(window, 2, EVEN_ODD_ROW, V_ROW, 0, positions);
        cache.append_at(
            0,
            &rows(0, positions, EVEN_ODD_ROW, 0),
            &rows(0, positions, EVEN_ODD_ROW, 1),
            &rows(0, positions, V_ROW, 2),
        ).expect("kv rows store");
        cache
    }

    fn adopted(
        window: Option<usize>,
        positions: usize,
        total_positions: usize,
    ) -> (DeviceKv, Vec<LayerCacheState>) {
        let mut caches = alloc::vec![LayerCacheState::Attention(host_cache(window, positions))];
        let widths = [LayerPadRowWidths::Attention {
            even_odd_row: EVEN_ODD_ROW,
            v_row: V_ROW,
        }];
        let device = DeviceKv::adopt(
            &mut caches,
            &widths,
            positions,
            (total_positions, 4),
            3,
            allocate_placed_buffer,
            NumericFormat::Float32,
        )
        .expect("device kv allocates on the real Metal device")
        .expect("a plain attention cache is adoptable");
        (device, caches)
    }

    fn write_step_rows(device: &DeviceKv, cached_len: usize) {
        let layer = device.layers[0].as_ref().expect("layer 0 is resident");
        let row = layer.output_row(cached_len);
        omega::try_write_placed_buffer_f32(
            &layer.k_even,
            row * layer.even_odd_row_bytes,
            &row_values(cached_len, EVEN_ODD_ROW, 0),
        )
        .expect("the output row lies inside k_even");
        omega::try_write_placed_buffer_f32(
            &layer.k_odd,
            row * layer.even_odd_row_bytes,
            &row_values(cached_len, EVEN_ODD_ROW, 1),
        )
        .expect("the output row lies inside k_odd");
        omega::try_write_placed_buffer_f32(
            &layer.v,
            row * layer.v_row_bytes,
            &row_values(cached_len, V_ROW, 2),
        )
        .expect("the output row lies inside v");
    }

    #[test]
    fn a_full_layer_round_trips_through_the_device_unchanged() {
        let (device, mut caches) = adopted(None, 37, 80);
        let LayerCacheState::Attention(released) = &caches[0] else {
            panic!("layer 0 stays an attention cache");
        };
        assert!(
            released.k_even.is_empty() && released.v.is_empty(),
            "adoption releases the host copy so a long context is held once"
        );

        device.flush(&mut caches, 37, 80).expect("device kv flushes to host");

        let LayerCacheState::Attention(restored) = &caches[0] else {
            panic!("layer 0 stays an attention cache");
        };
        assert_eq!(wide(&restored.k_even), rows(0, 37, EVEN_ODD_ROW, 0));
        assert_eq!(wide(&restored.k_odd), rows(0, 37, EVEN_ODD_ROW, 1));
        assert_eq!(wide(&restored.v), rows(0, 37, V_ROW, 2));
    }

    #[test]
    fn a_513_row_window_with_512_row_steps_gets_1280_rows_not_the_doubled_1538() {
        let (retain, step) = (513, 512);
        assert_eq!(
            window_capacity_rows(retain, step, omega::sized::KV_WINDOW_COMPACTION_GRANULE_ROWS as usize),
            1280,
            "retain + step = 1025 rounds up to five 256-row granules, llama's sliding cell count"
        );
        assert_eq!(window_capacity_rows(retain, step, 1), retain + step, "a granule of one is the bare minimum");
        assert_eq!(window_capacity_rows(retain, step, 0), retain + step, "a zero granule adds no slack");
    }

    #[test]
    fn a_sliding_layer_keeps_its_whole_ring_across_every_compaction() {
        let window = 8;
        let start = 21;
        let steps = 60;
        let (mut device, mut caches) = adopted(Some(window), start, start + steps);
        let mut compactions = 0;

        for step in 0..steps {
            let cached_len = start + step;
            let before = device.layers[0].as_ref().map(|layer| layer.base_position);
            device.layers[0]
                .as_mut()
                .expect("layer 0 is resident")
                .make_room(cached_len, 1, window)
                .expect("the kept rows lie inside the layer buffers");
            let layer = device.layers[0].as_ref().expect("layer 0 is resident");
            compactions += usize::from(Some(layer.base_position) != before);
            assert_eq!(
                layer.input_row(cached_len),
                cached_len - cached_len.min(window) - layer.base_position,
                "the leaf placement starts at the oldest live row"
            );
            write_step_rows(&device, cached_len);
        }

        assert!(
            compactions >= 3,
            "60 steps over a 2x8+3 buffer must compact repeatedly, saw {compactions}"
        );
        let final_len = start + steps;
        device.flush(&mut caches, final_len, start + steps).expect("device kv flushes to host");
        let LayerCacheState::Attention(restored) = &caches[0] else {
            panic!("layer 0 stays an attention cache");
        };
        let ring = restored
            .ring_geometry()
            .copied()
            .expect("sliding layer keeps its ring");
        for position in final_len - ring.capacity..final_len {
            let slot = position % ring.capacity;
            assert_eq!(
                wide(&restored.k_even)[slot * EVEN_ODD_ROW..(slot + 1) * EVEN_ODD_ROW],
                row_values(position, EVEN_ODD_ROW, 0)[..],
                "position {position} survives in its ring slot"
            );
            assert_eq!(
                wide(&restored.v)[slot * V_ROW..(slot + 1) * V_ROW],
                row_values(position, V_ROW, 2)[..]
            );
        }
    }

    #[test]
    fn a_ring_round_trips_through_the_device_with_its_rewind_slack_rows() {
        let positions = 21;
        let before = host_cache(Some(8), positions);
        let ring = before
            .ring_geometry()
            .copied()
            .expect("sliding layer has a ring");
        assert_eq!(ring.capacity, 10, "8 window rows plus 2 slack rows");
        let (device, mut caches) = adopted(Some(8), positions, 60);

        device.flush(&mut caches, positions, 60).expect("device kv flushes to host");

        let LayerCacheState::Attention(restored) = &caches[0] else {
            panic!("layer 0 stays an attention cache");
        };
        for position in positions - ring.capacity..positions {
            let slot = position % ring.capacity;
            assert_eq!(
                wide(&restored.k_even)[slot * EVEN_ODD_ROW..(slot + 1) * EVEN_ODD_ROW],
                wide(&before.k_even)[slot * EVEN_ODD_ROW..(slot + 1) * EVEN_ODD_ROW],
                "position {position}, older than the window but inside the slack, is still there"
            );
        }
    }

    #[test]
    fn seeding_zeroes_every_row_past_the_kept_ones_whatever_the_buffer_held() {
        let positions = 37;
        let capacity_rows = 48;
        let cache = host_cache(None, positions);
        let mut layer = DeviceKvLayer::allocate(
            0,
            &cache,
            (EVEN_ODD_ROW, V_ROW),
            capacity_rows,
            (3, BARE_MINIMUM_GRANULE_ROWS),
            &mut allocate_placed_buffer,
            (
                &NumericFormat::Float32,
                &NumericFormat::Float32,
            ),
        )
        .expect("device kv allocates on the real Metal device");
        let dirty = alloc::vec![f32::NAN; capacity_rows * EVEN_ODD_ROW.max(V_ROW)];
        omega::try_write_placed_buffer_f32(&layer.k_even, 0, &dirty[..capacity_rows * EVEN_ODD_ROW])
            .expect("capacity_rows rows fill k_even");
        omega::try_write_placed_buffer_f32(&layer.k_odd, 0, &dirty[..capacity_rows * EVEN_ODD_ROW])
            .expect("capacity_rows rows fill k_odd");
        omega::try_write_placed_buffer_f32(&layer.v, 0, &dirty[..capacity_rows * V_ROW])
            .expect("capacity_rows rows fill v");

        layer.seed(&cache, positions).expect("layer seeds");

        let tail_rows = capacity_rows - positions;
        let tail = |buffer: &PlacedBuffer, row_bytes: usize, row: usize| {
            omega::try_read_placed_buffer_f32(buffer, positions * row_bytes, tail_rows * row)
                .expect("the rows past the kept ones lie inside the buffer")
        };
        assert!(
            tail(&layer.k_even, layer.even_odd_row_bytes, EVEN_ODD_ROW)
                .iter()
                .chain(&tail(&layer.k_odd, layer.even_odd_row_bytes, EVEN_ODD_ROW))
                .chain(&tail(&layer.v, layer.v_row_bytes, V_ROW))
                .all(|value| value.to_bits() == 0),
            "rows the call never wrote read as +0.0, not the buffer's previous bytes"
        );
        assert_eq!(
            omega::try_read_placed_buffer_f32(&layer.k_even, 0, positions * EVEN_ODD_ROW)
                .expect("the kept rows lie inside k_even"),
            rows(0, positions, EVEN_ODD_ROW, 0),
            "the kept rows are the host rows, in order"
        );
    }

    fn adopted_before_prefill(
        window: Option<usize>,
        total_positions: usize,
        prefill_rows: usize,
    ) -> (DeviceKv, Vec<LayerCacheState>) {
        let mut caches = alloc::vec![LayerCacheState::Attention(host_cache(window, 0))];
        let widths = [LayerPadRowWidths::Attention {
            even_odd_row: EVEN_ODD_ROW,
            v_row: V_ROW,
        }];
        let device = DeviceKv::adopt(
            &mut caches,
            &widths,
            0,
            (total_positions + 3, 4),
            prefill_rows,
            allocate_placed_buffer,
            NumericFormat::Float32,
        )
        .expect("device kv allocates on the real Metal device")
        .expect("an empty attention cache is adoptable");
        (device, caches)
    }

    fn write_prefill_rows(device: &DeviceKv, rows_written: usize) {
        let layer = device.layers[0].as_ref().expect("layer 0 is resident");
        let row = layer.output_row(0);
        omega::try_write_placed_buffer_f32(
            &layer.k_even,
            row * layer.even_odd_row_bytes,
            &rows(0, rows_written, EVEN_ODD_ROW, 0),
        )
        .expect("the prefill rows lie inside k_even");
        omega::try_write_placed_buffer_f32(
            &layer.k_odd,
            row * layer.even_odd_row_bytes,
            &rows(0, rows_written, EVEN_ODD_ROW, 1),
        )
        .expect("the prefill rows lie inside k_odd");
        omega::try_write_placed_buffer_f32(
            &layer.v,
            row * layer.v_row_bytes,
            &rows(0, rows_written, V_ROW, 2),
        )
        .expect("the prefill rows lie inside v");
    }

    #[test]
    fn a_prefill_written_into_an_empty_adoption_flushes_back_as_host_rows() {
        let prefill_rows = 21;
        let (device, mut caches) = adopted_before_prefill(None, 40, prefill_rows);
        let layer = device.layers[0].as_ref().expect("layer 0 is resident");
        assert_eq!(
            layer.capacity_rows,
            kv_extent(40 + 3, usize::MAX, 4) + 4,
            "a prefill batch ends inside positions_needed, so it adds no full-layer rows"
        );
        assert_eq!(
            layer.output_row(0),
            0,
            "the prefill's rows start at the buffer's first row"
        );

        write_prefill_rows(&device, prefill_rows);
        device.flush(&mut caches, prefill_rows, 40).expect("device kv flushes to host");

        let LayerCacheState::Attention(restored) = &caches[0] else {
            panic!("layer 0 stays an attention cache");
        };
        assert_eq!(wide(&restored.k_even), rows(0, prefill_rows, EVEN_ODD_ROW, 0));
        assert_eq!(wide(&restored.k_odd), rows(0, prefill_rows, EVEN_ODD_ROW, 1));
        assert_eq!(wide(&restored.v), rows(0, prefill_rows, V_ROW, 2));
    }

    #[test]
    fn a_sliding_layer_holds_a_whole_prefill_longer_than_its_window_and_keeps_its_ring() {
        let prefill_rows = 21;
        let (mut device, mut caches) = adopted_before_prefill(Some(8), 60, prefill_rows);
        let layer = device.layers[0].as_mut().expect("layer 0 is resident");
        assert!(
            layer.capacity_rows >= prefill_rows,
            "the linear buffer must hold the whole prefill batch, saw {} rows",
            layer.capacity_rows
        );
        layer.make_room(0, prefill_rows, 8).expect("a linear layer never moves rows");
        assert_eq!(
            layer.base_position, 0,
            "a prefill from an empty cache never compacts"
        );

        write_prefill_rows(&device, prefill_rows);
        device.flush(&mut caches, prefill_rows, 60).expect("device kv flushes to host");

        let LayerCacheState::Attention(restored) = &caches[0] else {
            panic!("layer 0 stays an attention cache");
        };
        let ring = restored
            .ring_geometry()
            .copied()
            .expect("sliding layer keeps its ring");
        for position in prefill_rows - ring.capacity..prefill_rows {
            let slot = position % ring.capacity;
            assert_eq!(
                wide(&restored.k_even)[slot * EVEN_ODD_ROW..(slot + 1) * EVEN_ODD_ROW],
                row_values(position, EVEN_ODD_ROW, 0)[..],
                "position {position} of the prefill survives in its ring slot"
            );
            assert_eq!(
                wide(&restored.v)[slot * V_ROW..(slot + 1) * V_ROW],
                row_values(position, V_ROW, 2)[..]
            );
        }
    }

    static REQUESTED: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

    fn recording_source(byte_len: usize) -> Result<PlacedBuffer, omega::MetalError> {
        REQUESTED
            .lock()
            .expect("the recording lock is never poisoned")
            .push(byte_len);
        allocate_placed_buffer(byte_len)
    }

    #[test]
    fn adopt_takes_every_buffer_from_the_given_source() {
        let mut caches = alloc::vec![
            LayerCacheState::Attention(host_cache(None, 37)),
            LayerCacheState::Attention(host_cache(Some(8), 37)),
        ];
        let widths = [
            LayerPadRowWidths::Attention {
                even_odd_row: EVEN_ODD_ROW,
                v_row: V_ROW,
            },
            LayerPadRowWidths::Attention {
                even_odd_row: EVEN_ODD_ROW,
                v_row: V_ROW,
            },
        ];

        let device = DeviceKv::adopt(
            &mut caches,
            &widths,
            37,
            (80, 4),
            3,
            recording_source,
            NumericFormat::Float32,
        )
        .expect("device kv allocates on the real Metal device");

        assert!(device.is_some(), "two plain attention layers are adoptable");
        let requested = REQUESTED
            .lock()
            .expect("the recording lock is never poisoned");
        assert_eq!(requested.len(), 6, "three buffers per layer, two layers");
        for sizes in requested.chunks(3) {
            assert!(sizes[0] > 0, "a layer buffer is never empty");
            assert_eq!(sizes[0], sizes[1], "k_even and k_odd are the same size");
            assert_eq!(
                sizes[0] * V_ROW,
                sizes[2] * EVEN_ODD_ROW,
                "v holds the same rows at its own width"
            );
        }
    }

    #[test]
    fn buffers_one_generation_gives_back_serve_the_next_and_are_zeroed_past_its_rows() {
        let widths = [LayerPadRowWidths::Attention {
            even_odd_row: EVEN_ODD_ROW,
            v_row: V_ROW,
        }];
        let mut first_caches = alloc::vec![LayerCacheState::Attention(host_cache(None, 37))];
        let first = DeviceKv::adopt(
            &mut first_caches,
            &widths,
            37,
            (80, 4),
            3,
            allocate_placed_buffer,
            NumericFormat::Float32,
        )
        .expect("device kv allocates on the real Metal device")
        .expect("a plain attention layer is adoptable");
        let mut kept = first.into_buffers();
        assert_eq!(
            kept.len(),
            3,
            "k_even, k_odd and v of the one layer come back"
        );
        for buffer in &kept {
            let floats = resident_plans::kv_bytes(core::slice::from_ref(buffer)) / size_of::<f32>();
            omega::try_write_placed_buffer_f32(buffer, 0, &alloc::vec![f32::NAN; floats])
                .expect("the buffer holds the floats its own byte length names");
        }

        let mut allocations = 0;
        let mut second_caches = alloc::vec![LayerCacheState::Attention(host_cache(None, 37))];
        let second = DeviceKv::adopt(
            &mut second_caches,
            &widths,
            37,
            (80, 4),
            3,
            |bytes| {
                resident_plans::take_kv_buffer(&mut kept, bytes).map_or_else(
                    || {
                        allocations += 1;
                        allocate_placed_buffer(bytes)
                    },
                    Ok,
                )
            },
            NumericFormat::Float32,
        )
        .expect("device kv binds the kept buffers")
        .expect("a plain attention layer is adoptable");

        assert_eq!(
            allocations, 0,
            "every buffer of the second generation was bound, none allocated"
        );
        assert!(
            kept.is_empty(),
            "the second generation took all three kept buffers"
        );
        let layer = second.layers[0].as_ref().expect("layer 0 is resident");
        let tail_rows = layer.capacity_rows - 37;
        let tail = omega::try_read_placed_buffer_f32(
            &layer.k_even,
            37 * layer.even_odd_row_bytes,
            tail_rows * EVEN_ODD_ROW,
        )
        .expect("the rows past the seeded ones lie inside k_even");
        assert!(
            tail.iter().all(|value| *value == 0.0),
            "stale NaN rows from the first generation are zeroed past the seeded rows"
        );
        let head = omega::try_read_placed_buffer_f32(&layer.v, 0, 37 * V_ROW)
            .expect("the seeded rows lie inside v");
        assert_eq!(
            head,
            rows(0, 37, V_ROW, 2),
            "the seeded rows are the second generation's own"
        );
    }

    #[test]
    fn a_cache_that_misses_positions_is_not_adopted() {
        let mut caches = alloc::vec![LayerCacheState::Attention(host_cache(None, 10))];
        let widths = [LayerPadRowWidths::Attention {
            even_odd_row: EVEN_ODD_ROW,
            v_row: V_ROW,
        }];
        let adopted = DeviceKv::adopt(
            &mut caches,
            &widths,
            12,
            (40, 4),
            3,
            allocate_placed_buffer,
            NumericFormat::Float32,
        )
        .expect("the allocation path itself succeeds");
        assert!(
            adopted.is_none(),
            "10 host rows cannot stand for 12 positions"
        );
        let LayerCacheState::Attention(untouched) = &caches[0] else {
            panic!("layer 0 stays an attention cache");
        };
        assert_eq!(
            untouched.k_even.elements(),
            10 * EVEN_ODD_ROW,
            "a declined cache keeps its host rows"
        );
    }

    /// Quarter-step values: every one is exactly representable in binary16
    /// (at most 80.25 here, where the spacing is 0.0625), so a round trip
    /// through a half-width cache must return them bit for bit.
    fn half_exact_rows(first: usize, count: usize, width: usize, leaf: usize) -> Vec<f32> {
        (first..first + count)
            .flat_map(|position| {
                (0..width)
                    .map(move |column| (position * 8 + column) as f32 * 0.25 + leaf as f32 * 0.5)
            })
            .collect()
    }

    fn half_exact_cache(positions: usize) -> LayerCache {
        let mut cache = test_attention_cache(None, 2, EVEN_ODD_ROW, V_ROW, 0, positions);
        cache.append_at(
            0,
            &half_exact_rows(0, positions, EVEN_ODD_ROW, 0),
            &half_exact_rows(0, positions, EVEN_ODD_ROW, 1),
            &half_exact_rows(0, positions, V_ROW, 2),
        ).expect("kv rows store");
        cache
    }

    fn adopted_half(positions: usize, total_positions: usize) -> (DeviceKv, Vec<LayerCacheState>) {
        let mut caches = alloc::vec![LayerCacheState::Attention(half_exact_cache(positions))];
        let widths = [LayerPadRowWidths::Attention {
            even_odd_row: EVEN_ODD_ROW,
            v_row: V_ROW,
        }];
        let device = DeviceKv::adopt(
            &mut caches,
            &widths,
            positions,
            (total_positions, 4),
            3,
            allocate_placed_buffer,
            NumericFormat::Float16,
        )
        .expect("device kv allocates on the real Metal device")
        .expect("a plain attention cache is adoptable");
        (device, caches)
    }

    fn bf16_exact_rows(first: usize, count: usize, width: usize, leaf: usize) -> Vec<f32> {
        (first..first + count)
            .flat_map(|position| {
                (0..width).map(move |column| (position * 2 + column) as f32 + leaf as f32 * 0.5)
            })
            .collect()
    }

    /// Stores one step's rows at the output row through the host codec: what the plan's rendered encode does on the device.
    fn store_step_rows(device: &DeviceKv, cached_len: usize, rows: [&[f32]; 3]) {
        let layer = device.layers[0].as_ref().expect("layer 0 is resident");
        let output_row = layer.output_row(cached_len);
        layer
            .store_f32(&layer.k_even, output_row * layer.even_odd_row_bytes, rows[0])
            .expect("the even rows fit the cache");
        layer
            .store_f32(&layer.k_odd, output_row * layer.even_odd_row_bytes, rows[1])
            .expect("the odd rows fit the cache");
        layer
            .store_value_f32(&layer.v, output_row * layer.v_row_bytes, rows[2])
            .expect("the value rows fit the cache");
    }

    fn bf16_exact_cache(positions: usize) -> LayerCache {
        let mut cache = test_attention_cache(None, 2, EVEN_ODD_ROW, V_ROW, 0, positions);
        cache.append_at(
            0,
            &bf16_exact_rows(0, positions, EVEN_ODD_ROW, 0),
            &bf16_exact_rows(0, positions, EVEN_ODD_ROW, 1),
            &bf16_exact_rows(0, positions, V_ROW, 2),
        ).expect("kv rows store");
        cache
    }

    fn adopted_bf16(positions: usize, total_positions: usize) -> (DeviceKv, Vec<LayerCacheState>) {
        let mut caches = alloc::vec![LayerCacheState::Attention(bf16_exact_cache(positions))];
        let widths = [LayerPadRowWidths::Attention {
            even_odd_row: EVEN_ODD_ROW,
            v_row: V_ROW,
        }];
        let device = DeviceKv::adopt(
            &mut caches,
            &widths,
            positions,
            (total_positions, 4),
            3,
            allocate_placed_buffer,
            NumericFormat::BFloat16,
        )
        .expect("device kv allocates on the real Metal device")
        .expect("a plain attention cache is adoptable");
        (device, caches)
    }

    #[test]
    fn card_05_bf16_device_adopts_appends_and_flushes_rows() {
        let cached_len = 5;
        let (device, mut caches) = adopted_bf16(cached_len, 80);
        let layer = device.layers[0].as_ref().expect("layer 0 is resident");
        assert_eq!(layer.even_odd_row_bytes, EVEN_ODD_ROW * 2);
        assert_eq!(layer.v_row_bytes, V_ROW * 2);
        assert_eq!(device.cache_codec().expect("bf16 is readable"), Some(Codec::BFloat16));
        assert!(layer.key_encoded && layer.value_encoded, "BF16 cache rows are encoded into the cache");

        let even = [0.1f32, 0.2, 0.3, 0.4];
        let odd = [-0.1f32, -0.2, -0.3, -0.4];
        let value = [1.1f32, 1.2, 1.3, 1.4, 1.5, 1.6];
        store_step_rows(&device, cached_len, [&even, &odd, &value]);
        device.flush(&mut caches, cached_len + 1, 80).expect("device kv flushes to host");

        let LayerCacheState::Attention(restored) = &caches[0] else {
            panic!("layer 0 stays an attention cache");
        };
        let rounded = |values: &[f32]| {
            values
                .iter()
                .map(|value| half::bf16::from_f32(*value).to_f32())
                .collect::<Vec<_>>()
        };
        let mut expected_even = bf16_exact_rows(0, cached_len, EVEN_ODD_ROW, 0);
        expected_even.extend(rounded(&even));
        let mut expected_odd = bf16_exact_rows(0, cached_len, EVEN_ODD_ROW, 1);
        expected_odd.extend(rounded(&odd));
        let mut expected_value = bf16_exact_rows(0, cached_len, V_ROW, 2);
        expected_value.extend(rounded(&value));
        assert_eq!(wide(&restored.k_even), expected_even);
        assert_eq!(wide(&restored.k_odd), expected_odd);
        assert_eq!(wide(&restored.v), expected_value);
    }

    include!("../../../proxima-numeric/tests/support/production_text.rs");

    fn identifiers_after<'text>(text: &'text str, prefix: &str) -> Vec<&'text str> {
        text.match_indices(prefix)
            .map(|(offset, _)| {
                let tail = &text[offset + prefix.len()..];
                let end = tail
                    .find(|character: char| !character.is_alphanumeric() && character != '_')
                    .unwrap_or(tail.len());
                &tail[..end]
            })
            .collect()
    }

    #[test]
    fn device_kv_production_text_names_no_scalar_format_and_no_layout() {
        let production = production_of(include_str!("device_kv.rs"));
        let format_names = ["Float32", "Float16", "BFloat16", "BFloat8"];

        let named: Vec<&str> = ["NumericFormat::", "Codec::"]
            .into_iter()
            .flat_map(|prefix| identifiers_after(&production, prefix))
            .filter(|name| format_names.contains(name))
            .collect();
        let layouts: Vec<&str> = identifiers_after(&production, "FloatLayout::")
            .into_iter()
            .filter(|name| name.starts_with(|first: char| first.is_ascii_uppercase()))
            .collect();

        assert_eq!(named, Vec::<&str>::new());
        assert_eq!(layouts, Vec::<&str>::new());
    }

    #[test]
    fn device_kv_production_text_has_no_kv_element_type_and_never_names_ggml_type() {
        let production = production_of(include_str!("device_kv.rs"));

        let ggml_lines: Vec<&str> = production
            .lines()
            .filter(|line| !line.trim_start().starts_with("//") && line.contains("GgmlType"))
            .collect();

        assert!(!production.contains("KvElement"), "the cache element is a NumericFormat, not a parallel enum");
        assert_eq!(ggml_lines, Vec::<&str>::new());
    }

    #[test]
    fn a_binary32_cache_spelled_with_an_explicit_bias_is_adopted_as_a_plain_buffer_and_round_trips() {
        let respelled_f32 = NumericFormat::Scalar(FloatLayout {
            bias_override: Some(FloatLayout::BF32.bias()),
            ..FloatLayout::BF32
        });
        assert_ne!(respelled_f32, NumericFormat::Float32, "the spelling is not the named constant");
        let mut caches = alloc::vec![LayerCacheState::Attention(host_cache(None, 37))];
        let widths = [LayerPadRowWidths::Attention {
            even_odd_row: EVEN_ODD_ROW,
            v_row: V_ROW,
        }];

        let device = DeviceKv::adopt(
            &mut caches,
            &widths,
            37,
            (80, 4),
            3,
            allocate_placed_buffer,
            respelled_f32,
        )
        .expect("device kv allocates on the real Metal device")
        .expect("a plain attention cache is adoptable");

        assert_eq!(device.cache_codec().expect("respelled f32 is readable"), None);
        device.flush(&mut caches, 37, 80).expect("device kv flushes to host");
        let LayerCacheState::Attention(restored) = &caches[0] else {
            panic!("layer 0 stays an attention cache");
        };
        assert_eq!(wide(&restored.k_even), rows(0, 37, EVEN_ODD_ROW, 0));
        assert_eq!(wide(&restored.k_odd), rows(0, 37, EVEN_ODD_ROW, 1));
        assert_eq!(wide(&restored.v), rows(0, 37, V_ROW, 2));
    }

    fn bf8_exact_rows(first: usize, count: usize, width: usize, leaf: usize) -> Vec<f32> {
        let values = [0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.25];
        (first..first + count)
            .flat_map(|position| {
                (0..width).map(move |column| values[(position + column + leaf) % values.len()])
            })
            .collect()
    }

    fn adopted_bf8(positions: usize, total_positions: usize) -> (DeviceKv, Vec<LayerCacheState>) {
        let mut cache = test_attention_cache(None, 2, EVEN_ODD_ROW, V_ROW, 0, positions);
        cache.append_at(
            0,
            &bf8_exact_rows(0, positions, EVEN_ODD_ROW, 0),
            &bf8_exact_rows(0, positions, EVEN_ODD_ROW, 1),
            &bf8_exact_rows(0, positions, V_ROW, 2),
        ).expect("kv rows store");
        let mut caches = alloc::vec![LayerCacheState::Attention(cache)];
        let widths = [LayerPadRowWidths::Attention {
            even_odd_row: EVEN_ODD_ROW,
            v_row: V_ROW,
        }];
        let device = DeviceKv::adopt(
            &mut caches,
            &widths,
            positions,
            (total_positions, 4),
            3,
            allocate_placed_buffer,
            NumericFormat::BFloat8,
        )
        .expect("device kv allocates on the real Metal device")
        .expect("a plain attention cache is adoptable");
        (device, caches)
    }

    #[test]
    fn card_08_bf8_device_adopts_appends_and_flushes_rows() {
        let cached_len = 5;
        let (device, mut caches) = adopted_bf8(cached_len, 80);
        let layer = device.layers[0].as_ref().expect("layer 0 is resident");
        assert_eq!(layer.even_odd_row_bytes, EVEN_ODD_ROW);
        assert_eq!(layer.v_row_bytes, V_ROW);
        assert_eq!(device.cache_codec().expect("bf8 is readable"), Some(Codec::BFloat8));
        assert!(layer.key_encoded && layer.value_encoded, "BF8 cache rows are encoded into the cache");

        let even = [0.5f32, -0.5, 1.0, -1.0];
        let odd = [-1.0f32, 1.0, -2.0, 2.0];
        let value = [2.0f32, -2.0, 0.25, -0.25, 0.5, -0.5];
        store_step_rows(&device, cached_len, [&even, &odd, &value]);
        device.flush(&mut caches, cached_len + 1, 80).expect("device kv flushes to host");

        let LayerCacheState::Attention(restored) = &caches[0] else {
            panic!("layer 0 stays an attention cache");
        };
        let mut expected_even = bf8_exact_rows(0, cached_len, EVEN_ODD_ROW, 0);
        expected_even.extend(even);
        let mut expected_odd = bf8_exact_rows(0, cached_len, EVEN_ODD_ROW, 1);
        expected_odd.extend(odd);
        let mut expected_value = bf8_exact_rows(0, cached_len, V_ROW, 2);
        expected_value.extend(value);
        assert_eq!(wide(&restored.k_even), expected_even);
        assert_eq!(wide(&restored.k_odd), expected_odd);
        assert_eq!(wide(&restored.v), expected_value);
    }

    #[test]
    fn a_q4_0_key_beside_a_float16_value_sizes_blocks_for_k_and_halves_for_v_and_flushes_what_the_codecs_decode() {
        const WIDE_ROW: usize = 64;
        let positions = 9;
        let mut cache = test_attention_cache(None, 2, WIDE_ROW, V_ROW, 0, positions);
        let even = rows(0, positions, WIDE_ROW, 0);
        let odd = rows(0, positions, WIDE_ROW, 1);
        let value = rows(0, positions, V_ROW, 2);
        cache.append_at(0, &even, &odd, &value).expect("kv rows store");
        let mut caches = alloc::vec![LayerCacheState::Attention(cache)];
        let widths = [LayerPadRowWidths::Attention { even_odd_row: WIDE_ROW, v_row: V_ROW }];
        let q4_0 = NumericFormat::Q4_0;

        let device = DeviceKv::adopt_pair(
            &mut caches,
            &widths,
            positions,
            (40, 4),
            (3, 3, BARE_MINIMUM_GRANULE_ROWS),
            allocate_placed_buffer,
            (q4_0, NumericFormat::Float16),
        )
        .expect("a Q4_0 key beside a Float16 value is a readable pair")
        .expect("a plain attention cache is adoptable");

        let layer = device.layers[0].as_ref().expect("layer 0 is resident");
        assert_eq!(layer.even_odd_row_bytes, WIDE_ROW / 32 * 18, "two 18-byte Q4_0 blocks per 64-element row");
        assert_eq!(layer.v_row_bytes, V_ROW * 2, "binary16 value rows");
        assert!(
            layer.key_encoded && !layer.value_encoded,
            "the Q4_0 key is encoded by the plan; the binary16 value is a scalar layout the graph writes itself"
        );
        assert_eq!(device.cache_codec().expect("q4_0 is a readable key"), Some(q4_0));
        assert_eq!(device.value_cache_codec().expect("f16 is a readable value"), Some(Codec::Float16));

        device.flush(&mut caches, positions, 40).expect("device kv flushes to host");
        let LayerCacheState::Attention(restored) = &caches[0] else {
            panic!("layer 0 is an attention layer");
        };
        let (_, decoded_even) = {
            let codec = FormatCodec::for_format(&q4_0).expect("q4_0 resolves a codec");
            let mut packed = alloc::vec![0u8; codec.packed_len(even.len()).expect("whole blocks")];
            codec.encode(&even, &mut packed).expect("host encodes q4_0");
            let mut slots = alloc::vec![core::mem::MaybeUninit::<f32>::uninit(); even.len()];
            codec.decode(&packed, &mut slots).expect("host decodes q4_0");
            // decode returned Ok, which initialises every slot of its output
            (packed, slots.iter().map(|slot| unsafe { slot.assume_init() }).collect::<Vec<f32>>())
        };
        assert_eq!(wide(&restored.k_even), decoded_even, "the key rows read back as the Q4_0 codec decodes them");
        assert_eq!(wide(&restored.v), value, "the value rows are exact binary16 values");
    }

    #[test]
    fn a_scalar_key_beside_a_q4_0_value_is_refused_at_adoption_with_the_cache_types_named() {
        let mut caches = alloc::vec![LayerCacheState::Attention(host_cache(None, 10))];
        let widths = [LayerPadRowWidths::Attention {
            even_odd_row: EVEN_ODD_ROW,
            v_row: V_ROW,
        }];
        let error = DeviceKv::adopt_pair(
            &mut caches,
            &widths,
            10,
            (40, 4),
            (3, 3, BARE_MINIMUM_GRANULE_ROWS),
            allocate_placed_buffer,
            (NumericFormat::Float16, NumericFormat::Q4_0),
        )
        .err()
        .expect("a scalar key beside a block value has no read path");
        assert!(error.to_string().contains("device kv cache types"));
    }

    #[test]
    fn a_half_width_cache_round_trips_binary16_exact_rows_in_half_the_bytes() {
        let (device, mut caches) = adopted_half(37, 80);
        let layer = device.layers[0].as_ref().expect("layer 0 is resident");
        assert_eq!(layer.even_odd_row_bytes, EVEN_ODD_ROW * 2, "binary16 rows");
        assert_eq!(layer.v_row_bytes, V_ROW * 2, "binary16 rows");
        assert_eq!(device.cache_codec().expect("f16 is readable"), Some(Codec::Float16));

        device.flush(&mut caches, 37, 80).expect("device kv flushes to host");

        let LayerCacheState::Attention(restored) = &caches[0] else {
            panic!("layer 0 stays an attention cache");
        };
        assert_eq!(wide(&restored.k_even), half_exact_rows(0, 37, EVEN_ODD_ROW, 0));
        assert_eq!(wide(&restored.k_odd), half_exact_rows(0, 37, EVEN_ODD_ROW, 1));
        assert_eq!(wide(&restored.v), half_exact_rows(0, 37, V_ROW, 2));
    }

    #[test]
    fn a_half_cache_is_written_by_the_graph_and_encodes_nothing() {
        let cached_len = 5;
        let (device, _caches) = adopted_half(cached_len, 80);
        let layer = device.layers[0].as_ref().expect("layer 0 is resident");

        assert!(
            !layer.key_encoded && !layer.value_encoded,
            "binary16 is a layout the graph writes at the cache width, so the plan encodes no plane"
        );
    }

    #[test]
    fn a_half_width_step_reads_the_cache_and_writes_its_rows_into_it() {
        let cached_len = 5;
        let (mut device, _caches) = adopted_half(cached_len, 80);
        let leaves = [Some(KvLeafNodes {
            k_even: NodeId(1),
            k_odd: NodeId(2),
            v: NodeId(3),
        })];
        let roots = [LayerCacheRoots::Attention((NodeId(10), NodeId(11), NodeId(12)))];

        device.ready_step(cached_len, 1, 0).expect("a linear layer never moves rows");
        let placements = device.placements(cached_len, &leaves, &roots);

        let layer = device.layers[0].as_ref().expect("layer 0 is resident");
        assert!(core::ptr::eq(placements.inputs[0].1, &layer.k_even));
        assert_eq!(placements.inputs[0].2, 0, "a full layer reads from row 0");
        assert!(core::ptr::eq(placements.outputs[0].1, &layer.k_even));
        assert_eq!(
            placements.outputs[0].2,
            cached_len * layer.even_odd_row_bytes,
            "the new half row lands at the end of the cached rows"
        );
        assert!(placements.encoded.is_empty(), "the graph writes binary16 itself, so nothing is encoded");
    }

    #[test]
    fn an_f32_step_places_its_rows_straight_in_the_cache_and_commits_nothing() {
        let cached_len = 5;
        let (mut device, _caches) = adopted(None, cached_len, 80);
        let leaves = [Some(KvLeafNodes {
            k_even: NodeId(1),
            k_odd: NodeId(2),
            v: NodeId(3),
        })];
        let roots = [LayerCacheRoots::Attention((
            NodeId(10),
            NodeId(11),
            NodeId(12),
        ))];

        device.ready_step(cached_len, 1, 0).expect("a linear layer never moves rows");
        let placements = device.placements(cached_len, &leaves, &roots);

        let layer = device.layers[0].as_ref().expect("layer 0 is resident");
        assert!(!layer.key_encoded && !layer.value_encoded, "an f32 cache needs no encode");
        assert!(placements.encoded.is_empty());
        assert!(core::ptr::eq(placements.outputs[0].1, &layer.k_even));
        assert_eq!(
            placements.outputs[0].2,
            cached_len * layer.even_odd_row_bytes,
            "the new row lands at the end of the cached rows"
        );
        assert_eq!(device.cache_codec().expect("f32 is readable"), None);
    }

    fn adopted_bf16_with_step_rows(
        positions: usize,
        step_rows: usize,
    ) -> (DeviceKv, Vec<usize>) {
        let mut caches = alloc::vec![LayerCacheState::Attention(bf16_exact_cache(positions))];
        let widths = [LayerPadRowWidths::Attention {
            even_odd_row: EVEN_ODD_ROW,
            v_row: V_ROW,
        }];
        let mut acquired_bytes = Vec::new();
        let device = DeviceKv::adopt_pair(
            &mut caches,
            &widths,
            positions,
            (positions + 1, 4),
            (3, step_rows, BARE_MINIMUM_GRANULE_ROWS),
            |bytes| {
                acquired_bytes.push(bytes);
                allocate_placed_buffer(bytes)
            },
            (NumericFormat::BFloat16, NumericFormat::BFloat16),
        )
        .expect("device kv allocates on the real Metal device")
        .expect("a plain attention cache is adoptable");
        (device, acquired_bytes)
    }

    #[test]
    fn a_narrow_cache_allocates_only_its_three_cache_planes_and_no_binary32_staging() {
        let (device, acquired_bytes) = adopted_bf16_with_step_rows(37, 1);

        assert_eq!(device.max_step_rows(), 1, "a later step wider than the bound is refused");
        assert_eq!(
            acquired_bytes.len(),
            3,
            "the three cache planes come from the source and nothing else is allocated for the layer"
        );
    }

    #[test]
    fn a_q8_0_key_beside_a_binary32_value_is_refused_at_adoption() {
        let mut caches = alloc::vec![LayerCacheState::Attention(host_cache(None, 10))];
        let widths = [LayerPadRowWidths::Attention {
            even_odd_row: EVEN_ODD_ROW,
            v_row: V_ROW,
        }];

        let error = DeviceKv::adopt_pair(
            &mut caches,
            &widths,
            10,
            (40, 4),
            (3, 3, BARE_MINIMUM_GRANULE_ROWS),
            allocate_placed_buffer,
            (NumericFormat::Q8_0, NumericFormat::Float32),
        )
        .err()
        .expect("a block key beside a plain value is a mixed pair");

        assert!(error.to_string().contains("device kv cache types"));
    }

    #[test]
    fn a_block_format_is_a_readable_cache_exactly_when_its_description_is_a_readable_block() {
        let mut resolved = 0usize;
        let mut readable = 0usize;
        for format in NumericFormat::ALL
            .iter()
            .filter(|format| !matches!(format, NumericFormat::Scalar(_)))
        {
            if FormatCodec::for_format(format).is_none() {
                continue;
            }
            resolved += 1;

            let admitted = kernel_codecs(format, format).is_ok();

            assert_eq!(
                admitted,
                omega::msl::layout_render::readable_block_layout(format).is_some(),
                "{format:?}: admission must follow the description, not a list"
            );
            readable += usize::from(admitted);
        }

        assert_eq!(resolved, 14);
        assert!(readable >= 4, "Q4_0, Q8_0, MXFP4 and NVFP4 are readable block layouts, got {readable}");
    }

    #[test]
    fn leaf_nodes_resolve_by_declared_name_in_one_pass() {
        use proxima_numeric::DType;
        let input = |name: &str| Op::Input {
            dtype: DType::FLOAT32,
            shape: Vec::new(),
            name: Some(String::from(name)),
        };
        let program = [
            input("ids"),
            input("kv_cache.0.k_even"),
            input("kv_cache.0.k_odd"),
            input("kv_cache.0.v"),
        ];
        let names = [
            LayerCacheNames::Attention {
                k_even: String::from("kv_cache.0.k_even"),
                k_odd: String::from("kv_cache.0.k_odd"),
                v: String::from("kv_cache.0.v"),
            },
            LayerCacheNames::SharedFromLayer,
        ];

        let leaves = kv_leaf_nodes(&program, &names);

        let first = leaves[0].expect("layer 0 declares all three leaves");
        assert_eq!(
            (first.k_even, first.k_odd, first.v),
            (NodeId(1), NodeId(2), NodeId(3))
        );
        assert!(leaves[1].is_none(), "a shared-KV layer owns no leaf");
    }

    /// New K rows as an attention layer produces them: mixed magnitudes, both signs, exact zeros,
    /// and values below binary16's normal range.
    fn kv_like_rows(count: usize, width: usize) -> Vec<f32> {
        let magnitudes = [0.0f32, 3.1e-6, 0.0173, 0.4999, 1.0 / 3.0, 1.7, 12.37, 255.9, 1031.3];
        (0..count * width)
            .map(|index| {
                let magnitude = magnitudes[index % magnitudes.len()] * (1.0 + index as f32 * 0.0137);
                if index % 3 == 1 { -magnitude } else { magnitude }
            })
            .collect()
    }

    #[test]
    fn store_rows_as_half_writes_binary16_bits_at_the_output_row_of_a_half_cache() {
        const NEW_ROWS: usize = 3;
        const HEADS: usize = 2;
        const HEAD_DIM: usize = 4;
        const ROW_ELEMENTS: usize = HEADS * HEAD_DIM;
        const CACHED_ROWS: usize = 5;
        let half = omega::device_row_store_dtype(&NumericFormat::Float16).expect("binary16 is a layout the graph writes");
        let mut program = Vec::new();
        let rows = proxima_tensor::spec::input_leaf(
            &mut program,
            DType::FLOAT32,
            alloc::vec![
                Extent::Static(NEW_ROWS as u32),
                Extent::Static(HEADS as u32),
                Extent::Static(HEAD_DIM as u32),
            ],
            "rows",
        );
        let produced = proxima_tensor::spec::elementwise(&mut program, DType::FLOAT32, ScalarOp::Negate, &[(rows, "sui->sui")]).expect("a rank-3 f32 producer");
        let stored = store_rows_as(&mut program, produced, half).expect("a rank-3 root takes the cast");
        let input = kv_like_rows(NEW_ROWS, ROW_ELEMENTS);
        let plan = omega::plan(&program, &[], &[QuantizedBlock::Float32(&input)], &[stored], NumericPolicy::default())
            .expect("the cast program plans");
        let row_bytes = ROW_ELEMENTS * 2;
        let cache = allocate_placed_buffer((CACHED_ROWS + NEW_ROWS) * row_bytes).expect("the half cache allocates");
        omega::zero_placed_buffer(&cache, (CACHED_ROWS + NEW_ROWS) * row_bytes).expect("the cache starts zeroed");

        omega::execute_plan_with_placements(
            &plan,
            &[QuantizedBlock::Float32(&input)],
            &[],
            &[(stored, &cache, CACHED_ROWS * row_bytes)],
            &mut Vec::new(),
        )
        .expect("the cast places its rows in the cache");

        let codec = FormatCodec::for_format(&NumericFormat::Float16).expect("binary16 resolves a codec");
        let written = omega::read_placed_buffer(&cache, CACHED_ROWS * row_bytes, &codec, NEW_ROWS * ROW_ELEMENTS)
            .expect("the written rows decode");
        let expected: Vec<u32> = input.iter().map(|value| half::f16::from_f32(-*value).to_f32().to_bits()).collect();
        let actual: Vec<u32> = written.iter().map(|value| value.to_bits()).collect();
        assert_eq!(actual, expected, "the stored rows are binary16(f32) bit for bit");
        let before = omega::read_placed_buffer(&cache, 0, &codec, CACHED_ROWS * ROW_ELEMENTS).expect("the prefix decodes");
        assert!(before.iter().all(|value| *value == 0.0), "rows before the output row are untouched");
        assert_ne!(
            expected,
            input.iter().map(|value| value.to_bits()).collect::<Vec<u32>>(),
            "the inputs are not binary16 exact, so equality above proves the rounding ran"
        );
    }

    const SLIDING_WINDOW: usize = 8;
    const SEEDED_LEN: usize = 20;
    const GROWN_LEN: usize = 45;
    const ATTENTION_LAYERS: usize = 2;
    const GROWTH_GRANULE_ROWS: usize = 64;

    fn write_position_rows(layer: &DeviceKvLayer, position: usize) {
        let row = layer.output_row(position);
        for (buffer, row_bytes, width, leaf) in [
            (&layer.k_even, layer.even_odd_row_bytes, EVEN_ODD_ROW, 0),
            (&layer.k_odd, layer.even_odd_row_bytes, EVEN_ODD_ROW, 1),
            (&layer.v, layer.v_row_bytes, V_ROW, 2),
        ] {
            omega::try_write_placed_buffer_f32(buffer, row * row_bytes, &row_values(position, width, leaf))
                .expect("the output row lies inside the plane");
        }
    }

    /// A sliding layer and a full layer seeded at 20 positions, then decoded to 45 with a 64-row
    /// compaction granule: the sliding buffer never compacts, so its first row stays at position 10.
    pub(in crate::generate) fn grown_slot(ids_len: usize) -> ResidentSlot {
        let mut caches = alloc::vec![
            LayerCacheState::Attention(host_cache(Some(SLIDING_WINDOW), SEEDED_LEN)),
            LayerCacheState::Attention(host_cache(None, SEEDED_LEN)),
            LayerCacheState::SharedFromLayer,
        ];
        let attention = || LayerPadRowWidths::Attention { even_odd_row: EVEN_ODD_ROW, v_row: V_ROW };
        let widths = [attention(), attention(), LayerPadRowWidths::SharedFromLayer];
        let mut device = DeviceKv::adopt_pair(
            &mut caches,
            &widths,
            SEEDED_LEN,
            (60, 4),
            (3, 3, GROWTH_GRANULE_ROWS),
            allocate_placed_buffer,
            (NumericFormat::Float32, NumericFormat::Float32),
        )
        .expect("device kv allocates on the real Metal device")
        .expect("plain attention caches are adoptable");
        for position in SEEDED_LEN..GROWN_LEN {
            for layer in device.layers.iter_mut().flatten() {
                layer.make_room(position, 1, SLIDING_WINDOW).expect("the kept rows lie inside the buffers");
                write_position_rows(layer, position);
            }
        }
        let sliding = device.layers[0].as_ref().expect("layer 0 is resident");
        assert_eq!(sliding.base_position, 10, "the sliding buffer still starts at the position it was seeded from");
        let key = CacheKey::of(&ServingConfig::default(), false, RopeScaling::None, 0, 0);
        device.into_slot((0..ids_len as u32).collect(), key, Vec::new())
    }

    const UNSEEDED_GRANULE_ROWS: usize = 32;
    const UNSEEDED_RETAIN: usize = SLIDING_WINDOW;

    /// A sliding layer (no slack, so `retain == window`) and a full layer that start empty and are
    /// written by decode steps only, never seeded from host rows, until `stop` says the slot is
    /// ready. Returns the slot and the length it holds.
    fn decoded_slot(stop: impl Fn(&DeviceKvLayer, usize) -> bool) -> (ResidentSlot, usize) {
        let mut caches = alloc::vec![
            LayerCacheState::Attention(test_attention_cache(Some(SLIDING_WINDOW), 0, EVEN_ODD_ROW, V_ROW, 0, 0)),
            LayerCacheState::Attention(test_attention_cache(None, 0, EVEN_ODD_ROW, V_ROW, 0, 0)),
        ];
        let attention = || LayerPadRowWidths::Attention { even_odd_row: EVEN_ODD_ROW, v_row: V_ROW };
        let mut device = DeviceKv::adopt_pair(
            &mut caches,
            &[attention(), attention()],
            0,
            (100, 4),
            (3, 3, UNSEEDED_GRANULE_ROWS),
            allocate_placed_buffer,
            (NumericFormat::Float32, NumericFormat::Float32),
        )
        .expect("device kv allocates on the real Metal device")
        .expect("empty attention caches are adoptable");
        let mut len = 0;
        loop {
            for layer in device.layers.iter_mut().flatten() {
                layer.make_room(len, 1, SLIDING_WINDOW).expect("the kept rows lie inside the buffers");
                write_position_rows(layer, len);
            }
            len += 1;
            let sliding = device.layers[0].as_ref().expect("layer 0 is resident");
            if stop(sliding, len) {
                break;
            }
        }
        let key = CacheKey::of(&ServingConfig::default(), false, RopeScaling::None, 0, 0);
        (device.into_slot((0..len as u32).collect(), key, Vec::new()), len)
    }

    /// Reads back every row a rewind to `keep` keeps in each layer and returns how many it compared.
    fn kept_rows_match_what_was_written(device: &DeviceKv, keep: usize) -> usize {
        let mut compared = 0;
        for (index, slot) in device.layers.iter().enumerate() {
            let layer = slot.as_ref().expect("an attention layer is resident");
            for position in layer.first_kept(keep)..keep {
                let row = position - layer.base_position;
                let even = layer.load_f32(&layer.k_even, row * layer.even_odd_row_bytes, EVEN_ODD_ROW).expect("k_even row reads");
                let odd = layer.load_f32(&layer.k_odd, row * layer.even_odd_row_bytes, EVEN_ODD_ROW).expect("k_odd row reads");
                let value = layer.load_value_f32(&layer.v, row * layer.v_row_bytes, V_ROW).expect("v row reads");
                assert_eq!(even, row_values(position, EVEN_ODD_ROW, 0), "layer {index} position {position} k_even");
                assert_eq!(odd, row_values(position, EVEN_ODD_ROW, 1), "layer {index} position {position} k_odd");
                assert_eq!(value, row_values(position, V_ROW, 2), "layer {index} position {position} v");
                compared += 1;
            }
        }
        compared
    }

    #[test]
    fn an_unseeded_wrapped_sliding_layer_rewinds_in_place_up_to_capacity_minus_window_rows() {
        let (slot, len) = decoded_slot(|sliding, len| sliding.base_position > 0 && len - sliding.base_position == sliding.capacity_rows);
        let (capacity, base) = {
            let sliding = slot.device.layers[0].as_ref().expect("layer 0 is resident");
            (sliding.capacity_rows, sliding.base_position)
        };
        assert_eq!(capacity, 32, "11 rows of window plus step round up to one 32-row granule");
        assert_eq!(len - base, capacity, "the ring wrapped and the buffer holds exactly its capacity, the oldest row is len - capacity");
        let furthest = capacity - SLIDING_WINDOW;
        assert_eq!(furthest, 24, "capacity minus window is the slack the rewind may use");

        let mut compared_total = 0;
        for back in [1, 12, furthest] {
            let keep = len - back;
            let slot = decoded_slot(|_, next| next == len).0;
            let (device, outcome) = DeviceKv::adopt_resident(slot, keep).expect("keep lies inside the slot");
            assert!(matches!(outcome, RewindOutcome::InPlace), "a rewind of {back} rows is inside the {furthest} the buffer holds: {outcome:?}");
            let compared = kept_rows_match_what_was_written(&device, keep);
            assert_eq!(compared, UNSEEDED_RETAIN + keep, "retain rows of the sliding layer plus every row of the full layer, compared for a rewind of {back}");
            compared_total += compared;
        }
        assert_eq!(compared_total, 3 * UNSEEDED_RETAIN + (len - 1) + (len - 12) + (len - furthest), "every kept row of three rewinds was compared");

        let keep = len - furthest - 1;
        let (_, outcome) = DeviceKv::adopt_resident(decoded_slot(|_, next| next == len).0, keep).expect("keep lies inside the slot");
        assert!(
            matches!(outcome, RewindOutcome::Refused(MissReason::RingSlackExceeded { .. })),
            "one row past the slack needs a row the buffer no longer holds, and no checkpoint covers it: {outcome:?}"
        );
    }

    #[test]
    fn a_slot_that_never_wrapped_rewinds_in_place_to_every_keep() {
        let (slot, len) = decoded_slot(|sliding, len| len == 20 && sliding.base_position == 0);
        let capacity = slot.device.layers[0].as_ref().expect("layer 0 is resident").capacity_rows;
        assert!(len < capacity, "20 positions fit the 32-row buffer, so the ring never wrapped");
        let mut counted = 0;
        for keep in 1..=len {
            let slot = decoded_slot(|_, next| next == len).0;
            let (_, outcome) = DeviceKv::adopt_resident(slot, keep).expect("keep lies inside the slot");
            assert!(matches!(outcome, RewindOutcome::InPlace), "keep {keep} of {len}: row 0 is held, so every rewind is in place: {outcome:?}");
            counted += 1;
        }
        assert_eq!(counted, len, "every keep from 1 to len was asked");
    }

    fn checkpoint_at(position: usize) -> RingCheckpoint {
        let state = PrefixState {
            ids: (0..position as u32).collect(),
            layer_caches: alloc::vec![
                LayerCacheState::Attention(host_cache(Some(SLIDING_WINDOW), position)),
                LayerCacheState::Attention(host_cache(None, position)),
            ],
            cached_len: position,
        };
        RingCheckpoint::capture(&state).expect("a ring layer is captured")
    }

    #[test]
    fn adopt_resident_below_the_slot_length_leaves_the_host_rows_up_to_keep_and_no_row_uninitialised() {
        let keep = 40;
        let slot = grown_slot(GROWN_LEN);
        assert_eq!(slot.key, CacheKey::of(&ServingConfig::default(), false, RopeScaling::None, 0, 0), "into_slot keeps the identity the rows were built under");
        assert_eq!(slot.ids.len(), GROWN_LEN, "and the positions they cover");
        let (device, outcome) = DeviceKv::adopt_resident(slot, keep).expect("keep lies inside the slot");
        assert!(matches!(outcome, RewindOutcome::InPlace), "both layers still hold every row a rewind to 40 keeps: {outcome:?}");

        let references = [host_cache(Some(SLIDING_WINDOW), keep), host_cache(None, keep)];
        let (mut compared, mut uninitialised, mut zero_rows, mut stale_rows) = (0, 0, 0, 0);
        for (index, reference) in references.iter().enumerate() {
            let layer = device.layers[index].as_ref().expect("an attention layer is resident");
            for position in layer.first_kept(keep)..keep {
                let row = position - layer.base_position;
                let slot = reference.ring_geometry().map_or(position, |ring| position % ring.capacity);
                let even = layer.load_f32(&layer.k_even, row * layer.even_odd_row_bytes, EVEN_ODD_ROW).expect("k_even row reads");
                let odd = layer.load_f32(&layer.k_odd, row * layer.even_odd_row_bytes, EVEN_ODD_ROW).expect("k_odd row reads");
                let value = layer.load_value_f32(&layer.v, row * layer.v_row_bytes, V_ROW).expect("v row reads");
                assert_eq!(even, wide(&reference.k_even)[slot * EVEN_ODD_ROW..][..EVEN_ODD_ROW], "layer {index} position {position} k_even");
                assert_eq!(odd, wide(&reference.k_odd)[slot * EVEN_ODD_ROW..][..EVEN_ODD_ROW], "layer {index} position {position} k_odd");
                assert_eq!(value, wide(&reference.v)[slot * V_ROW..][..V_ROW], "layer {index} position {position} v");
            }
            compared += 1;
            for row in 0..layer.capacity_rows {
                let position = layer.base_position + row;
                let written = position < GROWN_LEN;
                for (buffer, width, leaf) in [(&layer.k_even, EVEN_ODD_ROW, 0), (&layer.k_odd, EVEN_ODD_ROW, 1), (&layer.v, V_ROW, 2)] {
                    let bytes = if leaf == 2 { layer.v_row_bytes } else { layer.even_odd_row_bytes };
                    let found = layer.load_f32(buffer, row * bytes, width).expect("a capacity row reads");
                    let is_written_row = found == row_values(position, width, leaf);
                    let is_zero_row = found.iter().all(|element| *element == 0.0);
                    uninitialised += usize::from(!(is_written_row && written) && !is_zero_row);
                    if leaf == 0 {
                        zero_rows += usize::from(is_zero_row && !written);
                        stale_rows += usize::from(is_written_row && position >= keep);
                    }
                }
            }
        }
        assert_eq!(compared, ATTENTION_LAYERS, "every attention layer of the fixture was compared");
        assert_eq!(device.layers.len(), ATTENTION_LAYERS + 1, "the fixture also holds one shared layer that owns no rows");
        assert_eq!(uninitialised, 0, "no row a step could read is neither a written row nor a finite zero");
        assert_eq!(stale_rows, 2 * (GROWN_LEN - keep), "the rows past keep stay as written, one run per layer");
        assert!(zero_rows > 0, "the buffers hold never-written rows, so the zero check had rows to find");
    }

    #[test]
    fn a_keep_past_the_sliding_retained_rows_is_never_in_place() {
        let (device, outcome) = DeviceKv::adopt_resident(grown_slot(GROWN_LEN), 14).expect("keep lies inside the slot");
        assert!(
            matches!(
                outcome,
                RewindOutcome::Refused(MissReason::RingSlackExceeded { rewind_rows, slack_rows: 2 }) if rewind_rows == GROWN_LEN - 14
            ),
            "the sliding layer starts at position 10 but keep 14 needs positions from 4, and no checkpoint is held: {outcome:?}"
        );
        assert_eq!(device.layers.len(), ATTENTION_LAYERS + 1, "the cache comes back whole");

        let mut slot = grown_slot(GROWN_LEN);
        slot.checkpoints = alloc::vec![checkpoint_at(6), checkpoint_at(12), checkpoint_at(30)];
        let (_, outcome) = DeviceKv::adopt_resident(slot, 14).expect("keep lies inside the slot");
        let RewindOutcome::Checkpoint(checkpoint) = outcome else {
            panic!("a checkpoint at 12 covers keep 14, got {outcome:?}");
        };
        assert_eq!(checkpoint.position(), 12, "the newest checkpoint at or before keep, not 6 and not 30");

        let mut slot = grown_slot(GROWN_LEN);
        slot.checkpoints = alloc::vec![checkpoint_at(20), checkpoint_at(30)];
        let (_, outcome) = DeviceKv::adopt_resident(slot, 14).expect("keep lies inside the slot");
        assert!(
            matches!(outcome, RewindOutcome::Refused(MissReason::RingSlackExceeded { .. })),
            "both checkpoints lie past keep 14, so none qualifies: {outcome:?}"
        );
    }

    #[test]
    fn adopt_resident_refuses_a_zero_keep_and_errors_on_a_keep_past_the_slot() {
        let (_, outcome) = DeviceKv::adopt_resident(grown_slot(GROWN_LEN), 0).expect("zero lies inside the slot");
        assert!(matches!(outcome, RewindOutcome::Refused(MissReason::NoCommonPrefix)), "{outcome:?}");
        assert!(DeviceKv::adopt_resident(grown_slot(GROWN_LEN), GROWN_LEN + 1).is_err(), "46 positions are not held");
    }
}
