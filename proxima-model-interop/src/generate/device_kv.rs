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
//! linear buffer of `2 * retain + step rows`: the leaf placement slides forward
//! by one row per token (the program reads `min(cached_len, window)` rows
//! starting `window` back), and when the read would run off the end the kept
//! rows are moved to the front. That trades the ring's modular indexing, which
//! the kernel does not do, for an amortised `retain`-row move once every
//! `retain + step` tokens.

use super::*;

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
}

struct DeviceKvLayer {
    k_even: PlacedBuffer,
    k_odd: PlacedBuffer,
    v: PlacedBuffer,
    /// `Some` for a half-width layer: the graph writes this step's rows as f32
    /// here (the cache buffers are binary16, so the program cannot place its
    /// f32 outputs in them), and [`DeviceKvLayer::commit`] rounds them into
    /// the cache once the step's command buffers have completed.
    staging: Option<[PlacedBuffer; 3]>,
    element: GgmlType,
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

pub(super) struct DeviceKv {
    layers: Vec<Option<DeviceKvLayer>>,
    max_step_rows: usize,
    element: GgmlType,
}

impl DeviceKvLayer {
    fn allocate(
        cache: &LayerCache,
        even_odd_row: usize,
        v_row: usize,
        full_capacity_rows: usize,
        max_step_rows: usize,
        source: KvBufferSource,
        element: GgmlType,
    ) -> Result<Self, InteropError> {
        let element_bytes = element_bytes(element)?;
        let window = cache.ring_geometry().map(|ring| ring.window);
        let retain = cache.ring_geometry().map_or(0, |ring| ring.capacity);
        let capacity_rows = if window.is_some() {
            (2 * retain + max_step_rows).min(full_capacity_rows)
        } else {
            full_capacity_rows
        };
        let even_odd_row_bytes = even_odd_row * element_bytes;
        let v_row_bytes = v_row * element_bytes;
        let k_even = source(capacity_rows * even_odd_row_bytes)?;
        let k_odd = source(capacity_rows * even_odd_row_bytes)?;
        let v = source(capacity_rows * v_row_bytes)?;
        let staging = if element == GgmlType::F32 {
            None
        } else {
            let f32_bytes = core::mem::size_of::<f32>();
            Some([
                allocate_placed_buffer(max_step_rows * even_odd_row * f32_bytes)?,
                allocate_placed_buffer(max_step_rows * even_odd_row * f32_bytes)?,
                allocate_placed_buffer(max_step_rows * v_row * f32_bytes)?,
            ])
        };
        Ok(Self {
            k_even,
            k_odd,
            v,
            staging,
            element,
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
    fn seed(&mut self, cache: &LayerCache, cached_len: usize) {
        let even_odd_row = self.even_odd_row;
        let v_row = self.v_row;
        let first = self.first_kept(cached_len);
        let kept = cached_len - first;
        self.base_position = first;
        for (slot_start, rows, target_row) in kept_segments(cache, first, kept)
            .into_iter()
            .filter(|(_, rows, _)| *rows > 0)
        {
            self.write_rows(cache, slot_start, rows, target_row, even_odd_row, v_row);
        }
        for (buffer, row_bytes) in [
            (&self.k_even, self.even_odd_row_bytes),
            (&self.k_odd, self.even_odd_row_bytes),
            (&self.v, self.v_row_bytes),
        ] {
            omega::metal::zero_placed_buffer_range(
                buffer,
                kept * row_bytes,
                (self.capacity_rows - kept) * row_bytes,
            );
        }
    }

    fn write_rows(
        &self,
        cache: &LayerCache,
        slot_start: usize,
        rows: usize,
        target_row: usize,
        even_odd_row: usize,
        v_row: usize,
    ) {
        let even_odd = slot_start * even_odd_row..(slot_start + rows) * even_odd_row;
        let value = slot_start * v_row..(slot_start + rows) * v_row;
        self.store_f32(
            &self.k_even,
            target_row * self.even_odd_row_bytes,
            &cache.k_even[even_odd.clone()],
        );
        self.store_f32(
            &self.k_odd,
            target_row * self.even_odd_row_bytes,
            &cache.k_odd[even_odd],
        );
        self.store_f32(&self.v, target_row * self.v_row_bytes, &cache.v[value]);
    }

    fn make_room(&mut self, cached_len: usize, new_count: usize, bound_extent: usize) {
        let Some(window) = self.window else {
            return;
        };
        let live_first_row = cached_len - cached_len.min(window) - self.base_position;
        let reach =
            (live_first_row + bound_extent).max(cached_len + new_count - self.base_position);
        if reach <= self.capacity_rows {
            return;
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
            );
        }
        self.base_position = kept_first;
    }

    fn input_row(&self, cached_len: usize) -> usize {
        self.window.map_or(0, |window| {
            cached_len - cached_len.min(window) - self.base_position
        })
    }

    fn output_row(&self, cached_len: usize) -> usize {
        cached_len - self.base_position
    }

    fn store_f32(&self, buffer: &PlacedBuffer, byte_offset: usize, values: &[f32]) {
        match self.element {
            GgmlType::F16 => omega::write_placed_buffer_f32_as_f16(buffer, byte_offset, values),
            _ => omega::write_placed_buffer_f32(buffer, byte_offset, values),
        }
    }

    fn load_f32(&self, buffer: &PlacedBuffer, byte_offset: usize, count: usize) -> Vec<f32> {
        match self.element {
            GgmlType::F16 => omega::read_placed_buffer_f16_as_f32(buffer, byte_offset, count),
            _ => omega::read_placed_buffer_f32(buffer, byte_offset, count),
        }
    }

    /// Rounds the `new_count` rows a finished step left in the f32 staging
    /// buffers into the cache at the rows [`Self::output_row`] names. A layer
    /// whose cache is f32 has its outputs placed in the cache itself, so there
    /// is nothing to do. Must run after the step's command buffers completed
    /// (`evaluate_with_placements` waits) and before the next step reads the
    /// cache.
    fn commit(&self, cached_len: usize, new_count: usize) {
        let Some([staging_even, staging_odd, staging_value]) = &self.staging else {
            return;
        };
        let output_row = self.output_row(cached_len);
        for (staging, cache, row_bytes, row) in [
            (staging_even, &self.k_even, self.even_odd_row_bytes, self.even_odd_row),
            (staging_odd, &self.k_odd, self.even_odd_row_bytes, self.even_odd_row),
            (staging_value, &self.v, self.v_row_bytes, self.v_row),
        ] {
            omega::narrow_placed_buffer_f32_to_f16(
                staging,
                0,
                cache,
                output_row * row_bytes,
                new_count * row,
            );
        }
    }
}

/// Bytes per stored element of a device cache of type `element`: a plain f32
/// buffer, or binary16 that the cached-attention decode kernel reads as
/// `Codec::Float16`.
fn element_bytes(element: GgmlType) -> Result<usize, InteropError> {
    match element {
        GgmlType::F32 => Ok(core::mem::size_of::<f32>()),
        GgmlType::F16 => Ok(core::mem::size_of::<u16>()),
        other => Err(InteropError::UnsupportedServingConfig(alloc::format!(
            "device kv cache type {other:?}: the device-resident cache stores f32 or f16"
        ))),
    }
}

impl DeviceKv {
    /// Moves every attention layer's host rows onto the device, or `None`
    /// when this call's caches are not the plain attention shape
    /// (`SharedFromLayer` layers are fine: they own nothing) or a host cache
    /// does not hold exactly `cached_len` rows. Host copies are released here
    /// and rebuilt by [`Self::flush`], so a long context is held once.
    ///
    /// `capacity_positions` is the most positions any step can reach: the call's
    /// `positions_needed` plus the rows of a speculative draft. `max_step_rows` is
    /// the widest step of any kind, a prefill batch included, and bounds every
    /// step and sizes a sliding layer's linear buffer. A prefill batch ends inside
    /// `positions_needed`, so it adds no full-layer rows.
    // clippy::too_many_arguments: the caches and their widths, the position
    // geometry, and the buffer hook plus the element type it allocates for
    #[allow(clippy::too_many_arguments)]
    pub(super) fn adopt(
        layer_caches: &mut [LayerCacheState],
        layer_row_widths: &[LayerPadRowWidths],
        cached_len: usize,
        capacity_positions: usize,
        bucket_tokens: usize,
        max_step_rows: usize,
        source: KvBufferSource,
        element: GgmlType,
    ) -> Result<Option<Self>, InteropError> {
        let full_capacity_rows =
            kv_extent(capacity_positions, usize::MAX, bucket_tokens) + bucket_tokens;
        let mut layers: Vec<Option<DeviceKvLayer>> = Vec::with_capacity(layer_caches.len());
        for (state, widths) in layer_caches.iter().zip(layer_row_widths) {
            match (state, widths) {
                (
                    LayerCacheState::Attention(cache),
                    LayerPadRowWidths::Attention {
                        even_odd_row,
                        v_row,
                    },
                ) => {
                    let host_rows = cache.k_even.len() / (*even_odd_row).max(1);
                    let holds_every_position =
                        cache.ring_geometry().is_some() || host_rows == cached_len;
                    if !holds_every_position {
                        return Ok(None);
                    }
                    let mut device_layer = DeviceKvLayer::allocate(
                        cache,
                        *even_odd_row,
                        *v_row,
                        full_capacity_rows,
                        max_step_rows,
                        source,
                        element,
                    )?;
                    device_layer.seed(cache, cached_len);
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
                cache.k_even = Vec::new();
                cache.k_odd = Vec::new();
                cache.v = Vec::new();
            }
        }
        Ok(Some(Self {
            layers,
            max_step_rows,
            element,
        }))
    }

    pub(super) fn max_step_rows(&self) -> usize {
        self.max_step_rows
    }

    /// The packed codec the leaf placements read this cache as: `Some(Float16)`
    /// for a binary16 cache, `None` for f32 (a plain buffer needs none).
    pub(super) fn cache_codec(&self) -> Option<Codec> {
        (self.element == GgmlType::F16).then_some(Codec::Float16)
    }

    /// Moves what a finished step wrote into the cache, for every layer whose
    /// outputs landed in f32 staging ([`DeviceKvLayer::commit`]). `cached_len`
    /// and `new_count` are the values [`Self::placements`] ran with.
    pub(super) fn commit_step(&self, cached_len: usize, new_count: usize) {
        for layer in self.layers.iter().flatten() {
            layer.commit(cached_len, new_count);
        }
    }

    /// One flag per layer, for the host paths that must skip a resident one.
    pub(super) fn resident_layers(&self) -> Vec<bool> {
        self.layers.iter().map(Option::is_some).collect()
    }

    /// Readies every sliding layer for a step at `cached_len` positions.
    /// `sliding_bound` is the program's sliding extent this step
    /// (`KvRing::bound_extent`). Separate from [`Self::placements`] so the
    /// placements borrow the cache shared, and [`Self::commit_step`] can run
    /// while the step's placement lists are still alive.
    pub(super) fn ready_step(&mut self, cached_len: usize, new_count: usize, sliding_bound: usize) {
        for layer in self.layers.iter_mut().flatten() {
            layer.make_room(cached_len, new_count, sliding_bound);
        }
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
        for (index, slot) in self.layers.iter().enumerate() {
            let (Some(layer), Some(Some(leaf)), Some(LayerCacheRoots::Attention((even, odd, value)))) =
                (slot, leaves.get(index), layer_roots.get(index))
            else {
                continue;
            };
            let input_row = layer.input_row(cached_len);
            let output_row = layer.output_row(cached_len);
            inputs.push((leaf.k_even, &layer.k_even, input_row * layer.even_odd_row_bytes));
            inputs.push((leaf.k_odd, &layer.k_odd, input_row * layer.even_odd_row_bytes));
            inputs.push((leaf.v, &layer.v, input_row * layer.v_row_bytes));
            match &layer.staging {
                Some([staging_even, staging_odd, staging_value]) => {
                    outputs.push((*even, staging_even, 0));
                    outputs.push((*odd, staging_odd, 0));
                    outputs.push((*value, staging_value, 0));
                }
                None => {
                    outputs.push((*even, &layer.k_even, output_row * layer.even_odd_row_bytes));
                    outputs.push((*odd, &layer.k_odd, output_row * layer.even_odd_row_bytes));
                    outputs.push((*value, &layer.v, output_row * layer.v_row_bytes));
                }
            }
        }
        KvPlacements { inputs, outputs }
    }

    /// Hands the rows back to the host caches so [`PrefixState`] and the
    /// prompt cache see exactly what a host-only run would have left.
    pub(super) fn flush(
        &self,
        layer_caches: &mut [LayerCacheState],
        cached_len: usize,
        positions_needed: usize,
    ) {
        for (slot, state) in self.layers.iter().zip(layer_caches.iter_mut()) {
            let (Some(layer), LayerCacheState::Attention(cache)) = (slot, state) else {
                continue;
            };
            let first = layer.first_kept(cached_len);
            let live = cached_len - first;
            let first_row = first - layer.base_position;
            let even_odd_row = layer.even_odd_row;
            let v_row = layer.v_row;
            let even = layer.load_f32(
                &layer.k_even,
                first_row * layer.even_odd_row_bytes,
                live * even_odd_row,
            );
            let odd = layer.load_f32(
                &layer.k_odd,
                first_row * layer.even_odd_row_bytes,
                live * even_odd_row,
            );
            let value = layer.load_f32(
                &layer.v,
                first_row * layer.v_row_bytes,
                live * v_row,
            );
            cache.k_even.clear();
            cache.k_odd.clear();
            cache.v.clear();
            cache.reserve_ring_rows(positions_needed);
            cache.append_at(first, &even, &odd, &value);
        }
        #[cfg(feature = "instrument")]
        debug!(
            host_mirror_bytes = host_mirror_bytes(layer_caches) as u64,
            cached_len = cached_len as u64,
            "kv_host_mirror"
        );
    }
}

/// Host bytes the attention layers' K/V row vectors hold by allocated
/// capacity: the copy [`DeviceKv::flush`] leaves in the host caches.
#[cfg(feature = "instrument")]
fn host_mirror_bytes(layer_caches: &[LayerCacheState]) -> usize {
    layer_caches
        .iter()
        .filter_map(|state| match state {
            LayerCacheState::Attention(cache) => Some(
                (cache.k_even.capacity() + cache.k_odd.capacity() + cache.v.capacity())
                    * size_of::<f32>(),
            ),
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
mod tests {
    use super::*;

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
        let mut cache = attention_cache(window, 2, EVEN_ODD_ROW, V_ROW, 0, positions);
        cache.append_at(
            0,
            &rows(0, positions, EVEN_ODD_ROW, 0),
            &rows(0, positions, EVEN_ODD_ROW, 1),
            &rows(0, positions, V_ROW, 2),
        );
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
            total_positions,
            4,
            3,
            allocate_placed_buffer,
            GgmlType::F32,
        )
            .expect("device kv allocates on the real Metal device")
            .expect("a plain attention cache is adoptable");
        (device, caches)
    }

    fn write_step_rows(device: &DeviceKv, cached_len: usize) {
        let layer = device.layers[0].as_ref().expect("layer 0 is resident");
        let row = layer.output_row(cached_len);
        omega::write_placed_buffer_f32(
            &layer.k_even,
            row * layer.even_odd_row_bytes,
            &row_values(cached_len, EVEN_ODD_ROW, 0),
        );
        omega::write_placed_buffer_f32(
            &layer.k_odd,
            row * layer.even_odd_row_bytes,
            &row_values(cached_len, EVEN_ODD_ROW, 1),
        );
        omega::write_placed_buffer_f32(
            &layer.v,
            row * layer.v_row_bytes,
            &row_values(cached_len, V_ROW, 2),
        );
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

        device.flush(&mut caches, 37, 80);

        let LayerCacheState::Attention(restored) = &caches[0] else {
            panic!("layer 0 stays an attention cache");
        };
        assert_eq!(restored.k_even, rows(0, 37, EVEN_ODD_ROW, 0));
        assert_eq!(restored.k_odd, rows(0, 37, EVEN_ODD_ROW, 1));
        assert_eq!(restored.v, rows(0, 37, V_ROW, 2));
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
                .make_room(cached_len, 1, window);
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
        device.flush(&mut caches, final_len, start + steps);
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
                restored.k_even[slot * EVEN_ODD_ROW..(slot + 1) * EVEN_ODD_ROW],
                row_values(position, EVEN_ODD_ROW, 0)[..],
                "position {position} survives in its ring slot"
            );
            assert_eq!(
                restored.v[slot * V_ROW..(slot + 1) * V_ROW],
                row_values(position, V_ROW, 2)[..]
            );
        }
    }

    #[test]
    fn a_ring_round_trips_through_the_device_with_its_rewind_slack_rows() {
        let positions = 21;
        let before = host_cache(Some(8), positions);
        let ring = before.ring_geometry().copied().expect("sliding layer has a ring");
        assert_eq!(ring.capacity, 10, "8 window rows plus 2 slack rows");
        let (device, mut caches) = adopted(Some(8), positions, 60);

        device.flush(&mut caches, positions, 60);

        let LayerCacheState::Attention(restored) = &caches[0] else {
            panic!("layer 0 stays an attention cache");
        };
        for position in positions - ring.capacity..positions {
            let slot = position % ring.capacity;
            assert_eq!(
                restored.k_even[slot * EVEN_ODD_ROW..(slot + 1) * EVEN_ODD_ROW],
                before.k_even[slot * EVEN_ODD_ROW..(slot + 1) * EVEN_ODD_ROW],
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
            &cache,
            EVEN_ODD_ROW,
            V_ROW,
            capacity_rows,
            3,
            allocate_placed_buffer,
            GgmlType::F32,
        )
            .expect("device kv allocates on the real Metal device");
        let dirty = alloc::vec![f32::NAN; capacity_rows * EVEN_ODD_ROW.max(V_ROW)];
        omega::write_placed_buffer_f32(&layer.k_even, 0, &dirty[..capacity_rows * EVEN_ODD_ROW]);
        omega::write_placed_buffer_f32(&layer.k_odd, 0, &dirty[..capacity_rows * EVEN_ODD_ROW]);
        omega::write_placed_buffer_f32(&layer.v, 0, &dirty[..capacity_rows * V_ROW]);

        layer.seed(&cache, positions);

        let tail_rows = capacity_rows - positions;
        let tail = |buffer: &PlacedBuffer, row_bytes: usize, row: usize| {
            omega::read_placed_buffer_f32(buffer, positions * row_bytes, tail_rows * row)
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
            omega::read_placed_buffer_f32(&layer.k_even, 0, positions * EVEN_ODD_ROW),
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
            total_positions + 3,
            4,
            prefill_rows,
            allocate_placed_buffer,
            GgmlType::F32,
        )
            .expect("device kv allocates on the real Metal device")
            .expect("an empty attention cache is adoptable");
        (device, caches)
    }

    fn write_prefill_rows(device: &DeviceKv, rows_written: usize) {
        let layer = device.layers[0].as_ref().expect("layer 0 is resident");
        let row = layer.output_row(0);
        omega::write_placed_buffer_f32(
            &layer.k_even,
            row * layer.even_odd_row_bytes,
            &rows(0, rows_written, EVEN_ODD_ROW, 0),
        );
        omega::write_placed_buffer_f32(
            &layer.k_odd,
            row * layer.even_odd_row_bytes,
            &rows(0, rows_written, EVEN_ODD_ROW, 1),
        );
        omega::write_placed_buffer_f32(
            &layer.v,
            row * layer.v_row_bytes,
            &rows(0, rows_written, V_ROW, 2),
        );
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
        assert_eq!(layer.output_row(0), 0, "the prefill's rows start at the buffer's first row");

        write_prefill_rows(&device, prefill_rows);
        device.flush(&mut caches, prefill_rows, 40);

        let LayerCacheState::Attention(restored) = &caches[0] else {
            panic!("layer 0 stays an attention cache");
        };
        assert_eq!(restored.k_even, rows(0, prefill_rows, EVEN_ODD_ROW, 0));
        assert_eq!(restored.k_odd, rows(0, prefill_rows, EVEN_ODD_ROW, 1));
        assert_eq!(restored.v, rows(0, prefill_rows, V_ROW, 2));
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
        layer.make_room(0, prefill_rows, 8);
        assert_eq!(layer.base_position, 0, "a prefill from an empty cache never compacts");

        write_prefill_rows(&device, prefill_rows);
        device.flush(&mut caches, prefill_rows, 60);

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
                restored.k_even[slot * EVEN_ODD_ROW..(slot + 1) * EVEN_ODD_ROW],
                row_values(position, EVEN_ODD_ROW, 0)[..],
                "position {position} of the prefill survives in its ring slot"
            );
            assert_eq!(
                restored.v[slot * V_ROW..(slot + 1) * V_ROW],
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
            80,
            4,
            3,
            recording_source,
            GgmlType::F32,
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
            40,
            4,
            3,
            allocate_placed_buffer,
            GgmlType::F32,
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
            untouched.k_even.len(),
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
                (0..width).map(move |column| {
                    (position * 8 + column) as f32 * 0.25 + leaf as f32 * 0.5
                })
            })
            .collect()
    }

    fn half_exact_cache(positions: usize) -> LayerCache {
        let mut cache = attention_cache(None, 2, EVEN_ODD_ROW, V_ROW, 0, positions);
        cache.append_at(
            0,
            &half_exact_rows(0, positions, EVEN_ODD_ROW, 0),
            &half_exact_rows(0, positions, EVEN_ODD_ROW, 1),
            &half_exact_rows(0, positions, V_ROW, 2),
        );
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
            total_positions,
            4,
            3,
            allocate_placed_buffer,
            GgmlType::F16,
        )
        .expect("device kv allocates on the real Metal device")
        .expect("a plain attention cache is adoptable");
        (device, caches)
    }

    #[test]
    fn a_half_width_cache_round_trips_binary16_exact_rows_in_half_the_bytes() {
        let (device, mut caches) = adopted_half(37, 80);
        let layer = device.layers[0].as_ref().expect("layer 0 is resident");
        assert_eq!(layer.even_odd_row_bytes, EVEN_ODD_ROW * 2, "binary16 rows");
        assert_eq!(layer.v_row_bytes, V_ROW * 2, "binary16 rows");
        assert_eq!(device.cache_codec(), Some(Codec::Float16));

        device.flush(&mut caches, 37, 80);

        let LayerCacheState::Attention(restored) = &caches[0] else {
            panic!("layer 0 stays an attention cache");
        };
        assert_eq!(restored.k_even, half_exact_rows(0, 37, EVEN_ODD_ROW, 0));
        assert_eq!(restored.k_odd, half_exact_rows(0, 37, EVEN_ODD_ROW, 1));
        assert_eq!(restored.v, half_exact_rows(0, 37, V_ROW, 2));
    }

    #[test]
    fn committing_a_step_rounds_the_staged_f32_rows_into_the_half_cache_at_the_output_row() {
        let cached_len = 5;
        let (device, _caches) = adopted_half(cached_len, 80);
        let layer = device.layers[0].as_ref().expect("layer 0 is resident");
        let [staging_even, staging_odd, staging_value] = layer
            .staging
            .as_ref()
            .expect("a half-width layer stages its f32 outputs");
        let even: Vec<f32> = (0..EVEN_ODD_ROW).map(|column| 0.1 + column as f32 / 3.0).collect();
        let odd: Vec<f32> = (0..EVEN_ODD_ROW).map(|column| -0.7 - column as f32 / 7.0).collect();
        let value: Vec<f32> = (0..V_ROW).map(|column| 1.0 / (column as f32 + 3.0)).collect();
        omega::write_placed_buffer_f32(staging_even, 0, &even);
        omega::write_placed_buffer_f32(staging_odd, 0, &odd);
        omega::write_placed_buffer_f32(staging_value, 0, &value);

        device.commit_step(cached_len, 1);

        let rounded = |values: &[f32]| -> Vec<f32> {
            values
                .iter()
                .map(|value| half::f16::from_f32(*value).to_f32())
                .collect()
        };
        let row = layer.output_row(cached_len);
        assert_eq!(
            omega::read_placed_buffer_f16_as_f32(
                &layer.k_even,
                row * layer.even_odd_row_bytes,
                EVEN_ODD_ROW
            ),
            rounded(&even)
        );
        assert_eq!(
            omega::read_placed_buffer_f16_as_f32(
                &layer.k_odd,
                row * layer.even_odd_row_bytes,
                EVEN_ODD_ROW
            ),
            rounded(&odd)
        );
        assert_eq!(
            omega::read_placed_buffer_f16_as_f32(&layer.v, row * layer.v_row_bytes, V_ROW),
            rounded(&value)
        );
        assert_ne!(
            rounded(&even),
            even,
            "the staged values are not binary16-exact, so equality above proves the rounding ran"
        );
    }

    #[test]
    fn a_half_width_step_reads_the_cache_and_writes_its_rows_to_staging() {
        let cached_len = 5;
        let (mut device, _caches) = adopted_half(cached_len, 80);
        let leaves = [Some(KvLeafNodes {
            k_even: NodeId(1),
            k_odd: NodeId(2),
            v: NodeId(3),
        })];
        let roots = [LayerCacheRoots::Attention((NodeId(10), NodeId(11), NodeId(12)))];

        device.ready_step(cached_len, 1, 0);
        let placements = device.placements(cached_len, &leaves, &roots);

        let layer = device.layers[0].as_ref().expect("layer 0 is resident");
        let [staging_even, ..] = layer.staging.as_ref().expect("half-width layers stage");
        assert!(core::ptr::eq(placements.inputs[0].1, &layer.k_even));
        assert_eq!(placements.inputs[0].2, 0, "a full layer reads from row 0");
        assert!(core::ptr::eq(placements.outputs[0].1, staging_even));
        assert!(
            placements.outputs.iter().all(|(_, _, offset)| *offset == 0),
            "staged rows start at the front of their buffer"
        );
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
        let roots = [LayerCacheRoots::Attention((NodeId(10), NodeId(11), NodeId(12)))];

        device.ready_step(cached_len, 1, 0);
        let placements = device.placements(cached_len, &leaves, &roots);

        let layer = device.layers[0].as_ref().expect("layer 0 is resident");
        assert!(layer.staging.is_none(), "an f32 cache needs no staging");
        assert!(core::ptr::eq(placements.outputs[0].1, &layer.k_even));
        assert_eq!(
            placements.outputs[0].2,
            cached_len * layer.even_odd_row_bytes,
            "the new row lands at the end of the cached rows"
        );
        assert_eq!(device.cache_codec(), None);
    }

    #[test]
    fn a_quantized_device_cache_is_refused_at_adoption() {
        let mut caches = alloc::vec![LayerCacheState::Attention(host_cache(None, 10))];
        let widths = [LayerPadRowWidths::Attention {
            even_odd_row: EVEN_ODD_ROW,
            v_row: V_ROW,
        }];

        let error = DeviceKv::adopt(
            &mut caches,
            &widths,
            10,
            40,
            4,
            3,
            allocate_placed_buffer,
            GgmlType::Q8_0,
        )
        .err()
        .expect("a Q8_0 device cache has no read path");

        assert!(error.to_string().contains("device kv cache type"));
    }

    #[test]
    fn leaf_nodes_resolve_by_declared_name_in_one_pass() {
        use proxima_tensor::DType;
        let input = |name: &str| Op::Input {
            dtype: DType::Float32,
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
}
