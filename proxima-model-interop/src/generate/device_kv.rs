//! Device-resident KV for the two-range decode loop.
//!
//! The host [`LayerCache`] is the source of truth between calls
//! ([`PrefixState`], the prompt cache), but inside one decode call it only
//! ever grew by the rows the device had just computed and was then copied
//! back to the device whole, every token: `KvPadScratch::fill` re-copied the
//! full history of every layer and each scratch became a fresh no-copy
//! `MTLBuffer` wrapper. [`DeviceKv`] keeps those rows in
//! [`PlacedBuffer`]s instead -- the same caller-owned-buffer mechanism the
//! qwen35 dense-attention placement and the single-range KV path already use
//! (`omega::execute_plan_named_with_placements`): the leaf the program reads
//! is an input placement over the buffer, the freshly computed rows are an
//! output placement written into the buffer's tail, and the host touches
//! neither. A call adopts its host caches once ([`DeviceKv::adopt`]) and hands
//! them back once ([`DeviceKv::flush`]), so everything that reads
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
}

impl DeviceKvLayer {
    fn allocate(
        cache: &LayerCache,
        even_odd_row: usize,
        v_row: usize,
        full_capacity_rows: usize,
        max_step_rows: usize,
    ) -> Result<Self, InteropError> {
        let window = cache.ring_geometry().map(|ring| ring.window);
        let retain = cache.ring_geometry().map_or(0, |ring| ring.capacity);
        let capacity_rows = if window.is_some() {
            (2 * retain + max_step_rows).min(full_capacity_rows)
        } else {
            full_capacity_rows
        };
        let even_odd_row_bytes = even_odd_row * core::mem::size_of::<f32>();
        let v_row_bytes = v_row * core::mem::size_of::<f32>();
        let k_even = allocate_placed_buffer(capacity_rows * even_odd_row_bytes)?;
        let k_odd = allocate_placed_buffer(capacity_rows * even_odd_row_bytes)?;
        let v = allocate_placed_buffer(capacity_rows * v_row_bytes)?;
        // the leaf reads past the live rows up to the bucket; a row never
        // written must be a finite zero, not undefined memory, because the
        // masked softmax weight 0.0 still multiplies it
        omega::metal::zero_placed_buffer(&k_even, capacity_rows * even_odd_row_bytes);
        omega::metal::zero_placed_buffer(&k_odd, capacity_rows * even_odd_row_bytes);
        omega::metal::zero_placed_buffer(&v, capacity_rows * v_row_bytes);
        Ok(Self {
            k_even,
            k_odd,
            v,
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

    fn seed(&mut self, cache: &LayerCache, cached_len: usize) {
        let even_odd_row = self.even_odd_row_bytes / core::mem::size_of::<f32>();
        let v_row = self.v_row_bytes / core::mem::size_of::<f32>();
        let first = self.first_kept(cached_len);
        let kept = cached_len - first;
        self.base_position = first;
        let mut even = Vec::with_capacity(kept * even_odd_row);
        let mut odd = Vec::with_capacity(kept * even_odd_row);
        let mut value = Vec::with_capacity(kept * v_row);
        for position in first..cached_len {
            let slot = cache
                .ring_geometry()
                .map_or(position, |ring| position % ring.capacity);
            even.extend_from_slice(&cache.k_even[slot * even_odd_row..(slot + 1) * even_odd_row]);
            odd.extend_from_slice(&cache.k_odd[slot * even_odd_row..(slot + 1) * even_odd_row]);
            value.extend_from_slice(&cache.v[slot * v_row..(slot + 1) * v_row]);
        }
        omega::write_placed_buffer_f32(&self.k_even, 0, &even);
        omega::write_placed_buffer_f32(&self.k_odd, 0, &odd);
        omega::write_placed_buffer_f32(&self.v, 0, &value);
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
}

impl DeviceKv {
    /// Moves every attention layer's host rows onto the device, or `None`
    /// when this call's caches are not the plain attention shape
    /// (`SharedFromLayer` layers are fine: they own nothing) or a host cache
    /// does not hold exactly `cached_len` rows. Host copies are released here
    /// and rebuilt by [`Self::flush`], so a long context is held once.
    pub(super) fn adopt(
        layer_caches: &mut [LayerCacheState],
        layer_row_widths: &[LayerPadRowWidths],
        cached_len: usize,
        positions_needed: usize,
        bucket_tokens: usize,
        max_step_rows: usize,
    ) -> Result<Option<Self>, InteropError> {
        let full_capacity_rows =
            kv_extent(positions_needed + max_step_rows, usize::MAX, bucket_tokens) + bucket_tokens;
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
        }))
    }

    pub(super) fn max_step_rows(&self) -> usize {
        self.max_step_rows
    }

    /// One flag per layer, for the host paths that must skip a resident one.
    pub(super) fn resident_layers(&self) -> Vec<bool> {
        self.layers.iter().map(Option::is_some).collect()
    }

    /// Readies every sliding layer for a step at `cached_len` positions,
    /// then names the placements the step runs under. `sliding_bound` is the
    /// program's sliding extent this step (`KvRing::bound_extent`).
    pub(super) fn placements<'buffers>(
        &'buffers mut self,
        cached_len: usize,
        new_count: usize,
        sliding_bound: usize,
        leaves: &[Option<KvLeafNodes>],
        layer_roots: &[Qwen35LayerRoots],
    ) -> KvPlacements<'buffers> {
        for layer in self.layers.iter_mut().flatten() {
            layer.make_room(cached_len, new_count, sliding_bound);
        }
        let mut inputs = Vec::with_capacity(self.layers.len() * 3);
        let mut outputs = Vec::with_capacity(self.layers.len() * 3);
        for (index, slot) in self.layers.iter().enumerate() {
            let (Some(layer), Some(Some(leaf)), Some(Qwen35LayerRoots::Attention((even, odd, value)))) =
                (slot, leaves.get(index), layer_roots.get(index))
            else {
                continue;
            };
            let input_row = layer.input_row(cached_len);
            let output_row = layer.output_row(cached_len);
            inputs.push((leaf.k_even, &layer.k_even, input_row * layer.even_odd_row_bytes));
            inputs.push((leaf.k_odd, &layer.k_odd, input_row * layer.even_odd_row_bytes));
            inputs.push((leaf.v, &layer.v, input_row * layer.v_row_bytes));
            outputs.push((*even, &layer.k_even, output_row * layer.even_odd_row_bytes));
            outputs.push((*odd, &layer.k_odd, output_row * layer.even_odd_row_bytes));
            outputs.push((*value, &layer.v, output_row * layer.v_row_bytes));
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
            let even_odd_row = layer.even_odd_row_bytes / core::mem::size_of::<f32>();
            let v_row = layer.v_row_bytes / core::mem::size_of::<f32>();
            let even = omega::read_placed_buffer_f32(
                &layer.k_even,
                first_row * layer.even_odd_row_bytes,
                live * even_odd_row,
            );
            let odd = omega::read_placed_buffer_f32(
                &layer.k_odd,
                first_row * layer.even_odd_row_bytes,
                live * even_odd_row,
            );
            let value = omega::read_placed_buffer_f32(
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
    }
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
        let device = DeviceKv::adopt(&mut caches, &widths, positions, total_positions, 4, 3)
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
    fn a_cache_that_misses_positions_is_not_adopted() {
        let mut caches = alloc::vec![LayerCacheState::Attention(host_cache(None, 10))];
        let widths = [LayerPadRowWidths::Attention {
            even_odd_row: EVEN_ODD_ROW,
            v_row: V_ROW,
        }];
        let adopted = DeviceKv::adopt(&mut caches, &widths, 12, 40, 4, 3)
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
