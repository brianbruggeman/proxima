use super::*;
use crate::block_file::{BlockFileHeader, BlockFileLayer, BlockFileView, encode_block};

fn malformed(reason: &'static str) -> InteropError {
    InteropError::BlockFileMalformed { reason }
}

fn narrow(value: usize) -> Result<u32, InteropError> {
    u32::try_from(value).map_err(|_| malformed("layer dimension exceeds u32"))
}

fn row_bytes(floats_per_row: usize) -> Result<u32, InteropError> {
    floats_per_row
        .checked_mul(size_of::<f32>())
        .ok_or(malformed("layer dimension exceeds u32"))
        .and_then(narrow)
}

fn full_layer(cache: &LayerCache, cached_len: usize) -> Result<BlockFileLayer, InteropError> {
    let whole = |plane: &[f32]| plane.len().is_multiple_of(cached_len);
    if cache.k_odd.len() != cache.k_even.len() || !whole(&cache.k_even) || !whole(&cache.v) {
        return Err(malformed("layer rows disagree with the cached length"));
    }
    Ok(BlockFileLayer {
        rows: narrow(cached_len)?,
        k_even_row_bytes: row_bytes(cache.k_even.len() / cached_len)?,
        k_odd_row_bytes: row_bytes(cache.k_odd.len() / cached_len)?,
        v_row_bytes: row_bytes(cache.v.len() / cached_len)?,
        ring_window: 0,
        ring_capacity: 0,
    })
}

fn ring_layer(cache: &LayerCache, ring: &KvRing) -> Result<BlockFileLayer, InteropError> {
    let disagree = malformed("layer rows disagree with the ring geometry");
    if ring.even_odd_row == 0 || ring.v_row == 0 {
        return Err(disagree);
    }
    let rows = cache.k_even.len() / ring.even_odd_row;
    let whole = cache.k_even.len().is_multiple_of(ring.even_odd_row)
        && cache.k_odd.len() == cache.k_even.len()
        && cache.v.len() == rows * ring.v_row;
    if !whole {
        return Err(disagree);
    }
    Ok(BlockFileLayer {
        rows: narrow(rows)?,
        k_even_row_bytes: row_bytes(ring.even_odd_row)?,
        k_odd_row_bytes: row_bytes(ring.even_odd_row)?,
        v_row_bytes: row_bytes(ring.v_row)?,
        ring_window: narrow(ring.window)?,
        ring_capacity: narrow(ring.capacity)?,
    })
}

fn attention_layer(cache: &LayerCache, cached_len: usize) -> Result<BlockFileLayer, InteropError> {
    if cache.k_even.is_empty() {
        return Err(malformed("layer holds no rows"));
    }
    match cache.ring.as_ref() {
        None => full_layer(cache, cached_len),
        Some(ring) if ring.write_offset != 0 => Err(malformed("ring is displaced")),
        Some(ring) => ring_layer(cache, ring),
    }
}

fn layer_record(entry: &LayerCacheState, cached_len: usize) -> Result<BlockFileLayer, InteropError> {
    match entry {
        LayerCacheState::SharedFromLayer => Ok(BlockFileLayer::default()),
        LayerCacheState::Attention(cache) => attention_layer(cache, cached_len),
        LayerCacheState::DenseAttention(_) | LayerCacheState::Ssm(_) => {
            Err(malformed("layer kind has no row planes"))
        }
    }
}

fn layer_planes(entry: &LayerCacheState) -> [&[f32]; 3] {
    match entry {
        LayerCacheState::Attention(cache) => [&cache.k_even, &cache.k_odd, &cache.v],
        _ => [&[], &[], &[]],
    }
}

fn read_plane(view: &BlockFileView<'_>, layer: usize, plane: usize) -> Result<Vec<f32>, InteropError> {
    let mut values = Vec::new();
    view.plane_f32(layer, plane, &mut values)
        .ok_or(malformed("plane is not whole f32 values"))?;
    Ok(values)
}

fn restored_ring(record: &BlockFileLayer) -> Result<Option<KvRing>, InteropError> {
    if record.ring_window == 0 {
        return Ok(None);
    }
    let window = record.ring_window as usize;
    let capacity = record.ring_capacity as usize;
    if window > capacity {
        return Err(malformed("ring window exceeds ring capacity"));
    }
    let even_odd_row = record.k_even_row_bytes as usize / size_of::<f32>();
    let v_row = record.v_row_bytes as usize / size_of::<f32>();
    Ok(Some(KvRing::new(window, capacity - window, even_odd_row, v_row, 0)))
}

fn restored_layer(view: &BlockFileView<'_>, layer: usize) -> Result<LayerCacheState, InteropError> {
    let record = &view.header.layers[layer];
    if record.rows == 0 {
        return Ok(LayerCacheState::SharedFromLayer);
    }
    let ring = restored_ring(record)?;
    Ok(LayerCacheState::Attention(LayerCache {
        k_even: read_plane(view, layer, 0)?,
        k_odd: read_plane(view, layer, 1)?,
        v: read_plane(view, layer, 2)?,
        ring,
        ..LayerCache::new()
    }))
}

impl PrefixState {
    /// Writes this state's rows into `out` as a kv block file whose `base_position` is 0.
    ///
    /// Composes [`encode_block`]: the planes are `pub(super)` fields no caller outside this
    /// module can read, so this wrapper is how a state is spilled. Full layers write every
    /// position; ring layers write every ring slot, slack included, so a restore reproduces the
    /// ring exactly; shared-KV layers write empty planes. Planes are borrowed, not cloned.
    ///
    /// # Errors
    ///
    /// [`InteropError::BlockFileMalformed`] when nothing is cached, a layer is dense or recurrent,
    /// a ring is displaced, or a layer's planes disagree with its geometry. `out` is untouched.
    pub fn to_block_file(
        &self,
        descriptor_digest: [u8; 16],
        content_key: u64,
        out: &mut Vec<u8>,
    ) -> Result<(), InteropError> {
        if self.cached_len == 0 {
            return Err(malformed("nothing cached"));
        }
        let layers = self
            .layer_caches
            .iter()
            .map(|entry| layer_record(entry, self.cached_len))
            .collect::<Result<Vec<_>, _>>()?;
        let header = BlockFileHeader { descriptor_digest, content_key, base_position: 0, layers };
        let planes = self.layer_caches.iter().flat_map(layer_planes).collect::<Vec<_>>();
        encode_block(&header, &planes, out)
    }

    /// Rebuilds a state from a decoded block file, the inverse of [`PrefixState::to_block_file`].
    ///
    /// Composes `BlockFileView::plane_f32`: each plane is read into a fresh vector, ring layers
    /// get their geometry back with a zero write offset, and layers with no rows become shared-KV
    /// layers. `ids` and `cached_len` come from the caller because the file carries rows only.
    ///
    /// # Errors
    ///
    /// [`InteropError::BlockFileMalformed`] when `cached_len` exceeds `ids`, a plane is not whole
    /// `f32` values, or a ring window exceeds its capacity.
    pub fn from_block_file(
        ids: Vec<u32>,
        cached_len: usize,
        view: &BlockFileView<'_>,
    ) -> Result<PrefixState, InteropError> {
        if cached_len > ids.len() {
            return Err(malformed("cached length exceeds ids"));
        }
        let layer_caches = (0..view.header.layers.len())
            .map(|layer| restored_layer(view, layer))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PrefixState { ids, layer_caches, cached_len })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::block_file::decode_block;

    fn ramp(start: f32, count: usize) -> Vec<f32> {
        (0..count).map(|index| start + index as f32 * 0.5).collect()
    }

    fn three_layer_state() -> PrefixState {
        let mut full = LayerCache::new();
        full.append(&ramp(0.0, 8), &ramp(100.0, 8), &ramp(200.0, 4));
        let mut ring = LayerCache::ring(KvRing::new(2, 1, 2, 1, 0), 4);
        ring.append_at(0, &ramp(0.0, 8), &ramp(100.0, 8), &ramp(200.0, 4));
        PrefixState {
            ids: vec![2, 818, 5279, 529],
            layer_caches: vec![
                LayerCacheState::Attention(full),
                LayerCacheState::Attention(ring),
                LayerCacheState::SharedFromLayer,
            ],
            cached_len: 4,
        }
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    #[test]
    fn prefix_state_file_encodes_full_ring_and_shared_layers() {
        let state = three_layer_state();
        let mut out = Vec::new();

        let outcome = state.to_block_file([0x33; 16], 0xABCD, &mut out);

        assert!(outcome.is_ok());
        let view = decode_block(&out).expect("the encoded state decodes");
        let ring_row = |rows, window, capacity| BlockFileLayer {
            rows,
            k_even_row_bytes: 8,
            k_odd_row_bytes: 8,
            v_row_bytes: 4,
            ring_window: window,
            ring_capacity: capacity,
        };
        assert_eq!(view.header.layers, vec![ring_row(4, 0, 0), ring_row(3, 2, 3), BlockFileLayer::default()]);
        assert_eq!(view.header.descriptor_digest, [0x33; 16]);
        assert_eq!(view.header.content_key, 0xABCD);
        assert_eq!(view.header.base_position, 0);
        let (LayerCacheState::Attention(full), LayerCacheState::Attention(ring)) =
            (&state.layer_caches[0], &state.layer_caches[1])
        else {
            panic!("fixture layers 0 and 1 are attention layers");
        };
        let mut plane = Vec::new();
        view.plane_f32(0, 0, &mut plane).expect("layer 0 k_even plane");
        assert_eq!(bits(&plane), bits(&full.k_even));
        view.plane_f32(1, 2, &mut plane).expect("layer 1 v plane");
        assert_eq!(bits(&plane), bits(&ring.v));
        assert_eq!(view.plane_f32(2, 1, &mut plane), Some(()));
        assert!(plane.is_empty());
    }

    #[test]
    fn prefix_state_file_refuses_layers_without_row_planes() {
        let state = PrefixState {
            ids: vec![2],
            layer_caches: vec![LayerCacheState::Ssm(SsmLayerCache::new(2, 2))],
            cached_len: 1,
        };
        let mut out = Vec::new();

        let refused = state.to_block_file([0; 16], 1, &mut out);

        assert!(matches!(
            refused,
            Err(InteropError::BlockFileMalformed { reason: "layer kind has no row planes" })
        ));
    }

    #[test]
    fn prefix_state_file_refuses_a_displaced_ring() {
        let mut ring = LayerCache::ring(KvRing::new(2, 1, 2, 1, 1), 4);
        ring.append_at(0, &ramp(0.0, 8), &ramp(100.0, 8), &ramp(200.0, 4));
        let state = PrefixState {
            ids: vec![2, 818, 5279, 529],
            layer_caches: vec![LayerCacheState::Attention(ring)],
            cached_len: 4,
        };
        let mut out = Vec::new();

        let refused = state.to_block_file([0; 16], 1, &mut out);

        assert!(matches!(refused, Err(InteropError::BlockFileMalformed { reason: "ring is displaced" })));
    }

    #[test]
    fn prefix_state_file_refuses_an_empty_state() {
        let mut state = three_layer_state();
        state.cached_len = 0;
        let mut out = Vec::new();

        let refused = state.to_block_file([0; 16], 1, &mut out);

        assert!(matches!(refused, Err(InteropError::BlockFileMalformed { reason: "nothing cached" })));
    }

    fn one_row_header(ring_window: u32, ring_capacity: u32) -> BlockFileHeader {
        let layer = BlockFileLayer {
            rows: 1,
            k_even_row_bytes: 4,
            k_odd_row_bytes: 4,
            v_row_bytes: 4,
            ring_window,
            ring_capacity,
        };
        BlockFileHeader { descriptor_digest: [0x33; 16], content_key: 1, base_position: 0, layers: vec![layer] }
    }

    fn one_row_bytes(header: &BlockFileHeader) -> Vec<u8> {
        let mut out = Vec::new();
        encode_block(header, &[&[1.0], &[2.0], &[3.0]], &mut out).expect("one row block encodes");
        out
    }

    #[test]
    fn prefix_state_file_round_trips_every_layer_bit_for_bit() {
        let state = three_layer_state();
        let mut first = Vec::new();
        state.to_block_file([0x33; 16], 0xABCD, &mut first).expect("the state encodes");
        let view = decode_block(&first).expect("the encoded state decodes");

        let restored = PrefixState::from_block_file(state.ids.clone(), state.cached_len, &view)
            .expect("the decoded state restores");

        let attention = |state: &PrefixState, layer: usize| match &state.layer_caches[layer] {
            LayerCacheState::Attention(cache) => cache.clone(),
            _ => panic!("fixture layer {layer} is an attention layer"),
        };
        for layer in 0..2 {
            let (source, back) = (attention(&state, layer), attention(&restored, layer));
            assert_eq!(bits(&back.k_even), bits(&source.k_even));
            assert_eq!(bits(&back.k_odd), bits(&source.k_odd));
            assert_eq!(bits(&back.v), bits(&source.v));
        }
        assert_eq!(attention(&restored, 1).ring_geometry(), Some(&KvRing::new(2, 1, 2, 1, 0)));
        assert!(matches!(restored.layer_caches[2], LayerCacheState::SharedFromLayer));
        assert_eq!(restored.cached_len, state.cached_len);
        assert_eq!(restored.ids, state.ids);
        let mut second = Vec::new();
        restored.to_block_file([0x33; 16], 0xABCD, &mut second).expect("the restored state encodes");
        assert_eq!(second, first);
    }

    #[test]
    fn prefix_state_file_restore_refuses_a_ring_window_over_its_capacity() {
        let bytes = one_row_bytes(&one_row_header(4, 3));
        let view = decode_block(&bytes).expect("the hand built block decodes");

        let refused = PrefixState::from_block_file(vec![2], 1, &view);

        assert!(matches!(
            refused,
            Err(InteropError::BlockFileMalformed { reason: "ring window exceeds ring capacity" })
        ));
    }

    #[test]
    fn prefix_state_file_restore_refuses_cached_length_over_ids() {
        let bytes = one_row_bytes(&one_row_header(0, 0));
        let view = decode_block(&bytes).expect("the hand built block decodes");

        let refused = PrefixState::from_block_file(vec![2, 818, 5279, 529], 5, &view);

        assert!(matches!(refused, Err(InteropError::BlockFileMalformed { reason: "cached length exceeds ids" })));
    }

    #[test]
    fn prefix_state_file_restore_refuses_a_plane_that_is_not_whole_floats() {
        let mut bytes = one_row_bytes(&one_row_header(0, 0));
        bytes[48 + 4..48 + 8].copy_from_slice(&6u32.to_le_bytes());
        bytes.extend_from_slice(&[0, 0]);
        let view = decode_block(&bytes).expect("the patched block still decodes");

        let refused = PrefixState::from_block_file(vec![2], 1, &view);

        assert!(matches!(
            refused,
            Err(InteropError::BlockFileMalformed { reason: "plane is not whole f32 values" })
        ));
    }
}
