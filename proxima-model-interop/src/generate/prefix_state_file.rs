use super::*;
use crate::block_file::{BlockFileHeader, BlockFileLayer, encode_block};

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
}
