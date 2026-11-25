//! Per-dispatch GPU time of the emitted Flash cached-attention kernel (`omega::emit_with_attention_variant`,
//! f16 cache, sinked, kv_heads 4 x groups 8, head_dim 128) at the full-layer shape and flint's sliding ring shape.
//! Arms are textual toggles of the emitted source's one dispatch condition, `(columns & 7) == 0`:
//! `peeled` as emitted, `branch_every_tile` forces the partial-tile instance on every block (the layout of 4401f597),
//! `no_partial_path` forces the whole-tile instance (the layout of 689ef8ce; reads past a ragged cache, timing only).
//!
//! ```sh
//! cargo run -p proxima-omega --release --features metal-attn-variants --example attn_flash_peel_bench
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)]

fn main() {
    #[cfg(all(feature = "metal", feature = "metal-attn-variants", target_os = "macos"))]
    imp::run();
    #[cfg(not(all(feature = "metal", feature = "metal-attn-variants", target_os = "macos")))]
    println!("attn_flash_peel_bench requires --features metal-attn-variants on macOS");
}

#[cfg(all(feature = "metal", feature = "metal-attn-variants", target_os = "macos"))]
mod imp {
    use core::ffi::c_void;
    use core::ptr::NonNull;

    use half::f16;
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions, MTLComputeCommandEncoder,
        MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLMathMode, MTLResourceOptions, MTLSize,
    };
    use omega::{AttentionDecomposition, AttentionKvStorage, AttentionVariant, Codec, PackedOperands};
    use proxima_numeric::{DType, NumericPolicy};
    use proxima_tensor::{BoundOp, BoundOpKind, CachedAttentionArithmetic, Layout, NodeId};

    const KV_HEADS: usize = 4;
    const GROUPS: usize = 8;
    const HEAD_DIM: usize = 128;
    const ROWS: usize = 512;
    const TOGGLE: &str = "(columns & 7) == 0";

    struct Shape {
        label: &'static str,
        cached_rows: usize,
        live: usize,
        lower: i64,
    }

    fn op(shape: &Shape) -> BoundOp {
        let layout = Layout { base: 0, strides: vec![1_i64].into() };
        BoundOp {
            node: NodeId(20),
            dtype: DType::FLOAT32,
            extents: vec![ROWS as u64, KV_HEADS as u64, GROUPS as u64, HEAD_DIM as u64],
            kind: BoundOpKind::CachedAttention {
                operands: (0..10).map(|index| (NodeId(index), layout.clone(), None)).collect(),
                query_rows: ROWS as u64,
                cached_key_rows: shape.cached_rows as u64,
                new_key_rows: ROWS as u64,
                kv_heads: KV_HEADS as u64,
                query_groups: GROUPS as u64,
                head_dim: HEAD_DIM as u64,
                rotary_dim: HEAD_DIM as u64,
                scale: 0.088_388_35,
                cached_lower_inclusive: shape.lower,
                new_upper_inclusive: 0,
                arithmetic: CachedAttentionArithmetic::OnlineF32,
                attention_sink: true, planes: proxima_tensor::bind::ScorePlanes::Paired,
            },
        }
    }

    fn values(state: &mut u64, count: usize, scale: f32) -> Vec<f32> {
        (0..count)
            .map(|_| {
                *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                ((*state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale
            })
            .collect()
    }

    fn buffer(device: &ProtocolObject<dyn MTLDevice>, bytes: &[u8]) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let pointer = NonNull::new(bytes.as_ptr().cast_mut().cast::<c_void>()).expect("slice pointer");
        unsafe { device.newBufferWithBytes_length_options(pointer, bytes.len(), MTLResourceOptions::StorageModeShared) }.expect("buffer")
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_le_bytes()).collect()
    }

    fn f16_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes()).collect()
    }

    fn pipeline(device: &ProtocolObject<dyn MTLDevice>, source: &str, entry: &str) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        let options = MTLCompileOptions::new();
        options.setMathMode(MTLMathMode::Relaxed);
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
            .unwrap_or_else(|error| panic!("compiles {entry}: {}", error.localizedDescription()));
        let function = library.newFunctionWithName(&NSString::from_str(entry)).expect("entry");
        device.newComputePipelineStateWithFunction_error(&function).expect("pipeline")
    }

    fn pass(
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        state: &ProtocolObject<dyn MTLComputePipelineState>,
        buffers: &[Retained<ProtocolObject<dyn MTLBuffer>>],
        threadgroups: usize,
        width: usize,
        repeats: usize,
    ) -> f64 {
        let command_buffer = queue.commandBuffer().expect("command buffer");
        let encoder = command_buffer.computeCommandEncoder().expect("encoder");
        for _ in 0..repeats {
            encoder.setComputePipelineState(state);
            for (index, held) in buffers.iter().enumerate() {
                unsafe { encoder.setBuffer_offset_atIndex(Some(held), 0, index) };
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: threadgroups, height: 1, depth: 1 },
                MTLSize { width, height: 1, depth: 1 },
            );
        }
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        (command_buffer.GPUEndTime() - command_buffer.GPUStartTime()) * 1e6 / repeats as f64
    }

    pub fn run() {
        let device = MTLCreateSystemDefaultDevice().expect("metal device");
        let queue = device.newCommandQueue().expect("queue");
        let shapes = [
            Shape { label: "full_q512_c2048", cached_rows: 2048, live: 1536, lower: i64::MIN },
            Shape { label: "sliding_q512_c513_w513", cached_rows: 513, live: 513, lower: -512 },
        ];
        let rounds: usize = std::env::var("PEEL_ROUNDS").ok().and_then(|value| value.parse().ok()).unwrap_or(15);
        for shape in &shapes {
            let bound = op(shape);
            let mut packed = PackedOperands::new();
            for index in [2, 3, 6] {
                packed.insert(bound.operands()[index].0, Codec::Float16);
            }
            let variant = AttentionVariant { kv_storage: AttentionKvStorage::F16, decomposition: AttentionDecomposition::Flash, ..AttentionVariant::default() };
            let kernel = omega::emit_with_attention_variant(&bound, &packed, NumericPolicy::llama_relaxed(), variant).expect("flash emits");
            assert!(kernel.source.contains(TOGGLE), "the emitted flash source carries the dispatch toggle");
            let width = kernel.grid.threadgroup_width.expect("flash threadgroup width") as usize;
            let threadgroups = kernel.grid.threads as usize / width;
            let mut state = 0x5eed_u64;
            let plane = HEAD_DIM / 2;
            let vectors = ROWS * KV_HEADS * GROUPS;
            let pad = 16;
            let cached_k = (shape.cached_rows + pad) * KV_HEADS * plane;
            let cached_v = (shape.cached_rows + pad) * KV_HEADS * HEAD_DIM;
            let fresh = ROWS * KV_HEADS;
            let buffers = vec![
                buffer(&device, &f32_bytes(&values(&mut state, vectors * plane, 1.0))),
                buffer(&device, &f32_bytes(&values(&mut state, vectors * plane, 1.0))),
                buffer(&device, &f16_bytes(&values(&mut state, cached_k, 1.0))),
                buffer(&device, &f16_bytes(&values(&mut state, cached_k, 1.0))),
                buffer(&device, &f32_bytes(&values(&mut state, fresh * plane, 1.0))),
                buffer(&device, &f32_bytes(&values(&mut state, fresh * plane, 1.0))),
                buffer(&device, &f16_bytes(&values(&mut state, cached_v, 1.0))),
                buffer(&device, &f32_bytes(&values(&mut state, fresh * HEAD_DIM, 1.0))),
                buffer(&device, &f32_bytes(&[shape.live as f32])),
                buffer(&device, &f32_bytes(&values(&mut state, KV_HEADS * GROUPS, 2.0))),
                buffer(&device, &vec![0u8; vectors * HEAD_DIM * 4]),
                buffer(&device, &[(vectors as i64).to_le_bytes(), 1i64.to_le_bytes()].concat()),
            ];
            let arms = [
                ("peeled", kernel.source.clone()),
                ("branch_every_tile", kernel.source.replace(TOGGLE, "false")),
                ("no_partial_path", kernel.source.replace(TOGGLE, "true")),
            ];
            let compiled: Vec<_> = arms.iter().map(|(name, source)| (*name, pipeline(&device, source, &kernel.entry))).collect();
            let mut samples: Vec<Vec<f64>> = vec![Vec::new(); compiled.len()];
            for (_, state) in &compiled {
                let _ = pass(&queue, state, &buffers, threadgroups, width, 3);
            }
            for round in 0..rounds {
                for offset in 0..compiled.len() {
                    let index = (round + offset) % compiled.len();
                    samples[index].push(pass(&queue, &compiled[index].1, &buffers, threadgroups, width, 5));
                }
            }
            for ((name, _), values) in compiled.iter().zip(&samples) {
                let mut sorted = values.clone();
                sorted.sort_by(f64::total_cmp);
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance = values.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / values.len() as f64;
                println!(
                    "shape={} arm={name} n={} median_us={:.1} min_us={:.1} cov_pct={:.2}",
                    shape.label,
                    values.len(),
                    sorted[sorted.len() / 2],
                    sorted[0],
                    100.0 * variance.sqrt() / mean
                );
            }
        }
    }
}
