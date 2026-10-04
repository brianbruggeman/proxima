//! Per-dispatch GPU time and correctness of the decode split partial kernel
//! (`CachedAttentionForm::TwoRangeDecodeSplit`, `omega/src/msl/
//! cached_attention_decode_split.rs`) at the gemma4-E2B layer shapes, with the
//! kernel's runtime knobs (`splits`, `chunks`) and its three layout
//! constants (`lanes_per_key`, `batch`, `cap`) swept from the environment, so one
//! binary measures every variant without a rebuild.
//!
//! The constants are rewritten in the emitted text before it is compiled:
//! `emit` bakes them from `[attention_decode]`, and the probe replaces the
//! `constexpr` declarations, which is how a sweep reaches kernel shapes the
//! sized defaults do not select. `splits` and `chunks` are the kernel's own
//! runtime uniforms, so they need no rewrite.
//!
//! Timing is `MTLCommandBuffer` GPU timestamps (`GPUEndTime - GPUStartTime`)
//! over `PROBE_REPEATS` in-order dispatches that rotate through
//! `PROBE_LAYERS` distinct K/V cache sets, so the working set is the model's
//! (one cache per layer), not one cache re-read warm. Correctness merges the
//! kernel's interleaved scratch on the CPU and compares it with a direct CPU
//! softmax attention over the same buffers.
//!
//! ```sh
//! PROBE_HEAD_DIM=512 PROBE_LIVE=1030 PROBE_CHUNKS=4 PROBE_BATCH=2 \
//!     cargo run -p omega --release --features metal --example attn_decode_split_probe
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)]

fn main() {
    #[cfg(all(feature = "metal", feature = "metal-attn-split-decode", target_os = "macos"))]
    imp::run();
    #[cfg(not(all(feature = "metal", feature = "metal-attn-split-decode", target_os = "macos")))]
    println!("attn_decode_split_probe requires --features metal on macOS");
}

#[cfg(all(feature = "metal", feature = "metal-attn-split-decode", target_os = "macos"))]
mod imp {
    use core::ffi::c_void;
    use core::ptr::NonNull;

    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
        MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
        MTLLibrary, MTLMathMode, MTLResourceOptions, MTLSize,
    };
    use omega::PackedOperands;
    use proxima_tensor::{BoundOp, BoundOpKind, DType, Layout, NodeId, NumericPolicy};

    const HEADS: usize = 8;
    const SIMD_WIDTH: u64 = 32;

    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name).map_or(default, |value| {
            value
                .parse()
                .unwrap_or_else(|error| panic!("{name}={value} is not an integer: {error}"))
        })
    }

    struct Shape {
        head_dim: usize,
        compiled_cached_rows: usize,
        live_rows: usize,
        lower: i64,
        layers: usize,
    }

    impl Shape {
        fn from_env() -> Self {
            let head_dim = env_usize("PROBE_HEAD_DIM", 512);
            let global = head_dim == 512;
            Self {
                head_dim,
                compiled_cached_rows: env_usize(
                    "PROBE_CACHED_ROWS",
                    if global { 2048 } else { 512 },
                ),
                live_rows: env_usize("PROBE_LIVE", if global { 1030 } else { 512 }),
                lower: if global { i64::MIN } else { -511 },
                layers: env_usize("PROBE_LAYERS", if global { 7 } else { 28 }),
            }
        }

        fn op(&self) -> BoundOp {
            let layout = Layout {
                base: 0,
                strides: vec![1_i64].into(),
            };
            BoundOp {
                node: NodeId(9),
                dtype: DType::Float32,
                extents: vec![1, 1, HEADS as u64, self.head_dim as u64],
                kind: BoundOpKind::CachedAttention {
                    operands: (0..9)
                        .map(|index| (NodeId(index), layout.clone(), None))
                        .collect(),
                    query_rows: 1,
                    cached_key_rows: self.compiled_cached_rows as u64,
                    new_key_rows: 1,
                    kv_heads: 1,
                    query_groups: HEADS as u64,
                    head_dim: self.head_dim as u64,
                    rotary_dim: self.head_dim as u64,
                    scale: 1.0,
                    cached_lower_inclusive: self.lower,
                    new_upper_inclusive: 0,
                },
            }
        }
    }

    struct Lcg(u64);

    impl Lcg {
        fn next_unit(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }

        fn vector(&mut self, count: usize, scale: f32) -> Vec<f32> {
            (0..count).map(|_| self.next_unit() * scale).collect()
        }
    }

    struct CacheSet {
        key_real: Vec<f32>,
        key_imag: Vec<f32>,
        value: Vec<f32>,
        new_key_real: Vec<f32>,
        new_key_imag: Vec<f32>,
        new_value: Vec<f32>,
    }

    fn cache_set(lcg: &mut Lcg, shape: &Shape) -> CacheSet {
        let plane = shape.head_dim / 2;
        let rows = shape.compiled_cached_rows;
        CacheSet {
            key_real: lcg.vector(rows * plane, 0.5),
            key_imag: lcg.vector(rows * plane, 0.5),
            value: lcg.vector(rows * shape.head_dim, 1.0),
            new_key_real: lcg.vector(plane, 0.5),
            new_key_imag: lcg.vector(plane, 0.5),
            new_value: lcg.vector(shape.head_dim, 1.0),
        }
    }

    fn compile(
        device: &ProtocolObject<dyn MTLDevice>,
        source: &str,
        entry: &str,
    ) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        let options = MTLCompileOptions::new();
        options.setMathMode(MTLMathMode::Relaxed);
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
            .unwrap_or_else(|error| panic!("compiles {entry}: {}", error.localizedDescription()));
        let function = library
            .newFunctionWithName(&NSString::from_str(entry))
            .unwrap_or_else(|| panic!("kernel entry `{entry}` missing from its own library"));
        device
            .newComputePipelineStateWithFunction_error(&function)
            .unwrap_or_else(|error| {
                panic!("creates the pipeline for {entry}: {}", error.localizedDescription())
            })
    }

    fn shared_buffer(
        device: &ProtocolObject<dyn MTLDevice>,
        bytes: &[u8],
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let pointer = NonNull::new(bytes.as_ptr().cast_mut().cast::<c_void>())
            .expect("a slice pointer is never null");
        unsafe {
            device.newBufferWithBytes_length_options(
                pointer,
                bytes.len(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .expect("device allocates a shared buffer")
    }

    fn f32_buffer(
        device: &ProtocolObject<dyn MTLDevice>,
        values: &[f32],
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
        shared_buffer(device, &bytes)
    }

    fn read_f32(buffer: &ProtocolObject<dyn MTLBuffer>, count: usize) -> Vec<f32> {
        let pointer = buffer.contents().as_ptr().cast::<f32>();
        unsafe { core::slice::from_raw_parts(pointer, count) }.to_vec()
    }

    struct LayerBuffers {
        inputs: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
        scratch: Retained<ProtocolObject<dyn MTLBuffer>>,
    }

    fn layer_buffers(
        device: &ProtocolObject<dyn MTLDevice>,
        query: &[Vec<f32>; 2],
        cache: &CacheSet,
        live: usize,
        scratch_floats: usize,
    ) -> LayerBuffers {
        let inputs = vec![
            f32_buffer(device, &query[0]),
            f32_buffer(device, &query[1]),
            f32_buffer(device, &cache.key_real),
            f32_buffer(device, &cache.key_imag),
            f32_buffer(device, &cache.new_key_real),
            f32_buffer(device, &cache.new_key_imag),
            f32_buffer(device, &cache.value),
            f32_buffer(device, &cache.new_value),
            f32_buffer(device, &[live as f32]),
        ];
        let scratch = f32_buffer(device, &vec![0.0_f32; scratch_floats]);
        LayerBuffers { inputs, scratch }
    }

    struct Launch<'a> {
        pipeline: &'a ProtocolObject<dyn MTLComputePipelineState>,
        uniforms: &'a ProtocolObject<dyn MTLBuffer>,
        threadgroups: usize,
        threads_per_threadgroup: usize,
    }

    fn encode(
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        launch: &Launch<'_>,
        layer: &LayerBuffers,
    ) {
        encoder.setComputePipelineState(launch.pipeline);
        for (index, buffer) in layer.inputs.iter().enumerate() {
            unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), 0, index) };
        }
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&layer.scratch), 0, 9);
            encoder.setBuffer_offset_atIndex(Some(launch.uniforms), 0, 10);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: launch.threadgroups,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: launch.threads_per_threadgroup,
                height: 1,
                depth: 1,
            },
        );
    }

    fn timed_pass(
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        launch: &Launch<'_>,
        layers: &[LayerBuffers],
        repeats: usize,
    ) -> f64 {
        let command_buffer = queue.commandBuffer().expect("command buffer");
        let encoder = command_buffer.computeCommandEncoder().expect("compute encoder");
        for repeat in 0..repeats {
            encode(&encoder, launch, &layers[repeat % layers.len()]);
        }
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        (command_buffer.GPUEndTime() - command_buffer.GPUStartTime()) * 1e9
    }

    fn attention_reference(
        query: &[Vec<f32>; 2],
        cache: &CacheSet,
        shape: &Shape,
        head: usize,
    ) -> Vec<f32> {
        let plane = shape.head_dim / 2;
        let first = if shape.lower == i64::MIN {
            0
        } else {
            (shape.live_rows as i64 + shape.lower).max(0) as usize
        };
        let last = shape.live_rows;
        let score = |key: usize| -> f32 {
            let (real, imag) = if key < shape.live_rows {
                (
                    &cache.key_real[key * plane..(key + 1) * plane],
                    &cache.key_imag[key * plane..(key + 1) * plane],
                )
            } else {
                (&cache.new_key_real[..], &cache.new_key_imag[..])
            };
            let q_real = &query[0][head * plane..(head + 1) * plane];
            let q_imag = &query[1][head * plane..(head + 1) * plane];
            (0..plane)
                .map(|pair| real[pair] * q_real[pair] + imag[pair] * q_imag[pair])
                .sum()
        };
        let scores: Vec<f32> = (first..=last).map(score).collect();
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f32> = scores.iter().map(|value| (value - maximum).exp()).collect();
        let total: f32 = weights.iter().sum();
        (0..shape.head_dim)
            .map(|dim| {
                (first..=last)
                    .zip(&weights)
                    .map(|(key, weight)| {
                        let value = if key < shape.live_rows {
                            cache.value[key * shape.head_dim + dim]
                        } else {
                            cache.new_value[dim]
                        };
                        weight * value
                    })
                    .sum::<f32>()
                    / total
            })
            .collect()
    }

    fn merged_from_scratch(
        scratch: &[f32],
        shape: &Shape,
        splits: usize,
        head: usize,
    ) -> Vec<f32> {
        let stats_base = HEADS * shape.head_dim * splits;
        let stats = |split: usize| {
            let at = stats_base + (head * splits + split) * 2;
            (scratch[at], scratch[at + 1])
        };
        let global_max = (0..splits)
            .map(|split| stats(split).0)
            .fold(f32::NEG_INFINITY, f32::max);
        let weight = |split: usize| {
            let maximum = stats(split).0;
            if maximum == f32::NEG_INFINITY {
                0.0
            } else {
                (maximum - global_max).exp()
            }
        };
        let total: f32 = (0..splits).map(|split| stats(split).1 * weight(split)).sum();
        (0..shape.head_dim)
            .map(|dim| {
                let quad = dim / 4;
                let lane = dim % 4;
                (0..splits)
                    .map(|split| {
                        let at = (((head * (shape.head_dim / 4) + quad) * splits + split) * 4) + lane;
                        scratch[at] * weight(split)
                    })
                    .sum::<f32>()
                    / total
            })
            .collect()
    }

    fn rewrite_constant(source: &str, declaration: &str, value: Option<usize>) -> String {
        let Some(value) = value else {
            return source.to_string();
        };
        let marker = format!("constexpr {declaration} = ");
        let start = source
            .find(&marker)
            .unwrap_or_else(|| panic!("emitted kernel has no `{marker}`"));
        let value_start = start + marker.len();
        let value_end = value_start
            + source[value_start..]
                .find(';')
                .expect("a constexpr line ends in a semicolon");
        format!(
            "{}{}{}",
            &source[..value_start],
            value,
            &source[value_end..]
        )
    }

    pub fn run() {
        let shape = Shape::from_env();
        let op = shape.op();
        let kernel = omega::emit(&op, &PackedOperands::new(), NumericPolicy::llama_relaxed())
            .expect("the decode split kernel emits");
        assert!(kernel.entry.ends_with("_ds"), "not the decode split form: {}", kernel.entry);
        let default_width = kernel
            .grid
            .threadgroup_width
            .or(kernel.grid.grid2d.map(|spec| spec.threads_per_threadgroup_x))
            .unwrap_or(SIMD_WIDTH);
        let default_chunks = (default_width / SIMD_WIDTH) as usize;
        let default_splits = (kernel.grid.threads / (HEADS as u64 * default_width)) as usize;
        let chunks = env_usize("PROBE_CHUNKS", default_chunks);
        let splits = env_usize("PROBE_SPLITS", default_splits);
        let lanes = std::env::var("PROBE_LANES").ok().map(|value| value.parse().unwrap());
        let batch = std::env::var("PROBE_BATCH").ok().map(|value| value.parse().unwrap());
        let repeats = env_usize("PROBE_REPEATS", 140);
        let passes = env_usize("PROBE_PASSES", 9);

        let source = rewrite_constant(&kernel.source, "short lanes_per_key", lanes);
        let source = rewrite_constant(&source, "short batch", batch);
        let cap = std::env::var("PROBE_CAP").ok().map(|value| value.parse().unwrap());
        let source = rewrite_constant(&source, "long cap", cap);

        let device = MTLCreateSystemDefaultDevice().expect("a metal device");
        let queue = device.newCommandQueue().expect("a command queue");
        let pipeline = compile(&device, &source, &kernel.entry);
        let threads_per_threadgroup = chunks * SIMD_WIDTH as usize;
        if threads_per_threadgroup > pipeline.maxTotalThreadsPerThreadgroup() {
            println!(
                "probe skipped reason=threadgroup_width width={threads_per_threadgroup} max_threads_per_threadgroup={}",
                pipeline.maxTotalThreadsPerThreadgroup()
            );
            return;
        }

        let mut lcg = Lcg(0x5eed_0001);
        let plane = shape.head_dim / 2;
        let query = [
            lcg.vector(HEADS * plane, 0.5),
            lcg.vector(HEADS * plane, 0.5),
        ];
        let scratch_floats = HEADS * shape.head_dim * splits + HEADS * splits * 2;
        let caches: Vec<CacheSet> = (0..shape.layers).map(|_| cache_set(&mut lcg, &shape)).collect();
        let layers: Vec<LayerBuffers> = caches
            .iter()
            .map(|cache| layer_buffers(&device, &query, cache, shape.live_rows, scratch_floats))
            .collect();
        let uniform_words = [HEADS as i64, chunks as i64, splits as i64];
        let uniform_bytes: Vec<u8> = uniform_words.iter().flat_map(|word| word.to_le_bytes()).collect();
        let uniforms = shared_buffer(&device, &uniform_bytes);
        let launch = Launch {
            pipeline: &pipeline,
            uniforms: &uniforms,
            threadgroups: HEADS * splits,
            threads_per_threadgroup,
        };

        let command_buffer = queue.commandBuffer().expect("command buffer");
        let encoder = command_buffer.computeCommandEncoder().expect("compute encoder");
        encode(&encoder, &launch, &layers[0]);
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        let scratch = read_f32(&layers[0].scratch, scratch_floats);
        let mut worst = 0.0_f32;
        let mut magnitude = 0.0_f32;
        for head in 0..HEADS {
            let want = attention_reference(&query, &caches[0], &shape, head);
            let got = merged_from_scratch(&scratch, &shape, splits, head);
            for (expected, actual) in want.iter().zip(&got) {
                worst = worst.max((expected - actual).abs());
                magnitude = magnitude.max(expected.abs());
            }
        }

        let _ = timed_pass(&queue, &launch, &layers, repeats);
        let mut per_dispatch_us: Vec<f64> = (0..passes)
            .map(|_| timed_pass(&queue, &launch, &layers, repeats) / repeats as f64 / 1e3)
            .collect();
        per_dispatch_us.sort_by(f64::total_cmp);
        let median = per_dispatch_us[per_dispatch_us.len() / 2];
        let mean = per_dispatch_us.iter().sum::<f64>() / per_dispatch_us.len() as f64;
        println!(
            "probe entry={} head_dim={} live={} compiled_cached_rows={} splits={splits} chunks={chunks} lanes={} batch={} layers={} repeats={repeats} passes={passes}",
            kernel.entry,
            shape.head_dim,
            shape.live_rows,
            shape.compiled_cached_rows,
            lanes.map_or_else(|| "default".to_string(), |value: usize| value.to_string()),
            batch.map_or_else(|| "default".to_string(), |value: usize| value.to_string()),
            shape.layers,
        );
        println!(
            "probe pipeline max_threads_per_threadgroup={} thread_execution_width={} static_threadgroup_bytes={}",
            pipeline.maxTotalThreadsPerThreadgroup(),
            pipeline.threadExecutionWidth(),
            pipeline.staticThreadgroupMemoryLength(),
        );
        println!(
            "probe parity max_abs_diff={worst:e} max_abs_reference={magnitude:e} relative={:e}",
            worst / magnitude.max(f32::MIN_POSITIVE)
        );
        println!(
            "probe timing per_dispatch_us median={median:.2} mean={mean:.2} min={:.2} max={:.2} all={per_dispatch_us:.2?}",
            per_dispatch_us[0],
            per_dispatch_us[per_dispatch_us.len() - 1]
        );
    }
}
