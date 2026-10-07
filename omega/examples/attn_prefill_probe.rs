//! Per-dispatch GPU time and correctness of one two-range cached-attention
//! kernel at a prefill shape (`PROBE_ROWS` new query rows over a bucketed cache
//! of `PROBE_CACHED_ROWS` rows holding `PROBE_LIVE`), so a kernel variant is
//! measured in seconds instead of through a model load.
//!
//! The kernel is the one `omega::emit` produces for the bound op, or, with
//! `PROBE_KERNEL_FILE`, the Metal source in that file under the same entry
//! name, which is how a variant is tried without rebuilding omega. The launch
//! is read off the emitted grid; `PROBE_THREADGROUPS` and `PROBE_THREADS`
//! override it for a variant that maps work differently.
//!
//! Correctness compares sampled query rows (all heads) with a direct softmax
//! attention in f64 over the same buffers, with the mask the unfused program
//! builds: a cached key `j` is visible to row `r` when `j - live - r >=
//! lower`, a new key `j` when `lower <= j - r <= upper`.
//!
//! ```sh
//! PROBE_HEAD_DIM=512 PROBE_ROWS=971 PROBE_CACHED_ROWS=992 \
//!     cargo run -p omega --release --example attn_prefill_probe
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)]

fn main() {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    imp::run();
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    println!("attn_prefill_probe requires --features metal on macOS");
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod imp {
    use std::collections::BTreeMap;

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
        kv_heads: usize,
        groups: usize,
        rows: usize,
        cached_rows: usize,
        live: usize,
        lower: i64,
        upper: i64,
        layers: usize,
        scale: f32,
    }

    impl Shape {
        fn from_env() -> Self {
            let head_dim = env_usize("PROBE_HEAD_DIM", 512);
            let window = env_usize("PROBE_WINDOW", 0);
            Self {
                head_dim,
                kv_heads: env_usize("PROBE_KV_HEADS", 1),
                groups: env_usize("PROBE_GROUPS", 8),
                rows: env_usize("PROBE_ROWS", 971),
                cached_rows: env_usize("PROBE_CACHED_ROWS", 992),
                live: env_usize("PROBE_LIVE", 0),
                lower: if window == 0 { i64::MIN } else { 1 - window as i64 },
                upper: env_usize("PROBE_UPPER", 0) as i64,
                layers: env_usize("PROBE_LAYERS", 7),
                scale: std::env::var("PROBE_SCALE").map_or(1.0, |value| value.parse().unwrap()),
            }
        }

        fn heads(&self) -> usize {
            self.kv_heads * self.groups
        }

        fn op(&self) -> BoundOp {
            let layout = Layout {
                base: 0,
                strides: vec![1_i64].into(),
            };
            BoundOp {
                node: NodeId(9),
                dtype: DType::Float32,
                extents: vec![
                    self.rows as u64,
                    self.kv_heads as u64,
                    self.groups as u64,
                    self.head_dim as u64,
                ],
                kind: BoundOpKind::CachedAttention {
                    operands: (0..9)
                        .map(|index| (NodeId(index), layout.clone(), None))
                        .collect(),
                    query_rows: self.rows as u64,
                    cached_key_rows: self.cached_rows as u64,
                    new_key_rows: self.rows as u64,
                    kv_heads: self.kv_heads as u64,
                    query_groups: self.groups as u64,
                    head_dim: self.head_dim as u64,
                    rotary_dim: self.head_dim as u64,
                    scale: self.scale,
                    cached_lower_inclusive: self.lower,
                    new_upper_inclusive: self.upper,
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
        let cached = shape.cached_rows * shape.kv_heads;
        let fresh = shape.rows * shape.kv_heads;
        CacheSet {
            key_real: lcg.vector(cached * plane, 0.5),
            key_imag: lcg.vector(cached * plane, 0.5),
            value: lcg.vector(cached * shape.head_dim, 1.0),
            new_key_real: lcg.vector(fresh * plane, 0.5),
            new_key_imag: lcg.vector(fresh * plane, 0.5),
            new_value: lcg.vector(fresh * shape.head_dim, 1.0),
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
        let scratch = f32_buffer(device, &vec![f32::NAN; scratch_floats]);
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
        row: usize,
        head: usize,
    ) -> Vec<f64> {
        let plane = shape.head_dim / 2;
        let kv_head = head / shape.groups;
        let query_offset = (row * shape.heads() + head) * plane;
        let dot = |real: &[f32], imag: &[f32], base: usize| -> f64 {
            (0..plane)
                .map(|pair| {
                    f64::from(real[base + pair]) * f64::from(query[0][query_offset + pair])
                        + f64::from(imag[base + pair]) * f64::from(query[1][query_offset + pair])
                })
                .sum::<f64>()
                * f64::from(shape.scale)
        };
        let mut keys: Vec<(f64, &[f32])> = Vec::new();
        for key in 0..shape.live {
            let relative = key as i64 - shape.live as i64 - row as i64;
            if relative >= shape.lower {
                let base = (key * shape.kv_heads + kv_head) * plane;
                let value_base = (key * shape.kv_heads + kv_head) * shape.head_dim;
                keys.push((
                    dot(&cache.key_real, &cache.key_imag, base),
                    &cache.value[value_base..value_base + shape.head_dim],
                ));
            }
        }
        for key in 0..shape.rows {
            let relative = key as i64 - row as i64;
            if relative >= shape.lower && relative <= shape.upper {
                let base = (key * shape.kv_heads + kv_head) * plane;
                let value_base = (key * shape.kv_heads + kv_head) * shape.head_dim;
                keys.push((
                    dot(&cache.new_key_real, &cache.new_key_imag, base),
                    &cache.new_value[value_base..value_base + shape.head_dim],
                ));
            }
        }
        let maximum = keys.iter().map(|(score, _)| *score).fold(f64::NEG_INFINITY, f64::max);
        let total: f64 = keys.iter().map(|(score, _)| (score - maximum).exp()).sum();
        (0..shape.head_dim)
            .map(|dim| {
                keys.iter()
                    .map(|(score, value)| (score - maximum).exp() * f64::from(value[dim]))
                    .sum::<f64>()
                    / total
            })
            .collect()
    }

    fn merged(
        out: &[f32],
        shape: &Shape,
        splits: usize,
        row: usize,
        head: usize,
    ) -> Vec<f64> {
        let vectors = shape.rows * shape.heads();
        let query_index = row * shape.heads() + head;
        if splits == 1 {
            return out[query_index * shape.head_dim..(query_index + 1) * shape.head_dim]
                .iter()
                .map(|value| f64::from(*value))
                .collect();
        }
        let stats_base = vectors * shape.head_dim * splits;
        let stats = |split: usize| {
            let at = stats_base + (query_index * splits + split) * 2;
            (f64::from(out[at]), f64::from(out[at + 1]))
        };
        let global_max = (0..splits).map(|split| stats(split).0).fold(f64::NEG_INFINITY, f64::max);
        let weight = |split: usize| {
            let maximum = stats(split).0;
            if maximum == f64::NEG_INFINITY {
                0.0
            } else {
                (maximum - global_max).exp()
            }
        };
        let total: f64 = (0..splits).map(|split| stats(split).1 * weight(split)).sum();
        (0..shape.head_dim)
            .map(|dim| {
                let at = |split: usize| {
                    (((query_index * (shape.head_dim / 4) + dim / 4) * splits + split) * 4)
                        + dim % 4
                };
                (0..splits)
                    .map(|split| f64::from(out[at(split)]) * weight(split))
                    .sum::<f64>()
                    / total
            })
            .collect()
    }

    fn entry_number(entry: &str, marker: &str) -> Option<usize> {
        let start = entry.find(marker)? + marker.len();
        let digits: String = entry[start..].chars().take_while(char::is_ascii_digit).collect();
        digits.parse().ok()
    }

    type Settings = BTreeMap<String, String>;

    struct ArmSpec {
        name: String,
        template: Option<String>,
        settings: Settings,
    }

    fn setting(settings: &Settings, key: &str, default: usize) -> usize {
        settings.get(key).map_or(default, |value| {
            value
                .parse()
                .unwrap_or_else(|error| panic!("{key}={value} is not an integer: {error}"))
        })
    }

    fn parse_arms() -> Vec<ArmSpec> {
        let Ok(arms) = std::env::var("PROBE_ARMS") else {
            let settings = std::env::vars()
                .filter_map(|(name, value)| {
                    name.strip_prefix("PROBE_DEFINE_").map(|key| (key.to_string(), value))
                })
                .chain(
                    [
                        ("TILE_ROWS", "PROBE_TILE_ROWS"),
                        ("SIMDGROUPS", "PROBE_SIMDGROUPS"),
                        ("BLOCK", "PROBE_BLOCK"),
                        ("SPLITS", "PROBE_SPLITS"),
                        ("THREADGROUPS", "PROBE_THREADGROUPS"),
                        ("THREADS", "PROBE_THREADS"),
                    ]
                    .into_iter()
                    .filter_map(|(key, name)| {
                        std::env::var(name).ok().map(|value| (key.to_string(), value))
                    }),
                )
                .collect();
            return vec![ArmSpec {
                name: "env".to_string(),
                template: std::env::var("PROBE_TEMPLATE").ok(),
                settings,
            }];
        };
        arms.split(';')
            .filter(|arm| !arm.trim().is_empty())
            .map(|arm| {
                let mut parts = arm.trim().split(',');
                let head = parts.next().expect("an arm names itself");
                let (name, template) = head
                    .split_once('=')
                    .unwrap_or_else(|| panic!("arm `{head}` is not name=template"));
                let settings = parts
                    .map(|pair| {
                        let (key, value) = pair
                            .split_once('=')
                            .unwrap_or_else(|| panic!("setting `{pair}` is not KEY=VALUE"));
                        (key.to_string(), value.to_string())
                    })
                    .collect();
                ArmSpec {
                    name: name.to_string(),
                    template: (template != "production").then(|| template.to_string()),
                    settings,
                }
            })
            .collect()
    }

    fn template_source(path: &str, shape: &Shape, settings: &Settings) -> String {
        let body = std::fs::read_to_string(path).unwrap_or_else(|error| panic!("reads {path}: {error}"));
        let lower = if shape.lower == i64::MIN {
            "-9223372036854775807L".to_string()
        } else {
            format!("{}L", shape.lower)
        };
        let mut substitutions: Settings = [
            ("ENTRY", "probe_attention".to_string()),
            ("KV_HEADS", shape.kv_heads.to_string()),
            ("QUERY_GROUPS", shape.groups.to_string()),
            ("HEAD_DIM", shape.head_dim.to_string()),
            ("SCALE", format!("{:?}", shape.scale)),
            ("CACHED_LOWER", lower),
            ("NEW_UPPER", format!("{}L", shape.upper)),
            ("TILE_ROWS", "1".to_string()),
            ("SIMDGROUPS", "8".to_string()),
            ("BLOCK", "64".to_string()),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect();
        if let Ok(defaults) = std::env::var("PROBE_DEFAULTS") {
            substitutions.extend(defaults.split(',').filter_map(|pair| {
                pair.split_once('=').map(|(key, value)| (key.to_string(), value.to_string()))
            }));
        }
        substitutions.extend(settings.clone());
        let mut text = body;
        for (token, value) in &substitutions {
            text = text.replace(&format!("@{token}@"), value);
        }
        format!("#include <metal_stdlib>\nusing namespace metal;\n\n{text}")
    }

    struct Arm {
        name: String,
        entry: String,
        pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        uniforms: Retained<ProtocolObject<dyn MTLBuffer>>,
        threadgroups: usize,
        threads_per_threadgroup: usize,
        splits: usize,
    }

    fn build_arm(
        device: &ProtocolObject<dyn MTLDevice>,
        shape: &Shape,
        spec: &ArmSpec,
        production: &omega::Kernel,
    ) -> Arm {
        let vectors = shape.rows * shape.heads();
        let (source, entry, default_width, default_groups, default_splits, row_tiled) =
            match &spec.template {
                Some(path) => {
                    let tile_rows = setting(&spec.settings, "TILE_ROWS", 1);
                    let simdgroups = setting(&spec.settings, "SIMDGROUPS", 8);
                    let splits = setting(&spec.settings, "SPLITS", 1);
                    let tiles = shape.rows.div_ceil(tile_rows);
                    (
                        template_source(path, shape, &spec.settings),
                        "probe_attention".to_string(),
                        simdgroups * 32,
                        shape.kv_heads * tiles * splits,
                        splits,
                        true,
                    )
                }
                None => {
                    let width = production
                        .grid
                        .threadgroup_width
                        .or(production.grid.grid2d.map(|grid| grid.threads_per_threadgroup_x))
                        .unwrap_or(SIMD_WIDTH) as usize;
                    let groups = (production.grid.threads as usize) / width;
                    let row_tiled = production.entry.ends_with("_rt");
                    let splits = match (row_tiled, entry_number(&production.entry, "_r")) {
                        (true, Some(tile_rows)) => {
                            (groups / (shape.kv_heads * shape.rows.div_ceil(tile_rows))).max(1)
                        }
                        _ => 1,
                    };
                    (
                        production.source.clone(),
                        production.entry.clone(),
                        width,
                        groups,
                        splits,
                        row_tiled,
                    )
                }
            };
        let threads_per_threadgroup = setting(&spec.settings, "THREADS", default_width);
        let threadgroups = setting(&spec.settings, "THREADGROUPS", default_groups);
        let splits = setting(&spec.settings, "SPLITS", default_splits);
        let uniform_total = if row_tiled {
            vectors
        } else {
            (production.grid.threads / SIMD_WIDTH) as usize
        };
        let uniform_bytes: Vec<u8> = [uniform_total as i64, splits as i64]
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        if let Ok(path) = std::env::var("PROBE_DUMP") {
            std::fs::write(format!("{path}.{}", spec.name), &source)
                .unwrap_or_else(|error| panic!("writes {path}: {error}"));
        }
        Arm {
            name: spec.name.clone(),
            pipeline: compile(device, &source, &entry),
            entry,
            uniforms: shared_buffer(device, &uniform_bytes),
            threadgroups,
            threads_per_threadgroup,
            splits,
        }
    }

    fn parity(
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        arm: &Arm,
        shape: &Shape,
        layer: &LayerBuffers,
        out_floats: usize,
        query: &[Vec<f32>; 2],
        cache: &CacheSet,
    ) -> (f64, f64, usize) {
        let launch = Launch {
            pipeline: &arm.pipeline,
            uniforms: &arm.uniforms,
            threadgroups: arm.threadgroups,
            threads_per_threadgroup: arm.threads_per_threadgroup,
        };
        let command_buffer = queue.commandBuffer().expect("command buffer");
        let encoder = command_buffer.computeCommandEncoder().expect("compute encoder");
        encode(&encoder, &launch, layer);
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        let out = read_f32(&layer.scratch, out_floats);
        let sampled_rows: Vec<usize> = [
            0,
            1,
            7,
            8,
            63,
            64,
            255,
            511,
            512,
            513,
            shape.rows / 2,
            shape.rows - 2,
            shape.rows - 1,
        ]
        .into_iter()
        .filter(|row| *row < shape.rows)
        .collect();
        let mut worst = 0.0_f64;
        let mut magnitude = 0.0_f64;
        let mut non_finite = 0usize;
        for row in &sampled_rows {
            for head in 0..shape.heads() {
                let want = attention_reference(query, cache, shape, *row, head);
                let got = merged(&out, shape, arm.splits, *row, head);
                for (expected, actual) in want.iter().zip(&got) {
                    if actual.is_finite() {
                        worst = worst.max((expected - actual).abs());
                        magnitude = magnitude.max(expected.abs());
                    } else {
                        non_finite += 1;
                    }
                }
            }
        }
        (worst, magnitude, non_finite)
    }

    fn median(sorted: &[f64]) -> f64 {
        sorted[sorted.len() / 2]
    }

    pub fn run() {
        let shape = Shape::from_env();
        let production = omega::emit(
            &shape.op(),
            &PackedOperands::new(),
            NumericPolicy::llama_relaxed(),
        )
        .expect("the attention kernel emits");
        let device = MTLCreateSystemDefaultDevice().expect("a metal device");
        let queue = device.newCommandQueue().expect("a command queue");
        let arms: Vec<Arm> = parse_arms()
            .iter()
            .map(|spec| build_arm(&device, &shape, spec, &production))
            .filter(|arm| {
                let fits = arm.threads_per_threadgroup <= arm.pipeline.maxTotalThreadsPerThreadgroup();
                if !fits {
                    println!(
                        "probe arm={} skipped width={} max={}",
                        arm.name,
                        arm.threads_per_threadgroup,
                        arm.pipeline.maxTotalThreadsPerThreadgroup()
                    );
                }
                fits
            })
            .collect();

        let mut lcg = Lcg(0x5eed_0001);
        let plane = shape.head_dim / 2;
        let vectors = shape.rows * shape.heads();
        let query = [lcg.vector(vectors * plane, 0.5), lcg.vector(vectors * plane, 0.5)];
        let max_splits = arms.iter().map(|arm| arm.splits).max().unwrap_or(1);
        let out_floats = if max_splits > 1 {
            vectors * shape.head_dim * max_splits + vectors * max_splits * 2
        } else {
            vectors * shape.head_dim
        };
        let caches: Vec<CacheSet> = (0..shape.layers).map(|_| cache_set(&mut lcg, &shape)).collect();
        let layers: Vec<LayerBuffers> = caches
            .iter()
            .map(|cache| layer_buffers(&device, &query, cache, shape.live, out_floats))
            .collect();

        println!(
            "probe shape rows={} cached_rows={} live={} kv_heads={} groups={} head_dim={} lower={} layers={}",
            shape.rows,
            shape.cached_rows,
            shape.live,
            shape.kv_heads,
            shape.groups,
            shape.head_dim,
            shape.lower,
            shape.layers,
        );
        for arm in &arms {
            let (worst, magnitude, non_finite) =
                parity(&queue, arm, &shape, &layers[0], out_floats, &query, &caches[0]);
            println!(
                "probe arm={} entry={} threadgroups={} threads={} splits={} static_threadgroup_bytes={} parity_max_abs_diff={worst:e} max_abs_reference={magnitude:e} non_finite={non_finite}",
                arm.name,
                arm.entry,
                arm.threadgroups,
                arm.threads_per_threadgroup,
                arm.splits,
                arm.pipeline.staticThreadgroupMemoryLength(),
            );
        }

        let repeats = env_usize("PROBE_REPEATS", 3 * shape.layers);
        let rounds = env_usize("PROBE_ROUNDS", 9);
        let launches: Vec<Launch<'_>> = arms
            .iter()
            .map(|arm| Launch {
                pipeline: &arm.pipeline,
                uniforms: &arm.uniforms,
                threadgroups: arm.threadgroups,
                threads_per_threadgroup: arm.threads_per_threadgroup,
            })
            .collect();
        for launch in &launches {
            let _ = timed_pass(&queue, launch, &layers, repeats);
        }
        let mut samples: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
        for round in 0..rounds {
            for offset in 0..arms.len() {
                let index = (round + offset) % arms.len();
                samples[index].push(
                    timed_pass(&queue, &launches[index], &layers, repeats) / repeats as f64 / 1e3,
                );
            }
        }
        let first = samples[0].clone();
        for (arm, values) in arms.iter().zip(&samples) {
            let mut sorted = values.clone();
            sorted.sort_by(f64::total_cmp);
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance =
                values.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / values.len() as f64;
            let mut ratios: Vec<f64> = values.iter().zip(&first).map(|(value, base)| value / base).collect();
            ratios.sort_by(f64::total_cmp);
            println!(
                "probe timing arm={} median_us={:.1} min_us={:.1} max_us={:.1} cov_pct={:.1} paired_ratio_vs_first={:.3} rounds={rounds} repeats={repeats}",
                arm.name,
                median(&sorted),
                sorted[0],
                sorted[sorted.len() - 1],
                100.0 * variance.sqrt() / mean,
                median(&ratios),
            );
        }
    }
}
