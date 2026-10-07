//! Times the production tiled `Q4_0` kernel, any number of hand-edited MSL variants of it, and ggml's own
//! `kernel_mul_mm_q4_0_f32` on one shape under one GPU clock, interleaved.
//!
//! Every arm reads the same synthetic `Q4_0` weight blocks and the same activation, and is timed by
//! `MTLCommandBuffer::GPUStartTime`/`GPUEndTime` around one dispatch with nothing else on the device.
//! Variants are whole MSL translation units with the production entry name and signature (dump the
//! production source with `LADDER_DUMP`, edit a copy, pass it back in `LADDER_VARIANTS`), so a kernel
//! change is tried without rebuilding Rust. Each variant prints the number of output words that differ
//! from the production kernel; ggml, which stages activations as `half`, prints its scaled relative error.
//!
//! ggml's source is read at run time from a llama.cpp checkout (`LADDER_GGML_DIR`, the `ggml/src/ggml-metal`
//! directory) and never copied into this repository: local `#include`s are pasted in header order, the
//! six `function_constant` declarations are replaced by the values ggml's host code computes for the shape,
//! and everything after the legacy `kernel_mul_mm` except the `q4_0_f32` instantiation is dropped.
//!
//! ```sh
//! LADDER_ROWS=12288 LADDER_K=1536 LADDER_TOKENS=971 LADDER_VARIANTS=a.metal,b.metal \
//!   cargo run --release -p omega --example mm_kernel_ladder --features metal
//! ```

fn main() -> anyhow::Result<()> {
    #[cfg(all(feature = "metal", feature = "metal-tiled-gemm", target_os = "macos"))]
    return ladder::run();
    #[cfg(not(all(feature = "metal", feature = "metal-tiled-gemm", target_os = "macos")))]
    {
        println!("mm_kernel_ladder requires --features metal on macOS");
        Ok(())
    }
}

#[cfg(all(feature = "metal", feature = "metal-tiled-gemm", target_os = "macos"))]
mod ladder {
    use core::ffi::c_void;
    use core::ptr::NonNull;
    use std::collections::BTreeSet;
    use std::path::PathBuf;

    use anyhow::Context;
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
        MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
        MTLLibrary, MTLMathMode, MTLResourceOptions, MTLSize,
    };
    use proxima_gguf::quant::q4_0::{BLOCK_BYTES, QK4_0, quantize};
    use proxima_tensor::test_support::Lcg;
    use proxima_tensor::{
        DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, Reduce, ReduceInit, ScalarOp,
        append, bind, correct_packed_matmul_layouts, infer, projection,
    };

    type Device = Retained<ProtocolObject<dyn MTLDevice>>;
    type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
    type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
    type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

    const DEFAULT_GGML_DIR: &str = "/Users/brianbruggeman/repos/others/llama.cpp/ggml/src/ggml-metal";

    struct Config {
        rows: usize,
        k: usize,
        tokens: usize,
        rounds: usize,
        warmup: usize,
        variants: Vec<PathBuf>,
        ggml_variants: Vec<PathBuf>,
        ggml_dir: PathBuf,
        dump: Option<PathBuf>,
    }

    fn env_number(name: &str, default: usize) -> anyhow::Result<usize> {
        match std::env::var(name) {
            Ok(text) => text.parse().with_context(|| format!("{name} is a whole number, got {text:?}")),
            Err(_) => Ok(default),
        }
    }

    fn path_list(name: &str) -> Vec<PathBuf> {
        std::env::var(name)
            .map(|list| list.split(',').filter(|part| !part.is_empty()).map(PathBuf::from).collect())
            .unwrap_or_default()
    }

    impl Config {
        fn from_env() -> anyhow::Result<Self> {
            Ok(Self {
                rows: env_number("LADDER_ROWS", 12288)?,
                k: env_number("LADDER_K", 1536)?,
                tokens: env_number("LADDER_TOKENS", 971)?,
                rounds: env_number("LADDER_ROUNDS", 21)?,
                warmup: env_number("LADDER_WARMUP", 30)?,
                variants: path_list("LADDER_VARIANTS"),
                ggml_variants: path_list("LADDER_GGML_VARIANTS"),
                ggml_dir: std::env::var("LADDER_GGML_DIR").map_or_else(|_| PathBuf::from(DEFAULT_GGML_DIR), PathBuf::from),
                dump: std::env::var("LADDER_DUMP").ok().map(PathBuf::from),
            })
        }
    }

    fn matmul_program(config: &Config) -> (Vec<Op>, NodeId, NodeId) {
        let (rows, k, tokens) = (config.rows as u32, config.k as u32, config.tokens as u32);
        let mut program = Vec::new();
        let weight = append(
            &mut program,
            Op::Input { dtype: DType::UInt8, shape: vec![Extent::Static(k), Extent::Static(rows)], name: None },
        );
        let activation = append(
            &mut program,
            Op::Input { dtype: DType::Float32, shape: vec![Extent::Static(tokens), Extent::Static(k)], name: None },
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (weight, IndexMap::Affine(projection(3, &[1, 2]))),
                    (activation, IndexMap::Affine(projection(3, &[0, 1]))),
                ],
                name: None,
            },
        );
        let sum = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(projection(3, &[0, 2])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        (program, weight, sum)
    }

    fn weight_blocks(rows: usize, k: usize) -> anyhow::Result<Vec<u8>> {
        let row_bytes = k / QK4_0 * BLOCK_BYTES;
        let mut lcg = Lcg(11);
        let values: Vec<f32> = (0..rows * k).map(|_| lcg.next_unit() * 2.0 - 1.0).collect();
        let mut blocks = vec![0u8; rows * row_bytes];
        for (row, row_out) in values.chunks_exact(k).zip(blocks.chunks_exact_mut(row_bytes)) {
            quantize(row, row_out).context("k is a whole number of q4_0 blocks")?;
        }
        Ok(blocks)
    }

    fn activation_values(count: usize) -> Vec<f32> {
        let mut lcg = Lcg(97);
        (0..count).map(|_| lcg.next_unit()).collect()
    }

    struct Production {
        source: String,
        entry: String,
        uniforms: Vec<u8>,
        threadgroups: MTLSize,
        threads: MTLSize,
        threadgroup_bytes: usize,
    }

    fn emit_production(config: &Config) -> anyhow::Result<Production> {
        let (program, weight_node, sum_node) = matmul_program(config);
        let policy = NumericPolicy::llama_relaxed();
        let shapes = infer(&program, &[]).context("matmul program infers")?;
        let mut bound_ops = bind(&program, &shapes, &[sum_node], policy).context("matmul program binds")?;
        let packed_nodes: BTreeSet<NodeId> = [weight_node].into_iter().collect();
        correct_packed_matmul_layouts(&mut bound_ops, &packed_nodes);
        let bound = bound_ops
            .into_iter()
            .find(|op| op.node == sum_node)
            .context("the fused reduce is in the bound program")?;
        let packed: omega::PackedOperands = std::collections::BTreeMap::from([(weight_node, omega::Codec::Q4_0)]);
        let kernel = omega::emit(&bound, &packed, policy).context("production kernel emits")?;
        let uniforms = omega::metal::pack_uniforms_for(&bound, policy).context("uniforms pack")?;
        let spec = kernel
            .grid
            .grid2d
            .context("the production tiled kernel takes the threadgroup-grid dispatch")?;
        Ok(Production {
            source: kernel.source,
            entry: kernel.entry,
            uniforms,
            threadgroups: MTLSize { width: spec.threadgroups_x as usize, height: spec.threadgroups_y as usize, depth: 1 },
            threads: MTLSize {
                width: spec.threads_per_threadgroup_x as usize,
                height: spec.threads_per_threadgroup_y as usize,
                depth: 1,
            },
            threadgroup_bytes: spec.threadgroup_bytes as usize,
        })
    }

    fn inline_includes(dirs: &[PathBuf], file: &str, seen: &mut BTreeSet<String>) -> anyhow::Result<String> {
        if !seen.insert(file.to_string()) {
            return Ok(String::new());
        }
        let path = dirs
            .iter()
            .map(|dir| dir.join(file))
            .find(|candidate| candidate.is_file())
            .with_context(|| format!("{file} not found under any of {dirs:?}"))?;
        let text = std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        let mut out = String::with_capacity(text.len());
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed == "#pragma once" {
                continue;
            }
            match trimmed.strip_prefix("#include \"").and_then(|rest| rest.strip_suffix('"')) {
                Some(name) => out.push_str(&inline_includes(dirs, name, seen)?),
                None => {
                    out.push_str(line);
                    out.push('\n');
                }
            }
        }
        Ok(out)
    }

    fn function_constant_literal(name: &str, config: &Config) -> anyhow::Result<&'static str> {
        let boundary_output = !config.rows.is_multiple_of(64) || !config.tokens.is_multiple_of(32);
        let boundary_input = !config.k.is_multiple_of(32);
        match name {
            "FC_mul_mm_bc_inp" => Ok(if boundary_input { "true" } else { "false" }),
            "FC_mul_mm_bc_out" => Ok(if boundary_output { "true" } else { "false" }),
            "FC_mul_mm_ne12" | "FC_mul_mm_ne13" | "FC_mul_mm_r2" | "FC_mul_mm_r3" => Ok("1"),
            "FC_mul_mm_id_amax" => Ok("false"),
            other => anyhow::bail!("no value for function constant {other}"),
        }
    }

    fn bake_function_constants(text: &str, config: &Config) -> anyhow::Result<String> {
        let mut out = String::with_capacity(text.len());
        for line in text.lines() {
            if line.starts_with("constant ") && line.contains("[[function_constant(") {
                let words: Vec<&str> = line.split_whitespace().collect();
                let name = words.get(2).context("function constant declaration has a name")?;
                let literal = function_constant_literal(name, config)?;
                out.push_str(&format!("constant constexpr {} {name} = {literal};\n", words[1]));
            } else {
                out.push_str(line);
                out.push('\n');
            }
        }
        Ok(out)
    }

    fn ggml_source(config: &Config) -> anyhow::Result<String> {
        let dirs = [config.ggml_dir.join("kernels"), config.ggml_dir.clone(), config.ggml_dir.join("..")];
        let mut seen = BTreeSet::new();
        let full = inline_includes(&dirs, "mul_mm.metal", &mut seen)?;
        let legacy_end = full
            .find("#endif // GGML_METAL_HAS_TENSOR\n")
            .context("mul_mm.metal closes its tensor/legacy split")?
            + "#endif // GGML_METAL_HAS_TENSOR\n".len();
        let typedef = full
            .lines()
            .find(|line| line.starts_with("typedef decltype(kernel_mul_mm<") && line.ends_with("mul_mm_t;"))
            .context("mul_mm_t typedef present")?;
        let instantiation = full
            .lines()
            .find(|line| line.contains("host_name(\"kernel_mul_mm_q4_0_f32\")"))
            .context("q4_0_f32 instantiation present")?;
        let trimmed = format!("{}\n{typedef}\n{instantiation}\n", &full[..legacy_end]);
        let prelude = "#define GGML_COMMON_DECL_METAL\n#define GGML_COMMON_IMPL_METAL\n";
        Ok(format!("{prelude}{}", bake_function_constants(&trimmed, config)?))
    }

    fn compile(device: &Device, source: &str, entry: &str) -> anyhow::Result<Pipeline> {
        let options = MTLCompileOptions::new();
        options.setMathMode(MTLMathMode::Relaxed);
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
            .map_err(|error| anyhow::anyhow!("compile {entry}: {}", error.localizedDescription()))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(entry))
            .with_context(|| format!("{entry} present in its library"))?;
        device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| anyhow::anyhow!("pipeline {entry}: {}", error.localizedDescription()))
    }

    fn buffer_from_bytes(device: &Device, bytes: &[u8]) -> anyhow::Result<Buffer> {
        // SAFETY: `bytes` is live for the call and the API copies it before returning.
        let pointer = unsafe { NonNull::new_unchecked(bytes.as_ptr().cast_mut().cast::<c_void>()) };
        unsafe { device.newBufferWithBytes_length_options(pointer, bytes.len(), MTLResourceOptions::StorageModeShared) }
            .context("shared buffer allocates")
    }

    fn read_f32(buffer: &Buffer, count: usize) -> Vec<f32> {
        let pointer = buffer.contents().as_ptr().cast::<f32>();
        // SAFETY: the buffer holds at least `count` floats and the command buffer completed.
        unsafe { std::slice::from_raw_parts(pointer, count) }.to_vec()
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct GgmlKargs {
        ne00: i32,
        ne02: i32,
        nb01: u64,
        nb02: u64,
        nb03: u64,
        ne12: i32,
        nb10: u64,
        nb11: u64,
        nb12: u64,
        nb13: u64,
        ne0: i32,
        ne1: i32,
        r2: i16,
        r3: i16,
    }

    enum Launch {
        Proxima { threadgroups: MTLSize, threads: MTLSize, dynamic_bytes: usize },
        Ggml { kargs: GgmlKargs, threadgroups: MTLSize },
    }

    struct Arm {
        label: String,
        pipeline: Pipeline,
        output: Buffer,
        launch: Launch,
    }

    struct Inputs {
        weight: Buffer,
        activation: Buffer,
        uniforms: Buffer,
    }

    fn dispatch_ns(queue: &Queue, arm: &Arm, inputs: &Inputs) -> anyhow::Result<f64> {
        let command_buffer = queue.commandBuffer().context("command buffer")?;
        let encoder = command_buffer.computeCommandEncoder().context("compute encoder")?;
        encoder.setComputePipelineState(&arm.pipeline);
        match &arm.launch {
            Launch::Proxima { threadgroups, threads, dynamic_bytes } => {
                // SAFETY: buffers outlive the command buffer, which is waited on below.
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&inputs.weight), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&inputs.activation), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(&arm.output), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(&inputs.uniforms), 0, 3);
                }
                if *dynamic_bytes > 0 {
                    // SAFETY: index 0 is the kernel's `[[threadgroup(0)]]` parameter.
                    unsafe { encoder.setThreadgroupMemoryLength_atIndex(*dynamic_bytes, 0) };
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(*threadgroups, *threads);
            }
            Launch::Ggml { kargs, threadgroups } => {
                // SAFETY: `kargs` is copied by `setBytes`; buffers outlive the waited-on command buffer.
                unsafe {
                    let pointer = NonNull::new_unchecked((kargs as *const GgmlKargs).cast_mut().cast::<c_void>());
                    encoder.setBytes_length_atIndex(pointer, size_of::<GgmlKargs>(), 0);
                    encoder.setBuffer_offset_atIndex(Some(&inputs.weight), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(&inputs.activation), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(&arm.output), 0, 3);
                    encoder.setThreadgroupMemoryLength_atIndex(8192, 0);
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(*threadgroups, MTLSize { width: 32, height: 4, depth: 1 });
            }
        }
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        Ok((command_buffer.GPUEndTime() - command_buffer.GPUStartTime()).max(0.0) * 1e9)
    }

    fn percentile(sorted: &[f64], percent: usize) -> f64 {
        sorted[(sorted.len() - 1) * percent / 100]
    }

    fn report(config: &Config, arms: &[Arm], samples: &[Vec<f64>], reference: usize) {
        let flops = 2.0 * config.rows as f64 * config.k as f64 * config.tokens as f64;
        let reference_p25 = {
            let mut sorted = samples[reference].clone();
            sorted.sort_by(f64::total_cmp);
            percentile(&sorted, 25)
        };
        for (arm, arm_samples) in arms.iter().zip(samples) {
            let mut sorted = arm_samples.clone();
            sorted.sort_by(f64::total_cmp);
            let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
            let variance = sorted.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / sorted.len() as f64;
            let p25 = percentile(&sorted, 25);
            println!(
                "TIME arm={} p25_us={:.1} median_us={:.1} min_us={:.1} cov_pct={:.2} tflops_p25={:.2} vs_ref_p25={:.4} n={}",
                arm.label,
                p25 / 1e3,
                percentile(&sorted, 50) / 1e3,
                sorted[0] / 1e3,
                variance.sqrt() / mean * 100.0,
                flops / p25 / 1e3,
                p25 / reference_p25,
                sorted.len(),
            );
        }
    }

    fn compare(label: &str, candidate: &[f32], reference: &[f32]) {
        let differing = candidate.iter().zip(reference).filter(|(left, right)| left.to_bits() != right.to_bits()).count();
        let square_error: f64 = candidate.iter().zip(reference).map(|(left, right)| f64::from(left - right).powi(2)).sum();
        let square_reference: f64 = reference.iter().map(|value| f64::from(*value).powi(2)).sum();
        println!(
            "CHECK arm={label} differing_words={differing}/{} rms_relative_error={:.3e}",
            reference.len(),
            (square_error / square_reference.max(f64::MIN_POSITIVE)).sqrt()
        );
    }

    fn build_arms(config: &Config, device: &Device, production: &Production) -> anyhow::Result<Vec<Arm>> {
        let output_bytes = config.rows * config.tokens * 4;
        let zeroed = |count: usize| buffer_from_bytes(device, &vec![0u8; count]);
        let proxima_launch = |dynamic_bytes: usize| Launch::Proxima {
            threadgroups: production.threadgroups,
            threads: production.threads,
            dynamic_bytes,
        };
        let variant_bytes = |source: &str| if source.contains("[[threadgroup(0)]]") { 8192 } else { 0 };
        let mut arms = vec![Arm {
            label: "prod".into(),
            pipeline: compile(device, &production.source, &production.entry)?,
            output: zeroed(output_bytes)?,
            launch: proxima_launch(production.threadgroup_bytes),
        }];
        for path in &config.variants {
            let source = std::fs::read_to_string(path).with_context(|| format!("read variant {}", path.display()))?;
            let label = path.file_stem().map_or_else(|| "variant".into(), |stem| stem.to_string_lossy().into_owned());
            arms.push(Arm {
                label,
                pipeline: compile(device, &source, &production.entry)?,
                output: zeroed(output_bytes)?,
                launch: proxima_launch(variant_bytes(&source)),
            });
        }
        let row_bytes = (config.k / QK4_0 * BLOCK_BYTES) as u64;
        let tokens = config.tokens as u64;
        let kargs = GgmlKargs {
            ne00: config.k as i32,
            ne02: 1,
            nb01: row_bytes,
            nb02: row_bytes * config.rows as u64,
            nb03: row_bytes * config.rows as u64,
            ne12: 1,
            nb10: 4,
            nb11: 4 * config.k as u64,
            nb12: 4 * config.k as u64 * tokens,
            nb13: 4 * config.k as u64 * tokens,
            ne0: config.rows as i32,
            ne1: config.tokens as i32,
            r2: 1,
            r3: 1,
        };
        let threadgroups = MTLSize { width: config.tokens.div_ceil(32), height: config.rows.div_ceil(64), depth: 1 };
        let ggml = ggml_source(config)?;
        if let Some(dump) = &config.dump {
            std::fs::write(dump.with_extension("ggml.metal"), &ggml).context("write ggml source dump")?;
        }
        for path in &config.ggml_variants {
            let source = std::fs::read_to_string(path).with_context(|| format!("read ggml variant {}", path.display()))?;
            let stem = path.file_stem().map_or_else(|| "variant".into(), |stem| stem.to_string_lossy().into_owned());
            arms.push(Arm {
                label: format!("g_{stem}"),
                pipeline: compile(device, &source, "kernel_mul_mm_q4_0_f32")?,
                output: zeroed(output_bytes)?,
                launch: Launch::Ggml { kargs, threadgroups },
            });
        }
        arms.push(Arm {
            label: "ggml".into(),
            pipeline: compile(device, &ggml, "kernel_mul_mm_q4_0_f32")?,
            output: zeroed(output_bytes)?,
            launch: Launch::Ggml { kargs, threadgroups },
        });
        Ok(arms)
    }

    fn run_rounds(config: &Config, queue: &Queue, arms: &[Arm], inputs: &Inputs) -> anyhow::Result<Vec<Vec<f64>>> {
        let mut samples = vec![Vec::with_capacity(config.rounds); arms.len()];
        for round in 0..config.warmup + config.rounds {
            for offset in 0..arms.len() {
                let index = (offset + round) % arms.len();
                let nanoseconds = dispatch_ns(queue, &arms[index], inputs)?;
                if round >= config.warmup {
                    samples[index].push(nanoseconds);
                }
            }
        }
        Ok(samples)
    }

    pub fn run() -> anyhow::Result<()> {
        let config = Config::from_env()?;
        let production = emit_production(&config)?;
        if let Some(dump) = &config.dump {
            std::fs::write(dump, &production.source).with_context(|| format!("write {}", dump.display()))?;
        }
        let device = MTLCreateSystemDefaultDevice().context("a Metal device")?;
        let queue = device.newCommandQueue().context("command queue")?;
        let weight_bytes = weight_blocks(config.rows, config.k)?;
        let activation: Vec<u8> = activation_values(config.tokens * config.k)
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let inputs = Inputs {
            weight: buffer_from_bytes(&device, &weight_bytes)?,
            activation: buffer_from_bytes(&device, &activation)?,
            uniforms: buffer_from_bytes(&device, &production.uniforms)?,
        };
        let arms = build_arms(&config, &device, &production)?;
        for arm in &arms {
            println!(
                "FACTS arm={} max_threads_per_threadgroup={} static_threadgroup_bytes={}",
                arm.label,
                arm.pipeline.maxTotalThreadsPerThreadgroup(),
                arm.pipeline.staticThreadgroupMemoryLength()
            );
            dispatch_ns(&queue, arm, &inputs)?;
        }
        let count = config.rows * config.tokens;
        let reference = read_f32(&arms[0].output, count);
        for arm in &arms[1..] {
            compare(&arm.label, &read_f32(&arm.output, count), &reference);
        }
        let samples = run_rounds(&config, &queue, &arms, &inputs)?;
        println!("SHAPE rows={} k={} tokens={} rounds={}", config.rows, config.k, config.tokens, config.rounds);
        report(&config, &arms, &samples, arms.len() - 1);
        Ok(())
    }
}
