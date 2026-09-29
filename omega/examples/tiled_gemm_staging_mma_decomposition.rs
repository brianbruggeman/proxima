//! Ablation decomposition of proxima's tiled Q4_0 GEMM (`[K=1536,M=12288]x
//! [N=510]`, the same shape `docs/model-interop/discipline.md` ROW C4.11
//! measured at 7.68ms GPU time via `execute_plan_op_timed`). Standalone
//! probe: no production
//! code changes. Every arm dispatches a TEXTUAL COPY of the exact source
//! `omega::emit` renders today (never a hand-retyped kernel), compiled and
//! dispatched by hand through `objc2_metal` with the SAME uniforms buffer
//! (`omega::metal::pack_uniforms_for`) and the SAME grid `omega::emit`
//! reports, so arm A is byte-for-byte production's own kernel on production's
//! own dispatch geometry.
//!
//! Arms:
//! - A: production kernel, unmodified emitted source.
//! - B: A with the MMA loop's `simdgroup_multiply_accumulate` calls removed
//!   (staging + store only). A single dependent write
//!   (`acc[0] = make_filled_simdgroup_matrix<float,8>(weight_tile[0] +
//!   act_tile[0])`) keeps `weight_tile`/`act_tile` live so the driver
//!   compiler cannot DCE the staging loops feeding a value the final store
//!   never reads.
//! - C: A with the weight (dequant) staging loop replaced by a constant
//!   `half` fill -- activation staging + MMA + store unchanged.
//! - D: A with the activation staging loop replaced by a constant `float`
//!   fill -- weight staging + MMA + store unchanged.
//! - E: A with BOTH staging loops removed from the per-K-substep loop body
//!   (tile memory constant-filled once, outside the loop) -- MMA + store
//!   only.
//!
//! Real Q4_0 weight bytes (`blk.0.ffn_gate.weight`, gemma4-E2B,
//! `in_dim=1536 -> out_dim=12288`, guiding-principles §9); synthetic
//! deterministic activation bytes (matches this repo's own
//! `q4_0_tiled_gemm_run8_speed_probe.rs` convention -- correctness against
//! real activations is a decode-time property, not this shape probe's job).
//!
//! GPU clock only (`MTLCommandBuffer::GPUStartTime`/`GPUEndTime`), never
//! host wall time -- the same clock `omega::metal::execute_plan_op_timed`
//! itself reports, so arm numbers are directly comparable to
//! `docs/model-interop/discipline.md` ROW C4.11's own 7.68ms figure.

fn main() -> anyhow::Result<()> {
    #[cfg(all(feature = "metal", feature = "metal-tiled-gemm", target_os = "macos"))]
    return run();
    #[cfg(not(all(feature = "metal", feature = "metal-tiled-gemm", target_os = "macos")))]
    {
        println!("tiled_gemm_staging_mma_decomposition requires --features metal,metal-tiled-gemm on macOS");
        Ok(())
    }
}

#[cfg(all(feature = "metal", feature = "metal-tiled-gemm", target_os = "macos"))]
fn run() -> anyhow::Result<()> {
    use anyhow::Context;
    use core::ffi::c_void;
    use core::ptr::NonNull;
    use std::collections::BTreeSet;
    use std::io::{Read, Seek, SeekFrom};

    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
        MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
        MTLLibrary, MTLMathMode, MTLResourceOptions, MTLSize,
    };

    use proxima_gguf::parser::{GgufEvent, GgufParser};
    use proxima_gguf::pipe::ParsedGguf;
    use proxima_gguf::types::GgmlType;
    use proxima_tensor::{
        DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce,
        ReduceInit, ScalarOp, append, bind, correct_packed_matmul_layouts, infer, projection,
    };

    const REAL_GEMMA4_GGUF_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";
    const TOKENS: u32 = 510;
    const EMBEDDING: u32 = 1536;
    const FEED_FORWARD: u32 = 6144;
    const ITERS: usize = 16;
    const WARMUP: usize = 4;

    // -- real-GGUF plumbing, restated per-binary (`omega/tests/*.rs` doc
    // convention: each integration-test/example crate restates its own GGUF
    // reader rather than sharing one across separately-compiled binaries) --

    fn real_gguf_header(path: &std::path::Path) -> anyhow::Result<Option<(ParsedGguf, u64, std::fs::File)>> {
        let Ok(mut file) = std::fs::File::open(path) else {
            return Ok(None);
        };
        let Ok(metadata) = file.metadata() else {
            return Ok(None);
        };
        let file_len = metadata.len();
        let mut prefix_len = 1usize << 20;
        loop {
            let mut buf = vec![0u8; prefix_len];
            file.seek(SeekFrom::Start(0)).context("seek to start")?;
            let read = file.read(&mut buf).context("read gguf prefix")?;
            buf.truncate(read);
            if let Ok((parser, events)) = GgufParser::new().push(&buf) {
                let mut version = None;
                let mut metadata = Vec::new();
                let mut tensors = Vec::new();
                let mut completion = None;
                for event in events {
                    match event {
                        GgufEvent::Header { version: value, .. } => version = Some(value),
                        GgufEvent::Metadata { key, value } => metadata.push((key, value)),
                        GgufEvent::Tensor(tensor) => tensors.push(tensor),
                        GgufEvent::Complete { data_offset, alignment } => {
                            completion = Some((data_offset, alignment));
                        }
                    }
                }
                if let (Some(version), Some((data_offset, alignment))) = (version, completion) {
                    parser.finish().context("parser reports complete and clean")?;
                    return Ok(Some((
                        ParsedGguf {
                            version,
                            tensor_count: tensors.len() as u64,
                            kv_count: metadata.len() as u64,
                            metadata,
                            tensors,
                            data_offset,
                            alignment,
                        },
                        file_len,
                        file,
                    )));
                }
            }
            if prefix_len as u64 >= file_len {
                return Ok(None);
            }
            prefix_len *= 2;
        }
    }

    fn real_tensor_bytes(
        file: &mut std::fs::File,
        parsed: &ParsedGguf,
        file_len: u64,
        name: &str,
    ) -> anyhow::Result<Vec<u8>> {
        let tensor = parsed
            .tensors
            .iter()
            .find(|candidate| candidate.name == name)
            .with_context(|| format!("{name} present in the real gemma4-E2B checkpoint"))?;
        eprintln!("real_tensor_bytes name={name} dims={:?} ggml_type={:?}", tensor.dims, tensor.ggml_type);
        anyhow::ensure!(tensor.ggml_type == GgmlType::Q4_0, "{name} is Q4_0 in this checkpoint, found {:?}", tensor.ggml_type);
        let range = parsed
            .tensor_data_range(tensor, file_len)
            .context("tensor byte range within file bounds")?;
        let mut buf = vec![0u8; (range.end - range.start) as usize];
        file.seek(SeekFrom::Start(range.start)).context("seek to tensor data")?;
        file.read_exact(&mut buf).context("read exact tensor byte range")?;
        Ok(buf)
    }

    fn random_activation(seed: u64, count: usize) -> Vec<f32> {
        use proxima_tensor::test_support::Lcg;
        let mut lcg = Lcg(seed);
        (0..count).map(|_| lcg.next_unit()).collect()
    }

    /// `[FEED_FORWARD, EMBEDDING] x [TOKENS, EMBEDDING] -> [TOKENS,
    /// FEED_FORWARD]`, reduced over `EMBEDDING` -- the exact node-2994/gate
    /// construction `omega/tests/replay_projection.rs::append_matmul` proves
    /// byte-exact against a real production capture at this same shape
    /// (only the token count differs: 600 there, 510 here, matching
    /// `docs/model-interop/discipline.md` ROW C4.11's own probed shape).
    fn gate_program() -> (Vec<Op>, NodeId, NodeId, NodeId) {
        let mut program = Vec::new();
        let weight = append(
            &mut program,
            Op::Input {
                dtype: DType::UInt8,
                shape: vec![Extent::Static(FEED_FORWARD), Extent::Static(EMBEDDING)],
                name: None,
            },
        );
        let activation = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(TOKENS), Extent::Static(EMBEDDING)],
                name: None,
            },
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (weight, IndexMap::Affine(projection(3, &[1, 2]))),
                    (activation, IndexMap::Affine(projection(3, &[0, 2]))),
                ],
                name: None,
            },
        );
        let gate = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(projection(3, &[0, 1])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        (program, weight, activation, gate)
    }

    fn compile_pipeline(
        device: &ProtocolObject<dyn MTLDevice>,
        source: &str,
        entry: &str,
        math_mode: MTLMathMode,
    ) -> anyhow::Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        let options = MTLCompileOptions::new();
        options.setMathMode(math_mode);
        let ns_source = NSString::from_str(source);
        let library = device
            .newLibraryWithSource_options_error(&ns_source, Some(&options))
            .map_err(|error| anyhow::anyhow!("compiles `{entry}`: {}", error.localizedDescription()))?;
        let ns_entry = NSString::from_str(entry);
        let function = library
            .newFunctionWithName(&ns_entry)
            .with_context(|| format!("entry `{entry}` present in its own compiled library"))?;
        device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| anyhow::anyhow!("creates the pipeline for `{entry}`: {}", error.localizedDescription()))
    }

    fn shared_buffer_from_bytes(
        device: &ProtocolObject<dyn MTLDevice>,
        bytes: &[u8],
    ) -> anyhow::Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
        // SAFETY: `bytes` outlives this call; `newBufferWithBytes_length_options`
        // copies once and never retains the pointer.
        let pointer = unsafe { NonNull::new_unchecked(bytes.as_ptr() as *mut c_void) };
        unsafe { device.newBufferWithBytes_length_options(pointer, bytes.len(), MTLResourceOptions::StorageModeShared) }
            .context("device allocates a fresh shared buffer")
    }

    fn zeroed_buffer(device: &ProtocolObject<dyn MTLDevice>, len: usize) -> anyhow::Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
        shared_buffer_from_bytes(device, &vec![0u8; len])
    }

    fn read_f32(buffer: &ProtocolObject<dyn MTLBuffer>, count: usize) -> Vec<f32> {
        let pointer = buffer.contents().as_ptr().cast::<f32>();
        // SAFETY: `buffer` was allocated with at least `count * 4` bytes and
        // is `StorageModeShared`, so the CPU-visible mapping is valid for
        // this read after `waitUntilCompleted`.
        unsafe { std::slice::from_raw_parts(pointer, count) }.to_vec()
    }

    // every argument is a distinct GPU resource/dispatch-shape parameter this
    // probe's arms genuinely vary independently -- bundling any of them into
    // a struct would just relocate the same eight fields.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_gpu_ns(
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        weight: &ProtocolObject<dyn MTLBuffer>,
        activation: &ProtocolObject<dyn MTLBuffer>,
        output: &ProtocolObject<dyn MTLBuffer>,
        uniform: &ProtocolObject<dyn MTLBuffer>,
        grid_threads: u64,
        threadgroup_width: usize,
    ) -> anyhow::Result<f64> {
        let command_buffer = queue.commandBuffer().context("command buffer")?;
        let encoder = command_buffer.computeCommandEncoder().context("compute encoder")?;
        encoder.setComputePipelineState(pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(weight), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(activation), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(output), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(uniform), 0, 3);
        }
        let grid = MTLSize { width: grid_threads as usize, height: 1, depth: 1 };
        let threadgroup = MTLSize { width: threadgroup_width, height: 1, depth: 1 };
        encoder.dispatchThreads_threadsPerThreadgroup(grid, threadgroup);
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        Ok((command_buffer.GPUEndTime() - command_buffer.GPUStartTime()).max(0.0) * 1e9)
    }

    struct Stats {
        mean: f64,
        cov_pct: f64,
        samples: Vec<f64>,
    }

    fn stats(samples: &[f64]) -> Stats {
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance = samples.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        let cov_pct = (variance.sqrt() / mean) * 100.0;
        Stats { mean, cov_pct, samples: samples.to_vec() }
    }

    if let Ok(output) = std::process::Command::new("pgrep").args(["-fl", "cargo|rustc"]).output()
        && !output.stdout.is_empty()
    {
        eprintln!("WARNING: cargo/rustc running elsewhere:\n{}", String::from_utf8_lossy(&output.stdout));
    }

    // -- real weight bytes --
    let (parsed, file_len, mut file) = real_gguf_header(std::path::Path::new(REAL_GEMMA4_GGUF_PATH))?
        .context("real gemma4-E2B gguf header parses")?;
    let weight_bytes = real_tensor_bytes(&mut file, &parsed, file_len, "blk.0.ffn_gate.weight")?;
    let activation_bytes: Vec<u8> = random_activation(97, (TOKENS as usize) * (EMBEDDING as usize))
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();

    // -- production kernel source + uniforms + grid, via the SAME public
    // path `omega::emit` itself takes (no hand-rolled uniform packer) --
    let (program, weight_node, _activation_node, gate_node) = gate_program();
    let numeric_policy = NumericPolicy::llama_relaxed();
    let shapes = infer(&program, &[]).context("gate program infers")?;
    let mut bound_ops = bind(&program, &shapes, &[gate_node], numeric_policy).context("gate program binds")?;
    let packed_node_set: BTreeSet<NodeId> = [weight_node].into_iter().collect();
    correct_packed_matmul_layouts(&mut bound_ops, &packed_node_set);
    let bound = bound_ops
        .into_iter()
        .find(|op| op.node == gate_node)
        .context("gate's own fused reduce is present in the bound program")?;
    let packed_operands: omega::PackedOperands = std::collections::BTreeMap::from([(weight_node, omega::Codec::Q4_0)]);
    let kernel = omega::emit(&bound, &packed_operands, numeric_policy).context("production kernel emits")?;
    let uniform_bytes = omega::metal::pack_uniforms_for(&bound, numeric_policy).context("production uniforms pack")?;

    eprintln!(
        "kernel entry={} grid_threads={} bindings={:?}",
        kernel.entry, kernel.grid.threads, kernel.bindings
    );

    let source_a = kernel.source.clone();
    let output_dir = std::env::temp_dir().join("proxima-tiled-gemm-staging-mma-decomposition");
    std::fs::create_dir_all(&output_dir).context("create staging-mma-decomposition output dir")?;
    std::fs::write(output_dir.join("live_source_a.metal"), &source_a).ok();
    let source_a_has_wide_activation_marker = source_a.contains("act_tile[a_col * 32 + a_k] = wide.x;");
    anyhow::ensure!(
        source_a_has_wide_activation_marker,
        "wide activation staging marker must be present, found={source_a_has_wide_activation_marker}"
    );
    let source_a_has_mma_call_marker =
        source_a.contains("simdgroup_multiply_accumulate(acc[i * 2 + j], a_frag[i], b_frag[j], acc[i * 2 + j]);");
    anyhow::ensure!(
        source_a_has_mma_call_marker,
        "MMA call marker must be present, found={source_a_has_mma_call_marker}"
    );
    let source_a_has_weight_tile_decl_marker = source_a.contains("threadgroup half *weight_tile");
    anyhow::ensure!(
        source_a_has_weight_tile_decl_marker,
        "weight_tile decl marker must be present, found={source_a_has_weight_tile_decl_marker}"
    );

    // -- Arm B: MMA removed, staging + store kept, dependency-forced --
    let mma_block = "\
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 2; ++j) {
                    simdgroup_multiply_accumulate(acc[i * 2 + j], a_frag[i], b_frag[j], acc[i * 2 + j]);
                }
            }";
    let mma_removed_with_dep = "\
            if (sub_k == 0) { acc[0] = make_filled_simdgroup_matrix<float, 8>(weight_tile[0] + act_tile[0]); }";
    let mma_block_count = source_a.matches(mma_block).count();
    anyhow::ensure!(
        mma_block_count == 1,
        "MMA block must appear exactly once, found {mma_block_count} occurrences"
    );
    let source_b = source_a.replacen(mma_block, mma_removed_with_dep, 1);

    // -- Arm C: weight (dequant) staging -> constant half fill --
    let weight_stage_block_start = "            if (w_feat < feature_extent) {";
    let weight_stage_block_end = "            } else {\n                for (long fill_k = 0; fill_k < 32; ++fill_k) { weight_tile[w_row * 32 + fill_k] = 0.0h; }\n            }";
    let weight_stage_start_index = source_a.find(weight_stage_block_start).context("weight staging start marker present")?;
    let weight_stage_end_index = source_a.find(weight_stage_block_end).context("weight staging end marker present")? + weight_stage_block_end.len();
    let weight_stage_full = &source_a[weight_stage_start_index..weight_stage_end_index];
    let weight_stage_constant = "\
                for (int j = 0; j < 32; ++j) { weight_tile[w_row * 32 + j] = 1.0h; }";
    let source_c = source_a.replacen(weight_stage_full, weight_stage_constant, 1);

    // -- Arm D: activation staging -> constant float fill --
    let act_stage_block_start = "        bool act_tile_interior";
    let act_stage_block_end = "\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n        for (int sub_k";
    let act_stage_start_index = source_a.find(act_stage_block_start).context("activation staging start marker present")?;
    let act_stage_barrier_index = source_a.find(act_stage_block_end).context("activation staging end marker present")?;
    let act_stage_full = &source_a[act_stage_start_index..act_stage_barrier_index];
    let act_stage_constant = "\
        for (long idx = tiitg; idx < 1024; idx += 128) { act_tile[idx] = 1.0f; }";
    let source_d = source_a.replacen(act_stage_full, act_stage_constant, 1);

    // -- Arm E: both staging steps removed from the per-K-substep loop;
    // tile memory constant-filled ONCE outside the loop instead --
    let loop_open = "    for (long k0 = 0; k0 < u.reduction_total; k0 += 32) {\n";
    let loop_open_index = source_a.find(loop_open).context("k0 loop open marker present")?;
    let sub_k_open = "        for (int sub_k = 0; sub_k < 4; ++sub_k) {";
    let sub_k_open_index = source_a.find(sub_k_open).context("sub_k loop open marker present")?;
    let once_fill = "\
    for (long fill_idx = tiitg; fill_idx < 2048; fill_idx += 128) { tg_shared[fill_idx] = 1; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
";
    let source_e = format!(
        "{}{}{}{}",
        &source_a[..loop_open_index],
        once_fill,
        loop_open,
        &source_a[sub_k_open_index..]
    );
    let source_e_has_mma = source_e.contains("simdgroup_multiply_accumulate");
    anyhow::ensure!(source_e_has_mma, "arm E must keep the MMA call, contains_mma={source_e_has_mma}");

    // -- Arm A': production kernel with `PROXIMA_TILED_GEMM_WIDE_WEIGHT_
    // STAGE=1` (see `docs/model-interop/discipline.md` ROW C4.12) -- and
    // Arm A'+dstore: the same, combined
    // with `PROXIMA_TILED_GEMM_DIRECT_STORE=1`. Both are re-emitted through
    // `omega::emit`, the same production path `source_a` came from, never a
    // hand-edited copy of `source_a`.
    let source_a_wide: anyhow::Result<String> =
        temp_env::with_var("PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE", Some("1"), || {
            Ok(omega::emit(&bound, &packed_operands, numeric_policy)
                .context("wide-weight-stage kernel emits")?
                .source)
        });
    let source_a_wide = source_a_wide?;
    let source_a_wide_has_q4_0_run8_wide = source_a_wide.contains("q4_0_run8_wide");
    let source_a_wide_has_wws_blk0 = source_a_wide.contains("wws_blk0");
    anyhow::ensure!(
        source_a_wide_has_q4_0_run8_wide && source_a_wide_has_wws_blk0,
        "arm A' must decode through q4_0_run8_wide and carry the wws_ per-thread block pointer, has_q4_0_run8_wide={source_a_wide_has_q4_0_run8_wide} has_wws_blk0={source_a_wide_has_wws_blk0}"
    );
    let source_a_wide_dstore: anyhow::Result<String> = temp_env::with_vars(
        [
            ("PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE", Some("1")),
            ("PROXIMA_TILED_GEMM_DIRECT_STORE", Some("1")),
        ],
        || {
            Ok(omega::emit(&bound, &packed_operands, numeric_policy)
                .context("wide-weight-stage + direct-store kernel emits")?
                .source)
        },
    );
    let source_a_wide_dstore = source_a_wide_dstore?;
    let source_a_wide_dstore_has_wide_stage = source_a_wide_dstore.contains("q4_0_run8_wide");
    let source_a_wide_dstore_has_direct_store = source_a_wide_dstore.contains("direct_store_interior");
    anyhow::ensure!(
        source_a_wide_dstore_has_wide_stage && source_a_wide_dstore_has_direct_store,
        "arm A'+dstore must carry both the wide-weight-stage and direct-store markers, has_wide_stage={source_a_wide_dstore_has_wide_stage} has_direct_store={source_a_wide_dstore_has_direct_store}"
    );

    let device = MTLCreateSystemDefaultDevice().context("a real Metal device")?;
    let queue = device.newCommandQueue().context("a real command queue")?;
    let math_mode = MTLMathMode::Relaxed;

    let weight_buffer = shared_buffer_from_bytes(&device, &weight_bytes)?;
    let activation_buffer = shared_buffer_from_bytes(&device, &activation_bytes)?;
    let uniform_buffer = shared_buffer_from_bytes(&device, &uniform_bytes)?;
    let output_len = (TOKENS as usize) * (FEED_FORWARD as usize) * size_of::<f32>();
    let grid_threads = kernel.grid.threads;
    let threadgroup_width = 128usize;

    let arms: Vec<(&str, String)> = vec![
        ("A_production", source_a.clone()),
        ("B_mma_removed", source_b),
        ("C_weight_const", source_c),
        ("D_act_const", source_d),
        ("E_both_const", source_e),
        ("A_wide_weight_stage", source_a_wide),
        ("A_wide_weight_stage_dstore", source_a_wide_dstore),
    ];

    type PipelineArm = (
        &'static str,
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        Retained<ProtocolObject<dyn MTLBuffer>>,
    );
    let pipelines: Vec<PipelineArm> = arms
        .iter()
        .map(|(label, source)| {
            let pipeline = compile_pipeline(&device, source, &kernel.entry, math_mode)?;
            let output_buffer = zeroed_buffer(&device, output_len)?;
            Ok((*label, pipeline, output_buffer))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    // -- correctness gate: arm A vs the real production path --
    let production_plan = omega::plan(&program, &[], &[
        QuantizedBlock::Packed { codec: omega::Codec::Q4_0, bytes: &weight_bytes },
        QuantizedBlock::Float32(unsafe { std::slice::from_raw_parts(activation_bytes.as_ptr().cast::<f32>(), (TOKENS as usize) * (EMBEDDING as usize)) }),
    ], &[gate_node], numeric_policy)
        .context("production plan compiles")?;
    let production_evaluated = omega::execute_plan(&production_plan, &[
        QuantizedBlock::Packed { codec: omega::Codec::Q4_0, bytes: &weight_bytes },
        QuantizedBlock::Float32(unsafe { std::slice::from_raw_parts(activation_bytes.as_ptr().cast::<f32>(), (TOKENS as usize) * (EMBEDDING as usize)) }),
    ]).context("production plan executes")?;
    let production_output: Vec<f32> = production_evaluated.root().to_vec();

    dispatch_gpu_ns(
        &queue,
        &pipelines[0].1,
        &weight_buffer,
        &activation_buffer,
        &pipelines[0].2,
        &uniform_buffer,
        grid_threads,
        threadgroup_width,
    )?;
    let arm_a_output = read_f32(&pipelines[0].2, (TOKENS as usize) * (FEED_FORWARD as usize));
    let differing_words = arm_a_output
        .iter()
        .zip(production_output.iter())
        .filter(|(replayed, production)| replayed.to_bits() != production.to_bits())
        .count();
    println!(
        "CORRECTNESS_GATE arm_a_vs_production differing_words={differing_words} total={}",
        arm_a_output.len()
    );

    // -- correctness gate: arm A' (wide-weight-stage) and A'+dstore vs A,
    // both dispatched off the same weight/activation/uniform buffers --
    for (label, pipeline, output_buffer) in pipelines
        .iter()
        .filter(|(label, ..)| *label == "A_wide_weight_stage" || *label == "A_wide_weight_stage_dstore")
    {
        dispatch_gpu_ns(
            &queue,
            pipeline,
            &weight_buffer,
            &activation_buffer,
            output_buffer,
            &uniform_buffer,
            grid_threads,
            threadgroup_width,
        )?;
        let arm_output = read_f32(output_buffer, (TOKENS as usize) * (FEED_FORWARD as usize));
        let differing = arm_output
            .iter()
            .zip(arm_a_output.iter())
            .filter(|(candidate, production)| candidate.to_bits() != production.to_bits())
            .count();
        println!(
            "CORRECTNESS_GATE {label}_vs_A differing_words={differing} total={}",
            arm_output.len()
        );
        anyhow::ensure!(
            differing == 0,
            "{label} must be BIT-IDENTICAL to arm A (production): differing_words={differing} expected=0"
        );
    }

    // -- interleaved timed runs, GPU clock, WARMUP dropped --
    let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(ITERS); pipelines.len()];
    for iteration in 0..ITERS {
        for (index, (_label, pipeline, output_buffer)) in pipelines.iter().enumerate() {
            let gpu_ns = dispatch_gpu_ns(
                &queue,
                pipeline,
                &weight_buffer,
                &activation_buffer,
                output_buffer,
                &uniform_buffer,
                grid_threads,
                threadgroup_width,
            )?;
            if iteration >= WARMUP {
                samples[index].push(gpu_ns / 1e6);
            }
        }
    }

    println!("STEADY_STATE iters={} warmup_dropped={WARMUP}", ITERS - WARMUP);
    let mut baseline_mean = 0.0;
    for (index, (label, _pipeline, _output_buffer)) in pipelines.iter().enumerate() {
        let run_stats = stats(&samples[index]);
        if *label == "A_production" {
            baseline_mean = run_stats.mean;
        }
        println!(
            "ARM label={label} mean_ms={:.4} cov_pct={:.2} share_of_A={:.3} samples={:?}",
            run_stats.mean,
            run_stats.cov_pct,
            run_stats.mean / baseline_mean.max(f64::EPSILON),
            run_stats.samples.iter().map(|value| format!("{value:.3}")).collect::<Vec<_>>()
        );
    }
    Ok(())
}
