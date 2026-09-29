//! Isolated A/B: the rewritten `push_tiled_gemm_restage_writeback` copy-out
//! loop (see `docs/model-interop/discipline.md` ROW C4.13) vs the previous
//! copy-out text, at real production dense-batched attention shapes and the
//! real Q4_0 FFN gate projection. Every "new" source is the LIVE text
//! `omega::emit` renders today. Every "old" source is built by textually
//! splicing the previous copy-out algorithm back into that same live text --
//! never a hand-typed kernel, never a reverted source tree. The splice logic
//! is verified against `GOLDEN_COPYOUT_TAIL` (the pre-rewrite copy-out
//! text's own tail, captured before the rewrite landed) before it is
//! trusted on any other shape.
//!
//! Family A ("score", batch-innermost) and family B ("attended"/P.V,
//! feature-innermost) op builders restate
//! `omega/tests/dense_batched_direct_store_parity.rs`'s own
//! `dense_batched_batch_innermost_program`/`dense_batched_feature_fastest_program`
//! (private to that binary; `omega/tests/*.rs` doc convention is that each
//! integration-test/example crate restates its own fixtures rather than
//! sharing across separately-compiled binaries). The Q4_0 gate-projection
//! builder restates `tiled_gemm_staging_mma_decomposition.rs`'s own
//! `gate_program`, at the real `blk.0.ffn_gate.weight` dims
//! (`in_dim=1536, out_dim=6144`).
//!
//! GPU clock only (`MTLCommandBuffer::GPUStartTime`/`GPUEndTime`), never
//! host wall time. Arms interleaved per iteration across ALL variants of ALL
//! three ops in one loop.

fn main() -> anyhow::Result<()> {
    #[cfg(all(feature = "metal", feature = "metal-tiled-gemm", target_os = "macos"))]
    return run();
    #[cfg(not(all(feature = "metal", feature = "metal-tiled-gemm", target_os = "macos")))]
    {
        println!("tiled_gemm_copyout_restage_ab requires --features metal,metal-tiled-gemm on macOS");
        Ok(())
    }
}

#[cfg(all(feature = "metal", feature = "metal-tiled-gemm", target_os = "macos"))]
fn run() -> anyhow::Result<()> {
    use anyhow::Context;
    use core::ffi::c_void;
    use core::ptr::NonNull;
    use std::collections::BTreeMap;
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
        DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, Reduce,
        ReduceInit, ScalarOp, append, bind, correct_packed_matmul_layouts, infer, projection,
        test_support::Lcg,
    };

    const REAL_GEMMA4_GGUF_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";
    const ITERS: usize = 16;
    const WARMUP: usize = 4;
    // captured tail (from the last threadgroup barrier onward) of the
    // pre-rewrite `push_tiled_gemm_restage_writeback` output at the real
    // Q4_0 ffn_gate shape, before `docs/model-interop/discipline.md` ROW
    // C4.13's address-arithmetic rewrite landed.
    const GOLDEN_COPYOUT_TAIL: &str = "    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (long idx = tiitg; idx < 2048; idx += 128) {
        long o_row = idx / 32;
        long o_col = idx % 32;
        long o_feat = row_tile * 64 + o_row;
        long o_tok = col_tile * 32 + o_col;
        if (o_feat < feature_extent && o_tok < token_extent) {
            long coord[3];
            for (int d = 0; d < 3; ++d) { coord[d] = 0; }
            coord[1] = o_feat;
            coord[0] = o_tok;
            long out_offset = u.out_base;
            out_offset += coord[0] * u.out_strides[0];
            out_offset += coord[1] * u.out_strides[1];
            out_offset += coord[2] * u.out_strides[2];
            out[out_offset] = (float)out_tile[idx];
        }
    }
}
";

    // -- shared plumbing, restated per omega/tests/*.rs convention --

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

    fn random_f32(seed: u64, count: usize) -> Vec<f32> {
        let mut lcg = Lcg(seed);
        (0..count).map(|_| lcg.next_unit() * 2.0 - 1.0).collect()
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_le_bytes()).collect()
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

    #[allow(clippy::too_many_arguments)]
    fn dispatch_gpu_ns(
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        bindings: &[omega::Binding],
        weight_node: NodeId,
        other_node: NodeId,
        weight: &ProtocolObject<dyn MTLBuffer>,
        other: &ProtocolObject<dyn MTLBuffer>,
        output: &ProtocolObject<dyn MTLBuffer>,
        uniform: &ProtocolObject<dyn MTLBuffer>,
        grid_threads: u64,
        threadgroup_width: usize,
    ) -> anyhow::Result<f64> {
        let command_buffer = queue.commandBuffer().context("command buffer")?;
        let encoder = command_buffer.computeCommandEncoder().context("compute encoder")?;
        encoder.setComputePipelineState(pipeline);
        for (index, binding) in bindings.iter().enumerate() {
            let buffer: &ProtocolObject<dyn MTLBuffer> = match *binding {
                omega::Binding::Input(id) if id == weight_node => weight,
                omega::Binding::Input(id) if id == other_node => other,
                omega::Binding::Output(_) => output,
                omega::Binding::Uniforms => uniform,
                other_binding => anyhow::bail!("unexpected binding {other_binding:?} at index {index}"),
            };
            unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), 0, index) };
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

    // -- copy-out splice: parse the current renderer's own emitted text and
    // reconstruct the previous algorithm's text at the SAME parameters,
    // rather than hand-typing per-shape kernel text --

    const BARRIER: &str = "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    const LOOP_PREFIX: &str = "    for (long idx = tiitg; idx < ";
    const IF_LINE: &str = "        if (o_feat < feature_extent && o_tok < token_extent) {\n";
    const EPILOGUE_LINE: &str = "            out[out_offset] = (float)out_tile[idx];\n";

    fn take_between<'a>(text: &'a str, prefix: &str, suffix: &str) -> anyhow::Result<&'a str> {
        let after = text
            .split_once(prefix)
            .with_context(|| format!("prefix `{prefix}` present in `{text}`"))?
            .1;
        Ok(after
            .split_once(suffix)
            .with_context(|| format!("suffix `{suffix}` present after prefix `{prefix}` in `{text}`"))?
            .0)
    }

    fn parse_batch_axes(pre_loop_setup: &str) -> anyhow::Result<Vec<u16>> {
        let mut batch_axes = Vec::new();
        for line in pre_loop_setup.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("coord[")
                && let Some((axis_str, rest2)) = rest.split_once(']')
                && rest2.trim_start().starts_with("= dense_batch_coord_")
            {
                batch_axes.push(axis_str.parse::<u16>().context("batch axis index parses")?);
            }
        }
        Ok(batch_axes)
    }

    fn parse_coord_axis(new_inner_body: &str, rhs: &str) -> anyhow::Result<u16> {
        for line in new_inner_body.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("coord[")
                && let Some((axis_str, rest2)) = rest.split_once(']')
                && rest2.trim_start() == format!("= {rhs};")
            {
                return axis_str.parse::<u16>().context("coord axis index parses");
            }
        }
        anyhow::bail!("no `coord[N] = {rhs};` line found in `{new_inner_body}`")
    }

    struct RestageParams {
        rank: usize,
        feature_axis: u16,
        token_axis: u16,
        batch_axes: Vec<u16>,
    }

    /// Splices the previous copy-out algorithm's text in place of the
    /// current rewrite's text, on a LIVE `omega::emit` render. Never edits
    /// `omega/src`; never reverts the tree. The previous algorithm
    /// (zero-init `coord[rank]` fresh every iteration, no `out_offset_base`
    /// hoist, full `0..rank` stride-accumulation loop every iteration) is
    /// confirmed identical across two independent saved samples spanning 0
    /// and 1 batch axis (`decomp/live_source_a.metal:992-999`,
    /// `ir/proxima_dense.metal:954-964`); this function generalizes it to N
    /// batch axes mechanically (one `coord[axis] = dense_batch_coord_axis;`
    /// line per axis, integer-addition order does not affect the computed
    /// offset), then self-verifies against the Q4_0 golden fixture below
    /// before being trusted on the dense-batched shapes.
    fn render_old_copyout(source: &str) -> anyhow::Result<(String, RestageParams)> {
        let barrier_idx = source
            .rfind(BARRIER)
            .context("restage tail has a preceding threadgroup barrier")?;
        let after_barrier = barrier_idx + BARRIER.len();
        let loop_rel = source[after_barrier..]
            .find(LOOP_PREFIX)
            .context("copy-out loop header present after the last barrier")?;
        let loop_idx = after_barrier + loop_rel;
        let pre_loop_setup = &source[after_barrier..loop_idx];

        let if_rel = source[loop_idx..].find(IF_LINE).context("bounds-check if present")?;
        let if_idx = loop_idx + if_rel;
        let after_if_open = if_idx + IF_LINE.len();
        let shared_loop_header = &source[loop_idx..after_if_open];

        let epi_rel = source[after_if_open..]
            .find(EPILOGUE_LINE)
            .context("identity epilogue write present")?;
        let epi_idx = after_if_open + epi_rel;
        let new_inner_body = &source[after_if_open..epi_idx];

        let rank: usize = take_between(pre_loop_setup, "long coord[", "];")?
            .trim()
            .parse()
            .context("rank parses")?;
        let batch_axes = parse_batch_axes(pre_loop_setup)?;
        let feature_axis = parse_coord_axis(new_inner_body, "o_feat")?;
        let token_axis = parse_coord_axis(new_inner_body, "o_tok")?;

        let mut old_inner = String::new();
        old_inner.push_str(&format!("            long coord[{rank}];\n"));
        old_inner.push_str(&format!("            for (int d = 0; d < {rank}; ++d) {{ coord[d] = 0; }}\n"));
        old_inner.push_str(&format!("            coord[{feature_axis}] = o_feat;\n"));
        old_inner.push_str(&format!("            coord[{token_axis}] = o_tok;\n"));
        for axis in &batch_axes {
            old_inner.push_str(&format!("            coord[{axis}] = dense_batch_coord_{axis};\n"));
        }
        old_inner.push_str("            long out_offset = u.out_base;\n");
        for dim in 0..rank {
            old_inner.push_str(&format!("            out_offset += coord[{dim}] * u.out_strides[{dim}];\n"));
        }

        let old_source = format!(
            "{}{}{}{}",
            &source[..after_barrier],
            shared_loop_header,
            old_inner,
            &source[epi_idx..],
        );
        Ok((old_source, RestageParams { rank, feature_axis, token_axis, batch_axes }))
    }

    // -- family A ("score", batch-innermost): restates
    // omega/tests/dense_batched_direct_store_parity.rs::dense_batched_batch_innermost_program --

    fn family_a_program(token: u32, reduce_len: u32, feature: u32, batch: u32) -> (Vec<Op>, NodeId, NodeId, NodeId) {
        let mut program = Vec::new();
        let weight = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(feature), Extent::Static(batch), Extent::Static(reduce_len)],
                name: None,
            },
        );
        let other = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(token), Extent::Static(batch), Extent::Static(reduce_len)],
                name: None,
            },
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (weight, IndexMap::Affine(projection(4, &[2, 3, 1]))),
                    (other, IndexMap::Affine(projection(4, &[0, 3, 1]))),
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
                in_map: IndexMap::Affine(projection(4, &[0, 1, 2, 3])),
                out_map: IndexMap::Affine(projection(4, &[0, 2, 3])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        (program, weight, other, sum)
    }

    // -- family B ("attended"/P.V, feature-fastest): restates
    // omega/tests/dense_batched_direct_store_parity.rs::dense_batched_feature_fastest_program --

    fn family_b_program(token: u32, reduce_len: u32, batch: u32, feature: u32) -> (Vec<Op>, NodeId, NodeId, NodeId) {
        let mut program = Vec::new();
        let weight = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(feature), Extent::Static(batch), Extent::Static(reduce_len)],
                name: None,
            },
        );
        let other = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(token), Extent::Static(batch), Extent::Static(reduce_len)],
                name: None,
            },
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (weight, IndexMap::Affine(projection(4, &[3, 2, 1]))),
                    (other, IndexMap::Affine(projection(4, &[0, 2, 1]))),
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
                in_map: IndexMap::Affine(projection(4, &[0, 1, 2, 3])),
                out_map: IndexMap::Affine(projection(4, &[0, 2, 3])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        (program, weight, other, sum)
    }

    // -- Q4_0 gate projection: restates
    // tiled_gemm_staging_mma_decomposition.rs::gate_program at the real
    // out_dim=6144 (not the synthetic 12288 the brief's own cited figure
    // used) --

    fn q4_0_gate_program(tokens: u32, embedding: u32, feed_forward: u32) -> (Vec<Op>, NodeId, NodeId, NodeId) {
        let mut program = Vec::new();
        let weight = append(
            &mut program,
            Op::Input {
                dtype: DType::UInt8,
                shape: vec![Extent::Static(feed_forward), Extent::Static(embedding)],
                name: None,
            },
        );
        let activation = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(tokens), Extent::Static(embedding)],
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

    // -- disclose foreign load immediately before the timed run, matching
    // the convention in reduction_literal_prefill_speed_probe.rs and
    // tiled_gemm_ggml_incumbent_arms.rs --
    if let Ok(output) = std::process::Command::new("pgrep").args(["-fl", "cargo|rustc"]).output()
        && !output.stdout.is_empty()
    {
        eprintln!(
            "WARNING: a cargo/rustc process is running elsewhere on this box:\n{}",
            String::from_utf8_lossy(&output.stdout)
        );
    }

    let device = MTLCreateSystemDefaultDevice().context("a real Metal device")?;
    let queue = device.newCommandQueue().context("a real command queue")?;
    let math_mode = MTLMathMode::Relaxed;

    struct PipelineEntry {
        label: String,
        group: usize,
        pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        output_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    }

    struct Group {
        op_label: &'static str,
        weight_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        other_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        uniform_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        bindings: Vec<omega::Binding>,
        weight_node: NodeId,
        other_node: NodeId,
        grid_threads: u64,
        element_count: usize,
    }

    let mut groups: Vec<Group> = Vec::new();
    let mut pipelines: Vec<PipelineEntry> = Vec::new();
    // (group_index, new_label, old_label) pairs for the correctness gate.
    let mut parity_pairs: Vec<(usize, String, String)> = Vec::new();

    // ==================== ARM 1: family A, score, weather shape ====================
    {
        let (token, reduce_len, feature, batch) = (563u32, 128u32, 576u32, 8u32);
        let (program, weight_node, other_node, sum) = family_a_program(token, reduce_len, feature, batch);
        let shapes = infer(&program, &[]).context("family A infers")?;
        let bound_ops = bind(&program, &shapes, &[sum], NumericPolicy::default()).context("family A binds")?;
        let bound = bound_ops.into_iter().find(|op| op.node == sum).context("family A fused reduce present")?;
        let packed_operands = omega::PackedOperands::new();
        let uniform_bytes = omega::metal::pack_uniforms_for(&bound, NumericPolicy::default()).context("family A uniforms pack")?;

        let kernel_new = temp_env::with_var("PROXIMA_TILED_GEMM_DENSE", Some("1"), || {
            omega::emit(&bound, &packed_operands, NumericPolicy::default()).context("family A new-copyout kernel emits")
        })?;
        let (old_source, params) = render_old_copyout(&kernel_new.source)?;
        anyhow::ensure!(
            params.rank == 4,
            "family A rank must be 4 (token,reduce,feature,batch): rank={} expected=4",
            params.rank
        );
        let family_a_batch_axes_len = params.batch_axes.len();
        anyhow::ensure!(
            family_a_batch_axes_len == 1,
            "family A must carry exactly one batch axis in this fixture, found {family_a_batch_axes_len}"
        );
        println!(
            "FAMILY_A parsed rank={} feature_axis={} token_axis={} batch_axes={:?}",
            params.rank, params.feature_axis, params.token_axis, params.batch_axes
        );

        let device_ref = &device;
        let weight_bytes = f32_bytes(&random_f32(11001, (feature * batch * reduce_len) as usize));
        let other_bytes = f32_bytes(&random_f32(11502, (token * batch * reduce_len) as usize));
        let group_index = groups.len();
        groups.push(Group {
            op_label: "family_A_score_weather",
            weight_buffer: shared_buffer_from_bytes(device_ref, &weight_bytes)?,
            other_buffer: shared_buffer_from_bytes(device_ref, &other_bytes)?,
            uniform_buffer: shared_buffer_from_bytes(device_ref, &uniform_bytes)?,
            bindings: kernel_new.bindings.clone(),
            weight_node,
            other_node,
            grid_threads: kernel_new.grid.threads,
            element_count: (token * feature * batch) as usize,
        });
        let output_len = (token as usize) * (feature as usize) * (batch as usize) * size_of::<f32>();
        let new_label = "A_new_copyout".to_string();
        let old_label = "A_old_copyout".to_string();
        pipelines.push(PipelineEntry {
            label: new_label.clone(),
            group: group_index,
            pipeline: compile_pipeline(device_ref, &kernel_new.source, &kernel_new.entry, math_mode)?,
            output_buffer: zeroed_buffer(device_ref, output_len)?,
        });
        pipelines.push(PipelineEntry {
            label: old_label.clone(),
            group: group_index,
            pipeline: compile_pipeline(device_ref, &old_source, &kernel_new.entry, math_mode)?,
            output_buffer: zeroed_buffer(device_ref, output_len)?,
        });
        parity_pairs.push((group_index, new_label, old_label));
    }

    // ==================== ARM 2: family B, attended/P.V, weather shape, DIRECT_STORE on/off ====================
    {
        let (token, reduce_len, batch, feature) = (563u32, 32u32, 8u32, 256u32);
        let (program, weight_node, other_node, sum) = family_b_program(token, reduce_len, batch, feature);
        let shapes = infer(&program, &[]).context("family B infers")?;
        let bound_ops = bind(&program, &shapes, &[sum], NumericPolicy::default()).context("family B binds")?;
        let bound = bound_ops.into_iter().find(|op| op.node == sum).context("family B fused reduce present")?;
        let packed_operands = omega::PackedOperands::new();
        let uniform_bytes = omega::metal::pack_uniforms_for(&bound, NumericPolicy::default()).context("family B uniforms pack")?;

        let device_ref = &device;
        let weight_bytes = f32_bytes(&random_f32(12001, (feature * batch * reduce_len) as usize));
        let other_bytes = f32_bytes(&random_f32(12502, (token * batch * reduce_len) as usize));
        let group_index = groups.len();
        let output_len = (token as usize) * (batch as usize) * (feature as usize) * size_of::<f32>();

        for (dstore_tag, dstore_env) in [("dstore0", None::<&str>), ("dstore1", Some("1"))] {
            let kernel_new = temp_env::with_vars(
                [("PROXIMA_TILED_GEMM_DENSE", Some("1")), ("PROXIMA_TILED_GEMM_DIRECT_STORE", dstore_env)],
                || omega::emit(&bound, &packed_operands, NumericPolicy::default()).context("family B new-copyout kernel emits"),
            )?;
            let (old_source, params) = render_old_copyout(&kernel_new.source)?;
            println!(
                "FAMILY_B[{dstore_tag}] parsed rank={} feature_axis={} token_axis={} batch_axes={:?}",
                params.rank, params.feature_axis, params.token_axis, params.batch_axes
            );

            if dstore_tag == "dstore0" {
                groups.push(Group {
                    op_label: "family_B_attended_weather",
                    weight_buffer: shared_buffer_from_bytes(device_ref, &weight_bytes)?,
                    other_buffer: shared_buffer_from_bytes(device_ref, &other_bytes)?,
                    uniform_buffer: shared_buffer_from_bytes(device_ref, &uniform_bytes)?,
                    bindings: kernel_new.bindings.clone(),
                    weight_node,
                    other_node,
                    grid_threads: kernel_new.grid.threads,
                    element_count: (token * batch * feature) as usize,
                });
            }
            let new_label = format!("B_new_copyout_{dstore_tag}");
            let old_label = format!("B_old_copyout_{dstore_tag}");
            pipelines.push(PipelineEntry {
                label: new_label.clone(),
                group: group_index,
                pipeline: compile_pipeline(device_ref, &kernel_new.source, &kernel_new.entry, math_mode)?,
                output_buffer: zeroed_buffer(device_ref, output_len)?,
            });
            pipelines.push(PipelineEntry {
                label: old_label.clone(),
                group: group_index,
                pipeline: compile_pipeline(device_ref, &old_source, &kernel_new.entry, math_mode)?,
                output_buffer: zeroed_buffer(device_ref, output_len)?,
            });
            parity_pairs.push((group_index, new_label, old_label));
        }
    }

    // ==================== ARM 3: Q4_0 real ffn_gate weight, WIDE_WEIGHT_STAGE=1, DIRECT_STORE on/off ====================
    {
        const TOKENS: u32 = 510;
        const EMBEDDING: u32 = 1536;
        const FEED_FORWARD: u32 = 6144;

        let (parsed, file_len, mut file) = real_gguf_header(std::path::Path::new(REAL_GEMMA4_GGUF_PATH))?
            .context("real gemma4-E2B gguf header parses")?;
        let weight_bytes = real_tensor_bytes(&mut file, &parsed, file_len, "blk.0.ffn_gate.weight")?;
        let activation_bytes = f32_bytes(&random_f32(97, (TOKENS as usize) * (EMBEDDING as usize)));

        let (program, weight_node, activation_node, gate_node) = q4_0_gate_program(TOKENS, EMBEDDING, FEED_FORWARD);
        let numeric_policy = NumericPolicy::llama_relaxed();
        let shapes = infer(&program, &[]).context("gate program infers")?;
        let mut bound_ops = bind(&program, &shapes, &[gate_node], numeric_policy).context("gate program binds")?;
        let packed_node_set: std::collections::BTreeSet<NodeId> = [weight_node].into_iter().collect();
        correct_packed_matmul_layouts(&mut bound_ops, &packed_node_set);
        let bound = bound_ops.into_iter().find(|op| op.node == gate_node).context("gate fused reduce present")?;
        let packed_operands: omega::PackedOperands = BTreeMap::from([(weight_node, omega::Codec::Q4_0)]);
        let uniform_bytes = omega::metal::pack_uniforms_for(&bound, numeric_policy).context("Q4_0 uniforms pack")?;

        // -- golden self-check: base kernel (no WWS, no DSTORE) must
        // reconstruct byte-identical to GOLDEN_COPYOUT_TAIL, captured at
        // this exact shape before the copy-out rewrite (ROW C4.13) landed --
        let kernel_base = omega::emit(&bound, &packed_operands, numeric_policy).context("Q4_0 base kernel emits")?;
        let (reconstructed_old, base_params) = render_old_copyout(&kernel_base.source)?;
        // Only the copy-out TAIL (from the last threadgroup barrier onward)
        // is what the rewrite touched; other uncommitted lever work
        // (grid_threads, WIDE_WEIGHT_STAGE plumbing, etc.) has touched the
        // kernel body text ABOVE this point even with every switch off, so
        // the golden compare is scoped to the tail this splice replaces.
        let reconstructed_tail = &reconstructed_old[reconstructed_old.rfind(BARRIER).context("reconstructed tail has a barrier")?..];
        let reconstructed_tail_trimmed = reconstructed_tail.trim_end();
        let golden_copyout_tail_trimmed = GOLDEN_COPYOUT_TAIL.trim_end();
        anyhow::ensure!(
            reconstructed_tail_trimmed == golden_copyout_tail_trimmed,
            "reconstructed OLD copy-out tail must match GOLDEN_COPYOUT_TAIL byte-for-byte: reconstructed={reconstructed_tail_trimmed:?} golden={golden_copyout_tail_trimmed:?}"
        );
        println!(
            "GOLDEN_CHECK reconstructed_old_matches_live_source_a=true rank={} feature_axis={} token_axis={} batch_axes={:?}",
            base_params.rank, base_params.feature_axis, base_params.token_axis, base_params.batch_axes
        );

        let device_ref = &device;
        let group_index = groups.len();
        let output_len = (TOKENS as usize) * (FEED_FORWARD as usize) * size_of::<f32>();

        for (dstore_tag, dstore_env) in [("dstore0", None::<&str>), ("dstore1", Some("1"))] {
            let kernel_new = temp_env::with_vars(
                [("PROXIMA_TILED_GEMM_WIDE_WEIGHT_STAGE", Some("1")), ("PROXIMA_TILED_GEMM_DIRECT_STORE", dstore_env)],
                || omega::emit(&bound, &packed_operands, numeric_policy).context("Q4_0 new-copyout kernel emits"),
            )?;
            let (old_source, params) = render_old_copyout(&kernel_new.source)?;
            println!(
                "Q4_0[{dstore_tag}] parsed rank={} feature_axis={} token_axis={} batch_axes={:?}",
                params.rank, params.feature_axis, params.token_axis, params.batch_axes
            );

            if dstore_tag == "dstore0" {
                groups.push(Group {
                    op_label: "Q4_0_ffn_gate_wide_weight_stage",
                    weight_buffer: shared_buffer_from_bytes(device_ref, &weight_bytes)?,
                    other_buffer: shared_buffer_from_bytes(device_ref, &activation_bytes)?,
                    uniform_buffer: shared_buffer_from_bytes(device_ref, &uniform_bytes)?,
                    bindings: kernel_new.bindings.clone(),
                    weight_node,
                    other_node: activation_node,
                    grid_threads: kernel_new.grid.threads,
                    element_count: (TOKENS as usize) * (FEED_FORWARD as usize),
                });
            }
            let new_label = format!("Q4_0_new_copyout_{dstore_tag}");
            let old_label = format!("Q4_0_old_copyout_{dstore_tag}");
            pipelines.push(PipelineEntry {
                label: new_label.clone(),
                group: group_index,
                pipeline: compile_pipeline(device_ref, &kernel_new.source, &kernel_new.entry, math_mode)?,
                output_buffer: zeroed_buffer(device_ref, output_len)?,
            });
            pipelines.push(PipelineEntry {
                label: old_label.clone(),
                group: group_index,
                pipeline: compile_pipeline(device_ref, &old_source, &kernel_new.entry, math_mode)?,
                output_buffer: zeroed_buffer(device_ref, output_len)?,
            });
            parity_pairs.push((group_index, new_label, old_label));
        }
    }

    println!("PIPELINES built={} groups={}", pipelines.len(), groups.len());

    // -- correctness gate: old == new, 0 differing words, BEFORE timing --

    let find_entry = |label: &str| -> anyhow::Result<usize> {
        pipelines
            .iter()
            .position(|entry| entry.label == label)
            .with_context(|| format!("pipeline `{label}` present"))
    };

    for (group_index, new_label, old_label) in &parity_pairs {
        let group = &groups[*group_index];
        let new_index = find_entry(new_label)?;
        let old_index = find_entry(old_label)?;

        dispatch_gpu_ns(
            &queue,
            &pipelines[new_index].pipeline,
            &group.bindings,
            group.weight_node,
            group.other_node,
            &group.weight_buffer,
            &group.other_buffer,
            &pipelines[new_index].output_buffer,
            &group.uniform_buffer,
            group.grid_threads,
            128,
        )?;
        dispatch_gpu_ns(
            &queue,
            &pipelines[old_index].pipeline,
            &group.bindings,
            group.weight_node,
            group.other_node,
            &group.weight_buffer,
            &group.other_buffer,
            &pipelines[old_index].output_buffer,
            &group.uniform_buffer,
            group.grid_threads,
            128,
        )?;
        let new_output = read_f32(&pipelines[new_index].output_buffer, group.element_count);
        let old_output = read_f32(&pipelines[old_index].output_buffer, group.element_count);
        let differing = new_output
            .iter()
            .zip(old_output.iter())
            .filter(|(new_value, old_value)| new_value.to_bits() != old_value.to_bits())
            .count();
        println!(
            "VALIDATE op={} new={new_label} old={old_label} differing_words={differing} total={}",
            group.op_label,
            group.element_count
        );
        anyhow::ensure!(
            differing == 0,
            "{new_label} vs {old_label} must be bit-identical: differing_words={differing} expected=0"
        );
    }

    // -- interleaved timed runs, GPU clock, WARMUP dropped --

    let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(ITERS); pipelines.len()];
    for iteration in 0..ITERS {
        for (index, entry) in pipelines.iter().enumerate() {
            let group = &groups[entry.group];
            let gpu_ns = dispatch_gpu_ns(
                &queue,
                &entry.pipeline,
                &group.bindings,
                group.weight_node,
                group.other_node,
                &group.weight_buffer,
                &group.other_buffer,
                &entry.output_buffer,
                &group.uniform_buffer,
                group.grid_threads,
                128,
            )?;
            if iteration >= WARMUP {
                samples[index].push(gpu_ns / 1e6);
            }
        }
    }

    println!("STEADY_STATE iters={} warmup_dropped={WARMUP} clock=GPUStartTime/GPUEndTime", ITERS - WARMUP);
    for (group_index, new_label, old_label) in &parity_pairs {
        let group = &groups[*group_index];
        let new_index = find_entry(new_label)?;
        let old_index = find_entry(old_label)?;
        let new_stats = stats(&samples[new_index]);
        let old_stats = stats(&samples[old_index]);
        println!(
            "ARM op={} new_label={new_label} new_mean_ms={:.5} new_cov_pct={:.2} new_samples={:?}",
            group.op_label, new_stats.mean, new_stats.cov_pct, new_stats.samples.iter().map(|value| format!("{value:.4}")).collect::<Vec<_>>()
        );
        println!(
            "ARM op={} old_label={old_label} old_mean_ms={:.5} old_cov_pct={:.2} old_samples={:?}",
            group.op_label, old_stats.mean, old_stats.cov_pct, old_stats.samples.iter().map(|value| format!("{value:.4}")).collect::<Vec<_>>()
        );
        println!(
            "RATIO op={} pair={new_label}_vs_{old_label} old_over_new={:.4} new_over_old={:.4}",
            group.op_label,
            old_stats.mean / new_stats.mean.max(f64::EPSILON),
            new_stats.mean / old_stats.mean.max(f64::EPSILON)
        );
    }
    Ok(())
}
