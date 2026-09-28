//! Isolated production-shape probe for `c4-7-reduction-literal.md`'s
//! `PROXIMA_REDUCTION_LITERAL` bake: the decode-shaped (`token_total = 1`)
//! Q4_0 packed-row-blocked matvec, at six real gemma4-E2B weight shapes
//! (`K` in {1536, 256, 2048, 4096, 6144, 12288}), two arms per shape in ONE
//! process -- runtime-bound (`PROXIMA_REDUCTION_LITERAL` unset when the
//! plan is built) and literal (`=1`).
//!
//! Timing measures GPU execution, not CPU: [`omega::execute_plan_timed`]
//! reads back ONE command buffer's `GPUStartTime`/`GPUEndTime` after
//! encoding `repeat_count` independent, structurally-identical dispatches
//! of the SAME op (same weight bytes, same activation, `repeat_count`
//! distinct output nodes) -- the batched-dispatch shape that function's own
//! doc names as the fix for `execute_plan_op_timed`'s ~700us per-buffer
//! submit/wait floor swamping a single dispatch's real cost.
//! `repeat_count = ceil(2e9 / weight_bytes)` so every timed sample streams
//! at least 2 GB of weight bytes through the same pipeline, matching
//! decode's own bandwidth-bound shape (each of the `repeat_count`
//! dispatches re-reads the full weight matrix from device memory, the same
//! way consecutive decode tokens do -- unlike a multi-row/prefill body,
//! which would reuse one weight load across many activation rows and
//! measure a different thing).
//!
//! Real gemma4-E2B `Q4_0` weight bytes read from the ollama blob, same
//! shapes and helper style as `omega/tests/reduction_literal_ab.rs`; the
//! multi-axis attn_output shapes use its `multi_axis_matmul_program`'s own
//! `(kv_heads=2, group=4)` fold.

#![allow(clippy::unwrap_used, clippy::expect_used)]

fn main() {
    #[cfg(all(
        feature = "metal",
        feature = "metal-reduction-literal",
        feature = "instrument",
        target_os = "macos"
    ))]
    run();
    #[cfg(not(all(
        feature = "metal",
        feature = "metal-reduction-literal",
        feature = "instrument",
        target_os = "macos"
    )))]
    println!(
        "reduction_literal_speed_probe requires --features metal,metal-reduction-literal,instrument on macOS"
    );
}

#[cfg(all(
    feature = "metal",
    feature = "metal-reduction-literal",
    feature = "instrument",
    target_os = "macos"
))]
fn run() {
    use std::io::{Read, Seek, SeekFrom};

    use proxima_gguf::parser::{GgufEvent, GgufParser};
    use proxima_gguf::pipe::ParsedGguf;
    use proxima_gguf::quant::q4_0;
    use proxima_gguf::types::GgmlType;
    use proxima_primitives::Codec;
    use proxima_tensor::test_support::Lcg;
    use proxima_tensor::{
        DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce,
        ReduceInit, ScalarOp, append, projection,
    };

    const REAL_GEMMA4_GGUF_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";
    const MIN_STREAM_BYTES: f64 = 2.0e9;
    const WARMUP_PAIRS: usize = 2;
    const KEPT_PAIRS: usize = 12;
    const TOTAL_PAIRS: usize = WARMUP_PAIRS + KEPT_PAIRS;

    // -- real gguf loading, same shape as omega/tests/reduction_literal_ab.rs --

    fn real_gguf_header(path: &std::path::Path) -> Option<(ParsedGguf, u64, std::fs::File)> {
        let mut file = std::fs::File::open(path).ok()?;
        let file_len = file.metadata().ok()?.len();
        let mut prefix_len = 1usize << 20;
        loop {
            let mut buf = vec![0u8; prefix_len];
            file.seek(SeekFrom::Start(0)).expect("seek to start");
            let read = file.read(&mut buf).expect("read gguf prefix");
            buf.truncate(read);
            if let Ok((parser, events)) = GgufParser::new().push(&buf) {
                let mut version = None;
                let mut metadata = Vec::new();
                let mut tensors = Vec::new();
                let mut completion = None;
                for event in events {
                    match event {
                        GgufEvent::Header {
                            version: version_value,
                            ..
                        } => version = Some(version_value),
                        GgufEvent::Metadata { key, value } => metadata.push((key, value)),
                        GgufEvent::Tensor(tensor) => tensors.push(tensor),
                        GgufEvent::Complete {
                            data_offset,
                            alignment,
                        } => completion = Some((data_offset, alignment)),
                    }
                }
                if let (Some(version), Some((data_offset, alignment))) = (version, completion) {
                    parser.finish().expect("parser reports complete and clean");
                    let parsed = ParsedGguf {
                        version,
                        tensor_count: tensors.len() as u64,
                        kv_count: metadata.len() as u64,
                        metadata,
                        tensors,
                        data_offset,
                        alignment,
                    };
                    return Some((parsed, file_len, file));
                }
            }
            if prefix_len as u64 >= file_len {
                return None;
            }
            prefix_len *= 2;
        }
    }

    fn real_tensor_bytes(
        file: &mut std::fs::File,
        parsed: &ParsedGguf,
        file_len: u64,
        name: &str,
    ) -> Option<(Vec<u8>, usize, usize)> {
        let tensor = parsed.tensors.iter().find(|candidate| candidate.name == name)?;
        if tensor.ggml_type != GgmlType::Q4_0 {
            eprintln!(
                "real_tensor_bytes: {name} is {:?} in this file, not Q4_0 -- test skipped, not faked",
                tensor.ggml_type
            );
            return None;
        }
        let in_dim = tensor.dims[0] as usize;
        let out_dim = tensor.dims[1] as usize;
        let range = parsed
            .tensor_data_range(tensor, file_len)
            .expect("tensor byte range within file bounds");
        let mut buf = vec![0u8; (range.end - range.start) as usize];
        file.seek(SeekFrom::Start(range.start)).expect("seek to tensor data");
        file.read_exact(&mut buf).expect("read exact tensor byte range");
        Some((buf, in_dim, out_dim))
    }

    fn dequantize_rows(packed: &[u8], in_dim: usize, out_rows: usize) -> Vec<Vec<f32>> {
        let blocks_per_row = in_dim / q4_0::QK4_0;
        let row_bytes = blocks_per_row * q4_0::BLOCK_BYTES;
        packed
            .chunks_exact(row_bytes)
            .take(out_rows)
            .map(|row_blocks| {
                let mut row = vec![0.0f32; in_dim];
                q4_0::dequantize(row_blocks, &mut row).expect("a whole number of q4_0 blocks per row");
                row
            })
            .collect()
    }

    fn expected_output(rows: &[Vec<f32>], activation: &[f32]) -> Vec<f32> {
        rows.iter()
            .map(|row| {
                row.iter()
                    .zip(activation.iter())
                    .map(|(weight, value)| weight * value)
                    .sum::<f32>()
            })
            .collect()
    }

    // -- program builders: `repeat_count` independent single-row matvecs
    // sharing ONE weight input node and ONE activation input node, each
    // writing its own output node -- `omega::plan`'s bind step never merges
    // structurally-identical-but-distinct nodes (`horizontal_merge_dispatch::
    // eight_independent_matvecs_run_unmerged_today` proves this is the
    // current default), so this really does encode `repeat_count` back-to-
    // back dispatches of one compiled pipeline.

    fn single_axis_repeated_program(in_dim: u32, out_dim: u32, repeat_count: usize) -> (Vec<Op>, Vec<NodeId>) {
        let mut program = Vec::new();
        let weight = append(
            &mut program,
            Op::Input {
                dtype: DType::UInt8,
                shape: vec![Extent::Static(in_dim), Extent::Static(out_dim)],
                name: Some("weight".into()),
            },
        );
        let activation = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(1), Extent::Static(in_dim)],
                name: Some("activation".into()),
            },
        );
        let mut roots = Vec::with_capacity(repeat_count);
        for _ in 0..repeat_count {
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
            roots.push(sum);
        }
        (program, roots)
    }

    fn multi_axis_repeated_program(
        kv_heads: u32,
        group: u32,
        head_dim: u32,
        embed: u32,
        repeat_count: usize,
    ) -> (Vec<Op>, Vec<NodeId>) {
        let mut program = Vec::new();
        let weight = append(
            &mut program,
            Op::Input {
                dtype: DType::UInt8,
                shape: vec![
                    Extent::Static(kv_heads),
                    Extent::Static(group),
                    Extent::Static(head_dim),
                    Extent::Static(embed),
                ],
                name: Some("weight".into()),
            },
        );
        let activation = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![
                    Extent::Static(1),
                    Extent::Static(kv_heads),
                    Extent::Static(group),
                    Extent::Static(head_dim),
                ],
                name: Some("activation".into()),
            },
        );
        let mut roots = Vec::with_capacity(repeat_count);
        for _ in 0..repeat_count {
            let product = append(
                &mut program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body: ScalarOp::Multiply,
                    operands: vec![
                        (weight, IndexMap::Affine(projection(5, &[1, 2, 3, 4]))),
                        (activation, IndexMap::Affine(projection(5, &[0, 1, 2, 3]))),
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
                    in_map: IndexMap::Affine(projection(5, &[0, 1, 2, 3, 4])),
                    out_map: IndexMap::Affine(projection(5, &[0, 4])),
                    keep: Keep::Reduce,
                    name: None,
                }),
            );
            roots.push(sum);
        }
        (program, roots)
    }

    struct Shape {
        label: &'static str,
        tensor: &'static str,
        multi_axis: bool,
    }

    const SHAPES: [Shape; 6] = [
        Shape { label: "attn_k_K1536", tensor: "blk.0.attn_k.weight", multi_axis: false },
        Shape { label: "proj_K256", tensor: "blk.0.proj.weight", multi_axis: false },
        Shape { label: "attn_output_K2048_multiaxis", tensor: "blk.0.attn_output.weight", multi_axis: true },
        Shape { label: "attn_output_K4096_multiaxis", tensor: "blk.4.attn_output.weight", multi_axis: true },
        Shape { label: "ffn_down_K6144", tensor: "blk.0.ffn_down.weight", multi_axis: false },
        Shape { label: "ffn_down_K12288", tensor: "blk.15.ffn_down.weight", multi_axis: false },
    ];

    fn median_and_min(mut values: Vec<f64>) -> (f64, f64) {
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if values.len().is_multiple_of(2) {
            (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
        } else {
            values[values.len() / 2]
        };
        (median, values[0])
    }

    fn cov_percent(values: &[f64]) -> f64 {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / values.len() as f64;
        100.0 * variance.sqrt() / mean
    }

    fn run_probe_pass(pass_label: &str) {
        let path = std::path::Path::new(REAL_GEMMA4_GGUF_PATH);
        let Some((parsed, file_len, mut file)) = real_gguf_header(path) else {
            println!("real gguf file not found at {REAL_GEMMA4_GGUF_PATH}; probe skipped");
            return;
        };

        println!("=== reduction_literal_speed_probe pass={pass_label} ===");

        for shape in &SHAPES {
            let Some((weight_bytes, in_dim, out_dim)) =
                real_tensor_bytes(&mut file, &parsed, file_len, shape.tensor)
            else {
                println!("{}: tensor not present or not Q4_0, skipped", shape.label);
                continue;
            };
            let repeat_count = (MIN_STREAM_BYTES / weight_bytes.len() as f64).ceil() as usize;
            let repeat_count = repeat_count.max(1);

            let mut lcg = Lcg(4200 + in_dim as u64);
            let activation: Vec<f32> = (0..in_dim).map(|_| lcg.next_unit() * 2.0 - 1.0).collect();

            let (program, roots) = if shape.multi_axis {
                let head_dim = (in_dim / 8) as u32;
                assert_eq!(head_dim as usize * 8, in_dim, "{}: in_dim must be a whole multiple of 8", shape.label);
                multi_axis_repeated_program(2, 4, head_dim, out_dim as u32, repeat_count)
            } else {
                single_axis_repeated_program(in_dim as u32, out_dim as u32, repeat_count)
            };

            let blocks = [
                QuantizedBlock::Packed { codec: Codec::Q4_0, bytes: &weight_bytes },
                QuantizedBlock::Float32(&activation),
            ];

            // pipeline compilation is LAZY -- `metal_specialization`/
            // `reduction_literal_override` run at DISPATCH time
            // (`encode_op`'s cache lookup), not at `plan()` time, which only
            // resolves `BoundOp`s. So `PROXIMA_REDUCTION_LITERAL` must be set
            // for the duration of every EXECUTE call for the literal arm's
            // plan, not merely its `plan()` call -- matching
            // `reduction_literal_ab.rs::run_with_env`'s own scoping (the env
            // wraps `omega::execute`, never just the plan build).
            let plan_runtime = omega::plan(&program, &[], &blocks, &roots, NumericPolicy::default())
                .expect("plan compiles for the repeated matvec program (runtime arm)");
            let plan_literal = temp_env::with_var("PROXIMA_REDUCTION_LITERAL", Some("1"), || {
                omega::plan(&program, &[], &blocks, &roots, NumericPolicy::default())
                    .expect("plan compiles for the repeated matvec program (literal arm)")
            });

            let run_runtime = |plan: &omega::Plan| {
                temp_env::with_var("PROXIMA_REDUCTION_LITERAL", None::<&str>, || {
                    omega::execute_plan_timed(plan, &blocks)
                        .expect("metal executes the runtime-bound repeated matvec")
                })
            };
            let run_literal = |plan: &omega::Plan| {
                temp_env::with_var("PROXIMA_REDUCTION_LITERAL", Some("1"), || {
                    omega::execute_plan_timed(plan, &blocks)
                        .expect("metal executes the literal-bound repeated matvec")
                })
            };

            let (evaluated_runtime, _) = run_runtime(&plan_runtime);
            let (evaluated_literal, _) = run_literal(&plan_literal);

            let compiled_keys = omega::pipeline_cache_keys();
            eprintln!("{}: compiled_keys={compiled_keys:?}", shape.label);
            let literal_marker = format!("_rl{in_dim}");
            let literal_has_marker = compiled_keys.iter().any(|key| key.contains(&literal_marker));
            println!(
                "{}: repeat_count={repeat_count} weight_bytes={} stream_bytes={:.3}GB literal_key_marker={literal_marker} present={literal_has_marker}",
                shape.label,
                weight_bytes.len(),
                repeat_count as f64 * weight_bytes.len() as f64 / 1e9,
            );
            assert!(
                literal_has_marker,
                "{}: expected a compiled pipeline-cache key containing {literal_marker} once the literal arm executed",
                shape.label
            );
            let mut compared_words = 0usize;
            let mut mismatched_words = 0usize;
            for &root in &roots {
                let (runtime_values, _) = evaluated_runtime.get(root).expect("runtime output present");
                let (literal_values, _) = evaluated_literal.get(root).expect("literal output present");
                for (runtime_value, literal_value) in runtime_values.iter().zip(literal_values.iter()) {
                    compared_words += 1;
                    if runtime_value.to_bits() != literal_value.to_bits() {
                        mismatched_words += 1;
                    }
                }
            }
            let dequantized = dequantize_rows(&weight_bytes, in_dim, out_dim.min(64));
            let oracle = expected_output(&dequantized, &activation);
            let (first_runtime, _) = evaluated_runtime.get(roots[0]).expect("first runtime output");
            let max_abs_vs_oracle = first_runtime
                .iter()
                .zip(oracle.iter())
                .map(|(got, want)| (got - want).abs())
                .fold(0.0f32, f32::max);
            println!(
                "{}: compared_words={compared_words} mismatched_words={mismatched_words} first_dispatch_vs_cpu_oracle_max_abs={max_abs_vs_oracle}",
                shape.label
            );
            assert_eq!(
                mismatched_words, 0,
                "{}: literal and runtime-bound arms produced {mismatched_words}/{compared_words} differing words",
                shape.label
            );

            // timed samples: interleaved A,B,A,B pairs, >=12 kept per arm,
            // 2 warmup pairs discarded.
            let mut runtime_ns: Vec<f64> = Vec::with_capacity(KEPT_PAIRS);
            let mut literal_ns: Vec<f64> = Vec::with_capacity(KEPT_PAIRS);
            let mut paired_diff_ns: Vec<f64> = Vec::with_capacity(KEPT_PAIRS);
            for pair_index in 0..TOTAL_PAIRS {
                let (_, runtime_gpu_ns) = run_runtime(&plan_runtime);
                let (_, literal_gpu_ns) = run_literal(&plan_literal);
                let runtime_per_dispatch = runtime_gpu_ns as f64 / repeat_count as f64;
                let literal_per_dispatch = literal_gpu_ns as f64 / repeat_count as f64;
                if pair_index >= WARMUP_PAIRS {
                    runtime_ns.push(runtime_per_dispatch);
                    literal_ns.push(literal_per_dispatch);
                    paired_diff_ns.push(literal_per_dispatch - runtime_per_dispatch);
                }
            }

            let (runtime_median, runtime_min) = median_and_min(runtime_ns.clone());
            let (literal_median, literal_min) = median_and_min(literal_ns.clone());
            let runtime_cov = cov_percent(&runtime_ns);
            let literal_cov = cov_percent(&literal_ns);
            let runtime_gbs = weight_bytes.len() as f64 / (runtime_median / 1e9) / 1e9;
            let literal_gbs = weight_bytes.len() as f64 / (literal_median / 1e9) / 1e9;
            let (diff_median, diff_min) = median_and_min(paired_diff_ns.clone());
            let diff_max = paired_diff_ns.iter().cloned().fold(f64::MIN, f64::max);

            println!(
                "{}: runtime_bound  median={runtime_median:.1}ns min={runtime_min:.1}ns CoV={runtime_cov:.2}% GBps={runtime_gbs:.2}",
                shape.label
            );
            println!(
                "{}: literal        median={literal_median:.1}ns min={literal_min:.1}ns CoV={literal_cov:.2}% GBps={literal_gbs:.2}",
                shape.label
            );
            println!(
                "{}: paired(literal-runtime) median={diff_median:.1}ns min={diff_min:.1}ns max={diff_max:.1}ns n={}",
                shape.label,
                paired_diff_ns.len()
            );
        }
    }

    if let Ok(output) = std::process::Command::new("pgrep").args(["-fl", "cargo|rustc"]).output()
        && !output.stdout.is_empty()
    {
        eprintln!(
            "WARNING: a cargo/rustc process is running elsewhere on this box:\n{}",
            String::from_utf8_lossy(&output.stdout)
        );
    }

    run_probe_pass("run1");
}
