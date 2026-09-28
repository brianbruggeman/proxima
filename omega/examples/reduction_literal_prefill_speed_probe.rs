//! Isolated production-shape probe for `c4-7-reduction-literal.md`'s
//! `PROXIMA_REDUCTION_LITERAL` bake, PREFILL-shaped (`token_total > 1`):
//! the multi-row packed-row-blocked Q4_0 matmul body
//! (`push_packed_row_multi_row_body`'s generic `else` arm,
//! `omega/src/msl/elementwise_reduce_core.rs:1210-1229`), at two real
//! gemma4-E2B weight shapes (`blk.0.attn_k` K=1536 rows=256, `blk.15.
//! ffn_down` K=12288 rows=1536) and two real prompt token counts (26, 510),
//! two arms per shape/token combo in ONE process -- runtime-bound
//! (`PROXIMA_REDUCTION_LITERAL` unset) and literal (`=1`).
//!
//! Sibling of `reduction_literal_speed_probe.rs` (the decode-shaped,
//! `token_total == 1` probe): same real-weight-loading, repeat_count,
//! `execute_plan_timed`, bit-exact-output-check methodology, but the
//! activation tensor carries a `tokens` axis instead of a fixed unit axis,
//! and the program shape mirrors `omega/src/msl/tests.rs`'s
//! `packed_row_multi_token_op` (weight declared `[rows, k]`, matching the
//! real Q4_0 GGUF row-major byte layout `[out_dim][in_dim]` exactly) instead
//! of the decode probe's `[in_dim, out_dim]` single-row convention.
//!
//! `repeat_count` is sized off the STREAMED bytes per single (non-repeated)
//! op, not off `weight_bytes` alone: the multi-row body tiles the token
//! axis in groups of `sized::PACKED_ROW_ACTIVATION_GROUP` (8, `omega-
//! runtime.toml`'s `[packed_row_multi_activation]`), re-streaming the full
//! weight matrix once per token-group, so one op already streams
//! `weight_bytes * ceil(tokens / 8)` bytes -- `repeat_count =
//! ceil(2e9 / (weight_bytes * ceil(tokens / 8)))`.

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
        "reduction_literal_prefill_speed_probe requires --features metal,metal-reduction-literal,instrument on macOS"
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
    const TOKEN_GROUP: usize = 8;
    const WARMUP_PAIRS: usize = 2;
    const KEPT_PAIRS: usize = 12;
    const TOTAL_PAIRS: usize = WARMUP_PAIRS + KEPT_PAIRS;

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
                "real_tensor_bytes: {name} is {:?} in this file, not Q4_0 -- probe skipped, not faked",
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

    // `weight[row] . activation[token]`, for a `rows_checked x tokens_checked`
    // corner of the full output -- an independent oracle, never omega's own
    // evaluator, per `feedback_incumbent_is_llama_not_us.md`.
    fn expected_output_multi_row(
        rows: &[Vec<f32>],
        activation_rows: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        activation_rows
            .iter()
            .map(|activation| {
                rows.iter()
                    .map(|row| {
                        row.iter()
                            .zip(activation.iter())
                            .map(|(weight, value)| weight * value)
                            .sum::<f32>()
                    })
                    .collect()
            })
            .collect()
    }

    fn multi_row_repeated_program(
        rows: u32,
        k: u32,
        tokens: u32,
        repeat_count: usize,
    ) -> (Vec<Op>, Vec<NodeId>) {
        let mut program = Vec::new();
        let weight = append(
            &mut program,
            Op::Input {
                dtype: DType::UInt8,
                shape: vec![Extent::Static(rows), Extent::Static(k)],
                name: Some("weight".into()),
            },
        );
        let activation = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(tokens), Extent::Static(k)],
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
                        (activation, IndexMap::Affine(projection(3, &[0, 2]))),
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
                    out_map: IndexMap::Affine(projection(3, &[0, 1])),
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
    }

    const SHAPES: [Shape; 2] = [
        Shape { label: "attn_k_K1536", tensor: "blk.0.attn_k.weight" },
        Shape { label: "ffn_down_K12288", tensor: "blk.15.ffn_down.weight" },
    ];
    const TOKEN_TOTALS: [u32; 2] = [26, 510];

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

    if let Ok(output) = std::process::Command::new("pgrep").args(["-fl", "cargo|rustc"]).output()
        && !output.stdout.is_empty()
    {
        eprintln!(
            "WARNING: a cargo/rustc process is running elsewhere on this box:\n{}",
            String::from_utf8_lossy(&output.stdout)
        );
    }

    let path = std::path::Path::new(REAL_GEMMA4_GGUF_PATH);
    let Some((parsed, file_len, mut file)) = real_gguf_header(path) else {
        println!("real gguf file not found at {REAL_GEMMA4_GGUF_PATH}; probe skipped");
        return;
    };

    println!("=== reduction_literal_prefill_speed_probe ===");

    for shape in &SHAPES {
        let Some((weight_bytes, in_dim, out_dim)) =
            real_tensor_bytes(&mut file, &parsed, file_len, shape.tensor)
        else {
            println!("{}: tensor not present or not Q4_0, skipped", shape.label);
            continue;
        };

        for &tokens in &TOKEN_TOTALS {
            let token_groups = (tokens as usize).div_ceil(TOKEN_GROUP);
            let bytes_per_op = weight_bytes.len() * token_groups;
            let repeat_count = (MIN_STREAM_BYTES / bytes_per_op as f64).ceil().max(1.0) as usize;
            let stream_gb = repeat_count as f64 * bytes_per_op as f64 / 1e9;

            let mut lcg = Lcg(4200 + in_dim as u64 + tokens as u64 * 97);
            let activation: Vec<f32> = (0..(in_dim * tokens as usize))
                .map(|_| lcg.next_unit() * 2.0 - 1.0)
                .collect();

            let (program, roots) =
                multi_row_repeated_program(out_dim as u32, in_dim as u32, tokens, repeat_count);

            let blocks = [
                QuantizedBlock::Packed { codec: Codec::Q4_0, bytes: &weight_bytes },
                QuantizedBlock::Float32(&activation),
            ];

            let plan_runtime = omega::plan(&program, &[], &blocks, &roots, NumericPolicy::default())
                .expect("plan compiles for the repeated prefill matmul program (runtime arm)");
            let plan_literal = temp_env::with_var("PROXIMA_REDUCTION_LITERAL", Some("1"), || {
                omega::plan(&program, &[], &blocks, &roots, NumericPolicy::default())
                    .expect("plan compiles for the repeated prefill matmul program (literal arm)")
            });

            let run_runtime = |plan: &omega::Plan| {
                temp_env::with_var("PROXIMA_REDUCTION_LITERAL", None::<&str>, || {
                    omega::execute_plan_timed(plan, &blocks)
                        .expect("metal executes the runtime-bound repeated prefill matmul")
                })
            };
            let run_literal = |plan: &omega::Plan| {
                temp_env::with_var("PROXIMA_REDUCTION_LITERAL", Some("1"), || {
                    omega::execute_plan_timed(plan, &blocks)
                        .expect("metal executes the literal repeated prefill matmul")
                })
            };

            let label = format!("{}_t{tokens}", shape.label);

            let (evaluated_runtime, _) = run_runtime(&plan_runtime);
            let (evaluated_literal, _) = run_literal(&plan_literal);

            let compiled_keys = omega::pipeline_cache_keys();
            let literal_marker = format!("_rl{in_dim}");
            let literal_has_marker = compiled_keys.iter().any(|key| key.contains(&literal_marker));
            println!(
                "{label}: repeat_count={repeat_count} token_groups={token_groups} weight_bytes={} bytes_per_op={bytes_per_op} stream_GB={stream_gb:.3} literal_key_marker={literal_marker} present={literal_has_marker}",
                weight_bytes.len(),
            );
            assert!(
                literal_has_marker,
                "{label}: expected a compiled pipeline-cache key containing {literal_marker} once the literal arm executed"
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
            let rows_checked = out_dim.min(32);
            let tokens_checked = (tokens as usize).min(4);
            let dequantized = dequantize_rows(&weight_bytes, in_dim, rows_checked);
            let activation_rows: Vec<Vec<f32>> = (0..tokens_checked)
                .map(|token_index| activation[token_index * in_dim..(token_index + 1) * in_dim].to_vec())
                .collect();
            let oracle = expected_output_multi_row(&dequantized, &activation_rows);
            let (first_runtime, _) = evaluated_runtime.get(roots[0]).expect("first runtime output");
            // output layout is [tokens, rows] row-major (out_map projection(3,&[0,1])).
            let mut max_abs_vs_oracle = 0.0f32;
            for (token_index, oracle_row) in oracle.iter().enumerate() {
                for (row_index, &want) in oracle_row.iter().enumerate() {
                    let got = first_runtime[token_index * out_dim + row_index];
                    max_abs_vs_oracle = max_abs_vs_oracle.max((got - want).abs());
                }
            }
            println!(
                "{label}: compared_words={compared_words} mismatched_words={mismatched_words} oracle_rows_checked={rows_checked} oracle_tokens_checked={tokens_checked} first_dispatch_vs_cpu_oracle_max_abs={max_abs_vs_oracle}"
            );
            assert_eq!(
                mismatched_words, 0,
                "{label}: literal and runtime-bound arms produced {mismatched_words}/{compared_words} differing words"
            );

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
            let (diff_median, diff_min) = median_and_min(paired_diff_ns.clone());
            let diff_max = paired_diff_ns.iter().cloned().fold(f64::MIN, f64::max);
            let faster_count = paired_diff_ns.iter().filter(|&&value| value < 0.0).count();

            println!(
                "{label}: runtime_bound  median={runtime_median:.1}ns min={runtime_min:.1}ns CoV={runtime_cov:.2}%"
            );
            println!(
                "{label}: literal        median={literal_median:.1}ns min={literal_min:.1}ns CoV={literal_cov:.2}%"
            );
            println!(
                "{label}: paired(literal-runtime) median={diff_median:.1}ns min={diff_min:.1}ns max={diff_max:.1}ns n={} literal_faster={faster_count}/{}",
                paired_diff_ns.len(),
                paired_diff_ns.len()
            );
        }
    }
}
