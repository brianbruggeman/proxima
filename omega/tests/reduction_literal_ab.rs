//! `c4-7-reduction-literal.md` AC2-AC5: `PROXIMA_REDUCTION_LITERAL=1` bakes a
//! packed-row-blocked op's flattened reduction length as a compiled literal.
//! Real gemma4-E2B `Q4_0` weight rows from the ollama blob (per
//! guiding-principles §9, the same file `q4_0_tiled_gemm_batched_run8_parity.rs`
//! already reads) drive three real collision pairs from the repro's own
//! `ops_C.tsv`: `blk.0.ffn_down.weight` (K=6144) vs `blk.15.ffn_down.weight`
//! (K=12288), `blk.0.attn_k.weight` (K=1536) vs `blk.0.proj.weight` (K=256),
//! and `blk.0.attn_output.weight` (K=2048=256x8) vs `blk.4.attn_output.weight`
//! (K=4096=512x8) folded across a synthetic (kv_head, group, head_dim) split
//! that reproduces the repro's rank-5 `ax0_4` shape on real bytes.
//!
//! Skips (does not fail) when the real file or a named tensor is not present
//! on this host, matching every other real-checkpoint test in this crate.

#![cfg(all(
    feature = "metal",
    feature = "metal-reduction-literal",
    feature = "instrument",
    target_os = "macos"
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::{Read, Seek, SeekFrom};

use proxima_gguf::parser::{GgufEvent, GgufParser};
use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::quant::q4_0;
use proxima_gguf::types::GgmlType;
use proxima_primitives::Codec;
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce, ReduceInit,
    ScalarOp, append, projection,
};

const REAL_GEMMA4_GGUF_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";

// Real weight rows compared per tensor -- bounded the same way
// `q4_0_real_checkpoint_parity.rs::ROWS_TO_CHECK` bounds its own device
// parity gate, so this stays a reasonable single-test runtime.
const ROWS_TO_CHECK: usize = 24;

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
                    } => {
                        completion = Some((data_offset, alignment));
                    }
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
    file.seek(SeekFrom::Start(range.start))
        .expect("seek to tensor data");
    file.read_exact(&mut buf).expect("read exact tensor byte range");
    Some((buf, in_dim, out_dim))
}

/// `[tokens, in_dim] x [in_dim, out_dim] -> [tokens, out_dim]`, single reduce
/// axis -- `q4_0_tiled_gemm_batched_run8_parity.rs::matmul_program`'s own
/// shape, `tokens=1` (decode-shaped) here so the tiled-gemm path never
/// engages and every op stays on the row-blocked packed path this spec's
/// helper actually renders.
fn single_axis_matmul_program(in_dim: u32, out_dim: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::UInt8,
            shape: vec![Extent::Static(in_dim), Extent::Static(out_dim)],
            name: None,
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(1), Extent::Static(in_dim)],
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
    (program, sum)
}

/// Rank-5, `output_axes=[0,4]` -- the repro's own `r5_ax0_4` shape:
/// `classify_packed_row_block`'s contiguous-fold rule folds `(kv_head,
/// group, head_dim)` into one flattened reduction of length `kv_heads *
/// group * head_dim`, matching `proxima_tensor::bind::tests::
/// correct_packed_matmul_layouts_derives_ggml_native_strides_for_a_multi_axis_contraction_group`'s
/// own construction. `native_packed_layout` rebuilds the packed weight's
/// `Layout` purely from `extents`/`output_axes` (see that function's own
/// doc), so declaring the weight's OWN local axis order as `[kv_head, group,
/// head_dim, embed]` and reading it with a plain identity `projection` is
/// enough -- no composed affine map is needed the way the internal bind test
/// used one, because that test was proving the affine-map path specifically,
/// not building a fixture for this one.
fn multi_axis_matmul_program(kv_heads: u32, group: u32, head_dim: u32, embed: u32) -> (Vec<Op>, NodeId) {
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
            name: None,
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
            name: None,
        },
    );
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
    (program, sum)
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

/// Independent oracle: dot each dequantized row against the flat activation
/// -- computed without going through either `proxima_tensor::cpu`'s own
/// quantized matmul path or omega's Metal emitter, same posture as
/// `attn_multi_axis_tiled_gemm_parity.rs::expected_output`.
fn expected_output(rows: &[Vec<f32>], in_dim: usize, activation: &[f32]) -> Vec<f32> {
    activation
        .chunks_exact(in_dim)
        .flat_map(|token_activation| {
            rows.iter().map(move |row| {
                row.iter()
                    .zip(token_activation.iter())
                    .map(|(weight, value)| weight * value)
                    .sum::<f32>()
            })
        })
        .collect()
}

struct ErrorSummary {
    max_abs: f32,
}

fn compare_to_oracle(actual: &[f32], oracle: &[f32]) -> ErrorSummary {
    let mut max_abs = 0.0f32;
    for (&got, &want) in actual.iter().zip(oracle.iter()) {
        max_abs = max_abs.max((got - want).abs());
    }
    ErrorSummary { max_abs }
}

fn differing_word_count(a: &[f32], b: &[f32]) -> usize {
    a.iter().zip(b.iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

/// Runs `program`/`sum` once under the given `PROXIMA_REDUCTION_LITERAL`
/// env value against `blocks`, returning the root output.
fn run_with_env(
    program: &[Op],
    sum: NodeId,
    blocks: &[QuantizedBlock<'_>],
    reduction_literal_env: Option<&str>,
    policy: NumericPolicy,
) -> Vec<f32> {
    temp_env::with_var("PROXIMA_REDUCTION_LITERAL", reduction_literal_env, || {
        omega::execute(program, &[], blocks, &[sum], policy)
            .expect("metal executes the packed-row-blocked matvec")
            .root()
            .to_vec()
    })
}

/// One collision pair's shared harness: `label` names the pair for
/// diagnostics, `weight_bytes`/`in_dim`/`rows` describe the real Q4_0 tensor,
/// `activation` is the flat `[1, in_dim]` input, `program`/`sum` is the op
/// graph reading it (single- or multi-axis).
struct CollisionOperand {
    label: String,
    program: Vec<Op>,
    sum: NodeId,
    weight_bytes: Vec<u8>,
    in_dim: usize,
    rows: usize,
    activation: Vec<f32>,
}

impl CollisionOperand {
    fn blocks(&self) -> [QuantizedBlock<'_>; 2] {
        [
            QuantizedBlock::Packed {
                codec: Codec::Q4_0,
                bytes: &self.weight_bytes,
            },
            QuantizedBlock::Float32(&self.activation),
        ]
    }
}

/// AC3: `bit_exact`/`llama_relaxed`/`fast` -- every policy's literal-baked
/// output must equal its own runtime-bound output word for word. A nonzero
/// count is reported, not silently weakened into a looser assertion.
fn assert_bit_exact_and_oracle_agreement(operand: &CollisionOperand) {
    let blocks = operand.blocks();
    let rows = dequantize_rows(&operand.weight_bytes, operand.in_dim, operand.rows);
    let oracle = expected_output(&rows, operand.in_dim, &operand.activation);

    for policy in [
        NumericPolicy::bit_exact(),
        NumericPolicy::llama_relaxed(),
        NumericPolicy::fast(),
    ] {
        let runtime_bound = run_with_env(&operand.program, operand.sum, &blocks, None, policy);
        let literal = run_with_env(&operand.program, operand.sum, &blocks, Some("1"), policy);
        assert_eq!(
            runtime_bound.len(),
            operand.rows,
            "{}: degenerate gate, runtime-bound produced the wrong element count",
            operand.label
        );
        assert_eq!(
            literal.len(),
            operand.rows,
            "{}: degenerate gate, literal produced the wrong element count",
            operand.label
        );
        let diff_words = differing_word_count(&literal, &runtime_bound);
        let literal_vs_oracle = compare_to_oracle(&literal, &oracle);
        let runtime_vs_oracle = compare_to_oracle(&runtime_bound, &oracle);
        eprintln!(
            "{} policy={policy:?}: differing_words={diff_words}/{} literal_vs_oracle_max_abs={} \
             runtime_vs_oracle_max_abs={}",
            operand.label,
            operand.rows,
            literal_vs_oracle.max_abs,
            runtime_vs_oracle.max_abs,
        );
        assert_eq!(
            diff_words, 0,
            "{}: policy={policy:?} PROXIMA_REDUCTION_LITERAL=1 produced {diff_words} words \
             differing from the runtime-bound default -- the baked literal must reproduce the \
             same loop bound, never a different one",
            operand.label
        );
        assert!(
            literal_vs_oracle.max_abs < 5e-2,
            "{}: policy={policy:?} literal path disagrees with the independent dequantize+dot \
             oracle (max_abs={})",
            operand.label,
            literal_vs_oracle.max_abs
        );
    }
}

fn run_on_fresh_thread<F: FnOnce() + Send + 'static>(body: F) {
    std::thread::Builder::new()
        .spawn(body)
        .expect("spawn a fresh thread for pipeline-cache isolation")
        .join()
        .expect("fresh-thread pipeline-cache body panicked");
}

/// AC2: both insertion orders through ONE `PIPELINE_CACHE` (one per spawned
/// thread -- there is no way to reset a thread's own cache from outside it,
/// so a fresh thread is this test's own cache reset, documented here rather
/// than assumed) must produce exactly 2 distinct keys and exactly 2 misses
/// (`PROXIMA_PIPELINE_KEY_AUDIT=1`'s own `audited` counter only increments on
/// a HIT, so `audited == hits_per_op * 2` after `1 miss + hits_per_op hits`
/// each proves the miss count indirectly: `pipeline_cache_keys().len() == 2`
/// is the direct miss-count proof, since every key only ever enters the
/// cache on the miss that compiles it).
fn assert_insertion_order_independent(first: &'static str, second: &'static str, build: fn(&str) -> CollisionOperand) {
    const HITS_PER_OP: u32 = 3;
    for order in [[first, second], [second, first]] {
        run_on_fresh_thread(move || {
            temp_env::with_var("PROXIMA_REDUCTION_LITERAL", Some("1"), || {
                temp_env::with_var("PROXIMA_PIPELINE_KEY_AUDIT", Some("1"), || {
                    for label in order {
                        let operand = build(label);
                        let blocks = operand.blocks();
                        for _ in 0..(1 + HITS_PER_OP) {
                            omega::execute(
                                &operand.program,
                                &[],
                                &blocks,
                                &[operand.sum],
                                NumericPolicy::default(),
                            )
                            .expect("metal executes the packed-row-blocked matvec");
                        }
                    }
                    let keys = omega::pipeline_cache_keys();
                    let (audited, mismatched) = omega::pipeline_key_audit_counts();
                    eprintln!(
                        "insertion order {order:?}: distinct_keys={} keys={keys:?} audited={audited} mismatched={mismatched}",
                        keys.len()
                    );
                    assert_eq!(
                        keys.len(),
                        2,
                        "order {order:?}: expected exactly 2 distinct pipeline-cache keys (one \
                         miss per distinct shape), got {}: {keys:?}",
                        keys.len()
                    );
                    assert_eq!(
                        audited,
                        u64::from(HITS_PER_OP) * 2,
                        "order {order:?}: expected {} audited hits (3 hits x 2 ops)",
                        u64::from(HITS_PER_OP) * 2
                    );
                    assert_eq!(
                        mismatched, 0,
                        "order {order:?}: pipeline key audit found {mismatched} key-completeness mismatches"
                    );
                });
            });
        });
    }
}

fn ffn_down_operand(label: &str) -> CollisionOperand {
    let path = std::path::Path::new(REAL_GEMMA4_GGUF_PATH);
    let (parsed, file_len, mut file) = real_gguf_header(path).expect("real gguf file present");
    let (weight_bytes, in_dim, out_dim) = real_tensor_bytes(&mut file, &parsed, file_len, label)
        .unwrap_or_else(|| panic!("{label} present and Q4_0"));
    let rows = ROWS_TO_CHECK.min(out_dim);
    let sliced_weight = weight_bytes[..rows * (in_dim / q4_0::QK4_0) * q4_0::BLOCK_BYTES].to_vec();
    let mut lcg = Lcg(7001 + in_dim as u64);
    let activation: Vec<f32> = (0..in_dim).map(|_| lcg.next_unit() * 2.0 - 1.0).collect();
    let (program, sum) = single_axis_matmul_program(in_dim as u32, rows as u32);
    CollisionOperand {
        label: label.to_string(),
        program,
        sum,
        weight_bytes: sliced_weight,
        in_dim,
        rows,
        activation,
    }
}

fn attn_output_multi_axis_operand(label: &str) -> CollisionOperand {
    let path = std::path::Path::new(REAL_GEMMA4_GGUF_PATH);
    let (parsed, file_len, mut file) = real_gguf_header(path).expect("real gguf file present");
    let (weight_bytes, in_dim, out_dim) = real_tensor_bytes(&mut file, &parsed, file_len, label)
        .unwrap_or_else(|| panic!("{label} present and Q4_0"));
    let head_dim = in_dim / 8;
    assert_eq!(head_dim * 8, in_dim, "{label}: in_dim must be a whole multiple of 8 (kv_heads*group)");
    let rows = ROWS_TO_CHECK.min(out_dim);
    let sliced_weight = weight_bytes[..rows * (in_dim / q4_0::QK4_0) * q4_0::BLOCK_BYTES].to_vec();
    let mut lcg = Lcg(9001 + in_dim as u64);
    let activation: Vec<f32> = (0..in_dim).map(|_| lcg.next_unit() * 2.0 - 1.0).collect();
    // kv_heads=2, group=4: 2*4=8, matching every real gemma4-E2B attn_output
    // in_dim's own "x8" factor (`c4-7-reduction-literal.md` section 4's
    // worked example) regardless of which layer's head_dim (256 or 512) it
    // multiplies.
    let (program, sum) = multi_axis_matmul_program(2, 4, head_dim as u32, rows as u32);
    CollisionOperand {
        label: label.to_string(),
        program,
        sum,
        weight_bytes: sliced_weight,
        in_dim,
        rows,
        activation,
    }
}

fn real_gguf_present() -> bool {
    let present = real_gguf_header(std::path::Path::new(REAL_GEMMA4_GGUF_PATH)).is_some();
    if !present {
        eprintln!("real gguf file not found at {REAL_GEMMA4_GGUF_PATH}; test skipped");
    }
    present
}

#[test]
fn q4_0_ffn_down_narrow_vs_wide_collision() {
    if !real_gguf_present() {
        return;
    }
    let narrow = ffn_down_operand("blk.0.ffn_down.weight");
    let wide = ffn_down_operand("blk.15.ffn_down.weight");
    assert_eq!(narrow.in_dim, 6144, "degenerate fixture: narrow ffn_down in_dim drifted");
    assert_eq!(wide.in_dim, 12288, "degenerate fixture: wide ffn_down in_dim drifted");
    assert_bit_exact_and_oracle_agreement(&narrow);
    assert_bit_exact_and_oracle_agreement(&wide);
    assert_insertion_order_independent(
        "blk.0.ffn_down.weight",
        "blk.15.ffn_down.weight",
        ffn_down_operand,
    );
}

#[test]
fn q4_0_attn_k_vs_proj_collision() {
    if !real_gguf_present() {
        return;
    }
    let wide = ffn_down_operand("blk.0.attn_k.weight");
    let narrow = ffn_down_operand("blk.0.proj.weight");
    assert_eq!(wide.in_dim, 1536, "degenerate fixture: attn_k in_dim drifted");
    assert_eq!(narrow.in_dim, 256, "degenerate fixture: proj in_dim drifted");
    assert_bit_exact_and_oracle_agreement(&wide);
    assert_bit_exact_and_oracle_agreement(&narrow);
    assert_insertion_order_independent(
        "blk.0.attn_k.weight",
        "blk.0.proj.weight",
        ffn_down_operand,
    );
}

#[test]
fn q4_0_attn_output_multi_axis_fold_collision() {
    if !real_gguf_present() {
        return;
    }
    let narrow = attn_output_multi_axis_operand("blk.0.attn_output.weight");
    let wide = attn_output_multi_axis_operand("blk.4.attn_output.weight");
    assert_eq!(narrow.in_dim, 2048, "degenerate fixture: narrow attn_output in_dim drifted");
    assert_eq!(wide.in_dim, 4096, "degenerate fixture: wide attn_output in_dim drifted");
    assert_bit_exact_and_oracle_agreement(&narrow);
    assert_bit_exact_and_oracle_agreement(&wide);
    assert_insertion_order_independent(
        "blk.0.attn_output.weight",
        "blk.4.attn_output.weight",
        attn_output_multi_axis_operand,
    );
}

/// [`single_axis_matmul_program`]'s own shape with `tokens` free instead of
/// pinned to 1 -- `tokens > 1` is what routes `push_packed_row_blocked_body`
/// into [`crate::msl::push_packed_row_multi_row_body`] at all
/// (`packed_row_block_token_total(block, extents) > 1`,
/// `packed_row_blocked_ggml.rs`), the family of render sites this AC
/// converts to [`packed_row_reduction_bound_token`] alongside the single-row
/// body.
fn multi_token_matmul_program(tokens: u32, in_dim: u32, out_dim: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::UInt8,
            shape: vec![Extent::Static(in_dim), Extent::Static(out_dim)],
            name: None,
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(tokens), Extent::Static(in_dim)],
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
    (program, sum)
}

/// Runs one `(program, sum)` under `reduction_literal_env`, with an optional
/// second env var set for the duration of the SAME run -- lets one call site
/// cover the plain generic arm (`extra_env = None`), the `Q4_0` header-hoist
/// arm (`PROXIMA_Q4_0_MULTI_ROW_HOIST=1`), and the unrolled arm
/// (`PROXIMA_MULTI_ROW_UNROLL=1`) without three copies of the same body.
fn run_multi_row(
    program: &[Op],
    sum: NodeId,
    blocks: &[QuantizedBlock<'_>],
    policy: NumericPolicy,
    extra_env: Option<(&str, &str)>,
    reduction_literal_env: Option<&str>,
) -> Vec<f32> {
    // `PROXIMA_TILED_GEMM_Q4_0` defaults ON now, and the one caller of this
    // helper runs at `tokens == TILED_GEMM_MIN_TOKENS` -- force it off so
    // Q4_0 still routes through `push_packed_row_multi_row_body`'s generic/
    // hoist/unroll arms this AC targets, not the tiled path.
    let inner = || {
        temp_env::with_var("PROXIMA_TILED_GEMM_Q4_0", Some("0"), || {
            temp_env::with_var("PROXIMA_REDUCTION_LITERAL", reduction_literal_env, || {
                omega::execute(program, &[], blocks, &[sum], policy)
                    .expect("metal executes the multi-row packed matvec")
                    .root()
                    .to_vec()
            })
        })
    };
    match extra_env {
        Some((name, value)) => temp_env::with_var(name, Some(value), inner),
        None => inner(),
    }
}

/// AC2/AC3 extension (`c4-7-reduction-literal.md`): the `token_total > 1`
/// family of render sites -- [`crate::msl::push_packed_row_multi_row_body`]'s
/// own generic arm, its `Q4_0` header-hoist fast arm
/// (`push_packed_row_multi_row_q4_0_body`, `PROXIMA_Q4_0_MULTI_ROW_HOIST=1`),
/// and its unrolled arm (`push_packed_row_multi_row_unroll_body`,
/// `PROXIMA_MULTI_ROW_UNROLL=1`) -- all read the reduction bound through
/// [`packed_row_reduction_bound_token`], never `u.reduction_total` directly.
/// A real, prefill-shaped (`token_total = PACKED_ROW_ACTIVATION_GROUP`, so
/// one simdgroup fold, no token-axis tiling) `Q4_0` `blk.0.ffn_down.weight`
/// matvec proves all three arms word for word: literal-baked
/// vs runtime-bound output must match exactly under every `NumericPolicy`
/// preset, and both must agree with an independent dequantize+dot oracle.
/// `q4_0_ffn_down_narrow_vs_wide_collision` above already proves this same
/// tensor's `token_total == 1` single-row body; this is the `token_total >
/// 1` sibling the spec names as still open.
#[test]
fn q4_0_multi_row_prefill_shaped_literal_matches_runtime() {
    if !real_gguf_present() {
        return;
    }
    let tokens = omega::sized::PACKED_ROW_ACTIVATION_GROUP as u32;
    assert!(
        tokens > 1,
        "degenerate fixture: PACKED_ROW_ACTIVATION_GROUP must exceed 1 for this to reach \
         push_packed_row_multi_row_body at all"
    );

    let label = "blk.0.ffn_down.weight";
    let path = std::path::Path::new(REAL_GEMMA4_GGUF_PATH);
    let (parsed, file_len, mut file) = real_gguf_header(path).expect("real gguf file present");
    let (weight_bytes, in_dim, out_dim) = real_tensor_bytes(&mut file, &parsed, file_len, label)
        .unwrap_or_else(|| panic!("{label} present and Q4_0"));
    let rows = ROWS_TO_CHECK.min(out_dim);
    let sliced_weight = weight_bytes[..rows * (in_dim / q4_0::QK4_0) * q4_0::BLOCK_BYTES].to_vec();
    let mut lcg = Lcg(31_337 + in_dim as u64);
    let activation: Vec<f32> = (0..(tokens as usize * in_dim))
        .map(|_| lcg.next_unit() * 2.0 - 1.0)
        .collect();
    let (program, sum) = multi_token_matmul_program(tokens, in_dim as u32, rows as u32);
    let dequantized = dequantize_rows(&sliced_weight, in_dim, rows);
    let oracle = expected_output(&dequantized, in_dim, &activation);
    let blocks = [
        QuantizedBlock::Packed {
            codec: Codec::Q4_0,
            bytes: &sliced_weight,
        },
        QuantizedBlock::Float32(&activation),
    ];
    let expected_len = tokens as usize * rows;

    for (arm, extra_env) in [
        ("generic", None),
        ("q4_0_hoist", Some(("PROXIMA_Q4_0_MULTI_ROW_HOIST", "1"))),
        ("unroll", Some(("PROXIMA_MULTI_ROW_UNROLL", "1"))),
    ] {
        for policy in [
            NumericPolicy::bit_exact(),
            NumericPolicy::llama_relaxed(),
            NumericPolicy::fast(),
        ] {
            let runtime_bound = run_multi_row(&program, sum, &blocks, policy, extra_env, None);
            let literal = run_multi_row(&program, sum, &blocks, policy, extra_env, Some("1"));
            assert_eq!(
                runtime_bound.len(),
                expected_len,
                "multi_row {arm} policy={policy:?}: degenerate gate, runtime-bound wrong element count"
            );
            assert_eq!(
                literal.len(),
                expected_len,
                "multi_row {arm} policy={policy:?}: degenerate gate, literal wrong element count"
            );
            let diff_words = differing_word_count(&literal, &runtime_bound);
            let literal_vs_oracle = compare_to_oracle(&literal, &oracle);
            eprintln!(
                "multi_row {arm} policy={policy:?}: differing_words={diff_words}/{expected_len} \
                 literal_vs_oracle_max_abs={}",
                literal_vs_oracle.max_abs,
            );
            assert_eq!(
                diff_words, 0,
                "multi_row {arm} policy={policy:?}: PROXIMA_REDUCTION_LITERAL=1 produced \
                 {diff_words} words differing from the runtime-bound default"
            );
            assert!(
                literal_vs_oracle.max_abs < 5e-2,
                "multi_row {arm} policy={policy:?}: literal path disagrees with the independent \
                 dequantize+dot oracle (max_abs={})",
                literal_vs_oracle.max_abs
            );
        }
    }
}
