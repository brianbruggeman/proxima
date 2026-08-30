//! Prune-after-training: train the SAME tiny causal transformer
//! `tests/language_model.rs` trains, then re-emit the FFN with the
//! lowest-magnitude hidden-unit BANDS entirely absent -- never masked to
//! zero, never read by any op -- and check whether the pruned, fine-tuned
//! graph still overfits the corpus.
//!
//! This is the SECONDARY arm of this session's sparse-graph question (the
//! primary, constructed-topology arm lives in `constructed_sparse.rs`).
//! Pruning starts from a DENSE, gradient-descended checkpoint and only
//! decides its topology from data AFTER training, which is a structurally
//! different, harder case than a topology fixed at construction time --
//! see this file's own report for why.
//!
//! `offset_gradient_probe.rs` (this session) found the mechanism this file
//! does NOT use: reading a band via a nonzero-`offset` `IndexMap::Affine`
//! slice of a single SHARED weight leaf shape-infers fine forward (with an
//! anchor pinning the axis extent), but its adjoint's routed `Reduce` --
//! writing a gradient into the full shared buffer through that same
//! offset -- panics in `proxima-tensor/src/cpu.rs:4461` (index out of
//! bounds), a genuine evaluator gap in `proxima-tensor`, not this crate,
//! and out of this session's scope to fix. So every retained band here
//! gets its OWN separate, offset-0 leaf tensor -- exactly the
//! `a_static_block_sparse_matmul_needs_no_data_dependent_map` precedent
//! (`proxima-tensor/src/cpu.rs:16236-16243`) and this session's own
//! `constructed_sparse.rs` -- with the pruning DECISION (which bands
//! survive) made once, host-side, from the trained dense checkpoint's own
//! magnitudes, and the surviving bands' values copied in and back out by
//! plain Rust slicing, the same division of labour
//! `combine_table_gradient` already uses in `language_model.rs` for the
//! tied embedding's gathered contribution.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::too_many_arguments)]

extern crate alloc;

use proxima_autograd::activation::{relu, softmax};
use proxima_autograd::adjoint::differentiate;
use proxima_autograd::optimizer::{AdamConfig, AdamOperands, adam_step, step_input};
use proxima_tensor::cpu::evaluate_named;
use proxima_tensor::dtype::DType;
use proxima_tensor::error::TensorError;
use proxima_tensor::map::{self, AxisIndex, AxisTerm, IndexMap};
use proxima_tensor::op::{self, Extent, NodeId, Op, ReduceInit, ScalarOp};
use proxima_tensor::shape;

const TOKENIZER_PATH: &str = "/Users/brianbruggeman/.lmstudio/models/HuggingFaceTB/SmolLM2-135M-Instruct/tokenizer.json";
const CORPUS: &str = "Four score and seven years ago our fathers brought forth on this continent a new nation, \
conceived in liberty, and dedicated to the proposition that all men are created equal.";

const SEQ_LEN: usize = 32;
const D_MODEL: usize = 12;
const N_HEADS: usize = 3;
const HEAD_DIM: usize = 5;
const FFN_HIDDEN: usize = 20;
const BAND_COUNT: usize = 4;
const BAND_WIDTH: usize = FFN_HIDDEN / BAND_COUNT;

/// Skips (does not fail) when the real tokenizer is not present on this
/// host -- matching `proxima-model-interop/tests/real_lfm2_checkpoint.rs`'s
/// own posture, which this file's fixture-reading code restates rather
/// than shares across the integration-test-binary boundary.
fn checkpoint_present() -> bool {
    std::path::Path::new(TOKENIZER_PATH).exists()
}

fn real_vocab() -> proxima_tokenizer::Vocab {
    let bytes = std::fs::read(TOKENIZER_PATH).expect("read the real SmolLM2 tokenizer.json this session's task names");
    proxima_tokenizer::hf::vocab_from_tokenizer_json(&bytes, None, None, None).expect("real tokenizer.json parses")
}

struct Corpus {
    compact_ids: alloc::vec::Vec<u32>,
    compact_to_real: alloc::vec::Vec<u32>,
    vocab_size: usize,
}

fn tokenize_corpus(vocab: &proxima_tokenizer::Vocab) -> Corpus {
    let real_ids = proxima_tokenizer::encode(CORPUS, vocab).expect("real tokenizer encodes the corpus");
    assert_eq!(real_ids.len(), SEQ_LEN + 1, "corpus must tokenize to exactly SEQ_LEN + 1 real tokens");
    let mut compact_to_real: alloc::vec::Vec<u32> = real_ids.clone();
    compact_to_real.sort_unstable();
    compact_to_real.dedup();
    let compact_ids = real_ids
        .iter()
        .map(|&real_id| compact_to_real.binary_search(&real_id).expect("id present") as u32)
        .collect();
    let vocab_size = compact_to_real.len();
    Corpus { compact_ids, compact_to_real, vocab_size }
}

fn decode_compact(ids: &[u32], corpus: &Corpus, vocab: &proxima_tokenizer::Vocab) -> String {
    let real_ids: alloc::vec::Vec<u32> = ids.iter().map(|&id| corpus.compact_to_real[id as usize]).collect();
    proxima_tokenizer::decode(&real_ids, vocab).expect("decodes a sequence of valid restricted-vocab ids")
}

fn leaf(program: &mut Vec<Op>, name: &str, shape: alloc::vec::Vec<Extent>) -> NodeId {
    op::append(program, Op::Input { dtype: DType::Float32, shape, name: Some(name.into()) })
}

fn int_leaf(program: &mut Vec<Op>, name: &str, extent: u32) -> NodeId {
    op::append(program, Op::Input { dtype: DType::Int32, shape: alloc::vec![Extent::Static(extent)], name: Some(name.into()) })
}

fn constant(program: &mut Vec<Op>, value: f32) -> NodeId {
    op::append(program, Op::Constant { dtype: DType::Float32, shape: alloc::vec::Vec::new(), value })
}

fn iota(program: &mut Vec<Op>, extent: u32) -> NodeId {
    op::append(program, Op::Iota { dtype: DType::Float32, extent: Extent::Static(extent) })
}

fn elementwise(program: &mut Vec<Op>, body: ScalarOp, operands: alloc::vec::Vec<(NodeId, IndexMap)>) -> NodeId {
    op::append(program, Op::Elementwise { dtype: DType::Float32, body, operands, name: None })
}

fn reduce_add(program: &mut Vec<Op>, operand: NodeId, in_map: IndexMap, out_map: IndexMap) -> NodeId {
    op::append(
        program,
        Op::Reduce(proxima_tensor::op::Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand,
            in_map,
            out_map,
            keep: proxima_tensor::op::Keep::Reduce,
            name: None,
        }),
    )
}

fn proj(iter_rank: u16, axes: &[u16]) -> IndexMap {
    IndexMap::Affine(map::projection(iter_rank, axes))
}

fn identity(rank: u16) -> IndexMap {
    proj(rank, &(0..rank).collect::<alloc::vec::Vec<u16>>())
}

fn broadcast(rank: u16) -> IndexMap {
    proj(rank, &[])
}

fn total_macs_from(program: &[Op], start: usize) -> Result<u64, TensorError> {
    let shapes = shape::infer(program, &[])?;
    let mut macs = 0u64;
    for op in program.iter().skip(start) {
        if let Op::Reduce(reduce) = op
            && matches!(reduce.body, ScalarOp::Add)
        {
            macs += shapes.of(reduce.operand).iter().product::<u64>();
        }
    }
    Ok(macs)
}

fn total_macs(program: &[Op]) -> Result<u64, TensorError> {
    total_macs_from(program, 0)
}

fn embedding_gather(program: &mut Vec<Op>, table: NodeId, ids: NodeId) -> NodeId {
    let gathered_map = IndexMap::Computed {
        indices: ids,
        index_map: map::projection(2, &[0]),
        base: proxima_tensor::map::IndexPattern {
            iter_rank: 2,
            axes: alloc::vec![
                AxisIndex::default(),
                AxisIndex { terms: core::iter::once(AxisTerm::projection(1)).collect(), offset: 0 },
            ],
        },
        gathered_dim: 0,
    };
    elementwise(program, ScalarOp::Identity, alloc::vec![(table, gathered_map)])
}

struct AttnParams {
    table: NodeId,
    wq: NodeId,
    wk: NodeId,
    wv: NodeId,
    wo: NodeId,
}

fn declare_attn_params(program: &mut Vec<Op>, vocab_size: usize) -> AttnParams {
    AttnParams {
        table: leaf(program, "table", alloc::vec![Extent::Static(vocab_size as u32), Extent::Static(D_MODEL as u32)]),
        wq: leaf(
            program,
            "wq",
            alloc::vec![Extent::Static(D_MODEL as u32), Extent::Static(N_HEADS as u32), Extent::Static(HEAD_DIM as u32)],
        ),
        wk: leaf(
            program,
            "wk",
            alloc::vec![Extent::Static(D_MODEL as u32), Extent::Static(N_HEADS as u32), Extent::Static(HEAD_DIM as u32)],
        ),
        wv: leaf(
            program,
            "wv",
            alloc::vec![Extent::Static(D_MODEL as u32), Extent::Static(N_HEADS as u32), Extent::Static(HEAD_DIM as u32)],
        ),
        wo: leaf(
            program,
            "wo",
            alloc::vec![Extent::Static(N_HEADS as u32), Extent::Static(HEAD_DIM as u32), Extent::Static(D_MODEL as u32)],
        ),
    }
}

/// Embedding gather through one causal self-attention block, residual
/// added -- identical to `language_model.rs`'s own prefix, duplicated here
/// (that file's own convention: `leaf`/`elementwise`/`reduce_add` are
/// duplicated rather than imported, since they are test-local, not library
/// surface). Returns `(x, residual1)`.
fn build_attention(program: &mut Vec<Op>, params: &AttnParams, ids: NodeId) -> (NodeId, NodeId) {
    let x = embedding_gather(program, params.table, ids);

    let q_product = elementwise(program, ScalarOp::Multiply, alloc::vec![(x, proj(4, &[0, 3])), (params.wq, proj(4, &[3, 1, 2]))]);
    let q = reduce_add(program, q_product, identity(4), proj(4, &[0, 1, 2]));
    let k_product = elementwise(program, ScalarOp::Multiply, alloc::vec![(x, proj(4, &[0, 3])), (params.wk, proj(4, &[3, 1, 2]))]);
    let k = reduce_add(program, k_product, identity(4), proj(4, &[0, 1, 2]));
    let v_product = elementwise(program, ScalarOp::Multiply, alloc::vec![(x, proj(4, &[0, 3])), (params.wv, proj(4, &[3, 1, 2]))]);
    let v = reduce_add(program, v_product, identity(4), proj(4, &[0, 1, 2]));

    let score_product = elementwise(program, ScalarOp::Multiply, alloc::vec![(q, proj(4, &[0, 2, 3])), (k, proj(4, &[1, 2, 3]))]);
    let scores = reduce_add(program, score_product, identity(4), proj(4, &[0, 1, 2]));
    let inv_sqrt_head_dim = constant(program, 1.0 / (HEAD_DIM as f32).sqrt());
    let scaled = elementwise(program, ScalarOp::Multiply, alloc::vec![(scores, identity(3)), (inv_sqrt_head_dim, broadcast(3))]);

    let query_index = iota(program, SEQ_LEN as u32);
    let key_index = iota(program, SEQ_LEN as u32);
    let is_future = elementwise(program, ScalarOp::Greater, alloc::vec![(key_index, proj(2, &[1])), (query_index, proj(2, &[0]))]);
    let neg_infinity = constant(program, f32::NEG_INFINITY);
    let masked = elementwise(
        program,
        ScalarOp::Select,
        alloc::vec![(is_future, proj(3, &[0, 1])), (neg_infinity, broadcast(3)), (scaled, identity(3))],
    );

    let probabilities = softmax(program, DType::Float32, masked, 3, 1);
    let attended_product = elementwise(program, ScalarOp::Multiply, alloc::vec![(probabilities, proj(4, &[0, 1, 2])), (v, proj(4, &[1, 2, 3]))]);
    let attended = reduce_add(program, attended_product, identity(4), proj(4, &[0, 2, 3]));

    let attn_product = elementwise(program, ScalarOp::Multiply, alloc::vec![(attended, proj(4, &[0, 1, 2])), (params.wo, proj(4, &[1, 2, 3]))]);
    let attn_out = reduce_add(program, attn_product, identity(4), proj(4, &[0, 3]));
    let residual1 = elementwise(program, ScalarOp::Add, alloc::vec![(x, identity(2)), (attn_out, identity(2))]);

    (x, residual1)
}

fn build_head_and_loss(program: &mut Vec<Op>, residual2: NodeId, table: NodeId, onehot: NodeId) -> (NodeId, NodeId) {
    let logits_product = elementwise(program, ScalarOp::Multiply, alloc::vec![(residual2, proj(3, &[0, 1])), (table, proj(3, &[2, 1]))]);
    let logits = reduce_add(program, logits_product, identity(3), proj(3, &[0, 2]));

    let probabilities_lm = softmax(program, DType::Float32, logits, 2, 1);
    let log_epsilon = constant(program, 1e-7);
    let probabilities_lm_stabilized = elementwise(program, ScalarOp::Add, alloc::vec![(probabilities_lm, identity(2)), (log_epsilon, broadcast(2))]);
    let log_probabilities = elementwise(program, ScalarOp::Logarithm, alloc::vec![(probabilities_lm_stabilized, identity(2))]);
    let weighted = elementwise(program, ScalarOp::Multiply, alloc::vec![(onehot, identity(2)), (log_probabilities, identity(2))]);
    let per_position_loss = reduce_add(program, weighted, identity(2), proj(2, &[0]));
    let negated = elementwise(program, ScalarOp::Negate, alloc::vec![(per_position_loss, identity(1))]);
    let total_sum = reduce_add(program, negated, identity(1), proj(1, &[]));
    let inv_seq_len = constant(program, 1.0 / SEQ_LEN as f32);
    let loss = elementwise(program, ScalarOp::Multiply, alloc::vec![(total_sum, identity(0)), (inv_seq_len, broadcast(0))]);
    (logits, loss)
}

struct DenseFfnParams {
    w1: NodeId,
    b1: NodeId,
    w2: NodeId,
    b2: NodeId,
}

fn declare_dense_ffn(program: &mut Vec<Op>) -> DenseFfnParams {
    DenseFfnParams {
        w1: leaf(program, "w1", alloc::vec![Extent::Static(D_MODEL as u32), Extent::Static(FFN_HIDDEN as u32)]),
        b1: leaf(program, "b1", alloc::vec![Extent::Static(FFN_HIDDEN as u32)]),
        w2: leaf(program, "w2", alloc::vec![Extent::Static(FFN_HIDDEN as u32), Extent::Static(D_MODEL as u32)]),
        b2: leaf(program, "b2", alloc::vec![Extent::Static(D_MODEL as u32)]),
    }
}

fn build_dense_ffn(program: &mut Vec<Op>, residual1: NodeId, ffn: &DenseFfnParams) -> NodeId {
    let gate_product = elementwise(program, ScalarOp::Multiply, alloc::vec![(residual1, proj(3, &[0, 1])), (ffn.w1, proj(3, &[1, 2]))]);
    let gate = reduce_add(program, gate_product, identity(3), proj(3, &[0, 2]));
    let gate_biased = elementwise(program, ScalarOp::Add, alloc::vec![(gate, identity(2)), (ffn.b1, proj(2, &[1]))]);
    let hidden = relu(program, DType::Float32, gate_biased, 2);

    let down_product = elementwise(program, ScalarOp::Multiply, alloc::vec![(hidden, proj(3, &[0, 1])), (ffn.w2, proj(3, &[1, 2]))]);
    let ffn_out = reduce_add(program, down_product, identity(3), proj(3, &[0, 2]));
    let ffn_out_biased = elementwise(program, ScalarOp::Add, alloc::vec![(ffn_out, identity(2)), (ffn.b2, proj(2, &[1]))]);
    elementwise(program, ScalarOp::Add, alloc::vec![(residual1, identity(2)), (ffn_out_biased, identity(2))])
}

struct SparseFfnParams {
    bands: alloc::vec::Vec<(usize, NodeId, NodeId, NodeId)>,
    b2: NodeId,
}

/// Declares ONE small `[D_MODEL, BAND_WIDTH]` / `[BAND_WIDTH]` /
/// `[BAND_WIDTH, D_MODEL]` leaf triple per RETAINED band -- never one
/// shared `[D_MODEL, FFN_HIDDEN]` buffer read at an offset (see this
/// file's own module doc for why). A pruned band contributes no leaf, no
/// op, nothing: it is absent from the program, not masked within it.
fn declare_sparse_ffn(program: &mut Vec<Op>, retained_bands: &[usize]) -> SparseFfnParams {
    let bands = retained_bands
        .iter()
        .map(|&band| {
            let w1_band = leaf(program, &alloc::format!("w1_band{band}"), alloc::vec![Extent::Static(D_MODEL as u32), Extent::Static(BAND_WIDTH as u32)]);
            let b1_band = leaf(program, &alloc::format!("b1_band{band}"), alloc::vec![Extent::Static(BAND_WIDTH as u32)]);
            let w2_band = leaf(program, &alloc::format!("w2_band{band}"), alloc::vec![Extent::Static(BAND_WIDTH as u32), Extent::Static(D_MODEL as u32)]);
            (band, w1_band, b1_band, w2_band)
        })
        .collect();
    let b2 = leaf(program, "b2", alloc::vec![Extent::Static(D_MODEL as u32)]);
    SparseFfnParams { bands, b2 }
}

/// Per retained band: `gate_band[s,g] = sum_o residual1[s,o] * w1_band[o,g]`,
/// biased and relu'd, then `partial[s,d] = sum_g hidden_band[s,g] *
/// w2_band[g,d]`. Every retained band's `partial` is FULL `[SEQ_LEN,
/// D_MODEL]` width already (the OUTPUT axis is never blocked, only the
/// hidden axis is), so summing them with plain `Add` needs no
/// concatenation primitive -- `proxima-tensor` has none
/// (`out_map`-free `Elementwise`; see this session's own notes on why a
/// column-blocked reassembly would need one and a row/reduction-blocked
/// one does not).
fn build_sparse_ffn(program: &mut Vec<Op>, residual1: NodeId, sparse: &SparseFfnParams) -> NodeId {
    let mut partials: alloc::vec::Vec<NodeId> = alloc::vec::Vec::new();
    for &(_band, w1_band, b1_band, w2_band) in &sparse.bands {
        let gate_product = elementwise(program, ScalarOp::Multiply, alloc::vec![(residual1, proj(3, &[0, 1])), (w1_band, proj(3, &[1, 2]))]);
        let gate = reduce_add(program, gate_product, identity(3), proj(3, &[0, 2]));
        let gate_biased = elementwise(program, ScalarOp::Add, alloc::vec![(gate, identity(2)), (b1_band, proj(2, &[1]))]);
        let hidden_band = relu(program, DType::Float32, gate_biased, 2);

        let down_product = elementwise(program, ScalarOp::Multiply, alloc::vec![(hidden_band, proj(3, &[0, 1])), (w2_band, proj(3, &[1, 2]))]);
        let partial = reduce_add(program, down_product, identity(3), proj(3, &[0, 2]));
        partials.push(partial);
    }

    let mut ffn_out = partials[0];
    for &partial in &partials[1..] {
        ffn_out = elementwise(program, ScalarOp::Add, alloc::vec![(ffn_out, identity(2)), (partial, identity(2))]);
    }
    let ffn_out_biased = elementwise(program, ScalarOp::Add, alloc::vec![(ffn_out, identity(2)), (sparse.b2, proj(2, &[1]))]);
    elementwise(program, ScalarOp::Add, alloc::vec![(residual1, identity(2)), (ffn_out_biased, identity(2))])
}

fn counter_pattern(seed: usize, count: usize) -> alloc::vec::Vec<f32> {
    (0..count).map(|index| (((seed + index) * 7 % 13) as f32 - 6.0) / 24.0).collect()
}

fn onehot_targets(compact_ids: &[u32], vocab_size: usize) -> alloc::vec::Vec<f32> {
    let mut onehot = alloc::vec![0.0f32; SEQ_LEN * vocab_size];
    for (position, &target_id) in compact_ids[1..=SEQ_LEN].iter().enumerate() {
        onehot[position * vocab_size + target_id as usize] = 1.0;
    }
    onehot
}

fn input_ids_as_f32(compact_ids: &[u32]) -> alloc::vec::Vec<f32> {
    compact_ids[0..SEQ_LEN].iter().map(|&id| id as f32).collect()
}

struct AttnWeights {
    table: alloc::vec::Vec<f32>,
    wq: alloc::vec::Vec<f32>,
    wk: alloc::vec::Vec<f32>,
    wv: alloc::vec::Vec<f32>,
    wo: alloc::vec::Vec<f32>,
}

fn initial_attn_weights(vocab_size: usize) -> AttnWeights {
    AttnWeights {
        table: counter_pattern(11, vocab_size * D_MODEL),
        wq: counter_pattern(23, D_MODEL * N_HEADS * HEAD_DIM),
        wk: counter_pattern(29, D_MODEL * N_HEADS * HEAD_DIM),
        wv: counter_pattern(31, D_MODEL * N_HEADS * HEAD_DIM),
        wo: counter_pattern(37, N_HEADS * HEAD_DIM * D_MODEL),
    }
}

struct FfnWeights {
    w1: alloc::vec::Vec<f32>,
    b1: alloc::vec::Vec<f32>,
    w2: alloc::vec::Vec<f32>,
    b2: alloc::vec::Vec<f32>,
}

fn initial_ffn_weights() -> FfnWeights {
    FfnWeights {
        w1: counter_pattern(41, D_MODEL * FFN_HIDDEN),
        b1: counter_pattern(43, FFN_HIDDEN),
        w2: counter_pattern(47, FFN_HIDDEN * D_MODEL),
        b2: counter_pattern(53, D_MODEL),
    }
}

/// Sum of absolute weight magnitude touching band `band`'s hidden units --
/// `w1`'s own `BAND_WIDTH` COLUMNS in that band (every input row) plus
/// `w2`'s own `BAND_WIDTH` ROWS (every output column). The lowest-magnitude
/// bands are the ones this file's pruning test drops.
fn band_magnitude(w1: &[f32], w2: &[f32], band: usize) -> f32 {
    let mut total = 0.0f32;
    for row in 0..D_MODEL {
        for local in 0..BAND_WIDTH {
            total += w1[row * FFN_HIDDEN + band * BAND_WIDTH + local].abs();
        }
    }
    for local in 0..BAND_WIDTH {
        for col in 0..D_MODEL {
            total += w2[(band * BAND_WIDTH + local) * D_MODEL + col].abs();
        }
    }
    total
}

fn w1_band_slice(w1: &[f32], band: usize) -> alloc::vec::Vec<f32> {
    let mut slice = alloc::vec![0.0f32; D_MODEL * BAND_WIDTH];
    for row in 0..D_MODEL {
        for local in 0..BAND_WIDTH {
            slice[row * BAND_WIDTH + local] = w1[row * FFN_HIDDEN + band * BAND_WIDTH + local];
        }
    }
    slice
}

fn b1_band_slice(b1: &[f32], band: usize) -> alloc::vec::Vec<f32> {
    b1[band * BAND_WIDTH..(band + 1) * BAND_WIDTH].to_vec()
}

fn w2_band_slice(w2: &[f32], band: usize) -> alloc::vec::Vec<f32> {
    w2[band * BAND_WIDTH * D_MODEL..(band + 1) * BAND_WIDTH * D_MODEL].to_vec()
}

/// `(band index, w1 band values, b1 band values, w2 band values)` -- one
/// retained band's host-side buffers, kept as a plain tuple (a named type
/// would need no more fields than this alias already states, so a struct
/// here would be a relocation, not a capability -- see this crate's own
/// root doc for that test applied elsewhere).
type BandValues = alloc::vec::Vec<(usize, alloc::vec::Vec<f32>, alloc::vec::Vec<f32>, alloc::vec::Vec<f32>)>;

struct AdamState {
    m: alloc::vec::Vec<f32>,
    v: alloc::vec::Vec<f32>,
}

fn zero_state(count: usize) -> AdamState {
    AdamState { m: alloc::vec![0.0f32; count], v: alloc::vec![0.0f32; count] }
}

struct AdamNodes {
    new_param: NodeId,
    new_m: NodeId,
    new_v: NodeId,
}

fn append_adam(program: &mut Vec<Op>, config: &AdamConfig, rank: u16, extents: &[u32], tag: &str, param: NodeId, grad: NodeId, step: NodeId) -> AdamNodes {
    let shape: alloc::vec::Vec<Extent> = extents.iter().map(|&extent| Extent::Static(extent)).collect();
    let m_in = leaf(program, &alloc::format!("{tag}_m"), shape.clone());
    let v_in = leaf(program, &alloc::format!("{tag}_v"), shape);
    let (new_param, new_m, new_v) = adam_step(program, config, rank, AdamOperands { param, grad, m: m_in, v: v_in }, step);
    AdamNodes { new_param, new_m, new_v }
}

fn greedy_sample_matches(
    program: &[Op],
    logits: NodeId,
    table: &str,
    attn: &AttnWeights,
    ffn_named: &[(&str, &[f32])],
    corpus: &Corpus,
    prompt_len: usize,
) -> (alloc::vec::Vec<u32>, usize) {
    let _ = table;
    let zero_onehot = alloc::vec![0.0f32; SEQ_LEN * corpus.vocab_size];
    let mut generated: alloc::vec::Vec<u32> = corpus.compact_ids[0..prompt_len].to_vec();
    while generated.len() < SEQ_LEN {
        let mut ids_buffer = alloc::vec![0.0f32; SEQ_LEN];
        for (position, &id) in generated.iter().enumerate() {
            ids_buffer[position] = id as f32;
        }
        let mut named: alloc::vec::Vec<(&str, &[f32])> = alloc::vec![
            ("ids", ids_buffer.as_slice()),
            ("onehot", zero_onehot.as_slice()),
            ("table", attn.table.as_slice()),
            ("wq", attn.wq.as_slice()),
            ("wk", attn.wk.as_slice()),
            ("wv", attn.wv.as_slice()),
            ("wo", attn.wo.as_slice()),
        ];
        named.extend_from_slice(ffn_named);
        let evaluated = evaluate_named(program, &[], &named, &[logits]).expect("forward-only program lowers and evaluates");
        let logits_values = evaluated.get(logits).expect("logits requested").0;
        let position = generated.len() - 1;
        let row = &logits_values[position * corpus.vocab_size..(position + 1) * corpus.vocab_size];
        let (best_index, _) = row
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |best, (index, &value)| if value > best.1 { (index, value) } else { best });
        generated.push(best_index as u32);
    }
    let training_prefix: alloc::vec::Vec<u32> = corpus.compact_ids[0..SEQ_LEN].to_vec();
    let matches = generated.iter().zip(training_prefix.iter()).filter(|(a, b)| a == b).count();
    (generated, matches)
}

/// Node/mac counts for the FFN sub-graph ALONE (dense vs 2-of-4 bands
/// retained), reported alongside the whole-network counts so the report
/// can state both: how much the FFN itself shrinks, and how much of the
/// WHOLE model's compute the FFN actually was.
#[proxima::test]
async fn structural_counts_dense_ffn_vs_two_of_four_bands_retained() {
    let mut dense_program = alloc::vec::Vec::new();
    let dense_ffn = declare_dense_ffn(&mut dense_program);
    let residual1_stand_in = leaf(&mut dense_program, "residual1", alloc::vec![Extent::Static(SEQ_LEN as u32), Extent::Static(D_MODEL as u32)]);
    build_dense_ffn(&mut dense_program, residual1_stand_in, &dense_ffn);

    let mut sparse_program = alloc::vec::Vec::new();
    let sparse_ffn = declare_sparse_ffn(&mut sparse_program, &[0, 1]);
    let residual1_stand_in_sparse = leaf(&mut sparse_program, "residual1", alloc::vec![Extent::Static(SEQ_LEN as u32), Extent::Static(D_MODEL as u32)]);
    build_sparse_ffn(&mut sparse_program, residual1_stand_in_sparse, &sparse_ffn);

    let dense_macs = total_macs(&dense_program).expect("dense ffn infers");
    let sparse_macs = total_macs(&sparse_program).expect("sparse ffn infers");
    std::eprintln!(
        "FFN alone: dense program.len()={} macs={dense_macs}; sparse(2/4 bands) program.len()={} macs={sparse_macs} \
         (ratio {:.3})",
        dense_program.len(),
        sparse_program.len(),
        sparse_macs as f64 / dense_macs as f64
    );
    assert_eq!(dense_macs, (SEQ_LEN * D_MODEL * FFN_HIDDEN * 2) as u64, "dense FFN macs = seq*d_model*hidden for w1 + same for w2");
    assert_eq!(sparse_macs, (SEQ_LEN * D_MODEL * BAND_WIDTH * 2 * 2) as u64, "2 retained bands each cost seq*d_model*band_width for w1 and w2");
    assert!(sparse_macs < dense_macs, "retaining half the bands must cost fewer macs than dense, not the same or more");
}

const GRADIENT_CHECK_ATOL: f32 = 1e-2;
const GRADIENT_CHECK_RTOL: f32 = 1e-2;

fn within_tolerance(analytic: f32, numeric: f32) -> bool {
    (analytic - numeric).abs() <= GRADIENT_CHECK_ATOL + GRADIENT_CHECK_RTOL * numeric.abs()
}

fn checked_indices(len: usize) -> impl Iterator<Item = usize> {
    let stride = (len / 6).max(1);
    (0..len).step_by(stride)
}

/// The full arc: train the dense model to near-zero loss (reusing
/// `language_model.rs`'s own recipe), rank the FFN's 4 hidden bands by
/// magnitude, drop the 2 lowest, re-emit the FFN with those bands entirely
/// absent, measure the loss break, fine-tune the surviving graph, and
/// check whether greedy sampling still reproduces the corpus.
#[proxima::test]
async fn pruned_ffn_bands_degrade_loss_then_recover_with_fine_tuning() {
    if !checkpoint_present() {
        eprintln!("skipping: no host-local SmolLM2 tokenizer.json fixture at {TOKENIZER_PATH}");
        return;
    }
    let vocab = real_vocab();
    let corpus = tokenize_corpus(&vocab);
    let ids = input_ids_as_f32(&corpus.compact_ids);
    let onehot = onehot_targets(&corpus.compact_ids, corpus.vocab_size);

    // ---- Phase 1: train the DENSE model to a working baseline. ----
    let mut dense_program = alloc::vec::Vec::new();
    let attn_params = declare_attn_params(&mut dense_program, corpus.vocab_size);
    let ffn_params = declare_dense_ffn(&mut dense_program);
    let ids_node = int_leaf(&mut dense_program, "ids", SEQ_LEN as u32);
    let onehot_node = leaf(&mut dense_program, "onehot", alloc::vec![Extent::Static(SEQ_LEN as u32), Extent::Static(corpus.vocab_size as u32)]);
    let (_x, residual1) = build_attention(&mut dense_program, &attn_params, ids_node);
    let residual2 = build_dense_ffn(&mut dense_program, residual1, &ffn_params);
    let (_dense_logits, dense_loss) = build_head_and_loss(&mut dense_program, residual2, attn_params.table, onehot_node);

    let differentiated = differentiate(&dense_program, dense_loss).expect("scalar loss differentiates");
    let grad_table = differentiated.gradient_of_named("table").expect("table feeds the loss");
    let grad_wq = differentiated.gradient_of_named("wq").expect("wq feeds the loss");
    let grad_wk = differentiated.gradient_of_named("wk").expect("wk feeds the loss");
    let grad_wv = differentiated.gradient_of_named("wv").expect("wv feeds the loss");
    let grad_wo = differentiated.gradient_of_named("wo").expect("wo feeds the loss");
    let grad_w1 = differentiated.gradient_of_named("w1").expect("w1 feeds the loss");
    let grad_b1 = differentiated.gradient_of_named("b1").expect("b1 feeds the loss");
    let grad_w2 = differentiated.gradient_of_named("w2").expect("w2 feeds the loss");
    let grad_b2 = differentiated.gradient_of_named("b2").expect("b2 feeds the loss");

    let mut program = differentiated.program;
    let config = AdamConfig { learning_rate: 0.015, ..AdamConfig::default() };
    let step_node = step_input(&mut program, "step");

    let qkv_extents = [D_MODEL as u32, N_HEADS as u32, HEAD_DIM as u32];
    let wo_extents = [N_HEADS as u32, HEAD_DIM as u32, D_MODEL as u32];
    let table_nodes = append_adam(&mut program, &config, 2, &[corpus.vocab_size as u32, D_MODEL as u32], "table", attn_params.table, grad_table, step_node);
    let wq_nodes = append_adam(&mut program, &config, 3, &qkv_extents, "wq", attn_params.wq, grad_wq, step_node);
    let wk_nodes = append_adam(&mut program, &config, 3, &qkv_extents, "wk", attn_params.wk, grad_wk, step_node);
    let wv_nodes = append_adam(&mut program, &config, 3, &qkv_extents, "wv", attn_params.wv, grad_wv, step_node);
    let wo_nodes = append_adam(&mut program, &config, 3, &wo_extents, "wo", attn_params.wo, grad_wo, step_node);
    let w1_nodes = append_adam(&mut program, &config, 2, &[D_MODEL as u32, FFN_HIDDEN as u32], "w1", ffn_params.w1, grad_w1, step_node);
    let b1_nodes = append_adam(&mut program, &config, 1, &[FFN_HIDDEN as u32], "b1", ffn_params.b1, grad_b1, step_node);
    let w2_nodes = append_adam(&mut program, &config, 2, &[FFN_HIDDEN as u32, D_MODEL as u32], "w2", ffn_params.w2, grad_w2, step_node);
    let b2_nodes = append_adam(&mut program, &config, 1, &[D_MODEL as u32], "b2", ffn_params.b2, grad_b2, step_node);

    let mut attn = initial_attn_weights(corpus.vocab_size);
    let mut ffn = initial_ffn_weights();
    let mut table_state = zero_state(corpus.vocab_size * D_MODEL);
    let mut wq_state = zero_state(D_MODEL * N_HEADS * HEAD_DIM);
    let mut wk_state = zero_state(D_MODEL * N_HEADS * HEAD_DIM);
    let mut wv_state = zero_state(D_MODEL * N_HEADS * HEAD_DIM);
    let mut wo_state = zero_state(N_HEADS * HEAD_DIM * D_MODEL);
    let mut w1_state = zero_state(D_MODEL * FFN_HIDDEN);
    let mut b1_state = zero_state(FFN_HIDDEN);
    let mut w2_state = zero_state(FFN_HIDDEN * D_MODEL);
    let mut b2_state = zero_state(D_MODEL);

    const DENSE_STEPS: u32 = 400;
    let mut dense_loss_curve: alloc::vec::Vec<f32> = alloc::vec::Vec::new();
    for step in 0..DENSE_STEPS {
        let step_value = [(step + 1) as f32];
        let evaluated = evaluate_named(
            &program,
            &[],
            &[
                ("ids", ids.as_slice()), ("onehot", onehot.as_slice()),
                ("table", &attn.table), ("wq", &attn.wq), ("wk", &attn.wk), ("wv", &attn.wv), ("wo", &attn.wo),
                ("w1", &ffn.w1), ("b1", &ffn.b1), ("w2", &ffn.w2), ("b2", &ffn.b2),
                ("step", &step_value),
                ("table_m", &table_state.m), ("table_v", &table_state.v),
                ("wq_m", &wq_state.m), ("wq_v", &wq_state.v),
                ("wk_m", &wk_state.m), ("wk_v", &wk_state.v),
                ("wv_m", &wv_state.m), ("wv_v", &wv_state.v),
                ("wo_m", &wo_state.m), ("wo_v", &wo_state.v),
                ("w1_m", &w1_state.m), ("w1_v", &w1_state.v),
                ("b1_m", &b1_state.m), ("b1_v", &b1_state.v),
                ("w2_m", &w2_state.m), ("w2_v", &w2_state.v),
                ("b2_m", &b2_state.m), ("b2_v", &b2_state.v),
            ],
            &[
                dense_loss,
                table_nodes.new_param, table_nodes.new_m, table_nodes.new_v,
                wq_nodes.new_param, wq_nodes.new_m, wq_nodes.new_v,
                wk_nodes.new_param, wk_nodes.new_m, wk_nodes.new_v,
                wv_nodes.new_param, wv_nodes.new_m, wv_nodes.new_v,
                wo_nodes.new_param, wo_nodes.new_m, wo_nodes.new_v,
                w1_nodes.new_param, w1_nodes.new_m, w1_nodes.new_v,
                b1_nodes.new_param, b1_nodes.new_m, b1_nodes.new_v,
                w2_nodes.new_param, w2_nodes.new_m, w2_nodes.new_v,
                b2_nodes.new_param, b2_nodes.new_m, b2_nodes.new_v,
            ],
        )
        .expect("dense training-step program lowers and evaluates");

        dense_loss_curve.push(evaluated.get(dense_loss).expect("loss requested").0[0]);
        attn.table = evaluated.get(table_nodes.new_param).expect("requested").0.to_vec();
        table_state.m = evaluated.get(table_nodes.new_m).expect("requested").0.to_vec();
        table_state.v = evaluated.get(table_nodes.new_v).expect("requested").0.to_vec();
        attn.wq = evaluated.get(wq_nodes.new_param).expect("requested").0.to_vec();
        wq_state.m = evaluated.get(wq_nodes.new_m).expect("requested").0.to_vec();
        wq_state.v = evaluated.get(wq_nodes.new_v).expect("requested").0.to_vec();
        attn.wk = evaluated.get(wk_nodes.new_param).expect("requested").0.to_vec();
        wk_state.m = evaluated.get(wk_nodes.new_m).expect("requested").0.to_vec();
        wk_state.v = evaluated.get(wk_nodes.new_v).expect("requested").0.to_vec();
        attn.wv = evaluated.get(wv_nodes.new_param).expect("requested").0.to_vec();
        wv_state.m = evaluated.get(wv_nodes.new_m).expect("requested").0.to_vec();
        wv_state.v = evaluated.get(wv_nodes.new_v).expect("requested").0.to_vec();
        attn.wo = evaluated.get(wo_nodes.new_param).expect("requested").0.to_vec();
        wo_state.m = evaluated.get(wo_nodes.new_m).expect("requested").0.to_vec();
        wo_state.v = evaluated.get(wo_nodes.new_v).expect("requested").0.to_vec();
        ffn.w1 = evaluated.get(w1_nodes.new_param).expect("requested").0.to_vec();
        w1_state.m = evaluated.get(w1_nodes.new_m).expect("requested").0.to_vec();
        w1_state.v = evaluated.get(w1_nodes.new_v).expect("requested").0.to_vec();
        ffn.b1 = evaluated.get(b1_nodes.new_param).expect("requested").0.to_vec();
        b1_state.m = evaluated.get(b1_nodes.new_m).expect("requested").0.to_vec();
        b1_state.v = evaluated.get(b1_nodes.new_v).expect("requested").0.to_vec();
        ffn.w2 = evaluated.get(w2_nodes.new_param).expect("requested").0.to_vec();
        w2_state.m = evaluated.get(w2_nodes.new_m).expect("requested").0.to_vec();
        w2_state.v = evaluated.get(w2_nodes.new_v).expect("requested").0.to_vec();
        ffn.b2 = evaluated.get(b2_nodes.new_param).expect("requested").0.to_vec();
        b2_state.m = evaluated.get(b2_nodes.new_m).expect("requested").0.to_vec();
        b2_state.v = evaluated.get(b2_nodes.new_v).expect("requested").0.to_vec();
    }

    let dense_initial_loss = dense_loss_curve[0];
    let dense_final_loss = *dense_loss_curve.last().expect("at least one step ran");
    std::eprintln!("dense phase: initial loss {dense_initial_loss}, final loss {dense_final_loss} over {DENSE_STEPS} steps");
    assert!(dense_loss_curve.iter().all(|value| value.is_finite()), "dense loss went non-finite");
    assert!(dense_final_loss < 0.05, "dense baseline must overfit near-zero before pruning is a meaningful comparison, got {dense_final_loss}");

    // ---- Phase 2: rank bands by magnitude, drop the 2 lowest. ----
    let mut magnitudes: alloc::vec::Vec<(usize, f32)> =
        (0..BAND_COUNT).map(|band| (band, band_magnitude(&ffn.w1, &ffn.w2, band))).collect();
    magnitudes.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("magnitudes are finite"));
    std::eprintln!("band magnitudes (ascending): {magnitudes:?}");
    let mut retained_bands: alloc::vec::Vec<usize> = magnitudes[2..].iter().map(|&(band, _)| band).collect();
    retained_bands.sort_unstable();
    std::eprintln!("retaining bands {retained_bands:?} (50% of {BAND_COUNT} dropped)");

    // ---- Phase 3: re-emit the FFN with the pruned bands absent. ----
    let mut sparse_program = alloc::vec::Vec::new();
    let sparse_attn_params = declare_attn_params(&mut sparse_program, corpus.vocab_size);
    let sparse_ffn_params = declare_sparse_ffn(&mut sparse_program, &retained_bands);
    let sparse_ids_node = int_leaf(&mut sparse_program, "ids", SEQ_LEN as u32);
    let sparse_onehot_node = leaf(&mut sparse_program, "onehot", alloc::vec![Extent::Static(SEQ_LEN as u32), Extent::Static(corpus.vocab_size as u32)]);
    let (_sparse_x, sparse_residual1) = build_attention(&mut sparse_program, &sparse_attn_params, sparse_ids_node);
    let sparse_residual2 = build_sparse_ffn(&mut sparse_program, sparse_residual1, &sparse_ffn_params);
    let (sparse_logits, sparse_loss) = build_head_and_loss(&mut sparse_program, sparse_residual2, sparse_attn_params.table, sparse_onehot_node);

    let ffn_named_from_trained: alloc::vec::Vec<(alloc::string::String, alloc::vec::Vec<f32>)> = retained_bands
        .iter()
        .flat_map(|&band| {
            alloc::vec![
                (alloc::format!("w1_band{band}"), w1_band_slice(&ffn.w1, band)),
                (alloc::format!("b1_band{band}"), b1_band_slice(&ffn.b1, band)),
                (alloc::format!("w2_band{band}"), w2_band_slice(&ffn.w2, band)),
            ]
        })
        .collect();
    let mut named_at_prune: alloc::vec::Vec<(&str, &[f32])> = alloc::vec![
        ("ids", ids.as_slice()), ("onehot", onehot.as_slice()),
        ("table", attn.table.as_slice()), ("wq", attn.wq.as_slice()), ("wk", attn.wk.as_slice()),
        ("wv", attn.wv.as_slice()), ("wo", attn.wo.as_slice()),
        ("b2", ffn.b2.as_slice()),
    ];
    for (name, values) in &ffn_named_from_trained {
        named_at_prune.push((name.as_str(), values.as_slice()));
    }
    let evaluated_at_prune = evaluate_named(&sparse_program, &[], &named_at_prune, &[sparse_loss])
        .expect("sparse program lowers and evaluates immediately after pruning");
    let loss_immediately_after_pruning = evaluated_at_prune.get(sparse_loss).expect("loss requested").0[0];
    std::eprintln!(
        "loss immediately after pruning 2/4 bands (no fine-tuning yet): {loss_immediately_after_pruning} \
         (dense trained loss was {dense_final_loss})"
    );
    assert!(
        loss_immediately_after_pruning > dense_final_loss,
        "dropping real, trained bands must make the loss WORSE immediately, not leave it unchanged \
         (loss {loss_immediately_after_pruning} vs dense {dense_final_loss}) -- otherwise the pruned bands \
         were not actually contributing, which would itself be a real but different finding"
    );

    // ---- Phase 4: gradient-check the SPARSE graph before trusting its Adam updates. ----
    let sparse_differentiated = differentiate(&sparse_program, sparse_loss).expect("sparse scalar loss differentiates");
    let sparse_grad_w1_band0 = sparse_differentiated.gradient_of_named(&alloc::format!("w1_band{}", retained_bands[0])).expect("retained band feeds the loss");
    let sparse_grad_w2_band0 = sparse_differentiated.gradient_of_named(&alloc::format!("w2_band{}", retained_bands[0])).expect("retained band feeds the loss");
    let check_named: alloc::vec::Vec<(&str, &[f32])> = named_at_prune.clone();
    let evaluated_grads = evaluate_named(&sparse_differentiated.program, &[], &check_named, &[sparse_grad_w1_band0, sparse_grad_w2_band0])
        .expect("sparse adjoint program lowers and evaluates");
    let analytic_w1_band0 = evaluated_grads.get(sparse_grad_w1_band0).expect("requested").0.to_vec();
    let analytic_w2_band0 = evaluated_grads.get(sparse_grad_w2_band0).expect("requested").0.to_vec();

    let mut w1_band0_values = w1_band_slice(&ffn.w1, retained_bands[0]);
    let mut w2_band0_values = w2_band_slice(&ffn.w2, retained_bands[0]);
    let step_size = 1e-3f32;
    let loss_at_band0 = |w1_band: &[f32], w2_band: &[f32]| -> f32 {
        let mut named: alloc::vec::Vec<(&str, &[f32])> = check_named.clone();
        let band0_w1_key = alloc::format!("w1_band{}", retained_bands[0]);
        let band0_w2_key = alloc::format!("w2_band{}", retained_bands[0]);
        named.retain(|(name, _)| *name != band0_w1_key && *name != band0_w2_key);
        named.push((band0_w1_key.leak(), w1_band));
        named.push((band0_w2_key.leak(), w2_band));
        evaluate_named(&sparse_program, &[], &named, &[sparse_loss])
            .expect("sparse forward program lowers and evaluates")
            .get(sparse_loss)
            .expect("loss requested")
            .0[0]
    };
    let mut worst_violation: Option<(&str, usize, f32, f32)> = None;
    for index in checked_indices(w1_band0_values.len()) {
        let original = w1_band0_values[index];
        w1_band0_values[index] = original + step_size;
        let plus = loss_at_band0(&w1_band0_values, &w2_band0_values);
        w1_band0_values[index] = original - step_size;
        let minus = loss_at_band0(&w1_band0_values, &w2_band0_values);
        w1_band0_values[index] = original;
        let numeric = (plus - minus) / (2.0 * step_size);
        if !within_tolerance(analytic_w1_band0[index], numeric) {
            worst_violation = Some(("w1_band0", index, analytic_w1_band0[index], numeric));
        }
    }
    for index in checked_indices(w2_band0_values.len()) {
        let original = w2_band0_values[index];
        w2_band0_values[index] = original + step_size;
        let plus = loss_at_band0(&w1_band0_values, &w2_band0_values);
        w2_band0_values[index] = original - step_size;
        let minus = loss_at_band0(&w1_band0_values, &w2_band0_values);
        w2_band0_values[index] = original;
        let numeric = (plus - minus) / (2.0 * step_size);
        if !within_tolerance(analytic_w2_band0[index], numeric) {
            worst_violation = Some(("w2_band0", index, analytic_w2_band0[index], numeric));
        }
    }
    std::eprintln!("sparse-graph gradient check on the retained band closest to band 0: violation={worst_violation:?}");
    assert!(worst_violation.is_none(), "sparse graph's adjoint disagreed with central difference: {worst_violation:?}");

    // ---- Phase 5: fine-tune the sparse graph. ----
    let sparse_grad_table = sparse_differentiated.gradient_of_named("table").expect("table feeds the loss");
    let sparse_grad_wq = sparse_differentiated.gradient_of_named("wq").expect("wq feeds the loss");
    let sparse_grad_wk = sparse_differentiated.gradient_of_named("wk").expect("wk feeds the loss");
    let sparse_grad_wv = sparse_differentiated.gradient_of_named("wv").expect("wv feeds the loss");
    let sparse_grad_wo = sparse_differentiated.gradient_of_named("wo").expect("wo feeds the loss");
    let sparse_grad_b2 = sparse_differentiated.gradient_of_named("b2").expect("b2 feeds the loss");
    let mut band_grads: alloc::vec::Vec<(usize, NodeId, NodeId, NodeId)> = alloc::vec::Vec::new();
    for &band in &retained_bands {
        let grad_w1_band = sparse_differentiated.gradient_of_named(&alloc::format!("w1_band{band}")).expect("band feeds the loss");
        let grad_b1_band = sparse_differentiated.gradient_of_named(&alloc::format!("b1_band{band}")).expect("band feeds the loss");
        let grad_w2_band = sparse_differentiated.gradient_of_named(&alloc::format!("w2_band{band}")).expect("band feeds the loss");
        band_grads.push((band, grad_w1_band, grad_b1_band, grad_w2_band));
    }

    let mut sparse_train_program = sparse_differentiated.program;
    let sparse_step_node = step_input(&mut sparse_train_program, "step");
    let sparse_table_nodes = append_adam(&mut sparse_train_program, &config, 2, &[corpus.vocab_size as u32, D_MODEL as u32], "s_table", sparse_attn_params.table, sparse_grad_table, sparse_step_node);
    let sparse_wq_nodes = append_adam(&mut sparse_train_program, &config, 3, &qkv_extents, "s_wq", sparse_attn_params.wq, sparse_grad_wq, sparse_step_node);
    let sparse_wk_nodes = append_adam(&mut sparse_train_program, &config, 3, &qkv_extents, "s_wk", sparse_attn_params.wk, sparse_grad_wk, sparse_step_node);
    let sparse_wv_nodes = append_adam(&mut sparse_train_program, &config, 3, &qkv_extents, "s_wv", sparse_attn_params.wv, sparse_grad_wv, sparse_step_node);
    let sparse_wo_nodes = append_adam(&mut sparse_train_program, &config, 3, &wo_extents, "s_wo", sparse_attn_params.wo, sparse_grad_wo, sparse_step_node);
    let sparse_b2_nodes = append_adam(&mut sparse_train_program, &config, 1, &[D_MODEL as u32], "s_b2", sparse_ffn_params.b2, sparse_grad_b2, sparse_step_node);
    let mut sparse_band_nodes: alloc::vec::Vec<(usize, AdamNodes, AdamNodes, AdamNodes)> = alloc::vec::Vec::new();
    for &(band, grad_w1_band, grad_b1_band, grad_w2_band) in &band_grads {
        let (_band_check, w1_leaf, b1_leaf, w2_leaf) = *sparse_ffn_params.bands.iter().find(|(candidate, ..)| *candidate == band).expect("band declared");
        let w1_adam = append_adam(&mut sparse_train_program, &config, 2, &[D_MODEL as u32, BAND_WIDTH as u32], &alloc::format!("s_w1_band{band}"), w1_leaf, grad_w1_band, sparse_step_node);
        let b1_adam = append_adam(&mut sparse_train_program, &config, 1, &[BAND_WIDTH as u32], &alloc::format!("s_b1_band{band}"), b1_leaf, grad_b1_band, sparse_step_node);
        let w2_adam = append_adam(&mut sparse_train_program, &config, 2, &[BAND_WIDTH as u32, D_MODEL as u32], &alloc::format!("s_w2_band{band}"), w2_leaf, grad_w2_band, sparse_step_node);
        sparse_band_nodes.push((band, w1_adam, b1_adam, w2_adam));
    }

    let mut sparse_table_state = zero_state(corpus.vocab_size * D_MODEL);
    let mut sparse_wq_state = zero_state(D_MODEL * N_HEADS * HEAD_DIM);
    let mut sparse_wk_state = zero_state(D_MODEL * N_HEADS * HEAD_DIM);
    let mut sparse_wv_state = zero_state(D_MODEL * N_HEADS * HEAD_DIM);
    let mut sparse_wo_state = zero_state(N_HEADS * HEAD_DIM * D_MODEL);
    let mut sparse_b2_state = zero_state(D_MODEL);
    let mut sparse_attn = AttnWeights { table: attn.table.clone(), wq: attn.wq.clone(), wk: attn.wk.clone(), wv: attn.wv.clone(), wo: attn.wo.clone() };
    let mut sparse_b2 = ffn.b2.clone();
    let mut band_states: alloc::vec::Vec<(usize, AdamState, AdamState, AdamState)> = retained_bands
        .iter()
        .map(|&band| (band, zero_state(D_MODEL * BAND_WIDTH), zero_state(BAND_WIDTH), zero_state(BAND_WIDTH * D_MODEL)))
        .collect();
    let mut band_values: BandValues = retained_bands
        .iter()
        .map(|&band| (band, w1_band_slice(&ffn.w1, band), b1_band_slice(&ffn.b1, band), w2_band_slice(&ffn.w2, band)))
        .collect();

    const FINE_TUNE_STEPS: u32 = 20;
    let mut sparse_loss_curve: alloc::vec::Vec<f32> = alloc::vec::Vec::new();
    for step in 0..FINE_TUNE_STEPS {
        let step_value = [(step + 1) as f32];
        let mut named: alloc::vec::Vec<(&str, &[f32])> = alloc::vec![
            ("ids", ids.as_slice()), ("onehot", onehot.as_slice()),
            ("table", sparse_attn.table.as_slice()), ("wq", sparse_attn.wq.as_slice()), ("wk", sparse_attn.wk.as_slice()),
            ("wv", sparse_attn.wv.as_slice()), ("wo", sparse_attn.wo.as_slice()), ("b2", sparse_b2.as_slice()),
            ("step", &step_value),
            ("s_table_m", &sparse_table_state.m), ("s_table_v", &sparse_table_state.v),
            ("s_wq_m", &sparse_wq_state.m), ("s_wq_v", &sparse_wq_state.v),
            ("s_wk_m", &sparse_wk_state.m), ("s_wk_v", &sparse_wk_state.v),
            ("s_wv_m", &sparse_wv_state.m), ("s_wv_v", &sparse_wv_state.v),
            ("s_wo_m", &sparse_wo_state.m), ("s_wo_v", &sparse_wo_state.v),
            ("s_b2_m", &sparse_b2_state.m), ("s_b2_v", &sparse_b2_state.v),
        ];
        for (band, w1_values, b1_values, w2_values) in &band_values {
            named.push((alloc::format!("w1_band{band}").leak(), w1_values.as_slice()));
            named.push((alloc::format!("b1_band{band}").leak(), b1_values.as_slice()));
            named.push((alloc::format!("w2_band{band}").leak(), w2_values.as_slice()));
        }
        for (band, m_state, v_state, _w2) in &band_states {
            named.push((alloc::format!("s_w1_band{band}_m").leak(), m_state.m.as_slice()));
            named.push((alloc::format!("s_w1_band{band}_v").leak(), m_state.v.as_slice()));
            named.push((alloc::format!("s_b1_band{band}_m").leak(), v_state.m.as_slice()));
            named.push((alloc::format!("s_b1_band{band}_v").leak(), v_state.v.as_slice()));
        }
        for (band, _m, _v, w2_state) in &band_states {
            named.push((alloc::format!("s_w2_band{band}_m").leak(), w2_state.m.as_slice()));
            named.push((alloc::format!("s_w2_band{band}_v").leak(), w2_state.v.as_slice()));
        }

        let mut outputs: alloc::vec::Vec<NodeId> = alloc::vec![
            sparse_loss,
            sparse_table_nodes.new_param, sparse_table_nodes.new_m, sparse_table_nodes.new_v,
            sparse_wq_nodes.new_param, sparse_wq_nodes.new_m, sparse_wq_nodes.new_v,
            sparse_wk_nodes.new_param, sparse_wk_nodes.new_m, sparse_wk_nodes.new_v,
            sparse_wv_nodes.new_param, sparse_wv_nodes.new_m, sparse_wv_nodes.new_v,
            sparse_wo_nodes.new_param, sparse_wo_nodes.new_m, sparse_wo_nodes.new_v,
            sparse_b2_nodes.new_param, sparse_b2_nodes.new_m, sparse_b2_nodes.new_v,
        ];
        for (_band, w1_adam, b1_adam, w2_adam) in &sparse_band_nodes {
            outputs.push(w1_adam.new_param);
            outputs.push(w1_adam.new_m);
            outputs.push(w1_adam.new_v);
            outputs.push(b1_adam.new_param);
            outputs.push(b1_adam.new_m);
            outputs.push(b1_adam.new_v);
            outputs.push(w2_adam.new_param);
            outputs.push(w2_adam.new_m);
            outputs.push(w2_adam.new_v);
        }

        let evaluated = evaluate_named(&sparse_train_program, &[], &named, &outputs).expect("sparse training-step program lowers and evaluates");
        sparse_loss_curve.push(evaluated.get(sparse_loss).expect("loss requested").0[0]);

        sparse_attn.table = evaluated.get(sparse_table_nodes.new_param).expect("requested").0.to_vec();
        sparse_table_state.m = evaluated.get(sparse_table_nodes.new_m).expect("requested").0.to_vec();
        sparse_table_state.v = evaluated.get(sparse_table_nodes.new_v).expect("requested").0.to_vec();
        sparse_attn.wq = evaluated.get(sparse_wq_nodes.new_param).expect("requested").0.to_vec();
        sparse_wq_state.m = evaluated.get(sparse_wq_nodes.new_m).expect("requested").0.to_vec();
        sparse_wq_state.v = evaluated.get(sparse_wq_nodes.new_v).expect("requested").0.to_vec();
        sparse_attn.wk = evaluated.get(sparse_wk_nodes.new_param).expect("requested").0.to_vec();
        sparse_wk_state.m = evaluated.get(sparse_wk_nodes.new_m).expect("requested").0.to_vec();
        sparse_wk_state.v = evaluated.get(sparse_wk_nodes.new_v).expect("requested").0.to_vec();
        sparse_attn.wv = evaluated.get(sparse_wv_nodes.new_param).expect("requested").0.to_vec();
        sparse_wv_state.m = evaluated.get(sparse_wv_nodes.new_m).expect("requested").0.to_vec();
        sparse_wv_state.v = evaluated.get(sparse_wv_nodes.new_v).expect("requested").0.to_vec();
        sparse_attn.wo = evaluated.get(sparse_wo_nodes.new_param).expect("requested").0.to_vec();
        sparse_wo_state.m = evaluated.get(sparse_wo_nodes.new_m).expect("requested").0.to_vec();
        sparse_wo_state.v = evaluated.get(sparse_wo_nodes.new_v).expect("requested").0.to_vec();
        sparse_b2 = evaluated.get(sparse_b2_nodes.new_param).expect("requested").0.to_vec();
        sparse_b2_state.m = evaluated.get(sparse_b2_nodes.new_m).expect("requested").0.to_vec();
        sparse_b2_state.v = evaluated.get(sparse_b2_nodes.new_v).expect("requested").0.to_vec();

        for ((_band, w1_values, b1_values, w2_values), (_band2, w1_adam, b1_adam, w2_adam)) in band_values.iter_mut().zip(sparse_band_nodes.iter()) {
            *w1_values = evaluated.get(w1_adam.new_param).expect("requested").0.to_vec();
            *b1_values = evaluated.get(b1_adam.new_param).expect("requested").0.to_vec();
            *w2_values = evaluated.get(w2_adam.new_param).expect("requested").0.to_vec();
        }
        for ((_band, m_w1, v_w1, _w2), (_band2, w1_adam, b1_adam, w2_adam)) in band_states.iter_mut().zip(sparse_band_nodes.iter()) {
            m_w1.m = evaluated.get(w1_adam.new_m).expect("requested").0.to_vec();
            m_w1.v = evaluated.get(w1_adam.new_v).expect("requested").0.to_vec();
            v_w1.m = evaluated.get(b1_adam.new_m).expect("requested").0.to_vec();
            v_w1.v = evaluated.get(b1_adam.new_v).expect("requested").0.to_vec();
            let _ = w2_adam;
        }
        for ((_band, _m, _v, w2_state), (_band2, _w1_adam, _b1_adam, w2_adam)) in band_states.iter_mut().zip(sparse_band_nodes.iter()) {
            w2_state.m = evaluated.get(w2_adam.new_m).expect("requested").0.to_vec();
            w2_state.v = evaluated.get(w2_adam.new_v).expect("requested").0.to_vec();
        }
    }

    let fine_tune_initial = sparse_loss_curve[0];
    let fine_tune_final = *sparse_loss_curve.last().expect("at least one fine-tune step ran");
    std::eprintln!(
        "fine-tune phase (2/4 bands retained, {FINE_TUNE_STEPS} steps): loss right after pruning {loss_immediately_after_pruning}, \
         first fine-tune step loss {fine_tune_initial}, final fine-tune loss {fine_tune_final}"
    );
    assert!(sparse_loss_curve.iter().all(|value| value.is_finite()), "fine-tune loss went non-finite: {sparse_loss_curve:?}");

    let ffn_named_final: alloc::vec::Vec<(alloc::string::String, alloc::vec::Vec<f32>)> = band_values
        .iter()
        .flat_map(|(band, w1_values, b1_values, w2_values)| {
            alloc::vec![
                (alloc::format!("w1_band{band}"), w1_values.clone()),
                (alloc::format!("b1_band{band}"), b1_values.clone()),
                (alloc::format!("w2_band{band}"), w2_values.clone()),
            ]
        })
        .chain(core::iter::once((alloc::string::String::from("b2"), sparse_b2.clone())))
        .collect();
    let ffn_named_final_refs: alloc::vec::Vec<(&str, &[f32])> = ffn_named_final.iter().map(|(name, values)| (name.as_str(), values.as_slice())).collect();
    let (generated, matches) = greedy_sample_matches(&sparse_program, sparse_logits, "table", &sparse_attn, &ffn_named_final_refs, &corpus, 6);
    let sampled_text = decode_compact(&generated, &corpus, &vocab);
    let training_text = decode_compact(&corpus.compact_ids[0..SEQ_LEN], &corpus, &vocab);
    std::eprintln!("training text:  {training_text:?}");
    std::eprintln!("sampled text (2/4 bands, fine-tuned):   {sampled_text:?}");
    std::eprintln!("greedy sample matched {matches}/{SEQ_LEN} training positions exactly");
}
