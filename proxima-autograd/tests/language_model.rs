//! Composition milestone: every piece [`proxima_autograd`] ships (gather
//! adjoint, `relu`/`softmax`, Adam, `differentiate`) assembled into one
//! tiny causal transformer, trained on real text tokenized by the real
//! SmolLM2 tokenizer -- not synthetic ids. Each piece is independently
//! gradient-checked elsewhere (`tests/training_loop.rs`); this file is the
//! one that checks the ASSEMBLY, which is the thing that was unverified.
//!
//! The tied embedding table is the deliberate stress case: it is read
//! twice -- once through the input gather (`embedding_gather`, an
//! `IndexMap::Computed` operand, landing in `Differentiated::gathered`)
//! and once as the dense LM-head operand (`IndexMap::Affine`, landing in
//! `Differentiated::gradients`) -- so its true gradient is the SUM of a
//! dense `grad_of` entry and a scatter-added `GatheredContribution`, a
//! combination no existing test exercises. Central difference over the
//! real loss cannot see that decomposition at all; it only sees the true
//! combined effect, so it is the correct oracle for whether this file's
//! own combination code (`combine_table_gradient`) got it right.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::too_many_arguments)]

extern crate alloc;

use proxima_autograd::activation::{relu, softmax};
use proxima_autograd::adjoint::differentiate;
use proxima_autograd::optimizer::{AdamConfig, AdamOperands, adam_step, step_input};
use proxima_autograd::sparse;
use proxima_tensor::cpu::evaluate_named;
use proxima_tensor::dtype::DType;
use proxima_tensor::map::{self, AxisIndex, AxisTerm, IndexMap};
use proxima_tensor::op::{self, Extent, NodeId, Op, ReduceInit, ScalarOp};

/// Real, on-disk SmolLM2 tokenizer.json this workspace already has cached
/// locally -- the same fixture `proxima-tokenizer/src/hf.rs`'s own doc
/// names (49152-entry BPE vocab). Not shipped in-repo: a caller without
/// this path fails loudly (no silent skip) rather than falling back to a
/// synthetic vocab.
const TOKENIZER_PATH: &str =
    "/Users/brianbruggeman/.lmstudio/models/HuggingFaceTB/SmolLM2-135M-Instruct/tokenizer.json";

/// Opening sentence of the Gettysburg Address (public domain, 1863).
/// Real English prose, not a fabricated fixture -- "and" and "," each
/// repeat, so an overfit model has real recurring structure to
/// memorize, not 33 tokens that each occur once. Kept to one sentence
/// (not the full two-sentence passage this file once used) so a full
/// forward pass -- O(seq^2) for self-attention -- stays cheap enough
/// for the whole-model gradient check to finish inside
/// `PROXIMA_TEST_TIMEOUT_MS`'s default execution budget.
const CORPUS: &str = "Four score and seven years ago our fathers brought forth on this continent a new nation, \
conceived in liberty, and dedicated to the proposition that all men are created equal.";

const SEQ_LEN: usize = 32;
const D_MODEL: usize = 12;
const N_HEADS: usize = 3;
const HEAD_DIM: usize = 5;
const FFN_HIDDEN: usize = 20;

/// Skips (does not fail) when the real tokenizer is not present on this
/// host -- matching `proxima-model-interop/tests/real_lfm2_checkpoint.rs`'s
/// own posture, which this file's fixture-reading code restates rather
/// than shares across the integration-test-binary boundary.
fn checkpoint_present() -> bool {
    std::path::Path::new(TOKENIZER_PATH).exists()
}

fn real_vocab() -> proxima_tokenizer::Vocab {
    let bytes = std::fs::read(TOKENIZER_PATH)
        .expect("read the real SmolLM2 tokenizer.json this session's task names");
    proxima_tokenizer::hf::vocab_from_tokenizer_json(&bytes, None, None, None)
        .expect("real tokenizer.json parses")
}

/// Tokenizes [`CORPUS`] with the real tokenizer, then restricts the vocab
/// to exactly the ids this corpus uses (guiding principle: "vocab can be
/// restricted" -- a 49152-row embedding table is not what this milestone
/// is checking) remapped to a dense `0..unique` id space in sorted order
/// of the real token id, so the mapping is deterministic and reproducible
/// from the corpus text alone.
struct Corpus {
    /// One real token id per corpus position, already remapped to the
    /// restricted `0..vocab_size` space.
    compact_ids: alloc::vec::Vec<u32>,
    /// `compact_ids_to_real[compact] = real` -- the inverse, needed only
    /// to decode generated ids back through the real tokenizer.
    compact_to_real: alloc::vec::Vec<u32>,
    vocab_size: usize,
}

fn tokenize_corpus(vocab: &proxima_tokenizer::Vocab) -> Corpus {
    let real_ids =
        proxima_tokenizer::encode(CORPUS, vocab).expect("real tokenizer encodes the corpus");
    assert_eq!(
        real_ids.len(),
        SEQ_LEN + 1,
        "corpus must tokenize to exactly SEQ_LEN + 1 real tokens"
    );

    let mut compact_to_real: alloc::vec::Vec<u32> = real_ids.clone();
    compact_to_real.sort_unstable();
    compact_to_real.dedup();

    let compact_ids = real_ids
        .iter()
        .map(|&real_id| {
            compact_to_real
                .binary_search(&real_id)
                .expect("every real id in the corpus is present in its own restricted vocab")
                as u32
        })
        .collect();

    let vocab_size = compact_to_real.len();
    Corpus {
        compact_ids,
        compact_to_real,
        vocab_size,
    }
}

fn decode_compact(ids: &[u32], corpus: &Corpus, vocab: &proxima_tokenizer::Vocab) -> String {
    let real_ids: alloc::vec::Vec<u32> = ids
        .iter()
        .map(|&id| corpus.compact_to_real[id as usize])
        .collect();
    proxima_tokenizer::decode(&real_ids, vocab)
        .expect("decodes a sequence of valid restricted-vocab ids")
}

fn leaf(program: &mut Vec<Op>, name: &str, shape: alloc::vec::Vec<Extent>) -> NodeId {
    op::append(
        program,
        Op::Input {
            dtype: DType::Float32,
            shape,
            name: Some(name.into()),
        },
    )
}

fn int_leaf(program: &mut Vec<Op>, name: &str, extent: u32) -> NodeId {
    op::append(
        program,
        Op::Input {
            dtype: DType::Int32,
            shape: alloc::vec![Extent::Static(extent)],
            name: Some(name.into()),
        },
    )
}

fn constant(program: &mut Vec<Op>, value: f32) -> NodeId {
    op::append(
        program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec::Vec::new(),
            value,
        },
    )
}

fn iota(program: &mut Vec<Op>, extent: u32) -> NodeId {
    op::append(
        program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(extent),
        },
    )
}

fn elementwise(
    program: &mut Vec<Op>,
    body: ScalarOp,
    operands: alloc::vec::Vec<(NodeId, IndexMap)>,
) -> NodeId {
    op::append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body,
            operands,
            name: None,
        },
    )
}

fn reduce_add(
    program: &mut Vec<Op>,
    operand: NodeId,
    in_map: IndexMap,
    out_map: IndexMap,
) -> NodeId {
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

/// `table[ids[s], d]` over iteration space `(s, d)` -- identical shape to
/// `tests/training_loop.rs`'s own `embedding_gather` (that helper is
/// `pub(crate)`-adjacent test code, not library surface, so it is
/// duplicated here rather than imported, the same way that file already
/// duplicates its own copy of `leaf`/`elementwise`/`reduce_add` instead of
/// reaching into `proxima_autograd::expr`, which is private).
fn embedding_gather(program: &mut Vec<Op>, table: NodeId, ids: NodeId) -> NodeId {
    let gathered_map = IndexMap::Computed {
        indices: ids,
        index_map: map::projection(2, &[0]),
        base: proxima_tensor::map::IndexPattern {
            iter_rank: 2,
            axes: alloc::vec![
                AxisIndex::default(),
                AxisIndex {
                    terms: core::iter::once(AxisTerm::projection(1)).collect(),
                    offset: 0
                },
            ],
        },
        gathered_dim: 0,
    };
    elementwise(
        program,
        ScalarOp::Identity,
        alloc::vec![(table, gathered_map)],
    )
}

struct Params {
    table: NodeId,
    wq: NodeId,
    wk: NodeId,
    wv: NodeId,
    wo: NodeId,
    w1: NodeId,
    b1: NodeId,
    w2: NodeId,
    b2: NodeId,
}

fn declare_params(program: &mut Vec<Op>, vocab_size: usize) -> Params {
    Params {
        table: leaf(
            program,
            "table",
            alloc::vec![
                Extent::Static(vocab_size as u32),
                Extent::Static(D_MODEL as u32)
            ],
        ),
        wq: leaf(
            program,
            "wq",
            alloc::vec![
                Extent::Static(D_MODEL as u32),
                Extent::Static(N_HEADS as u32),
                Extent::Static(HEAD_DIM as u32)
            ],
        ),
        wk: leaf(
            program,
            "wk",
            alloc::vec![
                Extent::Static(D_MODEL as u32),
                Extent::Static(N_HEADS as u32),
                Extent::Static(HEAD_DIM as u32)
            ],
        ),
        wv: leaf(
            program,
            "wv",
            alloc::vec![
                Extent::Static(D_MODEL as u32),
                Extent::Static(N_HEADS as u32),
                Extent::Static(HEAD_DIM as u32)
            ],
        ),
        wo: leaf(
            program,
            "wo",
            alloc::vec![
                Extent::Static(N_HEADS as u32),
                Extent::Static(HEAD_DIM as u32),
                Extent::Static(D_MODEL as u32)
            ],
        ),
        w1: leaf(
            program,
            "w1",
            alloc::vec![
                Extent::Static(D_MODEL as u32),
                Extent::Static(FFN_HIDDEN as u32)
            ],
        ),
        b1: leaf(
            program,
            "b1",
            alloc::vec![Extent::Static(FFN_HIDDEN as u32)],
        ),
        w2: leaf(
            program,
            "w2",
            alloc::vec![
                Extent::Static(FFN_HIDDEN as u32),
                Extent::Static(D_MODEL as u32)
            ],
        ),
        b2: leaf(program, "b2", alloc::vec![Extent::Static(D_MODEL as u32)]),
    }
}

struct Network {
    program: alloc::vec::Vec<Op>,
    params: Params,
    logits: NodeId,
    loss: NodeId,
}

/// `embedding -> one causal self-attention block (residual) -> one SwiGLU-free
/// dense FFN block (residual) -> tied LM head -> cross-entropy`, over a
/// fixed `SEQ_LEN`-token causal window. No normalization layer: not one of
/// the pieces the task names, and RMSNorm's reciprocal-sqrt adjoint is
/// already proven independently in `proxima-tensor`/`proxima-model-interop`
/// -- adding it here would not exercise anything new about *this* crate's
/// composition, only add node count to every gradient check below.
///
/// Dimensions are deliberately asymmetric throughout: `SEQ_LEN` (60) !=
/// `vocab_size` (45) != `D_MODEL` (12) != `N_HEADS * HEAD_DIM` (15) !=
/// `FFN_HIDDEN` (20) -- a transposed weight or swapped axis anywhere in
/// this function produces a shape-inference error, never a silently wrong
/// answer at matching sizes (`tests/training_loop.rs`'s own "ROW 135"
/// lesson: a hand test at one channel cannot see a transpose).
fn build_language_model(vocab_size: usize) -> Network {
    let mut program = alloc::vec::Vec::new();
    let params = declare_params(&mut program, vocab_size);
    let ids = int_leaf(&mut program, "ids", SEQ_LEN as u32);
    let onehot = leaf(
        &mut program,
        "onehot",
        alloc::vec![
            Extent::Static(SEQ_LEN as u32),
            Extent::Static(vocab_size as u32)
        ],
    );

    let x = embedding_gather(&mut program, params.table, ids);

    // q[s,h,d] = x[s,i] * wq[i,h,d], summed over i -- iter (s,h,d,i).
    let q_product = elementwise(
        &mut program,
        ScalarOp::Multiply,
        alloc::vec![(x, proj(4, &[0, 3])), (params.wq, proj(4, &[3, 1, 2]))],
    );
    let q = reduce_add(&mut program, q_product, identity(4), proj(4, &[0, 1, 2]));
    let k_product = elementwise(
        &mut program,
        ScalarOp::Multiply,
        alloc::vec![(x, proj(4, &[0, 3])), (params.wk, proj(4, &[3, 1, 2]))],
    );
    let k = reduce_add(&mut program, k_product, identity(4), proj(4, &[0, 1, 2]));
    let v_product = elementwise(
        &mut program,
        ScalarOp::Multiply,
        alloc::vec![(x, proj(4, &[0, 3])), (params.wv, proj(4, &[3, 1, 2]))],
    );
    let v = reduce_add(&mut program, v_product, identity(4), proj(4, &[0, 1, 2]));

    // scores[s,t,h] = sum_d q[s,h,d] * k[t,h,d] -- iter (s,t,h,d).
    let score_product = elementwise(
        &mut program,
        ScalarOp::Multiply,
        alloc::vec![(q, proj(4, &[0, 2, 3])), (k, proj(4, &[1, 2, 3]))],
    );
    let scores = reduce_add(
        &mut program,
        score_product,
        identity(4),
        proj(4, &[0, 1, 2]),
    );
    let inv_sqrt_head_dim = constant(&mut program, 1.0 / (HEAD_DIM as f32).sqrt());
    let scaled = elementwise(
        &mut program,
        ScalarOp::Multiply,
        alloc::vec![(scores, identity(3)), (inv_sqrt_head_dim, broadcast(3))],
    );

    // is_future[s,t] = key_index[t] > query_index[s] -- `Op::Iota`'s own
    // doc names this exact composition as what it is for.
    let query_index = iota(&mut program, SEQ_LEN as u32);
    let key_index = iota(&mut program, SEQ_LEN as u32);
    let is_future = elementwise(
        &mut program,
        ScalarOp::Greater,
        alloc::vec![(key_index, proj(2, &[1])), (query_index, proj(2, &[0]))],
    );
    let neg_infinity = constant(&mut program, f32::NEG_INFINITY);
    let masked = elementwise(
        &mut program,
        ScalarOp::Select,
        alloc::vec![
            (is_future, proj(3, &[0, 1])),
            (neg_infinity, broadcast(3)),
            (scaled, identity(3))
        ],
    );

    // softmax over the key axis (axis 1 of (s,t,h)) -- the masked-routing
    // Reduce(Maximum) adjoint, not a broadcast; this is the rule the task
    // names as the one to deliberately break and confirm the checker catches.
    let probabilities = softmax(&mut program, DType::Float32, masked, 3, 1);

    // attended[s,h,d] = sum_t probabilities[s,t,h] * v[t,h,d].
    let attended_product = elementwise(
        &mut program,
        ScalarOp::Multiply,
        alloc::vec![
            (probabilities, proj(4, &[0, 1, 2])),
            (v, proj(4, &[1, 2, 3]))
        ],
    );
    let attended = reduce_add(
        &mut program,
        attended_product,
        identity(4),
        proj(4, &[0, 2, 3]),
    );

    // attn_out[s,o] = sum_{h,d} attended[s,h,d] * wo[h,d,o].
    let attn_product = elementwise(
        &mut program,
        ScalarOp::Multiply,
        alloc::vec![
            (attended, proj(4, &[0, 1, 2])),
            (params.wo, proj(4, &[1, 2, 3]))
        ],
    );
    let attn_out = reduce_add(&mut program, attn_product, identity(4), proj(4, &[0, 3]));
    let residual1 = elementwise(
        &mut program,
        ScalarOp::Add,
        alloc::vec![(x, identity(2)), (attn_out, identity(2))],
    );

    // FFN: gate[s,g] = sum_o residual1[s,o] * w1[o,g] + b1[g], relu, then
    // sum_g hidden[s,g] * w2[g,o] + b2[o].
    let gate_product = elementwise(
        &mut program,
        ScalarOp::Multiply,
        alloc::vec![(residual1, proj(3, &[0, 1])), (params.w1, proj(3, &[1, 2]))],
    );
    let gate = reduce_add(&mut program, gate_product, identity(3), proj(3, &[0, 2]));
    let gate_biased = elementwise(
        &mut program,
        ScalarOp::Add,
        alloc::vec![(gate, identity(2)), (params.b1, proj(2, &[1]))],
    );
    let hidden = relu(&mut program, DType::Float32, gate_biased, 2);

    let down_product = elementwise(
        &mut program,
        ScalarOp::Multiply,
        alloc::vec![(hidden, proj(3, &[0, 1])), (params.w2, proj(3, &[1, 2]))],
    );
    let ffn_out = reduce_add(&mut program, down_product, identity(3), proj(3, &[0, 2]));
    let ffn_out_biased = elementwise(
        &mut program,
        ScalarOp::Add,
        alloc::vec![(ffn_out, identity(2)), (params.b2, proj(2, &[1]))],
    );
    let residual2 = elementwise(
        &mut program,
        ScalarOp::Add,
        alloc::vec![(residual1, identity(2)), (ffn_out_biased, identity(2))],
    );

    // Tied LM head: logits[s,v] = sum_o residual2[s,o] * table[v,o] -- the
    // SAME `params.table` NodeId the input gather above already reads.
    // This is the composition this file exists to prove: `table` now
    // carries both a `gathered_of` entry (from the gather) and a
    // `grad_of` entry (from this dense reduce) after `differentiate`.
    let logits_product = elementwise(
        &mut program,
        ScalarOp::Multiply,
        alloc::vec![
            (residual2, proj(3, &[0, 1])),
            (params.table, proj(3, &[2, 1]))
        ],
    );
    let logits = reduce_add(&mut program, logits_product, identity(3), proj(3, &[0, 2]));

    let probabilities_lm = softmax(&mut program, DType::Float32, logits, 2, 1);
    // `+ epsilon` before the log: as training overfits, non-target
    // probabilities underflow to exactly 0.0 in f32, and `0.0 *
    // log(0.0) == 0.0 * -inf == NaN` in IEEE 754 -- observed live in
    // this file's own training run (loss finite through step 32, NaN
    // from step 33 on) before this line existed. The epsilon keeps
    // `log_probabilities` finite everywhere without changing the loss
    // by more than float32 noise for any non-degenerate probability.
    let log_epsilon = constant(&mut program, 1e-7);
    let probabilities_lm_stabilized = elementwise(
        &mut program,
        ScalarOp::Add,
        alloc::vec![(probabilities_lm, identity(2)), (log_epsilon, broadcast(2))],
    );
    let log_probabilities = elementwise(
        &mut program,
        ScalarOp::Logarithm,
        alloc::vec![(probabilities_lm_stabilized, identity(2))],
    );
    let weighted = elementwise(
        &mut program,
        ScalarOp::Multiply,
        alloc::vec![(onehot, identity(2)), (log_probabilities, identity(2))],
    );
    let per_position_loss = reduce_add(&mut program, weighted, identity(2), proj(2, &[0]));
    let negated = elementwise(
        &mut program,
        ScalarOp::Negate,
        alloc::vec![(per_position_loss, identity(1))],
    );
    let total_sum = reduce_add(&mut program, negated, identity(1), proj(1, &[]));
    let inv_seq_len = constant(&mut program, 1.0 / SEQ_LEN as f32);
    let loss = elementwise(
        &mut program,
        ScalarOp::Multiply,
        alloc::vec![(total_sum, identity(0)), (inv_seq_len, broadcast(0))],
    );

    Network {
        program,
        params,
        logits,
        loss,
    }
}

fn counter_pattern(seed: usize, count: usize) -> alloc::vec::Vec<f32> {
    (0..count)
        .map(|index| (((seed + index) * 7 % 13) as f32 - 6.0) / 24.0)
        .collect()
}

fn onehot_targets(compact_ids: &[u32], vocab_size: usize) -> alloc::vec::Vec<f32> {
    let mut onehot = alloc::vec![0.0f32; SEQ_LEN * vocab_size];
    for (position, &target_id) in compact_ids[1..=SEQ_LEN].iter().enumerate() {
        onehot[position * vocab_size + target_id as usize] = 1.0;
    }
    onehot
}

fn input_ids_as_f32(compact_ids: &[u32]) -> alloc::vec::Vec<f32> {
    compact_ids[0..SEQ_LEN]
        .iter()
        .map(|&id| id as f32)
        .collect()
}

struct Weights {
    table: alloc::vec::Vec<f32>,
    wq: alloc::vec::Vec<f32>,
    wk: alloc::vec::Vec<f32>,
    wv: alloc::vec::Vec<f32>,
    wo: alloc::vec::Vec<f32>,
    w1: alloc::vec::Vec<f32>,
    b1: alloc::vec::Vec<f32>,
    w2: alloc::vec::Vec<f32>,
    b2: alloc::vec::Vec<f32>,
}

fn initial_weights(vocab_size: usize) -> Weights {
    Weights {
        table: counter_pattern(11, vocab_size * D_MODEL),
        wq: counter_pattern(23, D_MODEL * N_HEADS * HEAD_DIM),
        wk: counter_pattern(29, D_MODEL * N_HEADS * HEAD_DIM),
        wv: counter_pattern(31, D_MODEL * N_HEADS * HEAD_DIM),
        wo: counter_pattern(37, N_HEADS * HEAD_DIM * D_MODEL),
        w1: counter_pattern(41, D_MODEL * FFN_HIDDEN),
        b1: counter_pattern(43, FFN_HIDDEN),
        w2: counter_pattern(47, FFN_HIDDEN * D_MODEL),
        b2: counter_pattern(53, D_MODEL),
    }
}

fn loss_at(program: &[Op], loss: NodeId, ids: &[f32], onehot: &[f32], weights: &Weights) -> f32 {
    let evaluated = evaluate_named(
        program,
        &[],
        &[
            ("ids", ids),
            ("onehot", onehot),
            ("table", &weights.table),
            ("wq", &weights.wq),
            ("wk", &weights.wk),
            ("wv", &weights.wv),
            ("wo", &weights.wo),
            ("w1", &weights.w1),
            ("b1", &weights.b1),
            ("w2", &weights.w2),
            ("b2", &weights.b2),
        ],
        &[loss],
    )
    .expect("language model program lowers and evaluates");
    evaluated.get(loss).expect("loss requested").0[0]
}

/// PyTorch's own `torch.autograd.gradcheck` tolerance convention
/// (`|analytic - numeric| <= atol + rtol * |numeric|`), not a bare
/// relative error: a causally-masked attention position (position 0 can
/// only attend to itself) or a near-orthogonal head slice genuinely
/// produces analytic gradients on the order of 1e-3, and f32 central
/// difference at `step = 1e-3` has an absolute noise floor around the
/// same order -- this file's own bisection
/// (`scratch_isolated_attention_block_gradient_check`, since deleted, once
/// caught it live): `wq[22]` analytic -0.00195, numeric -0.00286, a 32%
/// RELATIVE error on an absolute difference of 0.0009. A floor-less
/// relative error cannot distinguish that from a genuine bug; only a
/// combined tolerance can, so this is the criterion that actually gates
/// pass/fail below -- the raw worst relative error is still reported for
/// every tensor, unconditionally, for transparency.
const GRADIENT_CHECK_ATOL: f32 = 1e-2;
const GRADIENT_CHECK_RTOL: f32 = 1e-2;

fn within_tolerance(analytic: f32, numeric: f32) -> bool {
    (analytic - numeric).abs() <= GRADIENT_CHECK_ATOL + GRADIENT_CHECK_RTOL * numeric.abs()
}

fn relative_error(analytic: f32, numeric: f32) -> f32 {
    (analytic - numeric).abs() / (analytic.abs().max(numeric.abs()) + 1e-6)
}

/// Central-difference check for one parameter tensor, against the whole
/// assembled model's real loss (not a per-layer probe) -- reports the
/// worst (index, relative-error) pair, matching `training_loop.rs`'s own
/// convention.
struct GradientCheckReport {
    worst_relative: f32,
    worst_relative_index: usize,
    violation: Option<(usize, f32, f32)>,
}

/// Central difference is one pair of forward evaluations per checked
/// index; checking every scalar of a ~1700-parameter model costs
/// thousands of full forward passes over the whole assembled program.
/// Standard gradcheck practice (e.g. PyTorch's own `gradcheck` sampling
/// knobs) checks a representative subset rather than every element --
/// this walks a fixed, deterministic stride so the same indices are
/// checked every run (no run-to-run flake), covering at least 10
/// spread-out positions per tensor regardless of its size.
fn checked_indices(len: usize) -> impl Iterator<Item = usize> {
    let stride = (len / 10).max(1);
    (0..len).step_by(stride)
}

fn gradient_check_tensor(
    program: &[Op],
    loss: NodeId,
    ids: &[f32],
    onehot: &[f32],
    weights: &mut Weights,
    which: fn(&mut Weights) -> &mut alloc::vec::Vec<f32>,
    analytic: &[f32],
    step: f32,
) -> GradientCheckReport {
    let mut worst = (0.0f32, 0usize);
    let mut violation: Option<(usize, f32, f32)> = None;
    for index in checked_indices(analytic.len()) {
        let original = which(weights)[index];
        which(weights)[index] = original + step;
        let plus = loss_at(program, loss, ids, onehot, weights);
        which(weights)[index] = original - step;
        let minus = loss_at(program, loss, ids, onehot, weights);
        which(weights)[index] = original;

        let numeric = (plus - minus) / (2.0 * step);
        let relative = relative_error(analytic[index], numeric);
        if relative > worst.0 {
            worst = (relative, index);
        }
        if !within_tolerance(analytic[index], numeric) {
            let is_worse_violation = violation.is_none_or(
                |(_, previous_analytic, previous_numeric): (usize, f32, f32)| {
                    (analytic[index] - numeric).abs() > (previous_analytic - previous_numeric).abs()
                },
            );
            if is_worse_violation {
                violation = Some((index, analytic[index], numeric));
            }
        }
    }
    GradientCheckReport {
        worst_relative: worst.0,
        worst_relative_index: worst.1,
        violation,
    }
}

/// The dense `grad_of` contribution (from the tied LM head's own matmul)
/// plus the scattered `GatheredContribution` (from the input embedding
/// gather) applied onto its full `[vocab, D_MODEL]` shape -- the exact
/// combination this file's own module doc calls out as unverified
/// elsewhere. `dense` is already full-shape (every vocab row is read by
/// the LM head regardless of whether that row was ever an input token),
/// so it is the accumulation base; the gathered rows are summed in on top.
fn combine_table_gradient(
    dense: &[f32],
    gathered_ids: &[u32],
    gathered_rows: &[f32],
    vocab_size: usize,
    dim: usize,
) -> alloc::vec::Vec<f32> {
    let mut combined = dense.to_vec();
    assert_eq!(combined.len(), vocab_size * dim);
    for (position, &id) in gathered_ids.iter().enumerate() {
        let source = &gathered_rows[position * dim..(position + 1) * dim];
        let destination = &mut combined[id as usize * dim..(id as usize + 1) * dim];
        for (accumulator, value) in destination.iter_mut().zip(source) {
            *accumulator += *value;
        }
    }
    combined
}

/// Gradient-checks every parameter tensor of the fully assembled model
/// (embedding, attention projections, FFN, tied LM head folded back into
/// the embedding) against central difference over the real loss.
#[proxima::test]
async fn whole_model_gradient_check_matches_central_difference_on_every_tensor() {
    if !checkpoint_present() {
        eprintln!("skipping: no host-local SmolLM2 tokenizer.json fixture at {TOKENIZER_PATH}");
        return;
    }
    let vocab = real_vocab();
    let corpus = tokenize_corpus(&vocab);
    let network = build_language_model(corpus.vocab_size);
    let differentiated =
        differentiate(&network.program, network.loss).expect("scalar loss differentiates");

    let ids = input_ids_as_f32(&corpus.compact_ids);
    let onehot = onehot_targets(&corpus.compact_ids, corpus.vocab_size);
    let mut weights = initial_weights(corpus.vocab_size);

    let grad_wq = differentiated
        .gradient_of_named("wq")
        .expect("wq feeds the loss");
    let grad_wk = differentiated
        .gradient_of_named("wk")
        .expect("wk feeds the loss");
    let grad_wv = differentiated
        .gradient_of_named("wv")
        .expect("wv feeds the loss");
    let grad_wo = differentiated
        .gradient_of_named("wo")
        .expect("wo feeds the loss");
    let grad_w1 = differentiated
        .gradient_of_named("w1")
        .expect("w1 feeds the loss");
    let grad_b1 = differentiated
        .gradient_of_named("b1")
        .expect("b1 feeds the loss");
    let grad_w2 = differentiated
        .gradient_of_named("w2")
        .expect("w2 feeds the loss");
    let grad_b2 = differentiated
        .gradient_of_named("b2")
        .expect("b2 feeds the loss");
    let grad_table_dense = differentiated
        .gradient_of_named("table")
        .expect("the tied LM head reads table densely");
    let table_gathered: alloc::vec::Vec<_> = differentiated
        .gathered_gradients_of_named("table")
        .collect();
    assert_eq!(
        table_gathered.len(),
        1,
        "table is gathered by exactly the one input-embedding site"
    );
    let table_gathered = table_gathered[0];

    let evaluated = evaluate_named(
        &differentiated.program,
        &[],
        &[
            ("ids", ids.as_slice()),
            ("onehot", onehot.as_slice()),
            ("table", &weights.table),
            ("wq", &weights.wq),
            ("wk", &weights.wk),
            ("wv", &weights.wv),
            ("wo", &weights.wo),
            ("w1", &weights.w1),
            ("b1", &weights.b1),
            ("w2", &weights.w2),
            ("b2", &weights.b2),
        ],
        &[
            grad_wq,
            grad_wk,
            grad_wv,
            grad_wo,
            grad_w1,
            grad_b1,
            grad_w2,
            grad_b2,
            grad_table_dense,
            table_gathered.values,
        ],
    )
    .expect("adjoint program lowers and evaluates");

    let analytic_wq = evaluated.get(grad_wq).expect("requested").0.to_vec();
    let analytic_wk = evaluated.get(grad_wk).expect("requested").0.to_vec();
    let analytic_wv = evaluated.get(grad_wv).expect("requested").0.to_vec();
    let analytic_wo = evaluated.get(grad_wo).expect("requested").0.to_vec();
    let analytic_w1 = evaluated.get(grad_w1).expect("requested").0.to_vec();
    let analytic_b1 = evaluated.get(grad_b1).expect("requested").0.to_vec();
    let analytic_w2 = evaluated.get(grad_w2).expect("requested").0.to_vec();
    let analytic_b2 = evaluated.get(grad_b2).expect("requested").0.to_vec();
    let dense_table = evaluated.get(grad_table_dense).expect("requested").0;
    let gathered_values = evaluated.get(table_gathered.values).expect("requested").0;

    let (unique_ids, summed) = sparse::dedupe_and_sum_rows(&ids, gathered_values, D_MODEL)
        .expect("ids and the compact contribution line up row for row");
    let analytic_table = combine_table_gradient(
        dense_table,
        &unique_ids,
        &summed,
        corpus.vocab_size,
        D_MODEL,
    );

    let step = 1e-3f32;
    let report: alloc::vec::Vec<(&str, GradientCheckReport)> = alloc::vec![
        (
            "wq",
            gradient_check_tensor(
                &network.program,
                network.loss,
                &ids,
                &onehot,
                &mut weights,
                |weights| &mut weights.wq,
                &analytic_wq,
                step
            )
        ),
        (
            "wk",
            gradient_check_tensor(
                &network.program,
                network.loss,
                &ids,
                &onehot,
                &mut weights,
                |weights| &mut weights.wk,
                &analytic_wk,
                step
            )
        ),
        (
            "wv",
            gradient_check_tensor(
                &network.program,
                network.loss,
                &ids,
                &onehot,
                &mut weights,
                |weights| &mut weights.wv,
                &analytic_wv,
                step
            )
        ),
        (
            "wo",
            gradient_check_tensor(
                &network.program,
                network.loss,
                &ids,
                &onehot,
                &mut weights,
                |weights| &mut weights.wo,
                &analytic_wo,
                step
            )
        ),
        (
            "w1",
            gradient_check_tensor(
                &network.program,
                network.loss,
                &ids,
                &onehot,
                &mut weights,
                |weights| &mut weights.w1,
                &analytic_w1,
                step
            )
        ),
        (
            "b1",
            gradient_check_tensor(
                &network.program,
                network.loss,
                &ids,
                &onehot,
                &mut weights,
                |weights| &mut weights.b1,
                &analytic_b1,
                step
            )
        ),
        (
            "w2",
            gradient_check_tensor(
                &network.program,
                network.loss,
                &ids,
                &onehot,
                &mut weights,
                |weights| &mut weights.w2,
                &analytic_w2,
                step
            )
        ),
        (
            "b2",
            gradient_check_tensor(
                &network.program,
                network.loss,
                &ids,
                &onehot,
                &mut weights,
                |weights| &mut weights.b2,
                &analytic_b2,
                step
            )
        ),
        (
            "table (dense + gathered combined)",
            gradient_check_tensor(
                &network.program,
                network.loss,
                &ids,
                &onehot,
                &mut weights,
                |weights| &mut weights.table,
                &analytic_table,
                step
            )
        ),
    ];

    for (name, result) in &report {
        std::eprintln!(
            "whole-model gradient check: {name} max relative error = {} (at flat index {}), tolerance violations (atol {GRADIENT_CHECK_ATOL} + rtol {GRADIENT_CHECK_RTOL}*|numeric|): {:?}",
            result.worst_relative,
            result.worst_relative_index,
            result.violation
        );
    }
    for (name, result) in &report {
        assert!(
            result.violation.is_none(),
            "tensor {name} disagreed with central difference beyond tolerance: {:?} (raw worst relative error {})",
            result.violation,
            result.worst_relative
        );
    }
}

struct AdamState {
    m: alloc::vec::Vec<f32>,
    v: alloc::vec::Vec<f32>,
}

fn zero_state(count: usize) -> AdamState {
    AdamState {
        m: alloc::vec![0.0f32; count],
        v: alloc::vec![0.0f32; count],
    }
}

struct AdamNodes {
    new_param: NodeId,
    new_m: NodeId,
    new_v: NodeId,
}

/// Appends one Adam update for `param` (reusing its already-in-graph
/// `grad` NodeId) onto `program`, declaring fresh `m`/`v` `Op::Input`
/// leaves named `{tag}_m`/`{tag}_v` for the caller to rebind every step.
fn append_adam(
    program: &mut Vec<Op>,
    config: &AdamConfig,
    rank: u16,
    extents: &[u32],
    tag: &str,
    param: NodeId,
    grad: NodeId,
    step: NodeId,
) -> AdamNodes {
    let shape: alloc::vec::Vec<Extent> = extents
        .iter()
        .map(|&extent| Extent::Static(extent))
        .collect();
    let m_in = leaf(program, &alloc::format!("{tag}_m"), shape.clone());
    let v_in = leaf(program, &alloc::format!("{tag}_v"), shape);
    let (new_param, new_m, new_v) = adam_step(
        program,
        config,
        rank,
        AdamOperands {
            param,
            grad,
            m: m_in,
            v: v_in,
        },
        step,
    );
    AdamNodes {
        new_param,
        new_m,
        new_v,
    }
}

/// Trains [`build_language_model`] with Adam over its own single-sequence
/// corpus repeatedly (full-batch, no mini-batching -- there is only one
/// training example, the whole corpus), printing the complete loss curve.
/// A correct LM overfitting ~60 real tokens must drive training loss
/// toward near-zero; a slowly-decreasing loss is exactly what a
/// partially-wrong gradient also produces (this file's own doc), so the
/// pass criterion below is deliberately a near-zero floor, not merely "it
/// decreased".
#[proxima::test]
async fn adam_training_overfits_the_real_corpus_toward_near_zero_loss() {
    if !checkpoint_present() {
        eprintln!("skipping: no host-local SmolLM2 tokenizer.json fixture at {TOKENIZER_PATH}");
        return;
    }
    let vocab = real_vocab();
    let corpus = tokenize_corpus(&vocab);
    let network = build_language_model(corpus.vocab_size);
    let differentiated =
        differentiate(&network.program, network.loss).expect("scalar loss differentiates");

    let grad_wq = differentiated
        .gradient_of_named("wq")
        .expect("wq feeds the loss");
    let grad_wk = differentiated
        .gradient_of_named("wk")
        .expect("wk feeds the loss");
    let grad_wv = differentiated
        .gradient_of_named("wv")
        .expect("wv feeds the loss");
    let grad_wo = differentiated
        .gradient_of_named("wo")
        .expect("wo feeds the loss");
    let grad_w1 = differentiated
        .gradient_of_named("w1")
        .expect("w1 feeds the loss");
    let grad_b1 = differentiated
        .gradient_of_named("b1")
        .expect("b1 feeds the loss");
    let grad_w2 = differentiated
        .gradient_of_named("w2")
        .expect("w2 feeds the loss");
    let grad_b2 = differentiated
        .gradient_of_named("b2")
        .expect("b2 feeds the loss");
    let grad_table_dense = differentiated
        .gradient_of_named("table")
        .expect("the tied LM head reads table densely");
    let table_gathered = differentiated
        .gathered_gradients_of_named("table")
        .next()
        .expect("table is gathered once");

    let mut program = differentiated.program;
    let config = AdamConfig {
        learning_rate: 0.015,
        ..AdamConfig::default()
    };
    let step_node = step_input(&mut program, "step");

    let qkv_extents = [D_MODEL as u32, N_HEADS as u32, HEAD_DIM as u32];
    let wo_extents = [N_HEADS as u32, HEAD_DIM as u32, D_MODEL as u32];
    let wq_nodes = append_adam(
        &mut program,
        &config,
        3,
        &qkv_extents,
        "wq",
        network.params.wq,
        grad_wq,
        step_node,
    );
    let wk_nodes = append_adam(
        &mut program,
        &config,
        3,
        &qkv_extents,
        "wk",
        network.params.wk,
        grad_wk,
        step_node,
    );
    let wv_nodes = append_adam(
        &mut program,
        &config,
        3,
        &qkv_extents,
        "wv",
        network.params.wv,
        grad_wv,
        step_node,
    );
    let wo_nodes = append_adam(
        &mut program,
        &config,
        3,
        &wo_extents,
        "wo",
        network.params.wo,
        grad_wo,
        step_node,
    );
    let w1_nodes = append_adam(
        &mut program,
        &config,
        2,
        &[D_MODEL as u32, FFN_HIDDEN as u32],
        "w1",
        network.params.w1,
        grad_w1,
        step_node,
    );
    let b1_nodes = append_adam(
        &mut program,
        &config,
        1,
        &[FFN_HIDDEN as u32],
        "b1",
        network.params.b1,
        grad_b1,
        step_node,
    );
    let w2_nodes = append_adam(
        &mut program,
        &config,
        2,
        &[FFN_HIDDEN as u32, D_MODEL as u32],
        "w2",
        network.params.w2,
        grad_w2,
        step_node,
    );
    let b2_nodes = append_adam(
        &mut program,
        &config,
        1,
        &[D_MODEL as u32],
        "b2",
        network.params.b2,
        grad_b2,
        step_node,
    );

    // table's Adam step runs over a SEPARATE small program: its true
    // gradient (dense LM-head contribution + scatter-added gathered
    // contribution) is only known after a host-side combine step this
    // crate's own `sparse::dedupe_and_sum_rows` cannot do inside the graph
    // (see this file's module doc).
    let mut table_program = alloc::vec::Vec::new();
    let table_param = leaf(
        &mut table_program,
        "table",
        alloc::vec![
            Extent::Static(corpus.vocab_size as u32),
            Extent::Static(D_MODEL as u32)
        ],
    );
    let table_grad_in = leaf(
        &mut table_program,
        "table_grad",
        alloc::vec![
            Extent::Static(corpus.vocab_size as u32),
            Extent::Static(D_MODEL as u32)
        ],
    );
    let table_step = step_input(&mut table_program, "step");
    let table_adam = append_adam(
        &mut table_program,
        &config,
        2,
        &[corpus.vocab_size as u32, D_MODEL as u32],
        "table",
        table_param,
        table_grad_in,
        table_step,
    );

    let ids = input_ids_as_f32(&corpus.compact_ids);
    let onehot = onehot_targets(&corpus.compact_ids, corpus.vocab_size);
    let mut weights = initial_weights(corpus.vocab_size);
    let mut wq_state = zero_state(D_MODEL * N_HEADS * HEAD_DIM);
    let mut wk_state = zero_state(D_MODEL * N_HEADS * HEAD_DIM);
    let mut wv_state = zero_state(D_MODEL * N_HEADS * HEAD_DIM);
    let mut wo_state = zero_state(N_HEADS * HEAD_DIM * D_MODEL);
    let mut w1_state = zero_state(D_MODEL * FFN_HIDDEN);
    let mut b1_state = zero_state(FFN_HIDDEN);
    let mut w2_state = zero_state(FFN_HIDDEN * D_MODEL);
    let mut b2_state = zero_state(D_MODEL);
    let mut table_state = zero_state(corpus.vocab_size * D_MODEL);

    const STEPS: u32 = 400;
    let mut loss_curve: alloc::vec::Vec<f32> = alloc::vec::Vec::new();

    for step in 0..STEPS {
        let step_value = [(step + 1) as f32];
        let evaluated = evaluate_named(
            &program,
            &[],
            &[
                ("ids", ids.as_slice()),
                ("onehot", onehot.as_slice()),
                ("table", &weights.table),
                ("wq", &weights.wq),
                ("wk", &weights.wk),
                ("wv", &weights.wv),
                ("wo", &weights.wo),
                ("w1", &weights.w1),
                ("b1", &weights.b1),
                ("w2", &weights.w2),
                ("b2", &weights.b2),
                ("step", &step_value),
                ("wq_m", &wq_state.m),
                ("wq_v", &wq_state.v),
                ("wk_m", &wk_state.m),
                ("wk_v", &wk_state.v),
                ("wv_m", &wv_state.m),
                ("wv_v", &wv_state.v),
                ("wo_m", &wo_state.m),
                ("wo_v", &wo_state.v),
                ("w1_m", &w1_state.m),
                ("w1_v", &w1_state.v),
                ("b1_m", &b1_state.m),
                ("b1_v", &b1_state.v),
                ("w2_m", &w2_state.m),
                ("w2_v", &w2_state.v),
                ("b2_m", &b2_state.m),
                ("b2_v", &b2_state.v),
            ],
            &[
                network.loss,
                grad_table_dense,
                table_gathered.values,
                wq_nodes.new_param,
                wq_nodes.new_m,
                wq_nodes.new_v,
                wk_nodes.new_param,
                wk_nodes.new_m,
                wk_nodes.new_v,
                wv_nodes.new_param,
                wv_nodes.new_m,
                wv_nodes.new_v,
                wo_nodes.new_param,
                wo_nodes.new_m,
                wo_nodes.new_v,
                w1_nodes.new_param,
                w1_nodes.new_m,
                w1_nodes.new_v,
                b1_nodes.new_param,
                b1_nodes.new_m,
                b1_nodes.new_v,
                w2_nodes.new_param,
                w2_nodes.new_m,
                w2_nodes.new_v,
                b2_nodes.new_param,
                b2_nodes.new_m,
                b2_nodes.new_v,
            ],
        )
        .expect("training-step program lowers and evaluates");

        loss_curve.push(evaluated.get(network.loss).expect("loss requested").0[0]);

        let dense_table = evaluated.get(grad_table_dense).expect("requested").0;
        let gathered_values = evaluated.get(table_gathered.values).expect("requested").0;
        let (unique_ids, summed) = sparse::dedupe_and_sum_rows(&ids, gathered_values, D_MODEL)
            .expect("ids/contribution line up");
        let combined_table_grad = combine_table_gradient(
            dense_table,
            &unique_ids,
            &summed,
            corpus.vocab_size,
            D_MODEL,
        );

        let table_evaluated = evaluate_named(
            &table_program,
            &[],
            &[
                ("table", &weights.table),
                ("table_grad", &combined_table_grad),
                ("step", &step_value),
                ("table_m", &table_state.m),
                ("table_v", &table_state.v),
            ],
            &[table_adam.new_param, table_adam.new_m, table_adam.new_v],
        )
        .expect("table adam program lowers and evaluates");

        weights.table = table_evaluated
            .get(table_adam.new_param)
            .expect("requested")
            .0
            .to_vec();
        table_state.m = table_evaluated
            .get(table_adam.new_m)
            .expect("requested")
            .0
            .to_vec();
        table_state.v = table_evaluated
            .get(table_adam.new_v)
            .expect("requested")
            .0
            .to_vec();

        weights.wq = evaluated
            .get(wq_nodes.new_param)
            .expect("requested")
            .0
            .to_vec();
        wq_state.m = evaluated.get(wq_nodes.new_m).expect("requested").0.to_vec();
        wq_state.v = evaluated.get(wq_nodes.new_v).expect("requested").0.to_vec();
        weights.wk = evaluated
            .get(wk_nodes.new_param)
            .expect("requested")
            .0
            .to_vec();
        wk_state.m = evaluated.get(wk_nodes.new_m).expect("requested").0.to_vec();
        wk_state.v = evaluated.get(wk_nodes.new_v).expect("requested").0.to_vec();
        weights.wv = evaluated
            .get(wv_nodes.new_param)
            .expect("requested")
            .0
            .to_vec();
        wv_state.m = evaluated.get(wv_nodes.new_m).expect("requested").0.to_vec();
        wv_state.v = evaluated.get(wv_nodes.new_v).expect("requested").0.to_vec();
        weights.wo = evaluated
            .get(wo_nodes.new_param)
            .expect("requested")
            .0
            .to_vec();
        wo_state.m = evaluated.get(wo_nodes.new_m).expect("requested").0.to_vec();
        wo_state.v = evaluated.get(wo_nodes.new_v).expect("requested").0.to_vec();
        weights.w1 = evaluated
            .get(w1_nodes.new_param)
            .expect("requested")
            .0
            .to_vec();
        w1_state.m = evaluated.get(w1_nodes.new_m).expect("requested").0.to_vec();
        w1_state.v = evaluated.get(w1_nodes.new_v).expect("requested").0.to_vec();
        weights.b1 = evaluated
            .get(b1_nodes.new_param)
            .expect("requested")
            .0
            .to_vec();
        b1_state.m = evaluated.get(b1_nodes.new_m).expect("requested").0.to_vec();
        b1_state.v = evaluated.get(b1_nodes.new_v).expect("requested").0.to_vec();
        weights.w2 = evaluated
            .get(w2_nodes.new_param)
            .expect("requested")
            .0
            .to_vec();
        w2_state.m = evaluated.get(w2_nodes.new_m).expect("requested").0.to_vec();
        w2_state.v = evaluated.get(w2_nodes.new_v).expect("requested").0.to_vec();
        weights.b2 = evaluated
            .get(b2_nodes.new_param)
            .expect("requested")
            .0
            .to_vec();
        b2_state.m = evaluated.get(b2_nodes.new_m).expect("requested").0.to_vec();
        b2_state.v = evaluated.get(b2_nodes.new_v).expect("requested").0.to_vec();
    }

    std::eprintln!("loss curve ({} steps): {loss_curve:?}", loss_curve.len());
    let initial = loss_curve[0];
    let final_loss = *loss_curve.last().expect("at least one step ran");
    std::eprintln!("initial loss {initial}, final loss {final_loss}");
    assert!(
        loss_curve.iter().all(|value| value.is_finite()),
        "loss went non-finite: {loss_curve:?}"
    );
    assert!(
        final_loss < 0.05,
        "expected near-zero overfit loss on {STEPS} steps, got {final_loss} (started at {initial})"
    );

    // Greedy sampling after overfitting: prompt with the corpus's own first
    // six tokens ("Four score and seven years ago"), let the model predict
    // every remaining position on its own, decode through the real
    // tokenizer, and compare against the real corpus text.
    const PROMPT_LEN: usize = 6;
    let zero_onehot = alloc::vec![0.0f32; SEQ_LEN * corpus.vocab_size];
    let mut generated: alloc::vec::Vec<u32> = corpus.compact_ids[0..PROMPT_LEN].to_vec();
    while generated.len() < SEQ_LEN {
        let mut ids_buffer = alloc::vec![0.0f32; SEQ_LEN];
        for (position, &id) in generated.iter().enumerate() {
            ids_buffer[position] = id as f32;
        }
        let evaluated = evaluate_named(
            &network.program,
            &[],
            &[
                ("ids", ids_buffer.as_slice()),
                ("onehot", zero_onehot.as_slice()),
                ("table", &weights.table),
                ("wq", &weights.wq),
                ("wk", &weights.wk),
                ("wv", &weights.wv),
                ("wo", &weights.wo),
                ("w1", &weights.w1),
                ("b1", &weights.b1),
                ("w2", &weights.w2),
                ("b2", &weights.b2),
            ],
            &[network.logits],
        )
        .expect("forward-only program lowers and evaluates");
        let logits = evaluated.get(network.logits).expect("logits requested").0;
        let position = generated.len() - 1;
        let row = &logits[position * corpus.vocab_size..(position + 1) * corpus.vocab_size];
        let (best_index, _) =
            row.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |best, (index, &value)| {
                    if value > best.1 { (index, value) } else { best }
                });
        generated.push(best_index as u32);
    }

    let sampled_text = decode_compact(&generated, &corpus, &vocab);
    let training_text = decode_compact(&corpus.compact_ids[0..SEQ_LEN], &corpus, &vocab);
    std::eprintln!("training text:  {training_text:?}");
    std::eprintln!("sampled text:   {sampled_text:?}");

    let training_prefix: alloc::vec::Vec<u32> = corpus.compact_ids[0..SEQ_LEN].to_vec();
    let matches = generated
        .iter()
        .zip(training_prefix.iter())
        .filter(|(a, b)| a == b)
        .count();
    std::eprintln!("greedy sample matched {matches}/{SEQ_LEN} training positions exactly");
    assert!(
        matches as f32 / SEQ_LEN as f32 > 0.9,
        "expected an overfit model's greedy sample to closely reproduce the training text, matched only {matches}/{SEQ_LEN}"
    );
}
