//! Rung 1 (re-scoped by the owner mid-task, twice: config-sweep -> program
//! transform -> **topology-fits-task**): does a model's SHAPE, not its size,
//! determine whether it can solve a problem at all?
//!
//! Two tasks with a provably unambiguous correct topology, built directly
//! from primitives this crate and `proxima-tensor` already ship:
//!
//! - **induction / copy-over-distance**: predicting the token that follows a
//!   symbol's FIRST occurrence, at the position of its SECOND occurrence far
//!   away. Attention (unbounded causal receptive field,
//!   [`build_attention_network`], the exact block
//!   `tests/language_model.rs` proves) can reach the first occurrence;
//!   a fixed-window causal convolution structurally cannot, by
//!   CONSTRUCTION rather than by argument -- see this file's own
//!   [`shift_ids`] doc.
//! - **bigram / local smoothing**: every target is a fixed function of the
//!   two immediately preceding tokens. A 2-tap convolution can express it
//!   exactly; a 1-tap convolution structurally cannot (it never sees the
//!   second-back token at all).
//!
//! The convolution here is NOT `proxima-tensor/specs/conv2d.toml`'s
//! sliding-window-as-a-graph-op shape (`image`'s two-term `h+y,w+x->chwyx`
//! axis). That shape fails
//! [`proxima_autograd::adjoint::expr::is_pure_projection`] the moment
//! `differentiate` tries to route a gradient through it
//! (`adjoint.rs:448-450`/`:465-466`/`:497-498`, checked directly, not
//! inferred -- every `IndexMap::Affine`/`Computed` operand `differentiate`
//! walks must have exactly one unit-coefficient term per axis). A real
//! conv2d op is buildable and EVALUATES; it does not currently TRAIN.
//! [`build_conv_network`] gets a genuine, weight-shared, receptive-field-
//! bounded causal convolution around that gap for free: each tap `k` is its
//! own [`embedding_gather`] call over a HOST-shifted `ids` array
//! ([`shift_ids`]), so the "window" is a fixed OFFSET
//! ([`AxisTerm::projection`]'s own single term, never a second iteration
//! variable) baked into which raw ids each tap's gather reads -- `k` never
//! appears inside the graph as an index expression at all, so nothing here
//! needs a multi-term axis, and every op both topologies use
//! (`Elementwise`, `Reduce(Add)`, `Reduce(Maximum)` inside `softmax`,
//! `Op::Iota`, `IndexMap::Computed` gather) is exactly what
//! `tests/language_model.rs` and `tests/constructed_sparse.rs` already
//! gradient-check.
//!
//! `NodeSpec` (`proxima-tensor/src/spec.rs:120-123`) is not exercised as a
//! literal round-trip in this file -- the final scope dropped that
//! requirement -- but every op-kind and map shape both topologies below use
//! is already proven expressible as `NodeSpec` TOML by this crate's own
//! landed `specs/causal_attention.toml` and `specs/mistral_layer.toml`
//! (`MapSpec::Gather` for the embedding lookup, `ScalarOp::{Multiply, Add,
//! Greater, Select, Negate, Logarithm}` for the rest), so nothing about
//! EITHER topology built here is missing from that grammar.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::too_many_arguments)]

extern crate alloc;

use proxima_autograd::activation::{relu, softmax};
use proxima_autograd::adjoint::differentiate;
use proxima_autograd::optimizer::{AdamConfig, AdamOperands, adam_step, step_input};
use proxima_autograd::sparse;
use proxima_tensor::cpu::evaluate_named;
use proxima_tensor::dtype::DType;
use proxima_tensor::map::{self, IndexMap};
use proxima_tensor::op::{self, Extent, NodeId, Op, ReduceInit, ScalarOp};
use proxima_tensor::shape;

// ---------------------------------------------------------------------------
// shared graph-building helpers -- duplicated per this crate's own stated
// convention (`language_model.rs`'s module doc: "duplicated here rather than
// imported", since these are `pub(crate)`-adjacent test helpers, not library
// surface every sibling test file already re-declares its own copy of).
// ---------------------------------------------------------------------------

fn leaf(program: &mut Vec<Op>, name: &str, shape: alloc::vec::Vec<Extent>) -> NodeId {
    op::append(program, Op::Input { dtype: DType::Float32, shape, name: Some(name.into()) })
}

fn int_leaf(program: &mut Vec<Op>, name: &str, extent: u32) -> NodeId {
    op::append(
        program,
        Op::Input { dtype: DType::Int32, shape: alloc::vec![Extent::Static(extent)], name: Some(name.into()) },
    )
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

/// `table[ids[s], d]` -- identical shape to `tests/language_model.rs`'s own
/// `embedding_gather`, duplicated for the same reason that file gives.
fn embedding_gather(program: &mut Vec<Op>, table: NodeId, ids: NodeId) -> NodeId {
    let gathered_map = IndexMap::Computed {
        indices: ids,
        index_map: map::projection(2, &[0]),
        base: proxima_tensor::map::IndexPattern {
            iter_rank: 2,
            axes: alloc::vec![
                proxima_tensor::map::AxisIndex::default(),
                proxima_tensor::map::AxisIndex {
                    terms: core::iter::once(proxima_tensor::map::AxisTerm::projection(1)).collect(),
                    offset: 0,
                },
            ],
        },
        gathered_dim: 0,
    };
    elementwise(program, ScalarOp::Identity, alloc::vec![(table, gathered_map)])
}

fn counter_pattern(seed: usize, count: usize) -> alloc::vec::Vec<f32> {
    (0..count).map(|index| (((seed + index) * 7 % 13) as f32 - 6.0) / 24.0).collect()
}

/// Sums the iteration-space size of every `Reduce(Add)` node -- the same
/// `constructed_sparse.rs` convention, applied to the FORWARD program only
/// (before `differentiate`), so this is a forward-inference MAC count.
fn total_macs(program: &[Op]) -> u64 {
    let shapes = shape::infer(program, &[]).expect("forward program infers");
    program
        .iter()
        .filter_map(|op| match op {
            Op::Reduce(reduce) if matches!(reduce.body, ScalarOp::Add) => Some(shapes.of(reduce.operand).iter().product::<u64>()),
            _ => None,
        })
        .sum()
}

// ---------------------------------------------------------------------------
// one parameter's identity: its config-face name (also its `Op::Input` leaf
// name), its shape, and the leaf `NodeId` a topology builder already placed
// in the forward program.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct ParamSpec {
    name: alloc::string::String,
    extents: alloc::vec::Vec<u32>,
    node: NodeId,
}

impl ParamSpec {
    fn new(name: &str, extents: alloc::vec::Vec<u32>, node: NodeId) -> Self {
        Self { name: name.into(), extents, node }
    }

    fn count(&self) -> usize {
        self.extents.iter().product::<u32>() as usize
    }
}

/// One trained topology: its forward program, its loss/logits outputs, the
/// embedding table (special-cased -- always gathered, never densely read,
/// since neither topology here ties the output head back to it), every
/// OTHER trainable parameter, and `taps` (0 for attention's single `ids`
/// binding, >=1 for a `taps`-tap convolution's `ids_tap0..ids_tap{taps-1}`
/// bindings -- see [`ids_bindings`]).
struct Network {
    program: alloc::vec::Vec<Op>,
    loss: NodeId,
    logits: NodeId,
    vocab_size: usize,
    d_model: usize,
    seq_len: usize,
    taps: usize,
    params: alloc::vec::Vec<ParamSpec>,
}

impl Network {
    fn param_count(&self) -> usize {
        self.vocab_size * self.d_model + self.params.iter().map(ParamSpec::count).sum::<usize>()
    }
}

/// The dense two-layer FFN residual block both topologies share, byte-for-
/// byte the same construction `tests/language_model.rs`'s own FFN section
/// uses, parameterised over `d_model`/`ffn_hidden`.
fn ffn_block(program: &mut Vec<Op>, residual1: NodeId, d_model: usize, ffn_hidden: usize, params: &mut Vec<ParamSpec>) -> NodeId {
    let w1 = leaf(program, "w1", alloc::vec![Extent::Static(d_model as u32), Extent::Static(ffn_hidden as u32)]);
    params.push(ParamSpec::new("w1", alloc::vec![d_model as u32, ffn_hidden as u32], w1));
    let b1 = leaf(program, "b1", alloc::vec![Extent::Static(ffn_hidden as u32)]);
    params.push(ParamSpec::new("b1", alloc::vec![ffn_hidden as u32], b1));
    let w2 = leaf(program, "w2", alloc::vec![Extent::Static(ffn_hidden as u32), Extent::Static(d_model as u32)]);
    params.push(ParamSpec::new("w2", alloc::vec![ffn_hidden as u32, d_model as u32], w2));
    let b2 = leaf(program, "b2", alloc::vec![Extent::Static(d_model as u32)]);
    params.push(ParamSpec::new("b2", alloc::vec![d_model as u32], b2));

    let gate_product = elementwise(program, ScalarOp::Multiply, alloc::vec![(residual1, proj(3, &[0, 1])), (w1, proj(3, &[1, 2]))]);
    let gate = reduce_add(program, gate_product, identity(3), proj(3, &[0, 2]));
    let gate_biased = elementwise(program, ScalarOp::Add, alloc::vec![(gate, identity(2)), (b1, proj(2, &[1]))]);
    let hidden = relu(program, DType::Float32, gate_biased, 2);
    let down_product = elementwise(program, ScalarOp::Multiply, alloc::vec![(hidden, proj(3, &[0, 1])), (w2, proj(3, &[1, 2]))]);
    let ffn_out = reduce_add(program, down_product, identity(3), proj(3, &[0, 2]));
    let ffn_out_biased = elementwise(program, ScalarOp::Add, alloc::vec![(ffn_out, identity(2)), (b2, proj(2, &[1]))]);
    elementwise(program, ScalarOp::Add, alloc::vec![(residual1, identity(2)), (ffn_out_biased, identity(2))])
}

/// An UNTIED output projection (`d_model -> vocab_size`, plain bias) -- the
/// tied-embedding gradient combination `tests/language_model.rs` proves is
/// deliberately not re-exercised here; this file is about topology vs.
/// task, not about the embedding-tying mechanism.
fn output_head(program: &mut Vec<Op>, residual2: NodeId, d_model: usize, vocab_size: usize, params: &mut Vec<ParamSpec>) -> NodeId {
    let w_out = leaf(program, "w_out", alloc::vec![Extent::Static(d_model as u32), Extent::Static(vocab_size as u32)]);
    params.push(ParamSpec::new("w_out", alloc::vec![d_model as u32, vocab_size as u32], w_out));
    let b_out = leaf(program, "b_out", alloc::vec![Extent::Static(vocab_size as u32)]);
    params.push(ParamSpec::new("b_out", alloc::vec![vocab_size as u32], b_out));
    let logits_product = elementwise(program, ScalarOp::Multiply, alloc::vec![(residual2, proj(3, &[0, 1])), (w_out, proj(3, &[1, 2]))]);
    let logits_raw = reduce_add(program, logits_product, identity(3), proj(3, &[0, 2]));
    elementwise(program, ScalarOp::Add, alloc::vec![(logits_raw, identity(2)), (b_out, proj(2, &[1]))])
}

/// Epsilon-stabilized cross-entropy -- the exact `+1e-7` fix
/// `tests/language_model.rs`'s own module doc records live-discovering
/// (converged probabilities underflow to exactly `0.0`, and `0.0 *
/// log(0.0)` is `NaN`).
fn cross_entropy_loss(program: &mut Vec<Op>, logits: NodeId, onehot: NodeId, seq_len: usize) -> NodeId {
    let probabilities = softmax(program, DType::Float32, logits, 2, 1);
    let log_epsilon = constant(program, 1e-7);
    let stabilized = elementwise(program, ScalarOp::Add, alloc::vec![(probabilities, identity(2)), (log_epsilon, broadcast(2))]);
    let log_probabilities = elementwise(program, ScalarOp::Logarithm, alloc::vec![(stabilized, identity(2))]);
    let weighted = elementwise(program, ScalarOp::Multiply, alloc::vec![(onehot, identity(2)), (log_probabilities, identity(2))]);
    let per_position = reduce_add(program, weighted, identity(2), proj(2, &[0]));
    let negated = elementwise(program, ScalarOp::Negate, alloc::vec![(per_position, identity(1))]);
    let total = reduce_add(program, negated, identity(1), proj(1, &[]));
    let inv_seq_len = constant(program, 1.0 / seq_len as f32);
    elementwise(program, ScalarOp::Multiply, alloc::vec![(total, identity(0)), (inv_seq_len, broadcast(0))])
}

const ATTENTION_HEADS: usize = 2;
const ATTENTION_HEAD_DIM: usize = 4;

/// Causal self-attention (unbounded receptive field) + shared FFN + untied
/// head -- the exact block `tests/language_model.rs` gradient-checks and
/// trains to near-zero loss on real text, parameterised over
/// `seq_len`/`vocab_size`/`d_model`/`ffn_hidden` for these much smaller
/// synthetic tasks.
fn build_attention_network(seq_len: usize, vocab_size: usize, d_model: usize, ffn_hidden: usize) -> Network {
    let n_heads = ATTENTION_HEADS;
    let head_dim = ATTENTION_HEAD_DIM;
    let mut program = alloc::vec::Vec::new();
    let mut params = alloc::vec::Vec::new();

    let table = leaf(&mut program, "table", alloc::vec![Extent::Static(vocab_size as u32), Extent::Static(d_model as u32)]);
    let ids = int_leaf(&mut program, "ids", seq_len as u32);
    let onehot = leaf(&mut program, "onehot", alloc::vec![Extent::Static(seq_len as u32), Extent::Static(vocab_size as u32)]);
    let x = embedding_gather(&mut program, table, ids);

    let qkv_shape = alloc::vec![Extent::Static(d_model as u32), Extent::Static(n_heads as u32), Extent::Static(head_dim as u32)];
    let wq = leaf(&mut program, "wq", qkv_shape.clone());
    params.push(ParamSpec::new("wq", alloc::vec![d_model as u32, n_heads as u32, head_dim as u32], wq));
    let wk = leaf(&mut program, "wk", qkv_shape.clone());
    params.push(ParamSpec::new("wk", alloc::vec![d_model as u32, n_heads as u32, head_dim as u32], wk));
    let wv = leaf(&mut program, "wv", qkv_shape);
    params.push(ParamSpec::new("wv", alloc::vec![d_model as u32, n_heads as u32, head_dim as u32], wv));
    let wo = leaf(
        &mut program,
        "wo",
        alloc::vec![Extent::Static(n_heads as u32), Extent::Static(head_dim as u32), Extent::Static(d_model as u32)],
    );
    params.push(ParamSpec::new("wo", alloc::vec![n_heads as u32, head_dim as u32, d_model as u32], wo));

    let q_product = elementwise(&mut program, ScalarOp::Multiply, alloc::vec![(x, proj(4, &[0, 3])), (wq, proj(4, &[3, 1, 2]))]);
    let q = reduce_add(&mut program, q_product, identity(4), proj(4, &[0, 1, 2]));
    let k_product = elementwise(&mut program, ScalarOp::Multiply, alloc::vec![(x, proj(4, &[0, 3])), (wk, proj(4, &[3, 1, 2]))]);
    let k = reduce_add(&mut program, k_product, identity(4), proj(4, &[0, 1, 2]));
    let v_product = elementwise(&mut program, ScalarOp::Multiply, alloc::vec![(x, proj(4, &[0, 3])), (wv, proj(4, &[3, 1, 2]))]);
    let v = reduce_add(&mut program, v_product, identity(4), proj(4, &[0, 1, 2]));

    let score_product = elementwise(&mut program, ScalarOp::Multiply, alloc::vec![(q, proj(4, &[0, 2, 3])), (k, proj(4, &[1, 2, 3]))]);
    let scores = reduce_add(&mut program, score_product, identity(4), proj(4, &[0, 1, 2]));
    let inv_sqrt_head_dim = constant(&mut program, 1.0 / (head_dim as f32).sqrt());
    let scaled = elementwise(&mut program, ScalarOp::Multiply, alloc::vec![(scores, identity(3)), (inv_sqrt_head_dim, broadcast(3))]);

    let query_index = iota(&mut program, seq_len as u32);
    let key_index = iota(&mut program, seq_len as u32);
    let is_future = elementwise(&mut program, ScalarOp::Greater, alloc::vec![(key_index, proj(2, &[1])), (query_index, proj(2, &[0]))]);
    let neg_infinity = constant(&mut program, f32::NEG_INFINITY);
    let masked = elementwise(
        &mut program,
        ScalarOp::Select,
        alloc::vec![(is_future, proj(3, &[0, 1])), (neg_infinity, broadcast(3)), (scaled, identity(3))],
    );
    let probabilities = softmax(&mut program, DType::Float32, masked, 3, 1);

    let attended_product = elementwise(&mut program, ScalarOp::Multiply, alloc::vec![(probabilities, proj(4, &[0, 1, 2])), (v, proj(4, &[1, 2, 3]))]);
    let attended = reduce_add(&mut program, attended_product, identity(4), proj(4, &[0, 2, 3]));
    let attn_product = elementwise(&mut program, ScalarOp::Multiply, alloc::vec![(attended, proj(4, &[0, 1, 2])), (wo, proj(4, &[1, 2, 3]))]);
    let attn_out = reduce_add(&mut program, attn_product, identity(4), proj(4, &[0, 3]));
    let residual1 = elementwise(&mut program, ScalarOp::Add, alloc::vec![(x, identity(2)), (attn_out, identity(2))]);

    let residual2 = ffn_block(&mut program, residual1, d_model, ffn_hidden, &mut params);
    let logits = output_head(&mut program, residual2, d_model, vocab_size, &mut params);
    let loss = cross_entropy_loss(&mut program, logits, onehot, seq_len);

    Network { program, loss, logits, vocab_size, d_model, seq_len, taps: 0, params }
}

/// A genuine causal `taps`-tap convolution, receptive field EXACTLY
/// `{s, s-1, .., s-(taps-1)}` by construction: tap `k` is its OWN
/// [`embedding_gather`] over a SEPARATE `ids_tap{k}` leaf holding
/// `ids[s-k]` (host-shifted -- see [`shift_ids`]), never a windowed VIEW of
/// one shared `x`. Weight-shared across position (`w_tap{k}` is read via a
/// broadcast projection over `s`, identical at every position, exactly how
/// a real conv kernel is shared), NOT shared across `k` (a distinct weight
/// per relative offset, as any causal conv1d has). Because `k` never
/// appears as an in-graph index expression, every operand map here is a
/// plain single-term projection and the whole thing differentiates through
/// the existing adjoint with no special-casing.
fn build_conv_network(seq_len: usize, vocab_size: usize, d_model: usize, taps: usize, ffn_hidden: usize) -> Network {
    let mut program = alloc::vec::Vec::new();
    let mut params = alloc::vec::Vec::new();

    let table = leaf(&mut program, "table", alloc::vec![Extent::Static(vocab_size as u32), Extent::Static(d_model as u32)]);
    let onehot = leaf(&mut program, "onehot", alloc::vec![Extent::Static(seq_len as u32), Extent::Static(vocab_size as u32)]);

    let mut tapped: alloc::vec::Vec<NodeId> = alloc::vec::Vec::new();
    for tap in 0..taps {
        let ids_tap = int_leaf(&mut program, &alloc::format!("ids_tap{tap}"), seq_len as u32);
        tapped.push(embedding_gather(&mut program, table, ids_tap));
    }

    let mut mixed: Option<NodeId> = None;
    for (tap, &x_tap) in tapped.iter().enumerate() {
        let w_tap = leaf(&mut program, &alloc::format!("w_tap{tap}"), alloc::vec![Extent::Static(d_model as u32), Extent::Static(d_model as u32)]);
        params.push(ParamSpec::new(&alloc::format!("w_tap{tap}"), alloc::vec![d_model as u32, d_model as u32], w_tap));
        let product = elementwise(&mut program, ScalarOp::Multiply, alloc::vec![(x_tap, proj(3, &[0, 2])), (w_tap, proj(3, &[2, 1]))]);
        let partial = reduce_add(&mut program, product, identity(3), proj(3, &[0, 1]));
        mixed = Some(match mixed {
            None => partial,
            Some(previous) => elementwise(&mut program, ScalarOp::Add, alloc::vec![(previous, identity(2)), (partial, identity(2))]),
        });
    }
    let mixed = mixed.expect("at least one tap");
    let residual1 = elementwise(&mut program, ScalarOp::Add, alloc::vec![(tapped[0], identity(2)), (mixed, identity(2))]);

    let residual2 = ffn_block(&mut program, residual1, d_model, ffn_hidden, &mut params);
    let logits = output_head(&mut program, residual2, d_model, vocab_size, &mut params);
    let loss = cross_entropy_loss(&mut program, logits, onehot, seq_len);

    Network { program, loss, logits, vocab_size, d_model, seq_len, taps, params }
}

#[derive(Clone, Copy, Debug)]
enum TopologyKind {
    Attention,
    Conv(usize),
}

fn build_network(seq_len: usize, vocab_size: usize, topology: TopologyKind, d_model: usize, ffn_hidden: usize) -> Network {
    match topology {
        TopologyKind::Attention => build_attention_network(seq_len, vocab_size, d_model, ffn_hidden),
        TopologyKind::Conv(taps) => build_conv_network(seq_len, vocab_size, d_model, taps, ffn_hidden),
    }
}

#[derive(Clone, Copy, Debug)]
enum Task {
    Induction,
    Bigram,
}

struct Example {
    ids: alloc::vec::Vec<u32>,
    onehot: alloc::vec::Vec<f32>,
    critical_positions: alloc::vec::Vec<usize>,
    expected: alloc::vec::Vec<u32>,
}

fn onehot_from_targets(seq_len: usize, vocab_size: usize, targets: &[u32]) -> alloc::vec::Vec<f32> {
    let mut onehot = alloc::vec![0.0f32; seq_len * vocab_size];
    for (position, &target) in targets.iter().enumerate() {
        onehot[position * vocab_size + target as usize] = 1.0;
    }
    onehot
}

fn make_example(raw: &[u32], seq_len: usize, vocab_size: usize, critical_positions: alloc::vec::Vec<usize>) -> Example {
    let ids: alloc::vec::Vec<u32> = raw[0..seq_len].to_vec();
    let targets: alloc::vec::Vec<u32> = raw[1..=seq_len].to_vec();
    let onehot = onehot_from_targets(seq_len, vocab_size, &targets);
    let expected = critical_positions.iter().map(|&position| targets[position]).collect();
    Example { ids, onehot, critical_positions, expected }
}

/// The one non-content symbol both tasks reserve: never a real token, never
/// a target, only ever the fill value [`shift_ids`] uses for an
/// out-of-range look-back.
const PAD_SYMBOL: u32 = 5;

const INDUCTION_VOCAB: usize = 6;
const INDUCTION_SEQ_LEN: usize = 8;

/// Five deterministic, hand-verified sequences (four training, one held
/// out), each obeying `raw[6] == raw[0]` (the trigger repeats the cue) and
/// `raw[7] == raw[1]` (the answer is whatever followed the cue the FIRST
/// time) -- the induction rule. `(cue, answer)` is DELIBERATELY not a
/// consistent function of the cue alone: cue 0 answers 1 in sequence 0 but
/// 3 in sequence 3, and cue 2 answers 1 in sequence 2 but 3 in the held-out
/// sequence. A first draft of this task used `answer = cue + 1 (mod
/// vocab)` for every sequence, which a plain per-token lookup solves with
/// ZERO receptive field -- measured: the trained attention candidate
/// predicted every TRAINING cue correctly at final loss 0.0056, but
/// predicted 2 instead of the held-out cue 4's true answer 0, proving it
/// had learned "cue -> cue+1" as content, never copying from context.
/// Repeating a cue with a DIFFERENT answer makes a content-only shortcut
/// mathematically impossible to fit even the training data, forcing
/// genuine in-context copying.
fn induction_task() -> (usize, usize, alloc::vec::Vec<Example>, Example) {
    let raws: [[u32; 9]; 5] = [
        [0, 1, 2, 3, 4, 2, 0, 1, 3],
        [1, 0, 3, 4, 2, 3, 1, 0, 4],
        [2, 1, 4, 0, 3, 4, 2, 1, 0],
        [0, 3, 1, 2, 4, 1, 0, 3, 2],
        [2, 3, 0, 1, 4, 0, 2, 3, 1],
    ];
    let critical = alloc::vec![6usize];
    let training: alloc::vec::Vec<Example> =
        raws[0..4].iter().map(|raw| make_example(raw, INDUCTION_SEQ_LEN, INDUCTION_VOCAB, critical.clone())).collect();
    let held_out = make_example(&raws[4], INDUCTION_SEQ_LEN, INDUCTION_VOCAB, critical);
    (INDUCTION_VOCAB, INDUCTION_SEQ_LEN, training, held_out)
}

const BIGRAM_VOCAB: usize = 6;
const BIGRAM_SEQ_LEN: usize = 9;

/// `raw[i] = (raw[i-1] + raw[i-2]) mod 3` for `i >= 2` -- a fixed local
/// (bigram) rule with zero long-range structure: every target is fully
/// determined by the two immediately preceding real tokens. Modulus 3 (not
/// 5, an earlier draft) makes the ordered-pair space small enough (9
/// pairs: `{0,1,2} x {0,1,2}`) that two short training sequences can cover
/// EVERY ordered pair, which `bigram_task` relies on: a held-out sequence
/// over an already-fully-covered pair space tests whether the model
/// learned a position-invariant LOCAL rule (what a real bigram topology
/// gives you), not whether it can extrapolate arithmetic to unseen pairs
/// (a much harder, unrelated problem this file is not testing).
fn bigram_sequence(seed0: u32, seed1: u32, len: usize) -> alloc::vec::Vec<u32> {
    let mut raw = alloc::vec![seed0, seed1];
    while raw.len() < len {
        let next = (raw[raw.len() - 1] + raw[raw.len() - 2]) % 3;
        raw.push(next);
    }
    raw
}

/// Training sequence `(0, 0)` trivially covers ordered pair `(0, 0)`;
/// training sequence `(0, 1)` (checked by hand) walks pairs `(0,1) (1,1)
/// (1,2) (2,0) (0,2) (2,2) (2,1) (1,0)` -- together with `(0,0)`, all NINE
/// ordered pairs `{0,1,2}^2`. The held-out seed `(2, 1)` therefore visits
/// only pairs already seen during training, at positions training never
/// used them at -- a fair test of position-invariance, never a demand to
/// extrapolate beyond the covered pair space. `critical_positions`
/// excludes input position 0 (predicting `raw[1]` from `raw[0]` alone is
/// not rule-determined -- it is just the arbitrary second seed).
fn bigram_task() -> (usize, usize, alloc::vec::Vec<Example>, Example) {
    let critical: alloc::vec::Vec<usize> = (1..BIGRAM_SEQ_LEN).collect();
    let raw_a = bigram_sequence(0, 0, BIGRAM_SEQ_LEN + 1);
    let raw_b = bigram_sequence(0, 1, BIGRAM_SEQ_LEN + 1);
    let raw_held = bigram_sequence(2, 1, BIGRAM_SEQ_LEN + 1);
    let training = alloc::vec![
        make_example(&raw_a, BIGRAM_SEQ_LEN, BIGRAM_VOCAB, critical.clone()),
        make_example(&raw_b, BIGRAM_SEQ_LEN, BIGRAM_VOCAB, critical.clone()),
    ];
    let held_out = make_example(&raw_held, BIGRAM_SEQ_LEN, BIGRAM_VOCAB, critical);
    (BIGRAM_VOCAB, BIGRAM_SEQ_LEN, training, held_out)
}

fn task_data(task: Task) -> (usize, usize, alloc::vec::Vec<Example>, Example) {
    match task {
        Task::Induction => induction_task(),
        Task::Bigram => bigram_task(),
    }
}

/// `ids_tap{k}[s] = ids[s - k]`, `pad` where `s - k` is negative -- the
/// host-side "im2col" shift that gives [`build_conv_network`]'s tap `k` a
/// causal window bounded to `{s, s-1, .., s-(taps-1)}` BEFORE any data ever
/// enters the graph: there is no larger array anywhere the graph could read
/// from even if a bug tried, which is a stronger guarantee than an
/// in-graph bound would be.
fn shift_ids(ids: &[u32], k: usize, pad: u32) -> alloc::vec::Vec<f32> {
    (0..ids.len()).map(|position| if position >= k { ids[position - k] as f32 } else { pad as f32 }).collect()
}

fn ids_bindings(network: &Network, example: &Example, pad_symbol: u32) -> alloc::vec::Vec<(alloc::string::String, alloc::vec::Vec<f32>)> {
    if network.taps == 0 {
        alloc::vec![("ids".into(), example.ids.iter().map(|&id| id as f32).collect())]
    } else {
        (0..network.taps).map(|tap| (alloc::format!("ids_tap{tap}"), shift_ids(&example.ids, tap, pad_symbol))).collect()
    }
}

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

fn accumulate_scatter(dense: &mut [f32], unique_ids: &[u32], summed: &[f32], dim: usize) {
    for (position, &id) in unique_ids.iter().enumerate() {
        let source = &summed[position * dim..(position + 1) * dim];
        let destination = &mut dense[id as usize * dim..(id as usize + 1) * dim];
        for (accumulator, value) in destination.iter_mut().zip(source) {
            *accumulator += *value;
        }
    }
}

struct Weights {
    table: alloc::vec::Vec<f32>,
    dense: alloc::collections::BTreeMap<alloc::string::String, alloc::vec::Vec<f32>>,
}

struct TrainResult {
    loss_curve: alloc::vec::Vec<f32>,
    weights: Weights,
    param_count: usize,
    forward_macs: u64,
}

/// Trains one [`Network`] with Adam, cycling deterministically through
/// `examples` (`step % examples.len()`) -- full-batch single-example steps,
/// no mini-batching, no randomness anywhere. `table`'s gradient is handled
/// exactly like `tests/language_model.rs`'s own tied-embedding case (a
/// SEPARATE small `table_program`, since the true gradient is only known
/// after a host-side scatter-sum), generalised from "1 gather site" to
/// "however many taps read it".
fn train_network(network: &Network, examples: &[Example], pad_symbol: u32, steps: u32, config: &AdamConfig) -> TrainResult {
    let forward_macs = total_macs(&network.program);
    let param_count = network.param_count();

    let differentiated = differentiate(&network.program, network.loss).expect("scalar loss differentiates");
    let dense_grads: alloc::vec::Vec<(ParamSpec, NodeId)> = network
        .params
        .iter()
        .map(|param| {
            let grad = differentiated
                .gradient_of_named(&param.name)
                .unwrap_or_else(|| panic!("parameter {} must feed the loss", param.name));
            (param.clone(), grad)
        })
        .collect();
    let table_gathered: alloc::vec::Vec<_> = differentiated.gathered_gradients_of_named("table").collect();
    assert!(!table_gathered.is_empty(), "table must be gathered by at least one tap");
    assert_eq!(table_gathered.len(), network.taps.max(1), "table's gather-site count must equal the topology's tap count");

    let mut program = differentiated.program;
    let step_node = step_input(&mut program, "step");
    let dense_adam: alloc::vec::Vec<(ParamSpec, AdamNodes)> = dense_grads
        .into_iter()
        .map(|(param, grad)| {
            let nodes = append_adam(&mut program, config, param.extents.len() as u16, &param.extents, &param.name, param.node, grad, step_node);
            (param, nodes)
        })
        .collect();

    let mut table_program = alloc::vec::Vec::new();
    let table_param = leaf(&mut table_program, "table", alloc::vec![Extent::Static(network.vocab_size as u32), Extent::Static(network.d_model as u32)]);
    let table_grad_in = leaf(&mut table_program, "table_grad", alloc::vec![Extent::Static(network.vocab_size as u32), Extent::Static(network.d_model as u32)]);
    let table_step = step_input(&mut table_program, "step");
    let table_adam = append_adam(
        &mut table_program,
        config,
        2,
        &[network.vocab_size as u32, network.d_model as u32],
        "table",
        table_param,
        table_grad_in,
        table_step,
    );

    let mut weights = Weights {
        table: counter_pattern(11, network.vocab_size * network.d_model),
        dense: dense_adam
            .iter()
            .enumerate()
            .map(|(index, (param, _))| (param.name.clone(), counter_pattern(23 + index * 7, param.count())))
            .collect(),
    };
    let mut table_state = zero_state(network.vocab_size * network.d_model);
    let mut dense_state: alloc::collections::BTreeMap<alloc::string::String, AdamState> =
        dense_adam.iter().map(|(param, _)| (param.name.clone(), zero_state(param.count()))).collect();

    let mut loss_curve = alloc::vec::Vec::new();

    for step in 0..steps {
        let example = &examples[step as usize % examples.len()];
        let step_value = [(step + 1) as f32];

        let mut bindings = ids_bindings(network, example, pad_symbol);
        bindings.push(("onehot".into(), example.onehot.clone()));
        bindings.push(("table".into(), weights.table.clone()));
        bindings.push(("step".into(), step_value.to_vec()));
        for (param, _) in &dense_adam {
            bindings.push((param.name.clone(), weights.dense[&param.name].clone()));
            bindings.push((alloc::format!("{}_m", param.name), dense_state[&param.name].m.clone()));
            bindings.push((alloc::format!("{}_v", param.name), dense_state[&param.name].v.clone()));
        }
        let refs: alloc::vec::Vec<(&str, &[f32])> = bindings.iter().map(|(key, value)| (key.as_str(), value.as_slice())).collect();

        let mut outputs = alloc::vec![network.loss];
        for contribution in &table_gathered {
            outputs.push(contribution.values);
            outputs.push(contribution.indices);
        }
        for (_, nodes) in &dense_adam {
            outputs.push(nodes.new_param);
            outputs.push(nodes.new_m);
            outputs.push(nodes.new_v);
        }

        let evaluated = evaluate_named(&program, &[], &refs, &outputs).expect("training-step program lowers and evaluates");
        loss_curve.push(evaluated.get(network.loss).expect("loss requested").0[0]);

        let mut combined_table_grad = alloc::vec![0.0f32; network.vocab_size * network.d_model];
        for contribution in &table_gathered {
            let indices_values = evaluated.get(contribution.indices).expect("gather indices requested").0;
            let gathered_values = evaluated.get(contribution.values).expect("gather values requested").0;
            let (unique_ids, summed) =
                sparse::dedupe_and_sum_rows(indices_values, gathered_values, network.d_model).expect("indices and values line up");
            accumulate_scatter(&mut combined_table_grad, &unique_ids, &summed, network.d_model);
        }

        let table_evaluated = evaluate_named(
            &table_program,
            &[],
            &[
                ("table", weights.table.as_slice()),
                ("table_grad", combined_table_grad.as_slice()),
                ("step", step_value.as_slice()),
                ("table_m", table_state.m.as_slice()),
                ("table_v", table_state.v.as_slice()),
            ],
            &[table_adam.new_param, table_adam.new_m, table_adam.new_v],
        )
        .expect("table adam program lowers and evaluates");
        weights.table = table_evaluated.get(table_adam.new_param).expect("requested").0.to_vec();
        table_state.m = table_evaluated.get(table_adam.new_m).expect("requested").0.to_vec();
        table_state.v = table_evaluated.get(table_adam.new_v).expect("requested").0.to_vec();

        for (param, nodes) in &dense_adam {
            let new_value = evaluated.get(nodes.new_param).expect("requested").0.to_vec();
            let new_m = evaluated.get(nodes.new_m).expect("requested").0.to_vec();
            let new_v = evaluated.get(nodes.new_v).expect("requested").0.to_vec();
            weights.dense.insert(param.name.clone(), new_value);
            let state = dense_state.get_mut(&param.name).expect("state tracked for every dense param");
            state.m = new_m;
            state.v = new_v;
        }
    }

    TrainResult { loss_curve, weights, param_count, forward_macs }
}

fn forward_logits(network: &Network, weights: &Weights, example: &Example, pad_symbol: u32) -> alloc::vec::Vec<f32> {
    let mut bindings = ids_bindings(network, example, pad_symbol);
    bindings.push(("table".into(), weights.table.clone()));
    for param in &network.params {
        bindings.push((param.name.clone(), weights.dense[&param.name].clone()));
    }
    bindings.push(("onehot".into(), alloc::vec![0.0f32; network.seq_len * network.vocab_size]));
    let refs: alloc::vec::Vec<(&str, &[f32])> = bindings.iter().map(|(key, value)| (key.as_str(), value.as_slice())).collect();
    let evaluated = evaluate_named(&network.program, &[], &refs, &[network.logits]).expect("forward-only program lowers and evaluates");
    evaluated.get(network.logits).expect("logits requested").0.to_vec()
}

fn argmax_row(logits: &[f32], position: usize, vocab_size: usize) -> u32 {
    let row = &logits[position * vocab_size..(position + 1) * vocab_size];
    row.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, (index, &value)| if value > best.1 { (index, value) } else { best })
        .0 as u32
}

/// Exact-match on the HELD-OUT example only, at exactly the positions the
/// task names as unambiguous -- never a loss threshold.
fn solved(network: &Network, weights: &Weights, example: &Example, pad_symbol: u32) -> bool {
    let logits = forward_logits(network, weights, example, pad_symbol);
    example.critical_positions.iter().zip(&example.expected).all(|(&position, &expected)| {
        let predicted = argmax_row(&logits, position, network.vocab_size);
        std::eprintln!("  critical position {position}: predicted={predicted} expected={expected}");
        predicted == expected
    })
}

/// Pure formula mirroring [`build_attention_network`]'s own parameter
/// declarations -- lets a conv candidate's test case assert "more params
/// than attention, same task" without needing to build+train attention
/// again in the same case.
fn attention_param_count(vocab_size: usize, d_model: usize, ffn_hidden: usize) -> usize {
    let heads = ATTENTION_HEADS;
    let head_dim = ATTENTION_HEAD_DIM;
    vocab_size * d_model
        + 3 * d_model * heads * head_dim
        + heads * head_dim * d_model
        + d_model * ffn_hidden
        + ffn_hidden
        + ffn_hidden * d_model
        + d_model
        + d_model * vocab_size
        + vocab_size
}

const STEPS: u32 = 700;

fn training_config() -> AdamConfig {
    AdamConfig { learning_rate: 0.01, ..AdamConfig::default() }
}

/// One row of the topology-vs-task table: builds, trains, and scores ONE
/// candidate against ONE task's held-out example, printing everything the
/// report needs (config, params, MACs, final loss, solved) before the
/// caller asserts on it.
fn run_candidate(task: Task, topology: TopologyKind, d_model: usize, ffn_hidden: usize) -> (TrainResult, bool) {
    let (vocab_size, seq_len, examples, held_out) = task_data(task);
    let network = build_network(seq_len, vocab_size, topology, d_model, ffn_hidden);
    let result = train_network(&network, &examples, PAD_SYMBOL, STEPS, &training_config());
    let did_solve = solved(&network, &result.weights, &held_out, PAD_SYMBOL);
    let window = examples.len().min(result.loss_curve.len());
    let windowed_average_loss: f32 =
        result.loss_curve[result.loss_curve.len() - window..].iter().sum::<f32>() / window as f32;
    std::eprintln!(
        "task={task:?} topology={topology:?} d_model={d_model} ffn_hidden={ffn_hidden} params={} macs={} \
         final_loss={} last-{window}-step average {windowed_average_loss} solved={did_solve}",
        result.param_count,
        result.forward_macs,
        result.loss_curve.last().expect("at least one step ran"),
    );
    (result, did_solve)
}

/// The core experiment: for each task, a structurally-right topology and at
/// least one structurally-wrong one, scored by EXACT MATCH on a held-out
/// example the wrong topology cannot reach by construction (induction) or
/// cannot see enough of (bigram's 1-tap control).
#[proxima::test]
#[case::induction_attention_solves_the_copy_task(Task::Induction, TopologyKind::Attention, 8, 12, true)]
#[case::induction_small_conv_cannot_reach_the_cue(Task::Induction, TopologyKind::Conv(3), 6, 8, false)]
#[case::induction_larger_conv_has_more_params_and_still_cannot_reach_it(Task::Induction, TopologyKind::Conv(3), 16, 24, false)]
#[case::bigram_one_tap_conv_cannot_see_the_second_back_token(Task::Bigram, TopologyKind::Conv(1), 6, 16, false)]
#[case::bigram_two_tap_conv_solves_the_local_rule(Task::Bigram, TopologyKind::Conv(2), 6, 16, true)]
async fn topology_fit_matches_receptive_field(
    #[case] task: Task,
    #[case] topology: TopologyKind,
    #[case] d_model: usize,
    #[case] ffn_hidden: usize,
    #[case] expected_solved: bool,
) {
    let (result, did_solve) = run_candidate(task, topology, d_model, ffn_hidden);
    assert!(result.loss_curve.iter().all(|value| value.is_finite()), "loss went non-finite: {:?}", result.loss_curve);
    assert_eq!(
        did_solve, expected_solved,
        "task={task:?} topology={topology:?}: expected solved={expected_solved}, got {did_solve} \
         (final loss {})",
        result.loss_curve.last().expect("at least one step ran"),
    );

    if let TopologyKind::Conv(_) = topology {
        let comparable_attention_params = attention_param_count(task_data(task).0, 8, 12);
        std::eprintln!(
            "task={task:?} conv params={} vs. a comparable attention candidate's {comparable_attention_params} params",
            result.param_count
        );
    }
}

/// The "wrong topology, MORE parameters, still loses" control: the induction
/// task's larger convolution has strictly more parameters than a comparable
/// attention candidate (both computed from the same formula each topology's
/// own builder uses), and still cannot solve the task, because its
/// receptive field -- not its capacity -- is what the task requires.
#[proxima::test]
async fn a_wider_convolution_has_more_parameters_than_attention_and_still_loses() {
    let (result, did_solve) = run_candidate(Task::Induction, TopologyKind::Conv(3), 16, 24);
    let comparable_attention_params = attention_param_count(INDUCTION_VOCAB, 8, 12);
    assert!(
        result.param_count > comparable_attention_params,
        "the wide conv control must have MORE parameters than a comparable attention candidate: \
         conv={}, attention={comparable_attention_params}",
        result.param_count
    );
    assert!(!did_solve, "the wide conv must still fail: its receptive field, not its parameter count, is the constraint");
}

/// Same seed, same steps, same optimiser, same corpus: rerunning the
/// training+scoring pipeline from scratch must reproduce byte-identical
/// loss curves, parameter counts, and MAC counts -- otherwise "the
/// structurally-right topology won" would be noise, not a result.
#[proxima::test]
async fn rerunning_the_pipeline_reproduces_byte_identical_results() {
    let candidates = [
        (Task::Induction, TopologyKind::Attention, 8usize, 12usize),
        (Task::Bigram, TopologyKind::Conv(2), 6usize, 16usize),
    ];
    for (task, topology, d_model, ffn_hidden) in candidates {
        let (first, first_solved) = run_candidate(task, topology, d_model, ffn_hidden);
        let (second, second_solved) = run_candidate(task, topology, d_model, ffn_hidden);
        assert_eq!(first.loss_curve, second.loss_curve, "task={task:?} topology={topology:?}: loss curve must rerun identically");
        assert_eq!(first.param_count, second.param_count, "param count must rerun identically");
        assert_eq!(first.forward_macs, second.forward_macs, "MAC count must rerun identically");
        assert_eq!(first_solved, second_solved, "the solved verdict must rerun identically");
    }
}
