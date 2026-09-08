//! Proves the program-composition surface `proxima_tensor::spec` exports is
//! actually sufficient for a foreign architecture crate composing a forward
//! program from OUTSIDE this crate -- the gap three prior landings hit one
//! symbol at a time (the `append_*` builders, then `elementwise`/`reduce`/
//! `symbolic_leaf`/`scalar_constant`/`rmsnorm`/`causal_mask`/`sigmoid`, all
//! `pub(crate)` or private until this pass). The in-crate
//! `public_builders_compose_a_one_layer_forward_program` test
//! (`src/spec.rs`) only ever proved in-crate composability -- it lives inside
//! `mod tests`, with sibling access to every private item regardless of its
//! declared visibility. An integration test under `tests/` compiles against
//! the crate's public API only, so it is the one artifact that can actually
//! settle whether the surface is public.
//!
//! Composes a tiny one-layer forward program using nothing but
//! `proxima_tensor::spec::*` (and the crate-root re-exports every builder's
//! signature already names): an entry self-attention block, broadcast into
//! hyper-connection streams via `elementwise`, a sigmoid-gated shared-expert
//! FFN alongside a routed `append_moe_ffn`, and a final `rmsnorm` +
//! `elementwise` + `reduce` lm-head chain -- the three shapes a foreign crate
//! could not previously express with any private symbol in the mix.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::too_many_lines)]

use proxima_tensor::spec::{
    ExpertGatingFunc, append_moe_ffn, causal_mask, elementwise, embedding_lookup, input_leaf,
    reduce, rmsnorm, scalar_constant, sigmoid, symbolic_leaf,
};
use proxima_tensor::{DType, Extent, ReduceInit, ScalarOp};

#[test]
fn external_crate_composes_a_one_layer_forward_program() {
    let tokens = 2usize;
    let vocab = 3u32;
    let embedding = 2u32;
    let hc = 2u32;
    let expert_count = 2u32;
    let expert_used_count = 1u32;
    let expert_hidden = 2u32;
    let shared_hidden = 2u32;

    let mut program = Vec::new();

    let ids = input_leaf(&mut program, DType::Int32, vec![Extent::Symbolic(0)], "ids");
    let table = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(vocab), Extent::Static(embedding)],
        "token_embd.weight",
    );
    let embedded = embedding_lookup(&mut program, table, ids);

    let inv_dim = scalar_constant(&mut program, 1.0 / embedding as f32);
    let eps = symbolic_leaf(&mut program, DType::Float32, "eps");
    let one = scalar_constant(&mut program, 1.0);

    // A tiny single-head self-attention: q/k/v from the embedding directly
    // (no learned projection weights -- the composition under test is the
    // masking/softmax/context chain, not a separate matmul this crate
    // already proves elsewhere), masked with the SAME `causal_mask` every
    // in-crate attention builder uses.
    let (is_future, neg_infinity) = causal_mask(&mut program).expect("causal_mask lowers");

    let scores_product = elementwise(
        &mut program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(embedded, "sd->std"), (embedded, "td->std")],
    )
    .expect("qk product lowers");
    let scores = reduce(
        &mut program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        scores_product,
        "std->std",
        "st->std",
    )
    .expect("qk reduce lowers");
    let scores_masked = elementwise(
        &mut program,
        DType::Float32,
        ScalarOp::Select,
        &[
            (is_future, "st->st"),
            (neg_infinity, "->st"),
            (scores, "st->st"),
        ],
    )
    .expect("causal select lowers");
    let score_max = reduce(
        &mut program,
        DType::Float32,
        ScalarOp::Maximum,
        ReduceInit::NegativeInfinity,
        scores_masked,
        "st->st",
        "s->st",
    )
    .expect("score max lowers");
    let shifted = elementwise(
        &mut program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(scores_masked, "st->st"), (score_max, "s->st")],
    )
    .expect("shift lowers");
    let weights = elementwise(&mut program, DType::Float32, ScalarOp::Exponential, &[(shifted, "st->st")])
        .expect("exponential lowers");
    let weight_sum = reduce(
        &mut program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        weights,
        "st->st",
        "s->st",
    )
    .expect("weight sum lowers");
    let inv_weight_sum = elementwise(&mut program, DType::Float32, ScalarOp::Reciprocal, &[(weight_sum, "s->s")])
        .expect("reciprocal lowers");
    let probabilities = elementwise(
        &mut program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(weights, "st->st"), (inv_weight_sum, "s->st")],
    )
    .expect("probability normalization lowers");
    let context_product = elementwise(
        &mut program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(probabilities, "st->std"), (embedded, "td->std")],
    )
    .expect("context product lowers");
    let context = reduce(
        &mut program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        context_product,
        "std->std",
        "sd->std",
    )
    .expect("context reduce lowers");

    // Entry broadcast into hyper-connection streams via `elementwise`, then
    // collapse them back with `reduce` -- the exact shape the foreign crate
    // could not spell with `elementwise` private.
    let hc_ones = proxima_tensor::append(
        &mut program,
        proxima_tensor::Op::Constant {
            dtype: DType::Float32,
            shape: vec![Extent::Static(hc)],
            value: 1.0,
        },
    );
    let streamed = elementwise(
        &mut program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(context, "sd->shd"), (hc_ones, "h->shd")],
    )
    .expect("hyper-connection stream broadcast lowers");
    let hidden = reduce(
        &mut program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        streamed,
        "shd->shd",
        "sd->shd",
    )
    .expect("hyper-connection stream collapse lowers");

    // Routed experts via the same `append_moe_ffn` every real forward
    // program calls per layer.
    let gate_inp = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(embedding), Extent::Static(expert_count)],
        "gate_inp",
    );
    let expert_w_gate = input_leaf(
        &mut program,
        DType::Float32,
        vec![
            Extent::Static(expert_count),
            Extent::Static(embedding),
            Extent::Static(expert_hidden),
        ],
        "expert_w_gate",
    );
    let expert_w_up = input_leaf(
        &mut program,
        DType::Float32,
        vec![
            Extent::Static(expert_count),
            Extent::Static(embedding),
            Extent::Static(expert_hidden),
        ],
        "expert_w_up",
    );
    let expert_w_down = input_leaf(
        &mut program,
        DType::Float32,
        vec![
            Extent::Static(expert_count),
            Extent::Static(expert_hidden),
            Extent::Static(embedding),
        ],
        "expert_w_down",
    );
    let routed_out = append_moe_ffn(
        &mut program,
        hidden,
        gate_inp,
        expert_w_gate,
        expert_w_up,
        expert_w_down,
        expert_count,
        expert_used_count,
        one,
        ExpertGatingFunc::Softmax,
        None,
    )
    .expect("routed moe ffn lowers");

    // A sigmoid-gated shared-expert FFN, run alongside the routed one and
    // added into its output -- the shape a private `sigmoid` could not
    // express outside this crate.
    let shared_w_up = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(embedding), Extent::Static(shared_hidden)],
        "shared_w_up",
    );
    let shared_w_down = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(shared_hidden), Extent::Static(embedding)],
        "shared_w_down",
    );
    let shared_gate_weight = input_leaf(&mut program, DType::Float32, vec![Extent::Static(embedding)], "shared_gate_weight");

    let shared_up_product = elementwise(
        &mut program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(hidden, "sd->sdf"), (shared_w_up, "df->sdf")],
    )
    .expect("shared ffn up product lowers");
    let shared_hidden_node = reduce(
        &mut program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        shared_up_product,
        "sdf->sdf",
        "sf->sdf",
    )
    .expect("shared ffn up reduce lowers");
    let shared_down_product = elementwise(
        &mut program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(shared_hidden_node, "sf->sfd"), (shared_w_down, "fd->sfd")],
    )
    .expect("shared ffn down product lowers");
    let shared_out = reduce(
        &mut program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        shared_down_product,
        "sfd->sfd",
        "sd->sfd",
    )
    .expect("shared ffn down reduce lowers");

    let gate_logit_product = elementwise(
        &mut program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(hidden, "sd->sd"), (shared_gate_weight, "d->sd")],
    )
    .expect("shared gate logit product lowers");
    let gate_logit = reduce(
        &mut program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        gate_logit_product,
        "sd->sd",
        "s->sd",
    )
    .expect("shared gate logit reduce lowers");
    let shared_gate = sigmoid(&mut program, gate_logit, one, "s->s").expect("sigmoid gate lowers");
    let shared_scaled = elementwise(
        &mut program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(shared_out, "sd->sd"), (shared_gate, "s->sd")],
    )
    .expect("shared ffn gating lowers");

    let ffn_out = elementwise(
        &mut program,
        DType::Float32,
        ScalarOp::Add,
        &[(routed_out, "sd->sd"), (shared_scaled, "sd->sd")],
    )
    .expect("routed + shared combination lowers");

    // The same `rmsnorm` + `elementwise` + `reduce` lm-head chain every
    // forward-program builder in this crate ends with.
    let output_norm_weight = input_leaf(&mut program, DType::Float32, vec![Extent::Static(embedding)], "output_norm.weight");
    let normed_final = rmsnorm(&mut program, ffn_out, output_norm_weight, inv_dim, eps).expect("final rmsnorm lowers");
    let lm_head_weight = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(embedding), Extent::Static(vocab)],
        "output.weight",
    );
    let logits_product = elementwise(
        &mut program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed_final, "sd->sdv"), (lm_head_weight, "dv->sdv")],
    )
    .expect("lm_head product lowers");
    let logits = reduce(
        &mut program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        logits_product,
        "sdv->sdv",
        "sv->sdv",
    )
    .expect("lm_head reduce lowers");

    let mut state = 0x2026_0907_dead_beefu64;
    let mut filled = |len: usize| -> Vec<f32> {
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
            })
            .collect()
    };

    let ids_data: Vec<f32> = (0..tokens as i32).map(|token| (token % vocab as i32) as f32).collect();
    let eps_data = vec![1e-6f32; tokens];
    let table_data = filled(vocab as usize * embedding as usize);
    let gate_inp_data = filled(embedding as usize * expert_count as usize);
    let expert_w_gate_data = filled(expert_count as usize * embedding as usize * expert_hidden as usize);
    let expert_w_up_data = filled(expert_count as usize * embedding as usize * expert_hidden as usize);
    let expert_w_down_data = filled(expert_count as usize * expert_hidden as usize * embedding as usize);
    let shared_w_up_data = filled(embedding as usize * shared_hidden as usize);
    let shared_w_down_data = filled(shared_hidden as usize * embedding as usize);
    let shared_gate_weight_data = filled(embedding as usize);
    let output_norm_data = filled(embedding as usize);
    let lm_head_data = filled(embedding as usize * vocab as usize);

    let named: Vec<(&str, &[f32])> = vec![
        ("ids", ids_data.as_slice()),
        ("eps", eps_data.as_slice()),
        ("token_embd.weight", table_data.as_slice()),
        ("gate_inp", gate_inp_data.as_slice()),
        ("expert_w_gate", expert_w_gate_data.as_slice()),
        ("expert_w_up", expert_w_up_data.as_slice()),
        ("expert_w_down", expert_w_down_data.as_slice()),
        ("shared_w_up", shared_w_up_data.as_slice()),
        ("shared_w_down", shared_w_down_data.as_slice()),
        ("shared_gate_weight", shared_gate_weight_data.as_slice()),
        ("output_norm.weight", output_norm_data.as_slice()),
        ("output.weight", lm_head_data.as_slice()),
    ];

    let evaluated = proxima_tensor::cpu::evaluate_named(&program, &[tokens as u64], &named, &[logits])
        .expect("the externally composed program evaluates on cpu");
    let (logits_values, logits_shape) = evaluated.get(logits).expect("logits output present");

    assert_eq!(
        logits_shape,
        [tokens as u64, vocab as u64],
        "logits must be [tokens, vocab]"
    );
    assert!(
        logits_values.iter().all(|value| value.is_finite()),
        "every logit must be finite: {logits_values:?}"
    );
}
