// a diagnostic binary, not library surface: every `.expect()` below is a
// setup precondition (real checkpoint present, program builds) whose only
// correct response is to panic with the failing step named, matching this
// crate's own sibling examples (`any_listener.rs` and friends carry the
// identical allow for the identical reason).
#![allow(clippy::expect_used)]

//! Within-layer bisection for a DENSE (non-MoE) LFM2 block -- `layer 0` and
//! `layer 1` on the real `LFM2.5-8B-A1B-Q4_K_M.gguf` checkpoint
//! (`leading_dense_block_count = 2`), the two layers
//! `lfm2_moe_route_diff.rs`'s own bisection cannot reach: that tool anchors
//! its backward walk on an `append_moe_ffn` `route` node, which does not
//! exist before layer 2. `lfm2_layer_oracle_diff.rs`'s corrected, RELATIVE
//! per-layer diff found the first out-of-noise-floor divergence already at
//! `l_out-0` (`relative_diff=0.805`, worst position `token=5, dim=126`) --
//! BEFORE any MoE layer runs at all, so the root cause cannot be an MoE
//! routing/gating defect (ROW 132's own suspect). This tool bisects INSIDE
//! layer 0's own conv-mixer + dense-FFN body to find which of `normed`
//! (mixer's own rmsnorm output, feeds the `B`/`C`/`x` branches),
//! `mixer_out` (pre-residual conv-mixer output), `post_mixer` (post-residual,
//! the dense FFN's own residual input), or `normed2` (`ffn_norm` output, the
//! dense FFN's own INPUT) is the first of the four to diverge.
//!
//! Anchored on `blk.{layer}.ffn_gate.weight`'s own [`NodeId`] (unique per
//! layer, always consumed by exactly one `Elementwise::Multiply` --
//! `spec.rs`'s `gate_product = elementwise(.., Multiply, &[(normed2, ..),
//! (w_gate, ..)])`) rather than a MoE route node, this walks BACKWARD through
//! the same rmsnorm-then-residual pattern
//! [`lfm2_moe_route_diff::mixer_pipeline_node_ids`] already established for
//! layer 5, then one layer further back through the conv-mixer's own
//! `out_proj`/gate/`branch_c` chain (`append_lfm2_conv_mixer`, `spec.rs`) to
//! reach `normed`. Every hop is asserted by `reduce_operand`/
//! `elementwise_operand`, so a wrong offset fails the walk instead of
//! silently reading an unrelated node's value -- the same discipline
//! `lfm2_moe_route_diff.rs` uses, duplicated rather than shared for the same
//! reason that file's own doc gives.

use std::env;
use std::fs;
use std::path::PathBuf;

use proxima_gguf::pipe::parse_complete;
use proxima_model_interop::{
    Lfm2Architecture, lfm2_architecture_from_metadata, lfm2_forward_values,
};
use proxima_tensor::op::{NodeId, Op, ReduceInit, ScalarOp};
use proxima_tensor::spec::lfm2_forward_program_with_experts;

fn read_oracle_activation(path: &PathBuf) -> Vec<f32> {
    let bytes = fs::read(path)
        .unwrap_or_else(|error| panic!("read oracle activation at {path:?}: {error}"));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

/// `lfm2_moe_route_diff.rs`'s own `reduce_operand`, duplicated -- see that
/// file's own doc.
fn reduce_operand(
    program: &[Op],
    node: NodeId,
    expected_body: ScalarOp,
    expected_init: ReduceInit,
) -> NodeId {
    match &program[node.0 as usize] {
        Op::Reduce(reduce) if reduce.body == expected_body && reduce.init == expected_init => {
            reduce.operand
        }
        other => panic!(
            "node {node:?}: expected a {expected_body:?}/{expected_init:?} reduce, got {other:?}"
        ),
    }
}

/// `lfm2_moe_route_diff.rs`'s own `elementwise_first_operand`, generalized to
/// an arbitrary operand INDEX -- layer 0's own walk needs `branch_c`
/// (`gated_output`'s operand 1), which the MoE walk never did.
fn elementwise_operand(
    program: &[Op],
    node: NodeId,
    expected_body: ScalarOp,
    index: usize,
) -> NodeId {
    match &program[node.0 as usize] {
        Op::Elementwise { body, operands, .. } if *body == expected_body => operands[index].0,
        other => panic!("node {node:?}: expected a {expected_body:?} elementwise, got {other:?}"),
    }
}

fn elementwise_first_operand(program: &[Op], node: NodeId, expected_body: ScalarOp) -> NodeId {
    elementwise_operand(program, node, expected_body, 0)
}

/// The one [`Op::Input`] named exactly `name` -- `lfm2_moe_route_diff.rs`'s
/// own `input_node_id`, duplicated.
fn input_node_id(program: &[Op], name: &str) -> NodeId {
    for (index, op) in program.iter().enumerate() {
        if let Op::Input {
            name: Some(candidate),
            ..
        } = op
            && candidate == name
        {
            return NodeId(index as u32);
        }
    }
    panic!("no Op::Input named {name:?} in this program");
}

/// The ONLY node in a dense layer's own body that multiplies `normed2`
/// (`ffn_norm`'s output) by `w_gate` -- `spec.rs`'s `gate_product =
/// elementwise(.., Multiply, &[(normed2, "sd->sdg"), (w_gate, "dg->sdg")])`,
/// unique because `w_gate` (`blk.{layer}.ffn_gate.weight`) is consumed
/// exactly once per layer. Scanning the WHOLE program (not just this layer's
/// own node range) is safe here specifically because `w_gate_id` already
/// identifies the layer uniquely -- no other layer's gate_product can name
/// this layer's own `w_gate` as an operand.
fn gate_product_node_id(program: &[Op], w_gate_id: NodeId) -> NodeId {
    for (index, op) in program.iter().enumerate() {
        if let Op::Elementwise {
            body: ScalarOp::Multiply,
            operands,
            ..
        } = op
            && operands.len() == 2
            && operands[1].0 == w_gate_id
        {
            return NodeId(index as u32);
        }
    }
    panic!("no Elementwise(Multiply) node consumes w_gate {w_gate_id:?}");
}

fn max_abs_diff(ours: &[f32], theirs: &[f32]) -> (f32, usize) {
    let mut worst = 0f32;
    let mut worst_index = 0usize;
    for (index, (&mine, &theirs_value)) in ours.iter().zip(theirs).enumerate() {
        let diff = (mine - theirs_value).abs();
        if diff > worst {
            worst = diff;
            worst_index = index;
        }
    }
    (worst, worst_index)
}

fn max_abs(values: &[f32]) -> f32 {
    values.iter().fold(0f32, |acc, value| acc.max(value.abs()))
}

fn report(label: &str, ours: &[f32], theirs: &[f32], embedding: usize) {
    let (diff, worst_index) = max_abs_diff(ours, theirs);
    let oracle_max = max_abs(theirs);
    let relative = if oracle_max > 0.0 {
        diff / oracle_max
    } else {
        f32::INFINITY
    };
    let worst_token = worst_index / embedding;
    let worst_dim = worst_index % embedding;
    println!(
        "{label}: oracle_max_abs={oracle_max:.6e} max_abs_diff={diff:.6e} relative_diff={relative:.6e} worst=(token={worst_token}, dim={worst_dim}) ours={:.6} theirs={:.6}",
        ours[worst_index], theirs[worst_index]
    );
}

fn main() {
    let model_path = env::args().nth(1).unwrap_or_else(|| {
        "/Users/brianbruggeman/.lmstudio/models/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf".to_string()
    });
    let oracle_dir = env::args().nth(2).unwrap_or_else(|| {
        "/private/tmp/claude-501/-Users-brianbruggeman-repos-slot-0/6cd9e134-c1a3-450a-be93-76dd95389bf4/scratchpad/oracle/dump_lfm2".to_string()
    });
    let prompt = env::args()
        .nth(3)
        .unwrap_or_else(|| "The capital of France is".to_string());
    let layer: u32 = env::args()
        .nth(4)
        .unwrap_or_else(|| "0".to_string())
        .parse()
        .expect("layer arg is a u32");

    let model_path = PathBuf::from(&model_path);
    let oracle_dir = PathBuf::from(&oracle_dir);
    if !model_path.exists() {
        println!("skipping: no host-local lfm2 gguf checkpoint at {model_path:?}");
        return;
    }
    if !oracle_dir.exists() {
        println!("skipping: no oracle layer-activation dump directory at {oracle_dir:?}");
        return;
    }

    let file_bytes = fs::read(&model_path).expect("read lfm2 gguf checkpoint");
    let parsed = parse_complete(&file_bytes).expect("parse lfm2 gguf checkpoint");
    let architecture: Lfm2Architecture = lfm2_architecture_from_metadata(&parsed)
        .expect("derive lfm2 architecture from gguf metadata");
    assert!(
        layer < architecture.leading_dense_block_count,
        "layer {layer} is not a dense layer (leading_dense_block_count={})",
        architecture.leading_dense_block_count
    );

    let vocab = proxima_tokenizer::gguf::vocab_from_metadata(&parsed)
        .expect("build vocab from gguf metadata");
    let add_bos = vocab.add_bos_token().unwrap_or(true);
    let ids = proxima_tokenizer::encode_with_bos_eos(&prompt, &vocab, add_bos, false)
        .expect("tokenize prompt");

    let (full_program, _logits_root) = lfm2_forward_program_with_experts(
        architecture.vocab,
        architecture.embedding,
        architecture.feed_forward,
        architecture.expert_feed_forward,
        architecture.query_heads,
        architecture.kv_heads,
        architecture.head_dim,
        architecture.block_count,
        architecture.expert_count,
        architecture.expert_used_count,
        architecture.leading_dense_block_count,
        architecture.l_cache,
        &architecture.layer_kinds,
    )
    .expect("build full lfm2 program to walk this layer's node ids");

    let w_gate_id = input_node_id(&full_program, &format!("blk.{layer}.ffn_gate.weight"));
    let gate_product_id = gate_product_node_id(&full_program, w_gate_id);
    let normed2_id = elementwise_operand(&full_program, gate_product_id, ScalarOp::Multiply, 0);
    let normed_inner_id = elementwise_first_operand(&full_program, normed2_id, ScalarOp::Multiply);
    let post_mixer_id =
        elementwise_first_operand(&full_program, normed_inner_id, ScalarOp::Multiply);
    let mixer_out_id = elementwise_first_operand(&full_program, post_mixer_id, ScalarOp::Add);
    let out_product_id =
        reduce_operand(&full_program, mixer_out_id, ScalarOp::Add, ReduceInit::Zero);
    let gated_output_id =
        elementwise_first_operand(&full_program, out_product_id, ScalarOp::Multiply);
    let branch_c_id = elementwise_operand(&full_program, gated_output_id, ScalarOp::Multiply, 1);
    let branch_c_product_id =
        reduce_operand(&full_program, branch_c_id, ScalarOp::Add, ReduceInit::Zero);
    let normed_id =
        elementwise_first_operand(&full_program, branch_c_product_id, ScalarOp::Multiply);
    // deeper still: `convolved` (post causal-conv, pre-C-gate --
    // `append_lfm2_conv_mixer`'s own `convolved` variable) and
    // `branch_b`/`branch_x` (the B/x streams feeding the conv), walked
    // BACKWARD through `causal_conv1d`'s own tail
    // (`masked_tap = Select([is_valid, tap_product, zero_tap])`,
    // `tap_product = Multiply([windowed, weight])`,
    // `windowed = Identity([(gated_input, gathered_map)])`,
    // `gated_input = Multiply([branch_b, branch_x])` -- `spec.rs`'s own
    // `causal_conv1d`/`append_lfm2_conv_mixer` bodies).
    let convolved_id =
        elementwise_first_operand(&full_program, gated_output_id, ScalarOp::Multiply);
    let masked_tap_id =
        reduce_operand(&full_program, convolved_id, ScalarOp::Add, ReduceInit::Zero);
    let tap_product_id = elementwise_operand(&full_program, masked_tap_id, ScalarOp::Select, 1);
    let windowed_id = elementwise_first_operand(&full_program, tap_product_id, ScalarOp::Multiply);
    let gated_input_id = elementwise_first_operand(&full_program, windowed_id, ScalarOp::Identity);
    let branch_b_id = elementwise_first_operand(&full_program, gated_input_id, ScalarOp::Multiply);
    let branch_x_id = elementwise_operand(&full_program, gated_input_id, ScalarOp::Multiply, 1);

    println!(
        "layer={layer} normed={normed_id:?} branch_b={branch_b_id:?} branch_x={branch_x_id:?} branch_c={branch_c_id:?} convolved={convolved_id:?} mixer_out={mixer_out_id:?} post_mixer={post_mixer_id:?} normed2={normed2_id:?}"
    );

    let all_node_ids = [
        normed_id,
        branch_b_id,
        branch_x_id,
        branch_c_id,
        convolved_id,
        mixer_out_id,
        post_mixer_id,
        normed2_id,
    ];
    let (_logits, values) =
        lfm2_forward_values(&parsed, &file_bytes, &architecture, &ids, &all_node_ids)
            .expect("evaluate this layer's own bisection node values");
    let ours_normed = values[0].as_slice();
    let ours_branch_b = values[1].as_slice();
    let ours_branch_x = values[2].as_slice();
    let ours_branch_c = values[3].as_slice();
    let ours_convolved = values[4].as_slice();
    let ours_mixer_out = values[5].as_slice();
    let ours_post_mixer = values[6].as_slice();
    let ours_normed2 = values[7].as_slice();

    let embedding = architecture.embedding as usize;

    let normed_path = oracle_dir.join(format!("model.layers.{{}}.operator_norm-{layer}.f32"));
    let bcx_path = oracle_dir.join(format!("model.layers.{{}}.conv.in_proj-{layer}.f32"));
    let conv_path = oracle_dir.join(format!("model.layers.{{}}.conv.conv-{layer}.f32"));
    let mixer_out_path = oracle_dir.join(format!("model.layers.{{}}.conv.out_proj-{layer}.f32"));
    let normed2_path = oracle_dir.join(format!("model.layers.{{}}.ffn_out-{layer}.f32"));
    // layer 0's own residual input is `inp_embd` (`model.embed_tokens.f32` --
    // see `lfm2_layer_oracle_diff.rs`'s own updated doc on why the probe's
    // stale `inp_embd` name is dead); every later layer's residual input is
    // the previous layer's own `l_out`.
    let layer_input_path = if layer == 0 {
        oracle_dir.join("model.embed_tokens.f32")
    } else {
        oracle_dir.join(format!("l_out-{}.f32", layer - 1))
    };

    for (label, path) in [
        ("normed", &normed_path),
        ("bcx", &bcx_path),
        ("conv", &conv_path),
        ("mixer_out", &mixer_out_path),
        ("normed2", &normed2_path),
        ("layer_input", &layer_input_path),
    ] {
        if !path.exists() {
            println!("MISSING_ORACLE_FILE for {label} at {path:?}");
            return;
        }
    }

    let their_normed = read_oracle_activation(&normed_path);
    let their_bcx = read_oracle_activation(&bcx_path);
    let their_conv = read_oracle_activation(&conv_path);
    let their_mixer_out = read_oracle_activation(&mixer_out_path);
    let their_normed2 = read_oracle_activation(&normed2_path);
    let their_layer_input = read_oracle_activation(&layer_input_path);
    let their_post_mixer: Vec<f32> = their_mixer_out
        .iter()
        .zip(&their_layer_input)
        .map(|(mixer, input)| mixer + input)
        .collect();

    // `their_bcx` is `[3*embedding]` per token, chunk 0 = B, chunk 1 = C,
    // chunk 2 = ungated x (`build_shortconv_block`'s own `ggml_view_3d`
    // offsets, `models/lfm2.cpp:182-187`) -- slicing it back into three
    // `[embedding]`-per-token vectors is what makes `branch_b`/`branch_c`
    // (post-projection, PRE `B*x` gate) directly comparable to our own.
    let n_tokens = ids.len();
    let mut their_branch_b = vec![0f32; n_tokens * embedding];
    let mut their_branch_c = vec![0f32; n_tokens * embedding];
    let mut their_branch_x = vec![0f32; n_tokens * embedding];
    for token in 0..n_tokens {
        let base = token * 3 * embedding;
        their_branch_b[token * embedding..(token + 1) * embedding]
            .copy_from_slice(&their_bcx[base..base + embedding]);
        their_branch_c[token * embedding..(token + 1) * embedding]
            .copy_from_slice(&their_bcx[base + embedding..base + 2 * embedding]);
        their_branch_x[token * embedding..(token + 1) * embedding]
            .copy_from_slice(&their_bcx[base + 2 * embedding..base + 3 * embedding]);
    }

    println!("\nWITHIN-layer-{layer} bisection (dense FFN, in program order):");
    report(
        "normed      (mixer rmsnorm output, feeds B/C/x)",
        ours_normed,
        &their_normed,
        embedding,
    );
    report(
        "branch_b    (post-projection, PRE B*x gate)     ",
        ours_branch_b,
        &their_branch_b,
        embedding,
    );
    report(
        "branch_x    (post-projection, PRE B*x gate)     ",
        ours_branch_x,
        &their_branch_x,
        embedding,
    );
    report(
        "branch_c    (post-projection, PRE conv gate)    ",
        ours_branch_c,
        &their_branch_c,
        embedding,
    );
    report(
        "convolved   (post causal-conv, PRE C-gate)      ",
        ours_convolved,
        &their_conv,
        embedding,
    );
    report(
        "mixer_out   (pre-residual conv-mixer output)    ",
        ours_mixer_out,
        &their_mixer_out,
        embedding,
    );
    report(
        "post_mixer  (post-residual, dense FFN's input)  ",
        ours_post_mixer,
        &their_post_mixer,
        embedding,
    );
    report(
        "normed2     (ffn_norm output, dense FFN's input)",
        ours_normed2,
        &their_normed2,
        embedding,
    );
}
