//! Layer-5 MoE routing diff for the real, downloaded
//! `LFM2.5-8B-A1B-Q4_K_M.gguf` checkpoint: this crate's own selected-expert
//! indices against `llama.cpp`'s own `build_moe_ffn`'s
//! `cb(selected_experts, "ffn_moe_topk", il)` dump
//! (`llama-graph.cpp:2059`, named `ffn_moe_topk-<il>` by
//! `llama_context::graph_get_cb`'s own `-%d` layer suffix) --
//! `lfm2_layer_oracle_diff.rs`'s bisection named layer 5, dimension 126 as
//! the first gross divergence; this tool answers the PRIMARY question that
//! bisection could not: did the two sides pick the SAME four experts per
//! token at that layer, or did routing itself diverge?
//!
//! [`route_node_ids`] finds [`proxima_tensor::spec::append_moe_ffn`]'s own
//! per-round `route` [`NodeId`]s by scanning the built program for their
//! exact, structurally unique shape (`Op::Reduce` with
//! `dtype: DType::Int32, body: ScalarOp::Maximum, init: ReduceInit::Zero` --
//! the ONLY node in this whole program built with that dtype/body/init
//! triple; `spec.rs`'s own doc on `append_moe_ffn` names this) rather than
//! by threading a new parameter through the public
//! `lfm2_forward_program_with_experts` signature -- a signature every other
//! checkpoint (openchat-3.5, SmolLM2, Mixtral) shares no part of, so adding
//! an output-collection parameter there would touch code paths this bug has
//! nothing to do with. Restricting the scan to layer 5's own node-id range
//! (via [`layer_boundary_node_id`]'s technique, duplicated from
//! `lfm2_layer_oracle_diff.rs` rather than shared, since sharing would need
//! a third crate-internal module neither example otherwise needs) rules out
//! misattributing a route node from an adjacent MoE layer.

use std::env;
use std::fs;
use std::path::PathBuf;

use proxima_gguf::pipe::parse_complete;
use proxima_model_interop::{Lfm2Architecture, lfm2_architecture_from_metadata, lfm2_forward_values};
use proxima_tensor::dtype::DType;
use proxima_tensor::op::{NodeId, Op, ReduceInit, ScalarOp};
use proxima_tensor::spec::lfm2_forward_program_with_experts;

fn read_oracle_route(path: &PathBuf) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read oracle route dump at {path:?}: {error}"));
    bytes.chunks_exact(4).map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])).collect()
}

/// [`lfm2_layer_oracle_diff.rs`]'s own `layer_boundary_node_id`, duplicated
/// rather than shared -- see this file's own doc.
fn layer_boundary_node_id(architecture: &Lfm2Architecture, depth: u32) -> NodeId {
    if depth == 0 {
        return NodeId(2);
    }
    let shallow_kinds = &architecture.layer_kinds[..depth as usize];
    let (shallow, _) = lfm2_forward_program_with_experts(
        architecture.vocab,
        architecture.embedding,
        architecture.feed_forward,
        architecture.expert_feed_forward,
        architecture.query_heads,
        architecture.kv_heads,
        architecture.head_dim,
        depth,
        architecture.expert_count,
        architecture.expert_used_count,
        architecture.leading_dense_block_count,
        architecture.l_cache,
        shallow_kinds,
    )
    .expect("build shallow throwaway lfm2 program");

    let mut deep_kinds = shallow_kinds.to_vec();
    deep_kinds.push(architecture.layer_kinds[(depth - 1) as usize]);
    let (deep, _) = lfm2_forward_program_with_experts(
        architecture.vocab,
        architecture.embedding,
        architecture.feed_forward,
        architecture.expert_feed_forward,
        architecture.query_heads,
        architecture.kv_heads,
        architecture.head_dim,
        depth + 1,
        architecture.expert_count,
        architecture.expert_used_count,
        architecture.leading_dense_block_count,
        architecture.l_cache,
        &deep_kinds,
    )
    .expect("build deep throwaway lfm2 program");

    let first_diff = shallow.iter().zip(deep.iter()).position(|(left, right)| left != right).unwrap_or(shallow.len());
    NodeId((first_diff - 1) as u32)
}

/// Every [`append_moe_ffn`]-shaped `route` [`NodeId`] whose own index falls
/// strictly between `layer_start` and `layer_end` -- one per expert-used
/// round, in program order (round 0 first).
fn route_node_ids(program: &[Op], layer_start: u32, layer_end: u32) -> Vec<NodeId> {
    let mut route_ids = Vec::new();
    for (index, op) in program.iter().enumerate() {
        let id = index as u32;
        if id <= layer_start || id >= layer_end {
            continue;
        }
        if let Op::Reduce(reduce) = op {
            if reduce.dtype == DType::Int32 && reduce.body == ScalarOp::Maximum && reduce.init == ReduceInit::Zero {
                route_ids.push(NodeId(id));
            }
        }
    }
    route_ids
}

/// One `Op::Reduce`'s own `operand`, asserting the `body`/`init` a caller
/// expects at that hop -- a wrong expectation means the backward walk landed
/// on the wrong node, and this fails loudly rather than silently reading an
/// unrelated value.
fn reduce_operand(program: &[Op], node: NodeId, expected_body: ScalarOp, expected_init: ReduceInit) -> NodeId {
    match &program[node.0 as usize] {
        Op::Reduce(reduce) if reduce.body == expected_body && reduce.init == expected_init => reduce.operand,
        other => panic!("node {node:?}: expected a {expected_body:?}/{expected_init:?} reduce, got {other:?}"),
    }
}

/// One `Op::Elementwise`'s own first operand, asserting `body` -- see
/// [`reduce_operand`]'s own doc for why the assertion travels with the walk.
fn elementwise_first_operand(program: &[Op], node: NodeId, expected_body: ScalarOp) -> NodeId {
    match &program[node.0 as usize] {
        Op::Elementwise { body, operands, .. } if *body == expected_body => operands[0].0,
        other => panic!("node {node:?}: expected a {expected_body:?} elementwise, got {other:?}"),
    }
}

/// [`append_moe_ffn`]'s own `logits`/`scores`/`selection_scores` [`NodeId`]s
/// for round 0 at one MoE layer, walked BACKWARD from that round's own
/// `route` node rather than forward from `layer_start` -- `spec.rs`'s own
/// node sequence for `route`: `route = Reduce(Max, candidate)`,
/// `candidate = Elementwise(Multiply, [mask, expert_index])`,
/// `mask = Elementwise(Equal, [selection_scores, max_selection])`,
/// `selection_scores = Elementwise(Add, [scores, bias])` (round 0 only --
/// later rounds' `mask` reads a `Select`-masked `selection_scores` instead,
/// which is why this walk is anchored on `route_ids[0]`, never a later
/// round), `scores = Reciprocal(..Negate(logits)..)` (the sigmoid path),
/// `logits = Reduce(Add, gate_product)`. Every hop is asserted by
/// [`reduce_operand`]/[`elementwise_first_operand`], so a wrong offset fails
/// the walk instead of silently returning an unrelated node's value.
fn gate_pipeline_node_ids(program: &[Op], route_round_0: NodeId) -> [NodeId; 3] {
    let candidate = reduce_operand(program, route_round_0, ScalarOp::Maximum, ReduceInit::Zero);
    let mask = elementwise_first_operand(program, candidate, ScalarOp::Multiply);
    let selection_scores = elementwise_first_operand(program, mask, ScalarOp::Equal);
    let scores = elementwise_first_operand(program, selection_scores, ScalarOp::Add);
    let one_plus_exp = elementwise_first_operand(program, scores, ScalarOp::Reciprocal);
    let exp_neg_logits = elementwise_first_operand(program, one_plus_exp, ScalarOp::Add);
    let neg_logits = elementwise_first_operand(program, exp_neg_logits, ScalarOp::Exponential);
    let logits = elementwise_first_operand(program, neg_logits, ScalarOp::Negate);
    [logits, scores, selection_scores]
}

/// The MoE gate's own `x` input (`normed2` in `spec.rs`'s naming: `rmsnorm`'s
/// output, so `append_moe_ffn`'s own `x` parameter) and the mixer's own
/// pre-residual output (`mixer_out` in `append_lfm2_conv_mixer`'s own
/// naming) -- walked BACKWARD from `logits_id` (`logits = Reduce(Add,
/// gate_product)`, `gate_product = Elementwise(Multiply, [x, gate_inp])` so
/// `x` is `gate_product`'s own first operand) through `rmsnorm`'s own
/// two-multiply tail (`spec.rs:682-693`: `gamma * (x * inv_rms)`, so `x`'s
/// own first operand two hops back is the norm's INPUT) and
/// `append_lfm2_conv_mixer`'s own residual add
/// (`spec.rs:1991`: `mixer_out + x`) -- bisects WITHIN layer 5 whether the
/// divergence is already present before `ffn_norm` (mixer bug) or enters at
/// the norm/gate projection itself.
fn mixer_pipeline_node_ids(program: &[Op], logits_id: NodeId) -> [NodeId; 3] {
    let gate_product = reduce_operand(program, logits_id, ScalarOp::Add, ReduceInit::Zero);
    let normed2 = elementwise_first_operand(program, gate_product, ScalarOp::Multiply);
    let normed_inner = elementwise_first_operand(program, normed2, ScalarOp::Multiply);
    let post_mixer = elementwise_first_operand(program, normed_inner, ScalarOp::Multiply);
    let mixer_out = elementwise_first_operand(program, post_mixer, ScalarOp::Add);
    [normed2, post_mixer, mixer_out]
}

/// The [`NodeId`] of the one `Op::Input` in `program` named exactly `name`
/// -- every per-layer weight `spec.rs` builds carries its own on-disk tensor
/// name (`blk.{layer}.ffn_norm.weight`, etc), so this is a direct,
/// unambiguous lookup, never a structural walk.
fn input_node_id(program: &[Op], name: &str) -> NodeId {
    for (index, op) in program.iter().enumerate() {
        if let Op::Input { name: Some(candidate), .. } = op {
            if candidate == name {
                return NodeId(index as u32);
            }
        }
    }
    panic!("no Op::Input named {name:?} in this program");
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

fn main() {
    let model_path = env::args().nth(1).unwrap_or_else(|| {
        "/Users/brianbruggeman/.lmstudio/models/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf".to_string()
    });
    let oracle_dir = env::args().nth(2).unwrap_or_else(|| {
        "/private/tmp/claude-501/-Users-brianbruggeman-repos-slot-0/6cd9e134-c1a3-450a-be93-76dd95389bf4/scratchpad/oracle/dump_lfm2_routing".to_string()
    });
    let prompt = env::args().nth(3).unwrap_or_else(|| "The capital of France is".to_string());
    let layer: u32 = env::args().nth(4).unwrap_or_else(|| "5".to_string()).parse().expect("layer arg is a u32");

    let model_path = PathBuf::from(&model_path);
    let oracle_dir = PathBuf::from(&oracle_dir);
    if !model_path.exists() {
        println!("skipping: no host-local lfm2 gguf checkpoint at {model_path:?}");
        return;
    }
    let oracle_path = oracle_dir.join(format!("ffn_moe_topk-{layer}.f32"));
    if !oracle_path.exists() {
        println!("skipping: no oracle route dump at {oracle_path:?}");
        return;
    }

    let file_bytes = fs::read(&model_path).expect("read lfm2 gguf checkpoint");
    let parsed = parse_complete(&file_bytes).expect("parse lfm2 gguf checkpoint");
    let architecture: Lfm2Architecture = lfm2_architecture_from_metadata(&parsed).expect("derive lfm2 architecture from gguf metadata");

    let vocab = proxima_tokenizer::gguf::vocab_from_metadata(&parsed).expect("build vocab from gguf metadata");
    let add_bos = vocab.add_bos_token().unwrap_or(true);
    let ids = proxima_tokenizer::encode_with_bos_eos(&prompt, &vocab, add_bos, false).expect("tokenize prompt");

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
    .expect("build full lfm2 program to scan for route nodes");

    let layer_start = layer_boundary_node_id(&architecture, layer);
    let layer_end = layer_boundary_node_id(&architecture, layer + 1);
    let route_ids = route_node_ids(&full_program, layer_start.0, layer_end.0);
    println!(
        "layer={layer} node_range=({}, {}) route_node_count={}",
        layer_start.0,
        layer_end.0,
        route_ids.len()
    );
    assert_eq!(
        route_ids.len(),
        architecture.expert_used_count as usize,
        "expected one route node per expert_used_count round"
    );

    let [logits_id, scores_id, selection_scores_id] = gate_pipeline_node_ids(&full_program, route_ids[0]);
    let [normed2_id, post_mixer_id, mixer_out_id] = mixer_pipeline_node_ids(&full_program, logits_id);
    let ffn_norm_weight_id = input_node_id(&full_program, &format!("blk.{layer}.ffn_norm.weight"));
    let mut all_node_ids = route_ids.clone();
    all_node_ids.extend_from_slice(&[logits_id, scores_id, selection_scores_id, normed2_id, post_mixer_id, mixer_out_id, ffn_norm_weight_id]);

    let (_logits, extras) = lfm2_forward_values(&parsed, &file_bytes, &architecture, &ids, &all_node_ids).expect("evaluate our own route node values");
    let (route_values, rest) = extras.split_at(route_ids.len());
    let (ours_logits, rest) = rest.split_at(1);
    let (ours_scores, rest) = rest.split_at(1);
    let (ours_selection_scores, rest) = rest.split_at(1);
    let (ours_normed2, rest) = rest.split_at(1);
    let (ours_post_mixer, rest) = rest.split_at(1);
    let (ours_mixer_out, rest) = rest.split_at(1);
    let ours_ffn_norm_weight = rest[0].as_slice();
    let ours_mixer_out = ours_mixer_out[0].as_slice();
    let ours_logits = ours_logits[0].as_slice();
    let ours_scores = ours_scores[0].as_slice();
    let ours_selection_scores = ours_selection_scores[0].as_slice();
    let ours_normed2 = ours_normed2[0].as_slice();
    let ours_post_mixer = ours_post_mixer[0].as_slice();

    let theirs = read_oracle_route(&oracle_path);
    let n_expert_used = architecture.expert_used_count as usize;
    assert_eq!(theirs.len(), ids.len() * n_expert_used, "oracle route dump element count mismatch");

    println!("\ntoken round ours_expert theirs_expert match");
    let mut any_mismatch = false;
    for (token, _) in ids.iter().enumerate() {
        for (round, values) in route_values.iter().enumerate() {
            let ours_expert = values[token].round() as i64;
            let theirs_expert = theirs[token * n_expert_used + round].round() as i64;
            let matches = ours_expert == theirs_expert;
            any_mismatch |= !matches;
            println!("{token} {round} {ours_expert} {theirs_expert} {matches}");
        }
    }

    if any_mismatch {
        println!("\nROUTING DIVERGES at layer {layer}: at least one token/round selected a different expert");
    } else {
        println!("\nrouting matches exactly at layer {layer}: divergence enters elsewhere");
    }

    let expert_count = architecture.expert_count as usize;
    let oracle_gate_dir = oracle_dir.parent().unwrap_or(&oracle_dir).join("dump_lfm2_gate");
    let logits_path = oracle_gate_dir.join(format!("ffn_moe_logits-{layer}.f32"));
    let probs_path = oracle_gate_dir.join(format!("ffn_moe_probs-{layer}.f32"));
    let probs_biased_path = oracle_gate_dir.join(format!("ffn_moe_probs_biased-{layer}.f32"));
    if logits_path.exists() && probs_path.exists() && probs_biased_path.exists() {
        let their_logits = read_oracle_route(&logits_path);
        let their_scores = read_oracle_route(&probs_path);
        let their_selection_scores = read_oracle_route(&probs_biased_path);

        println!("\ntoken expert ours_logit theirs_logit ours_score theirs_score ours_selection theirs_selection");
        for token in 0..ids.len() {
            let mut worst_logit_diff = 0f32;
            let mut worst_expert = 0usize;
            for expert in 0..expert_count {
                let index = token * expert_count + expert;
                let diff = (ours_logits[index] - their_logits[index]).abs();
                if diff > worst_logit_diff {
                    worst_logit_diff = diff;
                    worst_expert = expert;
                }
            }
            let index = token * expert_count + worst_expert;
            println!(
                "{token} {worst_expert} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
                ours_logits[index],
                their_logits[index],
                ours_scores[index],
                their_scores[index],
                ours_selection_scores[index],
                their_selection_scores[index]
            );
        }
    } else {
        println!("\nskipping gate-pipeline comparison: no oracle logits/probs dump at {oracle_gate_dir:?}");
    }

    let oracle_intra_dir = oracle_dir.parent().unwrap_or(&oracle_dir).join("dump_lfm2_intra");
    // `model.layers.{}.ffn_out-<il>` is llama.cpp's own naming quirk, not
    // ours -- `lfm2.cpp:270-275`'s two consecutive `cb()` calls both rename
    // the SAME `ffn_norm_out` tensor, so the value dumped under the
    // misleading `ffn_out` label is actually the norm's own output, exactly
    // what `normed2`/`append_moe_ffn`'s `x` parameter holds on our side.
    let mixer_out_path = oracle_intra_dir.join(format!("model.layers.{{}}.conv.out_proj-{layer}.f32"));
    let normed2_path = oracle_intra_dir.join(format!("model.layers.{{}}.ffn_out-{layer}.f32"));
    // `post_mixer = mixer_out + (layer `layer`'s own residual input)`
    // (`spec.rs:1991`) -- for `layer > 0` that residual input is exactly
    // `l_out-{layer-1}` (`lfm2_layer_oracle_diff.rs`'s own convention), so
    // `their_post_mixer` is computed EXACTLY here, never approximated by
    // reusing our own `post_mixer` as a stand-in for theirs.
    let layer_input_path = if layer == 0 { oracle_dir.join("inp_embd.f32") } else { oracle_dir.join(format!("l_out-{}.f32", layer - 1)) };
    if mixer_out_path.exists() && normed2_path.exists() && layer_input_path.exists() {
        let their_mixer_out = read_oracle_route(&mixer_out_path);
        let their_normed2 = read_oracle_route(&normed2_path);
        let their_layer_input = read_oracle_route(&layer_input_path);
        let their_post_mixer: Vec<f32> = their_mixer_out.iter().zip(&their_layer_input).map(|(mix, input)| mix + input).collect();

        let (mixer_out_diff, mixer_out_worst) = max_abs_diff(ours_mixer_out, &their_mixer_out);
        let (post_mixer_diff, post_mixer_worst) = max_abs_diff(ours_post_mixer, &their_post_mixer);
        let (normed2_diff, normed2_worst) = max_abs_diff(ours_normed2, &their_normed2);
        println!("\nWITHIN-layer-{layer} bisection (pre-gate):");
        println!(
            "mixer_out (pre-residual, {} elements) max_abs_diff={mixer_out_diff:e} worst_index={mixer_out_worst} ours={:.6} theirs={:.6}",
            ours_mixer_out.len(),
            ours_mixer_out[mixer_out_worst],
            their_mixer_out[mixer_out_worst]
        );
        println!(
            "post_mixer (residual sum, {} elements) max_abs_diff={post_mixer_diff:e} worst_index={post_mixer_worst} ours={:.6} theirs={:.6}",
            ours_post_mixer.len(),
            ours_post_mixer[post_mixer_worst],
            their_post_mixer[post_mixer_worst]
        );
        println!(
            "normed2 (post-ffn_norm, pre-gate, {} elements) max_abs_diff={normed2_diff:e} worst_index={normed2_worst} ours={:.6} theirs={:.6}",
            ours_normed2.len(),
            ours_normed2[normed2_worst],
            their_normed2[normed2_worst]
        );

        // `normed2[token,dim] = post_mixer[token,dim] * inv_rms[token] *
        // gamma[dim]` -- `inv_rms` is a single scalar per token, so the
        // RATIO `normed2 / post_mixer` at the worst dimension, divided by
        // that SAME ratio at a nearby reference dimension (to cancel
        // `inv_rms`), isolates `gamma[worst_dim] / gamma[reference_dim]`
        // alone -- computed from EACH side's own exact `post_mixer`, never
        // approximated. Comparing the two isolated ratios tells us directly
        // whether `gamma` itself differs between implementations at this
        // dimension.
        let embedding = architecture.embedding as usize;
        let worst_token = normed2_worst / embedding;
        let worst_dim = normed2_worst % embedding;
        let reference_dim = if worst_dim == 0 { 1 } else { 0 };
        let ours_index = worst_token * embedding + reference_dim;
        let our_ratio_worst = ours_normed2[normed2_worst] / ours_post_mixer[normed2_worst];
        let our_ratio_reference = ours_normed2[ours_index] / ours_post_mixer[ours_index];
        let their_ratio_worst = their_normed2[normed2_worst] / their_post_mixer[normed2_worst];
        let their_ratio_reference = their_normed2[ours_index] / their_post_mixer[ours_index];
        println!(
            "worst_dim={worst_dim} token={worst_token}: our_gamma_ratio={:.6} their_gamma_ratio={:.6} (bound gamma[{worst_dim}]={:.6}, bound gamma[{reference_dim}]={:.6})",
            our_ratio_worst / our_ratio_reference,
            their_ratio_worst / their_ratio_reference,
            ours_ffn_norm_weight[worst_dim],
            ours_ffn_norm_weight[reference_dim]
        );

        // `post_mixer = mixer_out + layer_input` (`spec.rs:1991`) exactly --
        // subtracting `ours_mixer_out` back out of `ours_post_mixer` recovers
        // OUR OWN value for `layer_input` (this layer's own residual INPUT,
        // `l_out-{layer-1}` on the oracle side) with no extra evaluation,
        // so this isolates whether the divergence at `(worst_token,
        // worst_dim)` was ALREADY present going INTO layer `layer`, or was
        // introduced by this layer's own mixer.
        let ours_layer_input_worst = ours_post_mixer[normed2_worst] - ours_mixer_out[normed2_worst];
        let their_layer_input_worst = their_layer_input[normed2_worst];
        let mixer_out_worst_at_dim = (ours_mixer_out[normed2_worst] - their_mixer_out[normed2_worst]).abs();
        println!(
            "at (token={worst_token}, dim={worst_dim}): layer_input(=l_out-{}) ours={:.6} theirs={:.6} diff={:.6} | this layer's own mixer_out diff={:.6}",
            layer.wrapping_sub(1),
            ours_layer_input_worst,
            their_layer_input_worst,
            (ours_layer_input_worst - their_layer_input_worst).abs(),
            mixer_out_worst_at_dim
        );
    } else {
        println!("\nskipping within-layer bisection: no oracle mixer_out/normed2/layer-input dump at {oracle_intra_dir:?} / {layer_input_path:?}");
    }
}
