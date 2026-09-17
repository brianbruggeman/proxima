//! Real-weight A/B parity probe (task: settle proxima-load-bug vs
//! checkpoint-layout for gemma4's garbage output). A = the ENGINE's own
//! computed layer-0 embedding, via `LoadedModel::forward_node_values`
//! against the real blob. B = an INDEPENDENT reference that dequantizes
//! the SAME real `token_embd.weight` (Q6_K) bytes directly with
//! `proxima_gguf::quant::q6_k::dequantize` and applies the same
//! `sqrt(embedding)` scale gemma4's own bind path
//! (`Gemma4Arch::bind`, `EmbeddingScale::Sqrt`) declares. Not library
//! surface -- a one-shot diagnostic, same convention as
//! `smollm2_layer_oracle_diff.rs`.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::env;
use std::fs::File;

use proxima_gguf::parse_complete;
use proxima_gguf::types::GgmlType;
use proxima_model_interop::LoadedModel;
use proxima_model_interop::gemma4::from_metadata;
use proxima_tensor::op::{NodeId, Op};
use proxima_tensor::spec::{
    Activation, AttentionScoreScale, EmbeddingScale, ExpertGatingFunc, FfnCombination,
    LayerAttentionConfig, LayerFfnConfig, LayerKind, RopePairing, RopeTableSel, ValueSourceKind,
    lfm2_forward_program_with_experts,
};

/// Finds the [`NodeId`] of the `Op::Input` leaf named `name` -- `op::append`'s
/// id-is-index invariant means the leaf's position in `program` IS its
/// `NodeId`, so this is a linear scan, not a lookup table.
fn find_input(program: &[Op], name: &str) -> NodeId {
    program
        .iter()
        .position(|op| matches!(op, Op::Input { name: Some(found), .. } if found == name))
        .map(|index| NodeId(index as u32))
        .unwrap_or_else(|| panic!("no Op::Input leaf named {name:?} in program"))
}

/// Reproduces `gemma4_attention_configs`/`gemma4_ffn_configs`
/// (`proxima-model-interop/src/gemma4/bind.rs`, both private to that crate)
/// so this diagnostic can call `lfm2_forward_program_with_experts` directly
/// -- pure graph construction, no weight bytes touched, so it is
/// near-instant next to the real `Gemma4Arch::bind`'s full weight bind
/// (proven too slow for this checkpoint's size in this same session: a
/// second independent `bind_gemma4_weights` call alone exceeded a 280s
/// budget). This reproduces the exact same `(Vec<Op>, NodeId, MoeSites)`
/// `LoadedModel::load` built internally, so its `NodeId`s are the same ones
/// `LoadedModel::forward_node_values` evaluates against the real blob.
fn gemma4_program(architecture: &proxima_model_interop::gemma4::Architecture) -> (Vec<Op>, NodeId) {
    let attention_configs: Vec<LayerAttentionConfig> = architecture
        .sliding_window_pattern
        .iter()
        .enumerate()
        .map(|(layer, &is_sliding)| {
            let kv_heads = architecture.kv_heads_by_layer[layer];
            if is_sliding {
                LayerAttentionConfig {
                    head_dim: architecture.key_length_swa,
                    kv_heads,
                    mask_window: Some(architecture.sliding_window),
                    value_source_kind: ValueSourceKind::ProjectedV,
                    rope_table: RopeTableSel {
                        cos_name: "rope_cos_swa",
                        sin_name: "rope_sin_swa",
                    },
                    rope_pairing: RopePairing::SplitHalf {
                        pairs: architecture.key_length_swa / 2,
                    },
                    score_scale: AttentionScoreScale::Unscaled,
                    value_norm: true,
                }
            } else {
                LayerAttentionConfig {
                    head_dim: architecture.key_length,
                    kv_heads,
                    mask_window: None,
                    value_source_kind: ValueSourceKind::SharedWithKey,
                    rope_table: RopeTableSel {
                        cos_name: "rope_cos",
                        sin_name: "rope_sin",
                    },
                    rope_pairing: RopePairing::SplitHalf {
                        pairs: architecture.key_length / 2,
                    },
                    score_scale: AttentionScoreScale::Unscaled,
                    value_norm: true,
                }
            }
        })
        .collect();
    let layer_kinds = vec![LayerKind::Attention; architecture.block_count as usize];
    let ffn_configs = vec![
        LayerFfnConfig {
            post_attention_norm: true,
            combination: FfnCombination::ParallelDenseMoe,
            dense_post_norm: true,
            routed_post_norm: true,
            combined_post_norm: true,
            output_scale: true,
            routed_gating: ExpertGatingFunc::Softmax,
            routed_expert_bias: false,
            routed_pre_norm: true,
            router_scale: true,
            expert_output_scale: true,
            activation: Activation::GeluTanh,
        };
        architecture.block_count as usize
    ];
    let logit_softcap = (architecture.final_logit_softcapping > 0.0)
        .then_some(architecture.final_logit_softcapping);
    let (program, logits, _moe_sites) = lfm2_forward_program_with_experts(
        architecture.vocab,
        architecture.embedding,
        architecture.feed_forward,
        architecture.expert_feed_forward,
        architecture.head_count,
        architecture.block_count,
        architecture.expert_count,
        architecture.expert_used_count,
        0,
        0,
        &layer_kinds,
        &attention_configs,
        &ffn_configs,
        Some(EmbeddingScale::Sqrt),
        logit_softcap,
        true,
    )
    .expect("build gemma4 program (mirrors Gemma4Arch::bind)");
    (program, logits)
}

/// Dequantizes ONE row (or the whole 1-D tensor, for a rank-1 leaf) of a
/// named real tensor directly from the mapped file bytes -- no weight bind,
/// so this stays cheap even though `bind_gemma4_weights` itself proved too
/// slow to call twice in this session's budget.
fn dequant_tensor(
    parsed: &proxima_gguf::ParsedGguf,
    file_bytes: &[u8],
    name: &str,
) -> (Vec<f32>, Vec<u64>) {
    let tensor = parsed
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .unwrap_or_else(|| panic!("tensor {name:?} present in real checkpoint"));
    let range = parsed
        .tensor_data_range(tensor, file_bytes.len() as u64)
        .unwrap_or_else(|error| panic!("tensor_data_range for {name:?}: {error:?}"));
    let source = &file_bytes[range.start as usize..range.end as usize];
    let element_count: u64 = tensor.dims.iter().product();
    let mut out = vec![0f32; element_count as usize];
    if tensor.ggml_type == GgmlType::F32 {
        // already raw f32 on disk -- no quant codec to dispatch, just a
        // byte reinterpret, the same convention `gguf_tensor_as_f32`
        // (`crate::bind`) uses for this exact tensor family.
        for (chunk, value) in source.chunks_exact(4).zip(out.iter_mut()) {
            *value = f32::from_le_bytes(chunk.try_into().expect("4-byte f32 chunk"));
        }
    } else {
        proxima_gguf::quant::dispatch::dequantize(tensor.ggml_type, source, &mut out)
            .unwrap_or_else(|error| {
                panic!("dequantize {name:?} ({:?}): {error:?}", tensor.ggml_type)
            });
    }
    (out, tensor.dims.to_vec())
}

/// Structural cross-check for the hand-counted mixer offsets: does `op`
/// read `target` as one of its operands, directly.
fn op_references(op: &Op, target: NodeId) -> bool {
    match op {
        Op::Elementwise { operands, .. } => operands.iter().any(|(node, _)| *node == target),
        Op::Reduce(reduce) => reduce.operand == target,
        _ => false,
    }
}

/// The `NodeId` of `op`'s operand at `index` -- used to structurally recover
/// a value that has no name-anchored leaf of its own (e.g. an attention
/// mixer's `x` residual argument), by reading it directly off the mixer's own
/// first-appended op instead of re-deriving it by counting.
fn operand_of(op: &Op, index: usize) -> NodeId {
    match op {
        Op::Elementwise { operands, .. } => operands[index].0,
        _ => panic!("operand_of: op is not Elementwise, has no indexed operands"),
    }
}

/// Layer-N FFN region NodeIds, all derived by walking a FIXED, known offset
/// from a name-anchored leaf -- zero hand-counting through the attention
/// mixer (whose own op count varies per layer: `ProjectedV` sliding layers
/// declare an extra `attn_v.weight` leaf and value-projection ops,
/// `SharedWithKey` full-attention layers do not). Every offset here is
/// justified by the FFN block's own straight-line source
/// (`attention_forward.rs:1319-1466`):
/// - `post_mixer` is the attention mixer's OWN last-appended op, and
///   `append_dense_swiglu_ffn`'s first action is declaring the
///   `ffn_gate.weight` leaf right after `rmsnorm(post_mixer, ffn_norm, ..)`'s
///   fixed 8-op run -- so `post_mixer = ffn_gate_leaf - 9`.
/// - `dense_raw`/`routed_raw` are each branch's own last op, immediately
///   followed by that branch's post-norm gamma leaf -- `dense_raw =
///   post_ffw_norm_1_leaf - 1`, `routed_raw = post_ffw_norm_2_leaf - 1`.
/// - `rmsnorm` is always exactly 8 ops, so each post-norm output sits at
///   `gamma_leaf + 8`.
/// - `router_logits` is `append_routed_expert_ffn`'s own straight-line
///   sequence off `ffn_gate_inp.weight`: `+1` router_scale leaf, `+2` scaled
///   router input, `+3` expert_scale leaf, `+4/+5/+6` expert gate/up/down
///   leaves, `+7` gate_product, `+8` router_logits reduce.
/// - the final layer output `x` is the residual add immediately followed by
///   the `layer_output_scale.weight` leaf, then the scale multiply --
///   `x = layer_output_scale_leaf + 1`.
struct LayerFfnTaps {
    post_mixer: NodeId,
    dense_raw: NodeId,
    dense_out: NodeId,
    router_logits: NodeId,
    routed_raw: NodeId,
    routed_out: NodeId,
    combined: NodeId,
    x: NodeId,
}

fn ffn_taps(program: &[Op], layer: u32) -> LayerFfnTaps {
    let ffn_gate_id = find_input(program, &format!("blk.{layer}.ffn_gate.weight"));
    let post_mixer = NodeId(ffn_gate_id.0 - 9);
    let post_ffw_norm_1_id = find_input(program, &format!("blk.{layer}.post_ffw_norm_1.weight"));
    let dense_raw = NodeId(post_ffw_norm_1_id.0 - 1);
    let dense_out = NodeId(post_ffw_norm_1_id.0 + 8);
    let gate_inp_id = find_input(program, &format!("blk.{layer}.ffn_gate_inp.weight"));
    // PROVEN WRONG at `+8` (this session, PRONG 3): that offset lands
    // mid-`rmsnorm` (a per-position SCALAR, shape [s] -- confirmed by a
    // runtime panic slicing it as [s, expert_count] and by the engine print
    // "router_logits |max| first8" showing only 6 values, one per
    // position, not 6*128). `append_routed_expert_ffn`'s ACTUAL straight-
    // line sequence off `gate_inp_id` (gemma4's config: `router_scale=true`,
    // `expert_output_scale=true`, `use_expert_bias=false`,
    // `attention_forward.rs:883-1003`): +1 router_scale_weight leaf, +2
    // inv_sqrt_embedding scalar, +3 rooted_router_scale_weight elementwise,
    // +4..+11 rmsnorm (a fixed 8-op run -- same convention every other
    // `+8` in this file documents), +12 expert_scale leaf, +13/+14/+15
    // expert gate/up/down leaves, +16 gate_product, +17 router_logits
    // reduce.
    let router_logits = NodeId(gate_inp_id.0 + 17);
    let post_ffw_norm_2_id = find_input(program, &format!("blk.{layer}.post_ffw_norm_2.weight"));
    let routed_raw = NodeId(post_ffw_norm_2_id.0 - 1);
    let routed_out = NodeId(post_ffw_norm_2_id.0 + 8);
    let post_ffw_norm_id = find_input(program, &format!("blk.{layer}.post_ffw_norm.weight"));
    let combined = NodeId(post_ffw_norm_id.0 + 8);
    let output_scale_id = find_input(program, &format!("blk.{layer}.layer_output_scale.weight"));
    let x = NodeId(output_scale_id.0 + 1);
    LayerFfnTaps {
        post_mixer,
        dense_raw,
        dense_out,
        router_logits,
        routed_raw,
        routed_out,
        combined,
        x,
    }
}

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| {
        "/Users/brianbruggeman/.ollama/models/blobs/sha256-ea549b7688d4c95019754880c21e3f29c58c985a7a1c3b37b9eebd0a95224129"
            .to_string()
    });
    let prompt = env::args()
        .nth(2)
        .unwrap_or_else(|| "The capital of France is the".to_string());

    let file = File::open(&path).expect("open gemma4 gguf");
    let mapping = unsafe { memmap2::Mmap::map(&file) }.expect("mmap gemma4 gguf");
    let file_bytes: &[u8] = &mapping;
    let parsed = parse_complete(file_bytes).expect("parse gemma4 header");
    let architecture = from_metadata(&parsed).expect("gemma4 hparams");
    let embedding = architecture.embedding as usize;

    println!("== stage: tokenize (matches LoadedModel::forward_node_values's own path) ==");
    let vocab =
        proxima_tokenizer::gguf::vocab_from_metadata(&parsed).expect("vocab from gguf metadata");
    let wants_bos = vocab
        .add_bos_token()
        .unwrap_or_else(|| vocab.bos_token_id().is_some());
    let add_eos = vocab.add_eos_token().unwrap_or(false);
    let ids = proxima_tokenizer::encode_with_bos_eos(&prompt, &vocab, wants_bos, add_eos)
        .expect("tokenize prompt");
    println!("prompt={prompt:?} wants_bos={wants_bos} add_eos={add_eos} ids={ids:?}");

    println!(
        "== deriving layer-0 attention-mixer NodeIds from the SAME program builder Gemma4Arch::bind calls =="
    );
    // `gemma4_program` calls `lfm2_forward_program_with_experts` with the
    // same args `Gemma4Arch::bind` (`gemma4/bind.rs:550`) does -- pure graph
    // construction, no weight bytes touched, so this reproduces the exact
    // `NodeId` numbering `LoadedModel::load` built internally without paying
    // for a second full weight bind (proven >280s and still incomplete for
    // this checkpoint in this same session -- see the dropped stage C).
    let (b_program, _b_logits) = gemma4_program(&architecture);
    let attn_norm_id = find_input(&b_program, "blk.0.attn_norm.weight");
    let post_attn_norm_id = find_input(&b_program, "blk.0.post_attention_norm.weight");
    let wq_id = find_input(&b_program, "blk.0.attn_q.weight");
    let wk_id = find_input(&b_program, "blk.0.attn_k.weight");
    let wv_id = find_input(&b_program, "blk.0.attn_v.weight");
    let wo_id = find_input(&b_program, "blk.0.attn_output.weight");
    let qn_id = find_input(&b_program, "blk.0.attn_q_norm.weight");
    let kn_id = find_input(&b_program, "blk.0.attn_k_norm.weight");
    // `append_attention_mixer` (`proxima-tensor/src/spec/attention_forward.rs:373`)
    // is a straight-line sequence for gemma4's layer 0 (ProjectedV, no
    // branches) -- these offsets are hand-counted node-for-node off that
    // function's own source, `rmsnorm`/`rmsnorm_per_head` each being a fixed
    // 8-op run (`primitives.rs:858`,`935`), `rmsnorm_per_head_no_scale` a
    // fixed 7-op run (`primitives.rs:1013`, no learned-gamma final multiply).
    // `base` is the first node the mixer itself appends, right after the
    // last leaf (`k_norm_weight`) the caller declares before invoking it
    // (`attention_forward.rs:1229-1257`).
    //
    // RE-DERIVED this pass: the PRIOR offsets below `v_out=base+29` were
    // hand-counted BEFORE `value_norm` (Gemma 4's weightless per-kv-head V
    // RMSNorm, `attention_forward.rs:492-500`) existed on this ProjectedV
    // path -- every op from `v` onward sits 7 slots later than the stale
    // count assumed. Full straight-line count for THIS layer's shape
    // (`ProjectedV` + `value_norm: true` + `SplitHalf{pairs: 128}`):
    // rmsnorm(x)=[+0..+7]->normed=+7; q_product=+8; q_raw(reduce)=+9;
    // rmsnorm_per_head(q)=[+10..+17]->q_normed=+17; k_product=+18;
    // k_raw(reduce)=+19; rmsnorm_per_head(k)=[+20..+27]->k_normed=+27;
    // v_product=+28; v_raw(reduce)=+29;
    // rmsnorm_per_head_no_scale(v_raw)=[+30..+36]->v_normed=+36 (7 ops, NO
    // final gamma multiply); rope Q=[+37..+42] (q_even_cos,q_odd_sin,
    // rotated_q_even=+39,q_even_sin,q_odd_cos,rotated_q_odd=+42); rope
    // K=[+43..+48] (k_even_cos,k_odd_sin,rotated_k_even=+45,k_even_sin,
    // k_odd_cos,rotated_k_odd=+48); GQA grouping+scores=[+49..+55]
    // (q_even_grouped,q_odd_grouped,score_even_product,score_even,
    // score_odd_product,score_odd,scores=+55); scores_scaled=+56;
    // scores_masked=+57; softmax=[+58..+63]
    // (score_max,shifted,weights,weight_sum,inv_weight_sum,
    // probabilities=+63); attended_product=+64; attended(reduce)=+65;
    // wo_product=+66; attn_out_raw(reduce)=+67;
    // rmsnorm(attn_out,post_attention_norm)=[+68..+75]->attn_out_normed=+75;
    // residual add->post_mixer=+76.
    let base = kn_id.0 + 1;
    let normed0 = NodeId(base + 7); // post attn_norm, pre q/k/v projection
    let q_raw = NodeId(base + 9); // q_product reduce: q BEFORE qk-norm
    let q_normed = NodeId(base + 17); // rmsnorm_per_head(q_raw, q_norm): q AFTER qk-norm, BEFORE rope
    let k_raw = NodeId(base + 19); // k_product reduce: k BEFORE qk-norm
    let k_normed = NodeId(base + 27); // rmsnorm_per_head(k_raw, k_norm): k AFTER qk-norm, BEFORE rope
    let v_raw0 = NodeId(base + 29); // v_product reduce: v projection output, BEFORE value_norm
    let v_normed0 = NodeId(base + 36); // rmsnorm_per_head_no_scale(v_raw0): V AFTER value_norm -- what attention actually consumes
    let rotated_q_even0 = NodeId(base + 39); // roped Q, head-dim channels [0, 128)
    let rotated_q_odd0 = NodeId(base + 42); // roped Q, head-dim channels [128, 256)
    let rotated_k_even0 = NodeId(base + 45); // roped K, head-dim channels [0, 128)
    let rotated_k_odd0 = NodeId(base + 48); // roped K, head-dim channels [128, 256)
    let scores_raw = NodeId(base + 55); // pre-scale rope'd q.k scores
    let scores_scaled = NodeId(base + 56); // post inv_sqrt_head_dim scale
    let probabilities = NodeId(base + 63); // softmax output
    let attended = NodeId(base + 65); // weighted-V sum, BEFORE o_proj
    let attn_out_raw = NodeId(base + 67); // o_proj output, BEFORE post_attention_norm
    let attn_out_normed0 = NodeId(base + 75); // post post_attention_norm, BEFORE residual add
    let post_mixer = NodeId(base + 76); // residual-added attention-block output (input to ffn_norm)
    println!(
        "wq={wq_id:?} wk={wk_id:?} wv={wv_id:?} wo={wo_id:?} qn={qn_id:?} kn={kn_id:?} base={base}"
    );
    println!(
        "taps: normed0={normed0:?} q_raw={q_raw:?} q_normed={q_normed:?} k_raw={k_raw:?} k_normed={k_normed:?} v_raw0={v_raw0:?} v_normed0={v_normed0:?} rotated_q_even0={rotated_q_even0:?} rotated_q_odd0={rotated_q_odd0:?} rotated_k_even0={rotated_k_even0:?} rotated_k_odd0={rotated_k_odd0:?} scores_raw={scores_raw:?} scores_scaled={scores_scaled:?} probabilities={probabilities:?} attended={attended:?} attn_out_raw={attn_out_raw:?} attn_out_normed0={attn_out_normed0:?} post_mixer={post_mixer:?}"
    );

    // Structural validation, same discipline layer 5 below uses: confirm the
    // RE-DERIVED offsets land on ops whose ACTUAL operands are the nodes the
    // source says they should be -- not trusting arithmetic alone.
    let normed_op = &b_program[(base + 7) as usize];
    let q_product_op = &b_program[(base + 8) as usize];
    let k_squared_op = &b_program[(base + 20) as usize];
    let v_squared_op = &b_program[(base + 30) as usize];
    let k_even_cos_op = &b_program[(base + 43) as usize];
    let post_attn_norm_final_op = &b_program[(base + 75) as usize];
    let x0 = operand_of(&b_program[base as usize], 0);
    let post_mixer_op = &b_program[post_mixer.0 as usize];
    println!(
        "validate normed(base+7) references attn_norm_weight={}: {}",
        attn_norm_id.0,
        op_references(normed_op, attn_norm_id)
    );
    println!(
        "validate q_product(base+8) references normed0={}: {}",
        normed0.0,
        op_references(q_product_op, normed0)
    );
    println!(
        "validate k rmsnorm_per_head(base+20) references k_raw={}: {}",
        k_raw.0,
        op_references(k_squared_op, k_raw)
    );
    println!(
        "validate v rmsnorm_per_head_no_scale(base+30) references v_raw0={}: {}",
        v_raw0.0,
        op_references(v_squared_op, v_raw0)
    );
    println!(
        "validate rope k_even_cos(base+43) references k_normed={}: {}",
        k_normed.0,
        op_references(k_even_cos_op, k_normed)
    );
    println!(
        "validate post_attn_norm final(base+75) references post_attention_norm.weight={}: {}",
        post_attn_norm_id.0,
        op_references(post_attn_norm_final_op, post_attn_norm_id)
    );
    println!(
        "validate post_mixer(base+76) references x0={} AND attn_out_normed0={}: {} {}",
        x0.0,
        attn_out_normed0.0,
        op_references(post_mixer_op, x0),
        op_references(post_mixer_op, attn_out_normed0)
    );

    println!(
        "\n== deriving layer-5 (FULL/global attention) mixer NodeIds -- diagnostic for v_norm/score-scaling/partial-rope =="
    );
    // Layer 5 is FULL attention: `ValueSourceKind::SharedWithKey` (no
    // `attn_v.weight` leaf, no V-projection ops) + `value_norm: true`
    // (`rmsnorm_per_head_no_scale`, 7 ops, no learned gamma). Hand-counted
    // node-for-node off `append_attention_mixer`'s straight-line source
    // (`attention_forward.rs:391-731`), same convention `base`/`q_raw`/etc.
    // above uses for layer 0's `ProjectedV` shape -- this is the DIFFERENT
    // straight-line shape `SharedWithKey` + `value_norm` produces.
    let attn_norm_id5 = find_input(&b_program, "blk.5.attn_norm.weight");
    let wq_id5 = find_input(&b_program, "blk.5.attn_q.weight");
    let wk_id5 = find_input(&b_program, "blk.5.attn_k.weight");
    let has_wv5 = b_program.iter().any(
        |op| matches!(op, Op::Input { name: Some(found), .. } if found == "blk.5.attn_v.weight"),
    );
    let qn_id5 = find_input(&b_program, "blk.5.attn_q_norm.weight");
    let kn_id5 = find_input(&b_program, "blk.5.attn_k_norm.weight");
    let base5 = kn_id5.0 + 1;
    let normed5 = NodeId(base5 + 7); // post attn_norm, pre q/k/v projection
    let q_raw5 = NodeId(base5 + 9); // q_product reduce: q BEFORE qk-norm
    let q_normed5 = NodeId(base5 + 17); // rmsnorm_per_head(q_raw5, q_norm): q AFTER qk-norm, BEFORE rope
    let k_raw5 = NodeId(base5 + 19); // k_product reduce: k BEFORE qk-norm
    let k_normed5 = NodeId(base5 + 27); // rmsnorm_per_head(k_raw5, k_norm): k AFTER qk-norm, BEFORE rope
    let v5 = NodeId(base5 + 34); // rmsnorm_per_head_no_scale(k_raw5): V post value_norm (SharedWithKey source: v_raw IS k_raw5)
    let rotated_q_even5 = NodeId(base5 + 37); // roped Q, head-dim channels [0, pairs)
    let rotated_q_odd5 = NodeId(base5 + 40); // roped Q, head-dim channels [pairs, 2*pairs)
    let rotated_k_even5 = NodeId(base5 + 43); // roped K, head-dim channels [0, pairs)
    let rotated_k_odd5 = NodeId(base5 + 46); // roped K, head-dim channels [pairs, 2*pairs)
    let scores_raw5 = NodeId(base5 + 53); // pre-scale q.k scores, feeds attention_forward.rs:627
    let scores_scaled5 = NodeId(base5 + 54); // post inv_sqrt_head_dim scale
    let probabilities5 = NodeId(base5 + 61); // causal softmax output
    let attended5 = NodeId(base5 + 63); // weighted-V sum, BEFORE o_proj
    let attn_out_raw5 = NodeId(base5 + 65); // o_proj output, BEFORE post_attention_norm
    let attn_out_normed5 = NodeId(base5 + 73); // post post_attention_norm, BEFORE residual add
    let post_mixer5 = NodeId(base5 + 74); // residual-added attention-block output
    println!(
        "layer5: attn_norm={attn_norm_id5:?} wq={wq_id5:?} wk={wk_id5:?} has_attn_v_weight_leaf={has_wv5} qn={qn_id5:?} kn={kn_id5:?} base5={base5}"
    );
    println!(
        "layer5 taps: normed={normed5:?} q_raw={q_raw5:?} q_normed={q_normed5:?} k_raw={k_raw5:?} k_normed={k_normed5:?} v={v5:?} rotated_q_even={rotated_q_even5:?} rotated_q_odd={rotated_q_odd5:?} rotated_k_even={rotated_k_even5:?} rotated_k_odd={rotated_k_odd5:?} scores_raw={scores_raw5:?} scores_scaled={scores_scaled5:?} probabilities={probabilities5:?} attended={attended5:?} attn_out_raw={attn_out_raw5:?} attn_out_normed={attn_out_normed5:?} post_mixer={post_mixer5:?}"
    );
    // structural validation, same discipline as the layer-0 checks above:
    // confirm the hand-counted offsets land on ops whose ACTUAL operands are
    // the nodes the source says they should be -- not trusting arithmetic
    // alone. `k_raw5`'s FIRST consumer inside `rmsnorm_per_head` is the
    // `squared = k_raw5 * k_raw5` elementwise at `base5+20`; `v5`'s FIRST
    // consumer inside `rmsnorm_per_head_no_scale` is the identical-shaped
    // `squared = v_raw * v_raw` at `base5+28`, and `v_raw` under
    // `SharedWithKey` IS `k_raw5` (no separate V-projection node exists) --
    // so this ALSO proves the `SharedWithKey` wiring at the graph level, not
    // just by architecture-config reading. `q_product`'s FIRST operand
    // (base5+8) is `normed5` under `append_attention_mixer`'s own
    // `q_product = elementwise(normed, wq, ...)` line; `post_mixer5`'s own
    // residual add is `elementwise(attn_out_normed5, x5)` where `x5` is the
    // mixer's own `x` argument, itself the FIRST operand of the mixer's own
    // opening op (base5+0, `squared = x*x` inside the leading `rmsnorm`).
    let q_product_op = &b_program[(base5 + 8) as usize];
    let k_squared_op = &b_program[(base5 + 20) as usize];
    let v_squared_op = &b_program[(base5 + 28) as usize];
    let k_even_cos_op = &b_program[(base5 + 41) as usize];
    let x5 = operand_of(&b_program[base5 as usize], 0);
    let post_mixer5_op = &b_program[post_mixer5.0 as usize];
    println!(
        "validate q_product(base5+8) references normed5={}: {}",
        normed5.0,
        op_references(q_product_op, normed5)
    );
    println!(
        "validate k rmsnorm_per_head(base5+20) references k_raw5={}: {}",
        k_raw5.0,
        op_references(k_squared_op, k_raw5)
    );
    println!(
        "validate v rmsnorm_per_head_no_scale(base5+28) references k_raw5 (SharedWithKey: v_raw==k_raw)={}: {}",
        k_raw5.0,
        op_references(v_squared_op, k_raw5)
    );
    println!(
        "validate rope k_even_cos(base5+41) references k_normed5={}: {}",
        k_normed5.0,
        op_references(k_even_cos_op, k_normed5)
    );
    println!(
        "validate post_mixer5(base5+74) references x5={} AND attn_out_normed5={}: {} {}",
        x5.0,
        attn_out_normed5.0,
        op_references(post_mixer5_op, x5),
        op_references(post_mixer5_op, attn_out_normed5)
    );

    println!("\n== stage A: engine forward, node 4 = post-sqrt-scale embedding ==");
    // traced from `proxima_tensor::spec::attention_forward::lfm2_forward_program_with_experts`:
    // ids=NodeId(0), token_embd.weight=NodeId(1), embedding_lookup=NodeId(2),
    // sqrt(embedding) scalar_constant=NodeId(3), elementwise multiply=NodeId(4).
    // gemma4 binds `EmbeddingScale::Sqrt` (`gemma4/bind.rs`), so NodeId(4) is
    // the scaled embedding every downstream layer actually consumes.
    println!(
        "\n== PRONG 1 setup: layer-by-layer FFN/MoE region NodeIds, name-anchored (no hand-counting through the mixer) =="
    );
    let probe_layers: [u32; 5] = [0, 1, 5, 15, 29];
    let layer_taps: Vec<LayerFfnTaps> = probe_layers
        .iter()
        .map(|&layer| ffn_taps(&b_program, layer))
        .collect();
    for (&layer, taps) in probe_layers.iter().zip(layer_taps.iter()) {
        println!(
            "  layer {layer}: post_mixer={:?} dense_raw={:?} dense_out={:?} router_logits={:?} routed_raw={:?} routed_out={:?} combined={:?} x={:?}",
            taps.post_mixer,
            taps.dense_raw,
            taps.dense_out,
            taps.router_logits,
            taps.routed_raw,
            taps.routed_out,
            taps.combined,
            taps.x
        );
    }
    let output_norm_id = find_input(&b_program, "output_norm.weight");
    // `rmsnorm` is a fixed 8-op run consuming (input, gamma_leaf) -- the
    // gamma leaf is declared immediately after its raw input is computed
    // (same convention `ffn_taps` documents for `dense_raw`/`routed_raw`),
    // so `x_final = output_norm_id - 1` is the last layer's residual output,
    // BEFORE output_norm -- the exact node the task calls "x_final".
    let x_final_id = NodeId(output_norm_id.0 - 1);
    let normed_final = NodeId(output_norm_id.0 + 8);
    let lm_head_id = find_input(&b_program, "output.weight");
    let logits_pre_softcap = NodeId(lm_head_id.0 + 2);
    let logits_post_softcap = NodeId(lm_head_id.0 + 6);
    println!(
        "final taps: x_final={x_final_id:?} normed_final={normed_final:?} logits_pre_softcap={logits_pre_softcap:?} logits_post_softcap={logits_post_softcap:?} logit_softcap={:?}",
        architecture.final_logit_softcapping
    );
    // structural validation: x_final must be the SAME node as layer 29's own
    // `x` (residual-add output after `layer_output_scale`) -- not just close
    // by arithmetic, actually identical, since layer 29 is the last layer in
    // `probe_layers` before `output_norm.weight` is declared.
    let layer29_taps = ffn_taps(&b_program, 29);
    println!(
        "validate x_final == layer29.x: x_final={x_final_id:?} layer29.x={:?} equal={}",
        layer29_taps.x,
        x_final_id == layer29_taps.x
    );

    let model = LoadedModel::load(&parsed, file_bytes).expect("load real gemma4 checkpoint");
    let mut requested = vec![
        NodeId(4),
        normed0,
        q_raw,
        q_normed,
        k_raw,
        k_normed,
        v_raw0,
        v_normed0,
        rotated_q_even0,
        rotated_q_odd0,
        rotated_k_even0,
        rotated_k_odd0,
        scores_raw,
        scores_scaled,
        probabilities,
        attended,
        attn_out_raw,
        attn_out_normed0,
        post_mixer,
        x0,
        normed5,
        q_raw5,
        q_normed5,
        k_raw5,
        k_normed5,
        v5,
        rotated_q_even5,
        rotated_q_odd5,
        rotated_k_even5,
        rotated_k_odd5,
        scores_raw5,
        scores_scaled5,
        probabilities5,
        attended5,
        attn_out_raw5,
        attn_out_normed5,
        post_mixer5,
        x5,
    ];
    for taps in &layer_taps {
        requested.extend([
            taps.post_mixer,
            taps.dense_raw,
            taps.dense_out,
            taps.router_logits,
            taps.routed_raw,
            taps.routed_out,
            taps.combined,
            taps.x,
        ]);
    }
    requested.push(x_final_id);
    requested.push(normed_final);
    requested.push(logits_pre_softcap);
    requested.push(logits_post_softcap);
    // layer-0 dense FFN A-vs-B needs the engine's own `normed2` (post
    // `ffn_norm`, pre-gate/up projection) as the reference matmul input --
    // `normed2 = find_input(ffn_gate) - 1`, the last op
    // `append_dense_swiglu_ffn` consumes before declaring its own first leaf.
    let normed2_layer0 = NodeId(find_input(&b_program, "blk.0.ffn_gate.weight").0 - 1);
    requested.push(normed2_layer0);
    // layer-0 routed-expert MoE A-vs-B needs the engine's own expert INPUT
    // `routed_input` -- `rmsnorm(post_mixer, pre_ffw_norm_2.weight, ..)`,
    // the value `append_routed_expert_ffn` actually receives as `normed`
    // (`attention_forward.rs:1449-1476`; NOT `normed2`, which is the
    // dense-branch's own `ffn_norm`-normed input). `rmsnorm` is a fixed
    // 8-op run, same convention `ffn_taps` uses for every other post-norm
    // output: `routed_input = pre_ffw_norm_2_leaf + 8`.
    let pre_ffw_norm_2_id = find_input(&b_program, "blk.0.pre_ffw_norm_2.weight");
    let routed_input_layer0 = NodeId(pre_ffw_norm_2_id.0 + 8);
    requested.push(routed_input_layer0);
    // sliding-window rope table (`rope_cos_swa`/`rope_sin_swa`) -- pushed
    // BEFORE the full-layer `rope_cos`/`rope_sin` pair below so the existing
    // `values.len() - 2` / `values.len() - 1` full-table lookups stay valid.
    let rope_cos_swa_id = find_input(&b_program, "rope_cos_swa");
    let rope_sin_swa_id = find_input(&b_program, "rope_sin_swa");
    requested.push(rope_cos_swa_id);
    requested.push(rope_sin_swa_id);
    let rope_cos_id = find_input(&b_program, "rope_cos");
    let rope_sin_id = find_input(&b_program, "rope_sin");
    requested.push(rope_cos_id);
    requested.push(rope_sin_id);
    let values = model
        .forward_node_values(&prompt, &requested)
        .expect("engine forward for embedding + layer-0 attention taps + PRONG 1/2 FFN/MoE taps");
    let engine_rope_cos_table = values[values.len() - 2].clone();
    let engine_rope_sin_table = values[values.len() - 1].clone();
    let engine_rope_cos_swa_table = values[values.len() - 4].clone();
    let engine_rope_sin_swa_table = values[values.len() - 3].clone();
    let rope_pairs_dump = 256usize;
    let rope_pairs_swa_dump = 128usize;
    let last_position = ids.len() - 1;
    println!(
        "DUMP rope_cos/sin at last position, pair 60..68: cos={:?} sin={:?}",
        &engine_rope_cos_table
            [last_position * rope_pairs_dump + 60..last_position * rope_pairs_dump + 68],
        &engine_rope_sin_table
            [last_position * rope_pairs_dump + 60..last_position * rope_pairs_dump + 68]
    );
    println!(
        "DUMP rope_cos_swa/sin_swa at last position, pair 0..8: cos={:?} sin={:?}",
        &engine_rope_cos_swa_table
            [last_position * rope_pairs_swa_dump..last_position * rope_pairs_swa_dump + 8],
        &engine_rope_sin_swa_table
            [last_position * rope_pairs_swa_dump..last_position * rope_pairs_swa_dump + 8]
    );
    let engine_embedding = &values[0];
    println!(
        "engine_embedding.len()={} positions={} embedding={}",
        engine_embedding.len(),
        engine_embedding.len() / embedding,
        embedding
    );

    println!("== stage B: independent Q6_K dequant of the same real token_embd.weight rows ==");
    let tensor = parsed
        .tensors
        .iter()
        .find(|tensor| tensor.name == "token_embd.weight")
        .expect("token_embd.weight tensor present");
    let layout = tensor.ggml_type.block_layout();
    assert_eq!(
        tensor.ggml_type,
        proxima_gguf::types::GgmlType::Q6_K,
        "token_embd.weight must be Q6_K on the real checkpoint"
    );
    let bytes_per_row = (embedding as u64 / layout.block_elements) * layout.block_bytes;
    let range = parsed
        .tensor_data_range(tensor, file_bytes.len() as u64)
        .expect("tensor data range");
    let source = &file_bytes[range.start as usize..range.end as usize];
    let scale = (embedding as f32).sqrt();

    let mut max_abs_diff = 0f32;
    let mut worst_position = 0usize;
    let mut worst_index = 0usize;
    let mut reference_embedding = vec![0f32; ids.len() * embedding];
    for (position, &token_id) in ids.iter().enumerate() {
        let row_start = (token_id as u64 * bytes_per_row) as usize;
        let row_bytes = &source[row_start..row_start + bytes_per_row as usize];
        let mut row = vec![0f32; embedding];
        proxima_gguf::quant::q6_k::dequantize(row_bytes, &mut row).expect("dequant token_embd row");
        for (local, value) in row.iter().enumerate() {
            let scaled = value * scale;
            reference_embedding[position * embedding + local] = scaled;
            let diff = (scaled - engine_embedding[position * embedding + local]).abs();
            if diff > max_abs_diff {
                max_abs_diff = diff;
                worst_position = position;
                worst_index = local;
            }
        }
    }

    println!(
        "A-vs-B embedding(after sqrt scale): max_abs_diff={max_abs_diff:e} worst_position={worst_position} worst_index={worst_index}"
    );
    let worst_flat = worst_position * embedding + worst_index;
    println!(
        "  engine={:.6} reference={:.6}",
        engine_embedding[worst_flat], reference_embedding[worst_flat]
    );
    println!(
        "  engine[0..8]={:?}",
        &engine_embedding[position_zero_slice(embedding)]
    );
    println!(
        "  reference[0..8]={:?}",
        &reference_embedding[position_zero_slice(embedding)]
    );

    // Stage C (fused ffn_gate_up_exps split, engine bind vs. independent byte
    // split) was already proven byte-exact in the prior pass on this same
    // checkpoint -- NOT re-run here: a second independent
    // `bind_gemma4_weights` full-weight bind alone exceeded a 280s budget in
    // THIS session (see git history / prior log), so re-proving it here would
    // burn the whole 30-minute ceiling on an already-settled stage. Priority
    // 1 (attention q/k/v) is what the task asked this pass to reach.

    let engine_normed0 = &values[1];
    let engine_q_raw = &values[2];
    let engine_q_normed = &values[3];
    let engine_k_raw = &values[4];
    let engine_k_normed = &values[5];
    let engine_v_raw0 = &values[6];
    let engine_v_normed0 = &values[7];
    let engine_rotated_q_even0 = &values[8];
    let engine_rotated_q_odd0 = &values[9];
    let engine_rotated_k_even0 = &values[10];
    let engine_rotated_k_odd0 = &values[11];
    let engine_scores_raw = &values[12];
    let engine_scores_scaled = &values[13];
    let engine_probabilities = &values[14];
    let engine_attended = &values[15];
    let engine_attn_out_raw = &values[16];
    let engine_attn_out_normed0 = &values[17];
    let engine_post_mixer = &values[18];
    let engine_x0 = &values[19];

    println!(
        "\n== stage: PRIORITY 1 -- independent Q3_K/Q5_K dequant of attn_q/attn_k/attn_v, matmul against the proven-correct embedding (node 4) =="
    );
    let (wq_bytes, wq_dims) = dequant_tensor(&parsed, file_bytes, "blk.0.attn_q.weight");
    let (wk_bytes, wk_dims) = dequant_tensor(&parsed, file_bytes, "blk.0.attn_k.weight");
    let (wv_bytes, wv_dims) = dequant_tensor(&parsed, file_bytes, "blk.0.attn_v.weight");
    let wq_type = tensor_type(&parsed, "blk.0.attn_q.weight");
    let wk_type = tensor_type(&parsed, "blk.0.attn_k.weight");
    let wv_type = tensor_type(&parsed, "blk.0.attn_v.weight");
    println!("attn_q.weight: dims={wq_dims:?} ggml_type={wq_type:?}");
    println!("attn_k.weight: dims={wk_dims:?} ggml_type={wk_type:?}");
    println!("attn_v.weight: dims={wv_dims:?} ggml_type={wv_type:?}");

    // GGUF convention (proven by the embedding stage above, byte-exact):
    // `dims[0]` is the fastest-varying (contraction) axis; every other dim
    // is a "row" of `dims[0]` contiguous elements at
    // `row_index * dims[0]`. `attn_norm_weight` is applied to `x` BEFORE the
    // q/k/v projection (`append_attention_mixer`'s own first line,
    // `attention_forward.rs:396`) -- the proven-correct embedding node 4 is
    // the RAW residual stream, not what q/k/v actually contract against.
    // Since this probe's job is isolating whether the q/k/v PROJECTION
    // (decode+transpose+matmul) is the bug, not the norm, it contracts
    // against the engine's OWN `normed0` tap (`NodeId(base+7)`, already
    // batched into the ONE `forward_node_values` call above) -- so any
    // divergence here is provably the projection, not a re-derived norm.
    let out_features_q = (wq_dims.iter().product::<u64>() / embedding as u64) as usize;
    let out_features_k = (wk_dims.iter().product::<u64>() / embedding as u64) as usize;
    let out_features_v = (wv_dims.iter().product::<u64>() / embedding as u64) as usize;
    let reference_q = matmul_rows(
        engine_normed0,
        &wq_bytes,
        ids.len(),
        embedding,
        out_features_q,
    );
    let reference_k = matmul_rows(
        engine_normed0,
        &wk_bytes,
        ids.len(),
        embedding,
        out_features_k,
    );
    let reference_v_raw0 = matmul_rows(
        engine_normed0,
        &wv_bytes,
        ids.len(),
        embedding,
        out_features_v,
    );

    report_diff("STAGE1 layer0 q_raw (pre qk-norm)", engine_q_raw, &reference_q);
    report_diff("STAGE1 layer0 k_raw (pre qk-norm)", engine_k_raw, &reference_k);
    report_diff(
        "STAGE1 layer0 v_raw0 (pre value_norm)",
        engine_v_raw0,
        &reference_v_raw0,
    );

    let kv_heads0 = architecture.kv_heads_by_layer[0] as usize;
    let head_dim0 = architecture.key_length_swa as usize;
    let query_heads0 = architecture.head_count as usize;
    let group0 = query_heads0 / kv_heads0;
    let rope_base0 = architecture.rope_freq_base_swa;
    let rope_dim0 = architecture.key_length_swa;
    let eps0 = architecture.rms_epsilon;
    let window0 = architecture.sliding_window;
    println!(
        "layer0 shapes: kv_heads={kv_heads0} head_dim={head_dim0} query_heads={query_heads0} group={group0} rope_base_swa={rope_base0} rope_dim_swa={rope_dim0} window={window0} positions={}",
        ids.len()
    );

    // STAGE 2: q_normed/k_normed -- independent rmsnorm_per_head with the
    // RAW (no +1) q_norm/k_norm gamma, over head_dim=256, same convention
    // layer 5's STAGE 2 uses.
    let (qn0_gamma, qn0_dims) = dequant_tensor(&parsed, file_bytes, "blk.0.attn_q_norm.weight");
    let (kn0_gamma, kn0_dims) = dequant_tensor(&parsed, file_bytes, "blk.0.attn_k_norm.weight");
    println!("blk.0.attn_q_norm.weight: dims={qn0_dims:?} blk.0.attn_k_norm.weight: dims={kn0_dims:?}");
    let reference_q_normed0 = rmsnorm_per_head_raw_gamma_ref(
        engine_q_raw,
        &qn0_gamma,
        ids.len(),
        query_heads0,
        head_dim0,
        eps0,
    );
    let reference_k_normed0 = rmsnorm_per_head_raw_gamma_ref(
        engine_k_raw,
        &kn0_gamma,
        ids.len(),
        kv_heads0,
        head_dim0,
        eps0,
    );
    report_diff(
        "STAGE2 layer0 q_normed vs rmsnorm_per_head(RAW q_norm gamma, head_dim=256)",
        engine_q_normed,
        &reference_q_normed0,
    );
    report_diff(
        "STAGE2 layer0 k_normed vs rmsnorm_per_head(RAW k_norm gamma, head_dim=256)",
        engine_k_normed,
        &reference_k_normed0,
    );

    // STAGE 3: v_normed0 -- weightless per-kv-head RMSNorm of the RAW
    // PROJECTED v (`engine_v_raw0`, NOT k_raw -- this layer is ProjectedV,
    // not SharedWithKey).
    let reference_v_normed0 =
        rmsnorm_per_head_no_scale_ref(engine_v_raw0, ids.len(), kv_heads0, head_dim0, eps0);
    report_diff(
        "STAGE3 layer0 v_normed0 (post value_norm) vs weightless-rmsnorm-per-head(v_raw0)",
        engine_v_normed0,
        &reference_v_normed0,
    );

    // STAGE 4 -- THE KEY SLIDING CHECK: independently rotate the engine's OWN
    // q_normed/k_normed with the sliding rope formula
    // (`gemma4_sliding_rope_table`, `gemma4/program.rs:27-44`, and its ONE
    // call site, `Gemma4Arch::step_inputs`, `gemma4/bind.rs:637`):
    // `theta = position * 1e4^(-2*pair/256)` for ALL 128 pairs (full
    // rotation, base 1e4, dim 256) -- NOT layer 5's base-1e6/full-512-dim
    // table and NOT a partial rotation.
    let engine_roped_q0 = reconstruct_roped(
        engine_rotated_q_even0,
        engine_rotated_q_odd0,
        ids.len(),
        query_heads0,
        head_dim0,
    );
    let engine_roped_k0 = reconstruct_roped(
        engine_rotated_k_even0,
        engine_rotated_k_odd0,
        ids.len(),
        kv_heads0,
        head_dim0,
    );
    let pairs_full0 = head_dim0 / 2;
    let reference_roped_q0 = rope_split_half_ref(
        engine_q_normed,
        ids.len(),
        query_heads0,
        head_dim0,
        pairs_full0,
        rope_base0,
        rope_dim0 as usize,
    );
    let reference_roped_k0 = rope_split_half_ref(
        engine_k_normed,
        ids.len(),
        kv_heads0,
        head_dim0,
        pairs_full0,
        rope_base0,
        rope_dim0 as usize,
    );
    report_diff_ranged(
        "STAGE4 layer0 roped Q vs SLIDING rope (base=1e4, dim=256, pairs=128, engine's own formula)",
        &engine_roped_q0,
        &reference_roped_q0,
        head_dim0,
    );
    report_diff_ranged(
        "STAGE4 layer0 roped K vs SLIDING rope (base=1e4, dim=256, pairs=128, engine's own formula)",
        &engine_roped_k0,
        &reference_roped_k0,
        head_dim0,
    );

    // STAGE 5: scores (pre/post-scale). Independent GQA contraction over the
    // engine's OWN roped Q/K, unscaled (gemma4 score_scale=Unscaled for every
    // layer, `gemma4_program` above). For 6-token prompts the 1024 window is
    // inert -- plain causal softmax below covers the window-mask check too.
    let reference_scores_raw0 = gqa_scores_ref(
        &engine_roped_q0,
        &engine_roped_k0,
        ids.len(),
        query_heads0,
        kv_heads0,
        head_dim0,
        group0,
    );
    report_diff(
        "STAGE5 layer0 scores (pre-scale) vs GQA contraction(engine roped Q, engine roped K)",
        engine_scores_raw,
        &reference_scores_raw0,
    );
    report_diff(
        "STAGE5 layer0 scores (post-scale, scale=1.0) vs SAME unscaled GQA contraction",
        engine_scores_scaled,
        &reference_scores_raw0,
    );

    // STAGE 6: probabilities. `window0` (1024) exceeds `ids.len()` for this
    // prompt, so the windowed mask is provably inert here -- plain causal
    // softmax is the correct reference at this sequence length.
    let reference_probabilities0 =
        causal_softmax_ref(engine_scores_scaled, ids.len(), kv_heads0, group0);
    println!(
        "STAGE6 layer0 window={window0} positions={} window_inert={}",
        ids.len(),
        window0 as usize >= ids.len()
    );
    report_diff(
        "STAGE6 layer0 probabilities vs independent causal softmax(engine scores_scaled)",
        engine_probabilities,
        &reference_probabilities0,
    );

    // STAGE 7: attended = probabilities . v_normed0 (the POST-value_norm V).
    let reference_attended0 = attended_from_probabilities_ref(
        engine_probabilities,
        engine_v_normed0,
        ids.len(),
        kv_heads0,
        group0,
        head_dim0,
    );
    report_diff(
        "STAGE7a layer0 attended vs independent probabilities.v_normed0 combine",
        engine_attended,
        &reference_attended0,
    );

    let (wo0_bytes, wo0_dims) = dequant_tensor(&parsed, file_bytes, "blk.0.attn_output.weight");
    let wo0_type = tensor_type(&parsed, "blk.0.attn_output.weight");
    let attended_in_features0 = query_heads0 * head_dim0;
    println!(
        "blk.0.attn_output.weight: dims={wo0_dims:?} ggml_type={wo0_type:?} expected_in_features={attended_in_features0} embedding={embedding}"
    );
    let out_features_o0 =
        (wo0_dims.iter().product::<u64>() / attended_in_features0 as u64) as usize;
    let reference_attn_out_raw0 = matmul_rows(
        &engine_attended.to_vec(),
        &wo0_bytes,
        ids.len(),
        attended_in_features0,
        out_features_o0,
    );
    report_diff(
        "STAGE7b layer0 attn_out (pre post_attention_norm) vs independent o_proj matmul(engine attended)",
        engine_attn_out_raw,
        &reference_attn_out_raw0,
    );

    let (pan0_gamma, pan0_dims) =
        dequant_tensor(&parsed, file_bytes, "blk.0.post_attention_norm.weight");
    println!("blk.0.post_attention_norm.weight: dims={pan0_dims:?}");
    let reference_attn_out_normed0 = rmsnorm_per_head_raw_gamma_ref(
        engine_attn_out_raw,
        &pan0_gamma,
        ids.len(),
        1,
        embedding,
        eps0,
    );
    report_diff(
        "STAGE7c layer0 attn_out_normed0 vs independent RAW-gamma post_attention_norm(engine attn_out_raw)",
        engine_attn_out_normed0,
        &reference_attn_out_normed0,
    );
    let reference_post_mixer0: Vec<f32> = engine_x0
        .iter()
        .zip(reference_attn_out_normed0.iter())
        .map(|(&residual, &normed)| residual + normed)
        .collect();
    report_diff(
        "STAGE7d layer0 post_mixer vs independent (engine_x0 + independent post_attention_norm(engine attn_out_raw))",
        engine_post_mixer,
        &reference_post_mixer0,
    );

    println!("\n== LAYER 5 (FULL attention) diagnostic: v_norm / score-scaling / partial-rope ==");
    let engine_normed5 = &values[20];
    let engine_q_raw5 = &values[21];
    let engine_q_normed5 = &values[22];
    let engine_k_raw5 = &values[23];
    let engine_k_normed5 = &values[24];
    let engine_v5 = &values[25];
    let engine_rotated_q_even5 = &values[26];
    let engine_rotated_q_odd5 = &values[27];
    let engine_rotated_k_even5 = &values[28];
    let engine_rotated_k_odd5 = &values[29];
    let engine_scores_raw5 = &values[30];
    let engine_scores_scaled5 = &values[31];
    let engine_probabilities5 = &values[32];
    let engine_attended5 = &values[33];
    let engine_attn_out_raw5 = &values[34];
    let engine_attn_out_normed5 = &values[35];
    let engine_post_mixer5 = &values[36];
    let engine_x5 = &values[37];

    let kv_heads5 = architecture.kv_heads_by_layer[5] as usize;
    let head_dim5 = architecture.key_length as usize;
    let query_heads5 = architecture.head_count as usize;
    let group5 = query_heads5 / kv_heads5;
    let rope_base5 = architecture.rope_freq_base;
    let eps5 = architecture.rms_epsilon;
    println!(
        "layer5 shapes: kv_heads={kv_heads5} head_dim={head_dim5} query_heads={query_heads5} group={group5} rope_base={rope_base5} positions={}",
        ids.len()
    );

    // STAGE 1: q_raw5/k_raw5 sanity -- independent dequant+matmul against the
    // engine's OWN post-attn-norm input (`normed5`), same convention the
    // layer-0 PRIORITY-1 check above uses.
    let (wq5_bytes, wq5_dims) = dequant_tensor(&parsed, file_bytes, "blk.5.attn_q.weight");
    let (wk5_bytes, wk5_dims) = dequant_tensor(&parsed, file_bytes, "blk.5.attn_k.weight");
    let wq5_type = tensor_type(&parsed, "blk.5.attn_q.weight");
    let wk5_type = tensor_type(&parsed, "blk.5.attn_k.weight");
    println!("blk.5.attn_q.weight: dims={wq5_dims:?} ggml_type={wq5_type:?}");
    println!("blk.5.attn_k.weight: dims={wk5_dims:?} ggml_type={wk5_type:?}");
    let out_features_q5 = (wq5_dims.iter().product::<u64>() / embedding as u64) as usize;
    let out_features_k5 = (wk5_dims.iter().product::<u64>() / embedding as u64) as usize;
    let reference_q_raw5 = matmul_rows(
        engine_normed5,
        &wq5_bytes,
        ids.len(),
        embedding,
        out_features_q5,
    );
    let reference_k_raw5 = matmul_rows(
        engine_normed5,
        &wk5_bytes,
        ids.len(),
        embedding,
        out_features_k5,
    );
    report_diff("STAGE1 layer5 q_raw (pre qk-norm)", engine_q_raw5, &reference_q_raw5);
    report_diff("STAGE1 layer5 k_raw (pre qk-norm)", engine_k_raw5, &reference_k_raw5);

    // STAGE 2: q_normed5/k_normed5 -- independent rmsnorm_per_head with the
    // RAW (no +1) q_norm/k_norm gamma, over the FULL head_dim=512, confirming
    // both the RAW-gamma binding AND the per-head reduction axis.
    let (qn5_gamma, qn5_dims) = dequant_tensor(&parsed, file_bytes, "blk.5.attn_q_norm.weight");
    let (kn5_gamma, kn5_dims) = dequant_tensor(&parsed, file_bytes, "blk.5.attn_k_norm.weight");
    println!("blk.5.attn_q_norm.weight: dims={qn5_dims:?} blk.5.attn_k_norm.weight: dims={kn5_dims:?}");
    let reference_q_normed5 = rmsnorm_per_head_raw_gamma_ref(
        engine_q_raw5,
        &qn5_gamma,
        ids.len(),
        query_heads5,
        head_dim5,
        eps5,
    );
    let reference_k_normed5 = rmsnorm_per_head_raw_gamma_ref(
        engine_k_raw5,
        &kn5_gamma,
        ids.len(),
        kv_heads5,
        head_dim5,
        eps5,
    );
    report_diff(
        "STAGE2 layer5 q_normed vs rmsnorm_per_head(RAW q_norm gamma, head_dim=512)",
        engine_q_normed5,
        &reference_q_normed5,
    );
    report_diff(
        "STAGE2 layer5 k_normed vs rmsnorm_per_head(RAW k_norm gamma, head_dim=512)",
        engine_k_normed5,
        &reference_k_normed5,
    );

    // CHECK 1 / STAGE 4: v_norm. HF-correct V = weightless per-kv-head RMSNorm of the
    // RAW k_proj output (`Gemma4TextAttention.forward`, no gamma, no RoPE).
    let reference_v5 =
        rmsnorm_per_head_no_scale_ref(engine_k_raw5, ids.len(), kv_heads5, head_dim5, eps5);
    report_diff(
        "layer5 v (post value_norm) vs weightless-rmsnorm-per-head(k_raw)",
        engine_v5,
        &reference_v5,
    );

    // CHECK 2: score scaling. HF-correct scaling = 1.0 -- gemma4 has no
    // `query_pre_attn_scalar` and no `1/sqrt(head_dim)`.
    // (`attention_forward.rs:627-632`: `scores_scaled = scores *
    // inv_sqrt_head_dim`, `inv_sqrt_head_dim = 1/sqrt(query_pre_attn_scalar)`,
    // `query_pre_attn_scalar` hard-set 256 for every gemma4 layer in
    // `gemma4_attention_configs`, `gemma4/bind.rs:502`.)
    let mut factor_samples: Vec<f32> = Vec::new();
    for (&raw, &scaled) in engine_scores_raw5.iter().zip(engine_scores_scaled5.iter()) {
        if raw.abs() > 1e-3 {
            factor_samples.push(scaled / raw);
        }
    }
    let factor_mean = factor_samples.iter().sum::<f32>() / factor_samples.len() as f32;
    let factor_max_dev = factor_samples
        .iter()
        .fold(0f32, |max, &factor| max.max((factor - factor_mean).abs()));
    println!(
        "layer5 score scaling: engine_factor(scaled/raw) mean={factor_mean:.6} max_dev_from_mean={factor_max_dev:e} n_samples={} | HF-correct=1.0 | 1/sqrt(head_dim=512)={:.6} | 1/sqrt(query_pre_attn_scalar=256)={:.6}",
        factor_samples.len(),
        1.0 / (head_dim5 as f32).sqrt(),
        1.0 / 256f32.sqrt()
    );
    println!(
        "layer5 score scaling: engine_factor_vs_HF_1.0 diff={:e} -- proxima SCALES where HF does NOT: {}",
        (factor_mean - 1.0).abs(),
        (factor_mean - 1.0).abs() > 1e-3
    );

    // CHECK 3: full-layer RoPE. HF-correct is PARTIAL: rotary_dim = 128 (only
    // the first 64 frequency pairs get a real rotation at base 1e6), the
    // remaining 384 dims (pairs 64..256) are identity. The engine's OWN
    // cos/sin table (`residency_caches.rs:939-966`, `build_position_inputs`)
    // computes ALL 256 pairs across the FULL head_dim=512 -- reconstruct
    // engine's actual roped K from `rotated_k_even5`/`rotated_k_odd5` and
    // compare independently against BOTH a full-256-pair rotation and a
    // partial-64-pair rotation of engine's own pre-rope `k_normed5`.
    let mut engine_roped_k5 = vec![0f32; ids.len() * kv_heads5 * head_dim5];
    let pairs_full = head_dim5 / 2;
    for s in 0..ids.len() {
        for u in 0..kv_heads5 {
            for pair in 0..pairs_full {
                let even_index = (s * kv_heads5 + u) * pairs_full + pair;
                let base_offset = (s * kv_heads5 + u) * head_dim5;
                engine_roped_k5[base_offset + pair] = engine_rotated_k_even5[even_index];
                engine_roped_k5[base_offset + pairs_full + pair] =
                    engine_rotated_k_odd5[even_index];
            }
        }
    }
    let full_reference_k5 = rope_split_half_ref(
        engine_k_normed5,
        ids.len(),
        kv_heads5,
        head_dim5,
        pairs_full,
        rope_base5,
        head_dim5,
    );
    // llama.cpp's authoritative gemma4 graph calls `ggml_rope_ext` with
    // `n_rot=head_dim` (512) for full/global layers -- the frequency
    // divisor in `theta = position * base^(-2*pair/n_rot)` stays `head_dim`
    // for EVERY pair, never the narrower rotary width. The partial-rotary
    // behaviour comes entirely from the `rope_freqs.weight` `freq_factors`
    // tensor (`[1.0]*64 + [1e30]*192`), which freezes pairs 64..256 to
    // `cos=1, sin=0` -- exactly what `rope_split_half_ref` already produces
    // for untouched channels `[2*pairs, head_dim)`. `rope_split_half_partial_ref`
    // (below) instead divides by the narrower `rotary_dim` (HF's naive
    // `partial_rotary_factor` convention), which is NOT what this
    // checkpoint's GGUF `rope_freqs.weight` tensor encodes -- kept only as
    // a labelled negative control.
    let partial_pairs = 64;
    let partial_reference_k5 = rope_split_half_ref(
        engine_k_normed5,
        ids.len(),
        kv_heads5,
        head_dim5,
        partial_pairs,
        rope_base5,
        head_dim5,
    );
    report_diff_ranged(
        "layer5 roped K vs FULL-256-pair rotation (engine's own formula)",
        &engine_roped_k5,
        &full_reference_k5,
        head_dim5,
    );
    report_diff_ranged(
        "layer5 roped K vs PARTIAL-64-pair rotation (HF-correct)",
        &engine_roped_k5,
        &partial_reference_k5,
        head_dim5,
    );

    // STAGE 3 (Q side): same FULL-vs-PARTIAL rope A/B as K above, applied to
    // the 16 query heads via [`reconstruct_roped`].
    let engine_roped_q5 = reconstruct_roped(
        engine_rotated_q_even5,
        engine_rotated_q_odd5,
        ids.len(),
        query_heads5,
        head_dim5,
    );
    let full_reference_q5 = rope_split_half_ref(
        engine_q_normed5,
        ids.len(),
        query_heads5,
        head_dim5,
        pairs_full,
        rope_base5,
        head_dim5,
    );
    let partial_reference_q5 = rope_split_half_ref(
        engine_q_normed5,
        ids.len(),
        query_heads5,
        head_dim5,
        partial_pairs,
        rope_base5,
        head_dim5,
    );
    report_diff_ranged(
        "layer5 roped Q vs FULL-256-pair rotation (engine's own formula)",
        &engine_roped_q5,
        &full_reference_q5,
        head_dim5,
    );
    report_diff_ranged(
        "layer5 roped Q vs PARTIAL-64-pair rotation (HF-correct)",
        &engine_roped_q5,
        &partial_reference_q5,
        head_dim5,
    );

    // STAGE 5: scores (pre-scale). Independent GQA contraction over the
    // engine's OWN roped Q/K (reconstructed from taps above) -- isolates the
    // score-contraction/GQA-grouping stage from whatever the rope-shape
    // stage (3) resolves to, since both reference and engine start from the
    // SAME roped vectors here.
    let reference_scores_raw5 = gqa_scores_ref(
        &engine_roped_q5,
        &engine_roped_k5,
        ids.len(),
        query_heads5,
        kv_heads5,
        head_dim5,
        group5,
    );
    report_diff(
        "STAGE5 layer5 scores (pre-scale) vs GQA contraction(engine roped Q, engine roped K)",
        engine_scores_raw5,
        &reference_scores_raw5,
    );
    report_diff(
        "STAGE5 layer5 scores (post-scale, scale=1.0) vs SAME unscaled GQA contraction",
        engine_scores_scaled5,
        &reference_scores_raw5,
    );

    // STAGE 6: probabilities (causal softmax) from the engine's OWN
    // scores_scaled5 -- isolates softmax/masking from the score contraction.
    let reference_probabilities5 =
        causal_softmax_ref(engine_scores_scaled5, ids.len(), kv_heads5, group5);
    report_diff(
        "STAGE6 layer5 probabilities vs independent causal softmax(engine scores_scaled)",
        engine_probabilities5,
        &reference_probabilities5,
    );

    // STAGE 7: attended = probabilities . V, from the engine's OWN
    // probabilities and V -- isolates the weighted-sum combine stage.
    let reference_attended5 = attended_from_probabilities_ref(
        engine_probabilities5,
        engine_v5,
        ids.len(),
        kv_heads5,
        group5,
        head_dim5,
    );
    report_diff(
        "STAGE7 layer5 attended vs independent probabilities.V combine(engine probabilities, engine V)",
        engine_attended5,
        &reference_attended5,
    );

    // STAGE 8: attn_out (post o_proj). Independent dequant+matmul of
    // `blk.5.attn_output.weight` against the engine's OWN `attended` --
    // checks the o_proj input layout (16 q heads * 512 head_dim = 8192 ->
    // embedding) matches the GGUF `dims[0]`-fastest convention every other
    // matmul in this file already relies on.
    let (wo5_bytes, wo5_dims) = dequant_tensor(&parsed, file_bytes, "blk.5.attn_output.weight");
    let wo5_type = tensor_type(&parsed, "blk.5.attn_output.weight");
    let attended_in_features5 = query_heads5 * head_dim5;
    println!(
        "blk.5.attn_output.weight: dims={wo5_dims:?} ggml_type={wo5_type:?} expected_in_features(16*512)={attended_in_features5} embedding={embedding}"
    );
    let out_features_o5 =
        (wo5_dims.iter().product::<u64>() / attended_in_features5 as u64) as usize;
    let reference_attn_out_raw5 = matmul_rows(
        &engine_attended5.to_vec(),
        &wo5_bytes,
        ids.len(),
        attended_in_features5,
        out_features_o5,
    );
    report_diff(
        "STAGE8 layer5 attn_out (pre post_attention_norm) vs independent o_proj matmul(engine attended)",
        engine_attn_out_raw5,
        &reference_attn_out_raw5,
    );

    // STAGE 9: post_mixer = x + post_attention_norm(attn_out) -- residual
    // add over the engine's OWN `x5` (mixer input, structurally extracted)
    // and RAW-gamma post_attention_norm of the engine's OWN `attn_out_raw5`.
    let (pan5_gamma, pan5_dims) =
        dequant_tensor(&parsed, file_bytes, "blk.5.post_attention_norm.weight");
    println!("blk.5.post_attention_norm.weight: dims={pan5_dims:?}");
    let reference_attn_out_normed5 = rmsnorm_per_head_raw_gamma_ref(
        engine_attn_out_raw5,
        &pan5_gamma,
        ids.len(),
        1,
        embedding,
        eps5,
    );
    report_diff(
        "STAGE9a layer5 attn_out_normed vs independent RAW-gamma post_attention_norm(engine attn_out_raw)",
        engine_attn_out_normed5,
        &reference_attn_out_normed5,
    );
    let reference_post_mixer5: Vec<f32> = engine_x5
        .iter()
        .zip(reference_attn_out_normed5.iter())
        .map(|(&residual, &normed)| residual + normed)
        .collect();
    report_diff(
        "STAGE9b layer5 post_mixer vs independent (engine_x5 + independent post_attention_norm(engine attn_out_raw))",
        engine_post_mixer5,
        &reference_post_mixer5,
    );

    println!(
        "\n== PRONG 1: layer-by-layer |max| magnitude progression (post_mixer / dense_raw / dense_out / router_logits / routed_raw / routed_out / combined / x) =="
    );
    let ffn_base = 38usize;
    let mut layer_values: Vec<[&[f32]; 8]> = Vec::new();
    for layer_index in 0..probe_layers.len() {
        let start = ffn_base + layer_index * 8;
        layer_values.push([
            &values[start],
            &values[start + 1],
            &values[start + 2],
            &values[start + 3],
            &values[start + 4],
            &values[start + 5],
            &values[start + 6],
            &values[start + 7],
        ]);
    }
    for (&layer, taps_values) in probe_layers.iter().zip(layer_values.iter()) {
        println!(
            "  layer {layer}: post_mixer={:.3} dense_raw={:.3} dense_out={:.3} router_logits={:.3} routed_raw={:.3} routed_out={:.3} combined={:.3} x={:.3}",
            max_abs(taps_values[0]),
            max_abs(taps_values[1]),
            max_abs(taps_values[2]),
            max_abs(taps_values[3]),
            max_abs(taps_values[4]),
            max_abs(taps_values[5]),
            max_abs(taps_values[6]),
            max_abs(taps_values[7]),
        );
    }
    let final_base = ffn_base + probe_layers.len() * 8;
    let engine_x_final = &values[final_base];
    let engine_normed_final = &values[final_base + 1];
    let engine_logits_pre = &values[final_base + 2];
    let engine_logits_post = &values[final_base + 3];
    let engine_normed2_layer0 = &values[final_base + 4];
    let engine_routed_input_layer0 = &values[final_base + 5];
    println!(
        "  final: x_final={:.3} output_norm={:.3} logits_pre_softcap={:.3} logits_post_softcap={:.3}",
        max_abs(engine_x_final),
        max_abs(engine_normed_final),
        max_abs(engine_logits_pre),
        max_abs(engine_logits_post)
    );

    println!(
        "\n== FFN-TAIL COMPOSITION: independent x reconstruction from engine dense_out/routed_out/post_mixer + dequantized post_ffw_norm.weight + layer_output_scale.weight, op order per gemma4.cpp:343-365,392-395 (residual add BEFORE the scale multiply, same `cur`/`x` node) =="
    );
    for &layer in &[0u32, 5u32] {
        let layer_index = probe_layers
            .iter()
            .position(|&candidate| candidate == layer)
            .unwrap_or_else(|| panic!("layer {layer} present in probe_layers"));
        let taps_values = &layer_values[layer_index];
        let engine_post_mixer = taps_values[0];
        let engine_dense_out = taps_values[2];
        let engine_routed_out = taps_values[5];
        let engine_combined = taps_values[6];
        let engine_x = taps_values[7];

        let (post_ffw_norm_gamma, _) =
            dequant_tensor(&parsed, file_bytes, &format!("blk.{layer}.post_ffw_norm.weight"));
        let (layer_output_scale, layer_output_scale_dims) =
            dequant_tensor(&parsed, file_bytes, &format!("blk.{layer}.layer_output_scale.weight"));
        let los_min = layer_output_scale.iter().cloned().fold(f32::INFINITY, f32::min);
        let los_max = layer_output_scale
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let los_mean = layer_output_scale.iter().sum::<f32>() / layer_output_scale.len() as f32;
        println!(
            "blk.{layer}.layer_output_scale.weight: dims={layer_output_scale_dims:?} min={los_min:.6} max={los_max:.6} mean={los_mean:.6} first8={:?}",
            &layer_output_scale[0..8.min(layer_output_scale.len())]
        );

        let reference_combined: Vec<f32> = engine_dense_out
            .chunks(embedding)
            .zip(engine_routed_out.chunks(embedding))
            .flat_map(|(dense_row, routed_row)| {
                let summed: Vec<f32> = dense_row
                    .iter()
                    .zip(routed_row.iter())
                    .map(|(&dense_value, &routed_value)| dense_value + routed_value)
                    .collect();
                rmsnorm_ref(&summed, &post_ffw_norm_gamma)
            })
            .collect();
        report_diff(
            &format!(
                "layer{layer} combined vs independent rmsnorm(dense_out+routed_out, post_ffw_norm.weight)"
            ),
            engine_combined,
            &reference_combined,
        );

        // x = layer_output_scale * (combined + post_mixer) -- residual add
        // THEN scale multiply, matching gemma4.cpp line 365 (`cur =
        // ggml_add(cur, attn_out)`) followed by line 392-395 (`cur =
        // ggml_mul(cur, out_scale)`), and proxima's own
        // `attention_forward.rs:1523-1543` (`x = ffn_out + post_mixer` THEN
        // `x = x * output_scale`) -- both scale the FULL post-residual `x`,
        // not the FFN delta alone.
        let reference_x: Vec<f32> = reference_combined
            .chunks(embedding)
            .zip(engine_post_mixer.chunks(embedding))
            .flat_map(|(combined_row, mixer_row)| {
                combined_row
                    .iter()
                    .zip(mixer_row.iter())
                    .zip(layer_output_scale.iter())
                    .map(|((&combined_value, &mixer_value), &scale)| {
                        (combined_value + mixer_value) * scale
                    })
                    .collect::<Vec<f32>>()
            })
            .collect();
        report_diff(
            &format!(
                "layer{layer} x vs independent layer_output_scale*(independent_combined + engine post_mixer)"
            ),
            engine_x,
            &reference_x,
        );
    }

    println!("\n== BISECTION: final-logit-path taps (last-token row) ==");
    let positions = ids.len();
    let last_token_slice = |values: &[f32], dim: usize| -> Vec<f32> {
        values[(positions - 1) * dim..positions * dim].to_vec()
    };
    let x_final_last = last_token_slice(engine_x_final, embedding);
    let normed_final_last = last_token_slice(engine_normed_final, embedding);
    let token_vocab = &vocab;
    let vocab = architecture.vocab as usize;
    let logits_pre_last = last_token_slice(engine_logits_pre, vocab);
    let logits_post_last = last_token_slice(engine_logits_post, vocab);

    println!(
        "x_final: |max|={:.4} mean={:.6} top5(channel,value)={:?}",
        max_abs(&x_final_last),
        x_final_last.iter().sum::<f32>() / x_final_last.len() as f32,
        top_k(&x_final_last, 5)
    );
    println!(
        "output_norm_out: |max|={:.4} mean={:.6} top5(channel,value)={:?}",
        max_abs(&normed_final_last),
        normed_final_last.iter().sum::<f32>() / normed_final_last.len() as f32,
        top_k(&normed_final_last, 5)
    );
    let logits_pre_top5 = top_k(&logits_pre_last, 5);
    println!(
        "engine logits_pre_softcap: |max|={:.4} top5(token_id,value)={:?}",
        max_abs(&logits_pre_last),
        logits_pre_top5
    );
    for (token_id, value) in &logits_pre_top5 {
        let piece = token_vocab.token_str(*token_id as u32);
        println!("  decode pre-softcap top: token_id={token_id} value={value:.4} piece={piece:?}");
    }
    println!(
        "engine logits_post_softcap: |max|={:.4} top5(token_id,value)={:?}",
        max_abs(&logits_post_last),
        top_k(&logits_post_last, 5)
    );

    println!("\n== BISECTION: independent recompute from engine x_final (last-token row only) ==");
    let output_norm_type = tensor_type(&parsed, "output_norm.weight");
    let (output_norm_gamma, output_norm_dims) =
        dequant_tensor(&parsed, file_bytes, "output_norm.weight");
    println!("output_norm.weight: dims={output_norm_dims:?} ggml_type={output_norm_type:?}");
    let reference_normed_final = rmsnorm_ref(&x_final_last, &output_norm_gamma);
    let mut norm_max_abs_diff = 0f32;
    for (engine_value, reference_value) in
        normed_final_last.iter().zip(reference_normed_final.iter())
    {
        norm_max_abs_diff = norm_max_abs_diff.max((engine_value - reference_value).abs());
    }
    println!(
        "output_norm INDEPENDENT-vs-ENGINE: max_abs_diff={norm_max_abs_diff:e} engine[0..4]={:?} reference[0..4]={:?}",
        &normed_final_last[0..4],
        &reference_normed_final[0..4]
    );

    println!(
        "dequantizing full token_embd.weight (tied lm_head, Q6_K, vocab x embedding) -- one-time cost for this probe"
    );
    let (token_embd_full, token_embd_dims) =
        dequant_tensor(&parsed, file_bytes, "token_embd.weight");
    println!("token_embd.weight: dims={token_embd_dims:?}");
    // gemma4 TIES output to token_embd: no separate `output.weight` tensor
    // bytes are read for the projection -- confirm that at the byte level,
    // not by name alone, since the earlier `find_input(&b_program,
    // "output.weight")` only proves the GRAPH declares an `output.weight`
    // leaf, not what bytes the loader bound to it.
    let has_separate_output_tensor = parsed
        .tensors
        .iter()
        .any(|tensor| tensor.name == "output.weight");
    println!("separate output.weight tensor present in checkpoint: {has_separate_output_tensor}");

    let independent_logits_pre = matmul_rows(
        &reference_normed_final,
        &token_embd_full,
        1,
        embedding,
        vocab,
    );
    let logit_softcap = architecture.final_logit_softcapping;
    let independent_logits_post: Vec<f32> = independent_logits_pre
        .iter()
        .map(|&logit| logit_softcap * (logit / logit_softcap).tanh())
        .collect();
    println!(
        "independent logits_pre_softcap: |max|={:.4} top5(token_id,value)={:?}",
        max_abs(&independent_logits_pre),
        top_k(&independent_logits_pre, 5)
    );
    println!(
        "independent logits_post_softcap: |max|={:.4} top5(token_id,value)={:?}",
        max_abs(&independent_logits_post),
        top_k(&independent_logits_post, 5)
    );

    // sanity: is the engine leaking the sqrt(embedding)~53x INPUT embedding
    // scale into the tied OUTPUT projection? Recompute logits a second way,
    // using token_embd rows multiplied by the same sqrt(embedding) scale the
    // input side applies, to see whether THAT version (not the raw-weight
    // version above) is the one that matches the engine.
    let embed_scale = (embedding as f32).sqrt();
    let scaled_token_embd: Vec<f32> = token_embd_full
        .iter()
        .map(|&value| value * embed_scale)
        .collect();
    let scaled_logits_pre = matmul_rows(
        &reference_normed_final,
        &scaled_token_embd,
        1,
        embedding,
        vocab,
    );
    println!(
        "independent logits_pre_softcap IF sqrt(embedding) scale leaked into lm_head: |max|={:.4} top5(token_id,value)={:?}",
        max_abs(&scaled_logits_pre),
        top_k(&scaled_logits_pre, 5)
    );

    println!(
        "\n== PRONG 2: layer-0 dense GeGLU FFN, independent dequant+matmul vs engine dense_raw =="
    );
    let (wgate_bytes, wgate_dims) = dequant_tensor(&parsed, file_bytes, "blk.0.ffn_gate.weight");
    let (wup_bytes, wup_dims) = dequant_tensor(&parsed, file_bytes, "blk.0.ffn_up.weight");
    let (wdown_bytes, wdown_dims) = dequant_tensor(&parsed, file_bytes, "blk.0.ffn_down.weight");
    println!(
        "ffn_gate.weight dims={wgate_dims:?} ffn_up.weight dims={wup_dims:?} ffn_down.weight dims={wdown_dims:?}"
    );
    let feed_forward = architecture.feed_forward as usize;
    let reference_gate = matmul_rows(
        engine_normed2_layer0,
        &wgate_bytes,
        ids.len(),
        embedding,
        feed_forward,
    );
    let reference_up = matmul_rows(
        engine_normed2_layer0,
        &wup_bytes,
        ids.len(),
        embedding,
        feed_forward,
    );
    let reference_hidden: Vec<f32> = reference_gate
        .iter()
        .zip(reference_up.iter())
        .map(|(&gate, &up)| gelu_tanh(gate) * up)
        .collect();
    let reference_dense_raw = matmul_rows(
        &reference_hidden,
        &wdown_bytes,
        ids.len(),
        feed_forward,
        embedding,
    );
    report_diff(
        "layer0 dense_raw (pre post_ffw_norm_1)",
        layer_values[0][1],
        &reference_dense_raw,
    );

    println!(
        "\n== PRONG 2: layer-0 router -- scale tensor magnitudes + engine router_logits values =="
    );
    let (router_scale_values, router_scale_dims) =
        dequant_tensor(&parsed, file_bytes, "blk.0.ffn_gate_inp.scale");
    let (expert_scale_values, expert_scale_dims) =
        dequant_tensor(&parsed, file_bytes, "blk.0.ffn_down_exps.scale");
    println!(
        "blk.0.ffn_gate_inp.scale dims={router_scale_dims:?} |max|={:.3} mean={:.4}",
        max_abs(&router_scale_values),
        router_scale_values.iter().sum::<f32>() / router_scale_values.len() as f32
    );
    println!(
        "blk.0.ffn_down_exps.scale dims={expert_scale_dims:?} |max|={:.3} mean={:.4} first8={:?}",
        max_abs(&expert_scale_values),
        expert_scale_values.iter().sum::<f32>() / expert_scale_values.len() as f32,
        &expert_scale_values[0..8.min(expert_scale_values.len())]
    );
    let engine_router_logits_layer0 = layer_values[0][3];
    println!(
        "engine router_logits(layer0) |max|={:.3} first8(position0)={:?}",
        max_abs(engine_router_logits_layer0),
        &engine_router_logits_layer0[0..8.min(engine_router_logits_layer0.len())]
    );

    println!(
        "\n== PRONG 3: layer-0 ROUTED EXPERT MoE, independent Q3_K gate/up + Q5_1 down dequant vs engine routed_raw (last-token row) =="
    );
    let expert_count = architecture.expert_count as usize;
    let expert_used_count = architecture.expert_used_count as usize;
    let last_position = ids.len() - 1;
    let router_logits_row = &engine_router_logits_layer0
        [last_position * expert_count..(last_position + 1) * expert_count];
    let routed_input_row =
        &engine_routed_input_layer0[last_position * embedding..(last_position + 1) * embedding];
    let engine_routed_raw = layer_values[0][4];
    let engine_routed_raw_row =
        &engine_routed_raw[last_position * embedding..(last_position + 1) * embedding];

    let selected = select_top_experts(router_logits_row, expert_used_count);
    println!("selected experts (id, raw_logit, normalized_weight, expert_scale, combine_weight):");
    for &(expert_id, raw_logit, normalized_weight) in &selected {
        let scale = expert_scale_values[expert_id];
        println!(
            "  expert={expert_id} raw_logit={raw_logit:.4} normalized_weight={normalized_weight:.6} expert_scale={scale:.4} combine_weight={:.6}",
            normalized_weight * scale
        );
    }

    let fused_name = "blk.0.ffn_gate_up_exps.weight";
    let fused_ggml_type = tensor_type(&parsed, fused_name);
    let fused_tensor = parsed
        .tensors
        .iter()
        .find(|tensor| tensor.name == fused_name)
        .expect("fused ffn_gate_up_exps.weight tensor present");
    let fused_range = parsed
        .tensor_data_range(fused_tensor, file_bytes.len() as u64)
        .expect("fused tensor data range");
    let fused_source = &file_bytes[fused_range.start as usize..fused_range.end as usize];
    let fused_layout = fused_ggml_type.block_layout();
    let bytes_per_row_gateup =
        (embedding as u64 / fused_layout.block_elements) * fused_layout.block_bytes;
    let gate_bytes = architecture.expert_feed_forward as u64 * bytes_per_row_gateup;

    let down_name = "blk.0.ffn_down_exps.weight";
    let down_ggml_type = tensor_type(&parsed, down_name);
    let down_tensor = parsed
        .tensors
        .iter()
        .find(|tensor| tensor.name == down_name)
        .expect("ffn_down_exps.weight tensor present");
    let down_range = parsed
        .tensor_data_range(down_tensor, file_bytes.len() as u64)
        .expect("down tensor data range");
    let down_full_source = &file_bytes[down_range.start as usize..down_range.end as usize];
    let down_layout = down_ggml_type.block_layout();
    let bytes_per_row_down = (architecture.expert_feed_forward as u64 / down_layout.block_elements)
        * down_layout.block_bytes;
    let per_expert_bytes_down = embedding as u64 * bytes_per_row_down;
    println!(
        "fused ffn_gate_up_exps.weight: ggml_type={fused_ggml_type:?} dims={:?} bytes_per_row={bytes_per_row_gateup} gate_bytes={gate_bytes}",
        fused_tensor.dims
    );
    println!(
        "ffn_down_exps.weight: ggml_type={down_ggml_type:?} dims={:?} bytes_per_row={bytes_per_row_down} per_expert_bytes={per_expert_bytes_down}",
        down_tensor.dims
    );

    let expert_feed_forward = architecture.expert_feed_forward as usize;
    let mut reference_routed_correct = vec![0f32; embedding];
    let mut reference_routed_swapped = vec![0f32; embedding];
    for &(expert_id, _raw_logit, normalized_weight) in &selected {
        let combine_weight = normalized_weight * expert_scale_values[expert_id];
        let expert_start = expert_id as u64 * 2 * gate_bytes;
        let gate_source =
            &fused_source[expert_start as usize..(expert_start + gate_bytes) as usize];
        let up_source = &fused_source
            [(expert_start + gate_bytes) as usize..(expert_start + 2 * gate_bytes) as usize];

        let down_start = expert_id as u64 * per_expert_bytes_down;
        let down_source =
            &down_full_source[down_start as usize..(down_start + per_expert_bytes_down) as usize];
        let down_dequant = dequant_range(
            down_ggml_type,
            down_source,
            embedding * expert_feed_forward,
        );

        let gate_dequant_correct =
            dequant_range(fused_ggml_type, gate_source, expert_feed_forward * embedding);
        let up_dequant_correct =
            dequant_range(fused_ggml_type, up_source, expert_feed_forward * embedding);
        let out_correct = compute_expert_ffn(
            routed_input_row,
            &gate_dequant_correct,
            &up_dequant_correct,
            &down_dequant,
            embedding,
            expert_feed_forward,
        );
        for (accumulated, value) in reference_routed_correct.iter_mut().zip(out_correct.iter()) {
            *accumulated += value * combine_weight;
        }

        // DEGENERATE CONTROL: gate/up rows swapped relative to
        // `bind_gemma4_fused_gate_up_experts`'s own split
        // (`gemma4/bind.rs:394-440`) -- must NOT match the engine.
        let gate_dequant_swapped =
            dequant_range(fused_ggml_type, up_source, expert_feed_forward * embedding);
        let up_dequant_swapped =
            dequant_range(fused_ggml_type, gate_source, expert_feed_forward * embedding);
        let out_swapped = compute_expert_ffn(
            routed_input_row,
            &gate_dequant_swapped,
            &up_dequant_swapped,
            &down_dequant,
            embedding,
            expert_feed_forward,
        );
        for (accumulated, value) in reference_routed_swapped.iter_mut().zip(out_swapped.iter()) {
            *accumulated += value * combine_weight;
        }
    }
    report_diff(
        "layer0 routed_raw (CORRECT gate/up split, last-token row)",
        engine_routed_raw_row,
        &reference_routed_correct,
    );
    report_diff(
        "layer0 routed_raw (SWAPPED gate/up split -- degenerate control, MUST diverge)",
        engine_routed_raw_row,
        &reference_routed_swapped,
    );

    // layer-0 FFN tail: `layer_output_scale.weight` value itself -- a per-
    // embedding-channel vector on disk (not a scalar), reported here as its
    // own min/max/mean rather than a single "the value" number.
    let (layer_output_scale0, layer_output_scale0_dims) =
        dequant_tensor(&parsed, file_bytes, "blk.0.layer_output_scale.weight");
    let los0_min = layer_output_scale0.iter().cloned().fold(f32::INFINITY, f32::min);
    let los0_max = layer_output_scale0
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let los0_mean = layer_output_scale0.iter().sum::<f32>() / layer_output_scale0.len() as f32;
    println!(
        "blk.0.layer_output_scale.weight: dims={layer_output_scale0_dims:?} min={los0_min:.6} max={los0_max:.6} mean={los0_mean:.6} first8={:?}",
        &layer_output_scale0[0..8.min(layer_output_scale0.len())]
    );

    println!("\n== summary ==");
    println!("embedding A-vs-B max_abs_diff={max_abs_diff:e}");
    println!(
        "engine tap magnitudes: q_raw|max|={:.3} k_raw|max|={:.3} v_normed0|max|={:.3} q_normed|max|={:.3} k_normed|max|={:.3} scores_raw|max|={:.3} scores_scaled|max|={:.3} probabilities|max|={:.3} attended|max|={:.3} attn_out_raw|max|={:.3} post_mixer|max|={:.3}",
        max_abs(engine_q_raw),
        max_abs(engine_k_raw),
        max_abs(engine_v_normed0),
        max_abs(engine_q_normed),
        max_abs(engine_k_normed),
        max_abs(engine_scores_raw),
        max_abs(engine_scores_scaled),
        max_abs(engine_probabilities),
        max_abs(engine_attended),
        max_abs(engine_attn_out_raw),
        max_abs(engine_post_mixer),
    );
}

fn tensor_type(parsed: &proxima_gguf::ParsedGguf, name: &str) -> GgmlType {
    parsed
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .unwrap_or_else(|| panic!("tensor {name:?} present"))
        .ggml_type
}

/// `weights` dequantized as `out_features` rows of `in_features` contiguous
/// elements each (GGUF's own `dims[0]`-fastest convention, the same one the
/// embedding stage above already proved byte-exact against the engine).
/// `reference[pos * out_features + o] = sum_i x[pos * in_features + i] *
/// weights[o * in_features + i]` -- the exact contraction
/// `append_attention_mixer`'s `"si->shdi","ihd->shdi"` einsum pattern
/// describes, with the head/head_dim axes flattened into `out_features`.
fn matmul_rows(
    x: &[f32],
    weights: &[f32],
    positions: usize,
    in_features: usize,
    out_features: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; positions * out_features];
    for position in 0..positions {
        let x_row = &x[position * in_features..(position + 1) * in_features];
        for feature in 0..out_features {
            let w_row = &weights[feature * in_features..(feature + 1) * in_features];
            let mut sum = 0f32;
            for (left, right) in x_row.iter().zip(w_row.iter()) {
                sum += left * right;
            }
            out[position * out_features + feature] = sum;
        }
    }
    out
}

/// Gemma 4's GeGLU activation, matching `append_activation`'s
/// [`Activation::GeluTanh`] arm (`attention_forward.rs:111-170`) op-for-op:
/// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
fn gelu_tanh(x: f32) -> f32 {
    let inner = x + 0.044_715 * x * x * x;
    0.5 * x * (1.0 + (0.797_884_6 * inner).tanh())
}

fn max_abs(values: &[f32]) -> f32 {
    values.iter().fold(0f32, |max, value| max.max(value.abs()))
}

/// Top-`count` (index, value) pairs by value, descending. `f32::total_cmp`
/// plus an index tiebreaker -- ties (e.g. a softcap wall at exactly 30.0)
/// must not resolve by nondeterministic scan order.
fn top_k(values: &[f32], count: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
    indexed.truncate(count);
    indexed
}

/// Gemma's RMSNorm: `x / sqrt(mean(x^2) + eps) * (1 + gamma)` -- the `1 +
/// gamma` convention (not bare `gamma`) is gemma-family specific, matching
/// `rmsnorm` (`proxima-tensor/src/spec/primitives.rs:858`).
fn rmsnorm_ref(x: &[f32], gamma: &[f32]) -> Vec<f32> {
    let mean_square = x.iter().map(|&value| value * value).sum::<f32>() / x.len() as f32;
    let inv_rms = 1.0 / (mean_square + 1e-6).sqrt();
    x.iter()
        .zip(gamma.iter())
        .map(|(&value, &gamma_value)| value * inv_rms * (1.0 + gamma_value))
        .collect()
}

fn report_diff(label: &str, engine: &[f32], reference: &[f32]) {
    assert_eq!(engine.len(), reference.len(), "{label}: length mismatch");
    let mut max_abs_diff = 0f32;
    let mut worst_index = 0usize;
    for (index, (&engine_value, &reference_value)) in
        engine.iter().zip(reference.iter()).enumerate()
    {
        let diff = (engine_value - reference_value).abs();
        if diff > max_abs_diff {
            max_abs_diff = diff;
            worst_index = index;
        }
    }
    println!(
        "A-vs-B {label}: max_abs_diff={max_abs_diff:e} worst_index={worst_index} engine={:.6} reference={:.6} engine[0..4]={:?} reference[0..4]={:?}",
        engine[worst_index],
        reference[worst_index],
        &engine[0..4.min(engine.len())],
        &reference[0..4.min(reference.len())],
    );
}

fn position_zero_slice(embedding: usize) -> std::ops::Range<usize> {
    0..embedding.min(8)
}

/// Reproduces `append_moe_ffn_with_projection_strategy_from_logits`'s own
/// top-k selection (`mistral_layer_moe.rs:1061-1221`) for
/// [`proxima_tensor::spec::ExpertGatingFunc::Softmax`] gating: each round
/// takes the arg-max of the still-unselected logits (ties break to the
/// HIGHEST index, matching the engine's `reduce Maximum` over
/// `mask * expert_index`), then `weight = exp(logit - first_round_max)`.
/// Since the engine's own combine step divides by `sum(weight)` over just
/// the selected set, this is mathematically the renormalized-top-k-softmax
/// HF describes (softmax-over-128 then renormalize-over-8 has the SAME
/// value as exp(shifted)-over-8/sum(exp(shifted)-over-8) -- the global
/// normalizer cancels). Returns `(expert_id, raw_logit, normalized_weight)`
/// in selection order.
fn select_top_experts(logits: &[f32], count: usize) -> Vec<(usize, f32, f32)> {
    let mut mask = vec![false; logits.len()];
    let mut selected: Vec<(usize, f32)> = Vec::with_capacity(count);
    let mut first_round_max: Option<f32> = None;
    for _ in 0..count {
        let mut best_index = 0usize;
        let mut best_value = f32::NEG_INFINITY;
        for (index, &value) in logits.iter().enumerate() {
            if mask[index] {
                continue;
            }
            let better = match value.total_cmp(&best_value) {
                std::cmp::Ordering::Greater => true,
                std::cmp::Ordering::Equal => index > best_index,
                std::cmp::Ordering::Less => false,
            };
            if better {
                best_value = value;
                best_index = index;
            }
        }
        first_round_max.get_or_insert(best_value);
        mask[best_index] = true;
        selected.push((best_index, best_value));
    }
    let global_max = first_round_max.expect("count > 0 selects at least one expert");
    let raw_weights: Vec<f32> = selected
        .iter()
        .map(|&(_, value)| (value - global_max).exp())
        .collect();
    let weight_total: f32 = raw_weights.iter().sum();
    selected
        .into_iter()
        .zip(raw_weights)
        .map(|((expert_id, raw_logit), raw_weight)| {
            (expert_id, raw_logit, raw_weight / weight_total)
        })
        .collect()
}

/// Dequantizes an arbitrary contiguous byte range of a quantized tensor --
/// [`dequant_tensor`] restricted to a caller-supplied sub-span (one expert's
/// gate/up/down slab) instead of a whole named tensor.
fn dequant_range(ggml_type: GgmlType, source: &[u8], element_count: usize) -> Vec<f32> {
    let mut out = vec![0f32; element_count];
    proxima_gguf::quant::dispatch::dequantize(ggml_type, source, &mut out)
        .unwrap_or_else(|error| panic!("dequantize byte range ({ggml_type:?}): {error:?}"));
    out
}

/// One expert's GeGLU FFN round: `down(gelu_tanh(gate . x) * (up . x))` --
/// `append_moe_round_output`'s math (`mistral_layer_moe.rs:897-930`)
/// independently reproduced against dequantized weight rows instead of the
/// engine's gathered-quantized kernel.
fn compute_expert_ffn(
    x_row: &[f32],
    gate_dequant: &[f32],
    up_dequant: &[f32],
    down_dequant: &[f32],
    embedding: usize,
    expert_feed_forward: usize,
) -> Vec<f32> {
    let gate = matmul_rows(x_row, gate_dequant, 1, embedding, expert_feed_forward);
    let up = matmul_rows(x_row, up_dequant, 1, embedding, expert_feed_forward);
    let hidden: Vec<f32> = gate
        .iter()
        .zip(up.iter())
        .map(|(&gate_value, &up_value)| gelu_tanh(gate_value) * up_value)
        .collect();
    matmul_rows(&hidden, down_dequant, 1, expert_feed_forward, embedding)
}

/// Gemma 4's `v_norm` reference: weightless per-(token, kv-head) RMSNorm over
/// the head-dim axis, no gamma, no RoPE -- `rmsnorm_per_head_no_scale`'s own
/// math (`primitives.rs:1014`) independently reproduced from `x`'s flat
/// `[position, kv_head, head_dim]` layout.
fn rmsnorm_per_head_no_scale_ref(
    x: &[f32],
    positions: usize,
    kv_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; positions * kv_heads * head_dim];
    for position in 0..positions {
        for head in 0..kv_heads {
            let base = (position * kv_heads + head) * head_dim;
            let row = &x[base..base + head_dim];
            let mean_square = row.iter().map(|&value| value * value).sum::<f32>() / head_dim as f32;
            let inv_rms = 1.0 / (mean_square + eps).sqrt();
            for (local, &value) in row.iter().enumerate() {
                out[base + local] = value * inv_rms;
            }
        }
    }
    out
}

/// Split-half RoPE reference over `pairs` frequency pairs computed against
/// `freq_dim` (the divisor inside `base^(-2*pair/freq_dim)`) -- the SAME
/// formula the engine's own table builder uses
/// (`residency_caches.rs:939-966`, `build_position_inputs`), independently
/// reproduced here so it can be applied to EITHER the engine's own
/// `pairs=head_dim/2` shape (the FULL check) or a narrower partial-rotary
/// shape (the PARTIAL check) via the `freq_dim`/`pairs` split. Positions run
/// `0..positions` (a single-shot forward with no prior KV cache, so absolute
/// position == row index). Channels `[0, pairs)` and `[pairs, 2*pairs)` are
/// rotated in place; channels at or beyond `2*pairs` are left untouched by
/// the caller (this function only returns the rotated `[0, 2*pairs)` prefix
/// plus a copy-through tail up to `head_dim`).
fn rope_split_half_ref(
    x: &[f32],
    positions: usize,
    kv_heads: usize,
    head_dim: usize,
    pairs: usize,
    base: f32,
    freq_dim: usize,
) -> Vec<f32> {
    let mut out = x.to_vec();
    for position in 0..positions {
        for head in 0..kv_heads {
            let row_base = (position * kv_heads + head) * head_dim;
            for pair in 0..pairs {
                let theta = position as f32 * base.powf(-((2 * pair) as f32) / freq_dim as f32);
                let (sin, cos) = theta.sin_cos();
                let first = x[row_base + pair];
                let second = x[row_base + pairs + pair];
                out[row_base + pair] = first * cos - second * sin;
                out[row_base + pairs + pair] = first * sin + second * cos;
            }
        }
    }
    out
}

/// [`report_diff`], plus a max_abs_diff split at `dim` = `head_dim` into a
/// "low" range `[0, split)` and a "high" range `[split, dim)` per row -- the
/// PARTIAL-rope hypothesis specifically predicts divergence concentrated in
/// ONE range and near-float-noise agreement in the other, which a single
/// whole-tensor max would hide.
fn report_diff_ranged(label: &str, engine: &[f32], reference: &[f32], dim: usize) {
    assert_eq!(engine.len(), reference.len(), "{label}: length mismatch");
    assert_eq!(
        engine.len() % dim,
        0,
        "{label}: length not a multiple of dim={dim}"
    );
    let split = dim / 4; // rotary_dim for the partial-rope hypothesis
    let mut max_abs_diff = 0f32;
    let mut max_low = 0f32;
    let mut max_high = 0f32;
    let mut worst_index = 0usize;
    for (index, (&engine_value, &reference_value)) in
        engine.iter().zip(reference.iter()).enumerate()
    {
        let diff = (engine_value - reference_value).abs();
        if diff > max_abs_diff {
            max_abs_diff = diff;
            worst_index = index;
        }
        let channel = index % dim;
        if channel < split {
            max_low = max_low.max(diff);
        } else {
            max_high = max_high.max(diff);
        }
    }
    println!(
        "A-vs-B {label}: max_abs_diff={max_abs_diff:e} worst_index={worst_index} engine={:.6} reference={:.6} | max_diff[dims 0..{split})={max_low:e} max_diff[dims {split}..{dim})={max_high:e}",
        engine[worst_index], reference[worst_index],
    );
}

/// [`rmsnorm_per_head_no_scale_ref`] plus the learned RAW gamma multiply --
/// gemma4's `q_norm`/`k_norm` under `bind_norm_plus_one` REMOVED (the
/// CURRENT tree state this probe targets): `Gemma4RMSNorm(head_dim, eps)`
/// with `with_scale=True` is `normed * weight` directly, no `1 + weight`
/// (that convention belongs to the OTHER `rmsnorm_ref` in this file, used
/// only for `output_norm`, a different learned-norm family). `gamma` here is
/// RAW bytes straight off `dequant_tensor` -- no `+1` applied.
fn rmsnorm_per_head_raw_gamma_ref(
    x: &[f32],
    gamma: &[f32],
    positions: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; positions * heads * head_dim];
    for position in 0..positions {
        for head in 0..heads {
            let base = (position * heads + head) * head_dim;
            let row = &x[base..base + head_dim];
            let mean_square = row.iter().map(|&value| value * value).sum::<f32>() / head_dim as f32;
            let inv_rms = 1.0 / (mean_square + eps).sqrt();
            for (local, &value) in row.iter().enumerate() {
                out[base + local] = value * inv_rms * gamma[local];
            }
        }
    }
    out
}

/// Reassembles a `[positions, heads, head_dim]` roped tensor from the
/// engine's own split even/odd rope taps -- the SAME reconstruction the
/// K-side diagnostic already does inline, generalized to `heads` so it also
/// serves the 16 query heads (`engine_rotated_q_even5`/`odd5`), not just the
/// 2 kv heads.
fn reconstruct_roped(
    even: &[f32],
    odd: &[f32],
    positions: usize,
    heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let pairs = head_dim / 2;
    let mut out = vec![0f32; positions * heads * head_dim];
    for position in 0..positions {
        for head in 0..heads {
            for pair in 0..pairs {
                let even_index = (position * heads + head) * pairs + pair;
                let base_offset = (position * heads + head) * head_dim;
                out[base_offset + pair] = even[even_index];
                out[base_offset + pairs + pair] = odd[even_index];
            }
        }
    }
    out
}

/// Independent GQA score contraction: `scores[s,t,u,g] = sum_d q[s,
/// group*u+g, d] * k[t, u, d]` -- `append_attention_mixer`'s own
/// `group_map = "s,{group}*u+g,i->sugi"` grouping (`attention_forward.rs:582`)
/// plus the `score_even`/`score_odd` reduce over `d`
/// (`attention_forward.rs:602-643`, here applied to the ALREADY-summed
/// even+odd roped vectors rather than the two half-dim partial sums the
/// engine keeps separate -- mathematically identical since
/// `sum_d(q*k) == sum_pairs(q_even*k_even) + sum_pairs(q_odd*k_odd)` for a
/// split-half layout). `q` is `[positions, query_heads, head_dim]`, `k` is
/// `[positions, kv_heads, head_dim]`; output is `[positions, positions,
/// kv_heads, group]` (`"stug"`, s slowest, g fastest -- the same row-major
/// convention every other letter-pattern in this file uses), unscaled
/// (`score_scale=1.0`, proven separately by the CHECK 2 factor probe above).
fn gqa_scores_ref(
    q: &[f32],
    k: &[f32],
    positions: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    group: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; positions * positions * kv_heads * group];
    for query_position in 0..positions {
        for key_position in 0..positions {
            for kv_head in 0..kv_heads {
                for group_index in 0..group {
                    let query_head = kv_head * group + group_index;
                    let q_base = (query_position * query_heads + query_head) * head_dim;
                    let k_base = (key_position * kv_heads + kv_head) * head_dim;
                    let mut sum = 0f32;
                    for dim in 0..head_dim {
                        sum += q[q_base + dim] * k[k_base + dim];
                    }
                    let out_index = ((query_position * positions + key_position) * kv_heads
                        + kv_head)
                        * group
                        + group_index;
                    out[out_index] = sum;
                }
            }
        }
    }
    out
}

/// Causal softmax over the `t` axis of a `[positions, positions, kv_heads,
/// group]` (`"stug"`) score tensor -- `key_position > query_position` masked
/// to `-inf` before the max-subtract/exp/normalize the engine's own
/// `score_max`/`shifted`/`weights`/`weight_sum`/`inv_weight_sum`/
/// `probabilities` chain performs (`attention_forward.rs:663-704`),
/// independently reproduced here directly from `f32` softmax math rather
/// than that op chain.
fn causal_softmax_ref(
    scores_scaled: &[f32],
    positions: usize,
    kv_heads: usize,
    group: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; positions * positions * kv_heads * group];
    for query_position in 0..positions {
        for kv_head in 0..kv_heads {
            for group_index in 0..group {
                let mut max_score = f32::NEG_INFINITY;
                for key_position in 0..=query_position {
                    let index = ((query_position * positions + key_position) * kv_heads
                        + kv_head)
                        * group
                        + group_index;
                    max_score = max_score.max(scores_scaled[index]);
                }
                let mut sum_exp = 0f32;
                for key_position in 0..=query_position {
                    let index = ((query_position * positions + key_position) * kv_heads
                        + kv_head)
                        * group
                        + group_index;
                    sum_exp += (scores_scaled[index] - max_score).exp();
                }
                for key_position in 0..positions {
                    let index = ((query_position * positions + key_position) * kv_heads
                        + kv_head)
                        * group
                        + group_index;
                    out[index] = if key_position <= query_position {
                        (scores_scaled[index] - max_score).exp() / sum_exp
                    } else {
                        0.0
                    };
                }
            }
        }
    }
    out
}

/// `attended[s, u, g, d] = sum_t probabilities[s, t, u, g] * v[t, u, d]` --
/// `append_attention_mixer`'s own `attended_product`/`attended` reduce
/// (`attention_forward.rs:706-720`), independently reproduced. Output flattens
/// `(u, g, d)` in that order (`u` outer, `d` innermost) -- the SAME order
/// `query_head = group*u + g` gives the flattened `query_head*head_dim + d`
/// layout `wo_product`'s own `"sugdo->sugdo"` pattern contracts against, so
/// this output can feed `matmul_rows` for the o_proj stage directly with no
/// re-layout.
fn attended_from_probabilities_ref(
    probabilities: &[f32],
    v: &[f32],
    positions: usize,
    kv_heads: usize,
    group: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; positions * kv_heads * group * head_dim];
    for query_position in 0..positions {
        for kv_head in 0..kv_heads {
            for group_index in 0..group {
                let out_base =
                    ((query_position * kv_heads + kv_head) * group + group_index) * head_dim;
                for key_position in 0..positions {
                    let probability_index = ((query_position * positions + key_position)
                        * kv_heads
                        + kv_head)
                        * group
                        + group_index;
                    let probability = probabilities[probability_index];
                    if probability == 0.0 {
                        continue;
                    }
                    let v_base = (key_position * kv_heads + kv_head) * head_dim;
                    for dim in 0..head_dim {
                        out[out_base + dim] += probability * v[v_base + dim];
                    }
                }
            }
        }
    }
    out
}
