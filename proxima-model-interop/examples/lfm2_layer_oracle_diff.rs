//! Depth-bisection of the real `LFM2.5-8B-A1B-Q4_K_M.gguf` forward pass
//! against `llama.cpp`'s own per-layer residual-stream dump (`l_out-<layer>`,
//! produced OUTSIDE this repo by the same `oracle-dump` probe
//! `lfm2_logit_oracle_diff.rs` uses) -- [`smollm2_layer_oracle_diff`]'s own
//! shape, reused rather than reinvented: this hybrid checkpoint's block
//! indices `{2,6,10,14,18,21}` are attention, the other 18 are
//! short-convolution, so the FIRST divergent layer index directly names
//! which subsystem (conv mixer / attention / MoE routing) is wrong, without
//! needing to inspect a single weight value first.
//!
//! [`lfm2_forward_values`]'s own doc explains the derivation:
//! [`lfm2_forward_program_with_experts`] built at a shorter `block_count`
//! against the SAME architecture shares every `NodeId` up to where it stops
//! (`NodeId`'s id-is-index invariant, `smollm2_layer_oracle_diff.rs`'s own
//! technique), so the last id two consecutive-depth throwaway builds still
//! agree on IS the residual-stream value right after that layer -- the exact
//! quantity `l_out-<layer>` names on the oracle side.

use std::env;
use std::fs;
use std::path::PathBuf;

use proxima_gguf::pipe::parse_complete;
use proxima_model_interop::{Lfm2Architecture, lfm2_architecture_from_metadata, lfm2_forward_values};
use proxima_telemetry::export::{Exporter, Formatter};
use proxima_telemetry::level::Level;
use proxima_telemetry::recorder::Recorder;
use proxima_tensor::op::NodeId;
use proxima_tensor::spec::lfm2_forward_program_with_experts;

fn read_oracle_activation(path: &PathBuf) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read oracle activation at {path:?}: {error}"));
    bytes.chunks_exact(4).map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])).collect()
}

/// The last `NodeId` a `depth`-deep throwaway program shares with a
/// `depth + 1`-deep one -- the residual-stream output right after layer
/// `depth - 1` -- [`smollm2_layer_oracle_diff.rs`]'s own
/// `layer_boundary_node_id`, adapted to this architecture's own program
/// builder (a per-layer `LayerKind` slice rather than a fixed dense shape).
///
/// Both builds are pure throwaways, never evaluated: only their `Op`
/// sequence's shared prefix matters, and
/// [`lfm2_forward_program_with_experts`]'s per-layer loop body (`spec.rs`)
/// depends only on that layer's own index and kind, never on the total
/// `block_count`/`layer_kinds` length it was called with -- so the deeper
/// build's imaginary extra layer can reuse ANY real `LayerKind` (this
/// function duplicates layer `depth - 1`'s own kind) without changing where
/// the two builds first diverge. That is what makes `depth == block_count`
/// safe here even though this checkpoint has no real `block_count + 1`th
/// layer to slice: the "deep" build's extra layer is a structural stand-in,
/// its weight names are never bound or evaluated.
fn layer_boundary_node_id(architecture: &Lfm2Architecture, depth: u32) -> NodeId {
    if depth == 0 {
        // `ids`, `token_embd.weight`, then the embedding gather itself --
        // always the 3rd op any depth of this program appends, matching
        // `lfm2_forward_program_with_experts`'s own opening three ops.
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

fn layer_kind_label(architecture: &Lfm2Architecture, layer: usize) -> &'static str {
    match architecture.layer_kinds[layer] {
        proxima_tensor::spec::LayerKind::Attention => "attention",
        proxima_tensor::spec::LayerKind::ShortConv => "shortconv",
    }
}

/// The residual stream's own per-layer noise floor on this real, Q4_K,
/// 24-layer, 2048-wide checkpoint -- MEASURED, not the smollm2 harness's own
/// `1e-1` (a much smaller f16 dense model's tolerance): layers 0-4 here
/// carry `2.4e-1..5.7e-1` of ordinary dequantization/summation-order noise,
/// so `1e-1` would flag EVERY layer as "gross" and hide the real signal --
/// the actual root-cause layer's jump to `4.6e1` (roughly two orders of
/// magnitude above the noise floor) is unmistakable at this threshold and
/// would remain so even at 10x the measured noise ceiling.
const GROSS_DIVERGENCE_THRESHOLD: f32 = 5.0;

fn main() {
    let model_path = env::args().nth(1).unwrap_or_else(|| {
        "/Users/brianbruggeman/.lmstudio/models/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf".to_string()
    });
    let oracle_dir = env::args().nth(2).unwrap_or_else(|| {
        "/private/tmp/claude-501/-Users-brianbruggeman-repos-slot-0/6cd9e134-c1a3-450a-be93-76dd95389bf4/scratchpad/oracle/dump_lfm2".to_string()
    });
    let prompt = env::args().nth(3).unwrap_or_else(|| "The capital of France is".to_string());
    let log_path = env::args().nth(4).unwrap_or_else(|| "lfm2_layer_diff.jsonl".to_string());

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

    let recorder = Recorder::builder().export(Exporter::file(&log_path).format(Formatter::Text)).expect("file exporter").install().expect("recorder");

    let file_bytes = fs::read(&model_path).expect("read lfm2 gguf checkpoint");
    let parsed = parse_complete(&file_bytes).expect("parse lfm2 gguf checkpoint");
    let architecture: Lfm2Architecture = lfm2_architecture_from_metadata(&parsed).expect("derive lfm2 architecture from gguf metadata");

    let vocab = proxima_tokenizer::gguf::vocab_from_metadata(&parsed).expect("build vocab from gguf metadata");
    let add_bos = vocab.add_bos_token().unwrap_or(true);
    let ids = proxima_tokenizer::encode_with_bos_eos(&prompt, &vocab, add_bos, false).expect("tokenize prompt");

    let block_count = architecture.block_count;
    let mut node_ids = Vec::with_capacity(block_count as usize);
    let mut labels: Vec<String> = Vec::with_capacity(block_count as usize);
    for layer in 0..block_count {
        node_ids.push(layer_boundary_node_id(&architecture, layer + 1));
        labels.push(format!("l_out-{layer}"));
    }

    let (_logits, activations) = lfm2_forward_values(&parsed, &file_bytes, &architecture, &ids, &node_ids).expect("compute our own layer activations");

    println!("label kind node_id max_abs_diff worst_index ours_value theirs_value");
    let mut first_gross_divergence: Option<(String, &'static str)> = None;
    for (layer, (label, values)) in labels.iter().zip(activations.iter()).enumerate() {
        let oracle_path = oracle_dir.join(format!("{label}.f32"));
        if !oracle_path.exists() {
            println!("{label} MISSING_ORACLE_FILE");
            continue;
        }
        let theirs = read_oracle_activation(&oracle_path);
        // the real checkpoint's own last layer: llama.cpp's `inp_out_ids`
        // reduces `l_out-{block_count - 1}` to only the requested output
        // positions (a single-sequence prefill's last token, here) before
        // dumping it, so that one label's oracle file is one position wide
        // while ours still carries every position -- compare only the last
        // position's slice in that case, never silently truncate any other
        // label's mismatch into a false pass.
        let ours: &[f32] = if theirs.len() == values.len() {
            values.as_slice()
        } else if theirs.len() < values.len() && values.len() % theirs.len() == 0 {
            &values[values.len() - theirs.len()..]
        } else {
            panic!("{label}: element count mismatch: ours={} theirs={}", values.len(), theirs.len());
        };

        let mut max_abs_diff = 0f32;
        let mut worst_index = 0usize;
        for index in 0..ours.len() {
            let diff = (ours[index] - theirs[index]).abs();
            if diff > max_abs_diff {
                max_abs_diff = diff;
                worst_index = index;
            }
        }
        let kind = layer_kind_label(&architecture, layer);
        println!("{label} {kind} node={} {:e} {worst_index} {:.6} {:.6}", node_ids[layer].0, max_abs_diff, ours[worst_index], theirs[worst_index]);

        let level = if max_abs_diff > GROSS_DIVERGENCE_THRESHOLD { Level::WARN } else { Level::DEBUG };
        let layer_label: &'static str = Box::leak(label.clone().into_boxed_str());
        recorder
            .log()
            .level(level)
            .message("lfm2 layer activation compared against oracle")
            .module_path(module_path!())
            .tag("layer", layer_label)
            .tag("kind", kind)
            .tag("max_abs_diff", f64::from(max_abs_diff))
            .tag("worst_index", worst_index as u64)
            .emit();

        if max_abs_diff > GROSS_DIVERGENCE_THRESHOLD && first_gross_divergence.is_none() {
            first_gross_divergence = Some((label.clone(), kind));
        }
    }
    while recorder.drain() > 0 {}

    match first_gross_divergence {
        Some((label, kind)) => println!("\nfirst gross divergence at: {label} (kind={kind})"),
        None => println!("\nno gross divergence found at any dumped layer"),
    }
}
