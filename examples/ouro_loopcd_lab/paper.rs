use std::fs::{self, File};
use std::io::{self, Write};

use serde_json::{Value, json};

use super::checkpoint::{Checkpoint, WeightMappingMode};
use super::guidance::Guidance;
use super::model::{Arm, Model, pick};
use super::runner::{Arguments, record};

pub fn run(arguments: &Arguments, writer: &mut impl Write) -> io::Result<()> {
    let guidance = arguments
        .guidance
        .ok_or_else(|| io::Error::other("selected guidance absent"))?;
    let mut prefix = arguments.prefix.clone();
    let mut generated = Vec::new();
    let mut forward_count = 0;
    let mut stop_reason = "token_budget";
    if arguments.max_new_tokens != 0 {
        let mapping_mode =
            WeightMappingMode::for_run(&arguments.backend, arguments.metal_copy_weights);
        let checkpoint = Checkpoint::open(&arguments.directory, mapping_mode)?;
        for value in &checkpoint.inventory {
            record(writer, value)?;
        }
        let config: Value =
            serde_json::from_slice(&fs::read(arguments.directory.join("config.json"))?)
                .map_err(io::Error::other)?;
        let eos_token = config["eos_token_id"]
            .as_u64()
            .and_then(|identifier| u32::try_from(identifier).ok())
            .ok_or_else(|| io::Error::other("checkpoint EOS token absent"))?;
        for step in 0..arguments.max_new_tokens {
            if prefix.len() > 65536 {
                return Err(io::Error::other("generated prefix exceeds context bound"));
            }
            let mut model = Model::prepare(&checkpoint, prefix.clone(), &arguments.backend)?;
            let arm = Arm {
                name: guidance.name(),
                depth: arguments.depth,
                guidance,
            };
            let output = model.forward_readouts(arm, false)?;
            forward_count += 1;
            let selected = pick(&output.logits)?;
            let hidden_mode = matches!(guidance, Guidance::FixedHidden { .. });
            let terminal = (!hidden_mode).then_some(&output.head_logits);
            let adaptive = output.adaptive.map(|value| {
                json!({
                    "top_probability":value.top_probability,
                    "second_probability":value.second_probability,
                    "margin":value.margin,
                    "effective_strength":value.effective_strength,
                    "probability_source":"unguided terminal full-vocabulary stable softmax",
                    "independent_confidence":false,
                })
            });
            record(
                writer,
                &json!({
                    "kind":"paper_selected_output", "step":step, "method":guidance.name(),
                    "context":model.context(arm), "prefix_ids":prefix, "depth":arguments.depth,
                    "strength":guidance.strength(), "effective_strength":output.effective_strength,
                    "reference_logits":output.reference, "terminal_logits":terminal,
                    "guided_logits":output.logits, "vocabulary_length":checkpoint.shape.vocab,
                    "raw_reference_hidden":output.raw_reference_hidden,
                    "raw_terminal_hidden":output.raw_terminal_hidden,
                    "contrasted_hidden":output.contrasted_hidden,
                    "normalized_hidden":output.normalized_hidden,
                    "hidden_width":checkpoint.shape.hidden,
                    "hidden_boundary":hidden_mode.then_some("completed iteration before model.norm.weight; last token"),
                    "hidden_contrast_scope":hidden_mode.then_some("complete prefix, row-independent contrast and normalization"),
                    "adaptive":adaptive,
                    "selected_id":selected, "selected_spelling":checkpoint.vocab.token_str(selected),
                    "selected_bytes":checkpoint.vocab.token_bytes(selected),
                    "head_executions":output.head_executions, "layer_executions":output.layer_executions,
                    "branch_norm_sites":output.branch_norm_sites, "loop_norm_sites":output.loop_norm_sites,
                    "diagnostic_extra_head_executions":0, "forward_count":forward_count,
                    "cache":false, "execution":"complete-prefix recomputation",
                    "cached_serving_performance":"unmeasured", "selected_is_eos":selected == eos_token,
                }),
            )?;
            prefix.push(selected);
            generated.push(selected);
            writer.flush()?;
            if selected == eos_token {
                stop_reason = "eos";
                break;
            }
        }
    }
    let summary = json!({
        "kind":"paper_generation_summary", "method":guidance.name(), "depth":arguments.depth,
        "strength":guidance.strength(), "initial_prefix_ids":arguments.prefix,
        "final_prefix_ids":prefix, "generated_ids":generated,
        "requested_max_new_tokens":arguments.max_new_tokens, "generated_token_count":generated.len(),
        "forward_count":forward_count, "stop_reason":stop_reason,
        "cache":false, "execution":"complete-prefix recomputation",
        "cached_serving_performance":"unmeasured", "quality_status":"unmeasured",
    });
    record(writer, &summary)?;
    serde_json::to_writer_pretty(
        File::create(arguments.output.join("summary.json"))?,
        &summary,
    )
    .map_err(io::Error::other)?;
    writer.flush()
}
