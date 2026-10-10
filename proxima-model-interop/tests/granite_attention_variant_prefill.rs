#![cfg(all(
    feature = "metal-attn-variants",
    feature = "instrument",
    target_os = "macos"
))]

use std::env;
use std::fs::{self, File};
use std::io::ErrorKind;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use conflaguration::Settings;
use memmap2::Mmap;
use omega::metal::MetalError;
use omega::msl::Binding;
use omega::{
    AttentionKvReuse, AttentionSimdgroupCount, AttentionTileHeight, AttentionVariant, CapturedDispatch,
    set_capture_step, take_captured_dispatches,
};
use proxima_gguf::parse_complete;
use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::types::GgmlType;
use proxima_model_interop::{
    AttentionTileHeightSetting, GPU_LAYERS_ALL, InteropError, LoadedModel, PromptCacheConfig,
    ServingConfig, ServingSettings,
    SpeculativeConfig,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

#[path = "../examples/cell_resources/cell.rs"]
mod cell_resources;

const GRANITE_MOE_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80";
const GRANITE_MOE_ENV: &str = "PROXIMA_ARCH_GRANITE_MOE_GGUF";
const GRANITE_EXPECTED_SHA256: &str =
    "cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80";
const PROMPT_TOKENS: usize = 971;
const SHORT_PROMPT_TOKENS: usize = 256;
const PASSAGE: &str = "To Sherlock Holmes she is always THE woman. I have seldom heard him mention her under any other name. In his eyes she eclipses and predominates the whole of her sex. It was not that he felt any emotion akin to love for Irene Adler. All emotions, and that one particularly, were abhorrent to his cold, precise but admirably balanced mind. He was, I take it, the most perfect reasoning and observing machine that the world has seen, but as a lover he would have placed himself in a false position. He never spoke of the softer passions, save with a gibe and a sneer. They were admirable things for the observer—excellent for drawing the veil from men's motives and actions. But for the trained reasoner to admit such intrusions into his own delicate and finely adjusted temperament was to introduce a distracting factor which might throw a doubt upon all his mental results.\n";

#[derive(Debug, Clone, PartialEq, Eq)]
struct DispatchIdentity {
    node: u32,
    extents: Vec<u64>,
    entry: String,
    source_sha: String,
    grid: omega::msl::GridSpec,
}

struct CapturedReplay {
    dispatch: CapturedDispatch,
    output: Vec<u8>,
    token_ids: Vec<u32>,
}

#[derive(Clone)]
struct ReplayEvidence {
    identity: DispatchIdentity,
    bindings: Vec<Binding>,
    output: Vec<u8>,
}

fn identity(dispatch: &CapturedDispatch) -> DispatchIdentity {
    DispatchIdentity {
        node: dispatch.node,
        extents: dispatch.extents.clone(),
        entry: dispatch.entry.clone(),
        source_sha: dispatch.msl_sha256.clone(),
        grid: dispatch.grid,
    }
}

fn assert_selected_dispatch_pair(
    legacy: &[DispatchIdentity],
    selected: &[DispatchIdentity],
) -> Result<usize, String> {
    let mut matched = 0;
    for legacy_dispatch in legacy {
        let Some(selected_dispatch) = selected.iter().find(|dispatch| {
            dispatch.node == legacy_dispatch.node && dispatch.extents == legacy_dispatch.extents
        }) else {
            continue;
        };
        if legacy_dispatch.extents.len() == 4
            && legacy_dispatch.extents[0] > 1
            && legacy_dispatch.extents[1..] == [8, 2, 64]
        {
            if legacy_dispatch.entry == selected_dispatch.entry {
                return Err(format!(
                    "node {} extents {:?} kept legacy entry {}",
                    legacy_dispatch.node, legacy_dispatch.extents, legacy_dispatch.entry
                ));
            }
            if legacy_dispatch.source_sha == selected_dispatch.source_sha {
                return Err(format!(
                    "node {} extents {:?} kept legacy source SHA {}",
                    legacy_dispatch.node, legacy_dispatch.extents, legacy_dispatch.source_sha
                ));
            }
            if selected_dispatch.grid.threads == 0 {
                return Err(format!(
                    "node {} selected an empty dispatch grid",
                    selected_dispatch.node
                ));
            }
            println!(
                "card_23 dispatch node={} extents={:?} legacy_entry={} selected_entry={} legacy_source_sha={} selected_source_sha={} grid={:?}",
                selected_dispatch.node,
                selected_dispatch.extents,
                legacy_dispatch.entry,
                selected_dispatch.entry,
                legacy_dispatch.source_sha,
                selected_dispatch.source_sha,
                selected_dispatch.grid
            );
            matched += 1;
        }
    }
    if matched == 0 {
        return Err("no matching multi-row Granite cached-attention dispatch pair".to_string());
    }
    Ok(matched)
}

fn serving_config(attention_variant: Option<AttentionVariant>) -> ServingConfig<'static> {
    ServingConfig {
        attention_variant,
        gpu_layers: GPU_LAYERS_ALL,
        kv_cache_key_quant: GgmlType::F32,
        kv_cache_value_quant: GgmlType::F32,
        flash_attention: false,
        reasoning_budget: 0,
        prompt_cache: PromptCacheConfig::off(),
        speculative: SpeculativeConfig::none(),
        ubatch_size: 0,
        ..ServingConfig::default()
    }
}

fn granite_checkpoint_path() -> String {
    let path = std::env::var(GRANITE_MOE_ENV).unwrap_or_else(|_| GRANITE_MOE_PATH.to_string());
    assert!(
        Path::new(&path).exists(),
        "Granite checkpoint is missing at {path}"
    );
    path
}

fn verify_granite_checkpoint(parsed: &ParsedGguf, mapping: &[u8]) -> (String, String) {
    let architecture = parsed
        .metadata_value("general.architecture")
        .and_then(|value| value.as_str())
        .expect("Granite checkpoint declares general.architecture");
    assert_eq!(architecture, "granitemoe", "checkpoint architecture");
    let model_name = parsed
        .metadata_value("general.name")
        .and_then(|value| value.as_str())
        .expect("Granite checkpoint declares general.name");
    assert_eq!(
        model_name, "Granite 3.1 1b A400M Instruct",
        "checkpoint model name"
    );
    let file_type = parsed
        .metadata_value("general.file_type")
        .and_then(|value| value.as_u32())
        .expect("Granite checkpoint declares general.file_type");
    assert_eq!(file_type, 7, "GGUF MOSTLY_Q8_0 file type");
    let checkpoint_sha256 = output_sha256(mapping);
    assert_eq!(
        checkpoint_sha256, GRANITE_EXPECTED_SHA256,
        "checkpoint content SHA256"
    );
    (model_name.to_string(), checkpoint_sha256)
}

fn clear_previous_report(path: &Path, checkpoint_path: &Path) {
    let report_path = if path.exists() {
        fs::canonicalize(path).expect("resolve previous Granite report path")
    } else {
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::canonicalize(parent)
            .expect("resolve Granite report directory")
            .join(path.file_name().expect("report path has a filename"))
    };
    let checkpoint_path =
        fs::canonicalize(checkpoint_path).expect("resolve Granite checkpoint path");
    assert_ne!(
        report_path, checkpoint_path,
        "report destination must not name the Granite checkpoint"
    );
    if path.exists() {
        let report_metadata = fs::metadata(path).expect("read previous Granite report metadata");
        let checkpoint_metadata =
            fs::metadata(&checkpoint_path).expect("read Granite checkpoint metadata");
        assert!(
            report_metadata.dev() != checkpoint_metadata.dev()
                || report_metadata.ino() != checkpoint_metadata.ino(),
            "report destination must not alias the Granite checkpoint"
        );
    }
    match fs::remove_file(path) {
        Ok(()) => {}
        Err(error) if error.kind() == ErrorKind::NotFound => {}
        Err(error) => panic!("cannot retire previous report {}: {error}", path.display()),
    }
}

fn prompt_of_tokens(vocab: &proxima_tokenizer::Vocab, target_tokens: usize) -> (String, usize) {
    let corpus = PASSAGE.repeat(target_tokens.div_ceil(40));
    let mut prompt = String::new();
    for word in corpus.split_inclusive(char::is_whitespace) {
        prompt.push_str(word);
        let token_count = proxima_tokenizer::encode(&prompt, vocab)
            .expect("tokenize prompt with the Granite vocab")
            .len();
        if token_count >= target_tokens {
            return (prompt, token_count);
        }
    }
    panic!("repeated Sherlock passage never reached {target_tokens} tokens");
}

fn prompt_prefix_of_tokens(
    vocab: &proxima_tokenizer::Vocab,
    target_tokens: usize,
) -> (String, usize) {
    let corpus = PASSAGE.repeat(target_tokens.div_ceil(40));
    let mut prompt = String::new();
    for word in corpus.split_inclusive(char::is_whitespace) {
        prompt.push_str(word);
        let candidate = prompt.trim_end_matches(char::is_whitespace);
        let token_count = proxima_tokenizer::encode(candidate, vocab)
            .expect("tokenize the word-bounded Granite prefix")
            .len();
        if token_count >= target_tokens {
            return (candidate.to_string(), token_count);
        }
    }
    panic!("repeated Sherlock passage never reached {target_tokens} tokens");
}

fn capture_prompt_dispatches(
    model: &LoadedModel<'_>,
    prompt: &str,
    variant: Option<AttentionVariant>,
) -> (Vec<DispatchIdentity>, Vec<u32>) {
    let (records, token_ids) = capture_prompt_records(model, prompt, variant);
    let identities = records
        .iter()
        .filter(|record| record.kind_name == "cached_attention")
        .map(identity)
        .collect();
    (identities, token_ids)
}

fn capture_prompt_records(
    model: &LoadedModel<'_>,
    prompt: &str,
    variant: Option<AttentionVariant>,
) -> (Vec<CapturedDispatch>, Vec<u32>) {
    let _ = take_captured_dispatches();
    set_capture_step(0);
    let (token_ids, _text, _stopped) = model
        .generate_with_serving_config(prompt, 1, serving_config(variant))
        .expect("Granite prefill and one-token forward run");
    let records = take_captured_dispatches();
    (records, token_ids)
}

fn capture_attention_replay(
    model: &LoadedModel<'_>,
    prompt: &str,
    variant: Option<AttentionVariant>,
    match_identity: Option<&CapturedDispatch>,
) -> CapturedReplay {
    let (records, token_ids) = capture_prompt_records(model, prompt, variant);
    let mut attention_records = records
        .into_iter()
        .filter(|record| record.kind_name == "cached_attention")
        .collect::<Vec<_>>();
    attention_records
        .sort_by(|left, right| (left.node, &left.extents).cmp(&(right.node, &right.extents)));
    let matching_records = attention_records
        .into_iter()
        .filter(|record| {
            record.extents.len() == 4
                && record.extents[0] > 1
                && record.extents[1..] == [8, 2, 64]
                && match_identity.is_none_or(|identity| {
                    record.node == identity.node && record.extents == identity.extents
                })
        })
        .collect::<Vec<_>>();
    if match_identity.is_some() {
        assert_eq!(
            matching_records.len(),
            1,
            "selected request must capture exactly one record matching legacy node and extents"
        );
    }
    let dispatch = matching_records
        .into_iter()
        .next()
        .expect("request captured a multi-row cached-attention dispatch");
    assert!(
        !dispatch
            .bindings
            .iter()
            .any(|binding| matches!(binding, Binding::Fault)),
        "selected cached-attention dispatch contains a fault binding"
    );
    assert!(
        dispatch.grid.threads > 0,
        "captured attention grid is empty"
    );
    assert!(
        dispatch.unreplayable.is_none(),
        "captured attention is unreplayable"
    );
    let expected_bytes = dispatch.extents.iter().product::<u64>() as usize * 4;
    let output = dispatch
        .replay_output_elements(Some(dispatch.extents.iter().product()))
        .expect("replay the complete captured attention output span");
    assert_eq!(
        output.len(),
        expected_bytes,
        "replay output span is incomplete"
    );
    assert!(
        output.chunks_exact(4).any(|word| word != [0x55; 4]),
        "replayed attention output contains only poison bytes"
    );
    CapturedReplay {
        dispatch,
        output,
        token_ids,
    }
}

fn replay_evidence(replay: &CapturedReplay) -> ReplayEvidence {
    ReplayEvidence {
        identity: identity(&replay.dispatch),
        bindings: replay.dispatch.bindings.clone(),
        output: replay.output.clone(),
    }
}

#[must_use]
fn compare_replay_pair(
    legacy: Option<&ReplayEvidence>,
    selected: Option<&ReplayEvidence>,
) -> Result<(), String> {
    let legacy = legacy.ok_or_else(|| "legacy dispatch record is missing".to_string())?;
    let selected = selected.ok_or_else(|| "selected dispatch record is missing".to_string())?;
    for (arm, evidence) in [("legacy", legacy), ("selected", selected)] {
        if evidence
            .bindings
            .iter()
            .any(|binding| matches!(binding, Binding::Fault))
        {
            return Err(format!("{arm} dispatch contains a fault binding"));
        }
    }
    if legacy.identity.node != selected.identity.node
        || legacy.identity.extents != selected.identity.extents
    {
        return Err("captured dispatch node or extents differ".to_string());
    }
    if legacy.identity.entry == selected.identity.entry {
        return Err("selected dispatch kept the legacy entry".to_string());
    }
    if legacy.identity.source_sha == selected.identity.source_sha {
        return Err("selected dispatch kept the legacy source SHA".to_string());
    }
    if legacy.output.is_empty() || legacy.output.len() != selected.output.len() {
        return Err("captured output spans are empty or have different lengths".to_string());
    }
    if let Some((offset, (expected, actual))) = legacy
        .output
        .iter()
        .zip(&selected.output)
        .enumerate()
        .find(|(_, (expected, actual))| expected != actual)
    {
        return Err(format!(
            "output byte {offset} differs: legacy={expected:#04x} selected={actual:#04x}"
        ));
    }
    Ok(())
}

fn compare_token_ids(expected: &[u32], actual: &[u32]) -> Result<(), String> {
    if expected.len() != actual.len() {
        return Err(format!(
            "token count differs: expected {}, received {}",
            expected.len(),
            actual.len()
        ));
    }
    for (index, (expected_id, actual_id)) in expected.iter().zip(actual).enumerate() {
        if expected_id != actual_id {
            return Err(format!(
                "token {index} differs: expected {expected_id}, received {actual_id}"
            ));
        }
    }
    Ok(())
}

#[must_use]
fn nearest_rank(sorted_samples: &[f64], percentile: f64) -> f64 {
    let rank = (percentile * sorted_samples.len() as f64).ceil() as usize;
    sorted_samples[rank - 1]
}

#[must_use]
fn summarize_samples(samples: &[f64]) -> Value {
    let mut sorted_samples = samples.to_vec();
    sorted_samples.sort_by(f64::total_cmp);
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let population_variance = samples
        .iter()
        .map(|sample| (sample - mean).powi(2))
        .sum::<f64>()
        / samples.len() as f64;
    let cov_percent = 100.0 * population_variance.sqrt() / mean;
    json!({
        "count": samples.len(),
        "min_gpu_ns": sorted_samples[0],
        "p50_gpu_ns": nearest_rank(&sorted_samples, 0.50),
        "p90_gpu_ns": nearest_rank(&sorted_samples, 0.90),
        "p99_gpu_ns": nearest_rank(&sorted_samples, 0.99),
        "max_gpu_ns": sorted_samples[sorted_samples.len() - 1],
        "mean_gpu_ns": mean,
        "cov_percent": cov_percent,
    })
}

fn resource_replays(dispatch: &CapturedDispatch, label: &str) -> String {
    let cell = cell_resources::Cell::begin();
    for _ in 0..5 {
        dispatch
            .time_gpu_ns(1)
            .expect("resource-context replay completes");
    }
    cell.end(label)
}

fn assert_active_grid_dimensions(dispatch: &CapturedDispatch) {
    assert!(
        dispatch.grid.threads > 0,
        "captured attention grid is empty"
    );
    assert!(
        dispatch.grid.depth > 0,
        "captured attention grid depth is zero"
    );
    assert!(
        dispatch
            .grid
            .threadgroup_width
            .is_none_or(|width| width > 0),
        "captured attention threadgroup width is zero"
    );
    assert!(
        dispatch.grid.grid2d.is_none_or(|grid| {
            grid.threadgroups_x > 0
                && grid.threadgroups_y > 0
                && grid.threads_per_threadgroup_x > 0
                && grid.threads_per_threadgroup_y > 0
        }),
        "captured attention 2D grid contains a zero dimension"
    );
}

#[must_use]
fn prompt_ids_sha256(token_ids: &[u32]) -> String {
    let mut digest = Sha256::new();
    for token_id in token_ids {
        digest.update(token_id.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

#[must_use]
fn output_sha256(output: &[u8]) -> String {
    format!("{:x}", Sha256::digest(output))
}

fn measured_arm_report(
    arm_name: &str,
    replay: &CapturedReplay,
    samples: Vec<Value>,
    resource: String,
) -> Value {
    let (tg_static_bytes, max_threads, exec_width) = replay.dispatch.pipeline_resources();
    let grid2d = replay.dispatch.grid.grid2d.map(|grid| {
        json!({
            "form": match grid.form {
                omega::msl::Grid2DForm::TileCoordinates => "tile_coordinates",
                omega::msl::Grid2DForm::FlatThreadgroupIndex => "flat_threadgroup_index",
            },
            "threadgroups_x": grid.threadgroups_x,
            "threadgroups_y": grid.threadgroups_y,
            "threads_per_threadgroup_x": grid.threads_per_threadgroup_x,
            "threads_per_threadgroup_y": grid.threads_per_threadgroup_y,
            "threadgroup_bytes": grid.threadgroup_bytes,
        })
    });
    let numeric_samples = samples
        .iter()
        .map(|sample| sample["gpu_ns"].as_f64().expect("sample time is numeric"))
        .collect::<Vec<_>>();
    let summary = summarize_samples(&numeric_samples);
    json!({
        "arm": arm_name,
        "node": replay.dispatch.node,
        "extents": replay.dispatch.extents,
        "entry": replay.dispatch.entry,
        "source_sha256": replay.dispatch.msl_sha256,
        "grid": {
            "threads": replay.dispatch.grid.threads,
            "threadgroup_width": replay.dispatch.grid.threadgroup_width,
            "depth": replay.dispatch.grid.depth,
            "grid2d": grid2d,
        },
        "pipeline_resources": {
            "tg_static_bytes": tg_static_bytes,
            "max_threads": max_threads,
            "exec_width": exec_width,
        },
        "bound_bytes": replay.dispatch.bound_buffer_bytes(),
        "fault_binding_present": replay.dispatch.bindings.iter().any(|binding| matches!(binding, Binding::Fault)),
        "output_bytes": replay.output.len(),
        "output_sha256": output_sha256(&replay.output),
        "generated_ids": replay.token_ids,
        "samples": samples,
        "summary": summary,
        "timing_attempts": 20,
        "resource_replay_attempts": 5,
        "replay_errors": 0,
        "resource": resource,
    })
}

fn write_report(path: &Path, report: &Value) {
    let serialized =
        serde_json::to_vec_pretty(report).expect("serialize Granite attention replay report");
    fs::write(path, serialized).expect("write Granite attention replay report");
}

fn route_mismatch_node(error: &InteropError) -> u32 {
    match error {
        InteropError::Metal(MetalError::RouteCompactionMismatch { node, .. }) => node.0,
        _ => u32::MAX,
    }
}

fn captured_route_prepass_summary(node: u32) -> String {
    take_captured_dispatches()
        .into_iter()
        .filter(|dispatch| dispatch.node == node)
        .map(|dispatch| {
            let route_inputs = dispatch
                .bindings
                .iter()
                .enumerate()
                .filter(|(_, binding)| matches!(binding, Binding::Indices(_)))
                .map(|(index, binding)| {
                    let head = dispatch
                        .bound_buffer_bytes_at(index)
                        .map(|bytes| {
                            bytes
                                .get(..bytes.len().min(32))
                                .unwrap_or_default()
                                .chunks_exact(4)
                                .map(|word| {
                                    f32::from_ne_bytes([word[0], word[1], word[2], word[3]])
                                })
                                .collect::<Vec<_>>()
                        });
                    format!("({index}, {binding:?}, {head:?})")
                })
                .collect::<Vec<_>>();
            let live_header = dispatch
                .bound_buffer_bytes_at(dispatch.bindings.len())
                .and_then(|bytes| {
                    bytes.get(..4).map(|header| {
                        u32::from_ne_bytes([header[0], header[1], header[2], header[3]])
                    })
                });
            let compaction_bytes = dispatch
                .bound_buffer_bytes_at(dispatch.bindings.len())
                .map(|bytes| bytes.len());
            let compact_head = dispatch
                .bound_buffer_bytes_at(dispatch.bindings.len())
                .map(|bytes| {
                    bytes
                        .get(..bytes.len().min(128))
                        .unwrap_or_default()
                        .chunks_exact(4)
                        .map(|word| u32::from_ne_bytes([word[0], word[1], word[2], word[3]]))
                        .collect::<Vec<_>>()
                });
            format!(
                "(node={}, step={}, chunk={}, entry={}, sha={}, bindings={:?}, prepass_dispatches={}, prepass_grid={:?}, max_threads={:?}, owner={:?}, compaction_bytes={compaction_bytes:?}, route_inputs={route_inputs:?}, uniforms_head={:?}, live_header={live_header:?}, compact_head={compact_head:?}, unreplayable={:?})",
                dispatch.node,
                dispatch.step,
                dispatch.chunk_index,
                dispatch.entry,
                dispatch.msl_sha256,
                dispatch.bindings,
                dispatch.route_prepass_dispatches,
                dispatch.route_prepass_grid,
                dispatch.route_prepass_max_threads,
                dispatch.prepass_owner,
                dispatch.uniform_bytes.get(..dispatch.uniform_bytes.len().min(32)),
                dispatch.unreplayable,
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn require_expected_answer(response: &str, expected_fact: &str) -> Result<(), String> {
    let normalized = response.to_lowercase();
    let expected_fact = expected_fact.to_lowercase();
    let has_negation =
        normalized.contains(" not ") || normalized.contains("n't") || normalized.contains("never");
    if normalized.contains(&expected_fact) && !has_negation {
        Ok(())
    } else {
        Err(format!(
            "response does not contain {expected_fact:?}: {response:?}"
        ))
    }
}

fn require_comparison_answer(
    response: &str,
    expected_answer: &str,
    expected_relation: &str,
    comparison_term: &str,
) -> Result<(), String> {
    let normalized = response.to_lowercase();
    let expected_answer = expected_answer.to_lowercase();
    let expected_relation = expected_relation.to_lowercase();
    let comparison_term = comparison_term.to_lowercase();
    let opposite_relation = if expected_relation == "bigger" {
        "smaller"
    } else {
        "bigger"
    };
    let expected_relation_present = normalized.contains(&expected_relation)
        || (expected_relation == "bigger" && normalized.contains("larger"));
    let opposite_relation_present = normalized.contains(opposite_relation)
        || (opposite_relation == "bigger" && normalized.contains("larger"));
    let answer_is_subject = normalized.contains(&format!("{expected_answer} is"));
    let comparison_is_subject = normalized.contains(&format!("{comparison_term} is"));
    let has_negation =
        normalized.contains(" not ") || normalized.contains("n't") || normalized.contains("never");
    if normalized.contains(&expected_answer)
        && normalized.contains(&comparison_term)
        && expected_relation_present
        && !opposite_relation_present
        && answer_is_subject
        && !comparison_is_subject
        && !has_negation
    {
        Ok(())
    } else {
        Err(format!(
            "response does not affirm {expected_answer} as {expected_relation} than {comparison_term}: {response:?}"
        ))
    }
}

fn granite_chat_prompt(user_turn: &str) -> String {
    format!("<|turn>user\n{user_turn}<turn|>\n<|turn>model\n")
}

struct ViabilityCheck {
    name: &'static str,
    prompt: String,
    expected_answer: &'static str,
    expected_relation: Option<(&'static str, &'static str)>,
}

fn granite_viability_checks() -> Vec<ViabilityCheck> {
    vec![
        ViabilityCheck {
            name: "paris",
            prompt: granite_chat_prompt("The capital of France is"),
            expected_answer: "paris",
            expected_relation: None,
        },
        ViabilityCheck {
            name: "soliloquy",
            prompt: granite_chat_prompt(
                "In drama, what is a speech in which a character, alone on stage, speaks their inner thoughts aloud called?",
            ),
            expected_answer: "soliloquy",
            expected_relation: None,
        },
        ViabilityCheck {
            name: "ant_vs_briefcase",
            prompt: granite_chat_prompt("Which is bigger, an ant or a briefcase?"),
            expected_answer: "briefcase",
            expected_relation: Some(("bigger", "ant")),
        },
        ViabilityCheck {
            name: "hippo_vs_building",
            prompt: granite_chat_prompt(
                "Which of these is smaller in size: a hippopotamus or a large office building?",
            ),
            expected_answer: "hippopotamus",
            expected_relation: Some(("smaller", "building")),
        },
    ]
}

fn granite_long_prompt_oracle() -> (String, Vec<u32>, Vec<u32>) {
    let fixture_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/llama-parity/granite_moe/long_prompt_llama_ids.json");
    let fixture_text = std::fs::read_to_string(&fixture_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", fixture_path.display()));
    let records: Vec<serde_json::Value> =
        serde_json::from_str(&fixture_text).expect("parse Granite llama prompt fixture");
    assert_eq!(records.len(), 1, "fixture must contain exactly one case");
    let record = &records[0];
    let prompt = record["prompt"]
        .as_str()
        .expect("fixture case has a prompt")
        .to_string();
    let ids = |field: &str| {
        record[field]
            .as_array()
            .unwrap_or_else(|| panic!("fixture field {field} must be an array"))
            .iter()
            .map(|value| {
                u32::try_from(
                    value
                        .as_u64()
                        .expect("fixture token ids must be unsigned integers"),
                )
                .expect("fixture token id fits u32")
            })
            .collect::<Vec<_>>()
    };
    (prompt, ids("prompt_ids"), ids("generated_ids"))
}

fn run_granite_viability_case(check: ViabilityCheck) {
    let path = granite_checkpoint_path();
    let file = File::open(&path).expect("open the real Granite checkpoint");
    // SAFETY: this test only reads the checkpoint and does not mutate it.
    let mapping = unsafe { Mmap::map(&file) }.expect("mmap the Granite checkpoint");
    let parsed = parse_complete(&mapping).expect("parse the Granite checkpoint");
    let model = LoadedModel::load(&parsed, &mapping)
        .unwrap_or_else(|error| panic!("bind Granite for {}: {error:?}", check.name));
    let (legacy_ids, legacy_text, _) = model
        .generate_with_serving_config(&check.prompt, 48, serving_config(None))
        .unwrap_or_else(|error| panic!("legacy {} request failed: {error:?}", check.name));
    let (selected_ids, selected_text, _) = model
        .generate_with_serving_config(&check.prompt, 48, serving_config(Some(shared_k_variant())))
        .unwrap_or_else(|error| panic!("selected {} request failed: {error:?}", check.name));
    assert!(
        !legacy_ids.is_empty() && !selected_ids.is_empty(),
        "{} returned empty token IDs; legacy={legacy_ids:?}; selected={selected_ids:?}; legacy_text={legacy_text:?}; selected_text={selected_text:?}",
        check.name
    );
    compare_token_ids(&legacy_ids, &selected_ids).unwrap_or_else(|error| {
        panic!(
            "{} token IDs differ; prompt={:?}; legacy_ids={legacy_ids:?}; selected_ids={selected_ids:?}; legacy_text={legacy_text:?}; selected_text={selected_text:?}; {error}",
            check.name, check.prompt
        )
    });
    assert_eq!(
        legacy_text, selected_text,
        "{} text differs; prompt={:?}; legacy_ids={legacy_ids:?}; selected_ids={selected_ids:?}",
        check.name, check.prompt
    );
    let mut semantic_matches = Vec::new();
    for (arm, response) in [
        ("legacy", legacy_text.as_str()),
        ("selected", selected_text.as_str()),
    ] {
        let answer_check = require_expected_answer(response, check.expected_answer);
        let relation_check = check.expected_relation.map(|(relation, comparison_term)| {
            require_comparison_answer(response, check.expected_answer, relation, comparison_term)
        });
        let matched = answer_check.is_ok() && relation_check.as_ref().is_none_or(Result::is_ok);
        semantic_matches.push((arm, matched));
        println!(
            "card_24 semantic name={} arm={} answer_check={:?} relation_check={:?}",
            check.name, arm, answer_check, relation_check
        );
    }
    println!(
        "card_24 viability name={} expected={:?} prompt={:?} semantic_matches={semantic_matches:?} legacy_ids={legacy_ids:?} selected_ids={selected_ids:?} legacy_text={legacy_text:?} selected_text={selected_text:?}",
        check.name, check.expected_answer, check.prompt
    );
}

#[proxima::test]
async fn card_23_granite_prefill_uses_selected_attention_dispatch() {
    assert!(
        std::env::var_os("PROXIMA_CAPTURE_LIVE").is_some(),
        "run with PROXIMA_CAPTURE_LIVE=1"
    );
    let path = granite_checkpoint_path();
    let file = File::open(&path).expect("open the real Granite checkpoint");
    // SAFETY: this test only reads the checkpoint and does not mutate it.
    let mapping = unsafe { Mmap::map(&file) }.expect("mmap the Granite checkpoint");
    let parsed = parse_complete(&mapping).expect("parse the Granite checkpoint");
    let vocab = proxima_tokenizer::gguf::vocab_from_metadata(&parsed)
        .expect("build the Granite checkpoint vocab");
    let (prompt, prompt_tokens) = prompt_of_tokens(&vocab, PROMPT_TOKENS);
    assert!(prompt_tokens >= PROMPT_TOKENS);
    let model = LoadedModel::load(&parsed, &mapping).expect("bind the Granite checkpoint");
    let (legacy_dispatches, legacy_tokens) = capture_prompt_dispatches(&model, &prompt, None);
    let mut selected = AttentionVariant::default();
    selected.kv_reuse = AttentionKvReuse::SharedK;
    let (selected_dispatches, selected_tokens) =
        capture_prompt_dispatches(&model, &prompt, Some(selected));
    let matched = assert_selected_dispatch_pair(&legacy_dispatches, &selected_dispatches)
        .expect("the selected row-tiled Granite dispatch differs from legacy");
    assert!(
        !legacy_tokens.is_empty(),
        "legacy forward produced no token ids"
    );
    assert_eq!(
        legacy_tokens, selected_tokens,
        "selected prefill changed generated token ids"
    );
    println!(
        "card_23 matched_multirow_dispatches={matched} generated_token_ids={} legacy_dispatches={} selected_dispatches={}",
        legacy_tokens.len(),
        legacy_dispatches.len(),
        selected_dispatches.len()
    );
}

#[proxima::test]
async fn card_23_legacy_dispatch_fails_selected_identity_control() {
    let legacy = DispatchIdentity {
        node: 12,
        extents: vec![1000, 8, 2, 64],
        entry: "legacy_entry".to_string(),
        source_sha: "legacy_sha".to_string(),
        grid: omega::msl::GridSpec {
            threads: 1,
            threadgroup_width: None,
            depth: 1,
            grid2d: None,
        },
    };
    let error = assert_selected_dispatch_pair(&[legacy.clone()], &[legacy])
        .expect_err("the control must reject identical legacy and selected identities");
    assert!(error.contains("kept legacy entry"));
}

#[proxima::test]
async fn card_24_granite_continuation_viability_and_public_request() {
    let path = granite_checkpoint_path();
    let file = File::open(&path).expect("open the real Granite checkpoint");
    // SAFETY: this test only reads the checkpoint and does not mutate it.
    let mapping = unsafe { Mmap::map(&file) }.expect("mmap the Granite checkpoint");
    let parsed = parse_complete(&mapping).expect("parse the Granite checkpoint");
    let vocab = proxima_tokenizer::gguf::vocab_from_metadata(&parsed)
        .expect("build the Granite checkpoint vocab");
    let (fixture_prompt, prompt_ids, llama_ids) = granite_long_prompt_oracle();
    assert_eq!(prompt_ids.len(), 1_000, "fixture prompt id count");
    assert_eq!(llama_ids.len(), 128, "fixture generated id count");
    let wants_bos = vocab
        .add_bos_token()
        .unwrap_or_else(|| vocab.bos_token_id().is_some());
    let wants_eos = vocab.add_eos_token().unwrap_or(false);
    let encoded_prompt =
        proxima_tokenizer::encode_with_bos_eos(&fixture_prompt, &vocab, wants_bos, wants_eos)
            .expect("encode the recorded Granite prompt");
    assert_eq!(
        encoded_prompt, prompt_ids,
        "text request tokenization must match the recorded llama prompt ids"
    );

    let model = LoadedModel::load(&parsed, &mapping).expect("bind the Granite checkpoint");
    let (legacy_ids, legacy_text, _) = model
        .generate_with_serving_config(&fixture_prompt, 128, serving_config(None))
        .unwrap_or_else(|error| panic!("legacy Granite request failed: {error:?}"));
    let (selected_ids, selected_text, _) = model
        .generate_with_serving_config(
            &fixture_prompt,
            128,
            serving_config(Some(shared_k_variant())),
        )
        .unwrap_or_else(|error| panic!("selected Granite request failed: {error:?}"));

    assert_eq!(legacy_ids.len(), 128, "legacy request must return 128 ids");
    assert_eq!(
        selected_ids.len(),
        128,
        "selected request must return 128 ids"
    );
    compare_token_ids(&llama_ids, &legacy_ids).unwrap_or_else(|error| {
        panic!("legacy output differs from the recorded llama ids: {error}")
    });
    compare_token_ids(&llama_ids, &selected_ids).unwrap_or_else(|error| {
        panic!("selected output differs from the recorded llama ids: {error}")
    });
    compare_token_ids(&legacy_ids, &selected_ids)
        .unwrap_or_else(|error| panic!("legacy and selected outputs differ: {error}"));
    assert!(
        !selected_text.trim().is_empty(),
        "selected public request returned no text"
    );
    compare_token_ids(&llama_ids[..8], &selected_ids[..8]).unwrap_or_else(|error| {
        panic!("selected public request prefix differs from the oracle: {error}")
    });

    println!(
        "card_24 oracle_cases=1 prompt_ids={} generated_ids={} selected_request_text_bytes={}",
        prompt_ids.len(),
        selected_ids.len(),
        selected_text.len(),
    );
    assert!(
        !legacy_text.trim().is_empty(),
        "legacy request returned no text"
    );
}

#[proxima::test]
async fn card_24_granite_viability_paris() {
    run_granite_viability_case(granite_viability_checks().remove(0));
}

#[proxima::test]
async fn card_24_granite_viability_soliloquy() {
    run_granite_viability_case(granite_viability_checks().remove(1));
}

#[proxima::test]
async fn card_24_granite_viability_ant_vs_briefcase() {
    run_granite_viability_case(granite_viability_checks().remove(2));
}

#[proxima::test]
async fn card_24_granite_viability_hippo_vs_building() {
    run_granite_viability_case(granite_viability_checks().remove(3));
}

#[proxima::test]
async fn card_25_granite_sequential_public_requests_complete_on_one_model() {
    let path = granite_checkpoint_path();
    let file = File::open(&path).expect("open the real Granite checkpoint");
    // SAFETY: this test only reads the checkpoint and does not mutate it.
    let mapping = unsafe { Mmap::map(&file) }.expect("mmap the Granite checkpoint");
    let parsed = parse_complete(&mapping).expect("parse the Granite checkpoint");
    let model = LoadedModel::load(&parsed, &mapping).expect("bind the Granite checkpoint");

    let (fixture_prompt, _, fixture_expected_ids) = granite_long_prompt_oracle();
    let (fixture_legacy_ids, fixture_legacy_text, _) = model
        .generate_with_serving_config(&fixture_prompt, 128, serving_config(None))
        .unwrap_or_else(|error| panic!("fixture legacy request failed: {error:?}"));
    let (fixture_selected_ids, fixture_selected_text, _) = model
        .generate_with_serving_config(
            &fixture_prompt,
            128,
            serving_config(Some(shared_k_variant())),
        )
        .unwrap_or_else(|error| panic!("fixture selected request failed: {error:?}"));
    compare_token_ids(&fixture_expected_ids, &fixture_legacy_ids)
        .unwrap_or_else(|error| panic!("fixture legacy output differs: {error}"));
    compare_token_ids(&fixture_expected_ids, &fixture_selected_ids)
        .unwrap_or_else(|error| panic!("fixture selected output differs: {error}"));
    assert_eq!(fixture_legacy_text, fixture_selected_text);
    println!(
        "card_25 fixture prompt_ids=1000 generated_ids={} legacy_text_bytes={} selected_text_bytes={}",
        fixture_expected_ids.len(),
        fixture_legacy_text.len(),
        fixture_selected_text.len()
    );

    for check in granite_viability_checks() {
        let _ = take_captured_dispatches();
        let (legacy_ids, legacy_text, _) = model
            .generate_with_serving_config(&check.prompt, 48, serving_config(None))
            .unwrap_or_else(|error| {
                let node = route_mismatch_node(&error);
                let route_state = captured_route_prepass_summary(node);
                panic!(
                    "{} legacy request failed: {error:?}; route prepass captures: {route_state}",
                    check.name
                )
            });
        let (selected_ids, selected_text, _) = match model.generate_with_serving_config(
            &check.prompt,
            48,
            serving_config(Some(shared_k_variant())),
        ) {
            Ok(result) => result,
            Err(error) => {
                let node = route_mismatch_node(&error);
                let route_state = captured_route_prepass_summary(node);
                panic!(
                    "{} selected request failed: {error:?}; route prepass captures: {route_state}",
                    check.name
                );
            }
        };
        assert!(
            !legacy_ids.is_empty()
                && !selected_ids.is_empty()
                && !legacy_text.trim().is_empty()
                && !selected_text.trim().is_empty(),
            "{} repeated request returned an empty result: legacy_ids={legacy_ids:?}, legacy_text={legacy_text:?}, selected_ids={selected_ids:?}, selected_text={selected_text:?}",
            check.name
        );
        compare_token_ids(&legacy_ids, &selected_ids)
            .unwrap_or_else(|error| panic!("{} repeated request arms differ: {error}", check.name));
        assert_eq!(
            legacy_text, selected_text,
            "{} repeated request text",
            check.name
        );
        let answer_check = require_expected_answer(&selected_text, check.expected_answer);
        let relation_check = check.expected_relation.map(|(relation, comparison_term)| {
            require_comparison_answer(
                &selected_text,
                check.expected_answer,
                relation,
                comparison_term,
            )
        });
        println!(
            "card_25 request name={} answer_check={answer_check:?} relation_check={relation_check:?} legacy_ids={legacy_ids:?} selected_ids={selected_ids:?} legacy_text={legacy_text:?} selected_text={selected_text:?}",
            check.name
        );
    }
}

fn shared_k_variant() -> AttentionVariant {
    let mut variant = AttentionVariant::default();
    variant.kv_reuse = AttentionKvReuse::SharedK;
    variant
}

fn f32_shared_prefetch_variant() -> AttentionVariant {
    let mut variant = AttentionVariant::default();
    variant.mma_precision = omega::AttentionMmaPrecision::F16;
    variant.kv_reuse = omega::AttentionKvReuse::SharedKv;
    variant.tile_height = omega::AttentionTileHeight::Rows8;
    variant.query_parallelism = omega::AttentionQueryParallelism::SimdgroupRows;
    variant.simd_topology = omega::AttentionSimdTopology::PerHead;
    variant.prefetch = omega::AttentionPrefetch::NextBlock;
    variant
}

fn f32_shared_variant() -> AttentionVariant {
    let mut variant = f32_shared_prefetch_variant();
    variant.prefetch = omega::AttentionPrefetch::Off;
    variant
}

fn f32_shared_k_variant() -> AttentionVariant {
    let mut variant = f32_shared_variant();
    variant.kv_reuse = AttentionKvReuse::SharedK;
    variant
}

fn f16_variant() -> AttentionVariant {
    let mut variant = AttentionVariant::default();
    variant.mma_precision = omega::AttentionMmaPrecision::F16;
    variant
}

fn f16_rows8_groups4_variant() -> AttentionVariant {
    let mut variant = f16_variant();
    variant.tile_height = omega::AttentionTileHeight::Rows8;
    variant.simdgroup_count = AttentionSimdgroupCount::Groups4;
    variant
}

fn f16_rows16_groups4_variant() -> AttentionVariant {
    let mut variant = f16_rows8_groups4_variant();
    variant.tile_height = omega::AttentionTileHeight::Rows16;
    variant
}

fn f16_shared_kv_simdgroup_rows_variant() -> AttentionVariant {
    let mut variant = f16_variant();
    variant.kv_reuse = omega::AttentionKvReuse::SharedKv;
    variant.query_parallelism = omega::AttentionQueryParallelism::SimdgroupRows;
    variant
}

fn f16_simdgroup_rows_variant() -> AttentionVariant {
    let mut variant = AttentionVariant::default();
    variant.mma_precision = omega::AttentionMmaPrecision::F16;
    variant.query_parallelism = omega::AttentionQueryParallelism::SimdgroupRows;
    variant
}

fn f16_shared_k_variant() -> AttentionVariant {
    let mut variant = AttentionVariant::default();
    variant.mma_precision = omega::AttentionMmaPrecision::F16;
    variant.kv_reuse = AttentionKvReuse::SharedK;
    variant
}

fn f16_rows16_variant() -> AttentionVariant {
    let mut variant = AttentionVariant::default();
    variant.mma_precision = omega::AttentionMmaPrecision::F16;
    variant.tile_height = omega::AttentionTileHeight::Rows16;
    variant
}

fn f16_rows16_shared_k_variant() -> AttentionVariant {
    let mut variant = f16_rows16_variant();
    variant.kv_reuse = omega::AttentionKvReuse::SharedK;
    variant
}

fn f32_rows16_shared_k_variant() -> AttentionVariant {
    let mut variant = f16_rows16_shared_k_variant();
    variant.mma_precision = omega::AttentionMmaPrecision::F32;
    variant
}

fn f16_rows16_shared_k_simdgroup_rows_variant() -> AttentionVariant {
    let mut variant = f16_rows16_shared_k_variant();
    variant.query_parallelism = omega::AttentionQueryParallelism::SimdgroupRows;
    variant
}

fn f16_rows16_simdgroup_rows_variant() -> AttentionVariant {
    let mut variant = f16_rows16_variant();
    variant.query_parallelism = omega::AttentionQueryParallelism::SimdgroupRows;
    variant
}

fn f32_output_difference(left: &[u8], right: &[u8]) -> Value {
    assert_eq!(left.len(), right.len(), "attention output byte lengths");
    assert_eq!(left.len() % 4, 0, "attention output is packed f32");
    let mut changed_bits = 0usize;
    let mut max_absolute_difference = 0.0f32;
    let mut squared_difference_sum = 0.0f64;
    let mut non_finite_pairs = 0usize;
    for (left_value, right_value) in left.chunks_exact(4).zip(right.chunks_exact(4)) {
        let left_value =
            f32::from_ne_bytes([left_value[0], left_value[1], left_value[2], left_value[3]]);
        let right_value = f32::from_ne_bytes([
            right_value[0],
            right_value[1],
            right_value[2],
            right_value[3],
        ]);
        changed_bits += usize::from(left_value.to_bits() != right_value.to_bits());
        if left_value.is_finite() && right_value.is_finite() {
            let difference = (left_value - right_value).abs();
            max_absolute_difference = max_absolute_difference.max(difference);
            squared_difference_sum += f64::from(difference) * f64::from(difference);
        } else {
            non_finite_pairs += 1;
        }
    }
    json!({
        "elements": left.len() / 4,
        "changed_bits": changed_bits,
        "max_absolute_difference": max_absolute_difference,
        "rms_difference": (squared_difference_sum / (left.len() / 4) as f64).sqrt(),
        "non_finite_pairs": non_finite_pairs,
    })
}

fn run_variant_prefill_probe(variant_name: &str, variant: AttentionVariant) {
    run_variant_prefill_probe_shapes(
        variant_name,
        None,
        variant,
        &[SHORT_PROMPT_TOKENS, PROMPT_TOKENS],
    );
}

fn run_variant_prefill_probe_shapes(
    variant_name: &str,
    baseline_variant: Option<AttentionVariant>,
    variant: AttentionVariant,
    prompt_shapes: &[usize],
) {
    run_variant_prefill_probe_shapes_with_grid_requirement(
        variant_name,
        baseline_variant,
        variant,
        prompt_shapes,
        false,
        false,
    );
}

fn run_variant_prefill_probe_shapes_with_grid_requirement(
    variant_name: &str,
    baseline_variant: Option<AttentionVariant>,
    variant: AttentionVariant,
    prompt_shapes: &[usize],
    require_equal_grid: bool,
    require_equal_output: bool,
) {
    let path = granite_checkpoint_path();
    let file = File::open(&path).expect("open the real Granite checkpoint");
    // SAFETY: this probe reads the checkpoint without modifying it.
    let mapping = unsafe { Mmap::map(&file) }.expect("mmap the Granite checkpoint");
    let parsed = parse_complete(&mapping).expect("parse the Granite checkpoint");
    let vocab = proxima_tokenizer::gguf::vocab_from_metadata(&parsed)
        .expect("build the Granite checkpoint vocab");
    let model = LoadedModel::load(&parsed, &mapping).expect("bind the Granite checkpoint");

    for &nominal_tokens in prompt_shapes {
        let (prompt, actual_tokens) = prompt_prefix_of_tokens(&vocab, nominal_tokens);
        let baseline = capture_attention_replay(&model, &prompt, baseline_variant, None);
        let selected =
            capture_attention_replay(&model, &prompt, Some(variant), Some(&baseline.dispatch));
        assert_eq!(
            baseline.dispatch.node, selected.dispatch.node,
            "attention node"
        );
        assert_eq!(
            baseline.dispatch.extents, selected.dispatch.extents,
            "attention extents"
        );
        assert_ne!(
            baseline.dispatch.entry, selected.dispatch.entry,
            "selected entry"
        );
        assert_ne!(
            baseline.dispatch.msl_sha256, selected.dispatch.msl_sha256,
            "selected source identity"
        );
        if require_equal_grid {
            assert_eq!(
                baseline.dispatch.grid, selected.dispatch.grid,
                "MMA precision variants must preserve dispatch grid"
            );
        }
        assert!(
            !baseline
                .dispatch
                .bindings
                .iter()
                .any(|binding| matches!(binding, Binding::Fault))
        );
        assert!(
            !selected
                .dispatch
                .bindings
                .iter()
                .any(|binding| matches!(binding, Binding::Fault))
        );
        let output_difference = f32_output_difference(&baseline.output, &selected.output);
        assert_eq!(
            output_difference["non_finite_pairs"], 0,
            "attention output comparison contains only finite values"
        );
        assert!(
            output_difference["max_absolute_difference"]
                .as_f64()
                .is_some_and(f64::is_finite),
            "maximum output difference is finite"
        );
        assert!(
            output_difference["rms_difference"]
                .as_f64()
                .is_some_and(f64::is_finite),
            "RMS output difference is finite"
        );
        if require_equal_output {
            assert_eq!(
                output_difference["changed_bits"], 0,
                "Rows16 must preserve every captured output bit"
            );
        }
        let token_comparison = compare_token_ids(&baseline.token_ids, &selected.token_ids);
        let ids_equal = token_comparison.is_ok();
        token_comparison
            .unwrap_or_else(|error| panic!("{variant_name} Granite generated IDs differ: {error}"));

        let mut baseline_samples = Vec::with_capacity(20);
        let mut selected_samples = Vec::with_capacity(20);
        let mut signed_deltas = Vec::with_capacity(20);
        for round in 0..20 {
            let (first, second) = if round % 2 == 0 {
                (&baseline.dispatch, &selected.dispatch)
            } else {
                (&selected.dispatch, &baseline.dispatch)
            };
            let first_ns = first.time_gpu_ns(1).expect("first replay completes");
            let second_ns = second.time_gpu_ns(1).expect("second replay completes");
            let (baseline_ns, selected_ns) = if round % 2 == 0 {
                (first_ns, second_ns)
            } else {
                (second_ns, first_ns)
            };
            assert!(
                baseline_ns.is_finite() && baseline_ns > 0.0,
                "baseline GPU replay time must be finite and positive"
            );
            assert!(
                selected_ns.is_finite() && selected_ns > 0.0,
                "selected GPU replay time must be finite and positive"
            );
            baseline_samples.push(baseline_ns);
            selected_samples.push(selected_ns);
            signed_deltas.push(selected_ns - baseline_ns);
        }
        let baseline_pipeline_resources = baseline.dispatch.pipeline_resources();
        let selected_pipeline_resources = selected.dispatch.pipeline_resources();
        println!(
            "granite_variant_probe name={variant_name} config={variant:?} nominal_tokens={nominal_tokens} actual_tokens={} node={} extents={:?} baseline_entry={} selected_entry={} baseline_msl_sha256={} selected_msl_sha256={} baseline_grid={:?} selected_grid={:?} baseline_pipeline_resources={baseline_pipeline_resources:?} selected_pipeline_resources={selected_pipeline_resources:?} output_diff={output_difference} ids_equal={ids_equal} baseline_ids={:?} selected_ids={:?} baseline_samples_ns={baseline_samples:?} selected_samples_ns={selected_samples:?} selected_minus_baseline_ns={signed_deltas:?} baseline_summary={:?} selected_summary={:?}",
            actual_tokens,
            baseline.dispatch.node,
            baseline.dispatch.extents,
            baseline.dispatch.entry,
            selected.dispatch.entry,
            baseline.dispatch.msl_sha256,
            selected.dispatch.msl_sha256,
            baseline.dispatch.grid,
            selected.dispatch.grid,
            baseline.token_ids,
            selected.token_ids,
            summarize_samples(&baseline_samples),
            summarize_samples(&selected_samples),
        );
    }
}

#[proxima::test]
async fn card_24_wrong_oracle_and_wrong_fact_are_rejected() {
    let expected_ids = [203, 433, 19482, 1236, 47615, 8558, 12011, 2783];
    let mut wrong_ids = expected_ids;
    wrong_ids[0] = 204;
    let error = compare_token_ids(&wrong_ids, &expected_ids)
        .expect_err("a changed llama token must be rejected");
    assert!(error.contains("token 0"), "unexpected mismatch: {error}");

    let error = require_expected_answer("The answer is unknown.", "paris")
        .expect_err("an answer without the expected fact must be rejected");
    assert!(
        error.contains("paris"),
        "unexpected fact rejection: {error}"
    );

    require_expected_answer("The capital of France is not Paris.", "paris")
        .expect_err("a negated Paris answer must be rejected");
    require_expected_answer("It isn't called a soliloquy.", "soliloquy")
        .expect_err("a negated soliloquy answer must be rejected");

    let error = require_comparison_answer(
        "The briefcase is bigger than an ant? No, it isn't.",
        "briefcase",
        "bigger",
        "ant",
    )
    .expect_err("a negated ant comparison must fail");
    assert!(error.contains("briefcase"));

    let error = require_comparison_answer(
        "A large office building is smaller than a hippopotamus.",
        "hippopotamus",
        "smaller",
        "building",
    )
    .expect_err("a reversed hippo comparison must fail");
    assert!(error.contains("hippopotamus"));
}

#[proxima::test]
async fn card_26_granite_attention_replay_pair_matches_complete_output() {
    assert!(
        std::env::var_os("PROXIMA_CAPTURE_LIVE").is_some(),
        "run with PROXIMA_CAPTURE_LIVE=1"
    );
    let path = granite_checkpoint_path();
    let file = File::open(&path).expect("open the real Granite checkpoint");
    // SAFETY: this test only reads the checkpoint and does not mutate it.
    let mapping = unsafe { Mmap::map(&file) }.expect("mmap the Granite checkpoint");
    let parsed = parse_complete(&mapping).expect("parse the Granite checkpoint");
    let vocab = proxima_tokenizer::gguf::vocab_from_metadata(&parsed)
        .expect("build the Granite checkpoint vocab");
    let (prompt, prompt_tokens) = prompt_of_tokens(&vocab, PROMPT_TOKENS);
    assert!(prompt_tokens >= PROMPT_TOKENS);
    let model = LoadedModel::load(&parsed, &mapping).expect("bind the Granite checkpoint");

    let legacy = capture_attention_replay(&model, &prompt, None, None);
    let legacy_evidence = replay_evidence(&legacy);
    let selected = capture_attention_replay(
        &model,
        &prompt,
        Some(shared_k_variant()),
        Some(&legacy.dispatch),
    );
    let selected_evidence = replay_evidence(&selected);
    compare_replay_pair(Some(&legacy_evidence), Some(&selected_evidence)).unwrap_or_else(|error| {
        panic!(
            "captured Granite attention outputs differ: {error}; legacy_node={} legacy_extents={:?} legacy_entry={} legacy_sha={} legacy_grid={:?}; selected_node={} selected_extents={:?} selected_entry={} selected_sha={} selected_grid={:?}",
            legacy.dispatch.node,
            legacy.dispatch.extents,
            legacy.dispatch.entry,
            legacy.dispatch.msl_sha256,
            legacy.dispatch.grid,
            selected.dispatch.node,
            selected.dispatch.extents,
            selected.dispatch.entry,
            selected.dispatch.msl_sha256,
            selected.dispatch.grid
        )
    });
    assert!(
        !legacy.token_ids.is_empty(),
        "legacy request produced no generated token IDs"
    );
    compare_token_ids(&legacy.token_ids, &selected.token_ids)
        .unwrap_or_else(|error| panic!("captured Granite request IDs differ: {error}"));
    println!(
        "card_26 matched_dispatch node={} extents={:?} legacy_entry={} selected_entry={} legacy_sha={} selected_sha={} legacy_grid={:?} selected_grid={:?} legacy_output_bytes={} selected_output_bytes={} output_equal=true legacy_ids={:?} selected_ids={:?} ids_equal=true fault_bindings=0",
        legacy.dispatch.node,
        legacy.dispatch.extents,
        legacy.dispatch.entry,
        selected.dispatch.entry,
        legacy.dispatch.msl_sha256,
        selected.dispatch.msl_sha256,
        legacy.dispatch.grid,
        selected.dispatch.grid,
        legacy.output.len(),
        selected.output.len(),
        legacy.token_ids,
        selected.token_ids,
    );
}

#[proxima::test]
async fn card_26_replay_comparator_rejects_three_false_pairs() {
    let identity_for = |entry: &str, source_sha: &str| DispatchIdentity {
        node: 12,
        extents: vec![1000, 8, 2, 64],
        entry: entry.to_string(),
        source_sha: source_sha.to_string(),
        grid: omega::msl::GridSpec {
            threads: 64,
            threadgroup_width: Some(64),
            depth: 1,
            grid2d: None,
        },
    };
    let reference_output = [0.25f32, -0.5, 1.0, 2.0]
        .into_iter()
        .flat_map(f32::to_ne_bytes)
        .collect::<Vec<_>>();
    let legacy = ReplayEvidence {
        identity: identity_for("legacy_entry", "legacy_sha"),
        bindings: Vec::new(),
        output: reference_output.clone(),
    };
    let selected = ReplayEvidence {
        identity: identity_for("selected_entry", "selected_sha"),
        bindings: Vec::new(),
        output: reference_output,
    };
    compare_replay_pair(Some(&legacy), Some(&selected))
        .expect("matching captured replay evidence is admitted");

    let mut changed_output = selected.output.clone();
    changed_output[2] ^= 1;
    let changed = ReplayEvidence {
        output: changed_output,
        ..selected.clone()
    };
    let changed_error = compare_replay_pair(Some(&legacy), Some(&changed))
        .expect_err("a changed output byte must be rejected");
    assert!(changed_error.contains("output byte 2"));

    let missing_error = compare_replay_pair(Some(&legacy), None)
        .expect_err("an absent selected record must be rejected");
    assert!(missing_error.contains("selected dispatch record is missing"));

    let fault_bound = ReplayEvidence {
        bindings: vec![Binding::Fault],
        ..selected
    };
    let fault_error = compare_replay_pair(Some(&legacy), Some(&fault_bound))
        .expect_err("a fault-bound selected record must be rejected");
    assert!(fault_error.contains("selected dispatch contains a fault binding"));
    println!("card_26 comparator_controls=3 rejected=3");
}

fn measure_granite_attention_shape(
    model: &LoadedModel<'_>,
    nominal_prompt_tokens: usize,
    prompt: &str,
    prompt_ids: &[u32],
    baseline_variant: Option<AttentionVariant>,
    selected_variant: AttentionVariant,
    selected_label: &str,
    warmup_rounds: usize,
) -> Value {
    let legacy = capture_attention_replay(model, prompt, baseline_variant, None);
    let legacy_evidence = replay_evidence(&legacy);
    let selected = capture_attention_replay(
        model,
        prompt,
        Some(selected_variant),
        Some(&legacy.dispatch),
    );
    assert_active_grid_dimensions(&legacy.dispatch);
    assert_active_grid_dimensions(&selected.dispatch);
    let selected_evidence = replay_evidence(&selected);
    compare_replay_pair(Some(&legacy_evidence), Some(&selected_evidence))
        .unwrap_or_else(|error| panic!("captured Granite attention outputs differ: {error}"));
    assert!(
        !legacy.token_ids.is_empty(),
        "legacy request returned no IDs"
    );
    compare_token_ids(&legacy.token_ids, &selected.token_ids)
        .unwrap_or_else(|error| panic!("captured Granite request IDs differ: {error}"));
    assert_eq!(
        legacy.dispatch.extents[0],
        prompt_ids.len() as u64,
        "captured attention query extent differs from tokenizer count"
    );

    let mut warmups = Vec::with_capacity(warmup_rounds);
    for round in 0..warmup_rounds {
        let arm_order = if round % 2 == 0 {
            ["legacy", selected_label]
        } else {
            [selected_label, "legacy"]
        };
        let mut samples = Vec::with_capacity(2);
        for (position, arm_name) in arm_order.iter().enumerate() {
            let dispatch = match *arm_name {
                "legacy" => &legacy.dispatch,
                label if label == selected_label => &selected.dispatch,
                _ => unreachable!("warmup order contains only admitted arms"),
            };
            let gpu_ns = dispatch
                .time_gpu_ns(1)
                .expect("single-dispatch GPU warmup completes");
            assert!(
                gpu_ns.is_finite() && gpu_ns > 0.0,
                "GPU warmup time must be finite and positive: {gpu_ns}"
            );
            samples.push(json!({
                "arm": arm_name,
                "position": position,
                "gpu_ns": gpu_ns,
            }));
        }
        warmups.push(json!({ "round": round, "arm_order": arm_order, "samples": samples }));
    }

    let mut legacy_samples = Vec::with_capacity(20);
    let mut selected_samples = Vec::with_capacity(20);
    let mut rounds = Vec::with_capacity(20);
    for round in 0..20 {
        let arm_order = if round % 2 == 0 {
            ["legacy", selected_label]
        } else {
            [selected_label, "legacy"]
        };
        let mut legacy_time = None;
        let mut selected_time = None;
        for (position, arm_name) in arm_order.iter().enumerate() {
            let dispatch = match *arm_name {
                "legacy" => &legacy.dispatch,
                label if label == selected_label => &selected.dispatch,
                _ => unreachable!("arm order contains only admitted arms"),
            };
            let gpu_ns = dispatch
                .time_gpu_ns(1)
                .expect("single-dispatch GPU replay completes");
            assert!(
                gpu_ns.is_finite() && gpu_ns > 0.0,
                "GPU replay time must be finite and positive: {gpu_ns}"
            );
            let sample = json!({
                "round": round,
                "position": position,
                "gpu_ns": gpu_ns,
            });
            match *arm_name {
                "legacy" => {
                    legacy_samples.push(sample);
                    legacy_time = Some(gpu_ns);
                }
                label if label == selected_label => {
                    selected_samples.push(sample);
                    selected_time = Some(gpu_ns);
                }
                _ => unreachable!("arm order contains only admitted arms"),
            }
        }
        let legacy_gpu_ns = legacy_time.expect("legacy arm ran once per round");
        let selected_gpu_ns = selected_time.expect("selected arm ran once per round");
        rounds.push(json!({
            "round": round,
            "arm_order": arm_order,
            "selected_minus_legacy_ns": selected_gpu_ns - legacy_gpu_ns,
        }));
    }

    let legacy_resource = resource_replays(&legacy.dispatch, "legacy");
    let selected_resource = resource_replays(&selected.dispatch, selected_label);
    json!({
        "nominal_prompt_tokens": nominal_prompt_tokens,
        "actual_prompt_tokens": prompt_ids.len(),
        "prompt_ids_sha256": prompt_ids_sha256(prompt_ids),
        "output_equal": true,
        "ids_equal": true,
        "warmup_rounds": warmups,
        "rounds": rounds,
        "arms": [
            measured_arm_report("legacy", &legacy, legacy_samples, legacy_resource),
            measured_arm_report(selected_label, &selected, selected_samples, selected_resource),
        ],
    })
}


fn run_granite_attention_measurement(
    nominal_targets: &[usize],
    baseline_variant: Option<AttentionVariant>,
    selected_variant: AttentionVariant,
    selected_label: &str,
) {
    run_granite_attention_measurement_with_warmup(
        nominal_targets,
        baseline_variant,
        selected_variant,
        selected_label,
        0,
    );
}

fn run_granite_attention_measurement_with_warmup(
    nominal_targets: &[usize],
    baseline_variant: Option<AttentionVariant>,
    selected_variant: AttentionVariant,
    selected_label: &str,
    warmup_rounds: usize,
) {
    assert!(
        env::var_os("PROXIMA_CAPTURE_LIVE").is_some(),
        "run with PROXIMA_CAPTURE_LIVE=1"
    );
    let report_path = env::var_os("PROXIMA_GRANITE_AB_REPORT")
        .map(PathBuf::from)
        .expect("set PROXIMA_GRANITE_AB_REPORT to the report destination");
    let checkpoint_path = granite_checkpoint_path();
    clear_previous_report(&report_path, Path::new(&checkpoint_path));
    assert!(
        !report_path.exists(),
        "previous Granite attention report remains at {}",
        report_path.display()
    );
    let file = File::open(&checkpoint_path).expect("open the real Granite checkpoint");
    // SAFETY: this test only reads the checkpoint and does not mutate it.
    let mapping = unsafe { Mmap::map(&file) }.expect("mmap the Granite checkpoint");
    let checkpoint_bytes = mapping.len() as u64;
    let parsed = parse_complete(&mapping).expect("parse the Granite checkpoint");
    let (model_name, checkpoint_sha256) = verify_granite_checkpoint(&parsed, mapping.as_ref());
    let vocab = proxima_tokenizer::gguf::vocab_from_metadata(&parsed)
        .expect("build the Granite checkpoint vocab");
    let prompt_shapes = nominal_targets
        .iter()
        .map(|&nominal| {
            let (prompt, actual) = if nominal == SHORT_PROMPT_TOKENS {
                prompt_prefix_of_tokens(&vocab, nominal)
            } else {
                prompt_of_tokens(&vocab, nominal)
            };
            let prompt_ids = proxima_tokenizer::encode(&prompt, &vocab)
                .expect("encode the measured Granite prompt");
            assert_eq!(prompt_ids.len(), actual, "prompt tokenizer count");
            assert!(
                actual >= nominal,
                "prompt did not reach nominal token target"
            );
            (nominal, prompt, prompt_ids)
        })
        .collect::<Vec<_>>();
    if prompt_shapes.len() == 2 {
        let (short_target, short_prompt, short_ids) = &prompt_shapes[0];
        let (long_target, long_prompt, long_ids) = &prompt_shapes[1];
        assert_eq!(*short_target, SHORT_PROMPT_TOKENS, "short prompt target");
        assert_eq!(*long_target, PROMPT_TOKENS, "long prompt target");
        assert!(
            long_prompt.starts_with(short_prompt),
            "short prompt text is not a prefix"
        );
        let first_mismatch = short_ids
            .iter()
            .zip(long_ids)
            .position(|(short, long)| short != long)
            .unwrap_or(short_ids.len());
        let context_start = first_mismatch.saturating_sub(4);
        let short_context_end = (first_mismatch + 8).min(short_ids.len());
        let long_context_end = (first_mismatch + 8).min(long_ids.len());
        let text_suffix = short_prompt
            .chars()
            .rev()
            .take(40)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<String>();
        let text_continuation = long_prompt[short_prompt.len()..]
            .chars()
            .take(40)
            .collect::<String>();
        let short_pieces = short_ids[context_start..short_context_end]
            .iter()
            .map(|&token_id| {
                (
                    token_id,
                    vocab.token_str(token_id),
                    vocab.token_bytes(token_id).map(String::from_utf8_lossy),
                )
            })
            .collect::<Vec<_>>();
        let long_pieces = long_ids[context_start..long_context_end]
            .iter()
            .map(|&token_id| {
                (
                    token_id,
                    vocab.token_str(token_id),
                    vocab.token_bytes(token_id).map(String::from_utf8_lossy),
                )
            })
            .collect::<Vec<_>>();
        assert!(
            long_ids.starts_with(short_ids),
            "short prompt IDs are not a prefix: short_count={} long_count={} first_mismatch={} short_ids={:?} long_ids={:?} short_pieces={short_pieces:?} long_pieces={long_pieces:?} short_text_suffix={text_suffix:?} long_text_continuation={text_continuation:?}",
            short_ids.len(),
            long_ids.len(),
            first_mismatch,
            &short_ids[context_start..short_context_end],
            &long_ids[context_start..long_context_end]
        );
        assert!(
            short_ids.len() < long_ids.len(),
            "prompt token counts are not distinct"
        );
    }
    let model = LoadedModel::load(&parsed, &mapping).expect("bind the Granite checkpoint");
    let shapes = prompt_shapes
        .iter()
        .map(|(nominal, prompt, prompt_ids)| {
            measure_granite_attention_shape(
                &model,
                *nominal,
                prompt,
                prompt_ids,
                baseline_variant,
                selected_variant,
                selected_label,
                warmup_rounds,
            )
        })
        .collect::<Vec<_>>();
    let report = json!({
        "version": 1,
        "model": "Granite 3.1 1B A400M Instruct",
        "selected_arm": selected_label,
        "checkpoint": {
            "path": checkpoint_path,
            "bytes": checkpoint_bytes,
            "sha256": checkpoint_sha256,
            "architecture": "granitemoe",
            "model_name": model_name,
            "gguf_file_type": 7,
            "weight_quant": "Q8_0",
        },
        "host": {
            "hostname": env::var("HOSTNAME").ok(),
            "arch": env::consts::ARCH,
            "os": env::consts::OS,
        },
        "device_description": Value::Null,
        "serving": {
            "gpu_layers": GPU_LAYERS_ALL,
            "kv_cache_key": "F32",
            "kv_cache_value": "F32",
            "flash_attention": false,
            "reasoning_budget": 0,
            "prompt_cache": "off",
            "ubatch_size": 0,
            "generated_tokens": 1,
        },
        "shapes": shapes,
    });
    write_report(&report_path, &report);
    for shape in report["shapes"].as_array().expect("shape records") {
        for arm in shape["arms"].as_array().expect("two arm records") {
            println!(
                "granite ab arm nominal={} actual={} arm={} samples={} resource={}",
                shape["nominal_prompt_tokens"]
                    .as_u64()
                    .expect("nominal count"),
                shape["actual_prompt_tokens"]
                    .as_u64()
                    .expect("actual count"),
                arm["arm"].as_str().expect("arm label"),
                arm["samples"].as_array().expect("raw samples").len(),
                arm["resource"].as_str().expect("resource observation")
            );
        }
    }
    println!("granite ab report={}", report_path.display());
}

#[proxima::test]
async fn perf_card_00_granite_attention_replay_cell() {
    run_granite_attention_measurement(&[PROMPT_TOKENS], None, shared_k_variant(), "shared_k");
}

#[proxima::test]
async fn perf_card_02_granite_two_prefill_shapes() {
    run_granite_attention_measurement(
        &[SHORT_PROMPT_TOKENS, PROMPT_TOKENS],
        None,
        shared_k_variant(),
        "shared_k",
    );
}


#[proxima::test]
async fn perf_granite_simdgroups4_warmup_control_against_f16_legacy() {
    let baseline = f16_variant();
    let selected = AttentionVariant {
        simdgroup_count: AttentionSimdgroupCount::Groups4,
        ..baseline
    };
    run_granite_attention_measurement_with_warmup(
        &[PROMPT_TOKENS],
        Some(baseline),
        selected,
        "simdgroups4",
        2,
    );
}


#[proxima::test]
async fn perf_granite_simdgroups4_warmup_control_two_shapes_against_f16_legacy() {
    let baseline = f16_variant();
    let settings = ServingSettings::from_env().expect("simdgroup count serving setting parses");
    let configured = settings
        .as_serving_config(&[])
        .attention_variant
        .expect("set PROXIMA_SERVING_ATTENTION_SIMDGROUP_COUNT=groups4");
    assert_eq!(configured.simdgroup_count, AttentionSimdgroupCount::Groups4);
    let selected = AttentionVariant {
        simdgroup_count: configured.simdgroup_count,
        ..baseline
    };
    run_granite_attention_measurement_with_warmup(
        &[SHORT_PROMPT_TOKENS, PROMPT_TOKENS],
        Some(baseline),
        selected,
        "simdgroups4",
        2,
    );
}


#[proxima::test]
async fn perf_granite_simdgroup_count_against_f16_legacy() {
    let baseline = f16_variant();
    let settings = ServingSettings::from_env().expect("simdgroup count serving setting parses");
    let configured = settings
        .as_serving_config(&[])
        .attention_variant
        .expect("set PROXIMA_SERVING_ATTENTION_SIMDGROUP_COUNT to groups2, groups4, or groups8");
    let selected = AttentionVariant {
        simdgroup_count: configured.simdgroup_count,
        ..baseline
    };
    assert_eq!(baseline.kv_storage, omega::AttentionKvStorage::F32);
    assert_eq!(selected.kv_storage, omega::AttentionKvStorage::F32);
    assert_eq!(baseline.mma_precision, omega::AttentionMmaPrecision::F16);
    assert_eq!(selected.mma_precision, omega::AttentionMmaPrecision::F16);
    assert_eq!(baseline.kv_reuse, omega::AttentionKvReuse::Legacy);
    assert_eq!(selected.kv_reuse, omega::AttentionKvReuse::Legacy);
    assert_eq!(baseline.tile_height, omega::AttentionTileHeight::Legacy);
    assert_eq!(selected.tile_height, omega::AttentionTileHeight::Legacy);
    assert_eq!(
        baseline.query_parallelism,
        omega::AttentionQueryParallelism::Legacy
    );
    assert_eq!(
        selected.query_parallelism,
        omega::AttentionQueryParallelism::Legacy
    );
    assert_eq!(baseline.simd_topology, omega::AttentionSimdTopology::Legacy);
    assert_eq!(selected.simd_topology, omega::AttentionSimdTopology::Legacy);
    assert_eq!(baseline.prefetch, omega::AttentionPrefetch::Off);
    assert_eq!(selected.prefetch, omega::AttentionPrefetch::Off);
    assert_eq!(
        baseline.simdgroup_count,
        omega::AttentionSimdgroupCount::Legacy
    );
    let selected_label = match selected.simdgroup_count {
        AttentionSimdgroupCount::Groups2 => "simdgroups2",
        AttentionSimdgroupCount::Groups4 => "simdgroups4",
        AttentionSimdgroupCount::Groups8 => "simdgroups8",
        AttentionSimdgroupCount::Legacy => {
            panic!("set PROXIMA_SERVING_ATTENTION_SIMDGROUP_COUNT to groups2, groups4, or groups8")
        }
    };

    run_granite_attention_measurement(&[PROMPT_TOKENS], Some(baseline), selected, selected_label);
}

#[proxima::test]
async fn perf_probe_granite_f16_mma_against_legacy() {
    run_variant_prefill_probe("f16_mma", f16_variant());
}

#[proxima::test]
async fn perf_probe_granite_f32_shared_prefetch_against_legacy() {
    run_variant_prefill_probe(
        "f32_f16_shared_kv_rows8_prefetch",
        f32_shared_prefetch_variant(),
    );
}

#[proxima::test]
async fn perf_probe_granite_f32_shared_prefetch_off_against_legacy() {
    run_variant_prefill_probe("f32_f16_shared_kv_rows8_prefetch_off", f32_shared_variant());
}

#[proxima::test]
async fn perf_probe_granite_f32_shared_k_parallel_against_legacy() {
    run_variant_prefill_probe(
        "f32_f16_shared_k_rows8_simdgroup_per_head",
        f32_shared_k_variant(),
    );
}

#[proxima::test]
async fn perf_granite_shared_kv_simdgroup_rows() {
    run_variant_prefill_probe_shapes(
        "f16_shared_kv_simdgroup_rows",
        Some(f16_variant()),
        f16_shared_kv_simdgroup_rows_variant(),
        &[SHORT_PROMPT_TOKENS, PROMPT_TOKENS],
    );
}

#[proxima::test]
async fn perf_granite_rows16_simdgroup_rows() {
    run_variant_prefill_probe_shapes(
        "f16_rows16_simdgroup_rows",
        Some(f16_variant()),
        f16_rows16_simdgroup_rows_variant(),
        &[SHORT_PROMPT_TOKENS, PROMPT_TOKENS],
    );
}

#[proxima::test]
async fn perf_granite_rows16_shared_k() {
    run_variant_prefill_probe_shapes(
        "f16_rows16_shared_k",
        Some(f16_variant()),
        f16_rows16_shared_k_variant(),
        &[SHORT_PROMPT_TOKENS, PROMPT_TOKENS],
    );
}

#[proxima::test]
async fn perf_granite_rows16_shared_k_f16_vs_f32_mma() {
    let baseline = f16_rows16_shared_k_variant();
    let selected = f32_rows16_shared_k_variant();
    for variant in [baseline, selected] {
        assert_eq!(variant.kv_storage, omega::AttentionKvStorage::F32);
        assert_eq!(variant.kv_reuse, omega::AttentionKvReuse::SharedK);
        assert_eq!(variant.tile_height, omega::AttentionTileHeight::Rows16);
        assert_eq!(
            variant.query_parallelism,
            omega::AttentionQueryParallelism::Legacy
        );
        assert_eq!(variant.simd_topology, omega::AttentionSimdTopology::Legacy);
        assert_eq!(variant.prefetch, omega::AttentionPrefetch::Off);
    }
    assert_eq!(baseline.mma_precision, omega::AttentionMmaPrecision::F16);
    assert_eq!(selected.mma_precision, omega::AttentionMmaPrecision::F32);

    run_variant_prefill_probe_shapes_with_grid_requirement(
        "f16_vs_f32_mma_rows16_shared_k",
        Some(baseline),
        selected,
        &[SHORT_PROMPT_TOKENS, PROMPT_TOKENS],
        true,
        false,
    );
}

#[proxima::test]
async fn perf_granite_rows16_only_repeat() {
    let baseline = f16_variant();
    let selected = f16_rows16_variant();
    for variant in [baseline, selected] {
        assert_eq!(variant.kv_storage, omega::AttentionKvStorage::F32);
        assert_eq!(variant.mma_precision, omega::AttentionMmaPrecision::F16);
        assert_eq!(variant.kv_reuse, omega::AttentionKvReuse::Legacy);
        assert_eq!(
            variant.query_parallelism,
            omega::AttentionQueryParallelism::Legacy
        );
        assert_eq!(variant.simd_topology, omega::AttentionSimdTopology::Legacy);
        assert_eq!(variant.prefetch, omega::AttentionPrefetch::Off);
    }
    assert_eq!(baseline.tile_height, omega::AttentionTileHeight::Legacy);
    assert_eq!(selected.tile_height, omega::AttentionTileHeight::Rows16);

    run_variant_prefill_probe_shapes_with_grid_requirement(
        "f16_rows16_only_repeat",
        Some(baseline),
        selected,
        &[SHORT_PROMPT_TOKENS, PROMPT_TOKENS],
        false,
        true,
    );
}

#[proxima::test]
async fn perf_granite_rows8_vs_rows16_groups4() {
    let baseline = f16_rows8_groups4_variant();
    let selected = f16_rows16_groups4_variant();
    for variant in [baseline, selected] {
        assert_eq!(variant.kv_storage, omega::AttentionKvStorage::F32);
        assert_eq!(variant.mma_precision, omega::AttentionMmaPrecision::F16);
        assert_eq!(variant.kv_reuse, AttentionKvReuse::Legacy);
        assert_eq!(
            variant.query_parallelism,
            omega::AttentionQueryParallelism::Legacy
        );
        assert_eq!(variant.simd_topology, omega::AttentionSimdTopology::Legacy);
        assert_eq!(variant.simdgroup_count, AttentionSimdgroupCount::Groups4);
        assert_eq!(variant.prefetch, omega::AttentionPrefetch::Off);
    }
    assert_eq!(baseline.tile_height, omega::AttentionTileHeight::Rows8);
    assert_eq!(selected.tile_height, omega::AttentionTileHeight::Rows16);

    run_variant_prefill_probe_shapes_with_grid_requirement(
        "f16_rows8_vs_rows16_groups4",
        Some(baseline),
        selected,
        &[SHORT_PROMPT_TOKENS, PROMPT_TOKENS],
        false,
        true,
    );
}


#[proxima::test]
async fn perf_granite_serving_tile_height_rows8_vs_rows16_groups4() {
    let rows8_settings = ServingSettings::builder()
        .attention_simdgroup_count(AttentionSimdgroupCount::Groups4)
        .attention_tile_height(AttentionTileHeightSetting::Rows8)
        .build();
    let rows16_settings = ServingSettings::builder()
        .attention_simdgroup_count(AttentionSimdgroupCount::Groups4)
        .attention_tile_height(AttentionTileHeightSetting::Rows16)
        .build();
    let rows8_config = rows8_settings.as_serving_config(&[]);
    let rows16_config = rows16_settings.as_serving_config(&[]);
    let rows8 = rows8_config
        .attention_variant
        .expect("Rows8 serving setting lowers to an attention variant");
    let rows16 = rows16_config
        .attention_variant
        .expect("Rows16 serving setting lowers to an attention variant");

    assert_eq!(rows8.tile_height, AttentionTileHeight::Rows8);
    assert_eq!(rows16.tile_height, AttentionTileHeight::Rows16);
    assert_eq!(rows8.simdgroup_count, AttentionSimdgroupCount::Groups4);
    assert_eq!(rows16.simdgroup_count, AttentionSimdgroupCount::Groups4);
    assert_eq!(rows8.mma_precision, rows16.mma_precision);
    assert_eq!(rows8.kv_reuse, rows16.kv_reuse);
    assert_eq!(rows8.query_parallelism, rows16.query_parallelism);
    assert_eq!(rows8.simd_topology, rows16.simd_topology);
    assert_eq!(rows8.prefetch, rows16.prefetch);

    run_variant_prefill_probe_shapes_with_grid_requirement(
        "serving_tile_height_rows8_vs_rows16_groups4",
        Some(rows8),
        rows16,
        &[SHORT_PROMPT_TOKENS, PROMPT_TOKENS],
        false,
        true,
    );
}


#[proxima::test]
async fn perf_granite_serving_legacy_tile_vs_rows8_groups4() {
    let legacy_settings = ServingSettings::builder()
        .attention_simdgroup_count(AttentionSimdgroupCount::Groups4)
        .attention_tile_height(AttentionTileHeightSetting::Legacy)
        .build();
    let rows8_settings = ServingSettings::builder()
        .attention_simdgroup_count(AttentionSimdgroupCount::Groups4)
        .attention_tile_height(AttentionTileHeightSetting::Rows8)
        .build();
    let legacy_config = legacy_settings.as_serving_config(&[]);
    let rows8_config = rows8_settings.as_serving_config(&[]);
    let legacy_tile = legacy_config
        .attention_variant
        .expect("Groups4 with legacy tile height lowers to an attention variant");
    let rows8 = rows8_config
        .attention_variant
        .expect("Rows8 serving setting lowers to an attention variant");

    assert_eq!(legacy_tile.tile_height, AttentionTileHeight::Legacy);
    assert_eq!(rows8.tile_height, AttentionTileHeight::Rows8);
    assert_eq!(
        legacy_tile.simdgroup_count,
        AttentionSimdgroupCount::Groups4
    );
    assert_eq!(rows8.simdgroup_count, AttentionSimdgroupCount::Groups4);
    assert_eq!(legacy_tile.kv_storage, rows8.kv_storage);
    assert_eq!(legacy_tile.mma_precision, rows8.mma_precision);
    assert_eq!(legacy_tile.kv_reuse, rows8.kv_reuse);
    assert_eq!(legacy_tile.query_parallelism, rows8.query_parallelism);
    assert_eq!(legacy_tile.simd_topology, rows8.simd_topology);
    assert_eq!(legacy_tile.prefetch, rows8.prefetch);

    run_variant_prefill_probe_shapes_with_grid_requirement(
        "serving_legacy_tile_vs_rows8_groups4",
        Some(legacy_tile),
        rows8,
        &[SHORT_PROMPT_TOKENS, PROMPT_TOKENS],
        false,
        true,
    );
}


#[proxima::test]
async fn perf_granite_rows16_shared_k_simdgroup_rows() {
    run_variant_prefill_probe_shapes(
        "f16_rows16_shared_k_simdgroup_rows",
        Some(f16_rows16_shared_k_variant()),
        f16_rows16_shared_k_simdgroup_rows_variant(),
        &[SHORT_PROMPT_TOKENS, PROMPT_TOKENS],
    );
}

#[proxima::test]
async fn perf_card_04_simdgroup_rows_only_against_f16() {
    run_variant_prefill_probe_shapes(
        "f16_simdgroup_rows_only",
        Some(f16_variant()),
        f16_simdgroup_rows_variant(),
        &[PROMPT_TOKENS],
    );
}

#[proxima::test]
async fn perf_card_04_shared_k_only_against_f16() {
    run_variant_prefill_probe_shapes(
        "f16_shared_k_only",
        Some(f16_variant()),
        f16_shared_k_variant(),
        &[PROMPT_TOKENS],
    );
}

#[proxima::test]
async fn perf_card_04_rows16_against_f16() {
    run_variant_prefill_probe_shapes(
        "f16_rows16_only",
        Some(f16_variant()),
        f16_rows16_variant(),
        &[PROMPT_TOKENS],
    );
}
