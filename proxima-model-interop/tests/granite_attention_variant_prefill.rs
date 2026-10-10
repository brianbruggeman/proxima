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

use memmap2::Mmap;
use omega::metal::MetalError;
use omega::msl::Binding;
use omega::{
    AttentionKvReuse, AttentionVariant, CapturedDispatch, set_capture_step,
    take_captured_dispatches,
};
use proxima_gguf::parse_complete;
use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::types::GgmlType;
use proxima_model_interop::{
    GPU_LAYERS_ALL, InteropError, LoadedModel, PromptCacheConfig, ServingConfig,
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

fn prompt_of_971_tokens(vocab: &proxima_tokenizer::Vocab) -> (String, usize) {
    let corpus = PASSAGE.repeat(PROMPT_TOKENS.div_ceil(40));
    let mut prompt = String::new();
    for word in corpus.split_inclusive(char::is_whitespace) {
        prompt.push_str(word);
        let token_count = proxima_tokenizer::encode(&prompt, vocab)
            .expect("tokenize prompt with the Granite vocab")
            .len();
        if token_count >= PROMPT_TOKENS {
            return (prompt, token_count);
        }
    }
    panic!("repeated Sherlock passage never reached {PROMPT_TOKENS} tokens");
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
    let (prompt, prompt_tokens) = prompt_of_971_tokens(&vocab);
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
    let (prompt, prompt_tokens) = prompt_of_971_tokens(&vocab);
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

#[proxima::test]
async fn perf_card_00_granite_attention_replay_cell() {
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
    let (prompt, prompt_tokens) = prompt_of_971_tokens(&vocab);
    let prompt_ids =
        proxima_tokenizer::encode(&prompt, &vocab).expect("encode the measured Granite prompt");
    assert_eq!(prompt_ids.len(), prompt_tokens, "prompt tokenizer count");
    let model = LoadedModel::load(&parsed, &mapping).expect("bind the Granite checkpoint");

    let legacy = capture_attention_replay(&model, &prompt, None, None);
    let legacy_evidence = replay_evidence(&legacy);
    let selected = capture_attention_replay(
        &model,
        &prompt,
        Some(shared_k_variant()),
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

    let mut legacy_samples = Vec::with_capacity(20);
    let mut selected_samples = Vec::with_capacity(20);
    let mut rounds = Vec::with_capacity(20);
    for round in 0..20 {
        let arm_order = if round % 2 == 0 {
            ["legacy", "shared_k"]
        } else {
            ["shared_k", "legacy"]
        };
        let mut legacy_time = None;
        let mut selected_time = None;
        for (position, arm_name) in arm_order.iter().enumerate() {
            let dispatch = match *arm_name {
                "legacy" => &legacy.dispatch,
                "shared_k" => &selected.dispatch,
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
                "shared_k" => {
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
    let selected_resource = resource_replays(&selected.dispatch, "shared_k");
    let report = json!({
        "version": 1,
        "model": "Granite 3.1 1B A400M Instruct",
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
        "shapes": [{
            "nominal_prompt_tokens": PROMPT_TOKENS,
            "actual_prompt_tokens": prompt_tokens,
            "prompt_ids_sha256": prompt_ids_sha256(&prompt_ids),
            "output_equal": true,
            "ids_equal": true,
            "rounds": rounds,
            "arms": [
                measured_arm_report("legacy", &legacy, legacy_samples, legacy_resource),
                measured_arm_report("shared_k", &selected, selected_samples, selected_resource),
            ],
        }],
    });
    write_report(&report_path, &report);
    for arm in report["shapes"][0]["arms"]
        .as_array()
        .expect("two arm records")
    {
        println!(
            "granite ab arm arm={} samples={} resource={}",
            arm["arm"].as_str().expect("arm label"),
            arm["samples"].as_array().expect("raw samples").len(),
            arm["resource"].as_str().expect("resource observation")
        );
    }
    println!("granite ab report={}", report_path.display());
}
