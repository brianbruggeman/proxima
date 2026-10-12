#![cfg(all(feature = "std", feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Round trip of the plain MLX export: a GGUF is exported to bf16 Hugging
//! Face files, the directory is read back by `LoadedModel::load_from_hf_directory`
//! and decodes greedily to the ids llama.cpp recorded for the same prompts.
//!
//! The export, the GGUF reference decode and the directory decode each run in
//! a child process of this test binary, so each reports a peak resident set
//! that holds nothing of the others. The directory's peak is held against the
//! GGUF's by the ratio the expert-stack bind test measured.

use std::fs::{File, create_dir_all, read_dir, write};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::process::Command;

use memmap2::Mmap;
use proxima_gguf::parse_complete;
use proxima_gguf::types::GgmlType;
use proxima_model_interop::profiles::binding_profile;
use proxima_model_interop::{
    DEFAULT_PLAIN_DTYPE, GPU_LAYERS_ALL, HfShard, Layout, LoadedModel, MLX_MAX_SHARD_BYTES,
    MlxExport, PromptCacheConfig, SANDWICH_NORM_LAYOUT, STANDARD_LAYOUT, ServingConfig,
    export_plain_named, export_tokenizer,
};
use proxima_primitives::pipe::sans_io::Outcome;
use proxima_safetensors::{Manifest, SafetensorsParser, TensorPayload};
use proxima_tokenizer::hf::{bos_eos_policy_from_tokenizer_config, vocab_from_tokenizer_json};
use proxima_tokenizer::{Vocab, encode_with_bos_eos};
use serde_json::Value;

const CHILD_ENV: &str = "PROXIMA_MLX_ROUND_TRIP_CHILD";
const DIRECTORY_ENV: &str = "PROXIMA_MLX_ROUND_TRIP_DIRECTORY";
const REUSE_ENV: &str = "PROXIMA_MLX_ROUND_TRIP_REUSE";
const CASE_MARKER: &str = "ROUND_TRIP_CASE";
const RSS_MARKER: &str = "ROUND_TRIP_RSS";
const HF_OVER_GGUF_PEAK_BOUND: f64 = 13.0;
const SHARD_SUFFIX: &str = ".safetensors";

struct Model {
    name: &'static str,
    gguf_env: &'static str,
    gguf_path: &'static str,
    layout: Layout,
    hf_prefix: &'static str,
    fixture: &'static str,
    expected_field: &'static str,
    stop_ids: &'static [u32],
}

const GRANITE: Model = Model {
    name: "granite",
    gguf_env: "PROXIMA_GRANITE_GGUF",
    gguf_path: "/Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80",
    layout: STANDARD_LAYOUT,
    hf_prefix: "model.",
    fixture: "llama-parity/granite_moe/llama_ids.json",
    expected_field: "generated_ids",
    stop_ids: &[],
};

const E2B: Model = Model {
    name: "e2b",
    gguf_env: "PROXIMA_GEMMA4_E2B_GGUF",
    gguf_path: "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd",
    layout: SANDWICH_NORM_LAYOUT,
    hf_prefix: "language_model.model.",
    fixture: "gemma4_e2b_llama_greedy_ids.json",
    expected_field: "oracle_ids",
    stop_ids: &[1, 106, 50],
};

struct Case {
    name: String,
    prompt: String,
    prompt_ids: Vec<u32>,
    expected_ids: Vec<u32>,
}

struct Generated {
    name: String,
    ids: Vec<u32>,
    tokenizer_match: bool,
}

fn ids_of(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .expect("ids are a json array")
        .iter()
        .map(|id| u32::try_from(id.as_u64().expect("an id is unsigned")).expect("an id fits u32"))
        .collect()
}

fn cases_of(model: &Model) -> Vec<Case> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(model.fixture);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    let records: Vec<Value> = serde_json::from_str(&text).expect("the fixture is a json array");
    records
        .iter()
        .enumerate()
        .map(|(position, record)| Case {
            name: record["name"]
                .as_str()
                .map_or_else(|| format!("prompt{position}"), String::from),
            prompt: record["prompt"]
                .as_str()
                .expect("record has a prompt")
                .to_owned(),
            prompt_ids: ids_of(&record["prompt_ids"]),
            expected_ids: ids_of(&record[model.expected_field]),
        })
        .collect()
}

fn cut_at_stop<'ids>(ids: &'ids [u32], stop_ids: &[u32]) -> &'ids [u32] {
    let end = ids
        .iter()
        .position(|id| stop_ids.contains(id))
        .unwrap_or(ids.len());
    &ids[..end]
}

fn first_divergence(expected: &[u32], actual: &[u32]) -> Option<usize> {
    let shared = expected.len().min(actual.len());
    (0..shared)
        .find(|&index| expected[index] != actual[index])
        .or_else(|| (expected.len() != actual.len()).then_some(shared))
}

fn divergence_after_stop(expected: &[u32], actual: &[u32], stop_ids: &[u32]) -> Option<usize> {
    first_divergence(
        cut_at_stop(expected, stop_ids),
        cut_at_stop(actual, stop_ids),
    )
}

fn peak_rss_bytes() -> u64 {
    process_usage::peak_rss_bytes().expect("peak rss must be readable for the current process")
}

fn gguf_path(model: &Model) -> PathBuf {
    PathBuf::from(std::env::var(model.gguf_env).unwrap_or_else(|_| model.gguf_path.into()))
}

fn map(path: &Path) -> Mmap {
    let file = File::open(path).unwrap_or_else(|error| panic!("open {}: {error}", path.display()));
    // SAFETY: the checkpoint and export files are not written while mapped.
    unsafe { Mmap::map(&file) }.expect("map file")
}

fn serving_config() -> ServingConfig<'static> {
    ServingConfig {
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

fn export_directory(model: &Model, directory: &Path) {
    let gguf = map(&gguf_path(model));
    let parsed = parse_complete(&gguf).expect("parse gguf");
    let family =
        proxima_model_interop::metadata_str(&parsed, "general.architecture").expect("family");
    let binding = binding_profile(family).expect("binding profile");
    let tensors = export_plain_named(
        &parsed,
        &gguf,
        DEFAULT_PLAIN_DTYPE,
        &binding,
        model.layout,
        model.hf_prefix,
    )
    .expect("plain export");
    let payloads = tensors
        .iter()
        .map(|tensor| TensorPayload {
            name: tensor.name.clone(),
            dtype: tensor.dtype,
            shape: tensor.shape.clone(),
            data: &tensor.bytes,
        })
        .collect();
    let export = MlxExport::assemble(payloads, MLX_MAX_SHARD_BYTES).expect("assemble shards");
    drop(tensors);
    create_dir_all(directory).expect("create export directory");
    export.shards.iter().for_each(|shard| {
        write(directory.join(&shard.file_name), &shard.bytes).expect("write shard")
    });
    let tokenizer = export_tokenizer(&parsed).expect("tokenizer files");
    write(directory.join("tokenizer.json"), &tokenizer.tokenizer_json)
        .expect("write tokenizer.json");
    write(
        directory.join("tokenizer_config.json"),
        &tokenizer.tokenizer_config_json,
    )
    .expect("write tokenizer_config.json");
}

fn manifest_of(file: &[u8]) -> (Manifest, u64) {
    let header_length = u64::from_le_bytes(file[..8].try_into().expect("8-byte prefix"));
    let data_start = 8 + header_length;
    let mut parser = SafetensorsParser::new();
    parser.feed(&file[..data_start as usize]);
    let Outcome::Event(manifest) = parser.poll().expect("header parses") else {
        panic!("a complete header must produce a manifest");
    };
    (manifest.clone(), data_start)
}

fn shard_paths(directory: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = read_dir(directory)
        .expect("read export directory")
        .map(|entry| entry.expect("directory entry").path())
        .filter(|path| path.to_string_lossy().ends_with(SHARD_SUFFIX))
        .collect();
    paths.sort();
    paths
}

fn special_id(tokenizer_json: &[u8], tokenizer_config: &[u8], key: &str) -> Option<u32> {
    let config: Value = serde_json::from_slice(tokenizer_config).ok()?;
    let token = config.get(key)?.as_str()?.to_owned();
    let tokenizer: Value = serde_json::from_slice(tokenizer_json).ok()?;
    u32::try_from(tokenizer["model"]["vocab"].get(token)?.as_u64()?).ok()
}

fn vocab_of(directory: &Path) -> Vocab {
    let tokenizer_json = std::fs::read(directory.join("tokenizer.json")).expect("tokenizer.json");
    let config =
        std::fs::read(directory.join("tokenizer_config.json")).expect("tokenizer_config.json");
    let [bos, eos, unk] = ["bos_token", "eos_token", "unk_token"]
        .map(|key| special_id(&tokenizer_json, &config, key));
    let vocab =
        vocab_from_tokenizer_json(&tokenizer_json, bos, eos, unk).expect("vocab reads back");
    let (add_bos, add_eos) =
        bos_eos_policy_from_tokenizer_config(&config).expect("policy reads back");
    vocab.with_bos_eos_policy(add_bos, add_eos)
}

fn tokenizer_matches(vocab: &Vocab, case: &Case) -> bool {
    let wants_bos = vocab
        .add_bos_token()
        .unwrap_or_else(|| vocab.bos_token_id().is_some());
    encode_with_bos_eos(
        &case.prompt,
        vocab,
        wants_bos,
        vocab.add_eos_token().unwrap_or(false),
    )
    .is_ok_and(|ids| ids == case.prompt_ids)
}

fn generate_cases(loaded: &LoadedModel<'_>, cases: &[Case], vocab: &Vocab) -> Vec<Generated> {
    let config = serving_config();
    cases
        .iter()
        .map(|case| {
            let (ids, _text, _stopped) = loaded
                .generate_from_ids(
                    &case.prompt_ids,
                    case.expected_ids.len(),
                    &config,
                    &mut |_event| ControlFlow::Continue(()),
                )
                .unwrap_or_else(|error| {
                    panic!("{}: generate_from_ids failed: {error:?}", case.name)
                });
            Generated {
                name: case.name.clone(),
                ids,
                tokenizer_match: tokenizer_matches(vocab, case),
            }
        })
        .collect()
}

fn report(generated: &[Generated]) {
    for item in generated {
        let ids: Vec<String> = item.ids.iter().map(u32::to_string).collect();
        println!(
            "{CASE_MARKER} name={} tokenizer_match={} ids={}",
            item.name,
            item.tokenizer_match,
            ids.join(",")
        );
    }
    println!("{RSS_MARKER} bytes={}", peak_rss_bytes());
}

fn generate_from_gguf(model: &Model) {
    let gguf = map(&gguf_path(model));
    let parsed = parse_complete(&gguf).expect("parse gguf");
    let loaded = LoadedModel::load(&parsed, &gguf).expect("gguf loads");
    let vocab = proxima_tokenizer::gguf::vocab_from_metadata(&parsed).expect("gguf vocab");
    report(&generate_cases(&loaded, &cases_of(model), &vocab));
}

fn generate_from_directory(model: &Model, directory: &Path) {
    let gguf = map(&gguf_path(model));
    let parsed = parse_complete(&gguf).expect("parse gguf header");
    let files: Vec<Mmap> = shard_paths(directory)
        .iter()
        .map(|path| map(path))
        .collect();
    let shards: Vec<HfShard<'_>> = files
        .iter()
        .map(|file| {
            let (manifest, data_start) = manifest_of(file);
            (manifest, &file[..], data_start)
        })
        .collect();
    let vocab = vocab_of(directory);
    let loaded = LoadedModel::load_from_hf_directory(&parsed, &gguf, &shards, vocab.clone())
        .expect("directory loads");
    report(&generate_cases(&loaded, &cases_of(model), &vocab));
}

fn spawn(test_name: &str, role: &str, directory: &Path) -> String {
    let output = Command::new(std::env::current_exe().expect("test binary path"))
        .args(["--exact", test_name, "--nocapture", "--test-threads=1"])
        .env(CHILD_ENV, role)
        .env(DIRECTORY_ENV, directory)
        .output()
        .expect("spawn child");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(
        output.status.success(),
        "{role} child failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    stdout
}

fn field<'line>(line: &'line str, key: &str) -> &'line str {
    line.split_whitespace()
        .find_map(|item| item.strip_prefix(key)?.strip_prefix('='))
        .unwrap_or_else(|| panic!("no {key} in {line}"))
}

fn parsed_generated(stdout: &str) -> Vec<Generated> {
    stdout
        .lines()
        .filter_map(|line| line.find(CASE_MARKER).map(|start| &line[start..]))
        .map(|line| Generated {
            name: field(line, "name").to_owned(),
            tokenizer_match: field(line, "tokenizer_match") == "true",
            ids: field(line, "ids")
                .split(',')
                .filter(|id| !id.is_empty())
                .map(|id| id.parse().expect("id"))
                .collect(),
        })
        .collect()
}

fn peak_of(stdout: &str) -> u64 {
    let line = stdout
        .lines()
        .find_map(|line| line.find(RSS_MARKER).map(|start| &line[start..]))
        .expect("child printed an rss line");
    field(line, "bytes").parse().expect("rss bytes")
}

fn divergences(model: &Model, generated: &[Generated]) -> Vec<String> {
    let cases = cases_of(model);
    generated
        .iter()
        .zip(&cases)
        .filter_map(|(item, case)| {
            divergence_after_stop(&case.expected_ids, &item.ids, model.stop_ids).map(|index| {
                format!(
                    "{}: first divergence at {index}; llama {:?}, directory {:?}",
                    case.name, case.expected_ids, item.ids
                )
            })
        })
        .collect()
}

fn run_child(model: &Model, role: &str) {
    let directory = PathBuf::from(std::env::var(DIRECTORY_ENV).expect("child directory"));
    match role {
        "export" => export_directory(model, &directory),
        "gguf" => generate_from_gguf(model),
        _ => generate_from_directory(model, &directory),
    }
}

fn has_shards(directory: &Path) -> bool {
    directory.is_dir() && !shard_paths(directory).is_empty()
}

fn exported_directory(test_name: &str) -> (Option<tempfile::TempDir>, PathBuf) {
    match std::env::var(REUSE_ENV) {
        Ok(reused) => {
            let reused = PathBuf::from(reused);
            if !has_shards(&reused) {
                spawn(test_name, "export", &reused);
            }
            (None, reused)
        }
        Err(_) => {
            let temporary = tempfile::tempdir().expect("export directory");
            spawn(test_name, "export", temporary.path());
            let path = temporary.path().to_path_buf();
            (Some(temporary), path)
        }
    }
}

fn plain_text_tokenizer_failures(model: &Model, generated: &[Generated]) -> Vec<String> {
    cases_of(model)
        .iter()
        .zip(generated)
        .filter(|(case, item)| !case.prompt.contains("<|") && !item.tokenizer_match)
        .map(|(case, _)| {
            format!(
                "{}: the directory tokenizer disagrees with the recorded prompt ids",
                case.name
            )
        })
        .collect()
}

fn summarize(model: &Model, hf_peak: u64, gguf_peak: u64, generated: &[Generated]) {
    let ratio = hf_peak as f64 / gguf_peak as f64;
    eprintln!(
        "{} hf_peak={hf_peak} gguf_peak={gguf_peak} ratio={ratio:.4} cases={}",
        model.name,
        generated.len()
    );
    for (case, item) in cases_of(model).iter().zip(generated) {
        let found = divergence_after_stop(&case.expected_ids, &item.ids, model.stop_ids);
        eprintln!(
            "{} case={} expected_len={} directory_len={} divergence={found:?} tokenizer_match={}",
            model.name,
            case.name,
            case.expected_ids.len(),
            item.ids.len(),
            item.tokenizer_match
        );
    }
}

fn round_trip(model: &Model, test_name: &str) {
    if let Ok(role) = std::env::var(CHILD_ENV) {
        return run_child(model, &role);
    }
    let (_keep, directory) = exported_directory(test_name);
    let hf_out = spawn(test_name, "hf", &directory);
    let gguf_out = spawn(test_name, "gguf", &directory);
    let generated = parsed_generated(&hf_out);
    let (hf_peak, gguf_peak) = (peak_of(&hf_out), peak_of(&gguf_out));
    summarize(model, hf_peak, gguf_peak, &generated);
    assert_eq!(
        generated.len(),
        cases_of(model).len(),
        "the directory decode must cover every recorded prompt"
    );
    assert_eq!(
        divergences(model, &parsed_generated(&gguf_out)),
        Vec::<String>::new(),
        "the gguf reference decode"
    );
    assert_eq!(
        divergences(model, &generated),
        Vec::<String>::new(),
        "the directory decode"
    );
    assert_eq!(
        plain_text_tokenizer_failures(model, &generated),
        Vec::<String>::new()
    );
    let ratio = hf_peak as f64 / gguf_peak as f64;
    assert!(
        ratio <= HF_OVER_GGUF_PEAK_BOUND,
        "directory peak {hf_peak} is {ratio:.3}x the gguf peak {gguf_peak}"
    );
}

#[test]
fn mlx_round_trip_granite_plain_export_generates_the_llama_ids() {
    round_trip(
        &GRANITE,
        "mlx_round_trip_granite_plain_export_generates_the_llama_ids",
    );
}

#[test]
fn mlx_round_trip_e2b_plain_export_generates_the_llama_oracle_ids() {
    round_trip(
        &E2B,
        "mlx_round_trip_e2b_plain_export_generates_the_llama_oracle_ids",
    );
}

#[test]
fn mlx_round_trip_identity_control_recorded_ids_compared_to_themselves_pass() {
    for model in [&GRANITE, &E2B] {
        let cases = cases_of(model);

        let compared: Vec<Option<usize>> = cases
            .iter()
            .map(|case| {
                divergence_after_stop(
                    &case.expected_ids,
                    &case.expected_ids.clone(),
                    model.stop_ids,
                )
            })
            .collect();

        assert!(!cases.is_empty(), "{} fixture holds no records", model.name);
        assert!(
            compared.iter().all(Option::is_none),
            "{}: {compared:?}",
            model.name
        );
    }
}

#[test]
fn mlx_round_trip_degenerate_control_one_altered_token_is_located() {
    for model in [&GRANITE, &E2B] {
        let case = &cases_of(model)[0];
        let mut altered = case.expected_ids.clone();
        altered[1] += 1;

        let found = divergence_after_stop(&case.expected_ids, &altered, model.stop_ids);

        assert_eq!(found, Some(1), "{}", model.name);
    }
}
