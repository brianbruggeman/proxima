//! Needle-in-a-haystack recall harness for long-context work
//! (`proxima-tensor/specs/long-context/SPEC.md` R16a1-R16c3, AC13b-AC21).
//!
//! Usage:
//! `cargo run -p proxima-model-interop --release --example long_context_niah --features std,metal -- \
//!   --model <gguf> --ctx N --needles K [--kv f32,f16,q8_0] \
//!   [--rope-scaling gguf,none,yarn:F:ORIG] [--context-extrapolate] [--control] \
//!   [--seed N] [--prefill-chunk N]`
//!
//! One prompt is built from a checked-in Project Gutenberg novel with K
//! seeded needles ("The special magic number for <noun> is <number>.") at
//! depths `(i + 0.5) * N / K`, and asking for every noun's number. That same
//! prompt string goes to each proxima arm (one per `--kv` x `--rope-scaling`
//! pair) and to Ollama (`raw`, greedy, `num_ctx = --ctx`), and one exact-match
//! scorer counts the numbers found. `--control` inserts no needles.
//!
//! `kv_bytes` prices the checkpoint's own per-layer KV layout
//! (`Architecture::kv_layers`, window included) at the arm's element size.
//! `peak_metal_bytes` is the largest `MTLDevice.currentAllocatedSize` sampled
//! after load, at the prefill boundary and at every generated token event.
//! Speculation runs at the production default: a verify evaluation delivers
//! several token events after one evaluation, so samples land per evaluation. A KV type
//! or scaling the runtime refuses fails that arm with the runtime's own error.

#[path = "long_context_niah/haystack.rs"]
mod haystack;
#[path = "long_context_niah/ollama.rs"]
mod ollama;
#[path = "long_context_niah/scoring.rs"]
mod scoring;

use std::env;
use std::fs;
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use memmap2::Mmap;
use proxima_gguf::{GgmlType, parse_complete};
use proxima_model_interop::{
    ArchitectureRegistry, ContextLength, GPU_LAYERS_ALL, InteropError, LoadedModel,
    PromptCacheConfig, RopeScaling, ServingConfig,
};
use proxima_tokenizer::gguf::vocab_from_metadata;
use proxima_tokenizer::{TokenizerError, encode_with_bos_eos};

use haystack::Case;

const DEFAULT_PREFILL_CHUNK_POSITIONS: usize = 2048;
const Q8_0_BLOCK_ELEMENTS: u64 = 32;
const Q8_0_BLOCK_BYTES: u64 = 34;

type KvLayer = (u32, u32, Option<u32>);

#[derive(Debug, thiserror::Error)]
enum NiahError {
    #[error(
        "usage: long_context_niah --model <gguf> --ctx N --needles K [--kv f32,f16,q8_0] \
         [--rope-scaling gguf,none,yarn:F:ORIG] [--context-extrapolate] [--control] \
         [--seed N] [--prefill-chunk N]: {0}"
    )]
    Usage(String),
    #[error("{context}: {source}")]
    Io {
        context: String,
        source: std::io::Error,
    },
    #[error("interop: {0}")]
    Interop(#[from] InteropError),
    #[error("tokenizer: {0}")]
    Tokenizer(#[from] TokenizerError),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("haystack source has no story start marker")]
    HaystackMarker,
    #[error("haystack holds {available} tokens, {requested} requested")]
    HaystackTooShort { requested: usize, available: usize },
    #[error("needle count {requested} outside 1..={available}")]
    NeedleCount { requested: usize, available: usize },
    #[error("context {context} cannot hold the needles, question and answer ({overhead} tokens)")]
    ContextTooSmall { context: usize, overhead: usize },
    #[error("prompt {prompt_tokens} + answer {max_new} tokens exceeds context {context}")]
    PromptExceedsContext {
        prompt_tokens: usize,
        max_new: usize,
        context: usize,
    },
    #[error("kv type {0:?} has no element size in this harness")]
    UnsupportedKv(GgmlType),
    #[error("no ollama manifest names blob {0}")]
    OllamaModelUnknown(String),
    #[error("{0}")]
    Failed(String),
}

impl NiahError {
    fn io(context: &str, source: std::io::Error) -> Self {
        Self::Io {
            context: context.to_string(),
            source,
        }
    }
}

struct Options {
    model: PathBuf,
    context: u32,
    needles: usize,
    kv_types: Vec<GgmlType>,
    rope_scalings: Vec<Option<RopeScaling>>,
    extrapolate: bool,
    control: bool,
    seed: Option<u64>,
    prefill_chunk_positions: usize,
}

impl Options {
    fn context_length(&self) -> ContextLength {
        if self.extrapolate {
            ContextLength::Extrapolate(self.context)
        } else {
            ContextLength::Within(self.context)
        }
    }
}

fn usage(reason: impl Into<String>) -> NiahError {
    NiahError::Usage(reason.into())
}

fn parse_number<T: std::str::FromStr>(flag: &str, value: &str) -> Result<T, NiahError> {
    value
        .parse()
        .map_err(|_| usage(format!("{flag} takes a number, got {value:?}")))
}

fn parse_kv_type(name: &str) -> Result<GgmlType, NiahError> {
    match name {
        "f32" => Ok(GgmlType::F32),
        "f16" => Ok(GgmlType::F16),
        "q8_0" => Ok(GgmlType::Q8_0),
        other => Err(usage(format!("--kv knows f32, f16, q8_0; got {other:?}"))),
    }
}

fn kv_label(kv: GgmlType) -> &'static str {
    match kv {
        GgmlType::F32 => "f32",
        GgmlType::F16 => "f16",
        GgmlType::Q8_0 => "q8_0",
        _ => "unsupported",
    }
}

fn parse_rope_scaling(spec: &str) -> Result<Option<RopeScaling>, NiahError> {
    let parts: Vec<&str> = spec.split(':').collect();
    match parts.as_slice() {
        ["gguf"] => Ok(None),
        ["none"] => Ok(Some(RopeScaling::None)),
        ["yarn", factor, original] => Ok(Some(RopeScaling::yarn(
            parse_number("--rope-scaling yarn factor", factor)?,
            parse_number("--rope-scaling yarn original", original)?,
        ))),
        _ => Err(usage(format!(
            "--rope-scaling knows gguf, none, yarn:F:ORIG; got {spec:?}"
        ))),
    }
}

fn rope_label(scaling: Option<RopeScaling>) -> String {
    match scaling {
        None => "gguf".to_string(),
        Some(RopeScaling::None) => "none".to_string(),
        Some(RopeScaling::Linear { factor }) => format!("linear:{factor}"),
        Some(RopeScaling::Yarn {
            factor,
            original_context,
            ..
        }) => format!("yarn:{factor}:{original_context}"),
    }
}

fn parse_list<T>(
    value: &str,
    parse: impl Fn(&str) -> Result<T, NiahError>,
) -> Result<Vec<T>, NiahError> {
    value.split(',').map(parse).collect()
}

fn parse_options(mut args: impl Iterator<Item = String>) -> Result<Options, NiahError> {
    let mut model = None;
    let mut context = None;
    let mut needles = None;
    let mut kv_types = vec![GgmlType::F32];
    let mut rope_scalings = vec![None];
    let mut extrapolate = false;
    let mut control = false;
    let mut seed = None;
    let mut prefill_chunk_positions = DEFAULT_PREFILL_CHUNK_POSITIONS;
    while let Some(flag) = args.next() {
        if flag == "--context-extrapolate" || flag == "--allow-extrapolation" {
            extrapolate = true;
            continue;
        }
        if flag == "--control" {
            control = true;
            continue;
        }
        let value = args
            .next()
            .ok_or_else(|| usage(format!("{flag} needs a value")))?;
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value)),
            "--ctx" => context = Some(parse_number(&flag, &value)?),
            "--needles" => needles = Some(parse_number(&flag, &value)?),
            "--kv" => kv_types = parse_list(&value, parse_kv_type)?,
            "--rope-scaling" => rope_scalings = parse_list(&value, parse_rope_scaling)?,
            "--seed" => seed = Some(parse_number(&flag, &value)?),
            "--prefill-chunk" => prefill_chunk_positions = parse_number(&flag, &value)?,
            other => return Err(usage(format!("unknown flag {other}"))),
        }
    }
    Ok(Options {
        model: model.ok_or_else(|| usage("--model is required"))?,
        context: context.ok_or_else(|| usage("--ctx is required"))?,
        needles: needles.ok_or_else(|| usage("--needles is required"))?,
        kv_types,
        rope_scalings,
        extrapolate,
        control,
        seed,
        prefill_chunk_positions,
    })
}

fn fresh_seed() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos());
    u64::try_from(nanos).unwrap_or(u64::MAX)
}

fn element_bytes(elements: u64, kv: GgmlType) -> Result<u64, NiahError> {
    match kv {
        GgmlType::F32 => Ok(elements * 4),
        GgmlType::F16 => Ok(elements * 2),
        GgmlType::Q8_0 => Ok(elements.div_ceil(Q8_0_BLOCK_ELEMENTS) * Q8_0_BLOCK_BYTES),
        other => Err(NiahError::UnsupportedKv(other)),
    }
}

/// K and V bytes for `context` positions over `layers`, the layout the
/// checkpoint's own architecture reports (own-KV layers only, sliding layers
/// capped at their window), stored as `kv`.
fn kv_cache_bytes(layers: &[KvLayer], context: u32, kv: GgmlType) -> Result<u64, NiahError> {
    layers
        .iter()
        .map(|&(kv_heads, head_dim, window)| {
            let rows = window.map_or(context, |window| window.min(context));
            let elements = u64::from(kv_heads) * u64::from(head_dim) * 2 * u64::from(rows);
            element_bytes(elements, kv)
        })
        .sum()
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_allocated_bytes() -> Option<u64> {
    omega::metal::current_allocated_size()
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn metal_allocated_bytes() -> Option<u64> {
    None
}

fn serving_config<'model>(
    options: &Options,
    model_path: &'model str,
    kv: GgmlType,
    rope_scaling: Option<RopeScaling>,
) -> ServingConfig<'model> {
    ServingConfig {
        model_path,
        gpu_layers: GPU_LAYERS_ALL,
        context_length: options.context_length(),
        rope_scaling,
        kv_cache_key_quant: kv,
        kv_cache_value_quant: kv,
        flash_attention: false,
        batch_size: 0,
        ubatch_size: 0,
        reasoning_budget: 0,
        prefill_chunk_positions: options.prefill_chunk_positions,
        prompt_cache: PromptCacheConfig::off(),
        ..ServingConfig::default()
    }
}

fn describe_bytes(bytes: Option<u64>) -> String {
    bytes.map_or_else(|| "unavailable".to_string(), |value| value.to_string())
}

fn run_proxima_arm(
    model: &LoadedModel,
    case: &Case,
    config: &ServingConfig,
    kv_bytes: u64,
) -> Result<String, NiahError> {
    let mut peak = metal_allocated_bytes();
    let (_ids, text, _stopped_by_eos) =
        model.generate_streaming(&case.prompt, case.max_new_tokens, *config, &mut |_event| {
            peak = peak.max(metal_allocated_bytes());
            ControlFlow::Continue(())
        })?;
    peak = peak.max(metal_allocated_bytes());
    let found = scoring::found_count(&text, &case.needles);
    Ok(format!(
        "found={found}/{} kv_bytes={kv_bytes} peak_metal_bytes={}",
        case.needles.len(),
        describe_bytes(peak)
    ))
}

fn run_ollama_arm(options: &Options, case: &Case) -> Result<(), NiahError> {
    let root = ollama::models_root()?;
    let name = ollama::name_for_blob(&root, &options.model)?;
    println!("ollama model={name}");
    let body = ollama::request_body(&name, &case.prompt, options.context, case.max_new_tokens);
    let text = ollama::answer(&body)?;
    println!(
        "ollama found={}/{}",
        scoring::found_count(&text, &case.needles),
        case.needles.len()
    );
    Ok(())
}

fn print_case(options: &Options, seed: u64, case: &Case) {
    println!(
        "seed={seed} ctx={} needles={} control={} prompt_tokens={} prompt_bytes={} max_new_tokens={} prefill_chunk_positions={} extrapolate={}",
        options.context,
        case.needles.len(),
        options.control,
        case.prompt_tokens,
        case.prompt.len(),
        case.max_new_tokens,
        options.prefill_chunk_positions,
        options.extrapolate
    );
    for (needle, placement) in case.needles.iter().zip(&case.placements) {
        println!(
            "needle noun={} number={} depth_target={} depth_measured={} word_index={}",
            needle.noun,
            needle.number,
            placement.depth_target,
            placement.depth_measured,
            placement.word_index
        );
    }
}

fn run_arms(options: &Options, model: &LoadedModel, case: &Case, layers: &[KvLayer]) -> usize {
    let model_path = options.model.display().to_string();
    let mut failures = 0;
    for &kv in &options.kv_types {
        for &rope in &options.rope_scalings {
            let label = format!("kv={} rope={}", kv_label(kv), rope_label(rope));
            let config = serving_config(options, &model_path, kv, rope);
            let arm = kv_cache_bytes(layers, options.context, kv)
                .and_then(|kv_bytes| run_proxima_arm(model, case, &config, kv_bytes));
            let (line, failed) = report_arm(arm);
            println!("{label} {line}");
            failures += usize::from(failed);
        }
    }
    failures
}

fn report_arm(arm: Result<String, NiahError>) -> (String, bool) {
    match arm {
        Ok(line) => (line, false),
        Err(NiahError::Interop(InteropError::UnsupportedServingConfig(reason))) => {
            (format!("rejected={reason}"), false)
        }
        Err(error) => (format!("error={error}"), true),
    }
}

fn run() -> Result<usize, NiahError> {
    let options = parse_options(env::args().skip(1))?;
    let seed = options.seed.unwrap_or_else(fresh_seed);
    let file = fs::File::open(&options.model)
        .map_err(|source| NiahError::io(&options.model.display().to_string(), source))?;
    // SAFETY: read-only mapping of a checkpoint no other process writes during this run.
    let mapping = unsafe { Mmap::map(&file) }
        .map_err(|source| NiahError::io(&options.model.display().to_string(), source))?;
    let parsed = parse_complete(&mapping)
        .map_err(|error| NiahError::Failed(format!("gguf parse: {error}")))?;
    let vocab = vocab_from_metadata(&parsed)?;
    let mut count = |text: &str| -> Result<usize, NiahError> {
        Ok(encode_with_bos_eos(text, &vocab, true, false)?.len())
    };
    let words = haystack::story_words()?;
    let context = usize::try_from(options.context)
        .map_err(|error| NiahError::Failed(format!("--ctx: {error}")))?;
    let case = haystack::build_case(
        &words,
        context,
        options.needles,
        options.control,
        seed,
        &mut count,
    )?;
    print_case(&options, seed, &case);
    let layers = ArchitectureRegistry::with_builtin()
        .resolve(&parsed)?
        .kv_layers(&parsed)?;
    let model = LoadedModel::load(&parsed, &mapping)?;
    let mut failures = run_arms(&options, &model, &case, &layers);
    if let Err(error) = run_ollama_arm(&options, &case) {
        failures += 1;
        println!("ollama error={error}");
    }
    Ok(failures)
}

fn main() -> ExitCode {
    match run() {
        Ok(0) => ExitCode::SUCCESS,
        Ok(failures) => {
            eprintln!("long_context_niah: {failures} arm(s) failed");
            ExitCode::FAILURE
        }
        Err(error) => {
            eprintln!("long_context_niah: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use haystack::{Needle, SOURCE};
    use proxima_tokenizer::Vocab;
    use sha2::{Digest, Sha256};

    const SOURCE_SHA256: &str = "d7a1d4c8d7427d526f8dc1fb89c4806c6f3276292190dc6fb1ff20a88b9bb7b5";
    const LICENSE: &str = include_str!("data/war_and_peace.LICENSE.txt");
    const FIXED_SEED: u64 = 20_260_929;
    const GEMMA4_TAG_PREFIX: &str = "gemma4:";

    fn piece_count(text: &str) -> Result<usize, NiahError> {
        Ok(text
            .split_whitespace()
            .map(|word| word.chars().count().div_ceil(4))
            .sum())
    }

    #[test]
    fn refused_serving_config_is_rejected_not_failed() {
        let refusal = NiahError::Interop(InteropError::UnsupportedServingConfig(
            "flash_attention=true is not served".to_string(),
        ));

        let (line, failed) = report_arm(Err(refusal));

        assert_eq!(line, "rejected=flash_attention=true is not served");
        assert!(!failed);
    }

    #[test]
    fn other_errors_still_count_as_failures() {
        let broken = NiahError::Failed("generation diverged".to_string());

        let (line, failed) = report_arm(Err(broken));

        assert_eq!(line, "error=generation diverged");
        assert!(failed);
    }

    #[test]
    fn completed_arm_is_not_a_failure() {
        let (line, failed) = report_arm(Ok("found=1/1".to_string()));

        assert_eq!(line, "found=1/1");
        assert!(!failed);
    }

    fn gemma4_vocab() -> Option<Vocab> {
        let root = ollama::models_root().ok()?;
        let entry = ollama::manifest_entries(&root)
            .ok()?
            .into_iter()
            .find(|entry| entry.name.starts_with(GEMMA4_TAG_PREFIX))?;
        let file = fs::File::open(root.join("blobs").join(&entry.blob)).ok()?;
        // SAFETY: read-only mapping of an Ollama blob nothing writes while the test runs.
        let mapping = unsafe { Mmap::map(&file) }.ok()?;
        let parsed = parse_complete(&mapping).ok()?;
        vocab_from_metadata(&parsed).ok()
    }

    fn small_case(context: usize, needle_count: usize, control: bool) -> Case {
        let words = haystack::story_words().expect("the story marker is in the checked-in text");
        haystack::build_case(
            &words,
            context,
            needle_count,
            control,
            FIXED_SEED,
            &mut piece_count,
        )
        .expect("a 2048-token window holds three needles")
    }

    #[test]
    fn niah_haystack_sha() {
        let digest = format!("{:x}", Sha256::digest(SOURCE.as_bytes()));

        assert_eq!(digest, SOURCE_SHA256);
        assert!(LICENSE.contains("THE FULL PROJECT GUTENBERG"));
        assert!(!SOURCE.contains("Project Gutenberg"));
    }

    #[test]
    fn niah_haystack_trim() {
        let words = haystack::story_words().expect("the story marker is in the checked-in text");

        let (kept, tokens) = haystack::trim_haystack(&words, 8192, &mut piece_count)
            .expect("the novel holds 8192 tokens");

        assert!((8110..=8274).contains(&tokens), "piece counter: {tokens}");
        assert!(tokens <= 8192);
        assert_eq!(
            piece_count(&words[..kept].join(" ")).expect("counting cannot fail"),
            tokens
        );
        let Some(vocab) = gemma4_vocab() else {
            eprintln!(
                "niah_haystack_trim: no gemma4 blob in the ollama manifests, real-tokenizer half skipped"
            );
            return;
        };
        let mut gemma_count = |text: &str| -> Result<usize, NiahError> {
            Ok(encode_with_bos_eos(text, &vocab, true, false)?.len())
        };
        let (_, gemma_tokens) = haystack::trim_haystack(&words, 8192, &mut gemma_count)
            .expect("the novel holds 8192 gemma4 tokens");
        assert!(
            (8110..=8274).contains(&gemma_tokens),
            "gemma4 tokenizer: {gemma_tokens}"
        );
    }

    #[test]
    fn niah_needle_depths() {
        let words = haystack::story_words().expect("the story marker is in the checked-in text");
        let needles = haystack::generate_needles(FIXED_SEED, 10).expect("10 nouns exist");
        let needle_tokens = needles
            .iter()
            .map(|needle| piece_count(&needle.sentence()).expect("counting cannot fail"))
            .max()
            .expect("10 needles");
        let (kept, _) = haystack::trim_haystack(&words, 8192, &mut piece_count)
            .expect("the novel holds 8192 tokens");

        let placements =
            haystack::place_needles(&words[..kept], 8192, needle_tokens, 10, &mut piece_count)
                .expect("placement converges");

        assert_eq!(placements.len(), 10);
        for (index, placement) in placements.iter().enumerate() {
            let expected_tenths = (2 * index + 1) * 4096;
            assert!(
                (placement.depth_measured * 10).abs_diff(expected_tenths) <= needle_tokens * 10,
                "needle {index}: measured {} vs {}.{}",
                placement.depth_measured,
                expected_tenths / 10,
                expected_tenths % 10
            );
        }
        assert_eq!(
            needles,
            haystack::generate_needles(FIXED_SEED, 10).expect("10 nouns exist")
        );
        assert_ne!(
            needles,
            haystack::generate_needles(FIXED_SEED + 1, 10).expect("10 nouns exist")
        );
        assert!(
            needles
                .iter()
                .all(|needle| (1_000_000..10_000_000).contains(&needle.number))
        );
    }

    #[test]
    fn niah_needle_sentences_reach_the_prompt_unless_control() {
        let with_needles = small_case(2048, 3, false);
        let control = small_case(2048, 3, true);

        assert!(
            with_needles.needles.iter().all(|needle| with_needles
                .prompt
                .matches(&needle.sentence())
                .count()
                == 1)
        );
        assert!(
            control
                .needles
                .iter()
                .all(|needle| !control.prompt.contains(&needle.number.to_string()))
        );
    }

    #[test]
    fn niah_ollama_request_prompt() {
        let case = small_case(2048, 3, false);

        let body =
            ollama::request_body("gemma4:e2b-it-qat", &case.prompt, 2048, case.max_new_tokens);
        let wire: serde_json::Value =
            serde_json::from_str(&body.to_string()).expect("the body is valid json");

        let sent = wire["prompt"].as_str().expect("prompt is a string");
        assert_eq!(sent.as_bytes(), case.prompt.as_bytes());
        assert!(case.prompt.chars().any(|character| !character.is_ascii()));
    }

    #[test]
    fn niah_ollama_request_options() {
        let body = ollama::request_body("qwen3:8b", "prompt", 131_072, 192);

        let wire: serde_json::Value =
            serde_json::from_str(&body.to_string()).expect("the body is valid json");

        assert_eq!(wire["options"]["num_ctx"], 131_072);
        assert_eq!(wire["options"]["temperature"], 0);
        assert_eq!(wire["raw"], true);
        assert_eq!(wire["stream"], false);
    }

    #[proxima::test]
    #[case::exact_needle("4830912", 1)]
    #[case::needle_inside_prose("The special magic number for marmot is 4830912.", 1)]
    #[case::wrong_digits("4830921", 0)]
    async fn niah_scoring_shared(#[case] response: &'static str, #[case] expected_found: usize) {
        let needle = Needle {
            noun: "marmot",
            number: 4_830_912,
        };

        let found = scoring::found_count(response, &[needle]);

        assert_eq!(found, expected_found);
    }

    #[test]
    fn niah_scoring_rejects_digit_runs_that_merely_contain_the_number() {
        assert!(!scoring::is_exact_match("48309120", "4830912"));
        assert!(!scoring::is_exact_match("14830912", "4830912"));
        assert!(scoring::is_exact_match(
            "marmot: 4830912\nyak: 1",
            "4830912"
        ));
    }

    #[test]
    fn niah_kv_bytes_matches_the_spec_table() {
        let gemma4: Vec<KvLayer> =
            [vec![(1, 256, Some(512)); 12], vec![(1, 512, None); 3]].concat();
        let qwen36: Vec<KvLayer> = vec![(2, 256, None); 10];

        let gemma4_f32 = kv_cache_bytes(&gemma4, 131_072, GgmlType::F32).expect("f32 has a size");
        let gemma4_f16 = kv_cache_bytes(&gemma4, 131_072, GgmlType::F16).expect("f16 has a size");
        let gemma4_q8 = kv_cache_bytes(&gemma4, 131_072, GgmlType::Q8_0).expect("q8_0 has a size");
        let qwen36_f16 = kv_cache_bytes(&qwen36, 262_144, GgmlType::F16).expect("f16 has a size");

        assert_eq!(gemma4_f32, 1_623_195_648);
        assert_eq!(gemma4_f16, 811_597_824);
        assert_eq!(gemma4_q8, 431_161_344);
        assert_eq!(qwen36_f16, 5_368_709_120);
    }

    #[test]
    fn niah_ollama_manifest_lookup() {
        let root = tempfile::tempdir().expect("a temp dir is available");
        let tag_dir = root
            .path()
            .join("manifests/registry.ollama.ai/library/gemma4");
        fs::create_dir_all(&tag_dir).expect("the manifest tree is creatable");
        let manifest = serde_json::json!({
            "schemaVersion": 2,
            "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "layers": [
                {"mediaType": "application/vnd.ollama.image.model",
                 "digest": "sha256:3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd",
                 "size": 7162394016_u64},
                {"mediaType": "application/vnd.ollama.image.license",
                 "digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
                 "size": 11357},
            ],
        });
        fs::write(tag_dir.join("e2b-it-qat"), manifest.to_string())
            .expect("the manifest is writable");
        let blobs = root.path().join("blobs");
        fs::create_dir_all(&blobs).expect("the blob dir is creatable");
        let blob =
            blobs.join("sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd");
        fs::write(&blob, b"GGUF").expect("the blob is writable");

        let name = ollama::name_for_blob(root.path(), &blob).expect("the digest is in a manifest");
        let unknown = ollama::name_for_blob(root.path(), &blobs);

        assert_eq!(name, "gemma4:e2b-it-qat");
        assert!(unknown.is_err());
    }
}
