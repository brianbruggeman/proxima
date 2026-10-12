use std::ops::ControlFlow;
use std::path::{Path, PathBuf};

use proxima_gguf::parse_complete;
use proxima_model_interop::{LoadedModel, PromptCacheConfig, ServingConfig};
use serde::Deserialize;

use proxima_gguf::GgmlType;

use crate::reencode::{Target, reencode};

#[derive(Deserialize)]
pub struct Case {
    pub prompt_ids: Vec<u32>,
    pub generated_ids: Vec<u32>,
}

pub fn cases(model_directory: &str) -> Vec<Case> {
    ["llama_ids.json", "long_prompt_llama_ids.json"]
        .into_iter()
        .flat_map(|name| {
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/llama-parity")
                .join(model_directory)
                .join(name);
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            serde_json::from_str::<Vec<Case>>(&text)
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
        })
        .collect()
}

pub fn agree(cases: &[Case], generations: &[Vec<u32>]) -> (usize, usize) {
    let total = cases.iter().map(|case| case.generated_ids.len()).sum();
    let agreeing = cases
        .iter()
        .zip(generations)
        .map(|(case, generated)| {
            case.generated_ids
                .iter()
                .enumerate()
                .filter(|(position, id)| generated.get(*position) == Some(*id))
                .count()
        })
        .sum();
    (agreeing, total)
}

pub fn recorded_generations(cases: &[Case]) -> Vec<Vec<u32>> {
    cases
        .iter()
        .map(|case| case.generated_ids.clone())
        .collect()
}

pub fn incremented_generations(cases: &[Case]) -> Vec<Vec<u32>> {
    cases
        .iter()
        .map(|case| case.generated_ids.iter().map(|id| id + 1).collect())
        .collect()
}

/// Loads `env`'s checkpoint, re-encodes its block weights into the family, runs free-running
/// greedy generation from each recorded prompt and prints `reencoded=<n>` and `agree=<n> of <N>`.
pub fn run(
    environment: &str,
    model_directory: &str,
    target_of: fn(GgmlType) -> Option<Target>,
) -> f64 {
    let path = std::env::var(environment)
        .unwrap_or_else(|_| panic!("set {environment} to the {model_directory} checkpoint path"));
    assert!(
        Path::new(&path).exists(),
        "no checkpoint at {path}: set {environment}"
    );
    let file = std::fs::File::open(&path).expect("open the checkpoint");
    // the checkpoint is not written while this process reads it
    let mapping = unsafe { memmap2::Mmap::map(&file) }.expect("map the checkpoint");
    let parsed = parse_complete(&mapping).expect("parse the checkpoint header");
    let rewritten = reencode(&parsed, &mapping, target_of);
    let reencoded: usize = rewritten.counts.iter().map(|(_, count)| count).sum();
    println!("reencoded={reencoded} by_target={:?}", rewritten.counts);
    assert!(
        reencoded > 0,
        "no tensor reached the family: the run exercised none of its codecs"
    );
    let cases = cases(model_directory);
    let reparsed = parse_complete(&rewritten.bytes).expect("parse the re-encoded model");
    let model = LoadedModel::load(&reparsed, &rewritten.bytes).expect("load the re-encoded model");
    let config = ServingConfig {
        prompt_cache: PromptCacheConfig::off(),
        ubatch_size: 0,
        ..ServingConfig::default()
    };
    let generations: Vec<Vec<u32>> = cases
        .iter()
        .map(|case| {
            model
                .generate_from_ids(
                    &case.prompt_ids,
                    case.generated_ids.len(),
                    &config,
                    &mut |_event| ControlFlow::Continue(()),
                )
                .expect("the re-encoded model generates")
                .0
        })
        .collect();
    for (position, (case, generated)) in cases.iter().zip(&generations).enumerate() {
        let first = case
            .generated_ids
            .iter()
            .zip(generated)
            .position(|(want, got)| want != got);
        println!(
            "case={position} positions={} first_divergence={first:?}",
            case.generated_ids.len()
        );
    }
    let (agreeing, total) = agree(&cases, &generations);
    let rate = agreeing as f64 / total as f64;
    println!("agree={agreeing} of {total} rate={rate:.4}");
    rate
}
