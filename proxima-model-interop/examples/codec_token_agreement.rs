//! Positional token agreement of fp6 and shipped generations against recorded llama ids:
//! `codec_token_agreement --fixtures <a.json,b.json> --allowed-gap-rate <rate>
//! --cell <name>=<fp6.gguf>,<shipped.gguf>,<ids-dir> ...`.
//!
//! Composes `parse_complete` (the checkpoint directory), `LoadedModel::load` and
//! `LoadedModel::generate_from_ids` (free-running greedy generation from the recorded
//! `prompt_ids`, no teacher forcing, the whole prompt in one evaluation like
//! `prefill_width_parity_with_llama`). Position `i` of a case matches when the generated id
//! equals the recorded `generated_ids[i]`; a generation that ends early misses the positions
//! it never reached. Each cell prints three summary lines and one detail line per case:
//! `cell=<name> codec=<fp6|shipped> match=<n> of <positions>` and
//! `cell=<name> bound allowed_gap=<floor(rate x positions)> gap=<shipped - fp6, floored at 0>
//! within=<bool>`. The detail lines list both directions of every flip.
//! The backend is `ServingConfig::default().gpu_layers`: Metal on a `metal` build, CPU otherwise.

use std::collections::BTreeMap;
use std::fs::{File, read_to_string};
use std::io;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use memmap2::Mmap;
use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::{GgufError, parse_complete};
use proxima_model_interop::{InteropError, LoadedModel, PromptCacheConfig, ServingConfig};
use serde::Deserialize;
use thiserror::Error;

const USAGE: &str = "usage: codec_token_agreement --fixtures <a.json,b.json> --allowed-gap-rate <rate> --cell <name>=<fp6.gguf>,<shipped.gguf>,<ids-dir> ...";

#[derive(Debug, Error)]
enum AgreementError {
    #[error("{0}")]
    Usage(String),
    #[error("{path}: {source}")]
    Io { path: PathBuf, source: io::Error },
    #[error("{path}: {source}")]
    Fixture {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("{path}: {source}")]
    Gguf { path: PathBuf, source: GgufError },
    #[error("{path}: {source}")]
    Model { path: PathBuf, source: InteropError },
}

#[derive(Deserialize)]
struct Record {
    prompt_ids: Vec<u32>,
    generated_ids: Vec<u32>,
}

struct Case {
    fixture: String,
    record: Record,
}

struct Cell {
    name: String,
    fp6: PathBuf,
    shipped: PathBuf,
    ids_directory: PathBuf,
}

struct Flags {
    fixtures: Vec<String>,
    allowed_gap_rate: f64,
    cells: Vec<Cell>,
}

fn usage(message: &str) -> AgreementError {
    AgreementError::Usage(format!("{message}\n{USAGE}"))
}

fn parse_cell(raw: &str) -> Result<Cell, AgreementError> {
    let (name, paths) = raw.split_once('=').ok_or_else(|| {
        usage(&format!(
            "cell {raw:?} is not <name>=<fp6>,<shipped>,<ids-dir>"
        ))
    })?;
    match paths.split(',').collect::<Vec<_>>()[..] {
        [fp6, shipped, ids_directory] => Ok(Cell {
            name: name.to_owned(),
            fp6: PathBuf::from(fp6),
            shipped: PathBuf::from(shipped),
            ids_directory: PathBuf::from(ids_directory),
        }),
        _ => Err(usage(&format!("cell {raw:?} does not name three paths"))),
    }
}

fn parse_rate(raw: &str) -> Result<f64, AgreementError> {
    raw.parse::<f64>()
        .ok()
        .filter(|rate| rate.is_finite() && (0.0..=1.0).contains(rate))
        .ok_or_else(|| {
            usage(&format!(
                "allowed-gap-rate {raw:?} is not a number in 0..=1"
            ))
        })
}

fn read_flags(mut arguments: impl Iterator<Item = String>) -> Result<Flags, AgreementError> {
    let mut fixtures = Vec::new();
    let mut allowed_gap_rate = None;
    let mut cells = Vec::new();
    while let Some(flag) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| usage(&format!("{flag} has no value")))?;
        match flag.as_str() {
            "--fixtures" => fixtures = value.split(',').map(str::to_owned).collect(),
            "--allowed-gap-rate" => allowed_gap_rate = Some(parse_rate(&value)?),
            "--cell" => cells.push(parse_cell(&value)?),
            other => return Err(usage(&format!("unknown flag {other}"))),
        }
    }
    let allowed_gap_rate =
        allowed_gap_rate.ok_or_else(|| usage("--allowed-gap-rate is required"))?;
    if fixtures.is_empty() || cells.is_empty() {
        return Err(usage("--fixtures and at least one --cell are required"));
    }
    Ok(Flags {
        fixtures,
        allowed_gap_rate,
        cells,
    })
}

fn read_case_file(path: &Path, fixture: &str) -> Result<Vec<Case>, AgreementError> {
    let text = read_to_string(path).map_err(|source| AgreementError::Io {
        path: path.to_owned(),
        source,
    })?;
    let records: Vec<Record> =
        serde_json::from_str(&text).map_err(|source| AgreementError::Fixture {
            path: path.to_owned(),
            source,
        })?;
    Ok(records
        .into_iter()
        .map(|record| Case {
            fixture: fixture.to_owned(),
            record,
        })
        .collect())
}

fn read_cases(cell: &Cell, fixtures: &[String]) -> Result<Vec<Case>, AgreementError> {
    let per_fixture = fixtures
        .iter()
        .map(|fixture| read_case_file(&cell.ids_directory.join(fixture), fixture))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(per_fixture.into_iter().flatten().collect())
}

fn serving_config() -> ServingConfig<'static> {
    ServingConfig {
        prompt_cache: PromptCacheConfig::off(),
        ubatch_size: 0,
        ..ServingConfig::default()
    }
}

fn position_hits(expected: &[u32], generated: &[u32]) -> Vec<bool> {
    expected
        .iter()
        .enumerate()
        .map(|(position, id)| generated.get(position) == Some(id))
        .collect()
}

fn hits_of(cases: &[Case], generations: &[Vec<u32>]) -> Vec<Vec<bool>> {
    cases
        .iter()
        .zip(generations)
        .map(|(case, generated)| position_hits(&case.record.generated_ids, generated))
        .collect()
}

fn generations_for(path: &Path, cases: &[Case]) -> Result<Vec<Vec<u32>>, AgreementError> {
    let model_error = |source| AgreementError::Model {
        path: path.to_owned(),
        source,
    };
    let file = File::open(path).map_err(|source| AgreementError::Io {
        path: path.to_owned(),
        source,
    })?;
    // the checkpoint is not written while this process reads it
    let mapping = unsafe { Mmap::map(&file) }.map_err(|source| AgreementError::Io {
        path: path.to_owned(),
        source,
    })?;
    let parsed = parse_complete(&mapping).map_err(|source| AgreementError::Gguf {
        path: path.to_owned(),
        source,
    })?;
    print_tensor_types(path, &parsed);
    let model = LoadedModel::load(&parsed, &mapping).map_err(model_error)?;
    let config = serving_config();
    cases
        .iter()
        .map(|case| {
            let (generated, _text, _stopped) = model
                .generate_from_ids(
                    &case.record.prompt_ids,
                    case.record.generated_ids.len(),
                    &config,
                    &mut |_event| ControlFlow::Continue(()),
                )
                .map_err(model_error)?;
            Ok(generated)
        })
        .collect()
}

fn print_tensor_types(path: &Path, parsed: &ParsedGguf) {
    let mut histogram = BTreeMap::<String, usize>::new();
    for tensor in &parsed.tensors {
        *histogram
            .entry(format!("{:?}", tensor.ggml_type))
            .or_default() += 1;
    }
    let listed = histogram
        .iter()
        .map(|(kind, count)| format!("{kind}:{count}"))
        .collect::<Vec<_>>()
        .join(",");
    println!("file={} tensor_types={listed}", path.display());
}

fn count(value: usize) -> Result<u32, AgreementError> {
    u32::try_from(value).map_err(|_| usage(&format!("{value} positions do not fit u32")))
}

fn matched(hits: &[Vec<bool>]) -> Result<u32, AgreementError> {
    count(hits.iter().flatten().filter(|hit| **hit).count())
}

fn allowed_gap(rate: f64, positions: u32) -> u32 {
    let ceiling = rate * f64::from(positions);
    (0..=positions)
        .rev()
        .find(|gap| f64::from(*gap) <= ceiling)
        .unwrap_or(0)
}

fn first_divergence(expected: &[u32], generated: &[u32], hits: &[bool]) -> String {
    hits.iter().position(|hit| !hit).map_or_else(
        || "none".to_owned(),
        |position| {
            let got = generated
                .get(position)
                .map_or_else(|| "none".to_owned(), ToString::to_string);
            format!("{position}:llama={}:got={got}", expected[position])
        },
    )
}

fn positions_where(left: &[bool], right: &[bool], left_hit: bool) -> Vec<usize> {
    left.iter()
        .zip(right)
        .enumerate()
        .filter(|(_, (left_value, right_value))| {
            **left_value == left_hit && **right_value != left_hit
        })
        .map(|(position, _)| position)
        .collect()
}

struct Observed {
    generations: Vec<Vec<u32>>,
    hits: Vec<Vec<bool>>,
}

fn observe(path: &Path, cases: &[Case]) -> Result<Observed, AgreementError> {
    let generations = generations_for(path, cases)?;
    let hits = hits_of(cases, &generations);
    Ok(Observed { generations, hits })
}

fn print_case(cell: &Cell, index: usize, case: &Case, fp6: &Observed, shipped: &Observed) {
    let expected = &case.record.generated_ids;
    println!(
        "detail cell={} fixture={} positions={} fp6_first_divergence={} shipped_first_divergence={} fp6_lost={:?} fp6_gained={:?} fp6_ids={:?}",
        cell.name,
        case.fixture,
        expected.len(),
        first_divergence(expected, &fp6.generations[index], &fp6.hits[index]),
        first_divergence(expected, &shipped.generations[index], &shipped.hits[index]),
        positions_where(&shipped.hits[index], &fp6.hits[index], true),
        positions_where(&fp6.hits[index], &shipped.hits[index], true),
        fp6.generations[index],
    );
}

fn report_cell(cell: &Cell, flags: &Flags) -> Result<(), AgreementError> {
    let cases = read_cases(cell, &flags.fixtures)?;
    let fp6 = observe(&cell.fp6, &cases)?;
    let shipped = observe(&cell.shipped, &cases)?;
    for (index, case) in cases.iter().enumerate() {
        print_case(cell, index, case, &fp6, &shipped);
    }
    let positions = count(
        cases
            .iter()
            .map(|case| case.record.generated_ids.len())
            .sum(),
    )?;
    let (fp6_match, shipped_match) = (matched(&fp6.hits)?, matched(&shipped.hits)?);
    let allowed = allowed_gap(flags.allowed_gap_rate, positions);
    let gap = shipped_match.saturating_sub(fp6_match);
    println!(
        "cell={} codec=fp6 match={fp6_match} of {positions}",
        cell.name
    );
    println!(
        "cell={} codec=shipped match={shipped_match} of {positions}",
        cell.name
    );
    println!(
        "cell={} bound allowed_gap={allowed} gap={gap} within={}",
        cell.name,
        gap <= allowed
    );
    Ok(())
}

fn run() -> Result<(), AgreementError> {
    let flags = read_flags(std::env::args().skip(1))?;
    println!("gpu_layers={}", serving_config().gpu_layers);
    flags
        .cells
        .iter()
        .try_for_each(|cell| report_cell(cell, &flags))
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
