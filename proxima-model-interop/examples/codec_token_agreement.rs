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
//!
//! `--teacher-forced` adds the distributional comparison (the free-running count stays a reported
//! line). The shipped model's free-running ids are the reference sequence; every candidate (fp6
//! and each `label=path` control appended to a cell) is fed that sequence through
//! `LoadedModel::teacher_forced_logits_from_ids`, and at every position the example records
//! top-1 agreement, `KL(reference || candidate)` over the full vocabulary, the reference's
//! top-1 minus top-2 logit margin, and the negative log-likelihood of the reference token under
//! each model. `--matched-control <label>` names the control fp6 is held against; the four
//! `relation=` lines print `holds=<bool>` for top-1 count, mean KL, NLL delta, and the identity
//! control reproducing the reference top-1 everywhere.

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
    // box: InteropError is >128 bytes under the metal feature (clippy result_large_err)
    Model {
        path: PathBuf,
        source: Box<InteropError>,
    },
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
    controls: Vec<(String, PathBuf)>,
}

struct Flags {
    fixtures: Vec<String>,
    allowed_gap_rate: f64,
    teacher_forced: bool,
    matched_control: Option<String>,
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
        [fp6, shipped, ids_directory, ref controls @ ..] => Ok(Cell {
            name: name.to_owned(),
            fp6: PathBuf::from(fp6),
            shipped: PathBuf::from(shipped),
            ids_directory: PathBuf::from(ids_directory),
            controls: controls
                .iter()
                .map(|control| parse_control(control))
                .collect::<Result<_, _>>()?,
        }),
        _ => Err(usage(&format!("cell {raw:?} does not name three paths"))),
    }
}

fn parse_control(raw: &str) -> Result<(String, PathBuf), AgreementError> {
    raw.split_once('=')
        .map(|(label, path)| (label.to_owned(), PathBuf::from(path)))
        .ok_or_else(|| usage(&format!("control {raw:?} is not <label>=<path>")))
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
    let mut teacher_forced = false;
    let mut matched_control = None;
    while let Some(flag) = arguments.next() {
        if flag == "--teacher-forced" {
            teacher_forced = true;
            continue;
        }
        let value = arguments
            .next()
            .ok_or_else(|| usage(&format!("{flag} has no value")))?;
        match flag.as_str() {
            "--fixtures" => fixtures = value.split(',').map(str::to_owned).collect(),
            "--allowed-gap-rate" => allowed_gap_rate = Some(parse_rate(&value)?),
            "--matched-control" => matched_control = Some(value),
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
        teacher_forced,
        matched_control,
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

struct ModelRun {
    generations: Vec<Vec<u32>>,
    passes: Vec<Vec<Vec<Vec<f32>>>>,
}

fn free_run(model: &LoadedModel, cases: &[Case]) -> Result<Vec<Vec<u32>>, InteropError> {
    let config = serving_config();
    cases
        .iter()
        .map(|case| {
            model
                .generate_from_ids(
                    &case.record.prompt_ids,
                    case.record.generated_ids.len(),
                    &config,
                    &mut |_event| ControlFlow::Continue(()),
                )
                .map(|(generated, _text, _stopped)| generated)
        })
        .collect()
}

fn forced_pass(
    model: &LoadedModel,
    cases: &[Case],
    forced: &[Vec<u32>],
) -> Result<Vec<Vec<Vec<f32>>>, InteropError> {
    let config = serving_config();
    cases
        .iter()
        .zip(forced)
        .map(|(case, ids)| {
            model.teacher_forced_logits_from_ids(&case.record.prompt_ids, ids, &config)
        })
        .collect()
}

fn run_model(
    path: &Path,
    cases: &[Case],
    forced: Option<&[Vec<u32>]>,
    pass_count: usize,
) -> Result<ModelRun, AgreementError> {
    with_model(path, |model| {
        let generations = free_run(model, cases)?;
        let sequence = forced.unwrap_or(&generations);
        let passes = (0..pass_count)
            .map(|_| forced_pass(model, cases, sequence))
            .collect::<Result<_, _>>()?;
        Ok(ModelRun {
            generations,
            passes,
        })
    })
}

fn with_model<T>(
    path: &Path,
    body: impl FnOnce(&LoadedModel) -> Result<T, InteropError>,
) -> Result<T, AgreementError> {
    let model_error = |source| AgreementError::Model {
        path: path.to_owned(),
        source: Box::new(source),
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
    body(&model).map_err(model_error)
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
    let generations = run_model(path, cases, None, 0)?.generations;
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

fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let peak = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let peak = f64::from(peak);
    let total: f64 = logits
        .iter()
        .map(|logit| (f64::from(*logit) - peak).exp())
        .sum();
    let log_total = peak + total.ln();
    logits
        .iter()
        .map(|logit| f64::from(*logit) - log_total)
        .collect()
}

fn top_two(logits: &[f32]) -> (usize, f64) {
    let (mut best, mut best_value, mut second_value) = (0, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (index, value) in logits.iter().copied().enumerate() {
        if value > best_value {
            (best, second_value, best_value) = (index, best_value, value);
        } else if value > second_value {
            second_value = value;
        }
    }
    (best, f64::from(best_value) - f64::from(second_value))
}

struct Position {
    top1_agrees: bool,
    kl: f64,
    margin: f64,
    nll_reference: f64,
    nll_candidate: f64,
}

fn score_position(reference: &[f32], candidate: &[f32], token: u32) -> Position {
    let (reference_log, candidate_log) = (log_softmax(reference), log_softmax(candidate));
    let kl = reference_log
        .iter()
        .zip(&candidate_log)
        .map(|(reference_value, candidate_value)| {
            reference_value.exp() * (reference_value - candidate_value)
        })
        .sum();
    let (reference_top, margin) = top_two(reference);
    Position {
        top1_agrees: top_two(candidate).0 == reference_top,
        kl,
        margin,
        nll_reference: -reference_log[token as usize],
        nll_candidate: -candidate_log[token as usize],
    }
}

struct Summary {
    positions: u32,
    top1: u32,
    kl_mean: f64,
    nll_reference: f64,
    nll_candidate: f64,
    free_match: u32,
}

impl Summary {
    fn nll_delta(&self) -> f64 {
        self.nll_candidate - self.nll_reference
    }
}

fn first_mismatch(reference: &[u32], generated: &[u32]) -> Option<usize> {
    (0..reference.len()).find(|position| generated.get(*position) != reference.get(*position))
}

fn print_position(cell: &Cell, label: &str, case: usize, index: usize, score: &Position) {
    println!(
        "position cell={} codec={label} case={case} index={index} top1_agrees={} kl={:.6} ref_margin={:.4} nll_ref={:.4} nll_candidate={:.4}",
        cell.name,
        score.top1_agrees,
        score.kl,
        score.margin,
        score.nll_reference,
        score.nll_candidate
    );
}

fn print_first_divergence(
    cell: &Cell,
    label: &str,
    case: usize,
    reference: &[Vec<f32>],
    reference_ids: &[u32],
    generated: &[u32],
) {
    let Some(index) = first_mismatch(reference_ids, generated) else {
        println!("firstdiv cell={} codec={label} case={case} none", cell.name);
        return;
    };
    let row = &reference[index];
    let (top, margin) = top_two(row);
    let chosen = generated.get(index).map(|id| *id as usize);
    let logit_gap = chosen.map_or(f64::NAN, |id| f64::from(row[top]) - f64::from(row[id]));
    println!(
        "firstdiv cell={} codec={label} case={case} index={index} ref_top1={top} candidate_token={} ref_margin={margin:.4} ref_logit_gap_to_candidate_token={logit_gap:.4}",
        cell.name,
        chosen.map_or_else(|| "none".to_owned(), |id| id.to_string()),
    );
}

fn summarize(
    cell: &Cell,
    label: &str,
    run: &ModelRun,
    pass: usize,
    reference: &ModelRun,
) -> Result<Summary, AgreementError> {
    let reference_ids = &reference.generations;
    let mut scores = Vec::new();
    for (case, ids) in reference_ids.iter().enumerate() {
        let (reference_rows, candidate_rows) =
            (&reference.passes[0][case], &run.passes[pass][case]);
        for (index, token) in ids.iter().enumerate() {
            let score = score_position(&reference_rows[index], &candidate_rows[index], *token);
            print_position(cell, label, case, index, &score);
            scores.push(score);
        }
        print_first_divergence(
            cell,
            label,
            case,
            reference_rows,
            ids,
            &run.generations[case],
        );
    }
    let positions = count(scores.len())?;
    let mean =
        |value: fn(&Position) -> f64| scores.iter().map(value).sum::<f64>() / f64::from(positions);
    let free_match = reference_ids
        .iter()
        .zip(&run.generations)
        .map(|(ids, generated)| {
            ids.iter()
                .zip(generated)
                .filter(|(left, right)| left == right)
                .count()
        })
        .sum();
    Ok(Summary {
        positions,
        top1: count(scores.iter().filter(|score| score.top1_agrees).count())?,
        kl_mean: mean(|score| score.kl),
        nll_reference: mean(|score| score.nll_reference),
        nll_candidate: mean(|score| score.nll_candidate),
        free_match: count(free_match)?,
    })
}

fn print_summary(cell: &Cell, label: &str, summary: &Summary) {
    println!(
        "summary cell={} codec={label} positions={} top1={} of {} kl_mean={:.6} nll_reference={:.6} nll_candidate={:.6} nll_delta={:.6} free_running_match={} of {}",
        cell.name,
        summary.positions,
        summary.top1,
        summary.positions,
        summary.kl_mean,
        summary.nll_reference,
        summary.nll_candidate,
        summary.nll_delta(),
        summary.free_match,
        summary.positions,
    );
}

fn print_relations(cell: &Cell, fp6: &Summary, matched: (&str, &Summary), identity: &Summary) {
    let (label, control) = matched;
    let name = &cell.name;
    let top1 = fp6.top1 >= control.top1;
    let kl = fp6.kl_mean <= control.kl_mean;
    let nll = fp6.nll_delta() <= control.nll_delta();
    let exact = identity.top1 == identity.positions;
    println!(
        "cell={name} relation=top1 fp6={} {label}={} holds={top1}",
        fp6.top1, control.top1
    );
    println!(
        "cell={name} relation=kl_mean fp6={:.6} {label}={:.6} holds={kl}",
        fp6.kl_mean, control.kl_mean
    );
    println!(
        "cell={name} relation=nll_delta fp6={:.6} {label}={:.6} holds={nll}",
        fp6.nll_delta(),
        control.nll_delta()
    );
    println!(
        "cell={name} relation=identity_top1 identity={} of {} holds={exact}",
        identity.top1, identity.positions
    );
}

fn report_teacher_forced(cell: &Cell, cases: &[Case], flags: &Flags) -> Result<(), AgreementError> {
    let matched = flags
        .matched_control
        .as_deref()
        .ok_or_else(|| usage("--teacher-forced requires --matched-control"))?;
    let reference = run_model(&cell.shipped, cases, None, 2)?;
    let mut summaries = vec![(
        "shipped_rerun".to_owned(),
        summarize(cell, "shipped_rerun", &reference, 1, &reference)?,
    )];
    let candidates = std::iter::once(("fp6", &cell.fp6)).chain(
        cell.controls
            .iter()
            .map(|(label, path)| (label.as_str(), path)),
    );
    for (label, path) in candidates {
        let run = run_model(path, cases, Some(&reference.generations), 1)?;
        summaries.push((
            label.to_owned(),
            summarize(cell, label, &run, 0, &reference)?,
        ));
    }
    summaries
        .iter()
        .for_each(|(label, summary)| print_summary(cell, label, summary));
    let find = |wanted: &str| {
        summaries
            .iter()
            .find(|(label, _)| label == wanted)
            .map(|(_, summary)| summary)
            .ok_or_else(|| usage(&format!("cell {} has no candidate {wanted:?}", cell.name)))
    };
    print_relations(
        cell,
        find("fp6")?,
        (matched, find(matched)?),
        find("identity")?,
    );
    Ok(())
}

fn report_cell(cell: &Cell, flags: &Flags) -> Result<(), AgreementError> {
    let cases = read_cases(cell, &flags.fixtures)?;
    if flags.teacher_forced {
        return report_teacher_forced(cell, &cases, flags);
    }
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
