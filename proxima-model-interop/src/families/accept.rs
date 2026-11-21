//! The golden gate: an inferred model runs only after its golden passed.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};
#[cfg(feature = "std")]
use std::{fs, io, path::Path};
use thiserror::Error;

use super::infer::Inferred;

/// One measured comparison against the incumbent, one JSONL line in the ledger.
///
/// Keyed on the full-file sha and the inferred digest, so a different file or a
/// rule change that alters an inferred value re-opens the golden.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldenRecord {
    pub sha: String,
    pub digest: String,
    pub stage: String,
    pub measured: f64,
    pub ceiling: f64,
}

/// Why an inferred model was refused.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum Refusal {
    #[error("golden_missing: no record for sha {sha} digest {digest}{}", stage_note(.stage))]
    GoldenMissing {
        sha: String,
        digest: String,
        stage: Option<String>,
    },
    #[error("golden_failed: stage {stage:?} measured {measured} against ceiling {ceiling}")]
    GoldenFailed {
        stage: String,
        measured: f64,
        ceiling: f64,
    },
}

fn stage_note(stage: &Option<String>) -> String {
    stage
        .as_ref()
        .map(|name| alloc::format!(" stage {name:?}"))
        .unwrap_or_default()
}

/// A ledger line that is not a [`GoldenRecord`].
#[derive(Debug, Error)]
#[error("ledger line {line} is not a golden record: {reason}")]
pub struct LedgerError {
    pub line: usize,
    pub reason: String,
}

/// An inferred model whose every required golden stage passed.
///
/// Only [`accept`] builds one, so a runner that takes `Accepted` cannot be
/// handed a hypothesis.
#[derive(Debug)]
pub struct Accepted<'inferred> {
    inferred: &'inferred Inferred,
}

impl Accepted<'_> {
    #[must_use]
    pub fn inferred(&self) -> &Inferred {
        self.inferred
    }
}

/// Accepts `inferred` only when `records` hold a passing golden for each of
/// the stages its family requires, under its sha and digest.
///
/// The ledger is append-only and records failures too, so the LAST record for
/// a (sha, digest, stage) is authoritative: a later failure revokes an earlier
/// pass and a later pass supersedes an earlier failure.
///
/// # Errors
///
/// [`Refusal::GoldenMissing`] when a stage has no record, [`Refusal::GoldenFailed`]
/// when its latest measurement is above its ceiling or not a number.
#[must_use = "a refused model must not be run"]
pub fn accept<'inferred>(
    inferred: &'inferred Inferred,
    records: &[GoldenRecord],
) -> Result<Accepted<'inferred>, Refusal> {
    let digest = inferred.digest();
    let sha = inferred.sha.as_str();
    let mine = |record: &&GoldenRecord| record.sha == sha && record.digest == digest;
    let missing = |stage: Option<String>| Refusal::GoldenMissing {
        sha: sha.to_string(),
        digest: digest.clone(),
        stage,
    };
    if !records.iter().any(|record| mine(&record)) {
        return Err(missing(None));
    }
    for stage in &inferred.required_stages {
        let latest = records
            .iter()
            .rev()
            .filter(mine)
            .find(|record| record.stage == *stage)
            .ok_or_else(|| missing(Some(stage.clone())))?;
        check_ceiling(latest)?;
    }
    Ok(Accepted { inferred })
}

fn check_ceiling(record: &GoldenRecord) -> Result<(), Refusal> {
    match record.measured <= record.ceiling {
        true => Ok(()),
        false => Err(Refusal::GoldenFailed {
            stage: record.stage.clone(),
            measured: record.measured,
            ceiling: record.ceiling,
        }),
    }
}

/// Parses a JSONL ledger; blank lines are skipped.
///
/// # Errors
///
/// [`LedgerError`] naming the first line that is not a record.
#[must_use = "a malformed ledger must be handled"]
pub fn parse_ledger(text: &str) -> Result<Vec<GoldenRecord>, LedgerError> {
    text.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| {
            serde_json::from_str(line).map_err(|error| LedgerError {
                line: index + 1,
                reason: error.to_string(),
            })
        })
        .collect()
}

/// Reads a JSONL ledger file; the file read is the only `std` part.
///
/// # Errors
///
/// [`LedgerFileError`] when the file cannot be read or a line is not a record.
#[cfg(feature = "std")]
#[must_use = "a malformed ledger must be handled"]
pub fn read_ledger(path: &Path) -> Result<Vec<GoldenRecord>, LedgerFileError> {
    let text = fs::read_to_string(path)?;
    Ok(parse_ledger(&text)?)
}

/// Why a ledger file was not read.
#[cfg(feature = "std")]
#[derive(Debug, Error)]
pub enum LedgerFileError {
    #[error("ledger file unreadable: {0}")]
    Io(#[from] io::Error),
    #[error(transparent)]
    Parse(#[from] LedgerError),
}
