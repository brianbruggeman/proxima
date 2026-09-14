//! Proves a checkpoint mapping is resident in physical memory before
//! [`crate::generate::LoadedModel::load_inner`] hands it to
//! `omega::backend::register_checkpoint_mapping` as a whole-mapping no-copy
//! `MTLBuffer` (`omega/src/metal.rs`'s own doc on that function) -- ROW 533's
//! own mechanism (`proxima-tensor/docs/discipline.md`): a non-resident page
//! behind a no-copy GPU buffer reads as zero rather than faulting, silently.
//!
//! [`prove_resident`] is the whole gate, walked as a [`ResidencyRung`]
//! ladder: [`crate::loader::prefault`] touches every page once, `mincore(2)`
//! proves it, and a retry absorbs the rare page a concurrent evictor
//! reclaimed between the touch and the check. The retry re-touches ONLY the
//! byte ranges [`missing_byte_ranges`] found still missing, not the whole
//! mapping again (ROW 535: a retry that re-touches everything gives a
//! concurrent evictor exactly as much time to reclaim the pages it already
//! evicted once, so it can never converge under sustained pressure --
//! narrowing the retry to the actual holes makes it finish in a fraction of
//! the time and lets it outrun the evictor). ROW 542: that retry still lost
//! the race at 81-82% memory free, so a final `mlock(2)` rung
//! ([`lock_resident`]) pins the whole mapping before the gate gives up --
//! a false refusal blocks serving, which is worse than the silent zero-read
//! ROW 533 named. [`count_missing_pages`] is split out of the `mincore` call
//! itself so the counting rule (which residency-vector bit means "resident")
//! is testable against a hand-built vector, with no real mapping or syscall
//! involved -- `omega::metal::checkpoint_mmap_resident_pages` fuses the
//! syscall and the count into one diagnostic-only return value and does not
//! expose the vector, so it cannot serve that test.

use crate::error::InteropError;

/// ROW 535 forensics: buckets `residency` into 1-GiB ranges and logs the
/// per-range missing-page count -- the isolated `#[ignore]`d test in this
/// module's own `tests` module proved a bare `prefault` + `mincore` round
/// trip on this exact checkpoint is 0% missing, so a real
/// [`crate::generate::LoadedModel::load_inner`] run that still fails needs
/// the RANGE of what went missing (tail-only vs. scattered) to distinguish
/// "concurrent eviction during the touch" from "a chunking bug that skips a
/// fixed region" -- an aggregate count cannot tell those apart.
#[cfg(feature = "instrument")]
fn log_missing_pattern(residency: &[i8], page: usize, stage: &'static str) {
    let pages_per_gib = (1024 * 1024 * 1024 / page).max(1);
    let missing_by_gib: std::vec::Vec<usize> =
        residency.chunks(pages_per_gib).map(count_missing_pages).collect();
    proxima_telemetry::debug!(
        stage,
        missing_by_gib_range = ?missing_by_gib,
        "checkpoint_mapping_residency: per-1-gib missing-page pattern"
    );
}

/// One [`prove_resident`] call's own accounting, carried out to the caller
/// so it can log `mapping_prefault_ms`/`mapping_resident_pages`/
/// `mapping_missing_pages` under `feature = "instrument"` the same way
/// `crate::generate::LoadedModel::apply_memory_fit_gate` logs its own
/// `memory_budget` event.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidencyReport {
    pub prefault_ms: f64,
    pub resident_pages: usize,
    pub missing_pages: usize,
    pub rung: ResidencyRung,
}

/// Which step of [`prove_resident`]'s ladder actually resolved every page --
/// ROW 542's fix for ROW 535's false refusal at 81-82% memory free: a plain
/// prefault + one holes-only retry can still lose the race to a concurrent
/// evictor under memory pressure, so `mlock(2)` (the same whole-mapping lock
/// llama.cpp's own `--mlock` flag takes) is the rung of last resort before
/// giving up and returning [`InteropError::MappingNotResident`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
pub enum ResidencyRung {
    /// The first prefault + `mincore` pass already found every page resident.
    Prefault,
    /// A holes-only re-prefault ([`missing_byte_ranges`]) resolved the rest.
    Retry,
    /// `mlock(2)` over the whole mapping was needed to pin the remainder.
    Mlock,
}

impl ResidencyRung {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            ResidencyRung::Prefault => "prefault",
            ResidencyRung::Retry => "retry",
            ResidencyRung::Mlock => "mlock",
        }
    }
}

/// The rung [`prove_resident`]'s last successful call resolved on --
/// `Prefault` (0) until a load has actually run. `omega/src/metal.rs`'s
/// `token_breakdown_metal` per-step line reads this through
/// [`active_residency_rung_str`] so ROW 551's reproduction table can read the
/// rung straight off the same log line as `mapping_rebound_blocks`, rather
/// than cross-referencing a separate one-shot load-time log record.
static ACHIEVED_RUNG: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// See [`ACHIEVED_RUNG`]'s own doc.
#[must_use]
pub fn active_residency_rung_str() -> &'static str {
    match ACHIEVED_RUNG.load(std::sync::atomic::Ordering::Relaxed) {
        1 => ResidencyRung::Retry.as_str(),
        2 => ResidencyRung::Mlock.as_str(),
        _ => ResidencyRung::Prefault.as_str(),
    }
}

/// Counts pages in `residency` -- one byte per page, in the exact
/// `mincore(2)` encoding (`libc::MINCORE_INCORE` set means resident) -- that
/// are NOT resident. A pure function over the syscall's own output shape, so
/// a fake vector with holes exercises the same counting rule the real probe
/// uses.
#[must_use]
pub fn count_missing_pages(residency: &[i8]) -> usize {
    residency
        .iter()
        .filter(|state| **state & libc::MINCORE_INCORE as i8 == 0)
        .count()
}

/// Collapses `residency`'s missing pages into contiguous byte ranges over a
/// `bytes_len`-byte mapping, so a retry can re-touch only the holes
/// (`prove_resident`'s own doc on why "only the holes" is the load-bearing
/// difference from re-touching the whole mapping). A pure function over the
/// same `mincore` output shape [`count_missing_pages`] already tests against
/// a hand-built vector.
#[must_use]
fn missing_byte_ranges(residency: &[i8], page: usize, bytes_len: usize) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut run_start: Option<usize> = None;
    for (page_index, state) in residency.iter().enumerate() {
        let missing = *state & libc::MINCORE_INCORE as i8 == 0;
        if missing {
            run_start.get_or_insert(page_index);
        } else if let Some(start) = run_start.take() {
            ranges.push((start * page, (page_index * page).min(bytes_len)));
        }
    }
    if let Some(start) = run_start {
        ranges.push((start * page, bytes_len));
    }
    ranges
}

/// `mincore(2)` over `bytes`' own page range, returning one residency byte
/// per page in the same order [`count_missing_pages`] expects.
///
/// # Errors
///
/// [`InteropError::Metal`] is never returned here; `mincore` failures surface
/// as [`InteropError::MappingNotResident`] with `bytes_total` pages all
/// marked missing, since a probe that cannot run proves nothing about
/// residency either way.
fn probe_residency(bytes: &[u8]) -> Vec<i8> {
    if bytes.is_empty() {
        return Vec::new();
    }
    let page = omega::metal::page_size();
    let page_count = bytes.len().div_ceil(page);
    let mut residency = vec![0_i8; page_count];
    // SAFETY: `bytes` is a live borrow for the duration of this call, and
    // `residency` is sized for exactly `page_count` bytes, matching
    // `mincore`'s own `len`-derived output length contract.
    let probed = unsafe {
        libc::mincore(
            bytes.as_ptr().cast::<libc::c_void>(),
            bytes.len(),
            residency.as_mut_ptr().cast::<libc::c_char>(),
        )
    };
    if probed != 0 {
        // a failed probe proves nothing; treat every page as missing rather
        // than reporting a false all-resident pass.
        return vec![0_i8; page_count];
    }
    residency
}

/// Walks [`ResidencyRung::Prefault`] -> [`ResidencyRung::Retry`] ->
/// [`ResidencyRung::Mlock`], calling `attempt` for each rung and stopping at
/// the first one whose reported missing-page count is zero. Extracted as a
/// pure control-flow function over an injected `attempt` so a unit test can
/// drive the whole ladder with a fake residency probe -- no real mapping or
/// syscall -- the same reason [`count_missing_pages`] is split out of
/// [`probe_residency`] above. `start` skips every earlier rung outright (used
/// by [`forced_start_rung`] to reproduce a specific rung deliberately,
/// ROW 551's own reproduction harness) -- production always calls this with
/// [`ResidencyRung::Prefault`].
fn walk_residency_ladder<F>(start: ResidencyRung, mut attempt: F) -> (usize, ResidencyRung)
where
    F: FnMut(ResidencyRung) -> usize,
{
    let mut missing = 0;
    for rung in [ResidencyRung::Prefault, ResidencyRung::Retry, ResidencyRung::Mlock] {
        if rung < start {
            continue;
        }
        missing = attempt(rung);
        if missing == 0 {
            return (0, rung);
        }
    }
    (missing, ResidencyRung::Mlock)
}

/// `PROXIMA_DEBUG_FORCE_RESIDENCY_RUNG=retry|mlock` skips [`prove_resident`]
/// straight to that rung of the ladder, so ROW 551's reproduction harness can
/// exercise the mlock rung deliberately instead of hoping a real evictor race
/// lands it. Unset or unrecognized falls through to the normal
/// [`ResidencyRung::Prefault`] start. `#[cfg(feature = "instrument")]`-gated:
/// a reproduction knob has no business compiling into a default build.
#[cfg(feature = "instrument")]
fn forced_start_rung() -> ResidencyRung {
    match std::env::var("PROXIMA_DEBUG_FORCE_RESIDENCY_RUNG").as_deref() {
        Ok("retry") => ResidencyRung::Retry,
        Ok("mlock") => ResidencyRung::Mlock,
        _ => ResidencyRung::Prefault,
    }
}

#[cfg(not(feature = "instrument"))]
fn forced_start_rung() -> ResidencyRung {
    ResidencyRung::Prefault
}

/// Best-effort `mlock(2)` over the whole mapping -- the ladder's last rung
/// before [`InteropError::MappingNotResident`]. llama.cpp's own `--mlock`
/// flag takes the identical whole-mapping lock (measured 0.8 s on the
/// qwen35moe checkpoint used in this module's own `#[ignore]`d test). A
/// failed lock is not fatal here: [`probe_residency`] re-checks afterward
/// and the ladder falls through to the existing error if pages are still
/// missing, mirroring [`probe_residency`]'s own "a failed probe proves
/// nothing" stance on a failed `mincore`.
fn lock_resident(bytes: &[u8]) {
    if bytes.is_empty() {
        return;
    }
    // SAFETY: `bytes` is a live borrow for the duration of this call;
    // `mlock` only pins pages already mapped into this process and never
    // writes through the pointer.
    let outcome = unsafe {
        rustix::mm::mlock(bytes.as_ptr().cast::<core::ffi::c_void>().cast_mut(), bytes.len())
    };
    #[cfg(feature = "instrument")]
    if let Err(error) = outcome {
        proxima_telemetry::debug!(
            ?error,
            "checkpoint_mapping_residency: mlock rung syscall failed, falling through to the existing error"
        );
    }
    #[cfg(not(feature = "instrument"))]
    let _ = outcome;
}

/// Touches every page of `bytes` ([`crate::loader::prefault`]), then proves
/// residency with `mincore(2)`. A hole after the first prefault retries by
/// re-touching ONLY [`missing_byte_ranges`]' holes (this module's own doc on
/// why), then re-checks; a hole that survives the retry gets one more rung
/// -- [`lock_resident`]'s whole-mapping `mlock(2)` -- before giving up. ROW
/// 542: the retry-only ladder still lost to a concurrent evictor at 81-82%
/// memory free (11.7-8.5 GB missing of 24 GB), a false refusal that is worse
/// than the silent zero-read it guards against (ROW 533), so `mlock` is the
/// rung that can never lose that race.
///
/// # Errors
///
/// [`InteropError::PrefaultPoolUnavailable`] if a prefault call fails.
/// [`InteropError::MappingNotResident`] if pages are still missing after
/// every rung, including `mlock`.
pub fn prove_resident(bytes: &[u8]) -> Result<ResidencyReport, InteropError> {
    let page = omega::metal::page_size();
    let page_bytes = u64::try_from(page).unwrap_or(1);
    let bytes_total = u64::try_from(bytes.len()).unwrap_or(u64::MAX);

    let started = std::time::Instant::now();
    let mut residency: Vec<i8> = Vec::new();
    let mut first_error: Option<InteropError> = None;

    let (missing_pages, rung) = walk_residency_ladder(forced_start_rung(), |rung| {
        if first_error.is_some() {
            return count_missing_pages(&residency);
        }
        let outcome = match rung {
            ResidencyRung::Prefault => crate::loader::prefault(bytes),
            ResidencyRung::Retry => {
                let mut result = Ok(());
                for (start, end) in missing_byte_ranges(&residency, page, bytes.len()) {
                    if let Err(error) = crate::loader::prefault(&bytes[start..end]) {
                        result = Err(error);
                        break;
                    }
                }
                result
            }
            ResidencyRung::Mlock => {
                lock_resident(bytes);
                Ok(())
            }
        };
        if let Err(error) = outcome {
            first_error = Some(error);
            return count_missing_pages(&residency);
        }
        residency = probe_residency(bytes);
        let missing = count_missing_pages(&residency);
        #[cfg(feature = "instrument")]
        log_missing_pattern(&residency, page, rung.as_str());
        missing
    });

    if let Some(error) = first_error {
        return Err(error);
    }
    let prefault_ms = started.elapsed().as_secs_f64() * 1000.0;
    let resident_pages = residency.len().saturating_sub(missing_pages);

    #[cfg(feature = "instrument")]
    proxima_telemetry::debug!(
        mapping_residency_rung = rung.as_str(),
        mapping_missing_pages = missing_pages,
        "checkpoint_mapping_residency: ladder outcome"
    );

    if missing_pages > 0 {
        let bytes_missing = (missing_pages as u64).saturating_mul(page_bytes).min(bytes_total);
        return Err(InteropError::MappingNotResident {
            bytes_missing,
            bytes_total,
        });
    }
    ACHIEVED_RUNG.store(rung as u8, std::sync::atomic::Ordering::Relaxed);
    Ok(ResidencyReport {
        prefault_ms,
        resident_pages,
        missing_pages,
        rung,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{
        ResidencyReport, ResidencyRung, count_missing_pages, missing_byte_ranges, prove_resident,
        walk_residency_ladder,
    };

    const MAPPING_SIZE_BYTES: usize = 64 * 1024 * 1024;

    #[test]
    fn count_missing_pages_counts_only_holes_in_a_fake_residency_vector() {
        let all_resident = vec![libc::MINCORE_INCORE as i8; 4];
        assert_eq!(count_missing_pages(&all_resident), 0);

        let one_hole = vec![
            libc::MINCORE_INCORE as i8,
            0,
            libc::MINCORE_INCORE as i8,
            libc::MINCORE_INCORE as i8,
        ];
        assert_eq!(count_missing_pages(&one_hole), 1);

        let all_holes = vec![0_i8; 3];
        assert_eq!(count_missing_pages(&all_holes), 3);
    }

    #[test]
    fn count_missing_pages_is_empty_for_an_empty_vector() {
        assert_eq!(count_missing_pages(&[]), 0);
    }

    /// ROW 535 fix: a retry must re-touch only the holes, not the whole
    /// mapping -- this asserts the exact byte ranges a scattered fake
    /// residency vector collapses to, over a 4-page (16 KiB/page) mapping.
    #[test]
    fn missing_byte_ranges_collapses_scattered_holes_into_contiguous_byte_spans() {
        const PAGE: usize = 16384;
        let resident = libc::MINCORE_INCORE as i8;
        // pages: [hole][resident][hole][hole] -- run 0 is one page, run 1 is
        // two pages, over a mapping exactly 4 pages long.
        let residency = [0_i8, resident, 0, 0];
        let ranges = missing_byte_ranges(&residency, PAGE, PAGE * 4);
        assert_eq!(ranges, vec![(0, PAGE), (PAGE * 2, PAGE * 4)]);
    }

    #[test]
    fn missing_byte_ranges_is_empty_when_every_page_is_resident() {
        const PAGE: usize = 16384;
        let residency = vec![libc::MINCORE_INCORE as i8; 4];
        assert!(missing_byte_ranges(&residency, PAGE, PAGE * 4).is_empty());
    }

    /// A trailing hole must clamp its end to the mapping's real byte length,
    /// not the page-rounded length -- the last page of a mapping whose size
    /// is not a page multiple is shorter than `page` bytes.
    #[test]
    fn missing_byte_ranges_clamps_a_trailing_hole_to_the_real_mapping_length() {
        const PAGE: usize = 16384;
        let bytes_len = PAGE + 100;
        let residency = [libc::MINCORE_INCORE as i8, 0];
        let ranges = missing_byte_ranges(&residency, PAGE, bytes_len);
        assert_eq!(ranges, vec![(PAGE, bytes_len)]);
    }

    /// A fake probe that reports zero missing at the very first rung must
    /// stop the ladder there -- the common case never needs a retry or an
    /// `mlock`.
    #[test]
    fn walk_residency_ladder_stops_at_prefault_when_it_already_resolves_everything() {
        let mut calls = Vec::new();
        let (missing, rung) = walk_residency_ladder(ResidencyRung::Prefault, |rung| {
            calls.push(rung);
            0
        });
        assert_eq!(missing, 0);
        assert_eq!(rung, ResidencyRung::Prefault);
        assert_eq!(calls, vec![ResidencyRung::Prefault]);
    }

    /// ROW 542's own reproduction shape: prefault and the holes-only retry
    /// both still report holes, and only the `mlock` rung resolves them --
    /// the ladder must walk all three rungs in order and report `Mlock` as
    /// the one that succeeded.
    #[test]
    fn walk_residency_ladder_falls_through_to_mlock_when_earlier_rungs_still_have_holes() {
        let mut calls = Vec::new();
        let (missing, rung) = walk_residency_ladder(ResidencyRung::Prefault, |rung| {
            calls.push(rung);
            match rung {
                ResidencyRung::Prefault => 42,
                ResidencyRung::Retry => 7,
                ResidencyRung::Mlock => 0,
            }
        });
        assert_eq!(missing, 0);
        assert_eq!(rung, ResidencyRung::Mlock);
        assert_eq!(
            calls,
            vec![ResidencyRung::Prefault, ResidencyRung::Retry, ResidencyRung::Mlock]
        );
    }

    /// A hole that survives every rung, including `mlock`, must report the
    /// final missing count at `Mlock` rather than silently claiming success
    /// -- this is what `prove_resident` turns into `MappingNotResident`.
    #[test]
    fn walk_residency_ladder_reports_the_residual_when_mlock_still_has_holes() {
        let (missing, rung) = walk_residency_ladder(ResidencyRung::Prefault, |_rung| 3);
        assert_eq!(missing, 3);
        assert_eq!(rung, ResidencyRung::Mlock);
    }

    #[test]
    fn prove_resident_reports_zero_missing_pages_for_a_patterned_temp_file() {
        let directory = tempfile::tempdir().expect("create a scratch tempdir for the mapping");
        let path = directory.path().join("checkpoint.bin");
        let pattern: Vec<u8> = (0..MAPPING_SIZE_BYTES).map(|index| (index % 251) as u8).collect();
        std::fs::write(&path, &pattern).expect("write the patterned fixture file");

        let file = std::fs::File::open(&path).expect("open the fixture file for mapping");
        // SAFETY: `file` is not written or truncated by any other process
        // while this mapping is alive for the duration of this test.
        let mapping = unsafe { memmap2::Mmap::map(&file) }.expect("mmap the fixture file");

        let report: ResidencyReport =
            prove_resident(&mapping).expect("prefault must resolve every page of a real mapping");
        assert_eq!(report.missing_pages, 0);
        assert!(report.resident_pages > 0);
    }

    /// ROW 535 forensics: `prove_resident` fails on the real 23,938,321,664
    /// byte qwen35moe blob with 49% of pages missing after prefault + one
    /// retry, at 82% memory free -- plenty of headroom for the whole mapping
    /// to stay resident. This test reproduces the probe against the real
    /// file and reports per-1-GiB-range resident/missing counts to stderr so
    /// the pattern (tail-only, alternating, or uniform loss) is visible
    /// instead of guessed from the aggregate count `run{1,2}.log` shows.
    #[test]
    #[ignore = "requires the real, local qwen3.6:35b-a3b GGUF blob; set PROXIMA_QWEN35MOE_GGUF"]
    fn prove_resident_per_gib_range_pattern_on_the_real_qwen35moe_checkpoint() {
        let path = std::env::var("PROXIMA_QWEN35MOE_GGUF").unwrap_or_else(|_| {
            "/Users/brianbruggeman/.ollama/models/blobs/\
             sha256-f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d"
                .to_string()
        });
        if !std::path::Path::new(&path).exists() {
            eprintln!("skipping: no checkpoint at {path} (set PROXIMA_QWEN35MOE_GGUF)");
            return;
        }
        let file = std::fs::File::open(&path).unwrap_or_else(|error| panic!("open {path}: {error}"));
        // SAFETY: read-only mapping of a file this test does not write or
        // truncate; the mapping is the whole test's scope.
        let mapping = unsafe { memmap2::Mmap::map(&file) }.expect("mmap the real checkpoint read-only");
        let bytes: &[u8] = &mapping;

        crate::loader::prefault(bytes).expect("the prefault pool builds and every chunk reports");
        let residency = super::probe_residency(bytes);
        let page = omega::metal::page_size();
        eprintln!("page_size = {page} bytes, mapping = {} bytes, {} pages", bytes.len(), residency.len());

        let pages_per_gib = (1024 * 1024 * 1024) / page;
        for (range_index, chunk) in residency.chunks(pages_per_gib.max(1)).enumerate() {
            let missing = count_missing_pages(chunk);
            eprintln!(
                "range {range_index:>3} [{range_index} GiB, {}): resident={} missing={} total={}",
                range_index + 1,
                chunk.len() - missing,
                missing,
                chunk.len(),
            );
        }

        let total_missing = count_missing_pages(&residency);
        eprintln!(
            "total: resident={} missing={} total={} ({:.1}% missing)",
            residency.len() - total_missing,
            total_missing,
            residency.len(),
            100.0 * total_missing as f64 / residency.len() as f64,
        );
    }

    /// ROW 542 root-cause: stages a second immediate prefault, an
    /// `MADV_WILLNEED`, and an `mlock` against the same real qwen35moe
    /// mapping in sequence, timing and `mincore`-checking each one, so the
    /// rung that actually resolves the holes ROW 535 hit is visible instead
    /// of guessed. A near-zero elapsed time for a stage over a 24 GB mapping
    /// (this module's own doc: a real touch is >= 500 ms) would mean that
    /// stage's touch is a no-op the optimizer elided.
    #[test]
    #[ignore = "requires the real, local qwen3.6:35b-a3b GGUF blob; set PROXIMA_QWEN35MOE_GGUF"]
    fn residency_ladder_stage_timings_on_the_real_qwen35moe_checkpoint() {
        let path = std::env::var("PROXIMA_QWEN35MOE_GGUF").unwrap_or_else(|_| {
            "/Users/brianbruggeman/.ollama/models/blobs/\
             sha256-f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d"
                .to_string()
        });
        if !std::path::Path::new(&path).exists() {
            eprintln!("skipping: no checkpoint at {path} (set PROXIMA_QWEN35MOE_GGUF)");
            return;
        }
        let file = std::fs::File::open(&path).unwrap_or_else(|error| panic!("open {path}: {error}"));
        // SAFETY: read-only mapping of a file this test does not write or
        // truncate; the mapping is the whole test's scope.
        let mapping = unsafe { memmap2::Mmap::map(&file) }.expect("mmap the real checkpoint read-only");
        let bytes: &[u8] = &mapping;

        let report = |label: &str, elapsed: std::time::Duration, residency: &[i8]| {
            let missing = count_missing_pages(residency);
            let ranges = missing_byte_ranges(residency, omega::metal::page_size(), bytes.len());
            let first_missing = ranges.first().map(|range| range.0);
            let last_missing = ranges.last().map(|range| range.1);
            eprintln!(
                "{label}: {:.1} ms, missing_pages={missing} of {}, contiguous_runs={}, \
                 first_missing_offset={first_missing:?}, last_missing_offset={last_missing:?}",
                elapsed.as_secs_f64() * 1000.0,
                residency.len(),
                ranges.len(),
            );
        };

        let started = std::time::Instant::now();
        crate::loader::prefault(bytes).expect("first prefault must resolve or report");
        let residency = super::probe_residency(bytes);
        report("stage 1: prefault", started.elapsed(), &residency);

        let started = std::time::Instant::now();
        crate::loader::prefault(bytes).expect("second immediate prefault must resolve or report");
        let residency = super::probe_residency(bytes);
        report("stage 2: second immediate prefault", started.elapsed(), &residency);

        let started = std::time::Instant::now();
        // SAFETY: `bytes` is a live borrow for the duration of this call;
        // `madvise` only advises the kernel and never writes through the
        // pointer.
        let advised = unsafe {
            rustix::mm::madvise(
                bytes.as_ptr().cast::<core::ffi::c_void>().cast_mut(),
                bytes.len(),
                rustix::mm::Advice::WillNeed,
            )
        };
        if let Err(error) = advised {
            eprintln!("stage 3: madvise(WILLNEED) failed: {error}");
        }
        let residency = super::probe_residency(bytes);
        report("stage 3: madvise(WILLNEED)", started.elapsed(), &residency);

        let started = std::time::Instant::now();
        super::lock_resident(bytes);
        let residency = super::probe_residency(bytes);
        report("stage 4: mlock", started.elapsed(), &residency);
    }
}
