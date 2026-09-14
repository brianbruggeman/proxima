//! Proves a checkpoint mapping is resident in physical memory before
//! [`crate::generate::LoadedModel::load_inner`] hands it to
//! `omega::backend::register_checkpoint_mapping` as a whole-mapping no-copy
//! `MTLBuffer` (`omega/src/metal.rs`'s own doc on that function) -- ROW 533's
//! own mechanism (`proxima-tensor/docs/discipline.md`): a non-resident page
//! behind a no-copy GPU buffer reads as zero rather than faulting, silently.
//!
//! [`prove_resident`] is the whole gate: [`crate::loader::prefault`] touches
//! every page once, `mincore(2)` proves it, and a single retry absorbs the
//! rare page a concurrent evictor reclaimed between the touch and the check.
//! The retry re-touches ONLY the byte ranges [`missing_byte_ranges`] found
//! still missing, not the whole mapping again (ROW 535: a retry that
//! re-touches everything gives a concurrent evictor exactly as much time to
//! reclaim the pages it already evicted once, so it can never converge under
//! sustained pressure -- narrowing the retry to the actual holes makes it
//! finish in a fraction of the time and lets it outrun the evictor).
//! [`count_missing_pages`] is split out of the `mincore` call itself so the
//! counting rule (which residency-vector bit means "resident") is testable
//! against a hand-built vector, with no real mapping or syscall involved --
//! `omega::metal::checkpoint_mmap_resident_pages` fuses the syscall and the
//! count into one diagnostic-only return value and does not expose the
//! vector, so it cannot serve that test.

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

/// Touches every page of `bytes` ([`crate::loader::prefault`]), then proves
/// residency with `mincore(2)`. A hole after the first prefault retries by
/// re-touching ONLY [`missing_byte_ranges`]' holes (not the whole mapping --
/// this module's own doc on why) and re-checks; a hole that survives the
/// retry is [`InteropError::MappingNotResident`] naming the exact byte
/// counts.
///
/// # Errors
///
/// [`InteropError::PrefaultPoolUnavailable`] if either prefault call fails.
/// [`InteropError::MappingNotResident`] if pages are still missing after the
/// retry.
pub fn prove_resident(bytes: &[u8]) -> Result<ResidencyReport, InteropError> {
    let page = omega::metal::page_size();
    let page_bytes = u64::try_from(page).unwrap_or(1);
    let bytes_total = u64::try_from(bytes.len()).unwrap_or(u64::MAX);

    let started = std::time::Instant::now();
    crate::loader::prefault(bytes)?;
    let mut residency = probe_residency(bytes);
    let mut missing_pages = count_missing_pages(&residency);
    if missing_pages > 0 {
        #[cfg(feature = "instrument")]
        log_missing_pattern(&residency, page, "first probe, before retry");
        for (start, end) in missing_byte_ranges(&residency, page, bytes.len()) {
            crate::loader::prefault(&bytes[start..end])?;
        }
        residency = probe_residency(bytes);
        missing_pages = count_missing_pages(&residency);
    }
    let prefault_ms = started.elapsed().as_secs_f64() * 1000.0;
    let resident_pages = residency.len().saturating_sub(missing_pages);

    if missing_pages > 0 {
        #[cfg(feature = "instrument")]
        log_missing_pattern(&residency, page, "second probe, after retry");
        let bytes_missing = (missing_pages as u64).saturating_mul(page_bytes).min(bytes_total);
        return Err(InteropError::MappingNotResident {
            bytes_missing,
            bytes_total,
        });
    }
    Ok(ResidencyReport {
        prefault_ms,
        resident_pages,
        missing_pages,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{ResidencyReport, count_missing_pages, missing_byte_ranges, prove_resident};

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
}
