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
//! [`count_missing_pages`] is split out of the `mincore` call itself so the
//! counting rule (which residency-vector bit means "resident") is testable
//! against a hand-built vector, with no real mapping or syscall involved --
//! `omega::metal::checkpoint_mmap_resident_pages` fuses the syscall and the
//! count into one diagnostic-only return value and does not expose the
//! vector, so it cannot serve that test.

use crate::error::InteropError;

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
/// residency with `mincore(2)`. A hole after the first prefault retries the
/// prefault once and re-checks; a hole that survives the retry is
/// [`InteropError::MappingNotResident`] naming the exact byte counts.
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
        crate::loader::prefault(bytes)?;
        residency = probe_residency(bytes);
        missing_pages = count_missing_pages(&residency);
    }
    let prefault_ms = started.elapsed().as_secs_f64() * 1000.0;
    let resident_pages = residency.len().saturating_sub(missing_pages);

    if missing_pages > 0 {
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
    use super::{ResidencyReport, count_missing_pages, prove_resident};

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
}
