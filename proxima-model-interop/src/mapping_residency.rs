//! Residency of a checkpoint mapping that is handed to
//! `omega::backend::register_checkpoint_mapping` as a whole-mapping no-copy
//! `MTLBuffer` (ROW 533's mechanism, `proxima-tensor/docs/discipline.md`: a
//! non-resident page behind a no-copy GPU buffer reads as zero rather than
//! faulting).
//!
//! The load path only asks the kernel to start reading the file in
//! ([`advise_will_need`], `posix_madvise(WILLNEED)`, the same single advice
//! llama.cpp issues). The guarantee that a GPU read never sees a non-resident
//! page lives where the GPU gets the mapping: omega adds the mapping's no-copy
//! buffer to its `MTLResidencySet` and requests residency when the buffer is
//! created (`omega::metal`'s `mapping_buffer`).
//!
//! [`probe_residency`], [`count_missing_pages`] and [`missing_byte_ranges`]
//! stay as measurement helpers for [`crate::probes`]. [`count_missing_pages`]
//! is split out of the `mincore` call so the counting rule (which
//! residency-vector bit means "resident") is testable against a hand-built
//! vector, with no real mapping or syscall involved.

/// Wall time and minor faults of one residency pass, logged at debug level so
/// each pass is attributable on its own. A zero-size no-op without
/// `instrument`.
#[cfg(feature = "instrument")]
struct PassClock {
    started: std::time::Instant,
    faults_before: i64,
}

#[cfg(feature = "instrument")]
impl PassClock {
    fn start() -> Self {
        Self {
            started: std::time::Instant::now(),
            faults_before: minor_faults(),
        }
    }

    fn finish(self, pass: &'static str) {
        proxima_telemetry::debug!(
            pass,
            pass_micros = self.started.elapsed().as_micros() as u64,
            minor_faults = minor_faults() - self.faults_before,
            "checkpoint_mapping_residency: pass timing"
        );
    }
}

#[cfg(feature = "instrument")]
fn minor_faults() -> i64 {
    // SAFETY: `rusage` is plain old data, so the zeroed value is valid, and
    // `getrusage` only writes through the pointer it is given.
    let mut usage: libc::rusage = unsafe { core::mem::zeroed() };
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    if status == 0 {
        usage.ru_minflt as i64
    } else {
        0
    }
}

#[cfg(not(feature = "instrument"))]
struct PassClock;

#[cfg(not(feature = "instrument"))]
impl PassClock {
    fn start() -> Self {
        Self
    }

    fn finish(self, _pass: &'static str) {}
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
/// `bytes_len`-byte mapping. A pure function over the same `mincore` output
/// shape [`count_missing_pages`] already tests against a hand-built vector.
#[must_use]
pub(crate) fn missing_byte_ranges(
    residency: &[i8],
    page: usize,
    bytes_len: usize,
) -> Vec<(usize, usize)> {
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
/// per page in the same order [`count_missing_pages`] expects. A probe that
/// cannot run proves nothing about residency, so a `mincore` failure reports
/// every page as missing.
pub(crate) fn probe_residency(bytes: &[u8]) -> Vec<i8> {
    if bytes.is_empty() {
        return Vec::new();
    }
    let page = omega::metal::page_size();
    let page_count = bytes.len().div_ceil(page);
    let mut residency = vec![0_i8; page_count];
    let clock = PassClock::start();
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
    clock.finish("mincore");
    if probed != 0 {
        return vec![0_i8; page_count];
    }
    residency
}

/// Asks the kernel to read `bytes` in ahead of use: one
/// `posix_madvise(WILLNEED)` over the whole mapping, with no per-page touch.
/// Advice only, so a failure leaves the mapping correct and merely colder;
/// it is surfaced as an `instrument`-gated warning.
pub(crate) fn advise_will_need(bytes: &[u8]) {
    if bytes.is_empty() {
        return;
    }
    let clock = PassClock::start();
    // SAFETY: `bytes` is a live borrow for the duration of this call;
    // `madvise` only advises the kernel and never writes through the pointer.
    let outcome = unsafe {
        rustix::mm::madvise(
            bytes.as_ptr().cast::<core::ffi::c_void>().cast_mut(),
            bytes.len(),
            rustix::mm::Advice::WillNeed,
        )
    };
    clock.finish("madvise_will_need");
    #[cfg(feature = "instrument")]
    if let Err(error) = outcome {
        proxima_telemetry::warn!(
            ?error,
            "checkpoint_mapping_residency: madvise(WILLNEED) failed, the mapping faults in on first use"
        );
    }
    #[cfg(not(feature = "instrument"))]
    let _ = outcome;
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{advise_will_need, count_missing_pages, missing_byte_ranges};

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

    /// the exact byte ranges a scattered fake residency vector collapses to,
    /// over a 4-page (16 KiB/page) mapping.
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

    /// advice is a hint: it must leave a real mapping's bytes exactly the
    /// file's, and accept an empty slice.
    #[test]
    fn advise_will_need_leaves_a_patterned_mapping_readable_and_accepts_empty() {
        let directory = tempfile::tempdir().expect("create a scratch tempdir for the mapping");
        let path = directory.path().join("checkpoint.bin");
        let pattern: Vec<u8> = (0..MAPPING_SIZE_BYTES)
            .map(|index| (index % 251) as u8)
            .collect();
        std::fs::write(&path, &pattern).expect("write the patterned fixture file");

        let file = std::fs::File::open(&path).expect("open the fixture file for mapping");
        // SAFETY: `file` is not written or truncated by any other process
        // while this mapping is alive for the duration of this test.
        let mapping = unsafe { memmap2::Mmap::map(&file) }.expect("mmap the fixture file");

        advise_will_need(&mapping);
        advise_will_need(&[]);

        assert_eq!(&mapping[..], &pattern[..]);
    }

    /// bare `prefault` + `mincore` round trip on the real checkpoint,
    /// reported per 1-GiB range so a hole pattern (tail-only, alternating,
    /// uniform) is visible instead of guessed from an aggregate.
    #[test]
    #[ignore = "requires the real, local qwen3.6:35b-a3b GGUF blob; set PROXIMA_QWEN35MOE_GGUF"]
    fn probe_per_gib_range_pattern_on_the_real_qwen35moe_checkpoint() {
        let path = std::env::var("PROXIMA_QWEN35MOE_GGUF").unwrap_or_else(|_| {
            "/Users/brianbruggeman/.ollama/models/blobs/\
             sha256-f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d"
                .to_string()
        });
        if !std::path::Path::new(&path).exists() {
            eprintln!("skipping: no checkpoint at {path} (set PROXIMA_QWEN35MOE_GGUF)");
            return;
        }
        let file =
            std::fs::File::open(&path).unwrap_or_else(|error| panic!("open {path}: {error}"));
        // SAFETY: read-only mapping of a file this test does not write or
        // truncate; the mapping is the whole test's scope.
        let mapping =
            unsafe { memmap2::Mmap::map(&file) }.expect("mmap the real checkpoint read-only");
        let bytes: &[u8] = &mapping;

        crate::loader::prefault(bytes).expect("the prefault pool builds and every chunk reports");
        let residency = super::probe_residency(bytes);
        let page = omega::metal::page_size();
        eprintln!(
            "page_size = {page} bytes, mapping = {} bytes, {} pages",
            bytes.len(),
            residency.len()
        );

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
