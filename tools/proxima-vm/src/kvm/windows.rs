use arrayvec::ArrayVec;

pub const MAX_WINDOWS: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentSpan {
    pub guest_address: u64,
    pub memory_size: u64,
    pub writable: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    pub start: u64,
    pub end: u64,
    pub writable: bool,
}

#[must_use]
pub fn merge_into_windows(spans: &[SegmentSpan], page_size: u64) -> ArrayVec<Window, MAX_WINDOWS> {
    let mut sorted: ArrayVec<SegmentSpan, MAX_WINDOWS> = spans.iter().take(MAX_WINDOWS).copied().collect();
    sorted.sort_by_key(|span| span.guest_address);
    let mut windows: ArrayVec<Window, MAX_WINDOWS> = ArrayVec::new();
    for span in &sorted {
        let start = span.guest_address / page_size * page_size;
        let bytes = span.guest_address - start + span.memory_size.max(1);
        let end = start + bytes.div_ceil(page_size) * page_size;
        match windows.last_mut() {
            Some(last) if start < last.end => {
                last.end = last.end.max(end);
                last.writable |= span.writable;
            }
            _ => {
                let _ = windows.try_push(Window { start, end, writable: span.writable });
            }
        }
    }
    windows
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    const PAGE: u64 = 4096;

    fn span(guest_address: u64, memory_size: u64, writable: bool) -> SegmentSpan {
        SegmentSpan { guest_address, memory_size, writable }
    }

    #[test]
    fn text_and_rodata_sharing_the_first_page_merge_into_one_read_only_window() {
        let windows = merge_into_windows(&[span(0, 291, false), span(291, 12, false)], PAGE);

        assert_eq!(windows.as_slice(), &[Window { start: 0, end: PAGE, writable: false }]);
    }

    #[test]
    fn a_writable_segment_in_a_shared_page_makes_the_whole_window_writable() {
        let windows = merge_into_windows(&[span(0, 100, false), span(200, 50, true)], PAGE);

        assert_eq!(windows.as_slice(), &[Window { start: 0, end: PAGE, writable: true }]);
    }

    #[test]
    fn windows_that_merely_touch_end_to_end_stay_separate() {
        let windows = merge_into_windows(&[span(0, PAGE, false), span(PAGE, PAGE, true)], PAGE);

        assert_eq!(
            windows.as_slice(),
            &[
                Window { start: 0, end: PAGE, writable: false },
                Window { start: PAGE, end: 2 * PAGE, writable: true },
            ]
        );
    }

    #[test]
    fn segments_are_sorted_before_merging() {
        let windows = merge_into_windows(&[span(2 * PAGE, PAGE, true), span(0, PAGE, false)], PAGE);

        assert_eq!(windows[0].start, 0);
        assert_eq!(windows[1].start, 2 * PAGE);
    }

    #[test]
    fn a_zero_sized_segment_still_claims_its_page() {
        let windows = merge_into_windows(&[span(PAGE, 0, true)], PAGE);

        assert_eq!(windows.as_slice(), &[Window { start: PAGE, end: 2 * PAGE, writable: true }]);
    }
}
