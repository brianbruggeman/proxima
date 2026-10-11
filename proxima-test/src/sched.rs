//! Scheduling metadata a `#[proxima::test]` writes into the test binary so an
//! external runner can read it from disk without executing anything.
//!
//! The macro emits one [`SchedRecord`] per generated `#[test]` fn as a
//! `#[used]` static in a dedicated linker section; nothing runs at test time.
//! Plain data composed from `core` only (no allocation, no `std`), so the
//! record is usable at every tier the macro-facing surface of this crate is.
//!
//! # Section names
//!
//! | object format | section                                             |
//! |---------------|-----------------------------------------------------|
//! | Mach-O        | `__DATA,__prox_test,regular,no_dead_strip`          |
//! | ELF           | `.proxima_test`                                     |
//! | PE / COFF     | `.prxtst$m` (the linker merges `$`-suffixed members into `.prxtst`) |
//!
//! # Reader contract
//!
//! The section is a flat array of [`SchedRecord`]: `size_of::<SchedRecord>()`
//! (328) bytes per element, 4-byte aligned, no inter-record padding, no
//! relocations (inline byte arrays, no pointers). Multi-byte fields use the
//! target's native endianness, which the object file declares. A reader
//! rejects any record whose `version` it does not know, ignores zero-length
//! tails, and decodes `group[..group_len]` / `path[..path_len]` as UTF-8.
//!
//! `path` is the libtest name exactly as `cargo test -- --list` prints it:
//! the module path with the crate-name segment removed, joined by `::` to the
//! fn name (or, for a parameterized test, to `<fn>::<case>`). The crate
//! segment is removed here at compile time, so the reader never strips.
//!
//! Teaching pointer: `#[proxima::test(serial)]`, `#[proxima::test(group = "...")]` and
//! `#[proxima::test(process)]` in `proxima-macros` are the only producer. Runner
//! equivalents: `serial` is nextest `threads-required = "num-test-threads"`;
//! `group` is a nextest test-group (at most N of the group at once, one process
//! per test, N from runner config, default 1); `process` makes tests share one
//! process per test binary, either every test of the same `group` or, with no
/// Bumped on any layout change; a reader refuses versions it does not know.
pub const SCHED_VERSION: u32 = 2;

/// Capacity of [`SchedRecord::group`] in bytes.
pub const SCHED_GROUP_CAP: usize = 64;

/// Capacity of [`SchedRecord::path`] in bytes.
pub const SCHED_PATH_CAP: usize = 256;

/// `serial`: no other test may be in flight (bit 0).
pub const SCHED_FLAG_SERIAL: u8 = 1;

/// `process`: tests share one process per test binary (bit 1).
pub const SCHED_FLAG_PROCESS: u8 = 2;

/// `group = "<name>"` was declared; `group[..group_len]` holds the name (bit 2).
pub const SCHED_FLAG_HAS_GROUP: u8 = 4;

/// One test's scheduling record. Fixed size, `repr(C)`, plain data.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SchedRecord {
    /// layout version, `SCHED_VERSION` (2)
    pub version: u32,
    /// bit 0 `SCHED_FLAG_SERIAL`, bit 1 `SCHED_FLAG_PROCESS`, bit 2
    /// `SCHED_FLAG_HAS_GROUP`; `0b110` is `group = "gpu", process`, `0b011` is
    /// `serial, process` with no group (one shared binary process running alone)
    pub flags: u8,
    /// used bytes of `group`; 0 unless `SCHED_FLAG_HAS_GROUP` is set, e.g. 15
    /// for `real-checkpoint`
    pub group_len: u8,
    /// used bytes of `path`, e.g. 41 for `loads::real_checkpoint::case_1`
    pub path_len: u16,
    /// UTF-8 nextest test-group name, zero padded, e.g. `real-checkpoint`.
    /// With `process` set, tests share one process only inside one test
    /// binary: the same name in the same binary, or, when this is empty, every
    /// `process` test of the binary (the runner's `@binary:<binary>` group).
    pub group: [u8; SCHED_GROUP_CAP],
    /// UTF-8 libtest path, zero padded, e.g. `gpu::decode::case_1`
    pub path: [u8; SCHED_PATH_CAP],
}

const _: () = assert!(core::mem::size_of::<SchedRecord>() == 328);
const _: () = assert!(core::mem::align_of::<SchedRecord>() == 4);

impl SchedRecord {
    /// Builds the record for one test.
    ///
    /// `flags` is an OR of `SCHED_FLAG_SERIAL` and `SCHED_FLAG_PROCESS`;
    /// `SCHED_FLAG_HAS_GROUP` is derived from `group`. `module_path` is
    /// `module_path!()` at the test fn (crate segment included); `leaf` is the
    /// fn name, or the case name for a parameterized case (whose
    /// `module_path!()` already ends in the fn name).
    ///
    /// Evaluated in a `static` an oversized path or group, an empty group, or
    /// an unknown flag bit is a compile error, never a runtime failure.
    ///
    /// # Panics
    ///
    /// Only at const-evaluation time of a `static` (a compile error there): the
    /// path or group exceeds its capacity, the group is empty, or `flags`
    /// carries a bit other than serial or process.
    #[must_use]
    pub const fn new(flags: u8, group: Option<&str>, module_path: &str, leaf: &str) -> Self {
        assert!(
            flags & !(SCHED_FLAG_SERIAL | SCHED_FLAG_PROCESS) == 0,
            "proxima::test flags may only combine SCHED_FLAG_SERIAL and SCHED_FLAG_PROCESS"
        );
        let (group_bytes, group_flag) = match group {
            Some(name) => {
                assert!(
                    !name.is_empty(),
                    "proxima::test group name must not be empty"
                );
                (name.as_bytes(), SCHED_FLAG_HAS_GROUP)
            }
            None => (&[] as &[u8], 0),
        };
        assert!(
            group_bytes.len() <= SCHED_GROUP_CAP,
            "proxima::test group name exceeds SCHED_GROUP_CAP bytes"
        );
        let mut group_buf = [0u8; SCHED_GROUP_CAP];
        copy_into(&mut group_buf, 0, group_bytes, 0);

        let module_bytes = module_path.as_bytes();
        let rest_start = after_crate_segment(module_bytes);
        let rest_len = module_bytes.len() - rest_start;
        let leaf_bytes = leaf.as_bytes();
        let separator = if rest_len == 0 { 0 } else { 2 };
        let path_len = rest_len + separator + leaf_bytes.len();
        assert!(
            path_len <= SCHED_PATH_CAP,
            "proxima::test libtest path exceeds SCHED_PATH_CAP bytes"
        );
        let mut path_buf = [0u8; SCHED_PATH_CAP];
        let mut cursor = copy_into(&mut path_buf, 0, module_bytes, rest_start);
        if separator != 0 {
            path_buf[cursor] = b':';
            path_buf[cursor + 1] = b':';
            cursor += 2;
        }
        copy_into(&mut path_buf, cursor, leaf_bytes, 0);

        Self {
            version: SCHED_VERSION,
            flags: flags | group_flag,
            group_len: count_u8(group_bytes.len()),
            path_len: count_u16(path_len),
            group: group_buf,
            path: path_buf,
        }
    }

    /// `serial`: no other test is in flight.
    #[must_use]
    pub const fn is_serial(&self) -> bool {
        self.flags & SCHED_FLAG_SERIAL != 0
    }

    /// `process`: tests share one process per test binary.
    #[must_use]
    pub const fn is_process(&self) -> bool {
        self.flags & SCHED_FLAG_PROCESS != 0
    }

    /// The declared `group` name bytes, or `None` when no group was declared.
    #[must_use]
    pub fn group_name(&self) -> Option<&[u8]> {
        (self.flags & SCHED_FLAG_HAS_GROUP != 0).then(|| &self.group[..usize::from(self.group_len)])
    }
}

// u8::try_from is not const and a cast truncates; counting up in the target
// type overflows (a const-eval error) instead of wrapping
const fn count_u8(length: usize) -> u8 {
    let mut counted = 0u8;
    let mut index = 0;
    while index < length {
        counted += 1;
        index += 1;
    }
    counted
}

const fn count_u16(length: usize) -> u16 {
    let mut counted = 0u16;
    let mut index = 0;
    while index < length {
        counted += 1;
        index += 1;
    }
    counted
}

/// Offset just past the first `::` (the crate segment), or the full length
/// when the path is the bare crate name.
const fn after_crate_segment(module_bytes: &[u8]) -> usize {
    let mut index = 0;
    while index + 1 < module_bytes.len() {
        if module_bytes[index] == b':' && module_bytes[index + 1] == b':' {
            return index + 2;
        }
        index += 1;
    }
    module_bytes.len()
}

/// Copies `source[source_from..]` into `buffer` at `at`; returns the new end.
const fn copy_into(buffer: &mut [u8], at: usize, source: &[u8], source_from: usize) -> usize {
    let mut index = source_from;
    let mut cursor = at;
    while index < source.len() {
        buffer[cursor] = source[index];
        index += 1;
        cursor += 1;
    }
    cursor
}

#[cfg(test)]
mod tests {
    use super::*;

    const NO_FLAGS: u8 = 0;

    fn path_of(record: &SchedRecord) -> &str {
        core::str::from_utf8(&record.path[..usize::from(record.path_len)])
            .unwrap_or("<path bytes are not utf-8>")
    }

    #[test]
    fn layout_is_328_bytes_four_aligned() {
        assert_eq!(core::mem::size_of::<SchedRecord>(), 328);
        assert_eq!(core::mem::align_of::<SchedRecord>(), 4);
    }

    #[test]
    fn version_is_two() {
        let record = SchedRecord::new(NO_FLAGS, None, "proxima::t", "a");

        assert_eq!(record.version, 2);
        assert_eq!(SCHED_VERSION, 2);
    }

    #[test]
    fn serial_alone_sets_only_bit_zero() {
        let record = SchedRecord::new(SCHED_FLAG_SERIAL, None, "model_loads", "loads_gemma");

        assert_eq!(path_of(&record), "loads_gemma");
        assert_eq!(record.flags, 0b001);
        assert_eq!(record.group_len, 0);
        assert_eq!(record.group_name(), None);
    }

    #[test]
    fn no_flags_leaves_the_byte_zero_and_drops_the_crate_segment() {
        let record = SchedRecord::new(
            NO_FLAGS,
            None,
            "proxima::gpu::decode",
            "q4_0_matches_oracle",
        );

        assert_eq!(path_of(&record), "gpu::decode::q4_0_matches_oracle");
        assert_eq!(record.flags, 0);
        assert!(!record.is_serial() && !record.is_process());
    }

    #[test]
    fn group_alone_sets_only_the_has_group_bit() {
        let record = SchedRecord::new(
            NO_FLAGS,
            Some("real-checkpoint"),
            "sched_records::loads",
            "small",
        );

        assert_eq!(path_of(&record), "loads::small");
        assert_eq!(record.flags, 0b100);
        assert_eq!(record.group_name(), Some(&b"real-checkpoint"[..]));
        assert_eq!(record.group_len, 15);
    }

    #[test]
    fn process_alone_sets_bit_one_with_an_empty_group() {
        let record = SchedRecord::new(SCHED_FLAG_PROCESS, None, "crate_a", "shares_binary");

        assert_eq!(record.flags, 0b010);
        assert!(record.is_process());
        assert_eq!(record.group_name(), None);
    }

    #[test]
    fn process_with_group_sets_bits_one_and_two() {
        let record = SchedRecord::new(SCHED_FLAG_PROCESS, Some("gpu"), "crate_a", "t");

        assert_eq!(record.flags, 0b110);
        assert_eq!(record.group_name(), Some(&b"gpu"[..]));
    }

    #[test]
    fn serial_with_process_and_no_group_sets_bits_zero_and_one() {
        let record = SchedRecord::new(
            SCHED_FLAG_SERIAL | SCHED_FLAG_PROCESS,
            None,
            "crate_a",
            "alone_in_shared_process",
        );

        assert_eq!(record.flags, 0b011);
        assert!(record.is_serial() && record.is_process());
        assert_eq!(record.group_name(), None);
    }

    #[test]
    fn unused_tail_is_zero_filled() {
        let record = SchedRecord::new(NO_FLAGS, Some("gpu"), "crate_a", "t");

        assert!(record.group[3..].iter().all(|&byte| byte == 0));
        assert!(record.path[1..].iter().all(|&byte| byte == 0));
    }

    #[test]
    fn path_at_capacity_records_its_full_length() {
        let leaf = "x".repeat(SCHED_PATH_CAP);
        let record = SchedRecord::new(NO_FLAGS, None, "crate_a", &leaf);

        assert_eq!(usize::from(record.path_len), SCHED_PATH_CAP);
    }

    #[test]
    #[should_panic(expected = "SCHED_PATH_CAP")]
    fn path_over_capacity_is_refused() {
        let long_leaf = "x".repeat(SCHED_PATH_CAP + 1);
        let _ = SchedRecord::new(SCHED_FLAG_SERIAL, None, "crate_a", &long_leaf);
    }

    #[test]
    #[should_panic(expected = "SCHED_GROUP_CAP")]
    fn group_over_capacity_is_refused() {
        let long_group = "g".repeat(SCHED_GROUP_CAP + 1);
        let _ = SchedRecord::new(NO_FLAGS, Some(&long_group), "crate_a", "t");
    }

    #[test]
    #[should_panic(expected = "must not be empty")]
    fn empty_group_is_refused() {
        let _ = SchedRecord::new(NO_FLAGS, Some(""), "crate_a", "t");
    }

    #[test]
    #[should_panic(expected = "may only combine")]
    fn caller_supplied_has_group_bit_is_refused() {
        let _ = SchedRecord::new(SCHED_FLAG_HAS_GROUP, None, "crate_a", "t");
    }
}
