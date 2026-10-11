//! `#[proxima::test(serial | group = .. | process)]` writes one scheduling record per
//! generated test into a linker section. This test opens its own executable
//! from disk, as an external runner would, and parses that section with the
//! `object` crate — no pointers followed, nothing executed.
#![allow(clippy::expect_used)]

use std::collections::BTreeSet;
use std::process::Command;

use object::{Endianness, Object, ObjectSection};

const RECORD_SIZE: usize = 328;
const VERSION: u32 = 2;
const FLAGS_NONE: u8 = 0b000;
const FLAG_SERIAL: u8 = 0b001;
const FLAG_PROCESS: u8 = 0b010;
const FLAG_HAS_GROUP: u8 = 0b100;

#[cfg(target_vendor = "apple")]
const SECTION_NAME: &str = "__prox_test";
#[cfg(windows)]
const SECTION_NAME: &str = ".prxtst";
#[cfg(not(any(target_vendor = "apple", windows)))]
const SECTION_NAME: &str = ".proxima_test";

#[proxima::test(runtime = "tokio", serial)]
async fn loads_whole_checkpoint() {
    assert_eq!(2 + 2, 4);
}

#[proxima::test(runtime = "tokio", group = "real-checkpoint")]
#[case::small(1u32)]
#[case::large(4_000_000_000u32)]
async fn decodes_each_size(#[case] value: u32) {
    assert!(value > 0);
}

#[proxima::test(runtime = "tokio", process)]
async fn shares_the_binary_process() {
    assert_eq!(1 + 2, 3);
}

#[proxima::test(runtime = "tokio", group = "gpu", process)]
async fn shares_the_gpu_group_process() {
    assert_eq!(3 * 3, 9);
}

#[proxima::test(runtime = "tokio", serial, process)]
async fn runs_alone_in_the_shared_process() {
    assert_eq!(4 - 1, 3);
}

mod nested {
    #[proxima::test(runtime = "tokio")]
    async fn unconstrained_in_a_module() {
        assert_eq!(u8::MAX, 255);
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Parsed {
    path: String,
    flags: u8,
    group: String,
}

fn parsed(path: &str, flags: u8, group: &str) -> Parsed {
    Parsed {
        path: path.to_string(),
        flags,
        group: group.to_string(),
    }
}

fn read_u16(bytes: &[u8], endianness: Endianness) -> u16 {
    let raw: [u8; 2] = bytes.try_into().expect("two bytes");
    match endianness {
        Endianness::Little => u16::from_le_bytes(raw),
        Endianness::Big => u16::from_be_bytes(raw),
    }
}

fn read_u32(bytes: &[u8], endianness: Endianness) -> u32 {
    let raw: [u8; 4] = bytes.try_into().expect("four bytes");
    match endianness {
        Endianness::Little => u32::from_le_bytes(raw),
        Endianness::Big => u32::from_be_bytes(raw),
    }
}

fn parse_section_of_own_executable() -> Vec<Parsed> {
    let exe = std::env::current_exe().expect("test executable path");
    let bytes = std::fs::read(&exe).expect("read own test executable");
    let file = object::File::parse(&*bytes).expect("parse own executable as an object file");
    let endianness = file.endianness();
    let section = file
        .sections()
        .find(|section| section.name() == Ok(SECTION_NAME))
        .expect("scheduling section present in the executable");
    let data = section.data().expect("section bytes");

    let (records, remainder) = data.as_chunks::<RECORD_SIZE>();
    assert!(remainder.is_empty(), "section is a whole number of records");
    records
        .iter()
        .map(|record| {
            assert_eq!(read_u32(&record[0..4], endianness), VERSION);
            let flags = record[4];
            let group_len = usize::from(record[5]);
            let path_len = usize::from(read_u16(&record[6..8], endianness));
            let group = &record[8..72];
            let path = &record[72..328];
            assert!(group[group_len..].iter().all(|byte| *byte == 0));
            assert!(path[path_len..].iter().all(|byte| *byte == 0));
            parsed(
                std::str::from_utf8(&path[..path_len]).expect("path is utf-8"),
                flags,
                std::str::from_utf8(&group[..group_len]).expect("group is utf-8"),
            )
        })
        .collect()
}

fn libtest_listing_of_own_executable() -> BTreeSet<String> {
    let exe = std::env::current_exe().expect("test executable path");
    let output = Command::new(exe)
        .args(["--list", "--format", "terse"])
        .output()
        .expect("run own executable with --list");
    assert!(output.status.success(), "--list exits cleanly");
    String::from_utf8(output.stdout)
        .expect("listing is utf-8")
        .lines()
        .filter_map(|line| line.strip_suffix(": test"))
        .map(str::to_string)
        .collect()
}

#[test]
fn section_holds_exactly_the_declared_records() {
    let mut records = parse_section_of_own_executable();
    records.sort();

    assert_eq!(
        records,
        vec![
            parsed(
                "decodes_each_size::large",
                FLAG_HAS_GROUP,
                "real-checkpoint"
            ),
            parsed(
                "decodes_each_size::small",
                FLAG_HAS_GROUP,
                "real-checkpoint"
            ),
            parsed("loads_whole_checkpoint", FLAG_SERIAL, ""),
            parsed("nested::unconstrained_in_a_module", FLAGS_NONE, ""),
            parsed(
                "runs_alone_in_the_shared_process",
                FLAG_SERIAL | FLAG_PROCESS,
                ""
            ),
            parsed("shares_the_binary_process", FLAG_PROCESS, ""),
            parsed(
                "shares_the_gpu_group_process",
                FLAG_HAS_GROUP | FLAG_PROCESS,
                "gpu"
            ),
        ]
    );
}

#[test]
fn record_paths_are_exactly_what_libtest_lists() {
    let listed = libtest_listing_of_own_executable();
    let record_paths: BTreeSet<String> = parse_section_of_own_executable()
        .into_iter()
        .map(|record| record.path)
        .collect();

    let proxima_tests: BTreeSet<String> = listed
        .into_iter()
        .filter(|name| !name.starts_with("section_holds") && !name.starts_with("record_paths"))
        .collect();
    assert_eq!(record_paths, proxima_tests);
}
