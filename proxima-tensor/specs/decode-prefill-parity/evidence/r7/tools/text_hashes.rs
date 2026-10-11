use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::process::ExitCode;

fn label_of(path: &str) -> Option<(String, String)> {
    let name = path.rsplit('/').next()?;
    let middle = name.strip_prefix("launches.raw.")?;
    let label = middle.rsplit_once(".p")?.0;
    match label.rsplit_once('.') {
        Some((case, arm)) => Some((case.to_string(), arm.to_string())),
        None => Some((String::new(), label.to_string())),
    }
}

fn hashes_in(path: &str) -> Vec<String> {
    let text = fs::read_to_string(path).expect("read raw child stderr");
    text.lines()
        .filter(|line| line.contains("run=done"))
        .filter_map(|line| {
            let at = line.find(" text_hash=")? + " text_hash=".len();
            let rest = &line[at..];
            Some(rest[..rest.find(' ').unwrap_or(rest.len())].to_string())
        })
        .collect()
}

fn main() -> ExitCode {
    let paths: Vec<String> = env::args().skip(1).collect();
    let mut by_case: BTreeMap<String, BTreeMap<String, Vec<String>>> = BTreeMap::new();
    for path in &paths {
        let Some((case, arm)) = label_of(path) else { continue };
        by_case.entry(case).or_default().entry(arm).or_default().extend(hashes_in(path));
    }
    if by_case.is_empty() {
        println!("N==0 raw files matched; RED");
        return ExitCode::FAILURE;
    }
    let mut all_equal = true;
    for (case, arms) in &by_case {
        let mut union: BTreeSet<String> = BTreeSet::new();
        for (arm, hashes) in arms {
            let distinct: BTreeSet<&String> = hashes.iter().collect();
            println!("case={case} arm={arm} runs={} distinct_text_hashes={} hashes={:?}", hashes.len(), distinct.len(), distinct);
            if hashes.is_empty() {
                all_equal = false;
            }
            union.extend(hashes.iter().cloned());
        }
        let equal = union.len() == 1;
        println!("case={case} arms={} text_hash_equal_across_arms_and_runs={equal}", arms.len());
        all_equal &= equal;
    }
    println!("all_cases_equal={all_equal}");
    if all_equal { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}
