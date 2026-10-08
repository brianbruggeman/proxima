use std::collections::BTreeMap;
use std::env;
use std::fs;

fn top_of_stack(path: &str) -> (BTreeMap<String, u64>, u64) {
    let text = fs::read_to_string(path).expect("read sample file");
    let mut counts = BTreeMap::new();
    let mut inside = false;
    for line in text.lines() {
        if line.starts_with("Sort by top of stack") {
            inside = true;
            continue;
        }
        if inside && line.starts_with("Binary Images") {
            break;
        }
        if !inside || line.trim().is_empty() {
            continue;
        }
        let trimmed = line.trim();
        let Some(split) = trimmed.rfind(char::is_whitespace) else { continue };
        let Ok(count) = trimmed[split..].trim().parse::<u64>() else { continue };
        let symbol = trimmed[..split].trim().replace("decode_gbps_baseline_base", "BIN").replace("decode_gbps_baseline_tip", "BIN");
        *counts.entry(symbol).or_insert(0) += count;
    }
    let busy = counts
        .iter()
        .filter(|(symbol, _)| !symbol.starts_with("__psynch_cvwait") && !symbol.starts_with("__workq_kernreturn") && !symbol.starts_with("start_wqthread"))
        .map(|(_, count)| *count)
        .sum();
    (counts, busy)
}

fn short(symbol: &str) -> String {
    symbol.chars().take(150).collect()
}

fn main() {
    let arguments: Vec<String> = env::args().collect();
    let (base, base_busy) = top_of_stack(&arguments[1]);
    let (tip, tip_busy) = top_of_stack(&arguments[2]);
    println!("top-of-stack samples that are not parked waits: base {base_busy}, tip {tip_busy}");
    let mut rows: Vec<(i64, u64, u64, String)> = base
        .keys()
        .chain(tip.keys())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .filter(|symbol| !symbol.starts_with("__psynch_cvwait") && !symbol.starts_with("__workq_kernreturn") && !symbol.starts_with("start_wqthread"))
        .map(|symbol| {
            let left = base.get(symbol).copied().unwrap_or(0);
            let right = tip.get(symbol).copied().unwrap_or(0);
            (right as i64 - left as i64, left, right, symbol.clone())
        })
        .collect();
    rows.sort_by_key(|row| -row.0.abs());
    println!("| delta (tip - base) | base | tip | frame (top of stack) |");
    println!("|---|---|---|---|");
    for (delta, left, right, symbol) in rows.iter().take(25) {
        println!("| {delta:+} | {left} | {right} | {} |", short(symbol));
    }
}
