use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::process::ExitCode;

const TIMING: [&str; 7] = ["ms_per_token", "prefill_ms", "ttft_ms", "wall_ms", "tokens_generated", "cpu_ms", "cpu_pct"];
const MEMORY: [&str; 6] = ["steady_rss_bytes", "steady_footprint_bytes", "steady_gpu_bytes", "peak_rss_bytes", "peak_footprint_bytes", "peak_gpu_bytes"];
const DELTA: [&str; 5] = ["ms_per_token", "ttft_ms", "wall_ms", "cpu_ms", "cpu_pct"];

#[derive(Clone, Copy)]
struct Stat {
    median: f64,
    min: f64,
    max: f64,
    cov_pct: f64,
}

struct Cell {
    stats: BTreeMap<(String, String), Stat>,
    summary_lines: usize,
    gpu_before_launch: BTreeMap<String, Vec<String>>,
    peers_nonempty: usize,
    peers_total: usize,
}

fn fields(line: &str) -> BTreeMap<&str, &str> {
    line.split_whitespace().filter_map(|token| token.split_once('=')).collect()
}

fn number(map: &BTreeMap<&str, &str>, key: &str) -> Option<f64> {
    map.get(key)?.parse().ok()
}

fn median_of(sorted: &[f64]) -> f64 {
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 1 { sorted[middle] } else { (sorted[middle - 1] + sorted[middle]) / 2.0 }
}

fn stat_of(values: &mut Vec<f64>) -> Option<Stat> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|left, right| left.total_cmp(right));
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / values.len() as f64;
    let cov_pct = if mean == 0.0 { 0.0 } else { variance.sqrt() / mean * 100.0 };
    Some(Stat { median: median_of(values), min: values[0], max: values[values.len() - 1], cov_pct })
}

fn load(path: &str) -> Result<Cell, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("read {path}: {error}"))?;
    let mut stats = BTreeMap::new();
    let mut memory: BTreeMap<(String, String), Vec<f64>> = BTreeMap::new();
    let mut gpu_before_launch: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let (mut summary_lines, mut peers_nonempty, mut peers_total) = (0, 0, 0);
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("summary ") {
            let map = fields(rest);
            if let (Some(arm), Some(metric), Some(median), Some(min), Some(max), Some(cov_pct)) =
                (map.get("arm"), map.get("metric"), number(&map, "median_all"), number(&map, "min_all"), number(&map, "max_all"), number(&map, "cov_all_pct"))
            {
                stats.insert((arm.to_string(), metric.to_string()), Stat { median, min, max, cov_pct });
                summary_lines += 1;
            }
        } else if let Some(rest) = line.strip_prefix("memory ") {
            let map = fields(rest);
            if let (Some(arm), Some(metric), Some(value)) = (map.get("arm"), map.get("metric"), number(&map, "value")) {
                memory.entry((arm.to_string(), metric.to_string())).or_default().push(value);
            }
        } else if let Some(rest) = line.strip_prefix("extra ") {
            let map = fields(rest);
            let arm = map.get("arm").copied().unwrap_or("").to_string();
            if let Some(value) = map.get("box_gpu_device_utilization_pct_before_launch") {
                gpu_before_launch.entry(arm).or_default().push((*value).to_string());
            } else if let Some(value) = map.get("peers_present_at_exit") {
                peers_total += 1;
                if *value != "[]" {
                    peers_nonempty += 1;
                }
            }
        }
    }
    for (key, mut values) in memory {
        if let Some(stat) = stat_of(&mut values) {
            stats.insert(key, stat);
        }
    }
    Ok(Cell { stats, summary_lines, gpu_before_launch, peers_nonempty, peers_total })
}

fn render(metric: &str, value: f64) -> String {
    if metric.ends_with("_gpu_bytes") {
        format!("{value:.0}")
    } else if metric.ends_with("_bytes") {
        format!("{:.1}", value / 1.0e6)
    } else if metric == "tokens_generated" {
        format!("{value:.0}")
    } else if metric == "ms_per_token" {
        format!("{value:.3}")
    } else {
        format!("{value:.1}")
    }
}

fn unit(metric: &str) -> &'static str {
    if metric.ends_with("_gpu_bytes") { " (B)" } else if metric.ends_with("_bytes") { " (MB)" } else { "" }
}

fn cases_and_arms(cell: &Cell) -> Vec<(String, String)> {
    let mut seen: Vec<(String, String)> = Vec::new();
    for (arm, _) in cell.stats.keys() {
        let (case, name) = arm.split_once('.').unwrap_or(("", arm));
        let pair = (case.to_string(), name.to_string());
        if !seen.contains(&pair) {
            seen.push(pair);
        }
    }
    seen.sort_by_key(|(case, name)| (case.clone(), ["base", "tip", "control"].iter().position(|known| known == name).unwrap_or(9)));
    seen
}

fn table(title: &str, cell: &Cell, metrics: &[&str]) {
    println!("{title} median [min-max] (CoV of all 15, % ; peak_* rows are 3 processes)");
    println!();
    let header: Vec<String> = metrics.iter().map(|metric| format!("{metric}{}", unit(metric))).collect();
    println!("| case | arm | {} |", header.join(" | "));
    println!("|---|---|{}", "---|".repeat(metrics.len()));
    for (case, name) in cases_and_arms(cell) {
        let arm = format!("{case}.{name}");
        let cells: Vec<String> = metrics
            .iter()
            .map(|metric| match cell.stats.get(&(arm.clone(), (*metric).to_string())) {
                Some(stat) => format!("{} [{}-{}] ({:.2})", render(metric, stat.median), render(metric, stat.min), render(metric, stat.max), stat.cov_pct),
                None => "MISSING".to_string(),
            })
            .collect();
        println!("| {case} | {name} | {} |", cells.join(" | "));
    }
    println!();
}

fn deltas(quiet: &Cell, loud: &Cell) {
    println!("loud (round seven lever-1 cells, same arm names) against quiet, median of all 15; delta = quiet minus loud");
    println!();
    println!("| case | arm | metric | loud median (CoV %) | quiet median (CoV %) | delta | delta as % of loud |");
    println!("|---|---|---|---|---|---|---|");
    for (case, name) in cases_and_arms(quiet) {
        let arm = format!("{case}.{name}");
        for metric in DELTA {
            let key = (arm.clone(), metric.to_string());
            let (Some(quiet_stat), Some(loud_stat)) = (quiet.stats.get(&key), loud.stats.get(&key)) else {
                println!("| {case} | {name} | {metric} | NO LOUD ARM OF THIS NAME | | | |");
                continue;
            };
            let delta = quiet_stat.median - loud_stat.median;
            let percent = if loud_stat.median == 0.0 { 0.0 } else { delta / loud_stat.median * 100.0 };
            println!(
                "| {case} | {name} | {metric} | {} ({:.2}) | {} ({:.2}) | {} | {percent:+.2} |",
                render(metric, loud_stat.median),
                loud_stat.cov_pct,
                render(metric, quiet_stat.median),
                quiet_stat.cov_pct,
                render(metric, delta)
            );
        }
    }
    println!();
}

fn box_lines(cell: &Cell) {
    println!("box-level GPU device utilization % sampled by decode_arms before each launch, per arm (ioreg, not the process):");
    for (arm, values) in &cell.gpu_before_launch {
        println!("- {arm}: {}", values.join(" "));
    }
    println!("peers_present_at_exit non-empty: {} of {}", cell.peers_nonempty, cell.peers_total);
    println!();
}

fn bounds(path: &str) -> Result<usize, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("read {path}: {error}"))?;
    let lines: Vec<&str> = text.lines().filter(|line| line.starts_with("bound ")).collect();
    println!("```");
    for line in &lines {
        println!("{line}");
    }
    println!("```");
    println!();
    Ok(lines.len())
}

fn run(args: &[String]) -> Result<(), String> {
    let [label, quiet_path, loud_path] = args else {
        return Err("usage: quiet_table <label> <quiet decode_arms.out> <loud decode_arms.out>".to_string());
    };
    let quiet = load(quiet_path)?;
    let loud = load(loud_path)?;
    println!("### {label}: {quiet_path}");
    println!();
    println!("summary lines read: quiet {} loud {}", quiet.summary_lines, loud.summary_lines);
    println!();
    if quiet.summary_lines == 0 || loud.summary_lines == 0 {
        return Err("N==0 summary lines; RED".to_string());
    }
    table("timing,", &quiet, &TIMING);
    table("memory,", &quiet, &MEMORY);
    box_lines(&quiet);
    deltas(&quiet, &loud);
    let count = bounds(quiet_path)?;
    println!("bound lines printed: {count}");
    if count == 0 {
        return Err("N==0 bound lines; RED".to_string());
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}
