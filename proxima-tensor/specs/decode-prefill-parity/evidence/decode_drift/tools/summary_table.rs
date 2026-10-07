use std::collections::BTreeMap;
use std::fs;

fn field<'text>(line: &'text str, key: &str) -> Option<&'text str> {
    let start = line.find(key)? + key.len();
    let rest = &line[start..];
    Some(&rest[..rest.find(' ').unwrap_or(rest.len())])
}

fn main() {
    let path = std::env::args().nth(1).expect("decode_arms.out path");
    let text = fs::read_to_string(&path).expect("read output");
    let mut table: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
    let mut summary_lines = 0usize;
    for line in text.lines().filter(|line| line.starts_with("summary arm=")) {
        let arm = field(line, "arm=").expect("arm").to_string();
        let metric = field(line, "metric=").expect("metric").to_string();
        if !["ms_per_token", "prefill_ms", "ttft_ms"].contains(&metric.as_str()) {
            continue;
        }
        summary_lines += 1;
        let cell = format!(
            "{} (kept {}, MAD {}, CoV all {}%, kept {}%; range {}-{})",
            field(line, "median_all=").expect("median_all"),
            field(line, "median_kept=").expect("median_kept"),
            field(line, "mad_kept=").expect("mad_kept"),
            field(line, "cov_all_pct=").expect("cov"),
            field(line, "cov_kept_pct=").expect("covk"),
            field(line, "min_all=").expect("min"),
            field(line, "max_all=").expect("max"),
        );
        table.entry(arm).or_default().insert(metric, cell);
    }
    println!("| arm | prefill ms: median all (kept, MAD, CoV, range) | decode ms/token: median all (kept, MAD, CoV, range) |");
    println!("|---|---|---|");
    for (arm, metrics) in &table {
        let none = String::from("n/a");
        println!("| {arm} | {} | {} |", metrics.get("prefill_ms").unwrap_or(&none), metrics.get("ms_per_token").unwrap_or(&none));
    }
    eprintln!("summary lines parsed: {summary_lines}, arms: {}", table.len());
}
