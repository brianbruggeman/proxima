use std::collections::BTreeMap;
use std::env;
use std::fs;

fn field(line: &str, name: &str) -> Option<String> {
    let marker = format!(" {name}=");
    let at = line.find(&marker)?;
    let rest = &line[at + marker.len()..];
    if rest.starts_with('[') {
        return Some(rest[..=rest.find(']')?].to_string());
    }
    Some(rest[..rest.find(' ').unwrap_or(rest.len())].to_string())
}

fn number(line: &str, name: &str) -> Option<f64> {
    field(line, name)?.parse().ok()
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).expect("finite"));
    sorted[sorted.len() / 2]
}

fn cov_percent(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / (values.len() as f64 - 1.0).max(1.0);
    100.0 * variance.sqrt() / mean
}

fn key_of(line: &str) -> Option<(String, String, String)> {
    let sha = field(line, "sha")?;
    let extents = field(line, "extents").unwrap_or_default();
    let arm = field(line, "arm")?;
    let arm = arm.strip_prefix(&sha).map_or(arm.clone(), |tail| tail.trim_start_matches('.').to_string());
    Some((sha[..8.min(sha.len())].to_string(), extents, if arm.is_empty() { "ctrl_name".into() } else { arm }))
}

fn main() {
    let arguments: Vec<String> = env::args().collect();
    let prefix = format!("ab {} ", arguments[1]);
    let value_key = &arguments[2];
    let files = &arguments[3..];
    let mut samples: BTreeMap<(String, String, String), Vec<f64>> = BTreeMap::new();
    let mut order: Vec<(String, String, String)> = Vec::new();
    let mut bits: BTreeMap<(String, String, String), String> = BTreeMap::new();
    let mut resources: BTreeMap<(String, String), (String, Vec<f64>, Vec<f64>, Vec<f64>, String)> = BTreeMap::new();
    for (index, path) in files.iter().enumerate() {
        for line in fs::read_to_string(path).expect("read ab output").lines() {
            if line.starts_with(&prefix) {
                if let (Some(key), Some(value)) = (key_of(line), number(line, value_key)) {
                    if index == 0 && !order.contains(&key) {
                        order.push(key.clone());
                    }
                    samples.entry(key).or_default().push(value);
                }
            } else if line.starts_with("ab bits ") && index == 0 {
                if let Some(key) = key_of(line) {
                    bits.insert(key, format!("differing={}/{} max_ulp={}", field(line, "differing").unwrap_or_default(), field(line, "elements").unwrap_or_default(), field(line, "max_ulp").unwrap_or_default()));
                }
            } else if line.starts_with("ab res ") {
                let (sha, arm) = (field(line, "sha").unwrap_or_default(), field(line, "arm").unwrap_or_default());
                let arm = arm.strip_prefix(&sha).map_or(arm.clone(), |tail| tail.trim_start_matches('.').to_string());
                let entry = resources.entry((sha[..8.min(sha.len())].to_string(), arm)).or_insert_with(|| {
                    (
                        format!(
                            "tg_static={} bound_buffer_mb={:.1} rss_peak_mb={} footprint_mb={} gpu_alloc_mb={} load={}",
                            field(line, "tg_static_bytes").unwrap_or_default(),
                            number(line, "bound_buffer_bytes").unwrap_or(0.0) / 1048576.0,
                            field(line, "rss_peak_mb").unwrap_or_default(),
                            field(line, "footprint_mb").unwrap_or_default(),
                            field(line, "gpu_alloc_mb").unwrap_or_default(),
                            field(line, "load_before").unwrap_or_default()
                        ),
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        String::new(),
                    )
                });
                entry.1.push(number(line, "cpu_ms").unwrap_or(0.0) / number(line, "iters").unwrap_or(1.0));
                entry.2.push(number(line, "cpu_pct").unwrap_or(0.0));
                entry.3.push(number(line, "wall_ms").unwrap_or(0.0) / number(line, "iters").unwrap_or(1.0));
            }
        }
    }
    assert!(!order.is_empty(), "N==0: no `{prefix}` lines with `{value_key}`");
    println!("{} files; value `{value_key}` from `{}` lines; median [CoV%] (min-max) over runs", files.len(), arguments[1]);
    println!("| group | extents | arm | n | {value_key} median [CoV%] (min-max) | bit compare vs base (first file) | cpu ms per replay | cpu % of wall | wall ms per replay | static tg bytes, bound buffer MB, rss peak MB, footprint MB, Metal MB, load |");
    println!("|---|---|---|---|---|---|---|---|---|---|");
    for key in &order {
        let values = &samples[key];
        let arm_key = (key.0.clone(), key.2.clone());
        let resource = resources.get(&arm_key);
        println!(
            "| {} | {} | {} | {} | {:.3} [{:.2}] ({:.3}-{:.3}) | {} | {} | {} | {} | {} |",
            key.0, key.1, key.2, values.len(), median(values), cov_percent(values),
            values.iter().copied().fold(f64::INFINITY, f64::min),
            values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            bits.get(key).cloned().unwrap_or_else(|| "-".into()),
            resource.map_or("-".into(), |entry| format!("{:.3}", median(&entry.1))),
            resource.map_or("-".into(), |entry| format!("{:.1}", median(&entry.2))),
            resource.map_or("-".into(), |entry| format!("{:.3}", median(&entry.3))),
            resource.map_or("-".into(), |entry| entry.0.clone())
        );
    }
}
