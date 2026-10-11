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

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).expect("finite"));
    sorted[sorted.len() / 2]
}

fn read(prefix: &str, processes: usize) -> BTreeMap<(String, String), (usize, Vec<f64>)> {
    let mut cells: BTreeMap<(String, String), (usize, Vec<f64>)> = BTreeMap::new();
    for process in 1..=processes {
        let path = format!("{prefix}_p{process}.out");
        for line in fs::read_to_string(&path).expect("read omit output").lines() {
            if !line.starts_with("ab omit ") {
                continue;
            }
            let key = (field(line, "entry").expect("entry"), field(line, "extents").expect("extents"));
            let count: usize = field(line, "count").expect("count").parse().expect("count value");
            let per_dispatch: f64 = field(line, "per_dispatch_us").expect("per dispatch").parse().expect("us");
            let cell = cells.entry(key).or_insert_with(|| (count, Vec::new()));
            cell.1.push(per_dispatch);
        }
    }
    cells
}

fn main() {
    let arguments: Vec<String> = env::args().collect();
    let [_, base_prefix, tip_prefix, processes] = &arguments[..] else {
        panic!("usage: omit_table <base prefix> <tip prefix> <processes>");
    };
    let processes: usize = processes.parse().expect("process count");
    let base = read(base_prefix, processes);
    let tip = read(tip_prefix, processes);
    println!("in-situ cost by omission (the step with the group removed against the whole step, program order, one command buffer, no cache flush); us per dispatch, per process p1 / p2 / p3");
    println!("| entry | extents | dispatches | base us per dispatch | tip us per dispatch | delta per dispatch (median of per-process pairs) | delta per step us (x dispatches) |");
    println!("|---|---|---|---|---|---|---|");
    let (mut total_delta, mut total_base) = (0.0f64, 0.0f64);
    for (key, (count, base_values)) in &base {
        let Some((_, tip_values)) = tip.get(key) else {
            continue;
        };
        let deltas: Vec<f64> = tip_values.iter().zip(base_values).map(|(after, before)| after - before).collect();
        let delta = median(&deltas);
        total_delta += delta * *count as f64;
        total_base += median(base_values) * *count as f64;
        let list = |values: &[f64]| values.iter().map(|value| format!("{value:.2}")).collect::<Vec<_>>().join(" / ");
        println!(
            "| {} | {} | {count} | {} | {} | {delta:+.2} | {:+.1} |",
            &key.0[..key.0.len().min(64)],
            key.1,
            list(base_values),
            list(tip_values),
            delta * *count as f64
        );
    }
    println!("sum over groups: base {total_base:.1} us (sum of group costs, not the step), delta {total_delta:+.1} us");
}
