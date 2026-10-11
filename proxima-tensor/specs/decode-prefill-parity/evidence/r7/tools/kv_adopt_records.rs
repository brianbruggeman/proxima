use std::collections::BTreeMap;
use std::env;
use std::fs;

#[derive(Default, Clone)]
struct Generation {
    acquire_ns: BTreeMap<String, Vec<f64>>,
    bytes: BTreeMap<String, u64>,
    write_ns: Vec<f64>,
    zero_ns: Vec<f64>,
    zero_bytes: Vec<f64>,
    adopt_ms: f64,
    bound: u64,
    allocated: u64,
}

fn field(line: &str, name: &str) -> String {
    let marker = format!(" {name}=");
    let Some(at) = line.find(&marker) else { return String::new() };
    let rest = &line[at + marker.len()..];
    rest[..rest.find(' ').unwrap_or(rest.len())].to_string()
}

fn number(line: &str, name: &str) -> f64 {
    field(line, name).parse().unwrap_or(f64::NAN)
}

fn median(values: &[f64]) -> f64 {
    assert!(!values.is_empty(), "median over N==0 values");
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).expect("finite"));
    sorted[sorted.len() / 2]
}

fn minimum(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::INFINITY, f64::min)
}

fn maximum(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

fn parse(path: &str) -> Vec<Generation> {
    let text = fs::read_to_string(path).expect("read telemetry log");
    let mut generations = Vec::new();
    let mut current = Generation::default();
    for line in text.lines() {
        if line.contains(" kv_adopt_buffer ") {
            let kind = field(line, "buffer");
            current.bytes.insert(kind.clone(), number(line, "bytes") as u64);
            current.acquire_ns.entry(kind).or_default().push(number(line, "acquire_ns"));
        } else if line.contains(" kv_adopt_seed ") {
            current.write_ns.push(number(line, "write_ns"));
            current.zero_ns.push(number(line, "zero_ns"));
            current.zero_bytes.push(number(line, "zero_bytes"));
        } else if line.contains("named_blocks_phase") && field(line, "phase") == "device_kv_adopt" {
            current.adopt_ms = number(line, "ms");
            current.bound = number(line, "buffers_bound") as u64;
            current.allocated = number(line, "buffers_allocated") as u64;
            generations.push(std::mem::take(&mut current));
        }
    }
    generations
}

fn total_ms(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / 1e6
}

fn acquire_total_ms(generation: &Generation) -> f64 {
    generation.acquire_ns.values().map(|values| total_ms(values)).sum()
}

fn summarize(label: &str, values: &[f64]) -> String {
    format!("{label} median {:.3} (min {:.3}, max {:.3}, N={})", median(values), minimum(values), maximum(values), values.len())
}

fn main() {
    let paths: Vec<String> = env::args().skip(1).collect();
    assert!(!paths.is_empty(), "usage: kv_adopt_records <telemetry.log>...");
    let mut warm: Vec<Generation> = Vec::new();
    println!("| process | generation | bound | allocated | adopt ms | acquire sum ms | seed write sum ms | seed zero sum ms | unattributed ms |");
    println!("|---|---|---|---|---|---|---|---|---|");
    for (process, path) in paths.iter().enumerate() {
        let generations = parse(path);
        assert!(!generations.is_empty(), "N==0 adopt events in {path}");
        for (index, generation) in generations.iter().enumerate() {
            let acquire = acquire_total_ms(generation);
            let write = total_ms(&generation.write_ns);
            let zero = total_ms(&generation.zero_ns);
            println!(
                "| {} | {} | {} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |",
                process + 1,
                index + 1,
                generation.bound,
                generation.allocated,
                generation.adopt_ms,
                acquire,
                write,
                zero,
                generation.adopt_ms - acquire - write - zero
            );
            if index > 0 {
                warm.push(generation.clone());
            }
        }
    }
    assert!(!warm.is_empty(), "N==0 warm generations");
    println!();
    println!("warm generations (generation 2 onward), N={}:", warm.len());
    let column = |select: &dyn Fn(&Generation) -> f64| -> Vec<f64> { warm.iter().map(select).collect() };
    println!("- {}", summarize("adopt ms", &column(&|generation| generation.adopt_ms)));
    println!("- {}", summarize("acquire sum ms", &column(&|generation| acquire_total_ms(generation))));
    println!("- {}", summarize("seed write sum ms", &column(&|generation| total_ms(&generation.write_ns))));
    println!("- {}", summarize("seed zero sum ms", &column(&|generation| total_ms(&generation.zero_ns))));
    println!("- {}", summarize("buffers bound", &column(&|generation| generation.bound as f64)));
    println!("- {}", summarize("buffers allocated", &column(&|generation| generation.allocated as f64)));
    println!();
    println!("| buffer | bytes each | calls per generation | acquire us median | min | max | per generation sum ms median |");
    println!("|---|---|---|---|---|---|---|");
    let kinds: Vec<String> = warm[0].acquire_ns.keys().cloned().collect();
    for kind in kinds {
        let calls: Vec<f64> = warm.iter().flat_map(|generation| generation.acquire_ns[&kind].iter().copied()).collect();
        let sums: Vec<f64> = warm.iter().map(|generation| total_ms(&generation.acquire_ns[&kind])).collect();
        println!(
            "| {kind} | {} | {} | {:.1} | {:.1} | {:.1} | {:.3} |",
            warm[0].bytes[&kind],
            warm[0].acquire_ns[&kind].len(),
            median(&calls) / 1e3,
            minimum(&calls) / 1e3,
            maximum(&calls) / 1e3,
            median(&sums)
        );
    }
    let zero_calls: Vec<f64> = warm.iter().flat_map(|generation| generation.zero_ns.iter().copied()).collect();
    let write_calls: Vec<f64> = warm.iter().flat_map(|generation| generation.write_ns.iter().copied()).collect();
    let zero_bytes = warm[0].zero_bytes.first().copied().unwrap_or(0.0);
    println!();
    println!(
        "seed per layer (all three buffers): zero {:.1} us median (min {:.1}, max {:.1}) over {zero_bytes:.0} bytes; write {:.1} us median",
        median(&zero_calls) / 1e3,
        minimum(&zero_calls) / 1e3,
        maximum(&zero_calls) / 1e3,
        median(&write_calls) / 1e3
    );
}
