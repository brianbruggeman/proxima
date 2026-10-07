use std::collections::BTreeMap;
use std::fs;

fn median(values: &mut Vec<f64>) -> f64 {
    values.sort_by(|left, right| left.partial_cmp(right).expect("finite"));
    let middle = values.len() / 2;
    if values.len() % 2 == 0 { (values[middle - 1] + values[middle]) / 2.0 } else { values[middle] }
}

fn field<'text>(line: &'text str, key: &str) -> Option<&'text str> {
    let start = line.find(key)? + key.len();
    let rest = &line[start..];
    Some(&rest[..rest.find(' ').unwrap_or(rest.len())])
}

fn main() {
    let mut arguments = std::env::args().skip(1);
    let path = arguments.next().expect("decode_arms.out path");
    let wanted: Vec<String> = arguments.collect();
    let text = fs::read_to_string(&path).expect("read");
    let mut by_arm: BTreeMap<String, BTreeMap<usize, Vec<f64>>> = BTreeMap::new();
    let mut raw_lines = 0usize;
    for line in text.lines().filter(|line| line.starts_with("raw arm=")) {
        let arm = field(line, "arm=").expect("arm").to_string();
        let process: usize = field(line, "process=").expect("process").parse().expect("int");
        let run: usize = field(line, "run=").expect("run").parse().expect("int");
        let value: f64 = field(line, "ms_per_token=").expect("value").parse().expect("float");
        raw_lines += 1;
        if run > 0 && (wanted.is_empty() || wanted.iter().any(|name| arm.contains(name.as_str()))) {
            by_arm.entry(arm).or_default().entry(process).or_default().push(value);
        }
    }
    for (arm, processes) in &mut by_arm {
        let cells: Vec<String> = processes.iter_mut().map(|(process, values)| format!("p{process}={:.3}", median(values))).collect();
        println!("{arm}: {}", cells.join(" "));
    }
    eprintln!("raw lines parsed: {raw_lines}");
}
