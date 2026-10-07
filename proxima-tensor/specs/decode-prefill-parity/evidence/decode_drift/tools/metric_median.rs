use std::collections::BTreeMap;
use std::fs;

fn field<'text>(line: &'text str, key: &str) -> Option<&'text str> {
    let start = line.find(key)? + key.len();
    let rest = &line[start..];
    Some(&rest[..rest.find(' ').unwrap_or(rest.len())])
}

fn main() {
    let mut arguments = std::env::args().skip(1);
    let path = arguments.next().expect("decode_arms.out path");
    let metric = arguments.next().expect("metric");
    let processes: Vec<usize> = arguments.map(|text| text.parse().expect("process index")).collect();
    let text = fs::read_to_string(&path).expect("read");
    let mut by_arm: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for line in text.lines().filter(|line| line.starts_with("arm=") && line.contains(&format!("metric={metric} "))) {
        let process: usize = field(line, "process=").expect("process").parse().expect("int");
        if processes.contains(&process) {
            by_arm.entry(field(line, "arm=").expect("arm").to_string()).or_default().push(field(line, "value=").expect("value").parse().expect("float"));
        }
    }
    for (arm, mut values) in by_arm {
        values.sort_by(|left, right| left.partial_cmp(right).expect("finite"));
        let middle = values.len() / 2;
        let median = if values.len() % 2 == 0 { (values[middle - 1] + values[middle]) / 2.0 } else { values[middle] };
        println!("{arm} n={} median={median:.3} min={:.3} max={:.3}", values.len(), values[0], values[values.len() - 1]);
    }
}
