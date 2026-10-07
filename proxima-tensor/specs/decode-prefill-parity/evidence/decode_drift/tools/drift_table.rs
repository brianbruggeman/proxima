use std::collections::BTreeMap;
use std::fs;

const MAPPING: [(&str, &str, &str); 20] = [
    ("a00_0b_9f0647da", "9f0647da (row 0b)", "named"),
    ("a01_0c_152467a9", "152467a9 (row 0c)", "named"),
    ("a02_slice5", "ecdd1bdd", "inferred from mtime 10-05 11:45"),
    ("a03_slice6", "tree before ce94f2d0", "inferred from mtime 10-05 15:20"),
    ("a04_slice7", "tree before 53064bff", "inferred from mtime 10-05 21:04"),
    ("a05_slice8", "3c42090d", "inferred from mtime 10-06 04:25"),
    ("a06_slice9", "bc686691", "inferred from mtime 10-06 09:48"),
    ("a07_slice10b", "tree before 5576a197", "inferred from mtime 10-06 13:15"),
    ("a08_slice10b_fix", "tree before 51df2ee7", "inferred from mtime 10-06 14:00"),
    ("a09_slice10c", "74bdbb87", "inferred from mtime 10-06 15:29"),
    ("a10_slice11", "44f48761 / f2aea8bc", "inferred from mtime 10-06 16:50"),
    ("a11_slice12", "e9f94b2e", "inferred from mtime 10-06 18:02"),
    ("a12_eor", "42f375e2", "inferred from mtime 10-06 20:12"),
    ("a13_f76b4a97", "f76b4a97", "named"),
    ("a14_tip_s1", "slice 1 tip, sha equals base_s2", "named, sha-identical to base_s2"),
    ("a15_s2_mml", "slice 2 tip 6e8729fe", "named s2_mml"),
    ("a16_s2fix_3a933038", "3a933038", "named"),
    ("a17_head_8ef12e94", "8ef12e94", "built this task"),
    ("a18_head_copy", "8ef12e94 byte copy", "control"),
    ("a19_0b_probe_9f0647da", "9f0647da probe build", "named"),
];

fn field<'text>(line: &'text str, key: &str) -> Option<&'text str> {
    let start = line.find(key)? + key.len();
    let rest = &line[start..];
    Some(&rest[..rest.find(' ').unwrap_or(rest.len())])
}

fn cell(line: &str) -> String {
    format!(
        "{:.1} / {:.1} / {:.2}",
        field(line, "median_all=").expect("median").parse::<f64>().expect("float"),
        field(line, "mad_kept=").expect("mad").parse::<f64>().expect("float"),
        field(line, "cov_all_pct=").expect("cov").parse::<f64>().expect("float"),
    )
}

fn decode_cell(line: &str) -> String {
    format!(
        "{:.3} / {:.3} / {:.2}",
        field(line, "median_all=").expect("median").parse::<f64>().expect("float"),
        field(line, "mad_kept=").expect("mad").parse::<f64>().expect("float"),
        field(line, "cov_all_pct=").expect("cov").parse::<f64>().expect("float"),
    )
}

fn main() {
    let path = std::env::args().nth(1).expect("decode_arms.out path");
    let text = fs::read_to_string(&path).expect("read output");
    let mut prefill: BTreeMap<String, String> = BTreeMap::new();
    let mut decode: BTreeMap<String, String> = BTreeMap::new();
    let mut count = 0usize;
    for line in text.lines().filter(|line| line.starts_with("summary arm=")) {
        let arm = field(line, "arm=").expect("arm").to_string();
        match field(line, "metric=").expect("metric") {
            "prefill_ms" => { prefill.insert(arm, cell(line)); count += 1; }
            "ms_per_token" => { decode.insert(arm, decode_cell(line)); count += 1; }
            _ => {}
        }
    }
    println!("| arm | commit | mapping | prefill ms: median / MAD / CoV% | decode ms/token: median / MAD / CoV% |");
    println!("|---|---|---|---|---|");
    for (arm, prefill_cell) in &prefill {
        let (commit, basis) = MAPPING.iter().find(|row| row.0 == arm).map_or(("reference", "llama-server f1ea20621"), |row| (row.1, row.2));
        println!("| {arm} | {commit} | {basis} | {prefill_cell} | {} |", decode.get(arm).expect("decode row"));
    }
    eprintln!("rows: {} (summary lines used: {count})", prefill.len());
}
