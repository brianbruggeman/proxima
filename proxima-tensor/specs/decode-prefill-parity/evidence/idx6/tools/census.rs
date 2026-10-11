use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

struct Group {
    sha: String,
    extents: String,
    count: usize,
    entry: String,
    grid_threads: String,
    width: String,
}

#[derive(Default)]
struct Tally {
    div_rem_i64_runtime_once: usize,
    div_rem_i64_runtime_loop: usize,
    div_rem_i64_const_once: usize,
    div_rem_i64_const_loop: usize,
    div_rem_i32_once: usize,
    div_rem_i32_loop: usize,
    mul_i64_once: usize,
    mul_i64_loop: usize,
    mul_i32_loop: usize,
    addsub_i64_loop: usize,
    phi_i64_loop: usize,
    phi_i32_loop: usize,
    loops: usize,
    divrem_i64_by_depth: [usize; 3],
    mul_i64_by_depth: [usize; 3],
}

fn parse_groups(path: &Path) -> Vec<Group> {
    let text = fs::read_to_string(path).expect("read describe output");
    let mut groups: Vec<Group> = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("ab describe sha=") {
            let sha = rest[..16].to_string();
            let extents_start = rest.find("extents=").expect("extents") + 8;
            let extents_end = rest.find(" count=").expect("count");
            let count_start = extents_end + 7;
            let entry_start = rest.find(" entry=").expect("entry");
            groups.push(Group {
                sha,
                extents: rest[extents_start..extents_end].replace(", ", "x").replace(['[', ']'], ""),
                count: rest[count_start..entry_start].parse().expect("count value"),
                entry: rest[entry_start + 7..].to_string(),
                grid_threads: String::new(),
                width: String::new(),
            });
        } else if line.contains("uniform_i64=") {
            if let Some(group) = groups.last_mut() {
                if let Some(at) = line.find("grid_threads=") {
                    let tail = &line[at + 13..];
                    let (grid, width) = tail.split_once(" threadgroup_width=").expect("width");
                    group.grid_threads = grid.to_string();
                    group.width = width.replace("Some(", "").replace(')', "").replace("None", "0");
                }
            }
        }
    }
    groups
}

fn compile_ir(source: &Path, out_dir: &Path, level: &str) -> PathBuf {
    let stem = source.file_stem().unwrap().to_string_lossy().to_string();
    let target = out_dir.join(format!("{stem}.{level}.ll"));
    let optimization = format!("-{level}");
    let output = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-S", "-emit-llvm", &optimization, "-ffast-math", "-Wno-unused-function", "-o"])
        .arg(&target)
        .arg(source)
        .output()
        .expect("run metal compiler");
    assert!(output.status.success(), "metal failed for {}", source.display());
    target
}

fn label_of(line: &str) -> Option<String> {
    if line.starts_with(' ') || line.starts_with(';') || line.starts_with('}') || line.starts_with('!') {
        return None;
    }
    let head = line.split(';').next().unwrap().trim_end();
    let name = head.strip_suffix(':')?;
    name.chars().all(|ch| ch.is_alphanumeric() || "_.$-".contains(ch)).then(|| name.to_string())
}

fn targets_of(line: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut rest = line;
    while let Some(at) = rest.find("label %") {
        let tail = &rest[at + 7..];
        let end = tail.find([',', ' ', ']']).unwrap_or(tail.len());
        found.push(tail[..end].to_string());
        rest = &tail[end..];
    }
    found
}

struct Instruction {
    block: usize,
    opcode: String,
    ty: String,
    divisor_constant: bool,
}

fn parse_instruction(line: &str, block: usize) -> Option<Instruction> {
    let body = line.trim();
    let after = body.split_once(" = ").map(|(_, rest)| rest)?;
    let mut tokens = after.split_whitespace();
    let opcode = tokens.next()?.to_string();
    let flags = ["nuw", "nsw", "exact", "disjoint", "nneg", "fast", "nnan", "ninf", "nsz", "arcp", "contract", "afn", "reassoc"];
    let mut ty = tokens.next()?;
    while flags.contains(&ty) {
        ty = tokens.next()?;
    }
    let operands: Vec<&str> = after.split(',').collect();
    let divisor_constant = operands.get(1).is_some_and(|operand| !operand.trim().starts_with('%'));
    Some(Instruction { block, opcode, ty: ty.to_string(), divisor_constant })
}

fn analyze(ir_path: &Path) -> Tally {
    let text = fs::read_to_string(ir_path).expect("read ir");
    let mut in_function = false;
    let mut block = 0usize;
    let mut labels: BTreeMap<String, usize> = BTreeMap::new();
    let mut instructions: Vec<Instruction> = Vec::new();
    let mut edges: Vec<(usize, String)> = Vec::new();
    for line in text.lines() {
        if line.starts_with("define ") {
            in_function = true;
            block = 0;
            labels.clear();
            continue;
        }
        if !in_function {
            continue;
        }
        if line.starts_with('}') {
            in_function = false;
            continue;
        }
        if let Some(label) = label_of(line) {
            block += 1;
            labels.insert(label, block);
            continue;
        }
        for target in targets_of(line) {
            edges.push((block, target));
        }
        if let Some(instruction) = parse_instruction(line, block) {
            instructions.push(instruction);
        }
    }
    classify(&instructions, &edges, &labels)
}

fn classify(instructions: &[Instruction], edges: &[(usize, String)], labels: &BTreeMap<String, usize>) -> Tally {
    let mut spans: Vec<(usize, usize)> = Vec::new();
    for (source, label) in edges {
        if let Some(&target) = labels.get(label) {
            if target <= *source {
                spans.push((target, *source));
            }
        }
    }
    let mut tally = Tally { loops: spans.len(), ..Tally::default() };
    for instruction in instructions {
        let depth = spans.iter().filter(|(head, latch)| (*head..=*latch).contains(&instruction.block)).count();
        let slot = depth.min(2);
        let wide = instruction.ty == "i64";
        let divides = matches!(instruction.opcode.as_str(), "sdiv" | "udiv" | "srem" | "urem");
        tally.divrem_i64_by_depth[slot] += usize::from(wide && divides);
        tally.mul_i64_by_depth[slot] += usize::from(wide && instruction.opcode == "mul");
        record(&mut tally, instruction, depth > 0);
    }
    tally
}

fn record(tally: &mut Tally, instruction: &Instruction, looped: bool) {
    let wide = instruction.ty == "i64";
    let narrow = instruction.ty == "i32";
    let divides = matches!(instruction.opcode.as_str(), "sdiv" | "udiv" | "srem" | "urem");
    match (instruction.opcode.as_str(), wide, narrow, looped) {
        (_, true, _, false) if divides && instruction.divisor_constant => tally.div_rem_i64_const_once += 1,
        (_, true, _, true) if divides && instruction.divisor_constant => tally.div_rem_i64_const_loop += 1,
        (_, true, _, false) if divides => tally.div_rem_i64_runtime_once += 1,
        (_, true, _, true) if divides => tally.div_rem_i64_runtime_loop += 1,
        (_, _, true, false) if divides => tally.div_rem_i32_once += 1,
        (_, _, true, true) if divides => tally.div_rem_i32_loop += 1,
        ("mul", true, _, false) => tally.mul_i64_once += 1,
        ("mul", true, _, true) => tally.mul_i64_loop += 1,
        ("mul", _, true, true) => tally.mul_i32_loop += 1,
        ("add" | "sub", true, _, true) => tally.addsub_i64_loop += 1,
        ("phi", true, _, true) => tally.phi_i64_loop += 1,
        ("phi", _, true, true) => tally.phi_i32_loop += 1,
        _ => {}
    }
}

fn read_csv(path: &Path) -> Vec<Vec<String>> {
    let text = fs::read_to_string(path).expect("read csv");
    text.lines().map(|line| line.split(',').map(str::to_string).collect()).collect()
}

fn median(values: &mut Vec<f64>) -> f64 {
    values.sort_by(|left, right| left.partial_cmp(right).unwrap());
    values[values.len() / 2]
}

fn instream_by_group(attr_dir: &Path) -> BTreeMap<(String, String), (usize, f64)> {
    let shapes = read_csv(&attr_dir.join("dispatch_shapes.csv"));
    let instream = read_csv(&attr_dir.join("dispatch_instream.csv"));
    let header = &instream[0];
    let column = header.iter().position(|name| name == "instream_ns_median").expect("instream column");
    let mut grouped: BTreeMap<(String, String), Vec<f64>> = BTreeMap::new();
    for (shape, row) in shapes.iter().skip(1).zip(instream.iter().skip(1)) {
        assert_eq!(shape[0], row[0], "index mismatch between shapes and instream");
        let value: f64 = row[column].parse().expect("ns");
        grouped.entry((shape[2].clone(), shape[3].clone())).or_default().push(value);
    }
    grouped.into_iter().map(|(key, mut values)| (key, (values.len(), median(&mut values) / 1000.0))).collect()
}

fn fits(extents: &str) -> (String, bool, bool) {
    let parts: Vec<u128> = extents.split('x').filter_map(|part| part.parse().ok()).collect();
    let product: u128 = parts.iter().product();
    let largest = parts.iter().copied().max().unwrap_or(0);
    (product.to_string(), product <= i32::MAX as u128, largest <= i32::MAX as u128)
}

fn main() {
    let arguments: Vec<String> = env::args().collect();
    let [_, model, describe, caps, attr_dir, ir_dir] = &arguments[..] else {
        panic!("usage: census <model> <describe.out> <caps dir> <attr5 cell dir> <ir out dir>");
    };
    fs::create_dir_all(ir_dir).expect("ir dir");
    let groups = parse_groups(Path::new(describe));
    let instream = instream_by_group(Path::new(attr_dir));
    println!("model,sha16,entry,extents,dispatches_per_step,grid_threads,threadgroup_width,instream_us_per_op_round5,round5_n,iteration_space,iteration_space_fits_i32,max_extent_fits_i32,o0_divrem_i64_literal,o0_mul_i64_literal,loops,div_rem_i64_runtime_once,div_rem_i64_runtime_loop,div_rem_i64_const_once,div_rem_i64_const_loop,div_rem_i32_once,div_rem_i32_loop,mul_i64_once,mul_i64_loop,mul_i32_loop,addsub_i64_loop,phi_i64_loop,phi_i32_loop,divrem_i64_depth0,divrem_i64_depth1,divrem_i64_depth2plus,mul_i64_depth0,mul_i64_depth1,mul_i64_depth2plus");
    let mut total = 0usize;
    for group in &groups {
        let source = Path::new(caps).join(format!("pipeline_{}.metal", group.sha));
        assert!(source.exists(), "N==0 guard: no source for {}", group.sha);
        let tally = analyze(&compile_ir(&source, Path::new(ir_dir), "O2"));
        let literal = analyze(&compile_ir(&source, Path::new(ir_dir), "O0"));
        let source_divrem = literal.div_rem_i64_runtime_once
            + literal.div_rem_i64_runtime_loop
            + literal.div_rem_i64_const_once
            + literal.div_rem_i64_const_loop;
        let source_mul = literal.mul_i64_once + literal.mul_i64_loop;
        let (product, fits_product, fits_extent) = fits(&group.extents);
        let (count, us) = instream.get(&(group.entry.clone(), group.extents.clone())).copied().unwrap_or((0, f64::NAN));
        total += 1;
        println!(
            "{model},{},{},{},{},{},{},{us:.2},{count},{product},{fits_product},{fits_extent},{source_divrem},{source_mul},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            group.sha, group.entry, group.extents, group.count, group.grid_threads, group.width,
            tally.loops, tally.div_rem_i64_runtime_once, tally.div_rem_i64_runtime_loop,
            tally.div_rem_i64_const_once, tally.div_rem_i64_const_loop, tally.div_rem_i32_once,
            tally.div_rem_i32_loop, tally.mul_i64_once, tally.mul_i64_loop, tally.mul_i32_loop,
            tally.addsub_i64_loop, tally.phi_i64_loop, tally.phi_i32_loop,
            tally.divrem_i64_by_depth[0], tally.divrem_i64_by_depth[1], tally.divrem_i64_by_depth[2],
            tally.mul_i64_by_depth[0], tally.mul_i64_by_depth[1], tally.mul_i64_by_depth[2],
        );
    }
    eprintln!("census processed {total} groups for {model}");
}
