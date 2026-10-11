#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Command;

use memmap2::Mmap;
use proxima_model_interop::profiles::{BindingProfile, TensorAlias, binding_profile};
use proxima_model_interop::{
    BoundWeights, KvLayout, bind_checkpoint, bind_safetensors_program_leaves, header_descriptor,
};
use proxima_primitives::pipe::sans_io::Outcome;
use proxima_safetensors::{Manifest, SafetensorsParser};
use proxima_tensor::op::Op;
use proxima_tensor::spec::build_forward;

const EXPORT_ENV: &str = "PROXIMA_GRANITE_PLAIN_EXPORT";
const GGUF_ENV: &str = "PROXIMA_GRANITE_GGUF";
const CHILD_ENV: &str = "PROXIMA_BIND_RSS_CHILD";
const DEFAULT_GGUF: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80";
const TEST_NAME: &str =
    "hf_expert_stack_bind_peak_rss_stays_within_the_measured_bound_of_the_gguf_bind";
const MARKER: &str = "BIND_RSS";
const EMBEDDING_LEAF: &str = "token_embd.weight";
// measured 11.60 to 12.17 over seven runs (hf 323.3 to 323.9 MB, gguf 26.7 to 27.9 MB): token_embd.weight decodes bf16 to 201338880 B of f32 and reads 100669440 B of mapped bf16; with it skipped hf peaks at 21.7 to 22.6 MB; the previous copying bind recorded 3242426368 B, ratio 125.25
const HF_OVER_GGUF_PEAK_BOUND: f64 = 13.0;

fn peak_rss_bytes() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: `usage` is a valid, writable `rusage`; `RUSAGE_SELF` takes no other pointer.
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    assert_eq!(status, 0, "getrusage(RUSAGE_SELF) failed");
    // SAFETY: `getrusage` returned 0, so it filled the struct.
    let maxrss = unsafe { usage.assume_init() }.ru_maxrss as u64;
    if cfg!(target_os = "macos") {
        maxrss
    } else {
        maxrss * 1024
    }
}

fn gguf_path() -> PathBuf {
    PathBuf::from(std::env::var(GGUF_ENV).unwrap_or_else(|_| DEFAULT_GGUF.into()))
}

fn map(path: &Path) -> Mmap {
    let file = File::open(path).unwrap_or_else(|error| panic!("open {}: {error}", path.display()));
    // SAFETY: the checkpoint files are not written while the bind runs.
    unsafe { Mmap::map(&file) }.expect("map checkpoint")
}

fn largest_owned(weights: &BoundWeights<'_>) -> (String, usize) {
    weights
        .owned()
        .iter()
        .map(|(name, values)| (name.clone(), values.len() * std::mem::size_of::<f32>()))
        .max_by_key(|(_, bytes)| *bytes)
        .unwrap_or_default()
}

fn report(mode: &str, weights: &BoundWeights<'_>) {
    let (owned, packed) = (weights.owned().len(), weights.packed().len());
    let (packed_owned, ranges) = (weights.packed_owned().len(), weights.expert_ranges().len());
    let owned_bytes: usize = weights
        .owned()
        .iter()
        .map(|(_, values)| values.len() * std::mem::size_of::<f32>())
        .sum();
    let (largest_name, largest_bytes) = largest_owned(weights);
    println!(
        "{MARKER} mode={mode} rss_peak_bytes={} owned={owned} packed={packed} packed_owned={packed_owned} expert_ranges={ranges} leaves={} owned_bytes={owned_bytes} largest_owned={largest_name}:{largest_bytes}",
        peak_rss_bytes(),
        owned + packed + packed_owned + ranges
    );
}

fn bind_gguf_child() {
    let mapping = map(&gguf_path());
    let parsed = proxima_gguf::parse_complete(&mapping).expect("parse gguf");
    let bound = bind_checkpoint(&parsed, &mapping).expect("gguf bind");
    report("gguf", &bound.weights);
}

fn manifest_of(file: &[u8]) -> (Manifest, u64) {
    let header_length = u64::from_le_bytes(file[..8].try_into().expect("8-byte prefix"));
    let data_start = 8 + header_length;
    let mut parser = SafetensorsParser::new();
    parser.feed(&file[..data_start as usize]);
    let Outcome::Event(manifest) = parser.poll().expect("header parses") else {
        panic!("a complete header must produce a manifest");
    };
    (manifest.clone(), data_start)
}

fn is_stored(manifest: &Manifest, binding: &BindingProfile, name: &str) -> bool {
    manifest.tensor(name).is_some()
        || binding
            .aliases_for(name)
            .any(|(alias, prefix)| match alias {
                TensorAlias::Rename { from, .. } | TensorAlias::Part { from, .. } => {
                    manifest.tensor(&format!("{prefix}{from}")).is_some()
                }
                TensorAlias::Join { .. } => false,
            })
}

fn runtime_inputs(
    program: &[Op],
    manifest: &Manifest,
    binding: &BindingProfile,
    skipped: &[&str],
) -> Vec<String> {
    program
        .iter()
        .filter_map(|operation| match operation {
            Op::Input {
                name: Some(name), ..
            } => Some(name.clone()),
            _ => None,
        })
        .filter(|name| skipped.contains(&name.as_str()) || !is_stored(manifest, binding, name))
        .collect()
}

fn bind_hf_child(export: &Path, mode: &str, skipped: &[&str]) {
    let gguf = map(&gguf_path());
    let parsed = proxima_gguf::parse_complete(&gguf).expect("parse gguf header");
    let descriptor = header_descriptor(&parsed, KvLayout::Full).expect("descriptor");
    let program = build_forward(&descriptor).expect("lowers").program;
    let file = map(&export.join("model.safetensors"));
    let (manifest, data_start) = manifest_of(&file);
    let binding = binding_profile("granitemoe").expect("granitemoe parses");
    let weights = bind_safetensors_program_leaves(
        &manifest,
        &file,
        data_start,
        &program,
        &binding,
        &runtime_inputs(&program, &manifest, &binding, skipped),
    )
    .expect("hf bind");
    report(mode, &weights);
}

fn field(line: &str, key: &str) -> u64 {
    line.split_whitespace()
        .find_map(|item| item.strip_prefix(key)?.strip_prefix('='))
        .unwrap_or_else(|| panic!("no {key} field in {line}"))
        .parse()
        .unwrap_or_else(|error| panic!("{key} is not a number in {line}: {error}"))
}

fn spawn_child(mode: &str, export: &Path) -> String {
    let output = Command::new(std::env::current_exe().expect("test binary path"))
        .args(["--exact", TEST_NAME, "--nocapture", "--test-threads=1"])
        .env(CHILD_ENV, mode)
        .env(EXPORT_ENV, export)
        .output()
        .expect("spawn child");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(
        output.status.success(),
        "{mode} child failed: {stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    stdout
        .lines()
        .find_map(|line| line.find(MARKER).map(|start| &line[start..]))
        .unwrap_or_else(|| panic!("{mode} child printed no {MARKER} line: {stdout}"))
        .to_string()
}

fn stack_bytes() -> (u64, u64) {
    let gguf = map(&gguf_path());
    let parsed = proxima_gguf::parse_complete(&gguf).expect("parse gguf header");
    parsed
        .tensors
        .iter()
        .filter(|tensor| tensor.name.contains("_exps."))
        .fold((0, 0), |(stored, bf16), tensor| {
            let range = parsed
                .tensor_data_range(tensor, gguf.len() as u64)
                .expect("range");
            (
                stored + (range.end - range.start),
                bf16 + tensor.element_count() * 2,
            )
        })
}

#[test]
fn hf_expert_stack_bind_peak_rss_stays_within_the_measured_bound_of_the_gguf_bind() {
    let Ok(export) = std::env::var(EXPORT_ENV) else {
        eprintln!(
            "SKIPPED {TEST_NAME}: {EXPORT_ENV} names no granite plain export directory (model.safetensors, written by the mlx_plain_export example); the peak-RSS contract was not checked"
        );
        return;
    };
    let export = PathBuf::from(export);
    match std::env::var(CHILD_ENV).as_deref() {
        Ok("gguf") => return bind_gguf_child(),
        Ok("hf") => return bind_hf_child(&export, "hf", &[]),
        Ok("hf_without_embedding") => {
            return bind_hf_child(&export, "hf_without_embedding", &[EMBEDDING_LEAF]);
        }
        _ => {}
    }
    let gguf_line = spawn_child("gguf", &export);
    let hf_line = spawn_child("hf", &export);
    let without_embedding_line = spawn_child("hf_without_embedding", &export);
    let (gguf_peak, hf_peak) = (
        field(&gguf_line, "rss_peak_bytes"),
        field(&hf_line, "rss_peak_bytes"),
    );
    let (stored, bf16) = stack_bytes();
    let ratio = hf_peak as f64 / gguf_peak as f64;
    eprintln!(
        "{gguf_line}\n{hf_line}\n{without_embedding_line}\nstack_bytes q8_0={stored} bf16={bf16} stack_ratio={:.4} hf_over_gguf_peak={ratio:.4}",
        bf16 as f64 / stored as f64
    );
    assert!(
        field(&gguf_line, "leaves") > 0,
        "the gguf bind bound no leaves"
    );
    assert_eq!(
        field(&hf_line, "leaves"),
        field(&gguf_line, "leaves"),
        "the hf bind must bind every leaf the gguf bind binds"
    );
    assert_eq!(
        field(&hf_line, "packed_owned"),
        0,
        "no fused expert part may be copied into owned bytes"
    );
    assert!(
        ratio <= HF_OVER_GGUF_PEAK_BOUND,
        "hf bind peak {hf_peak} is {ratio:.3}x the gguf bind peak {gguf_peak}"
    );
}
