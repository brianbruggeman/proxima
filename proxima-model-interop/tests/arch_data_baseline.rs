//! Consistency baseline for architecture-as-data (proxima-tensor/specs/architecture-as-data,
//! slice 0): for each real checkpoint, the lowered op graph and the bound weight set the
//! incumbent produces through the production path
//! (`ArchitectureRegistry::with_builtin().resolve(..)` then `bind_with_kv_layout(SlidingRing)`
//! and the verify program where one exists). Expected values live in
//! `tests/fixtures/llama-parity/<name>.digest` and `<name>.bound`; a refactor slice that changes
//! the program or the bound bytes fails here. This is a CONSISTENCY check against the incumbent,
//! never a correctness oracle -- the llama.cpp artifacts next to it are the oracle.
//!
//! A missing checkpoint fails with its path and env override; it never skips.
//! `PROXIMA_ARCH_DATA_CAPTURE=1` rewrites the expected files from the current build.
//! Model-loading tests must run with `--test-threads 1` (nextest `-j 1`).

#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fmt::Write as _;
use std::fs::File;
use std::path::{Path, PathBuf};

use proxima_gguf::parse_complete;
use proxima_model_interop::{
    Architecture, ArchitectureRegistry, BoundProgram, BoundWeights, Codec, KvLayout,
};
use proxima_tensor::cpu::QuantizedBlock;
use proxima_tensor::op::Op;
use sha2::{Digest, Sha256};

const CAPTURE_ENV: &str = "PROXIMA_ARCH_DATA_CAPTURE";
const CHECKPOINTS_TOML: &str = include_str!("fixtures/llama-parity/checkpoints.toml");

struct Checkpoint {
    name: &'static str,
    env: &'static str,
    path: &'static str,
    architecture: &'static str,
}

const GEMMA4_26B: Checkpoint = Checkpoint {
    name: "gemma4_26b",
    env: "PROXIMA_ARCH_GEMMA4_26B_GGUF",
    path: "/Users/brianbruggeman/.ollama/models/blobs/sha256-ea549b7688d4c95019754880c21e3f29c58c985a7a1c3b37b9eebd0a95224129",
    architecture: "gemma4",
};
const GEMMA4_E2B: Checkpoint = Checkpoint {
    name: "gemma4_e2b",
    env: "PROXIMA_ARCH_GEMMA4_E2B_GGUF",
    path: "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd",
    architecture: "gemma4",
};
const OPENCHAT: Checkpoint = Checkpoint {
    name: "openchat",
    env: "PROXIMA_ARCH_OPENCHAT_GGUF",
    path: "/Users/brianbruggeman/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf",
    architecture: "llama",
};
const QWEN2: Checkpoint = Checkpoint {
    name: "qwen2",
    env: "PROXIMA_ARCH_QWEN2_GGUF",
    path: "/Users/brianbruggeman/.ollama/models/blobs/sha256-c5396e06af294bd101b30dce59131a76d2b773e76950acc870eda801d3ab0515",
    architecture: "qwen2",
};
const QWEN3: Checkpoint = Checkpoint {
    name: "qwen3",
    env: "PROXIMA_ARCH_QWEN3_GGUF",
    path: "/Users/brianbruggeman/.ollama/models/blobs/sha256-a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f",
    architecture: "qwen3",
};
const QWEN35: Checkpoint = Checkpoint {
    name: "qwen35",
    env: "PROXIMA_ARCH_QWEN35_GGUF",
    path: "/Users/brianbruggeman/.ollama/models/blobs/sha256-afb707b6b8fac6e475acc42bc8380fc0b8d2e0e4190be5a969fbf62fcc897db5",
    architecture: "qwen35",
};
const QWEN35MOE: Checkpoint = Checkpoint {
    name: "qwen35moe",
    env: "PROXIMA_ARCH_QWEN35MOE_GGUF",
    path: "/Users/brianbruggeman/.ollama/models/blobs/sha256-f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d",
    architecture: "qwen35moe",
};

const ALL: [&Checkpoint; 7] = [
    &GEMMA4_26B,
    &GEMMA4_E2B,
    &OPENCHAT,
    &QWEN2,
    &QWEN3,
    &QWEN35,
    &QWEN35MOE,
];

impl Checkpoint {
    fn resolved_path(&self) -> String {
        std::env::var(self.env).unwrap_or_else(|_| self.path.to_string())
    }

    fn fixture(&self, extension: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/llama-parity")
            .join(format!("{}.{extension}", self.name))
    }

    fn open(&self) -> memmap2::Mmap {
        let path = self.resolved_path();
        assert!(
            Path::new(&path).exists(),
            "checkpoint {} is missing at {path}: stage a general.architecture = {} gguf there or set {}",
            self.name,
            self.architecture,
            self.env
        );
        let file = File::open(&path).unwrap_or_else(|error| panic!("open {path}: {error}"));
        unsafe { memmap2::Mmap::map(&file) }.expect("mmap the real checkpoint read-only")
    }
}

fn hex(bytes: impl AsRef<[u8]>) -> String {
    let mut text = String::new();
    for byte in bytes.as_ref() {
        write!(text, "{byte:02x}").expect("writing to a String cannot fail");
    }
    text
}

fn sha256_hex(chunks: &[&[u8]]) -> String {
    let mut hasher = Sha256::new();
    for chunk in chunks {
        hasher.update(chunk);
    }
    hex(hasher.finalize())
}

fn ops_digest(program: &[Op]) -> String {
    let mut hasher = Sha256::new();
    for (index, op) in program.iter().enumerate() {
        hasher.update(format!("{index}:{op:?}\n").as_bytes());
    }
    hex(hasher.finalize())
}

fn debug_digest(value: &impl std::fmt::Debug) -> String {
    sha256_hex(&[format!("{value:?}").as_bytes()])
}

fn describe_program(label: &str, bound: &BoundProgram<'_>) -> String {
    let mut text = String::new();
    writeln!(text, "{label}.ops={}", bound.program.len()).unwrap();
    writeln!(text, "{label}.ops_sha256={}", ops_digest(&bound.program)).unwrap();
    writeln!(text, "{label}.logits_root={:?}", bound.logits_root).unwrap();
    writeln!(text, "{label}.hidden_root={:?}", bound.hidden_root).unwrap();
    writeln!(
        text,
        "{label}.residual_roots={} sha256={}",
        bound.residual_roots.len(),
        debug_digest(&bound.residual_roots)
    )
    .unwrap();
    writeln!(
        text,
        "{label}.layer_roots={} sha256={}",
        bound.layer_roots.len(),
        debug_digest(&bound.layer_roots)
    )
    .unwrap();
    writeln!(
        text,
        "{label}.router_roots={} sha256={}",
        bound.router_roots.len(),
        debug_digest(&bound.router_roots)
    )
    .unwrap();
    writeln!(
        text,
        "{label}.single_position_step={}",
        bound.single_position_step
    )
    .unwrap();
    text
}

fn resolve(
    checkpoint: &Checkpoint,
    parsed: &proxima_gguf::pipe::ParsedGguf,
) -> &'static dyn Architecture {
    let registry = ArchitectureRegistry::with_builtin();
    registry
        .resolve(parsed)
        .unwrap_or_else(|error| panic!("{}: registry resolve failed: {error:?}", checkpoint.name))
}

fn assert_architecture_key(checkpoint: &Checkpoint, parsed: &proxima_gguf::pipe::ParsedGguf) {
    let declared = proxima_model_interop::metadata_str(parsed, "general.architecture")
        .expect("checkpoint declares general.architecture");
    assert_eq!(
        declared, checkpoint.architecture,
        "{} declares a different general.architecture than checkpoints.toml",
        checkpoint.name
    );
}

fn digest_record(checkpoint: &Checkpoint) -> String {
    let mapping = checkpoint.open();
    let file_bytes: &[u8] = &mapping;
    let parsed = parse_complete(file_bytes).expect("parses the real checkpoint's GGUF header");
    assert_architecture_key(checkpoint, &parsed);
    let route = resolve(checkpoint, &parsed);

    let bound = route
        .bind_with_kv_layout(&parsed, file_bytes, KvLayout::SlidingRing)
        .unwrap_or_else(|error| {
            panic!("{}: bind_with_kv_layout failed: {error:?}", checkpoint.name)
        });
    let verify = route
        .speculative_verify_program_with_kv_layout(&parsed, file_bytes, KvLayout::SlidingRing)
        .unwrap_or_else(|error| panic!("{}: verify bind failed: {error:?}", checkpoint.name));

    let mut record = String::new();
    writeln!(record, "checkpoint={}", checkpoint.name).unwrap();
    writeln!(record, "architecture={}", checkpoint.architecture).unwrap();
    writeln!(record, "registry_entry={}", route.name()).unwrap();
    writeln!(record, "kv_layout=SlidingRing").unwrap();
    record.push_str(&describe_program("bind", &bound));
    match verify {
        Some(program) => record.push_str(&describe_program("verify", &program)),
        None => record.push_str("verify=absent\n"),
    }
    record
}

fn digest_hex_of_block(block: &QuantizedBlock<'_>) -> (String, usize, String) {
    match block {
        QuantizedBlock::Float32(values) => {
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            ("f32".to_string(), bytes.len(), sha256_hex(&[&bytes]))
        }
        QuantizedBlock::Int32(values) => {
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            ("i32".to_string(), bytes.len(), sha256_hex(&[&bytes]))
        }
        QuantizedBlock::Packed { codec, bytes } => {
            (format!("{codec:?}"), bytes.len(), sha256_hex(&[bytes]))
        }
    }
}

fn bound_lines(weights: &BoundWeights<'_>) -> Vec<String> {
    let mut lines = Vec::new();
    for (name, values) in weights.owned() {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        lines.push(format!(
            "owned\t{name}\tf32\t{}\t{}",
            bytes.len(),
            sha256_hex(&[&bytes])
        ));
    }
    for (name, block) in weights.packed() {
        let (kind, length, digest) = digest_hex_of_block(block);
        lines.push(format!("packed\t{name}\t{kind}\t{length}\t{digest}"));
    }
    for (name, bytes, codec) in weights.packed_owned() {
        let codec: &Codec = codec;
        lines.push(format!(
            "packed_owned\t{name}\t{codec:?}\t{}\t{}",
            bytes.len(),
            sha256_hex(&[bytes])
        ));
    }
    lines.sort();
    lines
}

fn bound_record(checkpoint: &Checkpoint) -> String {
    let mapping = checkpoint.open();
    let file_bytes: &[u8] = &mapping;
    let parsed = parse_complete(file_bytes).expect("parses the real checkpoint's GGUF header");
    assert_architecture_key(checkpoint, &parsed);
    let route = resolve(checkpoint, &parsed);
    let bound = route
        .bind_with_kv_layout(&parsed, file_bytes, KvLayout::SlidingRing)
        .unwrap_or_else(|error| {
            panic!("{}: bind_with_kv_layout failed: {error:?}", checkpoint.name)
        });

    let lines = bound_lines(&bound.weights);
    assert!(
        !lines.is_empty(),
        "{}: the incumbent bound zero weights; the capture would assert nothing",
        checkpoint.name
    );
    let mut record = format!(
        "checkpoint={}\nbound_weights={}\nkind\tname\tcodec\tbytes\tsha256\n",
        checkpoint.name,
        lines.len()
    );
    record.push_str(&lines.join("\n"));
    record.push('\n');
    record
}

fn assert_matches_fixture(checkpoint: &Checkpoint, extension: &str, actual: &str) {
    let fixture = checkpoint.fixture(extension);
    if std::env::var(CAPTURE_ENV).is_ok_and(|value| value == "1") {
        std::fs::write(&fixture, actual)
            .unwrap_or_else(|error| panic!("write {}: {error}", fixture.display()));
        return;
    }
    let expected = std::fs::read_to_string(&fixture).unwrap_or_else(|error| {
        panic!(
            "no baseline at {}: {error}; capture it with {CAPTURE_ENV}=1",
            fixture.display()
        )
    });
    assert_eq!(
        expected,
        actual,
        "{} {extension} differs from the incumbent baseline {}",
        checkpoint.name,
        fixture.display()
    );
}

fn assert_digest(checkpoint: &Checkpoint) {
    let record = digest_record(checkpoint);
    assert_matches_fixture(checkpoint, "digest", &record);
}

fn assert_bound(checkpoint: &Checkpoint) {
    let record = bound_record(checkpoint);
    assert_matches_fixture(checkpoint, "bound", &record);
}

#[test]
fn arch_data_digest_gemma4_26b() {
    let record = digest_record(&GEMMA4_26B);
    assert!(
        record.contains("\nbind.ops=13314\n")
            && record.contains("\nbind.logits_root=NodeId(13313)\n"),
        "gemma4 26B must lower to 13314 ops ending at NodeId(13313), got:\n{record}"
    );
    assert_matches_fixture(&GEMMA4_26B, "digest", &record);
}

#[test]
fn arch_data_digest_gemma4_e2b() {
    assert_digest(&GEMMA4_E2B);
}

#[test]
fn arch_data_digest_openchat() {
    assert_digest(&OPENCHAT);
}

#[test]
fn arch_data_digest_qwen2() {
    assert_digest(&QWEN2);
}

#[test]
fn arch_data_digest_qwen3() {
    assert_digest(&QWEN3);
}

#[test]
fn arch_data_digest_qwen35() {
    assert_digest(&QWEN35);
}

#[test]
fn arch_data_digest_qwen35moe() {
    assert_digest(&QWEN35MOE);
}

#[test]
fn generic_binder_gemma4_26b() {
    assert_bound(&GEMMA4_26B);
}

#[test]
fn generic_binder_gemma4_e2b() {
    assert_bound(&GEMMA4_E2B);
}

#[test]
fn generic_binder_openchat() {
    assert_bound(&OPENCHAT);
}

#[test]
fn generic_binder_qwen2() {
    assert_bound(&QWEN2);
}

#[test]
fn generic_binder_qwen3() {
    assert_bound(&QWEN3);
}

#[test]
fn generic_binder_qwen35() {
    assert_bound(&QWEN35);
}

#[test]
fn generic_binder_qwen35moe() {
    assert_bound(&QWEN35MOE);
}

#[test]
fn checkpoints_toml_lists_every_baseline_checkpoint() {
    for checkpoint in ALL {
        for needle in [
            format!("name = \"{}\"", checkpoint.name),
            format!("path = \"{}\"", checkpoint.path),
            format!("env = \"{}\"", checkpoint.env),
            format!("architecture = \"{}\"", checkpoint.architecture),
        ] {
            assert!(
                CHECKPOINTS_TOML.contains(&needle),
                "checkpoints.toml lacks `{needle}` for {}",
                checkpoint.name
            );
        }
    }
}
