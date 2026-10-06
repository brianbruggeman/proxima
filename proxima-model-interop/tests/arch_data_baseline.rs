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

use core::ops::ControlFlow;
use std::fmt::Write as _;
use std::fs::File;
use std::path::{Path, PathBuf};
#[cfg(all(feature = "metal", target_os = "macos"))]
use std::sync::OnceLock;
#[cfg(all(feature = "metal", target_os = "macos"))]
use std::sync::atomic::{AtomicUsize, Ordering};

use proxima_gguf::parse_complete;
use proxima_model_interop::{
    Architecture, ArchitectureRegistry, BoundProgram, BoundWeights, Codec, KvLayout, LoadedModel,
    PromptCacheConfig, ServingConfig, SpeculativeDecodeStats, architecture_from_metadata, dense_descriptor_from_gguf, gemma4,
    metadata_f32_optional,
    metadata_str, metadata_u32, profiles::family_profile, qwen35_architecture_from_metadata,
    qwen35_descriptor_from_architecture, qwen35moe,
};
use proxima_tensor::cpu::QuantizedBlock;
use proxima_tensor::TensorError;
use proxima_tensor::op::{Extent, Op};
use proxima_tensor::spec::{
    Activation, AttentionScoreScale, CacheMask, CacheStrategy, EmbeddingScale, ExpertGatingFunc, FfnCombination,
    KeySourceKind, LayerAttentionConfig, LayerFfnConfig, LayerKind, LayerSchedule, ModelDescriptor,
    ParallelDenseMoeConfig, CachedLayerRoots, ForwardProgram, Qwen35LayerRoots, RopePairing, RopeTableSel,
    ValueSourceKind, build_forward,
    SLIDING_KV_SYMBOL, gemma4_descriptor_from_gguf, lfm2_two_range_cached_forward_program_with_experts,
    mistral_cached_forward_program_with_experts_and_layer_taps, mistral_descriptor_from_shape,
};
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
const GRANITE_MOE: Checkpoint = Checkpoint {
    name: "granite_moe",
    env: "PROXIMA_ARCH_GRANITE_MOE_GGUF",
    path: "/Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80",
    architecture: "granitemoe",
};

const ALL: [&Checkpoint; 8] = [
    &GEMMA4_26B,
    &GEMMA4_E2B,
    &OPENCHAT,
    &QWEN2,
    &QWEN3,
    &QWEN35,
    &QWEN35MOE,
    &GRANITE_MOE,
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

fn is_step_input(name: &str) -> bool {
    matches!(
        name,
        "ids" | "eps" | "rope_cos" | "rope_sin" | "rope_cos_swa" | "rope_sin_swa" | "cached_len" | "cached_len_swa" | "lm_head_row"
    ) || name.starts_with("kv_cache.")
        || name.starts_with("ssm_cache.")
}

fn assert_unbound_leaves_are_step_inputs(checkpoint: &Checkpoint, bound: &BoundProgram<'_>) {
    let bound_names: std::collections::BTreeSet<&str> = bound
        .weights
        .owned()
        .iter()
        .map(|(name, _)| name.as_str())
        .chain(bound.weights.packed().iter().map(|(name, _)| name.as_str()))
        .chain(bound.weights.packed_owned().iter().map(|(name, _, _)| name.as_str()))
        .collect();
    let weight_leaves_left_unbound: Vec<&str> = bound
        .program
        .iter()
        .filter_map(|op| match op {
            Op::Input { name: Some(name), .. } => Some(name.as_str()),
            _ => None,
        })
        .filter(|name| !bound_names.contains(name) && !is_step_input(name))
        .collect();
    assert!(
        weight_leaves_left_unbound.is_empty(),
        "{}: program leaves bound to nothing and fed by no step: {weight_leaves_left_unbound:?}",
        checkpoint.name
    );
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
        "{}: the binder bound zero weights; the comparison would assert nothing",
        checkpoint.name
    );
    assert_unbound_leaves_are_step_inputs(checkpoint, &bound);
    let mut record = format!(
        "checkpoint={}\nbound_weights={}\nkind\tname\tcodec\tbytes\tsha256\n",
        checkpoint.name,
        lines.len()
    );
    record.push_str(&lines.join("\n"));
    record.push('\n');
    record
}

/// The AC3 comparison drops the storage class of an `f32` tensor (a borrowed view of the mapping
/// or an owned buffer): every consumer reads the two the same, and the bytes and sha256 beside
/// it still have to match. Every other column is compared as captured, and the weight lines are
/// re-sorted because the capture sorted them on the storage class this drops.
fn without_f32_storage_class(record: &str) -> String {
    const HEADER_LINES: usize = 3;
    let lines: Vec<String> = record
        .lines()
        .map(|line| match line.split('\t').collect::<Vec<_>>().as_slice() {
            ["owned" | "packed", name, "f32", rest @ ..] => format!("f32\t{name}\tf32\t{}", rest.join("\t")),
            _ => line.to_string(),
        })
        .collect();
    let (header, weights) = lines.split_at(HEADER_LINES.min(lines.len()));
    let mut weights = weights.to_vec();
    weights.sort();
    let mut normalized: Vec<String> = header.to_vec();
    normalized.extend(weights);
    normalized.push(String::new());
    normalized.join("\n")
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
    let (expected, actual) = if extension == "bound" {
        (without_f32_storage_class(&expected), without_f32_storage_class(actual))
    } else {
        (expected, actual.to_string())
    };
    if expected != actual {
        let expected_lines: std::collections::BTreeSet<&str> = expected.lines().collect();
        let actual_lines: std::collections::BTreeSet<&str> = actual.lines().collect();
        let only_expected: Vec<&&str> = expected_lines.difference(&actual_lines).take(12).collect();
        let only_actual: Vec<&&str> = actual_lines.difference(&expected_lines).take(12).collect();
        panic!(
            "{} {extension} differs from the incumbent baseline {}\nexpected {} lines, got {}\nonly in the baseline: {only_expected:#?}\nonly in the run: {only_actual:#?}",
            checkpoint.name,
            fixture.display(),
            expected.lines().count(),
            actual.lines().count()
        );
    }
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
fn arch_data_digest_granite_moe() {
    let record = digest_record(&GRANITE_MOE);
    for expected in [
        "\nregistry_entry=dense\n",
        "\nbind.residual_roots=24 sha256=",
        "\nbind.layer_roots=24 sha256=",
        "\nbind.router_roots=0 sha256=",
        "\nbind.single_position_step=false\n",
    ] {
        assert!(record.contains(expected), "granite moe digest must contain {expected:?}, got:\n{record}");
    }
    assert!(record.ends_with("verify=absent\n"), "granite moe digest must end with verify=absent, got:\n{record}");
    assert_matches_fixture(&GRANITE_MOE, "digest", &record);
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
fn generic_binder_granite_moe() {
    let record = bound_record(&GRANITE_MOE);
    assert!(record.contains("\nbound_weights=243\n"), "granite moe must bind 243 weights (24 layers x 10 + token_embd + output_norm + output), got {:?}", record.lines().nth(1));
    assert_matches_fixture(&GRANITE_MOE, "bound", &record);
}

fn config_descriptors(
    checkpoint: &Checkpoint,
    parsed: &proxima_gguf::pipe::ParsedGguf,
) -> Vec<(&'static str, ModelDescriptor)> {
    if checkpoint.architecture == "gemma4" {
        let decode = gemma4::descriptor_from_gguf(parsed, true)
            .unwrap_or_else(|error| panic!("{}: descriptor_from_gguf failed: {error:?}", checkpoint.name));
        let verify = ModelDescriptor {
            last_row_only: false,
            ..decode.clone()
        };
        return vec![("bind", decode), ("verify", verify)];
    }
    if checkpoint.architecture == "qwen35" {
        let architecture = qwen35_architecture_from_metadata(parsed)
            .unwrap_or_else(|error| panic!("{}: qwen35_architecture_from_metadata failed: {error:?}", checkpoint.name));
        let descriptor = qwen35_descriptor_from_architecture(&architecture)
            .unwrap_or_else(|error| panic!("{}: the qwen35 descriptor failed: {error:?}", checkpoint.name));
        return vec![("bind", descriptor)];
    }
    if checkpoint.architecture == "qwen35moe" {
        let architecture = qwen35moe::from_metadata(parsed)
            .unwrap_or_else(|error| panic!("{}: qwen35moe hparams failed: {error:?}", checkpoint.name));
        let descriptor = qwen35moe::descriptor_from_architecture(&architecture, None)
            .unwrap_or_else(|error| panic!("{}: the qwen35moe descriptor failed: {error:?}", checkpoint.name));
        return vec![("bind", descriptor)];
    }
    let architecture = architecture_from_metadata(parsed)
        .unwrap_or_else(|error| panic!("{}: architecture_from_metadata failed: {error:?}", checkpoint.name));
    let descriptor = dense_descriptor_from_gguf(parsed, &architecture)
        .unwrap_or_else(|error| panic!("{}: dense_descriptor_from_gguf failed: {error:?}", checkpoint.name));
    vec![("bind", descriptor)]
}

fn model_config_roundtrip(checkpoint: &Checkpoint) {
    let mapping = checkpoint.open();
    let parsed = parse_complete(&mapping).expect("parses the real checkpoint's GGUF header");
    assert_architecture_key(checkpoint, &parsed);
    let incumbent = std::fs::read_to_string(checkpoint.fixture("digest"))
        .unwrap_or_else(|error| panic!("{}: no incumbent digest: {error}", checkpoint.name));

    let descriptors = config_descriptors(checkpoint, &parsed);
    let incumbent_labels = ["bind", "verify"]
        .into_iter()
        .filter(|label| incumbent.contains(&format!("\n{label}.ops=")))
        .collect::<Vec<_>>();
    assert_eq!(
        descriptors.iter().map(|(label, _)| *label).collect::<Vec<_>>(),
        incumbent_labels,
        "{}: the config set and the incumbent's programs disagree",
        checkpoint.name
    );
    for (label, descriptor) in descriptors {
        let text = toml::to_string(&descriptor)
            .unwrap_or_else(|error| panic!("{} {label}: descriptor does not serialize: {error}", checkpoint.name));
        let restored: ModelDescriptor = toml::from_str(&text)
            .unwrap_or_else(|error| panic!("{} {label}: descriptor toml does not parse: {error}\n{text}", checkpoint.name));
        assert_eq!(restored, descriptor, "{} {label}: toml round trip changed the config", checkpoint.name);
        conflaguration::Validate::validate(&restored)
            .unwrap_or_else(|error| panic!("{} {label}: the real checkpoint's config fails validation: {error}", checkpoint.name));

        let ForwardProgram { program, logits: logits_root, .. } = build_forward(&restored)
            .unwrap_or_else(|error| panic!("{} {label}: restored config does not lower: {error:?}", checkpoint.name));
        assert!(!program.is_empty(), "{} {label}: lowered zero ops", checkpoint.name);
        for expected in [
            format!("{label}.ops={}", program.len()),
            format!("{label}.ops_sha256={}", ops_digest(&program)),
            format!("{label}.logits_root={logits_root:?}"),
        ] {
            assert!(
                incumbent.lines().any(|line| line == expected),
                "{} {label}: the restored config lowers to `{expected}`, which the incumbent digest {} lacks",
                checkpoint.name,
                checkpoint.fixture("digest").display()
            );
        }
    }
}

#[test]
fn model_config_roundtrip_gemma4_26b() {
    model_config_roundtrip(&GEMMA4_26B);
}

#[test]
fn model_config_roundtrip_gemma4_e2b() {
    model_config_roundtrip(&GEMMA4_E2B);
}

#[test]
fn model_config_roundtrip_openchat() {
    model_config_roundtrip(&OPENCHAT);
}

#[test]
fn model_config_roundtrip_qwen2() {
    model_config_roundtrip(&QWEN2);
}

#[test]
fn model_config_roundtrip_qwen3() {
    model_config_roundtrip(&QWEN3);
}

#[test]
fn model_config_roundtrip_granite_moe() {
    model_config_roundtrip(&GRANITE_MOE);
}

#[test]
fn model_config_roundtrip_qwen35() {
    model_config_roundtrip(&QWEN35);
}

#[test]
fn model_config_roundtrip_qwen35moe() {
    model_config_roundtrip(&QWEN35MOE);
}

#[test]
fn model_config_text_edit_changes_the_lowered_program() {
    let mapping = GEMMA4_E2B.open();
    let parsed = parse_complete(&mapping).expect("parses the real checkpoint's GGUF header");
    let descriptor = gemma4::descriptor_from_gguf(&parsed, true).expect("the e2b header carries every key the descriptor reads");
    let text = toml::to_string(&descriptor).expect("a descriptor serializes to toml");
    assert!(text.contains("activation = \"GeluTanh\""), "e2b runs a gelu ffn, got:\n{text}");

    let edited: ModelDescriptor = toml::from_str(&text.replace("GeluTanh", "Silu")).expect("the edited toml parses");
    let ForwardProgram { program, .. } = build_forward(&edited).expect("the edited config lowers");

    let incumbent = std::fs::read_to_string(GEMMA4_E2B.fixture("digest")).expect("incumbent digest exists");
    assert!(
        !incumbent.lines().any(|line| line == format!("bind.ops_sha256={}", ops_digest(&program))),
        "a silu ffn must not lower to the gelu program"
    );
    assert_ne!(
        edited.layers.iter().map(|layer| layer.ffn.activation).collect::<Vec<_>>(),
        descriptor.layers.iter().map(|layer| layer.ffn.activation).collect::<Vec<_>>(),
    );
}

fn leaf_names(program: &[Op]) -> Vec<String> {
    program
        .iter()
        .filter_map(|op| match op {
            Op::Input { name: Some(name), .. } => Some(name.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn model_config_edit_moves_a_qwen35_attention_layer_to_the_recurrent_mixer() {
    let mapping = QWEN35.open();
    let parsed = parse_complete(&mapping).expect("parses the real checkpoint's GGUF header");
    let architecture = qwen35_architecture_from_metadata(&parsed).expect("the qwen35 header reads");
    let descriptor = qwen35_descriptor_from_architecture(&architecture).expect("the qwen35 descriptor builds");
    assert_eq!(descriptor.layers[3].kind, LayerKind::Attention, "the 0.8b checkpoint attends at layers 3, 7, 11, ...");
    let text = toml::to_string(&descriptor).expect("a descriptor serializes to toml");

    let mut edited: ModelDescriptor = toml::from_str(&text).expect("the descriptor toml parses");
    edited.layers[3].kind = LayerKind::Gdn;
    let edited: ModelDescriptor = toml::from_str(&toml::to_string(&edited).expect("serializes")).expect("the edited toml parses");
    let base = build_forward(&descriptor).expect("the header config lowers");
    let variant = build_forward(&edited).expect("the edited config lowers");

    let (base_leaves, variant_leaves) = (leaf_names(&base.program), leaf_names(&variant.program));
    assert!(base_leaves.contains(&"blk.3.attn_q.weight".to_string()));
    assert!(!variant_leaves.contains(&"blk.3.attn_q.weight".to_string()), "layer 3 no longer attends");
    assert!(variant_leaves.contains(&"ssm_cache.3.state".to_string()), "layer 3 now carries a recurrent state");
    assert_ne!(base.program.len(), variant.program.len());
}

#[test]
fn model_config_rejects_an_unknown_field() {
    let mapping = GEMMA4_E2B.open();
    let parsed = parse_complete(&mapping).expect("parses the real checkpoint's GGUF header");
    let descriptor = gemma4::descriptor_from_gguf(&parsed, true).expect("the e2b header carries every key the descriptor reads");
    let text = toml::to_string(&descriptor).expect("a descriptor serializes to toml");

    let outcome = toml::from_str::<ModelDescriptor>(&format!("block_cuont = 30\n{text}"));

    let error = outcome.expect_err("an unknown key must be an error, never a silent default");
    assert!(error.to_string().contains("unknown field"), "got: {error}");
}

fn owned_cache_roots(layer_roots: &[Qwen35LayerRoots]) -> Vec<CachedLayerRoots> {
    layer_roots
        .iter()
        .filter_map(|roots| match roots {
            Qwen35LayerRoots::Attention(cached) => Some(*cached),
            _ => None,
        })
        .collect()
}

fn assert_programs_identical(direct: &[Op], descriptor: &[Op]) {
    assert_eq!(direct.len(), descriptor.len(), "op count mismatch");
    let first_divergence = direct
        .iter()
        .zip(descriptor)
        .position(|(direct_op, descriptor_op)| direct_op != descriptor_op);
    assert!(
        first_divergence.is_none(),
        "op graphs diverge at index {first_divergence:?}: direct={:?} descriptor={:?}",
        first_divergence.map(|index| &direct[index]),
        first_divergence.map(|index| &descriptor[index]),
    );
}

/// The independent arm: layers written out from the real 26B-A4B header's own
/// values (every 6th layer full attention, 8/2 kv heads, window 1024), not
/// read from the descriptor under test.
fn gemma4_26b_hand_written_layers() -> Vec<LayerSchedule> {
    let ffn = LayerFfnConfig {
        post_attention_norm: true,
        combination: FfnCombination::ParallelDenseMoe(ParallelDenseMoeConfig {
            dense_post_norm: true,
            routed_post_norm: true,
            combined_post_norm: true,
            routed_pre_norm: true,
            router_scale: true,
            expert_output_scale: true,
        }),
        output_scale: true,
        routed_gating: ExpertGatingFunc::Softmax,
        routed_expert_bias: false,
        dense_feed_forward: None,
        exclusive_dense_post_norm: false,
        activation: Activation::GeluTanh,
        ple: false,
    };
    (0..30u32)
        .map(|layer| {
            let is_full = (layer + 1).is_multiple_of(6);
            let (head_dim, kv_heads, mask_window, value_source_kind, cos_name, sin_name) = if is_full {
                (512, 2, None, ValueSourceKind::SharedWithKey, "rope_cos", "rope_sin")
            } else {
                (256, 8, Some(1024), ValueSourceKind::ProjectedV, "rope_cos_swa", "rope_sin_swa")
            };
            LayerSchedule {
                kind: LayerKind::Attention,
                attention: LayerAttentionConfig {
                    head_dim,
                    kv_heads,
                    mask_window,
                    value_source_kind,
                    key_source_kind: KeySourceKind::ProjectedK,
                    rope_table: RopeTableSel { cos_name: cos_name.into(), sin_name: sin_name.into() },
                    rope_pairing: RopePairing::SplitHalf { pairs: head_dim / 2 },
                    score_scale: AttentionScoreScale::Unscaled,
                    value_norm: true,
                },
                ffn,
            }
        })
        .collect()
}

#[test]
fn descriptor_real_dims_gemma4_26b_program_equals_direct_builder() {
    let mapping = GEMMA4_26B.open();
    let parsed = parse_complete(&mapping).expect("the real gemma4 26B header parses");

    let profile = family_profile(GEMMA4_26B.architecture).expect("the gemma4 profile is embedded");
    let descriptor = gemma4_descriptor_from_gguf(&parsed, false, &profile)
        .expect("the production builder reads the real 26B header");

    assert_eq!(descriptor.block_count, 30);
    assert_eq!(descriptor.embedding, 2816);
    assert_eq!(descriptor.feed_forward, 2112);
    assert_eq!(descriptor.expert_feed_forward, 704);
    assert_eq!(descriptor.query_heads, 16);
    assert_eq!((descriptor.expert_count, descriptor.expert_used_count), (128, 8));
    assert_eq!(descriptor.logit_softcap, Some(30.0));
    assert_eq!((descriptor.cache_strategy, descriptor.cache_mask), (CacheStrategy::Cached, CacheMask::Padded));
    let (direct, direct_logits, direct_roots, direct_moe, _head_repeats) =
        lfm2_two_range_cached_forward_program_with_experts(
            descriptor.vocab,
            2816,
            2112,
            704,
            16,
            30,
            128,
            8,
            0,
            &gemma4_26b_hand_written_layers(),
            Some(EmbeddingScale::Sqrt),
            Some(30.0),
            true,
            None,
            false,
        )
        .expect("direct real-dims build");

    let ForwardProgram { program, logits, layer_roots, moe_sites, .. } =
        build_forward(&descriptor).expect("build_forward real-dims build");

    assert_programs_identical(&direct, &program);
    assert_eq!(direct_logits, logits, "root node id mismatch");
    assert_eq!(direct_roots, owned_cache_roots(&layer_roots), "cache roots mismatch");
    assert_eq!(direct_moe.0.len(), moe_sites.0.len(), "moe site count mismatch");
}

#[test]
fn descriptor_real_dims_openchat_program_equals_direct_builder() {
    let mapping = OPENCHAT.open();
    let parsed = parse_complete(&mapping).expect("the real openchat header parses");
    let shape = architecture_from_metadata(&parsed).expect("the real openchat header reads");
    assert_eq!(
        (shape.embedding, shape.feed_forward, shape.query_heads, shape.kv_heads, shape.head_dim, shape.block_count),
        (4096, 14336, 32, 8, 128, 32),
    );

    let (direct, direct_roots, direct_cache_roots, direct_residuals, direct_moe) =
        mistral_cached_forward_program_with_experts_and_layer_taps(
            shape.vocab,
            shape.embedding,
            shape.feed_forward,
            shape.query_heads,
            shape.kv_heads,
            shape.head_dim,
            shape.block_count,
            shape.expert_count,
            shape.expert_used_count,
            false,
            false,
            false,
            false,
            true,
        )
        .expect("direct real-dims build");
    let descriptor = mistral_descriptor_from_shape(
        shape.vocab,
        shape.embedding,
        shape.feed_forward,
        shape.query_heads,
        shape.kv_heads,
        shape.head_dim,
        shape.block_count,
        shape.expert_count,
        shape.expert_used_count,
        false,
        false,
        false,
        false,
        &family_profile(OPENCHAT.architecture).expect("the openchat family profile is embedded"),
    );

    let ForwardProgram { program, logits, layer_roots, moe_sites, layer_residuals, hidden, .. } =
        build_forward(&descriptor).expect("build_forward real-dims build");

    assert_programs_identical(&direct, &program);
    assert_eq!(direct_roots.logits, logits, "root node id mismatch");
    assert_eq!(Some(direct_roots.hidden), hidden, "hidden root mismatch");
    assert_eq!(direct_cache_roots, owned_cache_roots(&layer_roots), "cache roots mismatch");
    assert_eq!(direct_moe.0.len(), moe_sites.0.len(), "moe site count mismatch");
    assert_eq!(direct_residuals, layer_residuals, "layer-residual roots mismatch");
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

#[test]
fn granite_moe_header_declares_the_scales_the_profile_cannot_carry() {
    let mapping = GRANITE_MOE.open();
    let parsed = parse_complete(&mapping).expect("the real granite moe header parses");
    assert_architecture_key(&GRANITE_MOE, &parsed);

    let scale = |key: &str| metadata_f32_optional(&parsed, &format!("granitemoe.{key}"), -1.0);
    let count = |key: &str| {
        metadata_u32(&parsed, &format!("granitemoe.{key}")).expect("granite header declares the key")
    };
    assert_eq!(scale("embedding_scale"), 12.0);
    assert_eq!(scale("residual_scale"), 0.22_f32);
    assert_eq!(scale("logit_scale"), 6.0);
    assert_eq!(scale("attention.scale"), 0.015625);
    assert_eq!(count("expert_count"), 32);
    assert_eq!(count("expert_used_count"), 8);
    assert_eq!(count("block_count"), 24);
    assert_eq!(count("embedding_length"), 1024);
    assert_eq!(
        metadata_str(&parsed, "tokenizer.ggml.pre").expect("granite declares a pre-tokenizer"),
        "refact"
    );
}

fn constants_equal(program: &[Op], value: f32) -> usize {
    program
        .iter()
        .filter(|op| matches!(op, Op::Constant { value: found, .. } if *found == value))
        .count()
}

#[test]
fn granite_moe_program_carries_the_header_scales() {
    let mapping = GRANITE_MOE.open();
    let file_bytes: &[u8] = &mapping;
    let parsed = parse_complete(file_bytes).expect("the real granite moe header parses");
    let route = resolve(&GRANITE_MOE, &parsed);
    let bound = route
        .bind_with_kv_layout(&parsed, file_bytes, KvLayout::SlidingRing)
        .expect("granite moe binds under the sliding ring layout");

    assert_eq!(constants_equal(&bound.program, 12.0), 1, "embedding scale");
    assert_eq!(constants_equal(&bound.program, 1.0_f32 / 6.0_f32), 1, "reciprocal logit scale");
    assert_eq!(constants_equal(&bound.program, 0.22_f32), 1, "residual scale");
    assert_eq!(constants_equal(&bound.program, 0.015625), 1, "attention scale");
}

const LLAMA_GENERATED_TOKENS: usize = 32;

struct LlamaCase {
    prompt: String,
    prompt_ids: Vec<u32>,
    generated_ids: Vec<u32>,
}

fn llama_cases(checkpoint: &Checkpoint) -> Vec<LlamaCase> {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/llama-parity")
        .join(checkpoint.name)
        .join("llama_ids.json");
    let text = std::fs::read_to_string(&fixture).unwrap_or_else(|error| {
        panic!(
            "{}: no llama.cpp oracle ids at {}: {error}; llama.cpp f1ea20621 has to load this checkpoint to produce them",
            checkpoint.name,
            fixture.display()
        )
    });
    let records: Vec<serde_json::Value> =
        serde_json::from_str(&text).expect("llama_ids.json is a json array of records");
    assert!(
        !records.is_empty(),
        "{}: llama_ids.json holds zero records",
        checkpoint.name
    );
    records
        .iter()
        .map(|record| LlamaCase {
            prompt: record["prompt"]
                .as_str()
                .expect("record has a prompt")
                .to_owned(),
            prompt_ids: ids_of(&record["prompt_ids"]),
            generated_ids: ids_of(&record["generated_ids"]),
        })
        .collect()
}

fn ids_of(value: &serde_json::Value) -> Vec<u32> {
    value
        .as_array()
        .expect("ids are a json array")
        .iter()
        .map(|id| {
            u32::try_from(id.as_u64().expect("an id is an unsigned integer"))
                .expect("an id fits u32")
        })
        .collect()
}

fn first_divergence(expected: &[u32], actual: &[u32]) -> Option<usize> {
    let shared = expected.len().min(actual.len());
    (0..shared)
        .find(|&index| expected[index] != actual[index])
        .or_else(|| (expected.len() != actual.len()).then_some(shared))
}

fn llama_parity(checkpoint: &Checkpoint) {
    let mapping = checkpoint.open();
    let file_bytes: &[u8] = &mapping;
    let parsed = parse_complete(file_bytes).expect("parses the real checkpoint's GGUF header");
    let model = LoadedModel::load(&parsed, file_bytes)
        .unwrap_or_else(|error| panic!("{}: LoadedModel::load failed: {error:?}", checkpoint.name));
    assert_llama_ids(checkpoint, &parsed, &model);
}

fn assert_llama_ids(checkpoint: &Checkpoint, parsed: &proxima_gguf::pipe::ParsedGguf, model: &LoadedModel<'_>) {
    let cases = llama_cases(checkpoint);
    let config = ServingConfig {
        prompt_cache: PromptCacheConfig::off(),
        ..ServingConfig::default()
    };
    let vocab = proxima_tokenizer::gguf::vocab_from_metadata(parsed)
        .expect("builds the vocab from the checkpoint metadata");
    let wants_bos = vocab
        .add_bos_token()
        .unwrap_or_else(|| vocab.bos_token_id().is_some());
    let wants_eos = vocab.add_eos_token().unwrap_or(false);

    let mut failures = Vec::new();
    for case in &cases {
        let own_prompt_ids =
            proxima_tokenizer::encode_with_bos_eos(&case.prompt, &vocab, wants_bos, wants_eos)
                .expect("proxima tokenizes the prompt");
        if let Some(index) = first_divergence(&case.prompt_ids, &own_prompt_ids) {
            failures.push(format!(
                "TOKENIZER {} prompt {:?}: first divergent index {index}; llama prompt_ids {:?}, proxima {:?}",
                checkpoint.name, case.prompt, case.prompt_ids, own_prompt_ids
            ));
        }
        let (generated, _text, _stopped) = model
            .generate_from_ids(
                &case.prompt_ids,
                LLAMA_GENERATED_TOKENS,
                &config,
                &mut |_event| ControlFlow::Continue(()),
            )
            .unwrap_or_else(|error| {
                panic!("{}: generate_from_ids failed: {error:?}", checkpoint.name)
            });
        let compared_len = generated.len().min(case.generated_ids.len());
        let compared = &generated[..compared_len];
        if let Some(index) = first_divergence(&case.generated_ids, compared) {
            failures.push(format!(
                "MODEL {} prompt {:?}: first divergent index {index}; llama {:?}, proxima {:?}",
                checkpoint.name, case.prompt, case.generated_ids, generated
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} divergences from llama.cpp across {} prompts:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}

#[test]
fn llama_parity_gemma4_26b() {
    llama_parity(&GEMMA4_26B);
}

#[test]
fn llama_parity_gemma4_e2b() {
    llama_parity(&GEMMA4_E2B);
}

#[test]
fn llama_parity_granite_moe() {
    llama_parity(&GRANITE_MOE);
}

#[test]
fn llama_parity_openchat() {
    llama_parity(&OPENCHAT);
}

#[test]
fn llama_parity_qwen2() {
    llama_parity(&QWEN2);
}

#[test]
fn llama_parity_qwen3() {
    llama_parity(&QWEN3);
}

const FORCED_DRAFT_WIDTHS: [u16; 2] = [1, 3];

fn generic_verify_llama_parity(checkpoint: &Checkpoint) {
    let mapping = checkpoint.open();
    let file_bytes: &[u8] = &mapping;
    let parsed = parse_complete(file_bytes).expect("parses the real checkpoint's GGUF header");
    let armed = ModelDescriptor {
        speculative_verify: true,
        ..config_descriptors(checkpoint, &parsed).remove(0).1
    };
    let model = load_from_config(&parsed, file_bytes, &armed);
    let config = ServingConfig {
        prompt_cache: PromptCacheConfig::off(),
        ..ServingConfig::default()
    };
    let cases = llama_cases(checkpoint);
    let mut verify_steps_total = 0;
    let mut failures = Vec::new();
    for case in &cases {
        for width in FORCED_DRAFT_WIDTHS {
            let mut stats = SpeculativeDecodeStats::default();
            let (generated, _text, _stopped) = model
                .generate_from_ids_with_speculative_stats(
                    &case.prompt_ids,
                    LLAMA_GENERATED_TOKENS,
                    &config,
                    &mut |_event| ControlFlow::Continue(()),
                    &mut stats,
                    Some(width),
                )
                .unwrap_or_else(|error| panic!("{}: speculative generate failed: {error:?}", checkpoint.name));
            verify_steps_total += stats.verify_steps;
            let compared = &generated[..generated.len().min(case.generated_ids.len())];
            if let Some(index) = first_divergence(&case.generated_ids, compared) {
                failures.push(format!(
                    "{} prompt {:?} draft width {width}: first divergent index {index}; llama {:?}, proxima {:?}",
                    checkpoint.name, case.prompt, case.generated_ids, generated
                ));
            }
        }
    }
    assert!(
        verify_steps_total > 0,
        "{}: the verify program never ran across {} prompts, so nothing was checked",
        checkpoint.name,
        cases.len()
    );
    assert!(
        failures.is_empty(),
        "{} divergences from llama.cpp with speculation on:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn generic_verify_llama_parity_gemma4_e2b() {
    generic_verify_llama_parity(&GEMMA4_E2B);
}

#[test]
fn generic_verify_llama_parity_openchat() {
    generic_verify_llama_parity(&OPENCHAT);
}

#[test]
fn generic_verify_llama_parity_qwen2() {
    generic_verify_llama_parity(&QWEN2);
}

#[test]
fn generic_verify_llama_parity_qwen3() {
    generic_verify_llama_parity(&QWEN3);
}

#[test]
fn generic_verify_llama_parity_granite_moe() {
    generic_verify_llama_parity(&GRANITE_MOE);
}

fn load_from_config<'file>(
    parsed: &proxima_gguf::pipe::ParsedGguf,
    file_bytes: &'file [u8],
    config: &ModelDescriptor,
) -> LoadedModel<'file> {
    conflaguration::Validate::validate(config).expect("the config is internally consistent");
    LoadedModel::load_with_descriptor(parsed, file_bytes, config)
        .unwrap_or_else(|error| panic!("load_with_descriptor failed: {error:?}"))
}

fn hand_written_config(file: &str) -> ModelDescriptor {
    let config_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/model-configs").join(file);
    conflaguration::from_file(&config_path).unwrap_or_else(|error| panic!("{} does not load: {error}", config_path.display()))
}

fn assert_hand_written_config_equals_header(checkpoint: &Checkpoint, file: &str) {
    let written = hand_written_config(file);
    let mapping = checkpoint.open();
    let parsed = parse_complete(&mapping).expect("parses the real checkpoint's GGUF header");
    let architecture = architecture_from_metadata(&parsed).expect("the checkpoint declares an architecture");

    let derived = dense_descriptor_from_gguf(&parsed, &architecture)
        .unwrap_or_else(|error| panic!("{}: the header does not yield a descriptor: {error:?}", checkpoint.name));

    assert_eq!(written, derived, "{file} disagrees with {}'s own header", checkpoint.name);
}

#[test]
fn zero_rust_variant_hand_written_qwen2_config_reproduces_llama_ids() {
    let config = hand_written_config("qwen2.toml");
    let mapping = QWEN2.open();
    let file_bytes: &[u8] = &mapping;
    let parsed = parse_complete(file_bytes).expect("parses the real checkpoint's GGUF header");

    let model = load_from_config(&parsed, file_bytes, &config);

    let ForwardProgram { program, .. } = build_forward(&config).expect("the hand-written config lowers");
    assert_eq!(model.op_count(), program.len(), "the loaded model must run the config's lowering");
    assert_llama_ids(&QWEN2, &parsed, &model);
}

#[test]
fn hand_written_qwen2_config_equals_the_gguf_header_descriptor() {
    assert_hand_written_config_equals_header(&QWEN2, "qwen2.toml");
}

#[test]
fn hand_written_granite_moe_config_equals_the_gguf_header_descriptor() {
    assert_hand_written_config_equals_header(&GRANITE_MOE, "granite_moe.toml");
}

fn window_ceiling_constants(program: &[Op], window: u32) -> usize {
    let ceiling = (window as f32 - 1.0).to_bits();
    program
        .iter()
        .filter(|op| matches!(op, Op::Constant { value, .. } if value.to_bits() == ceiling))
        .count()
}

fn full_attention(base: &ModelDescriptor) -> ModelDescriptor {
    let layers = base
        .layers
        .iter()
        .map(|layer| LayerSchedule {
            attention: LayerAttentionConfig {
                mask_window: None,
                ..layer.attention.clone()
            },
            ..layer.clone()
        })
        .collect();
    ModelDescriptor {
        sliding_kv_ring: false,
        layers,
        ..base.clone()
    }
}

#[test]
fn zero_rust_variant_full_attention_gemma4_e2b_lowers_and_runs() {
    let variant = hand_written_config("gemma4_e2b_full_attention.toml");
    let mapping = GEMMA4_E2B.open();
    let file_bytes: &[u8] = &mapping;
    let parsed = parse_complete(file_bytes).expect("parses the real checkpoint's GGUF header");
    let base = gemma4::descriptor_from_gguf(&parsed, true).expect("the e2b header carries every key the descriptor reads");
    let window = base.layers[0].attention.mask_window.expect("e2b's first layer is a sliding layer");

    assert_eq!(variant, full_attention(&base), "the file must differ from e2b in the layer windows and the ring only");
    assert!(
        variant.layers.iter().all(|layer| layer.kind == LayerKind::Attention && layer.attention.mask_window.is_none()),
        "every layer of the file must be a full-attention layer"
    );

    let ForwardProgram { program: base_program, .. } = build_forward(&base).expect("the base lowers");
    let ForwardProgram { program: variant_program, .. } = build_forward(&variant).expect("the file lowers");
    assert!(
        window_ceiling_constants(&base_program, window) > 0,
        "e2b's program must carry the {window}-token window mask the variant removes"
    );
    assert_eq!(window_ceiling_constants(&variant_program, window), 0, "a full-attention program carries no window mask");

    let model = load_from_config(&parsed, file_bytes, &variant);
    assert_eq!(model.op_count(), variant_program.len(), "the loaded model must run the file's lowering");
    assert_ne!(model.op_count(), base_program.len(), "removing every window must change the graph");
    assert_llama_ids(&GEMMA4_E2B, &parsed, &model);
}

fn generated_ids_diverge_from_llama(checkpoint: &Checkpoint, config: &ModelDescriptor) -> bool {
    let mapping = checkpoint.open();
    let file_bytes: &[u8] = &mapping;
    let parsed = parse_complete(file_bytes).expect("parses the real checkpoint's GGUF header");
    let model = load_from_config(&parsed, file_bytes, config);
    let case = &llama_cases(checkpoint)[0];
    let serving = ServingConfig {
        prompt_cache: PromptCacheConfig::off(),
        ..ServingConfig::default()
    };

    let (generated, _text, _stopped) = model
        .generate_from_ids(&case.prompt_ids, LLAMA_GENERATED_TOKENS, &serving, &mut |_event| ControlFlow::Continue(()))
        .expect("the edited config runs");

    let compared = &generated[..generated.len().min(case.generated_ids.len())];
    first_divergence(&case.generated_ids, compared).is_some()
}

#[test]
fn model_config_edit_to_qwen2_changes_the_generated_ids() {
    let edited = ModelDescriptor {
        embedding_scale: Some(EmbeddingScale::Factor(3.0)),
        ..hand_written_config("qwen2.toml")
    };

    assert!(
        generated_ids_diverge_from_llama(&QWEN2, &edited),
        "an embedding scale of 3.0 left llama's ids unchanged, so the config is not what runs"
    );
}

#[test]
fn hand_written_granite_moe_config_reproduces_llama_ids() {
    let config = hand_written_config("granite_moe.toml");
    let mapping = GRANITE_MOE.open();
    let file_bytes: &[u8] = &mapping;
    let parsed = parse_complete(file_bytes).expect("parses the real checkpoint's GGUF header");

    let model = load_from_config(&parsed, file_bytes, &config);

    assert_llama_ids(&GRANITE_MOE, &parsed, &model);
}

#[test]
fn granite_moe_config_with_the_hf_split_half_pairing_is_refused_by_the_lowering() {
    let written = hand_written_config("granite_moe.toml");
    let layers = written
        .layers
        .iter()
        .map(|layer| LayerSchedule {
            attention: LayerAttentionConfig {
                rope_pairing: RopePairing::SplitHalf { pairs: layer.attention.head_dim / 2 },
                ..layer.attention.clone()
            },
            ..layer.clone()
        })
        .collect();
    let split_half = ModelDescriptor { layers, ..written };

    let refusal = build_forward(&split_half).expect_err("the moe layer expresses split-half only with qk-norm");

    assert!(
        matches!(refusal, TensorError::UnsupportedInBuilder { feature: "a rope pairing the moe layer cannot express", .. }),
        "got {refusal:?}"
    );
}

#[test]
fn a_config_that_pairs_gate_and_up_binds_the_fused_operand_it_adds() {
    let mapping = OPENCHAT.open();
    let file_bytes: &[u8] = &mapping;
    let parsed = parse_complete(file_bytes).expect("parses the real checkpoint's GGUF header");
    let architecture = architecture_from_metadata(&parsed).expect("hparams parse from the header");
    let bound = resolve(&OPENCHAT, &parsed)
        .bind_with_kv_layout(&parsed, file_bytes, KvLayout::SlidingRing)
        .expect("binds from the header descriptor");
    let fused_leaf = |layer: u32| format!("blk.{layer}.ffn_gate_up.weight");
    let holds = |weights: &BoundWeights<'_>, name: &str| {
        weights.packed().iter().any(|(bound_name, _)| bound_name == name)
            || weights.packed_owned().iter().any(|(bound_name, _, _)| bound_name == name)
    };
    assert!(!holds(&bound.weights, &fused_leaf(0)), "the header descriptor does not pair gate and up");

    let config = ModelDescriptor {
        paired_gate_up_reduce: true,
        ..dense_descriptor_from_gguf(&parsed, &architecture).expect("the header descriptor")
    };
    let lowered = bound
        .lowered_from(&config, &parsed, file_bytes)
        .expect("lowers the config and binds the leaves it adds");

    for layer in 0..architecture.block_count {
        assert!(holds(&lowered.weights, &fused_leaf(layer)), "layer {layer} fused gate/up operand is bound");
    }
    assert_unbound_leaves_are_step_inputs(&OPENCHAT, &lowered);
}

#[test]
fn model_config_file_layer_turns_the_decode_config_into_the_verify_program() {
    let mapping = GEMMA4_E2B.open();
    let parsed = parse_complete(&mapping).expect("parses the real checkpoint's GGUF header");
    let decode = gemma4::descriptor_from_gguf(&parsed, true).expect("the e2b header carries every key the descriptor reads");
    let directory = tempfile::tempdir().expect("a scratch directory");
    let layer_path = directory.path().join("verify.toml");
    std::fs::write(&layer_path, "last_row_only = false\n").expect("the layer file is written");

    let layered: ModelDescriptor = conflaguration::builder()
        .value(decode)
        .env()
        .file(&layer_path)
        .validate()
        .build()
        .expect("the layered descriptor validates");

    let ForwardProgram { program, logits: logits_root, .. } = build_forward(&layered).expect("the layered config lowers");
    let incumbent = std::fs::read_to_string(GEMMA4_E2B.fixture("digest")).expect("incumbent digest exists");
    for expected in [
        format!("verify.ops={}", program.len()),
        format!("verify.ops_sha256={}", ops_digest(&program)),
        format!("verify.logits_root={logits_root:?}"),
    ] {
        assert!(
            incumbent.lines().any(|line| line == expected),
            "the file layer lowers to `{expected}`, which the incumbent verify digest lacks"
        );
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
const ARENA_BYTES: usize = 4 << 30;
#[cfg(all(feature = "metal", target_os = "macos"))]
static ARENA_BASE: OnceLock<usize> = OnceLock::new();
#[cfg(all(feature = "metal", target_os = "macos"))]
static ARENA_USED: AtomicUsize = AtomicUsize::new(0);
#[cfg(all(feature = "metal", target_os = "macos"))]
static ARENA_REQUESTS: AtomicUsize = AtomicUsize::new(0);

#[cfg(all(feature = "metal", target_os = "macos"))]
fn map_arena() -> usize {
    let pointer = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            ARENA_BYTES,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_ANON | libc::MAP_PRIVATE | libc::MAP_NORESERVE,
            -1,
            0,
        )
    };
    assert_ne!(pointer, libc::MAP_FAILED, "the kv arena could not be reserved");
    pointer as usize
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn arena_source(byte_len: usize) -> Result<omega::PlacedBuffer, omega::MetalError> {
    let page = omega::page_size();
    let length = byte_len.max(1).div_ceil(page) * page;
    let offset = ARENA_USED.fetch_add(length, Ordering::SeqCst);
    assert!(offset + length <= ARENA_BYTES, "the kv arena is exhausted at {offset} bytes");
    ARENA_REQUESTS.fetch_add(1, Ordering::SeqCst);
    let base = *ARENA_BASE.get_or_init(map_arena);
    unsafe { omega::allocate_placed_buffer_over((base + offset) as *mut u8, length) }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn generated_ids(
    model: &LoadedModel<'_>,
    cases: &[LlamaCase],
    config: &ServingConfig<'_>,
) -> Vec<Vec<u32>> {
    cases
        .iter()
        .map(|case| {
            let (generated, _text, _stopped) = model
                .generate_from_ids(
                    &case.prompt_ids,
                    LLAMA_GENERATED_TOKENS,
                    config,
                    &mut |_event| ControlFlow::Continue(()),
                )
                .unwrap_or_else(|error| panic!("generate_from_ids failed: {error:?}"));
            generated
        })
        .collect()
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn kv_in_caller_memory(checkpoint: &Checkpoint, kv_owning_layers: usize, matches_llama: bool) {
    let cases = llama_cases(checkpoint);
    let mapping = checkpoint.open();
    let file_bytes: &[u8] = &mapping;
    let parsed = parse_complete(file_bytes).expect("parses the real checkpoint's GGUF header");
    let model = LoadedModel::load(&parsed, file_bytes)
        .unwrap_or_else(|error| panic!("{}: LoadedModel::load failed: {error:?}", checkpoint.name));
    let config = ServingConfig {
        prompt_cache: PromptCacheConfig::off(),
        ..ServingConfig::default()
    };

    let default_ids = generated_ids(&model, &cases, &config);
    assert_eq!(
        ARENA_REQUESTS.load(Ordering::SeqCst),
        0,
        "the default buffer source never touches the arena"
    );
    ARENA_USED.store(0, Ordering::SeqCst);
    ARENA_REQUESTS.store(0, Ordering::SeqCst);

    let hooked = model.with_kv_buffer_source(arena_source);
    let hooked_ids = generated_ids(&hooked, &cases, &config);
    assert_eq!(hooked_ids, default_ids, "kv in caller memory changes the ids");

    if matches_llama {
        for (index, case) in cases.iter().enumerate() {
            let compared = hooked_ids[index].len().min(case.generated_ids.len());
            assert_eq!(
                first_divergence(&case.generated_ids, &hooked_ids[index][..compared]),
                None,
                "{} prompt {:?} diverges from the recorded llama ids",
                checkpoint.name,
                case.prompt
            );
        }
    }
    assert_eq!(
        ARENA_REQUESTS.load(Ordering::SeqCst),
        3 * kv_owning_layers * cases.len(),
        "the arena serves three buffers per kv-owning layer per generate call"
    );
    assert!(ARENA_USED.load(Ordering::SeqCst) > 0, "the arena served no bytes");
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn kv_in_caller_memory_gemma4_e2b() {
    kv_in_caller_memory(&GEMMA4_E2B, 15, true);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn kv_in_caller_memory_gemma4_26b() {
    kv_in_caller_memory(&GEMMA4_26B, 30, true);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn kv_in_caller_memory_granite_moe() {
    kv_in_caller_memory(&GRANITE_MOE, 24, true);
}

fn ring_layers(descriptor: &ModelDescriptor) -> Vec<bool> {
    let ForwardProgram { program, .. } = build_forward(descriptor).expect("the descriptor lowers");
    let leading_extent = |name: &str| {
        program.iter().find_map(|op| match op {
            Op::Input { name: Some(leaf), shape, .. } if leaf == name => shape.first().copied(),
            _ => None,
        })
    };
    descriptor
        .layers
        .iter()
        .enumerate()
        .map(|(layer, entry)| {
            let owner = match entry.attention.key_source_kind {
                KeySourceKind::SharedFromLayer(source) => source as usize,
                _ => layer,
            };
            let extent = leading_extent(&format!("kv_cache.{owner}.k_even"));
            assert!(extent.is_some(), "layer {layer} reads kv_cache.{owner}, which the program does not declare");
            extent == Some(Extent::Symbolic(SLIDING_KV_SYMBOL))
        })
        .collect()
}

fn llama_swa_flags(checkpoint: &Checkpoint) -> Vec<bool> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/llama-parity")
        .join(checkpoint.name)
        .join("swa_layers.txt");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    let flags: Vec<(usize, bool)> = text
        .lines()
        .filter(|line| line.starts_with("load_tensors: layer"))
        .map(|line| {
            let layer = line["load_tensors: layer".len()..].split_whitespace().next().expect("a layer index");
            let flag = line.rsplit("is_swa = ").next().expect("an is_swa flag").trim();
            (layer.parse().expect("layer index is an integer"), flag == "1")
        })
        .collect();
    let indices: Vec<usize> = flags.iter().map(|(layer, _)| *layer).collect();
    assert_eq!(indices, (0..flags.len()).collect::<Vec<_>>(), "{}: layers listed out of order or with gaps", path.display());
    flags.into_iter().map(|(_, flag)| flag).collect()
}

#[test]
fn window_ring_layers_equal_the_oracle_swa_flags_on_both_gemma4_checkpoints() {
    for checkpoint in [&GEMMA4_E2B, &GEMMA4_26B] {
        let mapping = checkpoint.open();
        let parsed = parse_complete(&mapping).expect("parses the real checkpoint's GGUF header");
        let descriptor = gemma4::descriptor_from_gguf(&parsed, true)
            .unwrap_or_else(|error| panic!("{}: descriptor_from_gguf failed: {error:?}", checkpoint.name));

        let ring = ring_layers(&descriptor);
        let oracle = llama_swa_flags(checkpoint);

        assert!(oracle.iter().any(|flag| *flag) && oracle.iter().any(|flag| !*flag), "{}: the oracle must mix windowed and full layers", checkpoint.name);
        assert_eq!(ring, oracle, "{}: the layers lowered onto the ring are not the layers llama.cpp marks is_swa", checkpoint.name);
    }
}

#[test]
fn window_ring_layers_follow_a_window_on_one_dense_layer() {
    let mapping = QWEN2.open();
    let parsed = parse_complete(&mapping).expect("parses the real checkpoint's GGUF header");
    let architecture = architecture_from_metadata(&parsed).expect("the checkpoint declares an architecture");
    let mut descriptor = dense_descriptor_from_gguf(&parsed, &architecture).expect("the header yields a descriptor");
    let windowed_layer = 5;
    descriptor.layers[windowed_layer].attention.mask_window = Some(64);
    descriptor.sliding_kv_ring = true;

    let ring = ring_layers(&descriptor);

    let expected: Vec<bool> = (0..descriptor.layers.len()).map(|layer| layer == windowed_layer).collect();
    assert_eq!(ring, expected, "exactly the windowed dense layer sits on the ring");
    let unwindowed = ModelDescriptor { sliding_kv_ring: true, ..dense_descriptor_from_gguf(&parsed, &architecture).expect("the header yields a descriptor") };
    assert!(ring_layers(&unwindowed).iter().all(|on_ring| !on_ring), "control: a header with no window puts no layer on the ring");
}
