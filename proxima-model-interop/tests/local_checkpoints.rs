//! Differential: the profile built from a checkpoint's own header against the
//! profile today's serving path builds, on every local decoder-only
//! checkpoint. Headers only: the parser stops at the end of the tensor
//! directory, so no weight page is read. Run with
//! `--features std,host-checkpoints`; the checkpoint list and the recorded
//! unresolved set are data in `tests/fixtures/local_checkpoints.toml`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use proxima_gguf::parser::{GgufEvent, GgufParser};
use proxima_gguf::pipe::ParsedGguf;
use proxima_model_interop::families::{
    CheckpointSha, DecoderFamily, InferError, ProfileFacts, Slot, TensorShape, gguf_config, infer,
    infer_profile,
};
use proxima_model_interop::inferred_profile::{ProfileSourceSettings, inferred_profile};
use proxima_model_interop::probes::incumbent_profile;
use proxima_tensor::spec::FamilyProfile;

const CHUNK: usize = 1 << 20;

fn incumbent(parsed: &ParsedGguf, arch: &str) -> Result<(FamilyProfile, u32, bool), String> {
    incumbent_profile(parsed, arch).map_err(|error| error.to_string())
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").expect("HOME names the host's model directory"))
}

fn read_header(path: &Path) -> ParsedGguf {
    let mut file = File::open(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    let length = file.metadata().expect("file metadata").len();
    let mut parser = GgufParser::new().with_file_len(length);
    let mut events = Vec::new();
    let mut buffer = vec![0u8; CHUNK];
    while !parser.is_complete() {
        let read = file.read(&mut buffer).expect("header bytes");
        assert!(read > 0, "{} ended inside its header", path.display());
        let (advanced, unlocked) = parser.push(&buffer[..read]).expect("well-formed header");
        parser = advanced;
        events.extend(unlocked);
    }
    assemble(events)
}

fn assemble(events: Vec<GgufEvent>) -> ParsedGguf {
    let mut parsed = ParsedGguf {
        version: 0,
        tensor_count: 0,
        kv_count: 0,
        metadata: Vec::new(),
        tensors: Vec::new(),
        data_offset: 0,
        alignment: 0,
    };
    for event in events {
        match event {
            GgufEvent::Header {
                version,
                tensor_count,
                kv_count,
            } => {
                (parsed.version, parsed.tensor_count, parsed.kv_count) =
                    (version, tensor_count, kv_count);
            }
            GgufEvent::Metadata { key, value } => parsed.metadata.push((key, value)),
            GgufEvent::Tensor(tensor) => parsed.tensors.push(tensor),
            GgufEvent::Complete {
                data_offset,
                alignment,
            } => {
                (parsed.data_offset, parsed.alignment) = (data_offset, alignment);
            }
        }
    }
    parsed
}

fn library_blobs(locations: &Locations, name: &str) -> Vec<PathBuf> {
    let manifests = home().join(&locations.manifests).join(name);
    let mut blobs = BTreeSet::new();
    for entry in fs::read_dir(&manifests).unwrap_or_else(|error| panic!("{name}: {error}")) {
        let text = fs::read_to_string(entry.expect("manifest entry").path()).expect("manifest");
        let manifest: serde_json::Value = serde_json::from_str(&text).expect("manifest json");
        let model_layer = manifest["layers"]
            .as_array()
            .and_then(|layers| {
                layers
                    .iter()
                    .find(|layer| layer["mediaType"] == locations.model_layer.as_str())
            })
            .expect("a model layer");
        let digest = model_layer["digest"]
            .as_str()
            .expect("digest")
            .replace(':', "-");
        blobs.insert(home().join(&locations.blobs).join(digest));
    }
    blobs.into_iter().collect()
}

#[derive(serde::Deserialize)]
struct Locations {
    dirs: Vec<String>,
    library: Vec<String>,
    manifests: String,
    blobs: String,
    model_layer: String,
    unresolved: BTreeMap<String, Vec<String>>,
    named: Named,
}

#[derive(serde::Deserialize)]
struct Named {
    without_a_row: String,
    states_embedding_scale: String,
}

fn locations() -> Locations {
    toml::from_str(include_str!("fixtures/local_checkpoints.toml")).expect("locations parse")
}

fn checkpoints(locations: &Locations) -> Vec<(String, PathBuf)> {
    let mut found: Vec<(String, PathBuf)> = Vec::new();
    for directory in &locations.dirs {
        let mut files: Vec<PathBuf> = fs::read_dir(home().join(directory))
            .unwrap_or_else(|error| panic!("{directory}: {error}"))
            .map(|entry| entry.expect("entry").path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "gguf"))
            .collect();
        files.sort();
        found.extend(
            files
                .into_iter()
                .map(|path| (path.display().to_string(), path)),
        );
    }
    for name in &locations.library {
        for blob in library_blobs(locations, name) {
            found.push((format!("{name}:{}", blob.display()), blob));
        }
    }
    found
}

fn status<T: PartialEq + Debug>(name: &str, slot: Slot<T>, today: T, rows: &mut Vec<String>) {
    let word = match &slot {
        Slot::Known(found) if *found == today => "equal".to_string(),
        Slot::Known(found) => format!("DIFFERS inferred={found:?} today={today:?}"),
        Slot::NotApplicable => "n/a".to_string(),
        Slot::Unresolved => format!("unresolved (today={today:?})"),
    };
    rows.push(format!("    {name:<36} {word}"));
}

fn compare(
    facts: &ProfileFacts,
    profile: &FamilyProfile,
    experts: u32,
    qk_norm: bool,
) -> Vec<String> {
    let mut rows = Vec::new();
    status(
        "embedding_scale",
        facts.embedding_scale,
        profile.embedding_scale,
        &mut rows,
    );
    status(
        "ffn.post_attention_norm",
        facts.post_attention_norm,
        profile.ffn.post_attention_norm,
        &mut rows,
    );
    status(
        "ffn.combination",
        facts.layer_combination,
        profile.layer_ffn(experts).combination,
        &mut rows,
    );
    status(
        "ffn.output_scale",
        facts.output_scale,
        profile.ffn.output_scale,
        &mut rows,
    );
    status(
        "ffn.routed_gating",
        facts.routed_gating,
        profile.ffn.routed_gating,
        &mut rows,
    );
    status(
        "ffn.routed_expert_bias",
        facts.routed_expert_bias,
        profile.ffn.routed_expert_bias,
        &mut rows,
    );
    status(
        "ffn.activation",
        facts.activation,
        profile.ffn.activation,
        &mut rows,
    );
    status(
        "ffn.exclusive_dense_post_norm",
        facts.exclusive_dense_post_norm,
        profile.ffn.exclusive_dense_post_norm,
        &mut rows,
    );
    status(
        "ffn.shared_expert_gate",
        facts.shared_expert_gate,
        profile.ffn.shared_expert_gate,
        &mut rows,
    );
    status(
        "score_scale_inverse_sqrt_head_dim",
        facts.score_scale_inverse_sqrt_head_dim,
        profile.score_scale_inverse_sqrt_head_dim,
        &mut rows,
    );
    status(
        "value_norm",
        facts.value_norm,
        profile.value_norm,
        &mut rows,
    );
    status("qk_norm", facts.qk_norm, qk_norm, &mut rows);
    status(
        "rope_layout",
        facts.rope_layout,
        profile.rope_layout,
        &mut rows,
    );
    status(
        "recurrence",
        facts.recurrence,
        profile.recurrence,
        &mut rows,
    );
    status(
        "schedule_source",
        facts.schedule_source,
        profile.schedule_source,
        &mut rows,
    );
    status(
        "ssm_v_head_reordered",
        facts.ssm_v_head_reordered,
        profile.ssm_v_head_reordered,
        &mut rows,
    );
    rows
}

fn infer_facts(parsed: &ParsedGguf) -> Result<ProfileFacts, String> {
    let family = DecoderFamily::builtin().expect("embedded family parses");
    let templates = family.templates().expect("templates expand");
    let tensors: Vec<TensorShape> = parsed
        .tensors
        .iter()
        .map(|tensor| TensorShape {
            name: tensor.name.clone(),
            shape: tensor.dims.to_vec(),
        })
        .collect();
    let sha = CheckpointSha::parse(&"0".repeat(64)).expect("64 hex digits");
    let config = gguf_config(parsed);
    let inferred = infer(&tensors, &config, &templates, sha).map_err(describe)?;
    infer_profile(&family, &inferred, &config).map_err(|error| error.to_string())
}

fn describe(error: InferError) -> String {
    match error {
        InferError::UnclaimedTensors { names, .. } => {
            let heads: BTreeSet<String> = names
                .iter()
                .map(|name| name.split('.').take(2).collect::<Vec<_>>().join("."))
                .collect();
            format!("{} tensors unclaimed, name heads {heads:?}", names.len())
        }
        other => other.to_string(),
    }
}

#[test]
fn inferred_profile_equals_todays_profile_on_every_local_decoder_checkpoint() {
    let mut differs = Vec::new();
    let mut refused = Vec::new();
    let mut unexpected = Vec::new();
    let locations = locations();
    let mut unresolved = Vec::new();
    let mut compared = 0usize;
    let mut checkpoints_seen = 0usize;
    for (label, path) in checkpoints(&locations) {
        let parsed = read_header(&path);
        let arch = match parsed.metadata_value("general.architecture") {
            Some(proxima_gguf::value::MetadataValue::String(arch)) => arch.clone(),
            other => panic!("{label}: no architecture, got {other:?}"),
        };
        let facts = match infer_facts(&parsed) {
            Ok(facts) => facts,
            Err(error) => {
                refused.push(format!("{label}: {error}"));
                continue;
            }
        };
        checkpoints_seen += 1;
        let open: Vec<String> = facts.unresolved().into_iter().map(String::from).collect();
        if locations.unresolved.get(&arch) != Some(&open) {
            unexpected.push(format!("{label}: {arch} left {open:?} open"));
        }
        let Ok((profile, experts, qk_norm)) = incumbent(&parsed, &arch) else {
            println!("{label} (arch {arch}): no incumbent profile; inferred only");
            unresolved.push(format!("{label}: {:?}", facts.unresolved()));
            continue;
        };
        let rows = compare(&facts, &profile, experts, qk_norm);
        println!("{label} (arch {arch}, experts {experts})");
        rows.iter().for_each(|row| println!("{row}"));
        compared += rows.iter().filter(|row| row.ends_with("equal")).count();
        differs.extend(
            rows.iter()
                .filter(|row| row.contains("DIFFERS"))
                .map(|row| format!("{label}:{}", row.trim())),
        );
        unresolved.push(format!("{label}: {:?}", facts.unresolved()));
    }
    unresolved
        .iter()
        .for_each(|line| println!("UNRESOLVED {line}"));

    assert_eq!(
        refused,
        Vec::<String>::new(),
        "checkpoints the inferrer refused"
    );
    assert_eq!(checkpoints_seen, 16, "every local checkpoint inferred");
    assert!(compared > 0, "no field compared");
    assert_eq!(
        differs,
        Vec::<String>::new(),
        "inferred profile differs from today's"
    );
    assert_eq!(
        unexpected,
        Vec::<String>::new(),
        "unresolved set moved from the recorded one"
    );
}

fn serving(parsed: &ParsedGguf, arch: &str) -> Result<FamilyProfile, String> {
    let settings = ProfileSourceSettings::builder().inferred(true).build();
    inferred_profile(parsed, arch, &settings)
        .map_err(|error| error.to_string())?
        .ok_or_else(|| format!("{arch}: no template or no facts section"))
}

fn field<T: PartialEq + Debug>(name: &str, served: T, old: T, rows: &mut Vec<String>) {
    let word = if served == old {
        "equal".to_string()
    } else {
        format!("DIFFERS served={served:?} old={old:?}")
    };
    rows.push(format!("{name}: {word}"));
}

// a slot the file marks not applicable has no value to compare: the serving
// profile holds a neutral there and no lowering of this checkpoint reads it
fn served_vs_old(
    facts: &ProfileFacts,
    served: &FamilyProfile,
    old: &FamilyProfile,
    experts: u32,
    qk_norm: bool,
) -> Vec<String> {
    let (new_ffn, old_ffn) = (served.layer_ffn(experts), old.layer_ffn(experts));
    let mut rows = Vec::new();
    field("embedding_scale", served.embedding_scale, old.embedding_scale, &mut rows);
    field("ffn.post_attention_norm", new_ffn.post_attention_norm, old_ffn.post_attention_norm, &mut rows);
    field("ffn.combination", new_ffn.combination, old_ffn.combination, &mut rows);
    field("ffn.output_scale", new_ffn.output_scale, old_ffn.output_scale, &mut rows);
    field("ffn.activation", new_ffn.activation, old_ffn.activation, &mut rows);
    field(
        "ffn.exclusive_dense_post_norm",
        new_ffn.exclusive_dense_post_norm,
        old_ffn.exclusive_dense_post_norm,
        &mut rows,
    );
    if facts.routed_gating != Slot::NotApplicable {
        field("ffn.routed_gating", new_ffn.routed_gating, old_ffn.routed_gating, &mut rows);
    }
    if facts.routed_expert_bias != Slot::NotApplicable {
        field("ffn.routed_expert_bias", new_ffn.routed_expert_bias, old_ffn.routed_expert_bias, &mut rows);
    }
    if facts.shared_expert_gate != Slot::NotApplicable {
        field("ffn.shared_expert_gate", new_ffn.shared_expert_gate, old_ffn.shared_expert_gate, &mut rows);
    }
    field(
        "score_scale_inverse_sqrt_head_dim",
        served.score_scale_inverse_sqrt_head_dim,
        old.score_scale_inverse_sqrt_head_dim,
        &mut rows,
    );
    field("value_norm", served.value_norm, old.value_norm, &mut rows);
    field("qk_norm", served.qk_norm, qk_norm, &mut rows);
    field("rope_layout", served.rope_layout, old.rope_layout, &mut rows);
    field("recurrence", served.recurrence, old.recurrence, &mut rows);
    field("schedule_source", served.schedule_source, old.schedule_source, &mut rows);
    if facts.ssm_v_head_reordered != Slot::NotApplicable {
        field("ssm_v_head_reordered", served.ssm_v_head_reordered, old.ssm_v_head_reordered, &mut rows);
    }
    rows
}

#[test]
fn serving_profile_equals_the_old_row_profile_on_every_local_decoder_checkpoint() {
    let locations = locations();
    let (mut differs, mut checked, mut equal, mut seen) = (Vec::new(), 0usize, 0usize, 0usize);
    for (label, path) in checkpoints(&locations) {
        let parsed = read_header(&path);
        let arch = match parsed.metadata_value("general.architecture") {
            Some(proxima_gguf::value::MetadataValue::String(arch)) => arch.clone(),
            other => panic!("{label}: no architecture, got {other:?}"),
        };
        let served = serving(&parsed, &arch).unwrap_or_else(|error| panic!("{label}: {error}"));
        seen += 1;
        let Ok((old, experts, qk_norm)) = incumbent(&parsed, &arch) else {
            println!("{label} (arch {arch}): no old row; served profile complete");
            continue;
        };
        let facts = infer_facts(&parsed).expect("inferred");
        let rows = served_vs_old(&facts, &served, &old, experts, qk_norm);
        checked += 1;
        equal += rows.iter().filter(|row| row.ends_with("equal")).count();
        differs.extend(
            rows.iter()
                .filter(|row| row.contains("DIFFERS"))
                .map(|row| format!("{label}: {row}")),
        );
    }
    println!("SERVING served={seen} compared={checked} equal_fields={equal} differs={}", differs.len());
    assert_eq!(seen, 16, "every local checkpoint served");
    assert_eq!(checked, 14, "the checkpoints with an old row, compared field by field; the rest have none");
    assert_eq!(differs, Vec::<String>::new(), "serving profile differs from the old row");
    assert!(equal > 0, "no field compared");
}

#[test]
fn a_header_with_no_row_yields_a_complete_profile_from_inference_and_facts() {
    let locations = locations();
    let target = locations.named.without_a_row.clone();
    let mut found = 0usize;
    for (label, path) in checkpoints(&locations) {
        let parsed = read_header(&path);
        if !matches!(
            parsed.metadata_value("general.architecture"),
            Some(proxima_gguf::value::MetadataValue::String(arch)) if *arch == target
        ) {
            continue;
        }
        found += 1;
        let served = serving(&parsed, &target).unwrap_or_else(|error| panic!("{label}: {error}"));
        assert_eq!(served.rope_layout, proxima_tensor::spec::RopeLayout::SplitHalf, "{label}");
        assert_eq!(served.ffn.activation, proxima_tensor::spec::Activation::GeluTanh, "{label}");
        assert!(served.qk_norm && served.ffn.post_attention_norm, "{label}: this family carries q/k norms and a post-attention norm");
    }
    assert!(found > 0, "no {target} checkpoint among the local ones");
}

#[test]
fn a_fact_that_contradicts_the_header_is_refused_and_an_open_field_is_named() {
    let locations = locations();
    let target = locations.named.states_embedding_scale.clone();
    let (_, path) = checkpoints(&locations)
        .into_iter()
        .find(|(_, path)| header_arch(path) == target)
        .expect("a local checkpoint of the named architecture");
    let parsed = read_header(&path);
    let residue = "activation = \"Silu\"\nrope_layout = \"adjacent\"\nvalue_norm = false\nscore_scale_inverse_sqrt_head_dim = true\n";
    let contradicting = ProfileSourceSettings::builder()
        .facts_file(write_facts(
            "contradiction",
            &format!("[{target}]\n{residue}embedding_scale = \"None\"\n"),
        ))
        .build();
    let error = inferred_profile(&parsed, &target, &contradicting).expect_err("a contradiction is refused");
    assert!(error.to_string().contains("embedding_scale"), "{error}");
    let incomplete = ProfileSourceSettings::builder()
        .facts_file(write_facts("incomplete", &format!("[{target}]\nactivation = \"Silu\"\n")))
        .build();
    let error = inferred_profile(&parsed, &target, &incomplete).expect_err("an open field is refused");
    let message = error.to_string();
    assert!(message.contains(&target) && message.contains("rope_layout"), "{message}");
    fs::remove_dir_all(scratch()).expect("remove scratch");
}

fn header_arch(path: &Path) -> String {
    match read_header(path).metadata_value("general.architecture") {
        Some(proxima_gguf::value::MetadataValue::String(arch)) => arch.clone(),
        other => panic!("{}: no architecture, got {other:?}", path.display()),
    }
}

fn scratch() -> PathBuf {
    std::env::temp_dir().join(format!("facts-{}", std::process::id()))
}

fn write_facts(name: &str, text: &str) -> String {
    fs::create_dir_all(scratch()).expect("scratch dir");
    let path = scratch().join(format!("{name}.toml"));
    fs::write(&path, text).expect("write facts");
    path.display().to_string()
}

#[test]
fn every_local_checkpoint_yields_a_complete_inferred_profile() {
    let locations = locations();
    let mut served = 0usize;
    for (label, path) in checkpoints(&locations) {
        let parsed = read_header(&path);
        let arch = match parsed.metadata_value("general.architecture") {
            Some(proxima_gguf::value::MetadataValue::String(arch)) => arch.clone(),
            other => panic!("{label}: no architecture, got {other:?}"),
        };
        serving(&parsed, &arch).unwrap_or_else(|error| panic!("{label}: {error}"));
        served += 1;
    }
    assert_eq!(served, 16, "every local checkpoint yields a complete profile");
}
