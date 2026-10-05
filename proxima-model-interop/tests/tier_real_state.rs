#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs::File;
use std::path::Path;
use std::sync::OnceLock;

use proxima_gguf::parse_complete;
use proxima_model_interop::InteropError;
use proxima_model_interop::block_file::{MappedBlockFile, decode_block};
use proxima_model_interop::{LoadedModel, PrefixState, PromptCacheConfig, ServingConfig};

#[path = "support/block_write.rs"]
mod block_write;

use block_write::write_block_file;

const CHECKPOINT_ENV: &str = "PROXIMA_ARCH_GEMMA4_E2B_GGUF";
const CHECKPOINT_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";
const PROOF_DIGEST: [u8; 16] = [0x7e; 16];
const CONTENT_KEY: u64 = 7;

fn open_checkpoint() -> memmap2::Mmap {
    let path = std::env::var(CHECKPOINT_ENV).unwrap_or_else(|_| CHECKPOINT_PATH.to_string());
    assert!(
        Path::new(&path).exists(),
        "gemma4_e2b checkpoint is missing at {path}: stage it there or set {CHECKPOINT_ENV}"
    );
    let file = File::open(&path).unwrap_or_else(|error| panic!("open {path}: {error}"));
    unsafe { memmap2::Mmap::map(&file) }.expect("mmap the real checkpoint read-only")
}

fn longest_prompt_record() -> (String, Vec<u32>) {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/llama-parity/gemma4_e2b/llama_ids.json");
    let text = std::fs::read_to_string(&fixture)
        .unwrap_or_else(|error| panic!("read {}: {error}", fixture.display()));
    let records: Vec<serde_json::Value> = serde_json::from_str(&text).expect("llama_ids.json is a json array");
    let record = records
        .iter()
        .max_by_key(|record| record["prompt_ids"].as_array().expect("record has prompt_ids").len())
        .expect("llama_ids.json holds records");
    let prompt = record["prompt"].as_str().expect("record has a prompt").to_owned();
    let prompt_ids = record["prompt_ids"]
        .as_array()
        .expect("prompt_ids is an array")
        .iter()
        .map(|id| u32::try_from(id.as_u64().expect("an id is unsigned")).expect("an id fits u32"))
        .collect();
    (prompt, prompt_ids)
}

fn real_state() -> &'static (Vec<u32>, usize, Vec<u8>) {
    static STATE: OnceLock<(Vec<u32>, usize, Vec<u8>)> = OnceLock::new();
    STATE.get_or_init(|| {
        let (prompt, prompt_ids) = longest_prompt_record();
        let mapping = open_checkpoint();
        let file_bytes: &[u8] = &mapping;
        let parsed = parse_complete(file_bytes).expect("parses the gemma4 e2b header");
        let model = LoadedModel::load(&parsed, file_bytes).expect("loads gemma4 e2b");
        let serving = ServingConfig { prompt_cache: PromptCacheConfig::off(), ..ServingConfig::default() };
        let state = model.prefill_prefix(&prompt, &serving).expect("prefills the real prompt");
        assert_eq!(
            state.len(),
            prompt_ids.len(),
            "premise: state length {} differs from prompt id count {}",
            state.len(),
            prompt_ids.len()
        );
        let mut bytes = Vec::new();
        state.to_block_file(PROOF_DIGEST, CONTENT_KEY, &mut bytes).expect("encodes the real state");
        (prompt_ids, state.len(), bytes)
    })
}

fn restore_and_encode(bytes: &[u8]) -> (PrefixState, Vec<u8>) {
    let (prompt_ids, cached_len, _) = real_state();
    let view = decode_block(bytes).expect("decodes the block file");
    let restored = PrefixState::from_block_file(prompt_ids.clone(), *cached_len, &view).expect("restores the state");
    let mut again = Vec::new();
    restored.to_block_file(PROOF_DIGEST, CONTENT_KEY, &mut again).expect("re-encodes the restored state");
    (restored, again)
}

#[test]
fn tier_real_state_device_to_host() {
    let (_, cached_len, bytes) = real_state();

    let (restored, again) = restore_and_encode(bytes);
    let view = decode_block(bytes).expect("decodes the block file");
    let layers = &view.header.layers;
    let full = layers.iter().filter(|layer| layer.ring_window == 0 && layer.rows as usize == *cached_len).count();
    let ring = layers
        .iter()
        .filter(|layer| layer.ring_window == 512 && layer.ring_capacity >= 512)
        .count();
    let absent = layers.iter().filter(|layer| layer.rows == 0).count();

    assert_eq!(&again, bytes, "re-encoded bytes differ from the first encoding");
    assert_eq!(restored.len(), *cached_len);
    assert_eq!((full, ring, absent), (3, 12, 20), "full, ring and absent layer counts");
}

#[test]
fn tier_real_state_host_to_disk() {
    let (_, _, bytes) = real_state();
    let directory = tempfile::tempdir().expect("creates a tempdir");

    let path = write_block_file(directory.path(), CONTENT_KEY, bytes).expect("writes the block file");

    assert_eq!(path.file_name().and_then(|name| name.to_str()), Some("0000000000000007.pxkv"));
    assert_eq!(&std::fs::read(&path).expect("reads the file back"), bytes);
}

#[test]
fn tier_real_state_disk_to_host() {
    let (_, _, bytes) = real_state();
    let directory = tempfile::tempdir().expect("creates a tempdir");
    let path = write_block_file(directory.path(), CONTENT_KEY, bytes).expect("writes the block file");

    let mapped = MappedBlockFile::open(&path).expect("maps the block file");
    let view = mapped.view().expect("decodes the mapped bytes");
    let (prompt_ids, cached_len, _) = real_state();
    let restored = PrefixState::from_block_file(prompt_ids.clone(), *cached_len, &view).expect("restores from disk");
    let mut again = Vec::new();
    restored.to_block_file(PROOF_DIGEST, CONTENT_KEY, &mut again).expect("re-encodes the disk state");

    assert_eq!(&again, bytes);
}
