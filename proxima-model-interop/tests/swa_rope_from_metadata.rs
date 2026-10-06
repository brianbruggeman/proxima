//! The sliding-window RoPE table gemma4 feeds `rope_cos_swa`/`rope_sin_swa` is built from the
//! checkpoint's own `rope.freq_base_swa`/`rope.dimension_count_swa`. The expected values are parsed
//! from the llama.cpp oracle listing (`gguf_kv.txt`), never from proxima's own hparams parser.
//! Model-loading tests must run with `-j 1`.

#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs::File;
use std::path::Path;

use proxima_gguf::parse_complete;
use proxima_model_interop::gemma4::program::gemma4_sliding_rope_table;
use proxima_model_interop::{KvLayout, StepInput, bind_checkpoint_with_kv_layout, sliding_rope_inputs};

const POSITIONS: usize = 1500;

fn oracle_value(listing: &str, key: &str) -> String {
    listing
        .lines()
        .find_map(|line| {
            let (_, rest) = line.split_once('|')?;
            let (_, rest) = rest.split_once('|')?;
            let (name, value) = rest.split_once('=')?;
            (name.trim() == key).then(|| value.trim().to_owned())
        })
        .unwrap_or_else(|| panic!("{key} is absent from the oracle listing"))
}

fn assert_swa_rope_matches_oracle(name: &str, env: &str, path: &str) {
    let listing_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/llama-parity")
        .join(name)
        .join("gguf_kv.txt");
    let listing = std::fs::read_to_string(&listing_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", listing_path.display()));
    let freq_base: f32 = oracle_value(&listing, "gemma4.rope.freq_base_swa")
        .parse()
        .expect("oracle freq_base_swa is a float");
    let dimension_count: u32 = oracle_value(&listing, "gemma4.rope.dimension_count_swa")
        .parse()
        .expect("oracle dimension_count_swa is an integer");

    let resolved = std::env::var(env).unwrap_or_else(|_| path.to_owned());
    assert!(
        Path::new(&resolved).exists(),
        "checkpoint {name} is missing at {resolved}: stage it there or set {env}"
    );
    let file = File::open(&resolved).expect("open the real checkpoint");
    let mapping = unsafe { memmap2::Mmap::map(&file) }.expect("mmap the real checkpoint read-only");
    let file_bytes: &[u8] = &mapping;
    let parsed = parse_complete(file_bytes).expect("parses the real checkpoint's GGUF header");
    let bound = bind_checkpoint_with_kv_layout(&parsed, file_bytes, KvLayout::SlidingRing)
        .expect("binds the real checkpoint");

    let mut inputs: Vec<StepInput> = Vec::new();
    sliding_rope_inputs(&bound.architecture, 0, POSITIONS, &mut inputs);

    let positions: Vec<usize> = (0..POSITIONS).collect();
    let (expected_cos, expected_sin) =
        gemma4_sliding_rope_table(&positions, freq_base, dimension_count);
    let cos = inputs
        .iter()
        .find(|input| input.name == "rope_cos_swa")
        .expect("sliding_rope_inputs feeds rope_cos_swa");
    let sin = inputs
        .iter()
        .find(|input| input.name == "rope_sin_swa")
        .expect("sliding_rope_inputs feeds rope_sin_swa");
    assert_eq!(cos.values.len(), POSITIONS * dimension_count as usize / 2);
    assert_eq!(cos.values, expected_cos, "{name} cos table");
    assert_eq!(sin.values, expected_sin, "{name} sin table");
}

#[test]
fn swa_rope_from_metadata_gemma4_e2b() {
    assert_swa_rope_matches_oracle(
        "gemma4_e2b",
        "PROXIMA_ARCH_GEMMA4_E2B_GGUF",
        "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd",
    );
}

#[test]
fn swa_rope_from_metadata_gemma4_26b() {
    assert_swa_rope_matches_oracle(
        "gemma4_26b",
        "PROXIMA_ARCH_GEMMA4_26B_GGUF",
        "/Users/brianbruggeman/.ollama/models/blobs/sha256-ea549b7688d4c95019754880c21e3f29c58c985a7a1c3b37b9eebd0a95224129",
    );
}
