use std::io::{Read, Seek, SeekFrom};

use proxima_gguf::pipe::parse_complete;

use super::*;

const FIXTURE_PATH: &str = "/Users/brianbruggeman/.lmstudio/models/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf";

macro_rules! parse_header_region {
    ($file:expr, $header_buf:ident) => {{
        let mut parsed = None;
        for cap in [4usize << 20, 16 << 20, 64 << 20] {
            $header_buf.resize(cap, 0);
            $file.seek(SeekFrom::Start(0)).expect("seek to file start");
            let read = $file
                .read(&mut $header_buf)
                .expect("read gguf header region");
            $header_buf.truncate(read);
            if let Ok(result) = parse_complete(&$header_buf) {
                parsed = Some(result);
                break;
            }
        }
        parsed.expect("gguf metadata region did not fit in 64 MiB")
    }};
}

/// `architecture_from_metadata` must not hard-fail this real checkpoint
/// with [`InteropError::MissingMetadataKey`] on `rope.dimension_count`
/// (genuinely absent -- confirmed via `strings` on this exact file) --
/// before the derive-when-absent fallback landed, this call errored
/// outright, never reaching the `head_count_kv` array at all. This
/// checkpoint's own `attention.head_count_kv` genuinely disagrees across
/// layers (conv vs. attention), so `architecture_from_metadata` carries
/// the array as `kv_heads_by_layer` and leaves the uniform `kv_heads` view
/// at zero; a consumer that needs one scalar gets the NAMED error
/// ([`InteropError::HeterogeneousMetadataArray`]) from
/// [`ModelHparams::uniform_kv_heads`], never a silently wrong scalar.
#[test]
#[ignore = "depends on a ~5 GB host-local lfm2 gguf checkout outside this repo"]
fn architecture_from_metadata_names_the_heterogeneous_kv_heads_honestly() {
    crate::test_support::require_fixture(FIXTURE_PATH, None);
    let path = std::path::Path::new(FIXTURE_PATH);

    let mut file = std::fs::File::open(path).expect("open host-local lfm2 gguf fixture");
    let mut header_buf: Vec<u8> = Vec::new();
    let parsed = parse_header_region!(file, header_buf);

    let outcome = architecture_from_metadata(&parsed);
    std::println!("real_lfm2 architecture_from_metadata outcome={outcome:?}");
    let architecture = outcome.expect(
        "LFM2's real per-layer-varying head_count_kv must parse (never MissingMetadataKey)",
    );
    assert_eq!(architecture.kv_heads, 0, "the uniform view is zero when the array varies");
    assert_eq!(
        architecture.kv_heads_by_layer.len(),
        architecture.block_count as usize,
        "one head count per block"
    );
    assert!(
        architecture.kv_heads_by_layer.contains(&0)
            && architecture.kv_heads_by_layer.iter().any(|&heads| heads > 0),
        "conv layers carry zero and attention layers a positive count: {:?}",
        architecture.kv_heads_by_layer
    );
    assert!(
        matches!(
            architecture.uniform_kv_heads(),
            Err(InteropError::HeterogeneousMetadataArray { .. })
        ),
        "a uniform consumer must get the named error, never a silently-picked scalar"
    );
}

/// Same real file, isolating just the `rope.dimension_count` fallback:
/// reads `general.architecture`/`embedding_length`/`attention.head_count`
/// directly (bypassing `architecture_from_metadata`'s `head_count_kv`
/// step, which errors on this checkpoint) and confirms the derived
/// `embedding / query_heads` quotient matches this checkpoint's
/// independently-known real per-head dimension (`attn_q_norm.weight`'s
/// own declared shape is `[64]` on this file).
#[test]
#[ignore = "depends on a ~5 GB host-local lfm2 gguf checkout outside this repo"]
fn rope_dimension_count_absent_derives_the_real_lfm2_head_dim() {
    crate::test_support::require_fixture(FIXTURE_PATH, None);
    let path = std::path::Path::new(FIXTURE_PATH);

    let mut file = std::fs::File::open(path).expect("open host-local lfm2 gguf fixture");
    let mut header_buf: Vec<u8> = Vec::new();
    let parsed = parse_header_region!(file, header_buf);

    let architecture_name = metadata_str(&parsed, "general.architecture")
        .expect("general.architecture present on a real gguf");
    let embedding = metadata_u32(
        &parsed,
        &alloc::format!("{architecture_name}.embedding_length"),
    )
    .expect("embedding_length present on a real gguf");
    let query_heads = metadata_u32(
        &parsed,
        &alloc::format!("{architecture_name}.attention.head_count"),
    )
    .expect("attention.head_count present on a real gguf");
    let rope_dimension_count_key = alloc::format!("{architecture_name}.rope.dimension_count");
    let key_present = parsed.metadata_value(&rope_dimension_count_key).is_some();
    let derived_head_dim = metadata_u32_optional_or(
        &parsed,
        &rope_dimension_count_key,
        embedding / query_heads.max(1),
    );

    std::println!(
        "real_lfm2 architecture={architecture_name} embedding={embedding} query_heads={query_heads} \
         rope_dimension_count_key_present={key_present} derived_head_dim={derived_head_dim}"
    );
    assert!(
        !key_present,
        "this test's whole premise is that rope.dimension_count is ABSENT on this real checkpoint; \
         if this fails, the file changed and this test's fallback path is no longer exercised"
    );
    assert_eq!(
        derived_head_dim, 64,
        "LFM2.5-8B-A1B's real per-head dimension is 64 (independently confirmed via \
         attn_q_norm.weight's own declared [64] shape on this file)"
    );
}
