// workspace denies expect_used; a failed fixture must abort the test with its message
#![allow(clippy::expect_used)]

//! The recurrent tiny checkpoint's attention carries a non-unity scale (`1/sqrt(2)`) and no sink,
//! which `cached_attention_candidates` fuses into the plain `CachedAttention`. The reference
//! vendored under `fixtures/` was recorded at 92b0cf1f, where that candidate was declined and the
//! unfused reader ran; the fused plan must reproduce it.

use alloc::vec::Vec;
use core::ops::ControlFlow;
use std::sync::Arc;

use proxima_gguf::parse_complete;
use proxima_numeric::NumericFormat;
use proxima_primitives::sync::blocking::Mutex;
use proxima_telemetry::emit::{EnvFilter, global};
use proxima_telemetry::export::set_default_recorder;
use proxima_telemetry::pipes::InMemoryPipe;
use proxima_telemetry::recorder::Recorder;
use proxima_telemetry::tag::{ScalarValue, Tag};

use super::ouro_fixture::tiny_ouro_gguf;
use super::{LoadedModel, LogitsSink, NodeValuesSink, ServingConfig};
use crate::SpeculativeConfig;
use crate::serving::{GPU_LAYERS_ALL, PromptCacheConfig};

const REFERENCE: &str = include_str!("../../fixtures/recurrent_attention_unfused_reference.txt");

// the ambient recorder and level filter are process-wide; a bare blocking mutex is fine in test code
static AMBIENT_TELEMETRY: Mutex<()> = Mutex::new(());

struct Reference {
    ids: Vec<u32>,
    logits: Vec<Vec<f32>>,
}

fn reference() -> Reference {
    let mut ids = Vec::new();
    let mut logits = Vec::new();
    for line in REFERENCE.lines().filter(|line| !line.starts_with('#')) {
        if let Some(list) = line.strip_prefix("ids=[") {
            ids = list
                .trim_end_matches(']')
                .split(", ")
                .map(|id| id.parse().expect("the reference id parses"))
                .collect();
        } else if let Some((_, bits)) = line.split_once(':') {
            logits.push(
                bits.split_whitespace()
                    .map(|word| f32::from_bits(u32::from_str_radix(word, 16).expect("the reference bits parse")))
                    .collect(),
            );
        }
    }
    Reference { ids, logits }
}

fn binary32_device_config() -> ServingConfig<'static> {
    ServingConfig {
        kv_cache_key_quant: NumericFormat::Float32,
        kv_cache_value_quant: NumericFormat::Float32,
        flash_attention: false,
        batch_size: 0,
        ubatch_size: 0,
        gpu_layers: GPU_LAYERS_ALL,
        reasoning_budget: 0,
        temperature: 0.0,
        prompt_cache: PromptCacheConfig::off(),
        speculative: SpeculativeConfig::none(),
        ..ServingConfig::default()
    }
}

fn fused_attention_counts(recorder: &Recorder, pipe: &InMemoryPipe) -> Vec<u64> {
    while recorder.drain() > 0 {}
    pipe.logs()
        .iter()
        .flat_map(|log| log.attrs.iter())
        .filter_map(|tag| match tag {
            Tag::Scalar {
                key: "cached_attention_count",
                value: ScalarValue::U64(count),
            } => Some(*count),
            _ => None,
        })
        .collect()
}

#[test]
fn fused_recurrent_attention_matches_the_unfused_reference() {
    let _serialized = AMBIENT_TELEMETRY.lock();
    global::install(EnvFilter::parse("omega::metal::prepare_uniforms_pack=debug"));
    let pipe = InMemoryPipe::new();
    let recorder = Arc::new(
        Recorder::builder()
            .pipe(pipe.clone())
            .core_count(1)
            .start()
            .expect("the in-memory recorder starts"),
    );
    set_default_recorder(Arc::clone(&recorder));

    let bytes = tiny_ouro_gguf(2, false);
    let parsed = parse_complete(&bytes).expect("parse the tiny recurrent GGUF");
    let model = LoadedModel::load(&parsed, &bytes).expect("load");
    let serving = binary32_device_config();
    let mut runtime = model.backend_runtime(&serving);
    let mut logits: Vec<Vec<f32>> = Vec::new();
    let (ids, _, _, _) = model
        .run_decode_loop_through_cache(
            alloc::vec![0],
            &[],
            8,
            &serving,
            &mut runtime,
            None,
            &mut LogitsSink::Collect(&mut logits),
            &mut NodeValuesSink::Discard,
            &mut |_| ControlFlow::Continue(()),
            None,
            true,
            None,
            None,
        )
        .expect("decode eight steps on the device");

    let counts = fused_attention_counts(&recorder, &pipe);
    assert_eq!(
        counts,
        [2],
        "one plan reached the driver and it fused both logical passes' attention"
    );
    let expected = reference();
    assert_eq!(ids, expected.ids, "token ids against the unfused reference");
    assert_eq!(logits.len(), expected.logits.len(), "one logits row per step");
    for (step, (actual, reference)) in logits.iter().zip(&expected.logits).enumerate() {
        assert_eq!(actual.len(), reference.len(), "step {step} row width");
        for (index, (actual, reference)) in actual.iter().zip(reference).enumerate() {
            assert!(
                (actual - reference).abs() <= 0.000_001 + 0.000_001 * reference.abs(),
                "step {step} coordinate {index}: fused={actual} unfused_reference={reference}"
            );
        }
    }
}
