//! Warm speculation ON versus OFF, alternated, decode time only.
//!
//! Usage: `speculative_warm_bench <model.gguf> [tokens] [rounds]`. The model loads with the verify
//! program armed. Each prompt first runs ON and OFF twice untimed so the verify and decode plans are
//! built and the second request is the one measured. A round times one ON and one OFF request;
//! `decode_ms_per_token` runs from the first token event to the last, so prefill is excluded.

use core::ops::ControlFlow;
use std::env;
use std::fs::{self, File};
use std::time::Instant;

use memmap2::MmapOptions;
use proxima_gguf::parse_complete;
use proxima_gguf::types::GgmlType;
use proxima_model_interop::{
    GPU_LAYERS_ALL, KvLayout, LoadedModel, NgramMapParams, ServingConfig, SpeculativeConfig,
    SpeculativeDecodeStats, SpeculativeType, SpeculativeTypeSet, TokenEvent, header_descriptor,
};
use proxima_tensor::spec::ModelDescriptor;

const REPETITIVE: &str = "The capital of France is Paris. The capital of France is Paris. The capital of France is Paris. The capital of France is Paris. The capital of France is Paris. The capital of France is";
const NORMAL: &str = "Explain, in a few plain sentences, how a refrigerator keeps food cold and why the back of it feels warm.";

struct Run {
    ids: Vec<u32>,
    decode_ms_per_token: f64,
    stats: SpeculativeDecodeStats,
}

fn config(speculative: SpeculativeConfig<'static>) -> ServingConfig<'static> {
    ServingConfig {
        gpu_layers: GPU_LAYERS_ALL,
        kv_cache_key_quant: GgmlType::F32,
        kv_cache_value_quant: GgmlType::F32,
        flash_attention: false,
        batch_size: 0,
        ubatch_size: 0,
        reasoning_budget: 0,
        temperature: 0.0,
        repeat_penalty: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        speculative,
        ..ServingConfig::default()
    }
}

fn speculation_on() -> SpeculativeConfig<'static> {
    let defaults = SpeculativeConfig::none();
    SpeculativeConfig {
        speculative_types: SpeculativeTypeSet::single(SpeculativeType::NgramSimple),
        ngram_simple: NgramMapParams {
            size_n: 4,
            size_m: 15,
            ..defaults.ngram_simple
        },
        ..defaults
    }
}

fn run(model: &LoadedModel<'_>, prompt: &str, tokens: usize, on: bool) -> Run {
    let speculative = if on {
        speculation_on()
    } else {
        SpeculativeConfig::none()
    };
    let mut stats = SpeculativeDecodeStats::default();
    let mut first: Option<Instant> = None;
    let mut last = Instant::now();
    let mut on_token = |_event: TokenEvent<'_>| {
        last = Instant::now();
        first.get_or_insert(last);
        ControlFlow::Continue(())
    };
    let (ids, _text, _eos) = model
        .generate_streaming_with_speculative_stats(
            prompt,
            tokens,
            config(speculative),
            &mut on_token,
            &mut stats,
            None,
        )
        .unwrap_or_else(|error| panic!("decode: {error}"));
    println!("run on={on} stats_verify={} head={:?}", stats.verify_steps, &ids[..ids.len().min(6)]);
    let decode_ms = first.map_or(0.0, |started| {
        last.duration_since(started).as_secs_f64() * 1000.0
    });
    Run {
        decode_ms_per_token: decode_ms / ids.len().saturating_sub(1).max(1) as f64,
        ids,
        stats,
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn load_average() -> String {
    fs::read_to_string("/proc/loadavg")
        .map(|text| text.split_whitespace().next().unwrap_or("?").to_owned())
        .unwrap_or_default()
}

fn bench(model: &LoadedModel<'_>, label: &str, prompt: &str, tokens: usize, rounds: usize) {
    for _ in 0..2 {
        run(model, prompt, tokens, true);
        run(model, prompt, tokens, false);
    }
    let (mut on_ms, mut off_ms) = (Vec::new(), Vec::new());
    let mut last_on = None;
    for round in 0..rounds {
        let on = run(model, prompt, tokens, true);
        let off = run(model, prompt, tokens, false);
        println!(
            "round label={label} round={round} load={} on_ms_per_token={:.3} off_ms_per_token={:.3} identical={}",
            load_average(),
            on.decode_ms_per_token,
            off.decode_ms_per_token,
            on.ids == off.ids
        );
        if on.ids != off.ids {
            let split = on.ids.iter().zip(&off.ids).position(|(left, right)| left != right);
            println!("divergence label={label} round={round} first={split:?} verify_steps={} accepted={} on={:?} off={:?}", on.stats.verify_steps, on.stats.accepted_total, on.ids, off.ids);
        }
        on_ms.push(on.decode_ms_per_token);
        off_ms.push(off.decode_ms_per_token);
        last_on = Some(on);
    }
    let stats = last_on.map(|run| run.stats).unwrap_or_default();
    println!(
        "summary label={label} on_median={:.3} off_median={:.3} verify_steps={} drafted={} accepted={}",
        median(&mut on_ms),
        median(&mut off_ms),
        stats.verify_steps,
        stats.drafted_total,
        stats.accepted_total
    );
}

fn main() {
    let mut args = env::args().skip(1);
    let path = args
        .next()
        .unwrap_or_else(|| panic!("usage: speculative_warm_bench <model.gguf> [tokens] [rounds]"));
    let tokens: usize = args.next().and_then(|value| value.parse().ok()).unwrap_or(96);
    let rounds: usize = args.next().and_then(|value| value.parse().ok()).unwrap_or(5);
    let file = File::open(&path).unwrap_or_else(|error| panic!("open {path}: {error}"));
    // SAFETY: the file stays open and is only read for the life of the mapping.
    let bytes = unsafe { MmapOptions::new().map(&file) }
        .unwrap_or_else(|error| panic!("map {path}: {error}"));
    let parsed = parse_complete(&bytes).unwrap_or_else(|error| panic!("parse {path}: {error}"));
    let decode = header_descriptor(&parsed, KvLayout::Full)
        .unwrap_or_else(|error| panic!("header {path}: {error}"));
    let armed = ModelDescriptor {
        speculative_verify: true,
        ..decode
    };
    let model = LoadedModel::load_with_descriptor(&parsed, &bytes, &armed)
        .unwrap_or_else(|error| panic!("load {path}: {error}"));
    let only = args.next();
    if only.as_deref() != Some("normal") {
        bench(&model, "repetitive", REPETITIVE, tokens, rounds);
    }
    if only.as_deref() != Some("repetitive") {
        bench(&model, "normal", NORMAL, tokens, rounds);
    }
}
