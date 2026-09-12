//! Run a GGUF text-embedding checkpoint through the shared transformer graph.
//!
//! Usage: `embed_local <model.gguf> <cpu|cuda|vulkan> <last|mean> <text>`

#![allow(clippy::expect_used)]

use std::env;
use std::fs::File;
use std::time::Instant;

use memmap2::MmapOptions;
use proxima_gguf::parse_complete;
use proxima_model_interop::{GPU_LAYERS_ALL, LoadedModel, ModelTask, classify_task, metadata_u32};

fn usage() -> ! {
    eprintln!("usage: embed_local <model.gguf> <cpu|cuda|vulkan> <last|mean> <text>");
    std::process::exit(2)
}

fn main() {
    let mut args = env::args().skip(1);
    let path = args.next().unwrap_or_else(|| usage());
    let backend = args.next().unwrap_or_else(|| usage());
    let pooling = args.next().unwrap_or_else(|| usage());
    let text = args.next().unwrap_or_else(|| usage());
    if args.next().is_some()
        || !matches!(backend.as_str(), "cpu" | "cuda" | "vulkan")
        || !matches!(pooling.as_str(), "last" | "mean")
    {
        usage();
    }
    if backend == "cuda" && !cfg!(feature = "cuda") {
        eprintln!("embed_local: cuda mode requires --features cuda");
        std::process::exit(2);
    }
    if backend == "vulkan" && !cfg!(feature = "vulkan") {
        eprintln!("embed_local: vulkan mode requires --features vulkan");
        std::process::exit(2);
    }
    if backend != "cpu" && cfg!(all(feature = "cuda", feature = "vulkan")) {
        eprintln!("embed_local: build exactly one of cuda or vulkan");
        std::process::exit(2);
    }
    if backend == "cuda" {
        #[cfg(feature = "cuda")]
        {
            let driver = omega::CudaDriver::new(0)
                .unwrap_or_else(|error| panic!("cuda device unavailable: {error}"));
            driver.memory_info().expect("cuda memory query");
        }
    }
    if backend == "vulkan" {
        #[cfg(feature = "vulkan")]
        omega::probe_wgpu().expect("vulkan adapter");
    }

    let file = File::open(&path).expect("open GGUF checkpoint");
    // SAFETY: `file` remains alive for the mapping and `model` borrows it only
    // until this process exits.
    let mapping = unsafe { MmapOptions::new().map(&file) }.expect("map GGUF checkpoint");
    let parsed = parse_complete(&mapping).expect("parse GGUF checkpoint");
    let profile = classify_task(&parsed);
    if profile.task != ModelTask::Embedding {
        eprintln!(
            "embed_local: expected embedding task, detected {} ({:?})",
            profile.task.name(),
            profile.evidence
        );
        std::process::exit(3);
    }
    let architecture = profile
        .architecture
        .as_deref()
        .expect("embedding checkpoint architecture metadata");
    let embedding = metadata_u32(&parsed, &format!("{architecture}.embedding_length"))
        .expect("embedding length metadata");
    let model = LoadedModel::load(&parsed, &mapping).expect("bind embedding checkpoint");
    let hidden_root = model.hidden_root().expect("embedding graph hidden root");
    let compare_layers = backend != "cpu" && env::var_os("PROXIMA_COMPARE_LAYERS").is_some();
    let mut requested_nodes = vec![hidden_root];
    if compare_layers {
        requested_nodes.extend_from_slice(model.layer_residual_roots());
    }

    let started = Instant::now();
    let mut gpu_values = model
        .forward_node_values_on_backend(
            &text,
            &requested_nodes,
            if backend == "cpu" { 0 } else { GPU_LAYERS_ALL },
        )
        .expect("run embedding forward");
    let mut hidden = gpu_values.remove(0);
    let forward_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let cpu_values = if backend != "cpu" {
        Some(
            model
                .forward_node_values(&text, &requested_nodes)
                .expect("run CPU embedding reference"),
        )
    } else {
        None
    };
    let cpu_max_abs_diff = if let Some(cpu_values) = &cpu_values {
        let cpu_hidden = &cpu_values[0];
        if compare_layers {
            let mut first_bad = None;
            for (layer, (actual, expected)) in
                gpu_values.iter().zip(cpu_values.iter().skip(1)).enumerate()
            {
                let diff = actual
                    .iter()
                    .zip(expected)
                    .map(|(actual, expected)| f32::abs(*actual - *expected))
                    .fold(0.0, f32::max);
                if diff > 1.0e-3 && first_bad.is_none() {
                    first_bad = Some((layer, diff));
                }
                if env::var_os("PROXIMA_COMPARE_LAYER_ERRORS").is_some() {
                    eprintln!("embed_local: layer={layer} max_abs_diff={diff:.7}");
                }
            }
            eprintln!("embed_local: first_layer_difference={first_bad:?}");
        }
        Some(
            hidden
                .iter()
                .zip(cpu_hidden)
                .map(|(actual, expected)| f64::from((*actual - expected).abs()))
                .fold(0.0, f64::max),
        )
    } else {
        None
    };
    let width = embedding as usize;
    assert!(width != 0 && hidden.len().is_multiple_of(width));
    let rows = hidden.len() / width;
    if pooling == "mean" && rows == 1 {
        eprintln!(
            "embed_local: mean pooling requested, but the current graph exposes only the last hidden row"
        );
        std::process::exit(4);
    }
    if pooling == "last" {
        let start = hidden.len() - width;
        hidden = hidden.split_off(start);
    } else {
        let mut mean = vec![0.0f32; width];
        for row in hidden.chunks_exact(width) {
            for (output, value) in mean.iter_mut().zip(row) {
                *output += *value / rows as f32;
            }
        }
        hidden = mean;
    }
    let norm = hidden
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();
    if norm != 0.0 {
        for value in &mut hidden {
            *value /= norm as f32;
        }
    }
    println!(
        "{{\"task\":\"embedding\",\"pooling\":\"{pooling}\",\"dimensions\":{},\"forward_ms\":{forward_ms:.3},\"cpu_max_abs_diff\":{cpu_max_abs_diff:?},\"pre_normalize_l2_norm\":{norm:.7},\"first_values\":{:?}}}",
        hidden.len(),
        &hidden[..hidden.len().min(8)]
    );
}
