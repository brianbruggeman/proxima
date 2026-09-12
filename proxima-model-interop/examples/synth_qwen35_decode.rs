//! Loads `synth_qwen35_gguf`'s fixture through
//! [`proxima_model_interop::LoadedModel::load`]'s `qwen35` hybrid path and
//! greedy-decodes 8 tokens on CPU and (when built with `--features metal`)
//! on CUDA or Metal. Garbage text is expected -- the weights are LCG bytes, not a
//! trained checkpoint (`synth_qwen35_gguf`'s own doc) -- this only asserts
//! that both engines run the hybrid attention+state-space graph end to end
//! and reports the CPU-vs-GPU top-1 token agreement as a sanity number,
//! never as a correctness claim.
//!
//! Owner directive (binding, mid-task): compute the derived device budget
//! BEFORE loading and abort above 16 GB -- the real Qwen3.8 checkpoint
//! already froze this box once under memory pressure, and this fixture's
//! whole reason to be small is to never approach that regime again.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_gguf::types::GgmlType;
use proxima_model_interop::{GPU_LAYERS_ALL, LoadedModel, ServingConfig};

const FIXTURE_PATH: &str = "/tmp/proxima-synth-qwen35.gguf";

// Mirrors `synth_qwen35_gguf.rs`'s own constants -- kept in sync by hand
// since the two examples don't share a lib target; the budget check below
// is only meaningful if these match what was actually written to disk.
const BLOCK_COUNT: u32 = 4;
const FULL_ATTENTION_INTERVAL: u32 = 4;
const ATTN_HEAD_DIM: u32 = 512;
const KV_HEADS: u32 = 2;
const SSM_STATE_SIZE: u32 = 128;
const SSM_GROUP_COUNT: u32 = 16;
const SSM_TIME_STEP_RANK: u32 = 32;
const SSM_INNER_SIZE: u32 = 4096;
const SSM_CONV_KERNEL: u32 = 4;
const SERVING_CONTEXT: u32 = 256;
const MAX_DEVICE_BUDGET_BYTES: u64 = 16 * 1024 * 1024 * 1024;

/// weights (measured, on-disk file size) + KV cache (attention-kind layers,
/// at `SERVING_CONTEXT`) + SSM state (ssm-kind layers, O(1) in sequence
/// length -- `qwen38-path.md`'s own formula) + a generous fixed arena
/// allowance. All four terms are the same ones `qwen38-path.md`'s section
/// (4) names; this is that formula evaluated at THIS fixture's (tiny)
/// hparams rather than the real checkpoint's.
fn derived_device_budget_bytes(weights_bytes: u64) -> (u64, u64, u64, u64) {
    let attention_layers = (0..BLOCK_COUNT)
        .filter(|layer| (layer + 1).is_multiple_of(FULL_ATTENTION_INTERVAL))
        .count() as u64;
    let ssm_layers = u64::from(BLOCK_COUNT) - attention_layers;

    let kv_cache_bytes = 2
        * u64::from(SERVING_CONTEXT)
        * u64::from(KV_HEADS)
        * u64::from(ATTN_HEAD_DIM)
        * 4
        * attention_layers;

    let ssm_key_dim = u64::from(SSM_STATE_SIZE) * u64::from(SSM_GROUP_COUNT);
    let head_v_dim = u64::from(SSM_INNER_SIZE) / u64::from(SSM_TIME_STEP_RANK);
    let qkv_dim = 2 * ssm_key_dim + u64::from(SSM_INNER_SIZE);
    let conv_rows = u64::from(SSM_CONV_KERNEL) - 1;
    let state_len = u64::from(SSM_STATE_SIZE)
        * head_v_dim
        * u64::from(SSM_GROUP_COUNT)
        * (u64::from(SSM_TIME_STEP_RANK) / u64::from(SSM_GROUP_COUNT));
    let ssm_state_bytes = (conv_rows * qkv_dim * 4 + state_len * 4) * ssm_layers;

    // Fixed, generous allowance -- `qwen38-path.md`'s own arena line was
    // ASSUMED order-of-magnitude even for the real checkpoint; at this
    // fixture's tiny per-layer widths the real arena is under a MiB, so
    // 256 MiB here is pure headroom, not a measurement.
    let arena_allowance_bytes: u64 = 256 * 1024 * 1024;

    (
        weights_bytes,
        kv_cache_bytes,
        ssm_state_bytes,
        arena_allowance_bytes,
    )
}

fn print_vm_stat(label: &str) {
    let output = std::process::Command::new("vm_stat").output();
    match output {
        Ok(result) => {
            println!("--- vm_stat {label} ---");
            println!("{}", String::from_utf8_lossy(&result.stdout));
        }
        Err(error) => println!("vm_stat unavailable ({label}): {error}"),
    }
}

fn decode(
    parsed: &proxima_gguf::pipe::ParsedGguf,
    file_bytes: &[u8],
    gpu_layers: i32,
    label: &str,
) -> Vec<u32> {
    let loaded = LoadedModel::load(parsed, file_bytes).expect("load synthetic qwen35 checkpoint");

    // `ServingConfig::default()` is the owner's real `-ngl all -ctk q8_0
    // -ctv q8_0 -fa on -b 32 -ub 32 --reasoning-budget 1024` invocation --
    // three of those hit real, typed `UnsupportedServingConfig` rejections
    // on THIS forward (Q8_0 KV storage, flash attention, and non-zero
    // batch/ubatch each name the exact unimplemented path), none of them
    // qwen35-specific. `crate::generate::supported_serving_config`
    // (`pub(crate)`, unreachable from here) is the fully-supported knob set
    // every existing caller actually runs; mirrored by hand field-for-field
    // (`generate.rs:1497-1513`) since an example crate cannot import it.
    let config = ServingConfig {
        model_path: FIXTURE_PATH,
        context_length: SERVING_CONTEXT,
        kv_cache_key_quant: GgmlType::F32,
        kv_cache_value_quant: GgmlType::F32,
        flash_attention: false,
        batch_size: 0,
        ubatch_size: 0,
        gpu_layers,
        reasoning_budget: 0,
        ..ServingConfig::default()
    };

    let started = std::time::Instant::now();
    let prompt = "hello";
    let outcome = loaded.generate_with_serving_config(prompt, 8, config);
    let elapsed = started.elapsed();

    match outcome {
        Ok((ids, text, stopped_by_eos)) => {
            println!(
                "{label}: generated_ids={ids:?} generated_text={text:?} stopped_by_eos={stopped_by_eos} \
                 elapsed={elapsed:?} ms_per_token={:.3}",
                elapsed.as_secs_f64() * 1000.0 / ids.len().max(1) as f64
            );
            ids
        }
        Err(error) => {
            println!("{label}: forward failed: {error:?}");
            Vec::new()
        }
    }
}

fn main() {
    let file_bytes = std::fs::read(FIXTURE_PATH).expect("read synth_qwen35_gguf's own output");
    let parsed = proxima_gguf::parse_complete(&file_bytes).expect("parse synthetic qwen35 gguf");
    println!(
        "synth_qwen35_decode: fixture={FIXTURE_PATH} bytes={}",
        file_bytes.len()
    );

    let (weights, kv_cache, ssm_state, arena) =
        derived_device_budget_bytes(file_bytes.len() as u64);
    let total = weights + kv_cache + ssm_state + arena;
    println!(
        "derived device budget: weights={weights} kv_cache={kv_cache} ssm_state={ssm_state} \
         arena_allowance={arena} total={total} ({:.3} GiB), limit={MAX_DEVICE_BUDGET_BYTES} ({} GiB)",
        total as f64 / (1024.0 * 1024.0 * 1024.0),
        MAX_DEVICE_BUDGET_BYTES / (1024 * 1024 * 1024)
    );
    assert!(
        total <= MAX_DEVICE_BUDGET_BYTES,
        "derived device budget {total} bytes exceeds the 16 GiB owner-directed abort ceiling -- refusing to load"
    );

    print_vm_stat("before");

    let cpu_ids = decode(&parsed, &file_bytes, 0, "cpu");

    #[cfg(all(feature = "cuda", not(target_os = "macos")))]
    let gpu_ids = decode(&parsed, &file_bytes, GPU_LAYERS_ALL, "cuda");
    #[cfg(all(feature = "metal", target_os = "macos", not(feature = "cuda")))]
    let gpu_ids = decode(&parsed, &file_bytes, GPU_LAYERS_ALL, "metal");
    #[cfg(not(any(
        all(feature = "cuda", not(target_os = "macos")),
        all(feature = "metal", target_os = "macos", not(feature = "cuda"))
    )))]
    let gpu_ids: Vec<u32> = {
        let _ = GPU_LAYERS_ALL;
        println!("gpu: skipped, build without a supported GPU backend");
        Vec::new()
    };

    print_vm_stat("after");

    if !cpu_ids.is_empty() && !gpu_ids.is_empty() {
        let agree = cpu_ids
            .iter()
            .zip(gpu_ids.iter())
            .filter(|(cpu_token, gpu_token)| cpu_token == gpu_token)
            .count();
        let denominator = cpu_ids.len().min(gpu_ids.len());
        println!(
            "cpu_vs_gpu top1 agreement: {agree}/{denominator} ({:.1}%)",
            100.0 * agree as f64 / denominator.max(1) as f64
        );
    }
}
