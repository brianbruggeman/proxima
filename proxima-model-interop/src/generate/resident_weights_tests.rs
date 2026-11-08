use core::ops::ControlFlow;

use proxima_gguf::parse_complete;
use proxima_numeric::NumericPolicy;

use super::chunked_prefill_tests::{dense_checkpoint, dense_checkpoint_seeded};
use super::{LoadedModel, LogitsSink, NodeValuesSink, ServingConfig};
use crate::serving::{GPU_LAYERS_ALL, PromptCacheConfig};

const NEW_TOKENS: usize = 3;

type Outcome<Value = ()> = Result<Value, String>;

/// The cpu oracle runs the f32 online attention the gpu binds, not llama's f16 flash form the cpu binds by default.
fn oracle_policy(gpu_layers: i32) -> NumericPolicy {
    let policy = ServingConfig::default().numeric_policy;
    if gpu_layers == 0 { policy.with_flash_narrowing(true) } else { policy }
}

fn serving_config(gpu_layers: i32) -> ServingConfig<'static> {
    ServingConfig {
        gpu_layers,
        prompt_cache: PromptCacheConfig::off(),
        numeric_policy: oracle_policy(gpu_layers),
        ..ServingConfig::default()
    }
}

fn prompt_of(width: usize, salt: usize) -> Vec<u32> {
    (0..width).map(|index| ((index * 7 + salt * 13 + 3) % 250 + 1) as u32).collect()
}

fn decode_logits(model: &LoadedModel<'_>, prompt: &[u32], gpu_layers: i32) -> Outcome<Vec<Vec<f32>>> {
    let config = serving_config(gpu_layers);
    let mut runtime = model.backend_runtime(&config);
    let mut logits = Vec::new();
    model
        .run_decode_loop_through_cache(
            prompt.to_vec(),
            &[],
            NEW_TOKENS,
            &config,
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
        .map_err(|err| format!("decode on the tiny dense checkpoint at gpu_layers {gpu_layers}: {err:?}"))?;
    Ok(logits)
}

fn check_logits_match_oracle(gpu: &[Vec<f32>], oracle: &[Vec<f32>], width: usize) -> Outcome {
    if gpu.len() != oracle.len() {
        return Err(format!("width {width}: logit row count {} vs {}", gpu.len(), oracle.len()));
    }
    for (row, (got, want)) in gpu.iter().zip(oracle).enumerate() {
        for (column, (left, right)) in got.iter().zip(want).enumerate() {
            if (left - right).abs() > 1e-4 {
                return Err(format!("width {width} logit row {row} column {column}: cuda {left} vs cpu oracle {right}"));
            }
        }
    }
    Ok(())
}

fn check(condition: bool, message: &str) -> Outcome {
    condition.then_some(()).ok_or_else(|| message.to_owned())
}

fn resident_buffer_count() -> usize {
    omega::backend::cuda_resident_buffer_count().unwrap_or(0)
}

#[proxima::test]
#[case::two_rows(2)]
#[case::five_rows(5)]
#[case::seventeen_rows(17)]
#[case::thirty_seven_rows(37)]
#[case::one_hundred_thirty_rows(130)]
async fn a_freshly_built_cuda_plan_matches_the_cpu_oracle(#[case] width: usize) -> Outcome {
    let bytes = dense_checkpoint();
    let parsed = parse_complete(&bytes).map_err(|err| format!("parse the tiny dense GGUF: {err:?}"))?;
    let prompt = prompt_of(width, width);
    let oracle_model = LoadedModel::load(&parsed, &bytes).map_err(|err| format!("load the oracle model: {err:?}"))?;
    let oracle = decode_logits(&oracle_model, &prompt, 0)?;

    for _ in 0..3 {
        let fresh_model = LoadedModel::load(&parsed, &bytes).map_err(|err| format!("load a fresh model: {err:?}"))?;
        let fresh = decode_logits(&fresh_model, &prompt, GPU_LAYERS_ALL)?;
        check_logits_match_oracle(&fresh, &oracle, width)?;
    }
    Ok(())
}

#[proxima::test(process)]
async fn a_checkpoint_loaded_where_a_dropped_one_lived_runs_its_own_weights_on_cuda() -> Outcome {
    let first = dense_checkpoint_seeded(0x9e37_79b9_7f4a_7c15);
    let second = dense_checkpoint_seeded(0xd1b5_4a32_d192_ed03);
    check(first.len() == second.len(), "both checkpoints share one tensor layout")?;
    check(first != second, "the second checkpoint carries different weight values")?;
    let prompt = prompt_of(37, 3);
    let mut storage = first;
    let before_load = resident_buffer_count();

    {
        let parsed = parse_complete(&storage).map_err(|err| format!("parse the first checkpoint: {err:?}"))?;
        let model = LoadedModel::load(&parsed, &storage).map_err(|err| format!("load the first checkpoint: {err:?}"))?;
        let warmed = decode_logits(&model, &prompt, GPU_LAYERS_ALL)?;
        check(resident_buffer_count() > before_load, "the first checkpoint left no resident weight to alias")?;
        check(warmed.len() == NEW_TOKENS, "the first checkpoint decoded the requested token count")?;
    }
    check(
        resident_buffer_count() == before_load,
        "dropping the first checkpoint evicts every resident buffer it caused",
    )?;

    storage.copy_from_slice(&second);
    let parsed = parse_complete(&storage).map_err(|err| format!("parse the second checkpoint at the first one's address: {err:?}"))?;
    let model = LoadedModel::load(&parsed, &storage).map_err(|err| format!("load the second checkpoint: {err:?}"))?;
    let gpu = decode_logits(&model, &prompt, GPU_LAYERS_ALL)?;
    let oracle = decode_logits(&model, &prompt, 0)?;

    check_logits_match_oracle(&gpu, &oracle, prompt.len())
}
