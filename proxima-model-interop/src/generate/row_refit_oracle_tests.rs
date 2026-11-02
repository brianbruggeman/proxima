use core::ops::ControlFlow;

use proxima_gguf::parse_complete;

use super::chunked_prefill_tests::dense_checkpoint;
use super::ouro_fixture::{STATE_VOCABULARY, state_sensitive_ouro_gguf};
use super::{LoadedModel, LogitsSink, NodeValuesSink, ServingConfig};
use crate::serving::{GPU_LAYERS_ALL, PromptCacheConfig};

const NEW_TOKENS: usize = 4;

const DENSE_VOCABULARY: u32 = 250;

type Outcome<Value = ()> = Result<Value, String>;

struct Decoded {
    ids: Vec<u32>,
    logits: Vec<Vec<f32>>,
    plan_misses: usize,
}

fn ouro_checkpoint() -> Vec<u8> {
    state_sensitive_ouro_gguf(2)
}

fn serving_config(gpu_layers: i32) -> ServingConfig<'static> {
    ServingConfig {
        gpu_layers,
        prompt_cache: PromptCacheConfig::off(),
        ..ServingConfig::default()
    }
}

fn prompt_of(width: usize, vocabulary: u32) -> Vec<u32> {
    (0..width).map(|index| ((index * 7 + width * 13 + 3) as u32) % vocabulary).collect()
}

fn decode(model: &LoadedModel<'_>, prompt: &[u32], gpu_layers: i32) -> Outcome<Decoded> {
    let config = serving_config(gpu_layers);
    let mut runtime = model.backend_runtime(&config);
    let mut logits = Vec::new();
    let (ids, _, _, _) = model
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
        .map_err(|err| format!("decode {prompt:?} at gpu_layers {gpu_layers}: {err:?}"))?;
    Ok(Decoded {
        ids,
        logits,
        plan_misses: runtime.plan_misses,
    })
}

fn check_matches_oracle(label: &str, got: &Decoded, oracle: &Decoded) -> Outcome {
    if got.ids != oracle.ids {
        return Err(format!("{label}: ids {:?} vs cpu oracle {:?}", got.ids, oracle.ids));
    }
    if got.logits.len() != oracle.logits.len() {
        return Err(format!("{label}: {} logit rows vs cpu oracle {}", got.logits.len(), oracle.logits.len()));
    }
    for (row, (left_row, right_row)) in got.logits.iter().zip(&oracle.logits).enumerate() {
        for (column, (left, right)) in left_row.iter().zip(right_row).enumerate() {
            if (left - right).abs() > 1e-4 {
                return Err(format!("{label}: logit row {row} column {column}: cuda {left} vs cpu oracle {right}"));
            }
        }
    }
    Ok(())
}

#[proxima::test]
#[case::ouro_three_to_five(ouro_checkpoint, STATE_VOCABULARY, 3, 5, true)]
#[case::ouro_five_to_three(ouro_checkpoint, STATE_VOCABULARY, 5, 3, true)]
#[case::ouro_two_to_seventeen(ouro_checkpoint, STATE_VOCABULARY, 2, 17, false)]
#[case::dense_three_to_five(dense_checkpoint, DENSE_VOCABULARY, 3, 5, true)]
#[case::dense_five_to_three(dense_checkpoint, DENSE_VOCABULARY, 5, 3, true)]
#[case::dense_two_to_seventeen(dense_checkpoint, DENSE_VOCABULARY, 2, 17, false)]
async fn a_plan_moved_to_another_prompt_width_matches_a_fresh_plan_and_the_cpu_oracle(
    #[case] checkpoint: fn() -> Vec<u8>,
    #[case] vocabulary: u32,
    #[case] first_width: usize,
    #[case] second_width: usize,
    #[case] same_width_class: bool,
) -> Outcome {
    let bytes = checkpoint();
    let parsed = parse_complete(&bytes).map_err(|err| format!("parse the checkpoint: {err:?}"))?;
    let prompt = prompt_of(second_width, vocabulary);
    let oracle_model = LoadedModel::load(&parsed, &bytes).map_err(|err| format!("load the oracle model: {err:?}"))?;
    let oracle = decode(&oracle_model, &prompt, 0)?;
    let fresh_model = LoadedModel::load(&parsed, &bytes).map_err(|err| format!("load a fresh model: {err:?}"))?;
    let fresh = decode(&fresh_model, &prompt, GPU_LAYERS_ALL)?;
    let reused_model = LoadedModel::load(&parsed, &bytes).map_err(|err| format!("load the reused model: {err:?}"))?;
    decode(&reused_model, &prompt_of(first_width, vocabulary), GPU_LAYERS_ALL)?;

    let moved = decode(&reused_model, &prompt, GPU_LAYERS_ALL)?;

    check_matches_oracle("fresh plan", &fresh, &oracle)?;
    check_matches_oracle("moved plan", &moved, &oracle)?;
    if same_width_class && moved.plan_misses != 0 {
        return Err(format!(
            "{first_width} -> {second_width} rows built {} plans instead of moving the resident one",
            moved.plan_misses
        ));
    }
    Ok(())
}
