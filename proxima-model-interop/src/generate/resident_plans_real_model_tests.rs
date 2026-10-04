#![allow(clippy::expect_used)]

use omega::DispatchType;
use omega::metal::{PLAN_HANDOFF_REUSES, nocopy_cache_len};

use super::BackendRuntime;
use super::prefix_resume_long_prompt_tests::greedy_config;
use super::prompt_cache_real_model_tests::with_model;
use super::resident_plans;
use crate::LoadedModel;
use crate::serving::{PromptCacheConfig, ServingConfig};

const PROMPT: &str = "<|turn>user\nWhich of these is smaller in size: a hippopotamus or a large office building?<turn|>\n<|turn>model\n";

/// The prompt is 26 tokens and `kv_bucket_tokens` is 32: five tokens keep every
/// decode step of a generation in one bucket, so the plan one generation
/// leaves resident is an exact hit for the next.
const TOKENS: usize = 5;

struct Generation {
    ids: Vec<u32>,
    plan_hits: usize,
    plan_misses: usize,
    plan_refits: usize,
    weight_blocks_rebound: u64,
}

fn serving_config() -> ServingConfig<'static> {
    ServingConfig {
        prompt_cache: PromptCacheConfig::off(),
        ..greedy_config()
    }
}

fn generate(model: &LoadedModel<'_>, config: &ServingConfig<'_>) -> Generation {
    run(model, config, ResumeFrom::Resident)
}

#[derive(Clone, Copy)]
enum ResumeFrom {
    Resident,
    Nothing,
}

fn run(model: &LoadedModel<'_>, config: &ServingConfig<'_>, resume: ResumeFrom) -> Generation {
    let effective = model
        .effective_serving_config(config)
        .expect("the gemma4 config passes the model-dependent gates");
    let mut runtime = match resume {
        ResumeFrom::Resident => model.backend_runtime(&effective),
        ResumeFrom::Nothing => BackendRuntime::new(&effective),
    };
    let _ = PLAN_HANDOFF_REUSES.snapshot_and_reset();
    let (ids, _text, _stopped) = model
        .run_decode_loop(PROMPT, TOKENS, &effective, &mut runtime)
        .expect("greedy decode on the real gemma4-E2B checkpoint");
    Generation {
        ids,
        plan_hits: runtime.plan_hits,
        plan_misses: runtime.plan_misses,
        plan_refits: runtime.plan_refits,
        weight_blocks_rebound: PLAN_HANDOFF_REUSES.snapshot_and_reset(),
    }
}

#[test]
fn a_second_generation_on_the_same_model_builds_no_decode_plan() {
    with_model(|model| {
        let config = serving_config();

        let cold = generate(model, &config);
        let warm = generate(model, &config);
        let third = generate(model, &config);

        assert_eq!(
            (cold.plan_misses, cold.plan_refits),
            (2, 0),
            "the first generation builds a prefill plan and a decode plan"
        );
        assert_eq!(
            (warm.plan_misses, warm.plan_refits),
            (1, 0),
            "the second generation builds only the prompt-shaped prefill plan"
        );
        assert_eq!(
            warm.plan_hits,
            TOKENS - 1,
            "every decode step after the prefill reuses the resident decode plan"
        );
        assert_eq!(third.plan_misses, 1);
        assert_eq!(warm.ids, cold.ids, "a resumed plan decodes the same ids");
        assert_eq!(third.ids, cold.ids);
        assert!(
            warm.weight_blocks_rebound < cold.weight_blocks_rebound,
            "the decode plan's weight blocks are not walked again: cold {} warm {}",
            cold.weight_blocks_rebound,
            warm.weight_blocks_rebound
        );
    });
}

#[test]
fn a_runtime_built_with_new_keeps_every_plan_call_local() {
    with_model(|model| {
        let config = serving_config();

        let first = run(model, &config, ResumeFrom::Nothing);
        let second = run(model, &config, ResumeFrom::Nothing);

        assert_eq!(
            (first.plan_misses, first.plan_refits),
            (2, 0),
            "a generation builds a prefill plan and a decode plan"
        );
        assert_eq!(
            (second.plan_misses, second.plan_refits),
            (2, 0),
            "the control: with nothing resumed the next generation builds both again"
        );
        assert_eq!(
            resident_plans::resident_len(&model.plan_life),
            0,
            "a call-local runtime leaves nothing resident"
        );
    });
}

#[test]
fn weights_stay_bound_across_generations() {
    with_model(|model| {
        let config = serving_config();

        generate(model, &config);
        let wrappers_after_first = nocopy_cache_len();
        generate(model, &config);
        generate(model, &config);

        assert_eq!(
            nocopy_cache_len(),
            wrappers_after_first,
            "later generations add no resident weight wrapper"
        );
    });
}

#[test]
fn a_changed_dispatch_type_does_not_resume_the_resident_plan() {
    with_model(|model| {
        let serial = ServingConfig {
            dispatch_type: DispatchType::Serial,
            ..serving_config()
        };
        let concurrent = ServingConfig {
            dispatch_type: DispatchType::Concurrent,
            ..serving_config()
        };

        let first = generate(model, &serial);
        let resumed = generate(model, &serial);
        let switched = generate(model, &concurrent);
        let switched_back = generate(model, &serial);

        assert_eq!(resumed.plan_misses, 1, "the same config resumes");
        assert_eq!(
            switched.plan_misses, 2,
            "a plan built under another dispatch type is not served"
        );
        assert_eq!(
            switched_back.plan_misses, 2,
            "the entry now belongs to the other config, so this one rebuilds"
        );
        assert_eq!(switched.ids, first.ids);
        assert_eq!(switched_back.ids, first.ids);
    });
}

#[test]
fn dropping_the_model_frees_the_plans_it_left_resident() {
    with_model(|model| {
        generate(model, &serving_config());
        assert_eq!(
            resident_plans::resident_len(&model.plan_life),
            1,
            "one decode plan stays resident after a generation"
        );
        assert_eq!(resident_plans::entry_count(), 1);
    });

    assert_eq!(
        resident_plans::entry_count(),
        0,
        "the model's drop releases the plans this thread held for it"
    );
}
