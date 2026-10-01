#![allow(clippy::expect_used)]

use std::sync::mpsc;
use std::time::{Duration, Instant};

use super::prefix_resume_long_prompt_tests::{chat_prompt, corpus_document};
use super::prompt_cache_real_model_tests::long_chat_prompt;
use super::prompt_cache_real_model_tests::{
    TurnOutcome, cached_config, cached_tokens_after, encode_continuation, encode_opening,
    run_cached, run_fresh, uncached_config, with_model,
};
use super::{LoadedModel, PrewarmReport};
use crate::serving::{PromptCacheConfig, ServingConfig, SpeculativeConfig};

const TURN_BOUNDARY: &str = "<turn|>\n<|turn>user\n";
const USER_FROM_CHAR: usize = 1200;
const USER_CHARS: usize = 500;
const LONG_PREFIX_CHARS: usize = 14_000;
const PREWARM_CHUNK_TOKENS: u32 = 128;

fn turn_boundary_suffix(model: &LoadedModel<'_>) -> Vec<u32> {
    encode_continuation(model, TURN_BOUNDARY)
}

struct NextTurn {
    prompt: Vec<u32>,
    user_tokens: usize,
}

/// What a client sends for turn N+1: turn N's prompt and answer, the
/// turn-boundary suffix, then the new user text and the model-turn opener.
fn next_turn(
    model: &LoadedModel<'_>,
    previous_prompt: &[u32],
    generated: &[u32],
    suffix: &[u32],
) -> NextTurn {
    let user_text: String = corpus_document("rag004")
        .chars()
        .skip(USER_FROM_CHAR)
        .take(USER_CHARS)
        .collect();
    let user_ids = encode_continuation(model, &format!("{user_text}<turn|>\n<|turn>model\n"));
    let prompt = previous_prompt
        .iter()
        .chain(generated)
        .chain(suffix)
        .chain(&user_ids)
        .copied()
        .collect();
    NextTurn {
        prompt,
        user_tokens: user_ids.len(),
    }
}

fn assert_prefilled_only_the_user_turn(
    turn: &TurnOutcome,
    next: &NextTurn,
    held_tokens: usize,
    suffix_tokens: usize,
    answer_tokens: usize,
    opening_tokens: usize,
) {
    let shared = opening_tokens + answer_tokens + suffix_tokens;
    assert_eq!(turn.report.lcp, shared, "{:?}", turn.report);
    assert_eq!(
        turn.report.prefilled_tokens, next.user_tokens,
        "only the new user tokens may be prefilled: {:?}",
        turn.report
    );
    assert_eq!(
        turn.report.prewarm_hit_tokens,
        shared - held_tokens,
        "the rows past what the answer's request held were the prewarm's: {:?}",
        turn.report
    );
}

/// AC13: with the turn-boundary suffix registered once, no call after the
/// answer ends is needed -- the next request prefills only the user's new
/// tokens and its ids equal a full prefill's. The control turn, run first
/// with no suffix registered, prefills the answer's unforwarded tail and the
/// suffix as well: the trigger is off by default.
fn end_of_answer_trigger(speculative: SpeculativeConfig<'static>) {
    let control_prefilled = with_model(|model| {
        let opening = encode_opening(model, &chat_prompt(USER_FROM_CHAR));
        let suffix = turn_boundary_suffix(model);
        let first = run_cached(model, cached_config(speculative), &opening);
        let held = cached_tokens_after(opening.len(), &first);
        let next = next_turn(model, &opening, &first.generated, &suffix);
        let control = run_cached(model, cached_config(speculative), &next.prompt);

        assert_eq!(control.report.prewarm_hit_tokens, 0);
        assert_eq!(
            control.report.prefilled_tokens,
            next.prompt.len() - held,
            "no suffix registered: nothing was prewarmed, {:?}",
            control.report
        );
        control.report.prefilled_tokens
    });
    with_model(|model| {
        let opening = encode_opening(model, &chat_prompt(USER_FROM_CHAR));
        let suffix = turn_boundary_suffix(model);
        model.set_prewarm_suffix(&suffix);
        let first = run_cached(model, cached_config(speculative), &opening);
        let held = cached_tokens_after(opening.len(), &first);
        let next = next_turn(model, &opening, &first.generated, &suffix);
        let turn = run_cached(model, cached_config(speculative), &next.prompt);
        let fresh = run_fresh(model, uncached_config(speculative), &next.prompt);

        assert_eq!(
            turn.generated, fresh,
            "ids after a prewarm must equal a full prefill"
        );
        assert_prefilled_only_the_user_turn(
            &turn,
            &next,
            held,
            suffix.len(),
            first.generated.len(),
            opening.len(),
        );
        println!(
            "AC13 held_after_answer={held} prewarmed={} user_tokens={} prefilled={} control_prefilled={control_prefilled}",
            turn.report.prewarm_hit_tokens, next.user_tokens, turn.report.prefilled_tokens,
        );
    });
}

#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn the_end_of_answer_trigger_prewarms_the_next_turn_with_speculation_off() {
    end_of_answer_trigger(SpeculativeConfig::none());
}

#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn the_end_of_answer_trigger_prewarms_the_next_turn_with_speculation_on() {
    end_of_answer_trigger(SpeculativeConfig::default());
}

/// AC11: a caller-driven `prewarm` of the next turn's prefix (answer plus
/// suffix) leaves the next request only the user's new tokens, ids identical
/// to a full prefill.
#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn a_caller_prewarm_of_the_next_prefix_leaves_only_the_user_turn_to_prefill() {
    with_model(|model| {
        let speculative = SpeculativeConfig::none();
        let opening = encode_opening(model, &chat_prompt(USER_FROM_CHAR));
        let suffix = turn_boundary_suffix(model);
        let first = run_cached(model, cached_config(speculative), &opening);
        let held = cached_tokens_after(opening.len(), &first);
        let next = next_turn(model, &opening, &first.generated, &suffix);
        let expected_prefix = opening.len() + first.generated.len() + suffix.len();

        let warmed = model
            .prewarm(&next.prompt[..expected_prefix], &cached_config(speculative))
            .expect("prewarm the next turn's prefix");
        let turn = run_cached(model, cached_config(speculative), &next.prompt);
        let fresh = run_fresh(model, uncached_config(speculative), &next.prompt);

        assert_eq!(warmed.reused_tokens, held);
        assert_eq!(warmed.prefilled_tokens, expected_prefix - held);
        assert!(!warmed.preempted);
        assert_eq!(turn.generated, fresh);
        assert_prefilled_only_the_user_turn(
            &turn,
            &next,
            held,
            suffix.len(),
            first.generated.len(),
            opening.len(),
        );
        println!("AC11 prewarm={warmed:?} turn={:?}", turn.report);
    });
}

fn chunked_config() -> ServingConfig<'static> {
    ServingConfig {
        prompt_cache: PromptCacheConfig {
            prewarm_chunk_tokens: PREWARM_CHUNK_TOKENS,
            ..cached_config(SpeculativeConfig::none()).prompt_cache
        },
        ..cached_config(SpeculativeConfig::none())
    }
}

/// AC12: a request that arrives after the prewarm's first chunk waits for the
/// chunk in flight only, then reuses the rows prefilled so far.
#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn a_request_arriving_mid_prewarm_waits_one_chunk_and_reuses_the_partial_prewarm() {
    with_model(|model| {
        let config = chunked_config();
        let prefix = encode_opening(model, &long_chat_prompt(LONG_PREFIX_CHARS));
        let chunk = PREWARM_CHUNK_TOKENS as usize;
        assert!(
            prefix.len() > 8 * chunk,
            "{} tokens is too few chunks to preempt",
            prefix.len()
        );
        let request: Vec<u32> = prefix
            .iter()
            .copied()
            .chain(encode_continuation(
                model,
                "<turn|>\n<|turn>user\nSummarize it.<turn|>\n<|turn>model\n",
            ))
            .collect();
        let (first_chunk_sender, first_chunk_receiver) = mpsc::channel::<()>();

        let (turn, warmed, boundaries): (TurnOutcome, PrewarmReport, Vec<Instant>) =
            std::thread::scope(|scope| {
                let warming = scope.spawn(|| {
                    let mut boundaries = vec![Instant::now()];
                    let report = model
                        .prewarm_with_progress(&prefix, &config, &mut |_position| {
                            boundaries.push(Instant::now());
                            if boundaries.len() == 2 {
                                first_chunk_sender
                                    .send(())
                                    .expect("the request side listens");
                            }
                        })
                        .expect("prewarm the long prefix");
                    (report, boundaries)
                });
                first_chunk_receiver
                    .recv()
                    .expect("the prewarm reached its first chunk");
                let turn = run_cached(model, config, &request);
                let (report, boundaries) = warming.join().expect("the prewarm thread finishes");
                (turn, report, boundaries)
            });
        let fresh = run_fresh(model, uncached_config(SpeculativeConfig::none()), &request);

        assert!(warmed.preempted, "{warmed:?}");
        assert!(warmed.prefilled_tokens < prefix.len(), "{warmed:?}");
        assert_eq!(turn.generated, fresh);
        assert!(turn.report.prewarm_hit_tokens > 0, "{:?}", turn.report);
        assert_eq!(
            turn.report.reused_tokens, warmed.prefilled_tokens,
            "{:?}",
            turn.report
        );
        assert!(
            turn.report.prewarm_wait <= warmed.longest_chunk,
            "waited {:?} against a longest chunk of {:?}",
            turn.report.prewarm_wait,
            warmed.longest_chunk
        );
        let chunk_ms: Vec<String> = boundaries
            .windows(2)
            .map(|pair| format!("{:.1}", millis(pair[1] - pair[0])))
            .collect();
        println!(
            "AC12 chunk_tokens={chunk} chunk_ms={chunk_ms:?} prefix_tokens={} prewarmed_tokens={} chunks={} longest_chunk_ms={:.1} request_wait_ms={:.1} hit_tokens={} prefilled={}",
            prefix.len(),
            warmed.prefilled_tokens,
            warmed.chunks,
            millis(warmed.longest_chunk),
            millis(turn.report.prewarm_wait),
            turn.report.prewarm_hit_tokens,
            turn.report.prefilled_tokens,
        );
    });
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

const MODEL_TURN_CLOSING: &str = "<turn|>\n<|turn>model\n";
const FOLLOW_UP_BRANCHES: u32 = 3;

fn follow_up_config() -> ServingConfig<'static> {
    let base = cached_config(SpeculativeConfig::none());
    ServingConfig {
        prompt_cache: PromptCacheConfig {
            follow_up_branches: FOLLOW_UP_BRANCHES,
            max_entries: 8,
            prewarm_chunk_tokens: PREWARM_CHUNK_TOKENS,
            ..base.prompt_cache
        },
        ..base
    }
}

struct AnsweredTurn {
    base: Vec<u32>,
    closing: Vec<u32>,
}

fn answered_turn(model: &LoadedModel<'_>) -> AnsweredTurn {
    let opening = encode_opening(model, &chat_prompt(USER_FROM_CHAR));
    let first = run_cached(model, cached_config(SpeculativeConfig::none()), &opening);
    let base: Vec<u32> = opening
        .iter()
        .chain(&first.generated)
        .chain(&turn_boundary_suffix(model))
        .copied()
        .collect();
    AnsweredTurn {
        base,
        closing: encode_continuation(model, MODEL_TURN_CLOSING),
    }
}

/// AC14: the model drafts follow-up user turns behind an answer; a request
/// whose user turn begins like a draft reuses the rows of the matching
/// branch past the answer, and the ids still equal a full prefill's.
#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn a_user_turn_that_begins_like_a_drafted_follow_up_reuses_the_branch_past_the_answer() {
    with_model(|model| {
        let turn = answered_turn(model);
        model.set_follow_up_closing(&turn.closing);
        let config = follow_up_config();
        model
            .prewarm(&turn.base, &config)
            .expect("prewarm the answer and its suffix");

        let drafts = model
            .prewarm_follow_ups(&turn.base, &config)
            .expect("draft follow-up user turns");

        assert!(!drafts.is_empty(), "the model drafted no follow-up");
        for (index, draft) in drafts.iter().enumerate() {
            let text = proxima_tokenizer::decode(draft, &model.vocab).expect("decode a draft");
            println!("AC14 draft {index} tokens={} text={text:?}", draft.len());
        }
        let matched = &drafts[0];
        let half = matched.len().div_ceil(2);
        let diverging_tail = encode_continuation(model, " and then what about the weather?");
        let mut request: Vec<u32> = turn.base.clone();
        request.extend_from_slice(&matched[..half]);
        request.extend_from_slice(&diverging_tail);

        let outcome = run_cached(model, cached_config(SpeculativeConfig::none()), &request);
        let fresh = run_fresh(model, uncached_config(SpeculativeConfig::none()), &request);

        assert_eq!(outcome.generated, fresh);
        assert_eq!(
            outcome.report.follow_up_hit_tokens, half,
            "the request shared {half} drafted tokens past the answer: {:?}",
            outcome.report
        );
        assert_eq!(outcome.report.reused_tokens, turn.base.len() + half);
        println!(
            "AC14 branches={} draft0_tokens={} shared={half} report={:?}",
            drafts.len(),
            matched.len(),
            outcome.report
        );
    });
}

/// AC14, trigger: with follow-up drafting on, the end-of-answer prewarm leaves
/// the answer entry and a branch per draft behind, where the same request
/// with it off leaves the answer entry alone.
#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn the_end_of_answer_trigger_leaves_a_branch_entry_per_drafted_follow_up() {
    let bytes_with = |branches: u32| {
        with_model(|model| {
            let opening = encode_opening(model, &chat_prompt(USER_FROM_CHAR));
            model.set_prewarm_suffix(&turn_boundary_suffix(model));
            model.set_follow_up_closing(&encode_continuation(model, MODEL_TURN_CLOSING));
            let mut config = follow_up_config();
            config.prompt_cache.follow_up_branches = branches;
            run_cached(model, config, &opening);
            model.prompt_cache_bytes()
        })
    };

    let answer_only = bytes_with(0);
    let with_branches = bytes_with(FOLLOW_UP_BRANCHES);

    assert!(
        with_branches > answer_only * 2,
        "{FOLLOW_UP_BRANCHES} branches left {with_branches} bytes against {answer_only} for the answer alone"
    );
    println!("AC14 trigger answer_only_bytes={answer_only} with_branches_bytes={with_branches}");
}

/// AC14, preemption: a request arriving while a later branch is drafting
/// waits for the forward in flight and the one-token restore of the answer
/// entry, less than one prewarm chunk, and reuses the answer entry whole.
#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn a_request_arriving_mid_follow_up_drafting_waits_less_than_one_prewarm_chunk() {
    with_model(|model| {
        let config = follow_up_config();
        let reference_chunk = reference_chunk_time(model, &config);
        let turn = answered_turn(model);
        model.set_follow_up_closing(&turn.closing);
        model
            .prewarm(&turn.base, &config)
            .expect("prewarm the answer and its suffix");
        let mut request = turn.base.clone();
        request.extend(encode_continuation(model, "Thanks, and why?"));

        let (first_branch_sender, first_branch_receiver) = mpsc::channel::<()>();
        let (outcome, drafts) = std::thread::scope(|scope| {
            let drafting = scope.spawn(|| {
                model
                    .prewarm_follow_ups_with_progress(&turn.base, &config, &mut |kept| {
                        if kept == 1 {
                            first_branch_sender
                                .send(())
                                .expect("the request side listens");
                        }
                    })
                    .expect("draft follow-up user turns")
            });
            first_branch_receiver
                .recv()
                .expect("the first branch was stored");
            let outcome = run_cached(model, cached_config(SpeculativeConfig::none()), &request);
            (outcome, drafting.join().expect("the drafting thread ends"))
        });
        let fresh = run_fresh(model, uncached_config(SpeculativeConfig::none()), &request);

        assert_eq!(outcome.generated, fresh);
        assert!(
            drafts.len() < FOLLOW_UP_BRANCHES as usize,
            "the request arrived after every branch was drafted"
        );
        assert_eq!(
            outcome.report.reused_tokens,
            turn.base.len(),
            "the request must find the whole answer entry: {:?}",
            outcome.report
        );
        assert!(
            outcome.report.prewarm_wait <= reference_chunk,
            "waited {:?} against a {:?} chunk",
            outcome.report.prewarm_wait,
            reference_chunk
        );
        println!(
            "AC14 preempt branches_kept={} request_wait_ms={:.1} reference_chunk_ms={:.1} report={:?}",
            drafts.len(),
            millis(outcome.report.prewarm_wait),
            millis(reference_chunk),
            outcome.report
        );
    });
}

/// The time one warm [`PREWARM_CHUNK_TOKENS`]-token chunk takes: the second
/// chunk of a prewarm over a long document. Run before the conversation is
/// cached, because a prewarm reuses the entry with the longest common prefix
/// and any entry sharing the opening tokens would be rewound into it.
fn reference_chunk_time(model: &LoadedModel<'_>, config: &ServingConfig<'_>) -> Duration {
    let chunk = PREWARM_CHUNK_TOKENS as usize;
    let document = encode_opening(model, &long_chat_prompt(LONG_PREFIX_CHARS));
    let mut boundaries = vec![Instant::now()];
    model
        .prewarm_with_progress(&document[..chunk * 3], config, &mut |_position| {
            boundaries.push(Instant::now());
        })
        .expect("prewarm a reference document");
    boundaries[2] - boundaries[1]
}
