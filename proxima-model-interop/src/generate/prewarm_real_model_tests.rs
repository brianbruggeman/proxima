#![allow(clippy::expect_used)]

use std::sync::mpsc;
use std::time::{Duration, Instant};

use super::prefix_resume_long_prompt_tests::{chat_prompt, corpus_document};
use super::prompt_cache_real_model_tests::long_chat_prompt;
use super::prompt_cache_real_model_tests::{
    TurnOutcome, cached_config, cached_tokens_after, chat_prompt_of, encode_continuation,
    encode_opening, run_cached, run_fresh, uncached_config, with_model,
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

/// With the turn-boundary suffix registered once, no call after the
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
        model
            .run_pending_prewarm(&cached_config(speculative))
            .expect("the queued prewarm runs")
            .expect("the answer queued its next prefix");
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
            "held_after_answer={held} prewarmed={} user_tokens={} prefilled={} control_prefilled={control_prefilled}",
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

/// A caller-driven `prewarm` of the next turn's prefix (answer plus
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
        println!("prewarm={warmed:?} turn={:?}", turn.report);
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

/// A request that arrives after the prewarm's first chunk waits for the
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
            "chunk_tokens={chunk} chunk_ms={chunk_ms:?} prefix_tokens={} prewarmed_tokens={} chunks={} longest_chunk_ms={:.1} request_wait_ms={:.1} hit_tokens={} prefilled={}",
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

/// The model drafts follow-up user turns behind an answer; a request
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
            println!("draft {index} tokens={} text={text:?}", draft.len());
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
            "branches={} draft0_tokens={} shared={half} report={:?}",
            drafts.len(),
            matched.len(),
            outcome.report
        );
    });
}

/// Trigger: with follow-up drafting on, the end-of-answer prewarm leaves
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
            model
                .run_pending_prewarm(&config)
                .expect("the queued prewarm runs");
            model.prompt_cache_bytes()
        })
    };

    let answer_only = bytes_with(0);
    let with_branches = bytes_with(FOLLOW_UP_BRANCHES);

    assert!(
        with_branches > answer_only * 2,
        "{FOLLOW_UP_BRANCHES} branches left {with_branches} bytes against {answer_only} for the answer alone"
    );
    println!("trigger answer_only_bytes={answer_only} with_branches_bytes={with_branches}");
}

/// Preemption: a request arriving while a later branch is drafting
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
            "preempt branches_kept={} request_wait_ms={:.1} reference_chunk_ms={:.1} report={:?}",
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

/// The answer's call returns with the end-of-answer prefill still queued: the
/// rows it will add are not in the cache until something runs the queue, and
/// running it prefills the suffix the registered boundary added.
#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn an_answer_returns_with_its_end_of_answer_prefill_still_queued() {
    with_model(|model| {
        let config = cached_config(SpeculativeConfig::none());
        let opening = encode_opening(model, &chat_prompt(USER_FROM_CHAR));
        let suffix = turn_boundary_suffix(model);
        model.set_prewarm_suffix(&suffix);

        let first = run_cached(model, config, &opening);
        let held = cached_tokens_after(opening.len(), &first);
        let bytes_when_the_answer_returned = model.prompt_cache_bytes();
        let warmed = model
            .run_pending_prewarm(&config)
            .expect("the queued prewarm runs")
            .expect("the answer queued its next prefix");
        let nothing_left = model
            .run_pending_prewarm(&config)
            .expect("polling an empty queue is not an error");

        assert_eq!(warmed.reused_tokens, held);
        assert_eq!(
            warmed.prefilled_tokens,
            first.generated.len() + opening.len() + suffix.len() - held
        );
        assert!(
            model.prompt_cache_bytes() > bytes_when_the_answer_returned,
            "the prewarm added rows after the answer had returned"
        );
        assert!(nothing_left.is_none());
        println!(
            "WORKER answer_returned_with held={held} queued_prefill={}",
            warmed.prefilled_tokens
        );
    });
}

/// With a worker attached, no call after the answer is needed beyond waiting
/// for the user to type: the next turn prefills only the user's tokens and its
/// ids equal a full prefill's.
#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn a_worker_prewarms_the_answer_so_the_next_turn_prefills_only_the_user_tokens() {
    with_model(|model| {
        let config = cached_config(SpeculativeConfig::none());
        let opening = encode_opening(model, &chat_prompt(USER_FROM_CHAR));
        let suffix = turn_boundary_suffix(model);
        model.set_prewarm_suffix(&suffix);

        let (first, next, turn) = model
            .with_prewarm_worker(&config, || {
                let first = run_cached(model, config, &opening);
                model.wait_for_prewarm();
                let next = next_turn(model, &opening, &first.generated, &suffix);
                let turn = run_cached(model, config, &next.prompt);
                (first, next, turn)
            })
            .expect("start the prewarm worker");
        let fresh = run_fresh(
            model,
            uncached_config(SpeculativeConfig::none()),
            &next.prompt,
        );

        assert_eq!(turn.generated, fresh);
        assert_prefilled_only_the_user_turn(
            &turn,
            &next,
            cached_tokens_after(opening.len(), &first),
            suffix.len(),
            first.generated.len(),
            opening.len(),
        );
    });
}

const CONCURRENT_TURNS: usize = 3;

/// One conversation's turns, each prompt the previous one plus its answer, the
/// boundary suffix and a new user excerpt of `document_id`; returns every
/// prompt with the ids the cache produced for it.
fn converse_through_the_cache(
    model: &LoadedModel<'_>,
    config: ServingConfig<'static>,
    document_id: &str,
    suffix: &[u32],
) -> Vec<(Vec<u32>, TurnOutcome)> {
    let document = corpus_document(document_id);
    let mut prompt = encode_opening(model, &chat_prompt_of(document_id, USER_FROM_CHAR));
    let mut turns = Vec::new();
    for turn in 0..CONCURRENT_TURNS {
        let outcome = run_cached(model, config, &prompt);
        let mut following = prompt.clone();
        following.extend_from_slice(&outcome.generated);
        following.extend_from_slice(suffix);
        let excerpt: String = document
            .chars()
            .skip(USER_FROM_CHAR + turn * USER_CHARS)
            .take(USER_CHARS)
            .collect();
        following.extend(encode_continuation(
            model,
            &format!("{excerpt}<turn|>\n<|turn>model\n"),
        ));
        turns.push((prompt, outcome));
        prompt = following;
    }
    turns
}

/// Two conversations answering on two threads while the worker prewarms behind
/// both: the gate serializes who holds the device and an entry is out of the
/// cache while anyone uses it, so no turn may see another's rows. Every turn
/// of both conversations must generate what a full prefill generates.
#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn two_requests_and_a_background_prewarm_never_share_an_entry() {
    with_model(|model| {
        let config = cached_config(SpeculativeConfig::none());
        let suffix = turn_boundary_suffix(model);
        model.set_prewarm_suffix(&suffix);

        let (first, second) = model
            .with_prewarm_worker(&config, || {
                std::thread::scope(|scope| {
                    let first = scope
                        .spawn(|| converse_through_the_cache(model, config, "rag004", &suffix));
                    let second = scope
                        .spawn(|| converse_through_the_cache(model, config, "rag011", &suffix));
                    (
                        first.join().expect("conversation one finishes"),
                        second.join().expect("conversation two finishes"),
                    )
                })
            })
            .expect("start the prewarm worker");

        let mut compared = 0;
        for (prompt, outcome) in first.iter().chain(&second) {
            let fresh = run_fresh(model, uncached_config(SpeculativeConfig::none()), prompt);
            assert_eq!(
                outcome.generated,
                fresh,
                "a turn of {} tokens diverged from a full prefill: {:?}",
                prompt.len(),
                outcome.report
            );
            compared += 1;
        }
        assert_eq!(compared, 2 * CONCURRENT_TURNS);
        let reused: Vec<usize> = first
            .iter()
            .chain(&second)
            .map(|(_, outcome)| outcome.report.reused_tokens)
            .collect();
        println!("WORKER concurrent reused_per_turn={reused:?}");
    });
}

/// A gemma4 answer that finishes comes back with its end-of-turn token
/// repeated until the end-of-sequence token; the client's next prompt carries
/// that token once. The prewarm must prefill the client's boundary, not one
/// with the marker doubled.
#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn an_answer_that_ended_on_end_of_turn_tokens_is_prewarmed_for_the_clients_prompt() {
    with_model(|model| {
        const END_OF_TURN: u32 = 106;
        let config = cached_config(SpeculativeConfig::none());
        let prompt = encode_opening(
            model,
            "<|turn>user\nReply with only the word yes.<turn|>\n<|turn>model\n",
        );
        let suffix = turn_boundary_suffix(model);
        model.set_prewarm_suffix(&suffix);

        let first = run_cached(model, config, &prompt);
        let warmed = model
            .run_pending_prewarm(&config)
            .expect("the queued prewarm runs")
            .expect("the answer queued its next prefix");
        let answer_text: Vec<u32> = first
            .generated
            .iter()
            .copied()
            .filter(|id| *id != END_OF_TURN)
            .collect();
        let user_ids =
            encode_continuation(model, "And without the word no?<turn|>\n<|turn>model\n");
        let next: Vec<u32> = prompt
            .iter()
            .chain(&answer_text)
            .chain(&suffix)
            .chain(&user_ids)
            .copied()
            .collect();
        let turn = run_cached(model, config, &next);
        let fresh = run_fresh(model, uncached_config(SpeculativeConfig::none()), &next);

        assert!(first.stopped_by_eos, "the answer must finish on its own");
        assert!(
            first
                .generated
                .iter()
                .filter(|id| **id == END_OF_TURN)
                .count()
                > 1,
            "{:?}",
            first.generated
        );
        assert_eq!(turn.generated, fresh);
        assert_eq!(
            turn.report.prefilled_tokens,
            user_ids.len(),
            "{:?}",
            turn.report
        );
        assert!(turn.report.prewarm_hit_tokens > 0, "{:?}", turn.report);
        println!("EOS_ANSWER warmed={warmed:?} turn={:?}", turn.report);
    });
}
