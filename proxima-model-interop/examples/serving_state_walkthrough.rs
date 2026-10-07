//! The serving state machine driven through every legal transition by hand, over a stand-in
//! target function instead of a model: prefill, decode, a verify that accepts every draft, a
//! verify that rejects one, and finish. `LoadedModel`'s decode loop makes exactly these calls,
//! with a program evaluation between them. No checkpoint is read.
//!
//! Companion to `proxima-tensor/docs/the-serving-loop.md`.
//!
//! Usage:
//! `cargo run -p proxima-model-interop --example serving_state_walkthrough --features std`

#![allow(clippy::expect_used)]

use proxima_model_interop::ServingState;

type State = ServingState<u32, usize>;

fn target(token: u32) -> u32 {
    token + 1
}

fn main() {
    let prompt = vec![10_u32, 11, 12];
    let prompt_rows = prompt.len();
    let state: State = ServingState::start(prompt, 0);
    assert!(matches!(state, ServingState::Prefill { ref positions, cache: 0 } if positions.len() == 3));

    let state = state.advance_prefill(target(12), prompt_rows).expect("prefill settles into decode");
    assert_eq!(state, ServingState::Decode { last: 13, cache: 3 });

    let state = state.advance_decode(target(13), prompt_rows + 1).expect("one single-row decode step");
    assert_eq!(state, ServingState::Decode { last: 14, cache: 4 });

    let verifying = state.enter_verify(vec![15, 16]).expect("a decode state may draft");
    assert_eq!(verifying, ServingState::Verify { draft: vec![15, 16], snapshot: 4, cache: 4 });

    let accepted = verifying.accept_rows(2, &[15, 16], vec![5, 6]).expect("every drafted row matched");
    assert_eq!(accepted, ServingState::Accept { n: 2, next: 16, cache: 6 });
    let state = accepted.resume().expect("accept resumes decoding");
    assert_eq!(state, ServingState::Decode { last: 16, cache: 6 });

    let verifying = state.enter_verify(vec![17, 99]).expect("a second draft");
    let rejected = verifying.accept_rows(1, &[17, target(17)], vec![7, 8]).expect("row 1 differs from the draft");
    assert_eq!(rejected, ServingState::Rollback { snapshot: 8, to: 18 });
    let state = rejected.rollback().expect("rollback restores the cache cursor and resumes");
    assert_eq!(state, ServingState::Decode { last: 18, cache: 8 });

    let done = state.finish();
    assert_eq!(done, ServingState::Done { cache: 8 });
    let refusal = done.advance_decode(19, 9).expect_err("nothing evaluates after Done");
    println!("every legal transition walked; after Done: {refusal}");
}
