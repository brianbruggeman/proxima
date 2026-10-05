use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use super::ServingState;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Action {
    pub key: u8,
    pub delta: i64,
    pub version: u64,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct Environment {
    pub values: BTreeMap<u8, i64>,
    pub version: u64,
}

impl Environment {
    pub(crate) fn apply(&self, action: &Action) -> Self {
        let mut next = self.clone();
        *next.values.entry(action.key).or_insert(0) += action.delta;
        next.version += 1;
        next
    }
}

pub(crate) const TARGET_STEPS: usize = 12;

pub(crate) const DRAFT_ROWS: usize = 4;

pub(crate) fn true_action(environment: &Environment) -> Action {
    let key = u8::try_from(environment.version % 3).expect("a remainder below three fits a byte");
    let held = environment.values.get(&key).copied().unwrap_or(0);
    Action {
        key,
        delta: held.rem_euclid(5) + 1,
        version: environment.version,
    }
}

#[derive(Debug)]
pub(crate) struct Trace {
    pub history: Vec<Action>,
    pub environment: Environment,
    pub rounds: usize,
    pub rejected_rounds: usize,
    pub accepted_rows: usize,
}

pub(crate) fn run_sequential() -> Trace {
    let mut environment = Environment::default();
    let mut history = Vec::with_capacity(TARGET_STEPS);
    for _ in 0..TARGET_STEPS {
        let action = true_action(&environment);
        environment = environment.apply(&action);
        history.push(action);
    }
    Trace {
        history,
        environment,
        rounds: TARGET_STEPS,
        rejected_rounds: 0,
        accepted_rows: 0,
    }
}

pub(crate) fn guessed_action(environment: &Environment) -> Action {
    let mut action = true_action(environment);
    if environment.version % 4 == 3 {
        action.delta += 1;
    }
    action
}

pub(crate) fn draft_from(base: &Environment, rows: usize) -> Vec<Action> {
    let mut cursor = base.clone();
    let mut draft = Vec::with_capacity(rows);
    for _ in 0..rows {
        let guess = guessed_action(&cursor);
        cursor = cursor.apply(&guess);
        draft.push(guess);
    }
    draft
}

fn verify_rows(environment: &Environment, draft: &[Action]) -> (Vec<Action>, Vec<Environment>) {
    let mut cursor = environment.clone();
    let mut observed = Vec::with_capacity(draft.len());
    let mut caches = Vec::with_capacity(draft.len());
    for drafted in draft {
        let truth = true_action(&cursor);
        caches.push(cursor.apply(&truth));
        observed.push(truth);
        cursor = cursor.apply(drafted);
    }
    (observed, caches)
}

pub(crate) fn run_speculative(accept_policy: impl Fn(&[Action], &[Action]) -> usize) -> Trace {
    let genesis = Action { key: 0, delta: 0, version: 0 };
    let mut state = ServingState::start(Vec::new(), Environment::default())
        .advance_prefill(genesis, Environment::default())
        .expect("a fresh prefill state advances to decode");
    let mut previous = Environment::default();
    let mut history: Vec<Action> = Vec::with_capacity(TARGET_STEPS);
    let (mut rounds, mut rejected_rounds, mut accepted_rows) = (0, 0, 0);
    while history.len() < TARGET_STEPS {
        let ServingState::Decode { cache: environment, .. } = &state else {
            panic!("speculation loop expects a decode state at the top of a round");
        };
        let environment = environment.clone();
        let rows = (TARGET_STEPS - history.len()).min(DRAFT_ROWS);
        let base = if rounds % 3 == 2 { &previous } else { &environment };
        let draft = draft_from(base, rows);
        let (observed, row_caches) = verify_rows(&environment, &draft);
        let accepted = accept_policy(&draft, &observed);
        let settled = state
            .enter_verify(draft.clone())
            .expect("a decode state enters verify")
            .accept_rows(accepted, &observed, row_caches)
            .expect("the policy count is within the draft");
        state = match settled {
            ServingState::Accept { .. } => {
                history.extend(draft);
                accepted_rows += accepted;
                settled.resume().expect("an accept state resumes")
            }
            ServingState::Rollback { .. } => {
                history.extend(draft[..accepted].iter().cloned());
                history.push(observed[accepted].clone());
                accepted_rows += accepted;
                rejected_rounds += 1;
                settled.rollback().expect("a rollback state rolls back")
            }
            _ => panic!("accept_rows settles to accept or rollback only"),
        };
        previous = environment;
        rounds += 1;
    }
    let ServingState::Decode { cache: environment, .. } = state else {
        panic!("speculation loop ends in a decode state");
    };
    Trace { history, environment, rounds, rejected_rounds, accepted_rows }
}
