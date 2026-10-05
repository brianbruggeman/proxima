use alloc::collections::BTreeMap;
use alloc::vec::Vec;

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
}

pub(crate) fn run_sequential() -> Trace {
    let mut environment = Environment::default();
    let mut history = Vec::with_capacity(TARGET_STEPS);
    for _ in 0..TARGET_STEPS {
        let action = true_action(&environment);
        environment = environment.apply(&action);
        history.push(action);
    }
    Trace { history, environment }
}
