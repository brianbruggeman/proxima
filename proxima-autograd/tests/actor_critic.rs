//! Actor-critic on one `proxima_tensor` program with two loss nodes, proving
//! [`differentiate`] needs no stop-gradient primitive for the
//! two-parameter-set / two-optimizer shape `ai_lab`'s `burn` replacement
//! actually uses.
//!
//! # Why there is no `Op::Detach`
//!
//! Policy loss is `-log pi(a|s) * advantage`, `advantage = reward - V(s)`.
//! `differentiate(program, loss)` (`proxima-autograd/src/adjoint.rs:186`) is
//! already `&[Op], NodeId -> Differentiated` -- graph in, graph out -- so a
//! second loss node on the SAME program is just a second call, not a second
//! primitive: `differentiate(&program, policy_loss)` and
//! `differentiate(&program, value_loss)` each walk backward from their own
//! node and each return their own [`Differentiated`], with their own
//! `gradient_of_named` lookup table.
//!
//! Working the calculus first (per this session's design correction) rather
//! than reaching for a marker op: `d(policy_loss)/d(theta_policy) =
//! -d(log pi)/d(theta_policy) * advantage`, because `advantage` does not
//! depend on `theta_policy` at all -- it is a disjoint parameter set, so the
//! product rule's other term is exactly zero and the policy gradient is
//! correct with no severing of any edge. The contamination `.detach()`
//! prevents in an eager framework IS present here too: backprop from
//! `policy_loss` genuinely reaches `V`'s parameters (`advantage` reads `V(s)`
//! as its baseline), so `differentiate(&program,
//! policy_loss).gradient_of_named("w1v")` is `Some` and nonzero --
//! [`policy_loss_gradient_reaches_the_value_net_but_is_never_applied_to_it`]
//! proves it. What makes that harmless is composition, not erasure: the
//! value net is stepped from `differentiate(&program,
//! value_loss).gradient_of_named("w1v")` (a DIFFERENT, disjoint
//! [`Differentiated`] that never even contains a policy node, since
//! `value_loss`'s own index precedes every policy op in program order), and
//! that quantity is the only one ever handed to the value net's
//! [`proxima_autograd::optimizer::AdamStep`]. An eager tape has one
//! `backward()` call per step and therefore one gradient slot per
//! parameter, which is why it needs a marker to prevent contamination --
//! two independent `differentiate` calls each produce their own slot, so the
//! discipline that matters is which `Differentiated` a caller reads a
//! parameter's gradient FROM, never a graph edge to cut.
//!
//! This is not a claim that stop-gradient is never needed: a single
//! objective differentiated ONCE (`policy_loss + c * value_loss`, or a
//! bootstrapped TD target reading the same network's own output as a
//! constant within the very loss it feeds) would need exactly that -- this
//! environment's return is the real, single-step, non-bootstrapped reward,
//! so that case does not arise here, and no such primitive is added.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::too_many_arguments)]

extern crate alloc;

use proxima_autograd::activation::{relu, softmax};
use proxima_autograd::adjoint::{Differentiated, differentiate};
use proxima_autograd::optimizer::{AdamConfig, AdamOperands, adam_step, step_input};
use proxima_tensor::cpu::{Evaluated, evaluate_named};
use proxima_tensor::dtype::DType;
use proxima_tensor::map::{self, IndexMap};
use proxima_tensor::op::{self, Extent, NodeId, Op, ReduceInit, ScalarOp};

const STATE_DIM: usize = 3;
const ACTION_DIM: usize = 3;
const HIDDEN_DIM: usize = 6;

fn leaf(program: &mut Vec<Op>, name: &str, shape: Vec<Extent>) -> NodeId {
    op::append(
        program,
        Op::Input {
            dtype: DType::Float32,
            shape,
            name: Some(name.into()),
        },
    )
}

fn scalar_leaf(program: &mut Vec<Op>, name: &str) -> NodeId {
    leaf(program, name, Vec::new())
}

fn constant(program: &mut Vec<Op>, value: f32) -> NodeId {
    op::append(
        program,
        Op::Constant {
            dtype: DType::Float32,
            shape: Vec::new(),
            value,
        },
    )
}

fn elementwise(program: &mut Vec<Op>, body: ScalarOp, operands: Vec<(NodeId, IndexMap)>) -> NodeId {
    op::append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body,
            operands,
            name: None,
        },
    )
}

fn reduce_add(
    program: &mut Vec<Op>,
    operand: NodeId,
    in_map: IndexMap,
    out_map: IndexMap,
) -> NodeId {
    op::append(
        program,
        Op::Reduce(proxima_tensor::op::Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand,
            in_map,
            out_map,
            keep: proxima_tensor::op::Keep::Reduce,
            name: None,
        }),
    )
}

fn identity(rank: u16) -> IndexMap {
    IndexMap::Affine(map::projection(rank, &(0..rank).collect::<Vec<u16>>()))
}

fn broadcast(rank: u16) -> IndexMap {
    IndexMap::Affine(map::projection(rank, &[]))
}

/// `x @ w + b`, the same shape `proxima-autograd/tests/training_loop.rs`'s
/// own `dense` helper builds (matmul via `Elementwise(Multiply)` then
/// `Reduce(Add)`, plus a bias add).
fn dense(program: &mut Vec<Op>, x: NodeId, w: NodeId, b: NodeId) -> NodeId {
    let product = elementwise(
        program,
        ScalarOp::Multiply,
        alloc::vec![
            (w, identity(2)),
            (x, IndexMap::Affine(map::projection(2, &[0])))
        ],
    );
    let matmul = reduce_add(
        program,
        product,
        identity(2),
        IndexMap::Affine(map::projection(2, &[1])),
    );
    elementwise(
        program,
        ScalarOp::Add,
        alloc::vec![(matmul, identity(1)), (b, identity(1))],
    )
}

fn counter_pattern(seed: usize, count: usize) -> Vec<f32> {
    (0..count)
        .map(|index| (((seed + index) * 7 % 13) as f32 - 6.0) / 12.0)
        .collect()
}

fn one_hot(index: usize, dim: usize) -> Vec<f32> {
    let mut vector = alloc::vec![0.0f32; dim];
    vector[index] = 1.0;
    vector
}

/// The known-optimal action for state `s` is always `s` itself: a
/// deterministic identity-recognition contextual bandit, so "did the policy
/// converge" has an unambiguous, checkable answer per state.
fn reward_for(state: usize, action: usize) -> f32 {
    if action == state { 1.0 } else { -1.0 }
}

/// One forward+backward+loss program holding BOTH subgraphs -- the shape
/// this file's own module doc argues needs no `Op::Detach`: `policy_loss`
/// and `value_loss` are two taps on one program, differentiated
/// independently.
struct ActorCritic {
    program: Vec<Op>,
    probabilities: NodeId,
    value: NodeId,
    policy_loss: NodeId,
    value_loss: NodeId,
    policy_param_names: [&'static str; 4],
    policy_param_nodes: [NodeId; 4],
    policy_param_shapes: [Vec<Extent>; 4],
    value_param_names: [&'static str; 4],
    value_param_nodes: [NodeId; 4],
    value_param_shapes: [Vec<Extent>; 4],
}

/// `Policy`: `x -> dense -> relu -> dense -> softmax` over `ACTION_DIM`
/// discrete actions. `ValueNet`: `x -> dense -> relu -> dense -> scalar`.
/// Built value-net-first so `value_loss`'s own index precedes every policy
/// op -- `differentiate(&program, value_loss)` therefore truncates the
/// program before the policy net even exists, so there is no policy
/// gradient to accidentally read out of it (see
/// [`value_loss_gradient_never_reaches_policy_parameters`]).
fn build_actor_critic() -> ActorCritic {
    let mut program = Vec::new();
    let state = leaf(
        &mut program,
        "x",
        alloc::vec![Extent::Static(STATE_DIM as u32)],
    );

    let w1v = leaf(
        &mut program,
        "w1v",
        alloc::vec![
            Extent::Static(STATE_DIM as u32),
            Extent::Static(HIDDEN_DIM as u32)
        ],
    );
    let b1v = leaf(
        &mut program,
        "b1v",
        alloc::vec![Extent::Static(HIDDEN_DIM as u32)],
    );
    let w2v = leaf(
        &mut program,
        "w2v",
        alloc::vec![Extent::Static(HIDDEN_DIM as u32), Extent::Static(1)],
    );
    let b2v = leaf(&mut program, "b2v", alloc::vec![Extent::Static(1)]);
    let hidden_v_pre = dense(&mut program, state, w1v, b1v);
    let hidden_v = relu(&mut program, DType::Float32, hidden_v_pre, 1);
    let value_vec = dense(&mut program, hidden_v, w2v, b2v);
    let value = reduce_add(
        &mut program,
        value_vec,
        identity(1),
        IndexMap::Affine(map::projection(1, &[])),
    );

    let reward = scalar_leaf(&mut program, "reward");
    let diff = elementwise(
        &mut program,
        ScalarOp::Subtract,
        alloc::vec![(value, identity(0)), (reward, identity(0))],
    );
    let value_loss = elementwise(
        &mut program,
        ScalarOp::Multiply,
        alloc::vec![(diff, identity(0)), (diff, identity(0))],
    );

    let w1p = leaf(
        &mut program,
        "w1p",
        alloc::vec![
            Extent::Static(STATE_DIM as u32),
            Extent::Static(HIDDEN_DIM as u32)
        ],
    );
    let b1p = leaf(
        &mut program,
        "b1p",
        alloc::vec![Extent::Static(HIDDEN_DIM as u32)],
    );
    let w2p = leaf(
        &mut program,
        "w2p",
        alloc::vec![
            Extent::Static(HIDDEN_DIM as u32),
            Extent::Static(ACTION_DIM as u32)
        ],
    );
    let b2p = leaf(
        &mut program,
        "b2p",
        alloc::vec![Extent::Static(ACTION_DIM as u32)],
    );
    let hidden_p_pre = dense(&mut program, state, w1p, b1p);
    let hidden_p = relu(&mut program, DType::Float32, hidden_p_pre, 1);
    let logits = dense(&mut program, hidden_p, w2p, b2p);
    let probabilities = softmax(&mut program, DType::Float32, logits, 1, 0);

    let action_one_hot = leaf(
        &mut program,
        "action_one_hot",
        alloc::vec![Extent::Static(ACTION_DIM as u32)],
    );
    // `+ 1e-7` before the log: once the policy sharpens, an unchosen action's
    // probability can round to exactly 0.0 and `0.0 * log(0.0)` is NaN -- the
    // exact underflow `f7cab09` hit in the language-model milestone.
    let log_epsilon = constant(&mut program, 1e-7);
    let stabilized = elementwise(
        &mut program,
        ScalarOp::Add,
        alloc::vec![(probabilities, identity(1)), (log_epsilon, broadcast(1))],
    );
    let log_probabilities = elementwise(
        &mut program,
        ScalarOp::Logarithm,
        alloc::vec![(stabilized, identity(1))],
    );
    let weighted = elementwise(
        &mut program,
        ScalarOp::Multiply,
        alloc::vec![
            (action_one_hot, identity(1)),
            (log_probabilities, identity(1))
        ],
    );
    let log_pi_a = reduce_add(
        &mut program,
        weighted,
        identity(1),
        IndexMap::Affine(map::projection(1, &[])),
    );

    let advantage = elementwise(
        &mut program,
        ScalarOp::Subtract,
        alloc::vec![(reward, identity(0)), (value, identity(0))],
    );
    let product = elementwise(
        &mut program,
        ScalarOp::Multiply,
        alloc::vec![(log_pi_a, identity(0)), (advantage, identity(0))],
    );
    let policy_loss = elementwise(
        &mut program,
        ScalarOp::Negate,
        alloc::vec![(product, identity(0))],
    );

    ActorCritic {
        program,
        probabilities,
        value,
        policy_loss,
        value_loss,
        policy_param_names: ["w1p", "b1p", "w2p", "b2p"],
        policy_param_nodes: [w1p, b1p, w2p, b2p],
        policy_param_shapes: [
            alloc::vec![
                Extent::Static(STATE_DIM as u32),
                Extent::Static(HIDDEN_DIM as u32)
            ],
            alloc::vec![Extent::Static(HIDDEN_DIM as u32)],
            alloc::vec![
                Extent::Static(HIDDEN_DIM as u32),
                Extent::Static(ACTION_DIM as u32)
            ],
            alloc::vec![Extent::Static(ACTION_DIM as u32)],
        ],
        value_param_names: ["w1v", "b1v", "w2v", "b2v"],
        value_param_nodes: [w1v, b1v, w2v, b2v],
        value_param_shapes: [
            alloc::vec![
                Extent::Static(STATE_DIM as u32),
                Extent::Static(HIDDEN_DIM as u32)
            ],
            alloc::vec![Extent::Static(HIDDEN_DIM as u32)],
            alloc::vec![Extent::Static(HIDDEN_DIM as u32), Extent::Static(1)],
            alloc::vec![Extent::Static(1)],
        ],
    }
}

/// One parameter set's Adam wiring, appended onto `program` and sourcing
/// every gradient from `gradient_source` -- the one place a caller chooses
/// WHICH `Differentiated` a parameter is updated from, which is the entire
/// discipline this file's module doc describes in place of `.detach()`.
struct AdamWiring {
    step_name: &'static str,
    m_names: [alloc::string::String; 4],
    v_names: [alloc::string::String; 4],
    new_param: [NodeId; 4],
    new_m: [NodeId; 4],
    new_v: [NodeId; 4],
}

fn wire_adam(
    program: &mut Vec<Op>,
    config: &AdamConfig,
    step_name: &'static str,
    param_names: [&'static str; 4],
    param_nodes: [NodeId; 4],
    param_shapes: &[Vec<Extent>; 4],
    gradient_source: &Differentiated,
) -> AdamWiring {
    let step = step_input(program, step_name);
    let mut m_names: [alloc::string::String; 4] =
        core::array::from_fn(|_| alloc::string::String::new());
    let mut v_names: [alloc::string::String; 4] =
        core::array::from_fn(|_| alloc::string::String::new());
    let mut new_param = [NodeId(0); 4];
    let mut new_m = [NodeId(0); 4];
    let mut new_v = [NodeId(0); 4];

    for index in 0..4 {
        let name = param_names[index];
        let rank = param_shapes[index].len() as u16;
        let m_name = alloc::format!("m_{name}");
        let v_name = alloc::format!("v_{name}");
        let m_node = leaf(program, &m_name, param_shapes[index].clone());
        let v_node = leaf(program, &v_name, param_shapes[index].clone());
        let grad = gradient_source.gradient_of_named(name).unwrap_or_else(|| {
            panic!(
                "{name} must feed the loss node {:?} differentiate was called on",
                gradient_source.loss
            )
        });
        let (updated_param, updated_m, updated_v) = adam_step(
            program,
            config,
            rank,
            AdamOperands {
                param: param_nodes[index],
                grad,
                m: m_node,
                v: v_node,
            },
            step,
        );
        m_names[index] = m_name;
        v_names[index] = v_name;
        new_param[index] = updated_param;
        new_m[index] = updated_m;
        new_v[index] = updated_v;
    }

    AdamWiring {
        step_name,
        m_names,
        v_names,
        new_param,
        new_m,
        new_v,
    }
}

/// Deterministic splitmix64 -- no external RNG dependency, reproducible
/// action sampling across runs. Not cryptographic, not meant to be: the
/// only property needed is a fixed, portable stream from a fixed seed.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut mixed = self.0;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        mixed ^ (mixed >> 31)
    }

    fn next_unit_f32(&mut self) -> f32 {
        (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32
    }
}

fn sample_action(probabilities: &[f32], rng: &mut SplitMix64) -> usize {
    let threshold = rng.next_unit_f32();
    let mut cumulative = 0.0f32;
    for (action, probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if threshold < cumulative {
            return action;
        }
    }
    probabilities.len() - 1
}

struct NetState {
    params: [Vec<f32>; 4],
    m: [Vec<f32>; 4],
    v: [Vec<f32>; 4],
    step: u32,
}

fn shape_size(shape: &[Extent]) -> usize {
    shape
        .iter()
        .map(|extent| match extent {
            Extent::Static(size) => *size as usize,
            Extent::Symbolic(_) => 1,
        })
        .product::<usize>()
        .max(1)
}

fn init_net_state(param_shapes: &[Vec<Extent>; 4], seed_base: usize) -> NetState {
    let sizes: [usize; 4] = core::array::from_fn(|index| shape_size(&param_shapes[index]));
    NetState {
        params: core::array::from_fn(|index| counter_pattern(seed_base + index * 17, sizes[index])),
        m: core::array::from_fn(|index| alloc::vec![0.0f32; sizes[index]]),
        v: core::array::from_fn(|index| alloc::vec![0.0f32; sizes[index]]),
        step: 1,
    }
}

/// Every current binding across BOTH nets, built fresh each step. Extra
/// entries beyond what a given `program` actually declares are harmless --
/// `evaluate_named` resolves by scanning `program`'s own `Op::Input` names,
/// never the other direction -- so one combined vector safely serves
/// whichever of `value_program`/`policy_program` it is handed to.
fn build_bindings<'a>(
    x: &'a [f32],
    reward: &'a [f32],
    action_one_hot: &'a [f32],
    actor_critic: &ActorCritic,
    policy: &'a NetState,
    value: &'a NetState,
    policy_wiring: &'a AdamWiring,
    value_wiring: &'a AdamWiring,
    step_policy: &'a [f32],
    step_value: &'a [f32],
) -> Vec<(&'a str, &'a [f32])> {
    let mut named: Vec<(&str, &[f32])> = alloc::vec![
        ("x", x),
        ("reward", reward),
        ("action_one_hot", action_one_hot),
        (policy_wiring.step_name, step_policy),
        (value_wiring.step_name, step_value),
    ];
    for index in 0..4 {
        named.push((
            actor_critic.policy_param_names[index],
            policy.params[index].as_slice(),
        ));
        named.push((
            policy_wiring.m_names[index].as_str(),
            policy.m[index].as_slice(),
        ));
        named.push((
            policy_wiring.v_names[index].as_str(),
            policy.v[index].as_slice(),
        ));
        named.push((
            actor_critic.value_param_names[index],
            value.params[index].as_slice(),
        ));
        named.push((
            value_wiring.m_names[index].as_str(),
            value.m[index].as_slice(),
        ));
        named.push((
            value_wiring.v_names[index].as_str(),
            value.v[index].as_slice(),
        ));
    }
    named
}

fn apply_update(state: &mut NetState, wiring: &AdamWiring, evaluated: &Evaluated) {
    for index in 0..4 {
        state.params[index] = evaluated
            .get(wiring.new_param[index])
            .expect("new param requested")
            .0
            .to_vec();
        state.m[index] = evaluated
            .get(wiring.new_m[index])
            .expect("new m requested")
            .0
            .to_vec();
        state.v[index] = evaluated
            .get(wiring.new_v[index])
            .expect("new v requested")
            .0
            .to_vec();
    }
    state.step += 1;
}

struct TrainingResult {
    final_probabilities: [Vec<f32>; STATE_DIM],
    final_value_predictions: [f32; STATE_DIM],
    value_loss_curve: Vec<f32>,
    policy_loss_curve: Vec<f32>,
}

/// Runs the actor-critic loop. `use_contaminated_value_gradient` selects
/// which `Differentiated` the value net's Adam step reads its gradient
/// from: `false` is the correct, disjoint source (`differentiate(&program,
/// value_loss)`); `true` deliberately sources it from
/// `differentiate(&program, policy_loss)` instead -- the exact quantity
/// [`policy_loss_gradient_reaches_the_value_net_but_is_never_applied_to_it`]
/// proves is nonzero, applied here on purpose to show what breaks.
fn run_actor_critic(steps: u32, use_contaminated_value_gradient: bool) -> TrainingResult {
    let actor_critic = build_actor_critic();
    let differentiated_value = differentiate(&actor_critic.program, actor_critic.value_loss)
        .expect("value loss differentiates");
    let differentiated_policy = differentiate(&actor_critic.program, actor_critic.policy_loss)
        .expect("policy loss differentiates");

    let value_gradient_source = if use_contaminated_value_gradient {
        &differentiated_policy
    } else {
        &differentiated_value
    };
    let mut value_program = value_gradient_source.program.clone();
    let value_config = AdamConfig {
        learning_rate: 0.1,
        ..AdamConfig::default()
    };
    let value_wiring = wire_adam(
        &mut value_program,
        &value_config,
        "step_value",
        actor_critic.value_param_names,
        actor_critic.value_param_nodes,
        &actor_critic.value_param_shapes,
        value_gradient_source,
    );

    let mut policy_program = differentiated_policy.program.clone();
    let policy_config = AdamConfig {
        learning_rate: 0.05,
        ..AdamConfig::default()
    };
    let policy_wiring = wire_adam(
        &mut policy_program,
        &policy_config,
        "step_policy",
        actor_critic.policy_param_names,
        actor_critic.policy_param_nodes,
        &actor_critic.policy_param_shapes,
        &differentiated_policy,
    );

    let mut policy_state = init_net_state(&actor_critic.policy_param_shapes, 101);
    let mut value_state = init_net_state(&actor_critic.value_param_shapes, 211);
    let mut rng = SplitMix64(0x1234_5678_9ABC_DEF0);
    let mut value_loss_curve = Vec::new();
    let mut policy_loss_curve = Vec::new();

    let dummy_reward = [0.0f32];
    let dummy_action_one_hot = alloc::vec![0.0f32; ACTION_DIM];

    for step_index in 0..steps {
        let state = (step_index as usize) % STATE_DIM;
        let x = one_hot(state, STATE_DIM);
        let policy_step = [policy_state.step as f32];
        let value_step = [value_state.step as f32];

        let probe_bindings = build_bindings(
            &x,
            &dummy_reward,
            &dummy_action_one_hot,
            &actor_critic,
            &policy_state,
            &value_state,
            &policy_wiring,
            &value_wiring,
            &policy_step,
            &value_step,
        );
        let probe = evaluate_named(
            &actor_critic.program,
            &[],
            &probe_bindings,
            &[actor_critic.probabilities],
        )
        .expect("probe forward pass evaluates");
        let probabilities = probe
            .get(actor_critic.probabilities)
            .expect("probabilities requested")
            .0
            .to_vec();

        let action = sample_action(&probabilities, &mut rng);
        let reward_value = [reward_for(state, action)];
        let action_vector = one_hot(action, ACTION_DIM);

        let named = build_bindings(
            &x,
            &reward_value,
            &action_vector,
            &actor_critic,
            &policy_state,
            &value_state,
            &policy_wiring,
            &value_wiring,
            &policy_step,
            &value_step,
        );

        let mut value_outputs = alloc::vec![actor_critic.value_loss];
        value_outputs.extend_from_slice(&value_wiring.new_param);
        value_outputs.extend_from_slice(&value_wiring.new_m);
        value_outputs.extend_from_slice(&value_wiring.new_v);
        let value_evaluated = evaluate_named(&value_program, &[], &named, &value_outputs)
            .expect("value program evaluates");
        value_loss_curve.push(
            value_evaluated
                .get(actor_critic.value_loss)
                .expect("value loss requested")
                .0[0],
        );

        let mut policy_outputs = alloc::vec![actor_critic.policy_loss];
        policy_outputs.extend_from_slice(&policy_wiring.new_param);
        policy_outputs.extend_from_slice(&policy_wiring.new_m);
        policy_outputs.extend_from_slice(&policy_wiring.new_v);
        let policy_evaluated = evaluate_named(&policy_program, &[], &named, &policy_outputs)
            .expect("policy program evaluates");
        policy_loss_curve.push(
            policy_evaluated
                .get(actor_critic.policy_loss)
                .expect("policy loss requested")
                .0[0],
        );

        apply_update(&mut value_state, &value_wiring, &value_evaluated);
        apply_update(&mut policy_state, &policy_wiring, &policy_evaluated);
    }

    let mut final_probabilities: [Vec<f32>; STATE_DIM] = core::array::from_fn(|_| Vec::new());
    let mut final_value_predictions = [0.0f32; STATE_DIM];
    for state in 0..STATE_DIM {
        let x = one_hot(state, STATE_DIM);
        let policy_step = [policy_state.step as f32];
        let value_step = [value_state.step as f32];
        let bindings = build_bindings(
            &x,
            &dummy_reward,
            &dummy_action_one_hot,
            &actor_critic,
            &policy_state,
            &value_state,
            &policy_wiring,
            &value_wiring,
            &policy_step,
            &value_step,
        );
        let evaluated = evaluate_named(
            &actor_critic.program,
            &[],
            &bindings,
            &[actor_critic.probabilities, actor_critic.value],
        )
        .expect("final forward pass evaluates");
        final_probabilities[state] = evaluated
            .get(actor_critic.probabilities)
            .expect("probabilities requested")
            .0
            .to_vec();
        final_value_predictions[state] = evaluated
            .get(actor_critic.value)
            .expect("value requested")
            .0[0];
    }

    TrainingResult {
        final_probabilities,
        final_value_predictions,
        value_loss_curve,
        policy_loss_curve,
    }
}

/// The combined absolute+relative criterion this session's own report calls
/// out (`f7cab09` measured raw relative error, which blows up on a
/// near-zero numeric gradient purely from f32 noise) -- an absolute floor
/// plus a relative term, never bare relative error.
fn combined_tolerance_ok(analytic: f32, numeric: f32) -> bool {
    (analytic - numeric).abs() <= 1e-2 + 1e-2 * numeric.abs()
}

/// Evaluates either loss at a fixed `(x, reward, action_one_hot)` scenario
/// against the 8 raw parameter buffers -- both `policy_loss` and
/// `value_loss` live on the same [`ActorCritic::program`], so one forward
/// evaluator serves both gradient-check tests.
#[allow(clippy::too_many_arguments)]
fn forward_loss_at(
    program: &[Op],
    loss: NodeId,
    x: &[f32],
    reward: &[f32],
    action_one_hot: &[f32],
    w1v: &[f32],
    b1v: &[f32],
    w2v: &[f32],
    b2v: &[f32],
    w1p: &[f32],
    b1p: &[f32],
    w2p: &[f32],
    b2p: &[f32],
) -> f32 {
    evaluate_named(
        program,
        &[],
        &[
            ("x", x),
            ("reward", reward),
            ("action_one_hot", action_one_hot),
            ("w1v", w1v),
            ("b1v", b1v),
            ("w2v", w2v),
            ("b2v", b2v),
            ("w1p", w1p),
            ("b1p", b1p),
            ("w2p", w2p),
            ("b2p", b2p),
        ],
        &[loss],
    )
    .expect("forward program lowers and evaluates")
    .get(loss)
    .expect("loss requested")
    .0[0]
}

/// Central-difference estimate of `d(loss)/d(buffers[which][index])`,
/// holding every other buffer fixed -- the same shape
/// `training_loop.rs`'s own `numeric_gradient` uses, generalized from 4
/// buffers to the 8 this file's two networks together carry. `buffers`
/// order is fixed: `[w1v, b1v, w2v, b2v, w1p, b1p, w2p, b2p]`.
#[allow(clippy::too_many_arguments)]
fn numeric_gradient(
    program: &[Op],
    loss: NodeId,
    x: &[f32],
    reward: &[f32],
    action_one_hot: &[f32],
    buffers: &mut [&mut Vec<f32>; 8],
    which: usize,
    index: usize,
    step: f32,
) -> f32 {
    let original = buffers[which][index];

    buffers[which][index] = original + step;
    let plus = forward_loss_at(
        program,
        loss,
        x,
        reward,
        action_one_hot,
        buffers[0],
        buffers[1],
        buffers[2],
        buffers[3],
        buffers[4],
        buffers[5],
        buffers[6],
        buffers[7],
    );

    buffers[which][index] = original - step;
    let minus = forward_loss_at(
        program,
        loss,
        x,
        reward,
        action_one_hot,
        buffers[0],
        buffers[1],
        buffers[2],
        buffers[3],
        buffers[4],
        buffers[5],
        buffers[6],
        buffers[7],
    );

    buffers[which][index] = original;
    (plus - minus) / (2.0 * step)
}

/// Structural + numeric proof of this file's own module-doc claim: two
/// `differentiate` calls on the same program give two disjoint gradient
/// tables, and the one direction that WOULD need `.detach()` in an eager
/// framework -- `policy_loss`'s backward pass reaching the value net's
/// parameters through the `advantage` baseline -- is present and nonzero
/// here too. What prevents it from corrupting the value net is that
/// [`run_actor_critic`] never reads a value-net gradient out of
/// `differentiated_policy`; it only ever reads it out of
/// `differentiated_value`, a structurally separate `Differentiated` whose
/// own truncated program does not even contain a policy op.
#[proxima::test]
async fn policy_loss_gradient_reaches_the_value_net_but_is_never_applied_to_it() {
    let actor_critic = build_actor_critic();
    let differentiated_policy = differentiate(&actor_critic.program, actor_critic.policy_loss)
        .expect("policy loss differentiates");
    let differentiated_value = differentiate(&actor_critic.program, actor_critic.value_loss)
        .expect("value loss differentiates");

    assert!(
        differentiated_value.gradient_of_named("w1p").is_none(),
        "value_loss's own truncated program ends before the policy net's ops even exist -- there is \
         structurally no policy gradient to read out of it, not merely a zeroed one"
    );
    assert!(
        differentiated_value.gradient_of_named("b2p").is_none(),
        "same absence for every policy parameter"
    );

    let grad_w1v_from_policy_loss = differentiated_policy.gradient_of_named("w1v").expect(
        "policy_loss reads w1v through the value baseline inside `advantage = reward - value`",
    );

    let x = one_hot(0, STATE_DIM);
    let reward = [-1.0f32];
    let action_one_hot = one_hot(1, ACTION_DIM);
    let w1v = counter_pattern(401, STATE_DIM * HIDDEN_DIM);
    let b1v = counter_pattern(402, HIDDEN_DIM);
    let w2v = counter_pattern(403, HIDDEN_DIM);
    let b2v = counter_pattern(404, 1);
    let w1p = counter_pattern(405, STATE_DIM * HIDDEN_DIM);
    let b1p = counter_pattern(406, HIDDEN_DIM);
    let w2p = counter_pattern(407, HIDDEN_DIM * ACTION_DIM);
    let b2p = counter_pattern(408, ACTION_DIM);

    let evaluated = evaluate_named(
        &differentiated_policy.program,
        &[],
        &[
            ("x", x.as_slice()),
            ("reward", &reward),
            ("action_one_hot", action_one_hot.as_slice()),
            ("w1v", &w1v),
            ("b1v", &b1v),
            ("w2v", &w2v),
            ("b2v", &b2v),
            ("w1p", &w1p),
            ("b1p", &b1p),
            ("w2p", &w2p),
            ("b2p", &b2p),
        ],
        &[grad_w1v_from_policy_loss],
    )
    .expect("policy adjoint program lowers and evaluates");

    let contaminated = evaluated
        .get(grad_w1v_from_policy_loss)
        .expect("requested")
        .0;
    let max_abs_value = contaminated
        .iter()
        .fold(0.0f32, |worst, &value| worst.max(value.abs()));
    std::eprintln!(
        "policy_loss's gradient into w1v (computed, never applied): max |value| = {max_abs_value}, values = {contaminated:?}"
    );
    assert!(
        max_abs_value > 1e-4,
        "policy_loss's backward pass must genuinely reach w1v with a nonzero contribution -- this is the exact \
         quantity an eager framework's `.detach()` would zero; got {contaminated:?}"
    );
}

/// The convergence deliverable: a policy that learns the known-optimal
/// action per state, and a value net whose prediction tracks
/// `E_a~pi[reward(s,a)]` -- the quantity single-sample MSE regression
/// against the observed per-step reward actually converges to, not the
/// optimal-policy return, since the sampled action is stochastic.
#[proxima::test]
async fn policy_converges_to_the_known_optimal_action_and_value_predicts_the_expected_return() {
    let result = run_actor_critic(3000, false);

    for state in 0..STATE_DIM {
        let probabilities = &result.final_probabilities[state];
        let best_action = probabilities
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(action, _)| action)
            .expect("three actions");
        let expected_return: f32 = probabilities
            .iter()
            .enumerate()
            .map(|(action, &probability)| probability * reward_for(state, action))
            .sum();
        let value_prediction = result.final_value_predictions[state];
        std::eprintln!(
            "state {state}: probabilities {probabilities:?}, best action {best_action}, value prediction \
             {value_prediction}, E_a~pi[reward] {expected_return}"
        );

        assert_eq!(
            best_action, state,
            "state {state}'s known-optimal action is itself; got {probabilities:?}"
        );
        assert!(
            probabilities[state] > 0.6,
            "state {state} must clearly prefer the optimal action over uniform (1/3), got {probabilities:?}"
        );
        assert!(
            (value_prediction - expected_return).abs() < 0.2,
            "value net at state {state} must track E_a~pi[reward(s,a)] = {expected_return}, got {value_prediction}"
        );
    }

    assert!(
        result
            .value_loss_curve
            .iter()
            .all(|value| value.is_finite()),
        "value loss curve went non-finite: {:?}",
        &result.value_loss_curve[..10]
    );
    assert!(
        result
            .policy_loss_curve
            .iter()
            .all(|value| value.is_finite()),
        "policy loss curve went non-finite: {:?}",
        &result.policy_loss_curve[..10]
    );

    let window = 30;
    let early_value_loss: f32 =
        result.value_loss_curve[..window].iter().sum::<f32>() / window as f32;
    let late_value_loss: f32 = result.value_loss_curve[result.value_loss_curve.len() - window..]
        .iter()
        .sum::<f32>()
        / window as f32;
    std::eprintln!(
        "value loss: early {window}-step average {early_value_loss}, late average {late_value_loss}"
    );
    assert!(
        late_value_loss < early_value_loss,
        "value net's MSE must decrease over training: early {early_value_loss}, late {late_value_loss}"
    );
}

/// The assertion this session's design correction asked for in place of a
/// with/without-detach comparison: two otherwise-identical training runs,
/// differing ONLY in which `Differentiated` the value net's Adam step reads
/// its gradient from. The correct source converges the value net to the
/// true expected return; the contaminated source (the exact quantity
/// [`policy_loss_gradient_reaches_the_value_net_but_is_never_applied_to_it`]
/// proves is nonzero) does not -- proof that the composition discipline is
/// load-bearing, not merely harmless-looking.
#[proxima::test]
#[case::correct_gradient_source_converges(false)]
#[case::contaminated_gradient_source_fails_to_converge(true)]
async fn value_net_convergence_depends_on_which_differentiated_it_is_stepped_from(
    #[case] use_contaminated_value_gradient: bool,
) {
    let result = run_actor_critic(1200, use_contaminated_value_gradient);

    let mut worst_error = 0.0f32;
    for state in 0..STATE_DIM {
        let probabilities = &result.final_probabilities[state];
        let expected_return: f32 = probabilities
            .iter()
            .enumerate()
            .map(|(action, &probability)| probability * reward_for(state, action))
            .sum();
        let error = (result.final_value_predictions[state] - expected_return).abs();
        worst_error = worst_error.max(error);
    }
    std::eprintln!(
        "use_contaminated_value_gradient={use_contaminated_value_gradient}: worst |V(s) - E_a~pi[reward]| = \
         {worst_error}, final value predictions {:?}",
        result.final_value_predictions
    );

    if use_contaminated_value_gradient {
        assert!(
            worst_error > 0.3,
            "stepping the value net from policy_loss's gradient into w1v/etc (the quantity `.detach()` would \
             zero in an eager framework) must fail to track the true expected return -- got worst error \
             {worst_error}, which would falsely look converged"
        );
    } else {
        assert!(
            worst_error < 0.2,
            "stepping the value net from its own value_loss gradient must converge to the true expected \
             return -- got worst error {worst_error}"
        );
    }
}

/// Gradient check 1 of 2: the value net, under the combined
/// absolute+relative criterion.
#[proxima::test]
async fn value_network_gradient_check_matches_central_difference() {
    let actor_critic = build_actor_critic();
    let differentiated = differentiate(&actor_critic.program, actor_critic.value_loss)
        .expect("value loss differentiates");

    let x = alloc::vec![0.37f32, -0.52, 0.68];
    let reward = [-1.0f32];
    let action_one_hot = one_hot(2, ACTION_DIM);
    let mut w1v = counter_pattern(301, STATE_DIM * HIDDEN_DIM);
    let mut b1v = counter_pattern(302, HIDDEN_DIM);
    let mut w2v = counter_pattern(303, HIDDEN_DIM);
    let mut b2v = counter_pattern(304, 1);
    let mut w1p = counter_pattern(305, STATE_DIM * HIDDEN_DIM);
    let mut b1p = counter_pattern(306, HIDDEN_DIM);
    let mut w2p = counter_pattern(307, HIDDEN_DIM * ACTION_DIM);
    let mut b2p = counter_pattern(308, ACTION_DIM);

    let grad_w1v = differentiated
        .gradient_of_named("w1v")
        .expect("w1v feeds value_loss");
    let grad_b1v = differentiated
        .gradient_of_named("b1v")
        .expect("b1v feeds value_loss");
    let grad_w2v = differentiated
        .gradient_of_named("w2v")
        .expect("w2v feeds value_loss");
    let grad_b2v = differentiated
        .gradient_of_named("b2v")
        .expect("b2v feeds value_loss");

    let evaluated = evaluate_named(
        &differentiated.program,
        &[],
        &[
            ("x", x.as_slice()),
            ("reward", &reward),
            ("action_one_hot", action_one_hot.as_slice()),
            ("w1v", &w1v),
            ("b1v", &b1v),
            ("w2v", &w2v),
            ("b2v", &b2v),
            ("w1p", &w1p),
            ("b1p", &b1p),
            ("w2p", &w2p),
            ("b2p", &b2p),
        ],
        &[grad_w1v, grad_b1v, grad_w2v, grad_b2v],
    )
    .expect("adjoint program lowers and evaluates");

    let analytic_w1v = evaluated.get(grad_w1v).expect("requested").0.to_vec();
    let analytic_b1v = evaluated.get(grad_b1v).expect("requested").0.to_vec();
    let analytic_w2v = evaluated.get(grad_w2v).expect("requested").0.to_vec();
    let analytic_b2v = evaluated.get(grad_b2v).expect("requested").0.to_vec();

    let step = 1e-3f32;
    let mut worst = (0.0f32, "", 0usize);
    for (which, name, analytic) in [
        (0usize, "w1v", &analytic_w1v),
        (1, "b1v", &analytic_b1v),
        (2, "w2v", &analytic_w2v),
        (3, "b2v", &analytic_b2v),
    ] {
        for (index, &analytic_value) in analytic.iter().enumerate() {
            let numeric = numeric_gradient(
                &actor_critic.program,
                actor_critic.value_loss,
                &x,
                &reward,
                &action_one_hot,
                &mut [
                    &mut w1v, &mut b1v, &mut w2v, &mut b2v, &mut w1p, &mut b1p, &mut w2p, &mut b2p,
                ],
                which,
                index,
                step,
            );
            if !combined_tolerance_ok(analytic_value, numeric) {
                let deviation = (analytic_value - numeric).abs();
                if deviation > worst.0 {
                    worst = (deviation, name, index);
                }
            }
        }
    }

    std::eprintln!("value network gradient check: worst combined-criterion deviation {worst:?}");
    assert_eq!(
        worst.0, 0.0,
        "value network gradient failed the combined criterion at {worst:?}"
    );
}

/// Gradient check 2 of 2: the policy net, under the same combined
/// criterion, with the value net held at a fixed representative snapshot
/// (policy_loss's forward pass depends on it through `advantage`, so it
/// must be bound even though it is not being perturbed here).
#[proxima::test]
async fn policy_network_gradient_check_matches_central_difference() {
    let actor_critic = build_actor_critic();
    let differentiated = differentiate(&actor_critic.program, actor_critic.policy_loss)
        .expect("policy loss differentiates");

    let x = alloc::vec![-0.44f32, 0.81, 0.19];
    let reward = [1.0f32];
    let action_one_hot = one_hot(0, ACTION_DIM);
    let mut w1v = counter_pattern(501, STATE_DIM * HIDDEN_DIM);
    let mut b1v = counter_pattern(502, HIDDEN_DIM);
    let mut w2v = counter_pattern(503, HIDDEN_DIM);
    let mut b2v = counter_pattern(504, 1);
    let mut w1p = counter_pattern(505, STATE_DIM * HIDDEN_DIM);
    let mut b1p = counter_pattern(506, HIDDEN_DIM);
    let mut w2p = counter_pattern(507, HIDDEN_DIM * ACTION_DIM);
    let mut b2p = counter_pattern(508, ACTION_DIM);

    let grad_w1p = differentiated
        .gradient_of_named("w1p")
        .expect("w1p feeds policy_loss");
    let grad_b1p = differentiated
        .gradient_of_named("b1p")
        .expect("b1p feeds policy_loss");
    let grad_w2p = differentiated
        .gradient_of_named("w2p")
        .expect("w2p feeds policy_loss");
    let grad_b2p = differentiated
        .gradient_of_named("b2p")
        .expect("b2p feeds policy_loss");

    let evaluated = evaluate_named(
        &differentiated.program,
        &[],
        &[
            ("x", x.as_slice()),
            ("reward", &reward),
            ("action_one_hot", action_one_hot.as_slice()),
            ("w1v", &w1v),
            ("b1v", &b1v),
            ("w2v", &w2v),
            ("b2v", &b2v),
            ("w1p", &w1p),
            ("b1p", &b1p),
            ("w2p", &w2p),
            ("b2p", &b2p),
        ],
        &[grad_w1p, grad_b1p, grad_w2p, grad_b2p],
    )
    .expect("adjoint program lowers and evaluates");

    let analytic_w1p = evaluated.get(grad_w1p).expect("requested").0.to_vec();
    let analytic_b1p = evaluated.get(grad_b1p).expect("requested").0.to_vec();
    let analytic_w2p = evaluated.get(grad_w2p).expect("requested").0.to_vec();
    let analytic_b2p = evaluated.get(grad_b2p).expect("requested").0.to_vec();

    let step = 1e-3f32;
    let mut worst = (0.0f32, "", 0usize);
    for (which, name, analytic) in [
        (4usize, "w1p", &analytic_w1p),
        (5, "b1p", &analytic_b1p),
        (6, "w2p", &analytic_w2p),
        (7, "b2p", &analytic_b2p),
    ] {
        for (index, &analytic_value) in analytic.iter().enumerate() {
            let numeric = numeric_gradient(
                &actor_critic.program,
                actor_critic.policy_loss,
                &x,
                &reward,
                &action_one_hot,
                &mut [
                    &mut w1v, &mut b1v, &mut w2v, &mut b2v, &mut w1p, &mut b1p, &mut w2p, &mut b2p,
                ],
                which,
                index,
                step,
            );
            if !combined_tolerance_ok(analytic_value, numeric) {
                let deviation = (analytic_value - numeric).abs();
                if deviation > worst.0 {
                    worst = (deviation, name, index);
                }
            }
        }
    }

    std::eprintln!("policy network gradient check: worst combined-criterion deviation {worst:?}");
    assert_eq!(
        worst.0, 0.0,
        "policy network gradient failed the combined criterion at {worst:?}"
    );
}
