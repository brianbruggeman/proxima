//! Replays the CURRENT generic kernel against captured production buffers for
//! node 2994 (plain gate reduce, `[600,1536,12288]` Q4_0) and node 3009 (the
//! same-shaped up_proj reduce with the fused GeGLU epilogue, reading node
//! 2994's gate output 5x) -- see `arena_encode_dispatch_finish.rs`'s
//! `capture_dispatch`/`flush_pending_capture_dumps` for how the dump this
//! test reads was produced (`PROXIMA_CAPTURE_DUMP_DIR`).
//!
//! Ignored by default: this test reads a specific captured production dump
//! from `REPLAY_DUMP_DIR`, not a fixture this repo ships -- run explicitly
//! after a capture (`prefill599/replay/capture.log`'s own run) with
//! `REPLAY_DUMP_DIR=<dir> cargo test --release -p omega --features
//! metal,instrument --test replay_projection -- --ignored --nocapture`.

#![cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};
use std::time::Instant;

use proxima_tensor::spec::{elementwise, scalar_constant};
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce, ReduceInit,
    ScalarOp, append, projection,
};

const TOKENS: u32 = 600;
const EMBEDDING: u32 = 1536;
const FEED_FORWARD: u32 = 12288;
/// Q4_0: 18 bytes (2-byte fp16 scale + 16 nibble-packed bytes) per 32-element
/// block -- the real operand length; the dump's own file length (64 MiB) is
/// `flush_pending_capture_dumps`'s safety cap, not this weight's true size.
const WEIGHT_BYTES: usize = (FEED_FORWARD as usize) * (EMBEDDING as usize) * 18 / 32;
const ACTIVATION_BYTES: usize = (TOKENS as usize) * (EMBEDDING as usize) * 4;
const ROW_BYTES: usize = (TOKENS as usize) * (FEED_FORWARD as usize) * 4;

/// `REPLAY_ARM=current|unroll|index32|unroll_index32` runs only that one arm
/// (unset runs every arm in [`ARMS`]). `REPLAY_NODE=
/// 2994|3009` runs only that node's test (the other returns immediately).
/// `REPLAY_WARM`/`REPLAY_ITERS`/`REPLAY_IDLE_MS` override the defaults
/// below -- all four read once per test process, matching this crate's own
/// `PROXIMA_*` env posture (`kernel_types_identity.rs`'s overrides).
fn env_str(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

fn env_usize(name: &str, default: usize) -> usize {
    env_str(name).and_then(|value| value.parse().ok()).unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    env_str(name).and_then(|value| value.parse().ok()).unwrap_or(default)
}

/// `true` when this process should skip `node`'s test entirely -- `REPLAY_NODE`
/// set to a DIFFERENT node number. Unset or matching runs it.
fn node_deselected(node: u32) -> bool {
    env_str("REPLAY_NODE").is_some_and(|wanted| wanted.parse::<u32>() != Ok(node))
}

fn selected_arms() -> Vec<(&'static str, &'static [EnvVar])> {
    match env_str("REPLAY_ARM") {
        None => ARMS.to_vec(),
        Some(wanted) => ARMS
            .iter()
            .copied()
            .filter(|(suffix, _)| *suffix == wanted)
            .collect(),
    }
}

fn dump_dir() -> PathBuf {
    PathBuf::from(std::env::var("REPLAY_DUMP_DIR").expect(
        "REPLAY_DUMP_DIR must point at a PROXIMA_CAPTURE_DUMP_DIR capture (node2994/node3009 files)",
    ))
}

fn read_exact(dir: &Path, name: &str, len: usize) -> Vec<u8> {
    let bytes = std::fs::read(dir.join(name)).unwrap_or_else(|err| panic!("reading {name}: {err}"));
    assert!(
        bytes.len() >= len,
        "{name}: dumped {} bytes, need at least {len}",
        bytes.len()
    );
    bytes[..len].to_vec()
}

fn f32_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

/// `[out_dim, in_dim] x [tokens, in_dim] -> [tokens, out_dim]`, reduced over
/// `in_dim` -- the same construction `packed_row_multi_row_unroll_ab.rs`'s
/// own `matmul_program` proves bit-exact against a CPU f32 oracle.
fn append_matmul(program: &mut Vec<Op>, weight: NodeId, activation: NodeId) -> NodeId {
    let product = append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(projection(3, &[1, 2]))),
                (activation, IndexMap::Affine(projection(3, &[0, 2]))),
            ],
            name: None,
        },
    );
    append(
        program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: None,
        }),
    )
}

/// `proxima_tensor::spec::attention_forward::append_activation`'s
/// `Activation::GeluTanh` arm, copied node-for-node (that function is
/// `pub(super)`, unreachable from this crate) --
/// `gelu_pytorch_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 *
/// x^3)))`, Gemma's GeGLU nonlinearity.
fn append_gelu_tanh(program: &mut Vec<Op>, x: NodeId, ones: NodeId) -> NodeId {
    let half = scalar_constant(program, 0.5);
    let cubic_coeff = scalar_constant(program, 0.044_715);
    let sqrt_two_over_pi = scalar_constant(program, 0.797_884_6);
    let x_squared = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(x, "sg->sg"), (x, "sg->sg")]).unwrap();
    let x_cubed = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(x_squared, "sg->sg"), (x, "sg->sg")]).unwrap();
    let cubic_term = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(x_cubed, "sg->sg"), (cubic_coeff, "->sg")]).unwrap();
    let inner = elementwise(program, DType::Float32, ScalarOp::Add, &[(x, "sg->sg"), (cubic_term, "sg->sg")]).unwrap();
    let scaled_inner = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(inner, "sg->sg"), (sqrt_two_over_pi, "->sg")]).unwrap();
    let tanh_term = elementwise(program, DType::Float32, ScalarOp::Tanh, &[(scaled_inner, "sg->sg")]).unwrap();
    let one_plus_tanh = elementwise(program, DType::Float32, ScalarOp::Add, &[(tanh_term, "sg->sg"), (ones, "->sg")]).unwrap();
    let half_x = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(x, "sg->sg"), (half, "->sg")]).unwrap();
    elementwise(program, DType::Float32, ScalarOp::Multiply, &[(half_x, "sg->sg"), (one_plus_tanh, "sg->sg")]).unwrap()
}

/// Node 2994's own replay: plain `[600,1536,12288]` Q4_0 reduce, no epilogue.
fn gate_program() -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(&mut program, Op::Input { dtype: DType::UInt8, shape: vec![Extent::Static(FEED_FORWARD), Extent::Static(EMBEDDING)], name: None });
    let activation = append(&mut program, Op::Input { dtype: DType::Float32, shape: vec![Extent::Static(TOKENS), Extent::Static(EMBEDDING)], name: None });
    let gate = append_matmul(&mut program, weight, activation);
    (program, gate)
}

/// Node 3009's own replay: the up_proj `[600,1536,12288]` Q4_0 reduce fused
/// with the GeGLU epilogue, consuming node 2994's REAL captured gate output
/// directly (bound as a plain `Float32` input) rather than recomputing it --
/// isolating exactly the dispatch under test.
fn up_geglu_program() -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(&mut program, Op::Input { dtype: DType::UInt8, shape: vec![Extent::Static(FEED_FORWARD), Extent::Static(EMBEDDING)], name: None });
    let activation = append(&mut program, Op::Input { dtype: DType::Float32, shape: vec![Extent::Static(TOKENS), Extent::Static(EMBEDDING)], name: None });
    let gate = append(&mut program, Op::Input { dtype: DType::Float32, shape: vec![Extent::Static(TOKENS), Extent::Static(FEED_FORWARD)], name: None });
    let up = append_matmul(&mut program, weight, activation);
    let ones = scalar_constant(&mut program, 1.0);
    let activated_gate = append_gelu_tanh(&mut program, gate, ones);
    let ffn_hidden = elementwise(&mut program, DType::Float32, ScalarOp::Multiply, &[(activated_gate, "sg->sg"), (up, "sg->sg")]).unwrap();
    (program, ffn_hidden)
}

/// (arm suffix, env vars to set for that arm) -- shared by both nodes'
/// tests so a new arm is one row here, not two edits. Each arm sets only its
/// own var(s), leaving the others unset.
type EnvVar = (&'static str, Option<&'static str>);

const ARMS: &[(&str, &[EnvVar])] = &[
    ("current", &[]),
    ("unroll", &[("PROXIMA_MULTI_ROW_UNROLL", Some("1"))]),
    ("index32", &[("PROXIMA_MULTI_ROW_INDEX32", Some("1"))]),
    (
        "unroll_index32",
        &[
            ("PROXIMA_MULTI_ROW_UNROLL", Some("1")),
            ("PROXIMA_MULTI_ROW_INDEX32", Some("1")),
        ],
    ),
];

struct ReplayRun {
    label: String,
    wall_ms: Vec<f64>,
    gpu_ms: Vec<f64>,
    differing_words: usize,
    first_differences: Vec<(usize, f32, f32)>,
    kernel_keys: Vec<String>,
}

/// Checks `bits` against `production` element-wise, returning this call's
/// own differing count and (at most 3) first differences -- called once per
/// warm iteration and once per timed iteration, never only at the end, so a
/// regression on iteration 7 of 20 cannot hide behind iteration 20's own
/// correct output.
fn check_bits(bits: &[u32], production: &[f32]) -> (usize, Vec<(usize, f32, f32)>) {
    let mut differing_words = 0usize;
    let mut first_differences = Vec::new();
    for (index, (replayed_bits, &production_value)) in bits.iter().zip(production.iter()).enumerate() {
        let replayed_value = f32::from_bits(*replayed_bits);
        if replayed_value.to_bits() != production_value.to_bits() {
            differing_words += 1;
            if first_differences.len() < 3 {
                first_differences.push((index, replayed_value, production_value));
            }
        }
    }
    (differing_words, first_differences)
}

fn replay(program: &[Op], output: NodeId, blocks: &[QuantizedBlock<'_>], production: &[f32], env_vars: &[(&str, Option<&str>)], label: String) -> ReplayRun {
    let warm = env_usize("REPLAY_WARM", 2);
    let iters = env_usize("REPLAY_ITERS", 20);
    let idle_ms = env_u64("REPLAY_IDLE_MS", 1500);
    temp_env::with_vars(env_vars, || {
        // production's own numeric policy -- see omega/src/metal/
        // pipeline_buffers_upload.rs:22-28,101-109: NumericPolicy::default()
        // (bits 0x00) derives MathMode::Safe ('S'), but the captured
        // dispatch_capture key ends `...w6403R`: cooperative_width=64
        // (matches), numeric_policy_token="03" (contraction+reassociation,
        // exactly llama_relaxed()'s bit pattern), MathMode::Relaxed ('R').
        //
        // Warm phase: `plan()` (structural compile) plus `warm` full
        // execute_plan_timed calls (which is where `pipeline_for`'s
        // `PIPELINE_CACHE` miss actually JIT-compiles and prints its own
        // `pipeline_create key=... msl_sha256=...` line, `omega/src/metal/
        // pipeline_buffers_upload.rs`) all run BEFORE `window_begin` below --
        // grep this run's stderr for `pipeline_create` appearing before
        // `window_begin` to confirm the window holds execution only.
        let plan = omega::plan(program, &[], blocks, &[output], NumericPolicy::llama_relaxed())
            .expect("metal plans the replayed projection");
        let kernel_keys = plan.kernel_keys().expect("plan reports its own kernel cache keys");
        let mut warm_differing_words = 0usize;
        for _ in 0..warm {
            let (evaluated, _gpu_ns) =
                omega::execute_plan_timed(&plan, blocks).expect("metal runs the replayed projection (warm)");
            let bits: Vec<u32> = evaluated.root().iter().map(|value| value.to_bits()).collect();
            warm_differing_words += check_bits(&bits, production).0;
        }
        eprintln!(
            "warm_check label={label} warm_iters={warm} differing_words={warm_differing_words} \
             (0 expected; every pipeline_create above this line belongs to warm, not the window)"
        );

        let idle = std::time::Duration::from_millis(idle_ms);
        let epoch = Instant::now();
        let wall_now = || {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64()
        };
        eprintln!(
            "idle_pre_begin label={label} instant_s={:.6} wall_s={:.6}",
            epoch.elapsed().as_secs_f64(),
            wall_now()
        );
        std::thread::sleep(idle);
        eprintln!(
            "window_begin label={label} instant_s={:.6} wall_s={:.6}",
            epoch.elapsed().as_secs_f64(),
            wall_now()
        );

        let mut wall_ms = Vec::with_capacity(iters);
        let mut gpu_ms = Vec::with_capacity(iters);
        let mut differing_words = 0usize;
        let mut first_differences = Vec::new();
        for _ in 0..iters {
            let started = Instant::now();
            // `gpu_ns` is GPU-only (command-buffer start/end times, no host
            // encode/readback); `started.elapsed()` below is wall time
            // around the whole call, including any host conversion.
            let (evaluated, gpu_ns) =
                omega::execute_plan_timed(&plan, blocks).expect("metal runs the replayed projection");
            wall_ms.push(started.elapsed().as_secs_f64() * 1e3);
            gpu_ms.push(gpu_ns as f64 / 1e6);
            let bits: Vec<u32> = evaluated.root().iter().map(|value| value.to_bits()).collect();
            let (this_differing, this_first) = check_bits(&bits, production);
            differing_words += this_differing;
            if first_differences.is_empty() {
                first_differences = this_first;
            }
        }
        eprintln!(
            "window_end label={label} instant_s={:.6} wall_s={:.6}",
            epoch.elapsed().as_secs_f64(),
            wall_now()
        );
        std::thread::sleep(idle);
        eprintln!(
            "idle_post_end label={label} instant_s={:.6} wall_s={:.6}",
            epoch.elapsed().as_secs_f64(),
            wall_now()
        );

        ReplayRun { label, wall_ms, gpu_ms, differing_words, first_differences, kernel_keys }
    })
}

fn median_of(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

fn min_of(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::INFINITY, f64::min)
}

fn report(run: &ReplayRun) {
    eprintln!(
        "replay label={} min_wall_ms={:.3} median_wall_ms={:.3} min_gpu_ms={:.3} median_gpu_ms={:.3} \
         differing_words={} kernel_keys={:?}",
        run.label,
        min_of(&run.wall_ms),
        median_of(&run.wall_ms),
        min_of(&run.gpu_ms),
        median_of(&run.gpu_ms),
        run.differing_words,
        run.kernel_keys,
    );
    for (index, replayed, production) in &run.first_differences {
        eprintln!("  first_difference index={index} replayed={replayed} production={production}");
    }
}

#[test]
#[ignore = "reads a captured production dump from REPLAY_DUMP_DIR"]
fn replay_node2994_gate_matches_production_dump() {
    if node_deselected(2994) {
        return;
    }
    let dir = dump_dir();
    let weight = read_exact(&dir, "node2994_buf1_off2632245504_len67108864.bin", WEIGHT_BYTES);
    let activation = f32_le(&read_exact(&dir, "node2994_buf0_off0_len3686400.bin", ACTIVATION_BYTES));
    let production = f32_le(&read_exact(&dir, "node2994_buf2_off0_len29491200.bin", ROW_BYTES));
    let blocks = [
        QuantizedBlock::Packed { codec: omega::Codec::Q4_0, bytes: &weight },
        QuantizedBlock::Float32(&activation),
    ];
    let mut runs = Vec::new();
    for (suffix, env_vars) in selected_arms() {
        let label = format!("node2994_{suffix}");
        let (program, gate) = gate_program();
        let run = replay(&program, gate, &blocks, &production, env_vars, label);
        report(&run);
        runs.push(run);
    }
    for run in &runs {
        assert_eq!(run.differing_words, 0, "{}: replay diverged from the production dump", run.label);
    }
}

#[test]
#[ignore = "reads a captured production dump from REPLAY_DUMP_DIR"]
fn replay_node3009_up_geglu_matches_production_dump() {
    if node_deselected(3009) {
        return;
    }
    let dir = dump_dir();
    let weight = read_exact(&dir, "node3009_buf1_off2642868480_len67108864.bin", WEIGHT_BYTES);
    let activation = f32_le(&read_exact(&dir, "node3009_buf0_off0_len3686400.bin", ACTIVATION_BYTES));
    let gate = f32_le(&read_exact(&dir, "node2994_buf2_off0_len29491200.bin", ROW_BYTES));
    let production = f32_le(&read_exact(&dir, "node3009_buf11_off0_len29491200.bin", ROW_BYTES));
    let blocks = [
        QuantizedBlock::Packed { codec: omega::Codec::Q4_0, bytes: &weight },
        QuantizedBlock::Float32(&activation),
        QuantizedBlock::Float32(&gate),
    ];
    let mut runs = Vec::new();
    for (suffix, env_vars) in selected_arms() {
        let label = format!("node3009_{suffix}");
        let (program, ffn_hidden) = up_geglu_program();
        let run = replay(&program, ffn_hidden, &blocks, &production, env_vars, label);
        report(&run);
        runs.push(run);
    }
    for run in &runs {
        assert_eq!(run.differing_words, 0, "{}: replay diverged from the production dump", run.label);
    }
}
