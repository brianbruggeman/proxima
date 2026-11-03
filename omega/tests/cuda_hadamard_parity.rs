//! CPU-vs-CUDA parity of the Hadamard fold's activation side at the Bonsai-2 27B geometry:
//! block 1024 (32 x 32), widths 5120 / 6144 / 17408, attention heads of 256 and recurrence heads
//! of 128, so a disagreement names the construct that diverges.

#![cfg(all(feature = "cpu", feature = "cuda-driver"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use omega::Codec;
use omega::backend::{Engine, GpuDriver, execute_plan_named, plan_named};
use proxima_tensor::spec::{HadamardFold, input_leaf, merge_head_axes};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{DType, Extent, NodeId, NumericPolicy, Op, QuantizedBlock};

const BLOCK: u32 = 1024;
const OUTER: usize = 32;
const TOLERANCE: f32 = 1e-4;

fn sylvester(size: usize) -> Vec<f32> {
    let scale = 1.0 / (size as f32).sqrt();
    (0..size)
        .flat_map(|row| {
            (0..size).map(move |column| {
                if (row & column).count_ones() % 2 == 0 {
                    scale
                } else {
                    -scale
                }
            })
        })
        .collect()
}

fn signs(width: usize) -> Vec<f32> {
    (0..width)
        .map(|index| if (index * 2654435761usize) >> 7 & 1 == 0 { 1.0 } else { -1.0 })
        .collect()
}

fn random(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

struct Case {
    program: Vec<Op>,
    root: NodeId,
    named: Vec<(String, Vec<f32>)>,
}

fn fold_leaves(width: u32) -> Vec<(String, Vec<f32>)> {
    vec![
        (
            format!("prism.hadamard.rotation.{OUTER}"),
            sylvester(OUTER),
        ),
        (
            format!("prism.hadamard.signs.{width}"),
            signs(width as usize),
        ),
    ]
}

fn run(engine: Engine, driver: Option<GpuDriver>, case: &Case, policy: NumericPolicy) -> Vec<f32> {
    let named: Vec<(&str, QuantizedBlock<'_>)> = case
        .named
        .iter()
        .map(|(name, data)| (name.as_str(), QuantizedBlock::Float32(data)))
        .collect();
    let mut plan = plan_named(engine, driver, &case.program, &[], &named, &[case.root], policy)
        .expect("the case plans");
    execute_plan_named(&mut plan, &named)
        .expect("the case executes")
        .root()
        .to_vec()
}

fn assert_parity(label: &str, case: &Case) {
    let expected = run(Engine::Cpu, None, case, NumericPolicy::default());
    let actual = run(
        Engine::Gpu,
        Some(GpuDriver::Cuda),
        case,
        NumericPolicy::bit_exact().with_reassociation(true),
    );
    assert_eq!(expected.len(), actual.len(), "{label}: element count");
    let magnitude = expected.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
    let worst = expected
        .iter()
        .zip(&actual)
        .map(|(want, got)| if got.is_finite() { (want - got).abs() } else { f32::INFINITY })
        .fold(0.0f32, f32::max);
    let relative = worst / magnitude.max(f32::MIN_POSITIVE);
    assert!(
        relative < TOLERANCE,
        "{label}: cuda vs cpu relative={relative} (worst {worst}, magnitude {magnitude})"
    );
}

fn flat_case(width: u32, positions: u32) -> Case {
    let mut program = Vec::new();
    let x = input_leaf(
        &mut program,
        DType::FLOAT32,
        vec![Extent::Static(positions), Extent::Static(width)],
        "x",
    );
    let fold = HadamardFold::declare(&mut program, BLOCK, width, width, 1, true, &[]);
    let root = fold
        .rotate_flat(&mut program, x, width, "s")
        .expect("rotate_flat lowers");
    let mut named = vec![("x".to_owned(), random(3, (positions * width) as usize))];
    named.extend(fold_leaves(width));
    Case { program, root, named }
}

fn heads_case(heads: u32, head_dim: u32, positions: u32) -> Case {
    let width = heads * head_dim;
    let mut program = Vec::new();
    let x = input_leaf(
        &mut program,
        DType::FLOAT32,
        vec![
            Extent::Static(positions),
            Extent::Static(heads),
            Extent::Static(head_dim),
        ],
        "x",
    );
    let fold = HadamardFold::declare(&mut program, BLOCK, width, width, 1, true, &[]);
    let root = fold
        .rotate_heads(&mut program, x, heads, head_dim, "s")
        .expect("rotate_heads lowers");
    let mut named = vec![("x".to_owned(), random(5, (positions * width) as usize))];
    named.extend(fold_leaves(width));
    Case { program, root, named }
}

fn restore_case(width: u32, positions: u32) -> Case {
    let mut program = Vec::new();
    let z = input_leaf(
        &mut program,
        DType::FLOAT32,
        vec![Extent::Static(positions), Extent::Static(width)],
        "x",
    );
    let fold = HadamardFold::declare(&mut program, BLOCK, width, width, 1, true, &[]);
    let root = fold
        .restore_flat(&mut program, z, width)
        .expect("restore_flat lowers");
    let mut named = vec![("x".to_owned(), random(7, (positions * width) as usize))];
    named.extend(fold_leaves(width));
    Case { program, root, named }
}

fn merge_case(kv_heads: u32, group: u32, head_dim: u32, positions: u32) -> Case {
    let mut program = Vec::new();
    let x = input_leaf(
        &mut program,
        DType::FLOAT32,
        vec![
            Extent::Static(positions),
            Extent::Static(kv_heads),
            Extent::Static(group),
            Extent::Static(head_dim),
        ],
        "x",
    );
    let root = merge_head_axes(&mut program, x, "sugd", "s", kv_heads, group, true)
        .expect("merge_head_axes lowers");
    let named = vec![(
        "x".to_owned(),
        random(11, (positions * kv_heads * group * head_dim) as usize),
    )];
    Case { program, root, named }
}

fn projection_case(width: u32, outputs: u32, positions: u32) -> Case {
    let mut program = Vec::new();
    let x = input_leaf(
        &mut program,
        DType::FLOAT32,
        vec![Extent::Static(positions), Extent::Static(width)],
        "x",
    );
    let weight = input_leaf(
        &mut program,
        DType::FLOAT32,
        vec![Extent::Static(width), Extent::Static(outputs)],
        "weight",
    );
    let fold = HadamardFold::declare(&mut program, BLOCK, width, width, 1, true, &[]);
    let rotated = fold
        .rotate_flat(&mut program, x, width, "s")
        .expect("rotate_flat lowers");
    let root = fold
        .project_folded(&mut program, rotated, weight, "s", "o")
        .expect("project_folded lowers");
    let mut named = vec![
        ("x".to_owned(), random(13, (positions * width) as usize)),
        ("weight".to_owned(), random(17, (width * outputs) as usize)),
    ];
    named.extend(fold_leaves(width));
    Case { program, root, named }
}

#[test]
fn flat_rotation_matches_cpu_at_the_embedding_width() {
    assert_parity("flat 5120 one position", &flat_case(5120, 1));
    assert_parity("flat 5120 five positions", &flat_case(5120, 5));
}

#[test]
fn flat_rotation_matches_cpu_at_the_feed_forward_width() {
    assert_parity("flat 17408", &flat_case(17408, 1));
}

#[test]
fn head_rotation_matches_cpu_for_attention_and_recurrence_heads() {
    assert_parity("heads 24x256", &heads_case(24, 256, 1));
    assert_parity("heads 48x128", &heads_case(48, 128, 1));
    assert_parity("heads 48x128 five positions", &heads_case(48, 128, 5));
}

#[test]
fn embedding_restore_matches_cpu() {
    assert_parity("restore 5120", &restore_case(5120, 1));
    assert_parity("restore 5120 five positions", &restore_case(5120, 5));
}

#[test]
fn head_merge_matches_cpu() {
    assert_parity("merge 4x6x256", &merge_case(4, 6, 256, 1));
    assert_parity("merge 16x3x128", &merge_case(16, 3, 128, 1));
}

#[test]
fn folded_projection_matches_cpu_on_float_weights() {
    assert_parity("project 5120 -> 640", &projection_case(5120, 640, 1));
}

#[test]
fn folded_projection_on_a_packed_ternary_weight_matches_its_float_decode() {
    use proxima_gguf::quant::ptq1_0::{BLOCK_BYTES, QK_PTQ1_0, dequantize, quantize};
    let (width, outputs, positions) = (5120u32, 640u32, 1u32);
    let case = projection_case(width, outputs, positions);
    let rows = random(19, (outputs * width) as usize);
    let row_bytes = width as usize / QK_PTQ1_0 * BLOCK_BYTES;
    let mut packed = vec![0u8; outputs as usize * row_bytes];
    let mut decoded = vec![0.0f32; outputs as usize * width as usize];
    for ((row, bytes), row_decoded) in rows
        .chunks_exact(width as usize)
        .zip(packed.chunks_exact_mut(row_bytes))
        .zip(decoded.chunks_exact_mut(width as usize))
    {
        quantize(row, bytes).expect("whole blocks");
        dequantize(bytes, row_decoded).expect("whole blocks");
    }
    let mut transposed = vec![0.0f32; decoded.len()];
    for output in 0..outputs as usize {
        for input in 0..width as usize {
            transposed[input * outputs as usize + output] = decoded[output * width as usize + input];
        }
    }

    let mut oracle_case = Case {
        program: case.program.clone(),
        root: case.root,
        named: case.named.clone(),
    };
    for (name, data) in &mut oracle_case.named {
        if name == "weight" {
            *data = transposed.clone();
        }
    }
    let oracle = run(Engine::Cpu, None, &oracle_case, NumericPolicy::default());

    let packed_named = |case: &Case| -> Vec<(String, Vec<f32>)> {
        case.named
            .iter()
            .filter(|(name, _)| name != "weight")
            .cloned()
            .collect()
    };
    let run_packed = |engine: Engine, driver: Option<GpuDriver>, policy: NumericPolicy| {
        let owned = packed_named(&case);
        let mut named: Vec<(&str, QuantizedBlock<'_>)> = owned
            .iter()
            .map(|(name, data)| (name.as_str(), QuantizedBlock::Float32(data)))
            .collect();
        named.push((
            "weight",
            QuantizedBlock::Packed {
                codec: Codec::Ptq1_0,
                bytes: &packed,
            },
        ));
        let mut plan = plan_named(engine, driver, &case.program, &[], &named, &[case.root], policy)
            .expect("the packed case plans");
        execute_plan_named(&mut plan, &named)
            .expect("the packed case executes")
            .root()
            .to_vec()
    };
    let cpu = run_packed(Engine::Cpu, None, NumericPolicy::default());
    let cuda = run_packed(
        Engine::Gpu,
        Some(GpuDriver::Cuda),
        NumericPolicy::bit_exact().with_reassociation(true),
    );

    let magnitude = oracle.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
    for (label, actual) in [("cpu packed", &cpu), ("cuda packed", &cuda)] {
        let worst = oracle
            .iter()
            .zip(actual)
            .map(|(want, got)| (want - got).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst / magnitude < TOLERANCE,
            "{label} vs float decode: relative={}",
            worst / magnitude
        );
    }
}

#[cfg(feature = "cached-attention-streaming")]
mod folded_attention {
    use super::*;
    use proxima_tensor::spec::{
        DenseAttentionTaps, append_gated_attention_only_with_taps, causal_mask, scalar_constant,
    };
    use proxima_tensor::{append, infer};

    const EMBEDDING: u32 = 2048;
    const KV_HEADS: u32 = 4;
    const GROUP: u32 = 6;
    const HEAD_DIM: u32 = 256;
    const ROTARY: u32 = 64;
    const PASS: u32 = HEAD_DIM - ROTARY;
    const PAIRS: u32 = ROTARY / 2;
    const CACHED_EXTENT: u64 = 8;
    const CACHED_LEN: f32 = 5.0;

    struct Attention {
        program: Vec<Op>,
        symbols: Vec<u64>,
        taps: DenseAttentionTaps,
        residual: NodeId,
    }

    fn build() -> Attention {
        let mut program = Vec::new();
        let x = input_leaf(
            &mut program,
            DType::FLOAT32,
            vec![Extent::Symbolic(0), Extent::Static(EMBEDDING)],
            "x",
        );
        let inv_dim = count_constant(&mut program, EMBEDDING);
        let eps = input_leaf(&mut program, DType::FLOAT32, vec![Extent::Symbolic(0)], "eps");
        let ones = scalar_constant(&mut program, 1.0);
        let inv_sqrt = scalar_constant(&mut program, 1.0 / (HEAD_DIM as f32).sqrt());
        let inv_head = count_constant(&mut program, HEAD_DIM);
        let rotary_shape = vec![Extent::Symbolic(0), Extent::Static(PAIRS)];
        let cos = input_leaf(&mut program, DType::FLOAT32, rotary_shape.clone(), "cos");
        let sin = input_leaf(&mut program, DType::FLOAT32, rotary_shape, "sin");
        let group_ones = append(
            &mut program,
            Op::Constant {
                dtype: DType::FLOAT32,
                shape: vec![Extent::Static(KV_HEADS), Extent::Static(GROUP)],
                value: 1.0,
            },
        );
        let (is_future, _) = causal_mask(&mut program).expect("causal mask lowers");
        let cached_len = input_leaf(&mut program, DType::FLOAT32, Vec::new(), "cached_len");
        let fold = HadamardFold::declare(
            &mut program,
            BLOCK,
            EMBEDDING,
            EMBEDDING,
            KV_HEADS,
            true,
            &[KV_HEADS * GROUP * HEAD_DIM],
        );
        let norm = input_leaf(
            &mut program,
            DType::FLOAT32,
            vec![Extent::Static(EMBEDDING)],
            "attn_norm_weight",
        );
        let head_norm = |program: &mut Vec<Op>, name: &str| {
            input_leaf(program, DType::FLOAT32, vec![Extent::Static(HEAD_DIM)], name)
        };
        let q_norm = head_norm(&mut program, "q_norm_weight");
        let k_norm = head_norm(&mut program, "k_norm_weight");
        let wq_gate = input_leaf(
            &mut program,
            DType::FLOAT32,
            vec![
                Extent::Static(EMBEDDING),
                Extent::Static(KV_HEADS * GROUP * 2 * HEAD_DIM),
            ],
            "wq_gate",
        );
        let kv_shape = vec![Extent::Static(EMBEDDING), Extent::Static(KV_HEADS * HEAD_DIM)];
        let wk = input_leaf(&mut program, DType::FLOAT32, kv_shape.clone(), "wk");
        let wv = input_leaf(&mut program, DType::FLOAT32, kv_shape, "wv");
        let wo = input_leaf(
            &mut program,
            DType::FLOAT32,
            vec![
                Extent::Static(KV_HEADS * GROUP * HEAD_DIM),
                Extent::Static(EMBEDDING),
            ],
            "wo",
        );
        let cache = |program: &mut Vec<Op>, name: &str, width: u32| {
            input_leaf(
                program,
                DType::FLOAT32,
                vec![
                    Extent::Symbolic(1),
                    Extent::Static(KV_HEADS),
                    Extent::Static(width),
                ],
                name,
            )
        };
        let k_first = cache(&mut program, "k_first_cache", PAIRS);
        let k_second = cache(&mut program, "k_second_cache", PAIRS);
        let k_pass = cache(&mut program, "k_pass_cache", PASS);
        let v_cache = cache(&mut program, "v_cache", HEAD_DIM);
        let (residual, taps) = append_gated_attention_only_with_taps(
            &mut program,
            x,
            inv_dim,
            eps,
            ones,
            inv_sqrt,
            inv_head,
            cos,
            sin,
            group_ones,
            is_future,
            cached_len,
            GROUP,
            ROTARY,
            HEAD_DIM,
            norm,
            q_norm,
            k_norm,
            wq_gate,
            wk,
            wv,
            wo,
            k_first,
            k_second,
            k_pass,
            v_cache,
            Some(&fold),
        )
        .expect("the folded attention layer lowers");
        Attention {
            program,
            symbols: vec![1, CACHED_EXTENT],
            taps,
            residual,
        }
    }

    fn data(attention: &Attention) -> Vec<(String, Vec<f32>)> {
        let shapes = infer(&attention.program, &attention.symbols).expect("infers");
        attention
            .program
            .iter()
            .enumerate()
            .filter_map(|(position, op)| match op {
                Op::Input { name: Some(name), .. } => Some((position, name.clone())),
                _ => None,
            })
            .map(|(position, name)| {
                let count: usize = shapes
                    .of(NodeId(position as u32))
                    .iter()
                    .map(|extent| *extent as usize)
                    .product();
                let values = match name.as_str() {
                    "eps" => vec![1e-5; count],
                    "cached_len" => vec![CACHED_LEN],
                    name if name.starts_with("prism.hadamard.rotation.") => {
                        sylvester(name.rsplit('.').next().unwrap().parse().unwrap())
                    }
                    name if name.starts_with("prism.hadamard.signs.") => {
                        signs(name.rsplit('.').next().unwrap().parse().unwrap())
                    }
                    _ => random(position as u64 + 1, count.max(1)),
                };
                (name, values)
            })
            .collect()
    }

    #[test]
    fn folded_dense_attention_taps_match_cpu() {
        let attention = build();
        let owned = data(&attention);
        let named: Vec<(&str, QuantizedBlock<'_>)> = owned
            .iter()
            .map(|(name, values)| (name.as_str(), QuantizedBlock::Float32(values)))
            .collect();
        let taps = &attention.taps;
        let roots = [
            ("normed", taps.normed),
            ("q_split", taps.q_split),
            ("gate_split", taps.gate_split),
            ("q_normed", taps.q_normed),
            ("k_normed", taps.k_normed),
            ("attended", taps.attended),
            ("gated_attended", taps.gated_attended),
            ("o_proj_out", taps.o_proj_out),
            ("v_new", taps.v_new),
            ("residual", attention.residual),
        ];
        let nodes: Vec<NodeId> = roots.iter().map(|(_, node)| *node).collect();
        let mut cpu_plan = plan_named(
            Engine::Cpu,
            None,
            &attention.program,
            &attention.symbols,
            &named,
            &nodes,
            NumericPolicy::default(),
        )
        .expect("cpu plans");
        let cpu = execute_plan_named(&mut cpu_plan, &named).expect("cpu runs");
        let mut cuda_plan = plan_named(
            Engine::Gpu,
            Some(GpuDriver::Cuda),
            &attention.program,
            &attention.symbols,
            &named,
            &nodes,
            NumericPolicy::bit_exact().with_reassociation(true),
        )
        .expect("cuda plans");
        let cuda = execute_plan_named(&mut cuda_plan, &named).expect("cuda runs");

        let mut offenders = Vec::new();
        for (label, node) in roots {
            let (want, _) = cpu.get(node).expect("cpu returns the tap");
            let (got, _) = cuda.get(node).expect("cuda returns the tap");
            let magnitude = want.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
            let worst = want
                .iter()
                .zip(got)
                .map(|(want, got)| (want - got).abs())
                .fold(0.0f32, f32::max);
            let relative = worst / magnitude.max(f32::MIN_POSITIVE);
            if relative.is_nan() || relative >= TOLERANCE {
                offenders.push(format!("{label}={relative}"));
            }
        }
        assert!(offenders.is_empty(), "taps where cuda disagrees with cpu: {offenders:?}");
    }
}

#[cfg(feature = "cached-attention-streaming")]
mod packed_folded_attention {
    use super::*;
    use proxima_gguf::quant::ptq1_0::{BLOCK_BYTES, QK_PTQ1_0, dequantize, quantize};
    use proxima_tensor::spec::{
        append_gated_attention_only_with_taps, causal_mask, scalar_constant,
    };
    use proxima_tensor::{append, infer};

    const EMBEDDING: u32 = 2048;
    const KV_HEADS: u32 = 4;
    const GROUP: u32 = 6;
    const HEAD_DIM: u32 = 256;
    const ROTARY: u32 = 64;
    const PAIRS: u32 = ROTARY / 2;
    const PASS: u32 = HEAD_DIM - ROTARY;
    const WEIGHTS: [&str; 4] = ["wq_flat", "wk_flat", "wv_flat", "wo_flat"];

    struct Packed {
        bytes: Vec<u8>,
        transposed: Vec<f32>,
    }

    fn pack(seed: u64, inputs: u32, outputs: u32) -> Packed {
        let rows = random(seed, (outputs * inputs) as usize);
        let row_bytes = inputs as usize / QK_PTQ1_0 * BLOCK_BYTES;
        let mut bytes = vec![0u8; outputs as usize * row_bytes];
        let mut decoded = vec![0.0f32; rows.len()];
        for ((row, packed), out) in rows
            .chunks_exact(inputs as usize)
            .zip(bytes.chunks_exact_mut(row_bytes))
            .zip(decoded.chunks_exact_mut(inputs as usize))
        {
            quantize(row, packed).expect("whole blocks");
            dequantize(packed, out).expect("whole blocks");
        }
        let mut transposed = vec![0.0f32; decoded.len()];
        for output in 0..outputs as usize {
            for input in 0..inputs as usize {
                transposed[input * outputs as usize + output] =
                    decoded[output * inputs as usize + input];
            }
        }
        Packed { bytes, transposed }
    }

    fn weight_shape(name: &str) -> (u32, u32) {
        match name {
            "wq_flat" => (EMBEDDING, KV_HEADS * GROUP * HEAD_DIM * 2),
            "wk_flat" | "wv_flat" => (EMBEDDING, KV_HEADS * HEAD_DIM),
            _ => (KV_HEADS * GROUP * HEAD_DIM, EMBEDDING),
        }
    }

    #[test]
    fn packed_folded_dense_attention_matches_its_float_decode_on_cpu_and_cuda() {
        let mut program = Vec::new();
        let x = input_leaf(
            &mut program,
            DType::FLOAT32,
            vec![Extent::Symbolic(0), Extent::Static(EMBEDDING)],
            "x",
        );
        let inv_dim = count_constant(&mut program, EMBEDDING);
        let eps = input_leaf(&mut program, DType::FLOAT32, vec![Extent::Symbolic(0)], "eps");
        let ones = scalar_constant(&mut program, 1.0);
        let inv_sqrt = scalar_constant(&mut program, 1.0 / (HEAD_DIM as f32).sqrt());
        let inv_head = count_constant(&mut program, HEAD_DIM);
        let rotary_shape = vec![Extent::Symbolic(0), Extent::Static(PAIRS)];
        let cos = input_leaf(&mut program, DType::FLOAT32, rotary_shape.clone(), "cos");
        let sin = input_leaf(&mut program, DType::FLOAT32, rotary_shape, "sin");
        let group_ones = append(
            &mut program,
            Op::Constant {
                dtype: DType::FLOAT32,
                shape: vec![Extent::Static(KV_HEADS), Extent::Static(GROUP)],
                value: 1.0,
            },
        );
        let (is_future, _) = causal_mask(&mut program).expect("causal mask lowers");
        let cached_len = input_leaf(&mut program, DType::FLOAT32, Vec::new(), "cached_len");
        let fold = HadamardFold::declare(
            &mut program,
            BLOCK,
            EMBEDDING,
            EMBEDDING,
            KV_HEADS,
            true,
            &[KV_HEADS * GROUP * HEAD_DIM],
        );
        let norm = input_leaf(
            &mut program,
            DType::FLOAT32,
            vec![Extent::Static(EMBEDDING)],
            "attn_norm_weight",
        );
        let q_norm = input_leaf(&mut program, DType::FLOAT32, vec![Extent::Static(HEAD_DIM)], "q_norm_weight");
        let k_norm = input_leaf(&mut program, DType::FLOAT32, vec![Extent::Static(HEAD_DIM)], "k_norm_weight");
        let leaf = |program: &mut Vec<Op>, name: &str| {
            let (inputs, outputs) = weight_shape(name);
            input_leaf(
                program,
                DType::FLOAT32,
                vec![Extent::Static(inputs), Extent::Static(outputs)],
                name,
            )
        };
        let wq_flat = leaf(&mut program, "wq_flat");
        let wk_flat = leaf(&mut program, "wk_flat");
        let wv_flat = leaf(&mut program, "wv_flat");
        let wo = leaf(&mut program, "wo_flat");
        let cache = |program: &mut Vec<Op>, name: &str, width: u32| {
            input_leaf(
                program,
                DType::FLOAT32,
                vec![Extent::Symbolic(1), Extent::Static(KV_HEADS), Extent::Static(width)],
                name,
            )
        };
        let k_first = cache(&mut program, "k_first_cache", PAIRS);
        let k_second = cache(&mut program, "k_second_cache", PAIRS);
        let k_pass = cache(&mut program, "k_pass_cache", PASS);
        let v_cache = cache(&mut program, "v_cache", HEAD_DIM);
        let (_, taps) = append_gated_attention_only_with_taps(
            &mut program,
            x,
            inv_dim,
            eps,
            ones,
            inv_sqrt,
            inv_head,
            cos,
            sin,
            group_ones,
            is_future,
            cached_len,
            GROUP,
            ROTARY,
            HEAD_DIM,
            norm,
            q_norm,
            k_norm,
            wq_flat,
            wk_flat,
            wv_flat,
            wo,
            k_first,
            k_second,
            k_pass,
            v_cache,
            Some(&fold),
        )
        .expect("the folded packed attention lowers");
        let symbols = vec![1u64, 8];
        let shapes = infer(&program, &symbols).expect("infers");
        let packed: Vec<(&str, Packed)> = WEIGHTS
            .iter()
            .enumerate()
            .map(|(index, name)| {
                let (inputs, outputs) = weight_shape(name);
                (*name, pack(100 + index as u64, inputs, outputs))
            })
            .collect();
        let others: Vec<(String, Vec<f32>)> = program
            .iter()
            .enumerate()
            .filter_map(|(position, op)| match op {
                Op::Input { name: Some(name), .. } if !WEIGHTS.contains(&name.as_str()) => {
                    Some((position, name.clone()))
                }
                _ => None,
            })
            .map(|(position, name)| {
                let count: usize = shapes
                    .of(NodeId(position as u32))
                    .iter()
                    .map(|extent| *extent as usize)
                    .product();
                let values = match name.as_str() {
                    "eps" => vec![1e-5; count],
                    "cached_len" => vec![5.0],
                    name if name.starts_with("prism.hadamard.rotation.") => {
                        sylvester(name.rsplit('.').next().unwrap().parse().unwrap())
                    }
                    name if name.starts_with("prism.hadamard.signs.") => {
                        signs(name.rsplit('.').next().unwrap().parse().unwrap())
                    }
                    _ => random(position as u64 + 1, count.max(1)),
                };
                (name, values)
            })
            .collect();

        let roots = [taps.q_split, taps.k_normed, taps.v_new, taps.attended, taps.o_proj_out];
        let evaluate = |engine: Engine, driver: Option<GpuDriver>, policy: NumericPolicy, as_packed: bool| {
            let mut named: Vec<(&str, QuantizedBlock<'_>)> = others
                .iter()
                .map(|(name, values)| (name.as_str(), QuantizedBlock::Float32(values)))
                .collect();
            for (name, weight) in &packed {
                named.push((
                    name,
                    if as_packed {
                        QuantizedBlock::Packed { codec: Codec::Ptq1_0, bytes: &weight.bytes }
                    } else {
                        QuantizedBlock::Float32(&weight.transposed)
                    },
                ));
            }
            let mut plan = plan_named(engine, driver, &program, &symbols, &named, &roots, policy)
                .expect("plans");
            let evaluated = execute_plan_named(&mut plan, &named).expect("runs");
            roots
                .iter()
                .map(|node| evaluated.get(*node).expect("returns the tap").0.to_vec())
                .collect::<Vec<Vec<f32>>>()
        };
        let oracle = evaluate(Engine::Cpu, None, NumericPolicy::default(), false);
        let cpu = evaluate(Engine::Cpu, None, NumericPolicy::default(), true);
        let cuda = evaluate(
            Engine::Gpu,
            Some(GpuDriver::Cuda),
            NumericPolicy::bit_exact().with_reassociation(true),
            true,
        );

        let labels = ["q_split", "k_normed", "v_new", "attended", "o_proj_out"];
        let mut offenders = Vec::new();
        for (name, actuals) in [("cpu packed", &cpu), ("cuda packed", &cuda)] {
            for (index, label) in labels.iter().enumerate() {
                let want = &oracle[index];
                let got = &actuals[index];
                let magnitude = want.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
                let worst = want
                    .iter()
                    .zip(got)
                    .map(|(want, got)| (want - got).abs())
                    .fold(0.0f32, f32::max);
                let relative = worst / magnitude.max(f32::MIN_POSITIVE);
                if relative.is_nan() || relative >= TOLERANCE {
                    offenders.push(format!("{name} {label}={relative}"));
                }
            }
        }
        assert!(offenders.is_empty(), "taps that disagree with the float decode: {offenders:?}");
    }
}
