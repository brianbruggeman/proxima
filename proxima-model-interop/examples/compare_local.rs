//! Compare one-shot logits from the CPU oracle and a selected GPU driver.
//!
//! Usage: `compare_local <model.gguf> <cuda|vulkan> <prompt>`

#![allow(clippy::expect_used)]

use std::env;
use std::fs::File;

use memmap2::{Mmap, MmapOptions};
use proxima_gguf::{MetadataValue, ParsedGguf, parse_complete};
use proxima_model_interop::{GPU_LAYERS_ALL, LoadedModel};

#[cfg(feature = "metal")]
use omega::backend::{Engine, GpuDriver};
#[cfg(feature = "metal")]
use proxima_tensor::{DType, Extent, IndexMap, NodeId, Op, ScalarOp, append, projection};

fn main() {
    let mut args = env::args().skip(1);
    let model_path = args.next().expect("model path");
    let backend = args.next().expect("cuda or vulkan");
    let prompt = args.next().expect("prompt");
    assert!(matches!(backend.as_str(), "cuda" | "vulkan"));
    let file = File::open(&model_path).expect("open model");
    // SAFETY: `file` remains alive while the read-only mapping is borrowed.
    let bytes: Mmap = unsafe { MmapOptions::new().map(&file) }.expect("map model");
    let parsed = parse_complete(&bytes).expect("parse model");
    let model = LoadedModel::load(&parsed, &bytes).expect("bind model");
    if env::var_os("PROXIMA_DISABLE_CPU_FUSION").is_some() {
        proxima_tensor::cpu::set_epilogue_fuse_enabled(false);
    }
    let layer_roots = model.layer_residual_roots().to_vec();
    if !layer_roots.is_empty() {
        let requested_roots = if env::var_os("PROXIMA_COMPARE_LAYER").is_some()
            || env::var_os("PROXIMA_COMPARE_NODE").is_some()
        {
            // A layer root is topologically after every operation in that
            // layer. Requesting the complete prefix makes the evaluator keep
            // each intermediate alive, allowing this example to identify the
            // first divergent operation instead of only reporting the layer
            // aggregate.
            let through = env::var("PROXIMA_COMPARE_NODE")
                .ok()
                .and_then(|value| value.parse::<u32>().ok())
                .map(proxima_tensor::NodeId)
                .unwrap_or(layer_roots[0]);
            model.computed_node_ids_through(through)
        } else if env::var_os("PROXIMA_FIRST_LAYER_ONLY").is_some() {
            layer_roots[..1].to_vec()
        } else {
            layer_roots.clone()
        };
        let cpu_layers = model
            .forward_node_values_on_backend(&prompt, &requested_roots, 0)
            .expect("CPU layer residuals");
        let gpu_layers = model
            .forward_node_values_on_backend(&prompt, &requested_roots, GPU_LAYERS_ALL)
            .expect("GPU layer residuals");
        if env::var_os("PROXIMA_COMPARE_LAYER").is_some()
            || env::var_os("PROXIMA_COMPARE_NODE").is_some()
        {
            let threshold = 1e-3_f64;
            let mut first_bad = None;
            let sweep = env::var_os("PROXIMA_COMPARE_SWEEP").is_some();
            for (node_id, (cpu_node, gpu_node)) in requested_roots
                .iter()
                .zip(cpu_layers.iter().zip(&gpu_layers))
            {
                if cpu_node.len() != gpu_node.len() {
                    panic!(
                        "node {} output length differs: {} vs {}",
                        node_id.0,
                        cpu_node.len(),
                        gpu_node.len()
                    );
                }
                if cpu_node.is_empty() {
                    continue;
                }
                let (max_abs, max_index) = cpu_node
                    .iter()
                    .zip(gpu_node)
                    .enumerate()
                    .map(|(index, (left, right))| (f64::from((left - right).abs()), index))
                    .max_by(|left, right| left.0.total_cmp(&right.0))
                    .expect("nonempty intermediate");
                if sweep {
                    println!(
                        "{{\"sweep_node\":{},\"kind\":{:?},\"elements\":{},\"max_abs\":{max_abs},\"max_index\":{max_index}}}",
                        node_id.0,
                        model.node_kind(*node_id),
                        cpu_node.len()
                    );
                }
                if max_abs > threshold {
                    first_bad = Some((
                        node_id.0,
                        max_abs,
                        max_index,
                        cpu_node[max_index],
                        gpu_node[max_index],
                        cpu_node.len(),
                    ));
                    break;
                }
            }
            match first_bad {
                Some((node, max_abs, max_index, cpu_value, gpu_value, elements)) => println!(
                    "{{\"first_divergent_node\":{node},\"kind\":{:?},\"name\":{:?},\"elements\":{elements},\"max_abs\":{max_abs},\"max_index\":{max_index},\"cpu\":{cpu_value},\"gpu\":{gpu_value}}}",
                    model.node_kind(proxima_tensor::NodeId(node)),
                    model.node_name(proxima_tensor::NodeId(node))
                ),
                None => println!("{{\"first_divergent_node\":null}}"),
            }
            if let Some(target) = env::var("PROXIMA_COMPARE_NODE")
                .ok()
                .and_then(|value| value.parse::<u32>().ok())
            {
                let target = proxima_tensor::NodeId(target);
                if let Some(index) = requested_roots.iter().position(|node| *node == target) {
                    let cpu_target = &cpu_layers[index];
                    let gpu_target = &gpu_layers[index];
                    let (max_abs, max_index) = cpu_target
                        .iter()
                        .zip(gpu_target)
                        .enumerate()
                        .map(|(index, (left, right))| (f64::from((left - right).abs()), index))
                        .max_by(|left, right| left.0.total_cmp(&right.0))
                        .expect("target reduction is nonempty");
                    println!(
                        "{{\"target_node\":{},\"kind\":{:?},\"name\":{:?},\"description\":{:?},\"max_abs\":{max_abs},\"max_index\":{max_index},\"cpu\":{},\"gpu\":{},\"dependencies\":{:?}}}",
                        target.0,
                        model.node_kind(target),
                        model.node_name(target),
                        model.node_description(target),
                        cpu_target[max_index],
                        gpu_target[max_index],
                        model.node_dependencies(target)
                    );
                    if target.0 == 35 {
                        let reciprocal_index = env::var("PROXIMA_COMPARE_INDEX")
                            .ok()
                            .and_then(|value| value.parse::<usize>().ok())
                            .expect("PROXIMA_COMPARE_INDEX for reciprocal probe");
                        let dependency = proxima_tensor::NodeId(34);
                        let dependency_index = requested_roots
                            .iter()
                            .position(|node| *node == dependency)
                            .expect("node 34 requested for node 35 probe");
                        let cpu_input = cpu_layers[dependency_index][reciprocal_index];
                        let gpu_input = gpu_layers[dependency_index][reciprocal_index];
                        let host_cpu_reciprocal = 1.0_f32 / cpu_input;
                        let host_gpu_reciprocal = 1.0_f32 / gpu_input;
                        let gpu_output = gpu_layers[index][reciprocal_index];
                        println!(
                            "{{\"reciprocal_probe\":true,\"index\":{reciprocal_index},\"cpu_input\":{cpu_input},\"gpu_input\":{gpu_input},\"host_cpu_reciprocal\":{host_cpu_reciprocal},\"host_gpu_reciprocal\":{host_gpu_reciprocal},\"gpu_output\":{gpu_output},\"gpu_vs_host_gpu_abs\":{},\"gpu_vs_host_cpu_abs\":{}}}",
                            (gpu_output - host_gpu_reciprocal).abs(),
                            (gpu_output - host_cpu_reciprocal).abs()
                        );
                    }
                    for dependency in model.node_dependencies(target) {
                        if let Some(dep_index) = requested_roots
                            .iter()
                            .position(|node| *node == dependency)
                        {
                            let cpu_dep = &cpu_layers[dep_index];
                            let gpu_dep = &gpu_layers[dep_index];
                            let (dep_max_abs, dep_max_index) = cpu_dep
                                .iter()
                                .zip(gpu_dep)
                                .enumerate()
                                .map(|(index, (left, right))| {
                                    (f64::from((left - right).abs()), index)
                                })
                                .max_by(|left, right| left.0.total_cmp(&right.0))
                                .expect("reduction dependency is nonempty");
                            println!(
                                "{{\"dependency_node\":{},\"kind\":{:?},\"description\":{:?},\"elements\":{},\"max_abs\":{dep_max_abs},\"max_index\":{dep_max_index},\"cpu\":{},\"gpu\":{}}}",
                                dependency.0,
                                model.node_kind(dependency),
                                model.node_description(dependency),
                                cpu_dep.len(),
                                cpu_dep[dep_max_index],
                                gpu_dep[dep_max_index]
                            );
                        }
                    }
                }
            }

            #[cfg(feature = "metal")]
            if env::var_os("PROXIMA_RECIPROCAL_REPLAY").is_some() {
                replay_reciprocal(
                    &backend,
                    &requested_roots,
                    &gpu_layers,
                );
            }
            #[cfg(feature = "metal")]
            if env::var_os("PROXIMA_SQRT_REPLAY").is_some() {
                replay_square_root(&backend, &requested_roots, &gpu_layers);
            }
            #[cfg(feature = "metal")]
            if env::var_os("PROXIMA_EPSILON_REPLAY").is_some() {
                replay_epsilon_add(
                    &backend,
                    &requested_roots,
                    &gpu_layers,
                    declared_rms_epsilon(&parsed),
                );
            }
            #[cfg(feature = "metal")]
            if env::var_os("PROXIMA_MEAN_SQUARE_REPLAY").is_some() {
                replay_mean_square(&backend, &requested_roots, &gpu_layers);
            }
            #[cfg(feature = "metal")]
            if env::var_os("PROXIMA_SUM_SQUARES_REPLAY").is_some() {
                replay_sum_squares(&backend, &requested_roots, &gpu_layers);
            }
            #[cfg(feature = "metal")]
            if env::var_os("PROXIMA_RESIDUAL_ADD_REPLAY").is_some() {
                replay_residual_add(&backend, &requested_roots, &gpu_layers);
            }
            #[cfg(feature = "metal")]
            if env::var_os("PROXIMA_NODE70_REPLAY").is_some() {
                replay_node70_reduce(
                    &backend,
                    &requested_roots,
                    &cpu_layers,
                );
            }
        } else {
        for (layer, (cpu_layer, gpu_layer)) in cpu_layers.iter().zip(&gpu_layers).enumerate() {
            assert_eq!(cpu_layer.len(), gpu_layer.len());
            let (max_abs, max_index) = cpu_layer
                .iter()
                .zip(gpu_layer)
                .enumerate()
                .map(|(index, (left, right))| (f64::from((left - right).abs()), index))
                .max_by(|left, right| left.0.total_cmp(&right.0))
                .expect("nonempty layer residual");
            println!(
                "{{\"layer\":{layer},\"root\":{},\"elements\":{},\"max_abs\":{max_abs},\"max_index\":{max_index}}}",
                layer_roots[layer].0,
                cpu_layer.len()
            );
        }
        }
    }
    let cpu = model
        .forward_logits_on_backend(&prompt, 0)
        .expect("CPU logits");
    let gpu = model
        .forward_logits_on_backend(&prompt, GPU_LAYERS_ALL)
        .expect("GPU logits");
    assert_eq!(cpu.len(), gpu.len());
    let (max_abs, max_index) = cpu
        .iter()
        .zip(&gpu)
        .enumerate()
        .map(|(index, (left, right))| (f64::from((left - right).abs()), index))
        .max_by(|left, right| left.0.total_cmp(&right.0))
        .expect("nonempty logits");
    let cpu_top = cpu
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .expect("CPU top token");
    let gpu_top = gpu
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .expect("GPU top token");
    println!(
        "{{\"model\":{model_path:?},\"backend\":{backend:?},\"logits\":{},\"max_abs\":{max_abs},\"max_index\":{max_index},\"cpu_top\":{cpu_top},\"gpu_top\":{gpu_top}}}",
        cpu.len()
    );
}

#[cfg(feature = "metal")]
fn replay_reciprocal(
    backend: &str,
    requested_roots: &[NodeId],
    gpu_layers: &[Vec<f32>],
) {
    let input_node = NodeId(0);
    let output_node = NodeId(1);
    let input_index = requested_roots
        .iter()
        .position(|node| *node == NodeId(34))
        .expect("node 34 requested for reciprocal replay");
    let output_index = requested_roots
        .iter()
        .position(|node| *node == NodeId(35))
        .expect("node 35 requested for reciprocal replay");
    let input = &gpu_layers[input_index];
    let original = &gpu_layers[output_index];
    let mut program = vec![Op::Input {
        dtype: DType::Float32,
        shape: vec![Extent::Static(input.len() as u32)],
        name: Some("reciprocal_input".to_string()),
    }];
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Reciprocal,
            operands: vec![(input_node, IndexMap::Affine(projection(1, &[0])))],
            name: None,
        },
    );
    let driver = match backend {
        "vulkan" => GpuDriver::Wgpu,
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                GpuDriver::Cuda
            }
            #[cfg(not(feature = "cuda"))]
            {
                panic!("cuda reciprocal replay requires the cuda feature")
            }
        }
        _ => panic!("unsupported reciprocal replay backend"),
    };
    let named = [("reciprocal_input", proxima_tensor::QuantizedBlock::Float32(input))];
    let mut plan = omega::backend::plan_named(
        Engine::Gpu,
        Some(driver),
        &program,
        &[],
        &named,
        &[output_node],
        proxima_tensor::NumericPolicy::bit_exact(),
    )
    .expect("plan reciprocal replay");
    let mut replayed = Vec::new();
    for _ in 0..3 {
        let evaluated = omega::backend::execute_plan_named(&mut plan, &named)
            .expect("execute reciprocal replay");
        replayed.push(evaluated.get(output_node).expect("replay output").0.to_vec());
    }
    let host: Vec<f32> = input.iter().map(|value| 1.0_f32 / value).collect();
    for (run, values) in replayed.iter().enumerate() {
        let (host_max, host_index) = max_abs_and_index(values, &host);
        let (original_max, original_index) = max_abs_and_index(values, original);
        let exact_host = values == &host;
        let exact_original = values == original;
        println!(
            "{{\"reciprocal_replay\":true,\"backend\":{backend:?},\"run\":{run},\"exact_host\":{exact_host},\"exact_original\":{exact_original},\"max_host_abs\":{host_max},\"max_host_index\":{host_index},\"max_original_abs\":{original_max},\"max_original_index\":{original_index}}}"
        );
    }
}

#[cfg(feature = "metal")]
fn replay_square_root(
    backend: &str,
    requested_roots: &[NodeId],
    gpu_layers: &[Vec<f32>],
) {
    let input_node = NodeId(0);
    let output_node = NodeId(1);
    let input_index = requested_roots
        .iter()
        .position(|node| *node == NodeId(33))
        .expect("node 33 requested for square-root replay");
    let output_index = requested_roots
        .iter()
        .position(|node| *node == NodeId(34))
        .expect("node 34 requested for square-root replay");
    let input = &gpu_layers[input_index];
    let original = &gpu_layers[output_index];
    let mut program = vec![Op::Input {
        dtype: DType::Float32,
        shape: vec![Extent::Static(input.len() as u32)],
        name: Some("square_root_input".to_string()),
    }];
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::SquareRoot,
            operands: vec![(input_node, IndexMap::Affine(projection(1, &[0])))],
            name: None,
        },
    );
    let driver = match backend {
        "vulkan" => GpuDriver::Wgpu,
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                GpuDriver::Cuda
            }
            #[cfg(not(feature = "cuda"))]
            {
                panic!("cuda square-root replay requires the cuda feature")
            }
        }
        _ => panic!("unsupported square-root replay backend"),
    };
    let named = [("square_root_input", proxima_tensor::QuantizedBlock::Float32(input))];
    let mut plan = omega::backend::plan_named(
        Engine::Gpu,
        Some(driver),
        &program,
        &[],
        &named,
        &[output_node],
        proxima_tensor::NumericPolicy::bit_exact(),
    )
    .expect("plan square-root replay");
    let mut replayed = Vec::new();
    for _ in 0..3 {
        let evaluated = omega::backend::execute_plan_named(&mut plan, &named)
            .expect("execute square-root replay");
        replayed.push(evaluated.get(output_node).expect("replay output").0.to_vec());
    }
    let host: Vec<f32> = input.iter().map(|value| value.sqrt()).collect();
    for (run, values) in replayed.iter().enumerate() {
        let (host_max, host_index) = max_abs_and_index(values, &host);
        let (original_max, original_index) = max_abs_and_index(values, original);
        let exact_host = values == &host;
        let exact_original = values == original;
        println!(
            "{{\"square_root_replay\":true,\"backend\":{backend:?},\"run\":{run},\"exact_host\":{exact_host},\"exact_original\":{exact_original},\"max_host_abs\":{host_max},\"max_host_index\":{host_index},\"max_original_abs\":{original_max},\"max_original_index\":{original_index}}}"
        );
    }
}

#[cfg(feature = "metal")]
fn replay_epsilon_add(
    backend: &str,
    requested_roots: &[NodeId],
    gpu_layers: &[Vec<f32>],
    epsilon: f32,
) {
    let input_node = NodeId(0);
    let epsilon_node = NodeId(1);
    let output_node = NodeId(2);
    let input_index = requested_roots
        .iter()
        .position(|node| *node == NodeId(32))
        .expect("node 32 requested for epsilon replay");
    let output_index = requested_roots
        .iter()
        .position(|node| *node == NodeId(33))
        .expect("node 33 requested for epsilon replay");
    let input = &gpu_layers[input_index];
    let original = &gpu_layers[output_index];
    let epsilon_values = vec![epsilon; input.len()];
    let mut program = vec![Op::Input {
        dtype: DType::Float32,
        shape: vec![Extent::Static(input.len() as u32)],
        name: Some("epsilon_input".to_string()),
    }];
    append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(input.len() as u32)],
            name: Some("epsilon_value".to_string()),
        },
    );
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: vec![
                (input_node, IndexMap::Affine(projection(1, &[0]))),
                (epsilon_node, IndexMap::Affine(projection(1, &[0]))),
            ],
            name: None,
        },
    );
    let driver = match backend {
        "vulkan" => GpuDriver::Wgpu,
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                GpuDriver::Cuda
            }
            #[cfg(not(feature = "cuda"))]
            {
                panic!("cuda epsilon replay requires the cuda feature")
            }
        }
        _ => panic!("unsupported epsilon replay backend"),
    };
    let named = [
        (
            "epsilon_input",
            proxima_tensor::QuantizedBlock::Float32(input),
        ),
        (
            "epsilon_value",
            proxima_tensor::QuantizedBlock::Float32(&epsilon_values),
        ),
    ];
    let mut plan = omega::backend::plan_named(
        Engine::Gpu,
        Some(driver),
        &program,
        &[],
        &named,
        &[output_node],
        proxima_tensor::NumericPolicy::bit_exact(),
    )
    .expect("plan epsilon replay");
    let mut replayed = Vec::new();
    for _ in 0..3 {
        let evaluated = omega::backend::execute_plan_named(&mut plan, &named)
            .expect("execute epsilon replay");
        replayed.push(evaluated.get(output_node).expect("replay output").0.to_vec());
    }
    let host: Vec<f32> = input.iter().map(|value| value + epsilon).collect();
    for (run, values) in replayed.iter().enumerate() {
        let (host_max, host_index) = max_abs_and_index(values, &host);
        let (original_max, original_index) = max_abs_and_index(values, original);
        let exact_host = values == &host;
        let exact_original = values == original;
        println!(
            "{{\"epsilon_replay\":true,\"backend\":{backend:?},\"epsilon\":{epsilon},\"run\":{run},\"exact_host\":{exact_host},\"exact_original\":{exact_original},\"max_host_abs\":{host_max},\"max_host_index\":{host_index},\"max_original_abs\":{original_max},\"max_original_index\":{original_index}}}"
        );
    }
}

#[cfg(feature = "metal")]
fn replay_mean_square(
    backend: &str,
    requested_roots: &[NodeId],
    gpu_layers: &[Vec<f32>],
) {
    let input_node = NodeId(0);
    let constant_node = NodeId(1);
    let output_node = NodeId(2);
    let input_index = requested_roots
        .iter()
        .position(|node| *node == NodeId(31))
        .expect("node 31 requested for mean-square replay");
    let constant_index = requested_roots
        .iter()
        .position(|node| *node == NodeId(3))
        .expect("node 3 requested for mean-square replay");
    let output_index = requested_roots
        .iter()
        .position(|node| *node == NodeId(32))
        .expect("node 32 requested for mean-square replay");
    let input = &gpu_layers[input_index];
    let constant = gpu_layers[constant_index][0];
    let original = &gpu_layers[output_index];
    let mut program = vec![Op::Input {
        dtype: DType::Float32,
        shape: vec![Extent::Static(input.len() as u32)],
        name: Some("mean_square_input".to_string()),
    }];
    append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: Vec::new(),
            value: constant,
        },
    );
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (input_node, IndexMap::Affine(projection(1, &[0]))),
                (constant_node, IndexMap::Affine(projection(1, &[]))),
            ],
            name: None,
        },
    );
    let driver = match backend {
        "vulkan" => GpuDriver::Wgpu,
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                GpuDriver::Cuda
            }
            #[cfg(not(feature = "cuda"))]
            {
                panic!("cuda mean-square replay requires the cuda feature")
            }
        }
        _ => panic!("unsupported mean-square replay backend"),
    };
    let named = [("mean_square_input", proxima_tensor::QuantizedBlock::Float32(input))];
    let mut plan = omega::backend::plan_named(
        Engine::Gpu,
        Some(driver),
        &program,
        &[],
        &named,
        &[output_node],
        proxima_tensor::NumericPolicy::bit_exact(),
    )
    .expect("plan mean-square replay");
    let mut replayed = Vec::new();
    for _ in 0..3 {
        let evaluated = omega::backend::execute_plan_named(&mut plan, &named)
            .expect("execute mean-square replay");
        replayed.push(evaluated.get(output_node).expect("replay output").0.to_vec());
    }
    let host: Vec<f32> = input.iter().map(|value| value * constant).collect();
    for (run, values) in replayed.iter().enumerate() {
        let (host_max, host_index) = max_abs_and_index(values, &host);
        let (original_max, original_index) = max_abs_and_index(values, original);
        let exact_host = values == &host;
        let exact_original = values == original;
        println!(
            "{{\"mean_square_replay\":true,\"backend\":{backend:?},\"constant\":{constant},\"run\":{run},\"exact_host\":{exact_host},\"exact_original\":{exact_original},\"max_host_abs\":{host_max},\"max_host_index\":{host_index},\"max_original_abs\":{original_max},\"max_original_index\":{original_index}}}"
        );
    }
}

#[cfg(feature = "metal")]
fn replay_sum_squares(
    backend: &str,
    requested_roots: &[NodeId],
    gpu_layers: &[Vec<f32>],
) {
    let input_node = NodeId(0);
    let output_node = NodeId(1);
    let input_index = requested_roots
        .iter()
        .position(|node| *node == NodeId(30))
        .expect("node 30 requested for sum-squares replay");
    let output_index = requested_roots
        .iter()
        .position(|node| *node == NodeId(31))
        .expect("node 31 requested for sum-squares replay");
    let input = &gpu_layers[input_index];
    let original = &gpu_layers[output_index];
    let row_count = original.len();
    let column_count = input
        .len()
        .checked_div(row_count)
        .expect("sum-squares rows divide input length");
    let mut program = vec![Op::Input {
        dtype: DType::Float32,
        shape: vec![
            Extent::Static(row_count as u32),
            Extent::Static(column_count as u32),
        ],
        name: Some("sum_squares_input".to_string()),
    }];
    append(
        &mut program,
        Op::Reduce(proxima_tensor::Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: proxima_tensor::ReduceInit::Zero,
            operand: input_node,
            in_map: IndexMap::Affine(projection(2, &[0, 1])),
            out_map: IndexMap::Affine(projection(2, &[0])),
            keep: proxima_tensor::Keep::Reduce,
            name: None,
        }),
    );
    let driver = match backend {
        "vulkan" => GpuDriver::Wgpu,
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                GpuDriver::Cuda
            }
            #[cfg(not(feature = "cuda"))]
            {
                panic!("cuda sum-squares replay requires the cuda feature")
            }
        }
        _ => panic!("unsupported sum-squares replay backend"),
    };
    let named = [("sum_squares_input", proxima_tensor::QuantizedBlock::Float32(input))];
    let mut plan = omega::backend::plan_named(
        Engine::Gpu,
        Some(driver),
        &program,
        &[],
        &named,
        &[output_node],
        proxima_tensor::NumericPolicy::bit_exact(),
    )
    .expect("plan sum-squares replay");
    let mut replayed = Vec::new();
    for _ in 0..3 {
        let evaluated = omega::backend::execute_plan_named(&mut plan, &named)
            .expect("execute sum-squares replay");
        replayed.push(evaluated.get(output_node).expect("replay output").0.to_vec());
    }
    let host: Vec<f32> = input
        .chunks_exact(column_count)
        .map(|row| row.iter().copied().fold(0.0_f32, |sum, value| sum + value))
        .collect();
    let canonical_tree = input
        .chunks_exact(column_count)
        .map(|row| canonical_warp_sum(row))
        .collect::<Vec<_>>();
    for (run, values) in replayed.iter().enumerate() {
        let (host_max, host_index) = max_abs_and_index(values, &host);
        let (original_max, original_index) = max_abs_and_index(values, original);
        let (tree_max, tree_index) = max_abs_and_index(values, &canonical_tree);
        let exact_host = values == &host;
        let exact_original = values == original;
        let exact_tree = values == &canonical_tree;
        println!(
            "{{\"sum_squares_replay\":true,\"backend\":{backend:?},\"rows\":{row_count},\"columns\":{column_count},\"run\":{run},\"exact_host\":{exact_host},\"exact_original\":{exact_original},\"exact_canonical_tree\":{exact_tree},\"max_host_abs\":{host_max},\"max_host_index\":{host_index},\"max_original_abs\":{original_max},\"max_original_index\":{original_index},\"max_tree_abs\":{tree_max},\"max_tree_index\":{tree_index}}}"
        );
    }
}

#[cfg(feature = "metal")]
fn replay_residual_add(
    backend: &str,
    requested_roots: &[NodeId],
    gpu_layers: &[Vec<f32>],
) {
    let left_node = NodeId(0);
    let right_node = NodeId(1);
    let output_node = NodeId(2);
    let left_index = requested_roots
        .iter()
        .position(|node| *node == NodeId(68))
        .expect("node 68 requested for residual-add replay");
    let right_index = requested_roots
        .iter()
        .position(|node| *node == NodeId(70))
        .expect("node 70 requested for residual-add replay");
    let output_index = requested_roots
        .iter()
        .position(|node| *node == NodeId(71))
        .expect("node 71 requested for residual-add replay");
    let left = &gpu_layers[left_index];
    let right = &gpu_layers[right_index];
    assert_eq!(left.len(), right.len());
    let original = &gpu_layers[output_index];
    let mut program = vec![Op::Input {
        dtype: DType::Float32,
        shape: vec![Extent::Static(left.len() as u32)],
        name: Some("residual_add_left".to_string()),
    }];
    append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(right.len() as u32)],
            name: Some("residual_add_right".to_string()),
        },
    );
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: vec![
                (left_node, IndexMap::Affine(projection(1, &[0]))),
                (right_node, IndexMap::Affine(projection(1, &[0]))),
            ],
            name: None,
        },
    );
    let driver = match backend {
        "vulkan" => GpuDriver::Wgpu,
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                GpuDriver::Cuda
            }
            #[cfg(not(feature = "cuda"))]
            {
                panic!("cuda residual-add replay requires the cuda feature")
            }
        }
        _ => panic!("unsupported residual-add replay backend"),
    };
    let named = [
        (
            "residual_add_left",
            proxima_tensor::QuantizedBlock::Float32(left),
        ),
        (
            "residual_add_right",
            proxima_tensor::QuantizedBlock::Float32(right),
        ),
    ];
    let mut plan = omega::backend::plan_named(
        Engine::Gpu,
        Some(driver),
        &program,
        &[],
        &named,
        &[output_node],
        proxima_tensor::NumericPolicy::bit_exact(),
    )
    .expect("plan residual-add replay");
    let mut replayed = Vec::new();
    for _ in 0..3 {
        let evaluated = omega::backend::execute_plan_named(&mut plan, &named)
            .expect("execute residual-add replay");
        replayed.push(evaluated.get(output_node).expect("replay output").0.to_vec());
    }
    let host: Vec<f32> = left
        .iter()
        .zip(right)
        .map(|(left, right)| left + right)
        .collect();
    for (run, values) in replayed.iter().enumerate() {
        let (host_max, host_index) = max_abs_and_index(values, &host);
        let (original_max, original_index) = max_abs_and_index(values, original);
        let exact_host = values == &host;
        let exact_original = values == original;
        println!(
            "{{\"residual_add_replay\":true,\"backend\":{backend:?},\"elements\":{},\"run\":{run},\"exact_host\":{exact_host},\"exact_original\":{exact_original},\"max_host_abs\":{host_max},\"max_host_index\":{host_index},\"max_original_abs\":{original_max},\"max_original_index\":{original_index}}}",
            left.len()
        );
    }
}

#[cfg(feature = "metal")]
fn replay_node70_reduce(
    backend: &str,
    requested_roots: &[NodeId],
    cpu_layers: &[Vec<f32>],
) {
    let input_node = NodeId(0);
    let output_node = NodeId(1);
    let input_index = requested_roots
        .iter()
        .position(|node| *node == NodeId(69))
        .expect("node 69 requested for node-70 replay");
    let output_index = requested_roots
        .iter()
        .position(|node| *node == NodeId(70))
        .expect("node 70 requested for node-70 replay");
    let input = &cpu_layers[input_index];
    let original = &cpu_layers[output_index];
    let row_count = original.len();
    let column_count = input
        .len()
        .checked_div(row_count)
        .expect("node-70 rows divide input length");
    let mut program = vec![Op::Input {
        dtype: DType::Float32,
        shape: vec![
            Extent::Static(row_count as u32),
            Extent::Static(1),
            Extent::Static(1),
            Extent::Static(1),
            Extent::Static(column_count as u32),
        ],
        name: Some("node70_input".to_string()),
    }];
    append(
        &mut program,
        Op::Reduce(proxima_tensor::Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: proxima_tensor::ReduceInit::Zero,
            operand: input_node,
            in_map: IndexMap::Affine(projection(5, &[0, 1, 2, 3, 4])),
            out_map: IndexMap::Affine(projection(5, &[0, 1, 2, 3])),
            keep: proxima_tensor::Keep::Reduce,
            name: None,
        }),
    );
    let named = [("node70_input", proxima_tensor::QuantizedBlock::Float32(input))];
    let mut cpu_plan = omega::backend::plan_named(
        Engine::Cpu,
        None,
        &program,
        &[],
        &named,
        &[output_node],
        proxima_tensor::NumericPolicy::bit_exact(),
    )
    .expect("plan CPU node-70 replay");
    let mut cpu_replayed = Vec::new();
    for _ in 0..3 {
        let evaluated = omega::backend::execute_plan_named(&mut cpu_plan, &named)
            .expect("execute CPU node-70 replay");
        cpu_replayed.push(evaluated.get(output_node).expect("CPU replay output").0.to_vec());
    }
    let driver = match backend {
        "vulkan" => GpuDriver::Wgpu,
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                GpuDriver::Cuda
            }
            #[cfg(not(feature = "cuda"))]
            {
                panic!("cuda node-70 replay requires the cuda feature")
            }
        }
        _ => panic!("unsupported node-70 replay backend"),
    };
    let mut gpu_plan = omega::backend::plan_named(
        Engine::Gpu,
        Some(driver),
        &program,
        &[],
        &named,
        &[output_node],
        proxima_tensor::NumericPolicy::bit_exact(),
    )
    .expect("plan GPU node-70 replay");
    let mut gpu_replayed = Vec::new();
    for _ in 0..3 {
        let evaluated = omega::backend::execute_plan_named(&mut gpu_plan, &named)
            .expect("execute GPU node-70 replay");
        gpu_replayed.push(evaluated.get(output_node).expect("GPU replay output").0.to_vec());
    }
    for (run, (cpu_values, gpu_values)) in cpu_replayed.iter().zip(&gpu_replayed).enumerate() {
        let (cpu_graph_max, cpu_graph_index) = max_abs_and_index(cpu_values, original);
        let (cross_max, cross_index) = max_abs_and_index(cpu_values, gpu_values);
        let exact_cpu_graph = cpu_values == original;
        let exact_cross_backend = cpu_values == gpu_values;
        println!(
            "{{\"node70_replay\":true,\"backend\":{backend:?},\"rows\":{row_count},\"columns\":{column_count},\"run\":{run},\"exact_cpu_graph\":{exact_cpu_graph},\"exact_cross_backend\":{exact_cross_backend},\"max_cpu_graph_abs\":{cpu_graph_max},\"max_cpu_graph_index\":{cpu_graph_index},\"max_cross_backend_abs\":{cross_max},\"max_cross_backend_index\":{cross_index}}}"
        );
    }
}

#[cfg(feature = "metal")]
fn canonical_warp_sum(row: &[f32]) -> f32 {
    const WARP_SIZE: usize = 32;
    let mut partial = [0.0_f32; WARP_SIZE];
    for (lane, value) in row.iter().copied().enumerate() {
        partial[lane % WARP_SIZE] += value;
    }
    for shift in [16, 8, 4, 2, 1] {
        let previous = partial;
        for lane in 0..(WARP_SIZE - shift) {
            partial[lane] = previous[lane] + previous[lane + shift];
        }
    }
    partial[0]
}

#[cfg(feature = "metal")]
fn declared_rms_epsilon(parsed: &ParsedGguf) -> f32 {
    let architecture = parsed
        .metadata_value("general.architecture")
        .and_then(MetadataValue::as_str)
        .expect("architecture metadata");
    let key = format!("{architecture}.attention.layer_norm_rms_epsilon");
    match parsed.metadata_value(&key) {
        Some(MetadataValue::F32(value)) => *value,
        Some(MetadataValue::F64(value)) => *value as f32,
        _ => 1e-6,
    }
}

#[cfg(feature = "metal")]
fn max_abs_and_index(left: &[f32], right: &[f32]) -> (f32, usize) {
    left.iter()
        .zip(right)
        .enumerate()
        .map(|(index, (left, right))| ((left - right).abs(), index))
        .max_by(|left, right| left.0.total_cmp(&right.0))
        .expect("nonempty reciprocal comparison")
}
