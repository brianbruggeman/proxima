use super::*;
use crate::config::ResourceConstraints;
use proxima_primitives::Codec;
use proxima_tensor::map::{self, IndexMap};
use proxima_tensor::{DType, Extent, ScalarOp, bind_unfused, bind_with_fusion, infer};
#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    feature = "wgpu-backend",
    feature = "cuda-driver"
))]
use proxima_tensor::{Keep, Reduce, ReduceInit};

fn cpu_plan(plan: &Plan) -> &CpuPlan {
    match plan {
        Plan::Cpu(cpu) => cpu,
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Plan::Metal(_) => panic!("cpu requested"),
        #[cfg(feature = "wgpu-backend")]
        Plan::Wgpu(_) => panic!("cpu requested"),
        #[cfg(feature = "cuda-driver")]
        Plan::Cuda(_) => panic!("cpu requested"),
    }
}

#[test]
fn config_plan_cpu_executes_named_two_outputs() {
    let (program, outputs) = fixture();
    let named = named(&[1.0, -2.0, 3.0, -4.0], &[10.0, 20.0, 30.0, 40.0]);
    let (mut plan, report) =
        plan_named_with_config(&RuntimeConfig::default(), &program, &[], &named, &outputs)
            .expect("configured CPU fixture must plan");
    assert_eq!(report.len(), 1);
    assert_eq!(report[0].resolved_op_count, 2);
    assert_outputs(
        &execute_plan_named(&mut plan, &named).expect("CPU fixture must evaluate"),
        &[11.0, 18.0, 33.0, 36.0],
        &[-11.0, -18.0, -33.0, -36.0],
    );
}

#[test]
fn config_plan_cpu_reuses_same_shape_with_fresh_data() {
    let (program, outputs) = fixture();
    let first = named(&[1.0, -2.0, 3.0, -4.0], &[10.0, 20.0, 30.0, 40.0]);
    let (mut plan, _) =
        plan_named_with_config(&RuntimeConfig::default(), &program, &[], &first, &outputs)
            .expect("CPU repeat fixture must plan");
    assert_outputs(
        &execute_plan_named(&mut plan, &first).expect("first call"),
        &[11.0, 18.0, 33.0, 36.0],
        &[-11.0, -18.0, -33.0, -36.0],
    );
    let second = named(&[2.0, -1.0, 4.0, -3.0], &[10.0, 20.0, 30.0, 40.0]);
    assert_outputs(
        &execute_plan_named(&mut plan, &second).expect("repeat call"),
        &[12.0, 19.0, 34.0, 37.0],
        &[-12.0, -19.0, -34.0, -37.0],
    );
}

#[test]
fn config_plan_cpu_exact_flag_reaches_evaluator() {
    let (program, outputs) = fixture();
    let named = named(&[1.0, -2.0, 3.0, -4.0], &[10.0, 20.0, 30.0, 40.0]);
    for exact in [false, true] {
        let config = RuntimeConfig::builder()
            .with_exact_activations(exact)
            .build()
            .expect("boolean activation permission must validate");
        let (mut plan, _) = plan_named_with_config(&config, &program, &[], &named, &outputs)
            .expect("activation fixture must plan");
        let execution = cpu_plan(&plan)
            .execution_config
            .as_ref()
            .expect("configured evaluator options retained");
        assert_eq!(execution.exact_activations, exact);
        assert_eq!(execution.numeric, NumericPolicy::bit_exact());
        assert_outputs(
            &execute_plan_named(&mut plan, &named).expect("activation policy fixture"),
            &[11.0, 18.0, 33.0, 36.0],
            &[-11.0, -18.0, -33.0, -36.0],
        );
    }
}

#[test]
fn config_plan_numeric_bits_reach_cpu_binder() {
    let (program, outputs) = fixture();
    let named = named(&[1.0, -2.0, 3.0, -4.0], &[10.0, 20.0, 30.0, 40.0]);
    let mut numeric = NumericPolicy::bit_exact();
    numeric.contraction = true;
    numeric.approx_functions = true;
    let config = RuntimeConfig::builder()
        .with_numeric(numeric)
        .build()
        .expect("independent permissions must validate");
    let (plan, _) = plan_named_with_config(&config, &program, &[], &named, &outputs)
        .expect("numeric fixture must plan");
    let actual = cpu_plan(&plan)
        .execution_config
        .as_ref()
        .expect("stored binder options")
        .numeric;
    assert!(actual.contraction);
    assert!(actual.approx_functions);
    assert!(!actual.reassociation);
    assert!(!actual.nan_assumptions);
    assert!(!actual.signed_zero);
    assert!(!actual.epilogue_sources);
}

#[test]
fn config_plan_fusion_false_keeps_chain_nodes() {
    let (program, outputs) = fixture();
    let named = named(&[1.0, -2.0, 3.0, -4.0], &[10.0, 20.0, 30.0, 40.0]);
    let (plan, report) =
        plan_named_with_config(&RuntimeConfig::default(), &program, &[], &named, &outputs)
            .expect("unfused fixture must plan");
    assert!(
        !cpu_plan(&plan)
            .execution_config
            .as_ref()
            .expect("stored config")
            .allow_fusion
    );
    assert_eq!(report[0].resolved_op_count, 2);
    let shapes = infer(&program, &[]).expect("literal shapes");
    let bound = bind_unfused(&program, &shapes, &outputs, NumericPolicy::bit_exact())
        .expect("unfused chain");
    assert_eq!(
        bound.iter().map(|bound| bound.node).collect::<Vec<_>>(),
        outputs
    );
    assert_eq!(bound[0].element_body().steps[0].op, ScalarOp::Add);
    assert_eq!(bound[1].element_body().steps[0].op, ScalarOp::Negate);
    let fused = bind_with_fusion(
        &program,
        &shapes,
        &[NodeId(3)],
        true,
        NumericPolicy::bit_exact(),
    )
    .expect("enabled chain");
    assert_eq!(fused.len(), 1);
    assert_eq!(fused[0].element_body().steps.len(), 2);
}

#[test]
fn config_plan_explicit_unavailable_driver_does_not_fallback() {
    #[cfg(all(feature = "cuda-driver", target_os = "macos"))]
    {
        let error = CudaDriver::new(0)
            .err()
            .expect("Mac fixture has no CUDA driver");
        println!("cuda_availability_probe={error:?}");
        assert!(matches!(error, CudaDriverError::Driver(_)));
    }
    #[cfg(not(feature = "wgpu-backend"))]
    {
        let (program, outputs) = fixture();
        let named = named(&[1.0, -2.0, 3.0, -4.0], &[10.0, 20.0, 30.0, 40.0]);
        let config = RuntimeConfig::builder()
            .with_engine("gpu")
            .with_gpu_driver("wgpu")
            .build()
            .expect("portable driver config");
        assert!(matches!(
            plan_named_with_config(&config, &program, &[], &named, &outputs),
            Err(BackendError::NotCompiled {
                backend: "wgpu",
                feature: "wgpu-backend"
            })
        ));
    }
    #[cfg(feature = "wgpu-backend")]
    assert!(matches!(
        map_wgpu_configuration_error(WgpuError::NoAdapter),
        BackendError::Wgpu(WgpuError::NoAdapter)
    ));
}

#[test]
fn config_plan_npu_ane_vulkan_names_are_rejected() {
    for name in ["npu", "ane", "vulkan"] {
        let result = RuntimeConfig::builder().with_engine(name).build();
        let error = result.expect_err("absent engine driver must reject");
        assert!(error.to_string().contains(name));
    }
}

#[test]
fn config_plan_changed_input_codec_fails_before_mutation() {
    let (program, outputs) = fixture();
    let first = named(&[1.0, -2.0, 3.0, -4.0], &[10.0, 20.0, 30.0, 40.0]);
    let (mut plan, _) =
        plan_named_with_config(&RuntimeConfig::default(), &program, &[], &first, &outputs)
            .expect("input contract fixture");
    let before = cpu_plan(&plan)
        .free_buffers
        .iter()
        .map(Vec::capacity)
        .collect::<Vec<_>>();
    let changed = [
        (
            "input",
            QuantizedBlock::Packed {
                codec: Codec::Float16,
                bytes: &[0; 8],
            },
        ),
        ("bias", QuantizedBlock::Float32(&[10.0, 20.0, 30.0, 40.0])),
    ];
    assert!(matches!(
        execute_plan_named(&mut plan, &changed),
        Err(BackendError::InputContractChanged {
            node: NodeId(0),
            ..
        })
    ));
    assert_eq!(
        cpu_plan(&plan)
            .free_buffers
            .iter()
            .map(Vec::capacity)
            .collect::<Vec<_>>(),
        before
    );
    assert!(cpu_plan(&plan).validated_weight_nodes.is_none());
}

#[test]
fn config_plan_resource_sum_checks_overflow() {
    assert_eq!(checked_resource_sum([u64::MAX, 1]), None);
    assert_eq!(checked_resource_sum([16, 32, 8]), Some(56));
}

#[test]
fn config_plan_resource_replacement_counts_overlap() {
    let mut resources = BTreeMap::from([(1, 16_u64), (2, 32), (3, 8)]);
    resources.insert(2, 32);
    assert_eq!(checked_resource_sum(resources.values().copied()), Some(56));
    assert_eq!(
        checked_resource_sum(resources.values().copied().chain([64])),
        Some(120)
    );
    resources.insert(2, 64);
    assert_eq!(checked_resource_sum(resources.values().copied()), Some(88));
    resources.remove(&3);
    assert_eq!(checked_resource_sum(resources.values().copied()), Some(80));
}

#[test]
fn config_plan_device_single_cap_checks_reused_capacity() {
    let logical = 16_u64;
    let physical = 32_u64;
    let limit = 24_u64;
    assert!(logical <= limit);
    assert!(physical > limit);
    let rejection = BackendError::ResourceLimit {
        backend: "fake_device",
        field: "max_single_device_buffer_bytes",
        requested: physical,
        limit,
    };
    assert!(matches!(
        rejection,
        BackendError::ResourceLimit {
            requested: 32,
            limit: 24,
            ..
        }
    ));
}

#[test]
fn config_plan_output_negative_control_detects_wrong_vector() {
    let expected = [11.0, 18.0, 33.0, 36.0];
    assert_eq!(
        mismatch_positions(&expected, &expected),
        Vec::<usize>::new()
    );
    assert_eq!(
        mismatch_positions(&expected, &[11.0, 18.0, 33.0, 37.0]),
        [3]
    );
}

fn sample_row(engine: Engine, driver: Option<GpuDriver>, samples: &[u128]) -> CandidateMeasurement {
    let mut row = CandidateMeasurement::new(engine, driver);
    row.execution_ns = samples.to_vec();
    row.execution_sum_ns = Some(
        samples
            .iter()
            .try_fold(0_u128, |sum, sample| sum.checked_add(*sample))
            .expect("literal sample sum"),
    );
    row
}

#[test]
fn config_selection_configured_executes_no_trials() {
    let (program, outputs) = fixture();
    let named = named(&[1.0, -2.0, 3.0, -4.0], &[10.0, 20.0, 30.0, 40.0]);
    let (_, report) =
        plan_named_with_config(&RuntimeConfig::default(), &program, &[], &named, &outputs)
            .expect("configured mode");
    assert_eq!(report.len(), 1);
    assert!(report[0].warmup_ns.is_empty());
    assert!(report[0].execution_ns.is_empty());
    assert_eq!(report[0].execution_sum_ns, None);
    assert!(report[0].selected);
    #[cfg(not(any(
        all(feature = "metal", target_os = "macos"),
        feature = "wgpu-backend",
        feature = "cuda-driver"
    )))]
    {
        let mut config = RuntimeConfig::default();
        config.selection.mode = SelectionMode::MinimumMeasuredWallTime;
        let (mut plan, report) = plan_named_with_config(&config, &program, &[], &named, &outputs)
            .expect("CPU measured selection must sample the real evaluator");
        assert_eq!(report.len(), 1);
        assert_eq!(report[0].warmup_ns.len(), 1);
        assert_eq!(report[0].execution_ns.len(), 5);
        assert_eq!(
            report[0].execution_sum_ns,
            report[0]
                .execution_ns
                .iter()
                .try_fold(0_u128, |sum, sample| sum.checked_add(*sample))
        );
        assert!(report[0].selected);
        println!("cpu_selection_raw={report:?}");
        assert_outputs(
            &execute_plan_named(&mut plan, &named).expect("selected CPU payload"),
            &[11.0, 18.0, 33.0, 36.0],
            &[-11.0, -18.0, -33.0, -36.0],
        );
        config.engine = "gpu".into();
        config.gpu_driver = "wgpu".into();
        let (_, report) = plan_named_with_config(&config, &program, &[], &named, &outputs)
            .expect("measured mode must retain an unavailable preferred arm and sample CPU");
        assert_eq!(report.len(), 2);
        assert!(report[0].rejection.is_some());
        assert_eq!(report[0].driver, Some(GpuDriver::Wgpu));
        assert!(report[1].selected);
    }
}

#[test]
fn config_selection_stable_order_deduplicates_configured() {
    let mut names = vec!["cpu", "metal", "wgpu", "cuda"];
    prefer_candidate(&mut names, &"wgpu");
    assert_eq!(names, ["wgpu", "cpu", "metal", "cuda"]);
    assert_eq!(names.iter().collect::<BTreeSet<_>>().len(), 4);
}

#[test]
fn config_selection_mean_uses_raw_sample_order() {
    let cpu = sample_row(Engine::Cpu, None, &[9, 2, 5]);
    let metal = sample_row(Engine::Gpu, Some(GpuDriver::Metal), &[4, 4, 4]);
    assert!(candidate_is_better(&metal, &cpu));
    assert!(!candidate_is_better(&cpu, &metal));
    assert_eq!(cpu.execution_ns, [9, 2, 5]);
    assert_eq!(metal.execution_ns, [4, 4, 4]);
    assert_eq!(cpu.execution_sum_ns, Some(16));
    assert_eq!(metal.execution_sum_ns, Some(12));
}

#[test]
fn config_selection_tie_uses_stable_order() {
    let cpu = sample_row(Engine::Cpu, None, &[3, 5, 4]);
    let metal = sample_row(Engine::Gpu, Some(GpuDriver::Metal), &[4, 3, 5]);
    assert!(!candidate_is_better(&metal, &cpu));
    assert!(!candidate_is_better(&cpu, &metal));
    assert_eq!(cpu.execution_sum_ns, Some(12));
    assert_eq!(metal.execution_sum_ns, Some(12));
}

#[test]
fn config_selection_rejections_are_retained_and_empty_set_errors() {
    let mut rows = Vec::new();
    for (engine, driver, name, reason) in [
        (Engine::Cpu, None, "cpu", "resource refusal"),
        (Engine::Gpu, Some(GpuDriver::Metal), "metal", "absent"),
        (
            Engine::Gpu,
            Some(GpuDriver::Wgpu),
            "wgpu",
            "unsupported operation",
        ),
        (Engine::Gpu, None, "cuda", "not compiled"),
    ] {
        let mut row = CandidateMeasurement::new(engine, driver);
        row.device_identity = name.into();
        row.rejection = Some(reason.into());
        rows.push(row);
    }
    let error = BackendError::NoAdmissibleCandidate { candidates: rows };
    let BackendError::NoAdmissibleCandidate { candidates } = error else {
        panic!("negative selector fixture")
    };
    assert_eq!(candidates.len(), 4);
    assert_eq!(
        candidates
            .iter()
            .map(|row| row.device_identity.as_str())
            .collect::<Vec<_>>(),
        ["cpu", "metal", "wgpu", "cuda"]
    );
    assert!(
        candidates
            .iter()
            .all(|row| row.rejection.is_some() && !row.selected && row.execution_ns.is_empty())
    );
}

fn fixture() -> (Vec<Op>, [NodeId; 2]) {
    let program = vec![
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4)],
            name: Some("input".into()),
        },
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(4)],
            name: Some("bias".into()),
        },
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: vec![
                (NodeId(0), IndexMap::Affine(map::projection(1, &[0]))),
                (NodeId(1), IndexMap::Affine(map::projection(1, &[0]))),
            ],
            name: Some("sum".into()),
        },
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Negate,
            operands: vec![(NodeId(2), IndexMap::Affine(map::projection(1, &[0])))],
            name: Some("negated".into()),
        },
    ];
    (program, [NodeId(2), NodeId(3)])
}

fn named<'data>(
    input: &'data [f32],
    bias: &'data [f32],
) -> [(&'static str, QuantizedBlock<'data>); 2] {
    [
        ("input", QuantizedBlock::Float32(input)),
        ("bias", QuantizedBlock::Float32(bias)),
    ]
}

fn bounded(limit: u64) -> RuntimeConfig {
    RuntimeConfig::builder()
        .with_constraints(ResourceConstraints::default().with_max_host_tensor_bytes(Some(limit)))
        .build()
        .expect("positive host ceiling must validate")
}

fn assert_outputs(result: &Evaluated, sum: &[f32], negated: &[f32]) {
    let (observed_sum, sum_shape) = result.get(NodeId(2)).expect("sum output must be present");
    let (observed_negated, negated_shape) = result
        .get(NodeId(3))
        .expect("negated output must be present");
    assert_eq!(sum_shape, [4]);
    assert_eq!(negated_shape, [4]);
    assert_eq!(observed_sum, sum);
    assert_eq!(observed_negated, negated);
}

#[test]
fn config_cpu_budget_admits_literal_envelope() {
    let (program, outputs) = fixture();
    let named = named(&[1.0, -2.0, 3.0, -4.0], &[10.0, 20.0, 30.0, 40.0]);
    let (mut plan, report) = plan_named_with_config(&bounded(56), &program, &[], &named, &outputs)
        .expect("56-byte declared envelope must admit");
    assert_eq!(report.len(), 1);
    assert_eq!(report[0].host_tensor_reserved_bytes, Some(56));
    assert_outputs(
        &execute_plan_named(&mut plan, &named).expect("bounded chain must execute"),
        &[11.0, 18.0, 33.0, 36.0],
        &[-11.0, -18.0, -33.0, -36.0],
    );
}

#[test]
fn config_cpu_budget_rejects_one_byte_below_envelope() {
    let (program, outputs) = fixture();
    let named = named(&[1.0, -2.0, 3.0, -4.0], &[10.0, 20.0, 30.0, 40.0]);
    let failure = plan_named_with_config(&bounded(55), &program, &[], &named, &outputs);
    assert!(matches!(
        failure,
        Err(BackendError::ResourceLimit {
            backend: "cpu",
            field: "max_host_tensor_bytes",
            requested: 56,
            limit: 55
        })
    ));
}

#[test]
fn config_cpu_budget_trims_retained_capacity_before_repeat() {
    let (program, outputs) = fixture();
    let first = named(&[1.0, -2.0, 3.0, -4.0], &[10.0, 20.0, 30.0, 40.0]);
    let (mut plan, _) = plan_named_with_config(&bounded(56), &program, &[], &first, &outputs)
        .expect("first envelope must admit");
    let mut retained = Vec::with_capacity(8);
    retained.push(99.0);
    assert_eq!(retained.capacity(), 8);
    match &mut plan {
        Plan::Cpu(cpu) => cpu.free_buffers.push(retained),
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Plan::Metal(_) => panic!("cpu requested"),
        #[cfg(feature = "wgpu-backend")]
        Plan::Wgpu(_) => panic!("cpu requested"),
        #[cfg(feature = "cuda-driver")]
        Plan::Cuda(_) => panic!("cpu requested"),
    }
    let second = named(&[2.0, -1.0, 4.0, -3.0], &[10.0, 20.0, 30.0, 40.0]);
    assert_outputs(
        &execute_plan_named(&mut plan, &second).expect("retained capacity must trim before repeat"),
        &[12.0, 19.0, 34.0, 37.0],
        &[-12.0, -19.0, -34.0, -37.0],
    );
    match &plan {
        Plan::Cpu(cpu) => assert!(cpu.free_buffers.is_empty()),
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Plan::Metal(_) => panic!("cpu requested"),
        #[cfg(feature = "wgpu-backend")]
        Plan::Wgpu(_) => panic!("cpu requested"),
        #[cfg(feature = "cuda-driver")]
        Plan::Cuda(_) => panic!("cpu requested"),
    }
}

#[test]
fn config_cpu_budget_rejects_unbounded_specialized_workspace() {
    let (program, outputs) = fixture();
    let packed = QuantizedBlock::Packed {
        codec: Codec::Float16,
        bytes: &[0; 8],
    };
    let named = [
        ("input", packed),
        ("bias", QuantizedBlock::Float32(&[10.0, 20.0, 30.0, 40.0])),
    ];
    let failure = plan_named_with_config(&bounded(4096), &program, &[], &named, &outputs);
    assert!(matches!(
        failure,
        Err(BackendError::Tensor(TensorError::NotLowerable {
            node: NodeId(0),
            reason: "host tensor budget has no packed or converted input workspace bound"
        }))
    ));
}

#[test]
fn config_cpu_budget_rejects_device_only_constraint() {
    let (program, outputs) = fixture();
    let named = named(&[1.0, -2.0, 3.0, -4.0], &[10.0, 20.0, 30.0, 40.0]);
    let config = RuntimeConfig::builder()
        .with_constraints(
            ResourceConstraints::default().with_max_single_device_buffer_bytes(Some(24)),
        )
        .build()
        .expect("portable positive device ceiling must validate");
    assert!(matches!(
        plan_named_with_config(&config, &program, &[], &named, &outputs),
        Err(BackendError::UnsupportedConstraint {
            backend: "cpu",
            field: "max_single_device_buffer_bytes",
            ..
        })
    ));
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_config() -> RuntimeConfig {
    RuntimeConfig::builder()
        .with_engine("gpu")
        .with_gpu_driver("metal")
        .with_constraints(ResourceConstraints::default().with_max_host_tensor_bytes(Some(4096)))
        .build()
        .expect("bounded Metal fixture config must validate")
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn config_metal_named_program_and_repeat() {
    let (program, outputs) = fixture();
    let first = named(&[1.0, -2.0, 3.0, -4.0], &[10.0, 20.0, 30.0, 40.0]);
    let (mut plan, report) =
        plan_named_with_config(&metal_config(), &program, &[], &first, &outputs)
            .expect("available Metal device must plan the affine chain");
    assert_eq!(report[0].driver, Some(GpuDriver::Metal));
    assert_eq!(report[0].host_tensor_reserved_bytes, Some(32));
    assert_eq!(report[0].device_buffer_reserved_bytes, Some(144));
    assert_eq!(report[0].largest_device_buffer_bytes, Some(48));
    assert_outputs(
        &execute_plan_named(&mut plan, &first).expect("Metal first vectors must read back"),
        &[11.0, 18.0, 33.0, 36.0],
        &[-11.0, -18.0, -33.0, -36.0],
    );
    let second = named(&[2.0, -1.0, 4.0, -3.0], &[10.0, 20.0, 30.0, 40.0]);
    assert_outputs(
        &execute_plan_named(&mut plan, &second).expect("Metal repeat vectors must read back"),
        &[12.0, 19.0, 34.0, 37.0],
        &[-12.0, -19.0, -34.0, -37.0],
    );
    let expected = [11.0, 18.0, 33.0, 36.0];
    assert_eq!(
        mismatch_positions(&expected, &expected),
        Vec::<usize>::new()
    );
    assert_eq!(
        mismatch_positions(&expected, &[11.0, 18.0, 33.0, 37.0]),
        [3]
    );
    let mut config = metal_config();
    config.selection.mode = SelectionMode::MinimumMeasuredWallTime;
    let (mut selected, reports) = plan_named_with_config(&config, &program, &[], &first, &outputs)
        .expect("measured heterogeneous fixture must retain an admissible candidate");
    assert_eq!(reports.iter().filter(|row| row.selected).count(), 1);
    let admissible = reports
        .iter()
        .filter(|row| row.rejection.is_none())
        .collect::<Vec<_>>();
    assert!(admissible.iter().any(|row| row.engine == Engine::Cpu));
    assert!(
        admissible
            .iter()
            .any(|row| row.driver == Some(GpuDriver::Metal))
    );
    for row in &admissible {
        assert_eq!(row.warmup_ns.len(), 1);
        assert_eq!(row.execution_ns.len(), 5);
        assert_eq!(
            row.execution_sum_ns,
            row.execution_ns
                .iter()
                .try_fold(0_u128, |sum, sample| sum.checked_add(*sample))
        );
    }
    assert_eq!(
        reports
            .iter()
            .find(|row| row.selected)
            .expect("selected row")
            .execution_sum_ns,
        admissible
            .iter()
            .filter_map(|row| row.execution_sum_ns)
            .min()
    );
    println!("heterogeneous_selection_raw={reports:?}");
    assert_outputs(
        &execute_plan_named(&mut selected, &first).expect("selected heterogeneous payload"),
        &[11.0, 18.0, 33.0, 36.0],
        &[-11.0, -18.0, -33.0, -36.0],
    );
}

fn mismatch_positions(expected: &[f32], observed: &[f32]) -> Vec<usize> {
    assert_eq!(expected.len(), observed.len());
    expected
        .iter()
        .zip(observed)
        .enumerate()
        .filter_map(|(position, (expected, observed))| {
            (expected.to_bits() != observed.to_bits()).then_some(position)
        })
        .collect()
}

#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    feature = "wgpu-backend",
    feature = "cuda-driver"
))]
fn matrix_fixture() -> (Vec<Op>, NodeId) {
    let program = vec![
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(2), Extent::Static(3)],
            name: Some("left".into()),
        },
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(3), Extent::Static(2)],
            name: Some("right".into()),
        },
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (NodeId(0), IndexMap::Affine(map::projection(3, &[0, 1]))),
                (NodeId(1), IndexMap::Affine(map::projection(3, &[1, 2]))),
            ],
            name: None,
        },
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: NodeId(2),
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 2])),
            keep: Keep::Reduce,
            name: None,
        }),
    ];
    (program, NodeId(3))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn config_metal_affine_matmul_named_program() {
    let (program, output) = matrix_fixture();
    let named = [
        (
            "left",
            QuantizedBlock::Float32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ),
        (
            "right",
            QuantizedBlock::Float32(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
        ),
    ];
    let (mut plan, _) = plan_named_with_config(&metal_config(), &program, &[], &named, &[output])
        .expect("Metal affine contraction must plan");
    let result =
        execute_plan_named(&mut plan, &named).expect("Metal affine contraction must read back");
    assert_eq!(
        result.get(output).expect("requested matrix output"),
        (&[58.0, 64.0, 139.0, 154.0][..], &[2, 2][..])
    );
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn config_metal_cap_failure_precedes_upload_and_dispatch() {
    #[cfg(feature = "instrument")]
    let before = metal::metal_stage_totals();
    let (program, outputs) = fixture();
    let named = named(&[1.0, -2.0, 3.0, -4.0], &[10.0, 20.0, 30.0, 40.0]);
    let config = metal_config()
        .into_builder()
        .with_constraints(ResourceConstraints::default().with_max_device_buffer_bytes(Some(143)))
        .build()
        .expect("positive cap config must load");
    assert!(matches!(
        plan_named_with_config(&config, &program, &[], &named, &outputs),
        Err(BackendError::ResourceLimit {
            backend: "metal",
            field: "max_device_buffer_bytes",
            requested: 144,
            limit: 143
        })
    ));
    #[cfg(feature = "instrument")]
    {
        let after = metal::metal_stage_totals();
        assert_eq!(before.output_buffer_allocations, 0);
        assert_eq!(before.block_upload_calls, 0);
        assert_eq!(before.encode_dispatch_calls, 0);
        assert_eq!(after.output_buffer_allocations, 0);
        assert_eq!(after.block_upload_calls, 0);
        assert_eq!(after.encode_dispatch_calls, 0);
        println!("metal_refusal_stages_before={before:?} after={after:?}");
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn config_metal_same_shape_execution_resource_census() {
    let (program, outputs) = fixture();
    let named = named(&[1.0, -2.0, 3.0, -4.0], &[10.0, 20.0, 30.0, 40.0]);
    let (mut plan, report) =
        plan_named_with_config(&metal_config(), &program, &[], &named, &outputs)
            .expect("bounded census must plan");
    assert_eq!(report[0].device_buffer_reserved_bytes, Some(144));
    for _ in 0..2 {
        assert_outputs(
            &execute_plan_named(&mut plan, &named).expect("same-shape census call"),
            &[11.0, 18.0, 33.0, 36.0],
            &[-11.0, -18.0, -33.0, -36.0],
        );
        match &plan {
            Plan::Metal(metal) => {
                assert_eq!(metal.scoped_reservation(), Some((32, 144, 48)));
                let rows = metal.scoped_resource_rows();
                assert_eq!(rows.len(), 6);
                assert_eq!(
                    rows.iter()
                        .map(|(role, node, _, capacity)| (*role, *node, *capacity))
                        .collect::<Vec<_>>(),
                    [
                        ("input", NodeId(0), 16),
                        ("input", NodeId(1), 16),
                        ("output", NodeId(2), 16),
                        ("uniform", NodeId(2), 48),
                        ("output", NodeId(3), 16),
                        ("uniform", NodeId(3), 32)
                    ]
                );
                let physical: BTreeMap<_, _> = rows
                    .iter()
                    .map(|(_, _, identity, capacity)| (*identity, *capacity))
                    .collect();
                assert_eq!(physical.len(), 6);
                assert_eq!(checked_resource_sum(physical.values().copied()), Some(144));
                let mut aliases = rows.clone();
                aliases.push(rows[0]);
                let deduplicated: BTreeMap<_, _> = aliases
                    .iter()
                    .map(|(_, _, identity, capacity)| (*identity, *capacity))
                    .collect();
                assert_eq!(
                    checked_resource_sum(deduplicated.values().copied()),
                    Some(144)
                );
                println!("metal_resource_rows={rows:?}");
            }
            _ => panic!("Metal requested"),
        }
    }
}

#[cfg(feature = "wgpu-backend")]
fn wgpu_config() -> RuntimeConfig {
    RuntimeConfig::builder()
        .with_engine("gpu")
        .with_gpu_driver("wgpu")
        .with_constraints(ResourceConstraints::default().with_max_device_buffer_bytes(Some(208)))
        .build()
        .expect("wgpu fixture caps validate")
}

#[cfg(feature = "wgpu-backend")]
#[test]
fn config_wgpu_named_program_and_repeat() {
    let (program, outputs) = fixture();
    let first = named(&[1.0, -2.0, 3.0, -4.0], &[10.0, 20.0, 30.0, 40.0]);
    let (mut plan, report) =
        plan_named_with_config(&wgpu_config(), &program, &[], &first, &outputs)
            .expect("wgpu configured chain");
    assert_eq!(report[0].driver, Some(GpuDriver::Wgpu));
    assert_eq!(report[0].host_tensor_reserved_bytes, Some(48));
    assert_eq!(report[0].device_buffer_reserved_bytes, Some(208));
    assert_eq!(report[0].largest_device_buffer_bytes, Some(24));
    assert_outputs(
        &execute_plan_named(&mut plan, &first).expect("wgpu first payload"),
        &[11.0, 18.0, 33.0, 36.0],
        &[-11.0, -18.0, -33.0, -36.0],
    );
    let second = named(&[2.0, -1.0, 4.0, -3.0], &[10.0, 20.0, 30.0, 40.0]);
    assert_outputs(
        &execute_plan_named(&mut plan, &second).expect("wgpu repeat payload"),
        &[12.0, 19.0, 34.0, 37.0],
        &[-12.0, -19.0, -34.0, -37.0],
    );
    let expected = [11.0, 18.0, 33.0, 36.0];
    assert_eq!(
        mismatch_positions(&expected, &expected),
        Vec::<usize>::new()
    );
    assert_eq!(
        mismatch_positions(&expected, &[11.0, 18.0, 33.0, 37.0]),
        [3]
    );
}

#[cfg(feature = "wgpu-backend")]
#[test]
fn config_wgpu_affine_matmul_named_program() {
    let (program, output) = matrix_fixture();
    let named = [
        (
            "left",
            QuantizedBlock::Float32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ),
        (
            "right",
            QuantizedBlock::Float32(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
        ),
    ];
    let mut config = wgpu_config();
    config.constraints.max_device_buffer_bytes = Some(4096);
    let (mut plan, _) = plan_named_with_config(&config, &program, &[], &named, &[output])
        .expect("wgpu matrix plan");
    let result = execute_plan_named(&mut plan, &named).expect("wgpu matrix payload");
    assert_eq!(
        result.get(output).expect("matrix output"),
        (&[58.0, 64.0, 139.0, 154.0][..], &[2, 2][..])
    );
}

#[cfg(feature = "wgpu-backend")]
#[test]
fn config_wgpu_cap_failure_precedes_upload_and_dispatch() {
    #[cfg(feature = "instrument")]
    assert_eq!(wgpu_driver::wgpu_stage_totals(), (0, 0, 0, 0));
    let (program, outputs) = fixture();
    let named = named(&[1.0, -2.0, 3.0, -4.0], &[10.0, 20.0, 30.0, 40.0]);
    let mut config = wgpu_config();
    for (constraints, field, requested, limit) in [
        (
            ResourceConstraints::default().with_max_device_buffer_bytes(Some(207)),
            "max_device_buffer_bytes",
            208,
            207,
        ),
        (
            ResourceConstraints::default().with_max_single_device_buffer_bytes(Some(23)),
            "max_single_device_buffer_bytes",
            24,
            23,
        ),
        (
            ResourceConstraints::default().with_max_host_tensor_bytes(Some(47)),
            "max_host_tensor_bytes",
            48,
            47,
        ),
    ] {
        config.constraints = constraints;
        let error = plan_named_with_config(&config, &program, &[], &named, &outputs)
            .err()
            .expect("below exact reservation rejects");
        assert!(
            matches!(error, BackendError::ResourceLimit { backend: "wgpu", field: observed_field, requested: observed_requested, limit: observed_limit } if (observed_field, observed_requested, observed_limit) == (field, requested, limit))
        );
        println!("wgpu_preflight_refusal={error:?}");
        #[cfg(feature = "instrument")]
        {
            let counts = wgpu_driver::wgpu_stage_totals();
            assert_eq!(counts, (0, 0, 0, 0));
            println!("wgpu_refusal_allocation_upload_dispatch_submit={counts:?}");
        }
    }
}

#[cfg(feature = "wgpu-backend")]
#[test]
fn config_wgpu_same_shape_execution_resource_census() {
    #[cfg(feature = "instrument")]
    assert_eq!(wgpu_driver::wgpu_stage_totals(), (0, 0, 0, 0));
    let (program, outputs) = fixture();
    let named = named(&[1.0, -2.0, 3.0, -4.0], &[10.0, 20.0, 30.0, 40.0]);
    let (mut plan, _) = plan_named_with_config(&wgpu_config(), &program, &[], &named, &outputs)
        .expect("wgpu census plan");
    for repetition in 0..2 {
        assert_outputs(
            &execute_plan_named(&mut plan, &named).expect("wgpu census payload"),
            &[11.0, 18.0, 33.0, 36.0],
            &[-11.0, -18.0, -33.0, -36.0],
        );
        let Plan::Wgpu(wgpu) = &plan else {
            panic!("wgpu requested")
        };
        assert_eq!(
            wgpu.scoped_reservation().expect("census envelope"),
            (48, 208, 24)
        );
        assert_eq!(wgpu.buffer_allocations(), 4);
        let rows = wgpu.scoped_resource_rows();
        assert_eq!(rows.len(), 12);
        assert_eq!(
            rows.iter()
                .map(|(role, node, _, capacity)| (*role, *node, *capacity))
                .collect::<Vec<_>>(),
            [
                ("input", NodeId(0), 16),
                ("input", NodeId(1), 16),
                ("output", NodeId(2), 16),
                ("output", NodeId(3), 16),
                ("uniform", NodeId(2), 24),
                ("uniform", NodeId(3), 16),
                ("upload", NodeId(0), 16),
                ("upload", NodeId(1), 16),
                ("upload", NodeId(2), 24),
                ("upload", NodeId(3), 16),
                ("readback", NodeId(2), 16),
                ("readback", NodeId(3), 16),
            ]
        );
        let physical: BTreeMap<_, _> = rows
            .iter()
            .map(|(_, _, identity, capacity)| (*identity, *capacity))
            .collect();
        assert_eq!(physical.len(), 12);
        assert_eq!(checked_resource_sum(physical.values().copied()), Some(208));
        println!(
            "wgpu_device={} resource_rows={rows:?}",
            wgpu.device_identity()
        );
        #[cfg(feature = "instrument")]
        {
            let counts = wgpu_driver::wgpu_stage_totals();
            assert_eq!(counts, (if repetition == 0 { 12 } else { 8 }, 4, 2, 1));
            println!("wgpu_admitted_allocation_upload_dispatch_submit={counts:?}");
        }
        #[cfg(not(feature = "instrument"))]
        let _ = repetition;
    }
    let bad = [
        ("input", QuantizedBlock::Float32(&[1.0, 2.0, 3.0])),
        ("bias", QuantizedBlock::Float32(&[10.0, 20.0, 30.0, 40.0])),
    ];
    assert!(matches!(
        execute_plan_named(&mut plan, &bad),
        Err(BackendError::InputContractChanged {
            node: NodeId(0),
            ..
        })
    ));
    let Plan::Wgpu(wgpu) = &plan else {
        panic!("wgpu requested")
    };
    assert_eq!(wgpu.buffer_allocations(), 4);
}

#[cfg(feature = "cuda-driver")]
fn cuda_config() -> RuntimeConfig {
    RuntimeConfig::builder()
        .with_engine("gpu")
        .with_gpu_driver("cuda")
        .with_constraints(ResourceConstraints::default().with_max_device_buffer_bytes(Some(4096)))
        .build()
        .expect("CUDA fixture caps validate")
}

#[cfg(feature = "cuda-driver")]
#[test]
fn config_cuda_named_program_and_repeat() {
    let (program, outputs) = fixture();
    let first = named(&[1.0, -2.0, 3.0, -4.0], &[10.0, 20.0, 30.0, 40.0]);
    let (mut plan, report) =
        plan_named_with_config(&cuda_config(), &program, &[], &first, &outputs)
            .expect("CUDA hardware fixture must plan");
    assert_eq!(report[0].driver, Some(GpuDriver::Cuda));
    assert_eq!(report[0].host_tensor_reserved_bytes, Some(32));
    assert_eq!(report[0].device_buffer_reserved_bytes, Some(144));
    assert_eq!(report[0].largest_device_buffer_bytes, Some(48));
    assert_outputs(
        &execute_plan_named(&mut plan, &first).expect("CUDA first payload"),
        &[11.0, 18.0, 33.0, 36.0],
        &[-11.0, -18.0, -33.0, -36.0],
    );
    let second = named(&[2.0, -1.0, 4.0, -3.0], &[10.0, 20.0, 30.0, 40.0]);
    assert_outputs(
        &execute_plan_named(&mut plan, &second).expect("CUDA repeat payload"),
        &[12.0, 19.0, 34.0, 37.0],
        &[-12.0, -19.0, -34.0, -37.0],
    );
}

#[cfg(feature = "cuda-driver")]
#[test]
fn config_cuda_affine_matmul_named_program() {
    let (program, output) = matrix_fixture();
    let named = [
        (
            "left",
            QuantizedBlock::Float32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ),
        (
            "right",
            QuantizedBlock::Float32(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
        ),
    ];
    let (mut plan, _) = plan_named_with_config(&cuda_config(), &program, &[], &named, &[output])
        .expect("CUDA hardware matrix plan");
    let result = execute_plan_named(&mut plan, &named).expect("CUDA matrix payload");
    assert_eq!(
        result.get(output).expect("matrix output"),
        (&[58.0, 64.0, 139.0, 154.0][..], &[2, 2][..])
    );
}

#[cfg(feature = "cuda-driver")]
#[test]
fn config_cuda_cap_failure_precedes_upload_and_dispatch() {
    let (program, outputs) = fixture();
    let named = named(&[1.0, -2.0, 3.0, -4.0], &[10.0, 20.0, 30.0, 40.0]);
    let driver = CudaDriver::new(0).expect("CUDA hardware is required for resource preflight");
    let mut config = cuda_config();
    let plan = driver
        .plan_with_config(&program, &[], &outputs, &named, &config)
        .expect("CUDA baseline reservation");
    let (_, total, _) = plan.scoped_reservation().expect("CUDA arena envelope");
    assert_eq!(total, 144);
    assert_eq!(plan.allocations(), 0);
    config.constraints.max_device_buffer_bytes = Some(total - 1);
    let error = driver
        .plan_with_config(&program, &[], &outputs, &named, &config)
        .err()
        .expect("one byte below CUDA envelope rejects");
    assert!(
        matches!(error, CudaDriverError::ResourceLimit { field: "max_device_buffer_bytes", requested, limit } if requested == total && limit == total - 1)
    );
    let bad = [
        ("input", QuantizedBlock::Float32(&[1.0, 2.0, 3.0])),
        ("bias", QuantizedBlock::Float32(&[10.0, 20.0, 30.0, 40.0])),
    ];
    assert!(matches!(
        driver.plan_with_config(&program, &[], &outputs, &bad, &cuda_config()),
        Err(CudaDriverError::InputContractChanged {
            node: NodeId(0),
            ..
        })
    ));
}

#[cfg(feature = "cuda-driver")]
#[test]
fn config_cuda_same_shape_execution_resource_census() {
    let (program, outputs) = fixture();
    let named = named(&[1.0, -2.0, 3.0, -4.0], &[10.0, 20.0, 30.0, 40.0]);
    let (mut plan, report) =
        plan_named_with_config(&cuda_config(), &program, &[], &named, &outputs)
            .expect("CUDA hardware census plan");
    let expected = (
        report[0].host_tensor_reserved_bytes.expect("host envelope"),
        report[0]
            .device_buffer_reserved_bytes
            .expect("device envelope"),
        report[0]
            .largest_device_buffer_bytes
            .expect("largest envelope"),
    );
    for _ in 0..2 {
        assert_outputs(
            &execute_plan_named(&mut plan, &named).expect("CUDA census payload"),
            &[11.0, 18.0, 33.0, 36.0],
            &[-11.0, -18.0, -33.0, -36.0],
        );
        let Plan::Cuda(cuda) = &plan else {
            panic!("CUDA requested")
        };
        assert_eq!(
            cuda.scoped_reservation()
                .expect("CUDA repeated arena envelope"),
            expected
        );
        assert_eq!(cuda.allocations(), 6);
    }
    let bad = [
        ("input", QuantizedBlock::Float32(&[1.0, 2.0, 3.0])),
        ("bias", QuantizedBlock::Float32(&[10.0, 20.0, 30.0, 40.0])),
    ];
    assert!(matches!(
        execute_plan_named(&mut plan, &bad),
        Err(BackendError::InputContractChanged {
            node: NodeId(0),
            ..
        })
    ));
}
