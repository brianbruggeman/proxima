use std::io::Write;

use moe_test_json as serde_json;
use proxima_primitives::float_layout::{FloatLayout, SignEncoding, Specials};
use proxima_tensor::NumericPolicy;
use tempfile::NamedTempFile;

use super::{
    BatchingStrategy, DecodeKernelStrategy, DenseMmaInputLayout, FusionStrategy, ResidencyStrategy,
    ResourceConstraints, RuntimeConfig, SelectionConfig, SelectionMode, SpeculationStrategy,
    WeightConversionStrategy,
};

const CLEAN_ENVIRONMENT: [(&str, Option<&str>); 31] = [
    ("OMEGA_ENGINE", None),
    ("OMEGA_GPU_DRIVER", None),
    ("OMEGA_ALLOW_FUSION", None),
    ("OMEGA_EXACT_ACTIVATIONS", None),
    ("OMEGA_NUMERIC_CONTRACTION", None),
    ("OMEGA_NUMERIC_REASSOCIATION", None),
    ("OMEGA_NUMERIC_NAN_ASSUMPTIONS", None),
    ("OMEGA_NUMERIC_SIGNED_ZERO", None),
    ("OMEGA_NUMERIC_APPROX_FUNCTIONS", None),
    ("OMEGA_NUMERIC_EPILOGUE_SOURCES", None),
    ("OMEGA_CONSTRAINTS_MAX_HOST_TENSOR_BYTES", None),
    ("OMEGA_CONSTRAINTS_MAX_DEVICE_BUFFER_BYTES", None),
    ("OMEGA_CONSTRAINTS_MAX_SINGLE_DEVICE_BUFFER_BYTES", None),
    ("OMEGA_SELECTION_MODE", None),
    ("OMEGA_SELECTION_WARMUP_RUNS", None),
    ("OMEGA_SELECTION_MEASUREMENT_RUNS", None),
    ("OMEGA_CPU_WORKERS", None),
    ("OMEGA_INDEPENDENT_OPS", None),
    ("OMEGA_PARALLEL_SEQUENCES", None),
    ("OMEGA_MAX_IN_FLIGHT", None),
    ("OMEGA_OVERLAP_TRANSFER_COMPUTE", None),
    ("OMEGA_MAX_DEVICE_BYTES", None),
    ("OMEGA_DECODE_KERNEL", None),
    ("OMEGA_FUSION", None),
    ("OMEGA_RESIDENCY", None),
    ("OMEGA_BATCHING", None),
    ("OMEGA_SPECULATION", None),
    ("OMEGA_WEIGHT_CONVERSION", None),
    ("OMEGA_MOE_GROUPING", None),
    ("OMEGA_DENSE_MMA_INPUT_LAYOUT", None),
    ("OMEGA_NUMERIC_BF16_INPUT_ROUNDING", None),
];

fn validation_text(result: conflaguration::Result<RuntimeConfig>) -> String {
    match result {
        Ok(_) => panic!("expected runtime config validation to fail"),
        Err(error) => error.to_string(),
    }
}

fn write_config(contents: &str) -> NamedTempFile {
    let mut file = NamedTempFile::with_suffix(".toml")
        .expect("temporary config file is created with a TOML suffix");
    file.write_all(contents.as_bytes())
        .expect("temporary config contents are written");
    file
}

fn runtime_config_difference_count(left: &RuntimeConfig, right: &RuntimeConfig) -> usize {
    [
        left.engine != right.engine,
        left.gpu_driver != right.gpu_driver,
        left.allow_fusion != right.allow_fusion,
        left.exact_activations != right.exact_activations,
        left.pipeline_cache != right.pipeline_cache,
        left.pipeline_cache_dir != right.pipeline_cache_dir,
        left.pipeline_cache_max_bytes != right.pipeline_cache_max_bytes,
        left.cpu_workers != right.cpu_workers,
        left.independent_ops != right.independent_ops,
        left.parallel_sequences != right.parallel_sequences,
        left.max_in_flight != right.max_in_flight,
        left.overlap_transfer_compute != right.overlap_transfer_compute,
        left.max_device_bytes != right.max_device_bytes,
        left.decode_kernel != right.decode_kernel,
        left.fusion != right.fusion,
        left.residency != right.residency,
        left.batching != right.batching,
        left.speculation != right.speculation,
        left.weight_conversion != right.weight_conversion,
        left.moe_grouping != right.moe_grouping,
        left.numeric != right.numeric,
        left.dense_mma_input_layout != right.dense_mma_input_layout,
        left.constraints != right.constraints,
        left.selection != right.selection,
    ]
    .into_iter()
    .filter(|differs| *differs)
    .count()
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
struct RuntimeConfigFixture {
    cpu_workers: u16,
    independent_ops: u16,
    parallel_sequences: u16,
    max_in_flight: u16,
    overlap_transfer_compute: bool,
    max_device_bytes: Option<u64>,
    engine: String,
    reassociation: bool,
    decode_kernel: DecodeKernelStrategy,
    fusion: FusionStrategy,
    residency: ResidencyStrategy,
    batching: BatchingStrategy,
    speculation: SpeculationStrategy,
    weight_conversion: WeightConversionStrategy,
    moe_grouping: String,
    #[serde(default)]
    dense_mma_input_layout: DenseMmaInputLayout,
}

fn runtime_config_fixture_difference_count(
    actual: &RuntimeConfig,
    expected: &RuntimeConfigFixture,
) -> usize {
    [
        actual.cpu_workers != expected.cpu_workers,
        actual.independent_ops != expected.independent_ops,
        actual.parallel_sequences != expected.parallel_sequences,
        actual.max_in_flight != expected.max_in_flight,
        actual.overlap_transfer_compute != expected.overlap_transfer_compute,
        actual.max_device_bytes != expected.max_device_bytes,
        actual.engine != expected.engine,
        actual.numeric.reassociation != expected.reassociation,
        actual.decode_kernel != expected.decode_kernel,
        actual.fusion != expected.fusion,
        actual.residency != expected.residency,
        actual.batching != expected.batching,
        actual.speculation != expected.speculation,
        actual.weight_conversion != expected.weight_conversion,
        (actual.moe_grouping != (expected.moe_grouping == "enabled")),
        (actual.dense_mma_input_layout != expected.dense_mma_input_layout),
    ]
    .into_iter()
    .filter(|differs| *differs)
    .count()
}

#[test]
fn runtime_config_defaults_preserve_legacy_values() {
    let config = RuntimeConfig::default();

    assert_eq!(config.engine, "cpu");
    assert_eq!(config.gpu_driver, "auto");
    assert!(!config.allow_fusion);
    assert!(config.exact_activations);
    assert!(!config.numeric.contraction);
    assert!(!config.numeric.reassociation);
    assert!(!config.numeric.nan_assumptions);
    assert!(!config.numeric.signed_zero);
    assert!(!config.numeric.approx_functions);
    assert!(!config.numeric.epilogue_sources);
    assert!(!config.numeric.bf16_input_rounding);
    assert_eq!(
        config.dense_mma_input_layout,
        DenseMmaInputLayout::derived()
    );
    assert_eq!(config.constraints.max_host_tensor_bytes, None);
    assert_eq!(config.constraints.max_device_buffer_bytes, None);
    assert_eq!(config.constraints.max_single_device_buffer_bytes, None);
    assert_eq!(config.selection.mode, SelectionMode::Configured);
    assert_eq!(config.selection.warmup_runs, None);
    assert_eq!(config.selection.measurement_runs, None);
}

#[test]
fn runtime_config_legacy_toml_deserializes() {
    let config: RuntimeConfig = toml::from_str(
        "engine = 'gpu'\ngpu_driver = 'metal'\nallow_fusion = true\nexact_activations = false\n",
    )
    .expect("legacy runtime keys deserialize");

    assert_eq!(config.engine, "gpu");
    assert_eq!(config.gpu_driver, "metal");
    assert!(config.allow_fusion);
    assert!(!config.exact_activations);
    assert_eq!(config.numeric, NumericPolicy::bit_exact());
    assert_eq!(config.constraints, ResourceConstraints::default());
    assert_eq!(config.selection, SelectionConfig::default());
}

#[test]
fn dense_mma_layout_accepts_structured_settings_and_legacy_aliases() {
    let structured: RuntimeConfig = toml::from_str(
        "[dense_mma_input_layout]\nlayout = { exponent_bits = 8, mantissa_bits = 7, specials = 'InfinityOnOverflow', bias_override = 127, sign = 'Magnitude' }\n",
    )
    .expect("the float layout is configured as data");
    let alias: RuntimeConfig = toml::from_str("dense_mma_input_layout = 'bf16'\n")
        .expect("the previous short config spelling remains readable");

    assert_eq!(
        structured.dense_mma_input_layout,
        DenseMmaInputLayout::explicit(FloatLayout {
            exponent_bits: 8,
            mantissa_bits: 7,
            specials: Specials::InfinityOnOverflow,
            bias_override: Some(127),
            sign: SignEncoding::Magnitude,
        })
    );
    assert_eq!(
        alias.dense_mma_input_layout,
        DenseMmaInputLayout::explicit(FloatLayout::BF16)
    );
}

#[test]
fn dense_mma_layout_rejects_a_valid_float_format_the_lowerer_does_not_support() {
    let error = validation_text(
        RuntimeConfig::builder()
            .with_dense_mma_input_layout(DenseMmaInputLayout::explicit(FloatLayout::E3M2))
            .build(),
    );

    assert!(error.contains("dense_mma_input_layout.layout"));
    assert!(error.contains("only binary32 and bfloat16"));
}

#[test]
fn runtime_config_settings_uses_omega_legacy_prefix() {
    temp_env::with_vars(
        [
            ("OMEGA_ENGINE", Some("gpu")),
            ("OMEGA_GPU_DRIVER", Some("wgpu")),
            ("OMEGA_ALLOW_FUSION", Some("true")),
            ("OMEGA_EXACT_ACTIVATIONS", Some("false")),
            ("OMEGA_NUMERIC_CONTRACTION", None),
            ("OMEGA_NUMERIC_REASSOCIATION", None),
            ("OMEGA_NUMERIC_NAN_ASSUMPTIONS", None),
            ("OMEGA_NUMERIC_SIGNED_ZERO", None),
            ("OMEGA_NUMERIC_APPROX_FUNCTIONS", None),
            ("OMEGA_NUMERIC_EPILOGUE_SOURCES", None),
            ("OMEGA_CONSTRAINTS_MAX_HOST_TENSOR_BYTES", None),
            ("OMEGA_CONSTRAINTS_MAX_DEVICE_BUFFER_BYTES", None),
            ("OMEGA_CONSTRAINTS_MAX_SINGLE_DEVICE_BUFFER_BYTES", None),
            ("OMEGA_SELECTION_MODE", None),
            ("OMEGA_SELECTION_WARMUP_RUNS", None),
            ("OMEGA_SELECTION_MEASUREMENT_RUNS", None),
        ],
        || {
            let config = RuntimeConfig::from_env().expect("legacy env config validates");
            assert_eq!(config.engine, "gpu");
            assert_eq!(config.gpu_driver, "wgpu");
            assert!(config.allow_fusion);
            assert!(!config.exact_activations);
        },
    );
}

#[test]
fn runtime_config_nested_env_overrides_nested_defaults() {
    temp_env::with_vars(
        [
            ("OMEGA_ENGINE", None),
            ("OMEGA_GPU_DRIVER", None),
            ("OMEGA_ALLOW_FUSION", None),
            ("OMEGA_EXACT_ACTIVATIONS", None),
            ("OMEGA_NUMERIC_CONTRACTION", Some("true")),
            ("OMEGA_NUMERIC_REASSOCIATION", None),
            ("OMEGA_NUMERIC_NAN_ASSUMPTIONS", None),
            ("OMEGA_NUMERIC_SIGNED_ZERO", None),
            ("OMEGA_NUMERIC_APPROX_FUNCTIONS", None),
            ("OMEGA_NUMERIC_EPILOGUE_SOURCES", None),
            ("OMEGA_CONSTRAINTS_MAX_HOST_TENSOR_BYTES", None),
            ("OMEGA_CONSTRAINTS_MAX_DEVICE_BUFFER_BYTES", Some("4096")),
            ("OMEGA_CONSTRAINTS_MAX_SINGLE_DEVICE_BUFFER_BYTES", None),
            ("OMEGA_SELECTION_MODE", Some("minimum_measured_wall_time")),
            ("OMEGA_SELECTION_WARMUP_RUNS", Some("2")),
            ("OMEGA_SELECTION_MEASUREMENT_RUNS", Some("7")),
        ],
        || {
            let config = RuntimeConfig::from_env().expect("nested config overrides parse");
            assert_eq!(config.engine, "cpu");
            assert_eq!(config.gpu_driver, "auto");
            assert!(!config.allow_fusion);
            assert!(config.exact_activations);
            assert_eq!(
                config.selection.mode,
                SelectionMode::MinimumMeasuredWallTime
            );
            assert_eq!(config.selection.warmup_runs, Some(2));
            assert_eq!(config.selection.measurement_runs, Some(7));
            assert_eq!(config.constraints.max_host_tensor_bytes, None);
            assert_eq!(config.constraints.max_device_buffer_bytes, Some(4096));
            assert_eq!(config.constraints.max_single_device_buffer_bytes, None);
            assert!(config.numeric.contraction);
            assert!(!config.numeric.reassociation);
            assert!(!config.numeric.nan_assumptions);
            assert!(!config.numeric.signed_zero);
            assert!(!config.numeric.approx_functions);
            assert!(!config.numeric.epilogue_sources);
        },
    );
}

#[test]
fn runtime_config_builder_matches_serde_config() {
    let source = "engine = 'gpu'\ngpu_driver = 'metal'\n[constraints]\nmax_device_buffer_bytes = 4096\n[numeric]\nreassociation = true\n";
    let from_serde: RuntimeConfig = toml::from_str(source).expect("config TOML deserializes");
    let from_builder = RuntimeConfig::builder()
        .with_engine("gpu")
        .with_gpu_driver("metal")
        .with_constraints(ResourceConstraints::default().with_max_device_buffer_bytes(Some(4096)))
        .with_numeric(NumericPolicy::bit_exact().with_reassociation(true))
        .build()
        .expect("builder config validates");

    assert_eq!(from_builder, from_serde);
    assert_eq!(from_builder.engine, "gpu");
    assert_eq!(from_builder.gpu_driver, "metal");
    assert!(!from_builder.allow_fusion);
    assert!(from_builder.exact_activations);
    assert!(!from_builder.numeric.contraction);
    assert!(from_builder.numeric.reassociation);
    assert!(!from_builder.numeric.nan_assumptions);
    assert!(!from_builder.numeric.signed_zero);
    assert!(!from_builder.numeric.approx_functions);
    assert!(!from_builder.numeric.epilogue_sources);
    assert_eq!(from_builder.constraints.max_host_tensor_bytes, None);
    assert_eq!(from_builder.constraints.max_device_buffer_bytes, Some(4096));
    assert_eq!(
        from_builder.constraints.max_single_device_buffer_bytes,
        None
    );
    assert_eq!(from_builder.selection.mode, SelectionMode::Configured);
    assert_eq!(from_builder.selection.warmup_runs, None);
    assert_eq!(from_builder.selection.measurement_runs, None);
    let from_into_builder = from_serde
        .clone()
        .into_builder()
        .build()
        .expect("builder seeded from config validates");
    assert_eq!(from_into_builder, from_serde);
}

#[test]
fn runtime_config_layer_order_is_source_then_explicit_override() {
    let file = write_config(
        "engine = 'gpu'\ngpu_driver = 'metal'\n[constraints]\nmax_device_buffer_bytes = 8192\n",
    );
    let override_after_file = RuntimeConfig::builder()
        .from_file(file.path())
        .expect("file loads")
        .with_engine("cpu")
        .with_constraints(ResourceConstraints::default().with_max_device_buffer_bytes(Some(4096)))
        .build()
        .expect("later fluent overrides validate");
    assert_eq!(override_after_file.engine, "cpu");
    assert_eq!(
        override_after_file.constraints.max_device_buffer_bytes,
        Some(4096)
    );

    let override_before_file = RuntimeConfig::builder()
        .with_engine("cpu")
        .with_constraints(ResourceConstraints::default().with_max_device_buffer_bytes(Some(4096)))
        .from_file(file.path())
        .expect("source replaces previous value")
        .build()
        .expect("loaded file config validates");
    assert_eq!(override_before_file.engine, "gpu");
    assert_eq!(
        override_before_file.constraints.max_device_buffer_bytes,
        Some(8192)
    );

    temp_env::with_vars(
        [
            ("OMEGA_ENGINE", Some("gpu")),
            ("OMEGA_GPU_DRIVER", Some("wgpu")),
            ("OMEGA_ALLOW_FUSION", None),
            ("OMEGA_EXACT_ACTIVATIONS", None),
            ("OMEGA_NUMERIC_CONTRACTION", None),
            ("OMEGA_NUMERIC_REASSOCIATION", None),
            ("OMEGA_NUMERIC_NAN_ASSUMPTIONS", None),
            ("OMEGA_NUMERIC_SIGNED_ZERO", None),
            ("OMEGA_NUMERIC_APPROX_FUNCTIONS", None),
            ("OMEGA_NUMERIC_EPILOGUE_SOURCES", None),
            ("OMEGA_CONSTRAINTS_MAX_HOST_TENSOR_BYTES", None),
            ("OMEGA_CONSTRAINTS_MAX_DEVICE_BUFFER_BYTES", None),
            ("OMEGA_CONSTRAINTS_MAX_SINGLE_DEVICE_BUFFER_BYTES", None),
            ("OMEGA_SELECTION_MODE", None),
            ("OMEGA_SELECTION_WARMUP_RUNS", None),
            ("OMEGA_SELECTION_MEASUREMENT_RUNS", None),
        ],
        || {
            let from_env = RuntimeConfig::builder()
                .with_engine("cpu")
                .from_env()
                .expect("builder environment source loads")
                .build()
                .expect("environment config validates");
            assert_eq!(from_env.engine, "gpu");
            assert_eq!(from_env.gpu_driver, "wgpu");
        },
    );
}

#[test]
fn runtime_config_rejects_unknown_engine() {
    let file = write_config("engine = 'npu'\n");
    let message = validation_text(RuntimeConfig::from_file(file.path()));
    assert!(message.contains("engine"));
    assert!(message.contains("npu"));

    let file = write_config("engnie = 'gpu'\n");
    assert!(RuntimeConfig::from_file(file.path()).is_err());
}

#[test]
fn runtime_config_rejects_unknown_gpu_driver() {
    let file = write_config("gpu_driver = 'vulkan'\n");
    let message = validation_text(RuntimeConfig::from_file(file.path()));
    assert!(message.contains("gpu_driver"));
    assert!(message.contains("vulkan"));

    let file = write_config("engine = 'gpu'\ngpu_driver = 'cuda'\n");
    let config = RuntimeConfig::from_file(file.path()).expect("portable cuda config validates");
    assert_eq!(config.gpu_driver, "cuda");
}

#[test]
fn runtime_config_rejects_zero_resource_ceiling() {
    for (field, constraints) in [
        (
            "max_host_tensor_bytes",
            ResourceConstraints::default().with_max_host_tensor_bytes(Some(0)),
        ),
        (
            "max_device_buffer_bytes",
            ResourceConstraints::default().with_max_device_buffer_bytes(Some(0)),
        ),
        (
            "max_single_device_buffer_bytes",
            ResourceConstraints::default().with_max_single_device_buffer_bytes(Some(0)),
        ),
    ] {
        let message = validation_text(
            RuntimeConfig::builder()
                .with_constraints(constraints)
                .build(),
        );
        assert!(message.contains(field), "missing {field}: {message}");
    }

    let valid = RuntimeConfig::builder()
        .with_constraints(
            ResourceConstraints::default()
                .with_max_host_tensor_bytes(Some(1))
                .with_max_device_buffer_bytes(Some(1))
                .with_max_single_device_buffer_bytes(Some(1)),
        )
        .build()
        .expect("one-byte positive ceilings are valid config data");
    assert_eq!(valid.constraints.max_host_tensor_bytes, Some(1));
    assert_eq!(valid.constraints.max_device_buffer_bytes, Some(1));
    assert_eq!(valid.constraints.max_single_device_buffer_bytes, Some(1));

    let file = write_config("[constraints]\nmax_device_buffer_byte = 1\n");
    assert!(RuntimeConfig::from_file(file.path()).is_err());

    let mut environment = CLEAN_ENVIRONMENT;
    environment[10].1 = Some("0");
    temp_env::with_vars(environment, || {
        let message = validation_text(RuntimeConfig::from_env());
        assert!(message.contains("max_host_tensor_bytes"));
    });
}

#[test]
fn runtime_config_requires_measurement_samples() {
    let invalid = RuntimeConfig::builder()
        .with_selection(
            SelectionConfig::default()
                .with_mode(SelectionMode::MinimumMeasuredWallTime)
                .with_warmup_runs(Some(0))
                .with_measurement_runs(Some(2)),
        )
        .build();
    let message = validation_text(invalid);
    assert!(message.contains("warmup_runs"));
    assert!(message.contains("measurement_runs"));

    let valid = RuntimeConfig::builder()
        .with_selection(
            SelectionConfig::default()
                .with_mode(SelectionMode::MinimumMeasuredWallTime)
                .with_warmup_runs(Some(1))
                .with_measurement_runs(Some(3)),
        )
        .build()
        .expect("minimum measured counts validate");
    assert_eq!(valid.selection.resolved_warmup_runs(), 1);
    assert_eq!(valid.selection.resolved_measurement_runs(), 3);

    let configured_counts = RuntimeConfig::builder()
        .with_selection(
            SelectionConfig::default()
                .with_warmup_runs(Some(1))
                .with_measurement_runs(Some(3)),
        )
        .build();
    let configured_message = validation_text(configured_counts);
    assert!(configured_message.contains("selection.warmup_runs"));
    assert!(configured_message.contains("selection.measurement_runs"));

    let misspelled_selection = write_config("[selection]\nmeasuremnt_runs = 3\n");
    assert!(RuntimeConfig::from_file(misspelled_selection.path()).is_err());

    let mut environment = CLEAN_ENVIRONMENT;
    environment[13].1 = Some("unknown_mode");
    temp_env::with_vars(environment, || {
        assert!(RuntimeConfig::from_env().is_err());
    });
}

#[test]
fn runtime_config_numeric_permissions_round_trip_independently() {
    let source =
        "[numeric]\ncontraction = true\napprox_functions = true\nbf16_input_rounding = true\n";
    let from_serde: RuntimeConfig = toml::from_str(source).expect("numeric TOML deserializes");
    assert!(from_serde.numeric.contraction);
    assert!(!from_serde.numeric.reassociation);
    assert!(!from_serde.numeric.nan_assumptions);
    assert!(!from_serde.numeric.signed_zero);
    assert!(from_serde.numeric.approx_functions);
    assert!(!from_serde.numeric.epilogue_sources);
    assert!(from_serde.numeric.bf16_input_rounding);

    let serialized = toml::to_string(&from_serde).expect("runtime config serializes");
    let serde_round_trip: RuntimeConfig =
        toml::from_str(&serialized).expect("runtime config round-trips");
    assert_eq!(serde_round_trip, from_serde);

    let mut environment = CLEAN_ENVIRONMENT;
    environment[4].1 = Some("true");
    environment[8].1 = Some("true");
    environment[30].1 = Some("true");
    let scoped: Vec<_> = environment.into_iter().collect();
    temp_env::with_vars(scoped, || {
        let from_settings = RuntimeConfig::from_env().expect("numeric Settings parses");
        assert_eq!(from_settings.numeric, from_serde.numeric);
    });

    let from_builder = RuntimeConfig::builder()
        .with_numeric(
            NumericPolicy::bit_exact()
                .with_contraction(true)
                .with_approx_functions(true)
                .with_bf16_input_rounding(true),
        )
        .build()
        .expect("numeric builder config validates");
    assert_eq!(from_builder.numeric, from_serde.numeric);

    let misspelled_numeric = write_config("[numeric]\ncontracion = true\n");
    assert!(RuntimeConfig::from_file(misspelled_numeric.path()).is_err());
}

#[test]
fn runtime_config_defaults_track_sized_floor() {
    let config = RuntimeConfig::default();
    assert_eq!(crate::sized::RUNTIME_WARMUP_RUNS, 1);
    assert_eq!(crate::sized::RUNTIME_MEASUREMENT_RUNS, 5);
    assert_eq!(config.selection.warmup_runs, None);
    assert_eq!(config.selection.measurement_runs, None);
    assert_eq!(
        config.selection.resolved_warmup_runs(),
        crate::sized::RUNTIME_WARMUP_RUNS
    );
    assert_eq!(
        config.selection.resolved_measurement_runs(),
        crate::sized::RUNTIME_MEASUREMENT_RUNS
    );

    temp_env::with_vars(CLEAN_ENVIRONMENT, || {
        let from_env = RuntimeConfig::from_env().expect("defaults from env validate");
        assert_eq!(
            from_env.selection.resolved_warmup_runs(),
            crate::sized::RUNTIME_WARMUP_RUNS
        );
        assert_eq!(
            from_env.selection.resolved_measurement_runs(),
            crate::sized::RUNTIME_MEASUREMENT_RUNS
        );
    });
}

#[test]
fn unified_runtime_config_layered_fixture_resolves_file_env_fluent() {
    let fixtures: serde_json::Value = serde_json::from_str(include_str!(
        "../../proxima-tensor/specs/ouro-mainline/unified-programmable/fixtures.json"
    ))
    .expect("admitted fixture JSON parses");
    let expected: RuntimeConfigFixture =
        serde_json::from_value(fixtures["resolved_config"]["expected"].clone())
            .expect("resolved config fixture deserializes");
    let file = write_config(
        "cpu_workers = 2\nindependent_ops = 2\nparallel_sequences = 3\nmax_in_flight = 2\nmax_device_bytes = 64\nengine = 'cpu'\n",
    );
    let mut environment = CLEAN_ENVIRONMENT;
    environment[16].1 = Some("3");
    environment[17].1 = Some("3");
    environment[20].1 = Some("true");
    temp_env::with_vars(environment, || {
        let resolved = RuntimeConfig::builder()
            .from_file(file.path())
            .expect("file policy loads")
            .from_env()
            .expect("environment overrides file policy")
            .with_cpu_workers(4)
            .with_independent_ops(3)
            .with_overlap_transfer_compute(true)
            .build()
            .expect("resolved policy validates");
        let independent_expected = RuntimeConfig::builder()
            .with_cpu_workers(4)
            .with_independent_ops(3)
            .with_parallel_sequences(3)
            .with_max_in_flight(2)
            .with_max_device_bytes(Some(64))
            .with_overlap_transfer_compute(true)
            .build()
            .expect("independent expected policy validates");
        let serialized = serde_json::to_string(&resolved).expect("resolved policy serializes");
        println!("resolved_config={serialized}");
        let differences = runtime_config_fixture_difference_count(&resolved, &expected);
        println!("fixture_comparator_differences={differences}");
        assert_eq!(differences, 0);
        let mut corrupted = expected;
        corrupted.cpu_workers = 3;
        let control_differences = runtime_config_fixture_difference_count(&resolved, &corrupted);
        println!("one_field_corruption_differences={control_differences}");
        assert_eq!(control_differences, 1);
        assert_eq!(
            runtime_config_difference_count(&resolved, &independent_expected),
            0
        );
    });
}

#[test]
fn unified_runtime_config_sized_defaults_seed_runtime_defaults() {
    let config = RuntimeConfig::default();
    println!(
        "generated_defaults={{\"cpu_workers\":{},\"independent_ops\":{},\"parallel_sequences\":{},\"max_in_flight\":{}}}",
        config.cpu_workers, config.independent_ops, config.parallel_sequences, config.max_in_flight
    );
    assert_eq!(config.cpu_workers, crate::sized::RUNTIME_CPU_WORKERS);
    assert_eq!(
        config.independent_ops,
        crate::sized::RUNTIME_INDEPENDENT_OPS
    );
    assert_eq!(
        config.parallel_sequences,
        crate::sized::RUNTIME_PARALLEL_SEQUENCES
    );
    assert_eq!(config.max_in_flight, crate::sized::RUNTIME_MAX_IN_FLIGHT);
    assert_eq!(config.max_device_bytes, None);
}

#[test]
fn unified_runtime_config_serde_builder_round_trip_preserves_full_policy() {
    let source = "cpu_workers = 4\nindependent_ops = 2\nparallel_sequences = 3\nmax_in_flight = 2\noverlap_transfer_compute = true\nmax_device_bytes = 64\ndecode_kernel = 'fused_projection'\nfusion = 'required'\nresidency = 'exact_resident_first'\nbatching = 'compatible_rows'\nspeculation = 'disabled'\nweight_conversion = 'disabled'\nmoe_grouping = false\n[numeric]\nreassociation = false\n[dense_mma_input_layout]\nderive_from_operands = false\nlayout = { exponent_bits = 8, mantissa_bits = 7, specials = 'InfinityOnOverflow', bias_override = 127, sign = 'Magnitude' }\n";
    let from_serde: RuntimeConfig = toml::from_str(source).expect("policy TOML deserializes");
    let from_builder = RuntimeConfig::builder()
        .with_cpu_workers(4)
        .with_independent_ops(2)
        .with_parallel_sequences(3)
        .with_max_in_flight(2)
        .with_overlap_transfer_compute(true)
        .with_max_device_bytes(Some(64))
        .with_decode_kernel(DecodeKernelStrategy::FusedProjection)
        .with_fusion(FusionStrategy::Required)
        .with_residency(ResidencyStrategy::ExactResidentFirst)
        .with_batching(BatchingStrategy::CompatibleRows)
        .with_speculation(SpeculationStrategy::Disabled)
        .with_weight_conversion(WeightConversionStrategy::Disabled)
        .with_moe_grouping(false)
        .with_dense_mma_input_layout(DenseMmaInputLayout::explicit(FloatLayout {
            exponent_bits: 8,
            mantissa_bits: 7,
            specials: Specials::InfinityOnOverflow,
            bias_override: Some(127),
            sign: SignEncoding::Magnitude,
        }))
        .build()
        .expect("fluent policy validates");
    assert_eq!(
        runtime_config_difference_count(&from_builder, &from_serde),
        0
    );
    let serialized = toml::to_string(&from_builder).expect("policy serializes");
    let round_trip: RuntimeConfig = toml::from_str(&serialized).expect("policy round-trips");
    assert_eq!(
        runtime_config_difference_count(&from_builder, &round_trip),
        0
    );
}

#[test]
fn unified_runtime_config_settings_reads_parallelism_and_strategy_environment() {
    let mut environment = CLEAN_ENVIRONMENT;
    environment[16].1 = Some("4");
    environment[17].1 = Some("3");
    environment[18].1 = Some("3");
    environment[19].1 = Some("2");
    environment[20].1 = Some("true");
    environment[21].1 = Some("64");
    environment[22].1 = Some("fused_projection");
    environment[23].1 = Some("required");
    environment[28].1 = Some("false");
    temp_env::with_vars(environment, || {
        let config = RuntimeConfig::from_env().expect("policy environment parses");
        assert_eq!(config.cpu_workers, 4);
        assert_eq!(config.independent_ops, 3);
        assert_eq!(config.parallel_sequences, 3);
        assert_eq!(config.max_in_flight, 2);
        assert!(config.overlap_transfer_compute);
        assert_eq!(config.max_device_bytes, Some(64));
        assert_eq!(config.decode_kernel, DecodeKernelStrategy::FusedProjection);
        assert_eq!(config.fusion, FusionStrategy::Required);
        assert!(!config.moe_grouping);
    });
}

#[test]
fn unified_runtime_config_rejects_zero_cpu_workers() {
    let message = validation_text(RuntimeConfig::builder().with_cpu_workers(0).build());
    assert!(message.contains("cpu_workers"));
}

#[test]
fn unified_runtime_config_rejects_zero_independent_ops() {
    let message = validation_text(RuntimeConfig::builder().with_independent_ops(0).build());
    assert!(message.contains("independent_ops"));
}

#[test]
fn unified_runtime_config_rejects_zero_parallel_sequences() {
    let message = validation_text(RuntimeConfig::builder().with_parallel_sequences(0).build());
    assert!(message.contains("parallel_sequences"));
}

#[test]
fn unified_runtime_config_rejects_zero_in_flight_submissions() {
    let message = validation_text(RuntimeConfig::builder().with_max_in_flight(0).build());
    assert!(message.contains("max_in_flight"));
}

#[test]
fn unified_runtime_config_rejects_zero_device_residency_budget() {
    let message = validation_text(
        RuntimeConfig::builder()
            .with_max_device_bytes(Some(0))
            .build(),
    );
    assert!(message.contains("max_device_bytes"));
}

#[test]
fn unified_runtime_config_keeps_numeric_permission_separate_from_kernel_strategy() {
    let config = RuntimeConfig::builder()
        .with_decode_kernel(DecodeKernelStrategy::FusedProjection)
        .build()
        .expect("strategy request validates without numeric permission");
    assert_eq!(config.decode_kernel, DecodeKernelStrategy::FusedProjection);
    assert!(!config.numeric.reassociation);
}

#[test]
fn unified_runtime_config_rejects_unknown_decode_strategy() {
    let file = write_config("decode_kernel = 'speculative_magic'\n");
    assert!(RuntimeConfig::from_file(file.path()).is_err());
}

#[test]
fn unified_runtime_config_rejects_unknown_residency_strategy() {
    let file = write_config("residency = 'bloom_is_authoritative'\n");
    assert!(RuntimeConfig::from_file(file.path()).is_err());
}

#[test]
fn unified_runtime_config_explicit_fluent_value_wins_after_environment() {
    let mut environment = CLEAN_ENVIRONMENT;
    environment[16].1 = Some("3");
    temp_env::with_vars(environment, || {
        let config = RuntimeConfig::builder()
            .from_env()
            .expect("environment loads")
            .with_cpu_workers(4)
            .build()
            .expect("explicit override validates");
        assert_eq!(config.cpu_workers, 4);
    });
}

#[test]
fn unified_runtime_config_environment_value_overrides_file_value() {
    let file = write_config("cpu_workers = 2\n");
    let mut environment = CLEAN_ENVIRONMENT;
    environment[16].1 = Some("3");
    temp_env::with_vars(environment, || {
        let config = RuntimeConfig::builder()
            .from_file(file.path())
            .expect("file loads")
            .from_env()
            .expect("environment overlays file")
            .build()
            .expect("overlaid config validates");
        assert_eq!(config.cpu_workers, 3);
    });
}

#[test]
fn unified_runtime_config_resource_limits_do_not_grant_numeric_rewrites() {
    let config = RuntimeConfig::builder()
        .with_max_device_bytes(Some(64))
        .with_decode_kernel(DecodeKernelStrategy::FusedProjection)
        .build()
        .expect("resource and strategy requests validate");
    assert_eq!(config.max_device_bytes, Some(64));
    assert!(!config.numeric.reassociation);
}

#[test]
fn unified_runtime_config_defaults_select_safe_shared_strategies() {
    let config = RuntimeConfig::default();
    assert_eq!(config.decode_kernel, DecodeKernelStrategy::Automatic);
    assert_eq!(config.fusion, FusionStrategy::Automatic);
    assert_eq!(config.residency, ResidencyStrategy::ExactResidentFirst);
    assert_eq!(config.batching, BatchingStrategy::CompatibleRows);
    assert_eq!(config.speculation, SpeculationStrategy::Disabled);
    assert_eq!(config.weight_conversion, WeightConversionStrategy::Disabled);
    assert!(config.moe_grouping);
}

#[test]
fn runtime_config_pipeline_cache_builder_matches_serde_config() {
    let source = "pipeline_cache = false\npipeline_cache_dir = '/var/cache/omega-pipelines'\npipeline_cache_max_bytes = 1048576\n";
    let from_serde: RuntimeConfig =
        toml::from_str(source).expect("pipeline cache TOML deserializes");
    let from_builder = RuntimeConfig::builder()
        .with_pipeline_cache(false)
        .with_pipeline_cache_dir("/var/cache/omega-pipelines")
        .with_pipeline_cache_max_bytes(1_048_576)
        .build()
        .expect("builder config validates");

    assert_eq!(from_builder, from_serde);
    assert_eq!(
        runtime_config_difference_count(&from_builder, &RuntimeConfig::default()),
        3,
        "the difference count must see all three pipeline cache fields"
    );
}
