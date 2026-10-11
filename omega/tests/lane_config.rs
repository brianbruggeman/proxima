//! The lane options `compute_units`, `precision` and `model_cache_dir` are
//! first-class in both config surfaces: the fluent builder and the
//! conflaguration loaders (environment and file) build identical configs, and
//! `builder_from` overrides one seeded field.

#![cfg(all(feature = "onnx-door", feature = "mil-writer"))]
// every expect below runs against a config this file builds itself; a failure there IS the test failing.
#![allow(clippy::expect_used)]

use std::path::PathBuf;

use omega::onnx_door::{ComputeUnits, DoorConfig, DoorConfigError, Precision};

const PACKAGE_DIR: &str = "/Users/brianbruggeman/.cache/proxima/coreml-lane";

const LANE_VARIABLES: [&str; 3] = [
    "OMEGA_ONNX_DOOR_COMPUTE_UNITS",
    "OMEGA_ONNX_DOOR_PRECISION",
    "OMEGA_ONNX_DOOR_MODEL_CACHE_DIR",
];

fn built_for_all_units() -> DoorConfig {
    DoorConfig::builder()
        .compute_units("all")
        .precision("float32")
        .model_cache_dir(PACKAGE_DIR)
        .build()
}

#[test]
fn lane_config_builder_equals_environment_loader() {
    let set = [
        (LANE_VARIABLES[0], Some("all")),
        (LANE_VARIABLES[1], Some("float32")),
        (LANE_VARIABLES[2], Some(PACKAGE_DIR)),
    ];

    let loaded = temp_env::with_vars(set, DoorConfig::from_env).expect("set env loads");

    assert_eq!(loaded, built_for_all_units());
}

#[test]
fn lane_config_builder_equals_file_loader() {
    let directory = tempfile::tempdir().expect("a scratch directory");
    let path = directory.path().join("door.toml");
    let body = format!(
        "compute_units = \"all\"\nprecision = \"float32\"\nmodel_cache_dir = \"{PACKAGE_DIR}\"\n"
    );
    std::fs::write(&path, body).expect("the config file is written");

    let loaded = DoorConfig::from_file(&path).expect("the file loads");

    assert_eq!(loaded, built_for_all_units());
}

#[test]
fn lane_config_builder_from_config_overrides_one_seeded_field() {
    let seeded = built_for_all_units();

    let tweaked = DoorConfig::builder_from(&seeded)
        .compute_units("cpu_only")
        .build();

    assert_eq!(tweaked.compute_units, "cpu_only");
    assert_eq!(tweaked.precision, seeded.precision);
    assert_eq!(tweaked.model_cache_dir, seeded.model_cache_dir);
    assert_eq!(tweaked.engine, seeded.engine);
}

#[test]
fn lane_config_default_equals_builder_build() {
    let unset = LANE_VARIABLES.map(|name| (name, None::<&str>));

    let from_env = temp_env::with_vars(unset, DoorConfig::from_env).expect("empty env loads");
    let defaults = DoorConfig::default();
    let settings = defaults.settings().expect("the defaults resolve");

    assert_eq!(defaults, DoorConfig::builder().build());
    assert_eq!(from_env, defaults);
    assert_eq!(settings.compute_units, ComputeUnits::CpuAndNeuralEngine);
    assert_eq!(settings.precision, Precision::Float16);
    assert_eq!(settings.model_cache_dir, None);
}

#[test]
fn lane_config_unknown_compute_units_name_is_rejected_naming_it() {
    let config = DoorConfig::builder().compute_units("ane_only").build();

    let error = config.settings().expect_err("ane_only is not a unit");

    assert!(matches!(
        &error,
        DoorConfigError::UnknownComputeUnits { name } if name == "ane_only"
    ));
    assert!(error.to_string().contains("ane_only"));
}

#[test]
fn lane_config_precision_other_than_float16_is_rejected_for_cpu_and_neural_engine() {
    let rejected = DoorConfig::builder()
        .compute_units("cpu_and_neural_engine")
        .precision("float32")
        .build();
    let accepted = DoorConfig::builder()
        .compute_units("cpu_only")
        .precision("float32")
        .model_cache_dir(PACKAGE_DIR)
        .build();

    let error = rejected
        .settings()
        .expect_err("the engine computes float16 only");
    let settings = accepted.settings().expect("cpu_only may compute float32");

    assert!(matches!(
        error,
        DoorConfigError::PrecisionUnsupported { .. }
    ));
    assert_eq!(settings.precision, Precision::Float32);
    assert_eq!(settings.model_cache_dir, Some(PathBuf::from(PACKAGE_DIR)));
}
