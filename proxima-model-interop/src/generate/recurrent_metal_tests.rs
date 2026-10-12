// workspace denies expect_used; a failed fixture must abort the test with its message
#![allow(clippy::expect_used)]

//! The tiny recurrent checkpoint on the real device: the graph the CPU
//! evaluates in `recurrent_mainline_tests` must read back the same values from
//! Metal, because the backend gets the same lowered program and no recurrent
//! knowledge of its own.

use core::ops::ControlFlow;

use proxima_gguf::parse_complete;
use proxima_tensor::LoopCdMode;

use super::LoadedModel;
use super::ouro_fixture::tiny_ouro_gguf;
use crate::recurrent_config::RecurrentConfig;
use crate::serving::{GPU_LAYERS_ALL, ServingConfig};

fn on_device(recurrent: RecurrentConfig) -> ServingConfig<'static> {
    ServingConfig {
        gpu_layers: GPU_LAYERS_ALL,
        recurrent,
        ..ServingConfig::default()
    }
}

fn on_host(recurrent: RecurrentConfig) -> ServingConfig<'static> {
    ServingConfig {
        gpu_layers: 0,
        recurrent,
        ..ServingConfig::default()
    }
}

fn assert_close(device: &[f32], host: &[f32]) {
    assert_eq!(device.len(), host.len());
    for (index, (device, host)) in device.iter().zip(host).enumerate() {
        assert!(
            (device - host).abs() <= 1.0e-5 + 1.0e-5 * host.abs(),
            "coordinate {index}: device={device} host={host}"
        );
    }
}

#[test]
fn recurrent_metal_cached_literal() {
    let bytes = tiny_ouro_gguf(2, false);
    let parsed = parse_complete(&bytes).expect("parse the tiny recurrent GGUF");
    let model = LoadedModel::load(&parsed, &bytes).expect("load");
    let neutral = RecurrentConfig::neutral();
    let nodes = [model.logits_root()];

    let device = model
        .forward_node_values_from_ids(&[0], &nodes, &on_device(neutral))
        .expect("evaluate on the device");
    let host = model
        .forward_node_values_from_ids(&[0], &nodes, &on_host(neutral))
        .expect("evaluate on the host");
    let (generated, _, _) = model
        .generate_from_ids(&[0], 2, &on_device(neutral), &mut |_| {
            ControlFlow::Continue(())
        })
        .expect("prefill then decode on the device");

    assert_close(&device[0], &host[0]);
    assert_eq!(generated, [1, 1]);
}

#[test]
fn recurrent_metal_repeat_identity() {
    let bytes = tiny_ouro_gguf(2, false);
    let parsed = parse_complete(&bytes).expect("parse the tiny recurrent GGUF");
    let model = LoadedModel::load(&parsed, &bytes).expect("load");
    let config = on_device(RecurrentConfig::neutral());
    let decode = |model: &LoadedModel<'_>| {
        model
            .generate_from_ids(&[0], 2, &config, &mut |_| ControlFlow::Continue(()))
            .expect("generate on the device")
            .0
    };

    let first = decode(&model);
    let second = decode(&model);

    assert_eq!(
        first, second,
        "a repeated request on one model is the same request"
    );
}

#[test]
fn recurrent_metal_loopcd_readouts() {
    let bytes = tiny_ouro_gguf(2, false);
    let parsed = parse_complete(&bytes).expect("parse the tiny recurrent GGUF");
    let contrast = RecurrentConfig {
        iterations: None,
        mode: LoopCdMode::FixedLogits,
        strength: 0.5,
        reference_iteration: 1,
    };
    let model = LoadedModel::load_with_recurrent_config(&parsed, &bytes, contrast)
        .expect("load with fixed-logit guidance");
    let roots = *model.recurrent_readout_roots().expect("readout roots");
    let nodes = [
        model.logits_root(),
        roots.reference_logits.expect("reference head"),
        roots.terminal_logits.expect("terminal head"),
    ];

    let device = model
        .forward_node_values_from_ids(&[0], &nodes, &on_device(contrast))
        .expect("evaluate the readouts on the device");
    let host = model
        .forward_node_values_from_ids(&[0], &nodes, &on_host(contrast))
        .expect("evaluate the readouts on the host");

    for (device, host) in device.iter().zip(&host) {
        assert_close(device, host);
    }
}
