use std::collections::BTreeMap;

use super::binding::{bind_ordinal, release_after_bind};
use super::buffer::{offset_pointer, window_end};
use super::device::{HipDevice, compile_options, numeric_policy, open_ordinal};
use super::error::HipDriverError;
use super::shared::{resolve_wave, shared_driver};
use crate::OperandDtypes;
use crate::config::RuntimeConfig;
use crate::cuda::emit_hip_at_wave;
use crate::cuda::{EmitDevice, HipWave};
use crate::ffi::hip::{
    DevicePointer, HIP_DEVICE_PROPERTIES_BYTES, HipApi, gcn_arch_name, hiprtc_windows_candidates,
};
use proxima_numeric::{DType, NumericPolicy};
use proxima_tensor::{
    Extent, IndexMap, Keep, Op, Reduce, ReduceInit, ScalarOp, append, bind, infer, map,
};

fn activation_dtype() -> DType {
    proxima_numeric::Numerics::default()
        .activation
        .to_dtype()
        .expect("activation format has a dtype")
}

#[test]
fn hip_compile_options_name_the_open_devices_gcn_arch() {
    assert_eq!(
        compile_options(Some("gfx90a:sramecc-:xnack-"), &[]),
        vec![String::from("--offload-arch=gfx90a:sramecc-:xnack-")]
    );
}

#[test]
fn hip_compile_options_let_an_explicit_arch_override_the_device() {
    assert_eq!(
        compile_options(Some("gfx942"), &["--offload-arch=gfx90a"]),
        vec![String::from("--offload-arch=gfx90a")]
    );
    assert_eq!(
        compile_options(Some("gfx942"), &["--gpu-architecture=gfx90a"]),
        vec![String::from("--gpu-architecture=gfx90a")]
    );
}

#[test]
fn hip_compile_options_omit_the_arch_when_the_property_is_unreadable() {
    assert!(compile_options(None, &[]).is_empty());
    assert_eq!(
        compile_options(Some(""), &["-O3"]),
        vec![String::from("-O3")]
    );
}

#[test]
fn hip_properties_scan_finds_the_gcn_arch_name() {
    let token = b"gfx90a:sramecc-:xnack-";
    let mut properties = [0_u8; HIP_DEVICE_PROPERTIES_BYTES];
    properties[512..512 + token.len()].copy_from_slice(token);
    properties[512 + token.len()] = 0;

    assert_eq!(gcn_arch_name(&properties), Some("gfx90a:sramecc-:xnack-"));
    assert_eq!(gcn_arch_name(b"no architecture token here\0"), None);
    assert_eq!(
        gcn_arch_name(b"gfx\0"),
        None,
        "a bare gfx is not an arch name"
    );
}

#[test]
fn hip_numeric_policy_is_the_conflaguration_policy_with_the_shared_narrowing() {
    let numeric = RuntimeConfig::default().numeric;

    assert_eq!(
        numeric_policy(numeric),
        numeric
            .with_accumulator_narrowing()
            .with_flash_narrowing(true)
    );
}

#[test]
fn hip_open_ordinal_converts_the_configured_device() {
    assert!(matches!(open_ordinal(1), Ok(1)));
    assert!(matches!(open_ordinal(i32::MAX as u32), Ok(i32::MAX)));
    assert!(matches!(
        open_ordinal(i32::MAX as u32 + 1),
        Err(HipDriverError::DeviceOrdinalTooLarge(_))
    ));
    assert!(matches!(
        open_ordinal(u32::MAX),
        Err(HipDriverError::DeviceOrdinalTooLarge(u32::MAX))
    ));
}

#[test]
fn a_view_window_ends_inside_the_allocation_or_is_refused() {
    assert_eq!(
        window_end(64, 0, 64),
        Some(64),
        "the whole allocation is a window"
    );
    assert_eq!(
        window_end(64, 16, 48),
        Some(64),
        "a tail window ends on the last byte"
    );
    assert_eq!(
        window_end(64, 64, 0),
        Some(64),
        "an empty window at the end is legal"
    );
    assert_eq!(
        window_end(64, 16, 49),
        None,
        "one byte past the end is refused"
    );
    assert_eq!(
        window_end(64, 65, 0),
        None,
        "an offset past the end is refused"
    );
    assert_eq!(
        window_end(64, usize::MAX, 2),
        None,
        "an overflowing window is refused, not wrapped"
    );
}

#[test]
fn a_view_pointer_advances_by_bytes_not_elements() {
    let base = DevicePointer::from_address(0x1000);

    assert_eq!(offset_pointer(base, 0), base);
    assert_eq!(offset_pointer(base, 12).address(), 0x100c);
    assert_eq!(offset_pointer(base, 4096).address(), 0x2000);
}

#[test]
fn availability_is_whether_the_runtime_loads() {
    assert_eq!(HipApi::is_available(), HipApi::load().is_ok());
    if !HipApi::is_available() {
        assert!(matches!(
            HipApi::load(),
            Err(HipDriverError::RuntimeNotInstalled { .. } | HipDriverError::RtcNotInstalled { .. })
        ));
    }
}

#[test]
fn an_absent_runtime_is_a_typed_error_naming_the_sonames_tried() {
    match HipApi::load() {
        // a host with ROCm installed: the absent-runtime contract is
        // unexercised and the device test below is the live one
        Ok(_) => {}
        Err(HipDriverError::RuntimeNotInstalled { tried }) => {
            assert!(tried.contains("amdhip64"), "tried: {tried}");
        }
        Err(other) => panic!("unexpected load error: {other:?}"),
    }
}

#[test]
fn the_windows_hiprtc_name_is_derived_from_the_runtime_version_with_a_plain_fallback() {
    assert_eq!(
        hiprtc_windows_candidates(50_300_000),
        vec![String::from("hiprtc0503.dll"), String::from("hiprtc.dll")]
    );
    assert_eq!(
        hiprtc_windows_candidates(60_241_133),
        vec![String::from("hiprtc0602.dll"), String::from("hiprtc.dll")]
    );
    assert_eq!(
        hiprtc_windows_candidates(0),
        vec![String::from("hiprtc.dll")]
    );
}

fn count_binds(ordinals: &[i32]) -> Vec<i32> {
    let ordinals = ordinals.to_vec();
    std::thread::spawn(move || {
        let mut calls = Vec::new();
        for ordinal in ordinals {
            bind_ordinal(ordinal, |bound| {
                calls.push(bound);
                Ok::<(), HipDriverError>(())
            })
            .expect("the stand-in bind succeeds");
        }
        calls
    })
    .join()
    .expect("the bind thread runs")
}

#[test]
fn a_thread_binds_its_device_once_and_skips_the_repeat() {
    assert_eq!(count_binds(&[1, 1, 1]), vec![1]);
}

#[test]
fn a_different_ordinal_binds_again_and_a_return_rebinds() {
    assert_eq!(count_binds(&[0, 0, 1, 1, 0]), vec![0, 1, 0]);
}

#[test]
fn a_failed_bind_is_not_remembered() {
    let calls = std::thread::spawn(|| {
        let mut calls = 0;
        let mut attempt = |ok: bool| {
            bind_ordinal(2, |_| {
                calls += 1;
                if ok {
                    Ok(())
                } else {
                    Err(HipDriverError::BindingContract)
                }
            })
        };
        assert!(attempt(false).is_err());
        assert!(attempt(true).is_ok());
        assert!(attempt(true).is_ok());
        calls
    })
    .join()
    .expect("the bind thread runs");
    assert_eq!(calls, 2);
}

#[test]
fn a_release_runs_only_after_its_device_is_bound() {
    let (bound, released_after_bind, released_after_failure) = std::thread::spawn(|| {
        let mut bound = Vec::new();
        let mut released_after_bind = false;
        release_after_bind(
            3,
            |ordinal| {
                bound.push(ordinal);
                Ok::<(), HipDriverError>(())
            },
            || released_after_bind = true,
        );
        let mut released_after_failure = false;
        release_after_bind(
            4,
            |_| Err(HipDriverError::BindingContract),
            || released_after_failure = true,
        );
        (bound, released_after_bind, released_after_failure)
    })
    .join()
    .expect("the release thread runs");
    assert_eq!(bound, vec![3], "the drop binds the object's own ordinal");
    assert!(released_after_bind, "a bound device runs the release");
    assert!(!released_after_failure, "a failed bind skips the release");
}

#[test]
fn an_unconfigured_wave_takes_the_probed_width() {
    assert_eq!(
        resolve_wave(None, HipWave::Wave32).expect("probe wins"),
        HipWave::Wave32
    );
    assert_eq!(
        resolve_wave(None, HipWave::Wave64).expect("probe wins"),
        HipWave::Wave64
    );
}

#[test]
fn a_configured_wave_equal_to_the_probe_is_accepted() {
    assert_eq!(
        resolve_wave(Some(HipWave::Wave32), HipWave::Wave32).expect("equal"),
        HipWave::Wave32
    );
}

#[test]
fn a_configured_wave_that_differs_from_the_device_names_both_widths() {
    let error = resolve_wave(Some(HipWave::Wave64), HipWave::Wave32).expect_err("mismatch refused");
    assert!(
        matches!(
            error,
            HipDriverError::WaveMismatch {
                configured: 64,
                device: 32
            }
        ),
        "{error:?}"
    );
    let message = error.to_string();
    assert!(
        message.contains("64") && message.contains("32"),
        "{message}"
    );
}

#[test]
fn the_smoke_matches_the_host_oracle_when_a_device_is_present() {
    let Ok(driver) = shared_driver(&RuntimeConfig::default(), None) else {
        return;
    };
    let input = [0.0, 1.0, -2.5, 42.0, 1e6];
    let output = driver
        .smoke_add_one(&input)
        .expect("the shared smoke kernel runs through hiprtc");
    assert_eq!(output.len(), input.len());
    for (index, (found, expected)) in output.iter().zip(input.iter()).enumerate() {
        assert!(
            (found - (expected + 1.0)).abs() < 1e-6,
            "element {index}: {found} vs {}",
            expected + 1.0
        );
    }
}

/// hiprtc 9.x static-asserts that a `_sync` mask is 8 bytes at every wavefront width. This
/// compiles the emitted reduce at both widths through the same runtime path a plan uses;
/// a host without ROCm names the skip instead of pretending to have compiled.
#[test]
fn hiprtc_compiles_the_emitted_wave32_and_wave64_masks() {
    let dtype = activation_dtype();
    let mut program = Vec::new();
    let input = append(
        &mut program,
        Op::Input {
            dtype,
            shape: vec![Extent::Static(4), Extent::Static(4096)],
            name: None,
        },
    );
    let root = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: input,
            in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
            out_map: IndexMap::Affine(map::projection(2, &[0])),
            keep: Keep::Reduce,
            name: Some("hiprtc_mask_reduce".into()),
        }),
    );
    let shapes = infer(&program, &[]).expect("the reduce infers");
    let bound = bind(&program, &shapes, &[root], NumericPolicy::llama_relaxed())
        .expect("the reduce binds under the reassociation policy");
    let packed = BTreeMap::new();
    let device = match HipDevice::open(0) {
        Ok(device) => device,
        Err(HipDriverError::RuntimeNotInstalled { tried }) => {
            eprintln!("skipped: rocm runtime not installed ({tried})");
            return;
        }
        Err(error) => panic!("a ROCm host failed to open device 0: {error}"),
    };
    for wave in [HipWave::Wave32, HipWave::Wave64] {
        let kernel = emit_hip_at_wave(
            wave,
            &bound[0],
            &packed,
            NumericPolicy::llama_relaxed(),
            EmitDevice::default(),
            &OperandDtypes::new(),
        )
        .expect("the reduce emits");
        device
            .compile(&kernel.source, &[])
            .unwrap_or_else(|error| panic!("{wave:?} mask did not compile: {error}"));
    }
}
