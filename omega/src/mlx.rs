//! Compile-time MLX capability discovered by `omega/build.rs`.
//!
//! This module exposes only the provisioned installation path. Execution
//! remains behind a native adapter that consumes the backend-neutral tensor
//! contract; no model-family or parallel algebra is introduced here.

/// The MLX installation selected by `PROXIMA_MLX_PREFIX` at compile time.
pub const PREFIX: &str = env!("PROXIMA_MLX_PREFIX");

/// Native MLX is present in this build.
pub const AVAILABLE: bool = true;

use proxima_tensor::TensorError;
use proxima_tensor::gdn::GdnPrefillScan;

unsafe extern "C" {
    fn proxima_mlx_bridge_name() -> *const core::ffi::c_char;
    fn proxima_mlx_gdn_scan(
        query: *const f32,
        key: *const f32,
        value: *const f32,
        gate: *const f32,
        beta: *const f32,
        state: *mut f32,
        output: *mut f32,
        positions: i32,
        key_dim: i32,
        value_dim: i32,
        heads: i32,
        inv_sqrt_key_dim: f32,
    ) -> i32;
}

/// Confirms that the native MLX bridge was linked into this binary.
#[must_use]
pub fn linked() -> bool {
    // SAFETY: the build script emits this symbol only with the MLX bridge.
    unsafe { !proxima_mlx_bridge_name().is_null() }
}

/// Runs the GDN recurrence through the linked MLX native runtime.
pub fn run_gdn_prefill_scan(scan: GdnPrefillScan<'_>) -> Result<(), TensorError> {
    let shape = scan.shape;
    let status = unsafe {
        proxima_mlx_gdn_scan(
            scan.query.as_ptr(),
            scan.key.as_ptr(),
            scan.value.as_ptr(),
            scan.gate.as_ptr(),
            scan.beta.as_ptr(),
            scan.state.as_mut_ptr(),
            scan.output.as_mut_ptr(),
            i32::try_from(shape.positions).map_err(|_| TensorError::InvalidGdnPrefillShape {
                reason: "positions exceed MLX bridge integer range",
            })?,
            i32::try_from(shape.key_dim).map_err(|_| TensorError::InvalidGdnPrefillShape {
                reason: "key dimension exceeds MLX bridge integer range",
            })?,
            i32::try_from(shape.value_dim).map_err(|_| TensorError::InvalidGdnPrefillShape {
                reason: "value dimension exceeds MLX bridge integer range",
            })?,
            i32::try_from(shape.heads).map_err(|_| TensorError::InvalidGdnPrefillShape {
                reason: "head count exceeds MLX bridge integer range",
            })?,
            scan.inv_sqrt_key_dim,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(TensorError::InvalidGdnPrefillShape {
            reason: "MLX bridge rejected the GDN dimensions",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proxima_tensor::cpu::run_gdn_prefill_scan as run_cpu;
    use proxima_tensor::gdn::{GdnPrefillScan, GdnPrefillShape};

    #[test]
    fn mlx_matches_cpu_on_multi_position_scan() {
        let shape = GdnPrefillShape {
            positions: 3,
            key_dim: 4,
            value_dim: 5,
            heads: 2,
            kv_heads: 2,
        };
        let query: Vec<f32> = (0..24)
            .map(|index| (index as f32 - 7.0) * 0.03125)
            .collect();
        let key: Vec<f32> = (0..24)
            .map(|index| (index as f32 + 3.0) * 0.015625)
            .collect();
        let value: Vec<f32> = (0..30)
            .map(|index| (index as f32 - 11.0) * 0.0625)
            .collect();
        let gate: Vec<f32> = (0..6).map(|index| 0.1 + index as f32 * 0.03).collect();
        let beta: Vec<f32> = (0..6).map(|index| 0.2 + index as f32 * 0.02).collect();
        let mut cpu_state = vec![0.0; shape.state_len()];
        let mut cpu_output = vec![0.0; shape.output_len()];
        let mut mlx_state = vec![0.0; shape.state_len()];
        let mut mlx_output = vec![0.0; shape.output_len()];
        run_cpu(GdnPrefillScan {
            shape,
            query: &query,
            key: &key,
            query_key_head_stride: 1,
            query_key_dim_stride: 2,
            value: &value,
            gate: &gate,
            beta: &beta,
            inv_sqrt_key_dim: 0.5,
            state: &mut cpu_state,
            output: &mut cpu_output,
        })
        .expect("cpu recurrence accepts the fixture");
        run_gdn_prefill_scan(GdnPrefillScan {
            shape,
            query: &query,
            key: &key,
            query_key_head_stride: 1,
            query_key_dim_stride: 2,
            value: &value,
            gate: &gate,
            beta: &beta,
            inv_sqrt_key_dim: 0.5,
            state: &mut mlx_state,
            output: &mut mlx_output,
        })
        .expect("mlx recurrence accepts the fixture");
        let state_error = cpu_state
            .iter()
            .zip(&mlx_state)
            .map(|(cpu, mlx)| (cpu - mlx).abs())
            .fold(0.0_f32, f32::max);
        let output_error = cpu_output
            .iter()
            .zip(&mlx_output)
            .map(|(cpu, mlx)| (cpu - mlx).abs())
            .fold(0.0_f32, f32::max);
        assert!(state_error <= 1.0e-5, "state error {state_error}");
        assert!(output_error <= 1.0e-5, "output error {output_error}");
    }
}
