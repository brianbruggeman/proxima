//! Load-time warm-up of the model's device buffers.
//!
//! [`LoadedModel::warm_resident_buffers`] is the call a serving front end
//! makes right after `LoadedModel::load`, before it takes a request. It
//! composes `omega::metal::warm_resident_buffers` (one waited command buffer
//! that declares the checkpoint mapping and every resident buffer to the
//! driver) behind [`ServingConfig::warm_model_buffers_at_load`]. Call
//! `omega::metal::warm_resident_buffers` directly to warm without a
//! [`ServingConfig`]; this wrapper exists to apply the serving setting and to
//! skip the cases where decode would drop the mapping anyway.

use super::*;
#[cfg(any(test, all(feature = "metal", target_os = "macos")))]
use crate::serving::GPU_LAYERS_ALL;

/// Whether a warm-up should run for this configuration: the setting is on,
/// the GPU engine is selected, and decode will keep the checkpoint mapping
/// (the routed pre-gather path unregisters it, see
/// `LoadedModel::drive_serving_loop`, so warming it would build a
/// whole-checkpoint buffer that is dropped before use).
#[cfg(any(test, all(feature = "metal", target_os = "macos")))]
pub(super) fn should_warm(serving_config: &ServingConfig<'_>, routed_experts: bool) -> bool {
    let pre_gather = moe_pre_gather_enabled(serving_config.moe_pre_gather, routed_experts);
    let mapping_dropped = pre_gather && !serving_config.moe_monolithic_high_mmap;
    serving_config.warm_model_buffers_at_load
        && serving_config.gpu_layers == GPU_LAYERS_ALL
        && !mapping_dropped
}

impl LoadedModel<'_> {
    /// Declares this model's device buffers to the driver now instead of in
    /// the first command buffer that references them, when
    /// [`ServingConfig::warm_model_buffers_at_load`] asks for it. Returns how
    /// many buffers were declared; `Ok(0)` when the setting is off or the
    /// configuration does not keep the mapping. Must run on the thread that
    /// will serve: the buffer caches are thread-local.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn warm_resident_buffers(
        &self,
        serving_config: &ServingConfig<'_>,
    ) -> Result<usize, InteropError> {
        if !should_warm(serving_config, self.ffn_routing == FfnRouting::Routed) {
            return Ok(0);
        }
        omega::metal::warm_resident_buffers().map_err(InteropError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu_config() -> ServingConfig<'static> {
        ServingConfig {
            gpu_layers: GPU_LAYERS_ALL,
            ..ServingConfig::default()
        }
    }

    #[test]
    fn warm_up_runs_for_a_gpu_dense_model_by_default() {
        assert!(should_warm(&gpu_config(), false));
    }

    #[test]
    fn warm_up_is_skipped_when_the_setting_is_off() {
        let config = ServingConfig {
            warm_model_buffers_at_load: false,
            ..gpu_config()
        };
        assert!(!should_warm(&config, false));
    }

    #[test]
    fn warm_up_is_skipped_on_the_cpu_engine() {
        let config = ServingConfig {
            gpu_layers: 0,
            ..ServingConfig::default()
        };
        assert!(!should_warm(&config, false));
    }

    #[test]
    fn warm_up_is_skipped_when_routed_pre_gather_drops_the_mapping() {
        let config = ServingConfig {
            moe_pre_gather: true,
            ..gpu_config()
        };
        assert!(!should_warm(&config, true));
        assert!(should_warm(&config, false));
    }

    #[test]
    fn warm_up_runs_for_routed_experts_when_the_mapping_is_kept() {
        let config = ServingConfig {
            moe_pre_gather: true,
            moe_monolithic_high_mmap: true,
            ..gpu_config()
        };
        assert!(should_warm(&config, true));
    }
}
