use std::collections::BTreeMap;
use std::sync::OnceLock;

use proxima_tensor::{NodeId, Op, QuantizedBlock};

use crate::config::RuntimeConfig;
use crate::cuda::HipWave;
use crate::cuda_driver::{CudaDriver, CudaDriverError};

use super::error::HipDriverError;
use super::runtime::{HipPlan, HipRuntime};

/// The wave `RuntimeConfig::hip_wave` selects: `None` for 0, which means the
/// device reports it. The one place the config value becomes a [`HipWave`].
#[must_use]
pub fn hip_wave_of(config: &RuntimeConfig) -> Option<HipWave> {
    HipWave::from_lanes(u64::from(config.hip_wave))
}

/// The wave a runtime emits for: the device's `probed` width, which a `configured` width must equal
/// because the emitted kernels trap on any other `warpSize`.
///
/// # Errors
/// [`HipDriverError::WaveMismatch`] naming both widths.
pub(super) fn resolve_wave(
    configured: Option<HipWave>,
    probed: HipWave,
) -> Result<HipWave, HipDriverError> {
    let width = |wave: HipWave| u32::try_from(wave.lanes()).unwrap_or(u32::MAX);
    match configured {
        Some(wave) if wave != probed => Err(HipDriverError::WaveMismatch {
            configured: width(wave),
            device: width(probed),
        }),
        _ => Ok(probed),
    }
}

type HipDriverKey = (u32, Option<u64>);
type HipDriverTable =
    proxima_primitives::sync::blocking::Mutex<BTreeMap<HipDriverKey, CudaDriver<HipRuntime>>>;

static HIP_DRIVERS: OnceLock<HipDriverTable> = OnceLock::new();

/// The process's one driver for `(config.hip_device, wave)`, opened on first use: the HIP
/// counterpart of the single CUDA driver, so every plan shares its device context, weight ledger
/// and kernel cache. Clones share state (every field is an `Arc`). A failed open is not cached,
/// so a missing runtime stays the typed [`HipDriverError`] on each call.
///
/// # Errors
/// The refusals of [`HipRuntime::open`].
pub fn shared_driver(
    config: &RuntimeConfig,
    wave: Option<HipWave>,
) -> Result<CudaDriver<HipRuntime>, HipDriverError> {
    let drivers =
        HIP_DRIVERS.get_or_init(|| proxima_primitives::sync::blocking::Mutex::new(BTreeMap::new()));
    let key = (config.hip_device, wave.map(HipWave::lanes));
    let mut open = drivers.lock();
    if let Some(driver) = open.get(&key) {
        return Ok(driver.clone());
    }
    let driver = CudaDriver::<HipRuntime>::open(config, wave)?;
    open.insert(key, driver.clone());
    Ok(driver)
}

/// Plans `program` on the shared executor over [`shared_driver`]: the same `plan_with_config`
/// path a CUDA device takes, under the HIP grant of the resolved wave.
///
/// # Errors
/// The refusals of [`shared_driver`] and of `plan_with_config`.
pub fn plan(
    config: &RuntimeConfig,
    wave: Option<HipWave>,
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, QuantizedBlock<'_>)],
    outputs: &[NodeId],
) -> Result<HipPlan, CudaDriverError> {
    let driver = shared_driver(config, wave)?;
    driver.plan_with_config(program, symbols, outputs, named, config)
}
