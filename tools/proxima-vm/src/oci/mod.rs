//! OCI images as input to the linux flavor: pull a manifest and its layers from a
//! registry, flatten them to a rootfs, archive that as the initramfs the kernel unpacks.
//! Nothing here is vm-wide; the elf flavor and a future proxima-os flavor take other inputs.
//!
//! Every network read is one call of the transport pipe (`transport::fetch`, a
//! `Request -> Response` function) and every blob is checked against its sha256 digest
//! before it is cached or unpacked.

pub mod cache;
pub mod digest;
pub mod error;
pub mod initramfs;
pub mod manifest;
pub mod reference;
pub mod registry;
pub mod rootfs;
pub mod runspec;
pub mod transport;

use std::path::Path;

use prime::os::core_shard::run_on_this_thread;
use proxima_runtime::CoreId;

pub use error::OciError;
pub use reference::ImageRef;
pub use registry::PulledImage;

const PLATFORM_OS: &str = "linux";
const PLATFORM_ARCHITECTURE: &str = "amd64";

/// Pulls `image` through the real registry transport, blocking the caller. The prime worker
/// gets a thread of its own so a caller already inside an executor never nests reactors.
pub fn pull_blocking(cache_root: &Path, image: &ImageRef) -> Result<PulledImage, OciError> {
    let (cache_root, image) = (cache_root.to_path_buf(), image.clone());
    let worker = std::thread::spawn(move || {
        run_on_this_thread(CoreId(0), async move {
            registry::pull(&transport::fetch, &cache_root, &image, PLATFORM_OS, PLATFORM_ARCHITECTURE).await
        })
    });
    worker
        .join()
        .map_err(|_| OciError::Runtime("the pull worker thread panicked".into()))?
        .map_err(|error| OciError::Runtime(error.to_string()))?
}

/// The newc initramfs for a pulled image, with `init` as `/init`.
pub fn initramfs_for(pulled: &PulledImage, init: &[u8]) -> Result<Vec<u8>, proxima_core::ProximaError> {
    let mut tree = rootfs::flatten(&pulled.layers)?;
    initramfs::add_guest_files(&mut tree, init, &runspec::RunSpec::from_config(&pulled.run)?)?;
    initramfs::to_initramfs(&tree)
}
