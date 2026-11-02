//! `vm` upstream: a lambda as a registry entry, not a binary. `[upstreams.my-fn]
//! type = "vm"` in a proxima config boots a guest ELF through
//! `proxima_vm::dispatch::run_dispatch_loop` -- the M1 driver this module
//! composes, never rewrites (`tools/proxima-vm/ROADMAP.md`'s M2 section).
//!
//! One call is one invocation: the request body is the guest's input (handle
//! 0 of the fd-keyed `ChildRequest` protocol for an elf guest, a length-framed
//! COM2 channel for a Linux guest), the bytes it produces are the response
//! body. The guest files are read and validated once, when the factory builds
//! the pipe: an ELF image (`guest_image_path`), or a Linux bzImage plus the
//! initramfs archived around its `/init` (`kernel_path` + `init_path`), or the
//! same bzImage with an OCI image pulled, digest-verified and flattened into the
//! initramfs (`kernel_path` + `image` + `oci_init_path`; `proxima_vm::oci`).
//!
//! Mirrors [`crate::upstreams::process::ProcessConfig`] exactly: one config
//! struct deriving `Builder + Deserialize + Serialize + Settings + Validate`,
//! one factory resolved through the same `PipeFactoryRegistry` seam
//! (`src/settings/mod.rs`'s `RegistryEntry`, `src/app.rs`'s
//! `ProximaSettings.upstreams`).
//!
//! `max_hypercalls` / `output_capacity` / `guest_memory_size` are exactly the
//! parameters `run_dispatch_loop` accepts -- every field here traces to a real
//! parameter of the composed driver (principle 12). The PT_LOAD segment cap
//! and the four device-channel (mmio/net/blk/pl011) capacities are not
//! exposed: the elf guest never drives a device, and `parse_elf`'s
//! `MAX_SEGMENTS` is a const generic.
//!
//! The VM run is synchronous and blocks the calling thread for milliseconds.
//! When a runtime is installed (`crate::runtime::install_runtime`, what
//! `#[proxima::main]` does) the factory wraps the pipe in
//! [`proxima_listen::Offload`], which moves each call onto the runtime's
//! background pool via `Runtime::spawn_background_blocking`. `PipeFactory::
//! build` receives no runtime handle, so without an installed runtime the VM
//! runs on the polling thread; listeners wrap every served pipe in `Offload`
//! themselves, so a served `vm` upstream is off the executor either way.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

use bon::Builder;
use bytes::Bytes;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use proxima_listen::Offload;
use proxima_primitives::pipe::SendPipe;
use proxima_telemetry::warn;
use proxima_vm::dispatch;
use proxima_vm::elf;
use proxima_vm::linux::{self, LinuxGuestLimits};
use proxima_vm::oci;
use proxima_vm::x86_boot::SetupHeader;

use crate::error::ProximaError;
use crate::pipe::{PipeHandle, into_handle};
use crate::pipe_factory::PipeFactory;
use crate::request::{Request, Response};
use crate::runtime::Runtime;

use super::process::{RestartChoice, RestartPolicy, default_restart_delay_ms};

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
use super::vm_resident::build_resident;

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
async fn build_resident(label: &str, _spec: &VmSpec) -> Result<PipeHandle, ProximaError> {
    Err(ProximaError::Config(format!(
        "vm `{label}`: a resident guest needs linux on x86_64"
    )))
}

/// PT_LOAD segment cap `elf::parse_elf`'s const generic is instantiated
/// with. The elf guest links four (`.text`, `.rodata`, `.data`, and the
/// bss+stack segment `link-elf.ld` declares); headroom for a guest that
/// splits more.
const MAX_SEGMENTS: usize = 8;

/// Device-channel capacities `run_dispatch_loop` requires. The elf guest
/// never touches a virtio/pl011 window, so these only size buffers that stay
/// empty.
const EMITTED_CAPACITY: usize = 256;
const MMIO_EMITTED_CAPACITY: usize = 256;
const NET_EMITTED_CAPACITY: usize = 256;
const BLK_EMITTED_CAPACITY: usize = 2048;
const PL011_EMITTED_CAPACITY: usize = 256;

/// Both host mapping APIs (`hv_vm_map`, `KVM_SET_USER_MEMORY_REGION`) only
/// accept page-aligned ranges; 4 KiB is the base page size of the hosts this
/// crate drives.
const GUEST_MEMORY_ALIGNMENT: u64 = 4096;

fn default_name() -> String {
    "vm".to_string()
}

fn default_max_hypercalls() -> usize {
    16
}

fn default_output_capacity() -> usize {
    64 * 1024
}

fn default_guest_memory_size() -> u64 {
    2 * 1024 * 1024
}

/// The kernel command line the Linux guest was measured with
/// (`tools/proxima-vm/BENCH_LOG.md`, "Linux guest"): each flag there is
/// listed against the failure observed without it.
pub const DEFAULT_LINUX_CMDLINE: &str =
    "console=ttyS0 reboot=t panic=-1 nomodules i8042.noaux loglevel=6";

fn default_cmdline() -> String {
    DEFAULT_LINUX_CMDLINE.to_string()
}

fn default_image_cache_dir() -> PathBuf {
    std::env::temp_dir().join("proxima-oci-cache")
}

fn default_guest_port() -> u16 {
    80
}

fn default_connect_timeout_millis() -> u64 {
    60_000
}

fn default_modules_root() -> String {
    "/usr/lib/modules".to_string()
}

fn default_ram_mib() -> u32 {
    LinuxGuestLimits::default().ram_mib
}

fn default_watchdog_millis() -> u64 {
    LinuxGuestLimits::default().watchdog_millis
}

/// What fills the Linux flavor's initramfs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinuxRoot {
    /// One static executable archived as `/init`.
    Init { init_path: PathBuf },
    /// An OCI image reference pulled into `cache_dir`, flattened to a rootfs, with the
    /// static `oci-init` supervisor at `init_path` as `/init`.
    Image {
        reference: String,
        init_path: PathBuf,
        cache_dir: PathBuf,
    },
}

/// Which guest a [`VmSpec`] boots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuestSpec {
    /// A bare no_std ELF driven over the fd-keyed `ChildRequest` channel.
    Elf { image_path: PathBuf },
    /// A real bzImage under KVM whose initramfs is built from `root`.
    Linux {
        kernel_path: PathBuf,
        root: LinuxRoot,
        cmdline: String,
        ram_mib: u32,
        watchdog_millis: u64,
    },
}

/// Runtime spec a [`VmUpstream`] runs against -- the lowered form of
/// [`VmConfig`], mirroring [`crate::upstreams::process::ProcessSpec`]'s
/// role for `ProcessConfig`.
#[derive(Debug, Clone)]
pub struct VmSpec {
    pub guest: GuestSpec,
    pub max_hypercalls: usize,
    pub output_capacity: usize,
    pub guest_memory_size: u64,
    pub resident: bool,
    pub guest_port: u16,
    pub connect_timeout_millis: u64,
    pub modules_root: PathBuf,
    pub restart: RestartPolicy,
    pub restart_delay_ms: u64,
    pub max_restart_attempts: u32,
}

/// Typed config surface for the `vm` upstream -- either an ELF guest
/// driven through `proxima-vm`'s dispatch loop (`guest_image_path`) or a Linux
/// guest, a real kernel booted under KVM (`kernel_path` + `init_path`).
/// Exactly one of the two must be named. Mirrors [`crate::upstreams::process::
/// ProcessConfig`]'s shape: one struct deriving `Builder + Deserialize +
/// Serialize + Settings`, `#[settings(prefix = "PROXIMA_VM")]`.
#[derive(Debug, Clone, PartialEq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA_VM")]
#[builder(derive(Clone, Debug), on(String, into))]
pub struct VmConfig {
    /// Path to the ELF guest image to boot; mutually exclusive with
    /// `kernel_path`.
    #[setting(default)]
    #[serde(default)]
    pub guest_image_path: Option<String>,

    /// Path to the bzImage a Linux guest boots, e.g. `/boot/vmlinuz-linux`;
    /// mutually exclusive with `guest_image_path`.
    #[setting(default)]
    #[serde(default)]
    pub kernel_path: Option<String>,

    /// Path to the static executable that becomes the Linux guest's
    /// initramfs `/init`; required with `kernel_path`.
    #[setting(default)]
    #[serde(default)]
    pub init_path: Option<String>,

    /// OCI image the Linux guest runs, e.g. `traefik/whoami@sha256:1474...`; pulled,
    /// digest-verified and flattened into the initramfs at build. Mutually exclusive
    /// with `init_path`; requires `kernel_path` and `oci_init_path`. A call's body is
    /// the milliseconds the entrypoint is left running; the reply is `running`,
    /// `exited:<code>` or `signal:<number>`.
    #[setting(default)]
    #[serde(default)]
    pub image: Option<String>,

    /// Path to the static `oci-init` supervisor (`guests/linux-init`) that becomes `/init`
    /// beside the image; required with `image`.
    #[setting(default)]
    #[serde(default)]
    pub oci_init_path: Option<String>,

    /// Directory of the content-addressed blob cache an image is pulled into; defaults to
    /// `proxima-oci-cache` under the system temp directory. Only valid with `image`.
    #[setting(default)]
    #[serde(default)]
    pub image_cache_dir: Option<String>,

    /// Linux guest kernel command line.
    #[setting(default = "console=ttyS0 reboot=t panic=-1 nomodules i8042.noaux loglevel=6")]
    #[serde(default = "default_cmdline")]
    #[builder(default = default_cmdline())]
    pub cmdline: String,

    /// Linux guest RAM in MiB.
    #[setting(default = 128)]
    #[serde(default = "default_ram_mib")]
    #[builder(default = default_ram_mib())]
    pub ram_mib: u32,

    /// Linux guest wall-clock watchdog in milliseconds; a guest that never
    /// powers off fails the call when it expires. Per-request guests only: a
    /// `resident` guest runs until the upstream is dropped.
    #[setting(default = 10000)]
    #[serde(default = "default_watchdog_millis")]
    #[builder(default = default_watchdog_millis())]
    pub watchdog_millis: u64,

    /// Pipe label.
    #[setting(default = "vm")]
    #[serde(default = "default_name")]
    #[builder(default = default_name())]
    pub name: String,

    /// Hypercall-exit budget: the run loop reports a runaway guest instead
    /// of hanging the host once this many hypercalls have been driven. Every
    /// 256 bytes of request or response body cost one hypercall each way.
    #[setting(default = 16)]
    #[serde(default = "default_max_hypercalls")]
    #[builder(default = default_max_hypercalls())]
    pub max_hypercalls: usize,

    /// Capacity, in bytes, of the response body the guest may write; a write
    /// past it fails the call instead of truncating.
    #[setting(default = 65536)]
    #[serde(default = "default_output_capacity")]
    #[builder(default = default_output_capacity())]
    pub output_capacity: usize,

    /// Bytes of guest memory mapped for the VM; a multiple of 4096 that must
    /// cover the guest image's segments (an elf guest's `link-elf.ld` fits
    /// in 2 MiB, the M1/M6 guest's stack sits at 64 MiB). Every byte is
    /// first-touched at boot, so this is a per-call cost.
    #[setting(default = 2097152)]
    #[serde(default = "default_guest_memory_size")]
    #[builder(default = default_guest_memory_size())]
    pub guest_memory_size: u64,

    /// Boot the Linux guest once when the upstream is built and proxy every request
    /// into the HTTP server inside it, instead of booting a fresh guest per call.
    /// Only a Linux guest (`kernel_path`) can be resident.
    #[setting(default = false)]
    #[serde(default)]
    #[builder(default = false)]
    pub resident: bool,

    /// TCP port the resident guest's HTTP server listens on, e.g. `80` for whoami.
    #[setting(default = 80)]
    #[serde(default = "default_guest_port")]
    #[builder(default = default_guest_port())]
    pub guest_port: u16,

    /// Milliseconds the build waits for a resident guest to accept its first
    /// connection before failing.
    #[setting(default = 60000)]
    #[serde(default = "default_connect_timeout_millis")]
    #[builder(default = default_connect_timeout_millis())]
    pub connect_timeout_millis: u64,

    /// Directory holding one `<kernel release>` subdirectory of kernel modules,
    /// e.g. `/usr/lib/modules`; a resident guest loads its virtio net modules from there.
    #[setting(default = "/usr/lib/modules")]
    #[serde(default = "default_modules_root")]
    #[builder(default = default_modules_root())]
    pub modules_root: String,

    /// Whether a resident guest that ends is booted again. Defaults to `never`;
    /// the same policy values as the `process` upstream.
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub restart: RestartChoice,

    /// Delay before a resident guest is booted again, in ms.
    #[setting(default = 1000)]
    #[serde(default = "default_restart_delay_ms")]
    #[builder(default = default_restart_delay_ms())]
    pub restart_delay_ms: u64,

    /// Restart attempt ceiling (0 = unlimited).
    #[setting(default)]
    #[serde(default)]
    #[builder(default)]
    pub max_restart_attempts: u32,
}

impl Validate for VmConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        for (field, value) in [
            ("guest_image_path", &self.guest_image_path),
            ("kernel_path", &self.kernel_path),
            ("init_path", &self.init_path),
            ("image", &self.image),
            ("oci_init_path", &self.oci_init_path),
            ("image_cache_dir", &self.image_cache_dir),
        ] {
            if value.as_deref().is_some_and(str::is_empty) {
                errors.push(ValidationMessage::new(field, "must not be empty"));
            }
        }
        match (
            &self.guest_image_path,
            &self.kernel_path,
            &self.init_path,
            &self.image,
        ) {
            (Some(_), Some(_), _, _) => errors.push(ValidationMessage::new(
                "guest_image_path",
                "names an elf guest and kernel_path names a Linux guest; set exactly one",
            )),
            (None, None, _, _) => errors.push(ValidationMessage::new(
                "guest_image_path",
                "neither guest_image_path (elf) nor kernel_path (linux) is set",
            )),
            (Some(_), None, Some(_), _) => errors.push(ValidationMessage::new(
                "init_path",
                "only a Linux guest (kernel_path) takes an init_path",
            )),
            (Some(_), None, None, Some(_)) => errors.push(ValidationMessage::new(
                "image",
                "only a Linux guest (kernel_path) takes an image",
            )),
            (None, Some(_), Some(_), Some(_)) => errors.push(ValidationMessage::new(
                "image",
                "names an OCI image and init_path names a bare init; set exactly one",
            )),
            (None, Some(_), None, None) => errors.push(ValidationMessage::new(
                "init_path",
                "required with kernel_path",
            )),
            _ => {}
        }
        self.validate_image(&mut errors);
        self.validate_resident(&mut errors);
        if self.ram_mib == 0 {
            errors.push(ValidationMessage::new("ram_mib", "must be greater than zero"));
        }
        if self.watchdog_millis == 0 {
            errors.push(ValidationMessage::new(
                "watchdog_millis",
                "must be greater than zero",
            ));
        }
        if self.max_hypercalls == 0 {
            errors.push(ValidationMessage::new(
                "max_hypercalls",
                "must be greater than zero",
            ));
        }
        if self.output_capacity == 0 {
            errors.push(ValidationMessage::new(
                "output_capacity",
                "must be greater than zero",
            ));
        }
        if self.guest_memory_size == 0 || !self.guest_memory_size.is_multiple_of(GUEST_MEMORY_ALIGNMENT) {
            errors.push(ValidationMessage::new(
                "guest_memory_size",
                format!(
                    "must be a non-zero multiple of {GUEST_MEMORY_ALIGNMENT}; got {}",
                    self.guest_memory_size
                ),
            ));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

impl VmConfig {
    fn validate_image(&self, errors: &mut Vec<ValidationMessage>) {
        match (&self.image, &self.oci_init_path, &self.image_cache_dir) {
            (Some(reference), oci_init_path, _) => {
                if let Err(error) = proxima_vm::oci::ImageRef::parse(reference) {
                    errors.push(ValidationMessage::new("image", format!("{error}")));
                }
                if oci_init_path.is_none() {
                    errors.push(ValidationMessage::new(
                        "oci_init_path",
                        "required with image",
                    ));
                }
            }
            (None, Some(_), _) => errors.push(ValidationMessage::new(
                "oci_init_path",
                "only valid with image",
            )),
            (None, None, Some(_)) => errors.push(ValidationMessage::new(
                "image_cache_dir",
                "only valid with image",
            )),
            (None, None, None) => {}
        }
    }

    fn validate_resident(&self, errors: &mut Vec<ValidationMessage>) {
        if self.resident && self.kernel_path.is_none() {
            errors.push(ValidationMessage::new(
                "resident",
                "only a Linux guest (kernel_path) can run resident",
            ));
        }
        if !self.resident && self.restart != RestartChoice::Never {
            errors.push(ValidationMessage::new(
                "restart",
                "only a resident guest restarts",
            ));
        }
        if self.guest_port == 0 {
            errors.push(ValidationMessage::new("guest_port", "must be greater than zero"));
        }
        if self.connect_timeout_millis == 0 {
            errors.push(ValidationMessage::new(
                "connect_timeout_millis",
                "must be greater than zero",
            ));
        }
        if self.modules_root.is_empty() {
            errors.push(ValidationMessage::new("modules_root", "must not be empty"));
        }
    }

    /// Lower the wire config to the runtime [`VmSpec`].
    pub fn into_spec(self) -> Result<VmSpec, ProximaError> {
        self.validate()
            .map_err(|err| ProximaError::Config(format!("{err}")))?;
        let guest = match (
            self.guest_image_path,
            self.kernel_path,
            self.init_path,
            self.image,
        ) {
            (Some(image_path), None, None, None) => GuestSpec::Elf {
                image_path: PathBuf::from(image_path),
            },
            (None, Some(kernel_path), Some(init_path), None) => GuestSpec::Linux {
                kernel_path: PathBuf::from(kernel_path),
                root: LinuxRoot::Init {
                    init_path: PathBuf::from(init_path),
                },
                cmdline: self.cmdline,
                ram_mib: self.ram_mib,
                watchdog_millis: self.watchdog_millis,
            },
            (None, Some(kernel_path), None, Some(reference)) => {
                let init_path = self.oci_init_path.ok_or_else(|| {
                    ProximaError::Config("vm config names an image without oci_init_path".into())
                })?;
                GuestSpec::Linux {
                    kernel_path: PathBuf::from(kernel_path),
                    root: LinuxRoot::Image {
                        reference,
                        init_path: PathBuf::from(init_path),
                        cache_dir: self
                            .image_cache_dir
                            .map_or_else(default_image_cache_dir, PathBuf::from),
                    },
                    cmdline: self.cmdline,
                    ram_mib: self.ram_mib,
                    watchdog_millis: self.watchdog_millis,
                }
            }
            _ => {
                return Err(ProximaError::Config(
                    "vm config names neither or both guests".into(),
                ));
            }
        };
        Ok(VmSpec {
            guest,
            max_hypercalls: self.max_hypercalls,
            output_capacity: self.output_capacity,
            guest_memory_size: self.guest_memory_size,
            resident: self.resident,
            guest_port: self.guest_port,
            connect_timeout_millis: self.connect_timeout_millis,
            modules_root: PathBuf::from(self.modules_root),
            restart: self.restart.into(),
            restart_delay_ms: self.restart_delay_ms,
            max_restart_attempts: self.max_restart_attempts,
        })
    }
}

/// A guest ELF, driven through `proxima-vm`'s dispatch loop on every call.
/// Fresh VM per request, no snapshot, no fork -- the M2 exit criterion's own
/// baseline for later milestones to beat. The image is held in memory from
/// construction; each call re-parses its segment table (a header walk) and
/// boots a new VM.
pub struct VmUpstream {
    label: String,
    spec: VmSpec,
    guest: LoadedGuest,
}

/// The guest bytes read once at build: an ELF image, or a Linux kernel
/// with the initramfs already archived around its `/init`.
#[derive(Clone)]
enum LoadedGuest {
    Elf { image: Bytes },
    Linux { kernel: Bytes, initramfs: Bytes },
}

pub(super) fn read_guest_file(label: &str, what: &str, path: &Path) -> Result<Vec<u8>, ProximaError> {
    std::fs::read(path).map_err(|error| {
        ProximaError::Upstream(format!(
            "vm `{label}`: read {what} `{}`: {error}",
            path.display()
        ))
    })
}

fn load_elf(label: &str, image_path: &Path) -> Result<LoadedGuest, ProximaError> {
    let image = read_guest_file(label, "guest image", image_path)?;
    elf::parse_elf::<MAX_SEGMENTS>(&image).map_err(|error| {
        ProximaError::Upstream(format!(
            "vm `{label}`: parse guest ELF `{}`: {error}",
            image_path.display()
        ))
    })?;
    Ok(LoadedGuest::Elf {
        image: Bytes::from(image),
    })
}

fn initramfs_from_image(
    label: &str,
    reference: &str,
    init_path: &Path,
    cache_dir: &Path,
) -> Result<Vec<u8>, ProximaError> {
    let context = |error: &dyn std::fmt::Display| {
        ProximaError::Upstream(format!("vm `{label}`: image `{reference}`: {error}"))
    };
    let image = oci::ImageRef::parse(reference).map_err(|error| context(&error))?;
    let init = read_guest_file(label, "oci init", init_path)?;
    let pulled = oci::pull_blocking(cache_dir, &image).map_err(|error| context(&error))?;
    oci::initramfs_for(&pulled, &init).map_err(|error| context(&error))
}

fn load_linux(
    label: &str,
    kernel_path: &Path,
    root: &LinuxRoot,
) -> Result<LoadedGuest, ProximaError> {
    let kernel = read_guest_file(label, "kernel", kernel_path)?;
    SetupHeader::parse(&kernel).map_err(|error| {
        ProximaError::Upstream(format!(
            "vm `{label}`: parse kernel `{}`: {error}",
            kernel_path.display()
        ))
    })?;
    let initramfs = match root {
        LinuxRoot::Init { init_path } => {
            let init = read_guest_file(label, "init", init_path)?;
            linux::initramfs_from_init(&init).map_err(|error| {
                ProximaError::Upstream(format!(
                    "vm `{label}`: archive init `{}`: {error}",
                    init_path.display()
                ))
            })?
        }
        LinuxRoot::Image {
            reference,
            init_path,
            cache_dir,
        } => initramfs_from_image(label, reference, init_path, cache_dir)?,
    };
    Ok(LoadedGuest::Linux {
        kernel: Bytes::from(kernel),
        initramfs: Bytes::from(initramfs),
    })
}

impl VmUpstream {
    /// Reads the guest named by `spec` and validates it once: an elf image
    /// must parse as an ELF whose segments the loader accepts; a Linux kernel
    /// must carry a bzImage setup header, and its `/init` is archived into the
    /// initramfs here, not per call.
    ///
    /// # Errors
    ///
    /// Returns [`ProximaError::Upstream`] naming the label and path when a
    /// guest file cannot be read or fails validation.
    pub fn new(label: impl Into<String>, spec: VmSpec) -> Result<Self, ProximaError> {
        let label = label.into();
        let guest = match &spec.guest {
            GuestSpec::Elf { image_path } => load_elf(&label, image_path)?,
            GuestSpec::Linux {
                kernel_path, root, ..
            } => load_linux(&label, kernel_path, root)?,
        };
        Ok(Self { label, spec, guest })
    }

    #[must_use]
    pub fn spec(&self) -> &VmSpec {
        &self.spec
    }
}

/// Boots the guest with `input` as the invocation body, drives it to
/// completion, and returns the bytes it produced. A elf guest composes
/// exactly [`elf::parse_elf`] and [`dispatch::run_dispatch_loop`] -- the M1
/// driver; a Linux guest composes [`linux::run_linux_initramfs`]. Neither is
/// reimplemented.
fn boot_and_run(guest: &LoadedGuest, spec: &VmSpec, input: &[u8]) -> Result<Bytes, ProximaError> {
    match (guest, &spec.guest) {
        (LoadedGuest::Elf { image }, _) => boot_elf(image, spec, input),
        (
            LoadedGuest::Linux { kernel, initramfs },
            GuestSpec::Linux {
                cmdline,
                ram_mib,
                watchdog_millis,
                ..
            },
        ) => {
            let limits = LinuxGuestLimits {
                ram_mib: *ram_mib,
                output_capacity: spec.output_capacity,
                watchdog_millis: *watchdog_millis,
                ..LinuxGuestLimits::default()
            };
            let run = linux::run_linux_initramfs(kernel, initramfs, cmdline, input, &limits)?;
            Ok(Bytes::from(run.output))
        }
        (LoadedGuest::Linux { .. }, GuestSpec::Elf { .. }) => Err(ProximaError::Config(
            "linux guest loaded against an elf spec".into(),
        )),
    }
}

fn boot_elf(image: &[u8], spec: &VmSpec, input: &[u8]) -> Result<Bytes, ProximaError> {
    let (entry, segments) = elf::parse_elf::<MAX_SEGMENTS>(image)
        .map_err(|error| ProximaError::Upstream(format!("parse guest ELF: {error}")))?;
    let observed = dispatch::run_dispatch_loop(
        entry,
        &segments,
        input,
        spec.output_capacity,
        spec.max_hypercalls,
        EMITTED_CAPACITY,
        MMIO_EMITTED_CAPACITY,
        NET_EMITTED_CAPACITY,
        BLK_EMITTED_CAPACITY,
        PL011_EMITTED_CAPACITY,
        spec.guest_memory_size,
    );
    report_teardown_failures();
    Ok(Bytes::from(observed?.output))
}

fn report_teardown_failures() {
    for failure in dispatch::take_teardown_failures() {
        warn!(%failure, "vm teardown failed to release a resource");
    }
}

impl SendPipe for VmUpstream {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
        let label = self.label.clone();
        let spec = self.spec.clone();
        let guest = self.guest.clone();
        async move {
            let (_, input) = request.body_bytes().await?;
            boot_and_run(&guest, &spec, &input)
                .map(Response::ok)
                .map_err(|error| ProximaError::Upstream(format!("vm `{label}`: {error}")))
        }
    }
}

fn offloaded(handle: PipeHandle, runtime: Option<Arc<dyn Runtime>>) -> PipeHandle {
    match runtime {
        Some(runtime) => into_handle(Offload::new(handle, runtime)),
        None => handle,
    }
}

pub struct VmPipeFactory;

impl PipeFactory for VmPipeFactory {
    fn name(&self) -> &str {
        "vm"
    }

    fn path_fields(&self) -> &[&'static str] {
        &[
            "guest_image_path",
            "kernel_path",
            "init_path",
            "oci_init_path",
            "image_cache_dir",
            "modules_root",
        ]
    }

    fn build(
        &self,
        spec: &Value,
        _inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
        let spec = spec.clone();
        Box::pin(async move {
            let config: VmConfig = serde_json::from_value(spec)
                .map_err(|err| ProximaError::Config(format!("vm config: {err}")))?;
            let label = config.name.clone();
            let parsed = config.into_spec()?;
            if parsed.resident {
                return build_resident(&label, &parsed).await;
            }
            let upstream = into_handle(VmUpstream::new(label, parsed)?);
            let runtime = crate::runtime::installed_runtime().map(|selection| selection.runtime);
            Ok(offloaded(upstream, runtime))
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::any::Any;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use proxima_runtime::{BackgroundHandle, CoreId, SpawnError};
    use serde_json::json;

    // principle-4 parity: the fluent builder and the config value must lower
    // to identical VmSpec state.
    #[test]
    fn parity_fluent_builder_and_config_value_match() {
        let from_value: VmConfig = serde_json::from_value(json!({
            "name": "my-fn",
            "guest_image_path": "/tmp/guest.elf",
            "max_hypercalls": 8,
            "output_capacity": 128,
            "guest_memory_size": 1048576,
        }))
        .expect("from_value");
        let from_value = from_value.into_spec().expect("into_spec value");

        let from_builder = VmConfig::builder()
            .name("my-fn")
            .guest_image_path("/tmp/guest.elf")
            .max_hypercalls(8)
            .output_capacity(128)
            .guest_memory_size(1_048_576)
            .build()
            .into_spec()
            .expect("into_spec builder");

        assert_eq!(from_value.guest, from_builder.guest);
        assert_eq!(from_value.max_hypercalls, from_builder.max_hypercalls);
        assert_eq!(from_value.output_capacity, from_builder.output_capacity);
        assert_eq!(from_value.guest_memory_size, from_builder.guest_memory_size);
    }

    // config -> fluent round-trip: a config loaded from TOML/JSON reseeds the
    // builder field-by-field and keeps chaining, same shape `settings/mod.rs`
    // documents ("existing.builder().field(new).build()").
    #[test]
    fn loaded_config_reseeds_the_builder_and_keeps_chaining() {
        let loaded: VmConfig = serde_json::from_value(json!({
            "guest_image_path": "/tmp/a.elf",
            "max_hypercalls": 4,
        }))
        .expect("from_value");
        let extended = VmConfig::builder()
            .maybe_guest_image_path(loaded.guest_image_path.clone())
            .max_hypercalls(32)
            .build();
        assert_eq!(extended.guest_image_path, loaded.guest_image_path);
        assert_eq!(extended.max_hypercalls, 32);
    }

    #[test]
    fn defaults_describe_a_elf_lambda() {
        let config = VmConfig::builder()
            .guest_image_path("/tmp/guest.elf")
            .build();
        assert_eq!(config.name, "vm");
        assert_eq!(config.max_hypercalls, 16);
        assert_eq!(config.output_capacity, 65_536);
        assert_eq!(config.guest_memory_size, 2 * 1024 * 1024);
    }

    #[test]
    fn empty_guest_image_path_returns_config_error() {
        let outcome = VmConfig::builder().guest_image_path("").build().into_spec();
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[test]
    fn zero_max_hypercalls_returns_config_error() {
        let outcome = VmConfig::builder()
            .guest_image_path("/tmp/guest.elf")
            .max_hypercalls(0)
            .build()
            .into_spec();
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[test]
    fn zero_output_capacity_returns_config_error() {
        let outcome = VmConfig::builder()
            .guest_image_path("/tmp/guest.elf")
            .output_capacity(0)
            .build()
            .into_spec();
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[test]
    fn guest_memory_size_must_be_a_nonzero_page_multiple() {
        for size in [0_u64, 1, 4095, 4097, 2 * 1024 * 1024 + 1] {
            let outcome = VmConfig::builder()
                .guest_image_path("/tmp/guest.elf")
                .guest_memory_size(size)
                .build()
                .into_spec();
            let Err(ProximaError::Config(message)) = outcome else {
                panic!("guest_memory_size {size} must be rejected, got {outcome:?}");
            };
            assert!(
                message.contains("guest_memory_size") && message.contains("multiple of 4096"),
                "the rejection must name the field and the rule: {message}"
            );
        }
        for size in [4096_u64, 2 * 1024 * 1024, 64 * 1024 * 1024] {
            VmConfig::builder()
                .guest_image_path("/tmp/guest.elf")
                .guest_memory_size(size)
                .build()
                .into_spec()
                .expect("a page multiple is a valid guest_memory_size");
        }
    }

    #[proxima::test]
    async fn missing_guest_image_path_returns_config_error() {
        let factory = VmPipeFactory;
        let outcome = factory.build(&json!({"name": "no_path"}), None).await;
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    // M2 exit criterion 2: the same `VmConfig` built three ways -- via the
    // conflaguration env loader (`PROXIMA_VM_*`), via `ProximaSettings::
    // from_path` (the real file-loading path `[upstreams.my-fn] type =
    // "vm"` config resolves through), and via `.builder().build()` --
    // asserted to identical internal state (`tools/proxima-vm/ROADMAP.md`'s
    // M2 section, exit criterion 2; principle 4 config-as-mirror).
    #[test]
    fn env_loader_from_path_and_builder_yield_identical_state() {
        let directory = tempfile::tempdir().expect("create tempdir");
        let guest_path = directory.path().join("guest.elf");
        std::fs::write(&guest_path, b"not-a-real-elf-this-test-never-parses-it")
            .expect("write guest stub");
        let guest_image_path = guest_path.to_string_lossy().into_owned();

        let from_env: VmConfig = temp_env::with_vars(
            [
                (
                    "PROXIMA_VM_GUEST_IMAGE_PATH",
                    Some(guest_image_path.as_str()),
                ),
                ("PROXIMA_VM_NAME", Some("my-fn")),
                ("PROXIMA_VM_MAX_HYPERCALLS", Some("8")),
                ("PROXIMA_VM_OUTPUT_CAPACITY", Some("128")),
                ("PROXIMA_VM_GUEST_MEMORY_SIZE", Some("1048576")),
            ],
            || VmConfig::from_env().expect("VmConfig::from_env"),
        );

        let config_path = directory.path().join("proxima.toml");
        let toml = format!(
            "[upstreams.my-fn]\ntype = \"vm\"\nname = \"my-fn\"\nguest_image_path = \"{guest_image_path}\"\nmax_hypercalls = 8\noutput_capacity = 128\nguest_memory_size = 1048576\n"
        );
        std::fs::write(&config_path, toml).expect("write proxima.toml");
        let settings =
            crate::settings::ProximaSettings::from_path(&config_path).expect("load settings");
        let entry = settings
            .upstreams
            .get("my-fn")
            .expect("my-fn registered by config")
            .clone();
        let from_path: VmConfig =
            serde_json::from_value(entry.spec).expect("decode VmConfig from registry entry spec");

        let from_builder = VmConfig::builder()
            .name("my-fn")
            .guest_image_path(guest_image_path.clone())
            .max_hypercalls(8)
            .output_capacity(128)
            .guest_memory_size(1_048_576)
            .build();

        assert_eq!(from_env, from_path, "env loader and from_path diverged");
        assert_eq!(from_path, from_builder, "from_path and builder diverged");
    }

    #[test]
    fn linux_env_loader_from_path_and_builder_yield_identical_state() {
        let directory = tempfile::tempdir().expect("create tempdir");
        let kernel_path = "/boot/vmlinuz-linux";
        let init_path = directory.path().join("uppercase-init");
        let init_path = init_path.to_string_lossy().into_owned();
        let cmdline = "console=ttyS0 reboot=k panic=-1";

        let from_env: VmConfig = temp_env::with_vars(
            [
                ("PROXIMA_VM_KERNEL_PATH", Some(kernel_path)),
                ("PROXIMA_VM_INIT_PATH", Some(init_path.as_str())),
                ("PROXIMA_VM_NAME", Some("shout-linux")),
                ("PROXIMA_VM_CMDLINE", Some(cmdline)),
                ("PROXIMA_VM_RAM_MIB", Some("64")),
                ("PROXIMA_VM_WATCHDOG_MILLIS", Some("5000")),
                ("PROXIMA_VM_OUTPUT_CAPACITY", Some("4096")),
            ],
            || VmConfig::from_env().expect("VmConfig::from_env"),
        );

        let config_path = directory.path().join("proxima.toml");
        let toml = format!(
            "[upstreams.shout-linux]\ntype = \"vm\"\nname = \"shout-linux\"\nkernel_path = \"{kernel_path}\"\ninit_path = \"{init_path}\"\ncmdline = \"{cmdline}\"\nram_mib = 64\nwatchdog_millis = 5000\noutput_capacity = 4096\n"
        );
        std::fs::write(&config_path, toml).expect("write proxima.toml");
        let settings =
            crate::settings::ProximaSettings::from_path(&config_path).expect("load settings");
        let entry = settings
            .upstreams
            .get("shout-linux")
            .expect("shout-linux registered by config")
            .clone();
        let from_path: VmConfig =
            serde_json::from_value(entry.spec).expect("decode VmConfig from registry entry spec");

        let from_builder = VmConfig::builder()
            .name("shout-linux")
            .kernel_path(kernel_path)
            .init_path(init_path.clone())
            .cmdline(cmdline)
            .ram_mib(64)
            .watchdog_millis(5000)
            .output_capacity(4096)
            .build();

        assert_eq!(from_env, from_path, "env loader and from_path diverged");
        assert_eq!(from_path, from_builder, "from_path and builder diverged");
        assert!(matches!(
            from_builder.into_spec().expect("linux spec").guest,
            GuestSpec::Linux { ram_mib: 64, .. }
        ));
    }

    #[test]
    fn linux_defaults_carry_the_measured_cmdline_through_every_surface() {
        let from_env: VmConfig = temp_env::with_vars(
            [
                ("PROXIMA_VM_KERNEL_PATH", Some("/boot/vmlinuz-linux")),
                ("PROXIMA_VM_INIT_PATH", Some("/opt/init")),
            ],
            || VmConfig::from_env().expect("VmConfig::from_env"),
        );
        let from_builder = VmConfig::builder()
            .kernel_path("/boot/vmlinuz-linux")
            .init_path("/opt/init")
            .build();
        let from_value: VmConfig = serde_json::from_value(json!({
            "kernel_path": "/boot/vmlinuz-linux",
            "init_path": "/opt/init",
        }))
        .expect("from_value");

        assert_eq!(from_env.cmdline, DEFAULT_LINUX_CMDLINE);
        assert_eq!(from_builder.cmdline, DEFAULT_LINUX_CMDLINE);
        assert_eq!(from_value.cmdline, DEFAULT_LINUX_CMDLINE);
        assert_eq!(from_env, from_builder);
        assert_eq!(from_value, from_builder);
    }

    #[test]
    fn resident_env_loader_value_and_builder_yield_identical_state() {
        let from_env: VmConfig = temp_env::with_vars(
            [
                ("PROXIMA_VM_KERNEL_PATH", Some("/boot/vmlinuz-linux")),
                ("PROXIMA_VM_IMAGE", Some(WHOAMI_IMAGE)),
                ("PROXIMA_VM_OCI_INIT_PATH", Some("/opt/oci-init")),
                ("PROXIMA_VM_RESIDENT", Some("true")),
                ("PROXIMA_VM_GUEST_PORT", Some("8080")),
                ("PROXIMA_VM_CONNECT_TIMEOUT_MILLIS", Some("5000")),
                ("PROXIMA_VM_MODULES_ROOT", Some("/opt/modules")),
            ],
            || VmConfig::from_env().expect("VmConfig::from_env"),
        );
        let from_value: VmConfig = serde_json::from_value(json!({
            "kernel_path": "/boot/vmlinuz-linux",
            "image": WHOAMI_IMAGE,
            "oci_init_path": "/opt/oci-init",
            "resident": true,
            "guest_port": 8080,
            "connect_timeout_millis": 5000,
            "modules_root": "/opt/modules",
        }))
        .expect("from_value");
        let from_builder = VmConfig::builder()
            .kernel_path("/boot/vmlinuz-linux")
            .image(WHOAMI_IMAGE)
            .oci_init_path("/opt/oci-init")
            .resident(true)
            .guest_port(8080)
            .connect_timeout_millis(5000)
            .modules_root("/opt/modules")
            .build();

        assert_eq!(from_env, from_value, "env loader and value diverged");
        assert_eq!(from_value, from_builder, "value and builder diverged");
        let spec = from_builder.into_spec().expect("resident spec");
        assert!(spec.resident);
        assert_eq!(spec.guest_port, 8080);
        assert_eq!(spec.connect_timeout_millis, 5000);
        assert_eq!(spec.modules_root, PathBuf::from("/opt/modules"));
    }

    #[test]
    fn resident_defaults_are_per_call_whoami_friendly() {
        let config = VmConfig::builder().kernel_path("/k").init_path("/i").build();

        assert!(!config.resident);
        assert_eq!(config.guest_port, 80);
        assert_eq!(config.connect_timeout_millis, 60_000);
        assert_eq!(config.modules_root, "/usr/lib/modules");
    }

    #[proxima::test]
    #[case::resident_elf(
        VmConfig::builder().guest_image_path("/g.elf").resident(true).build(),
        "resident: only a Linux guest (kernel_path) can run resident"
    )]
    #[case::zero_guest_port(
        VmConfig::builder().kernel_path("/k").init_path("/i").resident(true).guest_port(0).build(),
        "guest_port: must be greater than zero"
    )]
    #[case::zero_connect_timeout(
        VmConfig::builder().kernel_path("/k").init_path("/i").resident(true).connect_timeout_millis(0).build(),
        "connect_timeout_millis: must be greater than zero"
    )]
    async fn invalid_resident_configs_are_rejected_with_the_field_named(
        #[case] config: VmConfig,
        #[case] expected: &str,
    ) {
        let outcome = config.into_spec();

        assert!(
            matches!(&outcome, Err(ProximaError::Config(message)) if message.contains(expected)),
            "expected a config error containing {expected:?}, got {outcome:?}"
        );
    }

    #[test]
    fn a_config_naming_both_guests_is_rejected() {
        let outcome = VmConfig::builder()
            .guest_image_path("/tmp/guest.elf")
            .kernel_path("/boot/vmlinuz-linux")
            .init_path("/opt/init")
            .build()
            .into_spec();

        let Err(ProximaError::Config(message)) = outcome else {
            panic!("naming both guests must be rejected, got {outcome:?}");
        };
        assert!(message.contains("set exactly one"), "{message}");
    }

    #[test]
    fn a_config_naming_neither_guest_is_rejected() {
        let outcome = VmConfig::builder().name("nothing").build().into_spec();

        let Err(ProximaError::Config(message)) = outcome else {
            panic!("naming no guest must be rejected, got {outcome:?}");
        };
        assert!(message.contains("neither guest_image_path"), "{message}");
    }

    #[test]
    fn a_kernel_without_an_init_and_an_init_without_a_kernel_are_rejected() {
        let kernel_only = VmConfig::builder()
            .kernel_path("/boot/vmlinuz-linux")
            .build()
            .into_spec();
        let elf_with_init = VmConfig::builder()
            .guest_image_path("/tmp/guest.elf")
            .init_path("/opt/init")
            .build()
            .into_spec();

        assert!(matches!(kernel_only, Err(ProximaError::Config(message)) if message.contains("required with kernel_path")));
        assert!(matches!(elf_with_init, Err(ProximaError::Config(message)) if message.contains("only a Linux guest")));
    }

    const WHOAMI_IMAGE: &str =
        "traefik/whoami@sha256:1474027c316661cdec87df2623e13a41e7e1ce0ba99c24917631de8f300b5420";

    #[test]
    fn image_env_loader_value_and_builder_yield_identical_state() {
        let from_env: VmConfig = temp_env::with_vars(
            [
                ("PROXIMA_VM_KERNEL_PATH", Some("/boot/vmlinuz-linux")),
                ("PROXIMA_VM_IMAGE", Some(WHOAMI_IMAGE)),
                ("PROXIMA_VM_OCI_INIT_PATH", Some("/opt/oci-init")),
                ("PROXIMA_VM_IMAGE_CACHE_DIR", Some("/var/cache/oci")),
            ],
            || VmConfig::from_env().expect("VmConfig::from_env"),
        );
        let from_value: VmConfig = serde_json::from_value(json!({
            "kernel_path": "/boot/vmlinuz-linux",
            "image": WHOAMI_IMAGE,
            "oci_init_path": "/opt/oci-init",
            "image_cache_dir": "/var/cache/oci",
        }))
        .expect("from_value");
        let from_builder = VmConfig::builder()
            .kernel_path("/boot/vmlinuz-linux")
            .image(WHOAMI_IMAGE)
            .oci_init_path("/opt/oci-init")
            .image_cache_dir("/var/cache/oci")
            .build();

        assert_eq!(from_env, from_value, "env loader and value diverged");
        assert_eq!(from_value, from_builder, "value and builder diverged");
        let GuestSpec::Linux { root, .. } = from_builder.into_spec().expect("image spec").guest
        else {
            panic!("an image config lowers to the linux flavor");
        };
        assert_eq!(
            root,
            LinuxRoot::Image {
                reference: WHOAMI_IMAGE.to_string(),
                init_path: PathBuf::from("/opt/oci-init"),
                cache_dir: PathBuf::from("/var/cache/oci"),
            }
        );
    }

    #[test]
    fn an_image_without_a_cache_dir_uses_the_temp_dir_default() {
        let spec = VmConfig::builder()
            .kernel_path("/boot/vmlinuz-linux")
            .image(WHOAMI_IMAGE)
            .oci_init_path("/opt/oci-init")
            .build()
            .into_spec()
            .expect("image spec");

        let GuestSpec::Linux {
            root: LinuxRoot::Image { cache_dir, .. },
            ..
        } = spec.guest
        else {
            panic!("an image config lowers to the linux image root");
        };
        assert_eq!(cache_dir, std::env::temp_dir().join("proxima-oci-cache"));
    }

    #[proxima::test]
    #[case::image_and_init_path_together(
        VmConfig::builder().kernel_path("/k").init_path("/i").image(WHOAMI_IMAGE).oci_init_path("/o").build(),
        "set exactly one"
    )]
    #[case::image_without_oci_init(
        VmConfig::builder().kernel_path("/k").image(WHOAMI_IMAGE).build(),
        "oci_init_path: required with image"
    )]
    #[case::oci_init_without_image(
        VmConfig::builder().kernel_path("/k").init_path("/i").oci_init_path("/o").build(),
        "oci_init_path: only valid with image"
    )]
    #[case::cache_dir_without_image(
        VmConfig::builder().kernel_path("/k").init_path("/i").image_cache_dir("/c").build(),
        "image_cache_dir: only valid with image"
    )]
    #[case::image_on_an_elf_guest(
        VmConfig::builder().guest_image_path("/g.elf").image(WHOAMI_IMAGE).oci_init_path("/o").build(),
        "only a Linux guest (kernel_path) takes an image"
    )]
    #[case::malformed_reference(
        VmConfig::builder().kernel_path("/k").image("traefik//whoami").oci_init_path("/o").build(),
        "empty component"
    )]
    #[case::empty_image(
        VmConfig::builder().kernel_path("/k").image("").oci_init_path("/o").build(),
        "must not be empty"
    )]
    async fn invalid_image_configs_are_rejected_with_the_field_named(
        #[case] config: VmConfig,
        #[case] expected: &str,
    ) {
        let outcome = config.into_spec();

        assert!(
            matches!(&outcome, Err(ProximaError::Config(message)) if message.contains(expected)),
            "expected a config error containing {expected:?}, got {outcome:?}"
        );
    }

    async fn load_vm_config(directory: &Path, toml: &str) -> ProximaError {
        let config_path = directory.join("lambda.toml");
        std::fs::write(&config_path, toml).expect("write lambda.toml");
        let context = crate::load::LoadContext::with_default_registry().expect("load context");
        match crate::load::load(config_path.as_path(), &context).await {
            Ok(_) => panic!("the guest named by {toml:?} must not boot"),
            Err(error) => error,
        }
    }

    #[proxima::test]
    async fn a_relative_guest_path_missing_from_the_config_directory_names_the_resolved_path() {
        let directory = tempfile::tempdir().expect("create tempdir");

        let error = load_vm_config(
            directory.path(),
            "type = \"vm\"\nname = \"ghost\"\nguest_image_path = \"absent.elf\"\n",
        )
        .await;

        let resolved = directory.path().join("absent.elf");
        let ProximaError::Upstream(message) = error else {
            panic!("an unreadable guest must fail the build with an upstream error: {error:?}");
        };
        assert!(
            message.contains(&format!("read guest image `{}`", resolved.display())),
            "the failure must name {}: {message}",
            resolved.display()
        );
    }

    #[proxima::test]
    async fn a_relative_guest_path_is_found_beside_the_config_file() {
        let directory = tempfile::tempdir().expect("create tempdir");
        std::fs::write(directory.path().join("stub.elf"), b"#!/bin/sh\n").expect("write stub");

        let error = load_vm_config(
            directory.path(),
            "type = \"vm\"\nname = \"stub\"\nguest_image_path = \"stub.elf\"\n",
        )
        .await;

        let resolved = directory.path().join("stub.elf");
        let ProximaError::Upstream(message) = error else {
            panic!("a non-ELF guest must fail the build with an upstream error: {error:?}");
        };
        assert!(
            message.contains(&format!("parse guest ELF `{}`", resolved.display())),
            "the read must succeed at {} and fail at the parse: {message}",
            resolved.display()
        );
    }

    #[proxima::test]
    async fn an_absolute_kernel_path_is_left_untouched_by_anchoring() {
        let directory = tempfile::tempdir().expect("create tempdir");

        let error = load_vm_config(
            directory.path(),
            "type = \"vm\"\nname = \"split\"\nkernel_path = \"/no/such/vmlinuz\"\ninit_path = \"init\"\n",
        )
        .await;

        let ProximaError::Upstream(message) = error else {
            panic!("an unreadable kernel must fail the build with an upstream error: {error:?}");
        };
        assert!(
            message.contains("read kernel `/no/such/vmlinuz`"),
            "an absolute kernel_path must pass through unchanged: {message}"
        );
    }

    #[proxima::test]
    async fn a_missing_kernel_fails_at_build_time_naming_the_path() {
        let directory = tempfile::tempdir().expect("create tempdir");
        let init_path = directory.path().join("init");
        std::fs::write(&init_path, b"init").expect("write init stub");

        let outcome = VmPipeFactory
            .build(
                &json!({
                    "name": "ghost-linux",
                    "kernel_path": "/no/such/vmlinuz",
                    "init_path": init_path.to_string_lossy(),
                }),
                None,
            )
            .await;

        let Err(ProximaError::Upstream(message)) = outcome else {
            panic!("an unreadable kernel must fail the build with an upstream error");
        };
        assert!(
            message.contains("vm `ghost-linux`") && message.contains("read kernel `/no/such/vmlinuz`"),
            "the failure must name the pipe and the path: {message}"
        );
    }

    #[proxima::test]
    async fn a_kernel_without_a_bzimage_header_fails_at_build_time() {
        let directory = tempfile::tempdir().expect("create tempdir");
        let kernel_path = directory.path().join("not-a-kernel");
        let init_path = directory.path().join("init");
        std::fs::write(&kernel_path, b"#!/bin/sh\necho not a kernel\n").expect("write kernel stub");
        std::fs::write(&init_path, b"init").expect("write init stub");

        let outcome = VmPipeFactory
            .build(
                &json!({
                    "name": "script-linux",
                    "kernel_path": kernel_path.to_string_lossy(),
                    "init_path": init_path.to_string_lossy(),
                }),
                None,
            )
            .await;

        let Err(ProximaError::Upstream(message)) = outcome else {
            panic!("a non-bzImage kernel must fail the build with an upstream error");
        };
        assert!(
            message.contains("vm `script-linux`") && message.contains("parse kernel"),
            "the failure must name the pipe and the parse stage: {message}"
        );
    }

    #[proxima::test]
    async fn nonexistent_guest_image_fails_at_build_time() {
        let factory = VmPipeFactory;
        let outcome = factory
            .build(
                &json!({
                    "name": "ghost",
                    "guest_image_path": "/no/such/guest.elf",
                }),
                None,
            )
            .await;
        let Err(ProximaError::Upstream(message)) = outcome else {
            panic!("an unreadable image must fail the build with an upstream error");
        };
        assert!(
            message.contains("vm `ghost`")
                && message.contains("read guest image `/no/such/guest.elf`"),
            "the failure must name the pipe and the path: {message}"
        );
    }

    #[proxima::test]
    async fn non_elf_guest_image_fails_at_build_time() {
        let directory = tempfile::tempdir().expect("create tempdir");
        let guest_path = directory.path().join("guest.elf");
        std::fs::write(&guest_path, b"#!/bin/sh\necho not an elf\n").expect("write guest stub");

        let factory = VmPipeFactory;
        let outcome = factory
            .build(
                &json!({
                    "name": "script",
                    "guest_image_path": guest_path.to_string_lossy(),
                }),
                None,
            )
            .await;

        let Err(ProximaError::Upstream(message)) = outcome else {
            panic!("a non-ELF image must fail the build with an upstream error");
        };
        assert!(
            message.contains("vm `script`") && message.contains("parse guest ELF"),
            "the failure must name the pipe and the parse stage: {message}"
        );
    }

    #[proxima::test]
    async fn empty_guest_image_fails_at_build_time() {
        let directory = tempfile::tempdir().expect("create tempdir");
        let guest_path = directory.path().join("empty.elf");
        std::fs::write(&guest_path, b"").expect("write empty guest");

        let outcome = VmUpstream::new(
            "empty",
            VmSpec {
                guest: GuestSpec::Elf {
                    image_path: guest_path,
                },
                max_hypercalls: 16,
                output_capacity: 64,
                guest_memory_size: 2 * 1024 * 1024,
                resident: false,
                guest_port: 80,
                connect_timeout_millis: 60_000,
                modules_root: PathBuf::from("/usr/lib/modules"),
                restart: RestartPolicy::Never,
                restart_delay_ms: 1_000,
                max_restart_attempts: 0,
            },
        );

        assert!(matches!(outcome, Err(ProximaError::Upstream(_))));
    }

    struct CountingRuntime {
        submissions: Arc<AtomicUsize>,
    }

    impl Runtime for CountingRuntime {
        fn spawn_on_current_core(&self, _future: Pin<Box<dyn Future<Output = ()> + 'static>>) {
            unreachable!("offload never calls spawn_on_current_core")
        }

        fn spawn_on_core(
            &self,
            _core_id: CoreId,
            _future: Pin<Box<dyn Future<Output = ()> + Send + 'static>>,
        ) -> Result<(), SpawnError> {
            unreachable!("offload never calls spawn_on_core")
        }

        fn spawn_factory_on_core(
            &self,
            _core_id: CoreId,
            _factory: Box<
                dyn FnOnce() -> Pin<Box<dyn Future<Output = ()> + 'static>> + Send + 'static,
            >,
        ) -> Result<(), SpawnError> {
            unreachable!("offload never calls spawn_factory_on_core")
        }

        fn spawn_background_blocking(
            &self,
            _work: Box<dyn FnOnce() -> Result<Box<dyn Any + Send>, ProximaError> + Send>,
        ) -> BackgroundHandle<Box<dyn Any + Send>> {
            self.submissions.fetch_add(1, Ordering::SeqCst);
            Box::pin(async {
                Err(ProximaError::Upstream(
                    "counting runtime declined the work".into(),
                ))
            })
        }

        fn timer_at(
            &self,
            _deadline: std::time::Instant,
        ) -> Pin<Box<dyn Future<Output = ()> + 'static>> {
            unreachable!("offload never calls timer_at")
        }

        fn current_core(&self) -> CoreId {
            CoreId(0)
        }

        fn num_cores(&self) -> usize {
            1
        }
    }

    struct EchoPipe;

    impl SendPipe for EchoPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
            async { Ok(Response::ok("ran inline")) }
        }
    }

    fn get_request() -> Request<Bytes> {
        Request::builder()
            .method("POST")
            .path("/")
            .build()
            .expect("request")
    }

    #[proxima::test]
    async fn an_installed_runtime_receives_the_call_on_its_background_pool() {
        let submissions = Arc::new(AtomicUsize::new(0));
        let runtime: Arc<dyn Runtime> = Arc::new(CountingRuntime {
            submissions: submissions.clone(),
        });

        let handle = offloaded(into_handle(EchoPipe), Some(runtime));
        let outcome = SendPipe::call(&handle, get_request()).await;

        assert_eq!(submissions.load(Ordering::SeqCst), 1);
        assert!(
            matches!(&outcome, Err(ProximaError::Upstream(message)) if message.contains("declined")),
            "the runtime's answer, not the inline pipe's, must come back: {outcome:?}"
        );
    }

    #[proxima::test]
    async fn without_a_runtime_the_call_runs_inline() {
        let handle = offloaded(into_handle(EchoPipe), None);

        let response = SendPipe::call(&handle, get_request())
            .await
            .expect("inline call");

        assert_eq!(&response.payload[..], b"ran inline");
    }
}
