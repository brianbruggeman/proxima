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
//! initramfs archived around its `/init` (`kernel_path` + `init_path`).
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
use proxima_vm::x86_boot::SetupHeader;

use crate::error::ProximaError;
use crate::pipe::{PipeHandle, into_handle};
use crate::pipe_factory::PipeFactory;
use crate::request::{Request, Response};
use crate::runtime::Runtime;

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

fn default_ram_mib() -> u32 {
    LinuxGuestLimits::default().ram_mib
}

fn default_watchdog_millis() -> u64 {
    LinuxGuestLimits::default().watchdog_millis
}

/// Which guest a [`VmSpec`] boots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuestSpec {
    /// A bare no_std ELF driven over the fd-keyed `ChildRequest` channel.
    Elf { image_path: PathBuf },
    /// A real bzImage under KVM whose initramfs `/init` is `init_path`.
    Linux {
        kernel_path: PathBuf,
        init_path: PathBuf,
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
    /// powers off fails the call when it expires.
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
}

impl Validate for VmConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        for (field, value) in [
            ("guest_image_path", &self.guest_image_path),
            ("kernel_path", &self.kernel_path),
            ("init_path", &self.init_path),
        ] {
            if value.as_deref().is_some_and(str::is_empty) {
                errors.push(ValidationMessage::new(field, "must not be empty"));
            }
        }
        match (&self.guest_image_path, &self.kernel_path, &self.init_path) {
            (Some(_), Some(_), _) => errors.push(ValidationMessage::new(
                "guest_image_path",
                "names an elf guest and kernel_path names a Linux guest; set exactly one",
            )),
            (None, None, _) => errors.push(ValidationMessage::new(
                "guest_image_path",
                "neither guest_image_path (elf) nor kernel_path (linux) is set",
            )),
            (Some(_), None, Some(_)) => errors.push(ValidationMessage::new(
                "init_path",
                "only a Linux guest (kernel_path) takes an init_path",
            )),
            (None, Some(_), None) => errors.push(ValidationMessage::new(
                "init_path",
                "required with kernel_path",
            )),
            _ => {}
        }
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
    /// Lower the wire config to the runtime [`VmSpec`].
    pub fn into_spec(self) -> Result<VmSpec, ProximaError> {
        self.validate()
            .map_err(|err| ProximaError::Config(format!("{err}")))?;
        let guest = match (self.guest_image_path, self.kernel_path, self.init_path) {
            (Some(image_path), None, None) => GuestSpec::Elf {
                image_path: PathBuf::from(image_path),
            },
            (None, Some(kernel_path), Some(init_path)) => GuestSpec::Linux {
                kernel_path: PathBuf::from(kernel_path),
                init_path: PathBuf::from(init_path),
                cmdline: self.cmdline,
                ram_mib: self.ram_mib,
                watchdog_millis: self.watchdog_millis,
            },
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

fn read_guest_file(label: &str, what: &str, path: &Path) -> Result<Vec<u8>, ProximaError> {
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

fn load_linux(
    label: &str,
    kernel_path: &Path,
    init_path: &Path,
) -> Result<LoadedGuest, ProximaError> {
    let kernel = read_guest_file(label, "kernel", kernel_path)?;
    SetupHeader::parse(&kernel).map_err(|error| {
        ProximaError::Upstream(format!(
            "vm `{label}`: parse kernel `{}`: {error}",
            kernel_path.display()
        ))
    })?;
    let init = read_guest_file(label, "init", init_path)?;
    let initramfs = linux::initramfs_from_init(&init).map_err(|error| {
        ProximaError::Upstream(format!(
            "vm `{label}`: archive init `{}`: {error}",
            init_path.display()
        ))
    })?;
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
                kernel_path,
                init_path,
                ..
            } => load_linux(&label, kernel_path, init_path)?,
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
        &["guest_image_path", "kernel_path", "init_path"]
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
