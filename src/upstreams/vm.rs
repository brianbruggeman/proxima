//! `vm` upstream: a lambda as a registry entry, not a binary. `[upstream.my-fn]
//! type = "vm"` in a proxima config boots a guest ELF through
//! `proxima_vm::dispatch::run_dispatch_loop` — the M1 driver this module
//! composes, never rewrites (`tools/proxima-vm/ROADMAP.md`'s M2 section).
//!
//! Mirrors [`crate::upstreams::process::ProcessConfig`] /
//! [`crate::upstreams::process::ProcessPipeFactory`] exactly: one config
//! struct deriving `Builder + Deserialize + Serialize + Settings +
//! Validate`, one factory resolved through the same `PipeFactoryRegistry`
//! seam (`src/settings/mod.rs`'s `RegistryEntry`, `src/app.rs`'s
//! `ProximaSettings.upstreams`).
//!
//! `max_hypercalls` / `emitted_capacity` / the canned response are exactly
//! the parameters `run_dispatch_loop` accepts today — every field here
//! traces to a real parameter of the composed driver (principle 12). Guest
//! memory size, PT_LOAD segment capacity, and the four device-channel
//! (mmio/net/blk/pl011) capacities are NOT exposed as config fields:
//! `run_dispatch_loop`'s C ABI fixes `GUEST_MEMORY_SIZE`, `parse_elf`'s
//! `MAX_SEGMENTS` is a const generic, and the device channels are only
//! driven by the lambda guest's own unconditional virtio bring-up
//! (`guests/lambda/src/main.rs`) — none of these are runtime knobs this
//! factory could thread through without rewriting the M1 driver's signature,
//! out of scope for M2 per the roadmap's own "compose, not rewrite"
//! instruction. `MAX_SEGMENTS` and the device-channel capacities below match
//! the values `tools/proxima-vm/src/bin/proxima-vm.rs` already uses for the
//! same call.

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;

use bon::Builder;
use bytes::Bytes;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use proxima_primitives::pipe::SendPipe;
use proxima_protocols::process::{ChildResponse, ReadResponse};
use proxima_vm::dispatch;
use proxima_vm::elf;

use crate::error::ProximaError;
use crate::pipe::{PipeHandle, into_handle};
use crate::pipe_factory::PipeFactory;
use crate::request::{Request, Response};

/// PT_LOAD segment cap `elf::parse_elf`'s const generic is instantiated
/// with. Matches `tools/proxima-vm/src/bin/proxima-vm.rs`'s `MAX_SEGMENTS`
/// — the largest segment count any lambda guest built so far links
/// (`.text`, `.rodata`, `.data`), with headroom.
const MAX_SEGMENTS: usize = 4;

/// Device-channel capacities `run_dispatch_loop` requires — the lambda
/// guest unconditionally drives virtio-console/net/blk after its
/// `ChildRequest` calls (`guests/lambda/src/main.rs`), so every call this
/// upstream makes must size these the same as
/// `tools/proxima-vm/src/bin/proxima-vm.rs`'s own constants or the guest's
/// device bring-up overruns the host buffer.
const MMIO_EMITTED_CAPACITY: usize = 256;
const NET_EMITTED_CAPACITY: usize = 256;
const BLK_EMITTED_CAPACITY: usize = 2048;
const PL011_EMITTED_CAPACITY: usize = 256;

fn default_name() -> String {
    "vm".to_string()
}

fn default_max_hypercalls() -> usize {
    16
}

fn default_emitted_capacity() -> usize {
    256
}

fn default_response_eof() -> bool {
    true
}

/// Runtime spec a [`VmUpstream`] runs against — the lowered form of
/// [`VmConfig`], mirroring [`crate::upstreams::process::ProcessSpec`]'s
/// role for `ProcessConfig`.
#[derive(Debug, Clone)]
pub struct VmSpec {
    pub guest_image_path: PathBuf,
    pub max_hypercalls: usize,
    pub emitted_capacity: usize,
    pub response_bytes: Vec<u8>,
    pub response_eof: bool,
}

/// Typed config surface for the `vm` upstream — a guest ELF driven through
/// `proxima-vm`'s M1 dispatch loop. Mirrors [`crate::upstreams::process::
/// ProcessConfig`]'s shape: one struct deriving `Builder + Deserialize +
/// Serialize + Settings`, `#[settings(prefix = "PROXIMA_VM")]`.
#[derive(Debug, Clone, PartialEq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA_VM")]
#[builder(derive(Clone, Debug), on(String, into))]
pub struct VmConfig {
    /// Path to the guest ELF image to boot.
    pub guest_image_path: String,

    /// Pipe label.
    #[setting(default = "vm")]
    #[serde(default = "default_name")]
    #[builder(default = default_name())]
    pub name: String,

    /// Hypercall-exit budget: the run loop reports a runaway guest instead
    /// of hanging the host once this many hypercalls have been driven.
    #[setting(default = 16)]
    #[serde(default = "default_max_hypercalls")]
    #[builder(default = default_max_hypercalls())]
    pub max_hypercalls: usize,

    /// Capacity, in bytes, of the buffer the guest's emitted bytes are
    /// collected into.
    #[setting(default = 256)]
    #[serde(default = "default_emitted_capacity")]
    #[builder(default = default_emitted_capacity())]
    pub emitted_capacity: usize,

    /// The single canned reply the host's dispatcher answers every guest
    /// hypercall with — `run_dispatch_loop`'s current one-`ChildResponse`
    /// shape (M2b replaces this with real host-side routing over the same
    /// channel; not this milestone's scope).
    #[setting(default)]
    #[serde(default)]
    #[builder(default)]
    pub response: String,

    /// Whether the canned reply reports end-of-stream. Defaults to true.
    #[setting(default = true)]
    #[serde(default = "default_response_eof")]
    #[builder(default = default_response_eof())]
    pub response_eof: bool,
}

impl Validate for VmConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.guest_image_path.is_empty() {
            errors.push(ValidationMessage::new(
                "guest_image_path",
                "must not be empty",
            ));
        }
        if self.max_hypercalls == 0 {
            errors.push(ValidationMessage::new(
                "max_hypercalls",
                "must be greater than zero",
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
        Ok(VmSpec {
            guest_image_path: PathBuf::from(self.guest_image_path),
            max_hypercalls: self.max_hypercalls,
            emitted_capacity: self.emitted_capacity,
            response_bytes: self.response.into_bytes(),
            response_eof: self.response_eof,
        })
    }
}

/// A guest ELF, driven through `proxima-vm`'s M1 dispatch loop on every
/// call. Fresh VM per request, no snapshot, no fork — the M2 exit
/// criterion's own baseline for later milestones to beat.
pub struct VmUpstream {
    label: String,
    spec: VmSpec,
}

impl VmUpstream {
    #[must_use]
    pub fn new(label: impl Into<String>, spec: VmSpec) -> Self {
        Self {
            label: label.into(),
            spec,
        }
    }

    #[must_use]
    pub fn spec(&self) -> &VmSpec {
        &self.spec
    }
}

/// Boots `guest_image_path`, drives it to completion, and returns the bytes
/// it emitted over the `ChildRequest` hvc channel. Composes exactly
/// [`elf::parse_elf`] and [`dispatch::run_dispatch_loop`] — the M1 driver —
/// never reimplements either. The mmio/net/blk/pl011 device channels
/// `run_dispatch_loop` now also returns (the lambda guest's unconditional
/// virtio bring-up, added after M1) are discarded here: this upstream's
/// contract is the `ChildRequest` channel's emitted bytes only.
fn boot_and_run(image: &[u8], spec: &VmSpec) -> Result<Bytes, ProximaError> {
    let (entry, segments) = elf::parse_elf::<MAX_SEGMENTS>(image)
        .map_err(|error| ProximaError::Upstream(format!("parse guest ELF: {error}")))?;
    let configured = ChildResponse::Read(ReadResponse {
        bytes: spec.response_bytes.clone(),
        eof: spec.response_eof,
    });
    let (_requests, emitted, _mmio_emitted, _net_emitted, _blk_emitted, _pl011_emitted, _, _, _) =
        dispatch::run_dispatch_loop(
            entry,
            &segments,
            configured,
            spec.max_hypercalls,
            spec.emitted_capacity,
            MMIO_EMITTED_CAPACITY,
            NET_EMITTED_CAPACITY,
            BLK_EMITTED_CAPACITY,
            PL011_EMITTED_CAPACITY,
            dispatch::GUEST_MEMORY_SIZE,
        )?;
    Ok(Bytes::from(emitted))
}

impl SendPipe for VmUpstream {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
        let label = self.label.clone();
        let spec = self.spec.clone();
        async move {
            let image = std::fs::read(&spec.guest_image_path).map_err(|error| {
                ProximaError::Upstream(format!(
                    "vm `{label}`: read guest image `{}`: {error}",
                    spec.guest_image_path.display()
                ))
            })?;
            boot_and_run(&image, &spec)
                .map(Response::ok)
                .map_err(|error| ProximaError::Upstream(format!("vm `{label}`: {error}")))
        }
    }
}

pub struct VmPipeFactory;

impl PipeFactory for VmPipeFactory {
    fn name(&self) -> &str {
        "vm"
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
            Ok(into_handle(VmUpstream::new(label, parsed)))
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    // principle-4 parity: the fluent builder and the config value must lower
    // to identical VmSpec state (guest_image_path/max_hypercalls/
    // emitted_capacity/response/response_eof).
    #[test]
    fn parity_fluent_builder_and_config_value_match() {
        let from_value: VmConfig = serde_json::from_value(json!({
            "name": "my-fn",
            "guest_image_path": "/tmp/guest.elf",
            "max_hypercalls": 8,
            "emitted_capacity": 128,
            "response": "canned",
            "response_eof": false,
        }))
        .expect("from_value");
        let from_value = from_value.into_spec().expect("into_spec value");

        let from_builder = VmConfig::builder()
            .name("my-fn")
            .guest_image_path("/tmp/guest.elf")
            .max_hypercalls(8)
            .emitted_capacity(128)
            .response("canned")
            .response_eof(false)
            .build()
            .into_spec()
            .expect("into_spec builder");

        assert_eq!(from_value.guest_image_path, from_builder.guest_image_path);
        assert_eq!(from_value.max_hypercalls, from_builder.max_hypercalls);
        assert_eq!(from_value.emitted_capacity, from_builder.emitted_capacity);
        assert_eq!(from_value.response_bytes, from_builder.response_bytes);
        assert_eq!(from_value.response_eof, from_builder.response_eof);
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
            .guest_image_path(loaded.guest_image_path.clone())
            .max_hypercalls(32)
            .build();
        assert_eq!(extended.guest_image_path, loaded.guest_image_path);
        assert_eq!(extended.max_hypercalls, 32);
    }

    #[test]
    fn defaults_match_the_composed_m1_driver_example() {
        let config = VmConfig::builder()
            .guest_image_path("/tmp/guest.elf")
            .build();
        assert_eq!(config.name, "vm");
        assert_eq!(config.max_hypercalls, 16);
        assert_eq!(config.emitted_capacity, 256);
        assert!(config.response_eof);
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

    #[proxima::test]
    async fn missing_guest_image_path_returns_config_error() {
        let factory = VmPipeFactory;
        let outcome = factory.build(&json!({"name": "no_path"}), None).await;
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    // M2 exit criterion 2: the same `VmConfig` built three ways — via the
    // conflaguration env loader (`PROXIMA_VM_*`), via `ProximaSettings::
    // from_path` (the real file-loading path `[upstreams.my-fn] type =
    // "vm"` config resolves through), and via `.builder().build()` —
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
                ("PROXIMA_VM_GUEST_IMAGE_PATH", Some(guest_image_path.as_str())),
                ("PROXIMA_VM_NAME", Some("my-fn")),
                ("PROXIMA_VM_MAX_HYPERCALLS", Some("8")),
                ("PROXIMA_VM_EMITTED_CAPACITY", Some("128")),
                ("PROXIMA_VM_RESPONSE", Some("canned")),
                ("PROXIMA_VM_RESPONSE_EOF", Some("false")),
            ],
            || VmConfig::from_env().expect("VmConfig::from_env"),
        );

        let config_path = directory.path().join("proxima.toml");
        let toml = format!(
            "[upstreams.my-fn]\ntype = \"vm\"\nname = \"my-fn\"\nguest_image_path = \"{guest_image_path}\"\nmax_hypercalls = 8\nemitted_capacity = 128\nresponse = \"canned\"\nresponse_eof = false\n"
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
            .emitted_capacity(128)
            .response("canned")
            .response_eof(false)
            .build();

        assert_eq!(from_env, from_path, "env loader and from_path diverged");
        assert_eq!(from_path, from_builder, "from_path and builder diverged");
    }

    #[proxima::test]
    async fn nonexistent_guest_image_returns_upstream_error_at_call_time() {
        let factory = VmPipeFactory;
        let handle = factory
            .build(
                &json!({
                    "name": "ghost",
                    "guest_image_path": "/no/such/guest.elf",
                }),
                None,
            )
            .await
            .expect("build succeeds — the image is only read at call time");
        let outcome = SendPipe::call(
            &handle,
            Request::builder()
                .method("GET")
                .path("/")
                .build()
                .expect("request"),
        )
        .await;
        assert!(matches!(outcome, Err(ProximaError::Upstream(_))));
    }
}
