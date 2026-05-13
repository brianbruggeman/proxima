//! Typed Profile struct + cross-axis validation.
//!
//! Resolution order (conflaguration's standard layering):
//!   1. Type-level defaults from `#[setting(default = ...)]`
//!   2. TOML file (default path: workspace-root `profiles/<name>.toml`,
//!      where `<name>` comes from `PROXIMA_PROFILE`)
//!   3. Env-var overrides via `PROXIMA_*` prefix
//!   4. Validation rules (`Validate` impl) — cross-axis sanity
//!
//! Step 1-3 are conflaguration's job; step 4 is the cross-axis rules
//! that catch impossible combinations (e.g. `tls=rustls` with `std=false`)
//! before downstream crates try to build something that can't link.

use conflaguration::{Settings, ValidationMessage};
use serde::{Deserialize, Serialize};

fn default_quic_impl() -> String {
    "none".into()
}

fn default_h3_impl() -> String {
    "none".into()
}

/// Top-level profile. Drives both the cargo invocation (via the xtask
/// wrapper) and the generated module that downstream `build.rs` files
/// include into their `OUT_DIR`.
#[derive(Debug, Clone, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA")]
pub struct Profile {
    /// Schema version. Bump on breaking changes to this struct.
    #[setting(default = 1_u32)]
    pub schema: u32,

    /// Whether the global allocator is available (`alloc` crate usable).
    /// Required for `Box`, `Vec`, `String`, `Arc`, `Rc`, growable
    /// containers, and `Box<dyn Trait>` polymorphism.
    #[setting(default = true)]
    pub alloc: bool,

    /// Whether the OS layer is available (`std` crate usable). Required
    /// for `std::io`, `std::time::Instant`, `std::thread`, sockets,
    /// environment vars. Implies `alloc`.
    #[setting(default = true)]
    pub std: bool,

    /// Executor selection (string-typed for conflaguration's FromEnvStr
    /// bound; use [`Profile::executor_kind`] for typed access). Validated
    /// against alloc / std axes.
    #[setting(default = "tokio")]
    pub executor: String,

    /// Reactor / I/O readiness layer (string-typed; use
    /// [`Profile::reactor_kind`] for typed access).
    #[setting(default = "tokio-epoll")]
    pub reactor: String,

    /// TLS backend (string-typed; use [`Profile::tls_backend`] for typed
    /// access).
    #[setting(default = "rustls")]
    pub tls: String,

    /// Enable HTTP/3 + QUIC. Coarse switch. Composes with [`Self::quic_impl`]
    /// to select WHICH QUIC implementation provides the wire stack; when this
    /// is `false`, the `quic_impl` axis is forced to `none` regardless.
    #[setting(default = false)]
    pub quic_enabled: bool,

    /// QUIC implementation selection (string-typed; use [`Profile::quic_impl_kind`]
    /// for typed access). Blessed values: `quinn` (default until C41 cutover —
    /// std-only, tokio-coupled), `native` (proxima-quic-proto + proxima-quic —
    /// runtime-agnostic, no tokio in production paths, tier-1 capable),
    /// `none` (no QUIC). Optional in TOML — defaults to `none` when omitted,
    /// matching the historical `quic_enabled = false` shape.
    #[setting(default = "none")]
    #[serde(default = "default_quic_impl")]
    pub quic_impl: String,

    /// HTTP/3 implementation selection (string-typed; use [`Profile::h3_impl_kind`]
    /// for typed access). Blessed values: `h3-quinn` (default until C41 cutover —
    /// std-only, depends on h3 + h3-quinn + quinn), `native` (proxima-h3-proto +
    /// proxima-h3 — runtime-agnostic, no tokio in production paths), `none`.
    /// Requires [`Self::quic_impl`] to be non-none. Optional in TOML — defaults
    /// to `none`.
    #[setting(default = "none")]
    #[serde(default = "default_h3_impl")]
    pub h3_impl: String,

    /// Timer driver selection (string-typed; use [`Profile::timer_kind`] for
    /// typed access). Blessed values: `std-thread`, `prime-wheel`,
    /// `embassy-time`, `mock`. Any string containing `::` is treated as a
    /// fully-qualified path to a user-supplied `&'static dyn Driver` symbol.
    #[setting(default = "std-thread")]
    pub timer: String,
}

/// Executor selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Executor {
    /// Standard tokio runtime (multi-thread or current-thread).
    Tokio,
    /// Proxima's prime per-core runtime.
    Prime,
    /// embassy bare-metal executor.
    Embassy,
    /// Static-pool prime variant (no heap; embassy-comparable).
    StaticPrime,
}

impl core::str::FromStr for Executor {
    type Err = ExecutorParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "tokio" => Ok(Self::Tokio),
            "prime" => Ok(Self::Prime),
            "embassy" => Ok(Self::Embassy),
            "static-prime" => Ok(Self::StaticPrime),
            other => Err(ExecutorParseError(other.into())),
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("unknown executor: {0}")]
pub struct ExecutorParseError(pub String);

/// Reactor / I/O readiness layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Reactor {
    TokioEpoll,
    IoUring,
    Wasi,
    EmbassyNet,
    None,
}

impl core::str::FromStr for Reactor {
    type Err = ReactorParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "tokio-epoll" => Ok(Self::TokioEpoll),
            "io-uring" => Ok(Self::IoUring),
            "wasi" => Ok(Self::Wasi),
            "embassy-net" => Ok(Self::EmbassyNet),
            "none" => Ok(Self::None),
            other => Err(ReactorParseError(other.into())),
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("unknown reactor: {0}")]
pub struct ReactorParseError(pub String);

/// TLS backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TlsBackend {
    Rustls,
    EmbeddedTls,
    None,
}

impl core::str::FromStr for TlsBackend {
    type Err = TlsParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "rustls" => Ok(Self::Rustls),
            "embedded-tls" => Ok(Self::EmbeddedTls),
            "none" => Ok(Self::None),
            other => Err(TlsParseError(other.into())),
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("unknown tls backend: {0}")]
pub struct TlsParseError(pub String);

/// QUIC implementation backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum QuicImpl {
    /// Today's quinn / quinn-proto wrapper. std-only, tokio-coupled.
    /// The default until the C41 cutover commit (Phase D2) flips it to
    /// [`Self::Native`].
    Quinn,
    /// Greenfield proxima-quic-proto + proxima-quic. Runtime-agnostic;
    /// no tokio in production paths; tier-1 capable (no_std + alloc);
    /// composes aws-lc-rs + rustls (or inline TLS spike fallback).
    Native,
    /// QUIC disabled. Equivalent to `quic_enabled=false`.
    None,
}

impl core::str::FromStr for QuicImpl {
    type Err = QuicImplParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "quinn" => Ok(Self::Quinn),
            "native" => Ok(Self::Native),
            "none" => Ok(Self::None),
            other => Err(QuicImplParseError(other.into())),
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("unknown quic_impl: {0}")]
pub struct QuicImplParseError(pub String);

/// HTTP/3 implementation backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum H3Impl {
    /// Today's h3 + h3-quinn wrapper. Depends on quinn; std-only.
    /// The default until the C41 cutover commit (Phase D2) flips it to
    /// [`Self::Native`].
    H3Quinn,
    /// Greenfield proxima-h3-proto + proxima-h3. Runtime-agnostic;
    /// no tokio in production paths.
    Native,
    /// H3 disabled.
    None,
}

impl core::str::FromStr for H3Impl {
    type Err = H3ImplParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "h3-quinn" => Ok(Self::H3Quinn),
            "native" => Ok(Self::Native),
            "none" => Ok(Self::None),
            other => Err(H3ImplParseError(other.into())),
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("unknown h3_impl: {0}")]
pub struct H3ImplParseError(pub String);

/// Timer driver selection. `Custom(path)` is the open extension point — any
/// string containing `::` is treated as a fully-qualified path to a user
/// `&'static dyn proxima_core::time::Driver` symbol baked into the build by
/// `proxima_build::emit_timer_binding`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case", untagged)]
pub enum Timer {
    /// `proxima_core::time::drivers::std_thread::DRIVER` — host fallback (default).
    StdThread,
    /// `prime::TIMER_DRIVER` — proxima's per-core timer wheel.
    PrimeWheel,
    /// `embassy_time::DRIVER` — embassy-time driver.
    EmbassyTime,
    /// `proxima_core::time::drivers::mock::DRIVER` — deterministic test driver.
    Mock,
    /// Arbitrary user-supplied driver path, e.g. `"my_hal::DRIVER"`.
    Custom(String),
}

impl Timer {
    /// The path baked into the static binding emitted by `proxima-core`'s
    /// `build.rs` via [`crate::emit_timer_binding`]. First-party drivers
    /// use `crate::time::...` because the generated binding is included
    /// from inside `proxima-core`'s `time` module; external drivers use
    /// absolute paths because they live in other crates and resolve via
    /// that crate's `extern crate` declaration in the proxima-core build
    /// tree.
    #[must_use]
    pub fn driver_path(&self) -> &str {
        match self {
            Self::StdThread => "crate::time::drivers::std_thread::DRIVER",
            // prime-wheel is link-injected, not a cargo dep: proxima-core
            // can't depend on prime (prime -> proxima-pipe -> proxima-core
            // cycles). The ExternalDriver routes to the `extern "Rust"`
            // symbols prime exports; the linker binds them in the final crate.
            Self::PrimeWheel => "crate::time::drivers::external::DRIVER",
            Self::EmbassyTime => "embassy_time::DRIVER",
            Self::Mock => "crate::time::drivers::mock::DRIVER",
            Self::Custom(path) => path,
        }
    }
}

impl core::str::FromStr for Timer {
    type Err = TimerParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "std-thread" => Ok(Self::StdThread),
            "prime-wheel" => Ok(Self::PrimeWheel),
            "embassy-time" => Ok(Self::EmbassyTime),
            "mock" => Ok(Self::Mock),
            path if path.contains("::") => Ok(Self::Custom(path.into())),
            other => Err(TimerParseError(other.into())),
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("unknown timer: {0}")]
pub struct TimerParseError(pub String);

impl Default for Profile {
    fn default() -> Self {
        Self {
            schema: 1,
            alloc: true,
            std: true,
            executor: "tokio".into(),
            reactor: "tokio-epoll".into(),
            tls: "rustls".into(),
            quic_enabled: false,
            quic_impl: "none".into(),
            h3_impl: "none".into(),
            timer: "std-thread".into(),
        }
    }
}

impl Profile {
    /// Typed view of the [`Self::executor`] string.
    pub fn executor_kind(&self) -> core::result::Result<Executor, ExecutorParseError> {
        self.executor.parse()
    }

    /// Typed view of the [`Self::reactor`] string.
    pub fn reactor_kind(&self) -> core::result::Result<Reactor, ReactorParseError> {
        self.reactor.parse()
    }

    /// Typed view of the [`Self::tls`] string.
    pub fn tls_backend(&self) -> core::result::Result<TlsBackend, TlsParseError> {
        self.tls.parse()
    }

    /// Typed view of the [`Self::timer`] string.
    pub fn timer_kind(&self) -> core::result::Result<Timer, TimerParseError> {
        self.timer.parse()
    }

    /// Typed view of the [`Self::quic_impl`] string.
    pub fn quic_impl_kind(&self) -> core::result::Result<QuicImpl, QuicImplParseError> {
        self.quic_impl.parse()
    }

    /// Typed view of the [`Self::h3_impl`] string.
    pub fn h3_impl_kind(&self) -> core::result::Result<H3Impl, H3ImplParseError> {
        self.h3_impl.parse()
    }
}

impl conflaguration::Validate for Profile {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();

        let executor = match self.executor_kind() {
            Ok(value) => Some(value),
            Err(err) => {
                errors.push(ValidationMessage::new("executor", err.to_string()));
                None
            }
        };
        let reactor = match self.reactor_kind() {
            Ok(value) => Some(value),
            Err(err) => {
                errors.push(ValidationMessage::new("reactor", err.to_string()));
                None
            }
        };
        let tls = match self.tls_backend() {
            Ok(value) => Some(value),
            Err(err) => {
                errors.push(ValidationMessage::new("tls", err.to_string()));
                None
            }
        };
        let timer = match self.timer_kind() {
            Ok(value) => Some(value),
            Err(err) => {
                errors.push(ValidationMessage::new("timer", err.to_string()));
                None
            }
        };
        let quic_impl = match self.quic_impl_kind() {
            Ok(value) => Some(value),
            Err(err) => {
                errors.push(ValidationMessage::new("quic_impl", err.to_string()));
                None
            }
        };
        let h3_impl = match self.h3_impl_kind() {
            Ok(value) => Some(value),
            Err(err) => {
                errors.push(ValidationMessage::new("h3_impl", err.to_string()));
                None
            }
        };
        let _ = reactor;

        if self.std && !self.alloc {
            errors.push(ValidationMessage::new(
                "std",
                "std implies alloc; cannot have std=true with alloc=false",
            ));
        }

        if matches!(tls, Some(TlsBackend::Rustls)) && !(self.std && self.alloc) {
            errors.push(ValidationMessage::new(
                "tls",
                "tls=rustls requires std=true and alloc=true",
            ));
        }

        if matches!(tls, Some(TlsBackend::EmbeddedTls)) && !self.alloc {
            errors.push(ValidationMessage::new(
                "tls",
                "tls=embedded-tls requires alloc=true",
            ));
        }

        if self.quic_enabled {
            if !matches!(tls, Some(TlsBackend::Rustls)) {
                errors.push(ValidationMessage::new(
                    "quic_enabled",
                    "quic requires tls=rustls (rustls is the only TLS path in v1)",
                ));
            }
            if matches!(quic_impl, Some(QuicImpl::None)) {
                errors.push(ValidationMessage::new(
                    "quic_enabled",
                    "quic_enabled=true requires quic_impl != none",
                ));
            }
        } else if !matches!(quic_impl, Some(QuicImpl::None) | None) {
            errors.push(ValidationMessage::new(
                "quic_impl",
                "quic_impl must be 'none' when quic_enabled=false",
            ));
        }

        match quic_impl {
            Some(QuicImpl::Quinn) => {
                // legacy path — quinn requires the full tokio + rustls + std + alloc stack
                if self.quic_enabled && !matches!(executor, Some(Executor::Tokio)) {
                    errors.push(ValidationMessage::new(
                        "quic_impl",
                        "quic_impl=quinn requires executor=tokio (legacy quinn is tokio-coupled)",
                    ));
                }
                if self.quic_enabled && !(self.std && self.alloc) {
                    errors.push(ValidationMessage::new(
                        "quic_impl",
                        "quic_impl=quinn requires std=true and alloc=true",
                    ));
                }
            }
            Some(QuicImpl::Native) => {
                // native path — runtime-agnostic; production target is executor=prime;
                // executor=tokio allowed only via the tokio-compat feature (not enforced
                // at profile-axis level — that's a Cargo feature gate).
                if self.quic_enabled && !(self.std && self.alloc) {
                    errors.push(ValidationMessage::new(
                        "quic_impl",
                        "quic_impl=native requires std=true and alloc=true at the facade tier",
                    ));
                }
                // tier-1 (no_std + alloc) compiles for proxima-quic-proto only; the
                // proxima-quic facade is tier-2.
            }
            Some(QuicImpl::None) | None => {}
        }

        match h3_impl {
            Some(H3Impl::H3Quinn) => {
                // legacy path — h3-quinn requires quic_impl=quinn
                if self.quic_enabled && !matches!(quic_impl, Some(QuicImpl::Quinn)) {
                    errors.push(ValidationMessage::new(
                        "h3_impl",
                        "h3_impl=h3-quinn requires quic_impl=quinn",
                    ));
                }
            }
            Some(H3Impl::Native) => {
                // native h3 requires native QUIC
                if !matches!(quic_impl, Some(QuicImpl::Native)) {
                    errors.push(ValidationMessage::new(
                        "h3_impl",
                        "h3_impl=native requires quic_impl=native",
                    ));
                }
            }
            Some(H3Impl::None) | None => {}
        }

        if matches!(executor, Some(Executor::Embassy)) {
            if self.std {
                errors.push(ValidationMessage::new(
                    "executor",
                    "executor=embassy requires std=false",
                ));
            }
            if !self.alloc {
                errors.push(ValidationMessage::new(
                    "executor",
                    "executor=embassy requires alloc=true",
                ));
            }
        }

        if matches!(executor, Some(Executor::StaticPrime)) && self.alloc {
            errors.push(ValidationMessage::new(
                "executor",
                "executor=static-prime requires alloc=false",
            ));
        }

        if matches!(executor, Some(Executor::Tokio)) && !self.std {
            errors.push(ValidationMessage::new(
                "executor",
                "executor=tokio requires std=true",
            ));
        }

        match &timer {
            Some(Timer::StdThread) if !self.std => {
                errors.push(ValidationMessage::new(
                    "timer",
                    "timer=std-thread requires std=true",
                ));
            }
            Some(Timer::PrimeWheel) => {
                if !matches!(executor, Some(Executor::Prime | Executor::StaticPrime)) {
                    errors.push(ValidationMessage::new(
                        "timer",
                        "timer=prime-wheel requires executor=prime or executor=static-prime",
                    ));
                }
                if !self.alloc {
                    errors.push(ValidationMessage::new(
                        "timer",
                        "timer=prime-wheel requires alloc=true",
                    ));
                }
            }
            Some(Timer::EmbassyTime) if !matches!(executor, Some(Executor::Embassy)) => {
                errors.push(ValidationMessage::new(
                    "timer",
                    "timer=embassy-time requires executor=embassy",
                ));
            }
            Some(Timer::Mock) if !self.alloc => {
                errors.push(ValidationMessage::new(
                    "timer",
                    "timer=mock requires alloc=true",
                ));
            }
            Some(Timer::Custom(path)) if path.is_empty() => {
                errors.push(ValidationMessage::new(
                    "timer",
                    "custom timer path must be non-empty",
                ));
            }
            _ => {}
        }

        if self.schema != 1 {
            errors.push(ValidationMessage::new(
                "schema",
                "only schema=1 is supported by this proxima-build",
            ));
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use conflaguration::Validate;

    fn linux_daemon() -> Profile {
        Profile {
            schema: 1,
            alloc: true,
            std: true,
            executor: "tokio".into(),
            reactor: "tokio-epoll".into(),
            tls: "rustls".into(),
            quic_enabled: false,
            quic_impl: "none".into(),
            h3_impl: "none".into(),
            timer: "std-thread".into(),
        }
    }

    #[test]
    fn linux_daemon_validates() {
        linux_daemon()
            .validate()
            .expect("linux-daemon should validate");
    }

    #[test]
    fn quic_requires_full_stack() {
        let mut profile = linux_daemon();
        profile.quic_enabled = true;
        profile.quic_impl = "quinn".into();
        profile.h3_impl = "h3-quinn".into();
        profile
            .validate()
            .expect("quic with tokio+rustls+std+alloc should validate");

        profile.tls = "none".into();
        assert!(
            profile.validate().is_err(),
            "quic without tls should reject"
        );
    }

    #[test]
    fn quic_impl_native_accepts_executor_prime() {
        let mut profile = linux_daemon();
        profile.quic_enabled = true;
        profile.quic_impl = "native".into();
        profile.h3_impl = "native".into();
        profile.executor = "prime".into();
        profile.reactor = "io-uring".into();
        profile.timer = "prime-wheel".into();
        profile
            .validate()
            .expect("native QUIC + executor=prime should validate (runtime-agnostic)");
    }

    #[test]
    fn quic_impl_quinn_rejects_executor_prime() {
        let mut profile = linux_daemon();
        profile.quic_enabled = true;
        profile.quic_impl = "quinn".into();
        profile.h3_impl = "h3-quinn".into();
        profile.executor = "prime".into();
        profile.reactor = "io-uring".into();
        profile.timer = "prime-wheel".into();
        assert!(
            profile.validate().is_err(),
            "legacy quinn requires executor=tokio"
        );
    }

    #[test]
    fn h3_impl_native_requires_quic_impl_native() {
        let mut profile = linux_daemon();
        profile.quic_enabled = true;
        profile.quic_impl = "quinn".into();
        profile.h3_impl = "native".into();
        assert!(
            profile.validate().is_err(),
            "h3_impl=native requires quic_impl=native"
        );
    }

    #[test]
    fn h3_impl_h3_quinn_requires_quic_impl_quinn() {
        let mut profile = linux_daemon();
        profile.quic_enabled = true;
        profile.quic_impl = "native".into();
        profile.h3_impl = "h3-quinn".into();
        assert!(
            profile.validate().is_err(),
            "h3_impl=h3-quinn requires quic_impl=quinn"
        );
    }

    #[test]
    fn quic_disabled_forces_quic_impl_none() {
        let mut profile = linux_daemon();
        profile.quic_enabled = false;
        profile.quic_impl = "quinn".into();
        assert!(
            profile.validate().is_err(),
            "quic_impl must be 'none' when quic_enabled=false"
        );
    }

    #[test]
    fn quic_impl_from_str_roundtrip() {
        use core::str::FromStr;
        assert_eq!(QuicImpl::from_str("quinn").unwrap(), QuicImpl::Quinn);
        assert_eq!(QuicImpl::from_str("native").unwrap(), QuicImpl::Native);
        assert_eq!(QuicImpl::from_str("none").unwrap(), QuicImpl::None);
        assert!(QuicImpl::from_str("garbage").is_err());
    }

    #[test]
    fn h3_impl_from_str_roundtrip() {
        use core::str::FromStr;
        assert_eq!(H3Impl::from_str("h3-quinn").unwrap(), H3Impl::H3Quinn);
        assert_eq!(H3Impl::from_str("native").unwrap(), H3Impl::Native);
        assert_eq!(H3Impl::from_str("none").unwrap(), H3Impl::None);
        assert!(H3Impl::from_str("garbage").is_err());
    }

    #[test]
    fn rustls_requires_std_and_alloc() {
        let mut profile = linux_daemon();
        profile.std = false;
        profile.alloc = true; // std=false alone is ok; rustls requires both
        let result = profile.validate();
        assert!(result.is_err(), "rustls without std should reject");
    }

    #[test]
    fn embedded_tls_requires_alloc() {
        let profile = Profile {
            schema: 1,
            alloc: false,
            std: false,
            executor: "static-prime".into(),
            reactor: "none".into(),
            tls: "embedded-tls".into(),
            quic_enabled: false,
            quic_impl: "none".into(),
            h3_impl: "none".into(),
            timer: "proxima_time::drivers::cortex_m_systick::DRIVER".into(),
        };
        assert!(
            profile.validate().is_err(),
            "embedded-tls without alloc should reject"
        );
    }

    #[test]
    fn embassy_requires_no_std_with_alloc() {
        let mut profile = linux_daemon();
        profile.executor = "embassy".into();
        profile.std = false;
        profile.tls = "none".into();
        profile.reactor = "embassy-net".into();
        profile.timer = "embassy-time".into();
        profile
            .validate()
            .expect("embassy + alloc + no std should validate");

        profile.std = true;
        assert!(
            profile.validate().is_err(),
            "embassy with std should reject"
        );
    }

    #[test]
    fn static_prime_requires_no_alloc() {
        let profile = Profile {
            schema: 1,
            alloc: false,
            std: false,
            executor: "static-prime".into(),
            reactor: "none".into(),
            tls: "none".into(),
            quic_enabled: false,
            quic_impl: "none".into(),
            h3_impl: "none".into(),
            timer: "proxima_time::drivers::cortex_m_systick::DRIVER".into(),
        };
        profile
            .validate()
            .expect("static-prime + no alloc + no std should validate");

        let mut alloc_on = profile.clone();
        alloc_on.alloc = true;
        // alloc=true breaks static-prime; switch timer to one that allows alloc
        // so the only failing axis is the executor/alloc combination.
        alloc_on.timer = "mock".into();
        assert!(
            alloc_on.validate().is_err(),
            "static-prime with alloc should reject"
        );
    }

    #[test]
    fn schema_version_must_be_1() {
        let mut profile = linux_daemon();
        profile.schema = 2;
        assert!(
            profile.validate().is_err(),
            "unsupported schema should reject"
        );
    }

    #[test]
    fn executor_from_str_roundtrip() {
        use core::str::FromStr;
        assert_eq!(Executor::from_str("tokio").unwrap(), Executor::Tokio);
        assert_eq!(Executor::from_str("prime").unwrap(), Executor::Prime);
        assert_eq!(Executor::from_str("embassy").unwrap(), Executor::Embassy);
        assert_eq!(
            Executor::from_str("static-prime").unwrap(),
            Executor::StaticPrime
        );
        assert!(Executor::from_str("nonsense").is_err());
    }

    #[test]
    fn timer_from_str_roundtrip() {
        use core::str::FromStr;
        assert_eq!(Timer::from_str("std-thread").unwrap(), Timer::StdThread);
        assert_eq!(Timer::from_str("prime-wheel").unwrap(), Timer::PrimeWheel);
        assert_eq!(Timer::from_str("embassy-time").unwrap(), Timer::EmbassyTime);
        assert_eq!(Timer::from_str("mock").unwrap(), Timer::Mock);
        assert_eq!(
            Timer::from_str("my_hal::DRIVER").unwrap(),
            Timer::Custom("my_hal::DRIVER".into())
        );
        assert!(Timer::from_str("nonsense").is_err());
    }

    #[test]
    fn timer_driver_path_is_baked_for_blessed_variants() {
        assert_eq!(
            Timer::StdThread.driver_path(),
            "crate::time::drivers::std_thread::DRIVER"
        );
        assert_eq!(
            Timer::PrimeWheel.driver_path(),
            "crate::time::drivers::external::DRIVER"
        );
        assert_eq!(Timer::EmbassyTime.driver_path(), "embassy_time::DRIVER");
        assert_eq!(Timer::Mock.driver_path(), "crate::time::drivers::mock::DRIVER");
        let custom = Timer::Custom("x::Y".into());
        assert_eq!(custom.driver_path(), "x::Y");
    }

    #[test]
    fn timer_std_thread_requires_std() {
        let mut profile = linux_daemon();
        profile.std = false;
        profile.alloc = true;
        profile.tls = "none".into();
        profile.executor = "embassy".into();
        profile.reactor = "embassy-net".into();
        // timer stays at "std-thread" — should now fail because std=false
        assert!(
            profile.validate().is_err(),
            "timer=std-thread without std should reject"
        );
    }

    #[test]
    fn timer_embassy_requires_embassy_executor() {
        let mut profile = linux_daemon();
        profile.timer = "embassy-time".into();
        // executor is still tokio → mismatch
        assert!(
            profile.validate().is_err(),
            "timer=embassy-time with executor=tokio should reject"
        );
    }

    #[test]
    fn timer_prime_wheel_requires_prime_executor() {
        let mut profile = linux_daemon();
        profile.timer = "prime-wheel".into();
        profile.executor = "prime".into();
        profile
            .validate()
            .expect("prime-wheel + executor=prime + alloc should validate");

        profile.executor = "tokio".into();
        assert!(
            profile.validate().is_err(),
            "timer=prime-wheel with executor=tokio should reject"
        );
    }

    #[test]
    fn timer_custom_path_validates() {
        let mut profile = linux_daemon();
        profile.timer = "user_crate::DRIVER".into();
        profile
            .validate()
            .expect("custom timer path should validate");
        assert!(matches!(
            profile.timer_kind().unwrap(),
            Timer::Custom(ref path) if path == "user_crate::DRIVER"
        ));
    }
}
