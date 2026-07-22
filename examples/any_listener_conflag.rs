#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Conflaguration as first-class: the SAME listener, the blacklist's strike
//! thresholds driven from a real TOML file instead of hardcoded `.with_*`
//! calls — `BlacklistConfig::layered().from_path(path)`
//! (`proxima-listen/src/admission/blacklist.rs`), the identical
//! `#[derive(Builder, Deserialize, Serialize, Settings)]` + layered-builder
//! house pattern `proxima-listen/src/config.rs`'s `ListenTuningConfig` also
//! uses.
//!
//! ## Two DIFFERENT TOML shapes — do not conflate them
//!
//! There are two genuinely different `[admission...]` TOML surfaces in this
//! codebase, and the runtime one this example drives is NOT nested the way
//! the build-time one is:
//!
//! 1. **Build-time SIZING TOML** (`proxima-listen/proxima-listen-core.toml`,
//!    read by `build.rs`, baked into `sized::` consts) — nested under
//!    `[admission]` / `[admission.blacklist]` table headers. This is the
//!    no_std+no_alloc FLOOR (`deny_strike_threshold = 1`, etc.) — it never
//!    changes without a rebuild.
//! 2. **Runtime layered TOML** (what `.from_path(path)` reads here) — FLAT,
//!    no section header at all: `conflaguration::from_file` deserializes the
//!    file straight into `BlacklistConfigPartial`
//!    (`proxima-listen/src/admission/blacklist.rs:152`), which has no outer
//!    table name. A file with a `[admission.blacklist]` header would parse
//!    as nested data the partial doesn't have fields for and silently keep
//!    every field at its `sized`-seeded default — a real, worth-knowing gotcha
//!    this example's own test (§2) proves directly.
//!
//! Run: `cargo run --example any_listener_conflag --features http1-native`

use std::future::Future;
use std::net::{Ipv4Addr, SocketAddr};

use bytes::Bytes;

use proxima::pipe::into_handle;
use proxima::request::{Request, Response};
use proxima::{Listener, ListenerBuilderEntry, ProximaError};
use proxima_listen::admission::BlacklistConfig;
use proxima_primitives::pipe::SendPipe;

struct LegitOk;

impl SendPipe for LegitOk {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move { Ok(Response::new(200).with_body(Bytes::from_static(b"ok"))) }
    }
}

fn free_loopback_addr() -> Result<SocketAddr, ProximaError> {
    let probe = std::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
    let addr = probe.local_addr()?;
    drop(probe);
    Ok(addr)
}

#[proxima::main]
async fn main() -> Result<(), ProximaError> {
    // ── §1: a real TOML file drives the SAME `.blacklist(config)` axis ──────
    let toml_dir = tempfile::tempdir().expect("tempdir for the conflag file");
    let toml_path = toml_dir.path().join("blacklist.toml");
    std::fs::write(
        &toml_path,
        "deny_strike_threshold = 1\n\
         unclassifiable_strike_threshold = 5\n\
         strike_window_ms = 60000\n\
         ban_duration_ms = 300000\n",
    )
    .expect("write conflag toml");

    let config = BlacklistConfig::layered()
        .from_path(&toml_path)
        .expect("a well-formed file loads")
        .build();
    assert_eq!(config.deny_strike_threshold, 1);
    assert_eq!(
        config.unclassifiable_strike_threshold, 5,
        "the file's own value, not the sized default (20)"
    );
    println!(
        "§1: BlacklistConfig loaded from {} — unclassifiable_strike_threshold={} (file wins over \
         the sized default)",
        toml_path.display(),
        config.unclassifiable_strike_threshold
    );

    let bind = free_loopback_addr()?;
    let server = Listener::builder()
        .bind(bind)
        .accept("h1")
        .deny("scanner", b"XSCANPROBE\r\n".to_vec())
        .blacklist(config)
        .handle(into_handle(LegitOk))
        .serve()
        .await?;
    println!("§1: listener {bind} running with the file-driven blacklist config");
    server.stop();

    // ── §2: the flat-vs-nested gotcha, proven directly ──────────────────────
    // A `[admission.blacklist]` section header — matching the BUILD-TIME
    // sizing TOML's shape — does NOT populate `BlacklistConfigPartial`'s
    // top-level fields. `.from_path` succeeds (it's syntactically valid
    // TOML) but every field silently stays at its `sized`-seeded default,
    // because the partial has no field literally named `admission`.
    let nested_path = toml_dir.path().join("nested-by-mistake.toml");
    std::fs::write(
        &nested_path,
        "[admission.blacklist]\ndeny_strike_threshold = 99\n",
    )
    .expect("write the nested-by-mistake toml");
    let nested_config = BlacklistConfig::layered()
        .from_path(&nested_path)
        .expect("syntactically valid TOML still loads")
        .build();
    assert_eq!(
        nested_config.deny_strike_threshold,
        BlacklistConfig::default().deny_strike_threshold,
        "a [admission.blacklist]-nested file is silently ignored by this runtime loader — \
         it stays at the sized default, NOT the file's 99"
    );
    println!(
        "§2: a [admission.blacklist]-nested TOML loads WITHOUT error but changes nothing \
         (deny_strike_threshold stayed at the default {}, not the file's 99) — the runtime \
         loader wants a FLAT file, unlike the build-time sizing TOML",
        nested_config.deny_strike_threshold
    );

    println!("\nany_listener_conflag: file-driven blacklist config + the flat-vs-nested gotcha OK");
    Ok(())
}
