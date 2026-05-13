use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::time::SystemTime;

use proxima_intercept::InterceptPipe;
use proxima_intercept::ca::{ca_cert_pem, ca_key_pem, generate_ca};
use proxima_intercept::capture::Capture;

const BIND: &str = "127.0.0.1:9090";

fn binary_mtime() -> Option<SystemTime> {
    std::env::current_exe()
        .ok()
        .and_then(|path| std::fs::metadata(path).ok())
        .and_then(|meta| meta.modified().ok())
}

// process-wide armed spigot: one 1-core prime runtime backs every capture's
// off-core blocking I/O (created once; cloning the Arc<OnceLock> is a bump).
fn armed_spigot() -> proxima_recording::pipe::DeferredRuntime {
    static SPIGOT: std::sync::OnceLock<proxima_recording::pipe::DeferredRuntime> =
        std::sync::OnceLock::new();
    SPIGOT
        .get_or_init(|| {
            let spigot = proxima_recording::pipe::deferred_runtime();
            spigot
                .set(
                    std::sync::Arc::new(proxima::runtime::PrimeRuntime::new(1).expect("prime"))
                        as std::sync::Arc<dyn proxima::runtime::Runtime>,
                )
                .ok();
            spigot
        })
        .clone()
}

fn main() -> Result<(), proxima_core::ProximaError> {
    let ca_dir =
        PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/tmp".into())).join(".proxima");
    std::fs::create_dir_all(&ca_dir)
        .map_err(|err| proxima_core::ProximaError::Config(format!("create ca dir: {err}")))?;
    let cert_path = ca_dir.join("ca.pem");
    let key_path = ca_dir.join("ca-key.pem");

    if !cert_path.exists() || !key_path.exists() {
        eprintln!("generating ca at {}", ca_dir.display());
        let ca = generate_ca()?;
        std::fs::write(&cert_path, ca_cert_pem(&ca)?).map_err(proxima_core::ProximaError::Io)?;
        std::fs::write(&key_path, ca_key_pem(&ca)).map_err(proxima_core::ProximaError::Io)?;
    } else {
        eprintln!("using ca from {}", ca_dir.display());
    }

    let capture_path = std::env::var("PROXIMA_INTERCEPT_CAPTURE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| ca_dir.join("intercept.bin"));

    let start_mtime = binary_mtime();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|err| proxima_core::ProximaError::Config(format!("tokio runtime: {err}")))?;

    rt.block_on(async {
        let capture = Capture::open(&capture_path, armed_spigot())?;
        eprintln!("recording to {}", capture_path.display());

        let pipe = InterceptPipe::with_ca_files(&cert_path, &key_path)?.with_capture(capture);
        let handle = proxima::into_handle(pipe);

        let mut app = proxima::App::new()?;
        app.pipe("intercept", proxima::Spec::Handle(handle)).await?;
        app.mount("/{*path}", proxima::MountTarget::Named("intercept".into()))?;

        let addr: std::net::SocketAddr = BIND
            .parse()
            .map_err(|err| proxima_core::ProximaError::Config(format!("parse bind: {err}")))?;

        eprintln!("listening on {addr}");
        eprintln!("usage: HTTPS_PROXY=http://{addr} agent -i \"hello\"");

        let shutdown = app.run_until_signal(proxima::RunConfig::http(addr)).await?;

        use tokio::signal::unix::{SignalKind, signal};
        let mut sigterm = signal(SignalKind::terminate()).map_err(|err| {
            proxima_core::ProximaError::Io(std::io::Error::other(format!("{err}")))
        })?;
        let mut sigint = signal(SignalKind::interrupt()).map_err(|err| {
            proxima_core::ProximaError::Io(std::io::Error::other(format!("{err}")))
        })?;

        let binary_changed = async {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                if binary_mtime() != start_mtime {
                    eprintln!("[hot-reload] binary changed, restarting...");
                    return;
                }
            }
        };

        tokio::select! {
            _ = sigterm.recv() => {}
            _ = sigint.recv() => {}
            _ = binary_changed => {}
        }
        // Race the drain against a second signal so a stuck connection
        // can't trap the operator: the first ctrl+c starts the drain;
        // a second ctrl+c (or SIGTERM) forces immediate exit. The decision
        // logic is the tested drain_or_force helper; this shell only maps the
        // outcome to a process exit code.
        use proxima_intercept::shutdown::{DrainOutcome, drain_or_force};
        let drain = async {
            shutdown.drain().await;
        };
        let on_interrupt = async {
            let _ = sigint.recv().await;
        };
        let on_terminate = async {
            let _ = sigterm.recv().await;
        };
        match drain_or_force(drain, on_interrupt, on_terminate).await {
            DrainOutcome::Drained => {}
            DrainOutcome::ForcedByInterrupt => {
                eprintln!("\n[force-exit] second SIGINT during drain — exiting immediately");
                std::process::exit(130);
            }
            DrainOutcome::ForcedByTerminate => std::process::exit(143),
        }
        Ok::<(), proxima_core::ProximaError>(())
    })?;

    if binary_mtime() != start_mtime {
        let exe = std::env::current_exe()
            .map_err(|err| proxima_core::ProximaError::Config(format!("current_exe: {err}")))?;
        let args: Vec<String> = std::env::args().collect();
        let err = std::process::Command::new(&exe).args(&args[1..]).exec();
        return Err(proxima_core::ProximaError::Config(format!(
            "exec failed: {err}"
        )));
    }

    Ok(())
}
