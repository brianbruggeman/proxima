use std::ffi::CString;
use std::fs;
use std::io;
use std::os::unix::process::ExitStatusExt;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use serde::Deserialize;

const RUN_SPEC: &str = "/.proxima/run.json";
const POLL_INTERVAL: Duration = Duration::from_millis(2);
const PSEUDO_FILESYSTEMS: [(&str, &str); 3] = [("devtmpfs", "/dev"), ("proc", "/proc"), ("sysfs", "/sys")];

#[derive(Deserialize)]
struct RunSpec {
    argv: Vec<String>,
    env: Vec<String>,
    cwd: String,
}

fn mount_pseudo_filesystems() {
    for (kind, target) in PSEUDO_FILESYSTEMS {
        if let Err(error) = mount(kind, target) {
            eprintln!("oci-init: mount {kind} on {target}: {error}");
        }
    }
}

fn mount(kind: &str, target: &str) -> io::Result<()> {
    fs::create_dir_all(target)?;
    let (kind, target) = (to_c(kind)?, to_c(target)?);
    let status = unsafe { libc::mount(kind.as_ptr(), target.as_ptr(), kind.as_ptr(), 0, std::ptr::null()) };
    if status == 0 { Ok(()) } else { Err(io::Error::last_os_error()) }
}

fn to_c(text: &str) -> io::Result<CString> {
    CString::new(text).map_err(io::Error::other)
}

fn linger_from(frame: &[u8]) -> io::Result<Duration> {
    let text = std::str::from_utf8(frame).map_err(io::Error::other)?;
    let millis: u64 = text.trim().parse().map_err(|error| io::Error::other(format!("linger `{text}` is not milliseconds: {error}")))?;
    Ok(Duration::from_millis(millis))
}

fn spawn(spec: &RunSpec) -> io::Result<Child> {
    let (program, arguments) = spec.argv.split_first().ok_or_else(|| io::Error::other("run spec has an empty argv"))?;
    Command::new(program)
        .args(arguments)
        .env_clear()
        .envs(spec.env.iter().filter_map(|entry| entry.split_once('=')))
        .stdin(Stdio::null())
        .spawn()
}

fn describe(status: ExitStatus) -> String {
    match (status.code(), status.signal()) {
        (Some(code), _) => format!("exited:{code}"),
        (None, Some(signal)) => format!("signal:{signal}"),
        (None, None) => "ended".to_string(),
    }
}

fn supervise(child: &mut Child, linger: Duration) -> io::Result<String> {
    let started = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(describe(status));
        }
        if started.elapsed() >= linger {
            child.kill()?;
            child.wait()?;
            return Ok("running".to_string());
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

fn step<Value>(what: &str, outcome: io::Result<Value>) -> io::Result<Value> {
    outcome.map_err(|error| io::Error::new(error.kind(), format!("{what}: {error}")))
}

fn serve() -> io::Result<()> {
    mount_pseudo_filesystems();
    let spec_bytes = step("read /.proxima/run.json", fs::read(RUN_SPEC))?;
    let spec: RunSpec = serde_json::from_slice(&spec_bytes).map_err(io::Error::other)?;
    let mut tty = step("open the invocation tty", linux_guest::open_invocation())?;
    let linger = linger_from(&step("read the linger frame", linux_guest::read_frame(&mut tty))?)?;
    linux_guest::mark_entrypoint();
    step(&format!("chdir {}", spec.cwd), std::env::set_current_dir(&spec.cwd))?;
    let mut child = step(&format!("spawn {:?}", spec.argv), spawn(&spec))?;
    let report = supervise(&mut child, linger)?;
    linux_guest::write_frame(&mut tty, report.as_bytes())
}

fn main() {
    linux_guest::mark_init_started();
    if let Err(error) = serve() {
        eprintln!("oci-init: {error}");
    }
    linux_guest::power_off()
}
