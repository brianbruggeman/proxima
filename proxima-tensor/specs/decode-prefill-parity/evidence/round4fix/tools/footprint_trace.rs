// Spawns one command and samples its physical footprint and resident set every millisecond until it exits.
//
// build: rustc -O footprint_trace.rs -o footprint_trace
// usage: footprint_trace <trace.tsv> <stderr.log> -- <command> [args...]    (environment is inherited; the caller clears it)
//
// The trace has one line per sample: milliseconds since spawn, footprint bytes, resident bytes. Child stderr
// lines are written to <stderr.log> prefixed with the same millisecond clock so a phase boundary the child
// prints lines up with the trace.

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

#[repr(C)]
#[derive(Default)]
struct RusageInfoV2 {
    uuid: [u8; 16],
    user_time: u64,
    system_time: u64,
    pkg_idle_wkups: u64,
    interrupt_wkups: u64,
    pageins: u64,
    wired_size: u64,
    resident_size: u64,
    phys_footprint: u64,
    proc_start_abstime: u64,
    proc_exit_abstime: u64,
    child_user_time: u64,
    child_system_time: u64,
    child_pkg_idle_wkups: u64,
    child_interrupt_wkups: u64,
    child_pageins: u64,
    child_elapsed_abstime: u64,
    diskio_bytesread: u64,
    diskio_byteswritten: u64,
}

unsafe extern "C" {
    fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut RusageInfoV2) -> i32;
}

const RUSAGE_INFO_V2: i32 = 2;

fn sample(pid: i32) -> Option<(u64, u64)> {
    let mut info = RusageInfoV2::default();
    // SAFETY: `info` is a valid, writable `rusage_info_v2`; the flavor matches its layout.
    let status = unsafe { proc_pid_rusage(pid, RUSAGE_INFO_V2, &mut info) };
    (status == 0).then_some((info.phys_footprint, info.resident_size))
}

fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    let separator = arguments.iter().position(|value| value == "--").expect("-- before the command");
    let trace_path = &arguments[1];
    let stderr_path = &arguments[2];
    let command = &arguments[separator + 1];
    let mut child = Command::new(command)
        .args(&arguments[separator + 2..])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn command");
    let started = Instant::now();
    let pid = child.id() as i32;
    let stderr = child.stderr.take().expect("piped stderr");
    let lines: Arc<Mutex<Vec<(u128, String)>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&lines);
    let reader = thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            sink.lock().expect("lock").push((started.elapsed().as_millis(), line));
        }
    });
    let mut trace = File::create(trace_path).expect("create trace");
    while child.try_wait().expect("poll child").is_none() {
        if let Some((footprint, resident)) = sample(pid) {
            writeln!(trace, "{}\t{footprint}\t{resident}", started.elapsed().as_millis()).expect("write trace");
        }
        thread::sleep(Duration::from_millis(1));
    }
    reader.join().expect("join stderr reader");
    let mut log = File::create(stderr_path).expect("create stderr log");
    for (millis, line) in lines.lock().expect("lock").iter() {
        writeln!(log, "{millis}\t{line}").expect("write stderr log");
    }
}
