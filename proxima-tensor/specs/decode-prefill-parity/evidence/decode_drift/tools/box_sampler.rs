use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::thread::sleep;
use std::time::Duration;

fn capture(program: &str, arguments: &[&str]) -> String {
    Command::new(program)
        .args(arguments)
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
        .unwrap_or_else(|error| format!("{program} failed: {error}\n"))
}

fn main() {
    let mut arguments = std::env::args().skip(1);
    let output_path = arguments.next().expect("output path");
    let stop_path = arguments.next().expect("stop file path");
    let interval_seconds: u64 = arguments.next().expect("interval seconds").parse().expect("integer");
    let mut file = OpenOptions::new().create(true).append(true).open(&output_path).expect("open output");
    while !Path::new(&stop_path).exists() {
        let stamp = capture("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"]);
        let load = capture("uptime", &[]);
        let top = capture("ps", &["-Aro", "pid,pcpu,state,comm"]);
        let therm = capture("pmset", &["-g", "therm"]);
        let top8: String = top.lines().take(9).collect::<Vec<_>>().join("\n");
        writeln!(file, "--- {}{}{}\n{}", stamp.trim(), "\n", load.trim(), top8).expect("write");
        writeln!(file, "{}", therm.trim()).expect("write");
        file.flush().expect("flush");
        sleep(Duration::from_secs(interval_seconds));
    }
}
