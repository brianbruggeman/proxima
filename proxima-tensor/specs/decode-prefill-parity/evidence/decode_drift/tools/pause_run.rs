use std::process::Command;

struct Resume {
    pid: String,
}

impl Drop for Resume {
    fn drop(&mut self) {
        let status = Command::new("kill").args(["-CONT", &self.pid]).status();
        eprintln!("pause_run: SIGCONT pid={} status={status:?}", self.pid);
    }
}

fn main() {
    let mut arguments = std::env::args().skip(1);
    let pid = arguments.next().expect("pid to pause");
    let program = arguments.next().expect("program");
    let rest: Vec<String> = arguments.collect();
    let state = Command::new("ps").args(["-o", "pid,state,args", "-p", &pid]).output().expect("ps");
    eprintln!("pause_run: before {}", String::from_utf8_lossy(&state.stdout).trim());
    let guard = Resume { pid: pid.clone() };
    let status = Command::new("kill").args(["-STOP", &pid]).status().expect("kill -STOP");
    assert!(status.success(), "SIGSTOP failed");
    eprintln!("pause_run: SIGSTOP pid={pid}");
    let result = Command::new(&program).args(&rest).status();
    eprintln!("pause_run: child status={result:?}");
    drop(guard);
    let state = Command::new("ps").args(["-o", "pid,state,args", "-p", &pid]).output().expect("ps");
    eprintln!("pause_run: after {}", String::from_utf8_lossy(&state.stdout).trim());
}
