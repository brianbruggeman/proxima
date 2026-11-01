const PEER: &str = "10.0.2.2:9";

fn step(message: &str) {
    println!("net-init: {message}");
}

fn run() -> std::io::Result<()> {
    linux_guest::mark_init_started();
    step("start");
    let socket = linux_guest::network::bring_up("net-init")?;
    socket.send_to(b"proxima", PEER)?;
    step("sent udp to 10.0.2.2:9");
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        step(&format!("failed: {error}"));
    }
    loop {
        unsafe { libc::pause() };
    }
}
