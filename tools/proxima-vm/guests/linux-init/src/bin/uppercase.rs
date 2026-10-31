use std::io;

fn serve() -> io::Result<()> {
    let mut tty = linux_guest::open_invocation()?;
    let request = linux_guest::read_frame(&mut tty)?;
    linux_guest::write_frame(&mut tty, &request.to_ascii_uppercase())
}

fn main() {
    linux_guest::mark_init_started();
    if let Err(error) = serve() {
        eprintln!("uppercase-init: {error}");
    }
    linux_guest::power_off()
}
