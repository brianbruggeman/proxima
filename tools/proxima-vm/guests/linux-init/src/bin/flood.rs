use std::io;

const FLOOD_BYTES: usize = 64 << 10;

fn serve() -> io::Result<()> {
    let mut tty = linux_guest::open_invocation()?;
    linux_guest::read_frame(&mut tty)?;
    linux_guest::write_frame(&mut tty, &vec![b'x'; FLOOD_BYTES])
}

fn main() {
    linux_guest::mark_init_started();
    if let Err(error) = serve() {
        eprintln!("flood-init: {error}");
    }
    linux_guest::power_off()
}
