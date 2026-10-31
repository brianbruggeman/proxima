use std::io;

fn serve() -> io::Result<()> {
    let mut tty = linux_guest::open_invocation()?;
    linux_guest::read_frame(&mut tty)?;
    let mut declared_100_sends_3 = 100_u64.to_le_bytes().to_vec();
    declared_100_sends_3.extend_from_slice(b"abc");
    linux_guest::write_raw(&mut tty, &declared_100_sends_3)
}

fn main() {
    linux_guest::mark_init_started();
    if let Err(error) = serve() {
        eprintln!("truncated-init: {error}");
    }
    linux_guest::power_off()
}
