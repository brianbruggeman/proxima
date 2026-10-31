use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;

const INVOCATION_TTY: &str = "/dev/ttyS1";
const MARKER_PORT: u16 = 0xe9;
const FRAME_PREFIX: usize = 8;
const MAX_REQUEST: u64 = 64 << 20;

const MARKER_INIT_STARTED: u8 = 1;
const MARKER_INVOCATION_READY: u8 = 2;

fn mark(value: u8) -> io::Result<()> {
    if unsafe { libc::syscall(libc::SYS_iopl, 3) } != 0 {
        return Err(io::Error::last_os_error());
    }
    // the host answers this port, so init can be timestamped and can release the input stream
    unsafe {
        core::arch::asm!("out dx, al", in("dx") MARKER_PORT, in("al") value, options(nomem, nostack));
    }
    Ok(())
}

pub fn mark_init_started() {
    if let Err(error) = mark(MARKER_INIT_STARTED) {
        eprintln!("linux guest: init marker unavailable: {error}");
    }
}

fn set_raw(tty: &File) -> io::Result<()> {
    let descriptor = tty.as_raw_fd();
    let mut settings = unsafe { std::mem::zeroed::<libc::termios>() };
    if unsafe { libc::tcgetattr(descriptor, &raw mut settings) } != 0 {
        return Err(io::Error::last_os_error());
    }
    unsafe { libc::cfmakeraw(&raw mut settings) };
    if unsafe { libc::tcsetattr(descriptor, libc::TCSANOW, &raw const settings) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

pub fn open_invocation() -> io::Result<File> {
    let tty = OpenOptions::new().read(true).write(true).custom_flags(libc::O_NOCTTY).open(INVOCATION_TTY)?;
    set_raw(&tty)?;
    mark(MARKER_INVOCATION_READY)?;
    Ok(tty)
}

pub fn read_frame(tty: &mut File) -> io::Result<Vec<u8>> {
    let mut prefix = [0_u8; FRAME_PREFIX];
    tty.read_exact(&mut prefix)?;
    let length = u64::from_le_bytes(prefix);
    if length > MAX_REQUEST {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("request of {length} bytes exceeds {MAX_REQUEST}")));
    }
    let mut body = vec![0_u8; length as usize];
    tty.read_exact(&mut body)?;
    Ok(body)
}

pub fn write_raw(tty: &mut File, bytes: &[u8]) -> io::Result<()> {
    tty.write_all(bytes)?;
    tty.flush()?;
    if unsafe { libc::tcdrain(tty.as_raw_fd()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

pub fn write_frame(tty: &mut File, payload: &[u8]) -> io::Result<()> {
    let mut framed = Vec::with_capacity(FRAME_PREFIX + payload.len());
    framed.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    framed.extend_from_slice(payload);
    write_raw(tty, &framed)
}

/// `reboot=k` on the kernel cmdline turns this into the host-visible i8042
/// reset write; init must never return, so a failed reboot parks.
pub fn power_off() -> ! {
    unsafe { libc::reboot(libc::RB_AUTOBOOT) };
    loop {
        std::hint::spin_loop();
    }
}
