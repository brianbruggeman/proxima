use std::ffi::CString;
use std::fs::{self, File};
use std::io;
use std::net::UdpSocket;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::thread;
use std::time::Duration;

const MODULES_DIR: &str = "/.proxima/modules";
const MODULE_INIT_COMPRESSED_FILE: libc::c_uint = 4;
const VIRTIO_MMIO_PARAMS: &str = "device=4K@0xd0000000:5";
const GUEST_ADDRESS: [u8; 4] = [10, 0, 2, 15];
const NETMASK: [u8; 4] = [255, 255, 255, 0];
const BRING_UP_ATTEMPTS: u32 = 100;
const BRING_UP_RETRY: Duration = Duration::from_millis(20);

fn module_params(name: &str) -> &'static str {
    match name {
        "virtio_mmio" => VIRTIO_MMIO_PARAMS,
        _ => "",
    }
}

fn module_name(file_name: &str) -> &str {
    let stripped = file_name.split_once('-').map_or(file_name, |(_, rest)| rest);
    stripped.split(".ko").next().unwrap_or(stripped)
}

fn load_module(path: &Path) -> io::Result<()> {
    let file_name = path.file_name().and_then(|name| name.to_str()).unwrap_or_default();
    let params = CString::new(module_params(module_name(file_name)))?;
    let compressed = !file_name.ends_with(".ko");
    let flags = if compressed { MODULE_INIT_COMPRESSED_FILE } else { 0 };
    let file = File::open(path)?;
    let result = unsafe { libc::syscall(libc::SYS_finit_module, file.as_raw_fd(), params.as_ptr(), flags) };
    if result == 0 { Ok(()) } else { Err(io::Error::last_os_error()) }
}

fn load_modules(label: &str) -> io::Result<()> {
    let mut paths: Vec<_> = fs::read_dir(MODULES_DIR)?.map(|entry| entry.map(|found| found.path())).collect::<io::Result<_>>()?;
    paths.sort();
    for path in paths {
        match load_module(&path) {
            Ok(()) => println!("{label}: loaded {}", path.display()),
            Err(error) => println!("{label}: load of {} failed: {error}", path.display()),
        }
    }
    Ok(())
}

fn request(name: &str) -> libc::ifreq {
    let mut request = unsafe { std::mem::zeroed::<libc::ifreq>() };
    for (slot, byte) in request.ifr_name.iter_mut().zip(name.bytes()) {
        *slot = byte as libc::c_char;
    }
    request
}

fn address(octets: [u8; 4]) -> libc::sockaddr {
    let mut inet = unsafe { std::mem::zeroed::<libc::sockaddr_in>() };
    inet.sin_family = libc::AF_INET as libc::sa_family_t;
    inet.sin_addr.s_addr = u32::from_ne_bytes(octets);
    unsafe { std::mem::transmute::<libc::sockaddr_in, libc::sockaddr>(inet) }
}

fn control(socket: &UdpSocket, code: libc::c_ulong, request: &mut libc::ifreq) -> io::Result<()> {
    if unsafe { libc::ioctl(socket.as_raw_fd(), code as libc::Ioctl, std::ptr::from_mut(request)) } == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

fn bring_up_interface(socket: &UdpSocket, name: &str, addressing: Option<([u8; 4], [u8; 4])>) -> io::Result<()> {
    if let Some((local, mask)) = addressing {
        let mut set_address = request(name);
        set_address.ifr_ifru.ifru_addr = address(local);
        control(socket, libc::SIOCSIFADDR, &mut set_address)?;
        let mut set_mask = request(name);
        set_mask.ifr_ifru.ifru_netmask = address(mask);
        control(socket, libc::SIOCSIFNETMASK, &mut set_mask)?;
    }
    let mut flags = request(name);
    control(socket, libc::SIOCGIFFLAGS, &mut flags)?;
    unsafe { flags.ifr_ifru.ifru_flags |= libc::IFF_UP as libc::c_short };
    control(socket, libc::SIOCSIFFLAGS, &mut flags)
}

fn bring_up_when_present(socket: &UdpSocket, name: &str) -> io::Result<()> {
    let mut last = io::Error::from(io::ErrorKind::NotFound);
    for _ in 0..BRING_UP_ATTEMPTS {
        match bring_up_interface(socket, name, Some((GUEST_ADDRESS, NETMASK))) {
            Ok(()) => return Ok(()),
            Err(error) => last = error,
        }
        thread::sleep(BRING_UP_RETRY);
    }
    Err(last)
}

/// Loads the packed virtio modules and brings lo and eth0 (10.0.2.15/24) up, printing progress as `label`.
pub fn bring_up(label: &str) -> io::Result<UdpSocket> {
    load_modules(label)?;
    let socket = UdpSocket::bind("0.0.0.0:0")?;
    bring_up_interface(&socket, "lo", None)?;
    println!("{label}: lo up");
    bring_up_when_present(&socket, "eth0")?;
    println!("{label}: eth0 up 10.0.2.15/24");
    Ok(socket)
}
