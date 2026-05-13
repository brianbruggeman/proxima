//! The XDP redirect program that steers RX frames into our AF_XDP socket,
//! driven entirely over the raw `bpf(2)` and netlink syscalls — no libbpf,
//! libxdp, or aya, and nothing C-linked. Three pieces:
//!
//! * an `XSKMAP` (`bpf(BPF_MAP_CREATE)`) keyed by RX queue index, valued by an
//!   AF_XDP socket fd;
//! * a hand-encoded eBPF program (`bpf(BPF_PROG_LOAD)`) that reads
//!   `xdp_md.rx_queue_index` and calls `bpf_redirect_map(xskmap, index,
//!   XDP_PASS)` — GPL-licensed because `bpf_redirect_map` is a GPL-only helper;
//! * a netlink `RTM_SETLINK`/`IFLA_XDP` attach in SKB (generic) mode, reliable
//!   on veth.
//!
//! [`XdpProgram`] owns the map and program fds and detaches the program from
//! the netdev on drop, so a dropped listener leaves the interface clean.

use super::error::XdpError;
use std::ffi::c_void;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::ptr;

const BPF_MAP_CREATE: u32 = 0;
const BPF_MAP_UPDATE_ELEM: u32 = 2;
const BPF_PROG_LOAD: u32 = 5;
const BPF_MAP_TYPE_XSKMAP: u32 = 17;
const BPF_PROG_TYPE_XDP: u32 = 6;
const BPF_FUNC_REDIRECT_MAP: i32 = 51;
const BPF_PSEUDO_MAP_FD: u8 = 1;

// eBPF instruction class/op bytes (linux/bpf_common.h + bpf.h).
const BPF_LDX_MEM_W: u8 = 0x61; // BPF_LDX | BPF_MEM | BPF_W
const BPF_LD_IMM_DW: u8 = 0x18; // BPF_LD  | BPF_IMM | BPF_DW (wide, 2 insns)
const BPF_ALU64_MOV_K: u8 = 0xb7; // BPF_ALU64 | BPF_MOV | BPF_K
const BPF_JMP_CALL: u8 = 0x85; // BPF_JMP | BPF_CALL
const BPF_JMP_EXIT: u8 = 0x95; // BPF_JMP | BPF_EXIT

// xdp_md.rx_queue_index byte offset in the context struct.
const XDP_MD_RX_QUEUE_INDEX: i16 = 16;

// netlink / rtnetlink constants.
const RTM_SETLINK: u16 = 19;
const NLM_F_REQUEST: u16 = 0x01;
const NLM_F_ACK: u16 = 0x04;
const NLMSG_ERROR: u16 = 2;
const IFLA_XDP: u16 = 43;
const IFLA_XDP_FD: u16 = 1;
const IFLA_XDP_FLAGS: u16 = 3;
const NLA_F_NESTED: u16 = 0x8000;
/// Generic/SKB attach mode — reliable on veth (no driver XDP needed).
pub const XDP_FLAGS_SKB_MODE: u32 = 1 << 1;
/// Native/DRV attach mode — the driver's own XDP hook (required for AF_XDP
/// zerocopy). Supported by veth's native XDP path.
pub const XDP_FLAGS_DRV_MODE: u32 = 1 << 2;

const VERIFIER_LOG_SIZE: usize = 16 * 1024;

/// One eBPF instruction (`struct bpf_insn`): 8 bytes, no padding.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct BpfInsn {
    code: u8,
    regs: u8,
    off: i16,
    imm: i32,
}

#[repr(C)]
struct MapCreateAttr {
    map_type: u32,
    key_size: u32,
    value_size: u32,
    max_entries: u32,
    map_flags: u32,
    inner_map_fd: u32,
    numa_node: u32,
    map_name: [u8; 16],
    map_ifindex: u32,
}

#[repr(C)]
struct ProgLoadAttr {
    prog_type: u32,
    insn_cnt: u32,
    insns: u64,
    license: u64,
    log_level: u32,
    log_size: u32,
    log_buf: u64,
    kern_version: u32,
    prog_flags: u32,
    prog_name: [u8; 16],
    prog_ifindex: u32,
    expected_attach_type: u32,
}

#[repr(C)]
struct MapUpdateAttr {
    map_fd: u32,
    _pad: u32,
    key: u64,
    value: u64,
    flags: u64,
}

fn reg(dst: u8, src: u8) -> u8 {
    dst | (src << 4)
}

fn last_errno() -> i32 {
    io::Error::last_os_error().raw_os_error().unwrap_or(-1)
}

fn bpf(cmd: u32, attr: *const c_void, size: u32) -> i64 {
    // SAFETY: raw `bpf(2)` syscall; `attr` points at a live, correctly-sized
    // `bpf_attr` prefix and `size` bounds how many bytes the kernel copies.
    unsafe { libc::syscall(libc::SYS_bpf, cmd, attr, size) }
}

/// A loaded XDP redirect program plus its XSKMAP. Detaches on drop.
pub struct XdpProgram {
    map: OwnedFd,
    prog: OwnedFd,
    ifindex: u32,
    // the XDP_FLAGS_* the program was attached with, so detach clears the same
    // mode (SKB vs DRV).
    attach_flags: u32,
}

impl XdpProgram {
    /// Create the XSKMAP (`max_entries` = queue count) and load the redirect
    /// program that references it. Not yet attached to any netdev.
    ///
    /// # Errors
    /// [`XdpError::BpfMapCreate`] if the map cannot be created, or
    /// [`XdpError::BpfLoad`] (carrying the verifier log) if the program is
    /// rejected.
    pub fn load(queue_count: u32) -> Result<Self, XdpError> {
        let map = create_xskmap(queue_count.max(1))?;
        let prog = load_redirect_program(map.as_raw_fd())?;
        Ok(Self {
            map,
            prog,
            ifindex: 0,
            attach_flags: XDP_FLAGS_SKB_MODE,
        })
    }

    /// The XSKMAP fd (for callers that want to update it directly).
    #[must_use]
    pub fn map_fd(&self) -> RawFd {
        self.map.as_raw_fd()
    }

    /// Point `xskmap[queue_id]` at a bound AF_XDP socket fd, so frames whose
    /// `rx_queue_index == queue_id` redirect into it.
    ///
    /// # Errors
    /// [`XdpError::BpfMapUpdate`] with the kernel's errno.
    pub fn update_map(&self, queue_id: u32, xsk_fd: RawFd) -> Result<(), XdpError> {
        let key = queue_id;
        let value = xsk_fd;
        let attr = MapUpdateAttr {
            map_fd: as_u32(self.map.as_raw_fd()),
            _pad: 0,
            key: ptr::from_ref(&key) as u64,
            value: ptr::from_ref(&value) as u64,
            flags: 0,
        };
        let ret = bpf(
            BPF_MAP_UPDATE_ELEM,
            ptr::from_ref(&attr).cast::<c_void>(),
            size_of::<MapUpdateAttr>() as u32,
        );
        if ret < 0 {
            return Err(XdpError::BpfMapUpdate {
                queue: queue_id,
                errno: last_errno(),
            });
        }
        Ok(())
    }

    /// Attach the program to `ifindex` in SKB (generic) mode.
    ///
    /// # Errors
    /// [`XdpError::Netlink`] / [`XdpError::BpfAttach`] with the kernel's errno.
    pub fn attach(&mut self, ifindex: u32) -> Result<(), XdpError> {
        self.attach_with_flags(ifindex, XDP_FLAGS_SKB_MODE)
    }

    /// Attach the program to `ifindex` with explicit `XDP_FLAGS_*` — e.g.
    /// [`XDP_FLAGS_DRV_MODE`] for the native driver hook that AF_XDP zerocopy
    /// requires.
    ///
    /// # Errors
    /// [`XdpError::Netlink`] / [`XdpError::BpfAttach`] with the kernel's errno.
    pub fn attach_with_flags(&mut self, ifindex: u32, flags: u32) -> Result<(), XdpError> {
        set_xdp_fd(ifindex, self.prog.as_raw_fd(), flags)?;
        self.ifindex = ifindex;
        self.attach_flags = flags;
        Ok(())
    }

    /// Detach the program (idempotent; also runs on drop).
    pub fn detach(&mut self) {
        if self.ifindex != 0 {
            let _ = set_xdp_fd(self.ifindex, -1, self.attach_flags);
            self.ifindex = 0;
        }
    }
}

impl Drop for XdpProgram {
    fn drop(&mut self) {
        self.detach();
    }
}

fn as_u32(fd: RawFd) -> u32 {
    u32::try_from(fd).unwrap_or(0)
}

fn create_xskmap(max_entries: u32) -> Result<OwnedFd, XdpError> {
    let mut name = [0u8; 16];
    let label = b"xsk_redirect";
    name[..label.len()].copy_from_slice(label);
    let attr = MapCreateAttr {
        map_type: BPF_MAP_TYPE_XSKMAP,
        key_size: 4,
        value_size: 4,
        max_entries,
        map_flags: 0,
        inner_map_fd: 0,
        numa_node: 0,
        map_name: name,
        map_ifindex: 0,
    };
    let ret = bpf(
        BPF_MAP_CREATE,
        ptr::from_ref(&attr).cast::<c_void>(),
        size_of::<MapCreateAttr>() as u32,
    );
    if ret < 0 {
        return Err(XdpError::BpfMapCreate(last_errno()));
    }
    // SAFETY: `ret` is a valid, just-created fd this call uniquely owns.
    Ok(unsafe { OwnedFd::from_raw_fd(ret as RawFd) })
}

// r2 = ctx->rx_queue_index; r1 = xskmap; r3 = XDP_PASS; call redirect_map; exit
fn redirect_program(map_fd: RawFd) -> [BpfInsn; 6] {
    [
        BpfInsn {
            code: BPF_LDX_MEM_W,
            regs: reg(2, 1),
            off: XDP_MD_RX_QUEUE_INDEX,
            imm: 0,
        },
        BpfInsn {
            code: BPF_LD_IMM_DW,
            regs: reg(1, BPF_PSEUDO_MAP_FD),
            off: 0,
            imm: map_fd,
        },
        BpfInsn {
            code: 0,
            regs: 0,
            off: 0,
            imm: 0,
        },
        BpfInsn {
            code: BPF_ALU64_MOV_K,
            regs: reg(3, 0),
            off: 0,
            imm: 2,
        },
        BpfInsn {
            code: BPF_JMP_CALL,
            regs: 0,
            off: 0,
            imm: BPF_FUNC_REDIRECT_MAP,
        },
        BpfInsn {
            code: BPF_JMP_EXIT,
            regs: 0,
            off: 0,
            imm: 0,
        },
    ]
}

fn load_redirect_program(map_fd: RawFd) -> Result<OwnedFd, XdpError> {
    let insns = redirect_program(map_fd);
    let license = b"GPL\0";
    let mut log = vec![0u8; VERIFIER_LOG_SIZE];
    let attr = ProgLoadAttr {
        prog_type: BPF_PROG_TYPE_XDP,
        insn_cnt: u32::try_from(insns.len()).unwrap_or(0),
        insns: insns.as_ptr() as u64,
        license: license.as_ptr() as u64,
        log_level: 1,
        log_size: u32::try_from(log.len()).unwrap_or(0),
        log_buf: log.as_mut_ptr() as u64,
        kern_version: 0,
        prog_flags: 0,
        prog_name: [0u8; 16],
        prog_ifindex: 0,
        expected_attach_type: 0,
    };
    let ret = bpf(
        BPF_PROG_LOAD,
        ptr::from_ref(&attr).cast::<c_void>(),
        size_of::<ProgLoadAttr>() as u32,
    );
    if ret < 0 {
        let errno = last_errno();
        let end = log.iter().position(|byte| *byte == 0).unwrap_or(0);
        let message = String::from_utf8_lossy(&log[..end]).into_owned();
        return Err(XdpError::BpfLoad {
            errno,
            log: message,
        });
    }
    // SAFETY: `ret` is a valid, just-created fd this call uniquely owns.
    Ok(unsafe { OwnedFd::from_raw_fd(ret as RawFd) })
}

// build the RTM_SETLINK request that sets (or clears, fd=-1) the netdev's XDP
// program, then read the ACK and surface any nlmsgerr.
fn set_xdp_fd(ifindex: u32, prog_fd: i32, flags: u32) -> Result<(), XdpError> {
    // SAFETY: plain integer args to socket(2); a failure is checked below.
    let raw = unsafe {
        libc::socket(
            libc::AF_NETLINK,
            libc::SOCK_RAW | libc::SOCK_CLOEXEC,
            libc::NETLINK_ROUTE,
        )
    };
    if raw < 0 {
        return Err(XdpError::Netlink(last_errno()));
    }
    // SAFETY: `raw` is a valid, just-created fd this scope uniquely owns; the
    // OwnedFd closes it on any early return below.
    let socket = unsafe { OwnedFd::from_raw_fd(raw) };

    let index = i32::try_from(ifindex).unwrap_or(-1);
    let mut message: Vec<u8> = Vec::with_capacity(64);
    message.extend_from_slice(&0u32.to_ne_bytes()); // nlmsg_len (patched below)
    message.extend_from_slice(&RTM_SETLINK.to_ne_bytes());
    message.extend_from_slice(&(NLM_F_REQUEST | NLM_F_ACK).to_ne_bytes());
    message.extend_from_slice(&1u32.to_ne_bytes()); // seq
    message.extend_from_slice(&0u32.to_ne_bytes()); // pid
    message.push(libc::AF_UNSPEC as u8); // ifi_family
    message.push(0); // pad
    message.extend_from_slice(&0u16.to_ne_bytes()); // ifi_type
    message.extend_from_slice(&index.to_ne_bytes()); // ifi_index
    message.extend_from_slice(&0u32.to_ne_bytes()); // ifi_flags
    message.extend_from_slice(&0u32.to_ne_bytes()); // ifi_change
    message.extend_from_slice(&20u16.to_ne_bytes()); // IFLA_XDP nla_len
    message.extend_from_slice(&(IFLA_XDP | NLA_F_NESTED).to_ne_bytes());
    message.extend_from_slice(&8u16.to_ne_bytes()); // IFLA_XDP_FD nla_len
    message.extend_from_slice(&IFLA_XDP_FD.to_ne_bytes());
    message.extend_from_slice(&prog_fd.to_ne_bytes());
    message.extend_from_slice(&8u16.to_ne_bytes()); // IFLA_XDP_FLAGS nla_len
    message.extend_from_slice(&IFLA_XDP_FLAGS.to_ne_bytes());
    message.extend_from_slice(&flags.to_ne_bytes());
    let total = u32::try_from(message.len()).unwrap_or(0);
    message[..4].copy_from_slice(&total.to_ne_bytes());

    // SAFETY: sends `message.len()` bytes from a live buffer to the kernel
    // (default netlink peer, pid 0); return value checked.
    let sent = unsafe {
        libc::send(
            socket.as_raw_fd(),
            message.as_ptr().cast::<c_void>(),
            message.len(),
            0,
        )
    };
    if sent < 0 {
        return Err(XdpError::BpfAttach {
            ifindex,
            errno: last_errno(),
        });
    }

    let mut response = [0u8; 128];
    // SAFETY: reads up to `response.len()` bytes into a live, owned buffer.
    let received = unsafe {
        libc::recv(
            socket.as_raw_fd(),
            response.as_mut_ptr().cast::<c_void>(),
            response.len(),
            0,
        )
    };
    if received < 0 {
        return Err(XdpError::BpfAttach {
            ifindex,
            errno: last_errno(),
        });
    }
    let length = received as usize;
    if length >= 20 {
        let nlmsg_type = u16::from_ne_bytes([response[4], response[5]]);
        if nlmsg_type == NLMSG_ERROR {
            let error =
                i32::from_ne_bytes([response[16], response[17], response[18], response[19]]);
            if error != 0 {
                return Err(XdpError::BpfAttach {
                    ifindex,
                    errno: -error,
                });
            }
        }
    }
    Ok(())
}
