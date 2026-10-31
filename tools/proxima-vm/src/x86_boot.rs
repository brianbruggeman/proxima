//! x86 Linux 64-bit boot protocol, sans-IO: parse a bzImage setup header and
//! fill the boot_params zero page into a caller-owned buffer.
//!
//! Field offsets are `struct setup_header` in the kernel's
//! `arch/x86/include/uapi/asm/bootparam.h` (header at `boot_params + 0x1f1`,
//! identical to the bzImage file offsets) and the table in
//! `Documentation/arch/x86/boot.rst`. Core only: no alloc, no IO, every buffer
//! caller-owned (tier 3).

use core::fmt;

pub const BOOT_PARAMS_SIZE: usize = 4096;

const HEADER_START: usize = 0x1f1;
const HEADER_MAGIC_OFFSET: usize = 0x202;
const HEADER_MAGIC: [u8; 4] = *b"HdrS";
const BOOT_FLAG_OFFSET: usize = 0x1fe;
const BOOT_FLAG: u16 = 0xaa55;
const SETUP_SECTS_OFFSET: usize = 0x1f1;
const VERSION_OFFSET: usize = 0x206;
const KERNEL_VERSION_OFFSET: usize = 0x20e;
const TYPE_OF_LOADER_OFFSET: usize = 0x210;
const LOADFLAGS_OFFSET: usize = 0x211;
const CODE32_START_OFFSET: usize = 0x214;
const RAMDISK_IMAGE_OFFSET: usize = 0x218;
const RAMDISK_SIZE_OFFSET: usize = 0x21c;
const CMD_LINE_PTR_OFFSET: usize = 0x228;
const INITRD_ADDR_MAX_OFFSET: usize = 0x22c;
const KERNEL_ALIGNMENT_OFFSET: usize = 0x230;
const RELOCATABLE_OFFSET: usize = 0x234;
const XLOADFLAGS_OFFSET: usize = 0x236;
const CMDLINE_SIZE_OFFSET: usize = 0x238;
const INIT_SIZE_OFFSET: usize = 0x260;
const HEADER_END_JUMP_OFFSET: usize = 0x201;
const PARSED_HEADER_END: usize = 0x268;

const E820_ENTRIES_OFFSET: usize = 0x1e8;
const E820_TABLE_OFFSET: usize = 0x2d0;
const E820_ENTRY_SIZE: usize = 20;
const E820_USABLE: u32 = 1;
const E820_RESERVED: u32 = 2;

const MINIMUM_PROTOCOL: u16 = 0x020c;
const LOADED_HIGH: u8 = 1;
const XLF_KERNEL_64: u16 = 1;
const UNDEFINED_LOADER: u8 = 0xff;
const SECTOR: usize = 512;

pub const BOOT_PARAMS_ADDRESS: u64 = 0x7000;
pub const CMDLINE_ADDRESS: u64 = 0x2_0000;
pub const GDT_ADDRESS: u64 = 0x500;
pub const PAGE_TABLES_ADDRESS: u64 = 0x9000;
pub const KERNEL_LOAD_ADDRESS: u64 = 0x10_0000;
pub const LOW_RAM_END: u64 = 0x9_fc00;
pub const ISA_HOLE_END: u64 = 0x10_0000;
const ENTRY_64_OFFSET: u64 = 0x200;
const PAGE: u64 = 4096;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BootError {
    Truncated { needed: usize, have: usize },
    BadBootFlag { found: u16 },
    BadMagic { found: [u8; 4] },
    ProtocolTooOld { version: u16 },
    NotLoadedHigh { loadflags: u8 },
    NotRelocatable,
    No64BitEntry { xloadflags: u16 },
    CmdlineTooLong { length: usize, maximum: u32 },
    RamTooSmall { needed: u64, have: u64 },
    InitrdAboveLimit { end: u64, limit: u64 },
    ParamsBufferSize { length: usize },
    GdtBufferSize { length: usize },
}

impl fmt::Display for BootError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { needed, have } => {
                write!(formatter, "kernel image truncated: needs {needed} bytes, has {have}")
            }
            Self::BadBootFlag { found } => {
                write!(formatter, "not a bzImage: boot flag 0x{found:04x}, expected 0xaa55")
            }
            Self::BadMagic { found } => {
                write!(formatter, "not a bzImage: header magic {found:02x?}, expected HdrS")
            }
            Self::ProtocolTooOld { version } => write!(
                formatter,
                "boot protocol 0x{version:04x} is older than the required 0x{MINIMUM_PROTOCOL:04x}"
            ),
            Self::NotLoadedHigh { loadflags } => {
                write!(formatter, "kernel is not a bzImage: loadflags 0x{loadflags:02x} lack LOADED_HIGH")
            }
            Self::NotRelocatable => write!(formatter, "kernel is not relocatable"),
            Self::No64BitEntry { xloadflags } => {
                write!(formatter, "kernel has no 64-bit entry: xloadflags 0x{xloadflags:04x}")
            }
            Self::CmdlineTooLong { length, maximum } => {
                write!(formatter, "cmdline is {length} bytes, kernel accepts at most {maximum}")
            }
            Self::RamTooSmall { needed, have } => {
                write!(formatter, "guest ram too small: kernel plan needs {needed} bytes, has {have}")
            }
            Self::InitrdAboveLimit { end, limit } => write!(
                formatter,
                "initrd ends at 0x{end:x}, above the kernel's initrd_addr_max 0x{limit:x}"
            ),
            Self::ParamsBufferSize { length } => {
                write!(formatter, "boot_params buffer is {length} bytes, must be {BOOT_PARAMS_SIZE}")
            }
            Self::GdtBufferSize { length } => {
                write!(formatter, "gdt buffer is {length} bytes, must be {GDT_SIZE}")
            }
        }
    }
}

impl core::error::Error for BootError {}

/// The fields of a bzImage setup header a 64-bit loader acts on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetupHeader {
    pub setup_sects: u8,
    pub version: u16,
    pub kernel_version_offset: u16,
    pub loadflags: u8,
    pub code32_start: u32,
    pub initrd_addr_max: u32,
    pub kernel_alignment: u32,
    pub xloadflags: u16,
    pub cmdline_size: u32,
    pub init_size: u32,
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]])
}

impl SetupHeader {
    /// Validates and decodes the header from the start of a bzImage.
    pub fn parse(image: &[u8]) -> Result<Self, BootError> {
        if image.len() < PARSED_HEADER_END {
            return Err(BootError::Truncated { needed: PARSED_HEADER_END, have: image.len() });
        }
        let boot_flag = read_u16(image, BOOT_FLAG_OFFSET);
        if boot_flag != BOOT_FLAG {
            return Err(BootError::BadBootFlag { found: boot_flag });
        }
        let magic = [
            image[HEADER_MAGIC_OFFSET],
            image[HEADER_MAGIC_OFFSET + 1],
            image[HEADER_MAGIC_OFFSET + 2],
            image[HEADER_MAGIC_OFFSET + 3],
        ];
        if magic != HEADER_MAGIC {
            return Err(BootError::BadMagic { found: magic });
        }
        let header = Self {
            setup_sects: image[SETUP_SECTS_OFFSET],
            version: read_u16(image, VERSION_OFFSET),
            kernel_version_offset: read_u16(image, KERNEL_VERSION_OFFSET),
            loadflags: image[LOADFLAGS_OFFSET],
            code32_start: read_u32(image, CODE32_START_OFFSET),
            initrd_addr_max: read_u32(image, INITRD_ADDR_MAX_OFFSET),
            kernel_alignment: read_u32(image, KERNEL_ALIGNMENT_OFFSET),
            xloadflags: read_u16(image, XLOADFLAGS_OFFSET),
            cmdline_size: read_u32(image, CMDLINE_SIZE_OFFSET),
            init_size: read_u32(image, INIT_SIZE_OFFSET),
        };
        header.validate(image)?;
        Ok(header)
    }

    fn validate(&self, image: &[u8]) -> Result<(), BootError> {
        if self.version < MINIMUM_PROTOCOL {
            return Err(BootError::ProtocolTooOld { version: self.version });
        }
        if self.loadflags & LOADED_HIGH == 0 {
            return Err(BootError::NotLoadedHigh { loadflags: self.loadflags });
        }
        if image[RELOCATABLE_OFFSET] == 0 {
            return Err(BootError::NotRelocatable);
        }
        if self.xloadflags & XLF_KERNEL_64 == 0 {
            return Err(BootError::No64BitEntry { xloadflags: self.xloadflags });
        }
        Ok(())
    }

    /// File offset of the protected-mode kernel; `setup_sects == 0` means 4.
    #[must_use]
    pub fn protected_mode_offset(&self) -> usize {
        let sectors = if self.setup_sects == 0 { 4 } else { usize::from(self.setup_sects) };
        (sectors + 1) * SECTOR
    }

    /// The kernel's own version string, a NUL-terminated string in the setup
    /// code at `0x200 + kernel_version`.
    #[must_use]
    pub fn version_string<'image>(&self, image: &'image [u8]) -> Option<&'image [u8]> {
        let start = 0x200 + usize::from(self.kernel_version_offset);
        let tail = image.get(start..)?;
        let length = tail.iter().position(|&byte| byte == 0)?;
        Some(&tail[..length])
    }
}

/// Where every piece of the boot lands in guest-physical memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BootPlan {
    pub ram_bytes: u64,
    pub kernel_address: u64,
    pub kernel_offset: usize,
    pub kernel_length: u64,
    pub entry: u64,
    pub boot_params_address: u64,
    pub cmdline_address: u64,
    pub cmdline_length: u32,
    pub initrd_address: u64,
    pub initrd_length: u64,
}

fn align_down(value: u64, alignment: u64) -> u64 {
    value - value % alignment
}

impl BootPlan {
    /// Lays out kernel at 1 MiB, initrd at the top of RAM, and checks every
    /// constraint the header states against `ram_bytes`.
    pub fn new(
        header: &SetupHeader,
        image_length: usize,
        ram_bytes: u64,
        cmdline_length: usize,
        initrd_length: usize,
    ) -> Result<Self, BootError> {
        let kernel_offset = header.protected_mode_offset();
        let kernel_length = image_length.checked_sub(kernel_offset).ok_or(BootError::Truncated {
            needed: kernel_offset,
            have: image_length,
        })? as u64;
        if cmdline_length as u64 >= u64::from(header.cmdline_size) {
            return Err(BootError::CmdlineTooLong { length: cmdline_length, maximum: header.cmdline_size });
        }
        let initrd_end = ram_bytes.min(u64::from(header.initrd_addr_max) + 1);
        let initrd_address = align_down(initrd_end.saturating_sub(initrd_length as u64), PAGE);
        let needed = KERNEL_LOAD_ADDRESS + kernel_length.max(u64::from(header.init_size));
        if initrd_address < needed || ram_bytes <= ISA_HOLE_END {
            return Err(BootError::RamTooSmall { needed: needed + initrd_length as u64, have: ram_bytes });
        }
        Ok(Self {
            ram_bytes,
            kernel_address: KERNEL_LOAD_ADDRESS,
            kernel_offset,
            kernel_length,
            entry: KERNEL_LOAD_ADDRESS + ENTRY_64_OFFSET,
            boot_params_address: BOOT_PARAMS_ADDRESS,
            cmdline_address: CMDLINE_ADDRESS,
            cmdline_length: cmdline_length as u32,
            initrd_address,
            initrd_length: initrd_length as u64,
        })
    }
}

fn write_e820(params: &mut [u8], index: usize, address: u64, size: u64, kind: u32) {
    let start = E820_TABLE_OFFSET + index * E820_ENTRY_SIZE;
    params[start..start + 8].copy_from_slice(&address.to_le_bytes());
    params[start + 8..start + 16].copy_from_slice(&size.to_le_bytes());
    params[start + 16..start + 20].copy_from_slice(&kind.to_le_bytes());
}

/// Fills the zero page: copies the image's setup header, patches the loader
/// fields, and writes the e820 map (usable low RAM, reserved ISA hole, usable
/// extended RAM).
pub fn fill_boot_params(image: &[u8], plan: &BootPlan, params: &mut [u8]) -> Result<(), BootError> {
    if params.len() != BOOT_PARAMS_SIZE {
        return Err(BootError::ParamsBufferSize { length: params.len() });
    }
    let header_end = 0x202 + usize::from(image[HEADER_END_JUMP_OFFSET]);
    if image.len() < header_end {
        return Err(BootError::Truncated { needed: header_end, have: image.len() });
    }
    params.fill(0);
    params[HEADER_START..header_end].copy_from_slice(&image[HEADER_START..header_end]);
    params[TYPE_OF_LOADER_OFFSET] = UNDEFINED_LOADER;
    params[CMD_LINE_PTR_OFFSET..CMD_LINE_PTR_OFFSET + 4]
        .copy_from_slice(&(plan.cmdline_address as u32).to_le_bytes());
    params[RAMDISK_IMAGE_OFFSET..RAMDISK_IMAGE_OFFSET + 4]
        .copy_from_slice(&(plan.initrd_address as u32).to_le_bytes());
    params[RAMDISK_SIZE_OFFSET..RAMDISK_SIZE_OFFSET + 4]
        .copy_from_slice(&(plan.initrd_length as u32).to_le_bytes());
    params[E820_ENTRIES_OFFSET] = 3;
    write_e820(params, 0, 0, LOW_RAM_END, E820_USABLE);
    write_e820(params, 1, LOW_RAM_END, ISA_HOLE_END - LOW_RAM_END, E820_RESERVED);
    write_e820(params, 2, ISA_HOLE_END, plan.ram_bytes - ISA_HOLE_END, E820_USABLE);
    Ok(())
}

pub const GDT_ENTRIES: usize = 4;
pub const GDT_SIZE: usize = GDT_ENTRIES * 8;
pub const BOOT_CODE_SELECTOR: u16 = 0x10;
pub const BOOT_DATA_SELECTOR: u16 = 0x18;

const BOOT_GDT: [u64; GDT_ENTRIES] = [
    0,
    0x00cf_9a00_0000_ffff,
    0x00af_9a00_0000_ffff,
    0x00cf_9200_0000_ffff,
];

/// Writes the boot-protocol GDT (`Documentation/arch/x86/boot.rst`,
/// "64-bit Boot Protocol"): a null descriptor, a 32-bit code segment, the
/// 64-bit `__BOOT_CS` at selector [`BOOT_CODE_SELECTOR`] and `__BOOT_DS` at
/// [`BOOT_DATA_SELECTOR`], all flat with base 0 and 4 GiB granular limits.
pub fn write_boot_gdt(out: &mut [u8]) -> Result<(), BootError> {
    if out.len() != GDT_SIZE {
        return Err(BootError::GdtBufferSize { length: out.len() });
    }
    let (slots, _) = out.as_chunks_mut::<8>();
    for (slot, descriptor) in slots.iter_mut().zip(BOOT_GDT) {
        *slot = descriptor.to_le_bytes();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    const REAL_SETUP: &[u8] = include_bytes!("../tests/fixtures/linux_7_2_11_arch1_bzimage_setup.bin");
    const CMDLINE: &str = "console=ttyS0 reboot=k panic=-1";

    fn header() -> SetupHeader {
        SetupHeader::parse(REAL_SETUP).expect("the real arch 7.2.11 setup header must parse")
    }

    #[test]
    fn real_arch_kernel_header_decodes_to_its_documented_fields() {
        let header = header();

        assert_eq!(header.version, 0x020f, "protocol 2.15, from `file /boot/vmlinuz-linux`");
        assert_eq!(header.setup_sects, 39);
        assert_eq!(header.protected_mode_offset(), 40 * 512);
        assert_eq!(header.loadflags & LOADED_HIGH, LOADED_HIGH);
        assert_eq!(header.xloadflags & XLF_KERNEL_64, XLF_KERNEL_64);
        assert_eq!(header.code32_start, 0x10_0000);
        assert_eq!(header.cmdline_size, 0x7ff);
    }

    #[test]
    fn real_header_names_the_kernel_version() {
        let version = header().version_string(REAL_SETUP).expect("version string is in the setup code");

        assert!(version.starts_with(b"7.2.11"), "got {:?}", core::str::from_utf8(version));
    }

    #[test]
    fn plan_places_entry_at_load_address_plus_0x200() {
        let plan = BootPlan::new(&header(), 17_244_672, 256 << 20, CMDLINE.len(), 4 << 20).unwrap();

        assert_eq!(plan.kernel_address, 0x10_0000);
        assert_eq!(plan.entry, 0x10_0200);
        assert_eq!(plan.kernel_length, 17_244_672 - 40 * 512);
        assert_eq!(plan.initrd_address, (256 << 20) - (4 << 20));
    }

    #[test]
    fn boot_params_carry_header_loader_cmdline_initrd_and_e820() {
        let plan = BootPlan::new(&header(), 17_244_672, 256 << 20, CMDLINE.len(), 4 << 20).unwrap();
        let mut params = [0_u8; BOOT_PARAMS_SIZE];

        fill_boot_params(REAL_SETUP, &plan, &mut params).unwrap();

        assert_eq!(&params[0x202..0x206], b"HdrS");
        assert_eq!(params[TYPE_OF_LOADER_OFFSET], 0xff);
        assert_eq!(read_u32(&params, CMD_LINE_PTR_OFFSET), 0x2_0000);
        assert_eq!(read_u32(&params, RAMDISK_IMAGE_OFFSET), 252 << 20);
        assert_eq!(read_u32(&params, RAMDISK_SIZE_OFFSET), 4 << 20);
        assert_eq!(params[E820_ENTRIES_OFFSET], 3);
        let last = E820_TABLE_OFFSET + 2 * E820_ENTRY_SIZE;
        assert_eq!(u64::from_le_bytes(params[last..last + 8].try_into().unwrap()), 0x10_0000);
        assert_eq!(u64::from_le_bytes(params[last + 8..last + 16].try_into().unwrap()), (256 << 20) - 0x10_0000);
        assert_eq!(read_u32(&params, last + 16), E820_USABLE);
    }

    #[test]
    fn non_bzimage_bytes_are_rejected_with_the_boot_flag_named() {
        let garbage = [0x7f_u8; 4096];

        let error = SetupHeader::parse(&garbage).expect_err("garbage is not a bzImage");

        assert_eq!(error.to_string(), "not a bzImage: boot flag 0x7f7f, expected 0xaa55");
    }

    #[test]
    fn wrong_magic_with_valid_boot_flag_is_rejected_naming_hdrs() {
        let mut image = REAL_SETUP.to_vec();
        image[0x202..0x206].copy_from_slice(b"NOPE");

        let error = SetupHeader::parse(&image).expect_err("magic must be HdrS");

        assert!(error.to_string().contains("expected HdrS"), "{error}");
    }

    #[test]
    fn short_image_reports_how_many_bytes_it_needed() {
        let error = SetupHeader::parse(&REAL_SETUP[..64]).expect_err("64 bytes cannot hold a header");

        assert_eq!(error, BootError::Truncated { needed: PARSED_HEADER_END, have: 64 });
    }

    #[test]
    fn old_protocol_is_rejected() {
        let mut image = REAL_SETUP.to_vec();
        image[VERSION_OFFSET..VERSION_OFFSET + 2].copy_from_slice(&0x0206_u16.to_le_bytes());

        let error = SetupHeader::parse(&image).expect_err("protocol 2.06 predates the 64-bit entry");

        assert_eq!(error, BootError::ProtocolTooOld { version: 0x0206 });
    }

    #[test]
    fn plan_rejects_ram_that_cannot_hold_kernel_and_initrd() {
        let error = BootPlan::new(&header(), 17_244_672, 16 << 20, CMDLINE.len(), 4 << 20)
            .expect_err("16 MiB cannot hold a 17 MB kernel");

        assert!(matches!(error, BootError::RamTooSmall { .. }), "{error}");
    }

    #[test]
    fn plan_rejects_cmdline_longer_than_the_kernel_accepts() {
        let error = BootPlan::new(&header(), 17_244_672, 256 << 20, 4096, 0).expect_err("cmdline too long");

        assert_eq!(error, BootError::CmdlineTooLong { length: 4096, maximum: 0x7ff });
    }

    fn descriptor(gdt: &[u8], selector: u16) -> u64 {
        let offset = usize::from(selector);
        u64::from_le_bytes(gdt[offset..offset + 8].try_into().unwrap())
    }

    #[test]
    fn the_boot_gdt_places_the_long_mode_code_and_data_segments_at_the_boot_selectors() {
        let mut gdt = [0xaa_u8; GDT_SIZE];

        write_boot_gdt(&mut gdt).expect("32 bytes is the gdt");

        assert_eq!(descriptor(&gdt, 0), 0, "null descriptor");
        let code = descriptor(&gdt, BOOT_CODE_SELECTOR);
        let data = descriptor(&gdt, BOOT_DATA_SELECTOR);
        assert_eq!(code >> 53 & 1, 1, "__BOOT_CS has the L (64-bit code) bit");
        assert_eq!(code >> 40 & 0xff, 0x9a, "present, ring 0, code, execute/read");
        assert_eq!(data >> 40 & 0xff, 0x92, "present, ring 0, data, read/write");
        assert_eq!(code >> 55 & 1, 1, "4 KiB granular limit");
        assert_eq!((code & 0xffff, data & 0xffff), (0xffff, 0xffff));
    }

    #[test]
    fn a_gdt_buffer_of_the_wrong_size_is_rejected() {
        let error = write_boot_gdt(&mut [0_u8; 31]).expect_err("31 bytes cannot hold four descriptors");

        assert_eq!(error, BootError::GdtBufferSize { length: 31 });
    }
}
