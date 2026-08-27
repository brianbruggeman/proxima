//! ELF64 program-header loader: validates `PT_LOAD` segments — bounds, size
//! consistency, address-space overflow, alignment congruence, overlap, and
//! W^X — directly against a borrowed byte buffer. No allocation, no host
//! memory operation.
//!
//! Mirrors the borrowed-view codec shape every other proxima wire parser
//! uses (`proxima_protocols::nvme::command::SubmissionEntry`,
//! `proxima_protocols::quic::packet::header::Header`,
//! `proxima_protocols::dns::frame_codec`): a free function and POD views
//! over `&[u8]`, never a `Pipe` — an ELF image is a byte buffer to decode,
//! not a stream to transform. [`parse_elf`] itself is a driver loop over
//! an explicit discriminated-enum state machine ([`Cursor`]/[`Step`]) per
//! `AGENTS.md` principle 11 — see [`Cursor`]'s doc comment for the stage
//! shape and why "consumed length" is per-step rather than cumulative for
//! a whole-buffer format like this one.
//!
//! # Tier
//!
//! Tier-3 (bare `no_std + no_alloc`). The only collection is
//! `arrayvec::ArrayVec`, and its capacity is a caller-chosen const generic
//! — never a hidden magic number (per `slot-0/AGENTS.md` principle 12,
//! `MAX_SEGMENTS` is data the caller supplies, not a constant this module
//! bakes in).
//!
//! # What this module does not do
//!
//! It never maps memory, never copies bytes into guest RAM, and never
//! touches a syscall. The [`Segment`]s [`parse_elf`] returns are borrowed
//! views; the tier-2 loader (`proxima-vm`'s host-memory component, a later
//! milestone step) copies them in with `mmap` / `hv_vm_map` /
//! `KVM_SET_USER_MEMORY_REGION`.
//!
//! # Reference
//!
//! ELF64 field layout and the Program Header Table's `p_vaddr` /
//! `p_offset` congruence rule follow the System V ABI gABI, chapter 5
//! ("Program Header"): <https://www.sco.com/developers/gabi/latest/ch5.pheader.html>.
//! Segment permission interpretation follows the gABI's "Exact" flag
//! table — a mapped segment's protection is exactly its `p_flags`, never a
//! widened approximation (this is where this loader diverges from QEMU's
//! own `load_elf`, which places `PT_LOAD` content but performs no
//! host-side protection enforcement at all — a bare `#![no_std]` guest has
//! no stage-1 page tables of its own to make that distinction, so the host
//! must, unlike QEMU's usual guests).

use arrayvec::ArrayVec;
// NVMe queue entries and ELF program headers are both host-DRAM,
// little-endian structures (the opposite of a network codec's big-endian
// wire) — reused rather than re-hand-rolling the identical trio
// (`proxima-protocols/src/nvme/raw.rs`, exposed `pub` for exactly this).
use proxima_protocols::nvme::raw::{read_u16, read_u32, read_u64};

const ELF_MAGIC: [u8; 4] = [0x7f, b'E', b'L', b'F'];
const ELF64_HEADER_LEN: usize = 64;
const PROGRAM_HEADER_LEN: usize = 56;

const EI_CLASS: usize = 4;
const EI_DATA: usize = 5;
const ELFCLASS64: u8 = 2;
const ELFDATA2LSB: u8 = 1;

const ET_EXEC: u16 = 2;

const PT_LOAD: u32 = 1;

const PF_EXECUTE: u32 = 1;
const PF_WRITE: u32 = 2;
const PF_READ: u32 = 4;

/// Why [`parse_elf`] rejected an image. Every variant names the exact field
/// that failed, so a caller can log the diagnosis without re-deriving it.
///
/// `header_index` fields count over the **whole** program-header table (as
/// `readelf -l` numbers "Program Headers:"), not just accepted `PT_LOAD`
/// entries — a non-`PT_LOAD` header (`PT_GNU_STACK`, `PT_NOTE`, …) still
/// occupies a slot in that numbering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LoaderError {
    /// Buffer shorter than a fixed-size structure needed it to be.
    Truncated { need: usize, got: usize },
    /// `e_ident[0..4]` was not `\x7fELF`.
    BadMagic,
    /// `e_ident[EI_CLASS]` was not `ELFCLASS64` — only 64-bit images load.
    UnsupportedClass { class: u8 },
    /// `e_ident[EI_DATA]` was not `ELFDATA2LSB` — only little-endian images
    /// load (both `aarch64` and `x86_64` targets are little-endian).
    UnsupportedEndianness { data: u8 },
    /// `e_type` was not `ET_EXEC`. This loader maps every segment at its
    /// literal `p_vaddr`, which is only correct for a non-relocatable
    /// executable; `ET_DYN`/`ET_REL`/`ET_CORE` need relocation processing
    /// this loader does not perform.
    UnsupportedType { elf_type: u16 },
    /// `e_phentsize` was not 56 (`size_of::<Elf64_Phdr>()`).
    BadProgramHeaderEntrySize { entry_size: u16 },
    /// The program header table itself (`e_phoff` for `e_phnum` entries of
    /// `e_phentsize` bytes) reaches past the end of the image.
    ProgramHeaderTableOutOfRange {
        offset: u64,
        len: usize,
        image_len: usize,
    },
    /// A `PT_LOAD` entry's `[p_offset, p_offset + p_filesz)` file window
    /// reaches past the end of the image.
    SegmentOutOfRange {
        header_index: usize,
        offset: u64,
        file_size: u64,
        image_len: usize,
    },
    /// A `PT_LOAD` entry's `p_filesz` exceeds its `p_memsz` — the file
    /// content would not fit in the memory region reserved for it.
    SegmentSizeInverted {
        header_index: usize,
        file_size: u64,
        memory_size: u64,
    },
    /// `p_vaddr + p_memsz` overflows the 64-bit address space.
    SegmentAddressOverflow {
        header_index: usize,
        virtual_address: u64,
        memory_size: u64,
    },
    /// `p_align` was neither 0 nor a power of two, or `p_vaddr` and
    /// `p_offset` were not congruent modulo `p_align` (gABI chapter 5,
    /// "Program Header" — the congruence rule every ELF loader depends on
    /// to place a segment without re-copying it byte-by-byte).
    SegmentMisaligned {
        header_index: usize,
        virtual_address: u64,
        offset: u64,
        align: u64,
    },
    /// A `PT_LOAD` entry carries both `PF_W` and `PF_X` — writable and
    /// executable at once, which this loader refuses to map (W^X).
    SegmentWriteAndExecute {
        header_index: usize,
        virtual_address: u64,
    },
    /// Two accepted `PT_LOAD` entries' `[p_vaddr, p_vaddr + p_memsz)`
    /// ranges intersect.
    SegmentOverlap {
        header_index: usize,
        other_header_index: usize,
    },
    /// More `PT_LOAD` entries were present than the caller's `MAX_SEGMENTS`
    /// capacity allows.
    TooManySegments { capacity: usize },
    /// `e_entry` does not fall inside any accepted, executable `PT_LOAD`
    /// segment — the loader would be asked to jump into unmapped memory,
    /// or into memory it never marked executable.
    EntryPointOutOfRange { entry: u64 },
}

impl core::fmt::Display for LoaderError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated { need, got } => {
                write!(formatter, "truncated: need {need} bytes, got {got}")
            }
            Self::BadMagic => write!(formatter, "bad ELF magic"),
            Self::UnsupportedClass { class } => {
                write!(formatter, "unsupported ELF class {class}, need ELFCLASS64")
            }
            Self::UnsupportedEndianness { data } => write!(
                formatter,
                "unsupported ELF endianness {data}, need ELFDATA2LSB"
            ),
            Self::UnsupportedType { elf_type } => {
                write!(formatter, "unsupported ELF type {elf_type}, need ET_EXEC")
            }
            Self::BadProgramHeaderEntrySize { entry_size } => write!(
                formatter,
                "bad program header entry size {entry_size}, need 56"
            ),
            Self::ProgramHeaderTableOutOfRange {
                offset,
                len,
                image_len,
            } => write!(
                formatter,
                "program header table at offset {offset} len {len} exceeds image length {image_len}"
            ),
            Self::SegmentOutOfRange {
                header_index,
                offset,
                file_size,
                image_len,
            } => write!(
                formatter,
                "segment {header_index} file window offset {offset} size {file_size} exceeds image length {image_len}"
            ),
            Self::SegmentSizeInverted {
                header_index,
                file_size,
                memory_size,
            } => write!(
                formatter,
                "segment {header_index} file size {file_size} exceeds memory size {memory_size}"
            ),
            Self::SegmentAddressOverflow {
                header_index,
                virtual_address,
                memory_size,
            } => write!(
                formatter,
                "segment {header_index} virtual address {virtual_address:#x} + memory size {memory_size:#x} overflows"
            ),
            Self::SegmentMisaligned {
                header_index,
                virtual_address,
                offset,
                align,
            } => write!(
                formatter,
                "segment {header_index} virtual address {virtual_address:#x} offset {offset:#x} not congruent modulo align {align:#x}"
            ),
            Self::SegmentWriteAndExecute {
                header_index,
                virtual_address,
            } => write!(
                formatter,
                "segment {header_index} at {virtual_address:#x} is writable and executable"
            ),
            Self::SegmentOverlap {
                header_index,
                other_header_index,
            } => write!(
                formatter,
                "segment {header_index} overlaps segment {other_header_index}"
            ),
            Self::TooManySegments { capacity } => {
                write!(formatter, "more PT_LOAD segments than capacity {capacity}")
            }
            Self::EntryPointOutOfRange { entry } => write!(
                formatter,
                "entry point {entry:#x} is not inside any executable segment"
            ),
        }
    }
}

impl core::error::Error for LoaderError {}

/// Borrowed view over one accepted `PT_LOAD` segment. `data()` is the
/// file-backed content only; a segment whose `memory_size()` exceeds
/// `data().len()` needs its trailing bytes zero-filled by the caller
/// (BSS-style) — this view never materializes them.
#[derive(Debug, Clone, Copy)]
pub struct Segment<'a> {
    virtual_address: u64,
    memory_size: u64,
    readable: bool,
    writable: bool,
    executable: bool,
    data: &'a [u8],
}

impl<'a> Segment<'a> {
    /// The guest-virtual address the tier-2 loader maps `data()` at.
    #[must_use]
    pub fn virtual_address(&self) -> u64 {
        self.virtual_address
    }

    /// Total mapped size, including any zero-fill past `data().len()`.
    #[must_use]
    pub fn memory_size(&self) -> u64 {
        self.memory_size
    }

    #[must_use]
    pub fn is_readable(&self) -> bool {
        self.readable
    }

    #[must_use]
    pub fn is_writable(&self) -> bool {
        self.writable
    }

    #[must_use]
    pub fn is_executable(&self) -> bool {
        self.executable
    }

    /// File-backed content, borrowed from the image `parse_elf` was given.
    #[must_use]
    pub fn data(&self) -> &'a [u8] {
        self.data
    }
}

/// Explicit parse progress through the gABI's decode stages, one variant per
/// stage — the house sans-IO shape (principle 11): a discriminated enum
/// instead of a linear function threading loop-carried state through local
/// variables. Every legal transition is a match arm in [`Cursor::advance`];
/// there is no runtime "which stage am I in" flag to get out of sync with
/// the data.
///
/// ELF is a whole-buffer format, not a byte stream — a `PT_LOAD` entry's
/// file window is an absolute offset anywhere in the image, not the next N
/// bytes after a read cursor. So unlike a streaming codec's single
/// monotonic cursor, each [`Step::Advance`] carries the byte span **that
/// step itself** validated, not a cumulative position; [`Cursor::Header`]
/// validates the fixed 64-byte ELF64 header, each
/// [`Cursor::ProgramHeaderTable`] step validates one fixed 56-byte program
/// header entry (by absolute offset into the table), and
/// [`Cursor::EntryPointCheck`] validates nothing new — it is a pure
/// transition consuming zero bytes.
enum Cursor<'a, const MAX_SEGMENTS: usize> {
    /// Nothing validated yet; `image` is exactly what the caller handed in.
    Header,
    /// The ELF64 header is valid. Walking `header_count` program-header
    /// entries starting at `next_index`, accumulating every `PT_LOAD` entry
    /// that passes validation into `accepted`.
    ProgramHeaderTable {
        entry: u64,
        table_offset: usize,
        header_count: u16,
        next_index: u16,
        accepted: ArrayVec<Segment<'a>, MAX_SEGMENTS>,
    },
    /// Every program-header entry has been walked; only the "does `entry`
    /// land inside an accepted executable segment" check remains.
    EntryPointCheck {
        entry: u64,
        accepted: ArrayVec<Segment<'a>, MAX_SEGMENTS>,
    },
}

/// One [`Cursor::advance`] transition outcome.
enum Step<'a, const MAX_SEGMENTS: usize> {
    /// Move to the next [`Cursor`] state. The `usize` is the byte span this
    /// step itself validated (see [`Cursor`]'s doc comment) — diagnostic
    /// only; [`parse_elf`] does not accumulate it into a position, because
    /// ELF has no single monotonic read position to accumulate into.
    Advance(Cursor<'a, MAX_SEGMENTS>, usize),
    /// Terminal: the entry point and every accepted `PT_LOAD` segment, in
    /// program-header order.
    Done(u64, ArrayVec<Segment<'a>, MAX_SEGMENTS>),
}

impl<'a, const MAX_SEGMENTS: usize> Cursor<'a, MAX_SEGMENTS> {
    /// Validate and consume exactly this state's stage of `image`, then
    /// name the next state — never touches bytes another stage owns.
    fn advance(self, image: &'a [u8]) -> Result<Step<'a, MAX_SEGMENTS>, LoaderError> {
        match self {
            Self::Header => Self::advance_header(image),
            Self::ProgramHeaderTable {
                entry,
                table_offset,
                header_count,
                next_index,
                accepted,
            } => Self::advance_program_header_table(
                image,
                entry,
                table_offset,
                header_count,
                next_index,
                accepted,
            ),
            Self::EntryPointCheck { entry, accepted } => {
                let entry_in_range = accepted.iter().any(|segment| {
                    segment.executable
                        && entry >= segment.virtual_address
                        && entry < segment.virtual_address + segment.memory_size
                });
                if entry_in_range {
                    Ok(Step::Done(entry, accepted))
                } else {
                    Err(LoaderError::EntryPointOutOfRange { entry })
                }
            }
        }
    }

    fn advance_header(image: &'a [u8]) -> Result<Step<'a, MAX_SEGMENTS>, LoaderError> {
        if image.len() < ELF64_HEADER_LEN {
            return Err(LoaderError::Truncated {
                need: ELF64_HEADER_LEN,
                got: image.len(),
            });
        }
        if image[0..4] != ELF_MAGIC {
            return Err(LoaderError::BadMagic);
        }

        let class = image[EI_CLASS];
        if class != ELFCLASS64 {
            return Err(LoaderError::UnsupportedClass { class });
        }
        let data_encoding = image[EI_DATA];
        if data_encoding != ELFDATA2LSB {
            return Err(LoaderError::UnsupportedEndianness {
                data: data_encoding,
            });
        }

        let elf_type = read_u16(image, 16);
        if elf_type != ET_EXEC {
            return Err(LoaderError::UnsupportedType { elf_type });
        }

        let entry = read_u64(image, 24);
        let program_header_offset = read_u64(image, 32);
        let program_header_entry_size = read_u16(image, 54);
        let header_count = read_u16(image, 56);

        if program_header_entry_size as usize != PROGRAM_HEADER_LEN {
            return Err(LoaderError::BadProgramHeaderEntrySize {
                entry_size: program_header_entry_size,
            });
        }

        let table_len = u64::from(header_count) * PROGRAM_HEADER_LEN as u64;
        let table_end = program_header_offset
            .checked_add(table_len)
            .filter(|end| *end <= image.len() as u64)
            .ok_or(LoaderError::ProgramHeaderTableOutOfRange {
                offset: program_header_offset,
                len: table_len as usize,
                image_len: image.len(),
            })?;
        debug_assert!(table_end <= image.len() as u64);

        Ok(Step::Advance(
            Self::ProgramHeaderTable {
                entry,
                table_offset: program_header_offset as usize,
                header_count,
                next_index: 0,
                accepted: ArrayVec::new(),
            },
            ELF64_HEADER_LEN,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn advance_program_header_table(
        image: &'a [u8],
        entry: u64,
        table_offset: usize,
        header_count: u16,
        next_index: u16,
        mut accepted: ArrayVec<Segment<'a>, MAX_SEGMENTS>,
    ) -> Result<Step<'a, MAX_SEGMENTS>, LoaderError> {
        if next_index == header_count {
            return Ok(Step::Advance(Self::EntryPointCheck { entry, accepted }, 0));
        }

        let header_index = usize::from(next_index);
        let phdr_offset = table_offset + header_index * PROGRAM_HEADER_LEN;
        let phdr = &image[phdr_offset..phdr_offset + PROGRAM_HEADER_LEN];

        if read_u32(phdr, 0) == PT_LOAD {
            let flags = read_u32(phdr, 4);
            let file_offset = read_u64(phdr, 8);
            let virtual_address = read_u64(phdr, 16);
            let file_size = read_u64(phdr, 32);
            let memory_size = read_u64(phdr, 40);
            let align = read_u64(phdr, 48);

            let file_end = file_offset
                .checked_add(file_size)
                .filter(|end| *end <= image.len() as u64)
                .ok_or(LoaderError::SegmentOutOfRange {
                    header_index,
                    offset: file_offset,
                    file_size,
                    image_len: image.len(),
                })?;

            if file_size > memory_size {
                return Err(LoaderError::SegmentSizeInverted {
                    header_index,
                    file_size,
                    memory_size,
                });
            }

            let virtual_end = virtual_address.checked_add(memory_size).ok_or(
                LoaderError::SegmentAddressOverflow {
                    header_index,
                    virtual_address,
                    memory_size,
                },
            )?;

            let congruent = align <= 1 || virtual_address % align == file_offset % align;
            if !congruent || (align != 0 && !align.is_power_of_two()) {
                return Err(LoaderError::SegmentMisaligned {
                    header_index,
                    virtual_address,
                    offset: file_offset,
                    align,
                });
            }

            let writable = flags & PF_WRITE != 0;
            let executable = flags & PF_EXECUTE != 0;
            if writable && executable {
                return Err(LoaderError::SegmentWriteAndExecute {
                    header_index,
                    virtual_address,
                });
            }

            if let Some(overlapping) = accepted.iter().enumerate().find(|(_, existing)| {
                let existing_end = existing.virtual_address + existing.memory_size;
                virtual_address < existing_end && existing.virtual_address < virtual_end
            }) {
                return Err(LoaderError::SegmentOverlap {
                    header_index,
                    other_header_index: overlapping.0,
                });
            }

            let readable = flags & PF_READ != 0;
            let data = &image[file_offset as usize..file_end as usize];

            accepted
                .try_push(Segment {
                    virtual_address,
                    memory_size,
                    readable,
                    writable,
                    executable,
                    data,
                })
                .map_err(|_capacity_error| LoaderError::TooManySegments {
                    capacity: MAX_SEGMENTS,
                })?;
        }

        Ok(Step::Advance(
            Self::ProgramHeaderTable {
                entry,
                table_offset,
                header_count,
                next_index: next_index + 1,
                accepted,
            },
            PROGRAM_HEADER_LEN,
        ))
    }
}

/// Parse an ELF64 image's `PT_LOAD` segments, validating every segment
/// before any host memory operation touches it. `MAX_SEGMENTS` is the
/// caller-chosen capacity for the fixed-cap `ArrayVec` — never a hidden
/// magic number; a caller sizes it for the largest guest it loads (the M1
/// scratch guest at `tools/proxima-vm/guests/lambda` links three loadable
/// sections — `.text`, `.rodata`, `.data` — so `MAX_SEGMENTS = 4` covers it
/// with headroom).
///
/// Drives [`Cursor::advance`] to completion — see [`Cursor`] for the state
/// shape. Every check below is one `Cursor` transition:
///
/// - the ELF64 magic, class (`ELFCLASS64`), and endianness (`ELFDATA2LSB`)
/// - `e_type == ET_EXEC` (see [`LoaderError::UnsupportedType`])
/// - the program header table itself stays inside the image
/// - every `PT_LOAD` entry's `[p_offset, p_offset + p_filesz)` file window
///   stays inside the image
/// - `p_filesz <= p_memsz`
/// - `p_vaddr + p_memsz` does not overflow
/// - the gABI `p_vaddr` / `p_offset` congruence rule modulo `p_align`
/// - no two accepted `PT_LOAD` segments' virtual-address ranges overlap
/// - no accepted `PT_LOAD` segment carries both `PF_W` and `PF_X` (W^X)
/// - `e_entry` falls inside an accepted, executable segment
///
/// Returns the entry point and the accepted segments, in program-header
/// order. Non-`PT_LOAD` entries (`PT_GNU_STACK`, `PT_NOTE`, …) are skipped
/// entirely — they carry nothing this loader maps.
///
/// # Errors
///
/// See [`LoaderError`] for every rejection this function can return.
pub fn parse_elf<const MAX_SEGMENTS: usize>(
    image: &[u8],
) -> Result<(u64, ArrayVec<Segment<'_>, MAX_SEGMENTS>), LoaderError> {
    let mut cursor = Cursor::Header;
    loop {
        match cursor.advance(image)? {
            Step::Advance(next, _step_span) => cursor = next,
            Step::Done(entry, accepted) => return Ok((entry, accepted)),
        }
    }
}

/// The canonical ELF64 test-fixture encoder, shared by this module's own
/// tests and by sibling modules (`loader.rs`) that need a real, valid
/// image to exercise `parse_elf`'s output against — never a hand-rolled
/// struct literal standing in for one (principle 9: "no field is ever
/// hand-typed twice").
#[cfg(all(test, feature = "std"))]
pub(crate) mod test_support {
    use super::{
        EI_CLASS, EI_DATA, ELF_MAGIC, ELF64_HEADER_LEN, ELFCLASS64, ELFDATA2LSB, ET_EXEC,
        PF_EXECUTE, PF_READ, PROGRAM_HEADER_LEN, PT_LOAD,
    };

    /// One `PT_LOAD` entry's fields, for building synthetic test images.
    pub(crate) struct TestSegment {
        pub(crate) virtual_address: u64,
        pub(crate) file_offset: u64,
        pub(crate) file_size: u64,
        pub(crate) memory_size: u64,
        pub(crate) flags: u32,
        pub(crate) align: u64,
        pub(crate) content: Vec<u8>,
    }

    impl TestSegment {
        pub(crate) fn readable_executable(
            virtual_address: u64,
            file_offset: u64,
            content: &[u8],
        ) -> Self {
            Self {
                virtual_address,
                file_offset,
                file_size: content.len() as u64,
                memory_size: content.len() as u64,
                flags: PF_READ | PF_EXECUTE,
                align: 0x1000,
                content: content.to_vec(),
            }
        }
    }

    /// The canonical encoder for these tests' fixtures: a minimal, valid
    /// ELF64 `ET_EXEC` image carrying exactly the given `PT_LOAD` entries.
    /// Negative tests build one, then mutate the field under test — this is
    /// the "synthesize one with the canonical encoder" half of principle 9;
    /// no field is ever hand-typed twice.
    pub(crate) fn build_elf(entry: u64, segments: &[TestSegment]) -> Vec<u8> {
        let program_header_table_offset = ELF64_HEADER_LEN;
        let mut image =
            vec![0_u8; program_header_table_offset + segments.len() * PROGRAM_HEADER_LEN];

        image[0..4].copy_from_slice(&ELF_MAGIC);
        image[EI_CLASS] = ELFCLASS64;
        image[EI_DATA] = ELFDATA2LSB;
        image[6] = 1; // EI_VERSION
        image[16..18].copy_from_slice(&ET_EXEC.to_le_bytes());
        image[20..24].copy_from_slice(&1_u32.to_le_bytes()); // e_version
        image[24..32].copy_from_slice(&entry.to_le_bytes());
        image[32..40].copy_from_slice(&(program_header_table_offset as u64).to_le_bytes());
        image[52..54].copy_from_slice(&(ELF64_HEADER_LEN as u16).to_le_bytes());
        image[54..56].copy_from_slice(&(PROGRAM_HEADER_LEN as u16).to_le_bytes());
        image[56..58].copy_from_slice(&(segments.len() as u16).to_le_bytes());

        for (index, segment) in segments.iter().enumerate() {
            let phdr = program_header_table_offset + index * PROGRAM_HEADER_LEN;
            image[phdr..phdr + 4].copy_from_slice(&PT_LOAD.to_le_bytes());
            image[phdr + 4..phdr + 8].copy_from_slice(&segment.flags.to_le_bytes());
            image[phdr + 8..phdr + 16].copy_from_slice(&segment.file_offset.to_le_bytes());
            image[phdr + 16..phdr + 24].copy_from_slice(&segment.virtual_address.to_le_bytes());
            image[phdr + 24..phdr + 32].copy_from_slice(&segment.virtual_address.to_le_bytes());
            image[phdr + 32..phdr + 40].copy_from_slice(&segment.file_size.to_le_bytes());
            image[phdr + 40..phdr + 48].copy_from_slice(&segment.memory_size.to_le_bytes());
            image[phdr + 48..phdr + 56].copy_from_slice(&segment.align.to_le_bytes());
        }

        for segment in segments {
            let start = segment.file_offset as usize;
            let end = start + segment.content.len();
            if end > image.len() {
                image.resize(end, 0);
            }
            image[start..end].copy_from_slice(&segment.content);
        }

        image
    }

    /// A real, valid two-segment image (readable+executable `.text`,
    /// readable-only `.rodata`, non-overlapping, correctly aligned) for
    /// tests that need more than one accepted `PT_LOAD` entry —
    /// `loader.rs`'s `GuestMemory` capacity test is the first consumer.
    // `loader.rs` is not declared in `lib.rs` yet: it links against
    // `proxima_vm_map_guest_memory`/`proxima_vm_unmap_guest_memory`, which
    // `build.rs` does not compile in (ROADMAP M1 step 4, not this reshape).
    // Wiring it in produces an undefined-symbol link error, not a warning,
    // so this fixture is genuinely unreferenced until that step lands.
    #[allow(dead_code)]
    pub(crate) fn build_two_segment_elf() -> Vec<u8> {
        let text = TestSegment::readable_executable(0, 0x1000, &[0xd4, 0x20, 0x00, 0x00]);
        let rodata = TestSegment {
            virtual_address: 0x2000,
            file_offset: 0x2000,
            file_size: 4,
            memory_size: 4,
            flags: PF_READ,
            align: 0x1000,
            content: vec![0x01, 0x02, 0x03, 0x04],
        };
        build_elf(0, &[text, rodata])
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::path::{Path, PathBuf};
    use std::process::Command;

    use proptest::prelude::*;

    use super::test_support::{TestSegment, build_elf};
    use super::*;

    fn minimal_valid_elf() -> Vec<u8> {
        build_elf(
            0,
            &[TestSegment::readable_executable(
                0,
                0x1000,
                &[0xd4, 0x20, 0x00, 0x00],
            )],
        )
    }

    #[test]
    fn truncated_header_is_rejected() {
        let full_image = minimal_valid_elf();
        let image = &full_image[..ELF64_HEADER_LEN - 1];
        assert_eq!(
            parse_elf::<4>(image).unwrap_err(),
            LoaderError::Truncated {
                need: ELF64_HEADER_LEN,
                got: ELF64_HEADER_LEN - 1
            }
        );
    }

    #[test]
    fn bad_magic_is_rejected() {
        let mut image = minimal_valid_elf();
        image[0] = 0x00;
        assert_eq!(parse_elf::<4>(&image).unwrap_err(), LoaderError::BadMagic);
    }

    #[test]
    fn unsupported_class_is_rejected() {
        let mut image = minimal_valid_elf();
        image[EI_CLASS] = 1; // ELFCLASS32
        assert_eq!(
            parse_elf::<4>(&image).unwrap_err(),
            LoaderError::UnsupportedClass { class: 1 }
        );
    }

    #[test]
    fn unsupported_endianness_is_rejected() {
        let mut image = minimal_valid_elf();
        image[EI_DATA] = 2; // ELFDATA2MSB
        assert_eq!(
            parse_elf::<4>(&image).unwrap_err(),
            LoaderError::UnsupportedEndianness { data: 2 }
        );
    }

    #[test]
    fn unsupported_elf_type_is_rejected() {
        let mut image = minimal_valid_elf();
        image[16..18].copy_from_slice(&3_u16.to_le_bytes()); // ET_DYN
        assert_eq!(
            parse_elf::<4>(&image).unwrap_err(),
            LoaderError::UnsupportedType { elf_type: 3 }
        );
    }

    #[test]
    fn bad_program_header_entry_size_is_rejected() {
        let mut image = minimal_valid_elf();
        image[54..56].copy_from_slice(&32_u16.to_le_bytes());
        assert_eq!(
            parse_elf::<4>(&image).unwrap_err(),
            LoaderError::BadProgramHeaderEntrySize { entry_size: 32 }
        );
    }

    #[test]
    fn program_header_table_out_of_range_is_rejected() {
        let mut image = minimal_valid_elf();
        let image_len = image.len() as u64;
        image[32..40].copy_from_slice(&(image_len + 1).to_le_bytes());
        assert_eq!(
            parse_elf::<4>(&image).unwrap_err(),
            LoaderError::ProgramHeaderTableOutOfRange {
                offset: image_len + 1,
                len: PROGRAM_HEADER_LEN,
                image_len: image.len(),
            }
        );
    }

    #[test]
    fn segment_file_window_out_of_range_is_rejected() {
        // `build_elf` grows its buffer to fit every segment's content, so a
        // claimed file window can only end up "out of range" by shrinking
        // the buffer back down after construction — never by asking for a
        // larger `file_offset`, which the builder would just satisfy.
        let segment = TestSegment::readable_executable(0, 0x1000, &[0xd4, 0x20, 0x00, 0x00]);
        let mut image = build_elf(0, &[segment]);
        image.truncate(0x1000);
        assert_eq!(
            parse_elf::<4>(&image).unwrap_err(),
            LoaderError::SegmentOutOfRange {
                header_index: 0,
                offset: 0x1000,
                file_size: 4,
                image_len: image.len(),
            }
        );
    }

    #[test]
    fn segment_size_inverted_is_rejected() {
        let mut segment = TestSegment::readable_executable(0, 0x1000, &[0xd4, 0x20, 0x00, 0x00]);
        segment.memory_size = 2;
        let image = build_elf(0, &[segment]);
        assert_eq!(
            parse_elf::<4>(&image).unwrap_err(),
            LoaderError::SegmentSizeInverted {
                header_index: 0,
                file_size: 4,
                memory_size: 2,
            }
        );
    }

    #[test]
    fn segment_address_overflow_is_rejected() {
        let mut segment = TestSegment::readable_executable(u64::MAX - 1, 0x1000, &[0xd4]);
        segment.memory_size = 4;
        segment.file_size = 1;
        let image = build_elf(u64::MAX - 1, &[segment]);
        assert_eq!(
            parse_elf::<4>(&image).unwrap_err(),
            LoaderError::SegmentAddressOverflow {
                header_index: 0,
                virtual_address: u64::MAX - 1,
                memory_size: 4,
            }
        );
    }

    #[test]
    fn segment_misaligned_is_rejected() {
        let mut segment = TestSegment::readable_executable(0x10, 0x1000, &[0xd4, 0x20, 0x00, 0x00]);
        segment.align = 0x1000;
        let image = build_elf(0x10, &[segment]);
        assert_eq!(
            parse_elf::<4>(&image).unwrap_err(),
            LoaderError::SegmentMisaligned {
                header_index: 0,
                virtual_address: 0x10,
                offset: 0x1000,
                align: 0x1000,
            }
        );
    }

    #[test]
    fn segment_align_not_power_of_two_is_rejected() {
        // vaddr 0 and offset 0x1002 are congruent modulo 3 (both remainder
        // 0), isolating the "align must be a power of two" branch from the
        // congruence branch `segment_misaligned_is_rejected` already covers.
        let mut segment = TestSegment::readable_executable(0, 0x1002, &[0xd4, 0x20, 0x00, 0x00]);
        segment.align = 3;
        let image = build_elf(0, &[segment]);
        assert_eq!(
            parse_elf::<4>(&image).unwrap_err(),
            LoaderError::SegmentMisaligned {
                header_index: 0,
                virtual_address: 0,
                offset: 0x1002,
                align: 3,
            }
        );
    }

    #[test]
    fn segment_write_and_execute_is_rejected() {
        let mut segment = TestSegment::readable_executable(0, 0x1000, &[0xd4, 0x20, 0x00, 0x00]);
        segment.flags = PF_READ | PF_WRITE | PF_EXECUTE;
        let image = build_elf(0, &[segment]);
        assert_eq!(
            parse_elf::<4>(&image).unwrap_err(),
            LoaderError::SegmentWriteAndExecute {
                header_index: 0,
                virtual_address: 0,
            }
        );
    }

    #[test]
    fn overlapping_segments_are_rejected() {
        let first = TestSegment::readable_executable(0, 0x1000, &[0xd4, 0x20, 0x00, 0x00]);
        let mut second = TestSegment::readable_executable(0x2, 0x2000, &[0xd4, 0x20, 0x00, 0x00]);
        // align 1 means "no alignment constraint" (gABI), decoupling this
        // overlap-focused case from the congruence rule under test elsewhere.
        second.align = 1;
        let image = build_elf(0, &[first, second]);
        assert_eq!(
            parse_elf::<4>(&image).unwrap_err(),
            LoaderError::SegmentOverlap {
                header_index: 1,
                other_header_index: 0,
            }
        );
    }

    #[test]
    fn too_many_segments_is_rejected() {
        let first = TestSegment::readable_executable(0, 0x1000, &[0xd4, 0x20, 0x00, 0x00]);
        let second = TestSegment::readable_executable(0x1000, 0x2000, &[0xd4, 0x20, 0x00, 0x00]);
        let image = build_elf(0, &[first, second]);
        assert_eq!(
            parse_elf::<1>(&image).unwrap_err(),
            LoaderError::TooManySegments { capacity: 1 }
        );
    }

    #[test]
    fn entry_point_out_of_range_is_rejected() {
        let segment = TestSegment::readable_executable(0, 0x1000, &[0xd4, 0x20, 0x00, 0x00]);
        let image = build_elf(0x9000, &[segment]);
        assert_eq!(
            parse_elf::<4>(&image).unwrap_err(),
            LoaderError::EntryPointOutOfRange { entry: 0x9000 }
        );
    }

    #[test]
    fn elf_with_no_load_segments_is_rejected() {
        let image = build_elf(0, &[]);
        assert_eq!(
            parse_elf::<4>(&image).unwrap_err(),
            LoaderError::EntryPointOutOfRange { entry: 0 }
        );
    }

    #[test]
    fn valid_minimal_elf_is_accepted() {
        let image = minimal_valid_elf();
        let (entry, segments) = parse_elf::<4>(&image).expect("valid ELF parses");
        assert_eq!(entry, 0);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].virtual_address(), 0);
        assert_eq!(segments[0].memory_size(), 4);
        assert!(segments[0].is_readable());
        assert!(!segments[0].is_writable());
        assert!(segments[0].is_executable());
        assert_eq!(segments[0].data(), &[0xd4, 0x20, 0x00, 0x00]);
    }

    proptest! {
        #[test]
        fn parse_never_panics_on_arbitrary_bytes(data in prop::collection::vec(any::<u8>(), 0..512)) {
            let _ = parse_elf::<8>(&data);
        }
    }

    /// Locates (building on demand) the real bare-metal guest ELF at
    /// `tools/proxima-vm/guests/lambda`, built for `target_triple`. Mirrors
    /// `tools/proxima-cli/tests/integration/pipeline_subcommand.rs`'s
    /// `CARGO_TARGET_DIR`-then-workspace-root discovery, extended with the
    /// per-target-triple artifact subdirectory a cross build produces.
    fn guest_elf_bytes(target_triple: &str) -> Vec<u8> {
        let target_dir = std::env::var("CARGO_TARGET_DIR").map_or_else(
            |_| {
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .expect("tools dir")
                    .parent()
                    .expect("workspace root")
                    .join("target")
            },
            PathBuf::from,
        );
        let artifact = target_dir
            .join(target_triple)
            .join("debug")
            .join("proxima-vm-guest-lambda");

        if !artifact.exists() {
            let manifest_path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("guests")
                .join("lambda")
                .join("Cargo.toml");
            let status = Command::new("cargo")
                .args([
                    "build",
                    "--manifest-path",
                    manifest_path.to_str().expect("utf8 manifest path"),
                    "--target",
                    target_triple,
                ])
                .status()
                .expect("run cargo build for the guest crate");
            assert!(status.success(), "cargo build for {target_triple} failed");
        }

        std::fs::read(&artifact).expect("read built guest elf")
    }

    /// Cross-checked against `llvm-readobj --program-headers` (LLVM 20, the
    /// toolchain's own bundled `rustlib/*/bin/llvm-readobj`; `readelf` is
    /// not installed on this host, so `llvm-readobj` is the incumbent here
    /// per principle 14) run against THIS function's own debug-profile
    /// build target (`guest_elf_bytes`'s `cargo build -p
    /// proxima-vm-guest-lambda --target …`, no `--release`) — the debug
    /// profile's unstripped, unoptimized code is materially larger than a
    /// release build's, so a release-profile pin here would be wrong.
    /// Re-pinned 2026-08-26 (M6 slice 5) against a from-scratch rebuild of
    /// `guests/lambda` in this tree: adding
    /// `guests/lambda/src/virtio_net.rs`'s mmio bring-up sequence grew both
    /// segments past the slice-3 pins (1940/920).
    ///
    /// Re-pinned again 2026-08-27 (host-workspace build fix): moving
    /// `guests/lambda` out of the host `Cargo.toml`'s `members` into
    /// `exclude` (so plain `cargo build --workspace` no longer tries to
    /// compile this no_std, no_alloc aarch64-unknown-none binary for the
    /// host) makes it its own workspace root. rustc now embeds this crate's
    /// `file!()`/panic-location strings relative to its OWN manifest dir
    /// (`src/main.rs`, …) instead of the host workspace root
    /// (`tools/proxima-vm/guests/lambda/src/main.rs`, …); those shorter
    /// strings shrink the `.rodata` `PT_LOAD` segment by 92 bytes
    /// (2984 -> 2892). The code emitting this binary did not change.
    ///
    /// ```text
    /// ProgramHeader { Type: PT_LOAD, Offset: 0x10000, VirtualAddress: 0x0,    FileSize: 10324, MemSize: 10324, Flags: PF_R | PF_X }
    /// ProgramHeader { Type: PT_LOAD, Offset: 0x12858, VirtualAddress: 0x2858, FileSize: 2892, MemSize: 2892, Flags: PF_R }
    /// ProgramHeader { Type: PT_GNU_STACK, ... }
    /// ```
    ///
    /// `parse_elf` must land on the same entry point and the same two
    /// `PT_LOAD` segments (in the same order); the `GNU_STACK` entry must
    /// not appear (it is not `PT_LOAD`).
    #[test]
    fn matches_readelf_on_the_real_aarch64_guest() {
        let image = guest_elf_bytes("aarch64-unknown-none");
        let (entry, segments) = parse_elf::<4>(&image).expect("real guest ELF parses");

        assert_eq!(entry, 0x0);
        assert_eq!(segments.len(), 2);

        assert_eq!(segments[0].virtual_address(), 0x0000);
        assert_eq!(segments[0].memory_size(), 10324);
        assert_eq!(segments[0].data().len(), 10324);
        assert!(segments[0].is_readable());
        assert!(!segments[0].is_writable());
        assert!(segments[0].is_executable());

        assert_eq!(segments[1].virtual_address(), 0x2858);
        assert_eq!(segments[1].memory_size(), 2892);
        assert_eq!(segments[1].data().len(), 2892);
        assert!(segments[1].is_readable());
        assert!(!segments[1].is_writable());
        assert!(!segments[1].is_executable());
    }

    /// Same cross-check as above, for the `x86_64-unknown-none` debug-
    /// profile build. Re-pinned 2026-08-26 (M6 slice 6) alongside the
    /// aarch64 pin, same cause: adding `virtio_blk.rs` grew the guest past
    /// the slice-5 pins (5797/1100/744). Unlike the aarch64 build, this
    /// debug rebuild's LLD output carries a THIRD `PT_LOAD` — a small
    /// writable, non-executable segment (`.data.rel.ro`, covered by the
    /// `PT_GNU_RELRO` entry `llvm-readobj` also reports and this loader
    /// correctly ignores, since it is not `PT_LOAD`) that the debug
    /// profile's unoptimized codegen emits and the release profile's does
    /// not; this is a real difference in this build's actual program
    /// headers, not a copy-paste of the aarch64 pin.
    ///
    /// Re-pinned again 2026-08-27, same cause and same 92-byte `.rodata`
    /// shrink as the aarch64 pin above (see that test's doc comment): moving
    /// `guests/lambda` to its own workspace root shortens its embedded
    /// `file!()` panic-location strings. The 92-byte-smaller segment 1
    /// (1692 -> 1600) shifts segment 2's start address down by the same
    /// page-aligned delta (0x2700 -> 0x26A0); segment 2's own size (1384) is
    /// unaffected, since it holds no `file!()`-derived strings.
    ///
    /// ```text
    /// ProgramHeader { Type: PT_LOAD, Offset: 0x1000, VirtualAddress: 0x0,    FileSize: 8287, MemSize: 8287, Flags: PF_R | PF_X }
    /// ProgramHeader { Type: PT_LOAD, Offset: 0x3060, VirtualAddress: 0x2060, FileSize: 1600, MemSize: 1600, Flags: PF_R }
    /// ProgramHeader { Type: PT_LOAD, Offset: 0x36A0, VirtualAddress: 0x26A0, FileSize: 1384, MemSize: 1384, Flags: PF_R | PF_W }
    /// ProgramHeader { Type: PT_GNU_RELRO, ... }
    /// ProgramHeader { Type: PT_GNU_STACK, ... }
    /// ```
    #[test]
    fn matches_readelf_on_the_real_x86_64_guest() {
        let image = guest_elf_bytes("x86_64-unknown-none");
        let (entry, segments) = parse_elf::<4>(&image).expect("real guest ELF parses");

        assert_eq!(entry, 0x0);
        assert_eq!(segments.len(), 3);

        assert_eq!(segments[0].virtual_address(), 0x0000);
        assert_eq!(segments[0].memory_size(), 8287);
        assert_eq!(segments[0].data().len(), 8287);
        assert!(segments[0].is_readable());
        assert!(!segments[0].is_writable());
        assert!(segments[0].is_executable());

        assert_eq!(segments[1].virtual_address(), 0x2060);
        assert_eq!(segments[1].memory_size(), 1600);
        assert_eq!(segments[1].data().len(), 1600);
        assert!(segments[1].is_readable());
        assert!(!segments[1].is_writable());
        assert!(!segments[1].is_executable());

        assert_eq!(segments[2].virtual_address(), 0x26A0);
        assert_eq!(segments[2].memory_size(), 1384);
        assert_eq!(segments[2].data().len(), 1384);
        assert!(segments[2].is_readable());
        assert!(segments[2].is_writable());
        assert!(!segments[2].is_executable());
    }
}
