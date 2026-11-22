//! Write encoded functions into an arm64 Mach-O relocatable object.
//!
//! Consumes [`crate::pipeline::Encoded`] and hands it to the `object` crate's writer: one global text
//! symbol per function, named with the Mach-O C prefix (`main` -> `_main`).
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use thiserror::Error;

use ::object::macho::{
    ARM64_RELOC_PAGE21, ARM64_RELOC_PAGEOFF12, PLATFORM_MACOS, RelocationType, Version,
};
use ::object::write::{
    MachOBuildVersion, Object, Relocation as WriteRelocation, SectionId, StandardSection, Symbol,
    SymbolId, SymbolSection,
};
use ::object::{
    Architecture, BinaryFormat, Endianness, RelocationEncoding, RelocationFlags,
    RelocationKind as ObjectKind, SymbolFlags, SymbolKind, SymbolScope,
};

use crate::aarch64::{EncodeError, LowerError};
use crate::pipeline::{Encoded, Relocation, RelocationKind};

/// Why an object was not written.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ObjectError {
    #[error("function `{name}`: {source}")]
    Lower { name: String, source: LowerError },
    #[error("function `{name}`: {source}")]
    Encode { name: String, source: EncodeError },
    #[error("object writer: {0}")]
    Write(String),
    #[error("data object `{symbol}`: {reason}")]
    Data {
        symbol: String,
        reason: &'static str,
    },
}

/// A pointer-sized slot of a data object that holds the address of `symbol`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PointerSlot {
    /// byte offset of the 8-byte slot within the object's bytes
    pub offset: u64,
    /// the symbol the slot points at, a function or data object fed in any order
    pub symbol: String,
}

/// One static: a symbol, its bytes, its alignment and the pointer slots inside the bytes.
///
/// An object with no slot goes to `__TEXT,__const`; one with slots goes to `__DATA,__const`, each
/// slot an `ARM64_RELOC_UNSIGNED` relocation of length 3, the form stock rustc's LLVM output uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataObject {
    pub symbol: String,
    pub bytes: Vec<u8>,
    /// a power of two, the object's address is a multiple of it
    pub align: u64,
    pub pointers: Vec<PointerSlot>,
}

struct PendingSlot {
    section: SectionId,
    offset: u64,
    symbol: String,
}

/// Fold encoded functions into one Mach-O object, one at a time.
///
/// `minos` is the deployment target packed as `xxxx.yy.zz` nibbles (11.0 is `0x000B_0000`); the
/// caller supplies it, as rustc does from `sess.apple_deployment_target()`. `feed` appends a
/// function's code and defines its symbol, `finish` resolves the stashed relocations and writes the
/// bytes. Symbol ids are defined symbols first in feed order, then undefined ones in first-reference order.
///
/// Not a [`Pipe`](proxima_primitives::pipe::Pipe): `Pipe::call` takes `&self`, a fold into one growing
/// object needs `&mut self`, and `finish` has no input value (the `ByteStreamParser` exception in
/// `proxima-primitives/src/pipe/sans_io.rs`). The stages before it are pipes, see
/// [`crate::pipeline::backend`].
pub struct ObjectBuilder {
    object: Object<'static>,
    text: SectionId,
    symbols: BTreeMap<String, SymbolId>,
    pending: Vec<(u64, Vec<Relocation>)>,
    slots: Vec<PendingSlot>,
}

impl ObjectBuilder {
    #[must_use]
    pub fn new(minos: u32) -> Self {
        let mut object = Object::new(
            BinaryFormat::MachO,
            Architecture::Aarch64,
            Endianness::Little,
        );
        // without LC_BUILD_VERSION ld warns "no platform load command found" for every nova object
        // MachOBuildVersion is non_exhaustive, so it is filled in from its Default
        let mut build_version = MachOBuildVersion::default();
        build_version.platform = PLATFORM_MACOS;
        build_version.minos = Version(minos);
        // sdk 0 is what rustc and LLVM write: it only matters at run time and ld sets it at the final link
        // (rustc_codegen_ssa back/metadata.rs:437)
        build_version.sdk = Version(0);
        object.set_macho_build_version(build_version);
        let text = object.section_id(::object::write::StandardSection::Text);
        Self {
            object,
            text,
            symbols: BTreeMap::new(),
            pending: Vec::new(),
            slots: Vec::new(),
        }
    }

    /// Append one function's code, define its symbol and stash its relocations.
    #[must_use = "a refused function must be handled"]
    pub fn feed(&mut self, function: Encoded) -> Result<(), ObjectError> {
        let value = self
            .object
            .append_section_data(self.text, &function.code, 4);
        let id = self.object.add_symbol(Symbol {
            name: function.name.as_bytes().to_vec(),
            value,
            size: function.code.len() as u64,
            kind: SymbolKind::Text,
            scope: SymbolScope::Dynamic,
            weak: false,
            section: SymbolSection::Section(self.text),
            flags: SymbolFlags::None,
        });
        self.symbols.insert(function.name, id);
        self.pending.push((value, function.relocations));
        Ok(())
    }

    /// Append one data object, define its symbol and stash its pointer slots.
    #[must_use = "a refused data object must be handled"]
    pub fn feed_data(&mut self, data: DataObject) -> Result<(), ObjectError> {
        let refuse = |reason| ObjectError::Data {
            symbol: data.symbol.clone(),
            reason,
        };
        if !data.align.is_power_of_two() {
            return Err(refuse("alignment is not a power of two"));
        }
        let in_range = |slot: &PointerSlot| {
            slot.offset
                .checked_add(POINTER_BYTES)
                .is_some_and(|end| end <= data.bytes.len() as u64)
        };
        if !data.pointers.iter().all(in_range) {
            return Err(refuse("pointer slot runs past the bytes"));
        }
        let standard = if data.pointers.is_empty() {
            StandardSection::ReadOnlyData
        } else {
            StandardSection::ReadOnlyDataWithRel
        };
        let section = self.object.section_id(standard);
        let value = self
            .object
            .append_section_data(section, &data.bytes, data.align);
        let id = self.object.add_symbol(Symbol {
            name: data.symbol.as_bytes().to_vec(),
            value,
            size: data.bytes.len() as u64,
            kind: SymbolKind::Data,
            scope: SymbolScope::Dynamic,
            weak: false,
            section: SymbolSection::Section(section),
            flags: SymbolFlags::None,
        });
        self.symbols.insert(data.symbol, id);
        self.slots
            .extend(data.pointers.into_iter().map(|slot| PendingSlot {
                section,
                offset: value + slot.offset,
                symbol: slot.symbol,
            }));
        Ok(())
    }

    /// Resolve every stashed relocation and write the object.
    #[must_use = "the object bytes or the refusal must be handled"]
    pub fn finish(mut self) -> Result<Vec<u8>, ObjectError> {
        for (start, relocations) in core::mem::take(&mut self.pending) {
            for relocation in relocations {
                let target = self.resolve(relocation.symbol);
                let placed = code_relocation(relocation.kind, start + relocation.offset, target);
                self.object
                    .add_relocation(self.text, placed)
                    .map_err(|error| ObjectError::Write(error.to_string()))?;
            }
        }
        for slot in core::mem::take(&mut self.slots) {
            let target = self.resolve(slot.symbol);
            self.object
                .add_relocation(slot.section, unsigned64(slot.offset, target))
                .map_err(|error| ObjectError::Write(error.to_string()))?;
        }
        self.object
            .write()
            .map_err(|error| ObjectError::Write(error.to_string()))
    }

    // a symbol a function or data object defined resolves to its id, in any feed order; any other
    // becomes undefined
    fn resolve(&mut self, name: String) -> SymbolId {
        match self.symbols.get(&name) {
            Some(id) => *id,
            None => {
                let id = self.object.add_symbol(undefined_text(&name));
                self.symbols.insert(name, id);
                id
            }
        }
    }
}

const POINTER_BYTES: u64 = 8;

fn code_relocation(kind: RelocationKind, offset: u64, symbol: SymbolId) -> WriteRelocation {
    match kind {
        RelocationKind::Branch26 => branch26(offset, symbol),
        RelocationKind::Page21 => page(offset, symbol, ARM64_RELOC_PAGE21, true),
        RelocationKind::PageOff12 => page(offset, symbol, ARM64_RELOC_PAGEOFF12, false),
    }
}

// the object writer has no generic form for these two, so the Mach-O flags go in directly:
// length 2 is the 4-byte word, and the writer keeps the addend out of the word for both types
// (object write/macho/object.rs macho_adjust_addend)
fn page(offset: u64, symbol: SymbolId, r_type: RelocationType, r_pcrel: bool) -> WriteRelocation {
    WriteRelocation {
        offset,
        symbol,
        addend: 0,
        flags: RelocationFlags::MachO {
            r_type,
            r_pcrel,
            r_length: 2,
        },
    }
}

// an absolute 64-bit slot is ARM64_RELOC_UNSIGNED at length 3 (object write/macho/object.rs:353)
fn unsigned64(offset: u64, symbol: SymbolId) -> WriteRelocation {
    WriteRelocation {
        offset,
        symbol,
        addend: 0,
        flags: RelocationFlags::Generic {
            kind: ObjectKind::Absolute,
            encoding: RelocationEncoding::Generic,
            size: 64,
        },
    }
}

fn undefined_text(name: &str) -> Symbol {
    Symbol {
        name: name.as_bytes().to_vec(),
        value: 0,
        size: 0,
        kind: SymbolKind::Text,
        scope: SymbolScope::Unknown,
        weak: false,
        section: SymbolSection::Undefined,
        flags: SymbolFlags::None,
    }
}

// pc-relative AArch64Call at size 26 is ARM64_RELOC_BRANCH26 (object write/macho/object.rs:362)
fn branch26(offset: u64, symbol: SymbolId) -> WriteRelocation {
    WriteRelocation {
        offset,
        symbol,
        addend: 0,
        flags: RelocationFlags::Generic {
            kind: ObjectKind::Relative,
            encoding: RelocationEncoding::AArch64Call,
            size: 26,
        },
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::fs;
    use std::os::unix::process::ExitStatusExt;
    use std::path::{Path, PathBuf};
    use std::process::{Command, ExitStatus, Stdio};
    use std::string::String;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;
    use std::vec;

    use alloc::string::ToString;

    use ::object::macho::{ARM64_RELOC_BRANCH26, ARM64_RELOC_UNSIGNED, MachHeader64};
    use ::object::read::macho::MachHeader;
    use ::object::read::{Object, ObjectSection, ObjectSymbol, RelocationTarget};

    use super::*;

    const MACOS_11: u32 = 0x000B_0000;
    const MACOS_12: u32 = 0x000C_0000;
    const RUN_TIMEOUT: Duration = Duration::from_secs(10);
    use crate::aarch64::{encode, lower_function};
    use crate::ir::{Block, BlockId, Cond, Function, Inst, Slot, SlotId, Terminator, Type, Value};

    fn iconst(result: u32, imm: i64, ty: Type) -> (Option<Value>, Inst) {
        (Some(Value(result)), Inst::Iconst { ty, imm })
    }

    fn main_with(blocks: vec::Vec<Block>) -> Function {
        Function {
            name: "main".to_string(),
            params: vec![],
            returns: Some(Type::I32),
            slots: vec![],
            blocks,
        }
    }

    fn ret_block(param: Option<u32>, value: u32, insts: vec::Vec<(Option<Value>, Inst)>) -> Block {
        Block {
            params: param.map(|id| (Value(id), Type::I32)).into_iter().collect(),
            insts,
            terminator: Some(Terminator::Return(Some(Value(value)))),
        }
    }

    fn encoded(function: &Function) -> Encoded {
        let lowered = lower_function(function).expect("lower");
        let code = lowered
            .insts
            .iter()
            .flat_map(|inst| encode(*inst).expect("encode").to_le_bytes())
            .collect();
        Encoded {
            name: function.name.clone(),
            code,
            relocations: lowered.relocations,
        }
    }

    fn build_object(functions: &[Encoded], minos: u32) -> Result<vec::Vec<u8>, ObjectError> {
        let mut builder = ObjectBuilder::new(minos);
        functions
            .iter()
            .try_for_each(|function| builder.feed(function.clone()))?;
        builder.finish()
    }

    // nm lines of the object for `_callee`: (defined `T`, undefined `U`)
    fn callee_symbol_counts(dir: &Path, bytes: &[u8]) -> (usize, usize) {
        let obj = dir.join("callee.o");
        fs::write(&obj, bytes).expect("write .o");
        let nm = Command::new("nm").arg(&obj).output().expect("nm");
        let listing = String::from_utf8_lossy(&nm.stdout).into_owned();
        let count = |tag: &str| {
            let wanted = std::format!(" {tag} _callee");
            listing
                .lines()
                .filter(|line| line.ends_with(&wanted))
                .count()
        };
        (count("T"), count("U"))
    }

    fn caller_of_callee() -> Function {
        let insts = vec![iconst(0, 3, Type::I32), call(1, "callee", &[0])];
        function_named("caller", 0, vec![entry_with(0, insts, 1)])
    }

    fn callee_function() -> Function {
        let insts = vec![(Some(Value(1)), Inst::Imul(Value(0), Value(0)))];
        function_named("callee", 1, vec![entry_with(1, insts, 1)])
    }

    #[test]
    fn builder_resolves_forward_call_to_defined_symbol() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut builder = ObjectBuilder::new(MACOS_11);
        builder
            .feed(encoded(&caller_of_callee()))
            .expect("feed caller");
        builder
            .feed(encoded(&callee_function()))
            .expect("feed callee");
        let both = builder.finish().expect("object");
        assert_eq!(callee_symbol_counts(dir.path(), &both), (1, 0));

        let mut alone = ObjectBuilder::new(MACOS_11);
        alone
            .feed(encoded(&caller_of_callee()))
            .expect("feed caller");
        let control = alone.finish().expect("object");
        assert_eq!(callee_symbol_counts(dir.path(), &control), (0, 1));
    }

    fn paged_function() -> Encoded {
        Encoded {
            name: "paged".to_string(),
            code: vec![0; 8],
            relocations: vec![
                Relocation {
                    offset: 0,
                    kind: RelocationKind::Page21,
                    symbol: "data".to_string(),
                },
                Relocation {
                    offset: 4,
                    kind: RelocationKind::PageOff12,
                    symbol: "data".to_string(),
                },
            ],
        }
    }

    // (relocations whose target `_data` is defined in the object, relocations whose target is undefined)
    fn data_target_counts(bytes: &[u8]) -> (usize, usize) {
        let file = ::object::File::parse(bytes).expect("parse object");
        let text = file.section_by_name("__text").expect("__text");
        let targets: vec::Vec<_> = text
            .relocations()
            .filter_map(|(_, relocation)| match relocation.target() {
                RelocationTarget::Symbol(index) => file.symbol_by_index(index).ok(),
                _ => None,
            })
            .filter(|symbol| symbol.name() == Ok("_data"))
            .collect();
        let undefined = targets
            .iter()
            .filter(|symbol| symbol.is_undefined())
            .count();
        (targets.len() - undefined, undefined)
    }

    #[test]
    fn builder_resolves_data_target_at_finish() {
        let mut builder = ObjectBuilder::new(MACOS_11);
        assert_eq!(builder.feed(paged_function()), Ok(()));
        builder
            .feed_data(data_object("data", &[1, 2, 3, 4], 4, &[]))
            .expect("feed data");
        let (defined, undefined) = data_target_counts(&builder.finish().expect("object"));
        std::println!("reloc_targets_defined={defined} reloc_targets_undefined={undefined}");

        let mut alone = ObjectBuilder::new(MACOS_11);
        assert_eq!(alone.feed(paged_function()), Ok(()));
        let (control_defined, control_undefined) =
            data_target_counts(&alone.finish().expect("object"));
        std::println!(
            "reloc_targets_defined={control_defined} reloc_targets_undefined={control_undefined} of={}",
            defined + undefined + control_defined + control_undefined
        );
        assert_eq!(
            (defined, undefined, control_defined, control_undefined),
            (2, 0, 0, 2)
        );

        let mut refusing = ObjectBuilder::new(MACOS_11);
        let odd_alignment = refusing.feed_data(data_object("odd", &[0; 4], 3, &[]));
        let overrun = refusing.feed_data(data_object("slot", &[0; 8], 8, &[(4, "odd")]));
        assert!(matches!(odd_alignment, Err(ObjectError::Data { .. })));
        assert!(matches!(overrun, Err(ObjectError::Data { .. })));
    }

    fn data_object(symbol: &str, bytes: &[u8], align: u64, slots: &[(u64, &str)]) -> DataObject {
        DataObject {
            symbol: symbol.to_string(),
            bytes: bytes.to_vec(),
            align,
            pointers: slots
                .iter()
                .map(|(offset, target)| PointerSlot {
                    offset: *offset,
                    symbol: (*target).to_string(),
                })
                .collect(),
        }
    }

    // the seven statics of AC23: five in __TEXT,__const, two holding pointers in __DATA,__const
    fn static_objects() -> vec::Vec<DataObject> {
        let outer: vec::Vec<u8> = 7u64.to_le_bytes().into_iter().chain([0; 8]).collect();
        vec![
            data_object("high", &70_000u32.to_le_bytes(), 4, &[]),
            data_object("low", &300u32.to_le_bytes(), 4, &[]),
            data_object("inner", &91u32.to_le_bytes(), 4, &[]),
            data_object("odd", &[1], 1, &[]),
            data_object("aligned", &9u64.to_le_bytes(), 8, &[]),
            data_object("first", &[0; 8], 8, &[(0, "low")]),
            data_object("outer", &outer, 8, &[(8, "inner")]),
        ]
    }

    fn load_through(symbol: &str, addr: u32, result: u32) -> [(Option<Value>, Inst); 2] {
        let symbol = symbol.to_string();
        [
            (Some(Value(addr)), Inst::DataAddr { symbol }),
            (
                Some(Value(result)),
                Inst::Load {
                    ty: Type::I32,
                    addr: Value(addr),
                    offset: 0,
                },
            ),
        ]
    }

    fn data_reading_main(with_data: bool) -> Function {
        let mut insts = vec![iconst(6, 5, Type::I32)];
        if with_data {
            insts = load_through("high", 0, 1).into_iter().collect();
            insts.extend(load_through("low", 2, 3));
            insts.extend(load_through("outer", 4, 5));
            insts.push((Some(Value(6)), Inst::Iadd(Value(1), Value(3))));
        }
        insts.push((Some(Value(7)), Inst::Iadd(Value(6), Value(6))));
        main_with(vec![ret_block(None, 7, insts)])
    }

    // the symbols of `statics` that sit in `__const` of `segment`
    fn const_symbols(file: &::object::File, statics: &[DataObject], segment: &str) -> usize {
        statics
            .iter()
            .filter(|data| {
                let name = std::format!("_{}", data.symbol);
                let section = file
                    .symbols()
                    .find(|symbol| symbol.name() == Ok(&*name))
                    .and_then(|symbol| symbol.section_index())
                    .and_then(|index| file.section_by_index(index).ok());
                section.is_some_and(|section| {
                    section.name() == Ok("__const") && section.segment_name() == Ok(Some(segment))
                })
            })
            .count()
    }

    // relocation counts by kind, the unsigned ones in address order, symbols by section, alignment
    fn data_report(bytes: &[u8], statics: &[DataObject]) -> String {
        let file = ::object::File::parse(bytes).expect("parse object");
        let (mut page21, mut pageoff12) = (0, 0);
        let mut unsigned: vec::Vec<(u64, String)> = vec![];
        for section in file.sections() {
            for (offset, relocation) in section.relocations() {
                let RelocationFlags::MachO {
                    r_type, r_length, ..
                } = relocation.flags()
                else {
                    continue;
                };
                let RelocationTarget::Symbol(index) = relocation.target() else {
                    panic!("relocation target is not a symbol");
                };
                let target = file.symbol_by_index(index).expect("target symbol");
                match r_type {
                    ARM64_RELOC_PAGE21 => page21 += 1,
                    ARM64_RELOC_PAGEOFF12 => pageoff12 += 1,
                    ARM64_RELOC_UNSIGNED => {
                        assert_eq!(r_length, 3, "a pointer slot is 8 bytes");
                        unsigned.push((offset, target.name().expect("name").to_string()));
                    }
                    _ => {}
                }
            }
        }
        unsigned.sort();
        let addrs: vec::Vec<_> = unsigned
            .iter()
            .map(|(offset, _)| offset.to_string())
            .collect();
        let targets: vec::Vec<_> = unsigned.iter().map(|(_, name)| name.as_str()).collect();
        let aligned = statics
            .iter()
            .filter(|data| {
                let name = std::format!("_{}", data.symbol);
                file.symbols()
                    .find(|symbol| symbol.name() == Ok(&*name))
                    .is_some_and(|symbol| symbol.address() % data.align == 0)
            })
            .count();
        std::format!(
            "page21={page21} pageoff12={pageoff12} unsigned={} reloc_addrs={} reloc_targets={} text_const={} data_const={} aligned={aligned} of={}",
            unsigned.len(),
            addrs.join(","),
            targets.join(","),
            const_symbols(&file, statics, "__TEXT"),
            const_symbols(&file, statics, "__DATA"),
            statics.len(),
        )
    }

    fn object_with_data(function: &Function, statics: &[DataObject]) -> vec::Vec<u8> {
        let mut builder = ObjectBuilder::new(MACOS_11);
        builder.feed(encoded(function)).expect("feed function");
        for data in statics {
            builder.feed_data(data.clone()).expect("feed data");
        }
        builder.finish().expect("object")
    }

    #[test]
    fn writes_data_relocations() {
        let statics = static_objects();
        let report = data_report(
            &object_with_data(&data_reading_main(true), &statics),
            &statics,
        );
        std::println!("{report}");
        let control = data_report(&object_with_data(&data_reading_main(false), &[]), &[]);
        std::println!("{control}");
        assert_eq!(
            report,
            "page21=3 pageoff12=3 unsigned=2 reloc_addrs=0,16 reloc_targets=_low,_inner text_const=5 data_const=2 aligned=7 of=7"
        );
        assert_eq!(
            control,
            "page21=0 pageoff12=0 unsigned=0 reloc_addrs= reloc_targets= text_const=0 data_const=0 aligned=0 of=0"
        );
    }

    fn link_with_data(dir: &Path, function: &Function, statics: &[DataObject]) -> PathBuf {
        let obj = dir.join("nova.o");
        let exe = dir.join("exe");
        fs::write(&obj, object_with_data(function, statics)).expect("write .o");
        let link = Command::new("cc")
            .arg(&obj)
            .arg("-o")
            .arg(&exe)
            .output()
            .expect("cc");
        assert!(
            link.status.success(),
            "cc: {}",
            String::from_utf8_lossy(&link.stderr)
        );
        exe
    }

    // returns `taken` when `check` equals `expected`, 3 otherwise
    fn return_when_equal(
        mut insts: vec::Vec<(Option<Value>, Inst)>,
        check: u32,
        expected: i64,
        taken: u32,
    ) -> Function {
        insts.push(iconst(90, expected, Type::I32));
        insts.push((
            Some(Value(91)),
            Inst::Icmp {
                cond: Cond::Eq,
                a: Value(check),
                b: Value(90),
            },
        ));
        let entry = Block {
            params: vec![],
            insts,
            terminator: Some(Terminator::Branch {
                cond: Value(91),
                then_block: BlockId(1),
                then_args: vec![Value(taken)],
                else_block: BlockId(2),
                else_args: vec![],
            }),
        };
        let found = ret_block(Some(92), 92, vec![]);
        let missed = ret_block(None, 93, vec![iconst(93, 3, Type::I32)]);
        main_with(vec![entry, found, missed])
    }

    #[test]
    fn links_and_reads_data() {
        let mut insts: vec::Vec<_> = load_through("high", 0, 1).into_iter().collect();
        insts.extend(load_through("low", 2, 3));
        insts.push((Some(Value(4)), Inst::Isub(Value(1), Value(3))));
        insts.push(iconst(5, 101, Type::I32));
        let function = return_when_equal(insts, 4, 69_700, 5);
        let dir = tempfile::tempdir().expect("tempdir");
        let exe = link_with_data(dir.path(), &function, &static_objects());
        assert_eq!(run_bounded(&exe, &[]), 101);
    }

    fn load_of(ty: Type, result: u32, addr: u32, offset: u32) -> (Option<Value>, Inst) {
        (
            Some(Value(result)),
            Inst::Load {
                ty,
                addr: Value(addr),
                offset,
            },
        )
    }

    #[test]
    fn links_and_reads_through_data_pointer() {
        let address = |result: u32, symbol: &str| {
            let symbol = symbol.to_string();
            (Some(Value(result)), Inst::DataAddr { symbol })
        };
        let insts = vec![
            address(0, "first"),
            load_of(Type::Ptr, 1, 0, 0),
            load_of(Type::I32, 2, 1, 0),
            address(3, "outer"),
            load_of(Type::Ptr, 4, 3, 8),
            load_of(Type::I32, 5, 4, 0),
        ];
        let function = return_when_equal(insts, 2, 300, 5);
        let dir = tempfile::tempdir().expect("tempdir");
        let exe = link_with_data(dir.path(), &function, &static_objects());
        assert_eq!(run_bounded(&exe, &[]), 91);
    }

    fn build_and_run(function: &Function) -> i32 {
        build_and_run_args(function, &[])
    }

    fn build_and_run_args(function: &Function, args: &[&str]) -> i32 {
        let dir = tempfile::tempdir().expect("tempdir");
        let exe = link(dir.path(), function, &[]);
        run_bounded(&exe, args)
    }

    // writes the .o, checks nm lists the symbol as `T`, links it with `cc` and any extra C sources
    fn link(dir: &Path, function: &Function, c_sources: &[&Path]) -> PathBuf {
        link_all(dir, std::slice::from_ref(function), c_sources)
    }

    // one object holding every function; nm must list each as `T`
    fn link_all(dir: &Path, functions: &[Function], c_sources: &[&Path]) -> PathBuf {
        let obj = dir.join("nova.o");
        let exe = dir.join("exe");
        let symbols: vec::Vec<Encoded> = functions.iter().map(encoded).collect();
        fs::write(&obj, build_object(&symbols, MACOS_11).expect("object")).expect("write .o");
        let nm = Command::new("nm").arg(&obj).output().expect("nm");
        let listing = String::from_utf8_lossy(&nm.stdout).into_owned();
        for function in functions {
            let symbol = std::format!(" T _{}", function.name);
            assert!(
                listing.lines().any(|line| line.ends_with(&symbol)),
                "nm listing: {listing}"
            );
        }
        let link = Command::new("cc")
            .arg(&obj)
            .args(c_sources)
            .arg("-o")
            .arg(&exe)
            .output()
            .expect("cc");
        assert!(
            link.status.success(),
            "cc: {}",
            String::from_utf8_lossy(&link.stderr)
        );
        exe
    }

    // a waiter thread owns the child and calls wait(); the test blocks on the channel, and on
    // expiry kills the child by pid, which lets the waiter return
    fn run_bounded(exe: &Path, args: &[&str]) -> i32 {
        run_bounded_output(exe, args).0
    }

    // same bound as above, also returns the child's stdout
    fn run_bounded_output(exe: &Path, args: &[&str]) -> (i32, String) {
        let (status, stdout) = run_bounded_status(exe, args);
        (status.code().expect("exit code"), stdout)
    }

    // same waiter and bound, returning the raw status so a death by signal is observable
    fn run_bounded_status(exe: &Path, args: &[&str]) -> (ExitStatus, String) {
        let child = Command::new(exe)
            .args(args)
            .stdout(Stdio::piped())
            .spawn()
            .expect("run");
        let pid = child.id();
        let (sender, receiver) = mpsc::channel();
        let waiter = thread::spawn(move || {
            let _ = sender.send(child.wait_with_output());
        });
        match receiver.recv_timeout(RUN_TIMEOUT) {
            Ok(output) => {
                waiter.join().expect("waiter thread");
                let output = output.expect("wait");
                let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
                (output.status, stdout)
            }
            Err(_) => {
                Command::new("kill")
                    .args(["-9", &pid.to_string()])
                    .status()
                    .expect("kill");
                waiter.join().expect("waiter thread");
                panic!("timed out after 10 s");
            }
        }
    }

    fn two_param_function() -> Function {
        let entry = Block {
            params: vec![(Value(0), Type::I32), (Value(1), Type::I32)],
            insts: vec![
                iconst(2, 100, Type::I32),
                (Some(Value(3)), Inst::Isub(Value(0), Value(1))),
                (Some(Value(4)), Inst::Iadd(Value(3), Value(2))),
            ],
            terminator: Some(Terminator::Return(Some(Value(4)))),
        };
        Function {
            name: "f".to_string(),
            params: vec![Type::I32, Type::I32],
            returns: Some(Type::I32),
            slots: vec![],
            blocks: vec![entry],
        }
    }

    fn run_with_c_main(function: &Function, c_main: &str) -> i32 {
        let dir = tempfile::tempdir().expect("tempdir");
        let source = dir.path().join("main.c");
        fs::write(&source, c_main).expect("write c main");
        let exe = link(dir.path(), function, &[&source]);
        run_bounded(&exe, &[])
    }

    #[test]
    fn links_two_params_from_c() {
        let function = two_param_function();
        let seven_three = run_with_c_main(
            &function,
            "int f(int a, int b);\nint main(void) { return f(7, 3); }\n",
        );
        let three_seven = run_with_c_main(
            &function,
            "int f(int a, int b);\nint main(void) { return f(3, 7); }\n",
        );
        assert_eq!((seven_three, three_seven), (104, 96));
    }

    // `wide(value)` takes the C caller's int64_t and returns an I64: the param is typed I32 under
    // `extend`, so the extension must ignore the upper half the caller left set
    fn wide_function(extend: Option<Inst>) -> Function {
        let ty = if extend.is_some() {
            Type::I32
        } else {
            Type::I64
        };
        let insts: vec::Vec<_> = extend
            .map(|inst| (Some(Value(1)), inst))
            .into_iter()
            .collect();
        let result = if insts.is_empty() { 0 } else { 1 };
        Function {
            name: "wide".to_string(),
            params: vec![ty],
            returns: Some(Type::I64),
            slots: vec![],
            blocks: vec![Block {
                params: vec![(Value(0), ty)],
                insts,
                terminator: Some(Terminator::Return(Some(Value(result)))),
            }],
        }
    }

    fn wide_main(argument: &str, expected: &str, hit: u8) -> String {
        std::format!(
            "#include <stdint.h>\nint64_t wide(int64_t value);\nint main(void) {{ return wide({argument}) == {expected} ? {hit} : {}; }}\n",
            hit % 10
        )
    }

    #[test]
    fn extends_over_garbage_upper_half() {
        let widen = |inst: fn(Type, Value) -> Inst| inst(Type::I64, Value(0));
        let uextend = wide_function(Some(widen(|ty, value| Inst::Uextend { ty, value })));
        let sextend = wide_function(Some(widen(|ty, value| Inst::Sextend { ty, value })));
        let control = wide_function(None);
        let unsigned = run_with_c_main(&uextend, &wide_main("0xDEADBEEF00000005ULL", "5", 41));
        let signed = run_with_c_main(
            &sextend,
            &wide_main("(int64_t)0xDEADBEEFFFFFFFFBULL", "-5", 42),
        );
        let kept = run_with_c_main(&control, &wide_main("0xDEADBEEF00000005ULL", "5", 41));
        assert_eq!((unsigned, signed, kept), (41, 42, 1));
    }

    #[test]
    fn links_and_exits_after_long_chain() {
        let mut insts = vec![iconst(0, 1, Type::I32)];
        let mut next = 1;
        for _ in 0..39 {
            insts.push(iconst(next, 3, Type::I32));
            insts.push((
                Some(Value(next + 1)),
                Inst::Iadd(Value(next - 1), Value(next)),
            ));
            next += 2;
        }
        let entry = ret_block(None, next - 1, insts);
        assert_eq!(build_and_run(&main_with(vec![entry])), 118);
    }

    #[test]
    fn links_and_exits_42() {
        let entry = Block {
            params: vec![],
            insts: vec![
                iconst(0, 40, Type::I32),
                iconst(1, 2, Type::I32),
                (Some(Value(2)), Inst::Iadd(Value(0), Value(1))),
            ],
            terminator: Some(Terminator::Jump {
                target: BlockId(1),
                args: vec![Value(2)],
            }),
        };
        let exit = ret_block(Some(3), 3, vec![]);
        assert_eq!(build_and_run(&main_with(vec![entry, exit])), 42);
    }

    #[test]
    fn links_and_exits_7() {
        let entry = ret_block(
            None,
            2,
            vec![
                iconst(0, 10, Type::I32),
                iconst(1, 3, Type::I32),
                (Some(Value(2)), Inst::Isub(Value(0), Value(1))),
            ],
        );
        assert_eq!(build_and_run(&main_with(vec![entry])), 7);
    }

    #[test]
    fn links_and_exits_through_stack_slot() {
        let entry = ret_block(
            None,
            3,
            vec![
                iconst(1, 23, Type::I32),
                (Some(Value(0)), Inst::SlotAddr(SlotId(0))),
                (
                    None,
                    Inst::Store {
                        value: Value(1),
                        addr: Value(0),
                        offset: 0,
                    },
                ),
                (Some(Value(2)), Inst::SlotAddr(SlotId(0))),
                (
                    Some(Value(3)),
                    Inst::Load {
                        ty: Type::I32,
                        addr: Value(2),
                        offset: 0,
                    },
                ),
            ],
        );
        let mut function = main_with(vec![entry]);
        function.slots = vec![Slot { size: 4, align: 4 }];
        assert_eq!(build_and_run(&function), 23);
    }

    // 30 slots, slot k holds k + 1; every access takes its own address, so 1 + 2 + ... + 30 = 465
    // survives only as 465 & 255 = 209 in the exit status
    #[test]
    fn links_and_exits_after_many_slots() {
        const SLOTS: u32 = 30;
        let mut insts = vec![];
        for slot in 0..SLOTS {
            insts.push(iconst(slot, i64::from(slot + 1), Type::I32));
            insts.push((Some(Value(100 + slot)), Inst::SlotAddr(SlotId(slot))));
            insts.push((
                None,
                Inst::Store {
                    value: Value(slot),
                    addr: Value(100 + slot),
                    offset: 0,
                },
            ));
        }
        for slot in 0..SLOTS {
            insts.push((Some(Value(200 + slot)), Inst::SlotAddr(SlotId(slot))));
            insts.push((
                Some(Value(300 + slot)),
                Inst::Load {
                    ty: Type::I32,
                    addr: Value(200 + slot),
                    offset: 0,
                },
            ));
            if slot > 0 {
                let previous = if slot == 1 { 300 } else { 400 + slot - 1 };
                insts.push((
                    Some(Value(400 + slot)),
                    Inst::Iadd(Value(previous), Value(300 + slot)),
                ));
            }
        }
        let entry = ret_block(None, 400 + SLOTS - 1, insts);
        let mut function = main_with(vec![entry]);
        function.slots = vec![Slot { size: 4, align: 4 }; SLOTS as usize];
        assert_eq!(build_and_run(&function), 209);
    }

    #[test]
    fn links_and_exits_from_param() {
        let entry = Block {
            params: vec![(Value(0), Type::I32)],
            insts: vec![
                (Some(Value(1)), Inst::SlotAddr(SlotId(0))),
                (
                    None,
                    Inst::Store {
                        value: Value(0),
                        addr: Value(1),
                        offset: 0,
                    },
                ),
                (Some(Value(5)), Inst::SlotAddr(SlotId(0))),
                (
                    Some(Value(2)),
                    Inst::Load {
                        ty: Type::I32,
                        addr: Value(5),
                        offset: 0,
                    },
                ),
                iconst(3, 6, Type::I32),
                (Some(Value(4)), Inst::Iadd(Value(2), Value(3))),
            ],
            terminator: Some(Terminator::Return(Some(Value(4)))),
        };
        let mut function = main_with(vec![entry]);
        function.params = vec![Type::I32];
        function.slots = vec![Slot { size: 4, align: 4 }];
        assert_eq!(build_and_run_args(&function, &[]), 7);
        assert_eq!(build_and_run_args(&function, &["a", "b"]), 9);
    }

    fn slot_store(addr: u32, slot: u32, value: u32) -> [(Option<Value>, Inst); 2] {
        [
            (Some(Value(addr)), Inst::SlotAddr(SlotId(slot))),
            (
                None,
                Inst::Store {
                    value: Value(value),
                    addr: Value(addr),
                    offset: 0,
                },
            ),
        ]
    }

    // slots land at 0 (16), 16 (4), 20 (4092) and 4112 (4): `high` is `low` plus 4096, so a
    // SlotAddr that drops the upper bits aliases it onto `low` and the exit becomes 7
    #[test]
    fn links_and_exits_over_4095_frame() {
        let mut insts = vec![iconst(0, 42, Type::I32), iconst(1, 7, Type::I32)];
        insts.extend(slot_store(2, 1, 0));
        insts.extend(slot_store(3, 3, 1));
        insts.push((Some(Value(4)), Inst::SlotAddr(SlotId(1))));
        insts.push((
            Some(Value(5)),
            Inst::Load {
                ty: Type::I32,
                addr: Value(4),
                offset: 0,
            },
        ));
        let mut function = main_with(vec![ret_block(None, 5, insts)]);
        function.slots = vec![
            Slot {
                size: 16,
                align: 16,
            },
            Slot { size: 4, align: 4 },
            Slot {
                size: 4092,
                align: 4,
            },
            Slot { size: 4, align: 4 },
        ];
        assert_eq!(build_and_run(&function), 42);
    }

    // frame 40016 = 9 pages + 3152: the probe loop and a remainder probe both run; a loop that
    // ran off the stack would fault
    #[test]
    fn links_and_exits_through_probe_loop() {
        let mut insts = vec![
            iconst(1, 41, Type::I32),
            (Some(Value(2)), Inst::Iadd(Value(0), Value(1))),
        ];
        insts.extend(slot_store(3, 1, 2));
        insts.push((Some(Value(4)), Inst::SlotAddr(SlotId(1))));
        insts.push((
            Some(Value(5)),
            Inst::Load {
                ty: Type::I32,
                addr: Value(4),
                offset: 0,
            },
        ));
        let entry = Block {
            params: vec![(Value(0), Type::I32)],
            insts,
            terminator: Some(Terminator::Return(Some(Value(5)))),
        };
        let mut function = main_with(vec![entry]);
        function.params = vec![Type::I32];
        function.slots = vec![
            Slot {
                size: 40000,
                align: 16,
            },
            Slot { size: 4, align: 4 },
        ];
        assert_eq!(build_and_run_args(&function, &[]), 42);
    }

    fn build_versions(bytes: &[u8]) -> vec::Vec<(u32, u32, u32)> {
        let header = MachHeader64::<Endianness>::parse(bytes, 0).expect("mach-o header");
        let endian = header.endian().expect("endian");
        let mut commands = header
            .load_commands(endian, bytes, 0)
            .expect("load commands");
        let mut found = vec![];
        while let Some(command) = commands.next().expect("next load command") {
            if let Some((version, _tools)) = command.build_version(endian).expect("build version") {
                found.push((
                    version.platform.get(endian).0,
                    version.minos.get(endian).0,
                    version.sdk.get(endian).0,
                ));
            }
        }
        found
    }

    #[test]
    fn writes_macos_build_version() {
        let function = main_with(vec![ret_block(None, 0, vec![iconst(0, 1, Type::I32)])]);
        let written = build_object(&[encoded(&function)], MACOS_11).expect("object");
        let mut bare = ::object::write::Object::new(
            BinaryFormat::MachO,
            Architecture::Aarch64,
            Endianness::Little,
        );
        bare.section_id(::object::write::StandardSection::Text);
        let control = bare.write().expect("bare object");
        assert_eq!(
            build_versions(&written),
            vec![(PLATFORM_MACOS.0, MACOS_11, 0)]
        );
        let chosen = build_object(&[encoded(&function)], MACOS_12).expect("object");
        assert_eq!(
            build_versions(&chosen),
            vec![(PLATFORM_MACOS.0, MACOS_12, 0)]
        );
        assert_eq!(build_versions(&control).len(), 0);
    }

    #[test]
    fn links_and_exits_254() {
        let entry = ret_block(None, 0, vec![iconst(0, -2, Type::I64)]);
        assert_eq!(build_and_run(&main_with(vec![entry])), 254);
    }

    fn val(id: u32) -> Value {
        Value(id)
    }

    fn typed(ids: &[u32]) -> vec::Vec<(Value, Type)> {
        ids.iter().map(|id| (Value(*id), Type::I32)).collect()
    }

    fn jump(target: u32, args: &[u32]) -> Option<Terminator> {
        Some(Terminator::Jump {
            target: BlockId(target),
            args: args.iter().map(|id| Value(*id)).collect(),
        })
    }

    // loop(px, py, pacc, pn): n1 = pn - one; acc1 = pacc + py; then edge swaps px/py
    fn loop_body() -> vec::Vec<(Option<Value>, Inst)> {
        vec![
            (Some(val(9)), Inst::Isub(val(8), val(4))),
            (Some(val(10)), Inst::Iadd(val(7), val(6))),
        ]
    }

    fn exit_block() -> Block {
        Block {
            params: typed(&[11, 12, 13]),
            insts: vec![
                iconst(14, 100, Type::I32),
                (Some(val(15)), Inst::Iadd(val(13), val(11))),
                (Some(val(16)), Inst::Isub(val(15), val(12))),
                (Some(val(17)), Inst::Iadd(val(16), val(14))),
            ],
            terminator: Some(Terminator::Return(Some(val(17)))),
        }
    }

    fn entry_block() -> Block {
        Block {
            params: vec![],
            insts: vec![
                iconst(0, 3, Type::I32),
                iconst(1, 5, Type::I32),
                iconst(2, 0, Type::I32),
                iconst(3, 7, Type::I32),
                iconst(4, 1, Type::I32),
            ],
            terminator: jump(1, &[0, 1, 2, 3]),
        }
    }

    fn branch(then_block: u32, then_args: &[u32], else_block: u32) -> Option<Terminator> {
        Some(Terminator::Branch {
            cond: val(8),
            then_block: BlockId(then_block),
            then_args: then_args.iter().map(|id| Value(*id)).collect(),
            else_block: BlockId(else_block),
            else_args: vec![val(5), val(6), val(7)],
        })
    }

    // the same loop with its back edge a Branch then edge, or a Jump from a body block
    fn swap_loop(jump_back_edge: bool) -> Function {
        let header_params = typed(&[5, 6, 7, 8]);
        let blocks = if jump_back_edge {
            let header = Block {
                params: header_params,
                insts: vec![],
                terminator: branch(2, &[], 3),
            };
            let body = Block {
                params: vec![],
                insts: loop_body(),
                terminator: jump(1, &[6, 5, 10, 9]),
            };
            vec![entry_block(), header, body, exit_block()]
        } else {
            let looped = Block {
                params: header_params,
                insts: loop_body(),
                terminator: branch(1, &[6, 5, 10, 9], 2),
            };
            vec![entry_block(), looped, exit_block()]
        };
        main_with(blocks)
    }

    #[test]
    fn links_and_exits_after_loop() {
        assert_eq!(build_and_run(&swap_loop(false)), 131);
        assert_eq!(build_and_run(&swap_loop(true)), 131);
        let dir = tempfile::tempdir().expect("tempdir");
        let exe = dir.path().join("swap_loop");
        let fixture = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/swap_loop.rs");
        let compiled = Command::new("rustc")
            .args(["--edition", "2024", fixture, "-o"])
            .arg(&exe)
            .output()
            .expect("rustc");
        assert!(
            compiled.status.success(),
            "rustc: {}",
            String::from_utf8_lossy(&compiled.stderr)
        );
        assert_eq!(run_bounded(&exe, &[]), 131);
    }

    fn compare_function(name: &str, cond: Cond, ty: Type) -> Function {
        let entry = Block {
            params: vec![(val(0), ty), (val(1), ty)],
            insts: vec![(
                Some(val(2)),
                Inst::Icmp {
                    cond,
                    a: val(0),
                    b: val(1),
                },
            )],
            terminator: Some(Terminator::Return(Some(val(2)))),
        };
        Function {
            name: name.to_string(),
            params: vec![ty, ty],
            returns: Some(Type::I8),
            slots: vec![],
            blocks: vec![entry],
        }
    }

    const COND_TABLE: [(&str, Cond); 10] = [
        ("cmp_eq", Cond::Eq),
        ("cmp_ne", Cond::Ne),
        ("cmp_slt", Cond::Slt),
        ("cmp_sle", Cond::Sle),
        ("cmp_sgt", Cond::Sgt),
        ("cmp_sge", Cond::Sge),
        ("cmp_ult", Cond::Ult),
        ("cmp_ule", Cond::Ule),
        ("cmp_ugt", Cond::Ugt),
        ("cmp_uge", Cond::Uge),
    ];

    // the table, with cmp_slt's body and param type replaceable for the controls
    fn compare_table(slt_cond: Cond, slt_ty: Type) -> vec::Vec<Function> {
        COND_TABLE
            .iter()
            .map(|(name, cond)| match *name {
                "cmp_slt" => compare_function(name, slt_cond, slt_ty),
                _ => compare_function(name, *cond, Type::I32),
            })
            .collect()
    }

    // the C caller declares `long` params, so it owns all 64 bits of x0 and x1
    const COMPARE_MAIN: &str = r#"
#include <stdint.h>
#include <stdio.h>
long cmp_eq(long a, long b);
long cmp_ne(long a, long b);
long cmp_slt(long a, long b);
long cmp_sle(long a, long b);
long cmp_sgt(long a, long b);
long cmp_sge(long a, long b);
long cmp_ult(long a, long b);
long cmp_ule(long a, long b);
long cmp_ugt(long a, long b);
long cmp_uge(long a, long b);

static long widen(int32_t value, uint64_t upper) {
    return (long)((upper << 32) | (uint32_t)value);
}

int main(void) {
    int32_t values[5] = {INT32_MIN, -2, 0, 3, INT32_MAX};
    int ran = 0, mismatched = 0;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            int a = values[i], b = values[j];
            unsigned ua = (unsigned)a, ub = (unsigned)b;
            long wa = widen(a, 0xDEADBEEFu), wb = widen(b, 0x12345678u);
#define CHECK(fn, expected) do { ran++; if (fn(wa, wb) != (long)(expected)) mismatched++; } while (0)
            CHECK(cmp_eq, a == b);
            CHECK(cmp_ne, a != b);
            CHECK(cmp_slt, a < b);
            CHECK(cmp_sle, a <= b);
            CHECK(cmp_sgt, a > b);
            CHECK(cmp_sge, a >= b);
            CHECK(cmp_ult, ua < ub);
            CHECK(cmp_ule, ua <= ub);
            CHECK(cmp_ugt, ua > ub);
            CHECK(cmp_uge, ua >= ub);
        }
    }
    printf("ran=%d mismatched=%d\n", ran, mismatched);
    return 0;
}
"#;

    fn run_compare_table(functions: &[Function]) -> String {
        let dir = tempfile::tempdir().expect("tempdir");
        let source = dir.path().join("main.c");
        fs::write(&source, COMPARE_MAIN).expect("write c main");
        let exe = link_all(dir.path(), functions, &[&source]);
        let (code, stdout) = run_bounded_output(&exe, &[]);
        assert_eq!(code, 0, "stdout: {stdout}");
        stdout
    }

    #[test]
    fn compare_conds_match_c() {
        let table = run_compare_table(&compare_table(Cond::Slt, Type::I32));
        std::print!("table: {table}");
        let unsigned_slt = run_compare_table(&compare_table(Cond::Ult, Type::I32));
        std::print!("control ult: {unsigned_slt}");
        let wide_slt = run_compare_table(&compare_table(Cond::Slt, Type::I64));
        std::print!("control i64: {wide_slt}");
        assert_eq!(table.trim(), "ran=250 mismatched=0");
        assert_eq!(unsigned_slt.trim(), "ran=250 mismatched=12");
        assert_eq!(wide_slt.trim(), "ran=250 mismatched=15");
    }

    // sum i*i while (i < 10), the loop flag round-tripped through a 1-byte slot; returns s - 200
    fn mul_cmp_loop() -> Function {
        let entry = Block {
            params: vec![],
            insts: vec![
                iconst(0, 0, Type::I32),
                iconst(1, 0, Type::I32),
                iconst(2, 10, Type::I32),
                iconst(3, 1, Type::I32),
                iconst(4, 200, Type::I32),
            ],
            terminator: jump(1, &[0, 1]),
        };
        let header = Block {
            params: typed(&[6, 7]),
            insts: vec![
                (
                    Some(val(8)),
                    Inst::Icmp {
                        cond: Cond::Slt,
                        a: val(6),
                        b: val(2),
                    },
                ),
                (Some(val(15)), Inst::SlotAddr(SlotId(0))),
                (
                    None,
                    Inst::Store {
                        value: val(8),
                        addr: val(15),
                        offset: 0,
                    },
                ),
                (Some(val(16)), Inst::SlotAddr(SlotId(0))),
                (
                    Some(val(9)),
                    Inst::Load {
                        ty: Type::I8,
                        addr: val(16),
                        offset: 0,
                    },
                ),
            ],
            terminator: Some(Terminator::Branch {
                cond: val(9),
                then_block: BlockId(2),
                then_args: vec![],
                else_block: BlockId(3),
                else_args: vec![val(7)],
            }),
        };
        let body = Block {
            params: vec![],
            insts: vec![
                (Some(val(10)), Inst::Imul(val(6), val(6))),
                (Some(val(11)), Inst::Iadd(val(7), val(10))),
                (Some(val(12)), Inst::Iadd(val(6), val(3))),
            ],
            terminator: jump(1, &[12, 11]),
        };
        let exit = Block {
            params: typed(&[13]),
            insts: vec![(Some(val(14)), Inst::Isub(val(13), val(4)))],
            terminator: Some(Terminator::Return(Some(val(14)))),
        };
        let mut function = main_with(vec![entry, header, body, exit]);
        function.slots = vec![Slot { size: 1, align: 1 }];
        function
    }

    #[test]
    fn links_and_exits_after_mul_and_compare() {
        assert_eq!(build_and_run(&mul_cmp_loop()), 85);
        let dir = tempfile::tempdir().expect("tempdir");
        let exe = dir.path().join("mul_cmp_loop");
        let fixture = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/mul_cmp_loop.rs"
        );
        let compiled = Command::new("rustc")
            .args(["--edition", "2024", fixture, "-o"])
            .arg(&exe)
            .output()
            .expect("rustc");
        assert!(
            compiled.status.success(),
            "rustc: {}",
            String::from_utf8_lossy(&compiled.stderr)
        );
        assert_eq!(run_bounded(&exe, &[]), 85);
    }

    fn call_typed(result: u32, callee: &str, args: &[u32], ret: Type) -> (Option<Value>, Inst) {
        (
            Some(Value(result)),
            Inst::Call {
                callee: callee.to_string(),
                args: args.iter().map(|id| Value(*id)).collect(),
                ret: Some(ret),
            },
        )
    }

    fn call(result: u32, callee: &str, args: &[u32]) -> (Option<Value>, Inst) {
        call_typed(result, callee, args, Type::I32)
    }

    fn call_for_effect(callee: &str) -> (Option<Value>, Inst) {
        (
            None,
            Inst::Call {
                callee: callee.to_string(),
                args: vec![],
                ret: None,
            },
        )
    }

    fn function_named(name: &str, params: u32, blocks: vec::Vec<Block>) -> Function {
        Function {
            name: name.to_string(),
            params: (0..params).map(|_| Type::I32).collect(),
            returns: Some(Type::I32),
            slots: vec![],
            blocks,
        }
    }

    fn entry_with(params: u32, insts: vec::Vec<(Option<Value>, Inst)>, result: u32) -> Block {
        Block {
            params: (0..params).map(|id| (Value(id), Type::I32)).collect(),
            insts,
            terminator: Some(Terminator::Return(Some(Value(result)))),
        }
    }

    fn square_function() -> Function {
        let insts = vec![(Some(Value(1)), Inst::Imul(Value(0), Value(0)))];
        function_named("square", 1, vec![entry_with(1, insts, 1)])
    }

    // returns the sum of its `count` params
    fn summing(name: &str, count: u32) -> Function {
        let insts = (1..count)
            .map(|index| {
                let left = if index == 1 { 0 } else { count + index - 1 };
                (
                    Some(Value(count + index)),
                    Inst::Iadd(Value(left), Value(index)),
                )
            })
            .collect();
        function_named(name, count, vec![entry_with(count, insts, 2 * count - 1)])
    }

    fn branch26_object(strip_relocations: bool) -> vec::Vec<u8> {
        let insts = vec![
            iconst(0, 3, Type::I32),
            call(1, "square", &[0]),
            call(2, "abs", &[1]),
        ];
        let functions = [
            square_function(),
            main_with(vec![ret_block(None, 2, insts)]),
        ];
        let mut symbols: vec::Vec<Encoded> = functions.iter().map(encoded).collect();
        if strip_relocations {
            symbols
                .iter_mut()
                .for_each(|symbol| symbol.relocations.clear());
        }
        build_object(&symbols, MACOS_11).expect("object")
    }

    // relocations that are ARM64_RELOC_BRANCH26 at a `bl` word inside `_main`, one per wanted target
    fn branch26_found(bytes: &[u8], wanted: &[&str]) -> usize {
        let file = ::object::File::parse(bytes).expect("parse object");
        let main = file
            .symbols()
            .find(|symbol| symbol.name() == Ok("_main"))
            .expect("_main");
        let section = file.section_by_name("__text").expect("__text");
        let data = section.data().expect("section data");
        // mach-o symbols carry no size, and `_main` is the last function in the section
        let (start, end) = (main.address() - section.address(), data.len() as u64);
        let relocations: vec::Vec<_> = section.relocations().collect();
        wanted
            .iter()
            .filter(|name| {
                relocations.iter().any(|(offset, relocation)| {
                    let word = data
                        .get(*offset as usize..*offset as usize + 4)
                        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                    let target = match relocation.target() {
                        RelocationTarget::Symbol(index) => file.symbol_by_index(index).ok(),
                        _ => None,
                    };
                    let is_branch26 = matches!(
                        relocation.flags(),
                        RelocationFlags::MachO {
                            r_type: ARM64_RELOC_BRANCH26,
                            r_pcrel: true,
                            ..
                        }
                    );
                    is_branch26
                        && (start..end).contains(offset)
                        && word.is_some_and(|word| word & 0xfc00_0000 == 0x9400_0000)
                        && target.is_some_and(|symbol| symbol.name() == Ok(**name))
                })
            })
            .count()
    }

    #[test]
    fn writes_branch26_relocation() {
        let wanted = ["_square", "_abs"];
        let object = branch26_object(false);
        let file = ::object::File::parse(&*object).expect("parse object");
        let total: usize = file
            .sections()
            .map(|section| section.relocations().count())
            .sum();
        let found = branch26_found(&object, &wanted);
        std::println!("branch26={found} of={}", wanted.len());
        assert_eq!(
            (found, total),
            (2, 2),
            "exactly two relocations, both branch26"
        );
        let control = branch26_found(&branch26_object(true), &wanted);
        std::println!("control branch26={control} of={}", wanted.len());
        assert_eq!(control, 0);
    }

    const SMASH_SOURCE: &str = "\
.text
.globl _smash
.p2align 2
_smash:
    movz x0, #0xBAD
    movz x1, #0xBAD
    movz x2, #0xBAD
    movz x3, #0xBAD
    movz x4, #0xBAD
    movz x5, #0xBAD
    movz x6, #0xBAD
    movz x7, #0xBAD
    movz x8, #0xBAD
    movz x9, #0xBAD
    movz x10, #0xBAD
    movz x11, #0xBAD
    movz x12, #0xBAD
    movz x13, #0xBAD
    movz x14, #0xBAD
    movz x15, #0xBAD
    movz x16, #0xBAD
    movz x17, #0xBAD
    ret
";

    fn run_with_sources(
        functions: &[Function],
        sources: &[(&str, &str)],
        args: &[&str],
    ) -> (i32, String) {
        let dir = tempfile::tempdir().expect("tempdir");
        let paths: vec::Vec<PathBuf> = sources
            .iter()
            .map(|(file, text)| {
                let path = dir.path().join(file);
                fs::write(&path, text).expect("write source");
                path
            })
            .collect();
        let refs: vec::Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
        let exe = link_all(dir.path(), functions, &refs);
        run_bounded_output(&exe, args)
    }

    fn calls_main() -> Function {
        let mut insts = vec![
            iconst(1, 4, Type::I32),
            (Some(Value(2)), Inst::Iadd(Value(0), Value(1))),
            call(3, "square", &[2]),
            call_for_effect("smash"),
            iconst(4, 12, Type::I32),
            call(5, "add3", &[2, 3, 4]),
        ];
        insts.extend((1..=8).map(|number| iconst(5 + number, i64::from(number), Type::I32)));
        insts.push(call(14, "sum8", &[6, 7, 8, 9, 10, 11, 12, 13]));
        insts.extend([
            iconst(15, 9, Type::I32),
            (Some(Value(16)), Inst::Isub(Value(0), Value(15))),
            call(17, "abs", &[16]),
            (Some(Value(18)), Inst::Iadd(Value(5), Value(2))),
            (Some(Value(19)), Inst::Iadd(Value(18), Value(17))),
            (Some(Value(20)), Inst::Iadd(Value(19), Value(14))),
            iconst(21, 36, Type::I32),
            (Some(Value(22)), Inst::Isub(Value(20), Value(21))),
            (Some(Value(23)), Inst::Iadd(Value(22), Value(0))),
            iconst(24, 1, Type::I32),
            (Some(Value(25)), Inst::Isub(Value(23), Value(24))),
        ]);
        function_named("main", 1, vec![entry_with(1, insts, 25)])
    }

    #[test]
    fn links_and_exits_after_calls() {
        let functions = [
            square_function(),
            summing("add3", 3),
            summing("sum8", 8),
            calls_main(),
        ];
        let (code, _) = run_with_sources(&functions, &[("smash.s", SMASH_SOURCE)], &[]);
        assert_eq!(code, 55, "nova");
        let dir = tempfile::tempdir().expect("tempdir");
        let exe = dir.path().join("calls");
        let fixture = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/calls.rs");
        let compiled = Command::new("rustc")
            .args(["--edition", "2024", fixture, "-o"])
            .arg(&exe)
            .output()
            .expect("rustc");
        assert!(
            compiled.status.success(),
            "rustc: {}",
            String::from_utf8_lossy(&compiled.stderr)
        );
        assert_eq!(run_bounded(&exe, &[]), 55, "stock rustc");
    }

    #[test]
    fn swaps_call_args_in_registers() {
        let g_insts = vec![
            (Some(Value(2)), Inst::Isub(Value(0), Value(1))),
            iconst(3, 120, Type::I32),
            (Some(Value(4)), Inst::Iadd(Value(2), Value(3))),
        ];
        let g = function_named("g", 2, vec![entry_with(2, g_insts, 4)]);
        let f = function_named("f", 2, vec![entry_with(2, vec![call(2, "g", &[1, 0])], 2)]);
        let c_main = "int f(int p, int q);\nint main(void) { return f(7, 3); }\n";
        let (code, _) = run_with_sources(&[g, f], &[("main.c", c_main)], &[]);
        assert_eq!(
            code, 116,
            "g(q, p) = 3 - 7 + 120; one-at-a-time moves give g(3, 3) = 120"
        );
    }

    const SAVED_SOURCE: &str = "\
.text
.globl _probe, _clobber, _clobber_fp, _unlinked, _run_saved
.p2align 2
_probe:
    ldr x0, [x29]
    ret
_clobber:
    movz x19, #0xDEAD
    movz x20, #0xBEEF
    mov x0, #0
    ret
_clobber_fp:
    mov x29, #0
    mov x0, #0
    ret
_unlinked:
    mov x17, x30
    bl _probe
    mov x30, x17
    ret
_run_saved:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    stp x25, x26, [sp, #-16]!
    stp x27, x28, [sp, #-16]!
    sub sp, sp, #16
    str x2, [sp]
    str x29, [sp, #8]
    mov x16, x0
    mov w0, w1
    mov x19, #19
    mov x20, #20
    mov x21, #21
    mov x22, #22
    mov x23, #23
    mov x24, #24
    mov x25, #25
    mov x26, #26
    mov x27, #27
    mov x28, #28
    blr x16
    ldr x1, [sp]
    ldr x2, [sp, #8]
    str x0, [x1]
    str x2, [x1, #8]
    mov x3, #0
    cmp x19, #19
    cinc x3, x3, ne
    cmp x20, #20
    cinc x3, x3, ne
    cmp x21, #21
    cinc x3, x3, ne
    cmp x22, #22
    cinc x3, x3, ne
    cmp x23, #23
    cinc x3, x3, ne
    cmp x24, #24
    cinc x3, x3, ne
    cmp x25, #25
    cinc x3, x3, ne
    cmp x26, #26
    cinc x3, x3, ne
    cmp x27, #27
    cinc x3, x3, ne
    cmp x28, #28
    cinc x3, x3, ne
    cmp x29, x2
    cinc x3, x3, ne
    str x3, [x1, #16]
    add sp, sp, #16
    ldp x27, x28, [sp], #16
    ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16
    ldp x29, x30, [sp], #16
    ret
";

    const SAVED_MAIN: &str = r#"#include <stdio.h>
struct out { long result; long frame; long changed; };
typedef long (*fn_t)(int);
void run_saved(fn_t fn, int arg, struct out *out);
long keep(int); long pick(int); long chain(int);
long unlinked(int); long clobber(int); long clobber_fp(int);
int main(void) {
    struct out o;
    run_saved(keep, 12, &o);
    printf("keep changed=%ld result=%ld\n", o.changed, o.result);
    run_saved(pick, 0, &o);
    printf("pick0 changed=%ld result=%ld\n", o.changed, o.result);
    run_saved(pick, 1, &o);
    printf("pick1 changed=%ld result=%ld\n", o.changed, o.result);
    run_saved(chain, 0, &o);
    printf("chain linked=%d\n", o.result == o.frame);
    run_saved(unlinked, 0, &o);
    printf("unlinked linked=%d\n", o.result == o.frame);
    run_saved(clobber, 0, &o);
    printf("clobber changed=%ld\n", o.changed);
    run_saved(clobber_fp, 0, &o);
    printf("clobber_fp changed=%ld\n", o.changed);
    return 0;
}
"#;

    fn keep_function() -> Function {
        let insts = vec![
            iconst(1, 10, Type::I32),
            iconst(2, 21, Type::I32),
            call_for_effect("smash"),
            (Some(Value(3)), Inst::Iadd(Value(0), Value(1))),
            (Some(Value(4)), Inst::Iadd(Value(3), Value(2))),
        ];
        function_named("keep", 1, vec![entry_with(1, insts, 4)])
    }

    fn pick_function() -> Function {
        let entry = Block {
            params: vec![(Value(0), Type::I32)],
            insts: vec![call_for_effect("smash")],
            terminator: Some(Terminator::Branch {
                cond: Value(0),
                then_block: BlockId(1),
                then_args: vec![],
                else_block: BlockId(2),
                else_args: vec![],
            }),
        };
        let arm = |constant: i64, id: u32| {
            ret_block(
                None,
                id + 1,
                vec![
                    iconst(id, constant, Type::I32),
                    (Some(Value(id + 1)), Inst::Iadd(Value(0), Value(id))),
                ],
            )
        };
        function_named("pick", 1, vec![entry, arm(59, 1), arm(50, 3)])
    }

    fn chain_function() -> Function {
        function_named(
            "chain",
            1,
            vec![entry_with(
                1,
                vec![call_typed(1, "probe", &[], Type::I64)],
                1,
            )],
        )
    }

    // main branches on `minus_one() < 0`; the call's declared type picks the compare width
    fn negative_result_main(ret: Type) -> Function {
        let entry = Block {
            params: vec![],
            insts: vec![
                call_typed(0, "minus_one", &[], ret),
                iconst(1, 0, Type::I32),
                (
                    Some(Value(2)),
                    Inst::Icmp {
                        cond: Cond::Slt,
                        a: Value(0),
                        b: Value(1),
                    },
                ),
            ],
            terminator: Some(Terminator::Branch {
                cond: Value(2),
                then_block: BlockId(1),
                then_args: vec![],
                else_block: BlockId(2),
                else_args: vec![],
            }),
        };
        let arm = |code: i64, id: u32| ret_block(None, id, vec![iconst(id, code, Type::I32)]);
        main_with(vec![entry, arm(71, 3), arm(72, 4)])
    }

    #[test]
    fn types_call_results_by_callee() {
        let helper = "int minus_one(void) { return -1; }\n";
        let source = [("minus_one.c", helper)];
        let (typed, _) = run_with_sources(&[negative_result_main(Type::I32)], &source, &[]);
        let (control, _) = run_with_sources(&[negative_result_main(Type::I64)], &source, &[]);
        assert_eq!(
            (typed, control),
            (71, 72),
            "i32 result compares as w and is negative; the same call typed i64 reads x0 zero-extended"
        );
    }

    #[test]
    fn traps_on_unreachable() {
        let main_ending = |terminator| {
            main_with(vec![Block {
                params: vec![],
                insts: vec![],
                terminator: Some(terminator),
            }])
        };
        let run = |function: &Function| {
            let dir = tempfile::tempdir().expect("tempdir");
            let exe = link(dir.path(), function, &[]);
            run_bounded_status(&exe, &[]).0
        };
        let trapped = run(&main_ending(Terminator::Unreachable));
        let returned = run(&main_with(vec![ret_block(
            None,
            0,
            vec![iconst(0, 7, Type::I32)],
        )]));
        assert_eq!(
            (trapped.signal(), returned.code()),
            (Some(5), Some(7)),
            "unreachable dies by SIGTRAP; the control returning 7 exits 7"
        );
    }

    #[test]
    fn preserves_callee_saved_registers() {
        let functions = [keep_function(), pick_function(), chain_function()];
        let sources = [
            ("saved.s", SAVED_SOURCE),
            ("smash.s", SMASH_SOURCE),
            ("main.c", SAVED_MAIN),
        ];
        let (code, stdout) = run_with_sources(&functions, &sources, &[]);
        let expected = "\
keep changed=0 result=43
pick0 changed=0 result=50
pick1 changed=0 result=60
chain linked=1
unlinked linked=0
clobber changed=2
clobber_fp changed=1
";
        assert_eq!((code, stdout.as_str()), (0, expected));
    }
}
