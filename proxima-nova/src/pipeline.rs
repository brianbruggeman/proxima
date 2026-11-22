//! The compiler stages as proxima pipes: `lower` -> `encode`, one function at a time.
//!
//! Each stage is a unit struct implementing [`Pipe`]; [`backend`] composes
//! them with [`PipeExt::and_then`]. All stages share [`ObjectError`] so the
//! `Next::Err: From<Self::Err>` bound of `and_then` holds by reflexivity.
use alloc::string::String;
use alloc::vec::Vec;

use proxima_primitives::PipeExt;
use proxima_primitives::pipe::{AndThen, Pipe};

use crate::aarch64::{MachInst, encode, lower_function};
use crate::ir::Function;
use crate::object::ObjectError;

/// How the linker patches a relocated word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelocationKind {
    /// the imm26 of a `bl`/`b`: Mach-O `ARM64_RELOC_BRANCH26`, pc-relative
    Branch26,
    /// the imm21 of an `adrp`: Mach-O `ARM64_RELOC_PAGE21`, pc-relative
    Page21,
    /// the imm12 of the `add` that follows an `adrp`: Mach-O `ARM64_RELOC_PAGEOFF12`
    PageOff12,
}

/// A word of the code that names a symbol the linker resolves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Relocation {
    /// byte offset of the instruction within the function's code
    pub offset: u64,
    pub kind: RelocationKind,
    /// the target symbol, un-prefixed (`square`, not `_square`)
    pub symbol: String,
}

/// A function after lowering: symbol name, machine instructions and the call sites to relocate.
#[derive(Debug, Clone, PartialEq)]
pub struct MachFunction {
    pub name: String,
    pub insts: Vec<MachInst>,
    pub relocations: Vec<Relocation>,
}

/// A function after encoding: symbol name, little-endian code bytes and their relocations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Encoded {
    pub name: String,
    pub code: Vec<u8>,
    pub relocations: Vec<Relocation>,
}

/// One function in, its machine instructions out.
pub struct LowerStage;
/// One lowered function in, its code bytes out.
pub struct EncodeStage;

impl Pipe for LowerStage {
    type In = Function;
    type Out = MachFunction;
    type Err = ObjectError;

    async fn call(&self, function: Self::In) -> Result<Self::Out, Self::Err> {
        let lowered = lower_function(&function).map_err(|source| ObjectError::Lower {
            name: function.name.clone(),
            source,
        })?;
        Ok(MachFunction {
            name: function.name,
            insts: lowered.insts,
            relocations: lowered.relocations,
        })
    }
}

impl Pipe for EncodeStage {
    type In = MachFunction;
    type Out = Encoded;
    type Err = ObjectError;

    async fn call(&self, function: Self::In) -> Result<Self::Out, Self::Err> {
        let code = function
            .insts
            .iter()
            .map(|inst| encode(*inst).map(u32::to_le_bytes))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|source| ObjectError::Encode {
                name: function.name.clone(),
                source,
            })?
            .concat();
        Ok(Encoded {
            name: function.name,
            code,
            relocations: function.relocations,
        })
    }
}

/// The per-function backend: a function in, its encoded code out. The concrete type (not `impl Pipe`)
/// keeps `Send` visible, which a worker pool needs; feed the results to
/// [`crate::object::ObjectBuilder`] in function order.
pub fn backend() -> AndThen<LowerStage, EncodeStage> {
    LowerStage.and_then(EncodeStage)
}

#[cfg(test)]
mod tests {
    extern crate std;

    use alloc::string::ToString;
    use alloc::vec;
    use prime::PrimeRuntime;
    use std::{fs, process::Command};

    use super::*;
    use crate::ir::{Block, BlockId, Inst, Terminator, Type, Value};
    use crate::object::ObjectBuilder;

    fn main_returning_42() -> Function {
        let entry = Block {
            params: vec![],
            insts: vec![
                (
                    Some(Value(0)),
                    Inst::Iconst {
                        ty: Type::I32,
                        imm: 40,
                    },
                ),
                (
                    Some(Value(1)),
                    Inst::Iconst {
                        ty: Type::I32,
                        imm: 2,
                    },
                ),
                (Some(Value(2)), Inst::Iadd(Value(0), Value(1))),
            ],
            terminator: Some(Terminator::Jump {
                target: BlockId(1),
                args: vec![Value(2)],
            }),
        };
        let exit = Block {
            params: vec![(Value(3), Type::I32)],
            insts: vec![],
            terminator: Some(Terminator::Return(Some(Value(3)))),
        };
        Function {
            name: "main".to_string(),
            params: vec![],
            returns: Some(Type::I32),
            slots: vec![],
            blocks: vec![entry, exit],
        }
    }

    #[test]
    fn compiles_on_prime() {
        let runtime = PrimeRuntime::new(1).expect("prime runtime");
        let pipeline = backend();
        let encoded = runtime
            .block_on(async move { pipeline.call(main_returning_42()).await })
            .expect("block_on dispatch")
            .expect("compile");
        let mut builder = ObjectBuilder::new(0x000B_0000);
        builder.feed(encoded).expect("feed");
        let bytes = builder.finish().expect("object");

        let dir = tempfile::tempdir().expect("tempdir");
        let obj = dir.path().join("main.o");
        let exe = dir.path().join("main");
        fs::write(&obj, bytes).expect("write .o");
        let link = Command::new("cc")
            .arg(&obj)
            .arg("-o")
            .arg(&exe)
            .output()
            .expect("cc");
        assert!(link.status.success(), "cc failed");
        let code = Command::new(&exe)
            .status()
            .expect("run")
            .code()
            .expect("exit code");
        assert_eq!(code, 42);
    }
}
