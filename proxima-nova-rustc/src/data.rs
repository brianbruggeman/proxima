//! Statics and allocations as nova data objects (R22 objects, R23 naming).
//!
//! A static is named by rustc's own mangling; any other allocation a pointer constant reaches
//! (`GlobalAlloc::Memory`) gets a name derived from its `AllocId`, so every use of one allocation
//! names one symbol and the adapter feeds it once.

use rustc_middle::mir::interpret::{AllocId, Allocation, GlobalAlloc};
use rustc_middle::ty::{Instance, TyCtxt};
use rustc_span::def_id::DefId;

use proxima_nova::object::{DataObject, PointerSlot};

use crate::mir_error::MirError;

const POINTER_BYTES: usize = 8;

pub fn static_symbol(tcx: TyCtxt<'_>, def_id: DefId) -> String {
    tcx.symbol_name(Instance::mono(tcx, def_id))
        .name
        .to_string()
}

pub fn memory_symbol(id: AllocId) -> String {
    format!("nova_alloc_{:x}", id.0.get())
}

/// The symbol an allocation is emitted under; a `Memory` allocation is queued in `pending` to be fed.
pub fn alloc_symbol(
    tcx: TyCtxt<'_>,
    id: AllocId,
    pending: &mut Vec<AllocId>,
) -> Result<String, MirError> {
    match tcx.global_alloc(id) {
        GlobalAlloc::Static(def_id) => Ok(static_symbol(tcx, def_id)),
        GlobalAlloc::Memory(_) => {
            pending.push(id);
            Ok(memory_symbol(id))
        }
        GlobalAlloc::Function { .. } => Err(MirError::FnPointer),
        GlobalAlloc::VTable(..) => Err(MirError::VTable),
        GlobalAlloc::TypeId { .. } => Err(MirError::TypeId),
    }
}

/// The data object of a static: the bytes `eval_static_initializer` returns, one slot per pointer.
pub fn static_object(
    tcx: TyCtxt<'_>,
    def_id: DefId,
    pending: &mut Vec<AllocId>,
) -> Result<DataObject, MirError> {
    let allocation =
        tcx.eval_static_initializer(def_id)
            .map_err(|error| MirError::ConstantRead {
                detail: format!("{error:?}"),
            })?;
    object_of(tcx, static_symbol(tcx, def_id), allocation.inner(), pending)
}

/// The data object of a `GlobalAlloc::Memory` allocation under its derived name.
pub fn memory_object(
    tcx: TyCtxt<'_>,
    id: AllocId,
    pending: &mut Vec<AllocId>,
) -> Result<DataObject, MirError> {
    match tcx.global_alloc(id) {
        GlobalAlloc::Memory(allocation) => {
            object_of(tcx, memory_symbol(id), allocation.inner(), pending)
        }
        _ => Err(MirError::ConstantRead {
            detail: format!("allocation {id:?} is not memory"),
        }),
    }
}

fn object_of(
    tcx: TyCtxt<'_>,
    symbol: String,
    allocation: &Allocation,
    pending: &mut Vec<AllocId>,
) -> Result<DataObject, MirError> {
    let bytes = allocation
        .inspect_with_uninit_and_ptr_outside_interpreter(0..allocation.len())
        .to_vec();
    let mut pointers = Vec::new();
    for (offset, provenance) in allocation.provenance().ptrs().iter() {
        let start = usize::try_from(offset.bytes()).map_err(|_| MirError::SlotOffset)?;
        let held = bytes
            .get(start..start + POINTER_BYTES)
            .ok_or(MirError::SlotOffset)?;
        if held.iter().any(|byte| *byte != 0) {
            return Err(MirError::SlotOffset);
        }
        pointers.push(PointerSlot {
            offset: offset.bytes(),
            symbol: alloc_symbol(tcx, provenance.alloc_id(), pending)?,
        });
    }
    Ok(DataObject {
        symbol,
        bytes,
        align: allocation.align.bytes(),
        pointers,
    })
}
