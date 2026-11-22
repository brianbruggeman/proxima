//! Lowers one internal `rustc_middle` MIR body to a nova [`Function`].
//!
//! Every non-zero-sized MIR local lives in a stack slot (spec decision "MIR locals"): the slots are
//! declared on the function, and each access takes the slot's address with a `SlotAddr` right before
//! its `Load` or `Store`, so no address outlives its use. Block 0 is a prologue that stores the
//! arguments into their slots, MIR bbN is nova block N+1. Anything outside the supported subset is
//! a typed refusal carrying the offending MIR text, never a guess. The body is the generic
//! `instance_mir` body (as rustc_codegen_cranelift reads it); every type goes through [`MonoTy`]
//! before a layout or a field is read.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use rustc_abi::{FieldsShape, Integer, Primitive, Scalar, TagEncoding, VariantIdx, Variants};
use rustc_attr_ir::lang_items::LangItem;
use rustc_middle::mir::interpret::{AllocId, CtfeProvenance, Pointer, Scalar as ConstScalar};
use rustc_middle::mir::{
    AggregateKind, AssertKind, BasicBlockData, BinOp, Body, CastKind, Const, ConstValue, Operand,
    Place, ProjectionElem, Rvalue, SourceInfo, StatementKind, SwitchTargets,
    Terminator as MirTerminator, TerminatorKind, UnwindAction,
};
use rustc_middle::ty::layout::TyAndLayout;
use rustc_middle::ty::print::with_forced_trimmed_paths;
use rustc_middle::ty::{
    EarlyBinder, Instance, InstanceKind, IntTy, Ty, TyCtxt, TyKind, TypingEnv, UintTy,
};
use rustc_public::rustc_internal;
use rustc_span::def_id::DefId;
use rustc_span::{Span, Symbol};

use proxima_nova::ir::{
    Block, BlockId, Cond, Function, Inst, Slot, SlotId, Terminator, Type, Value,
};

use crate::data;
use crate::mir_error::MirError;

// the stable conversion is crate-private to rustc_public, so refusal text is rendered where the type is
// stable: inside a throwaway `run`, only when a refusal or a listing prints it
macro_rules! refusal_text {
    ($tcx:expr, $item:expr) => {
        rustc_internal::run($tcx, || format!("{:?}", $item))
            .unwrap_or_else(|_| String::from("<unrenderable>"))
    };
}

/// a type that went through [`Lowerer::mono_ty`]; every layout, field and constant read takes one
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct MonoTy<'tcx>(Ty<'tcx>);

fn ty_text(tcx: TyCtxt<'_>, ty: MonoTy<'_>) -> String {
    refusal_text!(tcx, rustc_internal::stable(ty.0).kind())
}

fn place_text<'tcx>(tcx: TyCtxt<'tcx>, place: &Place<'tcx>) -> String {
    refusal_text!(tcx, rustc_internal::stable(place))
}

fn operand_text<'tcx>(tcx: TyCtxt<'tcx>, operand: &Operand<'tcx>) -> String {
    refusal_text!(tcx, rustc_internal::stable(operand))
}

fn rvalue_text<'tcx>(tcx: TyCtxt<'tcx>, rvalue: &Rvalue<'tcx>) -> String {
    refusal_text!(tcx, rustc_internal::stable(rvalue))
}

fn def_text(tcx: TyCtxt<'_>, def: DefId) -> String {
    with_forced_trimmed_paths!(tcx.def_path_str(def))
}

/// one struct field: its nova type and its byte offset in rustc's layout, which may reorder fields
#[derive(Clone)]
struct FieldSlot {
    name: Symbol,
    ty: Type,
    offset: u32,
}

/// a struct local's slot size and alignment, with its fields in declaration order
#[derive(Clone)]
struct StructLayout {
    name: Symbol,
    size: u32,
    align: u32,
    fields: Vec<FieldSlot>,
}

/// one enum variant: the value its tag holds and its fields at rustc's offsets
#[derive(Clone)]
struct VariantSlots {
    name: Symbol,
    idx: VariantIdx,
    discriminant: u128,
    fields: Vec<FieldSlot>,
}

/// a direct-tag enum local's slot; the tag sits at offset 0 and every variant's fields follow rustc's layout
#[derive(Clone)]
struct EnumLayout {
    def: DefId,
    size: u32,
    align: u32,
    tag: Type,
    tag_signed: bool,
    variants: Vec<VariantSlots>,
}

/// the nova representation of a MIR type: zero-sized types get no slot
#[derive(Clone)]
enum Repr {
    Zst,
    Scalar { ty: Type, size: u32, align: u32 },
    Struct(StructLayout),
    Enum(EnumLayout),
}

fn type_bits(ty: Type) -> u32 {
    match ty {
        Type::I8 => 8,
        Type::I16 => 16,
        Type::I32 => 32,
        Type::I64 | Type::Ptr => 64,
    }
}

#[derive(Clone, Copy)]
struct LocalSlot {
    id: SlotId,
    ty: Type,
}

/// what a local's MIR type means to a compare or branch; nova values carry no signedness
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Signed,
    Unsigned,
    Bool,
    Other,
}

fn kind_of(ty: MonoTy<'_>) -> Kind {
    match ty.0.kind() {
        TyKind::Int(IntTy::I32 | IntTy::I64 | IntTy::Isize) => Kind::Signed,
        TyKind::Uint(UintTy::U32 | UintTy::U64 | UintTy::Usize) => Kind::Unsigned,
        TyKind::Bool => Kind::Bool,
        _ => Kind::Other,
    }
}

fn is_reference(ty: MonoTy<'_>) -> bool {
    matches!(ty.0.kind(), TyKind::Ref(..))
}

fn local_of(place: &Place<'_>) -> usize {
    place.local.as_usize()
}

// a private copy of rustc_public_bridge context/impls.rs:81-104 (instance_has_body, item_has_body)
#[must_use]
pub fn has_body<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> bool {
    let def_id = instance.def_id();
    item_has_body(tcx, def_id)
        || !matches!(
            instance.def,
            InstanceKind::Virtual(..) | InstanceKind::Intrinsic(..) | InstanceKind::Item(..)
        )
}

fn item_has_body(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    let must_override = tcx
        .intrinsic(def_id)
        .is_some_and(|intrinsic| intrinsic.must_be_overridden);
    !must_override && tcx.is_mir_available(def_id) && !tcx.is_trivial_const(def_id)
}

// the instance whose body to read: an intrinsic's fallback body is its item (bridge builder.rs:18-23)
#[must_use]
pub fn body_instance<'tcx>(instance: Instance<'tcx>) -> Instance<'tcx> {
    match instance.def {
        InstanceKind::Intrinsic(def_id) => Instance::new_raw(def_id, instance.args),
        _ => instance,
    }
}

struct Lowerer<'tcx> {
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    body: &'tcx Body<'tcx>,
    monos: HashMap<Ty<'tcx>, MonoTy<'tcx>>,
    reprs: HashMap<MonoTy<'tcx>, Repr>,
    slots: Vec<Option<LocalSlot>>,
    // a struct local is in `structs`, never in `slots`, so no scalar access can reach it
    structs: Vec<Option<(SlotId, StructLayout)>>,
    enums: Vec<Option<(SlotId, EnumLayout)>>,
    frame: Vec<Slot>,
    tys: Vec<MonoTy<'tcx>>,
    next_value: u32,
    next_block: u32,
    extra: Vec<Block>,
    // memory allocations the constants name, fed after the functions
    allocs: Vec<AllocId>,
}

type Insts = Vec<(Option<Value>, Inst)>;

impl<'tcx> Lowerer<'tcx> {
    // the only constructor of MonoTy: the instance's args applied to one type, not to the body
    fn mono_ty(&mut self, ty: Ty<'tcx>) -> MonoTy<'tcx> {
        if let Some(mono) = self.monos.get(&ty) {
            return *mono;
        }
        let resolved = self.instance.instantiate_mir_and_normalize_erasing_regions(
            self.tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(self.tcx, ty),
        );
        let mono = MonoTy(resolved);
        self.monos.insert(ty, mono);
        mono
    }

    fn layout(&self, ty: MonoTy<'tcx>) -> Result<TyAndLayout<'tcx>, MirError> {
        self.tcx
            .layout_of(TypingEnv::fully_monomorphized().as_query_input(ty.0))
            .map_err(|error| MirError::Layout {
                ty: ty_text(self.tcx, ty),
                detail: format!("{error:?}"),
            })
    }

    fn size_align(
        &self,
        ty: MonoTy<'tcx>,
        layout: &TyAndLayout<'tcx>,
    ) -> Result<(u32, u32), MirError> {
        let overflow = || MirError::SizeOverflow {
            ty: ty_text(self.tcx, ty),
        };
        let size = u32::try_from(layout.size.bytes()).map_err(|_| overflow())?;
        let align = u32::try_from(layout.align.bytes()).map_err(|_| overflow())?;
        Ok((size, align))
    }

    fn variant_field_type(&self, ty: MonoTy<'tcx>) -> Result<Type, MirError> {
        match ty.0.kind() {
            TyKind::Int(IntTy::I32) | TyKind::Uint(UintTy::U32) => Ok(Type::I32),
            TyKind::Bool => Ok(Type::I8),
            _ => Err(MirError::UnsupportedType {
                ty: ty_text(self.tcx, ty),
            }),
        }
    }

    fn field_ty(
        &mut self,
        field: &rustc_middle::ty::FieldDef,
        args: rustc_middle::ty::GenericArgs<'tcx>,
    ) -> MonoTy<'tcx> {
        let declared = field.ty(self.tcx, args).skip_norm_wip();
        self.mono_ty(declared)
    }

    // the tag is read from the layout; a niche-encoded enum has no tag of its own and is refused
    fn enum_repr(
        &mut self,
        ty: MonoTy<'tcx>,
        layout: &TyAndLayout<'tcx>,
    ) -> Result<Repr, MirError> {
        let TyKind::Adt(def, args) = ty.0.kind() else {
            return Err(MirError::UnsupportedType {
                ty: ty_text(self.tcx, ty),
            });
        };
        let name = def_text(self.tcx, def.did());
        let Variants::Multiple {
            tag,
            tag_encoding,
            variants: placed,
            ..
        } = &layout.variants
        else {
            return Err(MirError::UnsupportedType { ty: name });
        };
        if !matches!(tag_encoding, TagEncoding::Direct) {
            return Err(MirError::NicheEncoding);
        }
        let Scalar::Initialized {
            value: Primitive::Int(integer, signed),
            ..
        } = tag
        else {
            return Err(MirError::UnsupportedType { ty: name });
        };
        let tag_ty = match integer {
            Integer::I8 => Type::I8,
            Integer::I16 => Type::I16,
            Integer::I32 => Type::I32,
            Integer::I64 => Type::I64,
            Integer::I128 => return Err(MirError::UnsupportedType { ty: name }),
        };
        let declared = def.variants();
        if declared.len() != placed.len() {
            return Err(MirError::EnumShape {
                ty: name,
                variants: declared.len(),
                placed: placed.len(),
            });
        }
        let mut variants = Vec::with_capacity(declared.len());
        for ((idx, variant), laid) in declared.iter_enumerated().zip(placed.iter()) {
            let discriminant =
                ty.0.discriminant_for_variant(self.tcx, idx)
                    .map(|discr| discr.val)
                    .ok_or_else(|| MirError::UnsupportedType { ty: name.clone() })?;
            if variant.fields.len() != laid.field_offsets.len() {
                return Err(MirError::StructShape {
                    ty: variant.name.to_string(),
                    fields: variant.fields.len(),
                    offsets: laid.field_offsets.len(),
                });
            }
            let mut fields = Vec::with_capacity(variant.fields.len());
            for (field, offset) in variant.fields.iter().zip(laid.field_offsets.iter()) {
                let field_ty = self.field_ty(field, *args);
                fields.push(FieldSlot {
                    name: field.name,
                    ty: self.variant_field_type(field_ty)?,
                    offset: u32::try_from(offset.bytes())
                        .map_err(|_| MirError::SizeOverflow { ty: name.clone() })?,
                });
            }
            variants.push(VariantSlots {
                name: variant.name,
                idx,
                discriminant,
                fields,
            });
        }
        let (size, align) = self.size_align(ty, layout)?;
        Ok(Repr::Enum(EnumLayout {
            def: def.did(),
            size,
            align,
            tag: tag_ty,
            tag_signed: *signed,
            variants,
        }))
    }

    // FieldsShape::Arbitrary offsets are indexed by source field order, the same order `fields` has
    fn struct_repr(
        &mut self,
        ty: MonoTy<'tcx>,
        layout: &TyAndLayout<'tcx>,
    ) -> Result<Repr, MirError> {
        let TyKind::Adt(def, args) = ty.0.kind() else {
            return Err(MirError::UnsupportedType {
                ty: ty_text(self.tcx, ty),
            });
        };
        let variant = def.non_enum_variant();
        let name = variant.name.to_string();
        let FieldsShape::Arbitrary { offsets, .. } = &layout.fields else {
            return Err(MirError::UnsupportedType { ty: name });
        };
        if offsets.len() != variant.fields.len() {
            return Err(MirError::StructShape {
                ty: name,
                fields: variant.fields.len(),
                offsets: offsets.len(),
            });
        }
        let mut fields = Vec::with_capacity(offsets.len());
        for (field, offset) in variant.fields.iter().zip(offsets.iter()) {
            let field_ty = self.field_ty(field, *args);
            let field_repr = self.repr(field_ty)?;
            let Repr::Scalar { ty: scalar, .. } = field_repr else {
                return Err(match field_repr {
                    Repr::Struct(_) => MirError::NestedStruct {
                        ty: name,
                        field: field.name.to_string(),
                    },
                    _ => MirError::UnsupportedType {
                        ty: ty_text(self.tcx, field_ty),
                    },
                });
            };
            if kind_of(field_ty) == Kind::Other {
                return Err(MirError::UnsupportedType {
                    ty: ty_text(self.tcx, field_ty),
                });
            }
            fields.push(FieldSlot {
                name: field.name,
                ty: scalar,
                offset: u32::try_from(offset.bytes())
                    .map_err(|_| MirError::SizeOverflow { ty: name.clone() })?,
            });
        }
        let (size, align) = self.size_align(ty, layout)?;
        Ok(Repr::Struct(StructLayout {
            name: variant.name,
            size,
            align,
            fields,
        }))
    }

    // layout_of runs once per type: the result is cached by MonoTy
    fn repr(&mut self, ty: MonoTy<'tcx>) -> Result<Repr, MirError> {
        if let Some(cached) = self.reprs.get(&ty) {
            return Ok(cached.clone());
        }
        let built = self.build_repr(ty)?;
        self.reprs.insert(ty, built.clone());
        Ok(built)
    }

    fn build_repr(&mut self, ty: MonoTy<'tcx>) -> Result<Repr, MirError> {
        let scalar = match ty.0.kind() {
            TyKind::Adt(def, _) if def.is_struct() => {
                let layout = self.layout(ty)?;
                return self.struct_repr(ty, &layout);
            }
            TyKind::Adt(def, _) if def.is_enum() => {
                let layout = self.layout(ty)?;
                return self.enum_repr(ty, &layout);
            }
            TyKind::Never => return Ok(Repr::Zst),
            TyKind::Tuple(items) if items.is_empty() => return Ok(Repr::Zst),
            TyKind::Bool => Type::I8,
            TyKind::Int(IntTy::I32) | TyKind::Uint(UintTy::U32) => Type::I32,
            TyKind::Int(IntTy::I64 | IntTy::Isize) | TyKind::Uint(UintTy::U64 | UintTy::Usize) => {
                Type::I64
            }
            TyKind::RawPtr(..) | TyKind::Ref(..) => Type::Ptr,
            _ => {
                return Err(MirError::UnsupportedType {
                    ty: ty_text(self.tcx, ty),
                });
            }
        };
        let layout = self.layout(ty)?;
        let (size, align) = self.size_align(ty, &layout)?;
        // a fat pointer is a reference with a wider layout than one nova Ptr
        if scalar == Type::Ptr && size != 8 {
            return Err(MirError::FatPointer {
                ty: ty_text(self.tcx, ty),
            });
        }
        Ok(Repr::Scalar {
            ty: scalar,
            size,
            align,
        })
    }

    fn place_mono_ty(&mut self, place: &Place<'tcx>) -> MonoTy<'tcx> {
        let body = self.body;
        let raw = place.ty(&body.local_decls, self.tcx).ty;
        self.mono_ty(raw)
    }

    fn unsupported_place(&self, place: &Place<'tcx>) -> MirError {
        MirError::UnsupportedOperand {
            operand: place_text(self.tcx, place),
        }
    }

    fn fresh(&mut self) -> Value {
        let value = Value(self.next_value);
        self.next_value += 1;
        value
    }

    // ids of extra blocks follow the MIR blocks; they are pushed to `extra` in the order reserved
    fn reserve_block(&mut self) -> BlockId {
        let id = BlockId(self.next_block);
        self.next_block += 1;
        id
    }

    fn enum_of(&self, local: usize) -> Option<&(SlotId, EnumLayout)> {
        self.enums.get(local).and_then(Option::as_ref)
    }

    // `(local as Variant).i` addresses the variant's own field offset, never the tag's
    fn variant_field_of(
        &self,
        place: &Place<'tcx>,
        variant: VariantIdx,
        index: usize,
    ) -> Result<Option<(SlotId, Type, u32)>, MirError> {
        let Some((id, layout)) = self.enum_of(local_of(place)) else {
            return Ok(None);
        };
        let field = layout
            .variants
            .iter()
            .find(|candidate| candidate.idx == variant)
            .and_then(|candidate| candidate.fields.get(index))
            .ok_or_else(|| self.unsupported_place(place))?;
        Ok(Some((*id, field.ty, field.offset)))
    }

    // the (slot, type, byte offset) a place addresses when it is one field of a struct or enum local
    fn field_of(&self, place: &Place<'tcx>) -> Result<Option<(SlotId, Type, u32)>, MirError> {
        let index = match &place.projection[..] {
            [
                ProjectionElem::Downcast(_, variant),
                ProjectionElem::Field(index, _),
            ] => {
                return self.variant_field_of(place, *variant, index.as_usize());
            }
            [ProjectionElem::Field(index, _)] => index.as_usize(),
            _ => return Ok(None),
        };
        let Some(Some((id, layout))) = self.structs.get(local_of(place)) else {
            return Ok(None);
        };
        let field = layout
            .fields
            .get(index)
            .ok_or_else(|| self.unsupported_place(place))?;
        Ok(Some((*id, field.ty, field.offset)))
    }

    // `(*p)` or `((*p).i)` for a shared or mutable reference local `p`: the pointer's slot, the
    // accessed nova type, its byte offset past the pointer's value, and its MIR type
    fn deref_of(
        &mut self,
        place: &Place<'tcx>,
    ) -> Result<Option<(LocalSlot, Type, u32, MonoTy<'tcx>)>, MirError> {
        let field = match &place.projection[..] {
            [ProjectionElem::Deref] => None,
            [
                ProjectionElem::Deref,
                ProjectionElem::Field(index, field_ty),
            ] => Some((index.as_usize(), *field_ty)),
            _ => return Ok(None),
        };
        let pointer = self
            .slot_of(local_of(place))?
            .filter(|slot| slot.ty == Type::Ptr)
            .ok_or_else(|| self.unsupported_place(place))?;
        let TyKind::Ref(_, pointee, _) = self.tys[local_of(place)].0.kind() else {
            return Err(self.unsupported_place(place));
        };
        let pointee = self.mono_ty(*pointee);
        match field {
            None => {
                let Repr::Scalar { ty, .. } = self.repr(pointee)? else {
                    return Err(self.unsupported_place(place));
                };
                if kind_of(pointee) == Kind::Other && !is_reference(pointee) {
                    return Err(self.unsupported_place(place));
                }
                Ok(Some((pointer, ty, 0, pointee)))
            }
            Some((index, field_ty)) => {
                let Repr::Struct(layout) = self.repr(pointee)? else {
                    return Err(self.unsupported_place(place));
                };
                let found = layout
                    .fields
                    .get(index)
                    .ok_or_else(|| self.unsupported_place(place))?;
                let field_ty = self.mono_ty(field_ty);
                Ok(Some((pointer, found.ty, found.offset, field_ty)))
            }
        }
    }

    // `(*p)[i]` for a reference local `p` whose pointee is an array with a scalar element: the
    // pointer's slot, the element's nova type, the stride and the local holding the index
    fn index_of(
        &mut self,
        place: &Place<'tcx>,
    ) -> Result<Option<(LocalSlot, Type, u32, usize)>, MirError> {
        let [ProjectionElem::Deref, ProjectionElem::Index(index)] = &place.projection[..] else {
            return Ok(None);
        };
        let pointer = self
            .slot_of(local_of(place))?
            .filter(|slot| slot.ty == Type::Ptr)
            .ok_or_else(|| self.unsupported_place(place))?;
        let TyKind::Ref(_, pointee, _) = self.tys[local_of(place)].0.kind() else {
            return Err(self.unsupported_place(place));
        };
        let pointee = self.mono_ty(*pointee);
        let TyKind::Array(element, _) = pointee.0.kind() else {
            return Err(self.unsupported_place(place));
        };
        let element = self.mono_ty(*element);
        let Repr::Scalar { ty, .. } = self.repr(element)? else {
            return Err(self.unsupported_place(place));
        };
        let layout = self.layout(pointee)?;
        let FieldsShape::Array { stride, .. } = &layout.fields else {
            return Err(self.unsupported_place(place));
        };
        let stride = u32::try_from(stride.bytes()).map_err(|_| self.unsupported_place(place))?;
        Ok(Some((pointer, ty, stride, index.as_usize())))
    }

    fn operand_ty(&mut self, operand: &Operand<'tcx>) -> Result<MonoTy<'tcx>, MirError> {
        match operand {
            Operand::Copy(place) | Operand::Move(place) => {
                if self.deref_of(place)?.is_some()
                    || self.index_of(place)?.is_some()
                    || place.projection.is_empty()
                    || self.field_of(place)?.is_some()
                {
                    Ok(self.place_mono_ty(place))
                } else {
                    Err(MirError::UnsupportedOperand {
                        operand: operand_text(self.tcx, operand),
                    })
                }
            }
            Operand::Constant(constant) => Ok(self.mono_ty(constant.const_.ty())),
            other => Err(MirError::UnsupportedOperand {
                operand: operand_text(self.tcx, other),
            }),
        }
    }

    fn operand_kind(&mut self, operand: &Operand<'tcx>) -> Result<Kind, MirError> {
        Ok(kind_of(self.operand_ty(operand)?))
    }

    fn operand_value(
        &mut self,
        operand: &Operand<'tcx>,
        insts: &mut Insts,
    ) -> Result<Value, MirError> {
        match operand {
            Operand::Copy(place) | Operand::Move(place) => self.load_place(place, insts),
            Operand::Constant(constant) => {
                let ty = self.mono_ty(constant.const_.ty());
                let Repr::Scalar { ty, .. } = self.repr(ty)? else {
                    return Err(MirError::UnsupportedOperand {
                        operand: operand_text(self.tcx, operand),
                    });
                };
                self.constant_value(constant.const_, ty, insts)
            }
            other => Err(MirError::UnsupportedOperand {
                operand: operand_text(self.tcx, other),
            }),
        }
    }

    fn load_place(&mut self, place: &Place<'tcx>, insts: &mut Insts) -> Result<Value, MirError> {
        if let Some((id, ty, offset)) = self.field_of(place)? {
            return Ok(self.load_at(id, ty, offset, insts));
        }
        if let Some((pointer, ty, offset, _)) = self.deref_of(place)? {
            let addr = self.load_slot(pointer, insts);
            let value = self.fresh();
            insts.push((Some(value), Inst::Load { ty, addr, offset }));
            return Ok(value);
        }
        if let Some((pointer, ty, stride, index)) = self.index_of(place)? {
            let ptr = self.load_slot(pointer, insts);
            let index_slot = self
                .slot_of(index)?
                .ok_or_else(|| self.unsupported_place(place))?;
            let index = self.load_slot(index_slot, insts);
            let scale = self.fresh();
            insts.push((
                Some(scale),
                Inst::Iconst {
                    ty: Type::I64,
                    imm: i64::from(stride),
                },
            ));
            let offset = self.fresh();
            insts.push((Some(offset), Inst::Imul(index, scale)));
            let addr = self.fresh();
            insts.push((Some(addr), Inst::Iadd(ptr, offset)));
            let value = self.fresh();
            insts.push((
                Some(value),
                Inst::Load {
                    ty,
                    addr,
                    offset: 0,
                },
            ));
            return Ok(value);
        }
        let local = local_of(place);
        if !place.projection.is_empty()
            || self.structs[local].is_some()
            || self.enums[local].is_some()
        {
            return Err(self.unsupported_place(place));
        }
        let slot = self
            .slot_of(local)?
            .ok_or_else(|| MirError::UnsupportedOperand {
                operand: format!("zero-sized {}", place_text(self.tcx, place)),
            })?;
        Ok(self.load_slot(slot, insts))
    }

    // a constant is evaluated only when read; the generic body's unevaluated constants are
    // instantiated first, as cranelift's monomorphize does
    fn const_value(&self, constant: Const<'tcx>) -> Result<ConstValue, MirError> {
        if let Const::Val(value, _) = constant {
            return Ok(value);
        }
        let env = TypingEnv::fully_monomorphized();
        let resolved = self.instance.instantiate_mir_and_normalize_erasing_regions(
            self.tcx,
            env,
            EarlyBinder::bind(self.tcx, constant),
        );
        resolved
            .eval(self.tcx, env, rustc_span::DUMMY_SP)
            .map_err(|error| MirError::ConstantRead {
                detail: format!("{error:?}"),
            })
    }

    fn constant_value(
        &mut self,
        constant: Const<'tcx>,
        ty: Type,
        insts: &mut Insts,
    ) -> Result<Value, MirError> {
        let value = self.const_value(constant)?;
        let ConstValue::Scalar(scalar) = value else {
            return Err(MirError::UnsupportedConstant {
                kind: refusal_text!(self.tcx, rustc_internal::stable(constant).kind()),
            });
        };
        let int = match scalar {
            ConstScalar::Int(int) => int,
            ConstScalar::Ptr(pointer, _) => return self.pointer_constant(pointer, ty, insts),
        };
        let bits = int.to_bits_unchecked();
        let wide = i128::try_from(bits).map_err(|_| MirError::ConstantRead {
            detail: format!("{bits} does not fit i128"),
        })?;
        let imm = if ty == Type::I64 {
            low_64_bits(wide)
        } else {
            i64::try_from(wide).map_err(|_| MirError::ConstantTooWide { value: wide })?
        };
        let result = self.fresh();
        insts.push((Some(result), Inst::Iconst { ty, imm }));
        Ok(result)
    }

    // codegen_ssa block.rs:2167-2176 with caller_location None: the span's Location allocation as a DataAddr
    fn caller_location(
        &mut self,
        source_info: SourceInfo,
        insts: &mut Insts,
    ) -> Result<Value, MirError> {
        let tcx = self.tcx;
        let location =
            self.body
                .caller_location_span(source_info, None::<ConstValue>, tcx, |span| {
                    tcx.span_as_caller_location(span)
                });
        match location {
            ConstValue::Scalar(ConstScalar::Ptr(pointer, _)) => {
                self.pointer_constant(pointer, Type::Ptr, insts)
            }
            other => Err(MirError::UnsupportedConstant {
                kind: format!("caller location {other:?}"),
            }),
        }
    }

    // a pointer constant is the address of its allocation's symbol plus the pointer's offset
    fn pointer_constant(
        &mut self,
        pointer: Pointer<CtfeProvenance>,
        ty: Type,
        insts: &mut Insts,
    ) -> Result<Value, MirError> {
        if ty != Type::Ptr {
            return Err(MirError::UnsupportedConstant {
                kind: format!("pointer constant of type {ty:?}"),
            });
        }
        let (provenance, offset) = pointer.prov_and_relative_offset();
        let symbol = data::alloc_symbol(self.tcx, provenance.alloc_id(), &mut self.allocs)?;
        let base = self.fresh();
        insts.push((Some(base), Inst::DataAddr { symbol }));
        if offset.bytes() == 0 {
            return Ok(base);
        }
        let imm = i64::try_from(offset.bytes()).map_err(|_| MirError::ConstantTooWide {
            value: i128::from(offset.bytes()),
        })?;
        let delta = self.fresh();
        insts.push((Some(delta), Inst::Iconst { ty: Type::I64, imm }));
        let moved = self.fresh();
        insts.push((Some(moved), Inst::Iadd(base, delta)));
        Ok(moved)
    }

    // the address is rematerialized right before each access, never carried across instructions
    fn slot_addr(&mut self, id: SlotId, insts: &mut Insts) -> Value {
        let addr = self.fresh();
        insts.push((Some(addr), Inst::SlotAddr(id)));
        addr
    }

    fn load_at(&mut self, id: SlotId, ty: Type, offset: u32, insts: &mut Insts) -> Value {
        let addr = self.slot_addr(id, insts);
        let value = self.fresh();
        insts.push((Some(value), Inst::Load { ty, addr, offset }));
        value
    }

    fn store_at(&mut self, id: SlotId, offset: u32, value: Value, insts: &mut Insts) {
        let addr = self.slot_addr(id, insts);
        insts.push((
            None,
            Inst::Store {
                value,
                addr,
                offset,
            },
        ));
    }

    fn load_slot(&mut self, slot: LocalSlot, insts: &mut Insts) -> Value {
        self.load_at(slot.id, slot.ty, 0, insts)
    }

    fn store_slot(&mut self, slot: LocalSlot, value: Value, insts: &mut Insts) {
        self.store_at(slot.id, 0, value, insts);
    }

    fn slot_of(&self, local: usize) -> Result<Option<LocalSlot>, MirError> {
        self.slots
            .get(local)
            .copied()
            .ok_or(MirError::LocalOutOfRange { local })
    }
}

// a scalar constant arrives zero-extended, so a negative i64 or a u64 above i64::MAX is a large positive value
fn low_64_bits(wide: i128) -> i64 {
    let bytes = wide.to_le_bytes();
    let mut low = [0u8; 8];
    low.copy_from_slice(&bytes[..8]);
    i64::from_le_bytes(low)
}

// by-value structs cross no call boundary: the callee side is refused here, the call side in lower_call
fn refuse_struct_boundary(
    tcx: TyCtxt<'_>,
    reprs: &[Repr],
    arg_count: usize,
) -> Result<(), MirError> {
    if let Some(name) = reprs.first().and_then(|repr| aggregate_name(tcx, repr)) {
        return Err(MirError::StructResult { ty: name });
    }
    match reprs[1..=arg_count]
        .iter()
        .find_map(|repr| aggregate_name(tcx, repr))
    {
        Some(name) => Err(MirError::StructArgument { ty: name }),
        None => Ok(()),
    }
}

fn aggregate_name(tcx: TyCtxt<'_>, repr: &Repr) -> Option<String> {
    match repr {
        Repr::Struct(layout) => Some(layout.name.to_string()),
        Repr::Enum(layout) => Some(def_text(tcx, layout.def)),
        Repr::Zst | Repr::Scalar { .. } => None,
    }
}

// one stdout line per struct local; size and align come back out of the lowered function's slot table
fn layout_lines(function: &Function, structs: &[Option<(SlotId, StructLayout)>]) -> Vec<String> {
    structs
        .iter()
        .flatten()
        .filter_map(|(id, layout)| {
            let slot = function.slots.get(id.0 as usize)?;
            let fields: Vec<String> = layout
                .fields
                .iter()
                .map(|field| format!("{}@{}", field.name, field.offset))
                .collect();
            Some(format!(
                "nova_layout type={} slot_size={} slot_align={} fields={}",
                layout.name,
                slot.size,
                slot.align,
                fields.join(",")
            ))
        })
        .collect()
}

fn type_bytes(ty: Type) -> u32 {
    type_bits(ty) / 8
}

fn defined_type(inst: &Inst, types: &BTreeMap<Value, Type>) -> Option<Type> {
    match inst {
        Inst::Iconst { ty, .. }
        | Inst::Sextend { ty, .. }
        | Inst::Uextend { ty, .. }
        | Inst::Ireduce { ty, .. }
        | Inst::Load { ty, .. } => Some(*ty),
        Inst::Iadd(left, _)
        | Inst::Isub(left, _)
        | Inst::Sdiv(left, _)
        | Inst::Udiv(left, _)
        | Inst::Imul(left, _) => types.get(left).copied(),
        Inst::Icmp { .. } => Some(Type::I8),
        Inst::SlotAddr(_) | Inst::DataAddr { .. } => Some(Type::Ptr),
        Inst::Call { ret, .. } => *ret,
        Inst::Store { .. } => None,
    }
}

enum Access {
    Load,
    Store,
}

// every Load and Store with the slot its address came from, if any, its offset and width; a Store's width is its value's type
fn for_each_access(function: &Function, mut visit: impl FnMut(Access, Option<SlotId>, u32, u32)) {
    let mut types: BTreeMap<Value, Type> = BTreeMap::new();
    let mut addrs: BTreeMap<Value, SlotId> = BTreeMap::new();
    for block in &function.blocks {
        types.extend(block.params.iter().copied());
        for (result, inst) in &block.insts {
            if let (Some(value), Some(ty)) = (result, defined_type(inst, &types)) {
                types.insert(*value, ty);
            }
            match inst {
                Inst::SlotAddr(slot) => {
                    if let Some(value) = result {
                        addrs.insert(*value, *slot);
                    }
                }
                Inst::Load { ty, addr, offset } => {
                    visit(
                        Access::Load,
                        addrs.get(addr).copied(),
                        *offset,
                        type_bytes(*ty),
                    );
                }
                Inst::Store {
                    value,
                    addr,
                    offset,
                } => {
                    if let Some(ty) = types.get(value) {
                        visit(
                            Access::Store,
                            addrs.get(addr).copied(),
                            *offset,
                            type_bytes(*ty),
                        );
                    }
                }
                _ => {}
            }
        }
    }
}

fn slot_accesses(function: &Function) -> BTreeMap<(SlotId, u32), BTreeSet<u32>> {
    let mut access: BTreeMap<(SlotId, u32), BTreeSet<u32>> = BTreeMap::new();
    for_each_access(function, |_, slot, offset, width| {
        if let Some(slot) = slot {
            access.entry((slot, offset)).or_default().insert(width);
        }
    });
    access
}

fn width_counts(counts: &BTreeMap<u32, u32>) -> String {
    if counts.is_empty() {
        return "-".to_owned();
    }
    let parts: Vec<String> = counts
        .iter()
        .map(|(width, count)| format!("{width}:{count}"))
        .collect();
    parts.join(",")
}

// one line when the function loads or stores through a pointer that is not a slot's own address
fn deref_line(function: &Function) -> Option<String> {
    let mut loads: BTreeMap<u32, u32> = BTreeMap::new();
    let mut stores: BTreeMap<u32, u32> = BTreeMap::new();
    for_each_access(function, |kind, slot, _, width| {
        if slot.is_some() {
            return;
        }
        let counts = match kind {
            Access::Load => &mut loads,
            Access::Store => &mut stores,
        };
        *counts.entry(width).or_default() += 1;
    });
    if loads.is_empty() && stores.is_empty() {
        return None;
    }
    Some(format!(
        "nova_deref fn={} loads={} stores={}",
        function.name,
        width_counts(&loads),
        width_counts(&stores)
    ))
}

// one stdout line per enum local; size, align and the access map come back out of the lowered function
fn enum_lines(
    tcx: TyCtxt<'_>,
    function: &Function,
    enums: &[Option<(SlotId, EnumLayout)>],
) -> Vec<String> {
    let access = slot_accesses(function);
    enums
        .iter()
        .flatten()
        .filter_map(|(id, layout)| {
            let slot = function.slots.get(id.0 as usize)?;
            let accessed: Vec<String> = access
                .iter()
                .filter(|((slot_id, _), _)| slot_id == id)
                .map(|((_, offset), widths)| {
                    let widths: Vec<String> = widths.iter().map(u32::to_string).collect();
                    format!("{offset}:{}", widths.join("/"))
                })
                .collect();
            let fields: Vec<String> =
                layout
                    .variants
                    .iter()
                    .flat_map(|variant| {
                        variant.fields.iter().enumerate().map(|(index, field)| {
                            format!("{}.{index}@{}", variant.name, field.offset)
                        })
                    })
                    .collect();
            Some(format!(
                "nova_layout type={} slot_size={} slot_align={} tag=0:{} access={} fields={}",
                def_text(tcx, layout.def),
                slot.size,
                slot.align,
                type_bytes(layout.tag),
                accessed.join(","),
                fields.join(",")
            ))
        })
        .collect()
}

/// Lower `body`, the generic MIR of `instance`, into a nova function named `name`.
#[must_use = "the lowered function or the refusal must be handled"]
pub fn lower_body<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    body: &'tcx Body<'tcx>,
    name: String,
    allocs: &mut Vec<AllocId>,
) -> Result<Function, MirError> {
    lower_core(tcx, instance, body, name, allocs).map(|(function, _, _)| function)
}

/// [`lower_body`] plus one `nova_layout` line per struct and enum local; only the listing modes call it.
#[must_use = "the lowered function or the refusal must be handled"]
pub fn lower_body_listed<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    body: &'tcx Body<'tcx>,
    name: String,
    allocs: &mut Vec<AllocId>,
) -> Result<(Function, Vec<String>), MirError> {
    let (function, structs, enums) = lower_core(tcx, instance, body, name, allocs)?;
    let mut lines = layout_lines(&function, &structs);
    lines.extend(enum_lines(tcx, &function, &enums));
    lines.extend(deref_line(&function));
    Ok((function, lines))
}

type StructLocals = Vec<Option<(SlotId, StructLayout)>>;
type EnumLocals = Vec<Option<(SlotId, EnumLayout)>>;

fn lower_core<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    body: &'tcx Body<'tcx>,
    name: String,
    allocs: &mut Vec<AllocId>,
) -> Result<(Function, StructLocals, EnumLocals), MirError> {
    let local_count = body.local_decls.len();
    let mut lowerer = Lowerer {
        tcx,
        instance,
        body,
        monos: HashMap::new(),
        reprs: HashMap::new(),
        slots: vec![None; local_count],
        structs: vec![None; local_count],
        enums: vec![None; local_count],
        frame: Vec::new(),
        tys: Vec::with_capacity(local_count),
        next_value: 0,
        next_block: 0,
        extra: Vec::new(),
        allocs: Vec::new(),
    };
    for decl in body.local_decls.iter() {
        let mono = lowerer.mono_ty(decl.ty);
        lowerer.tys.push(mono);
    }
    let reprs = lowerer
        .tys
        .clone()
        .into_iter()
        .map(|ty| lowerer.repr(ty))
        .collect::<Result<Vec<_>, _>>()?;
    refuse_struct_boundary(tcx, &reprs, body.arg_count)?;
    let block_count =
        u32::try_from(body.basic_blocks.len() + 1).map_err(|_| MirError::BlockOutOfRange {
            block: body.basic_blocks.len(),
        })?;
    lowerer.next_block = block_count;
    let (prologue, params) = lower_prologue(&mut lowerer, &reprs, body.arg_count)?;
    let returns = match reprs.first() {
        Some(Repr::Scalar { ty, .. }) => Some(*ty),
        _ => None,
    };
    let mut blocks = vec![prologue];
    for block in body.basic_blocks.iter() {
        blocks.push(lower_block(&mut lowerer, block)?);
    }
    blocks.append(&mut lowerer.extra);
    let function = Function {
        name,
        params: params.iter().map(|(_, ty)| *ty).collect(),
        returns,
        slots: lowerer.frame,
        blocks,
    };
    allocs.append(&mut lowerer.allocs);
    Ok((function, lowerer.structs, lowerer.enums))
}

fn lower_prologue(
    lowerer: &mut Lowerer<'_>,
    reprs: &[Repr],
    arg_count: usize,
) -> Result<(Block, Vec<(Value, Type)>), MirError> {
    let mut params = Vec::new();
    for repr in &reprs[1..=arg_count] {
        if let Repr::Scalar { ty, .. } = repr {
            params.push((lowerer.fresh(), *ty));
        }
    }
    let mut insts = Vec::new();
    for (local, repr) in reprs.iter().enumerate() {
        let (size, align) = match repr {
            Repr::Zst => continue,
            Repr::Scalar { size, align, .. } => (*size, *align),
            Repr::Struct(layout) => (layout.size, layout.align),
            Repr::Enum(layout) => (layout.size, layout.align),
        };
        let index =
            u32::try_from(lowerer.frame.len()).map_err(|_| MirError::TooManyLocals { local })?;
        let id = SlotId(index);
        lowerer.frame.push(Slot { size, align });
        match repr {
            Repr::Scalar { ty, .. } => lowerer.slots[local] = Some(LocalSlot { id, ty: *ty }),
            Repr::Struct(layout) => lowerer.structs[local] = Some((id, layout.clone())),
            Repr::Enum(layout) => lowerer.enums[local] = Some((id, layout.clone())),
            Repr::Zst => {}
        }
    }
    let mut incoming = params.iter();
    let arg_slots: Vec<LocalSlot> = lowerer.slots[1..=arg_count]
        .iter()
        .flatten()
        .copied()
        .collect();
    for slot in arg_slots {
        if let Some((value, _)) = incoming.next() {
            lowerer.store_slot(slot, *value, &mut insts);
        }
    }
    let entry = Block {
        params: params.clone(),
        insts,
        terminator: Some(Terminator::Jump {
            target: BlockId(1),
            args: vec![],
        }),
    };
    Ok((entry, params))
}

fn lower_block<'tcx>(
    lowerer: &mut Lowerer<'tcx>,
    block: &BasicBlockData<'tcx>,
) -> Result<Block, MirError> {
    let mut insts = Vec::new();
    for statement in &block.statements {
        lower_statement(lowerer, &statement.kind, &mut insts)?;
    }
    let terminator = lower_terminator(lowerer, block.terminator(), &mut insts)?;
    Ok(Block {
        params: vec![],
        insts,
        terminator: Some(terminator),
    })
}

fn lower_statement<'tcx>(
    lowerer: &mut Lowerer<'tcx>,
    kind: &StatementKind<'tcx>,
    insts: &mut Insts,
) -> Result<(), MirError> {
    match kind {
        StatementKind::StorageLive(_) | StatementKind::StorageDead(_) | StatementKind::Nop => {
            Ok(())
        }
        StatementKind::Assign(assign) => {
            let (place, rvalue) = &**assign;
            if place.projection.is_empty()
                || lowerer.field_of(place)?.is_some()
                || lowerer.deref_of(place)?.is_some()
            {
                return lower_assign(lowerer, place, rvalue, insts);
            }
            Err(unsupported_statement(lowerer.tcx, kind))
        }
        other => Err(unsupported_statement(lowerer.tcx, other)),
    }
}

fn unsupported_statement<'tcx>(tcx: TyCtxt<'tcx>, kind: &StatementKind<'tcx>) -> MirError {
    MirError::UnsupportedStatement {
        statement: refusal_text!(tcx, rustc_internal::stable(kind)),
    }
}

// one store per field, at the offset rustc's layout gave it
fn lower_aggregate<'tcx>(
    lowerer: &mut Lowerer<'tcx>,
    place: &Place<'tcx>,
    operands: &[Operand<'tcx>],
    insts: &mut Insts,
) -> Result<(), MirError> {
    let Some((id, layout)) = lowerer.structs[local_of(place)].clone() else {
        return Err(MirError::UnsupportedRvalue {
            rvalue: format!("aggregate into {}", place_text(lowerer.tcx, place)),
        });
    };
    if operands.len() != layout.fields.len() {
        return Err(MirError::UnsupportedRvalue {
            rvalue: format!(
                "aggregate of {} operands into struct {}",
                operands.len(),
                layout.name
            ),
        });
    }
    for (field, operand) in layout.fields.iter().zip(operands) {
        let value = lowerer.operand_value(operand, insts)?;
        lowerer.store_at(id, field.offset, value, insts);
    }
    Ok(())
}

// the variant's discriminant as rustc reports it, cut to the tag's width
fn tag_immediate(discriminant: u128, tag: Type) -> Result<i64, MirError> {
    let masked = discriminant & ((1u128 << type_bits(tag)) - 1);
    let bits = u64::try_from(masked).map_err(|_| MirError::ConstantTooWide {
        value: i128::try_from(masked).unwrap_or(i128::MAX),
    })?;
    Ok(i64::from_ne_bytes(bits.to_ne_bytes()))
}

fn enum_layout_of(lowerer: &Lowerer<'_>, local: usize) -> Result<(SlotId, EnumLayout), MirError> {
    lowerer
        .enum_of(local)
        .cloned()
        .ok_or(MirError::UnsupportedOperand {
            operand: format!("_{local} is not an enum local"),
        })
}

// fields first, then the discriminant at the tag
fn lower_variant_aggregate<'tcx>(
    lowerer: &mut Lowerer<'tcx>,
    place: &Place<'tcx>,
    variant: VariantIdx,
    operands: &[Operand<'tcx>],
    insts: &mut Insts,
) -> Result<(), MirError> {
    let (id, layout) = enum_layout_of(lowerer, local_of(place))?;
    let chosen = layout
        .variants
        .iter()
        .find(|candidate| candidate.idx == variant)
        .filter(|candidate| candidate.fields.len() == operands.len())
        .ok_or_else(|| MirError::UnsupportedRvalue {
            rvalue: format!(
                "aggregate of {} operands into enum {}",
                operands.len(),
                def_text(lowerer.tcx, layout.def)
            ),
        })?;
    for (field, operand) in chosen.fields.iter().zip(operands) {
        let value = lowerer.operand_value(operand, insts)?;
        lowerer.store_at(id, field.offset, value, insts);
    }
    let imm = tag_immediate(chosen.discriminant, layout.tag)?;
    let tag = lowerer.fresh();
    insts.push((
        Some(tag),
        Inst::Iconst {
            ty: layout.tag,
            imm,
        },
    ));
    lowerer.store_at(id, 0, tag, insts);
    Ok(())
}

// the tag, then every variant's every field, each at its own offset and width
fn lower_enum_copy<'tcx>(
    lowerer: &mut Lowerer<'tcx>,
    place: &Place<'tcx>,
    source: &Place<'tcx>,
    insts: &mut Insts,
) -> Result<(), MirError> {
    let (to_id, layout) = enum_layout_of(lowerer, local_of(place))?;
    let (from_id, _) = enum_layout_of(lowerer, local_of(source))?;
    if lowerer.tys[local_of(place)] != lowerer.tys[local_of(source)] {
        return Err(MirError::UnsupportedRvalue {
            rvalue: format!(
                "copy of {} into a differently typed enum {}",
                place_text(lowerer.tcx, source),
                place_text(lowerer.tcx, place)
            ),
        });
    }
    let tag = lowerer.load_at(from_id, layout.tag, 0, insts);
    lowerer.store_at(to_id, 0, tag, insts);
    for field in layout.variants.iter().flat_map(|variant| &variant.fields) {
        let value = lowerer.load_at(from_id, field.ty, field.offset, insts);
        lowerer.store_at(to_id, field.offset, value, insts);
    }
    Ok(())
}

fn lower_enum_assign<'tcx>(
    lowerer: &mut Lowerer<'tcx>,
    place: &Place<'tcx>,
    rvalue: &Rvalue<'tcx>,
    insts: &mut Insts,
) -> Result<(), MirError> {
    match rvalue {
        Rvalue::Aggregate(kind, operands) => match &**kind {
            AggregateKind::Adt(_, variant, ..) => {
                lower_variant_aggregate(lowerer, place, *variant, operands.raw.as_slice(), insts)
            }
            _ => Err(enum_rvalue_error(lowerer.tcx, rvalue)),
        },
        Rvalue::Use(Operand::Copy(source) | Operand::Move(source), _)
            if source.projection.is_empty() =>
        {
            lower_enum_copy(lowerer, place, source, insts)
        }
        other => Err(enum_rvalue_error(lowerer.tcx, other)),
    }
}

fn enum_rvalue_error<'tcx>(tcx: TyCtxt<'tcx>, rvalue: &Rvalue<'tcx>) -> MirError {
    MirError::UnsupportedRvalue {
        rvalue: format!("{} into an enum local", rvalue_text(tcx, rvalue)),
    }
}

// the tag is widened to the discriminant type by the tag's own signedness; equal widths emit nothing
fn lower_discriminant<'tcx>(
    lowerer: &mut Lowerer<'tcx>,
    source: &Place<'tcx>,
    dest: Type,
    insts: &mut Insts,
) -> Result<Value, MirError> {
    if !source.projection.is_empty() {
        return Err(lowerer.unsupported_place(source));
    }
    let (id, layout) = enum_layout_of(lowerer, local_of(source))?;
    let tag = lowerer.load_at(id, layout.tag, 0, insts);
    if layout.tag == dest {
        return Ok(tag);
    }
    if type_bits(layout.tag) > type_bits(dest) {
        return Err(MirError::UnsupportedCast {
            detail: format!("discriminant tag {:?} wider than {dest:?}", layout.tag),
        });
    }
    let inst = if layout.tag_signed {
        Inst::Sextend {
            ty: dest,
            value: tag,
        }
    } else {
        Inst::Uextend {
            ty: dest,
            value: tag,
        }
    };
    let result = lowerer.fresh();
    insts.push((Some(result), inst));
    Ok(result)
}

// a reference is its place's SlotAddr, plus the field's offset for one field of a struct local
fn lower_ref<'tcx>(
    lowerer: &mut Lowerer<'tcx>,
    place: &Place<'tcx>,
    insts: &mut Insts,
) -> Result<Value, MirError> {
    let unsupported = |lowerer: &Lowerer<'tcx>| MirError::UnsupportedRvalue {
        rvalue: format!("reference to {}", place_text(lowerer.tcx, place)),
    };
    let local = local_of(place);
    if place.projection.is_empty() {
        let id = lowerer
            .slot_of(local)?
            .map(|slot| slot.id)
            .or_else(|| lowerer.structs[local].as_ref().map(|(id, _)| *id))
            .or_else(|| lowerer.enums[local].as_ref().map(|(id, _)| *id))
            .ok_or_else(|| unsupported(lowerer))?;
        return Ok(lowerer.slot_addr(id, insts));
    }
    let Some((id, _, offset)) = lowerer.field_of(place)? else {
        return Err(unsupported(lowerer));
    };
    if lowerer.structs[local].is_none() {
        return Err(unsupported(lowerer));
    }
    let base = lowerer.slot_addr(id, insts);
    if offset == 0 {
        return Ok(base);
    }
    let delta = lowerer.fresh();
    insts.push((
        Some(delta),
        Inst::Iconst {
            ty: Type::I64,
            imm: i64::from(offset),
        },
    ));
    let result = lowerer.fresh();
    insts.push((Some(result), Inst::Iadd(base, delta)));
    Ok(result)
}

fn rvalue_value<'tcx>(
    lowerer: &mut Lowerer<'tcx>,
    rvalue: &Rvalue<'tcx>,
    dest: Type,
    insts: &mut Insts,
) -> Result<Value, MirError> {
    match rvalue {
        Rvalue::Discriminant(source) => lower_discriminant(lowerer, source, dest, insts),
        Rvalue::Use(operand, _) => lowerer.operand_value(operand, insts),
        Rvalue::Ref(_, _, place) => lower_ref(lowerer, place, insts),
        Rvalue::BinaryOp(op, operands) => {
            let (lhs, rhs) = &**operands;
            lower_binary(lowerer, *op, lhs, rhs, insts)
        }
        Rvalue::Cast(CastKind::IntToInt, operand, target) => {
            lower_int_cast(lowerer, operand, *target, insts)
        }
        other => Err(MirError::UnsupportedRvalue {
            rvalue: rvalue_text(lowerer.tcx, other),
        }),
    }
}

fn lower_assign<'tcx>(
    lowerer: &mut Lowerer<'tcx>,
    place: &Place<'tcx>,
    rvalue: &Rvalue<'tcx>,
    insts: &mut Insts,
) -> Result<(), MirError> {
    if let Some((id, ty, offset)) = lowerer.field_of(place)? {
        let value = rvalue_value(lowerer, rvalue, ty, insts)?;
        lowerer.store_at(id, offset, value, insts);
        return Ok(());
    }
    if let Some((pointer, ty, offset, _)) = lowerer.deref_of(place)? {
        let value = rvalue_value(lowerer, rvalue, ty, insts)?;
        let addr = lowerer.load_slot(pointer, insts);
        insts.push((
            None,
            Inst::Store {
                value,
                addr,
                offset,
            },
        ));
        return Ok(());
    }
    let local = local_of(place);
    if lowerer.enum_of(local).is_some() {
        return lower_enum_assign(lowerer, place, rvalue, insts);
    }
    if lowerer.structs[local].is_some() {
        return match rvalue {
            Rvalue::Aggregate(kind, operands) if matches!(&**kind, AggregateKind::Adt(..)) => {
                lower_aggregate(lowerer, place, operands.raw.as_slice(), insts)
            }
            other => Err(MirError::UnsupportedRvalue {
                rvalue: format!("{} into a struct local", rvalue_text(lowerer.tcx, other)),
            }),
        };
    }
    let Some(slot) = lowerer.slot_of(local)? else {
        return match rvalue {
            Rvalue::Use(..) => Ok(()),
            other => Err(MirError::UnsupportedRvalue {
                rvalue: format!(
                    "{} into a zero-sized local",
                    rvalue_text(lowerer.tcx, other)
                ),
            }),
        };
    };
    let value = rvalue_value(lowerer, rvalue, slot.ty, insts)?;
    lowerer.store_slot(slot, value, insts);
    Ok(())
}

fn compare_cond(op: BinOp, kind: Kind) -> Option<Cond> {
    let signed = kind == Kind::Signed;
    match (op, signed) {
        (BinOp::Eq, _) => Some(Cond::Eq),
        (BinOp::Ne, _) => Some(Cond::Ne),
        (BinOp::Lt, true) => Some(Cond::Slt),
        (BinOp::Lt, false) => Some(Cond::Ult),
        (BinOp::Le, true) => Some(Cond::Sle),
        (BinOp::Le, false) => Some(Cond::Ule),
        (BinOp::Gt, true) => Some(Cond::Sgt),
        (BinOp::Gt, false) => Some(Cond::Ugt),
        (BinOp::Ge, true) => Some(Cond::Sge),
        (BinOp::Ge, false) => Some(Cond::Uge),
        _ => None,
    }
}

fn binop_error(tcx: TyCtxt<'_>, op: BinOp) -> MirError {
    MirError::UnsupportedBinOp {
        op: refusal_text!(tcx, rustc_internal::stable(op)),
    }
}

fn div_inst(kind: Kind, left: Value, right: Value) -> Inst {
    if kind == Kind::Signed {
        Inst::Sdiv(left, right)
    } else {
        Inst::Udiv(left, right)
    }
}

// a - (a / b) * b; the assert before it has already excluded b == 0 and MIN / -1
fn lower_rem(
    lowerer: &mut Lowerer<'_>,
    kind: Kind,
    left: Value,
    right: Value,
    insts: &mut Insts,
) -> Value {
    let quotient = lowerer.fresh();
    insts.push((Some(quotient), div_inst(kind, left, right)));
    let product = lowerer.fresh();
    insts.push((Some(product), Inst::Imul(quotient, right)));
    let result = lowerer.fresh();
    insts.push((Some(result), Inst::Isub(left, product)));
    result
}

// bool `&` only: the product of the two zero-extended flags is nonzero exactly when both are 1
fn lower_bool_and<'tcx>(
    lowerer: &mut Lowerer<'tcx>,
    lhs: &Operand<'tcx>,
    rhs: &Operand<'tcx>,
    insts: &mut Insts,
) -> Result<Value, MirError> {
    if lowerer.operand_kind(lhs)? != Kind::Bool || lowerer.operand_kind(rhs)? != Kind::Bool {
        return Err(binop_error(lowerer.tcx, BinOp::BitAnd));
    }
    let left = lowerer.operand_value(lhs, insts)?;
    let right = lowerer.operand_value(rhs, insts)?;
    let left = widen_flag(lowerer, left, insts);
    let right = widen_flag(lowerer, right, insts);
    let product = lowerer.fresh();
    insts.push((Some(product), Inst::Imul(left, right)));
    let zero = lowerer.fresh();
    insts.push((
        Some(zero),
        Inst::Iconst {
            ty: Type::I32,
            imm: 0,
        },
    ));
    let result = lowerer.fresh();
    insts.push((
        Some(result),
        Inst::Icmp {
            cond: Cond::Ne,
            a: product,
            b: zero,
        },
    ));
    Ok(result)
}

fn widen_flag(lowerer: &mut Lowerer<'_>, value: Value, insts: &mut Insts) -> Value {
    let wide = lowerer.fresh();
    insts.push((
        Some(wide),
        Inst::Uextend {
            ty: Type::I32,
            value,
        },
    ));
    wide
}

fn lower_binary<'tcx>(
    lowerer: &mut Lowerer<'tcx>,
    op: BinOp,
    lhs: &Operand<'tcx>,
    rhs: &Operand<'tcx>,
    insts: &mut Insts,
) -> Result<Value, MirError> {
    if matches!(op, BinOp::BitAnd) {
        return lower_bool_and(lowerer, lhs, rhs, insts);
    }
    if !matches!(
        op,
        BinOp::Add
            | BinOp::Sub
            | BinOp::Mul
            | BinOp::Div
            | BinOp::Rem
            | BinOp::Eq
            | BinOp::Ne
            | BinOp::Lt
            | BinOp::Le
            | BinOp::Gt
            | BinOp::Ge
    ) {
        return Err(binop_error(lowerer.tcx, op));
    }
    let kind = lowerer.operand_kind(lhs)?;
    if !matches!(kind, Kind::Signed | Kind::Unsigned) {
        let ty = lowerer.operand_ty(lhs)?;
        return Err(MirError::UnsupportedOperandType {
            ty: ty_text(lowerer.tcx, ty),
        });
    }
    let left = lowerer.operand_value(lhs, insts)?;
    let right = lowerer.operand_value(rhs, insts)?;
    if op == BinOp::Rem {
        return Ok(lower_rem(lowerer, kind, left, right, insts));
    }
    let inst = match op {
        BinOp::Div => div_inst(kind, left, right),
        BinOp::Add => Inst::Iadd(left, right),
        BinOp::Sub => Inst::Isub(left, right),
        BinOp::Mul => Inst::Imul(left, right),
        _ => match compare_cond(op, kind) {
            Some(cond) => Inst::Icmp {
                cond,
                a: left,
                b: right,
            },
            None => return Err(binop_error(lowerer.tcx, op)),
        },
    };
    let result = lowerer.fresh();
    insts.push((Some(result), inst));
    Ok(result)
}

// a cast at equal width emits nothing; widening picks the extension by the source's signedness
fn lower_int_cast<'tcx>(
    lowerer: &mut Lowerer<'tcx>,
    operand: &Operand<'tcx>,
    target: Ty<'tcx>,
    insts: &mut Insts,
) -> Result<Value, MirError> {
    let from = lowerer.operand_ty(operand)?;
    let target = lowerer.mono_ty(target);
    let integer = |kind: Kind| matches!(kind, Kind::Signed | Kind::Unsigned);
    let cast_error = |lowerer: &Lowerer<'tcx>| MirError::UnsupportedCast {
        detail: format!(
            "IntToInt {} to {}",
            ty_text(lowerer.tcx, from),
            ty_text(lowerer.tcx, target)
        ),
    };
    if !(integer(kind_of(from)) && integer(kind_of(target))) {
        return Err(cast_error(lowerer));
    }
    let (Repr::Scalar { ty: from_ty, .. }, Repr::Scalar { ty: to_ty, .. }) =
        (lowerer.repr(from)?, lowerer.repr(target)?)
    else {
        return Err(cast_error(lowerer));
    };
    let value = lowerer.operand_value(operand, insts)?;
    let inst = match (from_ty, to_ty) {
        (Type::I32, Type::I64) if kind_of(from) == Kind::Signed => {
            Inst::Sextend { ty: to_ty, value }
        }
        (Type::I32, Type::I64) => Inst::Uextend { ty: to_ty, value },
        (Type::I64, Type::I32) => Inst::Ireduce { ty: to_ty, value },
        _ => return Ok(value),
    };
    let result = lowerer.fresh();
    insts.push((Some(result), inst));
    Ok(result)
}

fn lower_terminator<'tcx>(
    lowerer: &mut Lowerer<'tcx>,
    terminator: &MirTerminator<'tcx>,
    insts: &mut Insts,
) -> Result<Terminator, MirError> {
    let kind = &terminator.kind;
    match kind {
        TerminatorKind::Return => lower_return(lowerer, insts),
        TerminatorKind::Goto { target, .. } => Ok(Terminator::Jump {
            target: block_id(target.as_usize())?,
            args: vec![],
        }),
        TerminatorKind::SwitchInt { discr, targets } => {
            lower_switch(lowerer, discr, targets, insts)
        }
        TerminatorKind::Call {
            func,
            args,
            destination,
            target,
            unwind,
            ..
        } => {
            refuse_unwind(lowerer.tcx, unwind, kind)?;
            let operands: Vec<Operand<'tcx>> = args.iter().map(|arg| arg.node.clone()).collect();
            lower_call(
                lowerer,
                func,
                &operands,
                destination,
                target.map(|block| block.as_usize()),
                terminator.source_info,
                insts,
            )
        }
        TerminatorKind::Assert {
            cond,
            expected,
            msg,
            target,
            unwind,
        } => {
            refuse_unwind(lowerer.tcx, unwind, kind)?;
            let assert = AssertSite {
                cond,
                expected: *expected,
                msg,
                target: block_id(target.as_usize())?,
                source_info: terminator.source_info,
            };
            lower_assert(lowerer, &assert, insts)
        }
        TerminatorKind::Unreachable => Ok(Terminator::Unreachable),
        other => Err(MirError::UnsupportedTerminator {
            terminator: refusal_text!(lowerer.tcx, rustc_internal::stable(other)),
        }),
    }
}

const MAX_CALL_ARGS: usize = 8;

// the symbol is the resolved instance's symbol_name, the same call lower_instance names definitions
// with; for a foreign item rustc's symbol_name is its #[link_name] or else its item name
fn callee_symbol<'tcx>(
    lowerer: &mut Lowerer<'tcx>,
    func: &Operand<'tcx>,
) -> Result<(String, Instance<'tcx>), MirError> {
    let Operand::Constant(constant) = func else {
        return Err(MirError::UnsupportedOperand {
            operand: operand_text(lowerer.tcx, func),
        });
    };
    let callee_ty = lowerer.mono_ty(constant.const_.ty());
    let &TyKind::FnDef(def, args) = callee_ty.0.kind() else {
        return Err(MirError::UnsupportedOperand {
            operand: operand_text(lowerer.tcx, func),
        });
    };
    let tcx = lowerer.tcx;
    match Instance::try_resolve(
        tcx,
        TypingEnv::fully_monomorphized(),
        def,
        args.skip_binder(),
    ) {
        Ok(Some(instance)) => Ok((tcx.symbol_name(instance).name.to_string(), instance)),
        _ => Err(MirError::UnresolvedCallee {
            callee: format!("{}: unresolved", def_text(tcx, def)),
        }),
    }
}

fn lower_call<'tcx>(
    lowerer: &mut Lowerer<'tcx>,
    func: &Operand<'tcx>,
    args: &[Operand<'tcx>],
    destination: &Place<'tcx>,
    target: Option<usize>,
    source_info: SourceInfo,
    insts: &mut Insts,
) -> Result<Terminator, MirError> {
    if args.len() > MAX_CALL_ARGS {
        return Err(MirError::TooManyCallArgs { count: args.len() });
    }
    if !destination.projection.is_empty() {
        return Err(lowerer.unsupported_place(destination));
    }
    let dest_local = local_of(destination);
    if let Some((_, layout)) = &lowerer.structs[dest_local] {
        return Err(MirError::StructResult {
            ty: layout.name.to_string(),
        });
    }
    if let Some((_, layout)) = lowerer.enum_of(dest_local) {
        return Err(MirError::StructResult {
            ty: def_text(lowerer.tcx, layout.def),
        });
    }
    let (callee, instance) = callee_symbol(lowerer, func)?;
    let located = instance.def.requires_caller_location(lowerer.tcx);
    if located {
        refuse_track_caller_body(lowerer)?;
    }
    let mut values = Vec::with_capacity(args.len());
    for arg in args {
        if let Operand::Copy(place) | Operand::Move(place) = arg
            && place.projection.is_empty()
            && let Some((_, layout)) = &lowerer.structs[local_of(place)]
        {
            return Err(MirError::StructArgument {
                ty: layout.name.to_string(),
            });
        }
        if let Operand::Copy(place) | Operand::Move(place) = arg
            && place.projection.is_empty()
            && let Some((_, layout)) = lowerer.enum_of(local_of(place))
        {
            return Err(MirError::StructArgument {
                ty: def_text(lowerer.tcx, layout.def),
            });
        }
        let arg_ty = lowerer.operand_ty(arg)?;
        if kind_of(arg_ty) == Kind::Other && !is_reference(arg_ty) {
            return Err(MirError::UnsupportedOperandType {
                ty: ty_text(lowerer.tcx, arg_ty),
            });
        }
        values.push(lowerer.operand_value(arg, insts)?);
    }
    if located {
        values.push(lowerer.caller_location(source_info, insts)?);
    }
    if values.len() > MAX_CALL_ARGS {
        return Err(MirError::TooManyCallArgs {
            count: values.len(),
        });
    }
    let slot = lowerer.slot_of(dest_local)?;
    let dest_ty = lowerer.tys[dest_local];
    if kind_of(dest_ty) == Kind::Other && !is_reference(dest_ty) && slot.is_some() {
        return Err(MirError::UnsupportedOperandType {
            ty: ty_text(lowerer.tcx, dest_ty),
        });
    }
    let result = slot.map(|_| lowerer.fresh());
    insts.push((
        result,
        Inst::Call {
            callee,
            args: values,
            ret: slot.map(|local| local.ty),
        },
    ));
    if let (Some(local), Some(value)) = (slot, result) {
        lowerer.store_slot(local, value, insts);
    }
    match target {
        Some(block) => Ok(Terminator::Jump {
            target: block_id(block)?,
            args: vec![],
        }),
        None => Ok(Terminator::Unreachable),
    }
}

// unwind is always Unreachable under panic=abort; any cleanup edge is a refusal, never a dropped path
fn refuse_unwind<'tcx>(
    tcx: TyCtxt<'tcx>,
    unwind: &UnwindAction,
    kind: &TerminatorKind<'tcx>,
) -> Result<(), MirError> {
    if matches!(unwind, UnwindAction::Unreachable) {
        return Ok(());
    }
    Err(MirError::UnwindCleanup {
        terminator: refusal_text!(tcx, rustc_internal::stable(kind)),
    })
}

// a #[track_caller] body's location is an implicit parameter nova does not carry, so a body that
// would read it (an assert, or a call to another #[track_caller] fn) is refused
fn refuse_track_caller_body(lowerer: &Lowerer<'_>) -> Result<(), MirError> {
    if !lowerer.instance.def.requires_caller_location(lowerer.tcx) {
        return Ok(());
    }
    Err(MirError::TrackCaller {
        item: def_text(lowerer.tcx, lowerer.instance.def_id()),
    })
}

// rustc_codegen_cranelift base.rs:1113-1122: the lang item's mono instance names the symbol
fn panic_symbol(tcx: TyCtxt<'_>, item: LangItem, span: Span) -> String {
    let def = tcx.require_lang_item(item, span);
    tcx.symbol_name(Instance::mono(tcx, def)).name.to_string()
}

struct AssertSite<'a, 'tcx> {
    cond: &'a Operand<'tcx>,
    expected: bool,
    msg: &'a AssertKind<Operand<'tcx>>,
    target: BlockId,
    source_info: SourceInfo,
}

// rustc_codegen_ssa block.rs:768-832: a branch to a fail block that calls the lang item with the
// caller location and ends Unreachable; the argument table is the same
fn lower_assert<'tcx>(
    lowerer: &mut Lowerer<'tcx>,
    assert: &AssertSite<'_, 'tcx>,
    insts: &mut Insts,
) -> Result<Terminator, MirError> {
    let tcx = lowerer.tcx;
    if !tcx.sess.overflow_checks() && assert.msg.is_optional_overflow_check() {
        return Ok(Terminator::Jump {
            target: assert.target,
            args: vec![],
        });
    }
    refuse_track_caller_body(lowerer)?;
    let flag = lowerer.operand_value(assert.cond, insts)?;
    let span = assert.source_info.span;
    let mut fail = Vec::new();
    let (item, operands) = match assert.msg {
        AssertKind::BoundsCheck { len, index } => (LangItem::PanicBoundsCheck, vec![index, len]),
        AssertKind::InvalidEnumConstruction(_) => {
            return Err(MirError::UnsupportedTerminator {
                terminator: format!(
                    "Assert {}",
                    refusal_text!(tcx, rustc_internal::stable(assert.msg))
                ),
            });
        }
        AssertKind::MisalignedPointerDereference { required, found } => (
            LangItem::PanicMisalignedPointerDereference,
            vec![required, found],
        ),
        other => (other.panic_function(), vec![]),
    };
    let mut args = Vec::with_capacity(operands.len() + 1);
    for operand in operands {
        args.push(lowerer.operand_value(operand, &mut fail)?);
    }
    args.push(lowerer.caller_location(assert.source_info, &mut fail)?);
    let callee = panic_symbol(tcx, item, span);
    fail.push((
        None,
        Inst::Call {
            callee,
            args,
            ret: None,
        },
    ));
    let fail_block = lowerer.reserve_block();
    lowerer.extra.push(Block {
        params: vec![],
        insts: fail,
        terminator: Some(Terminator::Unreachable),
    });
    Ok(if assert.expected {
        branch(flag, assert.target, fail_block)
    } else {
        branch(flag, fail_block, assert.target)
    })
}

fn lower_return(lowerer: &mut Lowerer<'_>, insts: &mut Insts) -> Result<Terminator, MirError> {
    match lowerer.slot_of(0)? {
        None => Ok(Terminator::Return(None)),
        Some(slot) => Ok(Terminator::Return(Some(lowerer.load_slot(slot, insts)))),
    }
}

// MIR bbN is nova block N+1; block 0 is the prologue
fn block_id(mir_block: usize) -> Result<BlockId, MirError> {
    u32::try_from(mir_block + 1)
        .map(BlockId)
        .map_err(|_| MirError::BlockOutOfRange { block: mir_block })
}

fn branch(cond: Value, then_block: BlockId, else_block: BlockId) -> Terminator {
    Terminator::Branch {
        cond,
        then_block,
        then_args: vec![],
        else_block,
        else_args: vec![],
    }
}

fn lower_switch<'tcx>(
    lowerer: &mut Lowerer<'tcx>,
    discr: &Operand<'tcx>,
    targets: &SwitchTargets,
    insts: &mut Insts,
) -> Result<Terminator, MirError> {
    let branches: Vec<(u128, usize)> = targets
        .iter()
        .map(|(value, block)| (value, block.as_usize()))
        .collect();
    let otherwise = block_id(targets.otherwise().as_usize())?;
    match lowerer.operand_kind(discr)? {
        Kind::Bool => lower_bool_switch(lowerer, discr, &branches, otherwise, insts),
        Kind::Signed | Kind::Unsigned => {
            lower_int_switch(lowerer, discr, &branches, otherwise, insts)
        }
        Kind::Other => {
            let ty = lowerer.operand_ty(discr)?;
            Err(MirError::UnsupportedOperandType {
                ty: ty_text(lowerer.tcx, ty),
            })
        }
    }
}

fn lower_bool_switch<'tcx>(
    lowerer: &mut Lowerer<'tcx>,
    discr: &Operand<'tcx>,
    branches: &[(u128, usize)],
    otherwise: BlockId,
    insts: &mut Insts,
) -> Result<Terminator, MirError> {
    let flag = lowerer.operand_value(discr, insts)?;
    match branches {
        [(0, target)] => Ok(branch(flag, otherwise, block_id(*target)?)),
        [(1, target)] => Ok(branch(flag, block_id(*target)?, otherwise)),
        other => Err(MirError::UnsupportedSwitch {
            detail: format!("bool discriminant with branches {other:?}"),
        }),
    }
}

// each target tests the discriminant in its own block; the first test shares the MIR block, later
// ones are extra blocks reloading the discriminant, and the last test falls through to otherwise
fn lower_int_switch<'tcx>(
    lowerer: &mut Lowerer<'tcx>,
    discr: &Operand<'tcx>,
    branches: &[(u128, usize)],
    otherwise: BlockId,
    insts: &mut Insts,
) -> Result<Terminator, MirError> {
    let mut first = None;
    for (index, (wide, target)) in branches.iter().enumerate() {
        let else_block = if index + 1 == branches.len() {
            otherwise
        } else {
            lowerer.reserve_block()
        };
        let mut test = Vec::new();
        let terminator = lower_int_test(
            lowerer,
            discr,
            *wide,
            block_id(*target)?,
            else_block,
            &mut test,
        )?;
        if index == 0 {
            insts.append(&mut test);
            first = Some(terminator);
        } else {
            lowerer.extra.push(Block {
                params: vec![],
                insts: test,
                terminator: Some(terminator),
            });
        }
    }
    Ok(first.unwrap_or(Terminator::Jump {
        target: otherwise,
        args: vec![],
    }))
}

fn lower_int_test<'tcx>(
    lowerer: &mut Lowerer<'tcx>,
    discr: &Operand<'tcx>,
    wide: u128,
    then_block: BlockId,
    else_block: BlockId,
    insts: &mut Insts,
) -> Result<Terminator, MirError> {
    let discr_ty = lowerer.operand_ty(discr)?;
    let wide_discr = matches!(lowerer.repr(discr_ty)?, Repr::Scalar { ty: Type::I64, .. });
    let (ty, imm) = if wide_discr {
        let bits = u64::try_from(wide).map_err(|_| MirError::UnsupportedSwitch {
            detail: format!("target value {wide} wider than 64 bits"),
        })?;
        (Type::I64, i64::from_ne_bytes(bits.to_ne_bytes()))
    } else {
        let bits = u32::try_from(wide).map_err(|_| MirError::UnsupportedSwitch {
            detail: format!("target value {wide} wider than 32 bits"),
        })?;
        (Type::I32, i64::from(i32::from_ne_bytes(bits.to_ne_bytes())))
    };
    let value = lowerer.operand_value(discr, insts)?;
    let expected = lowerer.fresh();
    insts.push((Some(expected), Inst::Iconst { ty, imm }));
    let hit = lowerer.fresh();
    insts.push((
        Some(hit),
        Inst::Icmp {
            cond: Cond::Eq,
            a: value,
            b: expected,
        },
    ));
    Ok(branch(hit, then_block, else_block))
}
