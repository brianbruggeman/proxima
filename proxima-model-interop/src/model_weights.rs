//! Model-scope weight residency: materialize each owned checkpoint leaf once
//! per loaded model, and offer the same buffer on every call.
//!
//! A bind pass produces two kinds of weight storage. Leaves read straight out
//! of the mapped checkpoint borrow stable file bytes, so every pass offers the
//! same address. Leaves a bind has to materialize (an `in x out` transpose, a
//! widening decode, a precision-rule recode) live in a fresh heap buffer that
//! the pass owns, and a backend's name-keyed residency cache keys a name on
//! that buffer's address and length. A second pass that rebuilt the buffer
//! offered a new address under a name the cache already holds, which the
//! cache refuses as a rebind.
//!
//! [`ModelWeights`] is the model-scope owner of those materialized buffers.
//! [`ModelWeights::bind`] composes [`crate::bind_safetensors_program_leaves`]
//! and adopts a leaf already materialized instead of rebuilding it;
//! [`LeasedWeights`] hands the buffers back when the call ends, on the error
//! path as well, so the next call finds them. Moving a `Vec` moves its handle,
//! never its heap allocation, so the address a backend cached stays valid.
//!
//! This is the weight-side twin of call-scope residency: weights persist here
//! for the model's lifetime, while a call-scope resident (a decoder's
//! cross-attention K/V) is released by its owner when the call ends
//! (`BackendRuntime::release_call_residents`).

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::ops::Deref;

use proxima_numeric::DType;
use proxima_primitives::sync::blocking::Mutex;
use proxima_safetensors::{
    FORMAT_VERSION_KEY, Manifest, SafetensorsModel, TensorEntry, TensorPayload, is_filler_name,
    write_complete_aligned,
};
use proxima_tensor::op::Op;

use crate::bind::BoundWeights;
use crate::error::InteropError;
use crate::hf_bind::{bind_program_leaves_reusing, expert_stack_codec, tensor_bytes};
use crate::profiles::BindingProfile;
use crate::serving::WeightPrecisionRule;

/// `name` is `leaf` itself or its recoded `<leaf>@<codec>` form.
pub(crate) fn stored_for(name: &str, leaf: &str) -> bool {
    name == leaf
        || (name.len() > leaf.len()
            && name.as_bytes().get(leaf.len()) == Some(&b'@')
            && name.starts_with(leaf))
}

/// A buffer the model has never held under `name` is about to be kept: bytes a
/// backend may have cached under that name at a recycled address are not these
/// bytes, so the shared release advances the name's owner generation and
/// evicts them. A leaf adopted back (same buffer) never reaches here.
#[cfg(feature = "gpu")]
pub(crate) fn begin_owner_generation(name: &str) {
    omega::backend::release_resident_names([name]);
}

#[cfg(not(feature = "gpu"))]
pub(crate) fn begin_owner_generation(_name: &str) {}

/// The materialized stores of earlier passes, detached from any checkpoint
/// borrow: only the owned stores are ever filled, so the lifetime is free.
pub(crate) struct Materialized {
    stores: BoundWeights<'static>,
    // name -> heap address last kept; a kept buffer at the same address is the same bytes.
    seen: BTreeMap<String, usize>,
    // buffers kept at an address new to their name; a plan built before this moved may trust a recycled address
    allocations: u64,
}

impl Materialized {
    pub(crate) fn new() -> Self {
        Self {
            stores: BoundWeights::new(&[]),
            seen: BTreeMap::new(),
            allocations: 0,
        }
    }

    fn note_allocation(&mut self, name: &str, address: usize) {
        if self.seen.get(name) != Some(&address) {
            self.allocations += 1;
            self.seen.insert(name.to_string(), address);
            begin_owner_generation(name);
        }
    }

    /// Moves the entry stored for `leaf` (the leaf itself or its recoded
    /// `<leaf>@<codec>` form) into `weights`; `true` when one was found.
    pub(crate) fn adopt_into(&mut self, leaf: &str, weights: &mut BoundWeights<'_>) -> bool {
        let stores = &mut self.stores;
        if let Some(at) = stores
            .owned
            .iter()
            .position(|(name, _)| stored_for(name, leaf))
        {
            let entry = stores.owned.swap_remove(at);
            weights.resident_bytes += core::mem::size_of_val(entry.1.as_slice());
            weights.owned.push(entry);
            return true;
        }
        let Some(at) = stores
            .packed_owned
            .iter()
            .position(|(name, _, _)| stored_for(name, leaf))
        else {
            return false;
        };
        let entry = stores.packed_owned.swap_remove(at);
        weights.resident_bytes += entry.1.len();
        weights.packed_owned.push(entry);
        true
    }

    /// The recoded buffer held for `entry`'s tensor when it can replace the
    /// tensor in a written file: its codec has an element layout the shared
    /// binder reads back packed, and the bytes fill `entry`'s shape exactly.
    fn element_recode(&self, entry: &TensorEntry) -> Option<(DType, &[u8])> {
        let (_, bytes, codec) = self.stores.packed_owned.iter().find(|(name, _, _)| {
            name.len() > entry.name.len() && stored_for(name, &entry.name)
        })?;
        let dtype = DType::Layout(codec.element_layout()?);
        let elements = entry.shape.iter().product::<u64>();
        let fills_shape = bytes.len() as u64 == elements * dtype.size_bytes() as u64;
        (expert_stack_codec(dtype) == Some(*codec) && fills_shape)
            .then_some((dtype, bytes.as_slice()))
    }

    /// Takes `weights`' materialized entries, keeping the first buffer held
    /// under a name so the address a backend cached under it never changes.
    pub(crate) fn keep(&mut self, weights: &mut BoundWeights<'_>) {
        for entry in core::mem::take(&mut weights.owned) {
            if !self.stores.owned.iter().any(|(name, _)| *name == entry.0) {
                self.note_allocation(&entry.0, entry.1.as_ptr() as usize);
                self.stores.owned.push(entry);
            }
        }
        for entry in core::mem::take(&mut weights.packed_owned) {
            if !self
                .stores
                .packed_owned
                .iter()
                .any(|(name, _, _)| *name == entry.0)
            {
                self.note_allocation(&entry.0, entry.1.as_ptr() as usize);
                self.stores.packed_owned.push(entry);
            }
        }
    }
}

/// The weights one loaded model materialized, kept across calls.
///
/// Compose it with [`crate::bind_safetensors_program_leaves`]: use that
/// function directly for a one-shot bind, and this type when the same model
/// binds again on every call and a backend's name-keyed residency must see one
/// buffer per name.
pub struct ModelWeights {
    held: Mutex<Materialized>,
}

impl Default for ModelWeights {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelWeights {
    /// Whether [`Self::persist_aligned`] would substitute at least one tensor of `manifest`. A store entry is
    /// worth writing only then: without a recoded tensor it is a full copy of the source checkpoint.
    #[must_use]
    pub fn has_recoded(&self, manifest: &Manifest) -> bool {
        let held = self.held.lock();
        manifest
            .tensors
            .iter()
            .filter(|entry| !is_filler_name(&entry.name))
            .any(|entry| held.element_recode(entry).is_some())
    }

    /// How many buffers this model has kept at an address new to their name.
    #[must_use]
    #[cfg(all(feature = "std", feature = "proxima-autograd"))]
    pub(crate) fn allocation_epoch(&self) -> u64 {
        self.held.lock().allocations
    }

    #[must_use]
    pub fn new() -> Self {
        Self {
            held: Mutex::new(Materialized::new()),
        }
    }

    /// Binds `program`'s checkpoint leaves exactly as
    /// [`crate::bind_safetensors_program_leaves`] does, adopting a leaf this
    /// model already materialized rather than building it again. The returned
    /// lease parks what it materialized back here when it drops.
    #[allow(clippy::too_many_arguments)]
    pub fn bind<'file, 'store>(
        &'store self,
        manifest: &Manifest,
        file_bytes: &'file [u8],
        data_start: u64,
        program: &[Op],
        binding: &BindingProfile,
        runtime_inputs: &[String],
        weight_precision: &[WeightPrecisionRule<'_>],
    ) -> Result<LeasedWeights<'file, 'store>, InteropError> {
        let mut reuse = core::mem::replace(&mut *self.held.lock(), Materialized::new());
        let bound = bind_program_leaves_reusing(
            manifest,
            file_bytes,
            data_start,
            program,
            binding,
            runtime_inputs,
            weight_precision,
            &mut reuse,
        );
        {
            let mut held = self.held.lock();
            held.seen = core::mem::take(&mut reuse.seen);
            held.keep(&mut reuse.stores);
        }
        Ok(LeasedWeights {
            weights: bound?,
            home: Some(self),
        })
    }

    /// Rewrites the checkpoint `manifest`/`file_bytes` describe into `out`,
    /// each tensor this model recoded at an element layout replacing its source
    /// under the stored name and shape, and returns how many it replaced.
    /// Every other tensor, and the manifest's metadata, is carried byte for byte.
    ///
    /// A replaced tensor is stored in the codec's own element layout, so a
    /// later [`Self::bind`] over the written file finds the leaf already at its
    /// rule's target: [`crate::bind_safetensors_program_leaves`] filters a rule
    /// whose target equals the stored type (`precision_target_for`) and binds a
    /// zero-copy packed block, with no recode. The codec is read from the held
    /// `(name, bytes, codec)` tuple, never parsed from the `@<codec>` suffix,
    /// and the layout is accepted only where the shared binder reads it back
    /// packed (`expert_stack_codec`).
    ///
    /// Not persisted, and so still recoded by the next bind: a block-quant
    /// target (its stored form has no element dtype a safetensors header can
    /// declare) and a wide-target leaf, which the binder transposes to the
    /// program's `[in, out]` order and so no longer has the stored shape. Both
    /// are carried from the source unchanged.
    ///
    /// The write is [`Self::persist_aligned`] at an alignment of one byte:
    /// tensors are laid out back to back, so tensor offsets are not page aligned.
    ///
    /// # Errors
    ///
    /// As [`Self::persist_aligned`].
    pub fn persist(
        &self,
        manifest: &Manifest,
        file_bytes: &[u8],
        data_start: u64,
        out: &mut Vec<u8>,
    ) -> Result<usize, InteropError> {
        self.persist_aligned(manifest, file_bytes, data_start, 1, out)
    }

    /// As [`Self::persist`], composing
    /// [`proxima_safetensors::write_complete_aligned`]: every tensor of at
    /// least `alignment` bytes starts at an absolute file offset that is a
    /// multiple of `alignment`, with filler tensors in the gaps. A caller gets
    /// `alignment` from the backend (`omega::backend::no_copy_alignment`).
    ///
    /// # Errors
    ///
    /// [`InteropError::UnknownTensor`] when a manifest range lies outside
    /// `file_bytes`; [`InteropError::Safetensors`] from the writer.
    pub fn persist_aligned(
        &self,
        manifest: &Manifest,
        file_bytes: &[u8],
        data_start: u64,
        alignment: usize,
        out: &mut Vec<u8>,
    ) -> Result<usize, InteropError> {
        let held = self.held.lock();
        let mut substituted = 0;
        let mut tensors = Vec::with_capacity(manifest.tensors.len());
        for entry in manifest
            .tensors
            .iter()
            .filter(|entry| !is_filler_name(&entry.name))
        {
            let source = tensor_bytes(file_bytes, data_start, entry)?;
            let (dtype, data) = match held.element_recode(entry) {
                Some(recoded) => {
                    substituted += 1;
                    recoded
                }
                None => (entry.dtype, source),
            };
            tensors.push(TensorPayload {
                name: entry.name.clone(),
                dtype,
                shape: entry.shape.clone(),
                data,
            });
        }
        let metadata = manifest
            .metadata
            .iter()
            .filter(|(key, _)| key.as_str() != FORMAT_VERSION_KEY)
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        out.extend(write_complete_aligned(
            &SafetensorsModel { tensors, metadata },
            alignment,
        )?);
        Ok(substituted)
    }
}

/// A bind pass's [`BoundWeights`], on loan from a [`ModelWeights`]. Derefs to
/// [`BoundWeights`]; dropping it returns the materialized buffers to the model.
pub struct LeasedWeights<'file, 'store> {
    weights: BoundWeights<'file>,
    home: Option<&'store ModelWeights>,
}

impl LeasedWeights<'_, '_> {
    /// An empty lease that owns nothing and returns nothing, the starting
    /// value before a model has bound.
    #[must_use]
    pub fn detached() -> Self {
        Self {
            weights: BoundWeights::new(&[]),
            home: None,
        }
    }
}

impl<'file> Deref for LeasedWeights<'file, '_> {
    type Target = BoundWeights<'file>;

    fn deref(&self) -> &Self::Target {
        &self.weights
    }
}

impl Drop for LeasedWeights<'_, '_> {
    fn drop(&mut self) {
        if let Some(home) = self.home {
            home.held.lock().keep(&mut self.weights);
        }
    }
}
