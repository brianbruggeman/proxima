//! Reachable text generation: bind a checkpoint's weights once, then
//! generate text repeatedly against the bound weights without re-paying
//! the load cost.
//!
//! [`LoadedModel`] is the transform pipe (`In = (String, usize), Out =
//! (Vec<u32>, String, bool)` -- `proxima_primitives::pipe::Pipe`,
//! `proxima-primitives/src/pipe/primitives.rs:91-102`'s general form,
//! since neither `In` nor `Out` is `()`). [`LoadedModel::load`] is a plain
//! constructor, not a pipe: it pays the expensive one-time cost (mmap +
//! parse + bind 226 tensors, ~4 GB / ~120 ms prefault on the real
//! openchat-3.5 checkpoint -- `crate::bind::bind_all_weights`'s own doc)
//! and hands back a value that [`Pipe::call`] is then cheap to invoke many
//! times against, one call per generation request, without rebinding.
//! That two-step shape is the direct answer to "load once, generate
//! repeatedly": a caller holds one `LoadedModel` and calls it as many
//! times as it wants, exactly the way a caller holds one bound
//! `TcpListener` and accepts many connections from it.
//!
//! `call`'s body is synchronous CPU work wrapped in `async move { .. }`
//! with no internal `.await` -- the same shape `Pipe`'s own doc's
//! `Double`/`Always`/`Discard`/`Echo` examples use. It is still the right
//! trait: the algebra's whole point is that combinators (retry, tee,
//! rate-limit, ...) compose over `Pipe` regardless of whether a given
//! impl happens to yield control anywhere inside.
//!
//! # Stopping: the model's own signal, not just the caller's budget
//!
//! `Out`'s third field is `true` exactly when decoding stopped because the
//! model emitted its own end-of-sequence token, `false` when it stopped
//! because `max_tokens` ran out first -- the two are otherwise
//! indistinguishable to a caller (`generated_ids.len() < max_tokens` is
//! not proof of an early stop if `max_tokens` itself was small). A plain
//! `bool` earns this over a new enum because this checkpoint's own
//! metadata defines exactly one stopping condition to check, confirmed by
//! reading it rather than assumed: on the real openchat-3.5-1210 fixture
//! (`~/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf`),
//! `tokenizer.ggml.eos_token_id = 32000`, which is *not* the SentencePiece
//! `</s>` (id 2) -- it is `<|end_of_turn|>`, a [`proxima_tokenizer::vocab::TokenType::Control`]
//! entry, and the same id OpenChat's own `tokenizer.chat_template` emits
//! between turns. There is no separate `tokenizer.ggml.eot_token_id` (or
//! similar) key on this fixture; the GGUF writer already folded the
//! turn-boundary marker into the one `eos_token_id` slot
//! [`proxima_tokenizer::Vocab::eos_token_id`] reads. So checking a single
//! id against [`Vocab::eos_token_id`] is this fixture's whole stopping
//! condition -- a `bool` carries it exactly; an enum would be modeling a
//! multi-token-family case this checkpoint does not have.
//!
//! The stop token itself is excluded from both the returned ids and the
//! returned text (never pushed onto `generated_ids` before the loop
//! breaks) -- symmetric exclusion, not just from decoded text, because a
//! caller who re-feeds `generated_ids` as a future prompt's tokens should
//! never see a turn-boundary marker reappear as if it were generated
//! content.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use alloc::vec::Vec;
use core::future::Future;
use core::marker::PhantomData;
use core::ops::Range;

use memmap2::{Advice, Mmap};
#[cfg(feature = "mlx-gdn")]
use omega::mlx;
use proxima_gguf::GgmlType;
use proxima_gguf::pipe::ParsedGguf;
use proxima_primitives::pipe::Pipe;
#[cfg(any(
    not(feature = "metal"),
    all(feature = "instrument", target_os = "macos")
))]
use proxima_tensor::cpu::evaluate_quantized_named_with_scratch_and_experts;
use proxima_tensor::cpu::{
    Evaluated, ExpertSource, GdnPrefillScan, GdnPrefillShape, QuantizedBlock,
    evaluate_quantized_named_exact_with_scratch_and_experts, run_gdn_prefill_scan,
};
use proxima_tensor::op::{Extent, NodeId, Op};
#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
use proxima_tensor::spec::CachedLayerRoots;
use proxima_tensor::spec::{
    Qwen35LayerRoots, mistral_cached_forward_program_with_experts_and_layer_taps,
};
use proxima_tokenizer::{SamplingConfig, Vocab, sample_next_token};
use std::cell::RefCell;
use std::fs::File;
use std::sync::Arc;

#[cfg(all(
    feature = "instrument",
    feature = "metal",
    target_os = "macos",
    not(feature = "metal-output-placement")
))]
use omega::backend::execute_plan_named;
#[cfg(all(
    feature = "instrument",
    feature = "metal",
    target_os = "macos",
    not(feature = "metal-output-placement")
))]
use omega::backend::execute_plan_named_metal_op_timed;
#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
use omega::backend::execute_plan_named_metal_op_timed_with_expert_sources;
#[cfg(feature = "metal")]
use omega::backend::{
    Engine, Plan, clear_expert_source_cache, execute_plan_named_with_expert_sources, mark_resident,
    plan_named, plan_named_exact, release_resident_names, unregister_checkpoint_mapping,
};
// `set_math_mode` (unlike `mark_resident` above) takes `metal::MathMode` in
// its own signature, so unlike the ungated import above it needs the same
// `metal`+macos gate that type itself lives behind.
#[cfg(all(feature = "metal", target_os = "macos"))]
use omega::backend::set_math_mode;
// `set_dispatch_type` (unlike `mark_resident` above) takes `metal::DispatchType`
// in its own signature, so it needs the same `metal`+macos gate that type
// itself lives behind -- same reasoning as `set_math_mode` above.
#[cfg(all(feature = "metal", target_os = "macos"))]
use omega::backend::set_dispatch_type;
#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
use omega::metal::OpGpuTiming;
#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
use omega::metal::metal_stage_totals;
// Persistent device-resident KV: `PlacedBuffer`/`allocate_placed_buffer`/
// `execute_plan_named_with_placements` are `omega`'s own default-off
// `metal-output-placement` surface (`omega/src/metal.rs`'s own doc on
// `execute_plan_with_placements`) -- this crate's identically-named,
// identically default-off feature is a straight passthrough
// (`Cargo.toml`'s `metal-output-placement` entry), never a second gate.
// `plan_named` here is `omega::metal`'s own (the Metal-`Plan`-typed one,
// aliased to avoid colliding with `omega::backend::plan_named` above,
// which returns the backend-polymorphic `omega::backend::Plan` enum
// `PlacedBuffer` placement has no arm for) --
// `mistral_single_range_cached_forward_program` is this call's program
// builder, `proxima-tensor/src/spec.rs`'s own single-range counterpart to
// `mistral_cached_forward_program_with_experts`.
#[cfg(all(
    feature = "metal-output-placement",
    feature = "instrument",
    target_os = "macos"
))]
use omega::execute_plan_named_with_placements_dispatch_timed;
#[cfg(all(
    feature = "metal-output-placement",
    feature = "instrument",
    target_os = "macos"
))]
use omega::execute_plan_named_with_placements_op_timed;
#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
use omega::{
    PlacedBuffer, allocate_placed_buffer, execute_plan_named_with_placements,
    execute_plan_named_with_placements_and_expert_sources, plan_named_with_placed_inputs,
};
#[cfg(all(
    feature = "metal-output-placement",
    feature = "instrument",
    target_os = "macos"
))]
use omega::read_placed_buffer_f32;
#[cfg(feature = "instrument")]
use proxima_telemetry::{debug, info};
#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
use proxima_tensor::TensorError;
#[cfg(feature = "instrument")]
use proxima_tensor::instrument::{elapsed_ticks, read_ticks, ticks_to_nanos};
#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
use proxima_tensor::spec::{DuplicateHeadPosition, mistral_single_range_cached_forward_program};

use crate::architecture::{Architecture, StepInput, StepInputContext, bind_symbols};
use crate::bind::{
    BoundWeights, ModelArchitecture, PackedOwnedKind, architecture_from_metadata, bind_all_weights,
};
use crate::error::InteropError;
use crate::hf_bind::bind_all_weights_from_safetensors;
#[cfg(feature = "metal")]
use crate::serving::GPU_LAYERS_ALL;
use crate::serving::apply_serving_config;
use crate::serving::{GdnPrefillBackend, ServingConfig};


#[macro_use]
mod load_model;
#[macro_use]
mod pregather;
#[macro_use]
mod residency_caches;
#[macro_use]
mod decode;
mod tests_all;
pub use load_model::*;
pub(crate) use pregather::*;
pub use residency_caches::*;
use decode::*;
