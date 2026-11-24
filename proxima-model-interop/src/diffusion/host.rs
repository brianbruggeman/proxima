//! The two diffusion stages that run on the serving runtime: a [`Denoiser`]
//! (transformer backbone, one evaluation per Euler step) and a [`Decoder`]
//! (autoencoder decode). Each stage composes the same pieces the LLM decode
//! path does: one [`BackendRuntime`] built from [`ServingSettings`], weight
//! leaves named so the plan marks them resident, and one retained segment plan
//! per program.
//!
//! A runtime is per stage, not per run. A segment plan is keyed on the
//! program's address and never evicts, so the backbone's plan and device
//! arena must end with the backbone. Each stage owns its graph, and so its
//! program, for its whole life for the same reason: a program that moved or
//! was freed would key a new plan. Per-call inputs (latent, text, timestep)
//! arrive as already-built named blocks and are never resident: a reused
//! address with new contents would serve stale data.

use std::collections::BTreeSet;

use proxima_numeric::codec::Codec;
use proxima_tensor::op::NodeId;
use proxima_tensor::{Evaluated, QuantizedBlock};

use crate::ServingSettings;
use crate::generate::BackendRuntime;

use super::{Checkpoint, Graph, Weights, load_weights};

type Inputs<'a> = &'a [(&'a str, QuantizedBlock<'a>)];

/// One program, its weights and its runtime: what [`Denoiser`] and
/// [`Decoder`] share.
struct Stage {
    runtime: BackendRuntime,
    settings: ServingSettings,
    graph: Graph,
    weights: Weights,
    integers: Vec<Vec<i32>>,
    symbols: Vec<u64>,
    output: NodeId,
    #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
    profile: Option<BackendRuntime>,
}

fn runtime_for(settings: &ServingSettings) -> BackendRuntime {
    BackendRuntime::new(&settings.as_serving_config(&[]))
}

/// Every name the plan may keep on the device: the weights, then the host
/// leaves the stage owns for its whole life, never a name the caller supplies
/// per call.
fn resident_names<'a>(
    weight_names: impl Iterator<Item = &'a str>,
    host_names: impl Iterator<Item = &'a str>,
    input_names: &[&str],
) -> BTreeSet<&'a str> {
    let owned = host_names.filter(|name| !input_names.contains(name));
    weight_names.chain(owned).collect()
}

fn resident_of<'a>(weights: &'a Weights, graph: &'a Graph, inputs: Inputs<'_>) -> BTreeSet<&'a str> {
    let input_names: Vec<&str> = inputs.iter().map(|(name, _)| *name).collect();
    resident_names(
        weights.entries.iter().map(|(name, _)| name.as_str()),
        graph.hosts.iter().map(|host| host.name.as_str()),
        &input_names,
    )
}

/// The stage's own weights and host leaves, then the caller's per-call blocks;
/// a host leaf the caller supplies replaces the placeholder.
fn named_blocks<'a>(
    weights: &'a Weights,
    graph: &'a Graph,
    integers: &'a [Vec<i32>],
    inputs: Inputs<'a>,
) -> Vec<(&'a str, QuantizedBlock<'a>)> {
    let mut named = weights.blocks();
    named.extend(graph.host_blocks(&[], integers));
    named.retain(|(name, _)| !inputs.iter().any(|(input, _)| input == name));
    named.extend(inputs.iter().copied());
    named
}

impl Stage {
    fn open(
        checkpoint: &Checkpoint,
        graph: Graph,
        output: NodeId,
        symbols: &[u64],
        weights: (Codec, bool),
        settings: &ServingSettings,
    ) -> Result<Self, String> {
        let tokens = symbols
            .first()
            .ok_or_else(|| String::from("a stage needs its token count as the first symbol"))?;
        let mut staged = settings.clone();
        staged.context_length = u32::try_from(*tokens).map_err(|why| why.to_string())?;
        let loaded = load_weights(checkpoint, &graph.weights, weights.0, weights.1)?;
        Ok(Self {
            runtime: runtime_for(&staged),
            settings: staged,
            integers: graph.int_buffers(),
            graph,
            weights: loaded,
            symbols: symbols.to_vec(),
            output,
            #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
            profile: None,
        })
    }

    fn evaluate(&mut self, inputs: Inputs<'_>, outputs: &[NodeId]) -> Result<Evaluated, String> {
        let named = named_blocks(&self.weights, &self.graph, &self.integers, inputs);
        let resident = resident_of(&self.weights, &self.graph, inputs);
        self.runtime
            .evaluate_segment(
                &self.graph.program,
                &self.symbols,
                &named,
                outputs,
                &resident,
                None,
            )
            .map_err(|why| why.to_string())
    }

    /// The same program on a runtime of its own, planned afresh, so a capture
    /// that reads the environment at plan time sees it.
    #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
    fn evaluate_fresh(&self, inputs: Inputs<'_>) -> Result<Evaluated, String> {
        let named = named_blocks(&self.weights, &self.graph, &self.integers, inputs);
        let resident = resident_of(&self.weights, &self.graph, inputs);
        runtime_for(&self.settings)
            .evaluate_segment(
                &self.graph.program,
                &self.symbols,
                &named,
                &[self.output],
                &resident,
                None,
            )
            .map_err(|why| why.to_string())
    }

    #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
    fn evaluate_op_timed(
        &mut self,
        inputs: Inputs<'_>,
    ) -> Result<(Evaluated, Vec<omega::metal::OpGpuTiming>), String> {
        let settings = &self.settings;
        let runtime = self.profile.get_or_insert_with(|| runtime_for(settings));
        let named = named_blocks(&self.weights, &self.graph, &self.integers, inputs);
        let resident = resident_of(&self.weights, &self.graph, inputs);
        runtime
            .evaluate_segment_op_timed(
                &self.graph.program,
                &self.symbols,
                &named,
                &[self.output],
                &resident,
                &Default::default(),
            )
            .map_err(|why| why.to_string())
    }
}

/// The transformer backbone as a retained plan: build once from a graph and a
/// checkpoint, then [`Self::evaluate`] per Euler step. The first step plans
/// (one miss); every later step reuses the plan and the resident weights.
///
/// Composes [`load_weights`] over one serving runtime's `evaluate_segment`, the
/// path whisper's stages and the LLM decode take, rather than a bare
/// `plan_named` per caller. The caller builds the graph (its host leaves and
/// `build_dit`), so the model's constants stay at the call site as data.
pub struct Denoiser {
    stage: Stage,
}

impl Denoiser {
    /// Binds `graph`'s weights from `checkpoint` at `weights` and opens the
    /// stage's runtime from `settings`; `symbols[0]` is the stage's token
    /// count, which becomes the runtime's `context_length`.
    ///
    /// # Errors
    ///
    /// A message naming the weight load or the missing token count.
    pub fn new(
        checkpoint: &Checkpoint,
        graph: Graph,
        velocity: NodeId,
        symbols: &[u64],
        weights: Codec,
        settings: &ServingSettings,
    ) -> Result<Self, String> {
        let stage = Stage::open(checkpoint, graph, velocity, symbols, (weights, false), settings)?;
        Ok(Self { stage })
    }

    /// One Euler step's evaluation of the velocity node and `extra` outputs,
    /// with the step's latent, text and timestep as `inputs`.
    ///
    /// # Errors
    ///
    /// A message naming the plan or execution failure.
    pub fn evaluate(
        &mut self,
        inputs: &[(&str, QuantizedBlock<'_>)],
        extra: &[NodeId],
    ) -> Result<Evaluated, String> {
        let mut outputs = vec![self.stage.output];
        outputs.extend_from_slice(extra);
        self.stage.evaluate(inputs, &outputs)
    }

    /// The velocity node [`Self::evaluate`] always returns.
    #[must_use]
    pub fn velocity(&self) -> NodeId {
        self.stage.output
    }

    #[must_use]
    pub fn graph(&self) -> &Graph {
        &self.stage.graph
    }

    /// `(hits, misses)` of the stage's segment-plan cache.
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn plan_counts(&self) -> (usize, usize) {
        (self.stage.runtime.plan_hits, self.stage.runtime.plan_misses)
    }
}

/// The autoencoder decode as a retained plan, over the same serving runtime
/// the [`Denoiser`] uses but a runtime of its own.
pub struct Decoder {
    stage: Stage,
}

impl Decoder {
    /// Binds `graph`'s weights from `checkpoint` at `weights` (convolution
    /// kernels permuted to tap-inner order) and opens the stage's runtime from
    /// `settings`; `symbols[0]` is the stage's row count.
    ///
    /// # Errors
    ///
    /// A message naming the weight load or the missing row count.
    pub fn new(
        checkpoint: &Checkpoint,
        graph: Graph,
        image: NodeId,
        symbols: &[u64],
        weights: Codec,
        settings: &ServingSettings,
    ) -> Result<Self, String> {
        let stage = Stage::open(checkpoint, graph, image, symbols, (weights, true), settings)?;
        Ok(Self { stage })
    }

    /// Decodes one de-normalised, unpatchified latent given as `inputs`.
    ///
    /// # Errors
    ///
    /// A message naming the plan or execution failure.
    pub fn evaluate(&mut self, inputs: &[(&str, QuantizedBlock<'_>)]) -> Result<Evaluated, String> {
        self.stage.evaluate(inputs, &[self.stage.output])
    }

    /// The image node [`Self::evaluate`] returns.
    #[must_use]
    pub fn image(&self) -> NodeId {
        self.stage.output
    }

    #[must_use]
    pub fn graph(&self) -> &Graph {
        &self.stage.graph
    }

    /// `(hits, misses)` of the stage's segment-plan cache.
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn plan_counts(&self) -> (usize, usize) {
        (self.stage.runtime.plan_hits, self.stage.runtime.plan_misses)
    }

    /// [`Self::evaluate`] with one command buffer per op, returning each op's
    /// GPU time. It plans on a profile-local runtime, so a timed plan never
    /// enters the decode cache.
    ///
    /// # Errors
    ///
    /// A message naming the plan or execution failure.
    #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
    pub fn evaluate_op_timed(
        &mut self,
        inputs: &[(&str, QuantizedBlock<'_>)],
    ) -> Result<(Evaluated, Vec<omega::metal::OpGpuTiming>), String> {
        self.stage.evaluate_op_timed(inputs)
    }

    /// [`Self::evaluate`] on a fresh runtime that plans anew, for a capture
    /// that reads the environment at plan time.
    ///
    /// # Errors
    ///
    /// A message naming the plan or execution failure.
    #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
    pub fn evaluate_fresh(
        &self,
        inputs: &[(&str, QuantizedBlock<'_>)],
    ) -> Result<Evaluated, String> {
        self.stage.evaluate_fresh(inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_names_are_weights_and_owned_host_leaves_never_per_call_inputs() {
        let weights = ["blocks.0.norm.weight"];
        let hosts = ["latent", "timestep", "rope_cos", "gather.index"];

        let resident = resident_names(
            weights.into_iter(),
            hosts.into_iter(),
            &["latent", "timestep"],
        );

        assert_eq!(
            resident.into_iter().collect::<Vec<_>>(),
            vec!["blocks.0.norm.weight", "gather.index", "rope_cos"]
        );
    }
}
