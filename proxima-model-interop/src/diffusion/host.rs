//! The two diffusion stages that run on the serving runtime: a [`Denoiser`]
//! (transformer backbone, one evaluation per Euler step) and a [`Decoder`]
//! (autoencoder decode). Each stage composes the same pieces the LLM decode
//! path does: one [`BackendRuntime`] built from a [`ServingConfig`], weight
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

use proxima_numeric::PackedRows;
use proxima_numeric::codec::Codec;
use proxima_tensor::op::NodeId;
use proxima_tensor::{Evaluated, QuantizedBlock};

use crate::generate::BackendRuntime;
use crate::{BoundWeights, ContextLength, ServingConfig};

use super::{Checkpoint, Graph, load_weights};

type Inputs<'a> = &'a [(&'a str, QuantizedBlock<'a>)];

/// One program, its weights and its runtime: what [`Denoiser`] and
/// [`Decoder`] share.
struct Stage {
    runtime: BackendRuntime,
    graph: Graph,
    weights: BoundWeights<'static>,
    symbols: Vec<u64>,
    output: NodeId,
    /// Profile runs plan on runtimes of their own, built beside the stage's:
    /// a one-command-buffer-per-op plan and a capture plan share the stage's
    /// program key, so either in the stage's own cache would be served to a
    /// plain evaluation.
    #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
    profile: BackendRuntime,
    #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
    capture: BackendRuntime,
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

fn resident_of<'a>(
    weights: &'a BoundWeights<'static>,
    graph: &'a Graph,
    inputs: Inputs<'_>,
) -> BTreeSet<&'a str> {
    let input_names: Vec<&str> = inputs.iter().map(|(name, _)| *name).collect();
    resident_names(
        weights.names().into_iter(),
        graph.hosts.iter().map(|host| host.name.as_str()),
        &input_names,
    )
}

pub(super) fn widened(name: &str, rows: &PackedRows) -> Result<Option<PackedRows>, String> {
    if rows.format().is_host_element() {
        return Ok(None);
    }
    rows.to_host_element()
        .map(Some)
        .map_err(|why| format!("{name}: {why}"))
}

/// A leaf's bytes in the host element, bound as the decoded-values block the backends read. A
/// leaf whose allocation cannot be viewed in place is an error, never a silent codec-operand bind.
pub(super) fn leaf_block<'a>(
    name: &str,
    rows: &'a PackedRows,
    widened: Option<&'a PackedRows>,
) -> Result<QuantizedBlock<'a>, String> {
    let source = widened.unwrap_or(rows);
    let block = QuantizedBlock::Packed {
        codec: *source.format(),
        bytes: source.as_bytes(),
    };
    host_checked(name, block)
}

/// A block bound as the decoded-values view when its format is the host element. Bytes that
/// cannot be viewed in place are an error naming the input, never a silent codec-operand bind;
/// any other format passes through as it came.
pub(super) fn host_checked<'a>(
    name: &str,
    block: QuantizedBlock<'a>,
) -> Result<QuantizedBlock<'a>, String> {
    let host_element = matches!(
        &block,
        QuantizedBlock::Packed { codec, .. } if codec.is_host_element()
    );
    match block.host_view() {
        QuantizedBlock::Packed { .. } if host_element => Err(format!(
            "input {name}: its bytes cannot be viewed as the host element in place"
        )),
        viewed => Ok(viewed),
    }
}

/// The stage's own weights and host leaves, then the caller's per-call blocks;
/// a host leaf the caller supplies replaces the placeholder.
fn name_blocks<'a>(
    weights: &'a BoundWeights<'static>,
    graph: &'a Graph,
    inputs: Inputs<'a>,
) -> Result<Vec<(&'a str, QuantizedBlock<'a>)>, String> {
    let mut named = weights.blocks().map_err(|why| why.to_string())?;
    named.retain(|(name, _)| !inputs.iter().any(|(input, _)| input == name));
    named.reserve(graph.hosts.len() + inputs.len());
    for host in &graph.hosts {
        if !inputs.iter().any(|(input, _)| *input == host.name) {
            let block = leaf_block(&host.name, &host.values, host.widened.as_ref())?;
            named.push((host.name.as_str(), block));
        }
    }
    for (name, block) in inputs {
        named.push((*name, host_checked(name, *block)?));
    }
    Ok(named)
}

impl Stage {
    fn open(
        checkpoint: &Checkpoint,
        graph: Graph,
        output: NodeId,
        symbols: &[u64],
        weights: (Codec, bool),
        config: &ServingConfig<'_>,
    ) -> Result<Self, String> {
        let loaded = load_weights(checkpoint, &graph.weights, weights.0, weights.1)?;
        Self::with_weights(graph, output, symbols, loaded, config)
    }

    fn with_weights(
        graph: Graph,
        output: NodeId,
        symbols: &[u64],
        weights: BoundWeights<'static>,
        config: &ServingConfig<'_>,
    ) -> Result<Self, String> {
        graph.check_faults()?;
        let tokens = symbols
            .first()
            .ok_or_else(|| String::from("a stage needs its token count as the first symbol"))?;
        let mut staged = *config;
        staged.context_length =
            ContextLength::Within(u32::try_from(*tokens).map_err(|why| why.to_string())?);
        Ok(Self {
            runtime: BackendRuntime::new(&staged),
            graph,
            weights,
            symbols: symbols.to_vec(),
            output,
            #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
            profile: BackendRuntime::new(&staged),
            #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
            capture: BackendRuntime::new(&staged),
        })
    }

    fn evaluate(&mut self, inputs: Inputs<'_>, outputs: &[NodeId]) -> Result<Evaluated, String> {
        let resident = resident_of(&self.weights, &self.graph, inputs);
        let named = name_blocks(&self.weights, &self.graph, inputs)?;
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

    /// The same program on the capture runtime, which plans on its first call,
    /// so a capture that reads the environment at plan time sees it.
    #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
    fn evaluate_fresh(&mut self, inputs: Inputs<'_>) -> Result<Evaluated, String> {
        let resident = resident_of(&self.weights, &self.graph, inputs);
        let named = name_blocks(&self.weights, &self.graph, inputs)?;
        self.capture
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
        let resident = resident_of(&self.weights, &self.graph, inputs);
        let named = name_blocks(&self.weights, &self.graph, inputs)?;
        self.profile
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
    /// stage's runtime from `config` (a [`ServingConfig`], so a tile-layout A/B
    /// attaches its omega runtime with `with_omega_runtime`); `symbols[0]` is
    /// the stage's token count, which becomes the runtime's `context_length`.
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
        config: &ServingConfig<'_>,
    ) -> Result<Self, String> {
        let stage = Stage::open(checkpoint, graph, velocity, symbols, (weights, false), config)?;
        Ok(Self { stage })
    }

    /// A stage over weights the caller already holds, for a program whose
    /// weights do not come from a checkpoint (a micro-benchmark's synthetic
    /// leaf, or a graph with none). They are resident like a loaded stage's.
    ///
    /// # Errors
    ///
    /// A message naming the missing token count.
    pub fn with_weights(
        graph: Graph,
        velocity: NodeId,
        symbols: &[u64],
        weights: BoundWeights<'static>,
        config: &ServingConfig<'_>,
    ) -> Result<Self, String> {
        let stage = Stage::with_weights(graph, velocity, symbols, weights, config)?;
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

    /// [`Self::evaluate`] of the velocity node alone, with one command buffer
    /// per op and each op's GPU time. It plans on a profile-local runtime, so a
    /// timed plan never enters the step cache.
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
    /// `config`; `symbols[0]` is the stage's row count.
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
        config: &ServingConfig<'_>,
    ) -> Result<Self, String> {
        let stage = Stage::open(checkpoint, graph, image, symbols, (weights, true), config)?;
        Ok(Self { stage })
    }

    /// A decoder over weights the caller already holds; see
    /// [`Denoiser::with_weights`].
    ///
    /// # Errors
    ///
    /// A message naming the missing row count.
    pub fn with_weights(
        graph: Graph,
        image: NodeId,
        symbols: &[u64],
        weights: BoundWeights<'static>,
        config: &ServingConfig<'_>,
    ) -> Result<Self, String> {
        let stage = Stage::with_weights(graph, image, symbols, weights, config)?;
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

    /// [`Self::evaluate`] for exactly the nodes in `outputs`, for a probe that
    /// asks where a decode stage faults or drifts. Each distinct output set
    /// plans once.
    ///
    /// # Errors
    ///
    /// A message naming the plan or execution failure.
    pub fn evaluate_nodes(
        &mut self,
        inputs: &[(&str, QuantizedBlock<'_>)],
        outputs: &[NodeId],
    ) -> Result<Evaluated, String> {
        self.stage.evaluate(inputs, outputs)
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

    /// [`Self::evaluate`] on the capture runtime, which plans on its first
    /// call, for a capture that reads the environment at plan time.
    ///
    /// # Errors
    ///
    /// A message naming the plan or execution failure.
    #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
    pub fn evaluate_fresh(
        &mut self,
        inputs: &[(&str, QuantizedBlock<'_>)],
    ) -> Result<Evaluated, String> {
        self.stage.evaluate_fresh(inputs)
    }
}

#[cfg(test)]
mod tests {
    use proxima_numeric::NumericFormat;

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

    #[test]
    fn a_misaligned_host_element_input_is_refused_by_name() {
        let values = [1.5, -2.25, 0.0, 4.0, 7.0];
        let rows =
            PackedRows::from_values(&Codec::host_element(), &values).expect("host element encodes");
        let mut shifted = vec![0_u8; 8];
        shifted.extend_from_slice(rows.as_bytes());
        let skip = (0..8)
            .find(|skip| (shifted.as_ptr() as usize + skip) % 8 == 1)
            .expect("one offset in eight is one past an eight-byte boundary");
        let aligned = QuantizedBlock::Packed {
            codec: Codec::host_element(),
            bytes: rows.as_bytes(),
        };
        let misaligned = QuantizedBlock::Packed {
            codec: Codec::host_element(),
            bytes: &shifted[skip..skip + rows.as_bytes().len()],
        };
        let other_format = NumericFormat::ALL
            .iter()
            .find(|format| !format.is_host_element())
            .expect("a non host-element format exists");
        let codec_operand = QuantizedBlock::Packed {
            codec: *other_format,
            bytes: &shifted[skip..skip + rows.as_bytes().len()],
        };

        let refused = host_checked("latent", misaligned);
        let viewed = host_checked("latent", aligned).expect("aligned bytes are viewed");
        let passed = host_checked("text", codec_operand).expect("another format passes through");

        assert!(viewed.packed_bytes().is_none());
        assert!(matches!(passed, QuantizedBlock::Packed { .. }));
        let why = refused.expect_err("misaligned host-element bytes are refused");
        assert!(why.contains("latent"), "{why}");
    }
}
