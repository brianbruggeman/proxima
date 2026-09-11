//! Typed phase boundary for a routed `qwen35moe` decode step.
//!
//! The router and expert gather cannot be represented as one opaque callback
//! when a residency policy must run between them.  These consuming phase
//! values make the ordering explicit without a heap allocation or dynamic
//! dispatch.  The graph evaluator remains owned by the caller: this module
//! only carries the router result and the source table across the boundary.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use proxima_tensor::cpu::{Evaluated, QuantizedBlock};
use proxima_tensor::error::TensorError;
use proxima_tensor::op::{NodeId, Op};
use proxima_tensor::partition::{partition_at, partition_between, partition_between_with_mapping};

use crate::residency::{ExpertAddress, ServeDecision};

/// The two graph halves surrounding one routed layer's router output.
///
/// `router_program` produces the activation named by `cut_inputs`; the
/// `gather_program` accepts that activation as a named input.  The vectors
/// are built once, while the activation bytes remain caller-owned, so the
/// boundary can be reused by a CPU or device evaluator without a boxed
/// callback or a dynamic graph object.
#[derive(Debug, Clone)]
pub struct LayerProgramBoundary {
    router_program: Vec<Op>,
    cut_inputs: Vec<(NodeId, String)>,
    gather_program: Vec<Op>,
    router_output: NodeId,
    gather_output: NodeId,
}

/// A caller-owned activation crossing from the router program to the gather
/// program.  The evaluator may use `values` directly; `shape` documents the
/// concrete cut contract without copying the tensor or allocating a buffer.
#[derive(Debug, Clone, Copy)]
pub struct ActivationHandoff<'data> {
    name: &'data str,
    values: &'data [f32],
    shape: &'data [usize],
}

impl<'data> ActivationHandoff<'data> {
    /// Names and exposes the producer output for a consumer binding.
    #[must_use]
    pub const fn new(name: &'data str, values: &'data [f32], shape: &'data [usize]) -> Self {
        Self {
            name,
            values,
            shape,
        }
    }

    #[must_use]
    pub const fn name(&self) -> &'data str {
        self.name
    }

    #[must_use]
    pub const fn values(&self) -> &'data [f32] {
        self.values
    }

    #[must_use]
    pub const fn shape(&self) -> &'data [usize] {
        self.shape
    }
}

impl LayerProgramBoundary {
    /// Borrows producer cut values in the named-block representation consumed
    /// by the next segment. No activation copy is made.
    pub fn cut_bindings<'data>(
        &'data self,
        evaluated: &'data Evaluated,
    ) -> Result<Vec<(&'data str, QuantizedBlock<'data>)>, NodeId> {
        self.cut_inputs
            .iter()
            .map(|(node, name)| {
                let (values, _) = evaluated.get(*node).ok_or(*node)?;
                Ok((name.as_str(), QuantizedBlock::Float32(values)))
            })
            .collect()
    }

    /// Splits a graph immediately after `router_output`.  `gather_output`
    /// must be a later node in the same graph; this prevents a caller from
    /// accidentally treating a router-only result as the completed layer.
    pub fn split(
        program: &[Op],
        symbols: &[u64],
        router_output: NodeId,
        gather_output: NodeId,
    ) -> Result<Self, TensorError> {
        if gather_output.0 <= router_output.0 {
            return Err(TensorError::UnknownOutput(gather_output));
        }
        let (router_program, cut_inputs, gather_program) =
            partition_at(program, symbols, router_output)?;
        let gather_output = NodeId(
            cut_inputs
                .len()
                .saturating_add(gather_output.0 as usize)
                .saturating_sub(router_output.0 as usize + 1) as u32,
        );
        Ok(Self {
            router_program,
            cut_inputs,
            gather_program,
            router_output,
            gather_output,
        })
    }

    /// The producer program, evaluated before the residency boundary.
    #[must_use]
    pub fn router_program(&self) -> &[Op] {
        &self.router_program
    }

    /// The consumer program, evaluated after expert sources are selected.
    #[must_use]
    pub fn gather_program(&self) -> &[Op] {
        &self.gather_program
    }

    /// Producer node/name pairs used to bind the activation into the gather.
    #[must_use]
    pub fn cut_inputs(&self) -> &[(NodeId, String)] {
        &self.cut_inputs
    }

    /// Returns the consumer-side input name for an original producer node.
    /// A routed layer can cross several values at once (logits, selected
    /// experts, and routing weights); callers must bind each by identity
    /// rather than assuming the router value is the first cut.
    #[must_use]
    pub fn cut_input_name(&self, node: NodeId) -> Option<&str> {
        self.cut_inputs
            .iter()
            .find(|(cut_node, _)| *cut_node == node)
            .map(|(_, name)| name.as_str())
    }

    /// Turns a producer result into the zero-copy activation handoff accepted
    /// by the gather evaluator.
    pub fn activation_handoff<'data>(
        &'data self,
        values: &'data [f32],
        shape: &'data [usize],
    ) -> Result<ActivationHandoff<'data>, TensorError> {
        let (_, name) = self
            .cut_inputs
            .iter()
            .find(|(node, _)| *node == self.router_output)
            .ok_or(TensorError::UnknownOutput(self.router_output))?;
        Ok(ActivationHandoff::new(name, values, shape))
    }

    /// The producer-side router node to request from the first evaluation.
    #[must_use]
    pub const fn router_output(&self) -> NodeId {
        self.router_output
    }

    /// The renumbered gather root to request from the second evaluation.
    #[must_use]
    pub const fn gather_output(&self) -> NodeId {
        self.gather_output
    }
}

/// Builds the explicit router/gather handoff used by qwen35moe's per-layer
/// pre-gather execution.  This is deliberately a thin named wrapper around
/// the tensor partition algebra so future routed architectures share the
/// same cut semantics.
pub fn split_layer_program(
    program: &[Op],
    symbols: &[u64],
    router_output: NodeId,
    gather_output: NodeId,
) -> Result<LayerProgramBoundary, TensorError> {
    LayerProgramBoundary::split(program, symbols, router_output, gather_output)
}

/// Builds one sequential layer segment from the previous layer boundary to
/// the current router. Unlike [`split_layer_program`], this does not retain
/// the entire remaining graph, so a chain of routed layers can be evaluated
/// without duplicating suffix programs.
pub fn split_layer_segment(
    program: &[Op],
    symbols: &[u64],
    previous_output: Option<NodeId>,
    current_output: NodeId,
) -> Result<(Vec<Op>, Vec<(NodeId, String)>), TensorError> {
    partition_between(program, symbols, previous_output, current_output)
}

/// A sequential segment plus the original-to-dense ids produced by its
/// stable topological ordering.
pub type MappedLayerSegment = (Vec<Op>, Vec<(NodeId, String)>, BTreeMap<NodeId, NodeId>);

/// [`split_layer_segment`] with the node mapping required by execution.
pub fn split_mapped_layer_segment(
    program: &[Op],
    symbols: &[u64],
    previous_output: Option<NodeId>,
    current_output: NodeId,
) -> Result<MappedLayerSegment, TensorError> {
    partition_between_with_mapping(program, symbols, previous_output, current_output)
}

/// Builds executable router and gather segments with their dense node maps.
pub fn split_mapped_router_and_gather_segments(
    program: &[Op],
    symbols: &[u64],
    previous_layer_output: Option<NodeId>,
    router_output: NodeId,
    layer_output: NodeId,
) -> Result<(MappedLayerSegment, MappedLayerSegment), TensorError> {
    let router =
        split_mapped_layer_segment(program, symbols, previous_layer_output, router_output)?;
    let gather = split_mapped_layer_segment(program, symbols, Some(router_output), layer_output)?;
    Ok((router, gather))
}

/// Builds the two sequential segments for one routed layer. The first ends
/// at the router output; the second continues through the routed block output.
/// Keeping these as separate dense programs is what lets a caller apply a
/// residency transition between them without retaining the complete suffix.
pub fn split_router_and_gather_segments(
    program: &[Op],
    symbols: &[u64],
    previous_layer_output: Option<NodeId>,
    router_output: NodeId,
    layer_output: NodeId,
) -> Result<
    (
        (Vec<Op>, Vec<(NodeId, String)>),
        (Vec<Op>, Vec<(NodeId, String)>),
    ),
    TensorError,
> {
    let router = split_layer_segment(program, symbols, previous_layer_output, router_output)?;
    let gather = split_layer_segment(program, symbols, Some(router_output), layer_output)?;
    Ok((router, gather))
}

/// The ordinary decode path evaluates one graph and does not use the
/// pre-gather protocol.  `PreGather` opts a caller into the phase seam.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35MoeExecutionMode {
    SinglePass,
    PreGather,
}

/// Starts one pre-gather step.
#[derive(Debug, Clone, Copy, Default)]
pub struct PreGatherStep;

/// Router output handed to the residency boundary.
#[derive(Debug, Clone, Copy)]
pub struct RouterResult<'routes> {
    routes: &'routes [ServeDecision],
}

impl<'routes> RouterResult<'routes> {
    /// The routes selected by the router prepass.
    #[must_use]
    pub fn routes(&self) -> &'routes [ServeDecision] {
        self.routes
    }
}

/// The only phase in which a caller may apply HOBBIT/DynaExq actions.
#[derive(Debug, Clone, Copy)]
pub struct ResidencyBoundary<'routes> {
    result: RouterResult<'routes>,
}

impl PreGatherStep {
    /// Creates the first phase of a pre-gather step.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Completes the router prepass and opens the step-boundary phase.
    #[must_use]
    pub const fn router_complete<'routes>(
        self,
        routes: &'routes [ServeDecision],
    ) -> ResidencyBoundary<'routes> {
        ResidencyBoundary {
            result: RouterResult { routes },
        }
    }
}

impl<'routes> ResidencyBoundary<'routes> {
    /// Returns the router output that the residency policy must inspect.
    #[must_use]
    pub fn router_result(&self) -> RouterResult<'routes> {
        self.result
    }

    /// Applies the caller's already-computed boundary transition and opens
    /// the expert-gather phase.  The source table is generic so a fixed slab,
    /// an mmap-backed table, or another statically known representation can
    /// cross the boundary without `dyn` or boxing.
    #[must_use]
    pub const fn residency_complete<Source>(self, sources: Source) -> GatherPhase<'routes, Source> {
        GatherPhase {
            routes: self.result,
            sources,
        }
    }
}

/// Permission to evaluate the expert gather after residency has changed.
#[derive(Debug, Clone, Copy)]
pub struct GatherPhase<'routes, Source> {
    routes: RouterResult<'routes>,
    sources: Source,
}

/// Executes the three consuming phases of a routed step.  The callbacks are
/// deliberately supplied by the runtime: the router callback is the only
/// place that may produce routes, the boundary callback is the only place
/// that may change expert sources, and the gather callback cannot run until
/// both have returned.  This keeps the ordering usable with a fixed slab,
/// mmap view, or another caller-owned source table without `dyn` or boxing.
pub fn execute_pre_gather<Routes, Source, Output, Error, Router, Boundary, Gather>(
    router: Router,
    boundary: Boundary,
    gather: Gather,
) -> Result<Output, Error>
where
    Router: FnOnce() -> Result<Routes, Error>,
    Routes: AsRef<[ServeDecision]>,
    Boundary: FnOnce(RouterResult<'_>) -> Result<Source, Error>,
    Gather: FnOnce(GatherPhase<'_, Source>) -> Result<Output, Error>,
{
    // A callback-owned route buffer cannot be borrowed across this generic
    // boundary without imposing a lifetime on every caller.  The concrete
    // runtime uses `Vec<ServeDecision>` and this adapter keeps the phase
    // protocol itself independent of that storage choice.
    let routes = router()?;
    let route_result = RouterResult {
        routes: routes.as_ref(),
    };
    let source = boundary(route_result)?;
    gather(GatherPhase {
        routes: route_result,
        sources: source,
    })
}

impl<'routes, Source> GatherPhase<'routes, Source> {
    /// Returns the source table selected for this gather.
    #[must_use]
    pub const fn sources(&self) -> &Source {
        &self.sources
    }

    /// Returns the routes that caused this source selection.
    #[must_use]
    pub fn routes(&self) -> &'routes [ServeDecision] {
        self.routes.routes
    }

    /// Consumes the permit at the point where the caller invokes the gather.
    #[must_use]
    pub fn into_parts(self) -> (RouterResult<'routes>, Source) {
        (self.routes, self.sources)
    }
}

/// Makes a route address available to a policy without exposing the
/// internals of the phase values.
#[must_use]
pub const fn route_address(decision: ServeDecision) -> ExpertAddress {
    decision.address
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use proxima_tensor::cpu;
    use proxima_tensor::dtype::DType;
    use proxima_tensor::map::{self, IndexMap};
    use proxima_tensor::op::{Extent, NodeId, Op, ScalarOp, append};

    use super::{
        Qwen35MoeExecutionMode, execute_pre_gather, route_address, split_layer_program,
        split_layer_segment, split_mapped_layer_segment, split_mapped_router_and_gather_segments,
        split_router_and_gather_segments,
    };
    use crate::residency::{ExpertAddress, ServeDecision, ServePrecision};
    use core::cell::Cell;

    #[test]
    fn phases_force_router_boundary_then_source_gather() {
        let events = [const { Cell::new("unset") }; 3];
        let selected = [ServeDecision {
            address: ExpertAddress {
                layer: 2,
                expert: 7,
            },
            precision: ServePrecision::Low,
        }];
        let output = execute_pre_gather(
            || {
                events[0].set("router");
                Ok::<_, ()>(selected)
            },
            |router| {
                events[1].set("boundary");
                assert_eq!(router.routes(), &selected);
                assert_eq!(
                    route_address(selected[0]),
                    ExpertAddress {
                        layer: 2,
                        expert: 7
                    }
                );
                Ok::<_, ()>(["low-q2-expert-7"])
            },
            |gather| {
                events[2].set("gather");
                assert_eq!(gather.sources(), &["low-q2-expert-7"]);
                assert_eq!(gather.routes(), &selected);
                Ok::<_, ()>(gather.sources()[0])
            },
        )
        .expect("pre-gather phases execute in order");
        assert_eq!(output, "low-q2-expert-7");
        assert_eq!(
            events.map(|event| event.get()),
            ["router", "boundary", "gather"]
        );
    }

    #[test]
    fn default_mode_does_not_require_the_pre_gather_protocol() {
        assert_eq!(
            Qwen35MoeExecutionMode::SinglePass,
            Qwen35MoeExecutionMode::SinglePass
        );
        assert_ne!(
            Qwen35MoeExecutionMode::SinglePass,
            Qwen35MoeExecutionMode::PreGather
        );
    }

    #[test]
    fn layer_split_hands_router_activation_to_gather_on_cpu() {
        let identity = IndexMap::Affine(map::projection(1, &[0]));
        let mut program = Vec::new();
        let activation = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(2)],
                name: Some("activation".into()),
            },
        );
        let router = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: vec![
                    (activation, identity.clone()),
                    (activation, identity.clone()),
                ],
                name: Some("router".into()),
            },
        );
        let scale = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(2)],
                name: Some("scale".into()),
            },
        );
        let gather = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![(router, identity.clone()), (scale, identity.clone())],
                name: Some("gather".into()),
            },
        );
        let boundary = split_layer_program(&program, &[], router, gather)
            .expect("router/gather boundary is representable");
        assert_eq!(boundary.cut_inputs().len(), 1);
        assert_eq!(boundary.cut_input_name(router), Some("router"));
        assert_eq!(boundary.cut_input_name(gather), None);
        let (segment, segment_cuts) = split_layer_segment(&program, &[], Some(activation), gather)
            .expect("the sequential segment is representable");
        assert_eq!(segment.len(), 4);
        assert_eq!(segment_cuts, vec![(activation, String::from("activation"))]);
        let ((router_segment, router_cuts), (gather_segment, gather_cuts)) =
            split_router_and_gather_segments(&program, &[], None, router, gather)
                .expect("router and gather segments are representable");
        assert!(!router_segment.is_empty());
        assert!(!gather_segment.is_empty());

        let router_segment_root = NodeId((router_segment.len() - 1) as u32);
        let segmented_router = cpu::evaluate_named(
            &router_segment,
            &[],
            &[("activation", &[2.0_f32, 3.0])],
            &[router_segment_root],
        )
        .expect("router segment executes independently");
        let (segmented_router_values, _) = segmented_router
            .get(router_segment_root)
            .expect("router segment returns its boundary value");
        assert!(router_cuts.is_empty());
        assert_eq!(gather_cuts, vec![(router, String::from("router"))]);
        let gather_segment_root = NodeId((gather_segment.len() - 1) as u32);
        let segmented_gather = cpu::evaluate_named(
            &gather_segment,
            &[],
            &[
                ("router", segmented_router_values),
                ("scale", &[5.0_f32, 7.0]),
            ],
            &[gather_segment_root],
        )
        .expect("gather segment executes from the current router handoff");
        assert_eq!(segmented_gather.root(), &[20.0, 42.0]);

        let activation_data = [2.0_f32, 3.0];
        let routed = cpu::evaluate(
            boundary.router_program(),
            &[],
            &[&activation_data],
            &[boundary.router_output()],
        )
        .expect("router prepass evaluates");
        let (router_data, _) = routed
            .get(boundary.router_output())
            .expect("router activation is returned");
        let router_shape = [2_usize];
        let handoff = boundary
            .activation_handoff(router_data, &router_shape)
            .expect("router output crosses the boundary");
        assert_eq!(handoff.shape(), &router_shape);
        let cut_bindings = boundary
            .cut_bindings(&routed)
            .expect("producer result supplies every named cut");
        assert_eq!(cut_bindings.len(), 1);
        assert_eq!(cut_bindings[0].0, "router");
        let gathered = cpu::evaluate_named(
            boundary.gather_program(),
            &[],
            &[
                (handoff.name(), handoff.values()),
                ("scale", &[5.0_f32, 7.0]),
            ],
            &[boundary.gather_output()],
        )
        .expect("gather evaluates from the handed-off activation");
        assert_eq!(gathered.root(), &[20.0, 42.0]);
    }

    #[test]
    fn first_router_segment_is_the_graph_prefix() {
        let identity = IndexMap::Affine(map::projection(1, &[0]));
        let mut program = Vec::new();
        let activation = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(2)],
                name: Some("activation".into()),
            },
        );
        let router = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: vec![
                    (activation, identity.clone()),
                    (activation, identity.clone()),
                ],
                name: Some("router".into()),
            },
        );
        let gather = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![(router, identity.clone()), (activation, identity)],
                name: Some("gather".into()),
            },
        );

        let prefix = split_mapped_layer_segment(&program, &[], None, router)
            .expect("the graph prefix reaches the first router");
        let (first_router, _) =
            split_mapped_router_and_gather_segments(&program, &[], None, router, gather)
                .expect("the first routed layer has executable segments");

        assert_eq!(
            first_router, prefix,
            "executing a separate prefix would repeat the first router segment"
        );
    }
}
