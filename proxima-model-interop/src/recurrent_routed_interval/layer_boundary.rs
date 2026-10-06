//! Borrowed activation handoff for a single qwen35moe layer.
//!
//! A routed layer must publish its hidden activation before its router and
//! expert gather can be evaluated independently.  These phase values carry
//! that activation without owning or copying it; the evaluator remains the
//! owner of the backing buffer.

use proxima_tensor::NodeId;

/// One layer's activation buffer, borrowed for the duration of a split pass.
#[derive(Debug, Clone, Copy)]
pub struct LayerActivation<'values> {
    layer: u32,
    values: &'values [f32],
    width: usize,
}

impl<'values> LayerActivation<'values> {
    /// Creates a borrowed activation handoff for `layer`.
    #[must_use]
    pub const fn new(layer: u32, values: &'values [f32], width: usize) -> Self {
        Self {
            layer,
            values,
            width,
        }
    }

    #[must_use]
    pub const fn layer(self) -> u32 {
        self.layer
    }

    #[must_use]
    pub const fn values(self) -> &'values [f32] {
        self.values
    }

    #[must_use]
    pub const fn width(self) -> usize {
        self.width
    }
}

/// A layer activation with its router root selected for a prepass.
#[derive(Debug, Clone, Copy)]
pub struct RouterPhase<'values> {
    activation: LayerActivation<'values>,
    router_root: NodeId,
}

impl<'values> RouterPhase<'values> {
    /// Starts the router phase for one layer.
    pub const fn new(activation: LayerActivation<'values>, router_root: NodeId) -> Self {
        Self {
            activation,
            router_root,
        }
    }

    #[must_use]
    pub const fn activation(self) -> LayerActivation<'values> {
        self.activation
    }

    #[must_use]
    pub const fn router_root(self) -> NodeId {
        self.router_root
    }

    /// Attaches the route result and opens the expert-gather phase.
    #[must_use]
    pub const fn route_complete(self, selected_experts: &'values [u32]) -> GatherPhase<'values> {
        GatherPhase {
            activation: self.activation,
            router_root: self.router_root,
            selected_experts,
        }
    }
}

/// The only phase from which the layer's expert gather may be launched.
#[derive(Debug, Clone, Copy)]
pub struct GatherPhase<'values> {
    activation: LayerActivation<'values>,
    router_root: NodeId,
    selected_experts: &'values [u32],
}

impl<'values> GatherPhase<'values> {
    #[must_use]
    pub const fn activation(self) -> LayerActivation<'values> {
        self.activation
    }

    #[must_use]
    pub const fn router_root(self) -> NodeId {
        self.router_root
    }

    #[must_use]
    pub const fn selected_experts(self) -> &'values [u32] {
        self.selected_experts
    }
}

#[cfg(test)]
mod tests {
    use super::{GatherPhase, LayerActivation, RouterPhase};
    use proxima_tensor::NodeId;

    #[test]
    fn activation_is_borrowed_across_router_and_gather_phases() {
        let values = [1.0, 2.0, 3.0, 4.0];
        let selected = [3, 7];
        let router = RouterPhase::new(LayerActivation::new(4, &values, 4), NodeId(19));
        let gather = router.route_complete(&selected);
        assert_eq!(gather.activation().layer(), 4);
        assert_eq!(gather.activation().values(), &values);
        assert_eq!(gather.router_root(), NodeId(19));
        assert_eq!(gather.selected_experts(), &selected);
    }

    #[test]
    fn gather_phase_is_not_constructible_without_route_completion() {
        fn accepts_only_gather(_: GatherPhase<'_>) {}
        let values = [0.0];
        let selected = [1];
        let phase = RouterPhase::new(LayerActivation::new(0, &values, 1), NodeId(1));
        accepts_only_gather(phase.route_complete(&selected));
    }
}
