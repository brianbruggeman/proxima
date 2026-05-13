//! Pipeline executor: a DAG of stages, each one a child process, run
//! atop the `ProcessUpstream` + `ProcessEventBridge` composition.
//! `PipelineExecutor::run` consumes a `PipelineSpec`, allocates per-stage
//! `InteractionId`s, and emits universal `RecordingEvent`s into the
//! configured sink. See `executor.rs` for execution semantics and
//! `spec.rs` for the declarative shape.

pub mod control_plane;
pub mod dag;
pub mod executor;
pub mod explain;
pub mod fs_control_plane;
pub mod http_routes;
pub mod replay;
pub mod spec;

pub use control_plane::{
    DynPipelineControlPlane, EventFilter, EventStream, InMemoryPipelineControlPlane, ListFilter,
    PipelineControlPlane, PipelineRecord, PipelineStatus, PipelineSubmission, PipelineSummary,
};
pub use dag::topological_order;
pub use executor::{PipelineExecutor, PipelineRunReport, StageReport};
pub use explain::{ExplainStep, explain_stage};
pub use fs_control_plane::FsPipelineControlPlane;
pub use http_routes::PipelineControlPlanePipe;
pub use replay::{ReplayReport, replay_pipeline};
pub use spec::{PipelineSpec, StageSpec};
