// CEL evaluation against `orchestrator::CelBindings` — only orchestrator
// (tokio-only) calls this.
#[cfg(feature = "tokio")]
pub mod cel;
pub mod discover;
// external-process scenario runner (`tokio::process::Command` supervision +
// `tokio::io`/`tokio::fs`/`tokio::time`) — a genuine tokio::process
// capability with no prime equivalent today.
#[cfg(feature = "tokio")]
pub mod orchestrator;
pub mod spec;

pub use discover::{discover_by_name, discover_scenario, name_search_roots};
#[cfg(feature = "tokio")]
pub use orchestrator::{ScenarioReport, run_scenario, run_scenario_with_sink};
pub use spec::{
    CompareOp, DurationSpec, Expectation, OrchestrationMode, ProfileStep, Scenario,
    ScenarioPipeSpec, WorkloadMode, WorkloadSpec,
};
