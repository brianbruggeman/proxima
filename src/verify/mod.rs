//! `proxima verify` — static spec-graph and replay-with-policy
//! verification. Two CLI subcommands shell over the library API
//! here; both walkers return a [`Report`] the caller emits as text
//! or JSON.
//!
//! - [`verify_static`](static_walker::verify_static) — graph
//!   invariants over a parsed spec ([`serde_json::Value`]). v1 ships
//!   `no_cycles` and `all_upstreams_have_timeouts` plus the
//!   custom-predicate runner.
//! - Replay verification is in progress; see `replay_walker.rs`
//!   (W4.b).
//!
//! Discovery rules for zero-arg invocation live in [`discover`].

pub mod byte_drift;
pub mod discover;
pub mod policy;
pub mod repair;
pub mod replay_walker;
pub mod report;
pub mod static_walker;

pub use byte_drift::{skip_byte_drift_without_spec, verify_byte_drift};
pub use policy::Policy;
pub use repair::{
    RecordingRepairOutcome, RepairItem, RepairOutcome, Weight, project_max_coherent,
    repair_from_recording, repair_from_recording_file, repair_kind_claims, repair_spec_cycles,
    repair_static,
};
pub use replay_walker::{verify_replay, verify_replay_with_spec};
pub use report::{Level, Report, ReportEntry};
pub use static_walker::verify_static;
