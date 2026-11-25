//! The Metal driver's debug and diagnostic switches as one value.
//!
//! Teaching pointer: this is the data form of the `PROXIMA_*` environment reads the Metal driver
//! used to cache in a `OnceLock` per switch. It follows [`crate::msl::EmitKnobs`]: one struct with
//! consuming `with_*` setters, plus serde and `conflaguration::Settings` so
//! [`crate::config::RuntimeConfig`] nests it under the unchanged `PROXIMA_*` spellings. A Metal
//! plan resolves it once at construction and its readers take it from the plan, so two plans in
//! one process may carry different switches. It is a sibling of `EmitKnobs`, not a field of it:
//! `EmitKnobs` keys kernel caches and a diagnostic switch must never change a cache key.

use core::convert::Infallible;

use conflaguration::Settings;
use proxima_tensor::NodeId;
use serde::{Deserialize, Serialize};

const BLIT_AFTER_KERNEL: &str = "blit-after-kernel";

/// Every diagnostic switch the Metal driver honours, resolved to a value. All off by default.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA")]
#[serde(default, deny_unknown_fields)]
pub struct MetalDiagnostics {
    /// Ask Metal for per-encoder execution status on every command buffer; on only for the value `1`.
    #[setting(default = false, resolve_with = "parse_one")]
    pub metal_encoder_error_status: bool,
    /// Let the explicit-boundary diagnostic also split a prefill-shaped plan; presence alone enables it.
    #[setting(default = false, resolve_with = "parse_presence")]
    pub boundaries_prefill: bool,
    /// Wait for the previously committed chunk before each commit; on only for the value `1`.
    #[setting(default = false, resolve_with = "parse_one")]
    pub precommit_wait: bool,
    /// Commit and wait on an empty command buffer before the first commit of a step; on only for the value `1`.
    #[setting(default = false, resolve_with = "parse_one")]
    pub precommit_ping: bool,
    /// Enqueue each chunk's command buffer when it is created; on only for the value `1`.
    #[setting(default = false, resolve_with = "parse_one")]
    pub command_buffer_enqueue_early: bool,
    /// Fill each chunk head with a sentinel to expose reads of unwritten memory; presence alone enables it.
    #[setting(default = false, resolve_with = "parse_presence")]
    pub head_sentinel_fill: bool,
    /// Scan each op's output for non-finite values; presence alone enables it.
    #[setting(default = false, resolve_with = "parse_presence")]
    pub metal_nan_check: bool,
    /// Re-run the repeated nodes and compare the copies byte for byte; presence alone enables it.
    #[setting(default = false, resolve_with = "parse_presence")]
    pub repeat_verify: bool,
    /// Substitution mode; `blit-after-kernel` overwrites each kernel output from the dump directory.
    #[setting(default, resolve_with = "parse_text")]
    pub substitute_mode: String,
    /// Directory holding the substitute manifest; `None` substitutes nothing.
    #[setting(default, resolve_with = "parse_directory")]
    pub substitute_dump_dir: Option<String>,
    /// Node whose output is compared against a CPU re-evaluation, such as `Some(NodeId(2994))`.
    #[setting(default, resolve_with = "parse_node")]
    pub metal_compare_bound_node: Option<NodeId>,
    /// Nodes duplicated for run-to-run determinism checks; entries that do not parse are dropped.
    #[setting(default, resolve_with = "parse_node_list")]
    pub repeat_nodes: Vec<NodeId>,
    /// Copies made of each repeated node.
    #[setting(default = 1, resolve_with = "parse_repeat_count")]
    pub repeat_count: u32,
    /// Command buffers one decode step splits into; only a positive integer is honoured.
    #[setting(default, resolve_with = "parse_chunk_count")]
    pub command_buffer_chunks: Option<usize>,
    /// Explicit op positions that start a new command buffer; entries that do not parse are dropped.
    #[setting(default, resolve_with = "parse_boundaries")]
    pub command_buffer_boundaries: Option<Vec<usize>>,
    /// Audit every CPU write against the chunks already committed; presence alone enables it.
    #[setting(default = false, resolve_with = "parse_presence")]
    pub chunk_audit: bool,
    /// Ordered nodes whose buffer identity is printed and sentinel-filled before dispatch; index 0 is the head root and later indices are its duplicates. Entries that do not parse are dropped.
    #[setting(default, resolve_with = "parse_node_list")]
    pub head_debug_nodes: Vec<NodeId>,
    /// Keep the twin elementwise fusion pass off; presence alone disables it.
    #[setting(default = false, resolve_with = "parse_presence")]
    pub disable_twin_elementwise_fusion: bool,
}

impl Default for MetalDiagnostics {
    fn default() -> Self {
        Self {
            metal_encoder_error_status: false,
            boundaries_prefill: false,
            precommit_wait: false,
            precommit_ping: false,
            command_buffer_enqueue_early: false,
            head_sentinel_fill: false,
            metal_nan_check: false,
            repeat_verify: false,
            substitute_mode: String::new(),
            substitute_dump_dir: None,
            metal_compare_bound_node: None,
            repeat_nodes: Vec::new(),
            repeat_count: 1,
            command_buffer_chunks: None,
            command_buffer_boundaries: None,
            chunk_audit: false,
            head_debug_nodes: Vec::new(),
            disable_twin_elementwise_fusion: false,
        }
    }
}

impl MetalDiagnostics {
    /// Reads every switch from the process environment once; a plan built without a
    /// [`crate::config::RuntimeConfig`] resolves its diagnostics through this.
    #[must_use]
    pub fn from_env() -> Self {
        <Self as Settings>::from_env().unwrap_or_default()
    }

    /// Whether the substitution mode asks for a blit after each kernel.
    #[must_use]
    pub fn substitute_blit_after_kernel(&self) -> bool {
        self.substitute_mode == BLIT_AFTER_KERNEL
    }

    /// Set the per-encoder execution status request.
    #[must_use]
    pub fn with_metal_encoder_error_status(mut self, value: bool) -> Self {
        self.metal_encoder_error_status = value;
        self
    }

    /// Set whether explicit boundaries may split a prefill-shaped plan.
    #[must_use]
    pub fn with_boundaries_prefill(mut self, value: bool) -> Self {
        self.boundaries_prefill = value;
        self
    }

    /// Set the wait on the previously committed chunk.
    #[must_use]
    pub fn with_precommit_wait(mut self, value: bool) -> Self {
        self.precommit_wait = value;
        self
    }

    /// Set the empty-buffer ping before the first commit.
    #[must_use]
    pub fn with_precommit_ping(mut self, value: bool) -> Self {
        self.precommit_ping = value;
        self
    }

    /// Set early enqueue of each chunk's command buffer.
    #[must_use]
    pub fn with_command_buffer_enqueue_early(mut self, value: bool) -> Self {
        self.command_buffer_enqueue_early = value;
        self
    }

    /// Set the chunk-head sentinel fill.
    #[must_use]
    pub fn with_head_sentinel_fill(mut self, value: bool) -> Self {
        self.head_sentinel_fill = value;
        self
    }

    /// Set the non-finite output scan.
    #[must_use]
    pub fn with_metal_nan_check(mut self, value: bool) -> Self {
        self.metal_nan_check = value;
        self
    }

    /// Set the repeated-node byte comparison.
    #[must_use]
    pub fn with_repeat_verify(mut self, value: bool) -> Self {
        self.repeat_verify = value;
        self
    }

    /// Set the substitution mode.
    #[must_use]
    pub fn with_substitute_mode(mut self, value: impl Into<String>) -> Self {
        self.substitute_mode = value.into();
        self
    }

    /// Set the substitute dump directory.
    #[must_use]
    pub fn with_substitute_dump_dir(mut self, value: Option<String>) -> Self {
        self.substitute_dump_dir = value;
        self
    }

    /// Set the node compared against a CPU re-evaluation.
    #[must_use]
    pub fn with_metal_compare_bound_node(mut self, value: Option<NodeId>) -> Self {
        self.metal_compare_bound_node = value;
        self
    }

    /// Set the nodes duplicated for determinism checks.
    #[must_use]
    pub fn with_repeat_nodes(mut self, value: Vec<NodeId>) -> Self {
        self.repeat_nodes = value;
        self
    }

    /// Set how many copies each repeated node gets.
    #[must_use]
    pub fn with_repeat_count(mut self, value: u32) -> Self {
        self.repeat_count = value;
        self
    }

    /// Set the number of command buffers a decode step splits into.
    #[must_use]
    pub fn with_command_buffer_chunks(mut self, value: Option<usize>) -> Self {
        self.command_buffer_chunks = value;
        self
    }

    /// Set the explicit chunk-boundary op positions.
    #[must_use]
    pub fn with_command_buffer_boundaries(mut self, value: Option<Vec<usize>>) -> Self {
        self.command_buffer_boundaries = value;
        self
    }

    /// Set the chunked-submission write audit.
    #[must_use]
    pub fn with_chunk_audit(mut self, value: bool) -> Self {
        self.chunk_audit = value;
        self
    }

    /// Set the nodes whose head buffers are identity-printed and sentinel-filled.
    #[must_use]
    pub fn with_head_debug_nodes(mut self, value: Vec<NodeId>) -> Self {
        self.head_debug_nodes = value;
        self
    }

    /// Set whether the twin elementwise fusion pass is disabled.
    #[must_use]
    pub fn with_disable_twin_elementwise_fusion(mut self, value: bool) -> Self {
        self.disable_twin_elementwise_fusion = value;
        self
    }
}

fn parse_one(raw: &str) -> Result<bool, Infallible> {
    Ok(raw == "1")
}

fn parse_presence(_raw: &str) -> Result<bool, Infallible> {
    Ok(true)
}

fn parse_text(raw: &str) -> Result<String, Infallible> {
    Ok(raw.to_string())
}

fn parse_directory(raw: &str) -> Result<Option<String>, Infallible> {
    Ok(Some(raw.to_string()))
}

fn parse_node(raw: &str) -> Result<Option<NodeId>, Infallible> {
    Ok(raw.parse::<u32>().ok().map(NodeId))
}

fn parse_node_list(raw: &str) -> Result<Vec<NodeId>, Infallible> {
    Ok(raw
        .split(',')
        .filter_map(|entry| entry.trim().parse::<u32>().ok())
        .map(NodeId)
        .collect())
}

fn parse_repeat_count(raw: &str) -> Result<u32, Infallible> {
    Ok(raw.parse().unwrap_or(1))
}

fn parse_chunk_count(raw: &str) -> Result<Option<usize>, Infallible> {
    Ok(raw
        .trim()
        .parse::<usize>()
        .ok()
        .filter(|parsed| *parsed >= 1))
}

fn parse_boundaries(raw: &str) -> Result<Option<Vec<usize>>, Infallible> {
    Ok(Some(
        raw.split(',')
            .filter_map(|entry| entry.trim().parse::<usize>().ok())
            .collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RuntimeConfig;

    const SWITCHES: [&str; 18] = [
        "PROXIMA_METAL_ENCODER_ERROR_STATUS",
        "PROXIMA_BOUNDARIES_PREFILL",
        "PROXIMA_PRECOMMIT_WAIT",
        "PROXIMA_PRECOMMIT_PING",
        "PROXIMA_COMMAND_BUFFER_ENQUEUE_EARLY",
        "PROXIMA_HEAD_SENTINEL_FILL",
        "PROXIMA_METAL_NAN_CHECK",
        "PROXIMA_REPEAT_VERIFY",
        "PROXIMA_SUBSTITUTE_MODE",
        "PROXIMA_SUBSTITUTE_DUMP_DIR",
        "PROXIMA_METAL_COMPARE_BOUND_NODE",
        "PROXIMA_REPEAT_NODES",
        "PROXIMA_REPEAT_COUNT",
        "PROXIMA_COMMAND_BUFFER_CHUNKS",
        "PROXIMA_COMMAND_BUFFER_BOUNDARIES",
        "PROXIMA_CHUNK_AUDIT",
        "PROXIMA_HEAD_DEBUG_NODES",
        "PROXIMA_DISABLE_TWIN_ELEMENTWISE_FUSION",
    ];

    fn with_switches<R>(set: &[(&'static str, &'static str)], body: impl FnOnce() -> R) -> R {
        let mut variables: Vec<(&str, Option<&str>)> =
            SWITCHES.iter().map(|name| (*name, None)).collect();
        variables.extend([("OMEGA_ENGINE", None), ("OMEGA_GPU_DRIVER", None)]);
        for (name, value) in set {
            if let Some(slot) = variables.iter_mut().find(|slot| slot.0 == *name) {
                slot.1 = Some(*value);
            }
        }
        temp_env::with_vars(variables, body)
    }

    #[test]
    fn metal_diagnostics_default_is_all_off() {
        let defaults = MetalDiagnostics::default();
        let from_env = with_switches(&[], MetalDiagnostics::from_env);

        assert_eq!(from_env, defaults);
        assert!(!defaults.metal_encoder_error_status && !defaults.chunk_audit);
        assert!(!defaults.substitute_blit_after_kernel());
        assert_eq!(defaults.repeat_count, 1);
        assert!(defaults.repeat_nodes.is_empty());
        assert_eq!(defaults.metal_compare_bound_node, None);
        assert_eq!(defaults.command_buffer_chunks, None);
        assert_eq!(RuntimeConfig::default().metal_diagnostics, defaults);
    }

    #[test]
    fn metal_diagnostics_env_and_builder_agree() {
        let built = MetalDiagnostics::default()
            .with_metal_encoder_error_status(true)
            .with_chunk_audit(true)
            .with_substitute_mode("blit-after-kernel")
            .with_repeat_nodes(vec![NodeId(2994), NodeId(3009), NodeId(7)])
            .with_repeat_count(3)
            .with_metal_compare_bound_node(Some(NodeId(42)))
            .with_command_buffer_chunks(Some(4))
            .with_head_debug_nodes(vec![NodeId(11), NodeId(12), NodeId(13)])
            .with_disable_twin_elementwise_fusion(true);
        let environment = [
            ("PROXIMA_HEAD_DEBUG_NODES", "11, 12,not-a-node,,13"),
            ("PROXIMA_DISABLE_TWIN_ELEMENTWISE_FUSION", "1"),
            ("PROXIMA_METAL_ENCODER_ERROR_STATUS", "1"),
            ("PROXIMA_CHUNK_AUDIT", ""),
            ("PROXIMA_SUBSTITUTE_MODE", "blit-after-kernel"),
            ("PROXIMA_REPEAT_NODES", "2994, 3009,not-a-node,,7"),
            ("PROXIMA_REPEAT_COUNT", "3"),
            ("PROXIMA_METAL_COMPARE_BOUND_NODE", "42"),
            ("PROXIMA_COMMAND_BUFFER_CHUNKS", "4"),
        ];

        let loaded = with_switches(&environment, || RuntimeConfig::from_env().ok());

        assert_eq!(
            loaded.map(|config| config.metal_diagnostics),
            Some(built.clone())
        );
        assert_eq!(
            with_switches(&environment, MetalDiagnostics::from_env),
            built
        );
        assert_ne!(built, MetalDiagnostics::default());
        assert_eq!(
            built.head_debug_nodes,
            vec![NodeId(11), NodeId(12), NodeId(13)]
        );
    }

    #[test]
    fn value_switch_is_on_only_for_one() {
        let on = with_switches(
            &[("PROXIMA_PRECOMMIT_WAIT", "1")],
            MetalDiagnostics::from_env,
        );
        let zero = with_switches(
            &[("PROXIMA_PRECOMMIT_WAIT", "0")],
            MetalDiagnostics::from_env,
        );

        assert!(on.precommit_wait);
        assert!(!zero.precommit_wait);
    }

    #[test]
    fn presence_switch_is_on_for_any_value_including_empty() {
        let set = with_switches(
            &[("PROXIMA_BOUNDARIES_PREFILL", "1")],
            MetalDiagnostics::from_env,
        );
        let empty = with_switches(
            &[("PROXIMA_BOUNDARIES_PREFILL", "")],
            MetalDiagnostics::from_env,
        );
        let unset = with_switches(&[], MetalDiagnostics::from_env);

        assert!(set.boundaries_prefill);
        assert!(empty.boundaries_prefill);
        assert!(!unset.boundaries_prefill);
    }

    #[test]
    fn node_list_keeps_the_parsable_entries_in_order() {
        let parsed = parse_node_list("2994, 3009,not-a-node,,7").ok();

        assert_eq!(parsed, Some(vec![NodeId(2994), NodeId(3009), NodeId(7)]));
    }

    #[test]
    fn a_zero_chunk_count_falls_through_to_the_plan_default() {
        let zero = with_switches(
            &[("PROXIMA_COMMAND_BUFFER_CHUNKS", "0")],
            MetalDiagnostics::from_env,
        );

        assert_eq!(zero.command_buffer_chunks, None);
    }
}
