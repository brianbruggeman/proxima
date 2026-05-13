//! Coherence repair. Two failure modes feed one operator:
//!
//! - **Static failure** — the spec graph rejects (cycle, route
//!   conflict, missing field). Repair drops the minimum-weight set
//!   of spec items needed to pass verification.
//! - **Replay drift** — a recording shows a pipe behaving outside
//!   its declared kind. Repair drops the disagreeing kind claims,
//!   letting the declared model catch up to observed reality.
//!
//! Both reduce to one function: [`project_max_coherent`]. Given a
//! weighted set of removable items and a predicate that decides
//! whether a subset "holds together," it returns the largest-weight
//! subset that passes the predicate, plus the blame list of items
//! dropped to get there.
//!
//! The algorithm is greedy: sort by weight descending, include each
//! item that keeps the running set acceptable. Optimal min-blame is
//! NP-hard in general (it generalizes minimum feedback arc set and
//! set cover); the greedy variant is O(n²) in item count and
//! adequate for proxima-sized configs. Switch to ILP only if the
//! corpus shows the greedy result diverging from optimal often
//! enough to matter.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;

use futures::StreamExt;
use serde_json::Value;

use super::policy::Policy;
use super::replay_walker::verify_replay_events;
use super::report::Report;
use super::static_walker::verify_static;
use std::sync::Arc;

use crate::error::ProximaError;
use crate::recording::bin::source::BinSource;
use crate::recording::event::RecordingEvent;
use crate::recording::source::RecordingSource;
use crate::runtime::Runtime;

/// One removable thing. Repair drops these to recover acceptance.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum RepairItem {
    /// A chain edge in a pipe's `chain = [...]` list. Removing it
    /// breaks the dependency from `from` to `to`.
    ChainEdge { from: String, to: String },

    /// A whole named entry — drop the pipe / upstream / middleware.
    SpecEntry { section: String, name: String },

    /// A claim that a pipe behaves as a given kind. Drop to relax
    /// the claim — the declared model catches up to observed
    /// behavior.
    KindClaim { pipe: String, declared: String },

    /// A policy claim that a pipe's events must derive from the
    /// recording (vs. being inferred at replay time). Drop to revert
    /// the policy to whatever the recording actually shows.
    MustDeriveClaim { pipe: String },
}

impl fmt::Display for RepairItem {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChainEdge { from, to } => write!(formatter, "chain_edge {from} → {to}"),
            Self::SpecEntry { section, name } => write!(formatter, "{section}.{name}"),
            Self::KindClaim { pipe, declared } => {
                write!(formatter, "kind_claim {pipe} = {declared}")
            }
            Self::MustDeriveClaim { pipe } => {
                write!(formatter, "must_derive {pipe}")
            }
        }
    }
}

/// Per-item weight. Higher = more important to keep. The repair
/// operator orders by weight descending and includes each item that
/// doesn't break the coherence predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Weight(pub u32);

impl Default for Weight {
    fn default() -> Self {
        Self(1)
    }
}

/// Project a weighted item set to its max-weight coherent subset.
/// Returns `(kept, blame)`: `kept` is the surviving subset; `blame`
/// is what was dropped to get there.
///
/// `is_coherent` must hold on the empty set (otherwise nothing is
/// recoverable) and adding a coherence-preserving item must preserve
/// coherence (monotonicity within the kept set). Non-monotone
/// predicates still terminate but may yield non-maximal output.
pub fn project_max_coherent<Item, IsCoherent>(
    items: impl IntoIterator<Item = (Item, Weight)>,
    is_coherent: IsCoherent,
) -> (Vec<Item>, Vec<Item>)
where
    Item: Clone,
    IsCoherent: Fn(&[Item]) -> bool,
{
    let mut weighted: Vec<(Item, Weight)> = items.into_iter().collect();
    weighted.sort_by_key(|(_, weight)| std::cmp::Reverse(*weight));

    let mut kept: Vec<Item> = Vec::new();
    let mut blame: Vec<Item> = Vec::new();
    for (item, _) in weighted {
        kept.push(item.clone());
        if !is_coherent(&kept) {
            kept.pop();
            blame.push(item);
        }
    }
    (kept, blame)
}

/// Repair a spec rejected by the static walker. Drops chain edges to
/// break cycles. Returns the cleaned spec and the blame list.
pub fn repair_spec_cycles(spec: &Value) -> (Value, Vec<RepairItem>) {
    let edges = collect_chain_edges(spec);
    let weighted: Vec<(RepairItem, Weight)> = edges
        .into_iter()
        .map(|edge| (edge, Weight::default()))
        .collect();
    let (kept, blame) = project_max_coherent(weighted, |kept_edges| !contains_cycle(kept_edges));
    let repaired = apply_edge_filter(spec, &kept);
    (repaired, blame)
}

/// Repair a kind-claim set rejected by replay observation. Given the
/// declared `pipe → kind` map and the observed kinds per pipe, drops
/// the claims that disagree with observation. Pipes that never fired
/// are kept (vacuously fine — no evidence against them).
pub fn repair_kind_claims(
    declared: &BTreeMap<String, String>,
    observed: &BTreeMap<String, BTreeSet<String>>,
) -> (BTreeMap<String, String>, Vec<RepairItem>) {
    let claims: Vec<(RepairItem, Weight)> = declared
        .iter()
        .map(|(pipe, kind)| {
            (
                RepairItem::KindClaim {
                    pipe: pipe.clone(),
                    declared: kind.clone(),
                },
                Weight::default(),
            )
        })
        .collect();

    let (kept_items, blame) = project_max_coherent(claims, |kept| {
        for item in kept {
            let RepairItem::KindClaim { pipe, declared } = item else {
                continue;
            };
            match observed.get(pipe) {
                Some(observed_kinds) if observed_kinds.contains(declared) => {
                    continue;
                }
                Some(_) => return false,
                None => continue,
            }
        }
        true
    });

    let kept_map: BTreeMap<String, String> = kept_items
        .into_iter()
        .filter_map(|item| match item {
            RepairItem::KindClaim { pipe, declared } => Some((pipe, declared)),
            _ => None,
        })
        .collect();
    (kept_map, blame)
}

/// Output of a static repair run — both reports plus the blame list
/// and the post-repair spec.
#[derive(Debug, Clone)]
pub struct RepairOutcome {
    pub before: Report,
    pub after: Report,
    pub blame: Vec<RepairItem>,
    pub repaired_spec: Value,
}

/// Output of a recording-driven repair — both reports, the blame
/// list, and the reverted policy. The recording is the ground truth;
/// policy entries that the recording contradicts get dropped.
#[derive(Debug, Clone)]
pub struct RecordingRepairOutcome {
    pub before: Report,
    pub after: Report,
    pub blame: Vec<RepairItem>,
    pub repaired_policy: Policy,
}

impl RepairOutcome {
    #[must_use]
    pub fn improved(&self) -> bool {
        self.after.fail_count() < self.before.fail_count()
    }

    #[must_use]
    pub fn fully_repaired(&self) -> bool {
        self.after.fail_count() == 0
    }
}

impl RecordingRepairOutcome {
    #[must_use]
    pub fn improved(&self) -> bool {
        self.after.fail_count() < self.before.fail_count()
    }

    #[must_use]
    pub fn fully_repaired(&self) -> bool {
        self.after.fail_count() == 0
    }
}

/// Run [`verify_static`], project the spec to recover acceptance,
/// re-run verify. Wire as `proxima verify --repair`.
#[must_use]
pub fn repair_static(spec: &Value, policy: &Policy) -> RepairOutcome {
    let before = verify_static(spec, policy);
    let (repaired_spec, blame) = repair_spec_cycles(spec);
    let after = verify_static(&repaired_spec, policy);
    RepairOutcome {
        before,
        after,
        blame,
        repaired_spec,
    }
}

/// Revert a policy to match what the recording actually shows. For
/// each pipe currently in `policy.replay.must_derive_from_record`,
/// keep it only if doing so doesn't introduce a verify failure under
/// the supplied events and spec. Dropped pipes appear in `blame`.
///
/// "Reverting with a recording" — the recording is treated as ground
/// truth, and the policy gets relaxed to whatever subset the
/// recording can support. Use as `proxima replay --repair`.
#[must_use]
pub fn repair_from_recording(
    events: &[RecordingEvent],
    spec: Option<&Value>,
    policy: &Policy,
) -> RecordingRepairOutcome {
    let before = verify_replay_events(events, policy, spec);

    let candidates: Vec<(RepairItem, Weight)> = policy
        .replay
        .must_derive_from_record
        .iter()
        .map(|pipe| {
            (
                RepairItem::MustDeriveClaim { pipe: pipe.clone() },
                Weight::default(),
            )
        })
        .collect();

    let (kept_items, blame) = project_max_coherent(candidates, |kept| {
        let kept_pipes: Vec<String> = kept
            .iter()
            .filter_map(|item| match item {
                RepairItem::MustDeriveClaim { pipe } => Some(pipe.clone()),
                _ => None,
            })
            .collect();
        let projected = with_must_derive(policy, kept_pipes);
        verify_replay_events(events, &projected, spec).fail_count() == 0
    });

    let kept_pipes: Vec<String> = kept_items
        .into_iter()
        .filter_map(|item| match item {
            RepairItem::MustDeriveClaim { pipe } => Some(pipe),
            _ => None,
        })
        .collect();
    let repaired_policy = with_must_derive(policy, kept_pipes);
    let after = verify_replay_events(events, &repaired_policy, spec);

    RecordingRepairOutcome {
        before,
        after,
        blame,
        repaired_policy,
    }
}

/// Stream a `.bin` recording from disk and run [`repair_from_recording`]
/// on the collected events. Mirrors [`super::verify_replay_with_spec`].
pub async fn repair_from_recording_file(
    recording_path: &Path,
    spec: Option<&Value>,
    policy: &Policy,
    runtime: &Arc<dyn Runtime>,
) -> Result<RecordingRepairOutcome, ProximaError> {
    let source = BinSource::new(recording_path, Arc::clone(runtime));
    let mut stream = source.events();
    let mut events: Vec<RecordingEvent> = Vec::new();
    while let Some(item) = stream.next().await {
        events.push(item?);
    }
    Ok(repair_from_recording(&events, spec, policy))
}

fn with_must_derive(policy: &Policy, pipes: Vec<String>) -> Policy {
    let mut projected = policy.clone();
    projected.replay.must_derive_from_record = pipes;
    projected
}

fn collect_chain_edges(spec: &Value) -> Vec<RepairItem> {
    let mut edges: Vec<RepairItem> = Vec::new();
    for section in ["pipes", "middlewares"] {
        let Some(Value::Object(map)) = spec.get(section) else {
            continue;
        };
        for (name, entry) in map {
            let Some(Value::Array(chain)) = entry.get("chain") else {
                continue;
            };
            for item in chain {
                let Some(target) = item.as_str() else {
                    continue;
                };
                edges.push(RepairItem::ChainEdge {
                    from: name.clone(),
                    to: target.to_string(),
                });
            }
        }
    }
    edges
}

fn contains_cycle(edges: &[RepairItem]) -> bool {
    let mut adjacency: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for edge in edges {
        let RepairItem::ChainEdge { from, to } = edge else {
            continue;
        };
        adjacency.entry(from.clone()).or_default().push(to.clone());
    }
    let nodes: BTreeSet<String> = adjacency
        .iter()
        .flat_map(|(from, targets)| std::iter::once(from.clone()).chain(targets.iter().cloned()))
        .collect();

    let mut visited: BTreeSet<String> = BTreeSet::new();
    let mut on_stack: BTreeSet<String> = BTreeSet::new();
    for start in &nodes {
        if visited.contains(start) {
            continue;
        }
        if dfs_for_cycle(start, &adjacency, &mut visited, &mut on_stack) {
            return true;
        }
    }
    false
}

fn dfs_for_cycle(
    node: &String,
    adjacency: &BTreeMap<String, Vec<String>>,
    visited: &mut BTreeSet<String>,
    on_stack: &mut BTreeSet<String>,
) -> bool {
    visited.insert(node.clone());
    on_stack.insert(node.clone());
    if let Some(targets) = adjacency.get(node) {
        for target in targets {
            if on_stack.contains(target) {
                return true;
            }
            if !visited.contains(target) && dfs_for_cycle(target, adjacency, visited, on_stack) {
                return true;
            }
        }
    }
    on_stack.remove(node);
    false
}

fn apply_edge_filter(spec: &Value, kept: &[RepairItem]) -> Value {
    let mut kept_index: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for edge in kept {
        let RepairItem::ChainEdge { from, to } = edge else {
            continue;
        };
        kept_index
            .entry(from.clone())
            .or_default()
            .insert(to.clone());
    }
    let mut out = spec.clone();
    let Value::Object(top) = &mut out else {
        return out;
    };
    for section in ["pipes", "middlewares"] {
        let Some(Value::Object(map)) = top.get_mut(section) else {
            continue;
        };
        for (name, entry) in map.iter_mut() {
            let Value::Object(entry_map) = entry else {
                continue;
            };
            let Some(Value::Array(chain)) = entry_map.get_mut("chain") else {
                continue;
            };
            let allowed = kept_index.get(name).cloned().unwrap_or_default();
            chain.retain(|item| {
                item.as_str()
                    .map(|target| allowed.contains(target))
                    .unwrap_or(true)
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::field_reassign_with_default,
        clippy::type_complexity,
        clippy::useless_vec,
        clippy::needless_range_loop,
        clippy::default_constructed_unit_structs
    )]
    use super::*;
    use crate::recording::event::{
        EventSource, HttpEvent, InteractionId, ProtocolEvent, RecordMeta, RecordingEvent,
        RequestHeader,
    };
    use serde_json::json;
    use time::OffsetDateTime;

    fn make_event(
        id: InteractionId,
        parent: Option<InteractionId>,
        event: ProtocolEvent,
    ) -> RecordingEvent {
        RecordingEvent {
            id,
            ts_ms: 0,
            parent,
            event,
        }
    }

    fn http_started(pipe: &str) -> ProtocolEvent {
        ProtocolEvent::Http(HttpEvent::Started {
            ts: OffsetDateTime::UNIX_EPOCH,
            pipe: pipe.into(),
            request: RequestHeader::default(),
            meta: None,
        })
    }

    fn http_ended_with(source: EventSource) -> ProtocolEvent {
        let mut meta = RecordMeta::default();
        meta.source = Some(source);
        ProtocolEvent::Http(HttpEvent::Ended {
            latency_ms: 0,
            meta,
        })
    }

    fn fresh_id(byte: u8) -> InteractionId {
        InteractionId::from_bytes([byte; 16])
    }

    #[test]
    fn project_keeps_everything_when_input_already_coherent() {
        let input: Vec<(u32, Weight)> = vec![(1, Weight(1)), (2, Weight(1))];
        let (kept, blame) = project_max_coherent(input, |_| true);
        assert_eq!(kept, vec![1, 2]);
        assert!(blame.is_empty());
    }

    #[test]
    fn project_drops_low_weight_items_when_threshold_exceeded() {
        let input = vec![(10u32, Weight(3)), (20u32, Weight(2)), (30u32, Weight(1))];
        let (kept, blame) = project_max_coherent(input, |values| values.iter().sum::<u32>() <= 30);
        assert_eq!(kept, vec![10, 20]);
        assert_eq!(blame, vec![30]);
    }

    #[test]
    fn project_orders_inclusions_by_weight_descending() {
        let input = vec![(1u32, Weight(1)), (2u32, Weight(99))];
        let (kept, _blame) = project_max_coherent(input, |_| true);
        assert_eq!(kept, vec![2, 1]);
    }

    #[test]
    fn repair_breaks_two_node_cycle_by_dropping_one_edge() {
        let spec = json!({
            "pipes": {
                "alpha": { "chain": ["beta"] },
                "beta": { "chain": ["alpha"] },
            }
        });
        let (repaired, blame) = repair_spec_cycles(&spec);
        assert_eq!(blame.len(), 1, "exactly one edge dropped");
        let edges_after = collect_chain_edges(&repaired);
        assert!(!contains_cycle(&edges_after), "repaired spec is acyclic");
    }

    #[test]
    fn repair_acyclic_spec_keeps_every_edge() {
        let spec = json!({
            "pipes": {
                "alpha": { "chain": ["beta"] },
                "beta": { "chain": ["gamma"] },
            }
        });
        let (repaired, blame) = repair_spec_cycles(&spec);
        assert!(blame.is_empty());
        assert_eq!(spec, repaired);
    }

    #[test]
    fn repair_three_node_cycle_drops_one_edge() {
        let spec = json!({
            "pipes": {
                "alpha": { "chain": ["beta"] },
                "beta": { "chain": ["gamma"] },
                "gamma": { "chain": ["alpha"] },
            }
        });
        let (repaired, blame) = repair_spec_cycles(&spec);
        assert_eq!(blame.len(), 1);
        let edges_after = collect_chain_edges(&repaired);
        assert!(!contains_cycle(&edges_after));
    }

    #[test]
    fn repair_kind_claims_drops_observed_mismatch() {
        let declared: BTreeMap<String, String> = [
            ("pipe_one".to_string(), "Cache".to_string()),
            ("pipe_two".to_string(), "Passthrough".to_string()),
        ]
        .into_iter()
        .collect();
        let observed: BTreeMap<String, BTreeSet<String>> = [
            ("pipe_one".to_string(), {
                let mut set = BTreeSet::new();
                set.insert("Cache".to_string());
                set
            }),
            ("pipe_two".to_string(), {
                let mut set = BTreeSet::new();
                set.insert("Transform".to_string());
                set
            }),
        ]
        .into_iter()
        .collect();
        let (kept, blame) = repair_kind_claims(&declared, &observed);
        assert!(kept.contains_key("pipe_one"));
        assert!(!kept.contains_key("pipe_two"));
        assert_eq!(blame.len(), 1);
    }

    #[test]
    fn repair_kind_claims_keeps_unobserved_pipes() {
        let declared: BTreeMap<String, String> = [("silent_pipe".to_string(), "Cache".to_string())]
            .into_iter()
            .collect();
        let observed: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let (kept, blame) = repair_kind_claims(&declared, &observed);
        assert_eq!(kept.len(), 1);
        assert!(blame.is_empty());
    }

    #[test]
    fn repair_static_turns_cycle_fail_into_pass() {
        let spec = json!({
            "pipes": {
                "alpha": { "chain": ["beta"] },
                "beta": { "chain": ["alpha"] },
            }
        });
        let outcome = repair_static(&spec, &Policy::default());
        assert!(
            outcome.before.fail_count() >= 1,
            "cycle present before repair"
        );
        assert_eq!(outcome.after.fail_count(), 0, "no failures after repair");
        assert!(outcome.improved());
        assert!(outcome.fully_repaired());
        assert_eq!(outcome.blame.len(), 1);
    }

    // unification proof: same project_max_coherent, two domains.
    #[test]
    fn repair_from_recording_drops_must_derive_pipe_with_inferred_event() {
        let start_id = fresh_id(1);
        let end_id = fresh_id(2);
        let events = vec![
            make_event(start_id, None, http_started("fetch")),
            make_event(
                end_id,
                Some(start_id),
                http_ended_with(EventSource::Inferred),
            ),
        ];
        let policy = Policy::parse_str(
            r#"
                [replay]
                must_derive_from_record = ["fetch"]
            "#,
        )
        .expect("parse policy");

        let outcome = repair_from_recording(&events, None, &policy);
        assert!(outcome.before.fail_count() >= 1, "drift present pre-repair");
        assert_eq!(outcome.after.fail_count(), 0, "drift resolved post-repair");
        assert!(outcome.improved());
        assert_eq!(outcome.blame.len(), 1);
        assert!(matches!(
            &outcome.blame[0],
            RepairItem::MustDeriveClaim { pipe } if pipe == "fetch"
        ));
        assert!(
            outcome
                .repaired_policy
                .replay
                .must_derive_from_record
                .is_empty()
        );
    }

    #[test]
    fn repair_from_recording_keeps_must_derive_pipe_with_recorded_event() {
        let start_id = fresh_id(3);
        let end_id = fresh_id(4);
        let events = vec![
            make_event(start_id, None, http_started("fetch")),
            make_event(
                end_id,
                Some(start_id),
                http_ended_with(EventSource::Recorded),
            ),
        ];
        let policy = Policy::parse_str(
            r#"
                [replay]
                must_derive_from_record = ["fetch"]
            "#,
        )
        .expect("parse policy");

        let outcome = repair_from_recording(&events, None, &policy);
        assert_eq!(outcome.after.fail_count(), 0);
        assert!(outcome.blame.is_empty(), "no drift, nothing to drop");
        assert_eq!(
            outcome.repaired_policy.replay.must_derive_from_record.len(),
            1
        );
    }

    #[test]
    fn repair_from_recording_resolves_idempotence_contract() {
        // spec declares pipe non-idempotent; policy requires it derive
        // from recording. mutually contradictory — repair drops the
        // policy claim.
        let spec = json!({
            "pipes": {
                "fetch": { "idempotent": false, "http": "https://example.com" }
            }
        });
        let policy = Policy::parse_str(
            r#"
                [replay]
                must_derive_from_record = ["fetch"]
            "#,
        )
        .expect("parse policy");
        let events = vec![make_event(fresh_id(5), None, http_started("fetch"))];

        let outcome = repair_from_recording(&events, Some(&spec), &policy);
        assert!(outcome.before.fail_count() >= 1);
        assert_eq!(outcome.after.fail_count(), 0);
        assert_eq!(outcome.blame.len(), 1);
        assert!(
            outcome
                .repaired_policy
                .replay
                .must_derive_from_record
                .is_empty()
        );
    }

    #[test]
    fn repair_from_recording_only_drops_offending_entries() {
        // two pipes in must_derive: one drifts (inferred), one clean.
        // only the drifting one should be dropped.
        let drift_start = fresh_id(6);
        let drift_end = fresh_id(7);
        let clean_start = fresh_id(8);
        let clean_end = fresh_id(9);
        let events = vec![
            make_event(drift_start, None, http_started("drifter")),
            make_event(
                drift_end,
                Some(drift_start),
                http_ended_with(EventSource::Inferred),
            ),
            make_event(clean_start, None, http_started("steady")),
            make_event(
                clean_end,
                Some(clean_start),
                http_ended_with(EventSource::Recorded),
            ),
        ];
        let policy = Policy::parse_str(
            r#"
                [replay]
                must_derive_from_record = ["drifter", "steady"]
            "#,
        )
        .expect("parse policy");

        let outcome = repair_from_recording(&events, None, &policy);
        assert_eq!(outcome.after.fail_count(), 0);
        assert_eq!(outcome.blame.len(), 1, "only one drop");
        assert!(matches!(
            &outcome.blame[0],
            RepairItem::MustDeriveClaim { pipe } if pipe == "drifter"
        ));
        let kept = &outcome.repaired_policy.replay.must_derive_from_record;
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0], "steady");
    }

    #[test]
    fn repair_item_display_is_grep_friendly() {
        let edge = RepairItem::ChainEdge {
            from: "alpha".to_string(),
            to: "beta".to_string(),
        };
        assert_eq!(edge.to_string(), "chain_edge alpha → beta");
        let claim = RepairItem::KindClaim {
            pipe: "cache".to_string(),
            declared: "Cache".to_string(),
        };
        assert_eq!(claim.to_string(), "kind_claim cache = Cache");
        let derive = RepairItem::MustDeriveClaim {
            pipe: "fetch".to_string(),
        };
        assert_eq!(derive.to_string(), "must_derive fetch");
    }

    #[test]
    fn project_drives_static_and_replay_repair_with_one_function() {
        let edges = vec![
            (
                RepairItem::ChainEdge {
                    from: "alpha".to_string(),
                    to: "beta".to_string(),
                },
                Weight(2),
            ),
            (
                RepairItem::ChainEdge {
                    from: "beta".to_string(),
                    to: "alpha".to_string(),
                },
                Weight(1),
            ),
        ];
        let (edge_kept, edge_blame) = project_max_coherent(edges, |kept| !contains_cycle(kept));
        assert_eq!(edge_kept.len(), 1);
        assert_eq!(edge_blame.len(), 1);

        let claims = vec![(
            RepairItem::KindClaim {
                pipe: "pipe_one".to_string(),
                declared: "Cache".to_string(),
            },
            Weight(1),
        )];
        let mut observed_one = BTreeSet::new();
        observed_one.insert("Transform".to_string());
        let observed: BTreeMap<String, BTreeSet<String>> = [("pipe_one".to_string(), observed_one)]
            .into_iter()
            .collect();

        let (claim_kept, claim_blame) = project_max_coherent(claims, |kept| {
            kept.iter().all(|item| match item {
                RepairItem::KindClaim { pipe, declared } => observed
                    .get(pipe)
                    .is_none_or(|kinds| kinds.contains(declared)),
                _ => true,
            })
        });
        assert!(claim_kept.is_empty());
        assert_eq!(claim_blame.len(), 1);
    }
}
