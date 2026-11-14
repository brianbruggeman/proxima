//! Plan-time wavefront (level) schedule over `Prepared::resolved`.
//!
//! A sibling of `prune_dead` / `promote_output_placed_nodes` / `fuse_twin_elementwise`: a synchronous
//! `Vec<BoundOp> -> Vec<BoundOp>` permutation computed once in O(V + E), before `node_retirement`, so
//! lifetimes and arena slots are re-derived from the new order. Generic over each op's read set
//! ([`op_read_nodes`]), write set ([`extra_output_nodes`]) and a chain bitmask that pins ops sharing
//! graph-invisible state to program order. The runtime `HazardTracker` is unchanged and still decides
//! every barrier; this only puts independent ops next to each other.

use super::*;

/// Chain bit: ops sharing the plan-level attention scratch buffer keep program order.
pub(super) const CHAIN_ATTENTION_SCRATCH: u32 = 1 << 0;
/// Chain bit: ops touching a caller-placed (KV cache) buffer keep program order.
pub(super) const CHAIN_PLACED_BUFFER: u32 = 1 << 1;

/// Per-op chain bitmask. An op with a scratch length shares (conservatively, whatever the plan's
/// later row schedule says) the attention scratch; when any input is caller-placed, every op that
/// reads a placed input or writes a requested output may alias the placed buffer.
pub(super) fn chain_keys(
    resolved: &[BoundOp],
    numeric_policy: NumericPolicy,
    placed_input_nodes: &[NodeId],
    effective_outputs: &[NodeId],
) -> Vec<u32> {
    resolved
        .iter()
        .map(|bound| {
            let mut mask = 0;
            let schedule = crate::msl::AttentionRowSchedule::legacy();
            if cached_attention_scratch_len_with_schedule(bound, numeric_policy, schedule).is_some()
            {
                mask |= CHAIN_ATTENTION_SCRATCH;
            }
            if !placed_input_nodes.is_empty() {
                let mut touches_placed = effective_outputs.contains(&bound.node);
                op_read_nodes(bound, |read| {
                    touches_placed |= placed_input_nodes.contains(&read);
                });
                if touches_placed {
                    mask |= CHAIN_PLACED_BUFFER;
                }
            }
            mask
        })
        .collect()
}

/// Predecessor graph in CSR form: the producers of op `index` are
/// `targets[offsets[index]..offsets[index + 1]]`, all `< index`. A chain bit adds an edge from the
/// previous op carrying the same bit.
pub(super) fn dependency_csr(
    resolved: &[BoundOp],
    node_count: usize,
    chains: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    let mut producer = vec![u32::MAX; node_count];
    let mut offsets = Vec::with_capacity(resolved.len() + 1);
    let mut targets = Vec::new();
    let mut stamp = vec![u32::MAX; resolved.len()];
    let mut last_in_chain = [u32::MAX; u32::BITS as usize];
    offsets.push(0);
    for (index, bound) in resolved.iter().enumerate() {
        let mut add = |source: u32, targets: &mut Vec<u32>| {
            if source != u32::MAX && source as usize != index && stamp[source as usize] != index as u32
            {
                stamp[source as usize] = index as u32;
                targets.push(source);
            }
        };
        op_read_nodes(bound, |read| {
            if let Some(source) = producer.get(read.0 as usize) {
                add(*source, &mut targets);
            }
        });
        let mask = chains.get(index).copied().unwrap_or(0);
        for (bit, last) in last_in_chain.iter_mut().enumerate() {
            if mask & (1 << bit) != 0 {
                add(*last, &mut targets);
                *last = index as u32;
            }
        }
        if let Some(slot) = producer.get_mut(bound.node.0 as usize) {
            *slot = index as u32;
        }
        for (extra, _) in extra_output_nodes(bound) {
            if let Some(slot) = producer.get_mut(extra.0 as usize) {
                *slot = index as u32;
            }
        }
        offsets.push(targets.len() as u32);
    }
    (offsets, targets)
}

/// `level[index] = 1 + max(level[producer])`, 0 with none. Program order is topological, so one
/// forward pass suffices; the level count is the DAG depth, the minimum barrier count.
pub(super) fn wavefront_levels(offsets: &[u32], targets: &[u32]) -> Vec<u32> {
    let mut levels = vec![0u32; offsets.len().saturating_sub(1)];
    for index in 0..levels.len() {
        let producers = &targets[offsets[index] as usize..offsets[index + 1] as usize];
        levels[index] = producers
            .iter()
            .map(|producer| levels[*producer as usize] + 1)
            .max()
            .unwrap_or(0);
    }
    levels
}

/// Stable counting sort by `(level, original index)`: the new order and, per level, the position of
/// its first op.
pub(super) fn wavefront_order(levels: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let level_count = levels.iter().max().map_or(0, |deepest| *deepest as usize + 1);
    let mut level_starts = vec![0u32; level_count + 1];
    for level in levels {
        level_starts[*level as usize + 1] += 1;
    }
    for level in 0..level_count {
        level_starts[level + 1] += level_starts[level];
    }
    let mut cursor = level_starts.clone();
    let mut order = vec![0u32; levels.len()];
    for (index, level) in levels.iter().enumerate() {
        order[cursor[*level as usize] as usize] = index as u32;
        cursor[*level as usize] += 1;
    }
    level_starts.pop();
    (order, level_starts)
}

/// Last position of the level holding `position`.
fn level_end(level_starts: &[u32], total: usize, position: usize) -> usize {
    let next = level_starts.partition_point(|start| *start as usize <= position);
    level_starts
        .get(next)
        .map_or(total, |start| *start as usize)
        - 1
}

/// Moves every retirement to the last position of its level, so a slot released inside a level is
/// not reusable inside it (that reuse would be a WAR barrier inside one concurrent run).
pub(super) fn defer_retires_to_level_end(
    retires: Vec<Vec<NodeId>>,
    level_starts: &[u32],
) -> Vec<Vec<NodeId>> {
    let total = retires.len();
    let mut deferred = vec![Vec::new(); total];
    for (position, retired) in retires.into_iter().enumerate() {
        deferred[level_end(level_starts, total, position)].extend(retired);
    }
    deferred
}

/// Snaps each ascending chunk boundary to the nearest level start at or before it, so a command
/// buffer never cuts between independent ops of one level. Empty `level_starts` is program order.
pub(super) fn snap_to_level_starts(boundaries: &[usize], level_starts: &[u32]) -> Vec<usize> {
    let mut snapped: Vec<usize> = boundaries
        .iter()
        .map(|boundary| {
            let after = level_starts.partition_point(|start| *start as usize <= *boundary);
            after
                .checked_sub(1)
                .map_or(*boundary, |index| level_starts[index] as usize)
        })
        .collect();
    snapped.dedup();
    snapped
}

/// The reordered program and its level starts, or `None` when the order would not help or the
/// arena peak of the reordered plan exceeds `cap_bytes` (declined: program order stands).
#[must_use]
pub(super) fn wavefront_schedule(
    resolved: &[BoundOp],
    chains: &[u32],
    node_count: usize,
    effective_outputs: &[NodeId],
    cap_bytes: usize,
) -> Option<(Vec<BoundOp>, Vec<u32>)> {
    let (offsets, targets) = dependency_csr(resolved, node_count, chains);
    let levels = wavefront_levels(&offsets, &targets);
    let (order, level_starts) = wavefront_order(&levels);
    let reordered: Vec<BoundOp> = order
        .iter()
        .map(|index| resolved[*index as usize].clone())
        .collect();
    let peak_bytes = reordered_peak_bytes(&reordered, effective_outputs, &level_starts);
    if let Some(peak_bytes) = peak_bytes.filter(|peak| *peak > cap_bytes) {
        debug!(
            peak_bytes = peak_bytes as u64,
            cap_bytes = cap_bytes as u64,
            levels = level_starts.len() as u64,
            "wavefront reorder declined, program order stands: arena peak over transient cap"
        );
        return None;
    }
    Some((reordered, level_starts))
}

#[cfg(feature = "metal-plan-stable-buffers")]
fn reordered_peak_bytes(
    reordered: &[BoundOp],
    effective_outputs: &[NodeId],
    level_starts: &[u32],
) -> Option<usize> {
    let outputs: BTreeSet<NodeId> = effective_outputs.iter().copied().collect();
    let retires = defer_retires_to_level_end(node_retirement(reordered, effective_outputs), level_starts);
    let found = arena_allocations(reordered, &retires, &outputs, &BTreeSet::new(), |_| false);
    Some(peak_live_bytes(&found.allocations, &found.releases))
}

#[cfg(not(feature = "metal-plan-stable-buffers"))]
fn reordered_peak_bytes(_: &[BoundOp], _: &[NodeId], _: &[u32]) -> Option<usize> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Producer positions (window positions 56..75 as 0..19) from the design's section 4 edge list.
    const WINDOW_PREDS: [&[u32]; 20] = [
        &[],
        &[0],
        &[0],
        &[0],
        &[3],
        &[4],
        &[2],
        &[6],
        &[7, 5, 1],
        &[8],
        &[9],
        &[10],
        &[11],
        &[10, 12],
        &[13],
        &[14],
        &[10],
        &[16],
        &[17],
        &[15, 18],
    ];

    fn csr(preds: &[&[u32]]) -> (Vec<u32>, Vec<u32>) {
        let mut offsets = vec![0];
        let mut targets = Vec::new();
        for each in preds {
            targets.extend_from_slice(each);
            offsets.push(targets.len() as u32);
        }
        (offsets, targets)
    }

    fn barriers(order: &[u32], preds: &[&[u32]]) -> usize {
        use super::super::execute_and_hazards::{HazardTracker, hazard_step};
        const INCOMING: usize = 1000;
        let mut tracker = HazardTracker::<usize>::new();
        tracker.record(&[], [INCOMING]);
        let mut fired = 0;
        for index in order {
            let mut inputs: Vec<usize> = preds[*index as usize].iter().map(|pred| *pred as usize).collect();
            if *index == 0 {
                inputs.push(INCOMING);
            }
            fired += usize::from(hazard_step(&mut tracker, &inputs, &[*index as usize]));
        }
        fired
    }

    #[test]
    fn wavefront_levels_match_step31_attention_moe_window() {
        let (offsets, targets) = csr(&WINDOW_PREDS);

        let levels = wavefront_levels(&offsets, &targets);
        let (order, level_starts) = wavefront_order(&levels);

        assert_eq!(levels, [0, 1, 1, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 7, 8, 9, 12]);
        assert_eq!(level_starts.len(), 13);
        let identity: Vec<u32> = (0..20).collect();
        assert_eq!(barriers(&identity, &WINDOW_PREDS), 16);
        assert_eq!(barriers(&order, &WINDOW_PREDS), 13);
    }

    #[test]
    fn order_is_a_permutation_with_forward_edges_and_partitioned_levels() {
        let (offsets, targets) = csr(&WINDOW_PREDS);
        let levels = wavefront_levels(&offsets, &targets);

        let (order, level_starts) = wavefront_order(&levels);

        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..20).collect::<Vec<u32>>());
        let slot_of: Vec<usize> = (0..20)
            .map(|index| order.iter().position(|each| *each == index).expect("every op is placed"))
            .collect();
        for (index, preds) in WINDOW_PREDS.iter().enumerate() {
            assert!(preds.iter().all(|pred| slot_of[*pred as usize] < slot_of[index]));
        }
        assert_eq!(level_starts[0], 0);
        assert!(level_starts.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn flag_off_empty_level_starts_leave_boundaries_in_program_order() {
        let boundaries = [24, 60, 120];

        let snapped = snap_to_level_starts(&boundaries, &[]);

        assert_eq!(snapped, boundaries);
    }

    #[test]
    fn boundaries_snap_back_to_the_start_of_their_level() {
        let (offsets, targets) = csr(&WINDOW_PREDS);
        let (_, level_starts) = wavefront_order(&wavefront_levels(&offsets, &targets));

        let snapped = snap_to_level_starts(&[2, 7], &level_starts);

        assert_eq!(snapped, [1, 6]);
    }

    #[test]
    fn retires_defer_to_the_last_position_of_their_level() {
        let (offsets, targets) = csr(&WINDOW_PREDS);
        let (_, level_starts) = wavefront_order(&wavefront_levels(&offsets, &targets));
        let mut retires = vec![Vec::new(); 20];
        retires[1].push(NodeId(7));

        let deferred = defer_retires_to_level_end(retires, &level_starts);

        assert_eq!(deferred[3], [NodeId(7)]);
        assert!(deferred[1].is_empty());
    }
}

#[cfg(test)]
mod real_fixture_tests {
    use super::*;
    use proxima_tensor::{
        CachedAttentionArithmetic, Extent, IndexMap, Keep, Layout, Reduce, ReduceInit, ScalarOp,
        append, projection,
    };

    /// flint-shaped cached attention reading nodes 0..9 and writing `node`.
    fn attention(node: u32) -> BoundOp {
        let operands = (0..9)
            .map(|index| {
                (
                    NodeId(index),
                    Layout {
                        base: 0,
                        strides: vec![1].into(),
                    },
                    None,
                )
            })
            .collect();
        BoundOp {
            node: NodeId(node),
            dtype: DType::FLOAT32,
            extents: vec![49, 1, 8, 256],
            kind: BoundOpKind::CachedAttention {
                operands,
                query_rows: 49,
                cached_key_rows: 512,
                new_key_rows: 49,
                kv_heads: 1,
                query_groups: 8,
                head_dim: 256,
                rotary_dim: 256,
                scale: 1.0,
                cached_lower_inclusive: -511,
                new_upper_inclusive: 0,
                arithmetic: CachedAttentionArithmetic::OnlineF32,
                attention_sink: false,
            },
        }
    }

    /// The program the hazard tests use: `y`, a second reader of `y`, and a reduce with `y` in its
    /// epilogue; returns the program and its two requested outputs plus the placed leaf.
    fn program() -> (Vec<Op>, Vec<NodeId>, NodeId) {
        let mut program = Vec::new();
        let identity = || IndexMap::Affine(projection(1, &[0]));
        let raw = append(
            &mut program,
            Op::Input {
                dtype: DType::FLOAT32,
                shape: vec![Extent::Static(4)],
                name: None,
            },
        );
        let y = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::FLOAT32,
                body: ScalarOp::Identity,
                operands: vec![(raw, identity())],
                name: None,
            },
        );
        let weights = append(
            &mut program,
            Op::Input {
                dtype: DType::FLOAT32,
                shape: vec![Extent::Static(8), Extent::Static(4)],
                name: None,
            },
        );
        let reduced = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::FLOAT32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: weights,
                in_map: IndexMap::Affine(projection(2, &[0, 1])),
                out_map: IndexMap::Affine(projection(2, &[1])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        let consumer = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::FLOAT32,
                body: ScalarOp::Add,
                operands: vec![(reduced, identity()), (y, identity())],
                name: None,
            },
        );
        let extra = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::FLOAT32,
                body: ScalarOp::Negate,
                operands: vec![(y, identity())],
                name: None,
            },
        );
        (program, vec![consumer, extra], raw)
    }

    fn prepared(wavefront: bool, placed: &[NodeId]) -> Prepared {
        let (program, outputs, _) = program();
        let raw = [0.5f32; 4];
        let weights = [0.25f32; 32];
        let blocks = [QuantizedBlock::Float32(&raw), QuantizedBlock::Float32(&weights)];
        prepare_with_options(
            &program,
            &[],
            &blocks,
            &outputs,
            (NumericPolicy::default(), true, true, wavefront),
            placed,
        )
        .expect("the real fixture program prepares")
    }

    #[test]
    fn flag_off_plan_equals_the_default_prepare_path_in_program_order() {
        let (program, outputs, _) = program();
        let raw = [0.5f32; 4];
        let weights = [0.25f32; 32];
        let blocks = [QuantizedBlock::Float32(&raw), QuantizedBlock::Float32(&weights)];
        let pre_wavefront = prepare(&program, &[], &blocks, &outputs, NumericPolicy::default(), &[], (true, false))
            .expect("the default path prepares");

        let off = prepared(false, &[]);

        assert!(!off.resolved.is_empty());
        assert_eq!(off.resolved, pre_wavefront.resolved);
        assert_eq!(off.retires, pre_wavefront.retires);
        assert!(off.level_starts.is_empty());
    }

    #[test]
    fn the_scheduled_plan_option_reaches_prepare_and_off_leaves_no_levels() {
        let (program, outputs, _) = program();
        let raw = [0.5f32; 4];
        let weights = [0.25f32; 32];
        let blocks = [QuantizedBlock::Float32(&raw), QuantizedBlock::Float32(&weights)];
        let plan = |wavefront: bool| {
            plan_with_placed_inputs(
                &program,
                &[],
                &blocks,
                &outputs,
                NumericPolicy::default(),
                &[],
                (true, wavefront),
            )
            .expect("the real fixture plans")
        };

        let on = plan(true);
        let off = plan(false);

        assert!(!on.prepared.level_starts.is_empty());
        assert!(off.prepared.level_starts.is_empty());
    }

    #[test]
    fn flag_on_is_a_permutation_of_the_flag_off_plan_with_level_starts() {
        let off = prepared(false, &[]);

        let on = prepared(true, &[]);

        assert!(!on.level_starts.is_empty());
        assert_eq!(on.resolved.len(), off.resolved.len());
        assert!(off.resolved.iter().all(|bound| on.resolved.contains(bound)));
        assert_eq!(on.level_starts[0], 0);
    }

    #[test]
    fn placed_input_chain_keeps_program_order_of_every_op_touching_it() {
        let (_, _, raw) = program();
        let off = prepared(false, &[raw]);

        let on = prepared(true, &[raw]);

        let off_nodes: Vec<NodeId> = off.resolved.iter().map(|bound| bound.node).collect();
        let on_nodes: Vec<NodeId> = on.resolved.iter().map(|bound| bound.node).collect();
        let chained = |nodes: &[NodeId]| -> Vec<NodeId> {
            nodes
                .iter()
                .copied()
                .filter(|node| {
                    let bound = off.resolved.iter().find(|each| each.node == *node);
                    bound.is_some_and(|bound| {
                        off.effective_outputs.contains(node) || {
                            let mut reads_placed = false;
                            op_read_nodes(bound, |read| reads_placed |= read == raw);
                            reads_placed
                        }
                    })
                })
                .collect()
        };
        assert!(chained(&off_nodes).len() >= 2);
        assert_eq!(chained(&on_nodes), chained(&off_nodes));
    }

    #[test]
    fn attention_ops_sharing_a_scratch_never_share_a_level() {
        let resolved = vec![attention(20), attention(21)];
        let policy = NumericPolicy::default();

        let chains = chain_keys(&resolved, policy, &[], &[]);
        let chained = wavefront_levels_for(&resolved, &chains);
        let unchained = wavefront_levels_for(&resolved, &[0, 0]);

        assert_eq!(chains, [CHAIN_ATTENTION_SCRATCH, CHAIN_ATTENTION_SCRATCH]);
        assert_eq!(chained, [0, 1]);
        assert_eq!(unchained, [0, 0]);
    }

    fn wavefront_levels_for(resolved: &[BoundOp], chains: &[u32]) -> Vec<u32> {
        let (offsets, targets) = dependency_csr(resolved, 32, chains);
        wavefront_levels(&offsets, &targets)
    }

    #[test]
    fn arena_gate_declines_over_cap_and_accepts_under_it() {
        let resolved = vec![attention(20), attention(21)];
        let chains = [0, 0];

        let declined = wavefront_schedule(&resolved, &chains, 32, &[], 0);
        let accepted = wavefront_schedule(&resolved, &chains, 32, &[], usize::MAX);

        assert!(declined.is_none(), "a zero cap is below any peak, so program order stands");
        let (reordered, level_starts) = accepted.expect("an unbounded cap admits the reorder");
        assert_eq!(reordered, resolved);
        assert_eq!(level_starts, [0]);
    }
}
