use super::*;

/// The prefix state of op building: elementwise ops seen but not yet
/// materialized.
///
/// `retires` and `position` make [`BoundOpBuilder::push`] a single-argument-per-node
/// step (`expr`, `shapes`) rather than needing `node`/`retires` threaded in
/// by the caller on every call: `retires[i]` is node `i`'s kill-flag list
/// (see [`live::annotate`], computed once over the whole program before the
/// first push), and `position` is the node id the next push resolves to —
/// both pieces this type already needed to know, now carried as its own
/// state instead of repeated arguments.
pub struct BoundOpBuilder {
    pub(super) held: RefCell<BTreeMap<NodeId, HeldElementwise>>,
    pub(super) retires: Vec<Vec<NodeId>>,
    pub(super) position: Cell<u32>,
    /// `ones[node.0]` is `true` when `node` was pushed as an
    /// [`Op::Constant`] whose `value` is exactly `1.0` — grown one entry per
    /// [`push`](Self::push) call, never read ahead of the position that
    /// produced it. This is what lets [`compose_operand`] recognize (and
    /// drop) a `Multiply` operand that is algebraically a no-op without
    /// requiring the whole program in hand, honoring this module's own
    /// sans-IO streaming contract (see module doc) rather than threading a
    /// full `&[Op]` slice through every fusion call.
    pub(super) ones: RefCell<Vec<bool>>,
    /// `is_iota[node.0]` is `true` when `node` was pushed as an [`Op::Iota`]
    /// — the same one-entry-per-push discipline as `ones`, read by
    /// [`eliminate_masked_window_reduce`] to confirm a candidate operand is
    /// really one of `window_mask`'s three position markers rather than some
    /// other node that merely shares its `NodeId` shape.
    pub(super) is_iota: RefCell<Vec<bool>>,
    /// Whether this node or an elementwise descendant carries a multi-term
    /// index map. A descendant may already be materialized when a consuming
    /// reduce is pushed, so the original shape must survive outside `held`.
    pub(super) packed_mapping_subtree: RefCell<Vec<bool>>,
    /// `constant_value[node.0]` is `Some(value)` when `node` was pushed as an
    /// [`Op::Constant`] carrying `value` — a generalization of `ones` that
    /// keeps the actual stride literal (not just whether it is `1.0`), which
    /// [`eliminate_masked_window_reduce`]'s in-bounds proof needs.
    pub(super) constant_value: RefCell<Vec<Option<f32>>>,
    /// The [`NumericPolicy`] every [`Constants`] this builder hands to
    /// [`push_canonical_step`] carries — governs whether the
    /// `x+0` ([`NumericRewrite::IdentityEliminationSignedZero`]) and
    /// `max(x,-inf)`/`min(x,+inf)` ([`NumericRewrite::IdentityEliminationNanAssumption`])
    /// identity eliminations fire, on top of the always-on `x*1` case.
    pub(super) numeric_policy: NumericPolicy,
}

impl BoundOpBuilder {
    /// `retires` is normally [`live::annotate`]`(program, outputs)`.
    #[must_use]
    pub fn new(retires: Vec<Vec<NodeId>>, numeric_policy: NumericPolicy) -> Self {
        Self {
            held: RefCell::new(BTreeMap::new()),
            retires,
            position: Cell::new(0),
            ones: RefCell::new(Vec::new()),
            is_iota: RefCell::new(Vec::new()),
            packed_mapping_subtree: RefCell::new(Vec::new()),
            constant_value: RefCell::new(Vec::new()),
            numeric_policy,
        }
    }

    /// Judge one expression: hold an elementwise op, or emit whatever is now
    /// ready.
    ///
    /// May return more than one [`BoundOp`]: consuming a held elementwise op
    /// that turns out not to fuse must materialize it before the current
    /// expression can read it, so a single push can ready both that
    /// standalone op and the current expression's own — and, for an
    /// elementwise expression, one materialization per operand that fails to
    /// fuse, up to [`ScalarOp::arity`]'s current maximum
    /// (`READY_BATCH_CAPACITY`).
    pub fn push(&self, expr: &Op, shapes: &Shapes) -> Result<ReadyBatch, TensorError> {
        let node = NodeId(self.position.get());
        self.position.set(self.position.get() + 1);
        let empty = Vec::new();
        let retires = self.retires.get(node.0 as usize).unwrap_or(&empty);
        self.ones
            .borrow_mut()
            .push(matches!(expr, Op::Constant { value, .. } if *value == 1.0));
        self.is_iota
            .borrow_mut()
            .push(matches!(expr, Op::Iota { .. }));
        let packed_mapping = match expr {
            Op::Elementwise { operands, .. } => {
                let direct = operands.iter().any(|(_, map)| {
                    map.affine()
                        .axes
                        .iter()
                        .any(|axis| axis.terms.len() > 1)
                });
                let descendants = self.packed_mapping_subtree.borrow();
                direct
                    || operands.iter().any(|(operand, _)| {
                        descendants.get(operand.0 as usize).copied().unwrap_or(false)
                    })
            }
            _ => false,
        };
        self.packed_mapping_subtree
            .borrow_mut()
            .push(packed_mapping);
        self.constant_value
            .borrow_mut()
            .push(if let Op::Constant { value, .. } = expr {
                Some(*value)
            } else {
                None
            });

        let mut emitted = ReadyBatch::new();

        match expr {
            Op::Input { .. } => {}
            Op::Iota { dtype, .. } => {
                let extents = shapes.of(node).to_vec();
                push_ready(
                    &mut emitted,
                    node,
                    BoundOp {
                        node,
                        dtype: *dtype,
                        extents,
                        kind: BoundOpKind::Iota,
                    },
                )?;
            }
            Op::Constant { dtype, value, .. } => {
                let extents = shapes.of(node).to_vec();
                push_ready(
                    &mut emitted,
                    node,
                    BoundOp {
                        node,
                        dtype: *dtype,
                        extents,
                        kind: BoundOpKind::Constant { value: *value },
                    },
                )?;
            }
            Op::Elementwise {
                dtype,
                body,
                operands,
                ..
            } => {
                for (operand_node, map) in operands {
                    let still_live = !retires.contains(operand_node);
                    let non_identity = !is_identity_projection(map);
                    let not_held = !self.held.borrow().contains_key(operand_node);
                    let fuses = !still_live && !non_identity && !not_held;
                    #[cfg(feature = "instrument")]
                    {
                        let outcome = if fuses {
                            Ok(())
                        } else if still_live {
                            Err(instrument::FuseDeclineReason::StillLive)
                        } else if non_identity {
                            Err(instrument::FuseDeclineReason::NonIdentityProjection)
                        } else {
                            Err(instrument::FuseDeclineReason::NotHeld)
                        };
                        instrument::record_fuse_attempt(
                            *operand_node,
                            instrument::FuseSite::ElementwiseOperand,
                            outcome,
                        );
                        debug!(
                            node = operand_node.0,
                            kind = "elementwise_operand_fuse",
                            decision = if fuses { "fused" } else { "materialized" },
                            into = node.0,
                            still_live = still_live,
                            "single-consumer elementwise composition decision -- still_live is gated by the requested output set"
                        );
                    }
                    if !fuses {
                        self.materialize_if_held(*operand_node, shapes, &mut emitted)?;
                    }
                    self.materialize_computed_indices(map, shapes, &mut emitted)?;
                }
                self.held.borrow_mut().insert(
                    node,
                    HeldElementwise {
                        dtype: *dtype,
                        body: *body,
                        operands: operands.clone(),
                    },
                );
            }
            Op::Reduce(reduce) => {
                let window_elimination = eliminate_masked_window_reduce(
                    reduce,
                    &self.held,
                    &self.is_iota.borrow(),
                    &self.constant_value.borrow(),
                    shapes,
                );
                #[cfg(feature = "instrument")]
                instrument::record_window_reduce_attempt(window_elimination.is_some());
                if let Some((source_node, source_map)) = window_elimination {
                    // `source_map`'s windowed axis is a genuine two-term
                    // affine index (`stride*out + kernel`), not the plain
                    // single-term projection `compose_operand`'s own fusion
                    // contract requires of a map connecting to a still-held
                    // node (`is_identity_projection`'s own doc: "a window...
                    // materializes its operand instead of composing through
                    // it"). `source` may still be `held` here — e.g. a prior
                    // layer's bias-add fusing into this window on the
                    // ordinary identity-projection path it was pushed under
                    // — so it must be forced to materialize as a real buffer
                    // before this non-identity map ever reads it, or
                    // `compose_operand`'s recursive remap silently
                    // mis-addresses whatever was held beneath it.
                    self.materialize_if_held(source_node, shapes, &mut emitted)?;
                    let identity_operand = vec![(source_node, source_map)];
                    push_ready(
                        &mut emitted,
                        node,
                        build_elementwise_op(
                            node,
                            shapes,
                            &self.held,
                            reduce.dtype,
                            ScalarOp::Identity,
                            &identity_operand,
                            Constants {
                                ones: &self.ones.borrow(),
                                values: &self.constant_value.borrow(),
                                numeric_policy: self.numeric_policy,
                            },
                        ),
                    )?;
                    return Ok(emitted);
                }

                let still_live = !retires.contains(&reduce.operand);
                let non_identity = !is_identity_projection(&reduce.in_map);
                let not_held = !self.held.borrow().contains_key(&reduce.operand);
                let fuses = !still_live && !non_identity && !not_held;
                if fuses
                    && let Some(activation_node) =
                        composed_packed_product_activation(
                            &self.held,
                            &self.packed_mapping_subtree,
                            reduce.operand,
                        )
                {
                    // ROW 431 (`docs/discipline.md`): materialize ONLY the
                    // composed activation side of `Multiply(packed, a)` so
                    // `W * a` and this reduction stay fused --
                    // `run_reduce_quantized`'s admission contract
                    // (`packed_reduce_activation_operand`, `cpu.rs`) requires
                    // a bare two-operand product, and this is what makes
                    // that shape true without ever materializing the
                    // output-width product ROW 430 used to (superseded).
                    #[cfg(feature = "instrument")]
                    debug!(
                        node = reduce.operand.0,
                        activation = activation_node.0,
                        reduce = node.0,
                        "reduce fusion materializes the composed activation operand of a \
                         packed-weight product so W * a and the reduction stay fused \
                         (docs/discipline.md ROW 431, supersedes ROW 430)"
                    );
                    self.materialize_if_held(activation_node, shapes, &mut emitted)?;
                }
                #[cfg(feature = "instrument")]
                {
                    let outcome = if fuses {
                        Ok(())
                    } else if still_live {
                        Err(instrument::FuseDeclineReason::StillLive)
                    } else if non_identity {
                        Err(instrument::FuseDeclineReason::NonIdentityProjection)
                    } else {
                        Err(instrument::FuseDeclineReason::NotHeld)
                    };
                    instrument::record_fuse_attempt(
                        reduce.operand,
                        instrument::FuseSite::ReduceOperand,
                        outcome,
                    );
                }

                let (element_body, operands) = if fuses {
                    let reduce_extent: u64 = shape::fold_iteration_extents(node, reduce, shapes)?
                        .iter()
                        .product();
                    self.quarantine_broadcast_operands(
                        reduce.operand,
                        reduce_extent,
                        shapes,
                        &mut emitted,
                    )?;
                    compose_fused_operands(
                        shapes,
                        &self.held,
                        reduce.operand,
                        &reduce.in_map,
                        Constants {
                            ones: &self.ones.borrow(),
                            values: &self.constant_value.borrow(),
                            numeric_policy: self.numeric_policy,
                        },
                    )
                } else {
                    self.materialize_if_held(reduce.operand, shapes, &mut emitted)?;
                    self.materialize_computed_indices(&reduce.in_map, shapes, &mut emitted)?;
                    let operand = build_operand(reduce.operand, &reduce.in_map, shapes);
                    (ComposedBody::leaf(ScalarOp::Identity), vec![operand])
                };

                // A scatter's `out_map` names an `indices` node exactly the
                // way `in_map` can, and it is never covered by the
                // `in_map`-only walk above (`fuses`/the `else` branch both
                // only ever touch `reduce.operand`/`reduce.in_map`) — see
                // this method's own doc for why `materialize_computed_indices`
                // is unconditional for `in_map`; the same reasoning applies
                // here, independent of whether the operand fused.
                self.materialize_computed_indices(&reduce.out_map, shapes, &mut emitted)?;

                push_ready(
                    &mut emitted,
                    node,
                    build_reduce_op(node, reduce, shapes, element_body, operands)?,
                )?;
            }
        }

        Ok(emitted)
    }

    /// [`bind_plain`]'s reachability skip lane: advances the position
    /// counter and keeps every per-node bookkeeping vector
    /// (`ones`/`is_iota`/`packed_mapping_subtree`/`constant_value`) aligned
    /// to it, without running any of [`push`](Self::push)'s binding work.
    /// Safe because a program only ever references backwards
    /// (`op.rs`'s own module doc), so no node this method skips can be an
    /// operand, gather index, or reduce map of a node the caller does push
    /// — every read of these vectors is at a live node's own index.
    pub fn skip(&self) {
        self.position.set(self.position.get() + 1);
        self.ones.borrow_mut().push(false);
        self.is_iota.borrow_mut().push(false);
        self.packed_mapping_subtree.borrow_mut().push(false);
        self.constant_value.borrow_mut().push(None);
    }

    /// Flush every elementwise op still held: each was a requested output,
    /// and either way it materializes as its own op. A node reachable from
    /// no output never enters `held` at all — [`bind_plain`] never calls
    /// [`push`](Self::push) for it — so this no longer flushes dead code.
    /// Processed from the highest [`NodeId`] down: a still-held node can
    /// only ever be fused into a consumer with a *greater* id (references
    /// point backwards only), so visiting consumers first lets
    /// `build_elementwise_op` absorb whatever it still can before an
    /// earlier, now-absorbed node would otherwise be flushed standalone.
    pub fn finish(self, shapes: &Shapes) -> Result<Vec<BoundOp>, TensorError> {
        let mut remaining: Vec<NodeId> = self.held.borrow().keys().copied().collect();
        remaining.sort_unstable_by(|left, right| right.cmp(left));

        let mut built = Vec::new();
        for node in remaining {
            let mut materialized = self.materialize_node(node, shapes)?;
            materialized.reverse();
            built.extend(materialized);
        }
        built.reverse();
        Ok(built)
    }

    /// A [`IndexMap::Computed`]'s `indices` is a backwards [`NodeId`]
    /// reference that never appears in any op's own `operands` list — it
    /// only ever lives inside a sibling operand's *map* — so the operand
    /// walk in [`Self::push`] must force it here too, or a lone held
    /// `Elementwise` reached only this way sits un-materialized past the
    /// point a gather reads it (`docs/discipline.md` ROW 131 Limitation 2).
    /// `indices` is always read as a plain buffer at evaluation time
    /// ([`build_operand`]'s `Lookup`, never composed through by
    /// [`compose_operand`]), so unlike `operand_node` this is unconditional
    /// — there is no fusion path for it to opt out of.
    fn materialize_computed_indices(
        &self,
        map: &IndexMap,
        shapes: &Shapes,
        emitted: &mut ReadyBatch,
    ) -> Result<(), TensorError> {
        if let IndexMap::Computed { indices, .. } = map {
            self.materialize_if_held(*indices, shapes, emitted)?;
        }
        Ok(())
    }

    fn materialize_if_held(
        &self,
        node: NodeId,
        shapes: &Shapes,
        emitted: &mut ReadyBatch,
    ) -> Result<(), TensorError> {
        for materialized in self.materialize_node(node, shapes)? {
            push_ready(emitted, node, materialized)?;
        }
        Ok(())
    }

    fn materialize_node(&self, node: NodeId, shapes: &Shapes) -> Result<Vec<BoundOp>, TensorError> {
        let mut emitted = Vec::new();
        loop {
            let Some(held) = self.held.borrow().get(&node).cloned() else {
                return Ok(emitted);
            };
            if self.preview_elementwise_buffer_count(node, shapes)? <= 31 {
                break;
            }

            let candidate = held
                .operands
                .iter()
                .filter(|(operand, map)| {
                    is_identity_projection(map) && self.held.borrow().contains_key(operand)
                })
                .filter_map(|(operand, _)| {
                    let mut preview_held = self.held.borrow().clone();
                    preview_held.remove(operand);
                    let count = preview_elementwise_buffer_count(
                        node,
                        shapes,
                        &preview_held,
                        &self.ones.borrow(),
                        &self.constant_value.borrow(),
                        self.numeric_policy,
                    )
                    .ok()?;
                    Some((*operand, count))
                })
                .min_by_key(|(operand, count)| (*count, *operand))
                .map(|(operand, _)| operand)
                .ok_or(TensorError::NotLowerable {
                    node,
                    reason: "elementwise body exceeds Metal's 31-buffer ABI and has no composable child to materialize",
                })?;

            emitted.extend(self.materialize_node(candidate, shapes)?);
        }

        // see `finish`'s comment: the borrow must end before this `if let`
        // body runs, since `build_elementwise_op` borrows `self.held` too.
        let removed = self.held.borrow_mut().remove(&node);
        if let Some(held) = removed {
            emitted.push(build_elementwise_op(
                node,
                shapes,
                &self.held,
                held.dtype,
                held.body,
                &held.operands,
                Constants {
                    ones: &self.ones.borrow(),
                    values: &self.constant_value.borrow(),
                    numeric_policy: self.numeric_policy,
                },
            ));
        }
        Ok(emitted)
    }

    fn preview_elementwise_buffer_count(
        &self,
        node: NodeId,
        shapes: &Shapes,
    ) -> Result<usize, TensorError> {
        preview_elementwise_buffer_count(
            node,
            shapes,
            &self.held.borrow(),
            &self.ones.borrow(),
            &self.constant_value.borrow(),
            self.numeric_policy,
        )
    }

    /// Walks `node`'s still-held operands and materializes any whose own
    /// natural iteration space (`shapes.of(child)`) is smaller than
    /// `reduce_extent` — composing one through anyway would run its body
    /// once per `reduce_extent` element instead of once per its own, which
    /// is exactly the cost [`is_identity_projection`] cannot see: it only
    /// judges one map's shape, not what fusing recursively absorbs beneath
    /// it (see `compose_operand`'s own doc — it trusts every map it
    /// recurses through was already checked, but that check happened at a
    /// different, earlier `push`, against that op's own — smaller —
    /// iteration space, not against this reduce's). Safe children (same or
    /// larger extent) are walked further, since a broadcast can reappear
    /// several levels down.
    fn quarantine_broadcast_operands(
        &self,
        node: NodeId,
        reduce_extent: u64,
        shapes: &Shapes,
        emitted: &mut ReadyBatch,
    ) -> Result<(), TensorError> {
        let children = self
            .held
            .borrow()
            .get(&node)
            .map(|held| held.operands.clone());
        let Some(children) = children else {
            return Ok(());
        };
        // `composed_packed_product_activation`'s own admission contract
        // (ROW 431) keeps a packed-weight operand fused specifically
        // because it is broadcast over the reduce's own leading (batch)
        // axis -- `run_reduce_quantized`'s fast path is DESIGNED to stream
        // that operand's rows once and reuse them across every batch
        // position. The size-only test below cannot see that: it sees a
        // smaller-than-reduce extent and quarantines it, undoing ROW 431's
        // fusion and forcing the dense fallback (which then materializes
        // the packed weight's raw bytes into a buffer addressed by its
        // DECLARED axis order -- not the packed native row-major order
        // `materialize_quantized_weight_output` actually writes -- a
        // second, independent defect the fallback exposes but does not
        // cause). `packed_and_other_operand` (not
        // `composed_packed_product_activation` itself) because the latter
        // additionally requires the OTHER side still be a held elementwise,
        // which a plain activation `Op::Input` leaf never is -- irrelevant
        // to quarantine's own question of which side must stay fused.
        // Checked at every recursion level, not only the top, since a
        // packed product can repeat several levels down.
        let packed_operand = packed_and_other_operand(&self.held, &self.packed_mapping_subtree, node)
            .map(|(packed_node, _)| packed_node);
        for (child, _map) in children {
            if !self.held.borrow().contains_key(&child) {
                continue;
            }
            if packed_operand == Some(child) {
                self.quarantine_broadcast_operands(child, reduce_extent, shapes, emitted)?;
                continue;
            }
            let child_extent: u64 = shapes.of(child).iter().product();
            let quarantined = child_extent < reduce_extent;
            #[cfg(feature = "instrument")]
            instrument::record_fuse_attempt(
                child,
                instrument::FuseSite::QuarantineBroadcast,
                if quarantined {
                    Err(instrument::FuseDeclineReason::BroadcastQuarantined)
                } else {
                    Ok(())
                },
            );
            if quarantined {
                self.materialize_if_held(child, shapes, emitted)?;
            } else {
                self.quarantine_broadcast_operands(child, reduce_extent, shapes, emitted)?;
            }
        }
        Ok(())
    }
}

/// Appends one ready [`BoundOp`] to a [`ReadyBatch`], turning an overflow
/// into a [`TensorError`] instead of a panic -- the multi-position qwen35
/// mixer at real dims (M=13/16) did observe one at the old capacity of 3;
/// see [`READY_BATCH_CAPACITY`]'s own doc.
pub(super) fn push_ready(emitted: &mut ReadyBatch, node: NodeId, op: BoundOp) -> Result<(), TensorError> {
    emitted.try_push(op).map_err(|_| TensorError::NotLowerable {
        node,
        reason: "one push readied more BoundOps than the no-alloc batch capacity allows",
    })
}

/// `In = (Op, Shapes)` matches [`shape::ShapeTable`]'s own `Pipe::Out`
/// exactly, so `AndThen::new(ShapeTable, BoundOpBuilder)` (or
/// `shapes_instance.and_then(builder_instance)`) composes with no adapter:
/// shape resolution's snapshot travels alongside the `Op` it was resolved
/// for, and [`BoundOpBuilder::push`] reads both straight out of `Self::In`.
impl Pipe for BoundOpBuilder {
    type In = (Op, Shapes);
    type Out = ReadyBatch;
    type Err = TensorError;

    fn call(
        &self,
        (expr, shapes): Self::In,
    ) -> impl Future<Output = Result<ReadyBatch, TensorError>> {
        async move { self.push(&expr, &shapes) }
    }
}

pub(super) fn build_elementwise_op(
    node: NodeId,
    shapes: &Shapes,
    held: &RefCell<BTreeMap<NodeId, HeldElementwise>>,
    dtype: DType,
    body: ScalarOp,
    operands: &[(NodeId, IndexMap)],
    constants: Constants<'_>,
) -> BoundOp {
    let extents = shapes.of(node).to_vec();
    let (composed_body, built_operands) = compose(shapes, held, body, operands, constants);
    BoundOp {
        node,
        dtype,
        extents,
        kind: BoundOpKind::Elementwise {
            body: composed_body,
            operands: built_operands,
        },
    }
}

pub(super) fn preview_elementwise_buffer_count(
    node: NodeId,
    shapes: &Shapes,
    held: &BTreeMap<NodeId, HeldElementwise>,
    ones: &[bool],
    values: &[Option<f32>],
    numeric_policy: NumericPolicy,
) -> Result<usize, TensorError> {
    let held = RefCell::new(held.clone());
    let entry = held
        .borrow()
        .get(&node)
        .cloned()
        .ok_or(TensorError::NotLowerable {
            node,
            reason: "elementwise ABI preview requires a held node",
        })?;
    let (_, operands) = compose(
        shapes,
        &held,
        entry.body,
        &entry.operands,
        Constants {
            ones,
            values,
            numeric_policy,
        },
    );
    Ok(metal_buffer_binding_count(
        &operands,
        &alloc::vec::Vec::new(),
    ))
}

/// One operand's [`Layout`] (and, for a gather, its [`Lookup`]), built
/// directly from its [`IndexMap`] — the one place that decides how an
/// `Affine` vs a `Computed` map turns into what an executor reads.
pub(super) fn build_operand(
    node: NodeId,
    map: &IndexMap,
    shapes: &Shapes,
) -> (NodeId, Layout, Option<Lookup>) {
    match map {
        IndexMap::Affine(pattern) => (node, layout_of(pattern, shapes.of(node)), None),
        IndexMap::Computed {
            indices,
            index_map,
            base,
            gathered_dim,
        } => {
            let operand_shape = shapes.of(node);
            let layout = layout_of(base, operand_shape);
            let index_layout = layout_of(index_map, shapes.of(*indices));
            let element_stride = row_major_strides(operand_shape)[*gathered_dim as usize];
            let extent = operand_shape[*gathered_dim as usize];
            let lookup = Lookup {
                indices: *indices,
                index_layout,
                element_stride,
                extent,
            };
            (node, layout, Some(lookup))
        }
    }
}

pub(super) fn build_reduce_op(
    node: NodeId,
    reduce: &Reduce,
    shapes: &Shapes,
    element_body: ComposedBody,
    operands: BoundOperands,
) -> Result<BoundOp, TensorError> {
    let out_pattern = reduce.out_map.affine();
    let output_axes = pure_projection_axes(out_pattern);
    let (out_layout, out_scatter) = match &reduce.out_map {
        IndexMap::Affine(pattern) => (layout_of(pattern, shapes.of(node)), None),
        IndexMap::Computed {
            indices,
            index_map,
            base,
            gathered_dim,
        } => {
            let output_shape = shapes.of(node);
            let out_layout = build_scatter_out_layout(base, *gathered_dim, output_shape);
            let index_layout = layout_of(index_map, shapes.of(*indices));
            let element_stride = row_major_strides(output_shape)[*gathered_dim as usize];
            let extent = output_shape[*gathered_dim as usize];
            let lookup = Lookup {
                indices: *indices,
                index_layout,
                element_stride,
                extent,
            };
            (out_layout, Some(lookup))
        }
    };
    Ok(BoundOp {
        node,
        dtype: reduce.dtype,
        extents: shape::fold_iteration_extents(node, reduce, shapes)?,
        kind: BoundOpKind::Reduce {
            element_body,
            reduce_op: reduce.body,
            init: reduce.init,
            keep: reduce.keep,
            operands,
            output_axes,
            out_layout,
            out_scatter,
            // no epilogue at push time — `reduce_epilogue_fusion` (the
            // `reduce-epilogue-fusion` post-pass) is the only writer of a
            // non-default value for either field, and it runs after every
            // `BoundOp` here already exists.
            epilogue_body: ComposedBody::leaf(ScalarOp::Identity),
            epilogue_operands: Vec::new(),
            epilogue_broadcast_axes: SmallVec::new(),
        },
    })
}

/// [`layout_of`]'s counterpart for a scatter `out_map`'s `base` pattern:
/// identical walk, except `gathered_dim`'s own axis is skipped rather than
/// folded in. `layout_of` reads every axis's `offset` as a real address
/// contribution (`base += offset * element_stride`), which is exactly right
/// for a gather's `base` (that entry is `terms: [], offset: 0` there by
/// convention, so it contributes nothing) but would be wrong here: a
/// scatter's `base` repurposes that same slot's `offset` to carry the
/// destination's static extent (`map.rs`'s `IndexMap::Computed` doc), not an
/// address. Skipping the axis entirely is correct either way, since the
/// scattered axis's real address only ever comes from the fetched index at
/// evaluation time — see [`BoundOpKind::Reduce`]'s `out_scatter` field doc.
pub(super) fn build_scatter_out_layout(
    base: &IndexPattern,
    gathered_dim: u16,
    output_shape: &[u64],
) -> Layout {
    let element_strides = row_major_strides(output_shape);
    let mut strides = SmallVec::<[i64; MAX_INLINE_RANK]>::from_elem(0, base.iter_rank as usize);
    let mut layout_base = 0i64;
    for (axis_index, axis) in base.axes.iter().enumerate() {
        if axis_index as u16 == gathered_dim {
            continue;
        }
        let stride = element_strides[axis_index];
        layout_base += i64::from(axis.offset) * stride;
        for term in &axis.terms {
            strides[term.axis as usize] += i64::from(term.coeff) * stride;
        }
    }
    Layout {
        base: layout_base,
        strides,
    }
}

pub(super) fn pure_projection_axes(pattern: &IndexPattern) -> SmallVec<[u16; MAX_INLINE_RANK]> {
    pattern
        .axes
        .iter()
        .filter_map(|axis| match axis.terms.as_slice() {
            [term] if term.coeff == 1 => Some(term.axis),
            _ => None,
        })
        .collect()
}

/// ROW 431 (`docs/discipline.md`, supersedes ROW 430): whether `node`'s
/// held body is `Multiply(packed, composed)` -- one operand's own map OR a
/// map in the held body beneath it carrying a genuine multi-term axis (the
/// multi-letter packed-row contraction `wo`'s own reshape idiom,
/// `spec.rs:9017-9031`, uses -- a single-letter packed weight like
/// `wq`/`wk`/`wv` never has one) while the OTHER operand is STILL a held,
/// unmaterialized elementwise chain rather than a plain leaf. The recursive
/// walk matters because Qwen's gate-to-Q projection puts the multi-term
/// reshape inside the weight's own broadcast multiply, not in the outer
/// product's map. Returns that other (activation) node so the caller can
/// force just IT to materialize -- this is exactly the shape
/// `run_reduce_quantized`'s admission contract
/// (`packed_reduce_activation_operand`, `cpu.rs`) requires: a bare
/// two-operand product, weight times ONE already-materialized activation
/// buffer. ROW 430 instead declined to fuse the whole product here, which
/// materialized the OUTPUT-width `[u, g, d, o]` buffer; forcing only the
/// activation side (`[u, g, d]`) keeps `W * a` and the reduction fused and
/// never introduces the output axis into an intermediate buffer at all.
pub(super) fn composed_packed_product_activation(
    held: &RefCell<BTreeMap<NodeId, HeldElementwise>>,
    packed_mapping_subtree: &RefCell<Vec<bool>>,
    node: NodeId,
) -> Option<NodeId> {
    let (_, other_node) = packed_and_other_operand(held, packed_mapping_subtree, node)?;
    held.borrow().contains_key(&other_node).then_some(other_node)
}

/// [`composed_packed_product_activation`]'s own packed/other split, without
/// that function's extra "the other operand is still a held elementwise"
/// requirement -- [`BoundOpBuilder::quarantine_broadcast_operands`] needs the
/// PACKED side (never the other side, and regardless of whether the other
/// side is a plain leaf or its own held chain) so it can exempt exactly the
/// operand [`crate::cpu::run_reduce_quantized`]'s fast path is designed to
/// broadcast, at every recursion depth -- not only when
/// `composed_packed_product_activation`'s stricter contract also happens to
/// hold. Same one-of-two-operands-is-packed test as that function; kept as
/// the one shared derivation so the two callers cannot drift on which side
/// counts as "packed".
pub(super) fn packed_and_other_operand(
    held: &RefCell<BTreeMap<NodeId, HeldElementwise>>,
    packed_mapping_subtree: &RefCell<Vec<bool>>,
    node: NodeId,
) -> Option<(NodeId, NodeId)> {
    let (body, operands) = held
        .borrow()
        .get(&node)
        .map(|entry| (entry.body, entry.operands.clone()))?;
    if body != ScalarOp::Multiply {
        return None;
    }
    let [(first_node, first_map), (second_node, second_map)] = operands.as_slice() else {
        return None;
    };
    // a computed (data-dependent) map is ALSO the packed operand's own
    // signal, not just a multi-term affine reshape: `grouped_gathered_expert_product`
    // (`spec.rs`) gathers the packed weight stack with a single-term
    // `IndexMap::Computed` axis per operand axis, so the multi-term check
    // alone (tuned for `wo`'s reshape idiom) never fires for it and this
    // fusion silently declined, materializing the full `[sequence, selected,
    // d_in, d_out]` product instead (`docs/discipline.md` ROW 538).
    let first_packed = packed_mapping_in_held_tree(held, packed_mapping_subtree, *first_node)
        || first_map.is_data_dependent()
        || first_map
            .affine()
            .axes
            .iter()
            .any(|axis| axis.terms.len() > 1);
    let second_packed = packed_mapping_in_held_tree(held, packed_mapping_subtree, *second_node)
        || second_map.is_data_dependent()
        || second_map
            .affine()
            .axes
            .iter()
            .any(|axis| axis.terms.len() > 1);
    if first_packed == second_packed {
        return None;
    }
    Some(if first_packed {
        (*first_node, *second_node)
    } else {
        (*second_node, *first_node)
    })
}

pub(super) fn packed_mapping_in_held_tree(
    held: &RefCell<BTreeMap<NodeId, HeldElementwise>>,
    packed_mapping_subtree: &RefCell<Vec<bool>>,
    root: NodeId,
) -> bool {
    if packed_mapping_subtree
        .borrow()
        .get(root.0 as usize)
        .copied()
        .unwrap_or(false)
    {
        return true;
    }
    let mut pending = vec![root];
    let mut visited = BTreeSet::new();
    while let Some(node) = pending.pop() {
        if !visited.insert(node) {
            continue;
        }
        let Some(entry) = held.borrow().get(&node).cloned() else {
            continue;
        };
        if entry
            .operands
            .iter()
            .any(|(_, map)| map.affine().axes.iter().any(|axis| axis.terms.len() > 1))
        {
            return true;
        }
        pending.extend(entry.operands.iter().map(|(operand, _)| *operand));
    }
    false
}

/// A fusion can compose through: every axis a plain, unshifted projection.
/// Anything richer (a window, a slice, a stride) still resolves correctly,
/// it just materializes its operand instead of composing through it.
pub(super) fn is_identity_projection(map: &IndexMap) -> bool {
    if map.is_data_dependent() {
        return false;
    }
    map.affine()
        .axes
        .iter()
        .all(|axis| axis.offset == 0 && matches!(axis.terms.as_slice(), [term] if term.coeff == 1))
}

/// The three accumulators every `compose_*` call threads through its
/// recursion — a step list, the flat operand list an executor reads from,
/// and which held nodes this pass consumed — bundled for the same reason
/// [`WindowSpec`] bundles a parameter group: one field group traveling
/// together, not `clippy::too_many_arguments` positional soup.
pub(super) struct ComposeState<'a> {
    pub(super) steps: &'a mut Vec<BodyStep>,
    pub(super) operands: &'a mut BoundOperands,
    pub(super) absorbed: &'a mut Vec<NodeId>,
}

/// [`BoundOpBuilder`]'s own `ones`/`constant_value` per-node tables, bundled
/// for the same reason [`ComposeState`] bundles its own three fields: every
/// `compose_*` call threads both together (never one without the other), so
/// a bare pair of `&[bool]`/`&[Option<f32>]` parameters is exactly the
/// positional soup `clippy::too_many_arguments` flags at
/// [`build_elementwise_op`]'s own call depth.
#[derive(Clone, Copy)]
pub(super) struct Constants<'a> {
    pub(super) ones: &'a [bool],
    pub(super) values: &'a [Option<f32>],
    /// Threaded through to [`push_canonical_step`] — see
    /// [`BoundOpBuilder`]'s own field of the same name.
    pub(super) numeric_policy: NumericPolicy,
}

/// Composes the single still-held node `node` — reached from its consumer
/// through `map` — into a [`ComposedBody`] plus the flat, fully-addressed
/// operand list an executor reads from: the reduce-fusion entry point.
/// `node` is guaranteed present in `held` by every caller's own `fuses`
/// check, so this always absorbs at least one op; [`compose_operand`]
/// recurses through however many more are held beneath it.
///
/// [`compose_operand`]'s own ×1.0 elimination can, at THIS call depth only,
/// return a bare [`StepArg::Operand`] with nothing pushed to `steps` —
/// `node` resolved directly to `Multiply(real, Constant(1.0))` with no
/// further absorbing consumer above it (`MaxPool`/`AveragePool`'s own
/// `windowed` node passed straight to `build_reduce`, unlike `Conv`'s
/// `product = windowed * weight`, which always contributes its own step).
/// [`apply_body`](crate::cpu)'s `step_values[body.steps.len() - 1]` requires
/// at least one step, the same invariant the no-fusion branch already
/// guarantees via `ComposedBody::leaf(ScalarOp::Identity)` — so an empty
/// `steps` here gets exactly that: one trailing `Identity` step wrapping
/// the collapsed arg, restoring the invariant without re-introducing the
/// eliminated multiply.
pub(super) fn compose_fused_operands(
    shapes: &Shapes,
    held: &RefCell<BTreeMap<NodeId, HeldElementwise>>,
    node: NodeId,
    map: &IndexMap,
    constants: Constants<'_>,
) -> (ComposedBody, BoundOperands) {
    let mut steps = Vec::new();
    let mut operands = Vec::new();
    let mut absorbed = Vec::new();
    let mut state = ComposeState {
        steps: &mut steps,
        operands: &mut operands,
        absorbed: &mut absorbed,
    };
    let arg = compose_operand(shapes, held, &mut state, node, map, constants);
    if state.steps.is_empty() {
        push_canonical_step(&mut state, ScalarOp::Identity, alloc::vec![arg], constants);
    }
    drop_absorbed(held, absorbed);
    (ComposedBody { steps }, operands)
}

/// Composes an explicit `body` applied over `operands` — the
/// materialize-a-chain entry point [`build_elementwise_op`] uses, where the
/// top body and its immediate operand list are already in hand (the node
/// itself has already been removed from `held` by its caller).
///
/// Reached not only for a genuinely unfused node but also for one
/// [`quarantine_broadcast_operands`] forced standalone — a held node whose
/// own extent is smaller than its consumer's reduce extent gets materialized
/// here specifically so its body is computed once rather than re-read per
/// broadcast repetition. `window_materialize`'s `windowed` (`image * stamp`)
/// is exactly this shape for `Conv` (its `[n,c,oh,ow,kh,kw]` extent excludes
/// the `co` broadcast axis `product`'s own reduce walks), so this entry
/// point needs the same ×1.0 elimination [`compose_operand`] applies —
/// without it, `windowed` would still materialize with the stamp multiply
/// baked in, `compose_operand`'s own elimination never getting a chance to
/// run because [`held`] no longer holds `windowed` by the time the reduce
/// tries to fuse it (`proxima-tensor/docs/discipline.md` ROW 147).
pub(super) fn compose(
    shapes: &Shapes,
    held: &RefCell<BTreeMap<NodeId, HeldElementwise>>,
    body: ScalarOp,
    operands: &[(NodeId, IndexMap)],
    constants: Constants<'_>,
) -> (ComposedBody, BoundOperands) {
    let mut steps = Vec::new();
    let mut resolved_operands = Vec::new();
    let mut absorbed = Vec::new();
    let mut state = ComposeState {
        steps: &mut steps,
        operands: &mut resolved_operands,
        absorbed: &mut absorbed,
    };
    let arg = match eliminate_identity_multiply(body, operands, constants.ones) {
        Some((survivor_node, survivor_map)) => compose_operand(
            shapes,
            held,
            &mut state,
            survivor_node,
            survivor_map,
            constants,
        ),
        None => compose_body(shapes, held, &mut state, body, operands, constants),
    };
    if state.steps.is_empty() {
        push_canonical_step(&mut state, ScalarOp::Identity, alloc::vec![arg], constants);
    }
    drop_absorbed(held, absorbed);
    (ComposedBody { steps }, resolved_operands)
}

pub(super) fn drop_absorbed(held: &RefCell<BTreeMap<NodeId, HeldElementwise>>, absorbed: Vec<NodeId>) {
    let mut held_mut = held.borrow_mut();
    for node in absorbed {
        held_mut.remove(&node);
    }
}

/// A held `Multiply` whose two operands are one real operand and one
/// [`Op::Constant`] of value exactly `1.0` is algebraically a no-op —
/// `x * 1.0 == x` for every finite/inf/nan `f32` bar signaling-NaN quieting
/// (irrelevant to any real weight/activation this crate binds). Returns the
/// surviving operand when this pattern applies, `None` otherwise — the one
/// check both [`compose`] (a node materializing standalone, e.g. under
/// [`quarantine_broadcast_operands`]) and [`compose_operand`] (a node still
/// fusing into its consumer) run before ever pushing a [`BodyStep`], so the
/// constant is dropped from the body regardless of which path reaches it.
/// `window_materialize`'s all-ones shape-inference stamp
/// (`proxima-onnx/src/lower.rs`) is the motivating case, but this is
/// unconditional on which lowering produced the constant, so any future ×1
/// marker gets the same treatment (`proxima-tensor/docs/discipline.md`
/// ROW 147).
pub(super) fn eliminate_identity_multiply<'a>(
    body: ScalarOp,
    operands: &'a [(NodeId, IndexMap)],
    ones: &[bool],
) -> Option<(NodeId, &'a IndexMap)> {
    if body != ScalarOp::Multiply {
        return None;
    }
    let [(left_node, left_map), (right_node, right_map)] = operands else {
        return None;
    };
    let left_is_one = ones.get(left_node.0 as usize).copied().unwrap_or(false);
    let right_is_one = ones.get(right_node.0 as usize).copied().unwrap_or(false);
    if left_is_one && !right_is_one {
        return Some((*right_node, right_map));
    }
    if right_is_one {
        return Some((*left_node, left_map));
    }
    None
}

/// Reads one held node's `(body, operands)` by value — the same
/// clone-out-of-the-`RefCell` shape [`compose_operand`]'s own `entry` uses,
/// needed here because [`eliminate_masked_window_reduce`] walks three levels
/// of `held` chain while [`held`] itself may need a `borrow_mut` later (to
/// drop the matched nodes), and an outstanding `Ref` would collide with that.
pub(super) fn held_snapshot(
    held: &RefCell<BTreeMap<NodeId, HeldElementwise>>,
    node: NodeId,
) -> Option<(ScalarOp, Vec<(NodeId, IndexMap)>)> {
    held.borrow()
        .get(&node)
        .map(|entry| (entry.body, entry.operands.clone()))
}

/// One axis is a plain, unshifted single-term projection — the per-axis
/// version of [`is_identity_projection`], needed here because
/// [`eliminate_masked_window_reduce`] inspects individual [`AxisIndex`]
/// entries (a mask's own three selected axes, a source axis) rather than a
/// whole [`IndexMap`] at once.
pub(super) fn is_pure_axis(axis: &AxisIndex) -> bool {
    axis.offset == 0 && matches!(axis.terms.as_slice(), [term] if term.coeff == 1)
}

/// The stride literal and the three extents [`window_mask`]
/// (`proxima-autograd/src/conv.rs:201`) needs to have built `node`, read back
/// out of the [`BoundOpBuilder`]'s own side channels rather than the source
/// program (this module never holds the whole program, only what is still
/// `held`).
pub(super) struct WindowMatch {
    pub(super) stride: u64,
    pub(super) out_extent: u64,
    pub(super) kernel_extent: u64,
    pub(super) source_extent: u64,
    pub(super) combined_node: NodeId,
    pub(super) scaled_out_node: NodeId,
}

/// Confirms `node` is exactly `window_mask`'s `Equal(Iota, Add(Multiply(Iota,
/// Constant), Iota))` chain (`proxima-autograd/src/conv.rs:201-223`) and, if
/// so, extracts the stride and the three axis extents the in-bounds proof
/// needs. Any structural mismatch — a different `ScalarOp`, a non-`Iota`
/// operand where one is required, a non-pure-projection edge map, a missing
/// constant — returns `None`, leaving `held` untouched (the caller's own
/// contract).
pub(super) fn window_mask_match(
    held: &RefCell<BTreeMap<NodeId, HeldElementwise>>,
    node: NodeId,
    is_iota: &[bool],
    constant_value: &[Option<f32>],
    shapes: &Shapes,
) -> Option<WindowMatch> {
    let is_iota_node =
        |candidate: NodeId| is_iota.get(candidate.0 as usize).copied().unwrap_or(false);

    let (equal_body, equal_operands) = held_snapshot(held, node)?;
    if equal_body != ScalarOp::Equal {
        return None;
    }
    let [(source_iota, source_map), (combined_node, combined_map)] = equal_operands.as_slice()
    else {
        return None;
    };
    if !is_iota_node(*source_iota)
        || !is_identity_projection(source_map)
        || !is_identity_projection(combined_map)
    {
        return None;
    }

    let (add_body, add_operands) = held_snapshot(held, *combined_node)?;
    if add_body != ScalarOp::Add {
        return None;
    }
    let [(scaled_out_node, scaled_map), (kernel_iota, kernel_map)] = add_operands.as_slice() else {
        return None;
    };
    if !is_iota_node(*kernel_iota)
        || !is_identity_projection(scaled_map)
        || !is_identity_projection(kernel_map)
    {
        return None;
    }

    let (mul_body, mul_operands) = held_snapshot(held, *scaled_out_node)?;
    if mul_body != ScalarOp::Multiply {
        return None;
    }
    let [(out_iota, out_map), (stride_node, stride_map)] = mul_operands.as_slice() else {
        return None;
    };
    if !is_iota_node(*out_iota)
        || !is_identity_projection(out_map)
        || !is_identity_projection(stride_map)
    {
        return None;
    }
    // `f32::fract`/`round` need `std`'s libm; this crate's alloc tier does
    // not carry a libm dependency, so an exact round-trip through the
    // integer this node's own `stride as f32` construction produced is the
    // no_std-clean way to confirm `stride_value` is a nonnegative integer.
    let stride_value = constant_value
        .get(stride_node.0 as usize)
        .copied()
        .flatten()?;
    if stride_value < 0.0 {
        return None;
    }
    let stride_u64 = stride_value as u64;
    if stride_u64 as f32 != stride_value {
        return None;
    }

    Some(WindowMatch {
        stride: stride_u64,
        out_extent: *shapes.of(*out_iota).first()?,
        kernel_extent: *shapes.of(*kernel_iota).first()?,
        source_extent: *shapes.of(*source_iota).first()?,
        combined_node: *combined_node,
        scaled_out_node: *scaled_out_node,
    })
}

/// The class fix (`proxima-tensor/docs/discipline.md` ROW 147's ×1.0
/// precedent, generalized from a scalar marker to a shaped one): a
/// `Reduce(Add)` whose held operand is `Multiply(source, mask)`, where `mask`
/// is exactly [`window_mask_match`]'s `Equal`/`Iota` chain, is algebraically a
/// plain window read of `source` — for every `(out_position, kernel_position)`
/// pair, at most one `source_position` ever satisfies `source_position ==
/// out_position*stride + kernel_position`, so summing `source *
/// (source_position == that)` over `source_position` is just `source` indexed
/// at that position, proved in-bounds so the read never needs a fallback
/// branch. On a match, the whole `Multiply`/`Equal`/`Add`/`Multiply` subtree
/// is dropped from `held` (it would otherwise still flush as dead work at
/// [`BoundOpBuilder::finish`]) and the caller substitutes a single [`Op::Reduce`]
/// step, computed once per element instead of once per window position, for
/// the [`Op::Reduce`] it never fuses.
///
/// Any mismatch, or a failed in-bounds proof, returns `None` — the caller's
/// existing fuse-or-materialize path runs unchanged, exactly as if this
/// function did not exist.
pub(super) fn eliminate_masked_window_reduce(
    reduce: &Reduce,
    held: &RefCell<BTreeMap<NodeId, HeldElementwise>>,
    is_iota: &[bool],
    constant_value: &[Option<f32>],
    shapes: &Shapes,
) -> Option<(NodeId, IndexMap)> {
    if reduce.body != ScalarOp::Add
        || reduce.init != ReduceInit::Zero
        || reduce.keep != Keep::Reduce
    {
        return None;
    }
    if !is_identity_projection(&reduce.in_map) {
        return None;
    }

    let (masked_body, masked_operands) = held_snapshot(held, reduce.operand)?;
    if masked_body != ScalarOp::Multiply {
        return None;
    }
    let [(first_node, first_map), (second_node, second_map)] = masked_operands.as_slice() else {
        return None;
    };

    let (source_node, source_map, mask_node, mask_map, window) = if let Some(window) =
        window_mask_match(held, *second_node, is_iota, constant_value, shapes)
    {
        (
            *first_node,
            first_map.clone(),
            *second_node,
            second_map.clone(),
            window,
        )
    } else {
        let window = window_mask_match(held, *first_node, is_iota, constant_value, shapes)?;
        (
            *second_node,
            second_map.clone(),
            *first_node,
            first_map.clone(),
            window,
        )
    };

    if !is_identity_projection(&source_map) {
        return None;
    }
    let mask_pattern = mask_map.affine();
    let [windowed_axis, out_axis, kernel_axis] = mask_pattern.axes.as_slice() else {
        return None;
    };
    if !is_pure_axis(windowed_axis) || !is_pure_axis(out_axis) || !is_pure_axis(kernel_axis) {
        return None;
    }
    let windowed_axis = windowed_axis.terms[0].axis;
    let out_axis = out_axis.terms[0].axis;
    let kernel_axis = kernel_axis.terms[0].axis;

    let last_out = window.out_extent.checked_sub(1)?;
    let last_kernel = window.kernel_extent.checked_sub(1)?;
    let last_read = window
        .stride
        .checked_mul(last_out)?
        .checked_add(last_kernel)?;
    if last_read >= window.source_extent {
        return None;
    }
    let stride_coeff = i32::try_from(window.stride).ok()?;

    let out_pattern = reduce.out_map.affine();
    let keep_axes = pure_projection_axes(out_pattern);
    if keep_axes.len() != out_pattern.axes.len() || keep_axes.contains(&windowed_axis) {
        return None;
    }
    let new_out_position = keep_axes.iter().position(|&axis| axis == out_axis)? as u16;
    let new_kernel_position = keep_axes.iter().position(|&axis| axis == kernel_axis)? as u16;

    let mut new_axes: Vec<AxisIndex> = Vec::with_capacity(source_map.affine().axes.len());
    for axis in &source_map.affine().axes {
        if !is_pure_axis(axis) {
            return None;
        }
        let widened = axis.terms[0].axis;
        let new_axis = if widened == windowed_axis {
            AxisIndex {
                terms: [
                    AxisTerm::scaled(new_out_position, stride_coeff),
                    AxisTerm::scaled(new_kernel_position, 1),
                ]
                .into_iter()
                .collect(),
                offset: 0,
                len: None,
            }
        } else {
            let position = keep_axes.iter().position(|&kept| kept == widened)? as u16;
            AxisIndex {
                terms: core::iter::once(AxisTerm::projection(position)).collect(),
                offset: 0,
                len: None,
            }
        };
        new_axes.push(new_axis);
    }

    held.borrow_mut().remove(&reduce.operand);
    held.borrow_mut().remove(&mask_node);
    held.borrow_mut().remove(&window.combined_node);
    held.borrow_mut().remove(&window.scaled_out_node);

    Some((
        source_node,
        IndexMap::Affine(IndexPattern {
            iter_rank: keep_axes.len() as u16,
            axes: new_axes,
        }),
    ))
}

/// The scalar identity element for `op`'s own [`ScalarOp::is_associative`]
/// class that is bit-exact for EVERY `f32`, including NaN and signed zero —
/// `x * 1.0 == x` always, per IEEE 754 multiplication-by-one. Always
/// admitted regardless of policy
/// ([`NumericRewrite::IdentityElimination`] needs no permission). `Add`,
/// `Maximum`, and `Minimum` also have an algebraic identity element but are
/// NOT bit-exact on every input — see [`identity_element_signed_zero_nan`].
pub(super) const fn identity_element_bitexact(op: ScalarOp) -> Option<f32> {
    match op {
        ScalarOp::Multiply => Some(1.0),
        _ => None,
    }
}

/// The scalar identity element for `Add`/`Maximum`/`Minimum` — `x + 0.0`,
/// `max(x, -inf)`, `min(x, +inf)` all equal `x` for every FINITE `x`, but not
/// for every `f32`: `max(NaN, -inf)` evaluates to `-inf` ([`f32::max`]'s own
/// "if one argument is NaN, return the other" rule), while eliminating the
/// op would return the survivor, `NaN`; `(-0.0) + 0.0` evaluates to `+0.0`,
/// while eliminating the op would return `-0.0`. `Add` is classified
/// [`NumericRewrite::IdentityEliminationSignedZero`] (needs `signed_zero`
/// alone); `Maximum`/`Minimum` are classified
/// [`NumericRewrite::IdentityEliminationNanAssumption`] (needs
/// `nan_assumptions` alone) — [`identity_element_signed_zero_nan_rewrite`]
/// names which one a given op requires. [`push_canonical_step`] checks this
/// only after [`identity_element_bitexact`] misses, and only fires it once
/// [`admit`] clears the caller's [`NumericPolicy`] for that specific rewrite.
pub(super) const fn identity_element_signed_zero_nan(op: ScalarOp) -> Option<f32> {
    match op {
        ScalarOp::Add => Some(0.0),
        ScalarOp::Maximum => Some(f32::NEG_INFINITY),
        ScalarOp::Minimum => Some(f32::INFINITY),
        _ => None,
    }
}

/// The specific permission [`identity_element_signed_zero_nan`]'s
/// elimination needs for `op` — `Add`'s zero-literal case needs
/// `signed_zero` alone, `Maximum`/`Minimum`'s infinity-literal case needs
/// `nan_assumptions` alone. The two are independent permissions (a caller
/// may grant one without the other), so this is never a single shared
/// rewrite classification.
pub(super) const fn identity_element_signed_zero_nan_rewrite(op: ScalarOp) -> Option<NumericRewrite> {
    match op {
        ScalarOp::Add => Some(NumericRewrite::IdentityEliminationSignedZero),
        ScalarOp::Maximum | ScalarOp::Minimum => {
            Some(NumericRewrite::IdentityEliminationNanAssumption)
        }
        _ => None,
    }
}

/// The operand-order sort key [`push_canonical_step`] applies to a
/// commutative op's two args: a fused predecessor ([`StepArg::Step`]) always
/// sorts before a raw operand ([`StepArg::Operand`]), then by index —
/// `false < true` puts every `Step` ahead of every `Operand`. This is what
/// makes `a*b+c` and `c+a*b` mint the identical [`BodyStep`]: whichever
/// operand order the source authored, [`compose_operand`] has already turned
/// each into a `StepArg`, and this key sees only that shape, never the
/// original authoring order.
pub(super) fn step_arg_sort_key(arg: &StepArg) -> (bool, u16) {
    match *arg {
        StepArg::Step(index) => (false, index),
        StepArg::Operand(index) => (true, index),
    }
}

/// The literal value `arg` resolves to, if it is a [`StepArg::Operand`]
/// built from an [`crate::op::Op::Constant`] leaf — `state.operands[index].0`
/// is that operand's source [`NodeId`] ([`build_operand`]'s own first
/// field), and `constant_value[node.0]` is `Some` exactly when
/// [`BoundOpBuilder::push`] saw that node as a constant. A [`StepArg::Step`]
/// is a computed value, never a known literal at this point in composition,
/// so it always returns `None` here (no recursive constant-folding of a
/// step's own body in this slice).
pub(super) fn step_arg_constant(
    arg: StepArg,
    state: &ComposeState<'_>,
    constant_value: &[Option<f32>],
) -> Option<f32> {
    match arg {
        StepArg::Operand(index) => {
            let (node, _, _) = state.operands.get(index as usize)?;
            constant_value.get(node.0 as usize).copied().flatten()
        }
        StepArg::Step(_) => None,
    }
}

/// The one place a [`BodyStep`] enters a [`ComposedBody`] — replaces the raw
/// `state.steps.push(BodyStep { .. })` call this module used to make
/// directly. Canonicalizes a commutative binary op's operand order
/// ([`step_arg_sort_key`]) and eliminates an operand equal to `op`'s own
/// identity element before ever minting a step, so two authored orderings
/// of the same algebraic expression — `a*b+c` and `c+a*b`, or a chain with an
/// identity multiply/add folded away by an earlier rewrite — produce the
/// identical [`StepArg`], never a step whose recognizability depends on
/// which rewrite fired first in the same bind call
/// (`proxima-tensor/src/cpu.rs:2570-2626`'s own "hidden=1 confluence gap"
/// doc). Mints no new `ScalarOp`/`Op` variant: every value this returns is
/// either an existing `StepArg` unchanged or a freshly pushed `BodyStep`
/// using `op` exactly as given.
///
/// [`identity_element_bitexact`] (`x*1`) always fires. The remaining three
/// cases (`x+0`, `max(x,-inf)`, `min(x,+inf)`,
/// [`identity_element_signed_zero_nan`]) change bits on NaN/signed-zero
/// inputs and only fire once `constants.numeric_policy` clears the specific
/// rewrite [`identity_element_signed_zero_nan_rewrite`] names via [`admit`]
/// — under the library default ([`NumericPolicy::bit_exact`]) neither ever
/// fires, and a step carrying a `+0`/`max(-inf)`/`min(+inf)` operand
/// survives unreduced.
pub(super) fn push_canonical_step(
    state: &mut ComposeState<'_>,
    op: ScalarOp,
    mut args: Vec<StepArg>,
    constants: Constants<'_>,
) -> StepArg {
    if op.is_associative() && args.len() == 2 {
        args.sort_by_key(step_arg_sort_key);
    }
    if let [first, second] = args.as_slice() {
        if let Some(identity) = identity_element_bitexact(op) {
            if step_arg_constant(*first, state, constants.values) == Some(identity) {
                return *second;
            }
            if step_arg_constant(*second, state, constants.values) == Some(identity) {
                return *first;
            }
        }
        if let Some(identity) = identity_element_signed_zero_nan(op)
            && let Some(rewrite) = identity_element_signed_zero_nan_rewrite(op)
            && admit(constants.numeric_policy, rewrite).is_ok()
        {
            if step_arg_constant(*first, state, constants.values) == Some(identity) {
                return *second;
            }
            if step_arg_constant(*second, state, constants.values) == Some(identity) {
                return *first;
            }
        }
    }
    state.steps.push(BodyStep { op, args });
    StepArg::Step((state.steps.len() - 1) as u16)
}

/// Composes `body` applied over `body_operands` (expressed in the caller's
/// own iteration space), recursively composing each operand through
/// [`compose_operand`] first, then minting the step through
/// [`push_canonical_step`] — so a step this call mints is already in
/// canonical form, never a second pass over `state.steps`.
///
/// For a commutative binary `body`, `body_operands` is sorted by source
/// [`NodeId`] BEFORE recursing, not only after: `state.operands`'s indices
/// are assigned in visitation order, so `c + a*b` and `a*b + c` would
/// otherwise still number `c`/`a`/`b` differently depending on which
/// authored position each was in, even though `push_canonical_step`'s own
/// post-hoc `StepArg` sort puts the resulting args back in the same
/// `Step`-before-`Operand` shape. Sorting the source pairs first is what
/// makes the two authorings mint byte-identical operand slots, not merely
/// an equivalent argument order — the same "reordering a commutative binary
/// op's operands is exact" guarantee [`push_canonical_step`] documents,
/// applied one level earlier, before any operand slot exists to reorder.
pub(super) fn compose_body(
    shapes: &Shapes,
    held: &RefCell<BTreeMap<NodeId, HeldElementwise>>,
    state: &mut ComposeState<'_>,
    body: ScalarOp,
    body_operands: &[(NodeId, IndexMap)],
    constants: Constants<'_>,
) -> StepArg {
    let mut ordered: Vec<&(NodeId, IndexMap)> = body_operands.iter().collect();
    if body.is_associative() && ordered.len() == 2 {
        ordered.sort_by_key(|(node, _)| node.0);
    }
    let args = ordered
        .iter()
        .map(|(node, map)| compose_operand(shapes, held, state, *node, map, constants))
        .collect();
    push_canonical_step(state, body, args, constants)
}

/// Composes one operand reference `(node, map)` into `steps`/`operands`:
/// reads it directly from its own buffer when `node` is not (or is no
/// longer) held, or — when it is still held, meaning it satisfied the
/// fusion condition at the exact position that made this its last use —
/// absorbs its own body as one more [`BodyStep`], recursing through
/// however many further levels are held beneath it. `map`'s axes are
/// remapped through [`remap_sub_operands`] before recursing, since a held
/// node's own operand maps are expressed in *its* iteration space, not the
/// caller's. [`eliminate_identity_multiply`] is checked before ever pushing
/// a step — see that function's own doc.
pub(super) fn compose_operand(
    shapes: &Shapes,
    held: &RefCell<BTreeMap<NodeId, HeldElementwise>>,
    state: &mut ComposeState<'_>,
    node: NodeId,
    map: &IndexMap,
    constants: Constants<'_>,
) -> StepArg {
    let entry = held
        .borrow()
        .get(&node)
        .map(|held_elementwise| (held_elementwise.body, held_elementwise.operands.clone()));

    let Some((body, sub_operands)) = entry else {
        state.operands.push(build_operand(node, map, shapes));
        return StepArg::Operand((state.operands.len() - 1) as u16);
    };

    state.absorbed.push(node);
    let remapped = remap_sub_operands(&sub_operands, map);

    if let Some((survivor_node, survivor_map)) =
        eliminate_identity_multiply(body, &remapped, constants.ones)
    {
        return compose_operand(shapes, held, state, survivor_node, survivor_map, constants);
    }

    compose_body(shapes, held, state, body, &remapped, constants)
}

/// The outer iteration axis each of `map`'s own axes corresponds to — sound
/// only when `map` is [`is_identity_projection`], which every caller here
/// already checked before fusing through it.
pub(super) fn axis_correspondence(map: &IndexMap) -> Vec<u16> {
    map.affine()
        .axes
        .iter()
        .map(|axis| axis.terms[0].axis)
        .collect()
}

pub(super) fn remap_pattern(pattern: &IndexPattern, axis_map: &[u16], outer_iter_rank: u16) -> IndexPattern {
    let axes = pattern
        .axes
        .iter()
        .map(|axis_index| AxisIndex {
            terms: axis_index
                .terms
                .iter()
                .map(|term| AxisTerm {
                    axis: axis_map[term.axis as usize],
                    coeff: term.coeff,
                })
                .collect(),
            offset: axis_index.offset,
            len: axis_index.len,
        })
        .collect();
    IndexPattern {
        iter_rank: outer_iter_rank,
        axes,
    }
}

pub(super) fn remap_index_map(map: &IndexMap, axis_map: &[u16], outer_iter_rank: u16) -> IndexMap {
    match map {
        IndexMap::Affine(pattern) => {
            IndexMap::Affine(remap_pattern(pattern, axis_map, outer_iter_rank))
        }
        IndexMap::Computed {
            indices,
            index_map,
            base,
            gathered_dim,
        } => IndexMap::Computed {
            indices: *indices,
            index_map: remap_pattern(index_map, axis_map, outer_iter_rank),
            base: remap_pattern(base, axis_map, outer_iter_rank),
            gathered_dim: *gathered_dim,
        },
    }
}

/// Composes a held op's own operand maps (expressed in its own iteration
/// space) through `outer_map` — how its consumer reads it, always an
/// identity projection, the fusion precondition — into the consumer's own
/// iteration space. The symbolic counterpart of [`Layout`]-level stride
/// remapping, applied one level per absorbed node so [`compose_operand`]'s
/// recursion composes through as many levels as a chain has.
pub(super) fn remap_sub_operands(
    sub_operands: &[(NodeId, IndexMap)],
    outer_map: &IndexMap,
) -> Vec<(NodeId, IndexMap)> {
    let axis_map = axis_correspondence(outer_map);
    let outer_iter_rank = outer_map.affine().iter_rank;
    sub_operands
        .iter()
        .map(|(node, map)| (*node, remap_index_map(map, &axis_map, outer_iter_rank)))
        .collect()
}

pub(super) fn layout_of(pattern: &IndexPattern, operand_shape: &[u64]) -> Layout {
    let element_strides = row_major_strides(operand_shape);
    let mut strides = SmallVec::<[i64; MAX_INLINE_RANK]>::from_elem(0, pattern.iter_rank as usize);
    let mut base = 0i64;
    for (axis_index, axis) in pattern.axes.iter().enumerate() {
        let stride = element_strides[axis_index];
        base += i64::from(axis.offset) * stride;
        for term in &axis.terms {
            strides[term.axis as usize] += i64::from(term.coeff) * stride;
        }
    }
    Layout { base, strides }
}

pub(super) fn row_major_strides(shape: &[u64]) -> Vec<i64> {
    let mut strides = vec![0i64; shape.len()];
    let mut accumulator = 1i64;
    for (axis_index, extent) in shape.iter().enumerate().rev() {
        strides[axis_index] = accumulator;
        accumulator *= *extent as i64;
    }
    strides
}

