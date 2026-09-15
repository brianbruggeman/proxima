use super::*;

impl<'file> LoadedModel<'file> {
    #[must_use]
    pub fn token_id_for_piece(&self, piece: &str) -> Option<u32> {
        self.vocab.token_id(piece)
    }

    /// `true` when [`Self::load`] built a device-resident, single-range
    /// program for this checkpoint ([`Self::single_range`]'s own doc) --
    /// the ROW 373 evidence hook: a qk-norm (Qwen3) checkpoint returned
    /// `false` here before that row (rejected by
    /// `append_mistral_single_range_cached_layer`, `build_single_range_program`
    /// falling back to `Ok(None)`) and returns `true` after it, WITHOUT this
    /// crate's own decode output changing (`Self::run_decode_loop`'s own
    /// `Some(single_range)` branch is what a Metal decode step then takes).
    #[cfg(all(test, feature = "metal-output-placement", target_os = "macos"))]
    pub(crate) fn takes_placed_kv_path(&self) -> bool {
        self.single_range.is_some()
    }

    /// `general.name` off the checkpoint this call loaded, when the
    /// checkpoint declared one -- a live "what's running" indicator's own
    /// label ([`Self::model_name`]'s field doc). `None` on a safetensors
    /// checkpoint, or a GGUF checkpoint that omits the key.
    #[must_use]
    pub fn model_name(&self) -> Option<&str> {
        self.model_name.as_deref()
    }

    /// This checkpoint's pre-`lm_head` hidden-state root
    /// (`proxima_tensor::spec::ForwardRoots::hidden`), when the load path
    /// named one -- `None` on the qwen35 hybrid path
    /// (`crate::qwen35::qwen35_forward_program` carries no named
    /// hidden-state root yet). A caller composes this with
    /// [`Self::forward_node_values`] to read that tensor's row out
    /// directly; proxima names the node, it does not decide how a caller
    /// pools it (last-token, mean, or otherwise is the caller's own
    /// policy).
    #[must_use]
    pub fn hidden_root(&self) -> Option<NodeId> {
        self.hidden_root
    }

    /// Router-logit roots aligned with the routed layers of this model.
    /// These are the exact nodes already used by the production MoE graph;
    /// an empty slice means the loaded architecture has no exposed
    /// pre-gather roots.
    #[must_use]
    pub fn router_roots(&self) -> &[NodeId] {
        &self.router_roots
    }

    /// Evaluates the bound qwen35moe router roots for one prompt position.
    /// The returned vectors are the graph's actual per-layer router logits in
    /// layer order, so DynaExq can make a residency decision from execution
    /// data rather than from a duplicated host-side router. This diagnostic
    /// does not claim a memory reduction: it uses the ordinary evaluator
    /// until the production per-layer pre-gather path is enabled.
    pub fn qwen35moe_router_logits(
        &self,
        prompt: &str,
        gpu_layers: i32,
    ) -> Result<Vec<Vec<f32>>, InteropError> {
        if self.router_roots.is_empty()
            || self
                .architecture_impl
                .is_none_or(|architecture| architecture.name() != "qwen35moe")
        {
            return Err(InteropError::PreGatherExecutionUnsupported {
                architecture: String::from(
                    self.architecture_impl.map_or("unknown", Architecture::name),
                ),
                reason: String::from("the bound model does not expose qwen35moe router roots"),
            });
        }
        self.forward_node_values_on_backend(prompt, &self.router_roots, gpu_layers)
    }

    /// Builds the graph cuts that surround each qwen35moe routed layer. The
    /// cuts are plan-time data: callers evaluate one producer, apply the
    /// residency transition, then evaluate its consumer with the borrowed
    /// activation handoff. No cut is built for another architecture.
    pub fn qwen35moe_layer_boundaries(
        &self,
        symbols: &[u64],
    ) -> Result<Vec<crate::qwen35moe::execution::LayerProgramBoundary>, InteropError> {
        if self.qwen35moe_layer_diagnostics.is_empty()
            || self
                .architecture_impl
                .is_none_or(|architecture| architecture.name() != "qwen35moe")
        {
            return Err(InteropError::PreGatherExecutionUnsupported {
                architecture: String::from(
                    self.architecture_impl.map_or("unknown", Architecture::name),
                ),
                reason: String::from("the bound model has no qwen35moe layer diagnostics"),
            });
        }
        self.qwen35moe_layer_diagnostics
            .iter()
            .map(|diagnostic| {
                crate::qwen35moe::execution::split_layer_program(
                    &self.program,
                    symbols,
                    diagnostic.router_logits,
                    diagnostic.routed_output,
                )
                .map_err(InteropError::from)
            })
            .collect()
    }

    /// Builds dense router and gather programs for every routed layer in
    /// execution order. Each layer starts from the prior layer's block output,
    /// so callers can evaluate one layer, change resident expert sources, and
    /// continue without retaining a full-program suffix.
    pub fn qwen35moe_layer_segments(
        &self,
        symbols: &[u64],
    ) -> Result<
        Vec<(
            crate::qwen35moe::execution::ProgramSegment,
            crate::qwen35moe::execution::ProgramSegment,
        )>,
        InteropError,
    > {
        if self.qwen35moe_layer_diagnostics.is_empty()
            || self
                .architecture_impl
                .is_none_or(|architecture| architecture.name() != "qwen35moe")
        {
            return Err(InteropError::PreGatherExecutionUnsupported {
                architecture: String::from(
                    self.architecture_impl.map_or("unknown", Architecture::name),
                ),
                reason: String::from("the bound model has no qwen35moe layer diagnostics"),
            });
        }
        let mut segments = Vec::with_capacity(self.qwen35moe_layer_diagnostics.len());
        let mut previous_output = None;
        for diagnostic in &self.qwen35moe_layer_diagnostics {
            let pair = crate::qwen35moe::execution::split_router_and_gather_segments(
                &self.program,
                symbols,
                previous_output,
                diagnostic.router_logits,
                diagnostic.block_output,
            )
            .map_err(InteropError::from)?;
            segments.push(pair);
            previous_output = Some(diagnostic.block_output);
        }
        Ok(segments)
    }

    /// Builds one routed layer's segments on demand. The caller can drop the
    /// pair after the layer gather, keeping graph metadata bounded by one
    /// layer instead of materializing all 40 layer segments at once.
    pub fn qwen35moe_layer_segment(
        &self,
        layer: usize,
        symbols: &[u64],
        previous_layer_output: Option<NodeId>,
    ) -> Result<
        (
            crate::qwen35moe::execution::ProgramSegment,
            crate::qwen35moe::execution::ProgramSegment,
        ),
        InteropError,
    > {
        let diagnostic = self.qwen35moe_layer_diagnostics.get(layer).ok_or_else(|| {
            InteropError::PreGatherExecutionUnsupported {
                architecture: String::from("qwen35moe"),
                reason: String::from("requested routed layer is outside the bound diagnostics"),
            }
        })?;
        crate::qwen35moe::execution::split_router_and_gather_segments(
            &self.program,
            symbols,
            previous_layer_output,
            diagnostic.router_logits,
            diagnostic.block_output,
        )
        .map_err(InteropError::from)
    }

    pub(super) fn qwen35moe_pre_gather_plan(
        &self,
        symbols: &[u64],
        gdn_scan_enabled: bool,
        gdn_backend: GdnPrefillBackend,
        persistent_cuts: bool,
    ) -> Result<Qwen35MoePreGatherPlan, InteropError> {
        let last_layer_output = self
            .qwen35moe_layer_diagnostics
            .last()
            .map(|diagnostic| diagnostic.block_output)
            .ok_or_else(|| InteropError::PreGatherExecutionUnsupported {
                architecture: String::from("qwen35moe"),
                reason: String::from("the bound graph has no routed layer boundary"),
            })?;
        let suffix = crate::qwen35moe::execution::split_mapped_layer_segment(
            &self.program,
            symbols,
            Some(last_layer_output),
            self.logits_root,
        )
        .map_err(InteropError::from)?;

        let mut layer_parts = Vec::with_capacity(self.qwen35moe_layer_diagnostics.len());
        let mut prefix_required_nodes = BTreeSet::new();
        let mut previous_output = None;
        for diagnostic in &self.qwen35moe_layer_diagnostics {
            let (router, gdn_scan) = if let (Some(taps), Some(prefill)) =
                (diagnostic.ssm_taps.clone(), diagnostic.gdn_prefill)
                && gdn_scan_enabled
            {
                let producer = crate::qwen35moe::execution::split_mapped_layer_segment(
                    &self.program,
                    symbols,
                    previous_output,
                    taps.value_sequence,
                )
                .map_err(InteropError::from)?;
                let mut tail_symbols = symbols.to_vec();
                tail_symbols[0] = 1;
                let tail = crate::qwen35moe::execution::split_mapped_layer_segment(
                    &self.program,
                    &tail_symbols,
                    Some(prefill.delta_out_input),
                    prefill.router_logits,
                )
                .map_err(InteropError::from)?;
                if std::env::var_os("PROXIMA_DEBUG_GDN_COMPARE").is_some() {
                    eprintln!(
                        "gdn_tail_partition delta={} gated={} projected={} router={} cuts={:?} inputs={:?}",
                        prefill.delta_out_input.0,
                        prefill.gated_value.0,
                        prefill.projected.0,
                        prefill.router_logits.0,
                        tail.1,
                        tail.0
                            .iter()
                            .filter_map(|operation| operation.name())
                            .collect::<Vec<_>>(),
                    );
                }
                let router = crate::qwen35moe::execution::split_mapped_layer_segment(
                    &self.program,
                    symbols,
                    Some(diagnostic.post_mixer_residual),
                    diagnostic.router_logits,
                )
                .map_err(InteropError::from)?;
                let decode_router = crate::qwen35moe::execution::split_mapped_layer_segment(
                    &self.program,
                    symbols,
                    previous_output,
                    diagnostic.router_logits,
                )
                .map_err(InteropError::from)?;
                (
                    router,
                    Some(Qwen35MoeGdnScanSegment {
                        producer,
                        tail,
                        taps,
                        prefill,
                        post_mixer_residual: diagnostic.post_mixer_residual,
                        post_attention_norm_output: diagnostic.post_attention_norm_output,
                        router_logits: diagnostic.router_logits,
                        decode_router,
                    }),
                )
            } else {
                (
                    crate::qwen35moe::execution::split_mapped_layer_segment(
                        &self.program,
                        symbols,
                        previous_output,
                        diagnostic.router_logits,
                    )
                    .map_err(InteropError::from)?,
                    None,
                )
            };
            let gather = crate::qwen35moe::execution::split_mapped_layer_segment(
                &self.program,
                symbols,
                Some(diagnostic.router_logits),
                diagnostic.block_output,
            )
            .map_err(InteropError::from)?;
            let gather_next_router = self
                .qwen35moe_layer_diagnostics
                .get(layer_parts.len() + 1)
                .filter(|_| gdn_scan.is_none())
                .map(|next| {
                    crate::qwen35moe::execution::split_gather_and_next_router_segment(
                        &self.program,
                        symbols,
                        diagnostic.router_logits,
                        diagnostic.block_output,
                        next.router_logits,
                    )
                })
                .transpose()
                .map_err(InteropError::from)?;
            #[cfg(feature = "qwen35moe-linked-suffix")]
            let gather = if diagnostic.block_output == last_layer_output {
                crate::qwen35moe::execution::split_gather_and_suffix_segment(
                    &self.program,
                    symbols,
                    diagnostic.router_logits,
                    diagnostic.block_output,
                    self.logits_root,
                )
                .map_err(InteropError::from)?
            } else {
                gather
            };
            let entry = gdn_scan.as_ref().map_or(&router, |scan| &scan.producer);
            for (node, _) in &entry.1 {
                if !matches!(
                    self.program[node.0 as usize],
                    proxima_tensor::op::Op::Input { .. }
                ) {
                    prefix_required_nodes.insert(*node);
                }
            }
            // The decode-router fallback (single-position calls after the
            // initial multi-position scan) reads the SAME previous-layer
            // outputs the ordinary non-scan router would -- register its
            // cuts too, or a later single-token step finds them missing.
            if let Some(scan) = &gdn_scan {
                for (node, _) in &scan.decode_router.1 {
                    if !matches!(
                        self.program[node.0 as usize],
                        proxima_tensor::op::Op::Input { .. }
                    ) {
                        prefix_required_nodes.insert(*node);
                    }
                }
            }
            layer_parts.push((router, gather, gather_next_router, gdn_scan));
            previous_output = Some(diagnostic.block_output);
        }

        // layer zero's router is already the complete graph prefix, so a
        // second prefix segment would execute the same operations twice.
        let first_entry_mapping =
            layer_parts
                .first()
                .ok_or_else(|| InteropError::PreGatherExecutionUnsupported {
                    architecture: String::from("qwen35moe"),
                    reason: String::from("the bound graph has no first router segment"),
                })?;
        let first_entry_mapping = first_entry_mapping
            .3
            .as_ref()
            .map_or(&first_entry_mapping.0.2, |scan| &scan.producer.2);
        let prefix_carried_nodes = first_entry_mapping
            .keys()
            .filter(|node| prefix_required_nodes.contains(node))
            .copied()
            .collect();
        let global_cut_nodes = self
            .program
            .iter()
            .enumerate()
            .filter_map(|(index, operation)| {
                operation
                    .name()
                    .is_some_and(|name| name.starts_with("__cut_"))
                    .then_some(NodeId(index as u32))
            })
            .collect();

        let gather_cuts = layer_parts
            .iter()
            .map(|part| part.1.1.clone())
            .collect::<Vec<_>>();
        let mut layers = Vec::with_capacity(layer_parts.len());
        for layer in 0..layer_parts.len() {
            let (router, gather, gather_next_router, gdn_scan) = layer_parts[layer].clone();
            let future_gather_cuts = collect_future_gather_cuts(layer, &gather_cuts);
            let mut future_cuts = layer_parts[layer + 1..]
                .iter()
                .flat_map(|part| {
                    part.3
                        .as_ref()
                        .map_or(&part.0.1, |scan| &scan.producer.1)
                        .iter()
                        .chain(part.1.1.iter())
                })
                .cloned()
                .collect::<Vec<_>>();
            future_cuts.extend(suffix.1.iter().cloned());
            future_cuts.sort_by_key(|(node, _)| *node);
            future_cuts.dedup_by_key(|(node, _)| *node);
            let mut router_future_cuts = gather.1.clone();
            router_future_cuts.extend(future_cuts.iter().cloned());
            router_future_cuts.sort_by_key(|(node, _)| *node);
            router_future_cuts.dedup_by_key(|(node, _)| *node);
            let layer_window = if layer % 2 == 0 {
                self.qwen35moe_layer_diagnostics
                    .get(layer + 1)
                    .filter(|_| gdn_scan.is_none())
                    .map(|next| {
                        crate::qwen35moe::execution::split_two_layer_window_segment(
                            &self.program,
                            symbols,
                            (layer > 0)
                                .then(|| self.qwen35moe_layer_diagnostics[layer - 1].block_output),
                            self.qwen35moe_layer_diagnostics[layer].router_logits,
                            next.router_logits,
                            next.block_output,
                        )
                    })
                    .transpose()
                    .map_err(InteropError::from)?
            } else {
                None
            };
            layers.push(Qwen35MoeLayerSegments {
                router,
                gather,
                gather_next_router,
                layer_window,
                gdn_scan,
                router_future_cuts,
                next_cuts: future_cuts,
                future_gather_cuts,
            });
        }

        for (layer, segments) in layers.iter().enumerate() {
            let scan_programs = segments.gdn_scan.as_ref().into_iter().flat_map(|scan| {
                [
                    ("gdn-scan-producer", &scan.producer.0),
                    ("gdn-scan-tail", &scan.tail.0),
                ]
            });
            for (phase, program) in scan_programs.chain([
                ("router", &segments.router.0),
                ("gather", &segments.gather.0),
            ]) {
                proxima_tensor::shape::infer(program, symbols).map_err(|error| {
                    InteropError::PreGatherExecutionUnsupported {
                        architecture: String::from("qwen35moe"),
                        reason: alloc::format!("layer {layer} {phase} segment is invalid: {error}"),
                    }
                })?;
            }
            if let Some(fused) = &segments.gather_next_router {
                proxima_tensor::shape::infer(&fused.0, symbols).map_err(|error| {
                    InteropError::PreGatherExecutionUnsupported {
                        architecture: String::from("qwen35moe"),
                        reason: alloc::format!(
                            "layer {layer} gather-next-router segment is invalid: {error}"
                        ),
                    }
                })?;
            }
            if let Some(window) = &segments.layer_window {
                proxima_tensor::shape::infer(&window.0, symbols).map_err(|error| {
                    InteropError::PreGatherExecutionUnsupported {
                        architecture: String::from("qwen35moe"),
                        reason: alloc::format!(
                            "layer {layer} two-layer window is invalid: {error}"
                        ),
                    }
                })?;
            }
            if std::env::var_os("PROXIMA_DEBUG_QWEN35_PLAN").is_some() && layer < 3 {
                eprintln!(
                    "qwen35 plan layer={layer} router_cuts={:?} gather_cuts={:?} router_future={:?} map={:?} op493={:?} gather_ops={:?}",
                    segments.router.1,
                    segments.gather.1,
                    segments.router_future_cuts,
                    segments
                        .gather
                        .2
                        .iter()
                        .filter(|(_, mapped)| {
                            let maximum = std::env::var("PROXIMA_DEBUG_QWEN35_PLAN_MAX_MAPPED")
                                .ok()
                                .and_then(|value| value.parse::<u32>().ok())
                                .unwrap_or(24);
                            mapped.0 <= maximum
                        })
                        .collect::<Vec<_>>(),
                    self.program.get(493),
                    segments
                        .gather
                        .0
                        .iter()
                        .enumerate()
                        .map(|(index, operation)| (index, operation.name()))
                        .collect::<Vec<_>>()
                );
            }
        }

        #[cfg(feature = "instrument")]
        debug!(
            symbol_count = symbols.len() as u64,
            layer_count = layers.len() as u64,
            source_program_ops = self.program.len() as u64,
            cached_segment_ops = (layers
                .iter()
                .map(|segments| segments.router.0.len() + segments.gather.0.len())
                .sum::<usize>()
                + suffix.0.len()) as u64,
            "qwen35moe pre-gather partitions built because the concrete shape changed"
        );

        #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
        let router_cut_placements = if persistent_cuts {
            let shapes = proxima_tensor::shape::infer(&self.program, symbols).map_err(|error| {
                InteropError::PreGatherExecutionUnsupported {
                    architecture: String::from("qwen35moe"),
                    reason: alloc::format!("full graph shape inference failed: {error}"),
                }
            })?;
            let mut boundary_nodes = BTreeSet::new();
            for (layer, segments) in layers.iter().enumerate() {
                let diagnostic = &self.qwen35moe_layer_diagnostics[layer];
                boundary_nodes.insert(diagnostic.router_logits);
                boundary_nodes.insert(diagnostic.block_output);
                boundary_nodes.extend(segments.router.1.iter().map(|(node, _)| *node));
                boundary_nodes.extend(segments.gather.1.iter().map(|(node, _)| *node));
                boundary_nodes.extend(segments.router_future_cuts.iter().map(|(node, _)| *node));
                boundary_nodes.extend(segments.next_cuts.iter().map(|(node, _)| *node));
                boundary_nodes.extend(segments.future_gather_cuts.iter().copied());
                if let Some(fused) = &segments.gather_next_router {
                    boundary_nodes.extend(fused.1.iter().map(|(node, _)| *node));
                }
                if let Some(scan) = &segments.gdn_scan {
                    boundary_nodes.extend(scan.producer.1.iter().map(|(node, _)| *node));
                    boundary_nodes.extend(scan.tail.1.iter().map(|(node, _)| *node));
                }
            }
            let mut produced_cuts = BTreeSet::new();
            let mut consumed_cuts = BTreeSet::new();
            for segments in &layers {
                let mut segment_maps = vec![
                    (&segments.router.0, &segments.router.2),
                    (&segments.gather.0, &segments.gather.2),
                ];
                if let Some(fused) = &segments.gather_next_router {
                    segment_maps.push((&fused.0, &fused.2));
                }
                if let Some(scan) = &segments.gdn_scan {
                    segment_maps.push((&scan.producer.0, &scan.producer.2));
                    segment_maps.push((&scan.tail.0, &scan.tail.2));
                }
                for (program, mapping) in segment_maps {
                    for (original, mapped) in mapping {
                        if !boundary_nodes.contains(original) {
                            continue;
                        }
                        match program.get(mapped.0 as usize) {
                            Some(Op::Input { .. }) => {
                                consumed_cuts.insert(*original);
                            }
                            Some(_) => {
                                produced_cuts.insert(*original);
                            }
                            None => {}
                        }
                    }
                }
            }
            let placed_boundary_cuts = produced_cuts
                .intersection(&consumed_cuts)
                .copied()
                .collect::<BTreeSet<_>>();
            if std::env::var_os("PROXIMA_DEBUG_QWEN35_PLAN").is_some() {
                eprintln!(
                    "qwen35 placement candidates produced={} consumed={} placed={:?}",
                    produced_cuts.len(),
                    consumed_cuts.len(),
                    placed_boundary_cuts,
                );
            }
            let mut placements = BTreeMap::new();
            for node in placed_boundary_cuts {
                if matches!(
                    self.program.get(node.0 as usize),
                    Some(Op::Constant { .. } | Op::Input { .. })
                ) {
                    continue;
                }
                if !boundary_nodes.contains(&node) {
                    continue;
                }
                let element_count = shapes
                    .of(node)
                    .iter()
                    .try_fold(1usize, |product, extent| {
                        usize::try_from(*extent)
                            .ok()
                            .and_then(|extent| product.checked_mul(extent))
                    })
                    .ok_or_else(|| InteropError::PreGatherExecutionUnsupported {
                        architecture: String::from("qwen35moe"),
                        reason: alloc::format!(
                            "boundary cut {node:?} shape does not fit a placed buffer"
                        ),
                    })?;
                placements.insert(
                    node,
                    allocate_placed_buffer(
                        element_count
                            .checked_mul(core::mem::size_of::<f32>())
                            .ok_or_else(|| InteropError::PreGatherExecutionUnsupported {
                                architecture: String::from("qwen35moe"),
                                reason: alloc::format!(
                                    "boundary cut {node:?} byte size overflowed"
                                ),
                            })?,
                    )?,
                );
            }
            placements
        } else {
            BTreeMap::new()
        };

        Ok(Qwen35MoePreGatherPlan {
            symbols: symbols.to_vec(),
            gdn_scan_enabled,
            gdn_backend,
            persistent_cuts,
            layers,
            suffix,
            prefix_carried_nodes,
            global_cut_nodes,
            #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
            router_cut_placements,
        })
    }

    pub(super) fn evaluate_qwen35moe_gdn_scan_segment(
        &self,
        runtime: &mut BackendRuntime,
        layer: usize,
        scan: &Qwen35MoeGdnScanSegment,
        context: GdnScanSegmentContext<'_, '_>,
    ) -> Result<(), InteropError> {
        let GdnScanSegmentContext {
            state_cache,
            gdn_backend,
            future_cuts,
            symbols,
            named,
            outputs,
            resident_names,
            carried,
            results,
        } = context;
        #[cfg(not(feature = "mlx-gdn"))]
        let _ = gdn_backend;
        let (program, cuts, mapping) = &scan.producer;
        let mut segment_named: Vec<(&str, QuantizedBlock<'_>)> = named
            .iter()
            .copied()
            .filter(|(name, _)| {
                program
                    .iter()
                    .any(|operation| operation.name() == Some(*name))
            })
            .collect();
        let mut zero_delta = Vec::new();
        if let Some(mapped_delta) = mapping.get(&scan.prefill.delta_out_input).copied()
            && let Some((delta_name, _)) =
                program.iter().enumerate().find_map(|(index, operation)| {
                    (NodeId(index as u32) == mapped_delta).then_some(match operation {
                        Op::Input {
                            name: Some(name), ..
                        } => (name.as_str(), true),
                        _ => ("", false),
                    })
                })
            && !delta_name.is_empty()
        {
            let shapes = proxima_tensor::shape::infer(program, symbols).map_err(|error| {
                InteropError::PreGatherExecutionUnsupported {
                    architecture: String::from("qwen35moe"),
                    reason: alloc::format!("gdn scan producer shape inference failed: {error}"),
                }
            })?;
            let element_count = shapes
                .of(mapped_delta)
                .iter()
                .try_fold(1usize, |product, extent| {
                    product.checked_mul(*extent as usize)
                })
                .ok_or_else(|| InteropError::PreGatherExecutionUnsupported {
                    architecture: String::from("qwen35moe"),
                    reason: String::from("gdn scan delta input shape overflow"),
                })?;
            zero_delta.resize(element_count, 0.0);
            segment_named.push((delta_name, QuantizedBlock::Float32(&zero_delta)));
        }
        for (node, name) in cuts {
            if segment_named
                .iter()
                .any(|(candidate, _)| *candidate == name)
            {
                continue;
            }
            let (_, values) =
                carried
                    .get(node)
                    .ok_or_else(|| InteropError::PreGatherExecutionUnsupported {
                        architecture: String::from("qwen35moe"),
                        reason: alloc::format!(
                            "gdn scan producer missing cut node {node:?} ({name})"
                        ),
                    })?;
            if std::env::var_os("PROXIMA_DEBUG_GDN_CARRY").is_some() {
                eprintln!(
                    "gdn_scan_cut layer={} node={} name={} elements={} first4={:?}",
                    layer,
                    node.0,
                    name,
                    values.len(),
                    values.iter().take(4).copied().collect::<Vec<_>>(),
                );
            }
            segment_named.push((name.as_str(), QuantizedBlock::Float32(values)));
        }

        let taps = scan.taps.clone();
        let scan_inputs = [
            taps.query_sequence,
            taps.key_sequence,
            taps.value_sequence,
            taps.gate_sequence,
            taps.beta_sequence,
        ];
        let mut requested = BTreeMap::new();
        for original in scan
            .tail
            .1
            .iter()
            .map(|(node, _)| *node)
            .chain(scan_inputs)
            .chain(future_cuts.iter().map(|(node, _)| *node))
            .chain(outputs.iter().copied())
        {
            if let Some(mapped) = mapping.get(&original).copied() {
                let is_packed_weight = matches!(
                    &program[mapped.0 as usize],
                    Op::Input {
                        name: Some(name),
                        ..
                    } if name.ends_with(".weight")
                );
                if !is_packed_weight {
                    requested.insert(mapped, original);
                }
            }
        }
        let requested_nodes: Vec<NodeId> = requested.keys().copied().collect();
        // `state_in` is an Op::Input, not a computed output. The evaluator
        // therefore does not place it in `Evaluated`; the authoritative
        // recurrent state is the layer cache owned by the decode loop, not a
        // partition-local input reconstruction.
        let mapped_state =
            mapping
                .get(&taps.state_in)
                .copied()
                .ok_or(InteropError::MissingEvaluatedNode {
                    node: taps.state_in,
                })?;
        let state_shape = proxima_tensor::shape::infer(program, symbols)
            .map_err(|error| InteropError::PreGatherExecutionUnsupported {
                architecture: String::from("qwen35moe"),
                reason: alloc::format!("gdn scan state shape inference failed: {error}"),
            })?
            .of(mapped_state)
            .to_vec();
        let expected_state_elements = state_shape
            .iter()
            .try_fold(1usize, |product, extent| {
                product.checked_mul(*extent as usize)
            })
            .ok_or_else(|| InteropError::PreGatherExecutionUnsupported {
                architecture: String::from("qwen35moe"),
                reason: String::from("gdn scan state shape overflowed usize"),
            })?;
        if state_cache.len() != expected_state_elements {
            return Err(InteropError::PreGatherExecutionUnsupported {
                architecture: String::from("qwen35moe"),
                reason: alloc::format!(
                    "gdn scan layer cache state has {} elements but needs {expected_state_elements}",
                    state_cache.len()
                ),
            });
        }
        let evaluated = runtime.evaluate_segment(
            program,
            symbols,
            &segment_named,
            &requested_nodes,
            resident_names,
            None,
        )?;
        if std::env::var_os("PROXIMA_DEBUG_GDN_EXACT_COMPARE").is_some() {
            let mut exact_scratch = Vec::new();
            let mut exact_validated = None;
            let exact = evaluate_quantized_named_exact_with_scratch_and_experts(
                program,
                symbols,
                &segment_named,
                &requested_nodes,
                &mut exact_scratch,
                &mut exact_validated,
                None,
            )?;
            for (label, node) in [
                ("query_sequence", taps.query_sequence),
                ("key_sequence", taps.key_sequence),
                ("value_sequence", taps.value_sequence),
                ("gate_sequence", taps.gate_sequence),
                ("beta_sequence", taps.beta_sequence),
            ] {
                let Some(mapped) = mapping.get(&node).copied() else {
                    continue;
                };
                let Some((actual, actual_shape)) = evaluated.get(mapped) else {
                    continue;
                };
                let Some((expected, expected_shape)) = exact.get(mapped) else {
                    continue;
                };
                let maximum = actual
                    .iter()
                    .zip(expected)
                    .map(|(actual, expected)| (actual - expected).abs())
                    .fold(0.0_f32, f32::max);
                eprintln!(
                    "gdn_exact_compare layer={layer} label={label} node={} actual_shape={actual_shape:?} expected_shape={expected_shape:?} max_abs={maximum} actual_first4={:?} expected_first4={:?}",
                    node.0,
                    actual.iter().take(4).copied().collect::<Vec<_>>(),
                    expected.iter().take(4).copied().collect::<Vec<_>>(),
                );
            }
        }
        for (mapped, original) in requested {
            let (values, shape) = evaluated
                .get(mapped)
                .ok_or(InteropError::MissingEvaluatedNode { node: original })?;
            carried.insert(original, (shape.to_vec(), values.to_vec()));
            if outputs.contains(&original) {
                results.insert(original, (shape.to_vec(), values.to_vec()));
            }
        }

        let query_entry =
            carried
                .get(&taps.query_sequence)
                .ok_or(InteropError::MissingEvaluatedNode {
                    node: taps.query_sequence,
                })?;
        let query_shape = query_entry.0.as_slice();
        let query = query_entry.1.as_slice();
        let key_entry =
            carried
                .get(&taps.key_sequence)
                .ok_or(InteropError::MissingEvaluatedNode {
                    node: taps.key_sequence,
                })?;
        let key_shape = key_entry.0.as_slice();
        let key = key_entry.1.as_slice();
        let value_entry =
            carried
                .get(&taps.value_sequence)
                .ok_or(InteropError::MissingEvaluatedNode {
                    node: taps.value_sequence,
                })?;
        let value_shape = value_entry.0.clone();
        let value = value_entry.1.as_slice();
        let gate = carried
            .get(&taps.gate_sequence)
            .ok_or(InteropError::MissingEvaluatedNode {
                node: taps.gate_sequence,
            })?
            .1
            .as_slice();
        let beta = carried
            .get(&taps.beta_sequence)
            .ok_or(InteropError::MissingEvaluatedNode {
                node: taps.beta_sequence,
            })?
            .1
            .as_slice();
        let mut state = state_cache.to_vec();
        let mut output = vec![0.0_f32; value.len()];
        let positions = value_shape.first().copied().ok_or_else(|| {
            InteropError::PreGatherExecutionUnsupported {
                architecture: String::from("qwen35moe"),
                reason: String::from("gdn value sequence has no position dimension"),
            }
        })? as usize;
        let key_dim = state_shape.first().copied().ok_or_else(|| {
            InteropError::PreGatherExecutionUnsupported {
                architecture: String::from("qwen35moe"),
                reason: String::from("gdn state input has no key dimension"),
            }
        })? as usize;
        let value_dim = state_shape.get(1).copied().ok_or_else(|| {
            InteropError::PreGatherExecutionUnsupported {
                architecture: String::from("qwen35moe"),
                reason: String::from("gdn state input has no value dimension"),
            }
        })? as usize;
        let heads = state_shape
            .get(2..)
            .ok_or_else(|| InteropError::PreGatherExecutionUnsupported {
                architecture: String::from("qwen35moe"),
                reason: String::from("gdn state input has no head dimensions"),
            })?
            .iter()
            .try_fold(1_usize, |product, extent| {
                product.checked_mul(*extent as usize)
            })
            .ok_or_else(|| InteropError::PreGatherExecutionUnsupported {
                architecture: String::from("qwen35moe"),
                reason: String::from("gdn state head dimensions overflow usize"),
            })?;
        // `query`/`key` are sized by `kv_heads`, `value`/`gate`/`beta`/
        // `state` by `heads` (`GdnPrefillShape`'s own doc) -- the GQA case
        // has `kv_heads < heads`, so the head-axes count validated against
        // `query_shape` cannot reuse `heads` the way `value_shape` does.
        let head_axes_product = |shape: &[u64]| {
            shape
                .get(2..)?
                .iter()
                .try_fold(1_usize, |product, extent| {
                    product.checked_mul(*extent as usize)
                })
        };
        let kv_heads = head_axes_product(query_shape).ok_or_else(|| {
            InteropError::PreGatherExecutionUnsupported {
                architecture: String::from("qwen35moe"),
                reason: String::from("gdn query sequence has no head dimensions"),
            }
        })?;
        let projection_shape_is_valid = |shape: &[u64], first_dim: usize, head_count: usize| {
            let Some((&position_extent, rest)) = shape.split_first() else {
                return false;
            };
            let Some((&feature_extent, head_axes)) = rest.split_first() else {
                return false;
            };
            position_extent == positions as u64
                && feature_extent == first_dim as u64
                && head_axes.iter().try_fold(1usize, |product, extent| {
                    product.checked_mul(*extent as usize)
                }) == Some(head_count)
        };
        if !projection_shape_is_valid(query_shape, key_dim, kv_heads)
            || !projection_shape_is_valid(key_shape, key_dim, kv_heads)
        {
            return Err(InteropError::PreGatherExecutionUnsupported {
                architecture: String::from("qwen35moe"),
                reason: alloc::format!(
                    "gdn sequence projection shape is not [positions, feature, head axes...]: query={query_shape:?} key={key_shape:?} positions={positions} key_dim={key_dim} kv_heads={kv_heads}"
                ),
            });
        }
        if !projection_shape_is_valid(value_shape.as_slice(), value_dim, heads) {
            return Err(InteropError::PreGatherExecutionUnsupported {
                architecture: String::from("qwen35moe"),
                reason: alloc::format!(
                    "gdn value sequence shape is not [positions, feature, head axes...]: found={value_shape:?}"
                ),
            });
        }
        let gdn_recurrence = GdnPrefillScan {
            shape: GdnPrefillShape {
                positions,
                key_dim,
                value_dim,
                heads,
                kv_heads,
            },
            query,
            key,
            // This prefill path's own `projection_shape_is_valid` check above
            // proves `query`/`key` are `[positions, key_dim, kv_heads]` --
            // `kv_heads` fastest, `key_dim` slowest (`GdnPrefillScan`'s own
            // doc on why this differs from the decode-only bound kind's
            // pre-repeat, dim-fastest convention).
            query_key_head_stride: 1,
            query_key_dim_stride: kv_heads,
            value,
            gate,
            beta,
            // The graph's recurrent step applies this caller-supplied scale
            // after the per-head l2-normalized query tap.
            inv_sqrt_key_dim: 1.0 / (key_dim as f32).sqrt(),
            state: &mut state,
            output: &mut output,
        };
        #[cfg(feature = "mlx-gdn")]
        if matches!(gdn_backend, GdnPrefillBackend::Mlx) {
            mlx::run_gdn_prefill_scan(gdn_recurrence)?;
        } else {
            run_gdn_prefill_scan(gdn_recurrence)?;
        }
        #[cfg(not(feature = "mlx-gdn"))]
        run_gdn_prefill_scan(gdn_recurrence)?;
        if outputs.contains(&taps.delta_out) {
            results.insert(taps.delta_out, (value_shape.clone(), output.clone()));
        }
        if outputs.contains(&taps.state_out) {
            results.insert(taps.state_out, (state_shape.clone(), state.clone()));
        }
        carried.insert(taps.state_out, (state_shape, state));

        let (tail_program, tail_cuts, tail_mapping) = &scan.tail;
        // Keep the packed output projection internal to the tail. Requesting
        // either projection tap makes the binder retain its split weight and
        // lowers the projection as a generic reduce instead of the ordinary
        // packed matmul. Only the values consumed by the next partition are
        // execution outputs; diagnostic taps are read through an explicit
        // debug-only path rather than changing this production plan.
        let mut requested_pairs = alloc::vec![
            (scan.prefill.post_mixer_residual, scan.post_mixer_residual),
            (
                scan.prefill.post_attention_norm_output,
                scan.post_attention_norm_output,
            ),
            (scan.prefill.router_logits, scan.router_logits),
        ];
        if outputs.contains(&taps.gated_value) {
            requested_pairs.push((scan.prefill.gated_value, taps.gated_value));
        }
        if outputs.contains(&taps.ssm_out_result) {
            requested_pairs.push((scan.prefill.projected, taps.ssm_out_result));
        }
        let requested_tail: Vec<NodeId> = requested_pairs
            .iter()
            .map(|(node, _)| {
                tail_mapping
                    .get(node)
                    .copied()
                    .ok_or(InteropError::MissingEvaluatedNode { node: *node })
            })
            .collect::<Result<_, _>>()?;
        if std::env::var_os("PROXIMA_DEBUG_GDN_COMPARE").is_some() {
            eprintln!(
                "gdn_row_tail requested={requested_tail:?} program_len={}",
                tail_program.len(),
            );
        }
        let tail_original_named: Vec<(&str, QuantizedBlock<'_>)> = named
            .iter()
            .copied()
            .filter(|(name, _)| {
                !name.starts_with("gdn_prefill.")
                    && tail_program
                        .iter()
                        .any(|operation| operation.name() == Some(*name))
            })
            .collect();
        let mut row_symbols = symbols.to_vec();
        row_symbols[0] = 1;
        let delta_row_len = output.len() / positions;
        let mut accumulated_tail: BTreeMap<NodeId, (Vec<u64>, Vec<f32>)> = BTreeMap::new();

        for position in 0..positions {
            let mut row_named: Vec<(&str, QuantizedBlock<'_>)> = tail_original_named
                .iter()
                .map(|(name, block)| {
                    let expected_elements =
                        tail_program.iter().find_map(|operation| match operation {
                            Op::Input {
                                shape,
                                name: Some(input_name),
                                ..
                            } if input_name == name => {
                                shape
                                    .iter()
                                    .try_fold(1_usize, |product, extent| match extent {
                                        Extent::Static(value) => {
                                            product.checked_mul(*value as usize)
                                        }
                                        Extent::Symbolic(_) => None,
                                    })
                            }
                            _ => None,
                        });
                    match (block, expected_elements) {
                        (QuantizedBlock::Float32(values), Some(expected))
                            if values.len() == expected * positions =>
                        {
                            let start = position * expected;
                            (
                                *name,
                                QuantizedBlock::Float32(&values[start..start + expected]),
                            )
                        }
                        (QuantizedBlock::Int32(values), Some(expected))
                            if values.len() == expected * positions =>
                        {
                            let start = position * expected;
                            (
                                *name,
                                QuantizedBlock::Int32(&values[start..start + expected]),
                            )
                        }
                        _ => (*name, *block),
                    }
                })
                .collect();
            let mut row_cut_storage: Vec<(&str, Vec<f32>)> = Vec::new();
            for (node, name) in tail_cuts {
                if row_named.iter().any(|(candidate, _)| *candidate == name) {
                    continue;
                }
                if *node == scan.prefill.delta_out_input {
                    let row_start = position * delta_row_len;
                    row_cut_storage.push((
                        name.as_str(),
                        output[row_start..row_start + delta_row_len].to_vec(),
                    ));
                    continue;
                }
                let (shape, values) = carried.get(node).ok_or_else(|| {
                    InteropError::PreGatherExecutionUnsupported {
                        architecture: String::from("qwen35moe"),
                        reason: alloc::format!(
                            "gdn sequence tail missing cut node {node:?} ({name})"
                        ),
                    }
                })?;
                let target_node = tail_mapping
                    .get(node)
                    .copied()
                    .ok_or(InteropError::MissingEvaluatedNode { node: *node })?;
                let target_shape = match tail_program.get(target_node.0 as usize) {
                    Some(Op::Input { shape, .. }) => shape
                        .iter()
                        .map(|extent| match extent {
                            Extent::Static(size) => Ok(u64::from(*size)),
                            Extent::Symbolic(symbol) => row_symbols
                                .get(*symbol as usize)
                                .copied()
                                .ok_or(InteropError::PreGatherExecutionUnsupported {
                                    architecture: String::from("qwen35moe"),
                                    reason: alloc::format!(
                                        "gdn row tail cut {node:?} ({name}) uses an unbound symbol"
                                    ),
                                }),
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                    _ => Vec::new(),
                };
                let row_values = if shape.first().copied() == Some(positions as u64)
                    && target_shape.first().copied() == Some(1)
                    && shape.get(1..) == target_shape.get(1..)
                    && values.len() % positions == 0
                {
                    let row_len = values.len() / positions;
                    let row_start = position * row_len;
                    values[row_start..row_start + row_len].to_vec()
                } else if shape == &target_shape {
                    values.clone()
                } else {
                    return Err(InteropError::PreGatherExecutionUnsupported {
                        architecture: String::from("qwen35moe"),
                        reason: alloc::format!(
                            "gdn row tail cut {node:?} ({name}) source shape {shape:?} does not match target {target_shape:?}"
                        ),
                    });
                };
                row_cut_storage.push((name.as_str(), row_values));
            }
            row_named.extend(
                row_cut_storage
                    .iter()
                    .map(|(name, values)| (*name, QuantizedBlock::Float32(values))),
            );
            if std::env::var_os("PROXIMA_DEBUG_GDN_TAIL_DIGEST").is_some() && position < 2 {
                let cut_digest = row_cut_storage
                    .iter()
                    .map(|(name, values)| {
                        (*name, values.iter().take(4).copied().collect::<Vec<_>>())
                    })
                    .collect::<Vec<_>>();
                eprintln!("gdn_tail_digest position={position} named={cut_digest:?}");
            }
            let tail_evaluated = runtime.evaluate_segment(
                tail_program,
                &row_symbols,
                &row_named,
                &requested_tail,
                resident_names,
                None,
            )?;
            for (prefill_node, _) in &requested_pairs {
                let mapped = tail_mapping.get(prefill_node).copied().ok_or(
                    InteropError::MissingEvaluatedNode {
                        node: *prefill_node,
                    },
                )?;
                let (values, shape) =
                    tail_evaluated
                        .get(mapped)
                        .ok_or(InteropError::MissingEvaluatedNode {
                            node: *prefill_node,
                        })?;
                let entry = accumulated_tail
                    .entry(*prefill_node)
                    .or_insert_with(|| (shape.to_vec(), Vec::new()));
                entry.1.extend_from_slice(values);
            }
        }
        for (prefill_node, existing_node) in requested_pairs {
            let (mut shape, values) = accumulated_tail
                .remove(&prefill_node)
                .ok_or(InteropError::MissingEvaluatedNode { node: prefill_node })?;
            if let Some(position_extent) = shape.first_mut() {
                *position_extent = positions as u64;
            }
            carried.insert(existing_node, (shape.clone(), values.clone()));
            if outputs.contains(&existing_node) {
                results.insert(existing_node, (shape, values));
            }
        }
        Ok(())
    }

    pub(super) fn evaluate_qwen35moe_pre_gather<'mapping, BeforeGather>(
        &self,
        runtime: &mut BackendRuntime,
        plan: &Qwen35MoePreGatherPlan,
        symbols: &[u64],
        context: PreGatherContext<'_, 'mapping, 'file>,
        mut before_gather: BeforeGather,
    ) -> Result<Evaluated, InteropError>
    where
        BeforeGather: FnMut(
            usize,
            u64,
            &[crate::residency::RoutedExpert],
            &mut crate::expert_slab::ExpertSlab<'file>,
        ) -> Result<(), InteropError>,
    {
        let PreGatherContext {
            named,
            outputs,
            resident_names,
            layer_caches,
            expert_slab,
            sidecar_read_scratch,
            current_sources,
            position_offset,
            layer_window,
            gdn_backend,
            marker: _,
            #[cfg(feature = "metal")]
            sidecar,
            #[cfg(feature = "metal")]
            all_low_expert_scratch,
            #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
            ssm_placement,
            #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
            dense_attention_placement,
        } = context;
        #[cfg(not(feature = "metal"))]
        let _ = layer_window;
        let mut carried: BTreeMap<NodeId, (Vec<u64>, Vec<f32>)> = BTreeMap::new();
        let mut results: BTreeMap<NodeId, (Vec<u64>, Vec<f32>)> = BTreeMap::new();
        let mut placed_results = BTreeSet::new();
        let mut routed_experts = Vec::with_capacity(self.architecture.expert_used_count as usize);
        #[cfg(feature = "instrument")]
        let mut segment_execution_count = 0_u64;
        #[cfg(feature = "instrument")]
        let mut router_elapsed_us = 0_u64;
        #[cfg(feature = "instrument")]
        let mut gather_elapsed_us = 0_u64;
        #[cfg(feature = "instrument")]
        let mut router_readback_bytes = 0_u64;
        #[cfg(feature = "instrument")]
        let mut gather_readback_bytes = 0_u64;
        if std::env::var_os("PROXIMA_DEBUG_QWEN35_ROOTS").is_some() {
            eprintln!(
                "qwen35 pre-gather roots nodes={:?}",
                outputs
                    .iter()
                    .map(|node| {
                        (
                            node.0,
                            self.program
                                .get(node.0 as usize)
                                .and_then(|operation| operation.name()),
                        )
                    })
                    .collect::<Vec<_>>()
            );
        }
        for (index, operation) in self.program.iter().enumerate() {
            if let proxima_tensor::op::Op::Constant { value, .. } = operation {
                carried.insert(NodeId(index as u32), (Vec::new(), vec![*value]));
            }
        }
        #[cfg(feature = "metal")]
        let pair_window_enabled = layer_window == 2
            && plan
                .layers
                .iter()
                .any(|segments| segments.layer_window.is_some());
        let mut layer = 0usize;
        while layer < self.qwen35moe_layer_diagnostics.len() {
            let debug_layer = std::env::var("PROXIMA_DEBUG_EXPERT_GATHER_LAYER")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(0);
            // A prompt segment contains several rows, each with its own
            // router result. Build one union before the gather snapshot so
            // no row reads an unselected descriptor.
            let segments = &plan.layers[layer];
            // The scan's own conv branch (`causal_conv1d`) windows only
            // within this call's own `x` axis -- correct for the one call
            // that carries the model's entire causal context so far (the
            // initial multi-position prefill, `symbols.first() > 1`), wrong
            // for any later single-token step, which must fall through to
            // `scan.decode_router` (the ordinary persisted-history branch)
            // below instead.
            let use_gdn_scan_this_call =
                segments.gdn_scan.is_some() && symbols.first().copied().unwrap_or(1) > 1;
            if use_gdn_scan_this_call && let Some(scan) = &segments.gdn_scan {
                let state_cache = match layer_caches.get(layer) {
                    Some(LayerCacheState::Ssm(cache)) => cache.state.as_slice(),
                    _ => {
                        return Err(InteropError::PreGatherExecutionUnsupported {
                            architecture: String::from("qwen35moe"),
                            reason: alloc::format!(
                                "gdn scan layer {layer} has no ssm layer cache state"
                            ),
                        });
                    }
                };
                self.evaluate_qwen35moe_gdn_scan_segment(
                    runtime,
                    layer,
                    scan,
                    GdnScanSegmentContext {
                        state_cache,
                        gdn_backend,
                        future_cuts: &segments.router_future_cuts,
                        symbols,
                        named,
                        outputs,
                        resident_names,
                        carried: &mut carried,
                        results: &mut results,
                    },
                )?;
                let (router_shape, router_logits) =
                    carried
                        .get(&scan.router_logits)
                        .ok_or(InteropError::MissingEvaluatedNode {
                            node: scan.router_logits,
                        })?;
                visit_qwen35moe_router_boundary(
                    layer,
                    position_offset,
                    RouterLogits {
                        values: router_logits,
                        shape: router_shape,
                    },
                    RouterExpertCounts {
                        expert_count: self.architecture.expert_count as usize,
                        expert_used_count: self.architecture.expert_used_count as usize,
                    },
                    &mut routed_experts,
                    expert_slab,
                    &mut before_gather,
                )?;
            }
            #[cfg(feature = "metal")]
            if pair_window_enabled
                && layer.is_multiple_of(2)
                && layer + 1 < self.qwen35moe_layer_diagnostics.len()
                && segments.gdn_scan.is_none()
            {
                let sidecar =
                    sidecar.ok_or_else(|| InteropError::PreGatherExecutionUnsupported {
                        architecture: String::from("qwen35moe"),
                        reason: String::from(
                            "qwen35moe_layer_window=2 requires an attached exact sidecar",
                        ),
                    })?;
                if !sidecar.preserves_source_codecs() {
                    return Err(InteropError::PreGatherExecutionUnsupported {
                        architecture: String::from("qwen35moe"),
                        reason: String::from(
                            "qwen35moe_layer_window=2 requires byte-preserving sidecar codecs",
                        ),
                    });
                }
                let window = segments.layer_window.as_ref().ok_or_else(|| {
                    InteropError::PreGatherExecutionUnsupported {
                        architecture: String::from("qwen35moe"),
                        reason: alloc::format!(
                            "layer {layer} has no two-layer segment despite window=2"
                        ),
                    }
                })?;
                let all_low_sources = expert_slab.all_low_sources_for_layers(
                    sidecar,
                    layer,
                    2,
                    all_low_expert_scratch,
                )?;
                let mapped_expert_sources = map_expert_sources_to_segment(
                    layer,
                    &self.program,
                    &window.0,
                    &all_low_sources,
                )?;
                let mut segment_named: Vec<(&str, QuantizedBlock<'_>)> = named
                    .iter()
                    .copied()
                    .filter(|(name, _)| {
                        window
                            .0
                            .iter()
                            .any(|operation| operation.name() == Some(*name))
                    })
                    .collect();
                for (node, name) in &window.1 {
                    if segment_named
                        .iter()
                        .any(|(candidate, _)| *candidate == name)
                        || name.contains("_exps.weight")
                    {
                        continue;
                    }
                    let (_, values) = carried.get(node).ok_or_else(|| {
                        InteropError::PreGatherExecutionUnsupported {
                            architecture: String::from("qwen35moe"),
                            reason: alloc::format!(
                                "two-layer window missing cut node {node:?} ({name})"
                            ),
                        }
                    })?;
                    segment_named.push((name.as_str(), QuantizedBlock::Float32(values)));
                }
                let first_router = self.qwen35moe_layer_diagnostics[layer].router_logits;
                let second_router = self.qwen35moe_layer_diagnostics[layer + 1].router_logits;
                let second_output = self.qwen35moe_layer_diagnostics[layer + 1].block_output;
                let first_router_mapped = window
                    .2
                    .get(&first_router)
                    .copied()
                    .ok_or(InteropError::MissingEvaluatedNode { node: first_router })?;
                let second_router_mapped = window.2.get(&second_router).copied().ok_or(
                    InteropError::MissingEvaluatedNode {
                        node: second_router,
                    },
                )?;
                let second_output_mapped = window.2.get(&second_output).copied().ok_or(
                    InteropError::MissingEvaluatedNode {
                        node: second_output,
                    },
                )?;
                let mut requested = BTreeMap::from([
                    (first_router_mapped, first_router),
                    (second_router_mapped, second_router),
                    (second_output_mapped, second_output),
                ]);
                for original in segments
                    .next_cuts
                    .iter()
                    .map(|(node, _)| *node)
                    .chain(segments.future_gather_cuts.iter().copied())
                    .chain(plan.global_cut_nodes.iter().copied())
                    .chain(outputs.iter().copied())
                {
                    if let Some(mapped) = window.2.get(&original).copied()
                        && !matches!(
                            window.0.get(mapped.0 as usize),
                            Some(proxima_tensor::op::Op::Input { .. })
                        )
                    {
                        requested.insert(mapped, original);
                    }
                }
                let requested_nodes: Vec<NodeId> = requested.keys().copied().collect();
                let evaluated = runtime.evaluate_segment(
                    &window.0,
                    symbols,
                    &segment_named,
                    &requested_nodes,
                    resident_names,
                    Some(&mapped_expert_sources),
                )?;
                #[cfg(feature = "instrument")]
                {
                    segment_execution_count += 1;
                }
                let (first_logits, first_shape) = evaluated
                    .get(first_router_mapped)
                    .ok_or(InteropError::MissingEvaluatedNode { node: first_router })?;
                visit_qwen35moe_router_boundary(
                    layer,
                    position_offset,
                    RouterLogits {
                        values: first_logits,
                        shape: first_shape,
                    },
                    RouterExpertCounts {
                        expert_count: self.architecture.expert_count as usize,
                        expert_used_count: self.architecture.expert_used_count as usize,
                    },
                    &mut routed_experts,
                    expert_slab,
                    &mut before_gather,
                )?;
                let (second_logits, second_shape) = evaluated.get(second_router_mapped).ok_or(
                    InteropError::MissingEvaluatedNode {
                        node: second_router,
                    },
                )?;
                visit_qwen35moe_router_boundary(
                    layer + 1,
                    position_offset,
                    RouterLogits {
                        values: second_logits,
                        shape: second_shape,
                    },
                    RouterExpertCounts {
                        expert_count: self.architecture.expert_count as usize,
                        expert_used_count: self.architecture.expert_used_count as usize,
                    },
                    &mut routed_experts,
                    expert_slab,
                    &mut before_gather,
                )?;
                evaluated
                    .get(second_output_mapped)
                    .ok_or(InteropError::MissingEvaluatedNode {
                        node: second_output,
                    })?;
                for (mapped, original) in requested {
                    let (values, shape) = evaluated
                        .get(mapped)
                        .ok_or(InteropError::MissingEvaluatedNode { node: original })?;
                    carried.insert(original, (shape.to_vec(), values.to_vec()));
                    if outputs.contains(&original) {
                        results.insert(original, (shape.to_vec(), values.to_vec()));
                    }
                }
                omega::backend::clear_expert_source_cache();
                layer += 2;
                continue;
            }
            // A gdn-scan layer's `segments.router` is the SHORT post-mixer
            // tail (already evaluated above when `use_gdn_scan_this_call`);
            // a single-token call after the initial prefill instead needs
            // the layer's full `decode_router` (`previous_output ->
            // router_logits`), which reads the persisted conv history/state
            // this same layer's scan call just seeded.
            let router = if use_gdn_scan_this_call {
                &segments.router
            } else {
                segments
                    .gdn_scan
                    .as_ref()
                    .map_or(&segments.router, |scan| &scan.decode_router)
            };
            let gather = &segments.gather;
            let next_cuts = &segments.next_cuts;
            let future_gather_cuts = &segments.future_gather_cuts;
            // The linked boundary is the correctness-preserving default for
            // the routed decode path: it removes one command-buffer round
            // trip per layer while retaining the explicit router/gather
            // transition. GDN scan and placed recurrent/dense buffers still
            // fall back to the unfused segments below.
            let fused_boundary_requested =
                plan.layers
                    .iter()
                    .all(|segments| segments.gdn_scan.is_none())
                    && segments.gather_next_router.as_ref().is_some_and(|segment| {
                        fused_segment_experts_are_current_layer(segment, layer)
                    });
            #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
            let has_ssm_placement = ssm_placement
                .is_some_and(|placement| placement.buffers.iter().any(Option::is_some));
            #[cfg(not(all(feature = "metal-output-placement", target_os = "macos")))]
            let has_ssm_placement = false;
            #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
            let has_dense_attention_placement = dense_attention_placement
                .is_some_and(|placement| placement.buffers.iter().any(Option::is_some));
            #[cfg(not(all(feature = "metal-output-placement", target_os = "macos")))]
            let has_dense_attention_placement = false;
            // A linked boundary is only valid while every carried state is a
            // graph value. Placed recurrent or dense-attention outputs live
            // in caller-owned buffers, so keep the explicit router/gather
            // boundary whenever either placement is active.
            let fused_boundary_mode =
                fused_boundary_requested && !has_ssm_placement && !has_dense_attention_placement;
            if std::env::var_os("PROXIMA_DEBUG_QWEN35_FUSED").is_some() && layer < 3 {
                eprintln!(
                    "qwen35 fused boundary layer={} requested={} enabled={} has_segment={} ssm={} dense={}",
                    layer,
                    fused_boundary_requested,
                    fused_boundary_mode,
                    segments.gather_next_router.is_some(),
                    has_ssm_placement,
                    has_dense_attention_placement,
                );
            }
            for phase_index in 0..2 {
                let is_router = phase_index == 0;
                if fused_boundary_mode && is_router && layer > 0 {
                    continue;
                }
                let fused_gather = fused_boundary_mode
                    && !is_router
                    && layer + 1 < self.qwen35moe_layer_diagnostics.len();
                let phase_layer = if fused_gather { layer + 1 } else { layer };
                let diagnostic = self.qwen35moe_layer_diagnostics[phase_layer].clone();
                let (program, cuts, mapping, future_cuts, segment_output) = if fused_gather {
                    let fused = segments.gather_next_router.as_ref().ok_or_else(|| {
                        InteropError::PreGatherExecutionUnsupported {
                            architecture: String::from("qwen35moe"),
                            reason: String::from("fused boundary segment is absent"),
                        }
                    })?;
                    (
                        &fused.0,
                        &fused.1,
                        &fused.2,
                        segments.router_future_cuts.as_slice(),
                        diagnostic.router_logits,
                    )
                } else if is_router {
                    (
                        &router.0,
                        &router.1,
                        &router.2,
                        segments.router_future_cuts.as_slice(),
                        diagnostic.router_logits,
                    )
                } else {
                    (
                        &gather.0,
                        &gather.1,
                        &gather.2,
                        next_cuts.as_slice(),
                        diagnostic.block_output,
                    )
                };
                if is_router && use_gdn_scan_this_call {
                    continue;
                }
                let mut segment_named: Vec<(&str, QuantizedBlock<'_>)> = named
                    .iter()
                    .copied()
                    .filter(|(name, _)| {
                        program
                            .iter()
                            .any(|operation| operation.name() == Some(*name))
                    })
                    .collect();
                for (node, name) in cuts {
                    if segment_named
                        .iter()
                        .any(|(candidate, _)| *candidate == name)
                    {
                        continue;
                    }
                    if name.contains("_exps.weight") {
                        // The mapped expert source supplies this input at
                        // execution time; a placeholder would make the
                        // resolver classify the node as an ordinary f32
                        // binding and bypass the source table.
                        continue;
                    }
                    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                    if plan.router_cut_placements.contains_key(node) {
                        if !is_router && let Some((_, values)) = carried.get(node) {
                            // gdn scan outputs are produced on the host before
                            // the gather; bind those bytes instead of the
                            // caller-owned placement.
                            segment_named
                                .push((name.as_str(), QuantizedBlock::Float32(values.as_slice())));
                        }
                        // The placement-aware plan and resolver accept a
                        // missing named block and bind its caller-owned
                        // buffer directly. No host placeholder is needed.
                        continue;
                    }
                    let (_, values) = carried.get(node).ok_or_else(|| {
                        InteropError::PreGatherExecutionUnsupported {
                            architecture: String::from("qwen35moe"),
                            reason: alloc::format!(
                                "layer {layer} missing cut node {node:?} ({name})"
                            ),
                        }
                    })?;
                    if std::env::var_os("PROXIMA_DEBUG_GDN_CARRY").is_some() {
                        eprintln!(
                            "gdn_gather_cut layer={} phase={} node={} name={} elements={} first4={:?}",
                            layer,
                            if is_router { "router" } else { "gather" },
                            node.0,
                            name,
                            values.len(),
                            values.iter().take(4).copied().collect::<Vec<_>>(),
                        );
                    }
                    segment_named.push((name.as_str(), QuantizedBlock::Float32(values)));
                }

                if layer == 3
                    && is_router
                    && std::env::var_os("PROXIMA_DEBUG_DENSE_GRAPH").is_some()
                    && let Some(Op::Reduce(reduce)) = self.program.get(
                        self.qwen35moe_layer_diagnostics[layer]
                            .dense_attention_taps
                            .map_or(NodeId(u32::MAX), |taps| taps.q_split)
                            .0 as usize,
                    )
                    && let Some(Op::Elementwise { operands, .. }) =
                        self.program.get(reduce.operand.0 as usize)
                    && let Some((q_product, _)) = operands.first()
                    && let Some(mapped_product) = mapping.get(q_product)
                {
                    eprintln!(
                        "dense_segment_qg_layout original_product={} mapped_product={} local_op={:?} original_operands={:?} local_operands={:?} named_q={:?}",
                        q_product.0,
                        mapped_product.0,
                        program.get(mapped_product.0 as usize),
                        operands,
                        program
                            .get(mapped_product.0 as usize)
                            .and_then(|operation| match operation {
                                Op::Elementwise { operands, .. } => Some(operands.as_slice()),
                                _ => None,
                            })
                            .unwrap_or(&[]),
                        segment_named
                            .iter()
                            .map(|(name, block)| (*name, core::mem::discriminant(block)))
                            .filter(|(name, _)| name.contains("attn_q"))
                            .collect::<Vec<_>>(),
                    );
                    if let Some(Op::Reduce(local_reduce)) = program.get(mapped_product.0 as usize) {
                        eprintln!(
                            "dense_segment_qg_reduce original_operand={} mapped_operand={} original_operand_op={:?} local_operand_op={:?}",
                            self.program
                                .get(q_product.0 as usize)
                                .and_then(|operation| match operation {
                                    Op::Reduce(reduce) => Some(reduce.operand.0),
                                    _ => None,
                                })
                                .unwrap_or(u32::MAX),
                            local_reduce.operand.0,
                            self.program
                                .get(q_product.0 as usize)
                                .and_then(|operation| match operation {
                                    Op::Reduce(reduce) =>
                                        self.program.get(reduce.operand.0 as usize),
                                    _ => None,
                                }),
                            program.get(local_reduce.operand.0 as usize),
                        );
                        eprintln!(
                            "dense_segment_qg_inputs local_activation={:?} local_weight={:?}",
                            program.get(39),
                            program.get(16),
                        );
                        eprintln!(
                            "dense_segment_qg_weight original={:?} local={:?}",
                            self.program.get(1233),
                            program.get(16),
                        );
                        for (index, operation) in program.iter().enumerate() {
                            if let Op::Input { name, .. } = operation {
                                eprintln!("dense_segment_qg_input_node node={index} name={name:?}");
                            }
                        }
                    }
                }

                if std::env::var_os("PROXIMA_DEBUG_QWEN35_PLAN").is_some()
                    && !is_router
                    && layer == 0
                    && let Some((route_node, route_name)) = cuts
                        .iter()
                        .find(|(node, _)| *node == diagnostic.router_logits)
                    && let Some((route_shape, route_values)) = carried.get(route_node)
                {
                    eprintln!(
                        "qwen35 gather route input original={route_node:?} name={route_name} shape={route_shape:?} first={:?} min={} max={} finite={}",
                        route_values
                            .get(..route_values.len().min(8))
                            .unwrap_or_default(),
                        route_values.iter().copied().fold(f32::INFINITY, f32::min),
                        route_values
                            .iter()
                            .copied()
                            .fold(f32::NEG_INFINITY, f32::max),
                        route_values.iter().all(|value| value.is_finite()),
                    );
                }

                let mut requested = BTreeMap::new();
                let prefix_carried_nodes =
                    (layer == 0 && is_router).then_some(&plan.prefix_carried_nodes);
                for node in future_cuts
                    .iter()
                    .filter(|(_, name)| !named.iter().any(|(candidate, _)| *candidate == name))
                    .map(|(node, _)| node)
                    .chain(prefix_carried_nodes.into_iter().flatten())
                    .chain(plan.global_cut_nodes.iter())
                    .chain(future_gather_cuts.iter())
                    .chain(outputs)
                    .filter(|node| {
                        if outputs.contains(node) {
                            return true;
                        }
                        mapping.get(node).is_some_and(|mapped| {
                            !matches!(
                                program.get(mapped.0 as usize),
                                Some(proxima_tensor::op::Op::Input { .. })
                            )
                        })
                    })
                {
                    if let Some(mapped) = mapping.get(node).copied() {
                        requested.insert(mapped, *node);
                    }
                }
                requested.insert(
                    mapping.get(&segment_output).copied().ok_or(
                        InteropError::MissingEvaluatedNode {
                            node: segment_output,
                        },
                    )?,
                    segment_output,
                );
                #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                let placed_dense_roots = dense_attention_placement
                    .and_then(|placement| placement.buffers[layer].as_ref())
                    .and_then(|_| match self.layer_roots[layer] {
                        Qwen35LayerRoots::DenseAttention(roots) => Some(roots),
                        _ => None,
                    });
                #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                retain_qwen35_segment_readbacks(
                    &mut requested,
                    &plan.router_cut_placements,
                    placed_dense_roots,
                    program,
                );
                let mut requested_nodes: Vec<NodeId> = requested.keys().copied().collect();
                #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                let mut segment_input_placements = Vec::new();
                #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                let mut segment_output_placements = Vec::new();
                #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                for (node, buffer) in &plan.router_cut_placements {
                    if std::env::var_os("PROXIMA_DEBUG_QWEN35_PLACEMENTS").is_some() {
                        eprintln!(
                            "qwen35 placement phase={} layer={} original={:?} mapped={:?}",
                            if is_router { "router" } else { "gather" },
                            layer,
                            node,
                            mapping.get(node),
                        );
                    }
                    if let Some(mapped) = mapping.get(node).copied() {
                        if std::env::var_os("PROXIMA_DEBUG_QWEN35_PLACEMENTS").is_some()
                            && layer == 39
                            && *node == NodeId(15)
                        {
                            eprintln!(
                                "qwen35 placement detail original={node:?} mapped={mapped:?} op={:?}",
                                program.get(mapped.0 as usize),
                            );
                        }
                        if matches!(program.get(mapped.0 as usize), Some(Op::Input { .. })) {
                            if !is_router && carried.contains_key(node) {
                                // A gdn scan produced this boundary on the
                                // host. Let its carried block bind normally;
                                // an input placement would override it with
                                // the zero placeholder.
                                continue;
                            }
                            segment_input_placements.push((mapped, buffer, 0));
                        } else {
                            segment_output_placements.push((mapped, buffer, 0));
                        }
                    }
                }
                #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                if std::env::var_os("PROXIMA_DEBUG_QWEN35_PLACEMENTS").is_some()
                    && layer == 39
                    && is_router
                {
                    eprintln!(
                        "qwen35 placement bindings layer=39 inputs={:?} outputs={:?}",
                        segment_input_placements
                            .iter()
                            .map(|(node, _, offset)| (*node, *offset))
                            .collect::<Vec<_>>(),
                        segment_output_placements
                            .iter()
                            .map(|(node, _, offset)| (*node, *offset))
                            .collect::<Vec<_>>(),
                    );
                }
                #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                if is_router
                    && let Some(placement) = dense_attention_placement
                    && let (
                        Qwen35LayerRoots::DenseAttention(roots),
                        Some(input_nodes),
                        Some(buffers),
                    ) = (
                        self.layer_roots[layer],
                        placement.input_nodes[layer],
                        placement.buffers[layer].as_ref(),
                    )
                {
                    for (node, buffer) in [
                        (input_nodes.0, &buffers.k_first),
                        (input_nodes.1, &buffers.k_second),
                        (input_nodes.2, &buffers.k_pass),
                        (input_nodes.3, &buffers.value),
                    ] {
                        let mapped = mapping.get(&node).copied().ok_or_else(|| {
                            InteropError::PreGatherExecutionUnsupported {
                                architecture: String::from("qwen35moe"),
                                reason: alloc::format!(
                                    "layer {layer} dense-attention cache input {node:?} is absent from the router segment"
                                ),
                            }
                        })?;
                        segment_input_placements.push((mapped, buffer, 0));
                    }
                    for (node, buffer, row_bytes) in [
                        (roots.0, &buffers.k_first, buffers.even_odd_row_bytes),
                        (roots.1, &buffers.k_second, buffers.even_odd_row_bytes),
                        (roots.2, &buffers.k_pass, buffers.pass_row_bytes),
                        (roots.3, &buffers.value, buffers.value_row_bytes),
                    ] {
                        let mapped = mapping.get(&node).copied().ok_or_else(|| {
                            InteropError::PreGatherExecutionUnsupported {
                                architecture: String::from("qwen35moe"),
                                reason: alloc::format!(
                                    "layer {layer} dense-attention cache root {node:?} is absent from the router segment"
                                ),
                            }
                        })?;
                        let byte_offset =
                            position_offset.checked_mul(row_bytes).ok_or_else(|| {
                                InteropError::PreGatherExecutionUnsupported {
                                    architecture: String::from("qwen35moe"),
                                    reason: alloc::format!(
                                        "layer {layer} dense-attention cache offset overflowed"
                                    ),
                                }
                            })?;
                        segment_output_placements.push((mapped, buffer, byte_offset));
                        requested_nodes.push(mapped);
                    }
                }
                #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                if let Some(placement) = ssm_placement
                    && placement
                        .maximum_layer
                        .is_none_or(|maximum| layer <= maximum)
                    && let (
                        Qwen35LayerRoots::Ssm { state_out, .. },
                        Some(state_input),
                        Some((first_buffer, second_buffer)),
                    ) = (
                        &self.layer_roots[layer],
                        placement.input_nodes[layer],
                        placement.buffers[layer].as_ref(),
                    )
                {
                    let (input_buffer, output_buffer) = if placement.use_second_as_input {
                        (second_buffer, first_buffer)
                    } else {
                        (first_buffer, second_buffer)
                    };
                    if let Some(mapped_input) = mapping.get(&state_input).copied() {
                        segment_input_placements.push((mapped_input, input_buffer, 0));
                    }
                    if let Some(mapped_output) = mapping.get(state_out).copied() {
                        segment_output_placements.push((mapped_output, output_buffer, 0));
                        requested_nodes.push(mapped_output);
                    }
                }
                if layer == 3
                    && is_router
                    && std::env::var_os("PROXIMA_DEBUG_DENSE_GRAPH").is_some()
                    && let Some(Op::Reduce(reduce)) = self.program.get(
                        self.qwen35moe_layer_diagnostics[layer]
                            .dense_attention_taps
                            .map_or(NodeId(u32::MAX), |taps| taps.q_split)
                            .0 as usize,
                    )
                    && let Some(Op::Elementwise { operands, .. }) =
                        self.program.get(reduce.operand.0 as usize)
                    && let Some((q_product, _)) = operands.first()
                    && let Some(mapped) = mapping.get(q_product)
                {
                    requested_nodes.push(*mapped);
                    for node in [NodeId(39), NodeId(16), NodeId(14), NodeId(15)] {
                        requested_nodes.push(node);
                    }
                }
                requested_nodes.sort_unstable_by_key(|node| node.0);
                requested_nodes.dedup();
                if std::env::var_os("PROXIMA_DEBUG_QWEN35_REQUESTS").is_some() {
                    eprintln!(
                        "qwen35 segment requests phase={} layer={} program_ops={} named_inputs={} requested_nodes={} future_cuts={} global_cuts={} outputs={}",
                        if is_router { "router" } else { "gather" },
                        layer,
                        program.len(),
                        segment_named.len(),
                        requested_nodes.len(),
                        future_cuts.len(),
                        plan.global_cut_nodes.len(),
                        outputs.len(),
                    );
                }
                if layer == debug_layer
                    && !is_router
                    && std::env::var_os("PROXIMA_DEBUG_EXPERT_GATHER_PARITY").is_some()
                {
                    if let Some(debug_node) = std::env::var("PROXIMA_DEBUG_EXPERT_GATHER_NODE")
                        .ok()
                        .and_then(|value| value.parse::<u32>().ok())
                    {
                        let mut pending = vec![NodeId(debug_node)];
                        let mut visited = BTreeSet::new();
                        while let Some(debug_node) = pending.pop() {
                            if !visited.insert(debug_node) {
                                continue;
                            }
                            requested_nodes.push(debug_node);
                            match program.get(debug_node.0 as usize) {
                                Some(Op::Elementwise { operands, .. }) => {
                                    pending.extend(operands.iter().map(|(node, _)| *node));
                                }
                                Some(Op::Reduce(reduce)) => pending.push(reduce.operand),
                                _ => {}
                            }
                        }
                    }
                    requested_nodes.sort_unstable_by_key(|node| node.0);
                    requested_nodes.dedup();
                }
                let debug_nonfinite = layer == debug_layer
                    && !is_router
                    && std::env::var_os("PROXIMA_DEBUG_EXPERT_GATHER_NONFINITE").is_some();
                let evaluated = if is_router && !fused_gather {
                    let segment_started = std::time::Instant::now();
                    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                    let result = if segment_input_placements.is_empty()
                        && segment_output_placements.is_empty()
                    {
                        #[cfg(feature = "instrument")]
                        if routed_segment_profile_selected(layer, "router") {
                            let (evaluated, timings) = runtime.evaluate_segment_op_timed(
                                program,
                                symbols,
                                &segment_named,
                                &requested_nodes,
                                resident_names,
                                &BTreeMap::new(),
                            )?;
                            report_op_timings(position_offset, &timings, program);
                            Ok(evaluated)
                        } else {
                            runtime.evaluate_segment(
                                program,
                                symbols,
                                &segment_named,
                                &requested_nodes,
                                resident_names,
                                None,
                            )
                        }
                        #[cfg(not(feature = "instrument"))]
                        runtime.evaluate_segment(
                            program,
                            symbols,
                            &segment_named,
                            &requested_nodes,
                            resident_names,
                            None,
                        )
                    } else {
                        let empty_expert_sources = BTreeMap::new();
                        #[cfg(feature = "instrument")]
                        if routed_segment_profile_selected(layer, "router") {
                            let (evaluated, timings) = runtime
                                .evaluate_segment_op_timed_with_placements(
                                    program,
                                    symbols,
                                    &segment_named,
                                    &requested_nodes,
                                    resident_names,
                                    SegmentPlacements {
                                        input_placements: &segment_input_placements,
                                        output_placements: &segment_output_placements,
                                    },
                                )?;
                            report_op_timings(position_offset, &timings, program);
                            Ok(evaluated)
                        } else {
                            runtime.evaluate_segment_with_placements_and_expert_sources(
                                program,
                                symbols,
                                &segment_named,
                                &requested_nodes,
                                resident_names,
                                &SegmentMetalBindings {
                                    input_placements: &segment_input_placements,
                                    output_placements: &segment_output_placements,
                                    expert_sources: &empty_expert_sources,
                                },
                            )
                        }
                        #[cfg(not(feature = "instrument"))]
                        runtime.evaluate_segment_with_placements_and_expert_sources(
                            program,
                            symbols,
                            &segment_named,
                            &requested_nodes,
                            resident_names,
                            &SegmentMetalBindings {
                                input_placements: &segment_input_placements,
                                output_placements: &segment_output_placements,
                                expert_sources: &empty_expert_sources,
                            },
                        )
                    };
                    #[cfg(not(all(feature = "metal-output-placement", target_os = "macos")))]
                    let result = runtime.evaluate_segment(
                        program,
                        symbols,
                        &segment_named,
                        &requested_nodes,
                        resident_names,
                        None,
                    );
                    #[cfg(feature = "instrument")]
                    {
                        router_elapsed_us += segment_started.elapsed().as_micros() as u64;
                    }
                    if std::env::var_os("PROXIMA_DEBUG_QWEN35_SEGMENTS").is_some() {
                        eprintln!(
                            "qwen35 segment phase=router layer={} elapsed_us={}",
                            layer,
                            segment_started.elapsed().as_micros()
                        );
                    }
                    let evaluated = result?;
                    if layer == 3
                        && std::env::var_os("PROXIMA_DEBUG_DENSE_GRAPH").is_some()
                        && let Some(Op::Reduce(reduce)) = self.program.get(
                            self.qwen35moe_layer_diagnostics[layer]
                                .dense_attention_taps
                                .map_or(NodeId(u32::MAX), |taps| taps.q_split)
                                .0 as usize,
                        )
                        && let Some(Op::Elementwise { operands, .. }) =
                            self.program.get(reduce.operand.0 as usize)
                        && let Some((q_product, _)) = operands.first()
                        && let Some(mapped_product) = mapping.get(q_product)
                    {
                        let mut exact_scratch = Vec::new();
                        let mut exact_validated = None;
                        let exact = evaluate_quantized_named_exact_with_scratch_and_experts(
                            program,
                            symbols,
                            &segment_named,
                            &requested_nodes,
                            &mut exact_scratch,
                            &mut exact_validated,
                            None,
                        )?;
                        eprintln!(
                            "dense_segment_qg_compare mapped={} runtime={:?} exact={:?}",
                            mapped_product.0,
                            evaluated.get(*mapped_product).map(|(values, _)| values
                                .iter()
                                .take(4)
                                .copied()
                                .collect::<Vec<_>>()),
                            exact.get(*mapped_product).map(|(values, _)| values
                                .iter()
                                .take(4)
                                .copied()
                                .collect::<Vec<_>>()),
                        );
                        for node in [NodeId(39), NodeId(16), NodeId(14), NodeId(15)] {
                            eprintln!(
                                "dense_segment_qg_local node={} runtime={:?} exact={:?}",
                                node.0,
                                evaluated.get(node).map(|(values, shape)| (
                                    shape.to_vec(),
                                    values.iter().take(4).copied().collect::<Vec<_>>(),
                                )),
                                exact.get(node).map(|(values, shape)| (
                                    shape.to_vec(),
                                    values.iter().take(4).copied().collect::<Vec<_>>(),
                                )),
                            );
                        }
                    }
                    if std::env::var_os("PROXIMA_DEBUG_EXPERT_ROUTER_PARITY").is_some()
                        && layer == debug_layer
                    {
                        let mut scratch = Vec::new();
                        let mut validated = None;
                        let expected = evaluate_quantized_named_exact_with_scratch_and_experts(
                            program,
                            symbols,
                            &segment_named,
                            &requested_nodes,
                            &mut scratch,
                            &mut validated,
                            None,
                        )?;
                        let mapped_output = mapping.get(&segment_output).copied().ok_or(
                            InteropError::MissingEvaluatedNode {
                                node: segment_output,
                            },
                        )?;
                        if let (Some((actual_values, _)), Some((expected_values, _))) =
                            (evaluated.get(mapped_output), expected.get(mapped_output))
                        {
                            let (index, maximum) = actual_values
                                .iter()
                                .zip(expected_values)
                                .enumerate()
                                .map(|(index, (actual, expected))| {
                                    (index, (actual - expected).abs())
                                })
                                .max_by(|left, right| left.1.total_cmp(&right.1))
                                .unwrap_or((0, 0.0));
                            eprintln!(
                                "qwen35 router parity layer={layer} node={} max_abs={maximum} index={index} metal={} cpu={} shape={:?}",
                                segment_output.0,
                                actual_values.get(index).copied().unwrap_or_default(),
                                expected_values.get(index).copied().unwrap_or_default(),
                                actual_values.len(),
                            );
                        }
                    }
                    evaluated
                } else {
                    let selected_sidecar = if let Some(sidecar) = &self.expert_sidecar
                        && (sidecar.uses_file_reads() || sidecar.has_checkpoint_file())
                    {
                        #[cfg(feature = "instrument")]
                        let sidecar_read_started = read_ticks();
                        sidecar.read_selected_with_checkpoint_admitting(
                            layer,
                            expert_slab.selected_experts(layer),
                            expert_slab,
                            &mut *sidecar_read_scratch,
                            &crate::expert_sidecar::CheckpointAdmission {
                                checkpoint_mapping: Some(self.checkpoint_mapping),
                                current_decisions: current_sources.borrow().as_slice(),
                                admit_low_copy: qwen35moe_admit_low_copy,
                            },
                        )?;
                        if std::env::var_os("PROXIMA_DEBUG_EXPERT_UPLOADS").is_some() {
                            #[cfg(feature = "instrument")]
                            let sidecar_read_elapsed_us =
                                ticks_to_nanos(elapsed_ticks(sidecar_read_started)) / 1_000;
                            #[cfg(not(feature = "instrument"))]
                            let sidecar_read_elapsed_us = 0;
                            eprintln!(
                                "qwen35 bounded expert reads layer={} ranges={} bytes={} low_ranges={} high_ranges={} high_cache_hits={} high_cache_misses={} elapsed_us={}",
                                layer,
                                sidecar_read_scratch.ranges_read,
                                sidecar_read_scratch.bytes_read,
                                sidecar_read_scratch.low_ranges_read,
                                sidecar_read_scratch.high_ranges_read,
                                sidecar_read_scratch.high_cache_hits,
                                sidecar_read_scratch.high_cache_misses,
                                sidecar_read_elapsed_us,
                            );
                        }
                        Some(&*sidecar_read_scratch)
                    } else {
                        None
                    };
                    #[cfg(feature = "metal")]
                    let sidecar_mapping = self
                        .expert_sidecar
                        .as_ref()
                        .map(crate::expert_sidecar::MappedExpertSidecar::mapping_bytes);
                    #[cfg(not(feature = "metal"))]
                    let sidecar_mapping = None;
                    let mut expert_entries_scratch = Vec::with_capacity(3);
                    let expert_sources = expert_slab.sources_for_layer_with_sidecar(
                        layer,
                        sidecar_mapping,
                        selected_sidecar,
                        &mut expert_entries_scratch,
                    )?;
                    let mapped_expert_sources =
                        if std::env::var_os("PROXIMA_DEBUG_DISABLE_EXPERT_SOURCES").is_some() {
                            BTreeMap::new()
                        } else {
                            map_expert_sources_to_segment(
                                layer,
                                &self.program,
                                program,
                                &expert_sources,
                            )?
                        };
                    if std::env::var_os("PROXIMA_DEBUG_EXPERT_UPLOADS").is_some() {
                        eprintln!(
                            "qwen35 mapped expert source nodes={:?}",
                            mapped_expert_sources.keys().collect::<Vec<_>>()
                        );
                    }
                    let segment_started = std::time::Instant::now();
                    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                    let result = if segment_input_placements.is_empty()
                        && segment_output_placements.is_empty()
                    {
                        #[cfg(feature = "instrument")]
                        if routed_segment_profile_selected(layer, "gather") {
                            let (evaluated, timings) = runtime.evaluate_segment_op_timed(
                                program,
                                symbols,
                                &segment_named,
                                &requested_nodes,
                                resident_names,
                                &mapped_expert_sources,
                            )?;
                            report_op_timings(position_offset, &timings, program);
                            Ok(evaluated)
                        } else {
                            runtime.evaluate_segment(
                                program,
                                symbols,
                                &segment_named,
                                &requested_nodes,
                                resident_names,
                                Some(&mapped_expert_sources),
                            )
                        }
                        #[cfg(not(feature = "instrument"))]
                        runtime.evaluate_segment(
                            program,
                            symbols,
                            &segment_named,
                            &requested_nodes,
                            resident_names,
                            Some(&mapped_expert_sources),
                        )
                    } else {
                        runtime.evaluate_segment_with_placements_and_expert_sources(
                            program,
                            symbols,
                            &segment_named,
                            &requested_nodes,
                            resident_names,
                            &SegmentMetalBindings {
                                input_placements: &segment_input_placements,
                                output_placements: &segment_output_placements,
                                expert_sources: &mapped_expert_sources,
                            },
                        )
                    };
                    #[cfg(all(
                        feature = "instrument",
                        feature = "metal",
                        target_os = "macos",
                        not(feature = "metal-output-placement")
                    ))]
                    let result = if routed_segment_profile_selected(layer, "gather") {
                        let (evaluated, timings) = runtime.evaluate_segment_op_timed(
                            program,
                            symbols,
                            &segment_named,
                            &requested_nodes,
                            resident_names,
                            &mapped_expert_sources,
                        )?;
                        report_op_timings(position_offset, &timings, program);
                        Ok(evaluated)
                    } else {
                        runtime.evaluate_segment(
                            program,
                            symbols,
                            &segment_named,
                            &requested_nodes,
                            resident_names,
                            Some(&mapped_expert_sources),
                        )
                    };
                    #[cfg(not(any(
                        all(feature = "metal-output-placement", target_os = "macos"),
                        all(
                            feature = "instrument",
                            feature = "metal",
                            target_os = "macos",
                            not(feature = "metal-output-placement")
                        )
                    )))]
                    let result = runtime.evaluate_segment(
                        program,
                        symbols,
                        &segment_named,
                        &requested_nodes,
                        resident_names,
                        Some(&mapped_expert_sources),
                    );
                    if layer == 3
                        && is_router
                        && std::env::var_os("PROXIMA_DEBUG_DENSE_GRAPH").is_some()
                        && let Some(Op::Reduce(reduce)) = self.program.get(
                            self.qwen35moe_layer_diagnostics[layer]
                                .dense_attention_taps
                                .map_or(NodeId(u32::MAX), |taps| taps.q_split)
                                .0 as usize,
                        )
                        && let Some(Op::Elementwise { operands, .. }) =
                            self.program.get(reduce.operand.0 as usize)
                        && let Some((q_product, _)) = operands.first()
                        && let Some(mapped_product) = mapping.get(q_product)
                    {
                        let mut exact_scratch = Vec::new();
                        let mut exact_validated = None;
                        let exact = evaluate_quantized_named_exact_with_scratch_and_experts(
                            program,
                            symbols,
                            &segment_named,
                            &requested_nodes,
                            &mut exact_scratch,
                            &mut exact_validated,
                            Some(&mapped_expert_sources),
                        )?;
                        eprintln!(
                            "dense_segment_qg_compare mapped={} runtime={:?} exact={:?}",
                            mapped_product.0,
                            result.as_ref().ok().and_then(|values| {
                                values.get(*mapped_product).map(|(values, _)| {
                                    values.iter().take(4).copied().collect::<Vec<_>>()
                                })
                            }),
                            exact.get(*mapped_product).map(|(values, _)| values
                                .iter()
                                .take(4)
                                .copied()
                                .collect::<Vec<_>>()),
                        );
                    }
                    if (layer == debug_layer
                        && std::env::var_os("PROXIMA_DEBUG_EXPERT_GATHER_PARITY").is_some())
                        || debug_nonfinite
                    {
                        let mut scratch = Vec::new();
                        let mut validated = None;
                        let expected = evaluate_quantized_named_exact_with_scratch_and_experts(
                            program,
                            symbols,
                            &segment_named,
                            &requested_nodes,
                            &mut scratch,
                            &mut validated,
                            Some(&mapped_expert_sources),
                        )?;
                        if debug_nonfinite && let Ok(actual) = &result {
                            let first_actual = first_nonfinite_node_value(actual, &requested_nodes);
                            let first_cpu = first_nonfinite_node_value(&expected, &requested_nodes);
                            let first = first_actual.as_ref().or(first_cpu.as_ref());
                            if let Some(first) = first {
                                let cpu_counterpart = expected
                                    .get(first.node)
                                    .and_then(|(values, _)| values.get(first.index))
                                    .copied();
                                let metal_counterpart = actual
                                    .get(first.node)
                                    .and_then(|(values, _)| values.get(first.index))
                                    .copied();
                                eprintln!(
                                    "qwen35 first nonfinite gather layer={layer} local_node={:?} name={:?} op={:?} element={} value={} shape={:?} metal={metal_counterpart:?} cpu={cpu_counterpart:?}",
                                    first.node,
                                    program[first.node.0 as usize].name(),
                                    program[first.node.0 as usize],
                                    first.index,
                                    first.value,
                                    first.shape,
                                );
                                if let Some(Op::Elementwise { operands, .. }) =
                                    program.get(first.node.0 as usize)
                                {
                                    for (operand, _) in operands {
                                        let metal_value = actual
                                            .get(*operand)
                                            .and_then(|(values, _)| values.get(first.index))
                                            .copied();
                                        let cpu_value = expected
                                            .get(*operand)
                                            .and_then(|(values, _)| values.get(first.index))
                                            .copied();
                                        eprintln!(
                                            "qwen35 nonfinite operand local={operand:?} op={:?} metal={metal_value:?} cpu={cpu_value:?}",
                                            program.get(operand.0 as usize),
                                        );
                                    }
                                }
                            } else {
                                eprintln!(
                                    "qwen35 first nonfinite gather layer={layer} result=none local_nodes={}",
                                    requested_nodes.len(),
                                );
                            }
                        }
                        let debug_local_node = std::env::var("PROXIMA_DEBUG_EXPERT_GATHER_NODE")
                            .ok()
                            .and_then(|value| value.parse::<u32>().ok())
                            .map(NodeId);
                        let parity_node = debug_local_node
                            .and_then(|local| {
                                mapping.iter().find_map(|(original, mapped)| {
                                    (*mapped == local).then_some(*original)
                                })
                            })
                            .unwrap_or(segment_output);
                        let mapped_output = mapping
                            .get(&parity_node)
                            .copied()
                            .ok_or(InteropError::MissingEvaluatedNode { node: parity_node })?;
                        if let (Ok(actual), Some((expected_values, _))) =
                            (&result, expected.get(mapped_output))
                            && let Some((actual_values, _)) = actual.get(mapped_output)
                        {
                            let (index, maximum) = actual_values
                                .iter()
                                .zip(expected_values)
                                .enumerate()
                                .map(|(index, (actual, expected))| {
                                    (index, (actual - expected).abs())
                                })
                                .max_by(|left, right| left.1.total_cmp(&right.1))
                                .unwrap_or((0, 0.0));
                            eprintln!(
                                "qwen35 gather parity layer={layer} node={} original_op={:?} max_abs={maximum} index={index} metal={} cpu={}",
                                parity_node.0,
                                self.program
                                    .get(parity_node.0 as usize)
                                    .map(|operation| operation.name()),
                                actual_values.get(index).copied().unwrap_or_default(),
                                expected_values.get(index).copied().unwrap_or_default(),
                            );
                        }
                        if let Ok(actual) = &result {
                            if std::env::var_os("PROXIMA_DEBUG_EXPERT_GATHER_DIGEST").is_some() {
                                for node in &requested_nodes {
                                    let Some((actual_values, actual_shape)) = actual.get(*node)
                                    else {
                                        continue;
                                    };
                                    let Some((expected_values, expected_shape)) =
                                        expected.get(*node)
                                    else {
                                        continue;
                                    };
                                    let maximum = actual_values
                                        .iter()
                                        .zip(expected_values)
                                        .map(|(actual, expected)| (actual - expected).abs())
                                        .fold(0.0_f32, f32::max);
                                    eprintln!(
                                        "qwen35 gather digest layer={layer} node={node:?} name={:?} actual_shape={actual_shape:?} expected_shape={expected_shape:?} max_abs={maximum} actual_first={:?} expected_first={:?}",
                                        program[node.0 as usize].name(),
                                        actual_values.iter().take(4).copied().collect::<Vec<_>>(),
                                        expected_values.iter().take(4).copied().collect::<Vec<_>>(),
                                    );
                                }
                            }
                            if std::env::var_os("PROXIMA_DEBUG_EXPERT_GATHER_GRAPH").is_some() {
                                let graph_root =
                                    std::env::var("PROXIMA_DEBUG_EXPERT_GATHER_GRAPH_NODE")
                                        .ok()
                                        .and_then(|value| value.parse::<u32>().ok())
                                        .map(NodeId)
                                        .unwrap_or(parity_node);
                                let mut pending = vec![graph_root];
                                let mut visited = BTreeSet::new();
                                while let Some(node) = pending.pop() {
                                    if !visited.insert(node) {
                                        continue;
                                    }
                                    let Some(operation) = program.get(node.0 as usize) else {
                                        continue;
                                    };
                                    match operation {
                                        proxima_tensor::op::Op::Input { name, .. } => {
                                            eprintln!(
                                                "qwen35 gather graph input node={node:?} name={name:?}"
                                            );
                                        }
                                        proxima_tensor::op::Op::Elementwise {
                                            operands, ..
                                        } => {
                                            pending.extend(
                                                operands.iter().map(|(operand, _)| *operand),
                                            );
                                        }
                                        proxima_tensor::op::Op::Reduce(reduce) => {
                                            pending.push(reduce.operand)
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            for node in &requested_nodes {
                                let Some((actual_values, _)) = actual.get(*node) else {
                                    continue;
                                };
                                let Some((expected_values, _)) = expected.get(*node) else {
                                    continue;
                                };
                                let maximum = actual_values
                                    .iter()
                                    .zip(expected_values)
                                    .map(|(actual, expected)| (actual - expected).abs())
                                    .fold(0.0_f32, f32::max);
                                if maximum > 1.0e-3 {
                                    eprintln!(
                                        "qwen35 first gather divergence layer={layer} node={node:?} name={:?} op={:?} max_abs={maximum}",
                                        program[node.0 as usize].name(),
                                        program[node.0 as usize],
                                    );
                                    if let proxima_tensor::op::Op::Elementwise {
                                        operands, ..
                                    } = &program[node.0 as usize]
                                    {
                                        for (operand_index, (operand, _)) in
                                            operands.iter().enumerate()
                                        {
                                            if let (
                                                Some((actual_operand, _)),
                                                Some((expected_operand, _)),
                                            ) = (actual.get(*operand), expected.get(*operand))
                                            {
                                                let operand_maximum = actual_operand
                                                    .iter()
                                                    .zip(expected_operand)
                                                    .map(|(actual, expected)| {
                                                        (actual - expected).abs()
                                                    })
                                                    .fold(0.0_f32, f32::max);
                                                eprintln!(
                                                    "qwen35 gather operand node={operand:?} index={operand_index} max_abs={operand_maximum}"
                                                );
                                            }
                                        }
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    #[cfg(feature = "instrument")]
                    {
                        gather_elapsed_us += segment_started.elapsed().as_micros() as u64;
                    }
                    if std::env::var_os("PROXIMA_DEBUG_QWEN35_SEGMENTS").is_some() {
                        eprintln!(
                            "qwen35 segment phase=gather layer={} elapsed_us={}",
                            layer,
                            segment_started.elapsed().as_micros()
                        );
                    }
                    result?
                };
                if layer == 3
                    && is_router
                    && std::env::var_os("PROXIMA_DEBUG_DENSE_GRAPH").is_some()
                    && let Some(Op::Reduce(reduce)) = self.program.get(
                        self.qwen35moe_layer_diagnostics[layer]
                            .dense_attention_taps
                            .map_or(NodeId(u32::MAX), |taps| taps.q_split)
                            .0 as usize,
                    )
                    && let Some(Op::Elementwise { operands, .. }) =
                        self.program.get(reduce.operand.0 as usize)
                    && let Some((q_product, _)) = operands.first()
                    && let Some(mapped) = mapping.get(q_product)
                    && let Some((values, shape)) = evaluated.get(*mapped)
                {
                    eprintln!(
                        "dense_segment_qg_product mode={} layer=3 phase=router node={} mapped={} shape={shape:?} first4={:?}",
                        if position_offset > 0 {
                            "cached"
                        } else {
                            "scan"
                        },
                        q_product.0,
                        mapped.0,
                        values.iter().take(4).copied().collect::<Vec<_>>(),
                    );
                }
                #[cfg(feature = "instrument")]
                {
                    segment_execution_count += 1;
                    let readback_bytes = requested_nodes
                        .iter()
                        .filter_map(|node| evaluated.get(*node))
                        .map(|(values, _)| values.len() as u64 * core::mem::size_of::<f32>() as u64)
                        .sum::<u64>();
                    if is_router {
                        router_readback_bytes += readback_bytes;
                    } else {
                        gather_readback_bytes += readback_bytes;
                    }
                    debug!(
                        layer = layer as u64,
                        phase = if is_router { "router" } else { "gather" },
                        requested_nodes = requested_nodes.len() as u64,
                        readback_bytes,
                        "qwen35moe segment returned requested payload bytes"
                    );
                    if std::env::var_os("PROXIMA_DEBUG_QWEN35_REQUEST_BYTES").is_some() {
                        let mut returned = requested
                            .iter()
                            .filter_map(|(mapped, original)| {
                                evaluated.get(*mapped).map(|(values, shape)| {
                                    (
                                        original.0,
                                        mapped.0,
                                        values.len() as u64 * core::mem::size_of::<f32>() as u64,
                                        shape.to_vec(),
                                        program
                                            .get(mapped.0 as usize)
                                            .and_then(|operation| operation.name()),
                                    )
                                })
                            })
                            .collect::<Vec<_>>();
                        returned.sort_unstable_by_key(|entry| core::cmp::Reverse(entry.2));
                        eprintln!(
                            "qwen35 segment returned bytes phase={} layer={} nodes={:?}",
                            if is_router { "router" } else { "gather" },
                            layer,
                            returned,
                        );
                    }
                }
                if is_router || fused_gather {
                    let mapped_router = mapping.get(&diagnostic.router_logits).copied().ok_or(
                        InteropError::MissingEvaluatedNode {
                            node: diagnostic.router_logits,
                        },
                    )?;
                    let (router_logits, router_shape) =
                        evaluated
                            .get(mapped_router)
                            .ok_or(InteropError::MissingEvaluatedNode {
                                node: diagnostic.router_logits,
                            })?;
                    visit_qwen35moe_router_boundary(
                        phase_layer,
                        position_offset,
                        RouterLogits {
                            values: router_logits,
                            shape: router_shape,
                        },
                        RouterExpertCounts {
                            expert_count: self.architecture.expert_count as usize,
                            expert_used_count: self.architecture.expert_used_count as usize,
                        },
                        &mut routed_experts,
                        expert_slab,
                        &mut before_gather,
                    )?;
                }
                for (mapped, original) in requested {
                    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                    if plan.router_cut_placements.contains_key(&original) {
                        continue;
                    }
                    if evaluated.is_placed(mapped) {
                        // `finish` deliberately omits caller-owned output buffers from
                        // `Evaluated`; retain the original graph identity so the
                        // aggregate result preserves that placement contract.
                        placed_results.insert(original);
                        continue;
                    }
                    let (values, shape) = evaluated.get(mapped).ok_or_else(|| {
                        if std::env::var_os("PROXIMA_DEBUG_QWEN35_MISSING_NODE").is_some() {
                            #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                            let placed = plan.router_cut_placements.contains_key(&original);
                            #[cfg(not(all(feature = "metal-output-placement", target_os = "macos")))]
                            let placed = false;
                            eprintln!(
                                "qwen35 missing evaluated node phase={} original={:?} mapped={:?} placed={} op={:?}",
                                if is_router { "router" } else { "gather" },
                                original,
                                mapped,
                                placed,
                                self.program.get(original.0 as usize),
                            );
                        }
                        InteropError::MissingEvaluatedNode { node: original }
                    })?;
                    if future_cuts.iter().any(|(node, _)| *node == original)
                        || future_gather_cuts.contains(&original)
                        || plan.prefix_carried_nodes.contains(&original)
                        || plan.global_cut_nodes.contains(&original)
                        || original == segment_output
                    {
                        carried.insert(original, (shape.to_vec(), values.to_vec()));
                    }
                    if outputs.contains(&original) {
                        results.insert(original, (shape.to_vec(), values.to_vec()));
                    }
                }
                let output_id = mapping.get(&segment_output).copied().ok_or(
                    InteropError::MissingEvaluatedNode {
                        node: segment_output,
                    },
                )?;
                if let Some((values, shape)) = evaluated.get(output_id) {
                    if !is_router
                        && std::env::var_os("PROXIMA_DEBUG_GDN_BLOCK_OUTPUT").is_some()
                        && let (Some(selected_layer), Some(selected_position)) = (
                            std::env::var("PROXIMA_DEBUG_GDN_LAYER")
                                .ok()
                                .and_then(|value| value.parse::<usize>().ok()),
                            std::env::var("PROXIMA_DEBUG_GDN_POSITION")
                                .ok()
                                .and_then(|value| value.parse::<usize>().ok()),
                        )
                        && layer == selected_layer
                        && let Some(&rows) = shape.first()
                        && let Ok(rows) = usize::try_from(rows)
                        && rows > 0
                        && let Some(local_position) = selected_position.checked_sub(position_offset)
                        && local_position < rows
                    {
                        let row_length = values.len() / rows;
                        let row_start = local_position * row_length;
                        eprintln!(
                            "qwen35 block_output layer={} position={} node={:?} shape={:?} first4={:?}",
                            layer,
                            selected_position,
                            segment_output,
                            shape,
                            values[row_start..row_start + row_length]
                                .iter()
                                .take(4)
                                .copied()
                                .collect::<Vec<_>>(),
                        );
                    }
                    if !is_router && values.iter().any(|value| !value.is_finite()) {
                        let first_nonfinite = values
                            .iter()
                            .enumerate()
                            .find(|(_, value)| !value.is_finite())
                            .map(|(index, value)| (index, *value));
                        eprintln!(
                            "qwen35 nonfinite gather layer={layer} node={segment_output:?} first={first_nonfinite:?}"
                        );
                        return Err(InteropError::PreGatherExecutionUnsupported {
                            architecture: String::from("qwen35moe"),
                            reason: alloc::format!(
                                "layer {layer} expert gather produced a non-finite value"
                            ),
                        });
                    }
                    if std::env::var_os("PROXIMA_DEBUG_EXPERT_UPLOADS").is_some() {
                        let nan_count = values.iter().filter(|value| value.is_nan()).count();
                        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
                        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        eprintln!(
                            "qwen35 segment output layer={} phase={} node={:?} elements={} nan_count={} min={} max={} first={:?}",
                            layer,
                            if is_router { "router" } else { "gather" },
                            segment_output,
                            values.len(),
                            nan_count,
                            min,
                            max,
                            values.get(..values.len().min(4)).unwrap_or_default()
                        );
                    }
                    carried.insert(segment_output, (shape.to_vec(), values.to_vec()));
                }
            }
            #[cfg(unix)]
            if std::env::var("PROXIMA_EXPERT_SIDECAR_DISCARD_PER_LAYER")
                .ok()
                .as_deref()
                == Some("1")
                && let Some(sidecar) = &self.expert_sidecar
            {
                sidecar.discard_resident_pages()?;
            }
            // Sidecar payloads are an mmap, so leaving every low-codec page
            // resident turns a bounded device slab into an unbounded host
            // footprint over a long decode. Discard only the routes consumed
            // by this layer; `KEEP_PAGES` is an explicit diagnostic opt-out.
            #[cfg(all(feature = "metal", target_os = "macos"))]
            if std::env::var_os("PROXIMA_EXPERT_SIDECAR_KEEP_PAGES").is_none()
                && let Some(sidecar) = &self.expert_sidecar
            {
                let mut discarded = BTreeSet::new();
                for route in &routed_experts {
                    let address = crate::residency::ExpertAddress {
                        layer,
                        expert: route.expert,
                    };
                    if discarded.insert((address.layer, address.expert)) {
                        sidecar.discard_expert_low(address)?;
                    }
                }
            }
            #[cfg(all(feature = "metal", target_os = "macos"))]
            if std::env::var_os("PROXIMA_CHECKPOINT_DISCARD_PER_LAYER").is_some() {
                let discard = if std::env::var_os("PROXIMA_CHECKPOINT_DISCARD_PER_LAYER_IMMEDIATE")
                    .is_some()
                {
                    omega::discard_checkpoint_mmap_range_immediate
                } else {
                    omega::discard_checkpoint_mmap_range
                };
                discard(self.checkpoint_mapping).map_err(|error| {
                    InteropError::PreGatherExecutionUnsupported {
                        architecture: String::from("qwen35moe"),
                        reason: error.to_string(),
                    }
                })?;
            }
            let keep: BTreeSet<NodeId> = next_cuts.iter().map(|(node, _)| *node).collect();
            // A gather segment may carry an intermediate produced by the
            // preceding layer's gather rather than by its router. Retain
            // every such cut until its consumer layer instead of assuming
            // the next router cut list is complete.
            let keep: BTreeSet<NodeId> = keep
                .into_iter()
                .chain(segments.future_gather_cuts.iter().copied())
                .collect();
            // Keep only the handful of graph inputs that every layer may
            // reference, plus the explicit next-segment cuts. Expert stack
            // inputs are excluded above and can never enter this carry set.
            carried.retain(|node, _| {
                plan.prefix_carried_nodes.contains(node)
                    || keep.contains(node)
                    || matches!(
                        self.program[node.0 as usize],
                        proxima_tensor::op::Op::Constant { .. }
                    )
            });
            layer += 1;
        }

        let (suffix_program, suffix_cuts, suffix_mapping) = &plan.suffix;
        let mut requested = BTreeMap::new();
        for &node in outputs {
            if results.contains_key(&node) || placed_results.contains(&node) {
                continue;
            }
            if let Some(mapped) = suffix_mapping.get(&node).copied() {
                requested.insert(mapped, node);
            }
        }
        let suffix_executed = !requested.is_empty();
        if suffix_executed {
            let mut suffix_named: Vec<(&str, QuantizedBlock<'_>)> = named
                .iter()
                .copied()
                .filter(|(name, _)| {
                    suffix_program
                        .iter()
                        .any(|operation| operation.name() == Some(*name))
                })
                .collect();
            for (node, name) in suffix_cuts {
                if suffix_named.iter().any(|(candidate, _)| *candidate == name) {
                    continue;
                }
                let (_, values) = carried.get(node).ok_or_else(|| {
                    InteropError::PreGatherExecutionUnsupported {
                        architecture: String::from("qwen35moe"),
                        reason: alloc::format!("suffix missing cut node {node:?} ({name})"),
                    }
                })?;
                suffix_named.push((name.as_str(), QuantizedBlock::Float32(values)));
            }
            let requested_nodes: Vec<NodeId> = requested.keys().copied().collect();
            let evaluated = runtime.evaluate_segment(
                suffix_program,
                symbols,
                &suffix_named,
                &requested_nodes,
                resident_names,
                None,
            )?;
            #[cfg(feature = "instrument")]
            {
                segment_execution_count += 1;
            }
            for (mapped, original) in requested {
                let (values, shape) = evaluated.get(mapped).ok_or_else(|| {
                    if std::env::var_os("PROXIMA_DEBUG_QWEN35_MISSING_NODE").is_some() {
                        eprintln!(
                            "qwen35 missing evaluated node phase=suffix node={:?} mapped={:?}",
                            original, mapped
                        );
                    }
                    InteropError::MissingEvaluatedNode { node: original }
                })?;
                results.insert(original, (shape.to_vec(), values.to_vec()));
            }
        }
        #[cfg(feature = "instrument")]
        {
            debug!(
                position_offset = position_offset as u64,
                layer_count = plan.layers.len() as u64,
                segment_execution_count,
                suffix_executed,
                router_elapsed_us,
                gather_elapsed_us,
                router_readback_bytes,
                gather_readback_bytes,
                "qwen35moe pre-gather segment census recorded after requested outputs completed"
            );
            if std::env::var_os("PROXIMA_DEBUG_QWEN35_SEGMENTS").is_some() {
                eprintln!(
                    "qwen35 segment summary position={} layers={} segments={} suffix_executed={} router_elapsed_us={} gather_elapsed_us={} router_readback_bytes={} gather_readback_bytes={}",
                    position_offset,
                    plan.layers.len(),
                    segment_execution_count,
                    suffix_executed,
                    router_elapsed_us,
                    gather_elapsed_us,
                    router_readback_bytes,
                    gather_readback_bytes,
                );
            }
        }

        let ordered_results = outputs
            .iter()
            .filter(|node| !placed_results.contains(node))
            .map(|node| {
                results
                    .remove(node)
                    .map(|(shape, values)| (*node, shape, values))
                    .ok_or(InteropError::MissingEvaluatedNode { node: *node })
            })
            .collect::<Result<Vec<_>, _>>()?;
        #[cfg(unix)]
        if std::env::var("PROXIMA_EXPERT_SIDECAR_DISCARD")
            .ok()
            .as_deref()
            == Some("1")
            && let Some(sidecar) = &self.expert_sidecar
        {
            sidecar.discard_resident_pages()?;
        }
        Ok(Evaluated::from_parts_with_placed(
            self.logits_root,
            ordered_results,
            None,
            placed_results,
        ))
    }

    /// Graph-level producer boundaries for every qwen35moe layer, in layer
    /// order. Non-qwen35moe models return an empty slice.
    #[must_use]
    pub fn qwen35moe_layer_diagnostics(&self) -> &[crate::qwen35moe::Qwen35MoeLayerDiagnostics] {
        &self.qwen35moe_layer_diagnostics
    }

    /// This checkpoint's own transformer block count
    /// (`{architecture}.block_count`, [`ModelArchitecture::block_count`]).
    #[must_use]
    pub fn layer_count(&self) -> u32 {
        self.architecture.block_count
    }

    /// The checkpoint file's own byte length at load time (`file_bytes.len()`
    /// passed to [`Self::load`]/[`Self::load_from_safetensors`]) -- the
    /// on-disk size a live indicator reports, not this call's resident
    /// memory footprint (weights may be memory-mapped rather than copied;
    /// see `crate::bind::bind_all_weights`'s own doc for which tensors are
    /// borrowed versus owned).
    #[must_use]
    pub fn checkpoint_bytes(&self) -> usize {
        self.checkpoint_bytes
    }

    /// One post-layer residual root per dense layer
    /// (`crate::architecture::BoundProgram::residual_roots`'s own doc) --
    /// empty on the qwen35 hybrid path, which exposes no single per-layer
    /// residual node. `examples/compare_local.rs`/`examples/embed_local.rs`
    /// walk this to compare CPU/GPU at the first divergent layer instead of
    /// guessing [`NodeId`] arithmetic.
    #[must_use]
    pub fn layer_residual_roots(&self) -> &[NodeId] {
        &self.residual_roots
    }

    /// Every [`NodeId`] the program needs to evaluate `through`, inclusive.
    /// `self.program` is a flat, topologically-ordered slice (`op::append`'s
    /// own doc: "the last element is the root", by construction backwards-
    /// only), so the prefix `0..=through.0` is exactly that set -- no graph
    /// walk required.
    #[must_use]
    pub fn computed_node_ids_through(&self, through: NodeId) -> Vec<NodeId> {
        (0..=through.0).map(NodeId).collect()
    }

    /// This node's op kind (`Op::kind`), for diagnostics that print a
    /// divergent node without matching on the whole [`Op`] enum.
    #[must_use]
    pub fn node_kind(&self, node: NodeId) -> &'static str {
        self.program[node.0 as usize].kind()
    }

    /// This node's identity name (`Op::name`), when the underlying op
    /// carries one -- `None` for a computed leaf (`Op::Iota`/`Op::Constant`)
    /// or an unnamed intermediate.
    #[must_use]
    pub fn node_name(&self, node: NodeId) -> Option<&str> {
        self.program[node.0 as usize].name()
    }

    /// A full `Debug` rendering of this node's op, for a diagnostic that
    /// needs more than [`Self::node_kind`]/[`Self::node_name`] to identify
    /// which computation produced a divergent value.
    #[must_use]
    pub fn node_description(&self, node: NodeId) -> String {
        alloc::format!("{:?}", self.program[node.0 as usize])
    }

    /// This node's operand [`NodeId`]s (`Op::dependencies`), in the order
    /// the op itself addresses them.
    #[must_use]
    pub fn node_dependencies(&self, node: NodeId) -> Vec<NodeId> {
        self.program[node.0 as usize].dependencies()
    }

    /// Binds every weight the cached forward program needs out of
    /// `parsed`/`file_bytes` (`crate::bind::bind_all_weights`), derives
    /// [`ModelArchitecture`] from `parsed`'s own metadata
    /// ([`crate::bind::architecture_from_metadata`]), builds the vocab
    /// from the same metadata, and compiles the cached forward program
    /// once. Pays the whole load cost; every [`Pipe::call`] after reuses
    /// the result.
    ///
    /// # Errors
    ///
    /// Whatever [`crate::bind::architecture_from_metadata`],
    /// [`proxima_tokenizer::gguf::vocab_from_metadata`], or
    /// [`proxima_tensor::spec::mistral_cached_forward_program_with_experts`]
    /// can fail with.
    pub fn load(parsed: &ParsedGguf, file_bytes: &'file [u8]) -> Result<Self, InteropError> {
        Self::load_with_registry(
            parsed,
            file_bytes,
            &crate::architecture::ArchitectureRegistry::with_builtin(),
        )
    }

    /// [`Self::load`] with the [`crate::architecture::ArchitectureRegistry`]
    /// caller-supplied rather than fixed to
    /// [`crate::architecture::ArchitectureRegistry::with_builtin`] --
    /// the seam a foreign `Architecture` (registered against its own
    /// registry, never against this crate's) loads a checkpoint through,
    /// the same way [`crate::architecture::ArchitectureRegistry::resolve`]'s
    /// own doc describes. [`Self::load`] is this call with the builtin
    /// table, unchanged.
    ///
    /// # Errors
    ///
    /// Same as [`Self::load`].
    pub fn load_with_registry(
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
        registry: &crate::architecture::ArchitectureRegistry,
    ) -> Result<Self, InteropError> {
        Self::load_inner(parsed, file_bytes, false, false, registry)
    }

    /// [`Self::load`] with the paired gate/up reduce
    /// (`proxima_tensor::spec::append_mistral_cached_layer`'s
    /// `paired_gate_up_reduce`) flipped on: one `Op::Reduce` per layer over
    /// `blk.{layer}.ffn_gate_up.weight` (`crate::bind::bind_matmul_weight_paired`)
    /// in place of today's two independent `ffn_gate`/`ffn_up` matvecs.
    /// `false` at [`Self::load`] reproduces this crate's forward program
    /// byte-for-byte; this constructor is the seam a caller (or a future
    /// [`crate::serving::ServingConfig`] field) flips to measure the other
    /// side. No effect on a `qwen35` checkpoint (that branch never reads
    /// this flag) or a mixture-of-experts checkpoint (routed FFN weights are
    /// untouched by this flag either way).
    ///
    /// # Errors
    ///
    /// Same as [`Self::load`].
    pub fn load_with_paired_gate_up_reduce(
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
        paired_gate_up_reduce: bool,
    ) -> Result<Self, InteropError> {
        Self::load_inner(
            parsed,
            file_bytes,
            paired_gate_up_reduce,
            false,
            &crate::architecture::ArchitectureRegistry::with_builtin(),
        )
    }

    /// [`Self::load`] with the fused Q/K/V reduce
    /// (`proxima_tensor::spec::append_mistral_cached_layer`'s
    /// `fused_qkv_reduce`) flipped on: one `Op::Reduce` per layer over
    /// `blk.{layer}.attn_qkv.weight` (`crate::bind::bind_matmul_weight_triple`)
    /// in place of today's three independent `attn_q`/`attn_k`/`attn_v`
    /// matvecs. Unlike [`Self::load_with_paired_gate_up_reduce`], this does
    /// NOT remove dispatches -- q/k/v's three different GQA row counts mean
    /// none of them can be read back out of the shared reduce at zero
    /// extra cost (`append_mistral_cached_layer`'s `fused_qkv_reduce` doc
    /// traces the exact `shape::infer` limit this hits), so the measured
    /// effect, if any, is per-dispatch bandwidth on the one larger reduce,
    /// not fewer kernel launches. No effect on a `qwen35` checkpoint or a
    /// mixture-of-experts checkpoint, same carve-outs as the paired flag.
    ///
    /// # Errors
    ///
    /// Same as [`Self::load`].
    pub fn load_with_fused_qkv_reduce(
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
        fused_qkv_reduce: bool,
    ) -> Result<Self, InteropError> {
        Self::load_inner(
            parsed,
            file_bytes,
            false,
            fused_qkv_reduce,
            &crate::architecture::ArchitectureRegistry::with_builtin(),
        )
    }

    pub(super) fn load_inner(
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
        paired_gate_up_reduce: bool,
        fused_qkv_reduce: bool,
        registry: &crate::architecture::ArchitectureRegistry,
    ) -> Result<Self, InteropError> {
        // ROW 533's own mechanism (`proxima-tensor/docs/discipline.md`): a
        // non-resident page behind the no-copy `MTLBuffer`
        // `register_checkpoint_mapping` installs below reads as zero rather
        // than faulting, silently. `crate::mapping_residency::prove_resident`
        // is the gate that turns that into a typed refusal instead --
        // `PROXIMA_MAPPING_FIT_OVERRIDE=1` skips only the size-vs-host-limit
        // half of it, for a measurement run.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            if std::env::var_os("PROXIMA_MAPPING_FIT_OVERRIDE").is_none()
                && let Ok(facts) = omega::metal::system_memory_facts()
            {
                let limit = crate::memory_fit::HostMemoryLimit {
                    limit_bytes: facts
                        .recommended_max_working_set_size
                        .min(facts.physical_memory_bytes),
                    os_headroom_bytes: omega::sized::LOAD_TIME_FIT_OS_HEADROOM_BYTES,
                };
                crate::memory_fit::fit_mapping_bytes(file_bytes.len() as u64, limit)?;
            }
            let residency_report = crate::mapping_residency::prove_resident(file_bytes)?;
            #[cfg(feature = "instrument")]
            info!(
                mapping_prefault_ms = residency_report.prefault_ms,
                mapping_resident_pages = residency_report.resident_pages,
                mapping_missing_pages = residency_report.missing_pages,
                "checkpoint_mapping_residency: proved resident before metal no-copy registration"
            );
        }
        // registers `file_bytes` -- the checkpoint's own mmap, page-aligned
        // at its base by construction -- as the single mapping every packed
        // tensor's borrowed slice can be addressed into by OFFSET instead of
        // copied into its own device buffer; see
        // `omega::metal::register_checkpoint_mapping`'s own doc. `omega` is
        // an optional dependency gated behind the `metal` feature, so this
        // registration is a no-op (compiled out) when that feature is off.
        #[cfg(feature = "metal")]
        omega::backend::register_checkpoint_mapping(file_bytes);
        // `crate::memory_fit`'s own load-time gate needs these three sums
        // long after `parsed` has gone out of scope -- computed once, here,
        // before any weight is bound, shared by both the qwen35 and dense
        // branches below.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let (dense_weight_bytes, expert_weight_bytes, table_weight_bytes) =
            crate::bind::tensor_bytes_by_class(parsed);
        // The common case resolves the checkpoint's own `general.architecture`
        // against the registered `Architecture` table and lets that impl's
        // own `bind` do everything `load_inner`'s qwen35/dense arms used to
        // assemble by hand -- see `crate::architecture`'s own module doc
        // for why this seam exists. `paired_gate_up_reduce`/`fused_qkv_reduce`
        // are per-call diagnostic knobs `Architecture::bind`'s fixed
        // signature does not carry (`crate::dense::DenseArch`'s own doc on
        // why), and (documented on both flag-carrying constructors) have
        // "no effect on a qwen35 checkpoint" -- so a qwen35 checkpoint
        // always takes this registry path regardless of either flag, and
        // only a non-qwen35 checkpoint with a flag set falls through to
        // the narrow inline path below.
        let general_architecture = crate::bind::metadata_str(parsed, "general.architecture")?;
        if std::env::var_os("PROXIMA_DEBUG_ARCH_ROUTE").is_some() {
            eprintln!(
                "architecture route value={general_architecture:?} qwen35moe={} flags=({}, {})",
                general_architecture == "qwen35moe",
                paired_gate_up_reduce,
                fused_qkv_reduce,
            );
        }
        if matches!(general_architecture, "qwen35" | "qwen35moe")
            || (!paired_gate_up_reduce && !fused_qkv_reduce)
        {
            let resolved = registry.resolve(parsed)?;
            let bound = resolved.bind(parsed, file_bytes)?;
            #[cfg(all(feature = "metal", target_os = "macos"))]
            let step_state = resolved.step_state(parsed)?;
            let vocab = proxima_tokenizer::gguf::vocab_from_metadata(parsed)?;
            // The single-range program is dense-Mistral-only
            // (`SingleRangeProgram`'s own field doc): never built for
            // qwen35's hybrid attention+state-space layers, and
            // `build_single_range_program` itself already turns away any
            // mixture-of-experts checkpoint.
            #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
            let single_range = if resolved.name() == "qwen35" {
                None
            } else {
                let qk_norm = crate::bind::checkpoint_has_qk_norm(parsed);
                build_single_range_program(&bound.architecture, qk_norm)?
            };
            let bound = bound;
            let expert_slab =
                crate::bind::build_expert_slab(&bound.architecture, &bound.program, &bound.weights);
            return Self {
                expert_slab: std::sync::Mutex::new(expert_slab),
                expert_sidecar: None,
                weights: bound.weights,
                architecture: bound.architecture,
                architecture_impl: Some(resolved),
                #[cfg(all(feature = "metal", target_os = "macos"))]
                checkpoint_weight_bytes: crate::memory_fit::WeightClassBytes {
                    dense_bytes: dense_weight_bytes,
                    expert_bytes: expert_weight_bytes,
                    table_bytes: table_weight_bytes,
                    ssm_state_bytes: step_state.as_ref().map_or(0, |state| state.ssm_state_bytes),
                },
                vocab,
                program: bound.program,
                logits_root: bound.logits_root,
                hidden_root: bound.hidden_root,
                layer_roots: bound.layer_roots,
                residual_roots: bound.residual_roots,
                qwen35moe_layer_diagnostics: bound.qwen35moe_layer_diagnostics,
                router_roots: bound.router_roots,
                moe_sites: bound.moe_sites,
                single_position_step: bound.single_position_step,
                qwen35moe_hparams: (resolved.name() == "qwen35moe")
                    .then(|| crate::qwen35moe::hparams::from_metadata(parsed).ok())
                    .flatten(),
                model_name: crate::bind::metadata_str_opt(parsed, "general.name").map(String::from),
                checkpoint_bytes: file_bytes.len(),
                #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
                single_range,
                checkpoint_mapping: file_bytes,
            }
            .validated();
        }

        let architecture = architecture_from_metadata(parsed)?;
        let vocab = proxima_tokenizer::gguf::vocab_from_metadata(parsed)?;
        // `&[]`: `Self::load`/`load_with_*` take no `ServingConfig`, so
        // there is no `weight_precision` rule set to thread here yet --
        // `crate::bind::bind_all_weights`'s own doc names this as the
        // wiring a future slice does.
        let weights = bind_all_weights(
            parsed,
            file_bytes,
            &architecture,
            paired_gate_up_reduce,
            fused_qkv_reduce,
            &[],
        )?;
        // `architecture.expert_count`/`expert_used_count` read `0` for every
        // dense checkpoint (`ModelArchitecture`'s own doc), which selects
        // exactly the dense program this crate has always built -- a
        // mixture-of-experts checkpoint (`expert_count > 0`) is the only case
        // that changes which program gets compiled here. `qk_norm` is Qwen3's
        // own per-head QK-norm (`crate::bind::checkpoint_has_qk_norm`'s own
        // doc) -- `false` reproduces the identical program this call has
        // always compiled for a checkpoint that carries no
        // `attn_q_norm.weight` tensor.
        let qk_norm = crate::bind::checkpoint_has_qk_norm(parsed);
        let (program, forward_roots, cache_roots, layer_residuals, moe_sites) =
            mistral_cached_forward_program_with_experts_and_layer_taps(
                architecture.vocab,
                architecture.embedding,
                architecture.feed_forward,
                architecture.query_heads,
                architecture.kv_heads,
                architecture.head_dim,
                architecture.block_count,
                architecture.expert_count,
                architecture.expert_used_count,
                qk_norm,
                false,
                paired_gate_up_reduce,
                fused_qkv_reduce,
                true,
            )?;
        let logits_root = forward_roots.logits;
        // `mistral_single_range_cached_forward_program`'s own `w_gate`/`w_up`/
        // `wq`/`wk`/`wv` leaves (`build_single_range_program`) do not know
        // about `paired_gate_up_reduce`/`fused_qkv_reduce` yet --
        // `LoadedModel::run_decode_loop_observed`'s placed-KV fast path
        // would try to read `blk.{layer}.ffn_gate.weight`/
        // `blk.{layer}.attn_q.weight` against a `weights` set that, under
        // either flag, binds a differently-named fused tensor instead.
        // Forcing `None` here falls through to the two-range decode loop
        // below, which DOES thread both flags correctly, rather than a
        // `Metal(Tensor(UnboundInputName(..)))` panic -- the correct,
        // fusion-aware path over a crash, until the placed-KV builder
        // gains both flags (tracked, not done in this change: threading
        // `fused_qkv_reduce` through `mistral_single_range_cached_forward_program`
        // is a second builder needing the identical q/k/v leaf and
        // extract-op rewrite `append_mistral_cached_layer` just got, and is
        // out of this change's scope).
        //
        // `qk_norm` (ROW 373): `append_mistral_single_range_cached_layer`
        // (`proxima-tensor/src/spec.rs`) now takes the SAME
        // `Option<(NodeId, NodeId, NodeId)>` shape its two-range sibling
        // does and derives its RoPE pairing from `qk_norm.is_some()` the
        // same way -- a qk-norm checkpoint (Qwen3) now takes this
        // placed-KV fast path on Metal instead of falling back to the
        // two-range program (`build_single_range_program` no longer turns
        // a qk-norm request into `Ok(None)`; ROW 372's typed rejection is
        // gone from this builder). Before ROW 372, the same checkpoint
        // class silently ran attention on raw, un-normed Q/K with the wrong
        // RoPE pairing through this exact path -- no error, just a
        // structurally different (wrong) computation. ROW 372 made the
        // wrong path loud (reject); this row makes the right path fast
        // (build it correctly instead).
        #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
        let single_range = if paired_gate_up_reduce || fused_qkv_reduce {
            None
        } else {
            build_single_range_program(&architecture, qk_norm)?
        };
        let expert_slab = crate::bind::build_expert_slab(&architecture, &program, &weights);
        Self {
            expert_slab: std::sync::Mutex::new(expert_slab),
            expert_sidecar: None,
            weights,
            architecture,
            architecture_impl: None,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            checkpoint_weight_bytes: crate::memory_fit::WeightClassBytes {
                dense_bytes: dense_weight_bytes,
                expert_bytes: expert_weight_bytes,
                table_bytes: table_weight_bytes,
                ssm_state_bytes: 0,
            },
            vocab,
            program,
            logits_root,
            hidden_root: Some(forward_roots.hidden),
            layer_roots: cache_roots
                .into_iter()
                .map(Qwen35LayerRoots::Attention)
                .collect(),
            residual_roots: layer_residuals,
            qwen35moe_layer_diagnostics: Vec::new(),
            router_roots: Vec::new(),
            moe_sites,
            single_position_step: false,
            qwen35moe_hparams: None,
            model_name: crate::bind::metadata_str_opt(parsed, "general.name").map(String::from),
            checkpoint_bytes: file_bytes.len(),
            #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
            single_range,
            checkpoint_mapping: file_bytes,
        }
        .validated()
    }

    /// [`Self::load`]'s HF/safetensors counterpart: binds every weight out
    /// of a single safetensors buffer's [`proxima_safetensors::Manifest`]
    /// (`crate::hf_bind::bind_all_weights_from_safetensors`) instead of a
    /// `ParsedGguf` tensor directory, and takes `architecture`/`vocab`
    /// already built rather than deriving them from the checkpoint itself --
    /// unlike GGUF, safetensors carries neither: `architecture` comes from
    /// `config.json` ([`crate::hf_config::architecture_from_hf_config`]),
    /// and HF's own vocabulary lives in `tokenizer.json`/`tokenizer_config.json`,
    /// files this crate has no reader for yet (out of scope for this
    /// change -- a caller builds its own [`Vocab`] however it can, the same
    /// way any [`Pipe`] caller owns its own setup-path inputs).
    ///
    /// `data_start` (`8 + header_len`) is the byte offset into `file_bytes`
    /// where tensor data begins -- `manifest`'s own `data_offsets` are
    /// relative to that point, never to the start of the file (see
    /// `crate::hf_bind::bind_all_weights_from_safetensors`'s doc); a
    /// caller who just parsed `file_bytes`'s header into `manifest` already
    /// has this value.
    ///
    /// # Errors
    ///
    /// [`InteropError::HfMoeWeightsUnsupported`] if `architecture.expert_count`
    /// is nonzero; otherwise whatever
    /// `crate::hf_bind::bind_all_weights_from_safetensors` or
    /// [`mistral_cached_forward_program_with_experts`] can fail with.
    pub fn load_from_safetensors(
        manifest: &proxima_safetensors::Manifest,
        file_bytes: &'file [u8],
        data_start: u64,
        architecture: ModelArchitecture,
        vocab: Vocab,
    ) -> Result<Self, InteropError> {
        let weights =
            bind_all_weights_from_safetensors(manifest, file_bytes, data_start, &architecture)?;
        // safetensors carries no GGUF tensor directory to probe for
        // `attn_q_norm.weight`, and no HF/safetensors checkpoint this crate
        // binds today needs QK-norm -- see [`Self::load`]'s own `qk_norm` for
        // the GGUF path that does.
        let (program, forward_roots, cache_roots, layer_residuals, moe_sites) =
            mistral_cached_forward_program_with_experts_and_layer_taps(
                architecture.vocab,
                architecture.embedding,
                architecture.feed_forward,
                architecture.query_heads,
                architecture.kv_heads,
                architecture.head_dim,
                architecture.block_count,
                architecture.expert_count,
                architecture.expert_used_count,
                false,
                false,
                false,
                false,
                true,
            )?;
        let logits_root = forward_roots.logits;
        #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
        let single_range = build_single_range_program(&architecture, false)?;
        let expert_slab = crate::bind::build_expert_slab(&architecture, &program, &weights);
        Self {
            expert_slab: std::sync::Mutex::new(expert_slab),
            expert_sidecar: None,
            weights,
            architecture,
            architecture_impl: None,
            // safetensors carries no `_exps.`-style naming convention this
            // crate has confirmed against a real checkpoint the way
            // `crate::bind::tensor_bytes_by_class` has for GGUF -- every
            // byte counts as dense here rather than guessing a split;
            // `expert_bytes`/`table_bytes` stay `0` until a real HF MoE
            // checkpoint proves what its own expert-tensor names look like.
            #[cfg(all(feature = "metal", target_os = "macos"))]
            checkpoint_weight_bytes: crate::memory_fit::WeightClassBytes {
                dense_bytes: file_bytes.len() as u64,
                expert_bytes: 0,
                table_bytes: 0,
                ssm_state_bytes: 0,
            },
            vocab,
            program,
            logits_root,
            hidden_root: Some(forward_roots.hidden),
            layer_roots: cache_roots
                .into_iter()
                .map(Qwen35LayerRoots::Attention)
                .collect(),
            residual_roots: layer_residuals,
            qwen35moe_layer_diagnostics: Vec::new(),
            router_roots: Vec::new(),
            moe_sites,
            single_position_step: false,
            qwen35moe_hparams: None,
            // safetensors carries no `general.name`-equivalent key this
            // crate reads (`Self::model_name`'s own doc).
            model_name: None,
            checkpoint_bytes: file_bytes.len(),
            #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
            single_range,
            checkpoint_mapping: file_bytes,
        }
        .validated()
    }
}

pub(crate) enum LogitsSink<'sink> {
    Discard,
    Collect(&'sink mut Vec<Vec<f32>>),
    /// `real_openchat_file::decode_text_is_deterministic_across_repeated_runs`'s
    /// own sink: that test is `#[cfg(feature = "metal")]`, so this variant
    /// carries the same gate (plus `test`) rather than sitting dead in a
    /// `std`-only test build.
    #[cfg(all(test, feature = "metal"))]
    SumBarriers(&'sink mut u64),
}

