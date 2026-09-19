#[cfg(all(test, feature = "instrument", feature = "metal", target_os = "macos"))]
use super::phys_footprint_bytes;
#[cfg(all(test, feature = "metal"))]
use super::{
    BackendRuntime, InteropError, LogitsSink, NodeValuesSink, PrefixState, ServingConfig,
    map_expert_sources_to_segment, supported_serving_config, wants_bos,
};
#[cfg(test)]
use core::ops::ControlFlow;

#[cfg(test)]
use super::{
    DecodeMetrics, LoadedModel, Phase, RouterExpertCounts, RouterLogits, SsmLayerCache, TokenEvent,
    build_position_inputs, collect_future_gather_cuts, decode_until_stop_or_budget,
    first_nonfinite_node_value, kv_extent, lock_expert_slab, qwen35moe_admit_low_copy,
    qwen35moe_monolithic_all_low_enabled, qwen35moe_pre_gather_enabled,
    should_release_monolithic_sources, step_batch_needs_logits, visit_qwen35moe_router_boundary,
    visit_qwen35moe_router_selections,
};
#[cfg(all(test, feature = "metal-output-placement", target_os = "macos"))]
use super::{
    qwen35_dense_attention_placed_byte_length, qwen35_dense_attention_placement_enabled,
    retain_qwen35_segment_readbacks, use_metal_output_placements,
};
// Reads as unused under an explicit `--features std,metal` single-crate
// build (every reference below is the fully-qualified `super::PlanNumerics`,
// never the bare name this import binds), but dropping it breaks `cargo
// check --workspace --all-targets`: `super::PlanNumerics` stops resolving
// there ("not found in `super`") at every one of this module's 3 call
// sites. Cargo's per-target feature unification, not this file, decides
// whether `residency_caches`'s `pub(super) struct PlanNumerics` reaches
// `generate`'s glob re-export (`mod.rs`'s `pub use residency_caches::*;`)
// before this module compiles, and the workspace-wide pass resolves that
// differently than a single-crate `--features` invocation does -- keeping
// the explicit import is the one form both builds agree on.
#[cfg(all(test, feature = "metal-output-placement", target_os = "macos"))]
#[allow(unused_imports)]
use super::PlanNumerics;

#[cfg(all(test, feature = "std"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
pub(super) mod tests {
    #[cfg(feature = "qwen35moe-expert-prefetch")]
    use super::super::qwen35moe_expert_prefetch_requested;
    use super::{
        RouterExpertCounts, RouterLogits, SsmLayerCache, collect_future_gather_cuts,
        first_nonfinite_node_value, kv_extent, lock_expert_slab, qwen35moe_admit_low_copy,
        qwen35moe_monolithic_all_low_enabled, qwen35moe_pre_gather_enabled,
        should_release_monolithic_sources, step_batch_needs_logits,
        visit_qwen35moe_router_boundary, visit_qwen35moe_router_selections,
    };
    #[cfg(all(feature = "metal", target_os = "macos"))]
    use super::{map_expert_sources_to_segment, use_metal_output_placements};
    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
    use super::{
        qwen35_dense_attention_placed_byte_length, qwen35_dense_attention_placement_enabled,
        retain_qwen35_segment_readbacks,
    };
    use crate::bind::Codec;
    use alloc::string::String;
    use alloc::vec::Vec;

    #[cfg(any(feature = "metal", feature = "metal-output-placement"))]
    use alloc::collections::BTreeMap;

    #[cfg(all(feature = "metal", target_os = "macos"))]
    use alloc::collections::BTreeSet;
    #[cfg(all(feature = "metal", target_os = "macos"))]
    use proxima_gguf::value::MetadataArray;
    use proxima_gguf::value::MetadataValue as Value;
    use proxima_gguf::{GgmlType as WireType, GgufModel, TensorPayload, write_complete};
    use proxima_tensor::NodeId;
    use proxima_tensor::cpu::Evaluated;

    #[cfg(feature = "qwen35moe-expert-prefetch")]
    #[test]
    fn expert_prefetch_gate_requires_an_explicit_truthy_value() {
        assert!(!qwen35moe_expert_prefetch_requested(false));
        assert!(qwen35moe_expert_prefetch_requested(true));
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    use proxima_tensor::cpu::{ExpertEntry, ExpertSource, QuantizedBlock};
    #[cfg(all(feature = "metal", target_os = "macos"))]
    use proxima_tensor::{DType, Extent, Op};
    use proxima_tokenizer::Vocab;

    #[test]
    fn first_nonfinite_node_value_reports_program_order_and_payload_location() {
        let evaluated = Evaluated::from_parts(
            NodeId(9),
            vec![
                (NodeId(9), vec![2], vec![1.0, f32::INFINITY]),
                (NodeId(4), vec![1, 2], vec![f32::NAN, 3.0]),
            ],
            None,
        );

        let found = first_nonfinite_node_value(&evaluated, &[NodeId(4), NodeId(9)])
            .expect("the first requested non-finite payload is reported");

        assert_eq!(found.node, NodeId(4));
        assert_eq!(found.index, 0);
        assert!(found.value.is_nan());
        assert_eq!(found.shape, [1, 2]);
    }

    #[test]
    fn qwen35moe_pre_gather_admission_depends_only_on_config_and_architecture() {
        assert!(qwen35moe_pre_gather_enabled(true, Some("qwen35moe")));
        assert!(!qwen35moe_pre_gather_enabled(false, Some("qwen35moe")));
        assert!(!qwen35moe_pre_gather_enabled(true, Some("qwen3")));
        assert!(!qwen35moe_pre_gather_enabled(true, None));
    }

    #[test]
    fn qwen35moe_low_copy_admission_requires_byte_preserving_codec() {
        assert!(qwen35moe_admit_low_copy(Codec::Q4K, Codec::Q4K));
        assert!(qwen35moe_admit_low_copy(Codec::Q6K, Codec::Q6K));
        assert!(!qwen35moe_admit_low_copy(Codec::Q4K, Codec::Q3K));
        assert!(!qwen35moe_admit_low_copy(Codec::Q6K, Codec::Q2K));
    }

    #[test]
    fn linked_pre_gather_schedule_keeps_later_gathers_for_two_layers() {
        let gather_cuts = vec![
            vec![(NodeId(10), String::from("layer_zero_gather"))],
            vec![
                (NodeId(20), String::from("layer_one_kv")),
                (NodeId(21), String::from("layer_one_ssm")),
            ],
        ];

        assert_eq!(
            collect_future_gather_cuts(0, &gather_cuts),
            [NodeId(20), NodeId(21)]
        );
        assert!(collect_future_gather_cuts(1, &gather_cuts).is_empty());
    }

    #[test]
    fn qwen35moe_monolithic_all_low_is_default_off_and_gpu_only() {
        assert!(!qwen35moe_monolithic_all_low_enabled(true, true, false, 0));
        assert!(!qwen35moe_monolithic_all_low_enabled(false, true, true, 0));
        assert!(!qwen35moe_monolithic_all_low_enabled(true, false, true, 0));
        assert!(qwen35moe_monolithic_all_low_enabled(true, true, true, 0));
        assert!(qwen35moe_monolithic_all_low_enabled(true, true, true, 1));
    }

    #[test]
    fn monolithic_sources_release_only_at_decode_scope_exit() {
        assert!(!should_release_monolithic_sources(false, false));
        assert!(!should_release_monolithic_sources(true, true));
        assert!(should_release_monolithic_sources(true, false));
    }

    /// ROW 549 fix: recurrent state (`ssm_input_placements` non-empty) always
    /// requests the placed executor now, whether or not the same step also
    /// carries routed-expert substitutions -- qwen35moe's hybrid GDN+MoE
    /// decode step needs both together, and `evaluate_with_placements`
    /// already threads `expert_sources` through
    /// (`execute_plan_named_with_placements_and_expert_sources`).
    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
    #[test]
    fn recurrent_state_always_requests_placed_execution() {
        assert!(!use_metal_output_placements(false, false));
        assert!(use_metal_output_placements(true, false));
        assert!(use_metal_output_placements(false, true));
        assert!(use_metal_output_placements(true, true));
    }

    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
    #[test]
    fn qwen35_dense_attention_placement_is_qwen35moe_metal_only() {
        assert!(qwen35_dense_attention_placement_enabled(
            true, true, false, 0, true
        ));
        assert!(!qwen35_dense_attention_placement_enabled(
            false, true, false, 0, true
        ));
        assert!(!qwen35_dense_attention_placement_enabled(
            true, false, false, 0, true
        ));
        assert!(!qwen35_dense_attention_placement_enabled(
            true, true, true, 0, true
        ));
        assert!(!qwen35_dense_attention_placement_enabled(
            true, true, false, 1, true
        ));
        assert!(!qwen35_dense_attention_placement_enabled(
            true, true, false, 0, false
        ));
    }

    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
    #[test]
    fn qwen35_segment_readback_omits_only_placed_values() {
        let dense_roots = (NodeId(10), NodeId(11), NodeId(12), NodeId(13));
        let router_cut_placements = BTreeMap::from([(NodeId(20), ())]);
        let original = BTreeMap::from([
            (NodeId(1), NodeId(10)),
            (NodeId(2), NodeId(20)),
            (NodeId(3), NodeId(30)),
        ]);
        let computed = Op::Constant {
            dtype: DType::Float32,
            shape: Vec::new(),
            value: 0.0,
        };
        let router_program = vec![
            computed.clone(),
            computed.clone(),
            computed.clone(),
            computed,
        ];
        let gather_program = vec![
            Op::Constant {
                dtype: DType::Float32,
                shape: Vec::new(),
                value: 0.0,
            },
            Op::Constant {
                dtype: DType::Float32,
                shape: Vec::new(),
                value: 0.0,
            },
            Op::Input {
                dtype: DType::Float32,
                shape: Vec::new(),
                name: Some(String::from("__cut_20")),
            },
            Op::Constant {
                dtype: DType::Float32,
                shape: Vec::new(),
                value: 0.0,
            },
        ];

        let mut router_requested = original.clone();
        retain_qwen35_segment_readbacks(
            &mut router_requested,
            &router_cut_placements,
            Some(dense_roots),
            &router_program,
        );
        assert_eq!(
            router_requested,
            BTreeMap::from([(NodeId(2), NodeId(20)), (NodeId(3), NodeId(30))])
        );

        let mut gather_requested = original.clone();
        retain_qwen35_segment_readbacks(
            &mut gather_requested,
            &router_cut_placements,
            Some(dense_roots),
            &gather_program,
        );
        assert_eq!(gather_requested, BTreeMap::from([(NodeId(3), NodeId(30))]));

        let mut unplaced_requested = original.clone();
        retain_qwen35_segment_readbacks(
            &mut unplaced_requested,
            &BTreeMap::<NodeId, ()>::new(),
            None,
            &[],
        );
        assert_eq!(unplaced_requested, original);
    }

    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
    #[test]
    fn qwen35_dense_attention_placed_size_is_checked() {
        assert_eq!(
            qwen35_dense_attention_placed_byte_length(32, 128, 3, "value")
                .expect("a normal cache extent fits"),
            16_384
        );
        assert!(qwen35_dense_attention_placed_byte_length(usize::MAX, 2, 3, "value").is_err());
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn qwen35moe_expert_source_mapping_rejects_a_missing_segment_node() {
        let source_program = [Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(1)],
            name: Some(String::from("blk.0.ffn_gate_exps.weight")),
        }];
        let segment_program = [Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(1)],
            name: Some(String::from("activation")),
        }];
        let payload = [1.0_f32];
        let entries = [ExpertEntry {
            block: QuantizedBlock::Float32(&payload),
            out_dim: 1,
            in_dim: 1,
            epoch: 7,
        }];
        let sources = BTreeMap::from([(NodeId(0), ExpertSource::new(&entries))]);

        let error = map_expert_sources_to_segment(0, &source_program, &segment_program, &sources)
            .expect_err("a source absent from the gather program must fail before Metal staging");
        assert!(matches!(
            error,
            super::InteropError::PreGatherExecutionUnsupported { architecture, reason }
                if architecture == "qwen35moe"
                    && reason.contains("expert source node 0")
                    && reason.contains("absent from the gather segment")
        ));
    }

    #[cfg(feature = "metal")]
    #[test]
    fn full_plan_cache_distinguishes_output_roots_at_the_same_shape() {
        let mut cache = BTreeMap::new();
        let mut hits = 0;
        let mut misses = 0;
        let cache_roots = vec![NodeId(41)];
        let logits_and_cache_roots = vec![NodeId(15061), NodeId(41)];

        let cache_only = super::BackendRuntime::resolve_cached_plan(
            &mut cache,
            &mut hits,
            &mut misses,
            (2, 16, cache_roots),
            || Ok::<_, super::InteropError>(11_u32),
        )
        .expect("the cache-only prefill plan resolves");
        assert_eq!(*cache_only, 11);

        let with_logits = super::BackendRuntime::resolve_cached_plan(
            &mut cache,
            &mut hits,
            &mut misses,
            (2, 16, logits_and_cache_roots.clone()),
            || Ok::<_, super::InteropError>(22_u32),
        )
        .expect("the final prefill plan with logits resolves independently");
        assert_eq!(*with_logits, 22);

        let repeated_with_logits = super::BackendRuntime::resolve_cached_plan(
            &mut cache,
            &mut hits,
            &mut misses,
            (2, 16, logits_and_cache_roots),
            || Ok::<_, super::InteropError>(33_u32),
        )
        .expect("the identical final prefill plan is reused");
        assert_eq!(*repeated_with_logits, 22);
        assert_eq!(cache.len(), 1);
        assert_eq!(hits, 1);
        assert_eq!(misses, 2);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn segment_plan_cache_distinguishes_kv_bucket_extent() {
        let mut cache = BTreeMap::new();
        let mut hits = 0;
        let mut misses = 0;
        let first = super::BackendRuntime::resolve_segment_plan(
            &mut cache,
            &mut hits,
            &mut misses,
            (17, 1, 64, vec![NodeId(3)]),
            || Ok::<_, super::InteropError>(11_u32),
        )
        .expect("first segment shape resolves");
        assert_eq!(*first, 11);
        let second = super::BackendRuntime::resolve_segment_plan(
            &mut cache,
            &mut hits,
            &mut misses,
            (17, 1, 128, vec![NodeId(3)]),
            || Ok::<_, super::InteropError>(22_u32),
        )
        .expect("new KV bucket resolves independently");
        assert_eq!(*second, 22);
        assert_eq!(hits, 0);
        assert_eq!(misses, 2);
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    use super::{BackendRuntime, LoadedModel};
    use super::{
        ControlFlow, DecodeMetrics, Phase, TokenEvent, build_position_inputs,
        decode_until_stop_or_budget,
    };
    use crate::bind::architecture_from_metadata;

    #[test]
    fn residency_mutation_closes_only_for_the_expert_gather_phase() {
        let checkpoint_expert = [0_u8; 144];
        let routed_expert = [7_u8; 144];
        let slab = std::sync::Mutex::new(crate::expert_slab::ExpertSlab::new());
        lock_expert_slab(&slab)
            .bind_layer_stack(
                0,
                proxima_tensor::op::NodeId(1),
                crate::bind::Codec::Q4K,
                &checkpoint_expert,
                1,
                32,
                32,
            )
            .expect("the routed layer binds before evaluation");

        lock_expert_slab(&slab)
            .page_expert(0, 0, crate::bind::Codec::Q4K, &routed_expert, 32, 32)
            .expect("the current route may change residency before gather");

        let mut locked = lock_expert_slab(&slab);
        let mut gather_phase = locked.begin_step();
        // `StepGuard` forwards no paging method itself -- `as_slab_mut` is
        // the one crate-private escape this test uses to prove the
        // `step_in_progress` rejection still fires when something inside
        // the crate reaches past the guard, exactly as
        // `visit_qwen35moe_router_boundary`'s own residency callback would.
        let during_gather = gather_phase.as_slab_mut().page_expert(
            0,
            0,
            crate::bind::Codec::Q4K,
            &checkpoint_expert,
            32,
            32,
        );
        assert!(matches!(
            during_gather,
            Err(crate::InteropError::ExpertSwapDuringStep {
                layer: 0,
                expert: 0
            })
        ));

        drop(gather_phase);
        drop(locked);
        lock_expert_slab(&slab)
            .page_expert(0, 0, crate::bind::Codec::Q4K, &checkpoint_expert, 32, 32)
            .expect("paging succeeds again once the gather phase's StepGuard drops");
    }

    #[test]
    fn router_segment_visits_current_top_k_before_the_gather_boundary() {
        let logits = [0.5_f32, 7.0, 7.0, -1.0, 9.0, 1.0, 2.0, 8.0];
        let mut route_scratch = Vec::with_capacity(2);
        let mut visited = Vec::new();

        visit_qwen35moe_router_selections(
            3,
            41,
            RouterLogits {
                values: &logits,
                shape: &[2, 4],
            },
            RouterExpertCounts {
                expert_count: 4,
                expert_used_count: 2,
            },
            &mut route_scratch,
            &mut |layer, position, routes| {
                visited.push((
                    layer,
                    position,
                    routes.iter().map(|route| route.expert).collect::<Vec<_>>(),
                ));
                Ok(())
            },
        )
        .expect("the two real router rows expose their top-2 routes");

        assert_eq!(
            visited,
            [(3, 41, vec![1, 2]), (3, 42, vec![0, 3])],
            "the callback receives lower-index tie breaking and every position before its gather"
        );
        assert_eq!(
            route_scratch.capacity(),
            2,
            "the fixed top-k scratch does not grow while visiting router rows"
        );
    }

    #[test]
    fn router_boundary_pages_before_refreshing_the_gather_source() {
        let checkpoint_expert = [0_u8; 144];
        let routed_expert = [7_u8; 144];
        let weight_node = proxima_tensor::op::NodeId(1);
        let mut slab = crate::expert_slab::ExpertSlab::new();
        slab.bind_layer_stack(
            0,
            weight_node,
            crate::bind::Codec::Q4K,
            &checkpoint_expert,
            1,
            32,
            32,
        )
        .expect("the routed layer binds before evaluation");
        slab.open_step();
        let mut route_scratch = Vec::with_capacity(1);

        visit_qwen35moe_router_boundary(
            0,
            9,
            RouterLogits {
                values: &[3.0],
                shape: &[1, 1],
            },
            RouterExpertCounts {
                expert_count: 1,
                expert_used_count: 1,
            },
            &mut route_scratch,
            &mut slab,
            &mut |layer, position, routes, slab| {
                assert_eq!((layer, position), (0, 9));
                assert_eq!(routes[0].expert, 0);
                slab.page_expert(
                    layer,
                    routes[0].expert,
                    crate::bind::Codec::Q4K,
                    &routed_expert,
                    32,
                    32,
                )?;
                Ok(())
            },
        )
        .expect("the residency callback runs while the gather boundary is open");

        let mutation_after_boundary =
            slab.page_expert(0, 0, crate::bind::Codec::Q4K, &checkpoint_expert, 32, 32);
        assert!(matches!(
            mutation_after_boundary,
            Err(crate::InteropError::ExpertSwapDuringStep {
                layer: 0,
                expert: 0
            })
        ));

        let mut entries = Vec::new();
        let sources = slab
            .sources_for_step(&mut entries)
            .expect("the post-boundary gather source is complete");
        let source = sources
            .get(&weight_node)
            .expect("the routed layer has a refreshed source");
        let entry = source
            .entries()
            .first()
            .expect("the selected expert remains at its stable index");
        assert_eq!(entry.epoch, 1, "the gather sees the boundary page");
        assert!(matches!(
            entry.block,
            proxima_tensor::cpu::QuantizedBlock::Q4K(bytes) if bytes == routed_expert
        ));
    }

    #[test]
    fn kv_bucket_extent_includes_the_new_position() {
        let cached_len = 0usize;
        let new_count = 1usize;
        assert_eq!(
            kv_extent(cached_len + new_count, usize::MAX, 32),
            32,
            "the first position must allocate a non-empty KV bucket"
        );
        assert_eq!(
            kv_extent(cached_len, usize::MAX, 32),
            0,
            "the pre-step cache length is not a valid attention extent"
        );
        assert_eq!(
            kv_extent(15, usize::MAX, 32),
            32,
            "persistent KV capacity must not clip a bucket below its compiled extent"
        );
        assert_eq!(
            kv_extent(39, usize::MAX, 32),
            64,
            "a later reachable position must reserve the next full bucket"
        );
    }

    #[test]
    fn intermediate_split_prefill_batch_does_not_request_logits() {
        assert!(step_batch_needs_logits(false, false));
        assert!(step_batch_needs_logits(false, true));
        assert!(!step_batch_needs_logits(true, false));
        assert!(step_batch_needs_logits(true, true));
    }

    #[test]
    fn router_selection_callback_runs_before_gather() {
        let mut scratch = Vec::new();
        let mut observed = Vec::new();
        super::visit_qwen35moe_router_selections(
            2,
            11,
            super::RouterLogits {
                values: &[0.1, 0.9, 0.2, 0.8],
                shape: &[1, 4],
            },
            super::RouterExpertCounts {
                expert_count: 4,
                expert_used_count: 2,
            },
            &mut scratch,
            &mut |layer, position, routes| {
                observed.push((layer, position, routes[0].expert, routes[1].expert));
                Ok(())
            },
        )
        .expect("router callback receives one deterministic top-k row");
        assert_eq!(observed, vec![(2, 11, 1, 3)]);
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn metal_router_segment_readback_reaches_the_residency_boundary() {
        let config = super::supported_serving_config(GPU_LAYERS_ALL, omega::MathMode::default());
        let mut runtime = BackendRuntime::new(&config);
        let router = NodeId(0);
        let program = [Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(1), Extent::Static(4)],
            name: Some(String::from("router_logits")),
        }];
        let logits = [0.1_f32, 0.9, 0.2, 0.8];
        let named = [("router_logits", QuantizedBlock::Float32(&logits))];
        let evaluated = runtime
            .evaluate_segment(&program, &[1, 1], &named, &[router], &BTreeSet::new(), None)
            .expect("metal returns the requested router tensor to the host boundary");
        let (readback, shape) = evaluated
            .get(router)
            .expect("the requested router tensor is present after metal execution");
        let mut scratch = Vec::new();
        let mut observed = Vec::new();

        visit_qwen35moe_router_selections(
            2,
            11,
            RouterLogits {
                values: readback,
                shape,
            },
            RouterExpertCounts {
                expert_count: 4,
                expert_used_count: 2,
            },
            &mut scratch,
            &mut |layer, position, routes| {
                observed.push((layer, position, routes[0].expert, routes[1].expert));
                Ok(())
            },
        )
        .expect("the metal router readback drives the typed residency callback");

        assert_eq!(observed, vec![(2, 11, 1, 3)]);
    }

    #[test]
    fn decode_metrics_uses_cumulative_token_event_time() {
        let events = [
            TokenEvent {
                token_id: 1,
                text_piece: "",
                phase: Phase::Prefill { prompt_tokens: 7 },
                step: 0,
                elapsed_ms: 400,
            },
            TokenEvent {
                token_id: 1,
                text_piece: "a",
                phase: Phase::Token,
                step: 0,
                elapsed_ms: 400,
            },
            TokenEvent {
                token_id: 2,
                text_piece: "b",
                phase: Phase::Token,
                step: 1,
                elapsed_ms: 900,
            },
        ];
        let metrics = DecodeMetrics::from_events(&events, 3_500, 42.5, 1, 0.125);

        assert_eq!(metrics.prompt_tokens, 7);
        assert_eq!(metrics.generated_tokens, 2);
        assert_eq!(metrics.elapsed_ms, 900);
        assert!((metrics.tokens_per_second - (2.0 / 0.9)).abs() < f64::EPSILON);
        assert!((metrics.per_token_latency_ms - 450.0).abs() < f64::EPSILON);
        assert_eq!(metrics.peak_rss_bytes, 3_500);
        assert_eq!(metrics.cpu_percent, 42.5);
        assert_eq!(metrics.error_count, 1);
        assert_eq!(metrics.covariance, 0.125);
    }

    #[test]
    fn decode_metrics_keeps_provenance_labels_explicit() {
        let metrics = DecodeMetrics::from_events(&[], 0, 0.0, 0, 0.0);

        assert_eq!(metrics.timing_source, "TokenEvent::elapsed_ms");
        assert_eq!(metrics.memory_source, "caller_peak_rss_bytes");
        assert_eq!(
            metrics.resource_source,
            "caller_cpu_percent_and_error_count"
        );
    }

    #[test]
    fn ssm_state_replacement_preserves_fixed_capacity() {
        let mut cache = SsmLayerCache::new(4, 3);
        let state_capacity = cache.state.capacity();
        cache.advance(&[1.0, 2.0], &[3.0, 4.0, 5.0], 4);
        assert_eq!(cache.state, vec![3.0, 4.0, 5.0]);
        assert_eq!(cache.state.capacity(), state_capacity);
        cache.advance(&[6.0, 7.0], &[8.0, 9.0, 10.0], 4);
        assert_eq!(cache.state, vec![8.0, 9.0, 10.0]);
        assert_eq!(cache.state.capacity(), state_capacity);
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    use crate::serving::{GPU_LAYERS_ALL, ServingConfig};

    fn dims(values: &[u64]) -> arrayvec::ArrayVec<u64, { proxima_gguf::tensor::MAX_DIMS }> {
        values.iter().copied().collect()
    }

    /// A checkpoint that declares `llama.attention.layer_norm_rms_epsilon`
    /// (Qwen3's own value, `1e-6`, chosen because it differs from
    /// `crate::bind`'s own `RMS_EPSILON_DEFAULT` (`1e-5`) -- a test using
    /// the default would pass even if the metadata read were wired to
    /// nothing) must have that value flow all the way from
    /// [`architecture_from_metadata`] through [`build_position_inputs`]'s
    /// `epsilon` output, the exact vector `run_decode_loop` feeds every
    /// layer norm on the Metal/CPU decode path.
    #[test]
    fn checkpoint_declared_rms_epsilon_flows_into_position_inputs() {
        let embed_bytes = vec![0u8; 4 * 3 * 4]; // [embedding=4, vocab=3] f32
        let model = GgufModel {
            version: 3,
            metadata: vec![
                (
                    "general.architecture".to_string(),
                    Value::String("llama".to_string()),
                ),
                ("llama.embedding_length".to_string(), Value::U32(4)),
                ("llama.feed_forward_length".to_string(), Value::U32(8)),
                ("llama.attention.head_count".to_string(), Value::U32(2)),
                ("llama.attention.head_count_kv".to_string(), Value::U32(1)),
                ("llama.block_count".to_string(), Value::U32(1)),
                ("llama.rope.dimension_count".to_string(), Value::U32(2)),
                (
                    "llama.attention.layer_norm_rms_epsilon".to_string(),
                    Value::F32(1e-6),
                ),
            ],
            tensors: vec![TensorPayload {
                name: "token_embd.weight".to_string(),
                dims: dims(&[4, 3]),
                ggml_type: WireType::F32,
                data: &embed_bytes,
            }],
        };
        let file_bytes = write_complete(&model).expect("writes gguf with rms_epsilon metadata");
        let parsed = proxima_gguf::parse_complete(&file_bytes)
            .expect("parses gguf with rms_epsilon metadata");
        let architecture = architecture_from_metadata(&parsed)
            .expect("derive architecture from real metadata keys");

        assert_eq!(
            architecture.rms_epsilon, 1e-6,
            "architecture_from_metadata must read the checkpoint's own \
             layer_norm_rms_epsilon key, not a hard-coded default"
        );

        let inputs = build_position_inputs(
            &[7, 9],
            0,
            architecture.head_dim,
            architecture.rope_freq_base,
            architecture.rms_epsilon,
            false,
            None,
        );

        assert_eq!(
            inputs.epsilon,
            alloc::vec![1e-6, 1e-6],
            "build_position_inputs must feed the checkpoint's own epsilon into every \
             position, not RMS_EPSILON_DEFAULT"
        );
    }

    /// `int8-logs`, main `9a8b623c`: [`LoadedModel::forward_node_values_on_backend`]
    /// reported `NotLowerable { reason: "operand buffer missing at
    /// evaluation time" }` for a directly-requested `NodeId` once the
    /// requested `node_ids` window widened past roughly one layer's own
    /// node count, bisected in production between 35 and 40 requested
    /// nodes. Reproduces against the real host-local checkpoint (the exact
    /// quantized-weight, `cohort-staged-graph` CPU path production uses --
    /// a synthetic float32-only program does not engage the same matmul
    /// batching this needs): every `NodeId` in a wide window must come back
    /// with the identical value it has when requested alone.
    mod real_openchat_file {
        use core::ffi::c_void;
        use std::os::fd::AsFd;

        use proxima_tensor::op::NodeId;

        use super::super::LoadedModel;

        struct MappedGguf {
            base: *mut u8,
            len: usize,
            _file: std::fs::File,
        }

        impl MappedGguf {
            fn open(path: &std::path::Path) -> std::io::Result<Self> {
                let file = std::fs::File::open(path)?;
                let len = usize::try_from(file.metadata()?.len())
                    .expect("fixture file length fits in usize");
                // SAFETY: `len` matches the just-opened file's own length;
                // `file` is kept alive in `_file` for as long as `base` is
                // used, and the mapping is read-only/private so no writer
                // can observe or race it.
                let base = unsafe {
                    rustix::mm::mmap(
                        core::ptr::null_mut(),
                        len,
                        rustix::mm::ProtFlags::READ,
                        rustix::mm::MapFlags::PRIVATE,
                        file.as_fd(),
                        0,
                    )
                }
                .expect("mmap host-local openchat gguf fixture")
                .cast::<u8>();
                Ok(Self {
                    base,
                    len,
                    _file: file,
                })
            }

            fn as_slice(&self) -> &[u8] {
                // SAFETY: `base` points at `len` bytes mapped for `self`'s
                // whole lifetime; this borrows `self` immutably, so nothing
                // can unmap the region while the returned slice is alive.
                unsafe { core::slice::from_raw_parts(self.base, self.len) }
            }
        }

        impl Drop for MappedGguf {
            fn drop(&mut self) {
                // SAFETY: `base`/`len` are exactly what `open`'s `mmap`
                // call returned; nothing else unmaps this region.
                let _ = unsafe { rustix::mm::munmap(self.base.cast::<c_void>(), self.len) };
            }
        }

        #[test]
        #[ignore = "depends on a host-local openchat gguf checkout outside this repo"]
        fn forward_node_values_keeps_every_requested_node_live_across_a_layer_boundary() {
            let model_path = crate::test_support::openchat_gguf_path();
            crate::test_support::require_fixture(&model_path, Some("PROXIMA_OPENCHAT_GGUF"));
            let path = std::path::Path::new(&model_path);

            let mapped = MappedGguf::open(path).expect("mmap host-local openchat gguf fixture");
            let file_bytes = mapped.as_slice();
            let parsed = proxima_gguf::pipe::parse_complete(file_bytes)
                .expect("parse host-local openchat gguf fixture");
            let model = LoadedModel::load(&parsed, file_bytes)
                .expect("load real openchat checkpoint through the public path");

            let window: Vec<NodeId> = (0u32..40).map(NodeId).collect();
            let prompt = "The quick brown fox";

            let batch = model
                .forward_node_values(prompt, &window)
                .expect("every requested node across the layer boundary must evaluate");

            assert_eq!(
                batch.len(),
                window.len(),
                "one value must come back per requested node"
            );
            // A KV-cache `Op::Input` (`kv_cache.0.k_even` etc.) is legitimately
            // zero-length here -- this is a one-shot, fresh-KV-state forward
            // pass (`forward_node_values`'s own doc), so "empty" is a correct
            // answer for that node class, not evidence of eviction. NodeId(34)
            // is the exact node this defect's own bisection named: the
            // `activation * quantized-weight` multiply `is_quantized_matmul_operand`
            // ordinarily fuses into its reduce, forced standalone here only
            // because this window's own liveness protection keeps it alive --
            // its value coming back non-empty is the falsifiable claim this
            // test exists to prove.
            let quantized_matmul_multiply = NodeId(34);
            let position = window
                .iter()
                .position(|node| *node == quantized_matmul_multiply)
                .expect("the chosen window must include the node this defect was bisected to");
            assert!(
                !batch[position].is_empty(),
                "{quantized_matmul_multiply:?} came back with zero values -- evicted or never \
                 materialized despite being directly requested"
            );
        }
    }

    /// A minimal valid [`Vocab`] (every byte-level BPE vocab needs all 256
    /// base-byte tokens present or [`Vocab::new`] rejects it) plus one
    /// extra token at id `256` marked as this vocab's end-of-sequence id --
    /// enough to exercise [`decode_until_stop_or_budget`]'s stopping policy
    /// without a real checkpoint. Spells the base-byte alphabet as the
    /// SentencePiece `"<0xXX>"` fallback form directly (not through
    /// `proxima_tokenizer`'s private `byte_to_char`) since that spelling is
    /// public knowledge, not an internal detail this test needs to reach
    /// into the crate for.
    fn vocab_with_eos(eos_id: u32) -> Vocab {
        let mut tokens: Vec<String> = (0..=255u8)
            .map(|byte| alloc::format!("<0x{byte:02X}>"))
            .collect();
        tokens.push(String::from("<eos-marker>"));
        Vocab::new(tokens, &[], Some(0), Some(eos_id), None).expect("minimal vocab builds")
    }

    /// The defect this module exists to fix, proved directly: a scripted
    /// token source that would emit `999` on a 4th call never gets asked
    /// for it, because the 3rd call's token (`32000`, this vocab's eos id)
    /// stops the loop first. Also proves the eos id itself never lands in
    /// `generated_ids`.
    #[test]
    fn stops_early_when_eos_is_produced_and_excludes_it_from_ids() {
        let vocab = vocab_with_eos(32_000);
        let scripted_tokens = [10u32, 20, 32_000, 999];
        let mut calls = 0usize;

        let (generated_ids, stopped_by_eos) = decode_until_stop_or_budget(
            &vocab,
            4,
            0,
            |step| {
                calls += 1;
                Ok(scripted_tokens[step])
            },
            &mut |_event| ControlFlow::Continue(()),
        )
        .expect("scripted token source never errors");

        assert_eq!(
            generated_ids,
            alloc::vec![10, 20],
            "eos id must not be appended to the generated ids"
        );
        assert!(
            stopped_by_eos,
            "must report that the stop was the model's own eos signal"
        );
        assert_eq!(
            calls, 3,
            "must not pull a 4th token once eos is seen on the 3rd"
        );
    }

    /// The other half of the invariant: when the model never emits eos,
    /// decoding runs the full budget and reports that distinctly from an
    /// eos stop -- `stopped_by_eos == false` is the caller's only way to
    /// tell "ran out of budget" apart from "the model finished".
    #[test]
    fn exhausts_the_budget_and_reports_it_distinctly_from_an_eos_stop() {
        let vocab = vocab_with_eos(32_000);
        let scripted_tokens = [10u32, 20, 30, 40];

        let (generated_ids, stopped_by_eos) = decode_until_stop_or_budget(
            &vocab,
            scripted_tokens.len(),
            0,
            |step| Ok(scripted_tokens[step]),
            &mut |_event| ControlFlow::Continue(()),
        )
        .expect("scripted token source never errors");

        assert_eq!(
            generated_ids,
            alloc::vec![10, 20, 30, 40],
            "every scripted token is a real id, none is eos"
        );
        assert!(
            !stopped_by_eos,
            "budget exhaustion must not be reported as an eos stop"
        );
        assert_eq!(
            generated_ids.len(),
            scripted_tokens.len(),
            "budget exhaustion still runs every requested step"
        );
    }

    /// ROW 392's own class fix, proved directly against the real two-range
    /// path (not a hand-rolled stand-in): a synthetic one-layer
    /// mixture-of-experts checkpoint (`architecture.expert_count > 0` forces
    /// [`LoadedModel::single_range`] to `None`, `build_single_range_program`'s
    /// own doc, so `gpu_layers: GPU_LAYERS_ALL` here reaches
    /// [`BackendRuntime::evaluate`] through [`LoadedModel::run_decode_loop`],
    /// never `run_decode_loop_placed_kv`) drives the SAME 8-step greedy
    /// decode twice, once with `kv_bucket_tokens: 1` (today's pre-fix
    /// behavior: `kv_extent`'s own doc, `div_ceil(1)` is the identity) and
    /// once with `kv_bucket_tokens: 32` (`ServingConfig::default`'s own
    /// value). Two claims, both comparative rather than a single hard-coded
    /// constant, so neither depends on `omega`'s own internal per-node
    /// allocation count: bucketing must produce STRICTLY fewer plan misses
    /// and STRICTLY fewer [`omega::metal::OUTPUT_BUFFER_ALLOCATIONS`] than
    /// the unbucketed run (ROW 392's own finding: one miss, and one fresh
    /// `Plan` with its own device output buffers, per token before this
    /// fix), and the two runs must land on the IDENTICAL generated token
    /// ids -- `proxima_tensor::bind::cached_attention_candidates`'s own doc
    /// on the fused op's runtime bound is the numerics claim this equality
    /// is standing in for: a bucket's padding is invisible to softmax, so
    /// rounding `cached_len` up must never change what the model emits.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn two_range_plan_cache_buckets_cached_len_without_changing_generated_tokens() {
        fn f32_bytes(values: &[f32]) -> Vec<u8> {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        }

        let vocab_size = 257u64;
        let embedding = 2u64;
        let feed_forward = 2u64;
        let expert_count = 2u64;

        let mut tokens: Vec<String> = (0..=255u8)
            .map(|byte| alloc::format!("<0x{byte:02X}>"))
            .collect();
        tokens.push(String::from("<eos-marker>"));

        let token_embd = f32_bytes(&vec![0.05f32; (vocab_size * embedding) as usize]);
        let norm_weight = f32_bytes(&vec![1.0f32; embedding as usize]);
        let square = f32_bytes(&vec![0.05f32; (embedding * embedding) as usize]);
        let gate_inp = f32_bytes(&vec![0.05f32; (embedding * expert_count) as usize]);
        let expert_stack = f32_bytes(&vec![
            0.05f32;
            (expert_count * feed_forward * embedding) as usize
        ]);
        let output_weight = f32_bytes(&vec![0.05f32; (vocab_size * embedding) as usize]);

        let model = GgufModel {
            version: 3,
            metadata: vec![
                (
                    "general.architecture".to_string(),
                    Value::String("llama".to_string()),
                ),
                (
                    "llama.embedding_length".to_string(),
                    Value::U32(embedding as u32),
                ),
                (
                    "llama.feed_forward_length".to_string(),
                    Value::U32(feed_forward as u32),
                ),
                ("llama.attention.head_count".to_string(), Value::U32(1)),
                ("llama.attention.head_count_kv".to_string(), Value::U32(1)),
                ("llama.block_count".to_string(), Value::U32(1)),
                (
                    "llama.expert_count".to_string(),
                    Value::U32(expert_count as u32),
                ),
                ("llama.expert_used_count".to_string(), Value::U32(1)),
                (
                    "tokenizer.ggml.model".to_string(),
                    Value::String("gpt2".to_string()),
                ),
                (
                    "tokenizer.ggml.tokens".to_string(),
                    Value::Array(MetadataArray::String(tokens)),
                ),
                (
                    "tokenizer.ggml.merges".to_string(),
                    Value::Array(MetadataArray::String(Vec::new())),
                ),
            ],
            tensors: vec![
                TensorPayload {
                    name: "token_embd.weight".to_string(),
                    dims: dims(&[embedding, vocab_size]),
                    ggml_type: WireType::F32,
                    data: &token_embd,
                },
                TensorPayload {
                    name: "blk.0.attn_norm.weight".to_string(),
                    dims: dims(&[embedding]),
                    ggml_type: WireType::F32,
                    data: &norm_weight,
                },
                TensorPayload {
                    name: "blk.0.ffn_norm.weight".to_string(),
                    dims: dims(&[embedding]),
                    ggml_type: WireType::F32,
                    data: &norm_weight,
                },
                TensorPayload {
                    name: "blk.0.attn_q.weight".to_string(),
                    dims: dims(&[embedding, embedding]),
                    ggml_type: WireType::F32,
                    data: &square,
                },
                TensorPayload {
                    name: "blk.0.attn_k.weight".to_string(),
                    dims: dims(&[embedding, embedding]),
                    ggml_type: WireType::F32,
                    data: &square,
                },
                TensorPayload {
                    name: "blk.0.attn_v.weight".to_string(),
                    dims: dims(&[embedding, embedding]),
                    ggml_type: WireType::F32,
                    data: &square,
                },
                TensorPayload {
                    name: "blk.0.attn_output.weight".to_string(),
                    dims: dims(&[embedding, embedding]),
                    ggml_type: WireType::F32,
                    data: &square,
                },
                TensorPayload {
                    name: "blk.0.ffn_gate_inp.weight".to_string(),
                    dims: dims(&[embedding, expert_count]),
                    ggml_type: WireType::F32,
                    data: &gate_inp,
                },
                TensorPayload {
                    name: "blk.0.ffn_gate_exps.weight".to_string(),
                    dims: dims(&[embedding, feed_forward, expert_count]),
                    ggml_type: WireType::F32,
                    data: &expert_stack,
                },
                TensorPayload {
                    name: "blk.0.ffn_up_exps.weight".to_string(),
                    dims: dims(&[embedding, feed_forward, expert_count]),
                    ggml_type: WireType::F32,
                    data: &expert_stack,
                },
                TensorPayload {
                    name: "blk.0.ffn_down_exps.weight".to_string(),
                    dims: dims(&[feed_forward, embedding, expert_count]),
                    ggml_type: WireType::F32,
                    data: &expert_stack,
                },
                TensorPayload {
                    name: "output_norm.weight".to_string(),
                    dims: dims(&[embedding]),
                    ggml_type: WireType::F32,
                    data: &norm_weight,
                },
                TensorPayload {
                    name: "output.weight".to_string(),
                    dims: dims(&[embedding, vocab_size]),
                    ggml_type: WireType::F32,
                    data: &output_weight,
                },
            ],
        };

        let file_bytes =
            write_complete(&model).expect("writes a minimal one-layer MoE gguf fixture");
        let parsed = proxima_gguf::pipe::parse_complete(&file_bytes)
            .expect("parses the minimal one-layer MoE gguf fixture");
        let loaded = LoadedModel::load(&parsed, &file_bytes)
            .expect("loads the minimal one-layer MoE checkpoint through the public path");

        let base_config = ServingConfig {
            kv_cache_key_quant: WireType::F32,
            kv_cache_value_quant: WireType::F32,
            flash_attention: false,
            batch_size: 0,
            ubatch_size: 0,
            gpu_layers: GPU_LAYERS_ALL,
            reasoning_budget: 0,
            ..ServingConfig::default()
        };
        let max_tokens = 8usize;

        let unbucketed_config = ServingConfig {
            kv_bucket_tokens: 1,
            ..base_config
        };
        let mut unbucketed_runtime = BackendRuntime::new(&unbucketed_config);
        let _ = omega::metal::OUTPUT_BUFFER_ALLOCATIONS.snapshot_and_reset();
        let unbucketed = loaded
            .run_decode_loop("A", max_tokens, &unbucketed_config, &mut unbucketed_runtime)
            .expect("runs the unbucketed two-range MoE decode loop on the metal backend");
        let unbucketed_allocations = omega::metal::OUTPUT_BUFFER_ALLOCATIONS.snapshot_and_reset();

        let bucketed_config = ServingConfig {
            kv_bucket_tokens: 32,
            ..base_config
        };
        let mut bucketed_runtime = BackendRuntime::new(&bucketed_config);
        let _ = omega::metal::OUTPUT_BUFFER_ALLOCATIONS.snapshot_and_reset();
        let bucketed = loaded
            .run_decode_loop("A", max_tokens, &bucketed_config, &mut bucketed_runtime)
            .expect("runs the bucketed two-range MoE decode loop on the metal backend");
        let bucketed_allocations = omega::metal::OUTPUT_BUFFER_ALLOCATIONS.snapshot_and_reset();

        assert_eq!(
            unbucketed.0.len(),
            max_tokens,
            "no eos id was declared, so both runs must exhaust the full token budget"
        );
        assert_eq!(
            unbucketed_runtime.plan_misses, max_tokens,
            "kv_bucket_tokens=1 reproduces the pre-fix shape: cached_len is strictly \
             increasing, so every step is a fresh miss"
        );
        assert_eq!(
            unbucketed_runtime.plan_hits, 0,
            "an unbucketed extent never repeats within one decode call"
        );

        assert!(
            bucketed_runtime.plan_hits > 0,
            "a bucketed extent that never hits is the null result, not a pass"
        );
        assert!(
            bucketed_runtime.plan_misses < unbucketed_runtime.plan_misses,
            "kv_bucket_tokens=32 must reduce plan_misses below the unbucketed baseline, \
             or bucketing bought nothing on this fixture"
        );
        // Non-strict: measured `0` on both arms in this sandbox (no real
        // Metal device attached, `omega`'s own Gpu-arm buffer allocation
        // never fires at all rather than firing once per miss) -- the
        // plan_hits/plan_misses assertions above are this test's load-
        // bearing, environment-independent proof of ROW 392's fix; this one
        // only guards against a REGRESSION (bucketing must never allocate
        // MORE than the unbucketed baseline) on whatever device runs it.
        assert!(
            bucketed_allocations <= unbucketed_allocations,
            "bucketed_allocations={bucketed_allocations} unbucketed_allocations={unbucketed_allocations}: \
             bucketing must never allocate MORE device output buffers than the unbucketed baseline"
        );
        assert_eq!(
            bucketed.0, unbucketed.0,
            "the fused CachedAttention op's runtime bound must make a bucket's own \
             zero-padding invisible to softmax -- rounding cached_len up must never \
             change which token is emitted"
        );
    }

    /// [`two_range_plan_cache_buckets_cached_len_without_changing_generated_tokens`]'s
    /// counterpart for a qwen35 [`Qwen35LayerRoots::DenseAttention`] layer --
    /// this crate's fake-fixture fallback for that same claim, not the full
    /// 4-layer hybrid checkpoint `feat/synthetic-qwen38-fixture`'s own
    /// `examples/synth_qwen35_gguf.rs` builds (~918 MiB, out of this slice's
    /// time budget): one synthetic, `full_attention_interval: 1` layer (so
    /// every layer is [`crate::qwen35::Qwen35LayerKind::Attention`], no
    /// state-space mixer to also fixture), just wide enough
    /// (`query_heads = kv_heads = 1`, `attention.key_length = 4`,
    /// `rope.dimension_count = 2`, so `pass_dim = 2` is exercised alongside
    /// the rotated halves) to drive `qwen35_forward_program`'s
    /// `DenseAttention` cache path through
    /// [`Qwen35DenseAttentionPadScratch`] the same way the MoE test above
    /// drives `mistral_cached_forward_program_with_experts`'s `Attention`
    /// path through [`KvPadScratch`]. Asserts `plan_hits`/`plan_misses`
    /// only, per this card's own fake-fixture allowance -- no
    /// `OUTPUT_BUFFER_ALLOCATIONS`/CPU-vs-Metal comparison, since those need
    /// the real hybrid checkpoint's own numerics to be meaningful.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn qwen35_dense_attention_two_range_plan_cache_buckets_cached_len() {
        fn f32_bytes(values: &[f32]) -> Vec<u8> {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        }

        let vocab_size = 257u64;
        let embedding = 4u64;
        let feed_forward = 4u64;
        let query_heads = 1u64;
        let kv_heads = 1u64;
        let rotary_dim = 2u64;
        let attn_head_dim = 4u64;

        let mut tokens: Vec<String> = (0..=255u8)
            .map(|byte| alloc::format!("<0x{byte:02X}>"))
            .collect();
        tokens.push(String::from("<eos-marker>"));

        let token_embd = f32_bytes(&vec![0.05f32; (embedding * vocab_size) as usize]);
        let norm_weight = f32_bytes(&vec![1.0f32; embedding as usize]);
        let head_norm_weight = f32_bytes(&vec![1.0f32; attn_head_dim as usize]);
        let q_weight = f32_bytes(&vec![
            0.05f32;
            (embedding * query_heads * attn_head_dim * 2) as usize
        ]);
        let kv_weight = f32_bytes(&vec![
            0.05f32;
            (embedding * kv_heads * attn_head_dim) as usize
        ]);
        let output_weight = f32_bytes(&vec![
            0.05f32;
            (query_heads * attn_head_dim * embedding) as usize
        ]);
        let ffn_gate_up = f32_bytes(&vec![0.05f32; (embedding * feed_forward) as usize]);
        let ffn_down = f32_bytes(&vec![0.05f32; (feed_forward * embedding) as usize]);
        let output_table = f32_bytes(&vec![0.05f32; (embedding * vocab_size) as usize]);

        let model = GgufModel {
            version: 3,
            metadata: vec![
                (
                    "general.architecture".to_string(),
                    Value::String("qwen35".to_string()),
                ),
                (
                    "qwen35.embedding_length".to_string(),
                    Value::U32(embedding as u32),
                ),
                (
                    "qwen35.feed_forward_length".to_string(),
                    Value::U32(feed_forward as u32),
                ),
                (
                    "qwen35.attention.head_count".to_string(),
                    Value::U32(query_heads as u32),
                ),
                (
                    "qwen35.attention.head_count_kv".to_string(),
                    Value::U32(kv_heads as u32),
                ),
                ("qwen35.block_count".to_string(), Value::U32(1)),
                (
                    "qwen35.rope.dimension_count".to_string(),
                    Value::U32(rotary_dim as u32),
                ),
                (
                    "qwen35.attention.key_length".to_string(),
                    Value::U32(attn_head_dim as u32),
                ),
                ("qwen35.full_attention_interval".to_string(), Value::U32(1)),
                ("qwen35.ssm.conv_kernel".to_string(), Value::U32(2)),
                ("qwen35.ssm.state_size".to_string(), Value::U32(1)),
                ("qwen35.ssm.group_count".to_string(), Value::U32(1)),
                ("qwen35.ssm.time_step_rank".to_string(), Value::U32(1)),
                ("qwen35.ssm.inner_size".to_string(), Value::U32(1)),
                (
                    "tokenizer.ggml.model".to_string(),
                    Value::String("gpt2".to_string()),
                ),
                (
                    "tokenizer.ggml.tokens".to_string(),
                    Value::Array(MetadataArray::String(tokens)),
                ),
                (
                    "tokenizer.ggml.merges".to_string(),
                    Value::Array(MetadataArray::String(Vec::new())),
                ),
            ],
            tensors: vec![
                TensorPayload {
                    name: "token_embd.weight".to_string(),
                    dims: dims(&[embedding, vocab_size]),
                    ggml_type: WireType::F32,
                    data: &token_embd,
                },
                TensorPayload {
                    name: "blk.0.attn_norm.weight".to_string(),
                    dims: dims(&[embedding]),
                    ggml_type: WireType::F32,
                    data: &norm_weight,
                },
                TensorPayload {
                    name: "blk.0.post_attention_norm.weight".to_string(),
                    dims: dims(&[embedding]),
                    ggml_type: WireType::F32,
                    data: &norm_weight,
                },
                TensorPayload {
                    name: "blk.0.attn_q.weight".to_string(),
                    dims: dims(&[embedding, query_heads * attn_head_dim * 2]),
                    ggml_type: WireType::F32,
                    data: &q_weight,
                },
                TensorPayload {
                    name: "blk.0.attn_k.weight".to_string(),
                    dims: dims(&[embedding, kv_heads * attn_head_dim]),
                    ggml_type: WireType::F32,
                    data: &kv_weight,
                },
                TensorPayload {
                    name: "blk.0.attn_v.weight".to_string(),
                    dims: dims(&[embedding, kv_heads * attn_head_dim]),
                    ggml_type: WireType::F32,
                    data: &kv_weight,
                },
                TensorPayload {
                    name: "blk.0.attn_output.weight".to_string(),
                    dims: dims(&[query_heads * attn_head_dim, embedding]),
                    ggml_type: WireType::F32,
                    data: &output_weight,
                },
                TensorPayload {
                    name: "blk.0.attn_q_norm.weight".to_string(),
                    dims: dims(&[attn_head_dim]),
                    ggml_type: WireType::F32,
                    data: &head_norm_weight,
                },
                TensorPayload {
                    name: "blk.0.attn_k_norm.weight".to_string(),
                    dims: dims(&[attn_head_dim]),
                    ggml_type: WireType::F32,
                    data: &head_norm_weight,
                },
                TensorPayload {
                    name: "blk.0.ffn_gate.weight".to_string(),
                    dims: dims(&[embedding, feed_forward]),
                    ggml_type: WireType::F32,
                    data: &ffn_gate_up,
                },
                TensorPayload {
                    name: "blk.0.ffn_up.weight".to_string(),
                    dims: dims(&[embedding, feed_forward]),
                    ggml_type: WireType::F32,
                    data: &ffn_gate_up,
                },
                TensorPayload {
                    name: "blk.0.ffn_down.weight".to_string(),
                    dims: dims(&[feed_forward, embedding]),
                    ggml_type: WireType::F32,
                    data: &ffn_down,
                },
                TensorPayload {
                    name: "output_norm.weight".to_string(),
                    dims: dims(&[embedding]),
                    ggml_type: WireType::F32,
                    data: &norm_weight,
                },
                TensorPayload {
                    name: "output.weight".to_string(),
                    dims: dims(&[embedding, vocab_size]),
                    ggml_type: WireType::F32,
                    data: &output_table,
                },
            ],
        };

        let file_bytes =
            write_complete(&model).expect("writes a minimal one-layer qwen35 gguf fixture");
        let parsed = proxima_gguf::pipe::parse_complete(&file_bytes)
            .expect("parses the minimal one-layer qwen35 gguf fixture");
        let loaded = LoadedModel::load(&parsed, &file_bytes)
            .expect("loads the minimal one-layer qwen35 checkpoint through the public path");

        let base_config = ServingConfig {
            kv_cache_key_quant: WireType::F32,
            kv_cache_value_quant: WireType::F32,
            flash_attention: false,
            batch_size: 0,
            ubatch_size: 0,
            gpu_layers: GPU_LAYERS_ALL,
            reasoning_budget: 0,
            ..ServingConfig::default()
        };
        let max_tokens = 8usize;

        let unbucketed_config = ServingConfig {
            kv_bucket_tokens: 1,
            ..base_config
        };
        let mut unbucketed_runtime = BackendRuntime::new(&unbucketed_config);
        let unbucketed = loaded
            .run_decode_loop("A", max_tokens, &unbucketed_config, &mut unbucketed_runtime)
            .expect("runs the unbucketed qwen35 dense-attention decode loop");

        let bucketed_config = ServingConfig {
            kv_bucket_tokens: 32,
            ..base_config
        };
        let mut bucketed_runtime = BackendRuntime::new(&bucketed_config);
        let bucketed = loaded
            .run_decode_loop("A", max_tokens, &bucketed_config, &mut bucketed_runtime)
            .expect("runs the bucketed qwen35 dense-attention decode loop");

        assert_eq!(
            unbucketed.0.len(),
            max_tokens,
            "no eos id was declared, so both runs must exhaust the full token budget"
        );
        assert_eq!(
            unbucketed_runtime.plan_misses, max_tokens,
            "kv_bucket_tokens=1 reproduces the pre-fix shape on the DenseAttention arm: \
             cached_len is strictly increasing, so every step is a fresh miss"
        );
        assert_eq!(
            unbucketed_runtime.plan_hits, 0,
            "an unbucketed extent never repeats within one decode call"
        );
        assert!(
            bucketed_runtime.plan_hits > 0,
            "a bucketed extent that never hits on the DenseAttention arm is the null \
             result, not a pass"
        );
        assert!(
            bucketed_runtime.plan_misses < unbucketed_runtime.plan_misses,
            "kv_bucket_tokens=32 must reduce plan_misses below the unbucketed baseline \
             on the DenseAttention arm, or bucketing bought nothing on this fixture"
        );
        assert_eq!(
            bucketed.0, unbucketed.0,
            "the DenseAttention arm's padded cache must make a bucket's own \
             zero-padding invisible to softmax -- rounding cached_len up must never \
             change which token is emitted"
        );
    }

    /// [`qwen35_dense_attention_two_range_plan_cache_buckets_cached_len`]'s
    /// own fixture, driven through the raw-logit payload behind that test's
    /// argmax-only token-id comparison: that comparison alone cannot rule
    /// out a real but non-argmax-flipping corruption from
    /// [`Qwen35DenseAttentionPadScratch`]'s zero-padding on this fixture's
    /// tiny, uniform (`0.05` everywhere) weights, where two distinct logit
    /// vectors can still share an argmax. Runs the SAME 8-step greedy decode
    /// twice with [`LoadedModel::run_decode_loop_observed`]'s own
    /// `LogitsSink::Collect` hook, once with `kv_bucket_tokens: 1`
    /// (`kv_extent`'s own doc: `div_ceil(1)` is the identity, so `cached_len`
    /// is never rounded up and this arm pads NOTHING) and once with
    /// `kv_bucket_tokens: 32` (`ServingConfig::default`'s own value, so the
    /// early steps round a `cached_len` as small as `1` up to `32`, the
    /// widest possible padding this fixture can exercise). Compares the
    /// FINAL step's last-row logits bit-for-bit rather than only the
    /// emitted token ids -- `proxima_tensor::bind::cached_attention_
    /// candidates`'s own doc on the fused op's runtime bound is the
    /// numerics claim this equality is standing in for on the
    /// `DenseAttention` arm, which has no such fusion and instead relies on
    /// [`append_qwen35_dense_attention_layer`]'s own `cached_len` mask.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn qwen35_dense_attention_padding_is_invisible_to_softmax() {
        fn f32_bytes(values: &[f32]) -> Vec<u8> {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        }

        let vocab_size = 257u64;
        let embedding = 4u64;
        let feed_forward = 4u64;
        let query_heads = 1u64;
        let kv_heads = 1u64;
        let rotary_dim = 2u64;
        let attn_head_dim = 4u64;

        let mut tokens: Vec<String> = (0..=255u8)
            .map(|byte| alloc::format!("<0x{byte:02X}>"))
            .collect();
        tokens.push(String::from("<eos-marker>"));

        let token_embd = f32_bytes(&vec![0.05f32; (embedding * vocab_size) as usize]);
        let norm_weight = f32_bytes(&vec![1.0f32; embedding as usize]);
        let head_norm_weight = f32_bytes(&vec![1.0f32; attn_head_dim as usize]);
        let q_weight = f32_bytes(&vec![
            0.05f32;
            (embedding * query_heads * attn_head_dim * 2) as usize
        ]);
        let kv_weight = f32_bytes(&vec![
            0.05f32;
            (embedding * kv_heads * attn_head_dim) as usize
        ]);
        let output_weight = f32_bytes(&vec![
            0.05f32;
            (query_heads * attn_head_dim * embedding) as usize
        ]);
        let ffn_gate_up = f32_bytes(&vec![0.05f32; (embedding * feed_forward) as usize]);
        let ffn_down = f32_bytes(&vec![0.05f32; (feed_forward * embedding) as usize]);
        let output_table = f32_bytes(&vec![0.05f32; (embedding * vocab_size) as usize]);

        let model = GgufModel {
            version: 3,
            metadata: vec![
                (
                    "general.architecture".to_string(),
                    Value::String("qwen35".to_string()),
                ),
                (
                    "qwen35.embedding_length".to_string(),
                    Value::U32(embedding as u32),
                ),
                (
                    "qwen35.feed_forward_length".to_string(),
                    Value::U32(feed_forward as u32),
                ),
                (
                    "qwen35.attention.head_count".to_string(),
                    Value::U32(query_heads as u32),
                ),
                (
                    "qwen35.attention.head_count_kv".to_string(),
                    Value::U32(kv_heads as u32),
                ),
                ("qwen35.block_count".to_string(), Value::U32(1)),
                (
                    "qwen35.rope.dimension_count".to_string(),
                    Value::U32(rotary_dim as u32),
                ),
                (
                    "qwen35.attention.key_length".to_string(),
                    Value::U32(attn_head_dim as u32),
                ),
                ("qwen35.full_attention_interval".to_string(), Value::U32(1)),
                ("qwen35.ssm.conv_kernel".to_string(), Value::U32(2)),
                ("qwen35.ssm.state_size".to_string(), Value::U32(1)),
                ("qwen35.ssm.group_count".to_string(), Value::U32(1)),
                ("qwen35.ssm.time_step_rank".to_string(), Value::U32(1)),
                ("qwen35.ssm.inner_size".to_string(), Value::U32(1)),
                (
                    "tokenizer.ggml.model".to_string(),
                    Value::String("gpt2".to_string()),
                ),
                (
                    "tokenizer.ggml.tokens".to_string(),
                    Value::Array(MetadataArray::String(tokens)),
                ),
                (
                    "tokenizer.ggml.merges".to_string(),
                    Value::Array(MetadataArray::String(Vec::new())),
                ),
            ],
            tensors: vec![
                TensorPayload {
                    name: "token_embd.weight".to_string(),
                    dims: dims(&[embedding, vocab_size]),
                    ggml_type: WireType::F32,
                    data: &token_embd,
                },
                TensorPayload {
                    name: "blk.0.attn_norm.weight".to_string(),
                    dims: dims(&[embedding]),
                    ggml_type: WireType::F32,
                    data: &norm_weight,
                },
                TensorPayload {
                    name: "blk.0.post_attention_norm.weight".to_string(),
                    dims: dims(&[embedding]),
                    ggml_type: WireType::F32,
                    data: &norm_weight,
                },
                TensorPayload {
                    name: "blk.0.attn_q.weight".to_string(),
                    dims: dims(&[embedding, query_heads * attn_head_dim * 2]),
                    ggml_type: WireType::F32,
                    data: &q_weight,
                },
                TensorPayload {
                    name: "blk.0.attn_k.weight".to_string(),
                    dims: dims(&[embedding, kv_heads * attn_head_dim]),
                    ggml_type: WireType::F32,
                    data: &kv_weight,
                },
                TensorPayload {
                    name: "blk.0.attn_v.weight".to_string(),
                    dims: dims(&[embedding, kv_heads * attn_head_dim]),
                    ggml_type: WireType::F32,
                    data: &kv_weight,
                },
                TensorPayload {
                    name: "blk.0.attn_output.weight".to_string(),
                    dims: dims(&[query_heads * attn_head_dim, embedding]),
                    ggml_type: WireType::F32,
                    data: &output_weight,
                },
                TensorPayload {
                    name: "blk.0.attn_q_norm.weight".to_string(),
                    dims: dims(&[attn_head_dim]),
                    ggml_type: WireType::F32,
                    data: &head_norm_weight,
                },
                TensorPayload {
                    name: "blk.0.attn_k_norm.weight".to_string(),
                    dims: dims(&[attn_head_dim]),
                    ggml_type: WireType::F32,
                    data: &head_norm_weight,
                },
                TensorPayload {
                    name: "blk.0.ffn_gate.weight".to_string(),
                    dims: dims(&[embedding, feed_forward]),
                    ggml_type: WireType::F32,
                    data: &ffn_gate_up,
                },
                TensorPayload {
                    name: "blk.0.ffn_up.weight".to_string(),
                    dims: dims(&[embedding, feed_forward]),
                    ggml_type: WireType::F32,
                    data: &ffn_gate_up,
                },
                TensorPayload {
                    name: "blk.0.ffn_down.weight".to_string(),
                    dims: dims(&[feed_forward, embedding]),
                    ggml_type: WireType::F32,
                    data: &ffn_down,
                },
                TensorPayload {
                    name: "output_norm.weight".to_string(),
                    dims: dims(&[embedding]),
                    ggml_type: WireType::F32,
                    data: &norm_weight,
                },
                TensorPayload {
                    name: "output.weight".to_string(),
                    dims: dims(&[embedding, vocab_size]),
                    ggml_type: WireType::F32,
                    data: &output_table,
                },
            ],
        };

        let file_bytes =
            write_complete(&model).expect("writes a minimal one-layer qwen35 gguf fixture");
        let parsed = proxima_gguf::pipe::parse_complete(&file_bytes)
            .expect("parses the minimal one-layer qwen35 gguf fixture");
        let loaded = LoadedModel::load(&parsed, &file_bytes)
            .expect("loads the minimal one-layer qwen35 checkpoint through the public path");

        let base_config = ServingConfig {
            kv_cache_key_quant: WireType::F32,
            kv_cache_value_quant: WireType::F32,
            flash_attention: false,
            batch_size: 0,
            ubatch_size: 0,
            gpu_layers: GPU_LAYERS_ALL,
            reasoning_budget: 0,
            ..ServingConfig::default()
        };
        let max_tokens = 8usize;

        let unpadded_config = ServingConfig {
            kv_bucket_tokens: 1,
            ..base_config
        };
        let mut unpadded_runtime = BackendRuntime::new(&unpadded_config);
        let mut unpadded_logits: Vec<Vec<f32>> = Vec::new();
        let unpadded = loaded
            .run_decode_loop_observed(
                "A",
                max_tokens,
                &unpadded_config,
                &mut unpadded_runtime,
                None,
                &mut super::LogitsSink::Collect(&mut unpadded_logits),
                &mut |_event| ControlFlow::Continue(()),
            )
            .expect("runs the unpadded (kv_bucket_tokens=1) qwen35 dense-attention decode loop");

        let padded_config = ServingConfig {
            kv_bucket_tokens: 32,
            ..base_config
        };
        let mut padded_runtime = BackendRuntime::new(&padded_config);
        let mut padded_logits: Vec<Vec<f32>> = Vec::new();
        let padded = loaded
            .run_decode_loop_observed(
                "A",
                max_tokens,
                &padded_config,
                &mut padded_runtime,
                None,
                &mut super::LogitsSink::Collect(&mut padded_logits),
                &mut |_event| ControlFlow::Continue(()),
            )
            .expect("runs the padded (kv_bucket_tokens=32) qwen35 dense-attention decode loop");

        assert_eq!(
            unpadded.0, padded.0,
            "the DenseAttention arm's padded cache must make a bucket's own \
             zero-padding invisible to softmax -- rounding cached_len up must never \
             change which token is emitted"
        );

        let unpadded_last = unpadded_logits
            .last()
            .expect("the unpadded decode loop must observe at least one logits row");
        let padded_last = padded_logits
            .last()
            .expect("the padded decode loop must observe at least one logits row");

        let max_abs_diff = unpadded_last
            .iter()
            .zip(padded_last.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);

        std::println!(
            "qwen35_dense_attention_padding max_abs_diff={max_abs_diff} \
             unpadded={unpadded_last:?} padded={padded_last:?}"
        );
        assert!(
            max_abs_diff == 0.0,
            "kv_bucket_tokens=32's own zero-padded cached rows must be invisible to the \
             DenseAttention arm's softmax, the same guarantee the Attention arm's fused \
             CachedAttention op already provides -- max_abs_diff={max_abs_diff} between \
             unpadded (bucket=1) and padded (bucket=32) last-row logits proves it is not"
        );
    }

    /// Degenerate control: if the eos comparison were broken (e.g. always
    /// `false`), this test's scripted eos-first source would run the full
    /// budget instead of stopping on step 1 -- confirming the two tests
    /// above are not passing by coincidence of never actually comparing
    /// against `vocab.eos_token_id()`.
    #[test]
    fn stops_on_the_very_first_token_when_it_is_eos() {
        let vocab = vocab_with_eos(32_000);
        let mut calls = 0usize;

        let (generated_ids, stopped_by_eos) = decode_until_stop_or_budget(
            &vocab,
            10,
            0,
            |_step| {
                calls += 1;
                Ok(32_000)
            },
            &mut |_event| ControlFlow::Continue(()),
        )
        .expect("scripted token source never errors");

        assert!(
            generated_ids.is_empty(),
            "an immediate eos must produce zero generated ids"
        );
        assert!(stopped_by_eos);
        assert_eq!(
            calls, 1,
            "must stop after exactly one call, not run toward the budget of 10"
        );
    }

    /// [`TokenEvent`]'s own contract, proved end to end against a scripted
    /// source: exactly one [`Phase::Prefill`] event (carrying the prompt
    /// token count this call was given, at step `0`), then one
    /// [`Phase::Token`] event per generated token, in order -- concatenating
    /// every [`Phase::Token`] event's `text_piece` reproduces
    /// [`proxima_tokenizer::decode`]'s own output on the same ids, and
    /// every [`Phase::Token`] event's `token_id` is the matching entry of
    /// the returned `Vec<u32>`.
    #[test]
    fn streams_one_prefill_event_then_one_token_event_per_generated_token() {
        let vocab = vocab_with_eos(32_000);
        // 'H', 'i', '!' -- three tokens spelling one word this vocab's own
        // base-byte alphabet can decode without any multibyte splitting.
        let scripted_tokens = [b'H' as u32, b'i' as u32, b'!' as u32];
        let prompt_token_count = 5;

        let mut events: Vec<(Phase, u32, String, usize)> = Vec::new();
        let (generated_ids, stopped_by_eos) = decode_until_stop_or_budget(
            &vocab,
            scripted_tokens.len(),
            prompt_token_count,
            |step| Ok(scripted_tokens[step]),
            &mut |event: TokenEvent<'_>| {
                events.push((
                    event.phase,
                    event.token_id,
                    String::from(event.text_piece),
                    event.step,
                ));
                ControlFlow::Continue(())
            },
        )
        .expect("scripted token source never errors");

        assert!(!stopped_by_eos, "the scripted source never emits eos");
        assert_eq!(generated_ids, alloc::vec![72, 105, 33]);

        let prefill_events: Vec<_> = events
            .iter()
            .filter(|(phase, ..)| matches!(phase, Phase::Prefill { .. }))
            .collect();
        assert_eq!(
            prefill_events.len(),
            1,
            "exactly one prefill event, regardless of how many tokens follow"
        );
        let (prefill_phase, _, _, prefill_step) = prefill_events[0];
        assert_eq!(
            *prefill_phase,
            Phase::Prefill {
                prompt_tokens: prompt_token_count
            },
            "prefill must carry this call's own prompt token count"
        );
        assert_eq!(*prefill_step, 0, "prefill only ever happens at step 0");

        let token_events: Vec<_> = events
            .iter()
            .filter(|(phase, ..)| matches!(phase, Phase::Token))
            .collect();
        assert_eq!(
            token_events.len(),
            scripted_tokens.len(),
            "one Token event per generated token, none skipped or doubled"
        );
        let token_ids: Vec<u32> = token_events.iter().map(|(_, id, ..)| *id).collect();
        assert_eq!(
            token_ids, generated_ids,
            "Token event ids must equal the returned ids, in order"
        );

        let streamed_text: String = token_events
            .iter()
            .map(|(_, _, piece, _)| piece.as_str())
            .collect();
        let expected_text = proxima_tokenizer::decode(&generated_ids, &vocab)
            .expect("scripted ids all resolve to real vocab bytes");
        assert_eq!(
            streamed_text, expected_text,
            "concatenated Token event text must equal a one-shot decode of the same ids"
        );
    }

    /// [`ControlFlow::Break`]'s own contract: returning it from `on_token`
    /// ends decoding after that token, short of `max_tokens`, and is reported
    /// the same way running out of budget is -- never mistaken for the
    /// model's own eos.
    #[test]
    fn control_stop_ends_decoding_early_and_is_not_reported_as_eos() {
        let vocab = vocab_with_eos(32_000);
        let scripted_tokens = [b'H' as u32, b'i' as u32, b'!' as u32, b'?' as u32];
        let mut token_events_seen = 0usize;

        let (generated_ids, stopped_by_eos) = decode_until_stop_or_budget(
            &vocab,
            scripted_tokens.len(),
            0,
            |step| Ok(scripted_tokens[step]),
            &mut |event: TokenEvent<'_>| {
                if matches!(event.phase, Phase::Token) {
                    token_events_seen += 1;
                    if token_events_seen == 3 {
                        return ControlFlow::Break(());
                    }
                }
                ControlFlow::Continue(())
            },
        )
        .expect("scripted token source never errors");

        assert_eq!(
            generated_ids,
            alloc::vec![72, 105, 33],
            "must stop right after the 3rd token, never pulling the 4th"
        );
        assert_eq!(token_events_seen, 3);
        assert!(
            !stopped_by_eos,
            "a caller-requested Stop must report finished=false, same as budget exhaustion"
        );
    }

    /// The regression this module exists to fix, at proxima 818f5e46: a
    /// long prompt with a small `max_tokens` budget can legitimately end
    /// mid multibyte character (a real Qwen3 checkpoint stopping mid
    /// emoji/CJK glyph is the exact production report). Byte `0xE4` is
    /// `vocab_with_eos`'s own `<0xE4>` byte-fallback token -- the first of
    /// three bytes ([`crate::generate`]'s own `hex_fallback_token` fixture
    /// doc) a real 3-byte UTF-8 codepoint like `中` starts with -- and this
    /// vocab never supplies the other two, so the budget runs out with an
    /// incomplete lead byte still pending. Neither
    /// [`decode_until_stop_or_budget`] nor [`proxima_tokenizer::decode`] on
    /// its returned ids may fail over that: the caller-visible outcome is
    /// the valid prefix plus one U+FFFD, matching `proxima_tokenizer::decode`'s
    /// own "flag, don't drop" contract for a one-shot decode of the same
    /// truncated ids.
    #[test]
    fn budget_ending_mid_multibyte_character_flags_instead_of_erroring() {
        let vocab = vocab_with_eos(32_000);
        let scripted_tokens = [b'H' as u32, b'i' as u32, 0xE4u32];

        let (generated_ids, stopped_by_eos) = decode_until_stop_or_budget(
            &vocab,
            scripted_tokens.len(),
            0,
            |step| Ok(scripted_tokens[step]),
            &mut |_event| ControlFlow::Continue(()),
        )
        .expect("an incomplete trailing multibyte sequence must never error");

        assert_eq!(
            generated_ids,
            alloc::vec![72, 105, 0xE4],
            "the incomplete lead byte's own id is still a real generated id"
        );
        assert!(
            !stopped_by_eos,
            "the budget ran out; the model never emitted its own eos"
        );

        let text = proxima_tokenizer::decode(&generated_ids, &vocab)
            .expect("a one-shot decode of the same truncated ids must never error either");
        assert_eq!(
            text, "Hi\u{FFFD}",
            "the complete prefix stays intact and the unfinished tail becomes one U+FFFD"
        );
    }

    /// The other way an incomplete lead byte can resolve: not by the
    /// budget ending, but by the very next token NOT being a valid
    /// continuation byte (autoregressive sampling gives no guarantee that
    /// consecutive token ids retrace one contiguous encoder segmentation).
    /// [`decode_streamed_piece`]'s own doc: a stale pending tail that turns
    /// out unresolvable resolves to one U+FFFD and decoding resumes on
    /// whatever bytes follow, so decoding one bad run never fails the
    /// whole call.
    #[test]
    fn incomplete_lead_byte_followed_by_a_non_continuation_byte_flags_and_resumes() {
        let vocab = vocab_with_eos(32_000);
        // 0xE4 starts a 3-byte sequence; 'H'/'i' are plain ASCII and can
        // never be valid UTF-8 continuation bytes (those are 0x80-0xBF).
        let scripted_tokens = [0xE4u32, b'H' as u32, b'i' as u32];

        let mut events: Vec<String> = Vec::new();
        let (generated_ids, stopped_by_eos) = decode_until_stop_or_budget(
            &vocab,
            scripted_tokens.len(),
            0,
            |step| Ok(scripted_tokens[step]),
            &mut |event: TokenEvent<'_>| {
                if matches!(event.phase, Phase::Token) {
                    events.push(String::from(event.text_piece));
                }
                ControlFlow::Continue(())
            },
        )
        .expect("a non-continuing follow-on token must never error");

        assert_eq!(generated_ids, alloc::vec![0xE4, 72, 105]);
        assert!(!stopped_by_eos);
        assert_eq!(
            events.join(""),
            "\u{FFFD}Hi",
            "the unresolvable lead byte flags as one U+FFFD, then decoding \
             resumes normally on the bytes that follow it"
        );
    }

    /// The one real check for `LoadedModel`'s `Drop` impl: loads a real
    /// checkpoint on Metal, runs a handful of decode steps (so the resident
    /// weight buffers and the checkpoint-mapping no-copy buffer are both
    /// actually populated, not just registered), drops it, and prints
    /// `MTLDevice::currentAllocatedSize` alongside the process's own
    /// `phys_footprint` before load / after generate / after drop -- the
    /// artifact this row's own INVARIANT ("dropping a `LoadedModel` releases
    /// every device allocation it caused") is checked against. `#[ignore]`d
    /// like every other host-local fixture in this crate
    /// ([`real_openchat_file`]'s own doc): this prints evidence for a human
    /// to read, it does not assert a byte-exact threshold, because the OS's
    /// own `phys_footprint` also reflects unrelated process state (allocator
    /// arenas, thread stacks) this test does not control.
    #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
    mod release_on_drop_real_model {
        use core::ffi::c_void;
        use std::os::fd::AsFd;

        use super::super::LoadedModel;
        use crate::serving::GPU_LAYERS_ALL;

        struct MappedGguf {
            base: *mut u8,
            len: usize,
            _file: std::fs::File,
        }

        impl MappedGguf {
            fn open(path: &std::path::Path) -> std::io::Result<Self> {
                let file = std::fs::File::open(path)?;
                let len = usize::try_from(file.metadata()?.len())
                    .expect("fixture file length fits in usize");
                // SAFETY: `len` matches the just-opened file's own length;
                // `file` is kept alive in `_file` for as long as `base` is
                // used, and the mapping is read-only/private so no writer
                // can observe or race it.
                let base = unsafe {
                    rustix::mm::mmap(
                        core::ptr::null_mut(),
                        len,
                        rustix::mm::ProtFlags::READ,
                        rustix::mm::MapFlags::PRIVATE,
                        file.as_fd(),
                        0,
                    )
                }
                .expect("mmap host-local release-gate gguf fixture")
                .cast::<u8>();
                Ok(Self {
                    base,
                    len,
                    _file: file,
                })
            }

            fn as_slice(&self) -> &[u8] {
                // SAFETY: `base` points at `len` bytes mapped for `self`'s
                // whole lifetime; this borrows `self` immutably, so nothing
                // can unmap the region while the returned slice is alive.
                unsafe { core::slice::from_raw_parts(self.base, self.len) }
            }

            /// Explicit rather than left to `Drop` -- this test reads the
            /// device/process footprint immediately after unmapping, and
            /// that ordering (drop the model, THEN unmap, THEN measure) is
            /// the whole point: `register_checkpoint_mapping`'s own no-copy
            /// buffer aliases this mapping directly, so a real fix must
            /// release Metal's reference to it before the mapping itself
            /// goes away, not merely before the test happens to check.
            fn unmap(self) {
                // SAFETY: `base`/`len` are exactly what `open`'s `mmap`
                // call returned; nothing else unmaps this region.
                let _ = unsafe { rustix::mm::munmap(self.base.cast::<c_void>(), self.len) };
                core::mem::forget(self);
            }
        }

        impl Drop for MappedGguf {
            fn drop(&mut self) {
                // SAFETY: only reachable if `unmap` was never called --
                // `unmap` itself `mem::forget`s `self` after unmapping.
                let _ = unsafe { rustix::mm::munmap(self.base.cast::<c_void>(), self.len) };
            }
        }

        const FIXTURE_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-3e4cb14174460404e7a233e531675303b2fbf7749c02f91864fe311ab6344e4f";

        #[test]
        #[ignore = "depends on a host-local ollama gguf blob outside this repo"]
        fn dropping_a_loaded_model_releases_its_device_buffers() {
            crate::test_support::require_fixture(FIXTURE_PATH, None);
            let path = std::path::Path::new(FIXTURE_PATH);

            let before_load = omega::metal::current_allocated_size();
            let before_load_footprint = super::super::phys_footprint_bytes();
            std::println!(
                "before_load current_allocated_size={before_load:?} phys_footprint_bytes={before_load_footprint}"
            );

            let mapped = MappedGguf::open(path).expect("mmap host-local release-gate fixture");
            let file_bytes = mapped.as_slice();
            let parsed = proxima_gguf::pipe::parse_complete(file_bytes)
                .expect("parse host-local release-gate fixture");
            let model = LoadedModel::load(&parsed, file_bytes)
                .expect("load real checkpoint through the public path");

            let serving_config =
                super::super::supported_serving_config(GPU_LAYERS_ALL, omega::MathMode::default());
            let (generated_ids, _text, _stopped_by_eos) = model
                .generate_with_serving_config("The quick brown fox", 4, serving_config)
                .expect("decode 4 tokens on Metal");
            assert_eq!(generated_ids.len(), 4, "must actually run 4 decode steps");

            let after_generate = omega::metal::current_allocated_size();
            let after_generate_footprint = super::super::phys_footprint_bytes();
            std::println!(
                "after_generate current_allocated_size={after_generate:?} phys_footprint_bytes={after_generate_footprint}"
            );

            drop(model);
            mapped.unmap();

            let after_drop = omega::metal::current_allocated_size();
            let after_drop_footprint = super::super::phys_footprint_bytes();
            std::println!(
                "after_drop current_allocated_size={after_drop:?} phys_footprint_bytes={after_drop_footprint}"
            );
        }
    }
}

/// The defect ROW 329's slice found, proved directly: every
/// [`BackendRuntime::placed_plans`] build closure now routes through
/// [`BackendRuntime::build_placed_plan`], so a shape first resolved through
/// a diagnostic path (`evaluate_op_timed_with_placements`/
/// `evaluate_dispatch_timed_with_placements`) carries the SAME math mode
/// and dispatch type a later hit from the production path
/// ([`BackendRuntime::evaluate_with_placements`]) would have applied,
/// instead of silently keeping [`omega::metal::MathMode::default`]/
/// [`omega::metal::DispatchType::default`].
///
/// Exercises [`BackendRuntime::build_placed_plan`] directly against the
/// smallest program that plans without touching a Metal device
/// ([`omega::metal::plan`]/[`omega::metal::plan_named`] only resolve
/// shapes and codecs -- no `MTLDevice` is opened until an `execute_plan*`
/// call, per that function's own doc), rather than driving a full decode
/// step through the driver.
#[cfg(all(
    test,
    feature = "metal-output-placement",
    feature = "instrument",
    target_os = "macos"
))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
pub(super) mod placed_plan_mode_tests {
    use alloc::collections::BTreeSet;
    use alloc::string::String;
    use alloc::vec::Vec;

    use proxima_tensor::{
        DType, Extent, IndexMap, NodeId, Op, QuantizedBlock, ScalarOp, append, projection,
    };

    use super::BackendRuntime;

    /// `Input(name = "x") -> Elementwise(Identity)` -- the same minimal
    /// identity shape `omega`'s own `metal_output_placement.rs` test uses,
    /// named so [`proxima_tensor::resolve_named_blocks`] (which
    /// [`omega::plan_named`] calls) can bind it.
    fn named_identity_program() -> (Vec<Op>, NodeId) {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: Some(String::from("x")),
            },
        );
        let identity = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: alloc::vec![(source, IndexMap::Affine(projection(1, &[0])))],
                name: None,
            },
        );
        (program, identity)
    }

    #[test]
    fn build_placed_plan_applies_the_runtimes_math_mode_and_dispatch_type() {
        let (program, identity_node) = named_identity_program();
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let named: [(&str, QuantizedBlock<'_>); 1] = [("x", QuantizedBlock::Float32(&data))];
        let resident_names: BTreeSet<&str> = BTreeSet::new();

        let plan = BackendRuntime::build_placed_plan(
            &program,
            &[],
            &named,
            &[identity_node],
            &resident_names,
            &[],
            &super::PlanNumerics {
                math_mode: omega::metal::MathMode::Safe,
                numeric_policy: proxima_tensor::NumericPolicy::bit_exact(),
                dispatch_type: omega::metal::DispatchType::Serial,
                plan_time_constants: false,
            },
        )
        .expect("plans the identity program under an explicit non-default mode");

        assert_eq!(
            plan.math_mode(),
            omega::metal::MathMode::Safe,
            "a freshly built placed plan must carry the caller's math mode, \
             not MathMode::default() (Relaxed)"
        );
        assert_eq!(
            plan.dispatch_type(),
            omega::metal::DispatchType::Serial,
            "a freshly built placed plan must carry the caller's dispatch type, \
             not DispatchType::default() (Concurrent)"
        );
        assert_eq!(
            plan.numeric_policy(),
            proxima_tensor::NumericPolicy::bit_exact(),
            "a freshly built placed plan must carry the caller's OWN numeric policy -- the \
             policy `plan_named_placed` was constructed under, never a later math-mode setter"
        );
    }

    /// Names the invariant `build_placed_plan`'s construction-time wiring
    /// preserves: a caller declaring [`proxima_tensor::NumericPolicy::llama_relaxed`]
    /// (this crate's own default, `ServingConfig::numeric_policy`'s doc)
    /// alongside `MathMode::Safe` (a narrower compiled mode than the bound
    /// policy grants) must still see `llama_relaxed()` on the resulting
    /// plan's `numeric_policy()` -- `set_math_mode` only narrows the
    /// COMPILED mode within the already-bound policy
    /// (`omega::metal::Plan::set_math_mode`'s own doc); it can never widen
    /// or replace the policy the plan was constructed under.
    #[test]
    fn build_placed_plan_s_numeric_policy_survives_a_narrower_math_mode() {
        let (program, identity_node) = named_identity_program();
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let named: [(&str, QuantizedBlock<'_>); 1] = [("x", QuantizedBlock::Float32(&data))];
        let resident_names: BTreeSet<&str> = BTreeSet::new();

        let plan = BackendRuntime::build_placed_plan(
            &program,
            &[],
            &named,
            &[identity_node],
            &resident_names,
            &[],
            &super::PlanNumerics {
                math_mode: omega::metal::MathMode::Safe,
                numeric_policy: proxima_tensor::NumericPolicy::llama_relaxed(),
                dispatch_type: omega::metal::DispatchType::Serial,
                plan_time_constants: false,
            },
        )
        .expect("Safe never needs a permission llama_relaxed() withholds, so narrowing succeeds");

        assert_eq!(
            plan.numeric_policy(),
            proxima_tensor::NumericPolicy::llama_relaxed(),
            "the plan's bound numeric_policy must be exactly what it was constructed under, \
             regardless of the narrower compiled math_mode -- if this reads bit_exact(), the \
             construction-time wiring in build_placed_plan regressed"
        );
        assert_eq!(plan.math_mode(), omega::metal::MathMode::Safe);
    }

    /// The mismatch direction: a caller cannot narrow to a `MathMode` that
    /// needs a permission the plan's bound policy withholds.
    #[test]
    fn build_placed_plan_s_math_mode_narrowing_rejects_a_permission_the_bound_policy_withholds() {
        let (program, identity_node) = named_identity_program();
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let named: [(&str, QuantizedBlock<'_>); 1] = [("x", QuantizedBlock::Float32(&data))];
        let resident_names: BTreeSet<&str> = BTreeSet::new();

        let error = match BackendRuntime::build_placed_plan(
            &program,
            &[],
            &named,
            &[identity_node],
            &resident_names,
            &[],
            &super::PlanNumerics {
                math_mode: omega::metal::MathMode::Fast,
                numeric_policy: proxima_tensor::NumericPolicy::bit_exact(),
                dispatch_type: omega::metal::DispatchType::Serial,
                plan_time_constants: false,
            },
        ) {
            Ok(_) => panic!(
                "Fast needs nan_assumptions/signed_zero/approx_functions, bit_exact grants none"
            ),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("bit_exact") || error.to_string().contains("false"),
            "error must name the bound policy: {error}"
        );
    }
}

/// [`LoadedModel::apply_memory_fit_gate`]'s own contract, exercised
/// directly against a struct-literal [`LoadedModel`] -- private-field
/// construction is legitimate here (same module tree) and cheaper than a
/// full loadable checkpoint: the gate only ever reads
/// `self.checkpoint_weight_bytes`/`self.architecture`, never `self.weights`/
/// `self.program`/`self.vocab`, so those fields are empty stand-ins.
/// Requires a real Metal device (`omega::metal::system_memory_facts`), the
/// same requirement [`crate::memory_fit`]'s own doc names for anything
/// beyond its pure formulas.
#[cfg(all(test, feature = "metal", target_os = "macos"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
pub(super) mod memory_fit_gate_tests {
    use alloc::format;
    use alloc::string::String;
    use alloc::vec::Vec;

    use proxima_tokenizer::Vocab;

    use crate::bind::{BoundWeights, ModelArchitecture};
    use crate::serving::ServingConfig;

    use super::LoadedModel;

    /// A minimal valid byte-level BPE vocab -- every base-byte token
    /// present ([`Vocab::new`]'s own precondition), no merges, no special
    /// tokens beyond the one this gate never reads anyway (the fit gate
    /// touches `self.architecture`/`self.checkpoint_weight_bytes` only).
    fn tiny_vocab() -> Vocab {
        let tokens: Vec<String> = (0..=255u8).map(|byte| format!("<0x{byte:02X}>")).collect();
        Vocab::new(tokens, &[], None, None, None).expect("minimal vocab builds")
    }

    fn tiny_architecture() -> ModelArchitecture {
        ModelArchitecture {
            vocab: 1,
            embedding: 1,
            feed_forward: 1,
            query_heads: 1,
            kv_heads: 2,
            kv_heads_by_layer: vec![2; 2],
            head_dim: 64,
            block_count: 2,
            expert_count: 0,
            expert_used_count: 0,
            rope_freq_base: 10_000.0,
            rms_epsilon: 1e-5,
            tied_embeddings: false,
        }
    }

    fn model_with(dense_weight_bytes: u64) -> LoadedModel<'static> {
        LoadedModel {
            weights: BoundWeights {
                resident_bytes: 0,
                owned: Vec::new(),
                packed: Vec::new(),
                packed_owned: Vec::new(),
                precision: &[],
            },
            architecture: tiny_architecture(),
            architecture_impl: None,
            checkpoint_weight_bytes: crate::memory_fit::WeightClassBytes {
                dense_bytes: dense_weight_bytes,
                expert_bytes: 0,
                table_bytes: 0,
                ssm_state_bytes: 0,
            },
            mapping_residency_rung: crate::mapping_residency::ResidencyRung::Prefault,
            model_name: None,
            checkpoint_bytes: dense_weight_bytes as usize,
            checkpoint_mapping: &[],
            vocab: tiny_vocab(),
            program: Vec::new(),
            logits_root: proxima_tensor::op::NodeId(0),
            hidden_root: None,
            layer_roots: Vec::new(),
            residual_roots: Vec::new(),
            qwen35moe_layer_diagnostics: Vec::new(),
            router_roots: Vec::new(),
            moe_sites: proxima_tensor::spec::MoeSites::default(),
            single_position_step: false,
            qwen35moe_hparams: None,
            #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
            single_range: None,
            expert_slab: std::sync::Mutex::new(crate::expert_slab::ExpertSlab::new()),
            expert_sidecar: None,
        }
    }

    /// (d) the gate off: an absurd `context_length` that would otherwise
    /// need reducing (or would exceed the limit outright) passes through
    /// completely unchanged when `gpu_memory_fit` is `false` -- the
    /// caller's explicit opt-out.
    #[test]
    fn gate_off_leaves_an_otherwise_infeasible_context_length_untouched() {
        let model = model_with(1_000_000);
        let mut serving_config = ServingConfig {
            gpu_memory_fit: false,
            context_length: u32::MAX,
            ..ServingConfig::default()
        };

        model
            .apply_memory_fit_gate(&mut serving_config)
            .expect("gate must be a no-op when gpu_memory_fit is false");

        assert_eq!(
            serving_config.context_length,
            u32::MAX,
            "gate must not touch context_length when the caller opted out"
        );
    }

    /// (a) fits: this fixture's tiny weights and default 131072 context
    /// budget are well inside any real Metal device's own reported limit.
    #[test]
    fn gate_on_leaves_a_generously_fitting_context_length_unchanged() {
        let model = model_with(1_000_000);
        let mut serving_config = ServingConfig {
            gpu_memory_fit: true,
            ..ServingConfig::default()
        };
        let requested = serving_config.context_length;

        model
            .apply_memory_fit_gate(&mut serving_config)
            .expect("a real device's own limit must comfortably fit this fixture's budget");

        assert_eq!(
            serving_config.context_length, requested,
            "a generously fitting budget must not reduce context_length"
        );
    }

    #[test]
    fn configured_memory_ceiling_rejects_weights_before_device_allocation() {
        const FOUR_GIB: u64 = 4 * 1024 * 1024 * 1024;
        const FIVE_GIB: u64 = 5 * 1024 * 1024 * 1024;

        let model = model_with(FIVE_GIB);
        let mut serving_config = ServingConfig {
            gpu_memory_fit: true,
            gpu_memory_limit_bytes: Some(FOUR_GIB),
            context_length: 1,
            ..ServingConfig::default()
        };

        let error = model
            .apply_memory_fit_gate(&mut serving_config)
            .expect_err("five GiB of weights must not pass a four GiB device ceiling");

        assert!(
            matches!(
                error,
                crate::error::InteropError::MemoryBudgetExceeded {
                    dense_weights_bytes: FIVE_GIB,
                    limit_bytes: FOUR_GIB,
                    os_headroom_bytes: 0,
                    ..
                }
            ),
            "the typed error must retain the configured ceiling and offending weight class: {error:?}"
        );
    }

    /// [`PrefixState`] against the real host-local qwen3-1.7B checkpoint
    /// [`crate::test_support::qwen3_gguf_path`] resolves -- this checkpoint
    /// is qk-norm, so ordinary [`LoadedModel::generate_with_serving_config`]
    /// on Metal takes [`LoadedModel::run_decode_loop_placed_kv`]'s single-
    /// range fast path (`qwen3_split_half_rope_cpu_and_metal_greedy_decode_match`'s
    /// own doc, `proxima-model-interop/src/quality.rs`), which has no host-
    /// side [`LayerCacheState`] to report as a [`PrefixState`] at all --
    /// every test here therefore exercises the two-range SPLIT path
    /// [`LoadedModel::prefill_prefix`]/[`LoadedModel::generate_from_prefix`]
    /// force via `force_two_range: true`, not the fused single-range kernel
    /// ordinary callers get. `#[ignore]`d like every other host-local
    /// fixture in this crate.
    ///
    /// **Residual, run and measured, not hidden (ROW 406):** the two parity
    /// tests below currently FAIL on this real checkpoint --
    /// `generate_with_serving_config` over the concatenated prompt
    /// degenerates to a repeated-token greedy decode (a real, if
    /// uninteresting, model output at `temperature: 0.0`), while
    /// `generate_from_prefix`'s own resumed decode diverges to unrelated
    /// tokens from the very first generated position. The forced two-range
    /// path is exercised here in a shape that, so far as this landing's own
    /// reading of `proxima-tensor/src/spec.rs`/`omega/src/msl.rs` found, no
    /// existing caller ever produces on Metal: a multi-row query
    /// (`query_rows > 1`, the whole prefix) against a NONZERO cached range
    /// (`cached_len > 0` seeded from a prior [`PrefixState`]). Every
    /// existing two-range Metal caller's own multi-row step is prefill
    /// itself, always at `cached_len == 0`; every step with `cached_len >
    /// 0` is ordinary autoregressive decode, always `query_rows == 1`. This
    /// landing's own `prefill_prefix`/`generate_from_prefix` split is the
    /// first caller to combine the two, and the fused Metal
    /// `CachedAttention` kernel (or its uniform/identity/dispatch plumbing
    /// -- unattributed, no profiler/disassembly evidence gathered this
    /// slice) most likely does not handle that combination correctly. NOT
    /// verified against the CPU evaluator in the time this slice had --
    /// that comparison (does `run_cached_attention`'s CPU oracle, not the
    /// Metal kernel, also fail the same way?) is the next diagnostic step,
    /// left open rather than guessed at.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    mod prefix_state_real_model {
        use core::ffi::c_void;
        use std::os::fd::AsFd;

        #[cfg(all(feature = "metal", feature = "instrument"))]
        use proxima_telemetry::export::Exporter;
        #[cfg(all(feature = "metal", feature = "instrument"))]
        use proxima_telemetry::recorder::Recorder;

        use super::super::{
            BackendRuntime, ControlFlow, LoadedModel, LogitsSink, NodeValuesSink, Phase,
            PrefixState, supported_serving_config,
        };
        use crate::serving::GPU_LAYERS_ALL;

        struct MappedGguf {
            base: *mut u8,
            len: usize,
            _file: std::fs::File,
        }

        impl MappedGguf {
            fn open(path: &std::path::Path) -> std::io::Result<Self> {
                let file = std::fs::File::open(path)?;
                let len = usize::try_from(file.metadata()?.len())
                    .expect("fixture file length fits in usize");
                // SAFETY: `len` matches the just-opened file's own length;
                // `file` is kept alive in `_file` for as long as `base` is
                // used, and the mapping is read-only/private so no writer
                // can observe or race it.
                let base = unsafe {
                    rustix::mm::mmap(
                        core::ptr::null_mut(),
                        len,
                        rustix::mm::ProtFlags::READ,
                        rustix::mm::MapFlags::PRIVATE,
                        file.as_fd(),
                        0,
                    )
                }
                .expect("mmap host-local prefix-state gguf fixture")
                .cast::<u8>();
                Ok(Self {
                    base,
                    len,
                    _file: file,
                })
            }

            fn as_slice(&self) -> &[u8] {
                // SAFETY: `base` points at `len` bytes mapped for `self`'s
                // whole lifetime; this borrows `self` immutably, so nothing
                // can unmap the region while the returned slice is alive.
                unsafe { core::slice::from_raw_parts(self.base, self.len) }
            }
        }

        impl Drop for MappedGguf {
            fn drop(&mut self) {
                // SAFETY: `base`/`len` are exactly what `open`'s `mmap`
                // call returned; nothing else unmaps this region.
                let _ = unsafe { rustix::mm::munmap(self.base.cast::<c_void>(), self.len) };
            }
        }

        fn open_model(mapped: &MappedGguf) -> LoadedModel<'_> {
            let file_bytes = mapped.as_slice();
            let parsed = proxima_gguf::pipe::parse_complete(file_bytes)
                .expect("parse host-local qwen3 gguf fixture");
            LoadedModel::load(&parsed, file_bytes)
                .expect("load real qwen3 checkpoint through the public path")
        }

        /// Greedy (`temperature: 0.0`) so [`Self::generate_from_prefix`] and
        /// [`LoadedModel::generate_with_serving_config`] are directly
        /// comparable token-for-token -- any sampling randomness would make
        /// a mismatch ambiguous between "the primitive is wrong" and "the
        /// rng streams diverged".
        fn greedy_serving_config() -> super::super::ServingConfig<'static> {
            let mut config =
                supported_serving_config(GPU_LAYERS_ALL, crate::test_support::math_mode_from_env());
            config.temperature = 0.0;
            config
        }

        /// Real prose (Arthur Conan Doyle, public domain, "A Scandal in
        /// Bohemia"'s opening) rather than synthetic filler -- this is the
        /// byte-for-byte shape a real chat prompt's own shared system/
        /// history prefix takes. Ends on a newline: the tokenizer boundary
        /// this crate's vocabularies treat as a hard break, so
        /// `tokenize(PREFIX)` is a genuine prefix of `tokenize(PREFIX +
        /// SUFFIX_*)` for every `SUFFIX_*` below (each also opens on its own
        /// clause rather than continuing the prefix's last word).
        const PREFIX: &str = "To Sherlock Holmes she is always THE woman. I have seldom heard him mention her under any other name. In his eyes she eclipses and predominates the whole of her sex. It was not that he felt any emotion akin to love for Irene Adler. All emotions, and that one particularly, were abhorrent to his cold, precise but admirably balanced mind. He was, I take it, the most perfect reasoning and observing machine that the world has seen, but as a lover he would have placed himself in a false position. He never spoke of the softer passions, save with a gibe and a sneer. They were admirable things for the observer—excellent for drawing the veil from men's motives and actions. But for the trained reasoner to admit such intrusions into his own delicate and finely adjusted temperament was to introduce a distracting factor which might throw a doubt upon all his mental results.\n";

        const SUFFIX_A: &str = "Grit in a sensitive instrument, or a crack in one of his own high-power lenses, would not be more disturbing than a strong emotion in a nature such as his.";

        const SUFFIX_B: &str = "And yet there was but one woman to him, and that woman was the late Irene Adler, of dubious and questionable memory.";

        /// Root-cause proof for ROW 406's own open residual: direct
        /// (`force_two_range: false`, [`LoadedModel::generate_with_serving_config`]'s
        /// own path -- Metal's single-range placed-KV fast path for this
        /// checkpoint) vs resumed (`force_two_range: true`, the
        /// [`PrefixState`] two-range path) compared at LOGIT precision, not
        /// post-argmax, across 5 decode steps. `PREFIX` alone tokenizes to
        /// 183 rows -- already past `ATTENTION_SPLIT_KEYS_PER_SPLIT_AT_SCALE`
        /// (128, `omega/src/msl.rs`), the split-at-scale knee `omega/tests/
        /// qwen3_gqa_qk_norm_two_range_parity.rs`'s own regression test pins.
        /// Before `fix(omega): two-range cached attention skips split merge
        /// dispatch` (cherry-picked to `main` ahead of this test), the
        /// two-range `cached_attention_merge_needed` predicate answered
        /// `true` for this op past that knee and routed it through the
        /// single-range-only `ContextSplitMerge` protocol, which
        /// reinterpreted the resumed path's already-correct, already-
        /// normalized attention output as `(max, sum, weighted[head_dim])`
        /// triples and overwrote it with garbage -- exactly the shape the
        /// prior divergent-token failure this test replaces had. With that
        /// fix on `main`, `max_diff` here is noise-floor
        /// (~1e-5, ordinary Metal-vs-Metal reduction-order float noise) at
        /// every one of the 5 steps, and both paths agree on every argmax.
        #[test]
        #[ignore = "depends on a host-local qwen3 gguf checkout outside this repo, and a real Metal device"]
        fn generate_from_prefix_matches_generate_at_logit_precision_across_five_steps() {
            let model_path = crate::test_support::qwen3_gguf_path();
            crate::test_support::require_fixture(&model_path, Some("PROXIMA_QWEN3_GGUF"));
            let mapped = MappedGguf::open(std::path::Path::new(&model_path))
                .expect("mmap host-local qwen3 gguf fixture");
            let model = open_model(&mapped);
            let serving_config = greedy_serving_config();
            let steps = 5;

            let full_prompt = alloc::format!("{PREFIX}{SUFFIX_A}");
            let mut direct_runtime = BackendRuntime::new(&serving_config);
            let mut direct_logits: Vec<Vec<f32>> = Vec::new();
            let (direct_ids, _, _, _direct_final) = model
                .run_decode_loop_observed_seeded(
                    &full_prompt,
                    steps,
                    &serving_config,
                    &mut direct_runtime,
                    None,
                    &mut LogitsSink::Collect(&mut direct_logits),
                    &mut NodeValuesSink::Discard,
                    &mut |_event| ControlFlow::Continue(()),
                    None,
                    false,
                )
                .expect("direct greedy generate over the concatenated prompt");

            let prefix = model
                .prefill_prefix(PREFIX, &serving_config)
                .expect("prefill the shared prefix once");
            let mut resumed_runtime = BackendRuntime::new(&serving_config);
            let seed = PrefixState {
                ids: prefix.ids.clone(),
                layer_caches: prefix.layer_caches.clone(),
                cached_len: prefix.cached_len,
            };
            let mut resumed_logits: Vec<Vec<f32>> = Vec::new();
            let (resumed_ids, _, _, _resumed_final) = model
                .run_decode_loop_observed_seeded(
                    SUFFIX_A,
                    steps,
                    &serving_config,
                    &mut resumed_runtime,
                    None,
                    &mut LogitsSink::Collect(&mut resumed_logits),
                    &mut NodeValuesSink::Discard,
                    &mut |_event| ControlFlow::Continue(()),
                    Some(seed),
                    true,
                )
                .expect("resume decoding from the cached prefix");

            assert_eq!(
                direct_logits.len(),
                resumed_logits.len(),
                "both paths must run the same number of decode steps"
            );
            for (step, (direct_step, resumed_step)) in
                direct_logits.iter().zip(resumed_logits.iter()).enumerate()
            {
                let max_diff = direct_step
                    .iter()
                    .zip(resumed_step.iter())
                    .map(|(expected, actual)| (expected - actual).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_diff < 1e-3,
                    "step={step}: direct and resumed logits diverge past noise floor \
                     (max_diff={max_diff}) -- the two-range split/merge dispatch \
                     regression (omega/src/msl.rs's cached_attention_merge_needed) \
                     if it comes back"
                );
            }
            assert_eq!(
                resumed_ids, direct_ids,
                "resumed and direct must sample the identical greedy tokens over 5 steps"
            );
        }

        /// (a) parity: resuming from a cached prefix must produce the
        /// IDENTICAL greedy token ids as decoding the concatenated prompt
        /// in one call.
        #[test]
        #[ignore = "depends on a host-local qwen3 gguf checkout outside this repo, and a real Metal device"]
        fn generate_from_prefix_matches_generate_over_the_full_prompt_at_temperature_zero() {
            let model_path = crate::test_support::qwen3_gguf_path();
            crate::test_support::require_fixture(&model_path, Some("PROXIMA_QWEN3_GGUF"));
            let mapped = MappedGguf::open(std::path::Path::new(&model_path))
                .expect("mmap host-local qwen3 gguf fixture");
            let model = open_model(&mapped);
            let serving_config = greedy_serving_config();
            let max_tokens = 16;

            let full_prompt = alloc::format!("{PREFIX}{SUFFIX_A}");
            let (direct_ids, direct_text, _) = model
                .generate_with_serving_config(&full_prompt, max_tokens, serving_config)
                .expect("direct greedy generate over the concatenated prompt");

            let prefix = model
                .prefill_prefix(PREFIX, &serving_config)
                .expect("prefill the shared prefix once");
            let (resumed_ids, resumed_text, _) = model
                .generate_from_prefix(
                    &prefix,
                    SUFFIX_A,
                    max_tokens,
                    &serving_config,
                    &mut |_event| ControlFlow::Continue(()),
                )
                .expect("resume decoding from the cached prefix");

            std::println!(
                "prefix_parity direct={direct_text:?} resumed={resumed_text:?} \
                 prefix_len={}",
                prefix.len()
            );
            assert_eq!(
                resumed_ids, direct_ids,
                "resuming from a cached prefix must produce the same greedy token ids \
                 as decoding the full prompt in one call"
            );
        }

        /// (b) reuse: two different suffixes against the SAME prefill --
        /// both parity-correct against their own direct decode, and each
        /// `generate_from_prefix` call's own `Phase::Prefill` event reports
        /// a `prompt_tokens` count bounded by the suffix alone, never the
        /// prefix's own `cached_len` -- the load-bearing proof that the
        /// second (and first) call never re-ran the prefix's own forward
        /// pass, read off the SAME per-step counter
        /// [`super::super::TokenEvent::phase`]'s own doc already promises,
        /// never wall-clock.
        #[test]
        #[ignore = "depends on a host-local qwen3 gguf checkout outside this repo, and a real Metal device"]
        fn generate_from_prefix_reuses_one_prefill_across_two_suffixes() {
            let model_path = crate::test_support::qwen3_gguf_path();
            crate::test_support::require_fixture(&model_path, Some("PROXIMA_QWEN3_GGUF"));
            let mapped = MappedGguf::open(std::path::Path::new(&model_path))
                .expect("mmap host-local qwen3 gguf fixture");
            let model = open_model(&mapped);
            let serving_config = greedy_serving_config();
            let max_tokens = 12;

            let prefix = model
                .prefill_prefix(PREFIX, &serving_config)
                .expect("prefill the shared prefix once");

            for suffix in [SUFFIX_A, SUFFIX_B] {
                let full_prompt = alloc::format!("{PREFIX}{suffix}");
                let (direct_ids, _, _) = model
                    .generate_with_serving_config(&full_prompt, max_tokens, serving_config)
                    .expect("direct greedy generate over the concatenated prompt");

                let mut prefill_rows = 0_usize;
                let (resumed_ids, _, _) = model
                    .generate_from_prefix(
                        &prefix,
                        suffix,
                        max_tokens,
                        &serving_config,
                        &mut |event| {
                            if let Phase::Prefill { prompt_tokens } = event.phase {
                                prefill_rows = prompt_tokens;
                            }
                            ControlFlow::Continue(())
                        },
                    )
                    .expect("resume decoding from the cached prefix");

                assert_eq!(
                    resumed_ids, direct_ids,
                    "suffix {suffix:?} must match its own direct decode"
                );
                assert!(
                    prefill_rows < prefix.len(),
                    "prefill_rows={prefill_rows} must cover only the suffix's own tokens, \
                     never the {}-token cached prefix -- a value this large would mean \
                     the prefix was re-prefilled",
                    prefix.len()
                );
            }
        }

        /// (c) drop: once every [`PrefixState`] this test built goes out of
        /// scope, the model's own resident-buffer count
        /// (`ROW 403`'s `omega::metal::current_allocated_size`, the same
        /// counter `dropping_a_loaded_model_releases_its_device_buffers`
        /// checks) is unaffected -- [`PrefixState`]'s own doc: its fields
        /// are plain host `Vec<f32>` buffers, never a named device
        /// registration, so there is nothing for a device-buffer counter to
        /// see drop at all. This test's own real assertion is therefore
        /// that the count is IDENTICAL immediately before and after the
        /// drop, not merely "close" -- proving the negative
        /// [`PrefixState`]'s doc claims (no device identity to leak)
        /// rather than assuming it.
        #[test]
        #[ignore = "depends on a host-local qwen3 gguf checkout outside this repo, and a real Metal device"]
        fn dropping_a_prefix_state_leaves_the_models_resident_buffer_count_unchanged() {
            let model_path = crate::test_support::qwen3_gguf_path();
            crate::test_support::require_fixture(&model_path, Some("PROXIMA_QWEN3_GGUF"));
            let mapped = MappedGguf::open(std::path::Path::new(&model_path))
                .expect("mmap host-local qwen3 gguf fixture");
            let model = open_model(&mapped);
            let serving_config = greedy_serving_config();

            let prefix = model
                .prefill_prefix(PREFIX, &serving_config)
                .expect("prefill the shared prefix once");
            let before_drop = omega::metal::current_allocated_size();
            drop(prefix);
            let after_drop = omega::metal::current_allocated_size();

            std::println!(
                "prefix_state_drop before_drop={before_drop:?} after_drop={after_drop:?}"
            );
            assert_eq!(
                before_drop, after_drop,
                "PrefixState owns no device buffer, so dropping it must not move the \
                 model's own resident device-allocation count at all"
            );
        }

        /// Proves the prefill mechanism with a COUNT, not a read of the
        /// source (guiding-principle 18): `run_decode_loop_observed_seeded`
        /// on architectures that accept batched positions. Qwen35's
        /// single-position GDN path intentionally takes a different branch.
        /// with `max_tokens: 1` runs the `decode_until_stop_or_budget`
        /// `for step in 0..1` loop exactly once, and that single step's own
        /// closure calls `BackendRuntime::evaluate` exactly once regardless
        /// of how many rows `next_ids` carries -- `plan_hits`/`plan_misses`
        /// (`BackendRuntime`'s own doc: every `evaluate` call is exactly one
        /// hit or one miss, never both, never neither) sum to the number of
        /// `evaluate` calls this call made. If prefill were one evaluation
        /// PER PROMPT TOKEN, this sum would equal the prompt's own token
        /// count; measured here at a prompt tokenizing to well past 200
        /// rows, it is `1` -- the whole prompt already lands in ONE
        /// `[seq_len, embedding]` program evaluation, not one per token.
        #[test]
        #[ignore = "depends on a host-local qwen3 gguf checkout outside this repo, and a real Metal device"]
        fn prefill_evaluations_per_prompt_token() {
            let model_path = crate::test_support::qwen3_gguf_path();
            crate::test_support::require_fixture(&model_path, Some("PROXIMA_QWEN3_GGUF"));
            let mapped = MappedGguf::open(std::path::Path::new(&model_path))
                .expect("mmap host-local qwen3 gguf fixture");
            let model = open_model(&mapped);
            let serving_config = greedy_serving_config();
            let prompt = alloc::format!("{PREFIX}{SUFFIX_A} {SUFFIX_B}");

            let mut runtime = BackendRuntime::new(&serving_config);
            let (_generated_ids, _text, _stopped_by_eos, prefix_state) = model
                .run_decode_loop_observed_seeded(
                    &prompt,
                    1,
                    &serving_config,
                    &mut runtime,
                    None,
                    &mut LogitsSink::Discard,
                    &mut NodeValuesSink::Discard,
                    &mut |_event| ControlFlow::Continue(()),
                    None,
                    true,
                )
                .expect("prefill a real multi-hundred-token prompt");

            let prompt_token_count = prefix_state.len();
            let program_evaluations = runtime.plan_hits + runtime.plan_misses;
            std::println!(
                "prefill_evaluations_per_prompt_token prompt_token_count={prompt_token_count} \
                 program_evaluations={program_evaluations}"
            );
            assert!(
                prompt_token_count > 200,
                "fixture prompt must tokenize past 200 rows to distinguish \"one \
                 evaluation total\" from \"one evaluation per token\" (got \
                 {prompt_token_count})"
            );
            assert_eq!(
                program_evaluations, 1,
                "prefill already runs the whole prompt through ONE program \
                 evaluation (new_count == prompt_token_count on step 0, \
                 generate.rs's own `run_decode_loop_observed_seeded` closure) -- \
                 a count of {prompt_token_count} here would mean prefill was \
                 actually one evaluation per prompt token"
            );
        }

        /// [`install_stdout_telemetry`]'s handle. Mirrors `bind.rs`'s own
        /// `TelemetryProbe` (this crate's test modules each carry their own
        /// copy of this fixture rather than sharing one, the same
        /// convention `MappedGguf` already follows across this file and
        /// `bind.rs`): a background pump drains the log ring every 5ms so a
        /// real decode step's several-hundred per-op events never overflow
        /// it before this test's own final drain runs, while
        /// `drained_total` survives whichever side (pump or caller) drains
        /// any given batch.
        #[cfg(all(feature = "metal", feature = "instrument"))]
        struct TelemetryProbe {
            recorder: std::sync::Arc<Recorder>,
            drained_total: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        }

        #[cfg(all(feature = "metal", feature = "instrument"))]
        impl TelemetryProbe {
            fn drain_and_total(&self) -> usize {
                let final_pass = self.recorder.drain();
                self.drained_total
                    .fetch_add(final_pass, std::sync::atomic::Ordering::Relaxed)
                    + final_pass
            }
        }

        /// `generate.rs`'s `instrument`-gated `op_profile*` events are
        /// `info!` calls -- no-ops with no recorder installed. Installs a
        /// console recorder at `debug` so a `--nocapture` run of
        /// [`prefill_step_zero_op_profile`] shows every `op_profile_top`/
        /// `op_profile_bucket` line this test's own deliverable is.
        #[cfg(all(feature = "metal", feature = "instrument"))]
        fn install_stdout_telemetry() -> TelemetryProbe {
            proxima_telemetry::emit::global::install(proxima_telemetry::emit::EnvFilter::parse(
                "debug",
            ));
            let recorder = Recorder::builder()
                .export(Exporter::std())
                .expect("console exporter installs for an instrument-gated test")
                .install()
                .expect("stdout telemetry recorder installs for an instrument-gated test");
            let drained_total = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let pump_recorder = std::sync::Arc::clone(&recorder);
            let pump_total = std::sync::Arc::clone(&drained_total);
            std::thread::Builder::new()
                .name("prefill-op-profile-telemetry-drain".to_string())
                .spawn(move || {
                    loop {
                        let drained = pump_recorder.drain();
                        pump_total.fetch_add(drained, std::sync::atomic::Ordering::Relaxed);
                        std::thread::sleep(std::time::Duration::from_millis(5));
                    }
                })
                .expect("spawn the test-only telemetry drain thread");
            TelemetryProbe {
                recorder,
                drained_total,
            }
        }

        /// ROW 411's own open residual: the single `runtime.evaluate` call a
        /// prefill step runs (`program_evaluations=1`, proved by
        /// [`prefill_evaluations_per_prompt_token`] above) costs
        /// 59ms/token wall-clock on an 850-token prompt -- this test walks
        /// INSIDE that one call. Setting `PROXIMA_METAL_OP_PROFILE_STEP=0`
        /// swaps prefill's own step (`_step == 0`, `generate.rs:3585-3599`)
        /// from the production batched `runtime.evaluate` to
        /// `evaluate_op_timed`, which commits ONE command buffer PER
        /// `BoundOp` and reports each op's own GPU-only time
        /// (`report_op_timings`, this file's top-of-file doc). Not a
        /// pass/fail-on-numbers test -- the printed `op_profile_top`/
        /// `op_profile_bucket`/`op_profile_codec`/`op_profile_variant`
        /// lines ARE the deliverable, exactly like `bind.rs`'s
        /// `profiles_one_real_decode_step_by_per_op_gpu_time`.
        #[cfg(all(feature = "metal", feature = "instrument"))]
        #[test]
        #[ignore = "depends on a host-local qwen3 gguf checkout outside this repo, and a real Metal device"]
        fn prefill_step_zero_op_profile() {
            let model_path = crate::test_support::qwen3_gguf_path();
            crate::test_support::require_fixture(&model_path, Some("PROXIMA_QWEN3_GGUF"));
            let mapped = MappedGguf::open(std::path::Path::new(&model_path))
                .expect("mmap host-local qwen3 gguf fixture");
            let model = open_model(&mapped);
            let serving_config = greedy_serving_config();

            // five repeats of PREFIX (183 tokens each per
            // `prefill_evaluations_per_prompt_token`) lands well past the
            // ~850-token prompt the brief measured 50.3s TTFT on.
            let prompt = PREFIX.repeat(5);

            let telemetry_recorder = install_stdout_telemetry();

            // SAFETY: this test only runs via an explicit `--ignored`
            // invocation under nextest's one-process-per-test model, the
            // same convention `bind.rs`'s own
            // `profiles_one_real_decode_step_by_per_op_gpu_time` already
            // relies on for this exact env var.
            unsafe {
                std::env::set_var("PROXIMA_METAL_OP_PROFILE_STEP", "0");
            }
            let mut runtime = BackendRuntime::new(&serving_config);
            let (_generated_ids, _text, _stopped_by_eos, prefix_state) = model
                .run_decode_loop_observed_seeded(
                    &prompt,
                    1,
                    &serving_config,
                    &mut runtime,
                    None,
                    &mut LogitsSink::Discard,
                    &mut NodeValuesSink::Discard,
                    &mut |_event| ControlFlow::Continue(()),
                    None,
                    true,
                )
                .expect("prefill an ~850-token prompt with step 0's op-profile branch armed");
            // SAFETY: same justification as the `set_var` above.
            unsafe {
                std::env::remove_var("PROXIMA_METAL_OP_PROFILE_STEP");
            }

            let prompt_token_count = prefix_state.len();
            let flushed = telemetry_recorder.drain_and_total();
            std::println!(
                "prefill_step_zero_op_profile prompt_token_count={prompt_token_count} \
                 telemetry_records_flushed={flushed}"
            );
            assert!(
                prompt_token_count > 700,
                "fixture prompt must tokenize past 700 rows to land in the same \
                 ~850-token regime the brief measured 50.3s TTFT on (got \
                 {prompt_token_count})"
            );
            assert!(
                flushed > 0,
                "the op-profile step emitted no op_profile telemetry -- \
                 PROXIMA_METAL_OP_PROFILE_STEP=0 must have matched prefill's own step 0"
            );
        }

        /// Times prefill-to-first-token for the same ~850-950-token prompt
        /// [`prefill_step_zero_op_profile`] above uses (`PREFIX.repeat(5)`),
        /// 3 runs against a fresh [`BackendRuntime`] each time so no run
        /// benefits from another run's warm dispatch-plan cache.
        /// `max_tokens: 1` isolates prefill's own step-0 evaluation
        /// ([`prefill_evaluations_per_prompt_token`] above: ONE
        /// `runtime.evaluate` call regardless of prompt length) from any
        /// decode-step cost, so the timed interval IS time-to-first-token,
        /// not time-to-eighth-token. A second, untimed `max_tokens: 8` pass
        /// prints the full greedy id sequence so two builds of this SAME
        /// test -- one per feature set (`metal,instrument` vs
        /// `metal,instrument,metal-tiled-gemm`) -- can be diffed
        /// byte-for-byte: per ROW 105/107/109/113's own "does not earn the
        /// production default until ... the full stack wins with it on"
        /// framing, a tiled-gemm speedup that changes the greedy decode is
        /// not a win, it is a correctness regression.
        #[test]
        #[ignore = "depends on a host-local qwen3 gguf checkout outside this repo, and a real Metal device"]
        fn prefill_ttft_850() {
            let model_path = crate::test_support::qwen3_gguf_path();
            crate::test_support::require_fixture(&model_path, Some("PROXIMA_QWEN3_GGUF"));
            let mapped = MappedGguf::open(std::path::Path::new(&model_path))
                .expect("mmap host-local qwen3 gguf fixture");
            let model = open_model(&mapped);
            let serving_config = greedy_serving_config();
            let prompt = PREFIX.repeat(5);

            let mut ttft_ms: Vec<f64> = Vec::new();
            let mut prompt_token_count = 0usize;
            for _run in 0..3 {
                let mut runtime = BackendRuntime::new(&serving_config);
                let start = std::time::Instant::now();
                let (_generated_ids, _text, _stopped_by_eos, prefix_state) = model
                    .run_decode_loop_observed_seeded(
                        &prompt,
                        1,
                        &serving_config,
                        &mut runtime,
                        None,
                        &mut LogitsSink::Discard,
                        &mut NodeValuesSink::Discard,
                        &mut |_event| ControlFlow::Continue(()),
                        None,
                        true,
                    )
                    .expect("prefill an ~850-token prompt for one timed TTFT run");
                let elapsed = start.elapsed();
                prompt_token_count = prefix_state.len();
                ttft_ms.push(elapsed.as_secs_f64() * 1000.0);
            }
            ttft_ms.sort_by(|left, right| left.partial_cmp(right).expect("ttft_ms never NaN"));
            let min_ms = ttft_ms[0];
            let med_ms = ttft_ms[1];
            let max_ms = ttft_ms[2];
            let tokens_per_sec_at_median = prompt_token_count as f64 / (med_ms / 1000.0);
            std::println!(
                "prefill_ttft_850 prompt_token_count={prompt_token_count} \
                 ttft_ms_min={min_ms:.1} ttft_ms_med={med_ms:.1} ttft_ms_max={max_ms:.1} \
                 tokens_per_sec_at_median={tokens_per_sec_at_median:.1}"
            );
            assert!(
                prompt_token_count > 700,
                "fixture prompt must tokenize past 700 rows to land in the same \
                 ~850-token regime this brief measured TTFT on (got \
                 {prompt_token_count})"
            );

            let mut identity_runtime = BackendRuntime::new(&serving_config);
            let (generated_ids, _text, _stopped_by_eos, _final_prefix) = model
                .run_decode_loop_observed_seeded(
                    &prompt,
                    8,
                    &serving_config,
                    &mut identity_runtime,
                    None,
                    &mut LogitsSink::Discard,
                    &mut NodeValuesSink::Discard,
                    &mut |_event| ControlFlow::Continue(()),
                    None,
                    true,
                )
                .expect("greedy-decode 8 tokens for cross-feature-set identity comparison");
            std::println!("prefill_ttft_850 greedy_eight_token_ids={generated_ids:?}");
        }
    }

    /// Real-checkpoint oracle for the one-evaluation prefill
    /// (`Self::run_decode_loop_observed_seeded`'s own `one_evaluation_prefill`
    /// doc): a `new_count > 1` qwen35moe prefill built once through
    /// `qwen35moe_forward_program_at_width`'s `Extent::Static` branch must
    /// agree with the OLD `next_ids.len()`-way split, still reachable via
    /// `PROXIMA_PREFILL_SEQUENTIAL=1`, at both the decoded text AND the
    /// last prompt position's own logits.
    mod qwen35moe_one_evaluation_prefill_real_model {
        use core::ffi::c_void;
        use std::os::fd::AsFd;

        use proxima_tensor::op::NodeId;

        use super::super::{
            BackendRuntime, ControlFlow, LoadedModel, LogitsSink, NodeValuesSink,
            supported_serving_config,
        };
        use crate::serving::GPU_LAYERS_ALL;

        struct MappedGguf {
            base: *mut u8,
            len: usize,
            _file: std::fs::File,
        }

        impl MappedGguf {
            fn open(path: &std::path::Path) -> std::io::Result<Self> {
                let file = std::fs::File::open(path)?;
                let len = usize::try_from(file.metadata()?.len())
                    .expect("fixture file length fits in usize");
                // SAFETY: `len` matches the just-opened file's own length;
                // `file` is kept alive in `_file` for as long as `base` is
                // used, and the mapping is read-only/private so no writer
                // can observe or race it.
                let base = unsafe {
                    rustix::mm::mmap(
                        core::ptr::null_mut(),
                        len,
                        rustix::mm::ProtFlags::READ,
                        rustix::mm::MapFlags::PRIVATE,
                        file.as_fd(),
                        0,
                    )
                }
                .expect("mmap host-local qwen35moe gguf fixture")
                .cast::<u8>();
                Ok(Self {
                    base,
                    len,
                    _file: file,
                })
            }

            fn as_slice(&self) -> &[u8] {
                // SAFETY: `base` points at `len` bytes mapped for `self`'s
                // whole lifetime; this borrows `self` immutably, so nothing
                // can unmap the region while the returned slice is alive.
                unsafe { core::slice::from_raw_parts(self.base, self.len) }
            }
        }

        impl Drop for MappedGguf {
            fn drop(&mut self) {
                // SAFETY: `base`/`len` are exactly what `open`'s `mmap`
                // call returned; nothing else unmaps this region.
                let _ = unsafe { rustix::mm::munmap(self.base.cast::<c_void>(), self.len) };
            }
        }

        fn open_model(mapped: &MappedGguf) -> LoadedModel<'_> {
            let file_bytes = mapped.as_slice();
            let parsed = proxima_gguf::pipe::parse_complete(file_bytes)
                .expect("parse host-local qwen35moe gguf fixture");
            LoadedModel::load(&parsed, file_bytes)
                .expect("load real qwen35moe checkpoint through the public path")
        }

        /// Greedy (`temperature: 0.0`) so the one-evaluation and sequential
        /// prefill paths are directly comparable token-for-token -- any
        /// sampling randomness would make a mismatch ambiguous between "the
        /// recurrence is wrong" and "the rng streams diverged".
        fn greedy_serving_config() -> super::super::ServingConfig<'static> {
            let mut config =
                supported_serving_config(GPU_LAYERS_ALL, crate::test_support::math_mode_from_env());
            config.temperature = 0.0;
            config
        }

        /// # Safety
        ///
        /// Single-threaded within this one `#[ignore]`d test's own process
        /// (`cargo nextest` isolates every test in its own process by
        /// default) -- no other thread reads or writes
        /// `PROXIMA_PREFILL_ONE_EVALUATION` while this holds it set.
        unsafe fn with_one_evaluation_prefill_forced<T>(body: impl FnOnce() -> T) -> T {
            // SAFETY: see this function's own doc.
            unsafe {
                std::env::set_var("PROXIMA_PREFILL_ONE_EVALUATION", "1");
            }
            let result = body();
            // SAFETY: see this function's own doc.
            unsafe {
                std::env::remove_var("PROXIMA_PREFILL_ONE_EVALUATION");
            }
            result
        }

        #[test]
        #[ignore = "requires a real, local qwen3.6:35b-a3b GGUF blob; set PROXIMA_QWEN35MOE_GGUF"]
        fn one_evaluation_prefill_matches_sequential_prefill_on_the_real_checkpoint() {
            let model_path = crate::test_support::qwen35moe_gguf_path();
            crate::test_support::require_fixture(&model_path, Some("PROXIMA_QWEN35MOE_GGUF"));
            let mapped = MappedGguf::open(std::path::Path::new(&model_path))
                .expect("mmap host-local qwen35moe gguf fixture");
            let model = open_model(&mapped);
            let serving_config = greedy_serving_config();
            let prompt = "The capital of France is";
            let steps = 16;

            let mut sequential_runtime = BackendRuntime::new(&serving_config);
            let mut sequential_logits: Vec<Vec<f32>> = Vec::new();
            let (sequential_ids, sequential_text, _, _) = model
                .run_decode_loop_observed_seeded(
                    prompt,
                    steps,
                    &serving_config,
                    &mut sequential_runtime,
                    None,
                    &mut LogitsSink::Collect(&mut sequential_logits),
                    &mut NodeValuesSink::Discard,
                    &mut |_event| ControlFlow::Continue(()),
                    None,
                    false,
                )
                .expect("sequential prefill greedy generate");

            // SAFETY: no other thread touches `PROXIMA_PREFILL_ONE_EVALUATION`
            // during this call (this function's own doc).
            let (one_evaluation_ids, one_evaluation_text, one_evaluation_logits) = unsafe {
                with_one_evaluation_prefill_forced(|| {
                    let mut one_evaluation_runtime = BackendRuntime::new(&serving_config);
                    let mut one_evaluation_logits: Vec<Vec<f32>> = Vec::new();
                    let (ids, text, _, _) = model
                        .run_decode_loop_observed_seeded(
                            prompt,
                            steps,
                            &serving_config,
                            &mut one_evaluation_runtime,
                            None,
                            &mut LogitsSink::Collect(&mut one_evaluation_logits),
                            &mut NodeValuesSink::Discard,
                            &mut |_event| ControlFlow::Continue(()),
                            None,
                            false,
                        )
                        .expect("one-evaluation prefill greedy generate");
                    (ids, text, one_evaluation_logits)
                })
            };

            std::println!(
                "one_evaluation_prefill: ids={one_evaluation_ids:?} text={one_evaluation_text:?}"
            );
            std::println!("sequential_prefill: ids={sequential_ids:?} text={sequential_text:?}");

            let last_position_max_abs_diff = one_evaluation_logits
                .first()
                .zip(sequential_logits.first())
                .map(|(one_evaluation, sequential)| {
                    one_evaluation
                        .iter()
                        .zip(sequential.iter())
                        .map(|(left, right)| (left - right).abs())
                        .fold(0.0_f32, f32::max)
                })
                .expect("both paths capture the prompt's own first-step logits");
            std::println!(
                "one_evaluation_prefill vs sequential_prefill last_position_max_abs_diff={last_position_max_abs_diff}"
            );

            assert_eq!(
                one_evaluation_ids, sequential_ids,
                "one-evaluation and sequential prefill must decode the same ids"
            );
            assert_eq!(
                one_evaluation_text, sequential_text,
                "one-evaluation and sequential prefill must decode the same text"
            );
            assert!(
                last_position_max_abs_diff <= 1e-3,
                "one-evaluation vs sequential prefill last-position logits must agree within \
                 1e-3, got {last_position_max_abs_diff}"
            );
        }

        /// The oracle above drives `run_decode_loop_observed_seeded` off a
        /// hand-built `BackendRuntime`, never `generate_streaming` --
        /// `gguf_generate` and every other real caller's own entry point
        /// (`decode.rs`'s own `apply_memory_fit_gate` wiring only fires
        /// there, metal+macos gated). ROW 590's own fupan named this gap:
        /// the config-mirror test proves the FLAG plumbs through, not that
        /// the entry point real callers use still decodes through it.
        #[test]
        #[ignore = "requires a real, local qwen3.6:35b-a3b GGUF blob; set PROXIMA_QWEN35MOE_GGUF"]
        fn one_evaluation_prefill_through_generate_streaming_matches_sequential_on_the_real_checkpoint()
         {
            let model_path = crate::test_support::qwen35moe_gguf_path();
            crate::test_support::require_fixture(&model_path, Some("PROXIMA_QWEN35MOE_GGUF"));
            let mapped = MappedGguf::open(std::path::Path::new(&model_path))
                .expect("mmap host-local qwen35moe gguf fixture");
            let model = open_model(&mapped);
            let serving_config = greedy_serving_config();
            let prompt = "The capital of France is";
            let steps = 16;

            let (sequential_ids, sequential_text, _) = model
                .generate_streaming(prompt, steps, serving_config, &mut |_event| {
                    ControlFlow::Continue(())
                })
                .expect("sequential prefill through generate_streaming");

            // SAFETY: no other thread touches `PROXIMA_PREFILL_ONE_EVALUATION`
            // during this call (this function's own doc).
            let (one_evaluation_ids, one_evaluation_text, _) = unsafe {
                with_one_evaluation_prefill_forced(|| {
                    model
                        .generate_streaming(prompt, steps, serving_config, &mut |_event| {
                            ControlFlow::Continue(())
                        })
                        .expect("one-evaluation prefill through generate_streaming")
                })
            };

            std::println!(
                "generate_streaming one_evaluation: ids={one_evaluation_ids:?} text={one_evaluation_text:?}"
            );
            std::println!(
                "generate_streaming sequential: ids={sequential_ids:?} text={sequential_text:?}"
            );

            assert!(
                !one_evaluation_ids.is_empty(),
                "generate_streaming must not return an empty id sequence for one-evaluation prefill"
            );
            assert_eq!(
                one_evaluation_ids, sequential_ids,
                "generate_streaming must decode the same ids through the one-evaluation and \
                 sequential prefill paths"
            );
            assert_eq!(
                one_evaluation_text, sequential_text,
                "generate_streaming must decode the same text through the one-evaluation and \
                 sequential prefill paths"
            );
        }

        /// Diagnostic companion to the oracle above: when it fails, this
        /// names the first layer whose own `block_output` (post-residual,
        /// after FFN -- [`crate::qwen35moe::Qwen35MoeLayerDiagnostics::block_output`])
        /// disagrees between the one-evaluation and sequential prefill
        /// paths, at the prompt's own last position, relative to that
        /// row's own norm.
        #[test]
        #[ignore = "requires a real, local qwen3.6:35b-a3b GGUF blob; set PROXIMA_QWEN35MOE_GGUF"]
        fn one_evaluation_prefill_first_diverging_layer_on_the_real_checkpoint() {
            let model_path = crate::test_support::qwen35moe_gguf_path();
            crate::test_support::require_fixture(&model_path, Some("PROXIMA_QWEN35MOE_GGUF"));
            let mapped = MappedGguf::open(std::path::Path::new(&model_path))
                .expect("mmap host-local qwen35moe gguf fixture");
            let model = open_model(&mapped);
            let serving_config = greedy_serving_config();
            let prompt = "The capital of France is";

            let hparams = model
                .qwen35moe_hparams
                .as_ref()
                .expect("this checkpoint routes through the qwen35moe registry entry");
            let prompt_ids = proxima_tokenizer::encode_with_bos_eos(
                prompt,
                &model.vocab,
                super::super::wants_bos(&model.vocab),
                model.vocab.add_eos_token().unwrap_or(false),
            )
            .expect("tokenize the oracle prompt");
            let prompt_len = prompt_ids.len();
            let embedding = model.architecture.embedding as usize;

            let (_program, _roots, static_layer_roots, _moe_sites, static_diagnostics) =
                crate::qwen35moe::qwen35moe_forward_program_at_width(
                    hparams,
                    Some(prompt_len as u32),
                )
                .expect("static-width program builds");
            let _ = static_layer_roots;
            let one_evaluation_nodes: Vec<NodeId> = static_diagnostics
                .iter()
                .map(|diagnostic| diagnostic.block_output)
                .collect();
            let sequential_nodes: Vec<NodeId> = model
                .qwen35moe_layer_diagnostics
                .iter()
                .map(|diagnostic| diagnostic.block_output)
                .collect();

            let mut sequential_runtime = BackendRuntime::new(&serving_config);
            let mut sequential_steps: Vec<Vec<Vec<f32>>> = Vec::new();
            let mut sequential_sink = NodeValuesSink::Collect {
                nodes: &sequential_nodes,
                steps: &mut sequential_steps,
            };
            let _ = model
                .run_decode_loop_observed_seeded(
                    prompt,
                    1,
                    &serving_config,
                    &mut sequential_runtime,
                    None,
                    &mut LogitsSink::Discard,
                    &mut sequential_sink,
                    &mut |_event| ControlFlow::Continue(()),
                    None,
                    false,
                )
                .expect("sequential prefill diagnostic run");

            // SAFETY: no other thread touches `PROXIMA_PREFILL_ONE_EVALUATION`
            // during this call (`with_one_evaluation_prefill_forced`'s own
            // doc).
            let one_evaluation_steps: Vec<Vec<Vec<f32>>> = unsafe {
                with_one_evaluation_prefill_forced(|| {
                    let mut one_evaluation_runtime = BackendRuntime::new(&serving_config);
                    let mut one_evaluation_steps: Vec<Vec<Vec<f32>>> = Vec::new();
                    let mut one_evaluation_sink = NodeValuesSink::Collect {
                        nodes: &one_evaluation_nodes,
                        steps: &mut one_evaluation_steps,
                    };
                    let _ = model
                        .run_decode_loop_observed_seeded(
                            prompt,
                            1,
                            &serving_config,
                            &mut one_evaluation_runtime,
                            None,
                            &mut LogitsSink::Discard,
                            &mut one_evaluation_sink,
                            &mut |_event| ControlFlow::Continue(()),
                            None,
                            false,
                        )
                        .expect("one-evaluation prefill diagnostic run");
                    one_evaluation_steps
                })
            };

            let one_evaluation_last_position = one_evaluation_steps
                .first()
                .expect("one-evaluation prefill observes exactly one batch");
            let sequential_last_position = sequential_steps
                .last()
                .expect("sequential prefill observes at least one batch");

            let mut first_diverging_layer: Option<(usize, f32)> = None;
            for (layer, (one_evaluation_output, sequential_output)) in one_evaluation_last_position
                .iter()
                .zip(sequential_last_position.iter())
                .enumerate()
            {
                let last_row = &one_evaluation_output[one_evaluation_output.len() - embedding..];
                let row_norm = sequential_output
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    .sqrt()
                    .max(1e-6);
                let max_abs_diff = last_row
                    .iter()
                    .zip(sequential_output.iter())
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0_f32, f32::max);
                let relative_diff = max_abs_diff / row_norm;
                std::println!(
                    "layer={layer} block_output max_abs_diff={max_abs_diff} row_norm={row_norm} \
                     relative_diff={relative_diff}"
                );
                if relative_diff > 1e-3 && first_diverging_layer.is_none() {
                    first_diverging_layer = Some((layer, relative_diff));
                }
            }
            match first_diverging_layer {
                Some((layer, relative_diff)) => {
                    std::println!("first_diverging_layer={layer} relative_diff={relative_diff}")
                }
                None => std::println!("no layer's block_output diverged past 1e-3 relative"),
            }
        }

        /// Bisects layer 0 itself: `qkv_mixed` (shared code, computed
        /// identically on both branches of
        /// [`proxima_tensor::spec::append_qwen35_ssm_mixer_with_taps_and_layout`])
        /// vs `state_out`/`mixer_output` (the M>1 branch's own recurrence
        /// and tail) vs `post_mixer_residual`/`block_output` (the shared
        /// FFN after it).
        #[test]
        #[ignore = "requires a real, local qwen3.6:35b-a3b GGUF blob; set PROXIMA_QWEN35MOE_GGUF"]
        fn one_evaluation_prefill_layer_zero_bisection_on_the_real_checkpoint() {
            let model_path = crate::test_support::qwen35moe_gguf_path();
            crate::test_support::require_fixture(&model_path, Some("PROXIMA_QWEN35MOE_GGUF"));
            let mapped = MappedGguf::open(std::path::Path::new(&model_path))
                .expect("mmap host-local qwen35moe gguf fixture");
            let model = open_model(&mapped);
            let serving_config = greedy_serving_config();
            let prompt = "The capital of France is";

            let hparams = model
                .qwen35moe_hparams
                .as_ref()
                .expect("this checkpoint routes through the qwen35moe registry entry");
            let prompt_ids = proxima_tokenizer::encode_with_bos_eos(
                prompt,
                &model.vocab,
                super::super::wants_bos(&model.vocab),
                model.vocab.add_eos_token().unwrap_or(false),
            )
            .expect("tokenize the oracle prompt");
            let prompt_len = prompt_ids.len();

            let (_program, _roots, _static_layer_roots, _moe_sites, static_diagnostics) =
                crate::qwen35moe::qwen35moe_forward_program_at_width(
                    hparams,
                    Some(prompt_len as u32),
                )
                .expect("static-width program builds");
            let layer0_static = &static_diagnostics[0];
            let layer0_decode = &model.qwen35moe_layer_diagnostics[0];
            let static_ssm_taps = layer0_static
                .ssm_taps
                .clone()
                .expect("layer 0 is a GDN layer on this checkpoint");
            let decode_ssm_taps = layer0_decode
                .ssm_taps
                .clone()
                .expect("layer 0 is a GDN layer on this checkpoint");

            let names = ["qkv_mixed", "state_out"];
            let one_evaluation_nodes = [static_ssm_taps.qkv_mixed, static_ssm_taps.state_out];
            let sequential_nodes = [decode_ssm_taps.qkv_mixed, decode_ssm_taps.state_out];
            let _ = (layer0_static.mixer_output, layer0_decode.mixer_output);
            let _ = (
                layer0_static.post_mixer_residual,
                layer0_decode.post_mixer_residual,
            );
            let _ = (layer0_static.block_output, layer0_decode.block_output);

            let mut sequential_runtime = BackendRuntime::new(&serving_config);
            let mut sequential_steps: Vec<Vec<Vec<f32>>> = Vec::new();
            let mut sequential_sink = NodeValuesSink::Collect {
                nodes: &sequential_nodes,
                steps: &mut sequential_steps,
            };
            let _ = model
                .run_decode_loop_observed_seeded(
                    prompt,
                    1,
                    &serving_config,
                    &mut sequential_runtime,
                    None,
                    &mut LogitsSink::Discard,
                    &mut sequential_sink,
                    &mut |_event| ControlFlow::Continue(()),
                    None,
                    false,
                )
                .expect("sequential prefill layer-zero bisection run");

            // SAFETY: no other thread touches `PROXIMA_PREFILL_ONE_EVALUATION`
            // during this call (`with_one_evaluation_prefill_forced`'s own
            // doc).
            let one_evaluation_steps: Vec<Vec<Vec<f32>>> = unsafe {
                with_one_evaluation_prefill_forced(|| {
                    let mut one_evaluation_runtime = BackendRuntime::new(&serving_config);
                    let mut one_evaluation_steps: Vec<Vec<Vec<f32>>> = Vec::new();
                    let mut one_evaluation_sink = NodeValuesSink::Collect {
                        nodes: &one_evaluation_nodes,
                        steps: &mut one_evaluation_steps,
                    };
                    let _ = model
                        .run_decode_loop_observed_seeded(
                            prompt,
                            1,
                            &serving_config,
                            &mut one_evaluation_runtime,
                            None,
                            &mut LogitsSink::Discard,
                            &mut one_evaluation_sink,
                            &mut |_event| ControlFlow::Continue(()),
                            None,
                            false,
                        )
                        .expect("one-evaluation prefill layer-zero bisection run");
                    one_evaluation_steps
                })
            };

            let one_evaluation_values = one_evaluation_steps
                .first()
                .expect("one-evaluation prefill observes exactly one batch");
            let sequential_last_position = sequential_steps
                .last()
                .expect("sequential prefill observes at least one batch");

            for (name, one_evaluation_value, sequential_value) in
                itertools_zip3(&names, one_evaluation_values, sequential_last_position)
            {
                let embedding = sequential_value.len();
                let last_row = &one_evaluation_value[one_evaluation_value.len() - embedding..];
                let row_norm = sequential_value
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    .sqrt()
                    .max(1e-6);
                let max_abs_diff = last_row
                    .iter()
                    .zip(sequential_value.iter())
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0_f32, f32::max);
                std::println!(
                    "layer0 {name} max_abs_diff={max_abs_diff} row_norm={row_norm} \
                     relative_diff={}",
                    max_abs_diff / row_norm
                );
            }
        }

        fn itertools_zip3<'a, T>(
            names: &'a [&'a str],
            left: &'a [T],
            right: &'a [T],
        ) -> impl Iterator<Item = (&'a str, &'a T, &'a T)> {
            names
                .iter()
                .copied()
                .zip(left.iter())
                .zip(right.iter())
                .map(|((name, left_value), right_value)| (name, left_value, right_value))
        }

        /// Finer bisection than [`one_evaluation_prefill_layer_zero_bisection_on_the_real_checkpoint`]
        /// (which hits `MissingEvaluatedNode` requesting `state_out` alone):
        /// walks `block_input` (embedding row) -> `qkv_mixed` (shared code,
        /// pre-recurrence) -> `delta_out` (the recurrence's own per-position
        /// output) -> `mixer_output` -> `post_mixer_residual` -> `block_output`
        /// at layer 0's own last prompt position, one-evaluation vs
        /// sequential, each row's relative diff against that row's own norm.
        #[test]
        #[ignore = "requires a real, local qwen3.6:35b-a3b GGUF blob; set PROXIMA_QWEN35MOE_GGUF"]
        fn one_evaluation_prefill_layer_zero_tap_sweep_on_the_real_checkpoint() {
            let model_path = crate::test_support::qwen35moe_gguf_path();
            crate::test_support::require_fixture(&model_path, Some("PROXIMA_QWEN35MOE_GGUF"));
            let mapped = MappedGguf::open(std::path::Path::new(&model_path))
                .expect("mmap host-local qwen35moe gguf fixture");
            let model = open_model(&mapped);
            let serving_config = greedy_serving_config();
            let prompt = "The capital of France is";

            let hparams = model
                .qwen35moe_hparams
                .as_ref()
                .expect("this checkpoint routes through the qwen35moe registry entry");
            let prompt_ids = proxima_tokenizer::encode_with_bos_eos(
                prompt,
                &model.vocab,
                super::super::wants_bos(&model.vocab),
                model.vocab.add_eos_token().unwrap_or(false),
            )
            .expect("tokenize the oracle prompt");
            let prompt_len = prompt_ids.len();

            let (_program, _roots, _static_layer_roots, _moe_sites, static_diagnostics) =
                crate::qwen35moe::qwen35moe_forward_program_at_width(
                    hparams,
                    Some(prompt_len as u32),
                )
                .expect("static-width program builds");
            let layer0_static = &static_diagnostics[0];
            let layer0_decode = &model.qwen35moe_layer_diagnostics[0];
            let static_ssm_taps = layer0_static
                .ssm_taps
                .clone()
                .expect("layer 0 is a GDN layer on this checkpoint");
            let decode_ssm_taps = layer0_decode
                .ssm_taps
                .clone()
                .expect("layer 0 is a GDN layer on this checkpoint");

            let names = [
                "block_input",
                "qkv_mixed",
                "query",
                "key",
                "value",
                "beta",
                "gate",
                "z_head",
                "delta_out",
                "mixer_output",
                "post_mixer_residual",
                "block_output",
            ];
            let one_evaluation_nodes = [
                layer0_static.block_input,
                static_ssm_taps.qkv_mixed,
                static_ssm_taps.query,
                static_ssm_taps.key,
                static_ssm_taps.value,
                static_ssm_taps.beta,
                static_ssm_taps.gate,
                static_ssm_taps.z_head,
                static_ssm_taps.delta_out,
                layer0_static.mixer_output,
                layer0_static.post_mixer_residual,
                layer0_static.block_output,
                // extra, one-evaluation-only self-check (no sequential
                // counterpart requested): does the M>1 branch's own
                // per-position slice (`query`, index 2 above) actually equal
                // the LAST row of the shared multi-position tensor it was
                // sliced from (`query_sequence`)? Both come from the SAME
                // evaluation of the SAME graph, so any mismatch here is a
                // buffer-lifetime/aliasing bug in the interpreter, not a
                // wiring or algebra defect.
                static_ssm_taps.query_sequence,
                static_ssm_taps.beta_sequence,
            ];
            let sequential_nodes = [
                layer0_decode.block_input,
                decode_ssm_taps.qkv_mixed,
                decode_ssm_taps.query,
                decode_ssm_taps.key,
                decode_ssm_taps.value,
                decode_ssm_taps.beta,
                decode_ssm_taps.gate,
                decode_ssm_taps.z_head,
                decode_ssm_taps.delta_out,
                layer0_decode.mixer_output,
                layer0_decode.post_mixer_residual,
                layer0_decode.block_output,
            ];

            let mut sequential_runtime = BackendRuntime::new(&serving_config);
            let mut sequential_steps: Vec<Vec<Vec<f32>>> = Vec::new();
            let mut sequential_sink = NodeValuesSink::Collect {
                nodes: &sequential_nodes,
                steps: &mut sequential_steps,
            };
            let _ = model
                .run_decode_loop_observed_seeded(
                    prompt,
                    1,
                    &serving_config,
                    &mut sequential_runtime,
                    None,
                    &mut LogitsSink::Discard,
                    &mut sequential_sink,
                    &mut |_event| ControlFlow::Continue(()),
                    None,
                    false,
                )
                .expect("sequential prefill layer-zero tap sweep run");

            // SAFETY: no other thread touches `PROXIMA_PREFILL_ONE_EVALUATION`
            // during this call (`with_one_evaluation_prefill_forced`'s own
            // doc).
            let one_evaluation_steps: Vec<Vec<Vec<f32>>> = unsafe {
                with_one_evaluation_prefill_forced(|| {
                    let mut one_evaluation_runtime = BackendRuntime::new(&serving_config);
                    let mut one_evaluation_steps: Vec<Vec<Vec<f32>>> = Vec::new();
                    let mut one_evaluation_sink = NodeValuesSink::Collect {
                        nodes: &one_evaluation_nodes,
                        steps: &mut one_evaluation_steps,
                    };
                    let _ = model
                        .run_decode_loop_observed_seeded(
                            prompt,
                            1,
                            &serving_config,
                            &mut one_evaluation_runtime,
                            None,
                            &mut LogitsSink::Discard,
                            &mut one_evaluation_sink,
                            &mut |_event| ControlFlow::Continue(()),
                            None,
                            false,
                        )
                        .expect("one-evaluation prefill layer-zero tap sweep run");
                    one_evaluation_steps
                })
            };

            let one_evaluation_values = one_evaluation_steps
                .first()
                .expect("one-evaluation prefill observes exactly one batch");
            let sequential_last_position = sequential_steps
                .last()
                .expect("sequential prefill observes at least one batch");

            for (name, one_evaluation_value, sequential_value) in
                itertools_zip3(&names, one_evaluation_values, sequential_last_position)
            {
                let row_length = sequential_value.len();
                let last_row = &one_evaluation_value[one_evaluation_value.len() - row_length..];
                let row_norm = sequential_value
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    .sqrt()
                    .max(1e-6);
                let max_abs_diff = last_row
                    .iter()
                    .zip(sequential_value.iter())
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0_f32, f32::max);
                std::println!(
                    "layer0 {name} row_length={row_length} max_abs_diff={max_abs_diff} \
                     row_norm={row_norm} relative_diff={}",
                    max_abs_diff / row_norm
                );
            }

            // self-check: `query`/`beta` (index 2/5) MUST equal the last row
            // of `query_sequence`/`beta_sequence` (index 12/13) -- both read
            // from the SAME one-evaluation execution of the SAME graph.
            for (self_check_name, sliced_index, sequence_index) in
                [("query", 2usize, 12usize), ("beta", 5usize, 13usize)]
            {
                let sliced = &one_evaluation_values[sliced_index];
                let sequence = &one_evaluation_values[sequence_index];
                let row_length = sliced.len();
                let sequence_last_row = &sequence[sequence.len() - row_length..];
                let row_norm = sliced
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    .sqrt()
                    .max(1e-6);
                let max_abs_diff = sliced
                    .iter()
                    .zip(sequence_last_row.iter())
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0_f32, f32::max);
                std::println!(
                    "layer0 self_check {self_check_name}_vs_{self_check_name}_sequence_last_row \
                     row_length={row_length} max_abs_diff={max_abs_diff} row_norm={row_norm} \
                     relative_diff={}",
                    max_abs_diff / row_norm
                );
            }
        }

        /// Structural companion to the numeric tap sweep above: no
        /// evaluation, no weight bytes -- [`proxima_tensor::cpu::plan_trace_named`]'s
        /// own doc ("every decision here depends on `program`/`symbols`/
        /// `outputs` alone, never on tensor bytes"). Builds the SAME M=13
        /// one-evaluation program under the PRODUCTION output set (per-layer
        /// `{qkv_mixed, state_out}` for a GDN layer,
        /// `generate.rs`'s own `Qwen35LayerRoots::Ssm` root-push, plus the
        /// final `logits_root` -- never the diagnostic taps
        /// `one_evaluation_prefill_layer_zero_tap_sweep_on_the_real_checkpoint`
        /// requests) and lists every node with 2+ live consumers that a
        /// `raw_op`/"absorbed" or `resolved_node`/"retired" decision folds
        /// into (or retires against) one specific consumer -- the
        /// multi-consumer-absorption defect class this whole investigation
        /// is chasing, independent of which backend later executes the plan.
        #[test]
        #[ignore = "requires a real, local qwen3.6:35b-a3b GGUF blob; set PROXIMA_QWEN35MOE_GGUF"]
        fn plan_trace_reveals_multi_consumer_absorption_at_layer_zero_m13() {
            let model_path = crate::test_support::qwen35moe_gguf_path();
            crate::test_support::require_fixture(&model_path, Some("PROXIMA_QWEN35MOE_GGUF"));
            let mapped = MappedGguf::open(std::path::Path::new(&model_path))
                .expect("mmap host-local qwen35moe gguf fixture");
            let model = open_model(&mapped);
            let hparams = model
                .qwen35moe_hparams
                .as_ref()
                .expect("this checkpoint routes through the qwen35moe registry entry");

            let (program, roots, layer_roots, _moe_sites, _diagnostics) =
                crate::qwen35moe::qwen35moe_forward_program_at_width(hparams, Some(13))
                    .expect("static-width program builds");

            let mut production_outputs = alloc::vec![roots.logits];
            for layer_root in &layer_roots {
                match layer_root {
                    proxima_tensor::spec::Qwen35LayerRoots::Attention((even, odd, value)) => {
                        production_outputs.push(*even);
                        production_outputs.push(*odd);
                        production_outputs.push(*value);
                    }
                    proxima_tensor::spec::Qwen35LayerRoots::DenseAttention((
                        first,
                        second,
                        pass,
                        value,
                    )) => {
                        production_outputs.push(*first);
                        production_outputs.push(*second);
                        production_outputs.push(*pass);
                        production_outputs.push(*value);
                    }
                    proxima_tensor::spec::Qwen35LayerRoots::Ssm {
                        qkv_mixed,
                        state_out,
                    } => {
                        production_outputs.push(*qkv_mixed);
                        production_outputs.push(*state_out);
                    }
                }
            }

            let symbols: [u64; 2] = [13, 0];
            let decisions =
                proxima_tensor::cpu::plan_trace_named(&program, &symbols, &[], &production_outputs)
                    .expect("plan traces against the production output set");

            // `readers[source]` = every program position (== `NodeId.0`,
            // this crate numbers nodes by their own `Vec<Op>` index) that
            // reads `source` -- the same full-program scan
            // `live::annotate`/`node_retirement` themselves run, kept as the
            // raw position list (not just a count or the last one) so a
            // decision's own `into` position can be checked against the
            // TRUE last reader, not merely "does 2+ readers exist" (which is
            // routine and not itself a defect).
            let mut readers: alloc::collections::BTreeMap<proxima_tensor::op::NodeId, Vec<u32>> =
                alloc::collections::BTreeMap::new();
            for (position, operation) in program.iter().enumerate() {
                let uses = match operation {
                    proxima_tensor::Op::Elementwise { operands, .. } => {
                        operands.iter().map(|(node, _)| *node).collect::<Vec<_>>()
                    }
                    proxima_tensor::Op::Reduce(reduce) => alloc::vec![reduce.operand],
                    proxima_tensor::Op::Input { .. } | proxima_tensor::Op::Iota { .. } => {
                        Vec::new()
                    }
                    proxima_tensor::Op::Constant { .. } => Vec::new(),
                };
                for node in uses {
                    readers.entry(node).or_default().push(position as u32);
                }
            }

            let mut violations: Vec<(u32, &'static str, u32, Option<u32>, u32)> = Vec::new();
            for decision in &decisions {
                if !matches!(decision.decision, "absorbed" | "retired" | "fused") {
                    continue;
                }
                let Some(node_readers) = readers.get(&decision.node) else {
                    continue;
                };
                if node_readers.len() < 2 {
                    continue;
                }
                let true_last_reader = *node_readers.iter().max().expect("non-empty");
                let retires_at = decision.into.map_or(u32::MAX, |node| node.0);
                if retires_at < true_last_reader {
                    violations.push((
                        decision.node.0,
                        decision.decision,
                        node_readers.len() as u32,
                        decision.into.map(|node| node.0),
                        true_last_reader,
                    ));
                }
            }
            violations.sort_unstable();
            violations.dedup();

            for (node, decision, consumers, into, true_last_reader) in &violations {
                std::println!(
                    "multi_consumer_absorption_violation node={node} decision={decision} \
                     consumers={consumers} into={into:?} true_last_reader={true_last_reader}"
                );
            }
            std::println!(
                "multi_consumer_absorption_violation total_flagged={} decisions_scanned={}",
                violations.len(),
                decisions.len()
            );
        }
    }
}
