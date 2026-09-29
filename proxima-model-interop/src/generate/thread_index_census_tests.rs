// fixture setup; expect() carries the failure message
#![allow(clippy::expect_used)]

use std::collections::BTreeMap;
use std::fs::File;

use memmap2::Mmap;
use omega::msl::Grid2DForm;
use proxima_gguf::parse_complete;
use proxima_primitives::Codec;
use proxima_tensor::cpu::QuantizedBlock;
use proxima_tensor::spec::Qwen35LayerRoots;
use proxima_tensor::{NodeId, NumericPolicy, Op, bind_with_fusion, infer};

use crate::architecture::Architecture;
use crate::bind::BoundWeights;
use crate::GEMMA4;

/// One dispatch count per `(new rows, cached rows)` shape: how many of the
/// real program's dispatches launch past a 32-bit thread index, and of those
/// how many the emitter routes to the flat 2D form.
#[derive(Debug)]
struct Census {
    dispatches: usize,
    past_u32: usize,
    flat: usize,
    tiles: usize,
    widest_launch: u128,
    widest_node: u32,
    past_u32_kinds: BTreeMap<(&'static str, Vec<u64>), usize>,
}

fn packed_codecs(program: &[Op], weights: &BoundWeights<'_>) -> BTreeMap<NodeId, Codec> {
    let mut by_name: BTreeMap<&str, Codec> = BTreeMap::new();
    for (name, block) in &weights.packed {
        if let QuantizedBlock::Packed { codec, .. } = block {
            by_name.insert(name.as_str(), *codec);
        }
    }
    for (name, _, codec) in &weights.packed_owned {
        by_name.insert(name.as_str(), *codec);
    }
    program
        .iter()
        .enumerate()
        .filter_map(|(index, op)| {
            let codec = by_name.get(op.name()?)?;
            Some((NodeId(index as u32), *codec))
        })
        .collect()
}

fn census(
    program: &[Op],
    roots: &[NodeId],
    codecs: &BTreeMap<NodeId, Codec>,
    rows: u64,
    cached: u64,
    fuse_cached_attention: bool,
) -> Census {
    let policy = NumericPolicy::llama_relaxed();
    let shapes = infer(program, &[rows, cached]).expect("the real program infers at this shape");
    let resolved = bind_with_fusion(program, &shapes, roots, fuse_cached_attention, policy)
        .expect("the real program binds at this shape");
    let mut tally = Census {
        dispatches: 0,
        past_u32: 0,
        flat: 0,
        tiles: 0,
        widest_launch: 0,
        widest_node: 0,
        past_u32_kinds: BTreeMap::new(),
    };
    for bound in &resolved {
        let kernel = omega::emit(bound, codecs, policy).unwrap_or_else(|error| {
            panic!(
                "rows={rows} cached={cached}: node {} ({}) failed to emit: {error}",
                bound.node.0,
                bound.kind.name()
            )
        });
        let launch = u128::from(kernel.grid.threads) * u128::from(kernel.grid.depth);
        let past_u32 = launch > u128::from(u32::MAX);
        let form = kernel.grid.grid2d.map(|spec| spec.form);
        assert!(
            !past_u32 || form.is_some(),
            "rows={rows} cached={cached}: node {} ({}) launches {launch} threads through a 1D \
             32-bit thread index",
            bound.node.0,
            bound.kind.name()
        );
        assert!(
            past_u32 || form != Some(Grid2DForm::FlatThreadgroupIndex),
            "rows={rows} cached={cached}: node {} ({}) takes the flat form for a {launch}-thread launch that fits",
            bound.node.0,
            bound.kind.name()
        );
        tally.dispatches += 1;
        tally.past_u32 += usize::from(past_u32);
        if past_u32 {
            *tally
                .past_u32_kinds
                .entry((bound.kind.name(), bound.extents.clone()))
                .or_default() += 1;
        }
        tally.flat += usize::from(form == Some(Grid2DForm::FlatThreadgroupIndex));
        tally.tiles += usize::from(form == Some(Grid2DForm::TileCoordinates));
        if launch > tally.widest_launch {
            tally.widest_launch = launch;
            tally.widest_node = bound.node.0;
        }
    }
    tally
}

/// The census the overflow fix owes: not the one dispatch a bug report named
/// (`per_layer_model_proj`, node 17) but every dispatch of the real
/// gemma4-E2B program -- FFN gate/up reduces, score reduces, the per-layer
/// projection, anything else -- at prompt lengths from below the first
/// overflow (~1,365 rows) to 16,118, each either a 1D launch that fits 32
/// bits or a 2D/3D launch. Binds the real checkpoint's program and codecs and
/// calls `omega::emit`; no device is opened.
#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo"]
fn every_dispatch_of_the_real_gemma4_e2b_prefill_fits_32_bits_or_takes_a_2d_form() {
    let model_path = crate::test_support::gemma4_e2b_gguf_path();
    crate::test_support::require_fixture(&model_path, Some("PROXIMA_GEMMA4_E2B_GGUF"));
    let file = File::open(&model_path).expect("open the real gemma4-E2B checkpoint");
    // SAFETY: read-only mapping of a file nothing else writes during the test.
    let mapping = unsafe { Mmap::map(&file) }.expect("mmap the real gemma4-E2B checkpoint");
    let bytes: &[u8] = &mapping;
    let parsed = parse_complete(bytes).expect("parse the real gemma4-E2B header");
    let bound = GEMMA4.bind(&parsed, bytes).expect("bind the real gemma4-E2B program");
    let codecs = packed_codecs(&bound.program, &bound.weights);
    assert!(!codecs.is_empty(), "no packed weight was attributed a codec: the census would emit dense kernels");
    let kv_inputs = bound.program.iter().filter(|op| op.name().is_some_and(|name| name.starts_with("kv_cache"))).count();
    println!("program ops={} kv_cache inputs={kv_inputs} layer_roots={}", bound.program.len(), bound.layer_roots.len());
    let mut roots = vec![bound.logits_root];
    for layer in &bound.layer_roots {
        if let Qwen35LayerRoots::Attention((even, odd, value)) = layer {
            roots.extend([*even, *odd, *value]);
        }
    }

    for fuse_cached_attention in [true, false] {
        let control = census(&bound.program, &roots, &codecs, 512, 0, fuse_cached_attention);
        println!("census fuse={fuse_cached_attention} rows=512 cached=0 (control): {control:?}");
        assert_eq!(control.past_u32, 0, "a 512-row prefill must not need the flat form: {control:?}");
        assert_eq!(control.flat, 0);

        let prefill_rows: [u64; 4] = [1_400, 2_048, 7_895, 16_118];
        for cached_extent_is_rows in [false, true] {
            let mut previous_past_u32 = 0;
            for rows in prefill_rows {
                let cached = if cached_extent_is_rows { rows } else { 0 };
                let tally = census(&bound.program, &roots, &codecs, rows, cached, fuse_cached_attention);
                println!("census fuse={fuse_cached_attention} rows={rows} cached={cached}: {tally:?}");
                assert!(
                    tally.past_u32 > 0,
                    "rows={rows} cached={cached}: the census found nothing past 2^32, so it proves nothing: {tally:?}"
                );
                assert!(tally.flat + tally.tiles >= tally.past_u32, "{tally:?}");
                assert!(
                    tally.past_u32 >= previous_past_u32,
                    "past-2^32 dispatches must not shrink with rows: {tally:?}"
                );
                previous_past_u32 = tally.past_u32;
            }
        }

        for (rows, cached) in [(512u64, 15_360u64), (512, 16_384)] {
            let tally = census(&bound.program, &roots, &codecs, rows, cached, fuse_cached_attention);
            println!("census fuse={fuse_cached_attention} rows={rows} cached={cached} (chunked prefill): {tally:?}");
            assert!(tally.flat + tally.tiles >= tally.past_u32, "{tally:?}");
        }
    }
}
