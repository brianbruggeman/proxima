//! The bind half of the codec card's acceptance (PROXIMAOMEGA-20): the real
//! 23 GiB checkpoint, with its 69 `iq4_xs` and 2 `iq3_s` tensors, must bind every
//! leaf the flint program reads and claim every tensor the directory carries.
//!
//! Its own module, not a third test in `flint_greedy_parity_tests`: the parity
//! filter is `test(flint_greedy_parity_)`, which matches the module path, so a
//! sibling test there would silently change AC13's expected count from two to
//! three -- measured, after it did exactly that.
//!
//! Requires the real checkpoint: `PROXIMA_FLINT_GGUF`, defaulting to the path
//! the capture used.

#![allow(clippy::expect_used)]

use std::fs::File;

use memmap2::Mmap;
use proxima_gguf::parse_complete;

use crate::LoadedModel;

fn flint_path() -> String {
    std::env::var("PROXIMA_FLINT_GGUF").unwrap_or_else(|_| {
        String::from("/Users/brianbruggeman/models/llm/flint-gguf/flint-UD-Q4_K_XL.gguf")
    })
}

fn load_flint() -> LoadedModel<'static> {
    let path = flint_path();
    crate::test_support::require_fixture(&path, Some("PROXIMA_FLINT_GGUF"));
    let file = File::open(&path).expect("open the real flint checkpoint");
    // SAFETY: read-only mapping of a file nothing else writes during the test.
    let mapping = unsafe { Mmap::map(&file) }.expect("mmap the real flint checkpoint");
    // the model borrows the mapping for the whole test; leaking it keeps the
    // borrow alive without a self-referential struct.
    let bytes: &'static [u8] = Box::leak(Box::new(mapping));
    let parsed = parse_complete(bytes).expect("parse the real flint header");
    LoadedModel::load(&parsed, bytes).expect("bind the real flint checkpoint")
}

/// The bind half of the codec card's acceptance: the real 23 GiB checkpoint,
/// with its 69 `iq4_xs` and 2 `iq3_s` tensors, binds every leaf the flint
/// program reads and claims every tensor the directory carries -- the census
/// slice 9 built now runs against a file rather than a captured directory.
///
/// Reaching this assertion at all is the point: before the two codecs existed,
/// `bind_program_leaves` stopped at
/// `UnrepresentableGgmlType { tensor: "blk.2.ffn_gate_exps.weight", ggml_type: Iq3S }`.
#[test]
fn real_flint_checkpoint_binds_every_tensor() {
    let model = load_flint();
    // `load` is the assertion: `refuse_incomplete_bind` returns
    // `UnboundWeightLeaf`/`UnclaimedTensor` before a `LoadedModel` exists, so
    // reaching this line means every one of the 779 directory rows is claimed
    // by a leaf of the lowered program and every weight leaf the program reads
    // is in the directory. The two checks below LABEL the run -- they are read
    // straight off the header, so they cannot observe what lowered -- and exist
    // only so a different checkpoint behind the same env var is not silently
    // accepted as this one.
    assert_eq!(
        model.checkpoint_bytes() as u64,
        25_054_688_704,
        "the capture's own byte count labels this run"
    );
    assert_eq!(
        model.layer_count(),
        49,
        "the capture declares 49 blocks (a header read, not a lowering count)"
    );
}
