//! The route-compaction buffer an expert-grouped contraction's prepass writes and its grouped kernel reads.
//!
//! Words, in order: a header (word 0 is the total routed rows the prepass wrote, then one
//! three-word indirect-dispatch argument per row-tile slot a backend launches indirectly), the
//! per-expert row counts, the per-expert exclusive offsets into the list, the expert-major list of
//! flat token ids, the per-expert first-tile table, the tile-to-expert table, and the count pass's
//! partial counts. Metal and CUDA both size and address it through these items.

pub(crate) const COUNT_HEADER_WORDS: u64 = 1;
#[cfg(feature = "metal-grouped-gemm")]
const INDIRECT_ARGUMENT_WORDS: u64 = 3;

#[cfg(feature = "metal-grouped-gemm")]
pub(crate) const fn header_words(indirect_slots: u64) -> u64 {
    COUNT_HEADER_WORDS + INDIRECT_ARGUMENT_WORDS * indirect_slots
}

/// Tiles a launch must cover whatever the realized route is: every full tile of the whole route, plus one partial tile per expert.
pub(crate) const fn tile_upper_bound(tokens: u64, experts: u64, tile_tokens: u64) -> u64 {
    tokens.div_ceil(tile_tokens) + experts
}

pub(crate) const fn tile_table_words(tokens: u64, experts: u64, tile_tokens: u64) -> u64 {
    experts + tile_upper_bound(tokens, experts, tile_tokens)
}

pub(crate) const fn compaction_words(
    header: u64,
    tokens: u64,
    experts: u64,
    tile_tokens: u64,
    prepass_blocks: u64,
) -> u64 {
    header + 2 * experts + tokens + tile_table_words(tokens, experts, tile_tokens) + prepass_blocks * experts
}

/// The fault word a grouped kernel raises when the prepass's total disagrees with its own token extent.
pub(crate) const ROUTE_COMPACTION_MISMATCH_FAULT: u32 = u32::MAX;

/// C-family definitions of the three bases a kernel addresses the buffer by, for a device that
/// reads `experts` and `tokens` at run time: the list follows the header and the two per-expert
/// tables, the first-tile table follows the list, the tile-to-expert table follows that.
#[cfg(feature = "cuda")]
pub(crate) fn device_bases(header: u64, experts: &str, tokens: &str) -> alloc::string::String {
    alloc::format!(
        "    const long long list_base = {header}ll + 2ll * {experts};\n    const long long tile_start_base = list_base + {tokens};\n    const long long tile_expert_base = tile_start_base + {experts};\n"
    )
}
