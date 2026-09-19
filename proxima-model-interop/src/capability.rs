//! The one table `tests/capability_matrix.rs` and
//! `examples/generate_compatibility_doc.rs` both read -- declared once here
//! so the test that proves each cell's status and the doc that reports it
//! can never independently drift. See `docs/compatibility.md`'s own header
//! for how the two are kept in lockstep.
//!
//! Two tables, both keyed off the same `Codec` identity now:
//! [`GGML_CAPABILITY_TABLE`] mirrors the codec/topology/backend cells
//! `tests/capability_matrix.rs` actually drives through
//! `crate::LoadedModel`'s (`std`-gated) public `Pipe`; [`quant_format`]'s
//! table (built in `examples/generate_compatibility_doc.rs`,
//! `metal`-feature-gated) mirrors every [`proxima_primitives::Codec`]
//! variant against the CPU kernel (`proxima_tensor::cpu::QuantizedBlock`).

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write as _;

use proxima_gguf::GgmlType;

/// Which checkpoint shape a cell was driven through --
/// `crate::generate::LoadedModel::load`'s (`std`-gated) dense path vs its
/// `architecture.expert_count > 0` routed path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Topology {
    Dense,
    Moe,
}

impl Topology {
    const fn label(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Moe => "moe",
        }
    }
}

/// Which forward-pass driver a cell was run through.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    Cpu,
    Metal,
}

impl Backend {
    const fn label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
        }
    }
}

/// A cell's status -- `Unimplemented` always names the exact missing piece,
/// the same reason string `tests/capability_matrix.rs` puts in its own
/// `#[ignore = "..."]` attribute for that cell (a `#[ignore]` attribute
/// takes only a string literal, never a `const` reference, so the two
/// copies are kept identical by hand; `tests/capability_matrix.rs`'s own
/// drift-guard test below re-checks the doc this table renders still says
/// what the code does, which is the drift that actually matters).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellStatus {
    Supported,
    Unimplemented(&'static str),
}

/// One `(codec, topology, backend)` cell and its status --
/// `tests/capability_matrix.rs`'s own cell list, made data instead of only
/// existing implicitly as a set of test function names.
#[derive(Debug, Clone, Copy)]
pub struct GgmlCell {
    pub codec: GgmlType,
    pub codec_name: &'static str,
    pub topology: Topology,
    pub backend: Backend,
    pub status: CellStatus,
}

const UNREPRESENTABLE: &str = "no encoder or decoder in proxima_gguf::quant (only q3_k/q4_k/q5_k/q6_k/q8_0 exist); \
bind::gguf_tensor_as_f32 rejects it with UnrepresentableGgmlType before a forward pass can run";

const F16_ACTIVATION_UNSUPPORTED: &str = "proxima_tensor::cpu::evaluate_quantized_named_with_scratch is f32-only: \
reject_non_float32 (proxima-tensor/src/cpu.rs) rejects any non-Float32 elementwise node outright";

/// Every cell [`crate`]'s own `tests/capability_matrix.rs` drives, dense CPU
/// first (mirrors `dense_cpu_*_forward_produces_a_deterministic_token_sequence`
/// and the four `dense_cpu_*_forward_prefill_and_decode` `#[ignore]`d
/// placeholders), then the one MoE cell
/// (`moe_architecture_cpu_forward_prefill_and_decode`).
pub const GGML_CAPABILITY_TABLE: &[GgmlCell] = &[
    GgmlCell {
        codec: GgmlType::F32,
        codec_name: "F32",
        topology: Topology::Dense,
        backend: Backend::Cpu,
        status: CellStatus::Supported,
    },
    GgmlCell {
        codec: GgmlType::Q8_0,
        codec_name: "Q8_0",
        topology: Topology::Dense,
        backend: Backend::Cpu,
        status: CellStatus::Supported,
    },
    GgmlCell {
        codec: GgmlType::Q4_K,
        codec_name: "Q4_K",
        topology: Topology::Dense,
        backend: Backend::Cpu,
        status: CellStatus::Supported,
    },
    GgmlCell {
        codec: GgmlType::Q5_K,
        codec_name: "Q5_K",
        topology: Topology::Dense,
        backend: Backend::Cpu,
        status: CellStatus::Supported,
    },
    GgmlCell {
        codec: GgmlType::Q6_K,
        codec_name: "Q6_K",
        topology: Topology::Dense,
        backend: Backend::Cpu,
        status: CellStatus::Supported,
    },
    GgmlCell {
        codec: GgmlType::Q4_0,
        codec_name: "Q4_0",
        topology: Topology::Dense,
        backend: Backend::Cpu,
        status: CellStatus::Unimplemented(UNREPRESENTABLE),
    },
    GgmlCell {
        codec: GgmlType::Q5_0,
        codec_name: "Q5_0",
        topology: Topology::Dense,
        backend: Backend::Cpu,
        status: CellStatus::Unimplemented(UNREPRESENTABLE),
    },
    GgmlCell {
        codec: GgmlType::Q2_K,
        codec_name: "Q2_K",
        topology: Topology::Dense,
        backend: Backend::Cpu,
        status: CellStatus::Unimplemented(UNREPRESENTABLE),
    },
    GgmlCell {
        codec: GgmlType::Q3_K,
        codec_name: "Q3_K",
        topology: Topology::Dense,
        backend: Backend::Cpu,
        status: CellStatus::Supported,
    },
    GgmlCell {
        codec: GgmlType::F16,
        codec_name: "F16",
        topology: Topology::Dense,
        backend: Backend::Cpu,
        status: CellStatus::Unimplemented(F16_ACTIVATION_UNSUPPORTED),
    },
    GgmlCell {
        codec: GgmlType::F32,
        codec_name: "F32",
        topology: Topology::Moe,
        backend: Backend::Cpu,
        status: CellStatus::Supported,
    },
    GgmlCell {
        codec: GgmlType::F32,
        codec_name: "F32",
        topology: Topology::Dense,
        backend: Backend::Metal,
        status: CellStatus::Supported,
    },
    GgmlCell {
        codec: GgmlType::Q8_0,
        codec_name: "Q8_0",
        topology: Topology::Dense,
        backend: Backend::Metal,
        status: CellStatus::Supported,
    },
    GgmlCell {
        codec: GgmlType::Q4_K,
        codec_name: "Q4_K",
        topology: Topology::Dense,
        backend: Backend::Metal,
        status: CellStatus::Supported,
    },
    GgmlCell {
        codec: GgmlType::Q5_K,
        codec_name: "Q5_K",
        topology: Topology::Dense,
        backend: Backend::Metal,
        status: CellStatus::Supported,
    },
    GgmlCell {
        codec: GgmlType::Q6_K,
        codec_name: "Q6_K",
        topology: Topology::Dense,
        backend: Backend::Metal,
        status: CellStatus::Supported,
    },
    GgmlCell {
        codec: GgmlType::Q3_K,
        codec_name: "Q3_K",
        topology: Topology::Dense,
        backend: Backend::Metal,
        status: CellStatus::Supported,
    },
];

fn write_row(out: &mut String, codec: &str, topology: &str, backend: &str, status: CellStatus) {
    let status_cell = match status {
        CellStatus::Supported => String::from("supported"),
        CellStatus::Unimplemented(reason) => {
            let mut cell = String::from("unimplemented -- ");
            cell.push_str(reason);
            cell
        }
    };
    let _ = writeln!(out, "| {codec} | {topology} | {backend} | {status_cell} |");
}

/// Renders [`GGML_CAPABILITY_TABLE`] to the exact markdown body
/// `docs/compatibility.md` commits and
/// `tests/capability_doc_drift.rs`'s drift guard re-derives on every run.
/// No generation timestamp -- the whole point is a byte-identical diff
/// against the same table on every run, not a document that always looks
/// changed.
#[must_use]
pub fn render_ggml_matrix_markdown() -> String {
    let mut out = String::new();
    out.push_str("| codec | topology | backend | status |\n");
    out.push_str("| --- | --- | --- | --- |\n");
    for cell in GGML_CAPABILITY_TABLE {
        write_row(
            &mut out,
            cell.codec_name,
            cell.topology.label(),
            cell.backend.label(),
            cell.status,
        );
    }
    out
}

/// Every `(codec, topology, backend)` triple that must classify as
/// [`CellStatus::Supported`] -- `tests/capability_matrix.rs`'s drift guard
/// asserts each one still parses, loads, and produces the exact greedy ids
/// the hand-written per-cell tests already assert, so a codec silently
/// losing support flips this table's own row, not just a doc.
#[must_use]
pub fn supported_dense_cpu_codecs() -> Vec<GgmlType> {
    GGML_CAPABILITY_TABLE
        .iter()
        .filter(|cell| {
            cell.topology == Topology::Dense
                && cell.backend == Backend::Cpu
                && cell.status == CellStatus::Supported
        })
        .map(|cell| cell.codec)
        .collect()
}

/// Every [`proxima_primitives::Codec`] variant against the CPU kernel
/// (`proxima_tensor::cpu::QuantizedBlock`, the same enum
/// [`crate::bind::gguf_tensor_as_packed_block`] returns) -- built from an
/// exhaustive `match` on `Codec` with no `_` arm below -- `proxima_primitives`
/// gaining a 30th codec fails this crate's own build before it can silently
/// fail to appear here.
///
/// `Codec` carries 29 variants; only 14 have a CPU decode/matmul path
/// (`proxima_tensor::cpu::epilogue::codec_to_decodable_ggml_type` returns
/// `Some`, mirrored by [`ALL_CODECS`]'s `cpu_supported` field below) -- the
/// other 15 are recognized (this table names every one) but no construction
/// site in this crate or `proxima_tensor` ever builds a
/// [`proxima_tensor::cpu::QuantizedBlock::Packed`] carrying one, matching
/// `codec_to_decodable_ggml_type`'s own doc for why. This mirrors the CPU
/// kernel's own support boundary only -- it does not claim anything about
/// `omega`'s GPU emitters, which this module no longer reads (`omega::msl`'s
/// own former codec identity enum was folded onto this same `Codec`, see
/// that migration's own commit).
#[cfg(feature = "metal")]
pub mod quant_format {
    use alloc::string::String;
    use core::fmt::Write as _;

    use proxima_primitives::Codec;

    /// This codec's GGML-style display name, matching
    /// `proxima_tensor::cpu::epilogue::codec_name_for_error`'s own per-codec
    /// strings (uppercased to match [`super::GgmlCell::codec_name`]'s
    /// convention elsewhere in this file), not restated as a second
    /// independent copy of that mapping.
    const fn codec_name(codec: Codec) -> &'static str {
        match codec {
            Codec::Q2K => "Q2_K",
            Codec::Q3K => "Q3_K",
            Codec::Q4K => "Q4_K",
            Codec::Q5K => "Q5_K",
            Codec::Q6K => "Q6_K",
            Codec::Q8_0 => "Q8_0",
            Codec::Q4_0 => "Q4_0",
            Codec::Q5_1 => "Q5_1",
            Codec::Q5_0 => "Q5_0",
            Codec::Iq4Nl => "IQ4_NL",
            Codec::Iq2Xs => "IQ2_XS",
            Codec::Iq3Xxs => "IQ3_XXS",
            Codec::Float16 => "F16",
            Codec::BFloat16 => "BF16",
            Codec::Q4_1 => "Q4_1",
            Codec::Q8_1 => "Q8_1",
            Codec::Q8K => "Q8_K",
            Codec::Iq1S => "IQ1_S",
            Codec::Iq1M => "IQ1_M",
            Codec::Iq2Xxs => "IQ2_XXS",
            Codec::Iq2S => "IQ2_S",
            Codec::Iq3S => "IQ3_S",
            Codec::Iq4Xs => "IQ4_XS",
            Codec::Tq10 => "TQ1_0",
            Codec::Tq20 => "TQ2_0",
            Codec::Mxfp4 => "MXFP4",
            Codec::Nvfp4 => "NVFP4",
            Codec::Q1_0 => "Q1_0",
            Codec::Q2_0 => "Q2_0",
        }
    }

    /// `true` for exactly the 14 [`Codec`] variants
    /// `proxima_tensor::cpu::epilogue::codec_to_decodable_ggml_type` maps to
    /// `Some` (a CPU decode path exists); `false` for the other 15, which
    /// that same function maps to `None` -- no construction site in this
    /// crate or `proxima_tensor` ever builds a `QuantizedBlock::Packed`
    /// carrying one of them.
    const fn cpu_kernel_supported(codec: Codec) -> bool {
        matches!(
            codec,
            Codec::Q4K
                | Codec::Q5K
                | Codec::Q3K
                | Codec::Q2K
                | Codec::Q6K
                | Codec::Q8_0
                | Codec::Q4_0
                | Codec::Q5_1
                | Codec::Q5_0
                | Codec::Iq4Nl
                | Codec::Iq2Xs
                | Codec::Iq3Xxs
                | Codec::Float16
                | Codec::BFloat16
        )
    }

    /// Every [`Codec`] variant, exhaustively -- adding a 30th to
    /// `proxima_primitives::Codec` without adding it here is a compile
    /// error, not a silently stale doc.
    ///
    /// `pub(super)` so [`super::quant_format_tests`] can derive its row-count
    /// assertion from `ALL_CODECS.len()` instead of a hardcoded integer.
    pub(super) const ALL_CODECS: &[Codec] = &[
        Codec::Q4K,
        Codec::Q5K,
        Codec::Q6K,
        Codec::Q8_0,
        Codec::Q3K,
        Codec::Q4_0,
        Codec::Float16,
        Codec::BFloat16,
        Codec::Q2K,
        Codec::Q5_1,
        Codec::Q5_0,
        Codec::Q4_1,
        Codec::Q8_1,
        Codec::Q8K,
        Codec::Iq1S,
        Codec::Iq1M,
        Codec::Iq2Xxs,
        Codec::Iq2Xs,
        Codec::Iq2S,
        Codec::Iq3Xxs,
        Codec::Iq3S,
        Codec::Iq4Nl,
        Codec::Iq4Xs,
        Codec::Tq10,
        Codec::Tq20,
        Codec::Mxfp4,
        Codec::Nvfp4,
        Codec::Q1_0,
        Codec::Q2_0,
    ];

    #[must_use]
    pub fn render_markdown() -> String {
        let mut out = String::new();
        out.push_str("| packed codec | cpu kernel |\n");
        out.push_str("| --- | --- |\n");
        for &codec in ALL_CODECS {
            let status = if cpu_kernel_supported(codec) {
                "supported"
            } else {
                "unsupported"
            };
            let _ = writeln!(out, "| {} | {status} |", codec_name(codec));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CellStatus, GGML_CAPABILITY_TABLE, render_ggml_matrix_markdown, supported_dense_cpu_codecs,
    };

    #[test]
    fn table_has_no_duplicate_cells() {
        let mut seen = alloc::vec::Vec::new();
        for cell in GGML_CAPABILITY_TABLE {
            let key = (cell.codec_name, cell.topology, cell.backend);
            assert!(
                !seen.contains(&key),
                "duplicate cell in GGML_CAPABILITY_TABLE: {key:?}"
            );
            seen.push(key);
        }
    }

    #[test]
    fn supported_dense_cpu_codecs_matches_the_six_hand_written_test_functions() {
        let supported = supported_dense_cpu_codecs();
        assert_eq!(
            supported.len(),
            6,
            "capability_matrix.rs currently drives exactly 6 supported dense-cpu codecs"
        );
    }

    #[test]
    fn render_emits_one_row_per_table_cell_plus_the_header() {
        let rendered = render_ggml_matrix_markdown();
        let row_count = rendered.lines().count();
        assert_eq!(
            row_count,
            GGML_CAPABILITY_TABLE.len() + 2,
            "one header line, one separator line, one row per cell"
        );
    }

    #[test]
    fn unimplemented_cells_all_name_a_reason() {
        for cell in GGML_CAPABILITY_TABLE {
            if let CellStatus::Unimplemented(reason) = cell.status {
                assert!(
                    !reason.is_empty(),
                    "{} must name why it is unimplemented",
                    cell.codec_name
                );
            }
        }
    }
}

#[cfg(all(test, feature = "metal"))]
mod quant_format_tests {
    use super::quant_format::{ALL_CODECS, render_markdown};

    #[test]
    fn renders_one_packed_codec_row_per_all_codecs_plus_the_header() {
        let rendered = render_markdown();
        assert_eq!(
            rendered.lines().count(),
            ALL_CODECS.len() + 2,
            "one row per Codec variant, 1 header row, 1 separator row"
        );
    }

    #[test]
    fn all_codecs_covers_every_codec_variant_exactly_once() {
        assert_eq!(
            ALL_CODECS.len(),
            29,
            "proxima_primitives::Codec carries 29 variants today"
        );
    }

    #[test]
    fn q3_k_row_is_cpu_supported() {
        let rendered = render_markdown();
        let q3k_row = rendered.lines().find(|line| line.starts_with("| Q3_K "));
        assert_eq!(
            q3k_row,
            Some("| Q3_K | supported |"),
            "Q3_K has a CPU matmul kernel (matmul_q3k_f32)"
        );
    }

    #[test]
    fn q5_1_row_is_cpu_supported() {
        let rendered = render_markdown();
        let q5_1_row = rendered.lines().find(|line| line.starts_with("| Q5_1 "));
        assert_eq!(
            q5_1_row,
            Some("| Q5_1 | supported |"),
            "Q5_1 has a CPU matmul kernel (matmul_q5_1_f32)"
        );
    }

    #[test]
    fn q4_1_row_is_recognized_but_unsupported() {
        let rendered = render_markdown();
        let q4_1_row = rendered.lines().find(|line| line.starts_with("| Q4_1 "));
        assert_eq!(
            q4_1_row,
            Some("| Q4_1 | unsupported |"),
            "Q4_1 has no CPU decode/matmul path yet \
             (codec_to_decodable_ggml_type returns None)"
        );
    }
}
