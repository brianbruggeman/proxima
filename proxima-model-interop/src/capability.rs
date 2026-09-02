//! The one table `tests/capability_matrix.rs` and
//! `examples/generate_compatibility_doc.rs` both read -- declared once here
//! so the test that proves each cell's status and the doc that reports it
//! can never independently drift. See `docs/compatibility.md`'s own header
//! for how the two are kept in lockstep.
//!
//! Two tables, two different code enums as their source of truth:
//! [`GGML_CAPABILITY_TABLE`] mirrors the codec/topology/backend cells
//! `tests/capability_matrix.rs` actually drives through
//! `crate::LoadedModel`'s (`std`-gated) public `Pipe`; the quantized-packed-format table
//! (built in `examples/generate_compatibility_doc.rs`, `metal`-feature-gated
//! because it reads `omega::msl::PackedCodec`) mirrors every variant of that
//! enum against the CPU kernel (`proxima_tensor::cpu::QuantizedBlock`) and
//! the GPU emitters (`omega::msl`/`omega::wgsl`/`omega::cuda`) that dispatch
//! on it.

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

const UNREPRESENTABLE: &str = "no encoder or decoder in proxima_gguf::quant (only q4_k/q5_k/q6_k/q8_0 exist); \
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
        status: CellStatus::Unimplemented(UNREPRESENTABLE),
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

/// Every [`omega::msl::PackedCodec`] variant against the CPU kernel
/// (`proxima_tensor::cpu::QuantizedBlock`, the same enum
/// [`crate::bind::gguf_tensor_as_packed_block`] returns) and the three GPU
/// source emitters that dispatch on the same codec
/// (`omega::msl::emit`/`omega::wgsl::emit_wgsl`/`omega::cuda::emit`). Built
/// from an exhaustive `match` on `PackedCodec` with no `_` arm below --
/// `omega` gaining an eighth packed format fails this crate's own build
/// before it can silently fail to appear here.
#[cfg(feature = "metal")]
pub mod quant_format {
    use alloc::string::String;
    use core::fmt::Write as _;

    use omega::msl::PackedCodec;

    /// `(cpu kernel, metal emitter, wgsl emitter, cuda emitter)` -- every
    /// packed format is `Supported` on all four today (verified by reading
    /// the exhaustive `match` arms in `proxima-tensor/src/cpu.rs`,
    /// `omega/src/msl.rs`, `omega/src/wgsl.rs`, `omega/src/cuda.rs`; none
    /// carries a `todo!`/`unimplemented!` on any codec arm).
    const fn codec_name(codec: PackedCodec) -> &'static str {
        match codec {
            PackedCodec::Q4K => "Q4_K",
            PackedCodec::Q5K => "Q5_K",
            PackedCodec::Q6K => "Q6_K",
            PackedCodec::Q8_0 => "Q8_0",
            PackedCodec::Q4_0 => "Q4_0",
            PackedCodec::Float16 => "F16",
            PackedCodec::BFloat16 => "BF16",
        }
    }

    /// The 7 [`PackedCodec`] variants, exhaustively -- adding an 8th to
    /// `omega::msl::PackedCodec` without adding it here is a compile error,
    /// not a silently stale doc.
    const ALL_CODECS: &[PackedCodec] = &[
        PackedCodec::Q4K,
        PackedCodec::Q5K,
        PackedCodec::Q6K,
        PackedCodec::Q8_0,
        PackedCodec::Q4_0,
        PackedCodec::Float16,
        PackedCodec::BFloat16,
    ];

    #[must_use]
    pub fn render_markdown() -> String {
        let mut out = String::new();
        out.push_str(
            "| packed codec | cpu kernel | metal emitter | wgsl emitter | cuda emitter |\n",
        );
        out.push_str("| --- | --- | --- | --- | --- |\n");
        for &codec in ALL_CODECS {
            let _ = writeln!(
                out,
                "| {} | supported | supported | supported | supported |",
                codec_name(codec)
            );
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
    fn supported_dense_cpu_codecs_matches_the_five_hand_written_test_functions() {
        let supported = supported_dense_cpu_codecs();
        assert_eq!(
            supported.len(),
            5,
            "capability_matrix.rs currently drives exactly 5 supported dense-cpu codecs"
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
    use super::quant_format::render_markdown;

    #[test]
    fn renders_exactly_seven_packed_codec_rows_plus_the_header() {
        let rendered = render_markdown();
        assert_eq!(
            rendered.lines().count(),
            9,
            "7 PackedCodec variants, 1 header row, 1 separator row"
        );
    }
}
