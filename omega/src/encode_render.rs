//! The backend-neutral encode render: binary32 rows in, a cache format's bytes out, decided from
//! the format's description alone. A backend supplies an [`EncodeDialect`], the spelling of the
//! pieces that differ (prelude, kernel signature and thread index, storage token, the layout store
//! expression, the block function), and this module decides which formats render and assembles
//! the kernel. It names no format and no backend.
//!
//! Teaching pointer: [`proxima_numeric::FormatCodec::encode`] is the host algorithm.

use alloc::format;
use alloc::string::String;

use proxima_numeric::block_layout::BlockLayout;
use proxima_numeric::encode_rule::{CodeRounding, EncodeRule, ScaleStatistic, StoredMin};
use proxima_numeric::float_layout::FloatLayout;
use proxima_numeric::{FormatCodec, NumericFormat};

use crate::msl::layout_render::hardware_type;

const BYTE_BITS: u16 = u8::BITS as u16;

/// The pieces of an encode kernel a backend spells.
pub trait EncodeDialect {
    /// Text that opens the source (includes, namespaces).
    fn prelude(&self) -> String;
    /// The integer type that stores one scalar code of `bytes` bytes, `None` when the dialect has none.
    fn storage_token(&self, bytes: usize) -> Option<&'static str>;
    /// The function `name` that maps a binary32 value to the layout's code.
    fn scalar_store(&self, layout: FloatLayout, name: &str) -> String;
    /// Why the dialect cannot hold `layout` at all, `None` when it can.
    fn scalar_refusal(&self, layout: &FloatLayout) -> Option<&'static str>;
    /// The kernel signature through its opening brace and the declaration of `gid`, the thread's
    /// index; the input is `input` of `input_token` elements, the output `output` of `storage` elements.
    fn kernel_open(&self, name: &str, input: &str, storage: &str) -> String;
    /// The function `name` that encodes one block, `None` when the layout has no rule or the dialect no block rendering.
    fn block_function(&self, layout: &BlockLayout, name: &str) -> Option<String>;
    /// Why the dialect renders no block format, `None` when it renders them.
    fn block_refusal(&self) -> Option<&'static str>;
}

/// A rendered encode: the source of one kernel and the shape it dispatches over.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedEncode {
    /// Source defining the kernel named `entry`: input 0 is the binary32 input, output 1 the
    /// packed output; one thread encodes `elements_per_thread` values into `bytes_per_thread` bytes.
    pub source: String,
    /// The kernel function's name.
    pub entry: String,
    /// Elements one thread consumes: the block's element count, or one for a scalar.
    pub elements_per_thread: usize,
    /// Bytes one thread produces.
    pub bytes_per_thread: usize,
}

/// Why `format` has no rendered encode under `dialect`: the property it lacks.
///
/// # Errors
/// The property of the format or dialect that makes it unrenderable.
#[must_use = "the refusal names why a format has no rendered encode"]
pub fn encode_kernel_refusal(dialect: &dyn EncodeDialect, format: &NumericFormat) -> Result<(), &'static str> {
    let Some(codec) = FormatCodec::for_format(format) else {
        return Err("no host codec resolves the format");
    };
    if !codec.can_encode() {
        return Err("the host codec has no encode to match");
    }
    match format {
        NumericFormat::Scalar(layout) => scalar_refusal(dialect, layout),
        NumericFormat::Block(layout) => match layout.encode_rule() {
            Some(rule) => rule_refusal(layout, &rule).and_then(|()| dialect.block_refusal().map_or(Ok(()), Err)),
            None => Err("the layout has no encode rule"),
        },
        NumericFormat::Levels(_) => Err("a level format has no encode rule"),
    }
}

fn scalar_refusal(dialect: &dyn EncodeDialect, layout: &FloatLayout) -> Result<(), &'static str> {
    let width = u16::from(layout.width());
    if width % BYTE_BITS != 0 || dialect.storage_token(usize::from(width / BYTE_BITS)).is_none() {
        return Err("the scalar code does not fill a whole number of bytes the dialect stores");
    }
    dialect.scalar_refusal(layout).map_or(Ok(()), Err)
}

/// Why a layout that has an [`EncodeRule`] still has no rendering: the fit the rule names that this
/// emitter does not write (a searched or stored range, sub-block scales, ties to even), or the
/// layout packs independent scale runs.
fn rule_refusal(layout: &BlockLayout, rule: &EncodeRule) -> Result<(), &'static str> {
    if layout.scale_runs.is_some() && !rule.is_float_code() {
        return Err("the layout packs independent scale runs under an integer code");
    }
    if matches!(rule.statistic, ScaleStatistic::MinMax(_)) || rule.stored_min != StoredMin::Absent {
        return Err("the layout fits a stored range, which the emitter does not render");
    }
    if rule.sub.is_some() {
        return Err("the layout carries sub-block scales, which the emitter does not render");
    }
    if matches!(rule.rounding, CodeRounding::NearestEven) {
        return Err("the layout rounds ties to even through a magic-number add, which the emitter does not render");
    }
    Ok(())
}

/// Renders the encode kernel of `format` under `dialect`, entry point `{name}_kernel`.
///
/// # Errors
/// [`encode_kernel_refusal`]'s reason.
#[must_use = "the refusal names why a format has no rendered encode"]
pub fn format_encode_kernel(dialect: &dyn EncodeDialect, format: &NumericFormat, name: &str) -> Result<RenderedEncode, &'static str> {
    encode_kernel_refusal(dialect, format)?;
    match format {
        NumericFormat::Scalar(layout) => scalar_encode_kernel(dialect, *layout, name).ok_or("the dialect has no storage for the scalar code"),
        NumericFormat::Block(layout) => {
            let function = dialect.block_function(layout, name).ok_or("the layout has no encode rule")?;
            wrap_block_kernel(dialect, layout, name, function).ok_or("the dialect has no storage for a block byte")
        }
        NumericFormat::Levels(_) => Err("a level format has no encode rule"),
    }
}

fn scalar_encode_kernel(dialect: &dyn EncodeDialect, layout: FloatLayout, name: &str) -> Option<RenderedEncode> {
    let bytes = usize::from(layout.width()) / usize::from(BYTE_BITS);
    let storage = dialect.storage_token(bytes)?;
    let prelude = dialect.prelude();
    let store = dialect.scalar_store(layout, &format!("{name}_store"));
    let open = dialect.kernel_open(name, row_token()?, storage);
    let source = format!("{prelude}{store}\n{open}    output[gid] = ({storage}){name}_store(input[gid]);\n}}\n");
    Some(RenderedEncode { source, entry: format!("{name}_kernel"), elements_per_thread: 1, bytes_per_thread: bytes })
}

/// The layout of one host row element: the one scalar format the hardware stores in a whole word.
fn row_layout() -> Option<FloatLayout> {
    NumericFormat::every().find_map(|format| match format {
        NumericFormat::Scalar(layout) if usize::from(layout.width()) == u32::BITS as usize => Some(layout),
        _ => None,
    })
}

/// The token of one host row element.
fn row_token() -> Option<&'static str> {
    hardware_type(&row_layout()?)
}

/// Bytes of one host row element, what a source buffer's element must measure.
#[must_use]
pub fn row_bytes() -> Option<usize> {
    row_layout().map(|layout| usize::from(layout.width()) / usize::from(BYTE_BITS))
}

/// The kernel that runs `function` (named `name`) over one block per thread.
#[must_use]
pub fn wrap_block_kernel(dialect: &dyn EncodeDialect, layout: &BlockLayout, name: &str, function: String) -> Option<RenderedEncode> {
    let elements = usize::from(layout.elements);
    let bytes = usize::from(layout.block_bytes);
    let prelude = dialect.prelude();
    let storage = dialect.storage_token(core::mem::size_of::<u8>())?;
    let open = dialect.kernel_open(name, row_token()?, storage);
    let source = format!("{prelude}{function}\n{open}    {name}(input + gid * {elements}u, output + gid * {bytes}u);\n}}\n");
    Some(RenderedEncode { source, entry: format!("{name}_kernel"), elements_per_thread: elements, bytes_per_thread: bytes })
}

/// Which rows a dispatch of a rendered encode covers: the element count of the source rows and the
/// byte offset in the destination buffer where the packed bytes start. Every backend sizes and
/// splits the dispatch through this one type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodeRows {
    pub element_count: usize,
    pub byte_offset: usize,
}

impl EncodeRows {
    /// Threads to dispatch: one per `elements_per_thread` values.
    #[must_use]
    pub const fn threads(&self, elements_per_thread: usize) -> usize {
        self.element_count / elements_per_thread
    }

    /// Bytes the destination must hold past its offset, `None` when the rows are not whole blocks of `format`.
    #[must_use]
    pub fn byte_len(&self, format: &NumericFormat) -> Option<usize> {
        format.packed_len(self.element_count)
    }

    /// The byte range the packed rows occupy in the destination, `None` on overflow or partial blocks.
    #[must_use]
    pub fn byte_range(&self, format: &NumericFormat) -> Option<core::ops::Range<usize>> {
        let end = self.byte_offset.checked_add(self.byte_len(format)?)?;
        Some(self.byte_offset..end)
    }
}
