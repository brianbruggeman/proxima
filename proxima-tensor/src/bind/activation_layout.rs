//! Which block layouts can hold a quantized activation: the layout-only half of the activation
//! quantize decision, shared by every backend that derives or renders a `BlockQuantize` node.

use proxima_numeric::block_layout::BlockLayout;
use proxima_numeric::float_layout::FloatLayout;

/// Elements one activation scale covers: the 32-element run the int8 readers fold per step.
pub const ACTIVATION_GROUP_ELEMENTS: u64 = 32;

const QUANTIZE_DIVISOR: i32 = 127;

/// A piece a backend's render lacks for a block activation layout, with the line that found it
/// missing, so a refusal points at the code that would have to change and never at a format name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Missing {
    pub piece: &'static str,
    pub file: &'static str,
    pub line: u32,
}

fn codes_are_byte_runs(layout: &BlockLayout) -> bool {
    let element = &layout.element;
    let run = usize::from(layout.run_elements());
    element.code == FloatLayout::INT8
        && element.high.is_none()
        && element.low.bits == 8
        && (0..usize::from(layout.elements))
            .all(|index| element.low.address(index) == layout.run_code_byte((index / run) as u16) * 8 + index % run * 8)
}

/// Int8 codes in 32-element byte runs under an f16 scale, each run's scale followed by the two-byte
/// stored sum the int8 readers fold in.
#[must_use]
pub fn is_int8_run_layout(layout: &BlockLayout) -> bool {
    u64::from(layout.run_elements()) == ACTIVATION_GROUP_ELEMENTS
        && layout.scale.layout == FloatLayout::F16
        && layout.min.is_none()
        && layout.sub.is_none()
        && layout.multiplier == 1
        && layout.element.bias == 0
        && codes_are_byte_runs(layout)
        && (0..layout.runs()).all(|run| layout.run_code_byte(run) >= layout.run_scale_byte(run) + 4)
        && (0..layout.runs()).all(|run| layout.run_scale_byte(run) + 4 <= usize::from(layout.block_bytes))
}

/// Whether the layout's scale rule is the peak/127 rule with a stored sum that the quantize
/// producers write. A layout with independent scale runs (the grouped pair) has no single rule, so
/// its run geometry, checked by [`is_int8_run_layout`], is all there is to serve.
#[must_use]
pub fn quantize_rule_is_served(layout: &BlockLayout) -> bool {
    layout.encode_rule().is_none_or(|rule| rule.divisor == QUANTIZE_DIVISOR && rule.stored_sum)
}
