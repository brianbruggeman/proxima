//! The one hardware table: the float layouts a device scalar type equals field by field.
//!
//! Teaching pointer: the rows are looked up from `proxima_numeric::float_layout::NAMED_LAYOUTS` by
//! property, never spelled here. A dialect names a row ([`crate::render::Dialect::scalar`]); this
//! table only says which layouts have one.

use proxima_numeric::float_layout::{FloatLayout, NAMED_LAYOUTS, SignEncoding, Specials};

use super::dialect::Dialect;

const fn is_hardware(layout: &FloatLayout) -> bool {
    layout.is_float()
        && layout.accumulates_in_place()
        && layout.bias_override.is_none()
        && matches!(layout.specials, Specials::InfinityOnOverflow)
        && matches!(layout.sign, SignEncoding::Magnitude)
}

const fn hardware_count() -> usize {
    let mut count = 0;
    let mut index = 0;
    while index < NAMED_LAYOUTS.len() {
        if is_hardware(&NAMED_LAYOUTS[index].0) {
            count += 1;
        }
        index += 1;
    }
    count
}

/// How many layouts the table holds; a dialect's spelling table is sized by it.
pub(crate) const HARDWARE_COUNT: usize = hardware_count();

const fn hardware_table() -> [FloatLayout; HARDWARE_COUNT] {
    let mut table = [NAMED_LAYOUTS[0].0; HARDWARE_COUNT];
    let mut filled = 0;
    let mut index = 0;
    while index < NAMED_LAYOUTS.len() {
        if is_hardware(&NAMED_LAYOUTS[index].0) {
            table[filled] = NAMED_LAYOUTS[index].0;
            filled += 1;
        }
        index += 1;
    }
    table
}

/// The hardware layouts, in `NAMED_LAYOUTS` order.
pub(crate) const HARDWARE: [FloatLayout; HARDWARE_COUNT] = hardware_table();

/// One row of `HARDWARE`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HardwareScalar(u8);

impl HardwareScalar {
    /// Every row `dialect` spells, in table order.
    pub fn all<D: Dialect>(dialect: &D) -> impl Iterator<Item = Self> {
        (0..HARDWARE_COUNT)
            .filter_map(|index| u8::try_from(index).ok().map(Self))
            .filter(|row| dialect.scalar(*row).is_some())
    }

    /// The widest row `dialect` spells: the scalar every layout without a narrower row is
    /// computed in.
    #[must_use]
    pub fn widest<D: Dialect>(dialect: &D) -> Option<Self> {
        Self::all(dialect).max_by_key(|scalar| scalar.bytes())
    }

    /// The row `dialect` spells whose fields equal `layout`'s, compared field by field and never
    /// by name.
    #[must_use]
    pub fn of<D: Dialect>(dialect: &D, layout: &FloatLayout) -> Option<Self> {
        Self::all(dialect).find(|scalar| {
            let row = scalar.layout();
            row.exponent_bits == layout.exponent_bits
                && row.mantissa_bits == layout.mantissa_bits
                && row.specials == layout.specials
                && row.bias() == layout.bias()
                && row.sign == layout.sign
        })
    }

    /// Its position in `HARDWARE`; a dialect's spelling table is indexed by it.
    #[must_use]
    pub fn index(self) -> usize {
        usize::from(self.0)
    }

    /// The layout this row stands for.
    #[must_use]
    pub fn layout(self) -> FloatLayout {
        HARDWARE[self.index()]
    }

    /// Bytes one element occupies.
    #[must_use]
    pub fn bytes(self) -> u8 {
        self.layout().width() / 8
    }
}
