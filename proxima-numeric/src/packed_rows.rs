//! An owned run of values held as packed bytes in one [`NumericFormat`].
//!
//! Composes [`FormatCodec`] (`for_format`, `packed_len`, `encode`) for the write and
//! [`decode_blocks`] for the read: the bytes are the store, and a consumer that needs wide values
//! decodes the range it reads into a scratch it owns. [`Numeric`] is the borrowed
//! value that carries a format over wide values; this is the owned container that carries a
//! format over the packed bytes, so a binary32 run is just the format whose codec packs four
//! bytes per value. No branch here names a format.

use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::convert::{Overflow, Rounding};
use crate::decode::decode_blocks;
use crate::error::NumericError;
use crate::format::NumericFormat;
use crate::format_codec::FormatCodec;
use crate::sized::CONVERT_WINDOW_ELEMENTS;
use crate::value::Numeric;

const fn lcm(left: usize, right: usize) -> usize {
    let (mut high, mut low) = (left, right);
    while low != 0 {
        (high, low) = (low, high % low);
    }
    left / high * right
}

/// Values stored as the packed bytes of one format.
#[derive(Clone)]
pub struct PackedRows {
    // shared and heap-held: a `NumericFormat` is hundreds of bytes and would bloat every plane
    format: Arc<NumericFormat>,
    bytes: Vec<u8>,
    elements: usize,
}

impl core::fmt::Debug for PackedRows {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("PackedRows")
            .field("format", &self.format)
            .field("elements", &self.elements)
            .finish()
    }
}

impl PartialEq for PackedRows {
    fn eq(&self, other: &Self) -> bool {
        self.format == other.format && self.elements == other.elements && self.bytes == other.bytes
    }
}

impl PackedRows {
    /// An empty run of `format`.
    ///
    /// # Errors
    /// [`NumericError::UnsupportedEncode`] when `format` has no slice codec that encodes.
    pub fn new(format: &NumericFormat) -> Result<Self, NumericError> {
        Self::resolve(format)?;
        Ok(Self {
            format: Arc::new(*format),
            bytes: Vec::new(),
            elements: 0,
        })
    }

    // the codec is resolved per call, not stored: a stored one made every plane carry its function table
    fn resolve(format: &NumericFormat) -> Result<FormatCodec, NumericError> {
        match FormatCodec::for_format(format) {
            Some(codec) if codec.can_encode() => Ok(codec),
            _ => Err(NumericError::UnsupportedEncode {
                format: format.name().unwrap_or("unnamed"),
                reason: "no slice codec that encodes",
            }),
        }
    }

    fn codec(&self) -> Result<FormatCodec, NumericError> {
        Self::resolve(&self.format)
    }

    /// A run of `format` holding `values`, encoded.
    ///
    /// # Errors
    /// Whatever [`Self::new`] and [`Self::extend`] report.
    pub fn from_values(format: &NumericFormat, values: &[f32]) -> Result<Self, NumericError> {
        let mut rows = Self::new(format)?;
        rows.extend(values)?;
        Ok(rows)
    }

    /// A run of `format` over bytes that are already packed in it.
    ///
    /// # Errors
    /// [`NumericError::InputNotBlockMultiple`] when `bytes` is not whole blocks.
    pub fn from_bytes(format: &NumericFormat, bytes: Vec<u8>) -> Result<Self, NumericError> {
        let codec = Self::resolve(format)?;
        let mut rows = Self::new(format)?;
        let block_bytes = codec.block_bytes();
        if block_bytes == 0 || !bytes.len().is_multiple_of(block_bytes) {
            return Err(NumericError::InputNotBlockMultiple {
                format: codec.format_name(),
                found: bytes.len(),
                block_bytes,
            });
        }
        rows.elements = bytes.len() / block_bytes * codec.elements();
        rows.bytes = bytes;
        Ok(rows)
    }

    /// The format the bytes are packed in.
    #[must_use]
    pub fn format(&self) -> &NumericFormat {
        &self.format
    }

    /// Elements held.
    #[must_use]
    pub const fn elements(&self) -> usize {
        self.elements
    }

    /// Whether no element is held.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.elements == 0
    }

    /// The packed bytes, exactly as the format's encoder wrote them.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Bytes the allocation reserves.
    #[must_use]
    pub fn capacity_bytes(&self) -> usize {
        self.bytes.capacity()
    }

    fn not_whole_blocks(codec: &FormatCodec, found: usize) -> NumericError {
        NumericError::InputNotElementMultiple {
            format: codec.format_name(),
            found,
            elements: codec.elements(),
        }
    }

    fn range_bytes(&self, start: usize, len: usize) -> Result<core::ops::Range<usize>, NumericError> {
        let codec = self.codec()?;
        let from = codec.packed_len(start).ok_or_else(|| Self::not_whole_blocks(&codec, start))?;
        let upto = codec.packed_len(start + len).ok_or_else(|| Self::not_whole_blocks(&codec, len))?;
        Ok(from..upto)
    }

    /// Encodes `values` onto the end.
    ///
    /// # Errors
    /// [`NumericError::InputNotElementMultiple`] when `values` is not whole blocks; the encoder's
    /// own error otherwise. The run is unchanged on error.
    pub fn extend(&mut self, values: &[f32]) -> Result<(), NumericError> {
        let codec = self.codec()?;
        let at = self.bytes.len();
        let added = codec
            .packed_len(values.len())
            .ok_or_else(|| Self::not_whole_blocks(&codec, values.len()))?;
        self.bytes.resize(at + added, 0);
        if let Err(error) = codec.encode(values, &mut self.bytes[at..]) {
            self.bytes.truncate(at);
            return Err(error);
        }
        self.elements += values.len();
        Ok(())
    }

    /// Encodes `values` over the elements starting at `start`, which must already be held.
    ///
    /// # Errors
    /// [`NumericError::OutputSizeMismatch`] when the range runs past what is held; the encoder's
    /// own error otherwise.
    pub fn overwrite(&mut self, start: usize, values: &[f32]) -> Result<(), NumericError> {
        if start + values.len() > self.elements {
            return Err(NumericError::OutputSizeMismatch {
                found: start + values.len(),
                expected: self.elements,
            });
        }
        let range = self.range_bytes(start, values.len())?;
        self.codec()?.encode(values, &mut self.bytes[range])?;
        Ok(())
    }

    /// Appends `len` elements of `source` starting at element `start`, byte for byte: nothing is
    /// decoded or re-encoded, so the rows are exactly what `source` held.
    ///
    /// # Errors
    /// [`NumericError::OutputSizeMismatch`] when `source` is another format or the range runs past
    /// what it holds; [`NumericError::InputNotElementMultiple`] when it is not whole blocks.
    pub fn extend_from(&mut self, source: &Self, start: usize, len: usize) -> Result<(), NumericError> {
        let range = self.same_format_range(source, start, len)?;
        self.bytes.extend_from_slice(&source.bytes[range]);
        self.elements += len;
        Ok(())
    }

    /// [`Self::overwrite`] with `len` elements of `source` starting at `source_start`, byte for byte.
    ///
    /// # Errors
    /// As [`Self::extend_from`], plus [`NumericError::OutputSizeMismatch`] when the destination
    /// range runs past what is held.
    pub fn overwrite_from(
        &mut self,
        start: usize,
        source: &Self,
        source_start: usize,
        len: usize,
    ) -> Result<(), NumericError> {
        let from = self.same_format_range(source, source_start, len)?;
        if start + len > self.elements {
            return Err(NumericError::OutputSizeMismatch {
                found: start + len,
                expected: self.elements,
            });
        }
        let to = self.range_bytes(start, len)?;
        self.bytes[to].copy_from_slice(&source.bytes[from]);
        Ok(())
    }

    fn same_format_range(
        &self,
        source: &Self,
        start: usize,
        len: usize,
    ) -> Result<core::ops::Range<usize>, NumericError> {
        if source.format != self.format || start + len > source.elements {
            return Err(NumericError::OutputSizeMismatch {
                found: start + len,
                expected: source.elements,
            });
        }
        source.range_bytes(start, len)
    }

    /// Grows to `elements` with encoded zeros; never shrinks.
    ///
    /// # Errors
    /// Whatever [`Self::extend`] reports.
    pub fn grow_zeroed(&mut self, elements: usize) -> Result<(), NumericError> {
        if elements <= self.elements {
            return Ok(());
        }
        let zeros = alloc::vec![0.0f32; elements - self.elements];
        self.extend(&zeros)
    }

    /// Drops everything past `elements`; a no-op when no more than that is held.
    ///
    /// # Errors
    /// [`NumericError::InputNotElementMultiple`] when `elements` is not whole blocks.
    pub fn truncate(&mut self, elements: usize) -> Result<(), NumericError> {
        if elements >= self.elements {
            return Ok(());
        }
        let codec = self.codec()?;
        let keep = codec.packed_len(elements).ok_or_else(|| Self::not_whole_blocks(&codec, elements))?;
        self.bytes.truncate(keep);
        self.elements = elements;
        Ok(())
    }

    /// The packed bytes of the first `elements` elements.
    ///
    /// # Errors
    /// [`NumericError::OutputSizeMismatch`] when fewer are held; [`NumericError::InputNotElementMultiple`]
    /// when `elements` is not whole blocks.
    pub fn bytes_of(&self, elements: usize) -> Result<&[u8], NumericError> {
        if elements > self.elements {
            return Err(NumericError::OutputSizeMismatch {
                found: elements,
                expected: self.elements,
            });
        }
        let range = self.range_bytes(0, elements)?;
        Ok(&self.bytes[range])
    }

    /// Gives back the allocation's unused tail, so [`Self::capacity_bytes`] is the packed length.
    pub fn shrink_to_fit(&mut self) {
        self.bytes.shrink_to_fit();
    }

    /// Empties the run, keeping the allocation.
    pub fn clear(&mut self) {
        self.bytes.clear();
        self.elements = 0;
    }

    /// Empties the run and frees the allocation, so a later fill sizes to what it holds.
    pub fn release(&mut self) {
        self.bytes = Vec::new();
        self.elements = 0;
    }

    /// Decodes `out.len()` elements starting at element `start` into the caller's `out`.
    ///
    /// # Errors
    /// [`NumericError::OutputSizeMismatch`] when the range runs past what is held; the decoder's
    /// own error otherwise.
    pub fn decode_range(&self, start: usize, out: &mut [f32]) -> Result<(), NumericError> {
        if start + out.len() > self.elements {
            return Err(NumericError::OutputSizeMismatch {
                found: start + out.len(),
                expected: self.elements,
            });
        }
        let range = self.range_bytes(start, out.len())?;
        decode_blocks(&self.format, &self.bytes[range], out)?;
        Ok(())
    }

    /// Decodes every element into `out`, resized to fit.
    ///
    /// # Errors
    /// Whatever [`Self::decode_range`] reports.
    pub fn decode_all(&self, out: &mut Vec<f32>) -> Result<(), NumericError> {
        out.clear();
        out.resize(self.elements, 0.0);
        self.decode_range(0, out)
    }

    /// This run re-encoded in [`NumericFormat::host_element`]: one decode and one encode now, so
    /// no later view of the bytes needs either.
    ///
    /// # Errors
    /// Whatever [`Self::decode_all`] or the encode reports.
    pub fn to_host_element(&self) -> Result<Self, NumericError> {
        self.convert(
            &NumericFormat::host_element(),
            Rounding::NearestTiesEven,
            Overflow::Format,
        )
    }

    /// This run re-encoded in `target`, streamed through a window of
    /// [`CONVERT_WINDOW_ELEMENTS`] wide values (rounded down to a whole number of the source's and
    /// the target's blocks, never below one of each): each window is [`decode_blocks`]-decoded,
    /// then narrowed once by [`Numeric::to`] under `rounding` and `overflow`, so no whole-run wide
    /// buffer exists. A `target` that is the source format, or a scalar identity of it
    /// ([`NumericFormat::is_identity_of`]), returns the bytes unchanged. A target that is the host
    /// element takes the codec's own encode, which stores a decoded value exactly (a `NaN` too).
    ///
    /// # Errors
    /// [`NumericError::UnsupportedEncode`] when `target` has no slice codec that encodes;
    /// [`NumericError::InputNotElementMultiple`] when the run is not whole blocks of `target`; the
    /// decode's or [`Numeric::to`]'s own error otherwise (a `NaN` into a format that cannot store one).
    pub fn convert(
        &self,
        target: &NumericFormat,
        rounding: Rounding,
        overflow: Overflow,
    ) -> Result<Self, NumericError> {
        let target_codec = Self::resolve(target)?;
        if *self.format == *target || self.format.is_identity_of(target) {
            return Ok(Self {
                format: Arc::new(*target),
                bytes: self.bytes.clone(),
                elements: self.elements,
            });
        }
        let source_codec = self.codec()?;
        let total_bytes = target_codec
            .packed_len(self.elements)
            .ok_or_else(|| Self::not_whole_blocks(&target_codec, self.elements))?;
        let window =
            Self::window(source_codec.elements(), target_codec.elements()).min(self.elements);
        let mut wide = Vec::new();
        wide.resize_with(window, Default::default);
        let mut bytes = alloc::vec![0u8; total_bytes];
        let mut written = 0;
        for start in (0..self.elements).step_by(window.max(1)) {
            let len = window.min(self.elements - start);
            let values = &mut wide[..len];
            decode_blocks(
                &self.format,
                &self.bytes[self.range_bytes(start, len)?],
                values,
            )?;
            let packed = target_codec
                .packed_len(len)
                .ok_or_else(|| Self::not_whole_blocks(&target_codec, len))?;
            let out = &mut bytes[written..written + packed];
            written += packed;
            if target.is_host_element() {
                target_codec.encode(values, out)?;
            } else {
                Numeric::new(&self.format, values)?.to(target, rounding, overflow, out)?;
            }
        }
        Ok(Self {
            format: Arc::new(*target),
            bytes,
            elements: self.elements,
        })
    }

    // the largest whole number of both formats' blocks within the configured window, at least one
    fn window(source_elements: usize, target_elements: usize) -> usize {
        let block = lcm(source_elements.max(1), target_elements.max(1));
        (CONVERT_WINDOW_ELEMENTS / block).max(1) * block
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::float_layout::NAMED_LAYOUTS;

    fn every_format() -> impl Iterator<Item = NumericFormat> {
        NAMED_LAYOUTS
            .iter()
            .map(|(layout, _)| NumericFormat::Scalar(*layout))
            .chain(NumericFormat::ALL)
    }

    fn activations(count: usize) -> Vec<f32> {
        (0..count)
            .map(|index| ((index as f32) * 0.37).sin() * 3.1 + ((index as f32) * 0.011).cos())
            .collect()
    }

    #[test]
    fn a_binary32_run_is_four_bytes_a_value_and_reads_back_exactly() {
        let values = activations(64);
        let rows = PackedRows::from_values(&NumericFormat::Float32, &values).expect("binary32 encodes");
        let mut back = Vec::new();
        rows.decode_all(&mut back).expect("binary32 decodes");

        assert_eq!(rows.as_bytes().len(), 64 * 4);
        assert_eq!(back, values);
    }

    #[test]
    fn release_frees_the_allocation_where_clear_keeps_it_and_the_run_refills() {
        let values = activations(4096);
        let mut cleared = PackedRows::from_values(&NumericFormat::Float16, &values).expect("binary16 encodes");
        let mut released = cleared.clone();
        let held = cleared.capacity_bytes();

        cleared.clear();
        released.release();

        assert!(held >= 4096 * 2);
        assert_eq!(cleared.capacity_bytes(), held);
        assert_eq!(released.capacity_bytes(), 0);
        assert_eq!((cleared.elements(), released.elements()), (0, 0));
        released.extend(&values).expect("a released run refills");
        assert_eq!(released.as_bytes().len(), 4096 * 2);
    }

    #[test]
    fn a_q8_0_run_is_one_block_of_bytes_a_block_and_reads_back_what_the_codec_gives() {
        let values = activations(96);
        let rows = PackedRows::from_values(&NumericFormat::Q8_0, &values).expect("q8_0 encodes");
        let mut wide = Vec::new();
        rows.decode_all(&mut wide).expect("q8_0 decodes");
        let mut again = PackedRows::from_values(&NumericFormat::Q8_0, &wide).expect("re-encodes");
        let mut twice = Vec::new();
        again.decode_all(&mut twice).expect("decodes");

        assert_eq!(rows.as_bytes().len(), 3 * 34);
        assert_eq!(rows.elements(), 96);
        assert_eq!(wide, twice, "a decoded block re-encodes to the same values");
        again.clear();
        assert!(again.is_empty());
    }

    #[test]
    fn a_partial_block_is_refused_and_leaves_the_run_unchanged() {
        let mut rows = PackedRows::new(&NumericFormat::Q8_0).expect("q8_0 encodes");

        let refused = rows.extend(&activations(31));

        assert!(refused.is_err());
        assert_eq!(rows.elements(), 0);
        assert!(rows.as_bytes().is_empty());
    }

    #[test]
    fn truncate_overwrite_and_extend_from_address_whole_blocks_by_element() {
        let format = NumericFormat::Q8_0;
        let mut rows = PackedRows::from_values(&format, &activations(128)).expect("encodes");
        let donor = PackedRows::from_values(&format, &activations(64)).expect("encodes");

        rows.overwrite_from(32, &donor, 0, 32).expect("a whole block moves");
        let mut block = alloc::vec![0.0f32; 32];
        let mut expected = alloc::vec![0.0f32; 32];
        rows.decode_range(32, &mut block).expect("decodes");
        donor.decode_range(0, &mut expected).expect("decodes");
        assert_eq!(block, expected);

        rows.truncate(64).expect("a block boundary");
        rows.extend_from(&donor, 32, 32).expect("a whole block appends");
        assert_eq!(rows.elements(), 96);
        assert!(rows.truncate(40).is_err(), "40 elements is not whole blocks");
    }

    #[test]
    fn grow_zeroed_holds_encoded_zeros_and_never_shrinks() {
        let mut rows = PackedRows::new(&NumericFormat::Float16).expect("binary16 encodes");

        rows.grow_zeroed(8).expect("grows");
        rows.grow_zeroed(4).expect("never shrinks");
        let mut back = Vec::new();
        rows.decode_all(&mut back).expect("decodes");

        assert_eq!(back, alloc::vec![0.0f32; 8]);
    }

    #[test]
    fn bytes_of_another_format_are_not_a_run_of_this_one() {
        let donor = PackedRows::from_values(&NumericFormat::Float16, &activations(32)).expect("encodes");
        let mut rows = PackedRows::new(&NumericFormat::Float32).expect("encodes");

        assert!(rows.extend_from(&donor, 0, 32).is_err());
    }

    #[test]
    fn only_the_decoders_own_scalar_layout_is_the_host_element() {
        let host = NumericFormat::host_element();
        let (native, other): (Vec<NumericFormat>, Vec<NumericFormat>) =
            every_format().partition(NumericFormat::is_host_element);

        assert!(host.is_host_element());
        assert!(!native.is_empty());
        assert!(!other.is_empty());
        assert!(native.iter().all(|format| format.is_identity_of(&host)));
        assert!(other.iter().all(|format| !format.is_identity_of(&host)));
    }

    #[test]
    fn every_format_re_encodes_to_the_host_element_with_the_values_it_decoded() {
        let values = activations(64);
        let mut compared = 0;
        for format in every_format() {
            let Ok(rows) = PackedRows::from_values(&format, &values) else {
                continue;
            };
            let mut decoded = Vec::new();
            rows.decode_all(&mut decoded).expect("the run decodes");

            let host = rows.to_host_element().expect("the run re-encodes");
            let mut again = Vec::new();
            host.decode_all(&mut again).expect("the host run decodes");

            assert!(host.format().is_host_element());
            assert_eq!(again.len(), decoded.len());
            assert!(
                again
                    .iter()
                    .zip(&decoded)
                    .all(|(left, right)| left.to_bits() == right.to_bits()
                        || (left.is_nan() && right.is_nan())),
                "{format:?} changed a value"
            );
            compared += 1;
        }
        assert!(compared > 0, "no format encoded the run");
    }

    #[test]
    fn a_host_element_run_keeps_its_bytes() {
        let values = activations(64);
        let native = PackedRows::from_values(&NumericFormat::host_element(), &values)
            .expect("host element encodes");

        let kept = native.to_host_element().expect("host element re-encodes");

        assert!(!kept.as_bytes().is_empty());
        assert_eq!(kept.as_bytes(), native.as_bytes());
    }

    fn convert_nearest(
        rows: &PackedRows,
        target: &NumericFormat,
    ) -> Result<PackedRows, NumericError> {
        rows.convert(target, Rounding::NearestTiesEven, Overflow::Format)
    }

    // block formats the slice codec encodes, in the order the format table lists them
    fn encodable_blocks() -> Vec<NumericFormat> {
        NumericFormat::ALL
            .into_iter()
            .filter(|format| {
                matches!(format, NumericFormat::Block(_)) && PackedRows::new(format).is_ok()
            })
            .collect()
    }

    fn host_rows(count: usize) -> PackedRows {
        PackedRows::from_values(&NumericFormat::host_element(), &activations(count))
            .expect("host encodes")
    }

    #[test]
    fn convert_host_to_every_scalar_layout_and_back_matches_the_layout_codecs_narrowing() {
        let values = activations(257);
        let host = host_rows(257);
        let mut compared = 0;
        for (layout, _) in NAMED_LAYOUTS
            .iter()
            .filter(|(layout, _)| FormatCodec::scalar(layout).is_some())
        {
            let narrow = NumericFormat::Scalar(*layout);
            let back = convert_nearest(
                &convert_nearest(&host, &narrow).expect("narrows"),
                &NumericFormat::host_element(),
            )
            .expect("widens");
            let mut got = Vec::new();
            back.decode_all(&mut got).expect("decodes");

            let expected: Vec<_> = values
                .iter()
                .map(|value| {
                    crate::convert::narrow_to_layout(
                        *value,
                        *layout,
                        Rounding::NearestTiesEven,
                        Overflow::Format,
                    )
                })
                .collect::<Result<_, _>>()
                .expect("layout narrows");
            assert_eq!(
                got.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "{layout:?} differs from narrow_to_layout"
            );
            compared += 1;
        }
        assert!(compared > 0, "no scalar layout was compared");
    }

    #[test]
    fn convert_host_to_a_block_and_back_matches_encode_blocks_then_decode_blocks() {
        let block = encodable_blocks().remove(0);
        let count = usize::from(block.elements_per_block()) * 3;
        let values = activations(count);
        let host = host_rows(count);

        let quantized = convert_nearest(&host, &block).expect("block converts");
        let back =
            convert_nearest(&quantized, &NumericFormat::host_element()).expect("host converts");

        let mut packed = alloc::vec![0u8; 3 * usize::from(block.bytes_per_block())];
        crate::encode::encode_blocks(&block, &values, &mut packed).expect("encodes");
        let mut expected = Vec::new();
        expected.resize_with(count, Default::default);
        decode_blocks(&block, &packed, &mut expected).expect("decodes");
        let mut got = Vec::new();
        back.decode_all(&mut got).expect("decodes");
        assert_eq!(quantized.as_bytes(), packed.as_slice());
        assert_eq!(got, expected);
    }

    #[test]
    fn convert_between_two_block_formats_lands_in_the_target_geometry() {
        let blocks = encodable_blocks();
        let (first, second) = blocks
            .iter()
            .zip(blocks.iter().skip(1))
            .next()
            .expect("two block formats");
        let count = usize::from(first.elements_per_block())
            .max(usize::from(second.elements_per_block()))
            * 8;
        let source = convert_nearest(&host_rows(count), first).expect("scalar to block");

        let target = convert_nearest(&source, second).expect("block to block");

        let blocks_held = count / usize::from(second.elements_per_block());
        assert_eq!(target.elements(), count);
        assert_eq!(
            target.as_bytes().len(),
            blocks_held * usize::from(second.bytes_per_block())
        );
    }

    #[test]
    fn convert_to_the_same_format_returns_identical_bytes() {
        let block = encodable_blocks().remove(0);
        let narrow = NumericFormat::ALL
            .into_iter()
            .find(|format| {
                !format.is_host_element()
                    && matches!(format, NumericFormat::Scalar(_))
                    && PackedRows::new(format).is_ok()
            })
            .expect("a narrow scalar format");
        for format in [NumericFormat::host_element(), narrow, block] {
            let count = usize::from(format.elements_per_block()) * 4;
            let rows = PackedRows::from_values(&format, &activations(count)).expect("encodes");

            let same = convert_nearest(&rows, &format).expect("identity converts");

            assert_eq!(same, rows);
        }
    }

    #[test]
    fn convert_to_a_format_no_codec_encodes_is_a_typed_error() {
        let unsupported = NumericFormat::ALL
            .into_iter()
            .find(|format| PackedRows::new(format).is_err())
            .expect("some format has no encoding codec");

        let refused = convert_nearest(&host_rows(256), &unsupported);

        assert!(matches!(
            refused,
            Err(NumericError::UnsupportedEncode { .. })
        ));
    }

    #[test]
    fn convert_streams_a_run_larger_than_the_window_as_that_window_repeated() {
        let target = encodable_blocks().remove(0);
        let block = usize::from(target.elements_per_block());
        let window = PackedRows::window(1, block);
        let rows = host_rows(window * 2 + block);

        let whole = convert_nearest(&rows, &target).expect("converts");

        let mut stitched = PackedRows::new(&target).expect("new");
        let mut start = 0;
        while start < rows.elements() {
            let len = window.min(rows.elements() - start);
            let mut piece = PackedRows::new(rows.format()).expect("new");
            piece.extend_from(&rows, start, len).expect("slice");
            let converted = convert_nearest(&piece, &target).expect("converts");
            stitched.extend_from(&converted, 0, len).expect("append");
            start += len;
        }
        assert!(rows.elements() > window);
        assert_eq!(whole, stitched);
    }

    #[test]
    fn the_window_is_a_whole_number_of_both_formats_blocks() {
        let window = PackedRows::window(3, 256);

        assert_eq!(window % 3, 0);
        assert_eq!(window % 256, 0);
        assert!(window >= 768);
    }
}
