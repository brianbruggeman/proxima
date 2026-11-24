//! A checkpoint reader for the diffusion graphs: binds each declared weight
//! leaf from a safetensors file, re-encoding it into the packed codec the
//! backend accelerates and reordering a convolution kernel's axes so a tap's
//! reduce reads its channels contiguously.

use std::path::Path;

use proxima_numeric::codec::Codec;
use proxima_numeric::{DType, FormatCodec, NumericFormat, PackedRows};

use crate::CheckpointMapping;

use super::WeightBinding;

/// One bound leaf's bytes: a packed matrix the backend's matmul paths decode,
/// or the plain scalar vector a per-channel scale or bias binds (a packed
/// rank-1 operand has no matmul reader, so it stays plain). A packed matrix is
/// encoded in [`Weights::codec`], which keeps the 672-byte codec out of the
/// variant.
#[derive(Debug)]
pub enum WeightData {
    Packed(Vec<u8>),
    Plain(PackedRows),
}

/// The bound leaves, each holding the exact bytes the plan reads.
#[derive(Debug)]
pub struct Weights {
    pub codec: Codec,
    pub entries: Vec<(String, WeightData)>,
}

/// One safetensors file's tensor directory over its mapped bytes: the file is
/// addressed in place through [`CheckpointMapping`], never copied to the heap.
pub struct Checkpoint {
    mapping: CheckpointMapping,
    manifest: proxima_safetensors::Manifest,
    data_start: usize,
}

impl Checkpoint {
    /// Maps `path` and parses its header.
    ///
    /// # Errors
    ///
    /// A message naming the file and the mapping or parse failure.
    pub fn open(path: &Path) -> Result<Self, String> {
        let mapping = CheckpointMapping::open_file(path)
            .map_err(|why| format!("{}: {why}", path.display()))?;
        let bytes = mapping.bytes();
        let manifest = proxima_safetensors::parse_complete(bytes)
            .map_err(|why| format!("{}: {why}", path.display()))?;
        if bytes.len() < 8 {
            return Err(format!("{}: truncated header", path.display()));
        }
        let header = u64::from_le_bytes(
            bytes[..8]
                .try_into()
                .map_err(|_| format!("{}: truncated length", path.display()))?,
        );
        Ok(Self {
            mapping,
            manifest,
            data_start: 8 + header as usize,
        })
    }

    /// The tensor directory parsed from the file's header.
    #[must_use]
    pub fn manifest(&self) -> &proxima_safetensors::Manifest {
        &self.manifest
    }

    /// The whole file's mapped bytes, the view a loader borrows weights from.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        self.mapping.bytes()
    }

    /// Where tensor data begins in [`Self::bytes`], past the length prefix and header.
    #[must_use]
    pub fn data_start(&self) -> u64 {
        self.data_start as u64
    }

    /// One tensor's `(dtype, shape, bytes)`, offsets already absolute.
    ///
    /// # Errors
    ///
    /// A message naming the tensor when the header does not carry it.
    pub fn tensor(&self, name: &str) -> Result<(DType, Vec<u64>, &[u8]), String> {
        let entry = self
            .manifest
            .tensor(name)
            .ok_or_else(|| format!("checkpoint has no tensor {name}"))?;
        let start = self.data_start + entry.data_offsets.0 as usize;
        let end = self.data_start + entry.data_offsets.1 as usize;
        Ok((
            entry.dtype,
            entry.shape.clone(),
            &self.mapping.bytes()[start..end],
        ))
    }

    /// The header format the bindings' tensors share, `None` for no bindings.
    ///
    /// # Errors
    ///
    /// A message naming the tensor when it is missing, has no numeric format,
    /// or disagrees with the first binding's.
    pub fn stored_format(
        &self,
        bindings: &[WeightBinding],
    ) -> Result<Option<NumericFormat>, String> {
        let mut found: Option<NumericFormat> = None;
        for binding in bindings {
            let format = self
                .tensor(&binding.tensor)?
                .0
                .format()
                .ok_or_else(|| format!("{}: no numeric format", binding.tensor))?;
            match found {
                Some(first) if first != format => {
                    return Err(format!("{}: stored format differs from {first:?}", binding.tensor));
                }
                _ => found = Some(format),
            }
        }
        Ok(found)
    }

    /// The square window extent of a convolution weight stored `[out, in, kh, kw]`.
    ///
    /// # Errors
    ///
    /// A message naming the tensor when it is not rank 4 or `kh != kw`.
    pub fn window_extent(&self, name: &str) -> Result<u32, String> {
        let (_, shape, _) = self.tensor(name)?;
        match shape.as_slice() {
            [_, _, rows, columns] if rows == columns => {
                let window = u32::try_from(*rows).map_err(|why| format!("{name}: {why}"))?;
                if window == 0 || window % 2 == 0 {
                    return Err(format!("{name}: window {window} is not odd and nonzero"));
                }
                Ok(window)
            }
            _ => Err(format!("{name}: not a square [out, in, kh, kw] kernel")),
        }
    }

    /// Decodes one tensor, optionally slicing rows and permuting a convolution
    /// kernel's `[out, in, k, k]` axes to `[out, k, k, in]`, and returns it
    /// packed in the tensor's own format.
    ///
    /// # Errors
    ///
    /// A message naming the tensor and the failing step.
    pub fn values(
        &self,
        name: &str,
        rows: Option<(u32, u32)>,
        columns: Option<(u32, u32)>,
        permute_kernel: bool,
    ) -> Result<(Vec<u64>, PackedRows), String> {
        let (dtype, shape, bytes) = self.tensor(name)?;
        let format = dtype
            .format()
            .ok_or_else(|| format!("{name}: no numeric format"))?;
        let per_row: u64 = shape.iter().skip(1).product();
        let row_bytes = bytes.len() / (shape[0].max(1) as usize);
        let (first, last) = rows.unwrap_or((0, shape[0] as u32));
        let available = bytes.len() / row_bytes.max(1);
        let last = (last as usize).min(available);
        let first = (first as usize).min(last);
        let start = first * row_bytes;
        let end = last * row_bytes;
        let mut shape = shape;
        shape[0] = (last - first) as u64;
        if columns.is_none() && !(permute_kernel && shape.len() == 4) {
            let packed = PackedRows::from_bytes(&format, bytes[start..end].to_vec())
                .map_err(|why| format!("{name}: {why}"))?;
            return Ok((shape, packed));
        }
        let codec = FormatCodec::for_format(&format)
            .ok_or_else(|| format!("{name}: no codec for its format"))?;
        let count = (last - first) * per_row as usize;
        let mut slots = vec![core::mem::MaybeUninit::uninit(); count];
        codec
            .decode(&bytes[start..end], &mut slots)
            .map_err(|why| format!("{name}: {why}"))?;
        let mut values: Vec<_> = slots
            .into_iter()
            // SAFETY: decode returned Ok, which writes every one of the `count` slots it was given
            .map(|value| unsafe { value.assume_init() })
            .collect();
        if let Some((from, to)) = columns {
            let width = shape[1] as usize;
            let selected = (to - from) as usize;
            let mut windowed = Vec::with_capacity(values.len() / width * selected);
            for row in values.chunks(width) {
                windowed.extend_from_slice(&row[from as usize..(from as usize + selected).min(width)]);
            }
            values = windowed;
            shape[1] = selected as u64;
        }
        if permute_kernel && shape.len() == 4 {
            values = permute_kernel_axes(&values, &shape);
            shape = vec![shape[0], shape[2], shape[3], shape[1]];
        }
        let packed =
            PackedRows::from_values(&format, &values).map_err(|why| format!("{name}: {why}"))?;
        Ok((shape, packed))
    }
}

/// `[out, in, k, k]` row-major to `[out, k, k, in]`.
fn permute_kernel_axes<T: Copy>(values: &[T], shape: &[u64]) -> Vec<T> {
    let (out, into, kernel, columns) = (
        shape[0] as usize,
        shape[1] as usize,
        shape[2] as usize,
        shape[3] as usize,
    );
    let mut permuted = values.to_vec();
    for channel_out in 0..out {
        for channel_in in 0..into {
            for row in 0..kernel {
                for column in 0..columns {
                    let source = ((channel_out * into + channel_in) * kernel + row) * columns + column;
                    let target = ((channel_out * kernel + row) * columns + column) * into + channel_in;
                    permuted[target] = values[source];
                }
            }
        }
    }
    permuted
}

/// Binds every declared weight leaf, `codec`-encoded, with each convolution
/// kernel reordered to `[out, k, k, in]`.
///
/// # Errors
///
/// A message naming the leaf and the failing step.
pub fn load_weights(
    checkpoint: &Checkpoint,
    bindings: &[WeightBinding],
    codec: Codec,
    permute_kernels: bool,
) -> Result<Weights, String> {
    let mut entries = Vec::with_capacity(bindings.len());
    for binding in bindings {
        let (shape, values) = checkpoint.values(
            &binding.tensor,
            binding.rows,
            binding.columns,
            permute_kernels
                && (binding.tap.is_some() || binding.taps || binding.tensor.ends_with("conv.weight")),
        )?;
        if shape.len() == 1 {
            entries.push((binding.leaf.clone(), WeightData::Plain(values)));
            continue;
        }
        let mut wide = Vec::new();
        values
            .decode_all(&mut wide)
            .map_err(|why| format!("{}: {why}", binding.leaf))?;
        let wide = match binding.tap {
            Some((row, column)) => kernel_tap(&wide, &shape, row, column),
            None => wide,
        };
        let packed = PackedRows::from_values(&codec, &wide)
            .map_err(|why| format!("{}: {codec:?}: {why}", binding.leaf))?;
        entries.push((
            binding.leaf.clone(),
            WeightData::Packed(packed.as_bytes().to_vec()),
        ));
    }
    Ok(Weights { codec, entries })
}

/// `[out, k, k, in]`'s kernel position `(row, column)` as a contiguous
/// `[out, in]` matrix.
fn kernel_tap<T: Copy>(values: &[T], shape: &[u64], row: u32, column: u32) -> Vec<T> {
    let out = shape[0] as usize;
    let kernel = shape[1] as usize;
    let columns = shape[2] as usize;
    let into = shape[3] as usize;
    let mut tapped = Vec::with_capacity(out * into);
    for channel_out in 0..out {
        let base = ((channel_out * kernel + row as usize) * columns + column as usize) * into;
        tapped.extend_from_slice(&values[base..base + into]);
    }
    tapped
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(tensor: &str) -> WeightBinding {
        WeightBinding {
            leaf: tensor.into(),
            tensor: tensor.into(),
            rows: None,
            columns: None,
            tap: None,
            taps: false,
        }
    }

    fn open_header(dir: &Path, dtypes: [&str; 3]) -> Checkpoint {
        let header = format!(
            r#"{{"a":{{"dtype":"{}","shape":[1,2],"data_offsets":[0,2]}},"b":{{"dtype":"{}","shape":[1,2],"data_offsets":[2,4]}},"c":{{"dtype":"{}","shape":[1,2],"data_offsets":[4,6]}}}}"#,
            dtypes[0], dtypes[1], dtypes[2]
        );
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0u8; 6]);
        let path = dir.join("synthetic.safetensors");
        std::fs::write(&path, bytes).expect("the synthetic checkpoint writes");
        Checkpoint::open(&path).expect("the synthetic checkpoint opens")
    }

    #[test]
    fn the_stored_header_is_reported_and_a_disagreement_is_an_error() {
        let dir = tempfile::tempdir().expect("a scratch directory");
        let checkpoint = open_header(dir.path(), ["I8", "I8", "U8"]);
        let header = checkpoint.tensor("a").expect("a is declared").0.format();

        let none = checkpoint.stored_format(&[]).expect("an empty list is fine");
        let agreed = checkpoint.stored_format(&[binding("a"), binding("b")]);
        let split = checkpoint.stored_format(&[binding("a"), binding("c")]);

        assert_eq!(none, None);
        assert!(header.is_some(), "the header dtype has a numeric format");
        assert_eq!(agreed.expect("matching headers agree"), header);
        assert!(split.is_err(), "differing headers must not pick one");
    }
}
