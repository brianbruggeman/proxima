//! A minimal PNG codec: 8-bit RGB/RGBA, no interlacing. The writer emits
//! uncompressed deflate blocks so a written image costs no dependency; the
//! reader inflates through `flate2` and applies every scanline filter the
//! format defines, which is what reading another tool's image needs.

use std::path::Path;

use proxima_numeric::{NumericFormat, PackedRows};

/// A decoded image: interleaved channels as codes of the sample format,
/// row-major from the top.
#[derive(Debug, Clone, PartialEq)]
pub struct PngImage {
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub pixels: PackedRows,
}

const BITS_PER_BYTE: usize = 8;

/// Whether `sample` is a one-byte unsigned integer, the only sample this codec's all-ones full
/// scale, scanline filters and byte-wide IHDR bit depth are correct for.
#[must_use]
pub fn is_supported_sample(sample: &NumericFormat) -> bool {
    sample.is_integer()
        && !sample.is_signed_integer()
        && sample.size_bytes() == 1
        && sample.bits() == BITS_PER_BYTE
}

fn require_sample(sample: &NumericFormat) -> Result<u8, String> {
    if !is_supported_sample(sample) {
        return Err(String::from("png samples must be a one-byte unsigned integer format"));
    }
    u8::try_from(sample.size_bytes() * BITS_PER_BYTE).map_err(|why| why.to_string())
}

/// One sample at its format's largest code, decoded: the divisor that maps a
/// stored code onto `[0, 1]` and the multiplier that maps `[0, 1]` onto codes.
///
/// # Errors
///
/// A message when the format has no codec.
pub fn full_scale(sample: &NumericFormat) -> Result<PackedRows, String> {
    PackedRows::from_bytes(sample, vec![0xff; sample.size_bytes()]).map_err(|why| why.to_string())
}

/// Encodes `pixels` (interleaved `channels` per pixel, `[0, 1]`) as a PNG whose
/// samples are stored in `sample`.
///
/// # Errors
///
/// A message naming the path and the failure when the file cannot be written
/// or the dimensions do not match the buffer.
pub fn write_png(
    path: &Path,
    width: u32,
    height: u32,
    channels: u8,
    pixels: &PackedRows,
    sample: &NumericFormat,
) -> Result<(), String> {
    let bit_depth = require_sample(sample)?;
    if !matches!(channels, 3 | 4) {
        return Err(format!("{channels} channels is not RGB or RGBA"));
    }
    if pixels.elements() != (width as usize) * (height as usize) * channels as usize {
        return Err(String::from("pixel buffer does not match the dimensions"));
    }
    let mut scale = Vec::new();
    full_scale(sample)?
        .decode_all(&mut scale)
        .map_err(|why| why.to_string())?;
    let mut wide = Vec::new();
    pixels
        .decode_all(&mut wide)
        .map_err(|why| why.to_string())?;
    for value in &mut wide {
        *value = (value.clamp(0.0, 1.0) * scale[0]).round();
    }
    let codes = PackedRows::from_values(sample, &wide).map_err(|why| why.to_string())?;
    let stride = width as usize * channels as usize * sample.size_bytes();
    let mut raw = Vec::with_capacity((stride + 1) * height as usize);
    for row in codes.as_bytes().chunks(stride) {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    let mut encoded = Vec::new();
    encoded.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
    let mut header = Vec::with_capacity(13);
    header.extend_from_slice(&width.to_be_bytes());
    header.extend_from_slice(&height.to_be_bytes());
    header.extend_from_slice(&[bit_depth, if channels == 3 { 2 } else { 6 }, 0, 0, 0]);
    push_chunk(&mut encoded, b"IHDR", &header);
    push_chunk(&mut encoded, b"IDAT", &zlib_stored(&raw));
    push_chunk(&mut encoded, b"IEND", &[]);
    std::fs::write(path, encoded).map_err(|why| format!("{}: {why}", path.display()))
}

/// Reads a PNG whose bit depth is `sample`'s width, RGB/RGBA, into `sample` codes.
///
/// # Errors
///
/// A message naming the path when it is unreadable, and the unsupported
/// feature when the file uses one this reader does not cover.
pub fn read_png(path: &Path, sample: &NumericFormat) -> Result<PngImage, String> {
    let expected_depth = require_sample(sample)?;
    let bytes = std::fs::read(path).map_err(|why| format!("{}: {why}", path.display()))?;
    if bytes.len() < 8 || bytes[..8] != [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a] {
        return Err(format!("{}: not a PNG", path.display()));
    }
    let mut offset = 8;
    let (mut width, mut height, mut channels) = (0u32, 0u32, 0u8);
    let mut compressed = Vec::new();
    while offset + 8 <= bytes.len() {
        let length = u32::from_be_bytes(bytes[offset..offset + 4].try_into().map_err(|_| "chunk length")?) as usize;
        let kind = &bytes[offset + 4..offset + 8];
        let body = bytes
            .get(offset + 8..offset + 8 + length)
            .ok_or_else(|| format!("{}: truncated chunk", path.display()))?;
        match kind {
            b"IHDR" => {
                let bit_depth = body[8];
                let colour = body[9];
                let interlace = body[12];
                if bit_depth != expected_depth || interlace != 0 || !matches!(colour, 2 | 6) {
                    return Err(format!(
                        "{}: only {expected_depth}-bit non-interlaced RGB/RGBA is read (depth {bit_depth}, colour {colour}, interlace {interlace})",
                        path.display()
                    ));
                }
                width = u32::from_be_bytes(body[0..4].try_into().map_err(|_| "width")?);
                height = u32::from_be_bytes(body[4..8].try_into().map_err(|_| "height")?);
                channels = if colour == 2 { 3 } else { 4 };
            }
            b"IDAT" => compressed.extend_from_slice(body),
            b"IEND" => break,
            _ => {}
        }
        offset += 12 + length;
    }
    let raw = inflate(&compressed).ok_or_else(|| format!("{}: corrupt deflate stream", path.display()))?;
    let pixel_bytes = channels as usize * sample.size_bytes();
    let stride = width as usize * pixel_bytes;
    if raw.len() != (stride + 1) * height as usize {
        return Err(format!("{}: unexpected data length {}", path.display(), raw.len()));
    }
    let mut codes = Vec::with_capacity(stride * height as usize);
    let mut previous = vec![0u8; stride];
    let mut current = vec![0u8; stride];
    for row in 0..height as usize {
        let start = row * (stride + 1);
        let filter = raw[start];
        current.copy_from_slice(&raw[start + 1..start + 1 + stride]);
        unfilter(filter, pixel_bytes, &mut current, &previous)
            .ok_or_else(|| format!("{}: filter {filter} is not supported", path.display()))?;
        codes.extend_from_slice(&current);
        previous.copy_from_slice(&current);
    }
    let pixels = PackedRows::from_bytes(sample, codes).map_err(|why| why.to_string())?;
    Ok(PngImage {
        width,
        height,
        channels,
        pixels,
    })
}

fn unfilter(filter: u8, channels: usize, current: &mut [u8], previous: &[u8]) -> Option<()> {
    match filter {
        0 => Some(()),
        1 => {
            for index in channels..current.len() {
                current[index] = current[index].wrapping_add(current[index - channels]);
            }
            Some(())
        }
        2 => {
            for (index, value) in current.iter_mut().enumerate() {
                *value = value.wrapping_add(previous[index]);
            }
            Some(())
        }
        3 => {
            for index in 0..current.len() {
                let left = if index >= channels { u16::from(current[index - channels]) } else { 0 };
                let above = u16::from(previous[index]);
                current[index] = current[index].wrapping_add(((left + above) / 2) as u8);
            }
            Some(())
        }
        4 => {
            for index in 0..current.len() {
                let left = i32::from(if index >= channels { current[index - channels] } else { 0 });
                let above = i32::from(previous[index]);
                let corner = i32::from(if index >= channels { previous[index - channels] } else { 0 });
                current[index] = current[index].wrapping_add(paeth(left, above, corner));
            }
            Some(())
        }
        _ => None,
    }
}

fn paeth(left: i32, above: i32, corner: i32) -> u8 {
    let estimate = left + above - corner;
    let (to_left, to_above, to_corner) = (
        (estimate - left).abs(),
        (estimate - above).abs(),
        (estimate - corner).abs(),
    );
    let chosen = if to_left <= to_above && to_left <= to_corner {
        left
    } else if to_above <= to_corner {
        above
    } else {
        corner
    };
    chosen.clamp(0, 255) as u8
}

fn push_chunk(out: &mut Vec<u8>, kind: &[u8; 4], body: &[u8]) {
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(body);
    let mut crc = Crc32::new();
    crc.update(kind);
    crc.update(body);
    out.extend_from_slice(&crc.finish().to_be_bytes());
}

/// A zlib stream of stored deflate blocks: a valid encoding every reader takes,
/// with no compressor.
fn zlib_stored(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() + raw.len() / 65535 * 5 + 6);
    out.extend_from_slice(&[0x78, 0x01]);
    let mut blocks = raw.chunks(65535).peekable();
    if blocks.peek().is_none() {
        out.extend_from_slice(&[0x01, 0x00, 0x00, 0xff, 0xff]);
    }
    while let Some(block) = blocks.next() {
        out.push(u8::from(blocks.peek().is_none()));
        let length = block.len() as u16;
        out.extend_from_slice(&length.to_le_bytes());
        out.extend_from_slice(&(!length).to_le_bytes());
        out.extend_from_slice(block);
    }
    out.extend_from_slice(&adler32(raw).to_be_bytes());
    out
}

fn adler32(data: &[u8]) -> u32 {
    let (mut low, mut high) = (1u32, 0u32);
    for &byte in data {
        low = (low + u32::from(byte)) % 65521;
        high = (high + low) % 65521;
    }
    (high << 16) | low
}

struct Crc32 {
    table: [u32; 256],
    value: u32,
}

impl Crc32 {
    fn new() -> Self {
        let mut table = [0u32; 256];
        for (index, entry) in table.iter_mut().enumerate() {
            let mut crc = index as u32;
            for _ in 0..8 {
                crc = if crc & 1 == 1 { 0xedb8_8320 ^ (crc >> 1) } else { crc >> 1 };
            }
            *entry = crc;
        }
        Self { table, value: 0xffff_ffff }
    }

    fn update(&mut self, data: &[u8]) {
        for &byte in data {
            self.value = self.table[((self.value ^ u32::from(byte)) & 0xff) as usize] ^ (self.value >> 8);
        }
    }

    fn finish(self) -> u32 {
        self.value ^ 0xffff_ffff
    }
}

fn inflate(data: &[u8]) -> Option<Vec<u8>> {
    use std::io::Read;
    let mut decoder = flate2::read::ZlibDecoder::new(data);
    let mut out = Vec::new();
    decoder.read_to_end(&mut out).ok()?;
    Some(out)
}
