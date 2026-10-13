use cudarc::nvrtc::CompileOptions;
use sha2::{Digest, Sha256};

/// Bump to invalidate every cached PTX entry; it is part of both the key and the entry header.
pub const CACHE_SCHEMA_VERSION: u32 = 2;

const ENTRY_MAGIC: [u8; 4] = *b"PXKC";
const DIGEST_LEN: usize = 32;
const HEADER_LEN: usize = ENTRY_MAGIC.len() + 4 + DIGEST_LEN + 8 + DIGEST_LEN;

/// Content-addressed name of one compiled kernel: sha256 over every input that
/// can change the PTX NVRTC emits for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheKey([u8; DIGEST_LEN]);

impl CacheKey {
    /// `header` is the resolved `cuda_fp16.h` bytes when the source includes it, `None` otherwise;
    /// `nvrtc_version` and `compute_capability` are `(major, minor)`.
    #[must_use]
    pub fn new(
        source: &str,
        header: Option<&[u8]>,
        nvrtc_version: (i32, i32),
        options: &CompileOptions,
        compute_capability: (i32, i32),
    ) -> Self {
        Self::with_schema(
            CACHE_SCHEMA_VERSION,
            source,
            header,
            nvrtc_version,
            options,
            compute_capability,
        )
    }

    fn with_schema(
        schema: u32,
        source: &str,
        header: Option<&[u8]>,
        nvrtc_version: (i32, i32),
        options: &CompileOptions,
        compute_capability: (i32, i32),
    ) -> Self {
        let mut digest = Sha256::new();
        push_field(&mut digest, &schema.to_le_bytes());
        push_field(&mut digest, source.as_bytes());
        push_optional(&mut digest, header);
        push_field(&mut digest, &version_bytes(nvrtc_version));
        push_compile_options(&mut digest, options);
        push_field(&mut digest, &version_bytes(compute_capability));
        Self(digest.finalize().into())
    }

    #[must_use]
    pub fn to_hex(&self) -> String {
        self.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

fn push_field(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}

fn push_optional<T: AsRef<[u8]>>(digest: &mut Sha256, value: Option<T>) {
    match value {
        None => digest.update([0]),
        Some(bytes) => {
            digest.update([1]);
            push_field(digest, bytes.as_ref());
        }
    }
}

fn push_list(digest: &mut Sha256, items: &[String]) {
    push_field(digest, &(items.len() as u64).to_le_bytes());
    for item in items {
        push_field(digest, item.as_bytes());
    }
}

fn push_compile_options(digest: &mut Sha256, options: &CompileOptions) {
    let CompileOptions {
        ftz,
        prec_sqrt,
        prec_div,
        fmad,
        options: flags,
        use_fast_math,
        maxrregcount,
        include_paths,
        arch,
        name,
    } = options;
    push_optional(digest, ftz.map(flag_byte));
    push_optional(digest, prec_sqrt.map(flag_byte));
    push_optional(digest, prec_div.map(flag_byte));
    push_optional(digest, fmad.map(flag_byte));
    push_list(digest, flags);
    push_optional(digest, use_fast_math.map(flag_byte));
    push_optional(digest, maxrregcount.map(usize::to_le_bytes));
    push_list(digest, include_paths);
    push_optional(digest, *arch);
    push_optional(digest, name.as_deref());
}

fn flag_byte(flag: bool) -> [u8; 1] {
    [u8::from(flag)]
}

fn version_bytes((major, minor): (i32, i32)) -> [u8; 8] {
    let mut bytes = [0; 8];
    bytes[..4].copy_from_slice(&major.to_le_bytes());
    bytes[4..].copy_from_slice(&minor.to_le_bytes());
    bytes
}

/// Why a stored entry was refused. Every variant is a miss for the caller, never a failed run.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CacheEntryError {
    #[error("cache entry is {found} bytes, shorter than its {needed}-byte header")]
    Truncated { needed: usize, found: usize },
    #[error("cache entry does not start with the ptx cache magic")]
    BadMagic,
    #[error("cache entry schema {found} does not match schema {expected}")]
    SchemaMismatch { found: u32, expected: u32 },
    #[error("cache entry was stored under a different key")]
    KeyMismatch,
    #[error("cache entry declares a {declared}-byte payload but holds {found} bytes")]
    LengthMismatch { declared: u64, found: usize },
    #[error("cache entry payload sha256 does not match its header")]
    HashMismatch,
    #[error("cache entry payload is not utf-8 ptx text")]
    NotUtf8,
}

/// Header (magic, schema, key digest, payload length, payload sha256) followed by the PTX text.
#[must_use]
pub fn encode_entry(key: &CacheKey, payload: &str) -> Vec<u8> {
    let mut entry = Vec::with_capacity(HEADER_LEN + payload.len());
    entry.extend_from_slice(&ENTRY_MAGIC);
    entry.extend_from_slice(&CACHE_SCHEMA_VERSION.to_le_bytes());
    entry.extend_from_slice(&key.0);
    entry.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    entry.extend_from_slice(&payload_digest(payload.as_bytes()));
    entry.extend_from_slice(payload.as_bytes());
    entry
}

/// Verifies magic, schema, key, length and payload sha256 before handing back the
/// PTX text borrowed from `entry`. `key` must be the key the entry was requested under.
pub fn decode_entry<'entry>(
    key: &CacheKey,
    entry: &'entry [u8],
) -> Result<&'entry str, CacheEntryError> {
    let truncated = || CacheEntryError::Truncated {
        needed: HEADER_LEN,
        found: entry.len(),
    };
    let mut cursor = entry;
    let magic = take::<4>(&mut cursor).ok_or_else(truncated)?;
    if magic != ENTRY_MAGIC {
        return Err(CacheEntryError::BadMagic);
    }
    let found = u32::from_le_bytes(take::<4>(&mut cursor).ok_or_else(truncated)?);
    if found != CACHE_SCHEMA_VERSION {
        return Err(CacheEntryError::SchemaMismatch {
            found,
            expected: CACHE_SCHEMA_VERSION,
        });
    }
    let stored_key = take::<DIGEST_LEN>(&mut cursor).ok_or_else(truncated)?;
    if stored_key != key.0 {
        return Err(CacheEntryError::KeyMismatch);
    }
    let declared = u64::from_le_bytes(take::<8>(&mut cursor).ok_or_else(truncated)?);
    let declared_digest = take::<DIGEST_LEN>(&mut cursor).ok_or_else(truncated)?;
    let payload = cursor;
    if u64::try_from(payload.len()) != Ok(declared) {
        return Err(CacheEntryError::LengthMismatch {
            declared,
            found: payload.len(),
        });
    }
    if payload_digest(payload) != declared_digest {
        return Err(CacheEntryError::HashMismatch);
    }
    std::str::from_utf8(payload).map_err(|_| CacheEntryError::NotUtf8)
}

fn take<const N: usize>(cursor: &mut &[u8]) -> Option<[u8; N]> {
    let (head, tail) = cursor.split_first_chunk::<N>()?;
    *cursor = tail;
    Some(*head)
}

fn payload_digest(payload: &[u8]) -> [u8; DIGEST_LEN] {
    Sha256::digest(payload).into()
}

#[cfg(test)]
mod tests {
    use cudarc::nvrtc::CompileOptions;
    use sha2::{Digest, Sha256};

    use super::{
        CACHE_SCHEMA_VERSION, CacheEntryError, CacheKey, HEADER_LEN, decode_entry, encode_entry,
    };

    const SOURCE: &str = "#include <cuda_fp16.h>\nextern \"C\" __global__ void add_one(float *x) { x[0] += 1.0f; }\n";
    const HEADER: &[u8] =
        b"#define CUDA_FP16_H\ntypedef struct __half { unsigned short x; } __half;\n";
    const NVRTC: (i32, i32) = (12, 4);
    const CAPABILITY: (i32, i32) = (7, 5);
    const PTX: &str = ".version 7.5\n.target sm_75\n.address_size 64\n\n.visible .entry add_one(\n    .param .u64 add_one_param_0\n)\n{\n    ret;\n}\n";

    fn base_options() -> CompileOptions {
        CompileOptions {
            ftz: Some(false),
            prec_sqrt: Some(true),
            prec_div: Some(true),
            fmad: Some(true),
            options: vec!["--std=c++17".to_owned()],
            use_fast_math: Some(false),
            maxrregcount: Some(64),
            include_paths: vec!["/usr/local/cuda/include".to_owned()],
            arch: Some("sm_75"),
            name: Some("proxima_add_one.cu".to_owned()),
        }
    }

    fn base_key() -> CacheKey {
        CacheKey::new(SOURCE, Some(HEADER), NVRTC, &base_options(), CAPABILITY)
    }

    fn option_key(change: fn(&mut CompileOptions)) -> CacheKey {
        let mut options = base_options();
        change(&mut options);
        CacheKey::new(SOURCE, Some(HEADER), NVRTC, &options, CAPABILITY)
    }

    fn option_cases() -> Vec<(&'static str, CacheKey)> {
        vec![
            ("ftz", option_key(|o| o.ftz = Some(true))),
            ("ftz absent", option_key(|o| o.ftz = None)),
            ("prec_sqrt", option_key(|o| o.prec_sqrt = Some(false))),
            ("prec_div", option_key(|o| o.prec_div = Some(false))),
            ("fmad", option_key(|o| o.fmad = Some(false))),
            ("options", option_key(|o| o.options.push("-G".to_owned()))),
            (
                "use_fast_math",
                option_key(|o| o.use_fast_math = Some(true)),
            ),
            ("maxrregcount", option_key(|o| o.maxrregcount = Some(128))),
            ("maxrregcount absent", option_key(|o| o.maxrregcount = None)),
            (
                "include_paths",
                option_key(|o| o.include_paths.push("/opt/cuda/include".to_owned())),
            ),
            ("arch", option_key(|o| o.arch = Some("sm_86"))),
            ("arch absent", option_key(|o| o.arch = None)),
            ("name", option_key(|o| o.name = Some("other.cu".to_owned()))),
        ]
    }

    fn input_cases() -> Vec<(&'static str, CacheKey)> {
        let options = base_options();
        vec![
            (
                "source",
                CacheKey::new(
                    "extern \"C\" __global__ void add_two() {}",
                    Some(HEADER),
                    NVRTC,
                    &options,
                    CAPABILITY,
                ),
            ),
            (
                "header bytes",
                CacheKey::new(
                    SOURCE,
                    Some(b"#define CUDA_FP16_H 2\n"),
                    NVRTC,
                    &options,
                    CAPABILITY,
                ),
            ),
            (
                "header absent",
                CacheKey::new(SOURCE, None, NVRTC, &options, CAPABILITY),
            ),
            (
                "nvrtc version",
                CacheKey::new(SOURCE, Some(HEADER), (12, 5), &options, CAPABILITY),
            ),
            (
                "compute capability",
                CacheKey::new(SOURCE, Some(HEADER), NVRTC, &options, (8, 6)),
            ),
        ]
    }

    #[test]
    fn identical_inputs_give_identical_key() {
        let first = base_key();
        let second = base_key();
        assert_eq!(first, second, "same inputs must name the same entry");
        assert_eq!(first.to_hex().len(), 64, "sha256 hex is 64 characters");
    }

    #[test]
    fn every_option_field_changes_the_key() {
        let baseline = base_key();
        let cases = option_cases();
        assert_eq!(
            cases.len(),
            13,
            "option table covers every CompileOptions field"
        );
        for (label, key) in &cases {
            assert_ne!(*key, baseline, "changing `{label}` must change the key");
        }
    }

    #[test]
    fn every_input_changes_the_key() {
        let baseline = base_key();
        let cases = input_cases();
        assert_eq!(
            cases.len(),
            5,
            "input table covers source, header, nvrtc, capability"
        );
        for (label, key) in &cases {
            assert_ne!(*key, baseline, "changing `{label}` must change the key");
        }
    }

    #[test]
    fn all_key_variants_are_pairwise_distinct() {
        let mut keys: Vec<CacheKey> = option_cases().into_iter().map(|(_, key)| key).collect();
        keys.extend(input_cases().into_iter().map(|(_, key)| key));
        keys.push(base_key());
        let distinct: std::collections::HashSet<CacheKey> = keys.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            keys.len(),
            "no two distinct inputs may collide"
        );
    }

    #[test]
    fn schema_version_changes_the_key() {
        let options = base_options();
        let current = CacheKey::with_schema(
            CACHE_SCHEMA_VERSION,
            SOURCE,
            Some(HEADER),
            NVRTC,
            &options,
            CAPABILITY,
        );
        let bumped = CacheKey::with_schema(
            CACHE_SCHEMA_VERSION + 1,
            SOURCE,
            Some(HEADER),
            NVRTC,
            &options,
            CAPABILITY,
        );
        assert_ne!(current, bumped, "a schema bump must invalidate every entry");
    }

    #[test]
    fn source_and_header_boundary_is_length_prefixed() {
        let options = base_options();
        let split_one = CacheKey::new("ab", Some(b"c"), NVRTC, &options, CAPABILITY);
        let split_two = CacheKey::new("a", Some(b"bc"), NVRTC, &options, CAPABILITY);
        assert_ne!(split_one, split_two, "field bytes must not run together");
    }

    #[test]
    fn encode_then_decode_round_trips_real_ptx() {
        let entry = encode_entry(&base_key(), PTX);
        assert_eq!(
            entry.len(),
            HEADER_LEN + PTX.len(),
            "header plus payload, nothing else"
        );
        let decoded = decode_entry(&base_key(), &entry).expect("a freshly encoded entry decodes");
        assert_eq!(decoded, PTX, "decoded payload equals the encoded PTX");
    }

    #[test]
    fn truncation_below_header_is_rejected() {
        let entry = encode_entry(&base_key(), PTX);
        let cut = &entry[..HEADER_LEN - 1];
        assert_eq!(
            decode_entry(&base_key(), cut),
            Err(CacheEntryError::Truncated {
                needed: HEADER_LEN,
                found: HEADER_LEN - 1
            })
        );
    }

    #[test]
    fn truncated_payload_is_rejected() {
        let entry = encode_entry(&base_key(), PTX);
        let cut = &entry[..entry.len() - 9];
        assert_eq!(
            decode_entry(&base_key(), cut),
            Err(CacheEntryError::LengthMismatch {
                declared: PTX.len() as u64,
                found: PTX.len() - 9,
            })
        );
    }

    #[test]
    fn flipped_payload_byte_is_rejected() {
        let mut entry = encode_entry(&base_key(), PTX);
        let index = HEADER_LEN + PTX.find("add_one").expect("ptx names the kernel");
        entry[index] ^= 0x01;
        assert_eq!(decode_entry(&base_key(), &entry), Err(CacheEntryError::HashMismatch));
    }

    #[test]
    fn flipped_header_byte_is_rejected_at_every_offset() {
        let entry = encode_entry(&base_key(), PTX);
        let mut rejected = 0;
        for offset in 0..HEADER_LEN {
            let mut corrupt = entry.clone();
            corrupt[offset] ^= 0x80;
            assert!(
                decode_entry(&base_key(), &corrupt).is_err(),
                "flipped header byte {offset} must be rejected"
            );
            rejected += 1;
        }
        assert_eq!(rejected, HEADER_LEN, "every header byte was exercised");
    }

    #[test]
    fn flipped_magic_schema_and_length_map_to_their_variants() {
        let entry = encode_entry(&base_key(), PTX);
        let mut magic = entry.clone();
        magic[0] ^= 0x01;
        assert_eq!(decode_entry(&base_key(), &magic), Err(CacheEntryError::BadMagic));
        let mut schema = entry.clone();
        schema[4] ^= 0x01;
        assert_eq!(
            decode_entry(&base_key(), &schema),
            Err(CacheEntryError::SchemaMismatch {
                found: CACHE_SCHEMA_VERSION ^ 0x01,
                expected: CACHE_SCHEMA_VERSION,
            })
        );
        let mut length = entry;
        length[40] ^= 0x01;
        assert_eq!(
            decode_entry(&base_key(), &length),
            Err(CacheEntryError::LengthMismatch {
                declared: PTX.len() as u64 ^ 0x01,
                found: PTX.len(),
            })
        );
    }

    #[test]
    fn valid_looking_payload_with_wrong_hash_is_rejected() {
        let forged = PTX.replace("add_one", "add_two");
        assert_eq!(
            forged.len(),
            PTX.len(),
            "same-length swap keeps the header length valid"
        );
        let mut entry = encode_entry(&base_key(), &forged);
        let genuine = encode_entry(&base_key(), PTX);
        entry[48..HEADER_LEN].copy_from_slice(&genuine[48..HEADER_LEN]);
        assert_eq!(decode_entry(&base_key(), &entry), Err(CacheEntryError::HashMismatch));
    }

    #[test]
    fn non_utf8_payload_with_matching_hash_is_rejected() {
        let payload: &[u8] = &[0xff, 0xfe, 0xfd];
        let mut entry = Vec::new();
        entry.extend_from_slice(b"PXKC");
        entry.extend_from_slice(&CACHE_SCHEMA_VERSION.to_le_bytes());
        entry.extend_from_slice(&base_key().0);
        entry.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        entry.extend_from_slice(&Sha256::digest(payload));
        entry.extend_from_slice(payload);
        assert_eq!(decode_entry(&base_key(), &entry), Err(CacheEntryError::NotUtf8));
    }

    #[test]
    fn entry_stored_under_another_key_is_rejected_as_key_mismatch() {
        let other = CacheKey::new(
            "__global__ void add_two() {}",
            None,
            NVRTC,
            &base_options(),
            CAPABILITY,
        );
        let entry = encode_entry(&other, PTX);
        assert_eq!(
            decode_entry(&base_key(), &entry),
            Err(CacheEntryError::KeyMismatch),
            "a byte-valid entry copied under another key must not load"
        );
    }

    #[test]
    fn entry_decodes_under_the_key_it_was_stored_with() {
        let entry = encode_entry(&base_key(), PTX);
        assert_eq!(decode_entry(&base_key(), &entry), Ok(PTX));
    }

    #[test]
    fn schema_one_entry_is_rejected_as_schema_mismatch() {
        let mut entry = Vec::new();
        entry.extend_from_slice(b"PXKC");
        entry.extend_from_slice(&1_u32.to_le_bytes());
        entry.extend_from_slice(&(PTX.len() as u64).to_le_bytes());
        entry.extend_from_slice(&Sha256::digest(PTX));
        entry.extend_from_slice(PTX.as_bytes());
        assert_eq!(
            decode_entry(&base_key(), &entry),
            Err(CacheEntryError::SchemaMismatch {
                found: 1,
                expected: CACHE_SCHEMA_VERSION,
            })
        );
    }
}
