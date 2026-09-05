//! Build-time sizing config for this crate's own driver policy --
//! mirrors `proxima-tensor/src/sized.rs`'s pattern (guiding-principle 12):
//! `proxima-model-interop-runtime.toml` holds the number, `build.rs`'s
//! `emit_kv_sizing_consts` reads it (with a `PROXIMA_MODEL_INTEROP_<SECTION>_
//! <KEY>` env override, `resolve_int`, mirroring `proxima-tensor/build.rs`'s
//! own function of the same name), and this module includes the generated
//! file. `kv-capacity-bucket`-only: the whole module is compiled out when
//! that feature is off, since [`KV_BUCKET_TOKENS`] has no meaning without
//! `generate::kv_extent`'s plan-cache-key rounding.
//!
//! This constant used to live in `proxima-tensor::sized` (a Metal
//! plan-cache-key rounding policy for `generate::kv_extent`, this crate's
//! only consumer) -- moved here because the IR crate should not carry a
//! driver's cache policy. `proxima-tensor` keeps the `kv-capacity-bucket`
//! feature name (its own `spec.rs`'s `cpu_mask_zero_ulp` test still uses it
//! to prove the general bucketing mechanism 0-ULP-safe for any bucket size),
//! but no longer holds the specific bucket-size value.

#[cfg(feature = "kv-capacity-bucket")]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/proxima_model_interop_sized.rs"));
}

/// Plan-cache key bucket for [`crate::generate::kv_extent`]'s placed-KV
/// Metal decode path -- see `proxima-model-interop-runtime.toml`'s `[kv]`
/// section for the mechanism and the padding-vs-orchestration trade it
/// tunes (CARD 6.3). `kv-capacity-bucket`-only -- the constant has no
/// meaning without that feature's plan-cache-key rounding.
#[cfg(feature = "kv-capacity-bucket")]
pub const KV_BUCKET_TOKENS: usize = generated::KV_BUCKET_TOKENS;

/// Asserts [`KV_BUCKET_TOKENS`] still equals the value on record
/// (`proxima-model-interop-runtime.toml`'s own doc comment) -- catches a
/// TOML/doc drift the type system cannot: nothing stops someone editing
/// one file and not the other.
#[cfg(test)]
#[cfg(feature = "kv-capacity-bucket")]
mod tests {
    use super::*;

    #[test]
    fn kv_bucket_tokens_matches_the_runtime_toml() {
        assert_eq!(KV_BUCKET_TOKENS, 32);
    }
}
