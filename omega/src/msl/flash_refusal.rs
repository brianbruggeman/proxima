/// Why the flash decomposition cannot serve a cached-attention op: one variant per admission
/// condition of [`flash_admission`](super::cached_attention_flash_tiled::flash_admission), so the
/// decision returns its own reason instead of a bool a caller must re-derive.
///
/// A caller that selected `AttentionDecomposition::Flash` and is refused can log
/// [`FlashRefusal::reason`] and re-select `Legacy`; the validator surfaces the same reason as
/// `EmitError::CachedAttentionVariantAxisNotSupported { axis: "decomposition", .. }`. Head
/// dimension 512 is a `ThreadgroupBytes` refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum FlashRefusal {
    /// The op is not a cached attention at all.
    NotCachedAttention,
    /// The op carries no attention sink.
    NoAttentionSink,
    /// The op does not have eight query heads per kv head.
    QueryGroups,
    /// The head width is not the one the body's lane map assumes.
    HeadDim,
    /// The new keys are not causally masked to their own row.
    NewKeysNotCausal,
    /// The query rows and the new key rows differ.
    QueryRowsDifferFromNewKeys,
    /// The cached key extent is not a whole number of 8-key tiles.
    CachedRowsNotWholeTiles,
    /// The Q stage, output and score tiles together exceed the threadgroup memory ceiling.
    ThreadgroupBytes,
}

impl FlashRefusal {
    /// A static, lowercase reason fit for an error value or a log line.
    #[must_use]
    pub(super) const fn reason(self) -> &'static str {
        match self {
            Self::NotCachedAttention => "flash serves cached attention ops only",
            Self::NoAttentionSink => "flash needs an attention sink",
            Self::QueryGroups => "flash needs 8 query groups per kv head",
            Self::HeadDim => "flash needs head dim 128, 256 or 512",
            Self::NewKeysNotCausal => "flash needs causally masked new keys",
            Self::QueryRowsDifferFromNewKeys => "flash needs as many query rows as new key rows",
            Self::CachedRowsNotWholeTiles => "flash needs a cached key extent that is a whole number of 8-key tiles",
            Self::ThreadgroupBytes => "flash needs its staged Q, output and score tiles inside the threadgroup memory ceiling",
        }
    }
}
