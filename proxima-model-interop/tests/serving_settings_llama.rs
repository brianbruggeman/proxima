use proxima_model_interop::{ContextLength, ServingSettings};
use proxima_numeric::NumericFormat;

const CONFIG: &str = include_str!("../examples/llama_matched.toml");

#[test]
fn llama_matched_config_matches_the_incumbent_flags() {
    let settings: ServingSettings =
        conflaguration::from_toml_str(CONFIG).expect("the granite/llama TOML parses");
    let base = settings.as_serving_config(&[]);

    assert_eq!(base.batch_size, 2048, "llama's -b");
    assert_eq!(base.ubatch_size, 512, "llama's -ub");
    assert_eq!(base.kv_cache_key_quant, NumericFormat::Float16, "llama's -ctk");
    assert_eq!(base.kv_cache_value_quant, NumericFormat::Float16, "llama's -ctv");
    assert_eq!(base.context_length, ContextLength::Native, "the row supplies -c");
    assert_eq!(base.seed, 1, "the runner passes --seed 1");
    assert!(base.ignore_eos, "the request body sets ignore_eos");
    assert!(!base.flash_attention, "the admitted form is the decomposition");

    #[cfg(all(
        feature = "metal",
        feature = "metal-attn-variants",
        target_os = "macos"
    ))]
    assert_eq!(
        base.attention_variant
            .expect("flash_where_admitted is a variant")
            .decomposition,
        omega::AttentionDecomposition::FlashWhereAdmitted,
        "llama's -fa on"
    );
}

#[test]
fn llama_matched_ubatch_override_reaches_the_serving_config() {
    let settings: ServingSettings = conflaguration::from_toml_str("ubatch_size = 128")
        .expect("a one-key override parses");
    assert_eq!(settings.as_serving_config(&[]).ubatch_size, 128);
}
