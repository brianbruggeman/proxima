use super::*;
use proxima_primitives::pipe::sans_io::Outcome;
use proxima_safetensors::SafetensorsParser;

fn pinned_config_and_manifest() -> (HfConfig, Manifest) {
    let config_bytes = include_bytes!(
        "../../../proxima-tensor/specs/small-model-architecture-matrix/fixtures/semantic-pair-configs/lfm_text.json"
    );
    let header_bytes = include_bytes!(
        "../../../proxima-tensor/specs/small-model-architecture-matrix/fixtures/lfm_text.safetensors.header"
    );
    let config = crate::parse_hf_config(config_bytes).expect("pinned LFM config parses");
    let mut parser = SafetensorsParser::new();
    parser.feed(header_bytes);
    let manifest = match parser.poll().expect("pinned tensor header parses") {
        Outcome::Event(manifest) => manifest.clone(),
        Outcome::NeedMore => panic!("complete pinned header must produce a manifest"),
    };
    (config, manifest)
}

fn compact_layer_payload(manifest: &mut Manifest) -> Vec<u8> {
    let mut file_bytes = Vec::new();
    for entry in &mut manifest.tensors {
        if !(entry.name.starts_with("model.layers.0.") || entry.name.starts_with("model.layers.2."))
        {
            continue;
        }
        let start = u64::try_from(file_bytes.len()).expect("fixture offset fits u64");
        let byte_len = usize::try_from(entry.byte_len()).expect("fixture tensor fits usize");
        file_bytes.resize(file_bytes.len() + byte_len, 0xA5);
        let end = u64::try_from(file_bytes.len()).expect("fixture end fits u64");
        entry.data_offsets = (start, end);
    }
    file_bytes
}

fn assert_borrowed_view(
    view: LfmTensorView<'_, '_>,
    manifest: &Manifest,
    file_bytes: &[u8],
    name: &str,
    shape: &[u64],
) {
    let entry = manifest
        .tensor(name)
        .expect("view tensor exists in manifest");
    let start = usize::try_from(entry.data_offsets.0).expect("view start fits usize");
    let end = usize::try_from(entry.data_offsets.1).expect("view end fits usize");
    assert_eq!(view.name, name);
    assert_eq!(view.shape, shape);
    assert_eq!(view.dtype, DType::BFloat16);
    assert_eq!(view.bytes.len(), end - start);
    assert_eq!(view.bytes.as_ptr(), file_bytes[start..end].as_ptr());
}

#[test]
fn architecture_matrix_lfm_layer_bind() {
    let (config, mut manifest) = pinned_config_and_manifest();
    let file_bytes = compact_layer_payload(&mut manifest);
    let architecture = architecture_from_hf_config(&config)
        .expect("pinned LFM config derives its effective architecture");
    assert_eq!(architecture.feed_forward, 8192);
    assert!(architecture.tied_embeddings);
    let conv = bind_lfm_layer(&config, &manifest, &file_bytes, 0, 0)
        .expect("pinned LFM convolution layer binds");
    let attention = bind_lfm_layer(&config, &manifest, &file_bytes, 0, 2)
        .expect("pinned LFM attention layer binds");

    let conv_prefix = "model.layers.0.";
    for (view, suffix, shape) in [
        (
            conv.common.operator_norm,
            "operator_norm.weight",
            &[2048][..],
        ),
        (conv.common.ffn_norm, "ffn_norm.weight", &[2048][..]),
        (
            conv.common.ffn_w1,
            "feed_forward.w1.weight",
            &[8192, 2048][..],
        ),
        (
            conv.common.ffn_w2,
            "feed_forward.w2.weight",
            &[2048, 8192][..],
        ),
        (
            conv.common.ffn_w3,
            "feed_forward.w3.weight",
            &[8192, 2048][..],
        ),
    ] {
        assert_borrowed_view(
            view,
            &manifest,
            &file_bytes,
            &format!("{conv_prefix}{suffix}"),
            shape,
        );
    }
    let LfmMixerWeights::ShortConv {
        conv_weight,
        in_proj,
        out_proj,
    } = conv.mixer
    else {
        panic!("layer zero is the pinned short-convolution layer");
    };
    assert_borrowed_view(
        conv_weight,
        &manifest,
        &file_bytes,
        "model.layers.0.conv.conv.weight",
        &[2048, 1, 3],
    );
    assert_borrowed_view(
        in_proj,
        &manifest,
        &file_bytes,
        "model.layers.0.conv.in_proj.weight",
        &[6144, 2048],
    );
    assert_borrowed_view(
        out_proj,
        &manifest,
        &file_bytes,
        "model.layers.0.conv.out_proj.weight",
        &[2048, 2048],
    );

    let LfmMixerWeights::Attention {
        q_proj,
        k_proj,
        v_proj,
        out_proj,
        q_layernorm,
        k_layernorm,
    } = attention.mixer
    else {
        panic!("layer two is the pinned attention layer");
    };
    for (view, name, shape) in [
        (
            attention.common.operator_norm,
            "model.layers.2.operator_norm.weight",
            &[2048][..],
        ),
        (
            attention.common.ffn_norm,
            "model.layers.2.ffn_norm.weight",
            &[2048][..],
        ),
        (
            attention.common.ffn_w1,
            "model.layers.2.feed_forward.w1.weight",
            &[8192, 2048][..],
        ),
        (
            attention.common.ffn_w2,
            "model.layers.2.feed_forward.w2.weight",
            &[2048, 8192][..],
        ),
        (
            attention.common.ffn_w3,
            "model.layers.2.feed_forward.w3.weight",
            &[8192, 2048][..],
        ),
        (
            q_proj,
            "model.layers.2.self_attn.q_proj.weight",
            &[2048, 2048][..],
        ),
        (
            k_proj,
            "model.layers.2.self_attn.k_proj.weight",
            &[512, 2048][..],
        ),
        (
            v_proj,
            "model.layers.2.self_attn.v_proj.weight",
            &[512, 2048][..],
        ),
        (
            out_proj,
            "model.layers.2.self_attn.out_proj.weight",
            &[2048, 2048][..],
        ),
        (
            q_layernorm,
            "model.layers.2.self_attn.q_layernorm.weight",
            &[64][..],
        ),
        (
            k_layernorm,
            "model.layers.2.self_attn.k_layernorm.weight",
            &[64][..],
        ),
    ] {
        assert_borrowed_view(view, &manifest, &file_bytes, name, shape);
    }

    let mut wrong_conv_shape = manifest.clone();
    wrong_conv_shape
        .tensors
        .iter_mut()
        .find(|entry| entry.name == "model.layers.0.conv.conv.weight")
        .expect("pinned convolution weight exists")
        .shape[2] = 2;
    assert!(bind_lfm_layer(&config, &wrong_conv_shape, &file_bytes, 0, 0).is_err());

    let mut wrong_attention_shape = manifest.clone();
    wrong_attention_shape
        .tensors
        .iter_mut()
        .find(|entry| entry.name == "model.layers.2.self_attn.q_proj.weight")
        .expect("pinned query projection exists")
        .shape[0] = 2047;
    assert!(bind_lfm_layer(&config, &wrong_attention_shape, &file_bytes, 0, 2).is_err());

    println!("attention_layers=1 conv_layers=1 borrowed_views=19 bad_shapes_rejected=2");
}
