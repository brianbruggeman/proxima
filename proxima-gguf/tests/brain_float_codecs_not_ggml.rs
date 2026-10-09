use proxima_gguf::GgmlType;

#[test]
fn brain_float_codecs_do_not_change_ggml_wire_types() {
    assert_eq!(GgmlType::from_wire(29), Some(GgmlType::Iq1M));
    assert_eq!(GgmlType::from_wire(30), Some(GgmlType::Bf16));
}
