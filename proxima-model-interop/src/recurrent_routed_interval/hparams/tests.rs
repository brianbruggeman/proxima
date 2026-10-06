use alloc::string::ToString;
use alloc::vec;

use proxima_gguf::value::{MetadataArray, MetadataValue as Value};

use crate::memory_fit::{MemoryBudget, WeightClassBytes};
use crate::test_support::parsed_header;

/// `qwen3.6:35b-a3b`'s KV-relevant header (`ollama /api/show`,
/// 2026-09-29): 40 blocks, `head_count_kv` of 2 on every fourth layer
/// and 0 on the 30 GDN layers, key length 256.
#[test]
fn memory_budget_qwen35moe() {
    let parsed = parsed_header(vec![
        (
            "general.architecture",
            Value::String("qwen35moe".to_string()),
        ),
        ("qwen35moe.block_count", Value::U32(40)),
        (
            "qwen35moe.attention.head_count_kv",
            Value::Array(MetadataArray::U32(
                (0..40u32)
                    .map(|layer| if (layer + 1) % 4 == 0 { 2 } else { 0 })
                    .collect(),
            )),
        ),
        ("qwen35moe.attention.key_length", Value::U32(256)),
    ]);

    let layers = crate::lowering::kv_layers(&parsed)
        .expect("the header carries every key kv_layers reads");
    let budget = MemoryBudget::derive(WeightClassBytes::default(), &layers, 262_144, 0, 0);
    let every_layer = vec![(2u32, 256u32, None); 40];
    let today = MemoryBudget::derive(WeightClassBytes::default(), &every_layer, 262_144, 0, 0);

    assert_eq!(layers.len(), 10, "only layers with head_count_kv != 0");
    assert_eq!(budget.kv_cache_bytes, 10_737_418_240);
    assert_eq!(today.kv_cache_bytes, 42_949_672_960);
}
