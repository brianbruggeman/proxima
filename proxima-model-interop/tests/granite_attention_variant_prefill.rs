#![cfg(all(
    feature = "metal-attn-variants",
    feature = "instrument",
    target_os = "macos"
))]

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use omega::{
    AttentionKvReuse, AttentionVariant, CapturedDispatch, set_capture_step,
    take_captured_dispatches,
};
use proxima_gguf::parse_complete;
use proxima_gguf::types::GgmlType;
use proxima_model_interop::{GPU_LAYERS_ALL, LoadedModel, PromptCacheConfig, ServingConfig};

const GRANITE_MOE_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80";
const GRANITE_MOE_ENV: &str = "PROXIMA_ARCH_GRANITE_MOE_GGUF";
const PROMPT_TOKENS: usize = 971;
const PASSAGE: &str = "To Sherlock Holmes she is always THE woman. I have seldom heard him mention her under any other name. In his eyes she eclipses and predominates the whole of her sex. It was not that he felt any emotion akin to love for Irene Adler. All emotions, and that one particularly, were abhorrent to his cold, precise but admirably balanced mind. He was, I take it, the most perfect reasoning and observing machine that the world has seen, but as a lover he would have placed himself in a false position. He never spoke of the softer passions, save with a gibe and a sneer. They were admirable things for the observer—excellent for drawing the veil from men's motives and actions. But for the trained reasoner to admit such intrusions into his own delicate and finely adjusted temperament was to introduce a distracting factor which might throw a doubt upon all his mental results.\n";

#[derive(Debug, Clone, PartialEq, Eq)]
struct DispatchIdentity {
    node: u32,
    extents: Vec<u64>,
    entry: String,
    source_sha: String,
    grid: omega::msl::GridSpec,
}

fn identity(dispatch: &CapturedDispatch) -> DispatchIdentity {
    DispatchIdentity {
        node: dispatch.node,
        extents: dispatch.extents.clone(),
        entry: dispatch.entry.clone(),
        source_sha: dispatch.msl_sha256.clone(),
        grid: dispatch.grid,
    }
}

fn assert_selected_dispatch_pair(
    legacy: &[DispatchIdentity],
    selected: &[DispatchIdentity],
) -> Result<usize, String> {
    let mut matched = 0;
    for legacy_dispatch in legacy {
        let Some(selected_dispatch) = selected.iter().find(|dispatch| {
            dispatch.node == legacy_dispatch.node && dispatch.extents == legacy_dispatch.extents
        }) else {
            continue;
        };
        if legacy_dispatch.extents.len() == 4
            && legacy_dispatch.extents[0] > 1
            && legacy_dispatch.extents[1..] == [8, 2, 64]
        {
            if legacy_dispatch.entry == selected_dispatch.entry {
                return Err(format!(
                    "node {} extents {:?} kept legacy entry {}",
                    legacy_dispatch.node, legacy_dispatch.extents, legacy_dispatch.entry
                ));
            }
            if legacy_dispatch.source_sha == selected_dispatch.source_sha {
                return Err(format!(
                    "node {} extents {:?} kept legacy source SHA {}",
                    legacy_dispatch.node, legacy_dispatch.extents, legacy_dispatch.source_sha
                ));
            }
            if selected_dispatch.grid.threads == 0 {
                return Err(format!(
                    "node {} selected an empty dispatch grid",
                    selected_dispatch.node
                ));
            }
            println!(
                "card_23 dispatch node={} extents={:?} legacy_entry={} selected_entry={} legacy_source_sha={} selected_source_sha={} grid={:?}",
                selected_dispatch.node,
                selected_dispatch.extents,
                legacy_dispatch.entry,
                selected_dispatch.entry,
                legacy_dispatch.source_sha,
                selected_dispatch.source_sha,
                selected_dispatch.grid
            );
            matched += 1;
        }
    }
    if matched == 0 {
        return Err("no matching multi-row Granite cached-attention dispatch pair".to_string());
    }
    Ok(matched)
}

fn serving_config(attention_variant: Option<AttentionVariant>) -> ServingConfig<'static> {
    ServingConfig {
        attention_variant,
        gpu_layers: GPU_LAYERS_ALL,
        kv_cache_key_quant: GgmlType::F32,
        kv_cache_value_quant: GgmlType::F32,
        flash_attention: false,
        reasoning_budget: 0,
        prompt_cache: PromptCacheConfig::off(),
        ubatch_size: 0,
        ..ServingConfig::default()
    }
}

fn granite_checkpoint_path() -> String {
    let path = std::env::var(GRANITE_MOE_ENV).unwrap_or_else(|_| GRANITE_MOE_PATH.to_string());
    assert!(
        Path::new(&path).exists(),
        "Granite checkpoint is missing at {path}"
    );
    path
}

fn prompt_of_971_tokens(vocab: &proxima_tokenizer::Vocab) -> (String, usize) {
    let corpus = PASSAGE.repeat(PROMPT_TOKENS.div_ceil(40));
    let mut prompt = String::new();
    for word in corpus.split_inclusive(char::is_whitespace) {
        prompt.push_str(word);
        let token_count = proxima_tokenizer::encode(&prompt, vocab)
            .expect("tokenize prompt with the Granite vocab")
            .len();
        if token_count >= PROMPT_TOKENS {
            return (prompt, token_count);
        }
    }
    panic!("repeated Sherlock passage never reached {PROMPT_TOKENS} tokens");
}

fn capture_prompt_dispatches(
    model: &LoadedModel<'_>,
    prompt: &str,
    variant: Option<AttentionVariant>,
) -> (Vec<DispatchIdentity>, Vec<u32>) {
    let _ = take_captured_dispatches();
    set_capture_step(0);
    let (token_ids, _text, _stopped) = model
        .generate_with_serving_config(prompt, 1, serving_config(variant))
        .expect("Granite prefill and one-token forward run");
    let records = take_captured_dispatches();
    let identities = records
        .iter()
        .filter(|record| record.kind_name == "cached_attention")
        .map(identity)
        .collect();
    (identities, token_ids)
}

#[proxima::test]
async fn card_23_granite_prefill_uses_selected_attention_dispatch() {
    assert!(
        std::env::var_os("PROXIMA_CAPTURE_LIVE").is_some(),
        "run with PROXIMA_CAPTURE_LIVE=1"
    );
    let path = granite_checkpoint_path();
    let file = File::open(&path).expect("open the real Granite checkpoint");
    // SAFETY: this test only reads the checkpoint and does not mutate it.
    let mapping = unsafe { Mmap::map(&file) }.expect("mmap the Granite checkpoint");
    let parsed = parse_complete(&mapping).expect("parse the Granite checkpoint");
    let vocab = proxima_tokenizer::gguf::vocab_from_metadata(&parsed)
        .expect("build the Granite checkpoint vocab");
    let (prompt, prompt_tokens) = prompt_of_971_tokens(&vocab);
    assert!(prompt_tokens >= PROMPT_TOKENS);
    let model = LoadedModel::load(&parsed, &mapping).expect("bind the Granite checkpoint");
    let (legacy_dispatches, legacy_tokens) = capture_prompt_dispatches(&model, &prompt, None);
    let mut selected = AttentionVariant::default();
    selected.kv_reuse = AttentionKvReuse::SharedK;
    let (selected_dispatches, selected_tokens) =
        capture_prompt_dispatches(&model, &prompt, Some(selected));
    let matched = assert_selected_dispatch_pair(&legacy_dispatches, &selected_dispatches)
        .expect("the selected row-tiled Granite dispatch differs from legacy");
    assert!(
        !legacy_tokens.is_empty(),
        "legacy forward produced no token ids"
    );
    assert_eq!(
        legacy_tokens, selected_tokens,
        "selected prefill changed generated token ids"
    );
    println!(
        "card_23 matched_multirow_dispatches={matched} generated_token_ids={} legacy_dispatches={} selected_dispatches={}",
        legacy_tokens.len(),
        legacy_dispatches.len(),
        selected_dispatches.len()
    );
}

#[proxima::test]
async fn card_23_legacy_dispatch_fails_selected_identity_control() {
    let legacy = DispatchIdentity {
        node: 12,
        extents: vec![1000, 8, 2, 64],
        entry: "legacy_entry".to_string(),
        source_sha: "legacy_sha".to_string(),
        grid: omega::msl::GridSpec {
            threads: 1,
            threadgroup_width: None,
            depth: 1,
            grid2d: None,
        },
    };
    let error = assert_selected_dispatch_pair(&[legacy.clone()], &[legacy])
        .expect_err("the control must reject identical legacy and selected identities");
    assert!(error.contains("kept legacy entry"));
}
