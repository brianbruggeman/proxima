//! greedy decode through the pinned libllama, with llama-server `--ignore-eos` semantics.
//!
//! composes the bindgen `LlamaApi` calls (`llama_decode`, `llama_synchronize`,
//! `llama_get_logits_ith`, `llama_vocab_is_eog`) directly; there is no sampler object, because
//! greedy selection with an end-of-generation mask is one pass over the logits row.

use std::ffi::CString;
use std::path::Path;
use std::slice;
use std::time::Instant;

use crate::bindings::{
    ggml_type_GGML_TYPE_F16, llama_context, llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_ENABLED, llama_model,
};
use crate::{LlamaApi, LoadError};

/// every layer on Metal; larger than any model's layer count, which llama clamps.
pub const GPU_LAYERS: i32 = 99;

/// per step: the chosen ids, the wall ns of decode + synchronize that produced each id's logits
/// (step 0 is the prompt prefill), and the two best non-EOG `(id, logit)` pairs at that step.
pub type GreedyRun = (Vec<i32>, Vec<u64>, Vec<[(i32, f32); 2]>);

/// the two best non-EOG `(id, logit)` pairs of a logits row; `eog[id]` true removes `id`, which is
/// the `-inf` logit bias llama-server applies for `--ignore-eos` (common/common.cpp:1343-1355 at
/// ba3c05f71). ties keep the lowest id because the scan is ascending and comparison is strict.
/// an all-masked row yields id `-1` in both slots.
#[must_use]
pub fn masked_top2(logits: &[f32], eog: &[bool]) -> [(i32, f32); 2] {
    let mut best = [(-1, f32::NEG_INFINITY); 2];
    for ((id, &value), &masked) in (0_i32..).zip(logits).zip(eog) {
        if masked {
            continue;
        }
        if value > best[0].1 {
            best[1] = best[0];
            best[0] = (id, value);
        } else if value > best[1].1 {
            best[1] = (id, value);
        }
    }
    best
}

fn cstring(path: &Path) -> Result<CString, LoadError> {
    CString::new(path.as_os_str().as_encoded_bytes()).map_err(|_| LoadError::BadPath(path.to_path_buf()))
}

fn load_model(api: &LlamaApi, path: &Path, vocab_only: bool) -> Result<*mut llama_model, LoadError> {
    let c_path = cstring(path)?;
    // safety: the bindings were resolved from the sha-gated build, so the params layout matches
    let model = unsafe {
        let mut params = api.llama_model_default_params();
        params.n_gpu_layers = GPU_LAYERS;
        params.vocab_only = vocab_only;
        api.llama_model_load_from_file(c_path.as_ptr(), params)
    };
    if model.is_null() {
        return Err(LoadError::ModelLoad(path.to_path_buf()));
    }
    Ok(model)
}

/// tokenizes `text` with the model's own vocabulary: BOS added per the model's metadata and
/// control tokens like `<|turn>` parsed, which is the request shape llama-server's `/tokenize`
/// would give with `add_special` and `parse_special` true. loads vocabulary only.
///
/// # Errors
/// model load failure or a tokenizer error code.
pub fn tokenize(api: &LlamaApi, model_path: &Path, text: &str) -> Result<Vec<i32>, LoadError> {
    let model = load_model(api, model_path, true)?;
    let text_len = i32::try_from(text.len()).map_err(|_| LoadError::Tokenize(i32::MIN))?;
    let mut ids = vec![0_i32; text.len() + 8];
    // safety: `ids` is writable for the capacity passed; the vocab pointer lives as long as `model`
    let mut count = unsafe {
        let vocab = api.llama_model_get_vocab(model);
        api.llama_tokenize(vocab, text.as_ptr().cast(), text_len, ids.as_mut_ptr(), capacity(&ids), true, true)
    };
    if count < 0 {
        ids.resize(usize::try_from(-count).unwrap_or(0), 0);
        // safety: as above, with the capacity llama reported as needed
        count = unsafe {
            let vocab = api.llama_model_get_vocab(model);
            api.llama_tokenize(vocab, text.as_ptr().cast(), text_len, ids.as_mut_ptr(), capacity(&ids), true, true)
        };
    }
    // safety: `model` came from `load_model` and no pointer into it outlives this call
    unsafe { api.llama_model_free(model) };
    let kept = usize::try_from(count).map_err(|_| LoadError::Tokenize(count))?;
    ids.truncate(kept);
    Ok(ids)
}

fn capacity(ids: &[i32]) -> i32 {
    i32::try_from(ids.len()).unwrap_or(i32::MAX)
}

/// loads the model on Metal, builds an F16-K / F16-V flash-attention context of `n_ctx`, decodes
/// `prompt` as one batch and then one token per step, choosing each token by
/// [`masked_top2`] over the last logits row, for `n_predict` ids.
///
/// # Errors
/// load, context, decode or empty-candidate failure; handles are freed on every path.
pub fn greedy_decode(
    api: &LlamaApi,
    model_path: &Path,
    prompt: &[i32],
    n_ctx: u32,
    n_predict: usize,
) -> Result<GreedyRun, LoadError> {
    // safety: backend init/free bracket every handle created below
    unsafe { api.llama_backend_init() };
    let outcome = load_model(api, model_path, false).and_then(|model| {
        let run = run_in_context(api, model, prompt, n_ctx, n_predict);
        // safety: `model` is live, its context was freed inside `run_in_context`
        unsafe { api.llama_model_free(model) };
        run
    });
    // safety: nothing created after init is still live
    unsafe { api.llama_backend_free() };
    outcome
}

fn run_in_context(
    api: &LlamaApi,
    model: *mut llama_model,
    prompt: &[i32],
    n_ctx: u32,
    n_predict: usize,
) -> Result<GreedyRun, LoadError> {
    // safety: layouts match the gated build; `model` is live for the whole call
    let context = unsafe {
        let mut params = api.llama_context_default_params();
        params.n_ctx = n_ctx;
        params.n_batch = params.n_batch.max(u32::try_from(prompt.len()).unwrap_or(u32::MAX));
        params.type_k = ggml_type_GGML_TYPE_F16;
        params.type_v = ggml_type_GGML_TYPE_F16;
        params.flash_attn_type = llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_ENABLED;
        api.llama_init_from_model(model, params)
    };
    if context.is_null() {
        return Err(LoadError::ContextInit);
    }
    let run = decode_steps(api, model, context, prompt, n_predict);
    // safety: `context` is live and nothing borrows it past this point
    unsafe { api.llama_free(context) };
    run
}

fn decode_steps(
    api: &LlamaApi,
    model: *mut llama_model,
    context: *mut llama_context,
    prompt: &[i32],
    n_predict: usize,
) -> Result<GreedyRun, LoadError> {
    // safety: `model` is live
    let (vocab, n_vocab) = unsafe {
        let vocab = api.llama_model_get_vocab(model);
        (vocab, api.llama_vocab_n_tokens(vocab))
    };
    // safety: `vocab` is live; ids below `n_vocab` are valid arguments
    let eog: Vec<bool> = (0..n_vocab).map(|id| unsafe { api.llama_vocab_is_eog(vocab, id) }).collect();
    let (mut ids, mut step_ns, mut top2) = (Vec::new(), Vec::new(), Vec::new());
    let mut pending = prompt.to_vec();
    for step in 0..n_predict {
        let started = Instant::now();
        // safety: `pending` outlives the call; llama copies the tokens into its own batch
        let code = unsafe {
            let batch = api.llama_batch_get_one(pending.as_mut_ptr(), capacity(&pending));
            let code = api.llama_decode(context, batch);
            api.llama_synchronize(context);
            code
        };
        step_ns.push(u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX));
        if code != 0 {
            return Err(LoadError::Decode { step, code });
        }
        // safety: after a successful decode, row -1 is the last output row, `n_vocab` floats long
        let row = unsafe {
            let ptr = api.llama_get_logits_ith(context, -1);
            if ptr.is_null() {
                return Err(LoadError::NoLogits { step });
            }
            slice::from_raw_parts(ptr, usize::try_from(n_vocab).unwrap_or(0))
        };
        let best = masked_top2(row, &eog);
        if best[0].0 < 0 {
            return Err(LoadError::NoLogits { step });
        }
        pending.clear();
        pending.push(best[0].0);
        ids.push(best[0].0);
        top2.push(best);
    }
    Ok((ids, step_ns, top2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PINNED_BUILD_NUMBER, lib_dir_from_env, load};

    const GEMMA4_EOG_IDS: [usize; 2] = [1, 106];
    const MODEL_BLOB: &str =
        "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";
    const SHORT_PROMPT: &str =
        "<|turn>user\nWhich of these is smaller in size: a hippopotamus or a large office building?<turn|>\n<|turn>model\n";

    fn row_with_eog_on_top() -> (Vec<f32>, Vec<bool>) {
        let mut logits = vec![-4.0_f32; 300];
        logits[7] = 11.5;
        logits[42] = 11.0;
        logits[106] = 19.25;
        let mut eog = vec![false; 300];
        for id in GEMMA4_EOG_IDS {
            eog[id] = true;
        }
        (logits, eog)
    }

    #[test]
    fn masked_argmax_skips_an_eog_id_that_holds_the_global_max() {
        let (logits, eog) = row_with_eog_on_top();
        let best = masked_top2(&logits, &eog);
        assert_eq!(best[0], (7, 11.5));
        assert_eq!(best[1], (42, 11.0));
    }

    #[test]
    fn unmasked_control_picks_the_eog_id() {
        let (logits, _) = row_with_eog_on_top();
        let best = masked_top2(&logits, &vec![false; logits.len()]);
        assert_eq!(best[0], (106, 19.25));
    }

    #[test]
    fn ties_go_to_the_lowest_id() {
        let mut logits = vec![0.0_f32; 16];
        logits[9] = 3.0;
        logits[4] = 3.0;
        logits[12] = 3.0;
        let best = masked_top2(&logits, &[false; 16]);
        assert_eq!((best[0].0, best[1].0), (4, 9));
    }

    #[test]
    fn all_masked_row_has_no_candidate() {
        let best = masked_top2(&[1.0, 2.0], &[true, true]);
        assert_eq!(best[0].0, -1);
    }

    #[test]
    fn real_decode_never_emits_an_eog_id_and_first_step_is_the_prefill() {
        let lib = load(&lib_dir_from_env()).expect("pinned install must load");
        assert_eq!(lib.report.build_number, PINNED_BUILD_NUMBER);
        let model_path = Path::new(MODEL_BLOB);
        let prompt = tokenize(&lib.api, model_path, SHORT_PROMPT).expect("tokenize the scoreboard short prompt");
        assert!(prompt.len() > 10, "prompt ids: {prompt:?}");
        let (ids, step_ns, top2) = greedy_decode(&lib.api, model_path, &prompt, 256, 8).expect("greedy decode");
        assert_eq!((ids.len(), step_ns.len(), top2.len()), (8, 8, 8));
        for id in &ids {
            assert!(!GEMMA4_EOG_IDS.contains(&usize::try_from(*id).expect("non-negative id")), "{ids:?}");
        }
        assert!(top2.iter().all(|pair| pair[0].1 >= pair[1].1), "{top2:?}");
    }
}
