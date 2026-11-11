//! Renders a checkpoint's own Jinja chat template.
//!
//! The template is data the checkpoint ships: the GGUF key
//! `tokenizer.chat_template` (`gguf.rs`'s `CHAT_TEMPLATE_KEY`). Renderers that
//! hard-code one format (`ChatML`) mis-tokenize every model whose real turn
//! markers differ, e.g. flint's single special tokens `<|user|>`/`<|assistant|>`.
//!
//! [`render_chat`] composes [`minijinja`] (the Rust Jinja incumbent) with
//! `minijinja-contrib`'s `pycompat` callback, so templates written for
//! Python's Jinja (`.items()`, `.startswith()`) run unchanged, and registers
//! `raise_exception`, which HuggingFace-style templates call to reject bad
//! role orders. It is std-only (feature `chat-template`): the `no_std` + alloc
//! tokenizer core never sees it. Compose the primitives directly instead if you
//! need tools, multimodal content, or non-string message fields.
//!
//! The output is the prompt text only. Whether a BOS id is prepended at
//! tokenization is the caller's decision (`Vocab::add_bos`); pass the
//! checkpoint's BOS/EOS strings so templates that print them agree with it.

use std::collections::BTreeMap;

use minijinja::{Environment, Error, ErrorKind, Value, context};
use minijinja_contrib::pycompat::unknown_method_callback;
use thiserror::Error;

const TEMPLATE_NAME: &str = "chat_template";

/// Why a chat template could not be rendered.
#[derive(Debug, Error)]
pub enum ChatTemplateError {
    /// The template failed to compile or to render: syntax error, undefined
    /// filter or function, or `raise_exception` called by the template itself.
    #[error("chat template failed: {0}")]
    Template(#[from] Error),
}

/// Renders `template` over `messages` (`(role, content)` pairs, as ragd sends them).
///
/// `bos_token` / `eos_token` are the checkpoint's own strings (llama passes the
/// same two variables). `add_generation_prompt` is the template variable of the
/// same name: true appends the assistant opener. `flags` are extra boolean
/// template variables, e.g. `("enable_thinking", true)` for gemma4, whose
/// template reads it; flint's reads none (`&[]`).
///
/// # Errors
///
/// [`ChatTemplateError::Template`] when the template does not compile or
/// render. Never falls back to another format.
#[must_use = "the rendered prompt or the failure must be handled"]
pub fn render_chat(
    template: &str,
    messages: &[(&str, &str)],
    add_generation_prompt: bool,
    bos_token: &str,
    eos_token: &str,
    flags: &[(&str, bool)],
) -> Result<String, ChatTemplateError> {
    let mut environment = Environment::new();
    environment.set_unknown_method_callback(unknown_method_callback);
    environment.add_function("raise_exception", raise_exception);
    environment.add_template(TEMPLATE_NAME, template)?;
    let rendered_messages: Vec<Value> = messages
        .iter()
        .map(|(role, content)| context! { role => *role, content => *content })
        .collect();
    let mut variables: BTreeMap<&str, Value> = flags
        .iter()
        .map(|(name, flag)| (*name, Value::from(*flag)))
        .collect();
    variables.insert("messages", Value::from(rendered_messages));
    variables.insert("add_generation_prompt", Value::from(add_generation_prompt));
    variables.insert("bos_token", Value::from(bos_token));
    variables.insert("eos_token", Value::from(eos_token));
    Ok(environment.get_template(TEMPLATE_NAME)?.render(variables)?)
}

fn raise_exception(message: String) -> Result<String, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, message))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    const MESSAGES: [(&str, &str); 2] = [
        ("system", "You are a helpful assistant."),
        ("user", "Reply with the single word: flint."),
    ];

    #[test]
    fn undefined_filter_is_an_error_not_a_default_string() {
        let outcome = render_chat(
            "{{ messages[0].content | no_such_filter }}",
            &MESSAGES,
            true,
            "",
            "",
            &[],
        );
        assert!(
            matches!(outcome, Err(ChatTemplateError::Template(_))),
            "got {outcome:?}"
        );
    }

    #[test]
    fn raise_exception_in_template_surfaces_its_message() {
        let outcome = render_chat(
            "{{ raise_exception('roles must alternate') }}",
            &MESSAGES,
            false,
            "",
            "",
            &[],
        );
        let message = outcome.expect_err("template raised").to_string();
        assert!(message.contains("roles must alternate"), "got {message}");
    }

    #[cfg(feature = "gguf")]
    mod real_checkpoints {
        use std::fs::File;
        use std::io::{Read, Seek, SeekFrom};
        use std::path::PathBuf;

        use proxima_gguf::MetadataValue;
        use serde_json::Value as Json;

        use super::*;

        const FLINT_ENV: &str = "PROXIMA_FLINT_GGUF";
        const FLINT_DEFAULT: &str =
            "/Users/brianbruggeman/models/llm/flint-gguf/flint-UD-Q4_K_XL.gguf";
        const FLINT_PROMPT: &str = "<|system|>You are a helpful assistant.<|user|>Reply with the single word: flint.<|assistant|>";
        const GEMMA4_BLOB: &str =
            "sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";
        const GEMMA4_FIXTURE: &str =
            include_str!("../tests/fixtures/llama-chat-template/gemma4_apply_template.json");

        // reads only the metadata region (growing prefix), never the tensors.
        fn template_and_specials(path: &PathBuf) -> (String, String, String) {
            let mut file = File::open(path).unwrap_or_else(|error| {
                panic!("gguf absent at {path:?} ({error}); set {FLINT_ENV} for flint")
            });
            let mut header = Vec::new();
            for capacity in [4usize << 20, 16 << 20, 64 << 20] {
                header.resize(capacity, 0);
                file.seek(SeekFrom::Start(0)).expect("seek to gguf start");
                let read = file.read(&mut header).expect("read gguf header region");
                header.truncate(read);
                if let Ok(parsed) = proxima_gguf::pipe::parse_complete(&header) {
                    let template = string_key(&parsed, "tokenizer.chat_template");
                    let bos = special(&parsed, "tokenizer.ggml.bos_token_id");
                    let eos = special(&parsed, "tokenizer.ggml.eos_token_id");
                    return (template, bos, eos);
                }
            }
            panic!("gguf metadata region did not fit in 64 MiB of {path:?}");
        }

        fn string_key(parsed: &proxima_gguf::ParsedGguf, key: &str) -> String {
            match parsed.metadata_value(key) {
                Some(MetadataValue::String(value)) => value.clone(),
                other => panic!("{key} missing or not a string: {other:?}"),
            }
        }

        // the token strings are looked up by id; empty when the key is absent.
        fn special(parsed: &proxima_gguf::ParsedGguf, id_key: &str) -> String {
            let id = match parsed.metadata_value(id_key) {
                Some(MetadataValue::U32(id)) => *id as usize,
                _ => return String::new(),
            };
            match parsed.metadata_value("tokenizer.ggml.tokens") {
                Some(MetadataValue::Array(array)) => token_at(array, id),
                _ => String::new(),
            }
        }

        fn token_at(array: &proxima_gguf::MetadataArray, id: usize) -> String {
            match array {
                proxima_gguf::MetadataArray::String(tokens) => {
                    tokens.get(id).cloned().unwrap_or_default()
                }
                _ => String::new(),
            }
        }

        fn flint_path() -> PathBuf {
            PathBuf::from(std::env::var(FLINT_ENV).unwrap_or_else(|_| FLINT_DEFAULT.to_string()))
        }

        fn gemma4_path() -> PathBuf {
            let home = std::env::var("HOME").expect("HOME is set");
            PathBuf::from(home)
                .join(".ollama/models/blobs")
                .join(GEMMA4_BLOB)
        }

        #[test]
        fn flint_real_template_matches_llama_server_b12110() {
            let (template, bos, eos) = template_and_specials(&flint_path());
            let rendered = render_chat(&template, &MESSAGES, true, &bos, &eos, &[])
                .expect("flint template renders");
            assert_eq!(rendered, FLINT_PROMPT);
        }

        #[test]
        fn gemma4_real_template_matches_llama_server_b11671() {
            let (template, _bos, eos) = template_and_specials(&gemma4_path());
            // llama-server's /apply-template passes an empty bos_token (BOS is added at tokenization)
            // and enable_thinking=true; the fixture was captured under those defaults.
            let rendered = render_chat(
                &template,
                &MESSAGES,
                true,
                "",
                &eos,
                &[("enable_thinking", true)],
            )
            .expect("gemma4 template renders");
            let expected: Json = serde_json::from_str(GEMMA4_FIXTURE).expect("fixture is json");
            let expected = expected["prompt"]
                .as_str()
                .expect("fixture has a prompt string");
            assert_eq!(rendered, expected);
        }
    }
}
