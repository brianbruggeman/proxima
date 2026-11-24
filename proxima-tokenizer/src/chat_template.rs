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
const SECONDS_PER_DAY: i64 = 86_400;
const SECONDS_PER_HOUR: i64 = 3_600;
const SECONDS_PER_MINUTE: i64 = 60;
const DAYS_PER_WEEK: i64 = 7;
const EPOCH_WEEKDAY: i64 = 4;
const MONTH_NAMES: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];
const MONTH_DAYS: [i64; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
const WEEKDAY_NAMES: [&str; 7] = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
];
const ABBREVIATION_LEN: usize = 3;

/// Why a chat template could not be rendered.
#[derive(Debug, Error)]
pub enum ChatTemplateError {
    /// The template failed to compile or to render: syntax error, undefined
    /// filter or function, or `raise_exception` called by the template itself.
    #[error("chat template failed: {0}")]
    Template(#[from] Error),
}

/// Renders `template` over `messages` (`(role, content)` pairs, as callers send them).
///
/// `now_unix_seconds` is the UTC instant `strftime_now` formats; the caller owns it
/// so every render in one session shows the same date.
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
    now_unix_seconds: i64,
) -> Result<String, ChatTemplateError> {
    let mut environment = Environment::new();
    environment.set_unknown_method_callback(unknown_method_callback);
    environment.add_function("raise_exception", raise_exception);
    environment.add_function("strftime_now", move |format: String| {
        strftime(&format, now_unix_seconds)
    });
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

/// Days since 1970-01-01 to a proleptic Gregorian `(year, month, day)`.
#[must_use]
pub fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

fn is_leap_year(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn day_of_year(year: i64, month: i64, day: i64) -> i64 {
    let before = usize::try_from(month - 1).unwrap_or(0);
    let leap_day = i64::from(month > 2 && is_leap_year(year));
    MONTH_DAYS.iter().take(before).sum::<i64>() + leap_day + day
}

fn table_entry<'a>(table: &[&'a str], index: i64) -> &'a str {
    usize::try_from(index)
        .ok()
        .and_then(|slot| table.get(slot))
        .copied()
        .unwrap_or_default()
}

fn abbreviation(name: &str) -> &str {
    name.get(..ABBREVIATION_LEN).unwrap_or(name)
}

fn strftime(format: &str, unix_seconds: i64) -> Result<String, Error> {
    let days = unix_seconds.div_euclid(SECONDS_PER_DAY);
    let in_day = unix_seconds.rem_euclid(SECONDS_PER_DAY);
    let (year, month, day) = civil_from_days(days);
    let month_name = table_entry(&MONTH_NAMES, month - 1);
    let weekday_name = table_entry(
        &WEEKDAY_NAMES,
        (days + EPOCH_WEEKDAY).rem_euclid(DAYS_PER_WEEK),
    );
    let mut out = String::new();
    let mut chars = format.chars();
    while let Some(character) = chars.next() {
        if character != '%' {
            out.push(character);
            continue;
        }
        let directive = chars.next().ok_or_else(|| unsupported_directive("%"))?;
        match directive {
            'Y' => out.push_str(&format!("{year:04}")),
            'm' => out.push_str(&format!("{month:02}")),
            'd' => out.push_str(&format!("{day:02}")),
            'B' => out.push_str(month_name),
            'b' => out.push_str(abbreviation(month_name)),
            'A' => out.push_str(weekday_name),
            'a' => out.push_str(abbreviation(weekday_name)),
            'H' => out.push_str(&format!("{:02}", in_day / SECONDS_PER_HOUR)),
            'M' => out.push_str(&format!(
                "{:02}",
                in_day % SECONDS_PER_HOUR / SECONDS_PER_MINUTE
            )),
            'S' => out.push_str(&format!("{:02}", in_day % SECONDS_PER_MINUTE)),
            'j' => out.push_str(&format!("{:03}", day_of_year(year, month, day))),
            '%' => out.push('%'),
            other => return Err(unsupported_directive(&format!("%{other}"))),
        }
    }
    Ok(out)
}

fn unsupported_directive(directive: &str) -> Error {
    Error::new(
        ErrorKind::InvalidOperation,
        format!("strftime_now: unsupported directive {directive}"),
    )
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
            0,
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
            0,
        );
        let message = outcome.expect_err("template raised").to_string();
        assert!(message.contains("roles must alternate"), "got {message}");
    }

    const FIXED_INSTANT: i64 = 1_795_442_709;
    const LEAP_INSTANT: i64 = 1_709_251_200;
    const GRANITE_DATE_SNIPPET: &str = "{%- set system_message = \"Knowledge Cutoff Date: April 2024.\nToday's Date: \" + strftime_now('%B %d, %Y') + \".\nYou are Granite, developed by IBM.\" %}{{ system_message }}";

    fn render_at(template: &str, instant: i64) -> Result<String, ChatTemplateError> {
        render_chat(template, &MESSAGES, true, "", "", &[], instant)
    }

    #[test]
    fn strftime_now_formats_the_caller_instant() {
        let rendered = render_at("{{ strftime_now('%d %b %Y') }}", FIXED_INSTANT).unwrap();
        assert_eq!(rendered, "23 Nov 2026");
    }

    #[test]
    fn strftime_now_supports_every_listed_directive() {
        let rendered = render_at(
            "{{ strftime_now('%Y-%m-%d %H:%M:%S %A %a %B %b %j %%') }}",
            FIXED_INSTANT,
        )
        .unwrap();
        assert_eq!(
            rendered,
            "2026-11-23 14:05:09 Monday Mon November Nov 327 %"
        );
    }

    #[test]
    fn strftime_now_counts_the_leap_day() {
        let rendered = render_at("{{ strftime_now('%j %A') }}", LEAP_INSTANT).unwrap();
        assert_eq!(rendered, "061 Friday");
    }

    #[test]
    fn strftime_now_unsupported_directive_names_it() {
        let outcome = render_at("{{ strftime_now('%Q') }}", FIXED_INSTANT);
        let message = outcome.expect_err("unsupported directive").to_string();
        assert!(message.contains("%Q"), "got {message}");
    }

    #[test]
    fn granite_date_line_renders() {
        let rendered = render_at(GRANITE_DATE_SNIPPET, FIXED_INSTANT).unwrap();
        assert_eq!(
            rendered,
            "Knowledge Cutoff Date: April 2024.\nToday's Date: November 23, 2026.\nYou are Granite, developed by IBM."
        );
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
            let rendered = render_chat(&template, &MESSAGES, true, &bos, &eos, &[], 0)
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
                0,
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
