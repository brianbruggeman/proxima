//! End-of-generation (EOG) token set, inferred from the checkpoint alone.
//!
//! Order, strongest evidence first, none of it a built-in token list:
//! 1. GGUF keys: `tokenizer.ggml.eos_token_id`, `eot_token_id`, `eom_token_id`.
//! 2. The checkpoint's own chat template: the literals it emits at the very
//!    tail of its message loop close an assistant turn (or hand control to a
//!    tool), so every such literal that is exactly one vocab token is an EOG
//!    id ([`tail_literals`]).
//! 3. Config: `TokenizerConfig::eog_add` / `eog_remove` (token texts) replace
//!    the inference, add or remove.
//!
//! llama.cpp derives its set from a curated per-family text list
//! (`llama-vocab.cpp:3030-3057`); this module deliberately does not copy it.
//! Where inference cannot reach an id, the caller emits a warn event naming
//! the config key, see `gguf::vocab_from_metadata`.
//!
//! Composes [`crate::vocab::Vocab`]; read the result with
//! [`crate::vocab::Vocab::eog_token_ids`] / [`crate::vocab::Vocab::is_eog`].

use alloc::collections::BTreeSet;
use alloc::string::String;
use alloc::vec::Vec;

use crate::vocab::Vocab;

/// What the derivation saw, for the caller's diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EogEvidence {
    /// Ids the chat template resolved to (rule 2): its tail literals, or, when
    /// it closes no turn, the special tokens opening the turns that answer an
    /// assistant turn ([`opening_marker_ids`]).
    pub template_ids: Vec<u32>,
    /// `eog_add`/`eog_remove` texts that are not tokens of this vocab.
    pub unresolved_overrides: Vec<String>,
}

pub(crate) fn derive_eog_ids(
    vocab: &Vocab,
    eot: Option<u32>,
    eom: Option<u32>,
    chat_template: Option<&str>,
    add: &[String],
    remove: &[String],
) -> (Vec<u32>, EogEvidence) {
    let mut ids: BTreeSet<u32> = BTreeSet::new();
    ids.extend(vocab.eos_token_id());
    ids.extend(eot);
    ids.extend(eom);
    let mut template_ids: Vec<u32> = chat_template
        .map(tail_literals)
        .unwrap_or_default()
        .iter()
        .filter_map(|text| vocab.token_id(text))
        .collect();
    if template_ids.is_empty() {
        template_ids = chat_template.map(|template| opening_marker_ids(vocab, template)).unwrap_or_default();
    }
    ids.extend(template_ids.iter().copied());
    let mut unresolved_overrides = Vec::new();
    for text in add {
        match vocab.token_id(text) {
            Some(token_id) => {
                ids.insert(token_id);
            }
            None => unresolved_overrides.push(text.clone()),
        }
    }
    for text in remove {
        match vocab.token_id(text) {
            Some(token_id) => {
                ids.remove(&token_id);
            }
            None => unresolved_overrides.push(text.clone()),
        }
    }
    let evidence = EogEvidence {
        template_ids,
        unresolved_overrides,
    };
    (ids.into_iter().collect(), evidence)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Item {
    Open,
    Loop,
    Branch,
    Close,
    Output(usize),
    /// Raw template text between tags, trimmed and non-empty; indexes `literals`.
    Raw(usize),
    Other,
}

/// Roles of the chat messages API that answer an assistant turn.
const ANSWERING_ROLES: [&str; 2] = ["user", "tool"];

struct Parsed<'template> {
    items: Vec<Item>,
    /// `Output` slots hold an expression's string literals; `Raw` slots hold
    /// the one trimmed text.
    literals: Vec<Vec<String>>,
    /// Statement body per item, empty for non-statements.
    bodies: Vec<&'template str>,
}

/// Parses `{% %}` / `{{ }}` structure plus the raw text between tags.
fn parse(template: &str) -> Parsed<'_> {
    let mut parsed = Parsed {
        items: Vec::new(),
        literals: Vec::new(),
        bodies: Vec::new(),
    };
    let mut cursor = 0;
    let mut scan = 0;
    while let Some(found) = template[scan..].find('{') {
        let start = scan + found;
        let tail = &template[start..];
        let (close, kind) = match tail.get(..2) {
            Some("{#") => ("#}", '#'),
            Some("{%") => ("%}", '%'),
            Some("{{") => ("}}", '{'),
            _ => {
                scan = start + 1;
                continue;
            }
        };
        let Some(end) = tail.find(close) else { break };
        let body = tail[2..end].trim_matches(|character: char| {
            character == '-' || character == '+' || character.is_whitespace()
        });
        let raw = template[cursor..start].trim();
        if !raw.is_empty() {
            parsed.literals.push(alloc::vec![String::from(raw)]);
            parsed.items.push(Item::Raw(parsed.literals.len() - 1));
            parsed.bodies.push("");
        }
        match kind {
            '%' => {
                parsed.items.push(statement_item(body));
                parsed.bodies.push(body);
            }
            '{' => {
                parsed.literals.push(string_literals(body));
                parsed.items.push(Item::Output(parsed.literals.len() - 1));
                parsed.bodies.push("");
            }
            _ => {}
        }
        cursor = start + end + 2;
        scan = cursor;
    }
    parsed
}

fn statement_item(body: &str) -> Item {
    let keyword = body.split_whitespace().next().unwrap_or("");
    match keyword {
        "for" => Item::Loop,
        "if" | "macro" | "call" | "filter" | "block" => Item::Open,
        "set" if !body.contains('=') => Item::Open,
        "elif" | "else" => Item::Branch,
        "endfor" | "endif" | "endmacro" | "endcall" | "endfilter" | "endblock" | "endset" => {
            Item::Close
        }
        _ => Item::Other,
    }
}

fn string_literals(expression: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut chars = expression.chars();
    while let Some(quote) = chars.next() {
        if quote != '\'' && quote != '"' {
            continue;
        }
        let mut text = String::new();
        while let Some(character) = chars.next() {
            match character {
                '\\' => match chars.next() {
                    Some('n') => text.push('\n'),
                    Some('t') => text.push('\t'),
                    Some(other) => text.push(other),
                    None => break,
                },
                other if other == quote => break,
                other => text.push(other),
            }
        }
        found.push(text);
    }
    found
}

/// Matching close index for every `Open`, and each item's enclosing `Open`.
fn nesting(items: &[Item]) -> (Vec<Option<usize>>, Vec<Option<usize>>) {
    let mut close_of = alloc::vec![None; items.len()];
    let mut parent = alloc::vec![None; items.len()];
    let mut stack: Vec<usize> = Vec::new();
    for (index, item) in items.iter().enumerate() {
        parent[index] = stack.last().copied();
        match item {
            Item::Open | Item::Loop => stack.push(index),
            Item::Close => {
                if let Some(open) = stack.pop() {
                    close_of[open] = Some(index);
                }
            }
            _ => {}
        }
    }
    (close_of, parent)
}

fn last_top_level_loop(
    items: &[Item],
    close_of: &[Option<usize>],
    parent: &[Option<usize>],
) -> Option<usize> {
    (0..items.len())
        .rev()
        .find(|&index| items[index] == Item::Loop && parent[index].is_none() && close_of[index].is_some())
}

/// Raw text that opens the message-loop branches answering an assistant turn
/// (`== 'user'`, `== 'tool'` of the messages API), in template order. A
/// template whose assistant turn has no closing literal ends that turn where
/// one of these opens the next.
pub(crate) fn opening_markers(template: &str) -> Vec<String> {
    let Parsed { items, literals, bodies } = parse(template);
    let (close_of, parent) = nesting(&items);
    let Some(message_loop) = last_top_level_loop(&items, &close_of, &parent) else {
        return Vec::new();
    };
    let Some(loop_close) = close_of[message_loop] else {
        return Vec::new();
    };
    (message_loop + 1..loop_close)
        .filter(|&index| matches!(items[index], Item::Open | Item::Branch))
        .filter(|&index| answers_assistant(bodies[index]))
        .filter_map(|index| match items.get(index + 1) {
            Some(Item::Raw(slot)) => literals[*slot].first().cloned(),
            _ => None,
        })
        .collect()
}

fn answers_assistant(condition: &str) -> bool {
    !condition.contains("!=")
        && string_literals(condition)
            .iter()
            .any(|literal| ANSWERING_ROLES.contains(&literal.as_str()))
}

/// Ids of the opening markers that begin with a special (added) token of the
/// vocab; text that is not a special token at its start adds nothing.
pub(crate) fn opening_marker_ids(vocab: &Vocab, template: &str) -> Vec<u32> {
    let mut ids: Vec<u32> = opening_markers(template)
        .iter()
        .filter_map(|marker| vocab.longest_added_token_match(marker.as_bytes()))
        .map(|(token_id, _)| token_id)
        .collect();
    ids.dedup();
    ids
}

/// The string literals the template emits in the final conditional chain of
/// its last top-level loop (the message loop), found by descending through
/// every block that is the last thing inside its parent. In a chat template
/// that is where a turn is closed or handed to a tool; a literal is returned
/// only if it is the last literal of its output expression that is not pure
/// whitespace, trimmed. Empty when the template has no top-level loop.
pub(crate) fn tail_literals(template: &str) -> Vec<String> {
    let Parsed { items, literals, .. } = parse(template);
    let (close_of, parent) = nesting(&items);
    let Some(mut open) = last_top_level_loop(&items, &close_of, &parent) else {
        return Vec::new();
    };
    while let Some(close) = close_of[open] {
        let last_child = (open + 1..close)
            .rev()
            .find(|&index| {
                matches!(items[index], Item::Open | Item::Loop) && parent[index] == Some(open)
            });
        match last_child {
            Some(child) if close_of[child].is_some_and(|child_close| child_close + 1 == close) => {
                open = child;
            }
            _ => break,
        }
    }
    let Some(close) = close_of[open] else {
        return Vec::new();
    };
    items[open + 1..close]
        .iter()
        .filter_map(|item| match item {
            Item::Output(slot) => literals[*slot]
                .iter()
                .rev()
                .map(|text| text.trim())
                .find(|text| !text.is_empty())
                .map(String::from),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    //! worked example (the spec), flint's real `tokenizer.chat_template`:
    //! one top-level loop `for m in messages`; inside it the role chain
    //! `if system / elif developer / elif user / elif assistant / elif tool`.
    //! no `{{ '...' }}` string literal sits in any branch, so `tail_literals`
    //! is []. raw text opens every branch: system `<|system|>`, developer
    //! `<|developer|>`, user `<|user|>`, assistant `<|assistant|>`, tool
    //! `<|observation|><tool_response>`. the roles that answer an assistant
    //! turn are `user` and `tool`: markers `<|user|>` and
    //! `<|observation|><tool_response>`. the longest special token at the start
    //! of each is `<|user|>` (150003) and `<|observation|>` (150005). with
    //! eos 150001 the set is {150001, 150003, 150005}, llama.cpp's set.
    //! a closing literal, when present, is used alone.

    use super::*;
    use crate::byte_level::byte_to_char;
    use crate::vocab::TokenType;

    // verbatim tail structure of the real gemma4-E2B chat template
    // (tokenizer.chat_template of sha256-3646b4c1...), with the earlier
    // per-message literals that must NOT count as turn-closing.
    const GEMMA_SHAPED_TEMPLATE: &str = "{%- macro format_tool_response_block(tool_name, response) -%}\n    {{- '<|tool_response>' -}}\n    {{- '<tool_response|>' -}}\n{%- endmacro -%}\n{{- bos_token -}}\n{%- for message in loop_messages -%}\n    {%- if message['role'] != 'tool' -%}\n        {{- '<|turn>' + role + '\\n' }}\n        {%- if item['type'] == 'image' -%}\n            {{- '<|image|>' -}}\n        {%- endif -%}\n        {{- captured_content -}}\n        {%- if ns.prev_message_type == 'tool_call' and not ns_tr_out.flag -%}\n            {{- '<|tool_response>' -}}\n        {%- elif not (ns_tr_out.flag and not has_content) -%}\n            {{- '<turn|>\\n' -}}\n        {%- endif -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{- '<|turn>model\\n' -}}\n{%- endif -%}";

    const CHATML_SHAPED_TEMPLATE: &str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

    fn vocab_with(texts: &[&str], eos: Option<u32>) -> Vocab {
        let mut tokens: Vec<String> = (0..=255u8)
            .map(|byte| String::from(byte_to_char(byte)))
            .collect();
        tokens.extend(texts.iter().map(|text| String::from(*text)));
        Vocab::new(tokens, &[], None, eos, None).expect("vocab builds")
    }

    #[test]
    fn gemma_shaped_template_tail_names_the_turn_close_and_tool_response_only() {
        assert_eq!(
            tail_literals(GEMMA_SHAPED_TEMPLATE),
            ["<|tool_response>", "<turn|>"]
        );
    }

    #[test]
    fn chatml_shaped_template_tail_names_im_end() {
        assert_eq!(tail_literals(CHATML_SHAPED_TEMPLATE), ["<|im_end|>"]);
    }

    #[test]
    fn template_tail_tokens_join_the_key_ids_and_literals_outside_the_tail_do_not() {
        let vocab = vocab_with(&["<eos>", "<|tool_response>", "<turn|>", "<|image|>", "<|turn>"], Some(256));
        let (ids, evidence) =
            derive_eog_ids(&vocab, None, None, Some(GEMMA_SHAPED_TEMPLATE), &[], &[]);
        assert_eq!(ids, [256, 257, 258]);
        assert_eq!(evidence.template_ids, [257, 258]);
    }

    #[test]
    fn config_add_and_remove_override_the_inference_and_unknown_texts_are_reported() {
        let vocab = vocab_with(&["<eos>", "<|tool_response>", "<turn|>", "<|image|>"], Some(256));
        let add = [String::from("<|image|>"), String::from("<nope>")];
        let remove = [String::from("<|tool_response>")];
        let (ids, evidence) =
            derive_eog_ids(&vocab, None, None, Some(GEMMA_SHAPED_TEMPLATE), &add, &remove);
        assert_eq!(ids, [256, 258, 259]);
        assert_eq!(evidence.unresolved_overrides, ["<nope>"]);
    }

    #[test]
    fn a_template_without_a_loop_names_nothing() {
        assert_eq!(tail_literals("{{ 'hello' }}"), Vec::<String>::new());
    }

    // markers are control tokens, everything before them the 256 byte tokens.
    fn typed_vocab(markers: &[&str], eos: Option<u32>) -> Vocab {
        let mut vocab = vocab_with(markers, eos);
        let mut types = alloc::vec![TokenType::Normal; 256];
        types.extend(markers.iter().map(|_| TokenType::Control));
        vocab = vocab.with_token_types(types).expect("one type per token");
        vocab
    }

    // the real flint template's message loop, structure kept, bodies elided:
    // each turn opens with a raw marker, none closes, the tool role's marker
    // is followed by more raw text.
    const FLINT_SHAPED_TEMPLATE: &str = "{%- for m in messages -%}\n{%- if m.role == 'system' -%}\n<|system|>{{ m.content }}\n{%- elif m.role == 'user' -%}\n<|user|>{{ m.content }}\n{%- elif m.role == 'assistant' -%}\n<|assistant|>\n{%- if m.content %}{{ m.content }}{% endif -%}\n{%- elif m.role == 'tool' -%}\n<|observation|><tool_response>{{ m.content }}</tool_response>\n{%- endif -%}\n{%- endfor -%}";

    #[test]
    fn open_only_template_ends_the_assistant_turn_where_user_and_tool_turns_open() {
        let vocab = typed_vocab(&["<eos>", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"], Some(256));
        let (ids, evidence) = derive_eog_ids(&vocab, None, None, Some(FLINT_SHAPED_TEMPLATE), &[], &[]);
        assert_eq!(ids, [256, 258, 260]);
        assert_eq!(evidence.template_ids, [258, 260]);
    }

    #[test]
    fn closing_literal_wins_over_opening_markers() {
        let template = "{%- for m in messages -%}{%- if m.role == 'user' -%}<|user|>{{ m.content }}{%- elif m.role == 'assistant' -%}<|assistant|>{{ m.content }}{{ '<|end|>' }}{%- endif -%}{%- endfor -%}";
        let vocab = typed_vocab(&["<eos>", "<|user|>", "<|assistant|>", "<|end|>"], Some(256));
        let (ids, evidence) = derive_eog_ids(&vocab, None, None, Some(template), &[], &[]);
        assert_eq!(ids, [256, 259]);
        assert_eq!(evidence.template_ids, [259]);
    }

    #[test]
    fn opening_marker_that_is_not_a_special_token_is_not_added() {
        let template = "{%- for m in messages -%}{%- if m.role == 'user' -%}User: {{ m.content }}{%- elif m.role == 'tool' -%}<|observation|>{{ m.content }}{%- endif -%}{%- endfor -%}";
        let ordinary = typed_vocab(&["<eos>"], Some(256));
        let (ids, _) = derive_eog_ids(&ordinary, None, None, Some(template), &[], &[]);
        assert_eq!(ids, [256], "neither marker is a vocab token");

        let mut tokens: Vec<String> = (0..=255u8).map(|byte| String::from(byte_to_char(byte))).collect();
        tokens.extend(["<eos>", "<|observation|>"].map(String::from));
        let mut types = alloc::vec![TokenType::Normal; 256];
        types.extend([TokenType::Control, TokenType::Normal]);
        let plain = Vocab::new(tokens, &[], None, Some(256), None)
            .and_then(|built| built.with_token_types(types))
            .expect("vocab builds");
        let (ids, _) = derive_eog_ids(&plain, None, None, Some(template), &[], &[]);
        assert_eq!(ids, [256], "a vocab token that is not special is not added");
    }

    #[test]
    fn config_remove_overrides_an_opening_marker() {
        let vocab = typed_vocab(&["<eos>", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"], Some(256));
        let remove = [String::from("<|observation|>")];
        let (ids, _) = derive_eog_ids(&vocab, None, None, Some(FLINT_SHAPED_TEMPLATE), &[], &remove);
        assert_eq!(ids, [256, 258]);
    }

    #[test]
    fn opening_markers_skip_negated_and_non_answering_roles() {
        assert_eq!(opening_markers(FLINT_SHAPED_TEMPLATE), ["<|user|>", "<|observation|><tool_response>"]);
        let negated = "{%- for m in messages -%}{%- if m.role != 'user' -%}<|x|>{%- endif -%}{%- endfor -%}";
        assert_eq!(opening_markers(negated), Vec::<String>::new());
    }

    #[cfg(feature = "gguf")]
    fn header_of(env_key: &str, default_path: &str) -> proxima_gguf::ParsedGguf {
        use std::io::{Read, Seek, SeekFrom};

        let path = std::env::var(env_key).unwrap_or_else(|_| String::from(default_path));
        let mut file = std::fs::File::open(&path)
            .unwrap_or_else(|error| panic!("{env_key} must name an existing gguf, tried {path}: {error}"));
        let mut header = Vec::new();
        for cap in [4usize << 20, 16 << 20, 64 << 20, 256 << 20] {
            header.resize(cap, 0);
            file.seek(SeekFrom::Start(0)).expect("seek to file start");
            let read = file.read(&mut header).expect("read gguf header region");
            header.truncate(read);
            if let Ok(parsed) = proxima_gguf::pipe::parse_complete(&header) {
                return parsed;
            }
        }
        panic!("{env_key}: gguf metadata region did not fit in 256 MiB");
    }

    #[cfg(feature = "gguf")]
    #[test]
    fn real_flint_template_gives_eos_user_and_observation() {
        let parsed = header_of("PROXIMA_FLINT_GGUF", "/Users/brianbruggeman/models/llm/flint-gguf/flint-UD-Q4_K_XL.gguf");
        let vocab = crate::gguf::vocab_from_metadata(&parsed).expect("builds vocab from real flint metadata");
        assert_eq!(vocab.eog_token_ids(), &[150_001, 150_003, 150_005]);
        assert_eq!(
            [150_001, 150_003, 150_005].map(|token_id| vocab.token_str(token_id)),
            [Some("<|end_of_text|>"), Some("<|user|>"), Some("<|observation|>")]
        );
    }
}
