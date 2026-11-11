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
    /// Ids the chat template's tail literals resolved to (rule 2).
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
    let template_ids: Vec<u32> = chat_template
        .map(tail_literals)
        .unwrap_or_default()
        .iter()
        .filter_map(|text| vocab.token_id(text))
        .collect();
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
    Other,
}

/// Parses `{% %}` / `{{ }}` structure only: nesting plus, per output
/// expression, its string literals (returned in `literals`, indexed by the
/// `Output` payload).
fn parse(template: &str) -> (Vec<Item>, Vec<Vec<String>>) {
    let mut items = Vec::new();
    let mut literals: Vec<Vec<String>> = Vec::new();
    let mut rest = template;
    while let Some(start) = rest.find("{") {
        let tail = &rest[start..];
        let (close, kind) = match tail.get(..2) {
            Some("{#") => ("#}", '#'),
            Some("{%") => ("%}", '%'),
            Some("{{") => ("}}", '{'),
            _ => {
                rest = &rest[start + 1..];
                continue;
            }
        };
        let Some(end) = tail.find(close) else { break };
        let body = tail[2..end].trim_matches(|character: char| {
            character == '-' || character == '+' || character.is_whitespace()
        });
        match kind {
            '%' => items.push(statement_item(body)),
            '{' => {
                literals.push(string_literals(body));
                items.push(Item::Output(literals.len() - 1));
            }
            _ => {}
        }
        rest = &tail[end + 2..];
    }
    (items, literals)
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

/// The string literals the template emits in the final conditional chain of
/// its last top-level loop (the message loop), found by descending through
/// every block that is the last thing inside its parent. In a chat template
/// that is where a turn is closed or handed to a tool; a literal is returned
/// only if it is the last literal of its output expression that is not pure
/// whitespace, trimmed. Empty when the template has no top-level loop.
pub(crate) fn tail_literals(template: &str) -> Vec<String> {
    let (items, literals) = parse(template);
    let (close_of, parent) = nesting(&items);
    let last_loop = (0..items.len())
        .rev()
        .find(|&index| items[index] == Item::Loop && parent[index].is_none() && close_of[index].is_some());
    let Some(mut open) = last_loop else {
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
    use super::*;
    use crate::byte_level::byte_to_char;

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
        assert!(tail_literals("{{ 'hello' }}").is_empty());
    }
}
