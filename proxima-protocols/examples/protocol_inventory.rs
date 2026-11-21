//! Render the protocol surface. `markdown` writes `docs/protocols.md`;
//! `jsonl` writes `ai_docs/protocols.jsonl`.

use proxima_protocols::inventory::INVENTORY;

const INTRO: &str = "\
# The protocol surface

Every wire protocol is a sans-IO codec in `proxima-protocols`, one
feature-gated module each, folded from nineteen single-purpose parser crates.
The codec is buffer-in / buffer-out; the client or listener that owns the
transport lives in its own crate. Every protocol with a `bench` entry has a
criterion harness even where no number is published yet.

Rendered from `proxima-protocols/src/inventory.rs`. Regenerate:

```sh
cargo run -p proxima-protocols --example protocol_inventory -- markdown > docs/protocols.md
cargo run -p proxima-protocols --example protocol_inventory -- jsonl > ai_docs/protocols.jsonl
```

`inventory::tests::every_pub_module_is_in_the_inventory` fails when a module is
added without a row.
";

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("jsonl") => emit_jsonl(),
        _ => emit_markdown(),
    }
}

fn emit_markdown() {
    println!("{INTRO}");
    println!("| protocol | module | feature | transport | bench | status | last result |");
    println!("|---|---|---|---|---|---|---|");
    for entry in INVENTORY {
        println!(
            "| {} | `{}` | `{}` | {} | {} | {} | {} |",
            entry.protocol,
            entry.module,
            entry.feature,
            entry.transport,
            entry.bench,
            entry.status,
            entry.last_result,
        );
    }
}

fn emit_jsonl() {
    for entry in INVENTORY {
        println!(
            "{{\"protocol\":\"{}\",\"module\":\"{}\",\"kind\":\"{}\",\"feature\":\"{}\",\"transport\":\"{}\",\"bench\":\"{}\",\"status\":\"{}\",\"last_result\":\"{}\"}}",
            entry.protocol,
            entry.module,
            entry.kind,
            entry.feature,
            entry.transport,
            entry.bench,
            entry.status,
            entry.last_result,
        );
    }
}
