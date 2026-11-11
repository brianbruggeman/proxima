# llama.cpp chat-template fixtures

Oracle: llama-server build 11671 (b11671-ba3c05f71),
`/Users/brianbruggeman/.hermes/tools/llamacpp-metal-11671-darwin-arm64/llama-server`.

Captured once, 2026-11-11, then the server was stopped by PID:

    llama-server -m ~/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd --jinja --port 18755 -ngl 0 -c 512
    curl -s http://127.0.0.1:18755/apply-template -d @msgs.json > gemma4_apply_template.json

- `msgs.json`: the two messages ragd sends (system, user). Copied from
  `~/.cache/kvgate/flintchat/msgs.json`.
- `gemma4_apply_template.json`: gemma4's `/apply-template` output (`bos_token` `<bos>`, not in the prompt; llama adds BOS at tokenization).
- flint's expected prompt (llama-server b12110) is
  `<|system|>You are a helpful assistant.<|user|>Reply with the single word: flint.<|assistant|>`
  and is asserted inline in the test.

llama-server rendered gemma4 with `bos_token` empty and `enable_thinking` true
(the `<|think|>` line in the prompt); the test passes the same variables.
