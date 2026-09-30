export CARGO_TARGET_DIR=/private/tmp/cargo_target_long_ctx
export GEMMA4="$HOME/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd"
export QWEN36="$HOME/.ollama/models/blobs/sha256-f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d"
export QWEN3_8B="$HOME/.ollama/models/blobs/sha256-a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f"
export LONG_CTX_SPEC=proxima-tensor/specs/long-context

niah() {
    cargo run -p proxima-model-interop --release --example long_context_niah --features std,metal -- "$@"
}
