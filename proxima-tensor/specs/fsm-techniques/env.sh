export LOGS=/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm
export LLAMA=/Users/brianbruggeman/repos/slot-0/proxima-prefix-cache/scratchpad/bin/llama-f1ea2062/bin/llama-server
export LLAMA_TOKENIZE=/Users/brianbruggeman/repos/slot-0/proxima-prefix-cache/scratchpad/bin/llama-f1ea2062/bin/llama-tokenize
export GEMMA4_E2B="$HOME/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd"
export GEMMA4="$GEMMA4_E2B"
export GEMMA4_26B="$HOME/.ollama/models/blobs/sha256-ea549b7688d4c95019754880c21e3f29c58c985a7a1c3b37b9eebd0a95224129"
export GRANITE_MOE="$HOME/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80"

niah() {
    cargo run -p proxima-model-interop --release --example long_context_niah --features std,metal -- "$@"
}
