# MoE router sizes (config-sourced)

Router params = hidden x routed_experts x moe_layers (no router bias found in any config.json below; "unsourced" where config is silent).
fp16 MB = params x 2 / 1e6.

| model | layers / moe layers | hidden | experts | top-k | shared | expert ffn dim | router params | router fp16 MB | source URL |
|---|---|---|---|---|---|---|---|---|---|
| mistralai/Mixtral-8x7B-v0.1 | 32 / 32 | 4096 | 8 | 2 | none (absent) | 14336 | 1,048,576 | 2.097 | https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/raw/main/config.json |
| openai/gpt-oss-20b | 24 / 24 (no dense key) | 2880 | 32 | 4 | none (absent) | 2880 | 2,211,840 | 4.424 | https://huggingface.co/openai/gpt-oss-20b/raw/main/config.json |
| openai/gpt-oss-120b | 36 / 36 (no dense key) | 2880 | 128 | 4 | none (absent) | 2880 | 13,271,040 | 26.542 | https://huggingface.co/openai/gpt-oss-120b/raw/main/config.json |
| meta-llama/Llama-4-Scout-17B-16E | 48 / 48 (interleave_moe_layer_step=1) | 5120 | 16 | 1 | absent in config (unsourced) | 8192 (dense mlp 16384) | 3,932,160 | 7.864 | gated: 401 on meta-llama repo. Read via mirror https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct/raw/main/config.json (mirror, not the original repo) |
| Qwen/Qwen3-30B-A3B | 48 / 48 (decoder_sparse_step=1, mlp_only_layers=[]) | 2048 | 128 | 8 | none (absent) | 768 | 12,582,912 | 25.166 | https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json |
| Qwen/Qwen3-235B-A22B | 94 / 94 (decoder_sparse_step=1) | 4096 | 128 | 8 | none (absent) | 1536 | 49,283,072 | 98.566 | https://huggingface.co/Qwen/Qwen3-235B-A22B/raw/main/config.json |
| deepseek-ai/DeepSeek-V3 | 61 / 58 (first_k_dense_replace=3) | 7168 | 256 | 8 | 1 (n_shared_experts=1) | 2048 (dense 18432) | 106,430,464 | 212.861 | https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json |
| moonshotai/Kimi-K2-Instruct | 61 / 60 (first_k_dense_replace=1) | 7168 | 384 | 8 | 1 (n_shared_experts=1) | 2048 (dense 18432) | 165,150,720 | 330.301 | https://huggingface.co/moonshotai/Kimi-K2-Instruct/raw/main/config.json |
| ibm-granite/granite-3.1-1b-a400m-instruct | 24 / 24 (no dense key) | 1024 | 32 | 8 | none (absent) | 512 | 786,432 | 1.573 | https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-instruct/raw/main/config.json |
| google gemma-4-26B-A4B-it (Gemma 4 26B MoE) | 30 / 30 (enable_moe_block=true, all layers; assumed all) | 2816 | 128 | 8 (top_k_experts) | absent in config (unsourced; search snippet claims 1, not verified in config) | 704 (dense intermediate 2112) | 10,813,440 | 21.627 | https://huggingface.co/google/gemma-4-26B-A4B-it/raw/main/config.json |
| Qwen/Qwen3.6-35B-A3B | 40 / 40 (no sparse-step key; assumed all) | 2048 | 256 | 8 | 1 (shared_expert_intermediate_size=512; count key absent) | 512 | 20,971,520 | 41.943 | https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json |

Notes:
- All values came from WebFetch summaries of the raw config.json, not byte-for-byte copies. Re-check any value that feeds a decision.
- Llama 4 Scout: the original repo (meta-llama/Llama-4-Scout-17B-16E) returned HTTP 401; the -Instruct original also 401. Values are from the unsloth mirror.
- Qwen3.6-35B-A3B: hybrid layers (linear_attention + full_attention per layer_types); MoE-per-layer is inferred from absence of decoder_sparse_step / mlp_only_layers, not stated.
- Gemma 4 26B: the "26B-A4B" repo is the 26B MoE variant. Config keys are num_experts / top_k_experts (not top_k). Shared-expert count not in config.
- DeepSeek-V3 and Kimi K2 router score-correction bias vectors are not in config.json; router params above exclude them (unsourced).
- gpt-oss config sets attention_bias=true; no router bias key is present in config.
