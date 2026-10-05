# slice 0 (re-cut): oracles, granite as configuration, worked values for the hooks

anchors read at main a7c08c4c (full sha a7c08c4c95836f112742f0e30e4ddd672087cda0), read with `git -C /Users/brianbruggeman/repos/slot-0/proxima-windows show a7c08c4c:<path>`; llama.cpp at f1ea20621 (`/Users/brianbruggeman/repos/others/llama.cpp`). Main was at a7c08c4c when this file was first cut and at 0e2785bd at the audit that sent it back; a7c08c4c, one commit later, changes only `proxima-model-interop/tests/fixtures/llama-parity/gemma4_26b/llama_ids.json` and the architecture-as-data SPEC. At a7c08c4c the symbol each card names in `read first` was located again and its line hint corrected where it had moved by more than a few lines (`PromptCache::store` is ~942, was ~959; the tokenizer anchors are those of FT0.39); every `path::symbol (~line N)` below is the line at a7c08c4c, a hint only, and the executor re-locates by symbol. Nothing in this file was compiled; the numbers marked "measured" were taken with llama-server and llama-tokenize while cutting (single run each, status plausible), and every card that depends on one re-measures and asserts it. The records of the gemma4 26B and E2B chat and raw probes are kept under `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/probe_cut/` (token ids, per-step top-2, server logs); the earlier E2B follow-up and reuse probes and the granite probes were written to scratch space that was cleaned, so those figures are unrecorded and are re-taken by the cards.

Rules: CARDS.md (binding) and the pipeline-as-data SPEC. Owner direction that governs every card here: the hooks are built so a technique can be vetted later; a technique is never built. In this slice a technique appears only as the proof that a hook (or a worked value) expresses it. Training-side work (cartridge training, calibration fitters) is out of scope, and so is model merging.

Abbreviations. `PAD` is `proxima-tensor/specs/pipeline-as-data/` (SPEC.md, research.md, research-retrieval-attention.md, sketches/NN-*.md; on main). `SPECDIR` is `proxima-tensor/specs/fsm-techniques/` (committed on main, repo-relative). Paths are relative to the proxima repo root (the checkout holding main, `/Users/brianbruggeman/repos/slot-0/proxima-windows`) unless absolute. Cross-file `needs` use the ids as they stand in this file's id map.

Where `SPECDIR` lives. The fsm-techniques spec files (SPEC.md, CARDS.md, TASKS.md and these card files) are committed to main under `proxima-tensor/specs/fsm-techniques/` before execution, so `SPECDIR` in a card is that repo-relative directory in the checkout holding main (`/Users/brianbruggeman/repos/slot-0/proxima-windows`). Every card that writes a spec file (FT0.4, FT0.14 to FT0.29) writes it under `SPECDIR`, stages it with `git add`, checks `git diff --cached --stat` and commits with `git commit` (a plain conventional subject, for example `docs(fsm): derive the action speculation worked example`), all in that one checkout, with stage paths relative to its root; every command that reads `env.sh` or `worked-examples.md` names it by the repo-relative `SPECDIR` path and runs from the checkout root. Cards that run cargo run from that same checkout, with `env.sh` sourced. A card may anchor a symbol or section of SPEC.md as `proxima-tensor/specs/fsm-techniques/SPEC.md::symbol`; a card that needs a rule of the spec still states the rule in full in its own change text.

Test models (owner, 2026-10-04): gemma4 for dense or MoE, granite for MoE. No qwen of any kind appears in a test, fixture, oracle or validation added by a card of this file. Existing table-driven tests that already carry qwen rows are edited only to add a granite row; no card runs a test by a qwen name and no card adds a qwen row.
- gemma4 E2B (ollama `gemma4:e2b-it-qat`, dense, 35 layers), blob `/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd`, fixture name `gemma4_e2b`.
- gemma4 26B-A4B (ollama `batiai/gemma4-26b:latest`, MoE, 30 layers), blob `/Users/brianbruggeman/.ollama/models/blobs/sha256-ea549b7688d4c95019754880c21e3f29c58c985a7a1c3b37b9eebd0a95224129`, fixture name `gemma4_26b`.
- granite 3.1 MoE 1B-A400M (ollama `granite3.1-moe:1b`, `general.architecture = granitemoe`, 24 layers, 32 experts, 8 used, Q8_0, 1422239776 bytes), blob `/Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80`, fixture name `granite_moe`. Blob paths come from `ollama show --modelfile <name>` (the first `FROM` line).

Shared constants used by several cards:
- `LOGS=/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm`
- `LLAMA=/Users/brianbruggeman/repos/slot-0/proxima-prefix-cache/scratchpad/bin/llama-f1ea2062/bin/llama-server`; `LLAMA_TOKENIZE` is `llama-tokenize` in the same `bin/` (both present; `$LLAMA --version` prints `version: 0.5.0-dev (build 2633, commit f1ea20621)`).
- Oracles run once and are recorded. A vendored fixture is never regenerated by a test, and a test never starts llama-server.

## id map (old card to new card)

Ids of kept and recut cards do not change, so every other file's `needs` on a slice-0 id still resolves. New cards are FT0.31 to FT0.37 and FT0.39 to FT0.56 (FT0.38 is dropped, see below). The worked-examples file is now created by FT0.14 (it was FT0.9), so a card elsewhere that needs the file to exist needs FT0.14.

Where one old card was cut into several to meet the size rule (one model-loading run per card, one coherent public item per card), the last card of the group keeps the old id and needs the others, so a `needs: FT0.5`, `FT0.6` or `FT0.7` in another file still means every checkpoint's fixture exists.

| old | new | verdict |
|---|---|---|
| FT0.1, FT0.2, FT0.3 | none | dropped |
| FT0.4 | FT0.4 | recut: env.sh exports gemma4 and granite blobs, no qwen variable |
| FT0.5 | FT0.46, FT0.47, FT0.5 | recut and split, one server session each: follow-up ids for gemma4_e2b (FT0.46), gemma4_26b (FT0.47), granite_moe (FT0.5, needs the other two); new turn-1 prompt, `--swa-full`, 26B rendered through the chat template and top-2 recorded for turn 2, all measured |
| FT0.6 | FT0.48, FT0.49, FT0.6 | recut and split, one server session each: top-2 log probabilities for gemma4_e2b (FT0.48), gemma4_26b (FT0.49), granite_moe (FT0.6, needs the other two); step count equals recorded token count; the 26B file is cut from the confident chat-templated parity prompts |
| FT0.7 | FT0.50 to FT0.54, FT0.7 | recut and split, one server session each: shifted chunk reuse (FT0.50 e2b, FT0.52 26b, FT0.54 granite) and the unshifted generation added to each file by the card after it (FT0.51, FT0.53, FT0.7, which needs the rest); request arrangement corrected, measured |
| FT0.8 | FT0.8 | recut: controls over 3 checkpoints, counts 5 tests / 3 / 9 / 3 |
| FT0.9 to FT0.13 | none | dropped |
| FT0.14 | FT0.14 | kept; now creates worked-examples.md |
| FT0.15 | FT0.15 | recut: today's victim rule is the default row, three policies are proof instances |
| FT0.16, FT0.17 | FT0.16, FT0.17 | kept; FT0.17 restates the keep 0.75 selection under the non-local count rule and drops its dependency on the paper re-read |
| FT0.18 | FT0.18 | recut: gemma4 E2B, 7 reading layers |
| FT0.19 | FT0.19 | kept |
| FT0.20, FT0.21 | FT0.20, FT0.21 | recut: gemma4_e2b, gemma4_26b, granite_moe; the reduction-width model gains the attention output width |
| FT0.22 to FT0.27 | FT0.22 to FT0.27 | kept; FT0.24, FT0.25, FT0.26 lose their dependencies on dropped cards, FT0.25 states the integer disagreement bound |
| FT0.28 | FT0.28 | kept; adds the equal-score tie rule |
| FT0.29 | FT0.29 | kept; n counts non-local blocks |
| FT0.30 | none | dropped |
| FT0.38 | none | dropped: main already maps `refact` (see below) |
| new | FT0.31 to FT0.37, FT0.39 to FT0.45, FT0.55, FT0.56 | granite 3.1 MoE as configuration: header oracle (FT0.31), graph pin (FT0.32), four scale hooks (embedding FT0.33, logit FT0.34 and FT0.35, attention score FT0.36, residual FT0.44 and FT0.37), the MoE rope-pairing refusal (FT0.45), granite vocabulary oracle for the existing `refact` rule (FT0.39), family profile (FT0.40), recorded ids (FT0.41), graph digest (FT0.42), bound weights (FT0.56), program scales (FT0.55), token parity (FT0.43) |

## dropped

- FT0.1: re-reads a paper's `p` direction for the draft SPEC register; no hook reads it, `keep_ratio` is a configured value given in FT0.17.
- FT0.2: verifies a paper's accept rule for the draft SPEC; the classifier judge is a pure function of class probabilities (FT0.24) and needs no paper fact.
- FT0.3: checks Milvus segment ordering for a technique row; the seal decision is specified by its own worked trace (FT0.14).
- FT0.9: the PAV isotonic fit is a calibration fitter, out of scope; the isotonic judge reads a supplied table (FT0.26).
- FT0.10: computing q-hat from calibration scores is a calibration fitter; the serving-side accept rule is FT0.25.
- FT0.11: k-means plus silhouette is the offline fit for cluster-route-escalate, a calibration fitter.
- FT0.12: per-cluster `argmin(Error + lambda * Cost)` is offline route-table fitting, calibration-side.
- FT0.13: the Tchebycheff threshold search is a threshold fitter, calibration-side.
- FT0.30: KL divergence is the cartridge training loss, training-side.
- FT0.38: added a `PreType::Refact` pre-split rule for granite's `tokenizer.ggml.pre = "refact"`. Main already maps `"refact"` to `PreType::DigitIsolatedGpt2` (`proxima-tokenizer/src/pretokenize.rs::PreType::from_gguf_name`), so the card would have duplicated a library rule, put a second `"refact"` arm in the mapping, and staged a file that does not exist; the existing hook is used (FT0.39).

## what the recut changes, in English

- Hooks in this slice. The oracle and worked-value cards serve the assemble, read, seal, settle, readout and position hooks (each card names its hook). The one hook change built here is granite as configuration, which sits on the model descriptor and the single-range lowering (the tokenizer already carries the `refact` pre-split rule granite declares): a literal embedding multiplier, a logit divisor, a per-layer attention score literal and a residual multiplier, each default-off and each refused by every engine that cannot lower it, and a `granitemoe` family profile. No granite-specific code path exists in the library; granite reaches the generic engine through its profile and the values its own header declares. The recut oracle and worked-value cards (0.5 to 0.8, 0.15, 0.18, 0.20, 0.21) keep their kind, which is substrate: the sketch gaps they serve (assemble stage list, read selection, seal and victim decision, readouts) are built by the cards of the slice that owns each hook, and each card here names the hook whose decision it fixes the value of.
- Premises that measurement or reading falsified in the cards being replaced, worst first:
  - FT0.7: llama.cpp reuses a chunk only when the new prompt is the cached prompt with a middle span removed (`tools/server/server-context.cpp` ~3217-3262: the scan starts at the first mismatch and slides over the cache). The old arrangement (two different prefixes, one shared chunk) measured `cache_n = 1` of 798 prompt tokens on gemma4 E2B with `--cache-reuse 64 --swa-full`: no reuse. The arrangement now is prefix, removed span, moved chunk; E2B measured `cache_n = 553` of 554, granite `654` of 655.
  - FT0.5: the old turn-1 prompt (record 2 of the parity ids) generates one token on gemma4 E2B (recorded `generated_ids = [106]`), so the old assertion of 32 generated tokens fails. Without `--swa-full` the old `cache_n >= len(turn1.prompt_ids)` also fails on E2B: measured `cache_n = 101` for 106 prompt ids; with `--swa-full` it is 137 (106 + 32 - 1).
  - FT0.5 and FT0.7 on gemma4 26B: the old raw-prose prompts make the 26B degenerate (measured: its 32 greedy tokens on lines 3000-3010 read `mthemuch-appreciated much-appreciated ...`, smallest top-2 gap 0.084 nats), the same finding the architecture-as-data SPEC records for the 26B parity prompts (read at a7c08c4c, which also records that the 26B parity case now runs three chat-templated prompts that llama itself picked by a top-1 minus top-2 margin of at least 1.0 nats at every step, and that proxima's ids equal llama's on all three). The cards render the 26B prompts through the chat template (coherent when measured) and record the top-2 of the compared request so a first divergence can be read against its margin.
  - FT0.6: the old assertion of 32 steps per record fails on gemma4 E2B, whose recorded generations are 3, 32 and 1 tokens (stop at the end-of-generation token). Measured with `n_probs 2`: step count equals token count in all three records.
  - FT0.20: gemma4 E2B has `feed_forward_length` 6144 for layers 0 to 14 and 12288 for layers 15 to 34, so the reduction width is 12288, not 6144, and the old model ignored the attention output width (8 heads x 512 = 4096 on E2B, 16 x 512 = 8192 on 26B).
  - FT0.18: only the 7 full-attention layers of gemma4 E2B (layers 4, 9, 14, 19, 24, 29, 34 in `swa_layers.txt`) read the global rows; 35 layers would count the 28 windowed ones.
  - FT0.17: the keep 0.75 selection listed `{0,2,3}` under an all-blocks count; under the decided non-local count it is `{0,1,2,3}`.
  - A card to add the `refact` pre-split rule (the old FT0.38) had a false premise. Granite declares `tokenizer.ggml.pre = "refact"`, and main already maps it: `proxima-tokenizer/src/pretokenize.rs::PreType::from_gguf_name` (~65 at a7c08c4c) sends `"refact"` to `PreType::DigitIsolatedGpt2` (~77), whose passes live in `pretokenize_passes.rs` (`DIGIT_ISOLATED_GPT2_PASSES`). The card is dropped; FT0.39 vendors granite vocabulary ids from llama-tokenize and runs the existing rule over them.
- Capture commands in FT0.5 to FT0.7 (and the cards cut from them, FT0.46 to FT0.54), FT0.39 and FT0.41 are one-shot `curl`, `jq` and `llama-tokenize` invocations against the vendored oracle, as in the cards they replace; nothing scripted is committed except the fixtures they write.

Test-name rule for this slice: no test added here may contain `serving_state_`, `action_speculation_` or `accept_rule_`, or start with `sansio_` (other slices count those prefixes).

Card list: 0.4 env.sh; 0.31 to 0.37, 0.39 to 0.45, 0.55 and 0.56 granite as configuration; 0.5 to 0.7 and 0.46 to 0.54 oracle fixtures; 0.8 control tests; 0.14 to 0.29 worked values (one example per card; 0.14 creates the file).

## env.sh

### 0.4 Write fsm-techniques env.sh

- id: FT0.4
- needs: none
- budget: 20 min
- crate(s): none
- read first:
  - `ollama show --modelfile gemma4:e2b-it-qat`, `ollama show --modelfile batiai/gemma4-26b:latest`, `ollama show --modelfile granite3.1-moe:1b`: the first `FROM` line of each is the checkpoint blob (the E2B file lists a second `FROM`, the 987 MB vision projector, which is not exported);
  - `proxima-tensor/specs/long-context/env.sh` (the old card sourced it; it defines `QWEN3_8B` and `QWEN36`, which this file must not export, and the `niah` function and `GEMMA4`, which other slices use, so both are redefined here).
- change:
  1. New file `SPECDIR/env.sh` (that is `proxima-tensor/specs/fsm-techniques/env.sh`), exactly:
     ```
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
     ```
- test: none; the validate command is the check.
- validate: `bash -c 'source proxima-tensor/specs/fsm-techniques/env.sh && echo "e2b=$(test -f "$GEMMA4_E2B" && echo 1 || echo 0) g26b=$(test -f "$GEMMA4_26B" && echo 1 || echo 0) granite=$(test -f "$GRANITE_MOE" && echo 1 || echo 0) llama=$(test -x "$LLAMA" && echo 1 || echo 0) tokenize=$(test -x "$LLAMA_TOKENIZE" && echo 1 || echo 0) niah=$(type -t niah) qwen_lines=$(grep -ci qwen proxima-tensor/specs/fsm-techniques/env.sh)"'`
- expect: `e2b=1 g26b=1 granite=1 llama=1 tokenize=1 niah=function qwen_lines=0`
- also green: n/a
- stage: `proxima-tensor/specs/fsm-techniques/env.sh`
- commit: `chore(fsm): add env.sh for the fsm techniques checks`
- done when: the expect line printed, and `git diff --cached --stat` run in the checkout holding main equals the stage list (the stage list is relative to that checkout root; `git add` it there, and make the commit there with `git commit`)
- do not: edit `long-context/env.sh`; source it; export any variable whose name or value names qwen
- gpu: none

## granite 3.1 MoE as configuration (the hook change built in this slice)

Order: 0.31 (header and checkpoint row), 0.32 (pin the graph), then the scale hooks in this order: 0.33 (embedding), 0.34 (logit field), 0.35 (logit lowering), 0.36 (attention score), 0.44 (residual field), 0.37 (residual lowering), 0.45 (MoE rope-pairing refusal); then 0.39 (granite vocabulary oracle for the existing `refact` rule), 0.40 (profile and header scales), 0.41 (recorded ids), 0.42 (graph digest, one capture run), 0.56 (bound weights, one capture run), 0.55 (program scales, one run), 0.43 (token parity). The scale hooks share one test module, `forward_scales`, at the end of `proxima-tensor/src/spec/tests.rs`; each tensor card's validate runs that module, the four existing `head_repeats` tests and `layer_taps_variant_matches_the_plain_program_and_returns_one_tap_per_layer`, with the count stated per card, counted in the order above (7, 10, 13, 13, 15, 18, 19, 20).

What the granite header declares and what llama.cpp does with it (`src/models/granite.cpp`, the graph `granitemoe` shares through `using graph = llama_model_granite::graph` in `src/models/models.h`, f1ea20621): `embedding_scale` 12.0 multiplies the embedding lookup (`src/llama-graph.cpp:2442`); `attention.scale` 0.015625 replaces `1/sqrt(head_dim)` (`granite.cpp:235`); `residual_scale` 0.22 multiplies each sublayer output before its residual add, attention and FFN alike (`granite.cpp:250-251` and `:310-311`); `logit_scale` 6.0 divides the logits (`granite.cpp:198`, `1.0f / f_logit_scale`). A value of 0.0 means unset (`f_embedding_scale != 0.0f`, `if (hparams.f_residual_scale)`, `f_attention_scale == 0.0f`). RoPE is the adjacent pairing (`LLM_ARCH_GRANITE_MOE` is in the `LLAMA_ROPE_TYPE_NORM` group, `src/llama-model.cpp:3019`, returned at :3039). The MoE FFN is softmax gating over 32 experts, 8 used, weights renormalised, SiLU: the shape the single-range MoE layer already lowers. The checkpoint has no `output.weight` (242 tensors, tied embeddings, which `bind_all_weights` binds through `bind_matmul_weight_as`, `proxima-model-interop/src/bind.rs` ~2343).

### 0.31 Add the granite_moe checkpoint and its header oracle

- id: FT0.31
- needs: none
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/tests/arch_data_baseline.rs::GEMMA4_26B` (~line 48 at a7c08c4c: the `Checkpoint` static shape), `::ALL` (~91), `::checkpoints_toml_lists_every_baseline_checkpoint` (~592), `::assert_architecture_key`;
  - `proxima-model-interop/tests/fixtures/llama-parity/checkpoints.toml` (the row shape; `sha256` is the full-content digest) and `proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/gguf_kv.txt` (the three comment lines and the dump layout this fixture copies);
  - `proxima-model-interop/src/lib.rs` re-exports `metadata_f32_optional`, `metadata_str` and `metadata_u32` (std-gated).
- change:
  1. New fixture `proxima-model-interop/tests/fixtures/llama-parity/granite_moe/gguf_kv.txt`: `mkdir -p` the directory, then three comment lines, then the stdout of llama.cpp's own header dump. Lines: `# oracle tool: llama.cpp gguf-py gguf_dump.py (header only, CPU, no tensors) from /Users/brianbruggeman/repos/others/llama.cpp at commit f1ea206218210afb913ae2f5d2c51faed35915da`, `# command: cd /Users/brianbruggeman/repos/others/llama.cpp/gguf-py && PYTHONPATH=. uv run --no-project --with numpy --with pyyaml --with tqdm python gguf/scripts/gguf_dump.py --no-tensors /Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80`, `# checkpoint: granite_moe`. The dump (stderr dropped) must report `Dumping 45 key/value pair(s)` and contain `granitemoe.embedding_scale = 12.0`, `granitemoe.residual_scale = 0.2199999988079071`, `granitemoe.logit_scale = 6.0`, `granitemoe.attention.scale = 0.015625` and `tokenizer.ggml.pre = 'refact'`. Stop and report any difference.
  2. `proxima-model-interop/tests/fixtures/llama-parity/checkpoints.toml`: the first comment line says "the seven real checkpoints"; make it "the eight real checkpoints". Append, at the end of the file, a `[[checkpoint]]` with `name = "granite_moe"`, `path = "/Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80"`, `env = "PROXIMA_ARCH_GRANITE_MOE_GGUF"`, `architecture = "granitemoe"`, `sha256 = "cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80"` (`openssl dgst -sha256` of the blob prints exactly this), `size_bytes = 1422239776`, `size_note = "library/granite3.1-moe:1b, 1B-a400M MoE (expert_count 32, 8 used), Q8_0, 1.4 GB"`, `present = true`.
  3. `proxima-model-interop/tests/arch_data_baseline.rs`: add `const GRANITE_MOE: Checkpoint = Checkpoint { name: "granite_moe", env: "PROXIMA_ARCH_GRANITE_MOE_GGUF", path: "<the blob path above>", architecture: "granitemoe" };` after the last existing checkpoint static; `ALL` becomes `[&Checkpoint; 8]` with `&GRANITE_MOE` appended; add `metadata_f32_optional`, `metadata_str` and `metadata_u32` to the existing `use proxima_model_interop::{..}` list.
- test: add `granite_moe_header_declares_the_scales_the_profile_cannot_carry` in `proxima-model-interop/tests/arch_data_baseline.rs`: open `GRANITE_MOE`, `parse_complete`, `assert_architecture_key`; then assert, with `metadata_f32_optional(&parsed, "granitemoe.<key>", -1.0)`: `embedding_scale == 12.0`, `residual_scale == 0.22_f32`, `logit_scale == 6.0`, `attention.scale == 0.015625`; with `metadata_u32`: `granitemoe.expert_count == 32`, `granitemoe.expert_used_count == 8`, `granitemoe.block_count == 24`, `granitemoe.embedding_length == 1024`; and `metadata_str(&parsed, "tokenizer.ggml.pre") == "refact"`. The existing `checkpoints_toml_lists_every_baseline_checkpoint` now also covers the new row.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_0_31 cargo nextest run -p proxima-model-interop --features std -j 1 -E 'binary(arch_data_baseline) & test(/granite_moe_header|checkpoints_toml_lists/)'`
- expect: `2 passed` (both names appear)
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: `proxima-model-interop/tests/arch_data_baseline.rs proxima-model-interop/tests/fixtures/llama-parity/checkpoints.toml proxima-model-interop/tests/fixtures/llama-parity/granite_moe/gguf_kv.txt`
- commit: `test(llama-parity): add the granite moe checkpoint and header oracle`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: edit another checkpoint's row or fixture; read tensor data (the test maps the file and reads the header only); add a model-loading test
- gpu: none (a read-only mapping; no weight page is read)

### 0.32 Pin the single-range forward graph before the scale hooks land

- id: FT0.32
- needs: none
- budget: 20 min
- crate(s): proxima-tensor
- read first:
  - `proxima-tensor/src/spec/tests.rs::head_repeats` (~line 17466 at a7c08c4c: a module with a `descriptor` helper, a `built` helper, an FNV `digest` of the program's debug text, and pre-change node counts and digests as constants; this card copies the pattern for the single-range engine);
  - `proxima-tensor/src/spec/descriptor.rs::mistral_descriptor_from_shape` (~247) and `::build_forward`'s `CacheStrategy::SingleRange` arm (~451);
  - `proxima-tensor/src/spec/attention_forward.rs::mistral_cached_forward_program_with_experts_and_layer_taps_with_rope_pairing` (~2395).
- change:
  1. `proxima-tensor/src/spec/tests.rs`, in `mod head_repeats`: change `fn digest` (~17474) to `pub(super) fn digest` (one word). Nothing else in that module changes.
  2. Same file, appended after `mod head_repeats`: `mod forward_scales { use super::*; use super::head_repeats::digest; ... }` holding:
     - four constants, captured in step 3: `const DENSE_PRE_CHANGE_NODES: usize = 0;`, `const DENSE_PRE_CHANGE_DIGEST: u64 = 0;`, `const MOE_PRE_CHANGE_NODES: usize = 0;`, `const MOE_PRE_CHANGE_DIGEST: u64 = 0;`
     - `fn profile_text(rope_layout: &str) -> String` returning `format!("score_scale_inverse_sqrt_head_dim = true\nvalue_norm = false\nrope_layout = \"{rope_layout}\"\n\n[ffn]\npost_attention_norm = false\ncombination = \"Exclusive\"\noutput_scale = false\nrouted_gating = \"Softmax\"\nrouted_expert_bias = false\nactivation = \"Silu\"\nexclusive_dense_post_norm = false\n")`;
     - `fn descriptor(expert_count: u32, expert_used_count: u32) -> ModelDescriptor`: parse `profile_text("adjacent")` with `toml::from_str::<FamilyProfile>` (expect "the dense profile parses") and return `mistral_descriptor_from_shape(16, 8, 16, 2, 1, 4, 2, expert_count, expert_used_count, false, false, false, false, &profile)`;
     - `fn built(descriptor: &ModelDescriptor) -> (Vec<Op>, NodeId)`: `build_forward(descriptor, true)` with `.expect("the single-range descriptor lowers")`, returning the program and the logits root (the first two elements of the returned tuple).
  3. Capture, before any other edit: with the four constants at 0 run the validate command once. The two tests below fail and print `left` (the real node count, and the real digest in decimal). Paste the four numbers into the constants and rerun.
- test: add `dense_default_graph_equals_the_pre_change_graph` and `moe_default_graph_equals_the_pre_change_graph` in `mod forward_scales`: for `descriptor(0, 0)` and `descriptor(4, 2)` respectively, `program.len()` equals the node-count constant and `digest(&program)` equals the digest constant. They hold the same values in every later card; a card that changes either value has broken the default.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_0_32 cargo nextest run -p proxima-tensor -E 'test(/forward_scales::|head_repeats::|layer_taps_variant_matches/)'`
- expect: `7 passed` (the 2 new, the 4 `head_repeats` tests, and the layer-taps test)
- also green: `cargo clippy -p proxima-tensor --all-targets`
- stage: `proxima-tensor/src/spec/tests.rs`
- commit: `test(tensor): pin the single-range forward graph before scale knobs`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: edit any source file; run the capture on a tree carrying another card's edits; change a constant to make a later test pass
- gpu: none

### 0.33 Let a descriptor scale the embedding lookup by a literal

- id: FT0.33
- needs: FT0.32
- budget: 20 min
- crate(s): proxima-tensor
- read first:
  - `proxima-tensor/src/spec/attention_forward.rs::EmbeddingScale` (~112: one variant, `Sqrt`) and the same 12-line application written out three times: `lfm2_forward_program_with_experts_and_head_repeats` (~1803) and twice in `proxima-tensor/src/spec/lfm2_single_range_cached.rs` (~916 and ~1358);
  - `attention_forward.rs::mistral_cached_forward_program_with_experts_and_layer_taps_with_rope_pairing` (~2395: it reads `embedding_lookup` at ~2429 and never scales it, so the descriptor's `embedding_scale` is silently ignored by the single-range arm today) and the public wrapper `mistral_cached_forward_program_with_experts_and_layer_taps` (~2345);
  - `proxima-tensor/src/spec/descriptor.rs::FamilyProfile` (~152, derives `Eq`) and the `CacheStrategy::SingleRange` arm (~451).
- change:
  1. `proxima-tensor/src/spec/attention_forward.rs`: (a) `EmbeddingScale` gains `Factor(f32)` with doc "Multiply by a literal factor the checkpoint declares (`<family>.embedding_scale`, 12.0 for granite)"; drop `Eq` from its derive list (f32 has none). (b) Add `impl EmbeddingScale { #[must_use] pub fn multiplier(self, embedding: u32) -> f32 }` returning `(embedding as f32).sqrt()` for `Sqrt` and the factor for `Factor`. (c) Add `pub(crate) fn append_embedding_scale(program: &mut Vec<Op>, x: NodeId, embedding: u32, scale: EmbeddingScale) -> Result<NodeId, TensorError>`: `let multiplier = scalar_constant(program, scale.multiplier(embedding));` then `elementwise(program, DType::Float32, ScalarOp::Multiply, &[(x, "sd->sd"), (multiplier, "->sd")])`. (d) Replace the 12-line block at ~1804 by `if let Some(scale) = embedding_scale { x = append_embedding_scale(&mut program, x, embedding, scale)?; }` (node order is unchanged: constant first, then the multiply). (e) The `..._with_rope_pairing` builder gains a trailing parameter `embedding_scale: Option<EmbeddingScale>`; right after `let mut x = embedding_lookup(..)` (~2429) insert the same three-line `if let` block. (f) The public wrapper passes `None` as the new last argument.
  2. `proxima-tensor/src/spec/lfm2_single_range_cached.rs`: replace both 12-line blocks (~916 and ~1358) with the same three-line `if let` block.
  3. `proxima-tensor/src/spec/descriptor.rs`: drop `Eq` from `FamilyProfile`'s derive list (~150); the `SingleRange` arm passes `descriptor.embedding_scale` as the new last argument of the builder call (~490).
  4. The two direct calls of the builder in the test file (`proxima-tensor/src/spec/tests.rs`, ~725 and ~15090) get `None` appended, so each still compares graphs built with no scale.
- test: in `mod forward_scales` add the helper `fn constants_equal(program: &[Op], value: f32) -> usize` (the count of `Op::Constant { value: found, .. }` with `*found == value`), then three tests:
  - `embedding_factor_adds_one_constant_and_one_multiply`: `base = built(&descriptor(0, 0)).0`; a clone of the descriptor with `embedding_scale = Some(EmbeddingScale::Factor(12.0))` gives `scaled`; `scaled.len() == base.len() + 2`, `constants_equal(&scaled, 12.0) == 1`, `constants_equal(&base, 12.0) == 0`.
  - `embedding_sqrt_multiplies_by_the_square_root_of_the_width`: `Some(EmbeddingScale::Sqrt)` on the embedding width 8: `scaled.len() == base.len() + 2` and `constants_equal(&scaled, 8.0_f32.sqrt()) == 1`.
  - `profile_toml_can_carry_a_literal_embedding_factor`: `toml::from_str::<FamilyProfile>(&format!("embedding_scale = {{ Factor = 12.0 }}\n{}", profile_text("adjacent")))` gives `embedding_scale == Some(EmbeddingScale::Factor(12.0))`.
  The two pin tests and the four `head_repeats` tests (whose digests cover the two lfm2 engines that now share the helper) pass unchanged.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_0_33 cargo nextest run -p proxima-tensor -E 'test(/forward_scales::|head_repeats::|layer_taps_variant_matches/)'`
- expect: `10 passed`
- also green: `cargo clippy -p proxima-tensor --all-targets`
- stage: `proxima-tensor/src/spec/attention_forward.rs proxima-tensor/src/spec/lfm2_single_range_cached.rs proxima-tensor/src/spec/descriptor.rs proxima-tensor/src/spec/tests.rs`
- commit: `feat(tensor): let a descriptor scale the embedding lookup by a literal`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: change the order of nodes the lfm2 builders emit; touch `omega/` or `proxima-model-interop`; run rustfmt or `cargo fmt` on a source file
- gpu: none

### 0.34 Carry a logit scale in the descriptor and refuse the engines that cannot lower it

- id: FT0.34
- needs: FT0.33
- budget: 20 min
- crate(s): proxima-tensor
- read first:
  - `proxima-tensor/src/spec/descriptor.rs::ModelDescriptor` (~54; `logit_softcap` at ~81 is the sibling field, `head_repeats` at ~131 is the last), its other literals (`mistral_descriptor_from_shape` ~247, the literal at ~303, and `proxima-tensor/src/spec/gguf_descriptor.rs` ~133) and `::build_forward` (~387: three arms, `TwoRange` ~392, `Cacheless` ~422, `SingleRange` ~451);
  - `proxima-tensor/src/error.rs::TensorError::UnsupportedInBuilder` (~389: `{ builder: &'static str, feature: &'static str }`, the typed refusal the non-uniform-attention check at ~478 already returns).
- change:
  1. `proxima-tensor/src/spec/descriptor.rs`: `ModelDescriptor` gains, after `logit_softcap`, `pub logit_scale: Option<f32>` (doc: "Divisor on the final logits, `logits / logit_scale`: the checkpoint's `<family>.logit_scale`, 6.0 for granite; `None` leaves the logits untouched"). `mistral_descriptor_from_shape` sets it to `None`. Add a private `fn refuse_when(set: bool, builder: &'static str, feature: &'static str) -> Result<(), TensorError>` returning `Err(TensorError::UnsupportedInBuilder { builder, feature })` when `set`. At the top of the `TwoRange` arm add `refuse_when(descriptor.logit_scale.is_some(), "build_forward(CacheStrategy::TwoRange)", "a logit scale")?;`, at the top of the `Cacheless` arm the same with builder `"build_forward(CacheStrategy::Cacheless)"`, and in the `SingleRange` arm, after `let attention = first.attention;`, the same with builder `"build_forward(CacheStrategy::SingleRange)"`.
  2. `proxima-tensor/src/spec/gguf_descriptor.rs`: the `ModelDescriptor` literal (~133) gains `logit_scale: None`.
  3. `proxima-tensor/src/spec/tests.rs`: the two other `ModelDescriptor` literals (~14888 and the one inside `head_repeats::descriptor`, ~17504) gain `logit_scale: None`; in `mod head_repeats` change `fn descriptor` (~17482) to `pub(super) fn descriptor` (one word).
- test: add three tests in `mod forward_scales`:
  - `two_range_and_cacheless_refuse_a_logit_scale`: for each of `CacheStrategy::TwoRange` and `CacheStrategy::Cacheless`, `let mut descriptor = head_repeats::descriptor(strategy, 1);` set `logit_scale = Some(6.0)` and assert `build_forward(&descriptor, false)` matches `Err(TensorError::UnsupportedInBuilder { feature: "a logit scale", .. })`.
  - `single_range_refuses_a_logit_scale_until_it_lowers_one`: `descriptor(0, 0)` with `logit_scale = Some(6.0)` gives `Err(UnsupportedInBuilder { builder: "build_forward(CacheStrategy::SingleRange)", feature: "a logit scale" })`.
  - `new_descriptors_default_to_no_logit_scale`: `descriptor(0, 0)` has `logit_scale == None`.
  The pin tests and `embedding_*` tests still pass unchanged.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_0_34 cargo nextest run -p proxima-tensor -E 'test(/forward_scales::|head_repeats::|layer_taps_variant_matches/)'`
- expect: `13 passed` (8 in `forward_scales`, 4 in `head_repeats`, 1 layer-taps)
- also green: `cargo clippy -p proxima-tensor --all-targets`
- stage: `proxima-tensor/src/spec/descriptor.rs proxima-tensor/src/spec/gguf_descriptor.rs proxima-tensor/src/spec/tests.rs`
- commit: `feat(tensor): carry a logit scale and refuse engines that lack it`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: lower the scale in this card; change an engine's program; add a residual scale or a rope-pairing check (separate cards); add a field to `FamilyProfile`
- gpu: none

### 0.35 Divide the single-range logits by a descriptor scale

- id: FT0.35
- needs: FT0.34
- budget: 20 min
- crate(s): proxima-tensor
- read first:
  - `proxima-tensor/src/spec/attention_forward.rs::mistral_cached_forward_program_with_experts_and_layer_taps_with_rope_pairing` (~2395): the final `let logits = reduce(..)` (~2835) is the node `ForwardRoots::logits` names (~2847);
  - the gemma4 soft-cap epilogue in `lfm2_forward_program_with_experts_and_head_repeats` (the `match logit_softcap` after its own logits reduce, ~2134): the same `"sv->sv"` / `"->sv"` letter maps apply to a `[s, vocab]` logits node;
  - `proxima-tensor/src/spec/descriptor.rs` `SingleRange` arm (~451), where the previous card left a refusal of the logit scale.
- change:
  1. `proxima-tensor/src/spec/attention_forward.rs`: the builder gains a trailing parameter `logit_scale: Option<f32>`; immediately after its `let logits = reduce(..)?;` insert `let logits = match logit_scale { Some(scale) => { let inverse = scalar_constant(&mut program, 1.0 / scale); elementwise(&mut program, DType::Float32, ScalarOp::Multiply, &[(logits, "sv->sv"), (inverse, "->sv")])? } None => logits };` (the root the function returns is the new `logits`; `hidden` is unchanged). The public wrapper passes `None` as the new last argument.
  2. `proxima-tensor/src/spec/descriptor.rs`: in the `SingleRange` arm delete the `refuse_when` call for the logit scale and pass `descriptor.logit_scale` as the new last argument of the builder call. The `TwoRange` and `Cacheless` refusals stay.
  3. The two direct calls of the builder in the test file (`proxima-tensor/src/spec/tests.rs`, ~725 and ~15090) get `None` appended.
- test: in `mod forward_scales` delete `single_range_refuses_a_logit_scale_until_it_lowers_one` and add `logit_scale_divides_single_range_logits_by_the_scale`: `base_logits = built(&descriptor(0, 0)).1`; a clone with `logit_scale = Some(6.0)` gives `scaled` and `scaled_logits`; `scaled.len() == base.len() + 2`; `constants_equal(&scaled, 1.0_f32 / 6.0_f32) == 1` and `constants_equal(&base, 1.0_f32 / 6.0_f32) == 0`; `scaled_logits != base_logits` (the root moved to the new last node). The pin tests and `two_range_and_cacheless_refuse_a_logit_scale` pass unchanged.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_0_35 cargo nextest run -p proxima-tensor -E 'test(/forward_scales::|head_repeats::|layer_taps_variant_matches/)'`
- expect: `13 passed` (one test removed, one added)
- also green: `cargo clippy -p proxima-tensor --all-targets`
- stage: `proxima-tensor/src/spec/attention_forward.rs proxima-tensor/src/spec/descriptor.rs proxima-tensor/src/spec/tests.rs`
- commit: `feat(tensor): divide single-range logits by a descriptor scale`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: touch the gemma4 soft-cap code; apply the scale to `hidden`; add a logit scale to the two lfm2 engines
- gpu: none

### 0.36 Take the attention score scale from the descriptor

- id: FT0.36
- needs: FT0.35
- budget: 20 min
- crate(s): proxima-tensor
- read first:
  - `proxima-tensor/src/spec/attention_forward.rs::AttentionScoreScale` (~391), `build_attention_layer_resources` (~1425; its `match config.score_scale` at ~1450 is how the lfm2 engines read the scale) and the derive lists of `LayerAttentionConfig` (~412) and `LayerSchedule` (~456), the only containers that derive `Eq` over it (`ModelDescriptor` derives `PartialEq` only; `find_or_insert` needs `PartialEq` only);
  - the builder's `let inv_sqrt_head_dim = scalar_constant(.., 1.0 / (head_dim as f32).sqrt())` (~2434): built from `head_dim` alone, so a descriptor's `attention.score_scale` is silently ignored by the single-range arm today;
  - `proxima-tensor/src/spec/descriptor.rs` `SingleRange` arm (~451: `let attention = first.attention;`).
- change:
  1. `proxima-tensor/src/spec/attention_forward.rs`: (a) `AttentionScoreScale` gains `Factor(f32)` with doc "Multiply scores by a literal the checkpoint declares (`<family>.attention.scale`, 0.015625 for granite), replacing `1/sqrt(head_dim)`"; drop `Eq` from the derive lists of `AttentionScoreScale` (~390), `LayerAttentionConfig` (~412) and `LayerSchedule` (~456). (b) Add `impl AttentionScoreScale { #[must_use] pub fn multiplier(self) -> f32 }`: `InverseSqrtQueryPreAttnScalar(scalar)` gives `1.0 / (scalar as f32).sqrt()` (the same expression both call sites already evaluate), `Unscaled` gives `1.0`, `Factor(factor)` gives `factor`. (c) In `build_attention_layer_resources` replace the `match` inside the `find_or_insert` closure with `scalar_constant(program, config.score_scale.multiplier())`. (d) The builder gains a trailing parameter `score_scale: AttentionScoreScale`; the `inv_sqrt_head_dim` line becomes `scalar_constant(&mut program, score_scale.multiplier())`. The public wrapper passes `AttentionScoreScale::InverseSqrtQueryPreAttnScalar(head_dim)` as the new last argument.
  2. `proxima-tensor/src/spec/descriptor.rs`: the `SingleRange` arm passes `attention.score_scale` as the new last argument of the builder call.
  3. The two direct calls of the builder in the test file (`proxima-tensor/src/spec/tests.rs`, ~725 and ~15090) get `AttentionScoreScale::InverseSqrtQueryPreAttnScalar(64)` and `AttentionScoreScale::InverseSqrtQueryPreAttnScalar(REAL_HEAD_DIM)` appended respectively (each call's own head width).
- test: add two tests in `mod forward_scales`, both on `descriptor(0, 0)` (head width 4, so the default constant is `1/sqrt(4) = 0.5`) with every `layers[i].attention.score_scale` replaced:
  - `attention_factor_replaces_the_inverse_square_root_constant`: `Factor(0.015625)`: `scaled.len() == base.len()`, `constants_equal(&scaled, 0.015625) == 1`, `constants_equal(&scaled, 0.5) + 1 == constants_equal(&base, 0.5)`.
  - `unscaled_attention_uses_a_constant_of_one`: `Unscaled`: `scaled.len() == base.len()`, `constants_equal(&scaled, 1.0) == constants_equal(&base, 1.0) + 1`, `constants_equal(&scaled, 0.5) + 1 == constants_equal(&base, 0.5)`.
  The pin tests pass unchanged (the default `InverseSqrt` scale evaluates to the same bits as the constant it replaces).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_0_36 cargo nextest run -p proxima-tensor -E 'test(/forward_scales::|head_repeats::|layer_taps_variant_matches/)'`
- expect: `15 passed`
- also green: `cargo clippy -p proxima-tensor --all-targets`; `cargo check -p proxima-model-interop --features std --all-targets` (it builds `LayerAttentionConfig` in `src/lfm2.rs` ~626, and its tests, examples and benches build the same types; all must still compile without `Eq`)
- stage: `proxima-tensor/src/spec/attention_forward.rs proxima-tensor/src/spec/descriptor.rs proxima-tensor/src/spec/tests.rs`
- commit: `feat(tensor): take the attention score scale from the descriptor`
- done when: the expect line printed, clippy and the check clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: change the lfm2 engines' constants; add a per-head or per-layer scale rule; touch `proxima-model-interop` sources
- gpu: none

### 0.44 Carry a residual scale in the descriptor and refuse the engines that cannot lower it

- id: FT0.44
- needs: FT0.36
- budget: 20 min
- crate(s): proxima-tensor
- read first:
  - `proxima-tensor/src/spec/descriptor.rs::ModelDescriptor` (~54: the logit card put `logit_scale` right after `logit_softcap`, ~81; the new field goes right after it);
  - `proxima-tensor/src/spec/descriptor.rs::build_forward` (~387; FT0.34 adds a private `refuse_when` helper beside it, absent at a7c08c4c, so read it in the tree FT0.34 left) and its three arms (`TwoRange` ~392 and `Cacheless` ~422 each open with a `refuse_when` call for the logit scale; in `SingleRange` ~451 that call is gone, because the logit-lowering card removed it);
  - the `ModelDescriptor` literals the logit card extended: `proxima-tensor/src/spec/gguf_descriptor.rs` (~133) and, in `proxima-tensor/src/spec/tests.rs`, the one at ~14888 and the one inside `head_repeats::descriptor` (~17504);
  - `proxima-tensor/src/error.rs::TensorError::UnsupportedInBuilder` (~389).
- change:
  1. `proxima-tensor/src/spec/descriptor.rs`: `ModelDescriptor` gains, right after `logit_scale`, `pub residual_scale: Option<f32>` (doc: "Multiplier on each sublayer output before it joins the residual stream, `x + scale * sublayer`: the checkpoint's `<family>.residual_scale`, 0.22 for granite; `None` is the plain add"). `mistral_descriptor_from_shape` sets it to `None`. At the top of the `TwoRange` arm add `refuse_when(descriptor.residual_scale.is_some(), "build_forward(CacheStrategy::TwoRange)", "a residual scale")?;`, at the top of the `Cacheless` arm the same with builder `"build_forward(CacheStrategy::Cacheless)"`, and in the `SingleRange` arm, after `let attention = first.attention;`, the same with builder `"build_forward(CacheStrategy::SingleRange)"`.
  2. `proxima-tensor/src/spec/gguf_descriptor.rs`: the `ModelDescriptor` literal gains `residual_scale: None`.
  3. `proxima-tensor/src/spec/tests.rs`: the two other `ModelDescriptor` literals gain `residual_scale: None`.
- test: add three tests in `mod forward_scales`:
  - `two_range_and_cacheless_refuse_a_residual_scale`: for each of `CacheStrategy::TwoRange` and `CacheStrategy::Cacheless`, `let mut descriptor = head_repeats::descriptor(strategy, 1);` set `residual_scale = Some(0.22)` and assert `build_forward(&descriptor, false)` matches `Err(TensorError::UnsupportedInBuilder { feature: "a residual scale", .. })`.
  - `single_range_refuses_a_residual_scale_until_it_lowers_one`: `descriptor(0, 0)` and `descriptor(4, 2)`, each with `residual_scale = Some(0.22)`, give `Err(UnsupportedInBuilder { builder: "build_forward(CacheStrategy::SingleRange)", feature: "a residual scale" })`.
  - `new_descriptors_default_to_no_residual_scale`: `descriptor(0, 0)` has `residual_scale == None`.
  The pin tests, the `embedding_*` tests, the logit tests and the attention tests pass unchanged.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_0_44 cargo nextest run -p proxima-tensor -E 'test(/forward_scales::|head_repeats::|layer_taps_variant_matches/)'`
- expect: `18 passed` (13 in `forward_scales`, 4 in `head_repeats`, 1 layer-taps)
- also green: `cargo clippy -p proxima-tensor --all-targets`
- stage: `proxima-tensor/src/spec/descriptor.rs proxima-tensor/src/spec/gguf_descriptor.rs proxima-tensor/src/spec/tests.rs`
- commit: `feat(tensor): carry a residual scale and refuse engines that lack it`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: lower the scale in this card; change an engine's program; edit `attention_forward.rs` or `single_range_moe_cached.rs`; add a field to `FamilyProfile`
- gpu: none

### 0.37 Scale the single-range MoE layer's residual adds by a descriptor scale

- id: FT0.37
- needs: FT0.44
- budget: 20 min
- crate(s): proxima-tensor
- read first:
  - `proxima-tensor/src/spec/single_range_moe_cached.rs::append_mistral_cached_moe_layer` (~992): `residual1` (~1411) is `attn_out + x`, `x_next` (~1436) is `ffn_out + residual1`; `attn_out` is the `wo` reduce just above it (~1401) and `ffn_out` comes from `append_moe_ffn` (~1434);
  - `proxima-tensor/src/spec/attention_forward.rs` builder (~2395) and its two layer call sites (`append_mistral_cached_layer` ~2685 for dense, `append_mistral_cached_moe_layer` ~2774 for MoE); the dense layer (`mistral_forward_cached.rs::append_mistral_cached_layer`, ~472) is left alone on purpose: a dense layer with a residual scale is refused, not silently run unscaled;
  - `proxima-tensor/src/spec/descriptor.rs` `SingleRange` arm (~451), where an earlier card left a refusal of the residual scale.
- change:
  1. `proxima-tensor/src/spec/single_range_moe_cached.rs`: add `fn scale_residual(program: &mut Vec<Op>, branch: NodeId, scale: Option<NodeId>) -> Result<NodeId, TensorError>`: `Some(scale)` gives `elementwise(program, DType::Float32, ScalarOp::Multiply, &[(branch, "sd->sd"), (scale, "->sd")])`, `None` gives `Ok(branch)`. `append_mistral_cached_moe_layer` gains a trailing parameter `residual_scale: Option<NodeId>`; after the `attn_out` reduce and before `residual1` insert `let attn_out = scale_residual(program, attn_out, residual_scale)?;`; after `append_moe_ffn` and before `x_next` insert `let ffn_out = scale_residual(program, ffn_out, residual_scale)?;`.
  2. `proxima-tensor/src/spec/attention_forward.rs`: the builder gains a trailing parameter `residual_scale: Option<f32>`. First statement: `if residual_scale.is_some() && expert_count == 0 { return Err(TensorError::UnsupportedInBuilder { builder: "mistral_cached_forward_program_with_experts_and_layer_taps_with_rope_pairing", feature: "a residual scale on a dense layer" }); }`. Right after the `inv_head_dim` binding (~2438): `let residual_scale = residual_scale.map(|scale| scalar_constant(&mut program, scale));`. Pass `residual_scale` as the new last argument of the `append_mistral_cached_moe_layer` call (~2774); the dense call is unchanged. The public wrapper passes `None`.
  3. `proxima-tensor/src/spec/descriptor.rs`: in the `SingleRange` arm delete the `refuse_when` call for the residual scale and pass `descriptor.residual_scale` as the new last argument of the builder call.
  4. The two direct calls of the builder in the test file (`proxima-tensor/src/spec/tests.rs`, ~725 and ~15090) get `None` appended.
- test: in `mod forward_scales` delete `single_range_refuses_a_residual_scale_until_it_lowers_one` and add:
  - `moe_residual_scale_adds_one_constant_and_two_multiplies_per_layer`: `descriptor(4, 2)` with `residual_scale = Some(0.22)` (2 layers): `scaled.len() == base.len() + 5`, `constants_equal(&scaled, 0.22_f32) == 1`, `constants_equal(&base, 0.22_f32) == 0`.
  - `dense_single_range_refuses_a_residual_scale`: `descriptor(0, 0)` with `residual_scale = Some(0.22)` gives `Err(UnsupportedInBuilder { builder: "mistral_cached_forward_program_with_experts_and_layer_taps_with_rope_pairing", feature: "a residual scale on a dense layer" })`.
  The pin tests (both default graphs) pass unchanged.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_0_37 cargo nextest run -p proxima-tensor -E 'test(/forward_scales::|head_repeats::|layer_taps_variant_matches/)'`
- expect: `19 passed` (one test removed, two added)
- also green: `cargo clippy -p proxima-tensor --all-targets`
- stage: `proxima-tensor/src/spec/single_range_moe_cached.rs proxima-tensor/src/spec/attention_forward.rs proxima-tensor/src/spec/descriptor.rs proxima-tensor/src/spec/tests.rs`
- commit: `feat(tensor): scale moe layer residual adds by a descriptor scale`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: edit `mistral_forward_cached.rs`; scale the residual stream itself (only the two branches); touch the two lfm2 engines
- gpu: none

### 0.45 Refuse a rope pairing the single-range MoE layer cannot express

- id: FT0.45
- needs: FT0.37
- budget: 20 min
- crate(s): proxima-tensor
- read first:
  - `proxima-tensor/src/spec/single_range_moe_cached.rs::append_mistral_cached_moe_layer` (~992; the pairing comment at ~1075: the MoE layer picks split-half rope exactly when `qk_norm` is set and ignores the descriptor's `rope_pairing`, so a descriptor that asks for any other pairing would be built with the wrong rotation and no error);
  - `proxima-tensor/src/spec/descriptor.rs::build_forward`'s `SingleRange` arm (~451: `let attention = first.attention;` and the `refuse_when` calls the earlier cards left there) and `proxima-tensor/src/spec/primitives.rs::RopePairing` (~574: `Interleaved` and `SplitHalf { pairs }`);
  - `proxima-tensor/src/error.rs::TensorError::UnsupportedInBuilder` (~389).
- change:
  1. `proxima-tensor/src/spec/descriptor.rs`: in the `SingleRange` arm, after the existing `refuse_when` calls that follow `let attention = first.attention;`, add `let pairing_the_moe_layer_derives = if descriptor.qk_norm { RopePairing::SplitHalf { pairs: attention.head_dim / 2 } } else { RopePairing::Interleaved };` and `refuse_when(descriptor.expert_count > 0 && attention.rope_pairing != pairing_the_moe_layer_derives, "build_forward(CacheStrategy::SingleRange)", "a rope pairing the moe layer cannot express")?;`.
- test: add `single_range_moe_refuses_a_pairing_its_layer_cannot_express` in `mod forward_scales`: a MoE descriptor (`expert_count` 4, `expert_used_count` 2) built by calling `mistral_descriptor_from_shape` exactly as the module's `descriptor` helper does, from `profile_text("split_half")` and with the `qk_norm` argument `false`, gives `Err(UnsupportedInBuilder { feature: "a rope pairing the moe layer cannot express", .. })`; the same profile with `qk_norm` `true` lowers (`is_ok()`); the adjacent profile with `qk_norm` `false` lowers (`is_ok()`).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_0_45 cargo nextest run -p proxima-tensor -E 'test(/forward_scales::|head_repeats::|layer_taps_variant_matches/)'`
- expect: `20 passed` (15 in `forward_scales`, 4 in `head_repeats`, 1 layer-taps)
- also green: `cargo clippy -p proxima-tensor --all-targets`
- stage: `proxima-tensor/src/spec/descriptor.rs proxima-tensor/src/spec/tests.rs`
- commit: `fix(tensor): refuse a moe rope pairing the layer cannot express`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: change which pairing the MoE layer derives; touch `single_range_moe_cached.rs`; refuse a dense descriptor
- gpu: none

### 0.39 Vendor granite vocabulary ids from llama-tokenize and prove the existing refact rule on them

- id: FT0.39
- needs: none
- budget: 20 min
- crate(s): proxima-tokenizer (features: gguf)
- read first:
  - `proxima-tokenizer/src/pretokenize.rs::PreType::from_gguf_name` (~65 at a7c08c4c): its `"starcoder" | "refact" | ...` arm (~77) already maps `"refact"` to `PreType::DigitIsolatedGpt2`, and `::PreType` (~31) documents refact under that variant, so granite's `tokenizer.ggml.pre = "refact"` loads with no new rule; `proxima-tokenizer/src/gguf.rs::pre_type_from_metadata` (~152) returns that mapping or `UnsupportedPreTokenizer`;
  - `proxima-tokenizer/tests/pre_tokenizer_llama_oracle.rs` (the whole file: `load_vocab` ~31, `family_cases!` ~55, `assert_family_matches_llama` ~85, the per-family tests and blob constants ~19 to ~158) and `proxima-tokenizer/tests/fixtures/llama-pre-tokenize/README.md` (the exact `llama-tokenize` command and the family table; its last paragraphs already document the `starcoder/` and `refact/` families, whose tests are `#[ignore]` because their vocab-only GGUFs are not vendored, so granite's real vocabulary in the Ollama store is the first non-ignored test of this rule);
  - `proxima-tokenizer/tests/fixtures/llama-pre-tokenize/texts/` (the 13 shared input texts);
  - llama.cpp `src/llama-vocab.cpp` ~359-371 at f1ea20621: `LLAMA_VOCAB_PRE_TYPE_REFACT` uses the regexes `\p{N}` then the gpt2 word regex, and `tokenizer_pre == "refact"` selects it (~2252).
- change:
  1. `mkdir -p proxima-tokenizer/tests/fixtures/llama-pre-tokenize/granite_moe`. Then, from inside `proxima-tokenizer/tests/fixtures/llama-pre-tokenize`, for each of the 13 names in `texts/` (`arithmetic_with_contractions`, `code_with_numbers`, `combining_marks_with_digits`, `csv_rows`, `dates_times`, `digit_run_lengths`, `indic_thai_arabic_with_marks`, `large_and_decimal_numbers`, `non_ascii_digits`, `phone_numbers`, `prices_invoice`, `roman_numerals_and_circled_symbols`, `version_strings`) run `/Users/brianbruggeman/repos/slot-0/proxima-prefix-cache/scratchpad/bin/llama-f1ea2062/bin/llama-tokenize -m /Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80 --ids --log-disable --no-escape --no-bos -f texts/<name>.txt > granite_moe/<name>.ids`. The tool loads the vocabulary only, so no weights are read. Each file is one line, `[id, id, ...]` (measured on `csv_rows`: it opens `[314, 30, 15175, 30, 4415, 203, 35, 34, 34, 35, ...]`, digits one id each).
  2. `proxima-tokenizer/tests/fixtures/llama-pre-tokenize/README.md`: add the row `| granite_moe | Ollama blob sha256 cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80 (granite3.1-moe:1b) | `refact` | REFACT |` to the family table (the table has no row for this vocabulary; the existing refact paragraph below it is about the vocab-only file).
  3. `proxima-tokenizer/tests/pre_tokenizer_llama_oracle.rs`: add `const GRANITE_MOE_BLOB_SHA256: &str = "cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80";` beside the other blob constants.
- test: add `granite_refact_vocabulary_isolates_digits_like_llama` in `proxima-tokenizer/tests/pre_tokenizer_llama_oracle.rs`: `let vocab = load_vocab(&ollama_blob(GRANITE_MOE_BLOB_SHA256)); assert_family_matches_llama(&vocab, PreType::DigitIsolatedGpt2, &family_cases!("granite_moe"));` (13 cases; the helper asserts all 13 and the vocab's pre type). If any case differs, or the vocabulary fails to load, the test names it: that name and the first differing id pair, or the typed load error, is the finding to report, not a reason to change a fixture or to add a pre type.
- validate: `ls proxima-tokenizer/tests/fixtures/llama-pre-tokenize/granite_moe/*.ids | wc -l | tr -d ' ' && CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_0_39 cargo nextest run -p proxima-tokenizer --features gguf -E 'binary(pre_tokenizer_llama_oracle) & test(/granite/)'`
- expect: `13` then `1 passed`
- also green: `cargo clippy -p proxima-tokenizer --features gguf --all-targets`
- stage: the 13 files `proxima-tokenizer/tests/fixtures/llama-pre-tokenize/granite_moe/<name>.ids` (one per name listed in the change), `proxima-tokenizer/tests/fixtures/llama-pre-tokenize/README.md`, `proxima-tokenizer/tests/pre_tokenizer_llama_oracle.rs`
- commit: `test(tokenizer): vendor granite vocab ids from llama-tokenize`
- done when: the expect lines printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: edit any file under `proxima-tokenizer/src`; add a pre type or a second `"refact"` arm; edit an existing family's fixtures; regenerate a fixture to match proxima; start llama-server
- gpu: none (vocabulary-only load)

### 0.40 Load granitemoe from a family profile and the scales its header declares

- id: FT0.40
- needs: FT0.31, FT0.37
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/src/profiles/mod.rs::FAMILY_PROFILES` (~21), the tests `every_embedded_profile_parses` (~60), `every_profile_pairs_rope_as_llama_cpp_rope_type_says` (~104) and `real_header_metadata_selects_the_pairing_and_the_rotating_width` (~145, which reads `tests/fixtures/llama-parity/<name>/gguf_kv.txt` through `fixture_kv`), and `proxima-model-interop/src/profiles/mixtral.toml` (the adjacent-rope, softmax-routed, SiLU profile that granitemoe matches value for value);
  - `proxima-model-interop/src/dense.rs::DenseArch::bind` (~60: `family_profile(&architecture.family)`, then `mistral_descriptor_from_shape` at ~96 and `build_forward` at ~113) and the test helper `e2b_shaped` in `proxima-model-interop/src/gemma4/bind.rs` (~1050: builds a header with `GgufModel`, `write_complete` and `parse_complete`; `write_complete` accepts an empty tensor list);
  - `proxima-model-interop/src/bind.rs::metadata_f32_optional` (~852: the default for an absent or non-float key).
- change:
  1. New file `proxima-model-interop/src/profiles/granitemoe.toml`, exactly the contents of `mixtral.toml`: `score_scale_inverse_sqrt_head_dim = true`, `value_norm = false`, `rope_layout = "adjacent"`, a blank line, `[ffn]`, then `post_attention_norm = false`, `combination = "Exclusive"`, `output_scale = false`, `routed_gating = "Softmax"`, `routed_expert_bias = false`, `activation = "Silu"`, `exclusive_dense_post_norm = false`. (The score scale is replaced by the header's own value in step 3 when the header declares one.)
  2. `proxima-model-interop/src/profiles/mod.rs`: append `("granitemoe", include_str!("granitemoe.toml"))` to `FAMILY_PROFILES`; in `every_embedded_profile_parses` change the expected length from 8 to 9; add `("granitemoe", false)` to the table of `every_profile_pairs_rope_as_llama_cpp_rope_type_says` and a line to its doc comment: "GRANITE_MOE is NORM (`src/llama-model.cpp:3019`, the group returns at :3039)"; add `("granite_moe", RopePairing::Interleaved)` to the `cases` of `real_header_metadata_selects_the_pairing_and_the_rotating_width` (the fixture is the one the first granite card vendored).
  3. `proxima-model-interop/src/dense.rs`: import `AttentionScoreScale`, `EmbeddingScale`, `LayerAttentionConfig`, `LayerSchedule` and `ModelDescriptor` from `proxima_tensor::spec` and `metadata_f32_optional` from `crate::bind`. Add `fn header_scale(parsed: &ParsedGguf, family: &str, key: &str) -> Option<f32>`: `let value = metadata_f32_optional(parsed, &format!("{family}.{key}"), 0.0); (value != 0.0).then_some(value)` (llama.cpp treats 0.0 as unset). Add `fn with_header_scales(descriptor: ModelDescriptor, parsed: &ParsedGguf, family: &str) -> ModelDescriptor`: `layers` is `descriptor.layers` with every `attention.score_scale` replaced by `AttentionScoreScale::Factor(scale)` when `header_scale(.., "attention.scale")` is `Some(scale)`, and unchanged otherwise; the result is `ModelDescriptor { embedding_scale: header_scale(.., "embedding_scale").map(EmbeddingScale::Factor).or(descriptor.embedding_scale), logit_scale: header_scale(.., "logit_scale"), residual_scale: header_scale(.., "residual_scale"), layers, ..descriptor }`. In `bind`, right after `mistral_descriptor_from_shape(..)`, bind `let descriptor = with_header_scales(descriptor, parsed, &architecture.family);` so `build_forward` receives it. A family whose header carries none of the four keys gets the descriptor it got before.
- test: add four tests:
  - in `profiles/mod.rs`, `granitemoe_profile_is_adjacent_rope_softmax_routed_and_silu`: `family_profile("granitemoe")` has `rope_pairing(64) == RopePairing::Interleaved`, `ffn.routed_gating == ExpertGatingFunc::Softmax`, `ffn.activation == Activation::Silu`, `ffn.combination == FfnCombination::Exclusive` and `embedding_scale == None`.
  - in a new `#[cfg(test)]` module of `dense.rs` (same attribute lines as the `tests` module of `profiles/mod.rs`), with a helper that writes a header holding `general.architecture` plus the given `<family>.<key>` floats (zero tensors) and parses it:
    - `header_scales_reach_the_descriptor_of_a_granite_shaped_header`: `mistral_descriptor_from_shape(49155, 1024, 512, 16, 8, 64, 24, 32, 8, false, false, false, false, &family_profile("granitemoe").expect("profile embedded"))` (the real granite dims) and a header with `granitemoe.embedding_scale = 12.0`, `residual_scale = 0.22`, `logit_scale = 6.0`, `attention.scale = 0.015625`: the result has `embedding_scale == Some(EmbeddingScale::Factor(12.0))`, `logit_scale == Some(6.0)`, `residual_scale == Some(0.22)`, 24 layers each with `attention.score_scale == AttentionScoreScale::Factor(0.015625)`, and equals the input descriptor in every other field (compare `ModelDescriptor { embedding_scale: input.embedding_scale, logit_scale: None, residual_scale: None, layers: input.layers.clone(), ..result.clone() } == input`).
    - `a_header_without_scale_keys_leaves_the_descriptor_unchanged`: a descriptor from `mistral_descriptor_from_shape(32000, 4096, 14336, 32, 8, 128, 32, 0, 0, false, false, false, false, &family_profile("llama").expect("profile embedded"))` and a header with only `general.architecture = "llama"`: the result equals the input.
    - `a_zero_scale_means_unset_like_llama_cpp`: the same input with a header carrying `llama.residual_scale = 0.0` and `llama.logit_scale = 0.0`: the result equals the input.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_0_40 cargo nextest run -p proxima-model-interop --features std -E 'test(/granitemoe_profile_is_adjacent|every_embedded_profile_parses|every_profile_pairs_rope_as_llama_cpp_rope_type_says|real_header_metadata_selects_the_pairing|header_scales_reach_the_descriptor|a_header_without_scale_keys|a_zero_scale_means_unset/)'`
- expect: `7 passed` (the four new tests and the three edited ones)
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: `proxima-model-interop/src/profiles/granitemoe.toml proxima-model-interop/src/profiles/mod.rs proxima-model-interop/src/dense.rs`
- commit: `feat(interop): load granitemoe from a profile and its header scales`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a granite branch anywhere; add a family name outside `profiles/`; change what `llama`, `mistral`, `mixtral` or `qwen*` load (a header without the four keys must produce the same descriptor)
- gpu: none

### 0.41 Vendor granite greedy ids from llama-server

- id: FT0.41
- needs: FT0.4, FT0.31
- budget: 20 min
- crate(s): none (fixtures)
- read first:
  - `proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/llama_ids.json` (record shape: top-level array; each record has `prompt`, `prompt_ids`, `generated_ids`, `llama_commit` `"f1ea20621"` and `command`; the three prompts are `The capital of France is`, `def fibonacci(n):\n` and the "Rivers carry silt ..." paragraph);
  - `proxima-model-interop/tests/arch_data_baseline.rs::llama_parity` (~668) and `::llama_cases` (~617): the consumer of this shape (it tokenizes `prompt` with proxima and generates from `prompt_ids`);
  - measured while cutting, llama-server f1ea20621 with `-c 4096 -np 1`: tokenizing the three prompts with `add_special: true` gives 5, 6 and 72 ids (granite adds no BOS: its header has `tokenizer.ggml.add_bos_token = False`), and `n_predict 32, temperature 0` generates 32 tokens each (`stop_type limit`).
- change:
  1. Start one server: `source proxima-tensor/specs/fsm-techniques/env.sh` (defines `$LOGS` and `$LLAMA`; FT0.4 writes that file); `ps -axo comm | grep -E "decode_gbps|census|llama-server|decode_arms|norm_variant"` must print nothing (wait while it prints a match); `mkdir -p $LOGS/FT0.41`; `$LLAMA -m /Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80 -ngl 99 -c 4096 -np 1 --port 18171 > $LOGS/FT0.41/granite_moe.server.log 2>&1 &` keeping `$!`; wait on `http://127.0.0.1:18171/health`.
  2. For each prompt `p` of `proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/llama_ids.json` (read with `jq -r ".[i].prompt"`, `i` in 0..3): `ids=$(jq -n --arg c "$p" '{content: $c, add_special: true}' | curl -s localhost:18171/tokenize -d @- | jq -c .tokens)`; then `curl -s localhost:18171/completion -d '{"prompt": <ids>, "n_predict": 32, "temperature": 0, "cache_prompt": false, "stream": false, "return_tokens": true}'` and keep `.tokens` as the generated ids.
  3. Write `proxima-model-interop/tests/fixtures/llama-parity/granite_moe/llama_ids.json`: one array of the 3 records `{"prompt": p, "prompt_ids": ids, "generated_ids": tokens, "llama_commit": "f1ea20621", "command": "llama-server -m /Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80 -ngl 99 -c 4096 -np 1 --port 18171"}`.
  4. `kill $!; wait $!`; confirm no llama-server process remains.
- test: none; the validate command asserts the fixture.
- validate: `jq -e -n --slurpfile granite proxima-model-interop/tests/fixtures/llama-parity/granite_moe/llama_ids.json --slurpfile e2b proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/llama_ids.json '($granite[0]|length) == 3 and (($granite[0]|map(.prompt)) == ($e2b[0]|map(.prompt))) and ($granite[0]|all(.[]; (.prompt_ids|length) >= 1 and (.generated_ids|length) == 32))' | grep -c true`
- expect: `1`
- also green: n/a
- stage: `proxima-model-interop/tests/fixtures/llama-parity/granite_moe/llama_ids.json`
- commit: `test(llama-parity): vendor granite moe greedy ids from llama-server`
- done when: the expect line printed, one new file, no llama-server process left running, and the commit landed with that message
- do not: edit another checkpoint's `llama_ids.json`; change the prompts; start a second server; write logs under /tmp
- gpu: one run, waiting for a quiet box (the peer-gate check in CARDS "machine safety")

### 0.42 Baseline the granite moe graph digest

- id: FT0.42
- needs: FT0.40
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/tests/arch_data_baseline.rs`: the module doc (`PROXIMA_ARCH_DATA_CAPTURE=1` rewrites the expected files from the current build), `digest_record` (~210), `assert_matches_fixture` (~318: with the capture variable set it writes the file and returns) and `arch_data_digest_gemma4_26b` (~351: asserts structural lines before comparing the fixture);
  - `proxima-model-interop/tests/fixtures/llama-parity/openchat.digest` (the dense record: `registry_entry=dense`, `bind.single_position_step=false`, `verify=absent`).
  This baseline is a consistency check of the production path, never a correctness oracle; the token oracle is the parity card after the recorded ids.
- change:
  1. `proxima-model-interop/tests/arch_data_baseline.rs`: add the test below.
  2. Capture, in one invocation of one test (this is the card's single model-loading run): the validate command below, with the capture variable set, writes `proxima-model-interop/tests/fixtures/llama-parity/granite_moe.digest` after the structural assertions pass. The bound-weights record is the next card's; the comparison of both files against a fresh build, without the capture variable, is the slice exit.
- test: add `arch_data_digest_granite_moe` in `proxima-model-interop/tests/arch_data_baseline.rs`: `let record = digest_record(&GRANITE_MOE);` assert it contains `"\nregistry_entry=dense\n"`, `"\nbind.residual_roots=24 sha256="`, `"\nbind.layer_roots=24 sha256="`, `"\nbind.router_roots=0 sha256="` and `"\nbind.single_position_step=false\n"` and ends with `"verify=absent\n"`; then `assert_matches_fixture(&GRANITE_MOE, "digest", &record)`.
- validate: `PROXIMA_ARCH_DATA_CAPTURE=1 CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_0_42 cargo nextest run -p proxima-model-interop --features std -j 1 -E 'binary(arch_data_baseline) & test(/^arch_data_digest_granite_moe$/)'; grep -c '^bind.ops=' proxima-model-interop/tests/fixtures/llama-parity/granite_moe.digest`
- expect: `1 passed` then `1`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: `proxima-model-interop/tests/arch_data_baseline.rs proxima-model-interop/tests/fixtures/llama-parity/granite_moe.digest`
- commit: `test(llama-parity): baseline the granite moe graph digest`
- done when: the expect lines printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: edit another checkpoint's digest or bound file; write `granite_moe.bound`; commit a fixture captured from a tree with a failing structural assertion; run two model-loading processes at once
- gpu: one run (one test process, one CPU digest of a 1.4 GB checkpoint: the digest binds the program and then its verify program), waiting for a quiet box

### 0.56 Baseline the granite moe bound weights

- id: FT0.56
- needs: FT0.42
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/tests/arch_data_baseline.rs`: the module doc (`PROXIMA_ARCH_DATA_CAPTURE=1` rewrites the expected files from the current build), `bound_record` (~290), `assert_bound` (~345) and `assert_matches_fixture` (~318: with the capture variable set it writes the file and returns); the card before this one added `arch_data_digest_granite_moe`;
  - `proxima-model-interop/tests/fixtures/llama-parity/openchat.bound` (the record shape: `checkpoint=`, `bound_weights=`, a header line, then one tab-separated line per weight).
  This baseline is a consistency check of the production path, never a correctness oracle; the token oracle is the parity card after the recorded ids.
- change:
  1. `proxima-model-interop/tests/arch_data_baseline.rs`: add the test below.
  2. Capture, in one invocation of one test (this is the card's single model-loading run): the validate command below, with the capture variable set, writes `proxima-model-interop/tests/fixtures/llama-parity/granite_moe.bound`.
- test: add `generic_binder_granite_moe` in `proxima-model-interop/tests/arch_data_baseline.rs`: `let record = bound_record(&GRANITE_MOE); assert!(record.contains("\nbound_weights=243\n"), "granite moe must bind 243 weights (24 layers x 10 + token_embd + output_norm + output), got {:?}", record.lines().nth(1)); assert_matches_fixture(&GRANITE_MOE, "bound", &record);` (the shape of `arch_data_digest_gemma4_26b`, which asserts its op count before `assert_matches_fixture`). The assertion runs before `assert_matches_fixture`, so it fires under `PROXIMA_ARCH_DATA_CAPTURE=1` too. 243 is derived from reading code, not measured: `bind_all_weights` (`proxima-model-interop/src/bind.rs` ~2153, reached from `dense.rs` `bind` ~60 with `paired_gate_up_reduce` and `fused_qkv_reduce` both false) pushes `token_embd.weight` once, then per layer `attn_norm`, `ffn_norm`, `attn_q`, `attn_k`, `attn_v`, `attn_output`, `ffn_gate_inp` and, through `bind_moe_expert_weights` (one entry per projection whether the checkpoint stores stacked `_exps` tensors or per-expert tensors), `ffn_gate`, `ffn_up`, `ffn_down` = 10 entries, then `output_norm.weight`, then one `output.weight` entry (an on-disk tensor, or the `token_embd.weight` alias of a tied checkpoint via `bind_matmul_weight_as`): 24 x 10 + 3 = 243. The same rule gives 291 for openchat (32 layers x 9 + 3, `openchat.bound` line 2). It holds only if granite has no qkv bias and no q/k norm tensors (`checkpoint_qkv_biases` and `checkpoint_has_qk_norm` false); a printed count other than 243 means the derivation missed a tensor family, and the executor reports the printed count and the tensor names in `granite_moe.bound` instead of editing 243 to match.
- validate: `PROXIMA_ARCH_DATA_CAPTURE=1 CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_0_56 cargo nextest run -p proxima-model-interop --features std -j 1 -E 'binary(arch_data_baseline) & test(/^generic_binder_granite_moe$/)'; grep '^bound_weights=' proxima-model-interop/tests/fixtures/llama-parity/granite_moe.bound; wc -l < proxima-model-interop/tests/fixtures/llama-parity/granite_moe.bound | tr -d ' '`
- expect: `1 passed`, then `bound_weights=243`, then `246` (the 3 header lines plus 243 weight lines)
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: `proxima-model-interop/tests/arch_data_baseline.rs proxima-model-interop/tests/fixtures/llama-parity/granite_moe.bound`
- commit: `test(llama-parity): baseline the granite moe bound weights`
- done when: the expect lines printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: edit another checkpoint's digest or bound file; write `granite_moe.digest`; run two model-loading processes at once
- gpu: one run (one test process, one CPU bind of a 1.4 GB checkpoint), waiting for a quiet box

### 0.55 Prove the granite moe program carries the header scales

- id: FT0.55
- needs: FT0.56
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/tests/arch_data_baseline.rs`: `resolve` (~190: the registry lookup the digest test uses), `describe_program` (~154) and `assert_architecture_key` (~200); the two cards before this one added `arch_data_digest_granite_moe` and `generic_binder_granite_moe`;
  - `proxima-tensor/src/spec/tests.rs::head_repeats` (~17466 at a7c08c4c: the test-module style to follow; FT0.33 adds an eight-line `constants_equal` helper in a sibling module, absent at a7c08c4c, and a test file cannot import it, so this file types the body given in the change list below).
- change:
  1. `proxima-model-interop/tests/arch_data_baseline.rs`: add `fn constants_equal(program: &[Op], value: f32) -> usize` (the count of `Op::Constant { value: found, .. }` with `*found == value`) and the test below.
- test: add `granite_moe_program_carries_the_header_scales` in `proxima-model-interop/tests/arch_data_baseline.rs`: map the checkpoint, `resolve` it through the registry, `bind_with_kv_layout(.., KvLayout::SlidingRing)`, then assert on `bound.program`: `constants_equal(.., 12.0) == 1`, `constants_equal(.., 1.0_f32 / 6.0_f32) == 1`, `constants_equal(.., 0.22_f32) == 1` and `constants_equal(.., 0.015625) == 1` (the four values the real header declares, read through the descriptor, not typed into the builder).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_0_55 cargo nextest run -p proxima-model-interop --features std -j 1 -E 'binary(arch_data_baseline) & test(/^granite_moe_program_carries_the_header_scales$/)'`
- expect: `1 passed` (the new test only; the two baseline tests are compared against their committed fixtures by the slice exit)
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: `proxima-model-interop/tests/arch_data_baseline.rs`
- commit: `test(llama-parity): check granite moe program carries header scales`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: set the capture variable; edit the digest or bound fixtures; run two model-loading processes at once
- gpu: one run (one test process, one CPU bind of a 1.4 GB checkpoint), waiting for a quiet box

### 0.43 Compare granite moe tokens with the recorded llama.cpp ids

- id: FT0.43
- needs: FT0.4, FT0.39, FT0.41, FT0.55
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `proxima-model-interop/tests/arch_data_baseline.rs::llama_parity` (~668): for each recorded case it tokenizes the prompt with proxima and compares the ids with `prompt_ids` (a `TOKENIZER` line on the first divergent index), then `generate_from_ids` for 32 tokens and compares with `generated_ids` over the shorter length (a `MODEL` line), and fails with every divergence listed;
  - `llama_parity_gemma4_e2b` (~731): the one-line test shape to copy.
- change:
  1. `proxima-model-interop/tests/arch_data_baseline.rs`: add `#[test] fn llama_parity_granite_moe() { llama_parity(&GRANITE_MOE); }` after `llama_parity_gemma4_e2b`.
- test: `llama_parity_granite_moe`: asserts, for the 3 recorded prompts, that proxima's ids for each prompt equal the recorded `prompt_ids` (the existing `refact` pre-split rule, which the granite vocabulary oracle card checked) and that the first 32 generated ids equal the recorded `generated_ids` (each record holds 32; `llama_cases` panics on zero records, so the count is 3 by construction). A `TOKENIZER` line names a prompt the pre-split oracle did not cover; a `MODEL` line gives the first divergent index per prompt. Either is the finding to report with the printed line; do not widen a comparison, skip a prompt or edit a fixture.
- validate: `ps -axo comm | grep -E "decode_gbps|census|llama-server|decode_arms|norm_variant"` (must print nothing), then `source proxima-tensor/specs/fsm-techniques/env.sh && mkdir -p $LOGS/FT0.43 && CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_0_43 cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'binary(arch_data_baseline) & test(/^llama_parity_granite_moe$/)' > $LOGS/FT0.43/run.log 2>&1; echo "exit=$?"; grep -E "Summary" $LOGS/FT0.43/run.log`
- expect: `exit=0` and a summary line reporting `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/tests/arch_data_baseline.rs`
- commit: `test(llama-parity): compare granite moe tokens with recorded llama ids`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: run another model-loading process; add a granite branch to `src/`; add a tolerance (token ids compare exactly)
- gpu: one run, waiting for a quiet box

## oracle fixtures from llama-server f1ea20621

Cards 0.46, 0.47, 0.5 (follow-up turns), 0.48, 0.49, 0.6 (top-2 log probabilities) and 0.50 to 0.54, 0.7 (shifted chunk reuse) each run exactly one llama-server session, for one checkpoint, and commit one fixture file (the three unshifted-generation cards 0.51, 0.53 and 0.7 add one key to the file the reuse card before them wrote). Order inside each group is gemma4_e2b, then gemma4_26b, then granite_moe, and the last card of a group needs the earlier ones, so a `needs: FT0.5`, `FT0.6` or `FT0.7` in another file means the fixtures of all three checkpoints exist. Each card carries every command it needs. The rules every one of them follows, from the checkout holding main after `source proxima-tensor/specs/fsm-techniques/env.sh`:
- before the server starts: `ps -axo comm | grep -E "decode_gbps|census|llama-server|decode_arms|norm_variant"` must print nothing; wait while it prints a match (CARDS.md machine safety). This machine has kernel-panicked twice from concurrent model loads: never two servers at once;
- start: `$LLAMA -m <blob> -ngl 99 -c 8192 -np 1 <flags> --port <port> > $LOGS/<card>/server.log 2>&1 &`, keep `$!`, then wait on `http://127.0.0.1:<port>/health`; stop: `kill $!; wait $!`, and the `ps` line above prints nothing again;
- fixture records follow the shape of the existing `proxima-model-interop/tests/fixtures/llama-parity/<ckpt>/llama_ids.json`: JSON, integer-array id fields, plus `"llama_commit": "f1ea20621"` and `"command"`. `command` is the literal server command line the card gives, with `$LLAMA` written as `llama-server` and the blob as its absolute path, and without the `> ... server.log 2>&1 &` redirect (the redirect names the card's log directory, and a card id never enters a committed fixture). Files are written with `jq` and committed with the card.

Prompt rendering, decided from measurement. gemma4_e2b and granite_moe use the raw text (measured coherent: E2B continues the passage, granite continues it). gemma4_26b uses the gemma4 chat template with thinking off, `<|turn>user\n` + text + `<turn|>\n<|turn>model\n<|channel>thought\n<channel|>` (the form the architecture-as-data SPEC measured on the 26B, read at a7c08c4c), tokenized with `"add_special": true, "parse_special": true` so the BOS and the turn markers are ids. Reason, measured on lines 3000-3010 of `war_and_peace.txt` through llama-server f1ea20621: on the raw text the 26B degenerates (its 32 greedy tokens read `mthemuch-appreciated much-appreciated much-appreciated ...`, and the gap between its top two log probabilities falls to 0.084 nats), while through the template it answers coherently. Near-ties remain either way (smallest top-2 gaps measured: 26B chat 0.027 nats, E2B raw 0.022, E2B chat 0.021), so the follow-up and reuse fixtures also record `n_probs 2` for the request a later card compares, letting a first divergence be read against the margin at that step.

### 0.46 Vendor followup_ids.json for gemma4_e2b

- id: FT0.46
- needs: FT0.4
- budget: 20 min
- crate(s): none (fixtures)
- read first:
  - `proxima-model-interop/examples/speculative_bench.rs::run_llama_greedy_ids` (~line 2124 at a7c08c4c: the request shape, `prompt` as an id array, `n_predict`, `temperature` 0, `return_tokens` true);
  - `/Users/brianbruggeman/repos/others/llama.cpp/tools/server/server-common.cpp` ~86 (`cache_n` is in the response `timings` object) and `src/llama-kv-cache-iswa.cpp::get_can_shift` (~253: the reason a sliding-window model needs `--swa-full` for its slot to keep a long common prefix);
  - measured while cutting (llama-server f1ea20621, `-c 8192 -np 1 --swa-full`): raw text lines 3000-3010 of `war_and_peace.txt` (373 bytes) tokenizes to 106 ids and generates 32 tokens (`stop_type limit`); the suffix ` Why does that matter?` is `[8922, 1677, 600, 4217, 236881]`; turn 2 reports `cache_n = 137` (and `101` without `--swa-full`, below the 106 prompt ids). Adding `n_probs 2` to a request leaves the generated ids unchanged (checked on this checkpoint).
- change:
  1. `mkdir -p $LOGS/FT0.46`; `sed -n 3000,3010p proxima-model-interop/examples/data/war_and_peace.txt > $LOGS/FT0.46/turn1.txt`; `printf ' Why does that matter?' > $LOGS/FT0.46/suffix.txt`; the quiet-box check; start `$LLAMA -m $GEMMA4_E2B -ngl 99 -c 8192 -np 1 --swa-full --port 18141 > $LOGS/FT0.46/server.log 2>&1 &` and wait on `/health`.
  2. Turn-1 ids: `ids1=$(jq -Rs '{content: ., add_special: true}' $LOGS/FT0.46/turn1.txt | curl -s localhost:18141/tokenize -d @- | jq -c .tokens)`.
  3. Turn 1: `curl -s localhost:18141/completion -d '{"prompt": <ids1>, "n_predict": 32, "temperature": 0, "cache_prompt": true, "id_slot": 0, "stream": false, "return_tokens": true}'`. Keep `.tokens` as `gen1` and `.content` as `content1`.
  4. Suffix ids: `jq -Rs '{content: ., add_special: false}' $LOGS/FT0.46/suffix.txt` through `/tokenize`; keep `.tokens` as `suffix`.
  5. Turn 2 prompt ids = `ids1 + gen1 + suffix` (jq array concatenation). POST the same `/completion` body with that prompt plus `"n_probs": 2`. Keep `.tokens` as `gen2`, `.content` as `content2`, `.timings.cache_n` as `cache_n`, and `.completion_probabilities` as `steps2` in the step shape `{"id", "logprob", "top": [{"id", "logprob"}, {"id", "logprob"}]}` (`id` and `logprob` from the element, `top` from its first two `top_logprobs`).
  6. Write `proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/followup_ids.json` as a one-element array: `{"render": "raw", "source": "war_and_peace.txt lines 3000-3010", "turn1": {"prompt_ids": ids1, "generated_ids": gen1, "content": content1}, "suffix_text": " Why does that matter?", "suffix_ids": suffix, "turn2": {"prompt_ids": ids2, "generated_ids": gen2, "content": content2, "cache_n": cache_n, "steps": steps2}, "llama_commit": "f1ea20621", "command": "llama-server -m /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd -ngl 99 -c 8192 -np 1 --swa-full --port 18141"}`.
  7. Stop the server.
- test: none; the validate command asserts the fixture. If turn 1 or turn 2 generates fewer than 32 tokens, or `cache_n` is below the turn-1 prompt length, the premise is false: stop and report the numbers.
- validate: `jq -e '.[0] | (.turn2.cache_n >= (.turn1.prompt_ids|length)) and ((.turn1.generated_ids|length) == 32) and ((.turn2.generated_ids|length) == 32) and ((.turn2.steps|length) == 32) and ([.turn2.steps[].id] == .turn2.generated_ids)' proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/followup_ids.json | grep -c true`
- expect: `1`
- also green: n/a
- stage: `proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/followup_ids.json`
- commit: `test(llama-parity): vendor gemma4 e2b follow-up turn ids`
- done when: the expect line printed, `git status --short` shows only that file as new, no llama-server process is left running, and the commit landed with that message
- do not: edit an existing `llama_ids.json`; start a second server; write logs under /tmp; drop `--swa-full`; put a card id or a log path in the fixture
- gpu: one run (1 server session), waiting for a quiet box

### 0.47 Vendor followup_ids.json for gemma4_26b

- id: FT0.47
- needs: FT0.4, FT0.46
- budget: 20 min
- crate(s): none (fixtures)
- read first:
  - `proxima-model-interop/examples/speculative_bench.rs::run_llama_greedy_ids` (~line 2124 at a7c08c4c: the request shape);
  - `/Users/brianbruggeman/repos/others/llama.cpp/tools/server/server-common.cpp` ~86 (`cache_n` is in `timings`) and `src/llama-kv-cache-iswa.cpp::get_can_shift` (~253: why `--swa-full`);
  - measured while cutting (llama-server f1ea20621, `-c 8192 -np 1 --swa-full`), chat-rendered as defined above: turn 1 is 118 ids (opening `[2, 105, 2364, 107, 236913]`) and generates 32 tokens; the 19-id suffix is `[106, 107, 105, 2364, 107, 8922, 1677, 600, 4217, 236881, 106, 107, 105, 4368, 107, 100, 45518, 107, 101]`; the turn-2 prompt is 169 ids; `cache_n = 149` (118 + 32 - 1); turn 2 generates 32 tokens, coherent.
- change:
  1. `mkdir -p $LOGS/FT0.47`; `{ printf '<|turn>user\n'; sed -n 3000,3010p proxima-model-interop/examples/data/war_and_peace.txt; printf '<turn|>\n<|turn>model\n<|channel>thought\n<channel|>'; } > $LOGS/FT0.47/turn1.txt`; `printf '<turn|>\n<|turn>user\n Why does that matter?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>' > $LOGS/FT0.47/suffix.txt`; the quiet-box check; start `$LLAMA -m $GEMMA4_26B -ngl 99 -c 8192 -np 1 --swa-full --port 18142 > $LOGS/FT0.47/server.log 2>&1 &` and wait on `/health`.
  2. Turn-1 ids: `ids1=$(jq -Rs '{content: ., add_special: true, parse_special: true}' $LOGS/FT0.47/turn1.txt | curl -s localhost:18142/tokenize -d @- | jq -c .tokens)`.
  3. Turn 1: `curl -s localhost:18142/completion -d '{"prompt": <ids1>, "n_predict": 32, "temperature": 0, "cache_prompt": true, "id_slot": 0, "stream": false, "return_tokens": true}'`. Keep `.tokens` as `gen1` and `.content` as `content1`.
  4. Suffix ids: `jq -Rs '{content: ., add_special: false, parse_special: true}' $LOGS/FT0.47/suffix.txt` through `/tokenize`; keep `.tokens` as `suffix`.
  5. Turn 2 prompt ids = `ids1 + gen1 + suffix`. POST the same `/completion` body with that prompt plus `"n_probs": 2`. Keep `.tokens` as `gen2`, `.content` as `content2`, `.timings.cache_n` as `cache_n`, and `.completion_probabilities` as `steps2` in the step shape `{"id", "logprob", "top": [{"id", "logprob"}, {"id", "logprob"}]}`.
  6. Write `proxima-model-interop/tests/fixtures/llama-parity/gemma4_26b/followup_ids.json` as a one-element array: `{"render": "gemma4_chat_thinking_off", "source": "war_and_peace.txt lines 3000-3010", "turn1": {"prompt_ids": ids1, "generated_ids": gen1, "content": content1}, "suffix_text": <the suffix text>, "suffix_ids": suffix, "turn2": {"prompt_ids": ids2, "generated_ids": gen2, "content": content2, "cache_n": cache_n, "steps": steps2}, "llama_commit": "f1ea20621", "command": "llama-server -m /Users/brianbruggeman/.ollama/models/blobs/sha256-ea549b7688d4c95019754880c21e3f29c58c985a7a1c3b37b9eebd0a95224129 -ngl 99 -c 8192 -np 1 --swa-full --port 18142"}`.
  7. Stop the server.
- test: none; the validate command asserts the fixture. If turn 1 or turn 2 generates fewer than 32 tokens, or `cache_n` is below the turn-1 prompt length, the premise is false: stop and report the numbers.
- validate: `jq -e '.[0] | (.turn2.cache_n >= (.turn1.prompt_ids|length)) and ((.turn1.generated_ids|length) == 32) and ((.turn2.generated_ids|length) == 32) and ((.turn2.steps|length) == 32) and ([.turn2.steps[].id] == .turn2.generated_ids)' proxima-model-interop/tests/fixtures/llama-parity/gemma4_26b/followup_ids.json | grep -c true`
- expect: `1`
- also green: n/a
- stage: `proxima-model-interop/tests/fixtures/llama-parity/gemma4_26b/followup_ids.json`
- commit: `test(llama-parity): vendor gemma4 26b follow-up turn ids`
- done when: the expect line printed, `git status --short` shows only that file as new, no llama-server process is left running, and the commit landed with that message
- do not: edit an existing `llama_ids.json`; start a second server; write logs under /tmp; drop `--swa-full`; render the 26B prompt raw; put a card id or a log path in the fixture
- gpu: one run (1 server session), waiting for a quiet box

### 0.5 Vendor followup_ids.json for granite_moe

- id: FT0.5
- needs: FT0.4, FT0.31, FT0.47
- budget: 20 min
- crate(s): none (fixtures)
- read first:
  - `proxima-model-interop/examples/speculative_bench.rs::run_llama_greedy_ids` (~line 2124 at a7c08c4c: the request shape);
  - `/Users/brianbruggeman/repos/others/llama.cpp/tools/server/server-common.cpp` ~86 (`cache_n` is in `timings`);
  - measured while cutting (llama-server f1ea20621, `-c 8192 -np 1`, no `--swa-full`: granite has no sliding window): raw text lines 3000-3010 tokenizes to 120 ids (no BOS) and generates 32 tokens; the suffix ` Why does that matter?` is `[16239, 1957, 688, 15236, 49]`; turn 2 reports `cache_n = 151`.
- change:
  1. `mkdir -p $LOGS/FT0.5`; `sed -n 3000,3010p proxima-model-interop/examples/data/war_and_peace.txt > $LOGS/FT0.5/turn1.txt`; `printf ' Why does that matter?' > $LOGS/FT0.5/suffix.txt`; the quiet-box check; start `$LLAMA -m $GRANITE_MOE -ngl 99 -c 8192 -np 1 --port 18143 > $LOGS/FT0.5/server.log 2>&1 &` and wait on `/health`.
  2. Turn-1 ids: `ids1=$(jq -Rs '{content: ., add_special: true}' $LOGS/FT0.5/turn1.txt | curl -s localhost:18143/tokenize -d @- | jq -c .tokens)`.
  3. Turn 1: `curl -s localhost:18143/completion -d '{"prompt": <ids1>, "n_predict": 32, "temperature": 0, "cache_prompt": true, "id_slot": 0, "stream": false, "return_tokens": true}'`. Keep `.tokens` as `gen1` and `.content` as `content1`.
  4. Suffix ids: `jq -Rs '{content: ., add_special: false}' $LOGS/FT0.5/suffix.txt` through `/tokenize`; keep `.tokens` as `suffix`.
  5. Turn 2 prompt ids = `ids1 + gen1 + suffix`. POST the same `/completion` body with that prompt plus `"n_probs": 2`. Keep `.tokens` as `gen2`, `.content` as `content2`, `.timings.cache_n` as `cache_n`, and `.completion_probabilities` as `steps2` in the step shape `{"id", "logprob", "top": [{"id", "logprob"}, {"id", "logprob"}]}`.
  6. Write `proxima-model-interop/tests/fixtures/llama-parity/granite_moe/followup_ids.json` as a one-element array: `{"render": "raw", "source": "war_and_peace.txt lines 3000-3010", "turn1": {"prompt_ids": ids1, "generated_ids": gen1, "content": content1}, "suffix_text": " Why does that matter?", "suffix_ids": suffix, "turn2": {"prompt_ids": ids2, "generated_ids": gen2, "content": content2, "cache_n": cache_n, "steps": steps2}, "llama_commit": "f1ea20621", "command": "llama-server -m /Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80 -ngl 99 -c 8192 -np 1 --port 18143"}`.
  7. Stop the server.
- test: none; the validate command asserts the fixture. If turn 1 or turn 2 generates fewer than 32 tokens, or `cache_n` is below the turn-1 prompt length, the premise is false: stop and report the numbers.
- validate: `jq -e '.[0] | (.turn2.cache_n >= (.turn1.prompt_ids|length)) and ((.turn1.generated_ids|length) == 32) and ((.turn2.generated_ids|length) == 32) and ((.turn2.steps|length) == 32) and ([.turn2.steps[].id] == .turn2.generated_ids)' proxima-model-interop/tests/fixtures/llama-parity/granite_moe/followup_ids.json | grep -c true`
- expect: `1`
- also green: n/a
- stage: `proxima-model-interop/tests/fixtures/llama-parity/granite_moe/followup_ids.json`
- commit: `test(llama-parity): vendor granite moe follow-up turn ids`
- done when: the expect line printed, `git status --short` shows only that file as new, no llama-server process is left running, and the commit landed with that message
- do not: edit an existing `llama_ids.json`; start a second server; write logs under /tmp; put a card id or a log path in the fixture
- gpu: one run (1 server session), waiting for a quiet box

### 0.48 Vendor n_probs.json for gemma4_e2b

- id: FT0.48
- needs: FT0.4
- budget: 20 min
- crate(s): none (fixtures)
- read first:
  - `proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/llama_ids.json` (3 records with `prompt_ids`). Recorded generation lengths: 3, 32, 1 (it stops at the end-of-generation token);
  - `/Users/brianbruggeman/repos/others/llama.cpp/tools/server/server-task.cpp::completion_token_output::probs_vector_to_json` (~282 at f1ea20621: with `post_sampling_probs` unset each step is `{id, token, bytes, logprob, top_logprobs: [{id, token, bytes, logprob}, ...]}` and the response field is `completion_probabilities`);
  - measured while cutting on this checkpoint with `n_probs 2, cache_prompt false`: `completion_probabilities` has exactly as many steps as `tokens` (3, 32, 1), each has 2 `top_logprobs`, and `top_logprobs[0].id` equals the step's `id` in every step.
- change:
  1. `mkdir -p $LOGS/FT0.48`; the quiet-box check; start `$LLAMA -m $GEMMA4_E2B -ngl 99 -c 8192 -np 1 --port 18151 > $LOGS/FT0.48/server.log 2>&1 &` and wait on `/health`.
  2. For each of the 3 records `i` of `proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/llama_ids.json`, POST `/completion` with `{"prompt": <prompt_ids of record i>, "n_predict": 32, "temperature": 0, "n_probs": 2, "cache_prompt": false, "stream": false, "return_tokens": true}` and build the record `{"prompt_ids": ..., "generated_ids": .tokens, "steps": [ {"id": .id, "logprob": .logprob, "top": [ {"id": .top_logprobs[0].id, "logprob": .top_logprobs[0].logprob}, {"id": .top_logprobs[1].id, "logprob": .top_logprobs[1].logprob} ] } for each element of .completion_probabilities ], "llama_commit": "f1ea20621", "command": "llama-server -m /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd -ngl 99 -c 8192 -np 1 --port 18151"}`.
  3. Write the 3 records as one array to `proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/n_probs.json`. Stop the server.
- test: none; the validate command asserts the fixture.
- validate: `jq -e 'length == 3 and all(.[]; ((.steps|length) == (.generated_ids|length)) and (.steps|length) >= 1 and (.steps|length) <= 32 and ([.steps[].id] == .generated_ids) and all(.steps[]; (.top|length) == 2 and .top[0].id == .id))' proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/n_probs.json | grep -c true`
- expect: `1`
- also green: n/a
- stage: `proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/n_probs.json`
- commit: `test(llama-parity): vendor gemma4 e2b top-2 log probabilities`
- done when: the expect line printed, 1 new file, no llama-server process left running, and the commit landed with that message
- do not: edit an existing fixture; use `post_sampling_probs` (the readout compares raw-softmax log probabilities); pad a short record to 32 steps; put a card id or a log path in the fixture
- gpu: one run (1 server session), waiting for a quiet box

### 0.49 Vendor n_probs.json for gemma4_26b

- id: FT0.49
- needs: FT0.4, FT0.48
- budget: 20 min
- crate(s): none (fixtures)
- read first:
  - `proxima-model-interop/tests/fixtures/llama-parity/gemma4_26b/llama_ids.json` (3 records with `prompt_ids`, read at a7c08c4c: chat-templated prompts, `<|turn>user\nThe capital of France is<turn|>...`, `What is 17 plus 25?` and `Name the largest planet in our solar system.`, kept because llama's top-1 minus top-2 margin is at least 1.0 nats at every generated step (`min_margin_nats` 2.318, 4.898 and 10.403 per the architecture-as-data SPEC); recorded generation lengths 9, 12 and 12 ids, each ending at the end-of-generation token, so the step count here is those lengths, not 32). Because the margins are wide, this fixture is the one whose top-2 gap is read against a divergence; the tolerance is still on log probabilities;
  - `/Users/brianbruggeman/repos/others/llama.cpp/tools/server/server-task.cpp::completion_token_output::probs_vector_to_json` (~282 at f1ea20621: the response field is `completion_probabilities`, each step `{id, token, bytes, logprob, top_logprobs: [...]}`).
- change:
  1. `mkdir -p $LOGS/FT0.49`; the quiet-box check; start `$LLAMA -m $GEMMA4_26B -ngl 99 -c 8192 -np 1 --port 18152 > $LOGS/FT0.49/server.log 2>&1 &` and wait on `/health`.
  2. For each of the 3 records `i` of `proxima-model-interop/tests/fixtures/llama-parity/gemma4_26b/llama_ids.json`, POST `/completion` with `{"prompt": <prompt_ids of record i>, "n_predict": 32, "temperature": 0, "n_probs": 2, "cache_prompt": false, "stream": false, "return_tokens": true}` and build the record `{"prompt_ids": ..., "generated_ids": .tokens, "steps": [ {"id": .id, "logprob": .logprob, "top": [ {"id": .top_logprobs[0].id, "logprob": .top_logprobs[0].logprob}, {"id": .top_logprobs[1].id, "logprob": .top_logprobs[1].logprob} ] } for each element of .completion_probabilities ], "llama_commit": "f1ea20621", "command": "llama-server -m /Users/brianbruggeman/.ollama/models/blobs/sha256-ea549b7688d4c95019754880c21e3f29c58c985a7a1c3b37b9eebd0a95224129 -ngl 99 -c 8192 -np 1 --port 18152"}`.
  3. Write the 3 records as one array to `proxima-model-interop/tests/fixtures/llama-parity/gemma4_26b/n_probs.json`. Stop the server.
- test: none; the validate command asserts the fixture.
- validate: `jq -e 'length == 3 and all(.[]; ((.steps|length) == (.generated_ids|length)) and (.steps|length) >= 1 and (.steps|length) <= 32 and ([.steps[].id] == .generated_ids) and all(.steps[]; (.top|length) == 2 and .top[0].id == .id))' proxima-model-interop/tests/fixtures/llama-parity/gemma4_26b/n_probs.json | grep -c true`
- expect: `1`
- also green: n/a
- stage: `proxima-model-interop/tests/fixtures/llama-parity/gemma4_26b/n_probs.json`
- commit: `test(llama-parity): vendor gemma4 26b top-2 log probabilities`
- done when: the expect line printed, 1 new file, no llama-server process left running, and the commit landed with that message
- do not: edit an existing fixture; use `post_sampling_probs`; pad a short record to 32 steps; put a card id or a log path in the fixture
- gpu: one run (1 server session), waiting for a quiet box

### 0.6 Vendor n_probs.json for granite_moe

- id: FT0.6
- needs: FT0.4, FT0.41, FT0.49
- budget: 20 min
- crate(s): none (fixtures)
- read first:
  - `proxima-model-interop/tests/fixtures/llama-parity/granite_moe/llama_ids.json` (3 records with `prompt_ids`, the file the recorded-ids card vendored; recorded generation lengths 32, 32, 32);
  - `/Users/brianbruggeman/repos/others/llama.cpp/tools/server/server-task.cpp::completion_token_output::probs_vector_to_json` (~282 at f1ea20621: the response field is `completion_probabilities`, each step `{id, token, bytes, logprob, top_logprobs: [...]}`).
- change:
  1. `mkdir -p $LOGS/FT0.6`; the quiet-box check; start `$LLAMA -m $GRANITE_MOE -ngl 99 -c 8192 -np 1 --port 18153 > $LOGS/FT0.6/server.log 2>&1 &` and wait on `/health`.
  2. For each of the 3 records `i` of `proxima-model-interop/tests/fixtures/llama-parity/granite_moe/llama_ids.json`, POST `/completion` with `{"prompt": <prompt_ids of record i>, "n_predict": 32, "temperature": 0, "n_probs": 2, "cache_prompt": false, "stream": false, "return_tokens": true}` and build the record `{"prompt_ids": ..., "generated_ids": .tokens, "steps": [ {"id": .id, "logprob": .logprob, "top": [ {"id": .top_logprobs[0].id, "logprob": .top_logprobs[0].logprob}, {"id": .top_logprobs[1].id, "logprob": .top_logprobs[1].logprob} ] } for each element of .completion_probabilities ], "llama_commit": "f1ea20621", "command": "llama-server -m /Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80 -ngl 99 -c 8192 -np 1 --port 18153"}`.
  3. Write the 3 records as one array to `proxima-model-interop/tests/fixtures/llama-parity/granite_moe/n_probs.json`. Stop the server.
- test: none; the validate command asserts the fixture.
- validate: `jq -e 'length == 3 and all(.[]; ((.steps|length) == (.generated_ids|length)) and (.steps|length) >= 1 and (.steps|length) <= 32 and ([.steps[].id] == .generated_ids) and all(.steps[]; (.top|length) == 2 and .top[0].id == .id))' proxima-model-interop/tests/fixtures/llama-parity/granite_moe/n_probs.json | grep -c true`
- expect: `1`
- also green: n/a
- stage: `proxima-model-interop/tests/fixtures/llama-parity/granite_moe/n_probs.json`
- commit: `test(llama-parity): vendor granite moe top-2 log probabilities`
- done when: the expect line printed, 1 new file, no llama-server process left running, and the commit landed with that message
- do not: edit an existing fixture; use `post_sampling_probs`; pad a short record to 32 steps; put a card id or a log path in the fixture
- gpu: one run (1 server session), waiting for a quiet box

### 0.50 Vendor cache_reuse_ids.json for gemma4_e2b

- id: FT0.50
- needs: FT0.4
- budget: 20 min
- crate(s): none (fixtures)
- read first:
  - `/Users/brianbruggeman/repos/others/llama.cpp/tools/server/server-context.cpp` ~3206-3262 at f1ea20621: `can_cache_reuse` needs `llama_memory_can_shift` (~3208); the chunk scan (~3217) starts at `n_past`, the common prefix with the cached prompt, and for the first non-matching position of the new prompt slides over the cache looking for a run of at least `n_cache_reuse` equal tokens, then shifts it with `seq_add`. So a chunk is reused only when the new prompt equals the cached prompt with a middle span removed;
  - `src/llama-kv-cache-iswa.cpp::get_can_shift` (~253: needs the base and sliding caches the same size, which `--swa-full` gives);
  - `proxima-model-interop/examples/data/war_and_peace.txt` (65656 lines; prose well after the contents pages);
  - measured while cutting (`--cache-reuse 64 --swa-full -c 8192 -np 1`): token counts P (lines 3000-3010) 106, R (lines 4000-4020) 350, M (lines 2000-2040) 448; request 1 on `P+R+M`, then request 2 on `P+M` (554 ids) reports `cache_n = 553`, one less than the request-2 prompt length, with no `cache reuse is not supported` line in the log. The old arrangement (prefix `A1` then `M`, then a different prefix `A2` then `M`) measured `cache_n = 1` of 798: no reuse.
- change:
  1. `mkdir -p $LOGS/FT0.50`; `sed -n 3000,3010p proxima-model-interop/examples/data/war_and_peace.txt > $LOGS/FT0.50/prefix.txt`; `sed -n 4000,4020p ... > $LOGS/FT0.50/removed.txt`; `sed -n 2000,2040p ... > $LOGS/FT0.50/moved.txt`; the quiet-box check; start `$LLAMA -m $GEMMA4_E2B -ngl 99 -c 8192 -np 1 --cache-reuse 64 --swa-full --port 18161 > $LOGS/FT0.50/server.log 2>&1 &` and wait on `/health`.
  2. Tokenize through `POST /tokenize` with `jq -Rs`: the prefix with `{content: ., add_special: true}`, the removed and moved texts with `{content: ., add_special: false}`. Stop and report if the moved text has fewer than 200 ids.
  3. `ids1 = tokens(prefix) + tokens(removed) + tokens(moved)`, `ids2 = tokens(prefix) + tokens(moved)` (jq array concatenation).
  4. Request 1: POST `/completion` `{"prompt": ids1, "n_predict": 32, "temperature": 0, "cache_prompt": true, "id_slot": 0, "stream": false, "return_tokens": true}`; keep `.tokens` as `gen1`. Request 2 (on the same server, right after request 1, because its slot holds request 1): the same body with `ids2` plus `"n_probs": 2`; keep `.tokens` as `gen2`, `.timings.cache_n` as `cache_n` and `.completion_probabilities` as `steps2` in the step shape `{"id", "logprob", "top": [{"id", "logprob"}, {"id", "logprob"}]}`.
  5. The server log must contain no line `cache reuse is not supported`, and `cache_n` must equal `len(ids2) - 1`. If either fails, stop and report the log path and the number; write no file.
  6. Write `proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/cache_reuse_ids.json` as a one-element array: `{"render": "raw", "min_chunk": 64, "text_lines": {"prefix": "3000-3010", "removed": "4000-4020", "moved": "2000-2040"}, "request1": {"prompt_ids": ids1, "generated_ids": gen1}, "request2": {"prompt_ids": ids2, "generated_ids": gen2, "cache_n": cache_n, "steps": steps2}, "llama_commit": "f1ea20621", "command": "llama-server -m /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd -ngl 99 -c 8192 -np 1 --cache-reuse 64 --swa-full --port 18161"}`. Stop the server.
- test: none; the validate command asserts the fixture and the log.
- validate: `jq -e '.[0] | (.request2.cache_n == ((.request2.prompt_ids|length) - 1)) and ((.request1.generated_ids|length) == 32) and ((.request2.generated_ids|length) == 32) and ((.request2.steps|length) == 32) and ((.request2.prompt_ids|length) < (.request1.prompt_ids|length))' proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/cache_reuse_ids.json | grep -c true; grep -c 'cache reuse is not supported' $LOGS/FT0.50/server.log`
- expect: `1` then `0`
- also green: n/a
- stage: `proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/cache_reuse_ids.json`
- commit: `test(llama-parity): vendor gemma4 e2b shifted chunk reuse ids`
- done when: the expect lines printed, 1 new file, no llama-server process left running, and the commit landed with that message
- do not: change `--cache-reuse 64`; run request 2 on a fresh server; use the old two-prefix arrangement; put a card id or a log path in the fixture; add `request2_plain` (the next card does)
- gpu: one run (1 server session), waiting for a quiet box

### 0.51 Add the unshifted generation to cache_reuse_ids.json for gemma4_e2b

- id: FT0.51
- needs: FT0.50
- budget: 20 min
- crate(s): none (fixtures)
- read first:
  - `proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/cache_reuse_ids.json` (written by the previous card: `.[0].request2.prompt_ids` is the prompt both generations use);
  - the point of the key: `request2.generated_ids` came from a slot whose cache was shifted, `request2_plain` is the same prompt generated by a server with no chunk reuse, so a reader sees what the shift changed (a later slice reads `.[0].request2_plain.generated_ids`).
- change:
  1. `mkdir -p $LOGS/FT0.51`; the quiet-box check; start a fresh server without `--cache-reuse`: `$LLAMA -m $GEMMA4_E2B -ngl 99 -c 8192 -np 1 --swa-full --port 18161 > $LOGS/FT0.51/server.log 2>&1 &` and wait on `/health`.
  2. `ids2=$(jq -c '.[0].request2.prompt_ids' proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/cache_reuse_ids.json)`; POST `/completion` `{"prompt": <ids2>, "n_predict": 32, "temperature": 0, "cache_prompt": false, "stream": false, "return_tokens": true}` once; keep `.tokens` as `gen2_plain`. Stop the server.
  3. Add the key to the same file with `jq`: `.[0].request2_plain = {"prompt_ids": ids2, "generated_ids": gen2_plain, "command": "llama-server -m /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd -ngl 99 -c 8192 -np 1 --swa-full --port 18161"}`; write to a temporary file under `$LOGS/FT0.51/` and move it over the fixture (never redirect jq onto its own input).
- test: none; the validate command asserts the fixture.
- validate: `jq -e '.[0] | (.request2.cache_n == ((.request2.prompt_ids|length) - 1)) and ((.request2.generated_ids|length) == 32) and (.request2_plain.prompt_ids == .request2.prompt_ids) and ((.request2_plain.generated_ids|length) == 32)' proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/cache_reuse_ids.json | grep -c true`
- expect: `1`
- also green: n/a
- stage: `proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/cache_reuse_ids.json`
- commit: `test(llama-parity): vendor gemma4 e2b unshifted generation`
- done when: the expect line printed, `git diff --cached --stat` shows only that file, no llama-server process left running, and the commit landed with that message
- do not: run this on the reuse server; pass `--cache-reuse`; change `request1` or `request2`; put a card id or a log path in the fixture
- gpu: one run (1 server session), waiting for a quiet box

### 0.52 Vendor cache_reuse_ids.json for gemma4_26b

- id: FT0.52
- needs: FT0.4, FT0.51
- budget: 20 min
- crate(s): none (fixtures)
- read first:
  - `/Users/brianbruggeman/repos/others/llama.cpp/tools/server/server-context.cpp` ~3206-3262 at f1ea20621: `can_cache_reuse` needs `llama_memory_can_shift` (~3208); the chunk scan (~3217) starts at the common prefix with the cached prompt and slides over the cache for a run of at least `n_cache_reuse` equal tokens, then shifts it. A chunk is reused only when the new prompt equals the cached prompt with a middle span removed;
  - `src/llama-kv-cache-iswa.cpp::get_can_shift` (~253: `--swa-full`);
  - `proxima-model-interop/examples/data/war_and_peace.txt` (65656 lines);
  - measured while cutting (`--cache-reuse 64 --swa-full -c 8192 -np 1`), chat-rendered as defined above: prefix `P'` 109 ids, `R` 350, moved text `M'` 457; request-2 prompt 566 ids; `cache_n = 565`; no warning; coherent answer.
- change:
  1. `mkdir -p $LOGS/FT0.52`; `{ printf '<|turn>user\n'; sed -n 3000,3010p proxima-model-interop/examples/data/war_and_peace.txt; } > $LOGS/FT0.52/prefix.txt`; `sed -n 4000,4020p ... > $LOGS/FT0.52/removed.txt`; `{ sed -n 2000,2040p ...; printf '<turn|>\n<|turn>model\n<|channel>thought\n<channel|>'; } > $LOGS/FT0.52/moved.txt`; the quiet-box check; start `$LLAMA -m $GEMMA4_26B -ngl 99 -c 8192 -np 1 --cache-reuse 64 --swa-full --port 18162 > $LOGS/FT0.52/server.log 2>&1 &` and wait on `/health`.
  2. Tokenize through `POST /tokenize` with `jq -Rs`, all three with `parse_special: true`: the prefix with `add_special: true`, the removed and moved texts with `add_special: false`. Stop and report if the moved text has fewer than 200 ids.
  3. `ids1 = tokens(prefix) + tokens(removed) + tokens(moved)`, `ids2 = tokens(prefix) + tokens(moved)`.
  4. Request 1: POST `/completion` `{"prompt": ids1, "n_predict": 32, "temperature": 0, "cache_prompt": true, "id_slot": 0, "stream": false, "return_tokens": true}`; keep `.tokens` as `gen1`. Request 2 (same server, right after request 1): the same body with `ids2` plus `"n_probs": 2`; keep `.tokens` as `gen2`, `.timings.cache_n` as `cache_n`, `.completion_probabilities` as `steps2` in the step shape `{"id", "logprob", "top": [{"id", "logprob"}, {"id", "logprob"}]}`.
  5. The server log must contain no line `cache reuse is not supported`, and `cache_n` must equal `len(ids2) - 1`. If either fails, stop and report the log path and the number; write no file.
  6. Write `proxima-model-interop/tests/fixtures/llama-parity/gemma4_26b/cache_reuse_ids.json` as a one-element array: `{"render": "gemma4_chat_thinking_off", "min_chunk": 64, "text_lines": {"prefix": "3000-3010", "removed": "4000-4020", "moved": "2000-2040"}, "request1": {"prompt_ids": ids1, "generated_ids": gen1}, "request2": {"prompt_ids": ids2, "generated_ids": gen2, "cache_n": cache_n, "steps": steps2}, "llama_commit": "f1ea20621", "command": "llama-server -m /Users/brianbruggeman/.ollama/models/blobs/sha256-ea549b7688d4c95019754880c21e3f29c58c985a7a1c3b37b9eebd0a95224129 -ngl 99 -c 8192 -np 1 --cache-reuse 64 --swa-full --port 18162"}`. Stop the server.
- test: none; the validate command asserts the fixture and the log.
- validate: `jq -e '.[0] | (.request2.cache_n == ((.request2.prompt_ids|length) - 1)) and ((.request1.generated_ids|length) == 32) and ((.request2.generated_ids|length) == 32) and ((.request2.steps|length) == 32) and ((.request2.prompt_ids|length) < (.request1.prompt_ids|length))' proxima-model-interop/tests/fixtures/llama-parity/gemma4_26b/cache_reuse_ids.json | grep -c true; grep -c 'cache reuse is not supported' $LOGS/FT0.52/server.log`
- expect: `1` then `0`
- also green: n/a
- stage: `proxima-model-interop/tests/fixtures/llama-parity/gemma4_26b/cache_reuse_ids.json`
- commit: `test(llama-parity): vendor gemma4 26b shifted chunk reuse ids`
- done when: the expect lines printed, 1 new file, no llama-server process left running, and the commit landed with that message
- do not: change `--cache-reuse 64`; run request 2 on a fresh server; use the old two-prefix arrangement; render the 26B prompt raw; put a card id or a log path in the fixture; add `request2_plain` (the next card does)
- gpu: one run (1 server session), waiting for a quiet box

### 0.53 Add the unshifted generation to cache_reuse_ids.json for gemma4_26b

- id: FT0.53
- needs: FT0.52
- budget: 20 min
- crate(s): none (fixtures)
- read first:
  - `proxima-model-interop/tests/fixtures/llama-parity/gemma4_26b/cache_reuse_ids.json` (written by the previous card: `.[0].request2.prompt_ids` is the prompt both generations use);
  - the point of the key: `request2.generated_ids` came from a slot whose cache was shifted, `request2_plain` is the same prompt generated by a server with no chunk reuse.
- change:
  1. `mkdir -p $LOGS/FT0.53`; the quiet-box check; start a fresh server without `--cache-reuse`: `$LLAMA -m $GEMMA4_26B -ngl 99 -c 8192 -np 1 --swa-full --port 18162 > $LOGS/FT0.53/server.log 2>&1 &` and wait on `/health`.
  2. `ids2=$(jq -c '.[0].request2.prompt_ids' proxima-model-interop/tests/fixtures/llama-parity/gemma4_26b/cache_reuse_ids.json)`; POST `/completion` `{"prompt": <ids2>, "n_predict": 32, "temperature": 0, "cache_prompt": false, "stream": false, "return_tokens": true}` once; keep `.tokens` as `gen2_plain`. Stop the server.
  3. Add the key to the same file with `jq`: `.[0].request2_plain = {"prompt_ids": ids2, "generated_ids": gen2_plain, "command": "llama-server -m /Users/brianbruggeman/.ollama/models/blobs/sha256-ea549b7688d4c95019754880c21e3f29c58c985a7a1c3b37b9eebd0a95224129 -ngl 99 -c 8192 -np 1 --swa-full --port 18162"}`; write to a temporary file under `$LOGS/FT0.53/` and move it over the fixture (never redirect jq onto its own input).
- test: none; the validate command asserts the fixture.
- validate: `jq -e '.[0] | (.request2.cache_n == ((.request2.prompt_ids|length) - 1)) and ((.request2.generated_ids|length) == 32) and (.request2_plain.prompt_ids == .request2.prompt_ids) and ((.request2_plain.generated_ids|length) == 32)' proxima-model-interop/tests/fixtures/llama-parity/gemma4_26b/cache_reuse_ids.json | grep -c true`
- expect: `1`
- also green: n/a
- stage: `proxima-model-interop/tests/fixtures/llama-parity/gemma4_26b/cache_reuse_ids.json`
- commit: `test(llama-parity): vendor gemma4 26b unshifted generation`
- done when: the expect line printed, `git diff --cached --stat` shows only that file, no llama-server process left running, and the commit landed with that message
- do not: run this on the reuse server; pass `--cache-reuse`; change `request1` or `request2`; put a card id or a log path in the fixture
- gpu: one run (1 server session), waiting for a quiet box

### 0.54 Vendor cache_reuse_ids.json for granite_moe

- id: FT0.54
- needs: FT0.4, FT0.31, FT0.53
- budget: 20 min
- crate(s): none (fixtures)
- read first:
  - `/Users/brianbruggeman/repos/others/llama.cpp/tools/server/server-context.cpp` ~3206-3262 at f1ea20621: `can_cache_reuse` needs `llama_memory_can_shift` (~3208); the chunk scan (~3217) starts at the common prefix with the cached prompt and slides over the cache for a run of at least `n_cache_reuse` equal tokens, then shifts it. A chunk is reused only when the new prompt equals the cached prompt with a middle span removed;
  - `proxima-model-interop/examples/data/war_and_peace.txt` (65656 lines);
  - measured while cutting (`--cache-reuse 64 -c 8192 -np 1`, no `--swa-full`): token counts P (lines 3000-3010) 120, R (lines 4000-4020) 419, M (lines 2000-2040) 535 (no BOS); request 2 on `P+M` (655 ids) reports `cache_n = 654`, with no `cache reuse is not supported` line in the log.
- change:
  1. `mkdir -p $LOGS/FT0.54`; `sed -n 3000,3010p proxima-model-interop/examples/data/war_and_peace.txt > $LOGS/FT0.54/prefix.txt`; `sed -n 4000,4020p ... > $LOGS/FT0.54/removed.txt`; `sed -n 2000,2040p ... > $LOGS/FT0.54/moved.txt`; the quiet-box check; start `$LLAMA -m $GRANITE_MOE -ngl 99 -c 8192 -np 1 --cache-reuse 64 --port 18163 > $LOGS/FT0.54/server.log 2>&1 &` and wait on `/health`.
  2. Tokenize through `POST /tokenize` with `jq -Rs`: the prefix with `{content: ., add_special: true}`, the removed and moved texts with `{content: ., add_special: false}`. Stop and report if the moved text has fewer than 200 ids.
  3. `ids1 = tokens(prefix) + tokens(removed) + tokens(moved)`, `ids2 = tokens(prefix) + tokens(moved)`.
  4. Request 1: POST `/completion` `{"prompt": ids1, "n_predict": 32, "temperature": 0, "cache_prompt": true, "id_slot": 0, "stream": false, "return_tokens": true}`; keep `.tokens` as `gen1`. Request 2 (same server, right after request 1): the same body with `ids2` plus `"n_probs": 2`; keep `.tokens` as `gen2`, `.timings.cache_n` as `cache_n`, `.completion_probabilities` as `steps2` in the step shape `{"id", "logprob", "top": [{"id", "logprob"}, {"id", "logprob"}]}`.
  5. The server log must contain no line `cache reuse is not supported`, and `cache_n` must equal `len(ids2) - 1`. If either fails, stop and report the log path and the number; write no file.
  6. Write `proxima-model-interop/tests/fixtures/llama-parity/granite_moe/cache_reuse_ids.json` as a one-element array: `{"render": "raw", "min_chunk": 64, "text_lines": {"prefix": "3000-3010", "removed": "4000-4020", "moved": "2000-2040"}, "request1": {"prompt_ids": ids1, "generated_ids": gen1}, "request2": {"prompt_ids": ids2, "generated_ids": gen2, "cache_n": cache_n, "steps": steps2}, "llama_commit": "f1ea20621", "command": "llama-server -m /Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80 -ngl 99 -c 8192 -np 1 --cache-reuse 64 --port 18163"}`. Stop the server.
- test: none; the validate command asserts the fixture and the log.
- validate: `jq -e '.[0] | (.request2.cache_n == ((.request2.prompt_ids|length) - 1)) and ((.request1.generated_ids|length) == 32) and ((.request2.generated_ids|length) == 32) and ((.request2.steps|length) == 32) and ((.request2.prompt_ids|length) < (.request1.prompt_ids|length))' proxima-model-interop/tests/fixtures/llama-parity/granite_moe/cache_reuse_ids.json | grep -c true; grep -c 'cache reuse is not supported' $LOGS/FT0.54/server.log`
- expect: `1` then `0`
- also green: n/a
- stage: `proxima-model-interop/tests/fixtures/llama-parity/granite_moe/cache_reuse_ids.json`
- commit: `test(llama-parity): vendor granite moe shifted chunk reuse ids`
- done when: the expect lines printed, 1 new file, no llama-server process left running, and the commit landed with that message
- do not: change `--cache-reuse 64`; run request 2 on a fresh server; use the old two-prefix arrangement; put a card id or a log path in the fixture; add `request2_plain` (the next card does)
- gpu: one run (1 server session), waiting for a quiet box

### 0.7 Add the unshifted generation to cache_reuse_ids.json for granite_moe

- id: FT0.7
- needs: FT0.54
- budget: 20 min
- crate(s): none (fixtures)
- read first:
  - `proxima-model-interop/tests/fixtures/llama-parity/granite_moe/cache_reuse_ids.json` (written by the previous card: `.[0].request2.prompt_ids` is the prompt both generations use);
  - the point of the key: `request2.generated_ids` came from a slot whose cache was shifted, `request2_plain` is the same prompt generated by a server with no chunk reuse (a later slice reads `.[0].request2_plain.generated_ids` for the gemma4_e2b file, and the other two checkpoints carry the same key).
- change:
  1. `mkdir -p $LOGS/FT0.7`; the quiet-box check; start a fresh server without `--cache-reuse`: `$LLAMA -m $GRANITE_MOE -ngl 99 -c 8192 -np 1 --port 18163 > $LOGS/FT0.7/server.log 2>&1 &` and wait on `/health`.
  2. `ids2=$(jq -c '.[0].request2.prompt_ids' proxima-model-interop/tests/fixtures/llama-parity/granite_moe/cache_reuse_ids.json)`; POST `/completion` `{"prompt": <ids2>, "n_predict": 32, "temperature": 0, "cache_prompt": false, "stream": false, "return_tokens": true}` once; keep `.tokens` as `gen2_plain`. Stop the server.
  3. Add the key to the same file with `jq`: `.[0].request2_plain = {"prompt_ids": ids2, "generated_ids": gen2_plain, "command": "llama-server -m /Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80 -ngl 99 -c 8192 -np 1 --port 18163"}`; write to a temporary file under `$LOGS/FT0.7/` and move it over the fixture (never redirect jq onto its own input).
- test: none; the validate command asserts the fixture.
- validate: `jq -e '.[0] | (.request2.cache_n == ((.request2.prompt_ids|length) - 1)) and ((.request2.generated_ids|length) == 32) and (.request2_plain.prompt_ids == .request2.prompt_ids) and ((.request2_plain.generated_ids|length) == 32)' proxima-model-interop/tests/fixtures/llama-parity/granite_moe/cache_reuse_ids.json | grep -c true`
- expect: `1`
- also green: n/a
- stage: `proxima-model-interop/tests/fixtures/llama-parity/granite_moe/cache_reuse_ids.json`
- commit: `test(llama-parity): vendor granite moe unshifted generation`
- done when: the expect line printed, `git diff --cached --stat` shows only that file, no llama-server process left running, and the commit landed with that message
- do not: run this on the reuse server; pass `--cache-reuse`; change `request1` or `request2`; put a card id or a log path in the fixture
- gpu: one run (1 server session), waiting for a quiet box

## oracle control tests

### 0.8 Add the fsm_oracle_control_ tests

- id: FT0.8
- needs: FT0.5, FT0.6, FT0.7, FT0.21
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/tests/arch_data_baseline.rs::first_divergence` (~661 at a7c08c4c) and `::ids_of` (~649): the id comparator to copy (a test file cannot import it);
  - the three fixture kinds written by FT0.5, FT0.6 and FT0.7, for `gemma4_e2b`, `gemma4_26b` and `granite_moe`;
  - `SPECDIR/worked-examples.md`, section "readout tolerance" (the per-checkpoint tolerances).
- change:
  1. New file `proxima-model-interop/tests/fsm_oracle_control.rs`. It holds:
     - `#![allow(clippy::unwrap_used, clippy::expect_used)]` as the first line (workspace lints deny both and `proxima-model-interop` inherits them; `arch_data_baseline.rs:15` carries the same line), covering the `expect` in `fixture` and the unwraps in `ids_of`;
     - `const CHECKPOINTS: [&str; 3] = ["gemma4_e2b", "gemma4_26b", "granite_moe"];`
     - `fn tolerance(checkpoint: &str) -> f64`: `gemma4_e2b` 4.673e-3, `gemma4_26b` 3.719e-3, `granite_moe` 2.289e-3 (the readout tolerance RESULT line); any other name panics with the name;
     - `fn fixture(checkpoint: &str, name: &str) -> serde_json::Value` reading `CARGO_MANIFEST_DIR/tests/fixtures/llama-parity/<checkpoint>/<name>` with `expect`;
     - `fn ids_of(value: &serde_json::Value) -> Vec<u32>`;
     - `fn first_divergence(expected: &[u32], actual: &[u32]) -> Option<usize>` (a copy of the `arch_data_baseline` one);
     - `fn max_logprob_gap(left: &serde_json::Value, right: &serde_json::Value) -> f64`: the maximum over all steps and both `top` entries of `|left.logprob - right.logprob|`.
  2. Tests (all `#[test]`, sync, no sleeps):
     - `fsm_oracle_control_followup_ids_match_themselves`: for each of the 3 checkpoints, read `followup_ids.json`, assert `first_divergence(turn2.generated_ids, turn2.generated_ids)` is `None`, count the checkpoints processed and `assert_eq!(processed, 3)`.
     - `fsm_oracle_control_n_probs_match_themselves`: for each checkpoint and each of its 3 records, assert `max_logprob_gap(record, record) <= tolerance(checkpoint)` and that the record has at least one step; assert 9 records processed.
     - `fsm_oracle_control_cache_reuse_ids_match_themselves`: as the first, over `cache_reuse_ids.json` `request2.generated_ids`; assert 3 processed.
     - `fsm_oracle_control_flipped_id_is_rejected`: take gemma4_e2b `followup_ids.json` `turn2.generated_ids`, flip index 7 (`ids[7] ^= 1`), assert `first_divergence(original, flipped) == Some(7)`.
     - `fsm_oracle_control_perturbed_probability_is_rejected`: take gemma4_e2b `n_probs.json` record 0, clone it, add `2.0 * tolerance("gemma4_e2b")` to `steps[0].top[0].logprob`, assert `max_logprob_gap(original, perturbed) > tolerance("gemma4_e2b")`.
- test: the 5 tests above.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_0_8 cargo nextest run -p proxima-model-interop --features std -E 'test(/fsm_oracle_control_/)'`
- expect: `5 passed` (names above appear)
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: `proxima-model-interop/tests/fsm_oracle_control.rs`
- commit: `test(llama-parity): add self-comparison controls for vendored oracles`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: edit `arch_data_baseline.rs`; add a dependency; read any `llama_ids.json`; name a qwen checkpoint
- gpu: none

## worked examples (hand derivation first; `/algorithm-development`)

Conventions for every card 0.14 to 0.29:
- The deliverable is a hand derivation in `SPECDIR/worked-examples.md`: a heading `## <name>`, named for what it computes (never an id), an inputs block (copied from the card), numbered derivation steps each showing its arithmetic, and one line `RESULT <name>: <exact string>`.
- The card states the expected RESULT. The executor derives the numbers itself, step by step, and compares. A mismatch with the card's RESULT is a false premise: stop and report both.
- No code, no tests in these cards. They are the worked values later cards encode as tests.
- Card 0.14 creates the file; later cards append, in order 0.14, 0.15, and so on through 0.29.
- English only: section text, headings and RESULT lines carry no slice, stage, AC, R, W, FT or card id.
- Every card here is documentation only: `crate(s): none (docs)`, `test: none`, `also green: n/a`, `gpu: none`, `budget: 20 min`. validate is the `grep -c` of the exact RESULT text and expect is `1`.

### 0.14 Worked example block seal trace (10 appends, 3 rewinds)

- id: FT0.14
- needs: none
- budget: 20 min
- crate(s): none (docs)
- read first:
  - hook served: seal and tier. `PAD/SPEC.md`, the catalog row for the seal-and-tier hook (~line 90, default "host memory only") and the structural finding that no seal concept exists;
  - `PAD/sketches/08-rectified-sparse-attention.md`, section "HOOK GAPS", seal storage (~line 222): per-block summaries need a seal point, and `proxima-model-interop/src/generate/residency_caches.rs::LayerCache` (~112 at a7c08c4c) has no block structure while `::truncate` (~155) is infallible (`git grep -n "sealed_end" a7c08c4c` finds nothing);
  - `PAD/sketches/10-tiered-chunk-cache.md`, section "shape chosen" (~line 21): tiers use entry grain; this example is the block-grain seal the summary fold needs and makes no tier claim.
- change: create `SPECDIR/worked-examples.md` (that is `proxima-tensor/specs/fsm-techniques/worked-examples.md`) with the title line `# fsm techniques: worked examples (derived by hand before code)` and the section `## block seal trace`. Inputs: block size b = 4 rows, seal horizon H = 1 row. Rule: after every operation, `sealed_end = max(sealed_end, floor(max(0, len - H) / b) * b)`; a rewind to length t is refused with a typed rewind-into-sealed error, state unchanged, exactly when `t < sealed_end`. Sequence: a1 a2 a3 a4 a5, r1 (to 4), a6 a7 a8 a9 a10, r2 (to 8), r3 (to 7); an append adds one row, a rewind to length t keeps t rows. Table (len/sealed_end after the operation):
  - a1 1/0, a2 2/0, a3 3/0, a4 4/0 (block 0 is full but `len - H = 3 < 4`: inside the horizon, not sealed);
  - a5 5/4 (`len - H = 4`: block 0 sealed);
  - r1 to 4: `4 >= 4` allowed, 4/4; a6 5/4; a7 6/4; a8 7/4; a9 8/4 (`len - H = 7 < 8`);
  - a10 9/8 (`len - H = 8`: block 1 sealed);
  - r2 to 8: `8 >= 8` allowed, 8/8; r3 to 7: `7 < 8`, refused with the rewind-into-sealed error carrying keep 7 and sealed end 8, 8/8.
- test: none
- validate: `grep -c -F 'RESULT block seal trace: trace=[a1:1/0,a2:2/0,a3:3/0,a4:4/0,a5:5/4,r1->4:ok 4/4,a6:5/4,a7:6/4,a8:7/4,a9:8/4,a10:9/8,r2->8:ok 8/8,r3->7:RewindIntoSealed 8/8] final_len=8 sealed_end=8 sealed_blocks=2' proxima-tensor/specs/fsm-techniques/worked-examples.md`
- expect: `1`
- also green: n/a
- stage: `proxima-tensor/specs/fsm-techniques/worked-examples.md`
- commit: `docs(fsm): derive the block seal trace worked example`
- done when: the expect line printed, and `git diff --cached --stat` run in the checkout holding main equals the stage list (the stage list is relative to that checkout root; `git add` it there, and make the commit there with `git commit`)
- do not: touch any file but `worked-examples.md`; write code
- gpu: none

### 0.15 Worked example eviction victim (today's rule and three policies)

- id: FT0.15
- needs: FT0.14
- budget: 20 min
- crate(s): none (docs)
- read first:
  - hook served: seal and tier (the eviction decision). `proxima-model-interop/src/generate/prompt_cache.rs::PromptCache::eviction_victim` (~819 at a7c08c4c): today's rule, an unused follow-up branch (`branch_base` set) before anything a request produced, otherwise the entry with the lowest stamp, over `entries` in ascending stamp order; the stamp is reissued on every store (`PromptCache::store`, ~942), so the lowest stamp is the least recently stored entry;
  - `PAD/sketches/10-tiered-chunk-cache.md`, section 2 (the pure function `eviction_victim(entries, rules)` over an ordered rule list, `[Branch, Oldest]` reproducing today's code) and gap G1;
  - `PAD/SPEC.md`, the structural finding that the draft's `lru|lfu|fifo` default does not reproduce today's victim rule (~287).
- change: append `## eviction victim`. Rules, tried in the listed order, the first rule that finds a candidate decides: `branch_first` (the first entry in ascending stamp order whose branch flag is set), `lowest_stamp` (the first entry), `fewest_uses` (the lowest use count, ties to the lower stamp), `earliest_inserted` (the lowest insertion counter). Inputs, entries in ascending stamp order as `(stamp, branch, uses, inserted)`: A (3, no, 4, 1), B (5, yes, 0, 4), C (7, yes, 1, 2), D (9, no, 2, 3). Derivation, each as the walk over A to D:
  - today's rule `[branch_first, lowest_stamp]`: `branch_first` finds B at stamp 5, so the victim is 5;
  - `[lowest_stamp]`: A, stamp 3;
  - `[fewest_uses]`: use counts 4, 0, 1, 2, the lowest is B, stamp 5;
  - `[earliest_inserted]`: insertion counters 1, 4, 2, 3, the lowest is A, stamp 3;
  - today's rule over the two entries A and D (no branch flag set): `branch_first` finds nothing, `lowest_stamp` gives A, stamp 3;
  - `[fewest_uses]` over X (2, no, 1, 1) and Y (4, no, 1, 2): equal use counts, the tie goes to the lower stamp, stamp 2;
  - any rule list over an empty entry set: no victim.
  State that `uses` and `inserted` are fields the cache entry does not carry today; the example shows the decision function over them, not that they exist.
- test: none
- validate: `grep -c -F 'RESULT eviction victim: today=[branch_first,lowest_stamp] -> 5; lowest_stamp -> 3; fewest_uses -> 5; earliest_inserted -> 3; no_branch today -> 3; fewest_uses_tie -> 2; empty -> none' proxima-tensor/specs/fsm-techniques/worked-examples.md`
- expect: `1`
- also green: n/a
- stage: `proxima-tensor/specs/fsm-techniques/worked-examples.md`
- commit: `docs(fsm): derive the eviction victim worked example`
- done when: the expect line printed, and `git diff --cached --stat` run in the checkout holding main equals the stage list (the stage list is relative to that checkout root; `git add` it there, and make the commit there with `git commit`)
- do not: touch any file but `worked-examples.md`; carry the old two-tier demotion trace (the demotion sink is a separate hook change with its own card)
- gpu: none

### 0.16 Worked example per-layer recompute selection (3 chunks, chained)

- id: FT0.16
- needs: FT0.15
- budget: 20 min
- crate(s): none (docs)
- read first:
  - hook served: assemble. `PAD/sketches/09-cacheblend.md`, section "HOOK GAPS", gap 1 (more than one source entry, ~213) and gap 3 (a layer-boundary hand-off, ~233): selection chained layer to layer is the decision a blend-shaped stage needs; this card derives the decision only;
  - `PAD/SPEC.md`, the catalog row for the assemble hook (~86).
- change: append `## per-layer recompute selection`. Inputs: 3 chunks of 4 rows (M = 12 rows, indices 0 to 11, chunk c holds rows 4c to 4c+3), ratios per layer `[0.5, 0.25, 0.125]` (check layer first), count per layer = `ceil(ratio * 12)` = `[6, 3, 2]`. Each later layer selects only among the previous layer's selected rows. Deviation `d` = per-row sum of `|loaded - recomputed|`:
  - check layer, all 12 rows: `[0.05, 0.90, 0.10, 0.40, 0.02, 0.30, 0.75, 0.08, 0.60, 0.15, 0.04, 0.50]`; top 6 by d: rows 1 (0.90), 6 (0.75), 8 (0.60), 11 (0.50), 3 (0.40), 5 (0.30), listed sorted `[1,3,5,6,8,11]`; per chunk 2, 2, 2;
  - layer 2, deviations of the 6 selected rows: row 1 0.50, row 3 0.80, row 5 0.20, row 6 0.70, row 8 0.10, row 11 0.60; top 3: rows 3, 6, 11;
  - layer 3, deviations of rows 3, 6, 11: 0.30, 0.90, 0.50; top 2: rows 6, 11.
- test: none
- validate: `grep -c -F 'RESULT per-layer recompute selection: check=[1,3,5,6,8,11] layer2=[3,6,11] layer3=[6,11] counts=[6,3,2] per_chunk_check=[2,2,2]' proxima-tensor/specs/fsm-techniques/worked-examples.md`
- expect: `1`
- also green: n/a
- stage: `proxima-tensor/specs/fsm-techniques/worked-examples.md`
- commit: `docs(fsm): derive the per-layer recompute selection example`
- done when: the expect line printed, and `git diff --cached --stat` run in the checkout holding main equals the stage list (the stage list is relative to that checkout root; `git add` it there, and make the commit there with `git commit`)
- do not: touch any file but `worked-examples.md`; write code
- gpu: none

### 0.17 Worked example block scoring (4 blocks, non-local count)

- id: FT0.17
- needs: FT0.16
- budget: 20 min
- crate(s): none (docs)
- read first:
  - hook served: read. `PAD/sketches/08-rectified-sparse-attention.md`, section 4 (worked example) and section 3 (`block_score`), and gaps GAP-6 (a read field on the descriptor, ~186) and GAP-8 (no top-k op, ~208);
  - the non-local count rule decided 2026-10-04 (excluding the local blocks keeps the attended count data-independent), stated in full in the change below; no spec file is read;
  - the keep ratios 0.25 and 0.75 are given here; no paper fact is needed.
- change: append `## block scoring`. Inputs: head dimension 2, 4 sealed blocks with per-dimension key min and max: B0 min (0,0) max (1,1); B1 min (-2,-1) max (0,3); B2 min (1,-3) max (2,-1); B3 min (-1,-1) max (1,1). Pooled query q = (2,-1). Score of block i = sum over dims j of `max(q_j * kmax_ij, q_j * kmin_ij)`:
  - B0: max(2,0) + max(-1,0) = 2 + 0 = 2;
  - B1: max(0,-4) + max(-3,1) = 0 + 1 = 1;
  - B2: max(4,2) + max(1,3) = 4 + 3 = 7;
  - B3: max(2,-2) + max(-1,1) = 2 + 1 = 3.
  Selection: the most recent sealed block is local (`local_blocks = 1`, B3) and never competes for a top slot; the non-local count is `nonlocal = M - local = 3` and `n = min(nonlocal, max(n_min, ceil(keep_ratio * nonlocal)))` with `n_min = 1`, taken over the scores of B0 to B2 = `[2, 1, 7]`:
  - keep_ratio 0.25: `ceil(0.75) = 1`, n = 1, top = {B2}, attended = {B2} plus local {B3} = {2,3};
  - keep_ratio 0.75: `ceil(2.25) = 3`, n = min(3, 3) = 3, top = {B2, B0, B1}, attended = {0,1,2,3}.
  Note in the section that an earlier form counted n over all four blocks and let the local block compete (it listed `{0,2,3}` at 0.75); the stated rule counts non-local blocks only.
- test: none
- validate: `grep -c -F 'RESULT block scoring: scores=[2,1,7,3] keep0.25:n=1,attended={2,3} keep0.75:n=3,attended={0,1,2,3}' proxima-tensor/specs/fsm-techniques/worked-examples.md`
- expect: `1`
- also green: n/a
- stage: `proxima-tensor/specs/fsm-techniques/worked-examples.md`
- commit: `docs(fsm): derive the block scoring worked example`
- done when: the expect line printed, and `git diff --cached --stat` run in the checkout holding main equals the stage list (the stage list is relative to that checkout root; `git add` it there, and make the commit there with `git commit`)
- do not: touch any file but `worked-examples.md`; write code
- gpu: none

### 0.18 Worked example block read rows (gemma4 E2B)

- id: FT0.18
- needs: FT0.17
- budget: 20 min
- crate(s): none (docs)
- read first:
  - hook served: read. `PAD/sketches/08-rectified-sparse-attention.md`, section 4 and GAP-7 (the read selection is a visibility term inside the cached-attention graph; the windowed layers already bound their reads);
  - `proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/swa_layers.txt` (`is_swa = 0` for layers 4, 9, 14, 19, 24, 29, 34: 7 full-attention layers) and `.../gemma4_e2b/gguf_kv.txt` (`gemma4.block_count = 35`, `gemma4.attention.sliding_window = 512`);
  - `proxima-model-interop/examples/long_context_niah.rs` (the run that prints `prompt_tokens=` and `max_new_tokens=` on one line, ~341 at a7c08c4c; the formula below is evaluated with P = the printed `prompt_tokens` and T = the printed `max_new_tokens`, the number of decode steps).
- change: append `## block read rows`. The top-n blocks are chosen among the sealed non-local blocks and local blocks are always added, so the count is a pure function of lengths. Only the full-attention layers apply the block read (a windowed layer reads `min(len, window)` rows either way), so `layers` below counts the 7 full-attention layers of gemma4 E2B; each of them, including the ones that share another layer's cache, evaluates its own attention over the rows. Formula, per layer, per decode step s in 0 to T-1:
  - `len_s = P + s + 1` (rows visible, including the new row);
  - `sealed_end_s = floor(max(0, len_s - H) / b) * b`, `M_s = sealed_end_s / b`;
  - `L' = min(L, M_s)`; `n_s = min(M_s - L', max(n_min, ceil(keep_ratio * (M_s - L'))))`;
  - `rows_s = (n_s + L') * b + (len_s - sealed_end_s)`;
  - `kv_rows_read = layers * sum_s rows_s`.
  Instance: b = 64, H = 64, n_min = 16, L = 1, layers = 7, P = 4096, T = 2:
  - len = 4097, 4098; sealed_end = 4032 for both (floor(4033/64) = 63, floor(4034/64) = 63); M = 63, L' = 1, non-local 62;
  - keep_ratio 0.1: n = min(62, max(16, ceil(6.2) = 7)) = 16; rows = 17*64 + 65 = 1153 and 17*64 + 66 = 1154; per layer 2307; total 7 * 2307 = 16149;
  - keep_ratio 0.9: n = min(62, max(16, ceil(55.8) = 56)) = 56; rows = 57*64 + 65 = 3713 and 3714; per layer 7427; total 7 * 7427 = 51989.
  State in the section that 35 layers (all blocks of the model) would be wrong for this checkpoint: 28 of its layers are windowed.
- test: none
- validate: `grep -c -F 'RESULT block read rows: P=4096 T=2 layers=7 b=64 H=64 n_min=16 L=1 keep=0.1 -> 16149; keep=0.9 -> 51989' proxima-tensor/specs/fsm-techniques/worked-examples.md`
- expect: `1`
- also green: n/a
- stage: `proxima-tensor/specs/fsm-techniques/worked-examples.md`
- commit: `docs(fsm): derive the block read row count worked example`
- done when: the expect line printed, and `git diff --cached --stat` run in the checkout holding main equals the stage list (the stage list is relative to that checkout root; `git add` it there, and make the commit there with `git commit`)
- do not: touch any file but `worked-examples.md`; use a qwen shape
- gpu: none

### 0.19 Worked example sampled read budget (epsilon = delta = 0.05)

- id: FT0.19
- needs: FT0.18
- budget: 20 min
- crate(s): none (docs)
- read first:
  - hook served: read. `PAD/research-retrieval-attention.md` (~124: the draft already has a sampled read with `epsilon` and `delta`, a MagicPIG-shaped probabilistic read) and `PAD/SPEC.md`, the catalog row for the read hook (~89);
  - the statement the budget serves, given in full here: a sampled read's output has relative error at most epsilon with probability at least 1 - delta; no spec file is read.
- change: append `## sampled read budget`. Rule (two-sided normal quantile): `n = ceil((z * sd / (epsilon * mu))^2)` with `z = z_(1 - delta/2) = 1.959964`, `mu` and `sd` (sample standard deviation, n-1 denominator) over the base-rate sample. Inputs: base-rate sample of 8 per-row weights `[0.9, 1.1, 1.0, 1.2, 0.8, 1.0, 1.1, 0.9]`. Derivation: mu = 8.0/8 = 1.0; squared deviations 0.01, 0.01, 0, 0.04, 0.04, 0, 0.01, 0.01, sum 0.12; variance = 0.12/7 = 0.0171429; sd = 0.130931; z*sd/(0.05*1.0) = 5.13239; squared = 26.3414; n = 27.
- test: none
- validate: `grep -c -F 'RESULT sampled read budget: mu=1.0 sd=0.1309 z=1.959964 n=27' proxima-tensor/specs/fsm-techniques/worked-examples.md`
- expect: `1`
- also green: n/a
- stage: `proxima-tensor/specs/fsm-techniques/worked-examples.md`
- commit: `docs(fsm): derive the sampled read budget worked example`
- done when: the expect line printed, and `git diff --cached --stat` run in the checkout holding main equals the stage list (the stage list is relative to that checkout root; `git add` it there, and make the commit there with `git commit`)
- do not: touch any file but `worked-examples.md`; write code
- gpu: none

### 0.20 Worked example row tolerance (gemma4_e2b, gemma4_26b, granite_moe)

- id: FT0.20
- needs: FT0.19, FT0.31
- budget: 20 min
- crate(s): none (docs)
- read first:
  - hook served: none directly; shared conformance substrate for comparing rows produced by different read or assemble paths against dense ones. `proxima-model-interop/tests/fixtures/llama-parity/{gemma4_e2b,gemma4_26b,granite_moe}/gguf_kv.txt` (the granite one is vendored by FT0.31, which this card needs): `block_count`, `embedding_length`, `attention.head_count`, `attention.key_length` (or `rope.dimension_count` when absent), `feed_forward_length`, `expert_feed_forward_length`;
  - the per-layer `feed_forward_length` array of gemma4 E2B, which `gguf_kv.txt` truncates to its first six values: 6144 for layers 0 to 14 and 12288 for layers 15 to 34 (read from the header with llama.cpp's `gguf_dump.py`; the same split is written in `proxima-model-interop/src/gemma4/bind.rs::e2b_shaped`, ~1050).
- change: append `## row tolerance`. Model: each sublayer ends in a reduction accumulated in f32 as a tree; a tree reduction of n terms has relative error at most `ceil(log2 n) * u`, `u = 2^-24`. Rows are compared K and V at every layer: `tau = 2 * L * ceil(log2 n) * u` with L = block_count and n the widest reduction in the model, `n = max(embedding_length, head_count * key_length, max over layers of feed_forward_length, expert_feed_forward_length)` (the attention output projection reduces over `head_count * key_length`). The comparison is `max_i |a_i - b_i| <= tau * max_i |b_i|` per row. Inputs and results:
  - gemma4_e2b: L 35; embedding 1536; attention 8 x 512 = 4096; feed forward 12288 (layers 15 to 34); n = 12288, ceil(log2) = 14; tau = 2*35*14*2^-24 = 5.841e-05;
  - gemma4_26b: L 30; embedding 2816; attention 16 x 512 = 8192; feed forward 2112; expert 704; n = 8192, ceil(log2) = 13; tau = 2*30*13*2^-24 = 4.649e-05;
  - granite_moe: L 24; embedding 1024; attention 16 x 64 = 1024 (no `key_length` key: head width is `rope.dimension_count` 64); expert feed forward 512; n = 1024, ceil(log2) = 10; tau = 2*24*10*2^-24 = 2.861e-05.
  State the assumption: a kernel that reduces sequentially instead of as a tree would exceed these; if a measured gap exceeds tau the test fails and tau is not widened. State that the earlier gemma4 E2B value (5.424e-05) used n = 6144, missing the 20 shared-cache layers' 12288, and omitted the attention width.
- test: none
- validate: `grep -c -F 'RESULT row tolerance: gemma4_e2b=5.841e-05 gemma4_26b=4.649e-05 granite_moe=2.861e-05' proxima-tensor/specs/fsm-techniques/worked-examples.md`
- expect: `1`
- also green: n/a
- stage: `proxima-tensor/specs/fsm-techniques/worked-examples.md`
- commit: `docs(fsm): derive the row tolerance worked example`
- done when: the expect line printed, and `git diff --cached --stat` run in the checkout holding main equals the stage list (the stage list is relative to that checkout root; `git add` it there, and make the commit there with `git commit`)
- do not: touch any file but `worked-examples.md`; add a qwen shape
- gpu: none

### 0.21 Worked example readout tolerance (log probabilities)

- id: FT0.21
- needs: FT0.20, FT0.31
- budget: 20 min
- crate(s): none (docs)
- read first:
  - hook served: readout (the tap that gives a settle decision its inputs). `PAD/sketches/13-conformal-cascade.md`, gap G5 (`TokenEvent` carries no probability: `proxima-model-interop/src/generate/residency_caches.rs::TokenEvent`, ~3169 at a7c08c4c) and `PAD/SPEC.md`, the catalog row for the tap-and-edit stage (~104);
  - the row tolerance section just above; `.../gemma4_e2b/gguf_kv.txt` and `.../gemma4_26b/gguf_kv.txt` (`gemma4.final_logit_softcapping = 30.0`) and `.../granite_moe/gguf_kv.txt` (`granitemoe.logit_scale = 6.0`, no soft cap).
- change: append `## readout tolerance`. Model: `logprob = logit - logsumexp(logits)`. With logits carrying relative error tau (the row tolerance section) and an assumed bound `|logit| <= 40`, the logit error is at most `40 * tau`, and a logprob moves by the chosen logit's error plus the logsumexp's, so `tol_logprob = 2 * tau * 40`. For the top-1 minus top-2 probability margin, `|dp| <= p * |dlogp|` and `p1 + p2 <= 1`, so `tol_margin = tol_logprob`. The bound on the logit: both gemma4 headers declare a final soft cap of 30.0, so `|logit| <= 30 < 40` holds by construction; granite declares no cap, its logits are divided by its `logit_scale` 6.0 after the head, and `|logit| <= 40` on the scaled value is an assumption, unmeasured (state it as such). Results: gemma4_e2b 80 * 5.841e-05 = 4.673e-03; gemma4_26b 80 * 4.649e-05 = 3.719e-03; granite_moe 80 * 2.861e-05 = 2.289e-03.
- test: none
- validate: `grep -c -F 'RESULT readout tolerance: tol_logprob gemma4_e2b=4.673e-03 gemma4_26b=3.719e-03 granite_moe=2.289e-03 tol_margin=tol_logprob' proxima-tensor/specs/fsm-techniques/worked-examples.md`
- expect: `1`
- also green: n/a
- stage: `proxima-tensor/specs/fsm-techniques/worked-examples.md`
- commit: `docs(fsm): derive the readout tolerance worked example`
- done when: the expect line printed, and `git diff --cached --stat` run in the checkout holding main equals the stage list (the stage list is relative to that checkout root; `git add` it there, and make the commit there with `git commit`)
- do not: touch any file but `worked-examples.md`; present the granite bound as measured
- gpu: none

### 0.22 Worked example cartridge concatenation rotation

- id: FT0.22
- needs: FT0.21
- budget: 20 min
- crate(s): none (docs)
- read first:
  - hook served: position. `proxima-model-interop/src/generate/chunk_shift.rs::rotate_rows` (~173 at a7c08c4c, private: `even' = even * cos - odd * sin`, `odd' = odd * cos + even * sin` per pair, rows laid out `[row][head][pair]`) and `LoadedModel::delta_rotations` (~390, `pub(super)`: the per-layer `(cos, sin)` of a position delta, a negative delta rotating the other way; the working entry point, `PAD/sketches/11-cartridges-load.md` defect D2);
  - `PAD/SPEC.md`, the catalog row for the position stage (~111). Cartridges appear only as the example; the loaded rows are never trained here.
- change: append `## cartridge concatenation rotation`. Inputs: head dim 4 (2 pairs), per-pair angle per position theta = (pi/2, pi/4). Cartridge A holds 2 rows; cartridge B (2 rows, base position 0) is concatenated after A, so each B row moves by delta = 2 positions: angles (pi, pi/2), `cos = [-1, 0]`, `sin = [0, 1]` (exact in the hand arithmetic; in f32 `sin(pi)` is not exactly 0, so a test of this value uses a tolerance of 1e-6). B's K rows (layout [row][pair]): row 0 even `[1,2]` odd `[0,3]`; row 1 even `[0,4]` odd `[2,-1]`.
  - row 0 pair 0: even' = 1*(-1) - 0*0 = -1; odd' = 0*(-1) + 1*0 = 0; pair 1: even' = 2*0 - 3*1 = -3; odd' = 3*0 + 2*1 = 2;
  - row 1 pair 0: even' = 0*(-1) - 2*0 = 0; odd' = 2*(-1) + 0*0 = -2; pair 1: even' = 4*0 - (-1)*1 = 1; odd' = (-1)*0 + 4*1 = 4.
  V rows are not rotated.
- test: none
- validate: `grep -c -F 'RESULT cartridge concatenation rotation: k_b_rotated even=[[-1,-3],[0,1]] odd=[[0,2],[-2,4]] v_unchanged' proxima-tensor/specs/fsm-techniques/worked-examples.md`
- expect: `1`
- also green: n/a
- stage: `proxima-tensor/specs/fsm-techniques/worked-examples.md`
- commit: `docs(fsm): derive the cartridge rotation worked example`
- done when: the expect line printed, and `git diff --cached --stat` run in the checkout holding main equals the stage list (the stage list is relative to that checkout root; `git add` it there, and make the commit there with `git commit`)
- do not: touch any file but `worked-examples.md`; derive any training step
- gpu: none

### 0.23 Worked example threshold judge

- id: FT0.23
- needs: FT0.22
- budget: 20 min
- crate(s): none (docs)
- read first:
  - hook served: settle. `PAD/sketches/13-conformal-cascade.md`, sections 1 and 2 (one pure `settle` function; a threshold judge reads one readout) and gap G5 (the readout is a different hook's input);
  - `PAD/SPEC.md`, the catalog row for the settle hook (~94, default "always settle").
- change: append `## threshold judge`. Rule: readout = chosen-token logprob; settle iff `readout >= h`, h = -0.5 (inclusive). Cases: -0.2 >= -0.5 settle; -0.9 < -0.5 escalate; -0.5 >= -0.5 settle (boundary).
- test: none
- validate: `grep -c -F 'RESULT threshold judge: lp=-0.2 -> settle; lp=-0.9 -> escalate; lp=-0.5 -> settle' proxima-tensor/specs/fsm-techniques/worked-examples.md`
- expect: `1`
- also green: n/a
- stage: `proxima-tensor/specs/fsm-techniques/worked-examples.md`
- commit: `docs(fsm): derive the threshold judge worked example`
- done when: the expect line printed, and `git diff --cached --stat` run in the checkout holding main equals the stage list (the stage list is relative to that checkout root; `git add` it there, and make the commit there with `git commit`)
- do not: touch any file but `worked-examples.md`; write code
- gpu: none

### 0.24 Worked example classifier judge

- id: FT0.24
- needs: FT0.23
- budget: 20 min
- crate(s): none (docs)
- read first:
  - hook served: settle. `PAD/sketches/13-conformal-cascade.md`, section 2 (`settle` is the only decision; a classifier judge is one more pure function of class probabilities) and gap G5 (the probabilities are readouts the settle hook does not produce);
  - the rule below is given; no paper fact is needed.
- change: append `## classifier judge`. Rule: the judge model emits class probabilities `[p_accept, p_escalate]`; `accept_class = 0` is a configured value; settle iff `argmax = accept_class`. Cases: [0.7, 0.3] -> argmax 0 settle; [0.4, 0.6] -> argmax 1 escalate.
- test: none
- validate: `grep -c -F 'RESULT classifier judge: [0.7,0.3] -> settle; [0.4,0.6] -> escalate' proxima-tensor/specs/fsm-techniques/worked-examples.md`
- expect: `1`
- also green: n/a
- stage: `proxima-tensor/specs/fsm-techniques/worked-examples.md`
- commit: `docs(fsm): derive the classifier judge worked example`
- done when: the expect line printed, and `git diff --cached --stat` run in the checkout holding main equals the stage list (the stage list is relative to that checkout root; `git add` it there, and make the commit there with `git commit`)
- do not: touch any file but `worked-examples.md`; write code
- gpu: none

### 0.25 Worked example conformal judge (integer disagreement bound)

- id: FT0.25
- needs: FT0.24
- budget: 20 min
- crate(s): none (docs)
- read first:
  - hook served: settle. `PAD/sketches/13-conformal-cascade.md`, section 1 (`max_disagree` is q-hat in units of samples, an integer, because a division by N is not exact: one disagreeing draw of 16 is 62.5 per thousand) and section 2 (`settle(votes, samples, max_disagree)`: the answer is accepted when exactly one distinct answer has `samples - votes <= max_disagree`);
  - `PAD/research.md` CC-10 (calibration tables are keyed by the serving configuration); the bound is a supplied value here, fitting it is out of scope.
- change: append `## conformal judge`. Rule: `samples = 16`, `max_disagree = 13` (q-hat 0.8125 over 16 samples is 13 disagreements, given as a calibrated value); an answer is in the prediction set iff `samples - votes <= max_disagree`; settle iff the set has exactly one answer, and the settled answer is that one. Cases:
  - votes {A:9, B:4, C:3}: disagreements 7, 12, 13, all `<= 13`, set {A,B,C}, |C| = 3, escalate;
  - votes {A:12, B:2, C:2}: disagreements 4, 14, 14, only A is inside, |C| = 1, settle A.
- test: none
- validate: `grep -c -F 'RESULT conformal judge: samples=16 max_disagree=13 votes{A:9,B:4,C:3} -> escalate (|C|=3); votes{A:12,B:2,C:2} -> settle A (|C|=1)' proxima-tensor/specs/fsm-techniques/worked-examples.md`
- expect: `1`
- also green: n/a
- stage: `proxima-tensor/specs/fsm-techniques/worked-examples.md`
- commit: `docs(fsm): derive the conformal judge worked example`
- done when: the expect line printed, and `git diff --cached --stat` run in the checkout holding main equals the stage list (the stage list is relative to that checkout root; `git add` it there, and make the commit there with `git commit`)
- do not: touch any file but `worked-examples.md`; derive the calibration (a fitter, out of scope)
- gpu: none

### 0.26 Worked example isotonic judge (supplied table)

- id: FT0.26
- needs: FT0.25
- budget: 20 min
- crate(s): none (docs)
- read first:
  - hook served: settle. `PAD/sketches/13-conformal-cascade.md`, gap G5 (the margin readout) and defect D2 (one config section carries several hooks);
  - `PAD/research.md` CC-10 (the table is keyed by the serving configuration).
- change: append `## isotonic judge`. Rule: per-token margin `p_top1 - p_top2` over the answer's tokens; `u = 1 - mean(margins)`; `g(u)` is a supplied fitted table at the exact grid point u: grid `u = 0.125, 0.25, ..., 1.0` with `g = 0, 0, 0.5, 0.5, 0.5, 0.5, 1, 1` (a calibration output given here; producing it is a fitter and out of scope); settle iff `g(u) <= theta`, theta = 0.5. Cases:
  - margins [0.875, 0.625, 0.75]: mean 0.75, u = 0.25, g = 0 -> settle;
  - margins [0.5, 0.5, 0.5]: mean 0.5, u = 0.5, g = 0.5 <= 0.5 -> settle (boundary);
  - margins [0.25, 0.0, 0.125]: mean 0.125, u = 0.875, g = 1 -> escalate.
  (All means and u values are exact in binary.)
- test: none
- validate: `grep -c -F 'RESULT isotonic judge: u=0.25 g=0 -> settle; u=0.5 g=0.5 -> settle; u=0.875 g=1 -> escalate' proxima-tensor/specs/fsm-techniques/worked-examples.md`
- expect: `1`
- also green: n/a
- stage: `proxima-tensor/specs/fsm-techniques/worked-examples.md`
- commit: `docs(fsm): derive the isotonic judge worked example`
- done when: the expect line printed, and `git diff --cached --stat` run in the checkout holding main equals the stage list (the stage list is relative to that checkout root; `git add` it there, and make the commit there with `git commit`)
- do not: touch any file but `worked-examples.md`; fit the table
- gpu: none

### 0.27 Worked example always-settle judge (the default)

- id: FT0.27
- needs: FT0.26
- budget: 20 min
- crate(s): none (docs)
- read first:
  - hook served: settle. `PAD/SPEC.md`, the catalog row for the settle hook (~94: the default is "always settle") and `PAD/sketches/13-conformal-cascade.md`, section 1 (an absent cascade is one tier with judge `always`, one call);
  - the last tier always settles, so this judge is the row that proves the default.
- change: append `## always judge`. Rule: settle for every request regardless of readouts; in vote terms `samples = 1`, votes `[1]`, disagreement 0, inside every bound. Cases: 3 requests with logprobs -0.2, -3.0 and a missing readout all settle (3 of 3).
- test: none
- validate: `grep -c -F 'RESULT always judge: any readout -> settle (3 of 3 requests)' proxima-tensor/specs/fsm-techniques/worked-examples.md`
- expect: `1`
- also green: n/a
- stage: `proxima-tensor/specs/fsm-techniques/worked-examples.md`
- commit: `docs(fsm): derive the always-settle judge worked example`
- done when: the expect line printed, and `git diff --cached --stat` run in the checkout holding main equals the stage list (the stage list is relative to that checkout root; `git add` it there, and make the commit there with `git commit`)
- do not: touch any file but `worked-examples.md`; write code
- gpu: none

### 0.28 Worked example top-fraction selection (with the equal-score rule)

- id: FT0.28
- needs: FT0.27
- budget: 20 min
- crate(s): none (docs)
- read first:
  - hook served: read (selection). `PAD/sketches/08-rectified-sparse-attention.md`, GAP-8 (no top-k op; the in-algebra form is a rank count) and `PAD/research.md` CC-5 (~257: state the tie semantics so equal scores still select the stated count);
  - the selection rule (rank = count of scores strictly greater, selected = rank < count, plus kept rows) is stated in full in the change below; no spec file is read.
- change: append `## top-fraction selection`. Rule: `count = max(min_rows, ceil(fraction * M))`; `rank(i)` = the number of scores strictly greater than `score[i]` plus the number of equal scores at a lower index (so ranks are distinct and exactly `count` rows are selected); selected = `rank < count` or index in the kept rows. Inputs: fraction 0.25, min_rows 2, keep_rows {5}, score vector (12 values, indices 0 to 11, all distinct): `[0.31, 0.92, 0.15, 0.77, 0.64, 0.08, 0.55, 0.99, 0.23, 0.71, 0.40, 0.86]`.
  - M = 12 (fraction times M is an integer): count = max(2, ceil(0.25*12) = 3) = 3; ranks: 0.99 (idx 7) rank 0, 0.92 (idx 1) rank 1, 0.86 (idx 11) rank 2; selected {1,7,11} union keep {5} = {1,5,7,11};
  - M = 11 (the first 11 scores; fraction times M is not an integer): count = max(2, ceil(2.75) = 3) = 3; top three: 0.99 (7), 0.92 (1), 0.77 (3); union {5} = {1,3,5,7}.
  The two sets differ, so the count formula is distinguishable at the two sizes.
  Equal scores: M = 6, fraction 0.5, min_rows 1, no kept rows, scores `[0.5, 0.9, 0.5, 0.5, 0.1, 0.9]`: count = max(1, ceil(3.0) = 3) = 3. With the tie rule, ranks: idx 1 (0.9) greater 0, equal-lower 0, rank 0; idx 5 (0.9) rank 0 + 1 = 1; idx 0 (0.5) greater 2, rank 2; idx 2 (0.5) rank 2 + 1 = 3; idx 3 (0.5) rank 2 + 2 = 4; idx 4 (0.1) greater 5, rank 5; selected (rank < 3) = {0,1,5}, 3 rows. With strictly-greater rank alone: idx 0, 2, 3 all have rank 2, so {0,1,2,3,5} would be selected, 5 rows.
- test: none
- validate: `grep -c -F 'RESULT top-fraction selection: fraction=0.25 min_rows=2 keep={5} M=12 -> {1,5,7,11}; M=11 -> {1,3,5,7}; ties M=6 fraction=0.5 min_rows=1 scores=[0.5,0.9,0.5,0.5,0.1,0.9] -> {0,1,5} (3 selected; strictly-greater rank alone selects 5)' proxima-tensor/specs/fsm-techniques/worked-examples.md`
- expect: `1`
- also green: n/a
- stage: `proxima-tensor/specs/fsm-techniques/worked-examples.md`
- commit: `docs(fsm): derive the top-fraction selection worked example`
- done when: the expect line printed, and `git diff --cached --stat` run in the checkout holding main equals the stage list (the stage list is relative to that checkout root; `git add` it there, and make the commit there with `git commit`)
- do not: touch any file but `worked-examples.md`; write code
- gpu: none

### 0.29 Worked example block read selection (forced local block, unsealed tail)

- id: FT0.29
- needs: FT0.28
- budget: 20 min
- crate(s): none (docs)
- read first:
  - hook served: read. `PAD/sketches/08-rectified-sparse-attention.md`, section 4 and GAP-6 and GAP-7 (the read set is the union of the selected blocks, the local blocks and the unsealed tail);
  - the rule that n counts non-local blocks is stated in full in the change below; no spec file is read.
- change: append `## block read selection`. Inputs: block size b = 16, 4 sealed blocks (rows 0 to 63) plus an unsealed tail of 5 rows (rows 64 to 68, 69 rows total), keep_ratio 0.5, min_blocks 1, local_blocks 1, block scores `[6, 9, 4, 1]` (block 3, the most recent sealed block, has the lowest score). Derivation: local = block 3 (always, despite score 1), so the non-local blocks are 0, 1, 2 with scores 6, 9, 4 and `nonlocal = 3`; n = min(3, max(1, ceil(0.5 * 3) = 2)) = 2; top 2 of the non-local scores: block 1 (9), block 0 (6); the tail rows are all attended. Attended blocks {0,1,3} = 3 * 16 = 48 rows, plus 5 tail rows = 53 rows of 69; the 16 rows of block 2 are not attended. Note that counting n over all four blocks (`ceil(0.5 * 4) = 2`) also gives 2 here, so the selection does not distinguish the two forms; the non-local form is the stated rule.
- test: none
- validate: `grep -c -F 'RESULT block read selection: b=16 sealed=4 tail=5 n=2 top={0,1} local={3} attended_blocks={0,1,3} attended_rows=53' proxima-tensor/specs/fsm-techniques/worked-examples.md`
- expect: `1`
- also green: n/a
- stage: `proxima-tensor/specs/fsm-techniques/worked-examples.md`
- commit: `docs(fsm): derive the block read selection worked example`
- done when: the expect line printed, and `git diff --cached --stat` run in the checkout holding main equals the stage list (the stage list is relative to that checkout root; `git add` it there, and make the commit there with `git commit`)
- do not: touch any file but `worked-examples.md`; write code
- gpu: none

## spec drift

1. gemma4_26b is asserted by the existing `llama_parity_gemma4_26b` (three chat-templated prompts that llama picked by a top-1 minus top-2 margin of at least 1.0 nats at every generated step; the architecture-as-data SPEC read at a7c08c4c records that proxima's ids equal llama's on all three). No model-loading card in this file adds a 26B assertion. This file's 26B fixtures (follow-up, top-2 and reuse) are recorded from llama-server only, on chat-rendered prompts, and are exercised by the control tests (FT0.8) and by later slices; the margin recorded with them is the instrument for reading a near-tie divergence. The 26B top-2 card reads the three prompts of that parity fixture, whose generations end at the end-of-generation token (9, 12 and 12 ids), not at 32.
2. The granite parity card (FT0.43) rests on three things no run has shown: that `LoadedModel::load` binds a Q8_0 stacked-expert checkpoint through the single-range MoE layer, that the Metal path evaluates it, and that the four header scales are the whole difference from a llama-shaped MoE. Each is a false-premise stop with the printed first divergent index, not a tolerance.
3. FT0.20's model assumes tree-reduction error and FT0.21's rests on `|logit| <= 40`; both are assumptions recorded in their sections. For granite the bound is unmeasured.
4. The fsm-techniques SPEC (`proxima-tensor/specs/fsm-techniques/SPEC.md` on main) states the non-local count: the top blocks are chosen among the sealed blocks that are not local blocks. FT0.17 and FT0.29 use that rule and restate it in full, so they do not depend on the file and anchor nothing in it.
5. Ids elsewhere: a card in another file that cites `RESULT` text of a dropped example (the PAV fit, q-hat, k-means, per-cluster cost, Tchebycheff, distillation) has lost its source; those values are inputs of the dropped fitters. A card that `needs` FT0.9 only to have `worked-examples.md` exist needs FT0.14.
6. The label of the per-layer selection example is unchanged from the card it replaces: its RESULT line starts `RESULT per-layer recompute selection:`. A premise check in another file that greps `RESULT HKVD:` matches nothing against it; that check needs the label used here.
7. The four cards that change the positional parameter list of the single-range builder (FT0.33, FT0.35, FT0.36 and FT0.37 each append one parameter) edit the same two direct calls in the test file, so they are applied in id order; each lists the argument it appends.

## slice exit

Run after card 0.43, with cargo commands run from the checkout holding main and `env.sh` named by its absolute path, after `source proxima-tensor/specs/fsm-techniques/env.sh`. Model-loading rows (6 and 7) run alone, `-j 1`, after the quiet-box check. Each line is a command and the count it must print.

1. `cargo nextest run -p proxima-model-interop --features std -E 'test(/fsm_oracle_control_/)'` prints `5 passed`.
2. `git grep -nP '\b(struct|enum|trait)\s+\w*(Resa|ReSA|Cartridge|SleepTime|CacheBlend|Milvus|SealedSegment|GrowingSegment|Cascadia|Conformal|Ucci|UCCI|AoSpec|Sherlock|SpeculativeMacro|LmCache|VAttention)' -- proxima-model-interop/src proxima-tensor/src proxima-core/src omega/src | wc -l` prints `0`.
3. `cargo nextest run -p proxima-tensor -E 'test(/forward_scales::|head_repeats::|layer_taps_variant_matches/)'` prints `20 passed`.
4. `cargo nextest run -p proxima-tokenizer --features gguf -E 'binary(pre_tokenizer_llama_oracle) & test(/granite/)'` prints `1 passed` (the granite vocabulary oracle test; `pretokenize.rs` and `gguf.rs` are unchanged by this slice).
5. `cargo nextest run -p proxima-model-interop --features std -E 'test(/granitemoe_profile_is_adjacent|every_embedded_profile_parses|every_profile_pairs_rope_as_llama_cpp_rope_type_says|real_header_metadata_selects_the_pairing|header_scales_reach_the_descriptor|a_header_without_scale_keys|a_zero_scale_means_unset/)'` prints `7 passed`.
6. `cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'binary(arch_data_baseline) & test(/granite_moe/)'` prints `5 passed` (`granite_moe_header_declares_the_scales_the_profile_cannot_carry`, `arch_data_digest_granite_moe`, `generic_binder_granite_moe`, `granite_moe_program_carries_the_header_scales`, `llama_parity_granite_moe`).
7. `cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'binary(arch_data_baseline) & test(/^llama_parity_gemma4_e2b$/)'` prints `1 passed`.
8. `ls proxima-model-interop/tests/fixtures/llama-parity/{gemma4_e2b,gemma4_26b,granite_moe}/{followup_ids,n_probs,cache_reuse_ids}.json | wc -l` prints `9`.
9. `grep -c '^RESULT ' proxima-tensor/specs/fsm-techniques/worked-examples.md` prints `16` (one per card 0.14 to 0.29).
10. `for ckpt in gemma4_e2b gemma4_26b granite_moe; do jq -e '(.[0].request2_plain.generated_ids|length) == 32' proxima-model-interop/tests/fixtures/llama-parity/$ckpt/cache_reuse_ids.json; done | grep -c true` prints `3`.
11. `grep -lE 'FT[0-9]+\.[0-9]+|server\.log|\.long_ctx_backups' proxima-model-interop/tests/fixtures/llama-parity/{gemma4_e2b,gemma4_26b,granite_moe}/{followup_ids,n_probs,cache_reuse_ids}.json | wc -l` prints `0` (no card id or log path reached a committed fixture).
