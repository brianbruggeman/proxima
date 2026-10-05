# slice 13 cards (re-cut): the cascade's calibrated tier list, proved through the settle hook

anchors read at main a7c08c4c (full sha a7c08c4c95836f112742f0e30e4ddd672087cda0) with `git -C /Users/brianbruggeman/repos/slot-0/proxima-windows show main:<path>`. Every `path::symbol (~line N)` below is the line at that sha; the executor re-locates by symbol. Paths are relative to the proxima repo root.

Rules: CARDS.md applies to every card. Every card uses `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_13_<n>` and removes it when done. Logs go under `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/cards13/`. Test modules that use `unwrap` or `expect` carry `#[allow(clippy::unwrap_used, clippy::expect_used)]` (the workspace denies both; precedent `proxima-core/src/batch.rs` tests). No sleeps. No comments in code except a lowercase why.

## what this slice builds now

Owner direction (2026-10-04): the hooks are built so techniques can be vetted later; a technique is not built here, and training-side work (calibration fitting) is out of scope. The earlier cut built the calibration fitters, a 200 plus 200 row generator and a two-tier bench on qwen. This cut keeps what the settle stage still needs and nothing else.

The settle stage (a cascade of tiers, each with a judge) is a pure decision plus a configured list, and the cascade slice (`12-cascade.md`) builds the hook: the draw record, the tier, the cascade pipe and the validation of a tier list; the vote rule is not in the library, it is a function in that slice's own conformal proof test. This slice builds no library item. A calibration is only valid for the sampling configuration it was taken under, so the proof here looks its table up by a digest of that configuration (research change CC-10, card FT13.1); the digest lives in the proof test because nothing at serve time looks a calibration up.

The conformal technique appears only as the proof that the cascade's hooks express it on recorded model output: one test file, two replay closures of about 25 lines in all and a local vote rule, fed by draws recorded once from gemma4 (cards FT13.2 to FT13.4). The cascade slice's own conformal proof runs on scripted draws; this one runs the same pipe on real recorded draws and adds the digest lookup. No part of the technique is in the library.

Test models (owner, 2026-10-04): gemma4 E2B (`gemma4:e2b-it-qat`, dense) is tier 0 and gemma4 26B (`batiai/gemma4-26b`, MoE) is tier 1. No qwen of any kind. The 26B is the MoE arm; granite has no card here (see spec drift 5).

## old to new id map

| old card | new card | verdict |
|---|---|---|
| old FT13.1 to FT13.7 | none | dropped (listed below) |
| old FT13.8 | none | dropped: the tier list validation it cut is `CascadeError::Unsettled` and `CascadeError::Judge` of the cascade slice (`12-cascade.md`, FT12.5), which refuse a last tier whose judge declines and a judge that cannot use its draws |
| old FT13.9 | FT13.2, FT13.3 | recut: one recording run per tier, gemma4 E2B then gemma4 26B; the readout tap itself belongs to the readouts slice (spec drift 4) |
| old FT13.10 | FT13.1, FT13.4 | recut: the digest of the sampling fields, now a helper inside the proof test, and the proof test that replaces the in-example fit and report |

## dropped (no card body)

- old FT13.1 PAV isotonic fitter: calibration fitting, which the pipeline-as-data spec removes ("calibration fitters lose their cards"); nothing at serve time calls it, the judge only reads a fitted table.
- old FT13.2 conformal q-hat order statistic: a calibration fitter; the served judge needs only the integer `max_disagree` as configuration.
- old FT13.3 k-means and silhouette cluster fitter: training-side calibration that serves a router table no settle gap requires.
- old FT13.4 per-cluster lambda argmin: an offline fitter; the router consumes a fitted table and producing it is outside the serving pipeline.
- old FT13.5 Tchebycheff threshold choice: an offline fitter; no gap in sketch 13 needs it.
- old FT13.6 seeded 200-row needle generator: it only fed the dropped fitters and the bench calibration split; the proof needs 8 recorded rows, vendored by FT13.2.
- old FT13.7 vendored 200 plus 200 calibration and held-out rows: serves the dropped fitters; `tests/fixtures/cascade` does not exist at main and no hook needs 400 rows.

## prerequisites outside this slice (each card names the one it needs)

- The cascade slice (`12-cascade.md`), cards FT12.3, FT12.4, FT12.5 and FT12.6, for FT13.4 only. At main a7c08c4c none of it exists (`git ls-tree -r main --name-only | grep -E 'proxima-model-interop/src/cascade'` prints nothing), so FT13.4 runs after that slice lands. It must provide `proxima-model-interop/src/cascade.rs` with `CascadeError` (variant `Judge { reason: String }`), `Draw { ids, text, logprobs, margins }`, `Tier { serving, samples, draw, judge }` and `Cascade::new(&[Tier]) -> Result<Cascade, CascadeError>` implementing `Pipe` with `In = (String, usize)` and `Out = (usize, String)` (answering tier, its text), all reachable as `proxima_model_interop::cascade::*`. That slice adds no vote rule, no agreement type and no library file under `proxima-core`; the vote rule FT13.4 needs is written in its own test file. If any listed item is absent or differs, FT13.4 stops and reports; it does not define it.
- Test-name rule for this slice: every new test name says what it checks in English; none contains a card, slice or worked-example id, or the word `sketch`.

## cards

### 13.1 pin the sampling fields a calibration holds for

- id: FT13.1
- needs: none
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/src/serving.rs::ServingConfig (~line 720 at a7c08c4c)`: `temperature` (~796), `top_k` (~799), `top_p`, `min_p`, `repeat_last_n`, `repeat_penalty`, `frequency_penalty`, `presence_penalty` and `seed` (~829); the struct is `Copy` and reachable from a test crate as `proxima_model_interop::ServingConfig`;
  - `proxima-model-interop/src/serving.rs::Default for ServingConfig (~line 1119)`: the default values the pinned digests below were computed from (`temperature: 0.0`, `top_k: 0`, `top_p: 1.0`, `min_p: 0.0`, `repeat_last_n: 64`, `repeat_penalty: 1.0`, `frequency_penalty: 0.0`, `presence_penalty: 0.0`);
  - `proxima-model-interop/src/generate/decode.rs::logits_bits_hash (~line 1281)`: the eight-line FNV-1a fold over little-endian bytes to mirror (it is `cfg(metal)` and private, so the test file writes its own);
  - `proxima-model-interop/src/generate/prompt_cache_key.rs::CacheKey::of (~line 82)`: the prompt-cache key deliberately leaves the sampling fields out (`temperature: _` ~126); a calibration is the opposite case and needs exactly those fields.
- change:
  1. `proxima-model-interop/tests/cascade_recorded_draws.rs` (new). Header `#![cfg(feature = "std")]` and `#![allow(clippy::unwrap_used, clippy::expect_used)]`, then `use proxima_model_interop::ServingConfig;`, `const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;`, `const PRIME: u64 = 0x0000_0100_0000_01b3;` and, private to this file (the digest is a key for the proof in FT13.4; the library holds no calibration key, because nothing at serve time looks one up):
     ```rust
     fn calibration_digest(config: &ServingConfig<'_>) -> u64
     ```
     Body: destructure `let ServingConfig { temperature, top_k, top_p, min_p, repeat_last_n, repeat_penalty, frequency_penalty, presence_penalty, .. } = *config;` build `let fields = [temperature.to_bits().to_le_bytes(), top_k.to_le_bytes(), top_p.to_bits().to_le_bytes(), min_p.to_bits().to_le_bytes(), repeat_last_n.to_le_bytes(), repeat_penalty.to_bits().to_le_bytes(), frequency_penalty.to_bits().to_le_bytes(), presence_penalty.to_bits().to_le_bytes()];` and return `fields.iter().flatten().fold(OFFSET_BASIS, |hash, byte| (hash ^ u64::from(*byte)).wrapping_mul(PRIME))`. One lowercase why comment above the destructure: the seed is left out because the draws of one request are seeded base plus index, so any base seed is the same calibration. Model weights are not in the digest (a table lives beside its checkpoint); numeric policy and device are not in it either (unmeasured whether they move a calibration).
  2. Nothing else is edited; the test file needs no `Cargo.toml` entry.
- test: add in `cascade_recorded_draws.rs` (`#[cfg(test)] mod tests`, plain `#[test]`, a `ServingConfig { temperature: 0.7, ..ServingConfig::default() }` named `sampled` where used):
  - `calibration_digest_of_the_default_and_a_sampled_config_are_fixed_values`: `ServingConfig::default()` gives `0xe79f_03a4_b99c_7375`; `sampled` gives `0x87d6_d99e_3000_2dfd`; `sampled` with `top_k: 40` gives `0xbcbe_3907_e069_c0d5` (FNV-1a 64 over the 32 little-endian bytes in the order above, computed outside this repo from the default values listed in read first);
  - `calibration_digest_ignores_the_seed`: `sampled` with `seed: 7` and with `seed: 99` give equal digests;
  - `calibration_digest_moves_with_each_sampling_field`: `sampled` plus eight variants, each changing one field (`temperature: 0.8`, `top_k: 40`, `top_p: 0.9`, `min_p: 0.05`, `repeat_last_n: 32`, `repeat_penalty: 1.1`, `frequency_penalty: 0.5`, `presence_penalty: 0.5`); the nine digests collected into a `std::collections::HashSet` have length 9.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_13_1 cargo nextest run -p proxima-model-interop --features std --test cascade_recorded_draws -E 'test(/calibration_digest_/)'`
- expect: `3 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets` clean
- stage: `proxima-model-interop/tests/cascade_recorded_draws.rs`
- commit: `test(interop): pin the sampling fields a calibration holds for`
- done when: `3 passed` printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: edit `src/`, add a `calibration.rs`, a table type, a calibration struct or a fitter; hash `model_path` or `seed`; touch `prompt_cache_key.rs`
- gpu: none

### 13.2 record sampled draws from the small gemma4 tier

- id: FT13.2
- needs: none (reads the same sampling fields FT13.1 digests; shares no symbol)
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `proxima-model-interop/examples/long_context_niah.rs::NiahError (~line 57), parse_options (~line 191), run (~line 390)`: the `thiserror` error enum, the hand-rolled flag parser, and the mmap, `parse_complete`, `vocab_from_metadata`, `LoadedModel::load` sequence to mirror;
  - `proxima-model-interop/tests/arch_data_baseline.rs::llama_parity (~line 668)`: `wants_bos = vocab.add_bos_token().unwrap_or_else(|| vocab.bos_token_id().is_some())`, `wants_eos = vocab.add_eos_token().unwrap_or(false)` (~684) and `encode_with_bos_eos(.., wants_bos, wants_eos)`, then `generate_from_ids(&ids, n, &config, &mut |_event| ControlFlow::Continue(()))`; also `GEMMA4_E2B (~line 54)` and `GEMMA4_26B (~line 48)` for the blob paths;
  - `proxima-model-interop/src/generate/decode.rs::generate_from_ids (~line 2277)`: `(&self, prompt_ids: &[u32], max_tokens: usize, serving_config: &ServingConfig, on_token: &mut dyn FnMut(TokenEvent<'_>) -> ControlFlow<(), ()>) -> Result<(Vec<u32>, String, bool), InteropError>`;
  - `proxima-model-interop/src/serving.rs::ServingConfig (~line 720)`: `temperature` and `seed` are the two fields a draw varies; `Copy`, so a per-draw copy with `seed + index` is a struct update.
- change:
  1. `proxima-model-interop/tests/fixtures/tier_draws/rows.jsonl` (new): exactly these 8 lines (authored needle questions with a near-duplicate distractor; each prompt ends with `Answer:` and holds its label):
     ```
     {"prompt":"The rain had stopped by the time the carriage reached the edge of the village, and the lamps along the road were being lit one by one. The special magic number for marmot is 4830912. Nobody at the table spoke of the letter, though everyone had read it twice before supper. The special magic number for marten is 4830921. Across the river the mill kept turning, its wheel dark against the last of the light.\n\nWhat is the special magic number for marmot? Answer with the number only.\nAnswer:","label":"4830912"}
     {"prompt":"Nobody at the table spoke of the letter, though everyone had read it twice before supper. The special magic number for heron is 7216405. Across the river the mill kept turning, its wheel dark against the last of the light. The special magic number for egret is 7216450. A clerk in the next room copied figures into a ledger and did not look up when the door opened.\n\nWhat is the special magic number for egret? Answer with the number only.\nAnswer:","label":"7216450"}
     {"prompt":"Across the river the mill kept turning, its wheel dark against the last of the light. The special magic number for quince is 1957083. A clerk in the next room copied figures into a ledger and did not look up when the door opened. The special magic number for medlar is 1957038. The old general walked the length of the terrace twice before he decided to go in.\n\nWhat is the special magic number for quince? Answer with the number only.\nAnswer:","label":"1957083"}
     {"prompt":"A clerk in the next room copied figures into a ledger and did not look up when the door opened. The special magic number for lantern is 6024817. The old general walked the length of the terrace twice before he decided to go in. The special magic number for lamplighter is 6024871. Somewhere below, a cart rattled over the stones and a dog barked at it until it was gone.\n\nWhat is the special magic number for lamplighter? Answer with the number only.\nAnswer:","label":"6024871"}
     {"prompt":"The old general walked the length of the terrace twice before he decided to go in. The special magic number for harbor is 3398260. Somewhere below, a cart rattled over the stones and a dog barked at it until it was gone. The special magic number for harbour is 3398206. She folded the map along its worn creases and put it back in the drawer where it had always been kept.\n\nWhat is the special magic number for harbor? Answer with the number only.\nAnswer:","label":"3398260"}
     {"prompt":"Somewhere below, a cart rattled over the stones and a dog barked at it until it was gone. The special magic number for saffron is 8741530. She folded the map along its worn creases and put it back in the drawer where it had always been kept. The special magic number for turmeric is 8741503. By evening the wind had turned, and the smoke from the chimneys lay flat over the roofs.\n\nWhat is the special magic number for turmeric? Answer with the number only.\nAnswer:","label":"8741503"}
     {"prompt":"She folded the map along its worn creases and put it back in the drawer where it had always been kept. The special magic number for cobalt is 2569174. By evening the wind had turned, and the smoke from the chimneys lay flat over the roofs. The special magic number for cadmium is 2569147. The rain had stopped by the time the carriage reached the edge of the village, and the lamps along the road were being lit one by one.\n\nWhat is the special magic number for cobalt? Answer with the number only.\nAnswer:","label":"2569174"}
     {"prompt":"By evening the wind had turned, and the smoke from the chimneys lay flat over the roofs. The special magic number for juniper is 5083619. The rain had stopped by the time the carriage reached the edge of the village, and the lamps along the road were being lit one by one. The special magic number for cypress is 5083691. Nobody at the table spoke of the letter, though everyone had read it twice before supper.\n\nWhat is the special magic number for cypress? Answer with the number only.\nAnswer:","label":"5083691"}
     ```
  2. `proxima-model-interop/examples/tier_draws.rs` (new): a recorder whose non-test part holds at most 120 non-blank lines (counted by the command in `done when`), with these items and no others, the `#[cfg(test)] mod tests` last in the file:
     ```rust
     const TEMPERATURE: f32 = 0.7;
     const SEED: u64 = 7;
     const MAX_TOKENS: usize = 16;
     #[derive(Debug, thiserror::Error)]
     enum DrawError {
         #[error("usage: tier_draws <model.gguf> <rows.jsonl> <out.json> <samples>")] Usage,
         #[error("{path}: {source}")] Io { path: String, source: std::io::Error },
         #[error("gguf parse: {0}")] Gguf(String),
         #[error("{0}")] Json(#[from] serde_json::Error),
         #[error("{0}")] Interop(#[from] InteropError),
         #[error("{0}")] Tokenizer(#[from] TokenizerError),
     }
     #[derive(serde::Deserialize)] struct Row { prompt: String, label: String }
     #[derive(serde::Serialize, serde::Deserialize)] struct Draw { ids: Vec<u32>, text: String }
     #[derive(serde::Serialize, serde::Deserialize)] struct RowDraws { label: String, greedy: Draw, draws: Vec<Draw> }
     fn io_error(path: &str) -> impl Fn(std::io::Error) -> DrawError + '_
     fn load_rows(path: &str) -> Result<Vec<Row>, DrawError>
     fn draw(model: &LoadedModel, ids: &[u32], config: &ServingConfig) -> Result<Draw, DrawError>
     fn record_row(model: &LoadedModel, ids: &[u32], label: &str, samples: u64) -> Result<RowDraws, DrawError>
     fn run(args: &[String]) -> Result<Vec<RowDraws>, DrawError>
     fn main() -> ExitCode
     ```
     The program takes exactly four positional arguments, `<model.gguf> <rows.jsonl> <out.json> <samples>`; the temperature, base seed and token cap are the three constants (so the recording is reproducible from the command line alone). `run` matches `let [model_path, rows_path, out_path, samples] = args else { return Err(DrawError::Usage) };` and parses `samples` as `u64`, with a parse failure also `Usage`, both before any file is opened. `io_error` builds the `Io` variant from a path. `load_rows`: one `serde_json::from_str` per non-empty line. `draw`: `generate_from_ids(ids, MAX_TOKENS, config, &mut |_| ControlFlow::Continue(()))`, returning `Draw { ids, text }`. `record_row`: `greedy_config = ServingConfig { gpu_layers: GPU_LAYERS_ALL, prompt_cache: PromptCacheConfig::off(), temperature: 0.0, ..ServingConfig::default() }` and one greedy `draw`; then for `index` in `0..samples` a `draw` with the same config but `temperature: TEMPERATURE, seed: SEED + index` (so draw `i` is the exact call a user makes with seed `SEED + i`; no batching, no shared generator); returns `RowDraws { label: label.to_owned(), greedy, draws }`. `run` after the argument check: open and mmap the checkpoint read-only with a one-line lowercase why comment above the `unsafe`, `parse_complete` (its error through `to_string` into `Gguf`), `vocab_from_metadata`, `wants_bos` and `wants_eos` as in read first, `LoadedModel::load`, then per row tokenize `prompt` with `encode_with_bos_eos` and `record_row`; write `serde_json::to_string_pretty(&recorded)` (a JSON array of `RowDraws`) to `out_path` and return the rows. `main` collects `env::args().skip(1)`, prints `rows=<n> draws=<total draws over all rows> greedy=<rows whose greedy ids are non-empty>` on success (so every field is read in the non-test build), and on any error writes `tier_draws: <error>` to stderr and exits `FAILURE`. Why a loop of independent runs and not one batched decode: batched sampling is refused today (`parallel_sequences` must be 1), and N independent calls are what a deployed tier makes.
  3. `proxima-model-interop/Cargo.toml`: `[[example]] name = "tier_draws"`, `required-features = ["std", "metal"]`, `test = true` beside the `long_context_niah` entry (~line 434). `serde_json`, `serde` and `thiserror` are already dependencies (~lines 329 to 336).
  4. Run the recorder once (see validate) so `proxima-model-interop/tests/fixtures/tier_draws/tier0_e2b.json` exists; it is staged with the card.
- test: add in `tier_draws.rs` (`#[cfg(test)] mod tests`, `#[allow(clippy::unwrap_used, clippy::expect_used)]`; fixture paths are `Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tier_draws/<name>")`, passed to `load_rows` through `to_str`):
  - `tier_draws_refuses_missing_and_non_numeric_arguments`: `run(&[])` and `run` on three strings are `Err(DrawError::Usage)` (assert with `matches!`, `DrawError` is not `PartialEq`); `run` on `["m.gguf", "r.jsonl", "o.json", "abc"]` is `Err(DrawError::Usage)`; none of the three opens a file;
  - `tier_draws_rows_load_the_vendored_needle_rows`: `load_rows` on `rows.jsonl` returns 8 rows; every `label` is 7 ASCII digits; every `prompt` ends with `"Answer:"` and contains its own `label`;
  - `tier_draws_small_tier_recording_holds_every_draw_of_every_row`: `tier0_e2b.json` deserializes to a `Vec<RowDraws>` of length 8; every row has `draws.len() == 16`, non-empty `greedy.ids` and non-empty `ids` in every draw; the 8 `label`s equal the 8 labels of `rows.jsonl` in order.
- validate: after the quiet-box check, from the repo root, first `set -o pipefail; CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_13_2 cargo run -p proxima-model-interop --release --example tier_draws --features std,metal -- /Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd proxima-model-interop/tests/fixtures/tier_draws/rows.jsonl proxima-model-interop/tests/fixtures/tier_draws/tier0_e2b.json 16 | tee /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/cards13/tier0_e2b.log` (the blob is `gemma4:e2b-it-qat`; confirm with `ollama show --modelfile gemma4:e2b-it-qat`), then `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_13_2 cargo nextest run -p proxima-model-interop --features std,metal --example tier_draws -E 'test(/tier_draws_/)'`
- expect: the log prints `rows=8 draws=128 greedy=8` (zero on any count is RED), then `3 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets` clean
- stage: `proxima-model-interop/examples/tier_draws.rs`, `proxima-model-interop/Cargo.toml`, `proxima-model-interop/tests/fixtures/tier_draws/rows.jsonl`, `proxima-model-interop/tests/fixtures/tier_draws/tier0_e2b.json`
- commit: `test(interop): record sampled draws of the small gemma4 tier`
- done when: the expect lines printed, `sed '/^#\[cfg(test)\]/,$d' proxima-model-interop/examples/tier_draws.rs | grep -c '[^[:space:]]'` prints a number at most 120, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: query Ollama or llama; change a temperature, seed or row to change what was recorded; edit `src/`; if `apply_serving_config` or the memory fit refuses the config, report the refusal verbatim and stop
- gpu: one run (gemma4 E2B only), waiting for a quiet box (the peer-gate check in CARDS "machine safety"); one model-loading process at a time; no other agent or compile during the run

### 13.3 record greedy answers from the large gemma4 tier

- id: FT13.3
- needs: FT13.2
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `proxima-model-interop/examples/tier_draws.rs::record_row, run` (FT13.2): a `<samples>` argument of `0` records only the greedy answer, which is what an always-settle last tier returns;
  - `proxima-model-interop/tests/arch_data_baseline.rs::GEMMA4_26B (~line 48 at a7c08c4c)`: the 26B blob path (`gemma4_26b`, the MoE checkpoint, `batiai/gemma4-26b:latest`);
  - `proxima-model-interop/tests/fixtures/tier_draws/rows.jsonl` (FT13.2): the 8 rows both tiers answer.
- change:
  1. `proxima-model-interop/examples/tier_draws.rs`: add the test below; no other edit.
  2. Run the recorder once (see validate) so `proxima-model-interop/tests/fixtures/tier_draws/tier1_26b.json` exists; it is staged with the card.
- test: add `tier_draws_large_tier_recording_holds_a_greedy_answer_per_row` in `tier_draws.rs` tests: `tier1_26b.json` deserializes to a `Vec<RowDraws>` of length 8; every row has `draws.is_empty()` and non-empty `greedy.ids`; the 8 `label`s equal the labels of `rows.jsonl` in order.
- validate: after the quiet-box check, first `set -o pipefail; CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_13_3 cargo run -p proxima-model-interop --release --example tier_draws --features std,metal -- /Users/brianbruggeman/.ollama/models/blobs/sha256-ea549b7688d4c95019754880c21e3f29c58c985a7a1c3b37b9eebd0a95224129 proxima-model-interop/tests/fixtures/tier_draws/rows.jsonl proxima-model-interop/tests/fixtures/tier_draws/tier1_26b.json 0 | tee /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/cards13/tier1_26b.log` (confirm the blob with `ollama show --modelfile batiai/gemma4-26b:latest`), then `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_13_3 cargo nextest run -p proxima-model-interop --features std,metal --example tier_draws -E 'test(/tier_draws_/)'`
- expect: the log prints `rows=8 draws=0 greedy=8`, then `4 passed` (3 from FT13.2 plus 1)
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets` clean
- stage: `proxima-model-interop/examples/tier_draws.rs`, `proxima-model-interop/tests/fixtures/tier_draws/tier1_26b.json`
- commit: `test(interop): record the large gemma4 tier answers per needle row`
- done when: the expect lines printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: load the small tier in the same process; run another model-loading process alongside (the 26B is a 13 GB checkpoint; one model-loading process at a time, as the machine has panicked twice from concurrent loads); edit the recorder's non-test code
- gpu: one run (gemma4 26B only), waiting for a quiet box (the peer-gate check in CARDS "machine safety"); no other agent or compile during the run

### 13.4 replay recorded gemma4 draws through the cascade with a conformal vote judge

- id: FT13.4
- needs: FT13.1, FT13.2, FT13.3, FT12.3, FT12.4, FT12.5, FT12.6 (`CascadeError`, `Draw`, `Tier`, `Cascade`)
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/src/cascade.rs::{Cascade, Tier, Draw, CascadeError}` (cascade slice, cards FT12.3 to FT12.6; stop and report if absent): the pipe, the tier shape (`serving`, `samples`, `draw`, `judge`) and the judge closure type `Fn(&[Draw]) -> Result<Option<usize>, CascadeError>`; each draw runs with `seed = tier.serving.seed + index`, which is how the replay finds which recorded draw to return. The cascade library holds no vote rule and no agreement type, so this card writes its own in the test file, as the cascade slice's conformal test does;
  - `proxima-model-interop/tests/cascade_recorded_draws.rs::calibration_digest` (FT13.1): the key of the table the proof looks `max_disagree` up in; the file this card extends;
  - `proxima-model-interop/tests/fixtures/tier_draws/{rows.jsonl,tier0_e2b.json,tier1_26b.json}` (FT13.2, FT13.3): the prompts, the recorded draws and the recorded greedy answers the replay serves;
  - `proxima-model-interop/tests/arch_data_baseline.rs::use` list (~line 25 at a7c08c4c): `InteropError` and `ServingConfig` are imported from the crate root.
- change:
  1. `proxima-model-interop/tests/cascade_recorded_draws.rs` (extend the file FT13.1 created): the imports `proxima_model_interop::cascade::{Cascade, CascadeError, Draw, Tier}`, `proxima_model_interop::InteropError`, `proxima_primitives::pipe::Pipe`, `std::collections::HashMap`; these local items, which live only here:
     ```rust
     #[derive(serde::Deserialize)] struct Row { prompt: String, label: String }
     #[derive(serde::Deserialize)] struct Recorded { ids: Vec<u32>, text: String }
     #[derive(serde::Deserialize)] struct RowDraws { label: String, greedy: Recorded, draws: Vec<Recorded> }
     #[derive(Clone, Copy)] struct Agreement { samples: u32, max_disagree: u32 }
     fn fixture<Item: serde::de::DeserializeOwned>(name: &str) -> Vec<Item>
     fn replayed(recorded: &Recorded) -> Draw
     fn settle(agreement: Agreement, votes: &[u32]) -> Option<usize>
     fn vote_judge(agreement: Agreement) -> impl Fn(&[Draw]) -> Result<Option<usize>, CascadeError>
     ```
     `fixture` reads `tests/fixtures/tier_draws/<name>` under `CARGO_MANIFEST_DIR`: `rows.jsonl` through one `serde_json::from_str` per non-empty line, the two recordings as one JSON array; it panics with the path on failure. `replayed` builds `Draw { ids: recorded.ids.clone(), text: recorded.text.clone(), logprobs: Vec::new(), margins: Vec::new() }`. `settle` counts the votes whose disagreement `agreement.samples.saturating_sub(count)` is at most `agreement.max_disagree`: exactly one such vote returns `Some(its position)`, none or several return `None`. `vote_judge` returns a closure that errors with `CascadeError::Judge { reason: format!("agreement expects {} draws, the tier drew {}", agreement.samples, draws.len()) }` when `draws.len()` differs from `agreement.samples`, else groups the draws by equal `ids` in first-seen order (a `Vec<usize>` of first-draw positions and a `Vec<u32>` of counts), and returns `Ok(settle(agreement, &counts).map(|group| firsts[group]))`. The two replay closures are written inside the first test: the small tier's drawer finds the row whose `prompt` equals the prompt it is given, takes `index = serving.seed - 7`, and returns `replayed(&small[row].draws[index])`; the large tier's drawer finds the row the same way and returns `replayed(&large[row].greedy)`. Both drawers are annotated `|prompt: &str, _: usize, serving: ServingConfig<'_>| -> Result<Draw, InteropError>` and the small tier's judge is bound with `let` before the tier array is built.
  2. Nothing else is edited; the test file needs no `Cargo.toml` entry.
- test: two tests added in `cascade_recorded_draws.rs`:
  - `recorded_gemma4_rows_split_into_settled_and_escalated`: `sampled = ServingConfig { temperature: 0.7, seed: 7, ..ServingConfig::default() }`; `table = HashMap::from([(calibration_digest(&sampled), 4_u32)])` (4 is the value the nine-item calibration example yields, 16 draws per item at a miss rate of 0.2, not a fit on these rows); `max_disagree = *table.get(&calibration_digest(&sampled)).expect("the sampled config is calibrated")`; tier 0 is `Tier { serving: sampled, samples: 16, draw: &small_drawer, judge: &small_judge }` with `small_judge = vote_judge(Agreement { samples: 16, max_disagree })`, and tier 1 is `Tier { serving: ServingConfig::default(), samples: 1, draw: &large_drawer, judge: &|_: &[Draw]| Ok(Some(0)) }`; `rows` has 8 rows, both recordings have 8 rows, and their labels equal the `rows.jsonl` labels row by row. For each row run `proxima_primitives::block_on(cascade.call((row.prompt.clone(), 16)))` to get `(tier, text)`. Count rows answered by tier 0 (`settled`) and by tier 1 (`escalated`); print `settled={settled} escalated={escalated}` with `eprintln!`; assert `settled + escalated == 8`. Recount independently and per row, from the recorded draws and not through the cascade: a `HashMap<&[u32], u32>` of votes per distinct `ids`; the row is settled iff exactly one count `votes` has `16 - votes <= max_disagree`; assert that verdict equals the cascade's answering tier for the row. For a settled row assert the served text equals the `text` of the first draw holding that one member's ids; for an escalated row assert the served text equals tier 1's `greedy.text`. N is 8 rows, never 0;
  - `a_table_fitted_under_one_sampling_config_is_not_found_under_another` (the control that must fail): with the same `table`, `table.get(&calibration_digest(&sampled))` is `Some(&4)`; the same lookup for `ServingConfig::default()` (greedy) is `None`; for `sampled` with `top_k: 40` is `None`; a tier whose configuration is not in the table therefore has no `max_disagree` to hand its vote judge.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_13_4 cargo nextest run -p proxima-model-interop --features std --test cascade_recorded_draws`
- expect: `5 passed` (3 from FT13.1, 2 new), and the first new test prints `settled=<s> escalated=<e>` with `s + e = 8` (run it once with `-- --nocapture` and record the line in the report)
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets` clean
- stage: `proxima-model-interop/tests/cascade_recorded_draws.rs`
- commit: `test(interop): replay recorded gemma4 draws through the cascade`
- done when: `5 passed` printed, the `settled=` line recorded, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: edit `src/` or add a vote rule, agreement type, fitter or calibration routine to it; copy the cascade slice's scripted conformal cases or vote-tally cases; gate on how many rows settled (the count is a recorded fact, not a target); change the table value to move a count; load a model
- gpu: none

---

## spec drift

1. The earlier cut put five fitters, a row generator, the 200 plus 200 fixtures and a qwen bench in this slice. All are removed: the fitters are training-side calibration, the generator and fixtures fed only the fitters, and the bench pinned qwen3 tiers. The recorded 8-row proof replaces the bench; no coverage claim is made (spec drift 3).
2. The earlier cut judged conformal on a floating q-hat. This cut requires the integer `max_disagree` of sketch 13: one disagreeing draw of 16 is 62.5 per thousand, not an integer, so a float or a milli form does not compare exactly. The local `Agreement` in the proof test carries that form (`max_disagree: u32`); the library holds no agreement type.
3. Coverage is not claimed. The table value 4 in FT13.4 is the hand-derived value from the sketch's nine-item example (alpha 0.2), not a fit on the 8 recorded rows, and no calibration was run. The proof asserts that the hooks compose (each row's answering tier agrees with an independent recount of the recorded draws, the digest lookup misses under another configuration), not that the conformal guarantee holds for gemma4 on this corpus. Status: unmeasured.
4. The readout tap (a logprob and a top-2 margin on every token event, sketch 13 gap G5) belongs to the readouts slice, which already owns the token event and its oracles against llama-server. A threshold, classifier or isotonic judge would consume it; none is proved here because the conformal judge needs only votes. If the readouts slice does not land the tap, a readout-driven judge has no proof card.
5. Granite: this slice records gemma4 E2B (dense) and gemma4 26B (MoE). No granite card is here because granite loads as configuration through a family profile and descriptor owned by the oracle and descriptor slices; a granite tier would be one more recording command against FT13.2's recorder, not code.
6. The draws are recorded by proxima's own engine at temperature 0.7 with seeds 7 to 22, not by an incumbent. That is acceptable only because the proof is about control flow (tally, settle, escalate); proxima's sampled numerics are not claimed correct by this slice. The 8 rows are authored needle questions, not a corpus.
7. Draw votes key on exact token ids (sketch 13 section 2), so answers that differ only in wording count as different answers. The technique suits constrained answers such as a 7-digit number; the limit for free text is unmeasured.

## slice exit

- Every card FT13.1 to FT13.4 printed its expect line.
- `cargo nextest run -p proxima-model-interop --features std --test cascade_recorded_draws -E 'test(/calibration_digest_/)'` prints `3 passed`.
- `cargo nextest run -p proxima-model-interop --features std,metal --example tier_draws -E 'test(/tier_draws_/)'` prints `4 passed`.
- `cargo nextest run -p proxima-model-interop --features std --test cascade_recorded_draws` prints `5 passed`, and its first recorded-draws test printed `settled=<s> escalated=<e>` with `s + e = 8`.
- `wc -l < proxima-model-interop/tests/fixtures/tier_draws/rows.jsonl` prints `8`.
- `sed '/^#\[cfg(test)\]/,$d' proxima-model-interop/examples/tier_draws.rs | grep -c '[^[:space:]]'` prints a number at most 120.
- `git grep -n -i "qwen" -- proxima-model-interop/tests/fixtures/tier_draws proxima-model-interop/examples/tier_draws.rs proxima-model-interop/tests/cascade_recorded_draws.rs` prints nothing.
- `git grep -n -E "FT[0-9]+\.[0-9]+|AC[0-9]+|sketch" -- proxima-model-interop/tests/fixtures/tier_draws proxima-model-interop/examples/tier_draws.rs proxima-model-interop/tests/cascade_recorded_draws.rs` prints nothing.
