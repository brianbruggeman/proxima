# slice 15 cards (re-cut): long-context read arms as a driver of the read and step hooks

anchors read at main a7c08c4c (full sha a7c08c4c95836f112742f0e30e4ddd672087cda0). Read each with
`git -C /Users/brianbruggeman/repos/slot-0/proxima-windows show main:<path>`; the working tree is not the source.
Paths are relative to the proxima repo root. Every `~line N` is a hint; re-locate by symbol.
Every `path::symbol` anchor in a `read first` list exists on that main. Symbols that a `needs` card adds are absent on main by
design; they are named without a path, marked "added by FTn.m", and the executor locates them with `git grep -n <name>` once that card
has landed (if the grep finds nothing, the premise is false and the card stops).

Rules: CARDS.md applies to every card. Every card uses `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_15_<n>` and removes it when
done. Logs go under `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/cards15/`.

## what this file builds

Nothing in a library crate. The long-context harness (`proxima-model-interop/examples/long_context_niah.rs`) becomes the worked driver
of two hooks the other slices build, so a configured arm replaces a code change:

- the read stage: a `block` arm sets `ServingConfig.attention.read` to the read operand and hands the loaded model one pure read rule;
- the step stage: a `block-replay` arm adds one periodic entry rule to `ServingConfig.decode.enter`;
- the key-row counter the read stage exposes (`LoadedModel::take_kv_read_stats`) is printed on every arm line and checked against a
  pure row-count formula.

The technique appears only as proof that the hook expresses it. The block read rule here is about 25 lines, lives in the example's own
module file, and is a position rule: it keeps the leading blocks, the last sealed block and the unsealed tail. The model-side read rule
is a function of the layer index and the cached row count only (it cannot see keys), so a score-driven block selection cannot run live
through it; that selection is proven on real captured rows in the read-stage slice. This arm proves the binding and the counter, and
its `found=` is the recall of a position rule, not of any published technique.

## old to new id map (the previous cut of slice 15 is tasks/15-niah-read-arms.md)

| old | verdict | new |
|---|---|---|
| FT15.1 `--read` and `--rectify` flags select the arm's serving config | kept, premise changed | FT15.1 (the flag selects `dense`, `block`, `block-replay`; no keep ratio in the config, no rectify flag) |
| FT15.2 print rows read, check the formula, qwen3-8B run | recut | FT15.2 (the read rule and the layer pattern), FT15.3 (the printed counter, the per-layer formula, one gemma4 E2B run) |

## dropped

- The `RESA_KEEP_RATIO` constant read from a published-value line in the spec: the serving config no longer carries a keep ratio (the read
  rule is a function on the model), and the line is not needed to run the arm. The ratio is a `--keep-milli` flag (default 100).
- The `--rectify` flag and `DecodeConfig.rectify_every`: the periodic replay is one entry in the configured `decode.enter` list.
- The `$QWEN3_8B` run and the 36-layer flat formula with the 83052, 267372 and 295020 constants as run expectations: owner rule (gemma4
  for dense or MoE, granite for MoE, no qwen of any kind). The pure formula test keeps the worked 36-layer values as plain arithmetic
  with an explicit layer list; the run is on gemma4 E2B and gemma4 26B.
- A granite arm: the read operand program exists for gemma4 only (it is an engine change on a single-range cached engine), so a block
  arm on granite is refused by the model with a typed error and printed as `rejected=`. No granite card is cut.

## cross-file contract (symbols added by the `needs` cards; none of them is on main at a7c08c4c; the executor of a card stops when its premise is false)

- `ReadSpec` and `AttentionConfig`, exported from `proxima_model_interop` (added by FT2.1): `ServingConfig.attention.read` is a `Copy` `ReadSpec` with variants `Dense`
  (default) and `Operand` (the read-stage slice, FT2.1).
- `DecodeConfig` (added by FT7.3) and `EnterRule` (defined in `proxima-core`), both exported from `proxima_model_interop` by FT7.4: `ServingConfig.decode` is `DecodeConfig<'model> { enter: &'model [EnterRule] }`
  (one field)
  with `EnterRule::{DraftNonempty, Periodic { every_rows, replay_rows }}`, and a periodic rule runs in the decode loop (FT7.6).
- `LoadedModel::set_read_rule` (added by FT6.11; `LoadedModel` itself is `proxima-model-interop/src/generate/load_model.rs::LoadedModel`, ~line 803) is `(&mut self, rule: impl Fn(usize, usize, &mut [f32]) + Send + Sync + 'file)`: arguments are the layer index,
  the number of cached rows, and a slice of length `cached_rows` the rule marks with 1.0 for each row to skip (FT6.11).
- `LoadedModel::take_kv_read_stats` (added by FT6.12) is `(&self) -> (u64, u64)`: `(key rows read, single-row decode steps)`, reset on read; `(0, 0)` under
  `ReadSpec::Dense`; rows are summed over every attention layer that carries the read operand (35 for gemma4 E2B, 30 for gemma4 26B),
  and a step adds `cached_rows + 1` minus the flagged rows per layer (FT6.9).
- `seal_target` (added by FT4.0), a pure function in `proxima-core`; `proxima-model-interop` depends on `proxima-core` in its `std` feature (FT1.5). The block keep count and the attended row count are not library functions (slice 6 dropped them as technique arithmetic); this file writes them as private helpers in `read_arm.rs`.

## shared worked values

Block geometry used by the arms: block size 64 rows, seal horizon 64 rows, at least 16 kept blocks, 1 local block.

One decode step with `len = cached_rows + 1` rows after the step's own row is appended; `sealed_end = seal_target(len, 64, 64)`,
`sealed_blocks = sealed_end / 64`, `nonlocal = sealed_blocks - 1`:

| len | sealed_end | sealed_blocks | nonlocal | keep at 100 milli | keep at 900 milli | rows attended at 100 | rows attended at 900 |
|---|---|---|---|---|---|---|---|
| 4097 | 4032 | 63 | 62 | 16 | 56 | (16+1)*64 + 65 = 1153 | (56+1)*64 + 65 = 3713 |
| 4098 | 4032 | 63 | 62 | 16 | 56 | (16+1)*64 + 66 = 1154 | (56+1)*64 + 66 = 3714 |

The position rule at `cached_rows = 4096`, 100 milli: flags rows `1024..3968` (`keep * 64 .. nonlocal * 64`), which is 2944 rows; rows read
`= 4096 + 1 - 2944 = 1153`. At 900 milli the flagged range is `3584..3968` (384 rows) and rows read `= 4097 - 384 = 3713`.

Per-layer totals over two decode steps (prompt 4096): a full-attention layer reads `1153 + 1154 = 2307` rows at 100 milli (`3713 + 3714 = 7427`
at 900); a sliding layer carries no flags from this rule and reads every unflagged row the counter sees, `4097 + 4098 = 8195`.

| layout | full layers | sliding layers | block at 100 milli | block at 900 milli |
|---|---|---|---|---|
| 36 full layers (plain arithmetic) | 36 | 0 | 36 * 2307 = 83052 | 36 * 7427 = 267372 |
| gemma4 E2B: 35 layers, every 5th full | 7 | 28 | 7 * 2307 + 28 * 8195 = 245609 | 7 * 7427 + 28 * 8195 = 281449 |
| gemma4 26B: 30 layers, every 6th full | 5 | 25 | 5 * 2307 + 25 * 8195 = 216410 | 5 * 7427 + 25 * 8195 = 242010 |

Dense (`ReadSpec::Dense`): the counter reads `(0, 0)`, so the expected rows are 0 and the printed `decode_tokens=` is 0.

## decided in this file (and why)

- The arms are `dense` (read `Dense`, today's serving defaults), `block` (read operand, position rule, speculation off) and `block-replay`
  (the `block` arm plus the periodic entry rule `every_rows = 32, replay_rows = 8`). Speculation is off for the two block arms because the
  counter counts single-row steps only and the formula sums over consecutive single-row steps; a verify step would break that.
- The printed `decode_tokens` is the counter's second value (single-row steps with a read operand). The first emitted token comes from the
  prefill's logits, so decode step `s` (from 0) holds `prompt_tokens + s` cached rows before it appends its own.
- Which layers are full attention comes from the checkpoint header (`<arch>.attention.sliding_window_pattern`, true = sliding), not from
  `Architecture::kv_layers`: that list holds only the layers that own a cache (15 for E2B), and the counter sums over all attention layers
  (35), including those that share a donor's cache.
- The counter counts rows the read rule leaves unflagged. The sliding-window mask is a separate, existing mask, so a sliding layer's count
  is `cached_rows + 1` while the rule flags nothing there. If the counter on a real model reports a different value for a sliding layer, the
  arm prints `mismatch=1` and fails; the executor stops and reports the printed numbers, it does not adjust the formula.
- The `block-replay` arm prints its measured rows only: its replay passes run through the dense verify program and are not counted, but the
  single-row steps between them are, and the block arm's formula is not asserted for it.
- The keep ratio is a flag (`--keep-milli`, integer thousandths, default 100), because the keep count arithmetic (`keep_block_count`, private to `read_arm.rs`) takes integer thousandths.

## cards

### 15.1 `--read` selects the arm's serving config

- id: FT15.1
- needs: FT2.1, FT7.3, FT7.4, FT7.6
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `proxima-model-interop/examples/long_context_niah.rs::parse_options (~line 191 at a7c08c4c)`, `::Options (~line 106)` and `::parse_list (~line 184)`: the flag parser, and the `--kv` list flag as the precedent for a list flag;
  - `proxima-model-interop/examples/long_context_niah.rs::serving_config (~line 277)` and `::run_arms (~line 363)`: where a per-arm `ServingConfig` is built, and the arms loop;
  - `proxima-model-interop/examples/long_context_niah/scoring.rs` (a sibling module file declared with `#[path = "long_context_niah/scoring.rs"] mod scoring;` at the top of the example): the module layout the new file copies;
  - `proxima-model-interop/src/serving.rs::SpeculativeConfig::none (~line 412)`: the speculation off switch; `ReadSpec` and `AttentionConfig` (FT2.1), `DecodeConfig` (FT7.3) and the `proxima_model_interop` export of `DecodeConfig` and `EnterRule` (FT7.4) are absent on main, located with `git grep -n 'struct DecodeConfig'` and `git grep -n 'pub use proxima_core::serving_state::enter::EnterRule'` before editing (nothing found means the premise is false and the card stops).
- change:
  1. new file `proxima-model-interop/examples/long_context_niah/read_arm.rs`, declared in the example as `#[path = "long_context_niah/read_arm.rs"] mod read_arm;` beside the other three: `#[derive(Debug, Clone, Copy, PartialEq)] pub enum ReadArm { Dense, Block, BlockReplay }` with `pub fn label(self) -> &'static str` (`dense`, `block`, `block-replay`) and `pub fn from_label(name: &str) -> Option<Self>` (the inverse).
  2. `long_context_niah.rs`: `use read_arm::ReadArm;`, `fn parse_read_arm(name: &str) -> Result<ReadArm, NiahError>` = `ReadArm::from_label(name).ok_or_else(|| usage(format!("--read knows dense, block, block-replay; got {name}")))`. `Options` gains `read_arms: Vec<ReadArm>` (default `vec![ReadArm::Dense]`), parsed beside `--kv` as `"--read" => read_arms = parse_list(&value, parse_read_arm)?`. Add `[--read dense,block,block-replay]` to the `NiahError::Usage` text and to the module doc usage block.
  3. `serving_config` takes a trailing `arm: ReadArm`. Add `const REPLAY_ENTER: &[EnterRule] = &[EnterRule::Periodic { every_rows: 32, replay_rows: 8 }];` and, in the body, `let (read, speculative, enter) = match arm { ReadArm::Dense => (ReadSpec::Dense, defaults.speculative, defaults.decode.enter), ReadArm::Block => (ReadSpec::Operand, SpeculativeConfig::none(), defaults.decode.enter), ReadArm::BlockReplay => (ReadSpec::Operand, SpeculativeConfig::none(), REPLAY_ENTER) }` where `let defaults = ServingConfig::default();`; the returned config sets `attention: AttentionConfig { read }`, `speculative`, `decode: DecodeConfig { enter }` beside its existing fields (`DecodeConfig` has the one field, so no base expression).
  4. `run_arms` loops `for &arm in &options.read_arms` outside the kv and rope loops, passes `arm` to `serving_config`, and prints `arm=<label> ` before the existing `kv=.. rope=..` label.
- test: add in the example's `#[cfg(test)] mod tests` (helper `fn args(line: &str) -> impl Iterator<Item = String>` splitting on spaces):
  - `niah_read_flag_parses_the_arm_list`: `--model m --ctx 8 --needles 1 --read dense,block,block-replay` gives `read_arms == vec![ReadArm::Dense, ReadArm::Block, ReadArm::BlockReplay]`; `--model m --ctx 8 --needles 1` gives `vec![ReadArm::Dense]`;
  - `niah_read_flag_rejects_an_unknown_arm`: `--model m --ctx 8 --needles 1 --read dense,sparse` is `Err(NiahError::Usage(message))` with `message.contains("sparse")`;
  - `niah_serving_config_carries_each_arm`: with `Options` from `parse_options` of `--model m --ctx 8 --needles 1` and `GgmlType::F32`, `None` rope: `Dense` has `attention.read == ReadSpec::Dense`, `speculative == ServingConfig::default().speculative` and `decode.enter == [EnterRule::DraftNonempty]`; `Block` has `ReadSpec::Operand`, `speculative == SpeculativeConfig::none()` and `decode.enter == [EnterRule::DraftNonempty]`; `BlockReplay` has `ReadSpec::Operand`, `speculative == SpeculativeConfig::none()` and `decode.enter == [EnterRule::Periodic { every_rows: 32, replay_rows: 8 }]`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_15_1 cargo nextest run -p proxima-model-interop --example long_context_niah --features std,metal -E 'test(/niah_read_flag_|niah_serving_config_carries_each_arm/)'`
- expect: `3 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/examples/long_context_niah.rs`, `proxima-model-interop/examples/long_context_niah/read_arm.rs`
- commit: `feat(interop): select attention read arms in the niah harness`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: touch `src/` or `Cargo.toml`; change the ollama arm; add a keep-ratio flag yet; set a read rule (the next card does); put a technique in a library crate.
- gpu: none

### 15.2 the block arm sets a position read rule on the model

- id: FT15.2
- needs: FT15.1, FT6.11, FT4.0, FT1.5
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `proxima-model-interop/examples/long_context_niah.rs::run (~line 390 at a7c08c4c)` and `::run_arms (~line 363)`: where `parsed` and `model` live, and the arm loop (FT15.1) the rule is set in;
  - `seal_target` (added by FT4.0, `proxima-core/src/kv_decision.rs`, absent on main): the pure function the rule calls, located with `git grep -n 'fn seal_target'`; it is reached as `proxima_core::kv_decision::seal_target` (interop depends on `proxima-core`, FT1.5);
  - `set_read_rule` (added by FT6.11, a method on `proxima-model-interop/src/generate/load_model.rs::LoadedModel (~line 803)`; absent on main): the rule signature and the slice convention (length `cached_rows`, 1.0 skips the row);
  - `proxima-model-interop/src/bind.rs::metadata_bool_per_layer (~line 773)` and `proxima-model-interop/src/gemma4/hparams.rs::from_metadata (~line 82)`: how `general.architecture` and `<arch>.attention.sliding_window_pattern` (a `MetadataArray::Bool`, true = sliding layer) are read from the header; both are crate-private, so the example reads the key itself.
- change:
  1. `proxima-model-interop/examples/long_context_niah/read_arm.rs`: add constants `BLOCK_TOKENS: usize = 64`, `SEAL_HORIZON_ROWS: usize = 64`, `MIN_BLOCKS: usize = 16`, `LOCAL_BLOCKS: usize = 1`, and
     - `fn keep_block_count(sealed_blocks: usize, keep_milli: u32) -> usize` (private): `nonlocal = sealed_blocks - LOCAL_BLOCKS.min(sealed_blocks)`; `proportional = usize::try_from(keep_milli).map_or(usize::MAX, |milli| milli.saturating_mul(nonlocal)).div_ceil(1000)`; return `MIN_BLOCKS.max(proportional).min(nonlocal)` (the number of non-local sealed blocks attended; integer thousandths because a float ratio drifts at `ceil(0.1 * 30)`);
     - `pub fn skipped_row_range(len: usize, keep_milli: u32) -> Range<usize>`: `sealed_blocks = seal_target(len, BLOCK_TOKENS, SEAL_HORIZON_ROWS) / BLOCK_TOKENS`; `nonlocal = sealed_blocks - LOCAL_BLOCKS.min(sealed_blocks)`; `keep = keep_block_count(sealed_blocks, keep_milli)`; return `keep * BLOCK_TOKENS..nonlocal * BLOCK_TOKENS` (the leading `keep` blocks, the last sealed block and the tail stay visible);
     - `pub fn block_read_rule(full_layers: Vec<bool>, keep_milli: u32) -> impl Fn(usize, usize, &mut [f32]) + Send + Sync`: a `move` closure `|layer, cached_rows, skip|` that, when `full_layers.get(layer) == Some(&true)`, fills `skip[skipped_row_range(cached_rows + 1, keep_milli)]` with 1.0 and does nothing otherwise;
     - `pub fn full_layers_from_metadata(parsed: &ParsedGguf) -> Result<Vec<bool>, String>`: reads `general.architecture`, then `<arch>.attention.sliding_window_pattern` as `MetadataValue::Array(MetadataArray::Bool(values))` and returns each value negated (true = full attention); a missing key or any other type is `Err(format!("gguf has no {key} bool array"))`.
  2. `long_context_niah.rs`: `Options` gains `keep_milli: u32` (default 100), parsed as `"--keep-milli" => keep_milli = parse_number(&flag, &value)?`; add `[--keep-milli N]` to the usage text. Add `ReadArm::is_block(self) -> bool` (`Block` or `BlockReplay`) in `read_arm.rs`. In `run`: `let mut model = LoadedModel::load(..)?`; `let full_layers = if options.read_arms.iter().any(|arm| arm.is_block()) { read_arm::full_layers_from_metadata(&parsed).map_err(NiahError::Failed)? } else { Vec::new() };` passed to `run_arms(&options, &mut model, &case, &layers, &full_layers)`. In `run_arms`, before running a block arm (`arm.is_block()`): `model.set_read_rule(read_arm::block_read_rule(full_layers.to_vec(), options.keep_milli));`.
- test: add in `read_arm.rs` `#[cfg(test)] mod tests` (cases through `#[proxima::test]` as `niah_scoring_shared` does):
  - `niah_block_rule_flags_the_worked_row_range`: `block_read_rule(vec![true, false], 100)`; layer 0 with a zeroed slice of 4096 (cached rows): 2944 values are 1.0, indices 1023 and 3968 are 0.0, indices 1024 and 3967 are 1.0, and `4096 + 1 - 2944 == 1153`; layer 1 on a fresh zeroed slice flags 0 values;
  - `niah_keep_block_count_follows_the_worked_values`: `keep_block_count(63, 100) == 16`, `keep_block_count(63, 900) == 56`, `keep_block_count(0, 100) == 0`, `keep_block_count(1, 900) == 0` (every block local), each assert with a message naming the tuple;
  - `niah_block_rule_agrees_with_the_written_out_row_count`: for `keep_milli` in `[100, 900]` and `cached_rows` in `[4096, 4097]`, with `len = cached_rows + 1`, `sealed_end = 4032` (the shared worked value, not computed), `keep` the literal 16 at 100 milli and 56 at 900 milli: `cached_rows + 1 - flagged_count == (keep + 1) * 64 + (len - sealed_end)` (1153, 1154, 3713, 3714 in order);
  - `niah_block_rule_flags_nothing_before_a_block_is_sealed`: `cached_rows = 40` flags 0 values on a full layer;
  - `niah_full_layer_pattern_follows_the_header` with cases `e2b(35, 5, 7)` and `moe_26b(30, 6, 5)` (layers, period, full count): a `ParsedGguf` built with the seven public fields (`version: 3`, `tensor_count: 0`, `kv_count: 0`, `metadata`, `tensors: vec![]`, `data_offset: 0`, `alignment: 32`) whose metadata holds `general.architecture = "gemma4"` and `gemma4.attention.sliding_window_pattern` as a `Bool` array of `(layer + 1) % period != 0` over `layers`; the result has `layers` entries, exactly `full count` true values, `result[period - 1]` true and `result[0]` false;
  - `niah_full_layer_pattern_names_the_missing_key`: the same header without the pattern key is `Err(message)` with `message.contains("sliding_window_pattern")`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_15_2 cargo nextest run -p proxima-model-interop --example long_context_niah --features std,metal -E 'test(/niah_block_rule_|niah_keep_block_count_|niah_full_layer_pattern_/)'`
- expect: `7 passed` (three `niah_block_rule_`, one `niah_keep_block_count_`, and `niah_full_layer_pattern_`: the two `follows_the_header` cases count as two plus one `names_the_missing_key`; count derived from the filter, not run)
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`; `cargo check -p proxima-model-interop --features std,metal --examples`
- stage: `proxima-model-interop/examples/long_context_niah.rs`, `proxima-model-interop/examples/long_context_niah/read_arm.rs`
- commit: `feat(interop): drive block reads with a position rule in niah`
- done when: the expect line printed, clippy and the check clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: edit `src/` or `Cargo.toml`; read key rows or scores in the rule (it only sees layer and row count); add a function, type or field to a library crate; run the model.
- gpu: none

### 15.3 print each arm's key rows read and check them against the row formula

- id: FT15.3
- needs: FT15.2, FT6.12
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `proxima-model-interop/examples/long_context_niah.rs::run_proxima_arm (~line 304 at a7c08c4c)`: the arm result line (`found=.. kv_bytes=.. peak_metal_bytes=..`), and `::report_arm (~line 380)`: how `NiahError::Failed` becomes a failed arm line (`error=..`);
  - `proxima-model-interop/examples/long_context_niah.rs::run_arms` (FT15.1, FT15.2): where `arm`, `case`, `options` and `full_layers` are in scope;
  - `take_kv_read_stats` (added by FT6.12, a method on `proxima-model-interop/src/generate/load_model.rs::LoadedModel (~line 803)`; absent on main): `(rows, steps)`, reset on read;
  - `proxima-model-interop/examples/long_context_niah/read_arm.rs::keep_block_count` (FT15.2) and `seal_target` (added by FT4.0, `proxima-core`, absent on main): the formula reuses the keep count and the seal rule the position rule calls; the row count is written out by the formula itself, not read off the rule's flags.
- change:
  1. `proxima-model-interop/examples/long_context_niah/read_arm.rs`: add `pub fn expected_kv_rows_read(arm: ReadArm, full_layers: &[bool], keep_milli: u32, prompt_tokens: usize, decode_steps: u64) -> Option<u64>`: `Dense` is `Some(0)`; `BlockReplay` is `None`; `Block` is `None` when `full_layers` is empty, else the sum over `step in 0..decode_steps` of, with `len = prompt_tokens + step + 1` (converted with `usize::try_from`, `None` on failure), `sealed_end = seal_target(len, 64, 64)`, `sealed_blocks = sealed_end / 64`, `keep = keep_block_count(sealed_blocks, keep_milli)`: `full_count * ((keep + LOCAL_BLOCKS.min(sealed_blocks)) * BLOCK_TOKENS + (len - sealed_end)) + sliding_count * len` (the product converted with `u64::try_from`, `None` on failure), where `full_count` is the number of true values and `sliding_count` the number of false values in `full_layers`. Reuse `BLOCK_TOKENS`, `SEAL_HORIZON_ROWS`, `LOCAL_BLOCKS` and `keep_block_count` (FT15.2).
  2. `read_arm.rs`: add `pub fn check_rows(line: String, measured: u64, expected: Option<u64>) -> Result<String, String>`: `None` returns `Ok(line)`; `Some(value)` equal to `measured` returns `Ok(format!("{line} expected_kv_rows_read={value}"))`; otherwise `Err(format!("{line} expected_kv_rows_read={value} mismatch=1"))`.
  3. `long_context_niah.rs`: `run_proxima_arm` clears the counter before generating (`let _ = model.take_kv_read_stats();`), reads `let (rows, steps) = model.take_kv_read_stats();` after, appends `kv_rows_read={rows} prompt_tokens={case.prompt_tokens} decode_tokens={steps}` to its line, and returns `Result<(String, u64, u64), NiahError>` (line, rows, steps). `run_arms` maps that result through `expected_kv_rows_read(arm, full_layers, options.keep_milli, case.prompt_tokens, steps)` and `check_rows(line, rows, expected).map_err(NiahError::Failed)`, then `report_arm` as today (a mismatch prints `error=<line> .. mismatch=1` and counts as a failure).
- test: add in `read_arm.rs` tests (cases through `#[proxima::test]`):
  - `niah_expected_rows_match_the_worked_instance` with cases: `block_at_100_milli` (36 true layers, prompt 4096, 2 steps, keep 100) is `Some(83052)`; `block_at_900_milli` is `Some(267372)`; `dense_counts_nothing` is `Some(0)`; `replay_is_not_asserted` is `None`;
  - `niah_expected_rows_charge_sliding_layers_every_unflagged_row` with cases `e2b` (35 layers, `(layer + 1) % 5 == 0` full, keep 100) is `Some(245609)` and `moe_26b` (30 layers, `(layer + 1) % 6 == 0` full, keep 100) is `Some(216410)`, both prompt 4096 and 2 steps;
  - `niah_expected_rows_are_unknown_without_a_layer_pattern`: `Block` with an empty `full_layers` is `None`;
  - `niah_row_check_appends_the_expected_value_when_equal`: `check_rows("found=1/1".into(), 1153, Some(1153))` is `Ok("found=1/1 expected_kv_rows_read=1153")`;
  - `niah_row_check_marks_a_mismatch`: `check_rows("found=1/1".into(), 1154, Some(1153))` is `Err("found=1/1 expected_kv_rows_read=1153 mismatch=1")`;
  - `niah_row_check_leaves_an_unasserted_arm_alone`: `check_rows("found=1/1".into(), 99, None)` is `Ok("found=1/1")`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_15_3 cargo nextest run -p proxima-model-interop --example long_context_niah --features std,metal -E 'test(/niah_expected_rows_|niah_row_check_/)'`; then, after the quiet-box check (`ps -axo comm | grep -E "decode_gbps|census|llama-server|decode_arms|norm_variant"` prints nothing), the one run:
  `mkdir -p /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/cards15 && GEMMA4_E2B=$(ollama show --modelfile gemma4:e2b-it-qat | sed -n 's/^FROM //p' | head -1) && CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_15_3 cargo run -p proxima-model-interop --release --example long_context_niah --features std,metal -- --model "$GEMMA4_E2B" --ctx 8192 --needles 4 --read dense,block,block-replay > /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/cards15/e2b-read-arms.log 2>&1`,
  then `grep -c '^arm=' <log>`, then `grep -c '^arm=.*found=.*kv_rows_read=.*prompt_tokens=.*decode_tokens=' <log>`, then `grep -c 'mismatch=1' <log>`, then `grep '^arm=block ' <log>`.
- expect: `10 passed` (the four worked-instance cases, the two layer-pattern cases, the empty-pattern test and the three row-check tests); the run's counts print `3`, `3` and `0`; the `arm=block ` line has `kv_rows_read=` equal to its own `expected_kv_rows_read=` value; the `arm=dense ` line has `kv_rows_read=0` and `expected_kv_rows_read=0`. Record the printed `prompt_tokens=`, `decode_tokens=` and every arm's `found=` in the log directory, not in the repo. A `rejected=` or `mismatch=1` line is reported with the printed numbers and the card stops; the formula is not edited to fit.
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/examples/long_context_niah.rs`, `proxima-model-interop/examples/long_context_niah/read_arm.rs`
- commit: `feat(interop): report key rows read per arm in the niah harness`
- done when: the expect lines printed, clippy clean, `git diff --cached --stat` equals the stage list, the target dir removed, and the commit landed with that message
- do not: edit `src/`; print a timing; run the ollama arm concurrently with a proxima arm (the harness already runs them in turn); assert the `block-replay` arm's rows; use a qwen checkpoint.
- gpu: one run (the 8192-token run, one model, E2B), waiting for a quiet box (CARDS.md machine safety); the model-loading run is `-j 1` by construction (one process).

## spec drift

Each item: where an earlier statement says one thing and the cards do another, with the reason.

1. The earlier cut interleaved no arms and neither do these: the harness runs one case once, arms run in the order given on `--read`, and no timing is asserted, so no clock artifact applies. If a repeat flag is added, the arms loop is where to interleave.
2. The earlier acceptance line says each `arm=` line carries `prompt_tokens=`. The case line prints it once today; these cards repeat it on each arm line and leave the case line unchanged.
3. The earlier formula used the number of own-cache layers the harness holds (`layers.len()`), which is wrong for gemma4: shared-cache layers also read and the counter sums over them. The formula takes a per-layer full-attention list read from the header instead.
4. The earlier cut put the keep ratio in the serving config's read field. The read stage is now a function on the model plus the operand switch, so the ratio is a harness flag read by the example's own rule and formula.
5. The dense arm now prints `kv_rows_read=0` and `decode_tokens=0` because the counter is off under `ReadSpec::Dense` by design; a dense row total is not measured by this harness.

## slice exit

- Before the first card, record N: the count printed by `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_15_exit cargo nextest run -p proxima-model-interop --example long_context_niah --features std,metal` on main. After the three cards the same command prints `N+20 passed` (3 + 7 + 10 new tests), every earlier `niah_` test among them.
- The second model run, after a quiet-box check and with the E2B process finished: the same run command as card 15.3 with `GEMMA4_26B=$(ollama show --modelfile batiai/gemma4-26b:latest | sed -n 's/^FROM //p' | head -1)`, `--model "$GEMMA4_26B"` and the log `26b-read-arms.log`. Expect the three counts `3`, `3`, `0`, and an `arm=block ` line whose `kv_rows_read=` equals its `expected_kv_rows_read=` (30 layers, 5 full). Record the lines in the log directory. This run commits nothing.
- Not cuttable: none (each card is under the file-count and line limits; 15.2 adds about 40 non-test lines, 15.3 about 40).
- Decide-later items with owners: a formula for the dense rows of the replay passes (the `block-replay` arm's rows are printed, not asserted) belongs to the first measurement that needs it; a score-driven block read through the live model needs a read rule that can see keys, which is a change to the read stage's function signature owned by the read-stage slice, not to this harness.
