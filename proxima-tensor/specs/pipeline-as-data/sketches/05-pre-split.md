# sketch 5: a tokenizer that needs a different pre-split, via `tokenizer.ggml.pre` (H3 tokenize)

status: paper test. Read at main 4b4be6cf (`git show main:<path>`) plus the llama.cpp checkout at commit
f1ea20621 (`/Users/brianbruggeman/repos/others/llama.cpp`, read directly; it is the oracle, never a
dependency). Nothing was built or run; every "hand-derived" value is derived from cited lines, not
executed. Paths relative to the proxima repo root; `tok` = `proxima-tokenizer/src`;
`llama-vocab.cpp` = `src/llama-vocab.cpp` in the llama.cpp checkout.

## shape chosen, and the contested decision

Shape: H3's splitter is chosen by a small record, `[tokenizer.pre]`, selected from the GGUF key
`tokenizer.ggml.pre` through a name table. The record has the knobs the incumbent's pre types differ by,
and nothing else: `kind` (`word` or `newline_runs`), `digit_run_max`, `letter_marks`, `ignore_merges`.

Contested decision: should the pre-split be a regex string in config (the incumbent's shape,
`llama-vocab.cpp:283-460`), or a record of knobs over the existing hand-rolled scanner? Chosen: knobs. The
tokenizer core is `no_std + alloc` and ships no regex engine (`tok/pretokenize.rs:17-21`), so a regex
string would need an interpreter, which is a dependency decision; the three byte-level pre types that
matter here differ from each other in exactly two places (digit run length; whether `\p{M}` counts as a
letter), read from the regex text in section 1. Where a future pre type differs in a third place, that is a
new knob or a new scanner function in core, the same closed-set answer the sampler and drafter sketches
reach. Not claimed: that two knobs cover every pre type; only these three were compared.

Second gate, call site both ways, for a `Split: Pipe<In = &str, Out = Vec<Range<usize>>>` type: before,
`pretokenize(text)` (`tok/pipe.rs:76`); after, `split.call(text).await`. Same operand, same result, so it
is a relocation and is not minted.

## 1. ground: what happens today, and what the incumbent does

Today the splitter is chosen by the SHAPE of the vocab, never by the key. `encode_ordinary`
(`tok/pipe.rs:67-81`): a unigram vocab goes to SentencePiece (:68-71); a char-level BPE vocab goes to the
newline-run splitter `pretokenize_newline_runs` (:72-74, :85-92); everything else goes to `pretokenize`,
the LLAMA3 scanner (:75-79). `is_char_level_bpe` is "merges present, and both `▁` and `<0x0A>` are
tokens", derived once at construction (`tok/vocab.rs:474-482`). The GGUF reader never reads
`tokenizer.ggml.pre`: its key constants are `model, tokens, merges, scores, token_type, bos, eos, unknown,
add_bos, add_eos` (`tok/gguf.rs:40-49`), and `pre` appears only in the module doc (:17). A git grep for
`ggml.pre` over `tok/` and `interop/src` finds that one doc line.

The LLAMA3 scanner has two constants that differ from the other pre types:
- digits: `match_digits` takes a run of digits and returns `run.min(3)` (`tok/pretokenize.rs:167-178`),
  the regex `\p{N}{1,3}`;
- letters: `is_letter` is `char::is_alphabetic` (:31-33).

The incumbent, same commit, byte-level BPE regexes:
- LLAMA3 (`llama-vocab.cpp:290`): `...|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|...`
- QWEN2 (:389): identical except `\p{N}` (one digit per piece).
- QWEN35 (:396): `[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|...`: one digit per
  piece, and `\p{M}` joins both the letter run and the punctuation exclusion.
The key to type mapping: `llama3`, `llama-v3`, `llama-bpe`, `falcon3`, `falcon-h1`, `pixtral`, `midm-2.0`,
`lfm2`, `jina-v5-nano` give LLAMA3, with `ignore_merges = true` and `add_bos = true` (:2174-2185);
`qwen2`, `deepseek-r1-qwen`, `kormo`, `f2llmv2` give QWEN2 with `clean_spaces = false` (:2259-2264);
`qwen35` gives QWEN35 (:2266-2268); a missing key warns and uses `default` (:2160-2167); an unknown name
THROWS (:2423).

The checkpoints this repo holds (`proxima-model-interop/tests/fixtures/llama-parity/checkpoints.toml`),
from their fixture kv: qwen3 8B `pre = 'qwen2'` (kv :21), qwen2.5 0.5B `pre = 'qwen2'` (kv :28), qwen3.5
0.8B `pre = 'qwen35'` (kv :52), model `gpt2` in all three. gemma4 E2B and openchat carry no `pre` key at
all (their kv dumps list none), and take the other two splitters through shape.

So: qwen2, qwen3 and qwen3.5 all take the LLAMA3 scanner today (`pipe.rs:76`), and the incumbent gives
them single-digit splits.

Evidence about how much that is exercised: the three prompts in each qwen fixture
(`.../qwen3/llama_ids.json`, `.../qwen2/llama_ids.json`: `"The capital of France is"`,
`"def fibonacci(n):\n"`, and a paragraph of prose) contain no digit characters. The "token parity with
llama.cpp proven for qwen2, qwen3" row in the SPEC (`llama_parity_`) therefore says nothing about digit
splitting. Status: that the prompts are digit-free is read from the fixtures; whether any other test
covers digits for qwen is not claimed (the tokenizer-level oracle,
`proxima-tokenizer/tests/gemma4_llama_oracle.rs`, is gemma4 only).

## 2. the config a user would write

```toml
[tokenizer.pre]
source = "gguf"
unknown = "error"
missing = "shape"

[tokenizer.pre.rule.llama3]
names = ["llama3", "llama-v3", "llama-bpe", "falcon3", "falcon-h1", "pixtral", "midm-2.0", "lfm2", "jina-v5-nano"]
kind = "word"
digit_run_max = 3
letter_marks = false
ignore_merges = true

[tokenizer.pre.rule.qwen2]
names = ["qwen2", "deepseek-r1-qwen", "kormo", "f2llmv2"]
kind = "word"
digit_run_max = 1
letter_marks = false
ignore_merges = false

[tokenizer.pre.rule.qwen35]
names = ["qwen35"]
kind = "word"
digit_run_max = 1
letter_marks = true
ignore_merges = false

[tokenizer.pre.rule.gemma4]
names = ["gemma4"]
kind = "newline_runs"
```

Defaults equal today: `missing = "shape"` means a checkpoint with no `pre` key keeps today's shape probe
(unigram, char-level, else LLAMA3); and `source = "none"` reproduces today for every checkpoint including
the qwen ones. `source = "gguf"` changes qwen2, qwen3 and qwen3.5 output on digit runs. That is not a
default-preserving change, and the invariant ("default reproduces today") is honoured only by making
`source = "none"` the default until the oracle fixtures in section 5 pass; then it flips, with the evidence
(principle 14: the incumbent wins on correctness).

## 3. field map: every key to a consumer, or a gap

| key | consumer | status |
|---|---|---|
| `pre.source`, `pre.missing`, `pre.unknown` | none: `tokenizer.ggml.pre` is never read (`tok/gguf.rs:40-49`) | GAP-1 |
| `rule.*.names` | none | GAP-1 |
| `rule.*.kind` | shape probe: `is_char_level_bpe` (`tok/vocab.rs:480`), `is_unigram` | EXISTS by another route |
| `rule.*.digit_run_max` | the literal `3` in `match_digits` (`tok/pretokenize.rs:177`) | GAP-2 |
| `rule.*.letter_marks` | none: `is_letter` is `char::is_alphabetic` (:31-33); no `\p{M}` predicate in the crate (git grep `is_mark` over `tok/` finds nothing) | GAP-2, GAP-3 |
| `rule.*.ignore_merges` | none: `encode_pretoken` seeds bytes and merges, never looks up the whole pretoken first (`tok/bpe.rs:23-49`); git grep `ignore_merges` over `tok/` finds nothing | GAP-5 |

## 4. can the decision be a pure function in core plus a pipe?

Decision half: yes and it already is one, with two hard-coded constants. `pretokenize(text) ->
Vec<Range<usize>>` and `pretokenize_newline_runs(text)` are pure, allocation-bounded functions in a
`no_std + alloc` crate (`tok/pretokenize.rs:63`, :85). The catalog column "decision in proxima-core" is
again the tokenizer crate (sketches 6, 7).

Placement half: there is none beyond choosing the rule once at vocab construction and storing it on the
`Vocab`, the same shape as `with_token_types` and `with_bos_eos_policy` (`tok/vocab.rs:367-405`). No pipe is
needed. The change to core is the scanner taking the record:

```rust
pub struct PreRule {
    pub digit_run_max: usize,
    pub letter_marks: bool,
}

fn match_digits(chars: &[char], rule: PreRule) -> Option<usize> {
    let first = *chars.first()?;
    if !first.is_numeric() {
        return None;
    }
    let run = chars.iter().take_while(|character| character.is_numeric()).count();
    Some(run.min(rule.digit_run_max))
}
```

`PreRule` is the serde config shape the TOML needs, not a behaviour wrapper. `pretokenize(text)` keeps its
signature and calls the new function with `PreRule { digit_run_max: 3, letter_marks: false }`, so every
caller and test sees the same bytes.

## 5. worked example (doubles as the test)

Real strings from the crate's own tests (`tok/pretokenize.rs:268-270, 299-300`):

Example A, `"3333"`. LLAMA3 rule: `["333", "3"]` (the existing test, `digit_runs_cap_at_three`). QWEN2 and
QWEN35 rule (`digit_run_max = 1`): `["3", "3", "3", "3"]`.

Example B, `"In 2024, 3333 items."`, hand-derived with the scanner (`match_at` order: contraction, letters,
digits, punctuation, whitespace-with-newline, trailing whitespace, `pretokenize.rs:118-140`):
- LLAMA3: `In`, ` `, `202`, `4`, `,`, ` `, `333`, `3`, ` items`, `.` (10 spans). The space before `2` is a
  lone span because the letter rule needs a letter after it (:152-157) and the punctuation rule needs a
  punctuation char (:185); it falls through to `match_trailing_whitespace` (:228-244), which returns one
  char.
- QWEN2 (`digit_run_max = 1`): `In`, ` `, `2`, `0`, `2`, `4`, `,`, ` `, `3`, `3`, `3`, `3`, ` items`, `.`
  (14 spans). Only the digit runs differ.
Each span then goes through BPE independently (`pipe.rs:76-79`), so the id sequences differ wherever the
vocab has a multi-digit token. Which ids result needs the qwen vocab; not derived here.

Example C, `"apple1314151"` (also from the repo's tests, :300). LLAMA3: `apple`, `131`, `415`, `1`; QWEN2:
`apple`, then seven single-digit spans.

Test shape: A, B, C as span-equality assertions per rule, in `pretokenize.rs`'s existing `tests` module,
plus the missing evidence: tokenizer-level oracle fixtures for qwen, generated exactly as the gemma4 ones
(`proxima-tokenizer/tests/fixtures/llama-gemma4-tokenize/README.md`: `llama-tokenize -m <blob> --ids
--log-disable --no-escape -f <name>.txt`), with inputs `digits.txt` (Examples B and C) against the qwen3
blob named in `checkpoints.toml`. That run is the proof the incumbent-wins default flip needs; it was not
done here (no model was loaded).

Config parity (P4): the TOML loads to a value equal to the builder's by the `PromptCacheSettings` triple
(`interop/prompt_cache_settings.rs:17-100`). Not built here.

## 6. HOOK GAPS (stage, missing input, smallest generic change)

GAP-1. H3 tokenize. Missing input: `tokenizer.ggml.pre` is never read, and the splitter is chosen by
vocab shape. Smallest change: read the key in `vocab_from_metadata` (one more constant beside
`tok/gguf.rs:40-49`), carry it on the `Vocab` as `Option<String>` (the `with_bos_eos_policy` shape,
`tok/vocab.rs:397-405`), and have `encode_ordinary` look the rule up by name. Both missing-key and
unknown-name policies are data (`missing`, `unknown` above): the incumbent warns on missing (:2160) and
throws on unknown (:2423); today both are silent.

GAP-2. H3. Missing input: the scanner's two constants. Smallest change: section 4's `PreRule`, threaded
through `match_digits` and `match_letters`/`match_punct`. Call site both ways: before `pretokenize(text)`;
after `pretokenize_with(text, rule)`; they differ (the rule is an input), so this is a parameter, not a
relocation.

GAP-3. H3. Missing input: a `\p{M}` (Unicode mark) predicate for the `qwen35` rule. `core` offers
`char::is_alphabetic` and `char::is_numeric` and no general-category query (guess: not checked against the
toolchain docs; the crate has no such helper). Smallest change: a small generated range table for
Mn/Mc/Me in the tokenizer crate (build-time data, principle 12), used only when `letter_marks` is true.
Until then the `qwen35` rule is exact for text with no combining marks and unproven for text with them.
Status: plausible.

GAP-4. H3 / H1 seam. Missing input: the failure policies. A GGUF whose `pre` the table does not know
silently takes the shape default today. Covered by `unknown` in GAP-1; listed separately because it is the
one gap that changes behaviour for a checkpoint nobody tested.

GAP-5. H3. Missing input: `ignore_merges`. The LLAMA3 pre type sets it (`llama-vocab.cpp:2184`): a
pretoken that is itself a vocab token is emitted whole without running merges. `encode_pretoken` has no
such lookup (`tok/bpe.rs:23-49`). Whether outputs differ on real vocabs was not measured; the crate's own
ignored oracle test says exact id parity is "a bonus this test reports, not a correctness gate"
(`tok/tests.rs:370-376`). Smallest change: the `ignore_merges` knob and one `token_id(bytes)` lookup before
the merge loop. Status: unmeasured.

## 7. what is not a pipe, and why

- The scanner: a pure function over `&str`, run inside `encode`; there is nothing to compose with.
- The name table: a lookup at vocab construction, once per model.
- `PreRule`: plain data.

## 8. designs abandoned

- A regex string in config: needs a regex engine in a `no_std` crate (`pretokenize.rs:17-21`).
- A `trait Splitter` with one impl per pre type: the types differ by two constants and one predicate; a
  trait would be a blanket-impl-under-a-new-name.
- Selecting by checkpoint family name: the incumbent keys on the `pre` string, which is data in the GGUF;
  a family check would be an arbitrary per-instance rule.
- Flipping the default to `source = "gguf"` now: no qwen oracle fixture exists to prove it, so the
  default-preserving setting stays until one does.

## defects found in passing

- Two tokenizer tests carry bare `#[ignore]` and one of them is documented as "a bonus ... not a
  correctness gate" (`tok/tests.rs:332, 370-379`): the incumbent-wins rule (principle 14) forbids both.
- `tok/gguf.rs:17` documents `tokenizer.ggml.pre` as selecting the pretokenizer, while no code reads it.
- `tok/lib.rs:27-30` says the LLAMA3 pretokenizer is "the variant `tokenizer.ggml.pre = llama-bpe`
  identify"; qwen vocabs take the same path with a different `pre`.
- `wants_bos` falls back to "the vocab has a BOS id" (`interop/generate/residency_caches.rs:3595`), while
  the incumbent's LLAMA3 pre sets `add_bos = true` as a default (`llama-vocab.cpp:2185`): a second
  per-pre default with no home in the record. Not verified to differ on any checkpoint here.

## verdict

fits after 5 named hook changes: GAP-1 (read the key, carry it on the vocab, table lookup), GAP-2
(`PreRule` through the scanner), GAP-3 (a `\p{M}` table for `qwen35`), GAP-4 (missing/unknown policies),
GAP-5 (`ignore_merges`). The qwen2 and qwen3 case alone needs GAP-1 and GAP-2. Not claimed: that the
single-digit split changes any recorded token id (no digit-bearing qwen oracle fixture exists), or that it
preserves speed; none of it was run.
