---
name: disciplined-component
description: Bench-driven greenfield component development. Build a new performance-sensitive primitive (channel, allocator, datastructure, runtime piece) behind a default-off feature flag, with explicit comparison baselines, mandatory micro-benches per component, and a versioned discipline log where every tweak's delta is recorded — including rollbacks. Use when introducing a NEW primitive intended to replace or compete with an existing one (flume, tokio mpsc, dashmap, std HashMap, rayon, etc.), when the user says "do it disciplined", "we need to be super disciplined", "spin out a component", "bench it against X", or when the work spans 5+ components that each need to land independently. OUT OF SCOPE: drop-in dependency upgrades, refactors that don't change behavior, single-file features, code where comparison baselines don't exist or aren't meaningful.
---

# disciplined-component

When you're rolling your own performance-critical primitive, two failure modes dominate: shipping the whole thing then realizing pieces are bad, or claiming a win on vibes without numbers. This skill structures the work so each component lands fully validated against named baselines, every optimization is benched, and negative results are recorded — not buried.

The discipline is **mechanical**: a checklist gate per component, a log file that grows row-by-row, a script that runs the gate. Vibes don't pass; numbers do.

## When to use

- Implementing a new channel, queue, allocator, executor, hashmap, etc. that claims to beat or match an existing one
- A multi-component build where each piece (5+) needs to land independently before the next starts
- The user has said: "do it disciplined", "every new tweak rerun bench", "build it then bench it", "compare against flume/tokio/X", "spin out a component"
- Any greenfield primitive where "looks fast" isn't an answer — only "12.6 M/s vs flume's 22.0 M/s" is

## Project memory bootstrap

Before broad code exploration, check whether the project has an
agent-facing docs index. For Proxima work, read:

1. `proxima/ai_docs/AGENT.md`
2. `proxima/ai_docs/index.jsonl`
3. `proxima/ai_docs/task-routes.jsonl`
4. matching records from `proxima/ai_docs/invariants.jsonl`

If the requested component has no matching route or invariant record,
add the missing JSONL record before proceeding. The JSONL is the
machine-readable source of truth; markdown projections are only readable
views.

## When NOT to use

- Single-file features, refactors, typo fixes
- Dependency bumps and version migrations
- Code without meaningful comparison baselines (e.g., a config parser — there's no "fast" or "slow" comparable alternative worth structuring around)
- Behavior-only work where benchmarks aren't the success criterion
- The objective itself is uncertain or there's no known incumbent to beat — you're discovering whether a new capability *works*, not optimizing a known one against a named rival (a new architecture, a sparse-network LLM, an empirical heuristic) — use `/discovery-loop`. It is this skill's discovery-mode sibling: same substrate (default-off flag, discipline log, recorded negatives, CoV/variance, delegate-to-sonnet), but it adds pre-registration + held-out + ablation + scaling-ladder because the objective and baseline aren't given. A discovered win hands back here for the 13-point gate once it's a perf primitive vs a named incumbent.

## Architectural rules of thumb

The architectural rules-of-thumb that bind before the first line of a
component (RISC reuse, hot-path representation, allocation budget,
hot/cold separation, config + API as first-class, compile-time feature
gate, representation-shortcut regressions) live in `/guiding-principles`
under "Hot-path representation defaults" and principles 1, 3, 4, 11, 12.

Pull that skill before opening a discipline log row. Every row's
"Notes" column should state which guiding principles the component
engages and how. Updates to those rules belong in `/guiding-principles`,
not duplicated here.

## The 13-point gate (per component)

Every component must satisfy:

1. **Spin out** as an independently-buildable component with its own Cargo feature flag (compile-time, default-off). The flag is the firewall — until the e2e bench shows the full stack wins, downstream users opt in explicitly.
2. **Build clean** under that sub-flag (default-features off, --features sub-flag).
3. **Tests pass** — at minimum: happy path, sad path, edge cases, drop semantics, concurrency where applicable.
4. **Clippy pedantic clean** (or whatever the project's lint baseline is) — no `#[allow]` without justification.
5. **Micro-bench file exists** — one bench file per component.
6. **Compare-bench numbers** — bench against the *explicit alternatives* the component replaces (flume, tokio mpsc, std mpsc, etc.). Named, not "industry standard".
7. **E2E bench** — when downstream components compose this one, the e2e bench gains an arm with this swap.
8. **Optimization sweep recorded** — stack-over-heap, bytes-over-String, borrow-over-own, zero-copy-over-copy-over-clone, lock-free-over-mutex, slab-over-HashMap, fixed-cap-ring-over-VecDeque, hot/cold split, and explicit allocation budget. Before hand-rolling, sweep the reach-for crates:
   - `memchr` / `memchr::memmem` / `memchr2` / `memchr3` over `regex` for literal byte/substring patterns, line splits, delimiter scans, small alternations (SIMD-accelerated)
   - `aho-corasick` over hand-rolled multi-pattern scans when alternation count outgrows `memchr3`
   - `parking_lot::Mutex` / `RwLock` over `std::sync` (smaller footprint, no poisoning overhead, faster uncontended path)
   - `crossbeam_utils::CachePadded<T>` around atomics shared across cores to kill false sharing
   - `arc-swap` over `RwLock<Arc<T>>` for read-mostly hot-swappable config/state
   - `mmap` (via `memmap2`) over buffered `read` for cold-large read-only files where the OS page cache wins
9. **Re-bench every tweak** — every change gets a row in the discipline log with the delta vs prior. Including rollbacks. Especially rollbacks.
10. **SIMD / state-machine / no-Box / discriminated-enum pass** — could SIMD apply? Could `enum + match` replace `Box<dyn Trait>`? Is `Box::pin` actually needed or is a state-machine future cleaner?
11. **Strict O(1)** for steady-state operations — fixed-cap rings or slabs, not `VecDeque` (amortized + resize). No `HashMap` outside concurrent-shared cases (use slab + index).
12. **Config + API parity.** Component exposes both a `conflaguration`-derived config surface and a fluent Rust API; a test fixture constructs the component both ways and asserts equivalent state. Sensitive fields flagged `#[setting(sensitive)]`. **For composition-style components** (those whose purpose is to add variants *without* recompiling — integrations, notifiers, routers, pipelines), this also requires a **config-only-variant proof**: a fixture that registers a novel instance purely via config and exercises it against the compiled primitives, with zero new Rust. A per-variant Rust `impl` is a recompile and does not satisfy this — see `/guiding-principles` §4 "config-as-composition" (`integrations_are_config` / `three_axis_shape`).
13. **Home-turf bench arm = the 80% case.** For every named incumbent in point 6, include at least one bench arm that engages the **incumbent's design point AND the realistic 80% workload** — the operation called per-packet/per-request/per-event in steady-state successful traffic, run in a shape the incumbent's machinery was built for. These are the SAME concept: the incumbent's design point IS their 80% case; if they're built for line-rate AEAD, then per-packet AEAD at QUIC datagram sizes IS the home-turf arm. Isolated cold-path or primitive-op arms (loss events, error rejects, drop-on-overflow) test neither — "we win" on a <5% path while losing on the 100% path is a net loss masquerading as a win. The headline claim must be backed by a result on **their turf at realistic frequency**. If the incumbent's public surface hides their 80% case, bench the underlying library both wrappers share (e.g. `aws-lc-rs::aead` when both you and the incumbent wrap it), or mark the row "no comparable micro surface; e2e bench REQUIRED before claiming a verdict." Every compare-bench row's honest read MUST include a frequency-weighted scorecard (cost × call-frequency), not just per-arm ratios. See "Home-turf = 80% case" below for the full discipline.

## Sans-IO components — additional discipline

A component is **sans-IO** when its public API takes bytes in (`&[u8]`,
`Bytes`, `Cursor<&[u8]>`) and returns parsed values or bytes out, with
**no syscalls, no awaits, no allocations on the hot path, and no I/O
traits** in the signature. Wire-format parsers and codecs (HTTP framing,
MQTT packets, DNS messages, protobuf varints, Kafka headers, WebSocket
frames) are the canonical examples.

When a component is sans-IO, gate points 8 (Opt sweep) and 10 (SIMD /
state-machine / no-Box / discriminated-enum) get sharper, and the
opt-sweep findings table in the discipline log must explicitly address
each sans-IO axis — pass or punt — with rationale on every punt.

The full sans-IO discipline — six clauses, mandatory axes, opt-sweep
table shape, and bench design-point shifts — lives in
`/guiding-principles` under principle 11 ("Sans-IO: enum-shaped state
machine, low/no alloc, extreme benching, extreme performance"). A
sans-IO row cannot be sealed without each of those clauses marked DONE
with evidence.

## Delegate to subagents (sonnet)

This skill is a coordination layer. The grind — writing the bench harness, writing the test cases, exploring how an existing crate solves problem X — is delegated. The main thread stays on the gate, the log, and the judgment calls (keep vs roll back, what to try next).

Default model for delegated work is **sonnet** unless the task explicitly requires opus-level reasoning. Pass `model: "sonnet"` when launching the Agent.

Who does what:

- **Explore** (sonnet) — "where does flume implement its bounded ring buffer", "find all uses of CachePadded in the workspace", "what does crossbeam's ArrayQueue do on contention". Use for read-only code archaeology before you write or bench. Specify breadth: "quick" / "medium" / "very thorough".
- **test-writer** (sonnet) — writing the ≥6 cases (happy / sad / edge / drop / concurrency / property). Hand it the public surface + the language rules file. It runs the tests to confirm green via `cargo nextest run` (the project default; falls back to `cargo test` only for doctests).
- **perf-reviewer** (sonnet) — after a bench row goes red or mixed, hand it the file and ask for an in-place fix. Knows what to bench and what to leave alone.
- **code-reviewer** (sonnet) — independent read on the component before you mark it done. Won't see your reasoning, so the report is honest.
- **compliance-reviewer** (sonnet) — verifies the component against the repo AGENTS.md and language rules (no unwrap outside tests, no inline imports, edition 2024, etc.) before sealing the row.
- **debugger** (sonnet) — when a test or bench fails in a non-obvious way (UB, ordering bug, lost wakeup). Hand it the failure + the file.
- **doc-writer** (sonnet) — the discipline.md changelog rows themselves stay on the main thread (they're judgment), but if a component needs a real README or public docstring pass, delegate it.

Launch independent subagents **in parallel** — one message with multiple Agent tool uses. E.g. test-writer + bench scaffolding via Explore in the same turn. The main thread does not block on either.

Do NOT delegate:
- The gate verdict (which row to fill, what passes, what gets rolled back)
- The discipline.md log itself (the log is the contract)
- The decision to try optimization Y vs Z next

### Worktree isolation for parallel code-landing work

Subagents that *land code* (test-writer, perf-reviewer doing in-place fixes, doc-writer, anything with Edit/Write) run with `isolation: "worktree"` so concurrent component work does not collide on the same files. Read-only subagents (Explore, code-reviewer, compliance-reviewer, debugger) do not need worktree isolation — they observe, they don't write.

The main thread fans out: single message, multiple Agent tool uses, default model sonnet, code-landers in worktrees. After each returns, the main thread inspects the actual diff (trust but verify — the agent's summary describes intent, not necessarily what landed) before merging into the integration branch.

## Benchmarking is taxed — plan for it

The host you're running on may have other benches, builds, or background work competing for CPU. **High coefficient of variation (CoV) is the normal failure mode**, not a code problem. Criterion will print warnings like `Found N outliers among M measurements`; treat them as a signal to re-run or accept the noise, not to chase a phantom optimization.

Practical disciplines:

- **Don't trust a single criterion run when CoV > ~5%.** Re-run the bench in isolation if possible; if not, accept that a "tweak with +3% delta" is below the noise floor and the row should say "no signal, kept simpler form".
- **Do an in-the-moment criterion sweep.** When a tweak's delta is close to noise, run `cargo bench --bench <b> -- <pattern>` 3-5 times and record the *range*, not a single number. A range of `12.1-13.4 M/s` is honest; `12.6 M/s` alone is a lie when CoV is 7%.
- **Use criterion's `--save-baseline` and `--baseline` flags** to compare against a named prior run rather than eyeballing two log files. `cargo bench --bench bench_x -- --save-baseline pre-tweak`, then after the change, `--baseline pre-tweak`.
- **Sweep across at least two input sizes.** A tweak that wins at N=1k but ties at N=1M is a different story than one that wins at both. The changelog row should say which.
- **Pin the bench host loadout in the log.** "Ran with 2 other criterion benches active" is a valid notes-column entry. Future-you needs to know if the number came from a quiet box or a loaded one.
- **If results stay noisy, isolate.** Disable other background benches, kill heavy IDEs, run on AC power (laptop throttling is real). Document what you did to quiet the box in the changelog row.

Bench-running delegation: a sonnet Agent can drive the `cargo bench` runs and report back the parsed throughput + outlier count, so the main thread doesn't burn tokens on criterion's verbose output. Ask it to return only the `thrpt:` lines, CoV %, and outlier counts.

## Home-turf bench arms — the 80% case the incumbent was built for

The default failure mode of a compare-bench is choosing a workload where the incumbent's design choices are **not** engaged AND that isn't the realistic 80% case. Those are the same failure: the incumbent's design point IS the 80% case for the kind of system the incumbent was built for. Comparing an MPMC channel to your SPSC ring on single-producer single-consumer at small capacity is comparing the incumbent in their worst scenario AND on a workload that isn't the per-message hot path; "we're 11× faster" proves only that you wrote an SPSC ring.

Every named incumbent in point 6 has a design point — the specific workload shape their authors optimized for, which is also the operation called per-packet/per-request/per-event in realistic steady-state traffic. Find it. Build a bench arm that engages it. **If you can meet or beat them on their own turf at realistic frequency, that's a real result.** If you can't, that's also a real result — document the regime as "defers to incumbent on hot path; outside our design point."

### Two failure modes to avoid

1. **Win-on-niche, lose-on-hot.** Picking a cold-path operation (loss event in a CC controller, error-path reject in a parser, drop-on-overflow in a queue) and claiming a headline win there. The path fires at <5% rate; the per-operation cost gets multiplied by frequency, and the headline is meaningless or net-negative in steady state. **Historical example:** claiming a 12× win on `loss_event` (~1% of packets) while losing 1.78× on `on_packet_sent` (100% of packets) — net loss in normal traffic, but the headline claimed a 12× win.

2. **Skipping the 80% case because the incumbent's surface is awkward.** When the incumbent's API requires heavy setup (a full `Connection` to construct a private `RttEstimator`, a full `Endpoint` to construct a `PacketKey`), the temptation is to "defer that compare to an e2e bench" and headline the cheaper-to-bench cold-path operations. That's not honest. The right move is one of: (a) go through the heavy setup once and amortize, (b) bench the underlying library the incumbent wraps (e.g. `aws-lc-rs::aead` for an AEAD compare since both we and quinn wrap it), or (c) explicitly document the regime as "no comparable micro surface; e2e bench is the ONLY honest compare."

### Frequency bands — write these down BEFORE arms exist

Before adding the bench arm, in the discipline log opt-sweep section, classify each operation's call frequency in realistic production traffic:

- **80% case (hot path):** called per-packet / per-request / per-event in steady-state successful traffic. Examples: AEAD encrypt/decrypt, header-protect mask, varint parse, frame parse, `on_packet_sent`, `on_packet_acked`, `send_budget` query, hash function on insert. **Must have an incumbent compare arm.** Losses here are real losses regardless of how clean the micro looks.
- **Warm path (~5-20%):** called per-burst / per-window / per-batch. Examples: ACK frame emission, MAX_DATA grant, congestion-window growth, cache eviction. Should have a compare arm if cheap; documented if not.
- **Cold path (<5%):** per-rare-event / per-error / per-state-transition. Examples: loss event, connection close, key update, retry, error recovery. Compare arms here are nice-to-have, NOT headlines. A win here without a hot-path win is hollow.

### Frequency-weighted scorecard required

Every compare-bench row's "Honest read" section MUST include a frequency-weighted scorecard, not just per-arm absolute ratios:

```
| Frequency | Arm           | Verdict       | Absolute cost  |
|-----------|---------------|---------------|----------------|
| 100%      | on_packet_sent| LOSE 1.78x    | ~0.3 ns/call   |
| 100%      | send_budget   | TIED          | ~330 ps        |
| ~1%       | loss_event    | WIN 12x       | 33 ns saved/loss |
```

Then a one-sentence "headline reality" line: weighting cost × frequency, is this a net win, net loss, or tied in realistic traffic? If you can't answer that from the scorecard, the bench arms are wrong — the 80% case is missing.

### Honest framing of the loss — diagnose before declaring "we lost"

When the 80% case shows a loss, the framing matters as much as the number. Distinguish three causes before writing the headline:

1. **Backend-wiring drift** — plan said use library X (e.g. aws-lc-rs); code shipped library Y (e.g. RustCrypto). The "loss" is your wrapper invoking the wrong primitive. Honest framing: "config-truth bug — plan intent and shipped code diverged, bench caught it. Fix is a backend swap behind a feature flag, not an architecture change." This is the gate working as designed, not a capability gap.
2. **Underlying-primitive gap** — both you and the incumbent wrap the same library, the wrappers are equivalent, and the library itself is what's slow. Honest framing: "we and incumbent both inherit the library's limit; differentiating requires either contributing upstream or swapping the underlying primitive."
3. **Capability gap** — your design genuinely can't do what theirs can. Honest framing: "design trade-off; we accept N× loss on hot path in regime X to gain Y in regime Z. Headline scoped to regime Z."

Don't mash these together. "We lose 40× on the 80% case" reads as #3 (capability) when the cause is #1 (config-truth) — which leads to scheduling architecture work for a one-line backend swap.

### Which "80% path"? — match the bench to the product, not the substrate

This is the trap that loses turns: **which operation counts as the 80% path inverts based on what you're building.** The same component, same finding, can be P0 for one product and near-zero for another. The bench is honest, the framing isn't, because "80% path" is read against the wrong product.

Two canonical examples of the inversion:

| Product shape | What's the 80% path | What's the cold path |
|---|---|---|
| **Data-plane serving** (general proxy, gateway, runtime — competing on raw throughput) | Per-packet local work: AEAD, header protect, frame parse, varint, queue ops | The metered remote call (it's outside your latency budget, often outside your product) |
| **Money-plane orchestrator** (LLM gateway: route, cache, trim, inject — sitting in front of someone else's metered API) | The metered remote call to the upstream model (network RTT + token-gen latency, tens to thousands of ms) | All local work, including your AEAD and frame parsing — invisible behind the remote call |

A 40× AEAD-throughput loss is P0 against a "line-rate general proxy" claim and **approximately a no-op** against a "money-plane orchestrator" claim, because the buyer's bill is dominated by tokens at network RTT — local crypto at 140 MiB/s vs 5 GiB/s costs them nothing measurable.

**The filter that resolves this in one question:** *does the buyer's bill move?* If the bench result, when applied to realistic production volume on the actual paying product, changes the customer's invoice — it's a real 80%-case loss and a real headline. If it doesn't move the bill, it's a real finding on a non-revenue path: tracked, scheduled to the future claim that the path gates, but **not the priority of the current revenue work**.

**Why this trap is hard to avoid:** category reflex. Years spent on a runtime / serving plane train the instinct that per-packet local work IS the 80% case, because for that product it always is. When the product later pivots to a money-plane orchestrator, the instinct hasn't pivoted — every fresh substrate finding still presents itself as urgent. Most of them are P0 against the old positioning and near-zero against revenue. The filter has to be applied deliberately each time, or the deep-systems gravity wins.

**Scheduling discipline that survives the trap:**

- Name the product claim the loss invalidates. ("Line-rate general proxy" / "100k req/sec gateway" / "embedded edge inference at 10 Gbps" / "self-hosted control plane".)
- Check whether that claim is on the **near-term revenue path** or a **future positioning** (potentially months out, sometimes never).
- Apply the bill-mover filter: does the realistic production volume on the actual paying product turn this loss into a customer-visible cost?
- If the claim isn't near-term and the bill doesn't move, the loss is **sealed P0 against the future claim, parked until that claim ships** — bounded, specced, the fix is a known finite cost the team spends deliberately when the perf claim actually needs to be true.

Deep-systems wins exert strong gravity: a clean substrate finding feels like the right thing to fix immediately, and the framing tools that work for serving-plane work ("we lose 40× on the 80% case") activate that gravity directly. **Apply the bill-mover filter before writing the headline**, not after.

### When the incumbent's 80% surface is structurally inaccessible

Two honest moves:

1. **Bench the underlying library** the incumbent wraps. If both your code and the incumbent wrap `aws-lc-rs::aead` for AES-GCM (you via RustCrypto, them via aws-lc-rs), then `aws-lc-rs::aead` direct IS the comparable surface — it's the real underlying primitive, the comparison is meaningful even though the wrapper surface differs. Document the wrap.

2. **Mark the row "no comparable micro surface; e2e compare REQUIRED"** and schedule the e2e bench as a separate component. Do NOT claim a verdict on the row until that lands. A row with no hot-path compare data is a hypothesis, not a result.

### Anti-headlines (don't write these)

- "We won 12× on loss_event!" — when loss_event is ~1% of traffic. The headline is the frequency-weighted scorecard, not the biggest ratio.
- "We tied on send_budget" — `send_budget` is a single field load; both sides being ~330 ps tells you nothing about hot-path performance.
- "Documented as deliberate sans-IO trade-off" — when used to spin a real loss as a feature. Honest: "we lose N× because of <reason>; cost is X ns/call at frequency Y — net impact Z in realistic traffic." Then either fix it or scope it.
- **"We lose N× on the 80% case"** — without first diagnosing whether the cause is backend-wiring drift (config-truth bug, one-line fix), upstream-primitive gap (inherited from a shared library), or genuine capability gap (architecture). The framing changes the scheduling read.
- **"The headline, wrong direction"** — when the loss is on a path the buyer isn't paying for. A real 80%-case loss for a claim you're not making is a prerequisite for that future claim, not a fire today. Name the claim it gates before scheduling.

### Identifying the incumbent's design point

In rough order of authority:

1. **Their README headline** — what do they *say* they're for? `flume` ("a blazingly fast multi-producer, multi-consumer channel") → MPMC under contention. `hdrhistogram` ("a port of the High Dynamic Range Histogram for measuring values that span many orders of magnitude") → multi-decade range with quantile queries.
2. **Their own bench suite** — look at `benches/` in their repo. The workloads they bench are the workloads they're proud of. Mirror one of those.
3. **The crate's data structure** — `tracing::Subscriber` has layers; testing one layer isn't testing the system. `prometheus::IntCounterVec` has per-label-set storage; testing a single label isn't testing the registry.
4. **The complaint mode** — if the incumbent has known weaknesses in your regime, ALSO add an arm that hits those weaknesses. But the *primary* incumbent arm must engage their strength.

### Per-arm labeling

Every bench arm in the discipline log carries a `design-favors:` label:

- **`design-favors: incumbent`** — incumbent's machinery fully engaged. Meet-or-beat here is the load-bearing claim.
- **`design-favors: proxima`** — our machinery fully engaged. A win here is expected and not noteworthy on its own.
- **`design-favors: neutral`** — primitive operation; both implementations roughly comparable. Useful as a noise floor, not as a verdict.

**A component row that has no `design-favors: incumbent` arm has not been benched yet.** "Compare-bench" stays pending until at least one incumbent-favored arm produces a number — win, loss, or honest "regime not supported."

### Examples

| component built | incumbent | bad bench (incumbent in worst case) | good bench (incumbent's design point) |
|---|---|---|---|
| SPSC ring | flume / crossbeam ArrayQueue | SPSC at 1k cap, single iteration | MPMC under contention; 8 producers, 4 consumers, 1M items, deep cap |
| traceparent SIMD hex | faster-hex | 16-byte fixed input (one trace_id) | 1KB bulk hex decode (their SIMD chunks amortize) |
| Tag type | opentelemetry::KeyValue | static-key construct in tight loop | dynamic-key construct from HashMap<String, Value> (we don't even support this; arm becomes "feature gap documented") |
| Counter | prometheus::IntCounterVec | single counter, single label-set | 1024 unique label-sets driving the same counter (their per-label-set storage IS the design) |
| Histogram | hdrhistogram | record 1.5 in a tight loop | record values 10⁻³ to 10⁹, then query p99 (their multi-decade range + quantile query IS the design) |
| Span builder | opentelemetry::Tracer | `Tracer::start` with no Sampler/Processor/Exporter | full pipeline: Sampler → BatchSpanProcessor → InMemorySpanExporter |
| Log emit | tracing::Event + Subscriber | single Event + Discard subscriber | EnvFilter + fmt layer + opentelemetry layer + per-event filter rejection |
| Recorder | tracing-subscriber::Registry | single fmt layer | 4-layer composition with EnvFilter + Fmt + OpenTelemetry + custom layer |

### Honest verdicts

Possible outcomes when running a home-turf arm:

- **Match or beat:** the strongest possible claim. Document the workload and the number; this is the headline win.
- **Lose by a measurable margin:** also a real result. Document it. "Loses to flume's MPMC by 2.3× at 8-producer contention; out of scope for v1 SPSC primitive." Future opt-sweep target or explicit out-of-scope.
- **Cannot run the arm:** the regime the incumbent excels at isn't supported by your design (e.g. proxima Tag's `&'static str` keys can't accept dynamic keys). Document as **feature gap** — not a perf loss; the comparison can't even be made. Frame as a deliberate trade-off in the row notes.

**What's not OK:** omitting the arm because it would be unfavorable. That's the proxima-team-retraction shape — declaring "structural" before the audit finishes.

## Artifacts you produce

For each project that uses this skill:

```
project/
  scripts/
    component-gate.sh         # runs steps 2-4 mechanically per component
  src/<initiative>/...        # the components, each behind sub-flag
  benches/
    bench_<component>.rs      # one per component
    <e2e>.rs                  # extended with each new arm
```

**The discipline log lives in the slot-0 Obsidian vault, not the repo.** Code
artifacts (scripts, src, benches) stay in the repo; the log is a vault note:

```
<slot-0 Obsidian vault>/20 - Proxima/Discipline/<initiative>/
  discipline.md             # the versioned log (see template below)
```

Prepend Obsidian frontmatter (`title`, `type: discipline-log`,
`tags: [proxima, discipline, <initiative>]`) and add a path-qualified wikilink
to the new log from the Discipline MOC at
`.../20 - Proxima/Discipline/Discipline Logs.md`. When the initiative is indexed
in `proxima/ai_docs`, point its `path` / `source_paths` at the vault log, not a
repo `docs/` path. (Outside slot-0, fall back to
`project/docs/<initiative>/discipline.md` in-repo.)

`discipline.md` template:

```markdown
# <initiative> discipline log

The 13-point gate (see SKILL): build / tests / clippy / micro-bench /
compare-bench / E2E / opt-sweep / SIMD-SM-noBox / O(1) / Cfg-API /
home-turf / Δ.

## C1 — <Component name>

| Build | Tests | Clippy | Micro-bench | Compare-bench | E2E | Opt | SIMD/SM/no-Box | O(1) | Cfg/API | Home-turf | Δ | Notes |

**Incumbent design point(s):** [for each named incumbent: what they were built for. e.g. "flume → MPMC under contention"; "hdrhistogram → multi-decade range + quantile query"]
**Opt-sweep findings:** [bullets — what you tried, why each choice]
**SIMD/SM/no-Box pass:** [bullets — applicable or not, why]
**O(1):** [one sentence — what data structures and why each is strict O(1)]

### Changelog
| Date | Change | Δ vs prior | CoV / runs | Host loadout |
| 2026-MM-DD | initial landing of X | baseline | 2.1%, 3 runs | quiet |
| 2026-MM-DD | tried Y (rationale) | +12% on A, **-30% on B**, rolled back | 4%, 3 runs | 1 other bench active |
| 2026-MM-DD | tried Z | +3% on A (within noise), kept simpler form | 6%, 5 runs | 2 other benches active |
| 2026-MM-DD | tried W | +8% on A, +5% on B, kept | 2%, 5 runs | quiet |

### Initial bench results
| arm | this | baseline-1 | baseline-2 | design-favors |
| --- | --- | --- | --- | --- |
| primitive-op (small N) | ... | ... | ... | neutral |
| primitive-op (large N) | ... | ... | ... | neutral |
| **incumbent home turf** | ... | ... | ... | **incumbent** |
| our design point (e.g. per-core, no contention) | ... | ... | ... | proxima |

**Honest read:** [what the numbers actually say — including losses on incumbent's home turf]
**Implication:** [what comes next — keep iterating? defer? acknowledge regime as out-of-scope?]
```

## Loop body — what to do per component

```
1. Pick the next leaf in the DAG. Write down its public surface (types, fns) in 10 lines.
   → (optional) Explore (sonnet) for prior art / how baselines solve it.
2. Add the sub-flag to Cargo.toml / equivalent.
3. Write the impl. Smallest viable; clarity over premature optimization.
4. Write tests (≥6 cases — happy, sad, edge, drop, concurrency).
   → delegate to test-writer (sonnet). Pass public surface + language rules.
5. cargo nextest run -E 'test(runtime::...::<component>::)'  → green. (doctests still run via `cargo test --doc` when present.)
6. cargo clippy ... -- -D warnings  → clean.
7. Write bench_<component>.rs with EXPLICIT baselines (the alternatives this replaces).
   → delegate scaffolding to a sonnet Agent; main thread reviews baselines named.
   → for each incumbent, identify their **design point** (README headline, their own bench suite, their data structure). Add at least one arm that engages it. Label every arm `design-favors: incumbent | proxima | neutral`. (point 13 of the gate)
8. Register bench in Cargo.toml with required-features = [...sub-flag...].
9. cargo bench --bench bench_<component>  → record numbers in discipline.md.
   → if CoV is high (multiple benches active), sweep 3-5 runs; record the range.
   → bench-running can be delegated to sonnet; ask for parsed thrpt + outlier count only.
10. Identify 1-3 obvious optimizations (cache padding, lock-free swap, no_alloc tightening).
11. Apply ONE. Re-bench. Compare delta.
    → use `--save-baseline` / `--baseline` to compare numerically, not by eye.
12. Delta ≥ 0 on every metric → keep. Mixed → judge worst-metric; if worst is bad, roll back.
    → if delta is within noise floor (CoV), the row says "no signal, kept simpler form".
13. Append a Changelog row for the tweak, including ROLLBACK if rolled back.
14. Repeat 10-13 until you've exhausted the obvious wins.
15. (Optional) code-reviewer + compliance-reviewer (sonnet, in parallel) for an independent read.
16. Mark component done in TodoWrite. Move to next leaf.
```

## The critical disciplines (do NOT cheat these)

**Don't move on with blank cells.** If "Compare-bench" is empty, the component is not done. Even if you "know" it's faster.

**Negative deltas are not failures, they're knowledge.** A row that says "tried X, got -30% on the contended path, rolled back" is the most valuable kind of row — it documents why the simpler thing was the right thing. Future-you will not remember.

**Vibes don't replace baselines.** "It should be faster because no mutex" — until you measure, you don't know. Modern mutex paths on M1/AMD64 are extremely fast; the no-mutex win you imagine may not exist on your target hardware.

**Roll back when worse.** The temptation to keep an optimization that "feels right" but benched negative is high. Resist it. If the data says rollback, roll back. Document why you tried; document why you reverted.

**One tweak at a time.** If you cache-pad AND change the atomic ordering AND restructure the slot layout simultaneously, you don't know which one moved the number. Single change → bench → record → next.

**Default off.** The new code is gated. Until the e2e bench shows the full stack wins, the production default does not change. The flag is the firewall.

**Match or beat the named incumbent — on their home turf, not yours.** The component must be at least as fast as the explicit alternative it replaces, **measured on the workload the incumbent was designed for** (point 13 of the gate). Beating an MPMC channel on SPSC isn't a result — the incumbent never engaged the work they were built for. Pass requires a measured win or honest tie on the incumbent-favored arm. If the incumbent is genuinely better on their home turf, two options: (a) document the regime as out-of-scope and ship anyway, restricting the claim to the regime where we DO win; or (b) don't land — roll back and rethink. What's not OK: shipping a "win" backed only by arms where the incumbent's design point was bypassed.

**For slot-0 components, hot-path invariants bind.** Components living inside slot-0 projects must also satisfy the repo AGENTS.md hot-path requirements: 500MB RAM cap, ≥55MB/s sustained throughput, sub-1ms query latency, zero-copy on query/scoring/traversal paths, no heap allocation in inner loops (use mmap/LSM/prebuilt storage), lock-free concurrent reads, no O(n²) on query/scoring/index paths. These are gates, not aspirations — a component that breaks them does not land regardless of how clean the micro-bench looks.

**Representation shortcuts are regressions until proven otherwise.** `String` instead of bytes, owned structs instead of borrowed views, `Clone` instead of zero-copy/`Copy`, heap allocation instead of stack/slab storage, and scalar byte scans instead of SIMD-backed search all require a discipline-log note with measured evidence or a named semantic boundary.

## Common pitfalls

- **Hardcoding the bench items count** — make it a const at the top of the bench file; tune for your target measurement time
- **Forgetting `required-features` on the bench `[[bench]]` stanza** — bench won't compile under default features otherwise
- **Capturing `&Consumer` (which is !Send) in a `thread::spawn` closure** — the consumer stays on one thread by design; drain on main, push producers to threads
- **Comparing against the wrong baseline** — `flume::unbounded()` and `flume::bounded(N)` have different perf profiles; bench against the one your runtime will actually use
- **Testing the incumbent in their worst regime** — single-producer-single-consumer on an MPMC channel; 16-byte input on a SIMD bulk-decoder; one counter on a registry-vec; one Subscriber layer on a layer-composition system. Looks like "we win" but the incumbent never got to engage their actual design point. Add a `design-favors: incumbent` arm or the result is meaningless (gate point 13).
- **Skipping the rollback when a tweak is "mostly" better** — if one metric regresses badly, even a +20% win on another metric doesn't justify keeping it
- **Letting cargo bench's progress lines hide errors** — capture stdout/stderr to a file with `> /tmp/x.log 2>&1`, then grep for `error` and `thrpt:`
- **Chasing a delta below the noise floor** — if CoV is 7% and your tweak shows +3%, you measured nothing. Either re-run in isolation or record "no signal" and move on.
- **Single bench run = single sample** — criterion's internal sampling is not a substitute for running the binary multiple times on a contended host. Range > point estimate.
- **Doing the grind on the main thread** — writing tests and bench scaffolding burns the main context for work a sonnet subagent does as well or better. Main thread is for the gate, the log, and the call.
- **Win-on-niche, lose-on-hot** — claiming a 12× win on a cold-path operation (loss event, error reject, drop-on-overflow) while losing 1.78× on the 100%-frequency hot path. The headline is the frequency-weighted scorecard, not the biggest ratio. See gate point 14 + "Frequency-weighted bench design". Historical example: claiming a 12× `loss_event` win (1% of packets) while losing 1.78× on `on_packet_sent` (100% of packets) — net loss in normal traffic, the headline was a lie.
- **Picking the convenient bench, not the meaningful bench** — benching what was 5 lines of setup instead of what runs 5 billion times per connection. The set of arms you ship should be driven by call frequency, not by setup-friction.
- **"Documented as deliberate trade-off"** — used to spin a real loss as a feature. If the loss is on the 80% case, "deliberate trade-off" is a euphemism for "we lost." Honest: name the cost, the frequency, and the impact in realistic traffic; then either fix it or scope the regime where it doesn't matter.
- **Skipping the 80% case because the incumbent's surface is awkward** — when the incumbent's API requires heavy setup, the temptation is to "defer that compare to an e2e bench" and headline the cheaper cold-path arms. That's a hidden loss. Bench the underlying library both wrappers share, OR mark the row "no verdict until e2e bench lands."

## Output style

The user wants to see:
- The **established discipline.md template, gate table, and changelog row format** — no per-project format reinvention. If the template doesn't fit, propose a change to the template; don't fork it inline.
- Honest numbers, including losses
- Specific deltas vs specific baselines
- Each tweak as its own row in the changelog
- A concise "honest read" sentence under the bench table
- An "implication" sentence for what to do next

The user does NOT want:
- Hand-waved claims of speedup without numbers
- Buried negative results
- Bundled optimizations where you can't attribute the delta
- "Production-ready" without an e2e bench composing it in
