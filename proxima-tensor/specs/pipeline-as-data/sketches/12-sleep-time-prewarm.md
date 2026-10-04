# sketch 12: sleep-time prewarm at H10 (schedule, idle jobs)

status: paper test. Read at main 4b4be6cf with `git show main:<path>`. Nothing was built or run. `interop` =
`proxima-model-interop/src`; `PW` = `interop/generate/prewarm.rs`; `FU` = `interop/generate/prewarm_follow_up.rs`;
`PQ` = `interop/generate/prewarm_queue.rs`; `PG` = `interop/generate/prewarm_gate.rs`;
`PC` = `interop/generate/prompt_cache.rs`. `FT` = the draft under
`/Users/brianbruggeman/repos/slot-0/proxima-fsm-techniques/proxima-tensor/specs/fsm-techniques/` (read-only, one
opinion, not main).

Technique as the draft states it (`FT SPEC.md:72`, row T3): while idle, compute S(c) -> c' from a context c; at query
time answer from c' instead of c; up to 10 rethink calls, k in {1, 5, 10}. I did not read arXiv:2504.13171. The
draft's own note on this row: the prewarm form is lossless, the derived-context form is not (`FT SPEC.md:165`).

## ground facts on main (what exists, read)

The draft's audit row A17 says idle compute exists "only as anticipatory prefill" and has no job kind for derived
context (`FT SPEC.md:106`). Reading the files, main has more than that:
- Four moving parts. (1) Trigger: after a request stores its entry, `queue_prewarm_after_answer` queues the next
  turn's prefix, only if a turn-boundary suffix was registered (`PC:1288-1290`; `PW:278-298`, the early return on an
  empty suffix). (2) Queue: one slot, `submit` replaces an unrun job (`PQ:94-98`; the doc at `PQ:8-12` says the
  newest wins because an older prefix is one nobody will send). (3) Job body: `prewarm_queued` runs `prewarm_ids`, then,
  if that was neither skipped nor preempted, `follow_up_branches` (`PW:340-389`, the sequence at `PW:353-373`).
  (4) Admission: `PrewarmGate::try_begin` plus `request_waiting` (`PG:68`, `PG:77`), checked every chunk in the
  prefill (`PW:236-246`) and every decoded token while drafting (`FU:208-216`).
- The follow-up job is already a sleep-shaped job: it copies a stored entry's rows (`PrefixState::branch`,
  `PC:238-244`), samples a continuation from the copy at its own temperature and seed (`FU:196-199`), prefills the
  sample plus closing tokens, and stores the result as an ordinary entry marked `branch_base` (`FU:253`). Preemption
  is per token.
- Its knobs are four fields of `PromptCacheConfig`: `follow_up_branches` (default 0), `follow_up_max_tokens` (48),
  `follow_up_temperature_milli` (800), plus the closing tokens registered by setter (`serving.rs:524-531`,
  `prompt_cache_settings.rs:49-60`, `FU:36`).
- The drafts are returned only on the public path: `prewarm_follow_ups` returns `Vec<Vec<u32>>` (`FU:43-58`); the
  worker path discards them, matching only on `Err` (`PW:360-373`).
- Caller-driven entry points exist: `prewarm(ids, config)` (`PW:157`), `prewarm_follow_ups(ids, config)` (`FU:50`),
  `run_pending_prewarm` (`PW:321`), `with_prewarm_worker` (`PW:397`).
- `take_best` caps the resume length at `prompt_len - 1` (`PC:653`: `lcp.min(prompt_ids.len() - 1)`), which
  is why `follow_up_branches` can require `cached_len + 1 == base_ids.len()` (`FU:117`) after a prewarm that
  prefilled to `ids.len()` (`PW:216-222`). Both agree; recorded because they look contradictory on a first read.
- Branch entries are evicted before request entries (`PC:820-826`) and each holds a full copy of the rows
  (`serving.rs:524-529`).

## shape chosen, and the contested decision

Shape: configuration, ZERO pipes, and core changes at three places. The decision "which job runs next" is
`jobs.get(done)` guarded by `request_waiting()`, both already on main; no new pure function is warranted
(section 2).

Contested decision: is the sleep job a new job kind or the existing draft job with parameters? Chosen: the existing
draft job, parameterized per list element. The two differ in exactly three inputs (a lead instruction, a keep flag,
an output sink), listed as gaps. A separate "sleep" kind would copy `draft_branch` (`FU:183-266`) to change those three.

## 1. the configuration

```toml
[[schedule.idle]]
kind = "prewarm"

[[schedule.idle]]
kind = "draft"
branches = 5
max_tokens = 256
temperature_milli = 800
lead = "rethink"
keep = true
```

`lead` names a token list registered by setter, the same way `closing` is (`FU:36`); `keep = true` stores the result
as an ordinary entry instead of a `branch_base` entry.

Default reproduces today: an absent `schedule.idle` is `[prewarm, draft{branches = 0}]`, which is today's behaviour
(`follow_up_branches = 0`, `prompt_cache_settings.rs:50-52`; `FU:94` returns at once). The existing `follow_up_*`
fields become the first `draft` element's fields, one source for each number.

## 2. the pure function: there is none to add

Written as the expression it is: the next job is `jobs.get(completed)` when `!gate.request_waiting()`, else none.
`request_waiting` is one atomic load (`PG:77`). A function over `(jobs, completed, waiting)` has the same body as the
inline `get`, so it is not minted. The preemption points (per chunk, per token) are the existing checks; the draft's
R15b/R15d (`FT SPEC.md:280-283`) hold on main at chunk and token granularity, not the draft's "block boundary".

## 3. worked example (hand-derived, not executed)

Context c = a 6,000-token document, registered by the application with `prewarm(c)`, then `prewarm_follow_ups(c)` with
`branches = 3`, `lead = [L1, L2]` (instruction tokens), `closing = [E1, E2]`, `max_tokens = 4`.
- Entry for c: `cached_len = 6000` after the prefill (`PW:216-222`); `take_best` hands the drafter an entry rewound to
  5999 (`PC:653`), so the precondition at `FU:117` holds (`5999 + 1 = 6000`).
- Branch 0 samples at `seed + 1`, tokens `[a, b, c3, E1]`; the closing's first token ends the draft (`FU:224`
  trims it), so the draft is `[a, b, c3]` and the branch ids are `c ++ lead ++ [a, b, c3] ++ closing`. (The lead is
  gap G2.)
- Eviction with the default `max_entries = 4`: the answer entry for c plus 3 branches is 4 entries, exactly at the
  limit. `branches = 5` stores a fifth entry and the eviction rule picks a branch first (`PC:820-826`): an earlier
  draft is evicted by a later one. At `branches = 5` and `max_entries = 4`, at most 3 drafts survive, the one for c
  plus the last 3 drafted (derived from `PC:966-968` and the branch-first rule; not run).
- Consumption: a query prompt `c ++ lead ++ [a, b, c3] ++ closing ++ q` has `lcp` covering through the closing and
  hits that branch; only `q` prefills. A prompt that is `c ++ q` alone hits the entry for c and gets nothing from the
  draft.
- Control that must fail: with `keep = false`, the branch carries `branch_base` and is evicted first; a following
  unrelated request storing one entry at `max_entries = 4` and a full cache must evict it before any request entry.
  That is today's tested behaviour (`PC` test `unused_branches_are_evicted_before_entries_a_request_produced`,
  `PC:1642`); the new case is `keep = true` surviving the same pressure.

## 4. cache-key binding (cross-cutting rule, checked against `generate/prompt_cache_key.rs`)

- A derived-context entry is an entry whose ids are `c ++ lead ++ draft ++ closing`. Its rows are computed from those
  ids, so what a cached row means is unchanged, and the id-keyed lookup already separates a derived context from the
  original. No key field is needed. This differs from sketch 11 (cartridges) and sketch 10 (lossy codec), where
  the rows are not the rows the ids imply.
- Sampling fields are `_` in the key: temperature, top_k, top_p, seed and the rest (`prompt_cache_key.rs:126-134`)
  with the stated reason that the entry stores the ids actually forwarded. A draft sampled at 0.8 into an entry that a
  greedy request later resumes is therefore allowed by the key.
- One inherited assumption, not tested by me: branch rows come from a 1-row decode loop and then a rewind
  (`FU:236-249`), while a request's own rows come from M-row prefill; both sit under one key. I did not find a test
  asserting byte equality between the two row sources in the files I read. If they differ in the last bits, the
  key is silent about it already, today, for the existing follow-up path.
- `ring_slack_rows` is in the key (`prompt_cache_key.rs:67-70`) and a draft job does not change it.

## 5. HOOK GAPS (every place the sketch edits core)

G1. H10 job list. Missing input: a list. The two-step sequence is fixed in `prewarm_queued` (`PW:353-373`) and the
draft job's parameters are four `PromptCacheConfig` fields shared by every draft (`serving.rs:524-531`). Two draft
elements with different `max_tokens` or `lead` cannot coexist. Smallest change: `schedule.idle` read by
`prewarm_queued`, each `draft` element carrying its own parameters; `PromptCacheConfig` is `Copy` (`serving.rs:471`),
so the list lives where the closing tokens already live, in a setter into `PromptCache` (`PC:548-549`).

G2. H10 draft, lead tokens. Missing input: ids forwarded before the first sampled token. `draft_branch` feeds
`vec![base_ids[base_len - 1]]` as the decode input (`FU:203`) and appends `closing` only after the draft
(`FU:230-233`). An instruction between the context and the draft needs `[last_base_id] ++ lead` as the decode input,
and the check `state.ids != branch_ids[..keep]` (`FU:244`) must include the lead. Unread: whether the
decode loop accepts a multi-id first input from a seeded state; `prefill_through_stops` does (`PC:1139-1150` forwards
`ids[held..position]`).

G3. H10 draft, output sink. The worker path throws the drafts away (`PW:360-373`). The derived text is the product of
a sleep job; with the cache holding only ids behind `pub(super)` accessors (`PC:914` `entry_states`), the application
cannot read it. Smallest change: the worker takes an observer for `(source ids, drafts)`. Call site both ways: today
`prewarm_follow_ups` returns the drafts to its caller (`FU:43-58`); the worker path needs the same two values handed
to a caller-supplied closure, which `with_prewarm_worker<T>(&self, config, body: impl FnOnce() -> T)` (`PW:397`) could
take as one more generic parameter. A sink pipe (`Out = ()`) is the shape the SPEC already uses for placement; a
generic closure parameter is the same thing without a type.

G4. H10 queue policy. The slot holds one job and replaces on submit (`PQ:94-98`). Right for end-of-answer prefixes
(an older one is obsolete); wrong for sleep jobs on different contexts, where a second submit erases the first. Missing
input: a per-job policy. Smallest change: the queue's overflow behaviour chosen by the job's kind; a bounded queue is
what `proxima_core::ring` already provides (`proxima-core/src/ring/bounded.rs`; whether its `FailMode` fits is unread).

G5. H10/H13 keep marker. `branch_base.is_some()` means "evict me first" (`PC:820-826`) and also feeds the follow-up hit
count (`follow_up_hit_tokens` in `take_best_shifting`). A sleep result must survive pressure, so `keep = true` stores with
`branch_base = None`, which then reports no follow-up hit. Smallest change: the marker is set per element.

G6. Validation. `follow_up_branches` is documented to count against `byte_budget` and `max_entries`
(`serving.rs:524-529`) but I found no check: the only references outside the settings loader and the job are the field
and its default (`git grep -n follow_up_branches main -- proxima-model-interop/src` returns `serving.rs:530,573` plus the
job). With default `max_entries = 4`, `branches = 5` silently evicts its own earlier drafts (section 3). Add the rows
`branches + 1 <= max_entries` and `branches * entry_bytes <= byte_budget` (the second needs an entry size, unknown before
the first request; unmeasured, so it can only be a warning at store time).

## 6. is any of it a pipe? each candidate, against the two gates

- Job selection as a pipe: `jobs.get(completed)`; no work to wrap (section 2).
- The draft job as a pipe `In = &CacheEntry, Out = Vec<u32>`: it passes the pipe question. Call site both ways:
  `follow_up_branches(base_ids, ..)` versus `draft_job.call(entry).await`; identical work, plus a future around a
  synchronous decode loop that runs on a scoped worker thread (`PW:397-410`). Relocation.
- Observer for G3 as a sink pipe: the SPEC already has the placement precedent; the closure parameter does the same
  without minting a type. If a pipe is preferred it is a sink of about 5 lines (`In = (Vec<u32>, Vec<Vec<u32>>)`,
  `Out = ()`), at the one hook G3.
Therefore zero pipes beyond the optional 5-line sink.

## 7. designs abandoned

- A separate `sleep` job kind: abandoned for the parameterized draft job (shape chosen).
- A trigger list (`end_of_answer`, `on_context_registered`, `timer`): abandoned. The two triggers on main are the
  end-of-answer queue (`PC:1288`) and the caller's own call to `prewarm`; the second needs no configuration, and a timer
  trigger has no consumer in the files I read.
- Per-job priority and preemption classes (the draft's R15b): abandoned; the gate is a count of pending requests
  (`PG:77`) and any request preempts any idle job, which is all a single device allows.
- Storing derived context in a side map keyed by the source content hash: abandoned for entries keyed by ids, which
  the trie already indexes and the key already separates.

## defects found in passing

- D1. The draft's A17 (`FT SPEC.md:106`) misdescribes main: a draft-and-prefill idle job with per-token preemption
  exists (`FU:78-266`), default off. The audit row would have led to a duplicate job kind.
- D2. `queue_prewarm_after_answer` returns early on an empty suffix (`PW:291`) while the follow-up job also
  requires non-empty `closing` (`FU:95`): two registrations by setter, off by default, with no config surface and no
  report of why nothing ran when one is missing.
- D3. `PrewarmReport` has no field for the follow-up result and `run_queued_prewarm` returns only the prefill report
  (`PW:330-338`), so a caller of `run_pending_prewarm` cannot tell whether drafting ran.
- D4. A bare `std::sync::Mutex` guards the queue and the gate (`PQ:43`, `PG:22`); the module docs justify it as
  the dedicated blocking worker case and note that no `proxima-lock` crate exists in this workspace (`PG:11-15`).
  Recorded because principle 21 names `proxima_lock::Mutex`, which does not resolve here.

## verdict

fits after 6 named hook changes (G1 to G6); G4 and G6 are policy and validation rather than new capability, and the
caller-driven route (`prewarm`, then `prewarm_follow_ups`) runs a branch-sampling job today without G1 to G5 (not run;
the lead instruction and the keep flag are what it lacks). Configuration plus zero pipes. Not claimed: that a derived
context improves any answer, that branch rows equal prefill rows bit for bit, or any latency.
