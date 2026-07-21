# gate — the gate pipe pattern

Readiness and backpressure, composed from existing primitives — never a method on the pipe.

## What it demonstrates and why it matters

A pipe's `call` is always callable: there is no readiness method to poll first. Readiness lives outside the pipe contract and is composed on top, from a small gate vocabulary (a gate that answers "armed or closed right now") plus whichever primitive gives "not ready" the shape you need:

1. **SHED** — say no now. A `filter` that reads the gate rather than the item: closed, every call is rejected with a reason; armed, calls reach the inner pipe. The answer is immediate and the work never starts.
2. **WAIT** — do nothing, cheaply. While the gate is closed the call is a no-op: the inner pipe is never invoked, nothing is allocated, nothing spins. Re-arm and the next call dispatches normally. Dormant, not busy-failing.
3. **BALANCE** — route around. A `fan-in` merges the backends and skips any that is not ready, so gating a backend on its own health simply makes it not-ready and the merge picks another. Which ready backend it picks is a **strategy** (round-robin here, but least-loaded or random are the same shape). Readiness becomes routing, with no readiness method on the pipe and no polling loop for the caller to write. Each backend (`BackendQueue`) is written as `#[piped] impl BackendQueue { fn call(&self, ..) -> impl Future<..> + Unpin { .. } }` — the sync half of `#[proxima::piped]`'s stateful form (`00-foundations.md` section 7): `call` already returns the future it needs to return (no `async`/`.await`), so the macro relocates the body unchanged and only writes the `UnpinPipe` trait header around it.

The point: one gate seam, three consumers (the `filter`, the dormant wrapper, the `fan-in`), each expressing a different backpressure policy — shed, park, route-around — none of which added a method to the pipe.

## Run

```
cargo run --example gate
```

## Expected output and what it proves

```
shed: a filter reading the gate
  ingest processing job 1
job 1: accepted
job 2: shed (Refused)
  ingest processing job 3
job 3: accepted

wait: dormant while the gate is closed
ungated (AlwaysArmed): 1 dispatched
closed gate: 0 dispatched (dormant, no-op)
armed gate: 3 dispatched (resumed)
disarmed again: 3 dispatched (dormant again)

balance: the merge skips a gated backend, drains the ready one
step 0: drained (a, 1)
step 1: drained (b, 10)
-- backend a disarmed (unhealthy) --
step 2: drained (b, 20)
step 3: drained (b, 30)
step 4: drained (b, 40)
-- backend a recovered, backend b disarmed (over capacity) --
step 5: drained (a, 2)
step 6: drained (a, 3)
step 7: drained (a, 4)
step 8: nothing ready this poll (closed backend skipped, not failed)
-- backend b recovered --
step 9: all backends drained
drained 4 from a, 4 from b
```

- **SHED**: job 2 is shed with a refusal reason while the gate is disarmed, with no queueing and no retry — the inner pipe never even runs (no "ingest processing job 2" line). Re-arming resumes admission immediately.
- **WAIT**: the ungated baseline always dispatches — it is not a special case, just the identity gate (always armed). The gated pipe dispatches 0 calls while closed, exactly 3 while armed, and 0 more once disarmed again: dormancy is enforced synchronously at the call boundary, not eventually.
- **BALANCE**: while backend `a` is disarmed (steps 2-4), every drained item comes from `b`; once `a` recovers and `b` is disarmed (steps 5-8), every drained item comes from `a`. Step 8 lands exactly when `a`'s queue has just emptied and `b` is still closed — the merge correctly reports "nothing ready" instead of stalling or erroring. Both backends still fully drain (4 items each) once their windows reopen — no items lost to the gate flips, only reordered around them.

The example also asserts these invariants inline (`assert_eq!`), so a regression in any of the composed primitives fails the run, not just the eyeball check.

## In algebra terms

- the gate is the seam, not a form or a primitive on its own: a shared answer to "armed or closed right now", flippable from outside — never a method on the pipe, never part of a call contract
- SHED = filter: a predicate reads the gate instead of the item; closed gate drops every call with a refusal reason, armed gate lets calls through to the inner pipe
- WAIT = a dormant wrapper: while the gate is closed, the wrapped call is a synchronous no-op — no inner future is even built, no polling, no allocation; re-arming resumes normal dispatch on the next call
- BALANCE = fan-in: a closed gate makes one source "not ready" for a single poll, so the round-robin merge just steps past it to whichever source is ready — no stall, no error
- one gate seam, three primitives it composes with (filter, dormant wrapper, fan-in), three backpressure strategies (shed, park, route-around) — none of them added a method to the pipe
