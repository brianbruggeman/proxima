# send

One pipe, five rungs. The same transform — `In -> Out` — can stay in its
permissive local form, borrowing state and never leaving the thread that
made it; climb to the cross-core form, which owns its data outright so it
can be handed to another thread entirely; or climb again so its future can
be polled exactly where it sits, with no heap involved at all. This example
runs the first three rungs as a real program; the last two are proven by
the trait's own compiled tests, cited below, and used for real by
[fan-in](../fan_in/README.md).

## Builds on

[transform](../transform/README.md) — you already know how to write one `Pipe`;
here you learn its bounds are a ladder, not a fixed contract.

## What it demonstrates

proxima inverts the usual arrangement: the permissive, thread-local form is
the ROOT of the ladder, and every other capability — crossing a core,
being pollable in place — is something a pipe opts INTO, not something it
starts with.

The ladder has five rungs. Each is strictly additive over the one before
it — a real capability bought at a real, stated cost:

- **borrow** (`Pipe`) — the root: no ownership bound at all, so a pipe can
  hold a live reference into whatever created it. Zero-copy, zero
  allocation, but stuck on the stack frame and thread that made it.
- **own** (`Pipe + 'static`) — still the local form, but the pipe now owns
  everything it needs instead of borrowing it. Enough to be stored or moved
  around — not enough to cross threads.
- **cross a core** (`SendPipe`) — the pipe and everything it produces must
  be safe to hand to another thread outright. This rung costs a hard
  guarantee, not just data ownership.
- **poll in place** (`UnpinPipe`) — the pipe's returned future is `Unpin`,
  so a caller can drive it with `Pin::new(&mut future).poll(cx)` directly —
  no `unsafe`, no `Box`, no heap allocation, no pin-projection. This is the
  rung a caller needs when it must hold *several* pipes' in-flight calls at
  once and poll each where it sits — a fan-in merge, or a bare-metal target
  with no allocator at all.
- **cross a core AND poll in place** (`UnpinSendPipe`) — both constraints
  together, for a caller that is polling `Send` futures in place. Reach for
  this rung only when a concrete caller demands it; wanting it
  speculatively means nothing needs it yet.

Climbing only goes one way: a pipe built on the permissive root can never be
pushed onto another thread, outlive the borrow it holds, or be polled in
place if its `call` was written as an `async` block — all three are proven
by the compiler, not a convention. Each rung past the root is also its own
self-contained contract rather than "the root plus a marker" — reaching it
is a deliberate choice for a pipe with a genuinely different job, not a free
upgrade.

Climb only as far as the use case demands. Borrowing pipes are not a
fallback or a toy — they are the permissive default this example proves
first.

### Why the ladder is five rungs but four separate traits

Every rung's `call` returns `impl Future<Output = ...>` directly from the
trait method — Rust calls this shape "return-position `impl Trait` in
trait," or RPITIT. An RPITIT future's `Send`-ness (or `Unpin`-ness) cannot
be strengthened by a subtrait bound on stable Rust — writing
`trait SendPipe: Pipe` and asking for "the same `call`, but its future is
also `Send`" needs return-type notation, which is unstable (`rust#109417`).
So every additive constraint after the root costs a full standalone trait
that repeats the whole contract with the extra bound baked into `call`'s
return type:
`Pipe`, `SendPipe`, `UnpinPipe`, `UnpinSendPipe`. Two independent bounds
(`Send`, `Unpin`), each on or off, is four combinations; `Pipe` is the
"neither" case, so four traits cover the five-rung ladder. When return-type
notation stabilizes, every rung above `Pipe` collapses back into `Pipe` plus
a bound at the call site, and `SendPipe`/`UnpinPipe`/`UnpinSendPipe` all
become deletable.
(Source: `proxima-primitives/src/pipe/primitives.rs:114-201`.)

### What `Unpin` buys, and what it costs

An ordinary `Pipe`/`SendPipe` `call` is written `async move { .. }`, and
Rust compiles that into a `!Unpin` future — a compiler-generated state
machine that may hold a reference into its own fields across an `.await`
point, which makes it unsafe to move once you start polling it. Driving a
`!Unpin` future from behind a plain `&mut` is unsound; the usual fix is
`Box::pin`, a heap allocation.

`UnpinPipe` asks for a `call` whose returned future is `Unpin` instead —
safe to move even mid-poll — so `Pin::new(&mut future)` is valid with no
`unsafe` and no `Box`. The cost lands entirely on the implementor: you
cannot write `async move { .. }` for an `UnpinPipe`'s `call`, because that
is always `!Unpin`. You write a plain struct that implements `Future` by
hand, with its own `poll`. That is exactly the shape a lock-free ring
already has — popping an item is synchronous, so its "future" is just a
value wrapped in `Poll::Ready`, never a real state machine. (`Poll` is the
plain either-or every `Future::poll` answers — `Ready(value)` or `Pending`
— that `.await` normally hides from you; `Context` carries the waker a
genuinely-pending future would register on, unused here because these
futures never actually wait.) This is proven, not asserted — the trait's
own test module builds precisely that shape:

```rust,ignore
// proxima-primitives/src/pipe/primitives.rs:620-636 (unpin_tier_tests)
struct RingPop(u8);
impl Future for RingPop {
    type Output = Result<u8, Infallible>;
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Ready(Ok(self.0))
    }
}

struct Ring(u8);
impl UnpinPipe for Ring {
    type In = ();
    type Out = u8;
    type Err = Infallible;
    fn call(&self, (): ()) -> impl Future<Output = Result<u8, Infallible>> + Unpin {
        RingPop(self.0)
    }
}
```

and then polls two of them in place, out of a fixed-size array, with no
allocation and no `unsafe`:

```rust,ignore
// proxima-primitives/src/pipe/primitives.rs:648-659 (unpin_tier_tests)
fn merge_in_place<S: UnpinPipe<In = ()>, const N: usize>(
    sources: &[S; N],
    cx: &mut Context<'_>,
) -> Poll<Result<S::Out, S::Err>> {
    for source in sources {
        let mut call = source.call(());
        if let Poll::Ready(out) = Pin::new(&mut call).poll(cx) {
            return Poll::Ready(out);
        }
    }
    Poll::Pending
}
```

That loop is the mechanism behind [fan-in](../fan_in/README.md): merging
`N` sources means holding every source's in-flight call at once and polling
each in turn, and `UnpinPipe` is what makes that legal with no heap
involved. Run `cargo test -p proxima-primitives --lib pipe::primitives` to
see `unpin_tier_tests` pass — this example's own `cargo run --example send`
below only exercises the `Pipe`/`SendPipe` rungs; `fan_in`'s example is
where `UnpinPipe` runs as part of a whole program.

## Run

```
cargo run --example send
```

## What you'll see

```
--- local: Pipe borrows, stays on this thread, no allocation ---
  Borrows::call(10) -> running total 10
  Borrows::call(20) -> running total 30
  Borrows::call(30) -> running total 60
  final ledger total, read straight through the borrow: 60

--- cross-thread: SendPipe owns its data, spawned, joined ---
  main thread: handing [1, 2, 3, 4, 5] to a spawned worker
  worker: summed 5 values on a worker thread -> 15
  main thread: joined the worker, sum = 15

--- the ladder (additive, one-directional) ---
  Pipe            (borrow, !Send)  Borrows above -- the root form
  Pipe + 'static  (own it, erase)   what into_local_handle/DynPipe require
  SendPipe        (+Send, +Sync)    Summarize above -- crosses a thread
```

The borrowing pipe never copies the state it reaches into — it holds a
reference, mutates a cell behind that reference, and reads the running
total straight back on the same thread. Because that state uses interior
mutability that isn't safely shared across threads, the borrow itself can't
be sent to another thread either: this pipe is stuck on one thread by
construction, not by convention, and it still runs the full local-form
contract with zero fuss.

The cross-core pipe owns its data outright, and both the pipe and the work
it produces are safe to share across threads. The example proves that bound
is load-bearing, not decorative: it moves the pipe and its input onto a
real OS thread, runs the work there, and joins the result back on the main
thread.

The ladder only climbs. The borrowing pipe is tied to a lifetime that isn't
`'static` by construction, so it can never be erased into a longer-lived
handle, and it can never be moved onto another thread the way the owning
pipe was above. Owning your data and crossing a core are costs a pipe opts
into — not a tax the permissive root pays by default.

## In algebra terms

- This example is one FORM — a transform, `In -> Out` — run at rungs of a
  single bound ladder, not as unrelated primitives. The program above
  exercises the first three rungs; the last two (`UnpinPipe`,
  `UnpinSendPipe`) are proven by the trait's own tests, cited above, because
  writing them here would need a hand-rolled `Future`, not an `async` block
  — a lesson the [fan-in](../fan_in/README.md) example carries instead.
- The ladder is additive and climbed in one direction only: borrow (the
  root, `Pipe`) -> own your data (still local, but no longer tied to a
  stack frame, `Pipe + 'static`) -> cross a core (`SendPipe`) -> poll in
  place (`UnpinPipe`) -> both at once (`UnpinSendPipe`). Each rung is
  proven at compile time, not by convention.
- Each rung buys capability and spends a freedom: borrowing is free but
  stuck on one thread; owning frees the pipe from the stack frame but costs
  the copy; crossing a core requires the pipe and its output to be safely
  shareable; polling in place requires giving up `async move { .. }` for a
  hand-written `poll`, in exchange for no heap and no `unsafe`.
- Nothing forces a pipe up the ladder. It is climbed only as far as the
  concrete use case demands — a pipe that never leaves its thread stays at
  the root, permissive by default.
