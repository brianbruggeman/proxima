# Foundations, part 2: the pipe ergonomic surface

**Prerequisites:** [Foundations: the Pipe](./00-foundations.md), sections 1–7 and 13. You should already know: what `Pipe` is and its four shapes (source/sink/transform/observe); that pipes chain (`AndThen`); that proxima has four related traits — `Pipe`, `SendPipe`, `UnpinPipe`, `UnpinSendPipe` — because of a real compiler limitation; the free-function and stateful-`impl` forms of `#[proxima::piped]`; and `App::mount` + `app.serve(...)`.

**You will learn:** the layer of *sugar* built on top of everything Foundations taught — a fluent method-chaining trait (`PipeExt`), four macros that lift a bare closure into a pipe at the exact spot you write it (`pipe!`/`filter!`/`fanout!`/`fanin!`), why `#[proxima::piped]` emits *every* tier your function qualifies for rather than picking just one, why `App::mount` accepts four completely different kinds of value through one method with no runtime branching, and why time itself is never special-cased into the `Pipe` trait — it is composed, exactly like everything else here.

**New concepts (in order):** `PipeExt` · the four leaf macros · `#[proxima::piped]`'s tier closure, precisely · `App::mount`'s `IntoMountTarget<Via>` marker · the `Clock` capability.

Every code block below is either copied verbatim from a real file in this repository, or copied from `examples/pipe_ergonomics/main.rs` — a program written for this tutorial, compiled and run by `cargo run --example pipe_ergonomics`, whose transcript is reproduced at the end exactly as captured. Nothing here is invented; every claim is cited by `file:line`.

## Contents

1. Sugar, not algebra: what this whole document is
2. `PipeExt`: one blanket trait, four fluent methods
3. Why one trait was enough here (the short version)
4. Leaf macros: lifting a closure at the exact spot you write it
5. `filter!`, `fanout!`, `fanin!`: the same bridge, three shapes
6. `#[proxima::piped]` emits every tier you qualify for, not one
7. Why four traits at all: the long version, paid off
8. One `mount`, four shapes
9. Time is composed, not special-cased: the `Clock` capability
10. The whole transcript, and where to go next

## 1. Sugar, not algebra: what this whole document is

Everything Foundations taught you — `Pipe`, the four tiers, `AndThen`, `Handler`, `mount` — is the *algebra*: the fixed, minimal vocabulary the whole system is built from. Nothing in this document adds a new noun to that vocabulary. Every feature below either (a) builds an ordinary value you already know how to construct by hand, just with less typing, or (b) picks which of the four already-existing tiers a function belongs to. If you deleted every macro and extension trait this document teaches, every program you can write with them would still be expressible — just more verbosely, exactly the way `examples/send/main.rs` and `examples/rate_limit/main.rs`'s hand-written forms (cited in Foundations §7) still compile today. That is the test for whether something belongs in this document rather than in Foundations: does it add a capability, or does it just make an existing one nicer to spell?

## 2. `PipeExt`: one blanket trait, four fluent methods

Foundations §5 showed you `AndThen::new(first, second)` and `first.and_then(second)` side by side, without explaining where `.and_then()` actually lives. Here is the whole answer. It is a separate trait, `PipeExt`, blanket-implemented for every `Pipe` (`proxima-primitives/src/pipe/ext.rs:44,86`):

```rust
pub trait PipeExt: Pipe + Sized {
    // .and_then, .filter, .fanout, .fanin — see below
}

impl<P: Pipe> PipeExt for P {}
```

**A heads-up if you read older proxima docs:** some describe `and_then` as a *default method declared directly on `Pipe`*. That description is stale — it moved to this separate trait, and Foundations' own §5 has been corrected to match (see its "heads-up" callout). The reason it moved is worth knowing on its own terms, not just as a rename: `Pipe`'s job is the *minimal* contract every implementor must satisfy — the one thing `#[proxima::piped]` (Foundations §7) generates from a plain function. `.and_then()`/`.filter()`/`.fanout()`/`.fanin()` are convenience *on top* of that contract, wanted at nearly every call site but never load-bearing — nothing anywhere checks whether a type has them before trusting it as a pipe. Keeping "the contract" and "the sugar over the contract" as two traits means the sugar can grow new methods without ever touching the one trait every pipe author actually implements.

Four methods live on `PipeExt`, and you already understand three of their return types from Foundations:

| method | signature (abbreviated) | builds | source |
|---|---|---|---|
| `.and_then(next)` | `Next: Pipe<In = Self::Out>` | `AndThen<Self, Next>` | `ext.rs:47–53` |
| `.filter(predicate)` | `Pred: Pipe<Out = Self::In>` | `AndThen<Pred, Self>` | `ext.rs:56–62` |
| `.fanout(other)` | `Self: Clone, Self::In: Clone` | `FanOut<Self, AllOrNothing>` | `ext.rs:66–72` |
| `.fanin(other, strategy)` | `Self: UnpinPipe<In = (), Err = Exhausted>` | `FanIn<Self, Strategy, 2>` | `ext.rs:77–83` |

`.filter(predicate)` is worth a second look, because the argument order flips relative to what the name suggests: `self.filter(predicate)` builds `AndThen::new(predicate, self)` — the predicate runs *first*, and only an admitted item reaches `self`. Read it from the inner pipe's side: "run me, but filtered by this predicate first." Section 5 shows this end to end with `filter!`.

Real, compiled code — the first two rows, run for real (`examples/pipe_ergonomics/main.rs:24–39`):

```rust
async fn leaf_macro_demo() {
    let doubled = pipe!(|input: u64| -> Result<u64, Infallible> { Ok(input * 2) });
    let out = Pipe::call(&doubled, 21).await.expect("infallible");
    assert_eq!(out, 42);
    println!("pipe! : 21 -> {out}");
}

async fn and_then_demo() {
    let increment = pipe!(|input: u64| -> Result<u64, Infallible> { Ok(input + 1) });
    let double = pipe!(|input: u64| -> Result<u64, Infallible> { Ok(input * 2) });
    let chain = increment.and_then(double);
    let out = Pipe::call(&chain, 5).await.expect("infallible");
    assert_eq!(out, 12);
    println!("and_then: 5 -> increment -> double -> {out}");
}
```

(`pipe!` is section 4's macro — used here so this example needs no hand-written `struct`. Everything about `.and_then()` above applies identically to a hand-written `Pipe` impl.)

## 3. Why one trait was enough here (the short version)

Foundations §6 told you proxima needs *four separate* traits — `Pipe`, `SendPipe`, `UnpinPipe`, `UnpinSendPipe` — because an RPITIT future's `Send`/`Unpin`-ness can't be bounded generically on stable Rust (return-type notation, rust#109417, is unstable). A fair question: why doesn't `PipeExt` need the same treatment? Why is one `and_then` enough for all four tiers?

Because `PipeExt`'s methods never return a future. Look again at `.and_then`'s bound: `Next: Pipe<In = Self::Out>` — nothing about `Send` or `Unpin` anywhere. It returns a *value*, `AndThen<Self, Next>` — a plain struct that *wraps* the two pipes you gave it. The RTN limitation only bites the method that actually calls a pipe and hands back its future — `call` itself. `.and_then()` doesn't call anything; it just packages two values together. So the blanket `impl<P: Pipe> PipeExt for P {}` only ever needs `P: Pipe`, the one tier every pipe already has.

The tier tax doesn't disappear — it moves to where the future actually gets produced. `AndThen<First, Second>` itself pays it in full: `proxima-primitives/src/pipe/primitives.rs` has four separate impl blocks for `AndThen` — `Pipe` (`primitives.rs:203–221`), `SendPipe` (`primitives.rs:223–240`), `UnpinPipe` (`primitives.rs:327–343`), `UnpinSendPipe` (`primitives.rs:345–364`) — each conditioned on both `First` and `Second` implementing that same tier. `ext.rs`'s own module doc states the resulting split precisely (`ext.rs:11–15`): "the combinator value `and_then`/`filter`/`fanout`/`fanin` build carries whatever higher tiers its own stages qualify for regardless of which trait constructed it." Building the value is sugar, needing only the root tier; what the built value can later do is algebra, and pays the tier tax like everything else. Section 7 comes back to this from the other direction — why the *tax itself* costs four traits, not one.

## 4. Leaf macros: lifting a closure at the exact spot you write it

`#[proxima::piped]` (Foundations §7) needs a named, top-level `fn` or `impl` block to attach to. Sometimes you want a pipe as a throwaway value, right where you're building a chain — no name, no separate item. `pipe!` is that: a function-like macro (used as an expression, `pipe!(...)`, not a statement) that lifts a closure literal into a concrete pipe value inline (`proxima-macros/src/pipe_bang.rs:1–20`):

```rust
let doubled = pipe!(|input: u64| -> Result<u64, Infallible> { Ok(input * 2) });
```

Under the hood this expands to a fresh, unnamed tuple struct — `struct __ProximaPipeLeaf<F>(F)` — that owns the closure as a field and calls through it, plus whichever trait impls the closure's shape earns. That struct is minted *fresh at this call site*; it is never a shared type in `proxima-primitives` you could import (`pipe_bang.rs:142–151`). This is the one sanctioned pattern in this codebase for bridging a closure to a trait — a fresh struct per call site, never library machinery.

What tier(s) the resulting value reaches depends entirely on the closure's own shape, and this is where `pipe!` and `#[proxima::piped]` genuinely diverge, not just in spelling:

- **A plain (non-`async`) closure** — `|input: u64| -> Result<u64, Infallible> { .. }` — is wrapped in `core::future::ready`, whose future is `Unpin` unconditionally. It reaches `Pipe` *and* `UnpinPipe` by default, and — write `pipe!(.., send)` and nothing else changes about the closure — climbs to all four: `Pipe`, `SendPipe`, `UnpinPipe`, `UnpinSendPipe` in one expansion (`pipe_bang.rs:365–374`, `sync_closure_with_send_emits_all_four_tiers`), exactly the impl-all closure `#[proxima::piped]` computes (section 6).
- **An `async` closure** — `async move |input: u64| -> Result<u64, Infallible> { .. }`, stable since Rust 1.85 — is called straight through with no wrapper, zero extra cost. It reaches **`Pipe` only**, and two climbs are refused outright, each for a distinct, real reason (`pipe_bang.rs:212–233`, and proven by a real compile error below):

```rust
// examples/pipe_ergonomics — this does NOT compile, and the reason it
// doesn't is worth reading in full:
let leaf = pipe!(async move |input: u64| -> Result<u64, Infallible> { Ok(input) }, send);
```

```
error: `send` cannot be combined with an async closure lifted this way: proving the
closure's own returned future is `Send` requires naming `AsyncFnMut::CallRefFuture`,
which is `#[unstable(feature = "async_fn_traits")]` on stable Rust. Two ways around
it: (1) write a plain (non-`async`) closure instead — it reaches every tier,
including `send`; or (2) hand-write an `async fn` and lift it with
`#[proxima::piped(send)]`, whose `Send`-ness the compiler checks directly against
the concrete generated state machine.
```

This is a sharper distinction than "async closures are just more limited." An `async fn` *item* (what `#[proxima::piped(send)]` attaches to) compiles to one concrete, nameable state machine type — the compiler can inspect that specific type's captures and prove `Send` on stable Rust today, no unstable feature required. An `async` *closure* is different: proving something generic about the future *any* async closure returns needs to name the associated type on `AsyncFnMut`/`AsyncFn` that carries it (`CallRefFuture`), and naming that is gated behind the unstable `async_fn_traits` feature. So `#[proxima::piped(send)]` on a hand-written `async fn` can climb to `SendPipe` today; `pipe!` lifting an *async closure* with `send` genuinely cannot, on any stable compiler, until that feature stabilizes. The error message you just read states exactly that trade, not a vague "not supported."

`unpin` on an async closure is refused for an unrelated, equally real reason — this bridge is deliberately **zero-box**: it never reaches for `Box::pin` the way `#[proxima::piped(unpin, boxed)]` can (Foundations §7). An async closure's body is a compiler-generated state machine, which is `!Unpin` by construction, and there is no escape hatch here to make it `Unpin` without a heap allocation. If you need that climb, the macro's own error tells you the exit: hand-write the closure as a real `async fn` and lift *that* with `#[proxima::piped(unpin, boxed)]`, paying the one allocation per call explicitly, at the one macro that offers it.

A closure lifted either way must spell out its return type — `-> Result<Out, Err>` — because the macro reads that annotation; it does no type inference of its own (`pipe_bang.rs:250–260`). And passing an expression that is *not* a closure literal (an already-built pipe value) passes through unchanged — there is nothing to lift.

## 5. `filter!`, `fanout!`, `fanin!`: the same bridge, three shapes

`filter!` is the *decision* shape from the same bridge: `In -> Result<In, Err>` — `Ok` admits (the value survives, unchanged), `Err` rejects (`proxima-macros/src/filter_bang.rs:1–15`). It shares `pipe!`'s exact closure-lifting machinery, plus one macro-time check: the closure's admit type must equal its input type, checked by comparing the token strings of `In` and `Out` (`filter_bang.rs:40–54`) — not a trait bound, because the constraint is about the closure's *own* shape, not something `PipeExt::filter` could express generically (it only ever requires `Pred::Out == Self::In`, never `Pred::In == Pred::Out`).

Composed together, real and run (`examples/pipe_ergonomics/main.rs:42–57`):

```rust
#[derive(Debug, PartialEq, Eq)]
struct Odd;

async fn filter_demo() {
    let reject_odd = filter!(|input: u64| -> Result<u64, Odd> {
        if input.is_multiple_of(2) { Ok(input) } else { Err(Odd) }
    });
    let double = pipe!(|input: u64| -> Result<u64, Odd> { Ok(input * 2) });
    let gated = double.filter(reject_odd);

    let admitted = Pipe::call(&gated, 4).await.expect("4 is even");
    assert_eq!(admitted, 8);
    let rejected = Pipe::call(&gated, 3).await.expect_err("3 is odd");
    assert_eq!(rejected, Odd);
    println!("filter! : 4 -> {admitted}, 3 -> rejected({rejected:?})");
}
```

`fanout!(a, b, ..)` and `fanin!(a, b, ..)` are *variadic* — any number of arms — and each arm is either a closure literal (leaf-lifted the same way `pipe!` does) or an already-built pipe expression, passed through (`proxima-macros/src/fan_bang.rs:1–27`).

Variadic arity is the whole reason these two need their own macro rather than reusing `PipeExt::fanout`/`.fanin`, which only ever take *two* same-typed arms (the table in section 2). `FanOut<S, Policy>` and `FanIn<S, Strategy, N>` each hold a single, *homogeneous* array of one concrete type `S` — but `N` closure literals are `N` distinct, unnameable types, one freshly minted per call site. Reconciling "N distinct types" with "one `S`" without a `Box<dyn ..>` (which this codebase avoids by design — box-free is the house rule, `~/.claude/rules/rust.md`) is a sum type: the macro generates one enum, one variant per arm, each variant generic over that arm's own (possibly-anonymous) type, and that enum *becomes* `S`. `FanOut`/`FanIn`'s existing broadcast/merge loops are unchanged — they were already generic over `S: Pipe`/`S: UnpinPipe` — they just iterate over enum values now instead of one concrete struct (`fan_bang.rs:6–16`).

Its `Pipe`/`SendPipe` impls dispatch with an ordinary `match` inside one shared `async move { .. }` block — the same trick that unifies `AndThen`'s two stages into one anonymous future (section 3) unifies all `N` arms' distinct future types into one here. The `Unpin`/`UnpinSendPipe` impls can't use that trick (there is no `async move` on that tier to hide the union behind), so they get a second, hand-rolled poll-dispatch enum instead — one variant per arm, each holding that arm's own `Unpin` future (`fan_bang.rs:18–27`). Zero boxes either way; the cost is entirely in macro-generated code you never have to write or read.

`fanin!`'s arms carry one extra restriction `fanout!`'s don't, inherited directly from `FanIn` itself (Foundations §10): a synchronous, never-suspending merge loop needs `UnpinPipe`-shaped sources, so a closure-literal arm inside `fanin!` must be a plain, non-`async` closure — an async arm is refused with a specific, actionable message (`fan_bang.rs:133–141`):

```
error: fanin! arms must be plain (non-`async`) closures: FanIn's merge loop polls
each source synchronously in place and requires a genuinely `Unpin`,
never-suspending future. Lift an async source with `#[proxima::piped(unpin, boxed)]`
on a hand-written `async fn` first and pass the resulting value in as a
pass-through arm instead.
```

Both, run for real (`examples/pipe_ergonomics/main.rs:60–72`, `75–99` — `fanin_demo`'s poll loop elided below, it is the same `merge_in_place` shape Foundations §10 already showed you):

```rust
async fn fanout_demo() {
    let fan = fanout!(
        |input: u32| -> Result<(), Infallible> {
            println!("  fanout! arm a saw {input}");
            Ok(())
        },
        |input: u32| -> Result<(), Infallible> {
            println!("  fanout! arm b saw {input}");
            Ok(())
        },
    );
    Pipe::call(&fan, 7).await.expect("both arms accept");
}

fn fanin_demo() {
    let merged = fanin!(
        |(): ()| -> Result<u8, Exhausted> { Ok(1) },
        |(): ()| -> Result<u8, Exhausted> { Ok(2) },
    );
    // .. poll it in place, no allocation, no `unsafe` — see `examples/pipe_ergonomics/main.rs:80–95`
}
```

## 6. `#[proxima::piped]` emits every tier you qualify for, not one

Foundations §7 told you `#[proxima::piped]` "picks which of the four tiers a given function belongs to, and writes that tier's impl" — singular. **That description is stale**, and this document's own edits to Foundations correct it in place; the precise behavior is worth stating here because it is the single most surprising thing about the attribute macro if you only skimmed it once.

`#[proxima::piped]` computes the **full downward closure** of tiers your function's shape qualifies for, and writes one impl block for *every* tier in that closure — never just one — because the higher tiers are additive constraints on the same root contract, never a replacement for it (`proxima_primitives::pipe::primitives`'s own module doc says exactly this of the trait family; `proxima-macros/src/pipe_attr.rs:350–362`, `Tier::plan`):

```rust
pub(crate) fn plan(climbs_to_unpin: bool, send: bool) -> Vec<Tier> {
    let mut tiers = vec![Tier::Pipe];
    if send {
        tiers.push(Tier::SendPipe);
    }
    if climbs_to_unpin {
        tiers.push(Tier::UnpinPipe);
    }
    if climbs_to_unpin && send {
        tiers.push(Tier::UnpinSendPipe);
    }
    tiers
}
```

A plain `fn` with `send` reaches every tier in one expansion: `Pipe`, `SendPipe`, `UnpinPipe`, `UnpinSendPipe`. Not asserted — compiled and proven, four separate trait-bound assertions against the *same* macro-generated type (`examples/pipe_ergonomics/main.rs:102–118`; the demo continues to line 123 by also calling `triple` and checking its output, elided here since the point is the four assertions):

```rust
#[proxima::piped(send)]
fn triple(input: u64) -> Result<u64, Infallible> {
    Ok(input * 3)
}

fn assert_pipe<Value: Pipe>() {}
fn assert_send_pipe<Value: SendPipe>() {}
fn assert_unpin_pipe<Value: UnpinPipe>() {}
fn assert_unpin_send_pipe<Value: UnpinSendPipe>() {}

async fn piped_impl_all_demo() {
    // if `#[proxima::piped(send)]` picked only one tier, only one of these
    // four lines would compile — all four compiling is the impl-all proof.
    assert_pipe::<triple>();
    assert_send_pipe::<triple>();
    assert_unpin_pipe::<triple>();
    assert_unpin_send_pipe::<triple>();
    // .. also calls `Pipe::call(&triple, 14)` and checks the output is 42
}
```

If the macro really emitted "exactly one tier," three of those four lines would be a compile error (`error[E0277]: the trait bound triple: SendPipe is not satisfied`, or similar, for whichever tiers were withheld). All four compile, which is the falsifiable, mechanical version of the claim — the same discipline Foundations §14 already showed you for the algebra as a whole.

There is one real, and different, ambiguity this impl-all behavior creates, and it is worth naming precisely so you recognize it rather than mistake it for the (false) "only one tier" story: calling the shared method name `call` by plain dot-syntax on a value whose type implements more than one of these traits is genuinely ambiguous (`error[E0034]: multiple applicable items in scope`). `triple.call(14)` above would not compile on its own — you have to disambiguate with a fully-qualified path, `Pipe::call(&triple, 14)` (as the compiled example does) or `SendPipe::call(&triple, 14)`, depending on which tier's contract you're invoking. This is a call-site spelling question, not a reason for the macro to withhold a tier your function's shape already earned — and it is exactly why `PipeExt::and_then` in section 2 never runs into it: there is only ever *one* `PipeExt`, reaching every pipe through its single blanket impl, so `.and_then()` never has a second trait competing for the same name.

## 7. Why four traits at all: the long version, paid off

You now have enough vocabulary to see the whole shape of the constraint, not just its consequence. `Pipe::call` returns `impl Future<Output = ...>` — return-position `impl Trait` in a trait (RPITIT). The future that comes back is an *anonymous* type: you cannot write its name down anywhere, including in a `where` clause. That single fact is the entire reason this codebase has four traits instead of one:

- Writing `impl<P: Pipe + Send> SendPipe for P` — the blanket bridge you would reach for instinctively — requires bounding *the future `P::call` returns* as `Send`. But that future has no name to bound. The only stable mechanism that could name it is return-type notation (`P::call(..): Send`), tracked as rust#109417 and still unstable.
- So each additive promise — `Send`, `Unpin`, both — costs a full, separate, standalone trait, each with its *own* `call` method whose *own* RPITIT return type bakes the extra bound directly into its own signature (`+ Send`, `+ Unpin`, `+ Send + Unpin`). `proxima-primitives/src/pipe/primitives.rs:91–178` is literally the same four-field contract (`In`, `Out`, `Err`, `call`) written out four times — `Pipe` (91–102), `SendPipe` (104–124), `UnpinPipe` (142–152), `UnpinSendPipe` (165–178) — because there is no way to derive three of them from the fourth.
- Sections 3 and 6 of this document showed you the two faces of this cost from opposite directions: `PipeExt` (section 3) sidesteps it entirely because its methods never return a future, only a value. `#[proxima::piped]` (section 6) pays it in full and up front — the macro writes all four impl blocks so you never have to.

This is not a design proxima wants to keep. `proxima-primitives/src/pipe/primitives.rs:104–113` says so directly in its own doc comment: "when RTN stabilises, every tier below collapses back into `Pipe` plus a bound at the use site, and these traits are deletable." Until then, the tax is real, and every ergonomic feature in this document is a decision about *where* to pay it: once, by hand, in a leaf macro's generated code (`pipe!`/`filter!`/`fanout!`/`fanin!`, sections 4–5); once, by a proc-macro, at the definition site (`#[proxima::piped]`, section 6); or not at all, because the operation in question never needed to name a future (`PipeExt`, sections 2–3).

## 8. One `mount`, four shapes

Foundations §13 showed you `app.mount("/", hello)` once, with `hello` a `#[proxima::piped(send)]`-generated pipe. `App::mount` actually accepts four genuinely different kinds of value through that one method, with no runtime type-checking or branching on your part — the compiler picks the right path entirely at compile time (`src/app.rs:532–556`):

```rust
pub fn mount<Target, Via>(&self, path: &str, target: Target) -> Result<(), ProximaError>
where
    Target: IntoMountTarget<Via>,
{
    let target = target.into_mount_target();
    // ..
}
```

`Via` is a second, phantom type parameter you never name at the call site — Rust infers it. It exists to solve a coherence problem you would hit immediately without it: a handler-shaped pipe, a bare `async fn`, a registered name (`&str`/`String`), and an already-built `MountTarget` are four *overlapping-looking* shapes from the compiler's point of view, and writing one blanket `impl` per shape over the *same* trait would risk a coherence conflict (E0119) the moment some type could satisfy two of them at once. `src/app.rs:1159–1168` states the fix precisely: making `Via` part of the trait's own generic signature turns `IntoMountTarget<ViaName>`, `IntoMountTarget<ViaPipe>`, `IntoMountTarget<ViaFn>`, `IntoMountTarget<ViaTarget>` into four genuinely *distinct* trait instantiations. The compiler never needs to prove non-overlap between them — it's true by construction, because `ViaName` and `ViaPipe` are different concrete types, full stop.

The four shapes, each a real, already-tested call site:

- **`ViaPipe`** — a handler-shaped pipe: anything implementing `Handler` (which is blanket-implemented for every `SendPipe<In = Request<Bytes>, Out = Response<Bytes>, Err = ProximaError>`, per Foundations §13) — a `#[proxima::piped(send)]`-generated struct, or a hand-written one. `src/app.rs:1195–1202`.
- **`ViaFn`** — a *bare* `async fn(Request<Bytes>) -> Result<Response<Bytes>, ProximaError>`, with no `#[proxima::piped]` at all. `mount` wraps it in a small internal `FnHandler` adapter to reach `Handler` the same way `ViaPipe` does (`src/app.rs:1204–1236`) — the one sanctioned app-edge blanket this design needs, never library machinery.
- **`ViaName`** — a registered pipe name, `&str` or `String`, looked up in the app's own registry at mount time. Real usage: `app.mount("/foo", "cache")` (`src/app.rs:1547–1552`, test `unmatched_path_returns_404`).
- **`ViaTarget`** — an already-built `MountTarget` passed straight through, unchanged — the shape the daemon control plane uses when it has already resolved a handle or a name itself (`src/app.rs:1189–1193`).

The first two, compiled and mounted for real, no server actually started (`examples/pipe_ergonomics/main.rs:126–142`):

```rust
#[proxima::piped(send)]
async fn echo(request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    let (_, body) = request.body_bytes().await?;
    Ok(Response::ok(body))
}

async fn echo_fn(request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    let (_, body) = request.body_bytes().await?;
    Ok(Response::ok(body))
}

fn mount_shapes_demo() {
    let app = App::new().expect("app");
    app.mount("/via-pipe", echo).expect("mount a handler-shaped pipe");
    app.mount("/via-fn", echo_fn).expect("mount a bare async fn");
    println!("mount: ViaPipe and ViaFn both accepted by the same App::mount");
}
```

Notice `echo_fn` above has *no* `#[proxima::piped]` attribute anywhere — it is a completely ordinary `async fn`, and `App::mount` still accepts it. If your function's type does not satisfy any of the four `IntoMountTarget` arms, the compiler tells you so in plain language, not a wall of trait-resolution noise — `src/app.rs:1169–1172` puts a `#[diagnostic::on_unimplemented]` message directly on the trait (a stable, compile-error-only annotation; it changes nothing about what compiles, only what the error says when it doesn't):

```
error[E0277]: `{Self}` can't be mounted: expected a request handler — an
`async fn(Request<Bytes>) -> Result<Response<Bytes>, ProximaError>`, a
handler-shaped pipe, or a registered pipe name
```

## 9. Time is composed, not special-cased: the `Clock` capability

A natural question once you've seen `Pipe`/`PipeExt`/leaf macros/`mount` all avoid adding new nouns to the algebra: what about timing — readiness, delays, retries? Does `Pipe` have some special "are you ready yet" method?

No. `proxima_primitives::pipe::capabilities::Clock` is a small, ordinary trait — not a pipe, and not a method on `Pipe` (`proxima-primitives/src/pipe/capabilities.rs:51–57`):

```rust
pub trait Clock {
    type Delay: Future<Output = ()>;
    fn now_nanos(&self) -> u64;
    fn delay(&self, dur: Duration) -> Self::Delay;
}
```

`now_nanos` and `delay` are the *only* way time enters any logic built over `Clock` — nothing downstream can reach for the wall clock or a bare `sleep()` behind the trait's back. Production code injects `TimeClock` (`proxima-primitives/src/pipe/clock.rs:13`), which wraps `proxima-core`'s real monotonic driver; three real combinators in this codebase are generic over `Clock` today rather than sleeping directly — `Retry` (`proxima-primitives/src/pipe/retry.rs`), `Delay` (`delay.rs:31,88`), and `RateLimit` (`rate_limit.rs`).

Here is the honest correction to make about "the clock is a pipe," if you came in expecting `Clock` itself to implement `Pipe`: **it doesn't, and it isn't meant to.** `Clock` is a capability — a strategy a combinator is generic over — the same relationship `FanInStrategy` (Foundations §10) has to `FanIn`: it answers a control question, it never sees the payload, so it is a plain trait, not a pipe. What *does* compose as an ordinary pipe is whatever is *built on top of* a `Clock` — a timer, a retry loop, a rate limiter. `examples/clock/main.rs:85–101` makes this concrete with a `Timer` that is nothing more than the `Pipe` you already know how to write, generic over which `Clock` it holds:

```rust
struct Timer<ClockImpl: Clock> {
    clock: ClockImpl,
}

impl<ClockImpl: Clock> Pipe for Timer<ClockImpl> {
    type In = Duration;
    type Out = &'static str;
    type Err = Infallible;

    fn call(&self, wait: Duration) -> impl Future<Output = Result<&'static str, Infallible>> {
        let delay = self.clock.delay(wait);
        async move {
            delay.await;
            Ok("fired")
        }
    }
}
```

`Timer` never calls `sleep()`; it awaits whatever `self.clock.delay(wait)` hands back. Swap `TimeClock` (real monotonic time) for a `FakeClock` backed by nothing but a `Cell<u64>` that only moves when a test calls `.advance(..)` explicitly, and the exact same `Timer` becomes fully deterministic — no real time passes, no thread ever sleeps, and the test drives three `poll()`s by hand between `advance()` calls (`examples/clock/main.rs:113–158`, `cargo run --example clock`):

```
Timer scheduled for 30s against the fake clock. No thread sleeps, ever:
  poll #1 (t=0s):  Pending  — 30s hasn't happened, nothing is waiting on a clock tick
  advance(+15s) -> now_nanos = 15000000000
  poll #2 (t=15s): Pending  — halfway there, still not due
  advance(+15s) -> now_nanos = 30000000000
  poll #3 (t=30s): Ready("fired")  — deadline crossed, timer fires

zero real time passed. zero sleeps. zero threads parked — the fake clock made it deterministic.
```

That is the same lesson this entire document has been teaching from a different angle every time: proxima does not grow special methods on `Pipe` for readiness, timing, or dispatch shape. It composes an ordinary capability (`Clock`, `FanInStrategy`) *behind* an ordinary pipe (`Timer`, `FanIn`), and everything you already know about calling, chaining, and testing a pipe applies unchanged.

## 10. The whole transcript, and where to go next

Every code block in sections 2, 4–6, and 8 is one program, `examples/pipe_ergonomics/main.rs`. Run it yourself — `cargo run --example pipe_ergonomics` — and you should see exactly this, unedited:

```
pipe! : 21 -> 42
and_then: 5 -> increment -> double -> 12
filter! : 4 -> 8, 3 -> rejected(Odd)
  fanout! arm a saw 7
  fanout! arm b saw 7
fanin! : merged both sources -> [1, 2]
#[proxima::piped(send)] on a sync fn: reaches Pipe+SendPipe+UnpinPipe+UnpinSendPipe
mount: ViaPipe and ViaFn both accepted by the same App::mount
all pipe-ergonomics claims verified
```

You now have the complete picture: an algebra of four traits and a handful of combinators (Foundations), and a sugar layer over it (this document) that adds no new capability — only better spelling, and, in `#[proxima::piped]`'s case, boilerplate you no longer write by hand. Next:

- Foundations §8–§12 (filter, fan-out, fan-in, gate, signal) for the primitives `PipeExt`'s `.filter`/`.fanout`/`.fanin` build on top of.
- [the pattern gallery](../../book/src/algebra/patterns.md) for retry/auth/IAM/WAL/cron/ETL shapes built from this same algebra.
- [resilience/clock](../../book/src/resilience/clock.md) and [resilience/retry](../../book/src/resilience/retry.md) for `Clock` and `Retry` in full, beyond the single `Timer` this document showed.
- Any of the `Build a ...` tutorials in [the tutorials README](./README.md) — every one of them now reads faster, since `.and_then()`/`.filter()`/`.fanout()`/`.fanin()` and `#[proxima::piped]`'s impl-all behavior are assumed knowledge from here on.
