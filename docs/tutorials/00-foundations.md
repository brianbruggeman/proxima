# Foundations: the Pipe

This is the only tutorial you must read before any other. It assumes you can read a little Rust (a `struct`, a function, `async`/`.await`, `Result`) but **nothing** about proxima. Every proxima term is defined the first time it appears.

By the end you will understand: what a pipe is, the four shapes it can take, how to chain two of them, why proxima has four related `Pipe` traits instead of one, the idiomatic way to write a pipe with a macro, the handful of ways to connect many pipes together (the "pipe algebra"), and how to run a pipe as a real web server.

Every code block below is copied verbatim from a real file in this repository — either a doctest that `cargo test` compiles, a unit test, or a runnable `examples/*/main.rs`. Each is cited by path and line number so you can go read it yourself, and every `cargo run` transcript shown is real output this tutorial's author captured by actually running the command.

## Contents

1. The one idea (no code)
2. Your first pipe
3. The four shapes a pipe can take
4. When a pipe can fail
5. Chain two pipes: `and_then`
6. Four tiers, one idea: how much a pipe can promise
7. The idiomatic way to write one: `#[proxima::piped]`
8. Let some things through: filter
9. Send one thing to many: fan-out
10. Merge many into one: fan-in
11. Open or closed: gate
12. Wait for a one-time event: signal
13. A pipe that answers web requests, and how to serve it
14. The algebra is enforced by the compiler, not by us
15. Where to go next

## 1. The one idea

Picture an assembly line. Each station does exactly one job: something arrives, the station works on it, and something leaves (or the station flags a problem). A **pipe** is one such station.

That is proxima's whole idea: **everything is a pipe, and big things are just small pipes connected together.** A web server is a pipe. A filter that rejects bad requests is a pipe. A merge that combines several queues into one is a pipe. A complete API gateway is a handful of pipes wired together. Learn what one pipe is and the few ways to connect them, and you can build all of it — because it is all the same building block.

So we start with one pipe, and we do not touch networks or servers until the building block is firmly in hand.

## 2. Your first pipe

In Rust, a pipe is any type that implements the `Pipe` trait. Here is its definition, copied verbatim from `proxima-primitives/src/pipe/primitives.rs:89–99` (its doc comments are trimmed here for space; the file has the full rationale):

```rust
pub trait Pipe {
    type In;
    type Out;
    type Err: Debug + 'static;

    fn call(&self, input: Self::In) -> impl Future<Output = Result<Self::Out, Self::Err>>;
}
```

Three names describe the station: `In` is what comes in, `Out` is what goes out, `Err` is what a failure looks like (`Err` must implement Rust's `Debug` trait and not borrow anything — that is what `: Debug + 'static` means; more on `Err` in section 4). The one method, `call`, is the work: it takes an `In` and eventually produces either an `Out` (success) or an `Err` (failure). It is `async` — "eventually" means it may pause and resume, for example while waiting on the network — which is why it returns a `Future` you `.await`. If you have not met `async` yet, for now read `call` as "a function that returns `Result<Out, Err>`, possibly after some waiting."

(There is a way to chain two pipes together, `and_then` — but it is not a second method on this trait; it lives on a separate sugar trait every pipe gets for free. That is section 5. Ignore it for now.)

Here is a complete, real pipe — `Double`, copied verbatim from the doctest at `proxima-primitives/src/pipe/primitives.rs:46–54` (this exact block is compiled and run by `cargo test`, so it cannot describe a trait that no longer exists):

```rust
struct Double;
impl Pipe for Double {
    type In = u64;
    type Out = u64;
    type Err = Infallible;
    fn call(&self, input: u64) -> impl Future<Output = Result<u64, Infallible>> {
        async move { Ok(input * 2) }
    }
}
```

`Double` takes a `u64`, doubles it, and returns it. It can never fail, which Rust spells `Infallible` (a type with zero possible values — section 4 explains why that matters).

That is the entire building block. There is no shortcut and no macro required — you write a small type and implement one method. (`call` returns `impl Future` directly: proxima never puts a future in a `Box`, so you never need the usual async-trait boilerplate — `Box::pin`, `#[async_trait]` — to write a pipe. If you have not met those, good: you will not use them here.)

## 3. The four shapes a pipe can take

You do not learn four different tools. You choose what `In` and `Out` are (where `()` is Rust's "nothing" value), and the same trait becomes four familiar things. This table is copied verbatim from `Pipe`'s doc comment, `proxima-primitives/src/pipe/primitives.rs:30–35`:

| form      | shape       | what it is                                |
|-----------|-------------|--------------------------------------------|
| transform | `In -> Out` | turns one thing into another              |
| source    | `() -> Out` | takes nothing, produces something         |
| sink      | `In -> ()`  | takes something, produces nothing         |
| observe   | `In -> In`  | hands back its input; acts on the side    |

`Double` above is a **transform**. Here are the other three, also copied verbatim from the same doctest (`primitives.rs:56–87`):

```rust
// source: () -> Out. Nothing goes in.
struct Always;
impl Pipe for Always {
    type In = ();
    type Out = u64;
    type Err = Infallible;
    fn call(&self, _input: ()) -> impl Future<Output = Result<u64, Infallible>> {
        async move { Ok(7) }
    }
}

// sink: In -> (). Nothing comes out.
struct Discard;
impl Pipe for Discard {
    type In = u64;
    type Out = ();
    type Err = Infallible;
    fn call(&self, _input: u64) -> impl Future<Output = Result<(), Infallible>> {
        async move { Ok(()) }
    }
}

// observe: In -> In. Out == In is what makes it an observe.
struct Echo;
impl Pipe for Echo {
    type In = u64;
    type Out = u64;
    type Err = Infallible;
    fn call(&self, input: u64) -> impl Future<Output = Result<u64, Infallible>> {
        async move { Ok(input) }
    }
}
```

Four roles, one trait, no new API — you just chose the types.

To see this run as a real program, `examples/transform/main.rs` builds the same four shapes under different names (`Counter` = source, `Double` = transform, `Echo` = observe, `Print` = sink) and chains them in a loop. Its source form is worth reading because it shows a pipe holding its own state between calls — `Counter` uses a `Cell<u64>` (the standard library's way to mutate a value through a shared `&self`, since `call`'s signature only ever gives you `&self`, never `&mut self`) (`examples/transform/main.rs:22–38`):

```rust
struct Counter {
    next: Cell<u64>,
}

impl Pipe for Counter {
    type In = ();
    type Out = u64;
    type Err = Infallible;

    fn call(&self, (): ()) -> impl Future<Output = Result<u64, Infallible>> {
        let value = self.next.get();
        self.next.set(value + 1);
        async move { Ok(value) }
    }
}
```

Run it — this is the real transcript from `cargo run --example transform`, first round only:

```
--- round 0: one Pipe trait, four roles chosen by type ---
source    (In=(),    Out=u64):  () -> 0
transform (In=u64,   Out=u64):  0 -> 0
  echo: observed 0, call #1
observe   (In=Out=u64):         0 -> 0 (unchanged)
  sink: final value 0
```

## 4. When a pipe can fail

`Err` is whatever type best describes this pipe's failures. `Infallible` — every pipe above uses it — is the standard library's type with zero possible values: if a function's error type is `Infallible`, Rust's own type checker can prove that branch of the `Result` can never actually happen. `examples/send/main.rs:76–86` puts that to use directly: because `Err = Infallible` has no variants, matching it out is *exhaustive*, so there is no panic path, unlike calling `.expect()` on a `Result` that really could fail:

```rust
fn call_pipe<PipeImpl: Pipe<Err = Infallible>>(
    pipe: &PipeImpl,
    input: PipeImpl::In,
) -> PipeImpl::Out {
    match futures::executor::block_on(pipe.call(input)) {
        Ok(output) => output,
        Err(never) => match never {},
    }
}
```

`match never {}` compiles with no arms because `Infallible` has no values to match — the compiler is checking your pipe truly cannot fail, not just trusting you.

Use a real error type when a pipe *can* fail. Inside `call` you return failures the normal Rust way — `Err(...)`, or the `?` operator to pass a failure upward. The next section shows why `?` lines up cleanly once you connect two pipes with different error types.

## 5. Chain two pipes: `and_then`

Here is the first way to combine pipes, and the reason "everything is a pipe" pays off. If one pipe's output is another pipe's input, you can run them back to back. `.and_then()` is not a method on `Pipe` itself — it lives on a small, separate *extension* trait, `PipeExt` (`proxima-primitives/src/pipe/ext.rs:44–53`), that is blanket-implemented for every `Pipe`:

```rust
pub trait PipeExt: Pipe + Sized {
    fn and_then<Next>(self, next: Next) -> AndThen<Self, Next>
    where
        Next: Pipe<In = Self::Out>,
        Next::Err: From<Self::Err>,
    {
        AndThen::new(self, next)
    }
    // .filter/.fanout/.fanin live on the same trait — section 8 onward
}

impl<P: Pipe> PipeExt for P {}
```

**A heads-up if you read older proxima code or docs:** `and_then` used to be a *default* method declared directly on `Pipe` itself (`Self: Sized` in its own `where` clause, so every `Pipe` got it automatically). It moved to this separate `PipeExt` trait — the module's own doc comment explains why (`ext.rs:1–15`): `Pipe`'s job is the minimal contract every implementor must satisfy (and `#[proxima::piped]`, section 7, generates that contract from a plain function); `.and_then()`/`.filter()`/`.fanout()`/`.fanin()` are fluent sugar over it, wanted at *every* call site but never load-bearing — nothing gates on whether a type has them. Separating "the contract" from "the sugar over the contract" means the sugar can change (new combinator methods, better ergonomics) without touching the trait every pipe author actually implements. Since every pipe already implements the root `Pipe` (the other three tiers are additive, never a replacement — section 6), one blanket `impl<P: Pipe> PipeExt for P {}` reaches all of them, and there is no second trait competing for the same method names. If you see `and_then` described as living directly on `Pipe`, that description is stale; `PipeExt` is where it lives now, everywhere in this codebase.

`Self: Sized` (now on `PipeExt`'s own supertrait bound, `PipeExt: Pipe + Sized`, rather than on `and_then` itself) excludes the rare case where `Self` is an unsized type like `dyn Pipe` — every ordinary `struct` you write satisfies it automatically, so you will not think about this bound again. You call `and_then` like this: `first.and_then(second)` runs `first`, then feeds its output into `second`. Two rules make it type-check:

- `second`'s `In` must equal `first`'s `Out` (`Next: Pipe<In = Self::Out>`) — the output of one must fit the input of the next.
- `second`'s `Err` must be able to absorb `first`'s `Err` (`Next::Err: From<Self::Err>` — `From` is the standard library's "can be built from" conversion) — so a failure anywhere in the chain surfaces as one error type.

`and_then` returns `AndThen<First, Second>`, a real struct (`primitives.rs:189–193`) that is *itself* a `Pipe` (`primitives.rs:203–221`):

```rust
impl<First, Second> Pipe for AndThen<First, Second>
where
    First: Pipe,
    Second: Pipe<In = First::Out>,
    Second::Err: From<First::Err>,
{
    type In = First::In;
    type Out = Second::Out;
    type Err = Second::Err;

    fn call(&self, input: Self::In) -> impl Future<Output = Result<Self::Out, Self::Err>> {
        async move {
            let intermediate = self.first.call(input).await?;
            self.second.call(intermediate).await
        }
    }
}
```

Because `AndThen` is itself a `Pipe`, you can call `.and_then()` on it again — chains nest without limit: `a.and_then(b).and_then(c)`.

Here is a real, fallible chain, from the crate's own test suite (`primitives.rs:476–519`). `Increment` can overflow; `Halve` rejects odd numbers with its own, different error type, but absorbs `Increment`'s error via `From`:

```rust
struct Increment;
impl Pipe for Increment {
    type In = u64;
    type Out = u64;
    type Err = Overflow;
    fn call(&self, input: u64) -> impl Future<Output = Result<u64, Overflow>> {
        async move { input.checked_add(1).ok_or(Overflow) }
    }
}

enum HalveError {
    Odd,
    Upstream(Overflow),
}
impl From<Overflow> for HalveError {
    fn from(value: Overflow) -> Self { HalveError::Upstream(value) }
}

struct Halve;
impl Pipe for Halve {
    type In = u64;
    type Out = u64;
    type Err = HalveError;
    fn call(&self, input: u64) -> impl Future<Output = Result<u64, HalveError>> {
        async move {
            if input.is_multiple_of(2) { Ok(input / 2) } else { Err(HalveError::Odd) }
        }
    }
}
```

And the actual chain, from the same file's test module (`primitives.rs:599–604`):

```rust
// 5 -> increment -> 6 -> halve -> 3. Both stages succeed.
let chain = Increment.and_then(Halve);
let out = block_on(Pipe::call(&chain, 5)).expect("even intermediate halves");
assert_eq!(out, 3);
```

That `From<Overflow> for HalveError` impl is *why* `?` works inside `AndThen::call`: it is what lets `self.first.call(input).await?` convert an `Overflow` into a `HalveError` automatically. This is the composition law of the whole algebra: error types absorb their upstream's errors via `From`, so a failure anywhere in a long chain always surfaces as one well-typed error at the end.

The chain also genuinely stops on the first failure — it does not run the second stage "just in case." A test proves it by recording whether the second stage ever ran (`primitives.rs:696–708`):

```rust
#[test]
fn and_then_short_circuits_before_the_second_stage_on_first_stage_error() {
    let ran = core::cell::Cell::new(false);
    let chain = AlwaysFail.and_then(SpyRef { ran: &ran });

    let err = block_on(Pipe::call(&chain, 1)).expect_err("first stage always fails");

    assert_eq!(err, HalveError::Upstream(Overflow));
    assert!(
        !ran.get(),
        "second stage must not run after first stage errors"
    );
}
```

**A heads-up if you read older proxima code or docs:** this combinator used to be named `Series` (`Series::new(first, second)`). It was renamed to `AndThen` to match the `.and_then()` method name — the same convention Rust's own `Option`/`Result` use (`.map()` returns `Map`, `.and_then()` returns... `AndThen`). If you see `Series` anywhere, it is stale; the current name, everywhere in this codebase, is `AndThen`.

(One honest limitation — and it is a wall, not a to-do. `and_then` is implemented for the `Pipe` and `SendPipe` tiers (section 6 explains what those are) but **cannot** be implemented for `UnpinPipe`/`UnpinSendPipe` on stable Rust. To chain, the returned future has to hold stage one's future across polls and then stage two's — a state machine with those futures as named fields. But a pipe's `call` returns `impl Future`, an anonymous type you cannot name, so you cannot declare the field; and writing the body as an `async` block fails too, because an async block is `!Unpin` (`error[E0277]: async block cannot be unpinned`). The only escape is boxing, which allocates — on the one tier that exists to avoid allocating. This is the same limitation that shapes everything else here: `impl Trait` in associated position is unstable ([rust#109417](https://github.com/rust-lang/rust/issues/109417)). When it lands, this restriction disappears — along with the need for four separate traits at all. Until then: climb tiers only as far as you need, and chain at the root `Pipe` tier, which is what every example above does.)

## 6. Four tiers, one idea: how much a pipe can promise

So far every pipe has implemented plain `Pipe`. But proxima actually defines **four** related traits — `Pipe`, `SendPipe`, `UnpinPipe`, `UnpinSendPipe` — and you will see all four used in real code. They are not four different concepts to learn; they are the same one contract (`In`/`Out`/`Err`/`call`), with different extra promises bolted on. Think of them as a ladder you climb only when you need to:

- **`Pipe`** — the root. No extra promise. `call`'s returned future can borrow from `&self` and never has to leave the thread that created it. This is the permissive default: zero-copy, no allocation, no thread hop.
- **`SendPipe`** — adds `Send + Sync + 'static` on the pipe itself, and `Send` on the returned future. This means the pipe owns its data outright (nothing borrowed) and is safe to move across an OS thread or CPU core boundary.
- **`UnpinPipe`** — adds `Unpin` on the returned future, meaning a caller can poll it in place (`Pin::new(&mut future).poll(cx)`) without `unsafe`, `Box`, or heap allocation. An ordinary `async fn` body is *not* `Unpin` (the compiler-generated state machine can reference itself internally), so reaching this tier means hand-writing the future as a plain `poll` struct instead.
- **`UnpinSendPipe`** — both extra promises at once: `Send` and `Unpin`.

Here is why proxima writes these as four *separate*, standalone traits instead of one `Pipe` trait with optional bounds. `Pipe::call` returns `impl Future<...>` — this is called "return-position impl Trait in traits," or RPITIT. On today's stable Rust, there is no way to write a blanket impl like `impl<P: Pipe + Send> SendPipe for P` — doing so would require *bounding the future that `P::call` returns*, which needs a Rust feature called return-type notation (RTN) that is still unstable (tracked as [rust#109417](https://github.com/rust-lang/rust/issues/109417)). So each additive promise costs a full, separate copy of the trait, hand-written. `proxima-primitives/src/pipe/primitives.rs` says this plainly in its own doc comments (lines 118–123, on `SendPipe`):

> There is no blanket bridge from `Pipe`, and there cannot be one: writing `impl<P: Pipe + Send> SendPipe for P` requires bounding `P::call`'s returned future — a bound on an RPITIT return type, i.e. return-type notation, which is unstable (rust#109417). So each additive constraint costs a full standalone copy of the contract. When RTN stabilises, every tier below collapses back into `Pipe` plus a bound at the use site, and these traits are deletable.

That is not a design proxima wants to keep forever — it is a workaround for a compiler limitation that will one day go away. Until then, four traits it is. Section 5's `AndThen` struct showed you this tax directly: it has one `impl Pipe for AndThen<..>` (`primitives.rs:226`) and a second, nearly-identical `impl SendPipe for AndThen<..>` (`primitives.rs:246`) — the same logic, written twice, because there is no way to derive one from the other.

`examples/send/main.rs` demonstrates the first two rungs of the ladder with two real pipes. `Borrows` holds a reference and can never leave the stack frame that owns it — `Pipe` puts no `Send` or `'static` bound on `Self`, so this compiles even though it cannot outlive `ledger` (`examples/send/main.rs:30–49`):

```rust
struct Ledger {
    total: Cell<u64>,
}

struct Borrows<'a>(&'a Ledger);

impl Pipe for Borrows<'_> {
    type In = u64;
    type Out = u64;
    type Err = Infallible;

    fn call(&self, input: u64) -> impl Future<Output = Result<u64, Infallible>> {
        let total = self.0.total.get() + input;
        self.0.total.set(total);
        async move { Ok(total) }
    }
}
```

`Summarize` owns its data outright and satisfies `Send + Sync + 'static`, so it can implement `SendPipe` and be spawned onto a real OS thread (`examples/send/main.rs:54–74`, spawn call at `main.rs:96`):

```rust
struct Summarize {
    label: String,
}

impl SendPipe for Summarize {
    type In = Vec<u64>;
    type Out = u64;
    type Err = Infallible;

    fn call(&self, input: Vec<u64>) -> impl Future<Output = Result<u64, Infallible>> + Send {
        let label = self.label.clone();
        async move {
            let sum: u64 = input.iter().sum();
            println!("  {label}: summed {} values on a worker thread -> {sum}", input.len());
            Ok(sum)
        }
    }
}
```

The real, unedited transcript from `cargo run --example send`:

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

`UnpinPipe` is the tier you reach for when you need to hold several in-flight calls at once and poll each one in place — exactly what section 10 (fan-in) needs, and exactly why fan-in requires its sources to be `UnpinPipe`. Here is a real (test-only, not public API) example of what implementing it by hand looks like, from `primitives.rs`'s own `unpin_tier_tests` module (`primitives.rs:620–636`). It writes `Future` directly instead of using `async`/`.await`: `poll` is the plain either-or every `Future` answers when asked — `Poll::Ready(value)` (done) or `Poll::Pending` (not yet) — that `.await` normally hides from you; note there is no `async` block anywhere, because an `async` block's future is never `Unpin`:

```rust
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

`UnpinSendPipe` is the top rung — both promises at once — and the source code is direct about it not being a default choice (`primitives.rs:185–187`): "Do not reach for this rung by default. Wanting it usually means a caller is paying to poll `Send` futures in place; wanting it *speculatively* means nothing needs it yet." Climb the ladder only as far as your use case actually demands — every rung you skip is a cost you never pay.

## 7. The idiomatic way to write one: `#[proxima::piped]`

Sections 2 through 6 all hand-wrote a `struct` and an `impl` block. That is the *complete* contract, but it is boilerplate you rarely want to type by hand — `#[proxima::piped]` generates it from a plain function (`proxima-macros/src/pipe_attr.rs:1–3`):

```
`#[proxima::piped]` — generates a Pipe/SendPipe/UnpinPipe/UnpinSendPipe
impl from a plain function, removing the hand-written unit-struct-plus-impl
boilerplate every leaf pipe otherwise repeats.
```

`examples/runtime_select/main.rs:42–45` uses it — this is the whole pipe, macro and all:

```rust
#[proxima::piped(send)]
async fn select_pipe(_request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    Ok(Response::ok("hello from whichever runtime is listening\n"))
}
```

(`examples/hello/main.rs`'s own handler is deliberately the *other* shape — a bare `async fn` with no macro at all, mounted directly; section 13 shows it and explains why. `#[proxima::piped]` earns its keep when a handler needs to be a *named*, reusable pipe type — `select_pipe` above is exactly that case, since `examples/runtime_select/main.rs` mounts the same `PipeHandle` under two different runtimes.)

This makes `select_pipe` — the function you just wrote — into the pipe itself, giving it the `SendPipe` impl the function's own signature implies: `In = Request<Bytes>` (its one parameter), `Out = Response<Bytes>` and `Err = ProximaError` (from its `Result<Out, Err>` return type). It adds **no** new noun to the pipe algebra — it picks the *downward closure* of tiers from section 6 a given function's shape qualifies for, and writes one impl block per tier in that closure. **Not just one tier**: since the higher tiers are additive constraints on the same root contract, never a replacement for it (section 6), a pipe implements every tier it qualifies for at once. `select_pipe`'s `async fn` plus `#[proxima::piped(send)]` puts it at `Pipe` *and* `SendPipe` both (a plain `fn` with `send` reaches all four — [Foundations, part 2](./01-ergonomics.md) proves this by compiling four separate trait-bound assertions against one macro-generated type). The rule the macro follows to compute that closure, exactly as its own module doc states it (`proxima-macros/src/pipe_attr.rs:9–24`, the closure itself computed by `Tier::plan`):

- **whether the function is `async fn` decides the `Unpin` axis for free**: an `async fn`'s future is a compiler-generated state machine (`!Unpin`), so the macro emits `Pipe` (or `SendPipe`); a plain `fn` gets wrapped in `core::future::ready`, whose future *is* `Unpin` unconditionally and costs nothing, so the macro emits `UnpinPipe` (or `UnpinSendPipe`).
- **`send` is never inferred.** Only writing `#[proxima::piped(send)]` explicitly climbs to `SendPipe`/`UnpinSendPipe`. Nothing about your function's types is inspected to guess whether you "could" be `Send` — climbing a tier is a cost (see section 6) and the macro will not charge it to you without being asked.
- `#[proxima::piped(unpin, boxed)]` is how an `async fn` reaches the `Unpin` tier anyway: it wraps the call in `Box::pin`, which is `Unpin` for any future because a `Box` is a fixed heap address, not a self-referential state machine. That costs one heap allocation per call — and, like `send`, `boxed` is never inferred; you always ask for it explicitly.
- **the generated struct always derives `Clone`.** Every pipe the free-function form generates is a fieldless unit struct — like `Double` in section 2, it holds no data — so cloning it costs nothing: no heap, no allocator, not even a `memcpy` of anything but zero bytes. `#[proxima::piped]` puts `#[derive(::core::clone::Clone)]` on that struct unconditionally (`pipe_attr.rs:600`; the module doc states the reason at `pipe_attr.rs:40–44`), because `Clone` is the one bound a *combinator* — a pipe that wraps another pipe to add behavior like chaining, retrying, or rate-limiting; section 5's `AndThen` is the one you already met — commonly needs on the pipe it wraps. There is no `derive(...)` argument to opt in or out of this — it is always there.
- **the pipe wears the function's name, and the function itself moves aside.** `select_pipe` *is* the pipe now — that is the `select_pipe` in `into_handle(select_pipe)`. There is one consequence worth knowing before it surprises you: a unit struct and a function both live in Rust's *value* namespace, so both cannot be called `select_pipe`. The macro renames your function body out of the way (to `__proxima_pipe_select_pipe`) and the pipe takes the name, so `select_pipe(request)` is no longer a call you can write. If you want the plain function *and* a pipe, name the pipe yourself — `#[proxima::piped(send, name = Greet)]` leaves `select_pipe` callable and makes `Greet` the pipe.

`select_pipe` above is `async fn` plus `send`, so `Tier::plan`'s downward closure (`pipe_attr.rs`) puts it at `Pipe` *and* `SendPipe` — the async body never reaches the `Unpin` tier without the separate `unpin, boxed` opt-in (section 6), so those two are the whole closure here. `SendPipe` is the tier section 6 said an HTTP service needs, since a real server dispatches requests across many worker threads; `Pipe` comes along in the same expansion because `SendPipe` is additive on top of it, never a replacement for it.

**Auto-`Clone` is easiest to see paying for itself in a real example.** `RateLimit<Inner, Extractor, Clk>` (`proxima-primitives/src/pipe/rate_limit.rs`) wraps an inner pipe with a token-bucket admission check; every call clones `self.inner` to move it into the future that actually runs it, so its `SendPipe` impl requires `Inner: SendPipe + Clone + Send + Sync + 'static` (`rate_limit.rs:438–440`). Before this affordance existed, satisfying that bound meant writing the `Clone` derive yourself, alongside the struct and the impl — this is `examples/rate_limit/main.rs` as it read before `#[proxima::piped]` grew this affordance (the shape at commit `9f63d35b`, the base both `main` and this macro's branch built from — `main.rs:78–92` there):

```rust
#[derive(Clone)]
struct Backend;

impl SendPipe for Backend {
    type In = Request<bytes::Bytes>;
    type Out = Response<bytes::Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<bytes::Bytes>,
    ) -> impl Future<Output = Result<Response<bytes::Bytes>, ProximaError>> + Send {
        async { Ok(Response::ok("ok")) }
    }
}
```

The same pipe today (`examples/rate_limit/main.rs:83–88`; its own comment at `79–82` explains the same `name = Backend` point the next paragraph does):

```rust
#[piped(send, name = Backend)]
async fn respond_ok(
    _request: Request<bytes::Bytes>,
) -> Result<Response<bytes::Bytes>, ProximaError> {
    Ok(Response::ok("ok"))
}
```

Same tier (`SendPipe`, from `async fn` plus `send`), same `Clone` bound satisfied — the explicit `#[derive(Clone)]` is gone because the macro writes it unconditionally now, on every generated struct, whether or not the call site in front of you happens to need it yet. `name = Backend` is here for the ordinary reason the naming bullet above already covered: `respond_ok` says what the function does, but `RateLimit::new(Backend, ...)` and every other call site in the file already expect a type named `Backend`, so the function keeps the descriptive name and the generated pipe keeps the name the rest of the example uses.

**A real pipe often needs to hold onto something between calls — a client handle, a connection pool, a counter — and a fieldless struct can't do that.** `#[proxima::piped]` covers this shape too, by accepting a second, different kind of input: not a free function, but a plain `impl Foo { .. }` block naming no trait (Rust calls this an *inherent* impl — methods attached directly to a type, as opposed to `impl SomeTrait for Foo`). The macro tells the two shapes apart by grammar alone, before it inspects anything else — `impl ... { ... }` and `fn ... { ... }` never overlap, so trying the impl-block parse first can never mis-route an ordinary function (`pipe_attr.rs:450–455`):

```rust
pub fn expand(args: TokenStream, item: TokenStream) -> Result<TokenStream, Error> {
    match parse2::<ItemImpl>(item.clone()) {
        Ok(item_impl) => expand_impl_form(args, item_impl),
        Err(_) => expand_fn_form(args, item),
    }
}
```

(This is the macro's own source, not code you ever write — `TokenStream` is the proc-macro crate's name for "a chunk of Rust source not yet parsed," and `ItemImpl` is what a successfully-parsed `impl` block turns into. The only thing to take from it: the impl-block shape is tried first, and it can never accidentally swallow a plain function, so both forms coexist under the one attribute name with no ambiguity.)

For this shape the macro **generates no struct at all** — `Foo` already exists, with whatever fields you gave it, so `Foo` itself is relocated unchanged into `impl #trait for Foo`. The block must hold exactly one method named `call`, taking `&self` (never `&mut self` or `self` by value — a pipe is always called through a shared handle, so a call that needed exclusive access could never be satisfied) and exactly one parameter after it, the same `In`/`Out`/`Err` contract as always, just read off a method signature instead of a free function's (`pipe_attr.rs:637–653, 660–685, 731–736`). Anything else the block carries (a helper method) survives untouched, relocated into a plain leftover `impl Foo { .. }` next to the trait impl.

`examples/proxy/main.rs`'s `ProxyPipe` is exactly this shape — `client: Client` is state the free-function form has nowhere to put. Before (`main.rs:64–80` at `9f63d35b`):

```rust
struct ProxyPipe {
    client: Client,
}

impl SendPipe for ProxyPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let client = self.client.clone();
        async move { SendPipe::call(&client, request).await }
    }
}
```

Today (`examples/proxy/main.rs:53–63`):

```rust
struct ProxyPipe {
    client: Client,
}

#[piped(send)]
impl ProxyPipe {
    async fn call(&self, request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
        let client = self.client.clone();
        SendPipe::call(&client, request).await
    }
}
```

The struct is untouched — it was never the boilerplate. What disappeared is the trait header (`impl SendPipe for ProxyPipe { type In = ..; type Out = ..; type Err = ..; }`) and the `async move { .. }` wrapper the hand-written form needed to turn a plain `Result` into a `Future`: the macro reads `In`/`Out`/`Err` straight off `call`'s own signature, the same way it does for the free-function form, and — because `call` here is `async fn` — passes the relocated body straight through as its own future (wrapped only in `async move { .. }`, `pipe_attr.rs:817`), the same zero-cost RPITIT passthrough section 6 already described for `Pipe`/`SendPipe`.

The sync half of this form has a real, `UnpinPipe`-tier example too, in section 11's gate curriculum: `BackendQueue` holds a `RefCell<VecDeque<u32>>` (state), and its `call` was always hand-written to return `impl Future<..> + Unpin` directly, without `async`/`.await` — the same shape as `Ring`/`RingPop` back in section 6. Before (`examples/gate/main.rs:255–266` at `9f63d35b`):

```rust
impl UnpinPipe for BackendQueue {
    type In = ();
    type Out = (&'static str, u32);
    type Err = Exhausted;

    fn call(&self, (): ()) -> impl Future<Output = Result<Self::Out, Exhausted>> + Unpin {
        match self.items.borrow_mut().pop_front() {
            Some(value) => core::future::ready(Ok((self.label, value))),
            None => core::future::ready(Err(Exhausted)),
        }
    }
}
```

Today (`examples/gate/main.rs:259–267`):

```rust
#[piped]
impl BackendQueue {
    fn call(&self, (): ()) -> impl Future<Output = Result<(&'static str, u32), Exhausted>> + Unpin {
        match self.items.borrow_mut().pop_front() {
            Some(value) => core::future::ready(Ok((self.label, value))),
            None => core::future::ready(Err(Exhausted)),
        }
    }
}
```

No `async` anywhere in either version — this `call` already returns the future it needs to return, so the macro relocates the body exactly as written, with no wrapper at all (`ImplShape::Direct`, `pipe_attr.rs:818`): there is nothing here for the macro to change but the trait header.

**Put the two forms together and the rule is: you almost never write `impl Pipe for X` by hand anymore.** A stateless pipe — no fields, nothing to remember between calls — is a free function under `#[proxima::piped]`. A stateful pipe — a client, a pool, a counter, anything living in `&self` — is an inherent `impl Foo { fn call(..) { .. } }` under the same attribute. Both write the same four-tier impl section 6 describes; neither adds a fifth. What is left to hand-write is the tier-selection logic itself, plus a small, mechanical set of pipes the macro's own rules put out of scope — and that boundary is worth being precise about, because it is not "when the macro feels too heavy," it is a compile error with a name:

- `#[proxima::piped]` refuses a **generic** function or impl outright — `"#[proxima::piped] does not support a generic fn"` / `"...does not support a generic impl"` (`pipe_attr.rs:461–466, 705–709`) — because a macro that reads one concrete `In`/`Out`/`Err` off a signature has nothing to read when the types are still parameters. Every **combinator** in this codebase (a pipe generic over another pipe it wraps) is disqualified by this rule alone: `RateLimit<Inner, Extractor, Clk>` (`rate_limit.rs:113`), `Retry<Inner, Clk>` (`retry.rs:135`), `Delay<Inner, Clk>` (`delay.rs:88`), `Isolate<Inner>` (`isolate.rs:39`), `Diff<Inner>` (`diff.rs:31`), `Transform<Inner, InOp, OutOp>` (`transform.rs:38`), `Validate<Inner, Op>` (`validate.rs:43`) — plus `FanOut<S, Policy>` (`fanout.rs:71`) and `FanIn<S, Strategy, N>` (`fan_in.rs:131`), section 9 and section 10's own primitives. All nine hand-roll their `Pipe`/`SendPipe`/`UnpinPipe` impls, and will keep doing so until some future, generic form of this macro can express "one impl, over a type parameter" — a materially different job than "one impl, over one concrete function or method."
- The algebra-teaching examples this tutorial has already shown you — `Double`/`Always`/`Discard`/`Echo`/`Counter` (sections 2–3), `Ring`/`RingPop` (section 6) — stay hand-written **on purpose**, not because the macro can't reach them. They exist to show you the trait itself, one field at a time, before section 7 existed for you. Teaching the shortcut before the shape it shortcuts would leave you unable to read the four-tier ladder in section 6, or debug what the macro generates when something doesn't type-check.
- Inside `examples/fan_out/main.rs`, the `CapturingSink` *arms* plugged into `FanOut` are themselves ordinary stateful pipes (`examples/fan_out/main.rs:96–106`) — the macro applies to them exactly as it does to `ProxyPipe`. It is `FanOut<S, Policy>` itself, the generic combinator that holds the arms, that stays hand-rolled — for the reason just given. Both things are true in the same file: the leaf pipes are macro-written, the combinator they are wired into is not, because "generic over what it wraps" and "one concrete pipe" are different jobs, and the macro only ever does the second one.

## 8. Let some things through: filter

A **filter** puts a yes/no rule in front of a pipe. Each item is checked; only the approved ones reach the inner pipe, and the rest are dropped — the inner pipe is never even called for a dropped item.

The rule is an ordinary pipe: `In -> Result<In, Err>` — `Ok` admits (the item survives), `Err` rejects (carrying the reason). You compose it in front of the inner pipe with `.and_then(inner)` (section 5): `AndThen`'s own `?` already short-circuits before the inner pipe runs on a first-stage `Err`, so a rejected item never reaches it. This is a real, and honest, change from an older design: filter used to be a bespoke `Filter<Inner, Predicate>` combinator fed by a `Decide<In>::decide(&self, &In) -> bool` seam that threw the item and the rejection reason away, so two more types (`Rejectable`, `OnReject`) had to be grown just to carry them back — that whole apparatus is deleted now; the collapse is exactly this section's `predicate.and_then(inner)` (`proxima-primitives/src/pipe/filter.rs:24–33`, the module's own comment on why). Here is the real rule and the real call, copied from `examples/filter/main.rs:178–193,25–28`:

```rust
// the rule: an ordinary pipe, `In -> Result<In, Err>` — admits by returning
// `Ok`, rejects by returning `Err` with the reason
impl SendPipe for MinAmount {
    type In = Order;
    type Out = Order;
    type Err = Outcome;

    fn call(&self, order: Order) -> impl Future<Output = Result<Order, Outcome>> + Send {
        let admits = order.amount_cents >= self.threshold_cents;
        async move {
            if admits {
                Ok(order)
            } else {
                Err(Outcome::Dropped { id: order.id })
            }
        }
    }
}

let stack = MinAmount { threshold_cents: 2_000 }.and_then(ledger);
```

In that example, five orders go in; the two below the threshold are dropped. The proof that the inner pipe is *never even called* for them is not an assertion in the tutorial — it is asserted in the example itself, against a real counter the inner pipe increments on every call (`examples/filter/main.rs:85–90`):

```rust
assert_eq!(
    calls.load(Ordering::Relaxed),
    processed.len(),
    "the ledger's own call counter proves the gate runs before the inner pipe: \
     dropped orders never increment it"
);
```

Run it: `cargo run --example filter`.

## 9. Send one thing to many: fan-out

**Fan-out** takes one input and delivers a copy to several pipes at once — for example, handle a request *and* send a copy to an audit log. Each downstream pipe (each "arm") is an ordinary sink (section 3): it takes the item and returns `()`. Build it with `FanOut::all_or_nothing(vec![...])` (`proxima-primitives/src/pipe/fanout.rs:111`), from `examples/fan_out/main.rs:29–43`:

```rust
let primary = CapturingSink { label: "primary", log: Arc::clone(&primary_log) };
let mirror = CapturingSink { label: "mirror", log: Arc::clone(&mirror_log) };

let fan = FanOut::all_or_nothing(vec![primary, mirror]);
println!("fanning one request to {} arms", fan.sink_count());

fan.call(Message("checkout order 42".into()))
    .await
    .expect("fan-out delivers to every arm");
```

(`all_or_nothing` names the failure rule: if any arm fails, the whole call fails.) The real transcript, from `cargo run --example fan_out`, proves each arm's copy is independent:

```
fanning one request to 2 arms
primary arm received: ["primary: checkout order 42"]
mirror arm received:  ["mirror: checkout order 42"]
both arms received the one request, independently processed: fan-out proven
```

## 10. Merge many into one: fan-in

**Fan-in** is the opposite of fan-out: many sources merged into one stream. Where fan-out is push (one caller decides everyone gets a copy), fan-in is **pull-based**: each call to the merge scans its sources and returns an item only from a source that *has one ready right now*, skipping the rest. A source with nothing ready this round is skipped, not treated as failed — so one busy source can never starve the others. This is the primitive; keep it distinct from the *strategy* the next paragraph introduces — "many sources into one, taking only what is ready" is the whole idea, and it does not by itself say anything about fairness or order.

`FanIn<S, Strategy, const N: usize>` (`proxima-primitives/src/pipe/fan_in.rs:131`) is itself a pipe: its `In` is `()` (a source — it takes nothing, section 3), its `Out` is whatever the merged sources produce, and its `Err` is `Exhausted` — a marker type meaning "this source will never produce again" (`fan_in.rs:65–67`). This is a real, and honest, change from an older design: fan-in used to speak its own bespoke protocol (`PollSource::poll_next(&mut self, cx) -> Poll<Option<Item>>`) next to the pipe algebra, and that type is now deleted. Termination lives in the `Err` channel instead: a merge call resolves either `Ok(item)` or `Err(Exhausted)`, exactly the same `Result<Out, Err>` shape every other pipe uses — no second, parallel "am I done" protocol to learn.

*Which* ready source wins when more than one is ready is a separate seam, `FanInStrategy` (`fan_in.rs:93–99`) — a one-method trait, `index(&self, step, start, n) -> usize`, that never sees the merged item, only a position in the scan (the pipe-vs-strategy line the trait's own doc comment draws, `fan_in.rs:88–90`: "if the item passes through it, it is a pipe; if it only answers a control question and never sees the item, it is a strategy — a plain function"). `Select` (`fan_in.rs:104–114`) is the built-in implementor, passed in when you build the merge — it is a choice about fairness, not part of what a merge *is*:

```rust
pub enum Select {
    /// Resume the scan past whoever last emitted, so a perpetually-ready source
    /// cannot starve the rest. Fair; no source is preferred.
    RoundRobin,
    /// Always scan from the first source: earlier sources win every tie. This
    /// is also priority order — order the array by priority.
    Fifo,
    /// Always scan from the last source: later sources win every tie.
    Lifo,
}
```

There is deliberately no `Priority` variant — the source code explains why (`fan_in.rs:76–78`): "Priority is `Fifo` over an ordered array: put the sources in the order you want them preferred. That is why there is no `Priority` arm — it would be a second name for a choice you already made when you built the array." `FanIn::new` takes the sources *and* the strategy, always both — a `Strategy: FanInStrategy` is a required argument, not an optional extra (`fan_in.rs:143`, signature `pub fn new(sources: [S; N], strategy: Strategy) -> Self`); the common case still reads exactly as before, `FanIn::new(sources, Select::RoundRobin)`, with `Strategy` inferred as `Select`.

`FanInStrategy` is an *open* trait, not a closed enum welded onto `FanIn` — a deliberate, recent change (`67074baf`, "make the fan-in strategy an open trait"). Before it, `Select` was the only three-variant enum `FanIn` could take, so weighted, least-loaded, or a caller's own fairness rule were simply not expressible without editing this library. Now `Select` is one implementor of the open trait, and `FanIn` itself carries the `Strategy` type parameter shown above — a caller who needs a strategy this library never shipped just implements `FanInStrategy` for their own type and passes it to `FanIn::new` exactly the same way, no library change. Section 14's compile-time proof already reflects this: it is generic over `Strategy: FanInStrategy`, not hard-coded to `Select`.

Here is the real call, from `examples/fan_in/main.rs:30–34`:

```rust
let orders = Upstream::new("orders", [false, true, true], Counter::new(1, 1));
let payments = Upstream::new("payments", [false, true, true], Counter::new(10, 10));
let shipping = Upstream::new("shipping", [false, true], Counter::new(100, 100));

let drained = drain_merged(FanIn::new([orders, payments, shipping], Select::RoundRobin));
```

The real transcript from `cargo run --example fan_in`:

```
fan-in: merge 3 upstreams, pull only the ready
poll 1: nothing ready yet, all live upstreams pending
poll 2: drained ("orders", 1)
poll 3: drained ("payments", 10)
poll 4: drained ("shipping", 100)
poll 5: drained ("orders", 2)
poll 6: drained ("payments", 20)
poll 7: all upstreams drained
drained 5 items total: 2 orders, 2 payments, 1 shipping
```

One more thing worth knowing, because it explains a constraint you will hit if you write your own merged source: `FanIn`'s merged sources must implement `UnpinPipe` (section 6), not plain `Pipe`. That is not an arbitrary restriction — a merge over `N` sources has to hold every source's in-flight call future at once and poll each one in turn (exactly the `merge_in_place` shape section 6 showed you), and doing that with no heap and no `unsafe` is precisely what `UnpinPipe` buys.

## 11. Open or closed: gate

A **gate** is a switch you put in front of a pipe: it is either **armed** (open) or **disarmed** (closed). It controls *readiness* — whether work should flow right now. This is how proxima expresses backpressure and rate-limiting, without baking a special "are you ready?" method into every pipe — `examples/gate/main.rs`'s own module doc says this plainly (`examples/gate/main.rs:4–6`): "`proxima_primitives::pipe::SendPipe` has no such method — every gate shape below is composed from existing primitives instead of being baked into the trait."

You create a gate and its controller together with `AtomicGate::pair(initial_armed)` (`proxima-primitives/src/pipe/demand.rs:56`), and open or close it through the controller. The simplest way to use it is `Demand::new(pipe, gate)` (`demand.rs:100`): while the gate is closed, calls quietly do nothing (the inner pipe is never reached, a no-op `Ok`); while it is open, calls pass straight through. Here is the real setup, copied verbatim from `examples/gate/main.rs:203–210`:

```rust
let calls = Arc::new(AtomicUsize::new(0));
let (gate, controller) = AtomicGate::pair(false);
let production = Demand::new(
    CountingSink {
        calls: Arc::clone(&calls),
    },
    gate,
);
```

The rest of that function (`main.rs:212–234`) calls `SendPipe::call(&production, item)` three times while closed, then `controller.arm()` and three more calls, then `controller.disarm()` and one more call, asserting a call counter after each phase. The real transcript from `cargo run --example gate` (this is one of three shapes the example demonstrates; the transcript below is the middle one, "wait"):

```
wait: dormant while the gate is closed
ungated (AlwaysArmed): 1 dispatched
closed gate: 0 dispatched (dormant, no-op)
armed gate: 3 dispatched (resumed)
disarmed again: 3 dispatched (dormant again)
```

A closed gate can do one of three things, depending on how you wrap it — this list is copied directly from the example's own module doc (`examples/gate/main.rs:8–18`):

1. **SHED** — a decision pipe that reads the gate and answers from it alone: the same decision-pipe shape section 8 taught for `filter` (`In -> Result<In, Err>`), composed the same way, `.and_then(inner)`. Rejects (sheds) the item while closed, admits it while open. This is a real, and honest, change from an older design: the example used to wrap a `Filter` combinator around a separate `Decide` adapter; both are deleted now, for the same reason section 8's `Decide` was — see that section's note.
2. **WAIT** — `Demand`, shown above: the wrapped pipe goes dormant while closed, resumes once armed.
3. **BALANCE** — wrap each source of a `FanIn` (section 10) in a gate: a closed gate makes its source `Pending` for that call, so the round-robin merge simply skips it that pass and drains whichever backend is ready.

All three are a gate composed with an existing primitive — filter, `Demand`, or fan-in — no new machinery. Run all three: `cargo run --example gate`. Later, "rate-limit" is simply a gate that opens and closes on a token budget.

## 12. Wait for a one-time event: signal

A **signal** lets one part of your program wait for a one-time event — "the stream ended," "we are done draining" — without checking in a loop. `Signal::new()` creates one (`proxima-core/src/signal.rs:119`); someone calls `.fire()` once (`signal.rs:139`); anyone `.await`ing `.fired()` (`signal.rs:154`) wakes up. It is *sticky* — `.is_fired()` (`signal.rs:147`) stays `true` forever after — so even a latecomer who checks afterward sees it immediately, with no fresh wait.

`examples/signal/main.rs` shows this driving a real producer/consumer. A consumer task is spawned and yields once before the producer starts (`examples/signal/main.rs:42–43`), so it is genuinely parked, not merely about to run; the task itself does the parking, `.await`ing `signal.fired()` (`main.rs:112–120`):

```rust
async fn consumer_task(signal: Signal, polls: Arc<AtomicUsize>) {
    println!("consumer: parked on signal.fired() (no poll loop, no timeout)");
    CountingFuture {
        inner: signal.fired(),
        polls,
    }
    .await;
    println!("consumer: woken by fire() -> proceeding");
}
```

Meanwhile the producer runs a stream through an **observe** pipe (section 3), then a **filter** (section 8) that recognizes the one terminal item, whose inner pipe calls `signal.fire()` (`main.rs:230–234`) — every other item is dropped by the filter before that inner pipe is ever reached. The example instruments the consumer's `.await` point to count how many times it was actually polled, and asserts it was exactly two — once to park, once to wake (`main.rs:89–93`):

```rust
assert_eq!(
    polls, 2,
    "park (Pending, registers a waker) + wake (Ready) is the whole story; \
     a busy-poll loop would have called poll() far more than twice"
);
```

No poll loop, no timer, no sleep — the waiter is genuinely parked until the fire. Run it: `cargo run --example signal`.

That is the whole algebra: **`and_then` (chain), filter, fan-out, fan-in, gate, signal.** Every project you build later is these connectors around pipes you write. Now we turn a pipe into a web server.

## 13. A pipe that answers web requests, and how to serve it

To answer HTTP, a pipe's input is an HTTP request and its output an HTTP response. `Handler` (`proxima-primitives/src/pipe/handler.rs:73`) is exactly that shape, pinned down: it is a trait, but you never implement it yourself — it is *blanket*-implemented for every `SendPipe<In = Request<Bytes>, Out = Response<Bytes>, Err = ProximaError>` (`handler.rs:75–78`). Section 7's `select_pipe` reaches `Handler` this way directly, since `#[proxima::piped(send)]` already gave it that `SendPipe` impl.

A *bare* `async fn` — no `#[proxima::piped]` at all — reaches the server through an adjacent seam: `App::mount` accepts one directly via `IntoMountTarget<ViaFn>` (`src/app.rs:1286–1294`), which wraps the function in a small private `SendPipe` impl, `FnHandler` (`src/app.rs:1267`, `1269–1284`), before erasing it into a `Handler` the same way `select_pipe` was. This is the shape `hello` uses — mounted and served, this is `examples/hello/main.rs`, in full:

```rust
/// A handler is just an `async fn`: typed request in, typed response out, nothing
/// more. It never touches a socket — the listener owns that; the handler answers.
///
/// No attribute is needed to mount it: `App::mount` takes a bare
/// `async fn(Request<Bytes>) -> Result<Response<Bytes>, ProximaError>` directly.
/// `#[proxima::instrument]` wraps it in a span so every call is traced — one
/// attribute yields trace + metric + log. Reach for `#[proxima::piped]` only when
/// you want a *named*, reusable pipe type instead of a one-off handler.
#[proxima::instrument]
async fn hello(_request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
    Ok(Response::ok("hello, proxima\n"))
}

#[proxima::main]
async fn main() -> Result<(), ProximaError> {
    let bind = SocketAddr::from((Ipv4Addr::LOCALHOST, 8080));

    let app = App::new()?;
    app.mount("/", hello)?;

    let server = app.serve(RunConfig::http(bind)).await?;
    println!("listening on http://{bind}");

    server.run_until_signal().await;
    Ok(())
}
```

Three real pieces, each grounded in source:

- **`App::new()`** (`src/app.rs:253`) builds the app with a runtime already set up — the engine that actually drives async work. Under a bare `#[proxima::main]`, `App::new()` adopts that same runtime rather than building a second, independent one.
- **`app.mount("/", hello)`** (`mount` at `src/app.rs:577`) attaches your handler at a path. `mount` takes anything [`IntoMountTarget`] covers (`src/app.rs:1231`) — a handler-shaped pipe, a bare async fn, a registered pipe name, or an already-built `MountTarget`; `hello` here dispatches through the bare-fn arm just described. Underneath, that arm's own conversion runs `into_handle` (`proxima-primitives/src/pipe/handler.rs:86`), which wraps the erased `FnHandler` into a `PipeHandle`, one uniform type that can hold *any* handler. That is what lets `App` store handlers of different concrete types side by side, and mount several at different paths.
- **`app.serve(RunConfig::http(bind))`** (`serve` at `src/app.rs:829`) spawns the listener and returns a `Server` handle *only once the socket is genuinely accepting* — no polling, no sleeping, no discovering `ECONNREFUSED` the hard way.
- **`server.run_until_signal()`** (`src/server.rs:94`) blocks until SIGINT/SIGTERM (or a `stop()` from any clone), then stops accepting and lets in-flight requests drain. That one line is the entire shutdown story — no `ShutdownBarrier` ceremony to hand-roll.

Run it and, in another shell, curl it. `http1-native` is required, not default — it registers the h1+h2 listener `RunConfig::http` names, over the tokio-free sans-IO h1 driver (`serve_connection`/`serve_h1_connection` in `proxima-http`; `examples/hello/main.rs`'s own module doc explains this). `http1` (which layers the legacy hyper/tokio h1 stack on top of `http1-native`) is not needed for this example — `hello` is tokio-free end to end:

```
$ cargo run --example hello --features http1-native
listening on http://127.0.0.1:8080

$ curl http://127.0.0.1:8080/
hello, proxima
```

Ctrl-c the server (SIGINT) and `run_until_signal` returns: the listener stops, in-flight requests drain, the process exits `0`.

## 14. The algebra is enforced by the compiler, not by us

"Everything is a pipe" is a strong claim, and strong claims rot the moment someone forgets to check them. proxima does not leave that check to a human doing a grep. `proxima-primitives/src/pipe/mod.rs:275–314` has a test-only module, `algebra_claims`, whose entire job is to fail to *compile* — not fail a test, fail to build at all — the moment a primitive this tutorial teaches stops being a pipe. Its own doc comment explains why that is the only check worth trusting (`mod.rs:270–274`):

> "Everything is a pipe" is falsifiable, so falsify it mechanically: each line below fails to compile the moment a primitive we teach stops being a pipe. A grep for `impl .* Pipe for X` cannot answer this — it cannot see through generics, re-exports, macros, or a renamed type parameter, and it has been wrong every time it was asked. rustc is never wrong about it.

Here is the claim for fan-in (section 10) and chaining (section 5), copied verbatim — note the `Strategy: FanInStrategy` bound on the fan-in claim, mirroring section 10's open trait, not the closed `Select` enum an older version of this module checked. (`DropSafe` is a marker trait not otherwise covered by this tutorial — it says a type has no observable state left over if you drop it mid-call; fan-in's sources need it because, as section 10 explained, a source found `Pending` this scan has its transient call future dropped and gets asked fresh next time.)

```rust
fn assert_pipe<P: Pipe>() {}
fn assert_send_pipe<P: SendPipe>() {}

// the chain of two pipes is itself a pipe — the composition law, checked.
fn _a_chain_is_a_pipe<First, Second>()
where
    First: Pipe,
    Second: Pipe<In = First::Out>,
    Second::Err: From<First::Err>,
{
    assert_pipe::<super::primitives::AndThen<First, Second>>();
}

// fan-in IS a pipe, for any DropSafe UnpinPipe source.
fn _fan_in_is_a_pipe<S, Strategy, const N: usize>()
where
    S: super::primitives::UnpinPipe<In = (), Err = super::fan_in::Exhausted>
        + proxima_core::markers::DropSafe,
    Strategy: super::fan_in::FanInStrategy,
{
    assert_pipe::<super::fan_in::FanIn<S, Strategy, N>>();
}
```

This is why this tutorial can cite exact `file:line` locations instead of hedging with "should be" or "is designed to be": every claim in this document that a type is a `Pipe` is, in this codebase, a claim the compiler itself already checks on every build.

## 15. Where to go next

You now know the whole foundation:

- A **pipe** is one async step: `In -> Result<Out, Err>` (`Pipe`, section 2).
- **source / sink / observe / transform** are just choices of `In` and `Out` (section 3).
- `.and_then()` chains two pipes into one `AndThen`, which is itself a pipe (section 5).
- proxima has **four related `Pipe` traits** — `Pipe`, `SendPipe`, `UnpinPipe`, `UnpinSendPipe` — because a compiler limitation (RTN, unstable) forbids deriving one from another; climb only as far as your use case demands (section 6).
- `#[proxima::piped]` writes the boilerplate impl for you, picking a tier from your function's `async`-ness and an explicit `send` opt-in (section 7).
- The **pipe algebra** connects pipes: `.and_then()` (in a row), **filter** (let some through), **fan-out** (one to many), **fan-in** (many to one, pull-based, `Select` decides which ready source wins), **gate** (open/closed readiness), **signal** (a one-time event).
- `into_handle` holds any `SendPipe<Request<Bytes>, Response<Bytes>, ProximaError>` (a `Handler`) behind one uniform type; `App` listens on the network and routes requests to a mounted handle.
- The whole claim — everything here really is a `Pipe` — is checked by the compiler on every build, not by trusting this document (section 14).

Every "Build a …" section in the [index](./README.md) builds a real service out of exactly these pieces, and introduces nothing you have not seen here without teaching it first. Start with [Build an API gateway](./build-an-api-gateway.md).
