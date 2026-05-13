# fallback — try an alternate on failure

When the primary pipe errors, route to a secondary — cache, default, degraded response — instead of giving up.

## Builds on

[retry](../retry/README.md) — fallback is retry's sibling: try an alternate, not a re-run.

## What it demonstrates

`Fallback` (`proxima_primitives::pipe::resilience::fallback`) composes two pipes with the same `In`/`Out`/`Err` types:

```text
match primary.call(input.clone()).await {
    Ok(out) => Ok(out),           // secondary never runs
    Err(_)  => secondary.call(input).await,
}
```

`primary` gets tried first against a clone of the input (`P::In: Clone` is required so the original can be replayed). On success, `secondary` is skipped entirely. On any error, `secondary` gets the original input and its result — success or failure — is what `Fallback` returns.

The example runs the same `Fallback { primary: LiveService, secondary: Cache }` wiring twice, changing only whether `LiveService` is healthy:

1. **primary succeeds** — the live answer wins; `Cache`'s atomic hit counter proves it was never called.
2. **primary fails** — `Cache` serves a degraded-but-present answer, and the counter proves it was called exactly once.

That's graceful degradation as composition: no special-cased "if down, use default" branch in the caller, just two pipes and one combinator.

## Run

```
cargo run --example fallback
```

## What you'll see

```
fallback: try an alternate pipe on failure

-- primary healthy: live answer wins, cache untouched --
  query { id: 7 } -> Response { source: Live, value: 70 }
  cache hits: 0 (secondary untouched)

-- primary down: cache serves a degraded answer --
  query { id: 7 } -> Response { source: Cache, value: 7 }
  cache hits: 1 (secondary served)

same Fallback wiring both times; only LiveService's health changed which pipe answered the query
```

- **primary healthy**: `Response { source: Live, .. }` comes straight from `LiveService`, and `cache_hits` reads `0` — `Cache::call` was never invoked, not just cheaply invoked.
- **primary down**: `Response { source: Cache, .. }` comes from `Cache`, and `cache_hits` reads `1` — proof the failure actually routed to the secondary rather than the call silently erroring out.
- Both runs assert on the returned `Response` and the counter (`assert_eq!`), so a regression in `Fallback`'s routing fails the example, not just the eyeball check.
