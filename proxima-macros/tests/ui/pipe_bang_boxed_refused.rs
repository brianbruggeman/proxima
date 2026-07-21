use proxima_macros::pipe;

// `boxed` is never accepted by `pipe!` — this bridge is zero-box by
// construction, unlike `#[proxima::piped]`'s opt-in `unpin, boxed` escape
// hatch. Must be a compile error, not a silently-ignored arg.
fn main() {
    let _bad = pipe!(
        |input: u64| -> Result<u64, std::convert::Infallible> { Ok(input) },
        boxed
    );
}
