use proxima_macros::pipe;

// `unpin` on an async closure without `boxed` must be a compile error: its
// body compiles to a compiler-generated state machine, which is `!Unpin`.
fn main() {
    let _bad = pipe!(
        async move |input: u64| -> Result<u64, std::convert::Infallible> { Ok(input) },
        unpin
    );
}
