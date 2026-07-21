use proxima_macros::pipe;

// `send` on an async closure must be a compile error: proving the closure's
// own returned future is `Send` requires naming `AsyncFnMut::CallRefFuture`,
// which is unstable on stable Rust.
fn main() {
    let _bad = pipe!(
        async move |input: u64| -> Result<u64, std::convert::Infallible> { Ok(input) },
        send
    );
}
