use proxima_macros::pipe;

// #[pipe(unpin)] on an async fn must be a compile error: its future is a
// compiler-generated state machine, which is `!Unpin`.
#[pipe(unpin)]
async fn bad(input: u64) -> Result<u64, std::convert::Infallible> {
    Ok(input)
}

fn main() {}
