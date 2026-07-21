use proxima_macros::piped;

// #[piped(unpin)] on an async fn must be a compile error: its future is a
// compiler-generated state machine, which is `!Unpin`.
#[piped(unpin)]
async fn bad(input: u64) -> Result<u64, std::convert::Infallible> {
    Ok(input)
}

fn main() {}
