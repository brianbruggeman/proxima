use proxima_macros::pipe;

// `boxed` without `unpin` has no effect on an async fn (it only matters
// when climbing to the Unpin tier) — must be a compile error, not a
// silently-ignored flag.
#[pipe(boxed)]
async fn bad(input: u64) -> Result<u64, std::convert::Infallible> {
    Ok(input)
}

fn main() {}
