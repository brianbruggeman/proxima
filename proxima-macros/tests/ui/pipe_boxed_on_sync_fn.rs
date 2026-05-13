use proxima_macros::pipe;

// `boxed` on a plain fn is redundant — core::future::ready is already
// Unpin and allocates nothing — and must be a compile error.
#[pipe(boxed)]
fn bad(input: u64) -> Result<u64, std::convert::Infallible> {
    Ok(input)
}

fn main() {}
