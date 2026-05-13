use proxima_macros::pipe;

// an unrecognized #[pipe(...)] arg must be a compile error.
#[pipe(bogus)]
fn bad(input: u64) -> Result<u64, std::convert::Infallible> {
    Ok(input)
}

fn main() {}
