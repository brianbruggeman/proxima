use proxima_macros::piped;

// an unrecognized #[piped(...)] arg must be a compile error.
#[piped(bogus)]
fn bad(input: u64) -> Result<u64, std::convert::Infallible> {
    Ok(input)
}

fn main() {}
