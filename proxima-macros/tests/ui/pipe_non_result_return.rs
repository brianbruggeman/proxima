use proxima_macros::piped;

// a fn whose return type isn't Result<Out, Err> must be a compile error.
#[piped]
fn bad(input: u64) -> u64 {
    input
}

fn main() {}
