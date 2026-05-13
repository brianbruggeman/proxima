use proxima_macros::pipe;

// a fn whose return type isn't Result<Out, Err> must be a compile error.
#[pipe]
fn bad(input: u64) -> u64 {
    input
}

fn main() {}
