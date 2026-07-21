use proxima_macros::fanout;

// fanout!()/fanin!() need at least one arm — an empty call has nothing to
// fan anything to.
fn main() {
    let _bad = fanout!();
}
