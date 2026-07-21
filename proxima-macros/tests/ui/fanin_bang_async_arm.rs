use proxima_macros::fanin;

// fanin! arms must be plain (non-`async`) closures: FanIn's merge loop
// polls each source synchronously in place and requires a genuinely
// `Unpin`, never-suspending future.
fn main() {
    let _bad = fanin!(
        async move |(): ()| -> Result<u8, proxima::pipe::Exhausted> { Ok(1) },
        |(): ()| -> Result<u8, proxima::pipe::Exhausted> { Ok(2) }
    );
}
