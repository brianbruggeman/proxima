use std::env;
use std::error::Error;
use std::io::{self, Write};

use futures::executor::block_on;
use proxima_primitives::pipe::Request;
use proxima_primitives::pipe::SendPipe;
use proxima_vm::ScratchVm;

fn main() -> Result<(), Box<dyn Error>> {
    let guest = match env::args().nth(1) {
        Some(message) => ScratchVm::new(message),
        None => ScratchVm::hello(),
    };
    let request = Request::builder().method("BOOT").path("/scratch").build()?;
    let response = block_on(SendPipe::call(&guest, request))?;
    let bytes = block_on(response.collect_body())?;
    io::stdout().write_all(&bytes)?;
    Ok(())
}
