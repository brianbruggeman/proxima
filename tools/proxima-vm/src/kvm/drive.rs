use futures::FutureExt;
use proxima_primitives::pipe::Pipe;

use super::error::KvmError;
use super::trap::{End, RunError};

/// The one loop every lane runs: step the machine until a reply ends it. Each
/// stage of `step` completes on its first poll, so a stage that does not is a
/// defect, reported as one.
pub fn drive<Step>(step: &Step) -> Result<End, RunError>
where
    Step: Pipe<In = (), Out = (), Err = RunError>,
{
    loop {
        match step.call(()).now_or_never() {
            Some(Ok(())) => {}
            Some(Err(RunError::Ended(end))) => return Ok(end),
            Some(Err(error)) => return Err(error),
            None => {
                return Err(RunError::Failed(KvmError::guest("a pipe stage was not ready on its first poll")));
            }
        }
    }
}
