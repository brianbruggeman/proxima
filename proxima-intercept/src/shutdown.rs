use std::future::Future;

/// Outcome of racing a graceful drain against shutdown signals. The proxy's
/// first ctrl+c starts the drain; a second ctrl+c (or a SIGTERM) while the
/// drain is still in flight forces an immediate exit so a stuck upstream
/// connection can never trap the operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainOutcome {
    /// The drain finished on its own before any further signal arrived.
    Drained,
    /// A second SIGINT arrived mid-drain — caller should force-exit (130).
    ForcedByInterrupt,
    /// A SIGTERM arrived mid-drain — caller should force-exit (143).
    ForcedByTerminate,
}

/// Race a graceful `drain` against an `interrupt` (second SIGINT) and a
/// `terminate` (SIGTERM) signal. Whichever resolves first decides the
/// outcome. This isolates the decision from the side effects (`process::exit`,
/// real OS signal wiring) so it can be tested deterministically.
pub async fn drain_or_force<Drain, Interrupt, Terminate>(
    drain: Drain,
    interrupt: Interrupt,
    terminate: Terminate,
) -> DrainOutcome
where
    Drain: Future<Output = ()>,
    Interrupt: Future<Output = ()>,
    Terminate: Future<Output = ()>,
{
    // biased: poll drain first so a clean drain that finished in the same tick
    // as a late signal is reported Drained, never misreported as a forced exit.
    // without this, select! starts at a random arm and the outcome is racy.
    tokio::select! {
        biased;
        () = drain => DrainOutcome::Drained,
        () = interrupt => DrainOutcome::ForcedByInterrupt,
        () = terminate => DrainOutcome::ForcedByTerminate,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::future::{pending, ready};

    #[proxima::test]
    async fn drain_completing_first_yields_drained() {
        let outcome = drain_or_force(ready(()), pending::<()>(), pending::<()>()).await;
        assert_eq!(outcome, DrainOutcome::Drained);
    }

    #[proxima::test]
    async fn interrupt_mid_drain_forces_interrupt_exit() {
        // drain never finishes (stuck upstream); a second SIGINT arrives
        let outcome = drain_or_force(pending::<()>(), ready(()), pending::<()>()).await;
        assert_eq!(outcome, DrainOutcome::ForcedByInterrupt);
    }

    #[proxima::test]
    async fn terminate_mid_drain_forces_terminate_exit() {
        let outcome = drain_or_force(pending::<()>(), pending::<()>(), ready(())).await;
        assert_eq!(outcome, DrainOutcome::ForcedByTerminate);
    }

    #[proxima::test]
    async fn drain_wins_when_it_is_ready_alongside_signals() {
        // a completed drain alongside ready signals must still report Drained:
        // tokio::select polls in order and drain is the first arm, so a clean
        // shutdown is never misreported as a forced exit.
        let outcome = drain_or_force(ready(()), ready(()), ready(())).await;
        assert_eq!(outcome, DrainOutcome::Drained);
    }
}
