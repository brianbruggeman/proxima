//! RFC 793 connection control FSM. The transition table mirrors Figure 6;
//! see `docs/tcp-connection-fsm/discipline.md` for the hand-derived worked
//! example that these transitions and their tests are taken from.

/// The eleven RFC 793 connection states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Closed,
    Listen,
    SynSent,
    SynReceived,
    Established,
    FinWait1,
    FinWait2,
    CloseWait,
    Closing,
    LastAck,
    TimeWait,
}

impl State {
    /// A connection is synchronized once both sides have exchanged SYNs; in
    /// these states an incoming RST tears the connection down.
    #[must_use]
    pub const fn is_synchronized(self) -> bool {
        matches!(
            self,
            Self::SynReceived
                | Self::Established
                | Self::FinWait1
                | Self::FinWait2
                | Self::CloseWait
                | Self::Closing
                | Self::LastAck
                | Self::TimeWait
        )
    }
}

/// The control bits of an incoming segment. Sequence numbers, window, and
/// payload are not modeled here — only the flags that drive the FSM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Segment {
    pub syn: bool,
    pub ack: bool,
    pub fin: bool,
    pub rst: bool,
}

/// What drives a transition: a user call, an incoming segment, or the
/// TIME-WAIT 2MSL timer firing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Input {
    OpenActive,
    OpenPassive,
    Close,
    Segment(Segment),
    Timeout,
}

/// The control action the caller must perform after a transition. The FSM
/// names the action; emitting the segment (or notifying the user) is the
/// caller's job — this layer does no I/O.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    None,
    SendSyn,
    SendSynAck,
    SendAck,
    SendFin,
    ConnectionReset,
}

/// A TCP connection's control state. Construct with [`Connection::new`]
/// (starts CLOSED) and advance with [`Connection::step`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Connection {
    state: State,
}

impl Default for Connection {
    fn default() -> Self {
        Self::new()
    }
}

impl Connection {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: State::Closed,
        }
    }

    #[must_use]
    pub const fn state(&self) -> State {
        self.state
    }

    /// Apply one input, move to the next state, and return the action the
    /// caller must take. Inputs with no defined transition leave the state
    /// unchanged and return [`Action::None`].
    pub fn step(&mut self, input: Input) -> Action {
        let (next, action) = transition(self.state, input);
        self.state = next;
        action
    }
}

fn transition(state: State, input: Input) -> (State, Action) {
    if let Input::Segment(segment) = input
        && segment.rst
        && (state.is_synchronized() || state == State::SynSent)
    {
        return (State::Closed, Action::ConnectionReset);
    }

    match (state, input) {
        (State::Closed, Input::OpenPassive) => (State::Listen, Action::None),
        (State::Closed, Input::OpenActive) => (State::SynSent, Action::SendSyn),

        (State::Listen, Input::Segment(seg)) if seg.syn && !seg.ack => {
            (State::SynReceived, Action::SendSynAck)
        }

        (State::SynSent, Input::Segment(seg)) if seg.syn && seg.ack => {
            (State::Established, Action::SendAck)
        }
        // Simultaneous open: a bare SYN crossing our SYN on the wire.
        (State::SynSent, Input::Segment(seg)) if seg.syn => (State::SynReceived, Action::SendAck),

        (State::SynReceived, Input::Segment(seg)) if seg.ack => (State::Established, Action::None),

        (State::Established, Input::Close) => (State::FinWait1, Action::SendFin),
        (State::Established, Input::Segment(seg)) if seg.fin => (State::CloseWait, Action::SendAck),

        // FIN-WAIT-1 collapses three sub-cases off one segment: both bits ->
        // TIME-WAIT, peer-FIN only -> CLOSING, ACK-of-our-FIN only ->
        // FIN-WAIT-2. The combined case must be checked first or the close
        // strands in FIN-WAIT-2.
        (State::FinWait1, Input::Segment(seg)) if seg.fin && seg.ack => {
            (State::TimeWait, Action::SendAck)
        }
        (State::FinWait1, Input::Segment(seg)) if seg.fin => (State::Closing, Action::SendAck),
        (State::FinWait1, Input::Segment(seg)) if seg.ack => (State::FinWait2, Action::None),

        (State::FinWait2, Input::Segment(seg)) if seg.fin => (State::TimeWait, Action::SendAck),

        (State::Closing, Input::Segment(seg)) if seg.ack => (State::TimeWait, Action::None),

        (State::CloseWait, Input::Close) => (State::LastAck, Action::SendFin),

        (State::LastAck, Input::Segment(seg)) if seg.ack => (State::Closed, Action::None),

        (State::TimeWait, Input::Timeout) => (State::Closed, Action::None),

        (other, _) => (other, Action::None),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    const fn seg(syn: bool, ack: bool, fin: bool, rst: bool) -> Input {
        Input::Segment(Segment { syn, ack, fin, rst })
    }

    // Drive a connection from `start` through a script of
    // (input, expected action, expected resulting state), per the discipline
    // log's worked-example table.
    fn drive(start: State, script: &[(Input, Action, State)]) {
        let mut connection = Connection { state: start };
        for (step_index, (input, expected_action, expected_state)) in script.iter().enumerate() {
            let action = connection.step(*input);
            assert_eq!(action, *expected_action, "action at step {step_index}");
            assert_eq!(
                connection.state(),
                *expected_state,
                "state at step {step_index}"
            );
        }
    }

    #[test]
    fn passive_open_handshake() {
        drive(
            State::Closed,
            &[
                (Input::OpenPassive, Action::None, State::Listen),
                (
                    seg(true, false, false, false),
                    Action::SendSynAck,
                    State::SynReceived,
                ),
                (
                    seg(false, true, false, false),
                    Action::None,
                    State::Established,
                ),
            ],
        );
    }

    #[test]
    fn active_open_handshake() {
        drive(
            State::Closed,
            &[
                (Input::OpenActive, Action::SendSyn, State::SynSent),
                (
                    seg(true, true, false, false),
                    Action::SendAck,
                    State::Established,
                ),
            ],
        );
    }

    #[test]
    fn simultaneous_open() {
        drive(
            State::SynSent,
            &[(
                seg(true, false, false, false),
                Action::SendAck,
                State::SynReceived,
            )],
        );
    }

    #[test]
    fn active_close_to_time_wait() {
        drive(
            State::Established,
            &[
                (Input::Close, Action::SendFin, State::FinWait1),
                (
                    seg(false, true, false, false),
                    Action::None,
                    State::FinWait2,
                ),
                (
                    seg(false, false, true, false),
                    Action::SendAck,
                    State::TimeWait,
                ),
                (Input::Timeout, Action::None, State::Closed),
            ],
        );
    }

    #[test]
    fn simultaneous_close() {
        drive(
            State::FinWait1,
            &[
                (
                    seg(false, false, true, false),
                    Action::SendAck,
                    State::Closing,
                ),
                (
                    seg(false, true, false, false),
                    Action::None,
                    State::TimeWait,
                ),
            ],
        );
    }

    #[test]
    fn combined_fin_ack_collapses_to_time_wait() {
        drive(
            State::FinWait1,
            &[(
                seg(false, true, true, false),
                Action::SendAck,
                State::TimeWait,
            )],
        );
    }

    #[test]
    fn passive_close() {
        drive(
            State::Established,
            &[
                (
                    seg(false, false, true, false),
                    Action::SendAck,
                    State::CloseWait,
                ),
                (Input::Close, Action::SendFin, State::LastAck),
                (seg(false, true, false, false), Action::None, State::Closed),
            ],
        );
    }

    #[test]
    fn rst_resets_synchronized_connection() {
        drive(
            State::Established,
            &[(
                seg(false, false, false, true),
                Action::ConnectionReset,
                State::Closed,
            )],
        );
    }

    #[test]
    fn rst_with_ack_resets_syn_sent() {
        drive(
            State::SynSent,
            &[(
                seg(false, true, false, true),
                Action::ConnectionReset,
                State::Closed,
            )],
        );
    }

    #[test]
    fn undefined_input_is_inert() {
        let mut connection = Connection::new();
        assert_eq!(connection.step(Input::Close), Action::None);
        assert_eq!(connection.state(), State::Closed);
    }
}
