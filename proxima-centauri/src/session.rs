//! The handshake as a composable pipe.
//!
//! The state machine in [`crate::handshake`] is sans-IO: it takes bytes and
//! returns [`Progress`]. That makes it *drivable*, but not yet *composable* —
//! a caller still has to hand-write the loop that feeds it, draws its entropy,
//! and reads its output. [`Session`] is that loop expressed once, as a
//! `Pipe`, so a handshake drops into a pipeline beside every other proxima
//! stage instead of sitting outside the algebra.
//!
//! ```text
//! Pipe<In = (), Out = Progress, Err = CentauriError>
//! ```
//!
//! `In = ()` rather than `In = &[u8]` because bytes arrive before the step
//! runs: [`Session::feed`] stages them and the pipe polls, exactly the
//! `feed_bytes` / `poll` split `proxima-protocols`' HTTP/1 connection uses.
//! It also keeps the associated type free of a lifetime, which is what lets
//! the session compose with stages that know nothing about it.
//!
//! Entropy composes the same way: the session holds a reference to an entropy
//! *pipe* and draws through it, so a scripted source, a DRBG, and a cell are
//! interchangeable here for the same reason they are interchangeable anywhere
//! else.

use core::cell::{Cell, RefCell};
use core::future::Future;

use proxima_clock::ticks::Ticks;
use proxima_primitives::pipe::Pipe;

use crate::entropy::Entropy32;
use crate::error::CentauriError;
use crate::handshake::{Handshake, OUTBOUND_LEN, Progress};

/// Bytes a session will stage before a step consumes them. Sized to the
/// largest message the handshake defines, because a peer that sends more than
/// one message ahead of a step is not following the protocol.
pub const INBOUND_LEN: usize = OUTBOUND_LEN;

/// A handshake, its entropy source, and its clock, composed as one pipe.
///
/// Holds the state machine behind a [`RefCell`] because [`Pipe::call`] takes
/// `&self` while a step takes `&mut`. That is a single-threaded borrow, not a
/// lock: a session belongs to one connection and one task, so there is no
/// contention to arbitrate — and `RefCell` is `core`, so this still compiles
/// with no allocator and no operating system.
pub struct Session<Source> {
    handshake: RefCell<Handshake>,
    /// Held by value, because `Pipe` is implemented for `&CounterDrbg` rather
    /// than `CounterDrbg` — the source type *is* the reference, which is what
    /// keeps this struct free of a second lifetime.
    entropy: Source,
    now: Cell<Ticks>,
    inbound: RefCell<[u8; INBOUND_LEN]>,
    inbound_len: Cell<usize>,
}

impl<Source> Session<Source>
where
    Source: Pipe<In = (), Out = Entropy32, Err = CentauriError>,
{
    /// Wrap a handshake with the source it draws from.
    pub fn new(handshake: Handshake, entropy: Source) -> Self {
        Self {
            handshake: RefCell::new(handshake),
            entropy,
            now: Cell::new(Ticks::from_raw(0)),
            inbound: RefCell::new([0u8; INBOUND_LEN]),
            inbound_len: Cell::new(0),
        }
    }

    /// Set the time the next step is stamped with.
    ///
    /// A [`Ticks`] rather than a clock handle: the session does not read time
    /// any more than the state machine does. A driver composing a tick source
    /// pipe calls this from that source's output.
    pub fn set_now(&self, now: Ticks) {
        self.now.set(now);
    }

    /// Stage bytes for the next step.
    ///
    /// # Errors
    ///
    /// [`CentauriError::BufferTooSmall`] if more arrives than the largest
    /// message the protocol defines — which means the peer is not speaking it.
    pub fn feed(&self, bytes: &[u8]) -> Result<(), CentauriError> {
        let staged = self.inbound_len.get();
        // checked: a caller handing an enormous slice would otherwise wrap in
        // release, pass the bound below, and panic inside copy_from_slice —
        // a panic on attacker-shaped input is a denial of service
        let end = staged
            .checked_add(bytes.len())
            .ok_or(CentauriError::BufferTooSmall {
                needed: usize::MAX,
                available: INBOUND_LEN,
            })?;
        if end > INBOUND_LEN {
            return Err(CentauriError::BufferTooSmall {
                needed: end,
                available: INBOUND_LEN,
            });
        }

        self.inbound.borrow_mut()[staged..end].copy_from_slice(bytes);
        self.inbound_len.set(end);

        Ok(())
    }

    /// Bytes the last step staged for the peer, handed to a closure so the
    /// borrow cannot outlive it.
    pub fn outbound<Out>(&self, read: impl FnOnce(&[u8]) -> Out) -> Out {
        read(self.handshake.borrow().outbound())
    }

    /// Whether the handshake has finished and keys are available.
    pub fn is_established(&self) -> bool {
        self.handshake.borrow().keys().is_some()
    }

    /// Take the handshake back once the pipeline is done with it.
    pub fn into_handshake(self) -> Handshake {
        self.handshake.into_inner()
    }

    /// Advance once, drawing entropy through the source pipe if the step needs
    /// it.
    async fn advance(&self) -> Result<Progress, CentauriError> {
        let staged = self.inbound_len.get();

        // the staged bytes decide whether a draw is consumed, so the borrow
        // has to end before the await: a RefCell guard is not held across one
        let needs_entropy = {
            let inbound = self.inbound.borrow();
            self.handshake.borrow().needs_entropy(&inbound[..staged])
        };
        let entropy = if needs_entropy {
            Some(self.entropy.call(()).await?)
        } else {
            None
        };

        let inbound = self.inbound.borrow();
        let progress =
            self.handshake
                .borrow_mut()
                .step(&inbound[..staged], entropy, self.now.get())?;
        drop(inbound);

        // a step that consumed the message clears the stage; NeedInput leaves
        // it so the next feed appends rather than overwrites
        if progress != Progress::NeedInput {
            self.inbound_len.set(0);
        }

        Ok(progress)
    }
}

impl<Source> Pipe for &Session<Source>
where
    Source: Pipe<In = (), Out = Entropy32, Err = CentauriError>,
{
    type In = ();
    type Out = Progress;
    type Err = CentauriError;

    fn call(&self, (): ()) -> impl Future<Output = Result<Progress, CentauriError>> {
        self.advance()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use proxima_clock::ticks::Ticks;
    use proxima_primitives::pipe::Pipe;

    use super::{INBOUND_LEN, Session};
    use crate::entropy::{CounterDrbg, Entropy32, EntropyCell, FixedSequence};
    use crate::error::CentauriError;
    use crate::handshake::{Handshake, IkeSpi, Progress};

    const PSK: [u8; 32] = [0xAB; 32];

    fn block_on<Fut: core::future::Future>(future: Fut) -> Fut::Output {
        let mut pinned = core::pin::pin!(future);
        let mut context = core::task::Context::from_waker(core::task::Waker::noop());
        loop {
            if let core::task::Poll::Ready(output) = pinned.as_mut().poll(&mut context) {
                return output;
            }
        }
    }

    #[test]
    fn a_whole_handshake_runs_through_two_composed_pipes() {
        let initiator_entropy = CounterDrbg::new([0x11; 32]);
        let responder_entropy = CounterDrbg::new([0x22; 32]);

        let initiator = Session::new(
            Handshake::initiator(PSK, IkeSpi::new(1)),
            &initiator_entropy,
        );
        let responder = Session::new(
            Handshake::responder(PSK, IkeSpi::new(2)),
            &responder_entropy,
        );
        initiator.set_now(Ticks::from_raw(1_000));
        responder.set_now(Ticks::from_raw(1_000));

        // driving is now `call(())` on each side, with bytes moved between —
        // no hand-written step loop
        assert_eq!(block_on((&initiator).call(())).unwrap(), Progress::Advanced);
        let init = initiator.outbound(|bytes| {
            let mut copy = [0u8; INBOUND_LEN];
            copy[..bytes.len()].copy_from_slice(bytes);
            (copy, bytes.len())
        });

        responder.feed(&init.0[..init.1]).unwrap();
        assert_eq!(
            block_on((&responder).call(())).unwrap(),
            Progress::Established
        );
        let reply = responder.outbound(|bytes| {
            let mut copy = [0u8; INBOUND_LEN];
            copy[..bytes.len()].copy_from_slice(bytes);
            (copy, bytes.len())
        });

        initiator.feed(&reply.0[..reply.1]).unwrap();
        assert_eq!(
            block_on((&initiator).call(())).unwrap(),
            Progress::Established
        );

        assert!(initiator.is_established());
        assert!(responder.is_established());

        let initiator_keys = initiator.into_handshake();
        let responder_keys = responder.into_handshake();
        assert_eq!(
            initiator_keys.keys().unwrap().encrypt_key(),
            responder_keys.keys().unwrap().decrypt_key(),
            "composing through pipes must reach the same keys as driving by hand"
        );
    }

    #[test]
    fn a_partial_feed_leaves_the_stage_for_the_next_one() {
        let entropy = CounterDrbg::new([0x11; 32]);
        let initiator_entropy = CounterDrbg::new([0x33; 32]);
        let initiator = Session::new(
            Handshake::initiator(PSK, IkeSpi::new(1)),
            &initiator_entropy,
        );
        let _ = block_on((&initiator).call(())).unwrap();
        let init = initiator.outbound(|bytes| {
            let mut copy = [0u8; INBOUND_LEN];
            copy[..bytes.len()].copy_from_slice(bytes);
            (copy, bytes.len())
        });
        let init = &init.0[..init.1];

        let responder = Session::new(Handshake::responder(PSK, IkeSpi::new(2)), &entropy);

        // half a message: the pipe reports NeedInput and keeps what it has
        responder.feed(&init[..40]).unwrap();
        assert_eq!(
            block_on((&responder).call(())).unwrap(),
            Progress::NeedInput
        );

        // the rest appends rather than replacing
        responder.feed(&init[40..]).unwrap();
        assert_eq!(
            block_on((&responder).call(())).unwrap(),
            Progress::Established
        );
    }

    #[test]
    fn the_entropy_source_is_interchangeable_under_the_pipe() {
        // the whole point of the algebra: a scripted source drops in where a
        // DRBG was, and the session cannot tell
        let script = [[0x11u8; 32]];
        let scripted = FixedSequence::new(&script);
        let session = Session::new(Handshake::initiator(PSK, IkeSpi::new(1)), &scripted);

        assert_eq!(block_on((&session).call(())).unwrap(), Progress::Advanced);

        let scripted_bytes = session.outbound(|bytes| {
            let mut copy = [0u8; INBOUND_LEN];
            copy[..bytes.len()].copy_from_slice(bytes);
            (copy, bytes.len())
        });

        // the same seed by hand must produce the same message
        let mut by_hand = Handshake::initiator(PSK, IkeSpi::new(1));
        let _ = by_hand
            .step(&[], Some(Entropy32::new([0x11; 32])), Ticks::from_raw(0))
            .unwrap();

        assert_eq!(&scripted_bytes.0[..scripted_bytes.1], by_hand.outbound());
    }

    #[test]
    fn feeding_more_than_a_message_is_refused() {
        let entropy = CounterDrbg::new([0x11; 32]);
        let session = Session::new(Handshake::responder(PSK, IkeSpi::new(2)), &entropy);
        let flood = [0u8; super::INBOUND_LEN + 1];

        assert!(
            matches!(
                session.feed(&flood),
                Err(CentauriError::BufferTooSmall { .. })
            ),
            "a peer sending more than one message ahead is not speaking the protocol"
        );
    }

    /// Two peers all the way through SA_INIT and mutual AUTH, driven by hand
    /// so the session tests can start from a live, authenticated SA.
    fn authenticated_pair() -> (Handshake, Handshake) {
        let mut initiator = Handshake::initiator(PSK, IkeSpi::new(1))
            .with_identity(b"peer-a")
            .expect("identity fits");
        let mut responder = Handshake::responder(PSK, IkeSpi::new(2))
            .with_identity(b"peer-b")
            .expect("identity fits");
        let now = Ticks::from_raw(1_000);

        let mut relay = [0u8; INBOUND_LEN];
        let carry = |from: &Handshake, buffer: &mut [u8; INBOUND_LEN]| {
            let out = from.outbound();
            buffer[..out.len()].copy_from_slice(out);
            out.len()
        };

        let _ = initiator
            .step(&[], Some(Entropy32::new([0x11; 32])), now)
            .unwrap();
        let len = carry(&initiator, &mut relay);
        let _ = responder
            .step(&relay[..len], Some(Entropy32::new([0x22; 32])), now)
            .unwrap();
        let len = carry(&responder, &mut relay);
        let _ = initiator.step(&relay[..len], None, now).unwrap();

        let _ = initiator.send_auth().unwrap();
        let len = carry(&initiator, &mut relay);
        let _ = responder.step(&relay[..len], None, now).unwrap();
        let _ = responder.send_auth().unwrap();
        let len = carry(&responder, &mut relay);
        let _ = initiator.step(&relay[..len], None, now).unwrap();

        (initiator, responder)
    }

    #[test]
    fn a_session_can_answer_an_inbound_rekey() {
        // Regression for a defect the pipe form made unreachable rather than
        // merely awkward: `needs_entropy` answered `false` for an
        // authenticated SA, so the session fed `None` to the one step that
        // draws and every inbound rekey failed with EntropyUnavailable. A
        // take-once cell is the strict source here on purpose — it fails both
        // ways, on a draw that was not needed and on one that was skipped.
        let (mut initiator, responder) = authenticated_pair();
        let before = *responder.keys().unwrap().encrypt_key();

        let cell = EntropyCell::new();
        let session = Session::new(responder, &cell);
        session.set_now(Ticks::from_raw(2_000));

        // an idle poll must not touch the cell
        assert_eq!(
            block_on((&session).call(())).unwrap(),
            Progress::NeedInput,
            "nothing staged, so nothing to do"
        );
        assert!(!cell.is_full(), "and the empty cell was never asked");

        let _ = initiator
            .send_rekey(Some(Entropy32::new([0x31; 32])))
            .unwrap();
        session.feed(initiator.outbound()).unwrap();
        cell.set([0x32; 32]).expect("a fresh cell is empty");

        assert_eq!(block_on((&session).call(())).unwrap(), Progress::Rekeyed);
        assert!(!cell.is_full(), "the draw was claimed");

        let responder = session.into_handshake();
        assert_ne!(
            responder.keys().unwrap().encrypt_key(),
            &before,
            "a rekey that does not change the keys is not a rekey"
        );
    }

    #[test]
    fn an_exhausted_source_surfaces_through_the_pipe() {
        // an empty script: the draw fails, and the failure is the pipe's error
        let empty: [[u8; 32]; 0] = [];
        let exhausted = FixedSequence::new(&empty);
        let session = Session::new(Handshake::initiator(PSK, IkeSpi::new(1)), &exhausted);

        assert!(matches!(
            block_on((&session).call(())),
            Err(CentauriError::EntropyExhausted { .. })
        ));
    }
}
