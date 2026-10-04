use crate::tcp_listener::Endpoint;
use crate::tcp_stack::{ConnId, Outbound, TcpStack};
use proxima_protocols::tcp::time::Instant;
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};

const FREE: u8 = 0;
const PENDING: u8 = 1;
const CANCELLED: u8 = 2;

// a fixed local/remote tuple has one owner; Drop cannot wait for a driver lock.
pub(crate) struct DialLease {
    peer: ConnId,
    state: AtomicU8,
    cancelled_sequence: AtomicU32,
}

impl DialLease {
    pub(crate) fn new(peer: ConnId) -> Self {
        Self {
            peer,
            state: AtomicU8::new(FREE),
            cancelled_sequence: AtomicU32::new(0),
        }
    }

    pub(crate) fn reclaim(&self, stack: &mut TcpStack) {
        if self.state.load(Ordering::Acquire) == CANCELLED {
            stack.cancel_connect(self.peer, self.cancelled_sequence.load(Ordering::Relaxed));
            self.state.store(FREE, Ordering::Release);
        }
    }

    pub(crate) fn claim(self: &Arc<Self>, stack: &mut TcpStack) -> io::Result<DialGuard> {
        self.reclaim(stack);
        if stack.peer(self.peer).is_some()
            || self
                .state
                .compare_exchange(FREE, PENDING, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            return Err(io::Error::new(
                io::ErrorKind::AddrInUse,
                "TCP tuple already occupied",
            ));
        }
        Ok(DialGuard {
            lease: Arc::clone(self),
            sequence: None,
        })
    }
}

pub(crate) struct DialGuard {
    lease: Arc<DialLease>,
    sequence: Option<u32>,
}

impl DialGuard {
    pub(crate) fn connection(&self) -> Option<ConnId> {
        self.sequence.map(|_| self.lease.peer)
    }

    pub(crate) fn start(
        &mut self,
        stack: &mut TcpStack,
        peer: Endpoint,
    ) -> io::Result<Vec<Outbound>> {
        if stack.peer(self.lease.peer).is_some() {
            return Err(io::Error::new(
                io::ErrorKind::AddrInUse,
                "TCP tuple already occupied",
            ));
        }
        let (_, outbound) = stack.connect(peer);
        self.sequence = Some(outbound[0].1.seq);
        Ok(outbound)
    }

    pub(crate) fn owns(&self, stack: &TcpStack) -> bool {
        self.sequence.is_some_and(|sequence| {
            stack.active_connect_sequence(self.lease.peer) == Some(sequence)
        })
    }

    pub(crate) fn connected(&self, stack: &TcpStack) -> bool {
        self.sequence.is_some_and(|sequence| {
            stack.active_connection_sequence(self.lease.peer) == Some(sequence)
        })
    }

    pub(crate) fn complete(self, stack: &TcpStack) -> io::Result<Self> {
        self.flush(stack)?;
        Ok(self)
    }

    pub(crate) fn read(&self, stack: &mut TcpStack, buffer: &mut [u8]) -> io::Result<usize> {
        self.flush(stack)?;
        Ok(stack.read(self.lease.peer, buffer))
    }

    pub(crate) fn write(
        &self,
        stack: &mut TcpStack,
        buffer: &[u8],
        now: Instant,
    ) -> io::Result<Vec<Outbound>> {
        self.flush(stack)?;
        Ok(stack.write(self.lease.peer, buffer, now))
    }

    pub(crate) fn close(&self, stack: &mut TcpStack) -> io::Result<Vec<Outbound>> {
        self.flush(stack)?;
        Ok(stack.close(self.lease.peer))
    }

    pub(crate) fn flush(&self, stack: &TcpStack) -> io::Result<()> {
        if self.connected(stack) {
            Ok(())
        } else {
            Err(io::Error::from(io::ErrorKind::ConnectionReset))
        }
    }
}

impl Drop for DialGuard {
    fn drop(&mut self) {
        if let Some(sequence) = self.sequence {
            self.lease
                .cancelled_sequence
                .store(sequence, Ordering::Relaxed);
            self.lease.state.store(CANCELLED, Ordering::Release);
        } else {
            self.lease.state.store(FREE, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tcp_listener::Inbound;
    use proxima_protocols::inet::tcp::TcpFlags;
    use proxima_protocols::tcp::time::Instant;

    fn inbound(
        peer: Endpoint,
        sequence: u32,
        acknowledgement: u32,
        flags: TcpFlags,
        payload: &[u8],
    ) -> Inbound<'_> {
        Inbound {
            source_mac: peer.mac,
            source_ip: peer.ip,
            source_port: peer.port,
            flags,
            seq: sequence,
            ack: acknowledgement,
            window: 4096,
            payload,
        }
    }

    fn establish(stack: &mut TcpStack, peer: Endpoint, sequence: u32) {
        let response = stack.on_inbound(
            &inbound(
                peer,
                900,
                sequence.wrapping_add(1),
                TcpFlags {
                    syn: true,
                    ack: true,
                    ..TcpFlags::default()
                },
                &[],
            ),
            Instant::ZERO,
        );
        assert_eq!(response.len(), 1);
        assert_eq!(response[0].1.ack, 901);
    }

    #[test]
    fn fixed_tuple_preserves_payload_and_cancels_only_owned_generation() {
        let peer = Endpoint {
            mac: [2, 0, 0, 0, 0, 2],
            ip: [127, 0, 0, 2],
            port: 4321,
        };
        let identity = (peer.ip, peer.port);
        let mut stack = TcpStack::new([127, 0, 0, 1], 1234, 100);
        let lease = Arc::new(DialLease::new(identity));
        let mut first = lease.claim(&mut stack).expect("claim fixed tuple");
        let outbound = first.start(&mut stack, peer).expect("start owned dial");
        assert_eq!(outbound.len(), 1);
        assert_eq!(outbound[0].1.seq, 100);
        assert_eq!(first.connection(), Some(identity));
        assert!(first.owns(&stack));
        assert!(!first.connected(&stack));
        let rejected = lease
            .claim(&mut stack)
            .err()
            .expect("second call must reject occupied tuple");
        assert_eq!(rejected.kind(), io::ErrorKind::AddrInUse);
        assert_eq!(stack.peer(identity), Some(peer));
        establish(&mut stack, peer, 100);
        assert_eq!(stack.poll_connected(), Some(identity));
        stack.on_inbound(
            &inbound(
                peer,
                901,
                101,
                TcpFlags {
                    ack: true,
                    ..TcpFlags::default()
                },
                b"original",
            ),
            Instant::ZERO,
        );
        let mut payload = [0; 8];
        assert_eq!(
            first
                .read(&mut stack, &mut payload)
                .expect("read owned connection"),
            8
        );
        assert_eq!(&payload, b"original");
        let written = first
            .write(&mut stack, b"owned", Instant::ZERO)
            .expect("write owned connection");
        assert!(
            written
                .iter()
                .any(|(_, segment)| segment.payload == b"owned")
        );
        first.flush(&stack).expect("flush owned connection");
        assert!(
            first
                .close(&mut stack)
                .expect("close owned connection")
                .iter()
                .any(|(_, segment)| segment.flags.fin)
        );
        let first = first
            .complete(&stack)
            .expect("transfer dial ownership into stream");
        lease.reclaim(&mut stack);
        assert_eq!(
            stack.peer(identity),
            Some(peer),
            "delivered connection retains tuple ownership"
        );
        assert_eq!(
            lease
                .claim(&mut stack)
                .err()
                .expect("live tuple remains occupied")
                .kind(),
            io::ErrorKind::AddrInUse
        );
        drop(first);
        lease.reclaim(&mut stack);
        assert_eq!(
            stack.peer(identity),
            None,
            "delivered stream Drop reclaims active state"
        );

        for completed_handshake in [false, true] {
            let mut cancelled = lease.claim(&mut stack).expect("claim reclaimed tuple");
            let outbound = cancelled
                .start(&mut stack, peer)
                .expect("start cancelled dial");
            if completed_handshake {
                establish(&mut stack, peer, outbound[0].1.seq);
            }
            drop(cancelled);
            lease.reclaim(&mut stack);
            assert_eq!(stack.peer(identity), None);
            assert_eq!(
                stack.poll_connected(),
                None,
                "cancelled completion cannot reach another call"
            );
        }

        let unstarted = lease
            .claim(&mut stack)
            .expect("claim before ARP resolution");
        drop(unstarted);
        let mut stale = lease
            .claim(&mut stack)
            .expect("unstarted Drop releases tuple");
        let old = stale
            .start(&mut stack, peer)
            .expect("start stale generation");
        stack.on_inbound(
            &inbound(
                peer,
                900,
                0,
                TcpFlags {
                    rst: true,
                    ..TcpFlags::default()
                },
                &[],
            ),
            Instant::ZERO,
        );
        let (_, replacement) = stack.connect(peer);
        assert_ne!(old[0].1.seq, replacement[0].1.seq);
        drop(stale);
        lease.reclaim(&mut stack);
        assert_eq!(
            stack.peer(identity),
            Some(peer),
            "stale cancellation preserves replacement"
        );
        establish(&mut stack, peer, replacement[0].1.seq);
        assert_eq!(stack.poll_connected(), Some(identity));
        stack.on_inbound(
            &inbound(
                peer,
                901,
                replacement[0].1.seq.wrapping_add(1),
                TcpFlags {
                    ack: true,
                    ..TcpFlags::default()
                },
                b"replaced",
            ),
            Instant::ZERO,
        );
        assert_eq!(stack.read(identity, &mut payload), 8);
        assert_eq!(&payload, b"replaced");

        stack = TcpStack::new([127, 0, 0, 1], 1234, 500);
        let mut stale = lease
            .claim(&mut stack)
            .expect("claim active sequence control");
        let original = stale
            .start(&mut stack, peer)
            .expect("start active sequence control");
        establish(&mut stack, peer, original[0].1.seq);
        assert_eq!(stack.poll_connected(), Some(identity));
        let stale = stale.complete(&stack).expect("retain delivered old handle");
        stack.on_inbound(
            &inbound(
                peer,
                900,
                0,
                TcpFlags {
                    rst: true,
                    ..TcpFlags::default()
                },
                &[],
            ),
            Instant::ZERO,
        );
        assert_eq!(
            lease
                .claim(&mut stack)
                .err()
                .expect("old handle still owns reset tuple")
                .kind(),
            io::ErrorKind::AddrInUse
        );
        // reconstruct the sequence-wrap state directly instead of opening thousands of peers.
        stack = TcpStack::new([127, 0, 0, 1], 1234, original[0].1.seq);
        let passive = stack.on_inbound(
            &inbound(
                peer,
                900,
                0,
                TcpFlags {
                    syn: true,
                    ..TcpFlags::default()
                },
                &[],
            ),
            Instant::ZERO,
        );
        assert_eq!(passive[0].1.seq, original[0].1.seq);
        stack.on_inbound(
            &inbound(
                peer,
                901,
                passive[0].1.seq.wrapping_add(1),
                TcpFlags {
                    ack: true,
                    ..TcpFlags::default()
                },
                b"passive!",
            ),
            Instant::ZERO,
        );
        let mut untouched = [0x7f; 8];
        assert_eq!(
            stale
                .read(&mut stack, &mut untouched)
                .expect_err("old handle cannot read passive replacement")
                .kind(),
            io::ErrorKind::ConnectionReset
        );
        assert_eq!(untouched, [0x7f; 8]);
        assert_eq!(
            stale
                .write(&mut stack, b"corrupt!", Instant::ZERO)
                .expect_err("old handle cannot write replacement")
                .kind(),
            io::ErrorKind::ConnectionReset
        );
        assert_eq!(
            stale
                .close(&mut stack)
                .expect_err("old handle cannot close replacement")
                .kind(),
            io::ErrorKind::ConnectionReset
        );
        assert_eq!(
            stale
                .flush(&stack)
                .expect_err("old handle cannot flush replacement")
                .kind(),
            io::ErrorKind::ConnectionReset
        );
        drop(stale);
        lease.reclaim(&mut stack);
        assert_eq!(stack.poll_accept(), Some(identity));
        assert_eq!(
            stack.peer(identity),
            Some(peer),
            "passive same-sequence replacement survives active cancellation"
        );
        assert_eq!(stack.read(identity, &mut payload), 8);
        assert_eq!(&payload, b"passive!");

        stack = TcpStack::new([127, 0, 0, 1], 1234, 1000);
        let (_, previous) = stack.connect(peer);
        establish(&mut stack, peer, previous[0].1.seq);
        stack.on_inbound(
            &inbound(
                peer,
                900,
                0,
                TcpFlags {
                    rst: true,
                    ..TcpFlags::default()
                },
                &[],
            ),
            Instant::ZERO,
        );
        let mut pending = lease.claim(&mut stack).expect("claim after previous reset");
        let current = pending.start(&mut stack, peer).expect("start current dial");
        assert_eq!(
            stack.poll_connected(),
            Some(identity),
            "observe stale queued notification"
        );
        assert!(pending.owns(&stack));
        assert!(
            !pending.connected(&stack),
            "old notification cannot complete current SYN"
        );
        assert_eq!(
            pending
                .flush(&stack)
                .expect_err("pending is not a stream")
                .kind(),
            io::ErrorKind::ConnectionReset
        );
        establish(&mut stack, peer, current[0].1.seq);
        assert_eq!(stack.poll_connected(), Some(identity));
        let delivered = pending
            .complete(&stack)
            .expect("complete only current handshake");
        drop(delivered);
        lease.reclaim(&mut stack);
        assert_eq!(stack.peer(identity), None);
    }
}
