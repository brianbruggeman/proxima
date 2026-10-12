use std::os::windows::io::{
    AsHandle, AsRawSocket, AsSocket, BorrowedHandle, OwnedHandle, OwnedSocket,
};

use windows_sys::Win32::System::IO::{OVERLAPPED, OVERLAPPED_ENTRY};

use super::{
    Arc, AtomicBool, Duration, Interest, Ordering, SourceKey, Waker, Wakeup, WakeupInner, io,
};
use crate::core::sized;
use crate::os::afd::{Afd, AfdOutcome, AfdSlot};
use crate::os::iocp::{CompletionPort, WAKE_KEY};
use crate::os::waitpacket;

const AFD_KEY: usize = WAKE_KEY - 1;
const SHUTDOWN_DRAIN: Duration = Duration::from_millis(50);

// a handle slot's completion key is generation << INDEX_BITS | index, so a packet queued for a closed
// registration never matches the slot that later reuses its index. 32-bit targets split the word
// 16/16, which caps the slab at SLAB_CAP handle-capable slots and compares a 16-bit generation.
#[cfg(target_pointer_width = "64")]
const INDEX_BITS: u32 = 32;
#[cfg(not(target_pointer_width = "64"))]
const INDEX_BITS: u32 = 16;
const INDEX_MASK: usize = (1 << INDEX_BITS) - 1;
const SLAB_CAP: usize = INDEX_MASK - 1;

fn pack_key(key: SourceKey) -> usize {
    ((key.generation as usize) << INDEX_BITS) | key.index as usize
}

/// what a non-socket slot owns until deregistration; sockets use `socket` and `poll` instead.
enum Handle {
    None,
    Completion {
        handle: OwnedHandle,
        last: Option<OVERLAPPED_ENTRY>,
    },
    Signaled {
        target: OwnedHandle,
        packet: OwnedHandle,
        armed: bool,
    },
}

struct SourceSlot {
    socket: Option<OwnedSocket>,
    handle: Handle,
    // boxed: the kernel writes the poll in place, so its address must survive slab growth
    poll: Option<Box<AfdSlot>>,
    polled: Option<Interest>,
    generation: u32,
    read_waker: Option<Waker>,
    write_waker: Option<Waker>,
    read_ready_epoch: u32,
    write_ready_epoch: u32,
    read_armed: bool,
    write_armed: bool,
    next_free: Option<u32>,
}

impl SourceSlot {
    fn new(socket: OwnedSocket, generation: u32, interest: Interest) -> Self {
        Self {
            socket: Some(socket),
            poll: Some(Box::new(AfdSlot::new())),
            ..Self::with_handle(Handle::None, generation, interest)
        }
    }

    fn with_handle(handle: Handle, generation: u32, interest: Interest) -> Self {
        Self {
            socket: None,
            handle,
            poll: None,
            polled: None,
            generation,
            read_waker: None,
            write_waker: None,
            read_ready_epoch: 0,
            write_ready_epoch: 0,
            read_armed: interest.wants_read(),
            write_armed: interest.wants_write(),
            next_free: None,
        }
    }

    fn is_live(&self, key: SourceKey) -> bool {
        self.generation == key.generation && self.generation != 0
    }

    fn vacate(&mut self) {
        self.cancel_wait();
        self.socket = None;
        self.handle = Handle::None;
        self.generation = 0;
        self.read_waker = None;
        self.write_waker = None;
        self.read_armed = false;
        self.write_armed = false;
    }

    /// a queued wait packet would otherwise reach whichever slot reuses this index.
    fn cancel_wait(&mut self) {
        if let Handle::Signaled { packet, armed, .. } = &mut self.handle {
            let _ = waitpacket::cancel(packet.as_handle());
            *armed = false;
        }
    }

    fn bound_handle(&self) -> Option<BorrowedHandle<'_>> {
        match &self.handle {
            Handle::None => None,
            Handle::Completion { handle, .. } => Some(handle.as_handle()),
            Handle::Signaled { target, .. } => Some(target.as_handle()),
        }
    }

    fn complete_handle(&mut self, entry: &OVERLAPPED_ENTRY) -> usize {
        match &mut self.handle {
            Handle::None => 0,
            Handle::Completion { last, .. } => {
                *last = Some(*entry);
                self.fire_read(true)
            }
            Handle::Signaled { armed, .. } => match std::mem::take(armed) {
                true => self.fire_read(true),
                false => 0,
            },
        }
    }

    fn owns(&self, overlapped: *mut OVERLAPPED) -> bool {
        self.polled.is_some()
            && self
                .poll
                .as_ref()
                .is_some_and(|poll| poll.overlapped() == overlapped)
    }

    fn start_poll(&mut self, afd: &Afd, wanted: Interest, index: usize) -> io::Result<()> {
        let (Some(socket), Some(poll)) = (self.socket.as_ref(), self.poll.as_mut()) else {
            return Err(stale_source());
        };
        // SAFETY: `polled` is None, so the previous completion was dequeued; the boxed slot
        // stays put until `Sources::complete` dequeues its packet.
        unsafe { afd.poll(poll, socket.as_raw_socket(), wanted, index) }?;
        self.polled = Some(wanted);
        Ok(())
    }

    fn cancel_poll(&self, afd: &Afd) -> io::Result<()> {
        match self.poll.as_ref() {
            // SAFETY: the slot is the one armed through this device and is still boxed here.
            Some(poll) => unsafe { afd.cancel(poll) },
            None => Ok(()),
        }
    }

    fn deliver(&mut self, outcome: AfdOutcome) -> usize {
        let (readable, writable) = match outcome {
            AfdOutcome::Ready { readable, writable } => (readable, writable),
            AfdOutcome::LocalClose | AfdOutcome::Failed(_) => (true, true),
            AfdOutcome::Cancelled => (false, false),
        };
        self.fire_read(readable) + self.fire_write(writable)
    }

    fn fire_read(&mut self, signaled: bool) -> usize {
        if !(signaled && self.read_armed) {
            return 0;
        }
        self.read_armed = false;
        self.read_ready_epoch = self.read_ready_epoch.wrapping_add(1);
        if let Some(waker) = &self.read_waker {
            waker.wake_by_ref();
        }
        1
    }

    fn fire_write(&mut self, signaled: bool) -> usize {
        if !(signaled && self.write_armed) {
            return 0;
        }
        self.write_armed = false;
        self.write_ready_epoch = self.write_ready_epoch.wrapping_add(1);
        if let Some(waker) = &self.write_waker {
            waker.wake_by_ref();
        }
        1
    }
}

/// the source slab and the AFD device its polls run through.
struct Sources {
    afd: Afd,
    slab: Vec<SourceSlot>,
    free_head: Option<u32>,
    live_sources: usize,
}

impl Sources {
    fn live(&self, key: SourceKey) -> Option<&SourceSlot> {
        self.slab
            .get(key.index as usize)
            .filter(|slot| slot.is_live(key))
    }

    fn live_mut(&mut self, key: SourceKey) -> io::Result<&mut SourceSlot> {
        self.slab
            .get_mut(key.index as usize)
            .filter(|slot| slot.is_live(key))
            .ok_or_else(stale_source)
    }

    fn insert(&mut self, slot: SourceSlot) -> io::Result<SourceKey> {
        let generation = slot.generation;
        let index = match self.free_head {
            Some(index) => index,
            None => u32::try_from(self.slab.len())
                .ok()
                .filter(|index| (*index as usize) < SLAB_CAP)
                .ok_or_else(|| io::Error::other("reactor source capacity exhausted"))?,
        };
        let key = SourceKey { index, generation };
        match self.free_head {
            Some(_) => {
                let reused = &mut self.slab[index as usize];
                self.free_head = reused.next_free;
                *reused = slot;
            }
            None => self.slab.push(slot),
        }
        self.live_sources += 1;
        if let Err(error) = self.sync_poll(index as usize) {
            self.deregister(key)?;
            return Err(error);
        }
        Ok(key)
    }

    /// issue a poll for the armed directions unless one already covers them. a pending poll
    /// that is too narrow is cancelled and its completion re-issues the wider one.
    fn sync_poll(&mut self, index: usize) -> io::Result<()> {
        let slot = self.slab.get_mut(index).ok_or_else(stale_source)?;
        if slot.socket.is_none() {
            return Ok(());
        }
        let Some(wanted) = interest_of(slot.read_armed, slot.write_armed) else {
            return Ok(());
        };
        match slot.polled {
            Some(polled) if covers(polled, wanted) => Ok(()),
            Some(_) => slot.cancel_poll(&self.afd),
            None => slot.start_poll(&self.afd, wanted, index),
        }
    }

    fn deregister(&mut self, key: SourceKey) -> io::Result<()> {
        let Some(slot) = self
            .slab
            .get_mut(key.index as usize)
            .filter(|slot| slot.is_live(key))
        else {
            return Ok(());
        };
        let cancelled = slot.cancel_poll(&self.afd);
        slot.vacate();
        // a cancelled poll still owns this index until its packet is dequeued
        if slot.polled.is_none() {
            slot.next_free = self.free_head.replace(key.index);
        }
        self.live_sources -= 1;
        cancelled
    }

    fn complete(&mut self, entry: &OVERLAPPED_ENTRY) -> usize {
        match entry.lpCompletionKey {
            AFD_KEY => self.complete_poll(entry.lpOverlapped),
            WAKE_KEY => 0,
            _ => self.complete_handle(entry),
        }
    }

    fn complete_handle(&mut self, entry: &OVERLAPPED_ENTRY) -> usize {
        let (index, generation) = (
            entry.lpCompletionKey & INDEX_MASK,
            entry.lpCompletionKey >> INDEX_BITS,
        );
        self.slab
            .get_mut(index)
            .filter(|slot| {
                slot.generation != 0
                    && slot.generation as usize & (usize::MAX >> INDEX_BITS) == generation
            })
            .map_or(0, |slot| slot.complete_handle(entry))
    }

    fn complete_poll(&mut self, overlapped: *mut OVERLAPPED) -> usize {
        if overlapped.is_null() {
            return 0;
        }
        // SAFETY: an AFD_KEY packet comes only from a poll on this device, and every such slot
        // stays boxed in the slab until its packet is dequeued, so the address is a live `AfdSlot`.
        let index = unsafe { (*overlapped.cast::<AfdSlot>()).key() };
        let Some(slot) = self
            .slab
            .get_mut(index)
            .filter(|slot| slot.owns(overlapped))
        else {
            return 0;
        };
        slot.polled = None;
        if slot.generation == 0 {
            slot.next_free = self.free_head.replace(index as u32);
            return 0;
        }
        let outcome = slot.poll.as_ref().map(|poll| poll.outcome());
        let fired = outcome.map_or(0, |outcome| slot.deliver(outcome));
        match interest_of(slot.read_armed, slot.write_armed) {
            Some(wanted) if slot.start_poll(&self.afd, wanted, index).is_err() => {
                // waking both directions lets the next rearm re-issue the poll and report the error
                fired + slot.fire_read(true) + slot.fire_write(true)
            }
            _ => fired,
        }
    }

    fn in_flight(&self) -> bool {
        self.slab.iter().any(|slot| slot.polled.is_some())
    }

    fn cancel_all(&mut self) {
        for slot in &mut self.slab {
            let _ = slot.cancel_poll(&self.afd);
            slot.vacate();
        }
    }

    fn leak_in_flight(&mut self) {
        for slot in &mut self.slab {
            if slot.polled.is_some() {
                // no packet arrived, so the kernel may still write into the slot; leaking is the only safe release
                std::mem::forget(slot.poll.take());
            }
        }
    }
}

/// owns the completion port, the AFD socket polls, and the source slab.
pub struct Reactor {
    port: CompletionPort,
    sources: Sources,
    next_generation: u32,
    pub(super) wakeup: Wakeup,
}

// SAFETY: the reactor is single-thread-owned and moves to its worker once at launch. The raw
// pointers inside are kernel-visible poll buffers, boxed and owned by `Sources`, and handles
// the reactor alone uses.
unsafe impl Send for Reactor {}

impl Reactor {
    pub fn new() -> io::Result<Self> {
        let port = CompletionPort::new(sized::REACTOR_EVENT_BATCH)?;
        let afd = Afd::new()?;
        port.associate(afd.as_handle(), AFD_KEY)?;
        let wakeup = Wakeup {
            inner: Arc::new(WakeupInner {
                needs_wake: AtomicBool::new(false),
                alive: AtomicBool::new(true),
                cancellations: super::SegQueue::new(),
                port: port.shared_handle(),
            }),
        };
        Ok(Self {
            port,
            sources: Sources {
                afd,
                slab: Vec::with_capacity(sized::REACTOR_SLAB_INITIAL_CAP),
                free_head: None,
                live_sources: 0,
            },
            next_generation: 1,
            wakeup,
        })
    }

    #[must_use]
    pub fn live_sources(&self) -> usize {
        self.sources.live_sources
    }

    #[must_use]
    pub fn wakeup(&self) -> Wakeup {
        self.wakeup.clone()
    }

    pub fn arm_wakeup(&self) {
        self.wakeup.inner.needs_wake.store(true, Ordering::Release);
    }

    pub fn disarm_wakeup(&self) {
        self.wakeup.inner.needs_wake.store(false, Ordering::Release);
    }

    pub fn register(&mut self, source: impl AsSocket, interest: Interest) -> io::Result<SourceKey> {
        let socket = source.as_socket().try_clone_to_owned()?;
        let generation = self.take_generation();
        self.sources
            .insert(SourceSlot::new(socket, generation, interest))
    }

    /// the slot keeps the duplicate until deregistration; completions of overlapped I/O on it
    /// land on this reactor's port and the latest one is read back with [`Self::take_completion`].
    pub(crate) fn register_completion(&mut self, handle: OwnedHandle) -> io::Result<SourceKey> {
        let generation = self.take_generation();
        let handle = Handle::Completion { handle, last: None };
        let key =
            self.sources
                .insert(SourceSlot::with_handle(handle, generation, Interest::Read))?;
        let associated = self.sources.live(key).and_then(SourceSlot::bound_handle);
        let outcome = match associated {
            Some(bound) => self.port.associate(bound, pack_key(key)),
            None => Err(stale_source()),
        };
        self.vacate_on_error(key, outcome)
    }

    /// the gate opens once `handle` signals; an already-signaled handle opens it at registration.
    pub(crate) fn register_signaled(&mut self, handle: OwnedHandle) -> io::Result<SourceKey> {
        let generation = self.take_generation();
        let packet = waitpacket::create()?;
        let handle = Handle::Signaled {
            target: handle,
            packet,
            armed: false,
        };
        let key =
            self.sources
                .insert(SourceSlot::with_handle(handle, generation, Interest::Read))?;
        let outcome = self.arm_wait(key);
        self.vacate_on_error(key, outcome)
    }

    fn arm_wait(&mut self, key: SourceKey) -> io::Result<()> {
        let slot = self.sources.live_mut(key)?;
        let Handle::Signaled {
            target,
            packet,
            armed,
        } = &mut slot.handle
        else {
            return Err(stale_source());
        };
        let already_signaled = waitpacket::associate(
            packet.as_handle(),
            self.port.as_handle(),
            target.as_handle(),
            pack_key(key),
        )?;
        *armed = !already_signaled;
        if already_signaled {
            slot.fire_read(true);
        }
        Ok(())
    }

    fn vacate_on_error(
        &mut self,
        key: SourceKey,
        outcome: io::Result<()>,
    ) -> io::Result<SourceKey> {
        match outcome {
            Ok(()) => Ok(key),
            Err(error) => {
                let _ = self.sources.deregister(key);
                Err(error)
            }
        }
    }

    pub(crate) fn take_completion(&mut self, key: SourceKey) -> Option<OVERLAPPED_ENTRY> {
        match &mut self.sources.live_mut(key).ok()?.handle {
            Handle::Completion { last, .. } => last.take(),
            _ => None,
        }
    }

    fn take_generation(&mut self) -> u32 {
        let generation = self.next_generation;
        self.next_generation = generation.wrapping_add(1).max(1);
        generation
    }

    pub fn reregister(&mut self, key: SourceKey, interest: Interest) -> io::Result<()> {
        let slot = self.sources.live_mut(key)?;
        slot.read_armed = interest.wants_read();
        slot.write_armed = interest.wants_write();
        self.sources.sync_poll(key.index as usize)
    }

    /// rearm only a direction whose nonblocking operation returned pending.
    pub fn rearm(&mut self, key: SourceKey, interest: Interest) -> io::Result<()> {
        let slot = self.sources.live_mut(key)?;
        slot.read_armed |= interest.wants_read();
        slot.write_armed |= interest.wants_write();
        self.sources.sync_poll(key.index as usize)
    }

    pub fn deregister(&mut self, key: SourceKey) -> io::Result<()> {
        self.sources.deregister(key)
    }

    pub fn set_read_waker(&mut self, key: SourceKey, waker: Waker) -> bool {
        let Ok(slot) = self.sources.live_mut(key) else {
            return false;
        };
        slot.read_waker = Some(waker);
        true
    }

    pub fn set_write_waker(&mut self, key: SourceKey, waker: Waker) -> bool {
        let Ok(slot) = self.sources.live_mut(key) else {
            return false;
        };
        slot.write_waker = Some(waker);
        true
    }

    pub fn register_read_waker_ref(&mut self, key: SourceKey, waker: &Waker) -> bool {
        let Ok(slot) = self.sources.live_mut(key) else {
            return false;
        };
        if !slot
            .read_waker
            .as_ref()
            .is_some_and(|stored| stored.will_wake(waker))
        {
            slot.read_waker = Some(waker.clone());
        }
        true
    }

    pub fn register_write_waker_ref(&mut self, key: SourceKey, waker: &Waker) -> bool {
        let Ok(slot) = self.sources.live_mut(key) else {
            return false;
        };
        if !slot
            .write_waker
            .as_ref()
            .is_some_and(|stored| stored.will_wake(waker))
        {
            slot.write_waker = Some(waker.clone());
        }
        true
    }

    pub fn read_ready_epoch(&self, key: SourceKey) -> Option<u32> {
        self.sources.live(key).map(|slot| slot.read_ready_epoch)
    }

    pub fn write_ready_epoch(&self, key: SourceKey) -> Option<u32> {
        self.sources.live(key).map(|slot| slot.write_ready_epoch)
    }

    pub fn turn(&mut self, timeout: Option<Duration>) -> io::Result<usize> {
        self.drain_cancellations()?;
        let entries = match self.port.dequeue(timeout) {
            Err(error) if error.kind() == io::ErrorKind::Interrupted => return Ok(0),
            result => result?,
        };
        while let Some((key, _socket)) = self.wakeup.inner.cancellations.pop() {
            self.sources.deregister(key)?;
        }
        Ok(entries
            .iter()
            .map(|entry| self.sources.complete(entry))
            .sum())
    }

    fn drain_in_flight(&mut self) {
        while self.sources.in_flight() {
            let Ok(entries) = self.port.dequeue(Some(SHUTDOWN_DRAIN)) else {
                break;
            };
            if entries.is_empty() {
                break;
            }
            for entry in entries {
                self.sources.complete(entry);
            }
        }
    }
}

impl Drop for Reactor {
    fn drop(&mut self) {
        // retained wake handles keep the port alive; cancel every poll and wait for its packet
        // so the kernel no longer writes into slots this reactor frees.
        self.sources.cancel_all();
        self.drain_in_flight();
        self.sources.leak_in_flight();
        self.retire();
    }
}

fn stale_source() -> io::Error {
    io::Error::other("reactor source went stale")
}

fn interest_of(readable: bool, writable: bool) -> Option<Interest> {
    match (readable, writable) {
        (true, true) => Some(Interest::ReadWrite),
        (true, false) => Some(Interest::Read),
        (false, true) => Some(Interest::Write),
        (false, false) => None,
    }
}

fn covers(polled: Interest, wanted: Interest) -> bool {
    (!wanted.wants_read() || polled.wants_read()) && (!wanted.wants_write() || polled.wants_write())
}
