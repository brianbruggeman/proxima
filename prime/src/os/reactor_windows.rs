use std::os::windows::io::{AsSocket, OwnedSocket};

use polling::{Event, Events, Poller};

use super::{
    Arc, AtomicBool, Duration, Interest, Ordering, SourceKey, Waker, Wakeup, WakeupInner, io,
};

struct SourceSlot {
    socket: Option<OwnedSocket>,
    generation: u32,
    read_waker: Option<Waker>,
    write_waker: Option<Waker>,
    read_ready_epoch: u32,
    read_armed: bool,
    write_armed: bool,
}

/// owns the source slab; `polling` supplies IOCP waiting and notification.
pub struct Reactor {
    poller: Arc<Poller>,
    events: Events,
    slab: Vec<SourceSlot>,
    free: Vec<u32>,
    next_generation: u32,
    live_sources: usize,
    pub(super) wakeup: Wakeup,
}

impl Reactor {
    pub fn new() -> io::Result<Self> {
        let poller = Arc::new(Poller::new()?);
        let wakeup = Wakeup {
            inner: Arc::new(WakeupInner {
                needs_wake: AtomicBool::new(false),
                alive: AtomicBool::new(true),
                cancellations: super::SegQueue::new(),
                poller: poller.clone(),
            }),
        };
        Ok(Self {
            poller,
            events: Events::new(),
            slab: Vec::new(),
            free: Vec::new(),
            next_generation: 1,
            live_sources: 0,
            wakeup,
        })
    }

    #[must_use]
    pub fn live_sources(&self) -> usize {
        self.live_sources
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
        let index = match self.free.last() {
            Some(index) => *index,
            None => u32::try_from(self.slab.len())
                .map_err(|_| io::Error::other("reactor source capacity exhausted"))?,
        };
        let generation = self.next_generation;
        let key = SourceKey { index, generation };
        let mut slot = SourceSlot {
            socket: None,
            generation,
            read_waker: None,
            write_waker: None,
            read_ready_epoch: 0,
            read_armed: interest.wants_read(),
            write_armed: interest.wants_write(),
        };
        // safety: the slab owns the duplicate until deletion completes.
        unsafe {
            self.poller
                .add(&socket, event_for(key, slot.read_armed, slot.write_armed)?)?
        };
        slot.socket = Some(socket);
        if self.free.pop().is_some() {
            self.slab[index as usize] = slot;
        } else {
            self.slab.push(slot);
        }
        self.next_generation = generation.wrapping_add(1).max(1);
        self.live_sources += 1;
        Ok(key)
    }

    pub fn reregister(&mut self, key: SourceKey, interest: Interest) -> io::Result<()> {
        let slot = self
            .slab
            .get(key.index as usize)
            .filter(|slot| slot.generation == key.generation && slot.generation != 0)
            .ok_or_else(stale_source)?;
        let socket = slot.socket.as_ref().ok_or_else(stale_source)?;
        self.poller.modify(
            socket,
            event_for(key, interest.wants_read(), interest.wants_write())?,
        )?;
        let slot = self.slot_mut(key)?;
        slot.read_armed = interest.wants_read();
        slot.write_armed = interest.wants_write();
        Ok(())
    }

    /// rearm only a direction whose nonblocking operation returned pending.
    pub fn rearm(&mut self, key: SourceKey, interest: Interest) -> io::Result<()> {
        let slot = self
            .slab
            .get(key.index as usize)
            .filter(|slot| slot.generation == key.generation && slot.generation != 0)
            .ok_or_else(stale_source)?;
        let readable = slot.read_armed || interest.wants_read();
        let writable = slot.write_armed || interest.wants_write();
        if readable == slot.read_armed && writable == slot.write_armed {
            return Ok(());
        }
        // safety: the registered socket remains owned by its source until deregistration.
        let socket = slot.socket.as_ref().ok_or_else(stale_source)?;
        self.poller
            .modify(socket, event_for(key, readable, writable)?)?;
        let slot = self.slot_mut(key)?;
        slot.read_armed = readable;
        slot.write_armed = writable;
        Ok(())
    }

    pub fn deregister(&mut self, key: SourceKey) -> io::Result<()> {
        let Some(slot) = self
            .slab
            .get_mut(key.index as usize)
            .filter(|slot| slot.generation == key.generation && slot.generation != 0)
        else {
            return Ok(());
        };
        // safety: deletion runs before the source closes its registered socket.
        let socket = slot.socket.as_ref().ok_or_else(stale_source)?;
        self.poller.delete(socket)?;
        slot.socket = None;
        slot.generation = 0;
        slot.read_waker = None;
        slot.write_waker = None;
        slot.read_armed = false;
        slot.write_armed = false;
        self.free.push(key.index);
        self.live_sources -= 1;
        Ok(())
    }

    pub fn set_read_waker(&mut self, key: SourceKey, waker: Waker) -> bool {
        let Ok(slot) = self.slot_mut(key) else {
            return false;
        };
        slot.read_waker = Some(waker);
        true
    }

    pub fn set_write_waker(&mut self, key: SourceKey, waker: Waker) -> bool {
        let Ok(slot) = self.slot_mut(key) else {
            return false;
        };
        slot.write_waker = Some(waker);
        true
    }

    pub fn register_read_waker_ref(&mut self, key: SourceKey, waker: &Waker) -> bool {
        let Ok(slot) = self.slot_mut(key) else {
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
        let Ok(slot) = self.slot_mut(key) else {
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
        self.slab
            .get(key.index as usize)
            .filter(|slot| slot.generation == key.generation && slot.generation != 0)
            .map(|slot| slot.read_ready_epoch)
    }

    pub fn turn(&mut self, timeout: Option<Duration>) -> io::Result<usize> {
        self.drain_cancellations()?;
        self.events.clear();
        let count = match self.poller.wait(&mut self.events, timeout) {
            Err(error) if error.kind() == io::ErrorKind::Interrupted => return Ok(0),
            result => result?,
        };
        self.drain_cancellations()?;
        for event in self.events.iter() {
            let key = SourceKey {
                index: (event.key as u64 & u64::from(u32::MAX)) as u32,
                generation: ((event.key as u64) >> 32) as u32,
            };
            let Some(slot) = self
                .slab
                .get_mut(key.index as usize)
                .filter(|slot| slot.generation == key.generation && slot.generation != 0)
            else {
                continue;
            };
            if event.readable && slot.read_armed {
                slot.read_armed = false;
                slot.read_ready_epoch = slot.read_ready_epoch.wrapping_add(1);
                if let Some(waker) = &slot.read_waker {
                    waker.wake_by_ref();
                }
            }
            if event.writable && slot.write_armed {
                slot.write_armed = false;
                if let Some(waker) = &slot.write_waker {
                    waker.wake_by_ref();
                }
            }
            // one-shot consumption covers both directions; preserve an unfired waiter.
            if slot.read_armed || slot.write_armed {
                let socket = slot.socket.as_ref().ok_or_else(stale_source)?;
                self.poller
                    .modify(socket, event_for(key, slot.read_armed, slot.write_armed)?)?;
            }
        }
        Ok(count)
    }

    fn slot_mut(&mut self, key: SourceKey) -> io::Result<&mut SourceSlot> {
        self.slab
            .get_mut(key.index as usize)
            .filter(|slot| slot.generation == key.generation && slot.generation != 0)
            .ok_or_else(stale_source)
    }
}

impl Drop for Reactor {
    fn drop(&mut self) {
        // retained wake handles keep the poller alive; remove its owned registrations first.
        for slot in &self.slab {
            if let Some(socket) = &slot.socket {
                let _ = self.poller.delete(socket);
            }
        }
        self.retire();
    }
}

fn stale_source() -> io::Error {
    io::Error::other("reactor source went stale")
}

fn event_for(key: SourceKey, readable: bool, writable: bool) -> io::Result<Event> {
    let packed = (u64::from(key.generation) << 32) | u64::from(key.index);
    let token = usize::try_from(packed)
        .map_err(|_| io::Error::other("reactor requires 64-bit source tokens"))?;
    Ok(Event::new(token, readable, writable))
}
