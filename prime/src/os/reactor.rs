//! per-core I/O readiness reactor. POSIX-only (`kqueue` on macOS,
//! `epoll` on Linux) via raw `libc::*` syscalls — zero added deps beyond
//! the optional `libc` crate gated on this feature.
//!
//! design: single-thread-owned (`!Send`). source slab keyed by `SourceKey`
//! (slab index + generation); each slot stores read/write wakers inline.
//! `turn(timeout)` blocks until at least one source is ready, fires the
//! matching waker(s), returns event count.
//!
//! intentionally minimal: no edge/level toggle (defaults to level-triggered
//! oneshot on Linux for parity with the lazy-poll model), no priority,
//! no signalfd. those land as follow-up changelog rows when the executor
//! needs them.

use std::io;
use std::os::fd::RawFd;
use std::sync::Arc;
use std::sync::atomic::{self, AtomicBool, Ordering};
use std::task::Waker;
use std::time::Duration;

/// shared wake state. cheap to clone (shares an `Arc`). external threads
/// `fire()` it to interrupt the owning worker's `Reactor::turn` call.
///
/// implemented on macOS via `EVFILT_USER` (one user-event filter registered
/// with the kqueue, triggered with `NOTE_TRIGGER`); on Linux via `eventfd`
/// (a pre-registered fd that we write to). Both wake `epoll_wait`/`kevent`
/// immediately.
///
/// `needs_wake` is checked before firing the actual syscall — when the
/// worker isn't parked, the cost is one atomic load. mirrors the
/// `consumer_parked` elision pattern in the C1 inbox.
#[derive(Clone)]
pub struct Wakeup {
    inner: Arc<WakeupInner>,
}

struct WakeupInner {
    needs_wake: AtomicBool,
    #[cfg(target_os = "macos")]
    kq: RawFd,
    #[cfg(target_os = "macos")]
    ident: usize,
    #[cfg(target_os = "linux")]
    eventfd: RawFd,
}

impl Wakeup {
    /// fire the wakeup if the worker is currently parked on `Reactor::turn`.
    /// no-op when the worker is busy.
    ///
    /// **Dekker-pattern hazard.** Callers do something like:
    /// `inbox.tail.store(Release); wakeup.fire();`. The worker arms via
    /// `needs_wake.store(true, Release)` and then re-drains the inbox
    /// via `tail.load(Acquire)`. Release/Acquire on DIFFERENT atomics
    /// (`tail` vs `needs_wake`) does NOT establish cross-variable
    /// happens-before — both sides can observe stale values of the
    /// other's atomic, yielding a lost wake. Specifically:
    ///   - producer: `tail.store(Release)` then `needs_wake.load`
    ///     returns false (worker hasn't armed yet) → no syscall
    ///   - worker:   `needs_wake.store(true, Release)` then
    ///     `tail.load(Acquire)` reads OLD tail → recheck empty
    ///   - worker parks on `turn(None)`; task wedged in inbox
    ///
    /// The `SeqCst` fence below participates in the global SeqCst
    /// total order with the matching fence in `core_shard::worker_main`
    /// after `arm_wakeup`. Whichever fence sequenced-first in the total
    /// order, the OTHER side will see its preceding store: producer
    /// fires (worker's arm visible), or worker drains (producer's push
    /// visible). One side always acts.
    pub fn fire(&self) {
        atomic::fence(Ordering::SeqCst);
        if !self.inner.needs_wake.load(Ordering::Acquire) {
            return;
        }
        // race: another producer may have already won and cleared the flag.
        // we tolerate spurious syscalls; correctness rests on at-least-one
        // wake reaching the parked worker.
        if self.inner.needs_wake.swap(false, Ordering::AcqRel) {
            self.fire_syscall();
        }
    }

    #[cfg(target_os = "macos")]
    fn fire_syscall(&self) {
        use core::mem::MaybeUninit;
        let mut trigger: libc::kevent = unsafe { MaybeUninit::zeroed().assume_init() };
        trigger.ident = self.inner.ident;
        trigger.filter = libc::EVFILT_USER;
        trigger.flags = 0;
        trigger.fflags = libc::NOTE_TRIGGER;
        // SAFETY: kq is owned by the Reactor for the lifetime of this Arc;
        // libc::kevent with one change and zero events is a safe trigger.
        let _ = unsafe {
            libc::kevent(
                self.inner.kq,
                &trigger,
                1,
                core::ptr::null_mut(),
                0,
                core::ptr::null(),
            )
        };
    }

    #[cfg(target_os = "linux")]
    fn fire_syscall(&self) {
        let payload: u64 = 1;
        // SAFETY: eventfd is owned by the Reactor for the lifetime of this Arc.
        let _ = unsafe {
            libc::write(
                self.inner.eventfd,
                (&payload as *const u64).cast(),
                core::mem::size_of::<u64>(),
            )
        };
    }
}

/// what readiness to wait for on a source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interest {
    Read,
    Write,
    ReadWrite,
}

impl Interest {
    fn wants_read(self) -> bool {
        matches!(self, Self::Read | Self::ReadWrite)
    }
    fn wants_write(self) -> bool {
        matches!(self, Self::Write | Self::ReadWrite)
    }
}

/// opaque handle for a registered source. cheap to copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceKey {
    pub(crate) index: u32,
    pub(crate) generation: u32,
}

/// Per-source state. Packed (no `Option` wrap) — a slot with
/// `generation == 0` is dead; non-zero is live. Saves the 8-byte
/// `Option<...>` discriminant + padding per slot vs the prior
/// `Vec<Option<SourceSlot>>` shape. The slab stays denser in cache,
/// and the hot-path `register_*_waker_ref` / `turn` event scan
/// drops one `Option::as_mut()` dereference per access.
#[cfg(any(target_os = "macos", target_os = "linux"))]
struct SourceSlot {
    fd: RawFd,
    interest: Interest,
    /// `0` means dead. `register` assigns a nonzero generation;
    /// `deregister` clears it back to 0. The `bump_generation` helper
    /// ensures wrapping never returns 0.
    generation: u32,
    read_waker: Option<Waker>,
    write_waker: Option<Waker>,
    read_ready_epoch: u32,
    write_ready_epoch: u32,
}

/// Bump the generation counter, skipping 0 (the dead sentinel).
#[cfg(any(target_os = "macos", target_os = "linux"))]
#[inline]
fn bump_generation(current: u32) -> u32 {
    match current.wrapping_add(1) {
        0 => 1,
        next => next,
    }
}

#[cfg(target_os = "macos")]
mod kqueue {
    use super::*;
    use core::mem::MaybeUninit;

    /// ident used for the self-wake EVFILT_USER filter. arbitrary u32 that
    /// won't collide with real fd idents (file descriptors are ints; we use
    /// a high bit-set value to be safe).
    const WAKE_IDENT: usize = 0xFFFF_FFFF_0000_0001;

    pub struct Reactor {
        kq: RawFd,
        slab: Vec<SourceSlot>,
        free_head: Option<u32>,
        next_generation: u32,
        events_buf: Vec<libc::kevent>,
        live_sources: usize,
        wakeup: super::Wakeup,
    }

    impl Reactor {
        pub fn new() -> io::Result<Self> {
            // SAFETY: kqueue() is a thin syscall wrapper; returns fd or -1.
            let kq = unsafe { libc::kqueue() };
            if kq < 0 {
                return Err(io::Error::last_os_error());
            }
            // register the self-wake EVFILT_USER. EV_CLEAR makes it
            // edge-triggered: each NOTE_TRIGGER wakes kevent once.
            let mut user_event: libc::kevent = unsafe { MaybeUninit::zeroed().assume_init() };
            user_event.ident = WAKE_IDENT;
            user_event.filter = libc::EVFILT_USER;
            user_event.flags = libc::EV_ADD | libc::EV_CLEAR;
            let ret = unsafe {
                libc::kevent(
                    kq,
                    &user_event,
                    1,
                    core::ptr::null_mut(),
                    0,
                    core::ptr::null(),
                )
            };
            if ret < 0 {
                let err = io::Error::last_os_error();
                unsafe { libc::close(kq) };
                return Err(err);
            }
            let wakeup = super::Wakeup {
                inner: Arc::new(super::WakeupInner {
                    needs_wake: AtomicBool::new(false),
                    kq,
                    ident: WAKE_IDENT,
                }),
            };
            Ok(Self {
                kq,
                slab: Vec::new(),
                free_head: None,
                next_generation: 1,
                events_buf: vec![unsafe { MaybeUninit::zeroed().assume_init() }; 256],
                live_sources: 0,
                wakeup,
            })
        }

        /// cheap-to-clone wake handle. external threads call `fire()` to
        /// interrupt this reactor's blocking `turn` call.
        #[must_use]
        pub fn wakeup(&self) -> super::Wakeup {
            self.wakeup.clone()
        }

        /// arms the wakeup flag, indicating the worker is about to (or just
        /// did) call `turn` and producers should signal on send.
        pub fn arm_wakeup(&self) {
            self.wakeup.inner.needs_wake.store(true, Ordering::Release);
        }

        /// disarms the wakeup flag — call after `turn` returns and before
        /// proceeding to the next cycle of work.
        pub fn disarm_wakeup(&self) {
            self.wakeup.inner.needs_wake.store(false, Ordering::Release);
        }

        /// number of currently-registered sources. callers can skip the
        /// `turn` syscall entirely when this is 0 and they have nothing else
        /// to wait for.
        #[must_use]
        pub fn live_sources(&self) -> usize {
            self.live_sources
        }

        pub fn register(&mut self, fd: RawFd, interest: Interest) -> io::Result<SourceKey> {
            let generation = self.next_generation;
            self.next_generation = super::bump_generation(self.next_generation);
            let index = match self.free_head.take() {
                Some(free) => {
                    let slot = &mut self.slab[free as usize];
                    debug_assert!(slot.generation == 0, "free slot held a live source");
                    slot.fd = fd;
                    slot.interest = interest;
                    slot.generation = generation;
                    slot.read_waker = None;
                    slot.write_waker = None;
                    slot.read_ready_epoch = 0;
                    slot.write_ready_epoch = 0;
                    free
                }
                None => {
                    let raw = self.slab.len();
                    assert!(raw < u32::MAX as usize, "reactor slab > u32::MAX");
                    self.slab.push(SourceSlot {
                        fd,
                        interest,
                        generation,
                        read_waker: None,
                        write_waker: None,
                        read_ready_epoch: 0,
                        write_ready_epoch: 0,
                    });
                    raw as u32
                }
            };
            let udata = pack_udata(index, generation);
            self.kevent_change(fd, interest, libc::EV_ADD | libc::EV_CLEAR, udata)?;
            self.live_sources = self.live_sources.saturating_add(1);
            Ok(SourceKey { index, generation })
        }

        pub fn reregister(&mut self, key: SourceKey, interest: Interest) -> io::Result<()> {
            let slot_index = key.index as usize;
            let (fd, old_interest) = {
                let Some(slot) = self.slab.get(slot_index) else {
                    return Err(io::Error::other("Reactor: source went stale"));
                };
                if slot.generation != key.generation {
                    return Err(io::Error::other("Reactor: source went stale"));
                }
                if slot.interest == interest {
                    return Ok(());
                }
                (slot.fd, slot.interest)
            };

            if old_interest.wants_read() && !interest.wants_read() {
                let _ = self.kevent_change(fd, Interest::Read, libc::EV_DELETE, 0);
            }
            if old_interest.wants_write() && !interest.wants_write() {
                let _ = self.kevent_change(fd, Interest::Write, libc::EV_DELETE, 0);
            }
            let udata = pack_udata(key.index, key.generation);
            if interest.wants_read() && !old_interest.wants_read() {
                self.kevent_change(fd, Interest::Read, libc::EV_ADD | libc::EV_CLEAR, udata)?;
            }
            if interest.wants_write() && !old_interest.wants_write() {
                self.kevent_change(fd, Interest::Write, libc::EV_ADD | libc::EV_CLEAR, udata)?;
            }
            if let Some(slot) = self.slab.get_mut(slot_index)
                && slot.generation == key.generation
            {
                slot.interest = interest;
            }
            Ok(())
        }

        pub fn deregister(&mut self, key: SourceKey) -> io::Result<()> {
            let slot_index = key.index as usize;
            // pull fd out under a short-scoped borrow so the subsequent
            // self.kevent_change call doesn't conflict with the &mut self.slab.
            let (fd, interest) = {
                let Some(slot) = self.slab.get_mut(slot_index) else {
                    return Ok(());
                };
                if slot.generation != key.generation {
                    return Ok(());
                }
                (slot.fd, slot.interest)
            };
            // remove from kqueue (ignore errors — fd may already be closed).
            let _ = self.kevent_change(fd, interest, libc::EV_DELETE, 0);
            // re-borrow to clear the slot fields after the kevent syscall.
            if let Some(slot) = self.slab.get_mut(slot_index) {
                slot.generation = 0;
                slot.fd = -1;
                slot.interest = Interest::Read;
                slot.read_waker = None;
                slot.write_waker = None;
                slot.read_ready_epoch = 0;
                slot.write_ready_epoch = 0;
            }
            self.free_head = Some(key.index);
            self.live_sources = self.live_sources.saturating_sub(1);
            Ok(())
        }

        /// stores `waker` for the read-readiness slot. returns true if the
        /// slot is alive (generation matches), false if stale. Callers should
        /// prefer `register_read_waker_ref` on the hot path — it elides the
        /// `Arc<Waker>` increment when the same task re-polls.
        pub fn set_read_waker(&mut self, key: SourceKey, waker: Waker) -> bool {
            let Some(slot) = self.slab.get_mut(key.index as usize) else {
                return false;
            };
            if slot.generation != key.generation {
                return false;
            }
            slot.read_waker = Some(waker);
            true
        }

        pub fn set_write_waker(&mut self, key: SourceKey, waker: Waker) -> bool {
            let Some(slot) = self.slab.get_mut(key.index as usize) else {
                return false;
            };
            if slot.generation != key.generation {
                return false;
            }
            slot.write_waker = Some(waker);
            true
        }

        /// hot-path waker registration. clones the waker only if the slot's
        /// stored waker does not already `will_wake` the same task. Saves the
        /// per-poll Arc increment for the steady-state case where one task
        /// re-polls the same source.
        #[inline]
        pub fn register_read_waker_ref(&mut self, key: SourceKey, waker: &Waker) -> bool {
            let Some(slot) = self.slab.get_mut(key.index as usize) else {
                return false;
            };
            if slot.generation != key.generation {
                return false;
            }
            match &slot.read_waker {
                Some(existing) if existing.will_wake(waker) => return true,
                _ => {}
            }
            slot.read_waker = Some(waker.clone());
            true
        }

        #[inline]
        pub fn register_write_waker_ref(&mut self, key: SourceKey, waker: &Waker) -> bool {
            let Some(slot) = self.slab.get_mut(key.index as usize) else {
                return false;
            };
            if slot.generation != key.generation {
                return false;
            }
            match &slot.write_waker {
                Some(existing) if existing.will_wake(waker) => return true,
                _ => {}
            }
            slot.write_waker = Some(waker.clone());
            true
        }

        #[inline]
        pub fn read_ready_epoch(&self, key: SourceKey) -> Option<u32> {
            let slot = self.slab.get(key.index as usize)?;
            (slot.generation == key.generation).then_some(slot.read_ready_epoch)
        }

        pub fn turn(&mut self, timeout: Option<Duration>) -> io::Result<usize> {
            let timespec = timeout.map(|duration| libc::timespec {
                tv_sec: duration.as_secs() as libc::time_t,
                tv_nsec: libc::c_long::from(duration.subsec_nanos() as i32),
            });
            let timespec_ptr = timespec
                .as_ref()
                .map(|ts| ts as *const libc::timespec)
                .unwrap_or(core::ptr::null());
            // SAFETY: kq is owned + valid; events_buf is a valid &mut [kevent] of capacity.
            let count = unsafe {
                libc::kevent(
                    self.kq,
                    core::ptr::null(),
                    0,
                    self.events_buf.as_mut_ptr(),
                    self.events_buf.len() as i32,
                    timespec_ptr,
                )
            };
            if count < 0 {
                return Err(io::Error::last_os_error());
            }
            #[cfg(feature = "runtime-prime-reactor-trace")]
            if count == 0 {
                crate::trace::record_reactor_timeout();
            }
            let mut fired = 0;
            for event in &self.events_buf[..count as usize] {
                if event.filter == libc::EVFILT_USER {
                    #[cfg(feature = "runtime-prime-reactor-trace")]
                    crate::trace::record_reactor_wakeup_event();
                    // self-wake fired; not an I/O event. caller's outer loop
                    // re-drains the inbox; nothing to count.
                    continue;
                }
                let (index, generation) = unpack_udata(event.udata as u64);
                let Some(slot) = self.slab.get_mut(index as usize) else {
                    #[cfg(feature = "runtime-prime-reactor-trace")]
                    crate::trace::record_reactor_stale_event();
                    continue;
                };
                if slot.generation != generation {
                    #[cfg(feature = "runtime-prime-reactor-trace")]
                    crate::trace::record_reactor_stale_event();
                    continue;
                }
                // Use wake_by_ref instead of take()+wake(): the waker stays
                // in the slot, so the next register_*_waker_ref's will_wake
                // check succeeds and elides the Arc clone. Net per request:
                // 1 fewer Arc bump + 1 fewer slot mutation per I/O event.
                if event.filter == libc::EVFILT_READ {
                    if let Some(waker) = slot.read_waker.as_ref() {
                        slot.read_ready_epoch = super::bump_generation(slot.read_ready_epoch);
                        #[cfg(feature = "runtime-prime-reactor-trace")]
                        crate::trace::record_event_ready();
                        #[cfg(feature = "runtime-prime-reactor-trace")]
                        crate::trace::record_waker_start();
                        waker.wake_by_ref();
                        #[cfg(feature = "runtime-prime-reactor-trace")]
                        crate::trace::record_waker_end();
                        fired += 1;
                    } else {
                        #[cfg(feature = "runtime-prime-reactor-trace")]
                        crate::trace::record_reactor_ignored_read();
                    }
                } else if event.filter == libc::EVFILT_WRITE {
                    if let Some(waker) = slot.write_waker.as_ref() {
                        slot.write_ready_epoch = super::bump_generation(slot.write_ready_epoch);
                        waker.wake_by_ref();
                        fired += 1;
                    } else {
                        #[cfg(feature = "runtime-prime-reactor-trace")]
                        crate::trace::record_reactor_ignored_write();
                    }
                } else {
                    #[cfg(feature = "runtime-prime-reactor-trace")]
                    crate::trace::record_reactor_unknown_event();
                }
            }
            Ok(fired)
        }

        fn kevent_change(
            &self,
            fd: RawFd,
            interest: Interest,
            flags: u16,
            udata: u64,
        ) -> io::Result<()> {
            let mut changes: [libc::kevent; 2] = unsafe { MaybeUninit::zeroed().assume_init() };
            let mut count: usize = 0;
            if interest.wants_read() {
                changes[count] = libc::kevent {
                    ident: fd as usize,
                    filter: libc::EVFILT_READ,
                    flags,
                    fflags: 0,
                    data: 0,
                    udata: udata as *mut _,
                };
                count += 1;
            }
            if interest.wants_write() {
                changes[count] = libc::kevent {
                    ident: fd as usize,
                    filter: libc::EVFILT_WRITE,
                    flags,
                    fflags: 0,
                    data: 0,
                    udata: udata as *mut _,
                };
                count += 1;
            }
            // SAFETY: changes is owned + valid for `count` entries.
            let ret = unsafe {
                libc::kevent(
                    self.kq,
                    changes.as_ptr(),
                    count as i32,
                    core::ptr::null_mut(),
                    0,
                    core::ptr::null(),
                )
            };
            if ret < 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        }
    }

    impl Drop for Reactor {
        fn drop(&mut self) {
            // SAFETY: kq was created by kqueue(); closing once is correct.
            unsafe {
                libc::close(self.kq);
            }
        }
    }

    fn pack_udata(index: u32, generation: u32) -> u64 {
        ((generation as u64) << 32) | (index as u64)
    }

    fn unpack_udata(raw: u64) -> (u32, u32) {
        (raw as u32, (raw >> 32) as u32)
    }
}

#[cfg(target_os = "linux")]
mod epoll {
    use super::*;

    /// distinguished `u64` cookie stored in `epoll_event.u64` for the
    /// eventfd self-wake. high bit set to avoid collision with the
    /// packed (gen << 32) | index used for I/O sources.
    const WAKE_COOKIE: u64 = u64::MAX;

    pub struct Reactor {
        epfd: RawFd,
        eventfd: RawFd,
        slab: Vec<SourceSlot>,
        free_head: Option<u32>,
        next_generation: u32,
        events_buf: Vec<libc::epoll_event>,
        live_sources: usize,
        wakeup: super::Wakeup,
    }

    impl Reactor {
        pub fn new() -> io::Result<Self> {
            // SAFETY: epoll_create1() returns fd or -1.
            let epfd = unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) };
            if epfd < 0 {
                return Err(io::Error::last_os_error());
            }
            // SAFETY: eventfd() returns fd or -1.
            let eventfd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
            if eventfd < 0 {
                let err = io::Error::last_os_error();
                unsafe { libc::close(epfd) };
                return Err(err);
            }
            let mut event = libc::epoll_event {
                events: (libc::EPOLLIN | libc::EPOLLET) as u32,
                u64: WAKE_COOKIE,
            };
            // SAFETY: epfd and eventfd both valid + owned.
            let ret = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, eventfd, &mut event) };
            if ret < 0 {
                let err = io::Error::last_os_error();
                unsafe { libc::close(eventfd) };
                unsafe { libc::close(epfd) };
                return Err(err);
            }
            let wakeup = super::Wakeup {
                inner: Arc::new(super::WakeupInner {
                    needs_wake: AtomicBool::new(false),
                    eventfd,
                }),
            };
            Ok(Self {
                epfd,
                eventfd,
                slab: Vec::new(),
                free_head: None,
                next_generation: 1,
                events_buf: vec![libc::epoll_event { events: 0, u64: 0 }; 256],
                live_sources: 0,
                wakeup,
            })
        }

        #[must_use]
        pub fn live_sources(&self) -> usize {
            self.live_sources
        }

        #[must_use]
        pub fn wakeup(&self) -> super::Wakeup {
            self.wakeup.clone()
        }

        /// raw epoll fd backing this reactor. an epoll fd is itself readable
        /// whenever any monitored source (including the self-wake eventfd) has
        /// pending events, so the inverted compat worker registers it into the
        /// sister tokio runtime's reactor (`AsyncFd`) for a unified park: one
        /// wait that wakes on prime I/O OR a prime inbox wakeup while the
        /// sister's own reactor services tokio I/O. the fd stays owned by this
        /// reactor; callers must not close it.
        #[must_use]
        pub fn raw_poll_fd(&self) -> RawFd {
            self.epfd
        }

        pub fn arm_wakeup(&self) {
            self.wakeup.inner.needs_wake.store(true, Ordering::Release);
        }

        pub fn disarm_wakeup(&self) {
            self.wakeup.inner.needs_wake.store(false, Ordering::Release);
        }

        pub fn register(&mut self, fd: RawFd, interest: Interest) -> io::Result<SourceKey> {
            let generation = self.next_generation;
            self.next_generation = super::bump_generation(self.next_generation);
            let index = match self.free_head.take() {
                Some(free) => {
                    let slot = &mut self.slab[free as usize];
                    debug_assert!(slot.generation == 0, "free slot held a live source");
                    slot.fd = fd;
                    slot.interest = interest;
                    slot.generation = generation;
                    slot.read_waker = None;
                    slot.write_waker = None;
                    slot.read_ready_epoch = 0;
                    slot.write_ready_epoch = 0;
                    free
                }
                None => {
                    let raw = self.slab.len();
                    assert!(raw < u32::MAX as usize, "reactor slab > u32::MAX");
                    self.slab.push(SourceSlot {
                        fd,
                        interest,
                        generation,
                        read_waker: None,
                        write_waker: None,
                        read_ready_epoch: 0,
                        write_ready_epoch: 0,
                    });
                    raw as u32
                }
            };
            let mut events: u32 = libc::EPOLLET as u32;
            if interest.wants_read() {
                events |= libc::EPOLLIN as u32;
            }
            if interest.wants_write() {
                events |= libc::EPOLLOUT as u32;
            }
            let mut event = libc::epoll_event {
                events,
                u64: pack_udata(index, generation),
            };
            // SAFETY: epoll_ctl with valid fd and event.
            let ret = unsafe { libc::epoll_ctl(self.epfd, libc::EPOLL_CTL_ADD, fd, &mut event) };
            if ret < 0 {
                return Err(io::Error::last_os_error());
            }
            self.live_sources = self.live_sources.saturating_add(1);
            Ok(SourceKey { index, generation })
        }

        pub fn reregister(&mut self, key: SourceKey, interest: Interest) -> io::Result<()> {
            let slot_index = key.index as usize;
            let Some(slot) = self.slab.get_mut(slot_index) else {
                return Err(io::Error::other("Reactor: source went stale"));
            };
            if slot.generation != key.generation {
                return Err(io::Error::other("Reactor: source went stale"));
            }
            if slot.interest == interest {
                return Ok(());
            }
            let mut events: u32 = libc::EPOLLET as u32;
            if interest.wants_read() {
                events |= libc::EPOLLIN as u32;
            }
            if interest.wants_write() {
                events |= libc::EPOLLOUT as u32;
            }
            let mut event = libc::epoll_event {
                events,
                u64: pack_udata(key.index, key.generation),
            };
            let ret =
                unsafe { libc::epoll_ctl(self.epfd, libc::EPOLL_CTL_MOD, slot.fd, &mut event) };
            if ret < 0 {
                return Err(io::Error::last_os_error());
            }
            slot.interest = interest;
            Ok(())
        }

        pub fn deregister(&mut self, key: SourceKey) -> io::Result<()> {
            let slot_index = key.index as usize;
            let Some(slot) = self.slab.get_mut(slot_index) else {
                return Ok(());
            };
            if slot.generation != key.generation {
                return Ok(());
            }
            let fd = slot.fd;
            // SAFETY: epoll_ctl DEL with valid fd; ignore errors (fd may be closed).
            let _ = unsafe {
                libc::epoll_ctl(self.epfd, libc::EPOLL_CTL_DEL, fd, core::ptr::null_mut())
            };
            slot.generation = 0;
            slot.fd = -1;
            slot.interest = Interest::Read;
            slot.read_waker = None;
            slot.write_waker = None;
            slot.read_ready_epoch = 0;
            slot.write_ready_epoch = 0;
            self.free_head = Some(key.index);
            self.live_sources = self.live_sources.saturating_sub(1);
            Ok(())
        }

        /// see kqueue variant; same semantics.
        pub fn set_read_waker(&mut self, key: SourceKey, waker: Waker) -> bool {
            let Some(slot) = self.slab.get_mut(key.index as usize) else {
                return false;
            };
            if slot.generation != key.generation {
                return false;
            }
            slot.read_waker = Some(waker);
            true
        }

        pub fn set_write_waker(&mut self, key: SourceKey, waker: Waker) -> bool {
            let Some(slot) = self.slab.get_mut(key.index as usize) else {
                return false;
            };
            if slot.generation != key.generation {
                return false;
            }
            slot.write_waker = Some(waker);
            true
        }

        /// hot-path waker registration. clones only if the stored waker does
        /// not already `will_wake` the same task. matches the kqueue variant.
        #[inline]
        pub fn register_read_waker_ref(&mut self, key: SourceKey, waker: &Waker) -> bool {
            let Some(slot) = self.slab.get_mut(key.index as usize) else {
                return false;
            };
            if slot.generation != key.generation {
                return false;
            }
            match &slot.read_waker {
                Some(existing) if existing.will_wake(waker) => return true,
                _ => {}
            }
            slot.read_waker = Some(waker.clone());
            true
        }

        #[inline]
        pub fn register_write_waker_ref(&mut self, key: SourceKey, waker: &Waker) -> bool {
            let Some(slot) = self.slab.get_mut(key.index as usize) else {
                return false;
            };
            if slot.generation != key.generation {
                return false;
            }
            match &slot.write_waker {
                Some(existing) if existing.will_wake(waker) => return true,
                _ => {}
            }
            slot.write_waker = Some(waker.clone());
            true
        }

        #[inline]
        pub fn read_ready_epoch(&self, key: SourceKey) -> Option<u32> {
            let slot = self.slab.get(key.index as usize)?;
            (slot.generation == key.generation).then_some(slot.read_ready_epoch)
        }

        pub fn turn(&mut self, timeout: Option<Duration>) -> io::Result<usize> {
            let timeout_ms: i32 = match timeout {
                None => -1,
                Some(duration) => {
                    let millis = duration.as_millis();
                    if millis > i32::MAX as u128 {
                        i32::MAX
                    } else {
                        millis as i32
                    }
                }
            };
            // SAFETY: epfd valid, events_buf valid &mut [epoll_event].
            let count = unsafe {
                libc::epoll_wait(
                    self.epfd,
                    self.events_buf.as_mut_ptr(),
                    self.events_buf.len() as i32,
                    timeout_ms,
                )
            };
            if count < 0 {
                let err = io::Error::last_os_error();
                if err.raw_os_error() == Some(libc::EINTR) {
                    return Ok(0);
                }
                return Err(err);
            }
            #[cfg(feature = "runtime-prime-reactor-trace")]
            if count == 0 {
                crate::trace::record_reactor_timeout();
            }
            let mut fired = 0;
            for event in &self.events_buf[..count as usize] {
                if event.u64 == WAKE_COOKIE {
                    #[cfg(feature = "runtime-prime-reactor-trace")]
                    crate::trace::record_reactor_wakeup_event();
                    // drain the eventfd (read u64) so it doesn't re-fire on
                    // the next epoll_wait. EFD_NONBLOCK + EPOLLET means we
                    // can do this once per wake batch.
                    let mut sink: u64 = 0;
                    let _ = unsafe {
                        libc::read(
                            self.eventfd,
                            (&mut sink as *mut u64).cast(),
                            core::mem::size_of::<u64>(),
                        )
                    };
                    continue;
                }
                let (index, generation) = unpack_udata(event.u64);
                let Some(slot) = self.slab.get_mut(index as usize) else {
                    #[cfg(feature = "runtime-prime-reactor-trace")]
                    crate::trace::record_reactor_stale_event();
                    continue;
                };
                if slot.generation != generation {
                    #[cfg(feature = "runtime-prime-reactor-trace")]
                    crate::trace::record_reactor_stale_event();
                    continue;
                }
                // wake_by_ref: keep the waker in the slot so the next
                // register_*_waker_ref's will_wake check elides the clone.
                // (see kqueue variant for full rationale.) Note: EPOLLIN
                // and EPOLLOUT can BOTH be set on the same event; we fire
                // each respective waker in sequence, NOT as an else-if.
                if (event.events & libc::EPOLLIN as u32) != 0 {
                    if let Some(waker) = slot.read_waker.as_ref() {
                        slot.read_ready_epoch = super::bump_generation(slot.read_ready_epoch);
                        #[cfg(feature = "runtime-prime-reactor-trace")]
                        crate::trace::record_event_ready();
                        #[cfg(feature = "runtime-prime-reactor-trace")]
                        crate::trace::record_waker_start();
                        waker.wake_by_ref();
                        #[cfg(feature = "runtime-prime-reactor-trace")]
                        crate::trace::record_waker_end();
                        fired += 1;
                    } else {
                        #[cfg(feature = "runtime-prime-reactor-trace")]
                        crate::trace::record_reactor_ignored_read();
                    }
                }
                if (event.events & libc::EPOLLOUT as u32) != 0 {
                    if let Some(waker) = slot.write_waker.as_ref() {
                        slot.write_ready_epoch = super::bump_generation(slot.write_ready_epoch);
                        waker.wake_by_ref();
                        fired += 1;
                    } else {
                        #[cfg(feature = "runtime-prime-reactor-trace")]
                        crate::trace::record_reactor_ignored_write();
                    }
                }
                if (event.events & (libc::EPOLLIN | libc::EPOLLOUT) as u32) == 0 {
                    #[cfg(feature = "runtime-prime-reactor-trace")]
                    crate::trace::record_reactor_unknown_event();
                }
            }
            Ok(fired)
        }
    }

    impl Drop for Reactor {
        fn drop(&mut self) {
            // SAFETY: epfd and eventfd were created by us.
            unsafe {
                libc::close(self.eventfd);
                libc::close(self.epfd);
            }
        }
    }

    fn pack_udata(index: u32, generation: u32) -> u64 {
        ((generation as u64) << 32) | (index as u64)
    }

    fn unpack_udata(raw: u64) -> (u32, u32) {
        (raw as u32, (raw >> 32) as u32)
    }
}

#[cfg(target_os = "linux")]
pub use epoll::Reactor;
#[cfg(target_os = "macos")]
pub use kqueue::Reactor;

// SAFETY: Reactor owns its kqueue/epoll fd. Internal state (slab, event
// buffer with raw `*mut c_void` udata) is single-thread-owned; we move the
// Reactor to its worker thread once at launch and it never crosses threads
// after that. The raw udata pointers are not dereferenced — they carry a
// packed (gen, index) tuple, treated as `u64`.
#[cfg(any(target_os = "macos", target_os = "linux"))]
unsafe impl Send for Reactor {}
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
compile_error!("runtime-prime-reactor requires macOS (kqueue) or Linux (epoll)");

#[cfg(all(test, any(target_os = "macos", target_os = "linux")))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::os::fd::AsRawFd;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::Wake;

    struct CountingWaker(Arc<AtomicUsize>);
    impl Wake for CountingWaker {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::AcqRel);
        }
    }

    fn waker_for(count: &Arc<AtomicUsize>) -> Waker {
        Arc::new(CountingWaker(count.clone())).into()
    }

    fn socketpair() -> (
        std::os::unix::net::UnixStream,
        std::os::unix::net::UnixStream,
    ) {
        std::os::unix::net::UnixStream::pair().expect("socketpair")
    }

    fn set_nonblocking(stream: &std::os::unix::net::UnixStream) {
        stream.set_nonblocking(true).expect("nonblocking");
    }

    #[test]
    fn register_then_write_wakes_read_waker() {
        let mut reactor = Reactor::new().expect("reactor");
        let (left, mut right) = socketpair();
        set_nonblocking(&left);
        let key = reactor
            .register(left.as_raw_fd(), Interest::Read)
            .expect("register");
        let count = Arc::new(AtomicUsize::new(0));
        assert!(reactor.set_read_waker(key, waker_for(&count)));
        // before any data, turn(0) is non-blocking → no events.
        let fired = reactor.turn(Some(Duration::from_millis(0))).expect("turn0");
        assert_eq!(fired, 0);
        // write to the other side; reader becomes readable.
        right.write_all(b"x").expect("write");
        let fired = reactor.turn(Some(Duration::from_secs(1))).expect("turn");
        assert!(fired >= 1, "expected at least 1 fire, got {fired}");
        assert!(count.load(Ordering::Acquire) >= 1);
    }

    #[test]
    fn turn_with_zero_timeout_returns_immediately_when_idle() {
        let mut reactor = Reactor::new().expect("reactor");
        let started = std::time::Instant::now();
        let fired = reactor.turn(Some(Duration::from_millis(0))).expect("turn");
        let elapsed = started.elapsed();
        assert_eq!(fired, 0);
        assert!(elapsed < Duration::from_millis(100), "took {elapsed:?}");
    }

    #[test]
    fn deregister_stops_future_fires() {
        let mut reactor = Reactor::new().expect("reactor");
        let (left, mut right) = socketpair();
        set_nonblocking(&left);
        let key = reactor
            .register(left.as_raw_fd(), Interest::Read)
            .expect("register");
        let count = Arc::new(AtomicUsize::new(0));
        reactor.set_read_waker(key, waker_for(&count));
        reactor.deregister(key).expect("deregister");
        right.write_all(b"x").expect("write");
        let fired = reactor.turn(Some(Duration::from_millis(50))).expect("turn");
        assert_eq!(fired, 0);
        assert_eq!(count.load(Ordering::Acquire), 0);
    }

    #[test]
    fn stale_key_after_deregister_is_ignored() {
        let mut reactor = Reactor::new().expect("reactor");
        let (left, _right) = socketpair();
        set_nonblocking(&left);
        let key = reactor
            .register(left.as_raw_fd(), Interest::Read)
            .expect("register");
        reactor.deregister(key).expect("deregister");
        let count = Arc::new(AtomicUsize::new(0));
        // setting waker on stale key must return false, not panic.
        assert!(!reactor.set_read_waker(key, waker_for(&count)));
    }

    #[test]
    fn two_sources_each_get_own_waker() {
        let mut reactor = Reactor::new().expect("reactor");
        let (left_a, mut right_a) = socketpair();
        let (left_b, mut right_b) = socketpair();
        set_nonblocking(&left_a);
        set_nonblocking(&left_b);
        let key_a = reactor
            .register(left_a.as_raw_fd(), Interest::Read)
            .unwrap();
        let key_b = reactor
            .register(left_b.as_raw_fd(), Interest::Read)
            .unwrap();
        let count_a = Arc::new(AtomicUsize::new(0));
        let count_b = Arc::new(AtomicUsize::new(0));
        reactor.set_read_waker(key_a, waker_for(&count_a));
        reactor.set_read_waker(key_b, waker_for(&count_b));
        right_a.write_all(b"a").expect("write a");
        let _ = reactor.turn(Some(Duration::from_secs(1))).expect("turn");
        // drain socket to clear readiness; otherwise edge-triggered/oneshot
        // semantics may not re-fire.
        let mut buf = [0u8; 1];
        let _ = std::io::Read::read(&mut &left_a, &mut buf);
        assert!(count_a.load(Ordering::Acquire) >= 1);
        assert_eq!(count_b.load(Ordering::Acquire), 0);
        right_b.write_all(b"b").expect("write b");
        let _ = reactor.turn(Some(Duration::from_secs(1))).expect("turn");
        assert!(count_b.load(Ordering::Acquire) >= 1);
    }

    #[test]
    fn register_writable_socket_initially_ready() {
        let mut reactor = Reactor::new().expect("reactor");
        let (left, _right) = socketpair();
        set_nonblocking(&left);
        let key = reactor
            .register(left.as_raw_fd(), Interest::Write)
            .expect("register");
        let count = Arc::new(AtomicUsize::new(0));
        reactor.set_write_waker(key, waker_for(&count));
        let fired = reactor.turn(Some(Duration::from_millis(50))).expect("turn");
        assert!(
            fired >= 1,
            "an empty writable socket should fire immediately"
        );
    }
}
