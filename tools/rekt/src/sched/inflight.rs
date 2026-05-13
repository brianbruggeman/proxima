use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

/// bounded in-flight gate. an open-loop driver must cap concurrent in-flight
/// work without ever slowing the arrival schedule to shed it, so overflow is a
/// rejection the caller turns into a timeout, never a block.
#[derive(Debug)]
pub struct InFlight {
    cur: AtomicU32,
    max: u32,
}

impl InFlight {
    #[must_use]
    pub fn new(max: u32) -> Arc<Self> {
        Arc::new(Self { cur: AtomicU32::new(0), max })
    }

    #[must_use]
    pub fn try_enter(self: &Arc<Self>) -> Option<Permit> {
        let mut cur = self.cur.load(Ordering::Relaxed);
        loop {
            if cur >= self.max {
                return None;
            }
            match self
                .cur
                .compare_exchange_weak(cur, cur + 1, Ordering::AcqRel, Ordering::Relaxed)
            {
                Ok(_) => {
                    return Some(Permit { gate: Arc::clone(self) });
                }
                Err(observed) => cur = observed,
            }
        }
    }

    #[must_use]
    pub fn current(&self) -> u32 {
        self.cur.load(Ordering::Relaxed)
    }

    #[must_use]
    pub fn max(&self) -> u32 {
        self.max
    }
}

#[derive(Debug)]
pub struct Permit {
    gate: Arc<InFlight>,
}

impl Drop for Permit {
    fn drop(&mut self) {
        self.gate.cur.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    // tests assert on known states; unwrap/expect are the clearer failure here
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::thread;

    use super::*;

    #[proxima::test]
    async fn admits_up_to_max() {
        let gate = InFlight::new(2);
        let first = gate.try_enter();
        let second = gate.try_enter();
        assert!(first.is_some() && second.is_some());
        assert_eq!(gate.current(), 2);
    }

    #[proxima::test]
    async fn overflow_is_rejected_not_blocked() {
        let gate = InFlight::new(1);
        let _held = gate.try_enter().expect("first admits");
        assert!(gate.try_enter().is_none(), "overflow returns None, never blocks");
    }

    #[proxima::test]
    async fn drop_frees_a_slot() {
        let gate = InFlight::new(1);
        {
            let _held = gate.try_enter().expect("admit");
            assert_eq!(gate.current(), 1);
        }
        assert_eq!(gate.current(), 0);
        assert!(gate.try_enter().is_some(), "slot freed after drop");
    }

    #[proxima::test]
    async fn zero_cap_admits_nothing() {
        let gate = InFlight::new(0);
        assert!(gate.try_enter().is_none());
    }

    #[proxima::test]
    async fn concurrent_enters_never_exceed_max() {
        let gate = InFlight::new(8);
        let mut handles = Vec::new();
        for _ in 0..16 {
            let gate = Arc::clone(&gate);
            handles.push(thread::spawn(move || gate.try_enter()));
        }
        let permits: Vec<_> = handles
            .into_iter()
            .filter_map(|handle| handle.join().expect("thread joins"))
            .collect();
        assert!(permits.len() <= 8, "never admit past the cap under contention");
        assert_eq!(gate.current() as usize, permits.len());
    }
}
