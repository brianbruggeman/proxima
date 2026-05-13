//! single-producer push/drain hot loop + isolated, non-inlined try_send/try_recv
//! wrappers so the dynamic-vs-alloc hot path can be disassembled (objdump
//! --disassemble=recv_dynamic etc.) and root-caused instruction-by-instruction
//! instead of guessed. run: `inbox_hotloop alloc` | `inbox_hotloop dynamic`.

#![cfg(all(
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-inbox-dynamic"
))]

use std::env;
use std::hint::black_box;

use prime::core::inbox as alloc_inbox;
use prime::core::inbox_dynamic::{self, InboxDynamicConfig, ReleasePolicy, channel as dyn_channel};

const LANE_CAP: usize = 1024;
const ITEMS: usize = 400_000_000;

#[unsafe(no_mangle)]
#[inline(never)]
pub fn recv_alloc(consumer: &alloc_inbox::Consumer<u64>) -> bool {
    consumer.try_recv().is_ok()
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn recv_dynamic(consumer: &inbox_dynamic::Consumer<u64>) -> bool {
    consumer.try_recv().is_ok()
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn send_alloc(producer: &alloc_inbox::Producer<u64>, value: u64) -> bool {
    producer.try_send(value).is_ok()
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn send_dynamic(producer: &inbox_dynamic::Producer<u64>, value: u64) -> bool {
    producer.try_send(value).is_ok()
}

fn drive(mut try_send: impl FnMut(u64) -> bool, mut try_recv: impl FnMut() -> bool) {
    let mut count = 0usize;
    let mut index = 0u64;
    while count < ITEMS {
        let batch = LANE_CAP.min(ITEMS - count);
        let mut pushed = 0;
        while pushed < batch {
            if try_send(black_box(index)) {
                index += 1;
                pushed += 1;
            } else {
                break;
            }
        }
        let mut drained = 0;
        while drained < pushed {
            if try_recv() {
                drained += 1;
                count += 1;
            } else {
                break;
            }
        }
    }
    black_box(count);
}

fn main() {
    match env::args().nth(1).as_deref() {
        Some("dynamic") => {
            let config = InboxDynamicConfig {
                floor: 1,
                ceiling: 64,
                release: ReleasePolicy::Never,
                lane_capacity: LANE_CAP,
            };
            let (producer, consumer) = dyn_channel::<u64>(&config);
            drive(
                |value| send_dynamic(&producer, value),
                || recv_dynamic(&consumer),
            );
        }
        _ => {
            let (producer, consumer) = alloc_inbox::channel::<u64>(64, LANE_CAP);
            drive(
                |value| send_alloc(&producer, value),
                || recv_alloc(&consumer),
            );
        }
    }
}
