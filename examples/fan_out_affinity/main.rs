#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Fan-out AFFINITY: route one record to ONE of N partitions by a key, the way
//! Kafka's producer partitioner does — the same key always lands on the same
//! partition, so a whole customer's (or trace's) stream stays together.
//!
//! `proxima_primitives::pipe::FanOut` broadcasts one input to ALL arms — the
//! "everyone" distribution. Affinity is the OTHER distribution: route to one
//! arm by key. This example builds it consumer-side, EXTENDING the pattern
//! without adding anything to the library — which is exactly what `examples/`
//! are for. It also shows the line the algebra draws through the middle of it:
//!
//!   - keying is a PIPE. It reads the record to derive a routing key
//!     (`Record -> u64`), so the record passes THROUGH it. Reading the record
//!     is legal here — that is what makes it a pipe.
//!   - choosing the partition is a STRATEGY. It sees only the key, never the
//!     record, and answers one control question: which partition. A plain
//!     function. `HashAffinity`, `RoundRobin`, and a `Sticky` the library never
//!     heard of all plug into the same seam — extend, don't add.
//!
//! Run: `cargo run --example fan_out_affinity`

use core::cell::Cell;
use core::convert::Infallible;
use core::future::Future;
use core::task::{Context, Poll, Waker};

use proxima_primitives::pipe::Pipe;

const PARTITIONS: usize = 3;

fn main() {
    let records = order_stream();

    // affinity: every order for a customer lands on ONE partition, chosen by
    // hashing the customer key — the Kafka guarantee.
    let by_affinity = route(&records, &HashAffinity);
    print_partitions("HashAffinity (key -> hash -> one partition)", &by_affinity);
    assert_affinity_holds(&records, &by_affinity);

    // round-robin ignores the key entirely, so the same customer scatters —
    // the contrast that shows affinity is doing real work.
    let by_round_robin = route(&records, &RoundRobin::new());
    print_partitions("RoundRobin (key ignored, scatter)", &by_round_robin);
    assert_customer_scatters(&records, &by_round_robin);

    // a strategy the library never heard of: stay on one partition for a run
    // of records, then advance (Kafka's sticky partitioner). It plugs into the
    // same seam with zero library change — the whole point of "extend, not add".
    let by_sticky = route(&records, &Sticky::new(4));
    print_partitions("Sticky (caller-defined, library never heard of it)", &by_sticky);

    println!("\naffinity proven: same key -> same partition; the strategy never saw a record.");
}

/// The keying seam — a PIPE, because it reads the record. `Record -> u64`: the
/// record passes through and comes out a routing key. Everything record-shaped
/// a router could ever need is funnelled through this one pipe, which is why
/// the strategy below never has to.
struct PartitionKey;

impl Pipe for PartitionKey {
    type In = Record;
    type Out = u64;
    type Err = Infallible;

    fn call(&self, record: Record) -> impl Future<Output = Result<u64, Infallible>> {
        let key = fnv1a(record.customer.as_bytes());
        async move { Ok(key) }
    }
}

/// The distribution seam — a STRATEGY, because it never sees the record. It
/// answers one control question, "which partition", from the key alone. The
/// signature is the proof: `key`, not `Record`. Nothing here can be widened
/// into reading the payload without becoming a pipe instead.
trait Distribute {
    fn partition(&self, key: u64, partitions: usize) -> usize;
}

/// Kafka's key partitioner: `hash(key) % partitions`. Same key, same partition,
/// forever — the affinity guarantee. Stateless.
struct HashAffinity;

impl Distribute for HashAffinity {
    fn partition(&self, key: u64, partitions: usize) -> usize {
        (key % partitions as u64) as usize
    }
}

/// Keyless, fair: walk the partitions in turn. Uses `&self` state (the cursor),
/// never the key — so records for one customer spray across all partitions.
struct RoundRobin {
    cursor: Cell<usize>,
}

impl RoundRobin {
    fn new() -> Self {
        Self {
            cursor: Cell::new(0),
        }
    }
}

impl Distribute for RoundRobin {
    fn partition(&self, _key: u64, partitions: usize) -> usize {
        let index = self.cursor.get() % partitions;
        self.cursor.set(index + 1);
        index
    }
}

/// Kafka's sticky partitioner, defined entirely by the caller: stay on one
/// partition for a run of `batch` records (fat batches, fewer flushes), then
/// advance. Payload-blind and key-blind — pure `&self` state. The library has
/// no idea this exists; it plugs in anyway.
struct Sticky {
    batch: usize,
    partition: Cell<usize>,
    used: Cell<usize>,
}

impl Sticky {
    fn new(batch: usize) -> Self {
        Self {
            batch,
            partition: Cell::new(0),
            used: Cell::new(0),
        }
    }
}

impl Distribute for Sticky {
    fn partition(&self, _key: u64, partitions: usize) -> usize {
        if self.used.get() == self.batch {
            self.partition.set((self.partition.get() + 1) % partitions);
            self.used.set(0);
        }
        self.used.set(self.used.get() + 1);
        self.partition.get()
    }
}

/// Route every record: key it through the PIPE, then hand only the key to the
/// STRATEGY. The router itself is the composition — a keying pipe feeding a
/// distribution strategy — and never lets the strategy touch the record.
fn route(records: &[Record], strategy: &dyn Distribute) -> [Vec<Record>; PARTITIONS] {
    let keyer = PartitionKey;
    let mut partitions: [Vec<Record>; PARTITIONS] = Default::default();
    for record in records {
        let key = block_on_ready(keyer.call(record.clone())).expect("keying is infallible");
        let index = strategy.partition(key, PARTITIONS);
        partitions[index].push(record.clone());
    }
    partitions
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Record {
    customer: &'static str,
    order: u64,
}

fn order_stream() -> Vec<Record> {
    let customers = ["ada", "linus", "grace", "dennis", "ada", "grace", "linus", "ada"];
    customers
        .into_iter()
        .enumerate()
        .map(|(order, customer)| Record {
            customer,
            order: order as u64,
        })
        .collect()
}

// FNV-1a: a small, dependency-free, deterministic hash — enough to demonstrate
// key -> partition affinity. A real deployment would reach for the same hash on
// every producer so the mapping agrees fleet-wide (Kafka uses murmur2 for that).
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn print_partitions(title: &str, partitions: &[Vec<Record>; PARTITIONS]) {
    println!("\n{title}");
    for (index, partition) in partitions.iter().enumerate() {
        let labels: Vec<String> = partition
            .iter()
            .map(|record| format!("{}#{}", record.customer, record.order))
            .collect();
        println!("  partition {index}: {labels:?}");
    }
}

// affinity holds when every record for a given customer sits in exactly one
// partition, and that partition is the one the hash names — no customer's
// stream is ever split.
fn assert_affinity_holds(records: &[Record], partitions: &[Vec<Record>; PARTITIONS]) {
    for record in records {
        let expected = (fnv1a(record.customer.as_bytes()) % PARTITIONS as u64) as usize;
        let landed = partitions
            .iter()
            .position(|partition| partition.contains(record))
            .expect("every record lands somewhere");
        assert_eq!(
            landed, expected,
            "{}#{} must land on its hash partition {expected}, not {landed}",
            record.customer, record.order
        );
    }
}

// the contrast: with a key-blind strategy, at least one customer's records span
// more than one partition — proof the affinity above was the strategy's doing.
fn assert_customer_scatters(records: &[Record], partitions: &[Vec<Record>; PARTITIONS]) {
    let scattered = records.iter().any(|record| {
        let homes = partitions
            .iter()
            .filter(|partition| partition.iter().any(|other| other.customer == record.customer))
            .count();
        homes > 1
    });
    assert!(
        scattered,
        "round-robin must scatter at least one customer across partitions"
    );
}

// every future here resolves on its first poll (a plain hash, no real I/O), so
// a one-shot poll is a legitimate block_on — no executor needed to prove the
// pattern, same as the sibling fan_out example.
fn block_on_ready<F: Future>(future: F) -> F::Output {
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    let mut future = core::pin::pin!(future);
    match future.as_mut().poll(&mut context) {
        Poll::Ready(value) => value,
        Poll::Pending => unreachable!("fan_out_affinity example futures resolve on first poll"),
    }
}
