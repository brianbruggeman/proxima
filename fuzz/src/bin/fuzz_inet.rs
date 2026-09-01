//! Target 1: `proxima-protocols::inet` -- Ethernet/IPv4/UDP/TCP frame
//! decode. Builds on `proxima_protocols::inet::{ethernet, ipv4, tcp, udp}`,
//! each a borrowed-view `parse(bytes: &[u8]) -> Result<Self, DecodeError>`
//! (`proxima-protocols/src/inet/{ethernet,ipv4,tcp,udp}.rs`). Arbitrary
//! bytes must never panic -- reject or parse, always.
//!
//! Run: `cargo run --bin fuzz_inet`

use proxima_fuzz_harness::{run_no_panic_sweep, SweepReport};
use proxima_protocols::inet::{ethernet::EthernetFrame, ipv4::Ipv4Header, tcp::TcpHeader, udp::UdpHeader};

const SEED: u64 = 0x1E7A_1234_5678_9ABC;
const ITERATIONS: usize = 150_000;
const MAX_LEN: usize = 128;

fn main() {
    println!("fuzz_inet: proxima-protocols::inet::{{ethernet,ipv4,tcp,udp}} no-panic sweep");
    let reports = run(ITERATIONS);
    for report in &reports {
        report.print_line();
    }
}

fn run(iterations: usize) -> [SweepReport; 4] {
    [
        run_no_panic_sweep("inet::ethernet", SEED, iterations, MAX_LEN, |bytes| {
            EthernetFrame::parse(bytes).is_ok()
        }),
        run_no_panic_sweep("inet::ipv4", SEED.wrapping_add(1), iterations, MAX_LEN, |bytes| {
            Ipv4Header::parse(bytes).is_ok()
        }),
        run_no_panic_sweep("inet::udp", SEED.wrapping_add(2), iterations, MAX_LEN, |bytes| {
            UdpHeader::parse(bytes).is_ok()
        }),
        run_no_panic_sweep("inet::tcp", SEED.wrapping_add(3), iterations, MAX_LEN, |bytes| {
            TcpHeader::parse(bytes).is_ok()
        }),
    ]
}

#[cfg(test)]
mod tests {
    use super::run;

    #[test]
    fn no_panic_smoke() {
        for report in run(3_000) {
            assert_eq!(report.iterations, 3_000);
        }
    }
}
