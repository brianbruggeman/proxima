// Bring the dpdk stack up in our own code (not testpmd) over a net_tap vdev,
// then run a bounded L2-echo poll loop to prove the RX/TX path moves frames.
// Build/run on a dpdk host:  cargo run --features dpdk --bin tap-bringup  (sudo)

#[cfg(feature = "dpdk")]
fn main() -> std::process::ExitCode {
    imp::main()
}

#[cfg(not(feature = "dpdk"))]
fn main() {
    eprintln!("proxima-net-dpdk: build with --features dpdk on a dpdk host");
}

#[cfg(feature = "dpdk")]
mod imp {
    use proxima_net::dpdk::{Eal, Mempool, Port, RawMbuf, port};
    use std::env;
    use std::process::ExitCode;
    use std::time::{Duration, Instant};

    const BURST: usize = 32;

    fn run() -> Result<(), Box<dyn std::error::Error>> {
        let iface = env::var("PROXIMA_TAP_IFACE").unwrap_or_else(|_| "ptap0".to_string());
        let pmd_dir =
            env::var("PROXIMA_PMD_DIR").unwrap_or_else(|_| "/usr/lib/dpdk/pmds-26.0".to_string());
        let run_secs: u64 = env::var("PROXIMA_RUN_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);

        let vdev = format!("--vdev=net_tap0,iface={iface}");
        let eal_args = ["tap-bringup", "-l", "0", "--no-pci", "-d", &pmd_dir, &vdev];

        println!("eal init: {eal_args:?}");
        let _eal = Eal::init(&eal_args)?;

        let pool = Mempool::create("pndtap_pool", 8192, -1)?;
        println!("ports probed: {}", port::port_count());

        let dev = Port::init(0, &pool)?;
        let mac = dev.mac()?;
        println!(
            "port {} up: mac {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
            dev.id(),
            mac[0],
            mac[1],
            mac[2],
            mac[3],
            mac[4],
            mac[5],
        );

        println!("polling rx for {run_secs}s (echo + tx)...");
        let deadline = Instant::now() + Duration::from_secs(run_secs);
        let mut rx_total: usize = 0;
        let mut tx_total: usize = 0;
        let mut bufs: [RawMbuf; BURST] = [core::ptr::null_mut(); BURST];

        while Instant::now() < deadline {
            let received = usize::from(dev.rx_burst(&mut bufs));
            if received == 0 {
                continue;
            }
            rx_total += received;
            for &mbuf in &bufs[..received] {
                unsafe { port::eth_swap(mbuf) };
            }
            let sent = usize::from(dev.tx_burst(&mut bufs[..received]));
            tx_total += sent;
            for &mbuf in &bufs[sent..received] {
                unsafe { port::free(mbuf) };
            }
        }

        println!("done: rx_total={rx_total} tx_total={tx_total}");
        if rx_total == 0 {
            println!("note: no frames seen — bring the kernel tap up and generate traffic");
        }
        Ok(())
    }

    pub fn main() -> ExitCode {
        match run() {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("tap-bringup failed: {err}");
                ExitCode::FAILURE
            }
        }
    }
}
