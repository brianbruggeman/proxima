use thiserror::Error;

/// Failures bringing up or driving a dpdk port. The `i32` codes are dpdk's
/// negative errno returns, surfaced verbatim so the cause is recoverable from
/// the log alone.
#[derive(Debug, Error)]
pub enum DpdkError {
    #[error("an eal argument contained an interior nul byte")]
    ArgNul,

    #[error("too many eal arguments to fit in a c_int argc")]
    TooManyArgs,

    #[error("rte_eal_init failed (code {0})")]
    EalInit(i32),

    #[error("rte_pktmbuf_pool_create returned null (out of memory or hugepages)")]
    PoolCreate,

    #[error("no dpdk ports available; pass a --vdev (e.g. net_tap0) on the eal command line")]
    NoPorts,

    #[error("requested port {requested} but only {available} present")]
    PortOutOfRange { requested: u16, available: u16 },

    #[error("rte_eth port init failed (code {0})")]
    PortInit(i32),

    #[error("rte_eth_macaddr_get failed (code {0})")]
    MacAddr(i32),

    #[error(
        "dpdk abi self-check failed: {0} (built against dpdk that differs from the runtime .so?)"
    )]
    AbiSelfCheck(&'static str),
}
