//! Prime-native TCP `StreamListener` / `StreamUpstream` over dpdk. The driver is
//! [`crate::frame_stack`]; this module only supplies the dpdk port as its three
//! frame pipes ([`super::frame_pipes`]) and the net_tap bring-up.
//!
//! EAL is process-global, so there is one dpdk listener or upstream per process.

use super::frame_pipes::{DpdkMint, DpdkRx, DpdkTx, Hardware};
use super::DpdkError;
use crate::frame_stack::{Driver, FrameConnection, FrameListener, FrameUpstream};
use std::net::SocketAddrV4;

pub type DpdkStreamListener = FrameListener<DpdkRx, DpdkTx, DpdkMint>;
pub type DpdkStreamConnection = FrameConnection<DpdkRx, DpdkTx, DpdkMint>;
pub type DpdkStreamUpstream = FrameUpstream<DpdkRx, DpdkTx, DpdkMint>;

fn driver(
    pool_name: &str,
    local: SocketAddrV4,
    iface: &str,
    pmd_dir: &str,
    initial_sequence: u32,
) -> Result<Driver<DpdkRx, DpdkTx, DpdkMint>, DpdkError> {
    let (hardware, mac) = Hardware::bring_up(pool_name, iface, pmd_dir)?;
    Ok(Driver::new(Hardware::pipes(hardware), mac, local, initial_sequence))
}

impl DpdkStreamListener {
    /// Bring up a net_tap vdev and accept TCP connections on `bind`.
    ///
    /// # Errors
    /// Propagates EAL / pool / port bring-up failures as [`DpdkError`].
    pub fn bind(bind: SocketAddrV4, iface: &str, pmd_dir: &str) -> Result<Self, DpdkError> {
        let driver = driver("pnd_tcp_pool", bind, iface, pmd_dir, 0x1000)?;
        Ok(Self::new(driver, bind))
    }
}

impl DpdkStreamUpstream {
    /// Bring up a net_tap vdev and prepare to connect from `local` to `peer`.
    ///
    /// # Errors
    /// Propagates EAL / pool / port bring-up failures as [`DpdkError`].
    pub fn bind(
        local: SocketAddrV4,
        peer: SocketAddrV4,
        iface: &str,
        pmd_dir: &str,
    ) -> Result<Self, DpdkError> {
        let driver = driver("pnd_up_pool", local, iface, pmd_dir, 0x2000)?;
        Ok(Self::new(driver, peer))
    }
}
