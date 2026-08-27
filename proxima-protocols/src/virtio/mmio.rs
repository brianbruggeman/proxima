//! virtio-mmio transport register block (VIRTIO 1.2 spec §4.2.2): the
//! legacy-free, memory-mapped register layout a driver polls/pokes to
//! discover a device, negotiate features, and stand up virtqueues, before
//! any ring traffic (`super::avail`/`super::used`/`super::descriptor`) is
//! legal at all. This is the transport layer the module doc for
//! `super` names as sitting one layer above the ring codecs — the piece
//! that turns one raw `(offset, is_write, value)` register access into a
//! typed effect the host applies, mirroring how `super::status::Negotiation`
//! turns one status-byte write into a typed FSM transition.
//!
//! Every register is a fixed 32-bit little-endian word at a fixed offset
//! (spec §4.2.2 Table 4.1); there is no variable-length or streamed field
//! here, so [`MmioDevice::apply`] is a single match over `offset`, not a
//! cursor/state-machine walk the way `super::super::elf`'s whole-buffer
//! decode needs one. Fixed-shape, allocates nothing — `MAX_QUEUES` is the
//! caller-chosen const-generic capacity for the queue-config array
//! (per `slot-0/AGENTS.md` principle 12, never a hidden magic number).

use super::status::{DeviceStatus, Negotiation, NegotiationError};

/// `MagicValue` register (offset 0x000, read-only): ASCII `"virt"` read as
/// one little-endian `u32` (spec §4.2.2).
pub const MAGIC_VALUE: u32 = 0x7472_6976;

/// `Version` register (offset 0x004, read-only): 2 for the non-legacy,
/// non-transitional transport this codec models exclusively.
pub const TRANSPORT_VERSION: u32 = 2;

/// `DeviceID` value for the console device type (spec §5, Table 5.1).
pub const DEVICE_ID_CONSOLE: u32 = 3;

pub const REG_MAGIC_VALUE: u64 = 0x000;
pub const REG_VERSION: u64 = 0x004;
pub const REG_DEVICE_ID: u64 = 0x008;
pub const REG_VENDOR_ID: u64 = 0x00c;
pub const REG_DEVICE_FEATURES: u64 = 0x010;
pub const REG_DEVICE_FEATURES_SEL: u64 = 0x014;
pub const REG_DRIVER_FEATURES: u64 = 0x020;
pub const REG_DRIVER_FEATURES_SEL: u64 = 0x024;
pub const REG_QUEUE_SEL: u64 = 0x030;
pub const REG_QUEUE_NUM_MAX: u64 = 0x034;
pub const REG_QUEUE_NUM: u64 = 0x038;
pub const REG_QUEUE_READY: u64 = 0x044;
pub const REG_QUEUE_NOTIFY: u64 = 0x050;
pub const REG_INTERRUPT_STATUS: u64 = 0x060;
pub const REG_INTERRUPT_ACK: u64 = 0x064;
pub const REG_STATUS: u64 = 0x070;
pub const REG_QUEUE_DESC_LOW: u64 = 0x080;
pub const REG_QUEUE_DESC_HIGH: u64 = 0x084;
pub const REG_QUEUE_DRIVER_LOW: u64 = 0x090;
pub const REG_QUEUE_DRIVER_HIGH: u64 = 0x094;
pub const REG_QUEUE_DEVICE_LOW: u64 = 0x0a0;
pub const REG_QUEUE_DEVICE_HIGH: u64 = 0x0a4;
pub const REG_CONFIG_GENERATION: u64 = 0x0fc;

/// One raw register access, as the transport layer (`backend_macos.c`'s
/// data-abort decode, §4.2.2) recovers it from a trapped guest load/store:
/// the byte offset from the device's MMIO window base, whether it was a
/// write, and — for a write — the 32-bit value the guest stored. A read
/// carries `value: 0`, ignored by every read arm below.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MmioAccess {
    pub offset: u64,
    pub is_write: bool,
    pub value: u32,
}

/// What the transport must do in response to one applied [`MmioAccess`].
/// Every variant carries exactly the data the transport needs and nothing
/// it must re-derive — a read names the word to write back into the
/// guest's destination register; a queue-notify or queue-ready write names
/// which queue changed, so the transport does not have to re-read
/// [`MmioDevice`] state to find out what happened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmioEffect {
    /// A read register access: the value the transport writes back into the
    /// guest's destination register.
    ReadValue(u32),
    /// A write register access accepted with no further consequence — the
    /// register's new value has already been recorded in [`MmioDevice`].
    Applied,
    /// A `Status` register write drove the device-status FSM to this state.
    StatusTransition(DeviceStatus),
    /// A `QueueReady` register write changed the named queue's ready flag.
    QueueReady { queue: u16, ready: bool },
    /// A `QueueNotify` register write named the queue index the driver
    /// kicked — the transport walks that queue's avail ring next.
    QueueNotify { queue: u16 },
}

/// Why [`MmioDevice::apply`] rejected an access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmioError {
    /// No register is defined at this offset.
    UnknownRegister { offset: u64 },
    /// The register at this offset is read-only; the guest attempted a
    /// write.
    ReadOnlyRegister { offset: u64 },
    /// The register at this offset is write-only; the guest attempted a
    /// read.
    WriteOnlyRegister { offset: u64 },
    /// `QueueSel` named a queue index this device was not configured with.
    QueueSelectOutOfRange { queue: u16, queue_count: u16 },
    /// The `Status` register write was rejected by the device-status FSM.
    Negotiation(NegotiationError),
}

impl core::fmt::Display for MmioError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnknownRegister { offset } => {
                write!(formatter, "no mmio register defined at offset {offset:#x}")
            }
            Self::ReadOnlyRegister { offset } => {
                write!(formatter, "register at offset {offset:#x} is read-only")
            }
            Self::WriteOnlyRegister { offset } => {
                write!(formatter, "register at offset {offset:#x} is write-only")
            }
            Self::QueueSelectOutOfRange { queue, queue_count } => write!(
                formatter,
                "queue index {queue} is out of range for a device with {queue_count} queues"
            ),
            Self::Negotiation(inner) => write!(formatter, "{inner}"),
        }
    }
}

impl core::error::Error for MmioError {}

/// Per-queue configuration the driver assembles across several register
/// writes (`QueueNum`, `QueueReady`, the three split address-pair
/// registers) before the queue is legal to notify.
#[derive(Debug, Clone, Copy, Default)]
struct QueueConfig {
    num: u16,
    ready: bool,
    descriptor_address: u64,
    driver_address: u64,
    device_address: u64,
}

/// One virtio-mmio device's register-block state: the device-status FSM
/// (delegated to [`Negotiation`], never reimplemented here — reuse-first,
/// principle 1) plus the feature-selector and per-queue registers the FSM
/// does not own. `MAX_QUEUES` bounds the fixed-cap queue array; a
/// virtio-console device declares exactly 2 (receiveq, transmitq — spec
/// §5.3.2), so a caller sizes this generic for the device it is standing
/// up, never a baked-in constant.
#[derive(Debug, Clone)]
pub struct MmioDevice<const MAX_QUEUES: usize> {
    device_id: u32,
    negotiation: Negotiation,
    offered_features: u64,
    acked_features: u64,
    device_features_sel: u32,
    driver_features_sel: u32,
    queue_sel: u16,
    queue_num_max: u16,
    queues: [QueueConfig; MAX_QUEUES],
}

impl<const MAX_QUEUES: usize> MmioDevice<MAX_QUEUES> {
    /// A freshly reset device of the given type, offering `offered_features`
    /// (already including [`super::status::FEATURE_VERSION_1`] if the
    /// caller wants a modern-only device), with every queue's `QueueNumMax`
    /// fixed at `queue_num_max`.
    #[must_use]
    pub fn new(device_id: u32, queue_num_max: u16, offered_features: u64) -> Self {
        let mut negotiation = Negotiation::new();
        negotiation.offer_features(offered_features);
        Self {
            device_id,
            negotiation,
            offered_features,
            acked_features: 0,
            device_features_sel: 0,
            driver_features_sel: 0,
            queue_sel: 0,
            queue_num_max,
            queues: [QueueConfig::default(); MAX_QUEUES],
        }
    }

    #[must_use]
    pub fn status(&self) -> DeviceStatus {
        self.negotiation.status()
    }

    /// The fully assembled guest-physical descriptor-table address for
    /// queue `queue`, once both `QueueDescLow`/`QueueDescHigh` have been
    /// written — the transport reads this once [`MmioEffect::QueueReady`]
    /// names the queue live.
    #[must_use]
    pub fn queue_descriptor_address(&self, queue: u16) -> Option<u64> {
        self.queues
            .get(usize::from(queue))
            .map(|config| config.descriptor_address)
    }

    #[must_use]
    pub fn queue_driver_address(&self, queue: u16) -> Option<u64> {
        self.queues
            .get(usize::from(queue))
            .map(|config| config.driver_address)
    }

    #[must_use]
    pub fn queue_device_address(&self, queue: u16) -> Option<u64> {
        self.queues
            .get(usize::from(queue))
            .map(|config| config.device_address)
    }

    #[must_use]
    pub fn queue_size(&self, queue: u16) -> Option<u16> {
        self.queues.get(usize::from(queue)).map(|config| config.num)
    }

    #[must_use]
    pub fn queue_is_ready(&self, queue: u16) -> Option<bool> {
        self.queues.get(usize::from(queue)).map(|config| config.ready)
    }

    fn selected_queue_mut(&mut self) -> Result<&mut QueueConfig, MmioError> {
        let queue_count = self.queues.len() as u16;
        self.queues
            .get_mut(usize::from(self.queue_sel))
            .ok_or(MmioError::QueueSelectOutOfRange {
                queue: self.queue_sel,
                queue_count,
            })
    }

    /// Apply one register access, returning the effect the transport must
    /// carry out. `is_write` selects which half of a register's contract
    /// applies — most registers are legal in only one direction, so an
    /// access on the wrong side is [`MmioError::ReadOnlyRegister`] /
    /// [`MmioError::WriteOnlyRegister`] rather than silently ignored.
    pub fn apply(&mut self, access: MmioAccess) -> Result<MmioEffect, MmioError> {
        match (access.offset, access.is_write) {
            (REG_MAGIC_VALUE, false) => Ok(MmioEffect::ReadValue(MAGIC_VALUE)),
            (REG_MAGIC_VALUE, true) => Err(MmioError::ReadOnlyRegister { offset: access.offset }),

            (REG_VERSION, false) => Ok(MmioEffect::ReadValue(TRANSPORT_VERSION)),
            (REG_VERSION, true) => Err(MmioError::ReadOnlyRegister { offset: access.offset }),

            (REG_DEVICE_ID, false) => Ok(MmioEffect::ReadValue(self.device_id)),
            (REG_DEVICE_ID, true) => Err(MmioError::ReadOnlyRegister { offset: access.offset }),

            (REG_VENDOR_ID, false) => Ok(MmioEffect::ReadValue(0)),
            (REG_VENDOR_ID, true) => Err(MmioError::ReadOnlyRegister { offset: access.offset }),

            (REG_DEVICE_FEATURES, false) => {
                let word = if self.device_features_sel == 0 {
                    self.offered_features as u32
                } else {
                    (self.offered_features >> 32) as u32
                };
                Ok(MmioEffect::ReadValue(word))
            }
            (REG_DEVICE_FEATURES, true) => Err(MmioError::ReadOnlyRegister { offset: access.offset }),

            (REG_DEVICE_FEATURES_SEL, true) => {
                self.device_features_sel = access.value;
                Ok(MmioEffect::Applied)
            }
            (REG_DEVICE_FEATURES_SEL, false) => {
                Err(MmioError::WriteOnlyRegister { offset: access.offset })
            }

            (REG_DRIVER_FEATURES, true) => {
                self.acked_features = if self.driver_features_sel == 0 {
                    (self.acked_features & !0xffff_ffff) | u64::from(access.value)
                } else {
                    (self.acked_features & 0xffff_ffff) | (u64::from(access.value) << 32)
                };
                self.negotiation.ack_features(self.acked_features);
                Ok(MmioEffect::Applied)
            }
            (REG_DRIVER_FEATURES, false) => {
                Err(MmioError::WriteOnlyRegister { offset: access.offset })
            }

            (REG_DRIVER_FEATURES_SEL, true) => {
                self.driver_features_sel = access.value;
                Ok(MmioEffect::Applied)
            }
            (REG_DRIVER_FEATURES_SEL, false) => {
                Err(MmioError::WriteOnlyRegister { offset: access.offset })
            }

            (REG_QUEUE_SEL, true) => {
                self.queue_sel = access.value as u16;
                Ok(MmioEffect::Applied)
            }
            (REG_QUEUE_SEL, false) => Err(MmioError::WriteOnlyRegister { offset: access.offset }),

            (REG_QUEUE_NUM_MAX, false) => Ok(MmioEffect::ReadValue(u32::from(self.queue_num_max))),
            (REG_QUEUE_NUM_MAX, true) => Err(MmioError::ReadOnlyRegister { offset: access.offset }),

            (REG_QUEUE_NUM, true) => {
                let value = access.value as u16;
                self.selected_queue_mut()?.num = value;
                Ok(MmioEffect::Applied)
            }
            (REG_QUEUE_NUM, false) => Err(MmioError::WriteOnlyRegister { offset: access.offset }),

            (REG_QUEUE_READY, true) => {
                let ready = access.value != 0;
                let queue = self.queue_sel;
                self.selected_queue_mut()?.ready = ready;
                Ok(MmioEffect::QueueReady { queue, ready })
            }
            (REG_QUEUE_READY, false) => {
                let ready = self.selected_queue_mut()?.ready;
                Ok(MmioEffect::ReadValue(u32::from(ready)))
            }

            (REG_QUEUE_NOTIFY, true) => Ok(MmioEffect::QueueNotify {
                queue: access.value as u16,
            }),
            (REG_QUEUE_NOTIFY, false) => {
                Err(MmioError::WriteOnlyRegister { offset: access.offset })
            }

            (REG_INTERRUPT_STATUS, false) => Ok(MmioEffect::ReadValue(0)),
            (REG_INTERRUPT_STATUS, true) => {
                Err(MmioError::ReadOnlyRegister { offset: access.offset })
            }

            (REG_INTERRUPT_ACK, true) => Ok(MmioEffect::Applied),
            (REG_INTERRUPT_ACK, false) => {
                Err(MmioError::WriteOnlyRegister { offset: access.offset })
            }

            (REG_STATUS, true) => self
                .negotiation
                .write_status(access.value as u8)
                .map(MmioEffect::StatusTransition)
                .map_err(MmioError::Negotiation),
            (REG_STATUS, false) => Ok(MmioEffect::ReadValue(u32::from(
                self.negotiation.status().as_byte(),
            ))),

            (REG_QUEUE_DESC_LOW, true) => {
                let queue = self.selected_queue_mut()?;
                queue.descriptor_address =
                    (queue.descriptor_address & !0xffff_ffff) | u64::from(access.value);
                Ok(MmioEffect::Applied)
            }
            (REG_QUEUE_DESC_HIGH, true) => {
                let queue = self.selected_queue_mut()?;
                queue.descriptor_address =
                    (queue.descriptor_address & 0xffff_ffff) | (u64::from(access.value) << 32);
                Ok(MmioEffect::Applied)
            }
            (REG_QUEUE_DESC_LOW | REG_QUEUE_DESC_HIGH, false) => {
                Err(MmioError::WriteOnlyRegister { offset: access.offset })
            }

            (REG_QUEUE_DRIVER_LOW, true) => {
                let queue = self.selected_queue_mut()?;
                queue.driver_address =
                    (queue.driver_address & !0xffff_ffff) | u64::from(access.value);
                Ok(MmioEffect::Applied)
            }
            (REG_QUEUE_DRIVER_HIGH, true) => {
                let queue = self.selected_queue_mut()?;
                queue.driver_address =
                    (queue.driver_address & 0xffff_ffff) | (u64::from(access.value) << 32);
                Ok(MmioEffect::Applied)
            }
            (REG_QUEUE_DRIVER_LOW | REG_QUEUE_DRIVER_HIGH, false) => {
                Err(MmioError::WriteOnlyRegister { offset: access.offset })
            }

            (REG_QUEUE_DEVICE_LOW, true) => {
                let queue = self.selected_queue_mut()?;
                queue.device_address =
                    (queue.device_address & !0xffff_ffff) | u64::from(access.value);
                Ok(MmioEffect::Applied)
            }
            (REG_QUEUE_DEVICE_HIGH, true) => {
                let queue = self.selected_queue_mut()?;
                queue.device_address =
                    (queue.device_address & 0xffff_ffff) | (u64::from(access.value) << 32);
                Ok(MmioEffect::Applied)
            }
            (REG_QUEUE_DEVICE_LOW | REG_QUEUE_DEVICE_HIGH, false) => {
                Err(MmioError::WriteOnlyRegister { offset: access.offset })
            }

            (REG_CONFIG_GENERATION, false) => Ok(MmioEffect::ReadValue(0)),
            (REG_CONFIG_GENERATION, true) => {
                Err(MmioError::ReadOnlyRegister { offset: access.offset })
            }

            (offset, _) => Err(MmioError::UnknownRegister { offset }),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::virtio::status::{
        FEATURE_VERSION_1, STATUS_ACKNOWLEDGE, STATUS_DRIVER, STATUS_DRIVER_OK, STATUS_FEATURES_OK,
    };

    fn read(offset: u64) -> MmioAccess {
        MmioAccess {
            offset,
            is_write: false,
            value: 0,
        }
    }

    fn write(offset: u64, value: u32) -> MmioAccess {
        MmioAccess {
            offset,
            is_write: true,
            value,
        }
    }

    /// Worked example (principle 9 / /algorithm-development): the exact
    /// register sequence a minimal conformant driver performs to bring up
    /// virtio-console over the mmio transport, hand-derived from VIRTIO 1.2
    /// spec §3.1.1 ("Device Initialization") and §4.2.2 (register layout).
    /// `queue_num_max = 4`; the device offers `VIRTIO_F_VERSION_1` only, and
    /// the driver accepts exactly that.
    #[test]
    fn minimal_driver_brings_up_virtio_console_byte_exact() {
        let mut device = MmioDevice::<2>::new(DEVICE_ID_CONSOLE, 4, FEATURE_VERSION_1);

        // 1. probe: MagicValue, Version, DeviceID.
        assert_eq!(
            device.apply(read(REG_MAGIC_VALUE)).unwrap(),
            MmioEffect::ReadValue(MAGIC_VALUE)
        );
        assert_eq!(
            device.apply(read(REG_VERSION)).unwrap(),
            MmioEffect::ReadValue(TRANSPORT_VERSION)
        );
        assert_eq!(
            device.apply(read(REG_DEVICE_ID)).unwrap(),
            MmioEffect::ReadValue(DEVICE_ID_CONSOLE)
        );

        // 2. status handshake bytes 0x01, 0x03 (spec §3.1.1 steps 1-3).
        assert_eq!(
            device.apply(write(REG_STATUS, u32::from(STATUS_ACKNOWLEDGE))).unwrap(),
            MmioEffect::StatusTransition(DeviceStatus::Acknowledged)
        );
        assert_eq!(
            device
                .apply(write(
                    REG_STATUS,
                    u32::from(STATUS_ACKNOWLEDGE | STATUS_DRIVER)
                ))
                .unwrap(),
            MmioEffect::StatusTransition(DeviceStatus::DriverKnown)
        );

        // 3. feature negotiation: read both 32-bit halves, ack the low half
        // only (VIRTIO_F_VERSION_1 is bit 32, so only the high-half ack
        // matters here, but a real driver still selects sel=0 first).
        assert_eq!(
            device.apply(write(REG_DEVICE_FEATURES_SEL, 0)).unwrap(),
            MmioEffect::Applied
        );
        assert_eq!(
            device.apply(read(REG_DEVICE_FEATURES)).unwrap(),
            MmioEffect::ReadValue(0)
        );
        assert_eq!(
            device.apply(write(REG_DEVICE_FEATURES_SEL, 1)).unwrap(),
            MmioEffect::Applied
        );
        assert_eq!(
            device.apply(read(REG_DEVICE_FEATURES)).unwrap(),
            MmioEffect::ReadValue(1) // bit 32 - 32 = bit 0 of the high word
        );

        assert_eq!(
            device.apply(write(REG_DRIVER_FEATURES_SEL, 0)).unwrap(),
            MmioEffect::Applied
        );
        assert_eq!(device.apply(write(REG_DRIVER_FEATURES, 0)).unwrap(), MmioEffect::Applied);
        assert_eq!(
            device.apply(write(REG_DRIVER_FEATURES_SEL, 1)).unwrap(),
            MmioEffect::Applied
        );
        assert_eq!(device.apply(write(REG_DRIVER_FEATURES, 1)).unwrap(), MmioEffect::Applied);

        // 4. status byte 0x0b: FEATURES_OK.
        assert_eq!(
            device
                .apply(write(
                    REG_STATUS,
                    u32::from(STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK)
                ))
                .unwrap(),
            MmioEffect::StatusTransition(DeviceStatus::FeaturesOk {
                offered: FEATURE_VERSION_1,
                acked: FEATURE_VERSION_1,
            })
        );
        // driver re-reads status to confirm the device kept FEATURES_OK set.
        assert_eq!(
            device.apply(read(REG_STATUS)).unwrap(),
            MmioEffect::ReadValue(u32::from(
                STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK
            ))
        );

        // 5. queue setup: select queue 0, read QueueNumMax, set QueueNum,
        // program the three split address-pair registers, mark ready.
        assert_eq!(device.apply(write(REG_QUEUE_SEL, 0)).unwrap(), MmioEffect::Applied);
        assert_eq!(
            device.apply(read(REG_QUEUE_NUM_MAX)).unwrap(),
            MmioEffect::ReadValue(4)
        );
        assert_eq!(device.apply(write(REG_QUEUE_NUM, 4)).unwrap(), MmioEffect::Applied);
        assert_eq!(
            device.apply(write(REG_QUEUE_DESC_LOW, 0x1000)).unwrap(),
            MmioEffect::Applied
        );
        assert_eq!(
            device.apply(write(REG_QUEUE_DESC_HIGH, 0)).unwrap(),
            MmioEffect::Applied
        );
        assert_eq!(
            device.apply(write(REG_QUEUE_DRIVER_LOW, 0x2000)).unwrap(),
            MmioEffect::Applied
        );
        assert_eq!(
            device.apply(write(REG_QUEUE_DRIVER_HIGH, 0)).unwrap(),
            MmioEffect::Applied
        );
        assert_eq!(
            device.apply(write(REG_QUEUE_DEVICE_LOW, 0x3000)).unwrap(),
            MmioEffect::Applied
        );
        assert_eq!(
            device.apply(write(REG_QUEUE_DEVICE_HIGH, 0)).unwrap(),
            MmioEffect::Applied
        );
        assert_eq!(
            device.apply(write(REG_QUEUE_READY, 1)).unwrap(),
            MmioEffect::QueueReady { queue: 0, ready: true }
        );

        // 6. status byte 0x0f: DRIVER_OK — the device is live.
        assert_eq!(
            device
                .apply(write(
                    REG_STATUS,
                    u32::from(STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK | STATUS_DRIVER_OK)
                ))
                .unwrap(),
            MmioEffect::StatusTransition(DeviceStatus::DriverOk {
                offered: FEATURE_VERSION_1,
                acked: FEATURE_VERSION_1,
            })
        );

        // 7. the driver kicks queue 0 — QueueNotify names it directly.
        assert_eq!(
            device.apply(write(REG_QUEUE_NOTIFY, 0)).unwrap(),
            MmioEffect::QueueNotify { queue: 0 }
        );

        assert_eq!(device.queue_descriptor_address(0), Some(0x1000));
        assert_eq!(device.queue_driver_address(0), Some(0x2000));
        assert_eq!(device.queue_device_address(0), Some(0x3000));
        assert_eq!(device.queue_size(0), Some(4));
        assert_eq!(device.queue_is_ready(0), Some(true));
    }

    #[test]
    fn write_to_a_read_only_register_is_rejected() {
        let mut device = MmioDevice::<1>::new(DEVICE_ID_CONSOLE, 4, FEATURE_VERSION_1);
        assert_eq!(
            device.apply(write(REG_MAGIC_VALUE, 0)).unwrap_err(),
            MmioError::ReadOnlyRegister { offset: REG_MAGIC_VALUE }
        );
    }

    #[test]
    fn read_of_a_write_only_register_is_rejected() {
        let mut device = MmioDevice::<1>::new(DEVICE_ID_CONSOLE, 4, FEATURE_VERSION_1);
        assert_eq!(
            device.apply(read(REG_QUEUE_NOTIFY)).unwrap_err(),
            MmioError::WriteOnlyRegister { offset: REG_QUEUE_NOTIFY }
        );
    }

    #[test]
    fn unknown_offset_is_rejected() {
        let mut device = MmioDevice::<1>::new(DEVICE_ID_CONSOLE, 4, FEATURE_VERSION_1);
        assert_eq!(
            device.apply(read(0x200)).unwrap_err(),
            MmioError::UnknownRegister { offset: 0x200 }
        );
    }

    #[test]
    fn queue_select_out_of_range_is_rejected() {
        let mut device = MmioDevice::<1>::new(DEVICE_ID_CONSOLE, 4, FEATURE_VERSION_1);
        device.apply(write(REG_QUEUE_SEL, 5)).unwrap();
        assert_eq!(
            device.apply(write(REG_QUEUE_NUM, 4)).unwrap_err(),
            MmioError::QueueSelectOutOfRange { queue: 5, queue_count: 1 }
        );
    }

    #[test]
    fn out_of_order_status_write_surfaces_the_negotiation_error() {
        let mut device = MmioDevice::<1>::new(DEVICE_ID_CONSOLE, 4, FEATURE_VERSION_1);
        let attempted = u32::from(STATUS_ACKNOWLEDGE | STATUS_DRIVER_OK);
        let error = device.apply(write(REG_STATUS, attempted)).unwrap_err();
        assert!(matches!(error, MmioError::Negotiation(NegotiationError::OutOfOrder { .. })));
    }
}
