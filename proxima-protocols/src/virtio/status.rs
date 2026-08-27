use core::fmt;

/// Device status register bits (VIRTIO 1.2 spec §2.1, `VIRTIO_CONFIG_S_*`).
/// The driver accumulates these into the status byte one write at a time;
/// each write is either the next legal step or a protocol violation.
pub const STATUS_ACKNOWLEDGE: u8 = 1;
pub const STATUS_DRIVER: u8 = 2;
pub const STATUS_DRIVER_OK: u8 = 4;
pub const STATUS_FEATURES_OK: u8 = 8;
pub const STATUS_DEVICE_NEEDS_RESET: u8 = 64;
pub const STATUS_FAILED: u8 = 128;

/// `VIRTIO_F_VERSION_1` (spec §6): bit 32 of the 64-bit feature space. A
/// non-transitional (modern) device offering this bit requires the driver to
/// ack it before `FEATURES_OK` is legal — an ack that drops it means the
/// driver only speaks the legacy protocol, which this codec does not model.
pub const FEATURE_VERSION_1: u64 = 1 << 32;

/// Protocol-violation error for a device-status FSM transition: a status
/// byte legal for no reachable next state, or a `FEATURES_OK` write whose
/// backing feature-bit ack fails the subset/version rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NegotiationError {
    /// The written byte, once the `FAILED` bit is accounted for, matches no
    /// legal status value at all (an undefined combination of bits).
    IllegalStatusByte { byte: u8 },
    /// The written byte is a legal status value, but not the one reachable
    /// from `from` — e.g. `DRIVER_OK` written before `FEATURES_OK`.
    OutOfOrder { attempted: u8, from: DeviceStatus },
    /// The driver acked a feature bit the device never offered (spec §2.2:
    /// `acked ⊆ offered` is the only legal relation).
    AckedUnofferedFeatures { offered: u64, acked: u64, unoffered: u64 },
    /// The device offered `VIRTIO_F_VERSION_1` and the driver's ack did not
    /// include it.
    MissingVersion1 { offered: u64, acked: u64 },
}

impl fmt::Display for NegotiationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IllegalStatusByte { byte } => {
                write!(formatter, "illegal device status byte {byte:#04x}")
            }
            Self::OutOfOrder { attempted, from } => {
                write!(
                    formatter,
                    "status byte {attempted:#04x} is not reachable from {from:?}"
                )
            }
            Self::AckedUnofferedFeatures {
                offered,
                acked,
                unoffered,
            } => {
                write!(
                    formatter,
                    "driver acked features {unoffered:#x} the device never offered \
                     (offered {offered:#x}, acked {acked:#x})"
                )
            }
            Self::MissingVersion1 { offered, acked } => {
                write!(
                    formatter,
                    "device offered VIRTIO_F_VERSION_1 ({offered:#x}) but the driver's \
                     ack ({acked:#x}) dropped it"
                )
            }
        }
    }
}

impl core::error::Error for NegotiationError {}

/// The house enum-FSM shape (guiding-principles §11): each variant carries
/// exactly the data legal at that point in the handshake — the empty states
/// carry nothing, `FeaturesOk`/`DriverOk` carry the negotiated feature pair
/// once it exists, `Failed` carries the raw byte the device (or driver)
/// reported failure with. No runtime "which stage am I in" flag; the match
/// in [`Negotiation::write_status`] is the whole legality check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceStatus {
    /// Status byte 0: the device has just been reset (or never initialized).
    Reset,
    /// `ACKNOWLEDGE` (1): the driver has noticed the device.
    Acknowledged,
    /// `ACKNOWLEDGE | DRIVER` (3): the driver knows how to drive the device.
    DriverKnown,
    /// `ACKNOWLEDGE | DRIVER | FEATURES_OK` (11): feature negotiation is
    /// complete and legal.
    FeaturesOk { offered: u64, acked: u64 },
    /// `ACKNOWLEDGE | DRIVER | FEATURES_OK | DRIVER_OK` (15): the device is
    /// live.
    DriverOk { offered: u64, acked: u64 },
    /// `FAILED` (128) set on top of whatever bits were already legally
    /// accumulated; `status` is the raw byte as written.
    Failed { status: u8 },
}

impl DeviceStatus {
    /// The raw status byte a `Status` register read returns for this state
    /// (spec §4.2.2): the accumulated legal bits, or the exact byte the
    /// device (or driver) reported failure with for [`Self::Failed`].
    /// Reuses [`Self::canonical_bits`] rather than re-deriving the bit
    /// pattern a second time (the same `Negotiation::write_status` already
    /// computes and validates transitions against).
    #[must_use]
    pub fn as_byte(self) -> u8 {
        match self {
            Self::Failed { status } => status,
            other => other.canonical_bits(),
        }
    }

    /// The status bits this state implies, with `FAILED` excluded — used to
    /// check that a `FAILED` write only ever adds that one bit on top of
    /// the currently-legal accumulation, never rewrites earlier bits.
    fn canonical_bits(self) -> u8 {
        match self {
            Self::Reset => 0,
            Self::Acknowledged => STATUS_ACKNOWLEDGE,
            Self::DriverKnown => STATUS_ACKNOWLEDGE | STATUS_DRIVER,
            Self::FeaturesOk { .. } => STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK,
            Self::DriverOk { .. } => {
                STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK | STATUS_DRIVER_OK
            }
            Self::Failed { status } => status & !STATUS_FAILED,
        }
    }
}

/// Device-status FSM plus the feature-bit registers the spec keeps in
/// separate MMIO fields (`device_feature`/`driver_feature`) — modeled here
/// as two setters distinct from [`write_status`](Self::write_status) because
/// on real hardware they are separate register writes, not status-byte
/// bits. Fixed-shape, allocates nothing: two `u64`s and one small enum.
#[derive(Debug, Clone, Copy)]
pub struct Negotiation {
    state: DeviceStatus,
    offered: u64,
    acked: u64,
}

impl Negotiation {
    /// A freshly reset device: status byte 0, no features offered or acked.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: DeviceStatus::Reset,
            offered: 0,
            acked: 0,
        }
    }

    #[must_use]
    pub fn status(&self) -> DeviceStatus {
        self.state
    }

    /// Records the device's offered feature bits (`device_feature` register
    /// reads). Does not itself validate or transition — legality is checked
    /// once the driver claims `FEATURES_OK`.
    pub fn offer_features(&mut self, offered: u64) {
        self.offered = offered;
    }

    /// Records the driver's chosen subset (`driver_feature` register
    /// writes). Same non-validating shape as [`offer_features`](Self::offer_features).
    pub fn ack_features(&mut self, acked: u64) {
        self.acked = acked;
    }

    /// Applies one device-status register write, returning the resulting
    /// state or the named protocol violation. `byte == 0` is always legal
    /// and resets from any state, including `Failed` (spec §2.1: writing 0
    /// is how the driver responds to `DEVICE_NEEDS_RESET`).
    pub fn write_status(&mut self, byte: u8) -> Result<DeviceStatus, NegotiationError> {
        if byte == 0 {
            self.state = DeviceStatus::Reset;
            self.offered = 0;
            self.acked = 0;
            return Ok(self.state);
        }

        if byte & STATUS_FAILED != 0 {
            let current = self.state.canonical_bits();
            if byte & !STATUS_FAILED != current {
                return Err(NegotiationError::IllegalStatusByte { byte });
            }
            self.state = DeviceStatus::Failed { status: byte };
            return Ok(self.state);
        }

        let next = match (self.state, byte) {
            (DeviceStatus::Reset, STATUS_ACKNOWLEDGE) => DeviceStatus::Acknowledged,
            (DeviceStatus::Acknowledged, written)
                if written == STATUS_ACKNOWLEDGE | STATUS_DRIVER =>
            {
                DeviceStatus::DriverKnown
            }
            (DeviceStatus::DriverKnown, written)
                if written == STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK =>
            {
                self.validate_features()?;
                DeviceStatus::FeaturesOk {
                    offered: self.offered,
                    acked: self.acked,
                }
            }
            (DeviceStatus::FeaturesOk { offered, acked }, written)
                if written
                    == STATUS_ACKNOWLEDGE
                        | STATUS_DRIVER
                        | STATUS_FEATURES_OK
                        | STATUS_DRIVER_OK =>
            {
                DeviceStatus::DriverOk { offered, acked }
            }
            _ => {
                return Err(NegotiationError::OutOfOrder {
                    attempted: byte,
                    from: self.state,
                });
            }
        };
        self.state = next;
        Ok(next)
    }

    /// `acked ⊆ offered`, plus the `VIRTIO_F_VERSION_1` requirement — the
    /// arithmetic the device applies before it will honor `FEATURES_OK`.
    fn validate_features(&self) -> Result<(), NegotiationError> {
        let unoffered = self.acked & !self.offered;
        if unoffered != 0 {
            return Err(NegotiationError::AckedUnofferedFeatures {
                offered: self.offered,
                acked: self.acked,
                unoffered,
            });
        }
        if self.offered & FEATURE_VERSION_1 != 0 && self.acked & FEATURE_VERSION_1 == 0 {
            return Err(NegotiationError::MissingVersion1 {
                offered: self.offered,
                acked: self.acked,
            });
        }
        Ok(())
    }
}

impl Default for Negotiation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    /// Worked example (principle 9 / /algorithm-development): a full legal
    /// negotiation, hand-derived per VIRTIO 1.2 spec §2.1 — the exact byte
    /// sequence a conformant driver writes to the device-status register,
    /// with the device offering `VIRTIO_F_VERSION_1` plus one feature bit
    /// the driver accepts.
    ///
    /// device_feature = VIRTIO_F_VERSION_1 (bit 32) | bit 0
    /// driver_feature (ack) = the same two bits (subset, includes VERSION_1)
    /// status bytes written, in order: 0x01, 0x03, 0x0b, 0x0f
    #[test]
    fn full_legal_negotiation_reaches_driver_ok() {
        let mut negotiation = Negotiation::new();
        assert_eq!(negotiation.status(), DeviceStatus::Reset);

        assert_eq!(
            negotiation.write_status(STATUS_ACKNOWLEDGE).expect("ack"),
            DeviceStatus::Acknowledged
        );
        assert_eq!(
            negotiation
                .write_status(STATUS_ACKNOWLEDGE | STATUS_DRIVER)
                .expect("driver known"),
            DeviceStatus::DriverKnown
        );

        let offered = FEATURE_VERSION_1 | 0x1;
        negotiation.offer_features(offered);
        negotiation.ack_features(offered);

        assert_eq!(
            negotiation
                .write_status(STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK)
                .expect("features negotiated legally"),
            DeviceStatus::FeaturesOk {
                offered,
                acked: offered
            }
        );
        assert_eq!(
            negotiation
                .write_status(
                    STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK | STATUS_DRIVER_OK
                )
                .expect("device is live"),
            DeviceStatus::DriverOk {
                offered,
                acked: offered
            }
        );
    }

    /// Illegal sequence 1: the driver jumps straight to `DRIVER_OK`,
    /// skipping `FEATURES_OK` entirely — a protocol violation, not a panic.
    #[test]
    fn driver_ok_before_features_ok_is_rejected() {
        let mut negotiation = Negotiation::new();
        negotiation
            .write_status(STATUS_ACKNOWLEDGE)
            .expect("ack");
        negotiation
            .write_status(STATUS_ACKNOWLEDGE | STATUS_DRIVER)
            .expect("driver known");

        let attempted =
            STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK | STATUS_DRIVER_OK;
        assert_eq!(
            negotiation.write_status(attempted).unwrap_err(),
            NegotiationError::OutOfOrder {
                attempted,
                from: DeviceStatus::DriverKnown,
            }
        );
        // the illegal write does not advance the state.
        assert_eq!(negotiation.status(), DeviceStatus::DriverKnown);
    }

    /// Illegal sequence 2: the driver acks a feature bit the device never
    /// offered — `acked ⊄ offered`, rejected at the `FEATURES_OK` write.
    #[test]
    fn ack_of_unoffered_feature_bit_is_rejected() {
        let mut negotiation = Negotiation::new();
        negotiation
            .write_status(STATUS_ACKNOWLEDGE)
            .expect("ack");
        negotiation
            .write_status(STATUS_ACKNOWLEDGE | STATUS_DRIVER)
            .expect("driver known");

        let offered = FEATURE_VERSION_1;
        let acked = FEATURE_VERSION_1 | 0x2; // bit 1 was never offered
        negotiation.offer_features(offered);
        negotiation.ack_features(acked);

        assert_eq!(
            negotiation
                .write_status(STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK)
                .unwrap_err(),
            NegotiationError::AckedUnofferedFeatures {
                offered,
                acked,
                unoffered: 0x2,
            }
        );
        // the codec did not silently accept it — state stays pre-negotiation.
        assert_eq!(negotiation.status(), DeviceStatus::DriverKnown);
    }

    /// A driver dropping `VIRTIO_F_VERSION_1` from its ack is a named error,
    /// distinct from acking an unoffered bit.
    #[test]
    fn dropping_version_1_from_the_ack_is_rejected() {
        let mut negotiation = Negotiation::new();
        negotiation
            .write_status(STATUS_ACKNOWLEDGE)
            .expect("ack");
        negotiation
            .write_status(STATUS_ACKNOWLEDGE | STATUS_DRIVER)
            .expect("driver known");

        negotiation.offer_features(FEATURE_VERSION_1);
        negotiation.ack_features(0);

        assert_eq!(
            negotiation
                .write_status(STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK)
                .unwrap_err(),
            NegotiationError::MissingVersion1 {
                offered: FEATURE_VERSION_1,
                acked: 0,
            }
        );
    }

    /// Reset (byte 0) is legal from any state, including mid-negotiation.
    #[test]
    fn reset_is_legal_from_any_state() {
        let mut negotiation = Negotiation::new();
        negotiation
            .write_status(STATUS_ACKNOWLEDGE)
            .expect("ack");
        negotiation
            .write_status(STATUS_ACKNOWLEDGE | STATUS_DRIVER)
            .expect("driver known");

        assert_eq!(negotiation.write_status(0).expect("reset"), DeviceStatus::Reset);
        assert_eq!(negotiation.status(), DeviceStatus::Reset);
    }

    /// `FAILED` may be set on top of the currently-legal bits from any
    /// nonzero state; rewriting a *different* set of lower bits alongside
    /// `FAILED` is still a violation.
    #[test]
    fn failed_bit_layers_onto_the_current_state() {
        let mut negotiation = Negotiation::new();
        negotiation
            .write_status(STATUS_ACKNOWLEDGE)
            .expect("ack");
        negotiation
            .write_status(STATUS_ACKNOWLEDGE | STATUS_DRIVER)
            .expect("driver known");

        let failed_byte = STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FAILED;
        assert_eq!(
            negotiation.write_status(failed_byte).expect("device fails"),
            DeviceStatus::Failed {
                status: failed_byte
            }
        );

        // reset clears it, matching the driver's only legal recovery path.
        assert_eq!(negotiation.write_status(0).expect("reset"), DeviceStatus::Reset);
    }

    /// `FAILED` set together with bits that do not match the accumulated
    /// state (here, `DRIVER_OK` bolted on without ever reaching it) is an
    /// undefined byte, not a valid `Failed` report.
    #[test]
    fn failed_bit_with_mismatched_lower_bits_is_illegal() {
        let mut negotiation = Negotiation::new();
        negotiation
            .write_status(STATUS_ACKNOWLEDGE)
            .expect("ack");

        let bogus = STATUS_ACKNOWLEDGE | STATUS_DRIVER_OK | STATUS_FAILED;
        assert_eq!(
            negotiation.write_status(bogus).unwrap_err(),
            NegotiationError::IllegalStatusByte { byte: bogus }
        );
    }
}
