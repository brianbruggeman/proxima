//! ARM PrimeCell UART (PL011, ARM DDI 0183) register block — M5b's console
//! model, alongside the GICv3 model (`src/gic.rs`) and PSCI (`src/psci.rs`).
//! `src/dtb.rs`'s uart node advertises `compatible = "arm,pl011"` at QEMU
//! virt's fixed address (`hw/arm/virt.c`'s `VIRT_UART`); this module is the
//! pure decode/state machine for that window, mirroring `src/gic.rs`'s shape
//! exactly at a fraction of the size: one raw `(offset, is_write, value)`
//! access in, one typed effect or error out, a single match over offset, no
//! cursor, no I/O.
//!
//! # This is the byte channel M5's exit criterion asks for
//!
//! The effect that matters is [`Pl011Effect::TxByte`]: a write to `UARTDR`
//! (offset 0x000) is the guest emitting one console byte. Everything else
//! this module models exists only so a real kernel's earlycon/console probe
//! sequence completes without the guest wedging on an unmodeled register —
//! see `tests::linux_pl011_earlycon_probe_then_two_byte_write_sequence` for
//! the exact register sequence a Linux `pl011` earlycon performs.
//!
//! # Spec citations
//!
//! Register offsets and reset values are ARM DDI 0183 ("PrimeCell UART
//! (PL011) Technical Reference Manual"), chapter 3 ("Programmer's Model").
//! `UARTPeriphID0-3`/`UARTPCellID0-3` (offsets 0xfe0-0xffc) are the
//! architected identification bytes every PrimeCell peripheral exposes at
//! the top of its 4 KiB window; Linux's `amba-pl011` driver (via the
//! generic AMBA bus match in `drivers/amba/bus.c`) reads all eight before
//! binding, so they are modeled as fixed read-only constants rather than
//! RAZ/WI or unknown-register rejections.
//!
//! # Tier
//!
//! Tier-3 (bare `no_std + no_alloc`), landed unconditional like `src/gic.rs`
//! and `src/psci.rs`: every field is a plain scalar, no allocation, no
//! register access, no syscall.

const REG_UARTDR: u64 = 0x000;
const REG_UARTFR: u64 = 0x018;
const REG_UARTIBRD: u64 = 0x024;
const REG_UARTFBRD: u64 = 0x028;
const REG_UARTLCR_H: u64 = 0x02c;
const REG_UARTCR: u64 = 0x030;
/// `UARTIFLS` (offset 0x034, RW): interrupt FIFO level select — the wall
/// past `nosmp`'s own SMP-IPI routing (this module's own userspace-boot
/// investigation): the full `amba-pl011` driver probe (not just an earlycon
/// polling path) sets its RX/TX trigger levels here before enabling
/// interrupt-driven transmit.
const REG_UARTIFLS: u64 = 0x034;
const REG_UARTIMSC: u64 = 0x038;
const REG_UARTRIS: u64 = 0x03c;
const REG_UARTMIS: u64 = 0x040;
const REG_UARTICR: u64 = 0x044;
const REG_UARTPERIPHID0: u64 = 0xfe0;
const REG_UARTPERIPHID1: u64 = 0xfe4;
const REG_UARTPERIPHID2: u64 = 0xfe8;
const REG_UARTPERIPHID3: u64 = 0xfec;
const REG_UARTPCELLID0: u64 = 0xff0;
const REG_UARTPCELLID1: u64 = 0xff4;
const REG_UARTPCELLID2: u64 = 0xff8;
const REG_UARTPCELLID3: u64 = 0xffc;

/// `UARTFR.RXFE` (bit 4): receive FIFO empty. Always set — this model has no
/// guest-to-host input path, so the receive FIFO is permanently empty.
///
/// `UARTFR.TXFF` (bit 5, transmit FIFO full) and `UARTFR.BUSY` (bit 3, UART
/// busy transmitting) are both always clear — every `UARTDR` write is
/// accepted immediately as a [`Pl011Effect::TxByte`] and completes
/// synchronously, matching
/// [`GicDistributor`](crate::gic::GicDistributor)'s own "every write here
/// completes synchronously" `RWP` convention — so this constant alone is the
/// whole `UARTFR` reset value this model ever reports.
const FR_RESET_VALUE: u32 = 1 << 4;

/// `UARTPeriphID0-3` (offsets 0xfe0-0xfec): the architected PrimeCell
/// peripheral ID bytes a PL011 reports, per ARM DDI 0183 chapter 3.
const PERIPH_ID: [u32; 4] = [0x11, 0x10, 0x14, 0x00];

/// `UARTPCellID0-3` (offsets 0xff0-0xffc): the architected PrimeCell
/// identification bytes every AMBA PrimeCell peripheral reports, per ARM
/// DDI 0183 chapter 3 — identical across every PrimeCell device, not
/// PL011-specific.
const PCELL_ID: [u32; 4] = [0x0d, 0xf0, 0x05, 0xb1];

/// One raw register access recovered from a trapped guest load/store,
/// mirroring [`crate::gic::GicAccess`] exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pl011Access {
    pub offset: u64,
    pub is_write: bool,
    pub value: u32,
}

/// What the caller must do in response to one applied [`Pl011Access`] —
/// mirrors [`crate::gic::GicdEffect`]'s shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pl011Effect {
    /// A read register access: the value the caller writes back into the
    /// guest's destination register.
    ReadValue(u32),
    /// A write register access accepted with no further consequence.
    Applied,
    /// A `UARTDR` write: the guest emitted this byte on the console. This is
    /// the byte channel M5's exit criterion names.
    TxByte(u8),
}

/// Why [`Pl011Uart::apply`] rejected an access — mirrors
/// [`crate::gic::GicdError`]'s three cases exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pl011Error {
    /// No register this model implements exists at this offset.
    UnknownRegister { offset: u64 },
    /// The access offset was not naturally aligned to the register's 4-byte
    /// word size.
    UnalignedAccess { offset: u64 },
    /// The register at this offset is read-only; the guest attempted a
    /// write.
    ReadOnlyRegister { offset: u64 },
}

impl core::fmt::Display for Pl011Error {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnknownRegister { offset } => {
                write!(formatter, "no pl011 register defined at offset {offset:#x}")
            }
            Self::UnalignedAccess { offset } => {
                write!(
                    formatter,
                    "pl011 access at offset {offset:#x} is not 4-byte aligned"
                )
            }
            Self::ReadOnlyRegister { offset } => {
                write!(
                    formatter,
                    "pl011 register at offset {offset:#x} is read-only"
                )
            }
        }
    }
}

impl core::error::Error for Pl011Error {}

/// One PL011 UART's register-block state: the baud-rate divisor, line
/// control, control, and interrupt-mask registers a Linux `pl011` driver
/// writes at probe, stored verbatim (this model has no clock to derive an
/// actual baud rate from, and no interrupt controller wiring yet — storing
/// and echoing back is the whole contract these registers need to satisfy
/// for a guest console that only ever transmits).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pl011Uart {
    ibrd: u32,
    fbrd: u32,
    lcr_h: u32,
    cr: u32,
    imsc: u32,
    ifls: u32,
}

impl Pl011Uart {
    /// A freshly reset UART: every stored register at its architected
    /// power-on-reset value of 0, except `UARTIFLS`, whose architected reset
    /// value is both FIFOs at the 1/2-full trigger level (DDI 0183's
    /// `UARTIFLS` reset value, `0b010_010`).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            ibrd: 0,
            fbrd: 0,
            lcr_h: 0,
            cr: 0,
            imsc: 0,
            ifls: 0b010_010,
        }
    }

    #[must_use]
    pub fn ibrd(&self) -> u32 {
        self.ibrd
    }

    #[must_use]
    pub fn fbrd(&self) -> u32 {
        self.fbrd
    }

    #[must_use]
    pub fn lcr_h(&self) -> u32 {
        self.lcr_h
    }

    #[must_use]
    pub fn cr(&self) -> u32 {
        self.cr
    }

    #[must_use]
    pub fn imsc(&self) -> u32 {
        self.imsc
    }

    #[must_use]
    pub fn ifls(&self) -> u32 {
        self.ifls
    }

    /// Apply one register access, returning the effect the caller must carry
    /// out. Mirrors [`crate::gic::GicDistributor::apply`]'s single match
    /// over `(offset, is_write)` — every register here is a fixed 32-bit
    /// word at a fixed offset.
    pub fn apply(&mut self, access: Pl011Access) -> Result<Pl011Effect, Pl011Error> {
        if !access.offset.is_multiple_of(4) {
            return Err(Pl011Error::UnalignedAccess {
                offset: access.offset,
            });
        }

        match access.offset {
            REG_UARTDR => Ok(apply_dr(access)),
            REG_UARTFR => read_only(access, FR_RESET_VALUE),
            REG_UARTIBRD => Ok(apply_field(&mut self.ibrd, access)),
            REG_UARTFBRD => Ok(apply_field(&mut self.fbrd, access)),
            REG_UARTLCR_H => Ok(apply_field(&mut self.lcr_h, access)),
            REG_UARTCR => Ok(apply_field(&mut self.cr, access)),
            REG_UARTIFLS => Ok(apply_field(&mut self.ifls, access)),
            REG_UARTIMSC => Ok(apply_field(&mut self.imsc, access)),
            REG_UARTRIS => read_only(access, 0),
            REG_UARTMIS => read_only(access, 0),
            REG_UARTICR => Ok(apply_icr(access)),
            REG_UARTPERIPHID0 => read_only(access, PERIPH_ID[0]),
            REG_UARTPERIPHID1 => read_only(access, PERIPH_ID[1]),
            REG_UARTPERIPHID2 => read_only(access, PERIPH_ID[2]),
            REG_UARTPERIPHID3 => read_only(access, PERIPH_ID[3]),
            REG_UARTPCELLID0 => read_only(access, PCELL_ID[0]),
            REG_UARTPCELLID1 => read_only(access, PCELL_ID[1]),
            REG_UARTPCELLID2 => read_only(access, PCELL_ID[2]),
            REG_UARTPCELLID3 => read_only(access, PCELL_ID[3]),
            offset => Err(Pl011Error::UnknownRegister { offset }),
        }
    }
}

impl Default for Pl011Uart {
    fn default() -> Self {
        Self::new()
    }
}

/// `UARTDR` (offset 0x000): a write is the guest emitting one console byte
/// (the low 8 bits of `access.value`, per DDI 0183's `DATA` field); a read
/// returns 0 — this model never has receive data pending
/// (`UARTFR.RXFE` is always set, [`FR_RESET_VALUE`]), so a guest that reads `UARTDR` without first
/// checking `UARTFR.RXFE` observes an architecturally-undefined byte, and 0
/// is as good as any.
fn apply_dr(access: Pl011Access) -> Pl011Effect {
    if access.is_write {
        Pl011Effect::TxByte(access.value as u8)
    } else {
        Pl011Effect::ReadValue(0)
    }
}

/// `UARTICR` (offset 0x044, write-only per spec): writing clears pending
/// interrupt status bits. This model's `UARTRIS`/`UARTMIS` are always 0 (no
/// pending interrupts are ever raised), so a write is accepted with no
/// further effect and a read returns 0 — the architected "unpredictable on
/// read" contract resolved to the simplest legal value.
fn apply_icr(access: Pl011Access) -> Pl011Effect {
    if access.is_write {
        Pl011Effect::Applied
    } else {
        Pl011Effect::ReadValue(0)
    }
}

/// A read/write register whose value is a plain stored `u32` with no
/// side-effecting logic beyond store-then-echo.
fn apply_field(field: &mut u32, access: Pl011Access) -> Pl011Effect {
    if access.is_write {
        *field = access.value;
        Pl011Effect::Applied
    } else {
        Pl011Effect::ReadValue(*field)
    }
}

/// A read-only register: reads return `value`, writes are rejected.
fn read_only(access: Pl011Access, value: u32) -> Result<Pl011Effect, Pl011Error> {
    if access.is_write {
        return Err(Pl011Error::ReadOnlyRegister {
            offset: access.offset,
        });
    }
    Ok(Pl011Effect::ReadValue(value))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::{
        Pl011Access, Pl011Effect, Pl011Error, Pl011Uart, REG_UARTCR, REG_UARTDR, REG_UARTFBRD,
        REG_UARTFR, REG_UARTIBRD, REG_UARTICR, REG_UARTIFLS, REG_UARTIMSC, REG_UARTLCR_H,
        REG_UARTMIS, REG_UARTPCELLID0, REG_UARTPCELLID1, REG_UARTPCELLID2, REG_UARTPCELLID3,
        REG_UARTPERIPHID0, REG_UARTPERIPHID1, REG_UARTPERIPHID2, REG_UARTPERIPHID3, REG_UARTRIS,
    };

    fn read(offset: u64) -> Pl011Access {
        Pl011Access {
            offset,
            is_write: false,
            value: 0,
        }
    }

    fn write(offset: u64, value: u32) -> Pl011Access {
        Pl011Access {
            offset,
            is_write: true,
            value,
        }
    }

    /// Worked example (principle 9 / `algorithm-development`): the register
    /// sequence Linux's `drivers/tty/serial/amba-pl011.c` earlycon path
    /// performs to write one byte — poll `UARTFR.TXFF` until clear, write
    /// the byte to `UARTDR` — repeated for the two bytes of "OK", the exact
    /// bytes the M5b real-exit probe emits.
    #[test]
    fn linux_pl011_earlycon_probe_then_two_byte_write_sequence() {
        let mut uart = Pl011Uart::new();

        let periph_ids: [u32; 4] = core::array::from_fn(|index| {
            let offset = [
                REG_UARTPERIPHID0,
                REG_UARTPERIPHID1,
                REG_UARTPERIPHID2,
                REG_UARTPERIPHID3,
            ][index];
            match uart.apply(read(offset)).expect("periph id is readable") {
                Pl011Effect::ReadValue(value) => value,
                other => unreachable!("periph id read yields ReadValue, got {other:?}"),
            }
        });
        assert_eq!(periph_ids, [0x11, 0x10, 0x14, 0x00], "PL011 PeriphID0-3");

        let cell_ids: [u32; 4] = core::array::from_fn(|index| {
            let offset = [
                REG_UARTPCELLID0,
                REG_UARTPCELLID1,
                REG_UARTPCELLID2,
                REG_UARTPCELLID3,
            ][index];
            match uart.apply(read(offset)).expect("cell id is readable") {
                Pl011Effect::ReadValue(value) => value,
                other => unreachable!("cell id read yields ReadValue, got {other:?}"),
            }
        });
        assert_eq!(cell_ids, [0x0d, 0xf0, 0x05, 0xb1], "PrimeCell PCellID0-3");

        for byte in b"OK" {
            let flags = uart.apply(read(REG_UARTFR)).expect("fr is readable");
            assert_eq!(
                flags,
                Pl011Effect::ReadValue(1 << 4),
                "TXFF/BUSY clear, RXFE set"
            );

            let sent = uart
                .apply(write(REG_UARTDR, u32::from(*byte)))
                .expect("dr write is legal");
            assert_eq!(sent, Pl011Effect::TxByte(*byte), "dr write emits the byte");
        }
    }

    #[test]
    fn ibrd_fbrd_lcr_h_cr_ifls_imsc_round_trip_the_written_value() {
        let mut uart = Pl011Uart::new();
        for (offset, value) in [
            (REG_UARTIBRD, 0x1a),
            (REG_UARTFBRD, 0x03),
            (REG_UARTLCR_H, 0b0111_0000),
            (REG_UARTCR, 0x0301),
            (REG_UARTIFLS, 0b011_011),
            (REG_UARTIMSC, 0x50),
        ] {
            uart.apply(write(offset, value))
                .expect("register accepts writes");
            let read_back = uart.apply(read(offset)).expect("register is readable");
            assert_eq!(read_back, Pl011Effect::ReadValue(value));
        }
        assert_eq!(uart.ibrd(), 0x1a);
        assert_eq!(uart.fbrd(), 0x03);
        assert_eq!(uart.lcr_h(), 0b0111_0000);
        assert_eq!(uart.cr(), 0x0301);
        assert_eq!(uart.ifls(), 0b011_011);
        assert_eq!(uart.imsc(), 0x50);
    }

    /// Worked example (userspace-boot investigation's own next wall past
    /// `nosmp`): `UARTIFLS`'s architected reset value is both FIFOs at the
    /// 1/2-full trigger level (DDI 0183, `0b010_010`) — the value a guest
    /// that never wrote this register would observe on its own probe read,
    /// before the `amba-pl011` driver's real interrupt-mode setup
    /// overwrites it.
    #[test]
    fn ifls_reset_value_is_both_fifos_at_half_full() {
        let mut uart = Pl011Uart::new();
        let readback = uart.apply(read(REG_UARTIFLS)).expect("ifls is readable");
        assert_eq!(readback, Pl011Effect::ReadValue(0b010_010));
    }

    #[test]
    fn ris_and_mis_are_always_zero_and_read_only() {
        let mut uart = Pl011Uart::new();
        assert_eq!(uart.apply(read(REG_UARTRIS)), Ok(Pl011Effect::ReadValue(0)));
        assert_eq!(uart.apply(read(REG_UARTMIS)), Ok(Pl011Effect::ReadValue(0)));
        assert_eq!(
            uart.apply(write(REG_UARTRIS, 1)),
            Err(Pl011Error::ReadOnlyRegister {
                offset: REG_UARTRIS
            })
        );
    }

    #[test]
    fn icr_write_is_accepted_and_read_returns_zero() {
        let mut uart = Pl011Uart::new();
        assert_eq!(
            uart.apply(write(REG_UARTICR, 0xff)),
            Ok(Pl011Effect::Applied)
        );
        assert_eq!(uart.apply(read(REG_UARTICR)), Ok(Pl011Effect::ReadValue(0)));
    }

    #[test]
    fn periph_id_registers_reject_writes() {
        let mut uart = Pl011Uart::new();
        assert_eq!(
            uart.apply(write(REG_UARTPERIPHID0, 0)),
            Err(Pl011Error::ReadOnlyRegister {
                offset: REG_UARTPERIPHID0
            })
        );
    }

    #[test]
    fn unaligned_access_is_rejected() {
        let mut uart = Pl011Uart::new();
        assert_eq!(
            uart.apply(read(REG_UARTDR + 1)),
            Err(Pl011Error::UnalignedAccess {
                offset: REG_UARTDR + 1
            })
        );
    }

    #[test]
    fn unknown_offset_is_rejected() {
        let mut uart = Pl011Uart::new();
        const UNDEFINED_OFFSET: u64 = 0x900;
        assert_eq!(
            uart.apply(read(UNDEFINED_OFFSET)),
            Err(Pl011Error::UnknownRegister {
                offset: UNDEFINED_OFFSET
            })
        );
    }
}
