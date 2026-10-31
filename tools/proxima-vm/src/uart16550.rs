//! A 16550A UART as the Linux 8250 driver's autoconfig and startup probes see
//! it, as a tier-3 sans-IO codec: register state in, register state out, no
//! buffers and no clock.
//!
//! Transmit is instant: [`Uart16550::write`] hands the byte it accepted back to
//! the caller, and the THRE interrupt is pending the moment a byte is written.
//! Receive serves a caller-owned byte slice, held back until
//! [`Uart16550::enable_receive`]: the 8250 driver drains the receive FIFO while
//! probing and opening a port, so bytes offered before the guest has
//! configured its tty are lost on real hardware too. The interrupt line is
//! level: [`Uart16550::irq_level`] is a pure function of register state.
//!
//! Register map (offset from the port base), per the National Semiconductor
//! PC16550D datasheet, "Register Set":
//!
//! | offset | DLAB=0 read | DLAB=0 write | DLAB=1 |
//! |--------|-------------|--------------|--------|
//! | 0      | RBR         | THR          | DLL    |
//! | 1      | IER         | IER          | DLM    |
//! | 2      | IIR         | FCR          | IIR/FCR |
//! | 3      | LCR         | LCR          | LCR    |
//! | 4      | MCR         | MCR          | MCR    |
//! | 5      | LSR         | -            | LSR    |
//! | 6      | MSR         | -            | MSR    |
//! | 7      | SCR         | SCR          | SCR    |

pub const PORT_COUNT: u16 = 8;

const IER_RECEIVE_DATA: u8 = 0x01;
const IER_TRANSMIT_EMPTY: u8 = 0x02;
const IER_WRITABLE_BITS: u8 = 0x0f;
const LSR_DATA_READY: u8 = 0x01;
const LSR_TRANSMIT_EMPTY: u8 = 0x60;
const MCR_LOOPBACK: u8 = 0x10;
const MCR_WRITABLE_BITS: u8 = 0x1f;
const LCR_DIVISOR_LATCH: u8 = 0x80;
const FCR_FIFO_ENABLE: u8 = 0x01;
const IIR_FIFOS_ENABLED: u8 = 0xc0;
const IIR_RECEIVE_DATA: u8 = 0x04;
const IIR_TRANSMIT_EMPTY: u8 = 0x02;
const IIR_NONE_PENDING: u8 = 0x01;
const MSR_NOT_LOOPBACK: u8 = 0xb0;

#[derive(Clone, Debug)]
pub struct Uart16550<'rx> {
    interrupt_enable: u8,
    line_control: u8,
    modem_control: u8,
    scratch: u8,
    divisor_low: u8,
    divisor_high: u8,
    fifo_control: u8,
    transmit_interrupt_pending: bool,
    receive_enabled: bool,
    receive: &'rx [u8],
    receive_position: usize,
}

impl<'rx> Uart16550<'rx> {
    #[must_use]
    pub const fn new(receive: &'rx [u8]) -> Self {
        Self {
            interrupt_enable: 0,
            line_control: 0,
            modem_control: 0,
            scratch: 0,
            divisor_low: 0,
            divisor_high: 0,
            fifo_control: 0,
            transmit_interrupt_pending: false,
            receive_enabled: false,
            receive,
            receive_position: 0,
        }
    }

    pub fn enable_receive(&mut self) {
        self.receive_enabled = true;
    }

    #[must_use]
    pub fn receive_consumed(&self) -> usize {
        self.receive_position
    }

    fn receive_ready(&self) -> bool {
        self.receive_enabled && self.receive_position < self.receive.len()
    }

    fn divisor_latch(&self) -> bool {
        self.line_control & LCR_DIVISOR_LATCH != 0
    }

    #[must_use]
    pub fn irq_level(&self) -> bool {
        let receive = self.interrupt_enable & IER_RECEIVE_DATA != 0 && self.receive_ready();
        let transmit = self.interrupt_enable & IER_TRANSMIT_EMPTY != 0 && self.transmit_interrupt_pending;
        receive || transmit
    }

    fn modem_status(&self) -> u8 {
        if self.modem_control & MCR_LOOPBACK == 0 {
            return MSR_NOT_LOOPBACK;
        }
        ((self.modem_control & 0x02) << 3)
            | ((self.modem_control & 0x01) << 5)
            | ((self.modem_control & 0x04) << 4)
            | ((self.modem_control & 0x08) << 4)
    }

    fn read_interrupt_id(&mut self) -> u8 {
        let fifo_bits = if self.fifo_control & FCR_FIFO_ENABLE != 0 { IIR_FIFOS_ENABLED } else { 0 };
        if self.interrupt_enable & IER_RECEIVE_DATA != 0 && self.receive_ready() {
            return fifo_bits | IIR_RECEIVE_DATA;
        }
        if self.interrupt_enable & IER_TRANSMIT_EMPTY != 0 && self.transmit_interrupt_pending {
            self.transmit_interrupt_pending = false;
            return fifo_bits | IIR_TRANSMIT_EMPTY;
        }
        fifo_bits | IIR_NONE_PENDING
    }

    fn receive_byte(&mut self) -> u8 {
        let Some(byte) = self.receive.get(self.receive_position).copied().filter(|_| self.receive_enabled) else {
            return 0;
        };
        self.receive_position += 1;
        byte
    }

    pub fn read(&mut self, offset: u8) -> u8 {
        match (offset, self.divisor_latch()) {
            (0, true) => self.divisor_low,
            (0, false) => self.receive_byte(),
            (1, true) => self.divisor_high,
            (1, false) => self.interrupt_enable,
            (2, _) => self.read_interrupt_id(),
            (3, _) => self.line_control,
            (4, _) => self.modem_control,
            (5, _) => LSR_TRANSMIT_EMPTY | if self.receive_ready() { LSR_DATA_READY } else { 0 },
            (6, _) => self.modem_status(),
            _ => self.scratch,
        }
    }

    pub fn write(&mut self, offset: u8, value: u8) -> Option<u8> {
        match (offset, self.divisor_latch()) {
            (0, true) => self.divisor_low = value,
            (0, false) => return self.transmit(value),
            (1, true) => self.divisor_high = value,
            (1, false) => self.set_interrupt_enable(value),
            (2, _) => self.fifo_control = value,
            (3, _) => self.line_control = value,
            (4, _) => self.modem_control = value & MCR_WRITABLE_BITS,
            (7, _) => self.scratch = value,
            _ => {}
        }
        None
    }

    fn set_interrupt_enable(&mut self, value: u8) {
        let enabled_now = value & !self.interrupt_enable;
        self.interrupt_enable = value & IER_WRITABLE_BITS;
        if enabled_now & IER_TRANSMIT_EMPTY != 0 {
            self.transmit_interrupt_pending = true;
        }
    }

    fn transmit(&mut self, value: u8) -> Option<u8> {
        if self.modem_control & MCR_LOOPBACK != 0 {
            return None;
        }
        self.transmit_interrupt_pending = true;
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    const THR: u8 = 0;
    const IER: u8 = 1;
    const IIR: u8 = 2;
    const FCR: u8 = 2;
    const LCR: u8 = 3;
    const MCR: u8 = 4;
    const LSR: u8 = 5;
    const MSR: u8 = 6;
    const SCR: u8 = 7;

    #[test]
    fn a_reset_uart_reports_transmitter_empty_and_no_data() {
        let mut uart = Uart16550::new(b"");

        assert_eq!(uart.read(LSR), 0x60);
        assert_eq!(uart.read(IIR), 0x01);
        assert!(!uart.irq_level());
    }

    #[test]
    fn the_9600_baud_divisor_of_12_round_trips_through_the_latch() {
        let mut uart = Uart16550::new(b"");

        uart.write(LCR, 0x80);
        uart.write(THR, 12);
        uart.write(IER, 0);
        let latched = (uart.read(THR), uart.read(IER));
        uart.write(LCR, 0x03);

        assert_eq!(latched, (12, 0));
        assert_eq!(uart.read(LCR), 0x03);
    }

    #[test]
    fn a_byte_written_to_thr_is_handed_back_for_the_host_to_collect() {
        let mut uart = Uart16550::new(b"");
        uart.write(LCR, 0x03);

        assert_eq!(uart.write(THR, b'A'), Some(b'A'));
        assert_eq!(uart.read(LSR), 0x60, "transmit is instant");
    }

    #[test]
    fn enabling_the_thre_interrupt_asserts_the_line_until_iir_is_read() {
        let mut uart = Uart16550::new(b"");

        uart.write(IER, 0x02);
        let asserted = uart.irq_level();
        let id = uart.read(IIR);

        assert!(asserted);
        assert_eq!(id, 0x02, "THRE is interrupt id 0b001 in IIR bits 3:1");
        assert!(!uart.irq_level(), "reading IIR acknowledges THRE");
        assert_eq!(uart.read(IIR), 0x01);
    }

    #[test]
    fn writing_thr_re_arms_the_thre_interrupt() {
        let mut uart = Uart16550::new(b"");
        uart.write(IER, 0x02);
        uart.read(IIR);

        uart.write(THR, b'x');

        assert!(uart.irq_level());
    }

    #[test]
    fn the_fifo_enable_bit_shows_in_the_top_two_bits_of_iir() {
        let mut uart = Uart16550::new(b"");

        uart.write(FCR, 0x01);

        assert_eq!(uart.read(IIR), 0xc1);
    }

    #[test]
    fn received_bytes_are_held_until_the_guest_signals_readiness() {
        let mut uart = Uart16550::new(b"hi");

        let before = uart.read(LSR);
        uart.enable_receive();
        let after = uart.read(LSR);

        assert_eq!(before & 0x01, 0);
        assert_eq!(after & 0x01, 0x01);
    }

    #[test]
    fn receive_data_available_outranks_thre_and_drains_in_order() {
        let mut uart = Uart16550::new(b"ab");
        uart.enable_receive();
        uart.write(IER, 0x03);

        assert_eq!(uart.read(IIR), 0x04);
        assert!(uart.irq_level());
        assert_eq!((uart.read(THR), uart.read(THR)), (b'a', b'b'));
        assert_eq!(uart.read(LSR) & 0x01, 0, "the stream is exhausted");
        assert_eq!(uart.read(THR), 0, "an empty receive buffer reads zero");
        assert_eq!(uart.receive_consumed(), 2);
    }

    #[test]
    fn the_scratch_register_round_trips() {
        let mut uart = Uart16550::new(b"");

        uart.write(SCR, 0xa5);

        assert_eq!(uart.read(SCR), 0xa5);
    }

    #[test]
    fn modem_status_outside_loopback_is_cts_dsr_dcd() {
        let mut uart = Uart16550::new(b"");

        assert_eq!(uart.read(MSR), 0xb0);
    }

    #[test]
    fn loopback_wires_mcr_outputs_to_msr_inputs_and_swallows_transmit() {
        let mut uart = Uart16550::new(b"");

        uart.write(MCR, 0x10 | 0x02 | 0x01);
        let status = uart.read(MSR);
        let transmitted = uart.write(THR, b'z');

        assert_eq!(status, 0x10 | 0x20, "RTS drives CTS (bit 4), DTR drives DSR (bit 5)");
        assert_eq!(transmitted, None);
    }

    #[test]
    fn mcr_keeps_only_its_five_writable_bits() {
        let mut uart = Uart16550::new(b"");

        uart.write(MCR, 0xff);

        assert_eq!(uart.read(MCR), 0x1f);
    }
}
