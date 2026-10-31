//! The MC146818 CMOS RTC at ports 0x70 (index) and 0x71 (data), frozen at the
//! wall time the run began, as a tier-3 sans-IO codec.
//!
//! Without it every read returns 0xff, so the Linux kernel sees
//! "update in progress" forever and spins on `mc146818_avoid_UIP` for over a
//! second. BCD, 24-hour mode (x86 Linux decodes every RTC as BCD whatever the data-mode bit says, so binary registers read as the wrong date); the update-in-progress bit never sets.
//! Register layout per the Motorola MC146818A datasheet, "Time, Calendar and
//! Alarm Locations": 0x00 seconds, 0x02 minutes, 0x04 hours, 0x06 day of week
//! (1-7), 0x07 day of month, 0x08 month, 0x09 year (00-99); 0x0a/0x0b/0x0d are
//! status registers A/B/D; 0x32 is the IBM-PC century byte.

pub const INDEX_PORT: u16 = 0x70;
pub const DATA_PORT: u16 = 0x71;

const REGISTER_COUNT: usize = 128;
const INDEX_MASK: u8 = 0x7f;
const STATUS_A_STABLE: u8 = 0x26;
const STATUS_B_24H_BCD: u8 = 0x02;
const STATUS_D_VALID: u8 = 0x80;
const SECONDS_PER_DAY: i64 = 86_400;

#[derive(Clone, Debug)]
pub struct CmosRtc {
    index: u8,
    registers: [u8; REGISTER_COUNT],
}

fn bcd(value: u8) -> u8 {
    (value / 10) << 4 | (value % 10)
}

struct CivilTime {
    second: u8,
    minute: u8,
    hour: u8,
    weekday_from_sunday: u8,
    day: u8,
    month: u8,
    year: i64,
}

fn civil_from_epoch(seconds_since_epoch: i64) -> CivilTime {
    let days = seconds_since_epoch.div_euclid(SECONDS_PER_DAY);
    let second_of_day = seconds_since_epoch.rem_euclid(SECONDS_PER_DAY);
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097);
    let year_of_era = (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 { month_index + 3 } else { month_index - 9 };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    CivilTime {
        second: (second_of_day % 60) as u8,
        minute: (second_of_day / 60 % 60) as u8,
        hour: (second_of_day / 3600) as u8,
        weekday_from_sunday: (days + 4).rem_euclid(7) as u8,
        day: day as u8,
        month: month as u8,
        year,
    }
}

impl CmosRtc {
    #[must_use]
    pub fn at(seconds_since_epoch: i64) -> Self {
        let time = civil_from_epoch(seconds_since_epoch);
        let mut registers = [0_u8; REGISTER_COUNT];
        registers[0x00] = bcd(time.second);
        registers[0x02] = bcd(time.minute);
        registers[0x04] = bcd(time.hour);
        registers[0x06] = time.weekday_from_sunday + 1;
        registers[0x07] = bcd(time.day);
        registers[0x08] = bcd(time.month);
        registers[0x09] = bcd(time.year.rem_euclid(100) as u8);
        registers[0x0a] = STATUS_A_STABLE;
        registers[0x0b] = STATUS_B_24H_BCD;
        registers[0x0d] = STATUS_D_VALID;
        registers[0x32] = bcd(time.year.div_euclid(100) as u8);
        Self { index: 0, registers }
    }

    #[must_use]
    pub fn read(&self, port: u16) -> u8 {
        match port {
            DATA_PORT => self.registers[usize::from(self.index)],
            _ => 0xff,
        }
    }

    pub fn write(&mut self, port: u16, value: u8) {
        if port == INDEX_PORT {
            self.index = value & INDEX_MASK;
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    fn read_register(rtc: &mut CmosRtc, register: u8) -> u8 {
        rtc.write(INDEX_PORT, register);
        rtc.read(DATA_PORT)
    }

    fn time_registers(rtc: &mut CmosRtc) -> [u8; 8] {
        [0x00, 0x02, 0x04, 0x06, 0x07, 0x08, 0x09, 0x32].map(|register| read_register(rtc, register))
    }

    #[test]
    fn the_unix_epoch_is_thursday_1_january_1970_in_bcd() {
        let mut rtc = CmosRtc::at(0);

        assert_eq!(time_registers(&mut rtc), [0x00, 0x00, 0x00, 5, 0x01, 0x01, 0x70, 0x19]);
    }

    #[test]
    fn a_2023_instant_decodes_to_its_utc_calendar_fields() {
        let mut rtc = CmosRtc::at(1_700_000_000);

        assert_eq!(time_registers(&mut rtc), [0x20, 0x13, 0x22, 3, 0x14, 0x11, 0x23, 0x20]);
    }

    #[test]
    fn a_leap_day_in_a_century_year_divisible_by_400_is_february_29() {
        let mut rtc = CmosRtc::at(951_782_400);

        assert_eq!(time_registers(&mut rtc), [0x00, 0x00, 0x00, 3, 0x29, 0x02, 0x00, 0x20]);
    }

    #[test]
    fn the_status_registers_never_report_update_in_progress() {
        let mut rtc = CmosRtc::at(1_700_000_000);

        assert_eq!(read_register(&mut rtc, 0x0a) & 0x80, 0);
        assert_eq!(read_register(&mut rtc, 0x0b), 0x02);
        assert_eq!(read_register(&mut rtc, 0x0d), 0x80);
    }

    #[test]
    fn the_index_port_reads_ff_and_the_data_port_ignores_writes() {
        let mut rtc = CmosRtc::at(0);
        rtc.write(INDEX_PORT, 0x07);

        rtc.write(DATA_PORT, 0x55);

        assert_eq!(rtc.read(INDEX_PORT), 0xff);
        assert_eq!(rtc.read(DATA_PORT), 1, "day of month is unchanged");
    }

    #[test]
    fn the_index_register_keeps_seven_bits() {
        let mut rtc = CmosRtc::at(0);

        rtc.write(INDEX_PORT, 0x87);

        assert_eq!(rtc.read(DATA_PORT), 1, "bit 7 (NMI mask) is not part of the index");
    }

    #[test]
    fn an_instant_before_the_epoch_still_decodes() {
        let mut rtc = CmosRtc::at(-1);

        assert_eq!(time_registers(&mut rtc), [0x59, 0x59, 0x23, 4, 0x31, 0x12, 0x69, 0x19]);
    }
}
