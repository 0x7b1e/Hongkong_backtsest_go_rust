//! Hong Kong time handling shared by every module.
//!
//! All instants are stored as UTC nanoseconds since the Unix epoch, exactly
//! like ``DatetimeIndex.asi8`` in the Python implementation. Local values are
//! interpreted in ``Asia/Hong_Kong``.

use anyhow::{anyhow, bail, Result};
use chrono::{
    DateTime, Datelike, Duration, LocalResult, NaiveDate, NaiveDateTime, NaiveTime, Offset,
    TimeZone, Timelike,
};
use chrono_tz::Tz;

pub const TIMEZONE: &str = "Asia/Hong_Kong";
pub const NS_PER_SEC: i64 = 1_000_000_000;
pub const NS_PER_MIN: i64 = 60 * NS_PER_SEC;
pub const NS_PER_HOUR: i64 = 60 * NS_PER_MIN;
pub const NS_PER_DAY: i64 = 24 * NS_PER_HOUR;

/// Hong Kong has used a fixed UTC+08:00 offset since 1980.
const FIXED_FROM_UTC_NS: i64 = 315_504_000 * NS_PER_SEC; // 1979-12-31T16:00Z
const FIXED_OFFSET_NS: i64 = 8 * NS_PER_HOUR;

pub fn hk() -> Tz {
    chrono_tz::Asia::Hong_Kong
}

/// Offset (local - UTC) in nanoseconds at a UTC instant.
pub fn offset_at_utc(utc_ns: i64) -> i64 {
    if utc_ns >= FIXED_FROM_UTC_NS {
        return FIXED_OFFSET_NS;
    }
    let dt = DateTime::from_timestamp_nanos(utc_ns);
    let local = hk().from_utc_datetime(&dt.naive_utc());
    local.offset().fix().local_minus_utc() as i64 * NS_PER_SEC
}

/// Convert a UTC instant into HK wall-clock nanoseconds.
pub fn utc_to_local(utc_ns: i64) -> i64 {
    utc_ns + offset_at_utc(utc_ns)
}

/// Convert HK wall-clock nanoseconds into a UTC instant.
pub fn local_to_utc(local_ns: i64) -> Result<i64> {
    let candidate = local_ns - FIXED_OFFSET_NS;
    if candidate >= FIXED_FROM_UTC_NS {
        return Ok(candidate);
    }
    let naive = DateTime::from_timestamp_nanos(local_ns).naive_utc();
    match hk().from_local_datetime(&naive) {
        LocalResult::Single(dt) => Ok(dt
            .timestamp_nanos_opt()
            .ok_or_else(|| anyhow!("Timestamp out of range"))?),
        LocalResult::Ambiguous(_, _) => bail!("Ambiguous Hong Kong local time: {naive}"),
        LocalResult::None => bail!("Nonexistent Hong Kong local time: {naive}"),
    }
}

/// Convert wall-clock nanoseconds in an arbitrary IANA timezone to UTC.
pub fn zone_local_to_utc(local_ns: i64, zone: &str) -> Result<i64> {
    if zone == "UTC" || zone == "utc" || zone == "Etc/UTC" {
        return Ok(local_ns);
    }
    if zone == TIMEZONE {
        return local_to_utc(local_ns);
    }
    if let Some(offset) = parse_fixed_offset(zone) {
        return Ok(local_ns - offset);
    }
    let tz: Tz = zone
        .parse()
        .map_err(|_| anyhow!("Invalid timezone {zone:?}"))?;
    let naive = DateTime::from_timestamp_nanos(local_ns).naive_utc();
    match tz.from_local_datetime(&naive) {
        LocalResult::Single(dt) => Ok(dt.timestamp_nanos_opt().unwrap()),
        LocalResult::Ambiguous(_, _) => bail!("Ambiguous local time {naive} in {zone}"),
        LocalResult::None => bail!("Nonexistent local time {naive} in {zone}"),
    }
}

/// Convert a UTC instant to wall-clock nanoseconds in a timezone.
pub fn utc_to_zone_local(utc_ns: i64, zone: &str) -> Result<i64> {
    if zone == "UTC" || zone == "utc" || zone == "Etc/UTC" {
        return Ok(utc_ns);
    }
    if zone == TIMEZONE {
        return Ok(utc_to_local(utc_ns));
    }
    if let Some(offset) = parse_fixed_offset(zone) {
        return Ok(utc_ns + offset);
    }
    let tz: Tz = zone
        .parse()
        .map_err(|_| anyhow!("Invalid timezone {zone:?}"))?;
    let dt = DateTime::from_timestamp_nanos(utc_ns);
    let local = tz.from_utc_datetime(&dt.naive_utc());
    Ok(utc_ns + local.offset().fix().local_minus_utc() as i64 * NS_PER_SEC)
}

/// Validate a timezone name the way ``tz_localize`` would accept it.
pub fn validate_zone(zone: &str) -> Result<()> {
    if zone == "UTC" || zone == TIMEZONE || parse_fixed_offset(zone).is_some() {
        return Ok(());
    }
    zone.parse::<Tz>()
        .map(|_| ())
        .map_err(|_| anyhow!("Invalid source_timezone {zone:?}"))
}

fn parse_fixed_offset(zone: &str) -> Option<i64> {
    let bytes = zone.as_bytes();
    if bytes.len() == 6 && (bytes[0] == b'+' || bytes[0] == b'-') && bytes[3] == b':' {
        let hours: i64 = zone[1..3].parse().ok()?;
        let minutes: i64 = zone[4..6].parse().ok()?;
        let sign = if bytes[0] == b'-' { -1 } else { 1 };
        return Some(sign * (hours * NS_PER_HOUR + minutes * NS_PER_MIN));
    }
    None
}

/// Local midnight (as a UTC instant) of the HK day containing ``utc_ns``.
pub fn normalize(utc_ns: i64) -> i64 {
    let local = utc_to_local(utc_ns);
    let midnight = local.div_euclid(NS_PER_DAY) * NS_PER_DAY;
    local_to_utc(midnight).unwrap_or(midnight - FIXED_OFFSET_NS)
}

/// Local HK calendar date.
pub fn local_date(utc_ns: i64) -> NaiveDate {
    let local = utc_to_local(utc_ns);
    let days = local.div_euclid(NS_PER_DAY);
    NaiveDate::from_num_days_from_ce_opt((days + 719_163) as i32).expect("date in range")
}

/// Nanoseconds since local HK midnight.
pub fn local_time_of_day(utc_ns: i64) -> i64 {
    utc_to_local(utc_ns).rem_euclid(NS_PER_DAY)
}

/// Days since the Unix epoch for a date.
pub fn date_to_days(date: NaiveDate) -> i64 {
    date.num_days_from_ce() as i64 - 719_163
}

pub fn days_to_date(days: i64) -> NaiveDate {
    NaiveDate::from_num_days_from_ce_opt((days + 719_163) as i32).expect("date in range")
}

/// Midnight HK (as a UTC instant) for a calendar date.
pub fn date_midnight_utc(date: NaiveDate) -> i64 {
    let local = date_to_days(date) * NS_PER_DAY;
    local_to_utc(local).unwrap_or(local - FIXED_OFFSET_NS)
}

/// Parse an ISO ``YYYY-MM-DD`` date as Python's ``date.fromisoformat``.
pub fn parse_iso_date(text: &str) -> Result<NaiveDate> {
    let invalid = || anyhow!("Invalid isoformat string: {text:?}");
    let text = text.trim_end();
    if text.len() == 10 && text.as_bytes()[4] == b'-' && text.as_bytes()[7] == b'-' {
        return NaiveDate::parse_from_str(text, "%Y-%m-%d").map_err(|_| invalid());
    }
    if text.len() == 8 && text.bytes().all(|b| b.is_ascii_digit()) {
        return NaiveDate::parse_from_str(text, "%Y%m%d").map_err(|_| invalid());
    }
    Err(invalid())
}

/// Parse a local clock as Python's ``time.fromisoformat``; returns ns of day.
pub fn parse_iso_time(text: &str) -> Result<i64> {
    let invalid = || anyhow!("Invalid isoformat string: {text:?}");
    let mut s = text;
    if let Some(rest) = s.strip_prefix('T') {
        s = rest;
    }
    let (clock, fraction) = match s.find(['.', ',']) {
        Some(position) => (&s[..position], Some(&s[position + 1..])),
        None => (s, None),
    };
    let digits: String;
    let parts: Vec<&str> = if clock.contains(':') {
        clock.split(':').collect()
    } else {
        if !clock.bytes().all(|b| b.is_ascii_digit()) || clock.len() % 2 != 0 {
            return Err(invalid());
        }
        digits = clock.to_string();
        (0..digits.len())
            .step_by(2)
            .map(|i| &digits[i..i + 2])
            .collect()
    };
    if parts.is_empty() || parts.len() > 3 || parts.iter().any(|p| p.len() != 2) {
        return Err(invalid());
    }
    let values: Vec<i64> = parts
        .iter()
        .map(|p| p.parse::<i64>().map_err(|_| invalid()))
        .collect::<Result<_>>()?;
    let hour = values[0];
    let minute = *values.get(1).unwrap_or(&0);
    let second = *values.get(2).unwrap_or(&0);
    if hour > 23 || minute > 59 || second > 59 {
        return Err(invalid());
    }
    let mut nanos = 0;
    if let Some(fraction) = fraction {
        if parts.len() != 3
            || fraction.is_empty()
            || fraction.len() > 6 && fraction.len() != 9
            || !fraction.bytes().all(|b| b.is_ascii_digit())
        {
            return Err(invalid());
        }
        let padded = format!("{fraction:0<9}");
        nanos = padded[..9].parse::<i64>().map_err(|_| invalid())?;
        if fraction.len() <= 6 {
            nanos = nanos / 1000 * 1000;
        }
    }
    Ok(hour * NS_PER_HOUR + minute * NS_PER_MIN + second * NS_PER_SEC + nanos)
}

/// A parsed date/time string: naive wall-clock ns, optional UTC offset ns,
/// and whether the source text was a bare date.
#[derive(Debug, Clone, Copy)]
pub struct ParsedTime {
    pub naive_ns: i64,
    pub offset_ns: Option<i64>,
    pub date_only: bool,
}

impl ParsedTime {
    /// Resolve to UTC, localizing naive values in ``zone``.
    pub fn to_utc(self, zone: &str) -> Result<i64> {
        match self.offset_ns {
            Some(offset) => Ok(self.naive_ns - offset),
            None => zone_local_to_utc(self.naive_ns, zone),
        }
    }
}

/// Parse common ISO-like date/time spellings accepted by pandas' mixed parser.
pub fn parse_datetime_text(raw: &str) -> Result<ParsedTime> {
    let text = raw.trim();
    let invalid = || anyhow!("Invalid datetime string: {raw:?}");
    if text.is_empty() {
        return Err(invalid());
    }
    let (date_part, rest) = split_date(text).ok_or_else(invalid)?;
    let date = parse_date_token(date_part).ok_or_else(invalid)?;
    let mut naive = date_to_days(date) * NS_PER_DAY;
    let rest = rest.trim_start_matches(['T', ' ']);
    if rest.is_empty() {
        return Ok(ParsedTime {
            naive_ns: naive,
            offset_ns: None,
            date_only: true,
        });
    }
    // Split a trailing zone designator.
    let (clock, offset) = split_offset(rest).ok_or_else(invalid)?;
    let clock = clock.trim();
    if !clock.is_empty() {
        let clock_ns = if clock.len() == 2 && clock.bytes().all(|b| b.is_ascii_digit()) {
            parse_iso_time(&format!("{clock}:00"))
        } else {
            parse_iso_time(clock)
        }
        .map_err(|_| invalid())?;
        naive += clock_ns;
    }
    Ok(ParsedTime {
        naive_ns: naive,
        offset_ns: offset,
        date_only: false,
    })
}

fn split_date(text: &str) -> Option<(&str, &str)> {
    let bytes = text.as_bytes();
    // YYYY-MM-DD or YYYY/MM/DD
    if bytes.len() >= 10
        && bytes[..4].iter().all(u8::is_ascii_digit)
        && (bytes[4] == b'-' || bytes[4] == b'/')
    {
        // Allow single-digit month/day.
        let mut index = 5;
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            index += 1;
        }
        if index >= bytes.len() || bytes[index] != bytes[4] {
            return None;
        }
        index += 1;
        let start_day = index;
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            index += 1;
        }
        if index == start_day {
            return None;
        }
        return Some((&text[..index], &text[index..]));
    }
    if bytes.len() >= 8
        && bytes[..8].iter().all(u8::is_ascii_digit)
        && (bytes.len() == 8 || !bytes[8].is_ascii_digit())
    {
        return Some((&text[..8], &text[8..]));
    }
    // YYYY-M-D shorter forms
    if bytes.len() >= 8
        && bytes[..4].iter().all(u8::is_ascii_digit)
        && (bytes[4] == b'-' || bytes[4] == b'/')
    {
        let separator = bytes[4];
        let mut index = 5;
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            index += 1;
        }
        if index >= bytes.len() || bytes[index] != separator {
            return None;
        }
        index += 1;
        let start_day = index;
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            index += 1;
        }
        if index == start_day {
            return None;
        }
        return Some((&text[..index], &text[index..]));
    }
    None
}

fn parse_date_token(token: &str) -> Option<NaiveDate> {
    if token.len() == 8 && token.bytes().all(|b| b.is_ascii_digit()) {
        return NaiveDate::parse_from_str(token, "%Y%m%d").ok();
    }
    let separator = token.as_bytes()[4] as char;
    let parts: Vec<&str> = token.split(separator).collect();
    if parts.len() != 3 {
        return None;
    }
    let year: i32 = parts[0].parse().ok()?;
    let month: u32 = parts[1].parse().ok()?;
    let day: u32 = parts[2].parse().ok()?;
    NaiveDate::from_ymd_opt(year, month, day)
}

fn split_offset(rest: &str) -> Option<(&str, Option<i64>)> {
    if let Some(clock) = rest.strip_suffix('Z').or_else(|| rest.strip_suffix('z')) {
        return Some((clock, Some(0)));
    }
    if let Some(clock) = rest.strip_suffix(" UTC") {
        return Some((clock, Some(0)));
    }
    // Look for a sign after the clock digits.
    if let Some(position) = rest.rfind(['+', '-']) {
        if position == 0 && rest.len() <= 6 {
            // Offset only, no clock.
        }
        let zone = &rest[position..];
        let sign = if zone.starts_with('-') { -1 } else { 1 };
        let digits: String = zone[1..].chars().filter(|c| *c != ':').collect();
        if digits.bytes().all(|b| b.is_ascii_digit()) && (digits.len() == 2 || digits.len() == 4) {
            let hours: i64 = digits[..2].parse().ok()?;
            let minutes: i64 = if digits.len() == 4 {
                digits[2..].parse().ok()?
            } else {
                0
            };
            return Some((
                &rest[..position],
                Some(sign * (hours * NS_PER_HOUR + minutes * NS_PER_MIN)),
            ));
        }
    }
    Some((rest, None))
}

/// Format a UTC instant as an ISO string in HK, like ``Timestamp.isoformat``.
pub fn isoformat_hk(utc_ns: i64) -> String {
    isoformat_in(utc_ns, TIMEZONE)
}

/// Format a UTC instant as an ISO string in UTC with ``+00:00``.
pub fn isoformat_utc(utc_ns: i64) -> String {
    isoformat_in(utc_ns, "UTC")
}

fn isoformat_in(utc_ns: i64, zone: &str) -> String {
    let local = utc_to_zone_local(utc_ns, zone).unwrap_or(utc_ns);
    let offset = local - utc_ns;
    let dt = DateTime::from_timestamp_nanos(local).naive_utc();
    let nanos = dt.nanosecond();
    let mut out = dt.format("%Y-%m-%dT%H:%M:%S").to_string();
    if nanos != 0 {
        if nanos.is_multiple_of(1000) {
            out.push_str(&format!(".{:06}", nanos / 1000));
        } else {
            out.push_str(&format!(".{nanos:09}"));
        }
    }
    let sign = if offset < 0 { '-' } else { '+' };
    let minutes = offset.abs() / NS_PER_MIN;
    out.push_str(&format!("{sign}{:02}:{:02}", minutes / 60, minutes % 60));
    out
}

/// ``strftime`` of a UTC instant in HK.
pub fn strftime_hk(utc_ns: i64, format: &str) -> String {
    let local = utc_to_local(utc_ns);
    DateTime::from_timestamp_nanos(local)
        .naive_utc()
        .format(format)
        .to_string()
}

pub fn naive_from_ns(ns: i64) -> NaiveDateTime {
    DateTime::from_timestamp_nanos(ns).naive_utc()
}

pub fn time_of(ns_of_day: i64) -> NaiveTime {
    NaiveTime::from_num_seconds_from_midnight_opt(
        (ns_of_day / NS_PER_SEC) as u32,
        (ns_of_day % NS_PER_SEC) as u32,
    )
    .unwrap_or_default()
}

/// Parse a pandas-style ``Timedelta`` string: "0s", "5m", "5min", "1h30m",
/// "00:05:00", "1 days", "250ms", "1d".
pub fn parse_timedelta(text: &str) -> Result<i64> {
    let raw = text.trim();
    let invalid = || anyhow!("Invalid timedelta {text:?}");
    if raw.is_empty() {
        return Err(invalid());
    }
    let (sign, body) = match raw.strip_prefix('-') {
        Some(rest) => (-1, rest.trim()),
        None => (1, raw.strip_prefix('+').unwrap_or(raw).trim()),
    };
    // Clock form, optionally prefixed by "N days".
    let mut total: i128 = 0;
    let mut rest = body;
    if let Some(position) = rest.find("day") {
        let count: f64 = rest[..position].trim().parse().map_err(|_| invalid())?;
        total += (count * NS_PER_DAY as f64) as i128;
        rest = rest[position..]
            .trim_start_matches("days")
            .trim_start_matches("day");
        rest = rest.trim_start_matches(',').trim();
        if rest.is_empty() {
            return Ok(sign * total as i64);
        }
    }
    if rest.contains(':') {
        let clock = parse_clock_duration(rest).ok_or_else(invalid)?;
        total += clock as i128;
        return Ok(sign * total as i64);
    }
    // Unit components, e.g. "1h30m", "5 min", "0s", "250ms".
    let lower = rest.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let mut index = 0;
    let mut any = false;
    while index < bytes.len() {
        while index < bytes.len() && bytes[index] == b' ' {
            index += 1;
        }
        if index >= bytes.len() {
            break;
        }
        let start = index;
        while index < bytes.len() && (bytes[index].is_ascii_digit() || bytes[index] == b'.') {
            index += 1;
        }
        if start == index {
            return Err(invalid());
        }
        let number: f64 = lower[start..index].parse().map_err(|_| invalid())?;
        while index < bytes.len() && bytes[index] == b' ' {
            index += 1;
        }
        let unit_start = index;
        while index < bytes.len() && bytes[index].is_ascii_alphabetic() {
            index += 1;
        }
        let unit = &lower[unit_start..index];
        let scale: f64 = match unit {
            "" => {
                // Bare number: nanoseconds, as pandas.
                1.0
            }
            "w" | "week" | "weeks" => (7 * NS_PER_DAY) as f64,
            "d" | "day" | "days" => NS_PER_DAY as f64,
            "h" | "hr" | "hour" | "hours" => NS_PER_HOUR as f64,
            "m" | "min" | "mins" | "minute" | "minutes" | "t" => NS_PER_MIN as f64,
            "s" | "sec" | "secs" | "second" | "seconds" => NS_PER_SEC as f64,
            "ms" | "milli" | "millis" | "millisecond" | "milliseconds" | "l" => 1e6,
            "us" | "micro" | "micros" | "microsecond" | "microseconds" | "u" => 1e3,
            "ns" | "nano" | "nanos" | "nanosecond" | "nanoseconds" | "n" => 1.0,
            _ => return Err(invalid()),
        };
        total += (number * scale).round() as i128;
        any = true;
    }
    if !any {
        return Err(invalid());
    }
    if total > i64::MAX as i128 {
        return Err(invalid());
    }
    Ok(sign * total as i64)
}

fn parse_clock_duration(text: &str) -> Option<i64> {
    let (clock, fraction) = match text.split_once('.') {
        Some((c, f)) => (c, Some(f)),
        None => (text, None),
    };
    let parts: Vec<&str> = clock.split(':').collect();
    if parts.len() != 3 {
        return None;
    }
    let hours: i64 = parts[0].parse().ok()?;
    let minutes: i64 = parts[1].parse().ok()?;
    let seconds: i64 = parts[2].parse().ok()?;
    let mut nanos = 0;
    if let Some(fraction) = fraction {
        let padded = format!("{fraction:0<9}");
        nanos = padded[..9].parse().ok()?;
    }
    Some(hours * NS_PER_HOUR + minutes * NS_PER_MIN + seconds * NS_PER_SEC + nanos)
}

/// pandas ``Timestamp(x).value`` for display of durations not needed; helper.
pub fn duration_ns(duration: Duration) -> i64 {
    duration.num_nanoseconds().unwrap_or(i64::MAX)
}

pub fn year_of(utc_ns: i64) -> i32 {
    local_date(utc_ns).year()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_round_trip() {
        let utc = local_to_utc(
            date_to_days(NaiveDate::from_ymd_opt(2024, 1, 2).unwrap()) * NS_PER_DAY
                + 16 * NS_PER_HOUR,
        )
        .unwrap();
        assert_eq!(strftime_hk(utc, "%Y-%m-%d %H:%M"), "2024-01-02 16:00");
        assert_eq!(isoformat_hk(utc), "2024-01-02T16:00:00+08:00");
        assert_eq!(local_time_of_day(utc), 16 * NS_PER_HOUR);
    }

    #[test]
    fn parses_times() {
        assert_eq!(
            parse_iso_time("09:30").unwrap(),
            9 * NS_PER_HOUR + 30 * NS_PER_MIN
        );
        assert_eq!(
            parse_iso_time("0930").unwrap(),
            9 * NS_PER_HOUR + 30 * NS_PER_MIN
        );
        assert!(parse_iso_time("25:00").is_err());
        let parsed = parse_datetime_text("2024-01-02").unwrap();
        assert!(parsed.date_only);
        let parsed = parse_datetime_text("2024-01-02T16:00:00+08:00").unwrap();
        assert_eq!(parsed.offset_ns, Some(8 * NS_PER_HOUR));
        assert_eq!(parse_timedelta("5min").unwrap(), 5 * NS_PER_MIN);
        assert_eq!(parse_timedelta("0s").unwrap(), 0);
        assert_eq!(parse_timedelta("00:05:00").unwrap(), 5 * NS_PER_MIN);
    }
}
