//! Hong Kong cash-equity execution rules and dated transaction charges.

use anyhow::{bail, Result};

use crate::config::BacktestConfig;
use crate::timeutil::{self, NS_PER_HOUR, NS_PER_MIN};

/// Per-side charges for a filled child order, in Hong Kong dollars.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Fees {
    pub stamp_duty: f64,
    pub sfc_levy: f64,
    pub afrc_levy: f64,
    pub trading_fee: f64,
    pub trading_tariff: f64,
    pub commission: f64,
}

impl Fees {
    /// Sum of independently rounded charges (sequential, as Python ``sum``).
    pub fn total(&self) -> f64 {
        let mut total = 0.0;
        for value in [
            self.stamp_duty,
            self.sfc_levy,
            self.afrc_levy,
            self.trading_fee,
            self.trading_tariff,
            self.commission,
        ] {
            total += value;
        }
        total
    }
}

/// Round nonnegative amounts to cents using commercial half-up rules.
pub fn cent(value: f64) -> f64 {
    (value * 100.0 + 0.5 + 1e-9).floor() / 100.0
}

/// Calculate dated charges with supplied historical stamp exemptions.
///
/// Each child fill is one transaction. Stamp duty rounds up to HKD 1; the
/// levy, exchange fee and commission round separately to cents.
/// Clearing/custody charges and broker-specific taxes are not included.
/// Rates before November 2014 are deliberately unsupported.
pub fn transaction_fees(
    notional: f64,
    timestamp: i64,
    config: &BacktestConfig,
    stamp_duty_exempt: bool,
) -> Result<Fees> {
    if !notional.is_finite() || notional < 0.0 {
        bail!("Trade notional must be finite and nonnegative");
    }
    if notional == 0.0 {
        return Ok(Fees::default());
    }
    let day = timeutil::local_date(timestamp)
        .format("%Y-%m-%d")
        .to_string();
    if day.as_str() < "2014-11-01" {
        bail!("Transaction fees before 2014-11-01 unsupported");
    }
    let stamp_rate = if "2021-08-01" <= day.as_str() && day.as_str() < "2023-11-17" {
        0.0013
    } else {
        0.001
    };
    let trading_rate = if day.as_str() >= "2023-01-01" {
        0.0000565
    } else {
        0.00005
    };
    Ok(Fees {
        stamp_duty: if stamp_duty_exempt {
            0.0
        } else {
            (notional * stamp_rate - 1e-10).ceil()
        },
        sfc_levy: cent(notional * 0.000027),
        afrc_levy: if day.as_str() >= "2022-01-01" {
            cent(notional * 0.0000015)
        } else {
            0.0
        },
        trading_fee: cent(notional * trading_rate),
        trading_tariff: if day.as_str() >= "2023-01-01" {
            0.0
        } else {
            0.5
        },
        commission: cent(f64::max(
            config.minimum_commission,
            notional * config.commission_bps / 10_000.0,
        )),
    })
}

/// Return whether an open falls within a continuous trading session.
///
/// Calendar holidays and half days are controlled by the supplied panel.
/// Auction executions are outside this version's execution model.
pub fn is_continuous_session(timestamp: i64) -> bool {
    let current = timeutil::local_time_of_day(timestamp);
    let t = |h: i64, m: i64| h * NS_PER_HOUR + m * NS_PER_MIN;
    (t(9, 30) <= current && current < t(12, 0)) || (t(13, 0) <= current && current < t(16, 0))
}

/// Check an inclusive-start, exclusive-end local execution window.
pub fn within_window(timestamp: i64, start: i64, end: i64) -> bool {
    let current = timeutil::local_time_of_day(timestamp);
    start <= current && current < end
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(date: &str) -> i64 {
        let day = timeutil::parse_iso_date(date).unwrap();
        timeutil::date_midnight_utc(day) + 10 * NS_PER_HOUR
    }

    #[test]
    fn dated_rates() {
        let config = BacktestConfig::new("2024-01-02", "2024-01-03").unwrap();
        let fees = transaction_fees(100_000.0, at("2024-01-02"), &config, false).unwrap();
        assert_eq!(fees.stamp_duty, 100.0);
        assert_eq!(fees.sfc_levy, 2.7);
        assert_eq!(fees.afrc_levy, 0.15);
        assert_eq!(fees.trading_fee, 5.65);
        assert_eq!(fees.trading_tariff, 0.0);
        assert_eq!(fees.commission, 20.0);
        let fees = transaction_fees(100_000.0, at("2022-06-01"), &config, false).unwrap();
        assert_eq!(fees.stamp_duty, 130.0);
        assert_eq!(fees.trading_tariff, 0.5);
        assert!(transaction_fees(1.0, at("2014-10-31"), &config, false).is_err());
    }
}
