//! Generate reproducible synthetic fixtures for integration tests.

use std::fs;
use std::path::Path;

use anyhow::{bail, Result};
use chrono::{Datelike, Duration, Weekday};
use serde_json::{json, Map, Value};

use crate::config::BacktestConfig;
use crate::data;
use crate::errors::Error;
use crate::frames::{OutCol, OutFrame};
use crate::matrix;
use crate::nprandom::Generator;
use crate::pyjson::{self, Options};
use crate::timeutil::{self, NS_PER_DAY, NS_PER_HOUR, NS_PER_MIN};

/// Options of ``create_demo`` beyond the directory and frequency.
#[derive(Debug, Clone)]
pub struct DemoOptions {
    pub asset_count: usize,
    pub trading_days: Option<usize>,
    pub factor_count: usize,
    pub seed: u64,
}

impl Default for DemoOptions {
    fn default() -> Self {
        DemoOptions {
            asset_count: 20,
            trading_days: None,
            factor_count: 2,
            seed: 101,
        }
    }
}

/// ``pd.bdate_range(start, periods=n)``: weekdays from ``start``.
fn business_days(start: chrono::NaiveDate, periods: usize) -> Vec<chrono::NaiveDate> {
    let mut days = Vec::with_capacity(periods);
    let mut day = start;
    while days.len() < periods {
        if !matches!(day.weekday(), Weekday::Sat | Weekday::Sun) {
            days.push(day);
        }
        day += Duration::days(1);
    }
    days
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    fs::write(
        path,
        pyjson::dumps(
            value,
            Options {
                sort_keys: false,
                indent: Some(2),
                ensure_ascii: true,
            },
        ) + "\n",
    )?;
    Ok(())
}

/// Write a labelled synthetic multi-security, multi-factor dataset.
///
/// Returns paths and assumptions for running prepare and backtest commands.
/// Securities are synthetic, never real historical constituents.
pub fn create_demo(
    directory: &Path,
    frequency: &str,
    options: &DemoOptions,
) -> Result<Map<String, Value>> {
    if directory.exists() {
        bail!(Error::FileExists(directory.display().to_string()));
    }
    if options.asset_count < 10 || options.factor_count < 1 {
        bail!("Demo needs at least ten assets and one factor");
    }
    if !["1d", "5m", "1m"].contains(&frequency) {
        bail!("Unsupported demo frequency");
    }
    let trading_days = options
        .trading_days
        .unwrap_or(if frequency == "1d" { 40 } else { 8 });
    if trading_days < 2 {
        bail!("Demo needs at least two trading days");
    }
    let start = timeutil::parse_iso_date("2024-01-02")?;
    let days: Vec<i64> = business_days(start, trading_days)
        .into_iter()
        .map(timeutil::date_midnight_utc)
        .collect();
    let sessions: Vec<(i64, i64, i64, i64, i64)> = days
        .iter()
        .map(|&d| {
            (
                d,
                9 * NS_PER_HOUR + 30 * NS_PER_MIN,
                16 * NS_PER_HOUR,
                12 * NS_PER_HOUR,
                13 * NS_PER_HOUR,
            )
        })
        .collect();
    let (timestamps, _) = data::bar_grid(&sessions, frequency)?;
    let symbols: Vec<String> = (1..=options.asset_count)
        .map(|i| format!("{i:05}.HK"))
        .collect();
    let (rows, cols) = (timestamps.len(), symbols.len());
    let mut rng = Generator::default_rng(options.seed);
    let sigma = if frequency == "1d" { 0.008 } else { 0.001 };
    let innovations = rng.normal(0.0001, sigma, rows * cols);
    let mut cumulative = vec![0.0; cols];
    let mut closes = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            cumulative[c] += innovations[r * cols + c];
            closes[r * cols + c] = 30.0 * cumulative[c].exp();
        }
    }
    let mut opens = vec![0.0; rows * cols];
    for c in 0..cols {
        opens[c] = closes[c] / innovations[c].exp();
    }
    for r in 1..rows {
        for c in 0..cols {
            opens[r * cols + c] = closes[(r - 1) * cols + c];
        }
    }
    let volumes = rng.integers(200_000, 500_000, rows * cols);
    let repeated_times: Vec<i64> = timestamps
        .iter()
        .flat_map(|&t| std::iter::repeat_n(t, cols))
        .collect();
    let tiled_symbols: Vec<String> = (0..rows).flat_map(|_| symbols.iter().cloned()).collect();
    let mut prices = OutFrame::new();
    prices.push("timestamp", OutCol::Ts(repeated_times.clone()));
    prices.push("symbol", OutCol::Str(tiled_symbols.clone()));
    prices.push("open", OutCol::F64(opens.clone()));
    prices.push("close", OutCol::F64(closes.clone()));
    prices.push(
        "high",
        OutCol::F64(
            (0..rows * cols)
                .map(|i| opens[i].max(closes[i]) * 1.002)
                .collect(),
        ),
    );
    prices.push(
        "low",
        OutCol::F64(
            (0..rows * cols)
                .map(|i| opens[i].min(closes[i]) * 0.998)
                .collect(),
        ),
    );
    prices.push("volume", OutCol::I64(volumes));
    prices.push("lot_size", OutCol::I64(vec![100; rows * cols]));
    prices.push("price_basis", OutCol::Str(vec!["raw".into(); rows * cols]));
    let momentum: Vec<f64> = (0..rows * cols)
        .map(|i| {
            let r = i / cols;
            if r < 5 {
                f64::NAN
            } else {
                closes[i] / closes[i - 5 * cols] - 1.0
            }
        })
        .collect();
    fs::create_dir_all(directory)?;
    prices.write_parquet(&directory.join("prices.parquet"))?;
    let row_mean =
        |values: &[f64], r: usize| matrix::sum(&values[r * cols..(r + 1) * cols]) / cols as f64;
    let mut benchmark = OutFrame::new();
    benchmark.push("timestamp", OutCol::Ts(timestamps.clone()));
    benchmark.push(
        "open",
        OutCol::F64((0..rows).map(|r| row_mean(&opens, r) * 100.0).collect()),
    );
    benchmark.push(
        "close",
        OutCol::F64((0..rows).map(|r| row_mean(&closes, r) * 100.0).collect()),
    );
    benchmark.write_parquet(&directory.join("benchmark.parquet"))?;
    let mut constituents = OutFrame::new();
    constituents.push("effective_date", OutCol::Ts(vec![days[0]; cols]));
    constituents.push("available_at", OutCol::Ts(vec![days[0] - NS_PER_DAY; cols]));
    constituents.push("symbol", OutCol::Str(symbols.clone()));
    constituents.push("weight", OutCol::F64(vec![1.0 / cols as f64; cols]));
    constituents.write_parquet(&directory.join("constituents.parquet"))?;
    let mut calendar = OutFrame::new();
    calendar.push("date", OutCol::Ts(days.clone()));
    calendar.push("open", OutCol::Str(vec!["09:30".into(); days.len()]));
    calendar.push("close", OutCol::Str(vec!["16:00".into(); days.len()]));
    calendar.push("break_start", OutCol::Str(vec!["12:00".into(); days.len()]));
    calendar.push("break_end", OutCol::Str(vec!["13:00".into(); days.len()]));
    calendar.write_parquet(&directory.join("sessions.parquet"))?;
    for index in 0..options.factor_count {
        let (name, values): (String, Vec<f64>) = match index {
            0 => ("momentum_5".into(), momentum.clone()),
            1 => ("reversal_5".into(), momentum.iter().map(|v| -v).collect()),
            _ => {
                let noise = rng.normal(0.0, 0.002, momentum.len());
                (
                    format!("candidate_{index:05}"),
                    momentum.iter().zip(&noise).map(|(v, n)| v + n).collect(),
                )
            }
        };
        let partition = directory.join("signals").join(format!("factor={name}"));
        fs::create_dir_all(&partition)?;
        let mut frame = OutFrame::new();
        frame.push("timestamp", OutCol::Ts(repeated_times.clone()));
        frame.push("available_at", OutCol::Ts(repeated_times.clone()));
        frame.push("symbol", OutCol::Str(tiled_symbols.clone()));
        frame.push("value", OutCol::F64(values));
        frame.write_parquet(&partition.join("part.parquet"))?;
    }
    let mut values = Map::new();
    values.insert(
        "t_start".into(),
        Value::String(timeutil::local_date(days[0]).format("%Y-%m-%d").to_string()),
    );
    values.insert(
        "t_end".into(),
        Value::String(
            timeutil::local_date(*days.last().unwrap())
                .format("%Y-%m-%d")
                .to_string(),
        ),
    );
    values.insert(
        "mode".into(),
        Value::String(if frequency == "1d" { "T1" } else { "T0" }.into()),
    );
    values.insert("frequency".into(), Value::String(frequency.into()));
    values.insert(
        "order_slices".into(),
        Value::from(if frequency == "1d" { 1 } else { 3 }),
    );
    let config = BacktestConfig::from_map(&values)?;
    write_json(
        &directory.join("config.json"),
        &Value::Object(config.to_dict()),
    )?;
    let result = json!({
        "synthetic": true,
        "seed": options.seed,
        "symbols": options.asset_count,
        "sessions": days.len(),
        "bars": timestamps.len(),
        "frequency": frequency,
        "factors": options.factor_count,
        "note": "Not real HSI constituents or market performance",
    });
    write_json(&directory.join("demo.json"), &result)?;
    Ok(result.as_object().unwrap().clone())
}
