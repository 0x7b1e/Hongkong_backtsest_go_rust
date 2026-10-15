//! Stream validated market inputs into shared panels.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::errors::Error;
use crate::matrix::Mat;
use crate::models::MarketPanel;
use crate::npy;
use crate::table::{self, Col, Dataset, Frame};
use crate::timeutil::{self, NS_PER_DAY, NS_PER_HOUR, NS_PER_MIN, TIMEZONE};

const MATRIX_F64: &[&str] = &[
    "open",
    "high",
    "low",
    "close",
    "volume",
    "base_weights",
    "lot_sizes",
    "vcm_lower",
    "vcm_upper",
    "cash_dividend",
    "share_multiplier",
];

/// Interpret naive timestamps locally and date-only daily bars at 16:00.
pub fn normalize_times(col: &Col, name: &str, frequency: &str) -> Result<Vec<i64>> {
    let mut result = match col {
        Col::Int(values) => values
            .iter()
            .map(|v| {
                let v = v.ok_or_else(|| anyhow!("Timestamps cannot contain null values"))?;
                timeutil::local_to_utc(v)
            })
            .collect::<Result<Vec<_>>>()?,
        Col::Float(values) if values.iter().all(Option::is_none) && !values.is_empty() => {
            bail!("Timestamps cannot contain null values")
        }
        other => {
            if other.null_count() > 0 {
                bail!("Timestamps cannot contain null values");
            }
            table::column_instants(other, name, TIMEZONE)?.0
        }
    };
    if frequency == "1d" && result.iter().all(|&t| timeutil::normalize(t) == t) {
        for value in &mut result {
            *value += 16 * NS_PER_HOUR;
        }
    }
    Ok(result)
}

fn hash_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path)?;
    Ok(hex::encode(Sha256::digest(&bytes)))
}

/// Identify panel metadata and changes to any shared array file.
pub fn panel_fingerprint(directory: &Path) -> Result<String> {
    let metadata = directory.join("metadata.json");
    let mut digest = Sha256::new();
    digest.update(fs::read(&metadata).with_context(|| format!("{}", metadata.display()))?);
    let mut files: Vec<PathBuf> = fs::read_dir(directory)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|e| e == "npy").unwrap_or(false))
        .collect();
    files.sort_by_key(|p| p.file_name().map(|n| n.to_os_string()));
    for file in files {
        let stat = fs::metadata(&file)?;
        let mtime = stat
            .modified()?
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let name = file.file_name().unwrap().to_string_lossy();
        digest.update(format!("{name}:{}:{mtime}", stat.len()).as_bytes());
    }
    Ok(hex::encode(digest.finalize()))
}

/// Write arrays without pickle and record their content checksums.
pub fn save_panel(panel: &MarketPanel, directory: &Path) -> Result<()> {
    fs::create_dir_all(directory)?;
    if directory.join("metadata.json").exists() {
        bail!(Error::FileExists(format!(
            "Panel already exists: {}",
            directory.display()
        )));
    }
    let (rows, cols) = panel.shape();
    let f64_arrays: [(&str, &Mat<f64>); 11] = [
        ("open", &panel.open),
        ("high", &panel.high),
        ("low", &panel.low),
        ("close", &panel.close),
        ("volume", &panel.volume),
        ("base_weights", &panel.base_weights),
        ("lot_sizes", &panel.lot_sizes),
        ("vcm_lower", &panel.vcm_lower),
        ("vcm_upper", &panel.vcm_upper),
        ("cash_dividend", &panel.cash_dividend),
        ("share_multiplier", &panel.share_multiplier),
    ];
    for (name, array) in f64_arrays {
        npy::save_mat(&directory.join(format!("{name}.npy")), array)?;
    }
    npy::save_vec(
        &directory.join("benchmark_close.npy"),
        &panel.benchmark_close,
    )?;
    npy::save_mat(&directory.join("tradable.npy"), &panel.tradable)?;
    npy::save_mat(&directory.join("vcm_eligible.npy"), &panel.vcm_eligible)?;
    npy::save_vec(&directory.join("session_id.npy"), &panel.session_id)?;
    npy::save_vec(
        &directory.join("is_session_start.npy"),
        &panel.is_session_start,
    )?;
    npy::save_vec(&directory.join("is_session_end.npy"), &panel.is_session_end)?;
    let vcm_active = panel
        .vcm_active
        .clone()
        .unwrap_or_else(|| Mat::filled(rows, cols, -1i8));
    npy::save_mat(&directory.join("vcm_active.npy"), &vcm_active)?;
    let exempt = panel
        .stamp_duty_exempt
        .clone()
        .unwrap_or_else(|| Mat::filled(rows, cols, false));
    npy::save_mat(&directory.join("stamp_duty_exempt.npy"), &exempt)?;
    let benchmark_open = panel
        .benchmark_open
        .clone()
        .unwrap_or_else(|| vec![f64::NAN; rows]);
    npy::save_vec(&directory.join("benchmark_open.npy"), &benchmark_open)?;
    npy::save_vec(&directory.join("timestamps.npy"), &panel.timestamps)?;
    npy::save_vec(
        &directory.join("open_timestamps.npy"),
        &panel.open_timestamps,
    )?;
    let mut checksums = BTreeMap::new();
    let mut files: Vec<PathBuf> = fs::read_dir(directory)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|e| e == "npy").unwrap_or(false))
        .collect();
    files.sort();
    for file in files {
        checksums.insert(
            file.file_name().unwrap().to_string_lossy().to_string(),
            Value::String(hash_file(&file)?),
        );
    }
    let metadata = json!({
        "version": 1,
        "symbols": panel.symbols,
        "frequency": panel.frequency,
        "price_basis": panel.price_basis,
        "shape": [rows, cols],
        "timezone": TIMEZONE,
        "array_sha256": checksums,
    });
    fs::write(
        directory.join("metadata.json"),
        crate::pyjson::dumps(
            &metadata,
            crate::pyjson::Options {
                sort_keys: false,
                indent: Some(2),
                ensure_ascii: true,
            },
        ) + "\n",
    )?;
    Ok(())
}

/// Open immutable market arrays.
pub fn load_panel(directory: &Path) -> Result<MarketPanel> {
    let path = directory.join("metadata.json");
    let text = fs::read_to_string(&path).map_err(|_| {
        Error::FileNotFound(format!("No such file or directory: '{}'", path.display()))
    })?;
    let metadata: Value = serde_json::from_str(&text)?;
    if metadata.get("version").and_then(Value::as_i64) != Some(1) {
        bail!("Unsupported market-panel version");
    }
    let shape: Vec<usize> = metadata["shape"]
        .as_array()
        .ok_or_else(|| anyhow!("Invalid panel metadata"))?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as usize)
        .collect();
    let (rows, cols) = (shape[0], shape[1]);
    let file = |name: &str| directory.join(format!("{name}.npy"));
    let mat_f64 = |name: &str| -> Result<Mat<f64>> {
        let m = npy::load_mat::<f64>(&file(name))?;
        if m.shape() != (rows, cols) {
            bail!("Invalid array shape: {name}");
        }
        Ok(m)
    };
    let mat_bool = |name: &str| -> Result<Mat<bool>> {
        let m = npy::load_mat::<bool>(&file(name))?;
        if m.shape() != (rows, cols) {
            bail!("Invalid array shape: {name}");
        }
        Ok(m)
    };
    let vec_check = |name: &str, len: usize| -> Result<()> {
        if len != rows {
            bail!("Invalid array shape: {name}");
        }
        Ok(())
    };
    let benchmark_close = npy::load_vec::<f64>(&file("benchmark_close"))?;
    vec_check("benchmark_close", benchmark_close.len())?;
    let session_id = npy::load_vec::<i64>(&file("session_id"))?;
    vec_check("session_id", session_id.len())?;
    let is_session_start = npy::load_vec::<bool>(&file("is_session_start"))?;
    vec_check("is_session_start", is_session_start.len())?;
    let is_session_end = npy::load_vec::<bool>(&file("is_session_end"))?;
    vec_check("is_session_end", is_session_end.len())?;
    let vcm_active = if file("vcm_active").exists() {
        let m = npy::load_mat::<i8>(&file("vcm_active"))?;
        if m.shape() != (rows, cols) {
            bail!("Invalid array shape: vcm_active");
        }
        Some(m)
    } else {
        None
    };
    let stamp_duty_exempt = if file("stamp_duty_exempt").exists() {
        Some(mat_bool("stamp_duty_exempt")?)
    } else {
        None
    };
    let benchmark_open = if file("benchmark_open").exists() {
        let v = npy::load_vec::<f64>(&file("benchmark_open"))?;
        vec_check("benchmark_open", v.len())?;
        Some(v)
    } else {
        None
    };
    let timestamps = npy::load_vec::<i64>(&file("timestamps"))?;
    let open_timestamps = npy::load_vec::<i64>(&file("open_timestamps"))?;
    Ok(MarketPanel {
        timestamps,
        open_timestamps,
        symbols: metadata["symbols"]
            .as_array()
            .ok_or_else(|| anyhow!("Invalid panel metadata"))?
            .iter()
            .map(|s| s.as_str().unwrap_or_default().to_string())
            .collect(),
        frequency: metadata["frequency"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
        open: mat_f64("open")?,
        high: mat_f64("high")?,
        low: mat_f64("low")?,
        close: mat_f64("close")?,
        volume: mat_f64("volume")?,
        base_weights: mat_f64("base_weights")?,
        benchmark_close,
        lot_sizes: mat_f64("lot_sizes")?,
        tradable: mat_bool("tradable")?,
        vcm_lower: mat_f64("vcm_lower")?,
        vcm_upper: mat_f64("vcm_upper")?,
        vcm_eligible: mat_bool("vcm_eligible")?,
        cash_dividend: mat_f64("cash_dividend")?,
        share_multiplier: mat_f64("share_multiplier")?,
        session_id,
        is_session_start,
        is_session_end,
        price_basis: metadata["price_basis"]
            .as_str()
            .unwrap_or("raw")
            .to_string(),
        vcm_active,
        stamp_duty_exempt,
        benchmark_open,
    })
}

struct Snapshot {
    effective: i64,
    known: i64,
    weights: Vec<f64>,
}

fn snapshots(path: &Path) -> Result<(Vec<Snapshot>, Vec<String>)> {
    let frame = table::read_table(path)?;
    for required in ["available_at", "effective_date", "symbol", "weight"] {
        if !frame.has(required) {
            bail!("Constituents require ['available_at', 'effective_date', 'symbol', 'weight']");
        }
    }
    let symbols: Vec<String> = frame
        .col("symbol")?
        .to_strings()
        .into_iter()
        .map(|s| s.unwrap_or_else(|| "nan".into()))
        .collect();
    let effective: Vec<i64> =
        normalize_times(frame.col("effective_date")?, "effective_date", "5m")?
            .into_iter()
            .map(timeutil::normalize)
            .collect();
    let available = normalize_times(frame.col("available_at")?, "available_at", "5m")?;
    let mut seen = BTreeSet::new();
    for (e, s) in effective.iter().zip(&symbols) {
        if !seen.insert((*e, s.clone())) {
            bail!("Duplicate constituent snapshot entries");
        }
    }
    let weights = frame
        .col("weight")?
        .to_f64("weight")
        .map_err(|e| anyhow!("Unable to parse constituent weight: {e}"))?;
    if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
        bail!("Constituent weights must be finite and nonnegative");
    }
    let universe = table::sorted_unique(symbols.iter().cloned());
    let positions: HashMap<&String, usize> =
        universe.iter().enumerate().map(|(i, s)| (s, i)).collect();
    let mut groups: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (row, e) in effective.iter().enumerate() {
        groups.entry(*e).or_default().push(row);
    }
    let mut result = Vec::new();
    for (effective_date, rows) in groups {
        let total: f64 = crate::matrix::sum(&rows.iter().map(|&r| weights[r]).collect::<Vec<_>>());
        if (total - 1.0).abs() > 1e-8 + 1e-5 * 1.0 {
            bail!("Each FULL constituent snapshot must sum to 1");
        }
        let published: BTreeSet<i64> = rows.iter().map(|&r| available[r]).collect();
        if published.len() != 1 {
            bail!("Each snapshot must have one publication time");
        }
        let mut snapshot_weights = vec![0.0; universe.len()];
        for &r in &rows {
            snapshot_weights[positions[&symbols[r]]] = weights[r];
        }
        let available_at = *published.iter().next().unwrap();
        result.push(Snapshot {
            effective: effective_date,
            known: effective_date.max(available_at),
            weights: snapshot_weights,
        });
    }
    Ok((result, universe))
}

struct Session {
    date: i64,
    open: i64,
    close: i64,
    break_start: i64,
    break_end: i64,
}

fn clock(text: &Option<String>) -> Result<i64> {
    let text = text
        .as_ref()
        .ok_or_else(|| anyhow!("Invalid trading-calendar session"))?;
    timeutil::parse_timedelta(&format!("{text}:00"))
        .map_err(|_| anyhow!("Invalid trading-calendar clock {text:?}"))
}

fn calendar(
    path: Option<&Path>,
    dates: &[i64],
    t_start: Option<&str>,
    t_end: Option<&str>,
) -> Result<Vec<Session>> {
    let default = |date: i64| Session {
        date,
        open: 9 * NS_PER_HOUR + 30 * NS_PER_MIN,
        close: 16 * NS_PER_HOUR,
        break_start: 12 * NS_PER_HOUR,
        break_end: 13 * NS_PER_HOUR,
    };
    let path = match path {
        None => return Ok(dates.iter().map(|&d| default(d)).collect()),
        Some(path) => path,
    };
    let frame = table::read_table(path)?;
    for required in ["date", "open", "close"] {
        if !frame.has(required) {
            bail!("Sessions require date, open, close");
        }
    }
    let session_dates: Vec<i64> = normalize_times(frame.col("date")?, "date", "5m")?
        .into_iter()
        .map(timeutil::normalize)
        .collect();
    let unique: BTreeSet<i64> = session_dates.iter().copied().collect();
    if unique.len() != session_dates.len() {
        bail!("Duplicate trading-calendar dates");
    }
    let text = |name: &str, fallback: &str| -> Vec<Option<String>> {
        match frame.get(name) {
            Some(col) => col.to_strings(),
            None => vec![Some(fallback.to_string()); frame.len()],
        }
    };
    let opens = text("open", "");
    let closes = text("close", "");
    let pauses = text("break_start", "12:00");
    let resumes = text("break_end", "13:00");
    if !dates.iter().all(|d| unique.contains(d)) {
        bail!("Trading calendar does not cover quoted dates");
    }
    let bound = |value: Option<&str>| -> Result<Option<i64>> {
        value
            .map(|v| Ok(timeutil::date_midnight_utc(timeutil::parse_iso_date(v)?)))
            .transpose()
    };
    let min_date = *dates.iter().min().unwrap();
    let max_date = *dates.iter().max().unwrap();
    let start = bound(t_start)?.map(|s| s.min(min_date)).unwrap_or(min_date);
    let end = bound(t_end)?.map(|e| e.max(max_date)).unwrap_or(max_date);
    let mut sessions = Vec::new();
    for row in 0..session_dates.len() {
        let date = session_dates[row];
        if date < start || date > end {
            continue;
        }
        sessions.push(Session {
            date,
            open: clock(&opens[row])?,
            close: clock(&closes[row])?,
            break_start: clock(&pauses[row])?,
            break_end: clock(&resumes[row])?,
        });
    }
    sessions.sort_by_key(|s| s.date);
    Ok(sessions)
}

/// Bar-end and bar-open instants for a trading calendar.
pub fn bar_grid(
    sessions: &[(i64, i64, i64, i64, i64)],
    frequency: &str,
) -> Result<(Vec<i64>, Vec<i64>)> {
    let mut ends = Vec::new();
    let mut opens = Vec::new();
    let duration = if frequency == "1m" {
        NS_PER_MIN
    } else {
        5 * NS_PER_MIN
    };
    for &(day, open, close, pause, resume) in sessions {
        let start = day + open;
        let finish = day + close;
        let pause = day + pause;
        let resume = day + resume;
        if start >= finish || pause >= resume {
            bail!("Invalid trading-calendar session");
        }
        if frequency == "1d" {
            ends.push(day + 16 * NS_PER_HOUR);
            opens.push(start);
            continue;
        }
        for (left, right) in [(start, pause.min(finish)), (resume, finish)] {
            if left >= right {
                continue;
            }
            let mut bar = left + duration;
            while bar <= right {
                ends.push(bar);
                opens.push(bar - duration);
                bar += duration;
            }
        }
    }
    if ends.is_empty() {
        bail!("No price bars in requested range");
    }
    Ok((ends, opens))
}

fn base_weights(snapshots: &[Snapshot], opens: &[i64], destination: &mut Mat<f64>) -> Result<()> {
    for (i, &timestamp) in opens.iter().enumerate() {
        let chosen = snapshots
            .iter()
            .filter(|s| s.known <= timestamp)
            .max_by_key(|s| s.effective)
            .ok_or_else(|| anyhow!("No published constituent snapshot before open"))?;
        destination.row_mut(i).copy_from_slice(&chosen.weights);
    }
    Ok(())
}

fn format_thousands(value: u128) -> String {
    let digits = value.to_string();
    let mut out = String::new();
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

fn bool_column(frame: &Frame, name: &str) -> Result<Option<Vec<bool>>> {
    match frame.get(name) {
        None => Ok(None),
        Some(Col::Bool(values)) if values.iter().all(Option::is_some) => {
            Ok(Some(values.iter().map(|v| v.unwrap()).collect()))
        }
        Some(_) => bail!("{name} must be non-null booleans"),
    }
}

/// Prepare market arrays with bounded input-batch memory.
///
/// ``constituents`` are full point-in-time weight snapshots, never a current
/// constituent list retrospectively applied to history. ``t_start`` retains
/// 14 calendar days of warmup.
#[allow(clippy::too_many_arguments)]
pub fn prepare_market(
    prices: &Path,
    benchmark: &Path,
    constituents: &Path,
    output: &Path,
    frequency: &str,
    sessions: Option<&Path>,
    t_start: Option<&str>,
    t_end: Option<&str>,
    max_panel_bytes: u128,
) -> Result<MarketPanel> {
    if !["1d", "1m", "5m"].contains(&frequency) {
        bail!("Unsupported frequency");
    }
    if frequency != "1d" && sessions.is_none() {
        bail!("Intraday data require a historical session calendar");
    }
    if output.exists() {
        bail!(Error::FileExists(format!(
            "Cache already exists: {}",
            output.display()
        )));
    }
    let (snapshot_list, symbols) = snapshots(constituents)?;
    let dataset = Dataset::open(prices, false)?;
    let required = [
        "close",
        "high",
        "low",
        "open",
        "symbol",
        "timestamp",
        "volume",
    ];
    for name in required {
        if !dataset.has(name) {
            bail!("Stock prices require {:?}", required);
        }
    }
    let symbol_index: HashMap<String, usize> = symbols
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i))
        .collect();
    let lower = match t_start {
        Some(value) => {
            Some(timeutil::date_midnight_utc(timeutil::parse_iso_date(value)?) - 14 * NS_PER_DAY)
        }
        None => None,
    };
    let upper = match t_end {
        Some(value) => {
            Some(timeutil::date_midnight_utc(timeutil::parse_iso_date(value)?) + NS_PER_DAY)
        }
        None => None,
    };
    let keep_row = |time: i64| -> bool {
        lower.map(|l| time >= l).unwrap_or(true) && upper.map(|u| time < u).unwrap_or(true)
    };
    // First pass: quoted trading days.
    let mut quoted_days = BTreeSet::new();
    dataset.scan(
        &["timestamp".to_string(), "symbol".to_string()],
        65_536,
        |frame| {
            let symbol_values = frame.col("symbol")?.to_strings();
            let rows: Vec<usize> = (0..frame.len())
                .filter(|&r| {
                    symbol_values[r]
                        .as_ref()
                        .map(|s| symbol_index.contains_key(s))
                        .unwrap_or(false)
                })
                .collect();
            if rows.is_empty() {
                return Ok(());
            }
            let times =
                normalize_times(&frame.col("timestamp")?.take(&rows), "timestamp", frequency)?;
            for time in times {
                if keep_row(time) {
                    quoted_days.insert(timeutil::normalize(time));
                }
            }
            Ok(())
        },
    )?;
    if quoted_days.is_empty() {
        bail!("No constituent prices in requested date range");
    }
    let dates: Vec<i64> = quoted_days.into_iter().collect();
    let sessions_list = calendar(sessions, &dates, t_start, t_end)?;
    let grid_input: Vec<(i64, i64, i64, i64, i64)> = sessions_list
        .iter()
        .map(|s| (s.date, s.open, s.close, s.break_start, s.break_end))
        .collect();
    let (timestamps, open_timestamps) = bar_grid(&grid_input, frequency)?;
    let shape = (timestamps.len(), symbols.len());
    let estimate = shape.0 as u128 * shape.1 as u128 * (12 * 8 + 2) + shape.0 as u128 * 32;
    if estimate > max_panel_bytes {
        bail!(Error::Memory(format!(
            "Panel needs approximately {} bytes; budget is {}",
            format_thousands(estimate),
            format_thousands(max_panel_bytes)
        )));
    }
    let (rows, cols) = shape;
    let mut matrices: HashMap<&str, Mat<f64>> = HashMap::new();
    for name in MATRIX_F64 {
        let default = match *name {
            "volume" | "base_weights" | "cash_dividend" => 0.0,
            "share_multiplier" => 1.0,
            _ => f64::NAN,
        };
        matrices.insert(name, Mat::filled(rows, cols, default));
    }
    let mut tradable_matrix = Mat::filled(rows, cols, false);
    let mut vcm_eligible = Mat::filled(rows, cols, false);
    let mut stamp_exempt = Mat::filled(rows, cols, false);
    let mut vcm_active = Mat::filled(rows, cols, -1i8);
    let mut seen = Mat::filled(rows, cols, false);
    let row_of: HashMap<i64, usize> = timestamps
        .iter()
        .enumerate()
        .map(|(i, &t)| (t, i))
        .collect();
    let optional = [
        "cash_dividend",
        "halted",
        "lot_size",
        "price_basis",
        "security_type",
        "share_multiplier",
        "stamp_duty_exempt",
        "tradable",
        "vcm_active",
        "vcm_eligible",
        "vcm_lower",
        "vcm_upper",
    ];
    let mut columns: Vec<String> = required.iter().map(|s| s.to_string()).collect();
    for name in optional {
        if dataset.has(name) {
            columns.push(name.to_string());
        }
    }
    let reit_start = timeutil::date_midnight_utc(timeutil::parse_iso_date("2024-12-21")?);
    let etf_start = timeutil::date_midnight_utc(timeutil::parse_iso_date("2015-02-13")?);
    let mut price_bases: BTreeSet<String> = BTreeSet::new();
    dataset.scan(&columns, 65_536, |frame| {
        let symbol_values = frame.col("symbol")?.to_strings();
        let rows_in_universe: Vec<usize> = (0..frame.len())
            .filter(|&r| {
                symbol_values[r]
                    .as_ref()
                    .map(|s| symbol_index.contains_key(s))
                    .unwrap_or(false)
            })
            .collect();
        if rows_in_universe.is_empty() {
            return Ok(());
        }
        let frame = frame.take(&rows_in_universe);
        let all_times = normalize_times(frame.col("timestamp")?, "timestamp", frequency)?;
        let kept: Vec<usize> = (0..frame.len())
            .filter(|&r| keep_row(all_times[r]))
            .collect();
        if kept.is_empty() {
            return Ok(());
        }
        let frame = frame.take(&kept);
        let times: Vec<i64> = kept.iter().map(|&r| all_times[r]).collect();
        let tradable_flag = bool_column(&frame, "tradable")?;
        let halted_flag = bool_column(&frame, "halted")?;
        let eligible_flag = bool_column(&frame, "vcm_eligible")?;
        let active_flag = bool_column(&frame, "vcm_active")?;
        let exempt_flag = bool_column(&frame, "stamp_duty_exempt")?;
        let mut row_index = Vec::with_capacity(times.len());
        for time in &times {
            row_index.push(
                *row_of
                    .get(time)
                    .ok_or_else(|| anyhow!("Prices are off-grid or outside CTS hours"))?,
            );
        }
        let symbols_here = frame.col("symbol")?.to_strings();
        let col_index: Vec<usize> = symbols_here
            .iter()
            .map(|s| symbol_index[s.as_ref().unwrap()])
            .collect();
        let mut keys = BTreeSet::new();
        for (r, c) in row_index.iter().zip(&col_index) {
            if !keys.insert((*r, *c)) || seen.get(*r, *c) {
                bail!("Duplicate timestamp/symbol price rows");
            }
        }
        for (r, c) in row_index.iter().zip(&col_index) {
            seen.set(*r, *c, true);
        }
        let open = frame.col("open")?.to_f64("open")?;
        let high = frame.col("high")?.to_f64("high")?;
        let low = frame.col("low")?.to_f64("low")?;
        let close = frame.col("close")?.to_f64("close")?;
        let volume = frame.col("volume")?.to_f64("volume")?;
        for i in 0..times.len() {
            if ![open[i], high[i], low[i], close[i], volume[i]]
                .iter()
                .all(|v| v.is_finite())
            {
                bail!("Existing price rows must be finite");
            }
        }
        for i in 0..times.len() {
            if open[i] <= 0.0 || high[i] <= 0.0 || low[i] <= 0.0 || close[i] <= 0.0 {
                bail!("Prices must be positive");
            }
        }
        if volume.iter().any(|v| *v < 0.0) {
            bail!("Volume cannot be negative");
        }
        for i in 0..times.len() {
            if high[i] < open[i].max(close[i]) || low[i] > open[i].min(close[i]) || low[i] > high[i]
            {
                bail!("Invalid OHLC price envelope");
            }
        }
        for (name, values) in [
            ("open", &open),
            ("high", &high),
            ("low", &low),
            ("close", &close),
            ("volume", &volume),
        ] {
            let matrix = matrices.get_mut(name).unwrap();
            for i in 0..times.len() {
                matrix.set(row_index[i], col_index[i], values[i]);
            }
        }
        let mut lot_values = None;
        for (source, target) in [
            ("lot_size", "lot_sizes"),
            ("vcm_lower", "vcm_lower"),
            ("vcm_upper", "vcm_upper"),
            ("cash_dividend", "cash_dividend"),
            ("share_multiplier", "share_multiplier"),
        ] {
            if let Some(col) = frame.get(source) {
                let values = col.to_f64(source)?;
                let matrix = matrices.get_mut(target).unwrap();
                for i in 0..times.len() {
                    matrix.set(row_index[i], col_index[i], values[i]);
                }
                if source == "lot_size" {
                    lot_values = Some(values);
                }
            }
        }
        if let Some(lots) = lot_values {
            if lots
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.0 || v % 1.0 != 0.0)
            {
                bail!("lot_size must be a positive integer");
            }
        }
        for i in 0..times.len() {
            let mut tradable = volume[i] > 0.0;
            if let Some(flags) = &tradable_flag {
                tradable &= flags[i];
            }
            if let Some(flags) = &halted_flag {
                tradable &= !flags[i];
            }
            tradable_matrix.set(row_index[i], col_index[i], tradable);
            if let Some(flags) = &eligible_flag {
                vcm_eligible.set(row_index[i], col_index[i], flags[i]);
            }
            if let Some(flags) = &active_flag {
                vcm_active.set(row_index[i], col_index[i], flags[i] as i8);
            }
        }
        if let Some(types) = frame.get("security_type") {
            let types = types.to_strings();
            if types
                .iter()
                .any(|t| !matches!(t.as_deref(), Some("stock") | Some("reit") | Some("etf")))
            {
                bail!("security_type must be stock, reit, etf");
            }
            for i in 0..times.len() {
                let exemption = (types[i].as_deref() == Some("reit") && times[i] >= reit_start)
                    || (types[i].as_deref() == Some("etf") && times[i] >= etf_start);
                stamp_exempt.set(row_index[i], col_index[i], exemption);
            }
        }
        if let Some(flags) = &exempt_flag {
            for i in 0..times.len() {
                stamp_exempt.set(row_index[i], col_index[i], flags[i]);
            }
        }
        match frame.get("price_basis") {
            Some(col) => {
                for value in col.to_strings() {
                    price_bases.insert(value.unwrap_or_else(|| "None".into()));
                }
            }
            None => {
                price_bases.insert("raw".into());
            }
        }
        Ok(())
    })?;
    if price_bases.len() != 1 || !price_bases.iter().all(|b| b == "raw" || b == "adjusted") {
        bail!("Specify one price_basis: raw or adjusted");
    }
    let mut weights = matrices.remove("base_weights").unwrap();
    base_weights(&snapshot_list, &open_timestamps, &mut weights)?;
    for r in 0..rows {
        if !(0..cols).any(|c| seen.get(r, c) && weights.get(r, c) > 0.0) {
            bail!("Missing all constituent quotes in a session/bar");
        }
    }
    let mut close = matrices.remove("close").unwrap();
    let mut open = matrices.remove("open").unwrap();
    let mut lots = matrices.remove("lot_sizes").unwrap();
    for c in 0..cols {
        let mut last = f64::NAN;
        let mut last_lot = f64::NAN;
        for r in 0..rows {
            let value = close.get(r, c);
            if value.is_nan() {
                close.set(r, c, last);
            } else {
                last = value;
            }
            if !open.get(r, c).is_finite() {
                open.set(r, c, close.get(r, c));
            }
            let lot = lots.get(r, c);
            if lot.is_nan() {
                lots.set(r, c, last_lot);
            } else {
                last_lot = lot;
            }
        }
    }
    for i in 0..rows * cols {
        if weights.data[i] > 0.0 && !close.data[i].is_finite() {
            bail!("Constituent has no past quote for valuation");
        }
    }
    let multipliers = matrices.remove("share_multiplier").unwrap();
    if multipliers.data.iter().any(|v| *v <= 0.0 || !v.is_finite()) {
        bail!("Share multipliers must be finite and positive");
    }
    let dividends = matrices.remove("cash_dividend").unwrap();
    if dividends.data.iter().any(|v| !v.is_finite() || *v < 0.0) {
        bail!("Cash dividends must be finite and nonnegative");
    }
    let benchmark_frame = table::read_table(benchmark)?;
    if !benchmark_frame.has("timestamp") || !benchmark_frame.has("close") {
        bail!("Benchmark requires timestamp and close");
    }
    let benchmark_times =
        normalize_times(benchmark_frame.col("timestamp")?, "timestamp", frequency)?;
    let unique_times: BTreeSet<i64> = benchmark_times.iter().copied().collect();
    if unique_times.len() != benchmark_times.len() {
        bail!("Duplicate benchmark timestamps");
    }
    let lookup: HashMap<i64, usize> = benchmark_times
        .iter()
        .enumerate()
        .map(|(i, &t)| (t, i))
        .collect();
    let reindex = |values: &[f64]| -> Vec<f64> {
        timestamps
            .iter()
            .map(|t| lookup.get(t).map(|&i| values[i]).unwrap_or(f64::NAN))
            .collect()
    };
    let benchmark_close = reindex(&benchmark_frame.col("close")?.to_f64("close")?);
    let benchmark_open = match benchmark_frame.get("open") {
        Some(col) => Some(reindex(&col.to_f64("open")?)),
        None => None,
    };
    if let Some(values) = &benchmark_open {
        if values.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            bail!("Supplied benchmark open must cover all bars");
        }
    }
    if benchmark_close.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        bail!("Benchmark must cover every panel bar exactly");
    }
    let day_of: Vec<i64> = timestamps.iter().map(|&t| timeutil::normalize(t)).collect();
    let mut session_id = Vec::with_capacity(rows);
    let mut ids: HashMap<i64, i64> = HashMap::new();
    for day in &day_of {
        let next = ids.len() as i64;
        session_id.push(*ids.entry(*day).or_insert(next));
    }
    let is_session_start: Vec<bool> = (0..rows)
        .map(|i| i == 0 || day_of[i] != day_of[i - 1])
        .collect();
    let is_session_end: Vec<bool> = (0..rows)
        .map(|i| i == rows - 1 || day_of[i] != day_of[i + 1])
        .collect();
    let panel = MarketPanel {
        timestamps,
        open_timestamps,
        symbols,
        frequency: frequency.to_string(),
        open,
        high: matrices.remove("high").unwrap(),
        low: matrices.remove("low").unwrap(),
        close,
        volume: matrices.remove("volume").unwrap(),
        base_weights: weights,
        benchmark_close,
        lot_sizes: lots,
        tradable: tradable_matrix,
        vcm_lower: matrices.remove("vcm_lower").unwrap(),
        vcm_upper: matrices.remove("vcm_upper").unwrap(),
        vcm_eligible,
        cash_dividend: dividends,
        share_multiplier: multipliers,
        session_id,
        is_session_start,
        is_session_end,
        price_basis: price_bases.into_iter().next().unwrap(),
        vcm_active: Some(vcm_active),
        stamp_duty_exempt: Some(stamp_exempt),
        benchmark_open,
    };
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let name = output
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "panel".into());
    let staging = output.with_file_name(format!(
        ".{name}.{}.tmp",
        &hex::encode(Sha256::digest(format!(
            "{:?}{:?}",
            std::time::SystemTime::now(),
            std::process::id()
        )))[..32]
    ));
    fs::create_dir(&staging)?;
    let saved = save_panel(&panel, &staging)
        .and_then(|_| fs::rename(&staging, output).map_err(anyhow::Error::from));
    if let Err(error) = saved {
        let _ = fs::remove_dir_all(&staging);
        return Err(error);
    }
    drop(panel);
    load_panel(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_daily_and_intraday() {
        let day = timeutil::date_midnight_utc(timeutil::parse_iso_date("2024-01-02").unwrap());
        let session = [(
            day,
            9 * NS_PER_HOUR + 30 * NS_PER_MIN,
            16 * NS_PER_HOUR,
            12 * NS_PER_HOUR,
            13 * NS_PER_HOUR,
        )];
        let (ends, opens) = bar_grid(&session, "1d").unwrap();
        assert_eq!(timeutil::strftime_hk(ends[0], "%H:%M"), "16:00");
        assert_eq!(timeutil::strftime_hk(opens[0], "%H:%M"), "09:30");
        let (ends, _) = bar_grid(&session, "5m").unwrap();
        assert_eq!(ends.len(), 30 + 36);
        assert_eq!(timeutil::strftime_hk(ends[0], "%H:%M"), "09:35");
        assert_eq!(format_thousands(1234567), "1,234,567");
    }
}
