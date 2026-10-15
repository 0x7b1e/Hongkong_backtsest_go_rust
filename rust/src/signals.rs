//! Stream one long-form Parquet factor into an as-of signal matrix.

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Result};
use arrow::datatypes::DataType;

use crate::matrix::Mat;
use crate::models::{MarketPanel, SignalMatrix};
use crate::table::{Col, Dataset};
use crate::timeutil::{self, NS_PER_HOUR, TIMEZONE};

const BATCH_SIZE: usize = 65_536;

/// Map vendor columns to the long-form signal contract.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SignalColumns {
    pub timestamp: String,
    pub symbol: String,
    pub factor: String,
    pub value: String,
    pub available_at: String,
}

impl Default for SignalColumns {
    fn default() -> Self {
        SignalColumns {
            timestamp: "timestamp".into(),
            symbol: "symbol".into(),
            factor: "factor".into(),
            value: "value".into(),
            available_at: "available_at".into(),
        }
    }
}

impl SignalColumns {
    /// Reject missing or multiply mapped columns.
    pub fn validate(&self) -> Result<()> {
        let names = [
            &self.timestamp,
            &self.symbol,
            &self.factor,
            &self.value,
            &self.available_at,
        ];
        let unique: BTreeSet<&String> = names.iter().copied().collect();
        if names.iter().any(|n| n.is_empty()) || unique.len() != names.len() {
            bail!("Signal column names must be unique and nonempty");
        }
        Ok(())
    }
}

/// Convert mixed vendor timestamps to UTC nanoseconds without guessing.
fn normalize_signal_times(col: &Col, column: &str) -> Result<Vec<i64>> {
    if col.null_count() > 0 {
        bail!("Signal column {column:?} contains null timestamps");
    }
    match col {
        Col::Int(_) | Col::Float(_) | Col::Bool(_) => bail!(
            "Numeric signal timestamps need an explicit epoch unit; \
             convert them to typed datetimes with the signal adapter"
        ),
        Col::Ts { values, tz } => values
            .iter()
            .map(|v| match tz {
                Some(_) => Ok(v.unwrap()),
                None => timeutil::local_to_utc(v.unwrap()),
            })
            .collect(),
        Col::Date(values) => values
            .iter()
            .map(|v| timeutil::local_to_utc(v.unwrap() * timeutil::NS_PER_DAY + 16 * NS_PER_HOUR))
            .collect(),
        Col::Str(values) => {
            let parsed: Vec<timeutil::ParsedTime> = values
                .iter()
                .map(|v| {
                    timeutil::parse_datetime_text(v.as_ref().unwrap())
                        .map_err(|_| anyhow!("Invalid signal timestamp in {column:?}"))
                })
                .collect::<Result<_>>()?;
            let aware = parsed.iter().filter(|p| p.offset_ns.is_some()).count();
            if aware != 0 && aware != parsed.len() {
                bail!("Invalid signal timestamp in {column:?}");
            }
            parsed
                .iter()
                .zip(values)
                .map(|(p, text)| {
                    let date_only =
                        p.offset_ns.is_none() && is_iso_date_text(text.as_ref().unwrap());
                    let adjusted = timeutil::ParsedTime {
                        naive_ns: p.naive_ns + if date_only { 16 * NS_PER_HOUR } else { 0 },
                        ..*p
                    };
                    adjusted.to_utc(TIMEZONE)
                })
                .collect()
        }
        Col::Other { .. } => bail!("Invalid signal timestamp in {column:?}"),
    }
}

/// ``^\d{4}-\d{2}-\d{2}$``: date-only text labels a 16:00 HKT observation.
fn is_iso_date_text(text: &str) -> bool {
    let b = text.as_bytes();
    b.len() == 10
        && b[4] == b'-'
        && b[7] == b'-'
        && b.iter()
            .enumerate()
            .all(|(i, c)| i == 4 || i == 7 || c.is_ascii_digit())
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Event {
    timestamp: i64,
    available_at: i64,
    value: f64,
}

const EVENT_BYTES: usize = 24;

fn write_events(path: &Path, events: &[Event]) -> Result<()> {
    let file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    let mut writer = BufWriter::new(file);
    for event in events {
        writer.write_all(&event.timestamp.to_le_bytes())?;
        writer.write_all(&event.available_at.to_le_bytes())?;
        writer.write_all(&event.value.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

fn read_events(path: &Path) -> Result<Vec<Event>> {
    let mut bytes = Vec::new();
    fs::File::open(path)?.read_to_end(&mut bytes)?;
    Ok(bytes
        .chunks_exact(EVENT_BYTES)
        .map(|chunk| Event {
            timestamp: i64::from_le_bytes(chunk[0..8].try_into().unwrap()),
            available_at: i64::from_le_bytes(chunk[8..16].try_into().unwrap()),
            value: f64::from_le_bytes(chunk[16..24].try_into().unwrap()),
        })
        .collect())
}

/// ``np.searchsorted(sorted, value, side)``.
fn searchsorted(sorted: &[i64], value: i64, right: bool) -> usize {
    if right {
        sorted.partition_point(|&x| x <= value)
    } else {
        sorted.partition_point(|&x| x < value)
    }
}

/// Read and release only one security's complete publication history.
fn align_symbol(
    events_path: &Path,
    bar_times: &[i64],
    values: &mut Mat<f64>,
    updated: &mut [bool],
    column: usize,
    factor: &str,
) -> Result<()> {
    let mut events = read_events(events_path)?;
    let mut observed: Vec<i64> = events.iter().map(|e| e.timestamp).collect();
    observed.sort_unstable();
    if observed.windows(2).any(|w| w[0] == w[1]) {
        bail!("Duplicate symbol/timestamp for factor {factor:?}");
    }
    drop(observed);
    // np.lexsort((timestamp, available_at)): primary available_at.
    events.sort_by(|a, b| {
        a.available_at
            .cmp(&b.available_at)
            .then(a.timestamp.cmp(&b.timestamp))
    });
    let n = bar_times.len();
    let arrivals: Vec<usize> = events
        .iter()
        .map(|e| searchsorted(bar_times, e.available_at, false))
        .collect();
    for &arrival in &arrivals {
        if arrival < n {
            updated[arrival] = true;
        }
    }
    // Late features cannot overwrite an already available newer observation.
    let mut running = i64::MIN;
    let mut kept: Vec<(Event, usize)> = Vec::new();
    for (event, arrival) in events.iter().zip(&arrivals) {
        running = running.max(event.timestamp);
        if event.timestamp == running && *arrival < n {
            kept.push((*event, *arrival));
        }
    }
    if kept.is_empty() {
        return Ok(());
    }
    // Several publications can become known between the same two bar ends.
    let mut last: Vec<(Event, usize)> = Vec::with_capacity(kept.len());
    for i in 0..kept.len() {
        if i + 1 == kept.len() || kept[i + 1].1 != kept[i].1 {
            last.push(kept[i]);
        }
    }
    let arrival_bars: Vec<i64> = last.iter().map(|(_, a)| *a as i64).collect();
    for bar in 0..n {
        let selected = searchsorted(&arrival_bars, bar as i64, true);
        if selected > 0 {
            values.set(bar, column, last[selected - 1].0.value);
        }
    }
    Ok(())
}

/// Read single-file or hive-partitioned Parquet without a factor cube.
///
/// One factor is scanned once into temporary per-security binary files. Only
/// one security's history is sorted at a time, alongside the resulting
/// matrix. Historical observations before the panel are retained for warm
/// starts. Late publications cannot overwrite a newer feature observation.
pub struct SignalStore {
    pub path: PathBuf,
    pub columns: SignalColumns,
    dataset: Dataset,
}

fn is_text(data_type: &DataType) -> bool {
    match data_type {
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => true,
        DataType::Dictionary(_, value) => is_text(value),
        _ => false,
    }
}

impl SignalStore {
    /// Inspect the input schema without loading its scores.
    pub fn new(path: &Path, columns: SignalColumns) -> Result<Self> {
        columns.validate()?;
        let dataset = Dataset::open(path, true)?;
        let names = dataset.names();
        let required = [
            &columns.timestamp,
            &columns.symbol,
            &columns.factor,
            &columns.value,
        ];
        let missing: BTreeSet<&String> = required
            .iter()
            .copied()
            .filter(|n| !names.contains(n))
            .collect();
        if !missing.is_empty() {
            bail!(
                "Missing signal columns: {:?}",
                missing.into_iter().collect::<Vec<_>>()
            );
        }
        for column in [&columns.symbol, &columns.factor] {
            if !is_text(&dataset.field_type(column).unwrap()) {
                bail!("Signal identifier {column:?} must be text");
            }
        }
        Ok(SignalStore {
            path: path.to_path_buf(),
            columns,
            dataset,
        })
    }

    /// Discover partitions first, falling back to a factor-column scan.
    pub fn factors(&self) -> Result<Vec<String>> {
        let mut names: BTreeSet<String> = BTreeSet::new();
        let mut all_partitioned = true;
        for fragment in &self.dataset.fragments {
            match fragment
                .partitions
                .iter()
                .find(|(k, _)| *k == self.columns.factor)
            {
                Some((_, value)) => {
                    if value.is_empty() {
                        bail!("Factor identifiers cannot be empty");
                    }
                    names.insert(value.clone());
                }
                None => {
                    all_partitioned = false;
                    break;
                }
            }
        }
        if all_partitioned {
            return Ok(names.into_iter().collect());
        }
        self.dataset.scan(
            std::slice::from_ref(&self.columns.factor),
            BATCH_SIZE,
            |frame| {
                for value in frame.columns[0].to_strings() {
                    match value {
                        None => bail!("Factor identifiers cannot be null"),
                        Some(v) if v.is_empty() => bail!("Factor identifiers cannot be empty"),
                        Some(v) => {
                            names.insert(v);
                        }
                    }
                }
                Ok(())
            },
        )?;
        Ok(names.into_iter().collect())
    }

    /// Row filter pushed down on the publication (or observation) column.
    fn temporal_bound(&self, last_bar: i64) -> Option<(String, Box<dyn Fn(&Col, usize) -> bool>)> {
        let column = if self.dataset.has(&self.columns.available_at) {
            self.columns.available_at.clone()
        } else {
            self.columns.timestamp.clone()
        };
        match self.dataset.field_type(&column)? {
            DataType::Timestamp(_, tz) => {
                let bound = if tz.is_some() {
                    last_bar
                } else {
                    timeutil::utc_to_local(last_bar)
                };
                Some((
                    column,
                    Box::new(move |col: &Col, row: usize| match col {
                        Col::Ts { values, .. } => values[row].map(|v| v <= bound).unwrap_or(false),
                        _ => true,
                    }),
                ))
            }
            DataType::Date32 | DataType::Date64 => {
                let bound = timeutil::date_to_days(timeutil::local_date(last_bar));
                Some((
                    column,
                    Box::new(move |col: &Col, row: usize| match col {
                        Col::Date(values) => values[row].map(|v| v <= bound).unwrap_or(false),
                        _ => true,
                    }),
                ))
            }
            _ => None,
        }
    }

    /// Align publication events to bar ends without future information.
    pub fn read_factor(
        &self,
        factor: &str,
        panel: &MarketPanel,
        scratch_dir: Option<&Path>,
    ) -> Result<SignalMatrix> {
        if factor.is_empty() {
            bail!("factor must be a nonempty string");
        }
        if panel.timestamps.is_empty() {
            bail!("Cannot align signals to an empty market panel");
        }
        let names = &self.columns;
        let has_available = self.dataset.has(&names.available_at);
        let mut requested = vec![
            names.timestamp.clone(),
            names.symbol.clone(),
            names.value.clone(),
            names.factor.clone(),
        ];
        if has_available {
            requested.push(names.available_at.clone());
        }
        if let Some(dir) = scratch_dir {
            fs::create_dir_all(dir)?;
        }
        let temporary = match scratch_dir {
            Some(dir) => tempfile::Builder::new()
                .prefix(".signal-events-")
                .tempdir_in(dir)?,
            None => tempfile::Builder::new()
                .prefix(".signal-events-")
                .tempdir()?,
        };
        let directory = temporary.path().to_path_buf();
        let bar_times = &panel.timestamps;
        let last_bar = *bar_times.last().unwrap();
        let (rows, cols) = panel.shape();
        let mut updated = vec![false; rows];
        let mut values = Mat::filled(rows, cols, f64::NAN);
        let positions: HashMap<&String, usize> = panel
            .symbols
            .iter()
            .enumerate()
            .map(|(i, s)| (s, i))
            .collect();
        let bound = self.temporal_bound(last_bar);
        let mut written: BTreeSet<usize> = BTreeSet::new();
        self.dataset.scan(&requested, BATCH_SIZE, |frame| {
            let factors = frame.col(&names.factor)?.to_strings();
            let mut keep: Vec<usize> = (0..frame.len())
                .filter(|&r| factors[r].as_deref() == Some(factor))
                .collect();
            if let Some((column, test)) = &bound {
                let col = frame.col(column)?;
                keep.retain(|&r| test(col, r));
            }
            if keep.is_empty() {
                return Ok(());
            }
            let frame = frame.take(&keep);
            let observed = normalize_signal_times(frame.col(&names.timestamp)?, &names.timestamp)?;
            let available = if has_available {
                normalize_signal_times(frame.col(&names.available_at)?, &names.available_at)?
            } else {
                observed.clone()
            };
            if available.iter().zip(&observed).any(|(a, o)| a < o) {
                bail!("available_at cannot precede timestamp");
            }
            let symbols = frame.col(&names.symbol)?.to_strings();
            if symbols.iter().any(Option::is_none) {
                bail!("Signal symbols cannot be null");
            }
            let mut unknown: BTreeSet<String> = BTreeSet::new();
            let mut codes = Vec::with_capacity(symbols.len());
            for symbol in &symbols {
                let symbol = symbol.as_ref().unwrap();
                match positions.get(symbol) {
                    Some(&code) => codes.push(code),
                    None => {
                        unknown.insert(symbol.clone());
                        codes.push(usize::MAX);
                    }
                }
            }
            if !unknown.is_empty() {
                bail!(
                    "Unknown signal symbols: {:?}",
                    unknown.into_iter().collect::<Vec<_>>()
                );
            }
            let scores = frame
                .col(&names.value)?
                .to_f64(&names.value)
                .map_err(|_| anyhow!("Signal values must be numeric"))?;
            if scores.iter().any(|v| v.is_infinite()) {
                bail!("Signal values cannot be infinite");
            }
            let mut by_symbol: HashMap<usize, Vec<Event>> = HashMap::new();
            for i in 0..scores.len() {
                if available[i] <= last_bar {
                    by_symbol.entry(codes[i]).or_default().push(Event {
                        timestamp: observed[i],
                        available_at: available[i],
                        value: scores[i],
                    });
                }
            }
            let mut ordered: Vec<_> = by_symbol.into_iter().collect();
            ordered.sort_by_key(|(code, _)| *code);
            for (code, events) in ordered {
                write_events(&directory.join(format!("{code}.bin")), &events)?;
                written.insert(code);
            }
            Ok(())
        })?;
        if written.is_empty() {
            bail!("Factor {factor:?} has no available observations");
        }
        for column in written {
            align_symbol(
                &directory.join(format!("{column}.bin")),
                bar_times,
                &mut values,
                &mut updated,
                column,
                factor,
            )?;
        }
        drop(temporary);
        Ok(SignalMatrix {
            factor: factor.to_string(),
            values,
            updated,
        })
    }
}
