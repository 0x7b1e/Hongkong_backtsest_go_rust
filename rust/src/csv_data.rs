//! Read source CSV bars without changing labels, values, or source bytes.

use std::fs;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};

use crate::data;
use crate::frames::{OutCol, OutFrame};
use crate::table::{self, Col, Frame};
use crate::timeutil::{self, NS_PER_DAY, NS_PER_HOUR, NS_PER_MIN, TIMEZONE};

const COLUMNS: &[(&str, &str)] = &[
    ("代码", "symbol"),
    ("名称", "name"),
    ("日期", "date"),
    ("时间", "time"),
    ("开盘", "open"),
    ("最高", "high"),
    ("最低", "low"),
    ("收盘", "close"),
    ("成交量", "volume"),
    ("成交额", "amount"),
    ("昨收", "previous_close"),
];
const PRICE_COLUMNS: &[&str] = &["open", "high", "low", "close", "volume", "amount"];
/// Columns pandas reads as text (``dtype=str``) before renaming.
const TEXT_COLUMNS: &[&str] = &["代码", "日期", "时间", "symbol", "code", "date", "time"];
/// pandas' default ``na_values``.
const NA_VALUES: &[&str] = &[
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
    "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null",
];

/// Source encodings supported for CSV input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    Utf8Sig,
    Gb18030,
}

impl Encoding {
    pub fn name(self) -> &'static str {
        match self {
            Encoding::Utf8Sig => "utf-8-sig",
            Encoding::Gb18030 => "gb18030",
        }
    }

    fn decode(self, bytes: &[u8]) -> Option<String> {
        match self {
            Encoding::Utf8Sig => {
                let body = bytes.strip_prefix("\u{feff}".as_bytes()).unwrap_or(bytes);
                std::str::from_utf8(body).ok().map(str::to_string)
            }
            Encoding::Gb18030 => {
                let (text, _, errors) = encoding_rs::GB18030.decode(bytes);
                (!errors).then(|| text.into_owned())
            }
        }
    }
}

/// Detect UTF-8/GB18030 and an optional original HSI description row.
pub fn csv_options(source: &Path) -> Result<(Encoding, usize)> {
    let bytes = fs::read(source).with_context(|| format!("Cannot read {}", source.display()))?;
    let mut sample_end = 0;
    let mut lines = 0;
    while sample_end < bytes.len() && lines < 2 {
        if bytes[sample_end] == b'\n' {
            lines += 1;
        }
        sample_end += 1;
    }
    let sample = &bytes[..sample_end];
    if sample.iter().all(|b| b.is_ascii_whitespace()) {
        bail!("CSV has no header");
    }
    for encoding in [Encoding::Utf8Sig, Encoding::Gb18030] {
        if let Some(text) = encoding.decode(sample) {
            let first = text.lines().next().unwrap_or("");
            return Ok((encoding, first.starts_with("HSI ") as usize));
        }
    }
    bail!("Unsupported CSV encoding: {}", source.display())
}

fn canonical(name: &str) -> String {
    COLUMNS
        .iter()
        .find(|(source, _)| *source == name)
        .map(|(_, target)| target.to_string())
        .unwrap_or_else(|| name.to_string())
}

/// Read shared CSV syntax and map Chinese names to canonical fields.
///
/// Frames retain source values and leading-zero security codes; numeric
/// columns are inferred per chunk as ``pd.read_csv`` does.
pub fn csv_frames(source: &Path, chunksize: usize) -> Result<Vec<Frame>> {
    if chunksize < 1 {
        bail!("chunksize must be a positive integer");
    }
    let (encoding, skiprows) = csv_options(source)?;
    let bytes = fs::read(source)?;
    let text = encoding
        .decode(&bytes)
        .ok_or_else(|| anyhow!("Unsupported CSV encoding: {}", source.display()))?;
    let body: String = if skiprows > 0 {
        text.split_once('\n').map(|x| x.1).unwrap_or("").to_string()
    } else {
        text
    };
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(body.as_bytes());
    let raw_headers: Vec<String> = reader.headers()?.iter().map(str::to_string).collect();
    let headers: Vec<String> = raw_headers
        .iter()
        .enumerate()
        .map(|(i, h)| {
            if h.is_empty() {
                format!("Unnamed: {i}")
            } else {
                h.clone()
            }
        })
        .collect();
    let mut frames = Vec::new();
    let mut chunk: Vec<Vec<Option<String>>> = vec![Vec::new(); headers.len()];
    let mut count = 0;
    let flush = |chunk: &mut Vec<Vec<Option<String>>>, frames: &mut Vec<Frame>| -> Result<()> {
        let mut frame = Frame::default();
        for (i, header) in headers.iter().enumerate() {
            let values = std::mem::take(&mut chunk[i]);
            if header.starts_with("Unnamed:") {
                if values.iter().any(Option::is_some) {
                    bail!("Unexpected values in unnamed CSV columns");
                }
                continue;
            }
            let col = if TEXT_COLUMNS.contains(&header.as_str()) {
                Col::Str(values)
            } else {
                table::infer_text_column(values)
            };
            frame.push(&canonical(header), col);
        }
        frames.push(frame);
        Ok(())
    };
    for record in reader.records() {
        let record = record?;
        for i in 0..headers.len() {
            let field = record.get(i).unwrap_or("");
            chunk[i].push(if NA_VALUES.contains(&field) {
                None
            } else {
                Some(field.to_string())
            });
        }
        count += 1;
        if count == chunksize {
            flush(&mut chunk, &mut frames)?;
            count = 0;
        }
    }
    if count > 0 {
        flush(&mut chunk, &mut frames)?;
    }
    Ok(frames)
}

/// Combine YYYYMMDD dates and HHMM clocks into Hong Kong bar-end labels.
pub fn csv_timestamps(frame: &Frame, frequency: &str) -> Result<Vec<i64>> {
    if !["1d", "1m", "5m"].contains(&frequency) {
        bail!("frequency must be 1d, 1m, or 5m");
    }
    if let Some(col) = frame.get("timestamp") {
        return data::normalize_times(col, "timestamp", frequency);
    }
    let dates_col = frame
        .get("date")
        .ok_or_else(|| anyhow!("CSV requires a date or timestamp column"))?;
    let text = dates_col.to_strings();
    let mut dates = Vec::with_capacity(text.len());
    for value in &text {
        match value {
            None => dates.push(None),
            Some(v) => {
                let parsed = timeutil::parse_datetime_text(v)
                    .map_err(|_| anyhow!("Unable to parse date {v:?}"))?;
                dates.push(Some(parsed));
            }
        }
    }
    if frequency == "1d" {
        if dates.iter().any(Option::is_none) {
            bail!("Timestamps cannot contain null values");
        }
        let naive: Vec<i64> = dates.iter().map(|d| d.unwrap().naive_ns).collect();
        let mut instants: Vec<i64> = naive
            .iter()
            .map(|&n| timeutil::local_to_utc(n))
            .collect::<Result<_>>()?;
        if naive.iter().all(|n| n.rem_euclid(NS_PER_DAY) == 0) {
            for value in &mut instants {
                *value += 16 * NS_PER_HOUR;
            }
        }
        return Ok(instants);
    }
    let clocks = frame
        .get("time")
        .ok_or_else(|| anyhow!("Minute CSV requires a time column"))?
        .to_strings();
    let mut minutes_of_day = Vec::with_capacity(clocks.len());
    for clock in &clocks {
        let clock = clock.as_deref().map(str::trim).unwrap_or("");
        let compact =
            !clock.is_empty() && clock.len() <= 4 && clock.bytes().all(|b| b.is_ascii_digit());
        let colon = clock
            .split_once(':')
            .map(|(h, m)| {
                (1..=2).contains(&h.len())
                    && m.len() == 2
                    && h.bytes().chain(m.bytes()).all(|b| b.is_ascii_digit())
            })
            .unwrap_or(false);
        if !(compact || colon) {
            bail!("Invalid HHMM time value");
        }
        let digits = format!("{:0>4}", clock.replace(':', ""));
        let hours: i64 = digits[..2].parse()?;
        let minutes: i64 = digits[2..].parse()?;
        minutes_of_day.push((hours, minutes));
    }
    if minutes_of_day.iter().any(|(h, m)| *h > 23 || *m > 59) {
        bail!("Invalid HHMM time value");
    }
    if dates.iter().any(Option::is_none) {
        bail!("Dates cannot contain null values");
    }
    dates
        .iter()
        .zip(&minutes_of_day)
        .map(|(date, (h, m))| {
            let midnight = date.unwrap().naive_ns.div_euclid(NS_PER_DAY) * NS_PER_DAY;
            timeutil::local_to_utc(midnight + (h * 60 + m) * NS_PER_MIN)
        })
        .collect()
}

fn bound(value: Option<&str>, end: bool) -> Result<Option<i64>> {
    let Some(value) = value else { return Ok(None) };
    let parsed =
        timeutil::parse_datetime_text(value).map_err(|_| anyhow!("Date bounds cannot be null"))?;
    let mut result = parsed.to_utc(TIMEZONE)?;
    let date_only =
        (value.len() == 10 && value.as_bytes()[4] == b'-' && value.as_bytes()[7] == b'-')
            || (value.len() == 8 && value.bytes().all(|b| b.is_ascii_digit()));
    if end && date_only {
        result += NS_PER_DAY - 1;
    }
    Ok(Some(result))
}

/// Metadata attached to index CSV frames.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexAttrs {
    pub source: String,
    pub frequency: String,
    pub negative_amount_count: usize,
    pub timestamp_label: &'static str,
    pub volume_unit: &'static str,
}

fn empty_index() -> OutFrame {
    let mut frame = OutFrame::new();
    frame.push("timestamp", OutCol::Ts(Vec::new()));
    frame.push("symbol", OutCol::Str(Vec::new()));
    for column in PRICE_COLUMNS {
        frame.push(column, OutCol::F64(Vec::new()));
    }
    frame
}

/// Stream complete index bars and validate causal source ordering.
///
/// Source minute timestamps label bar ends. Missing bars and partial
/// sessions remain missing. No resampling, price adjustment, or stock volume
/// conversion is performed. Negative source amounts are retained for
/// reconciliation and counted in each frame's attributes.
pub fn iter_index_csv(
    source: &Path,
    frequency: &str,
    symbol: &str,
    t_start: Option<&str>,
    t_end: Option<&str>,
    chunksize: usize,
) -> Result<Vec<(OutFrame, IndexAttrs)>> {
    if symbol.is_empty() {
        bail!("symbol must be nonempty");
    }
    let start = bound(t_start, false)?;
    let end = bound(t_end, true)?;
    if let (Some(s), Some(e)) = (start, end) {
        if s > e {
            bail!("t_start must be on or before t_end");
        }
    }
    let mut last: Option<i64> = None;
    let mut output = Vec::new();
    for frame in csv_frames(source, chunksize)? {
        let missing: Vec<&str> = PRICE_COLUMNS
            .iter()
            .copied()
            .filter(|c| !frame.has(c))
            .collect();
        if !missing.is_empty() {
            let mut sorted = missing.clone();
            sorted.sort();
            bail!("Index CSV missing fields: {sorted:?}");
        }
        let times = csv_timestamps(&frame, frequency)?;
        let mut unique = times.clone();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() != times.len() {
            bail!("Duplicate index timestamp");
        }
        if times.windows(2).any(|w| w[1] < w[0]) {
            bail!("Index timestamps must be chronological");
        }
        if let (Some(first), Some(previous)) = (times.first(), last) {
            if *first <= previous {
                bail!("Duplicate or out-of-order index timestamp");
            }
        }
        if let Some(t) = times.last() {
            last = Some(*t);
        }
        let mut columns = Vec::new();
        for name in PRICE_COLUMNS {
            columns.push(frame.col(name)?.to_f64(name)?);
        }
        if columns.iter().flatten().any(|v| !v.is_finite()) {
            bail!("Index numeric values must be finite");
        }
        let (open, high, low, close, volume, amount) = (
            &columns[0],
            &columns[1],
            &columns[2],
            &columns[3],
            &columns[4],
            &columns[5],
        );
        if (0..times.len())
            .any(|i| open[i] <= 0.0 || high[i] <= 0.0 || low[i] <= 0.0 || close[i] <= 0.0)
        {
            bail!("Index prices must be positive");
        }
        if volume.iter().any(|v| *v < 0.0) {
            bail!("Index volume cannot be negative");
        }
        if (0..times.len()).any(|i| {
            high[i] < open[i].max(close[i]) || low[i] > open[i].min(close[i]) || low[i] > high[i]
        }) {
            bail!("Invalid index OHLC envelope");
        }
        if let Some(col) = frame.get("symbol") {
            if col
                .to_strings()
                .iter()
                .any(|s| s.as_deref() != Some(symbol))
            {
                bail!("CSV contains an unexpected index symbol");
            }
        }
        let keep: Vec<usize> = (0..times.len())
            .filter(|&i| {
                start.map(|s| times[i] >= s).unwrap_or(true)
                    && end.map(|e| times[i] <= e).unwrap_or(true)
            })
            .collect();
        if keep.is_empty() {
            continue;
        }
        let mut result = OutFrame::new();
        result.push(
            "timestamp",
            OutCol::Ts(keep.iter().map(|&i| times[i]).collect()),
        );
        result.push("symbol", OutCol::Str(vec![symbol.to_string(); keep.len()]));
        for (name, values) in PRICE_COLUMNS.iter().zip(&columns) {
            result.push(name, OutCol::F64(keep.iter().map(|&i| values[i]).collect()));
        }
        let negative = keep.iter().filter(|&&i| amount[i] < 0.0).count();
        output.push((
            result,
            IndexAttrs {
                source: source.display().to_string(),
                frequency: frequency.to_string(),
                negative_amount_count: negative,
                timestamp_label: "bar_end",
                volume_unit: "source_unchanged",
            },
        ));
    }
    Ok(output)
}

/// Read an index date slice in memory using the streaming parser.
pub fn read_index_csv(
    source: &Path,
    frequency: &str,
    symbol: &str,
    t_start: Option<&str>,
    t_end: Option<&str>,
    chunksize: usize,
) -> Result<(OutFrame, IndexAttrs)> {
    let parts = iter_index_csv(source, frequency, symbol, t_start, t_end, chunksize)?;
    let count = parts.iter().map(|(_, a)| a.negative_amount_count).sum();
    let frames: Vec<OutFrame> = parts.into_iter().map(|(f, _)| f).collect();
    let result = if frames.is_empty() {
        empty_index()
    } else {
        OutFrame::concat(&frames)?
    };
    Ok((
        result,
        IndexAttrs {
            source: source.display().to_string(),
            frequency: frequency.to_string(),
            negative_amount_count: count,
            timestamp_label: "bar_end",
            volume_unit: "source_unchanged",
        },
    ))
}
