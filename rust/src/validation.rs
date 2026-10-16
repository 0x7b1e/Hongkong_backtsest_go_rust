//! Local data audits, CSV adapters, and provider-snapshot comparisons.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use serde_json::{json, Map, Value};

use crate::csv_data;
use crate::data;
use crate::errors::Error;
use crate::frames::{python_float_repr, OutCol, OutFrame};
use crate::pyjson::{self, Options};
use crate::table::{Dataset, Frame};
use crate::timeutil;

/// An authenticated external provider supplied by the caller.
pub trait PriceProvider {
    /// Return canonical timestamp/symbol/OHLC/volume observations.
    fn fetch_prices(
        &self,
        symbol: &str,
        t_start: &str,
        t_end: &str,
        frequency: &str,
    ) -> Result<OutFrame>;
}

/// Adapt existing CSVs to Parquet without editing the source.
///
/// ``price_basis`` is raw or adjusted and is never inferred; ``kind`` is
/// stock or benchmark (benchmarks retain close and any available open).
pub fn import_csv(
    source: &Path,
    destination: &Path,
    frequency: &str,
    symbol: &str,
    price_basis: &str,
    kind: &str,
) -> Result<()> {
    if destination.exists() {
        bail!(Error::FileExists(destination.display().to_string()));
    }
    if !["raw", "adjusted"].contains(&price_basis) || !["stock", "benchmark"].contains(&kind) {
        bail!("Invalid CSV kind or price basis");
    }
    if let Some(parent) = destination.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let result = (|| -> Result<()> {
        let frames = csv_data::csv_frames(source, 65_536)?;
        let mut parts: Vec<OutFrame> = Vec::new();
        for frame in frames {
            let times = csv_data::csv_timestamps(&frame, frequency)?;
            let symbols: Vec<String> = match frame.get("symbol") {
                Some(col) => col
                    .to_strings()
                    .into_iter()
                    .map(|s| s.unwrap_or_else(|| "nan".into()))
                    .collect(),
                None => vec![symbol.to_string(); frame.len()],
            };
            if symbols.iter().any(|s| s != symbol) {
                bail!("CSV contains unexpected security identifiers");
            }
            let mut columns: Vec<&str> = vec!["timestamp", "symbol", "close"];
            if kind == "benchmark" {
                columns = vec!["timestamp", "symbol"];
                for name in ["open", "high", "low", "close", "volume", "amount"] {
                    if frame.has(name) {
                        columns.push(name);
                    }
                }
            } else {
                columns.extend(["open", "high", "low", "volume", "price_basis"]);
            }
            let mut out = OutFrame::new();
            for name in columns {
                match name {
                    "timestamp" => out.push(name, OutCol::Ts(times.clone())),
                    "symbol" => out.push(name, OutCol::Str(symbols.clone())),
                    "price_basis" => out.push(
                        name,
                        OutCol::Str(vec![price_basis.to_string(); frame.len()]),
                    ),
                    other => {
                        let col = frame
                            .get(other)
                            .ok_or_else(|| anyhow::anyhow!("\"['{other}'] not in index\""))?;
                        out.push(other, OutCol::F64(col.to_f64(other)?));
                    }
                }
            }
            parts.push(out);
        }
        if parts.is_empty() {
            bail!("CSV has no records");
        }
        OutFrame::concat(&parts)?.write_parquet(destination)
    })();
    if result.is_err() {
        let _ = fs::remove_file(destination);
    }
    result
}

fn csv_files(directory: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    if !directory.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.is_dir() {
            csv_files(&path, out)?;
        } else if path.extension().map(|e| e == "csv").unwrap_or(false) {
            out.push(path);
        }
    }
    Ok(())
}

/// Describe available files and research-blocking data gaps.
pub fn audit_local_data(directory: &Path) -> Result<Value> {
    let mut files = Vec::new();
    csv_files(directory, &mut files)?;
    files.sort();
    let mut records = Vec::new();
    let mut stock_symbols: BTreeSet<String> = BTreeSet::new();
    let mut has_stock_intraday = false;
    for filename in files {
        if filename
            .file_name()
            .map(|n| n.to_string_lossy().starts_with("._"))
            .unwrap_or(false)
        {
            continue;
        }
        let (encoding, skiprows) = csv_data::csv_options(&filename)?;
        let mut item = Map::new();
        item.insert("path".into(), Value::String(filename.display().to_string()));
        item.insert("encoding".into(), Value::String(encoding.name().into()));
        item.insert("header_row".into(), Value::from(skiprows + 1));
        item.insert("rows".into(), Value::from(0));
        let mut dates: Vec<i64> = Vec::new();
        let mut seen: HashSet<(i64, String)> = HashSet::new();
        let mut duplicates = 0usize;
        let mut missing_lots: Option<usize> = None;
        let mut rows = 0usize;
        for frame in csv_data::csv_frames(&filename, 65_536)? {
            rows += frame.len();
            item.insert("rows".into(), Value::from(rows));
            item.insert(
                "columns".into(),
                Value::Array(
                    frame
                        .names
                        .iter()
                        .map(|n| Value::String(n.clone()))
                        .collect(),
                ),
            );
            if frame.has("date") {
                let frequency = if frame.has("time") { "1m" } else { "1d" };
                let times = csv_data::csv_timestamps(&frame, frequency)?;
                if let (Some(min), Some(max)) = (times.iter().min(), times.iter().max()) {
                    dates.push(*min);
                    dates.push(*max);
                }
                let symbols: Vec<String> = match frame.get("symbol") {
                    Some(col) => col
                        .to_strings()
                        .into_iter()
                        .map(|s| s.unwrap_or_else(|| "nan".into()))
                        .collect(),
                    None => vec!["HSI".into(); frame.len()],
                };
                for key in times.iter().copied().zip(symbols) {
                    if !seen.insert(key) {
                        duplicates += 1;
                    }
                }
                if let Some(col) = frame.get("symbol") {
                    stock_symbols.extend(col.to_strings().into_iter().flatten());
                    has_stock_intraday |= frame.has("time");
                }
            }
            if let Some(col) = frame.get("lot_size") {
                *missing_lots.get_or_insert(0) += col.null_count();
            }
        }
        item.insert("duplicate_keys".into(), Value::from(duplicates));
        if !dates.is_empty() {
            let min = *dates.iter().min().unwrap();
            let max = *dates.iter().max().unwrap();
            item.insert(
                "date_start".into(),
                Value::String(timeutil::local_date(min).format("%Y-%m-%d").to_string()),
            );
            item.insert(
                "date_end".into(),
                Value::String(timeutil::local_date(max).format("%Y-%m-%d").to_string()),
            );
        }
        if let Some(missing) = missing_lots {
            item.insert("missing_lot_sizes".into(), Value::from(missing));
        }
        records.push(Value::Object(item));
    }
    Ok(json!({
        "files": records,
        "stock_symbols_with_prices": stock_symbols.into_iter().collect::<Vec<_>>(),
        "stock_intraday_available": has_stock_intraday,
        "ready_for_full_hsi_backtest": false,
        "requirements": [
            "Historical full HSI constituent snapshots with weights and publication/effective times",
            "Raw stock OHLCV covering every historical constituent",
            "Stock minute bars and official session calendar for T0",
            "Historical board-lot sizes; current reference rows are missing lot_size",
            "Corporate actions and VCM/halt metadata for rule coverage",
            "External factor Parquet data or mapped equivalent fields",
        ],
        "api_comparison": {
            "status": "not_run",
            "reason": "No provider or authenticated reference was supplied",
        },
    }))
}

fn dataset_symbols(dataset: &Dataset) -> Result<Vec<String>> {
    if !dataset.has("symbol") {
        return Ok(vec!["HSI".into()]);
    }
    let mut symbols = BTreeSet::new();
    dataset.scan(&["symbol".to_string()], 65_536, |frame| {
        symbols.extend(
            frame.columns[0]
                .to_strings()
                .into_iter()
                .map(|s| s.unwrap_or_else(|| "None".into())),
        );
        Ok(())
    })?;
    Ok(symbols.into_iter().collect())
}

/// One security's quotes keyed by timestamp.
struct SymbolQuotes {
    fields: BTreeMap<String, BTreeMap<i64, f64>>,
    index: BTreeSet<i64>,
}

const QUOTE_FIELDS: &[&str] = &["amount", "close", "high", "low", "open", "volume"];

fn symbol_frame(dataset: &Dataset, symbol: &str, frequency: &str) -> Result<SymbolQuotes> {
    let names = dataset.names();
    let mut collected: Vec<Frame> = Vec::new();
    dataset.scan(&names, 65_536, |frame| {
        if dataset.has("symbol") {
            let symbols = frame.col("symbol")?.to_strings();
            let rows: Vec<usize> = (0..frame.len())
                .filter(|&r| symbols[r].as_deref() == Some(symbol))
                .collect();
            if !rows.is_empty() {
                collected.push(frame.take(&rows));
            }
        } else {
            collected.push(frame);
        }
        Ok(())
    })?;
    if !names.iter().any(|n| n == "timestamp") {
        bail!("Comparison inputs require canonical timestamps");
    }
    let mut quotes = SymbolQuotes {
        fields: BTreeMap::new(),
        index: BTreeSet::new(),
    };
    let present: Vec<&str> = QUOTE_FIELDS
        .iter()
        .copied()
        .filter(|f| names.iter().any(|n| n == f))
        .collect();
    for field in &present {
        quotes.fields.insert(field.to_string(), BTreeMap::new());
    }
    for frame in collected {
        let times = data::normalize_times(frame.col("timestamp")?, "timestamp", frequency)?;
        for (row, time) in times.iter().enumerate() {
            if !quotes.index.insert(*time) {
                bail!("Comparison input has duplicate timestamp/symbol");
            }
            for field in &present {
                let values = frame.col(field)?.to_f64(field)?;
                quotes
                    .fields
                    .get_mut(*field)
                    .unwrap()
                    .insert(*time, values[row]);
            }
        }
    }
    Ok(quotes)
}

fn isclose(a: f64, b: f64, atol: f64, rtol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_infinite() || b.is_infinite() {
        return a == b;
    }
    (a - b).abs() <= atol + rtol * b.abs()
}

/// Compare provider-exported canonical quotes, one security at a time.
///
/// Returns exact mismatch and missing-observation counts, with truncation
/// status; detailed examples are bounded by ``max_differences``.
pub fn compare_snapshots(
    local: &Path,
    reference: &Path,
    output: &Path,
    frequency: &str,
    atol: f64,
    rtol: f64,
    max_differences: usize,
) -> Result<Value> {
    if atol.min(rtol) < 0.0 || max_differences < 1 {
        bail!("Invalid comparison tolerance or example budget");
    }
    if output.exists() {
        bail!(Error::FileExists(output.display().to_string()));
    }
    let left = Dataset::open(local, false)?;
    let right = Dataset::open(reference, false)?;
    let mut symbols: BTreeSet<String> = dataset_symbols(&left)?.into_iter().collect();
    symbols.extend(dataset_symbols(&right)?);
    let mut summaries: Vec<Map<String, Value>> = Vec::new();
    // (symbol, timestamp, reason, field, local, reference)
    let mut differences: Vec<(String, i64, &'static str, Option<String>, f64, f64)> = Vec::new();
    for symbol in &symbols {
        let local_frame = symbol_frame(&left, symbol, frequency)?;
        let reference_frame = symbol_frame(&right, symbol, frequency)?;
        let shared: Vec<i64> = local_frame
            .index
            .intersection(&reference_frame.index)
            .copied()
            .collect();
        let missing_local: Vec<i64> = reference_frame
            .index
            .difference(&local_frame.index)
            .copied()
            .collect();
        let missing_reference: Vec<i64> = local_frame
            .index
            .difference(&reference_frame.index)
            .copied()
            .collect();
        let fields: Vec<String> = local_frame
            .fields
            .keys()
            .filter(|k| reference_frame.fields.contains_key(*k))
            .cloned()
            .collect();
        if fields.is_empty() {
            bail!("No shared quote fields to compare");
        }
        let mut summary = Map::new();
        summary.insert("symbol".into(), Value::String(symbol.clone()));
        summary.insert("matched_rows".into(), Value::from(shared.len()));
        summary.insert("missing_local".into(), Value::from(missing_local.len()));
        summary.insert(
            "missing_reference".into(),
            Value::from(missing_reference.len()),
        );
        let mut mismatched = 0usize;
        for (kind, times) in [
            ("missing_local", &missing_local),
            ("missing_reference", &missing_reference),
        ] {
            for &timestamp in times
                .iter()
                .take(max_differences.saturating_sub(differences.len()))
            {
                differences.push((symbol.clone(), timestamp, kind, None, f64::NAN, f64::NAN));
            }
        }
        let mut per_field = Vec::new();
        for field in &fields {
            let actual = &local_frame.fields[field];
            let expected = &reference_frame.fields[field];
            let mut count = 0usize;
            for &timestamp in &shared {
                let (a, b) = (actual[&timestamp], expected[&timestamp]);
                if !isclose(a, b, atol, rtol) {
                    count += 1;
                    if differences.len() < max_differences {
                        differences.push((
                            symbol.clone(),
                            timestamp,
                            "value_mismatch",
                            Some(field.clone()),
                            a,
                            b,
                        ));
                    }
                }
            }
            per_field.push((format!("{field}_mismatches"), count));
            mismatched += count;
        }
        summary.insert("mismatched_cells".into(), Value::from(mismatched));
        for (name, count) in per_field {
            summary.insert(name, Value::from(count));
        }
        summaries.push(summary);
    }
    let total: u64 = summaries
        .iter()
        .map(|s| {
            s["mismatched_cells"].as_u64().unwrap()
                + s["missing_local"].as_u64().unwrap()
                + s["missing_reference"].as_u64().unwrap()
        })
        .sum();
    let result = json!({
        "status": "completed",
        "local": local.display().to_string(),
        "reference": reference.display().to_string(),
        "atol": atol,
        "rtol": rtol,
        "total_differences": total,
        "examples_truncated": total as usize > differences.len(),
        "symbols": summaries,
    });
    fs::create_dir_all(output)?;
    // summary.csv: union of keys in first-appearance order.
    let mut columns: Vec<String> = Vec::new();
    for summary in &summaries {
        for key in summary.keys() {
            if !columns.contains(key) {
                columns.push(key.clone());
            }
        }
    }
    let mut writer = csv::WriterBuilder::new().from_writer(Vec::new());
    writer.write_record(&columns)?;
    for summary in &summaries {
        writer.write_record(columns.iter().map(|c| match summary.get(c) {
            None | Some(Value::Null) => String::new(),
            Some(Value::Number(n))
                if n.as_f64().is_some() && n.as_i64().is_none() && n.as_u64().is_none() =>
            {
                python_float_repr(n.as_f64().unwrap())
            }
            Some(Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
        }))?;
    }
    fs::write(output.join("summary.csv"), writer.into_inner()?)?;
    let mut frame = OutFrame::new();
    frame.push(
        "symbol",
        OutCol::Str(differences.iter().map(|d| d.0.clone()).collect()),
    );
    frame.push(
        "timestamp",
        OutCol::Ts(differences.iter().map(|d| d.1).collect()),
    );
    frame.push(
        "reason",
        OutCol::Str(differences.iter().map(|d| d.2.to_string()).collect()),
    );
    let field_values: Vec<Option<String>> = differences.iter().map(|d| d.3.clone()).collect();
    frame.push(
        "local",
        OutCol::F64(differences.iter().map(|d| d.4).collect()),
    );
    frame.push(
        "reference",
        OutCol::F64(differences.iter().map(|d| d.5).collect()),
    );
    write_differences(&output.join("differences.parquet"), &frame, &field_values)?;
    fs::write(
        output.join("comparison.json"),
        pyjson::dumps(
            &result,
            Options {
                sort_keys: false,
                indent: Some(2),
                ensure_ascii: true,
            },
        ) + "\n",
    )?;
    Ok(result)
}

/// Write the differences table with a nullable ``field`` column.
fn write_differences(path: &Path, frame: &OutFrame, fields: &[Option<String>]) -> Result<()> {
    use arrow::array::{ArrayRef, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;
    let base = frame.to_batch(true)?;
    let mut columns: Vec<ArrayRef> = base.columns().to_vec();
    let mut schema_fields: Vec<Field> = base
        .schema()
        .fields()
        .iter()
        .map(|f| f.as_ref().clone())
        .collect();
    columns.insert(3, Arc::new(StringArray::from(fields.to_vec())));
    schema_fields.insert(3, Field::new("field", DataType::Utf8, true));
    let batch = RecordBatch::try_new(Arc::new(Schema::new(schema_fields)), columns)?;
    let file = fs::File::create(path)?;
    let mut writer = parquet::arrow::ArrowWriter::try_new(file, batch.schema(), None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

/// Compare caller-authorized API quotes without storing credentials.
pub fn compare_with_provider(
    local: &Path,
    provider: &dyn PriceProvider,
    output: &Path,
    frequency: &str,
) -> Result<Value> {
    let dataset = Dataset::open(local, false)?;
    let parent = output.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let temporary = tempfile::Builder::new()
        .prefix("api_quotes_")
        .tempdir_in(parent)?;
    for symbol in dataset_symbols(&dataset)? {
        let quotes = symbol_frame(&dataset, &symbol, frequency)?;
        if quotes.index.is_empty() {
            continue;
        }
        let first = timeutil::local_date(*quotes.index.iter().next().unwrap());
        let last = timeutil::local_date(*quotes.index.iter().last().unwrap());
        let fetched = provider.fetch_prices(
            &symbol,
            &first.format("%Y-%m-%d").to_string(),
            &last.format("%Y-%m-%d").to_string(),
            frequency,
        )?;
        for required in ["close", "symbol", "timestamp"] {
            if fetched.get(required).is_none() {
                bail!("Provider did not return canonical quote fields");
            }
        }
        if fetched.col("symbol").strs().iter().any(|s| s != &symbol) {
            bail!("Provider returned another security");
        }
        let name = hex::encode(<sha2::Sha256 as sha2::Digest>::digest(symbol.as_bytes()));
        fetched.write_parquet(&temporary.path().join(format!("{name}.parquet")))?;
    }
    compare_snapshots(
        local,
        temporary.path(),
        output,
        frequency,
        1e-8,
        1e-6,
        10_000,
    )
}
