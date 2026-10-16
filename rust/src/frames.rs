//! Typed output tables and their Parquet/CSV serialization.

use std::fs;
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Result};
use arrow::array::{
    ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray, TimestampNanosecondArray,
};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

use crate::timeutil::{self, TIMEZONE};

/// One output column.
#[derive(Debug, Clone, PartialEq)]
pub enum OutCol {
    /// Timezone-aware ``Asia/Hong_Kong`` instants (UTC ns).
    Ts(Vec<i64>),
    F64(Vec<f64>),
    I64(Vec<i64>),
    Str(Vec<String>),
    Bool(Vec<bool>),
}

impl OutCol {
    pub fn len(&self) -> usize {
        match self {
            OutCol::Ts(v) => v.len(),
            OutCol::F64(v) => v.len(),
            OutCol::I64(v) => v.len(),
            OutCol::Str(v) => v.len(),
            OutCol::Bool(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn empty_like(&self) -> OutCol {
        match self {
            OutCol::Ts(_) => OutCol::Ts(Vec::new()),
            OutCol::F64(_) => OutCol::F64(Vec::new()),
            OutCol::I64(_) => OutCol::I64(Vec::new()),
            OutCol::Str(_) => OutCol::Str(Vec::new()),
            OutCol::Bool(_) => OutCol::Bool(Vec::new()),
        }
    }

    fn extend(&mut self, other: &OutCol) -> Result<()> {
        match (self, other) {
            (OutCol::Ts(a), OutCol::Ts(b)) => a.extend_from_slice(b),
            (OutCol::F64(a), OutCol::F64(b)) => a.extend_from_slice(b),
            (OutCol::I64(a), OutCol::I64(b)) => a.extend_from_slice(b),
            (OutCol::Str(a), OutCol::Str(b)) => a.extend_from_slice(b),
            (OutCol::Bool(a), OutCol::Bool(b)) => a.extend_from_slice(b),
            _ => bail!("Cannot concatenate columns of different types"),
        }
        Ok(())
    }

    pub fn f64s(&self) -> &[f64] {
        match self {
            OutCol::F64(v) => v,
            _ => panic!("not a float column"),
        }
    }

    pub fn i64s(&self) -> &[i64] {
        match self {
            OutCol::I64(v) | OutCol::Ts(v) => v,
            _ => panic!("not an integer column"),
        }
    }

    pub fn strs(&self) -> &[String] {
        match self {
            OutCol::Str(v) => v,
            _ => panic!("not a string column"),
        }
    }

    /// Arrow array; NaN floats become nulls, as ``Table.from_pandas``.
    fn to_arrow(&self, nan_as_null: bool) -> ArrayRef {
        match self {
            OutCol::Ts(v) => {
                Arc::new(TimestampNanosecondArray::from(v.clone()).with_timezone(TIMEZONE))
            }
            OutCol::F64(v) => {
                if nan_as_null {
                    Arc::new(Float64Array::from(
                        v.iter()
                            .map(|x| (!x.is_nan()).then_some(*x))
                            .collect::<Vec<_>>(),
                    ))
                } else {
                    Arc::new(Float64Array::from(v.clone()))
                }
            }
            OutCol::I64(v) => Arc::new(Int64Array::from(v.clone())),
            OutCol::Str(v) => Arc::new(StringArray::from(v.clone())),
            OutCol::Bool(v) => Arc::new(BooleanArray::from(v.clone())),
        }
    }

    fn data_type(&self) -> DataType {
        match self {
            OutCol::Ts(_) => DataType::Timestamp(TimeUnit::Nanosecond, Some(TIMEZONE.into())),
            OutCol::F64(_) => DataType::Float64,
            OutCol::I64(_) => DataType::Int64,
            OutCol::Str(_) => DataType::Utf8,
            OutCol::Bool(_) => DataType::Boolean,
        }
    }

    fn display(&self, row: usize) -> String {
        match self {
            OutCol::Ts(v) => timeutil::strftime_hk(v[row], "%Y-%m-%d %H:%M:%S%:z"),
            OutCol::F64(v) => format_float(v[row]),
            OutCol::I64(v) => v[row].to_string(),
            OutCol::Str(v) => v[row].clone(),
            OutCol::Bool(v) => {
                if v[row] {
                    "True".into()
                } else {
                    "False".into()
                }
            }
        }
    }
}

/// Format a float as Python's ``repr`` (empty for NaN in CSV output).
pub fn format_float(value: f64) -> String {
    if value.is_nan() {
        return String::new();
    }
    python_float_repr(value)
}

/// Python ``repr(float)``: shortest round-trip with Python's exponent rules.
pub fn python_float_repr(value: f64) -> String {
    if value.is_nan() {
        return "nan".into();
    }
    if value.is_infinite() {
        return if value > 0.0 {
            "inf".into()
        } else {
            "-inf".into()
        };
    }
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    let exp_form = format!("{value:e}");
    let (mantissa, exponent) = exp_form.split_once('e').unwrap();
    let exponent: i32 = exponent.parse().unwrap();
    if (-4..16).contains(&exponent) {
        let text = format!("{value}");
        if text.contains('.') {
            text
        } else {
            format!("{text}.0")
        }
    } else {
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{mantissa}e{sign}{:02}", exponent.abs())
    }
}

/// A small named-column table.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OutFrame {
    pub names: Vec<String>,
    pub columns: Vec<OutCol>,
}

impl OutFrame {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, name: &str, col: OutCol) {
        if let Some(i) = self.names.iter().position(|n| n == name) {
            self.columns[i] = col;
        } else {
            self.names.push(name.into());
            self.columns.push(col);
        }
    }

    pub fn len(&self) -> usize {
        self.columns.first().map(OutCol::len).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, name: &str) -> Option<&OutCol> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| &self.columns[i])
    }

    pub fn col(&self, name: &str) -> &OutCol {
        self.get(name)
            .unwrap_or_else(|| panic!("missing column {name}"))
    }

    /// Typed empty table with the same columns.
    pub fn empty_like(&self) -> OutFrame {
        OutFrame {
            names: self.names.clone(),
            columns: self.columns.iter().map(OutCol::empty_like).collect(),
        }
    }

    /// ``pd.concat(frames, ignore_index=True)`` for identical schemas.
    pub fn concat(parts: &[OutFrame]) -> Result<OutFrame> {
        let mut iter = parts.iter();
        let mut result = match iter.next() {
            Some(first) => first.clone(),
            None => return Ok(OutFrame::new()),
        };
        for part in iter {
            if part.names != result.names {
                bail!("Cannot concatenate frames with different columns");
            }
            for (target, source) in result.columns.iter_mut().zip(&part.columns) {
                target.extend(source)?;
            }
        }
        Ok(result)
    }

    pub fn schema(&self) -> Schema {
        Schema::new(
            self.names
                .iter()
                .zip(&self.columns)
                .map(|(name, col)| Field::new(name, col.data_type(), true))
                .collect::<Vec<_>>(),
        )
    }

    pub fn to_batch(&self, nan_as_null: bool) -> Result<RecordBatch> {
        let schema = Arc::new(self.schema());
        let arrays: Vec<ArrayRef> = self
            .columns
            .iter()
            .map(|c| c.to_arrow(nan_as_null))
            .collect();
        if arrays.is_empty() {
            return Ok(RecordBatch::new_empty(schema));
        }
        Ok(RecordBatch::try_new(schema, arrays)?)
    }

    /// ``DataFrame.to_parquet(index=False)``; NaN floats are stored as nulls.
    pub fn write_parquet(&self, path: &Path) -> Result<()> {
        let batch = self.to_batch(true)?;
        let file = fs::File::create(path)?;
        let properties = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(properties))?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(())
    }

    /// ``DataFrame.to_csv(index=False)``, optionally with a UTF-8 BOM.
    pub fn write_csv(&self, path: &Path, bom: bool) -> Result<()> {
        let mut writer = csv::WriterBuilder::new().from_writer(Vec::new());
        writer.write_record(&self.names)?;
        for row in 0..self.len() {
            writer.write_record(self.columns.iter().map(|c| c.display(row)))?;
        }
        let mut bytes = Vec::new();
        if bom {
            bytes.extend_from_slice("\u{feff}".as_bytes());
        }
        bytes.extend(writer.into_inner()?);
        fs::write(path, bytes)?;
        Ok(())
    }
}

/// One execution audit row (trade or restriction).
#[derive(Debug, Clone, PartialEq)]
pub struct AuditRow {
    pub timestamp: i64,
    pub bar_end: i64,
    pub symbol: String,
    pub side: &'static str,
    pub requested: f64,
    pub filled: f64,
    pub trade: Option<TradeDetail>,
    pub reason: &'static str,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TradeDetail {
    pub price: f64,
    pub notional: f64,
    pub fees: f64,
    pub stamp_duty: f64,
    pub sfc_levy: f64,
    pub afrc_levy: f64,
    pub trading_fee: f64,
    pub trading_tariff: f64,
    pub commission: f64,
    pub cash: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditKind {
    Trades,
    Restrictions,
}

impl AuditKind {
    pub fn name(self) -> &'static str {
        match self {
            AuditKind::Trades => "trades",
            AuditKind::Restrictions => "restrictions",
        }
    }
}

/// Retained audit rows (empty when the table was streamed to Parquet).
#[derive(Debug, Clone, PartialEq)]
pub struct AuditTable {
    pub kind: AuditKind,
    pub rows: Vec<AuditRow>,
}

impl AuditTable {
    pub fn to_frame(&self) -> OutFrame {
        rows_to_frame(self.kind, &self.rows)
    }
}

pub fn rows_to_frame(kind: AuditKind, rows: &[AuditRow]) -> OutFrame {
    let mut frame = OutFrame::new();
    frame.push(
        "timestamp",
        OutCol::Ts(rows.iter().map(|r| r.timestamp).collect()),
    );
    frame.push(
        "bar_end",
        OutCol::Ts(rows.iter().map(|r| r.bar_end).collect()),
    );
    frame.push(
        "symbol",
        OutCol::Str(rows.iter().map(|r| r.symbol.clone()).collect()),
    );
    frame.push(
        "side",
        OutCol::Str(rows.iter().map(|r| r.side.to_string()).collect()),
    );
    frame.push(
        "requested",
        OutCol::F64(rows.iter().map(|r| r.requested).collect()),
    );
    frame.push(
        "filled",
        OutCol::F64(rows.iter().map(|r| r.filled).collect()),
    );
    if kind == AuditKind::Trades {
        let detail = |f: fn(&TradeDetail) -> f64| -> OutCol {
            OutCol::F64(
                rows.iter()
                    .map(|r| r.trade.as_ref().map(f).unwrap_or(f64::NAN))
                    .collect(),
            )
        };
        frame.push("price", detail(|t| t.price));
        frame.push("notional", detail(|t| t.notional));
        frame.push("fees", detail(|t| t.fees));
        frame.push("stamp_duty", detail(|t| t.stamp_duty));
        frame.push("sfc_levy", detail(|t| t.sfc_levy));
        frame.push("afrc_levy", detail(|t| t.afrc_levy));
        frame.push("trading_fee", detail(|t| t.trading_fee));
        frame.push("trading_tariff", detail(|t| t.trading_tariff));
        frame.push("commission", detail(|t| t.commission));
        frame.push("cash", detail(|t| t.cash));
    }
    frame.push(
        "reason",
        OutCol::Str(rows.iter().map(|r| r.reason.to_string()).collect()),
    );
    frame
}

pub const AUDIT_BUFFER_ROWS: usize = 4096;

/// Collect one fixed audit table in memory or stream it to Parquet.
///
/// Streaming retains at most 4096 finalized records. Empty streams still
/// create a typed Parquet file.
pub struct AuditRecords {
    pub kind: AuditKind,
    pub count: usize,
    pub peak_buffered_rows: usize,
    buffer_rows: usize,
    records: Vec<AuditRow>,
    writer: Option<ArrowWriter<fs::File>>,
    streaming: bool,
    closed: bool,
}

impl AuditRecords {
    pub fn new(kind: AuditKind, directory: Option<&Path>) -> Result<Self> {
        Self::with_buffer(kind, directory, AUDIT_BUFFER_ROWS)
    }

    pub fn with_buffer(
        kind: AuditKind,
        directory: Option<&Path>,
        buffer_rows: usize,
    ) -> Result<Self> {
        if !(1..=AUDIT_BUFFER_ROWS).contains(&buffer_rows) {
            bail!("Audit buffer must contain 1 to 4096 rows");
        }
        let mut writer = None;
        if let Some(directory) = directory {
            fs::create_dir_all(directory)?;
            let filename = directory.join(format!("{}.parquet", kind.name()));
            if filename.exists() {
                bail!(crate::errors::Error::FileExists(
                    filename.display().to_string()
                ));
            }
            let schema = Arc::new(rows_to_frame(kind, &[]).schema());
            let properties = WriterProperties::builder()
                .set_compression(Compression::ZSTD(ZstdLevel::default()))
                .build();
            writer = Some(ArrowWriter::try_new(
                fs::File::create(&filename)?,
                schema,
                Some(properties),
            )?);
        }
        Ok(AuditRecords {
            kind,
            count: 0,
            peak_buffered_rows: 0,
            buffer_rows,
            records: Vec::new(),
            streaming: writer.is_some(),
            writer,
            closed: false,
        })
    }

    pub fn buffered_rows(&self) -> usize {
        self.records.len()
    }

    pub fn append(&mut self, row: AuditRow) -> Result<()> {
        if self.closed {
            bail!("Execution audit is already closed");
        }
        self.records.push(row);
        self.count += 1;
        self.peak_buffered_rows = self.peak_buffered_rows.max(self.records.len());
        if self.writer.is_some() && self.records.len() >= self.buffer_rows {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if let Some(writer) = self.writer.as_mut() {
            if !self.records.is_empty() {
                let batch = rows_to_frame(self.kind, &self.records).to_batch(false)?;
                writer.write(&batch)?;
                self.records.clear();
            }
        }
        Ok(())
    }

    /// Retained records, or a typed empty table for streamed data.
    pub fn table(&self) -> AuditTable {
        AuditTable {
            kind: self.kind,
            rows: if self.streaming {
                Vec::new()
            } else {
                self.records.clone()
            },
        }
    }

    pub fn is_streaming(&self) -> bool {
        self.streaming
    }

    /// Flush remaining records and close a valid Parquet file.
    pub fn close(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        self.closed = true;
        if self.writer.is_some() {
            let flushed = self.flush();
            let writer = self.writer.take().unwrap();
            let closed = writer.close();
            self.records.clear();
            flushed?;
            closed?;
        }
        Ok(())
    }
}
