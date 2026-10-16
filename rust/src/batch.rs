//! Bounded, resumable factor jobs over a shared market panel.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};

use anyhow::{bail, Result};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::analysis;
use crate::config::BacktestConfig;
use crate::data;
use crate::errors::Error;
use crate::execution;
use crate::frames::{python_float_repr, OutFrame};
use crate::models::MarketPanel;
use crate::pyjson::{self, Options};
use crate::reporting;
use crate::signals::{SignalColumns, SignalStore};

const CACHE_VERSION: i64 = 2;

/// Represent unavailable scalar metrics as strict JSON null values.
pub fn metric_scalars(metrics: &Map<String, Value>) -> Map<String, Value> {
    metrics.clone()
}

/// Persist JSON (sorted keys, indent 2, no trailing newline).
fn dump_json(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        path,
        pyjson::dumps(
            value,
            Options {
                sort_keys: true,
                indent: Some(2),
                ensure_ascii: true,
            },
        ),
    )?;
    Ok(())
}

/// Return an unambiguous hash of a canonical JSON object.
fn hash_value(value: &Value) -> String {
    let encoded = pyjson::dumps(
        value,
        Options {
            sort_keys: true,
            indent: None,
            ensure_ascii: true,
        },
    );
    hex::encode(Sha256::digest(encoded.as_bytes()))
}

fn mtime_ns(path: &Path) -> Result<i128> {
    let modified = fs::metadata(path)?.modified()?;
    Ok(match modified.duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => d.as_nanos() as i128,
        Err(e) => -(e.duration().as_nanos() as i128),
    })
}

fn all_files(directory: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.is_dir() {
            all_files(&path, out)?;
        } else if path.is_file() {
            out.push(path);
        }
    }
    Ok(())
}

/// Fingerprint Parquet file identity without reading every factor score.
fn signal_identity(path: &Path) -> Result<Value> {
    let files: Vec<PathBuf> = if path.is_file() {
        vec![path.to_path_buf()]
    } else if path.is_dir() {
        let mut files = Vec::new();
        all_files(path, &mut files)?;
        files.sort();
        files
    } else {
        bail!(Error::FileNotFound(path.display().to_string()));
    };
    if files.is_empty() {
        bail!("Signal path contains no files");
    }
    let mut records = Vec::new();
    for item in &files {
        let name = if path.is_dir() {
            item.strip_prefix(path).unwrap().display().to_string()
        } else {
            item.file_name().unwrap().to_string_lossy().to_string()
        };
        records.push(json!({
            "path": name,
            "size": fs::metadata(item)?.len(),
            "mtime_ns": mtime_ns(item)? as i64,
        }));
    }
    Ok(json!({
        "path": fs::canonicalize(path)?.display().to_string(),
        "file_count": records.len(),
        "stat_fingerprint": hash_value(&Value::Array(records)),
    }))
}

/// Use a full digest so external factor names never become paths.
pub fn factor_directory(run_dir: &Path, factor: &str) -> PathBuf {
    run_dir
        .join("factors")
        .join(hex::encode(Sha256::digest(factor.as_bytes())))
}

fn error_kind(error: &anyhow::Error) -> &'static str {
    match error.downcast_ref::<Error>() {
        Some(Error::FileExists(_)) => "FileExistsError",
        Some(Error::FileNotFound(_)) => "FileNotFoundError",
        Some(Error::Memory(_)) => "MemoryError",
        None => {
            if error.downcast_ref::<std::io::Error>().is_some() {
                "OSError"
            } else {
                "ValueError"
            }
        }
    }
}

/// Describe failed jobs uniformly, including worker failures.
fn failed_summary(factor: &str, run_dir: &Path, error: &str) -> Map<String, Value> {
    let mut row = Map::new();
    row.insert("factor".into(), Value::String(factor.into()));
    row.insert("status".into(), Value::String("failed".into()));
    row.insert("error".into(), Value::String(error.into()));
    row.insert(
        "output_directory".into(),
        Value::String(factor_directory(run_dir, factor).display().to_string()),
    );
    row.insert("run_id".into(), Value::String(run_name(run_dir)));
    row
}

fn run_name(run_dir: &Path) -> String {
    run_dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default()
}

/// Accept only a completed output with an identical input contract.
fn cached_summary(directory: &Path, factor: &str, identity: &Value) -> Option<Map<String, Value>> {
    let text = fs::read_to_string(directory.join("completed.json")).ok()?;
    let saved: Value = serde_json::from_str(&text).ok()?;
    if saved.get("identity")? != identity || saved.get("factor")?.as_str()? != factor {
        return None;
    }
    let summary = saved.get("summary")?.as_object()?.clone();
    let artifacts = saved.get("artifacts")?.as_array()?;
    if summary.get("status")?.as_str()? != "success" || artifacts.is_empty() {
        return None;
    }
    for artifact in artifacts {
        let relative = PathBuf::from(artifact.get("path")?.as_str()?);
        if relative.is_absolute()
            || relative
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            return None;
        }
        let file = directory.join(&relative);
        let size = artifact.get("size")?.as_u64()?;
        if !file.is_file() || fs::metadata(&file).ok()?.len() != size {
            return None;
        }
    }
    let mut summary = summary;
    summary.insert(
        "output_directory".into(),
        Value::String(directory.display().to_string()),
    );
    Some(summary)
}

/// Shared, immutable inputs of every factor job.
struct Worker {
    panel: Arc<MarketPanel>,
    store: SignalStore,
    config: BacktestConfig,
    run_dir: PathBuf,
    identity: Value,
    plots: bool,
}

fn staging_directory(run_dir: &Path) -> Result<PathBuf> {
    Ok(tempfile::Builder::new()
        .prefix(".staging-")
        .tempdir_in(run_dir)?
        .keep())
}

impl Worker {
    /// Execute and atomically publish one independent factor experiment.
    fn run_factor(&self, factor: &str) -> Map<String, Value> {
        let staging = match staging_directory(&self.run_dir) {
            Ok(path) => path,
            Err(error) => {
                return failed_summary(factor, &self.run_dir, &format!("OSError: {error}"))
            }
        };
        let outcome = catch_unwind(AssertUnwindSafe(|| self.run_in(factor, &staging)));
        match outcome {
            Ok(Ok(summary)) => summary,
            Ok(Err(error)) => {
                let _ = fs::remove_dir_all(&staging);
                failed_summary(
                    factor,
                    &self.run_dir,
                    &format!("{}: {error}", error_kind(&error)),
                )
            }
            Err(panic) => {
                let _ = fs::remove_dir_all(&staging);
                let message = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "worker panicked".into());
                failed_summary(factor, &self.run_dir, &format!("RuntimeError: {message}"))
            }
        }
    }

    fn run_in(&self, factor: &str, staging: &Path) -> Result<Map<String, Value>> {
        let panel = self.panel.as_ref();
        let signal = self.store.read_factor(factor, panel, Some(staging))?;
        let result = execution::run_backtest(panel, &signal, &self.config, Some(staging))?;
        let mut analysis_result = analysis::analyze_factor(panel, &signal, &self.config)?;
        if self.config.simulate_layers {
            let mut accounts: Vec<OutFrame> = Vec::new();
            analysis::for_each_layer_signal(
                panel,
                &signal,
                &self.config,
                |layer, layer_signal| {
                    let layer_dir = staging.join("layers").join(layer.to_string());
                    let layer_result = execution::run_backtest(
                        panel,
                        &layer_signal,
                        &self.config,
                        Some(&layer_dir),
                    )?;
                    dump_json(
                        &layer_dir.join("metrics.json"),
                        &Value::Object(metric_scalars(&layer_result.metrics)),
                    )?;
                    let mut pnl = layer_result.pnl;
                    pnl.push("layer", crate::frames::OutCol::I64(vec![layer; pnl.len()]));
                    accounts.push(pnl);
                    Ok(())
                },
            )?;
            analysis_result.layer_pnl = Some(if accounts.is_empty() {
                OutFrame::new()
            } else {
                OutFrame::concat(&accounts)?
            });
        }
        let mut metadata = Map::new();
        metadata.insert("factor".into(), Value::String(factor.into()));
        metadata.insert("run_id".into(), Value::String(run_name(&self.run_dir)));
        metadata.insert("input_identity".into(), self.identity.clone());
        metadata.insert("config".into(), Value::Object(self.config.to_dict()));
        reporting::write_report(
            &result,
            &analysis_result,
            staging,
            Some(metadata),
            self.plots,
        )?;
        let directory = factor_directory(&self.run_dir, factor);
        let mut summary = Map::new();
        summary.insert("factor".into(), Value::String(factor.into()));
        for (key, value) in metric_scalars(&result.metrics) {
            summary.insert(key, value);
        }
        summary.insert("status".into(), Value::String("success".into()));
        summary.insert("error".into(), Value::String(String::new()));
        summary.insert(
            "output_directory".into(),
            Value::String(directory.display().to_string()),
        );
        summary.insert("run_id".into(), Value::String(run_name(&self.run_dir)));
        let mut files = Vec::new();
        all_files(staging, &mut files)?;
        files.sort();
        let artifacts: Vec<Value> = files
            .iter()
            .map(|file| {
                Ok(json!({
                    "path": file.strip_prefix(staging).unwrap().display().to_string(),
                    "size": fs::metadata(file)?.len(),
                }))
            })
            .collect::<Result<_>>()?;
        dump_json(
            &staging.join("completed.json"),
            &json!({
                "factor": factor,
                "identity": self.identity,
                "summary": summary,
                "artifacts": artifacts,
            }),
        )?;
        if directory.exists() {
            let backup = staging.with_file_name(format!(
                "{}-previous",
                staging.file_name().unwrap().to_string_lossy()
            ));
            fs::rename(&directory, &backup)?;
            if let Err(error) = fs::rename(staging, &directory) {
                fs::rename(&backup, &directory)?;
                return Err(error.into());
            }
            fs::remove_dir_all(&backup)?;
        } else {
            fs::rename(staging, &directory)?;
        }
        Ok(summary)
    }
}

/// A CSV cell for one JSON scalar, as pandas writes it.
fn csv_cell(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::Bool(b)) => {
            if *b {
                "True".into()
            } else {
                "False".into()
            }
        }
        Some(Value::Number(n)) => match n.as_i64() {
            Some(i) => i.to_string(),
            None => python_float_repr(n.as_f64().unwrap_or(f64::NAN)),
        },
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
    }
}

/// Summary table: fixed leading columns followed by sorted metric names.
pub struct Summary {
    pub columns: Vec<String>,
    pub rows: Vec<Map<String, Value>>,
}

impl Summary {
    fn new(mut rows: Vec<Map<String, Value>>) -> Summary {
        let leading = ["factor", "status", "error", "output_directory", "run_id"];
        rows.sort_by(|a, b| {
            let key = |r: &Map<String, Value>| {
                r.get("factor")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string()
            };
            key(a).cmp(&key(b))
        });
        let mut extra: Vec<String> = rows
            .iter()
            .flat_map(|r| r.keys().cloned())
            .filter(|k| !leading.contains(&k.as_str()))
            .collect();
        extra.sort();
        extra.dedup();
        let mut columns: Vec<String> = leading.iter().map(|s| s.to_string()).collect();
        columns.extend(extra);
        Summary { columns, rows }
    }

    pub fn write_csv(&self, path: &Path) -> Result<()> {
        let mut writer = csv::WriterBuilder::new().from_writer(Vec::new());
        writer.write_record(&self.columns)?;
        for row in &self.rows {
            writer.write_record(self.columns.iter().map(|c| csv_cell(row.get(c))))?;
        }
        let mut bytes = "\u{feff}".as_bytes().to_vec();
        bytes.extend(writer.into_inner()?);
        fs::write(path, bytes)?;
        Ok(())
    }

    pub fn failed(&self) -> bool {
        self.rows
            .iter()
            .any(|r| r.get("status").and_then(Value::as_str) == Some("failed"))
    }

    /// A plain-text table similar to ``DataFrame.to_string(index=False)``.
    pub fn to_text(&self) -> String {
        let cells: Vec<Vec<String>> = self
            .rows
            .iter()
            .map(|row| {
                self.columns
                    .iter()
                    .map(|c| match row.get(c) {
                        None | Some(Value::Null) => "NaN".to_string(),
                        Some(Value::Number(n)) if n.as_i64().is_none() => {
                            format!("{:.6}", n.as_f64().unwrap())
                        }
                        other => csv_cell(other),
                    })
                    .collect()
            })
            .collect();
        let widths: Vec<usize> = self
            .columns
            .iter()
            .enumerate()
            .map(|(i, c)| {
                cells
                    .iter()
                    .map(|r| r[i].chars().count())
                    .chain([c.chars().count()])
                    .max()
                    .unwrap_or(0)
            })
            .collect();
        let line = |values: Vec<&str>| -> String {
            values
                .iter()
                .zip(&widths)
                .map(|(v, w)| format!("{v:>w$}", w = *w))
                .collect::<Vec<_>>()
                .join(" ")
        };
        let mut out = vec![line(self.columns.iter().map(String::as_str).collect())];
        for row in &cells {
            out.push(line(row.iter().map(String::as_str).collect()));
        }
        out.join("\n")
    }
}

/// Publish the current requested-factor summary with atomic replacement.
fn write_summary(output_dir: &Path, rows: &[Map<String, Value>]) -> Result<Summary> {
    let summary = Summary::new(rows.to_vec());
    let temporary = tempfile::Builder::new()
        .prefix(".summary-")
        .suffix(".csv")
        .tempfile_in(output_dir)?;
    summary.write_csv(temporary.path())?;
    temporary.persist(output_dir.join("summary.csv"))?;
    Ok(summary)
}

/// Run factors with bounded submissions, failure isolation, and resume.
///
/// ``workers`` is the maximum number of concurrent factor jobs, each sharing
/// the read-only market panel; ``max_in_flight`` bounds unfinished jobs and
/// defaults to twice ``workers``. ``resume`` reuses only complete reports
/// matching all input identities. Failed factors retain their error details
/// in the run manifest.
#[allow(clippy::too_many_arguments)]
pub fn run_batch(
    market_dir: &Path,
    signal_path: &Path,
    output_dir: &Path,
    config: &BacktestConfig,
    workers: usize,
    max_in_flight: Option<usize>,
    resume: bool,
    plots: bool,
    columns: Option<SignalColumns>,
    factors: Option<Vec<String>>,
) -> Result<Summary> {
    if workers < 1 {
        bail!("workers must be a positive integer");
    }
    let limit = max_in_flight.unwrap_or(workers * 2);
    if limit < 1 {
        bail!("max_in_flight must be a positive integer");
    }
    let mapping = columns.unwrap_or_default();
    let store = SignalStore::new(signal_path, mapping.clone())?;
    let mut requested = match factors {
        Some(list) => list,
        None => store.factors()?,
    };
    if requested.iter().any(String::is_empty) {
        bail!("Factor identifiers must be nonempty strings");
    }
    let unique: std::collections::BTreeSet<&String> = requested.iter().collect();
    if unique.len() != requested.len() {
        bail!("Requested factor identifiers must be unique");
    }
    requested.sort();
    let identity = json!({
        "cache_version": CACHE_VERSION,
        "implementation_sha256": env!("HKFACTOR_SOURCE_SHA256"),
        "library_versions": {
            "hkfactor": env!("CARGO_PKG_VERSION"),
            "arrow": "60",
            "parquet": "60",
        },
        "config": Value::Object(config.to_dict()),
        "market": data::panel_fingerprint(market_dir)?,
        "signals": signal_identity(signal_path)?,
        "columns": serde_json::to_value(&mapping)?,
        "plots": plots,
    });
    let run_dir = output_dir.join("runs").join(hash_value(&identity));
    fs::create_dir_all(run_dir.join("factors"))?;
    dump_json(&run_dir.join("identity.json"), &identity)?;
    let mut rows: Vec<Map<String, Value>> = Vec::new();
    let mut pending = Vec::new();
    for factor in &requested {
        let cached = if resume {
            cached_summary(&factor_directory(&run_dir, factor), factor, &identity)
        } else {
            None
        };
        match cached {
            Some(row) => rows.push(row),
            None => pending.push(factor.clone()),
        }
    }
    log::info!(
        "Factors: {} total, {} cached, {} pending; workers={}, in-flight={}",
        requested.len(),
        rows.len(),
        pending.len(),
        workers,
        limit
    );
    let mut manifest = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(run_dir.join("manifest.jsonl"))?;
    let total = requested.len();
    let mut record = |row: Map<String, Value>, rows: &mut Vec<Map<String, Value>>| -> Result<()> {
        let line = pyjson::dumps(&Value::Object(row.clone()), Options::COMPACT);
        writeln!(manifest, "{line}")?;
        manifest.flush()?;
        rows.push(row.clone());
        log::info!(
            "Completed {}/{} factors: {} ({})",
            rows.len(),
            total,
            row.get("factor").and_then(Value::as_str).unwrap_or(""),
            row.get("status").and_then(Value::as_str).unwrap_or("")
        );
        Ok(())
    };
    if !pending.is_empty() {
        let panel = match MarketPanel::load(market_dir) {
            Ok(panel) => Arc::new(panel),
            Err(error) => {
                let message = format!("{}: {error}", error_kind(&error));
                for factor in &pending {
                    record(failed_summary(factor, &run_dir, &message), &mut rows)?;
                }
                let summary = write_summary(output_dir, &rows)?;
                summary.write_csv(&run_dir.join("summary.csv"))?;
                return Ok(summary);
            }
        };
        let worker = Arc::new(Worker {
            panel,
            store: SignalStore::new(signal_path, mapping.clone())?,
            config: config.clone(),
            run_dir: run_dir.clone(),
            identity: identity.clone(),
            plots,
        });
        if workers == 1 {
            for factor in &pending {
                record(worker.run_factor(factor), &mut rows)?;
            }
        } else {
            let threads = workers.min(limit).min(pending.len());
            let queue = Arc::new(Mutex::new(pending.clone().into_iter()));
            let (sender, receiver) = mpsc::channel::<Map<String, Value>>();
            let mut handles = Vec::new();
            for _ in 0..threads {
                let queue = Arc::clone(&queue);
                let sender = sender.clone();
                let worker = Arc::clone(&worker);
                handles.push(std::thread::spawn(move || loop {
                    let next = queue.lock().ok().and_then(|mut q| q.next());
                    match next {
                        Some(factor) => {
                            if sender.send(worker.run_factor(&factor)).is_err() {
                                break;
                            }
                        }
                        None => break,
                    }
                }));
            }
            drop(sender);
            let mut finished: BTreeMap<String, ()> = BTreeMap::new();
            for row in receiver {
                if let Some(name) = row.get("factor").and_then(Value::as_str) {
                    finished.insert(name.to_string(), ());
                }
                record(row, &mut rows)?;
            }
            for handle in handles {
                let _ = handle.join();
            }
            for factor in &pending {
                if !finished.contains_key(factor) {
                    record(
                        failed_summary(factor, &run_dir, "RuntimeError: worker terminated"),
                        &mut rows,
                    )?;
                }
            }
        }
    }
    let summary = write_summary(output_dir, &rows)?;
    summary.write_csv(&run_dir.join("summary.csv"))?;
    Ok(summary)
}
