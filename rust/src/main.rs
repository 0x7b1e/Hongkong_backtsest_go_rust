//! Command-line entry points for research and data validation.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde_json::{json, Map, Value};

use hkfactor::batch;
use hkfactor::config::BacktestConfig;
use hkfactor::data;
use hkfactor::demo::{self, DemoOptions};
use hkfactor::pyjson::{self, Options};
use hkfactor::signal_adapter;
use hkfactor::signals::SignalColumns;
use hkfactor::tools;
use hkfactor::validation;

#[derive(Parser)]
#[command(name = "hkfactor", about = "Hong Kong factor backtesting")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, ValueEnum)]
enum Frequency {
    #[value(name = "1d")]
    Daily,
    #[value(name = "5m")]
    FiveMinute,
    #[value(name = "1m")]
    OneMinute,
}

impl Frequency {
    fn as_str(self) -> &'static str {
        match self {
            Frequency::Daily => "1d",
            Frequency::FiveMinute => "5m",
            Frequency::OneMinute => "1m",
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum Kind {
    Stock,
    Benchmark,
}

#[derive(Clone, Copy, ValueEnum)]
enum PriceBasis {
    Raw,
    Adjusted,
}

#[derive(Clone, Copy, ValueEnum)]
enum Unit {
    S,
    Ms,
    Us,
    Ns,
}

#[derive(Clone, Copy, ValueEnum)]
enum Label {
    #[value(name = "bar_start")]
    BarStart,
    #[value(name = "bar_end")]
    BarEnd,
}

#[derive(Subcommand)]
enum Command {
    /// Create synthetic input fixtures
    Demo {
        #[arg(long)]
        output: PathBuf,
        #[arg(long, value_enum, default_value = "1d")]
        frequency: Frequency,
    },
    /// Validate and cache market
    Prepare {
        #[arg(long)]
        prices: PathBuf,
        #[arg(long)]
        benchmark: PathBuf,
        #[arg(long)]
        constituents: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        sessions: Option<PathBuf>,
        #[arg(long, value_enum, default_value = "1d")]
        frequency: Frequency,
        #[arg(long = "t-start")]
        t_start: Option<String>,
        #[arg(long = "t-end")]
        t_end: Option<String>,
        #[arg(long = "max-panel-gib", default_value_t = 4.0)]
        max_panel_gib: f64,
    },
    /// Run independent factors in batches
    Run {
        #[arg(long)]
        market: PathBuf,
        #[arg(long)]
        signals: PathBuf,
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 1)]
        workers: i64,
        #[arg(long = "max-in-flight")]
        max_in_flight: Option<i64>,
        #[arg(long = "factor")]
        factors: Vec<String>,
        #[arg(long = "no-resume")]
        no_resume: bool,
        #[arg(long = "no-plots")]
        no_plots: bool,
        #[arg(long = "no-layer-execution")]
        no_layer_execution: bool,
    },
    /// Describe local data gaps
    Audit {
        #[arg(long, default_value = "data")]
        data: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Adapt raw CSV to Parquet
    #[command(name = "import-csv")]
    ImportCsv {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        symbol: String,
        #[arg(long)]
        frequency: String,
        #[arg(long, value_enum)]
        kind: Kind,
        #[arg(long = "price-basis", value_enum, default_value = "adjusted")]
        price_basis: PriceBasis,
    },
    /// Convert per-factor wide Parquet to long signals
    #[command(name = "import-signals")]
    ImportSignals {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long = "timestamp-unit", value_enum)]
        timestamp_unit: Option<Unit>,
        #[arg(long = "timestamp-label", value_enum)]
        timestamp_label: Label,
        #[arg(long = "bar-duration")]
        bar_duration: Option<String>,
        #[arg(long = "publication-delay", default_value = "0s")]
        publication_delay: String,
        #[arg(long = "source-timezone", default_value = "UTC")]
        source_timezone: String,
        #[arg(long = "timestamp-column")]
        timestamp_column: Option<String>,
        #[arg(long = "batch-size", default_value_t = 65_536)]
        batch_size: i64,
    },
    /// Compare an API snapshot
    Compare {
        #[arg(long)]
        local: PathBuf,
        #[arg(long)]
        reference: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value = "1d")]
        frequency: String,
        #[arg(long, default_value_t = 1e-8)]
        atol: f64,
        #[arg(long, default_value_t = 1e-6)]
        rtol: f64,
    },
    /// Measure the complete pipeline on reproducible synthetic market data
    Benchmark {
        /// New directory for inputs, cache, and results
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 16, value_parser = clap::value_parser!(u64).range(1..))]
        factors: u64,
        /// Synthetic business-day sessions
        #[arg(long, default_value_t = 1008, value_parser = clap::value_parser!(u64).range(1..))]
        days: u64,
        #[arg(long, default_value_t = 80, value_parser = clap::value_parser!(u64).range(1..))]
        assets: u64,
        #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(u64).range(1..))]
        workers: u64,
        #[arg(long, value_enum, default_value = "1d")]
        frequency: Frequency,
        /// Execute ten additional layer accounts
        #[arg(long)]
        layers: bool,
    },
    /// Draw README charts from saved synthetic backtest accounts
    #[command(name = "plot-readme")]
    PlotReadme {
        #[arg(long)]
        t0: PathBuf,
        #[arg(long)]
        t1: PathBuf,
        #[arg(long, default_value = "momentum_5")]
        factor: String,
        #[arg(long, default_value = "docs/images")]
        output: PathBuf,
    },
}

/// Load a JSON or TOML configuration and optional signal column mapping.
fn load_config(path: &Path) -> Result<(BacktestConfig, SignalColumns)> {
    let text = std::fs::read_to_string(path).map_err(|_| {
        hkfactor::errors::Error::FileNotFound(format!(
            "No such file or directory: '{}'",
            path.display()
        ))
    })?;
    let raw: Value = if path.extension().map(|e| e == "toml").unwrap_or(false) {
        let parsed: toml::Value = toml::from_str(&text)?;
        serde_json::to_value(parsed)?
    } else {
        serde_json::from_str(&text)?
    };
    let root = raw
        .as_object()
        .ok_or_else(|| anyhow!("Configuration must be an object"))?;
    let mut values: Map<String, Value> = match root.get("backtest") {
        Some(Value::Object(section)) => section.clone(),
        Some(_) => return Err(anyhow!("backtest must be a table")),
        None => root.clone(),
    };
    let mapping = values
        .remove("signal_columns")
        .or_else(|| root.get("signal_columns").cloned())
        .unwrap_or_else(|| json!({}));
    let columns: SignalColumns =
        serde_json::from_value(mapping).map_err(|e| anyhow!("SignalColumns.__init__() {e}"))?;
    columns.validate()?;
    Ok((BacktestConfig::from_map(&values)?, columns))
}

fn print_json(value: &Value, ensure_ascii: bool) {
    println!(
        "{}",
        pyjson::dumps(
            value,
            Options {
                sort_keys: false,
                indent: Some(2),
                ensure_ascii
            }
        )
    );
}

fn execute(command: Command) -> Result<i32> {
    match command {
        Command::Demo { output, frequency } => {
            let result = demo::create_demo(&output, frequency.as_str(), &DemoOptions::default())?;
            print_json(&Value::Object(result), true);
        }
        Command::Prepare {
            prices,
            benchmark,
            constituents,
            output,
            sessions,
            frequency,
            t_start,
            t_end,
            max_panel_gib,
        } => {
            let budget = (max_panel_gib * 1024f64.powi(3)) as u128;
            let panel = data::prepare_market(
                &prices,
                &benchmark,
                &constituents,
                &output,
                frequency.as_str(),
                sessions.as_deref(),
                t_start.as_deref(),
                t_end.as_deref(),
                budget,
            )?;
            let (rows, cols) = panel.shape();
            println!(
                "{}",
                pyjson::dumps(
                    &json!({"shape": [rows, cols], "frequency": panel.frequency, "cache": output.display().to_string()}),
                    Options::COMPACT
                )
            );
        }
        Command::Run {
            market,
            signals,
            config,
            output,
            workers,
            max_in_flight,
            factors,
            no_resume,
            no_plots,
            no_layer_execution,
        } => {
            let (mut config, columns) = load_config(&config)?;
            if no_layer_execution {
                config = config.with_simulate_layers(false);
            }
            if workers < 1 {
                return Err(anyhow!("workers must be a positive integer"));
            }
            if matches!(max_in_flight, Some(v) if v < 1) {
                return Err(anyhow!("max_in_flight must be a positive integer"));
            }
            let summary = batch::run_batch(
                &market,
                &signals,
                &output,
                &config,
                workers as usize,
                max_in_flight.map(|v| v as usize),
                !no_resume,
                !no_plots,
                Some(columns),
                (!factors.is_empty()).then_some(factors),
            )?;
            println!("{}", summary.to_text());
            return Ok(summary.failed() as i32);
        }
        Command::Audit { data, output } => {
            if let Some(parent) = output.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)?;
                }
            }
            let report = validation::audit_local_data(&data)?;
            std::fs::write(
                &output,
                pyjson::dumps(
                    &report,
                    Options {
                        sort_keys: false,
                        indent: Some(2),
                        ensure_ascii: false,
                    },
                ),
            )?;
            println!("{}", output.display());
        }
        Command::ImportCsv {
            input,
            output,
            symbol,
            frequency,
            kind,
            price_basis,
        } => {
            validation::import_csv(
                &input,
                &output,
                &frequency,
                &symbol,
                match price_basis {
                    PriceBasis::Raw => "raw",
                    PriceBasis::Adjusted => "adjusted",
                },
                match kind {
                    Kind::Stock => "stock",
                    Kind::Benchmark => "benchmark",
                },
            )?;
            println!("{}", output.display());
        }
        Command::ImportSignals {
            input,
            output,
            timestamp_unit,
            timestamp_label,
            bar_duration,
            publication_delay,
            source_timezone,
            timestamp_column,
            batch_size,
        } => {
            if batch_size < 1 {
                return Err(anyhow!("batch_size must be a positive integer"));
            }
            let manifest = signal_adapter::import_wide_signals(
                &input,
                &output,
                timestamp_unit.map(|u| match u {
                    Unit::S => "s",
                    Unit::Ms => "ms",
                    Unit::Us => "us",
                    Unit::Ns => "ns",
                }),
                match timestamp_label {
                    Label::BarStart => "bar_start",
                    Label::BarEnd => "bar_end",
                },
                bar_duration.as_deref(),
                &publication_delay,
                &source_timezone,
                timestamp_column.as_deref(),
                batch_size as usize,
            )?;
            print_json(
                &json!({
                    "factors": manifest["factors"].as_array().map(Vec::len).unwrap_or(0),
                    "output_rows": manifest["output_rows"],
                    "manifest": output.join("import_manifest.json").display().to_string(),
                    "availability_policy": manifest["assumed_availability_policy"],
                }),
                false,
            );
        }
        Command::Compare {
            local,
            reference,
            output,
            frequency,
            atol,
            rtol,
        } => {
            let result = validation::compare_snapshots(
                &local, &reference, &output, &frequency, atol, rtol, 10_000,
            )?;
            print_json(&result, true);
        }
        Command::Benchmark {
            output,
            factors,
            days,
            assets,
            workers,
            frequency,
            layers,
        } => {
            return tools::benchmark(
                &output,
                factors,
                days,
                assets,
                workers,
                frequency.as_str(),
                layers,
            );
        }
        Command::PlotReadme {
            t0,
            t1,
            factor,
            output,
        } => {
            tools::plot_readme(&t0, &t1, &factor, &output)?;
            println!("Saved four English charts to {}", output.display());
        }
    }
    Ok(0)
}

fn main() -> ExitCode {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .format(|buffer, record| writeln!(buffer, "{} {}", record.level(), record.args()))
        .init();
    let cli = Cli::parse();
    match execute(cli.command) {
        Ok(code) => ExitCode::from(code as u8),
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(2)
        }
    }
}
