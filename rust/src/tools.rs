//! Pipeline benchmark and README chart rendering (ports of ``scripts/``).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, Result};
use plotters::coord::Shift;
use plotters::prelude::*;
use serde_json::{json, Map, Value};

use crate::batch;
use crate::config::BacktestConfig;
use crate::data;
use crate::demo::{self, DemoOptions};
use crate::pyjson::{self, Options};
use crate::reporting;
use crate::table;
use crate::timeutil;
use crate::viridis::VIRIDIS;

fn peak_rss_mib(who: libc::c_int) -> f64 {
    // SAFETY: getrusage writes into the provided, properly sized struct.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    let status = unsafe { libc::getrusage(who, &mut usage) };
    if status != 0 {
        return f64::NAN;
    }
    let divisor = if cfg!(target_os = "macos") {
        1024.0 * 1024.0
    } else {
        1024.0
    };
    usage.ru_maxrss as f64 / divisor
}

fn environment() -> Value {
    // SAFETY: sysconf has no preconditions.
    let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let memory = if pages > 0 && page_size > 0 {
        Value::from(pages as i64 * page_size as i64)
    } else {
        Value::Null
    };
    json!({
        "hardware": {
            "machine": std::env::consts::ARCH,
            "processor": std::env::consts::ARCH,
            "logical_cpu_count": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
            "physical_memory_bytes": memory,
        },
        "software": {
            "platform": format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
            "implementation": "rust",
            "packages": {
                "hkfactor": env!("CARGO_PKG_VERSION"),
                "arrow": "60",
                "parquet": "60",
                "plotters": "0.3",
            },
        },
    })
}

/// Generate data, prepare the panel, and measure factor backtesting.
pub fn benchmark(
    output: &Path,
    factors: u64,
    days: u64,
    assets: u64,
    workers: u64,
    frequency: &str,
    layers: bool,
) -> Result<i32> {
    if days < 6 {
        eprintln!("error: --days must be at least 6 for five-bar factor history");
        return Ok(2);
    }
    if assets < 10 {
        eprintln!("error: --assets must be at least 10 for the synthetic universe");
        return Ok(2);
    }
    if output.exists() {
        eprintln!("error: --output must be a new directory");
        return Ok(2);
    }
    fs::create_dir_all(output)?;
    let output = fs::canonicalize(output)?;
    let mut elapsed = Map::new();
    let total_start = Instant::now();
    let start = Instant::now();
    let inputs = output.join("inputs");
    let synthetic = demo::create_demo(
        &inputs,
        frequency,
        &DemoOptions {
            asset_count: assets as usize,
            trading_days: Some(days as usize),
            factor_count: factors as usize,
            seed: 101,
        },
    )?;
    elapsed.insert(
        "generate_seconds".into(),
        json!(start.elapsed().as_secs_f64()),
    );
    let config_value: Value =
        serde_json::from_str(&fs::read_to_string(inputs.join("config.json"))?)?;
    let config = BacktestConfig::from_json(&config_value)?.with_simulate_layers(layers);
    let market = output.join("market");
    let start = Instant::now();
    let panel = data::prepare_market(
        &inputs.join("prices.parquet"),
        &inputs.join("benchmark.parquet"),
        &inputs.join("constituents.parquet"),
        &market,
        frequency,
        Some(&inputs.join("sessions.parquet")),
        Some(&config.t_start),
        Some(&config.t_end),
        4 * 1024u128.pow(3),
    )?;
    elapsed.insert(
        "prepare_seconds".into(),
        json!(start.elapsed().as_secs_f64()),
    );
    let (rows, cols) = panel.shape();
    drop(panel);
    let start = Instant::now();
    let summary = batch::run_batch(
        &market,
        &inputs.join("signals"),
        &output.join("results"),
        &config,
        workers as usize,
        None,
        false,
        false,
        None,
        None,
    )?;
    let backtest_seconds = start.elapsed().as_secs_f64();
    elapsed.insert("backtest_seconds".into(), json!(backtest_seconds));
    elapsed.insert(
        "total_seconds".into(),
        json!(total_start.elapsed().as_secs_f64()),
    );
    let success = summary
        .rows
        .iter()
        .filter(|r| r.get("status").and_then(Value::as_str) == Some("success"))
        .count();
    let failed = summary.rows.len() - success;
    let mut report = Map::new();
    report.insert("synthetic".into(), json!(true));
    report.insert(
        "note".into(),
        json!("Synthetic business-day data; not real HSI constituents, an official Hong Kong calendar, or investment performance"),
    );
    report.insert(
        "measured_at_utc".into(),
        json!(chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, false)),
    );
    report.insert("dataset".into(), Value::Object(synthetic));
    report.insert("panel_shape".into(), json!([rows, cols]));
    report.insert("config".into(), Value::Object(config.to_dict()));
    report.insert("workers".into(), json!(workers));
    report.insert("plots".into(), json!(false));
    report.insert("elapsed".into(), Value::Object(elapsed.clone()));
    report.insert("successful_factors".into(), json!(success));
    report.insert("failed_factors".into(), json!(failed));
    report.insert(
        "successful_factors_per_second".into(),
        if backtest_seconds > 0.0 {
            json!(success as f64 / backtest_seconds)
        } else {
            Value::Null
        },
    );
    report.insert(
        "memory".into(),
        json!({
            "parent_peak_rss_mib": peak_rss_mib(libc::RUSAGE_SELF),
            "children_reported_peak_rss_mib": peak_rss_mib(libc::RUSAGE_CHILDREN),
            "scope": "Process-lifetime ru_maxrss counters; factor workers are threads of the parent process, so the parent counter covers them. macOS raw bytes and Linux raw KiB converted to MiB",
        }),
    );
    if let Value::Object(env) = environment() {
        for (key, value) in env {
            report.insert(key, value);
        }
    }
    let target = output.join("benchmark.json");
    fs::write(
        &target,
        pyjson::dumps(
            &Value::Object(report),
            Options {
                sort_keys: false,
                indent: Some(2),
                ensure_ascii: true,
            },
        ) + "\n",
    )?;
    println!(
        "{}",
        pyjson::dumps(
            &json!({
                "benchmark": target.display().to_string(),
                "elapsed": elapsed,
                "successful_factors": success,
                "failed_factors": failed,
            }),
            Options {
                sort_keys: false,
                indent: Some(2),
                ensure_ascii: true
            }
        )
    );
    Ok(if failed > 0 { 1 } else { 0 })
}

// ---------------------------------------------------------------------------
// README charts
// ---------------------------------------------------------------------------

const BLUE: RGBColor = RGBColor(0x38, 0x6C, 0xB0);
const GRAY: RGBColor = RGBColor(0x8B, 0x92, 0x9C);
const AMBER: RGBColor = RGBColor(0xC8, 0x84, 0x3B);
const INK: RGBColor = RGBColor(0x28, 0x34, 0x44);
const MUTED: RGBColor = RGBColor(0x74, 0x7E, 0x8D);
const GRID: RGBColor = RGBColor(0xE9, 0xED, 0xF2);
const ZERO: RGBColor = RGBColor(0xB8, 0xC0, 0xCB);
const SHADE: RGBColor = RGBColor(0xF2, 0xF4, 0xF7);

/// One successful factor's accounts and recorded configuration.
struct Report {
    factor: String,
    pnl: table::Frame,
    layers: table::Frame,
    ic: table::Frame,
    config: Value,
}

fn load_report(directory: &Path, factor: &str) -> Result<Report> {
    let summary = table::read_csv_text(&directory.join("summary.csv"))?;
    let factors = summary.col("factor")?.to_strings();
    let statuses = summary.col("status")?.to_strings();
    let outputs = summary.col("output_directory")?.to_strings();
    let selected: Vec<usize> = (0..summary.len())
        .filter(|&i| {
            factors[i].as_deref() == Some(factor) && statuses[i].as_deref() == Some("success")
        })
        .collect();
    if selected.len() != 1 {
        bail!(
            "Expected one successful {factor} run: {}",
            directory.display()
        );
    }
    let report = PathBuf::from(outputs[selected[0]].clone().unwrap_or_default());
    if !report.exists() {
        bail!(crate::errors::Error::FileNotFound(
            report.display().to_string()
        ));
    }
    let metadata: Value = serde_json::from_str(&fs::read_to_string(report.join("run.json"))?)?;
    Ok(Report {
        factor: factor.to_string(),
        pnl: table::read_parquet_file(&report.join("pnl.parquet"), None)?,
        layers: table::read_parquet_file(&report.join("layer_pnl.parquet"), None)?,
        ic: table::read_parquet_file(&report.join("ic_intraday.parquet"), None)?,
        config: metadata["config"].clone(),
    })
}

fn instants(frame: &table::Frame) -> Result<Vec<i64>> {
    Ok(table::column_instants(frame.col("timestamp")?, "timestamp", timeutil::TIMEZONE)?.0)
}

/// Each trading day's last observation of the given columns.
fn daily_last(
    frame: &table::Frame,
    columns: &[&str],
) -> Result<(Vec<chrono::NaiveDate>, Vec<Vec<f64>>)> {
    let times = instants(frame)?;
    let values: Vec<Vec<f64>> = columns
        .iter()
        .map(|c| frame.col(c).and_then(|col| col.to_f64(c)))
        .collect::<Result<_>>()?;
    let mut days: BTreeMap<chrono::NaiveDate, Vec<f64>> = BTreeMap::new();
    for (row, time) in times.iter().enumerate() {
        let slot = days
            .entry(timeutil::local_date(*time))
            .or_insert_with(|| vec![f64::NAN; columns.len()]);
        for (k, series) in values.iter().enumerate() {
            if !series[row].is_nan() {
                slot[k] = series[row];
            }
        }
    }
    let dates: Vec<chrono::NaiveDate> = days.keys().copied().collect();
    let series = (0..columns.len())
        .map(|k| days.values().map(|v| v[k]).collect())
        .collect();
    Ok((dates, series))
}

fn layer_daily(frame: &table::Frame) -> Result<(Vec<chrono::NaiveDate>, Vec<i64>, Vec<Vec<f64>>)> {
    let times = instants(frame)?;
    let navs = frame.col("net_nav")?.to_f64("net_nav")?;
    let layers: Vec<i64> = frame
        .col("layer")?
        .to_f64("layer")?
        .iter()
        .map(|v| *v as i64)
        .collect();
    let mut table: BTreeMap<chrono::NaiveDate, BTreeMap<i64, f64>> = BTreeMap::new();
    for row in 0..times.len() {
        let slot = table.entry(timeutil::local_date(times[row])).or_default();
        if !navs[row].is_nan() || !slot.contains_key(&layers[row]) {
            slot.insert(layers[row], navs[row]);
        }
    }
    let mut ids = layers.clone();
    ids.sort_unstable();
    ids.dedup();
    let dates: Vec<chrono::NaiveDate> = table.keys().copied().collect();
    let series = ids
        .iter()
        .map(|id| {
            table
                .values()
                .map(|m| *m.get(id).unwrap_or(&f64::NAN))
                .collect()
        })
        .collect();
    Ok((dates, ids, series))
}

fn plot_err<E: std::fmt::Debug>(error: E) -> anyhow::Error {
    anyhow!("Plot rendering failed: {error:?}")
}

fn tick_positions(count: usize) -> Vec<usize> {
    // np.unique(np.linspace(0, n - 1, 5).astype(int))
    let mut positions: Vec<usize> = (0..5)
        .map(|i| {
            if count <= 1 {
                0
            } else {
                ((count - 1) as f64 * i as f64 / 4.0) as usize
            }
        })
        .collect();
    positions.dedup();
    positions
}

fn nice_range(values: impl Iterator<Item = f64>, include_zero: bool) -> (f64, f64) {
    let mut low = f64::INFINITY;
    let mut high = f64::NEG_INFINITY;
    for v in values.filter(|v| v.is_finite()) {
        low = low.min(v);
        high = high.max(v);
    }
    if include_zero {
        low = low.min(0.0);
        high = high.max(0.0);
    }
    if !low.is_finite() {
        return (-0.01, 0.01);
    }
    let span = if high > low { high - low } else { 0.01 };
    (low - span * 0.06, high + span * 0.06)
}

fn header(
    root: &DrawingArea<BitMapBackend, Shift>,
    title: &str,
    subtitle: &str,
    note: &str,
) -> Result<()> {
    let (width, height) = root.dim_in_pixel();
    let x = (width as f64 * 0.065) as i32;
    root.draw(&Text::new(
        title.to_string(),
        (x, (height as f64 * 0.045) as i32),
        ("sans-serif", 40)
            .into_font()
            .style(FontStyle::Bold)
            .color(&INK),
    ))
    .map_err(plot_err)?;
    root.draw(&Text::new(
        subtitle.to_string(),
        (x, (height as f64 * 0.113) as i32),
        ("sans-serif", 24).into_font().color(&MUTED),
    ))
    .map_err(plot_err)?;
    root.draw(&Text::new(
        note.to_string(),
        (x, (height as f64 * 0.94) as i32),
        ("sans-serif", 20).into_font().color(&MUTED),
    ))
    .map_err(plot_err)?;
    Ok(())
}

fn percent(value: f64, decimals: usize) -> String {
    format!("{:.*}%", decimals, value * 100.0)
}

#[allow(clippy::too_many_arguments)]
fn line_panel(
    area: &DrawingArea<BitMapBackend, Shift>,
    title: &str,
    dates: &[chrono::NaiveDate],
    series: &[(Vec<f64>, RGBColor, u32, Option<String>, bool)],
    y_label: &str,
    legend_upper: bool,
) -> Result<()> {
    let n = dates.len().max(1);
    let (low, high) = nice_range(series.iter().flat_map(|s| s.0.iter().copied()), true);
    let margin = (n as f64 - 1.0).max(1.0) * 0.025;
    let mut chart = ChartBuilder::on(area)
        .caption(title, ("sans-serif", 26).into_font().color(&INK))
        .margin_top(10)
        .margin_right(20)
        .x_label_area_size(70)
        .y_label_area_size(110)
        .build_cartesian_2d(-margin..(n as f64 - 1.0) + margin, low..high)
        .map_err(plot_err)?;
    let ticks = tick_positions(dates.len());
    let labels: Vec<String> = dates
        .iter()
        .map(|d| d.format("%b %d").to_string())
        .collect();
    chart
        .configure_mesh()
        .disable_x_mesh()
        .y_labels(6)
        .x_labels(n.min(60))
        .x_label_formatter(&|x| {
            let i = x.round();
            if (x - i).abs() < 1e-6 && i >= 0.0 && ticks.contains(&(i as usize)) {
                labels[i as usize].clone()
            } else {
                String::new()
            }
        })
        .y_label_formatter(&|y| percent(*y, 1))
        .y_desc(y_label)
        .x_desc("Trading date")
        .axis_desc_style(("sans-serif", 22).into_font().color(&MUTED))
        .label_style(("sans-serif", 20).into_font().color(&MUTED))
        .bold_line_style(GRID.stroke_width(1))
        .light_line_style(WHITE.mix(0.0))
        .axis_style(RGBColor(0xD4, 0xDA, 0xE2))
        .draw()
        .map_err(plot_err)?;
    chart
        .draw_series(LineSeries::new(
            [(-margin, 0.0), (n as f64 - 1.0 + margin, 0.0)],
            ZERO.stroke_width(1),
        ))
        .map_err(plot_err)?;
    for (values, color, width, label, dashed) in series {
        let points: Vec<(f64, f64)> = values
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .map(|(i, v)| (i as f64, *v))
            .collect();
        let style = color.stroke_width(*width);
        let drawn = if *dashed {
            chart
                .draw_series(DashedLineSeries::new(points, 10, 7, style))
                .map_err(plot_err)?
        } else {
            chart
                .draw_series(LineSeries::new(points, style))
                .map_err(plot_err)?
        };
        if let Some(label) = label {
            let color = *color;
            let width = *width;
            drawn.label(label.clone()).legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 26, y)], color.stroke_width(width))
            });
        }
    }
    chart
        .configure_series_labels()
        .label_font(("sans-serif", 20).into_font().color(&INK))
        .background_style(WHITE.mix(0.0))
        .border_style(WHITE.mix(0.0))
        .position(if legend_upper {
            SeriesLabelPosition::UpperLeft
        } else {
            SeriesLabelPosition::LowerLeft
        })
        .draw()
        .map_err(plot_err)?;
    Ok(())
}

fn number(value: &Value) -> f64 {
    value.as_f64().unwrap_or(f64::NAN)
}

fn general_format(value: f64) -> String {
    // Python's ``{:g}``.
    let text = format!("{value}");
    if text.ends_with(".0") {
        text[..text.len() - 2].to_string()
    } else {
        text
    }
}

fn plot_accounts(reports: &[(&str, &Report)], output: &Path) -> Result<()> {
    let path = output.join("demo_account_nav.png");
    let root = BitMapBackend::new(&path, (2160, 864)).into_drawing_area();
    root.fill(&WHITE).map_err(plot_err)?;
    let body = root.margin(
        (864.0 * 0.26) as u32,
        (864.0 * 0.19) as u32,
        (2160.0 * 0.065) as u32,
        (2160.0 * 0.025) as u32,
    );
    let panels = body.split_evenly((1, 2));
    for ((mode, report), area) in reports.iter().zip(panels.iter()) {
        let (dates, series) = daily_last(&report.pnl, &["net_nav", "benchmark_nav"])?;
        let capital = number(&report.config["initial_capital"]);
        let to_return = |v: &Vec<f64>| v.iter().map(|x| x / capital - 1.0).collect::<Vec<_>>();
        let frequency = if *mode == "t0" {
            "5-minute bars"
        } else {
            "Daily bars"
        };
        line_panel(
            &area.margin(0, 0, 0, 60),
            &format!(
                "{}  /  {}  /  {} sessions",
                mode.to_uppercase(),
                frequency,
                dates.len()
            ),
            &dates,
            &[
                (
                    to_return(&series[0]),
                    BLUE,
                    4,
                    Some("Strategy (net)".into()),
                    false,
                ),
                (
                    to_return(&series[1]),
                    GRAY,
                    3,
                    Some("Synthetic benchmark".into()),
                    true,
                ),
            ],
            "Cumulative return",
            true,
        )?;
    }
    let config = &reports[0].1.config;
    header(
        &root,
        "Strategy vs. benchmark",
        &format!(
            "{}  /  keep = {:.2}  /  Initial capital: HKD {}m",
            reports[0].1.factor,
            number(&config["keep"]),
            general_format(number(&config["initial_capital"]) / 1e6)
        ),
        "Daily closes; returns relative to initial capital. Synthetic data, 20 stocks, seed 101. Sample windows differ.",
    )?;
    root.present().map_err(plot_err)?;
    Ok(())
}

fn colormap(n: usize) -> Vec<RGBColor> {
    // plt.colormaps["viridis"](np.linspace(0.12, 0.82, n))
    (0..n)
        .map(|i| {
            let x = if n == 1 {
                0.12
            } else {
                0.12 + (0.82 - 0.12) * i as f64 / (n - 1) as f64
            };
            let index = ((x * 256.0) as usize).min(255);
            let (r, g, b) = VIRIDIS[index];
            RGBColor(r, g, b)
        })
        .collect()
}

fn plot_layers(report: &Report, output: &Path, mode: &str) -> Result<()> {
    let path = output.join(format!("demo_{mode}_layers.png"));
    let (dates, ids, navs) = layer_daily(&report.layers)?;
    let capital = number(&report.config["initial_capital"]);
    let returns: Vec<Vec<f64>> = navs
        .iter()
        .map(|s| s.iter().map(|v| v / capital - 1.0).collect())
        .collect();
    let colors = colormap(ids.len());
    let root = BitMapBackend::new(&path, (2160, 864)).into_drawing_area();
    root.fill(&WHITE).map_err(plot_err)?;
    let body = root.margin(
        (864.0 * 0.26) as u32,
        (864.0 * 0.19) as u32,
        (2160.0 * 0.065) as u32,
        (2160.0 * 0.035) as u32,
    );
    let (width, _) = body.dim_in_pixel();
    let (left, right) = body.split_horizontally((width as f64 * 3.6 / 4.6 - 40.0) as u32);
    let series: Vec<(Vec<f64>, RGBColor, u32, Option<String>, bool)> = ids
        .iter()
        .enumerate()
        .map(|(k, id)| {
            let endpoint = k == 0 || k + 1 == ids.len();
            let color = if endpoint {
                colors[k]
            } else {
                RGBColor(colors[k].0, colors[k].1, colors[k].2)
            };
            (
                returns[k].clone(),
                color,
                if endpoint { 4 } else { 2 },
                Some(format!("D{id}")),
                false,
            )
        })
        .collect();
    line_panel(
        &left.margin(0, 0, 0, 40),
        "",
        &dates,
        &series,
        "Cumulative net return",
        true,
    )?;
    // Terminal bars.
    let finals: Vec<f64> = returns
        .iter()
        .map(|s| *s.last().unwrap_or(&f64::NAN))
        .collect();
    let scale = finals.iter().fold(
        0.001f64,
        |m, v| if v.is_finite() { m.max(v.abs()) } else { m },
    );
    let min = finals
        .iter()
        .cloned()
        .filter(|v| v.is_finite())
        .fold(0.0, f64::min);
    let max = finals
        .iter()
        .cloned()
        .filter(|v| v.is_finite())
        .fold(0.0, f64::max);
    let x_range = (min - scale * 0.4)..(max + scale * 0.4);
    let n = finals.len();
    let mut chart = ChartBuilder::on(&right.margin(0, 20, 60, 0))
        .caption("Final return", ("sans-serif", 24).into_font().color(&INK))
        .y_label_area_size(60)
        .build_cartesian_2d(x_range, -0.5f64..(n as f64 - 0.5))
        .map_err(plot_err)?;
    chart
        .draw_series(LineSeries::new(
            [(0.0, -0.5), (0.0, n as f64 - 0.5)],
            ZERO.stroke_width(1),
        ))
        .map_err(plot_err)?;
    for (k, value) in finals.iter().enumerate() {
        if !value.is_finite() {
            continue;
        }
        // Inverted y axis: first layer on top.
        let y = (n - 1 - k) as f64;
        let (a, b) = if *value >= 0.0 {
            (0.0, *value)
        } else {
            (*value, 0.0)
        };
        chart
            .draw_series(std::iter::once(Rectangle::new(
                [(a, y - 0.3), (b, y + 0.3)],
                colors[k].mix(0.9).filled(),
            )))
            .map_err(plot_err)?;
        let offset = scale * 0.045 * if *value >= 0.0 { 1.0 } else { -1.0 };
        let label = format!(
            "{}{}",
            if *value >= 0.0 { "+" } else { "" },
            percent(*value, 2)
        );
        let anchor = if *value >= 0.0 {
            plotters::style::text_anchor::Pos::new(
                plotters::style::text_anchor::HPos::Left,
                plotters::style::text_anchor::VPos::Center,
            )
        } else {
            plotters::style::text_anchor::Pos::new(
                plotters::style::text_anchor::HPos::Right,
                plotters::style::text_anchor::VPos::Center,
            )
        };
        chart
            .draw_series(std::iter::once(Text::new(
                label,
                (value + offset, y),
                ("sans-serif", 19).into_font().color(&INK).pos(anchor),
            )))
            .map_err(plot_err)?;
        chart
            .draw_series(std::iter::once(Text::new(
                format!("D{}", ids[k]),
                (x_range_start(&finals, scale), y),
                ("sans-serif", 19).into_font().color(&MUTED).pos(
                    plotters::style::text_anchor::Pos::new(
                        plotters::style::text_anchor::HPos::Right,
                        plotters::style::text_anchor::VPos::Center,
                    ),
                ),
            )))
            .map_err(plot_err)?;
    }
    let frequency = if mode == "t0" {
        "5-minute bars"
    } else {
        "Daily bars"
    };
    let keep = number(&report.config["keep"]);
    header(
        &root,
        &format!("{} decile accounts", mode.to_uppercase()),
        &format!(
            "{}  /  {}  /  {} sessions  /  D1: low signal  ->  D10: high signal",
            report.factor,
            frequency,
            dates.len()
        ),
        &format!(
            "Target: {:.0}% base + {:.0}% decile overlay. Costs and execution constraints included. Synthetic data, seed 101.",
            keep * 100.0,
            (1.0 - keep) * 100.0
        ),
    )?;
    root.present().map_err(plot_err)?;
    Ok(())
}

fn x_range_start(finals: &[f64], scale: f64) -> f64 {
    let min = finals
        .iter()
        .cloned()
        .filter(|v| v.is_finite())
        .fold(0.0, f64::min);
    min - scale * 0.4
}

fn plot_ic(report: &Report, output: &Path) -> Result<()> {
    let path = output.join("demo_t0_ic.png");
    let times = report.ic.col("local_time")?.to_strings();
    let minutes: Vec<i64> = times
        .iter()
        .map(|t| {
            let t = t.as_deref().unwrap_or("00:00");
            t[..2].parse::<i64>().unwrap_or(0) * 60 + t[3..].parse::<i64>().unwrap_or(0)
        })
        .collect();
    let pearson = report.ic.col("pearson_mean")?.to_f64("pearson_mean")?;
    let spearman = report.ic.col("spearman_mean")?.to_f64("spearman_mean")?;
    let counts = report.ic.col("pearson_count")?.to_f64("pearson_count")?;
    let root = BitMapBackend::new(&path, (2160, 864)).into_drawing_area();
    root.fill(&WHITE).map_err(plot_err)?;
    let body = root.margin(
        (864.0 * 0.26) as u32,
        (864.0 * 0.19) as u32,
        (2160.0 * 0.065) as u32,
        (2160.0 * 0.025) as u32,
    );
    let (low, high) = nice_range(pearson.iter().chain(&spearman).copied(), true);
    let mut chart = ChartBuilder::on(&body)
        .x_label_area_size(70)
        .y_label_area_size(110)
        .build_cartesian_2d(565f64..960f64, low..high)
        .map_err(plot_err)?;
    chart
        .draw_series(std::iter::once(Rectangle::new(
            [(720.0, low), (780.0, high)],
            SHADE.filled(),
        )))
        .map_err(plot_err)?;
    let ticks = [570i64, 630, 690, 780, 840, 900, 960];
    chart
        .configure_mesh()
        .disable_x_mesh()
        .y_labels(6)
        .x_labels(80)
        .x_label_formatter(&|x| {
            let v = x.round() as i64;
            if (x - x.round()).abs() < 1e-6 && ticks.contains(&v) {
                format!("{:02}:{:02}", v / 60, v % 60)
            } else {
                String::new()
            }
        })
        .y_label_formatter(&|y| format!("{y:.2}"))
        .y_desc("Mean IC")
        .x_desc("Entry time (Hong Kong)")
        .axis_desc_style(("sans-serif", 22).into_font().color(&MUTED))
        .label_style(("sans-serif", 20).into_font().color(&MUTED))
        .bold_line_style(GRID.stroke_width(1))
        .light_line_style(WHITE.mix(0.0))
        .draw()
        .map_err(plot_err)?;
    chart
        .draw_series(LineSeries::new(
            [(565.0, 0.0), (960.0, 0.0)],
            ZERO.stroke_width(1),
        ))
        .map_err(plot_err)?;
    chart
        .draw_series(std::iter::once(Text::new(
            "Lunch break".to_string(),
            (750.0, low + (high - low) * 0.95),
            ("sans-serif", 19).into_font().color(&MUTED).pos(
                plotters::style::text_anchor::Pos::new(
                    plotters::style::text_anchor::HPos::Center,
                    plotters::style::text_anchor::VPos::Top,
                ),
            ),
        )))
        .map_err(plot_err)?;
    for (values, color, label) in [
        (&pearson, BLUE, "Pearson IC"),
        (&spearman, AMBER, "Spearman IC"),
    ] {
        for (index, morning) in [true, false].into_iter().enumerate() {
            let points: Vec<(f64, f64)> = minutes
                .iter()
                .zip(values.iter())
                .filter(|(m, v)| (if morning { **m < 720 } else { **m >= 780 }) && v.is_finite())
                .map(|(m, v)| (*m as f64, *v))
                .collect();
            let drawn = chart
                .draw_series(LineSeries::new(points, color.stroke_width(3)))
                .map_err(plot_err)?;
            if index == 0 {
                drawn.label(label).legend(move |(x, y)| {
                    PathElement::new(vec![(x, y), (x + 26, y)], color.stroke_width(3))
                });
            }
        }
    }
    chart
        .configure_series_labels()
        .label_font(("sans-serif", 20).into_font().color(&INK))
        .background_style(WHITE.mix(0.0))
        .border_style(WHITE.mix(0.0))
        .position(SeriesLabelPosition::UpperLeft)
        .draw()
        .map_err(plot_err)?;
    let finite_counts: Vec<f64> = counts.iter().copied().filter(|c| c.is_finite()).collect();
    let min_count = finite_counts.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_count = finite_counts
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    header(
        &root,
        "Intraday information coefficient",
        &format!(
            "{}  /  5-minute bars  /  {}-{} observations per time slot",
            report.factor, min_count as i64, max_count as i64
        ),
        "Prior completed-bar signal vs. next open-to-close return. No overnight windows. Synthetic data, seed 101.",
    )?;
    root.present().map_err(plot_err)?;
    Ok(())
}

/// Render all README figures from two completed synthetic runs.
///
/// The inputs are task directories containing ``summary.csv``; their paths
/// must still resolve to the factor reports created by ``hkfactor run``.
pub fn plot_readme(t0: &Path, t1: &Path, factor: &str, output: &Path) -> Result<()> {
    reporting::register_fonts();
    let first = load_report(t0, factor)?;
    let second = load_report(t1, factor)?;
    fs::create_dir_all(output)?;
    plot_accounts(&[("t0", &first), ("t1", &second)], output)?;
    plot_layers(&first, output, "t0")?;
    plot_layers(&second, output, "t1")?;
    plot_ic(&first, output)?;
    Ok(())
}
