//! Persist account results and explicitly labelled research diagnostics.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::Once;

use anyhow::{anyhow, Result};
use plotters::prelude::*;
use plotters::style::FontStyle;
use serde_json::{Map, Value};

use crate::frames::OutFrame;
use crate::models::{AnalysisResult, BacktestResult};
use crate::timeutil;

static FONT: Once = Once::new();
const FONT_BYTES: &[u8] = include_bytes!("../assets/DejaVuSans.ttf");

pub fn register_fonts() {
    FONT.call_once(|| {
        let _ = plotters::style::register_font("sans-serif", FontStyle::Normal, FONT_BYTES);
        let _ = plotters::style::register_font("sans-serif", FontStyle::Bold, FONT_BYTES);
    });
}

/// Replace non-finite floats by null, recursively (strict JSON).
pub fn json_safe(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            Value::Object(map.into_iter().map(|(k, v)| (k, json_safe(v))).collect())
        }
        Value::Array(items) => Value::Array(items.into_iter().map(json_safe).collect()),
        other => other,
    }
}

/// Write readable UTF-8 metadata with strict JSON scalars.
pub fn save_json(path: &Path, values: &Value) -> Result<()> {
    let text = crate::pyjson::dumps(
        &json_safe(values.clone()),
        crate::pyjson::Options {
            sort_keys: false,
            indent: Some(2),
            ensure_ascii: false,
        },
    );
    fs::write(path, text + "\n")?;
    Ok(())
}

/// Matplotlib's default colour cycle.
pub const TAB10: [RGBColor; 10] = [
    RGBColor(0x1f, 0x77, 0xb4),
    RGBColor(0xff, 0x7f, 0x0e),
    RGBColor(0x2c, 0xa0, 0x2c),
    RGBColor(0xd6, 0x27, 0x28),
    RGBColor(0x94, 0x67, 0xbd),
    RGBColor(0x8c, 0x56, 0x4b),
    RGBColor(0xe3, 0x77, 0xc2),
    RGBColor(0x7f, 0x7f, 0x7f),
    RGBColor(0xbc, 0xbd, 0x22),
    RGBColor(0x17, 0xbe, 0xcf),
];

fn plot_error<E: std::fmt::Debug>(error: E) -> anyhow::Error {
    anyhow!("Plot rendering failed: {error:?}")
}

fn padded_range(values: impl Iterator<Item = f64>) -> (f64, f64) {
    let finite: Vec<f64> = values.filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return (-1.0, 1.0);
    }
    let low = finite.iter().cloned().fold(f64::INFINITY, f64::min);
    let high = finite.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let span = if high > low {
        high - low
    } else {
        high.abs().max(1.0) * 0.1
    };
    (low - span * 0.05, high + span * 0.05)
}

fn empty_figure(
    path: &Path,
    title: &str,
    x_label: &str,
    y_label: &str,
    message: &str,
) -> Result<()> {
    let root = BitMapBackend::new(path, (1650, 750)).into_drawing_area();
    root.fill(&WHITE).map_err(plot_error)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 26))
        .margin(30)
        .x_label_area_size(60)
        .y_label_area_size(90)
        .build_cartesian_2d(0f64..1f64, 0f64..1f64)
        .map_err(plot_error)?;
    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .label_style(("sans-serif", 18))
        .light_line_style(WHITE.mix(0.0))
        .draw()
        .map_err(plot_error)?;
    chart
        .draw_series(std::iter::once(Text::new(
            message.to_string(),
            (0.42, 0.5),
            ("sans-serif", 22),
        )))
        .map_err(plot_error)?;
    root.present().map_err(plot_error)?;
    Ok(())
}

/// Plot local-time average IC, separately from strategy account PnL.
fn plot_intraday(frame: &OutFrame, output_dir: &Path) -> Result<()> {
    let path = output_dir.join("ic_intraday.png");
    let title = "Research IC by intraday entry time";
    if frame.is_empty() {
        return empty_figure(
            &path,
            title,
            "Asia/Hong_Kong local time",
            "Mean IC",
            "No usable IC observations",
        );
    }
    let times = frame.col("local_time").strs();
    let pearson = frame.col("pearson_mean").f64s();
    let spearman = frame.col("spearman_mean").f64s();
    let n = times.len();
    let (mut low, mut high) = padded_range(pearson.iter().chain(spearman).copied().chain([0.0]));
    if low == high {
        low -= 1.0;
        high += 1.0;
    }
    let root = BitMapBackend::new(&path, (1650, 750)).into_drawing_area();
    root.fill(&WHITE).map_err(plot_error)?;
    let x_max = if n > 1 { (n - 1) as f64 } else { 1.0 };
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 26))
        .margin(30)
        .x_label_area_size(80)
        .y_label_area_size(100)
        .build_cartesian_2d(-0.05 * x_max..x_max * 1.05, low..high)
        .map_err(plot_error)?;
    let stride = n.div_ceil(16).max(1);
    let labels: Vec<String> = times.to_vec();
    chart
        .configure_mesh()
        .x_desc("Asia/Hong_Kong local time")
        .y_desc("Mean IC")
        .label_style(("sans-serif", 18))
        .x_labels(n.min(40))
        .x_label_formatter(&|x| {
            let i = x.round();
            if (x - i).abs() < 1e-6
                && i >= 0.0
                && (i as usize) < labels.len()
                && (i as usize).is_multiple_of(stride)
            {
                labels[i as usize].clone()
            } else {
                String::new()
            }
        })
        .light_line_style(WHITE.mix(0.0))
        .bold_line_style(BLACK.mix(0.2))
        .draw()
        .map_err(plot_error)?;
    chart
        .draw_series(LineSeries::new(
            [(-0.05 * x_max, 0.0), (x_max * 1.05, 0.0)],
            RGBColor(128, 128, 128).stroke_width(1),
        ))
        .map_err(plot_error)?;
    for (index, (name, series)) in [("Pearson IC", pearson), ("Spearman IC", spearman)]
        .into_iter()
        .enumerate()
    {
        let color = TAB10[index];
        let points: Vec<(f64, f64)> = series
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .map(|(i, v)| (i as f64, *v))
            .collect();
        chart
            .draw_series(LineSeries::new(points.clone(), color.stroke_width(2)))
            .map_err(plot_error)?
            .label(name)
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 24, y)], color.stroke_width(2))
            });
        chart
            .draw_series(points.iter().map(|p| Circle::new(*p, 3, color.filled())))
            .map_err(plot_error)?;
    }
    chart
        .configure_series_labels()
        .label_font(("sans-serif", 18))
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK.mix(0.3))
        .draw()
        .map_err(plot_error)?;
    root.present().map_err(plot_error)?;
    Ok(())
}

/// Plot separately executed layer account NAV by local trading date.
fn plot_layers(frame: &OutFrame, output_dir: &Path) -> Result<()> {
    let path = output_dir.join("layer_pnl.png");
    let title = "Executed layer accounts: net account value";
    if frame.is_empty() {
        return empty_figure(
            &path,
            title,
            "Hong Kong trading date",
            "Net NAV (HKD)",
            "No executed layer accounts",
        );
    }
    let nav_column = if frame.get("net_nav").is_some() {
        "net_nav"
    } else {
        "nav"
    };
    let navs = frame
        .get(nav_column)
        .ok_or_else(|| anyhow!("Layer account PnL requires a nav column"))?
        .f64s();
    let timestamps = frame.col("timestamp").i64s();
    let layers = frame.col("layer").i64s();
    // groupby(["date", "layer"]).last(): last non-null NAV per day and layer.
    let mut daily: BTreeMap<String, BTreeMap<i64, f64>> = BTreeMap::new();
    for i in 0..frame.len() {
        let date = timeutil::strftime_hk(timestamps[i], "%Y-%m-%d");
        let slot = daily.entry(date).or_default();
        if navs[i].is_finite() || !slot.contains_key(&layers[i]) {
            if navs[i].is_finite() {
                slot.insert(layers[i], navs[i]);
            } else {
                slot.entry(layers[i]).or_insert(f64::NAN);
            }
        }
    }
    let dates: Vec<String> = daily.keys().cloned().collect();
    let layer_ids: Vec<i64> = {
        let mut ids: Vec<i64> = layers.to_vec();
        ids.sort_unstable();
        ids.dedup();
        ids
    };
    let (low, high) = padded_range(daily.values().flat_map(|m| m.values().copied()));
    let root = BitMapBackend::new(&path, (1650, 750)).into_drawing_area();
    root.fill(&WHITE).map_err(plot_error)?;
    let n = dates.len();
    let x_range = if n == 1 {
        -0.5..0.5
    } else {
        -0.02 * (n - 1) as f64..(n - 1) as f64 * 1.02
    };
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 26))
        .margin(30)
        .x_label_area_size(90)
        .y_label_area_size(130)
        .build_cartesian_2d(x_range, low..high)
        .map_err(plot_error)?;
    let stride = n.div_ceil(10).max(1);
    chart
        .configure_mesh()
        .x_desc("Hong Kong trading date")
        .y_desc("Net NAV (HKD)")
        .label_style(("sans-serif", 18))
        .x_labels(n.min(40))
        .x_label_formatter(&|x| {
            let i = x.round();
            if (x - i).abs() < 1e-6
                && i >= 0.0
                && (i as usize) < dates.len()
                && (i as usize).is_multiple_of(stride)
            {
                dates[i as usize].clone()
            } else {
                String::new()
            }
        })
        .y_label_formatter(&|y| format!("{y:.0}"))
        .light_line_style(WHITE.mix(0.0))
        .bold_line_style(BLACK.mix(0.2))
        .draw()
        .map_err(plot_error)?;
    for (index, layer) in layer_ids.iter().enumerate() {
        let color = TAB10[index % TAB10.len()];
        let points: Vec<(f64, f64)> = dates
            .iter()
            .enumerate()
            .filter_map(|(i, d)| {
                daily[d]
                    .get(layer)
                    .filter(|v| v.is_finite())
                    .map(|v| (i as f64, *v))
            })
            .collect();
        chart
            .draw_series(LineSeries::new(points.clone(), color.stroke_width(2)))
            .map_err(plot_error)?
            .label(format!("Layer {layer}"))
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 24, y)], color.stroke_width(2))
            });
        chart
            .draw_series(points.iter().map(|p| Circle::new(*p, 2, color.filled())))
            .map_err(plot_error)?;
    }
    chart
        .configure_series_labels()
        .label_font(("sans-serif", 16))
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK.mix(0.3))
        .position(SeriesLabelPosition::UpperLeft)
        .draw()
        .map_err(plot_error)?;
    root.present().map_err(plot_error)?;
    Ok(())
}

/// Write machine-readable results and optional research/account plots.
///
/// The batch runner supplies a staging directory and atomically commits it
/// after this function completes. IC and research forward layers are gross
/// diagnostics; ``pnl`` and ``layer_pnl`` are engine accounts that include
/// configured execution costs and constraints.
pub fn write_report(
    result: &BacktestResult,
    analysis: &AnalysisResult,
    output_dir: &Path,
    metadata: Option<Map<String, Value>>,
    plots: bool,
) -> Result<()> {
    fs::create_dir_all(output_dir)?;
    let mut frames: Vec<(&str, OutFrame)> = vec![
        ("pnl", result.pnl.clone()),
        ("layers", analysis.layers.clone()),
        ("ic", analysis.ic.clone()),
        ("ic_intraday", analysis.ic_intraday.clone()),
        ("layer_summary", analysis.layer_summary.clone()),
    ];
    if !result.audit_streamed {
        frames.push(("trades", result.trades.to_frame()));
        frames.push(("restrictions", result.restrictions.to_frame()));
    }
    if let Some(layer_pnl) = &analysis.layer_pnl {
        frames.push(("layer_pnl", layer_pnl.clone()));
    }
    for (name, frame) in &frames {
        frame.write_parquet(&output_dir.join(format!("{name}.parquet")))?;
    }
    save_json(
        &output_dir.join("metrics.json"),
        &Value::Object(result.metrics.clone()),
    )?;
    let mut run = metadata.unwrap_or_default();
    run.insert(
        "analysis_assumptions".into(),
        Value::Object(analysis.assumptions.clone()),
    );
    let mut outputs = Map::new();
    outputs.insert(
        "pnl".into(),
        Value::String("Executed account; configured costs and constraints".into()),
    );
    outputs.insert(
        "layers".into(),
        Value::String("Gross forward research returns, not executed PnL".into()),
    );
    outputs.insert(
        "ic".into(),
        Value::String("Previous-bar signals versus next-open forward returns".into()),
    );
    outputs.insert(
        "layer_pnl".into(),
        Value::String(
            if analysis.layer_pnl.is_some() {
                "Independent executed accounts; same keep and costs"
            } else {
                "Not requested"
            }
            .into(),
        ),
    );
    run.insert("outputs".into(), Value::Object(outputs));
    save_json(&output_dir.join("run.json"), &Value::Object(run))?;
    if plots {
        register_fonts();
        plot_intraday(&analysis.ic_intraday, output_dir)?;
        if let Some(layer_pnl) = &analysis.layer_pnl {
            plot_layers(layer_pnl, output_dir)?;
        }
    }
    Ok(())
}
