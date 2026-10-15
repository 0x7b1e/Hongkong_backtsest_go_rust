//! Cross-sectional research diagnostics separate from execution accounts.

use std::collections::BTreeMap;

use anyhow::{bail, Result};
use serde_json::{Map, Value};

use crate::config::{BacktestConfig, Mode};
use crate::frames::{OutCol, OutFrame};
use crate::matrix::{self, Mat};
use crate::models::{AnalysisResult, MarketPanel, SignalMatrix};
use crate::timeutil;

const CHUNK_SIZE: usize = 4096;

/// Assign ascending groups with deterministic symbol ordering for ties.
///
/// ``values`` and ``eligible`` are row-major ``(rows, cols)`` blocks.
fn groups(
    values: &[f64],
    eligible: &[bool],
    rows: usize,
    cols: usize,
    quantiles: i64,
    symbol_order: &[usize],
) -> Vec<i64> {
    let mut out = vec![0i64; rows * cols];
    for r in 0..rows {
        let row_values = &values[r * cols..(r + 1) * cols];
        let row_eligible = &eligible[r * cols..(r + 1) * cols];
        let count = row_eligible.iter().filter(|e| **e).count() as i64;
        // Ordered by symbol; ineligible entries sort last as +inf.
        let ordered: Vec<f64> = symbol_order
            .iter()
            .map(|&c| {
                if row_eligible[c] {
                    row_values[c]
                } else {
                    f64::INFINITY
                }
            })
            .collect();
        let mut sorted: Vec<usize> = (0..cols).collect();
        sorted.sort_by(|&a, &b| {
            ordered[a]
                .partial_cmp(&ordered[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut ranks = vec![0i64; cols];
        for (rank, &position) in sorted.iter().enumerate() {
            ranks[position] = rank as i64;
        }
        for (position, &c) in symbol_order.iter().enumerate() {
            let mut label = ranks[position] * quantiles / count.max(1) + 1;
            if !row_eligible[c] || count < quantiles {
                label = 0;
            }
            out[r * cols + c] = label;
        }
    }
    out
}

fn symbol_order(panel: &MarketPanel) -> Vec<usize> {
    let mut order: Vec<usize> = (0..panel.symbols.len()).collect();
    order.sort_by(|&a, &b| panel.symbols[a].cmp(&panel.symbols[b]));
    order
}

/// Yield each layer's binary signals for separate execution accounts.
///
/// Groups use finite signals and membership known at each completed bar.
/// Layer 1 has the lowest raw factor values. Equal values are ordered by
/// symbol. Bars with fewer than ``quantiles`` members produce no signals.
/// The execution engine subsequently applies its own signal delay, ``keep``
/// floor, costs, and trading constraints. ``visit`` receives one layer at a
/// time so only one layer matrix is allocated.
pub fn for_each_layer_signal<F>(
    panel: &MarketPanel,
    signal: &SignalMatrix,
    config: &BacktestConfig,
    mut visit: F,
) -> Result<()>
where
    F: FnMut(i64, SignalMatrix) -> Result<()>,
{
    validate_shapes(panel, signal)?;
    let (rows, cols) = panel.shape();
    let order = symbol_order(panel);
    let mut labels = vec![0i64; rows * cols];
    let mut start = 0;
    while start < rows {
        let stop = (start + CHUNK_SIZE).min(rows);
        let values = &signal.values.data[start * cols..stop * cols];
        let eligible: Vec<bool> = (start * cols..stop * cols)
            .map(|i| panel.base_weights.data[i] > 0.0 && signal.values.data[i].is_finite())
            .collect();
        let block = groups(
            values,
            &eligible,
            stop - start,
            cols,
            config.quantiles,
            &order,
        );
        labels[start * cols..stop * cols].copy_from_slice(&block);
        start = stop;
    }
    let mut updated = signal.updated.clone();
    for r in 1..rows {
        if panel.base_weights.row(r) != panel.base_weights.row(r - 1) {
            updated[r] = true;
        }
    }
    for layer in 1..=config.quantiles {
        let mut values = Mat::filled(rows, cols, f64::NAN);
        for (i, label) in labels.iter().enumerate() {
            if *label == layer {
                values.data[i] = 1.0;
            }
        }
        visit(
            layer,
            SignalMatrix {
                factor: format!("{}:layer_{layer:02}", signal.factor),
                values,
                updated: updated.clone(),
            },
        )?;
    }
    Ok(())
}

/// Reject accidental factor/market misalignment.
fn validate_shapes(panel: &MarketPanel, signal: &SignalMatrix) -> Result<()> {
    if signal.values.shape() != panel.shape() {
        bail!("Signal and market dimensions do not match");
    }
    if signal.updated.len() != panel.shape().0 {
        bail!("Signal updated flags must have one value per bar");
    }
    Ok(())
}

/// Row-wise Pearson correlation, preserving missing pairs.
fn correlation(values: &[f64], returns: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; rows];
    for r in 0..rows {
        let x_row = &values[r * cols..(r + 1) * cols];
        let y_row = &returns[r * cols..(r + 1) * cols];
        let valid: Vec<bool> = (0..cols)
            .map(|c| x_row[c].is_finite() && y_row[c].is_finite())
            .collect();
        let count = valid.iter().filter(|v| **v).count();
        let denominator = count.max(1) as f64;
        let x_mean =
            matrix::pairwise_sum_map(cols, |c| if valid[c] { x_row[c] } else { 0.0 }) / denominator;
        let y_mean =
            matrix::pairwise_sum_map(cols, |c| if valid[c] { y_row[c] } else { 0.0 }) / denominator;
        let x: Vec<f64> = (0..cols)
            .map(|c| if valid[c] { x_row[c] - x_mean } else { 0.0 })
            .collect();
        let y: Vec<f64> = (0..cols)
            .map(|c| if valid[c] { y_row[c] - y_mean } else { 0.0 })
            .collect();
        let numerator = matrix::pairwise_sum_map(cols, |c| x[c] * y[c]);
        let scale = (matrix::pairwise_sum_map(cols, |c| x[c] * x[c])
            * matrix::pairwise_sum_map(cols, |c| y[c] * y[c]))
        .sqrt();
        let mut result = numerator / scale;
        if count < 2 || scale == 0.0 {
            result = f64::NAN;
        }
        out[r] = if result.is_nan() {
            f64::NAN
        } else {
            result.clamp(-1.0, 1.0)
        };
    }
    out
}

/// Entry-open to future-close total returns and window validity.
fn forward_returns(
    panel: &MarketPanel,
    entries: &[usize],
    horizon: usize,
    config: &BacktestConfig,
) -> Result<(Vec<f64>, Vec<bool>, Vec<usize>)> {
    let (rows, cols) = panel.shape();
    let t_end = timeutil::parse_iso_date(&config.t_end)?;
    let mut returns = vec![f64::NAN; entries.len() * cols];
    let mut valid_window = vec![false; entries.len()];
    let mut safe_ends = vec![0usize; entries.len()];
    for (k, &entry) in entries.iter().enumerate() {
        let end = entry + horizon - 1;
        let mut valid = end < rows;
        let safe_end = end.min(rows - 1);
        safe_ends[k] = safe_end;
        valid &= timeutil::local_date(panel.timestamps[safe_end]) <= t_end;
        if config.mode == Mode::T0 {
            valid &= panel.session_id[entry] == panel.session_id[safe_end];
        }
        valid_window[k] = valid;
        let mut shares = vec![1.0; cols];
        let mut dividends = vec![0.0; cols];
        for offset in 1..horizon {
            let action_row = (entry + offset).min(rows - 1);
            for c in 0..cols {
                dividends[c] += shares[c] * panel.cash_dividend.get(action_row, c);
                shares[c] *= panel.share_multiplier.get(action_row, c);
            }
        }
        for c in 0..cols {
            let price = panel.open.get(entry, c);
            let exit = panel.close.get(safe_end, c);
            let value = (exit * shares[c] + dividends[c]) / price - 1.0;
            let valid_quote = exit.is_finite() && exit > 0.0 && panel.tradable.get(safe_end, c);
            returns[k * cols + c] = if valid_quote && valid {
                value
            } else {
                f64::NAN
            };
        }
    }
    Ok((returns, valid_window, safe_ends))
}

/// Average ranks within each row, NaN preserved (``DataFrame.rank(axis=1)``).
fn average_ranks(values: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; rows * cols];
    for r in 0..rows {
        let row = &values[r * cols..(r + 1) * cols];
        let mut present: Vec<usize> = (0..cols).filter(|&c| !row[c].is_nan()).collect();
        present.sort_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap());
        let mut i = 0;
        while i < present.len() {
            let mut j = i + 1;
            while j < present.len() && row[present[j]] == row[present[i]] {
                j += 1;
            }
            let rank = (i + 1 + j) as f64 / 2.0;
            for &c in &present[i..j] {
                out[r * cols + c] = rank;
            }
            i = j;
        }
    }
    out
}

/// Count observations belonging to a repeated finite value per row.
fn tie_count(values: &[f64], rows: usize, cols: usize) -> Vec<i64> {
    let mut out = vec![0i64; rows];
    for r in 0..rows {
        let mut ordered: Vec<f64> = values[r * cols..(r + 1) * cols].to_vec();
        ordered.sort_by(|a, b| match (a.is_nan(), b.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            _ => a.partial_cmp(b).unwrap(),
        });
        let mut tied = vec![false; cols];
        for i in 1..cols {
            if ordered[i] == ordered[i - 1] && ordered[i].is_finite() {
                tied[i] = true;
                tied[i - 1] = true;
            }
        }
        out[r] = tied.iter().filter(|t| **t).count() as i64;
    }
    out
}

struct IcColumns {
    timestamp: Vec<i64>,
    signal_timestamp: Vec<i64>,
    forward_end: Vec<i64>,
    entry_n_assets: Vec<i64>,
    n_assets: Vec<i64>,
    tied_assets: Vec<i64>,
    pearson_ic: Vec<f64>,
    spearman_ic: Vec<f64>,
    reason: Vec<String>,
}

#[allow(clippy::too_many_arguments)]
fn ic_rows(
    panel: &MarketPanel,
    entries: &[usize],
    values: &[f64],
    eligible: &[bool],
    returns: &[f64],
    valid_window: &[bool],
    ends: &[usize],
    config: &BacktestConfig,
    out: &mut IcColumns,
) {
    let cols = panel.symbols.len();
    let rows = entries.len();
    let paired: Vec<bool> = (0..rows * cols)
        .map(|i| eligible[i] && returns[i].is_finite())
        .collect();
    let x: Vec<f64> = (0..rows * cols)
        .map(|i| if paired[i] { values[i] } else { f64::NAN })
        .collect();
    let y: Vec<f64> = (0..rows * cols)
        .map(|i| if paired[i] { returns[i] } else { f64::NAN })
        .collect();
    let mut pearson = correlation(&x, &y, rows, cols);
    let x_rank = average_ranks(&x, rows, cols);
    let y_rank = average_ranks(&y, rows, cols);
    let mut spearman = correlation(&x_rank, &y_rank, rows, cols);
    let masked: Vec<f64> = (0..rows * cols)
        .map(|i| if eligible[i] { values[i] } else { f64::NAN })
        .collect();
    let ties = tie_count(&masked, rows, cols);
    for r in 0..rows {
        let slice = r * cols..(r + 1) * cols;
        let count = paired[slice.clone()].iter().filter(|p| **p).count();
        let extremes = |data: &[f64]| -> bool {
            let mut max = f64::NEG_INFINITY;
            let mut min = f64::INFINITY;
            for c in 0..cols {
                if paired[r * cols + c] {
                    max = max.max(data[r * cols + c]);
                    min = min.min(data[r * cols + c]);
                }
            }
            max == min
        };
        let constant_x = extremes(&x);
        let constant_y = extremes(&y);
        let mut reason = "ok";
        if constant_y {
            reason = "constant_return";
        }
        if constant_x {
            reason = "constant_signal";
        }
        if (count as i64) < config.ic_min_assets {
            reason = "insufficient_assets";
        }
        if !valid_window[r] {
            reason = "invalid_forward_window";
        }
        if reason != "ok" {
            pearson[r] = f64::NAN;
            spearman[r] = f64::NAN;
        }
        out.timestamp.push(panel.open_timestamps[entries[r]]);
        out.signal_timestamp.push(panel.timestamps[entries[r] - 1]);
        out.forward_end.push(panel.timestamps[ends[r]]);
        out.entry_n_assets
            .push(eligible[slice].iter().filter(|e| **e).count() as i64);
        out.n_assets.push(count as i64);
        out.tied_assets.push(ties[r]);
        out.pearson_ic.push(pearson[r]);
        out.spearman_ic.push(spearman[r]);
        out.reason.push(reason.to_string());
    }
}

#[derive(Default)]
struct LayerColumns {
    timestamp: Vec<i64>,
    date: Vec<String>,
    layer: Vec<i64>,
    n_assets: Vec<i64>,
    forward_n_assets: Vec<i64>,
    one_bar_n_assets: Vec<i64>,
    mean_forward_return: Vec<f64>,
    mean_one_bar_return: Vec<f64>,
    gross_research_nav: Vec<f64>,
}

/// Gross research observations without compounding overlaps.
#[allow(clippy::too_many_arguments)]
fn layer_rows(
    panel: &MarketPanel,
    entries: &[usize],
    labels: &[i64],
    returns: &[f64],
    one_bar_returns: &[f64],
    config: &BacktestConfig,
    previous_nav: &mut [f64],
    out: &mut LayerColumns,
) {
    let cols = panel.symbols.len();
    let rows = entries.len();
    for layer in 1..=config.quantiles {
        let previous = previous_nav[(layer - 1) as usize];
        let mut cumulative = 1.0;
        let mut nav = previous;
        for r in 0..rows {
            let mut count = 0usize;
            let mut forward_values = Vec::new();
            let mut one_bar_values = Vec::new();
            for c in 0..cols {
                let i = r * cols + c;
                if labels[i] == layer {
                    count += 1;
                    if returns[i].is_finite() {
                        forward_values.push(returns[i]);
                    }
                    if one_bar_returns[i].is_finite() {
                        one_bar_values.push(one_bar_returns[i]);
                    }
                }
            }
            let forward_mean = mean_full(r, cols, labels, layer, returns, forward_values.len());
            let one_bar_mean = mean_full(
                r,
                cols,
                labels,
                layer,
                one_bar_returns,
                one_bar_values.len(),
            );
            // Complete same-bar baskets form a non-overlapping research curve.
            let curve_return = if count == 0 {
                0.0
            } else if one_bar_values.len() == count {
                one_bar_mean
            } else {
                f64::NAN
            };
            cumulative *= 1.0 + curve_return;
            nav = previous * cumulative;
            if count > 0 {
                let entry = entries[r];
                out.timestamp.push(panel.open_timestamps[entry]);
                out.date.push(timeutil::strftime_hk(
                    panel.open_timestamps[entry],
                    "%Y-%m-%d",
                ));
                out.layer.push(layer);
                out.n_assets.push(count as i64);
                out.forward_n_assets.push(forward_values.len() as i64);
                out.one_bar_n_assets.push(one_bar_values.len() as i64);
                out.mean_forward_return.push(forward_mean);
                out.mean_one_bar_return.push(one_bar_mean);
                out.gross_research_nav.push(nav);
            }
        }
        if rows > 0 {
            previous_nav[(layer - 1) as usize] = nav;
        }
    }
}

/// ``np.where(valid, values, 0).sum(axis=1) / count`` for one layer row.
fn mean_full(
    r: usize,
    cols: usize,
    labels: &[i64],
    layer: i64,
    values: &[f64],
    count: usize,
) -> f64 {
    if count == 0 {
        return f64::NAN;
    }
    let total = matrix::pairwise_sum_map(cols, |c| {
        let i = r * cols + c;
        if labels[i] == layer && values[i].is_finite() {
            values[i]
        } else {
            0.0
        }
    });
    total / count as f64
}

/// Analyze previous-completed-bar signals against future gross returns.
///
/// IC is diagnostic, independent of the execution strategy's holding rules.
/// Returns run from the next bar's open to the horizon's close, include
/// supplied intervening dividends and share multipliers, and exclude
/// cross-session T0 windows. Pearson and average-rank Spearman IC use the
/// same finite point-in-time constituent sample. Missing or untradable
/// entry/exit quotes are excluded and counts are retained.
///
/// Layers use entry-time membership and ascending raw signals. Their forward
/// returns are never compounded when horizons overlap. The separate gross
/// research curve compounds complete one-bar baskets, excluding overnight
/// returns and costs. Executed layer accounts, if requested, belong to
/// ``AnalysisResult::layer_pnl`` instead.
pub fn analyze_factor(
    panel: &MarketPanel,
    signal: &SignalMatrix,
    config: &BacktestConfig,
) -> Result<AnalysisResult> {
    validate_shapes(panel, signal)?;
    let (_, cols) = panel.shape();
    let start = timeutil::parse_iso_date(&config.t_start)?;
    let end = timeutil::parse_iso_date(&config.t_end)?;
    let entries: Vec<usize> = (1..panel.shape().0)
        .filter(|&i| {
            let day = timeutil::local_date(panel.open_timestamps[i]);
            start <= day && day <= end
        })
        .collect();
    let order = symbol_order(panel);
    let mut nav = vec![1.0; config.quantiles as usize];
    let mut ic = IcColumns {
        timestamp: Vec::new(),
        signal_timestamp: Vec::new(),
        forward_end: Vec::new(),
        entry_n_assets: Vec::new(),
        n_assets: Vec::new(),
        tied_assets: Vec::new(),
        pearson_ic: Vec::new(),
        spearman_ic: Vec::new(),
        reason: Vec::new(),
    };
    let mut layers = LayerColumns::default();
    for chunk in entries.chunks(CHUNK_SIZE) {
        let rows = chunk.len();
        let mut values = vec![0.0; rows * cols];
        let mut eligible = vec![false; rows * cols];
        for (k, &entry) in chunk.iter().enumerate() {
            for c in 0..cols {
                let value = signal.values.get(entry - 1, c);
                let open = panel.open.get(entry, c);
                values[k * cols + c] = value;
                eligible[k * cols + c] = panel.base_weights.get(entry, c) > 0.0
                    && panel.tradable.get(entry, c)
                    && panel.volume.get(entry, c) > 0.0
                    && open.is_finite()
                    && open > 0.0
                    && value.is_finite();
            }
        }
        let (returns, valid_window, ends) =
            forward_returns(panel, chunk, config.ic_horizon_bars as usize, config)?;
        let (one_bar, _, _) = forward_returns(panel, chunk, 1, config)?;
        ic_rows(
            panel,
            chunk,
            &values,
            &eligible,
            &returns,
            &valid_window,
            &ends,
            config,
            &mut ic,
        );
        let labels = groups(&values, &eligible, rows, cols, config.quantiles, &order);
        layer_rows(
            panel,
            chunk,
            &labels,
            &returns,
            &one_bar,
            config,
            &mut nav,
            &mut layers,
        );
    }
    let mut ic_frame = OutFrame::new();
    ic_frame.push("timestamp", OutCol::Ts(ic.timestamp));
    ic_frame.push("signal_timestamp", OutCol::Ts(ic.signal_timestamp));
    ic_frame.push("forward_end", OutCol::Ts(ic.forward_end));
    ic_frame.push("entry_n_assets", OutCol::I64(ic.entry_n_assets));
    ic_frame.push("n_assets", OutCol::I64(ic.n_assets));
    ic_frame.push("tied_assets", OutCol::I64(ic.tied_assets));
    ic_frame.push("pearson_ic", OutCol::F64(ic.pearson_ic));
    ic_frame.push("spearman_ic", OutCol::F64(ic.spearman_ic));
    ic_frame.push("reason", OutCol::Str(ic.reason));
    let mut layer_frame = OutFrame::new();
    layer_frame.push("timestamp", OutCol::Ts(layers.timestamp));
    layer_frame.push("date", OutCol::Str(layers.date));
    layer_frame.push("layer", OutCol::I64(layers.layer));
    layer_frame.push("n_assets", OutCol::I64(layers.n_assets));
    layer_frame.push("forward_n_assets", OutCol::I64(layers.forward_n_assets));
    layer_frame.push("one_bar_n_assets", OutCol::I64(layers.one_bar_n_assets));
    layer_frame.push(
        "mean_forward_return",
        OutCol::F64(layers.mean_forward_return),
    );
    layer_frame.push(
        "mean_one_bar_return",
        OutCol::F64(layers.mean_one_bar_return),
    );
    layer_frame.push("gross_research_nav", OutCol::F64(layers.gross_research_nav));
    let intraday = intraday_frame(&ic_frame);
    let summary = summary_frame(&layer_frame);
    let mut assumptions = Map::new();
    assumptions.insert(
        "signal_timing".into(),
        Value::String("previous completed bar; next bar open entry".into()),
    );
    assumptions.insert(
        "ic_horizon_bars".into(),
        Value::from(config.ic_horizon_bars),
    );
    assumptions.insert("ic_min_assets".into(), Value::from(config.ic_min_assets));
    assumptions.insert(
        "return_basis".into(),
        Value::String("gross shareholder return with supplied actions".into()),
    );
    assumptions.insert(
        "price_basis".into(),
        Value::String(panel.price_basis.clone()),
    );
    assumptions.insert(
        "layer_order".into(),
        Value::String("ascending raw factor; symbol-ordered ties".into()),
    );
    assumptions.insert(
        "research_nav".into(),
        Value::String(
            "non-overlapping one-bar open-to-close baskets; no overnight return or costs; \
             incomplete valuation invalidates subsequent NAV"
                .into(),
        ),
    );
    assumptions.insert(
        "account_pnl".into(),
        Value::String("execution-account layer PnL is layer_pnl".into()),
    );
    Ok(AnalysisResult {
        layers: layer_frame,
        ic: ic_frame,
        ic_intraday: intraday,
        layer_summary: summary,
        layer_pnl: None,
        assumptions,
    })
}

fn nan_stats(values: &[f64]) -> (f64, i64, f64) {
    let finite: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    let count = finite.len();
    let mean = if count == 0 {
        f64::NAN
    } else {
        matrix::sum(&finite) / count as f64
    };
    let std = if count < 2 {
        f64::NAN
    } else {
        let squares: Vec<f64> = finite.iter().map(|v| (v - mean) * (v - mean)).collect();
        (matrix::sum(&squares) / (count - 1) as f64).sqrt()
    };
    (mean, count as i64, std)
}

fn intraday_frame(ic: &OutFrame) -> OutFrame {
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, timestamp) in ic.col("timestamp").i64s().iter().enumerate() {
        groups
            .entry(timeutil::strftime_hk(*timestamp, "%H:%M"))
            .or_default()
            .push(i);
    }
    let pearson = ic.col("pearson_ic").f64s();
    let spearman = ic.col("spearman_ic").f64s();
    let mut frame = OutFrame::new();
    let mut local_time = Vec::new();
    let (mut pm, mut pc, mut ps, mut sm, mut sc, mut ss) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    for (time, rows) in groups {
        local_time.push(time);
        let (mean, count, std) = nan_stats(&rows.iter().map(|&i| pearson[i]).collect::<Vec<_>>());
        pm.push(mean);
        pc.push(count);
        ps.push(std);
        let (mean, count, std) = nan_stats(&rows.iter().map(|&i| spearman[i]).collect::<Vec<_>>());
        sm.push(mean);
        sc.push(count);
        ss.push(std);
    }
    frame.push("local_time", OutCol::Str(local_time));
    frame.push("pearson_mean", OutCol::F64(pm));
    frame.push("pearson_count", OutCol::I64(pc));
    frame.push("pearson_std", OutCol::F64(ps));
    frame.push("spearman_mean", OutCol::F64(sm));
    frame.push("spearman_count", OutCol::I64(sc));
    frame.push("spearman_std", OutCol::F64(ss));
    frame
}

fn summary_frame(layers: &OutFrame) -> OutFrame {
    let mut groups: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (i, layer) in layers.col("layer").i64s().iter().enumerate() {
        groups.entry(*layer).or_default().push(i);
    }
    let n_assets = layers.col("n_assets").i64s();
    let forward = layers.col("forward_n_assets").i64s();
    let mean_forward = layers.col("mean_forward_return").f64s();
    let mean_one = layers.col("mean_one_bar_return").f64s();
    let mut frame = OutFrame::new();
    let (mut layer_col, mut bars, mut assets, mut forward_obs, mut mf, mut mo) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    for (layer, rows) in groups {
        layer_col.push(layer);
        bars.push(rows.len() as i64);
        assets.push(rows.iter().map(|&i| n_assets[i]).sum());
        forward_obs.push(rows.iter().map(|&i| forward[i]).sum());
        mf.push(nan_stats(&rows.iter().map(|&i| mean_forward[i]).collect::<Vec<_>>()).0);
        mo.push(nan_stats(&rows.iter().map(|&i| mean_one[i]).collect::<Vec<_>>()).0);
    }
    frame.push("layer", OutCol::I64(layer_col));
    frame.push("bars", OutCol::I64(bars));
    frame.push("asset_observations", OutCol::I64(assets));
    frame.push("forward_observations", OutCol::I64(forward_obs));
    frame.push("mean_forward_return", OutCol::F64(mf));
    frame.push("mean_one_bar_return", OutCol::F64(mo));
    frame
}
