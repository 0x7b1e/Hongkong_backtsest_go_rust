//! Causal cash-equity simulation with bounded execution and an audit trail.

use std::collections::VecDeque;
use std::path::Path;

use anyhow::{bail, Result};
use serde_json::{Map, Value};

use crate::allocation;
use crate::config::{json_float, BacktestConfig, Mode};
use crate::errors::Error;
use crate::frames::{AuditKind, AuditRecords, AuditRow, OutCol, OutFrame, TradeDetail};
use crate::matrix;
use crate::models::{BacktestResult, MarketPanel, SignalMatrix};
use crate::rules::{self, Fees};
use crate::timeutil;

/// Mutable cash, shares, acquisition lots and outstanding targets.
struct Book {
    cash: f64,
    shares: Vec<f64>,
    /// Per asset: (acquisition session, remaining quantity), oldest first.
    lots: Vec<VecDeque<(i64, f64)>>,
    desired: Vec<f64>,
    slices: Vec<i64>,
}

/// Audit sinks of the costed account; the frictionless shadow has none.
struct Audits<'a> {
    trades: &'a mut AuditRecords,
    restrictions: &'a mut AuditRecords,
}

fn record(
    panel: &MarketPanel,
    bar: usize,
    asset: usize,
    side: &'static str,
    requested: f64,
    filled: f64,
    reason: &'static str,
) -> AuditRow {
    AuditRow {
        timestamp: panel.open_timestamps[bar],
        bar_end: panel.timestamps[bar],
        symbol: panel.symbols[asset].clone(),
        side,
        requested,
        filled,
        trade: None,
        reason,
    }
}

/// Round down without creating shares or borrowing money.
fn round_quantity(quantity: f64, lot: f64) -> f64 {
    f64::max(0.0, ((quantity + 1e-9) / lot).floor() * lot)
}

/// Remove sold shares from oldest acquisition lots first.
fn consume_lots(book: &mut Book, asset: usize, mut quantity: f64) {
    while quantity > 1e-8 {
        let lot = book.lots[asset].front_mut().expect("sold shares have lots");
        let used = quantity.min(lot.1);
        lot.1 -= used;
        quantity -= used;
        if lot.1 < 1e-8 {
            book.lots[asset].pop_front();
        }
    }
}

fn fees_total(
    notional: f64,
    timestamp: i64,
    config: &BacktestConfig,
    frictionless: bool,
    exempt: bool,
) -> Result<f64> {
    if frictionless {
        return Ok(0.0);
    }
    Ok(rules::transaction_fees(notional, timestamp, config, exempt)?.total())
}

/// Find the largest funded lot count including nonlinear charges.
#[allow(clippy::too_many_arguments)]
fn affordable(
    quantity: f64,
    unit: f64,
    price: f64,
    cash: f64,
    timestamp: i64,
    config: &BacktestConfig,
    frictionless: bool,
    exempt: bool,
) -> Result<f64> {
    let notional = quantity * price;
    let fees = fees_total(notional, timestamp, config, frictionless, exempt)?;
    if notional + fees <= cash + 1e-8 {
        return Ok(quantity);
    }
    let (mut low, mut high) = (0i64, (quantity / unit + 1e-9) as i64);
    while low < high {
        let middle = (low + high + 1) / 2;
        let notional = middle as f64 * unit * price;
        let fees = fees_total(notional, timestamp, config, frictionless, exempt)?;
        if notional + fees <= cash + 1e-8 {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    Ok(low as f64 * unit)
}

/// Attempt one child order and return filled consideration and fees.
#[allow(clippy::too_many_arguments)]
fn execute(
    book: &mut Book,
    panel: &MarketPanel,
    bar: usize,
    asset: usize,
    config: &BacktestConfig,
    frictionless: bool,
    audits: &mut Option<Audits>,
    exit_phase: bool,
) -> Result<(f64, f64)> {
    let delta = book.desired[asset] - book.shares[asset];
    if delta.abs() < 1e-8 {
        return Ok((0.0, 0.0));
    }
    let side: &'static str = if delta > 0.0 { "buy" } else { "sell" };
    let timestamp = panel.open_timestamps[bar];
    let count = book.slices[asset].max(1);
    let requested = delta.abs() / count as f64;
    let mut pending: Vec<AuditRow> = Vec::new();
    let reject = |pending: &mut Vec<AuditRow>, reason: &'static str, filled: f64| {
        if !frictionless {
            pending.push(record(panel, bar, asset, side, requested, filled, reason));
        }
    };
    // Report actual fills on every contributing restriction.
    let finish = |pending: Vec<AuditRow>,
                  audits: &mut Option<Audits>,
                  notional: f64,
                  fees: f64,
                  filled: f64|
     -> Result<(f64, f64)> {
        for mut row in pending {
            row.filled = filled;
            audits
                .as_mut()
                .expect("restrictions are recorded for costed accounts")
                .restrictions
                .append(row)?;
        }
        Ok((notional, fees))
    };
    if !rules::is_continuous_session(timestamp) {
        reject(&mut pending, "outside_continuous_session", 0.0);
        return finish(pending, audits, 0.0, 0.0, 0.0);
    }
    let use_exit_window = side == "sell" || exit_phase;
    let (start, end) = if use_exit_window {
        config.exit_window()
    } else {
        config.entry_window()
    };
    if !rules::within_window(timestamp, start, end) {
        reject(&mut pending, "outside_execution_window", 0.0);
        return finish(pending, audits, 0.0, 0.0, 0.0);
    }
    book.slices[asset] = (count - 1).max(1);
    let raw_price = panel.open.get(bar, asset);
    if !raw_price.is_finite() || raw_price <= 0.0 {
        reject(&mut pending, "missing_quote", 0.0);
        return finish(pending, audits, 0.0, 0.0, 0.0);
    }
    if !panel.tradable.get(bar, asset) {
        reject(&mut pending, "not_tradable", 0.0);
        return finish(pending, audits, 0.0, 0.0, 0.0);
    }
    let volume = panel.volume.get(bar, asset);
    if !volume.is_finite() || volume <= 0.0 {
        reject(&mut pending, "no_volume", 0.0);
        return finish(pending, audits, 0.0, 0.0, 0.0);
    }
    let unit = if config.round_lots {
        panel.lot_sizes.get(bar, asset)
    } else {
        1.0
    };
    if !unit.is_finite() || unit <= 0.0 {
        bail!("Historical board-lot sizes are required and must be finite and positive");
    }
    let rounded = round_quantity(requested, unit);
    if rounded < requested - 1e-8 {
        reject(&mut pending, "board_lot_rounding", rounded);
    }
    let mut quantity = rounded;
    if side == "sell" && config.mode == Mode::T1 {
        let session = panel.session_id[bar];
        let mut sellable = 0.0;
        for lot in &book.lots[asset] {
            if session - lot.0 >= config.holding_days {
                sellable += lot.1;
            }
        }
        let allowed = round_quantity(quantity.min(sellable), unit);
        if allowed < quantity - 1e-8 {
            reject(&mut pending, "holding_period_locked", allowed);
        }
        quantity = allowed;
    }
    let allowed = round_quantity(volume * config.participation_rate, unit);
    if quantity > allowed + 1e-8 {
        reject(&mut pending, "participation_limit", allowed);
        quantity = allowed;
    }
    let slippage = if frictionless {
        0.0
    } else {
        config.slippage_bps / 10_000.0
    };
    let price = raw_price
        * if side == "buy" {
            1.0 + slippage
        } else {
            1.0 - slippage
        };
    let lower = panel.vcm_lower.get(bar, asset);
    let upper = panel.vcm_upper.get(bar, asset);
    let active = panel
        .vcm_active
        .as_ref()
        .map(|states| states.get(bar, asset) as i64)
        .unwrap_or(-1);
    let complete_bands = lower.is_finite() && upper.is_finite();
    if active == 1 && !complete_bands {
        reject(&mut pending, "missing_vcm_event_bands", 0.0);
        return finish(pending, audits, 0.0, 0.0, 0.0);
    }
    if active != 0 && complete_bands && (price < lower || price > upper) {
        reject(&mut pending, "vcm_price_band", 0.0);
        return finish(pending, audits, 0.0, 0.0, 0.0);
    }
    let exempt = panel
        .stamp_duty_exempt
        .as_ref()
        .map(|flags| flags.get(bar, asset))
        .unwrap_or(false);
    if side == "buy" {
        let funded = affordable(
            quantity,
            unit,
            price,
            book.cash,
            timestamp,
            config,
            frictionless,
            exempt,
        )?;
        if funded < quantity - 1e-8 {
            reject(&mut pending, "insufficient_cash", funded);
        }
        quantity = funded;
    }
    if quantity <= 0.0 {
        return finish(pending, audits, 0.0, 0.0, 0.0);
    }
    let notional = quantity * price;
    let fees = if frictionless {
        Fees::default()
    } else {
        rules::transaction_fees(notional, timestamp, config, exempt)?
    };
    let total = fees.total();
    if side == "sell" && notional + book.cash < total {
        reject(&mut pending, "insufficient_cash_for_sell_fees", 0.0);
        return finish(pending, audits, 0.0, 0.0, 0.0);
    }
    if side == "buy" {
        book.cash -= notional + total;
        book.shares[asset] += quantity;
        let session = panel.session_id[bar];
        match book.lots[asset].back_mut() {
            Some(last) if last.0 == session => last.1 += quantity,
            _ => book.lots[asset].push_back((session, quantity)),
        }
    } else {
        book.cash += notional - total;
        book.shares[asset] -= quantity;
        consume_lots(book, asset, quantity);
    }
    book.cash = book.cash.max(0.0);
    if !frictionless {
        let audits_ref = audits
            .as_mut()
            .expect("trades are recorded for costed accounts");
        audits_ref.trades.append(AuditRow {
            timestamp,
            bar_end: panel.timestamps[bar],
            symbol: panel.symbols[asset].clone(),
            side,
            requested,
            filled: quantity,
            trade: Some(TradeDetail {
                price,
                notional,
                fees: total,
                stamp_duty: fees.stamp_duty,
                sfc_levy: fees.sfc_levy,
                afrc_levy: fees.afrc_levy,
                trading_fee: fees.trading_fee,
                trading_tariff: fees.trading_tariff,
                commission: fees.commission,
                cash: book.cash,
            }),
            reason: if exit_phase {
                "t0_restore_base"
            } else {
                "target_rebalance"
            },
        })?;
    }
    finish(pending, audits, notional, total, quantity)
}

/// One row of an account curve.
struct AccountRow {
    timestamp: i64,
    nav: f64,
    cash: f64,
    position_value: f64,
    turnover: f64,
    fees: f64,
    dividends: f64,
}

fn divide_where_positive(numerators: &[f64], valuation: &[f64]) -> Vec<f64> {
    numerators
        .iter()
        .zip(valuation)
        .map(|(n, v)| if *v > 0.0 { n / v } else { 0.0 })
        .collect()
}

/// Run one independently funded account using only completed signals.
fn simulate(
    panel: &MarketPanel,
    signal: &SignalMatrix,
    config: &BacktestConfig,
    indices: &[usize],
    frictionless: bool,
    mut audits: Option<Audits>,
) -> Result<Vec<AccountRow>> {
    let assets = panel.symbols.len();
    let mut book = Book {
        cash: config.initial_capital,
        shares: vec![0.0; assets],
        lots: vec![VecDeque::new(); assets],
        desired: vec![0.0; assets],
        slices: vec![1; assets],
    };
    let mut mark = vec![f64::NAN; assets];
    for bar in 0..indices[0] {
        for asset in 0..assets {
            let close = panel.close.get(bar, asset);
            if close.is_finite() && close > 0.0 {
                mark[asset] = close;
            }
        }
    }
    let nan_row = vec![f64::NAN; assets];
    let mut values: Vec<f64> = nan_row.clone();
    let mut previous_base = vec![f64::NAN; assets];
    let mut initialized = false;
    let mut flattening = false;
    let mut previous_valuation = vec![0.0; assets];
    let mut rows = Vec::with_capacity(indices.len());
    let flatten_time = config.flatten_time();
    for &bar in indices {
        let dividends = matrix::dot(&book.shares, panel.cash_dividend.row(bar));
        book.cash += dividends;
        let multipliers = panel.share_multiplier.row(bar);
        for asset in 0..assets {
            book.shares[asset] *= multipliers[asset];
            book.desired[asset] *= multipliers[asset];
            mark[asset] /= multipliers[asset];
            if multipliers[asset] != 1.0 {
                for lot in book.lots[asset].iter_mut() {
                    lot.1 *= multipliers[asset];
                }
            }
        }
        let open_prices = panel.open.row(bar);
        let usable: Vec<bool> = open_prices
            .iter()
            .map(|p| p.is_finite() && *p > 0.0)
            .collect();
        let valuation: Vec<f64> = (0..assets)
            .map(|a| {
                let v = if usable[a] { open_prices[a] } else { mark[a] };
                if v.is_nan() {
                    0.0
                } else if v == f64::INFINITY {
                    f64::MAX
                } else if v == f64::NEG_INFINITY {
                    f64::MIN
                } else {
                    v
                }
            })
            .collect();
        let nav = book.cash + matrix::dot(&book.shares, &valuation);
        let base = panel.base_weights.row(bar);
        let mut changed = !initialized || base != previous_base.as_slice();
        changed |= (0..assets)
            .any(|a| valuation[a] > 0.0 && previous_valuation[a] <= 0.0 && base[a] > 0.0);
        if panel.is_session_start[bar] {
            flattening = false;
        }
        let known_prior = bar >= 1 && panel.timestamps[bar - 1] <= panel.open_timestamps[bar];
        if config.mode == Mode::T1 && panel.is_session_start[bar] {
            if known_prior && panel.session_id[bar - 1] < panel.session_id[bar] {
                values = signal.values.row(bar - 1).to_vec();
            } else {
                values = nan_row.clone();
            }
            changed = true;
        }
        if config.mode == Mode::T0 {
            changed |= panel.is_session_start[bar];
            let was_flattening = flattening;
            flattening |= timeutil::local_time_of_day(panel.open_timestamps[bar]) >= flatten_time
                || panel.is_session_end[bar];
            if flattening {
                values = nan_row.clone();
                changed |= !was_flattening || panel.is_session_end[bar];
            } else if known_prior && (!initialized || signal.updated[bar - 1]) {
                values = signal.values.row(bar - 1).to_vec();
                changed = true;
            }
        }
        if changed {
            let weights = allocation::target_weights(base, &values, config.keep, config.direction);
            let scaled: Vec<f64> = weights.iter().map(|w| w * nav).collect();
            book.desired = divide_where_positive(&scaled, &valuation);
            let slices = if flattening && panel.is_session_end[bar] {
                1
            } else {
                config.order_slices
            };
            book.slices.iter_mut().for_each(|s| *s = slices);
            previous_base = base.to_vec();
            initialized = true;
        }
        let mut consideration = 0.0;
        let mut fees = 0.0;
        let sells: Vec<usize> = (0..assets)
            .filter(|&a| book.desired[a] < book.shares[a] - 1e-8)
            .collect();
        let mut buys: Vec<usize> = (0..assets)
            .filter(|&a| book.desired[a] > book.shares[a] + 1e-8)
            .collect();
        let floor_numerators: Vec<f64> = base.iter().map(|b| config.keep * b * nav).collect();
        let floor_shares = divide_where_positive(&floor_numerators, &valuation);
        if !buys.is_empty() {
            let priorities: Vec<f64> = buys
                .iter()
                .map(|&a| f64::max(floor_shares[a] - book.shares[a], 0.0))
                .collect();
            let mut order: Vec<usize> = (0..buys.len()).collect();
            // argsort(-priorities, kind="stable")
            order.sort_by(|&x, &y| {
                (-priorities[x])
                    .partial_cmp(&-priorities[y])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            buys = order.into_iter().map(|i| buys[i]).collect();
        }
        for asset in sells.into_iter().chain(buys) {
            let (traded, charged) = execute(
                &mut book,
                panel,
                bar,
                asset,
                config,
                frictionless,
                &mut audits,
                config.mode == Mode::T0 && flattening,
            )?;
            consideration += traded;
            fees += charged;
        }
        if let Some(audits) = audits.as_mut() {
            let executed_nav = book.cash + matrix::dot(&book.shares, &valuation);
            let actual_numerators: Vec<f64> = base
                .iter()
                .map(|b| config.keep * b * executed_nav)
                .collect();
            let actual_floor = divide_where_positive(&actual_numerators, &valuation);
            for asset in 0..assets {
                if valuation[asset] <= 0.0 && base[asset] > 0.0 {
                    audits.restrictions.append(record(
                        panel,
                        bar,
                        asset,
                        "buy",
                        0.0,
                        0.0,
                        "cannot_size_missing_quote",
                    ))?;
                }
            }
            for asset in 0..assets {
                if book.shares[asset] < actual_floor[asset] - 1e-8 {
                    audits.restrictions.append(record(
                        panel,
                        bar,
                        asset,
                        "buy",
                        actual_floor[asset] - book.shares[asset],
                        0.0,
                        "base_floor_shortfall",
                    ))?;
                }
            }
            if config.mode == Mode::T0 && panel.is_session_end[bar] {
                for asset in 0..assets {
                    let residual = book.desired[asset] - book.shares[asset];
                    if residual.abs() > 1e-8 {
                        audits.restrictions.append(record(
                            panel,
                            bar,
                            asset,
                            if residual > 0.0 { "buy" } else { "sell" },
                            residual.abs(),
                            0.0,
                            "t0_end_residual",
                        ))?;
                    }
                }
            }
        }
        let close = panel.close.row(bar);
        for asset in 0..assets {
            if usable[asset] {
                mark[asset] = open_prices[asset];
            }
            if close[asset].is_finite() && close[asset] > 0.0 {
                mark[asset] = close[asset];
            }
        }
        if (0..assets).any(|a| book.shares[a] > 0.0 && !mark[a].is_finite()) {
            bail!("Held security has no available valuation quote");
        }
        let marks: Vec<f64> = mark
            .iter()
            .map(|m| if m.is_nan() { 0.0 } else { *m })
            .collect();
        let position_value = matrix::dot(&book.shares, &marks);
        rows.push(AccountRow {
            timestamp: panel.timestamps[bar],
            nav: book.cash + position_value,
            cash: book.cash,
            position_value,
            turnover: if nav != 0.0 { consideration / nav } else { 0.0 },
            fees,
            dividends,
        });
        previous_valuation = valuation;
    }
    Ok(rows)
}

fn std_ddof1(values: &[f64]) -> f64 {
    let finite: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    if finite.len() < 2 {
        return f64::NAN;
    }
    let mean = matrix::sum(&finite) / finite.len() as f64;
    let squares: Vec<f64> = finite.iter().map(|v| (v - mean) * (v - mean)).collect();
    (matrix::sum(&squares) / (finite.len() - 1) as f64).sqrt()
}

fn mean_skipna(values: &[f64]) -> f64 {
    let finite: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    if finite.is_empty() {
        return f64::NAN;
    }
    matrix::sum(&finite) / finite.len() as f64
}

/// Backtest one factor and a separate frictionless shadow account.
///
/// Orders execute at a bar's open using earlier completed-bar signals.
/// Supplied bar volume caps simulated fills; it is an ex-post liquidity
/// constraint, not information used to generate the order. T1 holding locks
/// are strategy rules, not Hong Kong legal settlement restrictions. The base
/// floor applies to target weights. Lots, liquidity and cash may prevent
/// achieving it; every observed shortfall is recorded.
pub fn run_backtest(
    panel: &MarketPanel,
    signal: &SignalMatrix,
    config: &BacktestConfig,
    audit_dir: Option<&Path>,
) -> Result<BacktestResult> {
    if signal.values.shape() != panel.shape() {
        bail!("Signal values must match the market panel shape");
    }
    if signal.updated.len() != panel.shape().0 {
        bail!("Signal updated flags must align with panel bars");
    }
    if panel.frequency != config.frequency {
        bail!("Configured frequency must match panel frequency");
    }
    if config.slippage_bps >= 10_000.0 {
        bail!("Slippage must be less than 10000 basis points");
    }
    if panel.price_basis != "raw" && !config.allow_adjusted_prices {
        bail!("Raw execution prices required; research opt-in needed");
    }
    let start = timeutil::parse_iso_date(&config.t_start)?;
    let end = timeutil::parse_iso_date(&config.t_end)?;
    let indices: Vec<usize> = (0..panel.timestamps.len())
        .filter(|&i| {
            let day = timeutil::local_date(panel.timestamps[i]);
            start <= day && day <= end
        })
        .collect();
    if indices.is_empty() {
        bail!("No market bars fall within the requested dates");
    }
    if timeutil::local_date(panel.open_timestamps[indices[0]])
        .format("%Y-%m-%d")
        .to_string()
        .as_str()
        < "2014-11-01"
    {
        bail!("Transaction fees before 2014-11-01 unsupported");
    }
    if config.enforce_vcm {
        let cols = panel.symbols.len();
        for bar in indices[0]..=*indices.last().unwrap() {
            for asset in 0..cols {
                let complete = panel.vcm_lower.get(bar, asset).is_finite()
                    && panel.vcm_upper.get(bar, asset).is_finite();
                let eligible = panel.vcm_eligible.get(bar, asset);
                let requires = match &panel.vcm_active {
                    None => eligible,
                    Some(states) => {
                        let state = states.get(bar, asset);
                        state == 1 || (eligible && state == -1)
                    }
                };
                if requires && !complete {
                    bail!("Strict VCM requires complete bands for active events or unknown eligible states");
                }
            }
        }
    }
    if let Some(directory) = audit_dir {
        for filename in ["trades.parquet", "restrictions.parquet"] {
            let path = directory.join(filename);
            if path.exists() {
                bail!(Error::FileExists(path.display().to_string()));
            }
        }
    }
    let mut trades = AuditRecords::new(AuditKind::Trades, audit_dir)?;
    let mut restrictions = AuditRecords::new(AuditKind::Restrictions, audit_dir)?;
    let simulated = (|| -> Result<(Vec<AccountRow>, Vec<AccountRow>)> {
        let net = simulate(
            panel,
            signal,
            config,
            &indices,
            false,
            Some(Audits {
                trades: &mut trades,
                restrictions: &mut restrictions,
            }),
        )?;
        let gross = simulate(panel, signal, config, &indices, true, None)?;
        Ok((net, gross))
    })();
    let closed_trades = trades.close();
    let closed_restrictions = restrictions.close();
    let (net, gross) = simulated?;
    closed_trades?;
    closed_restrictions?;
    let capital = config.initial_capital;
    let n = net.len();
    let net_nav: Vec<f64> = net.iter().map(|r| r.nav).collect();
    let gross_nav: Vec<f64> = gross.iter().map(|r| r.nav).collect();
    let pnl_and_returns = |navs: &[f64]| -> (Vec<f64>, Vec<f64>) {
        let mut pnl = Vec::with_capacity(navs.len());
        let mut returns = Vec::with_capacity(navs.len());
        for i in 0..navs.len() {
            let previous = if i == 0 || navs[i - 1].is_nan() {
                capital
            } else {
                navs[i - 1]
            };
            pnl.push(navs[i] - previous);
            returns.push(navs[i] / previous - 1.0);
        }
        (pnl, returns)
    };
    let (net_pnl, net_return) = pnl_and_returns(&net_nav);
    let (gross_pnl, gross_return) = pnl_and_returns(&gross_nav);
    // Benchmark: forward-filled closes, first return from the first open.
    let mut benchmark = panel.benchmark_close.clone();
    let mut last = f64::NAN;
    for value in benchmark.iter_mut() {
        if value.is_nan() {
            *value = last;
        } else {
            last = *value;
        }
    }
    let selected: Vec<f64> = indices.iter().map(|&i| benchmark[i]).collect();
    let opening = panel
        .benchmark_open
        .as_ref()
        .map(|v| v[indices[0]])
        .unwrap_or(f64::NAN);
    let (first_previous, policy) = if opening.is_finite() && opening > 0.0 {
        (opening, "first_execution_open")
    } else {
        (selected[0], "first_close_missing_open")
    };
    let benchmark_return: Vec<f64> = (0..n)
        .map(|i| {
            let previous = if i == 0 {
                first_previous
            } else {
                selected[i - 1]
            };
            selected[i] / previous - 1.0
        })
        .collect();
    let excess_return: Vec<f64> = (0..n)
        .map(|i| net_return[i] - benchmark_return[i])
        .collect();
    let mut benchmark_nav = Vec::with_capacity(n);
    let mut product = 1.0;
    for value in &benchmark_return {
        product *= 1.0 + value;
        benchmark_nav.push(capital * product);
    }
    let excess_nav: Vec<f64> = (0..n).map(|i| net_nav[i] / benchmark_nav[i]).collect();
    let mut pnl = OutFrame::new();
    pnl.push(
        "timestamp",
        OutCol::Ts(net.iter().map(|r| r.timestamp).collect()),
    );
    pnl.push("net_nav", OutCol::F64(net_nav.clone()));
    pnl.push("cash", OutCol::F64(net.iter().map(|r| r.cash).collect()));
    pnl.push(
        "position_value",
        OutCol::F64(net.iter().map(|r| r.position_value).collect()),
    );
    pnl.push(
        "turnover",
        OutCol::F64(net.iter().map(|r| r.turnover).collect()),
    );
    pnl.push("fees", OutCol::F64(net.iter().map(|r| r.fees).collect()));
    pnl.push(
        "dividends",
        OutCol::F64(net.iter().map(|r| r.dividends).collect()),
    );
    pnl.push("gross_nav", OutCol::F64(gross_nav.clone()));
    pnl.push("net_pnl", OutCol::F64(net_pnl));
    pnl.push("net_return", OutCol::F64(net_return));
    pnl.push("gross_pnl", OutCol::F64(gross_pnl));
    pnl.push("gross_return", OutCol::F64(gross_return));
    pnl.push("benchmark_return", OutCol::F64(benchmark_return));
    pnl.push("excess_return", OutCol::F64(excess_return));
    pnl.push("benchmark_nav", OutCol::F64(benchmark_nav.clone()));
    pnl.push("excess_nav", OutCol::F64(excess_nav.clone()));
    // Daily NAV: last observation of each local calendar day.
    let mut daily_nav: Vec<f64> = Vec::new();
    let mut current_day = None;
    for (row, value) in net.iter().zip(&net_nav) {
        let day = timeutil::local_date(row.timestamp);
        if current_day == Some(day) {
            if !value.is_nan() {
                *daily_nav.last_mut().unwrap() = *value;
            }
        } else {
            current_day = Some(day);
            daily_nav.push(*value);
        }
    }
    daily_nav.retain(|v| !v.is_nan());
    let daily_returns: Vec<f64> = (0..daily_nav.len())
        .map(|i| {
            let previous = if i == 0 { capital } else { daily_nav[i - 1] };
            daily_nav[i] / previous - 1.0
        })
        .collect();
    let volatility = std_ddof1(&daily_returns);
    let sharpe = if volatility > 0.0 {
        mean_skipna(&daily_returns) / volatility * 252f64.sqrt()
    } else {
        f64::NAN
    };
    let mut running_max = f64::NEG_INFINITY;
    let mut max_drawdown = f64::INFINITY;
    for value in std::iter::once(capital).chain(net_nav.iter().copied()) {
        running_max = if value.is_nan() || running_max.is_nan() {
            f64::NAN
        } else {
            running_max.max(value)
        };
        let drawdown = value / running_max - 1.0;
        max_drawdown = if drawdown.is_nan() || max_drawdown.is_nan() {
            f64::NAN
        } else {
            max_drawdown.min(drawdown)
        };
    }
    let final_net = *net_nav.last().unwrap();
    let final_gross = *gross_nav.last().unwrap();
    let final_benchmark = *benchmark_nav.last().unwrap();
    let final_excess = *excess_nav.last().unwrap();
    let fees_sum = matrix::sum(
        &net.iter()
            .map(|r| r.fees)
            .filter(|v| !v.is_nan())
            .collect::<Vec<_>>(),
    );
    let turnover_sum = matrix::sum(
        &net.iter()
            .map(|r| r.turnover)
            .filter(|v| !v.is_nan())
            .collect::<Vec<_>>(),
    );
    let mut metrics = Map::new();
    metrics.insert("factor".into(), Value::String(signal.factor.clone()));
    metrics.insert("mode".into(), Value::String(config.mode.as_str().into()));
    metrics.insert(
        "net_total_return".into(),
        json_float(final_net / capital - 1.0),
    );
    metrics.insert(
        "gross_total_return".into(),
        json_float(final_gross / capital - 1.0),
    );
    metrics.insert(
        "benchmark_total_return".into(),
        json_float(final_benchmark / capital - 1.0),
    );
    metrics.insert(
        "benchmark_start_policy".into(),
        Value::String(policy.into()),
    );
    metrics.insert("excess_total_return".into(), json_float(final_excess - 1.0));
    metrics.insert("net_pnl".into(), json_float(final_net - capital));
    metrics.insert("gross_pnl".into(), json_float(final_gross - capital));
    metrics.insert("fees".into(), json_float(fees_sum));
    metrics.insert("turnover".into(), json_float(turnover_sum));
    metrics.insert("max_drawdown".into(), json_float(max_drawdown));
    metrics.insert("sharpe".into(), json_float(sharpe));
    metrics.insert("trade_count".into(), Value::from(trades.count as u64));
    metrics.insert(
        "restriction_count".into(),
        Value::from(restrictions.count as u64),
    );
    metrics.insert(
        "price_basis".into(),
        Value::String(panel.price_basis.clone()),
    );
    metrics.insert(
        "adjusted_price_research".into(),
        Value::Bool(panel.price_basis != "raw"),
    );
    metrics.insert(
        "gross_method".into(),
        Value::String("independent_zero_cost_zero_slippage_account".into()),
    );
    metrics.insert(
        "vcm_policy".into(),
        Value::String(
            if config.enforce_vcm {
                "strict_supplied_state_and_bands"
            } else {
                "supplied_events_missing_triggers_unknown"
            }
            .into(),
        ),
    );
    Ok(BacktestResult {
        pnl,
        trades: trades.table(),
        restrictions: restrictions.table(),
        metrics,
        audit_streamed: audit_dir.is_some(),
    })
}
