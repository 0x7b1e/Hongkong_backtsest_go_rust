"""Causal cash-equity simulation with bounded execution and an audit trail."""

import collections
import contextlib
import dataclasses
import datetime
import math
import pathlib

import numpy as np
import pandas as pd

from hkfactor import allocation
from hkfactor import audit
from hkfactor import config as backtest_config
from hkfactor import models
from hkfactor import rules


@dataclasses.dataclass
class _Book:
    """Mutable cash, shares, acquisition lots and outstanding targets."""

    cash: float
    shares: np.ndarray
    lots: list[collections.deque]
    desired: np.ndarray
    slices: np.ndarray


def _record(
    rows: audit.AuditRecords | list[dict],
    panel: models.MarketPanel,
    bar: int,
    asset: int,
    side: str,
    requested: float,
    filled: float,
    reason: str,
) -> None:
    """Append one restriction with quantities and an executable timestamp."""
    rows.append(
        {
            "timestamp": panel.open_timestamps[bar],
            "bar_end": panel.timestamps[bar],
            "symbol": panel.symbols[asset],
            "side": side,
            "requested": float(requested),
            "filled": float(filled),
            "reason": reason,
        }
    )


def _round_quantity(quantity: float, lot: float) -> float:
    """Round down without creating shares or borrowing money."""
    return max(0.0, math.floor((quantity + 1e-9) / lot) * lot)


def _consume_lots(book: _Book, asset: int, quantity: float) -> None:
    """Remove sold shares from oldest acquisition lots first."""
    while quantity > 1e-8:
        lot = book.lots[asset][0]
        used = min(quantity, lot[1])
        lot[1] -= used
        quantity -= used
        if lot[1] < 1e-8:
            book.lots[asset].popleft()


def _affordable(
    quantity: float,
    unit: float,
    price: float,
    cash: float,
    timestamp: pd.Timestamp,
    config: backtest_config.BacktestConfig,
    frictionless: bool,
    stamp_duty_exempt: bool = False,
) -> float:
    """Find the largest funded lot count including nonlinear charges."""
    notional = quantity * price
    fees = (
        0
        if frictionless
        else rules.transaction_fees(
            notional, timestamp, config, stamp_duty_exempt=stamp_duty_exempt
        ).total
    )
    if notional + fees <= cash + 1e-8:
        return quantity
    low, high = 0, int(quantity / unit + 1e-9)
    while low < high:
        middle = (low + high + 1) // 2
        notional = middle * unit * price
        fees = (
            0
            if frictionless
            else rules.transaction_fees(
                notional, timestamp, config, stamp_duty_exempt=stamp_duty_exempt
            ).total
        )
        if notional + fees <= cash + 1e-8:
            low = middle
        else:
            high = middle - 1
    return low * unit


def _execute(
    book: _Book,
    panel: models.MarketPanel,
    bar: int,
    asset: int,
    config: backtest_config.BacktestConfig,
    frictionless: bool,
    trades: audit.AuditRecords | None,
    restrictions: audit.AuditRecords | None,
    *,
    exit_phase: bool = False,
) -> tuple[float, float]:
    """Attempt one child order and return filled consideration and fees."""
    delta = book.desired[asset] - book.shares[asset]
    if abs(delta) < 1e-8:
        return 0.0, 0.0
    side = "buy" if delta > 0 else "sell"
    timestamp = panel.open_timestamps[bar]
    count = max(1, book.slices[asset])
    requested = abs(delta) / count
    pending_restrictions: list[dict] = []

    def reject(reason: str, filled: float = 0.0) -> None:
        if not frictionless:
            _record(
                pending_restrictions,
                panel,
                bar,
                asset,
                side,
                requested,
                filled,
                reason,
            )

    def finish(
        notional: float = 0.0,
        fees: float = 0.0,
        filled: float = 0.0,
    ) -> tuple[float, float]:
        """Report actual fills on every contributing restriction."""
        for row in pending_restrictions:
            row["filled"] = filled
            assert restrictions is not None
            restrictions.append(row)
        return notional, fees

    if not rules.is_continuous_session(timestamp):
        reject("outside_continuous_session")
        return finish()
    use_exit_window = side == "sell" or exit_phase
    start = config.exit_start if use_exit_window else config.entry_start
    end = config.exit_end if use_exit_window else config.entry_end
    if not rules.within_window(timestamp, start, end):
        reject("outside_execution_window")
        return finish()
    book.slices[asset] = max(1, count - 1)
    raw_price = panel.open[bar, asset]
    if not np.isfinite(raw_price) or raw_price <= 0:
        reject("missing_quote")
        return finish()
    if not panel.tradable[bar, asset]:
        reject("not_tradable")
        return finish()
    volume = panel.volume[bar, asset]
    if not np.isfinite(volume) or volume <= 0:
        reject("no_volume")
        return finish()
    unit = float(panel.lot_sizes[bar, asset]) if config.round_lots else 1.0
    if not math.isfinite(unit) or unit <= 0:
        raise ValueError(
            "Historical board-lot sizes are required and must "
            "be finite and positive"
        )
    rounded = _round_quantity(requested, unit)
    if rounded < requested - 1e-8:
        reject("board_lot_rounding", rounded)
    quantity = rounded
    if side == "sell" and config.mode == "T1":
        session = panel.session_id[bar]
        sellable = sum(
            lot[1]
            for lot in book.lots[asset]
            if session - lot[0] >= config.holding_days
        )
        allowed = _round_quantity(min(quantity, sellable), unit)
        if allowed < quantity - 1e-8:
            reject("holding_period_locked", allowed)
        quantity = allowed
    allowed = _round_quantity(volume * config.participation_rate, unit)
    if quantity > allowed + 1e-8:
        reject("participation_limit", allowed)
        quantity = allowed
    slippage = 0 if frictionless else config.slippage_bps / 10_000
    price = raw_price * (1 + slippage if side == "buy" else 1 - slippage)
    lower, upper = panel.vcm_lower[bar, asset], panel.vcm_upper[bar, asset]
    states = getattr(panel, "vcm_active", None)
    active = -1 if states is None else states[bar, asset]
    complete_bands = np.isfinite(lower) and np.isfinite(upper)
    if active == 1 and not complete_bands:
        reject("missing_vcm_event_bands")
        return finish()
    if active != 0 and complete_bands and (price < lower or price > upper):
        reject("vcm_price_band")
        return finish()
    exemptions = getattr(panel, "stamp_duty_exempt", None)
    exempt = False if exemptions is None else bool(exemptions[bar, asset])
    if side == "buy":
        funded = _affordable(
            quantity,
            unit,
            price,
            book.cash,
            timestamp,
            config,
            frictionless,
            stamp_duty_exempt=exempt,
        )
        if funded < quantity - 1e-8:
            reject("insufficient_cash", funded)
        quantity = funded
    if quantity <= 0:
        return finish()
    notional = quantity * price
    fees = (
        rules.Fees()
        if frictionless
        else rules.transaction_fees(
            notional, timestamp, config, stamp_duty_exempt=exempt
        )
    )
    if side == "sell" and notional + book.cash < fees.total:
        reject("insufficient_cash_for_sell_fees")
        return finish()
    if side == "buy":
        book.cash -= notional + fees.total
        book.shares[asset] += quantity
        if (
            book.lots[asset]
            and book.lots[asset][-1][0] == panel.session_id[bar]
        ):
            book.lots[asset][-1][1] += quantity
        else:
            book.lots[asset].append([panel.session_id[bar], quantity])
    else:
        book.cash += notional - fees.total
        book.shares[asset] -= quantity
        _consume_lots(book, asset, quantity)
    book.cash = max(0.0, book.cash)
    if not frictionless:
        assert trades is not None
        trades.append(
            {
                "timestamp": timestamp,
                "bar_end": panel.timestamps[bar],
                "symbol": panel.symbols[asset],
                "side": side,
                "requested": requested,
                "filled": quantity,
                "price": price,
                "notional": notional,
                "fees": fees.total,
                **dataclasses.asdict(fees),
                "cash": book.cash,
                "reason": "t0_restore_base"
                if exit_phase
                else "target_rebalance",
            }
        )
    return finish(notional, fees.total, quantity)


def _simulate(
    panel: models.MarketPanel,
    signal: models.SignalMatrix,
    config: backtest_config.BacktestConfig,
    indices: np.ndarray,
    frictionless: bool,
    trades: audit.AuditRecords | None = None,
    restrictions: audit.AuditRecords | None = None,
) -> pd.DataFrame:
    """Run one independently funded account using only completed signals."""
    assets = len(panel.symbols)
    book = _Book(
        config.initial_capital,
        np.zeros(assets),
        [collections.deque() for _ in range(assets)],
        np.zeros(assets),
        np.ones(assets, dtype=int),
    )
    mark = np.full(assets, np.nan)
    if indices[0] > 0:
        for bar in range(indices[0]):
            valid = np.isfinite(panel.close[bar]) & (panel.close[bar] > 0)
            mark[valid] = panel.close[bar, valid]
    values = np.full(assets, np.nan)
    previous_base = np.full(assets, np.nan)
    initialized = False
    flattening = False
    previous_valuation = np.zeros(assets)
    rows = []
    flatten_time = datetime.time.fromisoformat(config.t0_flatten_at)
    for bar in indices:
        dividends = float(np.dot(book.shares, panel.cash_dividend[bar]))
        book.cash += dividends
        multipliers = panel.share_multiplier[bar]
        book.shares *= multipliers
        book.desired *= multipliers
        mark /= multipliers
        for asset in np.flatnonzero(multipliers != 1):
            for lot in book.lots[asset]:
                lot[1] *= multipliers[asset]
        open_prices = panel.open[bar]
        usable = np.isfinite(open_prices) & (open_prices > 0)
        valuation = np.where(usable, open_prices, mark)
        valuation = np.nan_to_num(valuation)
        nav = book.cash + float(np.dot(book.shares, valuation))
        base = panel.base_weights[bar]
        changed = not initialized or not np.array_equal(base, previous_base)
        changed |= bool(
            np.any((valuation > 0) & (previous_valuation <= 0) & (base > 0))
        )
        if panel.is_session_start[bar]:
            flattening = False
        prior = bar - 1
        known_prior = (
            prior >= 0 and panel.timestamps[prior] <= panel.open_timestamps[bar]
        )
        if config.mode == "T1" and panel.is_session_start[bar]:
            if known_prior and panel.session_id[prior] < panel.session_id[bar]:
                values = signal.values[prior]
            else:
                values = np.full(assets, np.nan)
            changed = True
        if config.mode == "T0":
            changed |= bool(panel.is_session_start[bar])
            was_flattening = flattening
            flattening |= panel.open_timestamps[
                bar
            ].time() >= flatten_time or bool(panel.is_session_end[bar])
            if flattening:
                values = np.full(assets, np.nan)
                changed |= not was_flattening or panel.is_session_end[bar]
            elif known_prior and (
                not initialized or np.any(signal.updated[prior])
            ):
                values = signal.values[prior]
                changed = True
        if changed:
            weights = allocation.target_weights(
                base, values, config.keep, config.direction
            )
            book.desired = np.divide(
                weights * nav,
                valuation,
                out=np.zeros(assets),
                where=valuation > 0,
            )
            slices = (
                1
                if flattening and panel.is_session_end[bar]
                else (config.order_slices)
            )
            book.slices[:] = slices
            previous_base = base.copy()
            initialized = True
        consideration, fees = 0.0, 0.0
        sells = np.flatnonzero(book.desired < book.shares - 1e-8)
        buys = np.flatnonzero(book.desired > book.shares + 1e-8)
        floor_shares = np.divide(
            config.keep * base * nav,
            valuation,
            out=np.zeros(assets),
            where=valuation > 0,
        )
        if len(buys):
            priorities = np.maximum(floor_shares[buys] - book.shares[buys], 0)
            buys = buys[np.argsort(-priorities, kind="stable")]
        for asset in np.concatenate((sells, buys)):
            traded, charged = _execute(
                book,
                panel,
                bar,
                asset,
                config,
                frictionless,
                trades,
                restrictions,
                exit_phase=config.mode == "T0" and flattening,
            )
            consideration += traded
            fees += charged
        if not frictionless:
            assert restrictions is not None
            executed_nav = book.cash + float(np.dot(book.shares, valuation))
            actual_floor = np.divide(
                config.keep * base * executed_nav,
                valuation,
                out=np.zeros(assets),
                where=valuation > 0,
            )
            for asset in np.flatnonzero((valuation <= 0) & (base > 0)):
                _record(
                    restrictions,
                    panel,
                    bar,
                    asset,
                    "buy",
                    0,
                    0,
                    "cannot_size_missing_quote",
                )
            for asset in np.flatnonzero(book.shares < actual_floor - 1e-8):
                _record(
                    restrictions,
                    panel,
                    bar,
                    asset,
                    "buy",
                    actual_floor[asset] - book.shares[asset],
                    0,
                    "base_floor_shortfall",
                )
            if config.mode == "T0" and panel.is_session_end[bar]:
                residual = book.desired - book.shares
                for asset in np.flatnonzero(abs(residual) > 1e-8):
                    _record(
                        restrictions,
                        panel,
                        bar,
                        asset,
                        "buy" if residual[asset] > 0 else "sell",
                        abs(residual[asset]),
                        0,
                        "t0_end_residual",
                    )
        close = panel.close[bar]
        valid = np.isfinite(close) & (close > 0)
        mark[usable] = open_prices[usable]
        mark[valid] = close[valid]
        if np.any((book.shares > 0) & ~np.isfinite(mark)):
            raise ValueError("Held security has no available valuation quote")
        position_value = float(np.dot(book.shares, np.nan_to_num(mark)))
        rows.append(
            {
                "timestamp": panel.timestamps[bar],
                "nav": book.cash + position_value,
                "cash": book.cash,
                "position_value": position_value,
                "turnover": consideration / nav if nav else 0.0,
                "fees": fees,
                "dividends": dividends,
            }
        )
        previous_valuation = valuation
    return pd.DataFrame(rows)


def run_backtest(
    panel: models.MarketPanel,
    signal: models.SignalMatrix,
    config: backtest_config.BacktestConfig,
    audit_dir: pathlib.Path | None = None,
) -> models.BacktestResult:
    """Backtest one factor and a separate frictionless shadow account.

    Orders execute at a bar's open using earlier completed-bar signals.
    Supplied bar volume caps simulated fills; it is an ex-post liquidity
    constraint, not information used to generate the order. T1 holding
    locks are strategy rules, not Hong Kong legal settlement restrictions.
    The base floor applies to target weights. Lots, liquidity and cash may
    prevent achieving it; every observed shortfall is recorded.

    Args:
        panel: Raw prices, point-in-time constituents and trading events.
        signal: A single aligned factor known as of each completed bar.
        config: Date bounds, strategy and execution assumptions.
        audit_dir: Optional streaming trades/restrictions destination. Each
            audit buffer retains at most 4096 records. Returned audit frames
            are typed and empty when records are streamed to Parquet files.

    Returns:
        Net/gross accounts, filled orders, restrictions and summary metrics.

    Raises:
        ValueError: Data/configuration are incompatible or incomplete.
    """
    if signal.values.shape != panel.shape:
        raise ValueError("Signal values must match the market panel shape")
    if signal.updated.shape not in {(panel.shape[0],), panel.shape}:
        raise ValueError("Signal updated flags must align with panel bars")
    if panel.frequency != config.frequency:
        raise ValueError("Configured frequency must match panel frequency")
    if config.slippage_bps >= 10_000:
        raise ValueError("Slippage must be less than 10000 basis points")
    if panel.price_basis != "raw" and not config.allow_adjusted_prices:
        raise ValueError(
            "Raw execution prices required; research opt-in needed"
        )
    days = panel.timestamps.normalize().tz_localize(None)
    selected = (days >= pd.Timestamp(config.t_start).normalize()) & (
        days <= pd.Timestamp(config.t_end).normalize()
    )
    indices = np.flatnonzero(selected)
    if not len(indices):
        raise ValueError("No market bars fall within the requested dates")
    if panel.open_timestamps[indices[0]].date().isoformat() < "2014-11-01":
        raise ValueError("Transaction fees before 2014-11-01 unsupported")
    if config.enforce_vcm:
        included = slice(indices[0], indices[-1] + 1)
        complete = np.isfinite(panel.vcm_lower[included]) & np.isfinite(
            panel.vcm_upper[included]
        )
        eligible = panel.vcm_eligible[included]
        states = getattr(panel, "vcm_active", None)
        requires_bands = (
            eligible
            if states is None
            else (
                (states[included] == 1) | (eligible & (states[included] == -1))
            )
        )
        missing = requires_bands & ~complete
        if np.any(missing):
            raise ValueError(
                "Strict VCM requires complete bands for active "
                "events or unknown eligible states"
            )
    if audit_dir is not None:
        audit_dir = pathlib.Path(audit_dir)
        for filename in ("trades.parquet", "restrictions.parquet"):
            if (audit_dir / filename).exists():
                raise FileExistsError(audit_dir / filename)
    with contextlib.ExitStack() as stack:
        trades = stack.enter_context(audit.AuditRecords("trades", audit_dir))
        restrictions = stack.enter_context(
            audit.AuditRecords("restrictions", audit_dir)
        )
        net = _simulate(
            panel, signal, config, indices, False, trades, restrictions
        )
        gross = _simulate(panel, signal, config, indices, True)
    pnl = net.rename(columns={"nav": "net_nav"})
    pnl["gross_nav"] = gross["nav"]
    for prefix in ("net", "gross"):
        previous = pnl[f"{prefix}_nav"].shift(1).fillna(config.initial_capital)
        pnl[f"{prefix}_pnl"] = pnl[f"{prefix}_nav"] - previous
        pnl[f"{prefix}_return"] = pnl[f"{prefix}_nav"] / previous - 1
    benchmark = pd.Series(panel.benchmark_close, dtype=float).ffill()
    selected_benchmark = benchmark.iloc[indices].reset_index(drop=True)
    previous = selected_benchmark.shift(1)
    benchmark_open = getattr(panel, "benchmark_open", None)
    opening = np.nan if benchmark_open is None else benchmark_open[indices[0]]
    if np.isfinite(opening) and opening > 0:
        previous.iloc[0] = opening
        benchmark_start_policy = "first_execution_open"
    else:
        previous.iloc[0] = selected_benchmark.iloc[0]
        benchmark_start_policy = "first_close_missing_open"
    pnl["benchmark_return"] = selected_benchmark / previous - 1
    pnl["excess_return"] = pnl["net_return"] - pnl["benchmark_return"]
    pnl["benchmark_nav"] = (
        config.initial_capital * (1 + pnl["benchmark_return"]).cumprod()
    )
    pnl["excess_nav"] = pnl["net_nav"] / pnl["benchmark_nav"]
    final = pnl.iloc[-1]
    daily_nav = pnl.set_index("timestamp")["net_nav"].resample("D").last()
    daily_nav = daily_nav.dropna()
    daily_returns = (
        daily_nav / daily_nav.shift(1).fillna(config.initial_capital) - 1
    )
    volatility = daily_returns.std(ddof=1)
    sharpe = (
        float(daily_returns.mean() / volatility * np.sqrt(252))
        if volatility > 0
        else float("nan")
    )
    running_max = np.maximum.accumulate(
        np.r_[config.initial_capital, pnl["net_nav"].to_numpy()]
    )
    drawdown = (
        np.r_[config.initial_capital, pnl["net_nav"].to_numpy()] / running_max
        - 1
    )
    metrics = {
        "factor": signal.factor,
        "mode": config.mode,
        "net_total_return": final["net_nav"] / config.initial_capital - 1,
        "gross_total_return": final["gross_nav"] / config.initial_capital - 1,
        "benchmark_total_return": (
            final["benchmark_nav"] / config.initial_capital - 1
        ),
        "benchmark_start_policy": benchmark_start_policy,
        "excess_total_return": final["excess_nav"] - 1,
        "net_pnl": final["net_nav"] - config.initial_capital,
        "gross_pnl": final["gross_nav"] - config.initial_capital,
        "fees": float(pnl["fees"].sum()),
        "turnover": float(pnl["turnover"].sum()),
        "max_drawdown": float(drawdown.min()),
        "sharpe": sharpe,
        "trade_count": trades.count,
        "restriction_count": restrictions.count,
        "price_basis": panel.price_basis,
        "adjusted_price_research": panel.price_basis != "raw",
        "gross_method": "independent_zero_cost_zero_slippage_account",
        "vcm_policy": "strict_supplied_state_and_bands"
        if (config.enforce_vcm)
        else "supplied_events_missing_triggers_unknown",
    }
    return models.BacktestResult(
        pnl,
        trades.to_frame(),
        restrictions.to_frame(),
        metrics,
        audit_streamed=audit_dir is not None,
    )
