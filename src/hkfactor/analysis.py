"""Cross-sectional research diagnostics separate from execution accounts."""

from collections.abc import Iterator

import numpy as np
import pandas as pd

from hkfactor import config as backtest_config
from hkfactor import models

_CHUNK_SIZE = 4096
_IC_COLUMNS = [
    "timestamp",
    "signal_timestamp",
    "forward_end",
    "entry_n_assets",
    "n_assets",
    "tied_assets",
    "pearson_ic",
    "spearman_ic",
    "reason",
]
_LAYER_COLUMNS = [
    "timestamp",
    "date",
    "layer",
    "n_assets",
    "forward_n_assets",
    "one_bar_n_assets",
    "mean_forward_return",
    "mean_one_bar_return",
    "gross_research_nav",
]


def _groups(
    values: np.ndarray,
    eligible: np.ndarray,
    quantiles: int,
    symbol_order: np.ndarray,
) -> np.ndarray:
    """Assign ascending groups with deterministic symbol ordering for ties."""
    counts = eligible.sum(axis=1)
    ordered = np.where(
        eligible[:, symbol_order], values[:, symbol_order], np.inf
    )
    sorted_columns = np.argsort(ordered, axis=1, kind="stable")
    ranks = np.empty_like(sorted_columns)
    np.put_along_axis(
        ranks,
        sorted_columns,
        np.broadcast_to(np.arange(values.shape[1]), ranks.shape),
        axis=1,
    )
    labels = (ranks * quantiles // np.maximum(counts[:, None], 1)) + 1
    labels[~eligible[:, symbol_order]] = 0
    labels[counts < quantiles] = 0
    groups = np.empty_like(labels)
    groups[:, symbol_order] = labels
    return groups


def iter_layer_signals(
    panel: models.MarketPanel,
    signal: models.SignalMatrix,
    config: backtest_config.BacktestConfig,
) -> Iterator[tuple[int, models.SignalMatrix]]:
    """Yield one layer's binary signals for separate execution accounts.

    Groups use finite signals and membership known at each completed bar.
    Layer 1 has the lowest raw factor values. Equal values are ordered by
    symbol. Bars with fewer than ``quantiles`` members produce no signals.
    The execution engine subsequently applies its own signal delay,
    ``keep`` floor, costs, and trading constraints.

    Args:
        panel: Point-in-time market and membership arrays.
        signal: One factor known as of each bar end.
        config: Group count and execution assumptions.

    Yields:
        Layer number and its signal matrix. Only one layer matrix is
        allocated at a time; callers should release it before advancing.
    """
    _validate_shapes(panel, signal)
    rows, assets = panel.shape
    groups = np.zeros((rows, assets), dtype=np.int32)
    symbol_order = np.argsort(np.asarray(panel.symbols), kind="stable")
    for start in range(0, rows, _CHUNK_SIZE):
        stop = min(start + _CHUNK_SIZE, rows)
        values = signal.values[start:stop]
        eligible = (panel.base_weights[start:stop] > 0) & np.isfinite(values)
        groups[start:stop] = _groups(
            values,
            eligible,
            config.quantiles,
            symbol_order,
        )
    updated = np.array(signal.updated, dtype=bool, copy=True)
    if rows > 1:
        updated[1:] |= np.any(
            panel.base_weights[1:] != panel.base_weights[:-1],
            axis=1,
        )
    for layer in range(1, config.quantiles + 1):
        values = np.full(panel.shape, np.nan, dtype=np.float64)
        values[groups == layer] = 1.0
        yield (
            layer,
            models.SignalMatrix(
                factor=f"{signal.factor}:layer_{layer:02d}",
                values=values,
                updated=updated,
            ),
        )
        del values


def _validate_shapes(
    panel: models.MarketPanel, signal: models.SignalMatrix
) -> None:
    """Reject accidental factor/market misalignment."""
    if signal.values.shape != panel.shape:
        raise ValueError("Signal and market dimensions do not match")
    if np.asarray(signal.updated).shape != (panel.shape[0],):
        raise ValueError("Signal updated flags must have one value per bar")


def _correlation(values: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Calculate row-wise Pearson correlation, preserving missing pairs."""
    valid = np.isfinite(values) & np.isfinite(returns)
    count = valid.sum(axis=1)
    denominator = np.maximum(count, 1)
    x_mean = np.where(valid, values, 0).sum(axis=1) / denominator
    y_mean = np.where(valid, returns, 0).sum(axis=1) / denominator
    x = np.where(valid, values - x_mean[:, None], 0)
    y = np.where(valid, returns - y_mean[:, None], 0)
    numerator = np.einsum("ij,ij->i", x, y)
    scale = np.sqrt(np.einsum("ij,ij->i", x, x) * np.einsum("ij,ij->i", y, y))
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / scale
    result[(count < 2) | (scale == 0)] = np.nan
    return np.clip(result, -1.0, 1.0)


def _forward_returns(
    panel: models.MarketPanel,
    entries: np.ndarray,
    horizon: int,
    config: backtest_config.BacktestConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return entry-open to future-close total returns and window validity."""
    ends = entries + horizon - 1
    valid_window = ends < panel.shape[0]
    safe_ends = np.minimum(ends, panel.shape[0] - 1)
    end_dates = panel.timestamps[safe_ends].date
    valid_window &= end_dates <= pd.Timestamp(config.t_end).date()
    if config.mode == "T0":
        valid_window &= panel.session_id[entries] == panel.session_id[safe_ends]
    shares = np.ones((len(entries), panel.shape[1]), dtype=np.float64)
    dividends = np.zeros_like(shares)
    for offset in range(1, horizon):
        action_rows = np.minimum(entries + offset, panel.shape[0] - 1)
        dividends += shares * panel.cash_dividend[action_rows]
        shares *= panel.share_multiplier[action_rows]
    prices = panel.open[entries]
    exit_prices = panel.close[safe_ends]
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = (exit_prices * shares + dividends) / prices - 1
    valid_quote = (
        np.isfinite(exit_prices) & (exit_prices > 0) & panel.tradable[safe_ends]
    )
    returns[~valid_quote | ~valid_window[:, None]] = np.nan
    return returns, valid_window, safe_ends


def _tie_count(values: np.ndarray) -> np.ndarray:
    """Count observations belonging to a repeated finite value per row."""
    ordered = np.sort(values, axis=1)
    equal = ordered[:, 1:] == ordered[:, :-1]
    equal &= np.isfinite(ordered[:, 1:])
    tied = np.zeros(ordered.shape, dtype=bool)
    tied[:, 1:] |= equal
    tied[:, :-1] |= equal
    return tied.sum(axis=1)


def _ic_frame(
    panel: models.MarketPanel,
    entries: np.ndarray,
    values: np.ndarray,
    eligible: np.ndarray,
    returns: np.ndarray,
    valid_window: np.ndarray,
    ends: np.ndarray,
    config: backtest_config.BacktestConfig,
) -> pd.DataFrame:
    """Build IC observations with explicit missing/constant reasons."""
    paired = eligible & np.isfinite(returns)
    x = np.where(paired, values, np.nan)
    y = np.where(paired, returns, np.nan)
    count = paired.sum(axis=1)
    pearson = _correlation(x, y)
    x_rank = pd.DataFrame(x).rank(axis=1, method="average").to_numpy()
    y_rank = pd.DataFrame(y).rank(axis=1, method="average").to_numpy()
    spearman = _correlation(x_rank, y_rank)
    with np.errstate(invalid="ignore"):
        constant_x = np.where(paired, x, -np.inf).max(axis=1) == np.where(
            paired, x, np.inf
        ).min(axis=1)
        constant_y = np.where(paired, y, -np.inf).max(axis=1) == np.where(
            paired, y, np.inf
        ).min(axis=1)
    reason = np.full(len(entries), "ok", dtype=object)
    reason[constant_y] = "constant_return"
    reason[constant_x] = "constant_signal"
    reason[count < config.ic_min_assets] = "insufficient_assets"
    reason[~valid_window] = "invalid_forward_window"
    usable = reason == "ok"
    pearson[~usable] = np.nan
    spearman[~usable] = np.nan
    return pd.DataFrame(
        {
            "timestamp": panel.open_timestamps[entries],
            "signal_timestamp": panel.timestamps[entries - 1],
            "forward_end": panel.timestamps[ends],
            "entry_n_assets": eligible.sum(axis=1),
            "n_assets": count,
            "tied_assets": _tie_count(np.where(eligible, values, np.nan)),
            "pearson_ic": pearson,
            "spearman_ic": spearman,
            "reason": reason,
        }
    )


def _layer_frame(
    panel: models.MarketPanel,
    entries: np.ndarray,
    groups: np.ndarray,
    returns: np.ndarray,
    one_bar_returns: np.ndarray,
    config: backtest_config.BacktestConfig,
    previous_nav: np.ndarray,
) -> pd.DataFrame:
    """Build gross research observations without compounding overlaps."""
    output = []
    for layer in range(1, config.quantiles + 1):
        members = groups == layer
        count = members.sum(axis=1)
        forward_valid = members & np.isfinite(returns)
        one_bar_valid = members & np.isfinite(one_bar_returns)
        forward_count = forward_valid.sum(axis=1)
        one_bar_count = one_bar_valid.sum(axis=1)
        forward_mean = np.divide(
            np.where(forward_valid, returns, 0).sum(axis=1),
            forward_count,
            out=np.full(len(entries), np.nan),
            where=forward_count > 0,
        )
        one_bar_mean = np.divide(
            np.where(one_bar_valid, one_bar_returns, 0).sum(axis=1),
            one_bar_count,
            out=np.full(len(entries), np.nan),
            where=one_bar_count > 0,
        )
        # Complete same-bar baskets form a non-overlapping research curve.
        # Incomplete valuation makes the subsequent NAV indeterminate.
        curve_return = np.where(
            (count > 0) & (one_bar_count == count),
            one_bar_mean,
            np.nan,
        )
        curve_return[count == 0] = 0.0
        nav = previous_nav[layer - 1] * np.cumprod(1 + curve_return)
        previous_nav[layer - 1] = nav[-1]
        present = count > 0
        output.append(
            pd.DataFrame(
                {
                    "timestamp": panel.open_timestamps[entries[present]],
                    "date": panel.open_timestamps[entries[present]].strftime(
                        "%Y-%m-%d",
                    ),
                    "layer": layer,
                    "n_assets": count[present],
                    "forward_n_assets": forward_count[present],
                    "one_bar_n_assets": one_bar_count[present],
                    "mean_forward_return": forward_mean[present],
                    "mean_one_bar_return": one_bar_mean[present],
                    "gross_research_nav": nav[present],
                }
            )
        )
    return pd.concat(output, ignore_index=True)


def analyze_factor(
    panel: models.MarketPanel,
    signal: models.SignalMatrix,
    config: backtest_config.BacktestConfig,
) -> models.AnalysisResult:
    """Analyze previous-completed-bar signals against future gross returns.

    IC is diagnostic, independent of the execution strategy's holding
    rules. Returns run from the next bar's open to the horizon's close,
    include supplied intervening dividends and share multipliers, and
    exclude cross-session T0 windows. Pearson and average-rank Spearman
    IC use the same finite point-in-time constituent sample. Missing or
    untradable entry/exit quotes are excluded and counts are retained.

    Layers use entry-time membership and ascending raw signals. Their
    forward returns are never compounded when horizons overlap. The
    separate gross research curve compounds complete one-bar baskets,
    excluding overnight returns and costs. Executed layer accounts, if
    requested, belong to ``AnalysisResult.layer_pnl`` instead.

    Args:
        panel: Shared market, corporate actions, and membership arrays.
        signal: One externally generated factor.
        config: Date range, mode, horizon, and sample-size assumptions.

    Returns:
        Time-indexed IC, local intraday IC, layers, and layer statistics.
    """
    _validate_shapes(panel, signal)
    dates = panel.open_timestamps.date
    in_range = (dates >= pd.Timestamp(config.t_start).date()) & (
        dates <= pd.Timestamp(config.t_end).date()
    )
    entries = np.flatnonzero(in_range & (np.arange(panel.shape[0]) > 0))
    ic_parts = []
    layer_parts = []
    symbol_order = np.argsort(np.asarray(panel.symbols), kind="stable")
    nav = np.ones(config.quantiles)
    for start in range(0, len(entries), _CHUNK_SIZE):
        chunk_entries = entries[start : start + _CHUNK_SIZE]
        values = signal.values[chunk_entries - 1]
        open_prices = panel.open[chunk_entries]
        eligible = (
            (panel.base_weights[chunk_entries] > 0)
            & panel.tradable[chunk_entries]
            & (panel.volume[chunk_entries] > 0)
            & np.isfinite(open_prices)
            & (open_prices > 0)
            & np.isfinite(values)
        )
        returns, valid_window, ends = _forward_returns(
            panel,
            chunk_entries,
            config.ic_horizon_bars,
            config,
        )
        one_bar_returns, _, _ = _forward_returns(
            panel,
            chunk_entries,
            1,
            config,
        )
        ic_parts.append(
            _ic_frame(
                panel,
                chunk_entries,
                values,
                eligible,
                returns,
                valid_window,
                ends,
                config,
            )
        )
        groups = _groups(values, eligible, config.quantiles, symbol_order)
        layer_parts.append(
            _layer_frame(
                panel,
                chunk_entries,
                groups,
                returns,
                one_bar_returns,
                config,
                nav,
            )
        )
    ic = (
        pd.concat(ic_parts, ignore_index=True)
        if ic_parts
        else pd.DataFrame(columns=_IC_COLUMNS)
    )
    layers = (
        pd.concat(layer_parts, ignore_index=True)
        if layer_parts
        else pd.DataFrame(columns=_LAYER_COLUMNS)
    )
    intraday_columns = [
        "local_time",
        "pearson_mean",
        "pearson_count",
        "pearson_std",
        "spearman_mean",
        "spearman_count",
        "spearman_std",
    ]
    if ic.empty:
        intraday = pd.DataFrame(columns=intraday_columns)
    else:
        local_time = (
            ic["timestamp"]
            .dt.tz_convert(
                "Asia/Hong_Kong",
            )
            .dt.strftime("%H:%M")
        )
        intraday = (
            ic.groupby(local_time)
            .agg(
                pearson_mean=("pearson_ic", "mean"),
                pearson_count=("pearson_ic", "count"),
                pearson_std=("pearson_ic", "std"),
                spearman_mean=("spearman_ic", "mean"),
                spearman_count=("spearman_ic", "count"),
                spearman_std=("spearman_ic", "std"),
            )
            .rename_axis("local_time")
            .reset_index()
        )
    if layers.empty:
        summary = pd.DataFrame(
            columns=[
                "layer",
                "bars",
                "asset_observations",
                "forward_observations",
                "mean_forward_return",
                "mean_one_bar_return",
            ]
        )
    else:
        summary = (
            layers.groupby("layer")
            .agg(
                bars=("timestamp", "size"),
                asset_observations=("n_assets", "sum"),
                forward_observations=("forward_n_assets", "sum"),
                mean_forward_return=("mean_forward_return", "mean"),
                mean_one_bar_return=("mean_one_bar_return", "mean"),
            )
            .reset_index()
        )
    assumptions = {
        "signal_timing": "previous completed bar; next bar open entry",
        "ic_horizon_bars": config.ic_horizon_bars,
        "ic_min_assets": config.ic_min_assets,
        "return_basis": "gross shareholder return with supplied actions",
        "price_basis": panel.price_basis,
        "layer_order": "ascending raw factor; symbol-ordered ties",
        "research_nav": "non-overlapping one-bar open-to-close baskets; "
        "no overnight return or costs; incomplete "
        "valuation invalidates subsequent NAV",
        "account_pnl": "execution-account layer PnL is layer_pnl",
    }
    ic.attrs["assumptions"] = assumptions
    return models.AnalysisResult(layers, ic, intraday, summary)
