"""Verify research timing, membership, ties, and non-overlapping curves."""

import json

import numpy as np
import pandas as pd
import pytest

from hkfactor import analysis as factor_analysis
from hkfactor import config as backtest_config
from hkfactor import models
from hkfactor import reporting


def _panel(assets=10, times=None, symbols=None):
    """Build complete intraday market quotes with known membership."""
    if times is None:
        times = [
            "2026-01-05 09:35",
            "2026-01-05 09:40",
            "2026-01-05 09:45",
            "2026-01-05 09:50",
        ]
    timestamps = pd.DatetimeIndex(times, tz="Asia/Hong_Kong")
    shape = (len(times), assets)
    prices = np.full(shape, 100.0)
    close = prices * (1 + np.arange(1, assets + 1) / 100)
    session_id = pd.factorize(timestamps.date)[0]
    start = np.r_[True, session_id[1:] != session_id[:-1]]
    end = np.r_[session_id[1:] != session_id[:-1], True]
    return models.MarketPanel(
        timestamps=timestamps,
        open_timestamps=timestamps - pd.Timedelta(minutes=5),
        symbols=tuple(
            symbols or (f"{index:05d}.HK" for index in range(assets))
        ),
        frequency="5m",
        open=prices.copy(),
        high=close.copy(),
        low=prices.copy(),
        close=close,
        volume=np.full(shape, 10_000.0),
        base_weights=np.full(shape, 1 / assets),
        benchmark_close=np.ones(len(times)),
        lot_sizes=np.ones(shape),
        tradable=np.ones(shape, dtype=bool),
        vcm_lower=np.full(shape, np.nan),
        vcm_upper=np.full(shape, np.nan),
        vcm_eligible=np.zeros(shape, dtype=bool),
        cash_dividend=np.zeros(shape),
        share_multiplier=np.ones(shape),
        session_id=session_id,
        is_session_start=start,
        is_session_end=end,
    )


def _signal(panel, constant=False):
    """Create either ascending or constant signals known at each close."""
    values = np.ones(panel.shape)
    if not constant:
        values *= np.arange(panel.shape[1])
    return models.SignalMatrix(
        "test_factor", values, np.ones(panel.shape[0], dtype=bool)
    )


def _config(**changes):
    """Return an explicit intraday configuration for diagnostics."""
    values = dict(
        t_start="2026-01-05", t_end="2026-01-06", mode="T0", frequency="5m"
    )
    values.update(changes)
    return backtest_config.BacktestConfig(**values)


def test_ic_uses_previous_completed_bar():
    panel = _panel()
    signal = _signal(panel)
    signal.values[1] *= -1
    result = factor_analysis.analyze_factor(panel, signal, _config())
    assert result.ic.iloc[0].pearson_ic == pytest.approx(1)
    assert result.ic.iloc[1].pearson_ic == pytest.approx(-1)
    assert result.ic.iloc[0].signal_timestamp == panel.timestamps[0]
    assert result.ic.iloc[0].timestamp == panel.open_timestamps[1]
    assert result.ic.iloc[0].forward_end == panel.timestamps[1]


def test_constant_signals_have_explicit_reason():
    panel = _panel()
    result = factor_analysis.analyze_factor(
        panel, _signal(panel, True), _config()
    )
    assert result.ic.pearson_ic.isna().all()
    assert result.ic.spearman_ic.isna().all()
    assert set(result.ic.reason) == {"constant_signal"}
    assert (result.ic.tied_assets == 10).all()


def test_ties_are_ordered_by_symbol_and_not_array_position():
    symbols = tuple(f"{index:05d}.HK" for index in reversed(range(10)))
    panel = _panel(symbols=symbols)
    signal = _signal(panel, True)
    result = factor_analysis.analyze_factor(panel, signal, _config())
    first = result.layers[
        result.layers.timestamp == panel.open_timestamps[1]
    ].set_index("layer")
    assert first.loc[1, "mean_one_bar_return"] == pytest.approx(0.10)
    assert first.loc[10, "mean_one_bar_return"] == pytest.approx(0.01)
    iterator = factor_analysis.iter_layer_signals(panel, signal, _config())
    layer, layer_signal = next(iterator)
    assert layer == 1
    assert np.flatnonzero(np.isfinite(layer_signal.values[0]))[0] == 9


def test_small_universe_does_not_manufacture_ten_layers():
    panel = _panel(assets=9)
    signal = _signal(panel)
    result = factor_analysis.analyze_factor(panel, signal, _config())
    assert result.layers.empty
    assert result.layer_summary.empty
    assert (result.ic.n_assets == 9).all()
    assert next(factor_analysis.iter_layer_signals(panel, signal, _config()))[
        1
    ].values[0].shape == (9,)
    assert np.isnan(
        next(factor_analysis.iter_layer_signals(panel, signal, _config()))[
            1
        ].values
    ).all()


def test_minimum_ic_samples_preserves_counts():
    panel = _panel(assets=2)
    result = factor_analysis.analyze_factor(panel, _signal(panel), _config())
    assert (result.ic.n_assets == 2).all()
    assert set(result.ic.reason) == {"insufficient_assets"}
    assert result.ic.pearson_ic.isna().all()


def test_future_membership_never_changes_current_groups():
    panel = _panel(assets=11)
    panel.base_weights[1, -1] = 0
    result = factor_analysis.analyze_factor(
        panel, _signal(panel), _config(ic_horizon_bars=2)
    )
    first = result.layers[result.layers.timestamp == panel.open_timestamps[1]]
    assert first.n_assets.sum() == 10
    assert first.loc[first.layer == 10, "mean_forward_return"].iloc[
        0
    ] == pytest.approx(0.10)
    assert result.ic.iloc[0].n_assets == 10


def test_entry_zero_volume_and_missing_quotes_are_excluded():
    panel = _panel()
    panel.volume[1, 0] = 0
    panel.tradable[1, 1] = False
    panel.open[1, 2] = np.nan
    result = factor_analysis.analyze_factor(panel, _signal(panel), _config())
    assert result.ic.iloc[0].entry_n_assets == 7
    assert result.layers.timestamp.min() == panel.open_timestamps[2]


def test_future_missing_quotes_do_not_change_group_membership():
    panel = _panel()
    panel.tradable[2, 0] = False
    result = factor_analysis.analyze_factor(
        panel, _signal(panel), _config(ic_horizon_bars=2)
    )
    first = result.layers[result.layers.timestamp == panel.open_timestamps[1]]
    assert first.n_assets.sum() == 10
    assert first.forward_n_assets.sum() == 9
    assert result.ic.iloc[0].entry_n_assets == 10
    assert result.ic.iloc[0].n_assets == 9


def test_t0_rejects_cross_session_window_but_t1_allows_it():
    panel = _panel(
        times=[
            "2026-01-05 15:50",
            "2026-01-05 15:55",
            "2026-01-06 09:35",
            "2026-01-06 09:40",
        ]
    )
    signal = _signal(panel)
    t0 = factor_analysis.analyze_factor(
        panel, signal, _config(ic_horizon_bars=2)
    )
    t1 = factor_analysis.analyze_factor(
        panel, signal, _config(mode="T1", ic_horizon_bars=2)
    )
    assert t0.ic.iloc[0].reason == "invalid_forward_window"
    assert np.isnan(t0.ic.iloc[0].pearson_ic)
    assert t1.ic.iloc[0].pearson_ic == pytest.approx(1)


def test_end_date_and_warmup_filter_without_losing_known_signal():
    panel = _panel(
        times=[
            "2026-01-05 15:55",
            "2026-01-06 09:35",
            "2026-01-06 09:40",
            "2026-01-07 09:35",
        ]
    )
    result = factor_analysis.analyze_factor(
        panel,
        _signal(panel),
        _config(
            t_start="2026-01-06",
            t_end="2026-01-06",
            mode="T1",
            ic_horizon_bars=2,
        ),
    )
    assert len(result.ic) == 2
    assert result.ic.iloc[0].signal_timestamp == panel.timestamps[0]
    assert result.ic.iloc[0].reason == "ok"
    assert result.ic.iloc[1].reason == "invalid_forward_window"


def test_overlapping_forward_returns_are_not_compounded():
    panel = _panel()
    panel.close[2, -1] = 140
    result = factor_analysis.analyze_factor(
        panel, _signal(panel), _config(ic_horizon_bars=2)
    )
    layer = result.layers[result.layers.layer == 10].sort_values(
        "timestamp",
    )
    assert layer.iloc[0].mean_forward_return == pytest.approx(0.40)
    assert layer.iloc[0].gross_research_nav == pytest.approx(1.10)
    assert layer.iloc[1].gross_research_nav == pytest.approx(1.10 * 1.40)
    assert np.isnan(layer.iloc[-1].mean_forward_return)


def test_forward_window_includes_actions_after_entry_only():
    panel = _panel()
    panel.cash_dividend[1, :] = 100  # Already detached before entry.
    panel.cash_dividend[2, :] = 2
    panel.share_multiplier[2, :] = 2
    panel.close[2, :] /= 2
    result = factor_analysis.analyze_factor(
        panel, _signal(panel), _config(ic_horizon_bars=2)
    )
    first = result.layers[result.layers.timestamp == panel.open_timestamps[1]]
    assert first.loc[first.layer == 1, "mean_forward_return"].iloc[
        0
    ] == pytest.approx(0.03)


def test_intraday_ic_preserves_sample_counts():
    panel = _panel(
        times=[
            "2026-01-05 09:35",
            "2026-01-05 09:40",
            "2026-01-06 09:35",
            "2026-01-06 09:40",
        ]
    )
    result = factor_analysis.analyze_factor(panel, _signal(panel), _config())
    row = result.ic_intraday.set_index("local_time").loc["09:35"]
    assert row.pearson_count == 2
    assert row.pearson_mean == pytest.approx(1)
    assert row.spearman_count == 2


def test_reporting_separates_accounts_from_gross_research(tmp_path):
    panel = _panel()
    analysis = factor_analysis.analyze_factor(panel, _signal(panel), _config())
    analysis.layer_pnl = pd.DataFrame(
        {
            "timestamp": panel.timestamps,
            "net_nav": np.full(4, 1e7),
            "layer": np.ones(4, dtype=int),
        }
    )
    result = models.BacktestResult(
        pd.DataFrame({"net_nav": [1e7]}),
        pd.DataFrame(),
        pd.DataFrame(),
        {"unknown": np.nan},
    )
    reporting.write_report(
        result, analysis, tmp_path, {"config": _config().to_dict()}
    )
    assert (tmp_path / "layer_pnl.parquet").exists()
    assert (tmp_path / "layer_pnl.png").exists()
    assert (tmp_path / "ic_intraday.png").exists()
    assert (
        json.loads((tmp_path / "metrics.json").read_text())["unknown"] is None
    )
    metadata = json.loads((tmp_path / "run.json").read_text())
    assert metadata["analysis_assumptions"]["ic_horizon_bars"] == 1
    assert "not executed" in metadata["outputs"]["layers"]


def test_reporting_preserves_streamed_trade_and_restriction_audits(tmp_path):
    panel = _panel()
    analysis = factor_analysis.analyze_factor(panel, _signal(panel), _config())
    existing = {}
    for name in ["trades", "restrictions"]:
        path = tmp_path / f"{name}.parquet"
        pd.DataFrame({"audit": [name]}).to_parquet(path, index=False)
        existing[name] = path.read_bytes()
    result = models.BacktestResult(
        pd.DataFrame({"net_nav": [1e7]}),
        pd.DataFrame(),
        pd.DataFrame(),
        {},
        audit_streamed=True,
    )
    reporting.write_report(result, analysis, tmp_path, plots=False)
    for name, content in existing.items():
        assert (tmp_path / f"{name}.parquet").read_bytes() == content
