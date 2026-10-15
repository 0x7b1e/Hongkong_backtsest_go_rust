"""Causality, trading locks, constrained fills and account conservation."""

import dataclasses
import pathlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from hkfactor import allocation
from hkfactor import audit
from hkfactor import config as backtest_config
from hkfactor import execution
from hkfactor import models
from hkfactor import rules


def _panel(
    clocks: list[str],
    assets: int = 2,
    prices: np.ndarray | None = None,
    closes: np.ndarray | None = None,
    frequency: str = "5m",
) -> models.MarketPanel:
    """Create explicitly timestamped raw prices without provider adapters."""
    opens = pd.DatetimeIndex(clocks, tz="Asia/Hong_Kong")
    ends = (
        opens.normalize() + pd.Timedelta(hours=16)
        if frequency == "1d"
        else opens + pd.Timedelta(minutes=5)
    )
    shape = len(opens), assets
    prices = np.full(shape, 10.0) if prices is None else prices
    closes = prices.copy() if closes is None else closes
    session, _ = pd.factorize(opens.normalize())
    start = np.r_[True, session[1:] != session[:-1]]
    end = np.r_[session[1:] != session[:-1], True]
    return models.MarketPanel(
        timestamps=ends,
        open_timestamps=opens,
        symbols=tuple(f"{asset + 1:05}.HK" for asset in range(assets)),
        frequency=frequency,
        open=prices,
        high=np.maximum(prices, closes),
        low=np.minimum(prices, closes),
        close=closes,
        volume=np.full(shape, 1_000_000.0),
        base_weights=np.full(shape, 1 / assets),
        benchmark_close=np.full(len(opens), 100.0),
        lot_sizes=np.ones(shape),
        tradable=np.ones(shape, dtype=bool),
        vcm_lower=np.full(shape, np.nan),
        vcm_upper=np.full(shape, np.nan),
        vcm_eligible=np.zeros(shape, dtype=bool),
        cash_dividend=np.zeros(shape),
        share_multiplier=np.ones(shape),
        session_id=session,
        is_session_start=start,
        is_session_end=end,
    )


def _signal(
    panel: models.MarketPanel,
    values: np.ndarray | None = None,
) -> models.SignalMatrix:
    """Provide values completed at each bar end."""
    values = np.full(panel.shape, np.nan) if values is None else values
    return models.SignalMatrix(
        "test", values, np.ones(panel.shape[0], dtype=bool)
    )


def _config(
    panel: models.MarketPanel, **overrides
) -> backtest_config.BacktestConfig:
    """Use small funded accounts and unconstrained provider liquidity."""
    return backtest_config.BacktestConfig(
        t_start=str(panel.timestamps[0].date()),
        t_end=str(panel.timestamps[-1].date()),
        frequency=panel.frequency,
        initial_capital=10_000,
        participation_rate=1,
        commission_bps=0,
        **overrides,
    )


def test_target_floor_ties_and_unknown_signals():
    """Reserve exactly keep of every base weight and treat ties equally."""
    base = np.array([0.4, 0.4, 0.2])
    values = np.array([5.0, 5.0, np.nan])
    np.testing.assert_allclose(
        allocation.target_weights(base, values, 0.9, 1), [0.41, 0.41, 0.18]
    )
    np.testing.assert_allclose(
        allocation.target_weights(base, values, 0, 1), [0.5, 0.5, 0]
    )
    np.testing.assert_allclose(
        allocation.target_weights(base, values * np.nan, 0, 1), base
    )


def test_signal_at_execution_bar_end_cannot_change_earlier_trades():
    """Changing future signals leaves both earlier account paths unchanged."""
    panel = _panel(
        [
            "2024-01-02 09:30",
            "2024-01-02 09:35",
            "2024-01-02 09:40",
            "2024-01-02 15:55",
        ]
    )
    config = _config(panel, mode="T0", keep=0)
    values = np.array([[0, 10], [0, 10], [0, 10], [0, 10]], dtype=float)
    first = execution.run_backtest(panel, _signal(panel, values), config)
    values[1:] = [10, 0]
    second = execution.run_backtest(panel, _signal(panel, values), config)
    cutoff = panel.open_timestamps[2]
    pd.testing.assert_frame_equal(
        first.trades[first.trades.timestamp < cutoff],
        second.trades[second.trades.timestamp < cutoff],
    )
    assert not first.trades.equals(second.trades)


def test_t1_holding_lock_is_counted_in_supplied_trading_sessions():
    """Intraday sales of newly purchased shares are rejected by T1 policy."""
    panel = _panel(
        [
            "2024-01-05 09:30",
            "2024-01-05 09:35",
            "2024-01-08 09:30",
            "2024-01-08 09:35",
        ]
    )
    base = panel.base_weights.copy()
    base[1:] = [1, 0]
    panel = dataclasses.replace(panel, base_weights=base)
    result = execution.run_backtest(
        panel, _signal(panel), _config(panel, mode="T1", keep=1)
    )
    assert "holding_period_locked" in result.restrictions.reason.values
    sells = result.trades[result.trades.side == "sell"]
    assert len(sells) == 1
    assert sells.iloc[0].timestamp.date().isoformat() == "2024-01-08"


def test_t1_signal_is_frozen_from_previous_session_close():
    """T1 ignores all same-session updates despite available intraday bars."""
    panel = _panel(
        [
            "2024-01-02 09:30",
            "2024-01-02 09:35",
            "2024-01-03 09:30",
            "2024-01-03 09:35",
            "2024-01-03 09:40",
        ]
    )
    values = np.tile([0.0, 10.0], (5, 1))
    config = _config(panel, mode="T1", keep=0)
    first = execution.run_backtest(panel, _signal(panel, values), config)
    values[2:] = [10, 0]
    second = execution.run_backtest(panel, _signal(panel, values), config)
    pd.testing.assert_frame_equal(first.pnl, second.pnl)
    pd.testing.assert_frame_equal(first.trades, second.trades)


def test_buy_window_and_three_child_orders():
    """Slice across executable bars while respecting exclusive window ends."""
    panel = _panel(
        [
            f"2024-01-02 {clock}"
            for clock in ("09:30", "09:35", "09:40", "09:45", "09:50")
        ],
        1,
    )
    config = _config(
        panel, keep=1, order_slices=3, entry_start="09:35", entry_end="09:50"
    )
    result = execution.run_backtest(panel, _signal(panel), config)
    assert result.trades.timestamp.dt.strftime("%H:%M").tolist() == [
        "09:35",
        "09:40",
        "09:45",
    ]
    assert result.trades.iloc[0].filled == 333
    assert "outside_execution_window" in result.restrictions.reason.values
    assert (result.trades.cash >= 0).all()


def test_participation_caps_each_fill():
    """Volume caps are applied to actual filled shares, after lot rounding."""
    panel = _panel(["2024-01-02 09:30", "2024-01-02 09:35"], 1)
    panel = dataclasses.replace(
        panel,
        volume=np.full(panel.shape, 105.0),
        lot_sizes=np.full(panel.shape, 5.0),
    )
    config = dataclasses.replace(_config(panel, keep=1), participation_rate=0.1)
    result = execution.run_backtest(panel, _signal(panel), config)
    assert result.trades.filled.tolist() == [10, 10]
    assert (result.trades.filled <= panel.volume[:, 0] * 0.1).all()
    assert "participation_limit" in result.restrictions.reason.values


def test_t0_failed_flatten_records_actual_residual():
    """A suspension at the last bar cannot create an imaginary flat fill."""
    panel = _panel(["2024-01-02 09:30", "2024-01-02 10:00", "2024-01-02 15:55"])
    tradable = panel.tradable.copy()
    tradable[-1, 0] = False
    panel = dataclasses.replace(panel, tradable=tradable)
    values = np.tile([10.0, 0.0], (3, 1))
    result = execution.run_backtest(
        panel, _signal(panel, values), _config(panel, mode="T0", keep=0)
    )
    residual = result.restrictions[
        (result.restrictions.reason == "t0_end_residual")
        & (result.restrictions.symbol == "00001.HK")
    ]
    assert len(residual) == 1
    assert residual.iloc[0].side == "sell"
    assert residual.iloc[0].requested > 0
    assert residual.iloc[0].filled == 0
    final_sells = result.trades[
        (result.trades.timestamp == panel.open_timestamps[-1])
        & (result.trades.symbol == "00001.HK")
        & (result.trades.side == "sell")
    ]
    assert final_sells.empty


def test_half_day_forces_t0_restore_even_before_flatten_time():
    """The supplied final session bar overrides a regular-day cutoff."""
    panel = _panel(["2024-01-02 09:30", "2024-01-02 10:00", "2024-01-02 11:55"])
    values = np.tile([10.0, 0.0], (3, 1))
    result = execution.run_backtest(
        panel, _signal(panel, values), _config(panel, mode="T0", keep=0)
    )
    last = result.trades[result.trades.timestamp == panel.open_timestamps[-1]]
    assert ((last.side == "sell") & (last.symbol == "00001.HK")).any()


def test_t0_restoration_slices_start_at_configured_time():
    """Restore the base across bars and ignore later overlay updates."""
    panel = _panel(
        [
            f"2024-01-02 {clock}"
            for clock in (
                "09:30",
                "10:00",
                "10:05",
                "10:10",
                "15:30",
                "15:35",
                "15:40",
                "15:55",
            )
        ]
    )
    values = np.tile([10.0, 0.0], (8, 1))
    values[4:] = [0, 10]
    config = _config(panel, mode="T0", keep=0, order_slices=3)
    result = execution.run_backtest(panel, _signal(panel, values), config)
    sells = result.trades[
        (result.trades.side == "sell") & (result.trades.symbol == "00001.HK")
    ]
    assert sells.timestamp.dt.strftime("%H:%M").tolist()[:3] == [
        "15:30",
        "15:35",
        "15:40",
    ]
    assert abs(sells.filled.iloc[0] - sells.filled.iloc[1]) <= 1


def test_t0_restore_buys_use_exit_window_after_entry_window_ends():
    """Closing the overlay includes actual buys to restore underweight base."""
    panel = _panel(
        [
            f"2024-01-02 {clock}"
            for clock in ("09:30", "09:35", "10:00", "13:00", "15:30", "15:55")
        ]
    )
    signal = _signal(panel, np.tile([10.0, 0.0], (6, 1)))
    config = _config(
        panel,
        mode="T0",
        keep=0,
        entry_start="09:35",
        entry_end="11:30",
        exit_start="13:00",
        exit_end="16:00",
    )
    result = execution.run_backtest(panel, signal, config)
    restore_time = panel.open_timestamps[4]
    buys = result.trades[
        (result.trades.timestamp == restore_time)
        & (result.trades.symbol == "00002.HK")
        & (result.trades.side == "buy")
    ]
    assert len(buys) == 1
    assert buys.filled.iloc[0] > 0
    assert buys.reason.iloc[0] == "t0_restore_base"
    blocked = result.restrictions[
        (result.restrictions.timestamp == restore_time)
        & (result.restrictions.symbol == "00002.HK")
        & (result.restrictions.reason == "outside_execution_window")
    ]
    assert blocked.empty


def test_missing_quote_and_zero_volume_prevent_execution():
    """A valuation carry cannot make missing observations executable."""
    panel = _panel(
        ["2024-01-02 09:30", "2024-01-02 09:35", "2024-01-02 09:40"], 1
    )
    opens = panel.open.copy()
    opens[1] = np.nan
    volumes = np.zeros(panel.shape)
    volumes[1] = 1_000_000
    panel = dataclasses.replace(panel, open=opens, volume=volumes)
    result = execution.run_backtest(
        panel, _signal(panel), _config(panel, keep=1)
    )
    assert result.trades.empty
    assert {"missing_quote", "no_volume"} <= set(result.restrictions.reason)
    assert (result.restrictions.filled == 0).all()


def test_cash_conservation_and_frictionless_independent_book():
    """Fixed-price NAV loses exactly paid costs and never borrows cash."""
    panel = _panel(["2024-01-02 09:30", "2024-01-02 09:35"])
    result = execution.run_backtest(
        panel, _signal(panel), _config(panel, keep=1)
    )
    np.testing.assert_allclose(
        result.pnl.net_nav, 10_000 - result.pnl.fees.cumsum()
    )
    np.testing.assert_allclose(result.pnl.gross_nav, 10_000)
    np.testing.assert_allclose(
        result.pnl.net_nav, result.pnl.cash + result.pnl.position_value
    )
    assert (result.pnl.cash >= 0).all()
    assert result.trades.fees.sum() == pytest.approx(result.pnl.fees.sum())


def test_gross_account_keeps_opportunity_return_of_avoided_costs():
    """Gross NAV is not net NAV plus cumulative historical cash charges."""
    panel = _panel(
        ["2024-01-02 09:30", "2024-01-03 09:30"],
        1,
        prices=np.array([[10.0], [20.0]]),
        closes=np.array([[20.0], [30.0]]),
        frequency="1d",
    )
    result = execution.run_backtest(
        panel, _signal(panel), _config(panel, keep=1)
    )
    assert result.pnl.gross_nav.iloc[-1] == 30_000
    assert result.pnl.gross_nav.iloc[-1] - result.pnl.net_nav.iloc[-1] > (
        result.pnl.fees.sum()
    )


def test_warmup_supplies_signal_without_trading_or_fees():
    """Begin funded accounts at t_start using an earlier known session."""
    panel = _panel(["2024-01-02 09:30", "2024-01-03 09:30"], 2, frequency="1d")
    config = dataclasses.replace(_config(panel, keep=0), t_start="2024-01-03")
    result = execution.run_backtest(
        panel, _signal(panel, np.tile([10.0, 0.0], (2, 1))), config
    )
    assert len(result.pnl) == 1
    assert (result.trades.timestamp.dt.date.astype(str) == "2024-01-03").all()
    first_asset = result.trades[result.trades.symbol == "00001.HK"]
    second_asset = result.trades[result.trades.symbol == "00002.HK"]
    assert first_asset.filled.sum() > second_asset.filled.sum()


def test_benchmark_invests_at_first_execution_open():
    """Capture the first bar's open-to-close return for a newly funded book."""
    panel = _panel(
        ["2024-01-02 09:30"],
        1,
        prices=np.array([[10.0]]),
        closes=np.array([[11.0]]),
        frequency="1d",
    )
    panel = dataclasses.replace(
        panel,
        benchmark_close=np.array([110.0]),
        benchmark_open=np.array([100.0]),
    )
    result = execution.run_backtest(
        panel, _signal(panel), _config(panel, keep=1)
    )
    assert result.pnl.benchmark_return.iloc[0] == pytest.approx(0.1)
    assert result.pnl.benchmark_nav.iloc[0] == pytest.approx(11_000)
    assert result.metrics["benchmark_total_return"] == pytest.approx(0.1)
    assert result.metrics["gross_total_return"] == pytest.approx(0.1)
    assert result.metrics["benchmark_start_policy"] == "first_execution_open"


@pytest.mark.parametrize("has_open", [False, True])
def test_initial_benchmark_excludes_warmup_overnight_return(has_open: bool):
    """Warmup prices inform signals without anchoring the new investment."""
    panel = _panel(
        ["2024-01-02 09:30", "2024-01-03 09:30", "2024-01-04 09:30"],
        1,
        frequency="1d",
    )
    panel = dataclasses.replace(
        panel,
        benchmark_close=np.array([200.0, 110.0, 121.0]),
        benchmark_open=np.array([190.0, 100.0, 120.0]) if has_open else None,
    )
    config = dataclasses.replace(_config(panel, keep=1), t_start="2024-01-03")
    result = execution.run_backtest(panel, _signal(panel), config)
    expected_first = 0.1 if has_open else 0.0
    expected_total = 0.21 if has_open else 0.1
    expected_policy = (
        "first_execution_open" if has_open else "first_close_missing_open"
    )
    assert len(result.pnl) == 2
    assert result.pnl.benchmark_return.tolist() == pytest.approx(
        [expected_first, 0.1]
    )
    assert result.metrics["benchmark_total_return"] == pytest.approx(
        expected_total
    )
    assert result.metrics["benchmark_start_policy"] == expected_policy


def test_corporate_actions_use_existing_shares_before_trading():
    """Credit cash to entitled shares and preserve wealth through a split."""
    panel = _panel(
        ["2024-01-02 09:30", "2024-01-03 09:30"],
        1,
        prices=np.array([[10.0], [5.0]]),
        frequency="1d",
    )
    panel = dataclasses.replace(
        panel,
        cash_dividend=np.array([[0.0], [0.5]]),
        share_multiplier=np.array([[1.0], [2.0]]),
    )
    result = execution.run_backtest(
        panel, _signal(panel), _config(panel, keep=1)
    )
    entitled = result.trades[
        result.trades.timestamp == panel.open_timestamps[0]
    ].filled.sum()
    assert result.pnl.dividends.iloc[1] == entitled * 0.5
    assert result.pnl.net_pnl.iloc[1] == pytest.approx(
        entitled * 0.5 - result.pnl.fees.iloc[1]
    )


def test_vcm_supplied_band_rejects_price_and_strict_missing_fails():
    """Constrain actual event bands without inventing daily price limits."""
    panel = _panel(["2024-01-02 09:30"], 1)
    eligible = np.ones(panel.shape, dtype=bool)
    strict = _config(panel, keep=1, enforce_vcm=True)
    with pytest.raises(ValueError, match="Strict VCM"):
        execution.run_backtest(
            dataclasses.replace(panel, vcm_eligible=eligible),
            _signal(panel),
            strict,
        )
    panel = dataclasses.replace(
        panel,
        vcm_eligible=eligible,
        vcm_lower=np.full(panel.shape, 11.0),
        vcm_upper=np.full(panel.shape, 12.0),
    )
    result = execution.run_backtest(panel, _signal(panel), strict)
    assert result.trades.empty
    assert "vcm_price_band" in result.restrictions.reason.values


def test_strict_vcm_known_inactive_does_not_require_bands():
    """Normal trading is permitted when historical event state is known."""
    panel = _panel(["2024-01-02 09:30"], 1)
    panel = dataclasses.replace(
        panel,
        vcm_eligible=np.ones(panel.shape, dtype=bool),
        vcm_active=np.zeros(panel.shape, dtype=np.int8),
    )
    result = execution.run_backtest(
        panel, _signal(panel), _config(panel, keep=1, enforce_vcm=True)
    )
    assert not result.trades.empty
    assert "vcm_price_band" not in result.restrictions.reason.values


def test_vcm_known_inactive_ignores_stale_bands():
    """A historical inactive flag overrides bands carried from an old event."""
    panel = _panel(["2024-01-02 09:30"], 1)
    panel = dataclasses.replace(
        panel,
        vcm_eligible=np.ones(panel.shape, dtype=bool),
        vcm_active=np.zeros(panel.shape, dtype=np.int8),
        vcm_lower=np.full(panel.shape, 11.0),
        vcm_upper=np.full(panel.shape, 12.0),
    )
    result = execution.run_backtest(
        panel, _signal(panel), _config(panel, keep=1, enforce_vcm=True)
    )
    assert not result.trades.empty
    assert "vcm_price_band" not in result.restrictions.reason.values


def test_vcm_active_missing_band_rejects_in_default_and_strict_modes():
    """Known active cooling events cannot be simulated without their bands."""
    panel = _panel(["2024-01-02 09:30"], 1)
    panel = dataclasses.replace(
        panel,
        vcm_eligible=np.ones(panel.shape, dtype=bool),
        vcm_active=np.ones(panel.shape, dtype=np.int8),
        vcm_lower=np.full(panel.shape, 9.0),
    )
    config = _config(panel, keep=1)
    result = execution.run_backtest(panel, _signal(panel), config)
    assert result.trades.empty
    blocked = result.restrictions[
        result.restrictions.reason == "missing_vcm_event_bands"
    ]
    assert len(blocked) == 1
    assert blocked.filled.iloc[0] == 0
    with pytest.raises(ValueError, match="Strict VCM"):
        execution.run_backtest(
            panel, _signal(panel), dataclasses.replace(config, enforce_vcm=True)
        )


@pytest.mark.parametrize("lot", [np.nan, 0.0, -1.0])
def test_round_lots_requires_valid_historical_board_lots(lot: float):
    """A missing or invalid lot cannot produce a false successful fill."""
    panel = _panel(["2024-01-02 09:30"], 1)
    panel = dataclasses.replace(panel, lot_sizes=np.full(panel.shape, lot))
    config = _config(panel, keep=1)
    with pytest.raises(ValueError, match="Historical board-lot sizes"):
        execution.run_backtest(panel, _signal(panel), config)
    result = execution.run_backtest(
        panel, _signal(panel), dataclasses.replace(config, round_lots=False)
    )
    assert not result.trades.empty


def test_stamp_exemption_affects_funded_quantity_and_actual_fill_fees():
    """Funding checks use the same historical exemption as executed charges."""
    panel = _panel(["2024-12-23 09:30"], 1, prices=np.array([[1.0]]))
    config = dataclasses.replace(_config(panel, keep=1), initial_capital=1000)
    taxable = execution.run_backtest(panel, _signal(panel), config)
    exempt_panel = dataclasses.replace(
        panel, stamp_duty_exempt=np.ones(panel.shape, dtype=bool)
    )
    exempt = execution.run_backtest(exempt_panel, _signal(exempt_panel), config)
    assert taxable.trades.filled.iloc[0] == 998
    assert exempt.trades.filled.iloc[0] == 999
    assert taxable.trades.stamp_duty.iloc[0] == 1
    assert exempt.trades.stamp_duty.iloc[0] == 0
    assert taxable.trades.fees.iloc[0] == pytest.approx(1.09)
    assert exempt.trades.fees.iloc[0] == pytest.approx(0.09)
    assert exempt.pnl.cash.iloc[0] >= 0
    assert exempt.pnl.net_nav.iloc[0] == pytest.approx(999.91)


def test_adjusted_execution_requires_explicit_research_opt_in():
    """Do not use adjusted stock prices as historical executable quotes."""
    panel = dataclasses.replace(
        _panel(["2024-01-02 09:30"], 1), price_basis="qfq"
    )
    config = _config(panel, keep=1)
    with pytest.raises(ValueError, match="Raw execution"):
        execution.run_backtest(panel, _signal(panel), config)
    result = execution.run_backtest(
        panel,
        _signal(panel),
        dataclasses.replace(config, allow_adjusted_prices=True),
    )
    assert result.metrics["adjusted_price_research"] is True


def test_streamed_audits_equal_in_memory_results(
    tmp_path: pathlib.Path,
) -> None:
    """Streaming changes storage while preserving account and audit records."""
    panel = _panel(["2024-01-02 09:30", "2024-01-02 09:35", "2024-01-02 15:55"])
    panel = dataclasses.replace(
        panel,
        volume=np.full(panel.shape, 105.0),
        lot_sizes=np.full(panel.shape, 5.0),
    )
    config = dataclasses.replace(
        _config(panel, mode="T0", keep=0), participation_rate=0.1
    )
    signal = _signal(panel, np.tile([10.0, 0.0], (3, 1)))
    memory = execution.run_backtest(panel, signal, config)
    streamed = execution.run_backtest(panel, signal, config, audit_dir=tmp_path)
    assert streamed.audit_streamed is True
    assert memory.audit_streamed is False
    assert streamed.trades.empty and streamed.restrictions.empty
    pd.testing.assert_frame_equal(memory.pnl, streamed.pnl)
    pd.testing.assert_frame_equal(
        memory.trades, pd.read_parquet(tmp_path / "trades.parquet")
    )
    pd.testing.assert_frame_equal(
        memory.restrictions,
        pd.read_parquet(tmp_path / "restrictions.parquet"),
    )
    assert streamed.metrics["trade_count"] == len(memory.trades)
    assert streamed.metrics["restriction_count"] == len(memory.restrictions)
    assert len(list(tmp_path.glob("*.parquet"))) == 2
    limited = memory.restrictions[
        memory.restrictions.reason == "participation_limit"
    ]
    for row in limited.itertuples():
        fill = memory.trades[
            (memory.trades.timestamp == row.timestamp)
            & (memory.trades.symbol == row.symbol)
        ]
        assert row.filled == fill.filled.sum()


@pytest.mark.parametrize(
    "kind,schema",
    [
        ("trades", audit.TRADE_SCHEMA),
        ("restrictions", audit.RESTRICTION_SCHEMA),
    ],
)
def test_empty_streamed_audits_have_complete_schema(
    tmp_path: pathlib.Path,
    kind: str,
    schema,
) -> None:
    """An empty audit remains readable with the full stable field contract."""
    with audit.AuditRecords(kind, tmp_path) as records:
        assert records.count == 0
        assert records.to_frame().columns.tolist() == schema.names
    table = pq.read_table(tmp_path / f"{kind}.parquet")
    assert table.num_rows == 0
    assert table.schema.equals(schema)


def test_streaming_buffer_stays_bounded_and_closes_on_exception(
    tmp_path: pathlib.Path,
) -> None:
    """Multiple flushes retain at most 4096 rows and preserve finalized data."""
    timestamp = pd.Timestamp("2024-01-02 09:30", tz="Asia/Hong_Kong")
    row = {
        "timestamp": timestamp,
        "bar_end": timestamp,
        "symbol": "00001.HK",
        "side": "buy",
        "requested": 100.0,
        "filled": 10.0,
        "reason": "participation_limit",
    }
    total = audit.AUDIT_BUFFER_ROWS * 2 + 17
    with pytest.raises(RuntimeError, match="interrupted simulation"):
        with audit.AuditRecords("restrictions", tmp_path) as records:
            for _ in range(total):
                records.append(row.copy())
                assert records.buffered_rows <= audit.AUDIT_BUFFER_ROWS
            assert records.peak_buffered_rows == audit.AUDIT_BUFFER_ROWS
            assert records.count == total
            raise RuntimeError("interrupted simulation")
    table = pq.read_table(tmp_path / "restrictions.parquet")
    assert table.num_rows == total
    assert records.buffered_rows == 0
    assert table["filled"].to_pylist() == [10.0] * total


def test_execution_failure_closes_both_streamed_audits(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    """A simulation failure flushes finalized earlier orders before exit."""
    panel = _panel(["2024-01-02 09:30", "2024-01-03 09:30"], frequency="1d")
    config = _config(panel, mode="T1", keep=0)
    signal = _signal(panel, np.tile([10.0, 0.0], (2, 1)))
    original = rules.transaction_fees

    def failed_fee(notional, timestamp, fee_config, **kwargs):
        if timestamp == panel.open_timestamps[1]:
            raise RuntimeError("failed fee service")
        return original(notional, timestamp, fee_config, **kwargs)

    monkeypatch.setattr(rules, "transaction_fees", failed_fee)
    with pytest.raises(RuntimeError, match="failed fee service"):
        execution.run_backtest(panel, signal, config, audit_dir=tmp_path)
    trades = pq.read_table(tmp_path / "trades.parquet")
    restrictions = pq.read_table(tmp_path / "restrictions.parquet")
    assert trades.num_rows == 2
    assert trades.schema.equals(audit.TRADE_SCHEMA)
    assert restrictions.schema.equals(audit.RESTRICTION_SCHEMA)
    assert trades["timestamp"].to_pandas().max() == panel.open_timestamps[0]
