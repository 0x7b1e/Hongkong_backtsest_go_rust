"""Validate chronology, membership, and completeness before simulation."""

import pathlib

import numpy as np
import pandas as pd
import pytest

from hkfactor import data as market_data
from hkfactor import models


def _inputs(
    directory: pathlib.Path,
    timestamps: pd.DatetimeIndex | None = None,
) -> dict[str, pathlib.Path]:
    """Write a small, complete two-stock input dataset."""
    if timestamps is None:
        timestamps = pd.date_range(
            "2026-01-05 16:00",
            periods=3,
            freq="D",
            tz="Asia/Hong_Kong",
        )
    records = []
    for index, timestamp in enumerate(timestamps):
        for symbol, offset in [("00001.HK", 0), ("00002.HK", 5)]:
            price = 100 + index * 10 + offset
            records.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "open": price,
                    "high": price + 3,
                    "low": price - 1,
                    "close": price + 2,
                    "volume": 100_000,
                    "lot_size": 100,
                    "price_basis": "raw",
                }
            )
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        name: directory / f"{name}.parquet"
        for name in ["prices", "benchmark", "constituents", "sessions"]
    }
    # Reverse input order so tests cannot pass through accidental sorting.
    pd.DataFrame(records).iloc[::-1].to_parquet(paths["prices"], index=False)
    pd.DataFrame({"timestamp": timestamps, "close": 20_000}).to_parquet(
        paths["benchmark"],
        index=False,
    )
    pd.DataFrame(
        {
            "effective_date": ["2026-01-05", "2026-01-05"],
            "available_at": ["2026-01-04 10:00"] * 2,
            "symbol": ["00001.HK", "00002.HK"],
            "weight": [0.5, 0.5],
        }
    ).to_parquet(paths["constituents"], index=False)
    pd.DataFrame(
        {
            "date": timestamps.normalize().unique(),
            "open": "09:30",
            "close": "16:00",
        }
    ).to_parquet(paths["sessions"], index=False)
    return paths


def _prepare(
    paths: dict[str, pathlib.Path],
    output: pathlib.Path,
    **kwargs: object,
) -> models.MarketPanel:
    """Prepare a cache from the canonical test inputs."""
    return market_data.prepare_market(
        prices=paths["prices"],
        benchmark=paths["benchmark"],
        constituents=paths["constituents"],
        output=output,
        **kwargs,
    )


def test_prepare_load_retains_real_dates_and_readonly_arrays(tmp_path):
    paths = _inputs(tmp_path / "inputs")
    panel = _prepare(paths, tmp_path / "panel")
    expected = pd.date_range(
        "2026-01-05 16:00", periods=3, freq="D", tz="Asia/Hong_Kong"
    )
    assert panel.timestamps.equals(expected)
    assert panel.open_timestamps[0] == pd.Timestamp(
        "2026-01-05 09:30",
        tz="Asia/Hong_Kong",
    )
    np.testing.assert_array_equal(panel.close[:, 0], [102, 112, 122])
    reloaded = market_data.load_panel(tmp_path / "panel")
    assert reloaded.timestamps.equals(panel.timestamps)
    for name in [
        "open",
        "close",
        "volume",
        "base_weights",
        "tradable",
        "benchmark_close",
        "session_id",
    ]:
        array = getattr(reloaded, name)
        assert isinstance(array, np.memmap)
        assert not array.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        reloaded.close[0, 0] = 0
    with pytest.raises(FileExistsError):
        _prepare(paths, tmp_path / "panel")


def test_missing_quote_uses_only_past_valuation_and_never_trades(tmp_path):
    paths = _inputs(tmp_path / "inputs")
    prices = pd.read_parquet(paths["prices"])
    missing = (prices.symbol == "00002.HK") & (prices.timestamp.dt.day == 6)
    prices.loc[~missing].to_parquet(paths["prices"], index=False)
    panel = _prepare(paths, tmp_path / "panel")
    assert panel.close[1, 1] == panel.close[0, 1] == 107
    assert panel.close[2, 1] == 127
    assert panel.open[1, 1] == 107
    assert panel.volume[1, 1] == 0
    assert not panel.tradable[1, 1]
    assert np.isnan(panel.high[1, 1])
    assert np.isnan(panel.low[1, 1])


def test_new_members_future_quote_cannot_backfill_past(tmp_path):
    paths = _inputs(tmp_path / "inputs")
    snapshots = pd.read_parquet(paths["constituents"])
    future = pd.DataFrame(
        {
            "effective_date": ["2026-01-07"] * 2,
            "available_at": ["2026-01-06 10:00"] * 2,
            "symbol": ["00001.HK", "00003.HK"],
            "weight": [0.5, 0.5],
        }
    )
    pd.concat([snapshots, future]).to_parquet(
        paths["constituents"], index=False
    )
    prices = pd.read_parquet(paths["prices"])
    extra = prices[
        (prices.symbol == "00002.HK") & (prices.timestamp.dt.day >= 6)
    ].copy()
    extra["symbol"] = "00003.HK"
    pd.concat([prices, extra]).to_parquet(paths["prices"], index=False)
    panel = _prepare(paths, tmp_path / "panel")
    assert panel.symbols == ("00001.HK", "00002.HK", "00003.HK")
    assert np.isnan(panel.close[0, 2])
    assert np.isnan(panel.open[0, 2])
    assert np.isnan(panel.lot_sizes[0, 2])
    assert not panel.tradable[0, 2]
    assert panel.base_weights[0, 2] == panel.base_weights[1, 2] == 0
    assert panel.base_weights[2, 2] == 0.5


def test_late_announcement_keeps_previous_snapshot_until_known(tmp_path):
    paths = _inputs(tmp_path / "inputs")
    snapshots = pd.read_parquet(paths["constituents"])
    late = pd.DataFrame(
        {
            "effective_date": ["2026-01-06"] * 2,
            "available_at": ["2026-01-06 10:00"] * 2,
            "symbol": ["00001.HK", "00002.HK"],
            "weight": [0.8, 0.2],
        }
    )
    pd.concat([snapshots, late]).to_parquet(paths["constituents"], index=False)
    panel = _prepare(paths, tmp_path / "panel")
    np.testing.assert_array_equal(panel.base_weights[1], [0.5, 0.5])
    np.testing.assert_array_equal(panel.base_weights[2], [0.8, 0.2])


def test_required_first_valuation_cannot_use_a_future_quote(tmp_path):
    paths = _inputs(tmp_path / "inputs")
    prices = pd.read_parquet(paths["prices"])
    missing = (prices.symbol == "00001.HK") & (prices.timestamp.dt.day == 5)
    prices.loc[~missing].to_parquet(paths["prices"], index=False)
    with pytest.raises(ValueError, match="no past quote"):
        _prepare(paths, tmp_path / "panel")
    assert not (tmp_path / "panel").exists()


@pytest.mark.parametrize("corruption", ["duplicate", "ohlc", "benchmark"])
def test_invalid_market_input_is_rejected_without_partial_cache(
    tmp_path,
    corruption,
):
    paths = _inputs(tmp_path / "inputs")
    prices = pd.read_parquet(paths["prices"])
    if corruption == "duplicate":
        pd.concat([prices, prices.iloc[:1]]).to_parquet(
            paths["prices"], index=False
        )
        message = "Duplicate"
    elif corruption == "ohlc":
        prices.loc[prices.index[0], "high"] = 1
        prices.to_parquet(paths["prices"], index=False)
        message = "OHLC"
    else:
        benchmark = pd.read_parquet(paths["benchmark"])
        benchmark.iloc[:-1].to_parquet(paths["benchmark"], index=False)
        message = "Benchmark"
    with pytest.raises(ValueError, match=message):
        _prepare(paths, tmp_path / "panel")
    assert not (tmp_path / "panel").exists()
    assert not list(tmp_path.glob(".panel.*.tmp"))


def test_intraday_requires_explicit_historical_calendar(tmp_path):
    paths = _inputs(tmp_path / "inputs")
    with pytest.raises(ValueError, match="historical session calendar"):
        _prepare(paths, tmp_path / "panel", frequency="5m")
    assert not (tmp_path / "panel").exists()


def test_panel_budget_rejection_leaves_no_cache_and_allows_retry(tmp_path):
    paths = _inputs(tmp_path / "inputs")
    with pytest.raises(MemoryError, match="budget"):
        _prepare(paths, tmp_path / "panel", max_panel_bytes=1)
    assert not (tmp_path / "panel").exists()
    assert not list(tmp_path.glob(".panel.*.tmp"))
    assert _prepare(paths, tmp_path / "panel").shape == (3, 2)


def test_complete_missing_session_cannot_disappear_from_calendar(tmp_path):
    paths = _inputs(tmp_path / "inputs")
    prices = pd.read_parquet(paths["prices"])
    prices[prices.timestamp.dt.day != 6].to_parquet(
        paths["prices"], index=False
    )
    with pytest.raises(ValueError, match="(?i)(missing|no.*quote|no.*price)"):
        _prepare(paths, tmp_path / "panel", sessions=paths["sessions"])
    assert not (tmp_path / "panel").exists()


@pytest.mark.parametrize("missing_day", [5, 7])
def test_requested_date_boundaries_reject_entire_missing_sessions(
    tmp_path,
    missing_day,
):
    paths = _inputs(tmp_path / "inputs")
    prices = pd.read_parquet(paths["prices"])
    prices[prices.timestamp.dt.day != missing_day].to_parquet(
        paths["prices"],
        index=False,
    )
    with pytest.raises(ValueError, match="(?i)(missing|no.*quote|no.*price)"):
        _prepare(
            paths,
            tmp_path / "panel",
            sessions=paths["sessions"],
            t_start="2026-01-05",
            t_end="2026-01-07",
        )
    assert not (tmp_path / "panel").exists()


@pytest.mark.parametrize("column", ["tradable", "halted", "vcm_eligible"])
def test_string_false_flags_cannot_silently_become_true(tmp_path, column):
    paths = _inputs(tmp_path / "inputs")
    prices = pd.read_parquet(paths["prices"])
    prices[column] = "False"
    prices.to_parquet(paths["prices"], index=False)
    with pytest.raises(ValueError, match="boolean"):
        _prepare(paths, tmp_path / "panel")
    assert not (tmp_path / "panel").exists()


def test_complete_missing_intraday_bar_cannot_be_valued_as_real_bar(tmp_path):
    timestamps = pd.date_range(
        "2026-01-05 09:35", periods=3, freq="5min", tz="Asia/Hong_Kong"
    )
    paths = _inputs(tmp_path / "inputs", timestamps)
    calendar = pd.read_parquet(paths["sessions"])
    calendar["close"] = "09:45"
    calendar.to_parquet(paths["sessions"], index=False)
    prices = pd.read_parquet(paths["prices"])
    prices[prices.timestamp.dt.minute != 40].to_parquet(
        paths["prices"], index=False
    )
    with pytest.raises(ValueError, match="(?i)(missing|no.*quote|no.*price)"):
        _prepare(
            paths,
            tmp_path / "panel",
            frequency="5m",
            sessions=paths["sessions"],
        )
    assert not (tmp_path / "panel").exists()


def test_vcm_active_cache_distinguishes_unknown_inactive_and_active(tmp_path):
    paths = _inputs(tmp_path / "inputs")
    unknown = _prepare(paths, tmp_path / "unknown")
    np.testing.assert_array_equal(
        unknown.vcm_active, np.full(unknown.shape, -1, dtype=np.int8)
    )
    prices = pd.read_parquet(paths["prices"])
    prices["vcm_active"] = prices.timestamp.dt.day == 7
    missing = (prices.symbol == "00002.HK") & (prices.timestamp.dt.day == 6)
    prices.loc[~missing].to_parquet(paths["prices"], index=False)
    panel = _prepare(paths, tmp_path / "states")
    assert panel.vcm_active.dtype == np.dtype("int8")
    np.testing.assert_array_equal(panel.vcm_active[:, 1], [0, -1, 1])
    cached = market_data.load_panel(tmp_path / "states")
    np.testing.assert_array_equal(cached.vcm_active, panel.vcm_active)
    assert not cached.vcm_active.flags.writeable


def _dated_inputs(
    directory: pathlib.Path, dates: list[str]
) -> dict[str, pathlib.Path]:
    """Align historical quotes and membership for exemption-boundary tests."""
    timestamps = pd.DatetimeIndex(
        [f"{day} 16:00" for day in dates],
        tz="Asia/Hong_Kong",
    )
    paths = _inputs(directory, timestamps)
    snapshots = pd.read_parquet(paths["constituents"])
    snapshots["effective_date"] = dates[0]
    snapshots["available_at"] = timestamps[0].normalize() - pd.Timedelta(
        days=1,
    )
    snapshots.to_parquet(paths["constituents"], index=False)
    return paths


@pytest.mark.parametrize(
    "security_type, dates",
    [
        ("reit", ["2024-12-20", "2024-12-23"]),
        ("etf", ["2015-02-12", "2015-02-13"]),
    ],
)
def test_security_class_exemption_uses_historical_effective_date(
    tmp_path,
    security_type,
    dates,
):
    paths = _dated_inputs(tmp_path / "inputs", dates)
    prices = pd.read_parquet(paths["prices"])
    prices["security_type"] = np.where(
        prices.symbol == "00002.HK", security_type, "stock"
    )
    prices.to_parquet(paths["prices"], index=False)
    panel = _prepare(paths, tmp_path / "panel")
    np.testing.assert_array_equal(panel.stamp_duty_exempt[:, 0], [False, False])
    np.testing.assert_array_equal(panel.stamp_duty_exempt[:, 1], [False, True])
    cached = market_data.load_panel(tmp_path / "panel")
    np.testing.assert_array_equal(
        cached.stamp_duty_exempt, panel.stamp_duty_exempt
    )
    assert not cached.stamp_duty_exempt.flags.writeable


def test_explicit_stamp_exemption_overrides_class_derivation(tmp_path):
    paths = _dated_inputs(tmp_path / "inputs", ["2024-12-20", "2024-12-23"])
    prices = pd.read_parquet(paths["prices"])
    prices["security_type"] = np.where(
        prices.symbol == "00002.HK", "reit", "stock"
    )
    prices["stamp_duty_exempt"] = prices.symbol == "00001.HK"
    prices.to_parquet(paths["prices"], index=False)
    panel = _prepare(paths, tmp_path / "panel")
    np.testing.assert_array_equal(panel.stamp_duty_exempt[:, 0], [True, True])
    np.testing.assert_array_equal(panel.stamp_duty_exempt[:, 1], [False, False])
    cached = market_data.load_panel(tmp_path / "panel")
    np.testing.assert_array_equal(
        cached.stamp_duty_exempt, panel.stamp_duty_exempt
    )
