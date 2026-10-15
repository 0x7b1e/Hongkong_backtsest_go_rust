"""Generate reproducible synthetic fixtures for integration tests."""

import json
import pathlib

import numpy as np
import pandas as pd

from hkfactor import config as backtest_config
from hkfactor import data as market_data


def create_demo(
    directory: str | pathlib.Path,
    frequency: str = "1d",
    *,
    asset_count: int = 20,
    trading_days: int | None = None,
    factor_count: int = 2,
    seed: int = 101,
) -> dict:
    """Write a labelled 20-security, two-factor integration dataset.

    Args:
        directory: New fixture directory; existing directories are preserved.
        frequency: 1d for T1, or 5m/1m for intraday T0.
        asset_count: Synthetic stocks, never real historical constituents.
        trading_days: Optional number of sessions for scale experiments.
        factor_count: Factors emitted individually, without a factor cube.
        seed: Reproducible pseudorandom seed.

    Returns:
        Paths and assumptions for running prepare and backtest commands.
    """
    directory = pathlib.Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    if asset_count < 10 or factor_count < 1:
        raise ValueError("Demo needs at least ten assets and one factor")
    if frequency not in {"1d", "5m", "1m"}:
        raise ValueError("Unsupported demo frequency")
    trading_days = trading_days or (40 if frequency == "1d" else 8)
    if trading_days < 2:
        raise ValueError("Demo needs at least two trading days")
    days = pd.bdate_range(
        "2024-01-02", periods=trading_days, tz=market_data.TIMEZONE
    )
    calendar = pd.DataFrame(
        {
            "date": days,
            "open": "09:30",
            "close": "16:00",
            "break_start": "12:00",
            "break_end": "13:00",
        }
    )
    timestamps, _ = market_data._bar_grid(calendar, frequency)
    symbols = tuple(f"{i:05d}.HK" for i in range(1, asset_count + 1))
    rng = np.random.default_rng(seed)
    innovations = rng.normal(
        0.0001,
        0.008 if frequency == "1d" else 0.001,
        (len(timestamps), len(symbols)),
    )
    closes = 30 * np.exp(np.cumsum(innovations, axis=0))
    opens = np.vstack([closes[0] / np.exp(innovations[0]), closes[:-1]])
    prices = pd.DataFrame(
        {
            "timestamp": np.repeat(timestamps, len(symbols)),
            "symbol": np.tile(symbols, len(timestamps)),
            "open": opens.ravel(),
            "close": closes.ravel(),
            "high": np.maximum(opens, closes).ravel() * 1.002,
            "low": np.minimum(opens, closes).ravel() * 0.998,
            "volume": rng.integers(200_000, 500_000, closes.size),
            "lot_size": 100,
            "price_basis": "raw",
        }
    )
    momentum = pd.DataFrame(closes).pct_change(5, fill_method=None).to_numpy()
    signals = pd.DataFrame(
        {
            "timestamp": np.repeat(timestamps, len(symbols)),
            "available_at": np.repeat(timestamps, len(symbols)),
            "symbol": np.tile(symbols, len(timestamps)),
            "value": momentum.ravel(),
            "factor": "momentum_5",
        }
    )
    directory.mkdir(parents=True)
    prices.to_parquet(directory / "prices.parquet", index=False)
    pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens.mean(axis=1) * 100,
            "close": closes.mean(axis=1) * 100,
        }
    ).to_parquet(directory / "benchmark.parquet", index=False)
    pd.DataFrame(
        {
            "effective_date": days[0],
            "available_at": days[0] - pd.Timedelta(days=1),
            "symbol": symbols,
            "weight": 1 / len(symbols),
        }
    ).to_parquet(directory / "constituents.parquet", index=False)
    calendar.to_parquet(directory / "sessions.parquet", index=False)
    for index in range(factor_count):
        frame = signals.copy()
        if index == 1:
            frame["factor"] = "reversal_5"
            frame["value"] = -frame["value"]
        elif index > 1:
            frame["factor"] = f"candidate_{index:05d}"
            frame["value"] += rng.normal(0, 0.002, len(frame))
        partition = directory / "signals" / f"factor={frame.factor.iloc[0]}"
        partition.mkdir(parents=True)
        frame.drop(columns="factor").to_parquet(
            partition / "part.parquet", index=False
        )
    config = backtest_config.BacktestConfig(
        t_start=days[0].date().isoformat(),
        t_end=days[-1].date().isoformat(),
        mode="T1" if frequency == "1d" else "T0",
        frequency=frequency,
        order_slices=1 if frequency == "1d" else 3,
    )
    (directory / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    result = {
        "synthetic": True,
        "seed": seed,
        "symbols": asset_count,
        "sessions": len(days),
        "bars": len(timestamps),
        "frequency": frequency,
        "factors": factor_count,
        "note": "Not real HSI constituents or market performance",
    }
    (directory / "demo.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
