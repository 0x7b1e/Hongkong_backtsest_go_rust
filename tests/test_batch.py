"""Batch failure isolation, cache identity, and process determinism."""

import dataclasses
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from hkfactor import batch
from hkfactor import config as backtest_config
from hkfactor import models


@pytest.fixture
def inputs(
    tmp_path: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path, backtest_config.BacktestConfig]:
    """Create four realistic daily sessions for ten independent symbols."""
    timestamps = pd.date_range(
        "2024-01-02 16:00", periods=4, freq="D", tz="Asia/Hong_Kong"
    )
    symbols = tuple(f"{index:05d}.HK" for index in range(1, 11))
    shape = (len(timestamps), len(symbols))
    close = (
        np.arange(10, dtype=float)[None, :]
        + np.array([100.0, 101.0, 99.0, 103.0])[:, None]
    )
    opening = close - 0.25
    market = models.MarketPanel(
        timestamps=timestamps,
        open_timestamps=timestamps.normalize() + pd.Timedelta("9h30min"),
        symbols=symbols,
        frequency="1d",
        open=opening,
        high=close + 1,
        low=opening - 1,
        close=close,
        volume=np.full(shape, 1_000_000.0),
        base_weights=np.full(shape, 0.1),
        benchmark_close=np.array([100.0, 101.0, 99.0, 103.0]),
        lot_sizes=np.full(shape, 100, dtype=int),
        tradable=np.ones(shape, dtype=bool),
        vcm_lower=np.full(shape, np.nan),
        vcm_upper=np.full(shape, np.nan),
        vcm_eligible=np.zeros(shape, dtype=bool),
        cash_dividend=np.zeros(shape),
        share_multiplier=np.ones(shape),
        session_id=np.arange(len(timestamps)),
        is_session_start=np.ones(len(timestamps), dtype=bool),
        is_session_end=np.ones(len(timestamps), dtype=bool),
    )
    market_dir = tmp_path / "market"
    market.save(market_dir)
    rows = [
        {
            "timestamp": timestamp.strftime("%Y-%m-%d"),
            "symbol": symbol,
            "factor": factor,
            "value": float(index * direction),
        }
        for timestamp in timestamps
        for index, symbol in enumerate(symbols)
        for factor, direction in (("momentum", 1), ("reversal", -1))
    ]
    rows.append(
        {
            "timestamp": "2024-01-02",
            "symbol": "99999.HK",
            "factor": "broken",
            "value": 1.0,
        }
    )
    signals = tmp_path / "signals.parquet"
    pd.DataFrame(rows).to_parquet(signals, index=False)
    config = backtest_config.BacktestConfig(
        t_start="2024-01-02", t_end="2024-01-05", simulate_layers=False
    )
    return market_dir, signals, config


def test_failure_isolation_and_manifest(inputs, tmp_path: pathlib.Path) -> None:
    """A bad factor must not discard completed valid factor reports."""
    market, signals, config = inputs
    output = tmp_path / "output"
    summary = batch.run_batch(market, signals, output, config, plots=False)
    assert summary["factor"].tolist() == ["broken", "momentum", "reversal"]
    rows = summary.set_index("factor")
    assert rows.loc["broken", "status"] == "failed"
    assert "Unknown signal symbols" in rows.loc["broken", "error"]
    assert set(rows.loc[["momentum", "reversal"], "status"]) == {"success"}
    manifest = output / "runs" / summary.iloc[0]["run_id"] / "manifest.jsonl"
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(records) == 3
    assert (output / "summary.csv").exists()


def test_resume_requires_matching_inputs_and_complete_artifacts(
    inputs,
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    """Config edits and deleted artifacts must invalidate cached results."""
    market, signals, config = inputs
    output = tmp_path / "output"
    first = batch.run_batch(
        market, signals, output, config, plots=False, factors=["momentum"]
    )
    calls = []
    original = batch._run_factor

    def counting_job(factor: str) -> dict:
        calls.append(factor)
        return original(factor)

    monkeypatch.setattr(batch, "_run_factor", counting_job)
    cached = batch.run_batch(
        market, signals, output, config, plots=False, factors=["momentum"]
    )
    assert not calls
    pd.testing.assert_frame_equal(first, cached)
    changed = batch.run_batch(
        market,
        signals,
        output,
        dataclasses.replace(config, keep=0.8),
        plots=False,
        factors=["momentum"],
    )
    assert calls == ["momentum"]
    assert changed.iloc[0]["run_id"] != first.iloc[0]["run_id"]
    directory = pathlib.Path(first.iloc[0]["output_directory"])
    completed = json.loads((directory / "completed.json").read_text())
    (directory / completed["artifacts"][0]["path"]).unlink()
    batch.run_batch(
        market, signals, output, config, plots=False, factors=["momentum"]
    )
    assert calls == ["momentum", "momentum"]


def test_signal_changes_invalidate_resume(
    inputs, tmp_path: pathlib.Path
) -> None:
    """A rewritten source Parquet cannot inherit its previous output cache."""
    market, signals, config = inputs
    output = tmp_path / "output"
    first = batch.run_batch(
        market, signals, output, config, plots=False, factors=["momentum"]
    )
    frame = pd.read_parquet(signals)
    frame.loc[frame["factor"] == "momentum", "value"] *= -1
    frame.to_parquet(signals, index=False)
    changed = batch.run_batch(
        market, signals, output, config, plots=False, factors=["momentum"]
    )
    assert changed.iloc[0]["run_id"] != first.iloc[0]["run_id"]


def test_implementation_changes_invalidate_resume(
    inputs,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source changes must invalidate results even with identical inputs."""
    market, signals, config = inputs
    output = tmp_path / "output"
    monkeypatch.setattr(batch, "_implementation_fingerprint", lambda: "before")
    first = batch.run_batch(
        market, signals, output, config, plots=False, factors=["momentum"]
    )
    monkeypatch.setattr(batch, "_implementation_fingerprint", lambda: "after")
    changed = batch.run_batch(
        market, signals, output, config, plots=False, factors=["momentum"]
    )
    assert set(changed["status"]) == {"success"}
    assert changed.iloc[0]["run_id"] != first.iloc[0]["run_id"]
    metadata = json.loads(
        (
            output / "runs" / changed.iloc[0]["run_id"] / "identity.json"
        ).read_text()
    )
    assert metadata["implementation_sha256"] == "after"
    assert set(metadata["library_versions"]) == {"numpy", "pandas", "pyarrow"}


def test_single_and_parallel_results_match(
    inputs, tmp_path: pathlib.Path
) -> None:
    """Spawn workers must produce the same metrics and paths-safe accounts."""
    market, signals, config = inputs
    options = {"plots": False, "factors": ["momentum", "reversal"]}
    single = batch.run_batch(
        market, signals, tmp_path / "single", config, **options
    )
    parallel = batch.run_batch(
        market,
        signals,
        tmp_path / "parallel",
        config,
        workers=2,
        max_in_flight=1,
        **options,
    )
    assert set(parallel["status"]) == {"success"}
    pd.testing.assert_frame_equal(
        single.drop(columns="output_directory"),
        parallel.drop(columns="output_directory"),
    )
    for first, second in zip(
        single["output_directory"], parallel["output_directory"], strict=True
    ):
        pd.testing.assert_frame_equal(
            pd.read_parquet(pathlib.Path(first) / "pnl.parquet"),
            pd.read_parquet(pathlib.Path(second) / "pnl.parquet"),
        )


def test_factor_names_cannot_escape_output(
    inputs, tmp_path: pathlib.Path
) -> None:
    """External identifiers are names, never trusted filesystem paths."""
    market, signals, config = inputs
    frame = pd.read_parquet(signals)
    frame = frame[frame["factor"] == "momentum"].copy()
    frame["factor"] = "../../outside"
    frame.to_parquet(signals, index=False)
    output = tmp_path / "output"
    result = batch.run_batch(market, signals, output, config, plots=False)
    destination = pathlib.Path(result.iloc[0]["output_directory"])
    assert destination.is_relative_to(output)
    assert len(destination.name) == 64
    assert result.iloc[0]["status"] == "success"


def test_executed_layer_accounts_are_reported(
    inputs, tmp_path: pathlib.Path
) -> None:
    """Every requested layer has real net/gross engine account PnL."""
    market, signals, config = inputs
    result = batch.run_batch(
        market,
        signals,
        tmp_path / "layers",
        dataclasses.replace(config, simulate_layers=True),
        plots=False,
        factors=["momentum"],
    )
    assert result.iloc[0]["status"] == "success"
    destination = pathlib.Path(result.iloc[0]["output_directory"])
    accounts = pd.read_parquet(destination / "layer_pnl.parquet")
    assert sorted(accounts["layer"].unique()) == list(range(1, 11))
    assert accounts.groupby("layer").size().tolist() == [4] * 10
    assert {"net_nav", "gross_nav", "net_return", "benchmark_return"} <= (
        set(accounts.columns)
    )
    for layer in range(1, 11):
        directory = destination / "layers" / str(layer)
        metrics = json.loads((directory / "metrics.json").read_text())
        assert "restriction_count" in metrics
        trades = pd.read_parquet(directory / "trades.parquet")
        restrictions = pd.read_parquet(directory / "restrictions.parquet")
        assert len(trades) == metrics["trade_count"]
        assert len(restrictions) == metrics["restriction_count"]
