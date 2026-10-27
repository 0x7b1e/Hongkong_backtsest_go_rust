"""Import, time alignment and held-out screening of daily HK factors."""

import dataclasses
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from hkfactor import data
from hkfactor import hf_daily
from hkfactor import mining
from hkfactor import signals


def test_hf_import_checks_schema_and_raw_basis(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "dataset" / "不复权"
    root.mkdir(parents=True)
    source = root / "00700.HK_腾讯控股.csv"
    source.write_text(
        "日期,代码,开盘,最高,最低,收盘,成交量\n"
        "2024-01-03,00700,101,103,99,102,1000\n"
        "2024-01-02,00700,100,102,98,101,1100\n",
        encoding="utf-8-sig",
    )
    with pytest.raises(ValueError, match="volume is shares"):
        hf_daily.import_hf_daily(
            tmp_path / "dataset", tmp_path / "bad", volume_unit="lots"
        )
    manifest = hf_daily.import_hf_daily(
        tmp_path / "dataset", tmp_path / "prices", volume_unit="shares"
    )
    frame = pd.read_parquet(tmp_path / "prices" / "part-00700.HK.parquet")
    assert manifest["basis"] == "raw"
    assert frame["timestamp"].tolist() == list(
        pd.to_datetime(["2024-01-02 16:00", "2024-01-03 16:00"])
    )
    assert frame["symbol"].unique().tolist() == ["00700.HK"]
    assert frame["price_basis"].unique().tolist() == ["raw"]
    assert manifest["files"][0]["has_corporate_actions"] is False
    benchmark = tmp_path / "benchmark.parquet"
    pd.DataFrame(
        {"timestamp": ["2024-01-02", "2024-01-03"], "close": [100, 101]}
    ).to_parquet(benchmark, index=False)
    constituents = tmp_path / "constituents.parquet"
    pd.DataFrame(
        {
            "effective_date": ["2024-01-02"],
            "available_at": ["2024-01-01 16:00"],
            "symbol": ["00700.HK"],
            "weight": [1.0],
        }
    ).to_parquet(constituents, index=False)
    panel = data.prepare_market(
        tmp_path / "prices", benchmark, constituents, tmp_path / "panel"
    )
    assert panel.price_basis == "raw"
    assert panel.symbols == ("00700.HK",)
    source.write_text(
        "日子,代码,开盘,最高,最低,收盘,成交量\n"
        "2024-01-02,00700,100,102,98,101,1100\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Missing CSV fields"):
        hf_daily.import_hf_daily(
            tmp_path / "dataset", tmp_path / "missing", volume_unit="shares"
        )
    assert not (tmp_path / "missing").exists()
    hf_daily.import_hf_daily(
        tmp_path / "dataset",
        tmp_path / "mapped",
        volume_unit="shares",
        columns={"timestamp": "日子"},
    )


def _example_panel(tmp_path: pathlib.Path):
    source = pathlib.Path(__file__).parents[1] / "examples" / "daily"
    output = tmp_path / "market"
    panel = data.prepare_market(
        source / "prices.parquet",
        source / "benchmark.parquet",
        source / "constituents.parquet",
        output,
        "1d",
        source / "sessions.parquet",
    )
    return source, output, panel


def test_publication_delay_and_prefix_causality(tmp_path: pathlib.Path) -> None:
    _, output, panel = _example_panel(tmp_path)
    recipe = mining.Recipe("momentum_2", "momentum", 2)
    mining.generate_signals(panel, [recipe], tmp_path / "signals")
    store = signals.SignalStore(tmp_path / "signals")
    aligned = store.read_factor(recipe.factor_id, data.load_panel(output))
    column = next(
        i
        for i, symbol in enumerate(panel.symbols)
        if np.isfinite(panel.close[:5, i]).all()
    )
    expected = panel.close[2, column] / panel.close[0, column] - 1
    assert np.isnan(aligned.values[2, column])
    assert aligned.values[3, column] == pytest.approx(expected)
    recipe_2 = mining.Recipe("low_vol_5", "volatility", 5, -1)
    mining._audit_prefix(panel, recipe_2)


def test_momentum_accounts_for_a_published_split(
    tmp_path: pathlib.Path,
) -> None:
    _, _, original = _example_panel(tmp_path)
    close = np.array(original.close, copy=True)
    open_prices = np.array(original.open, copy=True)
    multipliers = np.array(original.share_multiplier, copy=True)
    volume = np.array(original.volume, copy=True)
    close[:3, 0] = [100, 50, 52]
    open_prices[:3, 0] = [100, 50, 52]
    multipliers[1, 0] = 2
    volume[:3, 0] = 1_000
    panel = dataclasses.replace(
        original,
        close=close,
        open=open_prices,
        share_multiplier=multipliers,
        volume=volume,
    )
    values = mining._series(panel, 0, mining.Recipe("split", "momentum", 2))
    assert values[2] == pytest.approx(0.04)


def test_mine_freezes_selection_before_test(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, market, _ = _example_panel(tmp_path)
    base = json.loads((source / "config.json").read_text(encoding="utf-8"))
    base.update({"simulate_layers": False, "round_lots": False})
    config = tmp_path / "config.json"
    config.write_text(json.dumps(base), encoding="utf-8")
    recipes = [
        mining.Recipe("intraday_1", "intraday", 1),
        mining.Recipe("momentum_3", "momentum", 3),
    ]
    actual_run = mining._engine_run

    def checked_run(
        engine,
        market_path,
        signal_path,
        config_object,
        output_path,
        factors,
        workers,
    ):
        if output_path.name == "test":
            frozen = json.loads(
                (tmp_path / "mine" / "selection.json").read_text()
            )
            assert frozen["frozen_before_test"]
            assert frozen["chosen"] == factors
        return actual_run(
            engine,
            market_path,
            signal_path,
            config_object,
            output_path,
            factors,
            workers,
        )

    monkeypatch.setattr(mining, "_engine_run", checked_run)
    report = mining.mine(
        market,
        config,
        tmp_path / "mine",
        train_end="2024-01-24",
        valid_end="2024-02-09",
        recipes=recipes,
        rust_bin=None,
        workers=1,
        top_k=2,
        select_k=1,
        min_trades=0,
        min_valid_excess=-10,
        price_only_research=True,
    )
    assert report["selection"]["frozen_before_test"]
    assert len(report["selection"]["shortlist"]) >= 1
    assert len(report["selection"]["chosen"]) == 1
    assert len(report["test_results"]) == 1
    assert (tmp_path / "mine" / "selection.json").exists()
    assert (tmp_path / "mine" / "test" / "results" / "summary.csv").exists()


def test_recipe_validation_forbids_unsafe_code(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "recipes.jsonl"
    path.write_text(
        json.dumps(
            {
                "factor_id": "bad",
                "kind": "python",
                "window": 1,
                "code": "exec('anything')",
            }
        )
    )
    with pytest.raises(TypeError):
        mining.read_recipes(path)
    with pytest.raises(ValueError, match="Invalid factor_id"):
        mining.Recipe("../../outside", "momentum", 3)
