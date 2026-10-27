"""Causal daily factor recipes, publication-aware signals and Rust screening."""

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from hkfactor import config as backtest_config
from hkfactor import data as market_data
from hkfactor import hf_daily
from hkfactor import models
from hkfactor import signals as factor_signals

_KINDS = {"momentum", "volatility", "relative_volume", "intraday", "range"}
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclasses.dataclass(frozen=True)
class Recipe:
    """One safe, bounded expression template; sign orients high scores up."""

    factor_id: str
    kind: str
    window: int
    sign: int = 1
    hypothesis: str = ""

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.factor_id):
            raise ValueError(f"Invalid factor_id: {self.factor_id!r}")
        if self.kind not in _KINDS:
            raise ValueError(f"Unsupported factor kind: {self.kind}")
        if type(self.window) is not int or not 1 <= self.window <= 252:
            raise ValueError("window must be an integer from 1 to 252")
        if type(self.sign) is not int or self.sign not in {-1, 1}:
            raise ValueError("sign must be +1 or -1")
        if self.kind == "intraday" and self.window != 1:
            raise ValueError("intraday requires window=1")
        if self.kind == "volatility" and self.window < 2:
            raise ValueError("volatility needs at least two returns")


DEFAULT_RECIPES = (
    Recipe("momentum_5", "momentum", 5, 1, "Medium-term price continuation"),
    Recipe("momentum_20", "momentum", 20, 1, "Slower price continuation"),
    Recipe("momentum_60", "momentum", 60, 1, "Quarterly trend"),
    Recipe("reversal_2", "momentum", 2, -1, "Short-term reversal"),
    Recipe("reversal_5", "momentum", 5, -1, "Weekly reversal"),
    Recipe("low_vol_20", "volatility", 20, -1, "Low realized volatility"),
    Recipe("volume_5", "relative_volume", 5, 1, "Unusual share volume"),
    Recipe("volume_20", "relative_volume", 20, 1, "Unusual monthly volume"),
    Recipe("intraday_1", "intraday", 1, 1, "Open-to-close strength"),
    Recipe("low_range_10", "range", 10, -1, "Low intraday range"),
)


def read_recipes(path: str | pathlib.Path | None) -> list[Recipe]:
    """Read JSONL recipes without executing arbitrary factor code."""
    if path is None:
        return list(DEFAULT_RECIPES)
    lines = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
    recipes = [Recipe(**json.loads(line)) for line in lines if line.strip()]
    if not recipes or len(recipes) > 64:
        raise ValueError("Supply 1 to 64 candidate recipes")
    if len({recipe.factor_id for recipe in recipes}) != len(recipes):
        raise ValueError("Duplicate factor_id in recipes")
    return recipes


def _series(
    panel: models.MarketPanel, column: int, recipe: Recipe
) -> np.ndarray:
    """Compute one security independently from current and past daily bars."""
    close = pd.Series(np.asarray(panel.close[:, column], dtype=float))
    opening = pd.Series(np.asarray(panel.open[:, column], dtype=float))
    high = pd.Series(np.asarray(panel.high[:, column], dtype=float))
    low = pd.Series(np.asarray(panel.low[:, column], dtype=float))
    volume = pd.Series(np.asarray(panel.volume[:, column], dtype=float))
    dividends = pd.Series(
        np.asarray(panel.cash_dividend[:, column], dtype=float)
    )
    multipliers = pd.Series(
        np.asarray(panel.share_multiplier[:, column], dtype=float)
    )
    window = recipe.window
    with np.errstate(divide="ignore", invalid="ignore"):
        if recipe.kind in {"momentum", "volatility"}:
            # Corporate actions on the current date affect shares and cash
            # owned at yesterday's close. Suspended sessions break a window.
            gross = (close * multipliers + dividends) / close.shift(1)
            gross[(volume <= 0) | (volume.shift(1) <= 0)] = np.nan
            if recipe.kind == "momentum":
                score = np.expm1(
                    np.log(gross).rolling(window, min_periods=window).sum()
                )
            else:
                score = (
                    (gross - 1).rolling(window, min_periods=window).std(ddof=0)
                )
        elif recipe.kind == "relative_volume":
            baseline = (
                volume.shift(1).rolling(window, min_periods=window).mean()
            )
            score = volume / baseline - 1
        elif recipe.kind == "intraday":
            score = close / opening - 1
        else:
            score = (
                ((high - low) / close)
                .rolling(window, min_periods=window)
                .mean()
            )
    values = score.to_numpy(dtype=float) * recipe.sign
    values[~np.isfinite(values)] = np.nan
    # Today's zero volume is known after close; clear yesterday's score.
    values[np.asarray(panel.volume[:, column]) <= 0] = np.nan
    return values


def _audit_prefix(panel: models.MarketPanel, recipe: Recipe) -> None:
    """Catch accidental future-dependence in a changed recipe implementation."""
    # _series only accesses one column; a truncated lightweight panel is
    # sufficient to detect a forward-looking operator without copying data.
    for column in {0, panel.shape[1] // 2, panel.shape[1] - 1}:
        full = _series(panel, column, recipe)
        for stop in {
            min(panel.shape[0], recipe.window + 2),
            panel.shape[0] // 2,
        }:
            if stop < 1:
                continue
            prefix = _series(_PrefixPanel(panel, stop), column, recipe)
            if not np.allclose(full[:stop], prefix, equal_nan=True):
                raise ValueError(
                    f"Prefix causality audit failed: {recipe.factor_id}"
                )


class _PrefixPanel:
    """Expose only an initial set of rows to the recipe calculator."""

    def __init__(self, panel: models.MarketPanel, stop: int) -> None:
        for name in (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "cash_dividend",
            "share_multiplier",
        ):
            setattr(self, name, getattr(panel, name)[:stop])


def generate_signals(
    panel: models.MarketPanel,
    recipes: list[Recipe],
    directory: pathlib.Path,
    *,
    publication_delay: str = "10min",
) -> dict:
    """Write one hive-partitioned Parquet factor at a time, including clears."""
    delay = pd.Timedelta(publication_delay)
    if delay < pd.Timedelta(0) or delay > pd.Timedelta(days=1):
        raise ValueError("publication_delay must be in [0, 1 day]")
    if panel.frequency != "1d" or panel.price_basis != "raw":
        raise ValueError("Daily mining requires a raw-price 1d market panel")
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(parents=True)
    timestamp = panel.timestamps.tz_convert("UTC")
    available = timestamp + delay
    schema = pa.schema(
        [
            pa.field("timestamp", pa.timestamp("ns", tz="UTC")),
            pa.field("available_at", pa.timestamp("ns", tz="UTC")),
            pa.field("symbol", pa.string()),
            pa.field("value", pa.float64()),
        ]
    )
    for recipe in recipes:
        _audit_prefix(panel, recipe)
        partition = directory / f"factor={recipe.factor_id}"
        partition.mkdir()
        with pq.ParquetWriter(
            partition / "part-00000.parquet", schema
        ) as writer:
            for start in range(0, panel.shape[1], 16):
                frames = []
                for column in range(start, min(start + 16, panel.shape[1])):
                    frames.append(
                        pd.DataFrame(
                            {
                                "timestamp": timestamp,
                                "available_at": available,
                                "symbol": panel.symbols[column],
                                "value": _series(panel, column, recipe),
                            }
                        )
                    )
                table = pa.Table.from_pandas(
                    pd.concat(frames, ignore_index=True),
                    schema=schema,
                    preserve_index=False,
                )
                writer.write_table(table)
    return {
        "signal_count": panel.shape[0] * panel.shape[1] * len(recipes),
        "factor_count": len(recipes),
        "publication_delay": str(delay),
        "observation_time": "vendor daily bar marked 16:00 Asia/Hong_Kong",
        "execution_timing": (
            "At delay 0, next session open (requires proof final prices are "
            "published by 16:00); with positive delay, current engine "
            "uses the signal at the open after the next completed session"
        ),
    }


def _splits(
    panel: models.MarketPanel,
    config: backtest_config.BacktestConfig,
    train_end: str,
    valid_end: str,
) -> dict[str, tuple[str, str]]:
    sessions = sorted(
        {
            day.isoformat()
            for day in panel.timestamps.date
            if config.t_start <= day.isoformat() <= config.t_end
        }
    )
    if train_end not in sessions or valid_end not in sessions:
        raise ValueError(
            "train_end and valid_end must be actual panel sessions"
        )
    if not (config.t_start <= train_end < valid_end < config.t_end):
        raise ValueError("Expected t_start <= train_end < valid_end < t_end")
    train = [day for day in sessions if day <= train_end]
    valid = [day for day in sessions if train_end < day <= valid_end]
    test = [day for day in sessions if day > valid_end]
    if min(len(train), len(valid), len(test)) < 2:
        raise ValueError(
            "Each of train, validation and test needs two sessions"
        )
    return {
        "train": (train[0], train[-1]),
        "validation": (valid[0], valid[-1]),
        "test": (test[0], test[-1]),
    }


def _shortlist(
    summary: pd.DataFrame,
    recipes: list[Recipe],
    panel: models.MarketPanel,
    train_end: str,
    top_k: int,
    min_trades: int,
    max_correlation: float,
) -> list[str]:
    """Use training account results and training signal correlations only."""
    frame = summary.loc[summary["status"].eq("success")].copy()
    frame["excess_total_return"] = pd.to_numeric(
        frame["excess_total_return"], errors="coerce"
    )
    frame["trade_count"] = pd.to_numeric(frame["trade_count"], errors="coerce")
    frame = frame.loc[
        frame["excess_total_return"].notna()
        & (frame["trade_count"] >= min_trades)
    ].sort_values(["excess_total_return", "factor"], ascending=[False, True])
    by_id = {recipe.factor_id: recipe for recipe in recipes}
    days = np.flatnonzero(
        np.asarray(panel.timestamps.date) <= pd.Timestamp(train_end).date()
    )
    days = days[:: max(1, len(days) // 250)]
    columns = np.unique(
        np.linspace(0, panel.shape[1] - 1, min(panel.shape[1], 32), dtype=int)
    )
    sampled = {}
    selected = []
    for name in frame["factor"]:
        recipe = by_id[name]
        values = np.stack(
            [_series(panel, int(col), recipe)[days] for col in columns], axis=1
        )
        eligible = np.asarray(panel.base_weights[np.ix_(days, columns)]) > 0
        values[~eligible] = np.nan
        # Center within each day's cross section.
        counts = np.isfinite(values).sum(axis=1, keepdims=True)
        centered = values - (
            np.nansum(values, axis=1, keepdims=True) / np.maximum(counts, 1)
        )
        current = centered.ravel()
        duplicate = False
        for previous in sampled.values():
            finite = np.isfinite(current) & np.isfinite(previous)
            if (
                finite.sum() >= 10
                and np.std(current[finite]) > 0
                and np.std(previous[finite]) > 0
            ):
                correlation = np.corrcoef(current[finite], previous[finite])[
                    0, 1
                ]
                if abs(correlation) >= max_correlation:
                    duplicate = True
                    break
        if not duplicate:
            selected.append(name)
            sampled[name] = current
        if len(selected) >= top_k:
            break
    return selected


def _engine_run(
    engine: list[str],
    market: pathlib.Path,
    signals: pathlib.Path,
    config: backtest_config.BacktestConfig,
    output: pathlib.Path,
    factors: list[str],
    workers: int,
) -> pd.DataFrame:
    output.mkdir(parents=True)
    config_path = output / "config.json"
    config_path.write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    if not factors:
        return pd.DataFrame()
    command = [
        *engine,
        "run",
        "--market",
        str(market),
        "--signals",
        str(signals),
        "--config",
        str(config_path),
        "--output",
        str(output / "results"),
        "--workers",
        str(workers),
        "--no-plots",
        "--no-layer-execution",
    ]
    for name in factors:
        command.extend(("--factor", name))
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False
    )
    (output / "engine.stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (output / "engine.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    summary_path = output / "results" / "summary.csv"
    if completed.returncode not in {0, 1} or not summary_path.exists():
        raise RuntimeError(
            f"Backtest failed: {command[0]} "
            f"(exit {completed.returncode}); see {output}"
        )
    frame = pd.read_csv(summary_path, encoding="utf-8-sig")
    if set(frame["factor"]) != set(factors):
        raise ValueError("Engine returned an incomplete factor summary")
    return frame


def _ic_report(frame: pd.DataFrame, split_end: str) -> list[dict[str, Any]]:
    result = []
    for row in frame.itertuples(index=False):
        if row.status != "success":
            continue
        ic = pd.read_parquet(pathlib.Path(row.output_directory) / "ic.parquet")
        # The engine labels the forward end; enforce split boundaries here too.
        ic = ic.loc[
            pd.to_datetime(ic["forward_end"]).dt.date
            <= pd.Timestamp(split_end).date()
        ]
        scores = pd.to_numeric(ic["spearman_ic"], errors="coerce")
        result.append(
            {
                "factor": row.factor,
                "spearman_ic_mean": float(scores.mean())
                if scores.notna().any()
                else None,
                "ic_sessions": int(scores.count()),
            }
        )
    return result


def mine(
    market: pathlib.Path,
    config_path: pathlib.Path,
    output: pathlib.Path,
    *,
    train_end: str,
    valid_end: str,
    recipes: list[Recipe],
    rust_bin: pathlib.Path | None,
    workers: int = 2,
    top_k: int = 8,
    select_k: int = 3,
    min_trades: int = 1,
    min_valid_excess: float = 0.0,
    max_correlation: float = 0.95,
    publication_delay: str = "10min",
    price_only_research: bool = False,
    corporate_actions_verified: bool = False,
) -> dict:
    """Select on train/validation, freeze choices, then open held-out test."""
    from hkfactor.cli import _config

    market = market.resolve()
    config_path = config_path.resolve()
    if output.exists():
        raise FileExistsError(output)
    if workers < 1 or top_k < 1 or select_k < 1 or min_trades < 0:
        raise ValueError(
            "workers/top_k/select_k must be positive; min_trades >= 0"
        )
    if not 0 < max_correlation <= 1:
        raise ValueError("max_correlation must be in (0, 1]")
    config, columns = _config(str(config_path))
    if columns != factor_signals.SignalColumns():
        raise ValueError(
            "Mining emits canonical signal columns; "
            "remove signal_columns mapping"
        )
    if config.frequency != "1d" or config.mode != "T1" or config.direction != 1:
        raise ValueError(
            "Mining requires daily T1 configuration with direction=1"
        )
    panel = market_data.load_panel(market)
    if panel.price_basis != "raw" or config.allow_adjusted_prices:
        raise ValueError("Executed mining must use raw prices only")
    if price_only_research and corporate_actions_verified:
        raise ValueError(
            "Choose either verified actions or price-only research"
        )
    if not price_only_research and not corporate_actions_verified:
        raise ValueError(
            "Verify complete historical dividend/split data with "
            "--corporate-actions-verified "
            "or explicitly use --price-only-research"
        )
    if config.round_lots and np.any(
        (~np.isfinite(panel.lot_sizes) | (panel.lot_sizes <= 0))
        & (panel.base_weights > 0)
        & panel.tradable
    ):
        raise ValueError(
            "Historical board-lot sizes required for round_lots=true"
        )
    splits = _splits(panel, config, train_end, valid_end)
    if rust_bin is None:
        engine = [sys.executable, "-m", "hkfactor"]
        engine_identity = "python development parity runner"
    else:
        binary = rust_bin.resolve()
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise FileNotFoundError(
                f"Build the Rust executable first: {binary}"
            )
        engine = [str(binary)]
        engine_identity = hashlib.sha256(binary.read_bytes()).hexdigest()
    output.mkdir(parents=True)
    (output / "recipes.jsonl").write_text(
        "".join(
            json.dumps(dataclasses.asdict(r), ensure_ascii=False) + "\n"
            for r in recipes
        ),
        encoding="utf-8",
    )
    signal_manifest = generate_signals(
        panel, recipes, output / "signals", publication_delay=publication_delay
    )
    manifest = {
        "market_fingerprint": market_data.panel_fingerprint(market),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "engine_sha256_or_description": engine_identity,
        "splits": splits,
        "signals": signal_manifest,
        "selection_rule": (
            "Train: rank by net excess total return and remove correlated "
            "training scores; validation: net excess >= threshold; "
            "freeze before test"
        ),
        "price_only_research": price_only_research,
        "corporate_actions_verified_by_caller": corporate_actions_verified,
        "no_future_bar_data_in_features": True,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    def run_split(name: str, names: list[str]) -> pd.DataFrame:
        first, last = splits[name]
        subconfig = dataclasses.replace(
            config, t_start=first, t_end=last, simulate_layers=False
        )
        return _engine_run(
            engine,
            market,
            output / "signals",
            subconfig,
            output / name,
            names,
            workers,
        )

    train = run_split("train", [r.factor_id for r in recipes])
    shortlist = _shortlist(
        train, recipes, panel, train_end, top_k, min_trades, max_correlation
    )
    validation = run_split("validation", shortlist)
    qualified = (
        validation.loc[validation["status"].eq("success")].copy()
        if not validation.empty
        else validation
    )
    if not qualified.empty:
        qualified["excess_total_return"] = pd.to_numeric(
            qualified["excess_total_return"], errors="coerce"
        )
        qualified = qualified.loc[
            qualified["excess_total_return"] >= min_valid_excess
        ].sort_values(
            ["excess_total_return", "factor"], ascending=[False, True]
        )
    chosen = (
        qualified["factor"].head(select_k).tolist()
        if not qualified.empty
        else []
    )
    selection = {
        "shortlist": shortlist,
        "chosen": chosen,
        "min_valid_excess": min_valid_excess,
        "training_ic": _ic_report(train, train_end),
        "validation_ic": _ic_report(validation, valid_end),
        "frozen_before_test": True,
    }
    (output / "selection.json").write_text(
        json.dumps(selection, indent=2, allow_nan=False), encoding="utf-8"
    )
    test = run_split("test", chosen)
    report = {
        **manifest,
        "selection": selection,
        "test_ic": _ic_report(test, splits["test"][1]),
        "test_results": json.loads(test.to_json(orient="records"))
        if not test.empty
        else [],
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False, default=str),
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """Provide separate import, recipe and Rust-guided mining entry points."""
    parser = argparse.ArgumentParser(prog="hkfactor-mine")
    commands = parser.add_subparsers(dest="command", required=True)
    importer = commands.add_parser("import-hf")
    importer.add_argument("--dataset-root", required=True)
    importer.add_argument("--output", required=True)
    importer.add_argument("--basis", choices=["raw", "adjusted"], default="raw")
    importer.add_argument(
        "--volume-unit", choices=["shares", "lots"], required=True
    )
    importer.add_argument(
        "--columns", help="JSON object canonical field -> exact CSV header"
    )
    importer.add_argument("--symbol", action="append", dest="symbols")
    defaults = commands.add_parser("recipes")
    defaults.add_argument("--output", required=True)
    runner = commands.add_parser("run")
    for key in ("market", "config", "output", "train-end", "valid-end"):
        runner.add_argument("--" + key, required=True)
    runner.add_argument("--recipes")
    runner.add_argument("--rust-bin", default="rust/target/release/hkfactor")
    runner.add_argument("--engine", choices=["rust", "python"], default="rust")
    runner.add_argument("--workers", type=int, default=2)
    runner.add_argument("--top-k", type=int, default=8)
    runner.add_argument("--select-k", type=int, default=3)
    runner.add_argument("--min-trades", type=int, default=1)
    runner.add_argument("--min-valid-excess", type=float, default=0.0)
    runner.add_argument("--max-correlation", type=float, default=0.95)
    runner.add_argument("--publication-delay", default="10min")
    runner.add_argument("--price-only-research", action="store_true")
    runner.add_argument("--corporate-actions-verified", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "import-hf":
            manifest = hf_daily.import_hf_daily(
                args.dataset_root,
                args.output,
                basis=args.basis,
                columns=json.loads(
                    pathlib.Path(args.columns).read_text(encoding="utf-8")
                )
                if args.columns
                else None,
                volume_unit=args.volume_unit,
                symbols=set(args.symbols) if args.symbols else None,
            )
            print(
                json.dumps(
                    {
                        "securities": manifest["security_count"],
                        "rows": manifest["row_count"],
                    }
                )
            )
        elif args.command == "recipes":
            path = pathlib.Path(args.output)
            if path.exists():
                raise FileExistsError(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "".join(
                    json.dumps(dataclasses.asdict(r), ensure_ascii=False) + "\n"
                    for r in DEFAULT_RECIPES
                ),
                encoding="utf-8",
            )
            print(path)
        else:
            result = mine(
                pathlib.Path(args.market),
                pathlib.Path(args.config),
                pathlib.Path(args.output),
                train_end=args.train_end,
                valid_end=args.valid_end,
                recipes=read_recipes(args.recipes),
                rust_bin=pathlib.Path(args.rust_bin)
                if args.engine == "rust"
                else None,
                workers=args.workers,
                top_k=args.top_k,
                select_k=args.select_k,
                min_trades=args.min_trades,
                min_valid_excess=args.min_valid_excess,
                max_correlation=args.max_correlation,
                publication_delay=args.publication_delay,
                price_only_research=args.price_only_research,
                corporate_actions_verified=args.corporate_actions_verified,
            )
            print(
                json.dumps(
                    {
                        "chosen": result["selection"]["chosen"],
                        "test_results": len(result["test_results"]),
                        "report": str(
                            pathlib.Path(args.output) / "report.json"
                        ),
                    }
                )
            )
    except (
        ValueError,
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
