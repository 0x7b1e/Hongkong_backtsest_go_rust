"""Build stress fixtures that exercise every execution and data branch.

Starting from the synthetic examples, this adds cash dividends, share
splits, halts, missing quotes, untradable rows, thin volume, changing board
lots, VCM bands and states, stamp-duty exemptions, two constituent
snapshots, a benchmark open, and late, stale, cleared and date-only factor
publications. Each case also gets configuration variants that switch on
slippage, minimum commission, holding locks, small capital, direction,
keep, horizons, quantiles and execution windows.

Usage:
    python parity/make_stress.py --output /tmp/stress
"""

import argparse
import datetime
import json
import pathlib
import shutil

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _prices(source: pathlib.Path, rng: np.random.Generator) -> pd.DataFrame:
    prices = pd.read_parquet(source / "prices.parquet")
    times = sorted(prices["timestamp"].unique())
    days = sorted(pd.DatetimeIndex(times).normalize().unique())
    symbols = sorted(prices["symbol"].unique())
    day = pd.DatetimeIndex(prices["timestamp"]).normalize()
    day_number = day.map({d: i for i, d in enumerate(days)}).to_numpy()
    prices["cash_dividend"] = 0.0
    prices["share_multiplier"] = 1.0
    prices["halted"] = False
    prices["tradable"] = True
    prices["vcm_eligible"] = prices["symbol"].isin(symbols[:8])
    prices["vcm_active"] = False
    prices["vcm_lower"] = np.nan
    prices["vcm_upper"] = np.nan
    prices["security_type"] = "stock"
    prices.loc[prices["symbol"] == symbols[3], "security_type"] = "etf"
    prices.loc[prices["symbol"] == symbols[4], "security_type"] = "reit"
    first_bar = prices.groupby(["symbol", day_number])["timestamp"].transform("min")
    is_first = prices["timestamp"].eq(first_bar)
    # Dividends on the first bar of an ex-date.
    for symbol, number, amount in [(symbols[0], 6, 0.35), (symbols[5], 12, 0.8), (symbols[9], 21, 0.12)]:
        mask = prices["symbol"].eq(symbol) & (day_number == number) & is_first
        prices.loc[mask, "cash_dividend"] = amount
    # A 2-for-1 split with raw prices halving from the ex-date onward.
    split = prices["symbol"].eq(symbols[1]) & (day_number >= 14)
    for column in ["open", "high", "low", "close"]:
        prices.loc[split, column] = prices.loc[split, column] / 2
    prices.loc[prices["symbol"].eq(symbols[1]) & (day_number == 14) & is_first, "share_multiplier"] = 2.0
    # The first post-split open must reflect the split.
    # Halt, untradable, thin volume and zero volume rows.
    prices.loc[prices["symbol"].eq(symbols[2]) & day_number.__ge__(10) & day_number.__le__(12), "halted"] = True
    prices.loc[prices["symbol"].eq(symbols[6]) & (day_number == 8), "tradable"] = False
    thin = prices["symbol"].eq(symbols[7]) & (day_number % 5 == 0)
    prices.loc[thin, "volume"] = 300
    prices.loc[prices["symbol"].eq(symbols[8]) & (day_number == 17), "volume"] = 0
    # Board lots change during the sample.
    lots = {symbols[10]: 500, symbols[11]: 1000, symbols[12]: 2000, symbols[13]: 50}
    for symbol, lot in lots.items():
        prices.loc[prices["symbol"].eq(symbol), "lot_size"] = lot
    prices.loc[prices["symbol"].eq(symbols[10]) & (day_number >= 20), "lot_size"] = 200
    # VCM: an active event with tight bands, an inactive band and a NaN band.
    vcm = prices["symbol"].eq(symbols[0]) & (day_number == 9)
    prices.loc[vcm, "vcm_active"] = True
    prices.loc[vcm, "vcm_lower"] = prices.loc[vcm, "open"] * 1.001
    prices.loc[vcm, "vcm_upper"] = prices.loc[vcm, "open"] * 1.01
    band = prices["symbol"].eq(symbols[5]) & (day_number == 15)
    prices.loc[band, "vcm_lower"] = prices.loc[band, "open"] * 0.95
    prices.loc[band, "vcm_upper"] = prices.loc[band, "open"] * 1.05
    # Missing quotes for several sessions (rows absent).
    missing = prices["symbol"].eq(symbols[14]) & day_number.__ge__(20) & day_number.__le__(22)
    prices = prices.loc[~missing].reset_index(drop=True)
    prices["lot_size"] = prices["lot_size"].astype("int64")
    return prices


def _constituents(source: pathlib.Path) -> pd.DataFrame:
    base = pd.read_parquet(source / "constituents.parquet")
    symbols = sorted(base["symbol"])
    effective = base["effective_date"].iloc[0]
    later = effective + pd.Timedelta(days=28)
    weights = np.linspace(1, 2, len(symbols) - 2)
    weights = weights / weights.sum()
    second = pd.DataFrame(
        {
            "effective_date": later,
            "available_at": later - pd.Timedelta(days=3) + pd.Timedelta(hours=18),
            "symbol": symbols[:17] + symbols[18:19],
            "weight": weights,
        }
    )
    return pd.concat([base, second], ignore_index=True)


def _signals(source: pathlib.Path, destination: pathlib.Path, rng) -> None:
    for partition in sorted((source / "signals").iterdir()):
        frame = pd.read_parquet(partition / "part.parquet")
        frame = frame.copy()
        delay = rng.integers(0, 4, len(frame))
        late = rng.random(len(frame)) < 0.1
        frame.loc[late, "available_at"] = frame.loc[late, "available_at"] + pd.to_timedelta(
            delay[late] * 7, unit="h"
        )
        clears = rng.random(len(frame)) < 0.03
        frame.loc[clears, "value"] = np.nan
        target = destination / "signals" / partition.name
        target.mkdir(parents=True)
        frame.to_parquet(target / "part.parquet", index=False)
    # A date-labelled factor with stale revisions published after newer data.
    base = pd.read_parquet(source / "signals" / "factor=momentum_5" / "part.parquet")
    daily = base.assign(date=base["timestamp"].dt.tz_convert("Asia/Hong_Kong").dt.date)
    daily = daily.groupby(["date", "symbol"], as_index=False)["value"].last()
    daily["timestamp"] = daily["date"]
    daily["available_at"] = pd.to_datetime(daily["date"]).dt.tz_localize("Asia/Hong_Kong") + pd.Timedelta(hours=17)
    stale = daily.sample(frac=0.05, random_state=7).copy()
    stale["available_at"] = stale["available_at"] + pd.Timedelta(days=4)
    stale["timestamp"] = stale["date"].map(lambda d: d - datetime.timedelta(days=9))
    stale["value"] = stale["value"] * 3
    combined = pd.concat([daily, stale], ignore_index=True)
    combined = combined.drop_duplicates(["timestamp", "symbol"], keep="first")
    columns = ["timestamp", "available_at", "symbol", "value"]
    # Sorted last, so the dataset schema comes from a timestamp partition and
    # these date32 labels are cast to timestamps (midnight UTC) on read.
    target = destination / "signals" / "factor=zz_dated_momentum"
    target.mkdir(parents=True)
    combined[columns].to_parquet(target / "part.parquet", index=False)
    # Alone, date32 labels follow the date-only rule (16:00 HKT).
    alone = destination / "signals_dated" / "factor=dated_momentum"
    alone.mkdir(parents=True)
    combined[columns].to_parquet(alone / "part.parquet", index=False)


def _variants(case: dict) -> dict[str, dict]:
    variants = {
        "costs": {"slippage_bps": 7.5, "minimum_commission": 50.0, "commission_bps": 3.0},
        "keep_direction": {"keep": 0.4, "direction": -1, "quantiles": 5},
        "small_capital": {"initial_capital": 150000.0, "participation_rate": 0.02},
        "no_lots": {"round_lots": False, "ic_horizon_bars": 3, "ic_min_assets": 5},
        "pure_base": {"keep": 1.0, "simulate_layers": False},
    }
    if case["mode"] == "T1":
        variants["holding"] = {"holding_days": 3, "keep": 0.0}
        variants["late_start"] = {"t_start": "2024-01-22"}
    else:
        variants["windows"] = {
            "entry_start": "10:00",
            "entry_end": "15:00",
            "exit_start": "09:45",
            "exit_end": "15:55",
            "t0_flatten_at": "14:30",
            "order_slices": 4,
        }
        variants["no_flatten_window"] = {"t0_flatten_at": "16:30", "order_slices": 2}
    return {name: {**case, **values} for name, values in variants.items()}


def build(source: pathlib.Path, destination: pathlib.Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    destination.mkdir(parents=True)
    _prices(source, rng).to_parquet(destination / "prices.parquet", index=False)
    benchmark = pd.read_parquet(source / "benchmark.parquet")
    benchmark["open"] = benchmark["close"].shift(1).fillna(benchmark["close"].iloc[0] * 0.999)
    benchmark.to_parquet(destination / "benchmark.parquet", index=False)
    _constituents(source).to_parquet(destination / "constituents.parquet", index=False)
    shutil.copy(source / "sessions.parquet", destination / "sessions.parquet")
    _signals(source, destination, rng)
    config = json.loads((source / "config.json").read_text())
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    variants = destination / "variants"
    variants.mkdir()
    for name, values in _variants(config).items():
        (variants / f"{name}.json").write_text(json.dumps(values, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        shutil.rmtree(args.output)
    build(ROOT / "examples" / "daily", args.output / "stress_daily", 11)
    build(ROOT / "examples" / "intraday", args.output / "stress_intraday", 12)
    print(args.output)


if __name__ == "__main__":
    main()
