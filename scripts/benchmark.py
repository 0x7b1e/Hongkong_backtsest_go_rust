"""Measure the complete pipeline on reproducible synthetic market data."""

import argparse
import dataclasses
import datetime
import importlib.metadata
import json
import logging
import os
import pathlib
import platform
import resource
import sys
import time

from hkfactor import batch
from hkfactor import config as backtest_config
from hkfactor import data as market_data
from hkfactor import demo as demo_data


def _positive_int(value: str) -> int:
    """Parse a positive integer without silently accepting invalid sizes."""
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Must be a positive integer") from (
            error
        )
    if result < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return result


def _arguments() -> argparse.Namespace:
    """Parse dataset sizes and execution options for this measurement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        type=pathlib.Path,
        help="New directory for inputs, cache, and results",
    )
    parser.add_argument("--factors", type=_positive_int, default=16)
    parser.add_argument(
        "--days",
        type=_positive_int,
        default=1008,
        help="Synthetic business-day sessions",
    )
    parser.add_argument("--assets", type=_positive_int, default=80)
    parser.add_argument("--workers", type=_positive_int, default=2)
    parser.add_argument("--frequency", choices=["1d", "5m", "1m"], default="1d")
    parser.add_argument(
        "--layers",
        action="store_true",
        help="Execute ten additional layer accounts",
    )
    arguments = parser.parse_args()
    if arguments.days < 6:
        parser.error("--days must be at least 6 for five-bar factor history")
    if arguments.assets < 10:
        parser.error("--assets must be at least 10 for the synthetic universe")
    if arguments.output.exists():
        parser.error("--output must be a new directory")
    return arguments


def _peak_rss_mib(who: int) -> float:
    """Return the operating system's maximum-RSS counter in MiB."""
    counter = resource.getrusage(who).ru_maxrss
    divisor = 1024**2 if sys.platform == "darwin" else 1024
    return counter / divisor


def _environment() -> dict:
    """Describe hardware and package versions relevant to reproducibility."""
    try:
        memory = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        memory = None
    versions = {}
    for package in [
        "hk-factor-backtest",
        "numpy",
        "pandas",
        "pyarrow",
        "matplotlib",
    ]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed as package"
    return {
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "physical_memory_bytes": memory,
        },
        "software": {
            "platform": platform.platform(),
            "python": sys.version,
            "packages": versions,
        },
    }


def main() -> int:
    """Generate data, prepare the panel, and measure factor backtesting."""
    arguments = _arguments()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    output = arguments.output.resolve()
    output.mkdir(parents=True)
    elapsed = {}
    total_start = time.perf_counter()
    start = time.perf_counter()
    inputs = output / "inputs"
    synthetic = demo_data.create_demo(
        inputs,
        arguments.frequency,
        asset_count=arguments.assets,
        trading_days=arguments.days,
        factor_count=arguments.factors,
    )
    elapsed["generate_seconds"] = time.perf_counter() - start
    config = backtest_config.BacktestConfig(
        **json.loads(
            (inputs / "config.json").read_text(encoding="utf-8"),
        )
    )
    config = dataclasses.replace(config, simulate_layers=arguments.layers)
    market = output / "market"
    start = time.perf_counter()
    panel = market_data.prepare_market(
        inputs / "prices.parquet",
        inputs / "benchmark.parquet",
        inputs / "constituents.parquet",
        market,
        frequency=arguments.frequency,
        sessions=inputs / "sessions.parquet",
        t_start=config.t_start,
        t_end=config.t_end,
    )
    elapsed["prepare_seconds"] = time.perf_counter() - start
    panel_shape = panel.shape
    del panel
    start = time.perf_counter()
    summary = batch.run_batch(
        market,
        inputs / "signals",
        output / "results",
        config,
        workers=arguments.workers,
        resume=False,
        plots=False,
    )
    elapsed["backtest_seconds"] = time.perf_counter() - start
    elapsed["total_seconds"] = time.perf_counter() - total_start
    success = int((summary["status"] == "success").sum())
    failed = len(summary) - success
    report = {
        "synthetic": True,
        "note": "Synthetic business-day data; not real HSI constituents, "
        "an official Hong Kong calendar, or investment performance",
        "measured_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "dataset": synthetic,
        "panel_shape": panel_shape,
        "config": config.to_dict(),
        "workers": arguments.workers,
        "plots": False,
        "elapsed": elapsed,
        "successful_factors": success,
        "failed_factors": failed,
        "successful_factors_per_second": (
            success / elapsed["backtest_seconds"]
            if elapsed["backtest_seconds"] > 0
            else None
        ),
        "memory": {
            "parent_peak_rss_mib": _peak_rss_mib(resource.RUSAGE_SELF),
            "children_reported_peak_rss_mib": _peak_rss_mib(
                resource.RUSAGE_CHILDREN,
            ),
            "scope": "Separate process-lifetime ru_maxrss counters; "
            "children counter is not aggregate worker memory. "
            "macOS raw bytes and Linux raw KiB converted to MiB",
        },
        **_environment(),
    }
    target = output / "benchmark.json"
    target.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "benchmark": str(target),
                "elapsed": elapsed,
                "successful_factors": success,
                "failed_factors": failed,
            },
            indent=2,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
