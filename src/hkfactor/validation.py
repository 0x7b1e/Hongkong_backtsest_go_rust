"""Local data audits, CSV adapters, and provider-snapshot comparisons."""

import hashlib
import json
import pathlib
import tempfile
from typing import Protocol

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from hkfactor import csv_data
from hkfactor import data as market_data


class PriceProvider(Protocol):
    """An authenticated external provider supplied by the caller."""

    def fetch_prices(
        self, symbol: str, t_start: str, t_end: str, frequency: str
    ) -> pd.DataFrame:
        """Return canonical timestamp/symbol/OHLC/volume observations."""
        ...


def import_csv(
    source: str | pathlib.Path,
    destination: str | pathlib.Path,
    frequency: str,
    symbol: str,
    price_basis: str = "adjusted",
    kind: str = "stock",
) -> None:
    """Adapt existing CSVs to Parquet without editing the source.

    Args:
        source: Existing Chinese-header CSV, optionally with a title row.
        destination: New Parquet file; existing files are never overwritten.
        frequency: Explicit source bar frequency.
        symbol: Canonical identifier if the source omits symbol.
        price_basis: raw or adjusted; never infer raw from adjusted files.
        kind: stock or benchmark; benchmarks retain close and available open.
    """
    source, destination = pathlib.Path(source), pathlib.Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if price_basis not in {"raw", "adjusted"} or kind not in {
        "stock",
        "benchmark",
    }:
        raise ValueError("Invalid CSV kind or price basis")
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for frame in csv_data.iter_csv_frames(source):
            frame["timestamp"] = csv_data.csv_timestamps(frame, frequency)
            if "symbol" not in frame:
                frame["symbol"] = symbol
            if (frame["symbol"] != symbol).any():
                raise ValueError("CSV contains unexpected security identifiers")
            columns = ["timestamp", "symbol", "close"]
            if kind == "benchmark":
                columns = ["timestamp", "symbol"] + [
                    column
                    for column in (
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "amount",
                    )
                    if column in frame
                ]
            if kind == "stock":
                frame["price_basis"] = price_basis
                columns += ["open", "high", "low", "volume", "price_basis"]
            for column in ["open", "high", "low", "close", "volume", "amount"]:
                if column in frame:
                    frame[column] = pd.to_numeric(
                        frame[column], errors="raise"
                    ).astype("float64")
            table = pa.Table.from_pandas(frame[columns], preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema)
            writer.write_table(table)
        if writer is None:
            raise ValueError("CSV has no records")
    except BaseException:
        if writer is not None:
            writer.close()
        destination.unlink(missing_ok=True)
        raise
    else:
        writer.close()


def audit_local_data(directory: str | pathlib.Path) -> dict:
    """Describe available files and research-blocking data gaps."""
    records = []
    stock_symbols = set()
    has_stock_intraday = False
    for filename in sorted(pathlib.Path(directory).rglob("*.csv")):
        if filename.name.startswith("._"):
            continue
        encoding, skiprows = csv_data.csv_options(filename)
        item = {
            "path": str(filename),
            "encoding": encoding,
            "header_row": skiprows + 1,
            "rows": 0,
        }
        dates, seen = [], set()
        duplicate_count, missing_lots = 0, None
        for frame in csv_data.iter_csv_frames(filename):
            item["rows"] += len(frame)
            item["columns"] = list(frame.columns)
            if "date" in frame:
                frequency = "1d" if "time" not in frame else "1m"
                times = csv_data.csv_timestamps(frame, frequency)
                dates.extend([times.min(), times.max()])
                symbols = frame.get(
                    "symbol", pd.Series("HSI", index=frame.index)
                )
                for key in zip(times.asi8, symbols, strict=True):
                    duplicate_count += key in seen
                    seen.add(key)
                if "symbol" in frame:
                    stock_symbols.update(frame["symbol"].dropna().astype(str))
                    has_stock_intraday |= "time" in frame
            if "lot_size" in frame:
                missing_lots = (missing_lots or 0) + int(
                    frame["lot_size"].isna().sum()
                )
        item["duplicate_keys"] = duplicate_count
        if dates:
            item["date_start"] = min(dates).date().isoformat()
            item["date_end"] = max(dates).date().isoformat()
        if missing_lots is not None:
            item["missing_lot_sizes"] = missing_lots
        records.append(item)
    return {
        "files": records,
        "stock_symbols_with_prices": sorted(stock_symbols),
        "stock_intraday_available": has_stock_intraday,
        "ready_for_full_hsi_backtest": False,
        "requirements": [
            "Historical full HSI constituent snapshots with weights and "
            "publication/effective times",
            "Raw stock OHLCV covering every historical constituent",
            "Stock minute bars and official session calendar for T0",
            "Historical board-lot sizes; current reference rows are missing "
            "lot_size",
            "Corporate actions and VCM/halt metadata for rule coverage",
            "External factor Parquet data or mapped equivalent fields",
        ],
        "api_comparison": {
            "status": "not_run",
            "reason": "No provider or authenticated reference was supplied",
        },
    }


def _symbols(dataset: ds.Dataset) -> list[str]:
    if "symbol" not in dataset.schema.names:
        return ["HSI"]
    symbols = set()
    for batch in dataset.scanner(columns=["symbol"]).to_batches():
        symbols.update(batch.column(0).to_pylist())
    return sorted(symbols)


def _symbol_frame(
    dataset: ds.Dataset, symbol: str, frequency: str
) -> pd.DataFrame:
    expression = (
        ds.field("symbol") == symbol
        if "symbol" in dataset.schema.names
        else None
    )
    frame = dataset.to_table(filter=expression).to_pandas()
    if "timestamp" not in frame:
        raise ValueError("Comparison inputs require canonical timestamps")
    frame["timestamp"] = market_data.normalize_times(
        frame["timestamp"], frequency
    )
    if frame["timestamp"].duplicated().any():
        raise ValueError("Comparison input has duplicate timestamp/symbol")
    return frame.set_index("timestamp").sort_index()


def compare_snapshots(
    local: str | pathlib.Path,
    reference: str | pathlib.Path,
    output: str | pathlib.Path,
    frequency: str = "1d",
    atol: float = 1e-8,
    rtol: float = 1e-6,
    max_differences: int = 10_000,
) -> dict:
    """Compare provider-exported canonical quotes, one security at a time.

    Args:
        local: Local Parquet dataset.
        reference: Authenticated API export in the same price/volume units.
        output: New output directory for full counts and bounded examples.
        frequency: Timestamp convention shared by both datasets.
        atol: Absolute price/value comparison tolerance.
        rtol: Relative comparison tolerance.
        max_differences: Maximum detailed examples, independent of full counts.

    Returns:
        Exact mismatch and missing-observation counts, with truncation status.
    """
    if min(atol, rtol) < 0 or max_differences < 1:
        raise ValueError("Invalid comparison tolerance or example budget")
    output = pathlib.Path(output)
    if output.exists():
        raise FileExistsError(output)
    left = ds.dataset(str(local), format="parquet", partitioning="hive")
    right = ds.dataset(str(reference), format="parquet", partitioning="hive")
    symbols = sorted(set(_symbols(left)) | set(_symbols(right)))
    summaries, differences = [], []
    for symbol in symbols:
        local_frame = _symbol_frame(left, symbol, frequency)
        reference_frame = _symbol_frame(right, symbol, frequency)
        shared = local_frame.index.intersection(reference_frame.index)
        missing_local = reference_frame.index.difference(local_frame.index)
        missing_reference = local_frame.index.difference(reference_frame.index)
        fields_to_compare = sorted(
            {"open", "high", "low", "close", "volume", "amount"}
            & set(local_frame)
            & set(reference_frame)
        )
        if not fields_to_compare:
            raise ValueError("No shared quote fields to compare")
        summary = {
            "symbol": symbol,
            "matched_rows": len(shared),
            "missing_local": len(missing_local),
            "missing_reference": len(missing_reference),
            "mismatched_cells": 0,
        }
        for kind, times in [
            ("missing_local", missing_local),
            ("missing_reference", missing_reference),
        ]:
            for timestamp in times[: max_differences - len(differences)]:
                differences.append(
                    {
                        "symbol": symbol,
                        "timestamp": timestamp,
                        "reason": kind,
                        "field": None,
                        "local": np.nan,
                        "reference": np.nan,
                    }
                )
        for field in fields_to_compare:
            actual = local_frame.loc[shared, field].to_numpy(float)
            expected = reference_frame.loc[shared, field].to_numpy(float)
            unequal = ~np.isclose(
                actual, expected, atol=atol, rtol=rtol, equal_nan=True
            )
            summary[f"{field}_mismatches"] = int(unequal.sum())
            summary["mismatched_cells"] += int(unequal.sum())
            indices = np.flatnonzero(unequal)
            for index in indices[: max_differences - len(differences)]:
                differences.append(
                    {
                        "symbol": symbol,
                        "timestamp": shared[index],
                        "reason": "value_mismatch",
                        "field": field,
                        "local": actual[index],
                        "reference": expected[index],
                    }
                )
        summaries.append(summary)
    total = sum(
        row["mismatched_cells"]
        + row["missing_local"]
        + row["missing_reference"]
        for row in summaries
    )
    result = {
        "status": "completed",
        "local": str(local),
        "reference": str(reference),
        "atol": atol,
        "rtol": rtol,
        "total_differences": total,
        "examples_truncated": total > len(differences),
        "symbols": summaries,
    }
    output.mkdir(parents=True)
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(
        differences,
        columns=[
            "symbol",
            "timestamp",
            "reason",
            "field",
            "local",
            "reference",
        ],
    ).to_parquet(output / "differences.parquet", index=False)
    (output / "comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def compare_with_provider(
    local: str | pathlib.Path,
    provider: PriceProvider,
    output: str | pathlib.Path,
    frequency: str = "1d",
) -> dict:
    """Compare caller-authorized API quotes without storing credentials."""
    dataset = ds.dataset(str(local), format="parquet", partitioning="hive")
    pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="api_quotes_", dir=pathlib.Path(output).parent
    ) as temp:
        for symbol in _symbols(dataset):
            frame = _symbol_frame(dataset, symbol, frequency)
            if frame.empty:
                continue
            fetched = provider.fetch_prices(
                symbol,
                frame.index.min().date().isoformat(),
                frame.index.max().date().isoformat(),
                frequency,
            )
            required = {"timestamp", "symbol", "close"}
            if not required.issubset(fetched):
                raise ValueError(
                    "Provider did not return canonical quote fields"
                )
            if not fetched["symbol"].eq(symbol).all():
                raise ValueError("Provider returned another security")
            filename = hashlib.sha256(symbol.encode()).hexdigest() + ".parquet"
            fetched.to_parquet(pathlib.Path(temp) / filename, index=False)
        return compare_snapshots(local, temp, output, frequency)
