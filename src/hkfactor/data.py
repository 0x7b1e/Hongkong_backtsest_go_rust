"""Stream validated market inputs into shared, memory-mapped panels."""

import hashlib
import json
import pathlib
import shutil
import uuid

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from hkfactor import models

TIMEZONE = "Asia/Hong_Kong"
_ARRAY_NAMES = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "base_weights",
    "benchmark_close",
    "lot_sizes",
    "tradable",
    "vcm_lower",
    "vcm_upper",
    "vcm_eligible",
    "cash_dividend",
    "share_multiplier",
    "session_id",
    "is_session_start",
    "is_session_end",
    "vcm_active",
    "stamp_duty_exempt",
    "benchmark_open",
)


def normalize_times(
    values: pd.Series | pd.DatetimeIndex | np.ndarray | list,
    frequency: str,
) -> pd.DatetimeIndex:
    """Interpret naive timestamps locally and date-only daily bars at 16:00.

    Args:
        values: Datetime-like values with one consistent timezone convention.
        frequency: Canonical bar frequency.

    Returns:
        Timezone-aware bar-end timestamps.
    """
    result = pd.DatetimeIndex(pd.to_datetime(values, format="mixed"))
    if result.hasnans:
        raise ValueError("Timestamps cannot contain null values")
    if result.tz is None:
        result = result.tz_localize(TIMEZONE)
    else:
        result = result.tz_convert(TIMEZONE)
    if frequency == "1d" and (result == result.normalize()).all():
        result += pd.Timedelta(hours=16)
    return result.as_unit("ns")


def _table(path: pathlib.Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={"symbol": str})
    return pd.read_parquet(path)


def _hash_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def panel_fingerprint(directory: str | pathlib.Path) -> str:
    """Identify panel metadata and changes to any shared array file."""
    directory = pathlib.Path(directory)
    metadata = directory / "metadata.json"
    digest = hashlib.sha256(metadata.read_bytes())
    for filename in sorted(directory.glob("*.npy")):
        stat = filename.stat()
        digest.update(
            f"{filename.name}:{stat.st_size}:{stat.st_mtime_ns}".encode()
        )
    return digest.hexdigest()


def save_panel(panel: models.MarketPanel, directory: pathlib.Path) -> None:
    """Write arrays without pickle and record their content checksums.

    Args:
        panel: Validated data contract to persist.
        directory: New cache directory, or the panel's existing staging path.
    """
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "metadata.json").exists():
        raise FileExistsError(f"Panel already exists: {directory}")
    for name in _ARRAY_NAMES:
        array = getattr(panel, name)
        if name == "vcm_active" and array is None:
            array = np.full(panel.shape, -1, dtype=np.int8)
        if name == "stamp_duty_exempt" and array is None:
            array = np.zeros(panel.shape, dtype=bool)
        if name == "benchmark_open" and array is None:
            array = np.full(panel.shape[0], np.nan)
        target = directory / f"{name}.npy"
        source = getattr(array, "filename", None)
        if (
            source is not None
            and pathlib.Path(source).resolve() == target.resolve()
        ):
            array.flush()
        else:
            np.save(target, array, allow_pickle=False)
    np.save(directory / "timestamps.npy", panel.timestamps.as_unit("ns").asi8)
    np.save(
        directory / "open_timestamps.npy",
        panel.open_timestamps.as_unit("ns").asi8,
    )
    metadata = {
        "version": 1,
        "symbols": panel.symbols,
        "frequency": panel.frequency,
        "price_basis": panel.price_basis,
        "shape": panel.shape,
        "timezone": TIMEZONE,
        "array_sha256": {
            filename.name: _hash_file(filename)
            for filename in sorted(directory.glob("*.npy"))
        },
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def load_panel(directory: pathlib.Path) -> models.MarketPanel:
    """Open immutable market arrays shared through the OS page cache."""
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("version") != 1:
        raise ValueError("Unsupported market-panel version")
    arrays = {
        name: np.load(
            directory / f"{name}.npy", mmap_mode="r", allow_pickle=False
        )
        for name in _ARRAY_NAMES
        if (directory / f"{name}.npy").exists()
    }
    timestamps = pd.to_datetime(
        np.load(directory / "timestamps.npy", allow_pickle=False), utc=True
    ).tz_convert(TIMEZONE)
    opens = pd.to_datetime(
        np.load(directory / "open_timestamps.npy", allow_pickle=False),
        utc=True,
    ).tz_convert(TIMEZONE)
    panel = models.MarketPanel(
        timestamps=timestamps,
        open_timestamps=opens,
        symbols=tuple(metadata["symbols"]),
        frequency=metadata["frequency"],
        price_basis=metadata["price_basis"],
        **arrays,
    )
    expected = tuple(metadata["shape"])
    for name in _ARRAY_NAMES:
        array = getattr(panel, name)
        if array is None:
            continue
        shape = expected if array.ndim == 2 else (expected[0],)
        if array.shape != shape:
            raise ValueError(f"Invalid array shape: {name}")
    return panel


def _snapshots(path: pathlib.Path) -> tuple[pd.DataFrame, tuple[str, ...]]:
    frame = _table(path)
    required = {"effective_date", "available_at", "symbol", "weight"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Constituents require {sorted(required)}")
    frame["symbol"] = frame["symbol"].astype(str)
    frame["effective_date"] = normalize_times(
        frame["effective_date"], "5m"
    ).normalize()
    frame["available_at"] = normalize_times(frame["available_at"], "5m")
    if frame.duplicated(["effective_date", "symbol"]).any():
        raise ValueError("Duplicate constituent snapshot entries")
    frame["weight"] = pd.to_numeric(frame["weight"], errors="raise")
    if not np.isfinite(frame["weight"]).all() or (frame["weight"] < 0).any():
        raise ValueError("Constituent weights must be finite and nonnegative")
    for _, group in frame.groupby("effective_date"):
        if not np.isclose(group["weight"].sum(), 1, atol=1e-8):
            raise ValueError("Each FULL constituent snapshot must sum to 1")
        if group["available_at"].nunique() != 1:
            raise ValueError("Each snapshot must have one publication time")
    return frame, tuple(sorted(frame["symbol"].unique()))


def _calendar(
    path: pathlib.Path | None,
    dates: pd.DatetimeIndex,
    t_start: str | None = None,
    t_end: str | None = None,
) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(
            {
                "date": dates,
                "open": "09:30",
                "close": "16:00",
                "break_start": "12:00",
                "break_end": "13:00",
            }
        )
    result = _table(path)
    if not {"date", "open", "close"}.issubset(result):
        raise ValueError("Sessions require date, open, close")
    result["date"] = normalize_times(result["date"], "5m").normalize()
    if result["date"].duplicated().any():
        raise ValueError("Duplicate trading-calendar dates")
    for column, default in [("break_start", "12:00"), ("break_end", "13:00")]:
        if column not in result:
            result[column] = default
    if not set(dates).issubset(set(result["date"])):
        raise ValueError("Trading calendar does not cover quoted dates")
    start = min(
        dates.min(),
        pd.Timestamp(t_start, tz=TIMEZONE).normalize()
        if t_start
        else dates.min(),
    )
    end = max(
        dates.max(),
        pd.Timestamp(t_end, tz=TIMEZONE).normalize() if t_end else dates.max(),
    )
    return result[result["date"].between(start, end)].sort_values("date")


def _bar_grid(
    calendar: pd.DataFrame, frequency: str
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    ends, opens = [], []
    duration = pd.Timedelta(minutes=1 if frequency == "1m" else 5)
    for row in calendar.itertuples(index=False):
        day = row.date
        start = day + pd.Timedelta(row.open + ":00")
        close = day + pd.Timedelta(row.close + ":00")
        pause = day + pd.Timedelta(row.break_start + ":00")
        resume = day + pd.Timedelta(row.break_end + ":00")
        if start >= close or pause >= resume:
            raise ValueError("Invalid trading-calendar session")
        if frequency == "1d":
            ends.append(day + pd.Timedelta(hours=16))
            opens.append(start)
            continue
        for left, right in [(start, min(pause, close)), (resume, close)]:
            if left >= right:
                continue
            bars = pd.date_range(left + duration, right, freq=duration)
            ends.extend(bars)
            opens.extend(bars - duration)
    if not ends:
        raise ValueError("No price bars in requested range")
    return (
        pd.DatetimeIndex(ends).as_unit("ns"),
        pd.DatetimeIndex(opens).as_unit("ns"),
    )


def _base_weights(
    snapshots: pd.DataFrame,
    opens: pd.DatetimeIndex,
    symbols: tuple[str, ...],
    destination: np.ndarray,
) -> None:
    positions = {symbol: i for i, symbol in enumerate(symbols)}
    groups = []
    for effective, group in snapshots.groupby("effective_date", sort=True):
        available = group["available_at"].iloc[0]
        known = max(effective, available).value
        weights = np.zeros(len(symbols))
        for row in group.itertuples():
            weights[positions[row.symbol]] = row.weight
        groups.append((effective.value, known, weights))
    for i, timestamp in enumerate(opens.asi8):
        eligible = [group for group in groups if group[1] <= timestamp]
        if not eligible:
            raise ValueError("No published constituent snapshot before open")
        destination[i] = max(eligible, key=lambda group: group[0])[2]


def prepare_market(
    prices: str | pathlib.Path,
    benchmark: str | pathlib.Path,
    constituents: str | pathlib.Path,
    output: str | pathlib.Path,
    frequency: str = "1d",
    sessions: str | pathlib.Path | None = None,
    t_start: str | None = None,
    t_end: str | None = None,
    max_panel_bytes: int = 4 * 1024**3,
) -> models.MarketPanel:
    """Prepare market arrays with bounded input-batch memory.

    Args:
        prices: Canonical long-form stock-price Parquet dataset.
        benchmark: Canonical timestamp/close benchmark Parquet or CSV.
        constituents: Full point-in-time weight snapshots, never a current
            constituent list retrospectively applied to history.
        output: New panel-cache directory; existing caches are not replaced.
        frequency: Price-bar frequency.
        sessions: Official historical session schedule; required intraday.
        t_start: Optional research start; retains 14 calendar days of warmup.
        t_end: Optional last included research date.
        max_panel_bytes: Reject panels exceeding this on-disk array budget.

    Returns:
        A read-only, memory-mapped panel.
    """
    if frequency not in {"1d", "1m", "5m"}:
        raise ValueError("Unsupported frequency")
    if frequency != "1d" and sessions is None:
        raise ValueError("Intraday data require a historical session calendar")
    output = pathlib.Path(output)
    if output.exists():
        raise FileExistsError(f"Cache already exists: {output}")
    snapshots, symbols = _snapshots(pathlib.Path(constituents))
    dataset = ds.dataset(str(prices), format="parquet", partitioning="hive")
    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    if not required.issubset(dataset.schema.names):
        raise ValueError(f"Stock prices require {sorted(required)}")
    expression = ds.field("symbol").isin(symbols)
    lower = (
        pd.Timestamp(t_start, tz=TIMEZONE).normalize() - pd.Timedelta(days=14)
        if t_start
        else None
    )
    upper = (
        pd.Timestamp(t_end, tz=TIMEZONE).normalize() + pd.Timedelta(days=1)
        if t_end
        else None
    )
    quoted_days = set()
    for batch in dataset.scanner(
        columns=["timestamp"], filter=expression, batch_size=65_536
    ).to_batches():
        times = normalize_times(batch.column(0).to_pandas(), frequency)
        mask = np.ones(len(times), dtype=bool)
        if lower is not None:
            mask &= times >= lower
        if upper is not None:
            mask &= times < upper
        quoted_days.update(times[mask].normalize())
    if not quoted_days:
        raise ValueError("No constituent prices in requested date range")
    dates = pd.DatetimeIndex(sorted(quoted_days))
    calendar = _calendar(
        pathlib.Path(sessions) if sessions else None, dates, t_start, t_end
    )
    timestamps, open_timestamps = _bar_grid(calendar, frequency)
    shape = len(timestamps), len(symbols)
    estimate = shape[0] * shape[1] * (12 * 8 + 2) + shape[0] * 32
    if estimate > max_panel_bytes:
        raise MemoryError(
            f"Panel needs approximately {estimate:,} bytes; "
            f"budget is {max_panel_bytes:,}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    staging.mkdir()
    try:
        arrays = {}
        matrix_defaults = {
            "open": np.nan,
            "high": np.nan,
            "low": np.nan,
            "close": np.nan,
            "volume": 0,
            "base_weights": 0,
            "lot_sizes": np.nan,
            "vcm_lower": np.nan,
            "vcm_upper": np.nan,
            "cash_dividend": 0,
            "share_multiplier": 1,
        }
        for name, default in matrix_defaults.items():
            array = np.lib.format.open_memmap(
                staging / f"{name}.npy",
                mode="w+",
                dtype="float64",
                shape=shape,
            )
            array[:] = default
            arrays[name] = array
        for name in ["tradable", "vcm_eligible", "stamp_duty_exempt"]:
            array = np.lib.format.open_memmap(
                staging / f"{name}.npy",
                mode="w+",
                dtype="bool",
                shape=shape,
            )
            array[:] = False
            arrays[name] = array
        arrays["vcm_active"] = np.lib.format.open_memmap(
            staging / "vcm_active.npy",
            mode="w+",
            dtype="int8",
            shape=shape,
        )
        arrays["vcm_active"][:] = -1
        seen = np.lib.format.open_memmap(
            staging / "seen.npy", mode="w+", dtype="bool", shape=shape
        )
        seen[:] = False
        symbol_index = {symbol: i for i, symbol in enumerate(symbols)}
        columns = list(required)
        optional = {
            "lot_size",
            "tradable",
            "halted",
            "vcm_lower",
            "vcm_upper",
            "vcm_eligible",
            "cash_dividend",
            "share_multiplier",
            "price_basis",
            "vcm_active",
            "stamp_duty_exempt",
            "security_type",
        }
        columns += sorted(optional.intersection(dataset.schema.names))
        price_bases = set()
        for batch in dataset.scanner(
            columns=columns, filter=expression, batch_size=65_536
        ).to_batches():
            frame = batch.to_pandas()
            times = normalize_times(frame["timestamp"], frequency)
            mask = np.ones(len(times), dtype=bool)
            if lower is not None:
                mask &= times >= lower
            if upper is not None:
                mask &= times < upper
            frame = frame.loc[mask].copy()
            times = times[mask]
            if frame.empty:
                continue
            for column in [
                "tradable",
                "halted",
                "vcm_eligible",
                "vcm_active",
                "stamp_duty_exempt",
            ]:
                if column in frame and (
                    frame[column].isna().any()
                    or not pd.api.types.is_bool_dtype(frame[column])
                ):
                    raise ValueError(f"{column} must be non-null booleans")
            row_index = timestamps.get_indexer(times)
            if (row_index < 0).any():
                raise ValueError("Prices are off-grid or outside CTS hours")
            col_index = frame["symbol"].map(symbol_index).to_numpy(int)
            key = row_index * shape[1] + col_index
            if (
                len(np.unique(key)) != len(key)
                or seen[row_index, col_index].any()
            ):
                raise ValueError("Duplicate timestamp/symbol price rows")
            seen[row_index, col_index] = True
            values = frame[["open", "high", "low", "close", "volume"]]
            if not np.isfinite(values.to_numpy(dtype=float)).all():
                raise ValueError("Existing price rows must be finite")
            if (frame[["open", "high", "low", "close"]] <= 0).any().any():
                raise ValueError("Prices must be positive")
            if (frame["volume"] < 0).any():
                raise ValueError("Volume cannot be negative")
            if (
                (frame["high"] < frame[["open", "close"]].max(axis=1))
                | (frame["low"] > frame[["open", "close"]].min(axis=1))
                | (frame["low"] > frame["high"])
            ).any():
                raise ValueError("Invalid OHLC price envelope")
            for name in ["open", "high", "low", "close", "volume"]:
                arrays[name][row_index, col_index] = frame[name]
            for source, target in [
                ("lot_size", "lot_sizes"),
                ("vcm_lower", "vcm_lower"),
                ("vcm_upper", "vcm_upper"),
                ("cash_dividend", "cash_dividend"),
                ("share_multiplier", "share_multiplier"),
            ]:
                if source in frame:
                    arrays[target][row_index, col_index] = pd.to_numeric(
                        frame[source], errors="raise"
                    )
            if (
                "lot_size" in frame
                and (
                    ~np.isfinite(frame["lot_size"])
                    | (frame["lot_size"] <= 0)
                    | (frame["lot_size"] % 1 != 0)
                ).any()
            ):
                raise ValueError("lot_size must be a positive integer")
            tradable = frame["volume"].to_numpy() > 0
            if "tradable" in frame:
                if frame["tradable"].isna().any():
                    raise ValueError("tradable cannot be null")
                tradable &= frame["tradable"].to_numpy(dtype=bool)
            if "halted" in frame:
                if frame["halted"].isna().any():
                    raise ValueError("halted cannot be null")
                tradable &= ~frame["halted"].to_numpy(dtype=bool)
            arrays["tradable"][row_index, col_index] = tradable
            if "vcm_eligible" in frame:
                if frame["vcm_eligible"].isna().any():
                    raise ValueError("vcm_eligible cannot be null")
                arrays["vcm_eligible"][row_index, col_index] = frame[
                    "vcm_eligible"
                ].to_numpy(dtype=bool)
            if "vcm_active" in frame:
                arrays["vcm_active"][row_index, col_index] = frame[
                    "vcm_active"
                ].to_numpy(dtype=np.int8)
            if "security_type" in frame:
                types = frame["security_type"]
                if (
                    types.isna().any()
                    or not types.isin(["stock", "reit", "etf"]).all()
                ):
                    raise ValueError("security_type must be stock, reit, etf")
                exemption = (
                    types.eq("reit").to_numpy()
                    & (times >= pd.Timestamp("2024-12-21", tz=TIMEZONE))
                ) | (
                    types.eq("etf").to_numpy()
                    & (times >= pd.Timestamp("2015-02-13", tz=TIMEZONE))
                )
                arrays["stamp_duty_exempt"][row_index, col_index] = exemption
            if "stamp_duty_exempt" in frame:
                arrays["stamp_duty_exempt"][row_index, col_index] = frame[
                    "stamp_duty_exempt"
                ].to_numpy(dtype=bool)
            price_bases.update(
                frame["price_basis"].astype(str).unique()
                if "price_basis" in frame
                else ["raw"]
            )
        if len(price_bases) != 1 or not price_bases <= {"raw", "adjusted"}:
            raise ValueError("Specify one price_basis: raw or adjusted")
        _base_weights(
            snapshots, open_timestamps, symbols, arrays["base_weights"]
        )
        if not (seen & (arrays["base_weights"] > 0)).any(axis=1).all():
            raise ValueError("Missing all constituent quotes in a session/bar")
        for j in range(shape[1]):
            arrays["close"][:, j] = (
                pd.Series(arrays["close"][:, j]).ffill().to_numpy()
            )
            missing = ~np.isfinite(arrays["open"][:, j])
            arrays["open"][missing, j] = arrays["close"][missing, j]
            arrays["lot_sizes"][:, j] = (
                pd.Series(arrays["lot_sizes"][:, j]).ffill().to_numpy()
            )
        if np.any((arrays["base_weights"] > 0) & ~np.isfinite(arrays["close"])):
            raise ValueError("Constituent has no past quote for valuation")
        if (
            np.any(arrays["share_multiplier"] <= 0)
            or not np.isfinite(arrays["share_multiplier"]).all()
        ):
            raise ValueError("Share multipliers must be finite and positive")
        if not np.isfinite(arrays["cash_dividend"]).all() or np.any(
            arrays["cash_dividend"] < 0
        ):
            raise ValueError("Cash dividends must be finite and nonnegative")
        benchmark_frame = _table(pathlib.Path(benchmark))
        if not {"timestamp", "close"}.issubset(benchmark_frame):
            raise ValueError("Benchmark requires timestamp and close")
        benchmark_frame.index = normalize_times(
            benchmark_frame["timestamp"], frequency
        )
        if benchmark_frame.index.duplicated().any():
            raise ValueError("Duplicate benchmark timestamps")
        arrays["benchmark_close"] = (
            benchmark_frame["close"].reindex(timestamps).to_numpy(dtype=float)
        )
        arrays["benchmark_open"] = (
            benchmark_frame["open"].reindex(timestamps).to_numpy(dtype=float)
            if "open" in benchmark_frame
            else None
        )
        if arrays["benchmark_open"] is not None and (
            not np.isfinite(arrays["benchmark_open"]).all()
            or np.any(arrays["benchmark_open"] <= 0)
        ):
            raise ValueError("Supplied benchmark open must cover all bars")
        if not np.isfinite(arrays["benchmark_close"]).all() or np.any(
            arrays["benchmark_close"] <= 0
        ):
            raise ValueError("Benchmark must cover every panel bar exactly")
        dates = timestamps.normalize()
        arrays["session_id"] = pd.factorize(dates, sort=False)[0]
        arrays["is_session_start"] = np.r_[True, dates[1:] != dates[:-1]]
        arrays["is_session_end"] = np.r_[dates[:-1] != dates[1:], True]
        panel = models.MarketPanel(
            timestamps=timestamps,
            open_timestamps=open_timestamps,
            symbols=symbols,
            frequency=frequency,
            price_basis=next(iter(price_bases)),
            **arrays,
        )
        del seen
        (staging / "seen.npy").unlink()
        save_panel(panel, staging)
        del panel, arrays
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return load_panel(output)
