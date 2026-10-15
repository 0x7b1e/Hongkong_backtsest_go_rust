"""Read source CSV bars without changing labels, values, or source bytes."""

import pathlib
import re
from collections.abc import Iterator

import numpy as np
import pandas as pd

from hkfactor import data as market_data

_COLUMNS = {
    "代码": "symbol",
    "名称": "name",
    "日期": "date",
    "时间": "time",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
    "昨收": "previous_close",
}
_PRICE_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]
_INDEX_COLUMNS = ["timestamp", "symbol", *_PRICE_COLUMNS]


def csv_options(source: str | pathlib.Path) -> tuple[str, int]:
    """Detect UTF-8/GB18030 and an optional original HSI description row."""
    with pathlib.Path(source).open("rb") as handle:
        sample = handle.readline() + handle.readline()
    if not sample.strip():
        raise ValueError("CSV has no header")
    for encoding in ["utf-8-sig", "gb18030"]:
        try:
            lines = sample.decode(encoding).splitlines()
            return encoding, int(lines[0].startswith("HSI "))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unsupported CSV encoding: {source}")


def iter_csv_frames(
    source: str | pathlib.Path,
    chunksize: int = 65_536,
) -> Iterator[pd.DataFrame]:
    """Read shared CSV syntax and map Chinese names to canonical fields.

    Args:
        source: CSV with Chinese or canonical column names.
        chunksize: Positive maximum rows per input batch.

    Yields:
        Frames retaining source values and leading-zero security codes.
    """
    if type(chunksize) is not int or chunksize < 1:
        raise ValueError("chunksize must be a positive integer")
    encoding, skiprows = csv_options(source)
    for frame in pd.read_csv(
        source,
        encoding=encoding,
        skiprows=skiprows,
        dtype={
            "代码": str,
            "日期": str,
            "时间": str,
            "symbol": str,
            "code": str,
            "date": str,
            "time": str,
        },
        chunksize=chunksize,
    ):
        unnamed = frame.columns.str.startswith("Unnamed:")
        if frame.loc[:, unnamed].notna().any().any():
            raise ValueError("Unexpected values in unnamed CSV columns")
        yield frame.loc[:, ~unnamed].rename(columns=_COLUMNS)


def csv_timestamps(
    frame: pd.DataFrame,
    frequency: str,
) -> pd.DatetimeIndex:
    """Combine YYYYMMDD dates and HHMM clocks into Hong Kong bar-end labels.

    Args:
        frame: Source fields containing timestamp, or date and minute time.
        frequency: 1d, 1m, or 5m; this does not resample observations.

    Returns:
        Nanosecond timestamps localized to Asia/Hong_Kong.
    """
    if frequency not in {"1d", "1m", "5m"}:
        raise ValueError("frequency must be 1d, 1m, or 5m")
    if "timestamp" in frame:
        return market_data.normalize_times(frame["timestamp"], frequency)
    if "date" not in frame:
        raise ValueError("CSV requires a date or timestamp column")
    dates = pd.to_datetime(frame["date"], format="mixed", errors="raise")
    if frequency == "1d":
        return market_data.normalize_times(dates, frequency)
    if "time" not in frame:
        raise ValueError("Minute CSV requires a time column")
    clock = frame["time"].astype("string").str.strip()
    compact = clock.str.fullmatch(r"\d{1,4}", na=False)
    colon = clock.str.fullmatch(r"\d{1,2}:\d{2}", na=False)
    if not (compact | colon).all():
        raise ValueError("Invalid HHMM time value")
    clock = clock.str.replace(":", "", regex=False).str.zfill(4)
    hours = pd.to_numeric(clock.str[:2])
    minutes = pd.to_numeric(clock.str[2:])
    if (hours > 23).any() or (minutes > 59).any():
        raise ValueError("Invalid HHMM time value")
    if dates.isna().any():
        raise ValueError("Dates cannot contain null values")
    times = dates.dt.normalize() + pd.to_timedelta(
        hours * 60 + minutes, unit="min"
    )
    return market_data.normalize_times(times, frequency)


def _bound(value: str | None, end: bool = False) -> pd.Timestamp | None:
    if value is None:
        return None
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError("Date bounds cannot be null")
    result = (
        result.tz_localize(market_data.TIMEZONE)
        if result.tzinfo is None
        else result.tz_convert(market_data.TIMEZONE)
    )
    if end and re.fullmatch(r"\d{4}-\d{2}-\d{2}|\d{8}", value):
        result += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return result


def _empty_index() -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="datetime64[ns, Asia/Hong_Kong]"),
            "symbol": pd.Series(dtype="str"),
            **{column: pd.Series(dtype="float64") for column in _PRICE_COLUMNS},
        }
    )
    return result[_INDEX_COLUMNS]


def iter_index_csv(
    source: str | pathlib.Path,
    frequency: str = "1m",
    *,
    symbol: str = "HSI",
    t_start: str | None = None,
    t_end: str | None = None,
    chunksize: int = 65_536,
) -> Iterator[pd.DataFrame]:
    """Stream complete index bars and validate causal source ordering.

    Source minute timestamps label bar ends. Missing bars and partial
    sessions remain missing. No resampling, price adjustment, or stock
    volume conversion is performed. Negative source amounts are retained
    for reconciliation and counted in each frame's attrs.

    Args:
        source: Original index CSV with optional title and empty tail columns.
        frequency: 1m, 5m, or 1d labels supplied by the caller.
        symbol: Index identifier; defaults to HSI.
        t_start: Inclusive local datetime or date bound.
        t_end: Inclusive datetime, or all bars of a date-only bound.
        chunksize: Maximum input rows processed at once.

    Yields:
        timestamp, symbol, open, high, low, close, volume and amount frames.

    Raises:
        ValueError: Missing fields, invalid numeric values or source ordering.
    """
    if not symbol:
        raise ValueError("symbol must be nonempty")
    start, end = _bound(t_start), _bound(t_end, end=True)
    if start is not None and end is not None and start > end:
        raise ValueError("t_start must be on or before t_end")
    last = None
    for frame in iter_csv_frames(source, chunksize):
        missing = set(_PRICE_COLUMNS).difference(frame.columns)
        if missing:
            raise ValueError(f"Index CSV missing fields: {sorted(missing)}")
        times = csv_timestamps(frame, frequency)
        if times.duplicated().any():
            raise ValueError("Duplicate index timestamp")
        if not times.is_monotonic_increasing:
            raise ValueError("Index timestamps must be chronological")
        if len(times) and last is not None and times[0] <= last:
            raise ValueError("Duplicate or out-of-order index timestamp")
        if len(times):
            last = times[-1]
        frame = frame.copy()
        for column in _PRICE_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(
                "float64"
            )
        if not np.isfinite(frame[_PRICE_COLUMNS].to_numpy()).all():
            raise ValueError("Index numeric values must be finite")
        if (frame[["open", "high", "low", "close"]] <= 0).any().any():
            raise ValueError("Index prices must be positive")
        if (frame["volume"] < 0).any():
            raise ValueError("Index volume cannot be negative")
        if (
            (frame["high"] < frame[["open", "close"]].max(axis=1))
            | (frame["low"] > frame[["open", "close"]].min(axis=1))
            | (frame["low"] > frame["high"])
        ).any():
            raise ValueError("Invalid index OHLC envelope")
        if "symbol" in frame and not frame["symbol"].eq(symbol).all():
            raise ValueError("CSV contains an unexpected index symbol")
        frame["timestamp"] = times
        frame["symbol"] = symbol
        mask = np.ones(len(frame), dtype=bool)
        if start is not None:
            mask &= times >= start
        if end is not None:
            mask &= times <= end
        result = frame.loc[mask, _INDEX_COLUMNS].reset_index(drop=True)
        if result.empty:
            continue
        result.attrs = {
            "source": str(source),
            "frequency": frequency,
            "negative_amount_count": int((result["amount"] < 0).sum()),
            "timestamp_label": "bar_end",
            "volume_unit": "source_unchanged",
        }
        yield result


def read_index_csv(
    source: str | pathlib.Path,
    frequency: str = "1m",
    *,
    symbol: str = "HSI",
    t_start: str | None = None,
    t_end: str | None = None,
    chunksize: int = 65_536,
) -> pd.DataFrame:
    """Read an index date slice in memory using the streaming parser.

    Args:
        source: Original index CSV.
        frequency: 1m, 5m, or 1d labels without resampling.
        symbol: Index identifier.
        t_start: Inclusive local datetime or date.
        t_end: Inclusive datetime or complete end date.
        chunksize: Input batch size; use iter_index_csv for bounded output RAM.

    Returns:
        Validated full OHLCVA bars; an empty range has the same typed schema.
    """
    frames = list(
        iter_index_csv(
            source,
            frequency,
            symbol=symbol,
            t_start=t_start,
            t_end=t_end,
            chunksize=chunksize,
        )
    )
    count = sum(frame.attrs["negative_amount_count"] for frame in frames)
    result = pd.concat(frames, ignore_index=True) if frames else _empty_index()
    result.attrs = {
        "source": str(source),
        "frequency": frequency,
        "negative_amount_count": count,
        "timestamp_label": "bar_end",
        "volume_unit": "source_unchanged",
    }
    return result
