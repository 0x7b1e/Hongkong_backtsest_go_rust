"""Stream one long-form Parquet factor into an as-of signal matrix."""

import dataclasses
import datetime
import pathlib
import re
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from hkfactor import models

_TIMEZONE = "Asia/Hong_Kong"
_DATE_TEXT = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_BATCH_SIZE = 65_536
_EVENT_DTYPE = np.dtype(
    [
        ("timestamp", "<i8"),
        ("available_at", "<i8"),
        ("value", "<f8"),
    ]
)


@dataclasses.dataclass(frozen=True)
class SignalColumns:
    """Map vendor columns to the long-form signal contract.

    Attributes:
        timestamp: Feature observation timestamp, or a date at 16:00 HKT.
        symbol: Security identifier matching the market panel exactly.
        factor: Factor identifier; hive partition columns are supported.
        value: Numeric factor score; null explicitly clears a known score.
        available_at: First publication time, defaulting to timestamp when
            the column is absent.
    """

    timestamp: str = "timestamp"
    symbol: str = "symbol"
    factor: str = "factor"
    value: str = "value"
    available_at: str = "available_at"

    def __post_init__(self) -> None:
        """Reject missing or multiply mapped columns."""
        names = [
            self.timestamp,
            self.symbol,
            self.factor,
            self.value,
            self.available_at,
        ]
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Signal column names must be unique and nonempty")


def _normalize_times(values: pd.Series, column: str) -> np.ndarray:
    """Convert mixed vendor timestamps to UTC nanoseconds without guessing."""
    if values.isna().any():
        raise ValueError(f"Signal column {column!r} contains null timestamps")
    if pd.api.types.is_numeric_dtype(values.dtype):
        raise ValueError(
            "Numeric signal timestamps need an explicit epoch unit; "
            "convert them to typed datetimes with the signal adapter"
        )
    if pd.api.types.is_datetime64_any_dtype(values.dtype):
        date_only = np.zeros(len(values), dtype=bool)
    else:
        date_only = values.map(
            lambda item: (
                (
                    isinstance(item, datetime.date)
                    and not isinstance(item, datetime.datetime)
                )
                or (isinstance(item, str) and bool(_DATE_TEXT.fullmatch(item)))
            )
        ).to_numpy(dtype=bool)
    try:
        parsed = pd.to_datetime(values, format="mixed", errors="raise")
        if isinstance(parsed.dtype, pd.DatetimeTZDtype):
            index = pd.DatetimeIndex(parsed).tz_convert(_TIMEZONE)
        elif pd.api.types.is_datetime64_dtype(parsed.dtype):
            index = pd.DatetimeIndex(parsed)
            if date_only.any():
                adjusted = index.to_numpy(copy=True)
                adjusted[date_only] += np.timedelta64(16, "h")
                index = pd.DatetimeIndex(adjusted)
            index = index.tz_localize(_TIMEZONE)
        else:
            normalized = []
            for item, is_date in zip(values, date_only, strict=True):
                timestamp = pd.Timestamp(item)
                if is_date:
                    timestamp += pd.Timedelta(hours=16)
                if timestamp.tzinfo is None:
                    timestamp = timestamp.tz_localize(_TIMEZONE)
                normalized.append(timestamp.tz_convert(_TIMEZONE))
            index = pd.DatetimeIndex(normalized)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"Invalid signal timestamp in {column!r}") from error
    return index.tz_convert("UTC").as_unit("ns").asi8


def _is_text(dtype: pa.DataType) -> bool:
    """Return whether Arrow stores a textual identifier."""
    if pa.types.is_dictionary(dtype):
        dtype = dtype.value_type
    return pa.types.is_string(dtype) or pa.types.is_large_string(dtype)


def _align_symbol(
    events_path: pathlib.Path,
    bar_times: np.ndarray,
    values: np.ndarray,
    updated: np.ndarray,
    column: int,
    factor: str,
) -> None:
    """Read and release only one security's complete publication history."""
    events = np.fromfile(events_path, dtype=_EVENT_DTYPE)
    observed = np.sort(events["timestamp"])
    if np.any(observed[1:] == observed[:-1]):
        raise ValueError(f"Duplicate symbol/timestamp for factor {factor!r}")
    del observed
    order = np.lexsort((events["timestamp"], events["available_at"]))
    events = events[order]
    del order
    arrivals = np.searchsorted(bar_times, events["available_at"])
    updated[arrivals[arrivals < len(bar_times)]] = True
    # Late features cannot overwrite an already available newer observation.
    active = events["timestamp"] == np.maximum.accumulate(events["timestamp"])
    events = events[active]
    arrivals = arrivals[active]
    known_events = arrivals < len(bar_times)
    events = events[known_events]
    arrivals = arrivals[known_events]
    if not len(events):
        return
    # Several publications can become known between the same two bar ends.
    last_in_bar = np.r_[arrivals[1:] != arrivals[:-1], True]
    events = events[last_in_bar]
    arrivals = arrivals[last_in_bar]
    selected = (
        np.searchsorted(arrivals, np.arange(len(bar_times)), side="right") - 1
    )
    known_bars = selected >= 0
    values[known_bars, column] = events["value"][selected[known_bars]]


class SignalStore:
    """Read single-file or hive-partitioned Parquet without a factor cube.

    One factor is scanned once into temporary per-security binary files.
    Only one security's history is sorted at a time, alongside the resulting
    matrix. Historical observations before the panel are retained for warm
    starts. Late publications cannot overwrite a newer feature observation.
    """

    def __init__(
        self,
        path: str | pathlib.Path,
        columns: SignalColumns | None = None,
    ) -> None:
        """Inspect the input schema without loading its scores."""
        self.path = pathlib.Path(path)
        self.columns = columns or SignalColumns()
        self._dataset = ds.dataset(
            self.path,
            format="parquet",
            partitioning="hive",
            exclude_invalid_files=True,
        )
        required = (
            self.columns.timestamp,
            self.columns.symbol,
            self.columns.factor,
            self.columns.value,
        )
        missing = set(required).difference(self._dataset.schema.names)
        if missing:
            raise ValueError(f"Missing signal columns: {sorted(missing)}")
        for column in (self.columns.symbol, self.columns.factor):
            if not _is_text(self._dataset.schema.field(column).type):
                raise ValueError(f"Signal identifier {column!r} must be text")

    def factors(self) -> list[str]:
        """Discover partitions first, falling back to a factor-column scan."""
        names: set[str] = set()
        for fragment in self._dataset.get_fragments():
            keys = ds.get_partition_keys(fragment.partition_expression)
            if self.columns.factor not in keys:
                break
            factor = keys[self.columns.factor]
            if factor is None:
                raise ValueError("Factor identifiers cannot be null")
            if not factor:
                raise ValueError("Factor identifiers cannot be empty")
            names.add(factor)
        else:
            return sorted(names)
        scanner = self._dataset.scanner(
            columns=[self.columns.factor],
            batch_size=_BATCH_SIZE,
            batch_readahead=1,
            fragment_readahead=1,
            use_threads=False,
        )
        for batch in scanner.to_batches():
            column = batch.column(0)
            if column.null_count:
                raise ValueError("Factor identifiers cannot be null")
            for factor in column.unique().to_pylist():
                if not factor:
                    raise ValueError("Factor identifiers cannot be empty")
                names.add(factor)
        return sorted(names)

    def _scan_filter(
        self, factor: str, last_bar: pd.Timestamp
    ) -> ds.Expression:
        """Push down safe temporal upper bounds, retaining warm-start rows."""
        expression = ds.field(self.columns.factor) == factor
        available = self.columns.available_at
        column = (
            available
            if available in self._dataset.schema.names
            else self.columns.timestamp
        )
        dtype = self._dataset.schema.field(column).type
        if pa.types.is_timestamp(dtype):
            if dtype.tz:
                bound = last_bar.to_pydatetime()
            else:
                bound = last_bar.tz_convert(_TIMEZONE).tz_localize(None)
                bound = bound.to_pydatetime()
            expression &= ds.field(column) <= pa.scalar(bound, type=dtype)
        elif pa.types.is_date(dtype):
            expression &= ds.field(column) <= last_bar.date()
        return expression

    def read_factor(
        self,
        factor: str,
        panel: models.MarketPanel,
        scratch_dir: pathlib.Path | None = None,
    ) -> models.SignalMatrix:
        """Align publication events to bar ends without future information.

        Args:
            factor: One factor identifier from the Parquet dataset.
            panel: Shared market bars and exact accepted security symbols.
            scratch_dir: Parent for temporary binary partitions. Defaults
                to the system temporary directory, honoring TMPDIR. Files
                are removed on success and on validation failures.

        Returns:
            Forward-filled scores and bars receiving publication events.

        Raises:
            ValueError: Input duplicates, unknown symbols, invalid values,
                publication before observation, or no usable observations.
        """
        if not factor or not isinstance(factor, str):
            raise ValueError("factor must be a nonempty string")
        if not len(panel.timestamps):
            raise ValueError("Cannot align signals to an empty market panel")
        names = self.columns
        has_available = names.available_at in self._dataset.schema.names
        requested = [names.timestamp, names.symbol, names.value]
        if has_available:
            requested.append(names.available_at)
        scanner = self._dataset.scanner(
            columns=requested,
            filter=self._scan_filter(factor, panel.timestamps[-1]),
            batch_size=_BATCH_SIZE,
            batch_readahead=1,
            fragment_readahead=1,
            use_threads=False,
        )
        if scratch_dir is not None:
            scratch_dir = pathlib.Path(scratch_dir)
            scratch_dir.mkdir(parents=True, exist_ok=True)
        bar_times = panel.timestamps.tz_convert("UTC").as_unit("ns").asi8
        updated = np.zeros(panel.shape[0], dtype=bool)
        values = np.full(panel.shape, np.nan, dtype=float)
        symbol_positions = {symbol: i for i, symbol in enumerate(panel.symbols)}
        with tempfile.TemporaryDirectory(
            prefix=".signal-events-",
            dir=scratch_dir,
        ) as temporary:
            directory = pathlib.Path(temporary)
            written_symbols: set[int] = set()
            for batch in scanner.to_batches():
                frame = batch.to_pandas()
                observed = _normalize_times(
                    frame[names.timestamp], names.timestamp
                )
                available = (
                    _normalize_times(
                        frame[names.available_at], names.available_at
                    )
                    if has_available
                    else observed
                )
                if np.any(available < observed):
                    raise ValueError("available_at cannot precede timestamp")
                symbols = frame[names.symbol]
                if symbols.isna().any():
                    raise ValueError("Signal symbols cannot be null")
                codes = symbols.map(symbol_positions)
                if codes.isna().any():
                    unknown = sorted(set(symbols[codes.isna()]))
                    raise ValueError(f"Unknown signal symbols: {unknown}")
                try:
                    scores = pd.to_numeric(frame[names.value], errors="raise")
                    scores = scores.to_numpy(dtype=float, na_value=np.nan)
                except (TypeError, ValueError) as error:
                    raise ValueError("Signal values must be numeric") from error
                if np.isinf(scores).any():
                    raise ValueError("Signal values cannot be infinite")
                in_range = available <= bar_times[-1]
                if not in_range.any():
                    continue
                records = np.empty(int(in_range.sum()), dtype=_EVENT_DTYPE)
                records["timestamp"] = observed[in_range]
                records["available_at"] = available[in_range]
                records["value"] = scores[in_range]
                indices = codes.to_numpy(dtype=np.int64)[in_range]
                order = np.argsort(indices, kind="stable")
                records = records[order]
                indices = indices[order]
                starts = np.r_[0, np.flatnonzero(np.diff(indices)) + 1]
                stops = np.r_[starts[1:], len(records)]
                for start, stop in zip(starts, stops, strict=True):
                    column = int(indices[start])
                    with (directory / f"{column}.bin").open("ab") as handle:
                        records[start:stop].tofile(handle)
                    written_symbols.add(column)
            if not written_symbols:
                raise ValueError(
                    f"Factor {factor!r} has no available observations"
                )
            # The scanner and final batch must release their buffers before
            # per-security history sorting begins.
            del scanner, batch, frame, observed, available, symbols, codes
            del scores, records, indices, order, starts, stops, in_range
            for column in sorted(written_symbols):
                _align_symbol(
                    directory / f"{column}.bin",
                    bar_times,
                    values,
                    updated,
                    column,
                    factor,
                )
        return models.SignalMatrix(
            factor=factor, values=values, updated=updated
        )
