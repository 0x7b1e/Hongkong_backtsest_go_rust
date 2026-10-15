"""Import one-factor wide Parquet files into publication-aware signals."""

import hashlib
import json
import pathlib
import shutil
import tempfile
import urllib.parse

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_UNITS = {"s", "ms", "us", "ns"}
_SCHEMA = pa.schema(
    [
        ("timestamp", pa.timestamp("ns", tz="UTC")),
        ("available_at", pa.timestamp("ns", tz="UTC")),
        ("symbol", pa.string()),
        ("factor", pa.string()),
        ("value", pa.float64()),
    ]
)


def _duration(value: str, name: str, *, positive: bool) -> pd.Timedelta:
    """Validate an explicit fixed interval without interpreting a cadence."""
    try:
        duration = pd.Timedelta(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Invalid {name}: {value!r}") from error
    if pd.isna(duration) or duration < pd.Timedelta(0):
        raise ValueError(f"{name} must be nonnegative")
    if positive and duration == pd.Timedelta(0):
        raise ValueError(f"{name} must be positive")
    return duration


def _timestamp_column(
    schema: pa.Schema, explicit: str | None
) -> tuple[str, str, list]:
    """Select a physical timestamp field without guessing index layouts."""
    metadata = schema.metadata or {}
    try:
        pandas_metadata = json.loads(metadata.get(b"pandas", b"{}"))
        index_columns = pandas_metadata.get("index_columns", [])
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError("Invalid pandas schema metadata") from error
    if not isinstance(index_columns, list):
        raise ValueError("Invalid pandas index_columns metadata")
    if explicit is not None:
        if explicit not in schema.names:
            raise ValueError(f"Missing timestamp column {explicit!r}")
        return explicit, "explicit", index_columns
    if len(index_columns) == 1 and isinstance(index_columns[0], str):
        column = index_columns[0]
        if column not in schema.names:
            raise ValueError("Pandas timestamp index is not a physical column")
        return column, "pandas_index", index_columns
    if len(index_columns) > 1:
        raise ValueError("MultiIndex requires an explicit timestamp_column")
    if "timestamp" in schema.names:
        return "timestamp", "canonical_timestamp", index_columns
    raise ValueError(
        "No physical timestamp index; set timestamp_column explicitly "
        "(RangeIndex is not a timestamp)"
    )


def _decode_times(
    column: pa.Array,
    name: str,
    timestamp_unit: str | None,
    source_timezone: str,
) -> pd.DatetimeIndex:
    """Decode integer epochs or temporal values into UTC nanoseconds."""
    if column.null_count:
        raise ValueError(f"Timestamp column {name!r} contains nulls")
    dtype = column.type
    try:
        if pa.types.is_integer(dtype):
            if timestamp_unit is None:
                raise ValueError(
                    "Integer timestamp index requires timestamp_unit "
                    "(s, ms, us, or ns); units are never inferred"
                )
            result = pd.DatetimeIndex(
                pd.to_datetime(
                    column.to_numpy(zero_copy_only=False),
                    unit=timestamp_unit,
                    utc=True,
                    errors="raise",
                )
            )
        elif pa.types.is_timestamp(dtype) or pa.types.is_date(dtype):
            result = pd.DatetimeIndex(column.to_pandas())
            if result.tz is None:
                result = result.tz_localize(source_timezone)
            result = result.tz_convert("UTC")
        elif pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
            normalized = []
            for value in column.to_pylist():
                timestamp = pd.Timestamp(value)
                if timestamp.tzinfo is None:
                    timestamp = timestamp.tz_localize(source_timezone)
                normalized.append(timestamp.tz_convert("UTC"))
            result = pd.DatetimeIndex(normalized)
        else:
            raise ValueError(
                f"Timestamp column {name!r} must contain integer epochs "
                "or datetimes"
            )
        if result.isna().any():
            raise ValueError(f"Timestamp column {name!r} contains nulls")
        return result.as_unit("ns")
    except (TypeError, ValueError, OverflowError) as error:
        message = f"Invalid timestamp column {name!r}: {error}"
        raise ValueError(message) from error


def _source_files(source: pathlib.Path) -> list[tuple[str, pathlib.Path]]:
    """Resolve the documented single-factor-per-file layout."""
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        paths = sorted(source.rglob("*.parquet"))
    else:
        raise FileNotFoundError(f"Signal source does not exist: {source}")
    if not paths:
        raise ValueError(f"No Parquet source files in {source}")
    factors: set[str] = set()
    files = []
    for path in paths:
        if path.suffix.lower() != ".parquet":
            raise ValueError(f"Signal source must be Parquet: {path}")
        factor = path.parent.name if path.stem == "factor" else path.stem
        if not factor or factor in {".", ".."}:
            raise ValueError(f"Invalid factor identifier from {path}")
        if factor in factors:
            raise ValueError(f"Duplicate source files for factor {factor!r}")
        factors.add(factor)
        files.append((factor, path))
    return files


def _sha256(path: pathlib.Path) -> str:
    """Hash source bytes without loading the file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _import_file(
    path: pathlib.Path,
    factor: str,
    destination: pathlib.Path,
    *,
    timestamp_unit: str | None,
    source_timezone: str,
    timestamp_column: str | None,
    observation_shift: pd.Timedelta,
    publication_delay: pd.Timedelta,
    batch_size: int,
) -> dict:
    """Stream a wide file while retaining null events and source provenance."""
    before = path.stat()
    source_hash = _sha256(path)
    with pq.ParquetFile(path) as parquet:
        schema = parquet.schema_arrow
        if len(schema.names) != len(set(schema.names)):
            raise ValueError(f"Duplicate source column names in {path}")
        time_column, selection, index_columns = _timestamp_column(
            schema, timestamp_column
        )
        symbols = [name for name in schema.names if name != time_column]
        if not symbols or any(not name for name in symbols):
            raise ValueError("Wide signals require nonempty asset columns")
        symbol_array = np.asarray(symbols, dtype=object)
        row_batch_size = max(1, batch_size // len(symbols))
        destination.mkdir()
        output_path = destination / "part-00000.parquet"
        null_counts = dict.fromkeys(symbols, 0)
        source_null_counts = dict.fromkeys(symbols, 0)
        source_rows = 0
        first_timestamp = None
        last_timestamp = None
        last_raw_time = None
        with pq.ParquetWriter(output_path, _SCHEMA) as writer:
            for batch in parquet.iter_batches(batch_size=row_batch_size):
                raw_times = _decode_times(
                    batch.column(time_column),
                    time_column,
                    timestamp_unit,
                    source_timezone,
                )
                raw_ns = raw_times.asi8
                if (
                    np.any(raw_ns[1:] <= raw_ns[:-1])
                    or last_raw_time is not None
                    and raw_ns[0] <= last_raw_time
                ):
                    raise ValueError(
                        "Source timestamps must be unique and strictly "
                        "increasing across all batches"
                    )
                last_raw_time = int(raw_ns[-1])
                try:
                    observed = raw_times + observation_shift
                    available = observed + publication_delay
                except (ValueError, OverflowError) as error:
                    raise ValueError("Timestamp interval overflows") from error
                if first_timestamp is None:
                    first_timestamp = observed[0].isoformat()
                last_timestamp = observed[-1].isoformat()
                values = []
                for symbol in symbols:
                    column = batch.column(symbol)
                    dtype = column.type
                    numeric = (
                        pa.types.is_integer(dtype)
                        or pa.types.is_floating(dtype)
                        or pa.types.is_decimal(dtype)
                    )
                    if not numeric and column.null_count != len(column):
                        raise ValueError(
                            f"Asset column {symbol!r} must be numeric or "
                            "entirely null"
                        )
                    if numeric:
                        scores = np.asarray(
                            column.to_numpy(zero_copy_only=False),
                            dtype=np.float64,
                        )
                    else:
                        scores = np.full(len(column), np.nan)
                    if np.isinf(scores).any():
                        raise ValueError(
                            f"Asset column {symbol!r} contains infinite values"
                        )
                    null_counts[symbol] += int(np.isnan(scores).sum())
                    source_null_counts[symbol] += column.null_count
                    values.append(scores)
                flat_values = np.column_stack(values).reshape(-1)
                # A row wider than the event budget is split by asset here.
                for offset in range(0, len(flat_values), batch_size):
                    stop = min(offset + batch_size, len(flat_values))
                    positions = np.arange(offset, stop)
                    row_positions = positions // len(symbols)
                    symbol_positions = positions % len(symbols)
                    table = pa.Table.from_arrays(
                        [
                            pa.array(
                                observed.asi8[row_positions],
                                type=_SCHEMA.field("timestamp").type,
                            ),
                            pa.array(
                                available.asi8[row_positions],
                                type=_SCHEMA.field("available_at").type,
                            ),
                            pa.array(
                                symbol_array[symbol_positions],
                                type=pa.string(),
                            ),
                            pa.array([factor] * len(positions)),
                            pa.array(
                                flat_values[offset:stop],
                                type=pa.float64(),
                                from_pandas=True,
                            ),
                        ],
                        schema=_SCHEMA,
                    )
                    writer.write_table(table)
                source_rows += len(batch)
        if not source_rows:
            raise ValueError(f"Signal source has no timestamp rows: {path}")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError(f"Signal source changed during import: {path}")
    return {
        "factor": factor,
        "name": path.name,
        "path": str(path),
        "sha256": source_hash,
        "size_bytes": before.st_size,
        "output_size_bytes": output_path.stat().st_size,
        "output_file": str(output_path.relative_to(destination.parent)),
        "source_schema": [
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": field.nullable,
            }
            for field in schema
        ],
        "source_index_dtype": str(schema.field(time_column).type),
        "pandas_index_columns": index_columns,
        "timestamp_column": time_column,
        "timestamp_selection": selection,
        "timestamp_unit_used": (
            timestamp_unit
            if pa.types.is_integer(schema.field(time_column).type)
            else None
        ),
        "source_rows": source_rows,
        "output_rows": source_rows * len(symbols),
        "symbols": symbols,
        "source_null_counts": source_null_counts,
        "null_counts": null_counts,
        "null_count": sum(null_counts.values()),
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
    }


def import_wide_signals(
    source: str | pathlib.Path,
    output: str | pathlib.Path,
    *,
    timestamp_unit: str | None,
    timestamp_label: str,
    bar_duration: str | None = None,
    publication_delay: str = "0s",
    source_timezone: str = "UTC",
    timestamp_column: str | None = None,
    batch_size: int = 65_536,
) -> dict:
    """Atomically convert wide factor files into long-form Hive Parquet.

    Args:
        source: One Parquet file or a recursively searched directory. A
            factor.parquet file uses its parent directory as the factor ID;
            other files use their filename stem. Each factor must be unique.
        output: New output directory, which must not already exist.
        timestamp_unit: Explicit s/ms/us/ns for integer Unix timestamps.
            Datetime columns retain their timezone or use source_timezone.
        timestamp_label: Explicit bar_start or bar_end interpretation.
        bar_duration: Positive interval required for bar_start. Observation
            time is the bar end, computed by adding this duration.
        publication_delay: Nonnegative assumed delay after observation time.
        source_timezone: Timezone assigned to naive datetime values.
        timestamp_column: Override the physical pandas index timestamp field.
        batch_size: Maximum number of long-form events in each write batch.

    Returns:
        The strict JSON provenance manifest also saved in the output.

    Raises:
        ValueError: Ambiguous timing, invalid values, repeated factor IDs,
            unordered timestamps, or malformed source data.
        FileExistsError: The output already exists.

    All asset nulls become explicit score-clearing events. Availability times
    are a caller-specified assumption, not verified publication timestamps.
    Asset names are preserved exactly; no market-symbol mapping is inferred.
    """
    if timestamp_unit is not None and timestamp_unit not in _UNITS:
        raise ValueError("timestamp_unit must be s, ms, us, or ns")
    if timestamp_label not in {"bar_start", "bar_end"}:
        raise ValueError("timestamp_label must be bar_start or bar_end")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    try:
        pd.Timestamp("2000-01-01").tz_localize(source_timezone)
    except (TypeError, ValueError, KeyError) as error:
        message = f"Invalid source_timezone {source_timezone!r}"
        raise ValueError(message) from error
    delay = _duration(publication_delay, "publication_delay", positive=False)
    shift = pd.Timedelta(0)
    if bar_duration is not None:
        duration = _duration(bar_duration, "bar_duration", positive=True)
        if timestamp_label == "bar_start":
            shift = duration
    elif timestamp_label == "bar_start":
        raise ValueError("bar_start requires a positive bar_duration")
    source_path = pathlib.Path(source).resolve()
    requested_output = pathlib.Path(output)
    if requested_output.exists() or requested_output.is_symlink():
        message = f"Signal output already exists: {requested_output}"
        raise FileExistsError(message)
    output_path = requested_output.resolve()
    files = _source_files(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}.import-", dir=output_path.parent
        )
    )
    try:
        file_manifests = []
        for factor, path in files:
            partition = "factor=" + urllib.parse.quote(factor, safe="")
            file_manifests.append(
                _import_file(
                    path,
                    factor,
                    staging / partition,
                    timestamp_unit=timestamp_unit,
                    source_timezone=source_timezone,
                    timestamp_column=timestamp_column,
                    observation_shift=shift,
                    publication_delay=delay,
                    batch_size=batch_size,
                )
            )
        manifest = {
            "format_version": 1,
            "source": str(source_path),
            "output": str(output_path),
            "factors": [item["factor"] for item in file_manifests],
            "output_rows": sum(item["output_rows"] for item in file_manifests),
            "timestamp_decode_assumption": {
                "integer_epoch_unit": timestamp_unit,
                "integer_epoch_timezone": "UTC",
                "naive_source_timezone": source_timezone,
                "output_timezone": "UTC",
                "output_unit": "ns",
            },
            "assumed_availability_policy": {
                "verified_publication_timestamps": False,
                "timestamp_label": timestamp_label,
                "bar_duration": bar_duration,
                "observation_shift_ns": shift.value,
                "publication_delay": publication_delay,
                "publication_delay_ns": delay.value,
                "rule": "available_at = observation_bar_end + delay",
            },
            "null_policy": "preserve every null as an explicit score clear",
            "symbol_policy": "preserve exact source asset names",
            "files": file_manifests,
        }
        (staging / "import_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        if output_path.exists() or output_path.is_symlink():
            message = f"Signal output already exists: {output_path}"
            raise FileExistsError(message)
        staging.rename(output_path)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)
