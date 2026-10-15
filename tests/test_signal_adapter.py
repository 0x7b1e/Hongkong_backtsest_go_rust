"""Wide factor conversion, explicit timing, and streaming regressions."""

import datetime
import hashlib
import json
import pathlib
import types

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from hkfactor import signal_adapter
from hkfactor import signals


def _wide(
    path: pathlib.Path,
    index,
    values: dict | None = None,
) -> pathlib.Path:
    """Write physical pandas index metadata matching the reference layout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = values or {"BTCUSDT": [1.0, 2.0, 3.0]}
    pd.DataFrame(data, index=index).to_parquet(path)
    return path


def _read(path: pathlib.Path) -> pd.DataFrame:
    """Read converted fragments while excluding the JSON manifest."""
    return (
        ds.dataset(
            path,
            format="parquet",
            partitioning="hive",
            exclude_invalid_files=True,
        )
        .to_table()
        .to_pandas()
    )


def test_wide_values_null_lifecycle_and_manifest(
    tmp_path: pathlib.Path,
) -> None:
    """Null score events clear old values after streaming conversion."""
    index = (
        pd.to_datetime(
            ["2024-01-02 01:35", "2024-01-02 01:40", "2024-01-02 01:45"],
            utc=True,
        )
        .as_unit("ms")
        .asi8
    )
    source = _wide(
        tmp_path / "alpha101_factor" / "alpha001" / "factor.parquet",
        index,
        {"BTCUSDT": [1.0, np.nan, 3.0], "币安人生USDT": [4.0, 5.0, 6.0]},
    )
    original = source.read_bytes()
    output = tmp_path / "converted"
    manifest = signal_adapter.import_wide_signals(
        source,
        output,
        timestamp_unit="ms",
        timestamp_label="bar_end",
        batch_size=2,
    )
    frame = _read(output)
    assert list(frame.columns) == [
        "timestamp",
        "available_at",
        "symbol",
        "factor",
        "value",
    ]
    assert len(frame) == 6
    assert frame["factor"].unique().tolist() == ["alpha001"]
    assert frame["symbol"].unique().tolist() == ["BTCUSDT", "币安人生USDT"]
    assert frame["timestamp"].dtype == pd.DatetimeTZDtype("ns", "UTC")
    assert frame["timestamp"].equals(frame["available_at"])
    np.testing.assert_allclose(
        frame["value"], [1, 4, np.nan, 5, 3, 6], equal_nan=True
    )
    assert source.read_bytes() == original
    assert manifest == json.loads(
        (output / "import_manifest.json").read_text(encoding="utf-8")
    )
    entry = manifest["files"][0]
    assert entry["sha256"] == hashlib.sha256(original).hexdigest()
    assert entry["source_index_dtype"] == "int64"
    assert entry["source_rows"] == 3
    assert entry["output_rows"] == 6
    assert entry["null_count"] == 1
    assert entry["null_counts"]["BTCUSDT"] == 1
    assert entry["timestamp_unit_used"] == "ms"
    assert (
        manifest["assumed_availability_policy"][
            "verified_publication_timestamps"
        ]
        is False
    )
    store = signals.SignalStore(output)
    panel_times = pd.date_range(
        "2024-01-02 01:35", periods=7, freq="150s", tz="UTC"
    )
    panel = types.SimpleNamespace(
        timestamps=panel_times,
        symbols=("BTCUSDT", "币安人生USDT"),
        shape=(7, 2),
    )
    aligned = store.read_factor("alpha001", panel)
    np.testing.assert_allclose(
        aligned.values[:, 0],
        [1, 1, np.nan, np.nan, 3, 3, 3],
        equal_nan=True,
    )
    assert list(tmp_path.glob(".converted.import-*")) == []


def test_integer_timestamp_requires_explicit_unit(
    tmp_path: pathlib.Path,
) -> None:
    """The importer never mistakes epoch milliseconds for nanoseconds."""
    source = _wide(tmp_path / "alpha007.parquet", [1, 2, 3])
    with pytest.raises(ValueError, match="requires timestamp_unit"):
        signal_adapter.import_wide_signals(
            source,
            tmp_path / "out",
            timestamp_unit=None,
            timestamp_label="bar_end",
        )
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out.import-*"))


def test_bar_start_shift_and_publication_delay(tmp_path: pathlib.Path) -> None:
    """Start-labeled observations are available only after end plus delay."""
    index = pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC")
    source = _wide(tmp_path / "alpha101.parquet", index)
    output = tmp_path / "out"
    manifest = signal_adapter.import_wide_signals(
        source,
        output,
        timestamp_unit=None,
        timestamp_label="bar_start",
        bar_duration="15min",
        publication_delay="30s",
    )
    frame = _read(output)
    pd.testing.assert_index_equal(
        pd.DatetimeIndex(frame["timestamp"]),
        (index + pd.Timedelta(minutes=15)).as_unit("ns").rename("timestamp"),
    )
    assert (
        frame["available_at"] - frame["timestamp"] == pd.Timedelta(seconds=30)
    ).all()
    assert manifest["factors"] == ["alpha101"]
    assert (
        manifest["assumed_availability_policy"]["observation_shift_ns"]
        == pd.Timedelta(minutes=15).value
    )


@pytest.mark.parametrize(
    "index,timezone,expected",
    [
        (
            pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
            "Asia/Hong_Kong",
            "2024-01-01 00:00Z",
        ),
        (
            pd.date_range("2024-01-01", periods=3, freq="h"),
            "Asia/Hong_Kong",
            "2023-12-31 16:00Z",
        ),
        (
            [datetime.date(2024, 1, day) for day in (1, 2, 3)],
            "UTC",
            "2024-01-01 00:00Z",
        ),
    ],
)
def test_datetime_timezone_and_dates(
    tmp_path: pathlib.Path, index, timezone: str, expected: str
) -> None:
    """Aware timezones persist and naive date times use the explicit zone."""
    source = _wide(tmp_path / "alpha.parquet", index)
    output = tmp_path / "out"
    signal_adapter.import_wide_signals(
        source,
        output,
        timestamp_unit=None,
        timestamp_label="bar_end",
        source_timezone=timezone,
    )
    assert _read(output)["timestamp"].iloc[0] == pd.Timestamp(expected)


@pytest.mark.parametrize(
    "arguments,match",
    [
        ({"timestamp_label": "unknown"}, "timestamp_label"),
        ({"timestamp_label": "bar_start"}, "requires.*bar_duration"),
        (
            {"timestamp_label": "bar_start", "bar_duration": "0s"},
            "positive",
        ),
        ({"publication_delay": "-1s"}, "nonnegative"),
        ({"timestamp_unit": "minutes"}, "timestamp_unit"),
        ({"batch_size": 0}, "batch_size"),
    ],
)
def test_explicit_policy_validation(
    tmp_path: pathlib.Path, arguments: dict, match: str
) -> None:
    """Unknown labels and invalid intervals fail before writing output."""
    source = _wide(tmp_path / "alpha.parquet", [1, 2, 3])
    options = {"timestamp_unit": "s", "timestamp_label": "bar_end"}
    options.update(arguments)
    with pytest.raises(ValueError, match=match):
        signal_adapter.import_wide_signals(source, tmp_path / "out", **options)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("index", [[1, 1, 2], [1, 3, 2]])
def test_duplicate_or_out_of_order_times_cleanup(
    tmp_path: pathlib.Path, index: list[int]
) -> None:
    """A late failing chunk removes the complete staged dataset."""
    source = _wide(tmp_path / "alpha.parquet", index)
    with pytest.raises(ValueError, match="unique and strictly increasing"):
        signal_adapter.import_wide_signals(
            source,
            tmp_path / "out",
            timestamp_unit="s",
            timestamp_label="bar_end",
            batch_size=1,
        )
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out.import-*"))


@pytest.mark.parametrize(
    "values,match",
    [
        ({"BTCUSDT": [1.0, 2.0, np.inf]}, "infinite"),
        ({"BTCUSDT": ["1", "2", "3"]}, "numeric or entirely null"),
    ],
)
def test_invalid_scores_cleanup(
    tmp_path: pathlib.Path, values: dict, match: str
) -> None:
    """Invalid asset values are rejected without retaining partial output."""
    source = _wide(tmp_path / "alpha.parquet", [1, 2, 3], values)
    original = source.read_bytes()
    with pytest.raises(ValueError, match=match):
        signal_adapter.import_wide_signals(
            source,
            tmp_path / "out",
            timestamp_unit="s",
            timestamp_label="bar_end",
            batch_size=1,
        )
    assert source.read_bytes() == original
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out.import-*"))


def test_recursive_files_and_duplicate_factor_rejection(
    tmp_path: pathlib.Path,
) -> None:
    """Independent factors stream separately and repeated IDs never merge."""
    source = tmp_path / "source"
    _wide(source / "alpha001" / "factor.parquet", [1, 2, 3])
    _wide(source / "alpha002" / "factor.parquet", [1, 2, 3])
    output = tmp_path / "out"
    manifest = signal_adapter.import_wide_signals(
        source,
        output,
        timestamp_unit="s",
        timestamp_label="bar_end",
    )
    assert manifest["factors"] == ["alpha001", "alpha002"]
    assert signals.SignalStore(output).factors() == ["alpha001", "alpha002"]
    _wide(source / "copy" / "alpha001" / "factor.parquet", [1, 2, 3])
    with pytest.raises(ValueError, match="Duplicate source files"):
        signal_adapter.import_wide_signals(
            source,
            tmp_path / "duplicate",
            timestamp_unit="s",
            timestamp_label="bar_end",
        )
    assert not (tmp_path / "duplicate").exists()


def test_range_and_multiindex_require_explicit_column(
    tmp_path: pathlib.Path,
) -> None:
    """Nonphysical RangeIndex and MultiIndex metadata are never guessed."""
    range_source = _wide(tmp_path / "range.parquet", pd.RangeIndex(3))
    multi_source = _wide(
        tmp_path / "multi.parquet",
        pd.MultiIndex.from_arrays([[1, 2, 3], [4, 5, 6]]),
    )
    for source in (range_source, multi_source):
        with pytest.raises(ValueError, match="RangeIndex|MultiIndex"):
            signal_adapter.import_wide_signals(
                source,
                tmp_path / "out",
                timestamp_unit="s",
                timestamp_label="bar_end",
            )


@pytest.mark.parametrize("explicit", [None, "time"])
def test_canonical_or_explicit_timestamp_column(
    tmp_path: pathlib.Path, explicit: str | None
) -> None:
    """An explicit vendor field or canonical timestamp supports flat files."""
    name = explicit or "timestamp"
    source = tmp_path / "alpha.parquet"
    pq.write_table(pa.table({name: [1, 2], "BTCUSDT": [4, 5]}), source)
    output = tmp_path / "out"
    manifest = signal_adapter.import_wide_signals(
        source,
        output,
        timestamp_unit="s",
        timestamp_label="bar_end",
        timestamp_column=explicit,
    )
    assert manifest["files"][0]["timestamp_column"] == name
    assert _read(output)["value"].tolist() == [4, 5]


def test_all_null_asset_and_event_batch_budget(tmp_path: pathlib.Path) -> None:
    """Even one very wide row splits into bounded output batches."""
    source = tmp_path / "alpha.parquet"
    pq.write_table(
        pa.table(
            {
                "timestamp": [1],
                "BTCUSDT": [np.nan],
                "ETHUSDT": pa.array([None], type=pa.string()),
                "XRPUSDT": [3.0],
            }
        ),
        source,
    )
    output = tmp_path / "out"
    manifest = signal_adapter.import_wide_signals(
        source,
        output,
        timestamp_unit="s",
        timestamp_label="bar_end",
        batch_size=1,
    )
    assert manifest["files"][0]["null_count"] == 2
    parquet = pq.ParquetFile(output / "factor=alpha" / "part-00000.parquet")
    assert parquet.metadata.num_row_groups == 3
    assert all(
        parquet.metadata.row_group(number).num_rows <= 1
        for number in range(parquet.metadata.num_row_groups)
    )
    assert _read(output)["value"].isna().tolist() == [True, True, False]


def test_existing_output_is_never_overwritten(tmp_path: pathlib.Path) -> None:
    """Import keeps an existing destination intact."""
    source = _wide(tmp_path / "alpha.parquet", [1, 2, 3])
    output = tmp_path / "out"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        signal_adapter.import_wide_signals(
            source,
            output,
            timestamp_unit="s",
            timestamp_label="bar_end",
        )
    assert marker.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_all_explicit_epoch_units(tmp_path: pathlib.Path, unit: str) -> None:
    """Supported epoch units decode the same UTC instant."""
    timestamp = pd.Timestamp("2024-01-02 01:35Z")
    epoch = pd.DatetimeIndex([timestamp]).as_unit(unit).asi8[0]
    source = _wide(tmp_path / "alpha.parquet", [epoch], {"BTCUSDT": [1.0]})
    output = tmp_path / "out"
    signal_adapter.import_wide_signals(
        source,
        output,
        timestamp_unit=unit,
        timestamp_label="bar_end",
    )
    assert _read(output)["timestamp"].iloc[0] == timestamp


def test_dangling_output_symlink_is_rejected(tmp_path: pathlib.Path) -> None:
    """An existing symlink occupies the output even if its target is absent."""
    source = _wide(tmp_path / "alpha.parquet", [1, 2, 3])
    output = tmp_path / "out"
    target = tmp_path / "target"
    output.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError, match="already exists"):
        signal_adapter.import_wide_signals(
            source,
            output,
            timestamp_unit="s",
            timestamp_label="bar_end",
        )
    assert output.is_symlink()
    assert not target.exists()
