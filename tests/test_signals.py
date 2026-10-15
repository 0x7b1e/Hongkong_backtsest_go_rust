"""Publication alignment and vendor signal-contract regression tests."""

import datetime
import pathlib
import types

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hkfactor import signals as signal_module


def _panel(
    times: list[str], symbols: tuple[str, ...] = ("00001.HK",)
) -> types.SimpleNamespace:
    """Supply the minimal read-only market contract used by the adapter."""
    timestamps = pd.DatetimeIndex(times).tz_localize("Asia/Hong_Kong")
    return types.SimpleNamespace(
        timestamps=timestamps,
        symbols=symbols,
        shape=(len(times), len(symbols)),
    )


def _write(path: pathlib.Path, rows: list[dict]) -> pathlib.Path:
    """Write long signals using the same Arrow path as external producers."""
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_late_publication_and_warm_start_are_asof(
    tmp_path: pathlib.Path,
) -> None:
    """An old late feature never replaces an already known newer feature."""
    path = _write(
        tmp_path / "signals.parquet",
        [
            {
                "timestamp": "2024-01-01 16:00",
                "available_at": "2024-01-01 16:00",
                "symbol": "00001.HK",
                "factor": "momentum",
                "value": 1.0,
            },
            {
                "timestamp": "2024-01-02 09:35",
                "available_at": "2024-01-02 09:45",
                "symbol": "00001.HK",
                "factor": "momentum",
                "value": 2.0,
            },
            {
                "timestamp": "2024-01-02 09:40",
                "available_at": "2024-01-02 09:40",
                "symbol": "00001.HK",
                "factor": "momentum",
                "value": 3.0,
            },
            {
                "timestamp": "2024-01-02 09:50",
                "available_at": "2024-01-02 09:55",
                "symbol": "00001.HK",
                "factor": "momentum",
                "value": 4.0,
            },
        ],
    )
    panel = _panel(
        [
            f"2024-01-02 {clock}"
            for clock in ("09:35", "09:40", "09:45", "09:50", "09:55")
        ]
    )
    signal = signal_module.SignalStore(path).read_factor("momentum", panel)
    np.testing.assert_array_equal(signal.values[:, 0], [1, 3, 3, 3, 4])
    np.testing.assert_array_equal(
        signal.updated, [True, True, True, False, True]
    )


def test_epoch_numbers_cannot_silently_become_1970_signals(
    tmp_path: pathlib.Path,
) -> None:
    """Undeclared milliseconds must not make future features seem old."""
    path = _write(
        tmp_path / "milliseconds.parquet",
        [
            {
                "timestamp": 1751620500000,
                "available_at": 1751620500000,
                "symbol": "00001.HK",
                "factor": "alpha001",
                "value": 1.0,
            }
        ],
    )
    with pytest.raises(ValueError, match="explicit epoch unit"):
        signal_module.SignalStore(path).read_factor(
            "alpha001", _panel(["2025-07-04 17:30"])
        )


def test_no_backfill_and_null_explicitly_clears(tmp_path: pathlib.Path) -> None:
    """Missing or not-yet-published features do not create future scores."""
    path = _write(
        tmp_path / "signals.parquet",
        [
            {
                "timestamp": "2024-01-02 09:40",
                "symbol": "00001.HK",
                "factor": "a",
                "value": 2.0,
            },
            {
                "timestamp": "2024-01-02 09:50",
                "symbol": "00001.HK",
                "factor": "a",
                "value": np.nan,
            },
        ],
    )
    panel = _panel(
        [
            f"2024-01-02 {clock}"
            for clock in ("09:35", "09:40", "09:45", "09:50", "09:55")
        ]
    )
    signal = signal_module.SignalStore(path).read_factor("a", panel)
    np.testing.assert_allclose(
        signal.values[:, 0], [np.nan, 2, 2, np.nan, np.nan], equal_nan=True
    )


@pytest.mark.parametrize("day", ["2024-01-02", datetime.date(2024, 1, 2)])
def test_plain_dates_arrive_at_hong_kong_close(
    tmp_path: pathlib.Path, day
) -> None:
    """Daily dates cannot accidentally become available at midnight."""
    path = _write(
        tmp_path / "signals.parquet",
        [
            {"timestamp": day, "symbol": "00001.HK", "factor": "a", "value": 2},
        ],
    )
    signal = signal_module.SignalStore(path).read_factor(
        "a",
        _panel(["2024-01-02 15:55", "2024-01-02 16:00", "2024-01-03 09:35"]),
    )
    np.testing.assert_allclose(
        signal.values[:, 0], [np.nan, 2, 2], equal_nan=True
    )


def test_aware_times_and_vendor_mapping(tmp_path: pathlib.Path) -> None:
    """A UTC publication is aligned to the same Hong Kong trading bar."""
    path = _write(
        tmp_path / "signals.parquet",
        [
            {
                "feature_time": "2024-01-02T01:35:00+00:00",
                "published": "2024-01-02T01:40:00+00:00",
                "ticker": "00001.HK",
                "alpha": "a",
                "score": 7.0,
            }
        ],
    )
    columns = signal_module.SignalColumns(
        timestamp="feature_time",
        symbol="ticker",
        factor="alpha",
        value="score",
        available_at="published",
    )
    store = signal_module.SignalStore(path, columns)
    assert store.factors() == ["a"]
    signal = store.read_factor(
        "a", _panel(["2024-01-02 09:35", "2024-01-02 09:40"])
    )
    np.testing.assert_allclose(signal.values[:, 0], [np.nan, 7], equal_nan=True)


@pytest.mark.parametrize(
    "change,expected",
    [
        ({"symbol": "99999.HK"}, "Unknown signal symbols"),
        ({"value": np.inf}, "infinite"),
        ({"available_at": "2024-01-02 09:30"}, "precede"),
    ],
)
def test_invalid_signals_fail_explicitly(
    tmp_path: pathlib.Path,
    change: dict,
    expected: str,
) -> None:
    """Data errors must not turn into apparently successful zero returns."""
    row = {
        "timestamp": "2024-01-02 09:35",
        "symbol": "00001.HK",
        "factor": "a",
        "value": 1.0,
    }
    row.update(change)
    path = _write(tmp_path / "signals.parquet", [row])
    with pytest.raises(ValueError, match=expected):
        signal_module.SignalStore(path).read_factor(
            "a", _panel(["2024-01-02 09:35"])
        )


def test_duplicates_across_parquet_fragments_are_rejected(
    tmp_path: pathlib.Path,
) -> None:
    """Duplicate checks apply across scanner batches and partition files."""
    partition = tmp_path / "factor=a"
    partition.mkdir()
    row = {"timestamp": "2024-01-02 09:35", "symbol": "00001.HK", "value": 1.0}
    for filename in ("part-0.parquet", "part-1.parquet"):
        _write(partition / filename, [row])
    store = signal_module.SignalStore(tmp_path)
    assert store.factors() == ["a"]
    with pytest.raises(ValueError, match="Duplicate"):
        store.read_factor("a", _panel(["2024-01-02 09:35"]))


def test_typed_timestamp_pushdown_preserves_warm_start(
    tmp_path: pathlib.Path,
) -> None:
    """Arrow predicate pushdown cannot discard the final pre-period feature."""
    times = pd.to_datetime(["2024-01-01 16:00", "2024-01-04 16:00"])
    table = pa.table(
        {
            "timestamp": times,
            "symbol": ["00001.HK", "00001.HK"],
            "factor": ["a", "a"],
            "value": [1.0, 999.0],
        }
    )
    path = tmp_path / "signals.parquet"
    pq.write_table(table, path)
    signal = signal_module.SignalStore(path).read_factor(
        "a", _panel(["2024-01-02 16:00", "2024-01-03 16:00"])
    )
    np.testing.assert_array_equal(signal.values[:, 0], [1, 1])


def test_streaming_partitions_are_per_security_and_cleaned(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shuffled batches produce exact signals with one history per read."""
    symbols = tuple(f"{number:05d}.HK" for number in range(1, 11))
    clocks = [f"2024-01-02 09:{minute:02d}" for minute in range(35, 40)]
    rows = [
        {
            "timestamp": clock,
            "symbol": symbol,
            "factor": "a",
            "value": float(bar * 100 + column),
        }
        for bar, clock in enumerate(clocks)
        for column, symbol in enumerate(symbols)
    ]
    path = tmp_path / "signals.parquet"
    pd.DataFrame(rows).sample(frac=1, random_state=7).to_parquet(
        path, index=False
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    marker = scratch / "keep.txt"
    marker.write_text("preserve caller-owned files")
    monkeypatch.setattr(signal_module, "_BATCH_SIZE", 7)
    original = signal_module._align_symbol
    histories = []

    def observe_partition(events_path: pathlib.Path, *args) -> None:
        histories.append(events_path.stat().st_size)
        assert events_path.parent.parent == scratch
        original(events_path, *args)

    monkeypatch.setattr(signal_module, "_align_symbol", observe_partition)
    signal = signal_module.SignalStore(path).read_factor(
        "a", _panel(clocks, symbols), scratch_dir=scratch
    )
    expected = np.arange(5)[:, None] * 100 + np.arange(10)[None, :]
    np.testing.assert_array_equal(signal.values, expected)
    assert histories == [5 * 24] * 10
    assert list(scratch.iterdir()) == [marker]


@pytest.mark.parametrize("failure", ["scan", "duplicate"])
def test_signal_scratch_is_cleaned_after_validation_failure(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Errors during scanning or history alignment leave no event files."""
    rows = [
        {
            "timestamp": "2024-01-02 09:35",
            "symbol": "00001.HK",
            "factor": "a",
            "value": 1.0,
        },
        {
            "timestamp": "2024-01-02 09:40",
            "symbol": "00001.HK",
            "factor": "a",
            "value": 2.0,
        },
    ]
    if failure == "scan":
        rows[1]["value"] = np.inf
        expected = "infinite"
    else:
        rows[1]["timestamp"] = rows[0]["timestamp"]
        expected = "Duplicate"
    path = _write(tmp_path / "signals.parquet", rows)
    scratch = tmp_path / "scratch"
    monkeypatch.setattr(signal_module, "_BATCH_SIZE", 1)
    with pytest.raises(ValueError, match=expected):
        signal_module.SignalStore(path).read_factor(
            "a", _panel(["2024-01-02 09:40"]), scratch_dir=scratch
        )
    assert list(scratch.iterdir()) == []


def test_hive_factor_discovery_reads_no_signal_batches(
    tmp_path: pathlib.Path,
) -> None:
    """Fully partitioned inputs expose names without scanning their rows."""
    for factor in ("momentum", "reversal"):
        directory = tmp_path / f"factor={factor}"
        directory.mkdir()
        _write(
            directory / "part.parquet",
            [
                {
                    "timestamp": "2024-01-02",
                    "symbol": "00001.HK",
                    "value": 1.0,
                }
            ],
        )
    store = signal_module.SignalStore(tmp_path)
    dataset = store._dataset

    class MetadataOnlyDataset:
        """Keep real Arrow partition expressions while prohibiting scans."""

        def get_fragments(self):
            return dataset.get_fragments()

        def scanner(self, **kwargs):
            pytest.fail("Partition discovery must not scan signal rows")

    store._dataset = MetadataOnlyDataset()
    assert store.factors() == ["momentum", "reversal"]


def test_mixed_factor_partition_and_single_file_falls_back(
    tmp_path: pathlib.Path,
) -> None:
    """Unpartitioned fragments retain names absent from hive metadata."""
    directory = tmp_path / "factor=momentum"
    directory.mkdir()
    _write(
        directory / "part.parquet",
        [
            {
                "timestamp": "2024-01-02",
                "symbol": "00001.HK",
                "value": 1.0,
            }
        ],
    )
    _write(
        tmp_path / "unpartitioned.parquet",
        [
            {
                "timestamp": "2024-01-02",
                "symbol": "00001.HK",
                "value": 2.0,
                "factor": "reversal",
            }
        ],
    )
    assert signal_module.SignalStore(tmp_path).factors() == [
        "momentum",
        "reversal",
    ]
