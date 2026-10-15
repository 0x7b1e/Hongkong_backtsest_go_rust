"""Read source index minutes without changing bar labels or source bytes."""

import hashlib
import pathlib

import pandas as pd
import pytest

from hkfactor import csv_data

_HEADER = "日期,时间,开盘,最高,最低,收盘,成交量,成交额\n"
_ROWS = (
    "20240102,930,100,102,99,101,5000,501000\n"
    "20240102,0931,101,104,100,103,6000,610000\n"
    "20240102,0932,103,105,102,104,7000,720000\n"
    "20240103,930,104,106,103,105,8000,830000\n"
    "20240103,0931,105,107,104,106,9000,940000\n"
)
_COLUMNS = [
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]


def _write_index(
    directory: pathlib.Path,
    *,
    rows: str = _ROWS,
    encoding: str = "utf-8-sig",
    title: str = "",
) -> pathlib.Path:
    """Write representative Chinese-header provider index records."""
    source = directory / "指数.csv"
    source.write_text(title + _HEADER + rows, encoding=encoding)
    return source


@pytest.mark.parametrize("frequency", ["1m", "5m"])
@pytest.mark.parametrize("encoding", ["utf-8-sig", "gb18030"])
@pytest.mark.parametrize("title", ["", "HSI 分钟前复权行情\n"])
def test_source_codecs_titles_and_exact_minute_labels(
    tmp_path: pathlib.Path,
    frequency: str,
    encoding: str,
    title: str,
) -> None:
    """Keep all source values and HHMM labels with Hong Kong timezone."""
    source = _write_index(tmp_path, encoding=encoding, title=title)
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    frame = csv_data.read_index_csv(source, frequency, chunksize=2)
    assert list(frame.columns) == _COLUMNS
    assert frame.timestamp.tolist() == [
        pd.Timestamp(text, tz="Asia/Hong_Kong")
        for text in [
            "2024-01-02 09:30",
            "2024-01-02 09:31",
            "2024-01-02 09:32",
            "2024-01-03 09:30",
            "2024-01-03 09:31",
        ]
    ]
    assert frame.symbol.tolist() == ["HSI"] * 5
    assert frame.open.tolist() == [100, 101, 103, 104, 105]
    assert frame.high.tolist() == [102, 104, 105, 106, 107]
    assert frame.low.tolist() == [99, 100, 102, 103, 104]
    assert frame.close.tolist() == [101, 103, 104, 105, 106]
    assert frame.volume.tolist() == [5000, 6000, 7000, 8000, 9000]
    assert frame.amount.tolist() == [501000, 610000, 720000, 830000, 940000]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == checksum


def test_chunked_read_matches_full_read_and_bounds_cross_chunks(
    tmp_path: pathlib.Path,
) -> None:
    """Inclusive local boundaries work across batches and date changes."""
    source = _write_index(tmp_path)
    full = csv_data.read_index_csv(source)
    chunks = list(csv_data.iter_index_csv(source, chunksize=2))
    assert all(len(chunk) <= 2 for chunk in chunks)
    pd.testing.assert_frame_equal(pd.concat(chunks, ignore_index=True), full)
    bounded = csv_data.read_index_csv(
        source,
        t_start="2024-01-02 09:31",
        t_end="2024-01-03 09:30",
        chunksize=2,
    )
    pd.testing.assert_frame_equal(
        bounded, full.iloc[1:4].reset_index(drop=True)
    )
    local_day = csv_data.read_index_csv(
        source, t_start="2024-01-02", t_end="2024-01-02", chunksize=2
    )
    pd.testing.assert_frame_equal(
        local_day, full.iloc[:3].reset_index(drop=True)
    )


def test_empty_time_range_retains_canonical_types(
    tmp_path: pathlib.Path,
) -> None:
    """No matching minutes still return a usable typed canonical frame."""
    source = _write_index(tmp_path)
    frame = csv_data.read_index_csv(source, t_start="2025-01-01")
    assert frame.empty
    assert list(frame.columns) == _COLUMNS
    assert str(frame.timestamp.dt.tz) == "Asia/Hong_Kong"
    for name in _COLUMNS[2:]:
        assert pd.api.types.is_numeric_dtype(frame[name])


def test_negative_source_amount_is_preserved_and_counted(
    tmp_path: pathlib.Path,
) -> None:
    """Keep known provider amount anomalies visible across streamed batches."""
    source = _write_index(
        tmp_path,
        rows=_ROWS.replace("501000", "-8192.2").replace("940000", "-75.36"),
    )
    chunks = list(csv_data.iter_index_csv(source, chunksize=2))
    assert [chunk.attrs["negative_amount_count"] for chunk in chunks] == [
        1,
        0,
        1,
    ]
    frame = csv_data.read_index_csv(source, chunksize=2)
    assert frame.amount.tolist() == [-8192.2, 610000, 720000, 830000, -75.36]
    assert frame.attrs["negative_amount_count"] == 2


@pytest.mark.parametrize("clock", ["2360", "2400", "093500", "935.5"])
def test_invalid_hhmm_is_rejected(
    tmp_path: pathlib.Path,
    clock: str,
) -> None:
    """Invalid times cannot be truncated, rolled forward, or relabeled."""
    source = _write_index(
        tmp_path, rows=f"20240102,{clock},100,102,99,101,5000,501000\n"
    )
    with pytest.raises(ValueError):
        csv_data.read_index_csv(source)


@pytest.mark.parametrize("clocks", [("0930", "0930"), ("0931", "0930")])
def test_duplicate_or_reversed_minutes_across_chunks_are_rejected(
    tmp_path: pathlib.Path,
    clocks: tuple[str, str],
) -> None:
    """Chunk boundaries must not hide repeated or reversed source keys."""
    source = _write_index(
        tmp_path,
        rows="".join(
            f"20240102,{clock},100,102,99,101,5000,501000\n" for clock in clocks
        ),
    )
    with pytest.raises(ValueError):
        list(csv_data.iter_index_csv(source, chunksize=1))


@pytest.mark.parametrize(
    "quotes",
    [
        "100,100,99,101,5000,501000",
        "100,102,101,101,5000,501000",
        "100,98,99,100,5000,501000",
        "100,102,99,inf,5000,501000",
        "100,102,99,NaN,5000,501000",
        "100,102,99,101,-1,501000",
        "100,102,99,101,5000,broken",
    ],
)
def test_invalid_ohlc_or_nonfinite_negative_volume_is_rejected(
    tmp_path: pathlib.Path,
    quotes: str,
) -> None:
    """Reject unusable numeric bars before downstream research consumes them."""
    source = _write_index(tmp_path, rows=f"20240102,0930,{quotes}\n")
    with pytest.raises(ValueError):
        csv_data.read_index_csv(source)
