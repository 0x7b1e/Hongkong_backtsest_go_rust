"""Source-preserving CSV import and exact provider comparison audits."""

import hashlib
import json
import pathlib

import pandas as pd
import pytest

from hkfactor import validation as data_validation


@pytest.mark.parametrize("title", ["", "HSI 日线前复权行情\n"])
def test_utf8_index_csv_import_leaves_source_unchanged(
    tmp_path: pathlib.Path,
    title: str,
) -> None:
    """Accept source title rows and retain Chinese UTF-8 bytes verbatim."""
    source = tmp_path / "指数.csv"
    source.write_text(
        title + "日期,开盘,最高,最低,收盘,成交量,成交额\n"
        "20240102,100,102,99,101,5000,501000\n"
        "20240103,101,104,100,103,6000,610000\n",
        encoding="utf-8-sig",
    )
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / "canonical" / "HSI.parquet"
    data_validation.import_csv(
        source, destination, "1d", "HSI", kind="benchmark"
    )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == checksum
    frame = pd.read_parquet(destination)
    assert {"timestamp", "symbol", "close", "open"} <= set(frame.columns)
    assert frame["open"].tolist() == [100, 101]
    assert frame["symbol"].tolist() == ["HSI", "HSI"]
    assert frame["timestamp"].dt.strftime("%H:%M").tolist() == [
        "16:00",
        "16:00",
    ]
    assert str(frame["timestamp"].dt.tz) == "Asia/Hong_Kong"
    assert frame["close"].tolist() == [101, 103]


def test_stock_csv_preserves_leading_zero_and_adjusted_price_basis(
    tmp_path: pathlib.Path,
) -> None:
    """Never strip code zeros or silently promote qfq prices to raw."""
    source = tmp_path / "00001.HK_qfq.csv"
    source.write_text(
        "代码,名称,日期,开盘,最高,最低,收盘,成交量,昨收\n"
        "00001.HK,长和,2024-01-02,10,12,9,11,1000,10\n",
        encoding="utf-8-sig",
    )
    before = source.read_bytes()
    destination = tmp_path / "stock.parquet"
    data_validation.import_csv(source, destination, "1d", "00001.HK")
    frame = pd.read_parquet(destination)
    assert frame.loc[0, "symbol"] == "00001.HK"
    assert frame.loc[0, "price_basis"] == "adjusted"
    assert frame.loc[0, "open"] == 10
    assert frame.loc[0, "volume"] == 1000
    assert source.read_bytes() == before
    with pytest.raises(FileExistsError):
        data_validation.import_csv(source, destination, "1d", "00001.HK")


def test_intraday_csv_hhmm_dates_and_failed_import_cleanup(
    tmp_path: pathlib.Path,
) -> None:
    """Map compact source clocks locally and remove failed output only."""
    source = tmp_path / "minutes.csv"
    source.write_text(
        "代码,日期,时间,开盘,最高,最低,收盘,成交量\n"
        "00001.HK,20240102,935,10,11,9,10.5,100\n",
        encoding="utf-8-sig",
    )
    before = source.read_bytes()
    destination = tmp_path / "minutes.parquet"
    data_validation.import_csv(
        source, destination, "5m", "00001.HK", price_basis="raw"
    )
    frame = pd.read_parquet(destination)
    assert frame.timestamp.iloc[0] == pd.Timestamp(
        "2024-01-02 09:35", tz="Asia/Hong_Kong"
    )
    assert frame.price_basis.iloc[0] == "raw"
    rejected = tmp_path / "unexpected.parquet"
    with pytest.raises(ValueError, match="unexpected security"):
        data_validation.import_csv(source, rejected, "5m", "00002.HK")
    assert not rejected.exists()
    assert source.read_bytes() == before


def _quotes(days: list[str], closes: list[float]) -> pd.DataFrame:
    """Produce comparable ordinary-stock quotes with unchanged volume."""
    return pd.DataFrame(
        {
            "timestamp": days,
            "symbol": "00001.HK",
            "close": closes,
            "volume": 1000.0,
        }
    )


def test_comparison_counts_all_differences_with_bounded_examples(
    tmp_path: pathlib.Path,
) -> None:
    """Example truncation must not truncate mismatch or missing-row counts."""
    local, reference = tmp_path / "local.parquet", tmp_path / "api.parquet"
    _quotes(
        ["2024-01-01", "2024-01-02", "2024-01-03"], [10, 11, 12]
    ).to_parquet(local, index=False)
    _quotes(
        ["2024-01-02", "2024-01-03", "2024-01-04"], [21, 22, 23]
    ).to_parquet(reference, index=False)
    output = tmp_path / "reports" / "comparison"
    result = data_validation.compare_snapshots(
        local, reference, output, max_differences=1
    )
    assert result["status"] == "completed"
    assert result["total_differences"] == 4
    assert result["examples_truncated"] is True
    summary = result["symbols"][0]
    assert summary["matched_rows"] == 2
    assert summary["missing_local"] == summary["missing_reference"] == 1
    assert summary["mismatched_cells"] == summary["close_mismatches"] == 2
    assert summary["volume_mismatches"] == 0
    assert len(pd.read_parquet(output / "differences.parquet")) == 1
    persisted = json.loads((output / "comparison.json").read_text())
    assert persisted["total_differences"] == 4
    with pytest.raises(FileExistsError):
        data_validation.compare_snapshots(local, reference, output)


def test_comparison_tolerances_and_duplicate_keys(
    tmp_path: pathlib.Path,
) -> None:
    """Apply explicit numeric tolerances and reject ambiguous snapshots."""
    local, reference = tmp_path / "local.parquet", tmp_path / "api.parquet"
    frame = _quotes(["2024-01-02"], [10.0])
    frame.to_parquet(local, index=False)
    _quotes(["2024-01-02"], [10.000001]).to_parquet(reference, index=False)
    result = data_validation.compare_snapshots(
        local, reference, tmp_path / "comparison", atol=0.000002, rtol=0
    )
    assert result["total_differences"] == 0
    assert result["examples_truncated"] is False
    pd.concat([frame, frame], ignore_index=True).to_parquet(local, index=False)
    with pytest.raises(ValueError, match="duplicate timestamp/symbol"):
        data_validation.compare_snapshots(
            local, reference, tmp_path / "duplicate"
        )


def test_provider_fetch_is_called_and_credentials_are_never_saved(
    tmp_path: pathlib.Path,
) -> None:
    """Compare fetched data and leave only permitted comparison artifacts."""
    local = tmp_path / "local.parquet"
    frame = _quotes(["2024-01-02", "2024-01-03"], [10.0, 11.0])
    frame.to_parquet(local, index=False)

    class Provider:
        """Minimal authenticated provider whose secrets are not quote fields."""

        credentials = "private_provider_token_for_test"

        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str, str]] = []

        def fetch_prices(
            self, symbol: str, t_start: str, t_end: str, frequency: str
        ) -> pd.DataFrame:
            self.calls.append((symbol, t_start, t_end, frequency))
            return frame.copy()

    provider = Provider()
    output = tmp_path / "provider" / "comparison"
    result = data_validation.compare_with_provider(local, provider, output)
    assert provider.calls == [("00001.HK", "2024-01-02", "2024-01-03", "1d")]
    assert result["status"] == "completed"
    assert result["total_differences"] == 0
    secret = provider.credentials.encode()
    assert all(
        secret not in path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    )
    assert not list(output.parent.glob("api_quotes_*"))
