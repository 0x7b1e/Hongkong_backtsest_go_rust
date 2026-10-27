"""Import the gated hongkong_day_kline directory into hkfactor's daily schema.

The dataset card does not publish its CSV header or volume unit. The importer
therefore checks both explicitly and never guesses whether volume is lots.
"""

import hashlib
import json
import pathlib
import re
import shutil
import uuid

import numpy as np
import pandas as pd

_FOLDERS = {"raw": "不复权", "adjusted": "后复权"}
_FILENAME = re.compile(r"^(\d{5}\.HK)(?:_.*)?\.csv$", re.IGNORECASE)
_ALIASES = {
    "timestamp": ("日期", "交易日期", "date", "timestamp"),
    "open": ("开盘", "开盘价", "open"),
    "high": ("最高", "最高价", "high"),
    "low": ("最低", "最低价", "low"),
    "close": ("收盘", "收盘价", "close"),
    "volume": ("成交量", "volume", "vol"),
    "symbol": ("代码", "股票代码", "symbol", "code"),
    "lot_size": ("每手股数", "lot_size"),
    "cash_dividend": ("每股现金股息", "cash_dividend"),
    "share_multiplier": ("股份倍数", "share_multiplier"),
}
_REQUIRED = ("timestamp", "open", "high", "low", "close", "volume")


def _columns(headers: list[str], mapping: dict[str, str]) -> dict[str, str]:
    if not isinstance(mapping, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in mapping.items()
    ):
        raise ValueError("Column mapping must be a JSON object of strings")
    if set(mapping) - set(_ALIASES):
        raise ValueError(
            f"Unknown column mappings: {sorted(set(mapping) - set(_ALIASES))}"
        )
    if len(headers) != len(set(headers)):
        raise ValueError("Duplicate CSV column headers")
    resolved = {}
    for canonical, aliases in _ALIASES.items():
        if canonical in mapping:
            matches = (
                [mapping[canonical]] if mapping[canonical] in headers else []
            )
        else:
            matches = [
                column
                for column in headers
                if column.strip().casefold()
                in {name.casefold() for name in aliases}
            ]
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous {canonical} columns: {matches}; supply --columns"
            )
        if matches:
            resolved[canonical] = matches[0]
    missing = set(_REQUIRED) - set(resolved)
    if missing:
        raise ValueError(
            f"Missing CSV fields {sorted(missing)}; headers={headers}. "
            "Supply a JSON mapping with canonical keys and exact CSV headers"
        )
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("One CSV column cannot map to multiple fields")
    return resolved


def _read_csv(path: pathlib.Path) -> tuple[pd.DataFrame, str]:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype=str), encoding
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unsupported CSV encoding: {path}")


def _prices(
    source: pd.DataFrame, columns: dict[str, str], symbol: str, basis: str
) -> pd.DataFrame:
    dates = pd.to_datetime(
        source[columns["timestamp"]], format="mixed", errors="raise"
    )
    if not isinstance(dates.dtype, pd.DatetimeTZDtype):
        dates = dates.dt.tz_localize("Asia/Hong_Kong")
    else:
        dates = dates.dt.tz_convert("Asia/Hong_Kong")
    if dates.isna().any() or not (dates == dates.dt.normalize()).all():
        raise ValueError(
            f"{symbol}: expected date-only daily rows in Hong Kong time"
        )
    if dates.duplicated().any():
        raise ValueError(f"{symbol}: duplicate trading dates")
    if "symbol" in columns:
        observed = source[columns["symbol"]].str.strip().str.upper()
        observed = observed.map(
            lambda code: f"{code.zfill(5)}.HK" if code.isdigit() else code
        )
        if not observed.eq(symbol).all():
            raise ValueError(f"{symbol}: CSV code disagrees with filename")
    result = pd.DataFrame(
        {"timestamp": dates.dt.tz_localize(None) + pd.Timedelta(hours=16)}
    )
    result["symbol"] = symbol
    for name in ("open", "high", "low", "close", "volume"):
        result[name] = pd.to_numeric(
            source[columns[name]].str.replace(",", "", regex=False),
            errors="raise",
        ).astype("float64")
    if (
        not np.isfinite(result[["open", "high", "low", "close", "volume"]])
        .all()
        .all()
    ):
        raise ValueError(f"{symbol}: missing or nonfinite OHLCV")
    if (result[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"{symbol}: nonpositive price")
    if (result["volume"] < 0).any():
        raise ValueError(f"{symbol}: negative share volume")
    if (
        (result["high"] < result[["open", "close"]].max(axis=1))
        | (result["low"] > result[["open", "close"]].min(axis=1))
        | (result["low"] > result["high"])
    ).any():
        raise ValueError(f"{symbol}: invalid OHLC envelope")
    for name, default in (("cash_dividend", 0.0), ("share_multiplier", 1.0)):
        result[name] = (
            pd.to_numeric(source[columns[name]], errors="raise").astype(
                "float64"
            )
            if name in columns
            else default
        )
    if "lot_size" in columns:
        result["lot_size"] = pd.to_numeric(
            source[columns["lot_size"]], errors="raise"
        ).astype("float64")
        if (
            ~np.isfinite(result["lot_size"])
            | (result["lot_size"] <= 0)
            | (result["lot_size"] % 1 != 0)
        ).any():
            raise ValueError(f"{symbol}: invalid historical board-lot size")
    if (
        not np.isfinite(result[["cash_dividend", "share_multiplier"]])
        .all()
        .all()
        or (result["cash_dividend"] < 0).any()
        or (result["share_multiplier"] <= 0).any()
    ):
        raise ValueError(f"{symbol}: invalid corporate action")
    result["price_basis"] = basis
    return result.sort_values("timestamp", kind="stable").reset_index(drop=True)


def import_hf_daily(
    dataset_root: str | pathlib.Path,
    output: str | pathlib.Path,
    *,
    basis: str = "raw",
    columns: dict[str, str] | None = None,
    volume_unit: str,
    symbols: set[str] | None = None,
) -> dict:
    """Convert one adjustment folder to per-security canonical Parquet files.

    The caller must verify the vendor's volume unit is shares. A half-import
    is removed on error and an existing output directory is never replaced.
    """
    if basis not in _FOLDERS:
        raise ValueError("basis must be raw or adjusted")
    if volume_unit != "shares":
        raise ValueError(
            "Verify volume is shares; lots require historical lot conversion"
        )
    folder = pathlib.Path(dataset_root) / _FOLDERS[basis]
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing dataset folder: {folder}")
    files = sorted(folder.glob("*.csv"))
    if symbols is not None:
        symbols = {item.upper() for item in symbols}
        files = [
            p
            for p in files
            if _FILENAME.fullmatch(p.name)
            and _FILENAME.fullmatch(p.name).group(1).upper() in symbols
        ]
    if not files:
        raise ValueError(f"No selected stock CSVs in {folder}")
    output = pathlib.Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}-{uuid.uuid4().hex}.tmp"
    staging.mkdir()
    entries = []
    seen = set()
    lot_size_coverage = None
    try:
        for path in files:
            matched = _FILENAME.fullmatch(path.name)
            if matched is None:
                raise ValueError(f"Unexpected filename: {path.name}")
            symbol = matched.group(1).upper()
            if symbol in seen:
                raise ValueError(f"Duplicate security CSV: {symbol}")
            seen.add(symbol)
            source, encoding = _read_csv(path)
            mapping = _columns(list(source.columns), columns or {})
            if lot_size_coverage is None:
                lot_size_coverage = "lot_size" in mapping
            elif lot_size_coverage != ("lot_size" in mapping):
                raise ValueError(
                    "Inconsistent lot_size fields across CSV files; "
                    "supply a complete historical lot source before import"
                )
            if source.empty:
                raise ValueError(f"{symbol}: empty CSV")
            frame = _prices(source, mapping, symbol, basis)
            frame.to_parquet(staging / f"part-{symbol}.parquet", index=False)
            entries.append(
                {
                    "symbol": symbol,
                    "source": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "encoding": encoding,
                    "columns": mapping,
                    "rows": len(frame),
                    "first_date": frame["timestamp"].iloc[0].date().isoformat(),
                    "last_date": frame["timestamp"].iloc[-1].date().isoformat(),
                    "has_lot_size": "lot_size" in mapping,
                    "has_corporate_actions": {
                        "cash_dividend",
                        "share_multiplier",
                    }.issubset(mapping),
                }
            )
        if symbols is not None and symbols - seen:
            raise ValueError(
                f"Missing requested securities: {sorted(symbols - seen)}"
            )
        manifest = {
            "source": "Crypt-dataset/hongkong_day_kline",
            "basis": basis,
            "volume_unit_verified_by_caller": volume_unit,
            "security_count": len(entries),
            "row_count": sum(item["rows"] for item in entries),
            "files": entries,
        }
        (staging / "_import_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
