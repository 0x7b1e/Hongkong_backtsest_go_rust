"""Fixed-schema execution audits with bounded Parquet write buffers."""

import pathlib
from typing import Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

AUDIT_BUFFER_ROWS = 4096
_TIME = pa.timestamp("ns", tz="Asia/Hong_Kong")
_COMMON_FIELDS = [
    ("timestamp", _TIME),
    ("bar_end", _TIME),
    ("symbol", pa.string()),
    ("side", pa.string()),
    ("requested", pa.float64()),
    ("filled", pa.float64()),
]
TRADE_SCHEMA = pa.schema(
    _COMMON_FIELDS
    + [
        ("price", pa.float64()),
        ("notional", pa.float64()),
        ("fees", pa.float64()),
        ("stamp_duty", pa.float64()),
        ("sfc_levy", pa.float64()),
        ("afrc_levy", pa.float64()),
        ("trading_fee", pa.float64()),
        ("trading_tariff", pa.float64()),
        ("commission", pa.float64()),
        ("cash", pa.float64()),
        ("reason", pa.string()),
    ]
)
RESTRICTION_SCHEMA = pa.schema(_COMMON_FIELDS + [("reason", pa.string())])


class AuditRecords:
    """Collect one fixed audit table in memory or stream it to Parquet.

    Streaming retains at most 4096 finalized record dictionaries. Empty
    streams still create a typed Parquet file. In-memory mode is provided
    for small interactive experiments and backwards-compatible results.
    """

    def __init__(
        self,
        kind: Literal["trades", "restrictions"],
        directory: pathlib.Path | None = None,
        buffer_rows: int = AUDIT_BUFFER_ROWS,
    ) -> None:
        """Open one audit sink with an explicit memory budget.

        Args:
            kind: One of the two execution audit tables.
            directory: Streaming destination; None retains records in memory.
            buffer_rows: Flush threshold, bounded by AUDIT_BUFFER_ROWS.

        Raises:
            ValueError: The table or buffer budget is invalid.
            FileExistsError: A previous audit would be overwritten.
        """
        if kind not in {"trades", "restrictions"}:
            raise ValueError("Unknown execution audit table")
        if not 1 <= buffer_rows <= AUDIT_BUFFER_ROWS:
            raise ValueError("Audit buffer must contain 1 to 4096 rows")
        self.schema = TRADE_SCHEMA if kind == "trades" else RESTRICTION_SCHEMA
        self.count = 0
        self.peak_buffered_rows = 0
        self._buffer_rows = buffer_rows
        self._records: list[dict] = []
        self._writer: pq.ParquetWriter | None = None
        self._closed = False
        if directory is not None:
            directory = pathlib.Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            filename = directory / f"{kind}.parquet"
            if filename.exists():
                raise FileExistsError(filename)
            self._writer = pq.ParquetWriter(
                filename, self.schema, compression="zstd"
            )

    @property
    def buffered_rows(self) -> int:
        """Return the number of record dictionaries currently retained."""
        return len(self._records)

    def append(self, row: dict) -> None:
        """Append a finalized immutable-by-convention execution record."""
        if self._closed:
            raise RuntimeError("Execution audit is already closed")
        self._records.append(row)
        self.count += 1
        self.peak_buffered_rows = max(
            self.peak_buffered_rows, len(self._records)
        )
        if self._writer is not None and len(self._records) >= self._buffer_rows:
            self._flush()

    def _flush(self) -> None:
        """Write a bounded record batch and release its dictionaries."""
        if self._writer is not None and self._records:
            table = pa.Table.from_pylist(self._records, schema=self.schema)
            self._writer.write_table(table)
            self._records.clear()

    def to_frame(self) -> pd.DataFrame:
        """Return retained records, or a typed empty frame for streamed data."""
        rows = [] if self._writer is not None else self._records
        return pa.Table.from_pylist(rows, schema=self.schema).to_pandas()

    def close(self) -> None:
        """Flush remaining records and close a valid Parquet file."""
        if self._closed:
            return
        self._closed = True
        if self._writer is not None:
            try:
                self._flush()
            finally:
                self._writer.close()

    def __enter__(self) -> "AuditRecords":
        """Return the opened audit sink."""
        return self

    def __exit__(self, *_error: object) -> None:
        """Close audit files even if execution raises an exception."""
        self.close()
