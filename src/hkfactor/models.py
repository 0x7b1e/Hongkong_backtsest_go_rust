"""Shared data contracts without execution or analysis dependencies."""

import dataclasses
import pathlib

import numpy as np
import pandas as pd


@dataclasses.dataclass(frozen=True)
class MarketPanel:
    """Read-only shared market arrays, shaped ``(bars, symbols)``.

    Timestamps label bar ends in Asia/Hong_Kong. Open timestamps indicate
    executable bar starts. Base weights are point-in-time snapshots known
    at the corresponding open. Session IDs count actual supplied trading
    dates. Missing quotes may be carried for valuation only; ``tradable``
    is false for such observations. VCM bands are NaN outside an explicitly
    supplied cooling-off interval. Cash dividends and share multipliers
    are applied before trading at each open.
    """

    timestamps: pd.DatetimeIndex
    open_timestamps: pd.DatetimeIndex
    symbols: tuple[str, ...]
    frequency: str
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    base_weights: np.ndarray
    benchmark_close: np.ndarray
    lot_sizes: np.ndarray
    tradable: np.ndarray
    vcm_lower: np.ndarray
    vcm_upper: np.ndarray
    vcm_eligible: np.ndarray
    cash_dividend: np.ndarray
    share_multiplier: np.ndarray
    session_id: np.ndarray
    is_session_start: np.ndarray
    is_session_end: np.ndarray
    price_basis: str = "raw"
    vcm_active: np.ndarray | None = None
    stamp_duty_exempt: np.ndarray | None = None
    benchmark_open: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        """Return the bar and security dimensions."""
        return self.open.shape

    def save(self, directory: str | pathlib.Path) -> None:
        """Persist an immutable panel for memory-mapped worker access."""
        from hkfactor import data as market_data

        market_data.save_panel(self, pathlib.Path(directory))

    @classmethod
    def load(cls, directory: str | pathlib.Path) -> "MarketPanel":
        """Load shared arrays without copying them into each worker."""
        from hkfactor import data as market_data

        return market_data.load_panel(pathlib.Path(directory))


@dataclasses.dataclass(frozen=True)
class SignalMatrix:
    """One factor's values known as of each bar end.

    Values are forward-filled only after available_at, never backward.
    ``updated`` flags bars receiving at least one new available observation.
    The engine uses an earlier completed bar, never its execution bar.
    """

    factor: str
    values: np.ndarray
    updated: np.ndarray


@dataclasses.dataclass
class BacktestResult:
    """Account results, filled orders, restrictions, and scalar metrics."""

    pnl: pd.DataFrame
    trades: pd.DataFrame
    restrictions: pd.DataFrame
    metrics: dict
    audit_streamed: bool = False


@dataclasses.dataclass
class AnalysisResult:
    """Gross research layers and cross-sectional factor statistics."""

    layers: pd.DataFrame
    ic: pd.DataFrame
    ic_intraday: pd.DataFrame
    layer_summary: pd.DataFrame
    layer_pnl: pd.DataFrame | None = None
