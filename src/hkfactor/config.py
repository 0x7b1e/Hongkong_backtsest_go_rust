"""Validated, explicit research and execution assumptions."""

import dataclasses
import datetime
import math
from typing import Literal

import pandas as pd


@dataclasses.dataclass(frozen=True)
class BacktestConfig:
    """Configure a single independent factor experiment.

    T0 executes the preceding bar's known signal and restores base targets
    from t0_flatten_at, with a final-bar attempt. T1 uses the prior session's
    known signal at the next session's open and prohibits selling shares
    younger than ``holding_days`` trading sessions. These are strategy
    definitions, not Hong Kong settlement restrictions.

    Attributes:
        t_start: First included local trading date, inclusive.
        t_end: Last included local trading date, inclusive.
        mode: Strategy holding convention.
        frequency: Canonical price-bar frequency.
        keep: Per-stock target-weight floor relative to base weights.
        initial_capital: Initial cash in HKD.
        direction: Positive means larger factor values receive more weight.
        holding_days: Minimum age of T1 shares in trading sessions.
        commission_bps: Broker commission per side in basis points.
        minimum_commission: Minimum commission per filled order in HKD.
        slippage_bps: Adverse execution slippage per side in basis points.
        participation_rate: Maximum filled shares divided by bar volume.
        round_lots: Whether to enforce provided historical board-lot sizes.
        ic_horizon_bars: Forward window used by research diagnostics.
        ic_min_assets: Minimum usable observations for cross-sectional IC.
        quantiles: Number of signal groups.
        enforce_vcm: Require supplied VCM bands for eligible instruments.
        allow_adjusted_prices: Opt into explicitly labelled research prices.
        entry_start: Inclusive local buy-window start.
        entry_end: Exclusive local buy-window end.
        exit_start: Inclusive local sell-window start.
        exit_end: Exclusive local sell-window end.
        t0_flatten_at: Time to begin restoring T0 base targets.
        order_slices: Number of child orders per target change, across bars.
        simulate_layers: Execute ten separate layer accounts with same rules.
    """

    t_start: str
    t_end: str
    mode: Literal["T0", "T1"] = "T1"
    frequency: Literal["1d", "5m", "1m"] = "1d"
    keep: float = 0.9
    initial_capital: float = 10_000_000.0
    direction: Literal[-1, 1] = 1
    holding_days: int = 1
    commission_bps: float = 2.0
    minimum_commission: float = 0.0
    slippage_bps: float = 0.0
    participation_rate: float = 0.1
    round_lots: bool = True
    ic_horizon_bars: int = 1
    ic_min_assets: int = 3
    quantiles: int = 10
    enforce_vcm: bool = False
    allow_adjusted_prices: bool = False
    entry_start: str = "09:30"
    entry_end: str = "16:00"
    exit_start: str = "09:30"
    exit_end: str = "16:00"
    t0_flatten_at: str = "15:30"
    order_slices: int = 1
    simulate_layers: bool = True

    def __post_init__(self) -> None:
        """Reject ambiguous or internally inconsistent configurations."""
        for value in [self.t_start, self.t_end]:
            if not isinstance(value, str):
                raise ValueError("Date bounds must be ISO YYYY-MM-DD strings")
            datetime.date.fromisoformat(value)
        integers = (
            self.holding_days,
            self.ic_horizon_bars,
            self.ic_min_assets,
            self.quantiles,
            self.order_slices,
            self.direction,
        )
        if any(type(value) is not int for value in integers):
            raise ValueError("Periods, layers, slices and direction need ints")
        toggles = (
            self.round_lots,
            self.enforce_vcm,
            self.allow_adjusted_prices,
            self.simulate_layers,
        )
        if any(type(value) is not bool for value in toggles):
            raise ValueError("Execution toggles must be true or false")
        if pd.Timestamp(self.t_start) > pd.Timestamp(self.t_end):
            raise ValueError("t_start must be on or before t_end")
        if self.mode not in {"T0", "T1"}:
            raise ValueError("mode must be T0 or T1")
        if self.frequency not in {"1d", "5m", "1m"}:
            raise ValueError("frequency must be 1d, 5m, or 1m")
        if self.mode == "T0" and self.frequency == "1d":
            raise ValueError("T0 requires individual-stock intraday bars")
        if not 0 <= self.keep <= 1:
            raise ValueError("keep must be between 0 and 1")
        if self.initial_capital <= 0 or self.direction not in {-1, 1}:
            raise ValueError("Invalid capital or factor direction")
        if not all(
            math.isfinite(value)
            for value in (
                self.initial_capital,
                self.keep,
                self.commission_bps,
                self.minimum_commission,
                self.slippage_bps,
                self.participation_rate,
            )
        ):
            raise ValueError("Numeric configuration must be finite")
        if self.holding_days < 1 or self.ic_horizon_bars < 1:
            raise ValueError("Holding periods must be positive")
        if self.ic_min_assets < 3 or self.quantiles < 2:
            raise ValueError("Invalid IC sample size or quantile count")
        if not 0 < self.participation_rate <= 1:
            raise ValueError("participation_rate must be in (0, 1]")
        if (
            min(
                self.commission_bps,
                self.minimum_commission,
                self.slippage_bps,
            )
            < 0
        ):
            raise ValueError("Transaction costs cannot be negative")
        if self.order_slices < 1:
            raise ValueError("order_slices must be positive")
        entry = (
            datetime.time.fromisoformat(self.entry_start),
            datetime.time.fromisoformat(self.entry_end),
        )
        exit_window = (
            datetime.time.fromisoformat(self.exit_start),
            datetime.time.fromisoformat(self.exit_end),
        )
        datetime.time.fromisoformat(self.t0_flatten_at)
        if entry[0] >= entry[1] or exit_window[0] >= exit_window[1]:
            raise ValueError("Execution-window start must precede end")

    def to_dict(self) -> dict:
        """Return JSON-serializable configuration values."""
        return dataclasses.asdict(self)
