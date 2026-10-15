"""Hong Kong cash-equity execution rules and dated transaction charges."""

import dataclasses
import datetime
import math

import pandas as pd

from hkfactor import config as backtest_config


@dataclasses.dataclass(frozen=True)
class Fees:
    """Per-side charges for a filled child order, in Hong Kong dollars."""

    stamp_duty: float = 0.0
    sfc_levy: float = 0.0
    afrc_levy: float = 0.0
    trading_fee: float = 0.0
    trading_tariff: float = 0.0
    commission: float = 0.0

    @property
    def total(self) -> float:
        """Return the sum of independently rounded charges."""
        return sum(
            (
                self.stamp_duty,
                self.sfc_levy,
                self.afrc_levy,
                self.trading_fee,
                self.trading_tariff,
                self.commission,
            )
        )


def _cent(value: float) -> float:
    """Round nonnegative amounts to cents using commercial half-up rules."""
    return math.floor(value * 100.0 + 0.5 + 1e-9) / 100.0


def transaction_fees(
    notional: float,
    timestamp: pd.Timestamp,
    config: backtest_config.BacktestConfig,
    *,
    stamp_duty_exempt: bool = False,
) -> Fees:
    """Calculate dated charges with supplied historical stamp exemptions.

    Each child fill is one transaction. Stamp duty rounds up to HKD 1;
    the levy, exchange fee and commission round separately to cents.
    Clearing/custody charges and broker-specific taxes are not included.
    Rates before November 2014 are deliberately unsupported.

    Args:
        notional: Filled consideration in HKD, excluding charges.
        timestamp: Local transaction timestamp.
        config: Broker commission assumptions.
        stamp_duty_exempt: Whether the instrument is exempt at this timestamp.
            This flag affects stamp duty only; no security type is inferred.

    Returns:
        Independently rounded fee components for one transaction side.

    Raises:
        ValueError: Consideration is invalid or historical rates unsupported.
    """
    if not math.isfinite(notional) or notional < 0:
        raise ValueError("Trade notional must be finite and nonnegative")
    if notional == 0:
        return Fees()
    day = pd.Timestamp(timestamp).date().isoformat()
    if day < "2014-11-01":
        raise ValueError("Transaction fees before 2014-11-01 unsupported")
    stamp_rate = 0.0013 if "2021-08-01" <= day < "2023-11-17" else 0.001
    trading_rate = 0.0000565 if day >= "2023-01-01" else 0.00005
    return Fees(
        stamp_duty=0.0
        if stamp_duty_exempt
        else float(math.ceil(notional * stamp_rate - 1e-10)),
        sfc_levy=_cent(notional * 0.000027),
        afrc_levy=_cent(notional * 0.0000015) if day >= "2022-01-01" else 0,
        trading_fee=_cent(notional * trading_rate),
        trading_tariff=0.0 if day >= "2023-01-01" else 0.5,
        commission=_cent(
            max(
                config.minimum_commission,
                notional * config.commission_bps / 10_000,
            )
        ),
    )


def is_continuous_session(timestamp: pd.Timestamp) -> bool:
    """Return whether an open falls within a continuous trading session.

    Calendar holidays and half days are controlled by the supplied panel.
    Auction executions are outside this version's execution model.
    """
    current = pd.Timestamp(timestamp).time()
    return datetime.time(9, 30) <= current < datetime.time(12) or datetime.time(
        13
    ) <= current < datetime.time(16)


def within_window(timestamp: pd.Timestamp, start: str, end: str) -> bool:
    """Check an inclusive-start, exclusive-end local execution window."""
    current = pd.Timestamp(timestamp).time()
    return (
        datetime.time.fromisoformat(start)
        <= current
        < datetime.time.fromisoformat(end)
    )
