"""Historical fee boundaries, rounding and local session checks."""

import pandas as pd
import pytest

from hkfactor import config as backtest_config
from hkfactor import rules


@pytest.mark.parametrize(
    "day,stamp,afrc,trading,tariff",
    [
        ("2021-07-31", 100, 0, 5, 0.5),
        ("2021-08-01", 130, 0, 5, 0.5),
        ("2022-01-01", 130, 0.15, 5, 0.5),
        ("2023-01-01", 130, 0.15, 5.65, 0),
        ("2023-11-16", 130, 0.15, 5.65, 0),
        ("2023-11-17", 100, 0.15, 5.65, 0),
    ],
)
def test_historical_fee_boundaries(day, stamp, afrc, trading, tariff):
    """Charge the correct rate on each effective-date boundary."""
    config = backtest_config.BacktestConfig(day, day, commission_bps=0)
    fees = rules.transaction_fees(100_000, pd.Timestamp(day), config)
    assert fees.stamp_duty == stamp
    assert fees.sfc_levy == 2.7
    assert fees.afrc_levy == afrc
    assert fees.trading_fee == trading
    assert fees.trading_tariff == tariff
    assert fees.total == pytest.approx(stamp + 2.7 + afrc + trading + tariff)


def test_round_components_and_stamp_up():
    """Use commercial cents rather than bankers rounding or one total."""
    config = backtest_config.BacktestConfig(
        "2021-07-01", "2021-07-01", commission_bps=0, minimum_commission=5.555
    )
    fees = rules.transaction_fees(100, pd.Timestamp("2021-07-01"), config)
    assert fees.stamp_duty == 1
    assert fees.trading_fee == 0.01
    assert fees.commission == 5.56
    assert (
        rules.transaction_fees(0, pd.Timestamp("2021-07-01"), config).total == 0
    )


def test_historical_stamp_exemption_changes_only_stamp_charge():
    """An explicit exemption must preserve every non-stamp transaction fee."""
    config = backtest_config.BacktestConfig("2024-12-23", "2024-12-23")
    timestamp = pd.Timestamp("2024-12-23")
    taxable = rules.transaction_fees(
        100_000, timestamp, config, stamp_duty_exempt=False
    )
    exempt = rules.transaction_fees(
        100_000, timestamp, config, stamp_duty_exempt=True
    )
    assert taxable.stamp_duty == 100
    assert exempt.stamp_duty == 0
    assert exempt.sfc_levy == taxable.sfc_levy == 2.7
    assert exempt.afrc_levy == taxable.afrc_levy == 0.15
    assert exempt.trading_fee == taxable.trading_fee == 5.65
    assert exempt.commission == taxable.commission == 20
    assert exempt.total == pytest.approx(28.5)
    assert taxable.total == pytest.approx(128.5)


def test_reject_unsupported_history():
    """Do not silently backfill contemporary rates into older history."""
    config = backtest_config.BacktestConfig("2014-10-31", "2014-10-31")
    with pytest.raises(ValueError, match="unsupported"):
        rules.transaction_fees(1000, pd.Timestamp("2014-10-31"), config)


@pytest.mark.parametrize(
    "clock,expected",
    [
        ("09:29", False),
        ("09:30", True),
        ("11:59", True),
        ("12:00", False),
        ("12:59", False),
        ("13:00", True),
        ("15:59", True),
        ("16:00", False),
    ],
)
def test_continuous_sessions(clock, expected):
    """Exclude lunch, pre-open and closing-auction execution times."""
    timestamp = pd.Timestamp(f"2024-01-02 {clock}", tz="Asia/Hong_Kong")
    assert rules.is_continuous_session(timestamp) is expected
