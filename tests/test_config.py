"""Reject research settings that could silently invalidate account results."""

import pytest

from hkfactor import config as backtest_config


@pytest.mark.parametrize(
    "overrides",
    [
        {"t_start": "2024-03-01", "t_end": "2024-01-01"},
        {"keep": 1.01},
        {"keep": -0.01},
        {"mode": "T0", "frequency": "1d"},
        {"initial_capital": float("nan")},
        {"slippage_bps": float("inf")},
        {"participation_rate": 0},
        {"entry_start": "11:30", "entry_end": "09:30"},
        {"order_slices": 0},
        {"order_slices": 1.5},
        {"holding_days": True},
        {"simulate_layers": "false"},
        {"t_start": 20240101},
    ],
)
def test_invalid_configuration(overrides: dict) -> None:
    """Invalid dates, risk bounds, times, and costs fail before data loading."""
    values = {"t_start": "2024-01-01", "t_end": "2024-03-01"}
    values.update(overrides)
    with pytest.raises(ValueError):
        backtest_config.BacktestConfig(**values)


def test_zero_and_full_base_are_valid() -> None:
    """Both edge allocations remain available for controlled experiments."""
    for keep in [0, 1]:
        config = backtest_config.BacktestConfig(
            "2024-01-01", "2024-01-31", keep=keep
        )
        assert config.to_dict()["keep"] == keep
