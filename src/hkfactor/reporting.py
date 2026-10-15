"""Persist account results and explicitly labelled research diagnostics."""

import datetime
import json
import pathlib

import matplotlib
import numpy as np
import pandas as pd

from hkfactor import models

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def _json_safe(value: object) -> object:
    """Convert scalar metadata to strict JSON without non-finite numbers."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (datetime.date, datetime.datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def _save_json(path: pathlib.Path, values: dict) -> None:
    """Write readable UTF-8 metadata with strict JSON scalars."""
    path.write_text(
        json.dumps(
            _json_safe(values), ensure_ascii=False, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )


def _plot_layers(frame: pd.DataFrame, output_dir: pathlib.Path) -> None:
    """Plot separately executed layer account NAV by local trading date."""
    fig, ax = plt.subplots(figsize=(11, 5))
    if frame.empty:
        ax.text(
            0.5,
            0.5,
            "No executed layer accounts",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        nav_column = "net_nav" if "net_nav" in frame else "nav"
        if nav_column not in frame:
            raise ValueError("Layer account PnL requires a nav column")
        dates = pd.to_datetime(frame["timestamp"])
        if dates.dt.tz is not None:
            dates = dates.dt.tz_convert("Asia/Hong_Kong")
        daily = frame.assign(date=dates.dt.strftime("%Y-%m-%d"))
        daily = (
            daily.groupby(["date", "layer"], sort=True)[nav_column]
            .last()
            .unstack("layer")
        )
        for layer in daily:
            ax.plot(
                pd.to_datetime(daily.index),
                daily[layer],
                label=f"Layer {layer}",
                marker="o",
                markersize=2,
            )
        if len(daily) == 1:
            only_date = pd.Timestamp(daily.index[0])
            ax.set_xlim(
                only_date - pd.Timedelta(hours=12),
                only_date + pd.Timedelta(hours=12),
            )
            ax.set_xticks([only_date], [daily.index[0]])
        ax.legend(ncol=5, fontsize=8)
    ax.set(
        title="Executed layer accounts: net account value",
        xlabel="Hong Kong trading date",
        ylabel="Net NAV (HKD)",
    )
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "layer_pnl.png", dpi=150)
    plt.close(fig)


def _plot_intraday(frame: pd.DataFrame, output_dir: pathlib.Path) -> None:
    """Plot local-time average IC, separately from strategy account PnL."""
    fig, ax = plt.subplots(figsize=(11, 5))
    if frame.empty:
        ax.text(
            0.5,
            0.5,
            "No usable IC observations",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        positions = np.arange(len(frame))
        ax.plot(
            positions, frame["pearson_mean"], label="Pearson IC", marker="."
        )
        ax.plot(
            positions, frame["spearman_mean"], label="Spearman IC", marker="."
        )
        stride = max(1, (len(frame) + 15) // 16)
        ax.set_xticks(
            positions[::stride], frame["local_time"].iloc[::stride], rotation=45
        )
        ax.legend()
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set(
        title="Research IC by intraday entry time",
        xlabel="Asia/Hong_Kong local time",
        ylabel="Mean IC",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "ic_intraday.png", dpi=150)
    plt.close(fig)


def write_report(
    result: models.BacktestResult,
    analysis: models.AnalysisResult,
    output_dir: pathlib.Path,
    metadata: dict | None = None,
    plots: bool = True,
) -> None:
    """Write machine-readable results and optional research/account plots.

    The batch runner supplies a staging directory and atomically commits
    it after this function completes. IC and research forward layers are
    gross diagnostics; ``pnl`` and ``layer_pnl`` are engine accounts that
    include configured execution costs and constraints.

    Args:
        result: Executed enhancement account outputs.
        analysis: Research diagnostics and optional executed layer PnL.
        output_dir: Empty or staging destination for this experiment.
        metadata: Factor identity, configuration, and input provenance.
        plots: Whether to write PNG charts using a headless backend.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "pnl": result.pnl,
        "layers": analysis.layers,
        "ic": analysis.ic,
        "ic_intraday": analysis.ic_intraday,
        "layer_summary": analysis.layer_summary,
    }
    if not result.audit_streamed:
        frames["trades"] = result.trades
        frames["restrictions"] = result.restrictions
    layer_pnl = analysis.layer_pnl
    if layer_pnl is not None:
        frames["layer_pnl"] = layer_pnl
    for name, frame in frames.items():
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)
    _save_json(output_dir / "metrics.json", result.metrics)
    run = dict(metadata or {})
    run["analysis_assumptions"] = analysis.ic.attrs.get("assumptions", {})
    run["outputs"] = {
        "pnl": "Executed account; configured costs and constraints",
        "layers": "Gross forward research returns, not executed PnL",
        "ic": "Previous-bar signals versus next-open forward returns",
        "layer_pnl": "Independent executed accounts; same keep and costs"
        if layer_pnl is not None
        else "Not requested",
    }
    _save_json(output_dir / "run.json", run)
    if plots:
        _plot_intraday(analysis.ic_intraday, output_dir)
        if layer_pnl is not None:
            _plot_layers(layer_pnl, output_dir)
