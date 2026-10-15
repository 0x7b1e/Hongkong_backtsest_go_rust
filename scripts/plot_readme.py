"""Draw English README charts from saved synthetic backtest accounts."""

import argparse
import json
import pathlib

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import dates as mdates  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib import ticker as mticker  # noqa: E402

BLUE = "#386CB0"
GRAY = "#8B929C"
AMBER = "#C8843B"
INK = "#283444"
MUTED = "#747E8D"


def _load_report(directory: pathlib.Path, factor: str) -> dict:
    """Read one successful factor's accounts and recorded configuration."""
    summary = pd.read_csv(directory / "summary.csv")
    selected = summary.loc[
        summary["factor"].eq(factor) & summary["status"].eq("success")
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one successful {factor} run: {directory}")
    report = pathlib.Path(selected["output_directory"].iloc[0])
    if not report.exists():
        raise FileNotFoundError(report)
    metadata = json.loads((report / "run.json").read_text(encoding="utf-8"))
    return {
        "pnl": pd.read_parquet(report / "pnl.parquet"),
        "layers": pd.read_parquet(report / "layer_pnl.parquet"),
        "ic": pd.read_parquet(report / "ic_intraday.parquet"),
        "config": metadata["config"],
        "factor": factor,
    }


def _daily(frame: pd.DataFrame, layers: bool = False) -> pd.DataFrame:
    """Keep each trading day's last account observation, in local time."""
    dates = (
        frame["timestamp"]
        .dt.tz_convert("Asia/Hong_Kong")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    labelled = frame.assign(date=dates)
    if layers:
        return (
            labelled.groupby(["date", "layer"])["net_nav"]
            .last()
            .unstack("layer")
        )
    return labelled.groupby("date")[["net_nav", "benchmark_nav"]].last()


def _style_axis(ax: plt.Axes, percent: bool = False) -> None:
    """Apply light grids and uncluttered axes."""
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E9EDF2", linewidth=0.7)
    ax.tick_params(axis="both", length=0, pad=7)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#D4DAE2")
    ax.axhline(0, color="#B8C0CB", linewidth=0.8, zorder=1)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    if percent:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1, decimals=1))


def _date_ticks(ax: plt.Axes, dates: pd.DatetimeIndex) -> None:
    """Label actual observations with compact English dates."""
    positions = np.unique(np.linspace(0, len(dates) - 1, 5).astype(int))
    ax.set_xticks(dates[positions])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.margins(x=0.025)
    ax.set_xlabel("Trading date", labelpad=9)


def _save(
    fig: plt.Figure,
    output: pathlib.Path,
    filename: str,
    title: str,
    subtitle: str,
    note: str,
) -> None:
    """Save a standalone chart with brief provenance and readable type."""
    fig.text(0.065, 0.955, title, va="top", fontsize=16, weight="bold")
    fig.text(0.065, 0.887, subtitle, va="top", fontsize=9.5, color=MUTED)
    fig.text(0.065, 0.035, note, fontsize=8, color=MUTED)
    fig.savefig(
        output / filename,
        dpi=180,
        metadata={"Description": "Synthetic backtest; English Matplotlib"},
    )
    plt.close(fig)


def _plot_accounts(reports: dict, output: pathlib.Path) -> None:
    """Show net account and benchmark returns for the two sample windows."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.subplots_adjust(
        left=0.065, right=0.975, bottom=0.19, top=0.74, wspace=0.24
    )
    for ax, mode in zip(axes, ("t0", "t1"), strict=True):
        report = reports[mode]
        daily = _daily(report["pnl"])
        capital = report["config"]["initial_capital"]
        ax.plot(
            daily.index,
            daily["net_nav"] / capital - 1,
            color=BLUE,
            linewidth=2.1,
            label="Strategy (net)",
        )
        ax.plot(
            daily.index,
            daily["benchmark_nav"] / capital - 1,
            color=GRAY,
            linewidth=1.7,
            linestyle=(0, (4, 3)),
            label="Synthetic benchmark",
        )
        frequency = "5-minute bars" if mode == "t0" else "Daily bars"
        ax.set_title(
            f"{mode.upper()}  /  {frequency}  /  {len(daily)} sessions",
            loc="left",
            fontsize=11,
            pad=14,
            weight="normal",
        )
        ax.set_ylabel("Cumulative return", labelpad=9)
        _style_axis(ax, percent=True)
        _date_ticks(ax, daily.index)
        ax.legend(frameon=False, fontsize=8, loc="best")
    config = reports["t0"]["config"]
    _save(
        fig,
        output,
        "demo_account_nav.png",
        "Strategy vs. benchmark",
        f"{reports['t0']['factor']}  /  keep = {config['keep']:.2f}  /  "
        f"Initial capital: HKD {config['initial_capital'] / 1e6:g}m",
        "Daily closes; returns relative to initial capital. "
        "Synthetic data, 20 stocks, seed 101. Sample windows differ.",
    )


def _plot_layers(report: dict, output: pathlib.Path, mode: str) -> None:
    """Pair ten ordered decile return curves with their final returns."""
    daily = _daily(report["layers"], layers=True)
    returns = daily / report["config"]["initial_capital"] - 1
    colors = plt.colormaps["viridis"](
        np.linspace(0.12, 0.82, len(returns.columns))
    )
    fig, (ax, terminal) = plt.subplots(
        1,
        2,
        figsize=(12, 4.8),
        gridspec_kw={"width_ratios": [3.6, 1], "wspace": 0.22},
    )
    fig.subplots_adjust(left=0.065, right=0.965, bottom=0.19, top=0.74)
    for color, layer in zip(colors, returns.columns, strict=True):
        endpoint = layer in (returns.columns[0], returns.columns[-1])
        ax.plot(
            returns.index,
            returns[layer],
            color=color,
            linewidth=2.1 if endpoint else 1.25,
            alpha=1 if endpoint else 0.78,
            label=f"D{layer}",
        )
    ax.set_ylabel("Cumulative net return", labelpad=9)
    _style_axis(ax, percent=True)
    _date_ticks(ax, returns.index)
    ax.legend(
        ncol=10,
        frameon=False,
        fontsize=7.5,
        loc="lower left",
        bbox_to_anchor=(-0.01, 1.025),
        columnspacing=0.9,
        handlelength=1.4,
        handletextpad=0.35,
    )
    final = returns.iloc[-1]
    positions = np.arange(len(final))
    terminal.barh(positions, final, color=colors, height=0.6, alpha=0.9)
    terminal.set_yticks(positions, [f"D{layer}" for layer in final.index])
    terminal.invert_yaxis()
    terminal.set_title("Final return", loc="left", fontsize=10.5, pad=14)
    terminal.axvline(0, color="#B8C0CB", linewidth=0.8)
    scale = max(float(final.abs().max()), 0.001)
    terminal.set_xlim(
        min(0, final.min()) - scale * 0.4, max(0, final.max()) + scale * 0.4
    )
    for position, value in enumerate(final):
        offset = scale * 0.045 * (1 if value >= 0 else -1)
        terminal.text(
            value + offset,
            position,
            f"{value:+.2%}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8,
            color=INK,
        )
    terminal.set_xticks([])
    terminal.tick_params(axis="y", length=0, pad=6, labelsize=8)
    for spine in terminal.spines.values():
        spine.set_visible(False)
    frequency = "5-minute bars" if mode == "t0" else "Daily bars"
    keep = report["config"]["keep"]
    _save(
        fig,
        output,
        f"demo_{mode}_layers.png",
        f"{mode.upper()} decile accounts",
        f"{report['factor']}  /  {frequency}  /  {len(daily)} sessions  /  "
        "D1: low signal  ->  D10: high signal",
        f"Target: {keep:.0%} base + {1 - keep:.0%} decile overlay. "
        "Costs and execution constraints included. Synthetic data, seed 101.",
    )


def _plot_ic(report: dict, output: pathlib.Path) -> None:
    """Show unsmoothed research IC without bridging the lunch break."""
    frame = report["ic"]
    minutes = np.array(
        [int(value[:2]) * 60 + int(value[3:]) for value in frame["local_time"]]
    )
    fig, ax = plt.subplots(figsize=(12, 4.8))
    fig.subplots_adjust(left=0.065, right=0.975, bottom=0.19, top=0.74)
    for column, color, label in (
        ("pearson_mean", BLUE, "Pearson IC"),
        ("spearman_mean", AMBER, "Spearman IC"),
    ):
        for index, mask in enumerate((minutes < 720, minutes >= 780)):
            ax.plot(
                minutes[mask],
                frame.loc[mask, column],
                color=color,
                linewidth=1.7,
                label=label if index == 0 else None,
            )
    _style_axis(ax)
    ax.axvspan(720, 780, color="#F2F4F7", zorder=0)
    ax.text(
        750,
        0.95,
        "Lunch break",
        ha="center",
        va="top",
        fontsize=8,
        color=MUTED,
        transform=ax.get_xaxis_transform(),
    )
    ticks = [570, 630, 690, 780, 840, 900, 960]
    ax.set_xticks(ticks, [f"{v // 60:02d}:{v % 60:02d}" for v in ticks])
    ax.set_xlim(565, 960)
    ax.set_ylabel("Mean IC", labelpad=9)
    ax.set_xlabel("Entry time (Hong Kong)", labelpad=9)
    ax.legend(
        ncol=2,
        frameon=False,
        fontsize=8,
        loc="lower left",
        bbox_to_anchor=(-0.005, 1.025),
    )
    counts = frame["pearson_count"]
    _save(
        fig,
        output,
        "demo_t0_ic.png",
        "Intraday information coefficient",
        f"{report['factor']}  /  5-minute bars  /  "
        f"{counts.min()}-{counts.max()} observations per time slot",
        "Prior completed-bar signal vs. next open-to-close return. "
        "No overnight windows. Synthetic data, seed 101.",
    )


def main() -> None:
    """Render all README figures from two completed synthetic runs.

    The inputs are task directories containing summary.csv. Their paths
    must still resolve to the factor reports created by hkfactor run.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t0", type=pathlib.Path, required=True)
    parser.add_argument("--t1", type=pathlib.Path, required=True)
    parser.add_argument("--factor", default="momentum_5")
    parser.add_argument("--output", type=pathlib.Path, default="docs/images")
    args = parser.parse_args()
    reports = {
        "t0": _load_report(args.t0, args.factor),
        "t1": _load_report(args.t1, args.factor),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    ):
        _plot_accounts(reports, args.output)
        for mode, report in reports.items():
            _plot_layers(report, args.output, mode)
        _plot_ic(reports["t0"], args.output)
    print(f"Saved four English charts to {args.output}")


if __name__ == "__main__":
    main()
