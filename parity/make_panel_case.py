"""Build a hand-edited market panel reaching engine-only branches.

Prepared panels never contain a tradable bar without an open or volume, an
open during the lunch break, a constituent with no quote at all, or an
active VCM event without bands. This script prepares the intraday example
with the Python implementation, injects those states directly into the
saved arrays, and writes a case directory (``market/``, ``signals/`` and
configurations) that both engines run unchanged.

Usage:
    python parity/make_panel_case.py --output /tmp/panel_case
"""

import argparse
import dataclasses
import json
import pathlib
import shutil
import tempfile

import numpy as np
import pandas as pd

from hkfactor import data

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    source = ROOT / "examples" / "intraday"
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    config = json.loads((source / "config.json").read_text())
    with tempfile.TemporaryDirectory() as temporary:
        panel = data.prepare_market(
            source / "prices.parquet",
            source / "benchmark.parquet",
            source / "constituents.parquet",
            pathlib.Path(temporary) / "market",
            frequency=config["frequency"],
            sessions=source / "sessions.parquet",
            t_start=config["t_start"],
            t_end=config["t_end"],
        )
        arrays = {
            field.name: np.array(getattr(panel, field.name))
            for field in dataclasses.fields(panel)
            if isinstance(getattr(panel, field.name), np.ndarray)
        }
        rows, cols = panel.shape
        # A tradable bar without an executable open, and one without volume.
        arrays["open"][100:104, 3] = np.nan
        arrays["volume"][150:160, 5] = 0
        # An active VCM event without bands, and an active event with bands
        # that exclude the open price.
        arrays["vcm_active"][200:206, 7] = 1
        arrays["vcm_active"][210:214, 8] = 1
        arrays["vcm_lower"][210:214, 8] = arrays["open"][210:214, 8] * 1.2
        arrays["vcm_upper"][210:214, 8] = arrays["open"][210:214, 8] * 1.3
        # A constituent that never trades or quotes.
        arrays["open"][:, 11] = np.nan
        arrays["close"][:, 11] = np.nan
        arrays["tradable"][:, 11] = False
        # Stamp-duty exemption and a missing benchmark close (forward-filled).
        arrays["stamp_duty_exempt"][:, 12] = True
        arrays["benchmark_close"][300] = np.nan
        arrays["benchmark_open"] = arrays["benchmark_close"].copy()
        arrays["benchmark_open"][0] = arrays["benchmark_close"][0] * 0.998
        arrays["benchmark_open"][300] = arrays["benchmark_close"][299]
        opens = panel.open_timestamps.copy()
        # One bar opening inside the lunch break.
        lunch = np.flatnonzero(
            (opens.hour == 11) & (opens.minute == 55)
        )[2]
        opens = opens.insert(lunch, opens[lunch] + pd.Timedelta(minutes=40)).delete(lunch + 1)
        edited = dataclasses.replace(
            panel,
            open_timestamps=opens,
            **{name: arrays[name] for name in arrays},
        )
        edited.save(args.output / "market")
    shutil.copytree(source / "signals", args.output / "signals")
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    variants = args.output / "variants"
    variants.mkdir()
    for name, values in {
        "tight": {"keep": 0.2, "order_slices": 2, "participation_rate": 0.005},
        "costly": {"slippage_bps": 30.0, "minimum_commission": 25.0},
    }.items():
        (variants / f"{name}.json").write_text(
            json.dumps({**config, **values}, indent=2) + "\n"
        )
    print(args.output)


if __name__ == "__main__":
    main()
