"""Check that the Rust port reproduces the Python implementation.

Runs ``prepare`` and ``run`` with both implementations on the same inputs and
compares every panel array and every published report table, metric and
audit row. Floats must agree to a relative tolerance (default 1e-9); counts,
timestamps, identifiers and reasons must match exactly.

Usage:
    python parity/compare.py --rust rust/target/release/hkfactor \
        --case examples/daily --case examples/intraday --work /tmp/parity
"""

import argparse
import json
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ARRAYS = (
    "open high low close volume base_weights benchmark_close lot_sizes "
    "tradable vcm_lower vcm_upper vcm_eligible cash_dividend share_multiplier "
    "session_id is_session_start is_session_end vcm_active stamp_duty_exempt "
    "benchmark_open timestamps open_timestamps"
).split()


class Mismatch(Exception):
    """A difference between the two implementations."""


def _run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"{' '.join(command)} failed ({result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return result.stdout


def _close(
    left: np.ndarray, right: np.ndarray, rtol: float, atol: float = 1e-12
) -> np.ndarray:
    if left.dtype.kind in "fc" or right.dtype.kind in "fc":
        left = left.astype(float)
        right = right.astype(float)
        return np.isclose(left, right, rtol=rtol, atol=atol, equal_nan=True)
    return left == right


def compare_arrays(name: str, left, right, rtol: float, atol=1e-12) -> None:
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape:
        raise Mismatch(f"{name}: shape {left.shape} != {right.shape}")
    equal = _close(left, right, rtol, atol)
    if not np.all(equal):
        where = np.argwhere(~equal)[:5]
        examples = [
            (tuple(int(v) for v in w), left[tuple(w)], right[tuple(w)])
            for w in where
        ]
        raise Mismatch(
            f"{name}: {int((~equal).sum())} differing values, e.g. {examples}"
        )


def compare_frames(name: str, left: pd.DataFrame, right: pd.DataFrame, rtol):
    if list(left.columns) != list(right.columns):
        raise Mismatch(f"{name}: columns {list(left.columns)} != {list(right.columns)}")
    if len(left) != len(right):
        raise Mismatch(f"{name}: {len(left)} rows != {len(right)} rows")
    for column in left.columns:
        a, b = left[column], right[column]
        if pd.api.types.is_datetime64_any_dtype(a):
            a = a.dt.tz_convert("UTC").dt.as_unit("ns").astype("int64")
            b = pd.to_datetime(b).dt.tz_convert("UTC").dt.as_unit("ns").astype("int64")
        if a.dtype == object or b.dtype == object:
            if not a.fillna("<null>").astype(str).equals(
                b.fillna("<null>").astype(str)
            ):
                diff = a.fillna("<null>").astype(str) != b.fillna("<null>").astype(str)
                raise Mismatch(
                    f"{name}.{column}: {int(diff.sum())} differing values, e.g. "
                    f"{list(zip(a[diff][:3], b[diff][:3]))}"
                )
            continue
        # PnL columns are differences of ~1e7 HKD account values, so one
        # unit in the last place of a NAV (~2e-9 HKD) is the natural floor;
        # share quantities such as floor shortfalls are differences of
        # holdings, compared to a millionth of a share.
        atol = 1e-6 if column.endswith("pnl") or column in ("requested", "filled") else 1e-12
        compare_arrays(f"{name}.{column}", a.to_numpy(), b.to_numpy(), rtol, atol)


def compare_json(name: str, left, right, rtol: float) -> None:
    if isinstance(left, dict):
        if list(left) != list(right):
            raise Mismatch(f"{name}: keys {list(left)} != {list(right)}")
        for key in left:
            compare_json(f"{name}.{key}", left[key], right[key], rtol)
    elif isinstance(left, float) or isinstance(right, float):
        if left is None or right is None:
            if left is not right:
                raise Mismatch(f"{name}: {left!r} != {right!r}")
            return
        atol = 1e-6 if name.endswith("pnl") else 1e-12
        if not np.isclose(left, right, rtol=rtol, atol=atol, equal_nan=True):
            raise Mismatch(f"{name}: {left!r} != {right!r}")
    elif left != right:
        raise Mismatch(f"{name}: {left!r} != {right!r}")


def compare_panels(left: pathlib.Path, right: pathlib.Path, rtol: float) -> int:
    for name in ARRAYS:
        compare_arrays(
            f"panel.{name}",
            np.load(left / f"{name}.npy"),
            np.load(right / f"{name}.npy"),
            rtol=0.0,
        )
    meta_left = json.loads((left / "metadata.json").read_text())
    meta_right = json.loads((right / "metadata.json").read_text())
    for key in ("version", "symbols", "frequency", "price_basis", "shape", "timezone"):
        compare_json(f"metadata.{key}", meta_left[key], meta_right[key], rtol)
    return len(ARRAYS)


def _factor_dirs(output: pathlib.Path) -> dict[str, pathlib.Path]:
    summary = pd.read_csv(output / "summary.csv", encoding="utf-8-sig")
    failed = summary.loc[summary["status"] != "success"]
    if len(failed):
        raise Mismatch(f"{output}: failed factors {failed.to_dict('records')}")
    return {
        row.factor: pathlib.Path(row.output_directory)
        for row in summary.itertuples()
    }


def compare_runs(left: pathlib.Path, right: pathlib.Path, rtol: float) -> int:
    checked = 0
    left_dirs, right_dirs = _factor_dirs(left), _factor_dirs(right)
    if sorted(left_dirs) != sorted(right_dirs):
        raise Mismatch(f"factors {sorted(left_dirs)} != {sorted(right_dirs)}")
    summary_left = pd.read_csv(left / "summary.csv", encoding="utf-8-sig")
    summary_right = pd.read_csv(right / "summary.csv", encoding="utf-8-sig")
    shared = [
        c for c in summary_left.columns
        if c not in ("output_directory", "run_id")
    ]
    compare_frames("summary", summary_left[shared], summary_right[shared], rtol)
    for factor, directory in left_dirs.items():
        other = right_dirs[factor]
        files = sorted(
            p.relative_to(directory)
            for p in directory.rglob("*")
            if p.is_file() and p.suffix in {".parquet", ".json"}
            and p.name not in {"completed.json", "run.json"}
        )
        files_other = sorted(
            p.relative_to(other)
            for p in other.rglob("*")
            if p.is_file() and p.suffix in {".parquet", ".json"}
            and p.name not in {"completed.json", "run.json"}
        )
        if files != files_other:
            raise Mismatch(f"{factor}: artifacts {files} != {files_other}")
        for relative in files:
            label = f"{factor}/{relative}"
            if relative.suffix == ".parquet":
                compare_frames(
                    label,
                    pd.read_parquet(directory / relative),
                    pd.read_parquet(other / relative),
                    rtol,
                )
            else:
                compare_json(
                    label,
                    json.loads((directory / relative).read_text()),
                    json.loads((other / relative).read_text()),
                    rtol,
                )
            checked += 1
        run_left = json.loads((directory / "run.json").read_text())
        run_right = json.loads((other / "run.json").read_text())
        for key in ("factor", "config", "analysis_assumptions", "outputs"):
            compare_json(f"{factor}/run.json.{key}", run_left[key], run_right[key], rtol)
        checked += 1
    return checked


def check_case(rust: str, case: pathlib.Path, work: pathlib.Path, rtol: float):
    config = json.loads((case / "config.json").read_text())
    frequency = config["frequency"]
    outputs = {}
    prepared = case / "market"
    for name, command in (
        ()
        if prepared.is_dir()
        else (("python", [sys.executable, "-m", "hkfactor"]), ("rust", [rust]))
    ):
        market = work / f"{name}-market"
        _run(
            command
            + [
                "prepare",
                "--prices", str(case / "prices.parquet"),
                "--benchmark", str(case / "benchmark.parquet"),
                "--constituents", str(case / "constituents.parquet"),
                "--sessions", str(case / "sessions.parquet"),
                "--frequency", frequency,
                "--t-start", config["t_start"],
                "--t-end", config["t_end"],
                "--output", str(market),
            ]
        )
        outputs[name] = market
    if prepared.is_dir():
        # A hand-edited panel shared by both engines.
        outputs = {"python": prepared, "rust": prepared}
        arrays = 0
    else:
        arrays = compare_panels(outputs["python"], outputs["rust"], rtol)
    # Both engines run on the Python-prepared panel so any difference is in
    # the engine itself, then Rust also runs on its own panel.
    configs = [case / "config.json"]
    if (case / "variants").is_dir():
        configs += sorted((case / "variants").glob("*.json"))
    checked = 0
    signal_sets = [case / "signals"] + sorted(case.glob("signals_*"))
    jobs = [(c, s) for s in signal_sets for c in (configs if s.name == "signals" else configs[:1])]
    for config_path, signal_path in jobs:
        runs = {}
        for name, command, market in (
            ("python", [sys.executable, "-m", "hkfactor"], outputs["python"]),
            ("rust", [rust], outputs["python"]),
            ("rust-own", [rust], outputs["rust"]),
        ):
            destination = work / f"{name}-{signal_path.name}-{config_path.stem}-run"
            _run(
                command
                + [
                    "run",
                    "--market", str(market),
                    "--signals", str(signal_path),
                    "--config", str(config_path),
                    "--output", str(destination),
                    "--no-plots",
                ]
            )
            runs[name] = destination
        try:
            checked += compare_runs(runs["python"], runs["rust"], rtol)
            checked += compare_runs(runs["python"], runs["rust-own"], rtol)
        except Mismatch as error:
            raise Mismatch(f"[{signal_path.name}/{config_path.name}] {error}") from error
    return arrays, checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust", required=True)
    parser.add_argument("--case", action="append", type=pathlib.Path, required=True)
    parser.add_argument("--work", type=pathlib.Path, required=True)
    parser.add_argument("--rtol", type=float, default=1e-9)
    args = parser.parse_args()
    failures = 0
    for case in args.case:
        work = args.work / case.name
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        try:
            arrays, checked = check_case(args.rust, case, work, args.rtol)
        except Mismatch as error:
            failures += 1
            print(f"FAIL {case}: {error}")
        else:
            print(f"OK   {case}: {arrays} panel arrays, {checked} report files identical")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
