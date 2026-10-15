"""Small command-line entry points for research and data validation."""

import argparse
import dataclasses
import json
import logging
import pathlib
import sys
import tomllib

from hkfactor import batch
from hkfactor import config as backtest_config
from hkfactor import data as market_data
from hkfactor import demo as demo_data
from hkfactor import signal_adapter
from hkfactor import signals as factor_signals
from hkfactor import validation as data_validation


def _config(
    path: str,
) -> tuple[backtest_config.BacktestConfig, factor_signals.SignalColumns]:
    source = pathlib.Path(path)
    with source.open("rb") as handle:
        raw = (
            tomllib.load(handle)
            if source.suffix == ".toml"
            else json.load(handle)
        )
    values = raw.get("backtest", raw).copy()
    mapping = values.pop("signal_columns", raw.get("signal_columns", {}))
    return backtest_config.BacktestConfig(
        **values
    ), factor_signals.SignalColumns(**mapping)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hkfactor", description="Hong Kong factor backtesting"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Create synthetic input fixtures")
    demo.add_argument("--output", required=True)
    demo.add_argument("--frequency", choices=["1d", "5m", "1m"], default="1d")
    prepare = commands.add_parser("prepare", help="Validate and cache market")
    for name in ["prices", "benchmark", "constituents", "output"]:
        prepare.add_argument(f"--{name}", required=True)
    prepare.add_argument("--sessions")
    prepare.add_argument(
        "--frequency", choices=["1d", "5m", "1m"], default="1d"
    )
    prepare.add_argument("--t-start")
    prepare.add_argument("--t-end")
    prepare.add_argument("--max-panel-gib", type=float, default=4)
    run = commands.add_parser("run", help="Run independent factors in batches")
    for name in ["market", "signals", "config", "output"]:
        run.add_argument(f"--{name}", required=True)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--max-in-flight", type=int)
    run.add_argument("--factor", action="append", dest="factors")
    run.add_argument("--no-resume", action="store_true")
    run.add_argument("--no-plots", action="store_true")
    run.add_argument("--no-layer-execution", action="store_true")
    audit = commands.add_parser("audit", help="Describe local data gaps")
    audit.add_argument("--data", default="data")
    audit.add_argument("--output", required=True)
    convert = commands.add_parser("import-csv", help="Adapt raw CSV to Parquet")
    for name in ["input", "output", "symbol", "frequency"]:
        convert.add_argument(f"--{name}", required=True)
    convert.add_argument(
        "--kind", choices=["stock", "benchmark"], required=True
    )
    convert.add_argument(
        "--price-basis", choices=["raw", "adjusted"], default="adjusted"
    )
    signal_import = commands.add_parser(
        "import-signals", help="Convert per-factor wide Parquet to long signals"
    )
    signal_import.add_argument("--input", required=True)
    signal_import.add_argument("--output", required=True)
    signal_import.add_argument(
        "--timestamp-unit", choices=["s", "ms", "us", "ns"]
    )
    signal_import.add_argument(
        "--timestamp-label", choices=["bar_start", "bar_end"], required=True
    )
    signal_import.add_argument("--bar-duration")
    signal_import.add_argument("--publication-delay", default="0s")
    signal_import.add_argument("--source-timezone", default="UTC")
    signal_import.add_argument("--timestamp-column")
    signal_import.add_argument("--batch-size", type=int, default=65_536)
    compare = commands.add_parser("compare", help="Compare an API snapshot")
    for name in ["local", "reference", "output"]:
        compare.add_argument(f"--{name}", required=True)
    compare.add_argument("--frequency", default="1d")
    compare.add_argument("--atol", type=float, default=1e-8)
    compare.add_argument("--rtol", type=float, default=1e-6)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute one command and return a nonzero status for failed factors."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    try:
        if args.command == "demo":
            print(
                json.dumps(
                    demo_data.create_demo(args.output, args.frequency), indent=2
                )
            )
        elif args.command == "prepare":
            panel = market_data.prepare_market(
                args.prices,
                args.benchmark,
                args.constituents,
                args.output,
                args.frequency,
                args.sessions,
                args.t_start,
                args.t_end,
                int(args.max_panel_gib * 1024**3),
            )
            print(
                json.dumps(
                    {
                        "shape": panel.shape,
                        "frequency": panel.frequency,
                        "cache": args.output,
                    }
                )
            )
        elif args.command == "run":
            config, columns = _config(args.config)
            if args.no_layer_execution:
                config = dataclasses.replace(config, simulate_layers=False)
            summary = batch.run_batch(
                pathlib.Path(args.market),
                pathlib.Path(args.signals),
                pathlib.Path(args.output),
                config,
                workers=args.workers,
                max_in_flight=args.max_in_flight,
                resume=not args.no_resume,
                plots=not args.no_plots,
                columns=columns,
                factors=args.factors,
            )
            print(summary.to_string(index=False))
            return int(summary["status"].eq("failed").any())
        elif args.command == "audit":
            output = pathlib.Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            report = data_validation.audit_local_data(args.data)
            output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(output)
        elif args.command == "import-csv":
            data_validation.import_csv(
                args.input,
                args.output,
                args.frequency,
                args.symbol,
                args.price_basis,
                args.kind,
            )
            print(args.output)
        elif args.command == "import-signals":
            manifest = signal_adapter.import_wide_signals(
                args.input,
                args.output,
                timestamp_unit=args.timestamp_unit,
                timestamp_label=args.timestamp_label,
                bar_duration=args.bar_duration,
                publication_delay=args.publication_delay,
                source_timezone=args.source_timezone,
                timestamp_column=args.timestamp_column,
                batch_size=args.batch_size,
            )
            print(
                json.dumps(
                    {
                        "factors": len(manifest["factors"]),
                        "output_rows": manifest["output_rows"],
                        "manifest": str(
                            pathlib.Path(args.output) / "import_manifest.json"
                        ),
                        "availability_policy": manifest[
                            "assumed_availability_policy"
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.command == "compare":
            print(
                json.dumps(
                    data_validation.compare_snapshots(
                        args.local,
                        args.reference,
                        args.output,
                        args.frequency,
                        args.atol,
                        args.rtol,
                    ),
                    indent=2,
                )
            )
    except (
        ValueError,
        FileExistsError,
        FileNotFoundError,
        MemoryError,
        TypeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
