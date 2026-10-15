"""Bounded, resumable factor jobs over a memory-mapped market panel."""

import dataclasses
import hashlib
import json
import logging
import multiprocessing
import os
import pathlib
import shutil
import tempfile
from concurrent import futures as concurrent_futures
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from hkfactor import config as backtest_config
from hkfactor import models
from hkfactor import signals as factor_signals

_CACHE_VERSION = 2
_LOGGER = logging.getLogger(__name__)
_WORKER_PANEL: models.MarketPanel | None = None
_WORKER_STORE: factor_signals.SignalStore | None = None
_WORKER_CONFIG: backtest_config.BacktestConfig | None = None
_WORKER_RUN: pathlib.Path | None = None
_WORKER_IDENTITY: dict | None = None
_WORKER_PLOTS = True


def _json_default(value: Any) -> Any:
    """Convert scalar numpy metrics into portable JSON values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _metric_scalars(metrics: dict) -> dict:
    """Represent unavailable scalar metrics as strict JSON null values."""
    converted = {}
    for name, value in metrics.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            value = None
        converted[name] = value
    return converted


def _dump_json(path: pathlib.Path, value: Any) -> None:
    """Persist JSON, used only inside unpublished output directories."""
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            default=_json_default,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def _hash(value: Any) -> str:
    """Return an unambiguous hash of a canonical JSON object."""
    encoded = json.dumps(
        value, sort_keys=True, default=_json_default, allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _signal_identity(path: pathlib.Path) -> dict:
    """Fingerprint Parquet file identity without reading every factor score."""
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(item for item in path.rglob("*") if item.is_file())
    else:
        raise FileNotFoundError(path)
    if not files:
        raise ValueError("Signal path contains no files")
    records = []
    for item in files:
        stat = item.stat()
        records.append(
            {
                "path": str(item.relative_to(path))
                if path.is_dir()
                else item.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return {
        "path": str(path.resolve()),
        "file_count": len(records),
        "stat_fingerprint": _hash(records),
    }


def _implementation_fingerprint() -> str:
    """Identify package source bytes without depending on worker count."""
    package = pathlib.Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for source in sorted(package.rglob("*.py")):
        name = source.relative_to(package).as_posix().encode("utf-8")
        content = source.read_bytes()
        for part in (name, content):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


def _factor_directory(run_dir: pathlib.Path, factor: str) -> pathlib.Path:
    """Use a full digest so external factor names never become paths."""
    return run_dir / "factors" / hashlib.sha256(factor.encode()).hexdigest()


def _failed_summary(
    factor: str, run_dir: pathlib.Path, error: Exception
) -> dict:
    """Describe failed jobs uniformly, including worker-pool failures."""
    return {
        "factor": factor,
        "status": "failed",
        "error": f"{type(error).__name__}: {error}",
        "output_directory": str(_factor_directory(run_dir, factor)),
        "run_id": run_dir.name,
    }


def _cached_summary(
    directory: pathlib.Path,
    factor: str,
    identity: dict,
) -> dict | None:
    """Accept only a completed output with an identical input contract."""
    try:
        saved = json.loads(
            (directory / "completed.json").read_text(encoding="utf-8")
        )
        if saved["identity"] != identity or saved["factor"] != factor:
            return None
        summary = saved["summary"]
        artifacts = saved["artifacts"]
        if summary["status"] != "success" or not artifacts:
            return None
        for artifact in artifacts:
            relative = pathlib.Path(artifact["path"])
            if relative.is_absolute() or ".." in relative.parts:
                return None
            file = directory / relative
            if not file.is_file() or file.stat().st_size != artifact["size"]:
                return None
        return {**summary, "output_directory": str(directory)}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _initialize_worker(
    market_dir: str,
    signal_path: str,
    config: backtest_config.BacktestConfig,
    run_dir: str,
    identity: dict,
    plots: bool,
    columns: factor_signals.SignalColumns,
) -> None:
    """Open immutable market arrays and one Parquet dataset once per worker."""
    global _WORKER_PANEL, _WORKER_STORE, _WORKER_CONFIG
    global _WORKER_RUN, _WORKER_IDENTITY, _WORKER_PLOTS
    _WORKER_PANEL = models.MarketPanel.load(market_dir)
    _WORKER_STORE = factor_signals.SignalStore(signal_path, columns)
    _WORKER_CONFIG = config
    _WORKER_RUN = pathlib.Path(run_dir)
    _WORKER_IDENTITY = identity
    _WORKER_PLOTS = plots


def _run_factor(factor: str) -> dict:
    """Execute and atomically publish one independent factor experiment."""
    from hkfactor import analysis as factor_analysis
    from hkfactor import execution
    from hkfactor import reporting

    if any(
        value is None
        for value in (
            _WORKER_PANEL,
            _WORKER_STORE,
            _WORKER_CONFIG,
            _WORKER_RUN,
            _WORKER_IDENTITY,
        )
    ):
        raise RuntimeError("Worker has not been initialized")
    panel = _WORKER_PANEL
    store = _WORKER_STORE
    config = _WORKER_CONFIG
    run_dir = _WORKER_RUN
    identity = _WORKER_IDENTITY
    directory = _factor_directory(run_dir, factor)
    staging = pathlib.Path(tempfile.mkdtemp(prefix=".staging-", dir=run_dir))
    try:
        signal = store.read_factor(factor, panel, scratch_dir=staging)
        result = execution.run_backtest(
            panel, signal, config, audit_dir=staging
        )
        analysis = factor_analysis.analyze_factor(panel, signal, config)
        if config.simulate_layers:
            layer_accounts = []
            for layer, layer_signal in factor_analysis.iter_layer_signals(
                panel, signal, config
            ):
                layer_dir = staging / "layers" / str(layer)
                layer_result = execution.run_backtest(
                    panel, layer_signal, config, audit_dir=layer_dir
                )
                _dump_json(
                    layer_dir / "metrics.json",
                    _metric_scalars(layer_result.metrics),
                )
                layer_accounts.append(layer_result.pnl.assign(layer=layer))
                del layer_result, layer_signal
            analysis.layer_pnl = (
                pd.concat(layer_accounts, ignore_index=True)
                if layer_accounts
                else pd.DataFrame()
            )
        reporting.write_report(
            result,
            analysis,
            staging,
            metadata={
                "factor": factor,
                "run_id": run_dir.name,
                "input_identity": identity,
                "config": config.to_dict(),
            },
            plots=_WORKER_PLOTS,
        )
        summary = {
            "factor": factor,
            **_metric_scalars(result.metrics),
            "status": "success",
            "error": "",
            "output_directory": str(directory),
            "run_id": run_dir.name,
        }
        artifacts = [
            {
                "path": str(file.relative_to(staging)),
                "size": file.stat().st_size,
            }
            for file in sorted(staging.rglob("*"))
            if file.is_file()
        ]
        _dump_json(
            staging / "completed.json",
            {
                "factor": factor,
                "identity": identity,
                "summary": summary,
                "artifacts": artifacts,
            },
        )
        if directory.exists():
            backup = staging.with_name(staging.name + "-previous")
            directory.rename(backup)
            try:
                staging.rename(directory)
            except BaseException:
                backup.rename(directory)
                raise
            shutil.rmtree(backup)
        else:
            staging.rename(directory)
        return summary
    except Exception as error:
        shutil.rmtree(staging, ignore_errors=True)
        return _failed_summary(factor, run_dir, error)


def _write_summary(output_dir: pathlib.Path, rows: list[dict]) -> pd.DataFrame:
    """Publish the current requested-factor summary with atomic replacement."""
    columns = ["factor", "status", "error", "output_directory", "run_id"]
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=columns)
    frame.sort_values("factor", kind="stable", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    extra = sorted(name for name in frame.columns if name not in columns)
    frame = frame.reindex(columns=columns + extra)
    descriptor, filename = tempfile.mkstemp(
        prefix=".summary-", suffix=".csv", dir=output_dir
    )
    os.close(descriptor)
    temporary = pathlib.Path(filename)
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(output_dir / "summary.csv")
    finally:
        temporary.unlink(missing_ok=True)
    return frame


def run_batch(
    market_dir: pathlib.Path,
    signal_path: pathlib.Path,
    output_dir: pathlib.Path,
    config: backtest_config.BacktestConfig,
    workers: int = 1,
    max_in_flight: int | None = None,
    resume: bool = True,
    plots: bool = True,
    columns: factor_signals.SignalColumns | None = None,
    factors: list[str] | None = None,
) -> pd.DataFrame:
    """Run factors with bounded submissions, failure isolation, and resume.

    Args:
        market_dir: Saved immutable MarketPanel directory.
        signal_path: Long-form Parquet file or hive partition directory.
        output_dir: Parent directory for manifests and factor reports.
        config: Explicit execution and diagnostic assumptions.
        workers: Maximum independent processes, each sharing market memmaps.
        max_in_flight: Maximum submitted unfinished jobs; defaults to twice
            workers. Signal matrices never enter the parent process.
        resume: Reuse only complete reports matching all input identities.
        plots: Whether to render per-factor plots; disable for large screens.
        columns: Optional vendor column mapping.
        factors: Optional subset; otherwise discover names without scores.

    Returns:
        Stable factor-ordered metrics with explicit success or failed status.
        Failed factors retain their exception details in the run manifest.

    Raises:
        ValueError: Invalid concurrency, factor names, or shared input schema.
    """
    from hkfactor import data as market_data

    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    limit = max_in_flight if max_in_flight is not None else workers * 2
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("max_in_flight must be a positive integer")
    market_dir, signal_path, output_dir = (
        pathlib.Path(market_dir),
        pathlib.Path(signal_path),
        pathlib.Path(output_dir),
    )
    mapping = columns or factor_signals.SignalColumns()
    store = factor_signals.SignalStore(signal_path, mapping)
    requested = factors if factors is not None else store.factors()
    if any(not isinstance(name, str) or not name for name in requested):
        raise ValueError("Factor identifiers must be nonempty strings")
    if len(set(requested)) != len(requested):
        raise ValueError("Requested factor identifiers must be unique")
    requested = sorted(requested)
    identity = {
        "cache_version": _CACHE_VERSION,
        "implementation_sha256": _implementation_fingerprint(),
        "library_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
        },
        "config": config.to_dict(),
        "market": market_data.panel_fingerprint(market_dir),
        "signals": _signal_identity(signal_path),
        "columns": dataclasses.asdict(mapping),
        "plots": plots,
    }
    run_dir = output_dir / "runs" / _hash(identity)
    (run_dir / "factors").mkdir(parents=True, exist_ok=True)
    _dump_json(run_dir / "identity.json", identity)
    rows: list[dict] = []
    pending = []
    for factor in requested:
        cached = (
            _cached_summary(
                _factor_directory(run_dir, factor), factor, identity
            )
            if resume
            else None
        )
        if cached is None:
            pending.append(factor)
        else:
            rows.append(cached)
    _LOGGER.info(
        "Factors: %d total, %d cached, %d pending; workers=%d, in-flight=%d",
        len(requested),
        len(rows),
        len(pending),
        workers,
        limit,
    )
    initializer_args = (
        str(market_dir),
        str(signal_path),
        config,
        str(run_dir),
        identity,
        plots,
        mapping,
    )
    with (run_dir / "manifest.jsonl").open("a", encoding="utf-8") as manifest:

        def record(row: dict) -> None:
            """Flush completions so failures survive later interruptions."""
            manifest.write(
                json.dumps(row, default=_json_default, allow_nan=False) + "\n"
            )
            manifest.flush()
            rows.append(row)
            _LOGGER.info(
                "Completed %d/%d factors: %s (%s)",
                len(rows),
                len(requested),
                row["factor"],
                row["status"],
            )

        if workers == 1 and pending:
            _initialize_worker(*initializer_args)
            for factor in pending:
                record(_run_factor(factor))
        elif pending:
            jobs = iter(pending)
            context = multiprocessing.get_context("spawn")
            with concurrent_futures.ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                initializer=_initialize_worker,
                initargs=initializer_args,
            ) as executor:
                futures = {}

                def submit_one() -> bool:
                    """Submit one next job within the external bound."""
                    factor = next(jobs, None)
                    if factor is None:
                        return False
                    try:
                        future = executor.submit(_run_factor, factor)
                    except Exception as error:
                        # A dead process pool cannot accept replacement jobs.
                        # Preserve a failure row for all remaining factors.
                        for name in [factor, *jobs]:
                            record(_failed_summary(name, run_dir, error))
                        return False
                    futures[future] = factor
                    return True

                for _ in range(min(limit, len(pending))):
                    submit_one()
                while futures:
                    complete, _ = concurrent_futures.wait(
                        futures, return_when=concurrent_futures.FIRST_COMPLETED
                    )
                    for future in complete:
                        factor = futures.pop(future)
                        try:
                            row = future.result()
                        except Exception as error:
                            row = _failed_summary(factor, run_dir, error)
                        record(row)
                        submit_one()
    frame = _write_summary(output_dir, rows)
    frame.to_csv(run_dir / "summary.csv", index=False, encoding="utf-8-sig")
    return frame
