"""Manifest-driven, resumable Spatial IPD backend experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from spatial_ipd.analytics import analyze_grid, cooperation_persistence
from spatial_ipd.engine import SimulationResult, simulate
from spatial_ipd.label import load_dotenv
from spatial_ipd.local_llm import (
    DEFAULT_LOCAL_LLM_ENDPOINT,
    DEFAULT_LOCAL_LLM_TIMEOUT,
    LOCAL_LLM_REASONING_EFFORTS,
    LocalLLMThinkerClient,
)
from spatial_ipd.telemetry import telemetry_summary
from spatial_ipd.think import RandomThinkerClient, ThinkerStats, simulate_with_thinkers

EXPERIMENT_MANIFEST_VERSION = 1
EXPERIMENT_RESULT_VERSION = 1
BACKEND_KINDS = ("baseline", "random", "jev", "local")
BACKEND_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
CSV_FIELDS = (
    "run_id",
    "backend_id",
    "backend_kind",
    "seed",
    "status",
    "final_cooperation_rate",
    "cooperation_auc",
    "cooperation_persistence",
    "cooperator_clusters",
    "largest_cooperator_cluster",
    "frontier_cells",
    "cooperator_frontier_cells",
    "mean_cooperator_payoff",
    "mean_defector_payoff",
    "backend_calls",
    "backend_call_seconds",
    "mean_backend_call_seconds",
    "backend_call_failures",
    "raw_response_bytes",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "think_calls",
    "holds",
    "flips",
    "imitates",
    "elapsed_seconds",
    "backend_error_category",
    "error_type",
    "error",
)

ClientFactory = Callable[[Mapping[str, object], int], object | None]


class ExperimentError(ValueError):
    """A malformed manifest or result file."""


def _require_int(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExperimentError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ExperimentError(f"{name} must be at least {minimum}")
    return value


def _require_float(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExperimentError(f"{name} must be a number")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ExperimentError(f"{name} must be at least {minimum}")
    if maximum is not None and number > maximum:
        raise ExperimentError(f"{name} must be at most {maximum}")
    return number


def _simulation_config(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ExperimentError("simulation must be an object")
    return {
        "height": _require_int(value.get("height"), "simulation.height", minimum=1),
        "width": _require_int(value.get("width"), "simulation.width", minimum=1),
        "generations": _require_int(value.get("generations"), "simulation.generations", minimum=0),
        "mutation_rate": _require_float(
            value.get("mutation_rate", 0.0),
            "simulation.mutation_rate",
            minimum=0.0,
            maximum=1.0,
        ),
        "cooperate_p": _require_float(
            value.get("cooperate_p", 0.5),
            "simulation.cooperate_p",
            minimum=0.0,
            maximum=1.0,
        ),
    }


def _thinker_config(value: object) -> dict[str, object]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ExperimentError("thinker must be an object")
    seats = value.get("seats", "frontier")
    if seats not in ("frontier", "random"):
        raise ExperimentError("thinker.seats must be 'frontier' or 'random'")
    return {
        "think_every": _require_int(value.get("think_every", 5), "thinker.think_every", minimum=0),
        "think_last": _require_int(value.get("think_last", 5), "thinker.think_last", minimum=0),
        "sticky": _require_int(value.get("sticky", 5), "thinker.sticky", minimum=0),
        "thinkers": _require_int(value.get("thinkers", 4), "thinker.thinkers", minimum=0),
        "seats": seats,
    }


def _backend_config(value: object, index: int) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ExperimentError(f"backends[{index}] must be an object")
    backend_id = value.get("id")
    if not isinstance(backend_id, str) or not BACKEND_ID_PATTERN.fullmatch(backend_id):
        raise ExperimentError(f"backends[{index}].id must match {BACKEND_ID_PATTERN.pattern}")
    kind = value.get("kind")
    if kind not in BACKEND_KINDS:
        raise ExperimentError(f"backends[{index}].kind must be one of {BACKEND_KINDS}")
    backend: dict[str, object] = {"id": backend_id, "kind": kind}
    if kind == "local":
        model = value.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ExperimentError(f"backends[{index}].model is required for local")
        endpoint = value.get("endpoint", DEFAULT_LOCAL_LLM_ENDPOINT)
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ExperimentError(f"backends[{index}].endpoint must be a non-empty string")
        timeout = _require_float(
            value.get("timeout", DEFAULT_LOCAL_LLM_TIMEOUT),
            f"backends[{index}].timeout",
            minimum=0.001,
        )
        reasoning_effort = value.get("reasoning_effort")
        if reasoning_effort is not None and reasoning_effort not in LOCAL_LLM_REASONING_EFFORTS:
            raise ExperimentError(f"backends[{index}].reasoning_effort must be one of {LOCAL_LLM_REASONING_EFFORTS}")
        backend.update(
            model=model,
            endpoint=endpoint,
            timeout=timeout,
            reasoning_effort=reasoning_effort,
        )
    return backend


def normalize_manifest(value: object) -> dict[str, object]:
    """Validate and normalize one versioned experiment manifest."""
    if not isinstance(value, Mapping):
        raise ExperimentError("manifest must be a JSON object")
    version = value.get("schema_version")
    if version != EXPERIMENT_MANIFEST_VERSION:
        raise ExperimentError(f"manifest schema_version must be {EXPERIMENT_MANIFEST_VERSION}, got {version!r}")
    seeds_value = value.get("seeds")
    if not isinstance(seeds_value, list) or not seeds_value:
        raise ExperimentError("seeds must be a non-empty list")
    seeds = [_require_int(seed, f"seeds[{index}]") for index, seed in enumerate(seeds_value)]
    if len(set(seeds)) != len(seeds):
        raise ExperimentError("seeds must be unique")
    backends_value = value.get("backends")
    if not isinstance(backends_value, list) or not backends_value:
        raise ExperimentError("backends must be a non-empty list")
    backends = [_backend_config(backend, index) for index, backend in enumerate(backends_value)]
    backend_ids = [str(backend["id"]) for backend in backends]
    if len(set(backend_ids)) != len(backend_ids):
        raise ExperimentError("backend ids must be unique")
    return {
        "schema_version": EXPERIMENT_MANIFEST_VERSION,
        "simulation": _simulation_config(value.get("simulation")),
        "thinker": _thinker_config(value.get("thinker")),
        "seeds": seeds,
        "backends": backends,
    }


def read_manifest(path: str | Path) -> dict[str, object]:
    """Read and normalize a JSON experiment manifest."""
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExperimentError(f"manifest {source} is invalid JSON") from exc
    return normalize_manifest(value)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def run_id_for(
    simulation: Mapping[str, object],
    thinker: Mapping[str, object],
    backend: Mapping[str, object],
    seed: int,
) -> str:
    """Build a stable run ID from every input that can affect a run."""
    payload = {
        "manifest_version": EXPERIMENT_MANIFEST_VERSION,
        "simulation": simulation,
        "thinker": thinker,
        "backend": backend,
        "seed": seed,
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:12]
    return f"{backend['id']}-{seed}-{digest}"


def planned_runs(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    """Expand a normalized manifest into backend-by-seed run plans."""
    simulation = manifest["simulation"]
    thinker = manifest["thinker"]
    return [
        {
            "run_id": run_id_for(simulation, thinker, backend, seed),
            "simulation": simulation,
            "thinker": thinker,
            "backend": backend,
            "seed": seed,
        }
        for backend in manifest["backends"]
        for seed in manifest["seeds"]
    ]


def cooperation_auc(rates: Sequence[float]) -> float:
    """Normalized trapezoidal area under a cooperation trajectory."""
    if not rates:
        raise ValueError("cooperation rates must not be empty")
    if len(rates) == 1:
        return float(rates[0])
    area = sum((left + right) / 2.0 for left, right in zip(rates, rates[1:], strict=False))
    return area / (len(rates) - 1)


def _default_client(backend: Mapping[str, object], seed: int) -> object | None:
    kind = backend["kind"]
    if kind == "random":
        return RandomThinkerClient(seed)
    if kind == "local":
        return LocalLLMThinkerClient(
            str(backend["model"]),
            endpoint=str(backend["endpoint"]),
            api_key=os.environ.get("LOCAL_LLM_API_KEY"),
            timeout=float(backend["timeout"]),
            reasoning_effort=backend.get("reasoning_effort"),
        )
    return None


def _run_simulation(
    plan: Mapping[str, object],
    client_factories: Mapping[str, ClientFactory],
) -> tuple[SimulationResult, ThinkerStats]:
    simulation = plan["simulation"]
    thinker = plan["thinker"]
    backend = plan["backend"]
    seed = int(plan["seed"])
    common = {
        "height": int(simulation["height"]),
        "width": int(simulation["width"]),
        "generations": int(simulation["generations"]),
        "seed": seed,
        "mutation_rate": float(simulation["mutation_rate"]),
        "cooperate_p": float(simulation["cooperate_p"]),
    }
    if backend["kind"] == "baseline":
        return simulate(**common), ThinkerStats()
    factory = client_factories.get(str(backend["kind"]))
    client = factory(backend, seed) if factory is not None else _default_client(backend, seed)
    return simulate_with_thinkers(
        **common,
        think_every=int(thinker["think_every"]),
        think_last=int(thinker["think_last"]),
        sticky=int(thinker["sticky"]),
        thinker_count=int(thinker["thinkers"]),
        client=client,
        seat_mode=str(thinker["seats"]),
    )


def execute_run(
    plan: Mapping[str, object],
    *,
    client_factories: Mapping[str, ClientFactory] | None = None,
    clock: Callable[[], float] = perf_counter,
) -> dict[str, object]:
    """Execute one plan and return a success or isolated failure record."""
    factories = {} if client_factories is None else client_factories
    started = clock()
    base = {
        "schema_version": EXPERIMENT_RESULT_VERSION,
        "run_id": plan["run_id"],
        "backend": plan["backend"],
        "seed": plan["seed"],
        "simulation": plan["simulation"],
        "thinker": plan["thinker"],
    }
    try:
        result, stats = _run_simulation(plan, factories)
    except Exception as exc:  # noqa: BLE001 - one backend failure must not stop the manifest
        failure = {
            **base,
            "status": "error",
            "elapsed_seconds": clock() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        backend_telemetry = getattr(exc, "spatial_ipd_backend_telemetry", None)
        if backend_telemetry is not None:
            failure.update(telemetry_summary([backend_telemetry]))
            failure["backend_error_category"] = backend_telemetry.error_category
        return failure
    spatial = analyze_grid([list(row) for row in result.grid])
    return {
        **base,
        "status": "success",
        "elapsed_seconds": clock() - started,
        "cooperation_rates": list(result.cooperation_rates),
        "final_cooperation_rate": result.final_cooperation_rate,
        "cooperation_auc": cooperation_auc(result.cooperation_rates),
        "cooperation_persistence": cooperation_persistence(result.cooperation_rates),
        **asdict(spatial),
        **telemetry_summary(stats.backend_telemetry),
        "think_calls": stats.calls,
        "holds": stats.holds,
        "flips": stats.flips,
        "imitates": stats.imitates,
    }


def read_result_records(path: str | Path) -> list[dict[str, object]]:
    """Read and validate existing experiment JSONL, if present."""
    source = Path(path)
    if not source.exists():
        return []
    records = []
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ExperimentError(f"result line {line_number} is invalid JSON") from exc
        if not isinstance(record, dict):
            raise ExperimentError(f"result line {line_number} must be an object")
        if record.get("schema_version") != EXPERIMENT_RESULT_VERSION:
            raise ExperimentError(f"result line {line_number} has an unsupported schema version")
        if not isinstance(record.get("run_id"), str) or record.get("status") not in ("success", "error"):
            raise ExperimentError(f"result line {line_number} is missing run_id or status")
        records.append(record)
    return records


def _csv_row(record: Mapping[str, object]) -> dict[str, object]:
    backend = record["backend"]
    return {
        "run_id": record["run_id"],
        "backend_id": backend["id"],
        "backend_kind": backend["kind"],
        "seed": record["seed"],
        "status": record["status"],
        "final_cooperation_rate": record.get("final_cooperation_rate", ""),
        "cooperation_auc": record.get("cooperation_auc", ""),
        "cooperation_persistence": record.get("cooperation_persistence", ""),
        "cooperator_clusters": record.get("cooperator_clusters", ""),
        "largest_cooperator_cluster": record.get("largest_cooperator_cluster", ""),
        "frontier_cells": record.get("frontier_cells", ""),
        "cooperator_frontier_cells": record.get("cooperator_frontier_cells", ""),
        "mean_cooperator_payoff": record.get("mean_cooperator_payoff", ""),
        "mean_defector_payoff": record.get("mean_defector_payoff", ""),
        "backend_calls": record.get("backend_calls", ""),
        "backend_call_seconds": record.get("backend_call_seconds", ""),
        "mean_backend_call_seconds": record.get("mean_backend_call_seconds", ""),
        "backend_call_failures": record.get("backend_call_failures", ""),
        "raw_response_bytes": record.get("raw_response_bytes", ""),
        "input_tokens": record.get("input_tokens", ""),
        "output_tokens": record.get("output_tokens", ""),
        "total_tokens": record.get("total_tokens", ""),
        "think_calls": record.get("think_calls", ""),
        "holds": record.get("holds", ""),
        "flips": record.get("flips", ""),
        "imitates": record.get("imitates", ""),
        "elapsed_seconds": record["elapsed_seconds"],
        "backend_error_category": record.get("backend_error_category", ""),
        "error_type": record.get("error_type", ""),
        "error": record.get("error", ""),
    }


def write_summary_csv(path: str | Path, records: Sequence[Mapping[str, object]]) -> Path:
    """Regenerate compact CSV using the latest record for each run ID."""
    latest = {str(record["run_id"]): record for record in records}
    destination = Path(path)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for run_id in sorted(latest):
            writer.writerow(_csv_row(latest[run_id]))
    return destination


def run_manifest(
    manifest: Mapping[str, object],
    *,
    jsonl_path: str | Path,
    csv_path: str | Path,
    client_factories: Mapping[str, ClientFactory] | None = None,
    clock: Callable[[], float] = perf_counter,
) -> dict[str, int]:
    """Run unfinished plans, append JSONL records, and regenerate CSV."""
    plans = planned_runs(manifest)
    existing = read_result_records(jsonl_path)
    succeeded_ids = {str(record["run_id"]) for record in existing if record["status"] == "success"}
    pending = [plan for plan in plans if plan["run_id"] not in succeeded_ids]
    new_records = []
    destination = Path(jsonl_path)
    with destination.open("a", encoding="utf-8") as handle:
        for plan in pending:
            record = execute_run(plan, client_factories=client_factories, clock=clock)
            handle.write(_canonical_json(record) + "\n")
            handle.flush()
            new_records.append(record)
    all_records = [*existing, *new_records]
    write_summary_csv(csv_path, all_records)
    return {
        "planned": len(plans),
        "executed": len(new_records),
        "skipped": len(plans) - len(new_records),
        "succeeded": sum(record["status"] == "success" for record in new_records),
        "failed": sum(record["status"] == "error" for record in new_records),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse experiment runner CLI flags."""
    parser = argparse.ArgumentParser(
        prog="python -m spatial_ipd.experiment",
        description="Run a resumable manifest across Spatial IPD thinker backends.",
    )
    parser.add_argument("--manifest", required=True, help="Versioned experiment manifest JSON.")
    parser.add_argument("--jsonl", required=True, help="Append-only per-run result JSONL.")
    parser.add_argument("--csv", required=True, help="Regenerated compact CSV summary.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one manifest and print a compact execution summary."""
    load_dotenv()
    args = parse_args(argv)
    manifest = read_manifest(args.manifest)
    summary = run_manifest(manifest, jsonl_path=args.jsonl, csv_path=args.csv)
    print(
        f"planned={summary['planned']} executed={summary['executed']} "
        f"skipped={summary['skipped']} succeeded={summary['succeeded']} "
        f"failed={summary['failed']} jsonl={args.jsonl} csv={args.csv}"
    )
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
