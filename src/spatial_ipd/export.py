"""Headless CSV export of a seeded Spatial IPD cooperation-rate series."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

from spatial_ipd.engine import SimulationResult, simulate

CSV_HEADER = ("generation", "cooperation_rate")


def write_cooperation_csv(result: SimulationResult, out_path: Path | str) -> Path:
    """Write one row per recorded generation (index 0 = initial lattice)."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(CSV_HEADER)
        for generation, rate in enumerate(result.cooperation_rates):
            writer.writerow([generation, rate])
    return path


def format_summary(result: SimulationResult, out_path: Path | str | None = None) -> str:
    """One-line final summary printed by the CLI."""
    parts = [
        f"final_cooperation_rate={result.final_cooperation_rate}",
        f"generations={result.generations}",
        f"seed={result.seed}",
        f"rows={len(result.cooperation_rates)}",
    ]
    if out_path is not None:
        parts.append(f"out={out_path}")
    return " ".join(parts)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse export CLI flags."""
    parser = argparse.ArgumentParser(
        prog="python -m spatial_ipd.export",
        description=(
            "Run a seeded Spatial IPD simulation and write a cooperation-rate time series to CSV (no display required)."
        ),
    )
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--generations", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mutation-rate", type=float, default=0.0)
    parser.add_argument("--out", required=True, help="Output CSV path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run a seeded simulation and write the cooperation-rate CSV."""
    args = parse_args(argv)
    result = simulate(
        args.height,
        args.width,
        args.generations,
        seed=args.seed,
        mutation_rate=args.mutation_rate,
    )
    write_cooperation_csv(result, args.out)
    print(format_summary(result, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
