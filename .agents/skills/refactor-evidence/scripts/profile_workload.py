"""Representative CPU and allocation workload for repository profiling."""

from __future__ import annotations

import argparse
import json

from spatial_ipd.analytics import analyze_grid
from spatial_ipd.engine import simulate


def parse_args() -> argparse.Namespace:
    """Parse repeatable workload dimensions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260316)
    return parser.parse_args()


def main() -> None:
    """Run deterministic simulation and analytics repeatedly."""
    args = parse_args()
    if args.size < 1 or args.generations < 0 or args.repeats < 1:
        raise SystemExit("size/repeats must be positive and generations non-negative")
    checksum = 0.0
    largest_cluster = 0
    for repeat in range(args.repeats):
        result = simulate(
            args.size,
            args.size,
            args.generations,
            seed=args.seed + repeat,
            mutation_rate=0.02,
        )
        metrics = analyze_grid([list(row) for row in result.grid])
        checksum += result.final_cooperation_rate
        largest_cluster = max(largest_cluster, metrics.largest_cooperator_cluster)
    print(
        json.dumps(
            {
                "size": args.size,
                "generations": args.generations,
                "repeats": args.repeats,
                "checksum": checksum,
                "largest_cluster": largest_cluster,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
