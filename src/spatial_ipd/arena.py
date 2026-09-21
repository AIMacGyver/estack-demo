"""Separate repeated-game arena for active deterministic policies."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Protocol

from spatial_ipd.payoffs import (
    COOPERATE,
    DEFECT,
    REWARD,
    TEMPTATION,
    payoff,
)


@dataclass(frozen=True)
class Observation:
    """Complete bounded history available before one simultaneous round."""

    round_index: int
    own_history: tuple[int, ...]
    opponent_history: tuple[int, ...]
    opponent_reputation: float | None = None
    opponent_warning_rate: float | None = None


class Policy(Protocol):
    """An active agent that chooses C or D from match history."""

    name: str

    def choose(self, observation: Observation) -> int:
        """Choose ``COOPERATE`` or ``DEFECT``."""


class AlwaysCooperate:
    """Cooperate every round."""

    name = "always_cooperate"

    def choose(self, observation: Observation) -> int:
        """Return C regardless of history."""
        del observation
        return COOPERATE


class AlwaysDefect:
    """Defect every round."""

    name = "always_defect"

    def choose(self, observation: Observation) -> int:
        """Return D regardless of history."""
        del observation
        return DEFECT


class TitForTat:
    """Cooperate first, then copy the opponent's previous action."""

    name = "tit_for_tat"

    def choose(self, observation: Observation) -> int:
        """Return C initially and then the opponent's last action."""
        if not observation.opponent_history:
            return COOPERATE
        return int(observation.opponent_history[-1])


class Pavlov:
    """Win-stay, lose-shift with C as the initial action."""

    name = "pavlov"

    def choose(self, observation: Observation) -> int:
        """Repeat after R/T; switch after P/S."""
        if not observation.own_history:
            return COOPERATE
        own = int(observation.own_history[-1])
        opponent = int(observation.opponent_history[-1])
        if payoff(own, opponent) in (REWARD, TEMPTATION):
            return own
        return COOPERATE if own == DEFECT else DEFECT


class ForgivingTitForTat:
    """Defect only after two consecutive opponent defections."""

    name = "forgiving_tit_for_tat"

    def choose(self, observation: Observation) -> int:
        """Forgive one defection; answer two consecutive defections with D."""
        if len(observation.opponent_history) < 2:
            return COOPERATE
        return DEFECT if observation.opponent_history[-2:] == (DEFECT, DEFECT) else COOPERATE


class MemoryWindowPolicy:
    """Expose only the most recent ``window`` rounds to another policy."""

    def __init__(self, policy: Policy, window: int):
        """Bind one policy to a positive history window."""
        if window < 1:
            raise ValueError("memory window must be positive")
        self.policy = policy
        self.window = window
        self.name = policy.name

    def choose(self, observation: Observation) -> int:
        """Delegate with histories truncated to the configured window."""
        return self.policy.choose(
            Observation(
                round_index=observation.round_index,
                own_history=observation.own_history[-self.window :],
                opponent_history=observation.opponent_history[-self.window :],
                opponent_reputation=observation.opponent_reputation,
                opponent_warning_rate=observation.opponent_warning_rate,
            )
        )


class ReputationGuard:
    """Cooperate with unknown/reputable opponents and defect below threshold."""

    name = "reputation_guard"

    def __init__(self, threshold: float = 0.5):
        """Bind a cooperation-rate threshold in [0, 1]."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("reputation threshold must be in [0, 1]")
        self.threshold = threshold

    def choose(self, observation: Observation) -> int:
        """Choose C without evidence; otherwise gate on opponent reputation."""
        reputation = observation.opponent_reputation
        if reputation is None or reputation >= self.threshold:
            return COOPERATE
        return DEFECT


class CommunicationGuard:
    """Cooperate without reports and defect when warnings reach threshold."""

    name = "communication_guard"

    def __init__(self, threshold: float = 0.5):
        """Bind a warning-rate threshold in [0, 1]."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("communication threshold must be in [0, 1]")
        self.threshold = threshold

    def choose(self, observation: Observation) -> int:
        """Gate cooperation on bounded warnings about the opponent."""
        warning_rate = observation.opponent_warning_rate
        if warning_rate is None or warning_rate < self.threshold:
            return COOPERATE
        return DEFECT


CANONICAL_POLICIES: tuple[type[Policy], ...] = (
    AlwaysCooperate,
    AlwaysDefect,
    TitForTat,
    Pavlov,
    ForgivingTitForTat,
)
GUARD_POLICIES: tuple[type[Policy], ...] = (
    *CANONICAL_POLICIES,
    ReputationGuard,
    CommunicationGuard,
)
POLICY_TYPES: dict[str, type[Policy]] = {policy_type.name: policy_type for policy_type in CANONICAL_POLICIES}
POLICY_TYPES[ReputationGuard.name] = ReputationGuard
POLICY_TYPES[CommunicationGuard.name] = CommunicationGuard
WARNING_THRESHOLD = 0.5
TOURNAMENT_CSV_FIELDS = (
    "information",
    "pass_name",
    "rank",
    "policy",
    "matches",
    "total_payoff",
    "average_payoff_per_round",
    "cooperation_rate",
    "reputation_signal",
    "warning_signal",
)


@dataclass(frozen=True)
class MatchResult:
    """Actions, payoffs, and cooperation summaries for one repeated match."""

    policy_a: str
    policy_b: str
    rounds: int
    actions_a: tuple[int, ...]
    actions_b: tuple[int, ...]
    payoff_a: int
    payoff_b: int
    cooperation_rate_a: float
    cooperation_rate_b: float
    mutual_cooperation_rounds: int


def _validated_action(action: int, policy_name: str) -> int:
    if int(action) not in (COOPERATE, DEFECT):
        raise ValueError(f"policy {policy_name!r} returned invalid action {action!r}")
    return int(action)


def play_match(
    policy_a: Policy,
    policy_b: Policy,
    *,
    rounds: int,
    opponent_reputation_a: float | None = None,
    opponent_reputation_b: float | None = None,
    opponent_warning_rate_a: float | None = None,
    opponent_warning_rate_b: float | None = None,
) -> MatchResult:
    """Play simultaneous repeated Prisoner's Dilemma for ``rounds``."""
    if rounds < 1:
        raise ValueError("rounds must be positive")
    actions_a: list[int] = []
    actions_b: list[int] = []
    payoff_a = 0
    payoff_b = 0
    mutual_cooperation = 0
    for round_index in range(rounds):
        action_a = _validated_action(
            policy_a.choose(
                Observation(
                    round_index=round_index,
                    own_history=tuple(actions_a),
                    opponent_history=tuple(actions_b),
                    opponent_reputation=opponent_reputation_a,
                    opponent_warning_rate=opponent_warning_rate_a,
                )
            ),
            policy_a.name,
        )
        action_b = _validated_action(
            policy_b.choose(
                Observation(
                    round_index=round_index,
                    own_history=tuple(actions_b),
                    opponent_history=tuple(actions_a),
                    opponent_reputation=opponent_reputation_b,
                    opponent_warning_rate=opponent_warning_rate_b,
                )
            ),
            policy_b.name,
        )
        actions_a.append(action_a)
        actions_b.append(action_b)
        payoff_a += payoff(action_a, action_b)
        payoff_b += payoff(action_b, action_a)
        mutual_cooperation += action_a == COOPERATE and action_b == COOPERATE
    return MatchResult(
        policy_a=policy_a.name,
        policy_b=policy_b.name,
        rounds=rounds,
        actions_a=tuple(actions_a),
        actions_b=tuple(actions_b),
        payoff_a=payoff_a,
        payoff_b=payoff_b,
        cooperation_rate_a=sum(actions_a) / rounds,
        cooperation_rate_b=sum(actions_b) / rounds,
        mutual_cooperation_rounds=mutual_cooperation,
    )


def roster_for(name: str) -> tuple[type[Policy], ...]:
    """Return the named deterministic tournament roster."""
    if name == "canonical":
        return CANONICAL_POLICIES
    if name == "guards":
        return GUARD_POLICIES
    raise ValueError("roster must be canonical or guards")


def round_robin(
    *,
    rounds: int,
    policy_types: Sequence[type[Policy]] = CANONICAL_POLICIES,
    signals: Mapping[str, Mapping[str, float]] | None = None,
) -> list[MatchResult]:
    """Play every unordered policy pairing, including self-play."""
    results = []
    for policy_a, policy_b in combinations_with_replacement(policy_types, 2):
        seen_by_a = (signals or {}).get(policy_b.name, {})
        seen_by_b = (signals or {}).get(policy_a.name, {})
        results.append(
            play_match(
                policy_a(),
                policy_b(),
                rounds=rounds,
                opponent_reputation_a=seen_by_a.get("reputation"),
                opponent_warning_rate_a=seen_by_a.get("warning_rate"),
                opponent_reputation_b=seen_by_b.get("reputation"),
                opponent_warning_rate_b=seen_by_b.get("warning_rate"),
            )
        )
    return results


def policy_aggregates(results: Sequence[MatchResult]) -> dict[str, dict[str, float]]:
    """Sum both sides of every pairing, including self-play."""
    totals: dict[str, dict[str, float]] = {}
    for result in results:
        for policy, payoff_total, cooperation in (
            (result.policy_a, result.payoff_a, result.cooperation_rate_a),
            (result.policy_b, result.payoff_b, result.cooperation_rate_b),
        ):
            current = totals.setdefault(
                policy,
                {"matches": 0, "total_payoff": 0, "cooperation_sum": 0.0, "rounds": 0},
            )
            current["matches"] += 1
            current["total_payoff"] += payoff_total
            current["cooperation_sum"] += cooperation
            current["rounds"] += result.rounds
    return totals


def information_signals(
    results: Sequence[MatchResult],
    *,
    threshold: float = WARNING_THRESHOLD,
) -> dict[str, dict[str, float]]:
    """Derive identity-level reputation and binary warnings from first-pass play."""
    signals = {}
    for policy, totals in policy_aggregates(results).items():
        cooperation = totals["cooperation_sum"] / totals["matches"]
        signals[policy] = {
            "reputation": cooperation,
            "warning_rate": 1.0 if cooperation < threshold else 0.0,
        }
    return signals


def rank_policies(
    results: Sequence[MatchResult],
    *,
    information: str,
    pass_name: str,
    signals: Mapping[str, Mapping[str, float]] | None = None,
) -> list[dict[str, object]]:
    """Rank policies by total payoff, breaking ties by name."""
    rows = []
    for policy, totals in policy_aggregates(results).items():
        signal = (signals or {}).get(policy, {})
        rows.append(
            {
                "information": information,
                "pass_name": pass_name,
                "policy": policy,
                "matches": int(totals["matches"]),
                "total_payoff": int(totals["total_payoff"]),
                "average_payoff_per_round": totals["total_payoff"] / totals["rounds"],
                "cooperation_rate": totals["cooperation_sum"] / totals["matches"],
                "reputation_signal": signal.get("reputation"),
                "warning_signal": signal.get("warning_rate"),
            }
        )
    rows.sort(key=lambda row: (-int(row["total_payoff"]), str(row["policy"])))
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def run_tournament(
    *,
    rounds: int,
    roster: str = "canonical",
    information: str = "off",
) -> dict[str, object]:
    """Run one deterministic tournament, optionally replaying with play-derived signals."""
    if information not in {"off", "on"}:
        raise ValueError("information must be off or on")
    policy_types = roster_for(roster)
    first = round_robin(rounds=rounds, policy_types=policy_types)
    passes = [
        {
            "pass_name": "uninformed",
            "matches": first,
            "rankings": rank_policies(first, information=information, pass_name="uninformed"),
        }
    ]
    if information == "on":
        signals = information_signals(first)
        second = round_robin(rounds=rounds, policy_types=policy_types, signals=signals)
        passes.append(
            {
                "pass_name": "informed",
                "matches": second,
                "rankings": rank_policies(
                    second,
                    information=information,
                    pass_name="informed",
                    signals=signals,
                ),
            }
        )
    return {
        "rounds": rounds,
        "roster": roster,
        "information": information,
        "policies": [policy_type.name for policy_type in policy_types],
        "passes": passes,
    }


def write_jsonl(path: str | Path, tournament: Mapping[str, object]) -> Path:
    """Write tournament matches, rankings, and one summary record."""
    destination = Path(path)
    records: list[dict[str, object]] = []
    for item in tournament["passes"]:
        for result in item["matches"]:
            records.append(
                {
                    "record_type": "match",
                    "information": tournament["information"],
                    "pass_name": item["pass_name"],
                    **asdict(result),
                }
            )
        for row in item["rankings"]:
            records.append({"record_type": "ranking", **row})
    records.append(
        {
            "record_type": "summary",
            "information": tournament["information"],
            "matches": sum(len(item["matches"]) for item in tournament["passes"]),
            "passes": [item["pass_name"] for item in tournament["passes"]],
            "policies": tournament["policies"],
            "roster": tournament["roster"],
            "rounds_per_match": tournament["rounds"],
        }
    )
    destination.write_text(
        "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    return destination


def write_csv(path: str | Path, tournament: Mapping[str, object]) -> Path:
    """Write compact ranked payoff tables for every tournament pass."""
    destination = Path(path)
    rows = [row for item in tournament["passes"] for row in item["rankings"]]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TOURNAMENT_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse active-agent arena flags."""
    parser = argparse.ArgumentParser(
        prog="python -m spatial_ipd.arena",
        description="Run deterministic repeated matches between canonical policies.",
    )
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--roster", choices=("canonical", "guards"), default="canonical")
    parser.add_argument("--information", choices=("off", "on"), default="off")
    parser.add_argument("--jsonl")
    parser.add_argument("--csv")
    args = parser.parse_args(argv)
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Print tournament matches and one summary record."""
    args = parse_args(argv)
    tournament = run_tournament(
        rounds=args.rounds,
        roster=args.roster,
        information=args.information,
    )
    if args.jsonl:
        write_jsonl(args.jsonl, tournament)
    if args.csv:
        write_csv(args.csv, tournament)
    if args.information == "off" and args.roster == "canonical" and not args.jsonl and not args.csv:
        results = tournament["passes"][0]["matches"]
        for result in results:
            print(
                json.dumps(
                    {"record_type": "match", **asdict(result)},
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        print(
            json.dumps(
                {
                    "record_type": "summary",
                    "policies": len(CANONICAL_POLICIES),
                    "matches": len(results),
                    "rounds_per_match": args.rounds,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    print(
        json.dumps(
            {
                "record_type": "summary",
                "information": tournament["information"],
                "matches": sum(len(item["matches"]) for item in tournament["passes"]),
                "passes": [item["pass_name"] for item in tournament["passes"]],
                "policies": tournament["policies"],
                "roster": tournament["roster"],
                "rounds_per_match": tournament["rounds"],
                **({} if not args.jsonl else {"jsonl": args.jsonl}),
                **({} if not args.csv else {"csv": args.csv}),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
