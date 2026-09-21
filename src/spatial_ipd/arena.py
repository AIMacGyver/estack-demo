"""Separate repeated-game arena for active deterministic policies."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import combinations_with_replacement
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


CANONICAL_POLICIES: tuple[type[Policy], ...] = (
    AlwaysCooperate,
    AlwaysDefect,
    TitForTat,
    Pavlov,
    ForgivingTitForTat,
)
POLICY_TYPES: dict[str, type[Policy]] = {policy_type.name: policy_type for policy_type in CANONICAL_POLICIES}
POLICY_TYPES[ReputationGuard.name] = ReputationGuard


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


def round_robin(
    *,
    rounds: int,
    policy_types: Sequence[type[Policy]] = CANONICAL_POLICIES,
) -> list[MatchResult]:
    """Play every unordered policy pairing, including self-play."""
    return [
        play_match(policy_a(), policy_b(), rounds=rounds)
        for policy_a, policy_b in combinations_with_replacement(policy_types, 2)
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse active-agent arena flags."""
    parser = argparse.ArgumentParser(
        prog="python -m spatial_ipd.arena",
        description="Run deterministic repeated matches between canonical policies.",
    )
    parser.add_argument("--rounds", type=int, default=20)
    args = parser.parse_args(argv)
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Print canonical round-robin matches and one summary record."""
    args = parse_args(argv)
    results = round_robin(rounds=args.rounds)
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


if __name__ == "__main__":
    raise SystemExit(main())
