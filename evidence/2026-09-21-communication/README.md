# Truthful bounded communication evidence — 2026-09-21

This experiment isolates a one-bit warning mechanism while preserving the
population, seed, pair schedule, rounds, payoffs, and disabled control.

## Mechanism

After each match, both participants publish one truthful report about the
opponent:

- warning when opponent cooperation rate was below 0.5
- no warning otherwise

Reports are bounded booleans. The global warning rate for an identified subject
is warnings divided by reports. `communication_guard` cooperates when no report
exists or warning rate is below 0.5, and defects otherwise.

The disabled control passes no warning signal, making the guard cooperate
unconditionally.

## Method

- two agents each: Always Cooperate, Always Defect, Communication Guard, TFT
- 50 seeded encounter shuffles
- 200 matches, 10 rounds per match
- identical disabled/enabled pair schedules
- reputation disabled; no selection, mutation, message decay, or deception

## Results

Communication Guard:

- cooperation rate: 1.000 → 0.780
- average payoff per round: 2.340 → 2.540

Always Defect:

- cooperation rate: 0.000 in both runs
- average payoff per round: 3.128 → 2.288

Always Cooperate and Tit-for-Tat summaries were unchanged.

Truthful warnings spread information quickly enough to protect warning-aware
agents from repeat exploitation. As with direct reputation, defectors did not
adapt. Total payoff fell from 9,956 to 9,316 and mean cooperation across policy
groups fell from 0.678 to 0.623 as more guard/defector encounters became D/D.

## Comparison with direct reputation

The direct cumulative-reputation guard reached payoff 2.550 and cooperation
0.790 in the comparable control. The communication guard reached 2.540 and
0.780. These single-seed results are very close; they do not establish that one
information channel is superior.

## Limits

Messages are truthful, globally visible, permanent, and code-generated. There
is no sender credibility, local range, lying, free text, or strategic messaging.
This is an information-propagation baseline, not a general communication model.

## Reproduce

```bash
for mode in disabled enabled; do
  uv run python -m spatial_ipd.population \
    --manifest "evidence/2026-09-21-communication/communication-$mode.json" \
    --jsonl "evidence/2026-09-21-communication/encounters-$mode.jsonl" \
    --csv "evidence/2026-09-21-communication/summary-$mode.csv"
done
```
