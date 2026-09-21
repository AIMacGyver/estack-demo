# Spatial IPD

A small **Spatial Iterated Prisoner's Dilemma** engine (Nowak & May 1992): cooperation evolving on a 2-D lattice. This repo is an Estack dry-run demo — clone it, run the tests, export a cooperation-rate CSV, optionally open a Pygame window. The engine is stdlib-only; pygame is an optional extra.

**Share / clone:** [github.com/aimacgyver/estack-demo](https://github.com/aimacgyver/estack-demo)

**How we build:** [Estack](https://x.ai/bot/R0acF6Pmp8YewSZm6fA-D) for plan → execute → ship (small scope, prove with pytest). In Cursor that is two project skills — [`estack-lock`](.agents/skills/estack-lock/SKILL.md) then [`estack-ship`](.agents/skills/estack-ship/SKILL.md). Invoke them on a concrete task in this repo (`/estack-lock`, `/estack-ship`). The agent runs the SOP locally; you review the diff and test output. No Cloud Agent unless you ask. [TypeSafe](https://docs.typesafe.ai) (Jev) for snap decisions — after a run (regime / survived?) and *inside* the game (thinker hold/flip). Jev does not replace payoffs, `simulate()`, or the Estack lock.

**Estack Grok Bot template:** [add the Estack bot](https://x.ai/bot/R0acF6Pmp8YewSZm6fA-D) — a recipe, not our chats or API keys.

## Try it

Python 3.12+ and [uv](https://docs.astral.sh/uv/). From a clone of this repo:

### 1. Install and test

```bash
uv sync --group dev
uv run pytest
```

Same thing via `make sync` and `make test`.

### 2. Headless CSV export

No display required. Writes `generation,cooperation_rate` (generation `0` is the initial lattice) and prints one summary line. This seed is the locked golden (final cooperation `2/144`):

```bash
uv run python -m spatial_ipd.export \
  --height 12 --width 12 --generations 30 \
  --seed 20260316 --mutation-rate 0.02 \
  --out coop_rates.csv
```

### 3. Optional viewer

Needs a display. Blue = Cooperate, red = Defect. Space pauses, `n` single-steps, `r` resets, click selects a cell, `c` clears selection, and `q` / Esc quits.

```bash
uv sync --extra viewer
uv run python -m spatial_ipd.viewer \
  --height 32 --width 32 --seed 42 \
  --mutation-rate 0.01 --cell-size 12 --fps 8 \
  --think-every 5 --thinkers 4 --backend random
```

## Model

Each cell is Cooperate (`C = 1`) or Defect (`D = 0`). Every generation the cell plays one-shot PD against its eight Moore neighbors on a torus, then imitates the strict highest scorer among itself and those neighbors (focal wins ties; scan NW → W). Optional mutation then flips each cell with probability `mutation_rate` via `random.Random(seed)`.

|            | vs C | vs D |
|------------|------|------|
| **C**      | 3, 3 | 0, 5 |
| **D**      | 5, 0 | 1, 1 |

`T = 5`, `R = 3`, `P = 1`, `S = 0`. `simulate` and `step` are deterministic for a fixed seed, size, mutation rate, and generation count.

## Optional TypeSafe labels

After a run, [TypeSafe](https://docs.typesafe.ai) (Jev) can label the cooperation-rate *shape*. It does not change payoffs or update rules. Needs `TYPESAFE_API_KEY` (env var or a local `.env`, which is gitignored).

```bash
uv sync --extra typesafe
uv run python -m spatial_ipd.label \
  --height 12 --width 12 --generations 30 \
  --seed 20260316 --mutation-rate 0.02
```

Questions and the survival threshold live in `src/spatial_ipd/judgments.py`.

## Optional Jev thinkers

Most cells still imitate. Every `--think-every` generations, a few seats ask Jev whether a **C→D** cell should stay C to keep a cluster even though `best_neighbor_score` beat `focal_score`. Resist is asked only on real C→D seats. Code applies hold if worth and resist are both ≥ 0.6. This demo is Jev agreeing with imitate-the-best on a collapsing lattice, not a leftover-C rescue. Thinker JSON includes those imitate scores from `score_cells(before)` (same tie-break as `adopt_best`). Default `--seats frontier` prefers cells that just changed, then C/D edges. Default `--think-last 5` also thinks on every generation in the last five. Default `--sticky 5` keeps an applied hold for five later gens. `--seats random`, `--think-last 0`, and `--sticky 0` keep the older schedule. Payoffs and the default `simulate()` golden do not change. Needs `TYPESAFE_API_KEY`.

The engine golden is still seed `20260316` (export / `think_every=0`). The thinker demo uses seed `1`. Same seats and 0.6 gate; `--backend` picks who answers worth/resist.

**Jev** (needs `TYPESAFE_API_KEY`): live `--compare` was `plain_final=0.01171875` `think_final=0.01171875` `delta=0.0` (`holds=0`). Resist on C→D sat around 0.32–0.38. Default `--backend jev`.

```bash
uv sync --extra typesafe
uv run python -m spatial_ipd.think \
  --height 16 --width 16 --generations 30 \
  --seed 1 --mutation-rate 0.02 \
  --think-every 5 --think-last 5 --sticky 5 --thinkers 4 \
  --seats frontier --backend jev --compare --verbose
```

**Random** (no API key): seeded Uniform[0, 1] control. Same command with `--backend random` produced `think_final=0.0234375` `delta=0.01171875` (`holds=3`) — luck holds on Cs that already lost on payoff.

```bash
uv run python -m spatial_ipd.think \
  --height 16 --width 16 --generations 30 \
  --seed 1 --mutation-rate 0.02 \
  --think-every 5 --think-last 5 --sticky 5 --thinkers 4 \
  --seats frontier --backend random --compare --verbose
```

**Local LLM** (no Python extra): any model behind an OpenAI-compatible chat-completions endpoint can answer the same bounded worth/resist questions. This Ollama example uses the default endpoint, `http://localhost:11434/v1/chat/completions`:

```bash
uv run python -m spatial_ipd.think \
  --height 16 --width 16 --generations 30 \
  --seed 1 --mutation-rate 0.02 \
  --think-every 5 --think-last 5 --sticky 5 --thinkers 4 \
  --seats frontier --backend local --local-model qwen3:8b \
  --local-reasoning-effort none \
  --compare --verbose
```

Use `--local-endpoint URL` and `--local-timeout SECONDS` for another server. `--local-reasoning-effort none` prevents thinking-capable Ollama models from spending the request on hidden reasoning; omit it for servers that do not support the OpenAI reasoning field. If the endpoint requires bearer authentication, set `LOCAL_LLM_API_KEY` in the environment or gitignored local `.env`. Invalid HTTP/OpenAI responses and malformed decision JSON fail with a `LocalLLMError`. The model must choose boolean worth/resist actions; code maps those actions into the existing gate. It also reports `[0, 1]` confidence separately for inspection, tagged `llm_self_report` because it is not a calibrated probability. Raw local output remains available in `ThinkerStats.backend_audits`.

`--compare` prints plain `simulate()` vs the thinker run. `--verbose` prints each seat’s act, confidence provenance, worth/resist signals and confidence (`n/a` when resist was not asked), and the imitate scores.

### Record and replay thinker decisions

Record a run once, then reproduce it without Jev, Ollama, or another backend. The versioned JSONL keeps each thinker state, question metadata, normalized response/confidence provenance, and available raw backend output.

```bash
uv run python -m spatial_ipd.think \
  --height 8 --width 8 --generations 10 --seed 1 \
  --mutation-rate 0.02 --think-every 5 --think-last 0 \
  --sticky 0 --thinkers 2 --backend random \
  --record-decisions decisions.jsonl

uv run python -m spatial_ipd.think \
  --height 8 --width 8 --generations 10 --seed 1 \
  --mutation-rate 0.02 --think-every 5 --think-last 0 \
  --sticky 0 --thinkers 2 \
  --replay-decisions decisions.jsonl
```

Replay validates call order, state, question IDs, and full consumption. Changed simulation arguments fail clearly instead of silently applying a decision to the wrong seat.

### Run reproducible backend experiments

The versioned manifest in `examples/experiment.json` expands fixed simulation settings across repeated seeds and backend definitions. Results append to resumable JSONL; the compact CSV is regenerated with the latest result for each stable run ID.

```bash
uv run python -m spatial_ipd.experiment \
  --manifest examples/experiment.json \
  --jsonl runs.jsonl \
  --csv summary.csv
```

Successful IDs are skipped on rerun; failures are recorded and retried later without repeating successful live calls. JSONL rows include the full cooperation trajectory, normalized trapezoidal AUC, persistence, final cooperator cluster count/size, strategy frontier size, final payoff means by strategy, normalized per-seat decisions, thinker counters, elapsed time, and backend metadata. Per-call telemetry adds backend latency, failure category, raw response bytes, and token usage when the backend provides it. Isolated failures retain timeout/transport/schema classification. The compact CSV carries the same summary metrics except the decision list. The example is API-free (`baseline` and seeded `random`). Manifests may also use `jev` or `local`; local entries require `model` and may set `endpoint`, `timeout`, `reasoning_effort`, `prompt_profile` (`cluster_guard_v1` or `score_defer_v1`), `temperature`, and `max_tokens`. Credentials still come only from environment variables or the gitignored `.env`.

For a manifest containing two or more local backend IDs over identical seeds, compare one configuration with all others offline:

```bash
uv run python -m spatial_ipd.ablate \
  --jsonl runs.jsonl \
  --reference local-guard
```

The report matches generation/row/column decisions and shows action agreement, resist-probability and worth-confidence shifts, latency, and failures. Agreement remains a behavior comparison—not evidence of calibration or correctness.

Committed research evidence lives under `evidence/`. Start with the
[first matched Jev/local/random run](evidence/2026-09-21-matched-backends/README.md);
its small sample is documented as diagnostic rather than conclusive. The
[counterfactual label-yield spike](evidence/2026-09-21-counterfactual-yield/README.md)
records why exact ties remain unresolved and calibration claims are deferred.

### Probe deterministic motifs

Send the same named, code-derived situations through a thinker backend and print one inspectable JSON object per motif:

```bash
uv run python -m spatial_ipd.probes --backend random --seed 1
```

The catalog covers lone-defector invasion, isolated-cooperator collapse, a pivotal cluster hold, toroidal cluster recovery, a stable mixed boundary, and uniform cooperation. Use `--backend jev` or `--backend local --local-model qwen3:8b --local-reasoning-effort none` for live comparisons. Tests lock only the engine-derived strategy transitions and fake-client mapping—not stochastic live-model answers.

### Check probability calibration against outcomes

Agreement between two backends does not establish which probability is better. The calibration diagnostic branches each C→D motif after imitation, forces either imitate or one-cell hold, advances both with identical RNG streams, and labels whether hold improves mean future cooperation:

```bash
uv run python -m spatial_ipd.calibrate \
  --backend random --seed 1 --horizon 3
```

Each JSON event includes both future trajectories, the outcome label, confidence provenance, and probability oriented as `P(resist=true)`. The final record reports Brier score and binary log loss over resolved outcomes; exact ties are reported and excluded. Use `--backend jev` or the same local-model flags as the probe command for live diagnostics. The built-in motif set is intentionally tiny, so its summary sets `calibration_claim_supported=false`; it is plumbing and evidence, not a scientific calibration claim.

Build a fuller reliability report from one or more captured calibration runs:

```bash
uv run python -m spatial_ipd.calibrate \
  --backend random --seed 1 --horizon 3 > calibration.jsonl

uv run python -m spatial_ipd.reliability \
  --jsonl calibration.jsonl \
  --bins 10 --confidence-threshold 0.8 --minimum-events 50
```

The report separates Brier/log loss, expected calibration error and reliability bins, confidence-gated coverage/selective accuracy, and consistency between the backend's resist decision and its oriented probability. Ties never become labels. `calibration_claim_supported` requires the configured minimum resolved sample and both positive and negative outcomes.

### Run the separate active-agent arena

The Nowak–May engine remains unchanged. A separate repeated-game arena lets policies actively choose C or D from bounded match history:

```bash
uv run python -m spatial_ipd.arena --rounds 20
```

The deterministic round robin includes Always Cooperate, Always Defect, Tit-for-Tat, Pavlov (win-stay/lose-shift), and forgiving TFT. JSON match records include both action histories, payoffs, cooperation rates, and mutual-cooperation rounds. This is the stable policy seam for later memory or mixed-population experiments—not a replacement for the spatial engine.

Run a seeded mixed population without reproduction or selection:

```bash
uv run python -m spatial_ipd.population \
  --manifest examples/population.json \
  --jsonl population.jsonl \
  --csv population-summary.csv
```

The manifest fixes canonical policy counts, encounter schedules, and rounds per match. Optional `memory_window` truncates the within-match history visible to every policy while preserving the pair schedule. JSONL preserves every identified pairing and action history; CSV aggregates payoff and cooperation by policy. Repeating the same seed is byte-identical, while another seed changes pairing order.

See the [first mixed-population evidence run](evidence/2026-09-21-mixed-population/README.md)
for the fixed composition, results, and limits, then the
[bounded-memory comparison](evidence/2026-09-21-bounded-memory/README.md) and
[cumulative-reputation control](evidence/2026-09-21-reputation/README.md).
The [truthful bounded-communication control](evidence/2026-09-21-communication/README.md)
separates shared warnings from direct reputation.

Run deterministic payoff-proportional selection over canonical policy
frequencies:

```bash
uv run python -m spatial_ipd.evolution \
  --manifest examples/evolution.json \
  --jsonl evolution.jsonl \
  --csv evolution.csv
```

Population size stays fixed; largest-remainder allocation converts each
policy's total-payoff share into next-generation integer counts. The
[first evolutionary evidence run](evidence/2026-09-21-evolution/README.md)
uses a larger population to avoid coarse rounding, repeats ten base seeds, and
documents its no-mutation limits. Manifests accept either `seed` or a unique
`seeds` list plus an optional policy `mutation_rate`. The
[matched mutation control](evidence/2026-09-21-evolution-mutation/README.md)
compares rates 0 and 0.02 over the same ten seeds.

The viewer accepts `--backend random|jev|local` when `--think-every` is set (`--seats` works there too). Applied holds are outlined green, other thinker decisions gold, and the selected cell cyan; its strategy and payoff appear in the title. The lower panel plots cooperation (blue) and largest-cluster share (green). Local uses the same `--local-model`, endpoint, timeout, and reasoning flags as other thinker commands.

```python
from spatial_ipd import simulate

print(simulate(20, 20, 50, seed=42, mutation_rate=0.01).final_cooperation_rate)
```

## Development

Same layout as SkillFlow: uv, Ruff (Google docstrings), pre-commit, Makefile. Planning and execution follow the Estack skills (lock a Goal / Scope / Done card, then ship the smallest pytest-backed diff).

```bash
make sync
make hooks
make lint
make test
```

```bash
uv sync --group dev
uv run pre-commit install
uv run ruff check .
uv run ruff format .
uv run pytest
```

Optional extras: `uv sync --extra typesafe` and/or `--extra viewer`. Copy `.env.example` to `.env` for a TypeSafe key (gitignored).

### Evidence-first refactoring

Install the optional analysis stack and run its static/benchmark entry points:

```bash
make analysis-sync
make analysis-static
make benchmark
```

The project skill [`refactor-evidence`](.agents/skills/refactor-evidence/SKILL.md) defines the CPU, memory, benchmark, dead-code, and complexity workflow. Checked-in [`thresholds.toml`](.agents/skills/refactor-evidence/thresholds.toml) supplies default investigate/ship gates; overrides must name a numeric threshold and rationale, while behavior preservation is non-overridable. Research rationale and initial observations live in [`docs/analysis-tooling.md`](docs/analysis-tooling.md). Generated profiles and benchmark baselines are gitignored.
