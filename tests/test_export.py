"""Golden CSV export for the locked seeded engine run."""

from pathlib import Path

from spatial_ipd.engine import simulate
from spatial_ipd.export import format_summary, main, write_cooperation_csv

from test_engine import (
    SEEDED_CELLS,
    SEEDED_FINAL_COOPERATION_RATE,
    SEEDED_GENERATIONS,
    SEEDED_HEIGHT,
    SEEDED_MUTATION_RATE,
    SEEDED_RATES,
    SEEDED_SEED,
    SEEDED_WIDTH,
)

SEEDED_CLI = [
    "--height",
    str(SEEDED_HEIGHT),
    "--width",
    str(SEEDED_WIDTH),
    "--generations",
    str(SEEDED_GENERATIONS),
    "--seed",
    str(SEEDED_SEED),
    "--mutation-rate",
    str(SEEDED_MUTATION_RATE),
]


def _expected_csv_text() -> str:
    lines = ["generation,cooperation_rate"]
    lines.extend(f"{generation},{rate}" for generation, rate in enumerate(SEEDED_RATES))
    return "\n".join(lines) + "\n"


def test_export_writes_exact_golden_csv(tmp_path: Path):
    result = simulate(
        SEEDED_HEIGHT,
        SEEDED_WIDTH,
        SEEDED_GENERATIONS,
        seed=SEEDED_SEED,
        mutation_rate=SEEDED_MUTATION_RATE,
    )
    assert result.final_cooperation_rate == SEEDED_FINAL_COOPERATION_RATE
    assert result.final_cooperation_rate == 2 / SEEDED_CELLS
    assert result.cooperation_rates == SEEDED_RATES

    out = tmp_path / "coop_rates.csv"
    write_cooperation_csv(result, out)
    assert out.read_text(encoding="utf-8") == _expected_csv_text()


def test_cli_writes_csv_and_prints_summary(tmp_path: Path, capsys):
    out = tmp_path / "coop_rates.csv"
    code = main([*SEEDED_CLI, "--out", str(out)])
    assert code == 0

    captured = capsys.readouterr()
    expected = simulate(
        SEEDED_HEIGHT,
        SEEDED_WIDTH,
        SEEDED_GENERATIONS,
        seed=SEEDED_SEED,
        mutation_rate=SEEDED_MUTATION_RATE,
    )
    assert captured.out == format_summary(expected, str(out)) + "\n"
    assert captured.err == ""
    assert out.read_text(encoding="utf-8") == _expected_csv_text()

    rows = out.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "generation,cooperation_rate"
    assert len(rows) == SEEDED_GENERATIONS + 2  # header + gen 0 + 30 updates
    assert rows[-1] == f"{SEEDED_GENERATIONS},{2 / SEEDED_CELLS}"
