"""Evidence-first refactoring skill and optional analysis group."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_refactor_evidence_skill_is_discoverable_and_concise():
    path = ROOT / ".agents" / "skills" / "refactor-evidence" / "SKILL.md"
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "name: refactor-evidence\n" in text
    assert "description:" in text
    assert "uv run pytest" in text
    assert "Pyinstrument" in text
    assert "Memray" in text
    assert len(text.splitlines()) < 500


def test_analysis_dependency_group_contains_selected_tools():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["dependency-groups"]["analysis"]
    names = {requirement.split(">", 1)[0] for requirement in dependencies}
    assert names == {
        "complexipy",
        "memray",
        "pyinstrument",
        "pytest-benchmark",
        "radon",
        "vulture",
    }


def test_profile_workload_script_is_present():
    path = ROOT / ".agents" / "skills" / "refactor-evidence" / "scripts" / "profile_workload.py"
    assert path.is_file()
