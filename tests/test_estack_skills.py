"""Estack project skills are present and well-formed."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / ".agents" / "skills"


def test_estack_skills_exist_with_matching_name():
    for name in ("estack-lock", "estack-ship"):
        path = SKILLS / name / "SKILL.md"
        text = path.read_text(encoding="utf-8")
        assert text.startswith("---\n")
        assert f"name: {name}\n" in text
        assert "description:" in text
        assert "Cloud Agent" in text
