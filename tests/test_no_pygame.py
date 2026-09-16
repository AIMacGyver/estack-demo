"""Guard: this engine is headless. Pygame must never appear."""

import sys
from pathlib import Path

import spatial_ipd

ROOT = Path(__file__).resolve().parents[1]


def test_no_pygame_import_in_engine_source():
    banned = ("import pygame", "from pygame")
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in banned):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_package_does_not_load_pygame():
    assert "pygame" not in sys.modules
    assert spatial_ipd.__name__ == "spatial_ipd"
