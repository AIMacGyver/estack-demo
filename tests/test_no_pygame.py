"""Guard: core engine stays headless. Only the optional viewer may mention pygame."""

import sys
from pathlib import Path

import spatial_ipd

ROOT = Path(__file__).resolve().parents[1]
VIEWER_FILES = frozenset({"viewer.py"})


def test_no_pygame_import_in_engine_source():
    banned = ("import pygame", "from pygame")
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        if path.name in VIEWER_FILES:
            continue
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in banned):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_package_does_not_load_pygame():
    assert "pygame" not in sys.modules
    assert spatial_ipd.__name__ == "spatial_ipd"


def test_core_engine_modules_do_not_import_pygame():
    """Importing payoffs / neighborhood / engine must not pull pygame in."""
    sys.modules.pop("pygame", None)
    for name in (
        "spatial_ipd.payoffs",
        "spatial_ipd.neighborhood",
        "spatial_ipd.engine",
        "spatial_ipd",
    ):
        sys.modules.pop(name, None)
    import spatial_ipd.engine  # noqa: F401
    import spatial_ipd.neighborhood  # noqa: F401
    import spatial_ipd.payoffs  # noqa: F401

    assert "pygame" not in sys.modules
