"""Shared test fixtures for Part B.

Lets classifier tests run against a hand-written library export without
depending on Part A's music.export_library() / load_library_export().
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from music import Song  # noqa: E402 -- must follow the sys.path insert

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "library_export.json"


def load_fixture_songs(path: Path = FIXTURE_PATH) -> list[Song]:
    with path.open() as f:
        return [Song(**entry) for entry in json.load(f)]


@pytest.fixture
def songs() -> list[Song]:
    return load_fixture_songs()


@pytest.fixture
def song(songs: list[Song]) -> Song:
    return songs[0]
