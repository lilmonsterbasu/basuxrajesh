"""app.py

Orchestration layer -- glues Part A (music.py export), Part B
(classifier.py), and Part C (music.py grouping/write-back) together
behind a small set of functions that both ui.py (Tkinter) and
webapp/server.py (web dashboard) call into.

No part-specific logic should live here; this just sequences calls in
the right order and handles the intermediate JSON files.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from classifier import (
    ClassificationResult,
    get_classifier,
    load_classified_songs,
    save_classified_songs,
)
from music import (
    Song,
    build_playlists,
    export_library,
    load_library_export,
    save_library_export,
    sync_playlists_to_music,
)

# Progress callback signature: on_progress(done, total). The GUI and the web
# dashboard both use this to drive a progress bar during classification.
ProgressCallback = Callable[[int, int], None]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
LIBRARY_EXPORT_PATH = DATA_DIR / "library_export.json"
CLASSIFIED_SONGS_PATH = DATA_DIR / "classified_songs.json"
PLAYLIST_PLAN_PATH = DATA_DIR / "playlist_plan.json"


def run_export_library() -> list[Song]:
    """Step 1: pull every song + playlist from Apple Music via AppleScript.

    Depends on: Part A's music.export_library().
    """
    DATA_DIR.mkdir(exist_ok=True)
    songs, _playlists = export_library()
    save_library_export(songs, LIBRARY_EXPORT_PATH)
    logger.info("Exported %d songs to %s", len(songs), LIBRARY_EXPORT_PATH)
    return songs


def run_analyze_songs(
    provider: str = "claude",
    on_progress: ProgressCallback | None = None,
) -> list[ClassificationResult]:
    """Step 2: classify every exported song with the configured LLM provider.

    Depends on: Part B's get_classifier() / Classifier.classify_batch().
    `on_progress` is forwarded straight through so the UI layers can show a
    progress bar without knowing anything about the provider.
    """
    songs = load_library_export(LIBRARY_EXPORT_PATH)
    classifier = get_classifier(provider=provider)
    results = classifier.classify_batch(songs, on_progress=on_progress)
    save_classified_songs(results, CLASSIFIED_SONGS_PATH)
    logger.info("Classified %d songs using provider=%s", len(results), provider)
    return results


def load_analysis() -> list[ClassificationResult]:
    """Re-read the last classification run from disk.

    Lets "Preview Results" rebuild a plan without paying for classification
    again, and lets the web dashboard survive a server restart.
    """
    return load_classified_songs(CLASSIFIED_SONGS_PATH)


def save_playlist_plan(plan: dict[str, list[Song]]) -> None:
    """Persist a playlist plan, including full song rows.

    Storing whole songs rather than bare titles is what makes the plan
    round-trippable: the web dashboard lets the user drag songs between
    playlists, and the edited plan has to come back as real Song objects
    before Part C can write it to Apple Music.
    """
    PLAYLIST_PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PLAYLIST_PLAN_PATH.open("w") as f:
        json.dump(
            {name: [asdict(song) for song in songs] for name, songs in plan.items()},
            f,
            indent=2,
        )


def load_playlist_plan() -> dict[str, list[Song]]:
    """Read back a plan saved by save_playlist_plan()."""
    if not PLAYLIST_PLAN_PATH.exists():
        raise FileNotFoundError(
            f"{PLAYLIST_PLAN_PATH} not found -- build a playlist plan first"
        )
    with PLAYLIST_PLAN_PATH.open() as f:
        payload = json.load(f)
    return {
        name: [Song(**row) for row in rows] for name, rows in payload.items()
    }


def run_build_playlist_plan(classified_songs: list[ClassificationResult]) -> dict[str, list[Song]]:
    """Step 3: group classified songs into playlist -> songs mapping.

    Depends on: Part C's music.build_playlists().
    """
    plan = build_playlists(classified_songs)
    save_playlist_plan(plan)
    logger.info("Built playlist plan with %d playlists", len(plan))
    return plan


def run_create_playlists(plan: dict[str, list[Song]]) -> None:
    """Step 4: create/populate playlists in Apple Music from the plan.

    Depends on: Part C's music.sync_playlists_to_music().
    """
    sync_playlists_to_music(plan)
    logger.info("Synced %d playlists to Apple Music", len(plan))


if __name__ == "__main__":
    # Simple CLI smoke-test path; the GUI (ui.py) and web app
    # (webapp/server.py) call the functions above directly.
    songs = run_export_library()
    results = run_analyze_songs()
    plan = run_build_playlist_plan(results)
    run_create_playlists(plan)
