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
from pathlib import Path

from classifier import ClassificationResult, get_classifier, save_classified_songs
from music import (
    Song,
    build_playlists,
    export_library,
    load_library_export,
    save_library_export,
    sync_playlists_to_music,
)

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


def run_analyze_songs(provider: str = "claude") -> list[ClassificationResult]:
    """Step 2: classify every exported song with the configured LLM provider.

    Depends on: Part B's get_classifier() / Classifier.classify_batch().
    """
    songs = load_library_export(LIBRARY_EXPORT_PATH)
    classifier = get_classifier(provider=provider)
    results = classifier.classify_batch(songs)
    save_classified_songs(results, CLASSIFIED_SONGS_PATH)
    logger.info("Classified %d songs using provider=%s", len(results), provider)
    return results


def run_build_playlist_plan(classified_songs: list[ClassificationResult]) -> dict[str, list[Song]]:
    """Step 3: group classified songs into playlist -> songs mapping.

    Depends on: Part C's music.build_playlists().
    """
    plan = build_playlists(classified_songs)
    with PLAYLIST_PLAN_PATH.open("w") as f:
        json.dump(
            {name: [s.title for s in songs] for name, songs in plan.items()},
            f,
            indent=2,
        )
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
