"""music.py

OWNER: Part A (export/read side) + Part C (playlist write-back side).
Split clearly marked below so both people can work in this file without
colliding -- consider splitting into music_export.py / music_write.py if
merge conflicts become a problem.

This module is the only place that shells out to AppleScript. Everything
else in the project should go through the functions defined here.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

APPLESCRIPT_DIR = Path(__file__).parent / "applescript"


@dataclass
class Song:
    """A single track from the Apple Music library.

    OWNER: Part A defines this; Part B/C/D just import and use it.
    """

    title: str
    artist: str
    album: str
    genre: str
    year: int
    play_count: int
    duration: float  # seconds

    def key(self) -> tuple[str, str]:
        """Identity used for de-duplication (title, artist)."""
        return (self.title.strip().lower(), self.artist.strip().lower())


@dataclass
class Playlist:
    """A playlist as read from Apple Music (existing playlists)."""

    name: str
    song_keys: list[tuple[str, str]] = field(default_factory=list)


class AppleScriptError(RuntimeError):
    """Raised when an osascript invocation fails or returns bad data."""


def _run_applescript(script_name: str, *args: str) -> str:
    """Run a .scpt file via osascript and return its stdout, stripped.

    OWNER: Part A. Shared helper used by both export and write-back code.

    Raises:
        AppleScriptError: if osascript exits non-zero.
    """
    script_path = APPLESCRIPT_DIR / script_name
    try:
        result = subprocess.run(
            ["osascript", str(script_path), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.error("AppleScript %s failed: %s", script_name, exc.stderr)
        raise AppleScriptError(f"{script_name} failed: {exc.stderr}") from exc
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Part A: export / read side
# ---------------------------------------------------------------------------


def export_library() -> tuple[list[Song], list[Playlist]]:
    """Run export_library.scpt, parse the JSON, return (songs, playlists).

    TODO(Part A):
    - Call _run_applescript("export_library.scpt")
    - json.loads the result
    - Build Song objects for each entry in result["songs"]
    - Build Playlist objects for each entry in result["playlists"],
      resolving song_titles -> song keys (title, artist) as needed
    - Cache/save to library_export.json (see save_library_export)
    """
    raise NotImplementedError("Part A: implement export_library()")


def save_library_export(songs: list[Song], path: Path) -> None:
    """Serialize songs to library_export.json for Part B to consume.

    TODO(Part A): decide final JSON shape and keep it in sync with
    classifier.py's expectations (list of song dicts).
    """
    raise NotImplementedError("Part A: implement save_library_export()")


def load_library_export(path: Path) -> list[Song]:
    """Load a previously exported library_export.json back into Song objects.

    TODO(Part A): mirror save_library_export()'s shape.
    """
    raise NotImplementedError("Part A: implement load_library_export()")


# ---------------------------------------------------------------------------
# Part C: playlist grouping + write-back side
# ---------------------------------------------------------------------------


def build_playlists(
    classified_songs: list[dict],
) -> dict[str, list[Song]]:
    """Group classified songs into playlists by their assigned contexts.

    Input: list of {"song": Song, "genre": str, "energy": int, "mood": str,
                     "contexts": list[str]} (see classifier.ClassificationResult)

    A song with contexts ["Coding", "Late Night"] must appear in both the
    "Coding" and "Late Night" playlist groups.

    TODO(Part C):
    - Iterate classified_songs, for each context tag append the song to
      that playlist's list
    - De-duplicate within each playlist using Song.key()
    - Return dict[playlist_name, list[Song]]
    """
    raise NotImplementedError("Part C: implement build_playlists()")


def create_playlist_if_missing(name: str) -> str:
    """Create a playlist in Apple Music if it doesn't already exist.

    Returns "created" or "skipped".

    TODO(Part C): call _run_applescript("create_playlist.scpt", name)
    """
    raise NotImplementedError("Part C: implement create_playlist_if_missing()")


def add_song_to_playlist(playlist_name: str, song: Song) -> str:
    """Add a single song to a playlist, avoiding duplicates.

    Returns "added", "duplicate", or "not_found".

    TODO(Part C): call _run_applescript(
        "add_song_to_playlist.scpt", playlist_name, song.title, song.artist
    )
    """
    raise NotImplementedError("Part C: implement add_song_to_playlist()")


def sync_playlists_to_music(plan: dict[str, list[Song]]) -> None:
    """Given a full playlist plan, create missing playlists and populate them.

    TODO(Part C):
    - For each (playlist_name, songs) in plan.items():
        - create_playlist_if_missing(playlist_name)
        - for song in songs: add_song_to_playlist(playlist_name, song)
    - Log progress (created/skipped/added/duplicate counts) for the GUI
      console output to display.
    """
    raise NotImplementedError("Part C: implement sync_playlists_to_music()")
