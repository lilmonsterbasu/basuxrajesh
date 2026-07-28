"""Tests for the Part D web dashboard API.

Uses FastAPI's TestClient against temp data files, so nothing here touches
Apple Music, the Claude API, or the real data/ directory.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest
from fastapi.testclient import TestClient

import app as orchestrator
from classifier import ClassificationResult, save_classified_songs
from webapp.server import api


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Point the orchestrator's data files at a temp directory."""
    monkeypatch.setattr(orchestrator, "LIBRARY_EXPORT_PATH", tmp_path / "library.json")
    monkeypatch.setattr(orchestrator, "CLASSIFIED_SONGS_PATH", tmp_path / "classified.json")
    monkeypatch.setattr(orchestrator, "PLAYLIST_PLAN_PATH", tmp_path / "plan.json")
    return TestClient(api)


def _write_library(songs) -> None:
    orchestrator.save_library_export(songs, orchestrator.LIBRARY_EXPORT_PATH)


def _write_analysis(songs) -> list[ClassificationResult]:
    results = [
        ClassificationResult(
            song=song,
            genre=song.genre,
            energy=5,
            mood="Chill",
            contexts=["Coding"] if i % 2 == 0 else ["Party"],
        )
        for i, song in enumerate(songs)
    ]
    save_classified_songs(results, orchestrator.CLASSIFIED_SONGS_PATH)
    return results


# --------------------------------------------------------------------------
# /api/stats -- the home page's headline numbers
# --------------------------------------------------------------------------


def test_stats_reports_nothing_before_export(client):
    """The home page must be able to tell "no data" from "zero songs"."""
    stats = client.get("/api/stats").json()

    assert stats["exported"] is False
    assert stats["analyzed"] is False
    assert stats == {**stats, "songs": 0, "artists": 0, "playlists": 0}


def test_stats_counts_songs_and_unique_artists(client, songs):
    _write_library(songs)
    stats = client.get("/api/stats").json()

    assert stats["exported"] is True
    assert stats["analyzed"] is False
    assert stats["songs"] == 15
    # Aphex Twin appears twice in the fixture, so artists < songs.
    assert stats["artists"] == 14


def test_stats_artist_count_ignores_case_and_padding(client, songs):
    from music import Song

    duped = songs[:2] + [
        Song(title="Another Track", artist=f"  {songs[0].artist.upper()} ",
             album="x", genre="x", year=2020, play_count=0, duration=1.0)
    ]
    _write_library(duped)

    assert client.get("/api/stats").json()["artists"] == 2


def test_stats_includes_playlists_once_analyzed(client, songs):
    _write_library(songs)
    results = _write_analysis(songs)
    orchestrator.run_build_playlist_plan(results)

    stats = client.get("/api/stats").json()
    assert stats["analyzed"] is True
    assert stats["analyzed_songs"] == 15
    assert stats["playlists"] == 2  # Coding + Party


# --------------------------------------------------------------------------
# Plan editing -- the drag-and-drop round trip
# --------------------------------------------------------------------------


def test_plan_survives_an_edit_round_trip(client, songs):
    _write_library(songs)
    results = _write_analysis(songs)
    orchestrator.run_build_playlist_plan(results)

    plan = client.get("/api/playlist-plan").json()
    coding = next(p for p in plan["playlists"] if p["name"] == "Coding")
    party = next(p for p in plan["playlists"] if p["name"] == "Party")
    moved = coding["songs"].pop(0)
    party["songs"].append(moved)

    saved = client.post("/api/playlist-plan", json=plan).json()
    assert saved["saved"] is True

    # It must come back as real Song objects -- that is Part C's input.
    reloaded = orchestrator.load_playlist_plan()
    assert moved["title"] in [s.title for s in reloaded["Party"]]
    assert moved["title"] not in [s.title for s in reloaded["Coding"]]


def test_plan_rejects_a_blank_playlist_name(client, songs):
    payload = {"playlists": [{"name": "  ", "songs": [asdict(songs[0])]}]}
    response = client.post("/api/playlist-plan", json=payload)

    assert response.status_code == 400
    assert "name" in response.json()["detail"].lower()


def test_plan_rejects_a_malformed_song(client):
    payload = {"playlists": [{"name": "Coding", "songs": [{"title": "only a title"}]}]}
    response = client.post("/api/playlist-plan", json=payload)

    assert response.status_code == 400
    assert "Coding" in response.json()["detail"]


def test_creating_playlists_without_a_plan_is_refused(client):
    """Better a clear 409 than an AppleScript run against an empty plan."""
    response = client.post("/api/create-playlists")

    assert response.status_code == 409
    assert "preview" in response.json()["detail"].lower()


# --------------------------------------------------------------------------
# Library / analysis views
# --------------------------------------------------------------------------


def test_library_is_empty_but_valid_before_export(client):
    body = client.get("/api/library").json()
    assert body == {"songs": [], "exported": False}


def test_analysis_returns_classification_per_song(client, songs):
    _write_library(songs)
    _write_analysis(songs)

    body = client.get("/api/analysis").json()
    assert body["analyzed"] is True
    first = body["songs"][0]
    assert first["title"] == songs[0].title
    assert first["classification"]["contexts"] == ["Coding"]


def test_unknown_job_id_is_a_404(client):
    assert client.get("/api/jobs/nope").status_code == 404


def test_static_index_is_served(client):
    """The dashboard itself must be reachable at /."""
    response = client.get("/")

    assert response.status_code == 200
    assert "Discover Your Sound" in response.text
    assert 'id="home-grid"' in response.text
