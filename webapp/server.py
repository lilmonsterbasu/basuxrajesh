"""webapp/server.py

OWNER: Part D (GUI + Web App Design) -- web dashboard half.

A thin FastAPI backend that reuses app.py's orchestration functions (which
in turn reuse music.py / classifier.py from Parts A/B/C) to serve a browser
dashboard: view library, view AI classifications, edit playlist groupings,
trigger playlist creation.

This is a companion surface to the Tkinter app, not a replacement --
both call into the same app.py functions.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# TODO(Part D):
# from fastapi import FastAPI
# from fastapi.staticfiles import StaticFiles
# import app as orchestrator
#
# api = FastAPI(title="Playlist AI")
#
# @api.get("/api/library")
# def get_library():
#     """Return the cached library_export.json contents."""
#     ...
#
# @api.post("/api/export")
# def export_library():
#     """Trigger orchestrator.export_library(), return songs."""
#     ...
#
# @api.post("/api/analyze")
# def analyze_songs():
#     """Trigger orchestrator.analyze_songs(), return classified songs."""
#     ...
#
# @api.get("/api/playlist-plan")
# def get_playlist_plan():
#     """Return dict[playlist_name, list[Song]] preview, with support for
#     the frontend to let the user edit groupings before confirming."""
#     ...
#
# @api.post("/api/create-playlists")
# def create_playlists():
#     """Trigger orchestrator.sync_playlists_to_music()."""
#     ...
#
# api.mount("/", StaticFiles(directory="webapp/static", html=True), name="static")


def main() -> None:
    """Run with: uvicorn webapp.server:api --reload"""
    raise NotImplementedError("Part D: wire up FastAPI app and static frontend")


if __name__ == "__main__":
    main()
