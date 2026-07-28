"""webapp/server.py

OWNER: Part D (GUI + Web App Design) -- web dashboard half.

A thin FastAPI backend that reuses app.py's orchestration functions (which in
turn reuse music.py / classifier.py from Parts A/B/C) to serve a browser
dashboard: view library, view AI classifications, edit playlist groupings,
trigger playlist creation.

This is a companion surface to the Tkinter app, not a replacement -- both call
into the same app.py functions.

Run with:  uvicorn webapp.server:api --reload
"""

from __future__ import annotations

import logging
import sys
import threading
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Let `uvicorn webapp.server:api` find app.py / music.py at the repo root no
# matter which directory uvicorn was launched from.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import app as orchestrator  # noqa: E402
from music import AppleScriptError, Song  # noqa: E402

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

api = FastAPI(title="Playlist AI")


# ---------------------------------------------------------------------------
# Background jobs
#
# Export and classification both take minutes. Doing that work inside the
# request would hold the HTTP connection open until it times out, so each one
# starts a job and the browser polls /api/jobs/{id} for progress and logs.
# ---------------------------------------------------------------------------


@dataclass
class Job:
    id: str
    kind: str
    status: str = "running"  # running | done | error
    done: int = 0
    total: int = 0
    logs: list[str] = field(default_factory=list)
    error: str | None = None

    def log(self, message: str) -> None:
        self.logs.append(message)
        logger.info("[%s] %s", self.kind, message)


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()


def _start_job(kind: str, target) -> Job:
    """Run `target(job)` on a daemon thread and return the Job immediately."""
    job = Job(id=uuid.uuid4().hex[:12], kind=kind)
    with _jobs_lock:
        _jobs[job.id] = job

    def runner() -> None:
        try:
            target(job)
            job.status = "done"
        except Exception as exc:  # noqa: BLE001 -- report every failure to the UI
            job.status = "error"
            job.error = str(exc)
            job.log(f"ERROR: {exc}")
            logger.error("Job %s failed:\n%s", job.id, traceback.format_exc())

    threading.Thread(target=runner, daemon=True).start()
    return job


@api.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    """Poll a running job for status, progress, and console output."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No such job: {job_id}")
    return asdict(job)


# ---------------------------------------------------------------------------
# Library and analysis
# ---------------------------------------------------------------------------


@api.get("/api/library")
def get_library() -> dict:
    """Return the cached library_export.json contents."""
    try:
        songs = orchestrator.load_library_export(orchestrator.LIBRARY_EXPORT_PATH)
    except FileNotFoundError:
        return {"songs": [], "exported": False}
    return {"songs": [asdict(song) for song in songs], "exported": True}


@api.post("/api/export")
def export_library() -> dict:
    """Trigger orchestrator.run_export_library() as a background job."""

    def work(job: Job) -> None:
        job.log("Reading your library from Apple Music...")
        songs = orchestrator.run_export_library()
        job.total = job.done = len(songs)
        job.log(f"Exported {len(songs)} songs.")

    return {"job_id": _start_job("export", work).id}


@api.get("/api/analysis")
def get_analysis() -> dict:
    """Return the cached classified_songs.json contents."""
    try:
        results = orchestrator.load_analysis()
    except FileNotFoundError:
        return {"songs": [], "analyzed": False}
    return {
        "songs": [
            {
                **asdict(result.song),
                "classification": {
                    "genre": result.genre,
                    "energy": result.energy,
                    "mood": result.mood,
                    "contexts": result.contexts,
                },
            }
            for result in results
        ],
        "analyzed": True,
    }


@api.post("/api/analyze")
def analyze_songs() -> dict:
    """Trigger orchestrator.run_analyze_songs() as a background job."""

    def work(job: Job) -> None:
        job.log("Classifying songs...")

        def on_progress(done: int, total: int) -> None:
            job.done, job.total = done, total

        results = orchestrator.run_analyze_songs(on_progress=on_progress)
        job.log(f"Classified {len(results)} songs.")

    return {"job_id": _start_job("analyze", work).id}


# ---------------------------------------------------------------------------
# Playlist plan
# ---------------------------------------------------------------------------


def _serialize_plan(plan: dict[str, list[Song]]) -> dict:
    return {
        "playlists": [
            {"name": name, "songs": [asdict(song) for song in plan[name]]}
            for name in sorted(plan)
        ]
    }


@api.get("/api/playlist-plan")
def get_playlist_plan(rebuild: bool = False) -> dict:
    """Return the playlist -> songs preview.

    By default this serves the saved plan, so drag-and-drop edits survive a
    page reload. `?rebuild=true` regenerates it from the latest
    classifications, discarding those edits.
    """
    if not rebuild:
        try:
            return _serialize_plan(orchestrator.load_playlist_plan())
        except FileNotFoundError:
            pass

    try:
        results = orchestrator.load_analysis()
    except FileNotFoundError:
        raise HTTPException(
            status_code=409,
            detail="No analysis yet -- run Export Library, then Analyze Songs.",
        ) from None
    return _serialize_plan(orchestrator.run_build_playlist_plan(results))


class PlaylistPayload(BaseModel):
    name: str
    songs: list[dict]


class PlanPayload(BaseModel):
    playlists: list[PlaylistPayload]


@api.post("/api/playlist-plan")
def save_playlist_plan(payload: PlanPayload) -> dict:
    """Persist the user's edited groupings before they confirm creation."""
    plan: dict[str, list[Song]] = {}
    for entry in payload.playlists:
        name = entry.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Playlist name cannot be empty.")
        try:
            plan[name] = [Song(**row) for row in entry.songs]
        except TypeError as exc:
            raise HTTPException(
                status_code=400, detail=f"Malformed song in '{name}': {exc}"
            ) from exc

    orchestrator.save_playlist_plan(plan)
    placements = sum(len(songs) for songs in plan.values())
    return {"saved": True, "playlists": len(plan), "placements": placements}


@api.post("/api/create-playlists")
def create_playlists() -> dict:
    """Trigger orchestrator.run_create_playlists() on the saved plan."""
    try:
        plan = orchestrator.load_playlist_plan()
    except FileNotFoundError:
        raise HTTPException(
            status_code=409,
            detail="No playlist plan saved yet -- preview one first.",
        ) from None

    def work(job: Job) -> None:
        job.total = len(plan)
        job.log(f"Creating {len(plan)} playlists in Apple Music...")
        orchestrator.run_create_playlists(plan)
        job.done = len(plan)
        job.log("Done. Check Apple Music for your new playlists.")

    return {"job_id": _start_job("create", work).id}


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@api.exception_handler(AppleScriptError)
def applescript_error_handler(_request: Any, exc: AppleScriptError) -> JSONResponse:
    """Turn AppleScript failures into a readable error instead of a traceback.

    This is nearly always the macOS Automation permission -- see the README.
    """
    return JSONResponse(
        status_code=502,
        content={
            "detail": (
                f"Apple Music automation failed: {exc}. Check System Settings > "
                "Privacy & Security > Automation and allow Music."
            )
        },
    )


# Mounted last: it claims "/", so every /api/* route must be registered above.
api.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def main() -> None:
    """Run with: uvicorn webapp.server:api --reload"""
    import uvicorn

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    uvicorn.run(api, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
