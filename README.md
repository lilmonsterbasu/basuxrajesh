# Playlist AI

Automatically organize your Apple Music library into intelligent playlists
using AI. macOS only, no Apple Developer account required — talks to
Apple Music via AppleScript.

## Project layout & ownership

| Part | Owner | Files |
|------|-------|-------|
| A — Data Export Layer | you | `applescript/export_library.scpt`, `music.py` (export functions), `Song`/`Playlist` dataclasses |
| B — AI Classification | Rajesh | `classifier.py`, `prompts.py` |
| C — Playlist Logic + Write-back | you | `applescript/create_playlist.scpt`, `applescript/add_song_to_playlist.scpt`, `music.py` (grouping/write-back functions) |
| D — GUI + Web App Design | Rajesh | `ui.py` (Tkinter), `webapp/` (FastAPI + frontend) |

`app.py` glues A → B → C together and is what `ui.py` and `webapp/server.py`
both call into. Every part's public functions are stubbed with
`NotImplementedError` and a `TODO(Part X)` docstring — grep for your letter
to find your work.

Shared JSON contracts (all under `data/`, gitignored):
- `library_export.json` — Part A's output, Part B's input
- `classified_songs.json` — Part B's output, Part C's input
- `playlist_plan.json` — Part C's grouping output (human-readable preview)

## Installation

Requires macOS with Apple Music (Music.app) and Python 3.12+.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set your Claude API key:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Permissions

### AppleScript / Automation permissions

The first time the app runs an AppleScript command against Music.app,
macOS will prompt:

> "Terminal" (or your IDE) wants access to control "Music.app"

Click **OK**. If you miss it or need to reset it:

**System Settings → Privacy & Security → Automation** → find your
terminal/IDE → enable the **Music** checkbox.

If AppleScript calls silently fail or hang, check that permission first —
it's the most common failure mode.

### Apple Music

No API keys or developer account needed for Music.app access — AppleScript
talks to the already-running (or auto-launched) local app directly.

## Running the app

Desktop GUI:

```bash
python ui.py
```

Web dashboard (companion, optional):

```bash
uvicorn webapp.server:api --reload
```

Then open `http://localhost:8000`.

CLI smoke test (runs the full pipeline once, no GUI):

```bash
python app.py
```

## Workflow

1. **Export Library** — reads every playlist and song from Apple Music into `data/library_export.json`
2. **Analyze Songs** — sends each song to the configured LLM, gets back genre/energy/mood/contexts, saves `data/classified_songs.json`
3. **Preview Results** — shows the proposed playlist → songs groupings before touching Apple Music
4. **Create Playlists** — creates any missing playlists and adds songs, skipping duplicates

## Adding another LLM provider

All classifiers implement the `Classifier` protocol in `classifier.py`:

```python
class Classifier(Protocol):
    def classify(self, song: Song) -> ClassificationResult: ...
    def classify_batch(self, songs: list[Song]) -> list[ClassificationResult]: ...
```

To add a new provider:

1. Subclass `BaseClassifier` in `classifier.py` (see `OpenAIClassifier`,
   `GeminiClassifier`, `OllamaClassifier` stubs for the pattern).
2. Implement `__init__` (read API key/config) and `classify()` (build
   the prompt via `prompts.build_classification_prompt`, call the
   provider's API, parse the JSON response into a `ClassificationResult`).
3. Register it in `get_classifier()`'s provider-name dispatch.
4. Nothing else in the codebase needs to change — `app.py`, `ui.py`, and
   `webapp/server.py` all go through `get_classifier(provider=...)`.
