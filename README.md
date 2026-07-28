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
- `classification_cache.json` — Part B's internal cache, keyed on
  `(title, artist)`; delete it to force a full re-classification

`tests/fixtures/library_export.json` is a checked-in 15-song sample with the
same shape as Part A's output, so Parts B/C/D can be developed and tested
before the AppleScript export lands.

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

Classify the sample library without Apple Music or Part A (one real API call
per batch of 20 songs):

```bash
python classifier.py tests/fixtures/library_export.json
```

## Tests

```bash
pytest
```

Part B's tests run against `tests/fixtures/library_export.json` and a fake
Anthropic client — no API key, no network, no Apple Music.

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
    def classify_batch(
        self,
        songs: list[Song],
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[ClassificationResult]: ...
```

Four providers ship today:

| `provider=` | Backend | Needs |
|-------------|---------|-------|
| `claude` (default) | Anthropic Messages API | `ANTHROPIC_API_KEY` |
| `openai` | OpenAI chat completions | `OPENAI_API_KEY`, `pip install openai` |
| `gemini` | Google Gemini | `GEMINI_API_KEY`, `pip install google-genai` |
| `ollama` | Local Ollama server | `ollama serve` — no key, no cost |

Provider SDKs are imported lazily, so you only install the ones you use.

To add a fifth:

1. Subclass `BaseClassifier` in `classifier.py` (see `OpenAIClassifier` for
   the pattern).
2. Implement `__init__` (read API key/config) and `_classify_song(song)`,
   which builds the prompt via `prompts.build_classification_prompt` and
   returns the raw fields dict from the model. If the provider can handle
   several songs per request, also set `batch_size` and implement
   `_classify_songs(songs) -> {index: fields}`.
3. Register it in `get_classifier()`'s provider-name dispatch.
4. Nothing else in the codebase needs to change — `app.py`, `ui.py`, and
   `webapp/server.py` all go through `get_classifier(provider=...)`.

`BaseClassifier` supplies the rest for free: the `(title, artist)` cache,
chunking, progress callbacks, per-song error isolation, markdown-fence
stripping, and validation of energy bounds and the context vocabulary. That
last part matters most for providers without schema enforcement — Gemini and
Ollama return free-form JSON, and `_normalize()` is what stops an invented
context tag from becoming a stray Apple Music playlist.
