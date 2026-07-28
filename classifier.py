"""classifier.py

OWNER: Part B (AI Classification)

Defines the Classifier interface and concrete provider implementations.
Any provider (Claude, OpenAI, Gemini, local Ollama) must implement the
same `Classifier` protocol so app.py / ui.py can swap providers via
config without touching calling code.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol

from music import Song
from prompts import (
    BATCH_SCHEMA,
    CLASSIFICATION_SCHEMA,
    CONTEXT_VOCABULARY,
    SYSTEM_PROMPT,
    build_batch_prompt,
    build_classification_prompt,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
CACHE_PATH = DATA_DIR / "classification_cache.json"

# Progress callback signature: on_progress(done, total).
ProgressCallback = Callable[[int, int], None]

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_CONTEXT_LOOKUP = {tag.lower(): tag for tag in CONTEXT_VOCABULARY}
_MAX_CONTEXTS = 3


class ClassificationError(RuntimeError):
    """Raised when a provider response can't be turned into a result."""


@dataclass
class ClassificationResult:
    """Structured output of classifying one song."""

    song: Song
    genre: str
    energy: int
    mood: str
    contexts: list[str]


class Classifier(Protocol):
    """Common interface every LLM provider adapter must implement.

    This is the single source of truth for what "a classifier" means in this
    project. app.py/ui.py should type against this Protocol, never a concrete
    class.
    """

    def classify(self, song: Song) -> ClassificationResult:
        """Classify a single song. Must return a ClassificationResult."""
        ...

    def classify_batch(
        self,
        songs: list[Song],
        on_progress: ProgressCallback | None = None,
    ) -> list[ClassificationResult]:
        """Classify many songs (default: loop over classify()).

        Override in a subclass if the provider supports true batch calls.
        """
        ...


class ClassificationCache:
    """Disk-backed map of Song.key() -> classification fields.

    Classifying a 2,000-song library is the expensive step of the whole
    pipeline, and re-running it after adding ten songs should not re-pay for
    the other 1,990. Keyed on (title, artist) so it survives re-exports.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else CACHE_PATH
        self._entries: dict[str, dict] = {}
        self._load()

    @staticmethod
    def _key(song: Song) -> str:
        return "".join(song.key())

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open() as f:
                self._entries = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable cache at %s: %s", self.path, exc)
            self._entries = {}

    def get(self, song: Song) -> dict | None:
        return self._entries.get(self._key(song))

    def put(self, song: Song, fields: dict) -> None:
        self._entries[self._key(song)] = fields

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w") as f:
            json.dump(self._entries, f, indent=2)


class BaseClassifier:
    """Shared helpers for concrete classifier implementations."""

    def classify(self, song: Song) -> ClassificationResult:  # pragma: no cover
        raise NotImplementedError

    def classify_batch(
        self,
        songs: list[Song],
        on_progress: ProgressCallback | None = None,
    ) -> list[ClassificationResult]:
        """Classify songs one at a time, reporting progress as we go.

        A song that fails to classify is logged and skipped rather than
        aborting the run, so one bad response can't kill a 500-song job. The
        returned list may therefore be shorter than `songs`.
        """
        results: list[ClassificationResult] = []
        total = len(songs)
        for i, song in enumerate(songs, start=1):
            try:
                results.append(self.classify(song))
            except Exception as exc:  # noqa: BLE001 -- one song must not kill the run
                logger.warning(
                    "Skipping %s by %s: %s", song.title, song.artist, exc
                )
            if on_progress is not None:
                on_progress(i, total)
        return results

    @staticmethod
    def _parse_response(raw_text: str) -> dict:
        """Strip markdown fences and json.loads() the model's output.

        Structured outputs make fenced responses unlikely on Claude, but the
        other providers have no equivalent, so every adapter goes through here.
        """
        cleaned = _FENCE_RE.sub("", raw_text.strip())
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            preview = cleaned[:200]
            raise ClassificationError(
                f"Model did not return valid JSON ({exc}). Got: {preview!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ClassificationError(
                f"Expected a JSON object, got {type(parsed).__name__}"
            )
        return parsed

    @classmethod
    def _normalize(cls, fields: dict, song: Song) -> ClassificationResult:
        """Validate and coerce one raw result dict into a ClassificationResult.

        The API's schema subset can't express "integer between 1 and 10" or
        "at most 3 items", so those bounds are enforced here.
        """
        missing = {"genre", "energy", "mood", "contexts"} - fields.keys()
        if missing:
            raise ClassificationError(
                f"Response missing required field(s): {sorted(missing)}"
            )

        try:
            energy = int(round(float(fields["energy"])))
        except (TypeError, ValueError) as exc:
            raise ClassificationError(
                f"energy must be a number, got {fields['energy']!r}"
            ) from exc
        energy = max(1, min(10, energy))

        raw_contexts = fields["contexts"]
        if not isinstance(raw_contexts, list):
            raise ClassificationError(
                f"contexts must be a list, got {type(raw_contexts).__name__}"
            )

        contexts: list[str] = []
        for tag in raw_contexts:
            canonical = _CONTEXT_LOOKUP.get(str(tag).strip().lower())
            if canonical is None:
                logger.debug(
                    "Dropping unknown context %r for %s", tag, song.title
                )
            elif canonical not in contexts:
                contexts.append(canonical)
        contexts = contexts[:_MAX_CONTEXTS]
        if not contexts:
            logger.warning(
                "%s by %s got no usable contexts; it will not be playlisted",
                song.title,
                song.artist,
            )

        return ClassificationResult(
            song=song,
            genre=str(fields["genre"]).strip(),
            energy=energy,
            mood=str(fields["mood"]).strip(),
            contexts=contexts,
        )

    @staticmethod
    def _result_fields(result: ClassificationResult) -> dict:
        """The cacheable part of a result (everything except the song)."""
        return {
            "genre": result.genre,
            "energy": result.energy,
            "mood": result.mood,
            "contexts": list(result.contexts),
        }


class ClaudeClassifier(BaseClassifier):
    """Classifier backed by the Anthropic Claude API.

    Two things make this usable on a real library rather than a demo:

    - Batching. One song per API call means a 2,000-song library is 2,000
      calls; `batch_size` songs per call is a ~20x cost and latency win.
    - Caching. Results are keyed on (title, artist) in
      data/classification_cache.json, so re-runs only pay for new songs.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-opus-5",
        batch_size: int = 20,
        max_tokens: int = 8000,
        effort: str = "low",
        use_cache: bool = True,
        cache_path: Path | None = None,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover -- environment problem
            raise ClassificationError(
                "The `anthropic` package is required for the Claude provider. "
                "Run: pip install -r requirements.txt"
            ) from exc

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ClassificationError(
                "No Claude API key found. Set ANTHROPIC_API_KEY in your "
                "environment, or pass api_key= explicitly."
            )

        self._anthropic = anthropic
        # The SDK retries 429s and 5xx with exponential backoff itself; 5 is a
        # friendlier ceiling than the default 2 for a long batch run.
        self._client = anthropic.Anthropic(api_key=key, max_retries=5)
        self.model = model
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.effort = effort
        self._cache = ClassificationCache(cache_path) if use_cache else None

    # -- API plumbing -------------------------------------------------------

    def _call(self, prompt: str, schema: dict) -> dict:
        """Send one request and return the parsed JSON object.

        `output_config.format` constrains the response to the given schema, so
        the model can't wrap the JSON in prose. Adaptive thinking at low effort
        is the recommended setting for routine classification -- disabling
        thinking on this model risks `<thinking>` tags leaking into the output.
        """
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                thinking={"type": "adaptive"},
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": schema},
                },
                messages=[{"role": "user", "content": prompt}],
            )
        except self._anthropic.APIStatusError as exc:
            raise ClassificationError(
                f"Claude API error ({exc.status_code}): {exc.message}"
            ) from exc
        except self._anthropic.APIConnectionError as exc:
            raise ClassificationError(f"Could not reach the Claude API: {exc}") from exc

        if response.stop_reason == "refusal":
            raise ClassificationError("Claude declined to classify this request.")
        if response.stop_reason == "max_tokens":
            raise ClassificationError(
                "Response hit max_tokens and was truncated. Lower batch_size "
                "or raise max_tokens."
            )

        # With thinking on, content[0] is a thinking block -- find the text.
        text = next(
            (block.text for block in response.content if block.type == "text"), None
        )
        if text is None:
            raise ClassificationError("Claude returned no text content.")
        return self._parse_response(text)

    # -- Classifier interface ----------------------------------------------

    def classify(self, song: Song) -> ClassificationResult:
        """Classify a single song (one API call)."""
        if self._cache is not None:
            cached = self._cache.get(song)
            if cached is not None:
                return self._normalize(cached, song)

        fields = self._call(build_classification_prompt(song), CLASSIFICATION_SCHEMA)
        result = self._normalize(fields, song)
        self._remember(result)
        return result

    def classify_batch(
        self,
        songs: list[Song],
        on_progress: ProgressCallback | None = None,
    ) -> list[ClassificationResult]:
        """Classify songs in batches, serving what it can from the cache.

        Songs that fail even after a per-song retry are logged and skipped, so
        the returned list may be shorter than `songs`.
        """
        total = len(songs)
        done = 0
        results: list[ClassificationResult] = []
        pending: list[Song] = []

        def report() -> None:
            if on_progress is not None:
                on_progress(done, total)

        for song in songs:
            cached = self._cache.get(song) if self._cache is not None else None
            if cached is None:
                pending.append(song)
                continue
            try:
                results.append(self._normalize(cached, song))
            except ClassificationError:
                pending.append(song)  # stale/corrupt entry -- just re-classify
                continue
            done += 1
            report()

        if pending:
            logger.info(
                "%d/%d songs served from cache; classifying %d",
                total - len(pending),
                total,
                len(pending),
            )

        for start in range(0, len(pending), self.batch_size):
            chunk = pending[start : start + self.batch_size]
            chunk_results = self._classify_chunk(chunk)
            results.extend(chunk_results)
            done += len(chunk)
            report()

        self._flush()
        return results

    # -- internals ----------------------------------------------------------

    def _classify_chunk(self, chunk: list[Song]) -> list[ClassificationResult]:
        """Classify one chunk, falling back to per-song calls on failure."""
        try:
            payload = self._call(build_batch_prompt(chunk), BATCH_SCHEMA)
            return self._results_from_batch(payload, chunk)
        except ClassificationError as exc:
            logger.warning(
                "Batch of %d failed (%s); retrying song by song", len(chunk), exc
            )

        results: list[ClassificationResult] = []
        for song in chunk:
            try:
                results.append(self.classify(song))
            except Exception as exc:  # noqa: BLE001 -- one song must not kill the run
                logger.warning("Skipping %s by %s: %s", song.title, song.artist, exc)
        return results

    def _results_from_batch(
        self, payload: dict, chunk: list[Song]
    ) -> list[ClassificationResult]:
        """Match batch entries back to songs via the echoed `index` field."""
        entries = payload.get("results")
        if not isinstance(entries, list):
            raise ClassificationError("Batch response had no `results` list.")

        results: list[ClassificationResult] = []
        seen: set[int] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            index = entry.get("index")
            if not isinstance(index, int) or not 0 <= index < len(chunk):
                logger.warning("Dropping batch entry with bad index %r", index)
                continue
            if index in seen:
                logger.warning("Dropping duplicate batch entry for index %d", index)
                continue
            song = chunk[index]
            try:
                result = self._normalize(entry, song)
            except ClassificationError as exc:
                logger.warning("Skipping %s by %s: %s", song.title, song.artist, exc)
                continue
            seen.add(index)
            results.append(result)
            self._remember(result)

        missing = [chunk[i] for i in range(len(chunk)) if i not in seen]
        for song in missing:
            logger.warning(
                "Batch response omitted %s by %s", song.title, song.artist
            )
        return results

    def _remember(self, result: ClassificationResult) -> None:
        if self._cache is not None:
            self._cache.put(result.song, self._result_fields(result))

    def _flush(self) -> None:
        if self._cache is not None:
            self._cache.save()


class OpenAIClassifier(BaseClassifier):
    """Stub adapter for OpenAI models. Same interface as ClaudeClassifier.

    TODO(Part B, optional/stretch): implement using the OpenAI SDK. Reuse
    BaseClassifier._parse_response() and ._normalize() -- only the API call
    itself should differ.
    """

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o") -> None:
        raise NotImplementedError("Part B (stretch): implement OpenAIClassifier")

    def classify(self, song: Song) -> ClassificationResult:
        raise NotImplementedError("Part B (stretch): implement OpenAIClassifier.classify()")


class GeminiClassifier(BaseClassifier):
    """Stub adapter for Google Gemini models. Same interface as ClaudeClassifier.

    TODO(Part B, optional/stretch): implement using the google-genai SDK.
    """

    def __init__(self, api_key: str | None = None, model: str = "gemini-1.5-pro") -> None:
        raise NotImplementedError("Part B (stretch): implement GeminiClassifier")

    def classify(self, song: Song) -> ClassificationResult:
        raise NotImplementedError("Part B (stretch): implement GeminiClassifier.classify()")


class OllamaClassifier(BaseClassifier):
    """Stub adapter for local Ollama models. Same interface as ClaudeClassifier.

    TODO(Part B, optional/stretch): implement via local HTTP call to
    http://localhost:11434/api/generate or the ollama python package.
    """

    def __init__(self, model: str = "llama3") -> None:
        raise NotImplementedError("Part B (stretch): implement OllamaClassifier")

    def classify(self, song: Song) -> ClassificationResult:
        raise NotImplementedError("Part B (stretch): implement OllamaClassifier.classify()")


_PROVIDERS: dict[str, type] = {
    "claude": ClaudeClassifier,
    "openai": OpenAIClassifier,
    "gemini": GeminiClassifier,
    "ollama": OllamaClassifier,
}


def get_classifier(provider: str = "claude", **kwargs) -> Classifier:
    """Factory: return a configured Classifier instance by provider name.

    This is the single function app.py/ui.py should call to obtain a
    classifier -- it is how the "swap providers" requirement is satisfied.
    """
    try:
        cls = _PROVIDERS[provider.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown provider {provider!r}. Expected one of: "
            f"{', '.join(sorted(_PROVIDERS))}"
        ) from None
    return cls(**kwargs)


def save_classified_songs(results: list[ClassificationResult], path) -> None:
    """Serialize classification results to classified_songs.json for Part C.

    Shape (agreed with Part C -- keep in sync with music.build_playlists()):
    [
      {"song": {...Song fields...}, "genre": ..., "energy": ...,
       "mood": ..., "contexts": [...]}
    ]
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump([asdict(result) for result in results], f, indent=2)
    logger.info("Wrote %d classified songs to %s", len(results), path)


def load_classified_songs(path) -> list[ClassificationResult]:
    """Read classified_songs.json back into ClassificationResult objects.

    Mirrors save_classified_songs(). Part D's "Preview Results" button uses
    this so it can build a playlist plan without re-running classification.
    """
    with Path(path).open() as f:
        payload = json.load(f)
    return [
        ClassificationResult(
            song=Song(**entry["song"]),
            genre=entry["genre"],
            energy=entry["energy"],
            mood=entry["mood"],
            contexts=list(entry["contexts"]),
        )
        for entry in payload
    ]


if __name__ == "__main__":
    # Dev entry point: classify a library_export.json without needing Part A.
    #   python classifier.py tests/fixtures/library_export.json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    source = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/library_export.json")
    with source.open() as f:
        library = [Song(**entry) for entry in json.load(f)]

    try:
        classifier = get_classifier("claude")
    except ClassificationError as exc:
        sys.exit(f"error: {exc}")

    classified = classifier.classify_batch(
        library,
        on_progress=lambda done, total: print(f"  {done}/{total}", flush=True),
    )
    for result in classified:
        print(
            f"{result.song.title} by {result.song.artist}\n"
            f"    {result.genre} | energy {result.energy} | {result.mood} "
            f"| {', '.join(result.contexts)}"
        )
