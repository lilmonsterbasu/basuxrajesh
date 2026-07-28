"""classifier.py

OWNER: Part B (AI Classification)

Defines the Classifier interface and concrete provider implementations.
Any provider (Claude, OpenAI, Gemini, local Ollama) must implement the
same `Classifier` protocol so app.py / ui.py can swap providers via
config without touching calling code.
"""

from __future__ import annotations

import importlib
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


class FatalClassificationError(ClassificationError):
    """An account-level problem that will fail identically for every song.

    Expired keys, revoked permissions, and exhausted credit balances are not
    per-song failures. Retrying a 2,000-song library one song at a time
    against a "credit balance too low" error just makes the user wait for
    2,000 identical rejections, so these abort the whole run immediately.
    """


#: Substrings that mark a provider error as account-level rather than per-song.
_FATAL_HINTS = (
    "credit balance",
    "billing",
    "quota",
    "insufficient_quota",
    "payment",
    "suspended",
    "invalid x-api-key",
    "invalid api key",
    "authentication",
)


def _is_fatal(message: str, status_code: int | None = None) -> bool:
    if status_code in (401, 403):
        return True
    lowered = message.lower()
    return any(hint in lowered for hint in _FATAL_HINTS)


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


def _require(module_name: str, install_hint: str):
    """Import an optional provider SDK, or explain how to install it.

    Provider SDKs are imported lazily so that using Claude doesn't require
    the OpenAI or Gemini packages to be installed, and vice versa.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ClassificationError(
            f"The `{module_name}` package is required for this provider. "
            f"Run: pip install {install_hint}"
        ) from exc


def _api_key(explicit: str | None, env_var: str, provider: str) -> str:
    key = explicit or os.environ.get(env_var)
    if not key:
        raise ClassificationError(
            f"No {provider} API key found. Set {env_var} in your environment, "
            "or pass api_key= explicitly."
        )
    return key


class BaseClassifier:
    """Shared machinery for every provider adapter.

    Subclasses implement `_classify_song()` (one song -> raw fields dict) and,
    if the provider can handle several songs per request, set `batch_size` and
    implement `_classify_songs()`. Caching, chunking, progress reporting, and
    per-song error isolation all live here so no provider has to repeat them.
    """

    #: Songs per request. 1 means "no batching" and only `_classify_song` is used.
    batch_size = 1

    def __init__(self, use_cache: bool = True, cache_path: Path | None = None) -> None:
        self._cache = ClassificationCache(cache_path) if use_cache else None

    # -- provider hooks -----------------------------------------------------

    def _classify_song(self, song: Song) -> dict:
        """Classify one song, returning the raw fields dict from the model."""
        raise NotImplementedError("Subclasses must implement _classify_song()")

    def _classify_songs(self, songs: list[Song]) -> dict[int, dict]:
        """Classify several songs at once, keyed by their index in `songs`.

        Only called when `batch_size` > 1.
        """
        raise NotImplementedError("This provider does not support batching")

    # -- Classifier interface ----------------------------------------------

    def classify(self, song: Song) -> ClassificationResult:
        """Classify a single song, serving from cache when possible."""
        if self._cache is not None:
            cached = self._cache.get(song)
            if cached is not None:
                try:
                    return self._normalize(cached, song)
                except ClassificationError:
                    pass  # stale or corrupt entry -- fall through and re-ask

        result = self._normalize(self._classify_song(song), song)
        self._remember(result)
        self._flush()
        return result

    def classify_batch(
        self,
        songs: list[Song],
        on_progress: ProgressCallback | None = None,
    ) -> list[ClassificationResult]:
        """Classify many songs, serving what it can from the cache.

        A song that fails to classify is logged and skipped rather than
        aborting the run, so one bad response can't kill a 500-song job. The
        returned list may therefore be shorter than `songs`.
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
                pending.append(song)
                continue
            done += 1
            report()

        if pending and total != len(pending):
            logger.info(
                "%d/%d songs served from cache; classifying %d",
                total - len(pending),
                total,
                len(pending),
            )

        size = max(1, self.batch_size)
        for start in range(0, len(pending), size):
            chunk = pending[start : start + size]
            results.extend(self._classify_chunk(chunk))
            done += len(chunk)
            report()

        self._flush()
        return results

    # -- internals ----------------------------------------------------------

    def _classify_chunk(self, chunk: list[Song]) -> list[ClassificationResult]:
        """Classify one chunk, falling back to single calls if the batch fails.

        FatalClassificationError is deliberately not caught here: an expired
        key or an empty credit balance fails the same way for every song, so
        the run should stop rather than retry the whole library one at a time.
        """
        if len(chunk) > 1:
            try:
                return self._collect(self._classify_songs(chunk), chunk)
            except FatalClassificationError:
                raise
            except ClassificationError as exc:
                logger.warning(
                    "Batch of %d failed (%s); retrying song by song", len(chunk), exc
                )

        results: list[ClassificationResult] = []
        for song in chunk:
            try:
                result = self._normalize(self._classify_song(song), song)
            except FatalClassificationError:
                raise
            except Exception as exc:  # noqa: BLE001 -- one song must not kill the run
                logger.warning("Skipping %s by %s: %s", song.title, song.artist, exc)
                continue
            self._remember(result)
            results.append(result)
        return results

    def _collect(
        self, fields_by_index: dict[int, dict], chunk: list[Song]
    ) -> list[ClassificationResult]:
        """Normalize a batch response and warn about anything the model dropped."""
        results: list[ClassificationResult] = []
        for index, song in enumerate(chunk):
            fields = fields_by_index.get(index)
            if fields is None:
                logger.warning(
                    "Batch response omitted %s by %s", song.title, song.artist
                )
                continue
            try:
                result = self._normalize(fields, song)
            except ClassificationError as exc:
                logger.warning("Skipping %s by %s: %s", song.title, song.artist, exc)
                continue
            self._remember(result)
            results.append(result)
        return results

    def _remember(self, result: ClassificationResult) -> None:
        if self._cache is not None:
            self._cache.put(result.song, self._result_fields(result))

    def _flush(self) -> None:
        if self._cache is not None:
            self._cache.save()

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

    Batches `batch_size` songs per request, which turns a 2,000-song library
    from 2,000 calls into ~100. Caching and error isolation come from
    BaseClassifier.
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
        super().__init__(use_cache=use_cache, cache_path=cache_path)
        anthropic = _require("anthropic", "anthropic")
        key = _api_key(api_key, "ANTHROPIC_API_KEY", "Claude")

        self._anthropic = anthropic
        # The SDK retries 429s and 5xx with exponential backoff itself; 5 is a
        # friendlier ceiling than the default 2 for a long batch run.
        self._client = anthropic.Anthropic(api_key=key, max_retries=5)
        self.model = model
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.effort = effort

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
            message = f"Claude API error ({exc.status_code}): {exc.message}"
            if _is_fatal(exc.message, exc.status_code):
                raise FatalClassificationError(message) from exc
            raise ClassificationError(message) from exc
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

    def _classify_song(self, song: Song) -> dict:
        return self._call(build_classification_prompt(song), CLASSIFICATION_SCHEMA)

    def _classify_songs(self, songs: list[Song]) -> dict[int, dict]:
        payload = self._call(build_batch_prompt(songs), BATCH_SCHEMA)
        return _index_batch(payload, len(songs))


class OpenAIClassifier(BaseClassifier):
    """Classifier backed by the OpenAI API.

    Uses the same prompts and the same JSON Schema as the Claude adapter --
    only the transport differs. Install with: pip install openai
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        batch_size: int = 20,
        use_cache: bool = True,
        cache_path: Path | None = None,
    ) -> None:
        super().__init__(use_cache=use_cache, cache_path=cache_path)
        openai = _require("openai", "openai")
        key = _api_key(api_key, "OPENAI_API_KEY", "OpenAI")

        self._openai = openai
        self._client = openai.OpenAI(api_key=key, max_retries=5)
        self.model = model
        self.batch_size = batch_size

    def _call(self, prompt: str, schema: dict, schema_name: str) -> dict:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": schema,
                        "strict": True,
                    },
                },
            )
        except Exception as exc:  # noqa: BLE001 -- SDK error types vary by version
            message = f"OpenAI API error: {exc}"
            if _is_fatal(str(exc), getattr(exc, "status_code", None)):
                raise FatalClassificationError(message) from exc
            raise ClassificationError(message) from exc

        choice = response.choices[0]
        if getattr(choice.message, "refusal", None):
            raise ClassificationError(
                f"OpenAI declined to classify: {choice.message.refusal}"
            )
        if choice.finish_reason == "length":
            raise ClassificationError(
                "Response was truncated. Lower batch_size for this provider."
            )
        return self._parse_response(choice.message.content or "")

    def _classify_song(self, song: Song) -> dict:
        return self._call(
            build_classification_prompt(song), CLASSIFICATION_SCHEMA, "classification"
        )

    def _classify_songs(self, songs: list[Song]) -> dict[int, dict]:
        payload = self._call(build_batch_prompt(songs), BATCH_SCHEMA, "classifications")
        return _index_batch(payload, len(songs))


class GeminiClassifier(BaseClassifier):
    """Classifier backed by Google Gemini.

    Gemini's `response_schema` accepts a different JSON Schema dialect than
    the one in prompts.py (no `additionalProperties`, for one), so this asks
    for JSON mode only and leans on BaseClassifier._normalize() to enforce the
    vocabulary and bounds. Install with: pip install google-genai
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-1.5-pro",
        batch_size: int = 20,
        use_cache: bool = True,
        cache_path: Path | None = None,
    ) -> None:
        super().__init__(use_cache=use_cache, cache_path=cache_path)
        genai = _require("google.genai", "google-genai")
        key = _api_key(api_key, "GEMINI_API_KEY", "Gemini")

        self._genai = genai
        self._client = genai.Client(api_key=key)
        self.model = model
        self.batch_size = batch_size

    def _call(self, prompt: str) -> dict:
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "response_mime_type": "application/json",
                },
            )
        except Exception as exc:  # noqa: BLE001 -- SDK error types vary by version
            raise ClassificationError(f"Gemini API error: {exc}") from exc

        text = getattr(response, "text", None)
        if not text:
            raise ClassificationError("Gemini returned no text content.")
        return self._parse_response(text)

    def _classify_song(self, song: Song) -> dict:
        return self._call(build_classification_prompt(song))

    def _classify_songs(self, songs: list[Song]) -> dict[int, dict]:
        return _index_batch(self._call(build_batch_prompt(songs)), len(songs))


class OllamaClassifier(BaseClassifier):
    """Classifier backed by a local Ollama server -- no API key, no cost.

    Talks to Ollama's HTTP API with the standard library rather than adding a
    dependency. Batching defaults to 5 rather than 20 because local models
    handle long multi-song prompts less reliably than hosted ones.
    """

    def __init__(
        self,
        model: str = "llama3",
        host: str = "http://localhost:11434",
        batch_size: int = 5,
        timeout: float = 180.0,
        use_cache: bool = True,
        cache_path: Path | None = None,
    ) -> None:
        super().__init__(use_cache=use_cache, cache_path=cache_path)
        self.model = model
        self.host = host.rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout

    def _call(self, prompt: str) -> dict:
        import urllib.error
        import urllib.request

        body = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "system": SYSTEM_PROMPT,
                "format": "json",
                "stream": False,
            }
        ).encode()
        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except urllib.error.URLError as exc:
            raise ClassificationError(
                f"Could not reach Ollama at {self.host} ({exc}). Is `ollama serve` "
                "running?"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ClassificationError(f"Ollama returned malformed JSON: {exc}") from exc

        return self._parse_response(payload.get("response", ""))

    def _classify_song(self, song: Song) -> dict:
        return self._call(build_classification_prompt(song))

    def _classify_songs(self, songs: list[Song]) -> dict[int, dict]:
        return _index_batch(self._call(build_batch_prompt(songs)), len(songs))


def _index_batch(payload: dict, count: int) -> dict[int, dict]:
    """Turn a `{"results": [...]}` batch response into {index: fields}.

    Matching on the echoed `index` rather than list position means a model
    that reorders or repeats entries can't silently misattribute a
    classification to the wrong song.
    """
    entries = payload.get("results")
    if not isinstance(entries, list):
        raise ClassificationError("Batch response had no `results` list.")

    by_index: dict[int, dict] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        index = entry.get("index")
        if not isinstance(index, int) or not 0 <= index < count:
            logger.warning("Dropping batch entry with bad index %r", index)
            continue
        if index in by_index:
            logger.warning("Dropping duplicate batch entry for index %d", index)
            continue
        by_index[index] = entry
    return by_index


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
