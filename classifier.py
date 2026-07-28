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
from dataclasses import dataclass
from typing import Protocol

from music import Song
from prompts import SYSTEM_PROMPT, build_classification_prompt

logger = logging.getLogger(__name__)


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

    TODO(Part B): keep this the single source of truth for what "a
    classifier" means in this project. app.py/ui.py should type against
    this Protocol, never a concrete class.
    """

    def classify(self, song: Song) -> ClassificationResult:
        """Classify a single song. Must return a ClassificationResult."""
        ...

    def classify_batch(self, songs: list[Song]) -> list[ClassificationResult]:
        """Classify many songs (default: loop over classify()).

        Override in a subclass if the provider supports true batch calls.
        """
        ...


class BaseClassifier:
    """Shared helpers for concrete classifier implementations.

    TODO(Part B):
    - Implement classify_batch() here as a simple loop calling self.classify()
      with logging/progress callback support (so the GUI progress bar can
      hook in via an optional `on_progress(i, total)` callback param).
    - Implement a `_parse_response(raw_text: str) -> dict` helper that
      strips markdown code fences and json.loads()'s the model output,
      raising a clear error on malformed JSON.
    """

    def classify_batch(self, songs: list[Song]) -> list[ClassificationResult]:
        raise NotImplementedError("Part B: implement BaseClassifier.classify_batch()")


class ClaudeClassifier(BaseClassifier):
    """Classifier backed by the Anthropic Claude API.

    TODO(Part B):
    - __init__(self, api_key: str | None = None, model: str = "claude-sonnet-5")
      -- read api_key from ANTHROPIC_API_KEY env var if not passed.
    - classify(): build prompt via build_classification_prompt(song), call
      the Claude Messages API with SYSTEM_PROMPT as system, parse JSON
      response into ClassificationResult.
    - Handle API errors / rate limits with retries + logging.
    """

    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-5") -> None:
        raise NotImplementedError("Part B: implement ClaudeClassifier.__init__()")

    def classify(self, song: Song) -> ClassificationResult:
        raise NotImplementedError("Part B: implement ClaudeClassifier.classify()")


class OpenAIClassifier(BaseClassifier):
    """Stub adapter for OpenAI models. Same interface as ClaudeClassifier.

    TODO(Part B, optional/stretch): implement using the OpenAI SDK.
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


def get_classifier(provider: str = "claude", **kwargs) -> Classifier:
    """Factory: return a configured Classifier instance by provider name.

    TODO(Part B): map "claude" -> ClaudeClassifier, "openai" -> OpenAIClassifier,
    "gemini" -> GeminiClassifier, "ollama" -> OllamaClassifier. Raise
    ValueError on unknown provider.

    This is the single function app.py/ui.py should call to obtain a
    classifier -- it is how the "swap providers" requirement is satisfied.
    """
    raise NotImplementedError("Part B: implement get_classifier()")


def save_classified_songs(results: list[ClassificationResult], path) -> None:
    """Serialize classification results to classified_songs.json for Part C.

    TODO(Part B): decide JSON shape, e.g.:
    [
      {"song": {...Song fields...}, "genre": ..., "energy": ...,
       "mood": ..., "contexts": [...]}
    ]
    Keep in sync with what music.build_playlists() (Part C) expects.
    """
    raise NotImplementedError("Part B: implement save_classified_songs()")
