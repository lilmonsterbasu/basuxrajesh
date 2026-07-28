"""Tests for the non-Claude provider adapters (Part B stretch goals).

Each provider's SDK is faked, so these run with no API keys, no network, and
none of the optional packages installed.
"""

from __future__ import annotations

import json
import sys
import types
import urllib.error

import pytest

from classifier import (
    BaseClassifier,
    ClassificationError,
    GeminiClassifier,
    OllamaClassifier,
    OpenAIClassifier,
    get_classifier,
)


def _entry(index: int, **overrides) -> dict:
    entry = {
        "index": index,
        "genre": "Shoegaze",
        "energy": 6,
        "mood": "Melancholic",
        "contexts": ["Late Night", "Coding"],
    }
    entry.update(overrides)
    return entry


def _batch(count: int) -> str:
    return json.dumps({"results": [_entry(i) for i in range(count)]})


# --------------------------------------------------------------------------
# OpenAI
# --------------------------------------------------------------------------


class _FakeChoice:
    def __init__(self, content, refusal=None, finish_reason="stop") -> None:
        self.message = types.SimpleNamespace(content=content, refusal=refusal)
        self.finish_reason = finish_reason


class _FakeCompletions:
    def __init__(self, responses) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return types.SimpleNamespace(choices=[response])


class _FakeOpenAIClient:
    def __init__(self, responses) -> None:
        self.completions = _FakeCompletions(responses)
        self.chat = types.SimpleNamespace(completions=self.completions)


@pytest.fixture
def make_openai(monkeypatch, tmp_path):
    def _make(responses, **kwargs):
        client = _FakeOpenAIClient(responses)
        fake_sdk = types.ModuleType("openai")
        fake_sdk.OpenAI = lambda **_: client
        monkeypatch.setitem(sys.modules, "openai", fake_sdk)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        kwargs.setdefault("cache_path", tmp_path / "openai-cache.json")
        return OpenAIClassifier(**kwargs)

    return _make


def test_openai_classifies_a_song(make_openai, song):
    clf = make_openai([_FakeChoice(json.dumps(_entry(0)))])
    result = clf.classify(song)

    assert result.genre == "Shoegaze"
    assert result.contexts == ["Late Night", "Coding"]


def test_openai_sends_strict_schema(make_openai, song):
    clf = make_openai([_FakeChoice(json.dumps(_entry(0)))])
    clf.classify(song)

    kwargs = clf._client.completions.calls[0]
    schema = kwargs["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["additionalProperties"] is False
    assert kwargs["messages"][0]["role"] == "system"


def test_openai_refusal_is_reported(make_openai, song):
    clf = make_openai([_FakeChoice(None, refusal="I can't help with that")])
    with pytest.raises(ClassificationError, match="declined"):
        clf.classify(song)


def test_openai_truncation_is_reported(make_openai, song):
    clf = make_openai([_FakeChoice("{", finish_reason="length")])
    with pytest.raises(ClassificationError, match="truncated"):
        clf.classify(song)


def test_openai_api_error_is_wrapped(make_openai, song):
    clf = make_openai([RuntimeError("connection reset")])
    with pytest.raises(ClassificationError, match="OpenAI API error"):
        clf.classify(song)


def test_openai_batches(make_openai, songs):
    clf = make_openai([_FakeChoice(_batch(5))], batch_size=5)
    results = clf.classify_batch(songs[:5])

    assert len(results) == 5
    assert len(clf._client.completions.calls) == 1


def test_openai_missing_key_is_clear(monkeypatch):
    fake_sdk = types.ModuleType("openai")
    fake_sdk.OpenAI = lambda **_: None
    monkeypatch.setitem(sys.modules, "openai", fake_sdk)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ClassificationError, match="OPENAI_API_KEY"):
        OpenAIClassifier()


# --------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------


class _FakeModels:
    def __init__(self, responses) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return types.SimpleNamespace(text=response)


@pytest.fixture
def make_gemini(monkeypatch, tmp_path):
    def _make(responses, **kwargs):
        models = _FakeModels(responses)
        fake_genai = types.ModuleType("google.genai")
        fake_genai.Client = lambda **_: types.SimpleNamespace(models=models)
        google_pkg = sys.modules.get("google") or types.ModuleType("google")
        monkeypatch.setitem(sys.modules, "google", google_pkg)
        monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        kwargs.setdefault("cache_path", tmp_path / "gemini-cache.json")
        classifier = GeminiClassifier(**kwargs)
        classifier._models = models
        return classifier

    return _make


def test_gemini_classifies_a_song(make_gemini, song):
    clf = make_gemini([json.dumps(_entry(0))])
    assert clf.classify(song).mood == "Melancholic"


def test_gemini_requests_json_mode(make_gemini, song):
    clf = make_gemini([json.dumps(_entry(0))])
    clf.classify(song)

    config = clf._models.calls[0]["config"]
    assert config["response_mime_type"] == "application/json"
    assert "Late Night" in config["system_instruction"]


def test_gemini_handles_fenced_json(make_gemini, song):
    """Gemini gets no schema enforcement, so fences are a real possibility."""
    clf = make_gemini(["```json\n" + json.dumps(_entry(0)) + "\n```"])
    assert clf.classify(song).genre == "Shoegaze"


def test_gemini_empty_response_is_reported(make_gemini, song):
    clf = make_gemini([""])
    with pytest.raises(ClassificationError, match="no text content"):
        clf.classify(song)


def test_gemini_normalizes_unknown_contexts(make_gemini, song):
    """No schema enum on this provider -- _normalize is the only guard."""
    clf = make_gemini([json.dumps(_entry(0, contexts=["Coding", "Vibing", "Sleep"]))])
    assert clf.classify(song).contexts == ["Coding"]


# --------------------------------------------------------------------------
# Ollama
# --------------------------------------------------------------------------


class _FakeHTTPResponse:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def read(self):
        return self._payload.encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture
def make_ollama(monkeypatch, tmp_path):
    def _make(responses, **kwargs):
        state = {"calls": []}
        queue = list(responses)

        def fake_urlopen(request, timeout=None):
            state["calls"].append(json.loads(request.data.decode()))
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return _FakeHTTPResponse(json.dumps({"response": item}))

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        kwargs.setdefault("cache_path", tmp_path / "ollama-cache.json")
        classifier = OllamaClassifier(**kwargs)
        classifier._sent = state["calls"]
        return classifier

    return _make


def test_ollama_classifies_a_song(make_ollama, song):
    clf = make_ollama([json.dumps(_entry(0))])
    assert clf.classify(song).energy == 6


def test_ollama_requests_json_format(make_ollama, song):
    clf = make_ollama([json.dumps(_entry(0))])
    clf.classify(song)

    sent = clf._sent[0]
    assert sent["format"] == "json"
    assert sent["stream"] is False
    assert sent["model"] == "llama3"


def test_ollama_needs_no_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    OllamaClassifier(cache_path=tmp_path / "c.json")  # must not raise


def test_ollama_connection_error_mentions_serve(make_ollama, song):
    clf = make_ollama([urllib.error.URLError("connection refused")])
    with pytest.raises(ClassificationError, match="ollama serve"):
        clf.classify(song)


def test_ollama_uses_a_smaller_default_batch(tmp_path):
    """Local models cope worse with long multi-song prompts than hosted ones."""
    assert OllamaClassifier(cache_path=tmp_path / "c.json").batch_size == 5


# --------------------------------------------------------------------------
# Shared behaviour across providers
# --------------------------------------------------------------------------


def test_every_provider_inherits_the_cache(make_openai, songs, tmp_path):
    """Caching lives in BaseClassifier, so it is not a Claude-only perk."""
    cache_path = tmp_path / "shared-cache.json"
    first = make_openai([_FakeChoice(_batch(3))], batch_size=3, cache_path=cache_path)
    first.classify_batch(songs[:3])
    assert len(first._client.completions.calls) == 1

    second = make_openai([], batch_size=3, cache_path=cache_path)
    results = second.classify_batch(songs[:3])

    assert len(results) == 3
    assert second._client.completions.calls == []


def test_batch_failure_falls_back_to_single_calls(make_openai, songs):
    clf = make_openai(
        [
            _FakeChoice("not json"),  # batch attempt fails
            _FakeChoice(json.dumps(_entry(0))),
            _FakeChoice(json.dumps(_entry(0))),
            _FakeChoice(json.dumps(_entry(0))),
        ],
        batch_size=3,
    )
    results = clf.classify_batch(songs[:3])

    assert len(results) == 3
    assert len(clf._client.completions.calls) == 4


def test_batch_omission_is_reported_not_guessed(make_openai, songs):
    """A song the model skipped must be dropped, never given a neighbour's tags."""
    payload = json.dumps({"results": [_entry(0), _entry(2)]})
    clf = make_openai([_FakeChoice(payload)], batch_size=3)
    results = clf.classify_batch(songs[:3])

    assert {r.song.title for r in results} == {songs[0].title, songs[2].title}


@pytest.mark.parametrize(
    ("name", "cls"),
    [("openai", OpenAIClassifier), ("gemini", GeminiClassifier), ("ollama", OllamaClassifier)],
)
def test_factory_dispatches_every_provider(name, cls, monkeypatch, tmp_path):
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = lambda **_: _FakeOpenAIClient([])
    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = lambda **_: types.SimpleNamespace(models=_FakeModels([]))
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setitem(sys.modules, "google", sys.modules.get("google") or types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    assert isinstance(get_classifier(name, cache_path=tmp_path / "c.json"), cls)


def test_missing_sdk_explains_the_install(monkeypatch):
    monkeypatch.delitem(sys.modules, "openai", raising=False)
    monkeypatch.setattr(
        "importlib.import_module",
        lambda name: (_ for _ in ()).throw(ImportError(f"no module {name}")),
    )
    with pytest.raises(ClassificationError, match="pip install openai"):
        OpenAIClassifier()


def test_every_provider_satisfies_the_protocol():
    """app.py types against Classifier, so every adapter must match it."""
    for cls in (OpenAIClassifier, GeminiClassifier, OllamaClassifier):
        assert issubclass(cls, BaseClassifier)
        assert cls._classify_song is not BaseClassifier._classify_song
        assert cls._classify_songs is not BaseClassifier._classify_songs
