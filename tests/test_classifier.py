"""Tests for Part B (AI Classification).

Runs entirely against the fixture library and a fake Anthropic client -- no
API key, no network, and no dependency on Part A or Part C.
"""

from __future__ import annotations

import json

import httpx
import pytest

import anthropic
from classifier import (
    ClaudeClassifier,
    ClassificationCache,
    ClassificationError,
    ClassificationResult,
    BaseClassifier,
    get_classifier,
    load_classified_songs,
    save_classified_songs,
)
from prompts import CONTEXT_VOCABULARY, build_batch_prompt, build_classification_prompt


# --------------------------------------------------------------------------
# Fake Anthropic client
# --------------------------------------------------------------------------


class _FakeBlock:
    def __init__(self, type_: str, text: str | None = None) -> None:
        self.type = type_
        self.text = text


class _FakeResponse:
    """Mimics a Message: a thinking block first, then the JSON text block."""

    def __init__(self, payload, stop_reason: str = "end_turn") -> None:
        text = payload if isinstance(payload, str) else json.dumps(payload)
        self.content = [_FakeBlock("thinking"), _FakeBlock("text", text)]
        self.stop_reason = stop_reason


class _FakeMessages:
    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("Fake client ran out of queued responses")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _FakeClient:
    def __init__(self, responses: list) -> None:
        self.messages = _FakeMessages(responses)


@pytest.fixture
def make_classifier(monkeypatch, tmp_path):
    """Build a ClaudeClassifier whose API calls return canned responses."""

    def _make(responses: list, **kwargs) -> ClaudeClassifier:
        client = _FakeClient(responses)
        monkeypatch.setattr(anthropic, "Anthropic", lambda **_: client)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        kwargs.setdefault("cache_path", tmp_path / "cache.json")
        return ClaudeClassifier(**kwargs)

    return _make


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


def _batch(count: int) -> dict:
    return {"results": [_entry(i) for i in range(count)]}


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------


def test_single_prompt_includes_every_signal(song):
    prompt = build_classification_prompt(song)
    for value in (song.title, song.artist, song.album, song.genre, str(song.year)):
        assert value in prompt


def test_batch_prompt_numbers_each_song(songs):
    prompt = build_batch_prompt(songs[:3])
    assert "[index: 0]" in prompt and "[index: 2]" in prompt
    assert "[index: 3]" not in prompt


# --------------------------------------------------------------------------
# Response parsing and normalization
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        '{"genre": "Pop"}',
        '```json\n{"genre": "Pop"}\n```',
        '```\n{"genre": "Pop"}\n```',
        '  \n{"genre": "Pop"}\n  ',
    ],
)
def test_parse_response_strips_fences(raw):
    assert BaseClassifier._parse_response(raw) == {"genre": "Pop"}


def test_parse_response_rejects_prose():
    with pytest.raises(ClassificationError, match="valid JSON"):
        BaseClassifier._parse_response("Sure! Here are the results.")


def test_parse_response_rejects_non_object():
    with pytest.raises(ClassificationError, match="JSON object"):
        BaseClassifier._parse_response("[1, 2, 3]")


def test_normalize_rejects_missing_fields(song):
    with pytest.raises(ClassificationError, match="missing required field"):
        BaseClassifier._normalize({"genre": "Pop", "energy": 5}, song)


@pytest.mark.parametrize(
    ("raw_energy", "expected"),
    [(0, 1), (-4, 1), (11, 10), (99, 10), (7, 7), ("8", 8), (6.6, 7)],
)
def test_normalize_clamps_energy(song, raw_energy, expected):
    result = BaseClassifier._normalize(_entry(0, energy=raw_energy), song)
    assert result.energy == expected


def test_normalize_rejects_non_numeric_energy(song):
    with pytest.raises(ClassificationError, match="energy must be a number"):
        BaseClassifier._normalize(_entry(0, energy="loud"), song)


def test_normalize_canonicalizes_and_drops_unknown_contexts(song):
    result = BaseClassifier._normalize(
        _entry(0, contexts=["late night", "  CODING ", "Nightcore Vibes"]), song
    )
    assert result.contexts == ["Late Night", "Coding"]


def test_normalize_dedupes_and_caps_contexts(song):
    result = BaseClassifier._normalize(
        _entry(0, contexts=["Coding", "coding", "Party", "Chill", "Workout"]), song
    )
    assert result.contexts == ["Coding", "Party", "Chill"]


def test_normalize_tolerates_no_usable_contexts(song):
    result = BaseClassifier._normalize(_entry(0, contexts=["Nonsense"]), song)
    assert result.contexts == []


def test_normalize_rejects_non_list_contexts(song):
    with pytest.raises(ClassificationError, match="contexts must be a list"):
        BaseClassifier._normalize(_entry(0, contexts="Coding"), song)


# --------------------------------------------------------------------------
# Claude classifier: single calls
# --------------------------------------------------------------------------


def test_classify_single_song(make_classifier, song):
    clf = make_classifier([_FakeResponse(_entry(0))])
    result = clf.classify(song)

    assert result.song is song
    assert result.genre == "Shoegaze"
    assert result.contexts == ["Late Night", "Coding"]


def test_classify_sends_schema_and_system_prompt(make_classifier, song):
    clf = make_classifier([_FakeResponse(_entry(0))])
    clf.classify(song)

    kwargs = clf._client.messages.calls[0]
    assert kwargs["model"] == "claude-opus-5"
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    # The closed context vocabulary must reach the model, or playlists sprawl.
    assert "Late Night" in kwargs["system"][0]["text"]
    assert "temperature" not in kwargs  # rejected by this model


def test_refusal_raises(make_classifier, song):
    clf = make_classifier([_FakeResponse(_entry(0), stop_reason="refusal")])
    with pytest.raises(ClassificationError, match="declined"):
        clf.classify(song)


def test_truncated_response_raises(make_classifier, song):
    clf = make_classifier([_FakeResponse(_entry(0), stop_reason="max_tokens")])
    with pytest.raises(ClassificationError, match="max_tokens"):
        clf.classify(song)


def test_api_status_error_is_wrapped(make_classifier, song):
    http_response = httpx.Response(
        500, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )
    boom = anthropic.APIStatusError("server exploded", response=http_response, body=None)
    clf = make_classifier([boom])
    with pytest.raises(ClassificationError, match="Claude API error"):
        clf.classify(song)


# --------------------------------------------------------------------------
# Claude classifier: batching
# --------------------------------------------------------------------------


def test_batch_uses_one_call_per_chunk(make_classifier, songs):
    clf = make_classifier([_FakeResponse(_batch(10)), _FakeResponse(_batch(5))],
                          batch_size=10)
    results = clf.classify_batch(songs)

    assert len(results) == 15
    assert len(clf._client.messages.calls) == 2


def test_batch_matches_results_by_index_not_position(make_classifier, songs):
    chunk = songs[:3]
    shuffled = {
        "results": [
            _entry(2, genre="Ambient"),
            _entry(0, genre="Indie Rock"),
            _entry(1, genre="Synthpop"),
        ]
    }
    clf = make_classifier([_FakeResponse(shuffled)], batch_size=3)
    by_title = {r.song.title: r.genre for r in clf.classify_batch(chunk)}

    assert by_title[chunk[0].title] == "Indie Rock"
    assert by_title[chunk[1].title] == "Synthpop"
    assert by_title[chunk[2].title] == "Ambient"


def test_batch_drops_bad_and_duplicate_indexes(make_classifier, songs):
    payload = {
        "results": [
            _entry(0),
            _entry(0),  # duplicate
            _entry(99),  # out of range
            _entry(1),
        ]
    }
    clf = make_classifier([_FakeResponse(payload)], batch_size=3)
    results = clf.classify_batch(songs[:3])

    assert {r.song.title for r in results} == {songs[0].title, songs[1].title}


def test_batch_failure_falls_back_to_single_calls(make_classifier, songs):
    clf = make_classifier(
        [
            _FakeResponse("not json at all"),  # batch call fails
            _FakeResponse(_entry(0)),
            _FakeResponse(_entry(0)),
            _FakeResponse(_entry(0)),
        ],
        batch_size=3,
    )
    results = clf.classify_batch(songs[:3])

    assert len(results) == 3
    assert len(clf._client.messages.calls) == 4


def test_one_bad_song_does_not_kill_the_run(make_classifier, songs):
    clf = make_classifier(
        [
            _FakeResponse("not json at all"),
            _FakeResponse(_entry(0)),
            _FakeResponse("still not json"),  # this song is skipped
            _FakeResponse(_entry(0)),
        ],
        batch_size=3,
    )
    results = clf.classify_batch(songs[:3])

    assert len(results) == 2


def test_progress_callback_reaches_total(make_classifier, songs):
    clf = make_classifier([_FakeResponse(_batch(10)), _FakeResponse(_batch(5))],
                          batch_size=10)
    seen: list[tuple[int, int]] = []
    clf.classify_batch(songs, on_progress=lambda done, total: seen.append((done, total)))

    assert seen[-1] == (15, 15)
    assert all(total == 15 for _, total in seen)


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------


def test_cache_avoids_reclassifying(make_classifier, songs, tmp_path):
    cache_path = tmp_path / "cache.json"
    first = make_classifier([_FakeResponse(_batch(3))], batch_size=3,
                            cache_path=cache_path)
    first.classify_batch(songs[:3])
    assert len(first._client.messages.calls) == 1

    # A second run over the same songs should make no API calls at all.
    second = make_classifier([], batch_size=3, cache_path=cache_path)
    results = second.classify_batch(songs[:3])

    assert len(results) == 3
    assert second._client.messages.calls == []


def test_cache_only_classifies_new_songs(make_classifier, songs, tmp_path):
    cache_path = tmp_path / "cache.json"
    first = make_classifier([_FakeResponse(_batch(3))], batch_size=10,
                            cache_path=cache_path)
    first.classify_batch(songs[:3])

    second = make_classifier([_FakeResponse(_batch(2))], batch_size=10,
                             cache_path=cache_path)
    results = second.classify_batch(songs[:5])

    assert len(results) == 5
    sent = second._client.messages.calls[0]["messages"][0]["content"]
    assert songs[3].title in sent and songs[4].title in sent
    assert songs[0].title not in sent


def test_cache_survives_a_corrupt_file(tmp_path, song):
    cache_path = tmp_path / "cache.json"
    cache_path.write_text("{ this is not json")
    cache = ClassificationCache(cache_path)

    assert cache.get(song) is None

    cache.put(song, {"genre": "Pop", "energy": 5, "mood": "Chill", "contexts": []})
    cache.save()
    assert ClassificationCache(cache_path).get(song)["genre"] == "Pop"


def test_cache_key_ignores_case_and_padding(tmp_path, song):
    from music import Song

    cache = ClassificationCache(tmp_path / "cache.json")
    cache.put(song, {"genre": "Pop", "energy": 5, "mood": "Chill", "contexts": []})

    same_song = Song(
        title=f"  {song.title.upper()}  ",
        artist=song.artist.lower(),
        album="different album",
        genre="different genre",
        year=1900,
        play_count=0,
        duration=1.0,
    )
    assert cache.get(same_song) is not None


def test_use_cache_false_disables_it(make_classifier, songs, tmp_path):
    cache_path = tmp_path / "cache.json"
    clf = make_classifier([_FakeResponse(_batch(3))], batch_size=3,
                          use_cache=False, cache_path=cache_path)
    clf.classify_batch(songs[:3])

    assert not cache_path.exists()


# --------------------------------------------------------------------------
# Factory and serialization
# --------------------------------------------------------------------------


def test_get_classifier_dispatches_to_claude(monkeypatch, tmp_path):
    monkeypatch.setattr(anthropic, "Anthropic", lambda **_: _FakeClient([]))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    assert isinstance(
        get_classifier("claude", cache_path=tmp_path / "c.json"), ClaudeClassifier
    )


def test_get_classifier_is_case_insensitive(monkeypatch, tmp_path):
    monkeypatch.setattr(anthropic, "Anthropic", lambda **_: _FakeClient([]))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    assert isinstance(
        get_classifier("Claude", cache_path=tmp_path / "c.json"), ClaudeClassifier
    )


def test_get_classifier_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_classifier("winamp")


def test_missing_api_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ClassificationError, match="ANTHROPIC_API_KEY"):
        ClaudeClassifier()


def test_save_and_load_round_trip(tmp_path, songs):
    results = [
        ClassificationResult(
            song=songs[0], genre="Indie Rock", energy=5,
            mood="Melancholic", contexts=["Late Night"],
        ),
        ClassificationResult(
            song=songs[1], genre="Synthpop", energy=9,
            mood="Euphoric", contexts=["Workout", "Party"],
        ),
    ]
    path = tmp_path / "nested" / "classified_songs.json"
    save_classified_songs(results, path)

    # The on-disk shape is Part C's input contract -- assert it explicitly.
    payload = json.loads(path.read_text())
    assert payload[0]["song"]["title"] == songs[0].title
    assert payload[0]["song"]["play_count"] == songs[0].play_count
    assert payload[1]["contexts"] == ["Workout", "Party"]

    reloaded = load_classified_songs(path)
    assert reloaded == results


def test_context_vocabulary_has_no_near_duplicates():
    normalized = [tag.lower().replace("-", " ").replace("_", " ") for tag in CONTEXT_VOCABULARY]
    assert len(set(normalized)) == len(CONTEXT_VOCABULARY)
