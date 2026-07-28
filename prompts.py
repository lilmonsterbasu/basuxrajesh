"""prompts.py

OWNER: Part B (AI Classification)

Central place for prompt templates so they can be tuned without touching
classifier.py's logic.
"""

from __future__ import annotations

from music import Song

# The closed set of situational tags a song may be assigned. Part C turns each
# tag into an Apple Music playlist, so this list *is* the playlist list --
# without it the model invents "Late Night", "Late-night", and "Nighttime" as
# three separate playlists.
CONTEXT_VOCABULARY = [
    "Workout",
    "Running",
    "Coding",
    "Studying",
    "Late Night",
    "Morning",
    "Road Trip",
    "Party",
    "Chill",
    "Dinner",
    "Cleaning",
    "Rainy Day",
    "Heartbreak",
    "Feel Good",
]

_SYSTEM_PROMPT_TEMPLATE = """\
You are a music categorization engine. Given a song's metadata, infer its
genre, energy level, mood, and situational contexts. Use the artist, album,
genre, and title to infer mood and context -- do not just repeat the given
genre field verbatim; refine it if you can be more specific.

Rules:
- energy is an integer from 1 (ambient, barely there) to 10 (peak intensity).
- mood is a single dominant word, e.g. Melancholic, Euphoric, Chill, Tense.
- contexts must be 1-3 tags chosen ONLY from this list, spelled exactly as
  written here:
{context_list}
- Pick contexts a listener would actually reach for the song in. Do not tag a
  song with every plausible context -- fewer, better tags make better playlists.
"""

SYSTEM_PROMPT = _SYSTEM_PROMPT_TEMPLATE.format(
    context_list="\n".join(f"  - {tag}" for tag in CONTEXT_VOCABULARY)
)

# JSON Schema handed to the API via output_config.format, which constrains the
# response to valid, parseable JSON. The structured-output schema subset does
# NOT support numeric bounds (minimum/maximum) or array length limits, so
# energy clamping and context trimming happen client-side in
# BaseClassifier._normalize().
_RESULT_PROPERTIES = {
    "genre": {"type": "string", "description": "Refined, specific genre."},
    "energy": {"type": "integer", "description": "Energy level, 1-10."},
    "mood": {"type": "string", "description": "Single dominant mood word."},
    "contexts": {
        "type": "array",
        "items": {"type": "string", "enum": CONTEXT_VOCABULARY},
        "description": "1-3 situational tags from the allowed list.",
    },
}

CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": dict(_RESULT_PROPERTIES),
    "required": ["genre", "energy", "mood", "contexts"],
    "additionalProperties": False,
}

BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "The song's index from the prompt.",
                    },
                    **_RESULT_PROPERTIES,
                },
                "required": ["index", "genre", "energy", "mood", "contexts"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


def _song_fields(song: Song) -> str:
    """Render one song's metadata as labeled lines.

    Year and play count are weak signals (era, and how much the user likes it)
    but cheap to include, so they go in.
    """
    return (
        f"Title: {song.title}\n"
        f"Artist: {song.artist}\n"
        f"Album: {song.album}\n"
        f"Genre (as tagged in the library): {song.genre}\n"
        f"Year: {song.year}\n"
        f"Play count: {song.play_count}"
    )


def build_classification_prompt(song: Song) -> str:
    """Build the user-turn prompt for classifying a single song."""
    return f"Classify this song:\n\n{_song_fields(song)}"


def build_batch_prompt(songs: list[Song]) -> str:
    """Build the user-turn prompt for classifying several songs in one call.

    Each song is numbered and the model echoes the number back as `index`, so
    results can be matched to songs even if the model reorders them.
    """
    blocks = [f"[index: {i}]\n{_song_fields(song)}" for i, song in enumerate(songs)]
    return (
        f"Classify each of these {len(songs)} songs. Return one result per song, "
        "echoing back each song's index.\n\n" + "\n\n".join(blocks)
    )
