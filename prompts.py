"""prompts.py

OWNER: Part B (AI Classification)

Central place for prompt templates so they can be tuned without touching
classifier.py's logic.
"""

from __future__ import annotations

from music import Song

SYSTEM_PROMPT = """\
You are a music categorization engine. Given a song's metadata, infer its
genre, energy level, mood, and situational contexts. Use the artist, album,
genre, and title to infer mood and context -- do not just repeat the given
genre field verbatim; refine it if you can be more specific.

Respond with ONLY valid JSON matching this exact shape, no prose:

{
  "genre": "string, refined/specific genre",
  "energy": integer 1-10,
  "mood": "string, single dominant mood e.g. Melancholic, Euphoric, Chill",
  "contexts": ["list", "of", "situational", "tags", "e.g. Coding, Late Night, Road Trip, Workout"]
}
"""


def build_classification_prompt(song: Song) -> str:
    """Build the user-turn prompt for classifying a single song.

    TODO(Part B):
    - Include title, artist, album, genre (and year/play_count if useful
      as weak signals) in a clear labeled format.
    - Keep this deterministic/templated so swapping models doesn't require
      prompt rewrites.
    """
    raise NotImplementedError("Part B: implement build_classification_prompt()")
