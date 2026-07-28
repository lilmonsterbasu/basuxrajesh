# Web app design reference

`home-grid-cards-reference.html` is a rough Apple-style mockup for the
dashboard home page (Part D). Open it directly in a browser to preview.

## What it shows

- Hero: "Discover Your Sound" headline + one-line summary of library size /
  playlist count, two CTAs ("Analyze Now", "View Results")
- 3-column grid of playlist cards: colored thumbnail block, small caps
  category label (e.g. "FOCUS WORK"), playlist name, song count
- Footer stats row: total songs analyzed, unique artists, number of AI
  playlists generated

## Mapping to real data (for whoever builds webapp/static/index.html)

- Hero subtitle count + footer "Songs Analyzed" / "AI Playlists" numbers
  come from `data/library_export.json` (song count, unique artist count)
  and `data/playlist_plan.json` (playlist count) once Export/Analyze have
  run — before that, these should show a "run Export Library first" state
  rather than fake numbers.
- Each grid card = one playlist from `data/playlist_plan.json`
  (`{name, song_count}`). The category label (e.g. "FOCUS WORK",
  "LATE NIGHT") is the classifier's `contexts` tag from
  `data/classified_songs.json` that this playlist was built from.
- The colored block per card is currently a flat placeholder color in the
  mockup — pick a stable color per context tag (e.g. hash the tag name to
  one of a fixed palette) rather than random, so a given playlist's card
  color doesn't change between reloads.
- "Analyze Now" / "View Results" buttons map to the same
  `/api/analyze` / `/api/playlist-plan` routes stubbed in
  `webapp/server.py`.

This is a starting point, not a spec to match pixel-for-pixel — adjust
freely once real data and the FastAPI routes are wired up.
