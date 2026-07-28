-- OWNER: Part A (Data Export Layer)
--
-- Reads every user playlist and every song in the Apple Music library,
-- and writes a JSON file to stdout (captured by music.py via osascript).
--
-- Expected output shape (see music.py Song dataclass / library_export.json):
-- {
--   "playlists": [
--     { "name": "My Playlist", "song_titles": ["Song A", "Song B"] }
--   ],
--   "songs": [
--     {
--       "title": "...",
--       "artist": "...",
--       "album": "...",
--       "genre": "...",
--       "year": 2020,
--       "play_count": 12,
--       "duration": 214.5
--     }
--   ]
-- }
--
-- TODO(Part A):
-- 1. tell application "Music" to get every playlist (user playlists only,
--    skip "Library"/"Podcasts"/smart playlists if desired).
-- 2. For each playlist, collect the names of its tracks.
-- 3. tell application "Music" to get every track of playlist "Library"
--    (or the main library source) and collect the 7 fields per song.
-- 4. Manually build a JSON string (AppleScript has no native JSON encoder)
--    -- escape quotes/backslashes in text fields.
-- 5. `return` the JSON string as the script's result so `osascript -e`
--    or `osascript export_library.scpt` prints it to stdout.
--
-- Run manually while building:
--   osascript applescript/export_library.scpt > library_export.json

on run
	return "{\"playlists\": [], \"songs\": []}"
end run
