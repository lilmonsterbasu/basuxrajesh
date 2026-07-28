-- OWNER: Part C (Playlist Logic + Apple Music Write-back)
--
-- Adds a single song (matched by title + artist) to an existing playlist,
-- avoiding duplicates.
--
-- Usage:
--   osascript applescript/add_song_to_playlist.scpt "Coding" "Song Title" "Artist Name"
--
-- TODO(Part C):
-- 1. Accept playlistName, songTitle, songArtist via `on run argv`.
-- 2. tell application "Music": find the track in the main library matching
--    title + artist (first match).
-- 3. Check whether a track with the same title+artist already exists in the
--    target playlist -- if so, return "duplicate" and do nothing.
-- 4. Otherwise `duplicate theTrack to playlistName` (Music.app's way of
--    adding an existing track to another playlist) and return "added".
-- 5. Handle the "track not found" case gracefully (return "not_found").

on run argv
	set playlistName to item 1 of argv
	set songTitle to item 2 of argv
	set songArtist to item 3 of argv
	tell application "Music"
		-- TODO(Part C): implement lookup + duplicate-check + add
		return "not_implemented"
	end tell
end run
