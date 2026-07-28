-- OWNER: Part C (Playlist Logic + Apple Music Write-back)
--
-- Creates a playlist with the given name in Apple Music if it does not
-- already exist. Idempotent: safe to call repeatedly.
--
-- Usage:
--   osascript applescript/create_playlist.scpt "Coding"
--
-- TODO(Part C):
-- 1. Accept playlist name via `on run argv`.
-- 2. tell application "Music": check `exists playlist named argv's item 1`.
-- 3. If it doesn't exist, `make new user playlist with properties {name: ...}`.
-- 4. Return "created" or "skipped" so music.py can log which happened.

on run argv
	set playlistName to item 1 of argv
	tell application "Music"
		if not (exists user playlist playlistName) then
			make new user playlist with properties {name:playlistName}
			return "created"
		else
			return "skipped"
		end if
	end tell
end run
