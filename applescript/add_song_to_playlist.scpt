-- Part C: Playlist Logic + Apple Music Write-back
--
-- Adds a single song (matched by title + artist) to an existing playlist,
-- avoiding duplicates.
--
-- Usage:
--   osascript applescript/add_song_to_playlist.scpt "Coding" "Song Title" "Artist Name"
--
-- Returns one of: "added", "duplicate", "not_found", "playlist_not_found"

on run argv
	set playlistName to item 1 of argv
	set songTitle to item 2 of argv
	set songArtist to item 3 of argv

	tell application "Music"
		if not (exists user playlist playlistName) then
			return "playlist_not_found"
		end if
		set targetPlaylist to user playlist playlistName

		set matchingTracks to (every track of library playlist 1 whose name is songTitle and artist is songArtist)
		if (count of matchingTracks) is 0 then
			return "not_found"
		end if
		set theTrack to item 1 of matchingTracks

		set existingTracks to (every track of targetPlaylist whose name is songTitle and artist is songArtist)
		if (count of existingTracks) > 0 then
			return "duplicate"
		end if

		duplicate theTrack to targetPlaylist
		return "added"
	end tell
end run
