-- Part A: Data Export Layer
--
-- Reads every user playlist and every song in the Apple Music library and
-- prints a single JSON object to stdout.
--
-- Output shape:
-- {
--   "songs": [
--     {"title": "...", "artist": "...", "album": "...", "genre": "...",
--      "year": 2020, "play_count": 12, "duration": 214.5}
--   ],
--   "playlists": [
--     {"name": "Coding Mix", "tracks": [{"title": "...", "artist": "..."}]}
--   ]
-- }
--
-- Performance note: property access is done in bulk
-- (e.g. `name of every track of thePlaylist`) rather than looping with a
-- per-track `tell` block. A per-track loop is a separate Apple Event
-- round-trip per property per track and is unusably slow on libraries of
-- more than a few hundred songs; bulk list retrieval is a single round
-- trip per property and stays fast into the tens of thousands of tracks.
--
-- Run manually while developing:
--   osascript applescript/export_library.scpt > library_export.json

on run
	set jsonOut to ""
	tell application "Music"
		-- ---- Songs (entire library) ----
		set libraryPlaylist to library playlist 1
		set trackNames to my safeList(name of every track of libraryPlaylist)
		set trackArtists to my safeList(artist of every track of libraryPlaylist)
		set trackAlbums to my safeList(album of every track of libraryPlaylist)
		set trackGenres to my safeList(genre of every track of libraryPlaylist)
		set trackYears to my safeList(year of every track of libraryPlaylist)
		set trackPlayCounts to my safeList(played count of every track of libraryPlaylist)
		set trackDurations to my safeList(duration of every track of libraryPlaylist)

		set songCount to count of trackNames
		set songEntries to {}
		repeat with i from 1 to songCount
			set end of songEntries to my songToJSON(item i of trackNames, item i of trackArtists, item i of trackAlbums, item i of trackGenres, item i of trackYears, item i of trackPlayCounts, item i of trackDurations)
		end repeat
		set songsJSON to "[" & my joinList(songEntries, ",") & "]"

		-- ---- User playlists ----
		set userPlaylists to every user playlist
		set playlistEntries to {}
		repeat with thePlaylist in userPlaylists
			set playlistName to name of thePlaylist
			set pTrackNames to my safeList(name of every track of thePlaylist)
			set pTrackArtists to my safeList(artist of every track of thePlaylist)
			set trackCount to count of pTrackNames
			set trackEntries to {}
			repeat with j from 1 to trackCount
				set end of trackEntries to "{\"title\":\"" & my escapeJSON(item j of pTrackNames) & "\",\"artist\":\"" & my escapeJSON(item j of pTrackArtists) & "\"}"
			end repeat
			set tracksJSON to "[" & my joinList(trackEntries, ",") & "]"
			set end of playlistEntries to "{\"name\":\"" & my escapeJSON(playlistName) & "\",\"tracks\":" & tracksJSON & "}"
		end repeat
		set playlistsJSON to "[" & my joinList(playlistEntries, ",") & "]"
	end tell

	set jsonOut to "{\"songs\":" & songsJSON & ",\"playlists\":" & playlistsJSON & "}"
	return jsonOut
end run

-- Coerces a bulk property list to a plain list even when Music.app returns
-- a single value instead of a list (happens when the playlist has exactly
-- one track).
on safeList(maybeList)
	try
		if class of maybeList is list then
			return maybeList
		else
			return {maybeList}
		end if
	on error
		return {}
	end try
end safeList

on songToJSON(theTitle, theArtist, theAlbum, theGenre, theYear, thePlayCount, theDuration)
	set yearStr to numToStr(theYear)
	set playCountStr to numToStr(thePlayCount)
	set durationStr to numToStr(theDuration)
	return "{\"title\":\"" & escapeJSON(theTitle) & "\",\"artist\":\"" & escapeJSON(theArtist) & "\",\"album\":\"" & escapeJSON(theAlbum) & "\",\"genre\":\"" & escapeJSON(theGenre) & "\",\"year\":" & yearStr & ",\"play_count\":" & playCountStr & ",\"duration\":" & durationStr & "}"
end songToJSON

on numToStr(n)
	try
		if n is missing value then return "0"
		return (n as text)
	on error
		return "0"
	end try
end numToStr

on escapeJSON(txt)
	try
		if txt is missing value then return ""
		set txt to txt as text
	on error
		return ""
	end try
	set txt to replaceText(txt, "\\", "\\\\")
	set txt to replaceText(txt, "\"", "\\\"")
	set txt to replaceText(txt, return, "\\n")
	set txt to replaceText(txt, linefeed, "\\n")
	set txt to replaceText(txt, tab, "\\t")
	return txt
end escapeJSON

on replaceText(theText, searchStr, replaceStr)
	set oldDelims to AppleScript's text item delimiters
	set AppleScript's text item delimiters to searchStr
	set theItems to text items of theText
	set AppleScript's text item delimiters to replaceStr
	set theText to theItems as text
	set AppleScript's text item delimiters to oldDelims
	return theText
end replaceText

on joinList(theList, delim)
	set oldDelims to AppleScript's text item delimiters
	set AppleScript's text item delimiters to delim
	set theText to theList as text
	set AppleScript's text item delimiters to oldDelims
	return theText
end joinList
