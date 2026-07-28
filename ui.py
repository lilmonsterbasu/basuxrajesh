"""ui.py

OWNER: Part D (GUI + Web App Design) -- Tkinter desktop half.

Provides the desktop GUI: 4 action buttons, a progress bar, and a console
output panel. Long-running work (export/classify/create) must run on a
background thread so the Tk mainloop doesn't freeze.
"""

from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from tkinter import ttk

logger = logging.getLogger(__name__)


class PlaylistAIApp:
    """Main Tkinter application window.

    TODO(Part D):
    - __init__: build the window layout:
        * Top: 4 buttons -- "Export Library", "Analyze Songs",
          "Preview Results", "Create Playlists"
        * Middle: a ttk.Progressbar
        * Bottom: a scrolled Text widget used as console output (read-only,
          appended to via a thread-safe queue)
        * A preview area (ttk.Treeview) showing proposed playlist -> songs
          groupings before the user confirms "Create Playlists"
    - Each button handler should:
        1. Disable buttons while running
        2. Spawn a background thread calling into app.py's orchestration
           functions (export_library, analyze_songs, build_playlists,
           sync_playlists_to_music)
        3. Use a queue.Queue + self.root.after(...) polling loop to safely
           push log lines / progress updates from the worker thread back
           to the Tk main thread
        4. Re-enable buttons when done, show success/error in console
    - Keep this module free of AppleScript/Claude API calls directly --
      always go through app.py so ui.py stays swappable for the web UI.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Playlist AI")
        self._log_queue: queue.Queue[str] = queue.Queue()
        raise NotImplementedError("Part D: build the Tkinter layout")

    def _on_export_library(self) -> None:
        raise NotImplementedError("Part D: wire up Export Library button")

    def _on_analyze_songs(self) -> None:
        raise NotImplementedError("Part D: wire up Analyze Songs button")

    def _on_preview_results(self) -> None:
        raise NotImplementedError("Part D: wire up Preview Results button")

    def _on_create_playlists(self) -> None:
        raise NotImplementedError("Part D: wire up Create Playlists button")

    def _run_in_background(self, target, *args) -> None:
        """Helper: run `target(*args)` on a daemon thread.

        TODO(Part D): wrap target so exceptions get logged to the console
        panel instead of crashing the thread silently.
        """
        thread = threading.Thread(target=target, args=args, daemon=True)
        thread.start()

    def log(self, message: str) -> None:
        """Thread-safe append to the console output panel.

        TODO(Part D): push to self._log_queue and have a self.root.after
        poller drain it into the Text widget.
        """
        raise NotImplementedError("Part D: implement thread-safe logging to console")


def main() -> None:
    root = tk.Tk()
    app = PlaylistAIApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
