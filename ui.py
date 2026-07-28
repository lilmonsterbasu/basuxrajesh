"""ui.py

OWNER: Part D (GUI + Web App Design) -- Tkinter desktop half.

Provides the desktop GUI: 4 action buttons, a progress bar, and a console
output panel. Long-running work (export/classify/create) runs on a background
thread so the Tk mainloop doesn't freeze.

Nothing here talks to AppleScript or the Claude API directly -- every action
goes through app.py, which keeps this module swappable with the web UI.
"""

from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
import traceback
from tkinter import messagebox, ttk

import app as orchestrator
from music import Song

logger = logging.getLogger(__name__)

POLL_INTERVAL_MS = 100


class QueueLogHandler(logging.Handler):
    """Funnels log records from worker threads into the GUI console.

    app.py, music.py, and classifier.py all report progress through the
    logging module, so routing that into the console panel is what makes
    "Creating playlist Coding..." appear without any extra plumbing.
    """

    def __init__(self, log_queue: queue.Queue) -> None:
        super().__init__()
        self._queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        self._queue.put(("log", self.format(record)))


class PlaylistAIApp:
    """Main Tkinter application window."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Playlist AI")
        self.root.geometry("900x680")
        self.root.minsize(720, 540)
        self._log_queue: queue.Queue = queue.Queue()
        self._plan: dict[str, list[Song]] = {}
        self._busy = False

        self._build_layout()
        self._attach_log_handler()
        self._drain_queue()

        self.log("Ready. Start with 'Export Library'.")

    # -- layout -------------------------------------------------------------

    def _build_layout(self) -> None:
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=3)  # preview
        root.rowconfigure(3, weight=2)  # console

        actions = ttk.Frame(root, padding=(12, 12, 12, 6))
        actions.grid(row=0, column=0, sticky="ew")
        for col in range(4):
            actions.columnconfigure(col, weight=1)

        self._buttons: dict[str, ttk.Button] = {}
        specs = [
            ("export", "1. Export Library", self._on_export_library),
            ("analyze", "2. Analyze Songs", self._on_analyze_songs),
            ("preview", "3. Preview Results", self._on_preview_results),
            ("create", "4. Create Playlists", self._on_create_playlists),
        ]
        for col, (key, label, command) in enumerate(specs):
            button = ttk.Button(actions, text=label, command=command)
            button.grid(row=0, column=col, sticky="ew", padx=4)
            self._buttons[key] = button

        progress_frame = ttk.Frame(root, padding=(12, 0, 12, 6))
        progress_frame.grid(row=1, column=0, sticky="ew")
        progress_frame.columnconfigure(0, weight=1)

        self._progress = ttk.Progressbar(progress_frame, mode="determinate", maximum=100)
        self._progress.grid(row=0, column=0, sticky="ew")

        self._status = ttk.Label(progress_frame, text="Idle", anchor="w")
        self._status.grid(row=1, column=0, sticky="ew", pady=(4, 0))

        preview_frame = ttk.LabelFrame(root, text="Playlist preview", padding=6)
        preview_frame.grid(row=2, column=0, sticky="nsew", padx=12, pady=6)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        self._tree = ttk.Treeview(preview_frame, columns=("artist",), show="tree headings")
        self._tree.heading("#0", text="Playlist / Song")
        self._tree.heading("artist", text="Artist")
        self._tree.column("#0", width=420, stretch=True)
        self._tree.column("artist", width=240, stretch=True)
        self._tree.grid(row=0, column=0, sticky="nsew")

        tree_scroll = ttk.Scrollbar(
            preview_frame, orient="vertical", command=self._tree.yview
        )
        tree_scroll.grid(row=0, column=1, sticky="ns")
        self._tree.configure(yscrollcommand=tree_scroll.set)

        console_frame = ttk.LabelFrame(root, text="Console", padding=6)
        console_frame.grid(row=3, column=0, sticky="nsew", padx=12, pady=(0, 12))
        console_frame.columnconfigure(0, weight=1)
        console_frame.rowconfigure(0, weight=1)

        self._console = tk.Text(console_frame, height=10, wrap="word", state="disabled")
        self._console.grid(row=0, column=0, sticky="nsew")

        console_scroll = ttk.Scrollbar(
            console_frame, orient="vertical", command=self._console.yview
        )
        console_scroll.grid(row=0, column=1, sticky="ns")
        self._console.configure(yscrollcommand=console_scroll.set)

    def _attach_log_handler(self) -> None:
        handler = QueueLogHandler(self._log_queue)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        if root_logger.level == logging.NOTSET or root_logger.level > logging.INFO:
            root_logger.setLevel(logging.INFO)

    # -- thread-safe plumbing ----------------------------------------------

    def log(self, message: str) -> None:
        """Thread-safe append to the console output panel."""
        self._log_queue.put(("log", message))

    def _drain_queue(self) -> None:
        """Poll the queue on the Tk main thread and apply updates.

        Every widget mutation happens here. Worker threads only ever put
        messages on the queue -- touching Tk from a worker thread is what
        makes Tkinter apps deadlock or crash.
        """
        try:
            while True:
                self._handle_message(self._log_queue.get_nowait())
        except queue.Empty:
            pass
        finally:
            self.root.after(POLL_INTERVAL_MS, self._drain_queue)

    def _handle_message(self, message: tuple) -> None:
        kind = message[0]
        if kind == "log":
            self._append_console(message[1])
        elif kind == "status":
            self._status.configure(text=message[1])
        elif kind == "progress":
            done, total = message[1], message[2]
            self._progress.stop()
            self._progress.configure(
                mode="determinate", maximum=max(total, 1), value=done
            )
            self._status.configure(text=f"Classifying {done}/{total} songs...")
        elif kind == "busy":
            if message[1]:
                self._progress.configure(mode="indeterminate")
                self._progress.start(12)
            else:
                self._progress.stop()
                self._progress.configure(mode="determinate", value=0)
        elif kind == "plan":
            self._plan = message[1]
            self._render_plan(self._plan)
        elif kind == "finished":
            self._set_busy(False)

    def _append_console(self, text: str) -> None:
        self._console.configure(state="normal")
        self._console.insert("end", text + "\n")
        self._console.see("end")
        self._console.configure(state="disabled")

    def _render_plan(self, plan: dict[str, list[Song]]) -> None:
        self._tree.delete(*self._tree.get_children())
        for name in sorted(plan):
            songs = plan[name]
            parent = self._tree.insert("", "end", text=f"{name}  ({len(songs)})")
            for song in songs:
                self._tree.insert(parent, "end", text=song.title, values=(song.artist,))

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        for button in self._buttons.values():
            button.configure(state=state)
        if not busy:
            self._progress.stop()
            self._progress.configure(mode="determinate", value=0)
            self._status.configure(text="Idle")

    def _run_in_background(self, target, *args) -> None:
        """Run `target(*args)` on a daemon thread.

        The wrapper is the important part: an unhandled exception in a worker
        thread would otherwise vanish silently, leaving the buttons disabled
        forever with no clue why.
        """
        if self._busy:
            return
        self._set_busy(True)

        def runner() -> None:
            try:
                target(*args)
            except Exception as exc:  # noqa: BLE001 -- surface everything to the console
                self.log(f"ERROR: {exc}")
                self.log(traceback.format_exc().rstrip())
            finally:
                self._log_queue.put(("finished",))

        threading.Thread(target=runner, daemon=True).start()

    # -- button handlers ----------------------------------------------------

    def _on_export_library(self) -> None:
        self._run_in_background(self._export_library)

    def _export_library(self) -> None:
        self._log_queue.put(("status", "Exporting library from Apple Music..."))
        self._log_queue.put(("busy", True))
        self.log("Exporting library from Apple Music (this may take a minute)...")
        songs = orchestrator.run_export_library()
        self.log(f"Exported {len(songs)} songs.")

    def _on_analyze_songs(self) -> None:
        self._run_in_background(self._analyze_songs)

    def _analyze_songs(self) -> None:
        self._log_queue.put(("status", "Classifying songs..."))
        self._log_queue.put(("busy", True))
        self.log("Sending songs to the classifier...")
        results = orchestrator.run_analyze_songs(
            on_progress=lambda done, total: self._log_queue.put(("progress", done, total))
        )
        self.log(f"Classified {len(results)} songs.")

    def _on_preview_results(self) -> None:
        self._run_in_background(self._preview_results)

    def _preview_results(self) -> None:
        self._log_queue.put(("status", "Building playlist plan..."))
        self._log_queue.put(("busy", True))
        self.log("Building playlist plan from the last analysis...")
        results = orchestrator.load_analysis()
        plan = orchestrator.run_build_playlist_plan(results)
        self._log_queue.put(("plan", plan))
        placements = sum(len(songs) for songs in plan.values())
        self.log(
            f"Plan ready: {len(plan)} playlists, {placements} song placements. "
            "Nothing has been written to Apple Music yet."
        )

    def _on_create_playlists(self) -> None:
        """Write the plan to Apple Music, after an explicit confirmation.

        This is the only step that changes the user's library, so it asks
        first and insists on a plan the user has actually seen.
        """
        plan = self._plan
        if not plan:
            try:
                plan = orchestrator.load_playlist_plan()
            except FileNotFoundError:
                messagebox.showwarning(
                    "No plan yet",
                    "Run 'Preview Results' first so you can see what will be "
                    "created before anything touches Apple Music.",
                )
                return
            self._plan = plan
            self._render_plan(plan)

        placements = sum(len(songs) for songs in plan.values())
        confirmed = messagebox.askokcancel(
            "Create playlists?",
            f"This will create or update {len(plan)} playlists in Apple Music "
            f"and add {placements} songs.\n\nExisting playlists with these names "
            "are added to, not replaced.",
        )
        if not confirmed:
            self.log("Cancelled -- nothing was written to Apple Music.")
            return

        self._run_in_background(self._create_playlists, plan)

    def _create_playlists(self, plan: dict[str, list[Song]]) -> None:
        self._log_queue.put(("status", "Writing playlists to Apple Music..."))
        self._log_queue.put(("busy", True))
        self.log(f"Creating {len(plan)} playlists in Apple Music...")
        orchestrator.run_create_playlists(plan)
        self.log("Done. Check Apple Music for your new playlists.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    root = tk.Tk()
    PlaylistAIApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
