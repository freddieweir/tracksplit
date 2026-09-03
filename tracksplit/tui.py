"""Triage: one row per segment across all manifests. k=keep d=discard space=preview (mpv) a=apply."""
from __future__ import annotations
import shutil, subprocess
from pathlib import Path
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Footer, Header
from . import manifest

class Triage(App):
    BINDINGS = [("k", "mark(True)", "Keep"), ("d", "mark(False)", "Discard"),
                ("space", "preview", "Preview"), ("a", "apply", "Apply"), ("q", "quit", "Quit")]

    def __init__(self, out: Path):
        super().__init__()
        self.out = out
        self.rows = []  # (manifest_path, idx)
        self.mans = {}

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(cursor_type="row")
        yield Footer()

    def on_mount(self):
        t = self.query_one(DataTable)
        t.add_columns("keep", "creator", "vod", "start", "len", "kind", "label", "conf")
        for mp in sorted(self.out.rglob("manifest.json")):
            m = manifest.read(mp); self.mans[mp] = m
            for i, s in enumerate(m.get("segments", [])):
                if s["kind"] in ("talk", "silence"):
                    continue
                self.rows.append((mp, i))
                t.add_row(self._mark(s), m["creator"], mp.parent.name,
                          f"{s['start']/60:.1f}m", f"{(s['end']-s['start'])/60:.1f}m",
                          s["kind"], f"{s['artist']} - {s['title']}" if s["kind"] == "song" else "?",
                          f"{s['confidence']:.0f}")
        self.title = f"tracksplit triage: {len(self.rows)} segments"

    @staticmethod
    def _mark(s): return {True: "✓", False: "✗"}.get(s.get("keep"), " ")

    def _cur(self):
        t = self.query_one(DataTable); mp, i = self.rows[t.cursor_row]
        return t, mp, self.mans[mp]["segments"][i]

    def action_mark(self, keep: bool):
        t, mp, s = self._cur(); s["keep"] = keep
        manifest.write(mp, self.mans[mp])
        t.update_cell_at((t.cursor_row, 0), self._mark(s))
        t.action_cursor_down()

    def action_preview(self):
        _, mp, s = self._cur()
        if shutil.which("mpv"):
            subprocess.Popen(["mpv", "--really-quiet", f"--start={s['start']}", f"--length=20",
                              self.mans[mp]["source"]])

    def action_apply(self):
        """Move discarded clips to _trash/ next to clips/."""
        n = 0
        for mp, m in self.mans.items():
            clips = {Path(c).name[:3]: Path(c) for c in m.get("clips", [])}
            for i, s in enumerate(m.get("segments", [])):
                c = clips.get(f"{i:03d}")
                if s.get("keep") is False and c and c.exists():
                    trash = c.parent.parent / "_trash"; trash.mkdir(exist_ok=True)
                    c.rename(trash / c.name); n += 1
        self.notify(f"moved {n} clips to _trash")
