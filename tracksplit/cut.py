"""ffmpeg -c copy cuts with keyframe padding. Originals never touched."""
from __future__ import annotations
import re, subprocess
from pathlib import Path
from .config import CreatorCfg
from .segment import Segment

def _safe(s: str) -> str:
    return re.sub(r"[^\w\-. ]+", "_", s).strip()[:80] or "untitled"

def cut_all(src: Path, segs: list[Segment], out_dir: Path, cfg: CreatorCfg) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, s in enumerate(segs):
        if s.kind in ("talk", "silence") and not cfg.export_talk:
            continue
        name = f"{i:03d}_{_safe(s.label)}.mp4"
        dst = out_dir / name
        if dst.exists():
            written.append(dst); continue
        ss = max(0.0, s.start - cfg.keyframe_pad_s)
        to = s.end + cfg.keyframe_pad_s
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-ss", f"{ss:.3f}", "-to", f"{to:.3f}", "-i", str(src),
             "-c", "copy", "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", str(dst)],
            check=True)
        written.append(dst)
    return written
