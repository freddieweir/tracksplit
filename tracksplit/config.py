from __future__ import annotations
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

@dataclass
class CreatorCfg:
    gate_window_s: float = 1.0
    gate_open_s: float = 10.0
    gate_close_s: float = 8.0
    music_threshold: float = 0.45
    fp_sample_s: int = 15
    fp_stride_s: int = 30
    min_segment_s: int = 45
    keyframe_pad_s: float = 2.0
    export_talk: bool = False

def load(path: Path | None) -> dict[str, dict]:
    if not path or not path.exists():
        return {"default": {}}
    return tomllib.loads(path.read_text())

def for_creator(raw: dict[str, dict], creator: str) -> CreatorCfg:
    merged = dict(raw.get("default", {}))
    merged.update(raw.get(creator, {}))
    names = {f.name for f in fields(CreatorCfg)}
    return CreatorCfg(**{k: v for k, v in merged.items() if k in names})
