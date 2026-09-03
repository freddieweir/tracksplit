"""Turn fingerprint hits + gate regions into cut segments.
Run-length merge same-track hits, smooth 1-window flicker, novelty-split unknown gaps."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, asdict, field
from pathlib import Path
from .config import CreatorCfg
from .gate import Region

@dataclass
class Segment:
    start: float
    end: float
    kind: str                # song | unknown | talk | silence
    artist: str = ""
    title: str = ""
    confidence: float = 0.0
    acr_id: str | None = None
    keep: bool | None = None  # triage decision
    def to_dict(self): return asdict(self)
    @property
    def label(self):
        return f"{self.artist} - {self.title}" if self.kind == "song" else self.kind

def _key(h): return h["hit"]["acr_id"] if h["hit"] else None

def _smooth(hits):
    """A lone window differing from both neighbours inherits the neighbour."""
    keys = [_key(h) for h in hits]
    for i in range(1, len(hits) - 1):
        if keys[i-1] == keys[i+1] and keys[i] != keys[i-1] and keys[i-1] is not None:
            hits[i] = dict(hits[i], hit=hits[i-1]["hit"])
    return hits

def build(regs: list[Region], hits: list[dict], cfg: CreatorCfg, wav: Path | None) -> list[Segment]:
    hits = _smooth(sorted(hits, key=lambda h: h["offset"]))
    segs: list[Segment] = []
    for r in regs:
        if r.kind != "music":
            segs.append(Segment(r.start, r.end, r.kind))
            continue
        rh = [h for h in hits if r.start <= h["offset"] < r.end]
        cur, cur_start = None, r.start
        for h in rh:
            k = _key(h)
            if k != cur:
                if cur is not None or h["offset"] > cur_start:
                    segs.append(_mk(cur_start, h["offset"], rh, cur))
                cur, cur_start = k, h["offset"]
        segs.append(_mk(cur_start, r.end, rh, cur))
    segs = _split_unknown(segs, cfg, wav)
    return _drop_short(segs, cfg)

def _mk(s, e, rh, key) -> Segment:
    if key is None:
        return Segment(s, e, "unknown")
    hs = [h["hit"] for h in rh if h["hit"] and h["hit"]["acr_id"] == key]
    h = hs[0]
    # ACR play_offset tells us where in the song the window landed -> back-calc true start
    return Segment(s, e, "song", h["artist"], h["title"],
                   float(np.mean([x["score"] for x in hs])), key)

def _split_unknown(segs, cfg, wav):
    if wav is None:
        return segs
    import librosa, soundfile as sf
    out = []
    for s in segs:
        if s.kind != "unknown" or s.end - s.start < 3 * cfg.min_segment_s:
            out.append(s); continue
        y, sr = sf.read(wav, start=int(s.start*16000), stop=int(s.end*16000), dtype="float32")
        bounds = librosa.segment.agglomerative(
            librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=4096),
            k=max(2, int((s.end - s.start) / 240)))
        times = librosa.frames_to_time(bounds, sr=sr, hop_length=4096) + s.start
        edges = [s.start, *times[1:], s.end]
        out += [Segment(a, b, "unknown") for a, b in zip(edges, edges[1:])]
    return out

def _drop_short(segs, cfg):
    out = []
    for s in segs:
        if s.end - s.start >= cfg.min_segment_s or s.kind in ("talk", "silence"):
            out.append(s)
        elif out and out[-1].kind == s.kind:
            out[-1].end = s.end
        elif out:
            out[-1].end = s.end  # absorb into previous
    return out
