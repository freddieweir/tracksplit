"""Turn fingerprint hits + gate regions into cut segments.
A *play* is a run of windows whose anchor (offset - play_offset_s = the song's start time in the VOD) is
continuous; catalogue ids and titles are labels only. Unknown gaps are cut at self-similarity novelty peaks."""
from __future__ import annotations
import math
import numpy as np
from collections import Counter
from dataclasses import dataclass, asdict
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
    anchor: float | None = None  # median (offset - play_offset_s) of the play, unclamped
    def to_dict(self): return asdict(self)
    @property
    def label(self):
        return f"{self.artist} - {self.title}" if self.kind == "song" else self.kind

def _anchor(h):
    return h["offset"] - h["hit"]["play_offset_s"] if h["hit"] else None

def _same_play(a, b, tol):
    return a is not None and b is not None and abs(a - b) <= tol

def _smooth(hits, tol):
    """A lone window disagreeing with both neighbours, which agree with each other, is assigned to their play."""
    for i in range(1, len(hits) - 1):
        a, b, c = _anchor(hits[i-1]), _anchor(hits[i]), _anchor(hits[i+1])
        if _same_play(a, c, tol) and not _same_play(a, b, tol):
            prev, shift = hits[i-1]["hit"], hits[i]["offset"] - hits[i-1]["offset"]
            hits[i] = dict(hits[i], hit=dict(prev, play_offset_s=prev["play_offset_s"] + shift))
    return hits

def _runs(rh, tol):
    """Consecutive windows grouped into plays (anchor-continuous, window to window) and unknown stretches."""
    runs, last = [], None
    for h in rh:
        a = _anchor(h)
        if runs and (a is None) == (last is None) and (a is None or _same_play(a, last, tol)):
            runs[-1].append(h)
        else:
            runs.append([h])
        last = a
    return runs

def build(regs: list[Region], hits: list[dict], cfg: CreatorCfg, wav: Path | None
          ) -> tuple[list[Segment], list[Segment]]:
    """Returns (segments, dropped). Dropped = too short with no same-label neighbour to join."""
    tol = cfg.fp_stride_s / 2
    hits = _smooth(sorted(hits, key=lambda h: h["offset"]), tol)
    segs: list[Segment] = []
    for r in regs:
        if r.kind != "music":
            segs.append(Segment(r.start, r.end, r.kind))
            continue
        rh = [h for h in hits if math.floor(r.start) <= h["offset"] < r.end]  # fingerprinter floors r.start
        out: list[Segment] = []
        for run in _runs(rh, tol):
            first = run[0]["offset"]
            if run[0]["hit"] is None:
                seg = Segment(first if out else r.start, 0.0, "unknown")
            else:
                seg = _mk(run, r.start)
                while out and out[-1].kind == "unknown" and out[-1].start >= seg.start:
                    out.pop()  # the play was already running under those unmatched windows
                if out and out[-1].start >= seg.start:
                    seg.start = first  # anchor would erase the previous play: distrust it
            if out:
                out[-1].end = seg.start
            elif seg.start > r.start:
                out.append(Segment(r.start, seg.start, "unknown"))
            out.append(seg)
        if out:
            out[-1].end = r.end
        else:
            out.append(Segment(r.start, r.end, "unknown"))
        segs += [s for s in out if s.end > s.start]
    segs = _split_unknown(segs, cfg, wav)
    return _drop_short(segs, cfg)

def _mk(run, region_start) -> Segment:
    """One play. Start = median anchor, i.e. where play_offset says the song began in the VOD; never later
    than the first matching window, never earlier than the region. Label = most frequent (artist, title)."""
    hs = [h["hit"] for h in run]
    anchor = float(np.median([_anchor(h) for h in run]))
    artist, title = Counter((h["artist"], h["title"]) for h in hs).most_common(1)[0][0]
    acr_id = next(h["acr_id"] for h in hs if (h["artist"], h["title"]) == (artist, title))
    start = max(min(anchor, run[0]["offset"]), region_start)
    return Segment(start, 0.0, "song", artist, title, float(np.mean([h["score"] for h in hs])), acr_id,
                   anchor=anchor)

def _split_unknown(segs, cfg, wav):
    """Cut long unknown stretches where the audio's self-similarity shows a boundary; otherwise leave whole."""
    if wav is None:
        return segs
    import soundfile as sf
    out = []
    for s in segs:
        if s.kind != "unknown" or s.end - s.start < 2 * cfg.min_segment_s:
            out.append(s); continue
        y, sr = sf.read(wav, start=int(s.start*16000), stop=int(s.end*16000), dtype="float32")
        edges = [s.start, *(s.start + _novelty_cuts(y, sr, cfg.min_segment_s)), s.end]
        out += [Segment(a, b, "unknown") for a, b in zip(edges, edges[1:])]
    return out

def _novelty_cuts(y, sr, min_s, kernel_s=40.0, delta=0.3) -> np.ndarray:
    """Foote novelty: cosine self-similarity of ~1 s log-mel frames, checkerboard kernel along the diagonal,
    librosa peak_pick with peaks >= min_s apart and >= min_s from either edge. Times relative to y."""
    import librosa
    hop = 4096
    M = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=hop, hop_length=hop, n_mels=64))
    M = librosa.util.sync(M, np.arange(0, M.shape[1], 4), aggregate=np.mean)  # ~1.02 s frames
    frame_s = 4 * hop / sr
    F = librosa.util.normalize(M - M.mean(axis=1, keepdims=True), norm=2, axis=0)
    S = F.T @ F
    L = int(kernel_s / frame_s)
    edge = np.r_[-np.ones(L), np.ones(L)] * np.hanning(2 * L)
    K = np.outer(edge, edge); K /= np.abs(K).sum()  # 0 on homogeneous audio, ~0.5 on a clean boundary
    nov = np.zeros(S.shape[0])
    for i in range(L, len(nov) - L):
        nov[i] = (S[i-L:i+L, i-L:i+L] * K).sum()
    nov = np.maximum(nov, 0)
    w = int(min_s / frame_s)
    peaks = librosa.util.peak_pick(nov, pre_max=w // 2, post_max=w // 2, pre_avg=w, post_avg=w, delta=delta, wait=w)
    return np.array([p for p in peaks if w <= p <= len(nov) - w], dtype=float) * frame_s

def _drop_short(segs, cfg):
    """Short music segments join a neighbour with the same label; otherwise they are dropped and returned."""
    out, dropped = [], []
    for i, s in enumerate(segs):
        if s.end - s.start >= cfg.min_segment_s or s.kind in ("talk", "silence"):
            out.append(s)
        elif out and out[-1].label == s.label:
            out[-1].end = s.end
        elif i + 1 < len(segs) and segs[i+1].label == s.label:
            segs[i+1].start = s.start
        else:
            dropped.append(s)
    return out, dropped
