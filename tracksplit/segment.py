"""Turn fingerprint hits + gate regions into cut segments.
A *play* is a run of windows whose anchor (offset - play_offset_s = the song's start time in the VOD) is
continuous; catalogue ids and titles are labels only. Unknown gaps are cut at self-similarity novelty peaks."""
from __future__ import annotations
import math, re
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

def _norm_title(t: str) -> str:
    """'Song A (Radio Mix)' -> 'song a'; strips bracketed/dashed edition suffixes and feat. tails."""
    t = re.sub(r"[\(\[].*?[\)\]]", " ", t.lower())
    t = re.sub(r"\s+-\s+((the )?best of|greatest hits|deluxe|edition|version|live|remix|radio|extended|original"
               r"|single|album|club|edit|remaster|\d{4} remaster).*$", " ", t)
    t = re.sub(r"\b(feat|ft|featuring)\.?\s.*$", " ", t)
    return re.sub(r"[^a-z0-9]+", " ", t).strip()

def _same_title(x: str, y: str) -> bool:
    """Equal after normalisation, or one is the other with an artist prefixed ('Some Artist - Some Title')."""
    a, b = _norm_title(x), _norm_title(y)
    return bool(a) and (a == b or a.endswith(" " + b) or b.endswith(" " + a))

def _same_label(p: "Segment", q: "Segment") -> bool:
    return _same_title(p.title, q.title) if p.kind == q.kind == "song" else p.kind == q.kind

def _continues(prev, h, tol) -> bool:
    """h extends prev's run: both unknown, or the same play by anchor, or the same title
    (loop-based tracks and live re-cues make play_offset jump although the play never stopped)."""
    a, b = _anchor(prev), _anchor(h)
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol or _same_title(prev["hit"]["title"], h["hit"]["title"])

def _smooth(hits, tol):
    """A lone window that does not continue its predecessor, while the neighbours continue each other
    (by anchor or title), is assigned to their play."""
    for i in range(1, len(hits) - 1):
        p, m, n = hits[i-1], hits[i], hits[i+1]
        if p["hit"] and n["hit"] and _continues(p, n, tol) and not _continues(p, m, tol):
            shift = m["offset"] - p["offset"]
            hits[i] = dict(m, hit=dict(p["hit"], play_offset_s=p["hit"]["play_offset_s"] + shift))
    return hits

def _runs(rh, tol):
    """Consecutive windows grouped into plays (window-to-window continuity) and unknown stretches."""
    runs = []
    for h in rh:
        if runs and _continues(runs[-1][-1], h, tol):
            runs[-1].append(h)
        else:
            runs.append([h])
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
    segs = _bridge(segs, cfg)
    segs = _split_unknown(segs, cfg, wav)
    return _drop_short(segs, cfg)

def _bridge_target(out, s, cfg):
    """Index of the earlier song that s continues, or None. Everything between must be shorter than a
    segment: up to 2*min_segment_s of it (gate breaks, misfires) when the anchors agree, up to
    min_segment_s of non-music when only the title agrees."""
    tol, mn = cfg.fp_stride_s / 2, cfg.min_segment_s
    gap, song_between = 0.0, False
    for k in range(len(out) - 1, -1, -1):
        p = out[k]
        if p.kind == "song" and gap > 0 and (_same_play(p.anchor, s.anchor, tol) or
                                             (not song_between and gap <= mn and _same_title(p.title, s.title))):
            return k
        if p.end - p.start >= mn:
            return None  # a real segment sits between
        gap += p.end - p.start
        song_between = song_between or p.kind == "song"
        if gap > 2 * mn:
            return None
    return None

def _bridge(segs, cfg):
    """One play split by a short gap (a breakdown the gate closed on, a misfire) is rejoined."""
    out: list[Segment] = []
    for s in segs:
        k = _bridge_target(out, s, cfg) if s.kind == "song" else None
        if k is None:
            out.append(s)
        else:
            p = out[k]; del out[k+1:]
            p.end, p.confidence = s.end, (p.confidence + s.confidence) / 2
    return out

def _mk(run, region_start) -> Segment:
    """One play. Start = median anchor, i.e. where play_offset says the song began in the VOD; never later
    than the first matching window, never earlier than the region. Label = most frequent (artist, title)."""
    hs = [h["hit"] for h in run]
    anchors = [_anchor(h) for h in run]
    tol = 15.0 if len(run) < 2 else max(15.0, abs(run[1]["offset"] - run[0]["offset"]) / 2)
    anchor = float(np.median([a for a in anchors if abs(a - anchors[0]) <= tol]))  # the first window's cluster;
    # later clusters are loop ambiguity or a re-cue, not where this play began
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
        elif out and _same_label(out[-1], s):
            out[-1].end = s.end
        elif i + 1 < len(segs) and _same_label(segs[i+1], s):
            segs[i+1].start = s.start
        else:
            dropped.append(s)
    return out, dropped
