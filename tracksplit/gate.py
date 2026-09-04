"""Music/no-music gate. PANNs CNN14 (AudioSet) on 1 s hops, hysteresis -> regions.
Talk over a beat still scores as music; pure speech, silence, alerts fall out."""
from __future__ import annotations
import numpy as np, soundfile as sf
from dataclasses import dataclass, asdict
from pathlib import Path
from .config import CreatorCfg

SR = 32000  # PANNs native rate

@dataclass
class Region:
    start: float
    end: float
    kind: str  # music | talk | silence
    def to_dict(self): return asdict(self)

_model = None
def _load():
    global _model
    if _model is None:
        from panns_inference import AudioTagging, labels
        import torch
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        at = AudioTagging(checkpoint_path=None, device=dev)
        if dev == "mps":  # panns_inference only knows cuda/cpu; move it ourselves
            at.model.to(dev); at.device = dev
        _model = (at, labels.index("Music"), labels.index("Speech"), labels.index("Silence"))
    return _model

def score(wav: Path, hop_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-hop probabilities (music, speech, silence)."""
    import librosa
    y, sr = sf.read(wav, dtype="float32")
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    model, mi, si, zi = _load()
    win = int(SR * max(hop_s, 1.0) * 2)  # 2 s analysis window
    hop = int(SR * hop_s)
    n = max(1, (len(y) - win) // hop + 1)
    frames = np.stack([y[i*hop:i*hop+win] for i in range(n)])
    probs = []
    for b in range(0, n, 64):
        clip, _ = model.inference(frames[b:b+64])
        probs.append(clip)
    p = np.concatenate(probs)
    return p[:, mi], p[:, si], p[:, zi]

def regions(music: np.ndarray, speech: np.ndarray, silence: np.ndarray,
            cfg: CreatorCfg, total_s: float) -> list[Region]:
    hop = cfg.gate_window_s
    open_n, close_n = int(cfg.gate_open_s / hop), int(cfg.gate_close_s / hop)
    on = music >= cfg.music_threshold
    out, state, run, start = [], "off", 0, 0.0
    for i, v in enumerate(on):
        t = i * hop
        if state == "off":
            run = run + 1 if v else 0
            if run >= open_n:
                if t - (run-1)*hop > start:
                    out.append(_nonmusic(start, t - (run-1)*hop, speech, silence, hop))
                start, state, run = t - (run-1)*hop, "on", 0
        else:
            run = run + 1 if not v else 0
            if run >= close_n:
                out.append(Region(start, t - (run-1)*hop, "music"))
                start, state, run = t - (run-1)*hop, "off", 0
    end = total_s
    out.append(Region(start, end, "music") if state == "on"
               else _nonmusic(start, end, speech, silence, hop))
    return [r for r in out if r.end - r.start > 0.5]

def _nonmusic(s, e, speech, silence, hop) -> Region:
    a, b = int(s / hop), max(int(e / hop), int(s / hop) + 1)
    kind = "talk" if speech[a:b].mean() > silence[a:b].mean() else "silence"
    return Region(s, e, kind)
