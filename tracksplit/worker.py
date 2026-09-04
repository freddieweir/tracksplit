"""Stage runner. Each stage is idempotent and persists its artefact under out/<creator>/<vod-stem>/."""
from __future__ import annotations
import os, traceback
from pathlib import Path
from . import config, extract, gate, fingerprint, segment, cut, manifest
from .queue import Queue

def workdir(out: Path, row) -> Path:
    return out / row["creator"] / Path(row["path"]).stem

def run(db: Path, out: Path, creators_toml: Path | None, stop_after: str | None):
    q = Queue(db)
    raw = config.load(creators_toml)
    acr = None
    if all(k in os.environ for k in ("ACR_HOST", "ACR_KEY", "ACR_SECRET")):
        acr = fingerprint.ACR()
    while (row := q.next_job(stop_after)) is not None:
        cfg = config.for_creator(raw, row["creator"])
        wd = workdir(out, row)
        src = Path(row["path"])
        wav = wd / "audio.wav"
        man = wd / "manifest.json"
        try:
            _step(q, row, src, wav, man, wd, cfg, acr)
        except Exception as e:
            q.fail(row["id"], f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
            print(f"[FAIL] {src.name}: {e}")

def _step(q, row, src, wav, man, wd, cfg, acr):
    stage = row["stage"]
    m = manifest.read(man) if man.exists() else {"source": str(src), "creator": row["creator"], "hash": row["hash"]}

    if stage == "pending":
        extract.extract_audio(src, wav)
        m["duration_s"] = extract.duration_s(src)
        manifest.write(man, m); q.advance(row["id"], "extracted")
        print(f"[extract] {src.name} {m['duration_s']/3600:.2f}h")

    elif stage == "extracted":
        mu, sp, si = gate.score(wav, cfg.gate_window_s)
        regs = gate.regions(mu, sp, si, cfg, m["duration_s"])
        m["regions"] = [r.to_dict() for r in regs]
        manifest.write(man, m); q.advance(row["id"], "gated")
        music = sum(r.end - r.start for r in regs if r.kind == "music")
        print(f"[gate] {src.name}: {len(regs)} regions, {music/3600:.2f}h music of {m['duration_s']/3600:.2f}h")
        for r in regs:
            print(f"    {r.start/60:7.1f}m -> {r.end/60:7.1f}m  {r.kind}")

    elif stage == "gated":
        regs = [gate.Region(**r) for r in m["regions"]]
        hits = fingerprint.fingerprint_regions(wav, row["hash"], regs, cfg, q, acr)
        m["hits"] = hits
        manifest.write(man, m); q.advance(row["id"], "fingerprinted")
        print(f"[fp] {src.name}: {len(hits)} queries, {sum(1 for h in hits if h['hit'])} hits")

    elif stage == "fingerprinted":
        regs = [gate.Region(**r) for r in m["regions"]]
        segs, dropped = segment.build(regs, m["hits"], cfg, wav)
        m["segments"] = [s.to_dict() for s in segs]
        m["dropped"] = [s.to_dict() for s in dropped]
        manifest.write(man, m); q.advance(row["id"], "segmented")
        print(f"[segment] {src.name}: {len(segs)} segments, {len(dropped)} dropped")
        for s in segs:
            a = f"  (anchor {s.anchor/60:.1f}m)" if s.anchor is not None else ""
            print(f"    {s.start/60:7.1f}m -> {s.end/60:7.1f}m  {s.label}{a}")

    elif stage == "segmented":
        segs = [segment.Segment(**s) for s in m["segments"]]
        files = cut.cut_all(src, segs, wd / "clips", cfg)
        m["clips"] = [str(f) for f in files]
        manifest.write(man, m); q.advance(row["id"], "cut")
        print(f"[cut] {src.name}: {len(files)} clips")
