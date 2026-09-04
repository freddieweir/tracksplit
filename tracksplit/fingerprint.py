"""ACRCloud identify, cached per (vod_hash, offset). Only called inside music regions.
The fingerprint is computed locally with the ACRCloud SDK; only that (~4 KB) is uploaded, never audio."""
from __future__ import annotations
import base64, hashlib, hmac, io, json, os, time
import numpy as np, requests, soundfile as sf
from acrcloud import acrcloud_extr_tool
from pathlib import Path
from .config import CreatorCfg
from .gate import Region
from .queue import Queue

class ACR:
    QPS = 2  # account limit (free trial: 2); requests are spaced >= 1/QPS apart

    def __init__(self):
        self.host = os.environ["ACR_HOST"]
        self.key = os.environ["ACR_KEY"]
        self.secret = os.environ["ACR_SECRET"].encode()
        self._last = 0.0

    def identify(self, pcm16k: np.ndarray) -> dict:
        buf = io.BytesIO(); sf.write(buf, pcm16k, 16000, format="WAV", subtype="PCM_16")
        fp = acrcloud_extr_tool.create_fingerprint_by_filebuffer(buf.getvalue(), 0, len(pcm16k) // 16000, False)
        if not fp:  # silence / no landmarks: nothing to send (the SDK refuses these too), treat as a miss
            return {"status": {"msg": "No result (empty fingerprint, nothing sent)", "code": 1001}}
        for attempt in (1, 2):
            time.sleep(max(0.0, self._last + 1 / self.QPS - time.monotonic()))
            self._last = time.monotonic()
            ts = str(int(time.time()))
            sig_str = "\n".join(["POST", "/v1/identify", self.key, "fingerprint", "1", ts])
            sig = base64.b64encode(hmac.new(self.secret, sig_str.encode(), hashlib.sha1).digest()).decode()
            r = requests.post(f"https://{self.host}/v1/identify",
                files={"sample": ("sample", fp, "application/octet-stream")},
                data={"access_key": self.key, "sample_bytes": len(fp), "timestamp": ts,
                      "signature": sig, "data_type": "fingerprint", "signature_version": "1"},
                timeout=30)
            r.raise_for_status()
            resp = json.loads(r.content)  # ACR serves UTF-8 JSON without a charset; r.json() would assume ISO-8859-1
            code = resp.get("status", {}).get("code", -1)
            if code != 3015 or attempt == 2:
                break
            time.sleep(1.0)  # QPS exceeded: back off once
        if code not in (0, 1001):  # 3001 key / 3003 quota / 3014 sig / 3015 qps / 2004 ...: don't cache, don't spend more
            raise RuntimeError(f"ACR status {code}: {resp.get('status', {}).get('msg')}")
        return resp

def parse(resp: dict) -> dict | None:
    """Flatten ACR response to {artist, title, score, play_offset_s} or None."""
    try:
        m = resp["metadata"]["music"][0]
    except (KeyError, IndexError):
        return None
    return {"artist": ", ".join(a["name"] for a in m.get("artists", [])),
            "title": m.get("title", ""), "score": m.get("score", 0),
            "play_offset_s": m.get("play_offset_ms", 0) / 1000,
            "acr_id": m.get("acrid")}

def fingerprint_regions(wav: Path, vod_hash: str, regs: list[Region], cfg: CreatorCfg,
                        q: Queue, acr: ACR | None) -> list[dict]:
    """Returns hits: [{offset, hit|None}] ordered by offset. Skips cached offsets."""
    y, sr = sf.read(wav, dtype="float32")
    assert sr == 16000
    hits = []
    for r in regs:
        if r.kind != "music":
            continue
        t = int(r.start)
        while t + cfg.fp_sample_s <= r.end:
            resp = q.cache_get(vod_hash, t)
            if resp is None:
                if acr is None:
                    raise RuntimeError("ACR creds missing and offset not cached")
                resp = acr.identify(y[t*sr:(t+cfg.fp_sample_s)*sr])
                q.cache_put(vod_hash, t, resp)
            hits.append({"offset": t, "hit": parse(resp)})
            t += cfg.fp_stride_s
    return hits
