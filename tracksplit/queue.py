"""SQLite job queue. Stages: pending -> extracted -> gated -> fingerprinted -> segmented -> cut."""
from __future__ import annotations
import hashlib, json, sqlite3, time
from pathlib import Path

STAGES = ["pending", "extracted", "gated", "fingerprinted", "segmented", "cut"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS vods (
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE NOT NULL,
  creator TEXT NOT NULL,
  hash TEXT NOT NULL,
  stage TEXT NOT NULL DEFAULT 'pending',
  error TEXT,
  updated REAL
);
CREATE TABLE IF NOT EXISTS acr_cache (
  vod_hash TEXT NOT NULL,
  offset_s INTEGER NOT NULL,
  response TEXT NOT NULL,
  PRIMARY KEY (vod_hash, offset_s)
);
"""

def quick_hash(p: Path) -> str:
    """Hash size + first/last 4 MB. Full hashing 100 GB is pointless."""
    h = hashlib.blake2b(digest_size=16)
    size = p.stat().st_size
    h.update(str(size).encode())
    with p.open("rb") as f:
        h.update(f.read(4 << 20))
        if size > 8 << 20:
            f.seek(-(4 << 20), 2)
            h.update(f.read())
    return h.hexdigest()

class Queue:
    def __init__(self, db: Path):
        db.parent.mkdir(parents=True, exist_ok=True)
        self.c = sqlite3.connect(db)
        self.c.row_factory = sqlite3.Row
        self.c.executescript(SCHEMA)

    def ingest(self, root: Path) -> int:
        n = 0
        for p in sorted(root.rglob("*.mp4")):
            creator = p.parent.name if p.parent != root else "default"
            try:
                self.c.execute(
                    "INSERT INTO vods(path, creator, hash, updated) VALUES (?,?,?,?)",
                    (str(p), creator, quick_hash(p), time.time()))
                n += 1
            except sqlite3.IntegrityError:
                pass
        self.c.commit()
        return n

    def next_job(self, stop_after: str | None):
        limit = STAGES.index(stop_after) if stop_after else len(STAGES) - 1
        return self.c.execute(
            "SELECT * FROM vods WHERE error IS NULL AND stage IN (%s) ORDER BY id LIMIT 1"
            % ",".join("?" * limit), STAGES[:limit]).fetchone()

    def advance(self, vod_id: int, stage: str):
        self.c.execute("UPDATE vods SET stage=?, updated=? WHERE id=?", (stage, time.time(), vod_id))
        self.c.commit()

    def fail(self, vod_id: int, err: str):
        self.c.execute("UPDATE vods SET error=?, updated=? WHERE id=?", (err[:2000], time.time(), vod_id))
        self.c.commit()

    def cache_get(self, vod_hash: str, offset: int):
        r = self.c.execute("SELECT response FROM acr_cache WHERE vod_hash=? AND offset_s=?",
                           (vod_hash, offset)).fetchone()
        return json.loads(r[0]) if r else None

    def cache_put(self, vod_hash: str, offset: int, resp: dict):
        self.c.execute("INSERT OR REPLACE INTO acr_cache VALUES (?,?,?)",
                       (vod_hash, offset, json.dumps(resp)))
        self.c.commit()

    def status(self):
        return self.c.execute("SELECT stage, COUNT(*) n, SUM(error IS NOT NULL) failed "
                              "FROM vods GROUP BY stage").fetchall()

    def reset(self):
        self.c.execute("UPDATE vods SET stage='pending', error=NULL")
        self.c.commit()
