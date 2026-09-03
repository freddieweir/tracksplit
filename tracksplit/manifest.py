import json
from pathlib import Path

def write(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))

def read(path: Path) -> dict:
    return json.loads(path.read_text())
