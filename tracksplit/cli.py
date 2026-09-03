import argparse, os
from pathlib import Path
from . import worker
from .queue import Queue, STAGES

def main():
    p = argparse.ArgumentParser(prog="tracksplit")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ingest"); a.add_argument("root", type=Path); a.add_argument("--db", type=Path, required=True)
    r = sub.add_parser("run"); r.add_argument("--db", type=Path, required=True); r.add_argument("--out", type=Path, required=True)
    r.add_argument("--creators", type=Path, default=Path("creators.toml"))
    r.add_argument("--stop-after", choices=["extracted", "gated", "fingerprinted", "segmented"], dest="stop")
    r.add_argument("--dry", action="store_true", help="alias for --stop-after gated")
    t = sub.add_parser("tui"); t.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("status"); s.add_argument("--db", type=Path, required=True)
    x = sub.add_parser("reset"); x.add_argument("--db", type=Path, required=True)
    args = p.parse_args()

    if args.cmd == "ingest":
        print(f"ingested {Queue(args.db).ingest(args.root)} new VODs")
    elif args.cmd == "run":
        stop = "gated" if args.dry else args.stop
        # Makefile passes the stage *name*; map "gate" -> "gated" for convenience
        stop = {"gate": "gated", "extract": "extracted", "fp": "fingerprinted", "segment": "segmented"}.get(stop, stop)
        worker.run(args.db, args.out, args.creators, stop)
    elif args.cmd == "tui":
        from .tui import Triage; Triage(args.out).run()
    elif args.cmd == "status":
        for row in Queue(args.db).status():
            print(f"{row['stage']:14} {row['n']:4}  failed={row['failed'] or 0}")
    elif args.cmd == "reset":
        Queue(args.db).reset(); print("reset")
