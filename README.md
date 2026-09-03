# tracksplit

DJ set VODs -> per-song clips, on Apple Silicon.

```
vods/<creator>/*.mp4  ->  out/<creator>/<vod>/{audio.wav, manifest.json, clips/, _trash/}
```

Pipeline (each stage idempotent, resumable, SQLite-tracked):

1. **extract**  ffmpeg -vn -> 16k mono wav
2. **gate**     PANNs CNN14 music/speech/silence per 1 s, hysteresis -> regions. Talk-over-beat stays music.
3. **fp**       ACRCloud on 15 s windows every 30 s *inside music regions only*, cached per (vod hash, offset)
4. **segment**  run-length merge same track, smooth flicker, chroma-novelty split unknown gaps, drop < 45 s
5. **cut**      ffmpeg -c copy with +-2 s keyframe pad; talk/silence logged but not written unless `export_talk`
6. **tui**      Textual triage: k/d/space/a

## Setup
```
brew install ffmpeg mpv uv
make setup
export ACR_HOST=identify-<region>.acrcloud.com ACR_KEY=... ACR_SECRET=...   # from your secret manager
make ingest VODS=~/vods
make dry          # extract + gate only; prints region map, zero ACR spend
make run          # full pipeline
make tui
```

First run downloads the PANNs checkpoint (~300 MB) to ~/panns_data.

## Notes
- `stop-after` lets you sanity-check the gate per creator before spending ACR queries. Tune `creators.toml`.
- Reset stages with `make reset`; ACR cache survives.
- Cuts snap to keyframes (stream VODs are typically ~2 s GOP). Re-encode path intentionally not included; add `-c:v h264_videotoolbox` if you need frame-accurate.
- `unknown` segments = music the fingerprinter couldn't ID (unreleased, mashups, edits). Still cut, still triageable.
