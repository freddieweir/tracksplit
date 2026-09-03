"""Track identity = continuity of anchor (offset - play_offset_s); ids and titles are labels only."""
import numpy as np
import soundfile as sf
from tracksplit import segment
from tracksplit.config import CreatorCfg
from tracksplit.gate import Region

CFG = CreatorCfg()  # fp_stride 30 s -> anchor tolerance 15 s; min_segment 45 s


def hit(offset, acr_id=None, anchor=None, title=None, score=90.0):
    """Fingerprint window at `offset`. A match reports play_offset such that offset - play_offset == anchor."""
    if acr_id is None:
        return {"offset": offset, "hit": None}
    return {"offset": offset, "hit": {"artist": "Artist", "title": title or acr_id, "score": score,
                                      "play_offset_s": offset - anchor, "acr_id": acr_id}}


def build(regs, hits, wav=None):
    return segment.build(regs, hits, CFG, wav)


def test_one_play_matched_under_two_catalogue_ids_is_one_segment():
    # ACR flips to the radio-edit entry for two windows; the anchors agree, so it is one play
    hits = [hit(t, "A-radio" if t in (60, 90) else "A-ext", anchor=0,
                title="Song A" if t in (60, 90) else "Song A (Extended Mix)") for t in range(0, 210, 30)]
    hits += [hit(t, "B", anchor=205) for t in range(210, 300, 30)]
    segs, dropped = build([Region(0, 300, "music")], hits)
    assert [(s.kind, s.title) for s in segs] == [("song", "Song A (Extended Mix)"), ("song", "B")]
    assert segs[0].acr_id == "A-ext"  # the majority label carries its id
    assert dropped == []


def test_short_misfire_between_different_labels_is_dropped_and_logged():
    hits = [hit(t, "A", anchor=0) for t in range(0, 120, 30)]
    hits += [hit(120, "B", anchor=110)]                          # B: 110 -> 150, 40 s, no B neighbour
    hits += [hit(t, "C", anchor=150) for t in range(150, 300, 30)]
    segs, dropped = build([Region(0, 300, "music")], hits)
    assert [s.title for s in segs] == ["A", "C"]
    assert [(d.title, d.start, d.end) for d in dropped] == [("B", 110.0, 150.0)]


def test_short_segment_merges_only_into_same_label_neighbour():
    hits = [hit(t, "A", anchor=0) for t in range(0, 120, 30)]
    hits += [hit(120, "A", anchor=115)]                          # A spun back from the top: new play, same label, 35 s
    hits += [hit(t, "B", anchor=150) for t in range(150, 300, 30)]
    segs, dropped = build([Region(0, 300, "music")], hits)
    assert [(s.title, s.start, s.end) for s in segs] == [("A", 0.0, 150.0), ("B", 150.0, 300.0)]
    assert dropped == []


def test_anchor_sets_start_and_separates_replays():
    region = Region(10, 400, "music")
    hits = [hit(10), hit(40)]                                    # ACR missed the intro
    hits += [hit(t, "A", anchor=5) for t in range(70, 220, 30)]  # play began at 5, before the gate opened
    hits += [hit(t, "A", anchor=250 + 0.2 * i)                   # replay from the top; anchors drift with pitch
             for i, t in enumerate(range(250, 400, 30))]
    segs, dropped = build([region], hits)
    assert [(s.title, s.start, s.end) for s in segs] == [("A", 10.0, 250.0), ("A", 250.0, 400.0)]
    assert segs[0].anchor == 5.0                                 # raw median anchor survives the clamp
    assert abs(segs[1].anchor - 250.4) < 1e-9
    assert dropped == []


def test_first_window_survives_fractional_region_start():
    # the fingerprinter floors the region start, so a region opening at 10.5 has its first window at 10
    segs, dropped = build([Region(10.5, 30, "music")], [hit(10, "A", anchor=10)])
    assert segs == []
    assert [(d.kind, d.title) for d in dropped] == [("song", "A")]  # seen as A, not as unknown


def _chords(roots, seconds, sr=16000, harmonics=(1.0, 0.5, 0.33)):
    out = []
    for root in roots:
        t = np.arange(2 * sr) / sr
        out.append(sum(a * np.sin(2 * np.pi * root * r * h * t)
                       for r in (1.0, 1.25, 1.5) for h, a in zip((1, 2, 3), harmonics)))
    loop = np.concatenate(out)
    y = np.tile(loop, int(np.ceil(seconds / 8)))[: seconds * sr]
    return (0.2 * y / np.abs(y).max()).astype(np.float32)


def test_unknown_is_cut_only_where_novelty_peaks(tmp_path):
    a = _chords([261.6, 392.0, 440.0, 349.2], 120)                      # C G Am F, bright
    b = _chords([146.8, 196.0, 220.0, 174.6], 120, harmonics=(1.0, 0.1, 0.02))  # D A Bm G, dark
    unknown = [hit(t) for t in range(0, 240, 30)]
    two = tmp_path / "two.wav"; sf.write(two, np.concatenate([a, b]), 16000)
    segs, _ = build([Region(0, 240, "music")], unknown, two)
    assert len(segs) == 2 and abs(segs[0].end - 120) < 10
    one = tmp_path / "one.wav"; sf.write(one, np.concatenate([a, a]), 16000)
    segs, _ = build([Region(0, 240, "music")], unknown, one)
    assert [(s.start, s.end) for s in segs] == [(0.0, 240.0)]