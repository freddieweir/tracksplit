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


def test_anchor_sets_start_and_replays_stay_separate_only_across_other_tracks():
    region = Region(10, 400, "music")
    hits = [hit(10), hit(40)]                                    # ACR missed the intro
    hits += [hit(t, "A", anchor=5) for t in range(70, 220, 30)]  # play began at 5, before the gate opened
    hits += [hit(t, "A", anchor=250 + 0.2 * i)                   # back-to-back replay: same title, joins the play
             for i, t in enumerate(range(250, 400, 30))]
    segs, dropped = build([region], hits)
    assert [(s.title, s.start, s.end) for s in segs] == [("A", 10.0, 400.0)]
    assert segs[0].anchor == 5.0                                 # anchor of the first cluster survives the clamp
    assert dropped == []
    # the same track after a different one is a separate play
    hits = [hit(t, "A", anchor=0) for t in range(0, 120, 30)] + [hit(t, "B", anchor=120) for t in range(120, 240, 30)]
    hits += [hit(t, "A", anchor=240 + 0.2 * i) for i, t in enumerate(range(240, 400, 30))]
    segs, _ = build([Region(0, 400, "music")], hits)
    assert [(s.title, s.start) for s in segs] == [("A", 0.0), ("B", 120.0), ("A", 240.0)]
    assert abs(segs[2].anchor - 240.5) < 1e-9                    # six anchors 240.0..241.0 drifting: median


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


def test_loop_track_with_jumping_anchors_is_one_play():
    # a 4-bar-loop track: ACR's play_offset is ambiguous, so the anchor jumps although the play never stopped
    anchors = [0, 0, 90, 30, 120, 0, 60, 0]
    hits = [hit(30 * i, "Loop", anchor=a, title="Loop Song") for i, a in enumerate(anchors)]
    hits += [hit(t, "B", anchor=240) for t in range(240, 330, 30)]
    segs, dropped = build([Region(0, 330, "music")], hits)
    assert [(s.title, s.start, s.end) for s in segs] == [("Loop Song", 0.0, 240.0), ("B", 240.0, 330.0)]
    assert segs[0].anchor == 0.0 and dropped == []    # anchor of the first window's cluster, not of the jumps


def test_release_variants_join_by_normalised_title():
    # radio mix vs album entry have different intro lengths, so anchors disagree; the title says one play
    hits = [hit(t, "A-radio", anchor=0, title="Song A (Radio Mix)") for t in (0, 30)]
    hits += [hit(t, "A-album", anchor=40, title="Song A") for t in (60, 90, 120)]
    hits += [hit(t, "B", anchor=150) for t in range(150, 240, 30)]
    segs, dropped = build([Region(0, 240, "music")], hits)
    assert [(s.title, s.acr_id) for s in segs] == [("Song A", "A-album"), ("B", "B")]  # majority label
    assert dropped == []


def test_artist_prefixed_title_matches_by_suffix():
    hits = [hit(t, "A1", anchor=0, title="Some Title") for t in (0, 30, 60)]
    hits += [hit(t, "A2", anchor=100, title="Some Artist - Some Title") for t in (90, 120, 150)]
    hits += [hit(t, "B", anchor=180) for t in range(180, 270, 30)]
    segs, _ = build([Region(0, 270, "music")], hits)
    assert [s.title for s in segs] == ["Some Title", "B"]
    # but a title that merely contains the other is a different song
    hits = [hit(t, "C1", anchor=0, title="Fade") for t in (0, 30, 60)]
    hits += [hit(t, "C2", anchor=100, title="Fade To The Edge") for t in (90, 120, 150)]
    segs, _ = build([Region(0, 180, "music")], hits)
    assert [s.title for s in segs] == ["Fade", "Fade To The Edge"]


def test_short_segment_merges_by_normalised_label():
    segs = [segment.Segment(0, 150, "song", "X", "Song A", 90, "a"),
            segment.Segment(150, 180, "song", "X", "Song A (Extended Mix)", 90, "a-ext"),
            segment.Segment(180, 400, "song", "Y", "Other", 90, "o")]
    kept, dropped = segment._drop_short(segs, CFG)
    assert [(s.title, s.start, s.end) for s in kept] == [("Song A", 0.0, 180.0), ("Other", 180.0, 400.0)]
    assert dropped == []


def test_same_play_is_bridged_across_short_gate_gap():
    # a breakdown the gate closed on: 15 s "talk" inside one play; anchors agree on both sides
    regs = [Region(0, 100, "music"), Region(100, 115, "talk"), Region(115, 300, "music")]
    hits = [hit(t, "A", anchor=0) for t in range(0, 90, 30)] + [hit(t, "A", anchor=0) for t in range(115, 300, 30)]
    segs, dropped = build(regs, hits)
    assert [(s.title, s.start, s.end, s.kind) for s in segs] == [("A", 0.0, 300.0, "song")]
    assert dropped == []


def test_gap_is_kept_when_sides_differ_or_gap_is_long():
    regs = [Region(0, 100, "music"), Region(100, 115, "talk"), Region(115, 300, "music")]
    hits = [hit(t, "A", anchor=0) for t in range(0, 90, 30)] + [hit(t, "B", anchor=115) for t in range(115, 300, 30)]
    segs, _ = build(regs, hits)
    assert [s.kind for s in segs] == ["song", "talk", "song"]
    regs = [Region(0, 100, "music"), Region(100, 160, "talk"), Region(160, 400, "music")]  # 60 s > min_segment_s
    hits = [hit(t, "A", anchor=0) for t in range(0, 90, 30)] + [hit(t, "A", anchor=0) for t in range(160, 400, 30)]
    segs, _ = build(regs, hits)
    assert [s.kind for s in segs] == ["song", "talk", "song"]


def test_same_play_is_bridged_across_short_song_misfires_when_anchors_agree():
    # two 30 s misfires (60 s) between halves of one play whose anchors are identical
    hits = [hit(t, "A", anchor=0) for t in range(0, 150, 30)]
    hits += [hit(150, "M1", anchor=140), hit(180, "M2", anchor=170)]
    hits += [hit(t, "A", anchor=0) for t in range(210, 400, 30)]
    segs, dropped = build([Region(0, 400, "music")], hits)
    assert [(s.title, s.start, s.end) for s in segs] == [("A", 0.0, 400.0)]
    assert dropped == []


def test_real_track_between_same_anchor_halves_is_not_swallowed():
    hits = [hit(t, "A", anchor=0) for t in range(0, 120, 30)]
    hits += [hit(t, "B", anchor=120) for t in range(120, 210, 30)]        # 90 s: a real segment
    hits += [hit(t, "A", anchor=0) for t in range(210, 400, 30)]
    segs, _ = build([Region(0, 400, "music")], hits)
    assert [s.title for s in segs] == ["A", "B", "A"]