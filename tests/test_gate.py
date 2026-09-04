"""Hysteresis boundaries land on the frame where the state changed, not one hop early."""
import numpy as np
from tracksplit import gate
from tracksplit.config import CreatorCfg

CFG = CreatorCfg()  # hop 1 s, open 10 s, close 8 s, threshold 0.45


def _regions(music, speech, silence, total):
    return [(r.start, r.end, r.kind) for r in gate.regions(music, speech, silence, CFG, total)]


def test_boundaries_land_on_state_change():
    music = np.zeros(200); music[0:100] = 0.9; music[120:200] = 0.9
    speech = np.zeros(200); speech[100:120] = 0.8
    assert _regions(music, speech, np.zeros(200), 200.0) == [
        (0.0, 100.0, "music"), (100.0, 120.0, "talk"), (120.0, 200.0, "music")]


def test_music_blip_shorter_than_open_time_stays_nonmusic():
    music = np.zeros(60); music[20:25] = 0.9; music[30:60] = 0.9
    assert _regions(music, np.full(60, 0.7), np.zeros(60), 60.0) == [(0.0, 30.0, "talk"), (30.0, 60.0, "music")]


def test_first_region_never_starts_negative():
    regs = gate.regions(np.full(50, 0.9), np.zeros(50), np.zeros(50), CFG, 50.0)
    assert [(r.start, r.end, r.kind) for r in regs] == [(0.0, 50.0, "music")]
