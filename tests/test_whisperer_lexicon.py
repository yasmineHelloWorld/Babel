"""Unit tests for backend.lexicon — sliding-window co-occurrence tracker."""
import pytest
from backend.lexicon import LexiconTracker


def test_stability_increases_with_repetition():
    lex = LexiconTracker(vocab=16, msg_len=2, window=200)
    for _ in range(80):
        lex.update(0, 1, [5, 3])   # meaning 0 (shape 0), colour 1
    stats = {e['meaning']: e for e in lex.stats()}
    # meaning 0 (shape 0): dominant is symbol 5 at position 0
    assert stats[0]['symbol'] == 5
    assert stats[0]['position'] == 0
    # With 2-symbol messages, each meaning records two (pos,sym) pairs per
    # update, so max stability with one consistent message is 0.5.
    assert stats[0]['stability'] == pytest.approx(0.5)
    # meaning 4 (color 0): position 1 → symbol 3
    # (obj_id 1 = s=0,c=1 — but we pass meaning 0 and 4 manually below)


def test_dominant_symbol_per_meaning():
    lex = LexiconTracker(vocab=16, msg_len=2, window=200)
    # meaning 3 always gets sym [7, 2]
    for _ in range(60):
        lex.update(3, 0, [7, 2])
    stats = {e['meaning']: e for e in lex.stats()}
    assert stats[3]['symbol']   == 7
    assert stats[3]['position'] == 0
    # Max stability with 2 positions is 0.5 (each update records both positions)
    assert stats[3]['stability'] == pytest.approx(0.5)


def test_window_expires_old_counts():
    lex = LexiconTracker(vocab=16, msg_len=2, window=10)
    # First 10 entries: meaning 0 → sym [1, 0]
    for _ in range(10):
        lex.update(0, 0, [1, 0])
    # Next 15 entries override with [7, 0]
    for _ in range(15):
        lex.update(0, 0, [7, 0])
    stats = {e['meaning']: e for e in lex.stats()}
    assert stats[0]['symbol'] == 7


def test_consistency_zero_on_fresh():
    lex = LexiconTracker()
    assert lex.consistency() == pytest.approx(0.0)


def test_consistency_range_after_training():
    lex = LexiconTracker()
    for _ in range(30):
        lex.update(0, 0, [5, 3])
    c = lex.consistency()
    assert 0.0 <= c <= 1.0


def test_consistency_rises_with_repetition():
    lex = LexiconTracker(window=200)
    before = lex.consistency()
    for _ in range(100):
        lex.update(0, 0, [5, 3])
        lex.update(1, 1, [5, 3])
        lex.update(2, 2, [6, 4])
        lex.update(3, 3, [6, 4])
    after = lex.consistency()
    assert after > before


def test_stats_covers_all_meanings():
    lex = LexiconTracker()
    stats = lex.stats()
    meanings = {e['meaning'] for e in stats}
    assert meanings == set(range(8))


def test_stats_stability_zero_on_no_data():
    lex = LexiconTracker()
    for e in lex.stats():
        assert e['stability'] == pytest.approx(0.0)


def test_reset_clears_state():
    lex = LexiconTracker(window=200)
    for _ in range(50):
        lex.update(0, 0, [5, 3])
    lex.reset()
    assert lex.consistency() == pytest.approx(0.0)
    for e in lex.stats():
        assert e['stability'] == pytest.approx(0.0)
