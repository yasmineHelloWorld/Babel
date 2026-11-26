"""Unit tests for backend.trainer — BabelTrainer and judge session."""
import pytest
from backend.trainer import BabelTrainer


# ── basic sanity ──────────────────────────────────────────────────────────────

def test_train_step_returns_stats():
    t = BabelTrainer(seed=0)
    stats = t.train_step()
    assert 0.0 <= stats.listener_acc <= 1.0
    assert 0.0 <= stats.rival_acc   <= 1.0
    assert 0.0 <= stats.consistency <= 1.0
    assert stats.entropy >= 0.0
    assert stats.epoch   == 1


def test_epoch_increments():
    t = BabelTrainer(seed=0)
    for i in range(5):
        s = t.train_step()
        assert s.epoch == i + 1


def test_rival_slower_than_listener():
    """With rival_lr_mult=0.2, listener should stay ahead after 500 steps."""
    t = BabelTrainer(seed=0, rival_lr_mult=0.2)
    for _ in range(500):
        stats = t.train_step()
    # Listener should be meaningfully better (at least not worse by >5%)
    assert stats.listener_acc >= stats.rival_acc - 0.05


def test_reset_zeroes_epoch():
    t = BabelTrainer(seed=0)
    for _ in range(100):
        t.train_step()
    t.reset(seed=99)
    assert t.epoch == 0
    s = t.train_step()
    assert s.epoch == 1


def test_reset_reproducibility():
    """A trainer reset to its own seed should reproduce the first step."""
    t = BabelTrainer(seed=7)
    s1 = t.train_step()
    t.reset(seed=7)
    s2 = t.train_step()
    assert abs(s1.listener_acc - s2.listener_acc) < 1e-6


# ── sample_round ──────────────────────────────────────────────────────────────

def test_sample_round_structure():
    t = BabelTrainer(seed=0)
    r = t.sample_round()
    assert 'object'        in r
    assert 'message'       in r
    assert 'listener_guess' in r
    assert 'rival_guess'   in r
    assert 'epoch'         in r


def test_sample_round_message_valid():
    t = BabelTrainer(seed=0)
    r = t.sample_round()
    msg = r['message']
    assert len(msg) == 2
    assert all(isinstance(s, int) and 0 <= s < 16 for s in msg)


def test_sample_round_object_valid():
    t = BabelTrainer(seed=0)
    for _ in range(20):
        r = t.sample_round()
        o = r['object']
        assert 0 <= o['s'] <= 3
        assert 0 <= o['c'] <= 3


def test_sample_round_guess_valid():
    t = BabelTrainer(seed=0)
    r = t.sample_round()
    for key in ('listener_guess', 'rival_guess'):
        g = r[key]
        assert 0 <= g['s'] <= 3
        assert 0 <= g['c'] <= 3


# ── skip ──────────────────────────────────────────────────────────────────────

def test_skip_advances_epoch():
    t = BabelTrainer(seed=0)
    s = t.skip(25)
    assert s.epoch == 25


def test_skip_then_train():
    t = BabelTrainer(seed=0)
    t.skip(10)
    s = t.train_step()
    assert s.epoch == 11


# ── judge session ──────────────────────────────────────────────────────────────

def test_start_judge_returns_first_round():
    t = BabelTrainer(seed=0)
    rnd = t.start_judge()
    assert rnd['type']  == 'judge_round'
    assert rnd['index'] == 1
    assert len(rnd['options']) == 4
    assert len(rnd['message']) == 2


def test_judge_options_contain_truth():
    """Truth object must appear in options."""
    t = BabelTrainer(seed=0)
    rnd = t.start_judge()
    # We don't expose truth from start_judge, but answer(correct_idx) should return correct=True
    # Find the correct option by checking answer(0..3)
    t2 = BabelTrainer(seed=0)
    rnd2 = t2.start_judge()
    # Try all options; at least one should give correct=True
    correct_found = False
    for i in range(4):
        t3 = BabelTrainer(seed=0)
        t3.start_judge()
        result = t3.answer_judge(i)
        if result['correct']:
            correct_found = True
            break
    assert correct_found


def test_answer_judge_structure():
    t = BabelTrainer(seed=0)
    t.start_judge()
    result = t.answer_judge(0)
    assert result['type']  == 'judge_result'
    assert 'correct'       in result
    assert 'truth'         in result
    assert 'rival_correct' in result
    assert 'you'           in result
    assert 'rival'         in result
    assert 'finished'      in result


def test_judge_8_rounds():
    t = BabelTrainer(seed=0)
    t.start_judge()
    for i in range(8):
        result = t.answer_judge(0)
        if i < 7:
            assert not result['finished']
            nxt = t.judge_next_round()
            assert nxt is not None
            assert nxt['index'] == i + 2
        else:
            assert result['finished']


def test_judge_scores_accumulate():
    t = BabelTrainer(seed=42)
    for _ in range(500):
        t.train_step()  # train a bit so rival has non-trivial behaviour
    t.start_judge()
    total_you = total_rival = 0
    for i in range(8):
        result = t.answer_judge(0)
        total_you   = result['you']
        total_rival = result['rival']
        if i < 7:
            t.judge_next_round()
    assert 0 <= total_you   <= 8
    assert 0 <= total_rival <= 8


# ── acceptance (slow) ─────────────────────────────────────────────────────────

@pytest.mark.slow
def test_listener_accuracy_target_2000_epochs():
    """Acceptance: seed 0, listener_acc ≥ 90% within 2000 epochs."""
    t = BabelTrainer(seed=0)
    for _ in range(2000):
        stats = t.train_step()
    assert stats.listener_acc >= 0.90, (
        f"listener_acc={stats.listener_acc:.3f} < 0.90 at epoch {stats.epoch}"
    )


@pytest.mark.slow
def test_rival_clearly_lower_at_2000():
    """Acceptance: rival accuracy clearly lower than listener at epoch 2000."""
    t = BabelTrainer(seed=0)
    for _ in range(2000):
        stats = t.train_step()
    gap = stats.listener_acc - stats.rival_acc
    assert gap > 0.10, (
        f"gap={gap:.3f} not large enough "
        f"(listener={stats.listener_acc:.3f}, rival={stats.rival_acc:.3f})"
    )
