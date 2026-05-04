import pytest
from ratelim import SlidingWindowRateLimiter


def make_clock(start=0.0):
    clock = [start]
    def time_fn():
        return clock[0]
    return clock, time_fn


def test_below_limit_returns_true():
    clock, time_fn = make_clock()
    rl = SlidingWindowRateLimiter(3, 10.0, time_fn)
    assert rl.try_acquire() is True
    assert rl.try_acquire() is True
    assert rl.try_acquire() is True


def test_at_limit_returns_false():
    clock, time_fn = make_clock()
    rl = SlidingWindowRateLimiter(2, 10.0, time_fn)
    assert rl.try_acquire() is True
    assert rl.try_acquire() is True
    assert rl.try_acquire() is False


def test_window_slides_allows_new_request():
    clock, time_fn = make_clock()
    rl = SlidingWindowRateLimiter(2, 10.0, time_fn)
    rl.try_acquire()          # t=0
    clock[0] = 5.0
    rl.try_acquire()          # t=5
    assert rl.try_acquire() is False  # still 2 in window
    clock[0] = 10.5           # first request (t=0) now outside window
    assert rl.try_acquire() is True


def test_per_key_isolation():
    clock, time_fn = make_clock()
    rl = SlidingWindowRateLimiter(1, 10.0, time_fn)
    assert rl.try_acquire("a") is True
    assert rl.try_acquire("a") is False
    assert rl.try_acquire("b") is True   # key b is independent


def test_current_count_within_window():
    clock, time_fn = make_clock()
    rl = SlidingWindowRateLimiter(5, 10.0, time_fn)
    rl.try_acquire()    # t=0
    rl.try_acquire()    # t=0
    assert rl.current_count() == 2
    clock[0] = 10.5     # both timestamps (t=0) now outside window
    assert rl.current_count() == 0


def test_current_count_partial_expiry():
    clock, time_fn = make_clock()
    rl = SlidingWindowRateLimiter(5, 10.0, time_fn)
    rl.try_acquire()          # t=0
    clock[0] = 5.0
    rl.try_acquire()          # t=5
    clock[0] = 10.5           # t=0 expired, t=5 still in window
    assert rl.current_count() == 1


def test_max_requests_one_rapid_calls():
    clock, time_fn = make_clock()
    rl = SlidingWindowRateLimiter(1, 1.0, time_fn)
    assert rl.try_acquire() is True
    assert rl.try_acquire() is False
    assert rl.try_acquire() is False
    clock[0] = 1.5
    assert rl.try_acquire() is True


def test_current_count_default_key_separate_from_named():
    clock, time_fn = make_clock()
    rl = SlidingWindowRateLimiter(5, 10.0, time_fn)
    rl.try_acquire("default")
    rl.try_acquire("other")
    rl.try_acquire("other")
    assert rl.current_count("default") == 1
    assert rl.current_count("other") == 2
