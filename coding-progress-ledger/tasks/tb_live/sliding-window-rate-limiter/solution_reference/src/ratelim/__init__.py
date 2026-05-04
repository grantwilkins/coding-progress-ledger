import time
from collections import deque


class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int, window_seconds: float, time_fn=time.time):
        self._max = max_requests
        self._window = window_seconds
        self._time_fn = time_fn
        self._buckets: dict[str, deque] = {}

    def _prune(self, key: str) -> deque:
        bucket = self._buckets.setdefault(key, deque())
        cutoff = self._time_fn() - self._window
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
        return bucket

    def try_acquire(self, key: str = "default") -> bool:
        bucket = self._prune(key)
        if len(bucket) >= self._max:
            return False
        bucket.append(self._time_fn())
        return True

    def current_count(self, key: str = "default") -> int:
        return len(self._prune(key))
