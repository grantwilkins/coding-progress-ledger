import threading
import random
from lru_cache_ts import LRUCache


def test_basic_insert_and_get():
    c = LRUCache(3)
    c.put("a", 1); c.put("b", 2)
    assert c.get("a") == 1
    assert c.get("b") == 2
    assert c.get("z") is None


def test_evicts_lru_on_overflow():
    c = LRUCache(3)
    for i in range(1, 5):
        c.put(i, str(i))   # key 4 evicts key 1
    assert c.get(1) is None
    assert c.get(2) == "2"
    assert c.get(4) == "4"


def test_get_updates_recency():
    c = LRUCache(2)
    c.put("x", 10); c.put("y", 20)
    c.get("x")          # x MRU, y LRU
    c.put("z", 30)      # evicts y
    assert c.get("y") is None
    assert c.get("x") == 10
    assert c.get("z") == 30


def test_put_update_changes_value_and_recency():
    c = LRUCache(2)
    c.put("a", 1); c.put("b", 2)
    c.put("a", 99)      # a becomes MRU, no eviction
    assert len(c) == 2
    c.put("c", 3)       # evicts b
    assert c.get("b") is None
    assert c.get("a") == 99
    assert c.get("c") == 3


def test_len_correct():
    c = LRUCache(4)
    assert len(c) == 0
    c.put("a", 1); c.put("b", 2); c.put("c", 3)
    assert len(c) == 3
    c.put("d", 4); c.put("e", 5)
    assert len(c) == 4


def test_contains_does_not_update_recency():
    c = LRUCache(2)
    c.put("A", 1); c.put("B", 2)   # A LRU, B MRU
    assert "A" in c and "B" in c   # no recency change
    c.put("C", 3)                   # evicts A (still LRU)
    assert "A" not in c
    assert "B" in c and "C" in c


def test_contains_recency_spec_example():
    # A, B inserted (maxsize=2); C inserted evicts A → B LRU, C MRU.
    # __contains__(B) must not promote B; D evicts B.
    c = LRUCache(2)
    c.put("A", 1); c.put("B", 2); c.put("C", 3)
    assert "B" in c     # no recency change
    c.put("D", 4)       # evicts B
    assert "B" not in c
    assert "C" in c and "D" in c


def test_maxsize_one():
    c = LRUCache(1)
    c.put("k", "v1")
    assert c.get("k") == "v1"
    c.put("j", "v2")
    assert c.get("k") is None
    assert c.get("j") == "v2"
    assert len(c) == 1


def test_concurrent_stress():
    maxsize, n_threads, n_ops, key_range = 10, 4, 500, 50
    cache = LRUCache(maxsize)
    errors = []
    barrier = threading.Barrier(n_threads)

    def worker(seed):
        rng = random.Random(seed)
        try:
            barrier.wait()
            for _ in range(n_ops):
                k = rng.randint(0, key_range - 1)
                if rng.random() < 0.4:
                    v = cache.get(k)
                    if v is not None and v != k * 10:
                        errors.append(f"corrupt value key={k} got={v}")
                else:
                    cache.put(k, k * 10)
        except Exception as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert not errors, errors
    assert len(cache) <= maxsize
