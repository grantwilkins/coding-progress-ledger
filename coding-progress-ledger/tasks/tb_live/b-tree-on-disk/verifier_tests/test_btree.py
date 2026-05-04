import random
import pytest
from btree import BTree


def test_empty_tree():
    t = BTree()
    assert len(t) == 0
    assert t.get(0) is None
    assert t.get(99) is None
    assert t.range(0, 10) == []


def test_insert_and_get():
    t = BTree()
    t.insert(10, "ten")
    assert t.get(10) == "ten"
    assert len(t) == 1


def test_duplicate_key_updates_value():
    t = BTree()
    t.insert(5, "a")
    t.insert(5, "b")
    assert t.get(5) == "b"
    assert len(t) == 1


def test_contains_true_and_false():
    t = BTree()
    t.insert(7, "seven")
    assert 7 in t
    assert 8 not in t
    assert 0 not in t


def test_keys_sorted():
    t = BTree(order=3)
    for k in [50, 10, 30, 20, 40]:
        t.insert(k, k)
    assert t.keys() == [10, 20, 30, 40, 50]


def test_range_half_open():
    t = BTree()
    for i in range(10):
        t.insert(i, i * 2)
    result = t.range(2, 5)
    assert result == [(2, 4), (3, 6), (4, 8)]


def test_range_inverted_bounds_empty():
    t = BTree()
    for i in range(10):
        t.insert(i, i)
    assert t.range(5, 2) == []
    assert t.range(5, 5) == []


def test_range_outside_data():
    t = BTree()
    for i in range(5, 10):
        t.insert(i, i)
    assert t.range(0, 4) == []
    assert t.range(11, 20) == []
    assert t.range(0, 6) == [(5, 5)]


def test_insert_200_random_keys_no_dupes():
    rng = random.Random(42)
    keys = rng.sample(range(10_000), 200)
    t = BTree()
    for k in keys:
        t.insert(k, k)
    assert len(t) == 200
    assert t.keys() == sorted(keys)


def test_stress_1000_keys_all_gettable():
    t = BTree(order=5)
    for i in range(1000):
        t.insert(i, i * 3)
    for i in range(1000):
        assert t.get(i) == i * 3
    assert len(t) == 1000


def test_various_value_types():
    t = BTree()
    t.insert(1, {"nested": True})
    t.insert(2, [1, 2, 3])
    t.insert(3, None)
    assert t.get(1) == {"nested": True}
    assert t.get(2) == [1, 2, 3]
    assert t.get(3) is None
    assert 3 in t
    assert len(t) == 3
