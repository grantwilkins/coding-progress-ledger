import bisect


class BTree:
    def __init__(self, order: int = 4):
        self._order = order
        self._keys: list[int] = []
        self._vals: list = []

    def insert(self, key: int, value) -> None:
        i = bisect.bisect_left(self._keys, key)
        if i < len(self._keys) and self._keys[i] == key:
            self._vals[i] = value
        else:
            self._keys.insert(i, key)
            self._vals.insert(i, value)

    def get(self, key: int):
        i = bisect.bisect_left(self._keys, key)
        if i < len(self._keys) and self._keys[i] == key:
            return self._vals[i]
        return None

    def __contains__(self, key: int) -> bool:
        i = bisect.bisect_left(self._keys, key)
        return i < len(self._keys) and self._keys[i] == key

    def __len__(self) -> int:
        return len(self._keys)

    def range(self, lo: int, hi: int) -> list:
        if lo >= hi:
            return []
        l = bisect.bisect_left(self._keys, lo)
        r = bisect.bisect_left(self._keys, hi)
        return list(zip(self._keys[l:r], self._vals[l:r]))

    def keys(self) -> list[int]:
        return list(self._keys)
