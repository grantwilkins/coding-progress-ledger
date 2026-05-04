from collections import OrderedDict
import threading


class LRUCache:
    def __init__(self, maxsize: int):
        self._maxsize = maxsize
        self._data: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key, value) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self._data[key] = value
            else:
                if len(self._data) >= self._maxsize:
                    self._data.popitem(last=False)
                self._data[key] = value

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __contains__(self, key) -> bool:
        with self._lock:
            return key in self._data
