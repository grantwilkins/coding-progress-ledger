import os
import sqlite3


def recover(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if os.path.getsize(path) == 0:
        return []
    rows: list[dict] = []
    con = sqlite3.connect(path)
    try:
        cur = con.execute("SELECT id, name, value FROM records ORDER BY id")
        while True:
            try:
                row = cur.fetchone()
            except sqlite3.DatabaseError:
                break
            if row is None:
                break
            rows.append({"id": row[0], "name": row[1], "value": row[2]})
    except sqlite3.DatabaseError:
        pass
    finally:
        con.close()
    return rows
