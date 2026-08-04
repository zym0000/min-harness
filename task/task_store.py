import asyncio
import json
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Protocol

class TaskStore(Protocol):
    async def save(self, checkpoint: Dict[str, Any]) -> None: ...
    async def load(self, task_id: str) -> Optional[Dict[str, Any]]: ...
    async def load_all(self) -> List[Dict[str, Any]]: ...
    async def delete(self, task_id: str) -> None: ...

class MemoryTaskStore:
    def __init__(self):
        self._data: Dict[str, Dict[str, Any]] = {}

    async def save(self, checkpoint: Dict[str, Any]) -> None:
        self._data[checkpoint["task_id"]] = checkpoint

    async def load(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self._data.get(task_id)

    async def load_all(self) -> List[Dict[str, Any]]:
        return list(self._data.values())

    async def delete(self, task_id: str) -> None:
        self._data.pop(task_id, None)


class SQLiteTaskStore:
    """跨进程持久化（标准库 sqlite3，异步方法经 to_thread 不阻塞事件循环）。"""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        with self._lock:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS task_checkpoints(
                       task_id TEXT PRIMARY KEY,
                       status TEXT NOT NULL,
                       blob TEXT NOT NULL,
                       updated_at REAL NOT NULL)"""
            )
            self._conn.commit()

    async def save(self, checkpoint: Dict[str, Any]) -> None:
        blob = json.dumps(checkpoint, ensure_ascii=False, default=repr)

        def _w() -> None:
            with self._lock:
                self._conn.execute(
                    "INSERT OR REPLACE INTO task_checkpoints VALUES(?,?,?,?)",
                    (checkpoint["task_id"], checkpoint.get("status", ""),
                     blob, time.time()))
                self._conn.commit()
        await asyncio.to_thread(_w)

    async def load(self, task_id: str) -> Optional[Dict[str, Any]]:
        def _r() -> Optional[Dict[str, Any]]:
            with self._lock:
                row = self._conn.execute(
                    "SELECT blob FROM task_checkpoints WHERE task_id=?",
                    (task_id,)).fetchone()
            return json.loads(row[0]) if row else None
        return await asyncio.to_thread(_r)

    async def load_all(self) -> List[Dict[str, Any]]:
        def _r() -> List[Dict[str, Any]]:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT blob FROM task_checkpoints ORDER BY updated_at").fetchall()
            return [json.loads(r[0]) for r in rows]
        return await asyncio.to_thread(_r)

    async def delete(self, task_id: str) -> None:
        def _w() -> None:
            with self._lock:
                self._conn.execute(
                    "DELETE FROM task_checkpoints WHERE task_id=?", (task_id,))
                self._conn.commit()
        await asyncio.to_thread(_w)

    def close(self) -> None:
        with self._lock:
            self._conn.close()