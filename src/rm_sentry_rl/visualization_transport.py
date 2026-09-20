"""本机RViz进程间通信，仅依赖标准库；不接收外部地图或路径。"""
from __future__ import annotations

import fcntl
import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class VisualizationChannel:
    """WAL快照与有界命令队列，每个进程独立持有连接。"""

    def __init__(self, directory: str | Path, *, producer: bool = False):
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = None
        if producer:
            self._lock = (self.directory / "producer.lock").open("a")
            try:
                fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self._lock.close()
                raise RuntimeError("该可视化目录已被另一训练/演示进程占用，请更换--session") from None
        try:
            self.connection = sqlite3.connect(self.directory / "session.sqlite3", timeout=2.0)
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=NORMAL")
            self.connection.executescript(
                "CREATE TABLE IF NOT EXISTS snapshots (name TEXT PRIMARY KEY, payload TEXT NOT NULL);"
                "CREATE TABLE IF NOT EXISTS commands (id INTEGER PRIMARY KEY, payload TEXT NOT NULL);"
            )
            if producer:
                with self.connection:
                    self.connection.execute("DELETE FROM snapshots")
                    self.connection.execute("DELETE FROM commands")
            self.connection.execute("PRAGMA busy_timeout=50")
        except Exception:
            self.close()
            raise

    def put(self, **values: Any) -> None:
        with self.connection:
            self.connection.executemany(
                "INSERT OR REPLACE INTO snapshots VALUES (?, ?)",
                [(key, json.dumps(value, allow_nan=False)) for key, value in values.items()],
            )

    def read(self, names: tuple[str, ...] | None = None) -> dict[str, Any]:
        query = "SELECT name, payload FROM snapshots"
        if names is not None:
            query += " WHERE name IN (" + ",".join("?" for _ in names) + ")"
        return {key: json.loads(payload) for key, payload in
                self.connection.execute(query, names or ())}

    def send(self, kind: str, value: Any = None) -> None:
        payload = json.dumps({"kind": kind, "value": value, "created": time.time()}, allow_nan=False)
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            if self.connection.execute("SELECT COUNT(*) FROM commands").fetchone()[0] >= 64:
                raise RuntimeError("控制队列已满，请检查环境是否仍在运行")
            self.connection.execute("INSERT INTO commands(payload) VALUES (?)", (payload,))

    def drain(self) -> list[dict[str, Any]]:
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            commands = [json.loads(row[0]) for row in
                        self.connection.execute("SELECT payload FROM commands ORDER BY id")]
            self.connection.execute("DELETE FROM commands")
        return commands

    def close(self) -> None:
        if hasattr(self, "connection"):
            self.connection.close()
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
