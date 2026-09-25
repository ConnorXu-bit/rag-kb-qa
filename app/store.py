"""向量存储 + 余弦检索。

故意用 SQLite + numpy 而不是向量数据库：
数据量在万级以内时，一次矩阵乘法比引入一个外部服务划算得多。
先讲清楚"检索本质是算相似度"，需要时再换 FAISS / pgvector 才是合理顺序。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .chunking import Chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    source  TEXT    NOT NULL,
    heading TEXT    NOT NULL,
    ordinal INTEGER NOT NULL,
    text    TEXT    NOT NULL,
    vector  BLOB    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks (source);
"""


@dataclass(frozen=True)
class Hit:
    source: str
    heading: str
    ordinal: int
    text: str
    score: float


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class VectorStore:
    def __init__(self, path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.executescript(SCHEMA)

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def iter_chunks(self) -> list[Chunk]:
        """按写入顺序取出全部片段，供关键词检索建索引使用。"""
        rows = self._conn.execute(
            "SELECT source, heading, ordinal, text FROM chunks ORDER BY id"
        ).fetchall()
        return [
            Chunk(text=row[3], source=row[0], heading=row[1], ordinal=row[2]) for row in rows
        ]

    def delete_source(self, source: str) -> int:
        cursor = self._conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
        self._conn.commit()
        return cursor.rowcount

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks 与 vectors 数量必须一致")
        rows = [
            (
                chunk.source,
                chunk.heading,
                chunk.ordinal,
                chunk.text,
                np.asarray(vector, dtype=np.float32).tobytes(),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self._conn.executemany(
            "INSERT INTO chunks (source, heading, ordinal, text, vector) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def search(self, query_vector: Sequence[float], top_k: int = 4) -> list[Hit]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        rows = self._conn.execute(
            "SELECT source, heading, ordinal, text, vector FROM chunks"
        ).fetchall()
        if not rows:
            return []

        matrix = np.vstack([np.frombuffer(row[4], dtype=np.float32) for row in rows])
        query = np.asarray(query_vector, dtype=np.float32)
        if query.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"查询向量维度 {query.shape[0]} 与库内 {matrix.shape[1]} 不一致"
            )

        scores = _normalize(matrix) @ _normalize(query.reshape(1, -1))[0]
        order = np.argsort(-scores)[:top_k]
        return [
            Hit(
                source=rows[index][0],
                heading=rows[index][1],
                ordinal=rows[index][2],
                text=rows[index][3],
                score=float(scores[index]),
            )
            for index in order
        ]

    def close(self) -> None:
        self._conn.close()
