"""
记忆存储层
使用 SQLite 存储记忆条目和嵌入向量
"""

import sqlite3
import struct
import numpy as np
from datetime import datetime
from config import DB_PATH, EMBEDDING_DIM


def _blob_to_vector(blob: bytes) -> list[float]:
    """将 SQLite BLOB 还原为 float 向量"""
    count = len(blob) // 4  # float32 = 4 bytes
    return list(struct.unpack(f"<{count}f", blob))


def _vector_to_blob(vec: list[float]) -> bytes:
    """将 float 向量转为 SQLite BLOB"""
    return struct.pack(f"<{len(vec)}f", *vec)


class MemoryStore:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_tables()

    def _init_tables(self):
        """初始化数据库表"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                embedding BLOB,
                tags TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def add(self, content: str, embedding: list[float], tags: str = "") -> int:
        """
        添加一条记忆

        Args:
            content: 记忆内容
            embedding: 嵌入向量
            tags: 标签（逗号分隔）

        Returns:
            新记忆的 ID
        """
        now = datetime.now().isoformat()
        blob = _vector_to_blob(embedding)
        cursor = self.conn.execute(
            "INSERT INTO memories (content, embedding, tags, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (content, blob, tags, now, now),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get(self, memory_id: int) -> dict | None:
        """根据 ID 获取一条记忆"""
        row = self.conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[dict]:
        """
        语义搜索最相关的记忆

        Args:
            query_embedding: 查询向量
            top_k: 返回条数

        Returns:
            匹配的记忆列表，包含相似度分数
        """
        rows = self.conn.execute(
            "SELECT id, content, embedding, tags, created_at, updated_at FROM memories WHERE embedding IS NOT NULL AND length(embedding) > 0"
        ).fetchall()

        if not rows:
            return []

        from similarity import search_similar

        all_embeddings = [_blob_to_vector(row[2]) for row in rows]
        results = search_similar(query_embedding, all_embeddings, top_k)

        memories = []
        for idx, score in results:
            row = rows[idx]
            memory = {
                "id": row[0],
                "content": row[1],
                "tags": row[3],
                "created_at": row[4],
                "updated_at": row[5],
                "score": round(score, 4),
            }
            memories.append(memory)

        return memories

    def keyword_search(self, keyword: str, top_k: int = 5) -> list[dict]:
        """
        关键词搜索（降级方案，Ollama 不可用时使用）

        Args:
            keyword: 关键词
            top_k: 返回条数
        """
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE content LIKE ? ORDER BY created_at DESC LIMIT ?",
            (f"%{keyword}%", top_k),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def list_all(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """列出所有记忆"""
        rows = self.conn.execute(
            "SELECT * FROM memories ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def delete(self, memory_id: int) -> bool:
        """删除一条记忆，返回是否成功"""
        cursor = self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def update(self, memory_id: int, content: str = None, tags: str = None) -> bool:
        """更新记忆内容或标签"""
        updates = []
        params = []

        if content is not None:
            updates.append("content = ?")
            params.append(content)
            updates.append("updated_at = ?")
            params.append(datetime.now().isoformat())

        if tags is not None:
            updates.append("tags = ?")
            params.append(tags)
            updates.append("updated_at = ?")
            params.append(datetime.now().isoformat())

        if not updates:
            return False

        params.append(memory_id)
        cursor = self.conn.execute(
            f"UPDATE memories SET {', '.join(updates)} WHERE id = ?", params
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """获取记忆总数"""
        row = self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        return row[0]

    def _row_to_dict(self, row: tuple) -> dict:
        """将数据库行转为字典"""
        return {
            "id": row[0],
            "content": row[1],
            "tags": row[3],
            "created_at": row[4],
            "updated_at": row[5],
        }

    def close(self):
        """关闭数据库连接"""
        self.conn.close()
