"""
知识图谱存储层
使用 SQLite 实现实体-关系-观察的知识图谱，支持向量语义搜索
"""

import sqlite3
import struct
import logging
from typing import Optional, List, Dict
from config import DB_PATH

logger = logging.getLogger("local-memory")


def _blob_to_vector(blob: bytes) -> List[float]:
    """将 SQLite BLOB 还原为 float 向量"""
    count = len(blob) // 4  # float32 = 4 bytes
    return list(struct.unpack(f"<{count}f", blob))


def _vector_to_blob(vec: List[float]) -> bytes:
    """将 float 向量转为 SQLite BLOB"""
    return struct.pack(f"<{len(vec)}f", *vec)


class KnowledgeGraph:
    def __init__(self, db_path: str = DB_PATH):
        """初始化数据库连接，创建表"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_tables()
        # 初始化 FAISS 向量索引
        self._init_vector_index()

    def _init_tables(self):
        """创建实体、关系、观察三张表"""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                name TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL DEFAULT 'unknown',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_entity TEXT NOT NULL REFERENCES entities(name) ON DELETE CASCADE,
                to_entity TEXT NOT NULL REFERENCES entities(name) ON DELETE CASCADE,
                relation_type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(from_entity, to_entity, relation_type)
            );

            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_name TEXT NOT NULL REFERENCES entities(name) ON DELETE CASCADE,
                content TEXT NOT NULL,
                embedding BLOB,
                tags TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_observations_entity ON observations(entity_name);
            CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(from_entity);
            CREATE INDEX IF NOT EXISTS idx_relations_to ON relations(to_entity);
        """)
        self.conn.commit()

    def _init_vector_index(self):
        """初始化 FAISS 向量索引"""
        try:
            from vector_index import VectorIndex
            self.vector_index = VectorIndex()
            if self.vector_index.total == 0:
                count = self.conn.execute(
                    "SELECT COUNT(*) FROM observations WHERE embedding IS NOT NULL AND length(embedding) > 0"
                ).fetchone()[0]
                if count > 0:
                    logger.info(f"数据库有 {count} 条向量但索引为空，正在重建索引...")
                    self.vector_index.rebuild_from_db(self.conn)
        except ImportError:
            logger.warning("FAISS 未安装，退回 numpy 搜索")
            self.vector_index = None

    def add_observation(
        self,
        entity_name: str,
        content: str,
        entity_type: str = "unknown",
        tags: str = "",
        relations: List[Dict] = None,
    ) -> dict:
        """
        添加观察（核心记忆）

        - 如果实体不存在，自动创建
        - 自动生成嵌入向量（如果 Ollama 可用）
        - 自动搜索冲突记忆，有冲突则更新旧记忆的 content 而不是新建
        - relations 格式: [{"to": "React", "type": "likes"}, ...]

        返回: {"entity": "张三", "observation_id": 1, "status": "created"/"updated", "conflict": {...} 或 None}
        """
        if relations is None:
            relations = []

        # 确保实体存在
        existing = self.conn.execute(
            "SELECT name FROM entities WHERE name = ?", (entity_name,)
        ).fetchone()
        if not existing:
            self.conn.execute(
                "INSERT OR IGNORE INTO entities (name, entity_type) VALUES (?, ?)",
                (entity_name, entity_type),
            )
            # 如果实体已存在但类型不同，更新类型
            self.conn.execute(
                "UPDATE entities SET entity_type = ? WHERE name = ?",
                (entity_type, entity_name),
            )
            self.conn.commit()

        # 生成嵌入向量
        embedding = None
        try:
            from embedding import get_embedding
            embedding = get_embedding(content)
        except Exception:
            pass

        # 冲突检测：直接在数据库中搜索相似记忆（复用已生成的 embedding）
        conflict = None
        if embedding:
            try:
                # 直接用向量搜索，不再调用 search() 避免双重生成 embedding
                rows = self.conn.execute(
                    "SELECT id, entity_name, content, embedding, tags, created_at "
                    "FROM observations WHERE embedding IS NOT NULL AND length(embedding) > 0"
                ).fetchall()

                if rows:
                    from similarity import search_similar
                    all_embeddings = [_blob_to_vector(row[3]) for row in rows]
                    results = search_similar(embedding, all_embeddings, top_k=3)

                    for idx, score in results:
                        if score > 0.85:
                            row = rows[idx]
                            if row[1] == entity_name:  # 同一实体
                                conflict = {
                                    "observation_id": row[0],
                                    "entity": row[1],
                                    "content": row[2],
                                    "tags": row[4],
                                    "score": round(score, 4),
                                }
                                break
            except Exception:
                pass

        # 无向量时：用关键词匹配做冲突检测（仅在完全相同内容时触发）
        if not conflict and not embedding:
            try:
                rows = self.conn.execute(
                    "SELECT id, content FROM observations WHERE entity_name = ? ORDER BY updated_at DESC LIMIT 5",
                    (entity_name,),
                ).fetchall()
                for row in rows:
                    old_content = row[1]
                    # 去掉标点和空白后比较
                    content_clean = ''.join(c for c in content if c.strip() and c not in "，。、！？,.!? \t\n")
                    old_clean = ''.join(c for c in old_content if c.strip() and c not in "，。、！？,.!? \t\n")
                    if content_clean == old_clean:
                        conflict = {"observation_id": row[0], "content": old_content}
                        break
            except Exception:
                pass

        if conflict:
            # 更新旧记忆
            obs_id = conflict.get("observation_id")
            blob = _vector_to_blob(embedding) if embedding else None
            self.conn.execute(
                "UPDATE observations SET content = ?, embedding = ?, tags = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (content, blob, tags, obs_id),
            )
            self.conn.commit()
            # 更新 FAISS 索引中的向量
            if embedding and self.vector_index:
                self.vector_index.remove(obs_id)
                self.vector_index.add(obs_id, embedding)
                self.vector_index.save()
            return {
                "entity": entity_name,
                "observation_id": obs_id,
                "status": "updated",
                "conflict": conflict,
            }

        # 无冲突，新建观察
        blob = _vector_to_blob(embedding) if embedding else None
        cursor = self.conn.execute(
            "INSERT INTO observations (entity_name, content, embedding, tags) VALUES (?, ?, ?, ?)",
            (entity_name, content, blob, tags),
        )
        self.conn.commit()
        obs_id = cursor.lastrowid

        # 同步 FAISS 索引
        if embedding and self.vector_index:
            self.vector_index.add(obs_id, embedding)
            self.vector_index.save()

        # 添加关系
        for rel in relations:
            to_entity = rel.get("to", "")
            rel_type = rel.get("type", "")
            if not to_entity or not rel_type:
                continue
            # 确保目标实体存在
            self.conn.execute(
                "INSERT OR IGNORE INTO entities (name, entity_type) VALUES (?, 'unknown')",
                (to_entity,),
            )
            self.conn.execute(
                "INSERT OR IGNORE INTO relations (from_entity, to_entity, relation_type) VALUES (?, ?, ?)",
                (entity_name, to_entity, rel_type),
            )
        if relations:
            self.conn.commit()

        return {
            "entity": entity_name,
            "observation_id": obs_id,
            "status": "created",
            "conflict": None,
        }

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        语义搜索（向量搜索 + 关键词降级）

        返回: [{"entity": "张三", "content": "是前端工程师", "type": "person",
                "tags": "工作", "score": 0.95, "observation_id": 1, "relations": [...]}]
        """
        # 尝试向量搜索（优先 FAISS，退回 numpy）
        try:
            from embedding import get_embedding
            query_embedding = get_embedding(query)

            # 优先使用 FAISS 索引
            if self.vector_index and self.vector_index.total > 0:
                faiss_results = self.vector_index.search(query_embedding, top_k)
                output = []
                for obs_id, score in faiss_results:
                    row = self.conn.execute(
                        "SELECT entity_name, content, tags, created_at FROM observations WHERE id = ?",
                        (obs_id,),
                    ).fetchone()
                    if not row:
                        continue
                    entity_name, content, tags, created_at = row
                    entity_row = self.conn.execute(
                        "SELECT entity_type FROM entities WHERE name = ?", (entity_name,)
                    ).fetchone()
                    entity_type = entity_row[0] if entity_row else "unknown"
                    rels = self._get_relations(entity_name)
                    output.append({
                        "observation_id": obs_id,
                        "entity": entity_name,
                        "content": content,
                        "type": entity_type,
                        "tags": tags,
                        "score": round(score, 4),
                        "created_at": created_at,
                        "relations": rels,
                    })
                if output:
                    return output

            # 退回 numpy 暴力搜索
            rows = self.conn.execute(
                "SELECT id, entity_name, content, embedding, tags, created_at "
                "FROM observations WHERE embedding IS NOT NULL AND length(embedding) > 0"
            ).fetchall()

            if rows:
                from similarity import search_similar
                all_embeddings = [_blob_to_vector(row[3]) for row in rows]
                results = search_similar(query_embedding, all_embeddings, top_k)

                output = []
                for idx, score in results:
                    row = rows[idx]
                    obs_id, entity_name, content, _, tags, created_at = row
                    entity_row = self.conn.execute(
                        "SELECT entity_type FROM entities WHERE name = ?", (entity_name,)
                    ).fetchone()
                    entity_type = entity_row[0] if entity_row else "unknown"
                    rels = self._get_relations(entity_name)
                    output.append({
                        "observation_id": obs_id,
                        "entity": entity_name,
                        "content": content,
                        "type": entity_type,
                        "tags": tags,
                        "score": round(score, 4),
                        "created_at": created_at,
                        "relations": rels,
                    })
                return output
        except Exception:
            pass

        # 降级：关键词搜索
        return self._keyword_search(query, top_k)

    def _keyword_search(self, keyword: str, top_k: int = 5) -> List[Dict]:
        """关键词搜索降级方案"""
        rows = self.conn.execute(
            "SELECT o.id, o.entity_name, o.content, o.tags, o.created_at, e.entity_type "
            "FROM observations o JOIN entities e ON o.entity_name = e.name "
            "WHERE o.content LIKE ? ORDER BY o.created_at DESC LIMIT ?",
            (f"%{keyword}%", top_k),
        ).fetchall()

        output = []
        for row in rows:
            obs_id, entity_name, content, tags, created_at, entity_type = row
            rels = self._get_relations(entity_name)
            output.append({
                "observation_id": obs_id,
                "entity": entity_name,
                "content": content,
                "type": entity_type,
                "tags": tags,
                "score": None,
                "created_at": created_at,
                "relations": rels,
            })
        return output

    def get_entity(self, name: str) -> Optional[dict]:
        """获取实体及其所有观察和关系"""
        entity_row = self.conn.execute(
            "SELECT name, entity_type, created_at FROM entities WHERE name = ?", (name,)
        ).fetchone()
        if not entity_row:
            return None

        obs_rows = self.conn.execute(
            "SELECT id, content, tags, created_at, updated_at FROM observations WHERE entity_name = ? ORDER BY created_at DESC",
            (name,),
        ).fetchall()

        observations = [
            {
                "id": row[0],
                "content": row[1],
                "tags": row[2],
                "created_at": row[3],
                "updated_at": row[4],
            }
            for row in obs_rows
        ]

        relations = self._get_relations(name)

        return {
            "name": entity_row[0],
            "type": entity_row[1],
            "created_at": entity_row[2],
            "observations": observations,
            "relations": relations,
        }

    def list_all(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        """列出所有记忆"""
        rows = self.conn.execute(
            "SELECT o.id, o.entity_name, o.content, o.tags, o.created_at, e.entity_type "
            "FROM observations o JOIN entities e ON o.entity_name = e.name "
            "ORDER BY o.created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()

        output = []
        for row in rows:
            obs_id, entity_name, content, tags, created_at, entity_type = row
            rels = self._get_relations(entity_name)
            output.append({
                "observation_id": obs_id,
                "entity": entity_name,
                "content": content,
                "type": entity_type,
                "tags": tags,
                "created_at": created_at,
                "relations": rels,
            })
        return output

    def delete_entity(self, name: str) -> bool:
        """删除实体及其所有观察和关系"""
        # 先获取该实体的所有观察 ID（用于清理 FAISS 索引）
        obs_ids = [row[0] for row in self.conn.execute(
            "SELECT id FROM observations WHERE entity_name = ?", (name,)
        ).fetchall()]
        cursor = self.conn.execute("DELETE FROM entities WHERE name = ?", (name,))
        self.conn.commit()
        if self.vector_index:
            for obs_id in obs_ids:
                self.vector_index.remove(obs_id)
            if obs_ids:
                self.vector_index.save()
        return cursor.rowcount > 0

    def delete_observation(self, observation_id: int) -> bool:
        """删除单条观察"""
        cursor = self.conn.execute(
            "DELETE FROM observations WHERE id = ?", (observation_id,)
        )
        self.conn.commit()
        if cursor.rowcount > 0 and self.vector_index:
            self.vector_index.remove(observation_id)
            self.vector_index.save()
        return cursor.rowcount > 0

    def count_entities(self) -> int:
        """实体总数"""
        row = self.conn.execute("SELECT COUNT(*) FROM entities").fetchone()
        return row[0]

    def count_observations(self) -> int:
        """观察总数"""
        row = self.conn.execute("SELECT COUNT(*) FROM observations").fetchone()
        return row[0]

    def _get_relations(self, entity_name: str) -> List[Dict]:
        """获取实体的所有关系"""
        rows = self.conn.execute(
            "SELECT from_entity, to_entity, relation_type, created_at FROM relations WHERE from_entity = ? OR to_entity = ?",
            (entity_name, entity_name),
        ).fetchall()
        return [
            {
                "from": row[0],
                "to": row[1],
                "type": row[2],
            }
            for row in rows
        ]

    def close(self):
        """关闭数据库连接"""
        self.conn.close()
