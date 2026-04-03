"""FAISS 向量索引
使用 Facebook FAISS 加速向量相似度搜索
支持增量添加、持久化存储
"""

import os
import logging
import numpy as np
from config import EMBEDDING_DIM, DATA_DIR

logger = logging.getLogger("local-memory")


class VectorIndex:
    """基于 FAISS 的向量索引，支持增量添加和持久化"""

    def __init__(self, dim: int = EMBEDDING_DIM, index_path: str = None):
        self.dim = dim
        if index_path is None:
            index_path = os.path.join(DATA_DIR, "faiss.index")
        self.index_path = index_path

        if os.path.exists(index_path):
            import faiss
            self.index = faiss.read_index(index_path)
            logger.info(f"FAISS 索引已加载: {self.index.ntotal} 条向量")
        else:
            import faiss
            # 使用 Inner Product（归一化后等价于余弦相似度）
            self.index = faiss.IndexFlatIP(dim)
            logger.info("FAISS 索引已创建（空）")

        # ID 映射：FAISS 内部索引 → 数据库观察 ID
        self._id_map: list[int] = []

    def add(self, observation_id: int, embedding: list[float]):
        """添加一条向量到索引"""
        import faiss

        vec = np.array([embedding], dtype=np.float32)
        # 自动适配维度
        if vec.shape[1] != self.dim:
            logger.warning(f"向量维度 {vec.shape[1]} 与索引维度 {self.dim} 不匹配，跳过")
            return
        faiss.normalize_L2(vec)
        self.index.add(vec)
        self._id_map.append(observation_id)

    def add_batch(self, ids: list[int], embeddings: list[list[float]]):
        """批量添加向量"""
        import faiss

        if not ids or not embeddings:
            return

        matrix = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(matrix)
        self.index.add(matrix)
        self._id_map.extend(ids)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[tuple[int, float]]:
        """
        搜索最相似的 top_k 个向量

        Returns:
            [(数据库观察ID, 相似度分数)] 列表，按相似度从高到低
        """
        if self.index.ntotal == 0:
            return []

        import faiss

        query = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)

        # 搜索 top_k（多取一些以防有无效 ID）
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, k)

        results = []
        for i in range(k):
            idx = int(indices[0][i])
            score = float(scores[0][i])
            if idx >= 0 and idx < len(self._id_map):
                results.append((self._id_map[idx], score))

        return results

    def remove(self, observation_id: int):
        """删除一条向量（重建索引，O(n)）"""
        if observation_id not in self._id_map:
            return

        import faiss

        # 标记要删除的索引位置
        keep_indices = [i for i, oid in enumerate(self._id_map) if oid != observation_id]
        if not keep_indices:
            # 全部删完，重建空索引
            self.index = faiss.IndexFlatIP(self.dim)
            self._id_map = []
        else:
            # 从 FAISS 获取保留的向量并重建
            vectors = faiss.rev_swig_ptr(
                self.index.get_xb(), self.index.ntotal * self.dim
            ).reshape(self.index.ntotal, self.dim)
            kept_vectors = vectors[keep_indices]

            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(kept_vectors.astype(np.float32))
            self._id_map = [self._id_map[i] for i in keep_indices]

    def save(self):
        """持久化索引到磁盘"""
        import faiss
        faiss.write_index(self.index, self.index_path)
        # 同时保存 ID 映射
        map_path = self.index_path + ".ids"
        with open(map_path, "w") as f:
            for oid in self._id_map:
                f.write(f"{oid}\n")
        logger.info(f"FAISS 索引已保存: {self.index.ntotal} 条向量")

    @property
    def total(self) -> int:
        """索引中的向量总数"""
        return self.index.ntotal

    def rebuild_from_db(self, db_conn, dim: int = EMBEDDING_DIM):
        """
        从数据库重建整个索引

        Args:
            db_conn: SQLite 连接
            dim: 向量维度
        """
        import struct
        import faiss

        rows = db_conn.execute(
            "SELECT id, embedding FROM observations WHERE embedding IS NOT NULL AND length(embedding) > 0"
        ).fetchall()

        if not rows:
            self.index = faiss.IndexFlatIP(dim)
            self._id_map = []
            logger.info("索引重建完成（空）")
            return

        ids = []
        vectors = []
        actual_dim = None
        for row in rows:
            obs_id, blob = row
            vec = list(struct.unpack(f"<{len(blob) // 4}f", blob))
            if actual_dim is None:
                actual_dim = len(vec)
            ids.append(obs_id)
            vectors.append(vec)

        # 自动适配数据库中的实际维度
        if actual_dim and actual_dim != dim:
            logger.warning(f"向量维度不匹配: 配置={dim}, 数据库={actual_dim}，使用实际维度")
            self.dim = actual_dim

        self.index = faiss.IndexFlatIP(self.dim)
        matrix = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(matrix)
        self.index.add(matrix)
        self._id_map = ids

        self.save()
        logger.info(f"索引重建完成: {len(ids)} 条向量, 维度={self.dim}")
