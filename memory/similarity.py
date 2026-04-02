"""
向量相似度计算
使用 numpy 计算余弦相似度
"""

import numpy as np


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    计算两个向量的余弦相似度

    Args:
        vec_a: 向量 A
        vec_b: 向量 B

    Returns:
        余弦相似度 [-1, 1]，越接近 1 越相似
    """
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def search_similar(
    query_embedding: list[float],
    all_embeddings: list[list[float]],
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """
    在所有嵌入向量中搜索与查询最相似的 top_k 个

    Args:
        query_embedding: 查询文本的嵌入向量
        all_embeddings: 所有记忆的嵌入向量列表
        top_k: 返回前 k 个最相似的结果

    Returns:
        [(索引, 相似度分数)] 列表，按相似度从高到低排序
    """
    if not all_embeddings:
        return []

    query = np.array(query_embedding, dtype=np.float32)
    matrix = np.array(all_embeddings, dtype=np.float32)

    # 批量计算余弦相似度
    norms = np.linalg.norm(matrix, axis=1)
    query_norm = np.linalg.norm(query)

    if query_norm == 0:
        return []

    similarities = np.dot(matrix, query) / (norms * query_norm + 1e-10)

    # 按相似度排序，取 top_k
    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [(int(idx), float(similarities[idx])) for idx in top_indices]
