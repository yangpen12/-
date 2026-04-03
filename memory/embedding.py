"""
Ollama 嵌入向量接口
调用本地 Ollama API 将文本转换为向量
"""

import httpx
from config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, EMBEDDING_DIM


def get_embedding(text: str) -> list[float]:
    """
    调用 Ollama API 获取文本的嵌入向量

    Args:
        text: 输入文本

    Returns:
        嵌入向量列表 (float)
    """
    url = f"{OLLAMA_BASE_URL}/api/embed"
    payload = {
        "model": OLLAMA_EMBED_MODEL,
        "input": text,
        "options": {
            "dimensions": EMBEDDING_DIM,
        },
    }

    # 首次调用超时设置长一些（Ollama 冷启动）
    response = httpx.post(url, json=payload, timeout=60.0)
    response.raise_for_status()
    data = response.json()

    # Ollama /api/embed 返回 { "embeddings": [[...]] }
    embeddings = data.get("embeddings", [])
    if not embeddings:
        raise ValueError("Ollama 未返回嵌入向量")

    return embeddings[0]


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    批量获取嵌入向量

    Args:
        texts: 文本列表

    Returns:
        嵌入向量列表的列表
    """
    url = f"{OLLAMA_BASE_URL}/api/embed"
    payload = {
        "model": OLLAMA_EMBED_MODEL,
        "input": texts,
        "options": {
            "dimensions": EMBEDDING_DIM,
        },
    }

    response = httpx.post(url, json=payload, timeout=60.0)
    response.raise_for_status()
    data = response.json()

    return data.get("embeddings", [])


def check_ollama_available() -> bool:
    """检查 Ollama 服务是否可用"""
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False
