"""
Local Memory MCP Server 配置文件
"""

# Ollama 配置（默认端口 11434，如果你的 Ollama 使用其他端口请修改）
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBED_MODEL = "qwen3-embedding:4b"

# 向量维度（qwen3-embedding:4b 实际输出 2560 维，Ollama 忽略 dimensions 参数）
EMBEDDING_DIM = 2560

# SQLite 数据库路径
import os
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "graph.db")

# 搜索默认返回条数
DEFAULT_TOP_K = 5
