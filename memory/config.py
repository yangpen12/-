"""
Local Memory MCP Server 配置文件
"""

# Ollama 配置（默认端口 11434，如果你的 Ollama 使用其他端口请修改）
OLLAMA_BASE_URL = "http://localhost:11500"
OLLAMA_EMBED_MODEL = "qwen3-embedding:4b"

# 对话模型（用于自动提取记忆，需要推理能力的 LLM）
OLLAMA_CHAT_MODEL = "qwen3:4b"

# 向量维度 (qwen3-embedding 支持 32~4096，1024 是精度和存储的平衡点)
EMBEDDING_DIM = 1024

# SQLite 数据库路径
import os
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "memory.db")

# 搜索默认返回条数
DEFAULT_TOP_K = 5
