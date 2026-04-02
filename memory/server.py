"""
Local Memory MCP Server
通过 MCP 协议为 AI 提供持久化记忆能力

MCP Tools:
  - memory_save: 保存一条记忆
  - memory_search: 语义搜索记忆
  - memory_list: 列出所有记忆
  - memory_delete: 删除指定记忆
  - auto_extract: 自动从对话中提取关键记忆
"""

from mcp.server.fastmcp import FastMCP
from embedding import get_embedding, check_ollama_available
from memory_store import MemoryStore
from extractor import extract_memories_sync
from config import DEFAULT_TOP_K

mcp = FastMCP("local-memory")

store = MemoryStore()


@mcp.tool()
def memory_save(content: str, tags: str = "") -> str:
    """
    保存一条记忆到本地记忆库。

    使用场景：当对话中出现值得记住的信息时，调用此工具保存。
    例如：用户偏好、重要决定、任务状态、关键事实等。

    Args:
        content: 要保存的记忆内容（一段话）
        tags: 标签，逗号分隔（可选，如 "偏好,工作,项目A"）

    Returns:
        保存结果，包含记忆 ID
    """
    try:
        embedding = get_embedding(content)
        memory_id = store.add(content, embedding, tags)
        return f"已保存记忆 (ID: {memory_id}): {content[:50]}..."
    except Exception as e:
        # Ollama 不可用时也能保存，但不带向量（后续可用关键词搜索）
        memory_id = store.add(content, [], tags)
        return f"已保存记忆 (ID: {memory_id}, 无向量-关键词模式): {content[:50]}... | 错误: {e}"


@mcp.tool()
def memory_search(query: str, top_k: int = DEFAULT_TOP_K) -> str:
    """
    语义搜索记忆库中的相关记忆。

    使用场景：当需要回忆之前对话中提到的信息时，调用此工具搜索。
    支持语义搜索（理解含义），不仅仅是关键词匹配。

    Args:
        query: 搜索查询（自然语言描述你想找什么）
        top_k: 返回条数，默认 5

    Returns:
        匹配的记忆列表，按相关性排序
    """
    try:
        query_embedding = get_embedding(query)
        results = store.search(query_embedding, top_k)
    except Exception:
        # Ollama 不可用时降级为关键词搜索
        results_raw = store.keyword_search(query, top_k)
        results = [{**r, "score": None} for r in results_raw]

    if not results:
        return f"未找到与「{query}」相关的记忆。"

    lines = [f"找到 {len(results)} 条相关记忆:\n"]
    for i, r in enumerate(results, 1):
        score_str = f" (相似度: {r['score']})" if r.get("score") is not None else " (关键词匹配)"
        lines.append(f"{i}. [ID:{r['id']}] {r['content']}{score_str}")
        if r.get("tags"):
            lines.append(f"   标签: {r['tags']} | 时间: {r['created_at']}")

    return "\n".join(lines)


@mcp.tool()
def memory_list(limit: int = 20, offset: int = 0) -> str:
    """
    列出记忆库中的所有记忆。

    使用场景：查看当前保存了哪些记忆，或管理记忆库。

    Args:
        limit: 返回条数，默认 20
        offset: 跳过前 N 条（分页用），默认 0

    Returns:
        记忆列表
    """
    memories = store.list_all(limit, offset)
    total = store.count()

    if not memories:
        return "记忆库为空，还没有保存任何记忆。"

    lines = [f"记忆列表 (共 {total} 条, 显示 {offset+1}-{min(offset+limit, total)} 条):\n"]
    for i, m in enumerate(memories, 1):
        lines.append(f"{i}. [ID:{m['id']}] {m['content'][:80]}")
        if m.get("tags"):
            lines.append(f"   标签: {m['tags']} | 时间: {m['created_at']}")

    return "\n".join(lines)


@mcp.tool()
def memory_delete(memory_id: int) -> str:
    """
    删除记忆库中的一条记忆。

    使用场景：当某条记忆已过时或错误时，删除它。

    Args:
        memory_id: 要删除的记忆 ID（从 memory_list 或 memory_search 获取）

    Returns:
        删除结果
    """
    success = store.delete(memory_id)
    if success:
        return f"已删除记忆 (ID: {memory_id})"
    return f"未找到 ID 为 {memory_id} 的记忆"


@mcp.tool()
def auto_extract(text: str, tags: str = "自动提取") -> str:
    """
    自动从对话文本中提取关键记忆并保存。

    使用场景：当对话中出现重要信息时，AI 应主动调用此工具。
    无需用户明确说"记住"，AI 应在以下情况自动调用：
    - 用户透露了个人偏好、习惯、喜好
    - 做出了重要决定或选择了某个方案
    - 提到了项目状态、工作进展、技术选型
    - 任何值得在未来对话中回忆的事实性信息

    工作原理：将对话文本发给本地 LLM，LLM 自动识别并提取关键信息，
    然后逐条存入记忆库。无提取结果则不保存。

    Args:
        text: 对话文本（可以是多轮对话，也可以是单条消息）
        tags: 标签，逗号分隔（可选，默认 "自动提取"）

    Returns:
        提取并保存的记忆列表
    """
    try:
        memories = extract_memories_sync(text)
    except Exception as e:
        return f"记忆提取失败: {e}"

    if not memories:
        return "未从对话中提取到值得记住的信息。"

    saved = []
    for memory in memories:
        try:
            embedding = get_embedding(memory)
            memory_id = store.add(memory, embedding, tags)
            saved.append(f"ID:{memory_id} - {memory}")
        except Exception:
            # 降级保存（无向量）
            memory_id = store.add(memory, [], tags)
            saved.append(f"ID:{memory_id} - {memory}")

    result = f"自动提取并保存了 {len(saved)} 条记忆:\n"
    result += "\n".join(f"  {i+1}. {s}" for i, s in enumerate(saved))
    return result


if __name__ == "__main__":
    ollama_ok = check_ollama_available()
    if not ollama_ok:
        print("[警告] 无法连接 Ollama，语义搜索不可用，将使用关键词搜索模式")
        print("[提示] 请确保 Ollama 正在运行: ollama serve")

    mcp.run(transport="stdio")
