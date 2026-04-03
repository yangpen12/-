"""
Local Memory MCP Server
通过 MCP 协议为 AI 提供持久化记忆能力（知识图谱存储）

MCP Tools:
  - memory_save: 保存记忆到知识图谱
  - memory_search: 语义搜索记忆
  - memory_list: 列出所有记忆
  - memory_delete: 删除记忆
"""

import logging
from mcp.server.fastmcp import FastMCP
from knowledge_graph import KnowledgeGraph
from config import DEFAULT_TOP_K

# 日志配置（输出到 stderr，不影响 MCP stdio 通信）
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("local-memory")

mcp = FastMCP("local-memory")

# 懒加载：第一次调用工具时才初始化数据库，避免 MCP 启动超时
_graph = None


def _get_graph() -> KnowledgeGraph:
    global _graph
    if _graph is None:
        logger.info("初始化知识图谱数据库...")
        _graph = KnowledgeGraph()
        logger.info("数据库初始化完成")
    return _graph


@mcp.tool()
def memory_save(
    content: str,
    entity_type: str = "person",
    entity_name: str = "用户",
    tags: str = "",
) -> str:
    """
保存记忆到知识图谱。

【重要：何时主动调用】
当对话中出现以下信息时，你应该主动调用此工具，无需用户明确说"记住"：
1. 用户透露个人偏好、习惯、爱好
2. 用户做出重要决定或选择
3. 用户提到项目状态、技术选型、工作内容
4. 用户提到人际关系、时间节点
5. 用户分析文档时提取到关键信息
6. 用户明确说"记住"、"别忘了"、"帮我记一下"

【参数说明】
- content: 要记住的内容（一段话），系统会自动提取实体和关系
- entity_type: 实体类型（person/technology/project/concept/other）
- entity_name: 实体名称（如"用户"、"张三"、"项目A"）
- tags: 标签，逗号分隔（可选）

Args:
    content: 要保存的记忆内容（一段话）
    entity_type: 实体类型，默认 "person"
    entity_name: 实体名称，默认 "用户"
    tags: 标签，逗号分隔（可选）

Returns:
    保存结果
    """
    try:
        g = _get_graph()
        result = g.add_observation(
            entity_name=entity_name,
            content=content,
            entity_type=entity_type,
            tags=tags,
        )
        status = result["status"]
        obs_id = result["observation_id"]
        entity = result["entity"]

        if status == "updated":
            conflict_info = result.get("conflict", {})
            old_content = conflict_info.get("content", "")[:50] if conflict_info else ""
            logger.info(f"更新记忆 ID={obs_id}, 实体={entity}")
            return (
                f"已更新记忆 (ID: {obs_id}, 实体: {entity})\n"
                f"旧内容: {old_content}...\n"
                f"新内容: {content[:80]}..."
            )
        logger.info(f"保存记忆 ID={obs_id}, 实体={entity}")
        return f"已保存记忆 (ID: {obs_id}, 实体: {entity}): {content[:80]}..."
    except Exception as e:
        logger.error(f"保存失败: {e}")
        return f"保存失败: {e}"


@mcp.tool()
def memory_search(query: str, top_k: int = DEFAULT_TOP_K) -> str:
    """
语义搜索记忆库中的相关记忆。

【重要：主动检索规则】
在以下情况下，你应该**主动调用此工具**检索用户的相关记忆，无需用户明确要求：
1. 新对话开始时（用户说"你好"或第一条消息），检索"用户基本信息、偏好、最近在做什么"
2. 用户提到"之前"、"上次"、"我记得"等时间相关词
3. 用户的问题可能涉及之前透露的偏好、决定、项目状态
4. 用户询问自己的情况（"我喜欢什么"、"我最近在忙什么"）

【禁止检索的情况】
用户明确说"不要用记忆库"、"忽略之前的对话"时，不要调用此工具。

【参数说明】
- query: 搜索查询（自然语言描述你想找什么）
- top_k: 返回条数，默认 5

Args:
    query: 搜索查询（自然语言描述你想找什么）
    top_k: 返回条数，默认 5

Returns:
    匹配的记忆列表，按相关性排序
    """
    try:
        g = _get_graph()
        logger.info(f"搜索: {query}")
        results = g.search(query, top_k)
    except Exception as e:
        logger.error(f"搜索失败: {e}")
        return f"搜索失败: {e}"

    if not results:
        return f"未找到与「{query}」相关的记忆。"

    lines = [f"找到 {len(results)} 条相关记忆:\n"]
    for i, r in enumerate(results, 1):
        score_str = f" (相似度: {r['score']})" if r.get("score") is not None else " (关键词匹配)"
        entity_info = f"[{r['entity']}({r.get('type', '')})]"
        lines.append(f"{i}. {entity_info} [ID:{r['observation_id']}] {r['content']}{score_str}")
        if r.get("tags"):
            lines.append(f"   标签: {r['tags']} | 时间: {r.get('created_at', '')}")
        if r.get("relations"):
            rel_strs = [f"{rel['from']} --{rel['type']}--> {rel['to']}" for rel in r["relations"]]
            lines.append(f"   关系: {', '.join(rel_strs)}")

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
    g = _get_graph()
    memories = g.list_all(limit, offset)
    total_obs = g.count_observations()
    total_entities = g.count_entities()

    if not memories:
        return "记忆库为空，还没有保存任何记忆。"

    lines = [
        f"记忆列表 (共 {total_entities} 个实体, {total_obs} 条观察, "
        f"显示 {offset + 1}-{min(offset + limit, total_obs)} 条):\n"
    ]
    for i, m in enumerate(memories, 1):
        entity_info = f"[{m['entity']}({m.get('type', '')})]"
        lines.append(f"{i}. {entity_info} [ID:{m['observation_id']}] {m['content'][:80]}")
        if m.get("tags"):
            lines.append(f"   标签: {m['tags']} | 时间: {m.get('created_at', '')}")

    return "\n".join(lines)


@mcp.tool()
def memory_delete(target: str, target_type: str = "entity") -> str:
    """
删除记忆库中的一条记忆或整个实体。

使用场景：当某条记忆已过时或错误时，删除它。

Args:
    target: 要删除的目标（实体名称或观察 ID）
    target_type: 目标类型，"entity" 删除整个实体及其所有观察和关系，"observation" 仅删除单条观察

Returns:
    删除结果
    """
    try:
        g = _get_graph()
        if target_type == "observation":
            obs_id = int(target)
            success = g.delete_observation(obs_id)
            if success:
                logger.info(f"删除观察 ID={obs_id}")
                return f"已删除观察 (ID: {obs_id})"
            return f"未找到 ID 为 {obs_id} 的观察"
        else:
            success = g.delete_entity(target)
            if success:
                logger.info(f"删除实体: {target}")
                return f"已删除实体「{target}」及其所有观察和关系"
            return f"未找到实体「{target}」"
    except ValueError:
        return f"无效的观察 ID: {target}（observation 类型需要数字 ID）"
    except Exception as e:
        logger.error(f"删除失败: {e}")
        return f"删除失败: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
