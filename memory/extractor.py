"""
记忆自动提取模块
调用本地 LLM 从对话文本中自动提取关键记忆
"""

import json
import httpx
from config import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL


EXTRACT_PROMPT = """你是一个记忆提取助手。从下面的对话文本中提取出值得长期记住的关键信息。

提取规则：
1. 只提取事实性、持久性的信息（用户偏好、重要决定、项目状态、关键事实等）
2. 忽略临时性、不重要的信息（寒暄、闲聊、临时提问等）
3. 每条记忆独立成行，简洁明了
4. 如果没有值得记住的信息，返回空列表

请以 JSON 格式返回，格式如下：
{{"memories": ["记忆1", "记忆2", ...]}}

只返回 JSON，不要返回其他内容。

对话文本：
{text}"""


def extract_memories(text: str) -> list[str]:
    """
    从对话文本中自动提取关键记忆

    Args:
        text: 对话文本（一段话或多轮对话）

    Returns:
        提取出的记忆列表
    """
    prompt = EXTRACT_PROMPT.format(text=text)

    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0.1,  # 低温度，保证输出稳定
        },
    }

    try:
        response = httpx.post(url, json=payload, timeout=300.0)
        response.raise_for_status()
        data = response.json()

        content = data.get("message", {}).get("content", "")

        # 解析 JSON（处理 LLM 可能包裹的 markdown 代码块）
        content = content.strip()
        if content.startswith("```"):
            # 去掉 ```json ... ``` 包裹
            lines = content.split("\n")
            # 过滤掉以 ``` 开头的行
            content = "\n".join(line for line in lines if not line.strip().startswith("```"))
        
        # 尝试直接解析
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # 尝试提取 JSON 部分
            import re
            match = re.search(r'\{[^{}]*"memories"[^{}]*\[.*?\][^{}]*\}', content, re.DOTALL)
            if match:
                result = json.loads(match.group())
            else:
                return []
        
        memories = result.get("memories", [])

        # 过滤空字符串
        return [m.strip() for m in memories if m.strip()]

    except json.JSONDecodeError:
        # LLM 返回了非 JSON 格式，返回空列表
        return []
    except Exception as e:
        raise RuntimeError(f"记忆提取失败: {e}")


# 兼容旧名称
extract_memories_sync = extract_memories