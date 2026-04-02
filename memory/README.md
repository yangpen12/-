# Local Memory — 本地 AI 记忆系统

一个极简的本地 AI 记忆系统，通过 MCP 协议接入 AI 工具（CodeBuddy / Claude），让 AI 拥有跨会话的持久化记忆能力。

**完全本地运行，零云端依赖，数据在你自己手里。**

## 它解决什么问题？

AI 每次新对话都"失忆"——你告诉过它的偏好、决定、上下文全都忘了。这个工具让 AI 能记住这些东西。

## 功能

| 工具 | 功能 | 示例 |
|------|------|------|
| `memory_save` | 手动保存一条记忆 | "记住：用户喜欢用 Python 写代码" |
| `memory_search` | 语义搜索记忆 | 搜索"用户的编程语言偏好" → 匹配到"喜欢用 Python" |
| `memory_list` | 列出所有记忆 | 查看记忆库里存了什么 |
| `memory_delete` | 删除一条记忆 | 清理过时的记忆 |
| `auto_extract` | ⭐ 自动提取记忆 | AI 自动从对话中识别关键信息并保存，无需用户说"记住" |

## 快速开始

### 1. 前置条件

- **Python 3.10+**（推荐 3.12+）
- **Ollama** 已安装并运行，需要两个模型：
  - 嵌入模型（用于语义搜索）：`qwen3-embedding:4b`
  - 对话模型（用于自动提取记忆）：`qwen3:4b`

```bash
# 拉取嵌入模型
ollama pull qwen3-embedding:4b

# 拉取对话模型（用于 auto_extract 自动提取）
ollama pull qwen3:4b
```

### 2. 安装依赖

```bash
cd d:/treaWJ/td/memory
pip install -r requirements.txt
```

依赖说明：
- `mcp[cli]` — Anthropic 官方 MCP Python SDK
- `httpx` — 调用 Ollama HTTP API
- `numpy` — 向量相似度计算

### 3. 配置

编辑 `config.py` 修改配置：

```python
# Ollama 地址（默认 11434，如果你的 Ollama 使用其他端口请修改）
# Windows 默认: http://localhost:11434
# 如果连接失败，打开终端执行 ollama serve 查看实际端口
OLLAMA_BASE_URL = "http://localhost:11434"

# 嵌入模型（文本 → 向量）
OLLAMA_EMBED_MODEL = "qwen3-embedding:4b"

# 对话模型（用于 auto_extract 自动提取记忆）
OLLAMA_CHAT_MODEL = "qwen3:4b"

# 向量维度
EMBEDDING_DIM = 1024
```

### 4. 接入 AI 工具

#### 接入 CodeBuddy

1. 打开 CodeBuddy → 侧边栏对话框右上角 **Settings**
2. 切换到 **MCP** 标签页
3. 点击 **Add MCP**
4. 填入以下配置：

```json
{
  "mcpServers": {
    "local-memory": {
      "type": "stdio",
      "command": "python",
      "args": ["d:/treaWJ/td/memory/server.py"],
      "description": "本地记忆系统 - 让AI记住对话内容"
    }
  }
}
```

5. 保存后，状态显示为绿色即接入成功

#### 接入 Claude Desktop

1. 打开配置文件：`%APPDATA%\Claude\claude_desktop_config.json`
2. 添加以下内容：

```json
{
  "mcpServers": {
    "local-memory": {
      "command": "python",
      "args": ["d:/treaWJ/td/memory/server.py"]
    }
  }
}
```

3. 重启 Claude Desktop

#### 接入 Claude Code

在终端执行：
```bash
claude mcp add local-memory -- python d:/treaWJ/td/memory/server.py
```

### 5. 使用

接入成功后，AI 会自动获得 5 个记忆工具。

#### 手动记忆（你说"记住"）

> "记住我喜欢用 Python 和 VS Code 写代码"

AI 会调用 `memory_save` 保存。

#### 自动记忆（AI 主动提取）⭐

你正常聊天就行，AI 会在以下情况**自动调用 `auto_extract`**：
- 你透露了个人偏好、习惯
- 做出了重要决定
- 提到项目状态、技术选型
- 任何值得长期记住的事实

例如你随便说：
> "我最近在用 Python 写一个爬虫，用的 Scrapy 框架，数据库用 MongoDB"

AI 会自动提取并保存：
- "用户会 Python 编程"
- "用户在做爬虫项目"
- "用户使用 Scrapy 框架"
- "用户项目使用 MongoDB 数据库"

#### 回忆

> "我的编程偏好是什么？"

AI 会调用 `memory_search` 搜索相关记忆并回答。

#### 管理

> "列出所有记忆" → `memory_list`
> "删除 ID 为 3 的记忆" → `memory_delete`

## 文件结构

```
memory/
├── server.py          # MCP Server 入口（注册 5 个工具）
├── memory_store.py    # SQLite 存储层（增删改查）
├── embedding.py       # Ollama 嵌入接口（文本→向量）
├── extractor.py       # 记忆自动提取（LLM 提取关键信息）
├── similarity.py      # 余弦相似度计算
├── config.py          # 配置文件
├── requirements.txt   # Python 依赖
├── run.bat            # Windows 一键启动
├── test_all.py        # 功能测试脚本
├── README.md          # 本文件
└── data/
    └── memory.db      # SQLite 数据库（运行后自动创建）
```

## 工作原理

### 手动记忆流程
```
用户说"记住xxx" → AI 调用 memory_save → Ollama 生成向量 → SQLite 存储
```

### 自动记忆流程
```
用户正常聊天 → AI 判断有重要信息 → 调用 auto_extract
→ 本地 LLM 分析对话文本 → 提取出关键记忆 → 逐条生成向量 → SQLite 存储
```

### 搜索流程
```
用户提问 → AI 调用 memory_search → 查询文本生成向量
→ 和库中所有向量算余弦相似度 → 返回最相似的几条
```

## 降级策略

如果 Ollama 没运行：
- 保存记忆：仍然可以保存，但不生成向量
- 搜索记忆：自动降级为 SQL `LIKE` 关键词匹配
- 自动提取：不可用（需要 LLM）

## 常见问题

**Q: 安装 mcp 失败？**
A: 需要 Python 3.10+，3.9 不支持。升级 Python 后重新 `pip install -r requirements.txt`。

**Q: 搜索不到记忆？**
A: 检查 Ollama 是否在运行。如果没运行，搜索会降级为关键词模式。

**Q: auto_extract 提取不到记忆？**
A: 需要 Ollama 中有对话模型（如 qwen3:4b）。检查 `config.py` 中的 `OLLAMA_CHAT_MODEL` 配置。

**Q: Ollama 端口不是 11434？**
A: 修改 `config.py` 中的 `OLLAMA_BASE_URL` 为你的实际端口。

**Q: 数据存在哪里？**
A: `memory/data/memory.db`，一个 SQLite 文件。备份直接复制这个文件就行。

**Q: 想换其他嵌入模型？**
A: 修改 `config.py` 中的 `OLLAMA_EMBED_MODEL` 和 `EMBEDDING_DIM`。

## License

MIT
