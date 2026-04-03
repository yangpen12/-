# Local Memory — 本地 AI 记忆系统 V2

一个极简的本地 AI 记忆系统，通过 MCP 协议接入 AI 工具（CodeBuddy / Claude），让 AI 拥有跨会话的持久化记忆能力。

**完全本地运行，零云端依赖，数据在你自己手里。**

## 它解决什么问题？

AI 每次新对话都"失忆"——你告诉过它的偏好、决定、上下文全都忘了。这个工具让 AI 能记住这些东西。

## V2 改动（相比 V1）

| 改动 | 说明 |
|------|------|
| 🗑️ 移除本地对话模型 | 不再需要跑 `gemma2:9b`，AI 自己就能判断何时保存记忆 |
| 🧠 知识图谱存储 | 从扁平文本升级为实体-关系-观察三表结构 |
| 🔍 FAISS 向量索引 | 集成 Facebook FAISS 加速语义搜索，支持大规模记忆 |
| 📐 2560 维向量 | 使用 `qwen3-embedding:4b` 嵌入，Ollama 忽略 dimensions 参数，实际输出 2560 维 |
| ⚡ 懒加载启动 | MCP Server 启动不再阻塞，首次调用工具时才初始化 |
| 🔄 冲突自动处理 | 相似度 > 0.85 自动更新旧记忆，避免重复存储 |
| 🛡️ 多级降级策略 | FAISS → numpy 暴力搜索 → SQL LIKE 关键词匹配 |
| 📝 统一 top_k=5 | 搜索默认返回 5 条，保持一致体验 |

## 核心设计

**知识图谱存储**（参考 Claude 官方 MCP Memory Server）：

- **实体**（Entities）：人、技术、项目、概念等
- **关系**（Relations）：实体之间的关联（喜欢、使用、属于…）
- **观察**（Observations）：关于实体的具体事实

例如你说"张三喜欢用 React 开发"，系统自动构建：
```
张三 --likes--> React
张三 --uses--> React
观察: "张三是一名前端工程师，喜欢用 React"
```

## 功能

| 工具 | 功能 | 说明 |
|------|------|------|
| `memory_save` | 保存记忆 | AI 自动判断何时保存，无需你手动说"记住" |
| `memory_search` | 语义搜索记忆 | FAISS 加速 + numpy 降级 + 关键词兜底 |
| `memory_list` | 列出所有记忆 | 查看记忆库内容，支持分页 |
| `memory_delete` | 删除记忆 | 删除实体或单条观察 |

### AI 主动记忆

AI 会在以下情况**自动调用** `memory_save` 保存记忆，无需你明确说"记住"：
1. 你透露个人偏好、习惯、爱好
2. 你做出重要决定或选择
3. 你提到项目状态、技术选型、工作内容
4. 你提到人际关系、时间节点
5. 你让 AI 分析文档时提取到关键信息
6. 你明确说"记住"、"别忘了"

### AI 主动检索

AI 会在以下情况**主动检索**你的记忆，无需你提醒：
1. **新对话开始时** — AI 会先检索你的基本信息、最近在做什么
2. 你提到"之前"、"上次"、"我记得"等时间相关词
3. 你问自己的情况（"我喜欢什么"、"我最近在忙什么"）
4. 你的问题可能涉及之前透露的偏好、决定、项目状态

> 这样你开新对话，AI 就知道你是谁、在准备什么面试、最近发生了什么。

### 记忆冲突自动处理

当新记忆与旧记忆相似度 > 0.85 时，系统自动**更新旧记忆**而不是重复新建。

冲突检测使用已生成的向量直接比对（不重复调用 Ollama），保证性能。

## 快速开始

### 1. 前置条件

- **Python 3.10+**（推荐 3.12+）
- **Ollama** 已安装并运行，需要一个嵌入模型：

```bash
ollama pull qwen3-embedding:4b
```

### 2. 安装依赖

```bash
cd memory
pip install -r requirements.txt
```

依赖说明：
- `mcp[cli]` — Anthropic 官方 MCP Python SDK
- `httpx` — 调用 Ollama HTTP API
- `numpy` — 向量相似度计算（FAISS 降级备选）
- `faiss-cpu` — Facebook FAISS 向量索引（加速语义搜索）

> `faiss-cpu` 是可选的。未安装时自动退回 numpy 暴力搜索，功能不受影响。

### 3. 配置

默认配置开箱即用，通常**不需要修改**。

如果你知道自己的 Ollama 端口不是默认的 `11434`，编辑 `config.py` 修改：

```python
# Ollama 地址（默认 11434，如果不是这个端口请修改）
OLLAMA_BASE_URL = "http://localhost:11434"

# 嵌入模型（文本 → 向量，用于语义搜索）
OLLAMA_EMBED_MODEL = "qwen3-embedding:4b"

# 向量维度（qwen3-embedding:4b 实际输出 2560 维，Ollama 忽略 dimensions 参数）
EMBEDDING_DIM = 2560
```

> **如何确认端口**：在终端运行 `ollama list` 能正常输出即端口正确。如果连接失败，检查 Ollama 系统托盘图标或环境变量 `OLLAMA_HOST`。

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
      "description": "本地记忆系统 - 知识图谱存储，让AI记住对话内容"
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

接入成功后，AI 会自动获得 4 个记忆工具。

#### 自动记忆（推荐）

你正常聊天就行，AI 会自动判断并保存重要信息：

> "我最近在用 Python 写一个爬虫，用的 Scrapy 框架，数据库用 MongoDB"

AI 会自动保存为知识图谱：
- 实体：用户（person）、Python（technology）、Scrapy（technology）、MongoDB（technology）
- 关系：用户 --uses--> Python、用户 --uses--> Scrapy、用户 --uses--> MongoDB
- 观察："用户在做爬虫项目，使用 Scrapy 框架，数据库用 MongoDB"

#### 手动记忆

> "记住我喜欢用 Python 和 VS Code 写代码"

AI 会调用 `memory_save` 保存。

#### 回忆

> "我的编程偏好是什么？"

AI 会调用 `memory_search` 搜索相关记忆并回答。

#### 管理

> "列出所有记忆" → `memory_list`
> "删除张三的所有记忆" → `memory_delete`

## 文件结构

```
memory/
├── server.py            # MCP Server 入口（4 个工具，懒加载启动）
├── knowledge_graph.py   # 知识图谱存储层（实体+关系+观察，集成 FAISS）
├── vector_index.py      # FAISS 向量索引（增删查、持久化、自动重建）
├── embedding.py         # Ollama 嵌入接口（文本→向量，60s 超时）
├── similarity.py        # Numpy 余弦相似度计算（FAISS 降级备选）
├── config.py            # 配置文件（端口、模型、维度）
├── requirements.txt     # Python 依赖
├── run.bat              # Windows 一键启动
├── test_all.py          # 功能测试脚本
├── README.md            # 本文件
└── data/
    ├── graph.db         # SQLite 数据库（运行后自动创建）
    ├── faiss.index      # FAISS 索引文件（运行后自动创建）
    └── faiss.index.ids  # FAISS ID 映射文件
```

## 工作原理

### 保存流程
```
AI 判断有重要信息 → 调用 memory_save
→ 自动创建/更新实体 → 生成嵌入向量（Ollama qwen3-embedding:4b）
→ 冲突检测（直接向量比对，相似度 > 0.85 自动更新，不重复调用 API）
→ SQLite 存储 + FAISS 索引同步
```

### 搜索流程
```
用户提问 → AI 调用 memory_search → 查询文本生成向量
→ FAISS 快速搜索（优先）→ numpy 暴力搜索（降级）→ SQL LIKE（兜底）
→ 返回最相似的 5 条（含实体关系信息）
```

### 降级策略

如果 Ollama 没运行：
- 保存记忆：仍然可以保存，但不生成向量
- 搜索记忆：自动降级为 SQL `LIKE` 关键词匹配
- 冲突检测：自动降级为内容完全匹配检测

如果 FAISS 未安装：
- 搜索自动退回 numpy 暴力搜索，功能不受影响
- 日志提示 `FAISS 未安装，退回 numpy 搜索`

### FAISS 索引管理

- **自动初始化**：启动时加载已有索引，若索引为空但数据库有向量则自动重建
- **自动同步**：增删改操作后自动同步索引并持久化到磁盘
- **维度自适应**：`rebuild_from_db` 自动检测数据库中的实际向量维度
- **持久化**：索引文件 `faiss.index` + ID 映射 `faiss.index.ids` 存储在 data 目录

## 常见问题

**Q: 安装 mcp 失败？**
A: 需要 Python 3.10+，3.9 不支持。升级 Python 后重新 `pip install -r requirements.txt`。

**Q: 搜索不到记忆？**
A: 检查 Ollama 是否在运行。如果没运行，搜索会降级为关键词模式。

**Q: Ollama 端口不是 11434？**
A: 修改 `config.py` 中的 `OLLAMA_BASE_URL` 为你的实际端口。

**Q: 数据存在哪里？**
A: `memory/data/graph.db`（SQLite 数据库）+ `memory/data/faiss.index`（向量索引）。备份直接复制整个 `data/` 目录。

**Q: 想换其他嵌入模型？**
A: 修改 `config.py` 中的 `OLLAMA_EMBED_MODEL`。只需确保模型支持嵌入接口（`/api/embed`）。
注意：Ollama 会忽略 `dimensions` 参数，实际输出维度取决于模型本身。首次使用新模型时，FAISS 索引会自动重建。

**Q: FAISS 安装失败怎么办？**
A: `faiss-cpu` 是可选依赖。未安装时搜索自动退回 numpy 实现，功能不受影响。
可尝试：`pip install faiss-cpu` 或 `conda install -c conda-forge faiss-cpu`。

**Q: 不需要本地对话模型吗？**
A: 不需要。AI（如 CodeBuddy 中的大模型）自己就能判断何时保存记忆，不需要本地再跑一个对话模型。

**Q: 第一次调用工具很慢？**
A: 这是正常的。MCP Server 使用懒加载——首次调用工具时才初始化知识图谱和 FAISS 索引（需要加载数据库和索引文件）。后续调用会很快。

## 参考方案

- [Claude 官方 MCP Memory Server](https://github.com/modelcontextprotocol/servers/tree/main/src/memory) — 知识图谱结构设计
- [Mem0](https://github.com/mem0ai/mem0) — 记忆冲突自动合并
- [mcp-memory-service](https://github.com/doobidoo/mcp-memory-service) — 混合搜索
- [FAISS](https://github.com/facebookresearch/faiss) — 高效向量相似度搜索

## License

MIT

