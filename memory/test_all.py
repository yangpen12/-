"""完整功能测试脚本"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embedding import get_embedding, check_ollama_available
from memory_store import MemoryStore
from similarity import cosine_similarity


def main():
    print("=" * 50)
    print("Local Memory MCP Server - 功能测试")
    print("=" * 50)
    
    # 1. Ollama 连接
    print("\n[1] Ollama 连接测试")
    ollama_ok = check_ollama_available()
    print(f"    Ollama 可用: {ollama_ok}")
    
    vec1, vec2 = None, None
    if ollama_ok:
        # 2. 嵌入测试
        print("\n[2] 嵌入向量测试")
        vec1 = get_embedding("用户喜欢骑自行车上班")
        vec2 = get_embedding("用户的工作是软件工程师")
        print(f"    向量维度: {len(vec1)}")
        print(f"    向量示例: {vec1[:3]}...")
        
        # 3. 相似度测试
        print("\n[3] 相似度测试")
        vec3 = get_embedding("天气预报说今天下雨")
        sim1 = cosine_similarity(vec1, vec2)
        sim2 = cosine_similarity(vec1, vec3)
        print(f"    相似文本相似度: {sim1:.4f}")
        print(f"    不相似文本相似度: {sim2:.4f}")
    
    # 4. 存储测试
    print("\n[4] SQLite 存储测试")
    test_db = os.path.join(os.path.dirname(__file__), "data", "test.db")
    if os.path.exists(test_db):
        os.remove(test_db)
    
    store = MemoryStore(test_db)
    id1 = store.add("用户喜欢骑自行车上班", vec1 or [], "偏好")
    id2 = store.add("用户是软件工程师", vec2 or [], "工作")
    # 测试空向量
    id3 = store.add("这是一条无向量的记忆", [], "测试")
    print(f"    添加: ID={id1}, ID={id2}, ID={id3}")
    print(f"    总数: {store.count()}")
    
    if vec1:
        results = store.search(vec1, top_k=3)
        print(f"    搜索结果: {len(results)} 条 (应只匹配有向量的记忆)")
        for r in results:
            print(f"      - ID:{r['id']} 分数:{r['score']:.3f}")
    
    store.delete(id1)
    store.delete(id3)
    print(f"    删除后总数: {store.count()}")
    store.close()
    os.remove(test_db)
    
    # 5. MCP 导入测试
    print("\n[5] MCP Server 导入测试")
    from server import mcp
    print(f"    MCP 名称: {mcp.name}")
    
    # 6. auto_extract 测试（需要对话模型）
    print("\n[6] auto_extract 测试")
    try:
        from extractor import extract_memories
        from config import OLLAMA_CHAT_MODEL, OLLAMA_BASE_URL
        print(f"    对话模型: {OLLAMA_CHAT_MODEL}")
        print(f"    Ollama 地址: {OLLAMA_BASE_URL}")
        
        # 检查对话模型是否可用
        import httpx
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            if any(OLLAMA_CHAT_MODEL in m for m in models):
                print(f"    对话模型已安装，正在测试提取...")
                memories = extract_memories(
                    "我叫小明，我是一名前端工程师，喜欢用 React 和 TypeScript。"
                    "我最近在做一个电商平台项目。"
                )
                if memories:
                    print(f"    提取到 {len(memories)} 条记忆:")
                    for i, m in enumerate(memories, 1):
                        print(f"      {i}. {m}")
                else:
                    print("    提取结果为空（LLM 可能未返回有效 JSON）")
            else:
                print(f"    [跳过] 对话模型 {OLLAMA_CHAT_MODEL} 未安装")
                print(f"    请执行: ollama pull {OLLAMA_CHAT_MODEL}")
        else:
            print("    [跳过] Ollama 未运行")
    except Exception as e:
        print(f"    [跳过] {e}")
    
    print("\n" + "=" * 50)
    print("测试完成!")
    print("=" * 50)

if __name__ == "__main__":
    main()
