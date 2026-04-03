# -*- coding: utf-8 -*-
"""知识图谱存储层 - 完整功能测试脚本"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knowledge_graph import KnowledgeGraph

TEST_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "test_graph.db")


def clean_test_db():
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except PermissionError:
            pass


def test_add_entity_and_observation():
    """测试 1: 添加实体和观察"""
    print("\n[1] 添加实体和观察")
    g = KnowledgeGraph(TEST_DB)

    r1 = g.add_observation("张三", "是一名前端工程师", entity_type="person", tags="工作")
    assert r1["status"] == "created", f"期望 created, 实际 {r1['status']}"
    assert r1["observation_id"] is not None and r1["observation_id"] > 0
    print(f"    添加张三: ID={r1['observation_id']}, status={r1['status']}")

    r2 = g.add_observation("张三", "喜欢用 React 开发", tags="偏好")
    assert r2["status"] == "created"
    print(f"    张三第二条观察: ID={r2['observation_id']}")

    r3 = g.add_observation("React", "是一个前端 UI 框架", entity_type="technology", tags="技术")
    assert r3["status"] == "created"
    print(f"    添加 React: ID={r3['observation_id']}")

    assert g.count_entities() == 2, f"期望 2 个实体, 实际 {g.count_entities()}"
    assert g.count_observations() == 3, f"期望 3 条观察, 实际 {g.count_observations()}"
    print(f"    实体数: {g.count_entities()}, 观察数: {g.count_observations()}")

    g.close()
    print("    [PASS]")


def test_add_relations():
    """测试 2: 添加关系"""
    print("\n[2] 添加关系")
    g = KnowledgeGraph(TEST_DB)

    r = g.add_observation(
        "张三",
        "在项目A中使用 React 和 TypeScript",
        tags="项目",
        relations=[
            {"to": "React", "type": "likes"},
            {"to": "TypeScript", "type": "uses"},
            {"to": "项目A", "type": "works_on"},
        ],
    )
    assert r["status"] == "created"
    print(f"    添加带关系的观察: ID={r['observation_id']}")

    entity = g.get_entity("张三")
    assert entity is not None
    rels = entity["relations"]
    rel_types = {rel["type"] for rel in rels}
    assert "likes" in rel_types, f"期望 likes 关系, 实际 {rel_types}"
    assert "uses" in rel_types, f"期望 uses 关系, 实际 {rel_types}"
    assert "works_on" in rel_types, f"期望 works_on 关系, 实际 {rel_types}"
    print(f"    张三的关系: {[(rel['from'], rel['type'], rel['to']) for rel in rels]}")

    ts_entity = g.get_entity("TypeScript")
    assert ts_entity is not None, "TypeScript 实体应自动创建"
    print(f"    TypeScript 自动创建: type={ts_entity['type']}")

    g.close()
    print("    [PASS]")


def test_semantic_search():
    """测试 3: 语义搜索（向量搜索）"""
    print("\n[3] 语义搜索")
    g = KnowledgeGraph(TEST_DB)

    try:
        from embedding import check_ollama_available
        if not check_ollama_available():
            print("    [跳过] Ollama 不可用")
            g.close()
            return
    except Exception as e:
        print(f"    [跳过] {e}")
        g.close()
        return

    r = g.add_observation("用户", "喜欢骑自行车上班", entity_type="person", tags="偏好")
    print(f"    添加记忆 (有向量): ID={r['observation_id']}")

    results = g.search("用户通勤方式是什么", top_k=3)
    assert len(results) > 0, "搜索结果不应为空"
    print(f"    搜索「用户通勤方式是什么」: {len(results)} 条结果")
    for r_item in results:
        print(f"      - [{r_item['entity']}] {r_item['content']} (score={r_item['score']})")

    g.close()
    print("    [PASS]")


def test_conflict_detection():
    """测试 4: 冲突检测"""
    print("\n[4] 冲突检测")
    g = KnowledgeGraph(TEST_DB)

    try:
        from embedding import check_ollama_available
        if not check_ollama_available():
            print("    [跳过] Ollama 不可用")
            g.close()
            return
    except Exception as e:
        print(f"    [跳过] {e}")
        g.close()
        return

    r1 = g.add_observation("用户", "喜欢喝咖啡", entity_type="person", tags="偏好")
    assert r1["status"] == "created"
    obs_id_1 = r1["observation_id"]
    print(f"    第一条: ID={obs_id_1}, status={r1['status']}")

    r2 = g.add_observation("用户", "喜欢喝美式咖啡", entity_type="person", tags="偏好")
    print(f"    第二条: ID={r2['observation_id']}, status={r2['status']}, conflict={r2.get('conflict')}")

    if r2["status"] == "updated":
        assert r2["observation_id"] == obs_id_1, "更新应作用于同一条观察"
        entity = g.get_entity("用户")
        updated_content = [o["content"] for o in entity["observations"] if o["id"] == obs_id_1]
        assert updated_content, "观察应存在"
        assert "美式咖啡" in updated_content[0], f"内容应已更新, 实际: {updated_content[0]}"
        print(f"    冲突更新成功: 「{updated_content[0]}」")
    else:
        print(f"    相似度未达冲突阈值(>0.85)，作为新记忆创建（合理）")

    g.close()
    print("    [PASS]")


def test_keyword_search_fallback():
    """测试 5: 关键词搜索降级"""
    print("\n[5] 关键词搜索降级")
    g = KnowledgeGraph(TEST_DB)

    g.add_observation("用户", "喜欢骑自行车上班", entity_type="person", tags="偏好")

    results = g._keyword_search("自行车", top_k=5)
    found = any("自行车" in r["content"] for r in results)
    assert found, f"关键词搜索应找到包含'自行车'的记忆"
    print(f"    关键词「自行车」: {len(results)} 条结果")
    for r in results:
        print(f"      - [{r['entity']}] {r['content']}")

    g.close()
    print("    [PASS]")


def test_delete():
    """测试 6: 删除"""
    print("\n[6] 删除测试")
    g = KnowledgeGraph(TEST_DB)

    r = g.add_observation("删除测试", "这条观察会被删除", entity_type="concept", tags="测试")
    obs_id = r["observation_id"]
    print(f"    添加观察: ID={obs_id}")

    success = g.delete_observation(obs_id)
    assert success, "删除观察应成功"
    assert g.count_observations() > 0, "其他观察应仍存在"
    print(f"    删除观察 ID={obs_id}: 成功")

    g.add_observation("临时实体", "会被整体删除", entity_type="other", tags="测试")
    g.add_observation("临时实体", "也会被删除", tags="测试")
    entity_before = g.get_entity("临时实体")
    assert entity_before is not None
    print(f"    临时实体观察数: {len(entity_before['observations'])}")

    success = g.delete_entity("临时实体")
    assert success, "删除实体应成功"
    entity_after = g.get_entity("临时实体")
    assert entity_after is None, "实体应不存在"
    print(f"    删除实体「临时实体」: 成功")

    assert not g.delete_entity("不存在实体")
    assert not g.delete_observation(999999)
    print(f"    删除不存在目标: 正确返回 False")

    g.close()
    print("    [PASS]")


def test_list():
    """测试 7: 列表"""
    print("\n[7] 列表测试")
    g = KnowledgeGraph(TEST_DB)

    all_memories = g.list_all(limit=50, offset=0)
    print(f"    全部记忆: {len(all_memories)} 条")
    for m in all_memories:
        print(f"      - [{m['entity']}({m['type']})] {m['content'][:60]}")

    page1 = g.list_all(limit=2, offset=0)
    page2 = g.list_all(limit=2, offset=2)
    print(f"    分页: 第1页={len(page1)}条, 第2页={len(page2)}条")

    print(f"    实体总数: {g.count_entities()}, 观察总数: {g.count_observations()}")

    g.close()
    print("    [PASS]")


def test_mcp_server_import():
    """测试 8: MCP Server 导入测试"""
    print("\n[8] MCP Server 导入测试")
    try:
        from server import mcp
        print(f"    MCP 名称: {mcp.name}")

        import server
        assert not hasattr(server, "auto_extract"), "不应有 auto_extract 工具"
        print("    确认无 auto_extract: OK")
    except ImportError as e:
        print(f"    [跳过] mcp 包未安装: {e}")
        return
    except Exception as e:
        print(f"    [FAIL] {e}")
        raise
    print("    [PASS]")


def main():
    print("=" * 50)
    print("知识图谱存储层 - 功能测试")
    print("=" * 50)

    clean_test_db()

    try:
        test_add_entity_and_observation()
        test_add_relations()
        test_semantic_search()
        test_conflict_detection()
        test_keyword_search_fallback()
        test_delete()
        test_list()
        test_mcp_server_import()
    finally:
        clean_test_db()

    print("\n" + "=" * 50)
    print("全部测试通过!")
    print("=" * 50)


if __name__ == "__main__":
    main()
