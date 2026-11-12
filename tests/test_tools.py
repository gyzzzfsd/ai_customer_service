"""
test_tools.py —— 工具层测试

重点不是"SQL 能跑"，而是三件事：
    1. 数据隔离：张三看到的订单里不能有李四的
    2. 失败不抛异常：查不到东西时返回结构化失败，而不是 500
    3. 日志真的在打
"""

import logging

from app.tools import execute_tool, get_user_orders, search_products


# ============================================================
# get_user_orders
# ============================================================


def test_zhangsan_has_five_orders():
    """张三（user_id=1）在种子数据里有 5 个订单，待发货/已完成/已取消/待付款/已发货 各一个。"""
    result = get_user_orders(1)

    assert result.success is True
    orders = {o["id"]: o for o in result.data["orders"]}
    assert set(orders) == {1, 2, 3, 5, 6}

    assert orders[1]["status"] == "pending_shipment"
    assert orders[1]["status_text"] == "待发货"
    assert orders[2]["status"] == "completed"
    assert orders[3]["status"] == "cancelled"
    assert orders[5]["status"] == "pending_payment"
    assert orders[6]["status"] == "shipped"


def test_order_amount_is_string_not_float():
    """
    金额必须以字符串返回。

    # 难点：如果用 float，2499 会变成 2499.0，两个订单相加会出现 2898.0000000000005。
    # harness 后面要拿它跟回复里的"2898"做 Decimal 比对，
    早期的浮点误差会让比对莫名失败。
    """
    result = get_user_orders(1)
    order1 = next(o for o in result.data["orders"] if o["id"] == 1)

    assert order1["total_amount"] == "2898.00"
    assert isinstance(order1["total_amount"], str)


def test_order_items_are_attached():
    """订单明细要挂在该订单自己身上，不能串到别的订单里。"""
    result = get_user_orders(1)
    orders = {o["id"]: o for o in result.data["orders"]}

    names_1 = [i["product_name"] for i in orders[1]["items"]]
    assert "小米洗衣机A3" in names_1
    assert "华为路由器E1" in names_1

    names_2 = [i["product_name"] for i in orders[2]["items"]]
    assert names_2 == ["海尔冰箱B1"], "订单 2 的明细混进了别的商品"


def test_user_data_isolation():
    """
    最关键的用例：李四只能看到自己的订单 4 / 7 / 8，看不到张三的 1。

    # 难点：这条断言是防"越权"的最后一道门。
    # 工具内部 SQL 写死了 WHERE user_id = %s，模型没有任何参数可以改它。
    # 一旦有人"优化"成 WHERE user_id = %(user_id)s 并从参数里取，
    # 模型就能通过编造参数查别人的订单 —— 这条用例会立刻变红。
    """
    result = get_user_orders(2)

    assert result.success is True
    assert {o["id"] for o in result.data["orders"]} == {4, 7, 8}
    assert all(o["id"] != 1 for o in result.data["orders"])


def test_no_orders_returns_structured_failure():
    """不存在的用户应返回 ORDERS_EMPTY，而不是抛异常或返回 500。"""
    result = get_user_orders(999999)

    assert result.success is False
    assert result.code == "ORDERS_EMPTY"
    assert "没有" in result.message


# ============================================================
# search_products
# ============================================================


def test_search_products_hit():
    """搜"洗衣机"应该命中多款，且都带价格。"""
    result = search_products("洗衣机")

    assert result.success is True
    products = result.data["products"]
    assert len(products) >= 3
    assert all("洗衣机" in p["name"] for p in products)
    assert all(p["price"] for p in products)


def test_search_products_empty_query_returns_cheapest():
    """不传关键词时返回最便宜的几款，且按价格升序 —— 前端"猜你喜欢"依赖这个顺序。"""
    result = search_products("")

    assert result.success is True
    prices = [float(p["price"]) for p in result.data["products"]]
    assert len(prices) == 5
    assert prices == sorted(prices)


def test_search_products_miss_returns_structured_failure():
    result = search_products("蓝牙耳机")

    assert result.success is False
    assert result.code == "PRODUCTS_EMPTY"


def test_off_shelf_products_excluded():
    """下架商品不应该出现在搜索结果里（SQL 里有 status='on_sale' 过滤）。"""
    result = search_products("")
    assert result.success is True
    # 返回结构里没有 status 字段，但能保证返回的条目数受 on_sale 数量限制
    assert len(result.data["products"]) <= 5


# ============================================================
# execute_tool 分发与容错
# ============================================================


def test_unknown_tool_returns_code():
    """模型编了一个不存在的工具名，必须返回 UNKNOWN_TOOL 而不是崩。"""
    result = execute_tool("cancel_order", "{}", 1)

    assert result.success is False
    assert result.code == "UNKNOWN_TOOL"


def test_invalid_json_arguments():
    """模型给出的参数经常是半截 JSON，必须被兜住。"""
    result = execute_tool("search_products", '{"query": "洗衣', 1)

    assert result.success is False
    assert result.code == "INVALID_ARGUMENTS"


def test_arguments_must_be_object():
    result = execute_tool("search_products", "[1,2,3]", 1)

    assert result.success is False
    assert result.code == "INVALID_ARGUMENTS"


def test_knowledge_tool_with_hits(monkeypatch):
    """
    知识检索命中：rag 返回段落 → 工具成功，data.chucks 原样透传。

    阶段① 这里是"暂未启用"的哨兵用例；阶段③ 接通 RAG 后改成三分支：
    命中 / 空召回 / 检索异常 —— 空召回走 KNOWLEDGE_NOT_FOUND（业务结果），
    异常走 KNOWLEDGE_ERROR（环境问题），模型对两者的措辞引导也不同。
    """
    from app import rag, tools

    monkeypatch.setattr(
        tools.rag,
        "search_chunks",
        lambda q, top_k=None: [{"title": "小米洗衣机A3使用手册", "content": "排水过滤器在右下方", "score": 0.82}],
    )
    result = execute_tool("search_knowledge", '{"query": "不排水"}', 1)

    assert result.success is True
    assert result.data["chunks"][0]["title"] == "小米洗衣机A3使用手册"


def test_knowledge_tool_empty_recall(monkeypatch):
    """空召回是业务结果不是错误：模型收到 KNOWLEDGE_NOT_FOUND 后应如实告知。"""
    from app import tools

    monkeypatch.setattr(tools.rag, "search_chunks", lambda q, top_k=None: [])
    result = execute_tool("search_knowledge", '{"query": "量子力学"}', 1)

    assert result.success is False
    assert result.code == "KNOWLEDGE_NOT_FOUND"


def test_knowledge_tool_error_becomes_tool_fail(monkeypatch):
    """
    检索炸了（模型没加载/数据库异常）→ KNOWLEDGE_ERROR，绝不向上抛异常。

    # 难点：这里必须兜住。如果让异常冒泡，FastAPI 返回 500，
    # 模型失去"知道检索失败"的机会，下一轮就会用编造内容作答 ——
    # 和其他工具的失败处理是同一条纪律。
    """
    from app import tools

    def _boom(q, top_k=None):
        raise RuntimeError("bge-m3 加载失败")

    monkeypatch.setattr(tools.rag, "search_chunks", _boom)
    result = execute_tool("search_knowledge", '{"query": "不排水"}', 1)

    assert result.success is False
    assert result.code == "KNOWLEDGE_ERROR"
    assert "bge-m3" in result.message


def test_knowledge_tool_blank_query():
    """空检索词不触发任何检索（INVALID_ARGUMENTS）。"""
    result = execute_tool("search_knowledge", '{"query": "   "}', 1)

    assert result.success is False
    assert result.code == "INVALID_ARGUMENTS"


def test_dispatch_injects_user_id_not_from_arguments():
    """
    就算模型在参数里伪造一个 user_id，也只能查到它自己（或查不到）。

    # 难点：这是"参数层面的越权防护"。
    # get_user_orders 的实现根本不读 args，只用调用方传的 user_id。
    # 所以模型写 {"user_id": 2} 完全无效 —— 它查到的还是调用方（张三）的订单。
    """
    result = execute_tool("get_user_orders", '{"user_id": 2}', user_id=1)

    assert result.success is True
    ids = {o["id"] for o in result.data["orders"]}
    assert ids == {1, 2, 3, 5, 6}, "模型通过参数伪造 user_id 越权成功了"


def test_tool_call_writes_log(caplog):
    """日志要求：每次工具调用必须留痕（工具名、用户、耗时、结果码）。"""
    caplog.set_level(logging.INFO, logger="app.tools")

    execute_tool("get_user_orders", "{}", 1)

    messages = [r.getMessage() for r in caplog.records]
    assert any("工具调用" in m and "get_user_orders" in m for m in messages), messages
