"""
test_api_actions.py —— 取消订单确认接口的测试

覆盖四段式最后一段的全部防线：
    正常取消、状态拦截、越权 404、不存在 404、确认消息写入。
订单状态在 SAVEPOINT 里改，用例结束自动回滚，种子数据不受影响。
"""


def test_cancel_pending_order_success(client, auth_headers, conversation_id, raw_conn):
    """张三取消待发货的订单1：成功，状态落库，确认消息写进会话。"""
    resp = client.post(
        "/api/orders/1/cancel",
        headers=auth_headers,
        json={"conversation_id": conversation_id},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "2898" in data["message"]

    with raw_conn.cursor() as cur:
        cur.execute("SELECT status FROM orders WHERE id = 1")
        assert cur.fetchone()[0] == "cancelled"
        cur.execute(
            "SELECT role, content FROM messages WHERE conversation_id = %s ORDER BY id DESC LIMIT 1",
            (conversation_id,),
        )
        role, content = cur.fetchone()
        assert role == "ai"
        assert "已为您取消订单 1" in content


def test_cancel_completed_order_blocked(client, auth_headers):
    """订单2 已完成：业务失败（200 + success=False），给用户可读的原因。"""
    resp = client.post("/api/orders/2/cancel", headers=auth_headers, json={})

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
    assert "已完成" in data["message"] or "completed" in data["message"]


def test_cancel_others_order_returns_404(client, auth_headers):
    """
    订单4 属于李四：张三来取消 → 404。

    # 难点：断言的是 404 不是 403 —— 与会话越权同理，
    # 403 会暴露"这个订单存在但不属于你"，可被用来枚举订单号。
    """
    resp = client.post("/api/orders/4/cancel", headers=auth_headers, json={})

    assert resp.status_code == 404


def test_cancel_nonexistent_order_returns_404(client, auth_headers):
    resp = client.post("/api/orders/999/cancel", headers=auth_headers, json={})

    assert resp.status_code == 404


def test_cancel_requires_auth(client):
    """没带 token：401，写操作更是必须验明身份。"""
    resp = client.post("/api/orders/1/cancel", json={})

    assert resp.status_code == 401


def test_confirm_message_not_written_to_others_conversation(client, auth_headers, lisi_headers, raw_conn):
    """
    取消成功但 conversation_id 是别人的：确认消息不落库（防串写）。

    # 难点：取消本身的校验在订单表（user_id 匹配），确认消息的校验在
    # 会话表（user_id 匹配）—— 两处防线互不代劳，各管各的表。
    """
    # 李四还没有会话，先给他建一个（SAVEPOINT 内回滚，不留数据）
    lisi_conv = client.post("/api/conversations", headers=lisi_headers).json()["id"]

    resp = client.post(
        "/api/orders/1/cancel",
        headers=auth_headers,
        json={"conversation_id": lisi_conv},
    )

    assert resp.status_code == 200 and resp.json()["success"] is True
    # 李四的会话一条确认消息都没多（zhangsan 被会话归属校验挡下）
    with raw_conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM messages WHERE conversation_id = %s AND role = 'ai'",
            (lisi_conv,),
        )
        assert cur.fetchone()[0] == 0


def test_cancel_is_idempotent_against_double_click(client, auth_headers, conversation_id):
    """连点两次按钮：第二次是干净的业务失败（已取消=不再是待发货），不是 500。"""
    first = client.post(
        "/api/orders/1/cancel", headers=auth_headers, json={"conversation_id": conversation_id}
    )
    second = client.post("/api/orders/1/cancel", headers=auth_headers, json={})

    assert first.json()["success"] is True
    assert second.status_code == 200
    assert second.json()["success"] is False
