"""
api_actions.py —— 写操作的确认执行（四段式的最后一段）

职责：
    把"AI 建议 + 用户点击确认"变成一次真实的数据库写操作。
    目前唯一的写操作是取消订单：
        POST /api/orders/{order_id}/cancel

四段式回顾（为什么这个接口是最后一段）：
    ① AI 只出建议      ACTION_GUIDE + action=CANCEL_ORDER（模型没有执行权）
    ② harness 校验建议 V4：订单必须被本轮工具查过、状态必须允许取消
    ③ 用户前端确认     页面上出现确认按钮，点了才发请求
    ④ 后端复核执行     本文件：UPDATE 的 WHERE 里再验一次归属和状态

# 难点：②已经校验过状态，④为什么还要再验一次？
# 因为②校验的是"AI 说话那一刻"的世界，④执行的是"用户点按钮那一刻"的世界。
# 两次之间可能隔了几十秒，库存并发、另一个浏览器 tab、甚至攻击者直接
# curl 这个接口，都可能让世界变了样。防线必须在执行现场再设一道 ——
# UPDATE ... WHERE id=? AND user_id=? AND status='pending_shipment'
# 三个条件写进同一条语句，靠数据库的原子性保证"检查和修改之间没有缝隙"。
# 这也是为什么不先 SELECT 检查再 UPDATE：两步之间同样有缝隙。

不负责：
    AI 怎么给出建议（agent.py）、建议是否可信（harness.py）、前端按钮（web/chat.html）。
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from psycopg.rows import dict_row

from app.db import get_conn
from app.schemas import CancelOrderRequest, CancelOrderResponse
from app.security import CurrentUser, get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/orders", tags=["写操作"])


@router.post("/{order_id}/cancel", response_model=CancelOrderResponse)
def cancel_order(
    order_id: int,
    body: CancelOrderRequest | None = None,
    current: CurrentUser = Depends(get_current_user),
) -> CancelOrderResponse:
    """
    用户确认后取消一笔自己的待发货订单。

    Args:
        order_id: 目标订单（orders.id，路径参数）。
        body: 可选的 conversation_id，用于把取消结果写进聊天记录。
        current: 当前登录用户（user_id 从 JWT 来，请求体里不接受）。

    Returns:
        CancelOrderResponse。注意"取消失败"（状态不允许）也是 200：
        这是用户可见的业务结果，不是服务器错误；
        404/403 只留给"订单不存在或不是你的"。

    Raises:
        HTTPException 404: 订单不存在，或存在但不属于当前用户
        （与查看会话的策略一致：响应对外统一 404，防止枚举别人的订单号）。
    """
    # body 可以为空（curl 不带 body 的调用），这里统一成可用的形状
    body = body or CancelOrderRequest()

    # ---------- ① 三重条件 UPDATE：检查与修改在同一语句里原子完成 ----------
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            UPDATE orders
            SET status = 'cancelled'
            WHERE id = %s AND user_id = %s AND status = 'pending_shipment'
            RETURNING id, total_amount
            """,
            (order_id, current.user_id),
        )
        row = cur.fetchone()

    # ---------- ② 影响行数 0：先分清"不是你的"还是"状态不允许" ----------
    if row is None:
        # 必须查一次才能给出准确的原因。安全纪律：只在"确认属于本人"后
        # 才暴露状态细节；查不到或不属于本人一律 404，不给枚举机会。
        with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT id, status FROM orders WHERE id = %s AND user_id = %s",
                (order_id, current.user_id),
            )
            mine = cur.fetchone()

        if mine is None:
            logger.warning(
                "取消订单越权/不存在拦截 | order=%s user=%s", order_id, current.user_id
            )
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="订单不存在")

        # 是本人的订单但状态不是待发货 → 业务失败（200 + success=False）
        reason = f"订单当前状态为「{mine['status']}」，只有待发货的订单可以取消"
        logger.info(
            "取消订单被状态拦截 | order=%s user=%s status=%s", order_id, current.user_id, mine["status"]
        )
        return CancelOrderResponse(order_id=order_id, success=False, message=reason)

    # ---------- ③ 成功：把确定性的确认消息写进会话 ----------
    # 这条消息由后端代码生成（不是模型生成），所以不需要过 harness ——
    # 它本身就是事实的来源。role 用 'ai'：对用户来说"客服侧说的话"，
    # 对模型来说作为 assistant 历史也成立（订单确实取消了，不是幻觉）。
    confirm = f"已为您取消订单 {order_id}（合计 {row['total_amount']} 元），退款将按原支付方式退回，请留意到账通知。"
    if body.conversation_id:
        with get_conn() as conn, conn.cursor() as cur:
            # 归属校验：确认消息只能写进"自己的"会话，防止拿别人的 conversation_id 乱写
            cur.execute(
                """
                INSERT INTO messages (conversation_id, role, content)
                SELECT %s, 'ai', %s
                WHERE EXISTS (SELECT 1 FROM conversations WHERE id = %s AND user_id = %s)
                """,
                (body.conversation_id, confirm, body.conversation_id, current.user_id),
            )

    logger.info(
        "订单已取消 | order=%s user=%s 金额=%s conv=%s",
        order_id, current.user_id, row["total_amount"], body.conversation_id,
    )
    return CancelOrderResponse(order_id=order_id, success=True, message=confirm)
