"""
tools.py —— 模型能调用的工具（证据来源）

职责：
    定义 3 个只读工具（查订单、查商品、查知识库）+ 给模型看的工具说明 + 统一分发入口。
    所有 SQL 都写在这个文件里 —— 因为 SQL 是工具的实现细节，跟着工具走，
    而不是单独抽一层 repository 让人来回跳。

不负责：
    决定调哪个工具（agent.py 里由模型决定）、校验工具结果是否被正确引用（harness.py）。

设计要点（整个项目最关键的三条）：
    1. 工具全部只读。没有 cancel_order 这类写工具 —— 从能力层面就不给模型改库的可能。
    2. user_id 永远不在参数表里，由后端从 JWT 取出后注入 SQL。
       模型只能决定"查不查"，改不了"查谁"，越权在 SQL 条件里就被掐死。
    3. 工具失败不抛异常，返回 ToolResult(success=False, code=..., message=...)。
       把"没查到"如实告诉模型，它才不会编造；异常会把整个请求打断，模型失去纠错机会。
"""

import json
import logging
import time
from typing import Any

from psycopg.rows import dict_row

from app import rag
from app.config import settings
from app.db import get_conn
from app.schemas import ToolResult

logger = logging.getLogger(__name__)

# 订单状态的中文说明。给模型看的是"待发货"而不是 pending_shipment，
# 否则模型容易在回答里直接蹦出英文状态码，体验很差。
ORDER_STATUS_TEXT = {
    "pending_payment": "待付款",
    "pending_shipment": "待发货",
    "shipped": "已发货",
    "completed": "已完成",
    "cancelled": "已取消",
}


# ============================================================
# 一、工具实现
# ============================================================


def get_user_orders(user_id: int) -> ToolResult:
    """
    查当前用户自己的全部订单（含商品明细）。

    Args:
        user_id: 当前登录用户 ID。**由 agent.py 从 JWT 解出后传入，不接受模型传参**。

    Returns:
        ToolResult，data 形如：
            {"orders": [{"id": 1, "status": "pending_shipment", "status_text": "待发货",
                         "total_amount": "2898.00", "address": "...", "created_at": "...",
                         "items": [{"product_name": "...", "quantity": 1, "unit_price": "2499.00"}]}]}
        用户没有订单时返回 success=False, code=ORDERS_EMPTY。

    # 难点（越权防护）：这条 SQL 的 WHERE 里写死了 user_id。
    # 工具的参数定义（见下方 TOOL_SCHEMAS）里根本没有 user_id 这一项，
    # 所以模型就算"想"查别人的订单也无从下手 —— 它连参数都填不了。
    # 把归属校验放在 SQL 里，而不是放在后面的业务代码里，是因为 SQL 是唯一必经之路，
    # 少一处 if 判断就少一个漏判的机会。
    """
    sql = """
        SELECT id, status, total_amount, address, created_at
        FROM orders
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT 10
    """
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, (user_id,))
        orders = cur.fetchall()

        if not orders:
            return ToolResult.fail("ORDERS_EMPTY", "当前用户没有任何订单")

        # 一次把所有订单的明细查回来，避免 N+1 查询（10 条订单循环查 10 次）
        order_ids = [o["id"] for o in orders]
        cur.execute(
            """
            SELECT order_id, product_name, quantity, unit_price
            FROM order_items
            WHERE order_id = ANY(%s)
            """,
            (order_ids,),
        )
        items_by_order: dict[int, list[dict]] = {}
        for item in cur.fetchall():
            items_by_order.setdefault(item["order_id"], []).append(
                {
                    "product_name": item["product_name"],
                    "quantity": item["quantity"],
                    # 金额统一转成字符串下发：JSON 里的 float 会出现 2499.0000000001 这种
                    # 浮点误差，而模型和 harness 都要拿它跟"2898"这类字面量比对。
                    "unit_price": str(item["unit_price"]),
                }
            )

    data = [
        {
            "id": o["id"],
            "status": o["status"],
            "status_text": ORDER_STATUS_TEXT.get(o["status"], o["status"]),
            "total_amount": str(o["total_amount"]),
            "address": o["address"],
            "created_at": o["created_at"].isoformat(timespec="minutes"),
            "items": items_by_order.get(o["id"], []),
        }
        for o in orders
    ]
    return ToolResult.ok({"orders": data}, message=f"共 {len(data)} 个订单")


def search_products(query: str = "") -> ToolResult:
    """
    按关键词搜在售商品。

    Args:
        query: 关键词，可为空。为空时返回价格最低的 5 个在售商品（用于"你们卖什么"）。

    Returns:
        ToolResult，data 形如 {"products": [{"id", "name", "category", "price", "stock", "description"}]}。
        没搜到时 success=False, code=PRODUCTS_EMPTY。
    """
    keyword = (query or "").strip()
    like = f"%{keyword}%"

    sql = """
        SELECT id, name, category, price, stock, description
        FROM products
        WHERE status = 'on_sale'
          AND (%s = '' OR name ILIKE %s OR category ILIKE %s OR description ILIKE %s)
        ORDER BY price ASC
        LIMIT 5
    """
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, (keyword, like, like, like))
        rows = cur.fetchall()

    if not rows:
        return ToolResult.fail(
            "PRODUCTS_EMPTY",
            f"没有找到与「{keyword}」相关的在售商品" if keyword else "当前没有在售商品",
        )

    data = [
        {
            "id": r["id"],
            "name": r["name"],
            "category": r["category"],
            "price": str(r["price"]),
            "stock": r["stock"],
            "description": (r["description"] or "")[:200],
        }
        for r in rows
    ]
    return ToolResult.ok({"products": data}, message=f"匹配到 {len(data)} 个商品")


def search_knowledge(query: str) -> ToolResult:
    """
    在操作手册知识库里检索与问题最相关的段落（RAG）。

    Args:
        query: 用户问题的关键内容。

    Returns:
        ToolResult，data 形如 {"chunks": [{"content": "...", "title": "...", "score": 0.82}]}。
        三种失败各走各的码：
            KNOWLEDGE_NOT_FOUND  检索正常但无相关内容（空召回是业务结果不是错误）
            KNOWLEDGE_ERROR      检索本身失败（模型没加载、数据库异常）
            INVALID_ARGUMENTS    query 为空

    # 阶段③ 起本工具已接真实向量检索（rag.search_chunks）。
    # 之前阶段① 的占位实现返回"暂未启用"，现在模型能拿到真实的手册段落，
    # 但"没查到就如实说"的行为约定从未变过 —— 这是防幻觉的关键一环。
    """
    if not (query or "").strip():
        return ToolResult.fail("INVALID_ARGUMENTS", "检索内容不能为空")

    # 阶段③ 已接通：rag.search_chunks 做向量检索，空列表表示知识库确实
    # 没有相关内容（阈值过滤后的正常业务结果），返回结构化失败让模型如实告知
    # 并建议转人工，而不是把"没查到"当成功硬答。签名与返回结构保持阶段① 的
    # 约定不变（chunks: [{title, content, score}]），agent.py 一行没改。
    # 首次调用会触发 bge-m3 懒加载（约 15 秒），embedding.py 的日志有提示。
    try:
        hits = rag.search_chunks(query)
    except Exception as exc:  # noqa: BLE001 —— 模型未配置/加载失败也不打断问答
        # exc_info=True 把堆栈打进日志：异常类型进了给模型的 message，
        # 但定位"为什么加载失败"需要堆栈（只有日志里有）
        logger.warning("知识库检索异常 | query=%s", query, exc_info=True)
        return ToolResult.fail(
            "KNOWLEDGE_ERROR",
            # 带上原始异常信息（但不带堆栈）：模型不需要，日志里 exception 会补全
            f"知识库检索失败（{type(exc).__name__}: {exc}），请如实告知用户暂时查不到，建议转人工",
        )

    if not hits:
        return ToolResult.fail(
            "KNOWLEDGE_NOT_FOUND",
            "知识库中未找到与问题相关的内容，请如实告知用户，不要编造手册内容，建议转人工",
        )
    return ToolResult.ok({"chunks": hits})


# ============================================================
# 二、给模型看的工具定义（OpenAI function calling 格式）
# ============================================================

# description 直接决定模型会不会在正确的时机调这个工具，写得越具体越好。
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_user_orders",
            "description": (
                "查询当前登录用户自己的订单列表（含商品明细、状态、金额）。"
                "用户问『我的订单』『订单到哪了』『我要退款/取消』时调用。"
                "注意：该工具不需要任何参数，系统会自动按当前登录用户过滤，"
                "你无法也无需指定用户。"
            ),
            # 难点：parameters 是空对象，且没有任何 user_id 参数。
            # 这是"物理隔离"而不是"约定"：模型即便幻觉出 user_id，也没有地方可填。
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": (
                "按关键词搜索本店在售商品，返回名称、价格、库存、简介。"
                "用户问『有没有 XX』『XX 多少钱』『推荐个洗衣机』时调用。"
                "关键词可以为空，此时返回价格最低的几款商品。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "商品关键词，例如『洗衣机』『海尔』『冰箱』；不确定就留空",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "在商品使用手册知识库里检索操作说明，返回最相关的段落。"
                "用户问『怎么用』『怎么安装』『报错怎么办』『保修多久』这类"
                "说明书里的问题时调用。问订单状态不要用它，用 get_user_orders。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要在手册里检索的问题要点，尽量保留用户原话里的产品型号",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


# ============================================================
# 三、统一分发入口
# ============================================================

# 工具名 → 实现函数。加新工具只需在这里加一行 + 上面写一个函数。
_TOOL_FUNCS = {
    "get_user_orders": lambda args, user_id: get_user_orders(user_id),
    "search_products": lambda args, user_id: search_products(args.get("query", "")),
    "search_knowledge": lambda args, user_id: search_knowledge(args.get("query", "")),
}


def execute_tool(name: str, arguments: str, user_id: int) -> ToolResult:
    """
    执行一个工具调用，**任何情况下都返回 ToolResult，绝不抛异常**。

    Args:
        name: 模型给出的工具名。
        arguments: 模型给出的参数 JSON 字符串（可能是残缺/非法的 JSON）。
        user_id: 当前登录用户，由 agent 从 JWT 注入，用于覆盖模型可能乱传的值。

    Returns:
        ToolResult。失败分为三类：
            INVALID_ARGUMENTS  参数不是合法 JSON
            UNKNOWN_TOOL       模型编了一个不存在的工具名
            TOOL_ERROR         工具内部异常（SQL 报错、连不上库等）

    本函数只做两件事：计时 + 打日志，具体的参数解析和分发在 _dispatch 里。
    这样"每次工具调用都一定有日志"是结构上保证的，而不是靠每个分支记得写。

    # 难点：为什么这里要 catch 住所有异常？
    # 因为工具是"给模型看的"，不是"给人看的"。
    # 一旦异常冒泡到 FastAPI，用户看到 500，模型也被剥夺了自我纠正的机会：
    # 它永远不知道刚才那次查询失败了，下一轮就会用编造的数据回答。
    # 把异常转成 success=False 的结构化结果回填给模型，它才有机会说"查询失败，请稍后重试"。
    """
    started = time.perf_counter()
    result = _dispatch(name, arguments, user_id)
    elapsed_ms = (time.perf_counter() - started) * 1000

    # 成功用 INFO，失败用 WARNING：日志里扫一眼就知道哪些工具经常查不到东西，
    # 这是后续判断"是不是该补数据/改工具描述"的依据。
    log = logger.info if result.success else logger.warning
    log(
        "工具调用 | name=%-16s user=%s 耗时=%5.0fms success=%-5s code=%-20s args=%s",
        name,
        user_id,
        elapsed_ms,
        result.success,
        result.code,
        _brief(arguments),
    )
    return result


def _dispatch(name: str, arguments: str, user_id: int) -> ToolResult:
    """
    解析参数并按工具名分发。所有失败都转成 ToolResult，不抛异常。

    Args:
        name: 工具名。
        arguments: 参数 JSON 字符串。
        user_id: 当前用户 ID。

    Returns:
        ToolResult。
    """
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return ToolResult.fail("INVALID_ARGUMENTS", f"参数不是合法 JSON：{arguments[:100]}")

    if not isinstance(parsed, dict):
        return ToolResult.fail("INVALID_ARGUMENTS", "参数必须是一个 JSON 对象")

    func = _TOOL_FUNCS.get(name)
    if func is None:
        return ToolResult.fail("UNKNOWN_TOOL", f"不存在名为 {name} 的工具")

    try:
        return func(parsed, user_id)
    except Exception as exc:  # noqa: BLE001 —— 这里就是要兜住所有异常
        # 异常堆栈要完整打出来（exc_info），否则"TOOL_ERROR"回到模型那里只是一句话，
        # 排查时完全不知道是 SQL 写错了还是连接断了。
        logger.exception("工具 %s 执行异常", name)
        return ToolResult.fail("TOOL_ERROR", f"工具执行失败：{type(exc).__name__}: {exc}")


def _brief(arguments: str, limit: int = 80) -> str:
    """把参数压成一行短文本用于日志，避免一次搜索把整段 JSON 打进日志。"""
    text = (arguments or "").replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."
