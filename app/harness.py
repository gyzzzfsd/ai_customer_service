"""
harness.py —— 五关卡校验（本项目的核心亮点）

职责：
    校验"AI 说的每句业务事实有没有工具依据"。模型负责生成，工具提供事实，
    本文件负责核对，核对不过的回复不发给用户。五道关卡顺序固定：

        V1 结构校验   reply_type/action 的组合是否合法（查表，零正则）
        V2 前置降级   业务工具全部失败 → 静默换固定话术，不再往下走
        V3 事实校验   正则提取订单号/金额/状态 → 与证据集合比对
        V4 动作校验   动作码白名单 + 订单已查询 + 状态可取消
        V5 纠错与兜底  由 refine.py 编排：反馈错误重写 1 次，仍不过换兜底

    配套：correction_feedback() 拼纠错反馈文本（V5 的"喂回去的错误"）。

不负责：
    生成回复（agent.py）、收口状态机（refine.py）、执行工具（tools.py）、
    提示词文案（prompts.py）、观测落库（run_log.py）。

# 难点：V5 的重试循环为什么不放在这里？
# 因为"重写"需要再调一次模型，那是编排行为。harness 若自己调模型，
# 就变成了"既当裁判又当教练"，单元测试也得 mock 模型才能跑。
# 拆开后 harness 是纯函数（进：输出+证据，出：结论），测试不需要任何 mock；
# refine 拿着结论决定"补查、重写还是兜底"。判断和执行分离，两边都简单。

证据规则（铁律）：
    只认本次运行 run_trace 里 success=True 的 get_user_orders / search_products；
    search_knowledge 的结果永远不进证据 —— 手册内容是"说法"不是"业务事实"，
    AI 引用手册说"怎么清洁滤网"无需证据，说"您的订单 1 是 2898 元"必须有工具背书。
"""

import logging
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from app.prompts import CORRECTION_FEEDBACK_TEMPLATE, TOOL_FAIL_CONTENT
from app.schemas import ALLOWED_ACTION_CODES, AgentOutput, ReplyType, ToolCallRecord

logger = logging.getLogger(__name__)

# ============================================================
# 一、错误码与提取规则
# ============================================================

# 三个稳定错误码：阶段④ 前端和 agent_runs 表都按它过滤统计
MODEL_OUTPUT_INVALID = "MODEL_OUTPUT_INVALID"          # 结构/组合不合法
UNSUPPORTED_FACT = "UNSUPPORTED_FACT"                  # 订单号/金额/状态查无实据
UNVERIFIED_ACTION_RESOURCE = "UNVERIFIED_ACTION_RESOURCE"  # 动作无依据或不可执行

# 订单号：只认"订单 + 数字"的写法，"订单 1""订单编号 3"都命中
ORDER_ID_RE = re.compile(r"订单\s*(?:编号\s*)?(\d+)")

# 否定语境线索：出现在订单号**前面**时，这个订单号不算"业务事实"。
# 用于区分两种句子：
#     "您的订单9已发货"          —— 正面陈述，必须有证据，拦
#     "没有订单9" / "查不到订单9" —— 否定陈述，说的就是"它不存在"，证据里没有它恰恰是对的
# 没有这条豁免，用户问一个不存在的订单，AI 如实回答"查无此单"也会被打回，
# 兜底成"转人工"—— 用户被无端拒绝，这正是 harness 最不该犯的错。
#
# 两种匹配方式（实测模型最爱的写法是"没有订单9"，必须两种都有）：
#   相邻式（前缀以线索结尾）：覆盖"没有订单9""查无订单9"——紧贴订单号，
#       用 endswith 而不是 in，防止"没有问题，订单1已发货"被误豁免；
#   短语式（前缀包含线索）：覆盖"我这边查不到订单9""未找到订单9"——中间隔了几个字。
_ADJACENT_CUES = ("没有", "无", "未", "查不到", "找不到", "不存在")
_PHRASE_CUES = ("查不到", "找不到", "没有查到", "未找到", "没有找到", "不存在", "无此", "没有这个")

# 金额：只认带业务标签的数字（合计/总价/金额/价格/售价……元）。
# 这是控制误杀的关键 —— "7 天无理由退货"的 7、"24 小时客服"的 24 都不带标签，
# 一律不管。宁可漏杀（放过个别编造），不能误杀（把正常回复打回）。
AMOUNT_RE = re.compile(r"(?:合计|总价|金额|价格|售价)\D{0,3}(\d+(?:\.\d{1,2})?)\s*元")

# 订单状态词表：覆盖 orders.status 的全部真实取值。
# 注意：状态校验不能只做"这个词出现过没有"——
# 用户有多笔订单时，"订单1已发货"里的"已发货"可能来自订单6，
# 集合比对会漏杀。所以 _check_facts 之后还有一道逐条一致性检查，
# 见 _check_order_status_consistency。
STATUS_WORDS = {
    "待发货": "pending_shipment",
    "已完成": "completed",
    "已取消": "cancelled",
    "已发货": "shipped",
}

# 唯一允许引导的写操作状态：待发货
CANCELABLE_STATUS = "pending_shipment"


# ============================================================
# 二、数据结构
# ============================================================


@dataclass
class Evidence:
    """本次运行从工具结果里构建的"盖章答案"。"""

    order_ids: set[int] = field(default_factory=set)        # 查到的订单号
    amounts: set[Decimal] = field(default_factory=set)      # 合法金额（订单总额+明细单价+商品售价）
    statuses: set[str] = field(default_factory=set)         # 出现过的订单状态码
    order_status: dict[int, str] = field(default_factory=dict)  # 订单号→状态，V4 用
    has_order_query: bool = False                           # 本次是否成功查过订单


@dataclass
class HarnessResult:
    """一次 validate 的完整结论。"""

    passed: bool                          # True 才允许发给用户
    output: AgentOutput                   # 最终输出（V2 降级时已被替换成话术）
    error_codes: list[str] = field(default_factory=list)
    error_message: str = ""               # 喂给模型重写的具体错误（V5 用）
    needs_order_query: bool = False       # 编排层据此强制补查一次（见下方说明）
    degraded: str | None = None           # 'tool_failure' / None

    # 难点：为什么要 needs_order_query 这个字段？
    # 实测发现：用户追问"帮我取消订单1"时，模型可能根本不调工具，
    # 直接沿用上一轮历史里的订单信息作答（日志里表现为 工具=0次）。
    # 此时 V3/V4 失败的原因是"没有证据"，而不是"证据矛盾"——
    # 直接打回重写，模型手里还是没有证据，只会被无端拒绝（用户被误伤）。
    # 正确动作是先强制补查一次 get_user_orders，再拿新证据复验。
    # "该不该补查"是判断，harness 给出；"怎么补查"是执行，agent 负责。


# ============================================================
# 三、对外入口
# ============================================================


def validate(output: AgentOutput, tool_trace: list[ToolCallRecord]) -> HarnessResult:
    """
    五关卡校验（V5 的重写循环在 agent.py，本函数只判断一轮）。

    Args:
        output: agent 结构化出的 AgentOutput。
        tool_trace: 本次运行的工具调用记录（证据唯一来源）。

    Returns:
        HarnessResult。passed=True 表示可以发给用户（V2 降级也算"通过"，
        但 output 已被替换且 degraded='tool_failure'）。
    """
    # ---------- V1 结构：组合规则 ----------
    err = _check_structure(output)
    if err:
        return HarnessResult(
            passed=False,
            output=output,
            error_codes=[MODEL_OUTPUT_INVALID],
            error_message=err,
        )

    # ---------- V2 前置降级：业务工具全部失败 ----------
    if _all_business_tools_failed(tool_trace):
        degraded_output = AgentOutput(
            reply_type=ReplyType.REQUEST_HANDOFF,
            content=TOOL_FAIL_CONTENT,
            action=None,
        )
        logger.warning("V2 前置降级 | 业务工具全部失败，静默换固定话术")
        return HarnessResult(passed=True, output=degraded_output, degraded="tool_failure")

    # ---------- V3 事实 + V4 动作 ----------
    evidence = build_evidence(tool_trace)
    order_ids, amounts, status_words = extract_facts(output.content)

    v3_msg = _check_facts(order_ids, amounts, status_words, evidence) or _check_order_status_consistency(
        output.content, evidence
    )
    if v3_msg:
        # 只有"提到订单但从未查过订单"才值得补查；查过仍对不上就是编造
        return HarnessResult(
            passed=False,
            output=output,
            error_codes=[UNSUPPORTED_FACT],
            error_message=v3_msg,
            needs_order_query=(bool(order_ids) and not evidence.has_order_query),
        )

    v4_msg = _check_action(output, evidence)
    if v4_msg:
        return HarnessResult(
            passed=False,
            output=output,
            error_codes=[UNVERIFIED_ACTION_RESOURCE],
            error_message=v4_msg,
            # 按钮指向的订单不在证据里，且从未查过订单 → 补查可能救回来
            needs_order_query=(
                output.action is not None
                and output.action.order_id not in evidence.order_ids
                and not evidence.has_order_query
            ),
        )

    return HarnessResult(passed=True, output=output)


def build_evidence(tool_trace: list[ToolCallRecord]) -> Evidence:
    """
    从工具轨迹构建证据集合。

    只收 success=True 的 get_user_orders / search_products；
    search_knowledge 的结果永远不进证据（手册是说法，不是业务事实）。

    # 难点：归属核验为什么不需要单独做？
    # get_user_orders 的 SQL 写死了 WHERE user_id=JWT 用户（见 tools.py），
    # 证据集合天然只含本人订单 —— 李四的订单 4 永远进不了张三的证据，
    # 越权动作会在 V4 的"订单号不在证据里"这一条被自然拦掉。
    """
    ev = Evidence()
    for record in tool_trace:
        if not record.success or record.result is None:
            continue

        if record.tool_name == "get_user_orders":
            ev.has_order_query = True
            for order in record.result.get("orders", []):
                ev.order_ids.add(order["id"])
                ev.statuses.add(order["status"])
                ev.order_status[order["id"]] = order["status"]
                ev.amounts.add(Decimal(str(order["total_amount"])))
                for item in order.get("items", []):
                    ev.amounts.add(Decimal(str(item["unit_price"])))

        elif record.tool_name == "search_products":
            for product in record.result.get("products", []):
                ev.amounts.add(Decimal(str(product["price"])))

    return ev


def extract_facts(text: str) -> tuple[set[int], set[Decimal], set[str]]:
    """
    从回复正文提取业务事实（正则在这里只干"提取"这一件事）。

    Args:
        text: AgentOutput.content。

    Returns:
        (订单号集合, 金额集合, 状态中文词集合)。
        提取结果为空 = 这句话不含需要证据的业务事实，直接放行。

    # 难点：正则只提取、不判断，这是本文件最重要的纪律。
    # "订单号对不对"是集合成员问题，正则管不了：金额的合法写法无穷多
    # （2898 元 / ¥2,898.00 / 2898.0），试图用一个大正则判断"整句是否合法"
    # 必然枚举不全。先提取、再用 Decimal 精确比对，把问题降维成查表。
    """
    order_ids = set()
    for m in ORDER_ID_RE.finditer(text):
        # 否定语境豁免：见 _ADJACENT_CUES / _PHRASE_CUES 的说明。这不算
        # "判断事实真伪"，只是提取阶段过滤"AI 自己声明不存在"的引用 ——
        # 和金额必须带标签是同一性质的提取规则，不是校验逻辑。
        # 窗口 8 个字符：盖住一个谓语短语，又不会跨过逗号误伤下一个分句。
        prefix = text[max(0, m.start() - 8) : m.start()]
        negated = prefix.endswith(_ADJACENT_CUES) or any(cue in prefix for cue in _PHRASE_CUES)
        if not negated:
            order_ids.add(int(m.group(1)))
    amounts = {Decimal(m) for m in AMOUNT_RE.findall(text)}
    status_words = {w for w in STATUS_WORDS if w in text}
    return order_ids, amounts, status_words



def correction_feedback(error_message: str) -> str:
    """拼 V5 纠错反馈的完整文本（模板在 prompts.py，话术统一放一处）。"""
    return CORRECTION_FEEDBACK_TEMPLATE.format(error=error_message)


# ============================================================
# 四、各关卡内部实现
# ============================================================


def _check_structure(output: AgentOutput) -> str | None:
    """
    V1：组合规则（字段类型已由 Pydantic 保证，这里只查跨字段契约）。

    Returns:
        错误描述（None 表示通过）。
    """
    if not (output.content or "").strip():
        return "content 为空字符串"
    if output.action is not None and output.reply_type != ReplyType.ACTION_GUIDE:
        return f"action 只能出现在 ACTION_GUIDE 中，当前 reply_type 是 {output.reply_type.value}"
    if output.reply_type == ReplyType.ACTION_GUIDE and output.action is None:
        return "ACTION_GUIDE 必须携带 action（code + order_id）"
    return None


def _all_business_tools_failed(tool_trace: list[ToolCallRecord]) -> bool:
    """
    V2：业务工具（订单/商品）是否"调过且全部失败"。

    两个容易搞错的边界：
        - 一个都没调 → 不算失败（走 V3，由证据缺失那套逻辑处理）；
        - search_knowledge 失败不算 —— "手册里没写"是正常业务结果，
          不是基础设施故障，模型应该转人工而不是被换话术。
    """
    business = [r for r in tool_trace if r.tool_name in ("get_user_orders", "search_products")]
    return bool(business) and not any(r.success for r in business)


def _check_facts(
    order_ids: set[int],
    amounts: set[Decimal],
    status_words: set[str],
    evidence: Evidence,
) -> str | None:
    """
    V3：提取出的事实逐个到证据集合里找，找不到的收集起来。

    Returns:
        错误描述（None 表示通过）。错误必须具体到"哪个事实没依据"，
        否则模型不知道该删哪句话。

    # 难点：金额比对为什么必须用 Decimal？
    # 数据库里是 NUMERIC("2898.00")，模型嘴里的可能是"2898"或"2898.0"。
    # float 会先把 2898.00 变成 2897.9999999 再比较；Decimal 则保证
    # Decimal("2898.00") == Decimal("2898") 为真，格式差异自动消失。
    """
    problems: list[str] = []

    bad_ids = order_ids - evidence.order_ids
    if bad_ids:
        problems.append("订单" + "、".join(str(i) for i in sorted(bad_ids)))

    bad_amounts = amounts - evidence.amounts
    if bad_amounts:
        problems.append("金额" + "、".join(str(a) for a in sorted(bad_amounts)) + "元")

    # 状态词先过映射表转成状态码，再和证据里的状态码集合比对
    bad_statuses = [w for w in sorted(status_words) if STATUS_WORDS[w] not in evidence.statuses]
    if bad_statuses:
        problems.append("状态" + "、".join(bad_statuses))

    if not problems:
        return None
    return "回复中的业务事实缺少工具支持：" + "；".join(problems)


# 分句符：逐条状态核对只在"一个分句里恰好一个订单号 + 一个状态词"时生效。
# 为什么必须分句：模型正常回答是"订单1待发货，订单2已完成"，
# 整句看有两个订单、两个状态，无法归属到某一单；切开后每个小句都能对上号。
_CLAUSE_SPLIT_RE = re.compile(r"[，。；;！!？?\n]")


def _check_order_status_consistency(text: str, evidence: Evidence) -> str | None:
    """
    V3 补强：逐条核对"订单 N + 状态"是否与证据一致（防跨订单串状态）。

    Returns:
        错误描述（None 表示通过或无法归属）。

    # 难点：为什么光有集合比对不够？
    # evidence.statuses 是"该用户全部订单状态的并集"。
    # 一个用户有 5 笔订单（待发货/已完成/已取消/待付款/已发货）时，
    # 模型把订单1说成"已发货"也能通过集合比对 —— 这个词确实在证据里，
    # 只是属于订单6。这与本项目早期"把 A 订单的商品金额安到 B 订单"是同一类错误，
    # 只有把状态重新绑定到订单号才能拦住。
    # 反过来，多订单/多状态的句子无法归属，宁可放过也不能误杀，
    # 所以只在"恰好一个订单号并恰好一个状态词"时下结论。
    """
    for clause in _CLAUSE_SPLIT_RE.split(text or ""):
        ids = {int(m.group(1)) for m in ORDER_ID_RE.finditer(clause)}
        words = {w for w in STATUS_WORDS if w in clause}
        if len(ids) != 1 or len(words) != 1:
            continue
        order_id = next(iter(ids))
        word = next(iter(words))
        actual = evidence.order_status.get(order_id)
        if actual is not None and actual != STATUS_WORDS[word]:
            return f"订单{order_id}的真实状态是 {actual}，回复里写成了「{word}」"
    return None


def _check_action(output: AgentOutput, evidence: Evidence) -> str | None:
    """
    V4：动作校验三条规则（查表 + 集合，零正则）。

    Returns:
        错误描述（None 表示通过）。没有 action 时直接通过。
    """
    action = output.action
    if action is None:
        return None

    if action.code not in ALLOWED_ACTION_CODES:
        return f"动作码 {action.code} 不在白名单内（仅支持 CANCEL_ORDER）"

    if action.order_id not in evidence.order_ids:
        return f"取消按钮指向的订单{action.order_id}未经本次工具查询确认（先查后引导）"

    status = evidence.order_status.get(action.order_id)
    if status != CANCELABLE_STATUS:
        return f"订单{action.order_id}状态为 {status}，只有待发货订单才能取消"

    return None
