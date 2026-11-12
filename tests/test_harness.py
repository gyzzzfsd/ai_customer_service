"""
test_harness.py —— 五关卡校验测试

设计文档《Harness校验设计.md》第 8 节的 8 条用例全部落在这里，
另加 V1 组合规则、V2 边界、补查信号、观测落库的用例。

证据全部从真实数据库构建（张三查一次订单就是种子数据的三张订单），
不造假证据 —— 证据造假会把"比对逻辑对不对"测成"假数据造得像不像"。
"""

from decimal import Decimal

import pytest

from app import harness, run_log, tools
from app.prompts import TOOL_FAIL_CONTENT
from app.schemas import AgentOutput, PageAction, ReplyType, ToolCallRecord


def _answer(content: str, action: PageAction | None = None, reply_type=ReplyType.ANSWER) -> AgentOutput:
    """造一个待校验的 AgentOutput。"""
    return AgentOutput(reply_type=reply_type, content=content, action=action)


@pytest.fixture
def zhangsan_trace(db_isolation) -> list[ToolCallRecord]:
    """张三真实查过订单的轨迹（订单 1/2/3，是构建证据的标准来源）。"""
    result = tools.execute_tool("get_user_orders", "{}", user_id=1)
    assert result.success
    return [ToolCallRecord(tool_name="get_user_orders", arguments={}, result=result.data, success=True)]


@pytest.fixture
def product_trace(db_isolation) -> list[ToolCallRecord]:
    """查过商品（洗衣机）的轨迹，金额证据来自商品售价。"""
    result = tools.execute_tool("search_products", '{"query": "洗衣机"}', user_id=1)
    assert result.success
    return [ToolCallRecord(tool_name="search_products", arguments={"query": "洗衣机"}, result=result.data, success=True)]


# ============================================================
# 一、提取（正则只提取不判断）
# ============================================================


def test_extract_order_ids():
    ids, _, _ = harness.extract_facts("您有订单 1 和订单2，订单编号 3 已发货")
    assert ids == {1, 2, 3}


def test_extract_order_ids_negation_exempt():
    """
    否定语境里的订单号不算业务事实。

    # 难点：没有这条豁免，"没有订单9"这种如实的回答也会被 V3 拦下，
    # 用户问一个不存在的订单，只能收到兜底转人工 —— 误伤比漏杀更伤体验。
    # 相邻式（"没有订单9"）与短语式（"我这边查不到订单9"）分别覆盖。
    """
    # 短语式：中间隔了几个字
    ids, _, _ = harness.extract_facts("我这边查不到订单9，请核对后再试")
    assert ids == set()

    # 相邻式：模型实测最爱的写法
    ids_adj, _, _ = harness.extract_facts("您只有3个订单，没有订单9。")
    assert ids_adj == set()

    # 混句：否定只豁免紧跟着它的那个订单号
    ids2, _, _ = harness.extract_facts("查不到订单9，但您的订单1已发货")
    assert ids2 == {1}

    # 正面陈述不受豁免保护；"没有问题，"不会误豁免后面的订单号
    ids3, _, _ = harness.extract_facts("您的订单9已发货")
    assert ids3 == {9}
    ids4, _, _ = harness.extract_facts("没有问题，订单1已发货")
    assert ids4 == {1}


def test_extract_amounts_only_with_business_label():
    """带标签的金额提取，裸数字不提取 —— 防误杀的核心设计。"""
    _, amounts, _ = harness.extract_facts("订单1合计 2898 元，支持 7 天无理由退货，24 小时客服在线")
    assert amounts == {Decimal("2898")}


@pytest.mark.parametrize(
    "text, expected",
    [
        ("订单1为待发货", {"待发货"}),
        ("订单 2 已完成，订单3 已取消", {"已完成", "已取消"}),
        ("您的包裹已发货", {"已发货"}),
        ("您好请问有什么可以帮您", set()),
    ],
    ids=["单状态", "双状态", "陷阱词", "无状态"],
)
def test_extract_status_words(text, expected):
    _, _, words = harness.extract_facts(text)
    assert words == expected


# ============================================================
# 二、证据构建（来自真实工具结果）
# ============================================================


def test_evidence_from_real_orders(zhangsan_trace):
    ev = harness.build_evidence(zhangsan_trace)

    assert ev.has_order_query is True
    assert ev.order_ids == {1, 2, 3, 5, 6}
    assert ev.order_status == {
        1: "pending_shipment", 2: "completed", 3: "cancelled",
        5: "pending_payment", 6: "shipped",
    }
    # 总额 + 明细单价都算合法金额
    assert Decimal("2898") in ev.amounts          # 订单1 总额
    assert Decimal("2499") in ev.amounts          # 订单1 里洗衣机单价
    assert Decimal("399") in ev.amounts           # 订单1 里路由器单价
    assert Decimal("4599") in ev.amounts          # 订单2 总额


def test_evidence_ignores_failed_and_knowledge():
    """失败的工具、知识库工具都不进证据。"""
    trace = [
        ToolCallRecord(tool_name="get_user_orders", arguments={}, result=None, success=False),
        ToolCallRecord(
            tool_name="search_knowledge",
            arguments={},
            result={"chunks": [{"content": "订单1金额为 99999 元"}]},
            success=True,
        ),
    ]
    ev = harness.build_evidence(trace)

    assert ev.has_order_query is False
    assert ev.order_ids == set()
    assert ev.amounts == set()   # 手册内容不能变成"合法金额"，否则幻觉有了挡箭牌


# ============================================================
# 三、设计文档的 8 条用例（全部基于种子数据）
# ============================================================


def test_case_1_hallucinated_order(zhangsan_trace):
    """用例1：幻觉订单。张三只有 1/2/3，说"订单9"必须拦。"""
    r = harness.validate(_answer("您的订单9已发货"), zhangsan_trace)

    assert r.passed is False
    assert r.error_codes == [harness.UNSUPPORTED_FACT]
    assert "订单9" in r.error_message


def test_case_2_hallucinated_amount(zhangsan_trace):
    """用例2：编造金额。订单1 实际 2898，说 3500 必须拦。"""
    r = harness.validate(_answer("订单1合计 3500 元"), zhangsan_trace)

    assert r.passed is False
    assert r.error_codes == [harness.UNSUPPORTED_FACT]
    assert "3500" in r.error_message


def test_case_3_hallucinated_status(zhangsan_trace):
    """用例3：编造状态。订单1 实际待发货，说"已发货"必须拦（陷阱词）。"""
    r = harness.validate(_answer("订单1已发货，物流单号 SF123"), zhangsan_trace)

    assert r.passed is False
    assert r.error_codes == [harness.UNSUPPORTED_FACT]
    assert "已发货" in r.error_message


def test_case_3b_status_stolen_from_another_order(zhangsan_trace):
    """
    状态词确实在证据里，但属于另一笔订单 —— 同样必须拦。

    # 难点：这条用例专治"集合比对"的漏杀。
    # 张三现在有 5 笔订单，其中订单6 就是已发货：
    #   说"订单6已发货" → 对，放行；
    #   说"订单1已发货" → 状态词在证据里（属于订单6），集合比对照样通过，
    #                      但这是把 B 订单的状态安到 A 订单头上，属于事实漂移。
    # 逐条一致性检查（_check_order_status_consistency）就是为这个场景加的。
    """
    ok = harness.validate(_answer("订单6已发货，物流单号 SF123"), zhangsan_trace)
    assert ok.passed is True, "订单6 本来就是已发货，不该被拦"

    bad = harness.validate(_answer("订单1已发货，物流单号 SF123"), zhangsan_trace)
    assert bad.passed is False
    assert bad.error_codes == [harness.UNSUPPORTED_FACT]
    assert "订单1" in bad.error_message


def test_case_4_cross_user_action(zhangsan_trace):
    """用例4：越权动作。订单4 属于李四，永远进不了张三的证据。

    正文故意不写订单号：写"订单4"的话会先被 V3 以 UNSUPPORTED_FACT 拦下
    （正文提到无依据订单同样是幻觉），这里专测 V4 对 action 的拦截。
    """
    action = PageAction(code="CANCEL_ORDER", order_id=4)
    r = harness.validate(_answer("好的，马上为您处理", action, ReplyType.ACTION_GUIDE), zhangsan_trace)

    assert r.passed is False
    assert r.error_codes == [harness.UNVERIFIED_ACTION_RESOURCE]
    assert "订单4" in r.error_message
    # 张三已经查过订单，补查救不回来（再查也没有订单4），不该要求补查
    assert r.needs_order_query is False


def test_case_5_uncancelable_order_action(zhangsan_trace):
    """用例5：不可取消订单给按钮。订单2 已完成。"""
    action = PageAction(code="CANCEL_ORDER", order_id=2)
    r = harness.validate(_answer("订单2可以取消", action, ReplyType.ACTION_GUIDE), zhangsan_trace)

    assert r.passed is False
    assert r.error_codes == [harness.UNVERIFIED_ACTION_RESOURCE]
    assert "completed" in r.error_message


def test_case_6_invented_action(zhangsan_trace):
    """用例6：发明动作。白名单里只有 CANCEL_ORDER。"""
    action = PageAction(code="REFUND_ORDER", order_id=1)
    r = harness.validate(_answer("好的，为您退款", action, ReplyType.ACTION_GUIDE), zhangsan_trace)

    assert r.passed is False
    assert r.error_codes == [harness.UNVERIFIED_ACTION_RESOURCE]
    assert "REFUND_ORDER" in r.error_message


def test_case_7_normal_pass(zhangsan_trace):
    """用例7：正常通过。订单号、金额、状态全部有据。"""
    r = harness.validate(_answer("订单1为待发货，合计 2898 元"), zhangsan_trace)

    assert r.passed is True
    assert r.error_codes == []
    assert r.degraded is None


def test_case_8_no_false_positive_on_bare_number(zhangsan_trace):
    """
    用例8：误杀回归。"7 天退货"的 7 不带业务标签，不得当金额校验。

    # 难点：这条比"能拦住"更重要。校验器拦得越狠，误杀越多，
    # 用户收到"已转人工"的次数就越多 —— 拦截是手段，不误伤正常回复才是底线。
    """
    r = harness.validate(_answer("支持 7 天无理由退货，订单1为待发货"), zhangsan_trace)

    assert r.passed is True


def test_product_price_is_valid_evidence(product_trace):
    """商品价格（来自 search_products）也是合法金额证据。"""
    ev = harness.build_evidence(product_trace)
    r = harness.validate(_answer("这款洗衣机价格 2499 元，现货充足"), product_trace)

    assert Decimal("2499") in ev.amounts
    assert r.passed is True


# ============================================================
# 四、V1 结构 / V2 前置降级 / 补查信号
# ============================================================


def test_v1_action_with_wrong_reply_type(zhangsan_trace):
    """action 只能出现在 ACTION_GUIDE 里。"""
    action = PageAction(code="CANCEL_ORDER", order_id=1)
    r = harness.validate(_answer("订单1待发货", action, ReplyType.ANSWER), zhangsan_trace)

    assert r.passed is False
    assert r.error_codes == [harness.MODEL_OUTPUT_INVALID]


def test_v1_action_guide_without_action(zhangsan_trace):
    """ACTION_GUIDE 必须携带 action。"""
    r = harness.validate(_answer("请确认取消订单1", None, ReplyType.ACTION_GUIDE), zhangsan_trace)

    assert r.passed is False
    assert r.error_codes == [harness.MODEL_OUTPUT_INVALID]


def test_v1_empty_content(zhangsan_trace):
    r = harness.validate(_answer("   "), zhangsan_trace)

    assert r.passed is False
    assert r.error_codes == [harness.MODEL_OUTPUT_INVALID]


def test_v2_all_business_tools_failed():
    """业务工具全部失败 → 静默换话术、清空 action，且算"通过"（内容是系统控制的）。"""
    trace = [
        ToolCallRecord(tool_name="get_user_orders", arguments={}, result=None, success=False),
        ToolCallRecord(tool_name="search_products", arguments={}, result=None, success=False),
    ]
    output = _answer("您的订单1合计 3500 元")  # 编造的话也无所谓，反正要被换掉

    r = harness.validate(output, trace)

    assert r.passed is True
    assert r.degraded == "tool_failure"
    assert r.output.content == TOOL_FAIL_CONTENT
    assert r.output.reply_type == ReplyType.REQUEST_HANDOFF
    assert r.output.action is None


def test_v2_not_triggered_when_no_tool_called():
    """一个工具都没调 → 不算基础设施失败（该拦还是拦，走补查/重写那套）。"""
    r = harness.validate(_answer("订单1待发货"), [])

    assert r.degraded is None
    assert r.passed is False                # 没证据 → UNSUPPORTED_FACT
    assert r.needs_order_query is True      # 但值得补查一次


def test_v2_knowledge_failure_is_not_infra_failure():
    """手册里没查到是正常业务结果，不是故障，不触发 V2。"""
    trace = [ToolCallRecord(tool_name="search_knowledge", arguments={}, result=None, success=False)]

    r = harness.validate(_answer("我查不到相关说明，建议转人工"), trace)

    assert r.passed is True
    assert r.degraded is None


def test_needs_order_query_when_action_without_evidence():
    """
    按钮指向订单但从未查过订单 → 要补查，不是直接拒。

    这是实测缺陷的对症设计：模型沿用历史直接给按钮时，
    先补一次查询，复验大概率通过，用户不会被无端拒绝。
    """
    action = PageAction(code="CANCEL_ORDER", order_id=1)
    r = harness.validate(_answer("好的，为您取消订单1", action, ReplyType.ACTION_GUIDE), [])

    assert r.passed is False
    assert r.needs_order_query is True


# ============================================================
# 五、summarize 与 agent_runs 落库
# ============================================================


class _FakeRun:
    """summarize 只取属性，用最小对象就能测，不必构造完整 AgentRun。"""

    def __init__(self, validation, corrected=False, error=None):
        self.validation = validation
        self.corrected = corrected
        self.error = error


def test_summarize_passed(zhangsan_trace):
    r = harness.validate(_answer("订单1为待发货"), zhangsan_trace)
    assert run_log.summarize(_FakeRun(r))["status"] == "passed"


def test_summarize_corrected():
    r = harness.HarnessResult(passed=True, output=_answer("x"))
    s = run_log.summarize(_FakeRun(r, corrected=True))
    assert s["status"] == "corrected"
    assert s["passed"] is True


def test_summarize_rejected():
    r = harness.HarnessResult(passed=False, error_codes=[harness.UNSUPPORTED_FACT], output=_answer("x"))
    s = run_log.summarize(_FakeRun(r))
    assert s["status"] == "rejected"
    assert s["error_codes"] == [harness.UNSUPPORTED_FACT]


def test_summarize_degraded_tools():
    r = harness.HarnessResult(passed=True, degraded="tool_failure", output=_answer("x"))
    assert run_log.summarize(_FakeRun(r))["status"] == "degraded_tools"


def test_summarize_degraded_model():
    s = run_log.summarize(_FakeRun(None, error="LLM_ERROR: 主备模型都失败"))
    assert s["status"] == "degraded_model"
    assert s["error_codes"] == ["LLM_ERROR"]


def test_record_run_writes_row(raw_conn):
    """观测表落库 + 查回验证（用例结束回滚，不留数据）。"""
    # 问题文本用唯一值：同一场测试里 api_chat 的用例也会落 agent_runs，
    # 按问题文本查回时会串行，用唯一值避免测试顺序依赖。
    question = "落库验证专用问题-唯一"
    run_log.record_run(
        conversation_id=1,
        user_id=1,
        question=question,
        reply_type="ACTION_GUIDE",
        status="corrected",
        error_codes=["UNVERIFIED_ACTION_RESOURCE"],
        tool_call_count=2,
        duration_ms=1234,
    )
    with raw_conn.cursor() as cur:
        cur.execute(
            "select question, validation_status, error_codes, tool_call_count "
            "from agent_runs where question = %s",
            (question,),
        )
        row = cur.fetchone()

    assert row is not None
    assert row[1] == "corrected"
    assert row[2] == ["UNVERIFIED_ACTION_RESOURCE"]
    assert row[3] == 2


def test_correction_feedback_contains_specific_error():
    text = harness.correction_feedback("订单9不在您的订单列表中")
    assert "订单9" in text
    assert "未通过服务端校验" in text
