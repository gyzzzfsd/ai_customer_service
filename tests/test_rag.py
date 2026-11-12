"""
test_rag.py —— 知识库检索的单元测试

策略：
    检索 SQL 的"正确性"（算符、JOIN、状态过滤）靠真实数据库验证最有说服力，
    但那需要真实向量 → 需要真模型 → 违背"测试不加载 bge-m3"的原则。
    所以这里把 SQL 结果用假连接喂给 search_chunks，专测**阈值过滤、
    空查询、参数传递**这些纯 Python 逻辑；SQL 本身的正确性由
    tests/test_rag_integration.py 的真实检索单独覆盖（标记 integration）。
"""

from contextlib import contextmanager

import numpy as np
import pytest

from app import rag


class _FakeCursor:
    """假游标：记住收到的 SQL 和参数，fetchall 吐出预置的行。支持上下文管理（rag 里 with 使用）。"""

    def __init__(self, rows):
        self._rows = rows
        self.executed_sql = ""
        self.executed_params: dict | None = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.executed_sql = sql
        self.executed_params = params

    def fetchall(self):
        return self._rows


class _FakeConn:
    """假连接：cursor() 忽略 row_factory，返回假游标。"""

    def __init__(self, rows):
        self._rows = rows
        self.cursor_obj: _FakeCursor | None = None

    def cursor(self, row_factory=None):
        self.cursor_obj = _FakeCursor(self._rows)
        return self.cursor_obj


@pytest.fixture
def patch_rag_deps(monkeypatch):
    """
    把 rag.search_chunks 的两个外部依赖都换成可控假件：
      - embedding.encode_one → 固定向量（反正 SQL 也是假的，向量值无所谓）
      - rag.get_conn → 假连接（预置返回行，同时记录 SQL 参数供断言）
    返回一个 install(rows) 函数，用例自己决定数据库"返回什么"。
    """

    def _install(rows):
        monkeypatch.setattr(
            rag.embedding, "encode_one", lambda q: np.zeros(1024, dtype=np.float32)
        )
        fake_conn = _FakeConn(rows)

        @contextmanager
        def _fake_get_conn():
            yield fake_conn

        monkeypatch.setattr(rag, "get_conn", _fake_get_conn, raising=True)
        return fake_conn

    return _install


def test_blank_query_returns_empty_without_db():
    """空/纯空白 query 直接返回 []，连数据库都不碰（省一次往返）。"""
    assert rag.search_chunks("") == []
    assert rag.search_chunks("   ") == []


def test_threshold_filters_low_scores(patch_rag_deps):
    """
    score 低于 RAG_SCORE_THRESHOLD 的命中被丢弃。

    # 难点：这是"宁可丢弃不硬答"的落地处 ——
    # top_k 取回来的 3 条里可能全是 0.3 分的不相关段落，
    # 不过滤的话模型会被喂一堆噪音，答非所问还带着手册的口吻。
    """
    patch_rag_deps(
        [
            {"title": "手册A", "content": "内容A", "score": 0.82},
            {"title": "手册B", "content": "内容B", "score": 0.45},  # 低于 0.5
            {"title": "手册C", "content": "内容C", "score": 0.51},
        ]
    )

    hits = rag.search_chunks("洗衣机不排水")

    assert [h["title"] for h in hits] == ["手册A", "手册C"]


def test_all_below_threshold_returns_empty(patch_rag_deps):
    """全部低于阈值 → 空列表是正常业务结果（知识库确实没有相关内容）。"""
    patch_rag_deps([{"title": "手册A", "content": "内容A", "score": 0.30}])

    assert rag.search_chunks("随便什么问题") == []


def test_top_k_default_and_override(patch_rag_deps):
    """top_k 未传时用 .env 的 RAG_TOP_K，传了用传的值（SQL 参数可断言）。"""
    conn = patch_rag_deps([])

    rag.search_chunks("问题")
    assert conn.cursor_obj.executed_params["top_k"] == 3  # .env RAG_TOP_K=3

    rag.search_chunks("问题", top_k=5)
    assert conn.cursor_obj.executed_params["top_k"] == 5


def test_sql_uses_cosine_operator(patch_rag_deps):
    """
    检索 SQL 必须用 <=>（余弦距离）算符。

    # 难点：document_chunks 的 HNSW 索引按 vector_cosine_ops 建立，
    # 只有 <=> 查询才走索引；这里把 SQL 原文断言下来，防止将来有人
    # 手滑改成 <-> 导致索引静默失效（数据量小时完全无感，量大会慢几十倍）。
    """
    conn = patch_rag_deps([])

    rag.search_chunks("问题")

    sql = conn.cursor_obj.executed_sql
    assert "<=>" in sql
    assert "d.status = 'ready'" in sql  # 未就绪文档不参与召回
