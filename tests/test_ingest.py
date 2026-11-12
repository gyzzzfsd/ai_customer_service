"""
test_ingest.py —— 向量回填脚本的单元测试

策略：
    回填的核心风险有三个：漏补（幂等性失效）、维度错配写到一半才发现、
    批次提交策略被悄悄改坏。这里全用假件验证控制流，真实回填
    （335 条、GPU 编码）只在上线时手动跑一次，不做自动化。
"""

import numpy as np
import pytest

from app import ingest


@pytest.fixture
def fake_flow(monkeypatch):
    """
    把 ingest.run 的三个协作者都换成记录器：
      fetch_pending → 预置待回填行
      embedding.encode → 预置向量
      write_vectors → 只记录调用不碰库
    返回 dict，用例从里面取记录结果。
    """
    state = {
        "pending": [(1, "文本一"), (2, "文本二"), (3, "文本三")],
        "encoded": [],
        "written": [],
        "cleared": False,
    }

    monkeypatch.setattr(ingest, "fetch_pending", lambda rebuild: state["pending"])
    monkeypatch.setattr(ingest, "clear_vectors", lambda: state.__setitem__("cleared", True) or 10)
    monkeypatch.setattr(
        ingest.embedding,
        "encode",
        lambda texts: state["encoded"].append(list(texts))
        or np.ones((len(texts), 1024), dtype=np.float32),
    )
    monkeypatch.setattr(
        ingest, "write_vectors", lambda rows: state["written"].append(rows)
    )
    return state


def test_run_encodes_and_writes_all(fake_flow):
    """正常路径：全部待回填行被编码、被写库，顺序保持 id 升序。"""
    ingest.run(rebuild=False)

    assert fake_flow["encoded"] == [["文本一", "文本二", "文本三"]]
    # write_vectors 收到 [(id, vector), ...]
    written_ids = [cid for cid, _vec in fake_flow["written"][0]]
    assert written_ids == [1, 2, 3]
    assert fake_flow["cleared"] is False  # 非重建模式不清空旧向量


def test_run_rebuild_clears_first(fake_flow):
    """rebuild=True 先清空旧向量再重算（全量重建语义）。"""
    ingest.run(rebuild=True)

    assert fake_flow["cleared"] is True


def test_run_noop_when_nothing_pending(fake_flow):
    """没有待回填行时是优雅的 no-op，不编码不写库。"""
    fake_flow["pending"] = []

    ingest.run(rebuild=False)

    assert fake_flow["encoded"] == []
    assert fake_flow["written"] == []


def test_run_aborts_on_dimension_mismatch(fake_flow, monkeypatch):
    """
    模型输出维度与 vector(1024) 列不符时，写库前就报错终止。

    # 难点：这个自检把失败从"SQL 报 dimensions do not match"提前到
    # "Python 侧报模型配置错"。前者只回滚最后一批、信息含糊；
    # 后者在写任何一行之前发生，且错误信息直接指向 .env 的两个配置项。
    """
    monkeypatch.setattr(
        ingest.embedding, "encode", lambda texts: np.ones((len(texts), 768), dtype=np.float32)
    )

    with pytest.raises(RuntimeError, match="768"):
        ingest.run(rebuild=False)

    # 关键：一条都没写进去
    assert fake_flow["written"] == []


def test_fetch_pending_only_null_when_not_rebuild(raw_conn):
    """
    幂等性验证（连真实库）：默认只取 embedding IS NULL 的行。
    335 条向量已回填，所以非重建模式下取到 0 行 ——
    这正好证明"重跑脚本不会重复编码已完成的部分"。
    """
    pending = ingest.fetch_pending(rebuild=False)

    assert pending == []


def test_fetch_pending_rebuild_takes_all(raw_conn):
    """重建模式取全部行（335 条种子切块）。"""
    pending = ingest.fetch_pending(rebuild=True)

    assert len(pending) == 335
    assert pending[0][0] == 1  # id 升序
