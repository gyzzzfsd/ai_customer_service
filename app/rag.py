"""
rag.py —— 知识库的向量检索

职责：
    把"用户的一句问题"变成"最相关的几段手册内容"：
      1. 问题向量化（embedding.encode_one）
      2. 到 document_chunks 里做余弦近邻检索（top_k）
      3. 相似度阈值过滤，宁可丢弃也不返回不相关的段落
    向量入库（回填 embedding）不在本文件，在 ingest.py —— 入库是一次性
    运维动作，检索是每次问答都要跑的在线路径，两者的失败处理策略完全不同
    （入库失败要立刻停下，检索失败要让模型知道"没查到"）。

不负责：
    向量化本身（embedding.py）、怎么把结果交给模型（tools.py 的
    search_knowledge 工具）、切块规则（document/build_seed.py 已在种子期完成）。

阈值的设计依据（script/RAG设计方案.md 5.2）：
    余弦距离 distance ≤ 0.5 ⇔ 相似度 score ≥ 0.5（因为 score = 1 - distance），
    两种写法等价。.env 的 RAG_SCORE_THRESHOLD 语义是"相似度下限"。
"""

import logging

import numpy as np
from psycopg.rows import dict_row

from app import embedding
from app.config import settings
from app.db import get_conn

logger = logging.getLogger(__name__)

# 检索 SQL：一次往返完成"按相似度排序 + 取前 k 条"。
# 难点（必须用 <=> 算符）：document_chunks 上的 HNSW 索引是按
# vector_cosine_ops 建的，只有查询算符同样是 <=> 时索引才会被使用；
# 换成 <->（欧氏距离）或 <#>（负内积），PG 会静默退化为全表扫描 ——
# 335 行时无感，库长到几十万行时会慢几个数量级，且没有任何报错提示。
_RETRIEVE_SQL = """
SELECT d.title AS title,
       c.content AS content,
       1 - (c.embedding <=> %(qvec)s) AS score
FROM document_chunks c
JOIN knowledge_documents d ON d.id = c.document_id
WHERE d.status = 'ready'
ORDER BY c.embedding <=> %(qvec)s
LIMIT %(top_k)s
"""


def search_chunks(query: str, top_k: int | None = None) -> list[dict]:
    """
    在知识库里检索与 query 最相关的段落。

    Args:
        query: 用户问题（或模型提炼后的检索词）。空串/纯空白返回空列表。
        top_k: 取几条，None 时用 .env 的 RAG_TOP_K（默认 3）。

    Returns:
        按 score 降序的字典列表，每项形如
        {"title": "小米洗衣机A3操作手册", "content": "……", "score": 0.82}。
        score 是余弦相似度（1=完全相同），已低于 RAG_SCORE_THRESHOLD 的结果
        会被丢弃 —— **返回空列表是正常业务结果**，含义是"知识库确实没有
        相关内容"，调用方（tools.search_knowledge）据此让模型如实告知，
        而不是把这里当成错误。

    Raises:
        RuntimeError: 向量化失败（模型没配好）。数据库异常由调用方
        （execute_tool 的兜底）转成 ToolResult.fail，这里不吞。

    # 难点：为什么把阈值过滤放在 SQL 外面而不是 WHERE 里？
    # 阈值过滤要求"先算 score 再比较"，等价于 WHERE 1-(embedding<=>q) >= 0.5。
    # pgvector 的 HNSW 索引只优化 ORDER BY ... LIMIT 的近邻搜索，
    # WHERE 里出现算符表达式不影响索引使用，写在哪边功能都对。
    # 放 Python 侧的真实原因是日志：过滤前后各几条、最低分多少，
    # 是联调阈值时最需要看的信息，放在 SQL 里就看不到了。
    """
    query = (query or "").strip()
    if not query:
        return []

    k = top_k if top_k is not None else settings.rag_top_k
    qvec: np.ndarray = embedding.encode_one(query)

    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_RETRIEVE_SQL, {"qvec": qvec, "top_k": k})
        rows = cur.fetchall()

    hits = [r for r in rows if float(r["score"]) >= settings.rag_score_threshold]
    if rows:
        logger.info(
            "知识库检索 | query=%s 召回=%d 过阈值=%d 最高分=%.3f",
            query[:40],
            len(rows),
            len(hits),
            float(rows[0]["score"]),
        )
    else:
        # 一条都没有 → 大概率 embedding 还没回填（阶段③ 之前的状态），
        # 这条 WARNING 就是为这种情况留的定位线索
        logger.warning("知识库检索空召回 | query=%s（embedding 是否已回填？）", query[:40])
    return hits
