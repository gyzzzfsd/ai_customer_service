"""
ingest.py —— 向 document_chunks 回填 embedding 的命令行脚本

职责：
    把 335 个知识切块的文本用 bge-m3 编码成 1024 维向量，写回
    document_chunks.embedding 列。这是阶段③ 的一次性入库动作，
    之后每次改动手册内容都可以重跑。

用法（Windows，项目根目录）：
    D:\\app\\anaconda\\envs\\llm1\\python.exe -m app.ingest
    加 --rebuild 参数强制全量重算（默认只补 embedding IS NULL 的行，
    与 .env 的 EMBEDDING_REBUILD=true 等效，命令行参数优先）。

不负责：
    切块本身（document/build_seed.py 在种子期已完成，本脚本只消费
    已有的行）、在线检索（rag.py）。

幂等性设计：
    默认"只补 NULL"使脚本可以放心重跑 —— 中途断掉、失败若干行，
    下次运行自动接着补，已完成的行不会被重复编码（省 GPU 也省时间）。
    全量重建模式则先全部置 NULL 再重算，适合手册内容更新后重刷。
"""

import logging
import sys

import numpy as np

from app import embedding
from app.config import settings
from app.db import close_pool, get_conn, init_pool
from app.log_setup import setup_logging

logger = logging.getLogger(__name__)

# 一次从库里取多少行做一批编码。值与 EMBEDDING_BATCH_SIZE 解耦：
# 编码批大小影响 GPU 上的吞吐，取行批大小影响内存占用和事务粒度，
# 两者没有理由绑在一起。335 条总量很小，64 是安全值。
_FETCH_BATCH = 64


def fetch_pending(rebuild: bool) -> list[tuple[int, str]]:
    """
    取出需要编码的切块（id, content）。

    Args:
        rebuild: True 取全部行（之后会先清空旧向量），False 只取
            embedding IS NULL 的行。

    Returns:
        [(chunk_id, text), ...] 列表，按 id 升序，方便日志和断点续跑。

    Raises:
        psycopg 异常：数据库连不上/表不存在。命令行脚本直接让异常
        冒出去打印堆栈，不需要包装成友好的错误对象。
    """
    sql = "SELECT id, content FROM document_chunks ORDER BY id"
    if not rebuild:
        sql = sql.replace("ORDER BY id", "WHERE embedding IS NULL ORDER BY id")
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)
        return [(r[0], r[1]) for r in cur.fetchall()]


def clear_vectors() -> int:
    """全量重建前清空所有 embedding，返回受影响行数。"""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("UPDATE document_chunks SET embedding = NULL")
        return cur.rowcount


def write_vectors(rows: list[tuple[int, np.ndarray]]) -> None:
    """
    把编码结果写回数据库。

    Args:
        rows: [(chunk_id, vector), ...]，vector 是 1024 维归一化向量。

    # 难点：为什么要按批 commit 而不是 335 条一个事务？
    # 一个事务意味着"任何一行写失败就全部回滚"，听起来更安全，实则不然：
    # 回填是最典型的幂等任务（重跑只补 NULL），分批提交后，前面已提交的
    # 批次在下一次运行时自动跳过，失败损失被限制在最后一个批次里。
    # 一个大事务还会在写期间持有行锁、撑大 WAL，对在线库不友好。
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.executemany(
            "UPDATE document_chunks SET embedding = %s WHERE id = %s",
            [(vec, cid) for cid, vec in rows],
        )
    # get_conn 正常退出即 commit（见 db.py 的说明）


def run(rebuild: bool) -> None:
    """
    入库主流程：统计 → 编码 → 写库 → 汇报。编码由 embedding.encode 批量完成。

    Args:
        rebuild: 是否全量重算（含清空旧向量）。

    Raises:
        RuntimeError: 模型加载失败时由 embedding 抛出，直接终止脚本 ——
        入库失败必须显式失败，绝不能"跳过向量继续跑"（那会让检索
        永远空召回且无人知晓）。
    """
    if rebuild:
        cleared = clear_vectors()
        logger.info("全量重建：已清空 %d 条旧向量", cleared)

    pending = fetch_pending(rebuild)
    if not pending:
        logger.info("没有待回填的切块（embedding 已全部就绪），无事可做")
        return
    logger.info("待回填切块：%d 条，开始编码", len(pending))

    total_written = 0
    for start in range(0, len(pending), _FETCH_BATCH):
        batch = pending[start : start + _FETCH_BATCH]
        ids = [cid for cid, _ in batch]
        texts = [text for _, text in batch]

        vectors = embedding.encode(texts)

        # 难点：维度自检。若模型被换成非 1024 维的（比如调成了 bge-large，
        # 768 维），PG 的 vector(1024) 列会在写入时报错，但那时已经编完
        # 一整批、报错信息只有 SQL 侧一句"dimensions do not match"。
        # 在写库前主动比对，能把错误定位到"模型配置"而不是"数据库"。
        if vectors.shape[1] != settings.embedding_dim:
            raise RuntimeError(
                f"模型输出 {vectors.shape[1]} 维，与库表 vector({settings.embedding_dim}) 不符，"
                f"请检查 EMBEDDING_MODEL_PATH / EMBEDDING_DIM 是否配套"
            )

        write_vectors(list(zip(ids, vectors)))
        total_written += len(batch)
        logger.info("已写入 %d / %d 条", total_written, len(pending))

    logger.info("回填完成：共写入 %d 条", total_written)


def main() -> int:
    """命令行入口。返回退出码（0 成功，1 失败），供脚本调用方判断。"""
    setup_logging()
    # 命令行参数优先于 .env：--rebuild 显式要求全量时覆盖 EMBEDDING_REBUILD
    rebuild = "--rebuild" in sys.argv or settings.embedding_rebuild

    init_pool()
    try:
        run(rebuild)
        return 0
    except Exception:
        logger.exception("向量回填失败")
        return 1
    finally:
        close_pool()


if __name__ == "__main__":
    sys.exit(main())
