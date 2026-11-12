"""
embedding.py —— 本地 bge-m3 模型的加载与文本向量化

职责：
    1. 进程内只加载一次 bge-m3（单例），谁先用到谁触发，之后的调用直接复用
    2. 对外只暴露一个函数 encode(texts)：一批文本进，一组 1024 维归一化向量出

不负责：
    检索逻辑（rag.py）、向量怎么入库（ingest.py）、配置从哪来（config.py）。

为什么做成懒加载单例而不是启动时就加载：
    bge-m3 有 2.27GB，加载到 GPU 约 15 秒。如果服务一启动就加载，
    那么只问订单的用户也要等这 15 秒；而懒加载把成本推迟到
    "第一个问知识库问题的人"，其余请求完全无感。
    代价是第一次知识库检索会慢十几秒（日志里会打 WARNING 提醒），
    这是服务型部署里常见的取舍。

线程安全说明：
    FastAPI 的同步路由跑在线程池里，多个请求可能同时进来。
    这里用一把简单的锁保证"加载"只发生一次 —— 加载完成后读 _model
    不加锁（CPython 读取属性是原子的，且 SentenceTransformer.encode
    本身可以并发调用，内部有批处理保护）。
"""

import logging
import threading
from typing import TYPE_CHECKING

import numpy as np

from app.config import settings

if TYPE_CHECKING:  # 只在类型检查时导入，运行时不依赖 sentence_transformers（测试环境可替换）
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_model: "SentenceTransformer | None" = None
_load_lock = threading.Lock()


def get_model() -> "SentenceTransformer":
    """
    返回进程内唯一的 bge-m3 模型实例，第一次调用时才真正加载。

    Returns:
        已加载到 EMBEDDING_DEVICE（cuda/cpu）的 SentenceTransformer 实例。

    Raises:
        RuntimeError: 模型路径不存在（.env 里 EMBEDDING_MODEL_PATH 配错），
            或显卡/CPU 不可用导致加载失败。异常会带上下文，方便定位是
            "路径错"还是"环境错"。

    # 难点：为什么加载失败要让整个请求失败（抛异常），而不是退回 CPU 再试？
    # 捕获 CUDA 加载失败自动降级 CPU 看似更健壮，实则掩盖配置问题：
    # 用户在 .env 里写 cuda，实际跑在 CPU 上编码 335 条要几分钟，
    # 检索延迟劣化 10 倍却没有任何报错，排查起来比直接失败难得多。
    # 所以这里选择"快失败 + 日志说清原因"，让运维自己决定要不要降级。
    """
    global _model
    if _model is not None:
        return _model

    with _load_lock:  # 双重检查：等锁的第二个线程进来时，第一个已经加载完
        if _model is not None:
            return _model

        if not settings.embedding_model_path:
            raise RuntimeError(
                "EMBEDDING_MODEL_PATH 未配置：请检查 .env 的 Embedding 段"
            )

        import time

        from sentence_transformers import SentenceTransformer  # 局部导入：不用向量时连库都不用加载

        started = time.time()
        logger.info(
            "加载 bge-m3 模型 | path=%s device=%s（首次调用约需 10~20 秒）",
            settings.embedding_model_path,
            settings.embedding_device,
        )
        try:
            _model = SentenceTransformer(
                settings.embedding_model_path,
                device=settings.embedding_device,
            )
        except Exception as exc:
            raise RuntimeError(
                f"bge-m3 加载失败（path={settings.embedding_model_path}, "
                f"device={settings.embedding_device}）：{exc}"
            ) from exc
        logger.info("bge-m3 加载完成，耗时 %.1fs", time.time() - started)
        return _model


def encode(texts: list[str]) -> np.ndarray:
    """
    把一批文本编码成向量（同步阻塞，跑在 FastAPI 的线程池里）。

    Args:
        texts: 文本列表。空列表直接返回空数组，不触发模型加载。

    Returns:
        形状 (len(texts), 1024) 的 float32 矩阵，每行已 L2 归一化。
        EMBEDDING_NORMALIZE=false 时返回原始向量（仅供实验，检索阈值会失配）。

    Raises:
        RuntimeError: 模型加载失败（见 get_model）。
    """
    if not texts:
        return np.zeros((0, settings.embedding_dim), dtype=np.float32)

    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=settings.embedding_batch_size,
        normalize_embeddings=settings.embedding_normalize,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def encode_one(text: str) -> np.ndarray:
    """
    单条文本便捷入口（检索时用）。语义等价于 encode([text])[0]。

    Args:
        text: 一条文本。

    Returns:
        形状 (1024,) 的一维向量，归一化与否与 encode 保持一致。

    # 难点：为什么单独包一个函数而不是让调用方自己 encode([q])[0]？
    # 两个原因：一是一维和二维数组在 pgvector 适配、余弦计算里的行为不同
    # （二维要 reshape），调用方容易踩坑；二是"单条"这个语义在检索侧
    # 出现频率极高，有个名字可读性明显更好。
    """
    return encode([text])[0]


def reset_model() -> None:
    """
    丢弃已加载的模型实例（测试专用）。

    正常业务代码永远用不到它；单测需要在"装了假模型"和"没装模型"
    两种状态间切换时调用，保证用例之间互不污染。
    """
    global _model
    with _load_lock:
        _model = None
