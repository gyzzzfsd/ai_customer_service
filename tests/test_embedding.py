"""
test_embedding.py —— 向量化模块的单元测试

策略：
    真模型（2.27GB）绝不在测试里加载（conftest 有哨兵拦截），
    这里注入一个假 SentenceTransformer，专测：
      - 空列表不触发模型加载
      - 归一化/批大小参数确实传给了模型
      - 单例缓存与 reset
      - 维度/路径配置错误的报错行为
"""

import numpy as np
import pytest

from app import embedding
from app.config import settings


class FakeModel:
    """假 bge-m3：记录每次 encode 的入参，返回固定形状的向量。"""

    def __init__(self, dim=1024):
        self.dim = dim
        self.calls: list[dict] = []

    def encode(self, texts, batch_size=None, normalize_embeddings=None, show_progress_bar=None):
        self.calls.append(
            {
                "texts": list(texts),
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
            }
        )
        return np.ones((len(texts), self.dim), dtype=np.float32)


@pytest.fixture
def fake_model():
    """注入假模型（覆盖 conftest 的哨兵），并保证用例结束后还原。"""
    m = FakeModel()
    embedding._model = m
    yield m
    embedding._model = None


def test_encode_empty_list_skips_model():
    """空列表直接返回 (0, dim)，不触发模型加载 —— conftest 哨兵在，没报错就说明没碰模型。"""
    out = embedding.encode([])

    assert out.shape == (0, settings.embedding_dim)
    assert out.dtype == np.float32


def test_encode_passes_normalization_and_batch(fake_model):
    """
    .env 的 EMBEDDING_NORMALIZE / EMBEDDING_BATCH_SIZE 必须真正传到模型。

    # 难点：归一化不是一个可有可无的开关 —— <=> 是"余弦距离"，
    # 只有归一化向量之间的 <=> 才等于余弦距离；不归一化的向量算出来的
    # 距离没有那个几何意义，阈值 0.5 的标定全部失效。
    # 入库和检索共用本函数，所以两边天然一致 —— 这也是
    # "不单独写一个检索侧 encode"的原因。
    """
    embedding.encode(["问题一", "问题二"])

    assert len(fake_model.calls) == 1
    call = fake_model.calls[0]
    assert call["texts"] == ["问题一", "问题二"]
    assert call["normalize_embeddings"] is settings.embedding_normalize
    assert call["batch_size"] == settings.embedding_batch_size


def test_encode_one_returns_single_vector(fake_model):
    """encode_one 返回一维向量（形状 (dim,)），供 pgvector 直接使用。"""
    vec = embedding.encode_one("单个问题")

    assert vec.shape == (settings.embedding_dim,)
    assert vec.dtype == np.float32


def test_get_model_returns_singleton(fake_model):
    """模型只加载一次：get_model 两次调用拿到同一个实例。"""
    assert embedding.get_model() is fake_model
    assert embedding.get_model() is fake_model


def test_get_model_raises_when_path_missing(monkeypatch):
    """EMBEDDING_MODEL_PATH 未配置时快失败，错误信息指向 .env。"""
    monkeypatch.setattr(embedding, "_model", None)
    monkeypatch.setattr(settings, "embedding_model_path", "")

    with pytest.raises(RuntimeError, match="EMBEDDING_MODEL_PATH"):
        embedding.get_model()


def test_reset_model_clears_singleton(fake_model):
    """reset_model 后 get_model 会重新走加载路径（测试间互不污染的保障）。"""
    embedding.reset_model()

    assert embedding._model is None
