import math
from types import SimpleNamespace

import pytest

from app.embeddings import FakeEmbedder, OpenAIEmbedder


def cosine(left, right):
    return sum(a * b for a, b in zip(left, right))


def test_same_text_always_gets_same_vector():
    embedder = FakeEmbedder(dim=64)

    first = embedder.embed(["短链接生成"])[0]
    second = embedder.embed(["短链接生成"])[0]

    assert first == second
    assert len(first) == 64
    assert math.isclose(cosine(first, first), 1.0, rel_tol=1e-9)


def test_similar_text_scores_higher_than_unrelated_text():
    embedder = FakeEmbedder(dim=256)
    query, similar, unrelated = embedder.embed(
        ["短链接怎么生成", "短链接生成要几步", "今天天气怎么样"]
    )

    assert cosine(query, similar) > cosine(query, unrelated)


def test_empty_text_gets_zero_vector():
    assert FakeEmbedder(dim=8).embed([""])[0] == [0.0] * 8


def test_dimension_must_be_positive():
    with pytest.raises(ValueError):
        FakeEmbedder(dim=0)


class StubEmbeddings:
    """假的 embeddings 端点：记录每次收到的批量，并故意乱序返回。"""

    def __init__(self):
        self.calls: list[list[str]] = []

    def create(self, model, input):
        self.calls.append(list(input))
        items = [
            SimpleNamespace(index=index, embedding=[float(len(text)), 0.0])
            for index, text in enumerate(input)
        ]
        return SimpleNamespace(data=list(reversed(items)))


def test_openai_embedder_splits_batches_and_restores_order():
    embedder = OpenAIEmbedder(api_key="test", batch_size=2)
    stub = StubEmbeddings()
    embedder._client = SimpleNamespace(embeddings=stub)

    vectors = embedder.embed(["a", "bb", "ccc", "dddd", "eeeee"])

    assert [len(call) for call in stub.calls] == [2, 2, 1]
    assert vectors == [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]


def test_openai_embedder_rejects_non_positive_batch_size():
    with pytest.raises(ValueError):
        OpenAIEmbedder(api_key="test", batch_size=0)
