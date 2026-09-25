import math

import pytest

from app.embeddings import FakeEmbedder


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
