import pytest

from app.chunking import Chunk
from app.embeddings import FakeEmbedder
from app.store import VectorStore


@pytest.fixture()
def store():
    store = VectorStore()
    yield store
    store.close()


def make_chunks(texts):
    return [
        Chunk(text=text, source="doc.md", heading="小节", ordinal=index)
        for index, text in enumerate(texts)
    ]


def ingest(store, embedder, texts):
    chunks = make_chunks(texts)
    store.add(chunks, embedder.embed([chunk.text for chunk in chunks]))
    return chunks


def test_search_returns_most_similar_chunk_first(store):
    embedder = FakeEmbedder(dim=256)
    ingest(
        store,
        embedder,
        ["短链接生成的方式有两种，随机码和自增转码。", "今天天气不错，适合出门散步。"],
    )

    hits = store.search(embedder.embed(["短链接生成"])[0], top_k=2)

    assert [hit.ordinal for hit in hits] == [0, 1]
    assert hits[0].score > hits[1].score


def test_top_k_limits_result_count(store):
    embedder = FakeEmbedder(dim=64)
    ingest(store, embedder, [f"第{index}段正文" for index in range(5)])

    assert len(store.search(embedder.embed(["正文"])[0], top_k=3)) == 3


def test_search_on_empty_store_returns_empty_list(store):
    assert store.search([0.0] * 64, top_k=3) == []


def test_delete_source_removes_only_that_document(store):
    embedder = FakeEmbedder(dim=64)
    store.add(make_chunks(["甲文"]), embedder.embed(["甲文"]))
    store.add(
        [Chunk(text="乙文", source="other.md", heading="小节", ordinal=0)],
        embedder.embed(["乙文"]),
    )

    assert store.delete_source("doc.md") == 1
    assert store.count() == 1
    assert store.search(embedder.embed(["乙文"])[0], top_k=1)[0].text == "乙文"


def test_vector_dimension_mismatch_is_reported(store):
    store.add(make_chunks(["甲文"]), FakeEmbedder(dim=64).embed(["甲文"]))

    with pytest.raises(ValueError, match="维度"):
        store.search([0.0] * 32, top_k=1)


def test_chunk_and_vector_count_must_match(store):
    with pytest.raises(ValueError, match="数量必须一致"):
        store.add(make_chunks(["甲文", "乙文"]), [[1.0, 0.0]])


def test_top_k_must_be_positive(store):
    with pytest.raises(ValueError):
        store.search([0.0] * 64, top_k=0)
