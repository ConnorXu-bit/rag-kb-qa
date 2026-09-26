import pytest

from app.chunking import Chunk
from app.embeddings import FakeEmbedder
from app.retrieval import (
    BM25Index,
    HybridRetriever,
    KeywordRetriever,
    VectorRetriever,
    keyword_tokenize,
    reciprocal_rank_fusion,
)
from app.store import Hit, VectorStore

DOCS = [
    "短链服务的短码使用 SETNX 原子写入，避免并发覆盖。",
    "查询短链时用 SCAN 游标分批遍历，避免阻塞 Redis。",
    "天气晴朗适合散步。",
]


def test_keyword_tokenize_splits_ascii_words_and_chinese_bigrams():
    tokens = keyword_tokenize("Redis SETNX 短链")

    assert "redis" in tokens
    assert "setnx" in tokens
    assert "短链" in tokens
    assert "短" not in tokens


def test_keyword_tokenize_keeps_single_chinese_character():
    assert keyword_tokenize("好") == ["好"]


def test_bm25_finds_document_with_rare_term():
    index = BM25Index().fit(DOCS)

    hits = index.search("SETNX", top_k=2)

    assert hits[0][0] == 0
    assert hits[0][1] > 0


def test_bm25_returns_nothing_when_no_term_matches():
    assert BM25Index().fit(DOCS).search("完全不相干的词", top_k=3) == []


def test_bm25_on_empty_index_returns_empty():
    assert BM25Index().fit([]).search("任意", top_k=3) == []


def test_bm25_top_k_must_be_positive():
    with pytest.raises(ValueError):
        BM25Index().fit(DOCS).search("短链", top_k=0)


def hit(source, ordinal, score=0.0):
    return Hit(source=source, heading="小节", ordinal=ordinal, text="正文", score=score)


def test_rrf_ranks_by_rank_not_by_score():
    vector_ranking = [hit("a.md", 0, score=0.99), hit("b.md", 0, score=0.98)]
    keyword_ranking = [hit("b.md", 0, score=12.5), hit("a.md", 0, score=11.0)]

    fused = reciprocal_rank_fusion([vector_ranking, keyword_ranking])

    assert [item.source for item in fused] == ["a.md", "b.md"]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_deduplicates_same_chunk_from_both_retrievers():
    ranking = [hit("a.md", 0)]

    fused = reciprocal_rank_fusion([ranking, ranking])

    assert len(fused) == 1
    assert fused[0].score == pytest.approx(2 / 61)


def test_rrf_rejects_bad_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[hit("a.md", 0)]], k=0)


def build_store():
    store = VectorStore()
    store.add(
        [Chunk(text=text, source="doc.md", heading="小节", ordinal=index) for index, text in enumerate(DOCS)],
        FakeEmbedder(dim=128).embed(DOCS),
    )
    return store


def test_vector_retriever_filters_by_min_score():
    store = build_store()
    retriever = VectorRetriever(FakeEmbedder(dim=128), store, min_score=1.1)

    assert retriever.search("SETNX", top_k=3) == []
    store.close()


def test_keyword_retriever_returns_hits_with_text():
    store = build_store()

    hits = KeywordRetriever(store).search("SETNX 原子写入", top_k=1)
    store.close()

    assert len(hits) == 1
    assert "SETNX" in hits[0].text
    assert hits[0].source == "doc.md"


def test_keyword_retriever_handles_empty_store():
    store = VectorStore()

    assert KeywordRetriever(store).search("任意问题", top_k=3) == []
    store.close()


def test_hybrid_retriever_merges_results_from_both_channels():
    store = build_store()
    hybrid = HybridRetriever(
        [VectorRetriever(FakeEmbedder(dim=128), store), KeywordRetriever(store)]
    )

    hits = hybrid.search("SETNX 原子写入", top_k=3)
    store.close()

    assert hits[0].source == "doc.md"
    assert len({(item.source, item.ordinal) for item in hits}) == len(hits)


def test_hybrid_retriever_requires_at_least_one_channel():
    with pytest.raises(ValueError):
        HybridRetriever([])


def test_hybrid_retriever_survives_one_empty_channel():
    store = build_store()
    hybrid = HybridRetriever(
        [VectorRetriever(FakeEmbedder(dim=128), store, min_score=1.1), KeywordRetriever(store)]
    )

    hits = hybrid.search("SETNX", top_k=2)
    store.close()

    assert len(hits) == 1
    assert "SETNX" in hits[0].text


class RecordingRetriever:
    """记录每次被要求返回多少条，用来验证候选池是否真的扩大了。"""

    def __init__(self, hits):
        self.hits = list(hits)
        self.requested: list[int] = []

    def search(self, question, top_k):
        self.requested.append(top_k)
        return self.hits[:top_k]


def make_hits(prefix: str, count: int) -> list[Hit]:
    return [
        Hit(source=f"{prefix}.md", heading=prefix, ordinal=index, text=f"{prefix}{index}", score=1.0)
        for index in range(count)
    ]


def test_hybrid_retriever_expands_candidate_pool_before_fusion():
    first = RecordingRetriever(make_hits("a", 10))
    second = RecordingRetriever(make_hits("b", 10))

    HybridRetriever([first, second], candidate_pool=3).search("问题", top_k=2)

    assert first.requested == [6]
    assert second.requested == [6]


def test_hybrid_retriever_rejects_non_positive_candidate_pool():
    with pytest.raises(ValueError):
        HybridRetriever([RecordingRetriever([])], candidate_pool=0)


def test_candidate_pool_rescues_a_hit_both_channels_ranked_low():
    """目标在两个通道里都排在尾部：不扩池会在融合前被截掉，扩池才能捞回来。"""
    target = Hit(source="g.md", heading="目标", ordinal=0, text="目标片段", score=1.0)
    first = RecordingRetriever(make_hits("a", 2) + [target])
    second = RecordingRetriever(make_hits("b", 5) + [target])

    without_pool = HybridRetriever([first, second], candidate_pool=1).search("问题", top_k=3)
    with_pool = HybridRetriever([first, second], candidate_pool=3).search("问题", top_k=3)

    def keys(hits):
        return [(hit.source, hit.ordinal) for hit in hits]

    assert ("g.md", 0) not in keys(without_pool)
    assert keys(with_pool)[0] == ("g.md", 0)
