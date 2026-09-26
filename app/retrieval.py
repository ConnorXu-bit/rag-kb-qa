"""检索层：向量检索、BM25 关键词检索、以及两者融合。

为什么需要混合检索：
纯向量检索对"意思相近"很擅长，但对型号、代码标识符、专有名词这类
低频精确词经常漏召（它们的语义邻居很少）。BM25 反过来，精确词一找一个准，
但不理解同义改写。两者融合，互相补短板。

为什么融合只用"名次"不用"分数"（RRF）：
余弦相似度在 [-1, 1]，BM25 是无上界的正数，两个分数量纲完全不同，
直接加权相加没有意义。RRF 只取名次，天然回避了这个问题。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import replace
from typing import Callable, Iterable, Protocol, Sequence

from .chunking import Chunk
from .embeddings import Embedder, _tokenize
from .store import Hit, VectorStore

ASCII_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")


class Retriever(Protocol):
    def search(self, question: str, top_k: int) -> list[Hit]:
        """返回按相关性降序的命中列表，已按各自阈值过滤。"""
        ...


def keyword_tokenize(text: str) -> list[str]:
    """英文按单词、中文按二元组切分。

    中文没有空格，单个汉字信息量太低（"的""是"到处都是），
    二元组（"短链""链接"）既保留了词序，又不需要引入分词库。
    """
    lowered = text.lower()
    tokens = ASCII_TOKEN_RE.findall(lowered)
    for run in CJK_RUN_RE.findall(lowered):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


class BM25Index:
    """Okapi BM25。k1 控制词频饱和，b 控制文档长度归一化的强度。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._term_freqs: list[Counter[str]] = []
        self._doc_freqs: Counter[str] = Counter()
        self._doc_lengths: list[int] = []
        self._avg_length = 0.0

    def fit(self, documents: Sequence[str]) -> "BM25Index":
        self._term_freqs = [Counter(keyword_tokenize(document)) for document in documents]
        self._doc_lengths = [sum(freq.values()) for freq in self._term_freqs]
        self._doc_freqs = Counter()
        for freqs in self._term_freqs:
            self._doc_freqs.update(freqs.keys())
        self._avg_length = (
            sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0.0
        )
        return self

    def _idf(self, term: str) -> float:
        total = len(self._term_freqs)
        doc_freq = self._doc_freqs.get(term, 0)
        return math.log(1 + (total - doc_freq + 0.5) / (doc_freq + 0.5))

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if not self._term_freqs:
            return []
        scores: list[tuple[int, float]] = []
        for index, freqs in enumerate(self._term_freqs):
            length_ratio = self._doc_lengths[index] / self._avg_length if self._avg_length else 0.0
            score = 0.0
            for term in keyword_tokenize(query):
                frequency = freqs.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (1 - self.b + self.b * length_ratio)
                score += self._idf(term) * frequency * (self.k1 + 1) / denominator
            if score > 0:
                scores.append((index, score))
        scores.sort(key=lambda item: (-item[1], item[0]))
        return scores[:top_k]


def hit_key(hit: Hit) -> tuple[str, int]:
    return (hit.source, hit.ordinal)


def reciprocal_rank_fusion(
    rankings: Iterable[Sequence[Hit]],
    k: int = 60,
) -> list[Hit]:
    """RRF：每个结果得分 = Σ 1/(k + 名次)。只看名次，不看分数。"""
    if k <= 0:
        raise ValueError("k 必须大于 0")
    scores: dict[tuple[str, int], float] = {}
    items: dict[tuple[str, int], Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            key = hit_key(hit)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            items.setdefault(key, hit)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    # 不四舍五入：融合后的分数只用来排序，保留原值避免制造并列
    return [replace(items[key], score=score) for key, score in ordered]


class VectorRetriever:
    def __init__(self, embedder: Embedder, store: VectorStore, min_score: float = 0.15) -> None:
        self.embedder = embedder
        self.store = store
        self.min_score = min_score

    def search(self, question: str, top_k: int) -> list[Hit]:
        vector = self.embedder.embed([question])[0]
        hits = self.store.search(vector, top_k=top_k)
        return [hit for hit in hits if hit.score >= self.min_score]


class KeywordRetriever:
    """BM25 关键词检索。语料变了要重新 fit，所以每次检索前重建索引。

    这是当前的取舍：万级以内片段重建索引只要几十毫秒，换来的是不用维护
    增量索引的一致性。语料再大就该换成持久化的倒排索引。
    """

    def __init__(self, store: VectorStore, min_score: float = 0.0) -> None:
        self.store = store
        self.min_score = min_score

    def search(self, question: str, top_k: int) -> list[Hit]:
        chunks = self.store.iter_chunks()
        if not chunks:
            return []
        hits = BM25Index().fit([chunk.text for chunk in chunks]).search(question, top_k)
        return [
            Hit(
                source=chunks[index].source,
                heading=chunks[index].heading,
                ordinal=chunks[index].ordinal,
                text=chunks[index].text,
                score=score,
            )
            for index, score in hits
            if score >= self.min_score
        ]


class HybridRetriever:
    """向量 + 关键词，用 RRF 融合名次。任一检索器没命中都不影响另一个。

    注意 candidate_pool：每个通道先多取 top_k * candidate_pool 条候选，
    融合之后再截回 top_k。如果只让每个通道取 top_k 条，某个通道排在第 k 位
    的结果会在融合前就被丢掉——评估里这种情况会让混合检索的召回率反而
    低于单通道，等于白融合。
    """

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        rrf_k: int = 60,
        candidate_pool: int = 3,
    ) -> None:
        if not retrievers:
            raise ValueError("至少需要一个检索器")
        if candidate_pool <= 0:
            raise ValueError("candidate_pool 必须大于 0")
        self.retrievers = list(retrievers)
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool

    def search(self, question: str, top_k: int) -> list[Hit]:
        pool = top_k * self.candidate_pool
        rankings = [retriever.search(question, pool) for retriever in self.retrievers]
        return reciprocal_rank_fusion(rankings, k=self.rrf_k)[:top_k]


def build_chunk_lookup(chunks: Sequence[Chunk]) -> Callable[[tuple[str, int]], Chunk]:
    """按 (source, ordinal) 反查片段，用于把检索结果映射回原文。"""
    index = {(chunk.source, chunk.ordinal): chunk for chunk in chunks}
    return lambda key: index[key]
