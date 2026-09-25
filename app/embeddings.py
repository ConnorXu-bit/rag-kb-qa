"""文本向量化。对外只暴露 embed(texts) -> list[list[float]]。

为什么要做成接口而不是直接调 API：
测试里不可能（也不该）每次都花钱调真实 embedding 接口，
把 Embedder 抽象出来后，测试注入 FakeEmbedder，业务代码一行不用改。
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, Sequence

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


def _tokenize(text: str) -> list[str]:
    """英文按单词、中文按单字切分，够用且不引入分词依赖。"""
    return TOKEN_RE.findall(text.lower())


class FakeEmbedder:
    """词袋哈希向量：确定性、不联网、可解释。

    同一段文本永远得到同一个向量；共享词汇越多，余弦相似度越高。
    所以测试里可以断言"排序正确"，而不是只能断言"返回了一个列表"。
    """

    def __init__(self, dim: int = 64) -> None:
        if dim <= 0:
            raise ValueError("dim 必须大于 0")
        self.dim = dim

    def _hash(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(digest, "big") % self.dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dim
            for token in _tokenize(text):
                vector[self._hash(token)] += 1.0
            norm = math.sqrt(sum(value * value for value in vector))
            vectors.append([value / norm for value in vector] if norm else vector)
        return vectors


class OpenAIEmbedder:
    """真实 embedding：任何 OpenAI 兼容的 /v1/embeddings 接口都能用。"""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self.model, input=list(texts))
        return [item.embedding for item in response.data]
