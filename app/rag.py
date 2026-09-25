"""RAG 主流程：检索 -> 拼上下文 -> 生成 -> 带引用返回。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .chunking import split_markdown
from .embeddings import Embedder
from .llm import LLM
from .retrieval import Retriever, VectorRetriever
from .store import Hit, VectorStore

NO_HIT_REPLY = "知识库中没有找到相关内容。"

PROMPT_TEMPLATE = """你是知识库问答助手。请严格依据下面的资料回答，不要编造资料之外的内容。
如果资料不足以回答，就直接说明资料里没有相关信息。
回答时用 [1] [2] 这样的编号标注依据，编号必须对应下面的资料。

资料：
{context}

问题：{question}
"""


@dataclass(frozen=True)
class Answer:
    answer: str
    sources: list[Hit]
    used_llm: bool


class RagService:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: LLM,
        top_k: int = 4,
        min_score: float = 0.15,
        retriever: Retriever | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score
        self.retriever: Retriever = retriever or VectorRetriever(embedder, store, min_score)

    def ingest_text(self, text: str, source: str) -> int:
        """入库一份文档。同一 source 重复导入会先清旧块，保证幂等。"""
        chunks = split_markdown(text, source)
        self.store.delete_source(source)
        if not chunks:
            return 0
        return self.store.add(chunks, self.embedder.embed([chunk.text for chunk in chunks]))

    def ingest_dir(self, directory: str | Path) -> int:
        root = Path(directory)
        if not root.is_dir():
            raise NotADirectoryError(f"知识库目录不存在：{root}")
        total = 0
        for path in sorted(root.rglob("*.md")):
            total += self.ingest_text(path.read_text(encoding="utf-8"), str(path.relative_to(root)))
        return total

    def retrieve(self, question: str) -> list[Hit]:
        return self.retriever.search(question, self.top_k)

    @staticmethod
    def build_context(hits: Sequence[Hit]) -> str:
        return "\n\n".join(
            f"[{index}] 来源：{hit.source}（{hit.heading}）\n{hit.text}"
            for index, hit in enumerate(hits, start=1)
        )

    def answer(self, question: str) -> Answer:
        hits = self.retrieve(question)
        if not hits:
            return Answer(answer=NO_HIT_REPLY, sources=[], used_llm=False)
        prompt = PROMPT_TEMPLATE.format(context=self.build_context(hits), question=question)
        return Answer(answer=self.llm.chat(prompt), sources=hits, used_llm=True)
