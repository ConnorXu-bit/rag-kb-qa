import pytest

from app.embeddings import FakeEmbedder
from app.llm import FakeLLM
from app.rag import NO_HIT_REPLY, RagService
from app.store import VectorStore

DOC = """# 短链接服务

## 短码生成

短码用 secrets 模块随机生成 6 位，写入时用 SETNX 保证原子性，冲突就重试。

## 为什么不用 KEYS

KEYS 会阻塞 Redis 主线程，遍历大量 key 时其他请求全部排队，改用 SCAN 游标分批遍历。
"""


@pytest.fixture()
def service():
    store = VectorStore()
    service = RagService(embedder=FakeEmbedder(dim=256), store=store, llm=FakeLLM(["答案 [1]"]))
    service.ingest_text(DOC, "shortlink.md")
    yield service
    store.close()


def test_answer_returns_sources_with_numbering(service):
    result = service.answer("短码是怎么生成的")

    assert result.used_llm is True
    assert result.answer == "答案 [1]"
    assert result.sources
    assert "SETNX" in result.sources[0].text
    assert result.sources[0].source == "shortlink.md"


def test_prompt_contains_retrieved_context_and_question(service):
    service.answer("短码是怎么生成的")

    prompt = service.llm.prompts[0]
    assert "[1] 来源：shortlink.md" in prompt
    assert "SETNX" in prompt
    assert "短码是怎么生成的" in prompt


def test_no_hit_returns_fallback_without_calling_llm(service):
    service.retriever.min_score = 1.1

    result = service.answer("今天天气怎么样")

    assert result.answer == NO_HIT_REPLY
    assert result.sources == []
    assert result.used_llm is False
    assert service.llm.prompts == []


def test_retrieve_respects_top_k(service):
    service.top_k = 1

    assert len(service.retrieve("短码生成")) == 1


def test_reingesting_same_source_does_not_duplicate(service):
    before = service.store.count()

    service.ingest_text(DOC, "shortlink.md")

    assert service.store.count() == before


def test_ingest_dir_reads_markdown_files_transitively(tmp_path):
    (tmp_path / "章节").mkdir()
    (tmp_path / "章节" / "a.md").write_text("# A\n\n正文甲。", encoding="utf-8")
    (tmp_path / "b.md").write_text("# B\n\n正文乙。", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("不是 markdown", encoding="utf-8")

    store = VectorStore()
    service = RagService(embedder=FakeEmbedder(dim=64), store=store, llm=FakeLLM())
    count = service.ingest_dir(tmp_path)
    store.close()

    assert count == 2


def test_ingest_dir_rejects_missing_directory(tmp_path):
    store = VectorStore()
    service = RagService(embedder=FakeEmbedder(dim=64), store=store, llm=FakeLLM())

    with pytest.raises(NotADirectoryError):
        service.ingest_dir(tmp_path / "不存在")
    store.close()


def test_top_k_must_be_positive():
    with pytest.raises(ValueError):
        RagService(embedder=FakeEmbedder(dim=8), store=VectorStore(), llm=FakeLLM(), top_k=0)
