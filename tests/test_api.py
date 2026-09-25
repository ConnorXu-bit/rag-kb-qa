
import pytest
from fastapi.testclient import TestClient

from app.embeddings import FakeEmbedder
from app.llm import FakeLLM
from app.main import create_app
from app.rag import RagService
from app.store import VectorStore

DOC = "# 短链服务\n\n## 短码生成\n\n短码用 secrets 随机生成 6 位，写入时用 SETNX 保证原子性。\n"


@pytest.fixture()
def client():
    store = VectorStore()
    service = RagService(
        embedder=FakeEmbedder(dim=256),
        store=store,
        llm=FakeLLM(["答案 [1]"]),
    )
    with TestClient(create_app(service)) as test_client:
        yield test_client
    store.close()


def test_health_reports_chunk_count(client):
    assert client.get("/health").json() == {"status": "ok", "chunks": 0}


def test_ingest_then_health_counts_chunks(client):
    response = client.post("/ingest", json={"source": "shortlink.md", "text": DOC})

    assert response.status_code == 200
    assert response.json() == {"source": "shortlink.md", "chunks": 1}
    assert client.get("/health").json()["chunks"] == 1


def test_query_returns_answer_and_sources(client):
    client.post("/ingest", json={"source": "shortlink.md", "text": DOC})

    payload = client.post("/query", json={"question": "短码怎么生成的"}).json()

    assert payload["answer"] == "答案 [1]"
    assert payload["used_llm"] is True
    assert payload["sources"][0]["source"] == "shortlink.md"
    assert "短码生成" in payload["sources"][0]["heading"]
    assert len(payload["sources"][0]["preview"]) <= 80


def test_query_without_relevant_content_skips_llm(client):
    client.post("/ingest", json={"source": "shortlink.md", "text": DOC})
    client.app.state.service.retriever.min_score = 1.1

    payload = client.post("/query", json={"question": "今天天气怎么样"}).json()

    assert payload["used_llm"] is False
    assert payload["sources"] == []


def test_empty_question_is_rejected(client):
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_empty_text_is_rejected(client):
    assert client.post("/ingest", json={"source": "a.md", "text": ""}).status_code == 422
