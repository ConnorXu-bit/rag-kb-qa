
from fastapi import FastAPI

from app.embeddings import FakeEmbedder
from app.llm import FakeLLM
from app.rag import RagService
from app.retrieval import HybridRetriever, KeywordRetriever, VectorRetriever
from app.schemas import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceItem,
)
from app.store import VectorStore


def build_service() -> RagService:
    store = VectorStore()
    embedder = FakeEmbedder(dim=512)
    retrievers = HybridRetriever(
        [VectorRetriever(embedder, store), KeywordRetriever(store)]
    )
    return RagService(
        embedder,
        store,
        FakeLLM(["（示例回答：换成真实模型就有真回答了）"]),
        top_k=4,
        retriever=retrievers,
    )


def create_app(service: RagService | None = None) -> FastAPI:
    app = FastAPI(title="RAG 知识库问答服务", version="0.1.0")
    app.state.service = service or build_service()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "chunks": app.state.service.store.count()}

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(payload: IngestRequest) -> IngestResponse:
        chunks = app.state.service.ingest_text(payload.text, payload.source)
        return IngestResponse(source=payload.source, chunks=chunks)

    @app.post("/query", response_model=QueryResponse)
    def query(payload: QueryRequest) -> QueryResponse:
        result = app.state.service.answer(payload.question)
        sources = [
            SourceItem(
                source=hit.source,
                heading=hit.heading,
                ordinal=hit.ordinal,
                score=hit.score,
                preview=hit.text[:80],
            )
            for hit in result.sources
        ]
        return QueryResponse(
            answer=result.answer, used_llm=result.used_llm, sources=sources
        )

    return app


app = create_app()
