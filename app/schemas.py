
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    source: str = Field(min_length=1, description="文档标识，例如 notes/ch1.md")
    text: str = Field(min_length=1, description="文档正文")


class IngestResponse(BaseModel):
    source: str
    chunks: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, description="用户问题")


class SourceItem(BaseModel):
    source: str
    heading: str
    ordinal: int
    score: float
    preview: str


class QueryResponse(BaseModel):
    answer: str
    used_llm: bool
    sources: list[SourceItem]
