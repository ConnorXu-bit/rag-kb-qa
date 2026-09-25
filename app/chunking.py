"""把 Markdown 讲义切成适合检索的片段。

两条经验规则：
1. 每个片段带上所属标题路径（"第2章 > 一元二次不等式"），
   否则切碎之后单个片段会失去上下文，检索命中率明显下降。
2. 相邻片段保留一段重叠（overlap），避免关键句子正好被切在边界上。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    """一个待向量化的片段。text 是最终送进 embedding 的字符串。"""

    text: str
    source: str
    heading: str
    ordinal: int


def _split_long(paragraph: str, chunk_size: int) -> list[str]:
    """段落本身超过 chunk_size 时按长度硬切。"""
    return [paragraph[i : i + chunk_size] for i in range(0, len(paragraph), chunk_size)]


def _iter_paragraphs(text: str):
    """按标题和空行切出 (标题路径, 段落正文)。"""
    heading_stack: list[tuple[int, str]] = []
    lines: list[str] = []

    def path() -> str:
        return " > ".join(title for _, title in heading_stack)

    def take() -> list[tuple[str, str]]:
        if not lines:
            return []
        paragraph = "\n".join(lines).strip()
        lines.clear()
        return [(path(), paragraph)] if paragraph else []

    for raw in text.splitlines():
        matched = HEADING_RE.match(raw)
        if matched:
            yield from take()
            level, title = len(matched.group(1)), matched.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            continue
        if not raw.strip():
            yield from take()
            continue
        lines.append(raw.rstrip())
    yield from take()


def split_markdown(
    text: str,
    source: str,
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[Chunk]:
    """切成片段。片段正文长度不超过 chunk_size + overlap。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap 必须满足 0 <= overlap < chunk_size")

    chunks: list[Chunk] = []
    buffer = ""
    buffer_heading = ""
    ordinal = 0

    def flush() -> None:
        nonlocal buffer, ordinal
        body = buffer.strip()
        if body:
            prefix = buffer_heading
            chunks.append(
                Chunk(
                    text=f"{prefix}\n{body}" if prefix else body,
                    source=source,
                    heading=prefix,
                    ordinal=ordinal,
                )
            )
            ordinal += 1
        buffer = ""

    for heading, paragraph in _iter_paragraphs(text):
        for piece in _split_long(paragraph, chunk_size):
            if not buffer:
                buffer_heading = heading
            elif heading != buffer_heading:
                flush()
                buffer_heading = heading
            elif len(buffer) + len(piece) > chunk_size:
                tail = buffer[-overlap:] if overlap else ""
                flush()
                buffer = tail
                buffer_heading = heading
            buffer += ("\n" if buffer else "") + piece

    flush()
    return chunks
