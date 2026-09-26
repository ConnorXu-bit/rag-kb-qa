"""评估用到的纯函数：标注的读写、命中判定、指标计算。

指标故意只吃「名次列表」这一种输入：第 i 项是第 i 个问题的命中名次，
没命中就是 None。这样指标和检索实现完全解耦，也才好写单元测试。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol, Sequence


class HitLike(Protocol):
    source: str
    heading: str
    text: str


@dataclass(frozen=True)
class QAPair:
    """一条标注：问题 + 标准答案所在的片段（用 source + heading 定位）。

    anchor 是答案里的一个锚点字符串。只有「命中片段的标题路径对得上、
    且正文里真的含这句锚点」才算召回，避免出现「标题对了但拿错块」的假阳性。
    """

    id: str
    question: str
    source: str
    heading: str
    anchor: str
    tag: str = "semantic"
    note: str = field(default="")

    @property
    def key(self) -> tuple[str, str]:
        return (self.source, self.heading)


def load_pairs(path: str | Path) -> list[QAPair]:
    pairs: list[QAPair] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"第 {number} 行不是合法 JSON：{error}") from error
        pairs.append(
            QAPair(
                id=payload["id"],
                question=payload["question"],
                source=payload["source"],
                heading=payload["heading"],
                anchor=payload["anchor"],
                tag=payload.get("tag", "semantic"),
                note=payload.get("note", ""),
            )
        )
    return pairs


def validate_pairs(pairs: Sequence[QAPair], chunk_index: dict[tuple[str, str], list[str]]) -> list[str]:
    """校验标注本身有没有写错，返回问题列表。

    这一步是评估可信度的前提：如果锚点在对不上的标题路径下，
    说明标注就是错的，必须先修标注，而不是调检索参数。
    """
    problems: list[str] = []
    seen: set[str] = set()
    for pair in pairs:
        if not pair.question.strip():
            problems.append(f"{pair.id}: question 为空")
        if not pair.anchor.strip():
            problems.append(f"{pair.id}: anchor 为空")
        if pair.id in seen:
            problems.append(f"{pair.id}: id 重复")
        seen.add(pair.id)
        texts = chunk_index.get(pair.key)
        if texts is None:
            problems.append(f"{pair.id}: 语料里没有这个标题路径 {pair.key}")
            continue
        if not any(pair.anchor in text for text in texts):
            problems.append(
                f"{pair.id}: 锚点「{pair.anchor}」不在 {pair.source} 的「{pair.heading}」里"
            )
    return problems


def matches(hit: HitLike, pair: QAPair) -> bool:
    """命中要求：同一份文档、同一个标题路径，且正文真的含锚点。"""
    return (
        hit.source == pair.source
        and hit.heading == pair.heading
        and pair.anchor in hit.text
    )


def first_match_rank(hits: Sequence[HitLike], pair: QAPair) -> int | None:
    """第一条正确命中的名次（从 1 开始）；没命中返回 None。"""
    for rank, hit in enumerate(hits, start=1):
        if matches(hit, pair):
            return rank
    return None


def recall_at_k(ranks: Sequence[int | None], k: int) -> float:
    if k <= 0:
        raise ValueError("k 必须大于 0")
    if not ranks:
        return 0.0
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def hit_rate_at_1(ranks: Sequence[int | None]) -> float:
    if not ranks:
        return 0.0
    return sum(1 for rank in ranks if rank == 1) / len(ranks)


def mean_reciprocal_rank(ranks: Sequence[int | None]) -> float:
    if not ranks:
        return 0.0
    return sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / len(ranks)


def summarize(ranks: Sequence[int | None], k: int) -> dict[str, float]:
    return {
        "recall": recall_at_k(ranks, k),
        "hit1": hit_rate_at_1(ranks),
        "mrr": mean_reciprocal_rank(ranks),
    }


def format_table(rows: Iterable[tuple[str, dict[str, float], int]], k: int) -> str:
    """把 (通道名, 指标, 无命中题数) 渲染成 Markdown 表格。"""
    lines = [
        f"| 检索通道 | Recall@{k} | Hit@1 | MRR | 返回 0 条的题数 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, stats, empty in rows:
        lines.append(
            f"| {name} | {stats['recall']:.3f} | {stats['hit1']:.3f} | "
            f"{stats['mrr']:.3f} | {empty} |"
        )
    return "\n".join(lines)
