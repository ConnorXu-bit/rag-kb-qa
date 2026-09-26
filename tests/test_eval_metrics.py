from pathlib import Path

import pytest

from eval.metrics import (
    QAPair,
    first_match_rank,
    hit_rate_at_1,
    load_pairs,
    matches,
    mean_reciprocal_rank,
    recall_at_k,
    summarize,
    validate_pairs,
)
from eval.run_eval import build_chunk_index, index_corpus, load_corpus
from app.embeddings import FakeEmbedder
from app.store import Hit

ROOT = Path(__file__).resolve().parents[1]


def make_hit(source="a.md", heading="一 > 二", text="正文里有锚点"):
    return Hit(source=source, heading=heading, ordinal=0, text=text, score=1.0)


def test_matches_requires_same_source_heading_and_anchor():
    pair = QAPair(id="q1", question="问题", source="a.md", heading="一 > 二", anchor="锚点")

    assert matches(make_hit(), pair)
    assert not matches(make_hit(source="b.md"), pair)
    assert not matches(make_hit(heading="一 > 三"), pair)
    assert not matches(make_hit(text="正文里没有那句话"), pair)


def test_first_match_rank_is_one_based_and_none_when_missing():
    pair = QAPair(id="q1", question="问题", source="a.md", heading="一 > 二", anchor="锚点")
    hits = [make_hit(source="x.md"), make_hit(heading="别的"), make_hit()]

    assert first_match_rank(hits, pair) == 3
    assert first_match_rank([make_hit(source="x.md")], pair) is None


def test_recall_counts_hits_within_k():
    ranks = [1, 2, 5, 6, None]

    assert recall_at_k(ranks, k=5) == pytest.approx(3 / 5)
    assert recall_at_k(ranks, k=1) == pytest.approx(1 / 5)
    assert recall_at_k([], k=5) == 0.0


def test_recall_rejects_non_positive_k():
    with pytest.raises(ValueError):
        recall_at_k([1], k=0)


def test_hit_rate_at_1_and_mrr():
    ranks = [1, 1, 2, None]

    assert hit_rate_at_1(ranks) == pytest.approx(2 / 4)
    assert mean_reciprocal_rank(ranks) == pytest.approx((1 + 1 + 0.5 + 0) / 4)
    assert summarize(ranks, k=5) == {
        "recall": pytest.approx(3 / 4),
        "hit1": pytest.approx(2 / 4),
        "mrr": pytest.approx(2.5 / 4),
    }


def test_validate_pairs_reports_missing_heading_and_bad_anchor():
    pairs = [
        QAPair(id="q1", question="问", source="a.md", heading="没有的标题", anchor="锚点"),
        QAPair(id="q2", question="问", source="a.md", heading="一 > 二", anchor="不存在的锚点"),
        QAPair(id="q2", question="问", source="a.md", heading="一 > 二", anchor="锚点"),
    ]

    problems = validate_pairs(pairs, {("a.md", "一 > 二"): ["正文里有锚点"]})

    assert len(problems) == 3
    assert any("没有的标题" in problem for problem in problems)
    assert any("不存在的锚点" in problem for problem in problems)
    assert any("id 重复" in problem for problem in problems)


def test_load_pairs_reads_jsonl(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        '{"id": "q1", "question": "问", "source": "a.md", "heading": "h", "anchor": "a", '
        '"tag": "lexical"}\n',
        encoding="utf-8",
    )

    pairs = load_pairs(path)

    assert len(pairs) == 1
    assert pairs[0].tag == "lexical"


def test_labeled_pairs_validate_against_repo_docs():
    """标注是评估的地基：锚点必须真的落在它声明的标题路径下。"""
    corpus = load_corpus(ROOT / "docs")
    _, chunks = index_corpus(corpus, FakeEmbedder(dim=512))
    pairs = load_pairs(ROOT / "eval" / "qa_pairs.jsonl")

    assert len(pairs) >= 30
    assert validate_pairs(pairs, build_chunk_index(chunks)) == []
