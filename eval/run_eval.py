"""在标注问答对上评估检索质量。

用法（在项目根目录跑）：
    python3 -m eval.run_eval                      # 默认语料 docs/，k=5
    python3 -m eval.run_eval --chunk-size 300     # 换切分参数重跑
    python3 -m eval.run_eval --sweep              # 切分参数扫描
    python3 -m eval.run_eval --embedder openai    # 换真实向量（需要 OPENAI_API_KEY）

为什么要有这个脚本：
切分参数、min_score 阈值、要不要上混合检索，这些选择都只能靠数据回答。
没有评估集的时候，调参就是在凭感觉改代码。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from app.chunking import Chunk, split_markdown
from app.embeddings import Embedder, FakeEmbedder
from app.retrieval import HybridRetriever, KeywordRetriever, Retriever, VectorRetriever
from app.store import VectorStore
from eval.metrics import (
    QAPair,
    first_match_rank,
    format_table,
    load_pairs,
    summarize,
    validate_pairs,
)

PRODUCTION = "hybrid RRF（生产默认）"


def load_corpus(root: Path) -> list[tuple[str, str]]:
    files = sorted(root.rglob("*.md"))
    if not files:
        raise FileNotFoundError(f"语料目录里没有 Markdown：{root}")
    return [(path.relative_to(root).as_posix(), path.read_text(encoding="utf-8")) for path in files]


def index_corpus(
    corpus: Sequence[tuple[str, str]],
    embedder: Embedder,
    chunk_size: int = 500,
    overlap: int = 80,
) -> tuple[VectorStore, list[Chunk]]:
    store = VectorStore()
    chunks: list[Chunk] = []
    for source, text in corpus:
        pieces = split_markdown(text, source, chunk_size=chunk_size, overlap=overlap)
        if pieces:
            store.add(pieces, embedder.embed([piece.text for piece in pieces]))
            chunks.extend(pieces)
    return store, chunks


def build_chunk_index(chunks: Sequence[Chunk]) -> dict[tuple[str, str], list[str]]:
    index: dict[tuple[str, str], list[str]] = {}
    for chunk in chunks:
        index.setdefault((chunk.source, chunk.heading), []).append(chunk.text)
    return index


def build_channels(
    store: VectorStore, embedder: Embedder, min_score: float = 0.15
) -> dict[str, Retriever]:
    """生产配置、两条单通道，外加候选池与阈值的对照。"""
    vector_default = VectorRetriever(embedder, store, min_score=min_score)
    keyword = KeywordRetriever(store, min_score=0.0)
    return {
        PRODUCTION: HybridRetriever([vector_default, keyword]),
        "hybrid（不做候选池扩展）": HybridRetriever(
            [vector_default, keyword], candidate_pool=1
        ),
        "hybrid（候选池 5×）": HybridRetriever([vector_default, keyword], candidate_pool=5),
        "vector 单通道": VectorRetriever(embedder, store, min_score=min_score),
        "BM25 单通道": KeywordRetriever(store, min_score=0.0),
    }


def evaluate_channel(
    pairs: Sequence[QAPair], retriever: Retriever, k: int
) -> tuple[list[int | None], list[str], int]:
    """返回（每条问题的命中名次, 未命中问题的 id, 返回 0 条的题数）。"""
    ranks: list[int | None] = []
    misses: list[str] = []
    empty = 0
    for pair in pairs:
        hits = retriever.search(pair.question, k)
        if not hits:
            empty += 1
        rank = first_match_rank(hits, pair)
        ranks.append(rank)
        if rank is None:
            misses.append(pair.id)
    return ranks, misses, empty


def make_embedder(
    name: str,
    dim: int,
    model: str | None = None,
    base_url: str | None = None,
    batch_size: int = 64,
) -> Embedder:
    if name == "fake":
        return FakeEmbedder(dim=dim)
    from app.embeddings import OpenAIEmbedder

    options = {"batch_size": batch_size}
    if model:
        options["model"] = model
    if base_url:
        options["base_url"] = base_url
    return OpenAIEmbedder(**options)


def render_report(
    results: dict[int, tuple[int, list[tuple[str, dict[str, float], int]], dict[str, list[int | None]]]],
    pairs: Sequence[QAPair],
    k: int,
    corpus: Sequence[tuple[str, str]],
    embedder_label: str,
    min_score: float,
) -> str:
    total_chunks = next(iter(results.values()))[0]
    lines = [
        "# 检索质量评估报告",
        "",
        f"- 语料：{len(corpus)} 篇文档；默认切分下 {total_chunks} 个片段",
        f"- 标注：{len(pairs)} 条问答对（其中语义改写类 "
        f"{sum(1 for pair in pairs if pair.tag != 'lexical')} 条）",
        f"- 向量模型：{embedder_label}",
        f"- 向量阈值：min_score={min_score}",
        f"- 指标口径：命中要求片段标题路径正确且正文含锚点，k={k}",
        "",
    ]
    for chunk_size, (chunk_count, rows, _) in results.items():
        lines += [f"## chunk_size={chunk_size}（{chunk_count} 个片段）", "", format_table(rows, k), ""]

    default_size = next(iter(results))
    default_rows, default_ranks = results[default_size][1], results[default_size][2]
    lines += [f"## 分类型对比（chunk_size={default_size}）", ""]
    lines.append("| 检索通道 | 类型 | 条数 | Recall@1 | Recall@k | MRR |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for name in default_ranks:
        for tag in ("lexical", "semantic"):
            subset = [pair for pair in pairs if pair.tag == tag]
            sizes = [pair.id for pair in subset]
            ranks = [
                rank
                for pair, rank in zip(pairs, default_ranks[name])
                if pair.id in set(sizes)
            ]
            stats = summarize(ranks, k)
            at1 = sum(1 for rank in ranks if rank == 1) / len(ranks) if ranks else 0.0
            lines.append(
                f"| {name} | {tag} | {len(ranks)} | {at1:.3f} | {stats['recall']:.3f} | {stats['mrr']:.3f} |"
            )

    lines += ["", "## 未命中的问题", ""]
    for name, channel_ranks in default_ranks.items():
        missed = [pair.id for pair, rank in zip(pairs, channel_ranks) if rank is None]
        lines.append(f"- {name}：{len(missed)} 条" + (f"（{'、'.join(missed)}）" if missed else ""))
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="在标注问答对上评估检索质量")
    parser.add_argument("--corpus", default="docs", help="语料目录（默认 docs）")
    parser.add_argument("--pairs", default="eval/qa_pairs.jsonl", help="标注文件")
    parser.add_argument("-k", type=int, default=5, help="Recall@k 的 k（默认 5）")
    parser.add_argument("--chunk-size", type=int, default=500, help="切分块大小（默认 500）")
    parser.add_argument("--overlap", type=int, default=80, help="切分重叠（默认 80）")
    parser.add_argument("--sweep", action="store_true", help="扫描多组切分参数")
    parser.add_argument("--dim", type=int, default=512, help="FakeEmbedder 维度")
    parser.add_argument("--embedder", choices=("fake", "openai"), default="fake")
    parser.add_argument(
        "--embedding-model", default="text-embedding-3-small", help="真实 embedding 的模型名"
    )
    parser.add_argument("--base-url", default=None, help="OpenAI 兼容接口的 base_url")
    parser.add_argument("--min-score", type=float, default=0.15, help="向量检索的相似度下限")
    parser.add_argument("--embedding-batch-size", type=int, default=64, help="每次请求提交的条数")
    parser.add_argument("--out", default="eval/report.md", help="报告输出路径")
    parser.add_argument("--verbose", action="store_true", help="打印每条问题的命中名次")
    args = parser.parse_args(argv)

    corpus = load_corpus(Path(args.corpus))
    pairs = load_pairs(args.pairs)
    embedder = make_embedder(
        args.embedder,
        args.dim,
        args.embedding_model,
        args.base_url,
        args.embedding_batch_size,
    )
    embedder_label = (
        f"fake（dim={args.dim}）"
        if args.embedder == "fake"
        else args.embedding_model
        + (f" @ {args.base_url}" if args.base_url else "")
        + f"（batch={args.embedding_batch_size}）"
    )
    sizes = (200, 300, 500, 800) if args.sweep else (args.chunk_size,)

    results = {}
    for chunk_size in sizes:
        store, chunks = index_corpus(corpus, embedder, chunk_size, args.overlap)
        problems = validate_pairs(pairs, build_chunk_index(chunks))
        if problems:
            print(f"chunk_size={chunk_size} 标注校验不通过，先修标注再谈指标：")
            for problem in problems:
                print(f"  - {problem}")
            store.close()
            return 2
        rows: list[tuple[str, dict[str, float], int]] = []
        ranks: dict[str, list[int | None]] = {}
        for name, retriever in build_channels(store, embedder, args.min_score).items():
            channel_ranks, _, empty = evaluate_channel(pairs, retriever, args.k)
            ranks[name] = channel_ranks
            rows.append((name, summarize(channel_ranks, args.k), empty))
        results[chunk_size] = (len(chunks), rows, ranks)
        store.close()

    print(f"语料 {len(corpus)} 篇，标注 {len(pairs)} 条，k={args.k}")
    for chunk_size, (chunk_count, rows, _) in results.items():
        print(f"\nchunk_size={chunk_size}（{chunk_count} 个片段）")
        print(format_table(rows, args.k))

    default_size = next(iter(results))
    default_ranks = results[default_size][2]
    if args.verbose:
        print(f"\n生产通道逐题命中名次（chunk_size={default_size}）：")
        for pair, rank in zip(pairs, default_ranks[PRODUCTION]):
            print(f"  {pair.id}  {'未命中' if rank is None else '第 ' + str(rank) + ' 名'}  {pair.question}")
    missed = [pair.id for pair, rank in zip(pairs, default_ranks[PRODUCTION]) if rank is None]
    print("\n生产通道未命中：", "、".join(missed) or "无")

    report = render_report(results, pairs, args.k, corpus, embedder_label, args.min_score)
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"\n报告已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
