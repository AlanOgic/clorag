"""Evaluate retrieval pipeline quality using a pre-generated Q&A dataset.

Loads eval_dataset.json, runs each question through the MultiSourceRetriever,
and computes Recall@5, MRR (Mean Reciprocal Rank), and NDCG.

Usage:
    uv run python scripts/eval_retrieval.py [--limit 0] [--top-k 5] [--no-rerank]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path
from typing import Any

import structlog

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clorag.core.retriever import MultiSourceRetriever, SearchSource  # noqa: E402

logger = structlog.get_logger(__name__)

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_dataset.json"


def url_match(result_url: str, expected_url: str) -> bool:
    """Check if a retrieved result URL matches the expected source URL.

    Handles trailing slashes and minor normalization.
    """
    a = result_url.rstrip("/").lower()
    b = expected_url.rstrip("/").lower()
    return a == b


def compute_recall_at_k(
    retrieved_urls: list[str],
    expected_url: str,
    k: int,
) -> float:
    """Recall@K: 1.0 if expected URL appears in top-K results, else 0.0."""
    for url in retrieved_urls[:k]:
        if url_match(url, expected_url):
            return 1.0
    return 0.0


def compute_reciprocal_rank(
    retrieved_urls: list[str],
    expected_url: str,
) -> float:
    """Reciprocal rank: 1/rank of first matching result, or 0.0 if not found."""
    for i, url in enumerate(retrieved_urls):
        if url_match(url, expected_url):
            return 1.0 / (i + 1)
    return 0.0


def compute_ndcg(
    retrieved_urls: list[str],
    expected_url: str,
    k: int,
) -> float:
    """NDCG@K with binary relevance (1 if URL matches, 0 otherwise).

    For a single relevant document, ideal DCG = 1.0 (relevant doc at rank 1).
    """
    # DCG: sum of rel_i / log2(i+1) for i in 1..k
    dcg = 0.0
    for i, url in enumerate(retrieved_urls[:k]):
        if url_match(url, expected_url):
            dcg += 1.0 / math.log2(i + 2)  # i+2 because i is 0-indexed

    # Ideal DCG: single relevant doc at rank 1 = 1/log2(2) = 1.0
    idcg = 1.0

    return dcg / idcg if idcg > 0 else 0.0


async def evaluate_question(
    retriever: MultiSourceRetriever,
    question: str,
    expected_url: str,
    top_k: int,
    source: SearchSource,
) -> dict[str, Any]:
    """Evaluate a single question against the retrieval pipeline."""
    try:
        result = await retriever.retrieve(
            query=question,
            source=source,
            limit=top_k,
        )

        retrieved_urls = [
            r.payload.get("url", "") for r in result.results
        ]

        recall = compute_recall_at_k(retrieved_urls, expected_url, top_k)
        mrr = compute_reciprocal_rank(retrieved_urls, expected_url)
        ndcg = compute_ndcg(retrieved_urls, expected_url, top_k)

        return {
            "question": question,
            "expected_url": expected_url,
            "retrieved_urls": retrieved_urls[:top_k],
            "recall": recall,
            "mrr": mrr,
            "ndcg": ndcg,
            "num_results": len(result.results),
            "reranked": result.reranked,
            "error": None,
        }
    except Exception as exc:
        logger.warning("Evaluation failed for question", question=question, error=str(exc))
        return {
            "question": question,
            "expected_url": expected_url,
            "retrieved_urls": [],
            "recall": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "num_results": 0,
            "reranked": False,
            "error": str(exc),
        }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval pipeline")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max questions to evaluate (0 = all, default: 0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve per query (default: 5)",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable reranking for evaluation",
    )
    parser.add_argument(
        "--source",
        choices=["docs", "hybrid"],
        default="docs",
        help="Search source: docs-only or hybrid (default: docs)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Concurrent retrieval queries per batch (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to save detailed results JSON",
    )
    args = parser.parse_args()

    # Load dataset
    if not DATASET_PATH.exists():
        print(f"Dataset not found at {DATASET_PATH}")
        print("Run generate_eval_dataset.py first.")
        sys.exit(1)

    dataset: list[dict[str, str]] = json.loads(DATASET_PATH.read_text())
    if args.limit > 0:
        dataset = dataset[: args.limit]

    print(f"Loaded {len(dataset)} evaluation questions from {DATASET_PATH}")

    # Initialize retriever
    source = SearchSource.DOCS if args.source == "docs" else SearchSource.HYBRID
    use_reranking = not args.no_rerank
    retriever = MultiSourceRetriever(rerank_enabled=use_reranking)

    print(f"Config: top_k={args.top_k}, rerank={use_reranking}, source={args.source}")
    print(f"Running evaluation...\n")

    # Evaluate in batches
    all_results: list[dict[str, Any]] = []
    total = len(dataset)

    for i in range(0, total, args.batch_size):
        batch = dataset[i : i + args.batch_size]
        batch_num = i // args.batch_size + 1
        total_batches = (total + args.batch_size - 1) // args.batch_size

        tasks = [
            evaluate_question(
                retriever=retriever,
                question=item["question"],
                expected_url=item["source_url"],
                top_k=args.top_k,
                source=source,
            )
            for item in batch
        ]
        results = await asyncio.gather(*tasks)
        all_results.extend(results)

        done = min(i + args.batch_size, total)
        print(f"  Progress: {done}/{total} ({done * 100 // total}%)", end="\r")

    print()

    # Compute aggregate metrics
    errors = [r for r in all_results if r["error"]]
    valid = [r for r in all_results if not r["error"]]

    if not valid:
        print("No valid results to evaluate.")
        sys.exit(1)

    n = len(valid)
    avg_recall = sum(r["recall"] for r in valid) / n
    avg_mrr = sum(r["mrr"] for r in valid) / n
    avg_ndcg = sum(r["ndcg"] for r in valid) / n

    # Print report
    print("=" * 60)
    print(f"  RETRIEVAL EVALUATION REPORT")
    print("=" * 60)
    print(f"  Questions evaluated:  {n}")
    print(f"  Errors/skipped:       {len(errors)}")
    print(f"  Reranking:            {'enabled' if use_reranking else 'disabled'}")
    print(f"  Source:               {args.source}")
    print(f"  Top-K:                {args.top_k}")
    print("-" * 60)
    print(f"  Recall@{args.top_k}:             {avg_recall:.4f}  ({avg_recall * 100:.1f}%)")
    print(f"  MRR:                  {avg_mrr:.4f}")
    print(f"  NDCG@{args.top_k}:              {avg_ndcg:.4f}")
    print("=" * 60)

    # Show worst performing questions
    misses = [r for r in valid if r["recall"] == 0.0]
    if misses:
        print(f"\nMissed questions ({len(misses)}/{n}):")
        for r in misses[:10]:
            print(f"  Q: {r['question']}")
            print(f"     Expected: {r['expected_url']}")
            if r["retrieved_urls"]:
                print(f"     Got:      {r['retrieved_urls'][0]}")
            print()

    # Save detailed results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "config": {
                "top_k": args.top_k,
                "rerank": use_reranking,
                "source": args.source,
                "total_questions": n,
            },
            "metrics": {
                "recall_at_k": round(avg_recall, 4),
                "mrr": round(avg_mrr, 4),
                "ndcg_at_k": round(avg_ndcg, 4),
            },
            "results": all_results,
        }
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"Detailed results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
