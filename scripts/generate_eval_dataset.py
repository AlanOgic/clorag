"""Generate evaluation Q&A dataset from existing Qdrant documentation chunks.

Uses Claude Sonnet to generate question-answer pairs from real documentation
chunks stored in the docusaurus_docs collection. Each pair includes the source
URL for retrieval evaluation.

Usage:
    uv run python scripts/generate_eval_dataset.py [--count 150] [--batch-size 10]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Any

import anthropic
import structlog

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clorag.config import get_settings  # noqa: E402
from clorag.core.vectorstore import VectorStore  # noqa: E402

logger = structlog.get_logger(__name__)

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_dataset.json"

GENERATION_PROMPT = """\
You are generating evaluation questions for a retrieval system about Cyanview \
camera control products (RCP, RIO, CI0, VP4, NIO) and their documentation.

Given the following documentation chunk, generate {count} distinct question-answer \
pairs that a real user might ask and that this chunk can answer.

Requirements:
- Questions should be natural, as a support engineer or integrator would ask
- Questions should vary in style: some short (2-4 words), some medium, some detailed
- The answer_snippet should be a SHORT phrase (5-20 words) extracted or closely \
paraphrased from the chunk text
- Each question must be answerable from THIS chunk specifically
- Avoid yes/no questions; prefer "how", "what", "which", "where" questions
- Do NOT reference "the chunk" or "the document" in questions

Source URL: {url}
Source Title: {title}

--- CHUNK TEXT ---
{text}
--- END CHUNK ---

Respond with a JSON array of objects, each with:
- "question": the evaluation question
- "answer_snippet": short answer extracted from the chunk
- "source_url": "{url}"

Return ONLY the JSON array, no other text."""


async def sample_chunks(
    vector_store: VectorStore,
    collection: str,
    sample_size: int,
) -> list[dict[str, Any]]:
    """Sample random chunks from a Qdrant collection via scroll.

    Scrolls through all chunks and randomly samples from them.
    """
    all_chunks: list[dict[str, Any]] = []
    offset: str | None = None

    logger.info("Scrolling collection", collection=collection)
    while True:
        batch, next_offset = await vector_store.scroll_chunks(
            collection=collection,
            limit=100,
            offset=offset,
        )
        if not batch:
            break
        all_chunks.extend(batch)
        if not next_offset:
            break
        offset = next_offset

    logger.info("Total chunks found", collection=collection, count=len(all_chunks))

    # Filter chunks with sufficient text and a URL
    valid_chunks = [
        c for c in all_chunks
        if c.get("payload", {}).get("text", "").strip()
        and len(c.get("payload", {}).get("text", "")) > 100
        and c.get("payload", {}).get("url")
    ]
    logger.info("Valid chunks with URL and text", count=len(valid_chunks))

    if len(valid_chunks) <= sample_size:
        return valid_chunks

    return random.sample(valid_chunks, sample_size)


async def generate_qa_pairs(
    client: anthropic.AsyncAnthropic,
    model: str,
    chunk: dict[str, Any],
    questions_per_chunk: int,
) -> list[dict[str, str]]:
    """Generate Q&A pairs from a single chunk using Claude."""
    payload = chunk.get("payload", {})
    text = payload.get("text", "")
    url = payload.get("url", "")
    title = payload.get("title", "Untitled")

    prompt = GENERATION_PROMPT.format(
        count=questions_per_chunk,
        url=url,
        title=title,
        text=text,
    )

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.content[0].text.strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[: content.rfind("```")]

        pairs: list[dict[str, str]] = json.loads(content)

        # Validate structure
        validated: list[dict[str, str]] = []
        for pair in pairs:
            if (
                isinstance(pair, dict)
                and "question" in pair
                and "answer_snippet" in pair
            ):
                validated.append({
                    "question": pair["question"],
                    "answer_snippet": pair["answer_snippet"],
                    "source_url": url,
                })
        return validated

    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.warning("Failed to parse LLM response", url=url, error=str(exc))
        return []
    except anthropic.APIError as exc:
        logger.warning("Anthropic API error", url=url, error=str(exc))
        return []


async def main() -> None:
    parser = argparse.ArgumentParser(description="Generate retrieval eval dataset")
    parser.add_argument(
        "--count",
        type=int,
        default=150,
        help="Target number of Q&A pairs (default: 150)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Concurrent LLM calls per batch (default: 5)",
    )
    parser.add_argument(
        "--questions-per-chunk",
        type=int,
        default=3,
        help="Questions to generate per chunk (default: 3)",
    )
    args = parser.parse_args()

    settings = get_settings()
    model = settings.sonnet_model
    api_key = settings.anthropic_api_key.get_secret_value()

    vector_store = VectorStore()
    client = anthropic.AsyncAnthropic(api_key=api_key)

    # Calculate how many chunks we need to sample
    chunks_needed = (args.count + args.questions_per_chunk - 1) // args.questions_per_chunk
    # Over-sample slightly to account for generation failures
    sample_size = int(chunks_needed * 1.2)

    logger.info(
        "Sampling chunks",
        target_pairs=args.count,
        chunks_needed=chunks_needed,
        sample_size=sample_size,
    )

    chunks = await sample_chunks(
        vector_store,
        collection=settings.qdrant_docs_collection,
        sample_size=sample_size,
    )

    if not chunks:
        logger.error("No valid chunks found in collection")
        sys.exit(1)

    logger.info("Generating Q&A pairs", chunks=len(chunks), model=model)

    all_pairs: list[dict[str, str]] = []

    # Process in batches for concurrency control
    for i in range(0, len(chunks), args.batch_size):
        batch = chunks[i : i + args.batch_size]
        batch_num = i // args.batch_size + 1
        total_batches = (len(chunks) + args.batch_size - 1) // args.batch_size

        logger.info("Processing batch", batch=batch_num, total=total_batches)

        tasks = [
            generate_qa_pairs(client, model, chunk, args.questions_per_chunk)
            for chunk in batch
        ]
        results = await asyncio.gather(*tasks)

        for pairs in results:
            all_pairs.extend(pairs)

        # Stop early if we have enough
        if len(all_pairs) >= args.count:
            break

    # Trim to target count
    all_pairs = all_pairs[: args.count]

    # Deduplicate by question text
    seen_questions: set[str] = set()
    unique_pairs: list[dict[str, str]] = []
    for pair in all_pairs:
        q = pair["question"].lower().strip()
        if q not in seen_questions:
            seen_questions.add(q)
            unique_pairs.append(pair)

    # Save dataset
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(unique_pairs, indent=2, ensure_ascii=False))

    # Summary stats
    urls = {p["source_url"] for p in unique_pairs}
    logger.info(
        "Dataset generated",
        total_pairs=len(unique_pairs),
        unique_urls=len(urls),
        output=str(OUTPUT_PATH),
    )
    print(f"\nGenerated {len(unique_pairs)} Q&A pairs from {len(urls)} unique pages")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
