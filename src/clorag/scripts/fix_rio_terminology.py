"""CLI script for fixing RIO terminology in vector database chunks."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import anyio
import structlog

from clorag.analysis.rio_analyzer import RIOTerminologyAnalyzer, apply_fix_to_text
from clorag.core.embeddings import EmbeddingsClient
from clorag.core.sparse_embeddings import SparseEmbeddingsClient
from clorag.core.terminology_db import TerminologyFix, get_terminology_fix_database
from clorag.core.vectorstore import VectorStore
from clorag.utils.logger import setup_logging

logger = structlog.get_logger(__name__)

# Collections to scan
COLLECTIONS = ["docusaurus_docs", "gmail_cases", "custom_docs"]


async def scan_for_rio_mentions(
    vectorstore: VectorStore,
    analyzer: RIOTerminologyAnalyzer,
    max_chunks: int | None = None,
) -> list[TerminologyFix]:
    """Scan all collections for chunks with RIO mentions.

    Args:
        vectorstore: VectorStore instance.
        analyzer: RIOTerminologyAnalyzer instance.
        max_chunks: Maximum chunks to scan (for testing).

    Returns:
        List of TerminologyFix suggestions.
    """
    all_fixes: list[TerminologyFix] = []
    chunks_scanned = 0
    chunks_with_rio = 0

    for collection in COLLECTIONS:
        logger.info("Scanning collection", collection=collection)

        try:
            offset: str | None = None
            while True:
                # Fetch batch of chunks
                chunks, next_offset = await vectorstore.scroll_chunks(
                    collection=collection,
                    limit=50,
                    offset=offset,
                )

                if not chunks:
                    break

                # Filter chunks with RIO mentions
                chunks_to_analyze: list[tuple[str, str, str]] = []
                for chunk in chunks:
                    text = chunk.get("payload", {}).get("text", "")
                    if text and analyzer.has_rio_mentions(text):
                        chunks_to_analyze.append(
                            (chunk["id"], collection, text)
                        )
                        chunks_with_rio += 1

                chunks_scanned += len(chunks)

                # Analyze chunks with RIO mentions
                if chunks_to_analyze:
                    fixes = await analyzer.analyze_chunks_batch(chunks_to_analyze)
                    all_fixes.extend(fixes)
                    logger.info(
                        "Analyzed batch",
                        collection=collection,
                        chunks_analyzed=len(chunks_to_analyze),
                        fixes_found=len(fixes),
                    )

                # Check if we've reached max chunks
                if max_chunks and chunks_scanned >= max_chunks:
                    logger.info("Reached max chunks limit", max_chunks=max_chunks)
                    break

                # Move to next page
                if not next_offset:
                    break
                offset = next_offset

        except Exception as e:
            logger.error("Error scanning collection", collection=collection, error=str(e))

    logger.info(
        "Scan complete",
        chunks_scanned=chunks_scanned,
        chunks_with_rio=chunks_with_rio,
        fixes_found=len(all_fixes),
    )

    return all_fixes


def update_keywords(
    keywords: list[str] | None,
    original: str,
    suggested: str,
) -> list[str] | None:
    """Update keywords list, replacing old terminology with new.

    Args:
        keywords: Original keywords list.
        original: Original terminology to find.
        suggested: New terminology to replace with.

    Returns:
        Updated keywords list, or None if no keywords.
    """
    if not keywords:
        return None

    # Normalize for comparison
    original_lower = original.lower()
    suggested_lower = suggested.lower()

    # Legacy terms to remove entirely (not replace)
    legacy_remove = ["rio-live", "rio live", "rio +wan live", "rio+wan live"]

    updated: list[str] = []
    added_suggested = False

    for kw in keywords:
        kw_lower = kw.lower()

        # Remove legacy terms entirely
        if kw_lower in legacy_remove:
            # Add suggested term once if we're removing legacy
            if not added_suggested and suggested_lower not in [k.lower() for k in updated]:
                updated.append(suggested)
                added_suggested = True
            continue

        # Replace matching terms
        if kw_lower == original_lower:
            if suggested_lower not in [k.lower() for k in updated]:
                updated.append(suggested)
            continue

        # Keep other keywords
        updated.append(kw)

    return updated if updated else None


async def apply_approved_fixes(
    vectorstore: VectorStore,
    embeddings: EmbeddingsClient,
    sparse_embeddings: SparseEmbeddingsClient,
) -> int:
    """Apply all approved terminology fixes to chunks.

    Also updates related metadata fields (subject, summaries, keywords).

    Args:
        vectorstore: VectorStore instance.
        embeddings: EmbeddingsClient for regenerating embeddings.
        sparse_embeddings: SparseEmbeddingsClient for regenerating sparse vectors.

    Returns:
        Number of fixes applied.
    """
    db = get_terminology_fix_database()
    approved_fixes = db.get_approved_fixes()

    if not approved_fixes:
        logger.info("No approved fixes to apply")
        return 0

    applied_count = 0
    failed_ids: list[str] = []

    for fix in approved_fixes:
        try:
            # Get current chunk
            chunk = await vectorstore.get_chunk(fix.collection, fix.chunk_id)
            if not chunk:
                logger.warning("Chunk not found", chunk_id=fix.chunk_id)
                failed_ids.append(fix.id)
                continue

            payload = chunk.get("payload", {})
            current_text = payload.get("text", "")
            if not current_text:
                logger.warning("Chunk has no text", chunk_id=fix.chunk_id)
                failed_ids.append(fix.id)
                continue

            # Apply the fix to main text
            new_text = apply_fix_to_text(current_text, fix.original_text, fix.suggested_text)

            # Build metadata updates for related fields
            metadata_updates: dict[str, str | list[str] | None] = {}

            # Apply fix to subject if present
            subject = payload.get("subject")
            if subject and fix.original_text.lower() in subject.lower():
                new_subject = apply_fix_to_text(subject, fix.original_text, fix.suggested_text)
                if new_subject != subject:
                    metadata_updates["subject"] = new_subject

            # Apply fix to problem_summary if present
            problem = payload.get("problem_summary")
            if problem and fix.original_text.lower() in problem.lower():
                new_problem = apply_fix_to_text(problem, fix.original_text, fix.suggested_text)
                if new_problem != problem:
                    metadata_updates["problem_summary"] = new_problem

            # Apply fix to solution_summary if present
            solution = payload.get("solution_summary")
            if solution and fix.original_text.lower() in solution.lower():
                new_solution = apply_fix_to_text(solution, fix.original_text, fix.suggested_text)
                if new_solution != solution:
                    metadata_updates["solution_summary"] = new_solution

            # Update keywords - remove legacy terms, add new terminology
            keywords = payload.get("keywords")
            if keywords and isinstance(keywords, list):
                new_keywords = update_keywords(keywords, fix.original_text, fix.suggested_text)
                if new_keywords != keywords:
                    metadata_updates["keywords"] = new_keywords

            # Check if any changes were made
            if new_text == current_text and not metadata_updates:
                logger.warning(
                    "No change after applying fix",
                    chunk_id=fix.chunk_id,
                    original=fix.original_text,
                )
                db.update_status(fix.id, "applied", datetime.utcnow())
                applied_count += 1
                continue

            # Regenerate embeddings for new text
            dense_result = await embeddings.embed_documents([new_text])
            sparse_vector = sparse_embeddings.embed_texts([new_text])[0]

            # Update chunk in vectorstore (text + metadata)
            success = await vectorstore.update_chunk(
                collection=fix.collection,
                chunk_id=fix.chunk_id,
                text=new_text,
                metadata_updates=metadata_updates if metadata_updates else None,
                dense_vector=dense_result.vectors[0],
                sparse_vector=sparse_vector,
            )

            if success:
                db.update_status(fix.id, "applied", datetime.utcnow())
                applied_count += 1
                logger.info(
                    "Applied fix",
                    chunk_id=fix.chunk_id,
                    original=fix.original_text,
                    suggested=fix.suggested_text,
                    metadata_updated=list(metadata_updates.keys()) if metadata_updates else [],
                )
            else:
                logger.error("Failed to update chunk", chunk_id=fix.chunk_id)
                failed_ids.append(fix.id)

        except Exception as e:
            logger.error("Error applying fix", fix_id=fix.id, error=str(e))
            failed_ids.append(fix.id)

    logger.info(
        "Applied fixes complete",
        applied=applied_count,
        failed=len(failed_ids),
    )

    return applied_count


async def run_preview(max_chunks: int | None = None) -> int:
    """Run preview mode: scan and save suggestions without applying.

    Args:
        max_chunks: Maximum chunks to scan.

    Returns:
        Number of suggestions found.
    """
    vectorstore = VectorStore()
    analyzer = RIOTerminologyAnalyzer()
    db = get_terminology_fix_database()

    # Clear any previous pending fixes
    cleared = db.clear_pending()
    if cleared:
        logger.info("Cleared previous pending fixes", count=cleared)

    # Scan for RIO mentions
    fixes = await scan_for_rio_mentions(vectorstore, analyzer, max_chunks)

    if not fixes:
        logger.info("No terminology fixes needed")
        return 0

    # Save to database
    count = db.insert_fixes_batch(fixes)

    # Also export to JSON for reference
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    json_path = data_dir / "rio_terminology_fixes.json"
    db.export_to_json(str(json_path))

    logger.info(
        "Preview complete - suggestions saved",
        count=count,
        json_path=str(json_path),
    )

    # Print summary
    stats = db.get_stats()
    print(f"\n{'=' * 60}")
    print("RIO Terminology Fix Preview")
    print(f"{'=' * 60}")
    print(f"Total suggestions: {stats['total']}")
    print("\nBy type:")
    by_type = stats.get("by_type", {})
    if isinstance(by_type, dict):
        for fix_type, count in by_type.items():
            print(f"  - {fix_type}: {count}")
    print("\nBy collection:")
    by_collection = stats.get("by_collection", {})
    if isinstance(by_collection, dict):
        for collection, count in by_collection.items():
            print(f"  - {collection}: {count}")
    print("\nReview suggestions at: /admin/terminology-fixes")
    print("Or run: uv run fix-rio-terminology --apply")
    print(f"{'=' * 60}\n")

    return len(fixes)


async def run_apply() -> int:
    """Run apply mode: apply all approved fixes.

    Returns:
        Number of fixes applied.
    """
    vectorstore = VectorStore()
    embeddings = EmbeddingsClient()
    sparse_embeddings = SparseEmbeddingsClient()

    count = await apply_approved_fixes(vectorstore, embeddings, sparse_embeddings)

    print(f"\n{'=' * 60}")
    print("RIO Terminology Fixes Applied")
    print(f"{'=' * 60}")
    print(f"Fixes applied: {count}")
    print(f"{'=' * 60}\n")

    return count


async def run_stats() -> None:
    """Show statistics about terminology fixes."""
    db = get_terminology_fix_database()
    stats = db.get_stats()

    print(f"\n{'=' * 60}")
    print("RIO Terminology Fix Statistics")
    print(f"{'=' * 60}")
    print(f"Total fixes: {stats['total']}")
    print("\nBy status:")
    by_status = stats.get("by_status", {})
    if isinstance(by_status, dict):
        for status, count in by_status.items():
            print(f"  - {status}: {count}")
    print("\nBy type:")
    by_type = stats.get("by_type", {})
    if isinstance(by_type, dict):
        for fix_type, count in by_type.items():
            print(f"  - {fix_type}: {count}")
    print("\nBy collection:")
    by_collection = stats.get("by_collection", {})
    if isinstance(by_collection, dict):
        for collection, count in by_collection.items():
            print(f"  - {collection}: {count}")
    print(f"{'=' * 60}\n")


def main() -> None:
    """Main entry point."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Fix RIO terminology in vector database chunks"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Scan and save suggestions without applying (default mode)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply all approved fixes to chunks",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics about terminology fixes",
    )
    parser.add_argument(
        "--max-chunks",
        "-m",
        type=int,
        help="Maximum chunks to scan (for testing)",
    )
    parser.add_argument(
        "--export",
        type=str,
        metavar="FILE",
        help="Export fixes to JSON file",
    )
    parser.add_argument(
        "--import",
        dest="import_file",
        type=str,
        metavar="FILE",
        help="Import fixes from JSON file",
    )

    args = parser.parse_args()

    try:
        if args.stats:
            anyio.run(run_stats)
        elif args.apply:
            logger.info("Starting terminology fix application")
            count = anyio.run(run_apply)
            logger.info("Terminology fixes applied", count=count)
        elif args.export:
            db = get_terminology_fix_database()
            count = db.export_to_json(args.export)
            print(f"Exported {count} fixes to {args.export}")
        elif args.import_file:
            db = get_terminology_fix_database()
            count = db.import_from_json(args.import_file)
            print(f"Imported {count} fixes from {args.import_file}")
        else:
            # Default: preview mode
            logger.info(
                "Starting terminology fix preview",
                max_chunks=args.max_chunks or "all",
            )
            count = anyio.run(run_preview, args.max_chunks)
            logger.info("Preview complete", suggestions=count)

    except Exception as e:
        logger.error("Terminology fix failed", error=str(e))
        raise


if __name__ == "__main__":
    main()
