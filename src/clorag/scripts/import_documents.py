"""Bulk import documents into the custom knowledge base."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anyio
from pypdf import PdfReader

from clorag.models.custom_document import CustomDocumentCreate, DocumentCategory
from clorag.services.custom_docs import CustomDocumentService
from clorag.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def extract_text_from_pdf(path: Path) -> str:
    """Extract text content from a PDF file.

    Args:
        path: Path to PDF file.

    Returns:
        Extracted text content.
    """
    reader = PdfReader(path)
    text_parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_parts.append(text)
    return "\n\n".join(text_parts)


def read_file_content(path: Path) -> str:
    """Read content from a file (txt, md, or pdf).

    Args:
        path: Path to file.

    Returns:
        File content as text.
    """
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    return path.read_text(encoding="utf-8")


def title_from_filename(path: Path) -> str:
    """Generate a title from filename.

    Args:
        path: Path to file.

    Returns:
        Title derived from filename.
    """
    name = path.stem
    # Replace underscores and hyphens with spaces, title-case
    title = name.replace("_", " ").replace("-", " ")
    return title.title()


def find_documents(folder: Path) -> list[Path]:
    """Recursively find all supported documents in a folder.

    Args:
        folder: Root folder to scan.

    Returns:
        List of document paths.
    """
    docs: list[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        docs.extend(folder.rglob(f"*{ext}"))
    return sorted(docs)


async def import_documents(
    folder: Path,
    category: DocumentCategory,
    tags: list[str],
    dry_run: bool = False,
) -> tuple[int, int]:
    """Import all documents from a folder.

    Args:
        folder: Folder containing documents.
        category: Category to assign to all documents.
        tags: Tags to assign to all documents.
        dry_run: If True, only preview without importing.

    Returns:
        Tuple of (imported count, skipped count).
    """
    docs = find_documents(folder)

    if not docs:
        logger.warning("No documents found", folder=str(folder))
        return 0, 0

    logger.info(
        "Found documents",
        count=len(docs),
        extensions=[ext for ext in SUPPORTED_EXTENSIONS],
    )

    if dry_run:
        print("\n[DRY RUN] Would import the following documents:\n")
        for doc_path in docs:
            rel_path = doc_path.relative_to(folder)
            title = title_from_filename(doc_path)
            print(f"  - {rel_path}")
            print(f"    Title: {title}")
            print(f"    Category: {category.value}")
            if tags:
                print(f"    Tags: {', '.join(tags)}")
            print()
        print(f"Total: {len(docs)} documents")
        return len(docs), 0

    service = CustomDocumentService()
    imported = 0
    skipped = 0

    for i, doc_path in enumerate(docs, 1):
        rel_path = doc_path.relative_to(folder)
        title = title_from_filename(doc_path)

        try:
            content = read_file_content(doc_path)

            if len(content.strip()) < 10:
                logger.warning(
                    "Skipping empty/short document",
                    path=str(rel_path),
                )
                skipped += 1
                continue

            doc_create = CustomDocumentCreate(
                title=title,
                content=content,
                category=category,
                tags=tags,
            )

            await service.create_document(doc_create, created_by="import-docs")
            imported += 1

            logger.info(
                "Imported document",
                progress=f"{i}/{len(docs)}",
                title=title,
                path=str(rel_path),
            )

        except Exception as e:
            logger.error(
                "Failed to import document",
                path=str(rel_path),
                error=str(e),
            )
            skipped += 1

    return imported, skipped


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Bulk import documents into the custom knowledge base.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import all docs from a folder
  uv run import-docs ./presales-folder --category pre_sales

  # With tags
  uv run import-docs ./docs --category product_info --tags "integration,api"

  # Preview without importing
  uv run import-docs ./docs --dry-run

Supported file types: .txt, .md, .pdf
        """,
    )

    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing documents to import",
    )

    parser.add_argument(
        "--category",
        "-c",
        type=str,
        default="other",
        choices=[cat.value for cat in DocumentCategory],
        help="Category to assign to all documents (default: other)",
    )

    parser.add_argument(
        "--tags",
        "-t",
        type=str,
        default="",
        help="Comma-separated tags to assign (e.g., 'presales,integration')",
    )

    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Preview documents without importing",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for bulk import."""
    setup_logging()
    args = parse_args()

    folder = args.folder.resolve()
    if not folder.is_dir():
        logger.error("Folder does not exist", path=str(folder))
        sys.exit(1)

    category = DocumentCategory(args.category)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    logger.info(
        "Starting bulk import",
        folder=str(folder),
        category=category.value,
        tags=tags,
        dry_run=args.dry_run,
    )

    try:
        imported, skipped = anyio.run(
            import_documents,
            folder,
            category,
            tags,
            args.dry_run,
        )

        if not args.dry_run:
            logger.info(
                "Import completed",
                imported=imported,
                skipped=skipped,
            )
            print(f"\nImported {imported} documents, skipped {skipped}")

    except Exception as e:
        logger.error("Import failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
