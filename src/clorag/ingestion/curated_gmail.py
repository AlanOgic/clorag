"""Curated Gmail ingestion pipeline with LLM analysis and QC."""

import uuid
from datetime import datetime

import structlog

from clorag.analysis import CameraExtractor, QualityController, ThreadAnalyzer
from clorag.config import get_settings
from clorag.core.database import get_camera_database
from clorag.core.embeddings import EmbeddingsClient
from clorag.core.sparse_embeddings import SparseEmbeddingsClient
from clorag.core.vectorstore import VectorStore
from clorag.ingestion.base import Document
from clorag.ingestion.chunker import TextChunker
from clorag.ingestion.gmail import GmailIngestionPipeline
from clorag.models import (
    CaseStatus,
    ResolutionQuality,
    SupportCase,
)
from clorag.models.camera import CameraSource
from clorag.utils.anonymizer import AnonymizationContext, TextAnonymizer
from clorag.utils.text_transforms import apply_product_name_transforms

logger = structlog.get_logger(__name__)


class CuratedGmailPipeline:
    """Curated Gmail ingestion with LLM analysis and quality control.

    Pipeline flow:
    1. Fetch Gmail threads
    2. Analyze with Haiku (parallel) - extract problem/solution/keywords
    3. Filter for resolved cases
    4. QC with Sonnet - refine and validate
    5. Chunk and embed with voyage-context-3 (contextualized)
    6. Store in Qdrant with rich metadata
    """

    def __init__(
        self,
        max_threads: int | None = None,
        offset: int = 0,
        min_confidence: float = 0.7,
        haiku_concurrent: int = 10,
        sonnet_concurrent: int = 5,
        extract_cameras: bool = True,
        since_days: int | None = None,
    ) -> None:
        """Initialize the curated pipeline.

        Args:
            max_threads: Maximum threads to fetch from Gmail.
            offset: Number of threads to skip (for incremental ingestion).
            min_confidence: Minimum confidence for resolved classification.
            haiku_concurrent: Concurrent Haiku requests.
            sonnet_concurrent: Concurrent Sonnet requests.
            extract_cameras: Whether to extract camera compatibility info.
            since_days: Only fetch threads from the last N days.
        """
        self._settings = get_settings()
        self._max_threads = max_threads
        self._offset = offset
        self._min_confidence = min_confidence

        # Initialize components
        self._gmail = GmailIngestionPipeline(
            max_threads=max_threads, offset=offset, since_days=since_days
        )
        self._analyzer = ThreadAnalyzer(max_concurrent=haiku_concurrent)
        self._qc = QualityController()
        self._embeddings = EmbeddingsClient()
        self._sparse_embeddings = SparseEmbeddingsClient()
        self._vectorstore = VectorStore()
        self._chunker = TextChunker()
        self._anonymizer = TextAnonymizer()

        self._sonnet_concurrent = sonnet_concurrent
        self._extract_cameras = extract_cameras

    async def run(self) -> int:
        """Run the full curated ingestion pipeline.

        Returns:
            Number of cases successfully ingested.
        """
        logger.info("Starting curated Gmail ingestion")

        # Step 1: Fetch raw threads
        logger.info("Step 1: Fetching Gmail threads")
        raw_documents = await self._gmail.fetch()
        if not raw_documents:
            logger.warning("No threads fetched from Gmail")
            return 0

        logger.info("Fetched Gmail threads", count=len(raw_documents))

        # Step 1.5: Filter out RMA cases
        original_count = len(raw_documents)
        rma_count = sum(1 for doc in raw_documents if doc.metadata.get("is_rma"))
        raw_documents = [doc for doc in raw_documents if not doc.metadata.get("is_rma")]
        logger.info(
            "Filtered RMA cases",
            original=original_count,
            rma_filtered=rma_count,
            remaining=len(raw_documents),
        )

        if not raw_documents:
            logger.warning("No threads remaining after RMA filtering")
            return 0

        # Step 1.6: Pre-anonymize thread content and apply product name transforms
        logger.info("Anonymizing thread content and applying product name transforms")
        anonymized_documents: list[Document] = []
        for doc in raw_documents:
            # Create a new context for each thread (consistent placeholders within thread)
            context = AnonymizationContext()
            anonymized_text, _ = self._anonymizer.anonymize(doc.text, context)
            # Apply product name transformations (RIO-Live -> RIO +LAN, RIO -> RIO +WAN)
            transformed_text = apply_product_name_transforms(anonymized_text)
            anonymized_documents.append(
                Document(
                    id=doc.id,
                    text=transformed_text,
                    metadata=doc.metadata,
                )
            )
        raw_documents = anonymized_documents
        logger.info("Anonymized and transformed threads", count=len(raw_documents))

        # Step 2: Analyze with Haiku
        logger.info("Step 2: Analyzing threads with Haiku")
        threads_for_analysis = [
            (doc.metadata.get("thread_id", doc.id), doc.text)
            for doc in raw_documents
        ]
        analyses = await self._analyzer.analyze_threads_batch(threads_for_analysis)

        # Step 3: Filter for resolved cases
        resolved = [
            a for a in analyses
            if a.is_resolved and a.confidence >= self._min_confidence
        ]
        logger.info(
            "Step 3: Filtered for resolved cases",
            total=len(analyses),
            resolved=len(resolved),
            min_confidence=self._min_confidence,
        )

        if not resolved:
            logger.warning("No resolved cases found")
            return 0

        # Create thread_id -> document mapping
        doc_by_thread = {
            doc.metadata.get("thread_id", doc.id): doc
            for doc in raw_documents
        }

        # Step 4: QC with Sonnet
        logger.info("Step 4: Quality control with Sonnet")
        cases_for_qc = [
            (analysis, doc_by_thread.get(analysis.thread_id, Document(id=analysis.thread_id, text="", metadata={})).text)
            for analysis in resolved
            if analysis.thread_id in doc_by_thread
        ]
        qc_results = await self._qc.review_cases_batch(
            cases_for_qc,
            max_concurrent=self._sonnet_concurrent,
        )

        logger.info("QC approved cases", approved=len(qc_results))

        if not qc_results:
            logger.warning("No cases approved by QC")
            return 0

        # Step 5: Build SupportCase objects
        logger.info("Step 5: Building support cases")
        support_cases: list[SupportCase] = []
        for analysis, qc_result in qc_results:
            doc = doc_by_thread.get(analysis.thread_id)
            if not doc:
                continue

            # Use anonymized title from QC, falling back to analysis, then original subject
            anonymized_subject = (
                qc_result.anonymized_title
                or analysis.anonymized_subject
                or "Support Case"
            )

            # Apply product name transforms to final document
            final_document = apply_product_name_transforms(qc_result.final_document)

            case = SupportCase(
                id=str(uuid.uuid4()),
                thread_id=analysis.thread_id,
                subject=anonymized_subject,  # Use anonymized title instead of original
                status=CaseStatus.RESOLVED,
                resolution_quality=ResolutionQuality(analysis.resolution_quality) if analysis.resolution_quality else None,
                problem_summary=apply_product_name_transforms(qc_result.refined_problem),
                solution_summary=apply_product_name_transforms(qc_result.refined_solution),
                keywords=qc_result.refined_keywords,
                category=qc_result.refined_category,
                product=analysis.product,
                document=final_document,
                raw_thread=doc.text,
                messages_count=doc.metadata.get("message_count", 0),
                created_at=datetime.fromisoformat(doc.metadata["date"]) if doc.metadata.get("date") else None,
                participants=[],  # Don't store participant emails (anonymization)
            )
            support_cases.append(case)

        logger.info("Built support cases", count=len(support_cases))

        # Step 6: Chunk and prepare for contextualized embedding
        logger.info("Step 6: Chunking and preparing embeddings")

        # Ensure collection exists with hybrid vector support (dense + sparse)
        await self._vectorstore.ensure_collections(hybrid=True)

        # Collect all chunks and their metadata for batched processing
        all_chunk_texts: list[str] = []
        documents_chunks: list[list[str]] = []
        # (case, chunk_texts, sparse_start_idx, sparse_end_idx)
        case_chunk_info: list[tuple[SupportCase, list[str], int, int]] = []

        for case in support_cases:
            # Chunk the final document
            chunks = self._chunker.chunk_text(case.document)
            chunk_texts = [chunk.text for chunk in chunks]

            if chunk_texts:
                sparse_start = len(all_chunk_texts)
                all_chunk_texts.extend(chunk_texts)
                sparse_end = len(all_chunk_texts)
                documents_chunks.append(chunk_texts)
                case_chunk_info.append((case, chunk_texts, sparse_start, sparse_end))

        if not all_chunk_texts:
            logger.warning("No chunks generated")
            return 0

        # Generate ALL sparse BM25 vectors in one batch (performance optimization)
        logger.info("Generating sparse BM25 embeddings", total_chunks=len(all_chunk_texts))
        all_sparse_vectors = self._sparse_embeddings.embed_batch(all_chunk_texts)

        # Generate contextualized dense embeddings (already batched by design)
        logger.info(
            "Generating contextualized dense embeddings",
            documents=len(documents_chunks),
            total_chunks=len(all_chunk_texts),
        )
        embeddings = await self._embeddings.embed_contextualized_batch(
            documents_chunks,
            batch_size=10,
        )

        # Step 7: Store in Qdrant with hybrid vectors
        logger.info("Step 7: Storing in Qdrant with hybrid vectors")

        total_stored = 0
        for case_idx, (case, chunk_texts, sparse_start, sparse_end) in enumerate(case_chunk_info):
            case_embeddings = embeddings[case_idx]
            # Slice pre-computed sparse vectors for this case
            sparse_vectors = all_sparse_vectors[sparse_start:sparse_end]

            # Prepare all chunks for batch upsert
            chunk_ids = [
                str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{case.id}_{chunk_idx}"))
                for chunk_idx in range(len(chunk_texts))
            ]
            metadata_list = [
                {
                    **case.to_metadata(),
                    "chunk_index": chunk_idx,
                    "parent_case_id": case.id,
                    "text": text,
                }
                for chunk_idx, text in enumerate(chunk_texts)
            ]

            # Upsert with hybrid vectors (dense + sparse)
            await self._vectorstore.upsert_documents_hybrid(
                collection=self._vectorstore.cases_collection,
                texts=chunk_texts,
                dense_vectors=case_embeddings,
                sparse_vectors=sparse_vectors,
                metadata=metadata_list,
                ids=chunk_ids,
            )
            total_stored += len(chunk_texts)

        logger.info(
            "Curated ingestion complete",
            cases=len(support_cases),
            chunks_stored=total_stored,
        )

        # Step 8: Extract camera compatibility info from support cases
        if self._extract_cameras:
            await self._extract_cameras_from_cases(support_cases)

        return len(support_cases)

    async def _extract_cameras_from_cases(
        self, cases: list[SupportCase]
    ) -> int:
        """Extract and store camera compatibility info from support cases.

        Args:
            cases: List of support cases to extract from.

        Returns:
            Number of cameras extracted/updated.
        """
        extractor = CameraExtractor()
        db = get_camera_database()

        # Prepare contents for batch extraction (use final document for context)
        contents: list[tuple[str, str | None]] = []
        for case in cases:
            # Use the final QC document which has the clean problem/solution
            contents.append((case.document, None))

        logger.info("Extracting cameras from support cases", cases=len(contents))

        try:
            cameras = await extractor.extract_from_batch(contents, concurrency=5)

            # Upsert cameras into database
            for camera in cameras:
                db.upsert_camera(camera, CameraSource.SUPPORT_CASE)

            logger.info(
                "Camera extraction from support cases complete",
                extracted=len(cameras),
            )
            return len(cameras)

        except Exception as e:
            logger.error("Camera extraction from support cases failed", error=str(e))
            return 0
        finally:
            await extractor.close()


async def run_curated_ingestion(
    max_threads: int | None = None,
    offset: int = 0,
    min_confidence: float = 0.7,
    since_days: int | None = None,
) -> int:
    """Run curated Gmail ingestion.

    Args:
        max_threads: Maximum threads to fetch.
        offset: Number of threads to skip (for incremental ingestion).
        min_confidence: Minimum confidence for resolved cases.
        since_days: Only fetch threads from the last N days.

    Returns:
        Number of cases ingested.
    """
    pipeline = CuratedGmailPipeline(
        max_threads=max_threads,
        offset=offset,
        min_confidence=min_confidence,
        since_days=since_days,
    )
    return await pipeline.run()
