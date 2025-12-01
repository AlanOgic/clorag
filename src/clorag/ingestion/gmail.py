"""Gmail threads ingestion pipeline."""

import asyncio
import base64
import re
import uuid
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from clorag.config import get_settings
from clorag.core.embeddings import EmbeddingsClient
from clorag.core.vectorstore import VectorStore
from clorag.ingestion.base import BaseIngestionPipeline, Document
from clorag.ingestion.chunker import TextChunker
from clorag.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


async def run_sync(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a synchronous function in the default executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))

# RMA detection patterns
RMA_SUBJECT_PATTERNS = [
    re.compile(r"\bRMA\b", re.IGNORECASE),
    re.compile(r"\breturn\s+(request|authorization|merchandise)\b", re.IGNORECASE),
    re.compile(r"\brefund\s+(request|processing)\b", re.IGNORECASE),
    re.compile(r"\breplacement\s+(request|unit|device)\b", re.IGNORECASE),
]

RMA_BODY_PATTERNS = [
    re.compile(r"\bRMA\s*(number|#|:)?\s*\d+", re.IGNORECASE),
    re.compile(r"\breturn\s+(shipping\s+)?label\b", re.IGNORECASE),
    re.compile(r"\breplacement\s+shipment\b", re.IGNORECASE),
    re.compile(r"\breturn\s+the\s+(defective|faulty)\s+(unit|device)\b", re.IGNORECASE),
    re.compile(r"\bship\s+(back|to\s+us)\b", re.IGNORECASE),
    re.compile(r"\breturn\s+address\b", re.IGNORECASE),
]

# Gmail API scopes (read-only)
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


class GmailIngestionPipeline(BaseIngestionPipeline):
    """Pipeline for ingesting Gmail threads as support cases.

    Fetches threads with a specific label (default: 'supports') and
    converts them into searchable case examples.
    """

    def __init__(
        self,
        label: str | None = None,
        credentials_path: Path | None = None,
        token_path: Path | None = None,
        embeddings_client: EmbeddingsClient | None = None,
        vector_store: VectorStore | None = None,
        chunk_size: int = 1500,
        chunk_overlap: int = 150,
        max_threads: int | None = None,
        offset: int = 0,
    ) -> None:
        """Initialize the pipeline.

        Args:
            label: Gmail label to filter threads. Defaults to GMAIL_LABEL env var.
            credentials_path: Path to OAuth credentials.json.
            token_path: Path to store OAuth token.
            embeddings_client: Client for generating embeddings.
            vector_store: Client for storing vectors.
            chunk_size: Size of text chunks.
            chunk_overlap: Overlap between chunks.
            max_threads: Maximum number of threads to fetch (most recent first).
            offset: Number of threads to skip (for incremental ingestion).
        """
        settings = get_settings()
        self._label = label or settings.gmail_label
        self._credentials_path = credentials_path or settings.google_credentials_path
        self._token_path = token_path or settings.google_token_path
        self._embeddings = embeddings_client or EmbeddingsClient()
        self._vectorstore = vector_store or VectorStore()
        self._chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._max_threads = max_threads
        self._offset = offset
        self._service = None

    def _get_credentials(self) -> Credentials:
        """Get or refresh OAuth credentials.

        Returns:
            Valid credentials for Gmail API.
        """
        creds = None

        # Load existing token
        if self._token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self._token_path), SCOPES)

        # Refresh or get new credentials
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not self._credentials_path.exists():
                    raise FileNotFoundError(
                        f"OAuth credentials not found at {self._credentials_path}. "
                        "Download from Google Cloud Console."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self._credentials_path), SCOPES
                )
                creds = flow.run_local_server(port=0)

            # Save token for next run
            with open(self._token_path, "w") as token:
                token.write(creds.to_json())

        return creds

    def _get_service(self):
        """Get Gmail API service.

        Returns:
            Gmail API service instance.
        """
        if self._service is None:
            creds = self._get_credentials()
            self._service = build("gmail", "v1", credentials=creds)
        return self._service

    async def fetch(self) -> list[Document]:
        """Fetch Gmail threads with the configured label.

        Returns:
            List of documents from Gmail threads.
        """
        documents: list[Document] = []
        service = self._get_service()

        # Get label ID
        label_id = await self._get_label_id(service, self._label)
        if not label_id:
            logger.error("Label not found", label=self._label)
            return []

        logger.info("Fetching threads with label", label=self._label, label_id=label_id)

        # List threads with this label
        threads = []
        page_token = None

        while True:
            results = await run_sync(
                service.users()
                .threads()
                .list(userId="me", labelIds=[label_id], pageToken=page_token)
                .execute
            )

            threads.extend(results.get("threads", []))
            page_token = results.get("nextPageToken")

            if not page_token:
                break

        # Apply offset and limit (Gmail returns most recent first)
        total_available = len(threads)
        if self._offset or self._max_threads:
            start_idx = self._offset
            end_idx = (self._offset + self._max_threads) if self._max_threads else len(threads)
            threads = threads[start_idx:end_idx]
            logger.info(
                "Applied offset/limit",
                total_available=total_available,
                offset=self._offset,
                max_threads=self._max_threads,
                selected=len(threads),
            )

        logger.info("Found threads", count=len(threads))

        # Fetch full thread content with progress logging
        total = len(threads)
        for i, thread_info in enumerate(threads):
            if (i + 1) % 100 == 0 or i == 0:
                logger.info("Fetching threads progress", current=i + 1, total=total)
            try:
                doc = await self._fetch_thread(service, thread_info["id"])
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.warning("Failed to fetch thread", thread_id=thread_info["id"], error=str(e))

        logger.info("Fetched Gmail threads", count=len(documents))
        return documents

    async def _get_label_id(self, service, label_name: str) -> str | None:
        """Get label ID by name.

        Args:
            service: Gmail API service.
            label_name: Label name to find.

        Returns:
            Label ID or None if not found.
        """
        results = await run_sync(service.users().labels().list(userId="me").execute)
        labels = results.get("labels", [])

        for label in labels:
            if label["name"].lower() == label_name.lower():
                return label["id"]

        return None

    def _is_rma_case(self, subject: str, body: str) -> bool:
        """Check if a thread is an RMA (Return Merchandise Authorization) case.

        RMA cases are procedural and don't add value to the knowledge base.

        Args:
            subject: Email subject line.
            body: Combined email body text.

        Returns:
            True if the thread appears to be an RMA case.
        """
        # Check subject patterns
        for pattern in RMA_SUBJECT_PATTERNS:
            if pattern.search(subject):
                return True

        # Check body patterns - require at least one match
        for pattern in RMA_BODY_PATTERNS:
            if pattern.search(body):
                return True

        return False

    async def _fetch_thread(self, service, thread_id: str) -> Document | None:
        """Fetch a single thread and format as document.

        Args:
            service: Gmail API service.
            thread_id: Thread ID to fetch.

        Returns:
            Document with thread content.
        """
        thread = await run_sync(
            service.users().threads().get(userId="me", id=thread_id).execute
        )
        messages = thread.get("messages", [])

        if not messages:
            return None

        # Format thread as case study
        formatted_messages = []
        subject = ""
        full_body = ""

        for msg in messages:
            headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
            if not subject:
                subject = headers.get("Subject", "No Subject")

            sender = headers.get("From", "Unknown")
            date = headers.get("Date", "")

            # Extract body
            body = self._extract_body(msg["payload"])
            full_body += " " + body

            formatted_messages.append(
                f"**From:** {sender}\n**Date:** {date}\n\n{body}"
            )

        text = f"# Support Case: {subject}\n\n" + "\n\n---\n\n".join(formatted_messages)

        # Check if this is an RMA case
        is_rma = self._is_rma_case(subject, full_body)

        return Document(
            id=str(uuid.uuid4()),
            text=text,
            metadata={
                "source": "gmail",
                "thread_id": thread_id,
                "subject": subject,
                "message_count": len(messages),
                "is_rma": is_rma,
            },
        )

    def _extract_body(self, payload) -> str:
        """Extract text body from message payload.

        Args:
            payload: Message payload.

        Returns:
            Decoded text body.
        """
        raw_text = ""
        if "body" in payload and payload["body"].get("data"):
            raw_text = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8")
        elif "parts" in payload:
            for part in payload["parts"]:
                if part["mimeType"] == "text/plain" and part.get("body", {}).get("data"):
                    raw_text = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
                    break
                elif "parts" in part:
                    result = self._extract_body(part)
                    if result:
                        raw_text = result
                        break

        return self._clean_email_text(raw_text) if raw_text else ""

    def _clean_email_text(self, text: str) -> str:
        """Clean email text by removing noise.

        Args:
            text: Raw email text.

        Returns:
            Cleaned text.
        """
        lines = text.split("\n")
        cleaned_lines = []
        in_signature = False

        for line in lines:
            stripped = line.strip()

            # Skip signature markers and everything after
            if stripped.startswith("--") and len(stripped) <= 3:
                in_signature = True
                continue
            if in_signature:
                continue

            # Skip common signature/footer patterns
            skip_patterns = [
                r"^Sent from my",
                r"^Envoyé de mon",
                r"^Get Outlook for",
                r"^_+$",
                r"^\*{3,}",
                r"^This email and any",
                r"^Ce message et toutes",
                r"^Confidential",
                r"^CONFIDENTIAL",
            ]
            if any(re.match(p, stripped, re.IGNORECASE) for p in skip_patterns):
                continue

            # Skip quoted text (replies)
            if stripped.startswith(">"):
                continue

            # Skip email headers in replies
            if re.match(r"^(On|Le) .+ wrote:$", stripped):
                continue
            if re.match(r"^From:|^To:|^Cc:|^Date:|^Subject:", stripped):
                continue

            cleaned_lines.append(line)

        # Remove excessive blank lines
        result = "\n".join(cleaned_lines)
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()

    async def process(self, documents: list[Document]) -> list[Document]:
        """Chunk documents for embedding.

        Args:
            documents: Raw documents.

        Returns:
            Chunked documents.
        """
        processed: list[Document] = []

        for doc in documents:
            chunks = self._chunker.chunk_text(doc.text)

            for chunk in chunks:
                # Generate deterministic UUID v5 for chunk ID
                chunk_id = str(
                    uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc.id}_{chunk.chunk_index}")
                )
                processed.append(
                    Document(
                        id=chunk_id,
                        text=chunk.text,
                        metadata={
                            **doc.metadata,
                            "chunk_index": chunk.chunk_index,
                            "parent_id": doc.id,
                        },
                    )
                )

        logger.info("Processed threads into chunks", original=len(documents), chunks=len(processed))
        return processed

    async def ingest(self, documents: list[Document]) -> int:
        """Embed and store documents.

        Args:
            documents: Processed documents.

        Returns:
            Number of documents ingested.
        """
        if not documents:
            return 0

        # Ensure collection exists
        await self._vectorstore.ensure_collections()

        # Generate embeddings in batches
        texts = [doc.text for doc in documents]
        result = await self._embeddings.embed_batch(texts, input_type="document")

        logger.info("Generated embeddings", count=len(result.vectors), tokens=result.total_tokens)

        # Store in Qdrant
        ids = [doc.id for doc in documents]
        metadata = [doc.metadata for doc in documents]

        await self._vectorstore.upsert_documents(
            collection=self._vectorstore.cases_collection,
            texts=texts,
            vectors=result.vectors,
            metadata=metadata,
            ids=ids,
        )

        logger.info("Ingested Gmail threads into Qdrant", count=len(documents))
        return len(documents)
