"""RAG-based response generator for support draft emails."""

import anthropic

from clorag.config import get_settings
from clorag.core.embeddings import EmbeddingsClient
from clorag.core.sparse_embeddings import SparseEmbeddingsClient
from clorag.core.vectorstore import VectorStore
from clorag.drafts.models import DraftPreview
from clorag.utils.logger import get_logger

logger = get_logger(__name__)

# System prompt for generating email draft replies
DRAFT_SYSTEM_PROMPT = """You are drafting a professional support email reply for Cyanview, specialists in broadcast camera control solutions.

CONTEXT: You will receive:
1. The customer's problem summary
2. The original email thread for context
3. Retrieved documentation and past support cases that may help

YOUR TASK: Write a helpful, professional email reply that addresses the customer's issue.

EMAIL STRUCTURE:
1. **Greeting**: Start with "Hello," or "Bonjour," (match the customer's language)
2. **Acknowledgment**: Briefly acknowledge their issue
3. **Solution**: Provide clear, actionable steps or information
   - Use numbered steps for procedures
   - Use bullet points for lists of options
   - Include specific settings, values, or commands when relevant
4. **Closing**: Offer further assistance and sign off professionally

FORMAT RULES:
- **Bold** product names: RCP, RIO, CI0, VP4, CVP, etc.
- Use code formatting for: IP addresses, firmware versions, menu paths, commands
- Keep paragraphs short and scannable
- Total length: 150-400 words (adapt to complexity)

CONTENT RULES:
- Use ONLY information from the provided context - never invent details
- Be specific and technical when the context supports it
- If the context doesn't fully answer, acknowledge what you can help with and offer to investigate further
- Never promise features or behaviors not mentioned in the context

TONE:
- Professional but warm
- Confident but not dismissive
- Helpful and solution-focused

SIGN OFF:
End with:
"Best regards,
Cyanview Support"

Match the customer's language (English or French based on their message)."""


class DraftResponseGenerator:
    """Generate support email draft responses using RAG.

    Uses hybrid search (dense + sparse vectors with RRF fusion) to retrieve
    relevant context, then synthesizes a professional email response.
    """

    def __init__(
        self,
        embeddings: EmbeddingsClient | None = None,
        sparse_embeddings: SparseEmbeddingsClient | None = None,
        vectorstore: VectorStore | None = None,
    ) -> None:
        """Initialize the generator.

        Args:
            embeddings: Dense embeddings client (Voyage AI).
            sparse_embeddings: Sparse embeddings client (BM25).
            vectorstore: Qdrant vector store.
        """
        self._embeddings = embeddings or EmbeddingsClient()
        self._sparse_embeddings = sparse_embeddings or SparseEmbeddingsClient()
        self._vectorstore = vectorstore or VectorStore()
        settings = get_settings()
        self._client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value()
        )
        self._model = settings.haiku_model

    async def generate_draft(
        self,
        problem_summary: str,
        thread_content: str,
        subject: str,
        to_address: str,
        thread_id: str,
    ) -> DraftPreview:
        """Generate a draft email response using RAG.

        Args:
            problem_summary: Summary of the customer's problem (used as search query).
            thread_content: Full thread content for context.
            subject: Email subject line.
            to_address: Customer's email address.
            thread_id: Gmail thread ID.

        Returns:
            DraftPreview with generated content and sources.
        """
        logger.info(
            "Generating draft response",
            thread_id=thread_id,
            subject=subject[:50],
        )

        # Perform hybrid search using problem summary as query
        chunks, sources = await self._search_context(problem_summary)

        # Calculate confidence based on search results
        confidence = self._calculate_confidence(chunks)

        # Generate email response
        content = await self._synthesize_response(
            problem_summary=problem_summary,
            thread_content=thread_content,
            chunks=chunks,
        )

        return DraftPreview(
            thread_id=thread_id,
            subject=subject,
            to_address=to_address,
            content=content,
            sources=sources,
            confidence=confidence,
            problem_summary=problem_summary,
        )

    async def _search_context(
        self,
        query: str,
        limit: int = 8,
    ) -> tuple[list[dict], list[dict]]:
        """Perform hybrid search for relevant context.

        Args:
            query: Search query (typically the problem summary).
            limit: Maximum number of results.

        Returns:
            Tuple of (chunks for synthesis, source links).
        """
        # Generate embeddings
        dense_vector = await self._embeddings.embed_query(query)
        sparse_vector = self._sparse_embeddings.embed_query(query)

        # Hybrid search across both collections
        results = await self._vectorstore.hybrid_search_rrf(
            dense_vector, sparse_vector, limit=limit
        )

        chunks = []
        sources = []
        seen_sources: set[str] = set()

        for item in results:
            source_type = item.payload.get("_source", "unknown")

            if source_type == "documentation":
                chunks.append({
                    "text": item.payload.get("text", ""),
                    "source_type": "documentation",
                    "url": item.payload.get("url"),
                    "title": item.payload.get("title", "Documentation"),
                })
                url = item.payload.get("url", "")
                if url and url not in seen_sources:
                    seen_sources.add(url)
                    sources.append({
                        "title": item.payload.get("title", "Documentation"),
                        "url": url,
                        "source_type": "documentation",
                    })
            else:
                chunks.append({
                    "text": item.payload.get("text", ""),
                    "source_type": "gmail_case",
                    "subject": item.payload.get("subject", "Support Case"),
                })
                subject = item.payload.get("subject", "")
                key = f"case:{subject}"
                if key not in seen_sources:
                    seen_sources.add(key)
                    sources.append({
                        "title": subject or "Support Case",
                        "url": None,
                        "source_type": "gmail_case",
                    })

        logger.info(
            "Retrieved context for draft",
            chunks=len(chunks),
            sources=len(sources),
        )

        return chunks, sources[:5]  # Limit to 5 sources

    def _build_context(self, chunks: list[dict], max_chunks: int = 8) -> str:
        """Build context string from chunks for Claude synthesis."""
        parts = []
        for i, chunk in enumerate(chunks[:max_chunks], 1):
            text = chunk.get("text", "")[:2000]
            if chunk.get("source_type") == "documentation":
                parts.append(f"[{i} Doc: {chunk.get('url', '')}]\n{text}")
            else:
                parts.append(f"[{i} Case: {chunk.get('subject', 'Support')}]\n{text}")
        return "\n---\n".join(parts)

    async def _synthesize_response(
        self,
        problem_summary: str,
        thread_content: str,
        chunks: list[dict],
    ) -> str:
        """Synthesize an email response using Claude.

        Args:
            problem_summary: Summary of the customer's problem.
            thread_content: Original thread content (truncated).
            chunks: Retrieved context chunks.

        Returns:
            Generated email response.
        """
        if not chunks:
            # Fallback response when no context is found
            return """Hello,

Thank you for reaching out to Cyanview Support.

I've received your message and I'm looking into your question. I'll need to investigate this further to provide you with the most accurate information.

I'll get back to you shortly with more details.

Best regards,
Cyanview Support"""

        context = self._build_context(chunks)

        # Truncate thread content to avoid token limits
        thread_excerpt = thread_content[:3000] if len(thread_content) > 3000 else thread_content

        user_content = f"""Problem Summary: {problem_summary}

Original Email Thread:
{thread_excerpt}

Retrieved Context (documentation and past support cases):
{context}

Write a professional email reply addressing this customer's issue. Use the retrieved context to provide accurate, helpful information."""

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=1000,
            system=DRAFT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )

        return response.content[0].text

    def _calculate_confidence(self, chunks: list[dict]) -> float:
        """Calculate confidence score based on retrieved chunks.

        Args:
            chunks: Retrieved context chunks.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        if not chunks:
            return 0.0

        # Base confidence on number and quality of results
        num_chunks = len(chunks)

        if num_chunks >= 6:
            return 0.9
        elif num_chunks >= 4:
            return 0.75
        elif num_chunks >= 2:
            return 0.6
        elif num_chunks >= 1:
            return 0.4
        else:
            return 0.2
