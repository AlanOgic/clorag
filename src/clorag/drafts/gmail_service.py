"""Gmail service with draft creation capability."""

import asyncio
import base64
from collections.abc import Callable
from email.message import EmailMessage
from email.utils import parsedate_to_datetime
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from clorag.config import get_settings
from clorag.drafts.models import DraftResult, PendingThread, ThreadDetail, ThreadMessage
from clorag.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# Scopes required for draft creation (upgrade from readonly)
DRAFT_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]


async def run_sync(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a synchronous function in the default executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))


class GmailDraftService:
    """Gmail service for reading threads and creating draft replies.

    This service extends the basic Gmail functionality to support
    creating draft replies in existing threads.
    """

    def __init__(
        self,
        credentials_path: Path | None = None,
        token_path: Path | None = None,
        from_email: str | None = None,
    ) -> None:
        """Initialize the Gmail draft service.

        Args:
            credentials_path: Path to OAuth credentials.json.
            token_path: Path to store OAuth token with compose scope.
            from_email: Email address for draft sender.
        """
        settings = get_settings()
        self._credentials_path = credentials_path or settings.google_credentials_path
        self._token_path = token_path or settings.draft_token_path
        self._from_email = from_email or settings.draft_from_email
        self._service = None

    def _get_credentials(self) -> Credentials:
        """Get or refresh OAuth credentials with compose scope.

        Returns:
            Valid credentials for Gmail API with draft creation.
        """
        creds = None

        # Load existing token
        if self._token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self._token_path), DRAFT_SCOPES)

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
                    str(self._credentials_path), DRAFT_SCOPES
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

    async def get_thread(self, thread_id: str) -> dict | None:
        """Fetch a single thread by ID.

        Args:
            thread_id: Gmail thread ID.

        Returns:
            Thread data or None if not found.
        """
        service = self._get_service()
        try:
            return await run_sync(
                service.users().threads().get(userId="me", id=thread_id).execute
            )
        except Exception as e:
            logger.warning("Failed to fetch thread", thread_id=thread_id, error=str(e))
            return None

    async def get_threads_by_label(
        self,
        label: str,
        max_results: int = 100,
    ) -> list[dict]:
        """Fetch threads with a specific label.

        Args:
            label: Gmail label name.
            max_results: Maximum number of threads to fetch.

        Returns:
            List of thread data.
        """
        service = self._get_service()

        # Get label ID
        label_id = await self._get_label_id(label)
        if not label_id:
            logger.error("Label not found", label=label)
            return []

        # List threads with this label
        threads = []
        page_token = None

        while len(threads) < max_results:
            results = await run_sync(
                service.users()
                .threads()
                .list(
                    userId="me",
                    labelIds=[label_id],
                    pageToken=page_token,
                    maxResults=min(100, max_results - len(threads)),
                )
                .execute
            )

            threads.extend(results.get("threads", []))
            page_token = results.get("nextPageToken")

            if not page_token:
                break

        return threads[:max_results]

    async def _get_label_id(self, label_name: str) -> str | None:
        """Get label ID by name.

        Args:
            label_name: Label name to find.

        Returns:
            Label ID or None if not found.
        """
        service = self._get_service()
        results = await run_sync(service.users().labels().list(userId="me").execute)
        labels = results.get("labels", [])

        for label in labels:
            if label["name"].lower() == label_name.lower():
                return label["id"]

        return None

    async def get_draft_thread_ids(self) -> set[str]:
        """Get thread IDs that already have drafts.

        Returns:
            Set of thread IDs with existing drafts.
        """
        service = self._get_service()
        thread_ids: set[str] = set()

        try:
            # List all drafts
            results = await run_sync(service.users().drafts().list(userId="me").execute)
            drafts = results.get("drafts", [])

            # Get thread ID for each draft
            for draft in drafts:
                try:
                    full_draft = await run_sync(
                        service.users()
                        .drafts()
                        .get(userId="me", id=draft["id"])
                        .execute
                    )
                    thread_id = full_draft.get("message", {}).get("threadId")
                    if thread_id:
                        thread_ids.add(thread_id)
                except Exception as e:
                    logger.warning(
                        "Failed to get draft thread ID",
                        draft_id=draft["id"],
                        error=str(e),
                    )

            logger.info("Found existing draft thread IDs", count=len(thread_ids))

        except Exception as e:
            logger.error("Failed to list drafts", error=str(e))

        return thread_ids

    def extract_thread_info(self, thread: dict) -> PendingThread | None:
        """Extract relevant info from a thread for draft creation.

        Args:
            thread: Full thread data from Gmail API.

        Returns:
            PendingThread with extracted information.
        """
        messages = thread.get("messages", [])
        if not messages:
            return None

        # Get info from the last message (the one we're replying to)
        last_message = messages[-1]
        headers = {h["name"].lower(): h["value"] for h in last_message["payload"]["headers"]}

        # Extract subject (from first message)
        first_headers = {h["name"].lower(): h["value"] for h in messages[0]["payload"]["headers"]}
        subject = first_headers.get("subject", "No Subject")

        # Get the customer's email (From header of last message)
        from_address = headers.get("from", "")
        # Extract just the email from "Name <email@domain.com>" format
        if "<" in from_address and ">" in from_address:
            from_address = from_address.split("<")[1].split(">")[0]

        # Get message ID for In-Reply-To header
        message_id = headers.get("message-id", "")

        # Parse date
        received_at = None
        date_str = headers.get("date", "")
        if date_str:
            try:
                received_at = parsedate_to_datetime(date_str)
            except Exception:
                pass

        return PendingThread(
            thread_id=thread["id"],
            subject=subject,
            from_address=from_address,
            last_message_id=message_id,
            message_count=len(messages),
            received_at=received_at,
            snippet=thread.get("snippet", "")[:200],
        )

    def extract_thread_content(self, thread: dict) -> str:
        """Extract full text content from a thread.

        Args:
            thread: Full thread data from Gmail API.

        Returns:
            Formatted thread content as string.
        """
        messages = thread.get("messages", [])
        if not messages:
            return ""

        formatted_messages = []

        for msg in messages:
            headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
            sender = headers.get("From", "Unknown")
            date = headers.get("Date", "")
            body = self._extract_body(msg["payload"])

            formatted_messages.append(
                f"**From:** {sender}\n**Date:** {date}\n\n{body}"
            )

        return "\n\n---\n\n".join(formatted_messages)

    def _extract_body(self, payload: dict) -> str:
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

        return raw_text.strip() if raw_text else ""

    def extract_thread_detail(self, thread: dict) -> ThreadDetail | None:
        """Extract full thread details with individual messages.

        Args:
            thread: Full thread data from Gmail API.

        Returns:
            ThreadDetail with all messages and metadata.
        """
        messages = thread.get("messages", [])
        if not messages:
            return None

        thread_id = thread["id"]
        gmail_link = f"https://mail.google.com/mail/u/0/#inbox/{thread_id}"

        # Get subject from first message
        first_headers = {
            h["name"].lower(): h["value"]
            for h in messages[0]["payload"]["headers"]
        }
        subject = first_headers.get("subject", "No Subject")

        # Cyanview email patterns (case insensitive check)
        cyanview_patterns = ["@cyanview.com", "cyanview"]

        thread_messages: list[ThreadMessage] = []

        for msg in messages:
            headers = {
                h["name"].lower(): h["value"]
                for h in msg["payload"]["headers"]
            }

            # Parse sender - handle Google Groups "via" format
            # Priority: X-Original-Sender > Reply-To > From (parsed for "via")
            from_raw = headers.get("from", "Unknown")
            from_name = ""
            from_address = from_raw

            # Check for X-Original-Sender (Google Groups)
            original_sender = headers.get("x-original-sender", "")
            reply_to = headers.get("reply-to", "")

            if original_sender:
                # X-Original-Sender contains the real sender email
                from_address = original_sender
                # Try to get name from From header
                if "<" in from_raw:
                    from_name = from_raw.split("<")[0].strip().strip('"')
                    # Remove "via ..." suffix from name
                    if " via " in from_name.lower():
                        from_name = from_name.lower().split(" via ")[0].strip()
                        from_name = from_raw.split("<")[0].strip().strip('"')
                        via_idx = from_name.lower().find(" via ")
                        if via_idx > 0:
                            from_name = from_name[:via_idx].strip()
            elif reply_to and "@cyanview.com" not in reply_to.lower():
                # Reply-To contains customer email (if not cyanview)
                if "<" in reply_to and ">" in reply_to:
                    from_address = reply_to.split("<")[1].split(">")[0]
                    reply_name = reply_to.split("<")[0].strip().strip('"')
                    if reply_name:
                        from_name = reply_name
                else:
                    from_address = reply_to
                # If no name from Reply-To, try From header
                if not from_name and "<" in from_raw:
                    from_name = from_raw.split("<")[0].strip().strip('"')
                    # Remove "via ..." suffix
                    via_idx = from_name.lower().find(" via ")
                    if via_idx > 0:
                        from_name = from_name[:via_idx].strip()
            elif "<" in from_raw and ">" in from_raw:
                # Standard From header parsing
                from_name = from_raw.split("<")[0].strip().strip('"')
                from_address = from_raw.split("<")[1].split(">")[0]

                # Handle "Name via Group <group@email>" format
                if " via " in from_name.lower():
                    via_idx = from_name.lower().find(" via ")
                    from_name = from_name[:via_idx].strip()

            # Check if message is from Cyanview (check all possible sender indicators)
            sender_indicators = [from_address, original_sender, from_raw]
            is_cyanview = any(
                pattern.lower() in indicator.lower()
                for pattern in cyanview_patterns
                for indicator in sender_indicators
                if indicator
            )

            # Parse date
            date_parsed = None
            date_str = headers.get("date", "")
            if date_str:
                try:
                    date_parsed = parsedate_to_datetime(date_str)
                except Exception:
                    pass

            # Extract body
            body = self._extract_body(msg["payload"])

            # Get message ID
            message_id = headers.get("message-id", msg.get("id", ""))

            # Snippet from Gmail
            snippet = msg.get("snippet", "")[:200]

            thread_messages.append(ThreadMessage(
                message_id=message_id,
                from_address=from_address,
                from_name=from_name if from_name != from_address else "",
                date=date_parsed,
                snippet=snippet,
                body=body,
                is_cyanview=is_cyanview,
            ))

        return ThreadDetail(
            thread_id=thread_id,
            subject=subject,
            gmail_link=gmail_link,
            messages=thread_messages,
        )

    async def create_draft_reply(
        self,
        thread_id: str,
        original_message_id: str,
        to_address: str,
        subject: str,
        content: str,
    ) -> DraftResult:
        """Create a draft reply in an existing thread.

        Args:
            thread_id: Gmail thread ID to reply to.
            original_message_id: Message-ID header of the message being replied to.
            to_address: Recipient email address.
            subject: Email subject (will be prefixed with Re: if needed).
            content: Email body content.

        Returns:
            DraftResult with draft information.
        """
        service = self._get_service()

        # Build the email message
        message = EmailMessage()
        message.set_content(content)
        message["To"] = to_address
        message["From"] = self._from_email
        message["Subject"] = f"Re: {subject}" if not subject.startswith("Re:") else subject

        # Required headers for threading
        if original_message_id:
            message["In-Reply-To"] = original_message_id
            message["References"] = original_message_id

        # Encode message
        encoded = base64.urlsafe_b64encode(message.as_bytes()).decode()

        # Create draft
        draft_body = {
            "message": {
                "raw": encoded,
                "threadId": thread_id,
            }
        }

        logger.info(
            "Creating draft reply",
            thread_id=thread_id,
            to=to_address,
            subject=subject[:50],
        )

        result = await run_sync(
            service.users().drafts().create(userId="me", body=draft_body).execute
        )

        draft_id = result.get("id", "")

        logger.info(
            "Draft created successfully",
            draft_id=draft_id,
            thread_id=thread_id,
        )

        return DraftResult(
            thread_id=thread_id,
            draft_id=draft_id,
            subject=subject,
            to_address=to_address,
        )
