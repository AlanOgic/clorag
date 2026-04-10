#!/usr/bin/env python3
"""Enrich support_cases with Gmail dates and raw thread content.

Re-fetches the 154 existing threads from Gmail API (read-only) and updates
the SQLite support_cases table with:
- created_at: date of the first message in the thread
- raw_thread: full anonymized email content

No re-ingestion, no Qdrant writes, no Sonnet calls.
"""

from __future__ import annotations

import asyncio
import base64
import re
from datetime import datetime
from email.utils import parsedate_to_datetime
from functools import partial
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from clorag.config import get_settings
from clorag.core.support_case_db import get_support_case_database
from clorag.utils.logger import get_logger
from clorag.utils.token_encryption import load_encrypted_token, save_encrypted_token

logger = get_logger(__name__)

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def _get_credentials() -> Credentials:
    """Get or refresh OAuth credentials."""
    settings = get_settings()
    token_path = Path(settings.google_token_path)
    credentials_path = Path(settings.google_credentials_path)

    creds = None
    token_data = load_encrypted_token(token_path)
    if token_data:
        creds = Credentials.from_authorized_user_info(token_data, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not credentials_path.exists():
                raise FileNotFoundError(
                    f"OAuth credentials not found at {credentials_path}. "
                    "Download from Google Cloud Console."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(credentials_path), SCOPES
            )
            creds = flow.run_local_server(port=0)

        import json

        token_data = json.loads(creds.to_json())
        save_encrypted_token(token_path, token_data)

    return creds


def _extract_body(payload: dict[str, Any]) -> str:
    """Extract text body from message payload."""
    raw_text = ""
    if "body" in payload and payload["body"].get("data"):
        raw_text = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8")
    elif "parts" in payload:
        for part in payload["parts"]:
            if part["mimeType"] == "text/plain" and part.get("body", {}).get("data"):
                raw_text = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
                break
            elif "parts" in part:
                result = _extract_body(part)
                if result:
                    raw_text = result
                    break
    return raw_text


def _clean_email_text(text: str) -> str:
    """Clean email text by removing noise (signatures, quoted text, headers)."""
    lines = text.split("\n")
    cleaned_lines = []
    in_signature = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("--") and len(stripped) <= 3:
            in_signature = True
            continue
        if in_signature:
            continue

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

        if stripped.startswith(">"):
            continue

        if re.match(r"^(On|Le) .+ wrote:$", stripped):
            continue
        if re.match(r"^From:|^To:|^Cc:|^Date:|^Subject:", stripped):
            continue

        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _parse_date(date_str: str) -> datetime | None:
    """Parse an email Date header into a datetime."""
    if not date_str:
        return None
    try:
        return parsedate_to_datetime(date_str)
    except (ValueError, TypeError):
        pass
    # Fallback: try ISO format
    try:
        return datetime.fromisoformat(date_str)
    except (ValueError, TypeError):
        return None


async def enrich() -> None:
    """Fetch dates and raw content from Gmail for existing support cases."""
    db = get_support_case_database()
    cases, total = db.list_cases(limit=1000, offset=0)
    thread_ids = {c.thread_id: c.id for c in cases}

    if not thread_ids:
        print("No support cases found in database.")
        return

    missing_date = sum(1 for c in cases if c.created_at is None)
    missing_raw = sum(1 for c in cases if not c.raw_thread)
    print(f"Support cases: {len(cases)}")
    print(f"  Missing created_at: {missing_date}")
    print(f"  Missing raw_thread: {missing_raw}")

    if missing_date == 0 and missing_raw == 0:
        print("All cases already enriched. Nothing to do.")
        return

    # Connect to Gmail
    creds = _get_credentials()
    service = build("gmail", "v1", credentials=creds)
    loop = asyncio.get_event_loop()

    enriched = 0
    failed = 0
    skipped = 0

    for i, (thread_id, case_id) in enumerate(thread_ids.items()):
        case = next(c for c in cases if c.thread_id == thread_id)

        # Skip if already complete
        if case.created_at is not None and case.raw_thread:
            skipped += 1
            continue

        if (i + 1) % 25 == 0 or i == 0:
            print(f"  Processing {i + 1}/{len(thread_ids)}...")

        try:
            thread = await loop.run_in_executor(
                None,
                partial(
                    service.users().threads().get(userId="me", id=thread_id).execute
                ),
            )
            messages = thread.get("messages", [])

            if not messages:
                failed += 1
                logger.warning("Thread has no messages", thread_id=thread_id)
                continue

            # Extract date from first message
            first_headers = {
                h["name"]: h["value"] for h in messages[0]["payload"]["headers"]
            }
            created_at = _parse_date(first_headers.get("Date", ""))

            # Build raw thread content
            formatted_messages = []
            for msg in messages:
                headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
                sender = headers.get("From", "Unknown")
                date = headers.get("Date", "")
                body = _extract_body(msg["payload"])
                cleaned = _clean_email_text(body) if body else ""
                formatted_messages.append(
                    f"**From:** {sender}\n**Date:** {date}\n\n{cleaned}"
                )

            subject = first_headers.get("Subject", case.subject)
            raw_thread = (
                f"# Support Case: {subject}\n\n"
                + "\n\n---\n\n".join(formatted_messages)
            )

            # Update SQLite directly (minimal update, no upsert overhead)
            with db._cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE support_cases
                    SET created_at = COALESCE(?, created_at),
                        raw_thread = COALESCE(?, raw_thread),
                        updated_at = datetime('now')
                    WHERE thread_id = ?
                    """,
                    (
                        created_at.isoformat() if created_at else None,
                        raw_thread if not case.raw_thread else None,
                        thread_id,
                    ),
                )

            enriched += 1

        except Exception as e:
            failed += 1
            logger.warning(
                "Failed to enrich thread", thread_id=thread_id, error=str(e)
            )

    print("\nEnrichment complete:")
    print(f"  Enriched: {enriched}")
    print(f"  Skipped (already complete): {skipped}")
    print(f"  Failed: {failed}")


def main() -> None:
    """Entry point for enrich-support-cases CLI."""
    asyncio.run(enrich())


if __name__ == "__main__":
    main()
