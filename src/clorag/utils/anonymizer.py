"""Text anonymization utilities for removing PII from support cases."""

import re
from dataclasses import dataclass, field


@dataclass
class AnonymizationContext:
    """Track anonymization state within a single case/thread.

    Maintains consistent replacement mappings so the same entity
    gets the same placeholder throughout a document.
    """

    _serial_counter: int = 0
    _email_counter: int = 0
    _phone_counter: int = 0
    _serial_map: dict[str, str] = field(default_factory=dict)
    _email_map: dict[str, str] = field(default_factory=dict)
    _phone_map: dict[str, str] = field(default_factory=dict)

    def get_serial_placeholder(self, serial: str, device_type: str) -> str:
        """Get consistent placeholder for a serial number."""
        if serial not in self._serial_map:
            self._serial_counter += 1
            self._serial_map[serial] = f"[SERIAL:{device_type}-{self._serial_counter}]"
        return self._serial_map[serial]

    def get_email_placeholder(self, email: str) -> str:
        """Get consistent placeholder for an email address."""
        if email not in self._email_map:
            self._email_counter += 1
            self._email_map[email] = f"[EMAIL-{self._email_counter}]"
        return self._email_map[email]

    def get_phone_placeholder(self, phone: str) -> str:
        """Get consistent placeholder for a phone number."""
        if phone not in self._phone_map:
            self._phone_counter += 1
            self._phone_map[phone] = f"[PHONE-{self._phone_counter}]"
        return self._phone_map[phone]


class TextAnonymizer:
    """Anonymize text by replacing PII with placeholder tokens.

    Handles:
    - Cyanview serial numbers (CY-RCP-48-12, CY-CI0-4-1, etc.)
    - Email addresses (preserves @cyanview.com)
    - Phone numbers (various international formats)
    """

    # Cyanview serial number pattern: CY-{TYPE}-{NUM}-{NUM}
    # Types: RCP, CI0, RIO, CVP, VP4, etc. (may include numbers like CI0)
    SERIAL_PATTERN = re.compile(
        r"\b(CY-)([A-Z][A-Z0-9]{1,3})(-\d{1,3}-\d{1,3})\b",
        re.IGNORECASE,
    )

    # Email pattern - matches standard email addresses
    EMAIL_PATTERN = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    )

    # Phone pattern - various international formats
    PHONE_PATTERN = re.compile(
        r"(?:\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}"
    )

    # Cyanview email domain to preserve
    CYANVIEW_DOMAIN = "@cyanview.com"

    def __init__(self) -> None:
        """Initialize the anonymizer."""
        pass

    def anonymize(
        self,
        text: str,
        context: AnonymizationContext | None = None,
    ) -> tuple[str, AnonymizationContext]:
        """Anonymize PII in text.

        Args:
            text: The text to anonymize.
            context: Optional context for consistent replacements.
                     If None, a new context is created.

        Returns:
            Tuple of (anonymized_text, context).
        """
        if context is None:
            context = AnonymizationContext()

        result = text

        # Anonymize serial numbers
        result = self._anonymize_serials(result, context)

        # Anonymize emails (but preserve cyanview.com emails)
        result = self._anonymize_emails(result, context)

        # Anonymize phone numbers
        result = self._anonymize_phones(result, context)

        return result, context

    def _anonymize_serials(
        self,
        text: str,
        context: AnonymizationContext,
    ) -> str:
        """Replace Cyanview serial numbers with placeholders."""
        def replace_serial(match: re.Match[str]) -> str:
            full_serial = match.group(0).upper()
            device_type = match.group(2).upper()
            return context.get_serial_placeholder(full_serial, device_type)

        return self.SERIAL_PATTERN.sub(replace_serial, text)

    def _anonymize_emails(
        self,
        text: str,
        context: AnonymizationContext,
    ) -> str:
        """Replace email addresses with placeholders, preserving Cyanview emails."""
        def replace_email(match: re.Match[str]) -> str:
            email = match.group(0)
            # Preserve Cyanview emails
            if email.lower().endswith(self.CYANVIEW_DOMAIN):
                return email
            return context.get_email_placeholder(email.lower())

        return self.EMAIL_PATTERN.sub(replace_email, text)

    def _anonymize_phones(
        self,
        text: str,
        context: AnonymizationContext,
    ) -> str:
        """Replace phone numbers with placeholders.

        Only matches sequences that look like phone numbers
        (at least 7 digits to avoid false positives).
        """
        def replace_phone(match: re.Match[str]) -> str:
            phone = match.group(0)
            # Only replace if it has at least 7 digits (to avoid version numbers, etc.)
            digits = re.sub(r"\D", "", phone)
            if len(digits) >= 7:
                return context.get_phone_placeholder(phone)
            return phone

        return self.PHONE_PATTERN.sub(replace_phone, text)

    def anonymize_batch(
        self,
        texts: list[str],
        context: AnonymizationContext | None = None,
    ) -> tuple[list[str], AnonymizationContext]:
        """Anonymize multiple texts with shared context.

        Useful for anonymizing all messages in a thread while
        maintaining consistent placeholders.

        Args:
            texts: List of texts to anonymize.
            context: Optional shared context.

        Returns:
            Tuple of (anonymized_texts, context).
        """
        if context is None:
            context = AnonymizationContext()

        anonymized = []
        for text in texts:
            anon_text, context = self.anonymize(text, context)
            anonymized.append(anon_text)

        return anonymized, context


def clean_thread_quotes(text: str, remove_signatures: bool = True) -> str:
    """Remove quoted reply history and signatures from email thread content.

    Cleans redundant content like:
    - Lines starting with > (quoted text)
    - "On [date], [person] wrote:" reply headers
    - "---------- Original Message ----------" separators
    - "From: ... Sent: ... To: ... Subject:" forwarded headers
    - Gmail-style "Le [date] à [time], [person] a écrit :" (French)
    - Email signatures (-- separator, "Best regards", "Sent from my iPhone", etc.)

    Args:
        text: Raw email thread text.
        remove_signatures: Whether to remove email signatures.

    Returns:
        Cleaned text with quoted content and signatures removed.
    """
    lines = text.split("\n")
    cleaned_lines: list[str] = []
    skip_until_blank = False
    in_quoted_block = False
    in_signature = False

    # Patterns for reply headers
    reply_header_patterns = [
        # English: "On Jan 15, 2025, at 10:30 AM, User wrote:"
        re.compile(r"^On .+wrote:\s*$", re.IGNORECASE),
        # English: "On 15/01/2025 10:30, User wrote:"
        re.compile(r"^On \d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}.+wrote:\s*$", re.IGNORECASE),
        # French: "Le 15 janv. 2025 à 10:30, User a écrit :"
        re.compile(r"^Le .+a écrit\s*:\s*$", re.IGNORECASE),
        # German: "Am 15.01.2025 um 10:30 schrieb User:"
        re.compile(r"^Am .+schrieb.+:\s*$", re.IGNORECASE),
        # Outlook style: "From: User Sent: Monday..."
        re.compile(r"^From:\s*.+$", re.IGNORECASE),
    ]

    # Patterns for separators
    separator_patterns = [
        re.compile(r"^-{3,}\s*(Original Message|Forwarded|Begin forwarded)", re.IGNORECASE),
        re.compile(r"^_{3,}\s*$"),
        re.compile(r"^={3,}\s*$"),
        re.compile(r"^\*{3,}\s*$"),
    ]

    # Patterns for signature starts
    signature_patterns = [
        # Standard signature separator
        re.compile(r"^--\s*$"),
        # Common sign-offs
        re.compile(r"^(Best|Kind|Warm)?\s*regards?,?\s*$", re.IGNORECASE),
        re.compile(r"^(Many\s+)?thanks?,?\s*$", re.IGNORECASE),
        re.compile(r"^Cheers,?\s*$", re.IGNORECASE),
        re.compile(r"^Sincerely,?\s*$", re.IGNORECASE),
        re.compile(r"^Cordialement,?\s*$", re.IGNORECASE),  # French
        re.compile(r"^Mit freundlichen Grüßen,?\s*$", re.IGNORECASE),  # German
        # Mobile signatures
        re.compile(r"^Sent from my (iPhone|iPad|Android|Galaxy|Pixel)", re.IGNORECASE),
        re.compile(r"^Envoyé de mon (iPhone|iPad)", re.IGNORECASE),  # French
        re.compile(r"^Get Outlook for", re.IGNORECASE),
    ]

    # Patterns that indicate a new message is starting (reset signature state)
    new_message_patterns = [
        re.compile(r"^-{3,}\s*Message\s*\d*\s*-{3,}$", re.IGNORECASE),
        re.compile(r"^-{3,}\s*$"),  # Simple separator
    ]

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Check if this looks like a new message boundary
        is_new_message = any(p.match(stripped) for p in new_message_patterns)
        if is_new_message:
            # Reset all states for new message
            in_signature = False
            in_quoted_block = False
            skip_until_blank = False
            # Keep the separator for context
            cleaned_lines.append(line)
            continue

        # Skip empty lines in quoted blocks or signatures
        if (in_quoted_block or in_signature) and not stripped:
            continue

        # Detect quoted lines (starting with >)
        if stripped.startswith(">"):
            in_quoted_block = True
            continue

        # End quoted block on non-quoted, non-empty line
        if in_quoted_block and stripped and not stripped.startswith(">"):
            in_quoted_block = False

        # Check for signature starts (only if enabled)
        if remove_signatures and not in_signature:
            is_signature_start = any(p.match(stripped) for p in signature_patterns)
            if is_signature_start:
                in_signature = True
                continue

        # End signature on reply header (new message starts)
        if in_signature:
            is_reply_header = any(p.match(stripped) for p in reply_header_patterns)
            if is_reply_header:
                in_signature = False
                skip_until_blank = True
                continue
            # Skip signature content
            continue

        # Check for reply headers
        is_reply_header = any(p.match(stripped) for p in reply_header_patterns)
        if is_reply_header:
            skip_until_blank = True
            continue

        # Check for separators (Original Message, Forwarded, etc.)
        is_separator = any(p.match(stripped) for p in separator_patterns)
        if is_separator:
            skip_until_blank = True
            continue

        # Skip lines after header until blank line
        if skip_until_blank:
            if not stripped:
                skip_until_blank = False
            continue

        # Keep non-quoted, non-header, non-signature lines
        cleaned_lines.append(line)

    # Clean up excessive blank lines
    result = "\n".join(cleaned_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()
