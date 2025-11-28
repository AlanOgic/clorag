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

    def __init__(self):
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
        def replace_serial(match: re.Match) -> str:
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
        def replace_email(match: re.Match) -> str:
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
        def replace_phone(match: re.Match) -> str:
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
