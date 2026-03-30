"""Tests for the messages feature."""

import os
import tempfile

import pytest

from clorag.core.messages_db import Message, MessagesDatabase


@pytest.fixture
def messages_db() -> MessagesDatabase:
    """Create a temporary messages database."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = MessagesDatabase(db_path=path)
    yield db
    os.unlink(path)


class TestMessagesDatabase:
    """Tests for MessagesDatabase CRUD operations."""

    def test_create_message(self, messages_db: MessagesDatabase) -> None:
        msg = messages_db.create_message(
            title="Test Update",
            body="This is a test message.",
            message_type="info",
        )
        assert msg.title == "Test Update"
        assert msg.body == "This is a test message."
        assert msg.message_type == "info"
        assert msg.is_active is True
        assert msg.sort_order == 0
        assert msg.link_url is None
        assert msg.id is not None

    def test_create_message_with_all_fields(self, messages_db: MessagesDatabase) -> None:
        msg = messages_db.create_message(
            title="Firmware Release",
            body="RIO firmware 2.5 available.",
            message_type="feature",
            link_url="https://support.cyanview.cloud/firmware",
            is_active=True,
            sort_order=1,
            expires_at="2026-12-31T23:59:59",
        )
        assert msg.message_type == "feature"
        assert msg.link_url == "https://support.cyanview.cloud/firmware"
        assert msg.sort_order == 1
        assert msg.expires_at is not None

    def test_create_message_invalid_type(self, messages_db: MessagesDatabase) -> None:
        with pytest.raises(ValueError, match="Invalid message_type"):
            messages_db.create_message(title="Bad", body="Bad", message_type="invalid")

    def test_get_active_messages_excludes_inactive(self, messages_db: MessagesDatabase) -> None:
        messages_db.create_message(title="Active", body="Visible", message_type="info")
        messages_db.create_message(
            title="Inactive", body="Hidden", message_type="warning", is_active=False
        )
        active = messages_db.get_active_messages()
        assert len(active) == 1
        assert active[0].title == "Active"

    def test_get_active_messages_excludes_expired(self, messages_db: MessagesDatabase) -> None:
        messages_db.create_message(title="Fresh", body="New", message_type="info")
        messages_db.create_message(
            title="Expired", body="Old", message_type="fix",
            expires_at="2020-01-01T00:00:00",
        )
        active = messages_db.get_active_messages()
        assert len(active) == 1
        assert active[0].title == "Fresh"

    def test_get_active_messages_ordered_by_sort_order(
        self, messages_db: MessagesDatabase
    ) -> None:
        messages_db.create_message(title="Second", body="B", message_type="info", sort_order=2)
        messages_db.create_message(title="First", body="A", message_type="info", sort_order=1)
        messages_db.create_message(title="Third", body="C", message_type="info", sort_order=3)
        active = messages_db.get_active_messages()
        assert [m.title for m in active] == ["First", "Second", "Third"]

    def test_get_all_messages_includes_inactive(self, messages_db: MessagesDatabase) -> None:
        messages_db.create_message(title="Active", body="A", message_type="info")
        messages_db.create_message(title="Inactive", body="B", message_type="info", is_active=False)
        all_msgs = messages_db.get_all_messages()
        assert len(all_msgs) == 2

    def test_update_message(self, messages_db: MessagesDatabase) -> None:
        msg = messages_db.create_message(title="Original", body="Old body", message_type="info")
        updated = messages_db.update_message(msg.id, title="Updated", body="New body")
        assert updated is not None
        assert updated.title == "Updated"
        assert updated.body == "New body"
        assert updated.message_type == "info"  # unchanged

    def test_update_message_toggle_active(self, messages_db: MessagesDatabase) -> None:
        msg = messages_db.create_message(title="Test", body="Body", message_type="info")
        updated = messages_db.update_message(msg.id, is_active=False)
        assert updated is not None
        assert updated.is_active is False

    def test_update_nonexistent_message(self, messages_db: MessagesDatabase) -> None:
        result = messages_db.update_message("nonexistent-id", title="Nope")
        assert result is None

    def test_delete_message(self, messages_db: MessagesDatabase) -> None:
        msg = messages_db.create_message(title="Delete Me", body="Gone", message_type="info")
        assert messages_db.delete_message(msg.id) is True
        assert messages_db.get_message(msg.id) is None

    def test_delete_nonexistent_message(self, messages_db: MessagesDatabase) -> None:
        assert messages_db.delete_message("nonexistent-id") is False

    def test_message_to_dict(self, messages_db: MessagesDatabase) -> None:
        msg = messages_db.create_message(title="Dict Test", body="Body", message_type="feature")
        d = msg.to_dict()
        assert d["title"] == "Dict Test"
        assert d["message_type"] == "feature"
        assert "id" in d
        assert "created_at" in d

    def test_valid_message_types(self, messages_db: MessagesDatabase) -> None:
        for msg_type in ["info", "warning", "feature", "fix"]:
            msg = messages_db.create_message(
                title=f"Type {msg_type}", body="Test", message_type=msg_type
            )
            assert msg.message_type == msg_type
