"""Persistent, bounded conversation memory."""

import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.core.exceptions import ConversationNotFoundError
from backend.database.models import Conversation, Message


class ConversationService:
    """Create conversations and retrieve isolated recent message history."""

    def create(self, session: Session, title: str | None = None) -> Conversation:
        conversation = Conversation(title=title)
        session.add(conversation)
        session.commit()
        session.refresh(conversation)
        return conversation

    def require(self, session: Session, conversation_id: str) -> Conversation:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(conversation_id)
        return conversation

    def history(
        self, session: Session, conversation_id: str, limit: int
    ) -> list[tuple[str, str]]:
        if limit <= 0:
            return []
        self.require(session, conversation_id)
        messages = session.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        ).all()
        return [(message.role, message.content) for message in reversed(messages)]

    def add_message(
        self,
        session: Session,
        conversation_id: str,
        role: str,
        content: str,
        sources: list[dict[str, object]] | None = None,
        retrieval_metadata: dict[str, object] | None = None,
    ) -> Message:
        if role not in {"user", "assistant"} or not content.strip():
            raise ValueError("Messages must have a supported role and non-empty content.")
        conversation = self.require(session, conversation_id)
        message = Message(
            conversation_id=conversation.id,
            role=role,
            content=content,
            sources_json=json.dumps(sources or []),
            retrieval_metadata_json=json.dumps(retrieval_metadata or {}),
        )
        session.add(message)
        session.commit()
        session.refresh(message)
        return message
