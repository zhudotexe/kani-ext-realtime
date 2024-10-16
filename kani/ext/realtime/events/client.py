from typing import Literal

from .base import ClientEvent as BaseEvent
from ..models import ConversationItem, ResponseConfig


# ===== events =====
class SessionUpdate(BaseEvent):
    type: Literal["session.update"] = "session.update"
    session: ResponseConfig


class InputAudioBufferAppend(BaseEvent):
    type: Literal["input_audio_buffer.append"] = "input_audio_buffer.append"
    audio: str

    # todo from_bytes


class InputAudioBufferCommit(BaseEvent):
    type: Literal["input_audio_buffer.commit"] = "input_audio_buffer.commit"


class InputAudioBufferClear(BaseEvent):
    type: Literal["input_audio_buffer.clear"] = "input_audio_buffer.clear"


class ConversationItemCreate(BaseEvent):
    type: Literal["conversation.item.create"] = "conversation.item.create"
    previous_item_id: str | None = None
    item: ConversationItem


class ConversationItemTruncate(BaseEvent):
    type: Literal["conversation.item.truncate"] = "conversation.item.truncate"
    item_id: str
    content_index: int
    audio_end_ms: int


class ConversationItemDelete(BaseEvent):
    type: Literal["conversation.item.delete"] = "conversation.item.delete"
    item_id: str


class ResponseCreate(BaseEvent):
    type: Literal["response.create"] = "response.create"
    response: ResponseConfig | None = None


class ResponseCancel(BaseEvent):
    type: Literal["response.cancel"] = "response.cancel"
