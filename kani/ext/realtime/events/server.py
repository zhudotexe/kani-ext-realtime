from typing import Literal

from pydantic import BaseModel

from .base import ServerEvent as BaseEvent
from .client import ConversationItem, ResponseConfig


# ===== models =====
# ---- error ----
class ErrorDetails(BaseModel):
    type: str
    code: str | None = None
    message: str
    param: str | None = None
    event_id: str | None = None


# ---- session.created ----
# ---- session.updated ----
class SessionDetails(ResponseConfig):
    id: str
    object: Literal["realtime.session"]


# ---- conversation.created ----
class ConversationDetails(ResponseConfig):
    id: str
    object: Literal["realtime.conversation"]


# ---- response.created ----
# ---- response.done ----
class UsageDetails(BaseModel):
    total_tokens: int
    input_tokens: int
    output_tokens: int


class RealtimeResponse(BaseModel):
    id: str
    object: Literal["realtime.response"]
    status: Literal["in_progress", "completed", "cancelled", "failed", "incomplete"]
    status_details: dict | None
    output: list[ConversationItem]
    usage: UsageDetails | None


# ---- rate_limits.updated ----
class RateLimitInfo(BaseModel):
    name: str
    limit: int
    remaining: int
    reset_seconds: float


# ===== events =====
class Error(BaseEvent):
    type: Literal["error"] = "error"
    error: ErrorDetails


class SessionCreated(BaseEvent):
    type: Literal["session.created"] = "session.created"
    session: SessionDetails


class SessionUpdated(BaseEvent):
    type: Literal["session.updated"] = "session.updated"
    session: SessionDetails


class ConversationCreated(BaseEvent):
    type: Literal["conversation.created"] = "conversation.created"
    conversation: ConversationDetails


class ConversationItemCreated(BaseEvent):
    type: Literal["conversation.item.created"] = "conversation.item.created"
    previous_item_id: str | None
    item: ConversationItem


class ConversationItemInputAudioTranscriptionCompleted(BaseEvent):
    type: Literal["conversation.item.input_audio_transcription.completed"] = (
        "conversation.item.input_audio_transcription.completed"
    )
    item_id: str
    content_index: int
    transcript: str


class ConversationItemInputAudioTranscriptionFailed(BaseEvent):
    type: Literal["conversation.item.input_audio_transcription.failed"] = (
        "conversation.item.input_audio_transcription.failed"
    )
    item_id: str
    content_index: int
    error: ErrorDetails


class ConversationItemTruncated(BaseEvent):
    type: Literal["conversation.item.truncated"] = "conversation.item.truncated"
    item_id: str
    content_index: int
    audio_end_ms: int


class ConversationItemDeleted(BaseEvent):
    type: Literal["conversation.item.deleted"] = "conversation.item.deleted"
    item_id: str


class InputAudioBufferCommitted(BaseEvent):
    type: Literal["input_audio_buffer.committed"] = "input_audio_buffer.committed"
    previous_item_id: str
    item_id: str


class InputAudioBufferCleared(BaseEvent):
    type: Literal["input_audio_buffer.cleared"] = "input_audio_buffer.cleared"


class InputAudioBufferSpeechStarted(BaseEvent):
    type: Literal["input_audio_buffer.speech_started"] = "input_audio_buffer.speech_started"
    audio_start_ms: int
    item_id: str


class InputAudioBufferSpeechStopped(BaseEvent):
    type: Literal["input_audio_buffer.speech_stopped"] = "input_audio_buffer.speech_stopped"
    audio_end_ms: int
    item_id: str


class ResponseCreated(BaseEvent):
    type: Literal["response.created"] = "response.created"
    response: RealtimeResponse


class ResponseDone(BaseEvent):
    type: Literal["response.done"] = "response.done"
    response: RealtimeResponse


class RateLimitsUpdated(BaseEvent):
    type: Literal["rate_limits.updated"] = "rate_limits.updated"
    rate_limits: list[RateLimitInfo]


# ---- streaming ----
# TODO:
# response.output_item.added
# response.output_item.done
# response.content_part.added
# response.content_part.done
# response.text.delta
# response.text.done
# response.audio_transcript.delta
# response.audio_transcript.done
# response.audio.delta
# response.audio.done
# response.function_call_arguments.delta
# response.function_call_arguments.done
