import abc
import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from kani import AIFunction
from .base import BaseEvent


# ===== models =====
# ---- session.update ----
class AudioTranscriptionConfig(BaseModel):
    enabled: bool = True
    model: str = "whisper-1"


class TurnDetectionConfig(BaseModel):
    type: str = "server_vad"
    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500


# ---- session.update ----
# ---- response.create ----
class FunctionDefinition(BaseModel):
    type: str = "function"
    name: str
    description: str
    parameters: dict

    @classmethod
    def from_ai_function(cls, f: AIFunction):
        return cls(name=f.name, description=f.desc, parameters=f.json_schema)


class ResponseConfig(BaseModel):
    modalities: list[str] = ["text", "audio"]
    instructions: str | None = None
    voice: Literal["alloy", "echo", "shimmer"] = "alloy"
    input_audio_format: Literal["pcm16", "g711_ulaw", "g711_alaw"] = "pcm16"
    output_audio_format: Literal["pcm16", "g711_ulaw", "g711_alaw"] = "pcm16"
    input_audio_transcription: AudioTranscriptionConfig | None = AudioTranscriptionConfig()
    turn_detection: TurnDetectionConfig | None = TurnDetectionConfig()
    tools: list[FunctionDefinition] = []
    tool_choice: Literal["auto", "none", "required"] | str = "auto"
    temperature: float = 0.8
    max_output_tokens: int | None = None


# ---- conversation.item.create ----
class TextContentPart(BaseModel):
    type: Literal["input_text", "text"]
    text: str


class AudioContentPart(BaseModel):
    type: Literal["input_audio", "audio"]
    audio: str
    transcript: str
    # todo transcript_bytes


ContentPart = Annotated[TextContentPart | AudioContentPart, Field(discriminator="type")]


class ConversationItemBase(BaseModel, abc.ABC):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str
    status: Literal["completed", "in_progress", "incomplete"] = "completed"


class MessageConversationItem(ConversationItemBase):
    type: Literal["message"]
    role: Literal["user", "assistant", "system"]
    content: list[ContentPart]


class FunctionCallConversationItem(ConversationItemBase):
    type: Literal["function_call"]
    call_id: str
    name: str
    arguments: str


class FunctionCallOutputConversationItem(ConversationItemBase):
    type: Literal["function_call_output"]
    output: str


ConversationItem = Annotated[
    MessageConversationItem | FunctionCallConversationItem | FunctionCallOutputConversationItem,
    Field(discriminator="type"),
]


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
