import abc
import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, Field


# ===== client =====
# ---- session.update ----
class AudioTranscriptionConfig(BaseModel):
    # enabled: bool = True
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

    # @classmethod
    # def from_ai_function(cls, f: AIFunction):
    #     return cls(name=f.name, description=f.desc, parameters=f.json_schema)


class ResponseConfig(BaseModel):
    modalities: list[str] = ["text", "audio"]
    instructions: str = ""
    voice: Literal["alloy", "echo", "shimmer"] = "alloy"
    input_audio_format: Literal["pcm16", "g711_ulaw", "g711_alaw"] = "pcm16"
    output_audio_format: Literal["pcm16", "g711_ulaw", "g711_alaw"] = "pcm16"
    input_audio_transcription: AudioTranscriptionConfig | None = AudioTranscriptionConfig()
    turn_detection: TurnDetectionConfig | None = TurnDetectionConfig()
    tools: list[FunctionDefinition] = []
    tool_choice: Literal["auto", "none", "required"] | str = "auto"
    temperature: float = 0.8
    # max_output_tokens: int | Literal["inf"] | None = "inf"


# ---- conversation.item.create ----
class TextContentPart(BaseModel):
    type: Literal["input_text", "text"] = "input_text"
    text: str


class AudioContentPart(BaseModel):
    type: Literal["input_audio", "audio"] = "input_audio"
    audio: str
    transcript: str
    # todo transcript_bytes


ContentPart = Annotated[TextContentPart | AudioContentPart, Field(discriminator="type")]


class ConversationItemBase(BaseModel, abc.ABC):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str
    status: Literal["completed", "in_progress", "incomplete"] = "completed"


class MessageConversationItem(ConversationItemBase):
    type: Literal["message"] = "message"
    role: Literal["user", "assistant", "system"]
    content: list[ContentPart]


class FunctionCallConversationItem(ConversationItemBase):
    type: Literal["function_call"] = "function_call"
    call_id: str
    name: str
    arguments: str


class FunctionCallOutputConversationItem(ConversationItemBase):
    type: Literal["function_call_output"] = "function_call_output"
    output: str


ConversationItem = Annotated[
    MessageConversationItem | FunctionCallConversationItem | FunctionCallOutputConversationItem,
    Field(discriminator="type"),
]


# ===== server =====
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
class ConversationDetails(BaseModel):
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
