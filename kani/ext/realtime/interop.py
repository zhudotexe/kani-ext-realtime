"""Translation between Kani models and OpenAI models"""

import base64
import itertools
from typing import TYPE_CHECKING

import openai.types.beta.realtime as oait
from kani import AIFunction, ChatMessage, ChatRole, FunctionCall, MessagePart, ToolCall
from openai.types.beta.realtime.session_create_params import Tool
from pydantic import Field

if TYPE_CHECKING:
    from .session import RealtimeSession


class TextPart(MessagePart):
    oai_type: str
    text: str

    def __str__(self):
        return self.text


class AudioPart(MessagePart):
    oai_type: str
    transcript: str | None
    audio_b64: str | None = Field(repr=False)

    @property
    def audio_bytes(self) -> bytes | None:
        if self.audio_b64 is None:
            return None
        return base64.b64decode(self.audio_b64)

    def __str__(self):
        return self.transcript if self.transcript is not None else ""

    def __repr__(self):
        if self.audio_b64 is None:
            audio_repr = "None"
        else:
            audio_duration = len(self.audio_bytes) / 48000
            audio_repr = f"[audio: {audio_duration:.3f}s]"
        return f'{self.__repr_name__()}({self.__repr_str__(", ")}, audio={audio_repr})'

    def __rich_repr__(self):
        if self.audio_b64 is None:
            audio_repr = "None"
        else:
            audio_duration = len(self.audio_bytes) / 48000
            audio_repr = f"[audio: {audio_duration:.3f}s]"

        yield "oai_type", self.oai_type
        yield "transcript", self.transcript
        yield "audio", audio_repr


# ===== translators =====
# ---- oai -> kani ----
def content_part_to_message_part(part: oait.ConversationItemContent) -> MessagePart:
    match part.model_dump():  # can be a Part sometimes which borks things
        case {"type": "input_text" | "text" as oai_type, "text": text}:
            return TextPart(oai_type=oai_type, text=text)
        case {"type": "input_audio" | "audio" as oai_type, "audio": audio, "transcript": transcript}:
            return AudioPart(oai_type=oai_type, audio_b64=audio, transcript=transcript)
    raise ValueError(f"Unknown content part: {part!r}")


def conv_items_to_chat_message(conv_items: list[oait.ConversationItem]) -> ChatMessage:
    # this takes a list because ASST messages can have tool calls as separate item in a single response
    out_role = None
    out_content = []
    out_tool_calls = []
    out_kwargs = {}
    for item in conv_items:
        match item:
            case oait.ConversationItem(type="function_call", call_id=call_id, name=name, arguments=args):
                out_tool_calls.append(ToolCall.from_function_call(FunctionCall(name=name, arguments=args), call_id))
            case oait.ConversationItem(type="function_call_output", call_id=call_id, output=output):
                out_role = ChatRole.FUNCTION
                out_kwargs["tool_call_id"] = call_id
                out_content.append(output)
            case oait.ConversationItem(type="message", role=role, content=content):
                if out_role is not None and out_role != role:
                    raise ValueError(f"Got 2 different message roles in response: {out_role}, {role}")
                out_role = ChatRole(role)
                out_content.extend(map(content_part_to_message_part, content))
            case other:
                raise ValueError(f"A response shouldn't have this but it did: {other!r}")
    return ChatMessage(
        role=out_role or ChatRole.ASSISTANT, content=out_content, tool_calls=out_tool_calls, **out_kwargs
    )


def response_to_chat_message(response: oait.RealtimeResponse) -> ChatMessage:
    return conv_items_to_chat_message(response.output)


def chat_history_from_session_state(session: "RealtimeSession"):
    """Given a session, get the chat history from cached responses."""
    # read chat items from session, grouping by responses (model output group or user input)
    history = []
    for resp_id, item_ids in itertools.groupby(
        session.conversation_item_order,
        key=lambda i: session.conversation_item_id_to_response_id.get(i),
    ):
        if resp_id is not None:
            msgs = [response_to_chat_message(session.responses[resp_id])]
        else:
            msgs = [conv_items_to_chat_message([session.conversation_items[iid]]) for iid in item_ids]
        # only add the message to history if it has any content - this can happen if we have a responsecreate but no
        # content yet
        history.extend(m for m in msgs if m.content)
    # todo return immutable
    return history


# ---- kani -> oai ----
def chat_message_to_conv_items(message: ChatMessage) -> list[oait.ConversationItem]:
    if message.role == ChatRole.FUNCTION:
        return [oait.ConversationItem(type="function_call_output", call_id=message.tool_call_id, output=message.text)]

    items = []
    # content
    content = []
    for part in message.parts:
        match part:
            case str():
                content.append(
                    oait.ConversationItemContent(
                        type="input_text" if message.role != ChatRole.ASSISTANT else "text", text=part
                    )
                )
            case TextPart(oai_type=oai_type, text=text):
                content.append(oait.ConversationItemContent(type=oai_type, text=text))
            case AudioPart(oai_type=oai_type, audio_b64=audio, transcript=transcript):
                content.append(oait.ConversationItemContent(type=oai_type, audio=audio, transcript=transcript))
            case _:
                raise ValueError(f"Unknown content part: {part!r}")
    if content:
        items.append(oait.ConversationItem(type="message", role=message.role.value, content=content))

    # tool calls
    if message.tool_calls:
        for tc in message.tool_calls:
            items.append(
                oait.ConversationItem(
                    type="function_call", call_id=tc.id, name=tc.function.name, arguments=tc.function.arguments
                )
            )

    return items


def ai_function_to_tool(func: AIFunction) -> Tool:
    return Tool(type="function", name=func.name, description=func.desc, parameters=func.json_schema)
