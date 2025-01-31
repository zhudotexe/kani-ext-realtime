"""Translation between Kani models and OpenAI models"""

import base64

import openai.types.beta.realtime as oait
from kani import AIFunction, ChatMessage, ChatRole, FunctionCall, MessagePart, ToolCall
from openai.types.beta.realtime.session_create_params import Tool


class TextPart(MessagePart):
    oai_type: str
    text: str

    def __str__(self):
        return self.text


class AudioPart(MessagePart):
    oai_type: str
    audio_b64: str | None
    transcript: str

    @property
    def audio_bytes(self) -> bytes | None:
        if self.audio_b64 is None:
            return None
        return base64.b64decode(self.audio_b64)

    def __str__(self):
        return self.transcript


# ===== translators =====
# ---- oai -> kani ----
def content_part_to_message_part(part: oait.ConversationItemContent) -> MessagePart:
    match part:
        case oait.ConversationItemContent(type="input_text" | "text" as oai_type, text=text):
            return TextPart(oai_type=oai_type, text=text)
        case oait.ConversationItemContent(type="input_audio" | "audio" as oai_type, audio=audio, transcript=transcript):
            return AudioPart(oai_type=oai_type, audio_b64=audio, transcript=transcript)
    raise ValueError(f"Unknown content part: {part!r}")


def response_to_chat_message(response: oait.RealtimeResponse) -> ChatMessage:
    out_role = None
    out_content = []
    out_tool_calls = []
    for item in response.output:
        match item:
            case oait.ConversationItem(type="function_call", call_id=call_id, name=name, arguments=args):
                out_tool_calls.append(ToolCall.from_function_call(FunctionCall(name=name, arguments=args), call_id))
            case oait.ConversationItem(role=role, content=content):
                if out_role is not None and out_role != role:
                    raise ValueError(f"Got 2 different message roles in response: {out_role}, {role}")
                out_role = ChatRole(role)
                out_content.extend(map(content_part_to_message_part, content))
            case other:
                raise ValueError(f"A response shouldn't have this but it did: {other!r}")
    return ChatMessage(role=out_role or ChatRole.ASSISTANT, content=out_content, tool_calls=out_tool_calls)


# ---- kani -> oai ----
def chat_message_to_conv_items(message: ChatMessage) -> list[oait.ConversationItem]:
    if message.role == ChatRole.FUNCTION:
        return [oait.ConversationItem(type="function_call_output", call_id=message.tool_call_id, output=message.text)]

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

    items = [oait.ConversationItem(type="message", role=message.role.value, content=content)]

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
