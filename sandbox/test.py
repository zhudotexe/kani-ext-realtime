import asyncio
import faulthandler
import logging
import os

from kani.ext.realtime import OpenAIRealtimeKani
from kani.ext.realtime.cli import chat_in_terminal_audio_async
from kani.ext.realtime.events import ConversationItemCreate, ResponseCreate
from kani.ext.realtime.models import MessageConversationItem, SessionConfig, TextContentPart
from kani.ext.realtime.session import RealtimeSession


async def test1():
    session = RealtimeSession(
        session_config=SessionConfig(modalities=["text"]),
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-realtime-preview-2024-10-01",
    )
    await session.connect()
    await session.send(
        ConversationItemCreate(
            item=MessageConversationItem(
                role="user",
                content=[TextContentPart(text="hello")],
            ),
        )
    )
    await session.send(ResponseCreate())
    await session.wait_for("response.created")
    response = await session.wait_for("response.done")
    print(response.response)


async def test2():
    ai = OpenAIRealtimeKani()
    await ai.connect()
    await chat_in_terminal_audio_async(ai, mode="stream")


if __name__ == "__main__":
    faulthandler.enable()
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(test2())
