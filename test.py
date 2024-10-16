import asyncio
import logging

from kani.ext.realtime.events import ConversationItemCreate, ResponseCreate
from kani.ext.realtime.models import MessageConversationItem, ResponseConfig, TextContentPart
from kani.ext.realtime.session import RealtimeSession


async def test():
    session = RealtimeSession(session_config=ResponseConfig(modalities=["text"]))
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(test())
