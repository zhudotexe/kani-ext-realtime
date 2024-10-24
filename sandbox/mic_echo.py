import asyncio

from kani.ext.realtime import play_audio
from kani.ext.realtime.audio import get_audio_stream


async def main():
    stream = get_audio_stream(0)
    async for frame in stream:
        await play_audio(frame)


if __name__ == "__main__":
    asyncio.run(main())
