import asyncio
import json
from typing import Annotated

import d20
import httpx
from easyaudiostream import list_mics
from kani import AIParam, ai_function
from kani.ext.realtime import OpenAIRealtimeKani
from kani.ext.realtime.cli import chat_in_terminal_audio_async


class MyRealtimeKani(OpenAIRealtimeKani):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wikipedia_client = httpx.AsyncClient(base_url="https://en.wikipedia.org/w/api.php", follow_redirects=True)

    @ai_function()
    async def wikipedia(
        self,
        title: Annotated[str, AIParam(desc='The article title on Wikipedia, e.g. "Train_station".')],
    ):
        """Get additional information about a topic from Wikipedia."""
        # https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&titles=Train&explaintext=1&formatversion=2
        resp = await self.wikipedia_client.get(
            "/",
            params={
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "titles": title,
                "explaintext": 1,
                "formatversion": 2,
            },
        )
        data = resp.json()
        page = data["query"]["pages"][0]
        if extract := page.get("extract"):
            return extract
        return f"The page {title!r} does not exist on Wikipedia."

    @ai_function()
    async def search(self, query: str):
        """Find titles of Wikipedia articles similar to the given query."""
        # https://en.wikipedia.org/w/api.php?action=opensearch&format=json&search=Train
        resp = await self.wikipedia_client.get("/", params={"action": "opensearch", "format": "json", "search": query})
        return json.dumps(resp.json()[1])

    @ai_function()
    def roll(self, dice: str):
        """Roll some dice in XdY notation. Math and complex operators like kh3 are supported."""
        return d20.roll(dice).result


async def test(mic_id):
    ai = MyRealtimeKani()
    await ai.connect(voice="ballad")
    await chat_in_terminal_audio_async(ai, mode="full_duplex", mic_id=mic_id, verbose=True)
    print(ai.chat_history)


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    list_mics()
    mid = int(input("Mic ID: "))
    asyncio.run(test(mid))
