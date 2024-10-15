import asyncio
import logging
import os
from typing import Any, Awaitable, Callable

import websockets
from websockets.asyncio.client import connect

from .events import ServerEvent
from .models import ConversationItem, RealtimeResponse, ResponseConfig

log = logging.getLogger(__name__)


class RealtimeSession:
    def __init__(
        self,
        api_key: str = None,
        model="gpt-4o-realtime-preview-2024-10-01",
        *,
        session_config: ResponseConfig = None,
        ws_base: str = "wss://api.openai.com/v1/realtime",
        headers: dict = None,
        # organization: str = None,  # todo is this supported?
        #  :param organization: The OpenAI organization to use in requests. By default, the org ID would be read from
        #      the `OPENAI_ORG_ID` environment variable (defaults to the API key's default org if not set).
        **generation_args,
    ):
        """
        :param api_key: Your OpenAI API key. By default, the API key will be read from the `OPENAI_API_KEY` environment
            variable.
        :param model: The id of the realtime model to use (e.g. "gpt-4o-realtime-preview-2024-10-01").

        :param ws_base: The base WebSocket URL to connect to.
        :param headers: A dict of HTTP headers to include with each request.
        :param client: An instance of ``httpx.AsyncClient`` (for reusing the same client in multiple engines).
        :param generation_args: The arguments to pass to the ``response.create`` call with each request. See
            https://platform.openai.com/docs/api-reference/realtime-client-events/response/create for a full list of
            params. Specifically, these arguments will be passed as the ``response`` key.
        """
        if session_config is None:
            session_config = ResponseConfig()
        if headers is None:
            headers = {}
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")

        # default headers
        headers.setdefault("Authorization", f"Bearer {api_key}")
        headers.setdefault("OpenAI-Beta", "realtime=v1")

        # client config
        self.ws_base = ws_base
        self.headers = headers
        self.model = model
        self.session_config = session_config
        self.generation_args = generation_args

        # state
        self.responses: dict[str, RealtimeResponse] = {}
        self.conversation_items: dict[str, ConversationItem] = {}

        # ws
        self.ws = None
        self.listeners = []
        self.ws_task = None

    # ==== lifecycle ====
    async def connect(self):
        """Connect to the WS and begin a task for event handling."""
        if self.ws_task is None:
            self.ws_task = asyncio.create_task(self._ws_task(), name="realtime-ws")

    async def close(self):
        if self.ws_task is not None:
            self.ws_task.cancel()



    # ==== events ====
    def add_listener(self, callback: Callable[[ServerEvent], Awaitable[Any]]):
        """
        Add a listener which is called for every event received from the WS.
        The listener must be an asynchronous function that takes in an event in a single argument.
        """
        self.listeners.append(callback)

    def remove_listener(self, callback):
        """Remove a listener added by :meth:`add_listener`."""
        self.listeners.remove(callback)

    async def wait_for(self, event_type: str, timeout: int = 60) -> ServerEvent:
        """Wait for the next event of a given type, and return it."""
        future = asyncio.get_running_loop().create_future()

        async def waiter(e: ServerEvent):
            if e.type == event_type:
                future.set_result(e)

        try:
            self.add_listener(waiter)
            return await asyncio.wait_for(future, timeout)
        finally:
            self.remove_listener(waiter)

    async def _ws_task(self):
        try:
            async with connect(f"{self.ws_base}?model={self.model}", additional_headers=self.headers) as self.ws:
                async for data in self.ws:
                    # noinspection PyBroadException
                    try:
                        event = ServerEvent.model_validate_json(data)
                        # process our event first, always
                        await self._handle_server_event(event)
                        # get listeners, call them - listeners can use the result of the processing if needed
                        await asyncio.gather(*(callback(event) for callback in self.listeners), return_exceptions=True)
                    except websockets.ConnectionClosedError as e:
                        log.error(f"WS connection closed unexpectedly: {e}")
                    except Exception:
                        log.exception("Exception when handling WS event:")
        except asyncio.CancelledError:
            return
        finally:
            self.ws = None

    # ==== ws event handler ====
    async def _handle_server_event(self, event: ServerEvent):
        """
        Main entrypoint for received server events.
        Will always be fully processed before WS events are dispatched to consumers to allow consumers to read from
        state instead of updating their state.
        """