import asyncio
import logging
from typing import Any, Awaitable, Callable

import openai.types.beta.realtime as oait
import websockets
from openai import AsyncOpenAI
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection
from openai.types.beta.realtime import RealtimeClientEvent, RealtimeServerEvent

from ._internal import get_server_event_handlers, server_event_handler

log = logging.getLogger(__name__)


class RealtimeSession:
    """This is an internal object used to manage the state of the OpenAI Realtime session."""

    def __init__(self, model: str, client: AsyncOpenAI):
        """
        :param model: The id of the realtime model to use.
        :param client: The OpenAI client to use.
        :param generation_args: The arguments to pass to the ``response.create`` call with each request. See
            https://platform.openai.com/docs/api-reference/realtime-client-events/response/create for a full list of
            params. Specifically, these arguments will be passed as the ``response`` key.
        """
        # client config
        self.model = model
        self.client = client

        # state
        self.session_config: oait.Session | None = None
        self.session_id: str | None = None
        self.conversation_id: str | None = None
        self.responses: dict[str, oait.RealtimeResponse] = {}
        self.conversation_items: dict[str, oait.ConversationItem] = {}

        # ws
        self._conn: AsyncRealtimeConnection | None = None
        self._ws_connected = asyncio.Event()
        self._session_created = asyncio.Event()
        self.listeners = []
        self.ws_task = None

        # event handlers
        self._server_event_handlers = get_server_event_handlers(self)

    # ==== lifecycle ====
    async def connect(self):
        """
        Connect to the WS, begin a task for event handling, and init the session.

        You should usually call :meth:`.OpenAIRealtimeKani.connect` instead of this.
        """
        if self.ws_task is None:
            self.ws_task = asyncio.create_task(self._ws_task(), name="realtime-ws")
        await self._ws_connected.wait()
        await self._session_created.wait()

    async def close(self):
        if self.ws_task is not None:
            self.ws_task.cancel()  # closes on cancel

    # ==== iface ====
    async def send(self, event: RealtimeClientEvent):
        """Send a client event to the websocket."""
        if self._conn is None:
            raise RuntimeError("Websocket is not yet initialized - call connect() first")
        log.debug(f">>> {event!r}")
        await self._conn.send(event)

    # ==== events ====
    def add_listener(self, callback: Callable[[RealtimeServerEvent], Awaitable[Any]]):
        """
        Add a listener which is called for every event received from the WS.
        The listener must be an asynchronous function that takes in an event in a single argument.
        """
        self.listeners.append(callback)

    def remove_listener(self, callback):
        """Remove a listener added by :meth:`add_listener`."""
        self.listeners.remove(callback)

    async def wait_for(
        self, event_type: str, predicate: Callable[[RealtimeServerEvent], bool] = None, timeout: int = 60
    ) -> RealtimeServerEvent:
        """Wait for the next event of a given type, and return it."""
        future = asyncio.get_running_loop().create_future()

        async def waiter(e: RealtimeServerEvent):
            if e.type == event_type:
                if predicate is None or predicate(e):
                    future.set_result(e)

        try:
            self.add_listener(waiter)
            return await asyncio.wait_for(future, timeout)
        finally:
            self.remove_listener(waiter)

    async def _ws_task(self):
        """Main websocket receive loop."""
        try:
            async with self.client.beta.realtime.connect(model=self.model) as self._conn:
                self._ws_connected.set()
                async for event in self._conn:
                    # noinspection PyBroadException
                    try:
                        log.debug(f"<<< {event!r}")
                        # process our event first, always
                        await self._handle_server_event(event)
                        # get listeners, call them - listeners can use the result of the processing if needed
                        await asyncio.gather(*(callback(event) for callback in self.listeners), return_exceptions=True)
                    except websockets.ConnectionClosedError as e:
                        log.error(f"WS connection closed unexpectedly: {e}")
                    except asyncio.CancelledError:
                        return
                    except Exception:
                        log.exception("Exception when handling WS event:")
        except asyncio.CancelledError:
            return
        except Exception:
            log.exception("Exception when connecting to the websocket:")
            raise
        finally:
            self._ws_connected.clear()
            self._conn = None

    # ==== ws event handlers ====
    async def _handle_server_event(self, event: RealtimeServerEvent):
        """
        Main entrypoint for received server events.
        Will always be fully processed before WS events are dispatched to consumers to allow consumers to read from
        state instead of updating their state.
        """

        handler = self._server_event_handlers.get(event.type)
        if handler is None:
            # warnings.warn(f"A server event with type {event.type!r} is being unhandled: {event!r}")
            return
        await handler(event)

    @server_event_handler("error")
    async def _handle_error(self, event: oait.ErrorEvent):
        log.error(event.error)

    @server_event_handler("session.created")
    async def _handle_session_created(self, event: oait.SessionCreatedEvent):
        self._session_created.set()
        self.session_id = event.session.id
        self.session_config = event.session

    @server_event_handler("session.updated")
    async def _handle_session_updated(self, event: oait.SessionUpdatedEvent):
        self.session_id = event.session.id
        self.session_config = event.session

    @server_event_handler("conversation.created")
    async def _handle_conversation_created(self, event: oait.ConversationCreatedEvent):
        self.conversation_id = event.conversation.id

    @server_event_handler("conversation.item.created")
    async def _handle_conversation_item_created(self, event: oait.ConversationItemCreatedEvent):
        item_id = event.item.id
        self.conversation_items[item_id] = event.item

    @server_event_handler("conversation.item.input_audio_transcription.completed")
    async def _handle_conversation_item_input_audio_transcription_completed(
        self, event: oait.ConversationItemInputAudioTranscriptionCompletedEvent
    ):
        content = self.get_item_content(event.item_id, event.content_index)
        content.transcript = event.transcript.strip()

    @server_event_handler("conversation.item.input_audio_transcription.failed")
    async def _handle_conversation_item_input_audio_transcription_failed(
        self, event: oait.ConversationItemInputAudioTranscriptionFailedEvent
    ):
        content = self.get_item_content(event.item_id, event.content_index)
        content.transcript = f"[transcript failed: {event.error}]"  # todo
        log.warning(f"Audio transcription failed: {event.error}")

    @server_event_handler("conversation.item.truncated")
    async def _handle_conversation_item_truncated(self, event: oait.ConversationItemTruncatedEvent):
        pass

    @server_event_handler("conversation.item.deleted")
    async def _handle_conversation_item_deleted(self, event: oait.ConversationItemDeletedEvent):
        self.conversation_items.pop(event.item_id, None)

    @server_event_handler("input_audio_buffer.committed")
    async def _handle_input_audio_buffer_committed(self, event: oait.InputAudioBufferCommittedEvent):
        pass  # todo create an in progress item id?

    @server_event_handler("input_audio_buffer.cleared")
    async def _handle_input_audio_buffer_cleared(self, event: oait.InputAudioBufferClearedEvent):
        pass

    @server_event_handler("input_audio_buffer.speech_started")
    async def _handle_input_audio_buffer_speech_started(self, event: oait.InputAudioBufferSpeechStartedEvent):
        pass  # todo create an in progress item id?

    @server_event_handler("input_audio_buffer.speech_stopped")
    async def _handle_input_audio_buffer_speech_stopped(self, event: oait.InputAudioBufferSpeechStoppedEvent):
        pass  # todo end an in progress item id?

    @server_event_handler("response.created")
    async def _handle_response_created(self, event: oait.ResponseCreatedEvent):
        self.responses[event.response.id] = event.response
        for item in event.response.output:
            self.conversation_items[item.id] = item

    @server_event_handler("response.done")
    async def _handle_response_done(self, event: oait.ResponseDoneEvent):
        self.responses[event.response.id] = event.response
        for item in event.response.output:
            self.conversation_items[item.id] = item

    @server_event_handler("rate_limits.updated")
    async def _handle_rate_limits_updated(self, event: oait.RateLimitsUpdatedEvent):
        pass  # todo

    # ==== helpers ====
    def get_item_content(self, item_id: str, content_index: int):
        item = self.conversation_items.get(item_id)
        if item is None:
            log.warning(f"Got event referencing item that does not exist: {item_id}")
            return
        if content_index < 0 or content_index >= len(item.content):
            log.warning(
                f"Got event referencing item content out of bounds (len {len(item.content)}, got {content_index})"
            )
            return
        return item.content[content_index]
