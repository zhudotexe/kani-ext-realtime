import asyncio
import base64
import enum
import functools
import itertools
import logging
from typing import Any, Awaitable, Callable, TYPE_CHECKING

import openai.types.beta.realtime as oait
import websockets
from openai import AsyncOpenAI
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection
from openai.types.beta.realtime import RealtimeClientEvent, RealtimeServerEvent
from typing_extensions import Unpack

from . import interop
from ._internal import create_id, create_task, get_server_event_handlers, server_event_handler
from .errors import OpenAIRealtimeError

if TYPE_CHECKING:
    from kani.models import ChatMessage
    from openai.types.beta.realtime.response_create_event_param import Response as ResponseCreateParams

log = logging.getLogger(__name__)


class ConnectionState(enum.Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    TERMINATING = "terminating"  # special case for when the program is terminating


class RealtimeSession:
    """This is an internal object used to manage the state of the OpenAI Realtime session."""

    def __init__(self, model: str, client: AsyncOpenAI, *, save_audio=True, retry_attempts: int = 5):
        """
        :param model: The id of the realtime model to use.
        :param client: The OpenAI client to use.
        :param save_audio: Whether to keep audio data in memory.
        :param retry_attempts: The number of times to retry failed WS events.
        """
        # client config
        self.model = model
        self.client = client
        self.save_audio = save_audio
        self.retry_attempts = retry_attempts

        # audio buffer
        self.input_audio_buffer = bytearray()
        self._last_speech_started = None
        self._last_speech_stopped = None
        self._input_audio_buffer_offset = 0  # the ms the input audio buffer in memory starts at from the server's view

        # state
        self.session_config: oait.Session | None = None
        self.session_id: str | None = None
        self.conversation_id: str | None = None
        self.responses: dict[str, oait.RealtimeResponse] = {}
        self.conversation_items: dict[str, oait.ConversationItem] = {}
        self.conversation_item_order: list[str] = []

        # ws
        self._conn: AsyncRealtimeConnection | None = None
        self.connection_state: ConnectionState = ConnectionState.DISCONNECTED
        self.has_connected_once = False
        self._ws_connected: asyncio.Future | None = None
        self._session_created = asyncio.Event()
        self.listeners = dict()
        self.lifecycle_listeners = []
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
            self._ws_connected = asyncio.get_running_loop().create_future()
            self.ws_task = asyncio.create_task(self._ws_task(), name="realtime-ws")
        await self._ws_connected
        await self._session_created.wait()

    async def close(self):
        if self.ws_task is not None:
            self.ws_task.cancel()  # closes on cancel

    async def _set_connection_state(self, new_state: ConnectionState):
        log.debug(f"New connection state: {new_state}")
        old_state = self.connection_state
        self.connection_state = new_state
        results = await asyncio.gather(
            *(callback(old_state, new_state) for callback in self.lifecycle_listeners), return_exceptions=True
        )
        for r in results:
            if isinstance(r, BaseException):
                log.exception("Exception when handling lifecycle event:", exc_info=r)

    # ==== iface ====
    async def send(self, event: RealtimeClientEvent):
        """Send a client event to the websocket."""
        if self.connection_state == ConnectionState.CONNECTING:
            await self._ws_connected
        if self._conn is None:
            raise RuntimeError("Websocket is not yet initialized - call connect() first")
        log.debug(f">>> {event!r}")
        await self._conn.send(event)

    async def send_with_retry(
        self,
        event: RealtimeClientEvent,
        ack_event_type: str,
        ack_event_predicate: Callable[[RealtimeServerEvent], bool] = lambda _: True,
        timeout: float = 60,
    ):
        """
        Send a client event to the websocket and wait for its ACKing response.

        If an causal error message or exception occurs or the ACK is not received by the timeout, try up to *attempts*
        times.
        Returns the ACKing response.
        """
        event.event_id = create_id("event")
        for attempt in range(self.retry_attempts):
            try:
                await self.send(event)
                return await self.wait_for(
                    ack_event_type, ack_event_predicate, timeout=timeout, raise_for=event.event_id
                )
            except Exception as e:
                if attempt + 1 == self.retry_attempts:
                    raise
                sleep_for = 2**attempt
                log.warning(
                    f"Exception in send_with_retry waiting for {ack_event_type} (attempt {attempt + 1} of"
                    f" {self.retry_attempts}), sleeping for {sleep_for} sec and retrying...",
                    exc_info=e,
                )
                await asyncio.sleep(sleep_for)

    # ---- high level iface ----
    async def session_update(self, session_config: oait.SessionCreateParams):
        event = oait.SessionUpdateEvent.model_validate({"type": "session.update", "session": session_config})
        return await self.send_with_retry(event, ack_event_type="session.updated")

    async def input_audio_buffer_append(self, frame: bytes):
        if self.save_audio:
            self.input_audio_buffer.extend(frame)
        data = base64.b64encode(frame).decode()
        await self.send(oait.InputAudioBufferAppendEvent(type="input_audio_buffer.append", audio=data))

    async def conversation_item_create_from_chat_message(self, message: "ChatMessage"):
        for item in interop.chat_message_to_conv_items(message):
            item_id = create_id("item")
            item.id = item_id
            event = oait.ConversationItemCreateEvent(type="conversation.item.create", item=item)
            # wait for confirmation and add audio bytes if we're logging
            created_event = await self.send_with_retry(
                event=event,
                ack_event_type="conversation.item.created",
                ack_event_predicate=lambda e: e.item.id == item_id,
            )
            if self.save_audio:
                created_item = created_event.item
                self.conversation_items[created_item.id] = merge_conversation_items(created_item, item)

    async def response_create(self, **generation_kwargs: Unpack["ResponseCreateParams"]) -> oait.ResponseCreatedEvent:
        if generation_kwargs:
            event = oait.ResponseCreateEvent.model_validate({"type": "response.create", "response": generation_kwargs})
        else:
            event = oait.ResponseCreateEvent.model_validate({"type": "response.create"})
        return await self.send_with_retry(event=event, ack_event_type="response.created")

    # ==== events ====
    def add_listener(self, callback: Callable[[RealtimeServerEvent], Awaitable[Any]], return_exceptions=False):
        """
        Add a listener which is called for every event received from the WS.
        The listener must be an asynchronous function that takes in an event in a single argument.

        If *return_exceptions* is True, the callback will also be called with any exceptions raised in the WS loop
        (usually disconnects).
        """
        if not return_exceptions:

            @functools.wraps(callback)
            def wrapped(e):
                if isinstance(e, BaseException):
                    return
                return callback(e)

            self.listeners[callback] = wrapped
        else:
            self.listeners[callback] = callback

    def remove_listener(self, callback):
        """Remove a listener added by :meth:`add_listener`."""
        self.listeners.pop(callback)

    def add_lifecycle_listener(self, callback: Callable[[ConnectionState, ConnectionState], Awaitable[Any]]):
        """
        Add a listener which is called for WS lifecycle changes.
        The listener must be an asynchronous function that takes in the (old_state, new_state).
        """
        self.lifecycle_listeners.append(callback)

    def remove_lifecycle_listener(self, callback):
        """Remove a listener added by :meth:`add_lifecycle_listener`."""
        self.lifecycle_listeners.remove(callback)

    async def wait_for(
        self,
        event_type: str,
        predicate: Callable[[RealtimeServerEvent], bool] = None,
        timeout: float = 60,
        raise_for: str | bool = None,
    ) -> RealtimeServerEvent:
        """
        Wait for the next event of a given type, and return it.

        :param raise_for: If given and an Error event is seen caused by this event ID, raise it. If True, raise for any
            Error event.
        """
        future = asyncio.get_running_loop().create_future()

        async def waiter(e: RealtimeServerEvent | BaseException):
            if isinstance(e, BaseException):
                future.set_exception(e)
            elif e.type == event_type and (predicate is None or predicate(e)):
                future.set_result(e)
            # raise an applicable Error
            elif isinstance(e, oait.ErrorEvent) and (raise_for is True or e.error.event_id == raise_for):
                err = OpenAIRealtimeError.from_ws_error(e.error)
                future.set_exception(err)

        try:
            self.add_listener(waiter, return_exceptions=True)
            return await asyncio.wait_for(future, timeout)
        finally:
            self.remove_listener(waiter)

    async def listen_until(
        self,
        event_type: str,
        predicate: Callable[[RealtimeServerEvent], bool] = None,
        inclusive=True,
        timeout: float = 600,
    ):
        """
        Yield all server events until the next event of a given type, waiting up to *timeout*.

        If an exception happens, raise it.
        """
        break_sentinel = object()
        q = asyncio.Queue()

        async def listener(e: RealtimeServerEvent | BaseException):
            if e.type == event_type and (predicate is None or predicate(e)):
                if inclusive:
                    await q.put(e)
                await q.put(break_sentinel)
            else:
                await q.put(e)

        self.add_listener(listener, return_exceptions=True)
        try:
            async with asyncio.timeout(timeout):
                while True:
                    item = await q.get()
                    if item is break_sentinel:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    yield item
        finally:
            self.remove_listener(listener)

    async def _ws_task(self):
        """Main websocket receive loop."""

        async def _call_listeners(data):
            results = await asyncio.gather(
                *(callback(data) for callback in self.listeners.values()), return_exceptions=True
            )
            # log any exceptions
            for r in results:
                if isinstance(r, BaseException):
                    log.exception("Exception in callback when handling WS event:", exc_info=r)

        is_terminating = False
        try:
            await self._set_connection_state(ConnectionState.CONNECTING)
            async with self.client.beta.realtime.connect(model=self.model) as self._conn:
                await self._set_connection_state(ConnectionState.CONNECTED)
                self.has_connected_once = True
                self._ws_connected.set_result(True)
                async for event in self._conn:
                    # noinspection PyBroadException
                    try:
                        log.debug(f"<<< {event!r}")
                        # process our event first, always
                        await self._handle_server_event(event)
                        # get listeners, call them - listeners can use the result of the processing if needed
                        await _call_listeners(event)
                    except NoPropagate:
                        continue
                    except Exception as e:
                        log.exception("Exception when handling WS event:")
                        await _call_listeners(e)
        except asyncio.CancelledError:
            # the ws loop was cancelled explicitly, so we are shutting down
            is_terminating = True
            raise
        except websockets.ConnectionClosedError as e:
            log.error(f"WS connection closed unexpectedly: {e}", exc_info=e)
            await _call_listeners(e)
        except Exception as e:
            log.exception("Exception when connecting to the websocket:")
            self._ws_connected.set_exception(e)
            await _call_listeners(e)
            raise
        finally:
            self._conn = None
            await self._set_connection_state(
                ConnectionState.DISCONNECTED if not is_terminating else ConnectionState.TERMINATING
            )
            self.ws_task = None

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
        if event.error.code == "session_expired":
            # close the WS - this will emit a DISCONNECTED lifecycle event and kill ws_task
            await self._conn.close()

    @server_event_handler("session.created")
    async def _handle_session_created(self, event: oait.SessionCreatedEvent):
        self._session_created.set()
        self.session_id = event.session.id
        self.session_config = event.session

    @server_event_handler("session.updated")
    async def _handle_session_updated(self, event: oait.SessionUpdatedEvent):
        self.session_id = event.session.id
        self.session_config = event.session

    # input_audio_buffer
    @server_event_handler("input_audio_buffer.committed")
    async def _handle_input_audio_buffer_committed(self, event: oait.InputAudioBufferCommittedEvent):
        if not self.save_audio:
            return

        # save a copy of the committed audio
        # if we're in VAD mode, we've received timestamps to slice the buffer
        # otherwise assume we just sent the whole buffer
        if (
            self._last_speech_started is not None
            and self._last_speech_stopped is not None
            and self._last_speech_stopped > self._last_speech_started
        ):
            # 24kHz * 2B samples = 48KBps -> 48 bytes/ms
            start_idx = (self._last_speech_started - self._input_audio_buffer_offset) * 48
            end_idx = (self._last_speech_stopped - self._input_audio_buffer_offset) * 48
            committed_audio_b64 = base64.b64encode(self.input_audio_buffer[start_idx:end_idx]).decode()
            self._input_audio_buffer_offset = self._last_speech_stopped
            del self.input_audio_buffer[:end_idx]
        else:
            committed_audio_b64 = base64.b64encode(self.input_audio_buffer).decode()
            self._input_audio_buffer_offset += len(self.input_audio_buffer) // 48
            self.input_audio_buffer.clear()
        self._last_speech_started = self._last_speech_stopped = None

        # in a task, wait for the conversation.item.created event that corresponds to this
        async def _wait_for_conv_item():
            await self.wait_for("conversation.item.created", lambda e: e.item.id == event.item_id, timeout=10)
            # item = self.conversation_items.get(event.item_id)
            # content = next(c for c in item.content if c.type == "input_audio")  # can probably do content 0?
            content = self.get_item_content(event.item_id, 0)
            if content.audio is None:
                content.audio = committed_audio_b64
            else:
                content.audio += committed_audio_b64

        create_task(_wait_for_conv_item())

    @server_event_handler("input_audio_buffer.cleared")
    async def _handle_input_audio_buffer_cleared(self, _: oait.InputAudioBufferClearedEvent):
        self.input_audio_buffer.clear()

    @server_event_handler("input_audio_buffer.speech_started")
    async def _handle_input_audio_buffer_speech_started(self, event: oait.InputAudioBufferSpeechStartedEvent):
        self._last_speech_started = event.audio_start_ms

    @server_event_handler("input_audio_buffer.speech_stopped")
    async def _handle_input_audio_buffer_speech_stopped(self, event: oait.InputAudioBufferSpeechStoppedEvent):
        self._last_speech_stopped = event.audio_end_ms

    @server_event_handler("conversation.created")
    async def _handle_conversation_created(self, event: oait.ConversationCreatedEvent):
        self.conversation_id = event.conversation.id

    # conversation.item
    @server_event_handler("conversation.item.created")
    async def _handle_conversation_item_created(self, event: oait.ConversationItemCreatedEvent):
        item_id = event.item.id
        self.conversation_items[item_id] = event.item
        if not event.previous_item_id:
            self.conversation_item_order.append(item_id)
        else:
            try:
                previous_item_idx = self.conversation_item_order.index(event.previous_item_id)
            except ValueError:
                self.conversation_item_order.append(item_id)
            else:
                self.conversation_item_order.insert(previous_item_idx + 1, item_id)

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

    # response
    @server_event_handler("response.created")
    async def _handle_response_created(self, event: oait.ResponseCreatedEvent):
        self.responses[event.response.id] = event.response
        for item in event.response.output:
            self.conversation_items[item.id] = item

    @server_event_handler("response.content_part.added")
    async def _handle_response_content_part_added(self, event: oait.ResponseContentPartAddedEvent):
        item = self.conversation_items.get(event.item_id)
        if item is None:
            log.warning(f"Got response.content_part.added event referencing item that does not exist: {event.item_id}")
            return
        if event.content_index < len(item.content):
            log.warning(f"Got response.content_part.added event but item content might be out of order!")
            log.debug(item)
            log.debug(event)
            return
        # noinspection PyTypeChecker
        # for some reason oait.ConversationItemContent doesn't believe in type="audio", but that's what we get
        item.content.append(event.part)

    @server_event_handler("response.audio.delta")
    async def _handle_response_audio_delta(self, event: oait.ResponseAudioDeltaEvent):
        if not self.save_audio:
            return
        content = self.get_item_content(event.item_id, event.content_index)
        if content.audio is None:
            content.audio = event.delta
        else:
            content.audio += event.delta

    @server_event_handler("response.done")
    async def _handle_response_done(self, event: oait.ResponseDoneEvent):
        # weird server-side thing where it can send cancelled after completed
        if (
            event.response.status == "cancelled"
            and event.response.id in self.responses
            and self.responses[event.response.id].status == "completed"
        ):
            raise NoPropagate
        self.responses[event.response.id] = event.response
        for item in event.response.output:
            self.conversation_items[item.id] = merge_conversation_items(item, self.conversation_items.get(item.id))

    @server_event_handler("rate_limits.updated")
    async def _handle_rate_limits_updated(self, event: oait.RateLimitsUpdatedEvent):
        pass  # todo

    # ==== helpers ====
    def get_item_content(self, item_id: str, content_index: int) -> oait.ConversationItemContent | None:
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


def merge_conversation_items(new: oait.ConversationItem, old: oait.ConversationItem) -> oait.ConversationItem:
    """
    Merge the final conversation item (i.e. *new* yielded by the server) with the partial cached by the client.
    This is needed because the server does not emit the final audio in the final conversation item.
    """
    if new.id != old.id:
        raise ValueError("Cannot merge conversation items with differing IDs.")
    if old.content is None:
        return new
    elif new.content is None:
        new_content = old.content
    else:
        # the only thing that really needs a merge is the content
        new_content = [
            merge_conversation_item_contents(n, o) for n, o in itertools.zip_longest(new.content, old.content)
        ]
    return new.model_copy(update={"content": new_content})


def merge_conversation_item_contents(
    new: oait.ConversationItemContent, old: oait.ConversationItemContent | None
) -> oait.ConversationItemContent:
    if old is None:
        return new
    # copy audio from old if new doesn't have it
    new_data = {}
    if not new.audio and old.audio:
        new_data["audio"] = old.audio
    return new.model_copy(update=new_data)


class NoPropagate(Exception):
    """Do not send this event to other listeners -- it is a serverside behavioural error."""
