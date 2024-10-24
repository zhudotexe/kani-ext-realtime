import asyncio
import base64
import collections
import contextlib
import logging
import os
import warnings
from typing import AsyncIterable, Awaitable, Callable

from kani import AIFunction, ChatMessage, ExceptionHandleResult, Kani, ToolCall
from kani.engines.base import BaseCompletion, BaseEngine, Completion
from kani.exceptions import FunctionCallException
from kani.models import ChatRole, FunctionCall, QueryType
from kani.streaming import DummyStream, StreamManager

from . import interop, models as oaimodels
from .events import client as client_events, server as server_events
from .session import RealtimeSession

log = logging.getLogger(__name__)


class DummyEngine(BaseEngine):
    max_context_size = 128000

    def message_len(self, message: ChatMessage) -> int:
        return len(message.text) // 4

    def function_token_reserve(self, functions: list[AIFunction]) -> int:
        return 0

    async def predict(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
    ) -> BaseCompletion:
        raise NotImplementedError


class OpenAIRealtimeKani(Kani):
    def __init__(
        self,
        # realtime session args
        api_key: str = None,
        model="gpt-4o-realtime-preview-2024-10-01",
        *,
        ws_base: str = "wss://api.openai.com/v1/realtime",
        headers: dict = None,
        # kani args
        system_prompt: str = None,
        always_included_messages: list[ChatMessage] = None,
        chat_history: list[ChatMessage] = None,
        **generation_args,
    ):
        if headers is None:
            headers = {}
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")

        self._has_connected = False
        self._always_included_messages = always_included_messages
        self._chat_history = chat_history

        Kani.__init__(
            self,
            engine=DummyEngine(),
            system_prompt=system_prompt,
            always_included_messages=always_included_messages,
            chat_history=chat_history,
        )
        self.lock = contextlib.nullcontext()

        self.session = RealtimeSession(
            api_key=api_key, model=model, ws_base=ws_base, headers=headers, **generation_args
        )
        """The underlying state of the OpenAI Realtime API. Used for lower-level API operations."""

    # ===== lifecycle =====
    async def connect(self, session_config: oaimodels.SessionConfig = None):
        """Connect to the WS and update the internal state until the engine is closed."""
        if self._has_connected:
            raise RuntimeError("This RealtimeKani has already connected to the socket.")
        if session_config is None:
            # we want input_audio_transcription to be on by default - see models for default config
            session_config = oaimodels.SessionConfig()

        self._has_connected = True
        await self.session.connect()

        # configure tools
        if session_config:
            tool_defs = session_config.tools + list(map(interop.ai_function_to_tool, self.functions.values()))
            session_config.tools = tool_defs
        else:
            tool_defs = list(map(interop.ai_function_to_tool, self.functions.values()))
            session_config = self.session.session_config.model_copy(update={"tools": tool_defs})

        # send session config over WS
        await self.session.send(client_events.SessionUpdate(session=session_config))
        await self.session.wait_for("session.updated")

        # send chat history over ws
        if self.always_included_messages:
            warnings.warn(
                "Due to the server-managed nature of the OpenAI realtime API, messages marked as always included may"
                " not always be included by the server."
            )
            for msg in self.always_included_messages:
                for item in interop.chat_message_to_conv_items(msg):
                    await self.session.send(client_events.ConversationItemCreate(item=item))
                    await self.session.wait_for("conversation.item.created")
        if self._chat_history:
            for msg in self._chat_history:
                for item in interop.chat_message_to_conv_items(msg):
                    await self.session.send(client_events.ConversationItemCreate(item=item))
                    await self.session.wait_for("conversation.item.created")

    # ===== weird overrides =====
    @property
    def always_included_messages(self):
        return self._always_included_messages

    @always_included_messages.setter
    def always_included_messages(self, value):
        if self._has_connected:
            raise ValueError("The chat history cannot be directly modified after connecting to the WS.")
        self._always_included_messages = value

    @property
    def chat_history(self):
        if not self._has_connected:
            return self._chat_history
        # todo read chat items from session
        # todo return immutable

    @chat_history.setter
    def chat_history(self, value):
        if self._has_connected:
            raise ValueError("The chat history cannot be directly modified after connecting to the WS.")
        self._chat_history = value

    async def add_to_history(self, message: ChatMessage):
        # intentionally do nothing here
        # todo maybe call a conversation.item.add if not in session?
        pass

    async def get_prompt(self) -> list[ChatMessage]:
        return []

    # ===== kani iface =====
    async def get_model_completion(self, include_functions: bool = True, **kwargs) -> Completion:
        """Request a completion now and return it."""
        if not include_functions:
            kwargs["tool_choice"] = "none"
        await self.session.send(
            client_events.ResponseCreate(response=self.session.session_config.model_copy(update=kwargs))
        )
        response_created_data: server_events.ResponseCreated = await self.session.wait_for("response.created")
        response: server_events.ResponseDone = await self.session.wait_for(
            "response.done", lambda e: e.response.id == response_created_data.response.id
        )
        message = interop.response_to_chat_message(response.response)
        return Completion(
            message=message, prompt_tokens=response.usage.input_tokens, completion_tokens=response.usage.output_tokens
        )

    async def get_model_stream(
        self, include_functions: bool = True, audio_callback: Callable[[bytes], Awaitable] = None, **kwargs
    ) -> AsyncIterable[str | BaseCompletion]:
        """
        Request a completion and stream from the model until the next response.done event. Only yield events from this
        completion.
        """
        if not include_functions:
            kwargs["tool_choice"] = "none"
        if audio_callback is None:

            async def audio_callback(_):
                pass

        response_config = self.session.session_config.model_copy(update=kwargs) if self.session.session_config else None
        await self.session.send(client_events.ResponseCreate(response=response_config))
        response_created_data: server_events.ResponseCreated = await self.session.wait_for("response.created")

        break_sentinel = object()
        completion = None
        q = asyncio.Queue()

        async def listener(e):
            match e:
                case server_events.ResponseTextDelta(response_id=response_created_data.response.id, delta=text):
                    await q.put(text)
                case server_events.ResponseAudioTranscriptDelta(
                    response_id=response_created_data.response.id, delta=text
                ):
                    await q.put(text)
                case server_events.ResponseAudioDelta(response_id=response_created_data.response.id, delta=audio_b64):
                    await audio_callback(base64.b64decode(audio_b64))
                case server_events.ResponseDone(response=response):
                    message = interop.response_to_chat_message(response)
                    nonlocal completion
                    completion = Completion(
                        message=message,
                        prompt_tokens=response.usage.input_tokens,
                        completion_tokens=response.usage.output_tokens,
                    )
                    await q.put(break_sentinel)

        self.session.add_listener(listener)
        try:
            while True:
                item = await q.get()
                if item is break_sentinel:
                    log.debug("Got break sentinel, yielding completion")
                    break
                yield item
        finally:
            self.session.remove_listener(listener)
            if completion:
                yield completion

    async def _full_round(self, query: QueryType, *, max_function_rounds: int, _kani_is_stream: bool, **kwargs):
        """Underlying handler for full_round with stream support."""
        retry = 0
        function_rounds = 0
        is_model_turn = True

        if query is not None:
            msg = ChatMessage.user(query)
            await self.add_to_history(msg)
            for item in interop.chat_message_to_conv_items(msg):
                await self.session.send(client_events.ConversationItemCreate(item=item))

        while is_model_turn:
            # do the model prediction (stream or no stream)
            if _kani_is_stream:
                stream = self.get_model_stream(**kwargs)
                manager = StreamManager(stream, role=ChatRole.ASSISTANT, after=self.add_completion_to_history)
                yield manager
                message = await manager.message()
            else:
                completion = await self.get_model_completion(**kwargs)
                message = await self.add_completion_to_history(completion)
                yield message

            # if function call, do it and attempt retry if it's wrong
            if not message.tool_calls:
                return

            # and update results after they are completed
            is_model_turn = False
            should_retry_call = False
            n_errs = 0
            results = await asyncio.gather(*(self._do_tool_call(tc, retry) for tc in message.tool_calls))
            for result in results:
                # save the result to the chat history
                await self.add_to_history(result.message)
                for item in interop.chat_message_to_conv_items(result.message):
                    await self.session.send(client_events.ConversationItemCreate(item=item))

                    # yield it, possibly in dummy streammanager
                    if _kani_is_stream:
                        yield DummyStream(result.message)
                    else:
                        yield result.message

                if isinstance(result, ExceptionHandleResult):
                    is_model_turn = True
                    n_errs += 1
                    # retry if any function says so
                    should_retry_call = should_retry_call or result.should_retry
                else:
                    # allow model to generate response if any function says so
                    is_model_turn = is_model_turn or result.is_model_turn

            # if we encountered an error, increment the retry counter and allow the model to generate a response
            if n_errs:
                retry += 1
                if not should_retry_call:
                    # disable function calling on the next go
                    kwargs["include_functions"] = False
            else:
                retry = 0

            # if we're at the max number of function rounds, don't include functions on the next go
            function_rounds += 1
            if max_function_rounds is not None and function_rounds >= max_function_rounds:
                kwargs["include_functions"] = False

    async def _do_tool_call(self, tc: ToolCall, retry: int):
        # call the method and set the is_tool_call_error attr (if the impl has not already set it)
        try:
            tc_result = await self.do_function_call(tc.function, tool_call_id=tc.id)
            if tc_result.message.is_tool_call_error is None:
                tc_result.message.is_tool_call_error = False
        except FunctionCallException as e:
            tc_result = await self.handle_function_call_exception(tc.function, e, retry, tool_call_id=tc.id)
            tc_result.message.is_tool_call_error = True
        return tc_result

    async def close(self):
        """Disconnect from the WS."""
        await self.session.close()

    # ===== full duplex =====
    async def full_duplex(
        self,
        audio_stream: AsyncIterable[bytes],  # todo what about manual response creates
        audio_callback: Callable[[bytes], Awaitable] = None,
        **kwargs,  # todo this might be a good place for session config too?
    ) -> AsyncIterable[StreamManager]:
        """
        Stream audio bytes from the given stream to the realtime model.

        Yields a stream for each conversation item created (both USER and ASSISTANT). Each stream will be related to
        exactly one conversation item (i.e., message), and multiple streams may emit simultaneously.

        To consume tokens from a stream, use this class as so:

        .. code-blocK:: python

            stream_tasks = set()

            async def handle_stream(stream):
                # do processing for a single message's stream here...
                # this example code does NOT account for multiple simultaneous messages
                async for token in stream:
                    print(token, end="")
                msg = await stream.message()

            async for stream in ai.full_duplex(audio_stream):
                task = asyncio.create_task(handle_stream(stream))
                # to keep a live reference to the task
                # see https://docs.python.org/3/library/asyncio-task.html#creating-tasks
                stream_tasks.add(task)
                task.add_done_callback(stream_tasks.discard)

        Check out the implementation of :func:`.chat_in_terminal_audio_async` for more in-depth stream handling (e.g.,
        printing out streams simultaneously without clobbering other messages' outputs).

        Each :class:`.StreamManager` object yielded by this method contains a :attr:`.StreamManager.role` attribute
        that can be used to determine if a message is from the user, engine or a function call. This attribute will be
        available *before* iterating over the stream.

        .. note::
            This method will exit once the ``audio_stream`` is exhausted (i.e., the iterator raises StopAsyncIteration).

        .. note::
            For lower-level control over the realtime chat session (e.g. to send events directly to the server), see
            :class:`.RealtimeSession` and :module:`.events`. For example, you might use the following to request a
            response when serverside VAD is disabled:

            .. code-block:: python

                from kani.ext.realtime import events

                await ai.session.send(events.client.ResponseCreate())

            See https://platform.openai.com/docs/api-reference/realtime-client-events for more details.

        :param audio_stream: An async iterator that emits audio frames (bytes). Audio frames should be encoded as
            raw 16 bit PCM audio at 24kHz, 1 channel, little-endian.
        :param audio_callback: An async function that consumes audio frames as emitted by the model. See
            :func:`.play_audio` for an example.
        """
        if audio_callback is None:

            async def audio_callback(_):
                pass

        break_sentinel = object()
        # streamer for item with given ID reads elements from their q, stored here
        streamer_queues = collections.defaultdict(asyncio.Queue)
        yielder_q = asyncio.Queue()  # streamers to yield

        # helper for yielding
        async def yield_from_queue(q: asyncio.Queue):
            while True:
                item_to_yield = await q.get()
                if item_to_yield is break_sentinel:
                    break
                yield item_to_yield

        # main event handler
        async def listener(e):
            """On event from server, route the event to the right streamer or yield a new streamer"""
            match e:
                # ===== new conversation item =====
                # we only care about messages here - function calls are handled elsewhere
                case server_events.ConversationItemCreated(
                    item=oaimodels.MessageConversationItem(id=item_id, role=role)
                ):
                    streamer_q = streamer_queues[item_id]
                    await yielder_q.put(StreamManager(yield_from_queue(streamer_q), role=ChatRole(role)))
                # ===== streaming items (asst) =====
                case server_events.ResponseTextDelta(
                    item_id=item_id, delta=text
                ) | server_events.ResponseAudioTranscriptDelta(item_id=item_id, delta=text):
                    await streamer_queues[item_id].put(text)
                case server_events.ResponseDone(response=response):
                    message = interop.response_to_chat_message(response)
                    completion = Completion(
                        message=message,
                        prompt_tokens=response.usage.input_tokens,
                        completion_tokens=response.usage.output_tokens,
                    )
                    for item_id in set(i.id for i in response.output if i.type == "message"):
                        q = streamer_queues[item_id]
                        await q.put(completion)
                        await q.put(break_sentinel)
                        streamer_queues.pop(item_id)
                # ===== streaming items (user) =====
                case server_events.ConversationItemInputAudioTranscriptionCompleted(item_id=item_id, transcript=text):
                    await streamer_queues[item_id].put(text.strip())
                    # emit a completion too
                    item = self.session.conversation_items.get(item_id)
                    assert isinstance(item, oaimodels.MessageConversationItem)
                    role = ChatRole(item.role)
                    content = list(map(interop.content_part_to_message_part, item.content))
                    message = ChatMessage(role=role, content=content)
                    completion = Completion(message=message, prompt_tokens=0, completion_tokens=0)
                    await streamer_queues[item_id].put(completion)
                    await streamer_queues[item_id].put(break_sentinel)
                    streamer_queues.pop(item_id)
                # ===== audio =====
                case server_events.ResponseAudioDelta(delta=audio_b64):
                    await audio_callback(base64.b64decode(audio_b64))
                # ===== function calling =====
                case server_events.ResponseOutputItemDone(
                    item=oaimodels.FunctionCallConversationItem(
                        status="completed", call_id=call_id, name=name, arguments=args
                    )
                ):
                    tc = ToolCall.from_function_call(FunctionCall(name=name, arguments=args), call_id)
                    # emit a dummystream with the function call
                    tc_message = ChatMessage.assistant(content=None, tool_calls=[tc])
                    await yielder_q.put(DummyStream(tc_message))
                    # actually call it and req a new completion with data
                    result = await self._do_tool_call(tc, 0)
                    # save the result to the chat history
                    await self.add_to_history(result.message)
                    for item in interop.chat_message_to_conv_items(result.message):
                        await self.session.send(client_events.ConversationItemCreate(item=item))
                        await yielder_q.put(DummyStream(result.message))
                    # request a new completion
                    await self.session.send(client_events.ResponseCreate())

        # audio sender
        async def audio_sender_task():
            async for frame in audio_stream:
                data = base64.b64encode(frame).decode()
                await self.session.send(client_events.InputAudioBufferAppend(audio=data))
            # when we are out of audio, tell the outer loop to break
            await yielder_q.put(break_sentinel)

        # add the listener, start the task to fwd audio frames, and start emitting
        self.session.add_listener(listener)
        audio_task = asyncio.create_task(audio_sender_task())
        try:
            async for stream_manager in yield_from_queue(yielder_q):
                yield stream_manager
        finally:
            audio_task.cancel()
            self.session.remove_listener(listener)
