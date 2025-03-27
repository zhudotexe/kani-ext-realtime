import asyncio
import base64
import collections
import contextlib
import itertools
import logging
import os
import warnings
from typing import Any, AsyncIterable, Callable

import openai.types.beta.realtime as oait
from kani import AIFunction, ChatMessage, ExceptionHandleResult, Kani, ToolCall
from kani.engines.base import BaseCompletion, BaseEngine, Completion
from kani.exceptions import FunctionCallException
from kani.models import ChatRole, FunctionCall, QueryType
from kani.streaming import DummyStream, StreamManager
from openai import AsyncOpenAI as OpenAIClient
from openai.types.beta.realtime.response_create_event_param import Response as ResponseCreateParams
from typing_extensions import Unpack

from . import interop
from ._internal import ensure_async
from .session import ConnectionState, RealtimeSession

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
    r"""
    In addition to all of :class:`kani.Kani`\ 's method, the OpenAIRealtimeKani provides the following two methods
    for interacting with the realtime API.
    """

    def __init__(
        self,
        # realtime session args
        api_key: str = None,
        model="gpt-4o-realtime-preview-2024-12-17",
        *,
        organization: str = None,
        retry: int = 5,
        ws_base: str = "wss://api.openai.com/v1",
        headers: dict = None,
        client: OpenAIClient = None,
        # kani args
        system_prompt: str = None,
        chat_history: list[ChatMessage] = None,
        always_included_messages: list[ChatMessage] = None,
        **generation_args: Unpack[ResponseCreateParams],
    ):
        """
        :param api_key: Your OpenAI API key. By default, the API key will be read from the `OPENAI_API_KEY` environment
            variable.
        :param model: The id of the realtime model to use (default "gpt-4o-realtime-preview-2024-10-01").
        :param system_prompt: The system prompt to provide to the LM. The prompt *will* be included in chat_history.

            .. note::
                For interacting with the Realtime API, you may instead wish to provide session instructions by providing
                the ``instructions`` key to :class:`.SessionConfig` in your :meth:`connect` call.
        :param chat_history: The chat history to start with (not including system prompt or always included messages),
            e.g. for few-shot prompting. By default, each kani starts with a new conversation session.
        :param always_included_messages: Prepended to ``chat_history``.

            .. warning::
                Unlike normal Kanis, due to the server-managed nature of the OpenAI realtime API, messages marked as
                always included may not always be included by the server. These messages will instead be prepended to
                any ``chat_history`` and *will* be included in the ``chat_history`` attribute.
        :param organization: The OpenAI organization to use in requests. By default, the org ID would be read from the
            `OPENAI_ORG_ID` environment variable (defaults to the API key's default org if not set).
        :param retry: How many times the engine should retry failed HTTP calls with exponential backoff (default 5).
        :param ws_base: The base WebSocket URL to connect to (default "wss://api.openai.com/v1/realtime").
        :param headers: A dict of HTTP headers to include with each request.
        :param client: An instance of `openai.AsyncOpenAI <https://github.com/openai/openai-python>`_
            (for reusing the same client in multiple engines).
            You must specify exactly one of ``(api_key, client)``. If this is passed the ``organization``, ``retry``,
            ``api_base``, and ``headers`` params will be ignored.
        :param generation_args: The arguments to pass to the ``response.create`` call with each request. See
            https://platform.openai.com/docs/api-reference/realtime-client-events/response/create for a full list of
            params. Specifically, these arguments will be passed as the ``response`` key.

            .. warning::
                This will only affect direct model queries (i.e., through :meth:`.chat_round` or :meth:`.full_round`) --
                these args will not affect the :meth:`.full_duplex` behaviour. Pass configuration options to
                :meth:`.connect` instead for full duplex configuration.
        """

        if headers is None:
            headers = {}
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")

        self._always_included_messages = always_included_messages
        self._chat_history = chat_history
        self.generation_args = generation_args
        self.response_timeout = 600
        self.retry_attempts = retry

        self.client = client or OpenAIClient(
            api_key=api_key,
            organization=organization,
            max_retries=retry,
            websocket_base_url=ws_base,
            default_headers=headers,
        )
        self.session = RealtimeSession(model=model, client=self.client, retry_attempts=retry)
        """The underlying state of the OpenAI Realtime API. Used for lower-level API operations."""

        Kani.__init__(
            self,
            engine=DummyEngine(),
            system_prompt=system_prompt,
            always_included_messages=always_included_messages,
            chat_history=chat_history,
        )
        self.lock = contextlib.nullcontext()

    # ===== lifecycle =====
    async def connect(self, **session_config: Unpack[oait.SessionCreateParams]):
        """Connect to the WS and update the internal state until the engine is closed."""
        if self.is_connected:
            raise RuntimeError("This RealtimeKani has already connected to the socket.")

        # set default kani kwargs
        session_config.setdefault("input_audio_format", "pcm16")
        session_config.setdefault("output_audio_format", "pcm16")
        session_config.setdefault("input_audio_transcription", {"model": "whisper-1"})
        session_config.setdefault("turn_detection", {"type": "server_vad"})
        session_config.setdefault("tool_choice", "auto")

        # connect to the WS
        await self.session.connect()

        # configure tools
        tool_defs = list(map(interop.ai_function_to_tool, self.functions.values()))
        session_config["tools"] = list(itertools.chain(session_config.get("tools", []), tool_defs))

        # send session config over WS
        await self.session.session_update(session_config)

        # send chat history over ws
        history_to_upload = []
        if self.always_included_messages:
            warnings.warn(
                "Due to the server-managed nature of the OpenAI realtime API, messages marked as always included may"
                " not always be included by the server."
            )
            history_to_upload.extend(self.always_included_messages)
        if self._chat_history:
            history_to_upload.extend(self._chat_history)

        for idx, msg in enumerate(history_to_upload):
            log.debug(f"Uploading conversation history item {idx} / {len(history_to_upload)}")
            await self.session.conversation_item_create_from_chat_message(msg)

    @property
    def is_connected(self):
        return (
            self.session.connection_state == ConnectionState.CONNECTED
            or self.session.connection_state == ConnectionState.CONNECTING
        )

    # ===== weird overrides =====
    @property
    def always_included_messages(self):
        return self._always_included_messages

    @always_included_messages.setter
    def always_included_messages(self, value):
        if self.is_connected:
            raise ValueError("The chat history cannot be directly modified after connecting to the WS.")
        self._always_included_messages = value

    @property
    def chat_history(self):
        if not self.session.has_connected_once:
            return self._chat_history
        # read chat items from session, grouping by responses (model output group or user input)
        return interop.chat_history_from_session_state(self.session)

    @chat_history.setter
    def chat_history(self, value):
        if self.is_connected:
            raise ValueError("The chat history cannot be directly modified after connecting to the WS.")
        self._chat_history = value

    async def add_to_history(self, message: ChatMessage):
        # intentionally do nothing here
        # todo maybe call a conversation.item.add if not in session?
        pass

    async def get_prompt(self) -> list[ChatMessage]:
        return []

    # ===== kani iface =====
    # noinspection PyMethodOverriding
    async def get_model_completion(
        self, include_functions: bool = True, **kwargs: Unpack[ResponseCreateParams]
    ) -> Completion:
        for retry_idx in range(self.retry_attempts):
            try:
                return await self._get_model_completion(include_functions, **kwargs)
            except Exception as e:
                if retry_idx + 1 == self.retry_attempts:
                    raise
                sleep_for = 2**retry_idx
                log.warning(
                    f"Got exception in get_model_completion (attempt {retry_idx + 1} of {self.retry_attempts}),"
                    f" sleeping for {sleep_for} sec and retrying...",
                    exc_info=e,
                )
                await asyncio.sleep(sleep_for)

    async def _get_model_completion(
        self, include_functions: bool = True, **kwargs: Unpack[ResponseCreateParams]
    ) -> Completion:
        """Request a completion now and return it."""
        if not include_functions:
            kwargs["tool_choice"] = "none"
        generation_kwargs = self.generation_args | kwargs
        response_created_data = await self.session.response_create(**generation_kwargs)
        response: oait.ResponseDoneEvent = await self.session.wait_for(
            "response.done", lambda e: e.response.id == response_created_data.response.id, timeout=self.response_timeout
        )
        message = interop.response_to_chat_message(response.response)
        return Completion(
            message=message,
            prompt_tokens=response.response.usage.input_tokens,
            completion_tokens=response.response.usage.output_tokens,
        )

    async def get_model_stream(
        self,
        include_functions: bool = True,
        audio_callback: Callable[[bytes], Any] = None,
        **kwargs: Unpack[ResponseCreateParams],
    ) -> AsyncIterable[str | BaseCompletion]:
        for retry_idx in range(self.retry_attempts):
            try:
                async for item in self._get_model_stream(include_functions, audio_callback, **kwargs):
                    yield item
            except Exception as e:
                if retry_idx + 1 == self.retry_attempts:
                    raise
                sleep_for = 2**retry_idx
                log.warning(
                    f"Got exception in get_model_stream (attempt {retry_idx + 1} of {self.retry_attempts}),"
                    f" sleeping for {sleep_for} sec and retrying...",
                    exc_info=e,
                )
                await asyncio.sleep(sleep_for)

    async def _get_model_stream(
        self,
        include_functions: bool = True,
        audio_callback: Callable[[bytes], Any] = None,
        **kwargs: Unpack[ResponseCreateParams],
    ) -> AsyncIterable[str | BaseCompletion]:
        """
        Request a completion and stream from the model until the next response.done event. Only yield events from this
        completion.
        """
        if not include_functions:
            kwargs["tool_choice"] = "none"
        audio_callback = ensure_async(audio_callback)
        generation_kwargs = self.generation_args | kwargs

        response_created_data = await self.session.response_create(**generation_kwargs)

        async for ws_event in self.session.listen_until(
            "response.done",
            lambda e: e.response.id == response_created_data.response.id,
            inclusive=True,
            timeout=self.response_timeout,
        ):
            match ws_event:
                case oait.ResponseTextDeltaEvent(response_id=response_created_data.response.id, delta=text):
                    yield text
                case oait.ResponseAudioTranscriptDeltaEvent(response_id=response_created_data.response.id, delta=text):
                    yield text
                case oait.ResponseAudioDeltaEvent(response_id=response_created_data.response.id, delta=audio_b64):
                    await audio_callback(base64.b64decode(audio_b64))
                case oait.ResponseDoneEvent(response=response):
                    message = interop.response_to_chat_message(response)
                    completion = Completion(
                        message=message,
                        prompt_tokens=response.usage.input_tokens,
                        completion_tokens=response.usage.output_tokens,
                    )
                    yield completion

    async def _full_round(self, query: QueryType, *, max_function_rounds: int, _kani_is_stream: bool, **kwargs):
        """Underlying handler for full_round with stream support."""
        retry = 0
        function_rounds = 0
        is_model_turn = True

        if query is not None:
            msg = ChatMessage.user(query)
            await self.add_to_history(msg)
            await self.session.conversation_item_create_from_chat_message(msg)

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
                await self.session.conversation_item_create_from_chat_message(result.message)

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
        audio_callback: Callable[[bytes], Any] = None,
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
                # this example code does NOT account for printing multiple concurrent message streams
                # it simply prints tokens as they are received
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
            :class:`.RealtimeSession`. For example, you might use the following to request a
            response when serverside VAD is disabled:

            .. code-block:: python

                import openai.types.beta.realtime as oait

                await ai.session.send(oait.ResponseCreateEvent(type="response.create"))

            See https://platform.openai.com/docs/api-reference/realtime-client-events for more details.

        :param audio_stream: An async iterator that emits audio frames (bytes). Audio frames should be encoded as
            raw 16 bit PCM audio at 24kHz, 1 channel, little-endian. You can use
            ``easyaudiostream.get_mic_stream_async()`` to get such a stream.
        :param audio_callback: An async function that consumes audio frames as emitted by the model. You can use
            ``easyaudiostream.play_raw_audio()`` to play audio over the system speakers.
        """
        audio_callback = ensure_async(audio_callback)
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
                case oait.ConversationItemCreatedEvent(
                    item=oait.ConversationItem(type="message", id=item_id, role=role)
                ):
                    streamer_q = streamer_queues[item_id]
                    await yielder_q.put(StreamManager(yield_from_queue(streamer_q), role=ChatRole(role)))
                # ===== streaming items (asst) =====
                case oait.ResponseTextDeltaEvent(item_id=item_id, delta=text) | oait.ResponseAudioTranscriptDeltaEvent(
                    item_id=item_id, delta=text
                ):
                    await streamer_queues[item_id].put(text)
                case oait.ResponseDoneEvent(response=response):
                    if response.status == "cancelled" and not response.output:
                        return
                    message = interop.response_to_chat_message(response)
                    completion = Completion(
                        message=message,
                        prompt_tokens=response.usage.input_tokens,
                        completion_tokens=response.usage.output_tokens,
                    )
                    await self.add_completion_to_history(completion)
                    for item_id in set(i.id for i in response.output if i.type == "message"):
                        q = streamer_queues[item_id]
                        await q.put(completion)
                        await q.put(break_sentinel)
                        streamer_queues.pop(item_id)
                # ===== streaming items (user) =====
                case oait.ConversationItemInputAudioTranscriptionCompletedEvent(item_id=item_id, transcript=text):
                    await streamer_queues[item_id].put(text.strip())
                    # emit a completion too
                    item = self.session.conversation_items.get(item_id)
                    assert item.type == "message"
                    role = ChatRole(item.role)
                    content = list(map(interop.content_part_to_message_part, item.content))
                    message = ChatMessage(role=role, content=content)
                    await self.add_to_history(message)
                    completion = Completion(message=message, prompt_tokens=0, completion_tokens=0)
                    await streamer_queues[item_id].put(completion)
                    await streamer_queues[item_id].put(break_sentinel)
                    streamer_queues.pop(item_id)
                # ===== audio =====
                case oait.ResponseAudioDeltaEvent(delta=audio_b64):
                    await audio_callback(base64.b64decode(audio_b64))
                # ===== function calling =====
                case oait.ResponseOutputItemDoneEvent(
                    item=oait.ConversationItem(
                        type="function_call", status="completed", call_id=call_id, name=name, arguments=args
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
                    await self.session.conversation_item_create_from_chat_message(result.message)
                    await yielder_q.put(DummyStream(result.message))
                    # request a new completion
                    await self.session.response_create()

        # audio sender
        async def audio_sender_task():
            async for frame in audio_stream:
                await self.session.input_audio_buffer_append(frame)
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
