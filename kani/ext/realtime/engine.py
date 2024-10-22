import asyncio
import base64
import contextlib
import logging
import os
import warnings
from typing import AsyncIterable, Awaitable, Callable

from kani import AIFunction, ChatMessage, ExceptionHandleResult, Kani, ToolCall
from kani.engines.base import BaseCompletion, BaseEngine, Completion
from kani.exceptions import FunctionCallException
from kani.models import ChatRole, QueryType
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
        self._pending_msgs = []

        Kani.__init__(
            self,
            engine=DummyEngine(),
            system_prompt=system_prompt,
            always_included_messages=always_included_messages,
            chat_history=chat_history,
        )
        self.session = RealtimeSession(
            api_key=api_key, model=model, ws_base=ws_base, headers=headers, **generation_args
        )

        self.lock = contextlib.nullcontext()

    # ===== lifecycle =====
    async def connect(self, session_config: oaimodels.SessionConfig = None):
        """Connect to the WS and update the internal state until the engine is closed."""
        if self._has_connected:
            raise RuntimeError("This RealtimeKani has already connected to the socket.")
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

    @chat_history.setter
    def chat_history(self, value):
        if self._has_connected:
            raise ValueError("The chat history cannot be directly modified after connecting to the WS.")
        self._chat_history = value

    async def add_to_history(self, message: ChatMessage):
        # intentionally do nothing here
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
                    # await q.put(interop.AudioPart(oai_type="audio", audio_b64=audio_b64, transcript=""))
                # case server_events.ResponseOutputItemDone(
                #     response_id=response_created_data.response.id,
                #     item=oaimodels.FunctionCallConversationItem(
                #         status="completed", call_id=call_id, name=name, arguments=args
                #     ),
                # ):
                #     await q.put(ToolCall.from_function_call(FunctionCall(name=name, arguments=args), call_id))
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
