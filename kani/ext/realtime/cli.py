"""The CLI utilities allow you to play with a chat session directly from a terminal."""

import asyncio
import logging
import os
import sys
import textwrap
from typing import Literal, overload

from kani.kani import Kani
from kani.models import ChatRole
from kani.streaming import StreamManager
from kani.utils.cli import print_stream, print_width
from kani.utils.message_formatters import assistant_message_contents_thinking, assistant_message_thinking

from . import interop
from .audio import get_audio_stream, play_audio
from .engine import OpenAIRealtimeKani
from .events import client as client_events


async def ainput(string: str) -> str:
    await asyncio.get_event_loop().run_in_executor(None, lambda s=string: sys.stdout.write(s))
    return await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)


# pretty much reimplementations of chat_in_terminal but it plays the output if there's audio


async def _chat_in_terminal_round_stream(
    query: str | None,
    kani: Kani,
    *,
    width: int = None,
    show_function_args: bool = False,
    show_function_returns: bool = False,
):
    async for stream in kani.full_round_stream(query, audio_callback=play_audio):
        # assistant
        if stream.role == ChatRole.ASSISTANT:
            await print_stream(stream, width=width, prefix="AI: ")
            msg = await stream.message()
            text = assistant_message_thinking(msg, show_args=show_function_args)
            if text:
                print_width(text, width=width, prefix="AI: ")
        # function
        elif stream.role == ChatRole.FUNCTION and show_function_returns:
            msg = await stream.message()
            print_width(msg.text, width=width, prefix="FUNC: ")


async def _chat_in_terminal_round_completion(
    query: str | None,
    kani: Kani,
    *,
    width: int = None,
    show_function_args: bool = False,
    show_function_returns: bool = False,
):
    async for msg in kani.full_round(query):
        # assistant
        if msg.role == ChatRole.ASSISTANT:
            text = assistant_message_contents_thinking(msg, show_args=show_function_args)
            print_width(text, width=width, prefix="AI: ")
            # play parts
            for part in msg.parts:
                if isinstance(part, interop.AudioPart) and part.audio_b64:
                    await play_audio(part.audio_bytes)
        # function
        elif msg.role == ChatRole.FUNCTION and show_function_returns:
            print_width(msg.text, width=width, prefix="FUNC: ")


async def _chat_in_terminal_full_duplex(
    kani: OpenAIRealtimeKani,
    *,
    ai_first: bool = False,
    width: int | None = None,
    show_function_args: bool = False,
    show_function_returns: bool = False,
    mic_id: int | None = None,
):
    try:
        from rich.live import Live
        import pyaudio
    except ImportError:
        raise ImportError(
            "You must install PyAudio and rich to use the built-in full duplex mode. You can install these dependencies"
            ' with `pip install "kani-ext-realtime[all]"`.'
        ) from None

    # get the audio stream iterator from pyaudio
    audio_stream = get_audio_stream(mic_id)

    # send all the info to a bg manager and start it
    manager = FullDuplexManager(
        kani,
        audio_stream,
        width=width,
        show_function_args=show_function_args,
        show_function_returns=show_function_returns,
    )
    await manager.start()

    # request an initial completion if we want it
    if ai_first:
        await kani.session.send(client_events.ResponseCreate())

    # then show the live data forever
    try:
        with Live(manager.get_display_text(), vertical_overflow="visible", auto_refresh=False) as live:
            while True:
                live.update(manager.get_display_text(), refresh=True)
                await asyncio.sleep(0.25)
    except (KeyboardInterrupt, asyncio.CancelledError):
        await manager.close()


async def chat_in_terminal_audio_async(
    kani: OpenAIRealtimeKani,
    *,
    rounds: int = 0,
    stopword: str = None,
    echo: bool = False,
    ai_first: bool = False,
    width: int = None,
    show_function_args: bool = False,
    show_function_returns: bool = False,
    verbose: bool = False,
    mode: Literal["chat", "stream", "full_duplex"] = "stream",
    mic_id: int = 0,
):
    """Async version of :func:`.chat_in_terminal_audio`.
    Use in environments when there is already an asyncio loop running (e.g. Google Colab).
    """
    if mode not in ("chat", "stream", "full_duplex"):
        raise ValueError('mode must be one of "chat", "stream", "full_duplex"')
    if os.getenv("KANI_DEBUG") is not None:
        logging.basicConfig(level=logging.DEBUG)
    if verbose:
        echo = show_function_args = show_function_returns = True

    if mode == "full_duplex":
        await _chat_in_terminal_full_duplex(
            kani,
            ai_first=ai_first,
            width=width,
            show_function_args=show_function_args,
            show_function_returns=show_function_returns,
            mic_id=mic_id,
        )
        return

    round_num = 0
    while round_num < rounds or not rounds:
        round_num += 1

        # get user query
        if not ai_first or round_num > 0:
            query = await ainput("USER: ")
            query = query.strip()
            if echo:
                print_width(query, width=width, prefix="USER: ")
            if stopword and query == stopword:
                break
        # print completion(s)
        else:
            query = None

        # print completion(s)
        if mode == "stream":
            await _chat_in_terminal_round_stream(
                query,
                kani,
                width=width,
                show_function_args=show_function_args,
                show_function_returns=show_function_returns,
            )
        # completions only
        else:
            await _chat_in_terminal_round_completion(
                query,
                kani,
                width=width,
                show_function_args=show_function_args,
                show_function_returns=show_function_returns,
            )


@overload
def chat_in_terminal_audio(
    kani: Kani,
    *,
    rounds: int = 0,
    stopword: str = None,
    echo: bool = False,
    ai_first: bool = False,
    width: int = None,
    show_function_args: bool = False,
    show_function_returns: bool = False,
    verbose: bool = False,
    mode: Literal["chat", "stream", "full_duplex"] = "stream",
    mic_id: int | None = None,
): ...


def chat_in_terminal_audio(kani: OpenAIRealtimeKani, **kwargs):
    """Chat with a kani right in your terminal.

    Useful for playing with kani, quick prompt engineering, or demoing the library.

    If the environment variable ``KANI_DEBUG`` is set, debug logging will be enabled.

    .. warning::

        This function is only a development utility and should not be used in production.

    :param int rounds: The number of chat rounds to play (defaults to 0 for infinite; chat or stream mode only).
    :param str stopword: Break out of the chat loop if the user sends this message (chat or stream mode only).
    :param bool echo: Whether to echo the user's input to stdout after they send a message (e.g. to save in interactive
        notebook outputs; default false; chat or stream mode only)
    :param bool ai_first: Whether the user should send the first message (default) or the model should generate a
        completion before prompting the user for a message.
    :param int width: The maximum width of the printed outputs (default unlimited).
    :param bool show_function_args: Whether to print the arguments the model is calling functions with for each call
        (default false).
    :param bool show_function_returns: Whether to print the results of each function call (default false).
    :param bool verbose: Equivalent to setting ``echo``, ``show_function_args``, and ``show_function_returns`` to True.
    :param str mode: The chat mode: "chat" for turn-based chat without streaming, "stream" for turn-based chat with
        streaming and audio, "full_duplex" for realtime conversation from the system default mic.
    :param int mic_id: The microphone ID to use for recording audio (default system default mic; full_duplex mode only)
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        try:
            # google colab comes with this pre-installed
            # let's try importing and patching the loop so that we can just use the normal asyncio.run call
            import nest_asyncio

            nest_asyncio.apply()
        except ImportError:
            print(
                f"WARNING: It looks like you're in an environment with a running asyncio loop (e.g. Google Colab).\nYou"
                f" should use `await chat_in_terminal_async(...)` instead or install `nest-asyncio`."
            )
            return
    asyncio.run(chat_in_terminal_audio_async(kani, **kwargs))


class FullDuplexManager:
    """Manager class for handling multiple concurrent streams for the full duplex mode."""

    def __init__(
        self,
        kani: OpenAIRealtimeKani,
        audio_stream,
        width: int | None,
        show_function_args: bool,
        show_function_returns: bool,
    ):
        self.kani = kani
        self.audio_stream = audio_stream
        self.width = width
        self.show_function_args = show_function_args
        self.show_function_returns = show_function_returns

        self.stream_tasks = set()
        self.stream_outputs = []

        self.main_task = None

    # ===== lifecycle =====
    async def start(self):
        self.main_task = asyncio.create_task(self._main_task())

    async def close(self):
        if self.main_task is not None:
            self.main_task.cancel()

    # ===== main =====
    async def _main_task(self):
        # bg task to handle getting stream info
        async for stream in self.kani.full_duplex(self.audio_stream, audio_callback=play_audio):
            # for each message stream emitted by the model, spawn a task to handle it
            idx = len(self.stream_outputs)
            self.stream_outputs.append([])
            task = asyncio.create_task(self._handle_one_stream_task(stream, idx))
            self.stream_tasks.add(task)
            task.add_done_callback(self.stream_tasks.discard)

    async def _handle_one_stream_task(self, stream: StreamManager, output_idx: int):
        output_buffer = self.stream_outputs[output_idx]

        # assistant
        if stream.role == ChatRole.ASSISTANT:
            await buffer_stream(stream, output_buffer, width=self.width, prefix="AI: ")
            msg = await stream.message()
            text = assistant_message_thinking(msg, show_args=self.show_function_args)
            if text:
                output_buffer.append(format_width(text, width=self.width, prefix="AI: "))
        # function
        elif stream.role == ChatRole.FUNCTION and self.show_function_returns:
            msg = await stream.message()
            output_buffer.append(format_width(msg.text, width=self.width, prefix="FUNC: "))
        # user
        elif stream.role == ChatRole.USER:
            await buffer_stream(stream, output_buffer, width=self.width, prefix="USER: ")

    def get_display_text(self):
        return "\n".join("".join(part for part in output) for output in self.stream_outputs)


# ===== format helpers =====
def format_width(msg: str, width: int = None, prefix: str = ""):
    """
    Format the given message such that the width of each line is less than *width*.
    If *prefix* is provided, indents each line after the first by the length of the prefix.

    .. code-block: pycon
        >>> format_width("Hello world I am a potato", width=15, prefix="USER: ")
        '''\
        USER: Hello
              world I
              am a
              potato\
        '''
    """
    if not width:
        return prefix + msg
    out = []
    wrapper = textwrap.TextWrapper(width=width, initial_indent=prefix, subsequent_indent=" " * len(prefix))
    lines = msg.splitlines()
    for line in lines:
        out.append(wrapper.fill(line))
        wrapper.initial_indent = wrapper.subsequent_indent
    return "\n".join(out)


async def buffer_stream(stream: StreamManager, buf: list, width: int = None, prefix: str = ""):
    """
    Buffer tokens from a stream to the given list, with the width of each line less than *width*.
    If *prefix* is provided, indents each line after the first by the length of the prefix.

    This is a helper function intended to be used with :meth:`.Kani.chat_round_stream` or
    :meth:`.Kani.full_round_stream`.
    """
    prefix_len = len(prefix)
    line_indent = " " * prefix_len
    prefix_printed = False

    # print tokens until they overflow width then newline and indent
    line_len = prefix_len
    async for token in stream:
        # only print the prefix if the model actually yields anything
        if not prefix_printed:
            buf.append(prefix)
            prefix_printed = True

        # split by newlines
        for part in token.splitlines(keepends=True):
            # then do bookkeeping
            line_len += len(part)
            if width and line_len > width:
                buf.append(f"\n{line_indent}")
                line_len = prefix_len

            # print the token
            buf.append(part.rstrip("\r\n"))

            # print a newline if the token had one
            if part.endswith("\n"):
                buf.append(f"\n{line_indent}")
                line_len = prefix_len

    return buf
