"""The CLI utilities allow you to play with a chat session directly from a terminal."""

import asyncio
import logging
import os
import sys
from typing import Literal, overload

from kani.kani import Kani
from kani.models import ChatRole
from kani.utils.cli import print_stream, print_width
from kani.utils.message_formatters import assistant_message_contents_thinking, assistant_message_thinking

from . import interop
from .utils import play_audio


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


async def chat_in_terminal_audio_async(
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
):
    """Async version of :func:`.chat_in_terminal`.
    Use in environments when there is already an asyncio loop running (e.g. Google Colab).
    """
    if mode not in ("chat", "stream", "full_duplex"):
        raise ValueError('mode must be one of "chat", "stream", "full_duplex"')
    if os.getenv("KANI_DEBUG") is not None:
        logging.basicConfig(level=logging.DEBUG)
    if verbose:
        echo = show_function_args = show_function_returns = True

    try:
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
    except (KeyboardInterrupt, asyncio.CancelledError):
        # we won't close the engine here since it's common enough that people close the session in colab
        # and if the process is closing then this will clean itself up anyway
        # await kani.engine.close()
        return


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
): ...


def chat_in_terminal_audio(kani: Kani, **kwargs):
    """Chat with a kani right in your terminal.

    Useful for playing with kani, quick prompt engineering, or demoing the library.

    If the environment variable ``KANI_DEBUG`` is set, debug logging will be enabled.

    .. warning::

        This function is only a development utility and should not be used in production.

    :param int rounds: The number of chat rounds to play (defaults to 0 for infinite).
    :param str stopword: Break out of the chat loop if the user sends this message.
    :param bool echo: Whether to echo the user's input to stdout after they send a message (e.g. to save in interactive
        notebook outputs; default false)
    :param bool ai_first: Whether the user should send the first message (default) or the model should generate a
        completion before prompting the user for a message.
    :param int width: The maximum width of the printed outputs (default unlimited).
    :param bool show_function_args: Whether to print the arguments the model is calling functions with for each call
        (default false).
    :param bool show_function_returns: Whether to print the results of each function call (default false).
    :param bool verbose: Equivalent to setting ``echo``, ``show_function_args``, and ``show_function_returns`` to True.
    :param str mode: The chat mode: "chat" for turn-based chat without streaming, "stream" for turn-based chat with
        streaming and audio, "full_duplex" for realtime conversation from the system default mic.
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