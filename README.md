# kani-ext-realtime

This repository adds the OpenAIRealtimeKani. Function calling works right out of the box!

https://github.com/user-attachments/assets/64aa852b-e97a-4d48-b092-e5b672a97e0f

This package is considered provisional and maintained on a best-effort basis. As such, it will not be released on
PyPI.

To install this package, you must install it using the git source:

```shell
$ pip install git+https://github.com/zhudotexe/kani-ext-realtime.git@main
```

See https://platform.openai.com/docs/guides/realtime for more information on the OpenAI Realtime API.

## Usage

```python
import asyncio

from kani.ext.realtime import OpenAIRealtimeKani, chat_in_terminal_audio_async


async def main():
    ai = OpenAIRealtimeKani()  # note - the OpenAIRealtimeKani does *not* take an engine!
    await ai.connect()  # additional step needed to connect to the Realtime API
    await chat_in_terminal_audio_async(ai, mode="full_duplex")


if __name__ == "__main__":
    asyncio.run(main())
```

## Programmatic Usage

```python
import asyncio

from kani.ext.realtime import OpenAIRealtimeKani, chat_in_terminal_audio_async


async def handle_stream(stream):
    # do processing for a single message's stream here...
    # this example code does NOT account for multiple simultaneous messages
    async for token in stream:
        print(token, end="")
    msg = await stream.message()


async def main():
    ai = OpenAIRealtimeKani()  # note - the OpenAIRealtimeKani does *not* take an engine!
    await ai.connect()  # additional step needed to connect to the Realtime API

    stream_tasks = set()

    async for stream in ai.full_duplex(audio_stream):
        task = asyncio.create_task(handle_stream(stream))
        # to keep a live reference to the task
        # see https://docs.python.org/3/library/asyncio-task.html#creating-tasks
        stream_tasks.add(task)
        task.add_done_callback(stream_tasks.discard)


if __name__ == "__main__":
    asyncio.run(main())
```

The OpenAIRealtimeKani is compatible with most standard kani interfaces -- you can, for example:

- Define `@ai_function`s which the realtime model will call (by subclassing `OpenAIRealtimeKani`)
- Supply a `system_prompt` or fewshot examples in `chat_history`
    - (note: you likely want to supply a system prompt through `SessionConfig.instructions` instead, see below)
- Get text/audio completions like a normal text-text model with `.full_round`

The new methods provided in the `OpenAIRealtimeKani` are:

- `connect(config: SessionConfig)`
- `full_duplex(input_audio_stream: AsyncIterable[bytes], output_audio_callback: AsyncCallable[[bytes], Any])`

It also supports the configuration options provided by the Realtime API. When you call `.connect`, you can supply a
`SessionConfig` object, which supports all of the options listed
at https://platform.openai.com/docs/api-reference/realtime-client-events/session/update.

For more information:

## [Read the Docs!](https://kani-ext-realtime.readthedocs.io)

<!--
- OpenAIRealtimeKani: manages a conversation
    - use `.session` for underlying OpenAI session
    - listen to events with `.session.add_listener` or `.session.wait_for`
- playing audio from streaming interfaces:
    - `ai.full_round_stream(..., audio_callback=play_audio)` (for example)
- `chat_in_terminal_audio_async`: use `mode="stream" | "full_duplex"` for audio I/O

## Publishing to PyPI

To publish your package to PyPI, this repo comes with a GitHub Action that will automatically build and upload new
releases. Alternatively, you can build and publish the package manually.

### GitHub Action

To use the GitHub Action, you must configure it as a publisher for your project on
PyPI: https://pypi.org/manage/account/publishing/

The workflow is configured with the following settings:

- workflow name: `pythonpublish.yml`
- environment name: `pypi`

Once you've configured this, each release you publish on GitHub will automatically be built and uploaded to PyPI.
You can also manually trigger the workflow.

Make sure to update the version number in `pyproject.toml` before releasing a new version!
-->
