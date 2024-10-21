# kani-ext-realtime

This repository adds the OpenAIRealtimeKani.

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

from kani.ext.realtime import OpenAIRealtimeKani
from kani.ext.realtime.cli import chat_in_terminal_audio_async


async def main():
    ai = OpenAIRealtimeKani()
    await ai.connect()
    await chat_in_terminal_audio_async(ai)


if __name__ == "__main__":
    asyncio.run(main())
```

- OpenAIRealtimeKani: manages a conversation
    - use `.session` for underlying OpenAI session
    - listen to events with `.session.add_listener` or `.session.wait_for`
- playing audio from streaming interfaces:
    - `ai.full_round_stream(..., audio_callback=play_audio)` (for example)
- `chat_in_terminal_audio_async`: use `mode="audio"|"full_duplex"` for audio I/O TODO

<!--
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