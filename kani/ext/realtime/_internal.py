import asyncio
import functools
import inspect
from typing import Any, Awaitable, Callable

from openai.types.beta.realtime import RealtimeServerEvent


def server_event_handler(event_type: str) -> Callable[
    [Callable[[RealtimeServerEvent], Awaitable[Any]]],
    Callable[[RealtimeServerEvent], Awaitable[Any]],
]:
    """Annotates the wrapped method with the type of event it handles.

    >>> class Foo:
    ...     @server_event_handler("session.created")
    ...     async def handle(self, event):
    ...         ...
    >>> Foo.handle.__realtime_event_handler__ == "session.created"
    """

    def wrapper(f: Callable[[RealtimeServerEvent], Awaitable[Any]]):
        f.__realtime_event_handler__ = event_type
        return f

    return wrapper


def get_server_event_handlers(inst) -> dict[str, Callable[[RealtimeServerEvent], Awaitable[Any]]]:
    """Get a mapping of all type -> handler on this instance."""
    handlers = {}
    for name, member in inspect.getmembers(inst, predicate=inspect.ismethod):
        if not hasattr(member, "__realtime_event_handler__"):
            continue
        if member.__realtime_event_handler__ in handlers:
            raise ValueError(f"Handler for event type {member.__realtime_event_handler__!r} is already defined!")
        handlers[member.__realtime_event_handler__] = member
    return handlers


def ensure_async(f, run_sync_in_executor=False):
    """
    Ensure the callable is an async function (or wrapped in one).

    If *run_sync_in_executor* is True and *f* is sync, the returned async function will be run in an executor.
    Otherwise, it will be run on the main event loop (be sure it's not blocking!).
    """
    if f is None:

        async def f(*_, **__):
            pass

    elif not inspect.iscoroutinefunction(f):
        original = f

        if run_sync_in_executor:

            @functools.wraps(original)
            async def f(*args, **kwargs):
                inner = functools.partial(original, *args, **kwargs)
                return await asyncio.get_event_loop().run_in_executor(None, inner)

        else:

            @functools.wraps(original)
            async def f(*args, **kwargs):
                return original(*args, **kwargs)

    return f


_global_bg_tasks = set()


def create_task(coro):
    """Helper with bookkeeping to prevent errant GCs."""
    task = asyncio.create_task(coro)
    _global_bg_tasks.add(task)
    task.add_done_callback(_global_bg_tasks.discard)
