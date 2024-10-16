import inspect
from typing import Any, Awaitable, Callable, TypeVar

from . import events

T = TypeVar("T", bound=events.ServerEvent)


def server_event_handler(event_type: str) -> Callable[
    [Callable[[T], Awaitable[Any]]],
    Callable[[T], Awaitable[Any]],
]:
    """Annotates the wrapped method with the type of event it handles.

    >>> class Foo:
    ...     @server_event_handler("session.created")
    ...     async def handle(self, event):
    ...         ...
    >>> Foo.handle.__realtime_event_handler__ == "session.created"
    """

    def wrapper(f: Callable[[T], Awaitable[Any]]):
        f.__realtime_event_handler__ = event_type
        return f

    return wrapper


def get_server_event_handlers(inst) -> dict[str, Callable[[T], Awaitable[Any]]]:
    """Get a mapping of all type -> handler on this instance."""
    handlers = {}
    for name, member in inspect.getmembers(inst, predicate=inspect.ismethod):
        if not hasattr(member, "__realtime_event_handler__"):
            continue
        if member.__realtime_event_handler__ in handlers:
            raise ValueError(f"Handler for event type {member.__realtime_event_handler__!r} is already defined!")
        handlers[member.__realtime_event_handler__] = member
    return handlers
