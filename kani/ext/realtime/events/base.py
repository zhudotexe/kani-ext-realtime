import abc
import uuid
import warnings
from typing import ClassVar, Type

from pydantic import BaseModel, Field, model_validator


class BaseEvent(BaseModel, abc.ABC):
    event_id: str | None = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str


class ServerEvent(BaseEvent, abc.ABC):
    # this class registers a list of subclasses that will be used later for deserialization
    # ==== serdes ====
    # used for saving/loading - map qualname to messagepart type
    _server_event_registry: ClassVar[dict[str, Type["ServerEvent"]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        event_type = cls.type
        if event_type in cls._server_event_registry:
            warnings.warn(
                f"The server event with type {event_type!r} was defined multiple times (perhaps a class is being"
                " defined in a function scope). This will likely cause undesired behaviour when deserializing server"
                " events."
            )
        cls._server_event_registry[event_type] = cls

    # noinspection PyNestedDecorators
    @model_validator(mode="wrap")
    @classmethod
    def _validate(cls, v, nxt):
        if isinstance(v, dict) and "type" in v:
            event_type = v["type"]
            try:
                klass = cls._server_event_registry[event_type]
            except KeyError:
                return UnknownEvent(event_id=v["event_id"], type=event_type, data=v)
            return klass.model_validate(v)
        return nxt(v)


class UnknownEvent(BaseEvent):
    """Catch-all unknown event type for server events."""

    data: dict
