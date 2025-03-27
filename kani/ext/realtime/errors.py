class RealtimeError(Exception):
    pass


# WS Error event
class OpenAIRealtimeError(RealtimeError):
    def __init__(
        self, message: str, type: str, code: str | None = None, event_id: str | None = None, param: str | None = None
    ):
        self.msg = message
        self.type = type
        self.code = code
        self.event_id = event_id
        self.param = param

    @classmethod
    def from_ws_error(cls, err):
        return cls(message=err.message, type=err.type, code=err.code, event_id=err.event_id, param=err.param)

    def __str__(self):
        return f"{self.type=} {self.code=} {self.event_id=} {self.param=} {self.msg=}"


# response.done with status=failed
class ResponseFailed(OpenAIRealtimeError):
    @classmethod
    def from_response_failure_error(cls, err):
        return cls(message=getattr(err, "message", "response failed (no message)"), type=err.type, code=err.code)


def exc_for_response_failure(status_details):
    if status_details is None:
        return
    if status_details.type != "failed":
        return
    return ResponseFailed.from_response_failure_error(status_details.error)


def raise_for_response_failure(status_details):
    if exc := exc_for_response_failure(status_details):
        raise exc
