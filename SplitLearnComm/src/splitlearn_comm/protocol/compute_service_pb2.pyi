from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ComputeRequest(_message.Message):
    __slots__ = ("data", "shape", "request_id", "metadata", "model_id")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    DATA_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    shape: _containers.RepeatedScalarFieldContainer[int]
    request_id: int
    metadata: _containers.ScalarMap[str, str]
    model_id: str
    def __init__(self, data: _Optional[bytes] = ..., shape: _Optional[_Iterable[int]] = ..., request_id: _Optional[int] = ..., metadata: _Optional[_Mapping[str, str]] = ..., model_id: _Optional[str] = ...) -> None: ...

class ComputeResponse(_message.Message):
    __slots__ = ("data", "shape", "request_id", "compute_time_ms")
    DATA_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    shape: _containers.RepeatedScalarFieldContainer[int]
    request_id: int
    compute_time_ms: float
    def __init__(self, data: _Optional[bytes] = ..., shape: _Optional[_Iterable[int]] = ..., request_id: _Optional[int] = ..., compute_time_ms: _Optional[float] = ...) -> None: ...

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("status", "message", "version", "uptime_seconds")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    UPTIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
    status: str
    message: str
    version: str
    uptime_seconds: float
    def __init__(self, status: _Optional[str] = ..., message: _Optional[str] = ..., version: _Optional[str] = ..., uptime_seconds: _Optional[float] = ...) -> None: ...

class ServiceInfoRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ServiceInfoResponse(_message.Message):
    __slots__ = ("service_name", "version", "device", "uptime_seconds", "total_requests", "custom_info")
    class CustomInfoEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    SERVICE_NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    UPTIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    CUSTOM_INFO_FIELD_NUMBER: _ClassVar[int]
    service_name: str
    version: str
    device: str
    uptime_seconds: float
    total_requests: int
    custom_info: _containers.ScalarMap[str, str]
    def __init__(self, service_name: _Optional[str] = ..., version: _Optional[str] = ..., device: _Optional[str] = ..., uptime_seconds: _Optional[float] = ..., total_requests: _Optional[int] = ..., custom_info: _Optional[_Mapping[str, str]] = ...) -> None: ...
