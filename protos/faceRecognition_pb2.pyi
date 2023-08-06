from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ImageRequest(_message.Message):
    __slots__ = ["image"]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    image: bytes
    def __init__(self, image: _Optional[bytes] = ...) -> None: ...

class MultipleImageRequest(_message.Message):
    __slots__ = ["images"]
    IMAGES_FIELD_NUMBER: _ClassVar[int]
    images: _containers.RepeatedCompositeFieldContainer[ImageRequest]
    def __init__(self, images: _Optional[_Iterable[_Union[ImageRequest, _Mapping]]] = ...) -> None: ...

class EmbeddingResponse(_message.Message):
    __slots__ = ["embedding"]
    EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    embedding: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, embedding: _Optional[_Iterable[float]] = ...) -> None: ...

class MultipleEmbeddingResponse(_message.Message):
    __slots__ = ["embeddings"]
    EMBEDDINGS_FIELD_NUMBER: _ClassVar[int]
    embeddings: _containers.RepeatedCompositeFieldContainer[EmbeddingResponse]
    def __init__(self, embeddings: _Optional[_Iterable[_Union[EmbeddingResponse, _Mapping]]] = ...) -> None: ...
