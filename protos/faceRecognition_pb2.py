# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: faceRecognition.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x15\x66\x61\x63\x65Recognition.proto\x12\x0f\x66\x61\x63\x65Recognition\";\n\x04Rect\x12\t\n\x01x\x18\x01 \x01(\x05\x12\t\n\x01y\x18\x02 \x01(\x05\x12\r\n\x05width\x18\x03 \x01(\x05\x12\x0e\n\x06height\x18\x04 \x01(\x05\"\x1d\n\x0cImageRequest\x12\r\n\x05image\x18\x01 \x01(\x0c\"E\n\x14MultipleImageRequest\x12-\n\x06images\x18\x01 \x03(\x0b\x32\x1d.faceRecognition.ImageRequest\"k\n\x15ImageWithRectsRequest\x12,\n\x05image\x18\x01 \x01(\x0b\x32\x1d.faceRecognition.ImageRequest\x12$\n\x05rects\x18\x02 \x03(\x0b\x32\x15.faceRecognition.Rect\"&\n\x11\x45mbeddingResponse\x12\x11\n\tembedding\x18\x01 \x03(\x02\"S\n\x19MultipleEmbeddingResponse\x12\x36\n\nembeddings\x18\x01 \x03(\x0b\x32\".faceRecognition.EmbeddingResponse2\xba\x02\n\x0f\x46\x61\x63\x65Recognition\x12U\n\x10getFaceEmbedding\x12\x1d.faceRecognition.ImageRequest\x1a\".faceRecognition.EmbeddingResponse\x12\x66\n\x11getFaceEmbeddings\x12%.faceRecognition.MultipleImageRequest\x1a*.faceRecognition.MultipleEmbeddingResponse\x12h\n\x12getFaceEmbeddings2\x12&.faceRecognition.ImageWithRectsRequest\x1a*.faceRecognition.MultipleEmbeddingResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'faceRecognition_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_RECT']._serialized_start=42
  _globals['_RECT']._serialized_end=101
  _globals['_IMAGEREQUEST']._serialized_start=103
  _globals['_IMAGEREQUEST']._serialized_end=132
  _globals['_MULTIPLEIMAGEREQUEST']._serialized_start=134
  _globals['_MULTIPLEIMAGEREQUEST']._serialized_end=203
  _globals['_IMAGEWITHRECTSREQUEST']._serialized_start=205
  _globals['_IMAGEWITHRECTSREQUEST']._serialized_end=312
  _globals['_EMBEDDINGRESPONSE']._serialized_start=314
  _globals['_EMBEDDINGRESPONSE']._serialized_end=352
  _globals['_MULTIPLEEMBEDDINGRESPONSE']._serialized_start=354
  _globals['_MULTIPLEEMBEDDINGRESPONSE']._serialized_end=437
  _globals['_FACERECOGNITION']._serialized_start=440
  _globals['_FACERECOGNITION']._serialized_end=754
# @@protoc_insertion_point(module_scope)
