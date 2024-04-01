import io
import os
import sys
import grpc
import numpy as np
from torchvision import transforms
import torch
import onnxruntime as ort
import protos.faceRecognition_pb2_grpc as faceRecognition_pb2_grpc
import protos.faceRecognition_pb2 as faceRecognition_pb2
from concurrent import futures
from PIL import Image

from grpc_health.v1 import health_pb2, health_pb2_grpc

env = os.environ.get('ENV')

class HealthServicer(health_pb2_grpc.HealthServicer):
  def __init__(self, face_recognition_service):
    self.face_recognition_service = face_recognition_service

  def Check(self, request, _): # _ is a placeholder for context
    if self.face_recognition_service.is_healthy():
      return health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.SERVING)
    else:
      return health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.NOT_SERVING)

  def Watch(self, request, _): # _ is a placeholder for context
    raise NotImplementedError()

class FaceRecognition(faceRecognition_pb2_grpc.FaceRecognitionServicer):

  def __init__(self):
    if (env == "prod"):
      self.model = ort.InferenceSession("models/model.onnx", providers=["CUDAExecutionProvider"])
    else:
      self.model = ort.InferenceSession("models/model.onnx")
    self.preprocess = transforms.Compose([
      transforms.Resize((112,112)),
      transforms.ToTensor(),
      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

  def is_healthy(self):
    return self.model is not None

  def getFaceEmbedding(self, request, _): # _ is a placeholder for context
    response = faceRecognition_pb2.EmbeddingResponse()

    image = self._process_image(request.image.image)

    image = image.unsqueeze(0)

    input = image.numpy()

    res = self.model.run(None, {self.model.get_inputs()[0].name: input})
    embedding = res[0][0].tolist()

    embedding_float64 = np.array(embedding)  # assuming embeddings is a list or array of float64
    embedding_float32 = embedding_float64.astype(np.float32)

    embedding = embedding_float32.tolist()

    response.embedding.extend(embedding)

    return response

  def getFaceEmbeddings(self, request, _): # _ is a placeholder for context
    response = faceRecognition_pb2.MultipleEmbeddingResponse()

    images = []
    for image_request in request.images:
      image = self._process_image(image_request.image)
      images.append(image)

    input = torch.stack(tuple(images), dim=0)

    input = input.numpy()

    res = self.model.run(None, {self.model.get_inputs()[0].name: input})
    embeddings = res[0].tolist()

    embeddings_float64 = np.array(embeddings)
    embeddings_float32 = embeddings_float64.astype(np.float32)

    embeddings = embeddings_float32.tolist()

    for embedding in embeddings:
      embedding_response = faceRecognition_pb2.EmbeddingResponse()
      embedding_response.embedding.extend(embedding)
      response.embeddings.extend([embedding_response])

    return response

  def getFaceEmbeddings2(self, request, _): # _ is a placeholder for context
    response = faceRecognition_pb2.MultipleEmbeddingResponse()
    whole_image = Image.open(io.BytesIO(request.image.image)).convert('RGB')

    # Crop faces
    images = []
    for rect in request.rects:
        left = rect.x
        top = rect.y
        right = left + rect.width
        bottom = top + rect.height

        cropped_pil_image = whole_image.crop((left, top, right, bottom))

        image_tensor = self.preprocess(cropped_pil_image)
        images.append(image_tensor)

    input = torch.stack(tuple(images), dim=0)

    input = input.numpy()

    res = self.model.run(None, {self.model.get_inputs()[0].name: input})
    embeddings = res[0].tolist()

    embeddings_float64 = np.array(embeddings)
    embeddings_float32 = embeddings_float64.astype(np.float32)

    embeddings = embeddings_float32.tolist()

    for embedding in embeddings:
      embedding_response = faceRecognition_pb2.EmbeddingResponse()
      embedding_response.embedding.extend(embedding)
      response.embeddings.extend([embedding_response])

    return response

  def _process_image(self, image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = self.preprocess(image)
    
    return image_tensor
  
def serve():
    max_message_length = 10 * 1024 * 1024 # 10MB

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ('grpc.max_send_message_length', max_message_length),
            ('grpc.max_receive_message_length', max_message_length)
        ]
    )
  
    face_recognition_service = FaceRecognition()
    faceRecognition_pb2_grpc.add_FaceRecognitionServicer_to_server(face_recognition_service, server)

    health_servicer = HealthServicer(face_recognition_service)
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    server.add_insecure_port('[::]:50051')

    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
