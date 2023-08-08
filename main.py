import io
import sys
import grpc
from torchvision import transforms
import torch
import onnxruntime as ort
import protos.faceRecognition_pb2_grpc as faceRecognition_pb2_grpc
import protos.faceRecognition_pb2 as faceRecognition_pb2
from concurrent import futures
from PIL import Image

from grpc_health.v1 import health
from grpc_health.v1 import health_pb2, health_pb2_grpc

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
    self.model = ort.InferenceSession("models/model.onnx")
    self.preprocess = transforms.Compose([
      # transforms.Resize((112,112)),
      transforms.ToTensor(),
      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

  def load_model(self):
    try:
      self.model = ort.InferenceSession("models/model.onnx")
    except Exception as e:
      print(f"Error loading model: {e}", file=sys.stderr)

  def is_healthy(self):
    return self.model is not None

  def getFaceEmbedding(self, request, _): # _ is a placeholder for context
    response = faceRecognition_pb2.EmbeddingResponse()

    image = self._process_image(request.image)

    image = image.unsqueeze(0)

    input = image.numpy()

    res = self.model.run(None, {self.model.get_inputs()[0].name: input})
    embedding = res[0][0].tolist()

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
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  
  face_recognition_service = FaceRecognition()
  faceRecognition_pb2_grpc.add_FaceRecognitionServicer_to_server(face_recognition_service, server)

  health_servicer = HealthServicer(face_recognition_service)
  health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

  server.add_insecure_port('[::]:50051')
  server.start()
  server.wait_for_termination()

if __name__ == "__main__":
  serve()