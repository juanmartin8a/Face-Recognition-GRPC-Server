import io
import os
import sys
import grpc
import numpy as np
from torchvision import transforms
import torch
import protos.faceRecognition_pb2_grpc as faceRecognition_pb2_grpc
import protos.faceRecognition_pb2 as faceRecognition_pb2
from concurrent import futures
from PIL import Image

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
    # self.model = ort.InferenceSession("models/model.onnx")
    self.model = torch.load("models/pytorch_model.pt")
    self.device = torch.device('cpu')
    self.preprocess = transforms.Compose([
      transforms.Resize((112,112), interpolation=Image.BICUBIC),
      transforms.ToTensor(),
      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

  def load_model(self):
    try:
      self.model = torch.load("models/pytorch_model.pt")
    except Exception as e:
      print(f"Error loading model: {e}", file=sys.stderr)

  def is_healthy(self):
    return self.model is not None

  def getFaceEmbedding(self, request, _): # _ is a placeholder for context
    response = faceRecognition_pb2.EmbeddingResponse()

    image = self._process_image(request.image)

    image = image.unsqueeze(0)

    input = image.to(self.device)

    self.model.eval()
    res = self.model(input)

    embedding = res.tolist()

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

    images_tensor = torch.stack(tuple(images), dim=0)

    input = images_tensor.to(self.device)

    self.model.eval()
    res = self.model(input)

    embeddings = res.tolist()

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
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1000000000000))
  
    face_recognition_service = FaceRecognition()
    faceRecognition_pb2_grpc.add_FaceRecognitionServicer_to_server(face_recognition_service, server)

    health_servicer = HealthServicer(face_recognition_service)
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    env = os.environ.get('ENV')

    if (env == "prod"):
        # Load the generated ECC private key and SSL certificate
        with open('keys/private.pem', 'rb') as f:
            private_key = f.read()
        with open('keys/certificate.pem', 'rb') as f:
            certificate_chain = f.read()

        server_credentials = grpc.ssl_server_credentials(
            ((private_key, certificate_chain,),)
        )

        server.add_secure_port('[::]:50051', server_credentials)
    else:
        server.add_insecure_port('[::]:50051')

    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
