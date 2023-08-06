import io
import grpc
from torchvision import transforms
import torch
import onnxruntime as ort
import protos.faceRecognition_pb2_grpc as faceRecognition_pb2_grpc
import protos.faceRecognition_pb2 as faceRecognition_pb2
from concurrent import futures
from PIL import Image

class FaceRecognition(faceRecognition_pb2_grpc.FaceRecognitionServicer):

  def __init__(self):
    self.model = ort.InferenceSession("models/model.onnx")
    self.preprocess = transforms.Compose([
      transforms.Resize((112,112)),
      transforms.ToTensor(),
      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

  def GetEmbedding(self, request, context):
    return self._process_image(request.image)

  def getFaceEmbeddings(self, request, context):

    response = faceRecognition_pb2.MultipleEmbeddingResponse()

    images = []
    for image_request in request.images:
      image = self._process_image(image_request.image)
      images.append(image)

    input = torch.stack(tuple(images), dim=0)

    input = input.numpy()

    res = self.model.run(None, {self.model.get_inputs()[0].name: input})
    embeddings = res[0].tolist()

    response = faceRecognition_pb2.MultipleEmbeddingResponse()
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
  faceRecognition_pb2_grpc.add_FaceRecognitionServicer_to_server(FaceRecognition(), server)
  server.add_insecure_port('[::]:50051')
  server.start()
  server.wait_for_termination()

if __name__ == "__main__":
  serve()