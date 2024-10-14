import qdrant_client
import torch

client = qdrant_client.QdrantClient("http://localhost:6333")

points = client.query_points("diffusers", query=torch.full((2048, ), 0.1).tolist(), limit=100, with_vectors=True).points
# scroll_points = client.scroll(
#     collection_name = "diffusers",
#     with_vectors=True
# )

for point in points:
    print(point.id, point.score)