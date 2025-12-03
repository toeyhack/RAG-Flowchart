from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams
import os
import uuid

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION = "rag_nodes"

def get_qdrant():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def ensure_collection(dim: int):
    client = get_qdrant()
    try:
        client.get_collection(COLLECTION)
    except Exception:
        client.recreate_collection(COLLECTION, vectors_config=VectorParams(size=dim, distance="Cosine"))

def upsert_node(vector: list, payload: dict):
    client = get_qdrant()
    point_id = str(uuid.uuid4())
    pt = PointStruct(id=point_id, vector=vector, payload=payload)
    client.upsert(collection_name=COLLECTION, points=[pt])
    return point_id
