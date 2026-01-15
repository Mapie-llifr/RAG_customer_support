from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from config import (
    ELASTIC_HOST,
    ELASTIC_INDEX,
    EMBEDDING_MODEL_NAME
)

def semantic_search(query: str, top_k: int = 3):
    es = Elasticsearch(ELASTIC_HOST)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    query_embedding = model.encode(query).tolist()

    response = es.search(
        index=ELASTIC_INDEX,
        knn={
            "field": "embedding",
            "query_vector": query_embedding,
            "k": top_k,
            "num_candidates": 10
        }
    )

    hits = response["hits"]["hits"]

    for i, hit in enumerate(hits, start=1):
        print(f"\nResult {i}:")
        print(hit["_source"]["content"])

if __name__ == "__main__":
    user_question = "A quoi ça sert la secu ?"
    #user_question = "A quoi ça sert Mon Espace Santé?"
    semantic_search(user_question)

