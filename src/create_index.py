#create_index.py

from elasticsearch import Elasticsearch
from config import ELASTIC_HOST, ELASTIC_INDEX, EMBEDDING_DIM

def main():
    es = Elasticsearch(ELASTIC_HOST)

    # Supprime l'index s'il existe déjà:
    if es.indices.exists(index=ELASTIC_INDEX):
        es.indices.delete(index=ELASTIC_INDEX)
        print(f"Deleted existing index: {ELASTIC_INDEX}")

    mapping = {
        "mappings": {
            "properties": {
                "content": {
                    "type": "text"
                },
                "embedding": {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIM,
                    "index": True,
                    "similarity": "cosine"
                },
                "source": {
                    "type": "keyword"
                }
            }
        }
    }

    es.indices.create(index=ELASTIC_INDEX, body=mapping)
    print(f"Created index: {ELASTIC_INDEX}")

if __name__ == "__main__":
    main()

