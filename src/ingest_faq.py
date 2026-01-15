import json
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from config import (
    ELASTIC_HOST,
    ELASTIC_INDEX,
    EMBEDDING_MODEL_NAME
)

def main():
    es = Elasticsearch(ELASTIC_HOST)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    with open("data/faq.json", "r") as f:
        faqs = json.load(f)

    for faq in faqs:
        content = f"Question: {faq['question']}\nAnswer: {faq['answer']}"
        embedding = model.encode(content).tolist()

        doc = {
            "content": content,
            "embedding": embedding,
            "source": "faq"
        }

        es.index(index=ELASTIC_INDEX, document=doc)

    print(f"Ingested {len(faqs)} FAQ documents into Elasticsearch.")

if __name__ == "__main__":
    main()

